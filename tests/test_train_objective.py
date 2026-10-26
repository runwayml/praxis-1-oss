"""Unit tests of the training objective and of the MoE balance (CPU only).

The flex attention of the backbone has no backward on the CPU. Thus these
tests check the parts of the objective that run on the CPU:

* the timestep schedule, the draws and the two-stage video noising
* the MoE balance terms, the routing bias update and the chunked dispatch
* the backward of the router and of the expert row permutation
* the action loss with a backbone stand-in, the text and proprio dropout
* the full loss forward of a tiny chunk-causal policy (eager flex attention)
* ``encode_window``

The GPU kernels have their own checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tiny_model import tiny_config, tiny_policy
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from praxis.backbone import FRAME_STRIDE, ROW_STRIDE, TapFeatures, build_backbone
from praxis.layers import MoEFeedForward, _PermuteRows, _router_logits
from praxis.model import PraxisPolicy
from praxis.robot_io import pack_action_tokens, q99_normalize
from praxis.train.moe import (
    RoutingStats,
    expert_counts,
    sequence_balance_loss,
    take_sequence_balance_losses,
    track_routing,
    update_expert_biases,
)
from praxis.train.objective import (
    EXPERT_FLEX_BACKWARD_OPTIONS,
    Draws,
    EncodedWindow,
    FinetuneModel,
    ObjectiveSettings,
    encode_window,
    shifted_timesteps,
)

H, W = 2, 4


def _bytes(t: Tensor) -> Tensor:
    """The raw bytes of ``t``, so that -0.0 and +0.0 differ."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _window(seed: int = 0, n_text: int = 7) -> EncodedWindow:
    g = torch.Generator().manual_seed(seed)
    return EncodedWindow(
        text=torch.randn(1, n_text, 32, generator=g).to(torch.bfloat16),
        anchor=torch.randn(1, 192, 1, H, W, generator=g).to(torch.bfloat16),
        chunks=[
            torch.randn(1, 192, 6, H, W, generator=g).to(torch.bfloat16)
            for _ in range(4)
        ],
        actions=torch.randn(1, 24, 40, generator=g).clamp(-1, 1),
        proprio=torch.randn(1, 4, 10, generator=g),
    )


UNCOND = torch.full((1, 1, 32), 0.25).to(torch.bfloat16)


def _model(dtype: torch.dtype = torch.bfloat16, **settings: object) -> FinetuneModel:
    s = ObjectiveSettings(**settings)  # ty: ignore[invalid-argument-type]
    return FinetuneModel(tiny_policy(dtype=dtype), s, UNCOND)


# ---------------------------------------------------------------------------
# Timesteps, draws, noising
# ---------------------------------------------------------------------------


def test_shifted_timesteps() -> None:
    u = torch.tensor([0.0, 0.25, 0.5, 1.0])
    t = shifted_timesteps(u, 5.0)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert torch.allclose(t, 5 * u / (1 + 4 * u))
    # The median of t is 5/6 for shift 5.
    assert abs(float(shifted_timesteps(torch.tensor(0.5), 5.0)) - 5 / 6) < 1e-6
    assert torch.equal(shifted_timesteps(u, 1.0), u)


def test_draws_follow_the_draw_order() -> None:
    """The draws are the plain generator calls in the documented order."""
    model = _model()
    batch = _window()
    d = model.draw(batch, torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(3)
    text_u = torch.rand(1, generator=g)
    assert torch.equal(d.text_u, text_u)
    assert torch.equal(d.video_u0, torch.rand(1, generator=g))
    eps0 = torch.randn(1, 192, 24, H, W, generator=g).to(torch.bfloat16)
    assert torch.equal(_bytes(d.video_noise), _bytes(eps0))
    assert torch.equal(d.video_u, torch.rand(4, generator=g))
    n_text = 1 if float(text_u) < 0.1 else 7
    spans = (W + H * ROW_STRIDE, n_text, W + H * ROW_STRIDE, 2 * W + 2 * H * ROW_STRIDE)
    shifts = tuple(
        int(torch.randint(0, FRAME_STRIDE - sp, (1,), generator=g)) for sp in spans
    )
    assert d.rope_shifts == shifts
    assert torch.equal(d.action_u, torch.rand(4, generator=g))
    assert torch.equal(d.action_noise, torch.randn(1, 24, 40, generator=g))
    assert torch.equal(d.proprio_u, torch.rand(1, generator=g))
    again = model.draw(batch, torch.Generator().manual_seed(3))
    assert again.rope_shifts == d.rope_shifts
    assert torch.equal(_bytes(again.video_noise), _bytes(d.video_noise))


def test_text_drop_changes_the_text_span() -> None:
    model = _model(text_dropout=1.0)
    d = model.draw(_window(n_text=7), torch.Generator().manual_seed(0))
    assert 0 <= d.rope_shifts[1] < FRAME_STRIDE - 1
    model.eval()
    # Eval mode: no dropout, the span of the real text.
    d = model.draw(_window(n_text=7), torch.Generator().manual_seed(0))
    assert 0 <= d.rope_shifts[1] < FRAME_STRIDE - 7


def test_draws_dataclass_holds_all_values() -> None:
    assert list(Draws.__dataclass_fields__) == [
        "text_u",
        "video_u0",
        "video_noise",
        "video_u",
        "rope_shifts",
        "action_u",
        "action_noise",
        "proprio_u",
    ]


def test_two_stage_video_noising() -> None:
    """The window noise at t0, the recovery of x0 and eps, the per-chunk noise."""
    model = _model()
    batch = _window(1)
    d = model.draw(batch, torch.Generator().manual_seed(5))
    noised, clean, t_k, target = model.video_inputs(batch, d)
    bf = torch.bfloat16
    x0 = torch.cat(batch.chunks, dim=2)
    # A 5-D float32 timestep: the products run in float32, as in the original training code.
    t0 = shifted_timesteps(d.video_u0, 5.0)
    t5 = t0.view(1, 1, 1, 1, 1)
    xt = (x0 * (1 - t5) + d.video_noise * t5).to(bf)
    v = d.video_noise - x0
    assert target.dtype == bf and torch.equal(_bytes(target), _bytes(v))
    x0r = xt.float() - t0.to(bf)[0].float() * v.float()
    epsr = x0r + v.float()
    tk = shifted_timesteps(d.video_u, 5.0).to(bf)
    assert torch.equal(t_k, tk.view(1, 4))
    for k in range(4):
        sl = slice(6 * k, 6 * k + 6)
        want = x0r[:, :, sl] * (1 - tk[k].float()) + epsr[:, :, sl] * tk[k].float()
        assert torch.equal(_bytes(noised[:, :, sl]), _bytes(want.to(bf)))
    assert clean is not None and clean.shape == (1, 192, 18, H, W)
    assert torch.equal(_bytes(clean), _bytes(x0r[:, :, :18].to(bf)))
    # The recovered x0 is close to the latents, but not equal (bf16 rounding).
    assert torch.allclose(clean.float(), x0[:, :, :18].float(), atol=0.1)


def test_layout_check() -> None:
    config = tiny_config()
    config["backbone"] = dict(config["backbone"], layout="serve")
    with pytest.raises(ValueError, match="layout"):
        build_backbone(config["backbone"])
    assert isinstance(tiny_policy(), PraxisPolicy)


def test_constructor_sets_the_training_kernels() -> None:
    model = _model()
    layers = model.moe_layers
    assert layers and all(m.dispatch_chunk_size == 65536 for m in layers)
    model = _model(moe_dispatch_chunk_size=7)
    assert all(m.dispatch_chunk_size == 7 for m in model.moe_layers)
    # The tiny expert runs SDPA: no flex options to extend.
    assert model.policy.action_expert.flex_kernel_options is None
    assert set(EXPERT_FLEX_BACKWARD_OPTIONS) == {
        "bwd_BLOCK_M1",
        "bwd_BLOCK_M2",
        "bwd_BLOCK_N1",
        "bwd_BLOCK_N2",
    }


# ---------------------------------------------------------------------------
# MoE balance
# ---------------------------------------------------------------------------


def test_expert_counts() -> None:
    top = torch.tensor([[0, 2], [2, 3], [2, 0]])
    assert torch.equal(expert_counts(top, 5), torch.tensor([2.0, 0.0, 3.0, 1.0, 0.0]))


def test_sequence_balance_loss_matches_the_formula() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 8, generator=g, dtype=torch.float32)
    picks = torch.topk(logits, 3, dim=-1)[1]
    loss = sequence_balance_loss(logits, picks)
    scores = torch.sigmoid(logits)
    p = (scores / scores.sum(-1, keepdim=True)).mean(0)
    f = expert_counts(picks, 8) / (50 * 3)
    assert torch.allclose(loss, 8 * (f * p).sum(), rtol=1e-6)
    # Uniform scores and uniform picks give 1.
    flat = torch.zeros(8, 8)
    picks = torch.stack([torch.arange(8), (torch.arange(8) + 1) % 8], 1)
    assert abs(float(sequence_balance_loss(flat, picks)) - 1.0) < 1e-6


def _stats(moe: MoEFeedForward) -> RoutingStats:
    stats = moe.routing_hook
    assert isinstance(stats, RoutingStats)
    return stats


def _moe() -> MoEFeedForward:
    torch.manual_seed(0)
    moe = MoEFeedForward(dim=16, hidden=8, num_experts=4, top_k=2)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0, 0.3)
    return moe


def test_routing_stats_record_and_bias_update() -> None:
    moe = _moe()
    model = torch.nn.Sequential(moe)
    layers = track_routing(model, seq_balance=True, unbiased_picks=True)
    assert layers == [moe]
    model.train()
    x = torch.randn(2, 10, 16)
    moe(x)
    moe(x[:, :5])
    stats = _stats(moe)
    assert stats.counts is not None
    assert stats.tokens == 30 and float(stats.counts.sum()) == 60.0
    losses = take_sequence_balance_losses(layers)
    assert len(losses) == 1 and losses[0].requires_grad
    assert take_sequence_balance_losses(layers) == []
    counts = stats.counts.clone()
    before = moe.expert_bias.clone()
    update_expert_biases(model, 0.5)
    target = 30 * 2 / 4
    step = ((counts < target).float() - (counts > target).float()) * 0.5
    assert torch.equal(moe.expert_bias, before + step)
    assert moe.expert_bias.dtype == torch.float32
    assert stats.counts is None and stats.tokens == 0
    # No tokens since the update: the bias does not move.
    update_expert_biases(model, 0.5)
    assert torch.equal(moe.expert_bias, before + step)


def test_routing_stats_skip_eval_and_recompute() -> None:
    moe = _moe()
    track_routing(moe, seq_balance=True, unbiased_picks=False)
    moe.eval()
    moe(torch.randn(1, 4, 16))
    assert _stats(moe).counts is None
    moe.train()
    x = torch.randn(1, 6, 16, requires_grad=True)
    out = checkpoint(moe, x, use_reentrant=False)
    assert isinstance(out, Tensor)
    out.sum().backward()
    # The recompute in the backward pass records nothing.
    assert _stats(moe).tokens == 6


def test_unbiased_picks_ignore_the_bias() -> None:
    moe = _moe()
    track_routing(moe, seq_balance=True, unbiased_picks=True)
    moe.train()
    x = torch.randn(1, 12, 16)
    moe(x)
    a = take_sequence_balance_losses([moe])[0]
    with torch.no_grad():
        moe.expert_bias.copy_(torch.tensor([5.0, 5.0, -5.0, -5.0]))
    moe(x)
    b = take_sequence_balance_losses([moe])[0]
    assert torch.equal(a, b)
    track_routing(moe, seq_balance=True, unbiased_picks=False)
    moe(x)
    c = take_sequence_balance_losses([moe])[0]
    assert not torch.equal(a, c)


def test_update_expert_biases_zero_speed_clears_counts() -> None:
    moe = _moe()
    track_routing(moe, seq_balance=False, unbiased_picks=False)
    moe.train()
    moe(torch.randn(1, 5, 16))
    before = moe.expert_bias.clone()
    update_expert_biases(moe, 0.0)
    assert torch.equal(moe.expert_bias, before)
    assert _stats(moe).counts is None


# ---------------------------------------------------------------------------
# Custom backward passes
# ---------------------------------------------------------------------------


def test_router_logits_backward() -> None:
    """The float32 router GEMM rounds its gradients once to bf16."""
    g = torch.Generator().manual_seed(0)
    xb = torch.randn(9, 16, generator=g).to(torch.bfloat16).requires_grad_(True)
    wb = torch.randn(4, 16, generator=g).to(torch.bfloat16).requires_grad_(True)
    out = _router_logits(wb, xb)
    assert out.dtype == torch.float32
    grad = torch.randn(9, 4, generator=g)
    out.backward(grad)
    assert xb.grad is not None and wb.grad is not None
    assert xb.grad.dtype == torch.bfloat16 and wb.grad.dtype == torch.bfloat16
    exp_x = (grad @ wb.detach().float()).to(torch.bfloat16)
    exp_w = (grad.t() @ xb.detach().float()).to(torch.bfloat16)
    assert torch.equal(_bytes(xb.grad), _bytes(exp_x))
    assert torch.equal(_bytes(wb.grad), _bytes(exp_w))
    # The forward without grad is the plain float32 GEMM.
    with torch.no_grad():
        ref = F.linear(xb.float(), wb.float())
        assert torch.equal(_router_logits(wb, xb), ref)


def test_permute_rows_backward_is_a_gather() -> None:
    src = torch.randn(6, 3, requires_grad=True)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(6)
    out = _PermuteRows.apply(src, order, inverse)
    assert torch.equal(out, src[order])
    grad = torch.randn(6, 3)
    grad[2, 1] = -0.0
    out.backward(grad)
    assert src.grad is not None
    assert torch.equal(_bytes(src.grad), _bytes(grad[inverse]))


def test_moe_backward_matches_eager_math() -> None:
    """The MoE backward in float32 against an eager per-token loop."""
    moe = _moe()
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = moe(x)
    # Eager reference with the same routing.
    xf = x.reshape(-1, 16)
    logits = xf @ moe.router.weight.t()
    scores = torch.sigmoid(logits)
    top = torch.topk(scores + moe.expert_bias, 2, dim=-1)[1]
    picked = scores.gather(-1, top)
    gate = picked / picked.sum(-1, keepdim=True)
    gu, down = moe.routed.gate_up_weight, moe.routed.down_weight
    rows = []
    for t in range(xf.shape[0]):
        acc = 0.0
        for j in range(2):
            e = int(top[t, j])
            h = xf[t] @ gu[e]
            a, b = h.chunk(2)
            acc = acc + gate[t, j] * ((F.silu(a) * b) @ down[e])
        rows.append(acc)
    ref = torch.stack(rows) * moe.routed_scaling_factor + moe.shared(xf)
    assert torch.allclose(out.reshape(-1, 16), ref, atol=1e-5)
    g = torch.randn_like(out)
    params = [x, moe.router.weight, moe.routed.gate_up_weight, moe.routed.down_weight]
    got = torch.autograd.grad(out, params, g, retain_graph=True)
    want = torch.autograd.grad(ref.reshape_as(out), params, g)
    for a, b in zip(got, want, strict=True):
        assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# Chunked expert dispatch
# ---------------------------------------------------------------------------


def test_chunked_dispatch_backward_matches_autograd() -> None:
    """The custom backward of the chunked dispatch equals autograd of the
    forward math in float32, for one and for several token chunks."""
    for chunk in (100, 4):
        moe = _moe()
        moe.dispatch_chunk_size = chunk
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = moe(x)
        moe.dispatch_chunk_size = None
        x2 = x.detach().clone().requires_grad_(True)
        ref = moe(x2)
        assert torch.allclose(out, ref, atol=1e-5)
        g = torch.randn_like(out)
        params = [moe.router.weight, moe.routed.gate_up_weight, moe.routed.down_weight]
        moe.dispatch_chunk_size = chunk
        got = torch.autograd.grad(moe(x), [x, *params], g)
        moe.dispatch_chunk_size = None
        want = torch.autograd.grad(moe(x2), [x2, *params], g)
        for a_, b_ in zip(got, want, strict=True):
            assert torch.allclose(a_, b_, atol=1e-5), (chunk, (a_ - b_).abs().max())


# ---------------------------------------------------------------------------
# The loss with a backbone stand-in
# ---------------------------------------------------------------------------


class _FakeBackbone:
    """Records the backbone inputs. Returns a zero video velocity and random
    taps in the chunk-causal layout (registers 4, anchor, text, 24 noised and
    18 clean frames of ``H * W`` tokens)."""

    def __init__(self, seen: dict) -> None:
        self.seen = seen
        self.g = torch.Generator().manual_seed(2)

    def __call__(self, text, anchor, clean, noised, timesteps, rope_shifts):
        self.seen.update(
            text=text, clean=clean, noised=noised, steps=timesteps, shifts=rope_shifts
        )
        tpf = H * W
        n_text = text.shape[1]
        n_g = 4 + tpf + n_text
        n_n, n_c = 24 * tpf, 18 * tpf
        total = n_g + n_n + n_c
        rope = torch.cat(
            [
                torch.zeros(4, dtype=torch.long),
                torch.arange(tpf) + 1,
                torch.arange(n_text) + FRAME_STRIDE + 1,
                torch.arange(n_n) + 2 * FRAME_STRIDE,
                torch.arange(n_c) + 2 * FRAME_STRIDE,
            ]
        )[None]
        taps = TapFeatures(
            taps=[torch.randn(1, total, 64, generator=self.g) for _ in range(2)],
            embeddings=torch.randn(1, total, 64, generator=self.g),
            frames_per_chunk=6,
            tokens_per_frame=tpf,
            rope_ids=rope,
            global_spans={
                "register": (0, 4),
                "anchor": (4, tpf),
                "text": (4 + tpf, n_text),
            },
            noised_span=(n_g, n_n),
            clean_span=(n_g + n_n, n_c),
            n_chunks=4,
        )
        out = torch.zeros_like(noised, requires_grad=False)
        return type("Out", (), {"velocity": out, "taps": taps})()


def _run_with_stand_in(
    monkeypatch: pytest.MonkeyPatch, model: FinetuneModel, draws: Draws
) -> tuple[Tensor, dict[str, Tensor], dict]:
    import praxis.train.objective as obj

    seen: dict = {}
    fake = _FakeBackbone(seen)

    def forward_chunk_causal(text, anchor, clean, noised, timesteps, rope_shifts):
        return fake(text, anchor, clean, noised, timesteps, rope_shifts)

    real_run = obj.run_expert_on_taps_k

    def run(*args, **kw):
        out = real_run(*args, **kw)
        seen.update(kw=kw, out=out)
        return out

    monkeypatch.setattr(
        model.policy.backbone, "forward_chunk_causal", forward_chunk_causal
    )
    monkeypatch.setattr(obj, "run_expert_on_taps_k", run)
    loss, metrics = model.loss(_window(), draws)
    return loss, metrics, seen


def test_action_loss_and_dropout_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``FinetuneModel.loss`` with a backbone stand-in (no flex attention).

    The real action expert runs on random taps. The test checks the noised
    actions, the blank caption of the text dropout, the zero proprio, the
    loss formula and the gradient path into the action expert.
    """
    model = _model(dtype=torch.float32)
    batch = _window()
    draws = model.draw(batch, torch.Generator().manual_seed(1))
    draws.text_u = torch.zeros(1)
    draws.proprio_u = torch.zeros(1)
    loss, metrics, seen = _run_with_stand_in(monkeypatch, model, draws)

    assert torch.equal(seen["text"], UNCOND.float())
    assert seen["shifts"] == draws.rope_shifts
    kw = seen["kw"]
    assert torch.count_nonzero(kw["proprio_per_chunk"]) == 0
    assert torch.equal(kw["action_t_per_chunk"], draws.action_u.view(1, 4))
    x0 = batch.actions
    t_tok = draws.action_u.repeat_interleave(6)[None, :, None]
    want = (1 - t_tok) * x0 + t_tok * draws.action_noise
    assert torch.equal(kw["noised_actions"], want)
    target = draws.action_noise - x0
    err = (seen["out"][0].float() - target[0]).square()
    exp_a = err.mean(-1).mean()
    assert torch.equal(metrics["loss_action"], exp_a.detach())
    assert float(metrics["action_dropped"]) == 0.0
    # Zero video velocity: the video loss is the mean of (eps0 - x0)**2.
    x0v = torch.cat(batch.chunks, dim=2).float()
    v = (draws.video_noise.float() - x0v).float()
    exp_v = v.pow(2).mean()
    assert abs(float(metrics["loss_video"]) - float(exp_v)) < 1e-4
    assert "loss_moe_balance" not in metrics  # the stand-in runs no MoE layer
    exp_total = metrics["loss_video"] + metrics["loss_action"]
    assert torch.allclose(loss.detach(), exp_total)
    loss.backward()
    head = model.policy.action_expert.head.weight.grad
    assert head is not None and torch.count_nonzero(head) > 0


def test_no_dropout_and_no_shift_in_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(dtype=torch.float32)
    batch = _window()
    draws = model.draw(batch, torch.Generator().manual_seed(1))
    draws.text_u = torch.zeros(1)
    draws.proprio_u = torch.zeros(1)
    model.eval()
    _, _, seen = _run_with_stand_in(monkeypatch, model, draws)
    assert torch.equal(seen["text"], batch.text.float())
    assert seen["shifts"] == (0, 0, 0, 0)
    assert torch.equal(seen["kw"]["proprio_per_chunk"], batch.proprio)


def test_action_guard_drops_the_action_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(dtype=torch.float32, action_guard=1e-6)
    draws = model.draw(_window(), torch.Generator().manual_seed(1))
    loss, metrics, _ = _run_with_stand_in(monkeypatch, model, draws)
    assert float(metrics["action_dropped"]) == 1.0
    assert float(metrics["loss_action"]) > 1e-6
    assert torch.allclose(loss.detach(), metrics["loss_video"])
    loss.backward()
    head = model.policy.action_expert.head.weight.grad
    assert head is not None and torch.count_nonzero(head) == 0


def test_text_dropout_needs_the_blank_caption() -> None:
    model = FinetuneModel(tiny_policy(), ObjectiveSettings(text_dropout=1.0))
    with pytest.raises(ValueError, match="blank caption"):
        model.draw(_window(), torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# The full loss forward of a tiny chunk-causal policy
# ---------------------------------------------------------------------------


def test_full_loss_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real tiny backbone (eager flex attention, forward only)."""
    from torch.nn.attention.flex_attention import flex_attention

    import praxis.layers

    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    model = _model(dtype=torch.float32, moe_dispatch_chunk_size=50)
    batch = _window()
    with torch.no_grad():
        draws = model.draw(batch, torch.Generator().manual_seed(4))
        loss, metrics = model.loss(batch, draws)
        assert torch.isfinite(loss)
        assert set(metrics) == {
            "loss_video",
            "loss_action",
            "action_dropped",
            "loss_moe_balance",
        }
        # The routing counts of the MoE layers are recorded once per layer.
        counts = [m.routing_hook.counts for m in model.moe_layers]  # ty: ignore[unresolved-attribute]
        assert all(c is not None for c in counts)
        # Eval: no balance term, the same video inputs.
        model.eval()
        loss_e, metrics_e = model.loss(batch, draws)
        assert torch.isfinite(loss_e) and "loss_moe_balance" not in metrics_e


# ---------------------------------------------------------------------------
# encode_window
# ---------------------------------------------------------------------------


class _FakeVae:
    """A stand-in autoencoder: 4x4 average pool and the causal time stride."""

    latent_mean = torch.zeros(1)

    def encode(self, clip: Tensor) -> Tensor:
        x = F.avg_pool3d(clip[:, :, :1], (1, 32, 32))
        first = x.repeat(1, 64, 1, 1, 1)[:, :192]
        rest = clip[:, :, 1:]
        if rest.shape[2] == 0:
            return first.to(torch.bfloat16)
        t = rest.shape[2] // 4
        y = F.avg_pool3d(rest[:, :, : 4 * t], (4, 32, 32)).repeat(1, 64, 1, 1, 1)
        return torch.cat([first, y[:, :192]], dim=2).to(torch.bfloat16)


class _FakeText:
    def encode(self, prompts: list[str]) -> list[Tensor]:
        return [torch.full((1, 3, 2048), float(len(p))) for p in prompts]


def test_encode_window_shapes_and_normalization() -> None:
    # The data module needs the packages of the "train" extra.
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    from praxis.train.data import TrainingWindow

    rng = np.random.default_rng(0)
    q01 = np.full(10, -0.5, dtype=np.float32)
    q99 = np.full(10, 0.5, dtype=np.float32)
    config = {
        "action_space": {"dim": 10},
        "chunking": {"n_chunks": 4, "action_chunk": 24},
        "normalization": {
            "action_q01": q01.tolist(),
            "action_q99": q99.tolist(),
            "q99_normed_dims": [0, 1, 2, 9],
            "proprio_mean": [0.1] * 10,
            "proprio_std": [2.0] * 10,
        },
    }
    actions = rng.uniform(-1, 1, (93, 10)).astype(np.float32)
    proprio = rng.uniform(-1, 1, (4, 10)).astype(np.float32)
    window = TrainingWindow(
        frames=torch.randint(0, 256, (93, 3, 64, 128), dtype=torch.uint8),
        prompt="put the cup",
        actions_rel=actions,
        proprio_abs=proprio,
        meta={},
    )
    vae = _FakeVae()
    enc = encode_window(window, vae, _FakeText(), config, "cpu")  # ty: ignore[invalid-argument-type]
    assert enc.text.shape == (1, 3, 2048) and enc.text.dtype == torch.bfloat16
    assert enc.anchor.shape == (1, 192, 1, 2, 4)
    assert len(enc.chunks) == 4 and all(
        c.shape == (1, 192, 6, 2, 4) for c in enc.chunks
    )
    assert enc.actions.shape == (1, 24, 40) and enc.actions.dtype == torch.float32
    assert enc.proprio.shape == (1, 4, 10)
    norm = q99_normalize(actions, q01, q99, dims=np.array([0, 1, 2, 9]))
    first = pack_action_tokens(norm[0:21], first_chunk=True)
    assert np.array_equal(enc.actions[0, :6].numpy(), first)
    second = pack_action_tokens(norm[21:45], first_chunk=False)
    assert np.array_equal(enc.actions[0, 6:12].numpy(), second)
    assert np.allclose(enc.proprio[0].numpy(), (proprio - 0.1) / 2.0)
    # The rot6d dims are not normalized.
    assert np.array_equal(norm[:, 3:9], actions[:, 3:9])
    # The anchor is a 1-frame encode of frame 0, the chunks one window encode.
    clip = window.frames.permute(1, 0, 2, 3)[None].float()
    assert torch.equal(enc.anchor, vae.encode(clip[:, :, :1]))
    assert torch.equal(torch.cat(enc.chunks, dim=2), vae.encode(clip))
    with pytest.raises(ValueError, match="frames"):
        short = TrainingWindow(window.frames[:89], "x", actions, proprio, {})
        encode_window(short, vae, _FakeText(), config, "cpu")  # ty: ignore[invalid-argument-type]


def test_objective_settings_convert_strings() -> None:
    s = ObjectiveSettings(moe_seq_balance_coef="1e-3", video_weight="5e-1")  # ty: ignore[invalid-argument-type]
    assert s.moe_seq_balance_coef == 1e-3 and s.video_weight == 0.5
    floats = [
        k
        for k in s.__dataclass_fields__
        if k not in ("fixed_draws", "moe_dispatch_chunk_size")
    ]
    assert all(type(getattr(s, k)) is float for k in floats)
    assert s.fixed_draws is False
    assert ObjectiveSettings(fixed_draws=True).fixed_draws is True
    with pytest.raises(ValueError):
        ObjectiveSettings(fixed_draws="yes")  # ty: ignore[invalid-argument-type]
    d = ObjectiveSettings()
    assert (d.video_weight, d.action_weight, d.action_guard) == (1.0, 1.0, 5.0)
    assert (d.text_dropout, d.proprio_dropout, d.video_timestep_shift) == (
        0.1,
        0.1,
        5.0,
    )
    assert (d.moe_balance_speed, d.moe_seq_balance_coef) == (1e-3, 1e-4)
    assert d.moe_dispatch_chunk_size == 65536


def test_blocks_are_the_transformer_blocks() -> None:
    model = _model()
    policy = model.policy
    blocks = model.blocks()
    n = (
        len(policy.backbone.encoder.core.blocks)
        + len(policy.backbone.decoder.core.blocks)
        + len(policy.action_expert.blocks)
    )
    assert len(blocks) == n
    names = {id(m) for m in blocks}
    assert id(policy.backbone.encoder.core.blocks[0]) in names
    assert id(policy.action_expert.blocks[-1]) in names


def test_decoder_block_call_runs_its_modulation() -> None:
    """The modulator of a decoder block runs inside the block call.

    Thus the activation checkpoint of a block also covers its modulation.
    """
    policy = tiny_policy(dtype=torch.float32)
    core = policy.backbone.decoder.core
    assert core.modulators is not None
    inside: list[bool] = []
    depth = [0]
    for block in core.blocks:
        block.register_forward_pre_hook(lambda *_: depth.__setitem__(0, 1))
        block.register_forward_hook(lambda *_: depth.__setitem__(0, 0))
    for mod in core.modulators:
        mod.register_forward_hook(lambda *_: inside.append(depth[0] == 1))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 40, 64, generator=g)
    x_mod = torch.randn(1, 40, 64, generator=g)
    rope = core.rope_table(torch.arange(40)[None], 40)
    with torch.no_grad():
        out = core.block_forward(0, x, rope, x_mod)
        shift, scale = core.modulators[0](x_mod).chunk(2, dim=-1)
        want = core.blocks[0](x * (1 + scale) + shift, rope)
    assert inside == [True, False]
    assert torch.allclose(out, want, atol=1e-6)


# ---------------------------------------------------------------------------
# Routing bias update with FSDP on two processes
# ---------------------------------------------------------------------------


def _bias_worker(rank: int, world: int, init: str, out: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        moe = _moe()
        model = torch.nn.Sequential(moe)
        track_routing(model, seq_balance=False, unbiased_picks=False)
        mesh = init_device_mesh("cpu", (world,))
        fully_shard(moe, mesh=mesh)
        model.train()
        g = torch.Generator().manual_seed(100 + rank)
        for _ in range(2):  # two micro-steps on each rank
            moe(torch.randn(1, 7, 16, generator=g)).sum().backward()
        update_expert_biases(model, 0.25)
        if rank == 0:
            torch.save(moe.expert_bias.clone(), out)
    finally:
        dist.destroy_process_group()


def test_bias_update_adds_micro_steps_and_ranks(tmp_path: object) -> None:
    import tempfile
    from pathlib import Path

    import torch.multiprocessing as mp

    out = str(Path(str(tmp_path)) / "bias.pt")
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_bias_worker, args=(2, f"file://{d}/pg", out), nprocs=2, join=True)
    got = torch.load(out)
    # The same counts on one process: 2 ranks x 2 micro-steps of 7 tokens.
    moe = _moe()
    counts = torch.zeros(4)
    for rank in range(2):
        g = torch.Generator().manual_seed(100 + rank)
        for _ in range(2):
            x = torch.randn(1, 7, 16, generator=g).reshape(-1, 16)
            scores = torch.sigmoid(_router_logits(moe.router.weight, x))
            top = torch.topk(scores + moe.expert_bias, 2, dim=-1)[1]
            counts += expert_counts(top, 4)
    target = 28 * 2 / 4
    want = ((counts < target).float() - (counts > target).float()) * 0.25
    assert torch.equal(got, want)
