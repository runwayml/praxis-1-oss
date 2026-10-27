"""Unit tests for the fine-tuning trainer (CPU only, no weights).

The distributed tests start 1 or 2 processes with the gloo backend. The model
is a small random praxis policy. The flex attention of the backbone has no
CPU backward, so the loss of these tests uses the parts of the policy that
run on the CPU: the input projections, the norm and feed-forward (dense and
MoE) of each encoder block, and the full action expert.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from tiny_model import tiny_config, tiny_policy
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from praxis.action_expert import run_expert_on_taps_k
from praxis.backbone import TEXT, VIDEO, chunk_causal_positions
from praxis.checkpoint import (
    CheckpointError,
    load_config,
    save_checkpoint,
    verify_checksums,
)
from praxis.layers import TransformerBlock
from praxis.model import PraxisPolicy
from praxis.train.lora import LoRASettings, apply_lora, merged_state_dict
from praxis.train.objective import EncodedWindow, FinetuneModel, ObjectiveSettings
from praxis.train.trainer import (
    FIXED_DRAWS_KEY,
    TRAINABLE_NAME,
    AdamW,
    TextCache,
    Trainer,
    TrainSettings,
    broadcast_windows,
    cycle_windows,
    derive_seed,
    round_stochastic_,
)

STEPS = 3


def _bytes(t: Tensor) -> Tensor:
    """The raw bytes of ``t``, so that -0.0 and +0.0 differ."""
    return t.detach().reshape(-1).view(torch.uint8)


def make_windows(n: int, seed: int = 1) -> list[EncodedWindow]:
    g = torch.Generator().manual_seed(seed)

    def lat(frames: int) -> Tensor:
        return torch.randn(1, 192, frames, 2, 4, generator=g).to(torch.bfloat16)

    return [
        EncodedWindow(
            text=torch.randn(1, 5, 32, generator=g).to(torch.bfloat16),
            anchor=lat(1),
            chunks=[lat(6) for _ in range(4)],
            actions=torch.rand(1, 24, 40, generator=g) * 2 - 1,
            proprio=torch.randn(1, 4, 10, generator=g),
        )
        for _ in range(n)
    ]


class StandIn(nn.Module):
    """A flow-matching loss over the CPU parts of a praxis policy.

    The video latents and the actions get noise and a timestep per chunk from
    ``generator``. The encoder blocks run without attention on the
    chunk-causal sequence ``[registers | anchor | text | noised | clean]``.
    The action expert reads the block outputs as its taps.
    """

    def __init__(self, policy: PraxisPolicy) -> None:
        super().__init__()
        self.policy = policy

    def encoder_blocks(self) -> list[TransformerBlock]:
        blocks = list(self.policy.backbone.encoder.core.blocks)
        assert all(isinstance(b, TransformerBlock) for b in blocks)
        return blocks  # ty: ignore[invalid-return-type]

    def blocks(self) -> list[nn.Module]:
        out: list[nn.Module] = [b.ffn for b in self.encoder_blocks()]
        return out + list(self.policy.action_expert.blocks)

    def forward(
        self, batch: EncodedWindow, generator: torch.Generator
    ) -> tuple[Tensor, dict[str, Tensor]]:
        enc = self.policy.backbone.encoder
        md = enc.registers.dtype
        t_v = torch.rand(4, generator=generator)
        noised = []
        for k, c in enumerate(batch.chunks):
            eps = torch.randn(c.shape, generator=generator)
            noised.append(((1 - t_v[k]) * c.float() + t_v[k] * eps).to(md))
        a = batch.actions
        eps_a = torch.randn(a.shape, generator=generator)
        t_a = torch.rand(1, 4, generator=generator)
        t_tok = t_a.repeat_interleave(a.shape[1] // 4, dim=1)[..., None]
        x_a = (1 - t_tok) * a + t_tok * eps_a

        text = enc.input_projections[TEXT](batch.text.to(md))
        anchor = enc.input_projections[VIDEO](batch.anchor.to(md)).flatten(1, -2)
        noised_v = torch.cat(noised, dim=2)
        clean_v = torch.cat([c.to(md) for c in batch.chunks[:3]], dim=2)
        flat_n = enc.input_projections[VIDEO](noised_v).flatten(1, -2)
        flat_c = enc.input_projections[VIDEO](clean_v).flatten(1, -2)
        x0 = torch.cat([enc.registers, anchor, text, flat_n, flat_c], dim=1)
        x, taps = x0, []
        for i, blk in enumerate(self.encoder_blocks()):
            x = x + blk.ffn(blk.ln_out(x) * blk.s)
            if i in enc.tap_blocks:
                taps.append(x)
        n_reg, n_anc, n_txt = enc.registers.shape[1], anchor.shape[1], text.shape[1]
        n_g = n_reg + n_anc + n_txt
        out = enc.output_projections[VIDEO](x[:, n_g : n_g + flat_n.shape[1]])
        target = torch.randn(out.shape, generator=generator)
        loss_v = (out.float() - target).pow(2).mean()
        h, w = batch.anchor.shape[-2:]
        rope = chunk_causal_positions(n_reg, (h, w), n_txt, (24, h, w), 18)[None]
        v_a = run_expert_on_taps_k(
            self.policy.action_expert,
            taps,
            x0,
            rope,
            global_spans={
                "register": (0, n_reg),
                "anchor": (n_reg, n_anc),
                "text": (n_reg + n_anc, n_txt),
            },
            noised_span=(n_g, flat_n.shape[1]),
            clean_span=(n_g + flat_n.shape[1], flat_c.shape[1]),
            n_chunks=4,
            frames_per_chunk=6,
            tokens_per_frame=h * w,
            noised_actions=x_a,
            action_t_per_chunk=t_a,
            proprio_per_chunk=batch.proprio,
        )
        loss_a = (v_a.float() - (eps_a - a)).pow(2).mean()
        loss = loss_v + loss_a
        return loss, {"loss_video": loss_v.detach(), "loss_action": loss_a.detach()}


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


def test_round_stochastic_is_exact_and_unbiased() -> None:
    nan = round_stochastic_(
        torch.full((4,), float("nan")), torch.full((4,), 0xFFFF, dtype=torch.int32)
    )
    assert bool(nan.isnan().all())
    exact = torch.tensor([0.0, -0.0, 1.0, -3.5, 2.0**-120], dtype=torch.bfloat16)
    x = exact.float()
    noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
    assert torch.equal(
        round_stochastic_(x.clone(), noise).view(torch.int32), x.view(torch.int32)
    )
    lo = torch.tensor(1.0)
    hi = torch.tensor(1.0 + 2.0**-7)  # the next bfloat16 value
    for frac in (0.1, 0.5, 0.9):
        x = torch.full((200_000,), 1.0 + frac * 2.0**-7)
        r = round_stochastic_(
            x.clone(), torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
        )
        assert bool(((r == lo) | (r == hi)).all())
        assert abs(float(r.double().mean()) - (1.0 + frac * 2.0**-7)) < 2e-5
        neg = round_stochastic_(
            -x, torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
        )
        assert bool((neg <= -lo).all()) and bool((neg >= -hi).all())


def test_adamw_float32_matches_torch() -> None:
    g = torch.Generator().manual_seed(0)
    p0 = torch.randn(64, 8, generator=g)
    grads = [torch.randn(64, 8, generator=g) for _ in range(5)]
    mine, ref = nn.Parameter(p0.clone()), nn.Parameter(p0.clone())
    opt_mine = AdamW([mine], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_ref = torch.optim.AdamW(
        [ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    for grad in grads:
        mine.grad, ref.grad = grad.clone(), grad.clone()
        opt_mine.step()
        opt_ref.step()
    torch.testing.assert_close(mine, ref, rtol=1e-6, atol=1e-7)


def test_adamw_bfloat16_moves_small_updates() -> None:
    """An update of 1/80 of the bfloat16 spacing still moves the mean."""
    p = nn.Parameter(torch.ones(20_000, dtype=torch.bfloat16))
    opt = AdamW([p], lr=1e-4, weight_decay=0.0, seed=3)
    for _ in range(100):
        p.grad = torch.ones_like(p)
        opt.step()
    assert abs(float(p.double().mean()) - (1.0 - 100 * 1e-4)) < 3e-4
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16


def test_save_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = tiny_policy()
    state = policy.state_dict()
    config = tiny_config()
    save_checkpoint(tmp_path / "ckpt", state, config, max_shard_bytes=200_000)
    verify_checksums(tmp_path / "ckpt")
    index = json.loads((tmp_path / "ckpt" / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) > 2 and shards[0].endswith(f"-of-{len(shards):05d}.safetensors")
    total = sum(t.numel() * t.element_size() for t in state.values())
    assert index["metadata"]["total_size"] == total
    assert load_config(tmp_path / "ckpt") == config
    loaded = PraxisPolicy.from_pretrained(tmp_path / "ckpt", device="cpu")
    for name, t in loaded.state_dict().items():
        assert t.dtype == state[name].dtype, name
        assert torch.equal(_bytes(t), _bytes(state[name])), name
    biases = [b for n, b in loaded.named_buffers() if n.endswith("expert_bias")]
    assert biases and all(b.dtype == torch.float32 for b in biases)
    with pytest.raises(CheckpointError):
        save_checkpoint(tmp_path / "ckpt", state, config)


def test_text_cache_keeps_recent_prompts() -> None:
    calls: list[str] = []

    class Encoder:
        def encode(self, prompts: list[str]) -> list[Tensor]:
            calls.extend(prompts)
            return [torch.full((1, 1, 2), float(len(p))) for p in prompts]

    cache = TextCache(Encoder(), size=2)  # ty: ignore[invalid-argument-type]
    for p in ["a", "bb", "a", "ccc", "bb", "a"]:
        assert float(cache.encode([p])[0][0, 0, 0]) == len(p)
    assert calls == ["a", "bb", "ccc", "bb", "a"]


def test_config_holds_the_recipe() -> None:
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "finetune_droid.yaml"
    cfg = yaml.safe_load(path.read_text())
    s = TrainSettings(**cfg["train"])
    assert cfg["model"]["checkpoint"] == "checkpoints/praxis-1-droid"
    assert (s.lr, s.warmup_steps, s.betas) == (3e-6, 50, (0.9, 0.95))
    assert (s.lr_schedule, s.min_lr, s.steps) == ("cosine", 3e-7, 1200)
    assert (s.grad_accum, s.save_every) == (16, 100)
    assert (s.weight_decay, s.eps, s.clip_grad_norm) == (1e-4, 1e-8, 1.0)
    assert cfg["data"]["source"] == "droid_hires"
    assert cfg["objective"]["text_dropout"] == 0.1
    # The overfit options are off.
    assert s.save is True and cfg["data"]["overfit_windows"] is None
    assert ObjectiveSettings(**cfg["objective"]).fixed_draws is False


def test_overfit_config() -> None:
    import yaml

    configs = Path(__file__).resolve().parents[1] / "configs"
    recipe = yaml.safe_load((configs / "finetune_droid.yaml").read_text())
    cfg = yaml.safe_load((configs / "finetune_droid_overfit.yaml").read_text())
    s = TrainSettings(**cfg["train"])
    o = ObjectiveSettings(**cfg["objective"])
    assert cfg["data"]["overfit_windows"] == 10 and o.fixed_draws is True
    assert (o.text_dropout, o.proprio_dropout, s.weight_decay) == (0.0, 0.0, 0.0)
    assert s.save is False and s.log_every == 1 and 0 < s.warmup_steps < s.steps
    assert (s.grad_accum, s.lr, s.lr_schedule) == (1, 5e-5, "flat")
    # All other values are the values of the recipe.
    changed = {
        ("data", "overfit_windows"),
        ("objective", "fixed_draws"),
        ("objective", "text_dropout"),
        ("objective", "proprio_dropout"),
        ("train", "steps"),
        ("train", "grad_accum"),
        ("train", "lr"),
        ("train", "warmup_steps"),
        ("train", "lr_schedule"),
        ("train", "weight_decay"),
        ("train", "save"),
        ("output", "dir"),
    }
    assert cfg.keys() == recipe.keys()
    for section, values in recipe.items():
        assert cfg[section].keys() == values.keys(), section
        for key, value in values.items():
            if (section, key) not in changed:
                assert cfg[section][key] == value, (section, key)


def test_command_line_values_with_an_exponent_are_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    # The script imports the data module, which needs the "train" extra.
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    # The script sets a default for this variable when it loads.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "finetune_script", root / "scripts" / "finetune.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    overrides = [
        "train.lr=1e-5",
        "objective.moe_seq_balance_coef=1e-3",
        "objective.text_dropout=5E-2",
        "data.max_episodes=1e3",
        "data.source=cadene/droid_1.0.1",
    ]
    cfg = script.load_config(str(root / "configs" / "finetune_droid.yaml"), overrides)
    assert cfg["train"]["lr"] == 1e-5
    assert cfg["objective"]["moe_seq_balance_coef"] == 1e-3
    assert cfg["objective"]["text_dropout"] == 0.05
    assert cfg["data"]["max_episodes"] == 1000.0
    assert cfg["data"]["source"] == "cadene/droid_1.0.1"
    assert cfg["train"]["steps"] == 1200 and cfg["output"]["resume"] is None


def test_settings() -> None:
    s = TrainSettings(betas=[0.9, 0.99], warmup_steps=4, lr="1e-4")  # ty: ignore[invalid-argument-type]
    assert s.betas == (0.9, 0.99) and s.lr == 1e-4
    assert [s.lr_at(i) for i in (0, 3, 4, 100)] == [2.5e-5, 1e-4, 1e-4, 1e-4]
    # Cosine: 5 warmup steps, then lr falls from 1e-4 to 1e-5 over steps 5..105.
    c = TrainSettings(
        steps=106, warmup_steps=5, lr=1e-4, lr_schedule="cosine", min_lr=1e-5
    )
    assert c.lr_at(0) == pytest.approx(2e-5) and c.lr_at(5) == pytest.approx(1e-4)
    assert c.lr_at(55) == pytest.approx(5.5e-5) and c.lr_at(105) == pytest.approx(1e-5)
    assert c.lr_at(500) == pytest.approx(1e-5)
    lrs = [c.lr_at(i) for i in range(5, 106)]
    assert lrs == sorted(lrs, reverse=True)
    with pytest.raises(ValueError):
        TrainSettings(lr_schedule="linear")
    with pytest.raises(ValueError):
        TrainSettings(grad_accum=0)
    with pytest.raises(ValueError):
        TrainSettings(save="no")  # ty: ignore[invalid-argument-type]
    assert derive_seed(1, 2) == derive_seed(1, 2) != derive_seed(2, 1)


# ---------------------------------------------------------------------------
# Distributed tests (gloo, 1 or 2 processes)
# ---------------------------------------------------------------------------


def _settings(**kw: object) -> TrainSettings:
    base = {"lr": 1e-3, "weight_decay": 1e-2, "seed": 7, "save_every": 10**6}
    return TrainSettings(**{**base, **kw})  # ty: ignore[invalid-argument-type]


def _batches(
    trainer: Trainer, windows: list[EncodedWindow]
) -> Callable[[], EncodedWindow]:
    """Give window ``(step * accum + micro) * world + rank`` of the list."""
    count = [0]

    def fn() -> EncodedWindow:
        n = trainer.settings.grad_accum
        micro = count[0] % n
        count[0] += 1
        g = (trainer.step * n + micro) * trainer.world + trainer.rank
        return windows[g % len(windows)]

    return fn


def _full_state(trainer: Trainer) -> dict[str, Tensor]:
    """The full state of the policy on every rank, on the CPU."""
    state = get_model_state_dict(
        trainer.model, options=StateDictOptions(full_state_dict=True)
    )
    cut = len(trainer.policy_prefix)
    return {k[cut:]: v.cpu() for k, v in state.items() if isinstance(v, Tensor)}


def _worker(rank: int, world: int, init: str, job: dict) -> None:
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        out = Path(job["out"])
        lora = job.get("lora")
        if job.get("resume") and not lora:
            policy = PraxisPolicy.from_pretrained(Path(job["resume"]) / "model", "cpu")
        else:
            policy = tiny_policy(dtype=job.get("dtype", torch.bfloat16))
        if lora:
            apply_lora(policy, LoRASettings(**lora))
            if job.get("resume"):
                saved = load_file(str(Path(job["resume"]) / TRAINABLE_NAME))
                assert not policy.load_state_dict(saved, strict=False).unexpected_keys
        model = StandIn(policy)
        settings = _settings(**job.get("settings", {}))
        trainer = Trainer(model, model.blocks(), settings, torch.device("cpu"))
        if job.get("resume"):
            trainer.load(job["resume"])
        batches = _batches(trainer, make_windows(8))
        result: dict = {"loss": [], "grad_norm": []}
        while trainer.step < job["steps"]:
            metrics = trainer.train_step(batches)
            result["loss"].append(metrics["loss"])
            result["grad_norm"].append(metrics["grad_norm"])
            if trainer.step == job.get("save_at"):
                trainer.save(out)
                result["saved_state"] = _full_state(trainer)
        result["state"] = _full_state(trainer)
        if rank == 0:
            torch.save(result, out / "result.pt")
    finally:
        dist.destroy_process_group()


def _run(world: int, out: Path, **job: object) -> dict:
    """Run ``_worker`` on ``world`` processes and return the result of rank 0."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        args = (world, f"file://{d}/pg", {"out": str(out), **job})
        mp.spawn(_worker, args=args, nprocs=world, join=True)
    return torch.load(out / "result.pt", weights_only=False)


def _plain_run(steps: int, per_step: int, dtype: torch.dtype) -> dict:
    """Training without FSDP: ``per_step`` windows in one backward per step.

    Window ``j`` of a step gets the generator seed of the trainer window
    ``micro * world + rank = j``.
    """
    windows = make_windows(8)
    model = StandIn(tiny_policy(dtype=dtype))
    model.requires_grad_(True)
    s = _settings()
    opt = AdamW(
        model.parameters(),
        lr=s.lr,
        betas=s.betas,
        eps=s.eps,
        weight_decay=s.weight_decay,
        seed=s.seed,
    )
    result: dict = {"loss": [], "grad_norm": []}
    for step in range(steps):
        total = 0.0
        for j in range(per_step):
            gen = torch.Generator().manual_seed(derive_seed(s.seed, step, j))
            loss, _ = model(windows[(step * per_step + j) % len(windows)], gen)
            (loss / per_step).backward()
            total += float(loss) / per_step
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        result["loss"].append(total)
        result["grad_norm"].append(float(torch.nn.utils.get_total_norm(grads)))
        opt.step()
        opt.zero_grad(set_to_none=True)
    result["state"] = {k: v.detach() for k, v in model.policy.state_dict().items()}
    return result


def _assert_exact(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        assert torch.equal(_bytes(a[k]), _bytes(b[k])), k


def _assert_close_run(got: dict, ref: dict, start: dict, rtol: float) -> None:
    """Compare the losses, the gradient norms and the weights of two runs.

    The weight check compares the change of the weights from ``start``: the
    difference of the two runs must be below ``rtol`` of the change.
    """
    np.testing.assert_allclose(got["loss"], ref["loss"], rtol=rtol)
    np.testing.assert_allclose(got["grad_norm"], ref["grad_norm"], rtol=rtol)
    a, b = got["state"], ref["state"]
    assert a.keys() == b.keys() == start.keys()
    diff = sum(float((a[k].double() - b[k].double()).norm() ** 2) for k in a)
    move = sum(float((b[k].double() - start[k].double()).norm() ** 2) for k in b)
    assert move > 0.0
    assert diff**0.5 <= rtol * move**0.5, (diff**0.5, move**0.5)
    for k in a:
        assert a[k].dtype == b[k].dtype, k


def test_fsdp_matches_plain_training_float32(tmp_path: Path) -> None:
    """Float32 weights: sharding, rank averaging and gradient accumulation give
    the result of one plain backward over all windows of a step, within the
    rounding of float32 sums."""
    start = tiny_policy(dtype=torch.float32).state_dict()
    f32 = torch.float32
    plain2, plain4 = _plain_run(STEPS, 2, f32), _plain_run(STEPS, 4, f32)
    accum = _run(1, tmp_path / "a", steps=STEPS, dtype=f32, settings={"grad_accum": 2})
    ranks = _run(2, tmp_path / "r", steps=STEPS, dtype=f32)
    both = _run(2, tmp_path / "b", steps=STEPS, dtype=f32, settings={"grad_accum": 2})
    for got, ref in ((accum, plain2), (ranks, plain2), (both, plain4)):
        _assert_close_run(got, ref, start, rtol=1e-4)


def test_fsdp_matches_plain_training_bfloat16(tmp_path: Path) -> None:
    """Bfloat16 weights: FSDP adds the gradient parts in another order than
    plain autograd, and bfloat16 sums round. The results thus differ in the
    last bits. The differences stay small against the change of the weights.
    The stochastic rounding uses the same noise for any sharding."""
    start = tiny_policy().state_dict()
    plain = _plain_run(STEPS, 2, torch.bfloat16)
    ranks = _run(2, tmp_path / "r", steps=STEPS)
    nosync = _run(
        1, tmp_path / "n", steps=STEPS, settings={"grad_accum": 2, "no_sync": True}
    )
    for got in (ranks, nosync):
        _assert_close_run(got, plain, start, rtol=0.05)
        same = sum(
            int((_bytes(got["state"][k]) == _bytes(v)).sum())
            for k, v in plain["state"].items()
        )
        size = sum(_bytes(v).numel() for v in plain["state"].values())
        assert same / size > 0.95


def test_save_load_and_resume(tmp_path: Path) -> None:
    """A save loads strictly with ``from_pretrained`` and holds the weights of
    its step. A resumed run continues bit for bit."""
    straight = _run(2, tmp_path / "a", steps=4, save_at=2)
    save = tmp_path / "a" / "step-000002"
    verify_checksums(save / "model")
    loaded = PraxisPolicy.from_pretrained(save / "model", device="cpu")
    _assert_exact(loaded.state_dict(), straight["saved_state"])
    assert load_config(save / "model") == tiny_config()
    assert json.loads((save / "trainer_state.json").read_text())["step"] == 2
    resumed = _run(2, tmp_path / "b", steps=4, resume=str(save))
    _assert_exact(resumed["state"], straight["state"])
    np.testing.assert_array_equal(resumed["loss"], straight["loss"][2:])


LORA = {"rank": 2, "alpha": 4.0, "targets": ["expert"]}
LORA_SCALE = 4.0 / 2


def test_lora_training_saves_merged_weights_and_resumes(tmp_path: Path) -> None:
    """A LoRA run trains only the adapters. Its save is a praxis checkpoint
    with the adapters folded in, and a resume from the start weights plus
    trainable.safetensors continues bit for bit."""
    start = tiny_policy(dtype=torch.bfloat16).state_dict()
    straight = _run(2, tmp_path / "a", steps=4, save_at=2, lora=LORA)
    save = tmp_path / "a" / "step-000002"
    verify_checksums(save / "model")
    loaded = PraxisPolicy.from_pretrained(save / "model", device="cpu")
    assert set(loaded.state_dict()) == set(start)
    scales = {
        k[: -len(".base.weight")]: LORA_SCALE
        for k in straight["saved_state"]
        if k.endswith(".base.weight")
    }
    want = merged_state_dict(straight["saved_state"], scales)
    _assert_exact(loaded.state_dict(), want)
    # The frozen weights keep their bytes; the target layers change.
    targets = {f"{m}.weight" for m in scales}
    changed = {
        k
        for k, v in loaded.state_dict().items()
        if not torch.equal(_bytes(v), _bytes(start[k]))
    }
    assert changed and changed <= targets
    trainable = load_file(str(save / TRAINABLE_NAME))
    assert trainable and all(".lora_" in k for k in trainable)
    resumed = _run(2, tmp_path / "b", steps=4, resume=str(save), lora=LORA)
    _assert_exact(resumed["state"], straight["state"])
    np.testing.assert_array_equal(resumed["loss"], straight["loss"][2:])


def _sr_worker(rank: int, world: int, init: str, out: str) -> None:
    """Three AdamW steps on sharded bfloat16 weights with fixed gradients."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, distribute_tensor

    import praxis.train.trainer as trainer_module

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        # Small noise blocks, so that blocks cross the shard boundaries.
        setattr(trainer_module, "SR_BLOCK", 16)  # noqa: B010
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(13, 5), nn.Linear(5, 7)).to(torch.bfloat16)
        fully_shard(model, mesh=init_device_mesh("cpu", (world,)))
        params = [p for p in model.parameters() if isinstance(p, DTensor)]
        opt = AdamW(params, lr=1e-2, weight_decay=0.1, seed=5)
        g = torch.Generator().manual_seed(1)
        for _ in range(3):
            for p in params:
                full = torch.randn(p.shape, generator=g).to(torch.bfloat16)
                p.grad = distribute_tensor(full, p.device_mesh, p.placements)
            opt.step()
        state = {str(i): p.full_tensor() for i, p in enumerate(params)}
        if rank == 0:
            torch.save(state, Path(out) / f"sr{world}.pt")
    finally:
        dist.destroy_process_group()


def test_stochastic_rounding_ignores_sharding(tmp_path: Path) -> None:
    """The same gradients give the same bfloat16 weights on 1 and 2 ranks."""
    for world in (1, 2):
        with tempfile.TemporaryDirectory() as d:
            args = (world, f"file://{d}/pg", str(tmp_path))
            mp.spawn(_sr_worker, args=args, nprocs=world, join=True)
    one = torch.load(tmp_path / "sr1.pt")
    two = torch.load(tmp_path / "sr2.pt")
    torch.manual_seed(0)
    start = nn.Sequential(nn.Linear(13, 5), nn.Linear(5, 7)).to(torch.bfloat16)
    assert len(one) == 4
    _assert_exact(one, two)
    first = next(start.parameters()).detach()
    assert not torch.equal(one["0"], first)


# ---------------------------------------------------------------------------
# Fixed windows and fixed draws (the overfit check)
# ---------------------------------------------------------------------------

# Recorded with the code before the overfit options: the noise seeds of the
# micro-steps of two steps with the seed 7. Index [rank][i] with
# i = step * grad_accum + micro.
RECORDED_SEEDS = {
    (1, 1): [[8949336737714337537, 1587299620702512783]],
    (2, 2): [
        [
            8949336737714337537,
            16472488366825535277,
            1587299620702512783,
            7450577197001350026,
        ],
        [
            8713865377778497563,
            14395007877634632326,
            2502279137656177121,
            8233261673722575741,
        ],
    ],
}


class Recorder(StandIn):
    """``StandIn`` that records the window id and the noise seed of each call."""

    def __init__(self, policy: PraxisPolicy) -> None:
        super().__init__(policy)
        self.seen: list[tuple[int | None, int]] = []

    def forward(
        self, batch: EncodedWindow, generator: torch.Generator
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self.seen.append((batch.window_id, int(generator.initial_seed())))
        return super().forward(batch, generator)


def _numbered(n: int) -> list[EncodedWindow]:
    return [dataclasses.replace(w, window_id=i) for i, w in enumerate(make_windows(n))]


def _cycle_worker(rank: int, world: int, init: str, job: dict) -> None:
    """Train ``job["steps"]`` steps with ``fit`` on fixed windows."""
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        model = Recorder(tiny_policy())
        settings = _settings(**job["settings"])
        cpu = torch.device("cpu")
        trainer = Trainer(
            model, model.blocks(), settings, cpu, fixed_draws=job["fixed_draws"]
        )
        windows = broadcast_windows(_numbered(job["n"]) if rank == 0 else None, cpu)
        # Each rank gets the same bits as rank 0.
        for got, want in zip(windows, _numbered(job["n"]), strict=True):
            assert got.window_id == want.window_id
            for a, b in zip(
                [got.text, got.anchor, *got.chunks, got.actions, got.proprio],
                [want.text, want.anchor, *want.chunks, want.actions, want.proprio],
                strict=True,
            ):
                assert a.dtype == b.dtype and torch.equal(_bytes(a), _bytes(b))
        out = Path(job["out"])
        trainer.fit(cycle_windows(trainer, windows), out)
        result: dict = {"seen": model.seen}
        if job["fixed_draws"]:
            plain = dataclasses.replace(windows[0], window_id=None)
            try:
                trainer.train_step(lambda: plain)
            except ValueError:
                result["no_id_error"] = True
        torch.save(result, out / f"result{rank}.pt")
    finally:
        dist.destroy_process_group()


def _cycle_run(world: int, out: Path, **job: object) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        args = (world, f"file://{d}/pg", {"out": str(out), **job})
        mp.spawn(_cycle_worker, args=args, nprocs=world, join=True)
    return [torch.load(out / f"result{r}.pt") for r in range(world)]


def _window_order(world: int, accum: int, steps: int, n: int) -> list[list[int]]:
    """The window of rank r at micro-step k of step s: ((s * A + k) * W + r) % n."""
    return [
        [((s * accum + k) * world + r) % n for s in range(steps) for k in range(accum)]
        for r in range(world)
    ]


def test_default_seeds_and_final_save_do_not_change(tmp_path: Path) -> None:
    """Without fixed draws, the noise seeds are the seeds of the code before
    the overfit options, also with fixed windows. The run saves after the
    last step."""
    for world, accum in RECORDED_SEEDS:
        out = tmp_path / f"w{world}a{accum}"
        settings = {"grad_accum": accum, "steps": 2}
        got = _cycle_run(world, out, settings=settings, fixed_draws=False, n=10)
        seeds = [[seed for _, seed in r["seen"]] for r in got]
        assert seeds == RECORDED_SEEDS[(world, accum)]
        ids = [[i for i, _ in r["seen"]] for r in got]
        assert ids == _window_order(world, accum, 2, 10)
        assert sorted(p.name for p in out.glob("step-*")) == ["step-000002"]
    assert derive_seed(7, 1, 3) == RECORDED_SEEDS[(2, 2)][1][3]


def test_fixed_windows_and_fixed_draws(tmp_path: Path) -> None:
    """Fixed windows come in turn on 1 and 2 ranks and with gradient
    accumulation. With fixed draws, window i gets the same noise seed in each
    step, on each rank and for each number of ranks. ``train.save`` false
    writes no save, also with ``save_every`` 1."""
    n, seeds = 10, {}
    for world, accum, steps in ((1, 1, 11), (2, 1, 6), (2, 2, 3)):
        out = tmp_path / f"w{world}a{accum}"
        settings = {"grad_accum": accum, "steps": steps, "save": False}
        settings["save_every"] = 1
        got = _cycle_run(world, out, settings=settings, fixed_draws=True, n=n)
        ids = [[i for i, _ in r["seen"][: steps * accum]] for r in got]
        assert ids == _window_order(world, accum, steps, n)
        for r in got:
            assert r["no_id_error"] is True
            for i, seed in r["seen"][: steps * accum]:
                assert seeds.setdefault(i, seed) == seed
        # No save and no partial save: only the log and the test results.
        results = [f"result{r}.pt" for r in range(world)]
        assert sorted(p.name for p in out.iterdir()) == ["metrics.jsonl", *results]
        assert len((out / "metrics.jsonl").read_text().splitlines()) == steps
    assert sorted(seeds) == list(range(n))
    assert all(seeds[i] == derive_seed(7, FIXED_DRAWS_KEY, i) for i in range(n))
    assert len(set(seeds.values())) == n
    defaults = {s for r in RECORDED_SEEDS.values() for row in r for s in row}
    assert not defaults & set(seeds.values())


def test_fixed_draws_repeat_for_a_window_and_differ_between_windows() -> None:
    uncond = torch.randn(1, 1, 32).to(torch.bfloat16)
    model = FinetuneModel(tiny_policy(), ObjectiveSettings(fixed_draws=True), uncond)
    batch = make_windows(1)[0]

    def draws(window_id: int) -> list[Tensor]:
        gen = torch.Generator().manual_seed(derive_seed(7, FIXED_DRAWS_KEY, window_id))
        d = model.draw(batch, gen)
        shifts = torch.tensor(d.rope_shifts, dtype=torch.float32)
        return [d.video_u, d.action_u, d.proprio_u, d.text_u, shifts]

    first, again, other = draws(3), draws(3), draws(4)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not any(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_cycle_windows_needs_numbered_windows() -> None:
    with pytest.raises(ValueError):
        cycle_windows(None, [])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        cycle_windows(None, make_windows(2))  # ty: ignore[invalid-argument-type]


def test_save_checkpoint_format(tmp_path: Path) -> None:
    from praxis.checkpoint import load_state_dict_from_dir, save_checkpoint

    tensors = {
        "b.blocks.10.w": torch.randn(64, 32).to(torch.bfloat16),
        "b.blocks.2.w": torch.randn(64, 32).to(torch.bfloat16),
        "b.blocks.2.ffn.expert_bias": torch.randn(8),
        "a.s": torch.tensor(0.5, dtype=torch.bfloat16),
    }
    save_checkpoint(tmp_path / "ckpt", tensors, {"x": 1}, max_shard_bytes=5000)
    root = tmp_path / "ckpt"
    verify_checksums(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    # Natural order: a.s, b.blocks.2.ffn.expert_bias, b.blocks.2.w (4130 bytes),
    # then b.blocks.10.w in the second shard.
    assert names == [f"model-{i:05d}-of-00002.safetensors" for i in (1, 2)]
    assert index["weight_map"]["b.blocks.2.w"] < index["weight_map"]["b.blocks.10.w"]
    assert index["metadata"]["total_size"] == 2 * 64 * 32 * 2 + 8 * 4 + 2
    assert json.loads((root / "config.json").read_text()) == {"x": 1}
    loaded = load_state_dict_from_dir(root)
    for name, t in tensors.items():
        assert loaded[name].dtype == t.dtype and torch.equal(loaded[name], t)
    with pytest.raises(CheckpointError, match="exists"):
        save_checkpoint(root, tensors, {})
