"""The chunk-causal layout of the backbone (CPU only)."""

from dataclasses import dataclass

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

from praxis.backbone import (
    BackboneOutput,
    VideoBackbone,
    _full_blocks,
    _TeacherForcingMaskCache,
    build_backbone,
    teacher_forcing_mask,
)

CPU = torch.device("cpu")


def _token_classes(
    slots: list[tuple[str, int, int]], chunk_size: int
) -> tuple[list[str], list[int]]:
    kinds: list[str] = []
    chunks: list[int] = []
    for kind, frames, tokens_per_frame in slots:
        for f in range(frames):
            kinds += [kind] * tokens_per_frame
            chunks += [f // chunk_size] * tokens_per_frame
    return kinds, chunks


def _brute_force(
    slots: list[tuple[str, int, int]], chunk_size: int, clean_past: bool
) -> torch.Tensor:
    kinds, chunks = _token_classes(slots, chunk_size)
    n = len(kinds)
    out = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            q, k = kinds[i], kinds[j]
            if k == "global":
                allow = True
            elif q == "global":
                allow = False
            elif q == "clean" and k == "clean":
                allow = chunks[j] <= chunks[i]
            elif q == "noised" and k == "clean":
                allow = clean_past and chunks[j] < chunks[i]
            elif q == "noised" and k == "noised":
                allow = chunks[j] == chunks[i]
            else:  # a clean token never reads a noised token
                allow = False
            out[i, j] = allow
    return out


def _dense(mask: BlockMask, n: int) -> torch.Tensor:
    q = torch.arange(n)[:, None].expand(n, n)
    k = torch.arange(n)[None, :].expand(n, n)
    zero = torch.zeros((), dtype=torch.int64)
    return mask.mask_mod(zero, zero, q, k)


def _tiles(counts: torch.Tensor | None, idx: torch.Tensor | None) -> set:
    if counts is None or idx is None:
        return set()
    out = set()
    c, ii = counts[0, 0], idx[0, 0]
    for qb in range(c.numel()):
        for j in range(int(c[qb])):
            out.add((qb, int(ii[qb, j])))
    return out


LAYOUTS = [
    # encoder windows: registers, anchor, text, K' noised chunks, K'-1 clean
    ([("global", 1, 4), ("global", 1, 6), ("global", 1, 5), ("noised", 3, 6)], True),
    (
        [
            ("global", 1, 4),
            ("global", 1, 6),
            ("global", 1, 5),
            ("noised", 6, 6),
            ("clean", 3, 6),
        ],
        True,
    ),
    (
        [
            ("global", 1, 4),
            ("global", 1, 6),
            ("global", 1, 3),
            ("noised", 9, 5),
            ("clean", 6, 5),
        ],
        True,
    ),
    # decoder window: registers, K' noised chunks
    ([("global", 1, 4), ("noised", 9, 7)], False),
]


@pytest.mark.parametrize(("slots", "clean_past"), LAYOUTS)
@pytest.mark.parametrize("full_blocks", [True, False])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("use_interval_mask", [False, True])
def test_teacher_forcing_mask_rules(
    monkeypatch: pytest.MonkeyPatch,
    slots: list[tuple[str, int, int]],
    clean_past: bool,
    full_blocks: bool,
    block_size: int,
    use_interval_mask: bool,
) -> None:
    monkeypatch.setattr("praxis.backbone.TF_PACKED_MASK", False)
    chunk_size = 3
    mask = teacher_forcing_mask(
        slots,
        chunk_size,
        clean_past,
        block_size,
        CPU,
        full_blocks,
        use_interval_mask=use_interval_mask,
    )
    want = _brute_force(slots, chunk_size, clean_past)
    n = want.size(0)
    got = _dense(mask, n)
    assert torch.equal(got, want)

    partial = _tiles(mask.kv_num_blocks, mask.kv_indices)
    full = _tiles(mask.full_kv_num_blocks, mask.full_kv_indices)
    assert not (partial & full)
    if not full_blocks:
        assert mask.full_kv_num_blocks is None
    # Every attending pair lies in a listed tile.
    nb = (n + block_size - 1) // block_size
    for qb in range(nb):
        for kb in range(nb):
            tile = want[
                qb * block_size : (qb + 1) * block_size,
                kb * block_size : (kb + 1) * block_size,
            ]
            if tile.any():
                assert (qb, kb) in partial | full
            if (qb, kb) in full:
                # A full tile is a whole tile where every pair attends.
                assert tile.shape == (block_size, block_size)
                assert tile.all()
    # The q lists are the transpose of the kv lists.
    assert _tiles(mask.q_num_blocks, mask.q_indices) == {(k, q) for q, k in partial}
    assert _tiles(mask.full_q_num_blocks, mask.full_q_indices) == {
        (k, q) for q, k in full
    }


@pytest.mark.parametrize("clean_past", [True, False])
@pytest.mark.parametrize(
    "slots",
    [
        [("global", 1, 3), ("noised", 8, 2), ("clean", 5, 2)],
        [("noised", 5, 3), ("global", 1, 2), ("clean", 7, 1)],
        [("clean", 4, 2), ("noised", 4, 3), ("global", 1, 2)],
        [("global", 1, 2), ("noised", 4, 2), ("noised", 3, 1)],
        [("clean", 7, 2)],
        [("global", 1, 5)],
    ],
)
def test_interval_mask_matches_metadata_with_padding_and_query_offsets(
    monkeypatch, slots, clean_past
):
    monkeypatch.setattr("praxis.backbone.TF_PACKED_MASK", False)
    n = sum(frames * width for _, frames, width in slots)
    masks = [
        teacher_forcing_mask(
            slots, 3, clean_past, 16, CPU, True, use_interval_mask=interval
        )
        for interval in (False, True)
    ]
    q = torch.arange(n)[:, None]
    k = torch.arange(n)[None, :]
    assert torch.equal(masks[0].mask_mod(0, 0, q, k), masks[1].mask_mod(0, 0, q, k))
    # Cached queries retain their absolute offset into the original sequence.
    assert torch.equal(
        masks[0].mask_mod(0, 0, q[2:-1], k),
        masks[1].mask_mod(0, 0, q[2:-1], k),
    )
    outside = torch.tensor([-1, n, n + 15])[:, None]
    assert not masks[1].mask_mod(0, 0, outside, k).any()
    assert not masks[1].mask_mod(0, 0, q, outside.T).any()
    for attr in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
    ):
        assert torch.equal(getattr(masks[0], attr), getattr(masks[1], attr))


@pytest.mark.parametrize(("slots", "clean_past"), LAYOUTS)
@pytest.mark.parametrize("full_blocks", [True, False])
@pytest.mark.parametrize("block_size", [16, 32])
def test_teacher_forcing_mask_packed_flags(
    monkeypatch: pytest.MonkeyPatch,
    slots: list[tuple[str, int, int]],
    clean_past: bool,
    full_blocks: bool,
    block_size: int,
) -> None:
    import praxis.backbone as PB

    monkeypatch.setattr(PB, "TF_PACKED_MASK", True)
    chunk_size = 3
    mask = teacher_forcing_mask(
        slots, chunk_size, clean_past, block_size, CPU, full_blocks
    )
    want = _brute_force(slots, chunk_size, clean_past)
    n = want.size(0)
    assert torch.equal(_dense(mask, n), want)
    partial = _tiles(mask.kv_num_blocks, mask.kv_indices)
    full = _tiles(mask.full_kv_num_blocks, mask.full_kv_indices)
    assert not (partial & full)
    monkeypatch.setattr(PB, "TF_PACKED_MASK", False)
    stock = teacher_forcing_mask(
        slots, chunk_size, clean_past, block_size, CPU, full_blocks
    )
    stock_partial = _tiles(stock.kv_num_blocks, stock.kv_indices)
    stock_full = _tiles(stock.full_kv_num_blocks, stock.full_kv_indices)
    assert partial == stock_partial and full == stock_full


@pytest.mark.parametrize("interval", [None, False, True])
def test_explicit_mask_choice_overrides_packed_default(monkeypatch, interval):
    from praxis import backbone as PB

    monkeypatch.setattr(PB, "TF_PACKED_MASK", True)
    built = []
    for name, label in [
        ("_tf_interval_mask", "interval"),
        ("_tf_packed_mask_mod", "packed"),
    ]:
        original = getattr(PB, name)

        def observe(*args, _original=original, _label=label):
            built.append(_label)
            return _original(*args)

        monkeypatch.setattr(PB, name, observe)
    slots = [("global", 1, 4), ("noised", 6, 4), ("clean", 3, 4)]
    mask = teacher_forcing_mask(
        slots, 3, True, 16, CPU, True, use_interval_mask=interval
    )
    assert built == (
        ["packed"] if interval is None else ["interval"] if interval else []
    )
    assert torch.equal(_dense(mask, 40), _brute_force(slots, 3, True))
    assert PB.TF_PACKED_MASK is True


@pytest.mark.parametrize(
    ("thor", "flag", "interval", "expected"),
    [
        (False, True, None, ["packed"]),
        (True, False, None, ["packed"]),
        (True, True, None, ["interval"]),
        (True, True, False, []),
    ],
)
def test_sm110_interval_mask_is_thor_only(monkeypatch, thor, flag, interval, expected):
    from praxis import backbone as PB

    monkeypatch.setattr(PB, "TF_PACKED_MASK", True)
    monkeypatch.setattr(PB, "SM110_INTERVAL_MASK", flag)
    monkeypatch.setattr(PB, "is_thor", lambda _device: thor)
    built = []
    for name, label in [
        ("_tf_interval_mask", "interval"),
        ("_tf_packed_mask_mod", "packed"),
    ]:
        original = getattr(PB, name)

        def observe(*args, _original=original, _label=label):
            built.append(_label)
            return _original(*args)

        monkeypatch.setattr(PB, name, observe)
    slots = [("global", 1, 4), ("noised", 6, 4), ("clean", 3, 4)]
    mask = teacher_forcing_mask(
        slots, 3, True, 16, CPU, True, use_interval_mask=interval
    )
    assert built == expected
    assert torch.equal(_dense(mask, 40), _brute_force(slots, 3, True))


def test_clean_past_false_hides_the_clean_tokens_from_the_noised_tokens() -> None:
    slots = [("global", 1, 4), ("noised", 6, 4), ("clean", 3, 4)]
    with_past = _dense(teacher_forcing_mask(slots, 3, True, 16, CPU, True), 40)
    without = _dense(teacher_forcing_mask(slots, 3, False, 16, CPU, True), 40)
    noised = slice(4, 28)
    clean = slice(28, 40)
    assert with_past[noised, clean].any()
    assert not without[noised, clean].any()
    # The clean tokens keep their chunk order in both cases.
    assert torch.equal(with_past[clean, clean], without[clean, clean])


def test_teacher_forcing_mask_cache() -> None:
    cache = _TeacherForcingMaskCache(block_size=16, max_entries=2)
    slots = [("global", 1, 4), ("noised", 3, 4)]
    a = cache.get(slots, 3, True, CPU, True)
    assert cache.get(slots, 3, True, CPU, True) is a
    assert cache.get(slots, 3, False, CPU, True) is not a
    cache.get(slots, 3, True, CPU, False)
    assert cache.get(slots, 3, True, CPU, True) is not a  # evicted


def test_unknown_slot_kind() -> None:
    with pytest.raises(ValueError, match="unknown slot kind"):
        teacher_forcing_mask([("audio", 1, 4)], 3, True, 16, CPU, True)


# ---------------------------------------------------------------------------
# The chunk-causal towers
# ---------------------------------------------------------------------------

BACKBONE_CHUNK_CAUSAL = {
    "layout": "chunk_causal",
    "dim": 64,
    "timestep_dim": 16,
    "num_registers": 4,
    "modulation_dtype": "mixed",
    "attention": {"num_kv_heads": 2, "head_dim": 16, "flex_block_size": 128},
    "rope": {"effective_dim": 8, "base": 26000000, "efficient_cache_max": 1000000},
    "moe": {
        "num_routed_experts": 4,
        "top_k_routed": 2,
        "expert_hidden": 16,
        "routed_scaling_factor": 2.5,
        "dispatch_chunk_size": 7,
    },
    "dense_ffn_hidden": 128,
    "encoder": {
        "depth": 3,
        "dense_blocks": [0],
        "tap_blocks": [0, 2],
        "input_projections": {
            "TextEncodingTensor": {"c_in": 8},
            "VideoTensor": {"c_in": 12},
        },
        "positive_zero_residual_add": True,
    },
    "decoder": {
        "depth": 2,
        "dense_blocks": [0],
        "modulation_dim": 64,
        "unpatch_factor": [1, 2, 2],
        "input_projections": {"VideoTensor": {"c_in": 3}},
        "positive_zero_residual_add": True,
    },
}
H, W, F = 2, 3, 6


@pytest.fixture(autouse=True)
def _eager_cpu_flex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run flex attention eagerly on the CPU.

    The compiled CPU kernel of torch 2.13 fails to build for some of these
    masks. The eager kernel computes the same attention.
    """
    from torch.nn.attention.flex_attention import flex_attention

    import praxis.layers

    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)


def _chunk_causal_backbone() -> VideoBackbone:
    backbone = build_backbone(BACKBONE_CHUNK_CAUSAL).eval()
    torch.manual_seed(0)
    for p in backbone.parameters():
        torch.nn.init.normal_(p, std=0.05)
    for name, buf in backbone.named_buffers():
        if name.endswith(("freqs", "phases")):
            buf.normal_()
    return backbone


@dataclass
class _Inputs:
    text: torch.Tensor
    anchor: torch.Tensor
    clean: torch.Tensor | None
    noised: torch.Tensor
    timesteps: torch.Tensor

    def run(self, backbone: VideoBackbone) -> BackboneOutput:
        with torch.no_grad():
            return backbone.forward_chunk_causal(
                self.text, self.anchor, self.clean, self.noised, self.timesteps
            )


def _chunk_causal_inputs(kk: int, seed: int = 1) -> _Inputs:
    g = torch.Generator().manual_seed(seed)
    return _Inputs(
        text=torch.randn(1, 5, 8, generator=g),
        anchor=torch.randn(1, 12, 1, H, W, generator=g),
        clean=torch.randn(1, 12, F * (kk - 1), H, W, generator=g) if kk > 1 else None,
        noised=torch.randn(1, 12, F * kk, H, W, generator=g),
        timesteps=torch.rand(1, kk, generator=g),
    )


def _reference_rope_ids(n_text: int, kk: int) -> list[int]:
    """The rope ids of the rules, written out token by token."""
    ids = [0] * 4
    ids += [1 + h * 360 + w for h in range(H) for w in range(W)]
    ids += [129_600 + 1 + i for i in range(n_text)]
    noised = [
        1 + (2 + f) * 129_600 + h * 360 + w
        for f in range(F * kk)
        for h in range(H)
        for w in range(W)
    ]
    return ids + noised + noised[: F * (kk - 1) * H * W]


@pytest.mark.parametrize("target", ["core", "block", "custom_forward"])
def test_paired_encoder_preserves_module_calls(target: str) -> None:
    encoder = _chunk_causal_backbone().encoder
    calls = [vars(_chunk_causal_inputs(1, seed)) for seed in (1, 2)]
    seen = []
    if target == "custom_forward":
        original = encoder.core.forward

        def forward(*args, **kwargs):
            seen.append(True)
            return original(*args, **kwargs)

        encoder.core.forward = forward
    else:
        module = encoder.core if target == "core" else encoder.core.blocks[0]
        module.register_forward_hook(lambda *args: seen.append(True))
    with torch.no_grad():
        expected = [encoder.forward_chunk_causal(**kw) for kw in calls]
        seen.clear()
        actual = encoder.forward_chunk_causal_pair(calls)
    assert len(seen) == 2
    for (cond, taps), (ref_cond, ref_taps) in zip(actual, expected, strict=True):
        torch.testing.assert_close(cond, ref_cond, rtol=0, atol=0)
        for tap, ref in zip(taps.taps, ref_taps.taps, strict=True):
            torch.testing.assert_close(tap, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kk", [1, 2])
def test_paired_decoder_matches_separate_passes(kk: int) -> None:
    backbone = _chunk_causal_backbone()
    calls = []
    with torch.no_grad():
        for seed in (1, 2):
            inp = _chunk_causal_inputs(kk, seed)
            cond, _ = backbone.encoder.forward_chunk_causal(
                inp.text, inp.anchor, inp.clean, inp.noised, inp.timesteps
            )
            calls.append(
                {"noised": inp.noised, "cond": cond, "timesteps": inp.timesteps}
            )
        expected = [backbone.decoder.forward_chunk_causal(**kw) for kw in calls]
        actual = backbone.decoder.forward_chunk_causal_pair(calls)
    for out, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_moe_tail_norm_keeps_cpu_outputs(monkeypatch) -> None:
    """MOE_TAIL_NORM only defers CUDA tails; elsewhere the ordinary path runs."""
    import praxis.layers

    backbone = _chunk_causal_backbone()
    inp = _chunk_causal_inputs(2)
    ref = inp.run(backbone)
    monkeypatch.setattr(praxis.layers, "MOE_TAIL_NORM", True)
    out = inp.run(backbone)
    assert ref.velocity is not None and out.velocity is not None
    torch.testing.assert_close(out.velocity, ref.velocity, rtol=0, atol=0)
    assert not praxis.layers._TAIL_INPUTS


@pytest.mark.parametrize("kk", [1, 2, 3])
def test_chunk_causal_layout(kk: int) -> None:
    backbone = _chunk_causal_backbone()
    inp = _chunk_causal_inputs(kk)
    out = inp.run(backbone)
    taps = out.taps
    tpf = H * W
    n_noised, n_clean = F * kk * tpf, F * (kk - 1) * tpf
    n_global = 4 + tpf + 5
    assert out.velocity is not None
    assert out.velocity.shape == inp.noised.shape
    assert taps.n_chunks == kk
    assert taps.frames_per_chunk == F and taps.tokens_per_frame == tpf
    assert taps.global_spans == {
        "register": (0, 4),
        "anchor": (4, tpf),
        "text": (4 + tpf, 5),
    }
    assert taps.noised_span == (n_global, n_noised)
    assert taps.clean_span == ((n_global + n_noised, n_clean) if kk > 1 else None)
    assert taps.rope_ids is not None
    assert taps.rope_ids.tolist() == [_reference_rope_ids(5, kk)]
    assert taps.embeddings.shape[1] == n_global + n_noised + n_clean
    assert all(t.shape == taps.embeddings.shape for t in taps.taps)


def test_chunk_causal_is_causal_over_chunks() -> None:
    """Chunk k (noised and clean) does not depend on the inputs of later chunks."""
    backbone = _chunk_causal_backbone()
    a = _chunk_causal_inputs(3, seed=1)
    assert a.clean is not None
    b = _Inputs(
        a.text, a.anchor, a.clean.clone(), a.noised.clone(), a.timesteps.clone()
    )
    assert b.clean is not None
    b.noised[:, :, 2 * F :] += 1.0  # the last noised chunk
    b.clean[:, :, F:] += 1.0  # the last clean chunk
    b.timesteps[:, 2] = 0.1
    oa, ob = a.run(backbone), b.run(backbone)
    assert oa.velocity is not None and ob.velocity is not None
    torch.testing.assert_close(oa.velocity[:, :, : 2 * F], ob.velocity[:, :, : 2 * F])
    assert not torch.allclose(oa.velocity[:, :, 2 * F :], ob.velocity[:, :, 2 * F :])
    start, _ = oa.taps.noised_span or (0, 0)
    first_two = slice(start, start + 2 * F * H * W)
    for ta, tb in zip(oa.taps.taps, ob.taps.taps, strict=True):
        torch.testing.assert_close(ta[:, first_two], tb[:, first_two])


def _with_backend(backend: str) -> dict:
    attention = {**BACKBONE_CHUNK_CAUSAL["attention"], "backend": backend}
    return {**BACKBONE_CHUNK_CAUSAL, "attention": attention}


@pytest.mark.parametrize(
    ("backend", "device", "major", "expected"),
    [
        ("flash", "cuda:0", 12, False),  # SM120 runs Triton: no full blocks
        ("flash", "cuda:0", 9, True),
        ("flash", "cuda:0", 10, True),
        ("flash", "cuda:0", 11, True),  # Thor: Flash on the 256-row mask blocks
        ("flash", "cpu", 12, True),
        ("triton", "cuda:0", 9, False),
        ("triton", "cuda:0", 12, False),
    ],
)
def test_full_blocks_follow_the_backend_that_runs(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    device: str,
    major: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (major, 0))
    backbone = build_backbone(_with_backend(backend))
    for tower in (backbone.encoder, backbone.decoder):
        block_size = tower._tf_masks.block_size
        assert _full_blocks(tower.core, torch.device(device), block_size) is expected


def test_flash_fallback_gets_the_triton_mask_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested flash backend that falls back to Triton (SM120) builds the
    Triton teacher-forcing mask and gives the bits of the Triton backbone."""
    import praxis.backbone
    from praxis.layers import Attention

    triton = _chunk_causal_backbone()
    flash = build_backbone(_with_backend("flash")).eval()
    flash.load_state_dict(triton.state_dict())
    original = Attention.effective_backend
    monkeypatch.setattr(
        Attention,
        "effective_backend",
        lambda self, device, block_mask=None, *, block_size=None: "triton"
        if self.backend == "flash"
        else original(self, device, block_mask, block_size=block_size),
    )
    built: list[tuple[str, BlockMask, bool]] = []
    build = praxis.backbone.teacher_forcing_mask

    def recording(slots, chunk_size, clean_past, block_size, device, full_blocks):
        mask = build(slots, chunk_size, clean_past, block_size, device, full_blocks)
        built.append((repr((slots, chunk_size, clean_past)), mask, full_blocks))
        return mask

    monkeypatch.setattr(praxis.backbone, "teacher_forcing_mask", recording)
    inp = _chunk_causal_inputs(2)
    reference = inp.run(triton)
    n_triton = len(built)
    actual = inp.run(flash)
    assert n_triton == 2 and len(built) == 4
    for (key_t, mask_t, full_t), (key_f, mask_f, full_f) in zip(
        built[:n_triton], built[n_triton:], strict=True
    ):
        assert key_t == key_f
        assert full_t is False and full_f is False
        assert mask_f.full_kv_num_blocks is None
        assert torch.equal(mask_f.kv_num_blocks, mask_t.kv_num_blocks)
        assert torch.equal(mask_f.kv_indices, mask_t.kv_indices)
    assert reference.velocity is not None and actual.velocity is not None
    assert torch.equal(actual.velocity, reference.velocity)
    for a, b in zip(actual.taps.taps, reference.taps.taps, strict=True):
        assert torch.equal(a, b)


def test_chunk_causal_chunked_moe_matches_unchunked() -> None:
    from praxis.layers import MoEFeedForward

    torch.manual_seed(0)
    moe = MoEFeedForward(16, 8, 4, 2, 2.5, dispatch_chunk_size=5)
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.2)
    x = torch.randn(23, 16)
    chunked = moe(x)
    moe.dispatch_chunk_size = None
    torch.testing.assert_close(chunked, moe(x), rtol=1e-5, atol=1e-6)


def test_chunk_causal_is_the_only_layout() -> None:
    config = {k: v for k, v in BACKBONE_CHUNK_CAUSAL.items() if k != "layout"}
    assert isinstance(build_backbone(config), VideoBackbone)
    for layout in ("serve", "other"):
        with pytest.raises(ValueError, match="layout"):
            build_backbone({**config, "layout": layout})


def test_decoder_output_projection_gives_a_contiguous_gradient() -> None:
    """The identity split of the decoder output projection hands its linear
    layer a contiguous output gradient (as the original implementation), also when the
    caller reorders the output (``movedim``)."""
    from praxis.backbone import DecoderOutputProjection

    torch.manual_seed(0)
    proj = DecoderOutputProjection(16, 6)
    seen: dict[str, bool] = {}

    def hook(
        _m: torch.nn.Module,
        _gin: tuple[torch.Tensor, ...] | torch.Tensor,
        gout: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        g = gout[0] if isinstance(gout, tuple) else gout
        seen["contiguous"] = bool(g.is_contiguous())

    proj.proj[1].register_full_backward_hook(hook)
    x = torch.randn(1, 10, 16, requires_grad=True)
    out = proj(x)
    assert torch.equal(out, proj.proj(x))
    out.movedim(-1, 1).sum(dim=-1).square().sum().backward()
    assert seen == {"contiguous": True}
