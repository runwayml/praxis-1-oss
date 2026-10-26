"""The trunk-rope layout of the action expert (CPU only, small random networks)."""

import pytest
import torch
from torch.nn.attention import flex_attention as flex_mod

import praxis.action_expert as ae
from praxis.action_expert import (
    ActionExpert,
    build_trunk_rope_masks,
    run_expert_on_taps_k,
    trunk_mask_mods,
    trunk_query_rope_ids,
)

MAX_INTER, MAX_INTRA = 129600, 360
TRUNK_ROPE = {
    "kind": "trunk",
    "dim": 16,
    "effective_dim": 16,
    "base": 26_000_000,
    "table_size": 1_000_000,
    "order": "before_norm",
}
SMALL: dict = {
    "action_patch_dim": 5,
    "proprio_patch_dim": 3,
    "trunk_dim": 24,
    "dim": 32,
    "num_layers": 4,
    "num_heads": 2,
    "head_dim": 16,
    "num_future_per_chunk": 2,
    "max_chunks": 4,
    "max_action_tokens": 8,
}


def _trunk_rope_expert(**overrides: object) -> ActionExpert:
    kwargs: dict = {
        **SMALL,
        "rope": TRUNK_ROPE,
        "global_order": ("register", "anchor", "text"),
        "attention": "sdpa",
        "head_plus_zero": True,
        "layout_stamp": 20260924,
        **overrides,
    }
    torch.manual_seed(0)
    expert = ActionExpert(**kwargs)
    for p in expert.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return expert.eval().requires_grad_(False)


def _rope_ids(
    n_reg: int, h: int, w: int, n_text: int, kp: int, fpc: int
) -> torch.Tensor:
    """Trunk rope ids of ``[regs | anchor | text | noised | clean]``."""
    hw = torch.tensor([i * MAX_INTRA + j for i in range(h) for j in range(w)])
    noised = [1 + (2 + f) * MAX_INTER + hw for f in range(fpc * kp)]
    ids = [
        torch.zeros(n_reg, dtype=torch.long),
        1 + hw,
        MAX_INTER + 1 + torch.arange(n_text),
        *noised,
        *noised[: fpc * (kp - 1)],
    ]
    return torch.cat(ids)[None]


def _window(kp: int, *, fpc: int = 2, h: int = 2, w: int = 3, seed: int = 0) -> dict:
    n_reg, n_text, tpf = 2, 3, h * w
    rope = _rope_ids(n_reg, h, w, n_text, kp, fpc)
    s = rope.shape[1]
    g = torch.Generator().manual_seed(seed)
    n0 = n_reg + tpf + n_text
    return {
        "taps": [torch.randn(1, s, 24, generator=g) for _ in range(2)],
        "embeddings": torch.randn(1, s, 24, generator=g),
        "rope_ids": rope,
        "global_spans": {
            "register": (0, n_reg),
            "anchor": (n_reg, tpf),
            "text": (n_reg + tpf, n_text),
        },
        "noised_span": (n0, fpc * kp * tpf),
        "clean_span": (n0 + fpc * kp * tpf, fpc * (kp - 1) * tpf) if kp > 1 else None,
        "n_chunks": kp,
        "frames_per_chunk": fpc,
        "tokens_per_frame": tpf,
        "noised_actions": torch.randn(1, fpc * kp, 5, generator=g),
        "action_t_per_chunk": torch.rand(1, kp, generator=g),
        "proprio_per_chunk": torch.randn(1, kp, 3, generator=g),
    }


# ---------------------------------------------------------------------------
# Rope ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kp", [1, 2, 3])
def test_query_rope_ids_are_band_starts_of_noised_frames(kp: int) -> None:
    fpc, h, w, n_g = 2, 2, 3, 2 + 6 + 3
    full = _rope_ids(2, h, w, 3, kp, fpc)
    n_noised, n_clean, tpf = fpc * kp, fpc * (kp - 1), h * w
    # KV order: globals, clean frames, noised frames.
    noised = full[:, n_g : n_g + n_noised * tpf]
    clean = full[:, n_g + n_noised * tpf :]
    kv = torch.cat([full[:, :n_g], clean, noised], dim=1)
    q = trunk_query_rope_ids(
        kv,
        num_global_tokens=n_g,
        num_clean_frames=n_clean,
        num_noised_frames=n_noised,
        tokens_per_frame=tpf,
        num_action_tokens=n_noised,
        num_future_per_chunk=2,
        chunk_size=fpc,
    )
    band = [1 + (2 + f) * MAX_INTER for f in range(n_noised)]
    proprio = [band[k * fpc] for k in range(kp)]
    future = [band[k * fpc] for k in range(kp) for _ in range(2)]
    assert q.tolist() == [proprio + future + band]


def test_query_rope_ids_checks_the_layout() -> None:
    kv = torch.zeros(1, 10, dtype=torch.long)
    with pytest.raises(ValueError, match="video rope ids"):
        trunk_query_rope_ids(
            kv,
            num_global_tokens=1,
            num_clean_frames=0,
            num_noised_frames=2,
            tokens_per_frame=4,
            num_action_tokens=2,
            num_future_per_chunk=1,
            chunk_size=2,
        )


def test_kv_rope_ids_follow_the_global_order() -> None:
    expert = _trunk_rope_expert()
    win = _window(2)
    seen: dict = {}
    orig = expert.forward_trunk_rope

    def spy(**kw: object) -> torch.Tensor:
        seen.update(kw)
        return orig(**seen)

    expert.forward_trunk_rope = spy  # ty: ignore[invalid-assignment]
    run_expert_on_taps_k(expert, **win)
    rope = win["rope_ids"][0]
    g = win["global_spans"]
    order = [*range(*_span(g["register"])), *range(*_span(g["anchor"]))]
    order += [*range(*_span(g["text"]))]
    order += [*range(*_span(win["clean_span"])), *range(*_span(win["noised_span"]))]
    assert seen["kv_rope_ids"].tolist() == [rope[order].tolist()]
    assert torch.equal(seen["global_tokens"], win["embeddings"][:, order[:11]])
    assert torch.equal(seen["trunk_features"][1], win["taps"][1][:, order[11:]])


def _span(span: tuple[int, int]) -> tuple[int, int]:
    return span[0], span[0] + span[1]


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def _dense_cross_mask(
    *,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    window_size: int,
    num_global_tokens: int,
    device: torch.device,
    include_noised: bool = True,
) -> torch.Tensor:
    """The ``(Q, KV)`` cross-attention rules, written with dense tensors.

    * A query can read all global tokens.
    * A query can read a clean frame strictly before the start of its chunk
      and at most ``window_size`` frames before it.
    * A query can read the noised frames of its own chunk only.
    """
    q_chunk = ae._chunk_of_query(
        num_action_tokens, num_future_per_chunk, chunk_size, device
    )
    offset = ae.noised_frame_offset(num_noised_frames, num_clean_frames, chunk_size)
    chunk_start = q_chunk * chunk_size + offset
    clean_tokens = num_clean_frames * tokens_per_frame
    noised_tokens = num_noised_frames * tokens_per_frame
    kv = torch.arange(num_global_tokens + clean_tokens + noised_tokens, device=device)
    is_global = kv < num_global_tokens
    rel = kv - num_global_tokens
    is_clean = (~is_global) & (rel < clean_tokens)
    is_noised = (~is_global) & (rel >= clean_tokens)
    clean_frame = torch.where(is_clean, rel // tokens_per_frame, torch.zeros_like(rel))
    noised_frame = torch.where(
        is_noised, (rel - clean_tokens) // tokens_per_frame, torch.zeros_like(rel)
    )
    cs = chunk_start[:, None]
    allow_clean = (
        is_clean[None, :]
        & (clean_frame[None, :] < cs)
        & (clean_frame[None, :] >= cs - window_size)
    )
    allow_noised = is_noised[None, :] & (
        (noised_frame[None, :] // chunk_size) == q_chunk[:, None]
    )
    if not include_noised:
        allow_noised = torch.zeros_like(allow_noised)
    return is_global[None, :].expand(q_chunk.numel(), -1) | allow_clean | allow_noised


def _dense_self_mask(
    *,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """The ``(Q, Q)`` self-attention rule: one block per chunk."""
    q_chunk = ae._chunk_of_query(
        num_action_tokens, num_future_per_chunk, chunk_size, device
    )
    return q_chunk[:, None] == q_chunk[None, :]


def test_trunk_masks_hand_made_case() -> None:
    # K' = 2, chunk 2, one token per frame, 2 globals, 1 future token per chunk.
    # Q = [p0 p1 | f0 f1 | a0 a1 a2 a3], KV = [g g | c0 c1 | n0 n1 n2 n3].
    cross, self_ = build_trunk_rope_masks(
        num_action_tokens=4,
        num_future_per_chunk=1,
        chunk_size=2,
        num_clean_frames=2,
        num_noised_frames=4,
        tokens_per_frame=1,
        window_size=6,
        num_global_tokens=2,
        device=torch.device("cpu"),
        block=False,
    )
    c0 = [1, 1, 0, 0, 1, 1, 0, 0]
    c1 = [1, 1, 1, 1, 0, 0, 1, 1]
    chunk = [0, 1, 0, 1, 0, 0, 1, 1]
    assert isinstance(cross, torch.Tensor) and isinstance(self_, torch.Tensor)
    assert cross.int().tolist() == [c0 if c == 0 else c1 for c in chunk]
    assert self_.int().tolist() == [[int(a == b) for b in chunk] for a in chunk]


@pytest.mark.parametrize("kp", [1, 2, 3])
@pytest.mark.parametrize("include_noised", [True, False])
def test_trunk_masks_equal_the_dense_rules(kp: int, include_noised: bool) -> None:
    layout: dict = {
        "num_action_tokens": 6 * kp,
        "num_future_per_chunk": 8,
        "chunk_size": 6,
        "num_clean_frames": 6 * (kp - 1),
        "num_noised_frames": 6 * kp,
        "tokens_per_frame": 5,
        "window_size": 6 * (2 * kp - 1),
        "num_global_tokens": 7,
        "device": torch.device("cpu"),
        "include_noised": include_noised,
    }
    cross, self_ = build_trunk_rope_masks(**layout, block=False)
    assert isinstance(cross, torch.Tensor) and isinstance(self_, torch.Tensor)
    assert torch.equal(cross, _dense_cross_mask(**layout))
    del layout["num_clean_frames"], layout["num_noised_frames"]
    del layout["tokens_per_frame"], layout["window_size"]
    del layout["num_global_tokens"], layout["include_noised"]
    assert torch.equal(self_, _dense_self_mask(**layout))


@pytest.mark.parametrize("kp", [1, 3])
def test_block_mask_tiles(kp: int) -> None:
    layout: dict = {
        "num_action_tokens": 6 * kp,
        "num_future_per_chunk": 8,
        "chunk_size": 6,
        "num_clean_frames": 6 * (kp - 1),
        "num_noised_frames": 6 * kp,
        "tokens_per_frame": 40,
        "window_size": 6 * (2 * kp - 1),
        "num_global_tokens": 150,
        "device": torch.device("cpu"),
    }
    cross_mod, self_mod, q_len, kv_len = trunk_mask_mods(**layout)
    cross, self_ = build_trunk_rope_masks(**layout, block=True)
    dense, _ = build_trunk_rope_masks(**layout, block=False)
    assert isinstance(dense, torch.Tensor)
    for mask, mod, kv in ((cross, cross_mod, kv_len), (self_, self_mod, q_len)):
        assert isinstance(mask, flex_mod.BlockMask)
        assert mask.BLOCK_SIZE == (128, 128)
        assert mask.seq_lengths == (q_len, kv)
        # The same tiles as torch's own dense-to-block conversion.
        grid = mod(None, None, torch.arange(q_len)[:, None], torch.arange(kv)[None])
        tiles = flex_mod._convert_mask_to_block_mask(grid, separate_full_blocks=True)
        ref = flex_mod._create_sparse_block_from_block_mask(tiles, mod, (q_len, kv))
        for name in (
            "kv_num_blocks",
            "kv_indices",
            "full_kv_num_blocks",
            "full_kv_indices",
        ):
            assert torch.equal(getattr(mask, name), getattr(ref, name)), name
    # Padded positions are never attended.
    q_pad, kv_pad = q_len + -q_len % 128, kv_len + -kv_len % 128
    padded = cross_mod(
        None, None, torch.arange(q_pad)[:, None], torch.arange(kv_pad)[None]
    )
    assert torch.equal(padded[:q_len, :kv_len], dense)
    assert not padded[q_len:].any() and not padded[:, kv_len:].any()


# ---------------------------------------------------------------------------
# The trunk-rope expert
# ---------------------------------------------------------------------------


def test_trunk_rope_expert_parameters() -> None:
    k = _trunk_rope_expert()
    assert not hasattr(k, "action_pos")
    assert tuple(k.future_chunk_embed.weight.shape) == (SMALL["max_chunks"], 32)
    assert "future_chunk_embed.weight" in k.state_dict()
    assert not any("action_pos" in name for name in k.state_dict())
    assert k.rope_table.base == 26_000_000


@pytest.mark.parametrize("kp", [1, 2, 3])
def test_trunk_rope_expert_shapes_and_finite(kp: int) -> None:
    out = run_expert_on_taps_k(_trunk_rope_expert(), **_window(kp))
    assert out.shape == (1, 2 * kp, 5)
    assert torch.isfinite(out).all()


def test_trunk_rope_expert_future_chunk_rows_start_at_zero() -> None:
    expert = _trunk_rope_expert()
    win = _window(1)
    base = run_expert_on_taps_k(expert, **win)
    assert expert.future_chunk_embed is not None
    expert.future_chunk_embed.weight[1:] += 5.0  # rows K'.. are unused at K' = 1
    assert torch.equal(run_expert_on_taps_k(expert, **win), base)
    expert.future_chunk_embed.weight[0] += 5.0
    assert not torch.equal(run_expert_on_taps_k(expert, **win), base)


class _NegativeZeroHead(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x.shape[:-1], 5, dtype=x.dtype).neg()


def test_trunk_rope_expert_head_plus_zero() -> None:
    expert = _trunk_rope_expert()
    expert.head = _NegativeZeroHead()  # ty: ignore[invalid-assignment]
    out = run_expert_on_taps_k(expert, **_window(1))
    assert (out == 0).all() and not torch.signbit(out).any()
    expert.head_plus_zero = False
    assert torch.signbit(run_expert_on_taps_k(expert, **_window(1))).all()


def test_trunk_rope_expert_chunk_isolation() -> None:
    # Changing the last noised chunk leaves the earlier chunks' actions unchanged.
    expert = _trunk_rope_expert()
    win = _window(3)
    base = run_expert_on_taps_k(expert, **win)
    start, length = win["noised_span"]
    last = length // 3
    for tap in win["taps"]:
        tap[:, start + 2 * last : start + length] += 3.0
    out = run_expert_on_taps_k(expert, **win)
    assert torch.equal(out[:, :4], base[:, :4])
    assert not torch.equal(out[:, 4:], base[:, 4:])


def test_flex_path_matches_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    # The block-mask path with the eager flex kernel equals the dense SDPA path.
    def eager(*args: object, kernel_options: object = None, **kw: object) -> object:
        return flex_mod.flex_attention(*args, **kw)  # ty: ignore[no-matching-overload]

    monkeypatch.setattr(ae, "_compiled_flex_attention_dynamic", eager)
    win = _window(2)
    sdpa = run_expert_on_taps_k(_trunk_rope_expert(), **win)
    flex = run_expert_on_taps_k(_trunk_rope_expert(attention="flex_dynamic"), **win)
    torch.testing.assert_close(flex, sdpa, atol=1e-5, rtol=1e-5)


def test_trunk_rope_expert_options() -> None:
    # The trunk rope is required, the frame rope and the absolute action
    # table are gone.
    with pytest.raises(TypeError):
        ActionExpert(**SMALL)
    for gone in ("expert_rope", "expert_absolute_positions", "action_pos_table"):
        with pytest.raises(TypeError):
            ActionExpert(**SMALL, rope=TRUNK_ROPE, **{gone: True})
    with pytest.raises(ValueError, match="unsupported rope"):
        _trunk_rope_expert(rope={**TRUNK_ROPE, "kind": "frame"})
    with pytest.raises(ValueError, match="global_order"):
        _trunk_rope_expert(global_order=("register", "text"))
    with pytest.raises(ValueError, match="flex_kernel_options"):
        _trunk_rope_expert(flex_kernel_options={"num_warps": 4})


def test_flex_kernel_options_default() -> None:
    expert = _trunk_rope_expert(attention="flex_dynamic")
    assert expert.flex_kernel_options == {
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 128,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert _trunk_rope_expert().flex_kernel_options is None


# ---------------------------------------------------------------------------
# Bitwise goldens, recorded on the code before the removal of the absolute
# action table (praxis d5678cb).
# ---------------------------------------------------------------------------

# The trunk-rope layout: ``run_expert_on_taps_k(_trunk_rope_expert(), **_window(2))``.
TRUNK_ROPE_GOLDEN = [
    -1.299107313156128, 1.2063676118850708, -3.4436731338500977,
    0.8848135471343994, 0.03333026170730591, -1.2002754211425781,
    0.5011886358261108, -2.9168128967285156, 0.1389312893152237,
    -0.27312636375427246, -1.8609845638275146, -0.03410938382148743,
    -2.8932137489318848, -0.29645735025405884, -0.7213178277015686,
    -1.569164514541626, 0.13758662343025208, -2.5755879878997803,
    0.48850929737091064, -0.531600832939148,
]  # fmt: skip


def test_trunk_rope_layout_is_unchanged() -> None:
    out = run_expert_on_taps_k(_trunk_rope_expert(), **_window(2))
    want = torch.tensor(TRUNK_ROPE_GOLDEN, dtype=torch.float32)
    assert out.shape == (1, 4, 5) and out.dtype == torch.float32
    assert torch.equal(out.flatten(), want)
