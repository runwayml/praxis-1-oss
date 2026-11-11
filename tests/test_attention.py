"""Optional DiT attention backends preserve masks and tower selection."""

import copy
import importlib.util

import pytest
import torch
from test_backbone import BACKBONE_CHUNK_CAUSAL
from torch.nn.attention.flex_attention import create_block_mask

from praxis.attention import (
    _interval_endpoints,
    flash_mask_block_size,
    optional_attention,
)
from praxis.backbone import _slice_q_blocks, build_backbone, teacher_forcing_mask
from praxis.layers import Attention


def _dense_intervals(endpoints, sk):
    keys = torch.arange(sk)[None, :]
    rows = endpoints[0].T
    visible = keys < rows[:, :1]
    for i in range(1, rows.shape[1], 2):
        visible |= (keys >= rows[:, i : i + 1]) & (keys < rows[:, i + 1 : i + 2])
    return visible


@pytest.mark.parametrize("kind", ["disjoint", "suffix", "empty", "full", "varying"])
def test_interval_conversion_preserves_visible_keys(kind):
    rules = {
        "disjoint": lambda b, h, q, k: (k < 3) | (k == 9),
        "suffix": lambda b, h, q, k: k >= q,
        "empty": lambda b, h, q, k: k < 0,
        "full": lambda b, h, q, k: k >= 0,
        "varying": lambda b, h, q, k: (k < q % 5) | ((k >= 130) & (k < q)),
    }
    rule = rules[kind]
    sq, sk = 257, 273
    mask = create_block_mask(rule, 1, 1, sq, sk, device="cpu")
    endpoints = _interval_endpoints(mask, sq, sk, "cpu")
    expected = rule(0, 0, torch.arange(sq)[:, None], torch.arange(sk)[None, :]).expand(
        sq, sk
    )
    assert endpoints.dtype == torch.int32 and endpoints.is_contiguous()
    assert endpoints.shape[1] % 2 == 1
    assert torch.equal(_dense_intervals(endpoints, sk), expected)


@pytest.mark.parametrize("cached_rows", [False, True])
def test_interval_conversion_preserves_window_context_mask(cached_rows):
    slots = [("global", 1, 5), ("noised", 12, 16), ("clean", 6, 16)]
    n = 5 + 12 * 16 + 6 * 16
    mask = teacher_forcing_mask(slots, 6, True, 128, torch.device("cpu"), False)
    if cached_rows:
        mask = _slice_q_blocks(mask, 128, n)
    sq = n - 128 if cached_rows else n
    expected = mask.mask_mod(0, 0, torch.arange(sq)[:, None], torch.arange(n)[None, :])
    endpoints = _interval_endpoints(mask, sq, n, "cpu")
    assert torch.equal(_dense_intervals(endpoints, n), expected)


def test_prefix_conversion_preserves_partial_rows():
    mask = create_block_mask(lambda b, h, q, k: k <= q, 1, 1, 257, 257, device="cpu")
    actual = _interval_endpoints(mask, 257, 257, "cpu")
    torch.testing.assert_close(
        actual, torch.arange(1, 258, dtype=torch.int32).view(1, 1, -1)
    )


@pytest.mark.parametrize("backend", ["triton", "flash", "cudnn"])
def test_towers_select_backends_independently(backend):
    cfg = copy.deepcopy(BACKBONE_CHUNK_CAUSAL)
    cfg["encoder"]["attention_backend"] = backend
    with torch.device("meta"):
        backbone = build_backbone(cfg)
    assert all(b.mha.backend == backend for b in backbone.encoder.core.blocks)
    assert all(b.mha.backend == "triton" for b in backbone.decoder.core.blocks)
    cfg["attention"]["backend"] = backend
    cfg["decoder"]["attention_backend"] = "triton"
    with torch.device("meta"):
        backbone = build_backbone(cfg)
    assert all(b.mha.backend == "triton" for b in backbone.decoder.core.blocks)


@pytest.mark.parametrize("backend", ["unknown", "fa4"])
def test_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="unknown attention backend"):
        Attention(backend=backend)


@pytest.mark.parametrize("backend", ["cudnn", "flash"])
@pytest.mark.parametrize("masked", [False, True, "disjoint"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend test")
def test_gpu_attention_matches_dense_reference(backend, masked):
    package = "cudnn" if backend == "cudnn" else "flash_attn"
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"optional {package} package is not installed")
    torch.manual_seed(7)
    # Unequal head counts and a partial final tile catch layout/mask mistakes.
    q = torch.randn(1, 4, 257, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(1, 2, 257, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    rule = (
        (lambda b, h, qi, ki: (ki < 3) | ((ki >= 130) & (ki <= qi)))
        if masked == "disjoint"
        else (lambda b, h, qi, ki: ki <= qi)
    )
    block_size = flash_mask_block_size(q.device) if backend == "flash" else 128
    mask = (
        create_block_mask(rule, 1, 1, 257, 257, device="cuda", BLOCK_SIZE=block_size)
        if masked
        else None
    )
    dense = (
        rule(
            0,
            0,
            torch.arange(257, device="cuda")[:, None],
            torch.arange(257, device="cuda")[None, :],
        )
        if masked
        else None
    )
    with torch.inference_mode():
        actual = optional_attention(backend, q, k, v, block_mask=mask, enable_gqa=True)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=dense,
            enable_gqa=True,
        ).to(q.dtype)
        repeat = optional_attention(backend, q, k, v, block_mask=mask, enable_gqa=True)
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
    assert torch.equal(actual, repeat)
