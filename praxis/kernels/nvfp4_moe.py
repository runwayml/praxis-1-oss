"""QAT-preserving reductions and activation kernels for routed NVFP4 experts."""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch import Tensor

from praxis.kernels.nvfp4 import _expert, _sm100_tile_metadata


@triton.jit
def _row_absmax(X, MAXIMUM, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row.to(tl.int64) * K + col, col < K, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=0)
    tl.store(MAXIMUM + row, maximum.to(tl.bfloat16))


@triton.jit
def _segment_global_scale(
    ROW_MAXIMUM,
    CSR,
    MAP,
    GLOBAL,
    EXPERTS: tl.constexpr,
    ROWS: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    begin = tl.load(CSR + expert)
    end = tl.load(CSR + expert + 1)
    begin = tl.maximum(tl.minimum(begin, ROWS), 0)
    end = tl.maximum(tl.minimum(end, ROWS), 0)
    # The reference assigns rows past the final CSR endpoint to the last expert.
    if expert == EXPERTS - 1:
        end = ROWS
    col = tl.arange(0, BLOCK)
    accumulated = tl.full((BLOCK,), 0, tl.float32)
    for start in range(begin, end, BLOCK):
        pos = start + col
        index = pos
        if GATHER:
            index = tl.load(MAP + pos, pos < end, other=0)
            index = tl.where(index < 0, index + SOURCE_ROWS, index)
        value = tl.load(ROW_MAXIMUM + index, pos < end, other=0).to(tl.float32)
        accumulated = tl.maximum(accumulated, value)
    maximum = tl.max(accumulated, axis=0)
    # QAT rounds the clamp to BF16 before multiplying by the FP32 reciprocal.
    clamped = tl.maximum(maximum, 1.0e-12).to(tl.bfloat16).to(tl.float32)
    tl.store(GLOBAL + expert, clamped * (1.0 / 2688.0))


def expert_scales_from_row_maximum(
    row_maximum: Tensor, csr: Tensor, row_indices: Tensor | None = None
) -> Tensor:
    """Reduce BF16 row maxima by expert without contended scatter atomics."""
    experts = csr.numel() - 1
    rows = row_maximum.numel() if row_indices is None else row_indices.numel()
    result = torch.empty(experts, device=row_maximum.device, dtype=torch.float32)
    _segment_global_scale[(experts,)](
        row_maximum,
        csr,
        row_maximum if row_indices is None else row_indices,
        result,
        experts,
        rows,
        row_maximum.numel(),
        row_indices is not None,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return result


def segmented_global_scales(
    x: Tensor, csr: Tensor, row_indices: Tensor | None = None
) -> Tensor:
    """Compute QAT global scales in two kernels for contiguous CUDA BF16 rows."""
    maximum = torch.empty(x.shape[0], device=x.device, dtype=torch.bfloat16)
    if x.shape[0]:
        _row_absmax[(x.shape[0],)](
            x,
            maximum,
            x.shape[1],
            triton.next_power_of_2(x.shape[1]),
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return expert_scales_from_row_maximum(maximum, csr, row_indices)


@triton.jit
def _rescale_swiglu_maximum(
    RAW,
    HIDDEN,
    MAXIMUM,
    CSR,
    ACTIVATION_GLOBAL,
    WEIGHT_GLOBAL,
    HIDDEN_WIDTH: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    expert = _expert(CSR, row, EXPERTS, (EXPERTS + 1).bit_length())
    multiplier = (
        (tl.load(ACTIVATION_GLOBAL + expert) * tl.load(WEIGHT_GLOBAL + expert))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    base = RAW + row.to(tl.int64) * (2 * HIDDEN_WIDTH)
    gate = tl.load(base + col, col < HIDDEN_WIDTH, other=0).to(tl.float32)
    up = tl.load(base + HIDDEN_WIDTH + col, col < HIDDEN_WIDTH, other=0).to(tl.float32)
    gate = (gate * multiplier).to(tl.bfloat16).to(tl.float32)
    up = (up * multiplier).to(tl.bfloat16).to(tl.float32)
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    hidden = (silu * up).to(tl.bfloat16)
    tl.store(HIDDEN + row.to(tl.int64) * HIDDEN_WIDTH + col, hidden, col < HIDDEN_WIDTH)
    maximum = tl.max(
        tl.abs(tl.where(col < HIDDEN_WIDTH, hidden.to(tl.float32), 0)), axis=0
    )
    tl.store(MAXIMUM + row, maximum.to(tl.bfloat16))


def rescale_swiglu_maximum(
    raw: Tensor, csr: Tensor, activation_globals: Tensor, weight_globals: Tensor
) -> tuple[Tensor, Tensor]:
    """Fuse rescale and SwiGLU; retain every intermediate BF16 rounding."""
    rows, doubled_hidden = raw.shape
    hidden_width = doubled_hidden // 2
    hidden = torch.empty((rows, hidden_width), device=raw.device, dtype=torch.bfloat16)
    maximum = torch.empty(rows, device=raw.device, dtype=torch.bfloat16)
    _rescale_swiglu_maximum[(rows,)](
        raw,
        hidden,
        maximum,
        csr,
        activation_globals,
        weight_globals,
        hidden_width,
        activation_globals.numel(),
        triton.next_power_of_2(hidden_width),
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return hidden, maximum


def _make_static_metadata(
    device_index: int, rows: int
) -> tuple[Tensor, Tensor, Tensor]:
    device = torch.device("cuda", device_index)
    return (
        torch.full((1,), rows // 128, device=device, dtype=torch.int32),
        torch.arange(rows, device=device, dtype=torch.int32),
        torch.ones((rows, 1), device=device, dtype=torch.float32),
    )


@lru_cache(maxsize=64)
def _static_metadata(
    device_index: int, stream: int, rows: int
) -> tuple[Tensor, Tensor, Tensor]:
    # Each entry is immutable and is only consumed on its creating stream.
    return _make_static_metadata(device_index, rows)


def grouped_metadata(csr: Tensor, rows: int) -> tuple[Tensor, ...]:
    """Build route metadata once for both fixed-tile SM100 grouped GEMMs."""
    tiles = rows // 128
    experts = torch.empty(tiles, device=csr.device, dtype=torch.int32)
    limits = torch.empty_like(experts)
    _sm100_tile_metadata[(triton.cdiv(tiles, 128),)](
        csr,
        experts,
        limits,
        csr.numel() - 1,
        tiles,
        128,
        num_warps=4,
    )
    device_index = csr.device.index
    assert device_index is not None
    with torch.cuda.device(csr.device):
        if torch.cuda.is_current_stream_capturing():
            # Captured initialization must not populate a cache shared by graphs.
            count, identity, unit_scale = _make_static_metadata(device_index, rows)
        else:
            stream = torch.cuda.current_stream(csr.device).cuda_stream
            count, identity, unit_scale = _static_metadata(device_index, stream, rows)
    return experts, limits, count, identity, unit_scale


@triton.jit
def _rescale_combine(
    RAW,
    INVERSE,
    TOP,
    GATE,
    CSR,
    PADDED_CSR,
    ACTIVATION_GLOBAL,
    WEIGHT_GLOBAL,
    SHARED,
    OUT,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALING: tl.constexpr,
    WITH_SHARED: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    a0 = tl.full((ROWS, COLS), 0, tl.float32)
    a1 = tl.full((ROWS, COLS), 0, tl.float32)
    a2 = tl.full((ROWS, COLS), 0, tl.float32)
    a3 = tl.full((ROWS, COLS), 0, tl.float32)
    mask = (rows[:, None] < TOKENS) & (cols[None, :] < WIDTH)
    for slot in tl.static_range(TOP_K):
        index = tl.load(INVERSE + rows * TOP_K + slot, rows < TOKENS, other=0)
        expert = tl.load(TOP + rows * TOP_K + slot, rows < TOKENS, other=0)
        source = tl.load(PADDED_CSR + expert) + index - tl.load(CSR + expert)
        multiplier = (
            (tl.load(ACTIVATION_GLOBAL + expert) * tl.load(WEIGHT_GLOBAL + expert))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        value = tl.load(
            RAW + source[:, None].to(tl.int64) * WIDTH + cols[None, :],
            mask,
            other=0,
        ).to(tl.float32)
        value = (value * multiplier[:, None]).to(tl.bfloat16).to(tl.float32)
        gate = tl.load(GATE + rows * TOP_K + slot, rows < TOKENS, other=0).to(
            tl.float32
        )
        scale = gate * SCALING
        product = value * scale[:, None]
        if slot % 4 == 0:
            a0 = a0 + product
        elif slot % 4 == 1:
            a1 = a1 + product
        elif slot % 4 == 2:
            a2 = a2 + product
        else:
            a3 = a3 + product
    result = (((a0 + a1) + a2) + a3).to(tl.bfloat16)
    if WITH_SHARED:
        shared = tl.load(
            SHARED + rows[:, None].to(tl.int64) * WIDTH + cols[None, :],
            mask,
            other=0,
        ).to(tl.float32)
        result = (result.to(tl.float32) + shared).to(tl.bfloat16)
    tl.store(OUT + rows[:, None].to(tl.int64) * WIDTH + cols[None, :], result, mask)


def rescale_combine(
    raw: Tensor,
    inverse: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    csr: Tensor,
    padded_csr: Tensor,
    activation_globals: Tensor,
    weight_globals: Tensor,
    scaling: float,
    shared: Tensor | None = None,
) -> Tensor:
    """Fuse down rescale and routed reduction without moving rounding points."""
    tokens, top_k = top_idx.shape
    width = raw.shape[1]
    result = torch.empty((tokens, width), device=raw.device, dtype=torch.bfloat16)
    if tokens:
        _rescale_combine[(triton.cdiv(tokens, 4), triton.cdiv(width, 256))](
            raw,
            inverse,
            top_idx,
            gate,
            csr,
            padded_csr,
            activation_globals,
            weight_globals,
            result if shared is None else shared,
            result,
            tokens,
            width,
            top_k,
            scaling,
            shared is not None,
            4,
            256,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return result
