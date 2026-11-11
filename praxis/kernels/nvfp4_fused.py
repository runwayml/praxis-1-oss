"""Fused rescaling, activation and reduction stages for native NVFP4 experts.

The unfused path (``NVFP4RoutedExperts._expert_outputs_sm110``) runs, per MoE layer:

    x.abs().amax(-1)  -> per-expert globals -> _quantize_q -> FP4 GEMM (gate|up)
    _scale_unpad (rescale) -> _swiglu_packed_kernel -> hidden.abs().amax(-1) -> globals -> _quantize_q
    FP4 GEMM (down) -> _scale_unpad (rescale, unpad) -> _combine_kernel

Each elementwise stage reads and writes a full (rows, width) bf16 tensor. This module keeps every rounding and
merges the stages that touch the same data:

* ``row_amax``: one read of x per row, the bf16 ``abs().amax(-1)`` (max is exact, so any order gives its bits);
* ``rescale_swiglu_amax``: the gate|up GEMM output -> rescale (bf16(bf16(g_a * g_w)) product, bf16 rounding) ->
  ``silu(gate) * up`` with the expression of ``_swiglu_packed_kernel`` -> hidden, plus the row amax of hidden
  (fp32 atomic max of exact bf16 magnitudes); the scaled gate|up tensor is never stored;
* ``combine_rescale``: the down GEMM output is rescaled inside the combine (``_scale_unpad`` and
  ``_combine_kernel`` in one pass); the unpadded rows are never stored.

The globals come from the row amaxes exactly as ``expert_global_scales`` computes them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from praxis.kernels.nvfp4 import _expert


@triton.jit
def _row_amax_kernel(X, OUT, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.full((BK,), 0.0, tl.float32)
    for k0 in tl.static_range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        v = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(v))
    tl.store(OUT + row, tl.max(acc, axis=0).to(tl.bfloat16))


def row_amax(x: torch.Tensor) -> torch.Tensor:
    """``x.abs().amax(-1)`` of a contiguous CUDA bf16 (rows, K) tensor (same bits, one read)."""
    out = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    if x.shape[0]:
        _row_amax_kernel[(x.shape[0],)](x, out, x.shape[1], 1024, num_warps=4)
    return out


def globals_from_row_amax(
    row_amax_bf16: torch.Tensor, csr: torch.Tensor, row_indices=None
) -> torch.Tensor:
    """``praxis.kernels.nvfp4.expert_global_scales`` after its ``x.abs().amax(-1)`` step."""
    e = csr.numel() - 1
    ra = row_amax_bf16 if row_indices is None else row_amax_bf16[row_indices]
    rows = torch.arange(ra.numel(), device=ra.device, dtype=csr.dtype)
    expert = torch.bucketize(rows, csr[1:], right=True).clamp_max_(e - 1)
    group_amax = torch.zeros(e, dtype=ra.dtype, device=ra.device)
    group_amax.scatter_reduce_(0, expert.long(), ra, "amax")
    return group_amax.to(ra.dtype).clamp_min(1e-12).float() * (1.0 / 2688.0)


@triton.jit
def _rescale_swiglu_amax_kernel(
    GU,
    OUT,
    AMAX,
    PCSR,
    G,
    W,
    H: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    # _scale_unpad with CSR = PCSR (padded -> padded): expert of the row, bf16 product of the two globals.
    expert = _expert(PCSR, tl.full((1,), 0, tl.int32) + row, E, SEARCH)
    multiplier = (
        (tl.load(G + expert) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
    )
    r64 = row.to(tl.int64)
    gate_raw = tl.load(GU + r64 * (2 * H) + col, mask=mask, other=0).to(tl.float32)
    up_raw = tl.load(GU + r64 * (2 * H) + H + col, mask=mask, other=0).to(tl.float32)
    gate_s = (gate_raw * multiplier).to(tl.bfloat16)
    up_s = (up_raw * multiplier).to(tl.bfloat16)
    # _swiglu_packed_kernel: gate loaded as float32, up kept in bf16.
    gate = gate_s.to(tl.float32)
    out = (gate * tl.sigmoid(gate)).to(up_s.dtype) * up_s
    tl.store(OUT + r64 * H + col, out, mask=mask)
    m = tl.max(tl.where(mask, tl.abs(out.to(tl.float32)), 0.0), axis=0)
    tl.atomic_max(AMAX + row, m)


def rescale_swiglu_amax(gu, padded_csr, activation_globals, weight_globals):
    """(hidden bf16 (P, H), row amax bf16 (P,)) of the padded gate|up GEMM output ``gu`` (P, 2H)."""
    p, h2 = gu.shape
    h = h2 // 2
    e = activation_globals.numel()
    out = torch.empty((p, h), dtype=torch.bfloat16, device=gu.device)
    amax = torch.zeros(p, dtype=torch.float32, device=gu.device)
    if p:
        block = 1024
        _rescale_swiglu_amax_kernel[(p, triton.cdiv(h, block))](
            gu,
            out,
            amax,
            padded_csr,
            activation_globals,
            weight_globals,
            h,
            e,
            (e + 1).bit_length(),
            block,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out, amax.to(torch.bfloat16)


@triton.jit
def _combine_rescale_kernel(
    RAW,
    INVERSE,
    SCALE,
    CSR,
    PCSR,
    G,
    W,
    OUT,
    T: tl.constexpr,
    C: tl.constexpr,
    TOP_K: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    a0 = tl.full((R, BC), 0.0, tl.float32)
    a1 = tl.full((R, BC), 0.0, tl.float32)
    a2 = tl.full((R, BC), 0.0, tl.float32)
    a3 = tl.full((R, BC), 0.0, tl.float32)
    for slot in tl.static_range(TOP_K):
        index = tl.load(INVERSE + rows * TOP_K + slot, rows < T, other=0)
        # _scale_unpad(raw, csr, padded_csr, ...): sorted row -> padded row, bf16 rescale.
        expert = _expert(CSR, index.to(tl.int32), E, SEARCH)
        source = tl.load(PCSR + expert) + index - tl.load(CSR + expert)
        multiplier = (
            (tl.load(G + expert) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
        )
        raw = tl.load(
            RAW + source[:, None].to(tl.int64) * C + cols[None, :],
            (rows[:, None] < T) & (cols[None, :] < C),
            other=0,
        ).to(tl.float32)
        value = (raw * multiplier[:, None]).to(tl.bfloat16).to(tl.float32)
        # _combine_kernel
        scale = tl.load(SCALE + index, rows < T, other=0)
        product = value * scale[:, None]
        if slot % 4 == 0:
            a0 = a0 + product
        elif slot % 4 == 1:
            a1 = a1 + product
        elif slot % 4 == 2:
            a2 = a2 + product
        else:
            a3 = a3 + product
    result = ((a0 + a1) + a2) + a3
    tl.store(
        OUT + rows[:, None].to(tl.int64) * C + cols[None, :],
        result.to(tl.bfloat16),
        (rows[:, None] < T) & (cols[None, :] < C),
    )


def combine_rescale(
    raw,
    inverse,
    sorted_scale,
    csr,
    padded_csr,
    activation_globals,
    weight_globals,
    tokens,
    top_k,
):
    """``combine_rows(rescale(raw, csr, padded_csr, g_a, g_w, rows), inverse, sorted_scale, tokens, top_k)``."""
    c = raw.shape[1]
    e = activation_globals.numel()
    out = torch.empty((tokens, c), device=raw.device, dtype=torch.bfloat16)
    if tokens:
        _combine_rescale_kernel[(triton.cdiv(tokens, 4), triton.cdiv(c, 256))](
            raw,
            inverse,
            sorted_scale,
            csr,
            padded_csr,
            activation_globals,
            weight_globals,
            out,
            tokens,
            c,
            top_k,
            e,
            (e + 1).bit_length(),
            4,
            256,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out
