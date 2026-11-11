"""Fused rotary embedding with the rounding of the eager ``apply_rope``.

``rope_fused(q, k, table)`` returns the same bits as
:func:`praxis.layers.apply_rope`: both inputs are widened to float32, each
rotated half is ``x0 * cos - x1 * sin`` and ``x1 * cos + x0 * sin`` with every
product and every sum rounded to float32 on its own (FP fusion is disabled,
so no FMA), and the result is rounded to the input dtype (round to nearest
even). The kernel reads ``q``, ``k`` and the table once and writes
contiguous ``(B, H, N, D)`` outputs, like the eager version.

Unsupported layouts or dtypes fall back to the eager function.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_kernel(
    x_ptr,
    out_ptr,
    tab_ptr,
    H,
    N,
    sxb,
    sxh,
    sxn,
    sxd,
    stb,
    stn,
    HALF: tl.constexpr,
    HALF_P2: tl.constexpr,
    ROWS: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    n = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    j = tl.arange(0, HALF_P2)
    m = (n < N)[:, None] & (j < HALF)[None, :]
    xb = (
        x_ptr
        + b.to(tl.int64) * sxb
        + h.to(tl.int64) * sxh
        + n[:, None].to(tl.int64) * sxn
    )
    x0 = tl.load(xb + j[None, :] * sxd, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(xb + (j[None, :] + HALF) * sxd, mask=m, other=0.0).to(tl.float32)
    tb = tab_ptr + b.to(tl.int64) * stb + n[:, None].to(tl.int64) * stn + j[None, :] * 2
    cos = tl.load(tb, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(tb + 1, mask=m, other=0.0).to(tl.float32)
    o0 = x0 * cos - x1 * sin
    o1 = x1 * cos + x0 * sin
    ob = (
        out_ptr
        + bh.to(tl.int64) * N * (2 * HALF)
        + n[:, None].to(tl.int64) * (2 * HALF)
    )
    tl.store(ob + j[None, :], o0.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(ob + HALF + j[None, :], o1.to(out_ptr.dtype.element_ty), mask=m)


def _rotate(x: Tensor, table: Tensor) -> Tensor:
    b, h, n, d = x.shape
    out = torch.empty((b, h, n, d), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    half = d // 2
    rows = 16
    grid = (b * h, triton.cdiv(n, rows))
    stb = table.stride(0) if table.shape[0] == b else 0
    _rope_kernel[grid](
        x,
        out,
        table,
        h,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stb,
        table.stride(1),
        HALF=half,
        HALF_P2=triton.next_power_of_2(half),
        ROWS=rows,
        enable_fp_fusion=False,
    )
    return out


def rope_fused(q: Tensor, k: Tensor, table: Tensor, eager) -> tuple[Tensor, Tensor]:
    """Fused ``eager(q, k, table)`` (``praxis.layers.apply_rope``) with the same bits."""
    ok = (
        q.is_cuda
        and q.dim() == 4
        and k.dim() == 4
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and q.shape[-1] % 2 == 0
        and k.shape[-1] == q.shape[-1]
        and q.shape[2] == k.shape[2]
        and q.shape[0] == k.shape[0]
        and table.dtype == torch.float32
        and table.dim() == 4
        and table.shape[1] == q.shape[2]
        and table.shape[2] == q.shape[-1] // 2
        and table.shape[3] == 2
        and table.stride(3) == 1
        and table.stride(2) == 2
        and table.shape[0] in (1, q.shape[0])
        and not torch.is_grad_enabled()
    )
    if not ok:
        return eager(q, k, table)
    return _rotate(q, table), _rotate(k, table)


@triton.jit
def _rope_norm_kernel(
    x_ptr,
    out_ptr,
    tab_ptr,
    w_ptr,
    H,
    N,
    eps,
    sxb,
    sxh,
    sxn,
    sxd,
    stb,
    stn,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
):
    # RoPE as _rope_kernel, the bf16 store of its output, then the RMSNorm of
    # at::native vectorized_layer_norm_kernel for a float32 row of 128: warp 0
    # lane t sums the squares of elements 4t..4t+3 in order (bf16 squares are
    # exact in float32, so FMA or not gives the same bits), a shuffle-down tree
    # with offsets 16, 8, 4, 2, 1 combines the lanes, rstd = rsqrt(sum / 128 +
    # eps), out = w * (rstd * x). HALF must be 64: lane l < 16 holds the first
    # half, lane 16 + l the second.
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    n = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    j = tl.arange(0, HALF)
    m = (n < N)[:, None] & (j < HALF)[None, :]
    xb = (
        x_ptr
        + b.to(tl.int64) * sxb
        + h.to(tl.int64) * sxh
        + n[:, None].to(tl.int64) * sxn
    )
    x0 = tl.load(xb + j[None, :] * sxd, mask=m, other=0.0).to(tl.float32)
    x1 = tl.load(xb + (j[None, :] + HALF) * sxd, mask=m, other=0.0).to(tl.float32)
    tb = tab_ptr + b.to(tl.int64) * stb + n[:, None].to(tl.int64) * stn + j[None, :] * 2
    cos = tl.load(tb, mask=m, other=0.0).to(tl.float32)
    sin = tl.load(tb + 1, mask=m, other=0.0).to(tl.float32)
    lo = (x0 * cos - x1 * sin).to(out_ptr.dtype.element_ty).to(tl.float32)
    hi = (x1 * cos + x0 * sin).to(out_ptr.dtype.element_ty).to(tl.float32)
    a02, a13 = tl.split(tl.reshape(lo, (ROWS, 16, 2, 2)))
    a0, a2 = tl.split(a02)
    a1, a3 = tl.split(a13)
    b02, b13 = tl.split(tl.reshape(hi, (ROWS, 16, 2, 2)))
    b0, b2 = tl.split(b02)
    b1, b3 = tl.split(b13)
    pa = ((a0 * a0 + a1 * a1) + a2 * a2) + a3 * a3
    pb = ((b0 * b0 + b1 * b1) + b2 * b2) + b3 * b3
    s = pa + pb  # offset 16
    s0, s1 = tl.split(tl.permute(tl.reshape(s, (ROWS, 2, 8)), (0, 2, 1)))
    s = s0 + s1  # offset 8
    s0, s1 = tl.split(tl.permute(tl.reshape(s, (ROWS, 2, 4)), (0, 2, 1)))
    s = s0 + s1  # offset 4
    s0, s1 = tl.split(tl.permute(tl.reshape(s, (ROWS, 2, 2)), (0, 2, 1)))
    s = s0 + s1  # offset 2
    s0, s1 = tl.split(s)
    r = tl.math.rsqrt((s0 + s1) / 128.0 + eps)  # offset 1
    w_lo = tl.load(w_ptr + j)
    w_hi = tl.load(w_ptr + HALF + j)
    o_lo = w_lo[None, :] * (r[:, None] * lo)
    o_hi = w_hi[None, :] * (r[:, None] * hi)
    ob = (
        out_ptr
        + bh.to(tl.int64) * N * (2 * HALF)
        + n[:, None].to(tl.int64) * (2 * HALF)
    )
    tl.store(ob + j[None, :], o_lo.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(ob + HALF + j[None, :], o_hi.to(out_ptr.dtype.element_ty), mask=m)


def _rotate_norm(x: Tensor, table: Tensor, w1: Tensor, eps: float) -> Tensor:
    b, h, n, d = x.shape
    out = torch.empty((b, h, n, d), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out
    rows = 16
    stb = table.stride(0) if table.shape[0] == b else 0
    _rope_norm_kernel[(b * h, triton.cdiv(n, rows))](
        x,
        out,
        table,
        w1,
        h,
        n,
        eps,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stb,
        table.stride(1),
        HALF=d // 2,
        ROWS=rows,
        enable_fp_fusion=False,
    )
    return out


def rope_norm_supported(q: Tensor, k: Tensor, table: Tensor, q_norm, k_norm) -> bool:
    """The layouts :func:`rope_norm_fused` reproduces: head dim 128, float32
    norms with a weight, half-precision CUDA ``q``/``k`` and the float32
    ``(B or 1, N, 64, 2)`` rotary table."""
    return (
        q.is_cuda
        and q.dim() == 4
        and k.dim() == 4
        and q.shape[-1] == 128
        and k.shape[-1] == 128
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and k.device == q.device
        and q.shape[2] == k.shape[2]
        and q.shape[0] == k.shape[0]
        and table.device == q.device
        and table.dtype == torch.float32
        and table.dim() == 4
        and table.shape[1] == q.shape[2]
        and table.shape[2] == 64
        and table.shape[3] == 2
        and table.stride(3) == 1
        and table.stride(2) == 2
        and table.shape[0] in (1, q.shape[0])
        and not torch.is_grad_enabled()
        and all(
            nm.cast_dtype == torch.float32
            and nm.weight is not None
            and nm.weight.device == q.device
            and tuple(nm.normalized_shape) == (128,)
            for nm in (q_norm, k_norm)
        )
    )


def rope_norm_fused(q: Tensor, k: Tensor, table: Tensor, q_norm, k_norm, eager):
    """``q_norm`` / ``k_norm`` of ``eager(q, k, table)`` (the RoPE, then the
    float32 RMSNorm with weight ``1 + w``, then the input dtype) with the same
    bits, in one kernel per tensor. Falls back to the unfused
    ops when a layout, dtype or norm setting differs from the one the kernel
    reproduces (head dim 128, float32 norm with a weight)."""
    if not rope_norm_supported(q, k, table, q_norm, k_norm):
        qr, kr = eager(q, k, table)
        return q_norm(qr), k_norm(kr)
    ws = [nm._inference_weight() for nm in (q_norm, k_norm)]
    return (
        _rotate_norm(q, table, ws[0], float(q_norm.eps)),
        _rotate_norm(k, table, ws[1], float(k_norm.eps)),
    )
