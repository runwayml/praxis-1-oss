"""Triton grouped GEMM for the routed experts (no host synchronization).

``grouped_mm(x, w, offsets)`` computes, for every expert ``e``, the rows
``x[offsets[e-1]:offsets[e]] @ w[e]`` like ``torch._grouped_mm(x, w, offsets)``
with a 2-D ``x`` (rows sorted by expert) and a 3-D ``w`` (E, K, N). The
offsets stay on the device: each program derives its expert and tile from
the cumulative per-expert tile counts, so there is no offset copy to the host
and no per-expert launch. The launch grid is the upper bound
``(cdiv(M, BM) + E) * cdiv(N, BN)``; programs past the last tile exit.

Tiles are ordered expert-major (all tiles of expert 0, then expert 1, ...)
and, inside an expert, row tiles vary fastest, so the programs that run
together share one weight column block and the expert's rows stay in L2. Each
weight byte is read from DRAM about once per call.

Accumulation is float32 with one rounding to the output dtype, but the K
order differs from cuBLAS, so results are not bitwise equal to
``torch._grouped_mm``. Unsupported layouts or dtypes fall back to it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

DEFAULT_CONFIG = {
    "BM": 128,
    "BN": 128,
    "BK": 64,
    "num_warps": 8,
    "num_stages": 3,
    "USE_TMA": False,
}


@triton.jit
def _grouped_mm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    offs_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, BN)
    idx = tl.arange(0, E_POW2)
    valid = idx < E
    ends_rows = tl.load(offs_ptr + idx, mask=valid, other=0)
    starts_rows = tl.load(offs_ptr + idx - 1, mask=valid & (idx > 0), other=0)
    m_tiles = tl.where(valid, (ends_rows - starts_rows + BM - 1) // BM, 0)
    tile_end = tl.cumsum(m_tiles * nt, 0)
    e = tl.sum((tile_end <= pid).to(tl.int32), 0)
    if e >= E:
        return
    sel = idx == e
    tile_start = tl.sum(tl.where(sel, tile_end - m_tiles * nt, 0), 0)
    mt = tl.sum(tl.where(sel, m_tiles, 0), 0)
    row0 = tl.sum(tl.where(sel, starts_rows, 0), 0)
    row1 = tl.sum(tl.where(sel, ends_rows, 0), 0)
    local = pid - tile_start
    pid_n = local // mt
    pid_m = local % mt

    rm = row0 + pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < row1
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if USE_TMA:
        # Tensor-memory-accelerator loads. x rows past this expert belong to the next
        # expert (or are zero past M); their products are computed but not stored.
        # Requires contiguous x (M, K) and w (E, K, N) viewed as (E*K, N).
        M = tl.load(offs_ptr + E - 1)
        xd = tl.make_tensor_descriptor(
            x_ptr, shape=[M, K], strides=[stride_xm, 1], block_shape=[BM, BK]
        )
        wd = tl.make_tensor_descriptor(
            w_ptr, shape=[E * K, N], strides=[stride_wn * N, 1], block_shape=[BK, BN]
        )
        m0 = row0 + pid_m * BM
        for k0 in range(0, K, BK):
            a = xd.load([m0, k0])
            b = wd.load([e * K + k0, pid_n * BN])
            acc = tl.dot(a, b, acc)
    else:
        xp = x_ptr + rm[:, None].to(tl.int64) * stride_xm + rk[None, :] * stride_xk
        wp = (
            w_ptr
            + e.to(tl.int64) * stride_we
            + rk[:, None] * stride_wk
            + rn[None, :] * stride_wn
        )
        for k0 in range(0, K, BK):
            k_ok = (k0 + rk) < K
            a = tl.load(xp, mask=m_ok[:, None] & k_ok[None, :], other=0.0)
            b = tl.load(wp, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            xp += BK * stride_xk
            wp += BK * stride_wk
    op = out_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :] * stride_on
    tl.store(op, acc.to(out_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def grouped_mm(
    x: Tensor, w: Tensor, offsets: Tensor, config: dict | None = None
) -> Tensor:
    """``torch._grouped_mm(x, w, offsets)`` for 2-D x, 3-D w, without host sync."""
    fast = (
        x.is_cuda
        and x.dim() == 2
        and w.dim() == 3
        and offsets.dim() == 1
        and offsets.dtype == torch.int32
        and offsets.numel() == w.shape[0]
        and x.dtype in (torch.bfloat16, torch.float16)
        and w.dtype == x.dtype
        and x.shape[1] == w.shape[1]
        and x.stride(1) == 1
        and not torch.is_grad_enabled()
    )
    if not fast:
        return torch._grouped_mm(x, w, offsets)
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    m, k = x.shape
    e, _, n = w.shape
    out = x.new_empty(m, n)
    if m == 0 or n == 0:
        return out
    # Rows past offsets[-1] are left untouched by torch._grouped_mm as well; the
    # routed path always has offsets[-1] == m.
    use_tma = (
        bool(cfg["USE_TMA"])
        and x.is_contiguous()
        and w.is_contiguous()
        and k % cfg["BK"] == 0
    )
    if use_tma:
        triton.set_allocator(_tma_alloc)
    grid = ((triton.cdiv(m, cfg["BM"]) + e) * triton.cdiv(n, cfg["BN"]),)
    _grouped_mm_kernel[grid](
        x,
        w,
        out,
        offsets,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        E=e,
        E_POW2=triton.next_power_of_2(e),
        BM=cfg["BM"],
        BN=cfg["BN"],
        BK=cfg["BK"],
        USE_TMA=use_tma,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


# ---------------------------------------------------------------------------
# Routed gate_up GEMM with the dispatch gather and the SwiGLU fused in
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_gate_up_swiglu_kernel(
    x_ptr,
    order_ptr,
    w_ptr,
    out_ptr,
    offs_ptr,
    H,
    K,
    stride_xm,
    stride_we,
    stride_wk,
    stride_wn,
    stride_om,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACT: tl.constexpr = 0,
):
    """``swiglu_packed(grouped_mm(rows, w, offs))`` for w (E, K, 2H), one launch.

    Output tile (BM, BN) of the (M, H) result accumulates the gate columns
    ``n`` and the up columns ``H + n`` of the same rows, each like an
    output tile of ``_grouped_mm_kernel`` (same tile walk over K). Both are
    rounded to the output dtype, as the GEMM stores them, before the SwiGLU of
    ``_swiglu_packed_kernel``. With GATHER, row ``i`` is read from
    ``x[order[i] // TOPK]`` (the row ``moe_dispatch`` would have copied).
    """
    pid = tl.program_id(0)
    nt = tl.cdiv(H, BN)
    idx = tl.arange(0, E_POW2)
    valid = idx < E
    ends_rows = tl.load(offs_ptr + idx, mask=valid, other=0)
    starts_rows = tl.load(offs_ptr + idx - 1, mask=valid & (idx > 0), other=0)
    m_tiles = tl.where(valid, (ends_rows - starts_rows + BM - 1) // BM, 0)
    tile_end = tl.cumsum(m_tiles * nt, 0)
    e = tl.sum((tile_end <= pid).to(tl.int32), 0)
    if e >= E:
        return
    sel = idx == e
    tile_start = tl.sum(tl.where(sel, tile_end - m_tiles * nt, 0), 0)
    mt = tl.sum(tl.where(sel, m_tiles, 0), 0)
    row0 = tl.sum(tl.where(sel, starts_rows, 0), 0)
    row1 = tl.sum(tl.where(sel, ends_rows, 0), 0)
    local = pid - tile_start
    pid_n = local // mt
    pid_m = local % mt

    rm = row0 + pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < row1
    n_ok = rn < H
    if GATHER:
        src = tl.load(order_ptr + rm, mask=m_ok, other=0) // TOPK
    else:
        src = rm.to(tl.int64)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    xp = x_ptr + src[:, None].to(tl.int64) * stride_xm + rk[None, :]
    wp = (
        w_ptr
        + e.to(tl.int64) * stride_we
        + rk[:, None] * stride_wk
        + rn[None, :] * stride_wn
    )
    wu = H * stride_wn
    for k0 in range(0, K, BK):
        k_ok = (k0 + rk) < K
        a = tl.load(xp, mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        bg = tl.load(wp, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
        bu = tl.load(wp + wu, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        xp += BK
        wp += BK * stride_wk
    up = acc_u.to(out_ptr.dtype.element_ty)
    gate = acc_g.to(out_ptr.dtype.element_ty).to(tl.float32)
    if ACT == 0:
        # _swiglu_packed_kernel (routed experts)
        out = (gate * tl.sigmoid(gate)).to(up.dtype) * up
    else:
        # F.silu(gate) * up (shared expert): ATen's CUDA silu is x / (1 + expf(-x)) in float32 with an
        # IEEE division and the precise expf; then the bf16 product.
        act = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate))
        out = act.to(up.dtype) * up
    op = out_ptr + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
    tl.store(op, out, mask=m_ok[:, None] & n_ok[None, :])


SWIGLU_CONFIG = {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 3}


def gate_up_swiglu_supported(x: Tensor, w: Tensor, offsets: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and w.dim() == 3
        and offsets.dim() == 1
        and offsets.dtype == torch.int32
        and offsets.numel() == w.shape[0]
        and x.dtype in (torch.bfloat16, torch.float16)
        and w.dtype == x.dtype
        and x.shape[1] == w.shape[1]
        and w.shape[2] % 2 == 0
        and x.stride(1) == 1
        and not torch.is_grad_enabled()
    )


def grouped_gate_up_swiglu(
    x: Tensor,
    w: Tensor,
    offsets: Tensor,
    order: Tensor | None = None,
    top_k: int = 1,
    rows: int | None = None,
    config: dict | None = None,
    act: int = 0,
) -> Tensor:
    """``swiglu_packed(grouped_mm(r, w, offsets))`` with ``r = x`` or, given
    ``order``, ``r[i] = x[order[i] // top_k]`` (``rows`` = len(order)).
    ``act=1``: ``F.silu(gate) * up`` instead (the shared expert's rounding)."""
    cfg = {**SWIGLU_CONFIG, **(config or {})}
    m = x.shape[0] if order is None else (rows if rows is not None else order.numel())
    k = x.shape[1]
    e, _, n2 = w.shape
    h = n2 // 2
    out = x.new_empty(m, h)
    if m == 0:
        return out
    grid = ((triton.cdiv(m, cfg["BM"]) + e) * triton.cdiv(h, cfg["BN"]),)
    _grouped_gate_up_swiglu_kernel[grid](
        x,
        order if order is not None else offsets,
        w,
        out,
        offsets,
        h,
        k,
        x.stride(0),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        E=e,
        E_POW2=triton.next_power_of_2(e),
        TOPK=top_k,
        GATHER=order is not None,
        BM=cfg["BM"],
        BN=cfg["BN"],
        BK=cfg["BK"],
        ACT=act,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


# ---------------------------------------------------------------------------
# Shared expert F.silu(gate) * up in one elementwise kernel ()
# ---------------------------------------------------------------------------


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, n, H, stride_gu, stride_out, BLOCK: tl.constexpr):
    """``F.silu(gu[:, :H]) * gu[:, H:]`` over the flattened (M, H) output."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n
    row = i // H
    col = i - row * H
    gp = gu_ptr + row * stride_gu + col
    gate = tl.load(gp, mask=ok, other=0.0).to(tl.float32)
    up = tl.load(gp + H, mask=ok, other=0.0)
    # ATen's CUDA silu (float32, IEEE division, precise expf), then the bf16 product.
    act = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate))
    tl.store(out_ptr + row * stride_out + col, act.to(up.dtype) * up, mask=ok)


SILU_MUL_BLOCK = 2048


def silu_mul(gate_up: Tensor, block: int | None = None) -> Tensor:
    """``F.silu(g) * u`` for ``g, u = gate_up.chunk(2, -1)`` (2-D, unit column stride), one kernel instead of
    two; same roundings, contiguous (M, H) output like the eager expression."""
    m, h2 = gate_up.shape
    h = h2 // 2
    out = gate_up.new_empty(m, h)
    n = m * h
    if n == 0:
        return out
    blk = block or SILU_MUL_BLOCK
    _silu_mul_kernel[(triton.cdiv(n, blk),)](
        gate_up, out, n, h, gate_up.stride(0), out.stride(0), BLOCK=blk, num_warps=8
    )
    return out
