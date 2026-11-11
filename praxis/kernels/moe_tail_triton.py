"""``praxis.layers.MOE_TAIL_NORM``: the routed-expert tail fused into the next norm.

See the section comment below and ``praxis.layers.MoETail``.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from praxis.kernels.norm_triton import _scale, _w1, mod_geom, mod_row
from praxis.kernels.nvfp4 import _expert
from praxis.kernels.nvfp4_pair import _row_map

# ---------------------------------------------------------------------------
# layers.MOE_TAIL_NORM: the routed-expert tail of the previous block (token combine of
# the routed rows, shared-expert add) computed inside the next block's residual add + norm, with the bits of the
# stock kernels it replaces:
#
# * MODE 1, BF16 chunked regime (layers_triton._moe_chunked_combine_kernel with ADD_SHARED): per slot
#   ``float(raw) * (float(gate) * SCALE)`` in float32, four interleaved accumulators, ``((a0 + a1) + a2) + a3``,
#   rounded to bf16, then ``+ float(shared)`` rounded to bf16;
# * MODE 2, NVFP4 one chunk (nvfp4_fused._combine_rescale_kernel, then the eager ``result + shared``): the padded
#   down-GEMM row of the sorted row, ``bf16(raw * float(bf16(g_a[e] * g_w[e])))``, times the float32 sorted scale,
#   the same accumulators, rounded to bf16, then the eager bf16 add of the shared row;
# * MODE 3, NVFP4 CFG pair (nvfp4_pair._combine_rescale_seg, then the eager add): MODE 2 with the (expert, pass)
#   segment global.
#
# The result ``y`` (bf16) then enters the residual add exactly as the stored ``y`` of add_norm_scale /
# add_mod_norm_scale did (``bf16(float(x) + float(y))``); the norm part is unchanged. ``y`` is never stored.
# ---------------------------------------------------------------------------


@triton.jit
def _tail_y(
    token,
    offs,
    m,
    N,
    EX,
    INV,
    GS,
    SH,
    SEG,
    CSR,
    PCSR,
    G,
    W,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    a0 = tl.zeros(offs.shape, dtype=tl.float32)
    a1 = tl.zeros(offs.shape, dtype=tl.float32)
    a2 = tl.zeros(offs.shape, dtype=tl.float32)
    a3 = tl.zeros(offs.shape, dtype=tl.float32)
    one = tl.arange(0, 1)
    for slot in tl.static_range(8):
        if MODE == 1:
            row = tl.load(INV + token * 8 + slot)
            gate = tl.load(GS + row).to(tl.float32)
            scale = gate * tl.full((), SCALE, tl.float32)
            raw = tl.load(EX + row * N + offs, mask=m, other=0).to(tl.float32)
            value = raw * scale
        else:
            index = tl.load(INV + token * 8 + slot + one)  # (1,)
            if MODE == 2:
                expert = _expert(CSR, index.to(tl.int32), E, SEARCH)
                source = tl.load(PCSR + expert) + index - tl.load(CSR + expert)
                gseg = expert
            else:
                gseg, expert, local = _row_map(
                    SEG, CSR, PCSR, index.to(tl.int32), E, SEARCH, SEARCH, False
                )
                source = tl.load(PCSR + expert) + local
            multiplier = (
                (tl.load(G + gseg) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
            )
            src = tl.sum(source.to(tl.int64), axis=0)
            mult = tl.sum(multiplier, axis=0)
            scale = tl.sum(tl.load(GS + index), axis=0)
            raw = tl.load(EX + src * N + offs, mask=m, other=0).to(tl.float32)
            value = (raw * mult).to(tl.bfloat16).to(tl.float32) * scale
        if slot % 4 == 0:
            a0 = a0 + value
        elif slot % 4 == 1:
            a1 = a1 + value
        elif slot % 4 == 2:
            a2 = a2 + value
        else:
            a3 = a3 + value
    routed = (((a0 + a1) + a2) + a3).to(tl.bfloat16).to(tl.float32)
    shared = tl.load(SH + token * N + offs, mask=m, other=0.0).to(tl.float32)
    return (routed + shared).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tail_add_norm_scale_kernel(
    x_ptr,
    xo_ptr,
    h_ptr,
    w_ptr,
    EX,
    INV,
    GS,
    SH,
    SEG,
    CSR,
    PCSR,
    G,
    W,
    N,
    s,
    eps,
    NI: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    """``_add_norm_scale_kernel`` (HAS_Y, DIV, no F32) with ``y`` from ``_tail_y``."""
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    w4 = tl.arange(0, 4)
    l32 = tl.arange(0, 32)
    e4 = tl.arange(0, 4)
    idx = w4[:, None, None] * 128 + l32[None, :, None] * 4 + e4[None, None, :]
    acc = tl.zeros((4, 32), dtype=tl.float32)
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        xv = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        yv = _tail_y(
            row,
            offs,
            m,
            N,
            EX,
            INV,
            GS,
            SH,
            SEG,
            CSR,
            PCSR,
            G,
            W,
            MODE,
            SCALE,
            E,
            SEARCH,
        )
        xb = (xv + yv).to(xo_ptr.dtype.element_ty)
        tl.store(xo_ptr + base + offs, xb, mask=m)
        xv = xb.to(tl.float32)
        sq = xv * xv
        q02, q13 = tl.split(tl.reshape(sq, (4, 32, 2, 2)))
        q0, q2 = tl.split(q02)
        q1, q3 = tl.split(q13)
        acc = acc + q0
        acc = acc + q1
        acc = acc + q2
        acc = acc + q3
    t = tl.permute(tl.reshape(acc, (4, 2, 16)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 8)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 4)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 2)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    a, b = tl.split(t)
    wsum = a + b
    t = tl.permute(tl.reshape(wsum, (2, 2)), (1, 0))
    a, b = tl.split(t)
    t = a + b
    a, b = tl.split(t)
    ss = a + b
    v = tl.math.div_rn(ss, N.to(tl.float32)) + eps
    r = tl.math.rsqrt(v)
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        xv = tl.load(xo_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=m, other=0.0)
        o = (wv * (r * xv)).to(h_ptr.dtype.element_ty).to(tl.float32)
        hb = (o * s).to(h_ptr.dtype.element_ty)
        tl.store(h_ptr + base + offs, hb, mask=m)


@triton.jit
def _tail_add_mod_norm_scale_kernel(
    x_ptr,
    m_ptr,
    xo_ptr,
    h_ptr,
    w_ptr,
    EX,
    INV,
    GS,
    SH,
    SEG,
    CSR,
    PCSR,
    G,
    W,
    N,
    s,
    eps,
    RF,
    RC,
    NREG,
    PW,
    WD,
    PH,
    NI: tl.constexpr,
    MODFMA: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    CMOD: tl.constexpr = False,
):
    """``_add_mod_norm_scale_kernel`` (HAS_Y) with ``y`` from ``_tail_y``; CMOD: compact modulation rows (MOD_COMPACT,
    ``norm_triton.mod_row``)."""
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    mbase = mod_row(row, RF, RC, NREG, PW, WD, PH, CMOD) * 2 * N
    w4 = tl.arange(0, 4)
    l32 = tl.arange(0, 32)
    e4 = tl.arange(0, 4)
    idx = w4[:, None, None] * 128 + l32[None, :, None] * 4 + e4[None, None, :]
    acc = tl.zeros((4, 32), dtype=tl.float32)
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        xv = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        yv = _tail_y(
            row,
            offs,
            m,
            N,
            EX,
            INV,
            GS,
            SH,
            SEG,
            CSR,
            PCSR,
            G,
            W,
            MODE,
            SCALE,
            E,
            SEARCH,
        )
        xv = (xv + yv).to(xo_ptr.dtype.element_ty).to(tl.float32)
        sh = tl.load(m_ptr + mbase + offs, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(m_ptr + mbase + N + offs, mask=m, other=0.0).to(tl.float32)
        sc1 = (sc + 1.0).to(xo_ptr.dtype.element_ty).to(tl.float32)
        if MODFMA:
            xm = tl.fma(xv, sc1, sh)
        else:
            xm = xv * sc1 + sh
        xb = xm.to(xo_ptr.dtype.element_ty)
        tl.store(xo_ptr + base + offs, xb, mask=m)
        xv = xb.to(tl.float32)
        sq = xv * xv
        q02, q13 = tl.split(tl.reshape(sq, (4, 32, 2, 2)))
        q0, q2 = tl.split(q02)
        q1, q3 = tl.split(q13)
        acc = acc + q0
        acc = acc + q1
        acc = acc + q2
        acc = acc + q3
    t = tl.permute(tl.reshape(acc, (4, 2, 16)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 8)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 4)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    t = tl.permute(tl.reshape(t, (4, 2, 2)), (0, 2, 1))
    a, b = tl.split(t)
    t = a + b
    a, b = tl.split(t)
    wsum = a + b
    t = tl.permute(tl.reshape(wsum, (2, 2)), (1, 0))
    a, b = tl.split(t)
    t = a + b
    a, b = tl.split(t)
    ss = a + b
    r = tl.math.rsqrt(tl.math.div_rn(ss, N.to(tl.float32)) + eps)
    tl.debug_barrier()
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        xv = tl.load(xo_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=m, other=0.0)
        o = (wv * (r * xv)).to(h_ptr.dtype.element_ty).to(tl.float32)
        tl.store(h_ptr + base + offs, (o * s).to(h_ptr.dtype.element_ty), mask=m)


def _tail_args(tail):
    """Kernel pointer args and constexprs of a ``layers.MoETail``."""
    sh = tail.shared
    seg = tail.seg if tail.seg is not None else sh
    csr = tail.csr if tail.csr is not None else sh
    pcsr = tail.pcsr if tail.pcsr is not None else sh
    g = tail.g if tail.g is not None else sh
    w = tail.w if tail.w is not None else sh
    ptrs = (tail.expert, tail.inverse, tail.gate, sh, seg, csr, pcsr, g, w)
    e = 1 if tail.csr is None else tail.csr.numel() - 1
    if tail.mode == 2:
        search = (e + 1).bit_length()
    elif tail.mode == 3:
        search = (2 * e + 1).bit_length()
    else:
        search = 1
    return ptrs, dict(MODE=tail.mode, SCALE=float(tail.scaling), E=e, SEARCH=search)


def tail_add_norm_scale(x: Tensor, tail, norm, s: Tensor) -> tuple[Tensor, Tensor]:
    """``add_norm_scale(x, tail.materialize(), norm, s)`` with the tail computed in the kernel (same bits)."""
    n = x.shape[-1]
    rows = x.numel() // n
    xc = x.contiguous()
    xo = torch.empty_like(xc)
    h = torch.empty_like(xc)
    if rows:
        ptrs, kw = _tail_args(tail)
        _tail_add_norm_scale_kernel[(rows,)](
            xc,
            xo,
            h,
            _w1(norm),
            *ptrs,
            n,
            _scale(s),
            float(norm.eps),
            NI=triton.cdiv(n, 512),
            **kw,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return xo, h


def tail_add_mod_norm_scale(
    x: Tensor, tail, mod: Tensor, norm, s: Tensor, modfma: bool = True, geom=None
):
    """``add_mod_norm_scale(x, tail.materialize(), mod, norm, s)`` with the tail computed in the kernel
    (``geom``: compact modulation rows, MOD_COMPACT)."""
    n = x.shape[-1]
    rows = x.numel() // n
    xc = x.contiguous()
    xo = torch.empty_like(xc)
    h = torch.empty_like(xc)
    if rows:
        ptrs, kw = _tail_args(tail)
        _tail_add_mod_norm_scale_kernel[(rows,)](
            xc,
            mod,
            xo,
            h,
            _w1(norm),
            *ptrs,
            n,
            _scale(s),
            float(norm.eps),
            *mod_geom(geom),
            NI=triton.cdiv(n, 512),
            MODFMA=modfma,
            **kw,
            CMOD=geom is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return xo, h


def tail_supported(x: Tensor, tail) -> bool:
    """The tails the kernels cover: rows of ``x`` = tokens of the tail, width = expert row width (top-8)."""
    return (
        tail.shared.shape[0] * tail.shared.shape[1] == x.numel()
        and tail.shared.shape[1] == x.shape[-1]
        and tail.expert.shape[1] == x.shape[-1]
        and tail.shared.dtype == x.dtype == torch.bfloat16
        and tail.expert.dtype == torch.bfloat16
        and tail.inverse.numel() >= 8 * tail.shared.shape[0]
    )


@triton.jit
def _tail_residual_add_kernel(
    x_ptr,
    xo_ptr,
    EX,
    INV,
    GS,
    SH,
    SEG,
    CSR,
    PCSR,
    G,
    W,
    N,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
):
    """``x + y`` (the eager bf16 add: float32 sum of the widened inputs, rounded) with ``y`` from ``_tail_y``."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    xv = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    yv = _tail_y(
        row, offs, m, N, EX, INV, GS, SH, SEG, CSR, PCSR, G, W, MODE, SCALE, E, SEARCH
    )
    tl.store(xo_ptr + row * N + offs, (xv + yv).to(xo_ptr.dtype.element_ty), mask=m)


def tail_residual_add(x: Tensor, tail) -> Tensor:
    """``x + tail.materialize()`` (the block's non-deferred residual add) with the tail computed in the kernel."""
    n = x.shape[-1]
    rows = x.numel() // n
    xc = x.contiguous()
    xo = torch.empty_like(xc)
    if rows:
        ptrs, kw = _tail_args(tail)
        _tail_residual_add_kernel[(rows, triton.cdiv(n, 512))](
            xc,
            xo,
            *ptrs,
            n,
            BLOCK=512,
            **kw,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return xo
