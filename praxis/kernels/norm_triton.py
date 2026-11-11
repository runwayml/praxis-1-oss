"""Inference kernels that retain the native RMSNorm reduction and rounding."""

import torch
import triton
import triton.language as tl
from torch import Tensor

# ---------------------------------------------------------------------------
# fused residual add + hidden RMSNorm +
# block scale, and the attention gate merge, with the eager bits.
#
# ``add_norm_scale(x, y, norm, s)`` returns ``(x + y, norm(x + y) * s)`` (``y``
# None: ``(x, norm(x) * s)``) with the bits of the eager path of
# :class:`praxis.layers.RMSNorm` (float32 cast, ``F.rms_norm`` with weight
# ``1 + w``, cast back) and the bf16 products/sums around it:
#
# * ``x + y`` is one float32 add of the widened bf16 inputs, rounded to bf16;
# * the norm reproduces ``at::native::vectorized_layer_norm_kernel<float, float,
#   rms>``: one 32 x 4 thread block per row; thread ``t = 32 * warp + lane``
#   reads the float4 vectors ``t, t + 128, ...`` and adds the squares of their 4
#   elements in order to its own sum; a shuffle-down tree (offsets 16, 8, 4, 2,
#   1) adds the lanes of each warp; the warps merge as ``(w0 + w2) + (w1 + w3)``;
#   ``rstd = rsqrt(sum / N + eps)`` with an IEEE division; ``out = gamma * (rstd
#   * x)`` in float32, rounded to bf16 (the eager cast back);
# * ``* s`` is one float32 product of the bf16 norm output and the bf16 scale,
#   rounded to bf16.
#
# Squares of bf16 values are exact in float32, so FMA or not does not matter in
# the sums; FP fusion is disabled anyway. Missing elements (rows not a multiple
# of 512 wide) add +0.0, which leaves a sum of squares unchanged.
#
# ``gate_merge(out, g)`` returns ``(out * g).permute(0, 2, 1, 3).reshape(B, S,
# H * D)``: one bf16 product per element (float32 product rounded to bf16),
# written in the merged-heads layout.
# ---------------------------------------------------------------------------


@triton.jit
def _add_norm_scale_kernel(
    x_ptr,
    y_ptr,
    xo_ptr,
    h_ptr,
    w_ptr,
    hf_ptr,
    N,
    s,
    eps,
    HAS_Y: tl.constexpr,
    NI: tl.constexpr,
    DIV: tl.constexpr,
    F32: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    w4 = tl.arange(0, 4)
    l32 = tl.arange(0, 32)
    e4 = tl.arange(0, 4)
    # element index within one 512-chunk: 4 * (32 * warp + lane) + e
    idx = w4[:, None, None] * 128 + l32[None, :, None] * 4 + e4[None, None, :]
    acc = tl.zeros((4, 32), dtype=tl.float32)
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        xv = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_Y:
            yv = tl.load(y_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
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
    # shuffle-down tree over the 32 lanes of each warp
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
    wsum = a + b  # (4,) one sum per warp
    # warps: offset 2 (w0 += w2, w1 += w3), then offset 1
    t = tl.permute(tl.reshape(wsum, (2, 2)), (1, 0))
    a, b = tl.split(t)
    t = a + b
    a, b = tl.split(t)
    ss = a + b
    if DIV:
        v = (
            tl.math.div_rn(ss, N.to(tl.float32)) + eps
        )  # IEEE division (Triton's '/' is approximate)
    else:
        v = ss * (1.0 / N.to(tl.float32)) + eps
    r = tl.math.rsqrt(v)
    for i in tl.static_range(NI):
        offs = i * 512 + idx
        m = offs < N
        if HAS_Y:
            xv = tl.load(xo_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        else:
            xv = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=m, other=0.0)
        o = (wv * (r * xv)).to(h_ptr.dtype.element_ty).to(tl.float32)
        hb = (o * s).to(h_ptr.dtype.element_ty)
        tl.store(h_ptr + base + offs, hb, mask=m)
        if F32:
            tl.store(hf_ptr + base + offs, hb.to(tl.float32), mask=m)


def _w1(norm) -> Tensor:
    """Reuse RMSNorm's versioned affine cache, also used by RoPE/QK norm."""
    return norm._inference_weight()


def _scale(s: Tensor) -> float:
    """Cache a scalar on its buffer until its version or storage changes."""
    if s.is_inference():
        # Versionless buffers cannot safely cache a mutable host scalar.
        return float(s.float())
    key = (s.data_ptr(), s._version, s.device, s.dtype)
    hit = getattr(s, "_praxis_norm_scale", None)
    if hit is None or hit[0] != key:
        hit = (key, float(s.float()))
        s._praxis_norm_scale = hit
    return hit[1]


def add_norm_scale(
    x: Tensor,
    y: Tensor | None,
    norm,
    s: Tensor,
    div: bool = True,
    out: Tensor | None = None,
    out_f32: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``(x + y, norm(x + y) * s)`` (or ``(x, norm(x) * s)``) with the eager bits; see the module doc.
    ``out`` (contiguous, the shape of ``x``) receives the normalized output; ``out_f32`` (contiguous
    float32, the shape of ``x``) also receives it widened to float32 (exact)."""
    n = x.shape[-1]
    rows = x.numel() // n
    xc = x.contiguous()
    yc = None if y is None else y.contiguous()
    xo = torch.empty_like(xc) if y is not None else xc
    h = (
        out
        if out is not None and out.is_contiguous() and out.shape == xc.shape
        else torch.empty_like(xc)
    )
    if rows:
        f32 = out_f32 is not None
        assert not f32 or (
            out_f32.is_contiguous()
            and out_f32.dtype == torch.float32
            and out_f32.shape == xc.shape
        )
        _add_norm_scale_kernel[(rows,)](
            xc,
            xc if yc is None else yc,
            xo,
            h,
            _w1(norm),
            out_f32 if f32 else h,
            n,
            _scale(s),
            float(norm.eps),
            HAS_Y=y is not None,
            NI=triton.cdiv(n, 512),
            DIV=div,
            F32=f32,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return xo, h


def supported(x: Tensor, norm, s: Tensor) -> bool:
    """The cases the kernel covers: bf16 CUDA rows, float32-cast norm with weight, bf16 0-dim scale."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and getattr(norm, "cast_dtype", None) == torch.float32
        and norm.weight is not None
        and tuple(norm.normalized_shape) == (x.shape[-1],)
        and x.shape[-1] % 4 == 0
        and s.dim() == 0
        and s.dtype == torch.bfloat16
        and not s.is_inference()
    )


@triton.jit
def _gate_merge_kernel(
    o_ptr,
    g_ptr,
    y_ptr,
    H,
    S,
    sob,
    soh,
    sos,
    sod,
    sgb,
    sgh,
    sgs,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // H
    hh = bh % H
    sr = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    m = (sr < S)[:, None]
    ov = tl.load(
        o_ptr
        + b.to(tl.int64) * sob
        + hh.to(tl.int64) * soh
        + sr[:, None].to(tl.int64) * sos
        + d[None, :] * sod,
        mask=m,
        other=0.0,
    ).to(tl.float32)
    gv = tl.load(
        g_ptr + b.to(tl.int64) * sgb + hh.to(tl.int64) * sgh + sr.to(tl.int64) * sgs,
        mask=sr < S,
        other=0.0,
    ).to(tl.float32)
    yv = (ov * gv[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr
        + (b.to(tl.int64) * S + sr[:, None].to(tl.int64)) * (H * D)
        + hh * D
        + d[None, :],
        yv,
        mask=m,
    )


def gate_merge(out: Tensor, g: Tensor) -> Tensor:
    """``(out * g).permute(0, 2, 1, 3).reshape(B, S, H * D)`` for ``out (B, H, S, D)``, ``g (B, H, S, 1)``."""
    b, h, s, d = out.shape
    y = torch.empty((b, s, h * d), dtype=out.dtype, device=out.device)
    if y.numel():
        rows = 16
        _gate_merge_kernel[(b * h, triton.cdiv(s, rows))](
            out,
            g,
            y,
            h,
            s,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            D=d,
            ROWS=rows,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y


def gate_merge_supported(out: Tensor, g: Tensor) -> bool:
    return (
        out.is_cuda
        and out.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and out.dim() == 4
        and g.shape == (*out.shape[:3], 1)
        and out.shape[-1] in (64, 128)
        and not torch.is_grad_enabled()
    )


# ---------------------------------------------------------------------------
# the action expert's rms_norm / AdaLN tail.
# ---------------------------------------------------------------------------


@triton.jit
def _rms_post_kernel(
    x_ptr,
    m_ptr,
    a_ptr,
    o_ptr,
    H,
    S,
    eps,
    sxb,
    sxh,
    sxs,
    smb,
    smh,
    sms,
    sab,
    sah,
    sas,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    ADA: tl.constexpr,
):
    bh = tl.program_id(0)
    b = (bh // H).to(tl.int64)
    h = (bh % H).to(tl.int64)
    sr = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    m = (sr < S)[:, None]
    srl = sr.to(tl.int64)
    xo = b * sxb + h * sxh + srl[:, None] * sxs + d[None, :]
    xv = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
    ms = tl.load(m_ptr + b * smb + h * smh + srl * sms, mask=sr < S, other=1.0)
    r = tl.math.rsqrt(ms + eps)
    o = (xv * r[:, None]).to(o_ptr.dtype.element_ty)
    if ADA:
        av = tl.load(
            a_ptr + b * sab + h * sah + srl[:, None] * sas + d[None, :],
            mask=m,
            other=0.0,
        ).to(tl.float32)
        o = ((1.0 + av) * o.to(tl.float32)).to(o_ptr.dtype.element_ty)
    tl.store(o_ptr + xo, o, mask=m)


def rms_post(
    x: Tensor, mean_sq: Tensor, eps: float, ada: Tensor | None = None
) -> Tensor:
    """The eager tail of ``praxis.action_expert.rms_norm`` after its float32 mean of squares:
    ``(x.float() * rsqrt(mean_sq + eps)).to(x.dtype)``, and with ``ada`` the AdaLN product
    ``((1 + ada.float()) * that).to(x.dtype)``, in one kernel with the eager roundings. The output has
    the strides of ``x`` (as the eager ops). ``x`` is 3-D or 4-D with a contiguous last dim."""
    x4 = x if x.dim() == 4 else x.unsqueeze(1)
    m4 = mean_sq if mean_sq.dim() == 4 else mean_sq.unsqueeze(1)
    out = torch.empty_like(x)
    o4 = out if out.dim() == 4 else out.unsqueeze(1)
    a4 = None if ada is None else (ada if ada.dim() == 4 else ada.unsqueeze(1))
    b, h, s, d = x4.shape
    if out.numel():
        rows = max(1, min(64, 4096 // d))
        _rms_post_kernel[(b * h, triton.cdiv(s, rows))](
            x4,
            m4,
            x4 if a4 is None else a4,
            o4,
            h,
            s,
            float(eps),
            x4.stride(0),
            x4.stride(1),
            x4.stride(2),
            m4.stride(0),
            m4.stride(1),
            m4.stride(2),
            *((0, 0, 0) if a4 is None else (a4.stride(0), a4.stride(1), a4.stride(2))),
            D=d,
            ROWS=rows,
            ADA=ada is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def rms_post_supported(x: Tensor, mean_sq: Tensor, ada: Tensor | None = None) -> bool:
    """The cases rms_post covers: half-precision CUDA rows of a power-of-two width with a contiguous last
    dim, a float32 keepdim mean, an output layout equal to ``x``'s."""
    ok = (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dim() in (3, 4)
        and x.stride(-1) == 1
        and mean_sq.dtype == torch.float32
        and mean_sq.shape == (*x.shape[:-1], 1)
        and x.shape[-1] in (64, 128, 256, 512, 1024, 2048)
        and not torch.is_grad_enabled()
        and torch.empty_like(x).stride() == x.stride()
    )
    if ada is not None:
        ok = (
            ok and ada.shape == x.shape and ada.dtype == x.dtype and ada.stride(-1) == 1
        )
    return ok


# ---------------------------------------------------------------------------
# decoder residual add + modulate + ln_in *
# s in one kernel.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOD_COMPACT: compact decoder modulation. The decoder's per-token conditioning is
# ``[zeros(n_reg) | cond upsampled by (ph, pw)]`` (DecoderTower._chunk_causal_prep), so its modulator rows repeat
# ph * pw times. With ``layers.MOD_COMPACT`` the modulator GEMM runs once per distinct row (the n_reg pad rows
# and the cf*h*w cond rows) and the modulate + norm kernels read row ``mod_row(r)`` of that compact (rows_c, 2N)
# tensor: the same values as row ``r`` of the full modulation.
# ---------------------------------------------------------------------------


@triton.jit
def mod_row(row, RF, RC, NREG, PW, WD, PH, CMOD: tl.constexpr):
    """The modulation row of token row ``row`` (int64). Full layout (CMOD False): ``row``. Compact: batch
    ``row // RF`` has RF full rows ``[n_reg | (f, y, i, x, j)]`` (i < PH, j < PW, x < WD) and RC compact rows
    ``[n_reg | (f, y, x)]``."""
    if CMOD:
        b = row // RF
        t = row - b * RF
        u = tl.maximum(t - NREG, 0)
        q = u // PW
        xx = q % WD
        fy = (q // WD) // PH
        c = tl.where(t < NREG, t, NREG + fy * WD + xx)
        return b * RC + c
    else:
        return row


def mod_geom(geom) -> tuple:
    """Kernel ints (RF, RC, NREG, PW, WD, PH) of a ``layers.CompactMod`` geometry (None: full rows)."""
    if geom is None:
        return (1, 1, 0, 1, 1, 1)
    return tuple(int(v) for v in geom)


@triton.jit
def _add_mod_norm_scale_kernel(
    x_ptr,
    y_ptr,
    m_ptr,
    xo_ptr,
    h_ptr,
    w_ptr,
    N,
    s,
    eps,
    RF,
    RC,
    NREG,
    PW,
    WD,
    PH,
    HAS_Y: tl.constexpr,
    NI: tl.constexpr,
    MODFMA: tl.constexpr,
    CMOD: tl.constexpr = False,
):
    """Row ``r``: ``xb = x + y`` (bf16); ``xm = modulate(xb, shift, 1 + scale)`` with shift = m[r, :N],
    scale = m[r, N:2N] (``1 + scale`` rounded to bf16, then ``xb * sc + shift`` in float32, fused multiply-add
    when MODFMA, rounded to bf16, as ``layers_triton._modulate_kernel``); writes xm and ``norm(xm) * s`` with
    the reduction order of ``_add_norm_scale_kernel``. CMOD (MOD_COMPACT): ``m`` holds the compact modulation rows
    (see ``mod_row``)."""
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
        if HAS_Y:
            yv = tl.load(y_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
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


def add_mod_norm_scale(
    x: Tensor,
    y: Tensor | None,
    mod: Tensor,
    norm,
    s: Tensor,
    modfma: bool = True,
    geom=None,
) -> tuple[Tensor, Tensor]:
    """``xm = modulate(x + y, shift, 1 + scale, "mixed")`` with ``shift, scale = mod.chunk(2, -1)``, and
    ``norm(xm) * s``; returns ``(xm, norm(xm) * s)`` with the eager/fused-kernel bits (see the kernel).
    ``geom`` (MOD_COMPACT): ``mod`` holds compact rows (``layers.CompactMod.geom``)."""
    n = x.shape[-1]
    rows = x.numel() // n
    xc = x.contiguous()
    yc = None if y is None else y.contiguous()
    xo = torch.empty_like(xc)
    h = torch.empty_like(xc)
    if rows:
        _add_mod_norm_scale_kernel[(rows,)](
            xc,
            xc if yc is None else yc,
            mod,
            xo,
            h,
            _w1(norm),
            n,
            _scale(s),
            float(norm.eps),
            *mod_geom(geom),
            HAS_Y=y is not None,
            NI=triton.cdiv(n, 512),
            MODFMA=modfma,
            CMOD=geom is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return xo, h


def add_mod_norm_scale_supported(
    x: Tensor, y: Tensor | None, mod: Tensor, norm, s: Tensor, geom=None
) -> bool:
    if geom is None:
        rows_ok = mod.shape[:-1] == x.shape[:-1]
    else:
        rf, rc = int(geom[0]), int(geom[1])
        rows_ok = (
            x.dim() == 3
            and mod.dim() == 3
            and x.shape[1] == rf
            and mod.shape[0] == x.shape[0]
            and mod.shape[1] == rc
        )
    return (
        supported(x, norm, s)
        and mod.is_cuda
        and mod.dtype == x.dtype
        and mod.is_contiguous()
        and rows_ok
        and mod.shape[-1] == 2 * x.shape[-1]
        and (y is None or (y.shape == x.shape and y.dtype == x.dtype))
    )


# Which modulate rounding reproduces layers_triton._modulate_kernel (fused multiply-add or not); set from the
# Modulation rounding probe.
MOD_FMA = True


# ---------------------------------------------------------------------------
# FUSED_KV_SOURCE: the action expert's KV-source tap norm tail in one kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _rms_scale_kernel(
    x_ptr,
    m_ptr,
    w_ptr,
    o_ptr,
    S,
    D,
    eps,
    ROWS: tl.constexpr,
    BD: tl.constexpr,
    NI: tl.constexpr,
):
    """Row r of a contiguous (S, D) ``x``: ``bf16(f32(bf16(x * rsqrt(mean_sq[r] + eps))) * w)`` - the eager
    ``(rms_norm(x) * w).to(x.dtype)`` of ``ActionExpert.forward_trunk_rope``: the float32 product of the widened
    input and the rstd rounded to the input dtype (rms_norm's cast back), then one float32 product with the
    widened weight, rounded to the input dtype (a bf16 x bf16 product is exact in float32, so a bf16 or a
    float32 weight gives the same bits). Elementwise per row: no reduction order to keep."""
    sr = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = sr < S
    srl = sr.to(tl.int64)
    ms = tl.load(m_ptr + srl, mask=rm, other=1.0)
    r = tl.math.rsqrt(ms + eps)
    for i in tl.static_range(NI):
        d = i * BD + tl.arange(0, BD)
        dm = d < D
        m = rm[:, None] & dm[None, :]
        xo = srl[:, None] * D + d[None, :]
        xv = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
        o = (xv * r[:, None]).to(o_ptr.dtype.element_ty).to(tl.float32)
        wv = tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32)
        tl.store(o_ptr + xo, (o * wv[None, :]).to(o_ptr.dtype.element_ty), mask=m)


def rms_scale(x: Tensor, mean_sq: Tensor, eps: float, w: Tensor) -> Tensor:
    """``(rms_norm(x) * w).to(x.dtype)`` from rms_norm's float32 mean of squares, in one kernel (FUSED_KV_SOURCE)."""
    out = torch.empty_like(x)
    d = x.shape[-1]
    s = x.numel() // d
    if s:
        bd = 1024
        rows = 4
        _rms_scale_kernel[(triton.cdiv(s, rows),)](
            x,
            mean_sq,
            w,
            out,
            s,
            d,
            float(eps),
            ROWS=rows,
            BD=bd,
            NI=triton.cdiv(d, bd),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def rms_scale_supported(x: Tensor, mean_sq: Tensor, w: Tensor) -> bool:
    """Contiguous half-precision CUDA rows (width a multiple of 8), a contiguous float32 keepdim mean, a
    contiguous (D,) bf16/float32 weight on the same device."""
    d = x.shape[-1]
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
        and d % 8 == 0
        and d <= 16384
        and not torch.is_grad_enabled()
        and mean_sq.dtype == torch.float32
        and mean_sq.shape == (*x.shape[:-1], 1)
        and mean_sq.is_contiguous()
        and w.device == x.device
        and w.dim() == 1
        and w.shape[0] == d
        and w.is_contiguous()
        and w.dtype in (torch.bfloat16, torch.float32)
    )
