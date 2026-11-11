"""NVFP4 activation packing and BF16 rescaling for Blackwell grouped GEMMs.

The block16 FP4 codes and swizzled E4M3 scales preserve the validated
single-device packing recipe. The routed reduction uses Praxis FP32 products.
This optional module is imported only by quantized inference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Fixed wider tiles for the portable native Blackwell packing kernel.
# Thor retains its independent scale layout and QUANT_WIDE option.
QUANT_WIDE = True


def _quantize_launch() -> tuple[int, int]:
    return (64, 8) if QUANT_WIDE else (8, 4)


@triton.jit
def _expert(CSR, rows, E: tl.constexpr, SEARCH: tl.constexpr):
    lo = tl.full(rows.shape, 0, tl.int32)
    hi = tl.full(rows.shape, E, tl.int32)
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        end = tl.load(CSR + mid + 1, mid < E, other=2147483647)
        right = (mid < E) & (end <= rows)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    return tl.minimum(lo, E - 1)


@triton.jit
def _pad_csr(CSR, OUT, E: tl.constexpr, B: tl.constexpr, ALIGN: tl.constexpr):
    i = tl.arange(0, B)
    value = tl.load(CSR + i, i <= E, other=0)
    tl.store(OUT + i, ((value + (ALIGN - 1) * i) // ALIGN) * ALIGN, i <= E)


@triton.jit
def _quantize(
    X,
    Q,
    SF,
    G,
    CSR,
    PCSR,
    MAP,
    M: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    GATHER: tl.constexpr,
    GROUPED_SF: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    expert = _expert(CSR, rows, E, SEARCH)
    start = tl.load(CSR + expert)
    pstart = tl.load(PCSR + expert)
    local = rows - start
    source = rows
    if GATHER:
        source = tl.load(MAP + rows, rows < M, other=0)
    values = tl.load(
        X + source[:, None].to(tl.int64) * K + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < K),
        other=0,
    ).to(tl.float32)
    blocks = values.reshape(R, C // 16, 16)
    block_scale = tl.max(tl.abs(blocks), axis=2) * (1.0 / 6.0)
    g = tl.load(G + expert)
    sf = tl.minimum(
        tl.maximum(libdevice.div_rn(block_scale, g[:, None]), 0.015625), 448.0
    ).to(tl.float8e4nv)
    reciprocal = libdevice.div_rn(libdevice.div_rn(1.0, g[:, None]), sf.to(tl.float32))
    norm = tl.minimum(tl.maximum(blocks * reciprocal[:, :, None], -6.0), 6.0).reshape(
        R, C
    )
    # The former threshold sequence encoded NaNs as signed zero.
    signed_zero = (norm.to(tl.uint32, bitcast=True) & 0x80000000).to(
        tl.float32, bitcast=True
    )
    norm = tl.where(norm != norm, signed_zero, norm)
    first, second = tl.split(norm.reshape(R, C // 2, 2))
    packed = tl.inline_asm_elementwise(
        """{
        .reg .b8 byte;
        cvt.rn.satfinite.e2m1x2.f32 byte, $2, $1;
        cvt.u32.u8 $0, byte;
        }""",
        constraints="=r,f,f",
        args=(first, second),
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    pcols = tl.program_id(1) * (C // 2) + tl.arange(0, C // 2)
    tl.store(
        Q + (pstart + local)[:, None].to(tl.int64) * (K // 2) + pcols[None, :],
        packed,
        (rows[:, None] < M) & (pcols[None, :] < K // 2),
    )
    scol = tl.program_id(1) * (C // 16) + tl.arange(0, C // 16)
    sfbase = pstart * (K // 16)
    if GROUPED_SF:
        sfbase = ((pstart + 127 * expert) // 128) * 128 * (K // 16)
    offset = sfbase[:, None] + (
        (local[:, None] // 128 * (K // 64) + scol[None, :] // 4) * 512
        + (local[:, None] % 32) * 16
        + ((local[:, None] % 128) // 32) * 4
        + scol[None, :] % 4
    )
    tl.store(
        SF + offset,
        sf.to(tl.uint8, bitcast=True),
        (rows[:, None] < M) & (scol[None, :] < K // 16),
    )


@triton.jit
def _scale_unpad(
    X,
    Y,
    CSR,
    PCSR,
    G,
    W,
    M: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    B: tl.constexpr,
):
    pos = tl.program_id(0) * B + tl.arange(0, B)
    row = pos // N
    expert = _expert(CSR, row, E, SEARCH)
    source = tl.load(PCSR + expert) + row - tl.load(CSR + expert)
    multiplier = (
        (tl.load(G + expert) * tl.load(W + expert)).to(tl.bfloat16).to(tl.float32)
    )
    value = tl.load(X + source.to(tl.int64) * N + pos % N, pos < M * N, other=0).to(
        tl.float32
    )
    tl.store(Y + pos, (value * multiplier).to(tl.bfloat16), pos < M * N)


def expert_global_scales(x, csr, row_indices=None):
    """Per-expert NVFP4 global scales of grouped rows, as the QAT forward rounds.

    The QAT training forward of the routed experts uses
    ``amax(rows of the expert) / 2688``: the amax is rounded to the row dtype
    and clamped at 1e-12, and the division is a multiply by the FP32
    reciprocal. Rows past
    the last group end belong to the last expert, as in ``_expert``. All work
    stays on the device: no host reads of offsets or routing.
    """
    if (
        x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] > 0
        and x.is_contiguous()
        and csr.device == x.device
        and csr.dtype in (torch.int32, torch.int64)
        and csr.is_contiguous()
        and (
            row_indices is None
            or (
                row_indices.ndim == 1
                and row_indices.device == x.device
                and row_indices.is_contiguous()
                and row_indices.dtype in (torch.int32, torch.int64)
            )
        )
    ):
        from praxis.kernels.nvfp4_moe import segmented_global_scales

        return segmented_global_scales(x, csr, row_indices)
    e = csr.numel() - 1
    row_amax = x.abs().amax(-1)
    if row_indices is not None:
        row_amax = row_amax[row_indices]
    rows = torch.arange(row_amax.numel(), device=x.device, dtype=csr.dtype)
    expert = torch.bucketize(rows, csr[1:], right=True).clamp_max_(e - 1)
    group_amax = torch.zeros(e, dtype=row_amax.dtype, device=x.device)
    group_amax.scatter_reduce_(0, expert.long(), row_amax, "amax")
    return group_amax.to(x.dtype).clamp_min(1e-12).float() * (1.0 / 2688.0)


def quantize_activations(
    x,
    csr,
    *,
    row_indices=None,
    padded_csr=None,
    padded_rows=None,
    alignment=4,
    grouped_scales=True,
    activation_globals=None,
):
    """Pack BF16 rows without CPU reads of offsets or routing decisions."""
    e = csr.numel() - 1
    m = x.shape[0] if row_indices is None else row_indices.numel()
    k = x.shape[1]
    if padded_csr is None:
        padded_rows = ((m + (alignment - 1) * e) // alignment) * alignment
        padded_csr = torch.empty_like(csr)
        _pad_csr[(1,)](csr, padded_csr, e, triton.next_power_of_2(e + 1), alignment)
    if activation_globals is not None and (
        activation_globals.shape != (e,)
        or activation_globals.dtype != torch.float32
        or activation_globals.device != x.device
    ):
        raise ValueError("Expected one FP32 activation global scale per expert")
    g = (
        expert_global_scales(x, csr, row_indices)
        if activation_globals is None
        else activation_globals
    )
    q = torch.zeros((padded_rows, k // 2), dtype=torch.uint8, device=x.device)
    sf_rows = (
        triton.cdiv(padded_rows + 127 * e, 128) * 128 if grouped_scales else padded_rows
    )
    sf = torch.zeros((sf_rows, k // 16), dtype=torch.uint8, device=x.device)
    if m:
        row_tile, warps = _quantize_launch()
        _quantize[(triton.cdiv(m, row_tile), triton.cdiv(k, 256))](
            x,
            q,
            sf,
            g,
            csr,
            padded_csr,
            csr if row_indices is None else row_indices,
            m,
            k,
            e,
            (e + 1).bit_length(),
            row_indices is not None,
            grouped_scales,
            row_tile,
            256,
            num_warps=warps,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return q, sf, g, padded_csr


def rescale(raw, csr, padded_csr, activation_globals, weight_globals, rows):
    """Round the global-scale product, then the scaled GEMM output, to BF16."""
    out = torch.empty((rows, raw.shape[1]), device=raw.device, dtype=torch.bfloat16)
    e = activation_globals.numel()
    if rows:
        _scale_unpad[(triton.cdiv(out.numel(), 512),)](
            raw,
            out,
            csr,
            padded_csr,
            activation_globals,
            weight_globals,
            rows,
            raw.shape[1],
            e,
            (e + 1).bit_length(),
            512,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out


@triton.jit
def _combine_kernel(
    ROWS,
    INVERSE,
    SCALE,
    OUT,
    T: tl.constexpr,
    C: tl.constexpr,
    TOP_K: tl.constexpr,
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
        value = tl.load(
            ROWS + index[:, None].to(tl.int64) * C + cols[None, :],
            (rows[:, None] < T) & (cols[None, :] < C),
            other=0,
        ).to(tl.float32)
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


def combine_rows(rows, inverse, sorted_scale, tokens, top_k):
    """Praxis chunked reduction: FP32 products and four FP32 accumulators."""
    out = torch.empty((tokens, rows.shape[1]), device=rows.device, dtype=torch.bfloat16)
    if tokens:
        _combine_kernel[(triton.cdiv(tokens, 4), triton.cdiv(rows.shape[1], 256))](
            rows,
            inverse,
            sorted_scale,
            out,
            tokens,
            rows.shape[1],
            top_k,
            4,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _sm100_tile_metadata(
    CSR, EXPERTS, LIMITS, E: tl.constexpr, TILES: tl.constexpr, B: tl.constexpr
):
    tiles = tl.program_id(0) * B + tl.arange(0, B)
    expert = _expert(CSR, tiles * 128, E, (E + 1).bit_length())
    limit = tl.load(CSR + expert + 1)
    tl.store(EXPERTS + tiles, expert, tiles < TILES)
    tl.store(LIMITS + tiles, limit, tiles < TILES)


def grouped_nvfp4_mm(a, b, a_scale, b_scale, csr, alpha, *, metadata=None):
    """Run native NVFP4 grouped GEMMs without host reads or atomic reduction.

    SM100/SM103 uses FlashInfer's deterministic contiguous grouped GEMM.
    Its finalization is an identity mapping; Praxis owns rescaling, SwiGLU,
    and the FP32 routed reduction. SM120 retains the original grouped path.
    """
    major, minor = torch.cuda.get_device_capability(a.device)
    if major == 12 and minor == 0:
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

        return group_gemm_nvfp4_nt_groupwise(
            a,
            b,
            a_scale,
            b_scale,
            csr,
            alpha,
            tile_n=128,
            tile_k=128,
            out_dtype=torch.bfloat16,
        )
    if (major, minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("NVFP4 requires SM100, SM103, or SM120; use BF16 on H100")
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
    )

    rows = a.shape[0]
    if rows % 128:
        raise ValueError("SM100 NVFP4 requires 128-row expert alignment")
    tiles = rows // 128
    if metadata is None:
        experts = torch.empty(tiles, device=a.device, dtype=torch.int32)
        limits = torch.empty_like(experts)
        _sm100_tile_metadata[(triton.cdiv(tiles, 128),)](
            csr,
            experts,
            limits,
            b.shape[0],
            tiles,
            128,
            num_warps=4,
        )
        num_tiles = torch.full((1,), tiles, device=a.device, dtype=torch.int32)
        identity = torch.arange(rows, device=a.device, dtype=torch.int32)
        unit_scale = torch.ones((rows, 1), device=a.device, dtype=torch.float32)
    else:
        experts, limits, num_tiles, identity, unit_scale = metadata
    return blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        experts,
        num_tiles,
        limits,
        identity,
        unit_scale,
        use_fused_finalize=False,
        enable_pdl=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )
