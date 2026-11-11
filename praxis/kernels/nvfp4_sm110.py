"""NVFP4 routed-expert grouped GEMMs on SM 11.0 (Jetson Thor).

The SM120 path (praxis/kernels/nvfp4.py + FlashInfer ``group_gemm_nvfp4_nt_groupwise``) builds only for SM 12.x.
This module keeps every rounding of that path and swaps only the GEMM:

* weights: the SM120 path's packed banks unchanged (FP4 codes, 128x4-swizzled E4M3 block scales, FP32 globals);
* activations: the SM120 path's ``_quantize`` math unchanged (per-expert global amax/2688, block16 E4M3 scales,
  FP4 codes, rows padded per expert to multiples of 4); only the placement of the scale rows differs, because
  quack's varlen_m kernel reads expert ``i``'s scale rows at tile ``pcsr[i] // 128 + i`` (FlashInfer:
  ``(pcsr[i] + 127 * i) // 128``);
* GEMM: quack's CuTe DSL tcgen05 block-scaled varlen_m grouped GEMM (NVFP4, unit alpha, BF16 out), the
  same NT problem as FlashInfer's: out[r] = sum_k a[r, k] * b[e(r), n, k] with FP32 accumulation;
* rescale / SwiGLU / combine: the SM120 path's kernels unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from praxis.hardware import require_thor
from praxis.kernels.nvfp4 import _expert, _pad_csr, expert_global_scales

# quack tile / cluster for the grouped GEMMs (both pass on Thor; tuning later).
TILE_M, TILE_N, CLUSTER_M, CLUSTER_N = 128, 128, 1, 1


@triton.jit
def _e2m1_pack(norm, R: tl.constexpr, C: tl.constexpr, NATIVE: tl.constexpr):
    """Packed FP4 (E2M1) codes, two per byte (even column in the low nibble), of ``norm`` (R, C) in [-6, 6] / NaN.

    NATIVE False: the SM120 path's threshold sequence (round to nearest even; sign from the float's sign bit, so
    -0 and small negatives give 0x8; NaN gives a signed zero). NATIVE True: the hardware conversion
    (cvt.rn.satfinite.e2m1x2.f32, also round to nearest even with the sign kept), NaN first mapped to the
    signed zero the thresholds produce. Same codes for every float32 input (checked exhaustively)."""
    if NATIVE:
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
    else:
        magnitude = tl.abs(norm)
        code = (
            (magnitude > 0.25).to(tl.uint8)
            + (magnitude >= 0.75).to(tl.uint8)
            + (magnitude > 1.25).to(tl.uint8)
            + (magnitude >= 1.75).to(tl.uint8)
            + (magnitude > 2.5).to(tl.uint8)
            + (magnitude >= 3.5).to(tl.uint8)
            + (magnitude > 5.0).to(tl.uint8)
        )
        code = code | ((norm.to(tl.uint32, bitcast=True) >> 28).to(tl.uint8) & 8)
        packed = tl.sum(
            code.reshape(R, C // 2, 2).to(tl.uint32)
            << (4 * tl.arange(0, 2)[None, None, :]),
            axis=2,
        ).to(tl.uint8)
    return packed


@triton.jit
def _quantize_q(
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
    R: tl.constexpr,
    C: tl.constexpr,
    NATIVE: tl.constexpr = False,
):
    # Body identical to praxis.kernels.nvfp4._quantize except ``sfbase`` (quack varlen_m SFA placement).
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
    packed = _e2m1_pack(norm, R, C, NATIVE)
    pcols = tl.program_id(1) * (C // 2) + tl.arange(0, C // 2)
    tl.store(
        Q + (pstart + local)[:, None].to(tl.int64) * (K // 2) + pcols[None, :],
        packed,
        (rows[:, None] < M) & (pcols[None, :] < K // 2),
    )
    scol = tl.program_id(1) * (C // 16) + tl.arange(0, C // 16)
    sfbase = (pstart // 128 + expert) * 128 * (K // 16)
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


# True: the FP4 code / scale buffers start uninitialized and only the padding rows of the
# expert groups (<= 6 rows each) are zeroed (every other row the GEMM stores is written by the quantize kernel; scale rows past a
# group's padded end belong to tile rows whose outputs are never stored). Same bits. Default off.
ZERO_PAD_ONLY = False

# True: the quantize kernels run 64-row x 256-column tiles with 8 warps instead of 8 x 256
# with 4 (Thor probe: bitwise at every config, faster on 63k x 3072 gathered rows).
# The math is per element and per 16-column block, so the tiling does not change any value. Default off.
QUANT_WIDE = False


# Thor (SM110) only, opt-in: the quantize kernels (_quantize_q, nvfp4_pair._quantize_seg) pack the FP4 codes with the native
# E2M1 conversion (cvt.rn.satfinite.e2m1x2.f32) instead of the 7-threshold sequence (the conversion the native SM100/SM120 packing kernel already uses). Same codes
# for every float32 input (checked exhaustively). NATIVE_E2M1_AUDIT = a JSON path (true: outputs/native_e2m1_audit.json):
# every quantize also runs both packings into zeroed buffers, continues with the threshold result and writes the
# code / scale agreement at exit (probe only). Default off.
NATIVE_E2M1 = False
NATIVE_E2M1_AUDIT = ""

# True: with FUSED_NVFP4 / PAIR_NVFP4 the gate|up FP4 GEMM applies the rescale, SwiGLU and
# hidden row amax in its epilogue (praxis/kernels/nvfp4_epi.py); the (P, 2H) gate|up tensor is never stored. The
# gate|up bank is used interleaved (converted in place on first use). Same bits. Default off.
EPI_SWIGLU = False


def quant_launch() -> tuple[int, int]:
    """(rows per program, warps) of the quantize kernels."""
    return (64, 8) if QUANT_WIDE else (8, 4)


@triton.jit
def _zero_padding(
    Q, SF, CSR, PCSR, K: tl.constexpr, BQ: tl.constexpr, BS: tl.constexpr
):
    e = tl.program_id(0)
    n = tl.load(CSR + e + 1) - tl.load(CSR + e)
    ps = tl.load(PCSR + e)
    npad = tl.load(PCSR + e + 1) - ps - n
    sfbase = (ps // 128 + e) * 128 * (K // 16)
    # pcsr = 4 * floor((csr + 3 i) / 4): a group's padding is pcsr[e+1] - pcsr[e] - n <= 6 rows.
    for j in tl.static_range(7):
        local = n + j
        if j < npad:
            for c0 in tl.static_range(0, K // 2, BQ):
                cols = c0 + tl.arange(0, BQ)
                tl.store(
                    Q + (ps + local).to(tl.int64) * (K // 2) + cols,
                    tl.zeros((BQ,), tl.uint8),
                    cols < K // 2,
                )
            for s0 in tl.static_range(0, K // 16, BS):
                scol = s0 + tl.arange(0, BS)
                off = sfbase + (
                    (local // 128 * (K // 64) + scol // 4) * 512
                    + (local % 32) * 16
                    + ((local % 128) // 32) * 4
                    + scol % 4
                )
                tl.store(SF + off, tl.zeros((BS,), tl.uint8), scol < K // 16)


def alloc_qsf(
    padded_rows: int, k: int, e: int, csr, pcsr, device, all_rows_written: bool
):
    """The FP4 code and scale buffers of one quantize: zeroed, or (ZERO_PAD_ONLY) uninitialized with the
    padding rows zeroed."""
    shape_q, shape_sf = (padded_rows, k // 2), (sf_rows_q(padded_rows, e), k // 16)
    if not ZERO_PAD_ONLY:
        return (
            torch.zeros(shape_q, dtype=torch.uint8, device=device),
            torch.zeros(shape_sf, dtype=torch.uint8, device=device),
        )
    q = torch.empty(shape_q, dtype=torch.uint8, device=device)
    sf = torch.empty(shape_sf, dtype=torch.uint8, device=device)
    if not all_rows_written and e:
        _zero_padding[(e,)](q, sf, csr, pcsr, k, 1024, 256, num_warps=4)
    return q, sf


_E2M1_AUDIT: dict = {}


def native_e2m1_launch(launch, q, sf, key: str) -> None:
    """``launch(q, sf, NATIVE_E2M1)``; with NATIVE_E2M1_AUDIT, the threshold packing into q / sf plus both
    packings into zeroed buffers, compared (codes and scales, bitwise)."""
    if not NATIVE_E2M1_AUDIT:
        launch(q, sf, NATIVE_E2M1)
        return
    import atexit

    launch(q, sf, False)
    qa, sfa = torch.zeros_like(q), torch.zeros_like(sf)
    qb, sfb = torch.zeros_like(q), torch.zeros_like(sf)
    launch(qa, sfa, False)
    launch(qb, sfb, True)
    if not _E2M1_AUDIT:
        atexit.register(_e2m1_audit_dump)
    a = _E2M1_AUDIT.setdefault(
        key,
        {
            "calls": 0,
            "code_bytes": 0,
            "code_bad": 0,
            "sf_bytes": 0,
            "sf_bad": 0,
            "nonzero_code_bytes": 0,
        },
    )
    a["calls"] += 1
    a["code_bytes"] += q.numel()
    a["sf_bytes"] += sf.numel()
    a["code_bad"] += int((qa != qb).sum())
    a["sf_bad"] += int((sfa != sfb).sum())
    a["nonzero_code_bytes"] += int((qa != 0).sum())


def _e2m1_audit_dump() -> None:
    import json
    import os

    tot = {
        k: sum(a[k] for a in _E2M1_AUDIT.values())
        for k in (
            "calls",
            "code_bytes",
            "code_bad",
            "sf_bytes",
            "sf_bad",
            "nonzero_code_bytes",
        )
    }
    tot["exact_all"] = tot["code_bad"] == 0 and tot["sf_bad"] == 0 and tot["calls"] > 0
    path = (
        NATIVE_E2M1_AUDIT
        if isinstance(NATIVE_E2M1_AUDIT, str)
        else "outputs/native_e2m1_audit.json"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"overall": tot, "shapes": _E2M1_AUDIT}, fh, indent=1, sort_keys=True)


def sf_rows_q(padded_rows: int, e: int) -> int:
    """Rows of the quack varlen_m SFA buffer (>= quack's minimum ceil(total/128) + e - 1 tiles)."""
    return (triton.cdiv(padded_rows, 128) + e) * 128


def quantize_activations_q(
    x, csr, *, row_indices=None, padded_csr=None, padded_rows=None, g=None
):
    """praxis.kernels.nvfp4.quantize_activations with quack's SFA placement (same codes, same scale values).
    ``g``: the per-expert globals when the caller already has them (see praxis.kernels.nvfp4_fused)."""
    require_thor(x.device)
    e = csr.numel() - 1
    m = x.shape[0] if row_indices is None else row_indices.numel()
    k = x.shape[1]
    if padded_csr is None:
        padded_rows = ((m + 3 * e) // 4) * 4
        padded_csr = torch.empty_like(csr)
        _pad_csr[(1,)](csr, padded_csr, e, triton.next_power_of_2(e + 1), 4)
    if g is None:
        g = expert_global_scales(x, csr, row_indices)
    q, sf = alloc_qsf(
        padded_rows,
        k,
        e,
        csr,
        padded_csr,
        x.device,
        all_rows_written=m == padded_rows and row_indices is None,
    )
    if m:
        rp, nw = quant_launch()

        def launch(qb, sfb, native):
            _quantize_q[(triton.cdiv(m, rp), triton.cdiv(k, 256))](
                x,
                qb,
                sfb,
                g,
                csr,
                padded_csr,
                csr if row_indices is None else row_indices,
                m,
                k,
                e,
                (e + 1).bit_length(),
                row_indices is not None,
                rp,
                256,
                native,
                num_warps=nw,
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )

        native_e2m1_launch(launch, q, sf, f"q.K{k}")
    return q, sf, g, padded_csr


def grouped_gemm_nvfp4(aq, asf, bq, bsf, padded_csr):
    """BF16 [padded_rows, N] = A_q[padded_rows, K] x B_q[e(row), N, K]^T with NVFP4 block scales, unit alpha.

    aq: uint8 [padded_rows, K/2]; asf: uint8 [sf_rows_q, K/16] (quack placement); bq: uint8 [E, N, K/2];
    bsf: uint8 [E, N_pad, K/16] (128x4 swizzled per expert, SM120 packing); padded_csr: [E+1] row offsets.
    """
    require_thor(aq.device)
    from quack.gemm import gemm

    e, n, k2 = bq.shape
    k = 2 * k2
    rows = aq.shape[0]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=aq.device)
    if rows == 0:
        return out
    sfa = asf.view(torch.float8_e4m3fn).view(1, asf.shape[0] // 128, k // 64, 32, 4, 4)
    sfb = bsf.view(torch.float8_e4m3fn).view(e, bsf.shape[1] // 128, k // 64, 32, 4, 4)
    gemm(
        aq.view(torch.float4_e2m1fn_x2),
        bq.view(torch.float4_e2m1fn_x2),
        out,
        None,
        None,
        tile_M=TILE_M,
        tile_N=TILE_N,
        cluster_M=CLUSTER_M,
        cluster_N=CLUSTER_N,
        cu_seqlens_m=padded_csr.to(torch.int32),
        SFA=sfa,
        SFB=sfb,
        bs_format_a="nvfp4",
        bs_format_b="nvfp4",
    )
    return out
