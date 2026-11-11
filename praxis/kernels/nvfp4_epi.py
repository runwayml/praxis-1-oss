"""``nvfp4_sm110.EPI_SWIGLU``: rescale + SwiGLU + hidden row amax in the FP4 gate|up GEMM epilogue.

Today (FUSED_NVFP4 / PAIR_NVFP4 on SM 11.0) the gate|up step is two kernels:

    gu = grouped_gemm_nvfp4(...)                        bf16 (P, 2H) = bf16(fp32 accumulator)
    _rescale_swiglu_amax_{kernel,seg}(gu)               per row p: m = float(bf16(g_a[seg(p)] * g_w[expert(p)]))
                                                        gate_s = bf16(gu_gate * m); up_s = bf16(gu_up * m)
                                                        hidden = bf16(bf16(gate_s * sigmoid(gate_s)) * up_s)
                                                        amax[p] = max |hidden[p]|

With EPI_SWIGLU the GEMM's epilogue takes the fp32 accumulator and applies exactly these roundings
(praxis/kernels/nvfp4_epi_cute.py): the (P, 2H) gu tensor is never written or read back. The per-row multiplier
is computed by a tiny Triton kernel with the expression (and compile flags) of the old kernel; the row amax comes
from the epilogue as per-(row, N-tile) partial maxima of the exact bf16 magnitudes, folded by ``amax(-1)``.

The paired epilogue pairs adjacent accumulator columns, so the gate|up bank is used INTERLEAVED (row 2j = gate j,
row 2j + 1 = up j, FP4 codes and E4M3 block scales permuted together; quack's ``concat_layout`` view cannot be used
because it does not permute SFB). ``sync_gate_up_layout`` converts a bank in place, one expert at a time, on its
first call in the other mode (and back when the flag is turned off), so every path sees the layout it reads. A
column permutation of B does not change any output element of the GEMM (each element is the dot product of one
A row and one B row with its own block scales).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from praxis.kernels.nvfp4 import _expert
from praxis.kernels.nvfp4_pair import _row_map


@triton.jit
def _row_multiplier_kernel(
    MULT,
    SEG,
    PCSR,
    G,
    W,
    P,
    E: tl.constexpr,
    SEARCH: tl.constexpr,
    PAIRED: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    mask = rows < P
    if PAIRED:
        seg, expert, local = _row_map(SEG, PCSR, PCSR, rows, E, SEARCH, SEARCH, True)
    else:
        expert = _expert(PCSR, rows, E, SEARCH)
        seg = expert
    # The multiplier expression of _rescale_swiglu_amax_kernel / _rescale_swiglu_amax_seg.
    multiplier = (
        (tl.load(G + seg, mask, other=0) * tl.load(W + expert, mask, other=0))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    tl.store(MULT + rows, multiplier, mask)


def row_multipliers(
    padded_csr, activation_globals, weight_globals, padded_rows, seg_csr=None
):
    """fp32 (P,) float(bf16(g_a[seg(p)] * g_w[expert(p)])) of every padded row (segments when ``seg_csr``)."""
    e = padded_csr.numel() - 1
    out = torch.empty(padded_rows, dtype=torch.float32, device=padded_csr.device)
    if padded_rows:
        r = 256
        _row_multiplier_kernel[(triton.cdiv(padded_rows, r),)](
            out,
            padded_csr if seg_csr is None else seg_csr,
            padded_csr,
            activation_globals,
            weight_globals,
            padded_rows,
            e,
            (e + 1).bit_length(),
            seg_csr is not None,
            r,
            num_warps=4,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return out


# ---- gate|up bank layout ----------------------------------------------------------------------------------------
def _perm(n, device, to_interleaved):
    j = torch.arange(n, device=device)
    h = n // 2
    if to_interleaved:  # new row j <- old (concat) row
        return (j % 2) * h + j // 2
    return torch.where(
        j < h, 2 * j, 2 * (j - h) + 1
    )  # new (concat) row j <- old (interleaved) row


def _unblock(sf):
    """(Npad, K/16) 128x4-blocked scale bytes (torchao to_blocked order) -> row-major (Npad, K/16)."""
    n, k16 = sf.shape
    return sf.view(n // 128, k16 // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(n, k16)


def _block(lin):
    n, k16 = lin.shape
    return lin.view(n // 128, 4, 32, k16 // 4, 4).permute(0, 3, 2, 1, 4).reshape(n, k16)


@torch.no_grad()
def permute_gate_up(q, sf, to_interleaved):
    """Permute the gate|up rows of an (E, 2H, K/2) code bank and its (E, 2H, K/16) blocked scales in place."""
    e, n, _ = q.shape
    assert n % 256 == 0 and sf.shape[1] == n, (
        "interleave needs 2H a multiple of 256 and unpadded scale rows"
    )
    p = _perm(n, q.device, to_interleaved)
    for i in range(e):
        q[i].copy_(q[i].index_select(0, p))
        sf[i].copy_(_block(_unblock(sf[i]).index_select(0, p)))


def sync_gate_up_layout(bank, interleaved: bool) -> None:
    """Make ``bank.gate_up_q`` / ``bank.gate_up_sf`` interleaved (fused epilogue) or concat [gate | up] (every other
    path), converting in place if needed."""
    if getattr(bank, "_gate_up_interleaved", False) == interleaved:
        return
    permute_gate_up(bank.gate_up_q, bank.gate_up_sf, interleaved)
    bank._gate_up_interleaved = interleaved


def gate_up_swiglu_amax(bank, aq, asf, padded_csr, activation_globals, seg_csr=None):
    """(hidden bf16 (P, H), row amax fp32 (P,)) = the old ``grouped_gemm_nvfp4`` + ``_rescale_swiglu_amax_*`` pair.
    ``bank`` must be interleaved (sync_gate_up_layout(bank, True))."""
    from praxis.kernels import nvfp4_epi_cute as C
    from praxis.kernels import nvfp4_sm110 as NS

    assert getattr(bank, "_gate_up_interleaved", False), (
        "gate|up bank is not interleaved"
    )
    mult = row_multipliers(
        padded_csr, activation_globals, bank.gate_up_global, aq.shape[0], seg_csr
    )
    return C.gate_up_swiglu_amax(
        aq,
        asf,
        bank.gate_up_q,
        bank.gate_up_sf,
        padded_csr,
        mult,
        (NS.TILE_M, NS.TILE_N, NS.CLUSTER_M, NS.CLUSTER_N),
    )
