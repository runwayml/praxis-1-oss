"""The FP4 gate|up GEMM with the rescale + SwiGLU + row amax in its epilogue (CuTe DSL part).

See praxis/kernels/nvfp4_epi.py for the contract. This module imports cutlass / quack at import time; import it
lazily (only on SM 11.0 with ``nvfp4_sm110.EPI_SWIGLU``).

quack 0.6.5's epilogue-mod frontend (``EpiMod.gemm`` / ``gemm_act``) rejects block-scaled operands with varlen_m
("does not support varlen/gather yet") and block-scaled operands with ``concat_layout`` (the B interleave view is not
applied to SFB). The kernel underneath (GemmSm100 + the mod's minted epilogue mixin) supports both block-scaled
varlen_m (it is what ``quack.gemm.gemm`` runs for ``grouped_gemm_nvfp4``) and the paired epilogue, so this module
mints the mod's kernel class and drives quack's generic host layer (``build_gemm_epi_plan`` /
``run_gemm_epi_plan``) itself, the same calls ``EpiMod.gemm`` makes after its checks. The gate|up weight bank is
passed already interleaved (row 2j = gate j, row 2j+1 = up j, codes and block scales; see
``nvfp4_epi.set_gate_up_interleaved``), which is what ``mode="acc_pair"`` pairs.
"""

from __future__ import annotations

import cutlass
import torch
from cutlass import Float32
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import unpack
from quack.epilogue.ops import ColVecReduce

from praxis.kernels.cute_act import _silu_triton


def _bf(v):
    """Round a Float32 to bf16 (RN) and back."""
    return v.to(cutlass.BFloat16).to(Float32)


@gemm_epilogue(
    outputs=("hidden",),
    reduces={"amax": ColVecReduce("amax", combine="max_abs")},
    mode="acc_pair",
)
def rescale_swiglu_amax_epi(acc, mult):
    """acc: (gate, up) FP32 accumulators of one hidden column; mult: the row's float(bf16(g_a * g_w)).

    gu = bf16(acc) (the plain GEMM's output); gate_s / up_s = bf16(gu * mult) (_rescale_swiglu_amax_*);
    hidden = bf16(bf16(gate_s * sigmoid(gate_s)) * up_s) with Triton's sigmoid 1 / (1 + ex2.approx(-g log2 e))
    (IEEE division, branch-free: praxis.kernels.cute_act._silu_triton); amax = max |hidden| per row."""
    gate, up = unpack(acc)
    g = _bf(_bf(gate) * mult)
    u = _bf(_bf(up) * mult)
    h = _bf(_bf(_silu_triton(g)) * u)
    return {"hidden": h, "amax": (h, h)}


# One live plan per (K, N, E, device, tile config); quack plans are pure metadata (M is symbolic in the kernel).
_PLANS: dict = {}


def _plan(aq, bq, mult, hidden, partial, sfa, sfb, tile):
    from quack.cute_dsl_utils import get_device_capacity
    from quack.gemm_runtime.host import build_gemm_epi_plan
    from quack.gemm_tvm_ffi_utils import (
        resolve_blockscaled_formats,
        validate_blockscaled_sf,
    )
    from quack.rounding import RoundingMode

    tile_m, tile_n, cluster_m, cluster_n = tile
    key = (tuple(aq.shape[1:]), tuple(bq.shape), tuple(sfb.shape), aq.device, tile)
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    cap = get_device_capacity(aq.device)
    fmt_a, fmt_b = resolve_blockscaled_formats("nvfp4", "nvfp4")
    sf_dtype, sf_vec = validate_blockscaled_sf(
        aq, bq, sfa, sfb, cap, num_batches=bq.shape[0], fmt_a=fmt_a, fmt_b=fmt_b
    )
    mod = rescale_swiglu_amax_epi
    # EpiMod.gemm's mint key for: mult a varlen colvec, paired accumulator, no C, RN, no pinned arg forms.
    mint_key = ((("mult", "col"),), cap[0], True, False, (), RoundingMode.RN, (), False)
    cls = mod._mint(*mint_key)
    plan = build_gemm_epi_plan(
        cls,
        cap,
        aq,
        bq,
        None,
        None,
        epi_values={"mult": mult, "hidden": hidden, "amax": partial},
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=cluster_m,
        cluster_N=cluster_n,
        persistent=True,
        varlen_m=True,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec,
        sf_batched=True,
        a_mma_dtype=fmt_a.to_cutlass_dtype(),
        b_mma_dtype=fmt_b.to_cutlass_dtype(),
        gemm_cls_ref=mod._class_ref(mint_key),
    )
    _PLANS[key] = plan
    return plan


def gate_up_swiglu_amax(
    aq, asf, bq_il, bsf_il, padded_csr, mult, tile=(128, 128, 1, 1)
):
    """(hidden bf16 (P, H), row amax fp32 (P,)) of the block-scaled varlen_m GEMM aq x bq_il^T with the fused epilogue.

    aq uint8 (P, K/2), asf uint8 (sf_rows_q, K/16) (quack placement, as for grouped_gemm_nvfp4); bq_il uint8
    (E, 2H, K/2) and bsf_il uint8 (E, 2H, K/16) the INTERLEAVED gate|up bank; padded_csr (E+1,) int32 offsets;
    mult fp32 (P,) per-row multipliers."""
    from quack.cute_dsl_utils import get_device_capacity
    from quack.gemm_config import cta_tile_shape_m
    from quack.gemm_runtime.host import run_gemm_epi_plan

    e, n, k2 = bq_il.shape
    k = 2 * k2
    rows = aq.shape[0]
    h = n // 2
    hidden = torch.empty((rows, h), dtype=torch.bfloat16, device=aq.device)
    if rows == 0:
        return hidden, torch.zeros(0, dtype=torch.float32, device=aq.device)
    cta_m = cta_tile_shape_m(tile[0], tile[2], get_device_capacity(aq.device)[0], True)
    partial = torch.empty(
        ColVecReduce("amax", combine="max_abs").sink_alloc_shape(
            (rows,), n, cta_m, tile[1]
        ),
        dtype=torch.float32,
        device=aq.device,
    )
    a = aq.view(torch.float4_e2m1fn_x2)
    b = bq_il.view(torch.float4_e2m1fn_x2)
    sfa = asf.view(torch.float8_e4m3fn).view(1, asf.shape[0] // 128, k // 64, 32, 4, 4)
    sfb = bsf_il.view(torch.float8_e4m3fn).view(
        e, bsf_il.shape[1] // 128, k // 64, 32, 4, 4
    )
    plan = _plan(a, b, mult, hidden, partial, sfa, sfb, tuple(tile))
    epi = {"mult": mult, "hidden": hidden, "amax": partial}
    # The generic host layer types SFA's batch mode with B's (symbolic l); under varlen_m the kernel never reads
    # it (GemmSm100 rebuilds the SFA layout from the rm/rk strides only), so pass the single M-padded buffer
    # broadcast over the experts (batch stride 0).
    sfa_l = sfa.expand(e, *sfa.shape[1:])
    run_gemm_epi_plan(
        plan,
        a,
        b,
        None,
        None,
        epi,
        cu_seqlens_m=padded_csr.to(torch.int32),
        SFA=sfa_l,
        SFB=sfb,
    )
    return hidden, partial.amax(dim=-1)
