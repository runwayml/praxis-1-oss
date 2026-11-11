"""SM100 FC1 with fused SwiGLU and NVFP4 output quantization.

Output scales are calibrated per bank. This changes QAT rounding and output
bits. Weight copies and scales are runtime state, outside checkpoint buffers.
"""

from __future__ import annotations

import torch


def prepare(gate_up_q, gate_up_sf, global_scale):
    """Copy gate/up weights into FlashInfer's 64-row up/gate order."""
    experts, doubled_hidden, _ = gate_up_q.shape
    hidden = doubled_hidden // 2
    indices = torch.arange(hidden, device=gate_up_q.device).reshape(-1, 64)
    order = torch.stack((indices + hidden, indices), dim=1).flatten()
    k16 = gate_up_sf.shape[-1]
    linear = (
        gate_up_sf.view(experts, doubled_hidden // 128, k16 // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(experts, doubled_hidden, k16)
    )
    scales = (
        linear[:, order]
        .reshape(experts, doubled_hidden // 128, 4, 32, k16 // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape_as(gate_up_sf)
    )
    global_ = torch.tensor([global_scale], device=gate_up_q.device, dtype=torch.float32)
    return (
        gate_up_q[:, order].contiguous(),
        scales,
        global_.reciprocal(),
        global_.expand(experts).contiguous(),
    )


def gate_up_fp4(aq, asf, activation_globals, weight_globals, prepared, metadata):
    """Return packed hidden values, blocked scales and FC2 global scales."""
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
    )

    rows, k2 = aq.shape
    k = k2 * 2
    # The gather GEMM reads row-major input scales. FC2 reads blocked scales.
    gather_sf = (
        asf.view(rows // 128, k // 64, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, k // 16)
    )
    weights, weight_scales, inverse_global, hidden_globals = prepared
    experts, limits, count, identity, _ = metadata
    q, sf = blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
        aq,
        weights,
        gather_sf,
        weight_scales,
        (activation_globals * weight_globals).to(torch.bfloat16).float(),
        experts,
        limits,
        identity,
        count,
        global_scale=inverse_global,
        c_dtype="float4_e2m1fn",
        topk=1,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        enable_pdl=False,
    )
    return q, sf, hidden_globals
