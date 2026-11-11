"""Custom Triton kernel for fused video-decoder normalization and SiLU.

The fused decoder path uses this kernel to replace a channel reduction,
normalization, learned channel scale and SiLU with one launch. Inputs are
strided CUDA BCTHW tensors in float16, bfloat16 or float32 with 256, 512 or
1024 channels. Gamma is a contiguous C-element tensor on the input device.
Multiple batch elements are supported; output uses channels-last-3d storage.

* ``decoder_norm_silu`` computes
  ``y = x / max(sqrt(sum(x*x)), 1e-12) * scale * gamma`` for each pixel,
  then ``y / (1 + exp(-y))``. Reduction and activation use float32 arithmetic.
  Default float32 output preserves normalized activation/history storage;
  optional bfloat16 output reduces storage and memory traffic. Writing the
  convolution's channels-last layout avoids a separate layout conversion.

The kernel does not autotune. Each program reduces all channels for four
pixels at 1024 channels or eight pixels otherwise, with four warps. FP fusion
is disabled to retain separate rounding steps. Reduction order can differ
from PyTorch, so output is not guaranteed to match the reference decoder
bitwise. The custom-op fake implementation exposes output shape, strides and
dtype to torch.compile and CUDA Graph capture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _decoder_norm_silu(
    x,
    gamma,
    out,
    M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    SB: tl.constexpr,
    SC: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    batch = tl.program_id(1)
    channel = tl.arange(0, BC)
    t = pos // (H * W)
    row = pos // W % H
    col = pos % W
    offset = batch * SB + t * ST + row * SH + col * SW
    value = tl.load(
        x + offset[:, None] + channel[None, :] * SC,
        (pos[:, None] < M) & (channel[None, :] < C),
        other=0,
    ).to(tl.float32)
    squared = tl.sum(value * value, axis=1)
    denominator = tl.maximum(libdevice.sqrt_rn(squared), 1.0e-12)
    weight = tl.load(gamma + channel, channel < C, other=0).to(tl.float32)
    normalized = ((value / denominator[:, None]) * SCALE) * weight[None, :] + 0.0
    result = normalized / (1.0 + libdevice.exp(-normalized))
    tl.store(
        out + batch * M * C + pos[:, None] * C + channel[None, :],
        result,
        (pos[:, None] < M) & (channel[None, :] < C),
    )


@torch.library.custom_op("praxis_vae::decoder_norm_silu", mutates_args=())
def decoder_norm_silu(
    x: torch.Tensor, gamma: torch.Tensor, scale: float, bf16_output: bool = False
) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 5 or x.shape[1] not in (256, 512, 1024):
        raise ValueError(
            "Decoder norm expects CUDA BCTHW input with 256/512/1024 channels"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Decoder norm expects floating-point input")
    if (
        gamma.device != x.device
        or gamma.numel() != x.shape[1]
        or not gamma.is_contiguous()
    ):
        raise ValueError(
            "Decoder norm gamma must be contiguous, on the input device, with C elements"
        )
    output = torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.bfloat16 if bf16_output else torch.float32,
        memory_format=torch.channels_last_3d,
    )
    b, c, t, h, w = x.shape
    m = t * h * w
    bm = 4 if c == 1024 else 8
    _decoder_norm_silu[(triton.cdiv(m, bm), b)](
        x,
        gamma,
        output,
        m,
        c,
        h,
        w,
        *x.stride(),
        scale,
        bm,
        triton.next_power_of_2(c),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@decoder_norm_silu.register_fake
def _(x, gamma, scale, bf16_output=False):
    return torch.empty(
        x.shape,
        device=x.device,
        dtype=torch.bfloat16 if bf16_output else torch.float32,
        memory_format=torch.channels_last_3d,
    )
