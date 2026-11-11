"""Custom Triton kernel for streamed video-decoder RGB assembly.

The fused decoder path uses this kernel to write each decoded chunk directly
into a preallocated uint8 video. This avoids collecting float chunks,
concatenating a full video and allocating separate unpatchify/RGB buffers.
Input is a batch-one CUDA BCTHW tensor with 12 channels; explicit strides
support both contiguous and channels-last storage.

* ``store_rgb`` expands each pixel's 12 channels into a 2x2 RGB patch and
  writes a (T, 3, 2*H, 2*W) output at the supplied frame offset. Output pixel
  (t, c, y, x) reads input channel ``4*c + 2*(x % 2) + y % 2`` at
  (t, y//2, x//2). Conversion clamps the float32 value to [-1, 1], applies
  ``v * 0.5 + 0.5``, clamps to [0, 1], multiplies by 255 and truncates to
  uint8. The caller supplies output storage; the operation returns no tensor.

The kernel uses fixed 1024-element blocks without autotuning. FP fusion is
disabled to preserve the reference RGB conversion's rounding. The custom
operator declares output mutation so torch.compile retains the write during
graph capture, and its fake implementation returns no tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _store_rgb(
    x,
    y,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    S_C: tl.constexpr,
    S_T: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ow = i % (2 * W)
    oh = i // (2 * W) % (2 * H)
    c = i // (4 * W * H) % 3
    t = i // (12 * W * H)
    channel = c * 4 + (ow % 2) * 2 + oh % 2
    v = tl.load(
        x + channel * S_C + t * S_T + (oh // 2) * S_H + (ow // 2) * S_W, i < N, other=0
    ).to(tl.float32)
    v = tl.minimum(tl.maximum(v, -1.0), 1.0)
    v = v * 0.5 + 0.5
    v = tl.minimum(tl.maximum(v, 0.0), 1.0) * 255.0
    tl.store(y + OFFSET * 12 * H * W + i, v.to(tl.uint8), i < N)


@torch.library.custom_op("praxis_vae::store_rgb", mutates_args=("output",))
def store_rgb(x: torch.Tensor, output: torch.Tensor, offset: int) -> None:
    # Write each temporal chunk into final uint8 storage.
    # This removes all-frame cat/unpatch/float32 video intermediates. Explicit
    # strides preserve both NCDHW and channels-last inputs; no FMA changes.
    assert x.shape[0] == 1 and x.shape[1] == 12
    h, w = x.shape[-2:]
    n = x.numel()
    _store_rgb[(triton.cdiv(n, 1024),)](
        x, output, n, h, w, *x.stride()[1:], offset, 1024, enable_fp_fusion=False
    )


@store_rgb.register_fake
def _(x, output, offset):
    return None
