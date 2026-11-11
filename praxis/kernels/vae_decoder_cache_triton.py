"""Assemble causal Conv3d inputs and next history in channels-last storage.

One memory pass reads strided BCTHW current/history tensors and writes the
causally padded convolution input plus a compact, independent next history.
Spatial padding stays implicit in cuDNN. The input can be cast to convolution
precision while history retains the original activation dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _assemble(
    X,
    Old,
    Input,
    History,
    C: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    CT: tl.constexpr,
    PT: tl.constexpr,
    OT: tl.constexpr,
    NT: tl.constexpr,
    N: tl.constexpr,
    NH: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    XS4: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    CS3: tl.constexpr,
    CS4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % C
    w = (i // C) % W
    h = (i // (C * W)) % H
    t = (i // (C * W * H)) % OT
    b = i // (C * W * H * OT)
    source_t = t - (PT - CT) - CT
    xv = tl.load(
        X + b * XS0 + c * XS1 + source_t * XS2 + h * XS3 + w * XS4,
        (i < N) & (source_t >= 0) & (source_t < T),
        other=0,
    )
    if CT:
        old_t = t - (PT - CT)
        cv = tl.load(
            Old + b * CS0 + c * CS1 + old_t * CS2 + h * CS3 + w * CS4,
            (i < N) & (old_t >= 0) & (old_t < CT),
            other=0,
        )
        value = tl.where(source_t >= 0, xv, cv)
    else:
        value = xv
    tl.store(Input + i, value, i < N)

    # Match _time_cache exactly: two current frames, or the last old frame
    # followed by the sole current frame. Padding never enters history.
    nt = (i // (C * W * H)) % NT
    nb = i // (C * W * H * NT)
    xt = nt + T - NT
    current = tl.load(
        X + nb * XS0 + c * XS1 + xt * XS2 + h * XS3 + w * XS4,
        (i < NH) & (xt >= 0),
        other=0,
    )
    if CT:
        previous = tl.load(
            Old + nb * CS0 + c * CS1 + (CT - 1) * CS2 + h * CS3 + w * CS4,
            (i < NH) & (xt < 0),
            other=0,
        )
        current = tl.where(xt >= 0, current, previous)
    tl.store(History + i, current, i < NH)


@torch.library.custom_op("praxis_vae::assemble_causal_input", mutates_args=())
def assemble_causal_input(
    x: torch.Tensor,
    cache: torch.Tensor | None,
    temporal_padding: int,
    input_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (padded input, next history), with independent CL3D buffers."""
    if not x.is_cuda or x.ndim != 5:
        raise ValueError("Causal assembly requires a CUDA BCTHW tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Causal assembly requires floating-point activations")
    if input_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unsupported convolution input dtype")
    b, c, t, h, w = x.shape
    ct = 0 if cache is None else cache.shape[2]
    if min(b, c, t, h, w) < 1 or temporal_padding < ct:
        raise ValueError("Invalid causal input dimensions or padding")
    if cache is not None and (
        cache.ndim != 5
        or cache.device != x.device
        or cache.dtype != x.dtype
        or cache.shape[:2] != x.shape[:2]
        or cache.shape[3:] != x.shape[3:]
    ):
        raise ValueError("History must match input batch, channels, space and dtype")
    shape = (b, c, t + temporal_padding, h, w)
    nt = min(2, t + (1 if ct else 0))
    output = torch.empty(
        shape, device=x.device, dtype=input_dtype, memory_format=torch.channels_last_3d
    )
    history = torch.empty(
        (b, c, nt, h, w),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last_3d,
    )
    old = x if cache is None else cache
    _assemble[(triton.cdiv(output.numel(), 512),)](
        x,
        old,
        output,
        history,
        c,
        t,
        h,
        w,
        ct,
        temporal_padding,
        shape[2],
        nt,
        output.numel(),
        history.numel(),
        *x.stride(),
        *old.stride(),
        512,
    )
    return output, history


@assemble_causal_input.register_fake
def _(x, cache, temporal_padding, input_dtype):
    b, c, t, h, w = x.shape
    ct = 0 if cache is None else cache.shape[2]
    return (
        torch.empty(
            (b, c, t + temporal_padding, h, w),
            device=x.device,
            dtype=input_dtype,
            memory_format=torch.channels_last_3d,
        ),
        torch.empty(
            (b, c, min(2, t + (1 if ct else 0)), h, w),
            device=x.device,
            dtype=x.dtype,
            memory_format=torch.channels_last_3d,
        ),
    )
