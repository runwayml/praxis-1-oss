"""Custom Triton kernels for the backbone transformer layers.

These kernels implement activation, modulation and mixture-of-experts routing
for inference. The wrappers retain dtype conversion and rounding boundaries
where required by the equivalent reference operations. Routing wrappers fall
back to eager PyTorch for unsupported layouts, dtypes, autocast or gradients.

* ``swiglu_packed`` computes ``silu(gate) * up`` from the two halves of a
  packed CUDA tensor with shape (..., 2*H). It computes SiLU in float32,
  rounds it to the input dtype and multiplies by up, returning (..., H).
* ``modulate_fused`` computes ``x * scale + shift`` in float32 and casts to
  the input dtype. Shift and scale either match (B, S, C) or broadcast from
  (B, 1, C). The wrapper makes inputs contiguous. Multiply-add fusion is
  permitted, so rounding can differ from separate eager operations.
* ``moe_dispatch`` gathers contiguous token rows in the supplied expert sort
  order. It also produces sorted gates, inverse indices and cumulative expert
  offsets, preserving the supplied ordering and the input value dtype.
* ``moe_combine`` weights sorted expert outputs, restores token order and
  sums eight routes. Weighted values round to the expert dtype before four
  interleaved float32 accumulators reproduce the eager reduction order.
* ``moe_scale_add`` scales the routed result, rounds to its dtype and adds
  the shared expert output, retaining the eager intermediate rounding.
* ``moe_chunked_combine`` scales gates and accumulates routed outputs in
  float32. It can round the routed sum and add the shared output in one pass.

The kernels do not autotune. SwiGLU and modulation use 1024-element blocks;
expert dispatch uses row tiles of up to 512 columns and 256-element metadata
blocks. Combination uses 256-column blocks and scale/add uses 1024 elements.
Combination and scale/add disable FP fusion to preserve rounding boundaries.
Launch sizes depend on tensor dimensions rather than timing measurements.

Each kernel has a fixed block size and no autotuning, so the output bits do
not depend on the launch.

Both functions support autograd. The backward passes are the backward passes
of the original training code:

* ``swiglu_packed``: the eager PyTorch backward of ``F.silu(gate) * up``
  (``aten.silu_backward`` and two products in the input dtype).
* ``modulate_fused``: float32 products, rounded to the dtype of each input.
  For ``rows=True`` a Triton kernel sums ``grad`` and ``grad * x`` over the
  tokens in float32, in blocks of ``MODULATE_REDUCE_BLOCK`` tokens.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

SWIGLU_BLOCK = 1024
MODULATE_BLOCK = 1024
MODULATE_REDUCE_BLOCK = 128


@triton.jit
def _swiglu_packed_kernel(
    gu_ptr,
    out_ptr,
    gu_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
) -> None:
    """out = silu(gate) * up over a packed (rows, 2 * n_cols) buffer."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < n_cols
    gu = gu_ptr + row * gu_row_stride
    gate = tl.load(gu + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu + n_cols + col, mask=mask, other=0.0)
    out = (gate * tl.sigmoid(gate)).to(up.dtype) * up
    tl.store(out_ptr + row * out_row_stride + col, out, mask=mask)


@triton.jit
def _modulate_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """out = x * scale + shift in float32, for three tensors of one shape."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    shift_raw = tl.load(shift_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + offs, mask=mask, other=0)
    out = x_raw.to(tl.float32) * scale_raw.to(tl.float32) + shift_raw.to(tl.float32)
    tl.store(out_ptr + offs, out.to(x_raw.dtype), mask=mask)


@triton.jit
def _modulate_rows_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    B,
    S,
    C,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """out = x * scale + shift in float32, x (B, S, C), shift/scale (B, C)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = offs % C
    tmp = offs // C
    b = tmp // S
    bc_idx = b * C + c
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    shift_raw = tl.load(shift_ptr + bc_idx, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + bc_idx, mask=mask, other=0)
    out = x_raw.to(tl.float32) * scale_raw.to(tl.float32) + shift_raw.to(tl.float32)
    tl.store(out_ptr + offs, out.to(x_raw.dtype), mask=mask)


@triton.jit
def _modulate_bwd_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    dx_ptr,
    dshift_ptr,
    dscale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """dx = grad * scale, dshift = grad, dscale = grad * x (float32 products)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    grad_raw = tl.load(grad_ptr + offs, mask=mask, other=0)
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + offs, mask=mask, other=0)
    grad_f = grad_raw.to(tl.float32)
    dx = grad_f * scale_raw.to(tl.float32)
    dscale = grad_f * x_raw.to(tl.float32)
    tl.store(dx_ptr + offs, dx.to(x_raw.dtype), mask=mask)
    tl.store(dshift_ptr + offs, grad_f, mask=mask)
    tl.store(dscale_ptr + offs, dscale, mask=mask)


@triton.jit
def _modulate_rows_bwd_dx_kernel(
    grad_ptr,
    scale_ptr,
    dx_ptr,
    S,
    C,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """dx = grad * scale for x (B, S, C) and scale (B, C), in float32."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = offs % C
    tmp = offs // C
    b = tmp // S
    bc_idx = b * C + c
    grad_raw = tl.load(grad_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + bc_idx, mask=mask, other=0)
    dx = grad_raw.to(tl.float32) * scale_raw.to(tl.float32)
    tl.store(dx_ptr + offs, dx.to(grad_raw.dtype), mask=mask)


@triton.jit
def _modulate_rows_bwd_reduce_kernel(
    grad_ptr,
    x_ptr,
    dshift_ptr,
    dscale_ptr,
    S,
    C,
    BLOCK_S: tl.constexpr,
) -> None:
    """dshift = sum_s grad and dscale = sum_s grad * x, for one (b, c) pair."""
    pid_bc = tl.program_id(0).to(tl.int64)
    b = pid_bc // C
    c = pid_bc % C
    acc_shift = tl.zeros((), dtype=tl.float32)
    acc_scale = tl.zeros((), dtype=tl.float32)
    s_start = 0
    while s_start < S:
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        idx = ((b * S + s_offs) * C) + c
        grad = tl.load(grad_ptr + idx, mask=s_mask, other=0).to(tl.float32)
        x = tl.load(x_ptr + idx, mask=s_mask, other=0).to(tl.float32)
        acc_shift += tl.sum(grad, axis=0)
        acc_scale += tl.sum(grad * x, axis=0)
        s_start += BLOCK_S
    tl.store(dshift_ptr + pid_bc, acc_shift)
    tl.store(dscale_ptr + pid_bc, acc_scale)


@triton.jit
def _swiglu_packed_bwd_kernel(
    gu_ptr,
    dout_ptr,
    dgu_ptr,
    gu_row_stride,
    dout_row_stride,
    dgu_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
) -> None:
    """d(silu(gate) * up) over a packed (rows, 2 * n_cols) buffer, in float32."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < n_cols
    gu = gu_ptr + row * gu_row_stride
    gate = tl.load(gu + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu + n_cols + col, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + row * dout_row_stride + col, mask=mask, other=0.0).to(
        tl.float32
    )
    sig = tl.sigmoid(gate)
    silu = gate * sig
    dgate = dout * up * (silu * (1 - sig) + sig)
    dup = dout * silu
    dgu = dgu_ptr + row * dgu_row_stride
    tl.store(dgu + col, dgate, mask=mask)
    tl.store(dgu + n_cols + col, dup, mask=mask)


def swiglu_packed_backward_fused(gate_up: Tensor, grad: Tensor) -> Tensor:
    """Return the gradient of ``silu(gate) * up`` with all products in float32.

    This is the backward of the routed experts in the chunked MoE dispatch
    of the original training code (one rounding to the input dtype at the
    end). On the CPU the same formula runs in eager PyTorch.
    """
    if not gate_up.is_cuda:
        gate, up = gate_up.chunk(2, dim=-1)
        sig = torch.sigmoid(gate)
        d_gate = grad * up * (sig * (1 + gate * (1 - sig)))
        d_up = grad * (gate * sig)
        return torch.cat([d_gate, d_up], dim=-1)
    n_cols = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * n_cols)
    dout = grad.reshape(-1, n_cols).contiguous()
    dgu = torch.empty_like(gu)
    if gu.shape[0]:
        grid = (gu.shape[0], triton.cdiv(n_cols, SWIGLU_BLOCK))
        _swiglu_packed_bwd_kernel[grid](
            gu,
            dout,
            dgu,
            gu.stride(0),
            dout.stride(0),
            dgu.stride(0),
            n_cols,
            BLOCK=SWIGLU_BLOCK,  # ty: ignore[invalid-argument-type]
        )
    return dgu.view_as(gate_up)


def _swiglu_forward(gate_up: Tensor) -> Tensor:
    n_cols = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * n_cols)
    out = gate_up.new_empty(gu.shape[0], n_cols)
    if gu.shape[0]:
        grid = (gu.shape[0], triton.cdiv(n_cols, SWIGLU_BLOCK))
        _swiglu_packed_kernel[grid](
            gu, out, gu.stride(0), out.stride(0), n_cols, BLOCK=SWIGLU_BLOCK
        )
    return out.view(*gate_up.shape[:-1], n_cols)


@triton.jit
def _moe_combine_kernel(
    expert_ptr,
    gate_ptr,
    inverse_ptr,
    result_ptr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    # Match PyTorch's four interleaved FP32 accumulators for sum(dim=1)
    # of a contiguous (tokens, 8, channels) tensor. The multiplication must
    # round to the expert dtype BEFORE accumulation, just as eager does.
    for slot in tl.static_range(8):
        row = tl.load(inverse_ptr + token * 8 + slot)
        gate = tl.load(gate_ptr + row).to(tl.float32)
        raw = tl.load(expert_ptr + row * C + col, col < C, other=0)
        value = (raw.to(tl.float32) * gate).to(raw.dtype).to(tl.float32)
        if slot % 4 == 0:
            a0 = a0 + value
        elif slot % 4 == 1:
            a1 = a1 + value
        elif slot % 4 == 2:
            a2 = a2 + value
        else:
            a3 = a3 + value
    total = ((a0 + a1) + a2) + a3
    tl.store(result_ptr + token * C + col, total, col < C)


def moe_combine(
    expert_output: Tensor, sorted_gate: Tensor, inverse: Tensor, top_k: int
) -> Tensor:
    """Gate, undo expert sorting and sum, retaining eager rounding.

    The inference kernel specializes the model's top-eight, non-chunked
    reduction. Other layouts/dtypes, autocast and differentiable calls use
    eager ops. Autocast can promote the eager sum to FP32.
    Scaling and the shared-expert addition retain their original boundaries.
    """
    channels = expert_output.shape[1]
    if (
        expert_output.is_cuda
        and sorted_gate.device == expert_output.device
        and inverse.device == expert_output.device
        and inverse.dtype == torch.int64
        and expert_output.dtype in (torch.bfloat16, torch.float16)
        and sorted_gate.dtype == expert_output.dtype
        and top_k == 8
        and channels >= 128
        and expert_output.is_contiguous()
        and sorted_gate.is_contiguous()
        and inverse.is_contiguous()
        and not torch.is_autocast_enabled()
        and not (
            torch.is_grad_enabled()
            and (expert_output.requires_grad or sorted_gate.requires_grad)
        )
    ):
        result = expert_output.new_empty(inverse.numel() // top_k, channels)
        if result.numel():
            _moe_combine_kernel[(result.shape[0], triton.cdiv(channels, 256))](
                expert_output,
                sorted_gate,
                inverse,
                result,
                channels,
                256,
                enable_fp_fusion=False,
            )
        return result
    weighted = expert_output * sorted_gate.unsqueeze(-1)
    return weighted[inverse].reshape(-1, top_k, channels).sum(dim=1)


@triton.jit
def _moe_chunked_combine_kernel(
    expert_ptr,
    gate_ptr,
    inverse_ptr,
    shared_ptr,
    result_ptr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    ADD_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(8):
        row = tl.load(inverse_ptr + token * 8 + slot)
        gate = tl.load(gate_ptr + row).to(tl.float32)
        scale = gate * tl.full((), SCALE, tl.float32)
        raw = tl.load(expert_ptr + row * C + col, col < C, other=0).to(tl.float32)
        value = raw * scale
        if slot % 4 == 0:
            a0 = a0 + value
        elif slot % 4 == 1:
            a1 = a1 + value
        elif slot % 4 == 2:
            a2 = a2 + value
        else:
            a3 = a3 + value
    total = ((a0 + a1) + a2) + a3
    if ADD_SHARED:
        # The chunked path rounds the routed sum before adding the shared
        # expert. Products above must not round to BF16/FP16 first.
        rounded = total.to(shared_ptr.dtype.element_ty).to(tl.float32)
        total = rounded + tl.load(shared_ptr + token * C + col, col < C, other=0).to(
            tl.float32
        )
    tl.store(result_ptr + token * C + col, total, col < C)


def moe_chunked_combine(
    expert_output: Tensor,
    sorted_gate: Tensor,
    inverse: Tensor,
    top_k: int,
    scaling: float,
    *,
    shared: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Preserve the chunked FP32 scaling and four-accumulator reduction.

    Without ``shared``, return FP32. With it, round the routed sum to the
    shared dtype before addition. ``out`` can be a contiguous chunk slice.
    Unsupported and differentiable inputs use the original eager sequence.
    """
    tokens, channels = inverse.numel() // top_k, expert_output.shape[1]
    dtype = torch.float32 if shared is None else shared.dtype
    fast = (
        expert_output.is_cuda
        and sorted_gate.device == expert_output.device
        and inverse.device == expert_output.device
        and inverse.dtype == torch.int64
        and expert_output.dtype in (torch.bfloat16, torch.float16)
        and sorted_gate.dtype == expert_output.dtype
        and top_k == 8
        and channels >= 128
        and all(t.is_contiguous() for t in (expert_output, sorted_gate, inverse))
        and not torch.is_autocast_enabled()
        and not (
            torch.is_grad_enabled()
            and any(
                t.requires_grad
                for t in (expert_output, sorted_gate)
                + (() if shared is None else (shared,))
            )
        )
        and (
            shared is None
            or (
                shared.shape == (tokens, channels)
                and shared.dtype == expert_output.dtype
                and shared.device == expert_output.device
                and shared.is_contiguous()
            )
        )
        and (
            out is None
            or (
                out.shape == (tokens, channels)
                and out.dtype == dtype
                and out.device == expert_output.device
                and out.is_contiguous()
                and not out.requires_grad
            )
        )
    )
    if fast:
        result = (
            torch.empty((tokens, channels), device=expert_output.device, dtype=dtype)
            if out is None
            else out
        )
        if result.numel():
            _moe_chunked_combine_kernel[(tokens, triton.cdiv(channels, 256))](
                expert_output,
                sorted_gate,
                inverse,
                expert_output if shared is None else shared,
                result,
                channels,
                scaling,
                shared is not None,
                256,
                enable_fp_fusion=False,
            )
        return result
    indices = inverse.view(tokens, top_k)
    scale = sorted_gate.to(torch.float32) * scaling
    accumulators = [
        torch.zeros(
            (tokens, channels), device=expert_output.device, dtype=torch.float32
        )
        for _ in range(4)
    ]
    for slot in range(top_k):
        row = indices[:, slot]
        accumulators[slot % 4] = accumulators[slot % 4] + expert_output[
            row
        ].float() * scale[row].unsqueeze(-1)
    result = ((accumulators[0] + accumulators[1]) + accumulators[2]) + accumulators[3]
    if shared is not None:
        result = result.to(shared.dtype) + shared
    return result if out is None else out.copy_(result)


@triton.jit
def _moe_dispatch_rows_kernel(
    x,
    order,
    rows,
    N: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(order + row) // K
    value = tl.load(x + source * C + col, col < C, other=0)
    tl.store(rows + row * C + col, value, col < C)


@triton.jit
def _moe_dispatch_metadata_kernel(
    experts,
    gate,
    order,
    sorted_gate,
    inverse,
    offsets,
    N: tl.constexpr,
    E: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(order + i, i < N, other=0)
    weight = tl.load(gate + source, i < N, other=0)
    tl.store(sorted_gate + i, weight, i < N)
    tl.store(inverse + source, i, i < N)
    if tl.program_id(0) == 0:
        expert = tl.arange(0, EXPERT_BLOCK)
        lo = tl.full((EXPERT_BLOCK,), 0, tl.int32)
        hi = tl.full((EXPERT_BLOCK,), N, tl.int32)
        # Upper bounds in the original stable sort; absent experts get
        # repeated offsets, exactly as torch.searchsorted does.
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            active = (lo < hi) & (expert < E)
            original = tl.load(order + mid, active, other=0)
            value = tl.load(experts + original, active, other=0)
            right = value <= expert
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
            hi = tl.where(expert < E, hi, lo)
        tl.store(offsets + expert, lo, expert < E)


def moe_dispatch(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    order: Tensor,
    num_experts: int,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gather expert rows and prepare metadata, preserving stable sort order."""
    k = top_idx.shape[1]
    n, channels = order.numel(), x.shape[1]
    fast = (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == output_dtype
        and gate.dtype == x.dtype
        and all(t.device == x.device for t in (top_idx, gate, order))
        and all(t.is_contiguous() for t in (x, top_idx, gate, order))
        and top_idx.dtype == torch.int64
        and order.dtype == torch.int64
        and n < 2**31
        and not torch.is_autocast_enabled()
        and not (torch.is_grad_enabled() and (x.requires_grad or gate.requires_grad))
    )
    if not fast:
        sorted_tokens = torch.arange(x.shape[0], device=x.device).repeat_interleave(k)[
            order
        ]
        rows = x[sorted_tokens].to(output_dtype)
        sorted_gate = gate.reshape(-1)[order]
        offsets = torch.searchsorted(
            top_idx.reshape(-1)[order],
            torch.arange(1, num_experts + 1, device=x.device),
            out_int32=True,
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=x.device)
        return rows, sorted_gate, inverse, offsets
    rows = x.new_empty(n, channels)
    sorted_gate = gate.new_empty(n)
    inverse = torch.empty_like(order)
    offsets = torch.empty(num_experts, dtype=torch.int32, device=x.device)
    if n * channels:
        # H100 replay sweeps favor 512-column tiles over full-row tiles:
        # more independent row reads hide the indirect gather latency.
        block = triton.next_power_of_2(min(channels, 512))
        _moe_dispatch_rows_kernel[(n, triton.cdiv(channels, block))](
            x, order, rows, n, channels, k, block
        )
    _moe_dispatch_metadata_kernel[(max(1, triton.cdiv(n, 256)),)](
        top_idx,
        gate,
        order,
        sorted_gate,
        inverse,
        offsets,
        n,
        num_experts,
        triton.next_power_of_2(num_experts),
        256,
    )
    return rows, sorted_gate, inverse, offsets


@triton.jit
def _moe_scale_add_kernel(
    routed, shared, result, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(routed + i, i < N, other=0)
    shared_value = tl.load(shared + i, i < N, other=0)
    # Round the scaling result before addition, retaining eager boundaries.
    scaled = (value.to(tl.float32) * SCALE).to(value.dtype).to(tl.float32)
    tl.store(result + i, scaled + shared_value.to(tl.float32), i < N)


def moe_scale_add(routed: Tensor, shared: Tensor, scale: float) -> Tensor:
    """Fuse routed scaling and shared addition, with an intermediate rounding."""
    if (
        routed.is_cuda
        and shared.device == routed.device
        and routed.dtype in (torch.bfloat16, torch.float16)
        and shared.dtype == routed.dtype
        and shared.shape == routed.shape
        and routed.is_contiguous()
        and shared.is_contiguous()
        and not torch.is_autocast_enabled()
        and not (
            torch.is_grad_enabled() and (routed.requires_grad or shared.requires_grad)
        )
    ):
        result = torch.empty_like(routed)
        if result.numel():
            _moe_scale_add_kernel[(triton.cdiv(result.numel(), 1024),)](
                routed,
                shared,
                result,
                result.numel(),
                scale,
                1024,
                enable_fp_fusion=False,
            )
        return result
    return routed * scale + shared


def swiglu_packed_backward(gate_up: Tensor, grad: Tensor) -> Tensor:
    """Return the gradient of ``silu(gate) * up`` with respect to ``gate_up``.

    The function runs the eager PyTorch backward of ``F.silu(gate) * up``:
    ``dgate = silu_backward(grad * up, gate)`` and ``dup = grad * silu(gate)``.
    Each product rounds to the input dtype, as in the eager backward.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    dgu = torch.empty_like(gate_up)
    dgate, dup = dgu.chunk(2, dim=-1)
    torch.mul(grad, up, out=dgate)
    torch.ops.aten.silu_backward.grad_input(dgate, gate, grad_input=dgate)
    torch.ops.aten.silu.out(gate, out=dup)
    torch.mul(grad, dup, out=dup)
    return dgu


class _SwiGLUPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, gate_up: Tensor) -> Tensor:
        ctx.save_for_backward(gate_up)
        return _swiglu_forward(gate_up)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> Tensor:
        (gate_up,) = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        return swiglu_packed_backward(gate_up, grad)


def swiglu_packed(gate_up: Tensor) -> Tensor:
    """Return ``silu(gate) * up`` for the two halves of a CUDA ``(rows, 2 * H)``."""
    if torch.is_grad_enabled() and gate_up.requires_grad:
        return _SwiGLUPacked.apply(gate_up)
    return _swiglu_forward(gate_up)


def _modulate_forward(x: Tensor, shift: Tensor, scale: Tensor, rows: bool) -> Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, MODULATE_BLOCK),)
    if not rows:
        _modulate_kernel[grid](
            x, shift, scale, out, n_elements, BLOCK_SIZE=MODULATE_BLOCK
        )
        return out
    b, s, c = x.shape
    _modulate_rows_kernel[grid](
        x, shift, scale, out, b, s, c, n_elements, BLOCK_SIZE=MODULATE_BLOCK
    )
    return out


def modulate_fused_backward(
    grad: Tensor,
    x: Tensor,
    scale: Tensor,
    shift_dtype: torch.dtype,
    rows: bool,
    needs: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Return the gradients ``(dx, dshift, dscale)`` of ``modulate_fused``.

    All tensors are contiguous CUDA tensors. ``x`` is ``(B, S, C)``. For
    ``rows=False``, ``scale`` has the shape of ``x``. For ``rows=True``,
    ``scale`` is ``(B, C)`` and the returned ``dshift`` and ``dscale`` are
    ``(B, C)``. The products run in float32. ``dx`` rounds to the dtype of
    ``x``. ``dshift`` rounds to ``shift_dtype`` and ``dscale`` to the dtype of
    ``scale``. ``needs`` selects the gradients to compute.
    """
    n_elements = grad.numel()
    grid = (triton.cdiv(n_elements, MODULATE_BLOCK),)
    if not rows:
        dx = torch.empty_like(x)
        dshift = torch.empty(x.shape, device=x.device, dtype=shift_dtype)
        dscale = torch.empty_like(scale)
        if n_elements:
            _modulate_bwd_kernel[grid](
                grad,
                x,
                scale,
                dx,
                dshift,
                dscale,
                n_elements,
                BLOCK_SIZE=MODULATE_BLOCK,
            )
        return (
            dx if needs[0] else None,
            dshift if needs[1] else None,
            dscale if needs[2] else None,
        )
    b, s, c = x.shape
    dx = dshift = dscale = None
    if needs[0]:
        dx = torch.empty_like(x)
        if n_elements:
            _modulate_rows_bwd_dx_kernel[grid](
                grad, scale, dx, s, c, n_elements, BLOCK_SIZE=MODULATE_BLOCK
            )
    if needs[1] or needs[2]:
        dshift_bc = torch.empty((b, c), device=x.device, dtype=shift_dtype)
        dscale_bc = torch.empty((b, c), device=x.device, dtype=scale.dtype)
        if b * c:
            _modulate_rows_bwd_reduce_kernel[(b * c,)](
                grad, x, dshift_bc, dscale_bc, s, c, BLOCK_S=MODULATE_REDUCE_BLOCK
            )
        dshift = dshift_bc if needs[1] else None
        dscale = dscale_bc if needs[2] else None
    return dx, dshift, dscale


class _ModulateFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        rows: bool,
    ) -> Tensor:
        ctx.save_for_backward(x, scale)
        ctx.rows = rows  # ty: ignore[unresolved-attribute]
        ctx.shift_dtype = shift.dtype  # ty: ignore[unresolved-attribute]
        return _modulate_forward(x, shift, scale, rows)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        x, scale = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        needs = tuple(ctx.needs_input_grad[:3])  # ty: ignore[unresolved-attribute]
        dx, dshift, dscale = modulate_fused_backward(
            grad.contiguous(),
            x,
            scale,
            ctx.shift_dtype,  # ty: ignore[unresolved-attribute]
            ctx.rows,  # ty: ignore[unresolved-attribute]
            needs,
        )
        return dx, dshift, dscale, None


def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor, rows: bool) -> Tensor:
    """Return ``x * scale + shift`` in float32, cast to the dtype of ``x``.

    All three tensors are 3-D CUDA tensors. With ``rows=False`` they have one
    shape. With ``rows=True``, ``shift`` and ``scale`` are ``(B, 1, C)`` and
    apply to every token of ``x`` ``(B, S, C)``.
    """
    x = x.contiguous()
    if rows:
        shift, scale = shift[:, 0, :], scale[:, 0, :]
    shift, scale = shift.contiguous(), scale.contiguous()
    if torch.is_grad_enabled() and (
        x.requires_grad or shift.requires_grad or scale.requires_grad
    ):
        return _ModulateFused.apply(x, shift, scale, rows)
    return _modulate_forward(x, shift, scale, rows)


# ---------------------------------------------------------------------------
# unchunked combine + scale + shared
# add in one kernel, and the dispatch metadata without the row gather.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_combine_scale_add_kernel(
    expert_ptr,
    gate_ptr,
    inverse_ptr,
    shared_ptr,
    result_ptr,
    T0,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """``moe_scale_add(moe_combine(...)[T0 + t], shared[t], SCALE)`` for token t."""
    t = tl.program_id(0).to(tl.int64)
    token = T0 + t
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(8):
        row = tl.load(inverse_ptr + token * 8 + slot)
        gate = tl.load(gate_ptr + row).to(tl.float32)
        raw = tl.load(expert_ptr + row * C + col, col < C, other=0)
        value = (raw.to(tl.float32) * gate).to(raw.dtype).to(tl.float32)
        if slot % 4 == 0:
            a0 = a0 + value
        elif slot % 4 == 1:
            a1 = a1 + value
        elif slot % 4 == 2:
            a2 = a2 + value
        else:
            a3 = a3 + value
    # moe_combine stores its sum in the expert dtype; moe_scale_add rounds the
    # scaled value to that dtype before the float32 add and the final store.
    routed = (((a0 + a1) + a2) + a3).to(result_ptr.dtype.element_ty)
    scaled = (routed.to(tl.float32) * SCALE).to(routed.dtype).to(tl.float32)
    shared_value = tl.load(shared_ptr + t * C + col, col < C, other=0)
    tl.store(result_ptr + t * C + col, scaled + shared_value.to(tl.float32), col < C)


def moe_combine_scale_add(
    expert_output: Tensor,
    sorted_gate: Tensor,
    inverse: Tensor,
    shared: Tensor,
    scale: float,
    token_start: int = 0,
) -> Tensor:
    """``moe_scale_add(moe_combine(expert_output, sorted_gate, inverse, 8)
    [token_start:token_start + len(shared)], shared, scale)`` in one kernel
    (CUDA, contiguous bf16/fp16 fast path only; callers check
    ``moe_combine_scale_add_supported``)."""
    channels = expert_output.shape[1]
    result = torch.empty_like(shared)
    if result.numel():
        _moe_combine_scale_add_kernel[(shared.shape[0], triton.cdiv(channels, 256))](
            expert_output,
            sorted_gate,
            inverse,
            shared,
            result,
            token_start,
            channels,
            scale,
            256,
            enable_fp_fusion=False,
        )
    return result


def moe_combine_scale_add_supported(
    expert_output: Tensor,
    sorted_gate: Tensor,
    inverse: Tensor,
    shared: Tensor,
    top_k: int,
) -> bool:
    return (
        expert_output.is_cuda
        and expert_output.dtype in (torch.bfloat16, torch.float16)
        and sorted_gate.dtype == expert_output.dtype
        and shared.dtype == expert_output.dtype
        and shared.dim() == 2
        and shared.shape[1] == expert_output.shape[1]
        and top_k == 8
        and expert_output.shape[1] >= 128
        and expert_output.is_contiguous()
        and sorted_gate.is_contiguous()
        and inverse.is_contiguous()
        and shared.is_contiguous()
        and not torch.is_autocast_enabled()
        and not torch.is_grad_enabled()
    )


def moe_dispatch_metadata(
    top_idx: Tensor, gate: Tensor, order: Tensor, num_experts: int
) -> tuple[Tensor, Tensor, Tensor]:
    """``moe_dispatch(...)[1:]`` (sorted gates, inverse, offsets) without the
    row gather, for a GEMM that reads ``x[order[i] // top_k]`` itself. CUDA,
    contiguous int64 ``top_idx``/``order`` only."""
    n = order.numel()
    sorted_gate = gate.new_empty(n)
    inverse = torch.empty_like(order)
    offsets = torch.empty(num_experts, dtype=torch.int32, device=order.device)
    _moe_dispatch_metadata_kernel[(max(1, triton.cdiv(n, 256)),)](
        top_idx,
        gate,
        order,
        sorted_gate,
        inverse,
        offsets,
        n,
        num_experts,
        triton.next_power_of_2(num_experts),
        256,
    )
    return sorted_gate, inverse, offsets
