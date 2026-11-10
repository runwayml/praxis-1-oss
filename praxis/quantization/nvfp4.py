"""Optional, inference-only NVFP4 storage for routed expert banks.

Only the routed MLP weights and its GEMMs are quantized. Router, shared
expert, and surrounding model precision are owned by the original layers.
TorchAO packs one CPU expert at a time. FlashInfer is imported at execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PACKING_RECIPE = "qat-reciprocal-v1"
_BUFFER_NAMES = frozenset(
    (
        "gate_up_q",
        "gate_up_sf",
        "gate_up_global",
        "down_q",
        "down_sf",
        "down_global",
        "_unit_alpha",
    )
)


def _pack_matrix_cpu(weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pack one contiguous BF16 [out, in] CPU matrix, with no GPU work.

    Rounds exactly like the CUDA QAT forward the routed experts were trained
    with: FP32 reciprocal multiplies by 2688 and 6, not true division.
    """
    try:
        from torchao.prototype.mx_formats.kernels import (
            f32_to_f4_unpacked,
            pack_uint4,
        )
        from torchao.prototype.mx_formats.utils import to_blocked
    except ImportError as exc:
        raise ImportError(
            "NVFP4 packing requires the optional torchao package"
        ) from exc
    if not torch.isfinite(weight).all().item():
        raise ValueError("Routed source weights contain nonfinite values")
    n, k = weight.shape
    global_scale = weight.abs().amax().clamp_min(1e-12).float() * (1.0 / 2688.0)
    blocks = weight.float().reshape(n, k // 16, 16)
    block_scale = (
        (blocks.abs().amax(-1) * (1.0 / 6.0) / global_scale)
        .clamp(2.0**-6, 448.0)
        .to(torch.float8_e4m3fn)
    )
    reciprocal = (1.0 / global_scale) / block_scale.float()
    scaled = (blocks * reciprocal.unsqueeze(-1)).clamp(-6.0, 6.0).reshape(n, k)
    return (
        pack_uint4(f32_to_f4_unpacked(scaled)),
        to_blocked(block_scale)
        .view(torch.uint8)
        .reshape(((n + 127) // 128) * 128, k // 16),
        global_scale.reshape(()),
    )


def pack_routed_experts(
    gate_up_weight: Tensor,
    down_weight: Tensor,
) -> NVFP4RoutedExperts:
    """Pack expert-first CPU weights [E,C,2H]/[E,H,C]; retain no BF16 master."""
    for weight in (gate_up_weight, down_weight):
        if (
            not isinstance(weight, Tensor)
            or weight.ndim != 3
            or weight.device.type != "cpu"
            or weight.dtype != torch.bfloat16
            or weight.layout != torch.strided
        ):
            raise ValueError("Packing requires real CPU BF16 expert-first banks")
    e, c, doubled_hidden = gate_up_weight.shape
    h = doubled_hidden // 2
    if (
        e <= 0
        or c <= 0
        or h <= 0
        or doubled_hidden % 2
        or c % 128
        or h % 128
        or down_weight.shape != (e, h, c)
    ):
        raise ValueError("Inconsistent routed banks; C and H must be multiples of 128")
    state: dict[str, Tensor] = {}
    with torch.no_grad():
        for name, weight, n, k in (
            ("gate_up", gate_up_weight, 2 * h, c),
            ("down", down_weight, c, h),
        ):
            state[name + "_q"] = torch.empty((e, n, k // 2), dtype=torch.uint8)
            state[name + "_sf"] = torch.empty((e, n, k // 16), dtype=torch.uint8)
            state[name + "_global"] = torch.empty(e, dtype=torch.float32)
            for expert in range(e):
                matrix = weight[expert].detach().t().contiguous()
                q, sf, global_scale = _pack_matrix_cpu(matrix)
                state[name + "_q"][expert].copy_(q)
                state[name + "_sf"][expert].copy_(sf)
                state[name + "_global"][expert].copy_(global_scale)
        state["_unit_alpha"] = torch.ones(e, dtype=torch.float32)
    return NVFP4RoutedExperts.from_packed(state)


class NVFP4RoutedExperts(nn.Module):
    """Packed buffers with the Praxis chunked routed-sum precision contract.

    Device moves are supported. Dtype casts, training and gradients are not.
    Shapes can vary between calls; captured calls retain their normal graph
    allocations and never read routing tensors or offsets back to the CPU.
    """

    format = "nvfp4"
    backend = "flashinfer_sm120_grouped_nvfp4"

    def __init__(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        super().__init__()
        if not isinstance(state, Mapping) or set(state) != _BUFFER_NAMES:
            raise ValueError("Unexpected NVFP4 packed buffer keys")
        if any(
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.requires_grad
            for value in state.values()
        ):
            raise ValueError("Reconstruction requires contiguous detached CPU buffers")
        gu = state["gate_up_q"]
        if gu.ndim != 3:
            raise ValueError("Expected grouped gate/up packed matrix")
        e, n, k2 = gu.shape
        c, h = k2 * 2, n // 2
        if e <= 0 or c <= 0 or h <= 0 or n % 2 or c % 128 or h % 128:
            raise ValueError("Invalid grouped NVFP4 dimensions")
        expected = {
            "gate_up_q": ((e, 2 * h, c // 2), torch.uint8),
            "gate_up_sf": ((e, 2 * h, c // 16), torch.uint8),
            "gate_up_global": ((e,), torch.float32),
            "down_q": ((e, c, h // 2), torch.uint8),
            "down_sf": ((e, c, h // 16), torch.uint8),
            "down_global": ((e,), torch.float32),
            "_unit_alpha": ((e,), torch.float32),
        }
        for name, (shape, dtype) in expected.items():
            value = state[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Invalid NVFP4 buffer shape/dtype: {name}")
            if name.endswith("_global"):
                if (
                    not torch.isfinite(value).all().item()
                    or not (value > 0).all().item()
                ):
                    raise ValueError(f"Invalid positive global scales: {name}")
            if name.endswith("_sf"):
                decoded = value.view(torch.float8_e4m3fn).float()
                if (
                    not torch.isfinite(decoded).all().item()
                    or (decoded < 0).any().item()
                ):
                    raise ValueError(f"Invalid E4M3 block scales: {name}")
        if not torch.equal(state["_unit_alpha"], torch.ones(e, dtype=torch.float32)):
            raise ValueError("NVFP4 GEMM alpha must be one")
        self.num_experts, self.in_features, self.hidden_features = e, c, h
        for name in expected:
            self.register_buffer(name, state[name].detach())
        self.train(False)

    @classmethod
    def from_packed(
        cls,
        state: Mapping[str, Tensor],
    ) -> NVFP4RoutedExperts:
        """Reconstruct validated plain CPU buffers from a weights-only cache."""
        return cls(state)

    def train(self, mode: bool = True) -> NVFP4RoutedExperts:
        if mode:
            raise RuntimeError("NVFP4 routed experts support inference only")
        super().train(False)
        return self

    def _apply(self, fn, recurse: bool = True) -> NVFP4RoutedExperts:
        moved = {name: fn(value) for name, value in self._buffers.items()}
        for name, value in self._buffers.items():
            other = moved[name]
            if other.dtype != value.dtype or other.shape != value.shape:
                raise TypeError(
                    "NVFP4 buffers support device moves, not dtype/shape casts"
                )
        self._buffers.update(moved)
        return self

    @torch.compiler.disable
    def _expert_outputs(self, x: Tensor, offsets: Tensor, token_rows: Tensor) -> Tensor:
        # FlashInfer owns its JIT cache and GEMM dispatch. Keep that Python
        # machinery outside Dynamo; its CUDA work remains graph-capturable.
        from praxis.kernels.layers_triton import swiglu_packed
        from praxis.kernels.nvfp4 import quantize_activations, rescale

        try:
            from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
        except ImportError as exc:
            raise ImportError(
                "NVFP4 execution requires FlashInfer with SM120 grouped GEMM"
            ) from exc
        csr = F.pad(offsets, (1, 0))
        aq, asf, ag, pcsr = quantize_activations(
            x,
            csr,
            row_indices=token_rows,
        )
        gu = group_gemm_nvfp4_nt_groupwise(
            aq,
            self.gate_up_q,
            asf,
            self.gate_up_sf,
            pcsr,
            self._unit_alpha,
            tile_n=128,
            tile_k=128,
            out_dtype=torch.bfloat16,
        )
        del aq, asf
        padded_rows = gu.shape[0]
        scaled = rescale(gu, pcsr, pcsr, ag, self.gate_up_global, padded_rows)
        del gu, ag
        # The routed SwiGLU of the bf16 path (swiglu_packed, Triton sigmoid).
        hidden = swiglu_packed(scaled)
        del scaled
        hq, hsf, hg, _ = quantize_activations(
            hidden,
            pcsr,
            padded_csr=pcsr,
            padded_rows=padded_rows,
        )
        del hidden
        raw = group_gemm_nvfp4_nt_groupwise(
            hq,
            self.down_q,
            hsf,
            self.down_sf,
            pcsr,
            self._unit_alpha,
            tile_n=128,
            tile_k=128,
            out_dtype=torch.bfloat16,
        )
        del hq, hsf
        return rescale(raw, csr, pcsr, hg, self.down_global, token_rows.numel())

    def routed_forward(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
    ) -> Tensor:
        """Return the scaled routed sum; caller adds its unchanged shared expert."""
        if self.training or (
            torch.is_grad_enabled() and (x.requires_grad or gate.requires_grad)
        ):
            raise RuntimeError("NVFP4 routed experts support inference only")
        if (
            x.ndim != 2
            or x.shape[1] != self.in_features
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.device.type != "cuda"
        ):
            raise ValueError("Expected contiguous CUDA BF16 routed activations")
        if (
            top_idx.ndim != 2
            or top_idx.shape[0] != x.shape[0]
            or not 1 <= top_idx.shape[1] <= min(8, self.num_experts)
            or top_idx.dtype != torch.int64
            or top_idx.device != x.device
            or not top_idx.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous CUDA int64 expert indices, top-k <= 8"
            )
        if (
            gate.shape != top_idx.shape
            or gate.dtype != torch.bfloat16
            or gate.device != x.device
            or not gate.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA BF16 gate weights")
        if any(value.device != x.device for value in self.buffers()):
            raise ValueError("Packed expert buffers and inputs must share a device")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("NVFP4 inference requires a positive dispatch chunk size")
        if (
            isinstance(scaling, bool)
            or not isinstance(scaling, float | int)
            or not math.isfinite(scaling)
        ):
            raise ValueError("Routed scaling must be finite")
        if torch.cuda.get_device_capability(x.device) != (12, 0):
            raise RuntimeError("The NVFP4 grouped backend requires an SM120 GPU")
        from praxis.kernels.nvfp4 import combine_rows
        from praxis.layers import _dispatch_rows

        top_k = top_idx.shape[1]
        # The widest padded matrix is the gate/up output, 2 * hidden columns.
        if (min(chunk_size, x.shape[0]) * top_k + 3 * self.num_experts) * max(
            self.in_features, 2 * self.hidden_features
        ) >= 2**31:
            raise ValueError("Routed chunk exceeds NVFP4 kernel indexing bounds")
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            order, tokens, sorted_gate, offsets = _dispatch_rows(
                top_idx[start:end],
                gate[start:end],
                self.num_experts,
            )
            rows = self._expert_outputs(x[start:end], offsets, tokens)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=x.device)
            # Preserve _routed_chunk's FP32 gate * scaling before multiplication.
            scale = sorted_gate.to(torch.float32) * scaling
            out[start:end] = combine_rows(rows, inverse, scale, end - start, top_k)
        return out

    def forward(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
    ) -> Tensor:
        return self.routed_forward(
            x, top_idx, gate, scaling=scaling, chunk_size=chunk_size
        )
