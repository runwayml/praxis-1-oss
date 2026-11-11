"""Optional, inference-only NVFP4 storage for routed expert banks.

Only the routed MLP weights and its GEMMs are quantized. Router, shared
expert, and surrounding model precision are owned by the original layers.
TorchAO packs one CPU expert at a time. FlashInfer is imported at execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from praxis.hardware import is_thor, require_thor

# Enable fused Triton stages when the selected NVFP4 backend supports them.
# Paired dispatch requires the SM110 scale layout. Elementwise fusion also
# applies to SM120; SM100/SM103 retain their existing native fusion paths.
FUSED_NVFP4 = True
FUSED_SWIGLU_FP4 = True
PAIR_NVFP4 = True


class _FusedRows(NamedTuple):
    raw: Tensor
    csr: Tensor
    pcsr: Tensor
    ag: Tensor
    wg: Tensor


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


def backend_for_device(device: torch.device | str) -> str:
    """Select a native NVFP4 backend; BF16 callers never enter this module."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("NVFP4 execution requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    if capability in ((10, 0), (10, 3)):
        return "flashinfer_sm100_contiguous_nvfp4"
    if capability == (11, 0):
        return "quack_sm110_grouped_nvfp4"
    if capability == (12, 0):
        return "flashinfer_sm120_grouped_nvfp4"
    raise RuntimeError(
        "NVFP4 requires SM100, SM103, SM110, or SM120; use routed_precision='bf16' on H100"
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
    backend = "flashinfer_grouped_nvfp4"
    ACCEPTS_DISPATCH = (
        True  # routed_forward(_with_shared)(..., dispatch=router_fused.Routing)
    )

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
        self._fc1_fp4_scale = None
        self._fc1_fp4_cache = None
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

    def set_fused_swiglu_scale(self, scale: float | None) -> None:
        """Set a calibrated FC1 output scale, or disable FP4 output with None.

        This changes output bits. Configure before inference or graph capture.
        The scale and prepared weight copies are not checkpoint buffers.
        """
        if scale is not None and (
            isinstance(scale, bool)
            or not isinstance(scale, float | int)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("FC1 output scale must be finite and positive")
        if (
            scale is not None
            and self.gate_up_q.device.type == "cuda"
            and backend_for_device(self.gate_up_q.device)
            != "flashinfer_sm100_contiguous_nvfp4"
        ):
            raise RuntimeError("Fused SwiGLU FP4 output requires SM100 or SM103")
        self._fc1_fp4_scale = None if scale is None else float(scale)
        self._fc1_fp4_cache = None
        if scale is not None and self.gate_up_q.device.type == "cuda":
            from praxis.kernels.nvfp4_swiglu import prepare

            self._fc1_fp4_cache = prepare(self.gate_up_q, self.gate_up_sf, scale)

    def _load_from_state_dict(self, *args, **kwargs):
        self._fc1_fp4_scale = None
        self._fc1_fp4_cache = None
        super()._load_from_state_dict(*args, **kwargs)

    def _restore_gate_up_layout(self) -> None:
        if getattr(self, "_gate_up_interleaved", False):
            from praxis.kernels.nvfp4_epi import sync_gate_up_layout

            sync_gate_up_layout(self, False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._restore_gate_up_layout()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _apply(self, fn, recurse: bool = True) -> NVFP4RoutedExperts:
        self._restore_gate_up_layout()
        moved = {name: fn(value) for name, value in self._buffers.items()}
        for name, value in self._buffers.items():
            other = moved[name]
            if other.dtype != value.dtype or other.shape != value.shape:
                raise TypeError(
                    "NVFP4 buffers support device moves, not dtype/shape casts"
                )
        self._buffers.update(moved)
        self._fc1_fp4_cache = None
        return self

    @torch.compiler.disable
    def _expert_outputs(
        self,
        x: Tensor,
        offsets: Tensor,
        token_rows: Tensor,
        *,
        raw_output: bool = False,
    ) -> Tensor | _FusedRows | tuple[Tensor, Tensor, Tensor, Tensor]:
        if is_thor(x.device):
            if raw_output:
                raise ValueError("SM110 uses its own fused-combine output contract")
            return self._expert_outputs_sm110(x, offsets, token_rows)
        if getattr(self, "_gate_up_interleaved", False):
            from praxis.kernels.nvfp4_epi import sync_gate_up_layout

            sync_gate_up_layout(self, False)
        # FlashInfer owns its JIT cache and GEMM dispatch. Keep that Python
        # machinery outside Dynamo; its CUDA work remains graph-capturable.
        from praxis.kernels.layers_triton import swiglu_packed
        from praxis.kernels.nvfp4 import grouped_nvfp4_mm, quantize_activations, rescale
        from praxis.kernels.nvfp4_moe import (
            expert_scales_from_row_maximum,
            grouped_metadata,
            rescale_swiglu_maximum,
        )

        sm100 = backend_for_device(x.device) == "flashinfer_sm100_contiguous_nvfp4"
        packing = dict(alignment=128, grouped_scales=False) if sm100 else {}
        csr = F.pad(offsets, (1, 0))
        aq, asf, ag, pcsr = quantize_activations(
            x,
            csr,
            row_indices=token_rows,
            **packing,
        )
        metadata = grouped_metadata(pcsr, aq.shape[0]) if sm100 else None
        if sm100 and FUSED_SWIGLU_FP4 and self._fc1_fp4_scale is not None:
            from praxis.kernels.nvfp4_swiglu import gate_up_fp4, prepare

            if self._fc1_fp4_cache is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Prepare fused SwiGLU before CUDA Graph capture")
                self._fc1_fp4_cache = prepare(
                    self.gate_up_q, self.gate_up_sf, self._fc1_fp4_scale
                )
            hq, hsf, hg = gate_up_fp4(
                aq, asf, ag, self.gate_up_global, self._fc1_fp4_cache, metadata
            )
            del aq, asf, ag
        else:
            gu = grouped_nvfp4_mm(
                aq,
                self.gate_up_q,
                asf,
                self.gate_up_sf,
                pcsr,
                self._unit_alpha,
                metadata=metadata,
            )
            del aq, asf
            padded_rows = gu.shape[0]
            if sm100 or FUSED_NVFP4:
                hidden, row_maximum = rescale_swiglu_maximum(
                    gu, pcsr, ag, self.gate_up_global
                )
                hidden_globals = expert_scales_from_row_maximum(row_maximum, pcsr)
                del row_maximum
            else:
                scaled = rescale(gu, pcsr, pcsr, ag, self.gate_up_global, padded_rows)
                # Praxis retains Triton's sigmoid and each BF16 rounding boundary.
                hidden = swiglu_packed(scaled)
                hidden_globals = None
                del scaled
            del gu, ag
            hq, hsf, hg, _ = quantize_activations(
                hidden,
                pcsr,
                padded_csr=pcsr,
                padded_rows=padded_rows,
                activation_globals=hidden_globals,
                **packing,
            )
            del hidden
        raw = grouped_nvfp4_mm(
            hq,
            self.down_q,
            hsf,
            self.down_sf,
            pcsr,
            self._unit_alpha,
            metadata=metadata,
        )
        del hq, hsf
        if raw_output:
            return raw, csr, pcsr, hg
        if FUSED_NVFP4:
            return _FusedRows(raw, csr, pcsr, hg, self.down_global)
        return rescale(raw, csr, pcsr, hg, self.down_global, token_rows.numel())

    @torch.compiler.disable
    def _expert_outputs_sm110(
        self, x: Tensor, offsets: Tensor, token_rows: Tensor
    ) -> Tensor:
        """SM 11.0 (Thor): the same roundings as _expert_outputs; the grouped GEMMs run in quack
        (CuTe DSL tcgen05 block-scaled varlen_m) instead of FlashInfer's SM120 kernel. See
        praxis/kernels/nvfp4_sm110.py."""
        require_thor(x.device)
        from praxis.kernels.layers_triton import swiglu_packed
        from praxis.kernels.nvfp4 import rescale
        from praxis.kernels.nvfp4_sm110 import (
            grouped_gemm_nvfp4,
            quantize_activations_q,
        )

        csr = F.pad(offsets, (1, 0))
        if FUSED_NVFP4:
            return self._expert_outputs_sm110_fused(x, csr, token_rows)
        if getattr(self, "_gate_up_interleaved", False):
            from praxis.kernels.nvfp4_epi import sync_gate_up_layout

            sync_gate_up_layout(self, False)
        aq, asf, ag, pcsr = quantize_activations_q(x, csr, row_indices=token_rows)
        gu = grouped_gemm_nvfp4(aq, asf, self.gate_up_q, self.gate_up_sf, pcsr)
        del aq, asf
        padded_rows = gu.shape[0]
        scaled = rescale(gu, pcsr, pcsr, ag, self.gate_up_global, padded_rows)
        del gu, ag
        hidden = swiglu_packed(scaled)
        del scaled
        hq, hsf, hg, _ = quantize_activations_q(
            hidden, pcsr, padded_csr=pcsr, padded_rows=padded_rows
        )
        del hidden
        raw = grouped_gemm_nvfp4(hq, hsf, self.down_q, self.down_sf, pcsr)
        del hq, hsf
        return rescale(raw, csr, pcsr, hg, self.down_global, token_rows.numel())

    @torch.compiler.disable
    def _expert_outputs_sm110_fused(
        self, x: Tensor, csr: Tensor, token_rows: Tensor
    ) -> Tensor:
        """_expert_outputs_sm110 with the elementwise stages fused (FUSED_NVFP4, see
        praxis/kernels/nvfp4_fused.py): same roundings. Returns the rescaled rows only through the fused
        combine, so the caller passes ``combine`` instead of combining itself."""
        require_thor(x.device)
        from praxis.kernels import nvfp4_fused as NF
        from praxis.kernels import nvfp4_sm110 as NS
        from praxis.kernels.nvfp4_epi import gate_up_swiglu_amax, sync_gate_up_layout
        from praxis.kernels.nvfp4_sm110 import (
            grouped_gemm_nvfp4,
            quantize_activations_q,
        )

        sync_gate_up_layout(self, NS.EPI_SWIGLU)
        ag = NF.globals_from_row_amax(NF.row_amax(x), csr, token_rows)
        aq, asf, _, pcsr = quantize_activations_q(x, csr, row_indices=token_rows, g=ag)
        padded_rows = aq.shape[0]
        if NS.EPI_SWIGLU:
            # Rescale + SwiGLU + row amax in the GEMM epilogue (same bits).
            hidden, hamax = gate_up_swiglu_amax(self, aq, asf, pcsr, ag)
            hamax = hamax.to(torch.bfloat16)
            del aq, asf
        else:
            gu = grouped_gemm_nvfp4(aq, asf, self.gate_up_q, self.gate_up_sf, pcsr)
            del aq, asf
            hidden, hamax = NF.rescale_swiglu_amax(gu, pcsr, ag, self.gate_up_global)
            del gu
        hg = NF.globals_from_row_amax(hamax, pcsr)
        hq, hsf, _, _ = quantize_activations_q(
            hidden, pcsr, padded_csr=pcsr, padded_rows=padded_rows, g=hg
        )
        del hidden
        raw = grouped_gemm_nvfp4(hq, hsf, self.down_q, self.down_sf, pcsr)
        del hq, hsf
        return _FusedRows(raw, csr, pcsr, hg, self.down_global)

    def routed_forward_pair(
        self, xs, top_idxs, gates, *, scaling: float, tail_shared=None, dispatch=None
    ) -> list[Tensor]:
        """``[self.routed_forward(x, i, g, scaling=scaling, chunk_size=len(x)) for ...]`` of two inputs in one
        paired dispatch (PAIR_NVFP4; SM 11.0; each input one dispatch chunk). ``tail_shared``: see
        ``pair_expert_outputs`` (MOE_TAIL_NORM). ``dispatch``: a ``praxis.layers.ROUTER_FUSED``
        router_fused.Routing (P = 2) of these inputs: its metadata replaces the dispatch sort (same values)."""
        if not PAIR_NVFP4 or not xs:
            raise RuntimeError("PAIR_NVFP4 must be enabled for Thor paired dispatch")
        for x in xs:
            require_thor(x.device)
        from praxis.kernels.nvfp4_pair import pair_expert_outputs

        return pair_expert_outputs(
            self,
            list(xs),
            list(top_idxs),
            list(gates),
            scaling,
            tail_shared=tail_shared,
            dispatch=dispatch,
        )

    def routed_forward(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
        dispatch=None,
    ) -> Tensor:
        """Return the scaled routed sum with the original BF16 output boundary."""
        return self._routed_forward(
            x, top_idx, gate, scaling=scaling, chunk_size=chunk_size, dispatch=dispatch
        )

    def routed_forward_with_shared(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
        shared: Tensor,
        dispatch=None,
    ) -> Tensor:
        """Add the native shared expert after rounding the routed sum to BF16. ``dispatch``: a
        router_fused.Routing (P = 1) of the whole input, used when the input is one dispatch chunk."""
        if (
            shared.shape != x.shape
            or shared.dtype != x.dtype
            or shared.device != x.device
            or not shared.is_contiguous()
            or (torch.is_grad_enabled() and shared.requires_grad)
        ):
            return (
                self.routed_forward(
                    x,
                    top_idx,
                    gate,
                    scaling=scaling,
                    chunk_size=chunk_size,
                    dispatch=dispatch,
                )
                + shared
            )
        return self._routed_forward(
            x,
            top_idx,
            gate,
            scaling=scaling,
            chunk_size=chunk_size,
            shared=shared,
            dispatch=dispatch,
        )

    def routed_tail(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
        shared: Tensor,
        shape,
    ) -> object | None:
        """``praxis.layers.MOE_TAIL_NORM``: ``routed_forward_with_shared`` of one dispatch chunk on SM
        11.0 with FUSED_NVFP4, up to (not including) the combine: the same dispatch and expert outputs, returned
        as a ``praxis.layers.MoETail`` (mode 2) that the next norm combines. None (before any work) where the
        stock path differs (several chunks, other backends, unfused path, unusual inputs)."""
        from praxis.layers import MoETail, _dispatch_rows

        if not (
            FUSED_NVFP4
            and not self.training
            and not torch.is_grad_enabled()
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.device.type == "cuda"
            and x.shape[1] == self.in_features
            and 0 < x.shape[0] <= chunk_size
            and top_idx.ndim == 2
            and top_idx.shape == (x.shape[0], 8)
            and top_idx.dtype == torch.int64
            and top_idx.is_contiguous()
            and gate.shape == top_idx.shape
            and gate.dtype == torch.bfloat16
            and gate.is_contiguous()
            and shared.shape == x.shape
            and shared.dtype == x.dtype
            and shared.device == x.device
            and shared.is_contiguous()
            and backend_for_device(x.device) == "quack_sm110_grouped_nvfp4"
            and all(value.device == x.device for value in self.buffers())
            and isinstance(scaling, float | int)
            and not isinstance(scaling, bool)
            and math.isfinite(scaling)
            and (x.shape[0] * 8 + 3 * self.num_experts)
            * max(self.in_features, 2 * self.hidden_features)
            < 2**31
        ):
            return None
        order, tokens, sorted_gate, offsets = _dispatch_rows(
            top_idx, gate, self.num_experts
        )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=x.device)
        rows = self._expert_outputs(x, offsets, tokens)
        assert isinstance(rows, _FusedRows)
        scale = sorted_gate.to(torch.float32) * scaling
        return MoETail(
            2,
            rows.raw,
            inverse,
            scale,
            shared,
            float(scaling),
            shape,
            csr=rows.csr,
            pcsr=rows.pcsr,
            g=rows.ag,
            w=rows.wg,
        )

    def _routed_forward(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        *,
        scaling: float,
        chunk_size: int,
        shared: Tensor | None = None,
        dispatch=None,
    ) -> Tensor:
        """Return the scaled routed sum and optionally add the shared expert. ``dispatch``: a
        router_fused.Routing (P = 1) of the whole input; used when the input is one chunk (the values of the
        _dispatch_rows sort of its top_idx / gate and of the inverse permutation)."""
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
        backend = backend_for_device(x.device)
        padding = 127 if backend == "flashinfer_sm100_contiguous_nvfp4" else 3
        from praxis.kernels.nvfp4 import combine_rows
        from praxis.kernels.nvfp4_moe import rescale_combine
        from praxis.layers import _dispatch_rows

        sm100 = backend == "flashinfer_sm100_contiguous_nvfp4"
        top_k = top_idx.shape[1]
        # The widest padded matrix is the gate/up output, 2 * hidden columns.
        if (min(chunk_size, x.shape[0]) * top_k + padding * self.num_experts) * max(
            self.in_features, 2 * self.hidden_features
        ) >= 2**31:
            raise ValueError("Routed chunk exceeds NVFP4 kernel indexing bounds")
        out = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            if (
                dispatch is not None
                and start == 0
                and end == x.shape[0]
                and len(dispatch.rows) == 1
            ):
                order, tokens, sorted_gate, offsets = (
                    dispatch.order,
                    dispatch.tokens,
                    dispatch.sorted_gate,
                    dispatch.offsets,
                )
                inverse = dispatch.inverse
            else:
                order, tokens, sorted_gate, offsets = _dispatch_rows(
                    top_idx[start:end],
                    gate[start:end],
                    self.num_experts,
                )
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(order.numel(), device=x.device)
            if sm100:
                raw, csr, pcsr, hidden_globals = self._expert_outputs(
                    x[start:end], offsets, tokens, raw_output=True
                )
                out[start:end] = rescale_combine(
                    raw,
                    inverse,
                    top_idx[start:end],
                    gate[start:end],
                    csr,
                    pcsr,
                    hidden_globals,
                    self.down_global,
                    scaling,
                    None if shared is None else shared[start:end],
                )
                del raw, csr, pcsr, hidden_globals
            else:
                rows = self._expert_outputs(x[start:end], offsets, tokens)
                # Preserve FP32 gate * scaling before routed multiplication.
                scale = sorted_gate.to(torch.float32) * scaling
                if isinstance(rows, _FusedRows):
                    from praxis.kernels.nvfp4_fused import combine_rescale

                    result = combine_rescale(
                        rows.raw,
                        inverse,
                        scale,
                        rows.csr,
                        rows.pcsr,
                        rows.ag,
                        rows.wg,
                        end - start,
                        top_k,
                    )
                else:
                    result = combine_rows(rows, inverse, scale, end - start, top_k)
                out[start:end] = (
                    result if shared is None else result + shared[start:end]
                )
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
