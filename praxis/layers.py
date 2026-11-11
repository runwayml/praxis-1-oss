"""Shared transformer layers of the video backbone.

The encoder tower and the decoder tower use the same layers: a zero-centred
RMSNorm, rotary position embedding, grouped-query attention with a per-head
output gate, a SwiGLU feed-forward or a mixture-of-experts feed-forward, and a
block stack that threads the rotary table, the attention mask and the optional
per-token modulation.

With the default Triton attention backend, inference follows the reference
implementation op by op, so bf16 outputs are bitwise equal. Optional attention
backends can round differently. Comments mark every float32 and float64 island.

Two custom Triton kernels (``praxis.kernels.layers_triton``) compute the
expert SwiGLU and the decoder modulation. The original implementation runs
the same kernels.

Training uses the same forward code. Five ops get a custom backward. Thus the
gradients follow the original training code. The ops are the two Triton
kernels, the float32 router GEMM (``_RouterLogits``), the row permutation
of the expert outputs (``_PermuteRows``) and the chunked expert dispatch
(``_ChunkedRoutedExperts``).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Generator, Sequence
from functools import partial
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from praxis import nvtx as NV
from praxis.hardware import is_blackwell_tcgen05, is_thor
from praxis.kernels.layers_triton import (
    modulate_fused,
    moe_chunked_combine,
    moe_combine,
    moe_combine_scale_add,
    moe_combine_scale_add_supported,
    moe_dispatch,
    moe_dispatch_metadata,
    moe_scale_add,
    swiglu_packed,
    swiglu_packed_backward_fused,
)

ModulationDtype = Literal["bfloat16", "float32", "mixed"]
AttentionBackend = Literal["triton", "flash", "cudnn"]

# One compiled flex-attention callable for the whole process. The eager
# flex_attention falls back to a dense kernel that does not fit long sequences.
_compiled_flex_attention = torch.compile(flex_attention)
# Shape-polymorphic twin for callers whose sequence lengths change from call to
# call (the action expert in the "flex_dynamic" mode). It has its own compile
# cache, so its entries do not count against the static one.
#
# Short queries go to the flex decoding kernel. Its split combine is a
# persistent reduction, and by default inductor benchmarks several block
# configs for it at run time. The chosen config changes the summation order,
# so two processes can give outputs that differ in the last bits. The option
# below makes inductor pick one fixed config by a rule (no benchmark). The
# original implementation runs with the same option.
_compiled_flex_attention_dynamic = torch.compile(
    flex_attention,
    dynamic=True,
    options={"test_configs.force_filter_reduction_configs": True},
)

# The router GEMM runs over row chunks of this size. A different chunking can
# make cuBLAS pick a different kernel, which can flip near-tied experts.
_ROUTER_CHUNK_ROWS = 65536


def to_device_async(t: Tensor, device: torch.device) -> Tensor:
    """Copy a small CPU tensor to ``device`` without a host synchronize.

    A ``.to(device)`` of a pageable CPU tensor waits until the GPU has run
    every queued kernel. The CPU then cannot queue the next kernels ahead of
    the GPU. This copy goes through pinned memory with ``non_blocking=True``.
    The values are those of ``t.to(device)``.
    """
    if device.type != "cuda" or t.device.type != "cpu":
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

_MODULATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _modulate_eager(
    x: Tensor,
    shift: Tensor,
    scale: Tensor,
    compute: Literal["bfloat16", "float32"] | None,
) -> Tensor:
    orig_dtype = x.dtype
    if compute is not None:
        dt = torch.float32 if compute == "float32" else torch.bfloat16
        x, shift, scale = x.to(dt), shift.to(dt), scale.to(dt)
    out = x * scale + shift
    if compute is not None:
        out = out.to(orig_dtype)
    return out


def _fused_modulate_case(x: Tensor, shift: Tensor, scale: Tensor) -> str | None:
    """Return "same", "rows" or None for the fused modulate kernel."""
    tensors = (x, shift, scale)
    if not all(t.is_cuda and t.device == x.device for t in tensors):
        return None
    if not all(t.dtype in _MODULATE_DTYPES and t.ndim == 3 for t in tensors):
        return None
    if x.shape == shift.shape == scale.shape:
        return "same"
    if (
        shift.shape == scale.shape
        and shift.shape[0] == x.shape[0]
        and shift.shape[1] == 1
        and shift.shape[2] == x.shape[2]
    ):
        return "rows"
    return None


def modulate(
    x: Tensor,
    shift: Tensor,
    scale: Tensor,
    dtype: ModulationDtype | None = None,
) -> Tensor:
    """Return ``x * scale + shift`` in the dtype of ``x``.

    ``"mixed"`` and ``"float32"`` compute in float32. ``"bfloat16"`` computes
    in bf16. ``None`` uses the normal type promotion. With ``"mixed"``, 3-D
    CUDA inputs of one shape (or ``shift`` and ``scale`` of shape
    ``(B, 1, C)``) go to a Triton kernel. That kernel can fuse the multiply
    and the add, so its rounding is not the rounding of the eager path.
    """
    compute: Literal["bfloat16", "float32"] | None = (
        "float32" if dtype == "mixed" else dtype
    )
    case = _fused_modulate_case(x, shift, scale) if dtype == "mixed" else None
    if case is None:
        return _modulate_eager(x, shift, scale, compute)
    return modulate_fused(x, shift, scale, rows=case == "rows")


# ---------------------------------------------------------------------------
# Normalization and rotary embedding
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm. The scale is ``1 + weight``.

    With ``cast_dtype`` set, the input and the weight are cast to that dtype,
    the norm runs in that dtype, and the output is cast back.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        cast_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.cast_dtype = cast_dtype
        self.weight = nn.Parameter(torch.zeros(dim)) if elementwise_affine else None
        self._norm_weight_cache: tuple | None = None

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        self._norm_weight_cache = None
        return super()._apply(fn, recurse)

    def _inference_weight(self) -> Tensor | None:
        """Reuse the FP32 zero-centred weight until the parameter changes."""
        weight = self.weight
        if weight is None:
            return None
        try:
            version = weight._version
        except RuntimeError:
            # Inference tensors have no version counter. Do not cache them.
            return 1 + weight.to(torch.float32)
        key = (weight.data_ptr(), version, weight.device, weight.dtype)
        if self._norm_weight_cache is None or self._norm_weight_cache[0] != key:
            self._norm_weight_cache = (key, 1 + weight.to(torch.float32))
        return self._norm_weight_cache[1]

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        orig_dtype = x.dtype
        if self.cast_dtype is not None:
            x = x.to(self.cast_dtype)
            if weight is not None:
                weight = weight.to(self.cast_dtype)
        if weight is not None:
            weight = 1 + weight
        out = F.rms_norm(x, self.normalized_shape, weight=weight, eps=self.eps)
        if self.cast_dtype is not None:
            out = out.to(orig_dtype)
        return out


def rms_norm_scale(norm: RMSNorm, x: Tensor, scale: Tensor) -> Tensor:
    """Plain normalization and scaling for unsupported or unfused calls."""
    return norm(x) * scale


def residual_norm_scale(
    x: Tensor, delta: Tensor, norm: RMSNorm, scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Plain residual addition and normalization for fallback calls."""
    residual = x + delta
    return residual, norm(residual) * scale


class RotaryEmbedding(nn.Module):
    """Rotary (cos, sin) table for integer positions.

    Only the first ``effective_dim // 2`` frequency pairs rotate. The other
    pairs have zero frequency and pass through unchanged.

    A position is split as ``high * table_size + low``. For each part the
    angle is a float64 product, and cos and sin are taken in float64 on the
    device and rounded to float32. The two parts are then combined with the
    angle-sum identity in float64 and rounded to float32. This equals a lookup
    in precomputed ``table_size``-row tables, bit for bit.
    """

    def __init__(
        self,
        dim: int = 128,
        effective_dim: int = 64,
        base: int = 26_000_000,
        table_size: int = 1_000_000,
    ) -> None:
        super().__init__()
        assert effective_dim <= dim
        self.dim = dim
        self.effective_dim = effective_dim
        self.base = base
        self.table_size = table_size
        # The frequencies are computed on the CPU in float32. This plain
        # attribute is not a buffer, so module casts and moves do not touch it.
        exponent = (
            torch.arange(0, effective_dim, 2, device="cpu")[
                : effective_dim // 2
            ].float()
            / effective_dim
        )
        theta = 1.0 / (base**exponent)
        pad = torch.zeros((dim - effective_dim) // 2, device="cpu")
        self.theta = torch.cat([theta, pad], dim=0)

    @staticmethod
    def _cos_sin(angle: Tensor) -> tuple[Tensor, Tensor]:
        """Return float64 cos and sin, each rounded to float32 and back."""
        return angle.cos().float().double(), angle.sin().float().double()

    def forward(self, positions: Tensor) -> Tensor:
        """Map ``positions`` (B, N) to a float32 table (B, N, dim // 2, 2)."""
        if CACHE_DEVICE_TABLES:
            # The same float64 values, copied to the device once instead of a
            # blocking host-to-device copy on every call.
            cache = self.__dict__.setdefault("_theta_on", {})
            key = (positions.device, id(self.theta))
            theta = cache.get(key)
            if theta is None:
                theta = cache[key] = self.theta.to(positions.device).double()
        else:
            theta = self.theta.to(positions.device).double()
        high = positions // self.table_size
        low = positions % self.table_size
        # Table rows hold float32 positions. Both casts below are exact.
        high_pos = (high.to(torch.float32) * self.table_size).double()
        low_pos = low.to(torch.float32).double()
        cos_high, sin_high = self._cos_sin(high_pos[..., None] * theta)
        cos_low, sin_low = self._cos_sin(low_pos[..., None] * theta)
        cos = cos_high * cos_low - sin_high * sin_low
        sin = sin_high * cos_low + cos_high * sin_low
        table = torch.stack([cos, sin], dim=-1).float()
        if table.dim() == 3:
            table = table.unsqueeze(0)
        return table


# True: RotaryEmbedding keeps its frequencies, and the chunk-causal towers their
# rotary position ids, on the device after the first call (the same values; no
# blocking host-to-device copy per call).
CACHE_DEVICE_TABLES = True

# True: apply_rope runs the fused Triton kernel of praxis.kernels.rope_triton,
# which keeps the eager float32 op order and rounding (same bits).
FUSED_ROPE = True

# ---------------------------------------------------------------------------
# Inference optimizations. Portable kernels check their supported inputs.
# Fused expert GEMMs remain opt-in after H100 and B200 policy measurements.
# Optional SM110 kernels retain their device guards and default to off.
# ---------------------------------------------------------------------------

# Fixed tiles for paired expert GEMMs. Each output row keeps its own input
# row accumulation order when two guidance passes share a launch.
PAIR_GEMM_CONFIG = {"BN": 256, "num_stages": 4}

# Fuse RoPE and Q/K RMSNorm while preserving rotation rounding and the native
# FP32 reduction order. None selects supported BF16/FP16 CUDA inference.
# Explicit booleans override selection. Gradients, autocast and hooks fall back.
FUSED_ROPE_QK_NORM: bool | None = None

# Fuse residual add, RMSNorm and layer scaling with native reduction order and
# BF16 rounding. Unmodulated stacks can defer a residual add into the next norm.
# None selects supported BF16 CUDA inference, including NVFP4 activations.
# Explicit booleans override selection. Gradients, autocast and hooks fall back.
FUSED_NORM: bool | None = None

# Merge attention heads and apply the gate in one elementwise kernel.
FUSED_GATE = True

# Replace the current key/value rows in place. Other cached rows stay intact.
KV_INPLACE = True

# Preserve full attention-gate and router GEMM shapes during context reuse.
# Generator inputs carry their row intervals through paired expert dispatch.
CTX_FULL_ROWS = True
_ROUTER_ROWS: dict[int, tuple[int, int, int]] = {}

# Reuse FP32 copies written by fused normalization for router GEMMs.
ROUTER_F32 = True

# Fused MoE routing and dispatch (CuTe DSL, praxis.kernels.router_fused): after
# the ordinary FP32 router GEMM, sigmoid + bias + top-8 + gates + the dispatch
# metadata (order, inverse, sorted gates, tokens, offsets, pair segments) run
# as one barrier-free kernel instead of about ten PyTorch ops. Reproduces
# torch.topk's slot order (including ties) and the gate-sum order of the
# pinned torch (2.13) bitwise; a torch upgrade needs the routing audit rerun. Default on for Blackwell tcgen05 and Thor only;
# other architectures keep ordinary routing. Requires CuTe DSL.
ROUTER_FUSED = True
# Probe only: a JSON path (True: outputs/router_fused_audit.json). Every fused
# routing also runs the ordinary ops, the layer continues with the ordinary
# values, and per-layer agreement is written at exit.
ROUTER_FUSED_AUDIT = ""

# Fuse paired expert gather, gate/up projection, SwiGLU and output combine.
# Opt-in on every architecture. Set before warmup and graph capture.
FUSED_MOE: bool | None = False

# Fuse unchunked expert combine, scaling and shared-expert addition.
# Chunked layers already fuse these operations. None enables supported CUDA
# calls except SM90. Explicit booleans override selection.
FUSED_MOE_SOLO: bool | None = None

# Use fused expert GEMMs outside the paired driver. Opt-in on every
# architecture after H100 and B200 policy regressions. DEC_*_CONFIG controls
# tiles; empty dictionaries restore kernel defaults. Unsupported calls fall back.
FUSED_MOE_DEC: bool | None = False
# Measured tile defaults for the portable Triton expert kernels. Explicit {}
# restores the kernel defaults for comparisons on any architecture.
DEC_GATE_UP_CONFIG: dict = {"BM": 256, "BN": 128, "num_warps": 8, "num_stages": 3}
DEC_DOWN_CONFIG: dict = {
    "BM": 256,
    "BN": 256,
    "BK": 64,
    "num_warps": 8,
    "num_stages": 3,
    "USE_TMA": False,
}

# Run the decoder's two guidance passes block by block in lockstep, like the
# paired encoder, so each MoE layer reads its routed weights once for both
# passes (MoEFeedForward.forward_pair: the FUSED_MOE / CUTE_EXPERTS BF16 pair
# path or PAIR_NVFP4). Routing, shared expert and per-pass combine keep the
# unpaired roundings. Needs Session.pair_cfg and both decoder passes. Opt-in.
PAIR_DEC = False

# Fuse the shared expert SiLU and product after its ordinary GEMM.
SHARED_SILU_MUL = True

# Optional Thor (SM110) shared-expert CuTe GEMM with fused SiLU and product.
# Requires Quack. Smaller inputs retain the ordinary GEMM and elementwise path.
SHARED_QUACK = False
SHARED_QUACK_MIN_ROWS = 2048

# Select paired gate/up GEMM tiles by routed row count. Explicit call-level
# configurations take precedence over this selection.
ENC_GATE_UP_AUTO = True
ENC_GATE_UP_BM256 = {"BM": 256, "BN": 128, "num_warps": 8, "num_stages": 3}
ENC_GATE_UP_BM128 = {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 4}

# Optional Thor (SM110) routed down projection using a persistent CuTe GEMM.
# Requires Quack and an explicit opt-in.
ROUTED_QUACK_DOWN = False

# Optional Thor (SM110) CuTe shared and routed expert GEMMs. Requires Quack.
# Gate/up projections fuse SwiGLU. Routed weights get a cached K-major layout
# on first use. Explicit opt-in; ordinary CUDA paths remain available.
CUTE_EXPERTS = False

# Dense SwiGLU feed-forward (encoder blocks 0-1, decoder block 0) on tcgen05
# GPUs (SM100/SM103/SM110): gate|up and F.silu(gate) * up as one CuTe GEMM with
# the eager roundings, so the gate/up outputs never reach memory
# (praxis.kernels.cute_dense, needs Quack). Rows dispatch through a per-
# architecture table; only SM110 has a measured one, other GPUs run the
# ordinary ops until scripts/tune_cute_dense.py fills theirs. The down
# projection stays cuBLAS. a/b weights become views of one concatenated buffer
# on first use (same values). Bitwise equal to the ordinary module. Opt-in.
CUTE_DENSE_FFN = False

# Fuse decoder residual add, modulation, RMSNorm and layer scaling.
# The preceding feed-forward residual add can be deferred into this kernel.
FUSED_MOD_NORM = True

# Write paired feed-forward inputs into adjacent buffer slices, so expert
# dispatch can read the merged rows without concatenation.
PAIR_BUFFER = True

# Opt-in (with FUSED_NORM): a mixture-of-experts feed-forward driven by TransformerBlock.forward_gen in
# the chunked regime (BF16 FUSED_MOE / FUSED_MOE_DEC paths, NVFP4 FUSED_NVFP4 / PAIR_NVFP4 one-chunk paths on
# SM 11.0) returns its routed tail unevaluated (MoETail: the expert rows, the inverse permutation, the gates and
# the shared-expert rows); when the block defers its output, the next block's add_norm_in / add_mod_norm_in
# computes the token combine + shared add + residual add + norm in one kernel; a block that does not defer (tap
# blocks, the last block) runs combine + shared add + residual add in one kernel (praxis.kernels.moe_tail_triton;
# the roundings of the stock combine, the shared add, the residual add and the norm, same bits). Anywhere else the
# tail is materialized with the stock kernels. Default off.
MOE_TAIL_NORM = False
# data_ptr of the MoE feed-forward inputs yielded by TransformerBlock.forward_gen (MOE_TAIL_NORM), and whether the
# MoE call in progress may return MoETail objects.
_TAIL_INPUTS: set[int] = set()
_TAIL_NOW = False


class MoETail:
    """An unevaluated MoE feed-forward output (``MOE_TAIL_NORM``): ``materialize()`` returns the tensor
    the stock path returns (same kernels). ``mode`` 1: BF16 chunked combine (``moe_chunked_combine`` with the
    shared add); 2: NVFP4 one-chunk ``combine_rescale`` + shared; 3: NVFP4 CFG pair ``_combine_rescale_seg`` +
    shared. ``inverse`` holds this output's ``8 * tokens`` sorted-row indices first; ``gate`` is the bf16
    sorted gate (mode 1) or the float32 sorted scale (2, 3)."""

    __slots__ = (
        "csr",
        "expert",
        "g",
        "gate",
        "inverse",
        "mode",
        "pcsr",
        "scaling",
        "seg",
        "shape",
        "shared",
        "w",
    )

    def __init__(
        self,
        mode,
        expert,
        inverse,
        gate,
        shared,
        scaling,
        shape,
        seg=None,
        csr=None,
        pcsr=None,
        g=None,
        w=None,
    ):
        self.mode, self.expert, self.inverse, self.gate = mode, expert, inverse, gate
        self.shared, self.scaling, self.shape = shared, scaling, shape
        self.seg, self.csr, self.pcsr, self.g, self.w = seg, csr, pcsr, g, w

    @property
    def dtype(self) -> torch.dtype:
        return self.shared.dtype

    def materialize(self) -> Tensor:
        t, c = self.shared.shape
        if self.mode == 1:
            with NV.op("moe", "combine"):
                y = moe_chunked_combine(
                    self.expert,
                    self.gate,
                    self.inverse,
                    8,
                    float(self.scaling),
                    shared=self.shared,
                )
            return y.reshape(self.shape)
        if self.mode == 2:
            from praxis.kernels.nvfp4_fused import combine_rescale

            result = combine_rescale(
                self.expert,
                self.inverse,
                self.gate,
                self.csr,
                self.pcsr,
                self.g,
                self.w,
                t,
                8,
            )
        else:
            from praxis.kernels.nvfp4_pair import combine_pass

            result = combine_pass(
                self.expert,
                self.inverse,
                self.gate,
                self.seg,
                self.csr,
                self.pcsr,
                self.g,
                self.w,
                t,
            )
        return (result + self.shared).reshape(self.shape)


def _tail_wanted(xs) -> bool:
    """MOE_TAIL_NORM and every input is a registered deferred feed-forward input."""
    return (
        MOE_TAIL_NORM
        and bool(_TAIL_INPUTS)
        and all(x.data_ptr() in _TAIL_INPUTS for x in xs)
    )


def _tail_bf16_ok(out, sorted_gate, inverse, shared, k) -> bool:
    """The inputs on which moe_chunked_combine(..., shared=) runs its fast kernel (the one the tail kernel
    reproduces), for bf16 rows."""
    return (
        k == 8
        and out.is_cuda
        and out.dtype == torch.bfloat16
        and sorted_gate.dtype == out.dtype
        and inverse.dtype == torch.int64
        and out.shape[1] >= 128
        and out.is_contiguous()
        and sorted_gate.is_contiguous()
        and inverse.is_contiguous()
        and all(
            s.dtype == out.dtype
            and s.dim() == 2
            and s.shape[1] == out.shape[1]
            and s.is_contiguous()
            for s in shared
        )
        and not torch.is_autocast_enabled()
        and not torch.is_grad_enabled()
    )


def _tail_mat(p):
    """``p`` as a tensor (a MoETail is materialized)."""
    return p.materialize() if isinstance(p, MoETail) else p


# Portable torch + Triton: the decoder keeps its per-token conditioning as its distinct rows (CompactMod: the n_reg zero rows
# and the cf*h*w cond rows; the stock x_mod repeats each cond row ph*pw times), each block's modulator GEMM runs on
# those rows only, and the FUSED_MOD_NORM kernels (add_mod_norm_scale, MOE_TAIL_NORM's tail variant) read the
# shift/scale of token r from compact row mod_row(r). Same values: each GEMM row depends only on its own input row
# (checked bitwise on Thor at the decoder shapes). Any other consumer gets the
# stock tensors (CompactMod.full / .expand). The modulator GEMM
# runs on 4x fewer rows (checkpoint weights, bitwise). Default on.
MOD_COMPACT = True


class CompactMod:
    """MOD_COMPACT: the decoder modulation input ``cat([zeros(n_reg), up(cond)])`` as its distinct rows.

    ``rows`` (B, n_reg + cf*h*w, C): the pad rows then ``cond`` (B, cf, h, w, C) flattened; ``geom`` the kernel
    ints (RF, RC, n_reg, pw, w, ph) of ``norm_triton.mod_row``."""

    __slots__ = ("_cond", "_idx", "_ph", "_pw", "geom", "rows")

    def __init__(
        self, rows: Tensor, geom: tuple, cond: Tensor, ph: int, pw: int
    ) -> None:
        self.rows, self.geom, self._cond, self._ph, self._pw, self._idx = (
            rows,
            geom,
            cond,
            ph,
            pw,
            None,
        )

    @classmethod
    def decoder(cls, cond: Tensor, n_reg: int, ph: int, pw: int) -> CompactMod:
        b, cf, h, w, c = cond.shape
        pad = torch.zeros(b, n_reg, c, device=cond.device, dtype=cond.dtype)
        rows = torch.cat([pad, cond.reshape(b, cf * h * w, c)], dim=1)
        rf = n_reg + cf * h * ph * w * pw
        return cls(rows, (rf, n_reg + cf * h * w, n_reg, pw, w, ph), cond, ph, pw)

    def full(self) -> Tensor:
        """The stock modulation input (DecoderTower._chunk_causal_prep without the flag)."""
        cond, ph, pw = self._cond, self._ph, self._pw
        b, cf, h, w, c = cond.shape
        up = cond[:, :, :, None, :, None, :].expand(b, cf, h, ph, w, pw, c)
        up = up.reshape(b, cf * h * ph * w * pw, c)
        pad = torch.zeros(b, self.geom[2], c, device=up.device, dtype=up.dtype)
        return torch.cat([pad, up], dim=1)

    def index(self) -> Tensor:
        """(RF,) int64: the compact row of each full row (norm_triton.mod_row)."""
        if self._idx is None:
            rf, _, nreg, pw, wd, ph = self.geom
            t = torch.arange(rf, device=self.rows.device)
            q = (t - nreg).clamp(min=0) // pw
            self._idx = torch.where(t < nreg, t, nreg + (q // wd) // ph * wd + q % wd)
        return self._idx

    def expand(self, y: Tensor) -> Tensor:
        """The full-row tensor of a row-wise result ``y`` (B, RC, ...) of the compact rows (same values)."""
        return y.index_select(1, self.index())


def _full_mod(x_mod):
    """The stock modulation input of ``x_mod`` (a CompactMod is expanded)."""
    return x_mod.full() if isinstance(x_mod, CompactMod) else x_mod


_PAIR_SLOTS: dict | None = None


def _pair_slot(like: Tensor) -> Tensor | None:
    """The slice of the pair buffer for the next pass's feed-forward input (see
    PAIR_BUFFER). The passes call it in turn, once per block each; the row
    counts of the previous block size the buffer."""
    st = _PAIR_SLOTS
    if st is None:
        return None
    rows = like.numel() // like.size(-1)
    i = st["i"]
    st["i"] = (i + 1) % st["n"]
    st["seen"][i] = rows
    sizes = st["sizes"]
    if i == st["n"] - 1:
        st["sizes"] = list(st["seen"])
    if sizes is None or sizes[i] != rows:
        if i == 0:
            st["buf"] = None
        return None
    if i == 0:
        st["buf"] = like.new_empty(sum(sizes), like.size(-1))
    buf = st["buf"]
    if buf is None or buf.dtype != like.dtype or buf.device != like.device:
        return None
    off = sum(sizes[:i])
    return buf[off : off + rows].view(like.shape)


def apply_rope(q: Tensor, k: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate ``q`` and ``k`` (B, H, N, D) in float32.

    Channel ``j`` pairs with channel ``j + D // 2``. The outputs have the
    input dtype.
    """
    if FUSED_ROPE and q.is_cuda:
        from praxis.kernels.rope_triton import rope_fused

        return rope_fused(q, k, table, _apply_rope_eager)
    return _apply_rope_eager(q, k, table)


def _apply_rope_eager(q: Tensor, k: Tensor, table: Tensor) -> tuple[Tensor, Tensor]:
    """The eager float32 rotation of :func:`apply_rope`."""
    orig_dtype = q.dtype
    q0, q1 = q.float().chunk(2, dim=-1)
    k0, k1 = k.float().chunk(2, dim=-1)
    seq_len, half = q0.shape[2], q0.shape[3]
    cos, sin = table.float().chunk(2, dim=-1)
    assert cos.shape[1] == seq_len and cos.shape[2] == half and cos.shape[3] == 1
    cos = cos.reshape(-1, 1, seq_len, half)
    sin = sin.reshape(-1, 1, seq_len, half)
    q_out = torch.stack([q0 * cos - q1 * sin, q1 * cos + q0 * sin], dim=-2)
    k_out = torch.stack([k0 * cos - k1 * sin, k1 * cos + k0 * sin], dim=-2)
    return q_out.flatten(-2).to(orig_dtype), k_out.flatten(-2).to(orig_dtype)


def _fused_moe_enabled(setting: bool | None, x: Tensor) -> bool:
    """Keep the measured SM90 expert path unless fusion is explicitly selected."""
    if setting is not None:
        return setting
    return x.is_cuda and torch.cuda.get_device_capability(x.device) != (9, 0)


def _fused_rope_qk_norm_enabled(q: Tensor) -> bool:
    """Select fusion without an architecture allowlist; preserve overrides."""
    if FUSED_ROPE_QK_NORM is not None:
        return FUSED_ROPE_QK_NORM
    return (
        not torch.is_grad_enabled()
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and not torch.is_autocast_enabled()
    )


def _rope_qk_norm_ok(q_norm: RMSNorm, k_norm: RMSNorm) -> bool:
    """Plain RMSNorm modules without hooks, outside autocast."""
    return (
        not torch.is_autocast_enabled()
        and not torch.nn.modules.module._global_forward_hooks
        and not torch.nn.modules.module._global_forward_pre_hooks
        and all(
            type(norm) is RMSNorm
            and not norm._forward_hooks
            and not norm._forward_pre_hooks
            and "forward" not in vars(norm)
            for norm in (q_norm, k_norm)
        )
    )


def apply_rope_norm(
    q: Tensor, k: Tensor, table: Tensor | None, q_norm: RMSNorm, k_norm: RMSNorm
) -> tuple[Tensor, Tensor]:
    """Fuse rotary embedding and Q/K RMSNorm on supported inference inputs."""
    if (
        FUSED_ROPE
        and not torch.is_grad_enabled()
        and table is not None
        and q.is_cuda
        and _fused_rope_qk_norm_enabled(q)
    ):
        from praxis.kernels.rope_triton import rope_norm_fused, rope_norm_supported

        if _rope_qk_norm_ok(q_norm, k_norm) and rope_norm_supported(
            q, k, table, q_norm, k_norm
        ):
            return rope_norm_fused(q, k, table, q_norm, k_norm, apply_rope)
    if table is not None:
        q, k = apply_rope(q, k, table)
    return q_norm(q), k_norm(k)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class QKVProjection(nn.Module):
    """Separate bias-free projections for queries, keys and values."""

    def __init__(
        self, dim: int, head_dim: int, num_heads: int, num_kv_heads: int
    ) -> None:
        super().__init__()
        self.q_weight = nn.Parameter(torch.empty(num_heads * head_dim, dim))
        self.k_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))
        self.v_weight = nn.Parameter(torch.empty(num_kv_heads * head_dim, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = F.linear(x, self.q_weight)
        k = F.linear(x, self.k_weight)
        v = F.linear(x, self.v_weight)
        return q, k, v


def _pad_context_rows(x: Tensor, rows: tuple[int, int, int]) -> Tensor:
    """Restore the full token dimension with zeros in the unused rows."""
    r0, r1, total = rows
    return F.pad(x, (0, 0, r0, total - r1))


class KVSplice:
    """Keys and values of one attention layer across the Euler steps of a call.

    With ``rows`` None (the first step) the layer stores its full keys and
    values. With ``rows = (r0, r1)`` the layer ran only on the query rows
    ``r0:r1`` and its keys and values for those rows replace the stored ones;
    the rows outside are context tokens whose keys and values do not change
    between the steps of a call.
    """

    def __init__(self, store: dict, key: int, rows: tuple[int, int] | None) -> None:
        self.store, self.key, self.rows = store, key, rows

    def full_rows(self) -> tuple[int, int, int] | None:
        """Return the query interval and original token count on cache reuse."""
        if self.rows is None:
            return None
        return (*self.rows, self.store[self.key][0].shape[2])

    def __call__(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.rows is None:
            self.store[self.key] = (k, v)
            return k, v
        r0, r1 = self.rows
        ck, cv = self.store[self.key]
        if KV_INPLACE and ck.shape[:2] == k.shape[:2] and cv.shape[:2] == v.shape[:2]:
            ck[:, :, r0:r1].copy_(k)
            cv[:, :, r0:r1].copy_(v)
            return ck, cv
        k = torch.cat([ck[:, :, :r0], k, ck[:, :, r1:]], dim=2)
        v = torch.cat([cv[:, :, :r0], v, cv[:, :, r1:]], dim=2)
        return k, v


class Attention(nn.Module):
    """Grouped-query self-attention with a sigmoid gate per head.

    The order is: project, rotate q and k, RMS-normalize q and k per head,
    run flex attention, multiply by the gate, project out.
    """

    def __init__(
        self,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        backend: AttentionBackend = "triton",
    ):
        super().__init__()
        if backend not in ("triton", "flash", "cudnn"):
            raise ValueError(f"unknown attention backend {backend!r}")
        self.backend: AttentionBackend = backend
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.num_kv_heads = num_kv_heads
        self.in_proj = QKVProjection(dim, head_dim, self.num_heads, num_kv_heads)
        self.gate_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, dim)

    def effective_backend(
        self,
        device: torch.device | str,
        block_mask: BlockMask | None = None,
        *,
        block_size: int | None = None,
    ) -> AttentionBackend:
        """Use the existing Triton path where Flash cannot run block sparsity.

        SM 12.x: always (SM120 rule). SM 11.x (Thor): when the mask's block size is
        not a multiple of 256, where FA4 block sparsity raises. ``block_size``
        stands for the block size of a mask that is not built yet.
        """
        device = torch.device(device)
        if self.backend == "flash" and device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            if major == 12:
                return "triton"
            if major == 11:
                bs = block_size if block_mask is None else block_mask.BLOCK_SIZE
                bs = bs[0] if isinstance(bs, tuple) else bs
                if bs is None or bs % 256:
                    return "triton"
        return self.backend

    def backend_metadata(self, device: torch.device | str) -> dict[str, str | None]:
        """Report the requested and actual backend without changing the config."""
        effective = self.effective_backend(device)
        return {
            "requested": self.backend,
            "effective": effective,
            "reason": "Flash block sparsity is unsupported for this architecture or block size"
            if effective != self.backend
            else None,
        }

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | None,
        kv: KVSplice | None = None,
    ) -> Tensor:
        b, s, _ = x.shape
        d = self.head_dim
        backend = self.effective_backend(x.device, block_mask)
        q, k, v = self.in_proj(x)
        rows = (
            kv.full_rows()
            if kv is not None and (backend == "flash" or CTX_FULL_ROWS)
            else None
        )
        if rows is None:
            g = self.gate_proj(x).sigmoid()
        else:
            # Keep the original GEMM shape: fewer rows can change its bits.
            g = self.gate_proj(_pad_context_rows(x, rows))[:, rows[0] : rows[1]]
            g = g.sigmoid()
        g = g.permute(0, 2, 1).reshape(b, self.num_heads, s, 1)
        q = q.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        k = k.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        v = v.reshape(b, s, -1, d).permute(0, 2, 1, 3)
        with NV.op("attn", "qk_norm_rope"):
            q, k = apply_rope_norm(q, k, rope, self.q_norm, self.k_norm)
        if kv is not None:
            with NV.op("attn", "kv_cache"):
                k, v = kv(k, v)
        attention = _compiled_flex_attention
        options = {"kernel_options": None}
        if backend != "triton":
            from praxis.attention import optional_attention

            attention = partial(optional_attention, backend)
            options = {}
        with NV.op("attn", "attn_core"):
            out = attention(  # ty: ignore[no-matching-overload]
                q,
                k,
                v,
                block_mask=block_mask,
                scale=None,
                enable_gqa=k.shape[1] != q.shape[1],
                **options,
            )
        if FUSED_GATE:
            from praxis.kernels import norm_triton as NT

            if NT.gate_merge_supported(out, g):
                with NV.op("attn", "gate_merge"):
                    merged = NT.gate_merge(out, g)
                with NV.op("attn", "out_proj"):
                    return self.out_proj(merged)
        with NV.op("attn", "gate_merge"):
            out = out * g
            out = out.permute(0, 2, 1, 3).reshape(b, s, -1)
        with NV.op("attn", "out_proj"):
            return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-forward layers
# ---------------------------------------------------------------------------


class SwiGLUProjection(nn.Module):
    """The two bias-free input projections of a SwiGLU."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.a_weight = nn.Parameter(torch.empty(hidden, dim))
        self.b_weight = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return F.linear(x, self.a_weight), F.linear(x, self.b_weight)


class SwiGLU(nn.Module):
    """``silu(x @ a.T) * (x @ b.T)``."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.linear = SwiGLUProjection(dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.linear(x)
        return F.silu(a) * b


class SwiGLUFeedForward(nn.Sequential):
    """Dense SwiGLU feed-forward: ``[0]`` is the SwiGLU, ``[1]`` projects out."""

    def __init__(self, dim: int = 3072, hidden: int = 11520) -> None:
        super().__init__(SwiGLU(dim, hidden), nn.Linear(hidden, dim, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        if CUTE_DENSE_FFN:
            h = self._cute_hidden(x)
            if h is not None:
                with NV.op("ffn", "down"):
                    return self._down(h)
        if not NV.ON:
            return super().forward(x)
        # PRAXIS_NVTX=1 only: the same three ops with a range each.
        with NV.op("ffn", "gate_up"):
            a, b = self[0].linear(x)
        with NV.op("ffn", "activation"):
            h = F.silu(a) * b
        with NV.op("ffn", "down"):
            return self[1](h)

    def _cute_hidden(self, x: Tensor) -> Tensor | None:
        """CUTE_DENSE_FFN: F.silu(x A^T) * (x B^T) as one CuTe GEMM, or None
        where it does not apply (the ordinary operations run)."""
        from praxis.kernels import cute_dense as CD

        proj = self[0].linear
        rows = x.numel() // x.shape[-1] if x.dim() >= 2 else 0
        if rows == 0 or not CD.supported():
            return None
        cfg = CD.gate_up_config(rows)
        if cfg is None or not CD.ok(x, proj.a_weight, proj.b_weight):
            return None
        x2 = x.reshape(rows, x.shape[-1])
        if x2.stride(0) != x2.shape[1]:
            x2 = x2.contiguous()
        with NV.op("ffn", "gate_up_act"):
            h = CD.gate_up_silu(x2, CD.packed_weight(proj.a_weight, proj.b_weight), cfg)
        return h.view(*x.shape[:-1], h.shape[-1])

    def _down(self, h: Tensor) -> Tensor:
        """The down projection: nn.Linear (cuBLAS), or a Quack GEMM where the
        architecture's cute_dense.DOWN_TABLES entry lists one."""
        from praxis.kernels import cute_dense as CD

        cfg = CD.down_config(h.numel() // h.shape[-1]) if CD.has_down_table() else None
        if cfg is None:
            return self[1](h)
        y = CD.down(h.reshape(-1, h.shape[-1]), self[1].weight, cfg)
        return y.view(*h.shape[:-1], y.shape[-1])


class RoutedExperts(nn.Module):
    """Expert weight banks, expert-first: gate|up (E, C, 2H), down (E, H, C)."""

    def __init__(self, num_experts: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_weight = nn.Parameter(torch.empty(num_experts, dim, 2 * hidden))
        self.down_weight = nn.Parameter(torch.empty(num_experts, hidden, dim))


class SharedExpert(nn.Module):
    """The always-on expert, a dense SwiGLU with packed gate|up weights."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Parameter(torch.empty(dim, 2 * hidden))
        self.down = nn.Parameter(torch.empty(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        with NV.op("moe", "shared_expert"):
            return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        if (
            CUTE_EXPERTS
            and is_thor(x.device)
            and x.is_cuda
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and self.gate_up.dtype == torch.bfloat16
            and self.down.dtype == torch.bfloat16
            and x.stride(1) == 1
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        ):
            from praxis.kernels import cute_experts as CE

            if CE.supported(x.device):
                return CE.shared_down(
                    CE.shared_gate_up_silu(x, CE.kmajor_cached(self.gate_up)), self.down
                )
        if (
            SHARED_QUACK
            and is_thor(x.device)
            and x.shape[0] >= SHARED_QUACK_MIN_ROWS
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        ):
            from praxis.kernels import shared_quack as SQ

            if SQ.supported(x, self.gate_up):
                return SQ.gate_up_silu(x, self.gate_up) @ self.down
        gate_up = x @ self.gate_up
        if (
            SHARED_SILU_MUL
            and gate_up.is_cuda
            and gate_up.dim() == 2
            and gate_up.stride(1) == 1
            and gate_up.dtype in (torch.bfloat16, torch.float16)
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        ):
            from praxis.kernels.moe_gemm_triton import silu_mul

            return silu_mul(gate_up) @ self.down
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down


def _expert_swiglu(gate_up: Tensor) -> Tensor:
    """``silu(gate) * up`` of the routed experts.

    On CUDA this runs the custom kernel of the original implementation. Its sigmoid rounds
    differently from ``F.silu``.
    """
    if gate_up.is_cuda:
        return swiglu_packed(gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _router_logits_forward(
    weight: Tensor, x: Tensor, xf: Tensor | None = None, out: Tensor | None = None
) -> Tensor:
    w = weight.to(torch.float32)
    if out is None:
        out = x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.float32)
    for start in range(0, x.shape[0], _ROUTER_CHUNK_ROWS):
        end = min(start + _ROUTER_CHUNK_ROWS, x.shape[0])
        # ROUTER_F32: xf holds the same rows, already widened.
        rows = x[start:end].to(torch.float32) if xf is None else xf[start:end]
        out[start:end] = F.linear(rows, w)
    return out


# ROUTER_F32: float32 copies of feed-forward inputs, written by the fused
# norm kernel next to its bf16 output, keyed by the bf16 tensor's data pointer.
# Each entry holds the bf16 tensor, so no other tensor can take its address
# while the entry exists; the router pops the entry when it uses it.
_F32_ROWS: dict[int, tuple[Tensor, Tensor]] = {}


def _put_f32_rows(h: Tensor, hf: Tensor) -> None:
    if len(_F32_ROWS) > 8:
        _F32_ROWS.clear()
    _F32_ROWS[h.data_ptr()] = (h, hf)


def _take_f32_rows(x: Tensor) -> Tensor | None:
    """The float32 copy of the 2-D bf16 ``x``, if the fused norm wrote one."""
    hit = _F32_ROWS.pop(x.data_ptr(), None) if _F32_ROWS else None
    if hit is None:
        return None
    h, hf = hit
    if not (
        x.is_contiguous()
        and x.dim() == 2
        and x.numel() == h.numel()
        and x.dtype == h.dtype
        and x.untyped_storage().data_ptr() == h.untyped_storage().data_ptr()
    ):
        return None
    return hf.view(x.shape)


class _RouterLogits(torch.autograd.Function):
    """The router GEMM with the backward of the original training code.

    The backward runs two float32 GEMMs per row chunk:
    ``dx = (g @ w).to(x.dtype)`` and ``dw += g.T @ x``. ``dw`` starts at zero
    in float32 and rounds once to the dtype of the weight.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: Tensor, weight: Tensor
    ) -> Tensor:
        ctx.save_for_backward(x, weight)
        return _router_logits_forward(weight, x)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, weight = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        w = weight.to(torch.float32)
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, x.shape[0], _ROUTER_CHUNK_ROWS):
            end = min(start + _ROUTER_CHUNK_ROWS, x.shape[0])
            g = grad[start:end].to(torch.float32)
            grad_x[start:end] = (g @ w).to(x.dtype)
            grad_w += g.t() @ x[start:end].to(torch.float32)
        return grad_x, grad_w.to(weight.dtype)


def _router_logits(weight: Tensor, x: Tensor) -> Tensor:
    """Float32 router logits, computed over fixed row chunks."""
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _RouterLogits.apply(x, weight)
    if ROUTER_F32:
        return _router_logits_forward(weight, x, _take_f32_rows(x))
    return _router_logits_forward(weight, x)


# Routing audit (ROUTER_FUSED_AUDIT): per layer, the fused kernel's outputs vs the stock ops on the same logits.
_FUSED_AUDIT: dict = {}
_FUSED_AUDIT_IDS: dict = {}
_FUSED_AUDIT_ROWS: list = []


def _fused_audit(layer: MoEFeedForward, logits: Tensor, routing, path: str):
    """Compare ``routing`` (router_fused) with the stock routing + dispatch of ``logits``; return a Routing of
    the STOCK values (the layer continues with them)."""
    import atexit

    from praxis.kernels import router_fused as RF
    from praxis.kernels.layers_triton import moe_dispatch_metadata

    if not _FUSED_AUDIT:
        atexit.register(_fused_audit_dump)
    rows = routing.rows
    n = _FUSED_AUDIT_IDS.setdefault(id(layer), len(_FUSED_AUDIT_IDS))
    key = f"m{n:02d}.P{len(rows)}.{path}"
    tops, gates = [], []
    t0 = 0
    for r in rows:
        ti, gt = RF.route_ref(logits[t0 : t0 + r], layer.expert_bias, layer.top_k)
        tops.append(ti)
        gates.append(gt)
        t0 += r
    top = torch.cat(tops) if len(tops) > 1 else tops[0]
    gate = torch.cat(gates) if len(gates) > 1 else gates[0]
    ref = RF.dispatch_ref(top, gate, rows, layer.num_experts)
    order = top.reshape(-1).argsort(stable=True)
    sg, inv, offs = moe_dispatch_metadata(top, gate, order, layer.num_experts)
    i16 = torch.int16
    top_bad = (top != routing.top_idx).any(-1)
    gate_bad = (gate.view(i16) != routing.gate.view(i16)).any(-1)
    f = {
        "order": torch.equal(ref.order, routing.order),
        "inverse": torch.equal(ref.inverse, routing.inverse),
        "sorted_gate": torch.equal(
            ref.sorted_gate.view(i16), routing.sorted_gate.view(i16)
        ),
        "tokens": torch.equal(ref.tokens, routing.tokens),
        "tokens32": torch.equal(ref.tokens32, routing.tokens32),
        "seg_csr": torch.equal(ref.seg_csr, routing.seg_csr),
        "offsets": torch.equal(ref.offsets, routing.offsets),
        "stock_argsort_order": torch.equal(order, routing.order),
        "stock_metadata": torch.equal(sg.view(i16), routing.sorted_gate.view(i16))
        and torch.equal(inv, routing.inverse)
        and torch.equal(offs, routing.offsets),
    }
    a = _FUSED_AUDIT.setdefault(
        key,
        {
            "calls": 0,
            "tokens": 0,
            "top_bad_rows": 0,
            "gate_bad_rows": 0,
            "field_fail": {},
            "ties_rows": 0,
        },
    )
    a["calls"] += 1
    a["tokens"] += int(top.shape[0])
    a["top_bad_rows"] += int(top_bad.sum())
    a["gate_bad_rows"] += int(gate_bad.sum())
    # rows with a tie among the biased values of the 8 picked (where the slot order is the unstable network's)
    sc = torch.sigmoid(logits)
    bv = (sc + layer.expert_bias).gather(-1, top).sort(-1).values
    a["ties_rows"] += int((bv[:, 1:] == bv[:, :-1]).any(-1).sum())
    for k, v in f.items():
        if not v:
            a["field_fail"][k] = a["field_fail"].get(k, 0) + 1
    bad = top_bad | gate_bad
    if bool(bad.any()) and sum(x.shape[0] for x in _FUSED_AUDIT_ROWS) < 4096:
        _FUSED_AUDIT_ROWS.append(
            torch.cat(
                [logits[bad], layer.expert_bias[None].expand(int(bad.sum()), -1)], 1
            ).cpu()
        )
    return ref


def _fused_audit_dump() -> None:
    import json
    import os

    tot = {
        "calls": 0,
        "tokens": 0,
        "top_bad_rows": 0,
        "gate_bad_rows": 0,
        "ties_rows": 0,
    }
    for a in _FUSED_AUDIT.values():
        for k in tot:
            tot[k] += a[k]
    tot["field_fail_layers"] = sum(1 for a in _FUSED_AUDIT.values() if a["field_fail"])
    tot["layers"] = len(_FUSED_AUDIT)
    tot["exact_all"] = (
        tot["top_bad_rows"] == 0
        and tot["gate_bad_rows"] == 0
        and tot["field_fail_layers"] == 0
    )
    path = (
        ROUTER_FUSED_AUDIT
        if isinstance(ROUTER_FUSED_AUDIT, str)
        else "outputs/router_fused_audit.json"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {"overall": tot, "layers": _FUSED_AUDIT}, fh, indent=1, sort_keys=True
        )
    if _FUSED_AUDIT_ROWS:
        torch.save(torch.cat(_FUSED_AUDIT_ROWS), path + ".bad_rows.pt")


class _PermuteRows(torch.autograd.Function):
    """``src[index]`` for a row permutation ``index``.

    The backward is the gather ``grad[inverse]`` with the inverse permutation,
    as in the original training code. A scatter-add backward gives the same values, but it
    turns a gradient of -0.0 into +0.0.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        src: Tensor,
        index: Tensor,
        inverse: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(inverse)
        return src[index]

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor, None, None]:
        (inverse,) = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        return grad[inverse], None, None


def _dispatch_rows(
    top_idx: Tensor, gate: Tensor, num_experts: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sort the (token, expert) rows of one chunk by expert.

    Returns the sort order, the token and the gate of each sorted row and the
    int32 end offsets of the expert groups.
    """
    num_tokens, top_k = top_idx.shape
    device = top_idx.device
    experts = top_idx.reshape(-1)
    _, order = experts.sort(stable=True)
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    counts = torch.zeros(num_experts, device=device, dtype=torch.int64)
    counts.scatter_add_(0, experts.long(), torch.ones_like(experts, dtype=torch.int64))
    offsets = counts.cumsum(dim=0).to(torch.int32)
    return order, tokens[order], gate.reshape(-1)[order], offsets


def _sum_rows_of_tokens(rows: Tensor, rows_of_token: Tensor, scale: Tensor) -> Tensor:
    """``out[t] = sum_k rows[r] * scale[r]`` with ``r = rows_of_token[t, k]``.

    The sum runs in float32 with four interleaved accumulators: slot ``s``
    goes to accumulator ``s % 4``, then ``((a0 + a1) + a2) + a3``. This is the
    reduction of the fused combine kernel of the original implementation.
    """
    num_tokens, top_k = rows_of_token.shape
    acc = [
        torch.zeros(num_tokens, rows.size(1), device=rows.device, dtype=torch.float32)
        for _ in range(4)
    ]
    for slot in range(top_k):
        r = rows_of_token[:, slot]
        acc[slot % 4] = acc[slot % 4] + rows[r].float() * scale[r].unsqueeze(-1)
    return ((acc[0] + acc[1]) + acc[2]) + acc[3]


def _routed_chunk(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
    *,
    fuse: bool = False,
    shared: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Float32 scaled routed sum of one token chunk (see ``MoEFeedForward``)."""
    num_tokens, top_k = top_idx.shape
    fast = fuse and _can_fuse_chunked_moe(x, top_idx, gate, gate_up_w, down_w)
    if fast:
        order = top_idx.flatten().argsort(stable=True)
        rows, sorted_gate, inverse, offsets = moe_dispatch(
            x, top_idx, gate, order, gate_up_w.shape[0], gate_up_w.dtype
        )
    else:
        order, sorted_tokens, sorted_gate, offsets = _dispatch_rows(
            top_idx, gate, gate_up_w.shape[0]
        )
        rows = x[sorted_tokens].to(gate_up_w.dtype)
    gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
    expert_out = torch._grouped_mm(_expert_swiglu(gate_up), down_w, offsets)
    if fast:
        return moe_chunked_combine(
            expert_out, sorted_gate, inverse, top_k, scaling, shared=shared, out=out
        )
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    scale = sorted_gate.to(torch.float32) * scaling
    result = _sum_rows_of_tokens(expert_out, inverse.view(num_tokens, top_k), scale)
    if shared is not None:
        result = result.to(shared.dtype) + shared
    return result if out is None else out.copy_(result)


def _can_fuse_chunked_moe(x, top_idx, gate, gate_up_w, down_w) -> bool:
    """The fixed top-eight CUDA inference path; other calls retain eager math."""
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == gate.dtype == gate_up_w.dtype == down_w.dtype
        and top_idx.dtype == torch.int64
        and top_idx.shape[1] == 8
        and x.shape[1] >= 128
        and top_idx.numel() < 2**31
        and all(t.is_contiguous() for t in (x, top_idx, gate, gate_up_w, down_w))
        and not torch.is_autocast_enabled()
        and not (
            torch.is_grad_enabled()
            and any(t.requires_grad for t in (x, gate, gate_up_w, down_w))
        )
    )


def _routed_chunk_backward(
    grad_out: Tensor,
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The gradients of ``_routed_chunk``, as in the original training code.

    The chunk recomputes ``gate_up``. Returns the float32 gradients of ``x``
    and of the gates and the gradients of the two expert weight banks (in
    the weight dtype).
    """
    num_tokens, top_k = top_idx.shape
    order, sorted_tokens, sorted_gate, offsets = _dispatch_rows(
        top_idx, gate, gate_up_w.shape[0]
    )
    rows = x[sorted_tokens].to(gate_up_w.dtype)
    gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
    scale = sorted_gate.to(torch.float32) * scaling
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=x.device)
    grad_rows_out = grad_out[sorted_tokens]
    hidden = _expert_swiglu(gate_up)
    grad_hidden_unscaled = torch._grouped_mm(
        grad_rows_out.to(gate_up.dtype), down_w.transpose(1, 2), offsets
    )
    grad_scale = (grad_hidden_unscaled * hidden).sum(dim=-1, dtype=torch.float32)
    del grad_hidden_unscaled
    grad_rows_scaled = (grad_rows_out.float() * scale.unsqueeze(-1)).to(gate_up.dtype)
    del grad_rows_out
    grad_down = torch._grouped_mm(hidden.t(), grad_rows_scaled, offsets)
    del hidden
    grad_hidden = torch._grouped_mm(grad_rows_scaled, down_w.transpose(1, 2), offsets)
    del grad_rows_scaled
    grad_gate_up = swiglu_packed_backward_fused(gate_up, grad_hidden)
    del grad_hidden
    grad_gate_up_w = torch._grouped_mm(rows.t(), grad_gate_up, offsets)
    grad_rows_in = torch._grouped_mm(grad_gate_up, gate_up_w.transpose(1, 2), offsets)
    del grad_gate_up
    ones = torch.ones(order.numel(), device=x.device, dtype=torch.float32)
    grad_x = _sum_rows_of_tokens(grad_rows_in, inverse.view(num_tokens, top_k), ones)
    routed = grad_scale * scaling if scaling != 1.0 else grad_scale
    grad_flat = torch.empty_like(routed)
    grad_flat[order] = routed
    return grad_x, grad_flat.view(num_tokens, top_k), grad_gate_up_w, grad_down


class _ChunkedRoutedExperts(torch.autograd.Function):
    """The chunked expert dispatch with the backward of the original training code.

    The forward is ``MoEFeedForward._routed_forward_chunked``. It saves only
    its inputs. The backward runs the same token chunks again: it recomputes
    the first grouped GEMM, takes the gate gradients from the SwiGLU outputs
    and sums the weight gradients of the chunks in the weight dtype.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        gate_up_w: Tensor,
        down_w: Tensor,
        scaling: float,
        chunk: int,
    ) -> Tensor:
        ctx.save_for_backward(x, top_idx, gate, gate_up_w, down_w)
        ctx.scaling, ctx.chunk = scaling, chunk  # ty: ignore[unresolved-attribute]
        return _routed_chunks(x, top_idx, gate, gate_up_w, down_w, scaling, chunk)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad_out: Tensor
    ) -> tuple[Tensor, None, Tensor, Tensor, Tensor, None, None]:
        x, top_idx, gate, gate_up_w, down_w = (
            ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        )
        scaling, step = ctx.scaling, ctx.chunk  # ty: ignore[unresolved-attribute]
        grad_out = grad_out.contiguous()
        num_tokens = x.shape[0]
        grad_x = torch.empty_like(x)
        grad_gate = torch.empty_like(gate, dtype=torch.float32)
        grad_gu: Tensor | None = None
        grad_down: Tensor | None = None
        for start in range(0, max(num_tokens, 1), step):
            end = min(start + step, num_tokens)
            if end <= start:
                break
            gx, gg, ggu, gd = _routed_chunk_backward(
                grad_out[start:end],
                x[start:end],
                top_idx[start:end],
                gate[start:end],
                gate_up_w,
                down_w,
                scaling,
            )
            grad_x[start:end] = gx.to(x.dtype)
            grad_gate[start:end] = gg
            # The first chunk gives the sum. Later chunks add in the weight
            # dtype, as in the original training code.
            grad_gu = ggu.to(gate_up_w.dtype) if grad_gu is None else grad_gu.add_(ggu)
            grad_down = gd.to(down_w.dtype) if grad_down is None else grad_down.add_(gd)
        if grad_gu is None or grad_down is None:
            grad_gu, grad_down = torch.zeros_like(gate_up_w), torch.zeros_like(down_w)
        return (
            grad_x,
            None,
            grad_gate.to(gate.dtype),
            grad_gu.to(gate_up_w.dtype),
            grad_down.to(down_w.dtype),
            None,
            None,
        )


def _routed_chunks(
    x: Tensor,
    top_idx: Tensor,
    gate: Tensor,
    gate_up_w: Tensor,
    down_w: Tensor,
    scaling: float,
    step: int,
    *,
    fuse: bool = False,
    shared: Tensor | None = None,
) -> Tensor:
    """The scaled routed sum over token chunks of ``step``, rounded once."""
    num_tokens, c = x.shape
    out = x.new_empty(num_tokens, c, dtype=torch.float32 if shared is None else x.dtype)
    for start in range(0, max(num_tokens, 1), step):
        end = min(start + step, num_tokens)
        if end <= start:
            break
        result = _routed_chunk(
            x[start:end],
            top_idx[start:end],
            gate[start:end],
            gate_up_w,
            down_w,
            scaling,
            fuse=fuse,
            shared=None if shared is None else shared[start:end],
            out=None if shared is None else out[start:end],
        )
        if shared is None:
            out[start:end] = result
    return out.to(x.dtype) if shared is None else out


RoutingHook = Callable[["MoEFeedForward", Tensor, Tensor], None]


class MoEFeedForward(nn.Module):
    """Mixture of experts: one shared expert plus ``top_k`` of ``num_experts``.

    The router runs in float32 with sigmoid scores. The routing bias changes
    which experts are picked but not their weights. The picked scores are
    normalized to sum to one and cast to the activation dtype. Each expert is
    a SwiGLU. The routed sum is scaled by ``routed_scaling_factor`` and added
    to the shared expert output.

    ``expert_bias`` stays float32 under module dtype casts.

    ``routing_hook`` is None at inference. Training code can set it to a
    callable ``hook(layer, logits, top_idx)``. The layer calls it after the
    routing and before the experts, with the float32 router logits and the
    picked experts.
    """

    expert_bias: Tensor
    routing_hook: RoutingHook | None = None

    def __init__(
        self,
        dim: int = 3072,
        hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        dispatch_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if dispatch_chunk_size is not None and dispatch_chunk_size <= 0:
            raise ValueError("dispatch_chunk_size must be positive")
        self.dispatch_chunk_size = dispatch_chunk_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.routed = RoutedExperts(num_experts, dim, hidden)
        self.shared = SharedExpert(dim, hidden)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        bias = self.expert_bias
        super()._apply(fn, recurse)
        # Keep the routing bias in float32. Move it only to the new device.
        self.expert_bias = bias.to(device=self.expert_bias.device)
        return self

    def _route(self, x_flat: Tensor) -> tuple[Tensor, Tensor]:
        """The routing of ``forward`` (same ops, same order): the top-k expert
        indices and the gates in the dtype of ``x_flat``."""
        rows = _ROUTER_ROWS.get(x_flat.data_ptr()) if _ROUTER_ROWS else None
        with NV.op("moe", "router"):
            if rows is None:
                logits = _router_logits(self.router.weight, x_flat)
            else:
                r0, r1, total = rows
                x3 = x_flat.reshape(-1, r1 - r0, x_flat.size(-1))
                padded = _pad_context_rows(x3, rows).reshape(-1, x_flat.size(-1))
                logits = _router_logits(self.router.weight, padded)
                logits = logits.reshape(x3.shape[0], total, self.num_experts)
                logits = logits[:, r0:r1].reshape(-1, self.num_experts)
        with NV.op("moe", "topk_dispatch"):
            scores = torch.sigmoid(logits)
            biased = scores + self.expert_bias.to(scores.dtype)
            _, top_idx = torch.topk(biased, self.top_k, dim=-1)
            picked = scores.gather(dim=-1, index=top_idx)
            gate = picked / picked.sum(dim=-1, keepdim=True)
            gate = gate.to(x_flat.dtype)
        if self.routing_hook is not None:
            self.routing_hook(self, logits, top_idx)
        return top_idx, gate

    def _router_logits_into(
        self, x_flat: Tensor, out: Tensor, rows: tuple[int, int, int] | None
    ) -> None:
        """``out`` = the logits of ``_route`` / ``forward`` for ``x_flat`` (same GEMM calls and shapes)."""
        if rows is None:
            if ROUTER_F32:
                _router_logits_forward(
                    self.router.weight, x_flat, _take_f32_rows(x_flat), out=out
                )
            else:
                _router_logits_forward(self.router.weight, x_flat, out=out)
            return
        r0, r1, total = rows
        x3 = x_flat.view(-1, r1 - r0, x_flat.size(-1))
        padded = _pad_context_rows(x3, rows).reshape(-1, x_flat.size(-1))
        logits = _router_logits(self.router.weight, padded)
        logits = logits.view(x3.shape[0], total, self.num_experts)
        out.view(x3.shape[0], r1 - r0, self.num_experts).copy_(logits[:, r0:r1])

    def _route_fused(
        self, flats: Sequence[Tensor], ctx_rows: Sequence | None = None, path: str = ""
    ):
        """ROUTER_FUSED: the router_fused.Routing of the inputs ``flats`` (back to back; P = len(flats)):
        the stock FP32 logits of each input, then one fused routing + dispatch kernel. None (before any work)
        where it does not apply. ``ctx_rows``: the context-cache rows per input (default: the registry)."""
        if (
            not ROUTER_FUSED
            or self.routing_hook is not None
            or not 1 <= len(flats) <= 2
        ):
            return None
        f0 = flats[0]
        bias = self.expert_bias
        if not (
            self.top_k == 8
            and self.num_experts == 128
            and f0.is_cuda
            and f0.dtype == torch.bfloat16
            and is_blackwell_tcgen05(f0.device)
            and bias.dtype == torch.float32
            and bias.is_contiguous()
            and bias.device == f0.device
            and all(
                f.dim() == 2
                and f.device == f0.device
                and f.dtype == f0.dtype
                and f.shape[0] > 0
                and f.size(1) == self.router.weight.size(1)
                for f in flats
            )
            and sum(f.shape[0] for f in flats) * 8 < 2**31
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        ):
            return None
        from praxis.kernels import router_fused as RF

        if importlib.util.find_spec("cutlass") is None:
            return None
        rows = [f.shape[0] for f in flats]
        logits = torch.empty(
            sum(rows), self.num_experts, dtype=torch.float32, device=f0.device
        )
        t0 = 0
        for i, (f, lab) in enumerate(
            zip(flats, NV.pass_labels(len(flats)), strict=True)
        ):
            if ctx_rows is not None:
                r = ctx_rows[i]
            else:
                r = _ROUTER_ROWS.get(f.data_ptr()) if _ROUTER_ROWS else None
            with NV.ctx(**lab), NV.op("moe", "router"):
                self._router_logits_into(f, logits[t0 : t0 + rows[i]], r)
            t0 += rows[i]
        with NV.op("moe", "topk_dispatch"):
            routing = RF.route_dispatch(logits, bias, rows)
        if ROUTER_FUSED_AUDIT:
            routing = _fused_audit(self, logits, routing, path)
        return routing

    def _moe_regime(self, rows: Sequence[int]) -> str | None:
        """The combine math of ``forward`` for inputs of ``rows`` tokens each.

        ``"unchunked"`` (no ``dispatch_chunk_size``): ``moe_combine`` (bf16
        products, bf16 sum) then ``moe_scale_add``. ``"chunked"`` (every input
        fits one dispatch chunk): ``moe_chunked_combine`` (float32 scaled
        products and sum, one rounding, then the shared add). None: several
        chunks, an optional routed bank, a routing hook or an empty input; the
        fused expert paths then do not apply and ``forward`` runs.
        """
        if not isinstance(self.routed, RoutedExperts) or self.routing_hook is not None:
            return None
        if not rows or min(rows) <= 0:
            return None
        if self.dispatch_chunk_size is None:
            return "unchunked"
        if max(rows) <= self.dispatch_chunk_size:
            return "chunked"
        return None

    def _moe_inputs_supported(self, flats: Sequence[Tensor]) -> bool:
        """CUDA half-precision inference rows of one device and dtype that
        match the expert banks (the cases the fused expert paths cover)."""
        f0 = flats[0]
        gate_up_w = self.routed.gate_up_weight
        return (
            f0.is_cuda
            and f0.dtype in (torch.bfloat16, torch.float16)
            and gate_up_w.dtype == f0.dtype
            and self.routed.down_weight.dtype == f0.dtype
            and gate_up_w.device == f0.device
            and all(
                f.dim() == 2
                and f.device == f0.device
                and f.dtype == f0.dtype
                and f.size(1) == f0.size(1)
                for f in flats
            )
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        )

    def forward_pair(self, xs: Sequence[Tensor]) -> list[Tensor]:
        """``_forward_pair`` (may return MoETail objects for deferred outputs, see MOE_TAIL_NORM)."""
        global _TAIL_NOW
        prev = _TAIL_NOW
        _TAIL_NOW = _tail_wanted(xs)
        try:
            return self._forward_pair(xs)
        finally:
            _TAIL_NOW = prev

    def _forward_pair(self, xs: Sequence[Tensor]) -> list[Tensor]:
        """``[self(x) for x in xs]`` with one routed-expert pass for all inputs.

        Session.pair_cfg. The router, the gates and the shared expert run
        per input, as in ``forward``. The routed rows of all inputs are
        dispatched together (expert-major; inside an expert, the rows of
        ``xs[0]`` first), so the grouped GEMMs read each expert's weights once.
        They run the Triton grouped GEMM with the fixed ``PAIR_GEMM_CONFIG``,
        whose rows have the bits of ``torch._grouped_mm`` on each input alone.
        Dispatch and combine are row-wise and the combine is the one of
        ``forward`` for the layer's dispatch layout (see ``_moe_regime``), so
        every output has the bits of ``forward``.
        """
        flats = [x.reshape(-1, x.size(-1)) for x in xs]
        if hasattr(self.routed, "routed_forward_pair"):
            paired = self._nvfp4_pair(flats, xs)
            if paired is not None:
                return paired
        regime = self._moe_regime([f.shape[0] for f in flats])
        if regime is None or not self._moe_inputs_supported(flats):
            return [self(x) for x in xs]
        from praxis.kernels.moe_gemm_triton import grouped_mm

        labels = NV.pass_labels(len(flats))
        routing = self._route_fused(flats, path="pair") if ROUTER_FUSED else None
        routes = []
        if routing is None:
            for f, lab in zip(flats, labels, strict=True):
                with NV.ctx(**lab):
                    routes.append(self._route(f))
        shared = []
        for f, lab in zip(flats, labels, strict=True):
            with NV.ctx(**lab):
                shared.append(self.shared(f))
        if routing is not None:
            x = _adjacent_rows(flats)
            top_idx, gate = routing.top_idx, routing.gate
        else:
            with NV.op("moe", "topk_dispatch"):
                x = _adjacent_rows(flats)
                top_idx = torch.cat([r[0] for r in routes])
                gate = torch.cat([r[1] for r in routes])
        if _fused_moe_enabled(FUSED_MOE, x):
            fused = self._routed_fused(
                x, top_idx, gate, regime, shared, xs, dispatch=routing
            )
            if fused is not None:
                return fused
        gate_up_w = self.routed.gate_up_weight
        with NV.op("moe", "topk_dispatch"):
            order = top_idx.reshape(-1).argsort(stable=True)
            rows, sorted_gate, inverse, offsets = moe_dispatch(
                x, top_idx, gate, order, self.num_experts, gate_up_w.dtype
            )
        with NV.op("moe", "routed_gate_up_swiglu"):
            gate_up = grouped_mm(rows, gate_up_w, offsets, config=PAIR_GEMM_CONFIG)
            act = _expert_swiglu(gate_up)
        with NV.op("moe", "routed_down"):
            out = grouped_mm(
                act,
                self.routed.down_weight,
                offsets,
                config=PAIR_GEMM_CONFIG,
            )
        return self._pair_combine(regime, out, sorted_gate, inverse, shared, xs)

    def _nvfp4_pair(
        self, flats: Sequence[Tensor], xs: Sequence[Tensor]
    ) -> list[Tensor] | None:
        """``forward_pair`` for an SM 11.0 NVFP4 routed bank (``nvfp4.PAIR_NVFP4``): routing and
        the shared expert per input as in ``forward``, one paired NVFP4 dispatch (praxis/kernels/nvfp4_pair.py)
        whose per-(expert, input) globals keep the bits of each input's own ``routed_forward``. None (before
        any work) where it does not apply."""
        from praxis.quantization import nvfp4 as Q

        if not (
            Q.PAIR_NVFP4
            and isinstance(self.routed, Q.NVFP4RoutedExperts)
            and len(flats) == 2
            and self.routing_hook is None
            and not self.routed.training
            and not torch.is_autocast_enabled()
            and all(
                f.is_cuda
                and f.dtype == torch.bfloat16
                and f.is_contiguous()
                and f.device == flats[0].device
                and all(value.device == f.device for value in self.routed.buffers())
                and 0 < f.shape[0] <= (self.dispatch_chunk_size or f.shape[0])
                for f in flats
            )
            and torch.cuda.get_device_capability(flats[0].device) == (11, 0)
            and not torch.is_grad_enabled()
        ):
            return None
        labels = NV.pass_labels(len(flats))
        routing = self._route_fused(flats, path="nvfp4_pair") if ROUTER_FUSED else None
        routes = []
        if routing is not None:
            t0 = flats[0].shape[0]
            routes = [
                (routing.top_idx[:t0], routing.gate[:t0]),
                (routing.top_idx[t0:], routing.gate[t0:]),
            ]
        else:
            for f, lab in zip(flats, labels, strict=True):
                with NV.ctx(**lab):
                    routes.append(self._route(f))
        shared = []
        for f, lab in zip(flats, labels, strict=True):
            with NV.ctx(**lab):
                shared.append(self.shared(f))
        if (
            _TAIL_NOW
            and self.top_k == 8
            and all(
                s.shape == f.shape and s.dtype == f.dtype and s.is_contiguous()
                for s, f in zip(shared, flats, strict=True)
            )
        ):
            # MOE_TAIL_NORM: the combine + shared add run inside the next block's norm.
            with NV.op("moe", "routed"):
                tails = self.routed.routed_forward_pair(
                    flats,
                    [r[0] for r in routes],
                    [r[1] for r in routes],
                    scaling=self.routed_scaling_factor,
                    tail_shared=shared,
                    **({} if routing is None else {"dispatch": routing}),
                )
            for t, x in zip(tails, xs, strict=True):
                t.shape = x.shape
            return tails
        with NV.op("moe", "routed"):
            ys = self.routed.routed_forward_pair(
                flats,
                [r[0] for r in routes],
                [r[1] for r in routes],
                scaling=self.routed_scaling_factor,
                **({} if routing is None else {"dispatch": routing}),
            )
        return [
            (y + s).reshape(x.shape) for y, s, x in zip(ys, shared, xs, strict=True)
        ]

    def _routed_fused(
        self,
        x: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        regime: str,
        shared: Sequence[Tensor],
        xs: Sequence[Tensor],
        gate_up_config: dict | None = None,
        down_config: dict | None = None,
        dispatch=None,
    ) -> list[Tensor] | None:
        """The FUSED_MOE routed experts of the rows ``x`` (the inputs ``xs``
        back to back, ``shared`` their shared-expert outputs); None (before any
        work) if not supported. ``dispatch``: a router_fused.Routing of
        these rows (its order / sorted gates / inverse / offsets are the stock
        dispatch metadata of its top_idx / gate)."""
        from praxis.kernels.moe_gemm_triton import gate_up_swiglu_supported

        gate_up_w = self.routed.gate_up_weight
        if not (
            x.is_contiguous()
            and x.dtype == gate_up_w.dtype
            and gate.dtype == x.dtype
            and gate.is_contiguous()
            and top_idx.dtype == torch.int64
            and top_idx.is_contiguous()
            and self.top_k == 8
        ):
            return None
        a_idx = None
        if dispatch is not None:
            order, sorted_gate, inverse, offsets = (
                dispatch.order,
                dispatch.sorted_gate,
                dispatch.inverse,
                dispatch.offsets,
            )
            a_idx = dispatch.tokens32
        else:
            with NV.op("moe", "topk_dispatch"):
                order = top_idx.reshape(-1).argsort(stable=True)
                sorted_gate, inverse, offsets = moe_dispatch_metadata(
                    top_idx, gate, order, self.num_experts
                )
        if not gate_up_swiglu_supported(x, gate_up_w, offsets):
            return None
        out = self._fused_expert_gemms(
            x, order, offsets, gate_up_config, down_config, a_idx=a_idx
        )
        return self._fused_combine(regime, out, sorted_gate, inverse, shared, xs)

    def _fused_expert_gemms(
        self,
        x: Tensor,
        order: Tensor,
        offsets: Tensor,
        gate_up_config: dict | None,
        down_config: dict | None,
        a_idx: Tensor | None = None,
    ) -> Tensor:
        """``grouped_mm(swiglu_packed(grouped_mm(x[order // top_k], gate_up)),
        down)`` of the sorted rows, with the gather and the SwiGLU inside the
        first GEMM. ``a_idx``: ``(order // top_k).int()`` if the caller has it."""
        from praxis.kernels.moe_gemm_triton import grouped_gate_up_swiglu, grouped_mm

        if CUTE_EXPERTS and is_thor(x.device) and x.dtype == torch.bfloat16:
            from praxis.kernels import cute_experts as CE

            if CE.supported(x.device):
                with NV.op("moe", "routed_gate_up_swiglu"):
                    h = CE.routed_gate_up_swiglu(
                        x,
                        CE.routed_bank_kmajor(self.routed.gate_up_weight),
                        offsets,
                        order,
                        self.top_k,
                        **({} if a_idx is None else {"a_idx": a_idx}),
                    )
                with NV.op("moe", "routed_down"):
                    return CE.routed_down(h, self.routed.down_weight, offsets)
        if gate_up_config is None and ENC_GATE_UP_AUTO:
            rows = order.numel()
            gate_up_config = (
                ENC_GATE_UP_BM128 if 40_000 <= rows < 48_000 else ENC_GATE_UP_BM256
            )
        with NV.op("moe", "routed_gate_up_swiglu"):
            h = grouped_gate_up_swiglu(
                x,
                self.routed.gate_up_weight,
                offsets,
                order=order,
                top_k=self.top_k,
                config=gate_up_config,
            )
        with NV.op("moe", "routed_down"):
            if ROUTED_QUACK_DOWN and is_thor(h.device):
                from praxis.kernels import routed_quack as RQ

                return RQ.down(h, self.routed.down_weight, offsets)
            return grouped_mm(
                h,
                self.routed.down_weight,
                offsets,
                config=PAIR_GEMM_CONFIG if down_config is None else down_config,
            )

    def _fused_combine(
        self,
        regime: str,
        out: Tensor,
        sorted_gate: Tensor,
        inverse: Tensor,
        shared: Sequence[Tensor],
        xs: Sequence[Tensor],
    ) -> list[Tensor]:
        """Unchunked: ``moe_combine_scale_add`` per input (the roundings of
        moe_combine -> moe_scale_add in one kernel). Otherwise, or if that
        kernel does not cover the case: ``_pair_combine``."""
        if regime == "unchunked" and all(
            moe_combine_scale_add_supported(out, sorted_gate, inverse, s, self.top_k)
            for s in shared
        ):
            res, t0 = [], 0
            for s, xi, lab in zip(shared, xs, NV.pass_labels(len(xs)), strict=True):
                with NV.op("moe", "combine", **NV.pas_kw(lab)):
                    res.append(
                        moe_combine_scale_add(
                            out, sorted_gate, inverse, s, self.routed_scaling_factor, t0
                        ).reshape(xi.shape)
                    )
                t0 += s.shape[0]
            return res
        return self._pair_combine(regime, out, sorted_gate, inverse, shared, xs)

    def _pair_combine(
        self,
        regime: str,
        out: Tensor,
        sorted_gate: Tensor,
        inverse: Tensor,
        shared: Sequence[Tensor],
        xs: Sequence[Tensor],
    ) -> list[Tensor]:
        """Per input: the combine + scale + shared add of ``forward`` over the
        input's own tokens of the merged dispatch (rows ``t0 .. t0 + n``)."""
        k = self.top_k
        if regime == "unchunked":
            with NV.op("moe", "combine"):
                y = moe_combine(out, sorted_gate, inverse, k)
                parts = y.split([s.shape[0] for s in shared])
                return [
                    moe_scale_add(p, s, self.routed_scaling_factor).reshape(x.shape)
                    for p, s, x in zip(parts, shared, xs, strict=True)
                ]
        res, t0 = [], 0
        if _TAIL_NOW and _tail_bf16_ok(out, sorted_gate, inverse, shared, k):
            for s, x in zip(shared, xs, strict=True):
                n = s.shape[0]
                res.append(
                    MoETail(
                        1,
                        out,
                        inverse[t0 * k : (t0 + n) * k],
                        sorted_gate,
                        s,
                        float(self.routed_scaling_factor),
                        x.shape,
                    )
                )
                t0 += n
            return res
        for s, x, lab in zip(shared, xs, NV.pass_labels(len(xs)), strict=True):
            n = s.shape[0]
            with NV.op("moe", "combine", **NV.pas_kw(lab)):
                res.append(
                    moe_chunked_combine(
                        out,
                        sorted_gate,
                        inverse[t0 * k : (t0 + n) * k],
                        k,
                        float(self.routed_scaling_factor),
                        shared=s,
                    ).reshape(x.shape)
                )
            t0 += n
        return res

    def forward(
        self, x: Tensor, *, router_rows: tuple[int, int, int] | None = None
    ) -> Tensor:
        """``_forward`` (may return a MoETail for a deferred output, see MOE_TAIL_NORM)."""
        global _TAIL_NOW
        prev = _TAIL_NOW
        _TAIL_NOW = _tail_wanted((x,))
        try:
            return self._forward(x, router_rows=router_rows)
        finally:
            _TAIL_NOW = prev

    def _forward(
        self, x: Tensor, *, router_rows: tuple[int, int, int] | None = None
    ) -> Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        if router_rows is None and _ROUTER_ROWS:
            router_rows = _ROUTER_ROWS.get(x_flat.data_ptr())
        if router_rows is None and _fused_moe_enabled(FUSED_MOE_DEC, x_flat):
            fused = self._forward_fused_solo(x_flat)
            if isinstance(fused, MoETail):
                fused.shape = shape
                return fused
            if fused is not None:
                return fused.reshape(shape)
        routing = None
        if ROUTER_FUSED and (router_rows is None or x.dim() in (2, 3)):
            routing = self._route_fused([x_flat], [router_rows], path="forward")
        if routing is not None:
            top_idx, gate = routing.top_idx, routing.gate
        else:
            with NV.op("moe", "router"):
                if router_rows is None:
                    logits = _router_logits(self.router.weight, x_flat)
                else:
                    # Preserve the full FP32 router GEMM and its row-chunk shapes.
                    r0, r1, total = router_rows
                    padded = _pad_context_rows(x, router_rows)
                    logits = _router_logits(
                        self.router.weight, padded.reshape(-1, x.size(-1))
                    )
                    logits = logits.reshape(*shape[:-2], total, self.num_experts)
                    logits = logits[..., r0:r1, :].reshape(-1, self.num_experts)
            with NV.op("moe", "topk_dispatch"):
                scores = torch.sigmoid(logits)
                biased = scores + self.expert_bias.to(scores.dtype)
                _, top_idx = torch.topk(biased, self.top_k, dim=-1)
                picked = scores.gather(dim=-1, index=top_idx)
                gate = picked / picked.sum(dim=-1, keepdim=True)
                gate = gate.to(x_flat.dtype)
            if self.routing_hook is not None:
                self.routing_hook(self, logits, top_idx)
        y_shared = self.shared(x_flat)
        if _TAIL_NOW and self.dispatch_chunk_size is not None:
            # MOE_TAIL_NORM: NVFP4 one-chunk tail, combined inside the next block's norm.
            routed_tail = getattr(self.routed, "routed_tail", None)
            if routed_tail is not None:
                with NV.op("moe", "routed"):
                    tail = routed_tail(
                        x_flat,
                        top_idx,
                        gate,
                        scaling=self.routed_scaling_factor,
                        chunk_size=self.dispatch_chunk_size,
                        shared=y_shared,
                        shape=shape,
                    )
                if tail is not None:
                    return tail
        with NV.op("moe", "routed"):
            if routing is not None:
                out = self._routed_tail(
                    x_flat, top_idx, gate, y_shared, dispatch=routing
                )
            else:
                out = self._routed_tail(x_flat, top_idx, gate, y_shared)
        return out.reshape(shape)

    def _forward_fused_solo(self, x_flat: Tensor) -> Tensor | None:
        """``forward`` of the flat rows with the FUSED_MOE routed experts and
        the DEC_*_CONFIG tiles (FUSED_MOE_DEC); None (before any work)
        where the fused expert paths do not apply."""
        regime = self._moe_regime([x_flat.shape[0]])
        if regime is None or not self._moe_inputs_supported([x_flat]):
            return None
        routing = self._route_fused([x_flat], path="solo") if ROUTER_FUSED else None
        if routing is not None:
            top_idx, gate = routing.top_idx, routing.gate
        else:
            top_idx, gate = self._route(x_flat)
        y_shared = self.shared(x_flat)
        fused = self._routed_fused(
            x_flat,
            top_idx,
            gate,
            regime,
            [y_shared],
            [x_flat],
            gate_up_config=DEC_GATE_UP_CONFIG,
            down_config=DEC_DOWN_CONFIG,
            dispatch=routing,
        )
        if fused is not None:
            return fused[0]
        with NV.op("moe", "routed"):
            return self._routed_tail(x_flat, top_idx, gate, y_shared)

    def _routed_tail(
        self,
        x_flat: Tensor,
        top_idx: Tensor,
        gate: Tensor,
        y_shared: Tensor,
        dispatch=None,
    ) -> Tensor:
        """The routed experts and the shared add of ``forward``, on the flat
        rows (FUSED_MOE_SOLO controls the unchunked combine)."""
        if not isinstance(self.routed, RoutedExperts):
            # Packed banks include routed scaling; the shared branch stays native.
            with_shared = getattr(self.routed, "routed_forward_with_shared", None)
            # The fused routing kernel's dispatch of the whole input (one chunk).
            dkw = (
                {"dispatch": dispatch}
                if dispatch is not None
                and getattr(self.routed, "ACCEPTS_DISPATCH", False)
                else {}
            )
            if with_shared is not None:
                out = with_shared(
                    x_flat,
                    top_idx,
                    gate,
                    scaling=self.routed_scaling_factor,
                    chunk_size=self.dispatch_chunk_size or max(1, x_flat.shape[0]),
                    shared=y_shared,
                    **dkw,
                )
            else:
                y_routed = self.routed.routed_forward(
                    x_flat,
                    top_idx,
                    gate,
                    scaling=self.routed_scaling_factor,
                    chunk_size=self.dispatch_chunk_size or max(1, x_flat.shape[0]),
                    **dkw,
                )
                out = y_routed + y_shared
        elif self.dispatch_chunk_size is not None:
            if (
                y_shared.dtype == x_flat.dtype
                and y_shared.is_contiguous()
                and not (torch.is_grad_enabled() and y_shared.requires_grad)
                and _can_fuse_chunked_moe(
                    x_flat,
                    top_idx,
                    gate,
                    self.routed.gate_up_weight,
                    self.routed.down_weight,
                )
            ):
                return _routed_chunks(
                    x_flat,
                    top_idx,
                    gate,
                    self.routed.gate_up_weight,
                    self.routed.down_weight,
                    float(self.routed_scaling_factor),
                    int(self.dispatch_chunk_size),
                    fuse=True,
                    shared=y_shared,
                )
            # The chunked path applies the scaling factor itself.
            y_routed = self._routed_forward_chunked(x_flat, top_idx, gate)
            out = y_routed + y_shared
        else:
            out = None
            if _fused_moe_enabled(FUSED_MOE_SOLO, x_flat):
                out = self._routed_forward_scale_add(x_flat, top_idx, gate, y_shared)
            if out is None:
                y_routed = self._routed_forward(x_flat, top_idx, gate)
                out = moe_scale_add(y_routed, y_shared, self.routed_scaling_factor)
        return out

    def _routed_forward_scale_add(
        self, x: Tensor, top_idx: Tensor, gate: Tensor, shared: Tensor
    ) -> Tensor | None:
        """``moe_scale_add(self._routed_forward(x, top_idx, gate), shared,
        routed_scaling_factor)`` with the combine, the scale and the shared add
        in one kernel (FUSED_MOE_SOLO; the roundings of moe_combine ->
        moe_scale_add), or None (before any work) if unsupported."""
        gate_up_w = self.routed.gate_up_weight
        if not (
            shared.is_contiguous()
            and shared.dim() == 2
            and self.top_k == 8
            and shared.dtype == x.dtype
            and gate.dtype == x.dtype
            and x.is_cuda
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled()
        ):
            return None
        with NV.op("moe", "topk_dispatch"):
            order = top_idx.reshape(-1).argsort(stable=True)
            rows, sorted_gate, inverse, offsets = moe_dispatch(
                x, top_idx, gate, order, self.num_experts, gate_up_w.dtype
            )
        with NV.op("moe", "routed_gate_up_swiglu"):
            gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
            act = _expert_swiglu(gate_up)
        with NV.op("moe", "routed_down"):
            out = torch._grouped_mm(act, self.routed.down_weight, offsets)
        with NV.op("moe", "combine"):
            if moe_combine_scale_add_supported(
                out, sorted_gate, inverse, shared, self.top_k
            ):
                return moe_combine_scale_add(
                    out, sorted_gate, inverse, shared, self.routed_scaling_factor
                )
            y = moe_combine(out, sorted_gate, inverse, self.top_k)
            return moe_scale_add(y, shared, self.routed_scaling_factor)

    def _routed_forward_chunked(
        self, x: Tensor, top_idx: Tensor, gate: Tensor
    ) -> Tensor:
        """Routed experts over token chunks of ``dispatch_chunk_size``.

        Each chunk sorts its (token, expert) rows by expert and runs the two
        grouped GEMMs. The row scale is the gate in float32 times
        ``routed_scaling_factor``. The ``top_k`` scaled rows of a token are
        summed in float32 with four interleaved accumulators (slot ``s`` goes
        to accumulator ``s % 4``, then ``((a0 + a1) + a2) + a3``). The sum is
        rounded to the dtype of ``x`` once, after all chunks. Returns the
        scaled routed sum. With grad, ``_ChunkedRoutedExperts`` gives the
        backward of the original training code.
        """
        assert self.dispatch_chunk_size is not None
        args = (
            x,
            top_idx,
            gate,
            self.routed.gate_up_weight,
            self.routed.down_weight,
            float(self.routed_scaling_factor),
            int(self.dispatch_chunk_size),
        )
        if torch.is_grad_enabled() and (
            x.requires_grad
            or gate.requires_grad
            or self.routed.gate_up_weight.requires_grad
            or self.routed.down_weight.requires_grad
        ):
            out = _ChunkedRoutedExperts.apply(*args)
            assert isinstance(out, Tensor)
            return out
        return _routed_chunks(*args)

    def _routed_forward(self, x: Tensor, top_idx: Tensor, gate: Tensor) -> Tensor:
        """Sort the (token, expert) rows by expert, run two grouped GEMMs,
        scale each row by its gate, undo the sort and sum over ``top_k``."""
        experts = top_idx.reshape(-1)
        order = experts.argsort(stable=True)
        gate_up_w = self.routed.gate_up_weight
        rows, sorted_gate, inverse, offsets = moe_dispatch(
            x,
            top_idx,
            gate,
            order,
            self.num_experts,
            gate_up_w.dtype,
        )
        gate_up = torch._grouped_mm(rows, gate_up_w, offsets)
        out = torch._grouped_mm(
            _expert_swiglu(gate_up), self.routed.down_weight, offsets
        )
        if torch.is_grad_enabled() and (out.requires_grad or sorted_gate.requires_grad):
            weighted = out * sorted_gate.unsqueeze(-1)
            unsorted = _PermuteRows.apply(weighted, inverse, order)
            return unsorted.reshape(x.shape[0], self.top_k, x.shape[1]).sum(dim=1)
        return moe_combine(out, sorted_gate, inverse, self.top_k)


# ---------------------------------------------------------------------------
# Block and stack
# ---------------------------------------------------------------------------


Modulation = tuple[nn.Module, Tensor, ModulationDtype | None]


def _fused_norm_ok(x: Tensor, norms: Sequence[RMSNorm], s: Tensor) -> bool:
    """The rows and norms ``norm_triton.add_norm_scale`` covers: bf16 CUDA
    inference rows, plain float32-cast RMSNorm modules with a weight and no
    hooks, a bf16 0-dim scale, outside autocast."""
    from praxis.kernels import norm_triton as NT

    return (
        not torch.is_autocast_enabled()
        and not torch.nn.modules.module._global_forward_hooks
        and not torch.nn.modules.module._global_forward_pre_hooks
        and all(
            type(norm) is RMSNorm
            and NT.supported(x, norm, s)
            and norm.weight.device == x.device
            and s.device == x.device
            and not norm._forward_hooks
            and not norm._forward_pre_hooks
            and "forward" not in vars(norm)
            for norm in norms
        )
    )


def _fused_norm_enabled(x: Tensor) -> bool:
    """Select fusion without an architecture allowlist; preserve overrides."""
    if FUSED_NORM is not None:
        return FUSED_NORM
    return (
        not torch.is_grad_enabled()
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_autocast_enabled()
    )


def _ctx_rows_enter(ffn: nn.Module, h: Tensor, kv: KVSplice | None) -> int:
    """Keep a yielded MoE input's full row shape until its generator resumes."""
    if not CTX_FULL_ROWS or kv is None or not isinstance(ffn, MoEFeedForward):
        return 0
    rows = kv.full_rows()
    if rows is None:
        return 0
    key = h.data_ptr()
    _ROUTER_ROWS[key] = rows
    return key


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x) * s)``, then ``x + ffn(norm(x) * s)``.

    ``s`` is ``1 / sqrt(layer_index)``. It is a persistent buffer, so the
    checkpoint value (rounded to bf16) is the value in use.

    ``modulation`` is ``(modulator, x_mod, dtype)`` in a modulated stack. The
    block then first replaces ``x`` with ``modulate(x, shift, 1 + scale)``,
    with ``(shift, scale)`` from ``modulator(x_mod)``. Thus one block call
    holds all the work of a layer. An activation checkpoint of the block
    also recomputes the modulation.
    """

    s: Tensor

    def __init__(
        self,
        ffn: nn.Module,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        layer_index: int = 1,
        attention_backend: AttentionBackend = "triton",
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1 / (layer_index**0.5)))
        self.ln_in = RMSNorm(dim)
        self.mha = Attention(dim, head_dim, num_kv_heads, attention_backend)
        self.ln_out = RMSNorm(dim)
        self.ffn = ffn

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | None = None,
        modulation: Modulation | None = None,
        kv: KVSplice | None = None,
    ) -> Tensor:
        if modulation is not None:
            modulator, x_mod, dtype = modulation
            shift, scale = modulator(x_mod).chunk(2, dim=-1)
            x = modulate(x, shift, 1 + scale, dtype=dtype)
        x, normed = residual_norm_scale(
            x,
            self.mha(rms_norm_scale(self.ln_in, x, self.s), rope, block_mask, kv),
            self.ln_out,
            self.s,
        )
        rows = (
            kv.full_rows()
            if kv is not None and (self.mha.backend == "flash" or CTX_FULL_ROWS)
            else None
        )
        if rows is not None and isinstance(self.ffn, MoEFeedForward):
            x = x + self.ffn(normed, router_rows=rows)
        else:
            x = x + self.ffn(normed)
        return x

    def forward_gen(
        self,
        x: Tensor,
        rope: Tensor | None,
        block_mask: BlockMask | None = None,
        modulation: Modulation | None = None,
        kv: KVSplice | None = None,
        pending: Tensor | None = None,
        defer: bool = False,
    ) -> Generator[tuple[Callable[[Tensor], Tensor], Tensor], Tensor, Any]:
        """``forward`` as a generator that yields ``(ffn, input)`` and is sent
        the feed-forward output (see :func:`run_solo`, :func:`run_pair`).

        The ops are those of ``forward``, in its order. The block's own module
        hooks do not run (the driver calls the feed-forward module itself).
        ``pending`` (FUSED_NORM) is the previous block's feed-forward output,
        still to add to ``x`` first. With ``defer`` the generator returns
        ``(x, ffn_out)`` instead of their sum.
        """
        h = None
        if modulation is not None:
            modulator, x_mod, dtype = modulation
            geom = None
            with NV.op("add_norm", "mod_linear"):
                if isinstance(x_mod, CompactMod):
                    mod = modulator(x_mod.rows)
                    geom = x_mod.geom
                else:
                    mod = modulator(x_mod)
            if (
                FUSED_MOD_NORM
                and _fused_norm_enabled(x)
                and dtype == "mixed"
                and _fused_norm_ok(x, (self.ln_in, self.ln_out), self.s)
            ):
                from praxis.kernels import norm_triton as NT

                if isinstance(pending, MoETail):
                    from praxis.kernels import moe_tail_triton as MT

                    if MT.tail_supported(
                        x, pending
                    ) and NT.add_mod_norm_scale_supported(
                        x, None, mod, self.ln_in, self.s, geom
                    ):
                        with NV.op("add_norm", "add_mod_norm_in"):
                            if geom is None:
                                x, h = MT.tail_add_mod_norm_scale(
                                    x,
                                    pending,
                                    mod,
                                    self.ln_in,
                                    self.s,
                                    modfma=NT.MOD_FMA,
                                )
                            else:
                                x, h = MT.tail_add_mod_norm_scale(
                                    x,
                                    pending,
                                    mod,
                                    self.ln_in,
                                    self.s,
                                    modfma=NT.MOD_FMA,
                                    geom=geom,
                                )
                        pending = None
                    else:
                        pending = pending.materialize()
                if h is None and NT.add_mod_norm_scale_supported(
                    x, pending, mod, self.ln_in, self.s, geom
                ):
                    with NV.op("add_norm", "add_mod_norm_in"):
                        if geom is None:
                            x, h = NT.add_mod_norm_scale(
                                x, pending, mod, self.ln_in, self.s, modfma=NT.MOD_FMA
                            )
                        else:
                            x, h = NT.add_mod_norm_scale(
                                x,
                                pending,
                                mod,
                                self.ln_in,
                                self.s,
                                modfma=NT.MOD_FMA,
                                geom=geom,
                            )
                    pending = None
            if h is None and geom is not None:
                mod = x_mod.expand(mod)
            if h is None:
                pending = _tail_mat(pending)
                with NV.op("add_norm", "modulate"):
                    shift, scale = mod.chunk(2, dim=-1)
                    if pending is not None:
                        x = x + pending
                        pending = None
                    x = modulate(x, shift, 1 + scale, dtype=dtype)
        if _fused_norm_enabled(x) and (
            h is not None or _fused_norm_ok(x, (self.ln_in, self.ln_out), self.s)
        ):
            from praxis.kernels import norm_triton as NT

            if h is None and isinstance(pending, MoETail):
                from praxis.kernels import moe_tail_triton as MT

                if MT.tail_supported(x, pending):
                    with NV.op("add_norm", "add_norm_in"):
                        x, h = MT.tail_add_norm_scale(x, pending, self.ln_in, self.s)
                else:
                    pending = pending.materialize()
            if h is None:
                with NV.op("add_norm", "add_norm_in"):
                    x, h = NT.add_norm_scale(x, pending, self.ln_in, self.s)
            hf = None
            if ROUTER_F32 and isinstance(self.ffn, MoEFeedForward):
                hf = torch.empty(x.shape, dtype=torch.float32, device=x.device)
            a = self.mha(h, rope, block_mask, kv)
            with NV.op("add_norm", "add_norm_out"):
                x, h = NT.add_norm_scale(
                    x,
                    a,
                    self.ln_out,
                    self.s,
                    out=_pair_slot(x) if PAIR_BUFFER else None,
                    out_f32=hf,
                )
            if hf is not None:
                _put_f32_rows(h, hf)
            key = _ctx_rows_enter(self.ffn, h, kv)
            tkey = 0
            if MOE_TAIL_NORM and isinstance(self.ffn, MoEFeedForward):
                tkey = h.data_ptr()
                _TAIL_INPUTS.add(tkey)
            try:
                f = yield self.ffn, h
            finally:
                _ROUTER_ROWS.pop(key, None)
                if tkey:
                    _TAIL_INPUTS.discard(tkey)
            if defer:
                return x, f
            if isinstance(f, MoETail):
                from praxis.kernels import moe_tail_triton as MT

                if MT.tail_supported(x, f) and f.shape == x.shape:
                    with NV.op("add_norm", "residual_add"):
                        return MT.tail_residual_add(x, f)
                f = f.materialize()
            with NV.op("add_norm", "residual_add"):
                return x + f
        pending = _tail_mat(pending)
        if pending is not None:
            with NV.op("add_norm", "residual_add"):
                x = x + pending
        with NV.op("add_norm", "norm_in"):
            normed_in = rms_norm_scale(self.ln_in, x, self.s)
        a = self.mha(normed_in, rope, block_mask, kv)
        with NV.op("add_norm", "add_norm_out"):
            x, normed = residual_norm_scale(x, a, self.ln_out, self.s)
        key = _ctx_rows_enter(self.ffn, normed, kv)
        try:
            f = yield self.ffn, normed
        finally:
            _ROUTER_ROWS.pop(key, None)
        if defer:
            return x, f
        with NV.op("add_norm", "residual_add"):
            return x + f


def _adjacent_rows(flats: Sequence[Tensor]) -> Tensor:
    """``torch.cat(flats)``, as a view when the row blocks already lie back to
    back in one buffer (PAIR_BUFFER)."""
    f0 = flats[0]
    pos = f0.data_ptr()
    ok = all(
        f.is_contiguous() and f.dtype == f0.dtype and f.size(1) == f0.size(1)
        for f in flats
    )
    if ok:
        for f in flats:
            if (
                f.data_ptr() != pos
                or f.untyped_storage().data_ptr() != f0.untyped_storage().data_ptr()
            ):
                ok = False
                break
            pos += f.numel() * f.element_size()
    if not ok:
        return torch.cat(flats)
    return f0.as_strided((sum(f.size(0) for f in flats), f0.size(1)), (f0.size(1), 1))


def run_solo(
    gen: Generator[tuple[Callable[[Tensor], Tensor], Tensor], Tensor, Any],
) -> Any:
    """Drive a ``forward_gen`` generator, running each feed-forward on its own."""
    gen = NV.tag_current(gen)
    try:
        ffn, h = next(gen)
        while True:
            with NV.like(gen):
                y = ffn(h)
            ffn, h = gen.send(y)
    except StopIteration as stop:
        return stop.value
    finally:
        gen.close()


def run_pair(
    gens: Sequence[Generator[tuple[Callable[[Tensor], Tensor], Tensor], Tensor, Any]],
) -> list[Any]:
    """Drive several ``forward_gen`` generators of the same stack in lockstep.

    At each feed-forward, a mixture of experts shared by all of them runs
    once through :meth:`MoEFeedForward.forward_pair`; any other feed-forward
    runs per generator. Each result has the bits of :func:`run_solo`.
    """
    global _PAIR_SLOTS
    prev = _PAIR_SLOTS
    if PAIR_BUFFER and FUSED_NORM is not False:
        _PAIR_SLOTS = {
            "n": len(gens),
            "i": 0,
            "seen": [0] * len(gens),
            "sizes": None,
            "buf": None,
        }
    try:
        return _run_pair(gens)
    finally:
        _PAIR_SLOTS = prev
        for gen in gens:
            gen.close()


def _run_pair(
    gens: Sequence[Generator[tuple[Callable[[Tensor], Tensor], Tensor], Tensor, Any]],
) -> list[Any]:
    done: list[Any] = []
    reqs = []
    for g in gens:
        try:
            reqs.append(next(g))
        except StopIteration as stop:
            done.append(stop.value)
    if done:
        if len(done) != len(gens):
            raise RuntimeError("run_pair: the stacks ended at different blocks")
        return done
    while True:
        ffns = [r[0] for r in reqs]
        if isinstance(ffns[0], MoEFeedForward) and all(f is ffns[0] for f in ffns):
            with NV.like(gens[0], **{"pass": "p"}):
                ys = ffns[0].forward_pair([r[1] for r in reqs])
        else:
            ys = []
            for g, (f, h) in zip(gens, reqs, strict=True):
                with NV.like(g):
                    ys.append(f(h))
        nxt = []
        for g, y in zip(gens, ys, strict=True):
            try:
                nxt.append(g.send(y))
            except StopIteration as stop:
                done.append(stop.value)
        if done:
            if len(done) != len(gens):
                raise RuntimeError("run_pair: the stacks ended at different blocks")
            return done
        reqs = nxt


class Transformer(nn.Module):
    """A stack of blocks with a shared rotary table.

    The first ``num_dense`` blocks have a dense SwiGLU, the others a mixture of
    experts. With ``mod_dim`` set, each block has a modulator: a linear map of
    the per-token conditioning ``x_mod`` to ``(shift, scale)``. The block input
    is then ``modulate(x, shift, 1 + scale)``.
    """

    def __init__(
        self,
        depth: int = 14,
        num_dense: int = 2,
        dim: int = 3072,
        head_dim: int = 128,
        num_kv_heads: int = 8,
        dense_hidden: int = 11520,
        moe_hidden: int = 1280,
        num_experts: int = 128,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        mod_dim: int | None = None,
        modulation_dtype: ModulationDtype | None = "mixed",
        rope_effective_dim: int = 64,
        rope_base: int = 26_000_000,
        rope_table_size: int = 1_000_000,
        attention_backend: AttentionBackend = "triton",
        moe_dispatch_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.rope = RotaryEmbedding(
            head_dim, rope_effective_dim, rope_base, rope_table_size
        )
        blocks = []
        for i in range(depth):
            ffn: nn.Module
            if i < num_dense:
                ffn = SwiGLUFeedForward(dim, dense_hidden)
            else:
                ffn = MoEFeedForward(
                    dim,
                    moe_hidden,
                    num_experts,
                    top_k,
                    routed_scaling_factor,
                    moe_dispatch_chunk_size,
                )
            blocks.append(
                TransformerBlock(
                    ffn,
                    dim,
                    head_dim,
                    num_kv_heads,
                    layer_index=i + 1,
                    attention_backend=attention_backend,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.modulators: nn.ModuleList | None = None
        if mod_dim is not None:
            self.modulators = nn.ModuleList(
                [nn.Linear(mod_dim, 2 * dim) for _ in range(depth)]
            )
        self.modulation_dtype: ModulationDtype | None = modulation_dtype

    def rope_table(self, rope_idx: Tensor, seq_len: int) -> Tensor:
        """Rotary table for ``rope_idx`` (B, N), cut to ``seq_len`` tokens."""
        return self.rope(rope_idx)[:, :seq_len]

    def block_forward(
        self,
        i: int,
        x: Tensor,
        rope: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | None = None,
        kv: KVSplice | None = None,
    ) -> Tensor:
        """Run block ``i``, with its modulation first when the stack has one."""
        if self.modulators is None:
            return self.blocks[i](x, rope, block_mask, kv=kv)
        assert x_mod is not None, "a modulated stack needs x_mod"
        modulation = (self.modulators[i], _full_mod(x_mod), self.modulation_dtype)
        return self.blocks[i](x, rope, block_mask, modulation, kv=kv)

    def forward(
        self,
        x: Tensor,
        rope_idx: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | None = None,
        tap_blocks: Sequence[int] = (),
        kv_store: dict | None = None,
        rows: tuple[int, int] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Map ``x`` (B, N, C) to (B, N, C) and return the taps.

        ``rope_idx`` is (B, N) int64. The taps are the outputs of the blocks in
        ``tap_blocks``, in the order of ``tap_blocks``. With ``kv_store`` each
        attention layer stores (``rows`` None) or splices (``rows``) its keys
        and values, see :class:`KVSplice`.
        """
        if (
            (_fused_norm_enabled(x) or NV.ON)
            and not torch.is_grad_enabled()
            and self._can_use_generator()
        ):
            return run_solo(
                self.forward_gen(
                    x, rope_idx, x_mod, block_mask, tap_blocks, kv_store, rows
                )
            )
        rope = self.rope_table(rope_idx, x.size(-2))
        x_mod = _full_mod(x_mod)
        taps: dict[int, Tensor] = {}
        for i in range(len(self.blocks)):
            kv = None if kv_store is None else KVSplice(kv_store, i, rows)
            x = self.block_forward(i, x, rope, x_mod, block_mask, kv)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]

    def _can_use_generator(self) -> bool:
        """Keep hooks and customized block forwards on the module-call path."""
        return (
            type(self) is Transformer
            and "block_forward" not in vars(self)
            and not torch.nn.modules.module._global_forward_hooks
            and not torch.nn.modules.module._global_forward_pre_hooks
            and all(
                type(block) is TransformerBlock
                and "forward" not in vars(block)
                and not block._forward_hooks
                and not block._forward_pre_hooks
                and not block._backward_hooks
                and not block._backward_pre_hooks
                for block in self.blocks
            )
        )

    def block_forward_gen(
        self,
        i: int,
        x: Tensor,
        rope: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | None = None,
        kv: KVSplice | None = None,
        pending: Tensor | None = None,
        defer: bool = False,
    ) -> Generator[tuple[Callable[[Tensor], Tensor], Tensor], Tensor, Any]:
        """``block_forward`` as a generator (see ``TransformerBlock.forward_gen``)."""
        modulation = None
        if self.modulators is not None:
            assert x_mod is not None, "a modulated stack needs x_mod"
            modulation = (self.modulators[i], x_mod, self.modulation_dtype)
        return (
            yield from self.blocks[i].forward_gen(
                x, rope, block_mask, modulation, kv, pending, defer
            )
        )

    def forward_gen(
        self,
        x: Tensor,
        rope_idx: Tensor,
        x_mod: Tensor | None = None,
        block_mask: BlockMask | None = None,
        tap_blocks: Sequence[int] = (),
        kv_store: dict | None = None,
        rows: tuple[int, int] | None = None,
    ) -> Generator[
        tuple[Callable[[Tensor], Tensor], Tensor], Tensor, tuple[Tensor, list[Tensor]]
    ]:
        """``forward`` as a generator that yields at each feed-forward (see
        ``TransformerBlock.forward_gen``); :func:`run_pair` drives several in
        lockstep."""
        with NV.op("embed", "rope_table", block="pre"):
            rope = self.rope_table(rope_idx, x.size(-2))
        taps: dict[int, Tensor] = {}
        pending = None
        last = len(self.blocks) - 1
        for i in range(len(self.blocks)):
            NV.set_block(i)
            kv = None if kv_store is None else KVSplice(kv_store, i, rows)
            defer = (
                _fused_norm_enabled(x)
                and (self.modulators is None or FUSED_MOD_NORM)
                and i != last
                and i not in tap_blocks
            )
            r = yield from self.block_forward_gen(
                i, x, rope, x_mod, block_mask, kv, pending, defer
            )
            x, pending = r if defer else (r, None)
            if i in tap_blocks:
                taps[i] = x
        NV.set_block("post")
        return x, [taps[i] for i in tap_blocks]
