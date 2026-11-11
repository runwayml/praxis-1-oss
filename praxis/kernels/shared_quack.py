"""Shared-expert ``F.silu(g) * u`` for ``g, u = (x @ gate_up).chunk(2, -1)`` as one CuTe DSL GEMM (quack's SM100-family
tcgen05 kernel: 2-CTA cluster, 256x256 tile, persistent with the dynamic tile scheduler) whose epilogue keeps the
eager roundings: the accumulator rounded to bf16 (the bf16 GEMM output), ATen's CUDA silu in float32 (precise
expf, IEEE division) rounded to bf16, then the bf16 product. The GEMM accumulates K in the same order as cuBLAS
(no split-K), so the result has the bits of the eager expression."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Float32
from torch import Tensor

from praxis.hardware import is_thor, require_thor


def _rb(v):
    if isinstance(v, tuple):
        return type(v)(*[_rb(t) for t in v])
    return v.to(cutlass.BFloat16).to(Float32)


def _silu(v):
    if isinstance(v, tuple):
        return type(v)(*[_silu(t) for t in v])
    return v / (Float32(1.0) + cute.math.exp(-v, fastmath=False))


def _mul(a, b):
    if isinstance(a, tuple):
        return type(a)(*[_mul(x, y) for x, y in zip(a, b, strict=False)])
    return a * b


def swiglu_exact(gate, up):
    """``bf16(silu(bf16(gate))) * bf16(up)`` per accumulator pair (rounded to bf16 on store)."""
    return _mul(_rb(_silu(_rb(gate))), _rb(up))


_INSTALLED = False
_KMAJOR: dict = {}
CONFIG = {"tile_m": 256, "tile_n": 256, "cluster_m": 2, "cluster_n": 1, "dynamic": True}


def _install():
    global _INSTALLED
    if _INSTALLED:
        return
    import quack.activation as QA
    import quack.gemm_interface as QI

    QA.gate_fn_map["swiglu_exact"] = swiglu_exact
    QI.gated_to_pytorch_fn_map["swiglu_exact"] = lambda gate, up: F.silu(gate) * up
    _INSTALLED = True


def _kmajor(w: Tensor) -> Tensor:
    """A K-major copy of the (K, 2H) gate|up weight (gate and up concatenated along the non-contiguous N, the
    layout quack's concat_layout reads), cached per weight."""
    key = (w.data_ptr(), tuple(w.shape))
    t = _KMAJOR.get(key)
    if t is None:
        t = w.t().contiguous().t()
        _KMAJOR[key] = t
    return t


def supported(x: Tensor, w: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.stride(1) == 1
        and is_thor(x.device)
    )


def gate_up_silu(x: Tensor, w: Tensor) -> Tensor:
    """``(lambda gu: F.silu(gu[:, :H]) * gu[:, H:])(x @ w)`` for x (M, K), w (K, 2H); contiguous (M, H)."""
    require_thor(x.device)
    _install()
    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_act

    c = CONFIG
    cfg = GemmConfig(
        tile_m=c["tile_m"],
        tile_n=c["tile_n"],
        cluster_m=c["cluster_m"],
        cluster_n=c["cluster_n"],
        pingpong=False,
        is_dynamic_persistent=c["dynamic"],
        device_capacity=torch.cuda.get_device_capability(x.device)[0],
    )
    return gemm_act(
        x,
        _kmajor(w),
        activation="swiglu_exact",
        store_preact=False,
        concat_layout=("B",),
        config=cfg,
    )[1]
