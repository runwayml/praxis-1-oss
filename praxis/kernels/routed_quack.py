"""Routed experts as CuTe DSL grouped GEMMs (quack varlen-M kernels, tcgen05, 2-CTA clusters).

``gate_up_swiglu``: ``_swiglu_packed_kernel(grouped_mm(x[order // top_k], gate_up, offsets))`` in one launch: the
rows are gathered by the kernel (A_idx), each expert's (K, 2H) weight is read once per tile column, and the
epilogue keeps the rounding of the reference SwiGLU kernel: gate and up rounded to bf16 (the GEMM output), then
Triton's ``gate * sigmoid(gate)`` in float32 (``1 / (1 + ex2.approx(-gate * log2 e))``),
rounded to bf16, times up (bf16 product). ``down``: the grouped down GEMM (plain epilogue).

The weights are read in a layout quack's paired epilogue can use: gate|up concatenated along the non-contiguous
dim of each expert's weight (``concat_layout``), i.e. K-major. ``kmajor_inplace`` converts the (E, K, 2H)
parameter storage in place, one expert at a time (one 15.7 MB scratch), so no second copy of the bank exists.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass._mlir_helpers import math as mlir_math
from torch import Tensor

from praxis.hardware import require_thor

_LOG2E = math.log2(math.e)


def _map(fn, *vs):
    if isinstance(vs[0], tuple):
        return type(vs[0])(*[_map(fn, *z) for z in zip(*vs, strict=False)])
    return fn(*vs)


def _rb(v):
    return _map(lambda t: t.to(cutlass.BFloat16).to(Float32), v)


def _ex2(v):
    return cute.math.exp2(v, fastmath=True)


def _sig_rcp(g):
    # tl.sigmoid(g) = 1 / (1 + exp(-g)); Triton lowers f32 exp to ex2.approx(x * log2 e); the reciprocal of
    # 1 + e (>= 1, normal) by rcp.approx (what div.full computes for a unit numerator in range).
    return mlir_math.rcp(
        Float32(1.0) + _ex2((-g) * Float32(_LOG2E)), approx=True, ftz=False
    )


def _sig_rcp_ftz(g):
    return mlir_math.rcp(
        Float32(1.0) + _ex2((-g) * Float32(_LOG2E)), approx=True, ftz=True
    )


def _sig_div_rn(g):
    return Float32(1.0) / (Float32(1.0) + _ex2((-g) * Float32(_LOG2E)))


def _swiglu_with(sig):
    def fn(gate, up):
        g = _rb(gate)
        return _map(lambda a, b: a * b, _rb(_map(lambda t: t * sig(t), g)), _rb(up))

    return fn


swiglu_routed = _swiglu_with(_sig_rcp)
swiglu_routed_ftz = _swiglu_with(_sig_rcp_ftz)
swiglu_routed_divrn = _swiglu_with(_sig_div_rn)
VARIANTS = {
    "swiglu_routed": swiglu_routed,
    "swiglu_routed_ftz": swiglu_routed_ftz,
    "swiglu_routed_divrn": swiglu_routed_divrn,
}
ACT = "swiglu_routed"
_INSTALLED = False
GATE_UP_CONFIG = {
    "tile_m": 256,
    "tile_n": 256,
    "cluster_m": 2,
    "cluster_n": 1,
    "dynamic": True,
}
DOWN_CONFIG = {
    "tile_m": 256,
    "tile_n": 256,
    "cluster_m": 2,
    "cluster_n": 1,
    "dynamic": True,
}


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import quack.activation as QA
    import quack.gemm_interface as QI

    from praxis.kernels.layers_triton import swiglu_packed

    def ref(gate, up):
        return swiglu_packed(torch.cat([gate, up], -1))

    for k, f in VARIANTS.items():
        QA.gate_fn_map[k] = f
        QI.gated_to_pytorch_fn_map[k] = ref
    _INSTALLED = True


def _cfg(c: dict, dev) -> object:
    from quack.gemm_config import GemmConfig

    return GemmConfig(
        tile_m=c["tile_m"],
        tile_n=c["tile_n"],
        cluster_m=c["cluster_m"],
        cluster_n=c["cluster_n"],
        pingpong=False,
        is_dynamic_persistent=c["dynamic"],
        use_tma_gather=c.get("tma_gather", False),
        device_capacity=torch.cuda.get_device_capability(dev)[0],
    )


def cu_seqlens(offsets: Tensor) -> Tensor:
    """(E,) cumulative int32 row ends -> (E + 1,) starting with 0."""
    return torch.cat([offsets.new_zeros(1), offsets])


def is_kmajor(w: Tensor) -> bool:
    return w.dim() == 3 and w.stride(1) == 1 and w.stride(2) == w.shape[1]


def kmajor_inplace(p: torch.nn.Parameter) -> None:
    """Rewrite an (E, K, N) n-major parameter's storage so it holds each expert's (N, K) row-major matrix, and
    make the parameter that storage seen as (E, K, N) (strides (K N, 1, K)). Same values; no second bank."""
    w = p.data
    if is_kmajor(w):
        return
    e, k, n = w.shape
    assert w.is_contiguous()
    flat = w.view(e, n * k)
    with torch.no_grad():
        for i in range(e):
            t = w[i].t().contiguous()  # (N, K)
            flat[i].copy_(t.view(-1))
    p.data = torch.as_strided(w, (e, k, n), (k * n, 1, k))


def gate_up_swiglu(
    x: Tensor,
    w: Tensor,
    offsets: Tensor,
    order: Tensor,
    top_k: int,
    act: str | None = None,
    config: dict | None = None,
) -> Tensor:
    """(M, H) for M = order.numel() rows ``x[order[i] // top_k]``; w (E, K, 2H) K-major."""
    require_thor(x.device)
    install()
    from quack.gemm_interface import gemm_act

    a_idx = (order // top_k).to(torch.int32)
    return gemm_act(
        x,
        w,
        activation=act or ACT,
        store_preact=False,
        concat_layout=("B",),
        cu_seqlens_m=cu_seqlens(offsets),
        A_idx=a_idx,
        config=_cfg(config or GATE_UP_CONFIG, x.device),
    )[1]


def down(h: Tensor, w: Tensor, offsets: Tensor, config: dict | None = None) -> Tensor:
    require_thor(h.device)
    from quack.gemm_interface import gemm_act

    # gemm_act without an activation is the plain GEMM; unlike gemm() it takes an explicit config (quack's
    # autotuner and default configs have no SM110 entries).
    return gemm_act(
        h,
        w,
        activation=None,
        store_preact=False,
        cu_seqlens_m=cu_seqlens(offsets),
        config=_cfg(config or DOWN_CONFIG, h.device),
    )[1]
