"""Branch-free exact SwiGLU epilogues for the quack gated GEMMs.

The existing exact epilogues divide with IEEE ``div.rn.f32``. On SM100/SM110 ptxas lowers that to a MUFU.RCP fast path
plus an FCHK test and a CALL to a slow-path subroutine per element; the per-element call splits the epilogue into
basic blocks, so the independent elements of a subtile cannot be interleaved and 4 epilogue warps run latency-bound.

``_div_rn(n, d)`` computes the same correctly rounded quotient for d in [1, +inf] without the call: the operands are
scaled by 2^-64 when d > 2^64 (exact; the quotient is unchanged and the reciprocal stays normal, so the ftz rcp is
safe), then MUFU.RCP + one Newton step for the reciprocal, q0 = n * r, and one FMA residual correction
(rem = n - d*q0 exact by FMA, q = q0 + rem*r) - the fast path of the PTX div.rn sequence. d = +inf and n = +inf
take the IEEE results by select. Because the GEMM output is rounded to bf16 before the activation, each epilogue is
a function of one bf16 input: a probe checked it against the reference kernel for ALL
65,536 bf16 gate values (with random up values), through the real fused GEMM.

    swiglu_exact_nr       == shared_quack.swiglu_exact  (ATen F.silu: g / (1 + expf(-g)), precise expf, IEEE div)
    swiglu_routed_nr      == routed_quack.swiglu_routed_divrn (Triton: g * (1 / (1 + ex2.approx(-g log2 e))))
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import arith
from cutlass._mlir_helpers import math as mlir_math

_LOG2E = math.log2(math.e)
_INF = float("inf")


def _map(fn, *vs):
    if isinstance(vs[0], tuple):
        return type(vs[0])(*[_map(fn, *z) for z in zip(*vs, strict=False)])
    return fn(*vs)


def _rb(v):
    return _map(lambda t: t.to(cutlass.BFloat16).to(Float32), v)


def _sel(pred, a, b):
    return Float32(arith.select(pred, Float32(a).ir_value(), Float32(b).ir_value()))


def _cmp(pred, a, b):
    return arith.cmpf(pred, Float32(a).ir_value(), Float32(b).ir_value())


def _div_rn(n, d):
    """IEEE round-to-nearest n / d for d in [1, +inf] (n any), branch-free."""
    big = _cmp(arith.CmpFPredicate.OGT, d, Float32(2.0**64))
    s = _sel(big, Float32(2.0**-64), Float32(1.0))
    n2 = n * s
    d2 = d * s
    r = mlir_math.rcp(d2, approx=True, ftz=True)
    e = mlir_math.fma(-d2, r, Float32(1.0))
    r = mlir_math.fma(r, e, r)
    q = n2 * r
    rem = mlir_math.fma(-d2, q, n2)
    q = mlir_math.fma(rem, r, q)
    q = _sel(_cmp(arith.CmpFPredicate.OEQ, d, Float32(_INF)), n * Float32(0.0), q)
    q = _sel(_cmp(arith.CmpFPredicate.OEQ, n, Float32(_INF)), n, q)
    return q


def _silu_aten(g):
    return _div_rn(g, Float32(1.0) + cute.math.exp(-g, fastmath=False))


def _silu_triton(g):
    sig = _div_rn(
        Float32(1.0),
        Float32(1.0) + cute.math.exp2((-g) * Float32(_LOG2E), fastmath=True),
    )
    return g * sig


def swiglu_exact_nr(gate, up):
    """bf16(silu(bf16(gate))) * bf16(up), ATen rounding (precise expf, IEEE division)."""
    return _map(lambda a, b: a * b, _rb(_map(_silu_aten, _rb(gate))), _rb(up))


def swiglu_routed_nr(gate, up):
    """bf16(bf16(gate) * (1 / (1 + ex2.approx(-bf16(gate) log2 e)))) * bf16(up), Triton rounding."""
    return _map(lambda a, b: a * b, _rb(_map(_silu_triton, _rb(gate))), _rb(up))


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import quack.activation as QA
    import quack.gemm_interface as QI
    import torch.nn.functional as F

    QA.gate_fn_map["swiglu_exact_nr"] = swiglu_exact_nr
    QI.gated_to_pytorch_fn_map["swiglu_exact_nr"] = lambda gate, up: F.silu(gate) * up
    QA.gate_fn_map["swiglu_routed_nr"] = swiglu_routed_nr
    try:
        import torch

        from praxis.kernels.layers_triton import swiglu_packed

        QI.gated_to_pytorch_fn_map["swiglu_routed_nr"] = lambda gate, up: swiglu_packed(
            torch.cat([gate, up], -1)
        )
    except ImportError:
        pass
    _INSTALLED = True
