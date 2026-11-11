"""All-CuTe-DSL expert GEMMs. quack 0.6.5 GemmSm100 (tcgen05, persistent,
2-CTA clusters) with configs tuned on Thor. Every entry point is
bitwise with the current path (checked on Thor incl. skewed real routing with empty experts and 2^-12..2^5 stress
rows).

    shared_gate_up_silu(x, w_gu_kmajor)        F.silu(g) * u for g, u = (x @ gate_up).chunk(2, -1)   (one GEMM)
    shared_down(h, w_down)                     h @ down                                               (one GEMM)
    routed_gate_up_swiglu(x, w_gu_kmajor, offsets, order, top_k)
                                               swiglu_packed(grouped_mm(x[order // top_k], gate_up))  (one GEMM)
    routed_down(h, w_down, offsets)            grouped_mm(h, down, offsets)                            (one GEMM)

Rules learned on Thor:
* The plain GEMMs (down projections) must use quack's default epilogue (``gemm_tuned.fn``, D store path), not
  ``gemm_act(activation=None)``.
* quack's ``gemm_act`` frontend keys its plan cache by ``id(config)``: pass a module-level GemmConfig (never a
  per-call temporary — a recycled id replays another config's plan).
* quack has no SM110 autotune entries: always an explicit GemmConfig(device_capacity=11, pingpong=False).
* Exact epilogues use the branch-free IEEE division of praxis/kernels/cute_act.py (``swiglu_exact_nr``,
  ``swiglu_routed_nr``): ptxas lowers div.rn.f32 to a per-element slow-path CALL that serializes the epilogue.
  Same bits for all 65,280 finite bf16 gate values.
* Routed gate_up: max_swizzle_size=1 (each expert's M tiles run back to back per weight N column:
  weight-stationary order), dynamic persistent scheduler.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from praxis.hardware import is_thor, require_thor

C, H = 3072, 1280


@functools.cache
def _cfg(tile_m, tile_n, cm, cn, dyn, sw, swap=False, tg=False):
    """One live GemmConfig per config (see the id(config) note above)."""
    from quack.gemm_config import GemmConfig

    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cm,
        cluster_n=cn,
        pingpong=False,
        is_dynamic_persistent=dyn,
        swap_ab=swap,
        max_swizzle_size=sw,
        use_tma_gather=tg,
        device_capacity=torch.cuda.get_device_capability()[0],
    )


# ---- configs (Thor, cold L2) -------------------------------------------------------------------------------------
def shared_down_config(m: int):
    # swap-AB (D^T = down^T h^T) 2-CTA 256x224 tiles; 4-CTA cluster (B multicast) from ~3k rows.
    return (
        _cfg(256, 224, 4, 1, False, 8, swap=True)
        if m >= 3000
        else _cfg(256, 224, 2, 1, False, 8, swap=True)
    )


def shared_gate_up_config(m: int):
    if m >= 3000:
        return _cfg(256, 256, 2, 2, True, 8)
    if m >= 2048:
        return _cfg(128, 256, 4, 1, False, 4)
    return _cfg(256, 256, 2, 1, False, 8)


def routed_gate_up_config(rows: int):
    if rows >= 48_000:  # decoder (9,988 x 8) and the encoder pair at 2 x 3,964
        return _cfg(256, 256, 2, 1, True, 1)
    if rows >= 40_000:  # encoder pair 2 x 2,688
        return _cfg(256, 256, 4, 1, True, 8)
    return _cfg(128, 256, 4, 1, False, 8)  # encoder pairs 2 x 1,468 / 2 x 1,340


def routed_down_config(rows: int):
    return _cfg(256, 256, 2, 1, True, 1)


# ---- epilogue activations ----------------------------------------------------------------------------------------
_INSTALLED = False


def install():
    """Register the exact-rounding gated epilogues with quack: the impl's div.rn versions and the branch-free
    versions of praxis/kernels/cute_act.py (exhaustively bitwise-equal over all bf16 gate values)."""
    global _INSTALLED
    if _INSTALLED:
        return
    from praxis.kernels import cute_act as CA
    from praxis.kernels import routed_quack as RQ
    from praxis.kernels import shared_quack as SQ

    CA.install()
    SQ._install()  # "swiglu_exact": bf16(silu(bf16 gate)) * bf16(up), precise expf + IEEE div (ATen F.silu)
    RQ.install()  # "swiglu_routed_divrn": Triton _swiglu_packed_kernel rounding (ex2.approx, IEEE div)
    _INSTALLED = True


# ---- entry points ------------------------------------------------------------------------------------------------
def kmajor(w: Tensor) -> Tensor:
    """(..., K, N) n-major -> same shape, K-major storage (what concat_layout=("B",) reads for gate|up)."""
    return w.transpose(-1, -2).contiguous().transpose(-1, -2)


def shared_gate_up_silu(x: Tensor, w_kmajor: Tensor, config=None) -> Tensor:
    """x (M, 3072) bf16 row-major; w_kmajor (3072, 2560) = gate|up K-major (kmajor(gate_up)). -> (M, 1280)."""
    require_thor(x.device)
    install()
    from quack.gemm_interface import gemm_act

    return gemm_act(
        x,
        w_kmajor,
        activation="swiglu_exact_nr",
        store_preact=False,
        concat_layout=("B",),
        config=config or shared_gate_up_config(x.shape[0]),
    )[1]


def shared_down(h: Tensor, w: Tensor, out: Tensor | None = None, config=None) -> Tensor:
    """h (M, 1280) bf16 row-major; w (1280, 3072) as stored (n-major). -> (M, 3072)."""
    require_thor(h.device)
    from quack.gemm_interface import gemm_tuned

    if out is None:
        out = h.new_empty(h.shape[0], w.shape[1])
    gemm_tuned.fn(h, w, out, C=None, config=config or shared_down_config(h.shape[0]))
    return out


def cu_seqlens(offsets: Tensor) -> Tensor:
    """(E,) cumulative int32 row ends -> (E + 1,) starting with 0."""
    return torch.cat([offsets.new_zeros(1), offsets])


def routed_gate_up_swiglu(
    x: Tensor,
    w_kmajor: Tensor,
    offsets: Tensor,
    order: Tensor,
    top_k: int,
    config=None,
    a_idx: Tensor | None = None,
) -> Tensor:
    """Rows x[order[i] // top_k] (gathered in-kernel), expert e = rows offsets[e-1]:offsets[e];
    w_kmajor (E, 3072, 2560) K-major (routed_quack.kmajor_inplace). -> (rows, 1280)."""
    require_thor(x.device)
    install()
    from quack.gemm_interface import gemm_act

    rows = order.numel()
    if a_idx is None:  # ROUTER_FUSED passes it (int32 order // top_k, the same values)
        a_idx = (order // top_k).to(torch.int32)
    return gemm_act(
        x,
        w_kmajor,
        activation="swiglu_routed_nr",
        store_preact=False,
        concat_layout=("B",),
        cu_seqlens_m=cu_seqlens(offsets),
        A_idx=a_idx,
        config=config or routed_gate_up_config(rows),
    )[1]


def routed_down(
    h: Tensor, w: Tensor, offsets: Tensor, out: Tensor | None = None, config=None
) -> Tensor:
    """h (rows, 1280); w (E, 1280, 3072) as stored (n-major). -> (rows, 3072)."""
    require_thor(h.device)
    from quack.gemm_interface import gemm_tuned

    if out is None:
        out = h.new_empty(h.shape[0], w.shape[2])
    gemm_tuned.fn(
        h,
        w,
        out,
        C=None,
        cu_seqlens_m=cu_seqlens(offsets),
        config=config or routed_down_config(h.shape[0]),
    )
    return out


# ---- weight layouts ------------------------------------------------------------------------------------------------
_KMAJOR: dict = {}


def kmajor_cached(w: Tensor) -> Tensor:
    """A K-major copy of a 2-D shared gate|up weight (7.9 MB per layer), cached per weight."""
    key = (w.data_ptr(), tuple(w.shape))
    t = _KMAJOR.get(key)
    if t is None:
        with torch.no_grad():
            t = kmajor(w)
        _KMAJOR[key] = t
    return t


def routed_bank_kmajor(p: torch.nn.Parameter) -> Tensor:
    """Make the (E, K, 2H) routed gate|up bank K-major in place (one expert at a time; no second bank) and return
    it. The parameter keeps its shape and values; only its strides change, which every consumer (the Triton
    grouped kernels take strides) reads correctly."""
    from praxis.kernels import routed_quack as RQ

    if not RQ.is_kmajor(p.data):
        with torch.no_grad():
            RQ.kmajor_inplace(p)
    return p.data


def supported(device: torch.device | str | None = None) -> bool:
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda", torch.cuda.current_device())
    return is_thor(device)
