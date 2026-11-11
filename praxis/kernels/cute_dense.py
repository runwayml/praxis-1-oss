"""Dense SwiGLU feed-forward on a CuTe DSL gate|up GEMM (``layers.CUTE_DENSE_FFN``).

A memory-traffic fusion for tcgen05 GPUs (SM100/SM103/SM110). The dense FFN of encoder blocks 0-1 and decoder
block 0 (``layers.SwiGLUFeedForward``: dim 3072, hidden 11520) runs eagerly as five kernels: ``a = x @ A^T`` and
``b = x @ B^T`` (two cuBLAS GEMMs), ``F.silu(a)`` and ``* b`` (two elementwise kernels), then ``h @ W^T``
(cuBLAS). With ``layers.CUTE_DENSE_FFN``:

    gate_up_silu(x, w_cat_t)   F.silu(x @ A^T) * (x @ B^T)    ONE Quack tcgen05 GEMM over [A; B] with
                               SiLU(gate) * up in the epilogue ("swiglu_exact_nr" of cute_act: bf16(silu(bf16 gate))
                               * bf16(up), ATen's precise expf + IEEE division); the gate/up outputs never reach
                               memory, only the (M, 11520) hidden
    down                       h @ W^T, stock F.linear (cuBLAS) unless the architecture's down table names a Quack
                               config for the row count (Thor: cuBLAS wins at every row count).

Row dispatch is per architecture: ``GATE_UP_TABLES`` / ``DOWN_TABLES`` map a CUDA capability to
((lo, hi, config | None), ...); the first entry with lo <= rows < hi decides; None, no entry or no table for the
device -> the ordinary ops. Only SM110 (Thor) has a measured table. SM100/SM103 have none, so they run the ordinary
ops until ``scripts/tune_cute_dense.py`` has been run there: it checks every candidate config bitwise against the
eager module, times it, and prints the table to add here. SM90 and SM120 are not selected (no tcgen05 path).

Exactness: no split-K, K accumulated in order like cuBLAS, the gate and up accumulators rounded to bf16 before the
activation; the tuning script checks every config bitwise against the eager module before it can be picked.
Validate a new table with a bitwise action / command / latent comparison of the full policy, flag off vs on.

The concatenated weight: Quack's concat_layout=("B",) reads gate|up as ONE (K, 2H) operand. ``packed_weight``
makes ``a_weight`` and ``b_weight`` views of one (2H, K) buffer (in place, once per module; same values, no second
copy kept), so the K-major (K, 2H) operand is a transpose view. cuBLAS gives the same bits on the views.

Plan cache: Quack's gemm_act keys its compiled plans by id(config). Every config here comes from
``cute_experts._cfg`` (one live GemmConfig per config for the process), never a per-call temporary.
"""

from __future__ import annotations

import functools
import importlib.util

import torch
from torch import Tensor

from praxis.hardware import is_blackwell_tcgen05

# (lo, hi, (tile_m, tile_n, cluster_m, cluster_n, dynamic, max_swizzle, swap_ab) | None); first match wins.
_BIG = 1 << 30
# SM110 (Thor): every config is bitwise; the fused GEMM beats the eager 2 GEMMs + silu + mul at every probed
# count, 256..17,480 rows. Below 256 rows: not probed -> ordinary ops.
_SM110_GATE_UP: tuple = (
    (256, 2048, (256, 256, 2, 2, True, 8, False)),
    (2048, 4608, (256, 256, 2, 1, True, 8, False)),
    (4608, 6144, (256, 224, 2, 1, True, 8, False)),
    (6144, 8192, (256, 256, 2, 1, True, 4, False)),
    (8192, _BIG, (256, 256, 2, 2, True, 8, False)),
)
GATE_UP_TABLES: dict[tuple[int, int], tuple] = {(11, 0): _SM110_GATE_UP}
# Down: on Thor cuBLAS wins at every real count. No entries.
DOWN_TABLES: dict[tuple[int, int], tuple] = {}


def _capability(device: torch.device | str | None = None) -> tuple[int, int]:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return tuple(torch.cuda.get_device_capability(device))


def _pick(table, m: int):
    from praxis.kernels.cute_experts import _cfg

    for lo, hi, c in table:
        if lo <= m < hi:
            return None if c is None else _cfg(*c)
    return None


def gate_up_config(m: int, capability: tuple[int, int] | None = None):
    """The gate|up config of ``m`` rows on this device (None: ordinary ops)."""
    return _pick(GATE_UP_TABLES.get(capability or _capability(), ()), m)


def down_config(m: int, capability: tuple[int, int] | None = None):
    """The down config of ``m`` rows on this device (None: cuBLAS)."""
    return _pick(DOWN_TABLES.get(capability or _capability(), ()), m)


def has_down_table() -> bool:
    return bool(DOWN_TABLES.get(_capability()))


@functools.cache
def quack_available() -> bool:
    return importlib.util.find_spec("quack") is not None


def supported() -> bool:
    """A tcgen05 Blackwell GPU (SM100/SM103/SM110) with Quack installed. The row
    count still needs an entry in that architecture's table."""
    return (
        torch.cuda.is_available()
        and is_blackwell_tcgen05(torch.device("cuda", torch.cuda.current_device()))
        and quack_available()
    )


def _require_tcgen05(device: torch.device) -> None:
    if not is_blackwell_tcgen05(device):
        raise RuntimeError("cute_dense needs a tcgen05 GPU (SM100/SM103/SM110)")


def _adjacent(a: Tensor, b: Tensor) -> bool:
    return (
        a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
        and a.dtype == b.dtype
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and b.data_ptr() == a.data_ptr() + a.numel() * a.element_size()
    )


def packed_weight(a_param: torch.nn.Parameter, b_param: torch.nn.Parameter) -> Tensor:
    """The (K, 2H) K-major gate|up operand: ``cat([A, B]).t()`` with A, B the (H, K) weights.

    On first use the two parameters are re-pointed at the halves of one (2H, K) buffer (same values and
    shapes; the old storages are released), so later calls take the view for free."""
    a, b = a_param.data, b_param.data
    if not _adjacent(a, b):
        with torch.inference_mode(False), torch.no_grad():
            cat = torch.cat([a, b], 0)
            h = a.shape[0]
            a_param.data = cat[:h]
            b_param.data = cat[h:]
        a, b = a_param.data, b_param.data
    h, k = a.shape
    return a.as_strided((2 * h, k), (k, 1)).t()


def gate_up_silu(x: Tensor, w_cat_t: Tensor, config) -> Tensor:
    """x (M, K) bf16 row-major, w_cat_t (K, 2H) K-major = cat([A, B]).t() -> F.silu(x A^T) * (x B^T), (M, H)."""
    _require_tcgen05(x.device)
    from quack.gemm_interface import gemm_act

    from praxis.kernels import cute_experts as CE

    CE.install()
    return gemm_act(
        x,
        w_cat_t,
        activation="swiglu_exact_nr",
        store_preact=False,
        concat_layout=("B",),
        config=config,
    )[1]


def down(h: Tensor, w: Tensor, config, out: Tensor | None = None) -> Tensor:
    """h (M, H) bf16 row-major, w (N, H) the nn.Linear weight -> h @ w.T, (M, N)."""
    _require_tcgen05(h.device)
    from quack.gemm_interface import gemm_tuned

    if out is None:
        out = h.new_empty(h.shape[0], w.shape[0])
    gemm_tuned.fn(h, w.t(), out, C=None, config=config)
    return out


def ok(x: Tensor, *ws: Tensor) -> bool:
    """The fused path applies: CUDA bf16 inference with a row-contiguous input."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() >= 2
        and x.stride(-1) == 1
        and x.numel() > 0
        and all(w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2 for w in ws)
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled()
    )
