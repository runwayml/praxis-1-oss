"""The exact bilinear resizes of the tile canvas, on any device.

The tile canvas resizes each view with ``torch.nn.functional.interpolate``
(bilinear, ``align_corners=False``, with or without antialias) in float32 on
the CPU, then rounds and clamps to uint8. The CUDA kernel of the same call
computes the sums in another order, and the CPU kernel itself fuses
multiply-adds or not depending on the build and the CPU, so in general a
resize on another device can differ from the CPU resize by one pixel level.

A resize is *exact* (:func:`exact`) when every product and every sum of it
is a float32 value with no rounding, for any uint8 input: the weights are
nonnegative dyadic fractions with few enough bits (the 9/8 downscale of
288x468 to 256x416, a 2x downscale, ...). An exact resize gives the exact
real result in any order, with or without fused multiply-adds, so
:func:`resize` on a CUDA device gives the bits of the CPU ``interpolate``.
Other resizes stay on the CPU.

* The weights are the CPU kernel's own weights. They are read out of the CPU
  ``interpolate`` call itself: a resize of a one-hot basis along one axis
  returns each weight exactly (one product by 1.0, sums with zeros).
* :func:`resize` resizes the width, then the height. Each output value is
  ``x[i0] * w0``, then ``+ x[i] * w`` for each next tap: one gather, one
  product and one sum per tap. An axis whose size does not change is not
  resized.

``tests/test_tile_resize.py`` checks :func:`resize` against ``interpolate``
bit for bit on the CPU, and the CUDA result against the CPU result.
"""

from __future__ import annotations

import functools
import threading

import torch

# A float32 significand: an integer below 2**24 times a power of two is exact.
_FLOAT32_BITS = 24

_lock = threading.Lock()
# (n_in, n_out, antialias, device) -> (index (n_out, K) int64, weight (n_out, K) float32)
_TAPS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def cpu_weights(n_in: int, n_out: int, antialias: bool) -> torch.Tensor:
    """Return the ``(n_out, n_in)`` float32 weights of the CPU resize of one axis.

    A one-hot basis (``n_in`` channels, one row of ``n_in`` pixels) resized
    to one row of ``n_out`` pixels: channel ``j`` of output ``i`` is the
    weight of input ``j`` in output ``i``, as the CPU kernel computes it.
    """
    eye = torch.eye(n_in, dtype=torch.float32).view(1, n_in, 1, n_in)
    out = torch.nn.functional.interpolate(
        eye,
        size=(1, n_out),
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )
    return out[0, :, 0, :].t().contiguous()


def _taps(
    n_in: int, n_out: int, antialias: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input index and weight of each tap of each output, on ``device``.

    Tap ``k`` of output ``i`` reads input ``first_i + k``, from the first
    nonzero weight of the output on. Taps past the support have weight 0 (a
    sum with +0.0 keeps the value).
    """
    key = (n_in, n_out, bool(antialias), device)
    with _lock:
        hit = _TAPS.get(key)
    if hit is not None:
        return hit
    w = cpu_weights(n_in, n_out, antialias)
    nz = w != 0
    if not bool(nz.any(dim=1).all()):
        raise ValueError(f"an output of the {n_in} -> {n_out} resize has no weight")
    cols = torch.arange(n_in)
    first = torch.where(nz, cols, n_in).amin(dim=1)
    last = torch.where(nz, cols, -1).amax(dim=1)
    k = int((last - first).max()) + 1
    idx = first[:, None] + torch.arange(k)[None, :]
    inside = idx < n_in
    idx = idx.clamp(max=n_in - 1)
    weight = torch.where(inside, w.gather(1, idx), torch.zeros(()))
    hit = (idx.to(device), weight.to(device))
    with _lock:
        _TAPS[key] = hit
    return hit


def _axis_bits(n_in: int, n_out: int, antialias: bool) -> tuple[int, float] | None:
    """The fraction bits and the largest weight sum of one axis, or ``None``.

    Every weight of the axis is an integer times ``2 ** -bits``. ``None``:
    a negative weight, or a weight with more than 24 fraction bits.
    """
    if n_in == n_out:
        return 0, 1.0
    w = cpu_weights(n_in, n_out, antialias).double()
    if bool((w < 0).any()):
        return None
    for bits in range(_FLOAT32_BITS + 1):
        scaled = w * 2.0**bits
        if torch.equal(scaled, scaled.round()):
            return bits, float(w.sum(dim=1).max())
    return None


@functools.lru_cache(maxsize=64)
def exact(in_hw: tuple[int, int], size: tuple[int, int], antialias: bool) -> bool:
    """True when the resize of ``in_hw`` to ``size`` rounds nothing in float32.

    The inputs are integers in [0, 255]. The width pass gives multiples of
    ``2 ** -bits_w`` up to ``255 * sum_w``; the height pass multiples of
    ``2 ** -(bits_w + bits_h)`` up to ``255 * sum_w * sum_h`` (the products
    and the partial sums of nonnegative weights stay below the total). Each
    is exact when it is below ``2 ** 24`` of its units. The bound does not
    depend on the order of the passes or of the sums.
    """
    axes = [
        _axis_bits(int(n_in), int(n_out), bool(antialias))
        for n_in, n_out in zip(in_hw, size, strict=True)
    ]
    if None in axes:
        return False
    (bits_h, sum_h), (bits_w, sum_w) = axes
    top = 255.0 * max(1.0, sum_w) * max(1.0, sum_h)
    return top * 2.0 ** (bits_h + bits_w) < 2.0**_FLOAT32_BITS


def _resize_axis(
    x: torch.Tensor, n_out: int, dim: int, antialias: bool
) -> torch.Tensor:
    """Resize float32 ``x`` along ``dim`` (-1 or -2), one tap at a time."""
    idx, weight = _taps(x.shape[dim], n_out, antialias, x.device)
    shape = (n_out, 1) if dim == -2 else (n_out,)
    out = None
    for k in range(idx.shape[1]):
        term = x.index_select(dim, idx[:, k]).mul_(weight[:, k].view(shape))
        out = term if out is None else out.add_(term)
    return out


def resize(x: torch.Tensor, size: tuple[int, int], antialias: bool) -> torch.Tensor:
    """Bilinear resize of float32 ``(N, C, H, W)`` ``x`` to ``size``.

    The weights, dtype and memory format are those of ``interpolate(x, size,
    mode="bilinear", align_corners=False, antialias=antialias)`` on the CPU.
    The values are the CPU values when the resize is :func:`exact` and the
    values of ``x`` are integers in [0, 255]; otherwise they can differ
    slightly (for such shapes the CPU values themselves change with the
    number of CPU threads).
    """
    if x.dtype != torch.float32 or x.dim() != 4:
        raise ValueError(
            f"needs a float32 (N, C, H, W) tensor, got {x.dtype} {tuple(x.shape)}"
        )
    h, w = (int(s) for s in size)
    fmt = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous()
        else torch.contiguous_format
    )
    out = x
    if out.shape[-1] != w:
        out = _resize_axis(out, w, -1, antialias)
    if out.shape[-2] != h:
        out = _resize_axis(out, h, -2, antialias)
    if out is x:
        out = x.clone()
    return out.contiguous(memory_format=fmt)
