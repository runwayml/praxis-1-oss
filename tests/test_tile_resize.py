"""The exact canvas resizes of praxis.tile_resize and their CUDA tile path.

The CPU tests check :func:`tile_resize.resize` against ``interpolate`` bit
for bit and the choice of the tile path. A CPU tensor that reports
``is_cuda`` (:class:`OnDevice`) drives the device branch of
:func:`single_arm._resize` without a GPU. The CUDA tests (skipped without a
GPU) check the CUDA canvas against the CPU canvas bit for bit.
"""

import numpy as np
import pytest
import torch

from praxis import single_arm as sa
from praxis import tile_resize

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def no_architecture_query(*args):
    pytest.fail("The exact tile path must not query the GPU architecture")


class OnDevice(torch.Tensor):
    """A CPU tensor that reports ``is_cuda``: the device branch of ``_resize``."""

    @property
    def is_cuda(self):
        return True


def spy_resize(monkeypatch):
    """Record the ``(in_hw, size, antialias)`` of each ``tile_resize.resize`` call."""
    calls = []
    resize = tile_resize.resize

    def spy(x, size, antialias):
        calls.append((tuple(x.shape[-2:]), tuple(size), antialias))
        return resize(x, size, antialias)

    monkeypatch.setattr(tile_resize, "resize", spy)
    return calls


def _frames(shape, seed=0, cl=True):
    """Float32 ``(T, 3, H, W)`` uint8 values, channels last as the canvas makes them."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    # Saturated pixels: the largest products and sums.
    a[0, :4] = 255
    fr = torch.from_numpy(a).permute(0, 3, 1, 2)
    return (fr if cl else fr.contiguous()).float()


def _interpolate(x, size, antialias):
    return torch.nn.functional.interpolate(
        x, size=size, mode="bilinear", align_corners=False, antialias=antialias
    )


@pytest.mark.parametrize(
    ("in_hw", "size", "antialias", "want"),
    [
        ((288, 468), (256, 416), False, True),  # 288x512 views, 256x416 tiles
        ((512, 832), (256, 416), False, True),  # 2x
        ((256, 416), (256, 416), False, True),  # no resize
        ((234, 520), (234, 416), False, True),  # one axis
        ((300, 500), (256, 416), False, False),
        ((276, 512), (224, 416), False, False),
        ((720, 1280), (234, 416), True, False),  # the fit rule of praxis-1-droid
        ((288, 512), (234, 416), True, False),
        ((117, 208), (234, 416), True, True),  # an antialiased 2x upscale
    ],
)
def test_exact_shapes(in_hw, size, antialias, want):
    assert tile_resize.exact(in_hw, size, antialias) is want


@pytest.mark.parametrize("cl", [True, False])
@pytest.mark.parametrize(
    ("in_hw", "size", "antialias"),
    [
        ((288, 468), (256, 416), False),
        ((512, 832), (256, 416), False),
        ((234, 520), (234, 416), False),
        ((300, 500), (256, 416), False),
        ((720, 1280), (234, 416), True),
        ((288, 512), (234, 416), True),
        ((117, 208), (234, 416), True),
    ],
)
def test_resize_matches_cpu_interpolate_when_exact(cl, in_hw, size, antialias):
    """Exact: the bits of interpolate. Always: its dtype, shape and strides.

    Not exact: within one uint8 level after the round. For these shapes the
    CPU ``interpolate`` itself changes with the number of CPU threads, by
    more than the last bit, so no tighter bound holds on every machine.
    """
    x = _frames((3, *in_hw), cl=cl)
    want = _interpolate(x, size, antialias)
    got = tile_resize.resize(x, size, antialias)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.stride() == want.stride()
    if tile_resize.exact(in_hw, size, antialias):
        assert torch.equal(got, want)
    else:
        assert (got.round() - want.round()).abs().max() <= 1


def test_exact_bound_is_24_bits():
    """255 times 2 ** bits must stay below 2 ** 24: 16 fraction bits, not 17.

    A 128x upscale has 8 fraction bits per axis, a 256x upscale 9.
    """
    assert tile_resize._axis_bits(2, 256, False) == (8, 1.0)
    assert tile_resize._axis_bits(2, 512, False) == (9, 1.0)
    assert tile_resize.exact((2, 2), (256, 256), False)
    assert not tile_resize.exact((2, 2), (256, 512), False)
    assert not tile_resize.exact((2, 2), (512, 256), False)
    x = _frames((2, 2, 2))
    want = _interpolate(x, (256, 256), False)
    assert torch.equal(tile_resize.resize(x, (256, 256), False), want)


def test_exact_is_sound_on_a_sweep():
    """Every shape that exact() accepts gives the CPU bits."""
    rng = np.random.default_rng(1)
    accepted = 0
    for _ in range(300):
        size = (int(rng.integers(4, 40)), int(rng.integers(4, 40)))
        in_hw = tuple(
            int(n * rng.choice([1, 2, 4, 3 / 2, 5 / 4, 9 / 8, 7 / 5, 1 / 2]))
            for n in size
        )
        antialias = bool(rng.integers(0, 2))
        if min(in_hw) < 1 or not tile_resize.exact(in_hw, size, antialias):
            continue
        accepted += 1
        x = _frames((2, *in_hw), seed=accepted)
        want = _interpolate(x, size, antialias)
        assert torch.equal(tile_resize.resize(x, size, antialias), want), (
            in_hw,
            size,
            antialias,
        )
    assert accepted > 20


@pytest.mark.parametrize(
    ("shape", "tile", "antialias", "want"),
    [
        ((288, 512), (256, 416), False, True),  # exact resize
        ((234, 416), (224, 416), True, True),  # no resize
        ((288, 512), (224, 416), False, False),
        ((288, 512), (224, 416), True, False),
        ((720, 1280), (224, 416), True, False),
        ((117, 208), (224, 416), True, True),  # exact 2x upscale to the fit
        ((117, 240), (224, 416), True, True),  # crop to 117x208, then 2x
        ((130, 208), (224, 416), True, True),  # crop to 117x208, then 2x
        ((120, 208), (224, 416), True, False),
    ],
)
def test_tile_on_device_takes_exact_resizes(monkeypatch, shape, tile, antialias, want):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(sa, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(sa, "THOR_TILE_RESIZE", False)
    frames = np.zeros((2, *shape, 3), dtype=np.uint8)
    on = sa._tile_on_device(frames, frames, *tile, 0.03, "cuda", antialias)
    assert on is want
    if shape[-2:] != sa.fit_resize_size(tile[1]):
        # The off switch keeps every resize on the CPU.
        monkeypatch.setattr(sa, "TILE_ON_DEVICE_RESIZE", False)
        assert not sa._tile_on_device(frames, frames, *tile, 0.03, "cuda", antialias)


def test_tile_on_device_checks_both_views(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(sa, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(sa, "THOR_TILE_RESIZE", False)
    exact = np.zeros((1, 288, 512, 3), dtype=np.uint8)
    other = np.zeros((1, 300, 500, 3), dtype=np.uint8)
    assert sa._tile_on_device(exact, exact, 256, 416, 0.03, "cuda", False)
    assert not sa._tile_on_device(exact, other, 256, 416, 0.03, "cuda", False)
    assert not sa._tile_on_device(other, exact, 256, 416, 0.03, "cuda", False)


@pytest.mark.parametrize(
    ("in_hw", "size", "antialias"),
    [
        ((288, 468), (256, 416), False),
        ((234, 520), (234, 416), False),
        ((117, 208), (234, 416), True),
        ((300, 500), (256, 416), False),
        ((720, 1280), (234, 416), True),
        ((120, 208), (234, 416), True),
    ],
)
def test_device_resize_is_tile_resize_exactly_when_exact(
    monkeypatch, in_hw, size, antialias
):
    """On the device, _resize takes tile_resize.resize for exact resizes only."""
    calls = spy_resize(monkeypatch)
    x = _frames((2, *in_hw))
    want = _interpolate(x, size, antialias)
    got = sa._resize(x.as_subclass(OnDevice), size, antialias)
    exact = tile_resize.exact(in_hw, size, antialias)
    assert calls == ([(in_hw, size, antialias)] if exact else [])
    assert got.stride() == want.stride()
    assert torch.equal(got.as_subclass(torch.Tensor), want)
    # Never on the CPU, and never with the off switch.
    calls.clear()
    assert torch.equal(sa._resize(x, size, antialias), want)
    monkeypatch.setattr(sa, "TILE_ON_DEVICE_RESIZE", False)
    sa._resize(x.as_subclass(OnDevice), size, antialias)
    assert calls == []


@pytest.mark.parametrize(
    ("shape", "tile", "antialias", "device_resize"),
    [
        ((288, 512), (256, 416), False, True),
        ((117, 240), (224, 416), True, True),  # the fit rule's crop and 2x
        ((234, 416), (224, 416), True, False),  # crop only
        ((300, 500), (256, 416), False, False),
        ((288, 512), (224, 416), True, False),
    ],
)
def test_device_canvas_resizes_with_tile_resize(
    monkeypatch, shape, tile, antialias, device_resize
):
    """The device branch of every canvas resize: the CPU canvas, bit for bit.

    tile_resize.resize runs for each view exactly when _tile_on_device takes
    the shape for its resize.
    """
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_architecture_query)
    monkeypatch.setattr(sa, "TILE_ON_DEVICE", True)
    monkeypatch.setattr(sa, "THOR_TILE_RESIZE", False)
    rng = np.random.default_rng(4)
    ext = rng.integers(0, 256, (2, *shape, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (2, *shape, 3), dtype=np.uint8)
    want = sa.tile_clip(ext, wrist, *tile, 0.03, "cpu", antialias)
    on = sa._tile_on_device(ext, wrist, *tile, 0.03, "cuda", antialias)
    calls = spy_resize(monkeypatch)
    resize = sa._resize

    def on_device(x, size, antialias):
        return resize(x.as_subclass(OnDevice), size, antialias).as_subclass(
            torch.Tensor
        )

    monkeypatch.setattr(sa, "_resize", on_device)
    got = sa.tile_clip(ext, wrist, *tile, 0.03, "cpu", antialias)
    assert got.stride() == want.stride()
    assert torch.equal(got, want)
    assert len(calls) == (2 if device_resize else 0)
    if device_resize:
        assert on


def test_cpu_tile_clip_does_not_use_the_exact_resize(monkeypatch):
    """The CPU canvas is unchanged: interpolate, never tile_resize."""

    def fail(*args):
        pytest.fail("tile_resize.resize on the CPU path")

    monkeypatch.setattr(tile_resize, "resize", fail)
    rng = np.random.default_rng(2)
    ext = rng.integers(0, 256, (2, 288, 512, 3), dtype=np.uint8)
    clip = sa.tile_clip(ext, ext, 256, 416, 0.03)
    assert clip.shape == (1, 3, 2, 256, 832)


@cuda
@pytest.mark.parametrize(
    ("in_hw", "size", "antialias"),
    [
        ((288, 468), (256, 416), False),
        ((512, 832), (256, 416), False),
        ((234, 520), (234, 416), False),
        ((117, 208), (234, 416), True),
    ],
)
def test_cuda_resize_is_the_cpu_resize(in_hw, size, antialias):
    x = _frames((5, *in_hw), seed=3)
    want = _interpolate(x, size, antialias)
    got = tile_resize.resize(x.cuda(), size, antialias)
    assert got.stride() == want.stride()
    assert torch.equal(got.cpu(), want)


@cuda
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("frames", [1, 24, 25, 33])
@pytest.mark.parametrize(
    ("shape", "tile", "antialias", "device_resize"),
    [
        ((288, 512), (256, 416), False, True),  # exact resize on CUDA
        ((117, 240), (224, 416), True, True),  # exact fit-rule resize on CUDA
        ((300, 500), (256, 416), False, False),  # CPU resize
        ((234, 416), (224, 416), True, False),  # crop only
        ((288, 512), (224, 416), True, False),  # CPU resize
    ],
)
def test_cuda_tile_clip_is_the_cpu_clip(
    monkeypatch, pinned, frames, shape, tile, antialias, device_resize
):
    """The CUDA canvas has the bits, dtype and strides of the CPU canvas.

    tile_resize.resize runs exactly for the exact resizes with
    TILE_ON_DEVICE_RESIZE on.
    """
    monkeypatch.setattr(sa, "TILE_U8_PINNED", pinned)
    calls = spy_resize(monkeypatch)
    rng = np.random.default_rng(frames)
    ext = rng.integers(0, 256, (frames, *shape, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (frames, *shape, 3), dtype=np.uint8)
    want = sa.tile_clip(ext, wrist, *tile, 0.03, "cpu", antialias)
    for on in (True, False):
        monkeypatch.setattr(sa, "TILE_ON_DEVICE_RESIZE", on)
        calls.clear()
        got = sa.tile_clip(ext, wrist, *tile, 0.03, "cuda", antialias)
        assert bool(calls) is (on and device_resize)
        assert got.is_cuda and got.dtype == want.dtype
        assert got.stride() == want.stride()
        assert torch.equal(got.cpu(), want)
