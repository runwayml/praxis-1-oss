"""Wire codec of the Positronic policy protocol: msgpack with numpy values and images.

A message is one msgpack map. The codec packs numpy values into marker maps
with ``bytes`` keys:

* ``{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": [...]}``
* ``{b"__npgeneric__": True, b"data": value, b"dtype": str}``

Object, void and complex dtypes are refused in both directions. An object
dtype would read client bytes as object pointers.

A client can also send an image stack ``(T, H, W, 3)`` or one image
``(H, W, 3)`` in a compressed form:

* ``{b"__jpeg__": True, b"frames": [bytes, ...], b"ndim": 3 | 4}``: one JPEG
  file per frame.
* ``{b"__h264__": True, b"data": bytes, b"shape": [...], b"ndim": 3 | 4}``: one
  H.264 group of pictures (an Annex B stream). The encoder pads odd sizes to
  even sizes, so the decoder crops each frame back to ``shape``.

:func:`deserialise_request` also measures the image decode of one request for
the ``server_timing`` block of the answer.
"""

from __future__ import annotations

import collections.abc as cabc
import contextlib
import functools
import io
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np
from PIL import Image

IMAGE_CODECS: tuple[str, ...] = ("h264", "jpeg")
MAX_IMAGE_FRAMES = 25
# Match the WebSocket message limit; apply across all cameras in a request.
MAX_DECODED_BYTES = 256 * 1024 * 1024

# The standard luminance quantization table of the IJG JPEG library (JPEG
# Annex K.1, natural order).
_JPEG_STD_LUMA = np.array(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)  # fmt: skip


@dataclass
class DecodeStats:
    """Image decode statistics of one request."""

    jpeg_ms: float = 0.0
    video_ms: float = 0.0
    codecs: set[str] = field(default_factory=set)
    jpeg_quality: int | None = None
    _decoded_bytes: int = field(default=0, repr=False)


# The statistics of the request that this thread decodes now, or None.
_ACTIVE = threading.local()


def _stats() -> DecodeStats | None:
    return getattr(_ACTIVE, "stats", None)


def jpeg_quality_estimate(luma_table: cabc.Iterable[int] | None) -> int | None:
    """Return the IJG quality that made a luminance quantization table, or None.

    The IJG library scales the standard table by ``(200 - 2q) %`` for
    ``q >= 50`` and by ``5000 / q %`` below. The median ratio of the sorted
    tables inverts this scale. The element order of the table has no effect.
    """
    try:
        t = np.sort(np.asarray(list(luma_table or ()), dtype=np.float64))
    except (TypeError, ValueError):
        return None
    if t.size != 64 or not np.all(t > 0):
        return None
    scale = float(np.median(100.0 * t / np.sort(_JPEG_STD_LUMA)))
    q = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return round(min(100.0, max(1.0, q)))


def pack_numpy(obj: object) -> object:
    """Return the wire form of a value that msgpack cannot pack (``default`` hook)."""
    if isinstance(obj, cabc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _safe_dtype(dtype: np.dtype) -> np.dtype:
    """Refuse the dtypes that :func:`pack_numpy` never sends."""
    if dtype.hasobject or dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype


def _charge(size: int) -> None:
    stats = _stats()
    total = size + (stats._decoded_bytes if stats is not None else 0)
    if total > MAX_DECODED_BYTES:
        raise ValueError("Decoded arrays exceed the request byte limit")
    if stats is not None:
        stats._decoded_bytes = total


def _image_ndim(obj: dict) -> int:
    ndim = obj[b"ndim"]
    if type(ndim) is not int or ndim not in (3, 4):
        raise ValueError("Image ndim must be 3 or 4")
    return ndim


def _decode_jpeg(obj: dict) -> np.ndarray:
    t0 = time.monotonic()
    stats = _stats()
    ndim = _image_ndim(obj)
    buffers = obj[b"frames"]
    if not isinstance(buffers, list) or not 1 <= len(buffers) <= MAX_IMAGE_FRAMES:
        raise ValueError("JPEG stack must contain 1 to 25 frames")
    if ndim == 3 and len(buffers) != 1:
        raise ValueError("A 3D image must contain one frame")
    frames = []
    for buf in buffers:
        im = Image.open(io.BytesIO(buf), formats=["JPEG"])
        _charge(im.width * im.height * 3)
        if stats is not None and stats.jpeg_quality is None:
            with contextlib.suppress(Exception):
                tables = getattr(im, "quantization", None) or {}
                stats.jpeg_quality = jpeg_quality_estimate(tables.get(0))
        frames.append(np.asarray(im.convert("RGB")))
    stacked = np.stack(frames)
    if stats is not None:
        stats.jpeg_ms += (time.monotonic() - t0) * 1e3
        stats.codecs.add("jpeg")
    return stacked if ndim == 4 else stacked[0]


def _decode_h264(obj: dict) -> np.ndarray:
    import av  # only H.264 requests need PyAV

    t0 = time.monotonic()
    ndim = _image_ndim(obj)
    shape = obj[b"shape"]
    if (
        len(shape) != ndim
        or any(type(x) is not int or x <= 0 for x in shape)
        or shape[-1] != 3
    ):
        raise ValueError("H.264 shape must describe positive RGB dimensions")
    t, h, w = (shape[0], shape[1], shape[2]) if ndim == 4 else (1, shape[0], shape[1])
    if not 1 <= t <= MAX_IMAGE_FRAMES:
        raise ValueError("H.264 stack must contain 1 to 25 frames")
    _charge(t * h * w * 3)
    frames = []
    with av.open(io.BytesIO(obj[b"data"]), mode="r", format="h264") as container:
        for frame in container.decode(video=0):
            if len(frames) == t:
                raise ValueError(f"H.264 stack exceeds expected {t} frames")
            if (frame.height, frame.width) != (h + h % 2, w + w % 2):
                raise ValueError(
                    "H.264 frame dimensions do not match the declared shape"
                )
            frames.append(frame.to_ndarray(format="rgb24")[:h, :w])
    if len(frames) != t:
        raise ValueError(f"H.264 stack decoded to {len(frames)} frames, expected {t}")
    stacked = np.stack(frames)
    stats = _stats()
    if stats is not None:
        stats.video_ms += (time.monotonic() - t0) * 1e3
        stats.codecs.add("h264")
    return stacked if ndim == 4 else stacked[0]


def unpack_numpy(obj: dict) -> object:
    """Return the value of a marker map (``object_hook``). Other maps pass unchanged.

    An array is a read-only view of the message bytes. A shape that does not
    match the data raises ``ValueError``.
    """
    if b"__ndarray__" in obj:
        _charge(len(obj[b"data"]))
        dtype = _safe_dtype(np.dtype(obj[b"dtype"]))
        return np.frombuffer(obj[b"data"], dtype=dtype).reshape(obj[b"shape"])
    if b"__npgeneric__" in obj:
        dtype = _safe_dtype(np.dtype(obj[b"dtype"]))
        return dtype.type(obj[b"data"])
    if b"__jpeg__" in obj:
        return _decode_jpeg(obj)
    if b"__h264__" in obj:
        return _decode_h264(obj)
    if b"__cmd__" in obj:
        # A robot command of the client (the plan of a real-time call): the
        # command dict that the server sends.
        return obj[b"__cmd__"]
    return obj


serialise = functools.partial(msgpack.packb, default=pack_numpy)


def _unpack_pairs(pairs: list[tuple]) -> object:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate msgpack key: {key!r}")
        obj[key] = value
    markers = (b"__ndarray__", b"__npgeneric__", b"__jpeg__", b"__h264__")
    if sum(key in obj for key in markers) > 1:
        raise ValueError("Conflicting wire markers")
    return unpack_numpy(obj)


def deserialise(message: bytes) -> Any:
    return deserialise_request(message)[0]


def deserialise_request(message: bytes) -> tuple[Any, DecodeStats]:
    """Decode one request. Return the value and the statistics of its image decode."""
    stats = DecodeStats()
    _ACTIVE.stats = stats
    try:
        return msgpack.unpackb(message, object_pairs_hook=_unpack_pairs), stats
    finally:
        _ACTIVE.stats = None


def compress_images() -> bool:
    """Return the ``compress_images`` value of the handshake.

    True asks the client to send JPEG images. Set ``PRAXIS_COMPRESS_IMAGES=0``
    to receive raw arrays.
    """
    return os.environ.get("PRAXIS_COMPRESS_IMAGES", "1") != "0"


def image_codecs() -> list[str]:
    """Return the ``image_codecs`` value of the handshake (preferred codec first).

    ``PRAXIS_IMAGE_CODECS`` sets it, for example ``jpeg``. The default is
    ``h264,jpeg``. An unknown codec raises ``ValueError``.
    """
    raw = os.environ.get("PRAXIS_IMAGE_CODECS", "h264,jpeg")
    codecs = [c.strip().lower() for c in raw.split(",") if c.strip()]
    unknown = [c for c in codecs if c not in IMAGE_CODECS]
    if unknown or not codecs:
        raise ValueError(
            f"PRAXIS_IMAGE_CODECS={raw!r}: expected a comma list of {IMAGE_CODECS}"
        )
    return codecs
