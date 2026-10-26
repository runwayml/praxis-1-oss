"""MP4 output of a sim run and the frame layout of the human view.

``VideoWriter`` sends each frame to an ffmpeg process while the run continues,
so the video never stays in memory. The file is H.264 in yuv420p, which all
browsers can play. It needs the ``imageio-ffmpeg`` package (the ``video`` or
the ``sim`` extra).

``compose_panels`` puts camera views side by side, for example
``[overview | exterior | wrist]``. ``overlay_text`` writes lines of text,
such as the time, the call index, the prompt and the success flag, on a dark
band of the frame.
"""

from __future__ import annotations

import functools
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXT_COLOR = (255, 255, 255)
BAND_ALPHA = 0.55


def _check_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"frame must be uint8 HxWx3, got {arr.dtype} {tuple(arr.shape)}"
        )
    return arr


def even_size(frame: np.ndarray) -> np.ndarray:
    """Return the frame with an even height and width.

    yuv420p needs even sizes. An odd side gets one more row or column, a copy
    of the last one.
    """
    h, w = frame.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if not pad_h and not pad_w:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


class VideoWriter:
    """Write uint8 RGB frames to an H.264 MP4 file, one frame per call.

    The first frame sets the size. ffmpeg writes to ``<stem>.partial.mp4``.
    ``close`` renames the file to ``path``, so the final name only holds
    complete files. ``close`` also runs after an exception in a ``with``
    block, so the video of a failed run is available.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 15.0,
        crf: int = 20,
        preset: str = "fast",
        faststart: bool = True,
    ) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError(f"the video path must end with .mp4, got {self.path}")
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = preset
        self.faststart = faststart
        self.frames = 0
        self.size: tuple[int, int] | None = None  # (height, width) of the file
        self._gen: Generator[None, bytes | None, None] | None = None
        self._closed = False
        self._failed = False
        self._result: Path | None = None

    @property
    def partial_path(self) -> Path:
        """Return the path that ffmpeg writes to before ``close``."""
        return self.path.with_name(self.path.stem + ".partial.mp4")

    def _open(self, height: int, width: int) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise ImportError(
                "VideoWriter needs imageio-ffmpeg: pip install 'praxis[sim]'"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        params = ["-preset", self.preset, "-crf", str(self.crf)]
        if self.faststart:
            # The index goes to the start of the file, so a browser can play
            # the video before the download ends.
            params += ["-movflags", "+faststart"]
        self._gen = imageio_ffmpeg.write_frames(
            str(self.partial_path),
            (width, height),
            fps=self.fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=params,
        )
        self._gen.send(None)
        self.size = (height, width)

    def write(self, frame: np.ndarray) -> None:
        """Send one frame (uint8, HxWx3) to the encoder."""
        if self._closed:
            raise RuntimeError(f"the writer of {self.path} is closed")
        arr = even_size(_check_frame(frame))
        if self._gen is None:
            self._open(arr.shape[0], arr.shape[1])
        if arr.shape[:2] != self.size:
            raise ValueError(
                f"frame size {arr.shape[:2]} differs from the video size {self.size}"
            )
        assert self._gen is not None
        try:
            self._gen.send(np.ascontiguousarray(arr).tobytes())
        except BaseException:
            self._failed = True
            raise
        self.frames += 1

    def close(self) -> Path | None:
        """Stop ffmpeg and move the file to ``path``.

        Return the path. Return None when the writer got no frames or when
        ffmpeg failed. After a failure, the partial file stays for inspection.
        """
        if self._closed:
            return self._result
        self._closed = True
        if self._gen is None:
            return None
        self._gen.close()
        self._gen = None
        if self._failed or not self.partial_path.is_file():
            return None
        self.partial_path.replace(self.path)
        self._result = self.path
        return self._result

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_mp4(
    path: str | Path, frames: Iterable[np.ndarray], fps: float = 15.0
) -> Path:
    """Write all frames to one MP4 file and return its path."""
    with VideoWriter(path, fps=fps) as writer:
        for frame in frames:
            writer.write(frame)
    if writer.frames == 0:
        raise ValueError("write_mp4 got no frames")
    return writer.path


@functools.lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has one fixed-size default font
        return ImageFont.load_default()


def _fit(
    text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int
) -> str:
    """Cut the text at the end until it fits in ``width`` pixels."""
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + "...") > width:
        text = text[:-1]
    return text + "..."


def overlay_text(
    frame: np.ndarray,
    lines: Sequence[str],
    font_size: int | None = None,
    position: str = "bottom",
) -> np.ndarray:
    """Return a copy of the frame with the lines on a dark band.

    ``position`` is ``"top"`` or ``"bottom"``. The default font size is 1/24 of
    the frame height (at least 10 px). Lines that are too long end in "...".
    """
    if position not in ("top", "bottom"):
        raise ValueError(f"position must be 'top' or 'bottom', got {position!r}")
    arr = _check_frame(frame)
    if not lines:
        return arr.copy()
    h, w = arr.shape[:2]
    size = int(font_size) if font_size else max(10, round(h / 24))
    font = _font(size)
    pad = max(2, size // 4)
    line_h = size + pad
    band_h = min(h, len(lines) * line_h + pad)
    y0 = 0 if position == "top" else h - band_h
    out = arr.copy()
    band = out[y0 : y0 + band_h].astype(np.float32) * (1.0 - BAND_ALPHA)
    img = Image.fromarray(band.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text(
            (pad, pad + i * line_h),
            _fit(line, font, w - 2 * pad),
            fill=TEXT_COLOR,
            font=font,
        )
    out[y0 : y0 + band_h] = np.asarray(img)
    return out


def status_lines(
    time_s: float,
    call_index: int | None = None,
    prompt: str | None = None,
    success: bool | None = None,
    extra: dict[str, object] | None = None,
) -> list[str]:
    """Return the overlay lines of one tick: the state line, then the prompt."""
    parts = [f"t {time_s:6.2f} s"]
    if call_index is not None:
        parts.append(f"call {call_index}")
    if success is not None:
        parts.append("success" if success else "no success")
    for key, value in (extra or {}).items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        parts.append(f"{key} {value}")
    lines = ["   ".join(parts)]
    if prompt:
        lines.append(prompt)
    return lines


def compose_panels(
    panels: Sequence[np.ndarray],
    height: int | None = None,
    labels: Sequence[str] | None = None,
    text: Sequence[str] | None = None,
    gap: int = 0,
) -> np.ndarray:
    """Put the panels side by side at one height and return the frame.

    Each panel is uint8 HxWx3. A panel of a different height gets a bilinear
    resize that keeps its aspect. ``height`` defaults to the height of the
    first panel. ``labels`` adds a small name at the top left of each panel.
    ``text`` adds lines on a dark band at the bottom (see ``status_lines``).
    ``gap`` adds dark columns between the panels.
    """
    if not panels:
        raise ValueError("compose_panels needs at least one panel")
    if labels is not None and len(labels) != len(panels):
        raise ValueError(f"got {len(labels)} labels for {len(panels)} panels")
    arrs = [_check_frame(p) for p in panels]
    target_h = int(height) if height else arrs[0].shape[0]
    cols: list[np.ndarray] = []
    for i, arr in enumerate(arrs):
        h, w = arr.shape[:2]
        if h != target_h:
            new_w = max(1, round(w * target_h / h))
            img = Image.fromarray(arr).resize(
                (new_w, target_h), Image.Resampling.BILINEAR
            )
            arr = np.asarray(img)
        if labels is not None and labels[i]:
            arr = overlay_text(
                arr, [labels[i]], font_size=max(10, target_h // 28), position="top"
            )
        if i and gap > 0:
            cols.append(np.full((target_h, gap, 3), 32, dtype=np.uint8))
        cols.append(arr)
    frame = np.ascontiguousarray(np.concatenate(cols, axis=1))
    if text:
        frame = overlay_text(frame, text)
    return frame
