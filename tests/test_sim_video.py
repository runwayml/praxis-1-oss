"""Unit tests for the sim MP4 writer and the panel layout (CPU only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from praxis.sim.video import (
    VideoWriter,
    compose_panels,
    even_size,
    overlay_text,
    status_lines,
    write_mp4,
)


def _frames(n: int, h: int, w: int) -> list[np.ndarray]:
    out = []
    for k in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : (k * 7) % w + 1] = (255, 120, 0)
        frame[..., 2] = 30 * k % 256
        out.append(frame)
    return out


def _read_back(path: Path) -> tuple[dict, list[np.ndarray]]:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    w, h = meta["size"]
    frames = [np.frombuffer(b, dtype=np.uint8).reshape(h, w, 3) for b in gen]
    return meta, frames


def test_write_and_read_back(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "out" / "run.mp4"
    src = _frames(20, 72, 128)
    with VideoWriter(path, fps=15) as writer:
        for frame in src:
            writer.write(frame)
        assert writer.partial_path.is_file() and not path.exists()
    assert path.is_file() and not writer.partial_path.exists()
    assert writer.frames == 20 and writer.size == (72, 128)

    meta, frames = _read_back(path)
    assert meta["codec"] == "h264"
    assert meta["pix_fmt"].startswith("yuv420p")
    assert meta["size"] == (128, 72)
    assert meta["fps"] == 15.0
    assert len(frames) == 20
    # Lossy, but close to the source.
    err = np.abs(frames[10].astype(int) - src[10].astype(int)).mean()
    assert err < 6.0


def test_odd_size_gets_padded(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    path = write_mp4(tmp_path / "odd.mp4", _frames(5, 71, 125), fps=10)
    meta, frames = _read_back(path)
    assert meta["size"] == (126, 72)
    assert len(frames) == 5
    assert even_size(np.zeros((71, 125, 3), np.uint8)).shape == (72, 126, 3)


def test_writer_checks(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "run.avi")
    writer = VideoWriter(tmp_path / "empty.mp4")
    assert writer.close() is None
    assert not (tmp_path / "empty.mp4").exists()
    with pytest.raises(RuntimeError):
        writer.write(_frames(1, 8, 8)[0])

    pytest.importorskip("imageio_ffmpeg")
    with VideoWriter(tmp_path / "size.mp4") as writer:
        writer.write(_frames(1, 16, 16)[0])
        with pytest.raises(ValueError):
            writer.write(_frames(1, 16, 32)[0])
        with pytest.raises(ValueError):
            writer.write(np.zeros((16, 16, 3), dtype=np.float32))
    assert (tmp_path / "size.mp4").is_file()


def test_compose_panels() -> None:
    overview = np.full((360, 640, 3), 50, dtype=np.uint8)
    exterior = np.full((288, 512, 3), 100, dtype=np.uint8)
    wrist = np.full((288, 512, 3), 150, dtype=np.uint8)
    frame = compose_panels([overview, exterior, wrist], height=288)
    assert frame.dtype == np.uint8
    assert frame.shape == (288, 512 + 512 + 512, 3)
    assert frame[200, 100, 0] == 50 and frame[200, 700, 0] == 100
    assert frame[200, 1300, 0] == 150

    framed = compose_panels([exterior, wrist], gap=4, labels=["exterior", "wrist"])
    assert framed.shape == (288, 1028, 3)
    assert framed[200, 513, 0] == 32
    # The label band darkens the top left corner.
    assert framed[2, 2, 0] < 100

    lines = status_lines(1.5, call_index=2, prompt="lift the red cube", success=True)
    assert lines == ["t   1.50 s   call 2   success", "lift the red cube"]
    with_text = compose_panels([exterior, wrist], text=lines)
    assert with_text.shape == (288, 1024, 3)
    assert with_text[-3, 1000, 0] < 100
    assert (with_text[-30:] != 100).any()

    with pytest.raises(ValueError):
        compose_panels([])
    with pytest.raises(ValueError):
        compose_panels([exterior], labels=["a", "b"])


def test_overlay_text_leaves_the_input() -> None:
    frame = np.full((100, 200, 3), 200, dtype=np.uint8)
    out = overlay_text(frame, ["a very long line " * 20], position="top")
    assert (frame == 200).all()
    assert out.shape == frame.shape
    assert out[1, 1, 0] < 200 and out[-1, -1, 0] == 200
    with pytest.raises(ValueError):
        overlay_text(frame, ["x"], position="left")
