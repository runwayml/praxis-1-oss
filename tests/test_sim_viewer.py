"""Unit tests for the live sim viewer (CPU only, loopback address only)."""

from __future__ import annotations

import http.client
import io
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("praxis_sim")  # the `sim` extra

from praxis.sim.viewer import BOUNDARY, Viewer, parse_range


def _frame(k: int, h: int = 72, w: int = 128) -> np.ndarray:
    """Return a frame with a solid colour per k, so JPEG colours stay clear."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = (40 * k) % 256
    frame[..., 1] = 200
    frame[..., 2] = 255 - (40 * k) % 256
    return frame


def _get(
    port: int, path: str, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _read_part(resp: http.client.HTTPResponse) -> bytes:
    """Read one MJPEG part: boundary line, part headers, JPEG body."""
    line = resp.fp.readline()
    while line in (b"\r\n", b"\n"):
        line = resp.fp.readline()
    assert line.strip() == f"--{BOUNDARY}".encode(), line
    headers = {}
    while True:
        line = resp.fp.readline().strip()
        if not line:
            break
        key, _, value = line.decode().partition(":")
        headers[key.strip().lower()] = value.strip()
    assert headers["content-type"] == "image/jpeg"
    return resp.fp.read(int(headers["content-length"]))


@pytest.fixture
def viewer():
    v = Viewer(host="127.0.0.1", port=0, max_fps=50.0)
    v.start()
    yield v
    v.stop()


def test_page_and_status(viewer: Viewer) -> None:
    assert viewer.url == f"http://127.0.0.1:{viewer.port}/"
    status, headers, body = _get(viewer.port, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"/stream.mjpg" in body and b"/status" in body and b"<video" in body

    status, _, body = _get(viewer.port, "/status")
    assert status == 200
    assert json.loads(body) == {"status": {}, "frames": 0, "video": None}

    viewer.publish(_frame(1), {"tick": 3, "dist": np.float32(0.25), "ok": False})
    data = json.loads(_get(viewer.port, "/status")[2])
    assert data["frames"] == 1
    assert data["status"] == {"tick": 3, "dist": 0.25, "ok": False}

    viewer.set_status({"tick": 4, "nan": float("nan"), "xyz": np.zeros(2)})
    data = json.loads(_get(viewer.port, "/status")[2])
    assert data["status"] == {"tick": 4, "nan": "nan", "xyz": [0.0, 0.0]}

    viewer.set_status({"tick": 4})
    data = json.loads(_get(viewer.port, "/status")[2])
    assert data["status"] == {"tick": 4} and data["frames"] == 1

    assert _get(viewer.port, "/video")[0] == 404
    assert _get(viewer.port, "/nothing")[0] == 404


def test_publish_rejects_bad_frames(viewer: Viewer) -> None:
    with pytest.raises(ValueError):
        viewer.publish(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        viewer.publish(np.zeros((4, 4), dtype=np.uint8))


def test_publish_copies_the_frame(viewer: Viewer) -> None:
    frame = _frame(0)
    viewer.publish(frame)
    frame[:] = 0
    status, _, body = _get(viewer.port, "/frame.jpg")
    assert status == 200
    img = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
    assert abs(int(img[..., 1].mean()) - 200) < 8


def test_mjpeg_stream_gives_new_frames(viewer: Viewer) -> None:
    viewer.publish(_frame(0))
    conn = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=5)
    conn.request("GET", "/stream.mjpg")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == (
        f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    )

    stop = threading.Event()

    def producer() -> None:
        k = 1
        while not stop.is_set():
            viewer.publish(_frame(k), {"tick": k})
            k += 1
            time.sleep(0.02)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    try:
        images = []
        for _ in range(2):
            img = Image.open(io.BytesIO(_read_part(resp)))
            img.load()
            assert img.format == "JPEG" and img.size == (128, 72)
            images.append(np.asarray(img.convert("RGB")))
    finally:
        stop.set()
        thread.join()
        conn.close()
    # The two parts hold two different frames.
    assert abs(int(images[0][..., 0].mean()) - int(images[1][..., 0].mean())) > 10


def test_mjpeg_frame_rate_cap() -> None:
    with Viewer(host="127.0.0.1", port=0, max_fps=10.0) as v:
        v.publish(_frame(0))
        sock = socket.create_connection(("127.0.0.1", v.port), timeout=5)
        sock.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
        data = b""
        t_end = time.monotonic() + 1.0
        k = 1
        while time.monotonic() < t_end:
            v.publish(_frame(k))
            k += 1
            time.sleep(0.005)
        sock.settimeout(0.2)
        try:
            while chunk := sock.recv(65536):
                data += chunk
        except TimeoutError:
            pass
        sock.close()
    parts = data.count(b"Content-Type: image/jpeg")
    # About 100 frames in 1 s, at most 10 per second plus the first one.
    assert k > 50
    assert 5 <= parts <= 12


def test_video_range_requests(viewer: Viewer, tmp_path: Path) -> None:
    payload = np.random.default_rng(0).integers(0, 256, 5000, dtype=np.uint8)
    path = tmp_path / "run.mp4"
    path.write_bytes(payload.tobytes())
    viewer.set_video(path)

    data = json.loads(_get(viewer.port, "/status")[2])
    assert data["video"].startswith("/video")

    status, headers, body = _get(viewer.port, "/video")
    assert status == 200
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "video/mp4"
    assert body == payload.tobytes()

    status, headers, body = _get(viewer.port, "/video", {"Range": "bytes=100-199"})
    assert status == 206
    assert headers["Content-Range"] == "bytes 100-199/5000"
    assert body == payload[100:200].tobytes()

    status, headers, body = _get(viewer.port, "/video", {"Range": "bytes=4990-"})
    assert status == 206 and body == payload[4990:].tobytes()

    status, headers, body = _get(viewer.port, "/video", {"Range": "bytes=-10"})
    assert status == 206 and body == payload[-10:].tobytes()

    status, headers, _ = _get(viewer.port, "/video", {"Range": "bytes=9000-"})
    assert status == 416
    assert headers["Content-Range"] == "bytes */5000"

    viewer.set_video(None)
    assert json.loads(_get(viewer.port, "/status")[2])["video"] is None
    with pytest.raises(FileNotFoundError):
        viewer.set_video(tmp_path / "missing.mp4")


def test_parse_range() -> None:
    assert parse_range(None, 10) is None
    assert parse_range("bytes=2-5", 10) == (2, 5)
    assert parse_range("bytes=2-50", 10) == (2, 9)
    assert parse_range("bytes=7-", 10) == (7, 9)
    assert parse_range("bytes=-3", 10) == (7, 9)
    assert parse_range("bytes=-30", 10) == (0, 9)
    assert parse_range("bytes=0-1,4-5", 10) is None
    assert parse_range("items=0-1", 10) is None
    for bad in ("bytes=10-", "bytes=5-2", "bytes=-0"):
        with pytest.raises(ValueError):
            parse_range(bad, 10)


def test_client_disconnect_does_not_stop_the_server(viewer: Viewer) -> None:
    viewer.publish(_frame(0))
    for _ in range(3):
        sock = socket.create_connection(("127.0.0.1", viewer.port), timeout=5)
        sock.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
        sock.recv(64)
        sock.close()
        viewer.publish(_frame(1))
    # A client that stops reading must not block publish.
    slow = socket.create_connection(("127.0.0.1", viewer.port), timeout=5)
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    slow.sendall(b"GET /stream.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
    t0 = time.monotonic()
    noise = np.random.default_rng(1).integers(0, 256, (288, 512, 3), dtype=np.uint8)
    for _ in range(200):
        viewer.publish(noise)
    assert time.monotonic() - t0 < 1.0
    status, _, body = _get(viewer.port, "/status")
    assert status == 200 and json.loads(body)["frames"] == 204
    slow.close()


def test_stop_closes_streams_and_port() -> None:
    v = Viewer(host="127.0.0.1", port=0)
    v.start()
    port = v.port
    v.publish(_frame(0))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/stream.mjpg")
    resp = conn.getresponse()
    _read_part(resp)
    t0 = time.monotonic()
    v.stop()
    assert time.monotonic() - t0 < 2.0
    conn.close()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    with pytest.raises(RuntimeError):
        v.start()


def test_context_manager() -> None:
    with Viewer(host="127.0.0.1", port=0) as v:
        assert _get(v.port, "/status")[0] == 200
        port = v.port
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
