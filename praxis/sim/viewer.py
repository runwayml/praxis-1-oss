"""Live view of a sim run in a web browser. Standard library and Pillow only.

The viewer is an HTTP server in a daemon thread. It holds the latest frame and
the latest status dict of the run. These are the routes:

* ``/``: an HTML page with the live stream, a status line and, when the run
  has a finished MP4, a video player.
* ``/stream.mjpg``: the latest frame as an MJPEG stream
  (``multipart/x-mixed-replace``).
* ``/frame.jpg``: the latest frame as one JPEG.
* ``/status``: the status as JSON.
* ``/video``: the MP4 of the run. The route accepts HTTP Range requests, so
  the browser can seek.

``publish`` only stores a copy of the frame. The stream threads encode the
JPEG, so the sim loop does not wait for the network. A slow client gets
fewer frames, because each client always gets the newest frame.

The default address is 127.0.0.1. On a cloud VM, forward the port with SSH::

    ssh -L 8000:127.0.0.1:8000 user@vm

Then open http://127.0.0.1:8000 on the local machine.

A host of 0.0.0.0 makes the stream available to all machines that can reach
the port. The stream has no password. Use it only on a private network.
"""

from __future__ import annotations

import html
import io
import json
import logging
import math
import re
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BOUNDARY = "frame"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
_CHUNK = 256 * 1024
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body { background: #111; color: #ddd; font-family: sans-serif; margin: 16px; }
img, video { display: block; max-width: 100%; border: 1px solid #333; }
#status { font-family: monospace; margin: 8px 0 16px; white-space: pre-wrap; }
h2 { font-size: 1em; font-weight: normal; color: #999; margin: 16px 0 4px; }
</style></head>
<body>
<h2>live</h2>
<img id="live" src="/stream.mjpg" alt="live view">
<div id="status">waiting for the first frame</div>
<h2 id="video-title" hidden>video of the run</h2>
<video id="video" controls loop muted playsinline hidden></video>
<script>
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const videoTitle = document.getElementById("video-title");
let videoSrc = null;
function fmt(v) {
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(3);
  if (typeof v === "object" && v !== null) return JSON.stringify(v);
  return String(v);
}
async function poll() {
  try {
    const r = await fetch("/status", {cache: "no-store"});
    const s = await r.json();
    const parts = Object.entries(s.status).map(([k, v]) => k + ": " + fmt(v));
    if (parts.length) statusEl.textContent = parts.join("   ");
    if (s.video !== videoSrc) {
      videoSrc = s.video;
      videoEl.hidden = videoTitle.hidden = !videoSrc;
      if (videoSrc) { videoEl.src = videoSrc; videoEl.play().catch(() => {}); }
      else { videoEl.removeAttribute("src"); videoEl.load(); }
    }
  } catch (e) {
    statusEl.textContent = "the viewer does not answer";
  }
  setTimeout(poll, 1000);
}
poll();
</script>
</body></html>
"""


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode one uint8 RGB frame as JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def jsonable(value: object) -> object:
    """Return a copy of ``value`` that ``json.dumps`` accepts with strict JSON.

    numpy values become Python values. NaN and inf become the strings "nan"
    and "inf", because a browser cannot parse them as JSON. Other types become
    their ``str``.
    """
    if isinstance(value, np.ndarray | np.generic):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return the inclusive byte span of a single Range header.

    Return None for no header or for a header that this server ignores (a
    multi-range or an unknown unit). Then the server sends the full file.
    Raise ValueError when the span is not satisfiable.
    """
    if not header:
        return None
    m = _RANGE_RE.match(header.strip())
    if m is None:
        return None
    first, last = m.groups()
    if not first and not last:
        return None
    if not first:
        # Suffix span: the last N bytes.
        n = int(last)
        if n == 0 or size == 0:
            raise ValueError(f"unsatisfiable range {header!r}")
        return max(0, size - n), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError(f"unsatisfiable range {header!r}")
    return start, end


class _Hub:
    """Shared state of the server threads and the sim thread."""

    def __init__(self, jpeg_quality: int) -> None:
        self.jpeg_quality = int(jpeg_quality)
        self._cond = threading.Condition()
        self._encode_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._status: object = {}
        self._video: Path | None = None
        self._video_version = 0
        self._closed = False
        self._conns: set[socket.socket] = set()

    @property
    def closed(self) -> bool:
        return self._closed

    def publish(self, frame: np.ndarray, status: object) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            if status is not None:
                self._status = status
            self._cond.notify_all()

    def set_status(self, status: object) -> None:
        with self._cond:
            self._status = status

    def set_video(self, path: Path | None) -> None:
        with self._cond:
            self._video = path
            self._video_version += 1

    def video(self) -> Path | None:
        with self._cond:
            return self._video

    def status_json(self) -> bytes:
        with self._cond:
            status = self._status
            seq = self._seq
            video = self._video
            version = self._video_version
        body = {
            "status": status,
            "frames": seq,
            "video": f"/video?v={version}" if video is not None else None,
        }
        return json.dumps(body, allow_nan=False).encode()

    def wait_frame(self, last_seq: int, timeout: float) -> int:
        """Wait for a frame newer than ``last_seq``. Return the newest sequence."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq or self._closed, timeout=timeout
            )
            return self._seq

    def sleep(self, seconds: float) -> bool:
        """Sleep for ``seconds`` or until close. Return True after close."""
        with self._cond:
            return self._cond.wait_for(lambda: self._closed, timeout=seconds)

    def latest_jpeg(self) -> tuple[int, bytes | None]:
        """Return the newest frame as JPEG. Encode each frame at most once."""
        with self._encode_lock:
            with self._cond:
                seq, frame = self._seq, self._frame
            if frame is None:
                return 0, None
            if self._jpeg_seq != seq:
                self._jpeg = encode_jpeg(frame, self.jpeg_quality)
                self._jpeg_seq = seq
            return seq, self._jpeg

    def add_conn(self, conn: socket.socket) -> None:
        with self._cond:
            if not self._closed:
                self._conns.add(conn)
                return
        # The viewer stops. Refuse new connections.
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def drop_conn(self, conn: socket.socket) -> None:
        with self._cond:
            self._conns.discard(conn)
            self._cond.notify_all()

    def close(self, timeout: float) -> None:
        """Wake all stream threads, shut down open sockets and wait for them."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        with self._cond:
            self._cond.wait_for(lambda: not self._conns, timeout=timeout)


class _Server(ThreadingHTTPServer):
    """HTTP server that logs client disconnects at debug level only."""

    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError | TimeoutError):
            logger.debug("viewer client %s went away: %s", client_address, exc)
            return
        logger.error("viewer request from %s failed", client_address, exc_info=True)


def _make_handler(
    hub: _Hub, page: bytes, max_fps: float
) -> type[BaseHTTPRequestHandler]:
    period = 1.0 / max_fps if max_fps > 0 else 0.0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # A client that stops reading frees its thread after this time.
        timeout = 30

        def setup(self) -> None:
            super().setup()
            hub.add_conn(self.connection)

        def finish(self) -> None:
            try:
                super().finish()
            finally:
                hub.drop_conn(self.connection)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("viewer %s: %s", self.address_string(), format % args)

        def do_HEAD(self) -> None:
            self.do_GET()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_bytes(page, "text/html; charset=utf-8")
            elif path == "/status":
                self._send_bytes(hub.status_json(), "application/json")
            elif path == "/frame.jpg":
                _, jpeg = hub.latest_jpeg()
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
                else:
                    self._send_bytes(jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self._stream()
            elif path == "/video":
                video = hub.video()
                if video is None or not video.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "no video yet")
                else:
                    self._send_file(video, "video/mp4")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_bytes(self, body: bytes, ctype: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_file(self, path: Path, ctype: str) -> None:
            size = path.stat().st_size
            try:
                span = parse_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if span is None:
                start, end = 0, size - 1
                self.send_response(HTTPStatus.OK)
            else:
                start, end = span
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command == "HEAD":
                return
            remaining = end - start + 1
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    while remaining > 0 and not hub.closed:
                        chunk = f.read(min(_CHUNK, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except OSError:
                pass
            if remaining > 0:
                # The response is short. The client must not reuse the socket.
                self.close_connection = True

        def _stream(self) -> None:
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command == "HEAD":
                return
            # Each part ends with the next boundary line. Some browsers show a
            # part only when the next boundary arrives.
            delim = f"\r\n--{BOUNDARY}\r\n".encode()
            seq = 0
            next_send = 0.0
            try:
                self.wfile.write(delim[2:])
                while not hub.closed:
                    if hub.wait_frame(seq, timeout=1.0) == seq:
                        continue
                    delay = next_send - time.monotonic()
                    if delay > 0 and hub.sleep(delay):
                        break
                    seq, jpeg = hub.latest_jpeg()
                    if jpeg is None:
                        continue
                    head = (
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode()
                    self.wfile.write(head + jpeg + delim)
                    next_send = time.monotonic() + period
            except OSError:
                # The client closed the connection or stopped reading.
                pass

    return Handler


class Viewer:
    """Live browser view of a sim run: MJPEG stream, status line and MP4 player.

    Use it as a context manager, or call ``start`` and ``stop``::

        with Viewer(port=8000) as viewer:
            print(viewer.url)
            viewer.publish(frame, {"tick": 0})
            viewer.set_video("run.mp4")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        jpeg_quality: int = 80,
        max_fps: float = 15.0,
        title: str = "praxis sim",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.max_fps = float(max_fps)
        self.title = title
        self._hub = _Hub(jpeg_quality)
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None

    @property
    def url(self) -> str | None:
        """Return the URL of the page, or None before ``start``."""
        return self._url

    def start(self) -> str:
        """Bind the port, serve in a daemon thread and return the page URL."""
        if self._server is not None:
            raise RuntimeError("the viewer is already started")
        if self._hub.closed:
            raise RuntimeError("a stopped viewer cannot start again")
        page = _PAGE.replace("__TITLE__", html.escape(self.title)).encode()
        handler = _make_handler(self._hub, page, self.max_fps)
        self._server = _Server((self.host, self.port), handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="praxis-sim-viewer",
            daemon=True,
        )
        self._thread.start()
        if self.host not in LOOPBACK_HOSTS:
            logger.warning(
                "The viewer listens on %s:%d. All machines that can reach this "
                "port can see the stream. Use 127.0.0.1 and SSH port forwarding "
                "when possible.",
                self.host,
                self.port,
            )
        url_host = "127.0.0.1" if self.host in ("", "0.0.0.0") else self.host
        self._url = f"http://{url_host}:{self.port}/"
        logger.info("viewer: %s", self._url)
        return self._url

    def publish(self, frame: np.ndarray, status: dict | None = None) -> None:
        """Store a copy of the frame (uint8, HxWx3) and replace the status.

        The call does not encode and does not wait for clients.
        """
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"frame must be uint8 HxWx3, got {arr.dtype} {tuple(arr.shape)}"
            )
        self._hub.publish(
            np.array(arr, copy=True, order="C"),
            jsonable(dict(status)) if status is not None else None,
        )

    def set_status(self, status: dict) -> None:
        """Replace the status without a new frame."""
        self._hub.set_status(jsonable(dict(status)))

    def set_video(self, path: str | Path | None) -> None:
        """Serve this MP4 on ``/video`` and show the player. None hides it."""
        if path is None:
            self._hub.set_video(None)
            return
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        self._hub.set_video(p)

    def stop(self, timeout: float = 2.0) -> None:
        """Close all streams, stop the server and release the port."""
        self._hub.close(timeout=timeout)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._server = None
        self._thread = None

    def __enter__(self) -> Viewer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
