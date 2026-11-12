#!/usr/bin/env python
"""Run an so101 praxis policy server on an SO-100 / SO-101 follower arm through LeRobot.

The remote client of a praxis serve endpoint on an so101 checkpoint
(``python -m praxis.serve --config configs/serve_so101.yaml`` on a local
GPU, for example a 5090 with ``routed_precision: nvfp4``, or the Modal app).
It is one file without a praxis import: next to the robot it needs
``lerobot`` (with the Feetech extra), ``numpy``, ``msgpack``, ``websockets``
and ``Pillow``.

    python eval/so101/so101_client.py \\
        --url ws://<host>:8000/api/v1/session \\
        --robot-port /dev/ttyACM0 --robot-id my_so101 \\
        --exterior-cam 0 [--wrist-cam 2] \\
        --grip-closed 1.4 --grip-open 42.1 \\
        --prompt "Pick up the red cube and put it in the box." \\
        [--dry-run] [--control-hz 30] [--max-step-deg 4]

A Modal endpoint takes its proxy auth headers from ``--headers-file`` (a JSON
file with ``Modal-Key`` and ``Modal-Secret``).

Observation contract (what the client sends): the frame stack that the
server declares in its handshake, one entry per executed 15 Hz row, oldest
first, the current entry last: ``state`` ``(T, 6)`` = the follower's
``shoulder_pan.pos .. wrist_roll.pos`` in degrees (``use_degrees=True``, the
LeRobot default) and ``gripper.pos`` in 0..100; ``image.exterior`` and (with
``--wrist-cam``) ``image.wrist``, each frame bounded to a 416x416 box with
its aspect kept (a 640x480 camera sends 416x312), as JPEG. The client
records one entry per 15 Hz row of the executed chunk, on the execution
clock, so the stack holds the rows the model trained on (``stack_entries``
of the handshake: 25 for the default context of one chunk).

Model space (the server's business, ``praxis/so101.py``): radians and a
gripper closure 0..1 between the readings ``--grip-closed`` (closure 1) and
``--grip-open`` (closure 0). Training mapped each table's own p1 / p99
``gripper.pos`` to closure 1 / 0, so "low = closed" is an assumption about the
training data, not a rule of LeRobot: give the readings of your follower
(``--measure-grip`` reads them with the torque off), ``closed > open`` is
fine. The defaults 1.4 / 42.1 are the training medians, to be replaced.

Command contract (what the client plays): the answer's ``actions``
``(R, 6)``, absolute targets in the units of ``state``, one row per 1/15 s
(21 rows on the first chunk of a context, 24 after). ``--control-hz`` (30 by
default, the LeRobot control rate) sends the arm a linear interpolation
between the 15 Hz rows. A deadline that passed during a stall (a camera
read, a GC pause) slips: the client slows down, it never sends the overdue
rows back to back. Safety is LeRobot's: ``--max-relative-target``
(the follower's ``max_relative_target``, LeRobot's own cap per command);
``--max-step-deg`` / ``--max-grip-step`` add a per-row clamp of the client
(off by default). At the start the client checks that the arm reports the
folded rest pose of the training episodes (``shoulder_lift`` about -100 deg,
``elbow_flex`` about 95 deg): a far reading means another calibration (not
the ``use_degrees`` zero of training) or a start out of distribution, and
stops the client (``--no-rest-check`` warns instead). Ctrl-C holds the current
position, then disconnects; LeRobot releases the torque on disconnect
(``disable_torque_on_disconnect``) unless ``--hold-torque-on-exit``.
``--dry-run`` prints the targets and never moves the arm; ``--fake-robot``
also skips the hardware (synthetic frames, a pose that follows the
commands), for a test of the endpoint.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import msgpack
import numpy as np

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
FPS = 15.0
FIT_SIDE = 416
# The follower's folded rest pose in the training data (use_degrees), and how
# far from it the start may be.
REST_POSE_DEG = {"shoulder_lift": -100.0, "elbow_flex": 95.0}
REST_TOL_DEG = 25.0
# The training medians of the gripper readings at closure 1 / 0.
DEFAULT_GRIP_CLOSED = 1.4
DEFAULT_GRIP_OPEN = 42.1
MAX_MESSAGE_BYTES = int(1.9 * 1024 * 1024)
# The session parameters of the server that the client sets from its flags.
SESSION_FLAGS = ("grip_closed", "grip_open", "max_context_chunks", "exec_horizon")


# -- wire (positronic msgpack numpy / JPEG records) ----------------------------


def _pack(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str,
                b"shape": obj.shape}  # fmt: skip
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj: dict) -> object:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])  # fmt: skip
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def dumps(x: object) -> bytes:
    return msgpack.packb(x, default=_pack)


def loads(b: bytes) -> Any:
    return msgpack.unpackb(b, object_hook=_unpack, strict_map_key=False)


def _s(d: Any, k: str) -> Any:
    """The value of ``k`` in a decoded message (str or bytes keys)."""
    return d[k] if k in d else d.get(k.encode())


def fit_frame(frame: np.ndarray, side: int = FIT_SIDE) -> np.ndarray:
    """Downscale (aspect kept, antialiased) so the long side is at most
    ``side``: the model's fit-416 canvas never upscales, so nothing is lost."""
    from PIL import Image

    h, w = frame.shape[:2]
    if max(h, w) <= side:
        return frame
    nh, nw = min(h, side, h * side // w), min(w, side, w * side // h)
    return np.asarray(
        Image.fromarray(frame).resize((nw, nh), Image.Resampling.BILINEAR)
    )


def jpeg(frames: list[np.ndarray], quality: int) -> dict:
    from PIL import Image

    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return {b"__jpeg__": True, b"frames": out, b"ndim": 4}


def build_request(
    entries: list[dict], views: list[str], prompt: str, quality: int
) -> bytes:
    """The stack (oldest first; the last entry is the current one) as one
    msgpack request, the JPEG quality lowered until it fits a 2 MB message."""
    state = np.stack([e["state"] for e in entries]).astype(np.float32)
    q = quality
    while True:
        msg = {"state": state, "task": prompt}
        for v in views:
            msg[f"image.{v}"] = jpeg([e[v] for e in entries], q)
        raw = dumps(msg)
        if len(raw) <= MAX_MESSAGE_BYTES or q <= 40:
            return raw
        q -= 10


# -- robot -----------------------------------------------------------------------


class FakeRobot:
    """No hardware: the rest pose that follows the commands, noise frames."""

    def __init__(self, cams: list[str], hw: tuple[int, int] = (480, 640)) -> None:
        self.cams, self.hw = cams, hw
        self.q = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])
        self.rng = np.random.default_rng(0)
        self.torque = True

    def connect(self) -> None:
        pass

    def get_observation(self) -> dict:
        obs = {f"{j}.pos": float(v) for j, v in zip(JOINTS, self.q, strict=True)}
        for c in self.cams:
            obs[c] = self.rng.integers(0, 255, (*self.hw, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict) -> dict:
        self.q = np.array([action[f"{j}.pos"] for j in JOINTS])
        return action

    def disable_torque(self) -> None:
        self.torque = False

    def disconnect(self) -> None:
        pass


def camera_source(spec: str) -> int | Path:
    """An OpenCV index (``0``) or a device path (``/dev/video0``)."""
    return int(spec) if spec.isdigit() else Path(spec)


def camera_configs(a: argparse.Namespace, cams: dict[str, str]) -> dict:
    """The LeRobot camera config of each view.

    ``--exterior-cam`` / ``--wrist-cam`` name a camera of ``--cameras`` (the
    ``--robot.cameras`` dict of LeRobot: ``{name: {type, index_or_path,
    width, height, fps}}``), or give an OpenCV index or a device path for an
    OpenCV camera at ``--cam-width`` x ``--cam-height``, ``--cam-fps``.
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    given = dict(json.loads(a.cameras)) if a.cameras else {}
    out = {}
    for view, spec in cams.items():
        if spec in given:
            cfg = dict(given[spec])
            kind = cfg.pop("type", "opencv")
            if kind == "opencv":
                out[view] = OpenCVCameraConfig(**cfg)
            elif kind in ("intelrealsense", "realsense"):
                from lerobot.cameras.realsense.configuration_realsense import (
                    RealSenseCameraConfig,
                )

                out[view] = RealSenseCameraConfig(**cfg)
            else:
                sys.exit(f"camera {spec!r}: unknown type {kind!r}")
        else:
            out[view] = OpenCVCameraConfig(
                index_or_path=camera_source(spec),
                fps=a.cam_fps,
                width=a.cam_width,
                height=a.cam_height,
            )
    return out


def make_robot(a: argparse.Namespace, cams: dict[str, str]) -> Any:
    if a.fake_robot:
        return FakeRobot(list(cams))
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # LeRobot before the so_follower module
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=a.robot_port,
        id=a.robot_id,
        use_degrees=True,
        max_relative_target=a.max_relative_target,
        disable_torque_on_disconnect=not a.hold_torque_on_exit,
        cameras=camera_configs(a, cams),
    )
    return SO101Follower(cfg)


def read_entry(robot: Any, views: list[str]) -> dict[str, Any]:
    obs = robot.get_observation()
    e: dict[str, Any] = {"state": np.array([float(obs[f"{j}.pos"]) for j in JOINTS])}
    for v in views:
        e[v] = fit_frame(np.asarray(obs[v], np.uint8))
    return e


def send(robot: Any, q: np.ndarray, dry: bool) -> None:
    if not dry:
        robot.send_action(
            {f"{j}.pos": float(x) for j, x in zip(JOINTS, q, strict=True)}
        )


def rest_check(state: np.ndarray, tol: float = REST_TOL_DEG) -> list[str]:
    """The joints of the folded rest pose that read more than ``tol`` deg off."""
    far = []
    for name, want in REST_POSE_DEG.items():
        got = float(state[JOINTS.index(name)])
        if abs(got - want) > tol:
            far.append(f"{name}.pos {got:.1f} deg (rest {want:.0f})")
    return far


def clamp_chunk(
    targets: np.ndarray,
    start: np.ndarray,
    max_deg: float | None,
    max_grip: float | None,
) -> np.ndarray:
    """Each row's move from the previous (clamped) command limited per joint.

    ``None`` leaves a limit off. The gripper stays in 0..100.
    """
    out = np.array(targets, np.float64, copy=True)
    if max_deg is not None or max_grip is not None:
        lim = np.array([np.inf if max_deg is None else max_deg] * 5
                       + [np.inf if max_grip is None else max_grip])  # fmt: skip
        prev = np.asarray(start, np.float64)
        for i, t in enumerate(targets):
            prev = prev + np.clip(t - prev, -lim, lim)
            out[i] = prev
    out[:, 5] = np.clip(out[:, 5], 0.0, 100.0)
    return out


def interpolate_rows(prev: np.ndarray, row: np.ndarray, sub: int) -> np.ndarray:
    """The ``sub`` commands from ``prev`` (not included) to ``row`` (included):
    the control ticks between two 15 Hz rows."""
    k = np.arange(1, sub + 1, dtype=np.float64)[:, None] / sub
    return prev[None] + (row - prev)[None] * k


def play_rows(
    robot: Any,
    rows: np.ndarray,
    start: np.ndarray,
    anchor: dict[str, Any],
    hist: collections.deque,
    views: list[str],
    sub: int,
    period: float,
    dry: bool,
    paced: bool,
) -> np.ndarray:
    """Play the 15 Hz ``rows`` of a chunk on the control clock; return the last row.

    Each row gets the ``sub`` commands of :func:`interpolate_rows` from the
    previous one (``start`` before row 0), one command per ``period`` seconds
    (``paced``; off with the fake robot). The stack entry of a row is read as
    the row starts and appended to ``hist`` (row 0's is ``anchor``), so the
    stack holds one entry per executed 15 Hz row.

    A deadline that is already past when a command goes out (a camera or USB
    stall in ``read_entry``, a GC pause) slips: the next deadline is one
    ``period`` after now, not after the missed one. The rows that were due
    during the stall are not sent back to back, so a stall slows the arm
    down instead of making it catch up (and the entries read after it stay
    one row period apart).
    """
    prev = start
    tick = time.monotonic()
    for i, row in enumerate(rows):
        hist.append(anchor if i == 0 else read_entry(robot, views))
        for q in interpolate_rows(prev, row, sub):
            send(robot, q, dry)
            tick = max(tick, time.monotonic()) + period
            if paced:
                time.sleep(max(0.0, tick - time.monotonic()))
        prev = row
    return prev


def measure_grip(robot: Any) -> tuple[float, float]:
    """Read ``gripper.pos`` with the torque off: closed by hand, then open."""
    bus = getattr(robot, "bus", robot)
    bus.disable_torque()
    out = []
    for what in ("CLOSE the gripper fully by hand", "OPEN the gripper fully by hand"):
        input(f"{what}, then press Enter... ")
        reading = float(robot.get_observation()["gripper.pos"])
        print(f"gripper.pos = {reading:.2f}")
        out.append(reading)
    return out[0], out[1]


# -- main ------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True, help="ws[s]://.../api/v1/session")
    ap.add_argument("--headers-file", default="",
                    help="JSON file of request headers (e.g. Modal-Key / Modal-Secret)")  # fmt: skip
    ap.add_argument("--robot-port", default="/dev/ttyACM0")
    ap.add_argument(
        "--robot-id", default="so101_follower", help="LeRobot calibration id"
    )
    ap.add_argument("--cameras", default="",
                    help="JSON: LeRobot --robot.cameras dict, {name: {type, index_or_path, ...}}")  # fmt: skip
    ap.add_argument("--exterior-cam", default="0",
                    help="view 0: a camera name of --cameras, an OpenCV index or a path")  # fmt: skip
    ap.add_argument("--wrist-cam", default=None,
                    help="view 1 (same forms); omit for a 1-view session")  # fmt: skip
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--prompt", default=None, help="the task (required to run)")
    ap.add_argument("--grip-closed", type=float, default=DEFAULT_GRIP_CLOSED,
                    help="this follower's gripper.pos fully closed (--measure-grip)")  # fmt: skip
    ap.add_argument("--grip-open", type=float, default=DEFAULT_GRIP_OPEN,
                    help="this follower's gripper.pos fully open (--measure-grip)")  # fmt: skip
    ap.add_argument("--max-context-chunks", type=int, default=None,
                    help="context chunks of the window, 0..3 (server default 1)")  # fmt: skip
    ap.add_argument(
        "--exec-horizon", type=int, default=None, help="rows executed per chunk"
    )
    ap.add_argument("--session", default=None,
                    help='JSON of other session parameters, e.g. \'{"num_steps": 4, "sample_seed": 0}\'')  # fmt: skip
    ap.add_argument("--control-hz", type=float, default=30.0,
                    help="commands per second: 15, or a multiple (interpolated)")  # fmt: skip
    ap.add_argument("--max-step-deg", type=float, default=None,
                    help="client clamp per 15 Hz row, per joint (default off)")  # fmt: skip
    ap.add_argument("--max-grip-step", type=float, default=None,
                    help="client clamp per 15 Hz row, gripper.pos units (default off)")  # fmt: skip
    ap.add_argument("--max-relative-target", type=float, default=None,
                    help="LeRobot's max_relative_target of the follower (its own cap)")  # fmt: skip
    ap.add_argument(
        "--max-chunks", type=int, default=0, help="stop after N chunks (0 = forever)"
    )
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--connect-timeout", type=float, default=1800.0,
                    help="seconds; a cold Modal container boots in minutes")  # fmt: skip
    ap.add_argument("--recv-timeout", type=float, default=60.0,
                    help="seconds to wait for a chunk before holding and stopping")  # fmt: skip
    ap.add_argument("--dry-run", action="store_true", help="print targets, never move")
    ap.add_argument(
        "--fake-robot", action="store_true", help="no hardware (implies --dry-run)"
    )
    ap.add_argument("--hold-torque-on-exit", action="store_true",
                    help="keep the torque on at disconnect (LeRobot releases it by default)")  # fmt: skip
    ap.add_argument("--no-rest-check", action="store_true",
                    help="warn instead of stopping when the arm is far from the rest pose")  # fmt: skip
    ap.add_argument("--measure-grip", action="store_true",
                    help="torque off, read gripper.pos closed and open, then exit")  # fmt: skip
    ap.add_argument("--log-jsonl", default=None, help="append per-chunk records here")
    a = ap.parse_args(argv)
    if a.fake_robot:
        a.dry_run = True
    if a.control_hz < FPS:
        ap.error("--control-hz must be >= 15")
    if not a.measure_grip and a.prompt is None:
        ap.error("--prompt is required")
    if abs(a.grip_open - a.grip_closed) < 1.0:
        ap.error("--grip-closed and --grip-open must differ by at least 1")
    return a


def auth_headers(a: argparse.Namespace) -> dict[str, str]:
    """The request headers of the session (never printed)."""
    if a.headers_file:
        return dict(json.loads(Path(a.headers_file).expanduser().read_text()))
    return {}


def session_url(a: argparse.Namespace, n_views: int) -> str:
    """The session URL: the rig parameters and the other session parameters
    of the server, under the server's own names."""
    q: dict[str, str] = {"views": str(n_views), "prompt": a.prompt}
    for k in SESSION_FLAGS:
        v = getattr(a, k)
        if v is not None:
            q[k] = json.dumps(v)
    for k, v in json.loads(a.session or "{}").items():
        q[k] = json.dumps(v)
    return f"{a.url}?{urlencode(q)}"


def handshake(ws: Any) -> dict:
    while True:
        m = loads(ws.recv())
        st = _s(m, "status")
        st = st.decode() if isinstance(st, bytes) else st
        if st == "ready":
            return _s(m, "meta")
        if st == "error":
            sys.exit(f"server refused the session: {_s(m, 'error')}")
        print(f"[server] {st}: {_s(m, 'message')}", flush=True)


def main(argv: list[str] | None = None) -> None:
    from websockets.sync.client import connect

    a = parse_args(argv)
    cams = {"exterior": a.exterior_cam}
    if a.wrist_cam is not None:
        cams["wrist"] = a.wrist_cam
    views = list(cams)
    robot = make_robot(a, cams)
    robot.connect()
    if a.measure_grip:
        try:
            closed, opened = measure_grip(robot)
        finally:
            robot.disconnect()
        print(f"run with: --grip-closed {closed:.2f} --grip-open {opened:.2f}")
        return
    start_state = read_entry(robot, [])["state"]
    far = rest_check(start_state)
    if far:
        msg = "the arm is not at the folded rest pose of training: " + ", ".join(far)
        if not a.no_rest_check:
            robot.disconnect()
            sys.exit(
                f"{msg}. Move it to the rest pose (or check the calibration: "
                "use_degrees=True, 0 = the mid-range), or pass --no-rest-check"
            )
        print(f"WARNING: {msg}", flush=True)
    url = session_url(a, len(views))
    sub = max(1, round(a.control_hz / FPS))
    period = 1.0 / (FPS * sub)
    log = open(a.log_jsonl, "a") if a.log_jsonl else None
    last_cmd: np.ndarray | None = None
    try:
        with connect(url, additional_headers=auth_headers(a),
                     max_size=16 * 1024 * 1024, open_timeout=a.connect_timeout,
                     close_timeout=5) as ws:  # fmt: skip
            meta = handshake(ws)
            print(f"[server] ready: {_s(meta, 'arm')} ({_s(meta, 'embodiment')}), views "
                  f"{_s(meta, 'views')}, steps {_s(meta, 'num_steps')}, cfg_video "
                  f"{_s(meta, 'cfg_video_scale')}, context {_s(meta, 'max_context_chunks')}, "
                  f"grip {_s(meta, 'grip_range')}", flush=True)  # fmt: skip
            hist: collections.deque = collections.deque(
                maxlen=max(0, int(_s(meta, "stack_entries")) - 1)
            )
            chunk = 0
            while not a.max_chunks or chunk < a.max_chunks:
                anchor = read_entry(robot, views)
                req = build_request([*hist, anchor], views, a.prompt, a.jpeg_quality)
                t0 = time.monotonic()
                ws.send(req)
                try:
                    rep = loads(ws.recv(timeout=a.recv_timeout))
                except TimeoutError:
                    sys.exit(
                        f"no answer in {a.recv_timeout:.0f} s (connection lost?): "
                        "holding and stopping"
                    )
                rtt = (time.monotonic() - t0) * 1e3
                err = _s(rep, "error")
                if err:
                    sys.exit(f"server error: {err}")
                targets = np.asarray(_s(rep, "actions"), np.float64)
                if not np.isfinite(targets).all() or targets.shape[1:] != (6,):
                    sys.exit(f"bad chunk {targets.shape}")
                start = anchor["state"] if last_cmd is None else last_cmd
                safe = clamp_chunk(targets, start, a.max_step_deg, a.max_grip_step)
                timing = _s(rep, "server_timing") or {}
                print(f"chunk {chunk}: {len(safe)} rows, rtt {rtt:.0f} ms "
                      f"(policy {_s(timing, 'policy_ms') or 0:.0f} ms), request "
                      f"{len(req) / 1e6:.2f} MB, stack {len(hist) + 1}, clamped "
                      f"{float(np.abs(safe - targets).max()):.2f}", flush=True)  # fmt: skip
                if a.dry_run:
                    for i in (0, len(safe) // 2, len(safe) - 1):
                        cols = " ".join(
                            f"{j}={x:7.2f}"
                            for j, x in zip(JOINTS, safe[i], strict=True)
                        )
                        print(f"   row {i:2d}: {cols}")
                if log is not None:
                    log.write(json.dumps({"chunk": chunk, "rtt_ms": rtt, "anchor": anchor["state"].tolist(),
                                          "targets": targets.tolist(), "sent": safe.tolist(),
                                          "server_timing": {k.decode() if isinstance(k, bytes) else k: v
                                                            for k, v in dict(timing).items()}}) + "\n")  # fmt: skip
                    log.flush()
                prev = play_rows(robot, safe, start, anchor, hist, views, sub, period,
                                 dry=a.dry_run, paced=not a.fake_robot)  # fmt: skip
                last_cmd = prev if not a.dry_run else None
                chunk += 1
    except KeyboardInterrupt:
        print("\nCtrl-C: holding the current position and stopping", flush=True)
    finally:
        try:
            if not a.dry_run:
                hold = read_entry(robot, [])["state"]
                send(robot, hold, False)
        finally:
            robot.disconnect()
            if log is not None:
                log.close()


if __name__ == "__main__":
    main()
