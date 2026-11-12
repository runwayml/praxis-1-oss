#!/usr/bin/env python
"""Run an so101 praxis policy server on an SO-100 / SO-101 follower arm through LeRobot.

The remote client of a praxis serve endpoint on an so101 checkpoint
(``python -m praxis.serve --config configs/serve_so101.yaml`` on a local
GPU, for example a 5090 with ``routed_precision: nvfp4``, or the Modal app).
It is one file without a praxis import: next to the robot it needs
``lerobot`` (with the Feetech extra), ``numpy``, ``msgpack``, ``websockets``
and ``Pillow``.

    python demo/so101/so101_client.py \\
        --url ws://<host>:8000/api/v1/session \\
        --robot-port /dev/ttyACM0 --robot-id my_so101 \\
        --exterior-cam 0 [--wrist-cam 2] \\
        --grip-closed 1.4 --grip-open 42.1 \\
        --prompt "Pick up the red cube and put it in the box." \\
        [--dry-run] [--control-hz 30] [--max-step-deg 4] [--ui-port 8780]

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
(off by default).

Rest pose: the client moves the arm to the folded rest pose of the training
episodes before the first chunk, and back to it when the client ends
(Ctrl-C, ``--max-chunks`` or a server error without the page, SIGTERM or
SIGHUP). The pose is :data:`REST_JOINTS_DEG` (``--rest-pose`` gives another)
and the gripper at ``--grip-closed``. :func:`move_to_rest` moves in three
phases (shoulder lift and elbow, then the pan, then the wrist and the
gripper) at :data:`REST_SPEED` at most, and no command leads the reading by
more than :data:`REST_LEAD`, so a blocked joint gets a bounded push. A wrist
that is flexed past the rest value unflexes first: it would block the fold.
After the start move the client checks ``shoulder_lift`` (about -100 deg)
and ``elbow_flex`` (about 95 deg):
a far reading means another calibration (not the ``use_degrees`` zero of
training) or an obstacle, and stops the client (``--no-rest-check`` warns
instead). After the end move LeRobot releases the torque on disconnect
(``disable_torque_on_disconnect``) unless ``--hold-torque-on-exit``. Ctrl-C
during a rest move stops it: the arm is released where it is.
``--no-rest-move`` turns both moves off: the client then only checks the
start pose and releases the arm where the run ends.

Page: ``--ui-port 8780`` serves a page on ``--ui-host`` (127.0.0.1): the
live views as the model gets them, the prompt with a field to change it, the
state of the client and three buttons. The client then waits for ``Start``
(``--autostart`` starts at once). ``Start`` runs a session; ``Stop`` ends
the session or the rest move that runs, and the arm holds its position;
``Return to rest`` moves the arm to the rest pose. No button ends the
client: Ctrl-C does, and the arm then returns to the rest pose. With the
page, ``--max-chunks`` or a problem of the server ends the session, not the
client. A new prompt goes out with the next chunk request (the ``task`` of
the request); the page shows the prompt that the server answers with. The
page has no password: keep the default host.

``--dry-run`` prints the targets and never moves the arm (no rest move);
``--fake-robot`` also skips the hardware (synthetic frames, a pose that
follows the commands), for a test of the endpoint.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
#: Seconds to wait for a camera frame that is late before the session ends.
FRAME_WAIT_S = 6.0
# The five joints of that rest pose, the target of the rest moves (the
# gripper rests at --grip-closed).
REST_JOINTS_DEG = (0.0, -100.0, 95.0, 70.0, 0.0)
# The phases of a rest move: fold the arm, turn it, then set the wrist and
# the gripper. The wrist comes last: a wrist that flexes down while the arm
# is still low would push the gripper into the table. One case comes first
# (:func:`rest_phases`): a wrist that is flexed past the rest value unflexes
# before the fold, because the gripper then blocks the elbow (measured on an
# SO-101: with the wrist of a reach pose the elbow stopped at 69.5 deg).
REST_PHASES = (
    ("shoulder_lift", "elbow_flex"),
    ("shoulder_pan",),
    ("wrist_flex", "wrist_roll", "gripper"),
)
# The move runs a second time when the fold ends more than REST_RETRY from
# the rest value and the pass moved another joint: that may have cleared
# what blocked the fold.
REST_RETRY = 10.0
# In a rest move the farthest joint of a phase goes at REST_SPEED at most
# (degrees per second, gripper.pos units for the gripper), and a command
# leads the reading by at most REST_LEAD. A phase ends when its joints are
# within REST_ARRIVE, or REST_SETTLE_S after the command reached the rest
# value, or when no joint came REST_PROGRESS closer for REST_STALL_S (a
# blocked joint).
REST_SPEED = 40.0
REST_LEAD = 8.0
REST_ARRIVE = 3.0
REST_SETTLE_S = 1.0
REST_PROGRESS = 0.5
REST_STALL_S = 1.5
# The longest prompt that the page accepts, in characters.
MAX_PROMPT_CHARS = 500
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


def read_joints(robot: Any) -> dict[str, Any]:
    """The joint readings without the cameras. LeRobot's ``get_observation``
    also reads each camera and raises on a late frame; the hold, the rest
    moves and the readings of the page must not stop on a camera."""
    bus = getattr(robot, "bus", None)
    if hasattr(bus, "sync_read"):
        # LeRobot's own readings pass these retries; sync_read has none by default
        retries = getattr(getattr(robot, "config", None), "num_read_retries", 2)
        read = bus.sync_read("Present_Position", num_retry=retries)
        return {f"{m}.pos": v for m, v in read.items()}
    return robot.get_observation()


CAMERA_LOCK = threading.Lock()


def revive_cameras(robot: Any) -> list[str]:
    """Connect again each camera whose read thread has ended; return their
    names. LeRobot's OpenCV camera ends its read thread after about ten
    failed reads in a row (a short USB stall is enough) and never starts it
    again: every later read then raises."""
    revived = []
    cams = getattr(robot, "cameras", None)
    if not isinstance(cams, dict):
        return revived
    with CAMERA_LOCK:
        for name, cam in cams.items():
            thread = getattr(cam, "thread", None)
            if thread is None or thread.is_alive():
                continue
            try:
                with contextlib.suppress(Exception):
                    cam.disconnect()
                cam.connect()
                revived.append(name)
                say(f"[client] camera {name}: its reader had ended, connected again")
            except Exception as e:
                say(f"[client] camera {name}: no reconnect: {type(e).__name__}: {e}")
    return revived


def read_observation(robot: Any) -> dict[str, Any]:
    """LeRobot's ``get_observation`` (the joints and the cameras). A late
    frame is asked again, and a camera whose reader has ended is connected
    again (:func:`revive_cameras`), for :data:`FRAME_WAIT_S`; then the
    session ends (:class:`SessionError`), not the client."""
    t0 = time.monotonic()
    while True:
        try:
            return robot.get_observation()
        except (TimeoutError, RuntimeError) as e:
            if time.monotonic() - t0 > FRAME_WAIT_S:
                raise SessionError(f"a camera sends no frame: {e}") from e
            if not revive_cameras(robot):
                time.sleep(0.05)


def read_entry(robot: Any, views: list[str], ui: Ui | None = None) -> dict[str, Any]:
    """One stack entry: the joint readings and the fitted frame of each view.
    ``ui`` gets the frames and the readings for the page."""
    obs = read_observation(robot) if views else read_joints(robot)
    e: dict[str, Any] = {"state": np.array([float(obs[f"{j}.pos"]) for j in JOINTS])}
    for v in views:
        e[v] = fit_frame(np.asarray(obs[v], np.uint8))
        if ui is not None:
            ui.publish(v, e[v])
    if ui is not None:
        ui.update(state=[round(float(x), 1) for x in e["state"]])
    return e


def send(robot: Any, q: np.ndarray, dry: bool) -> None:
    if not dry:
        robot.send_action(
            {f"{j}.pos": float(x) for j, x in zip(JOINTS, q, strict=True)}
        )


def rest_check(
    state: np.ndarray, tol: float = REST_TOL_DEG, rest: np.ndarray | None = None
) -> list[str]:
    """The joints of the folded rest pose that read more than ``tol`` deg off
    (from ``rest``, the target of the rest move, when given)."""
    far = []
    for name, want in REST_POSE_DEG.items():
        want = want if rest is None else float(rest[JOINTS.index(name)])
        got = float(state[JOINTS.index(name)])
        if abs(got - want) > tol:
            far.append(f"{name}.pos {got:.1f} deg (rest {want:.0f})")
    return far


def rest_pose_arg(text: str) -> tuple[float, ...]:
    """``--rest-pose``: 5 joint angles in degrees, comma separated."""
    try:
        values = tuple(float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not 5 numbers: {text!r}") from None
    if len(values) != 5 or not all(np.isfinite(v) and abs(v) <= 200.0 for v in values):
        raise argparse.ArgumentTypeError(
            f"the rest pose is 5 angles in degrees (pan,lift,elbow,wrist_flex,"
            f"wrist_roll), got {text!r}"
        )
    return values


def rest_vector(a: argparse.Namespace) -> np.ndarray:
    """The rest pose in the units of ``state``: ``--rest-pose`` (default
    :data:`REST_JOINTS_DEG`) and the gripper at ``--grip-closed``."""
    joints = REST_JOINTS_DEG if a.rest_pose is None else a.rest_pose
    return np.array([*joints, min(100.0, max(0.0, a.grip_closed))], np.float64)


def rest_phases(q: np.ndarray, rest: np.ndarray) -> list[tuple[str, ...]]:
    """The phases of a rest move from the reading ``q``: :data:`REST_PHASES`,
    after a first phase that unflexes the wrist when it is flexed past the rest
    value (the rest value lies between the reading and 0). That move brings
    the gripper toward the line of the forearm, so it frees the fold and does
    not push the gripper down."""
    w = JOINTS.index("wrist_flex")
    past = q[w] != 0 and 0 <= rest[w] / q[w] < 1 and abs(q[w] - rest[w]) > REST_ARRIVE
    return [*([("wrist_flex",)] if past else []), *REST_PHASES]


def rest_phase(
    robot: Any,
    rest: np.ndarray,
    cmd: np.ndarray,
    names: tuple[str, ...],
    period: float,
    paced: bool,
    stop: Callable[[], bool] | None = None,
) -> bool:
    """Move the joints ``names`` to their rest values; the other joints keep
    ``cmd``. ``cmd`` is updated in place. Return False when the joints stopped
    short (no progress for :data:`REST_STALL_S`). ``stop`` is asked before
    each command: True raises :class:`StopRun`."""
    idx = [JOINTS.index(n) for n in names]
    origin = cmd[idx].copy()
    line = rest[idx] - origin
    step = REST_SPEED * period / max(float(np.abs(line).max()), 1e-9)
    frac = 0.0  # the fraction of the line that the command has reached
    best, t_best, t_rest = np.inf, time.monotonic(), None
    tick = time.monotonic()
    while True:
        if stop is not None and stop():
            raise StopRun
        q = read_entry(robot, [])["state"]
        far = float(np.abs(rest[idx] - q[idx]).max())
        now = time.monotonic()
        if far <= REST_ARRIVE and frac >= 1.0:
            return True  # the command is the rest value
        if far < best - REST_PROGRESS:
            best, t_best = far, now
        elif now - t_best > REST_STALL_S:
            cmd[idx] = q[idx]  # blocked: end the push
            send(robot, cmd, False)
            return False
        if frac >= 1.0:  # the command is the rest value from here on
            t_rest = now if t_rest is None else t_rest
            if now - t_rest > REST_SETTLE_S:
                if far > REST_ARRIVE:  # short of rest: end the push, as for a stall
                    cmd[idx] = q[idx]
                    send(robot, cmd, False)
                return True
        ahead = min(1.0, frac + step)
        if float(np.abs(origin + ahead * line - q[idx]).max()) <= REST_LEAD:
            frac = ahead
        lead = np.clip(origin + frac * line - q[idx], -REST_LEAD, REST_LEAD)
        cmd[idx] = q[idx] + lead
        send(robot, cmd, False)
        tick = max(tick, time.monotonic()) + period
        if paced:
            time.sleep(max(0.0, tick - time.monotonic()))


def move_to_rest(
    robot: Any,
    rest: np.ndarray,
    hz: float = 30.0,
    paced: bool = True,
    stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Move the arm to ``rest`` (the units of ``state``); return the last reading.

    The phases of :func:`rest_phases` run one after the other
    (:func:`rest_phase`). In a phase the command goes along the straight line
    from the start of the phase to the rest value, the same fraction for each
    joint, so the joints of a phase arrive together. The farthest joint goes
    at :data:`REST_SPEED`. The command waits while it would lead the reading
    of a joint by more than :data:`REST_LEAD`, and it is clipped to that
    lead: a slow joint slows the move down, and a blocked joint gets a
    bounded push. The joints of the earlier phases hold the rest value, the
    joints of the later phases hold the reading of the start.

    A phase ends when its joints are within :data:`REST_ARRIVE` of the rest
    value, or :data:`REST_SETTLE_S` after the command reached it. A phase
    also ends when no joint came :data:`REST_PROGRESS` closer for
    :data:`REST_STALL_S`: the command of the blocked joints then goes back
    to the reading, and a line names them. When the fold ends more than
    :data:`REST_RETRY` from the rest value and the pass moved another joint
    (the pan, the wrist or the gripper), the phases run a second time.
    The caller checks the last reading (:func:`rest_check`). ``stop`` (the
    Stop button of the page) is asked before each command: True raises
    :class:`StopRun`, and the caller holds the arm.
    """
    period = 1.0 / hz
    fold = [JOINTS.index(n) for n in REST_PHASES[0]]
    others = [i for i in range(len(JOINTS)) if i not in fold]
    q = read_entry(robot, [])["state"]
    cmd = np.array(q, np.float64)
    for _ in range(2):
        before = q
        folded = True
        for names in rest_phases(q, rest):
            if names == ("shoulder_pan",) and not folded:
                say("[client] rest move: the fold stopped short, the pan stays")
                continue  # no swing of an arm that is not folded
            if not rest_phase(robot, rest, cmd, names, period, paced, stop):
                folded = folded and names != REST_PHASES[0]
                q = read_entry(robot, [])["state"]
                stopped = ", ".join(
                    f"{n} at {q[JOINTS.index(n)]:.1f} (rest {rest[JOINTS.index(n)]:.1f})"
                    for n in names
                )
                say(f"[client] rest move: no progress for {REST_STALL_S} s: {stopped}")
        q = read_entry(robot, [])["state"]
        short = float(np.abs(rest[fold] - q[fold]).max()) > REST_RETRY
        cleared = float(np.abs(q[others] - before[others]).max()) > REST_ARRIVE
        if not (short and cleared):  # a second pass needs something to have changed
            break
    return q


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
    *,
    ui: Ui | None = None,
    stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Play the 15 Hz ``rows`` of a chunk on the control clock; return the last row.

    Each row gets the ``sub`` commands of :func:`interpolate_rows` from the
    previous one (``start`` before row 0), one command per ``period`` seconds
    (``paced``; off with the fake robot). The stack entry of a row is read as
    the row starts and appended to ``hist`` (row 0's is ``anchor``), so the
    stack holds one entry per executed 15 Hz row. ``ui`` gets the frames of
    each entry. ``stop`` is asked before each row: True raises
    :class:`StopRun` (the stop button of the page).

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
        if stop is not None and stop():
            raise StopRun
        hist.append(anchor if i == 0 else read_entry(robot, views, ui))
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


# -- page ------------------------------------------------------------------------


class StopRun(Exception):
    """A stop or rest request of the page: end the session or the rest move
    that runs. The client holds the arm and goes on."""


def say(text: str) -> None:
    """Print a line. A closed terminal (SIGHUP) must not end the rest move."""
    with contextlib.suppress(OSError, ValueError):
        print(text, flush=True)


# The page: one HTML file, no request to another host (system fonts, no
# library). The views at the left, the state, the prompt and the buttons in
# the panel at the right.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 client</title>
<style>
  :root { --bg: #ffffff; --text: #111111; --muted: #6b6b6b; --border: #e6e6e6;
    --surface: #f2f2f2; --surface-hover: #e8e8e8; --radius: 12px; --radius-sm: 6px; }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  body { margin: 0; min-height: 100vh; display: flex; flex-direction: column;
    background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.4;
    font-family: Inter, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased; }
  :focus-visible { outline: 2px solid var(--text); outline-offset: 2px; }

  .pill { align-self: flex-start; padding: 4px 10px; border-radius: 999px; background: var(--surface);
    font-size: 12px; font-weight: 500; }

  .stage { flex: 1; min-height: 0; display: grid; gap: 24px;
    grid-template-columns: minmax(0, 1fr) 400px; padding: 24px 32px 32px; }

  /* The views: the first one large, the second and the readings stacked. */
  .feeds { --gap: 16px; display: grid; gap: var(--gap); align-content: start;
    grid-template-columns: minmax(0, 1fr) calc((100% - 7 * var(--gap) / 3) / 3); }
  .feed, .readout { aspect-ratio: 4 / 3; margin: 0; border: 1px solid var(--border);
    border-radius: var(--radius); }
  .feed { position: relative; overflow: hidden; background: #000; }
  .feed:first-child { grid-row: span 2; aspect-ratio: auto; }
  .feeds.single .feed { grid-row: auto; aspect-ratio: 4 / 3; }
  .feeds.single .readout { aspect-ratio: auto; align-self: start; }
  .feed img { position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: contain; }
  .feed figcaption { position: absolute; top: 12px; left: 12px; padding: 4px 10px;
    border-radius: 999px; background: rgba(255, 255, 255, 0.9); font-size: 12px;
    font-weight: 500; }
  .feed-status { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; padding: 0 16px; color: #8a8a8a; font-size: 13px;
    text-align: center; }
  .feed-status:empty { display: none; }

  .readout { display: flex; flex-direction: column; justify-content: center;
    gap: 12px; padding: 16px 20px; font-variant-numeric: tabular-nums; }
  .readout dl { display: grid; gap: 4px 20px; margin: 0; }
  .readout dl + dl { padding-top: 12px; border-top: 1px solid var(--border);
    grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); }
  .readout dl div { display: flex; justify-content: space-between; gap: 8px; }
  .readout dt { color: var(--muted); }
  .readout dd { margin: 0; white-space: nowrap; }

  .panel { display: flex; flex-direction: column; gap: 20px; padding: 28px;
    border: 1px solid var(--border); border-radius: var(--radius); }
  .panel h1 { margin: 0; font-size: 30px; font-weight: 400; line-height: 1.15;
    letter-spacing: -0.03em; }

  .state { display: flex; flex-direction: column; gap: 6px; padding: 18px 20px;
    border-radius: var(--radius); background: var(--surface);
    transition: background 0.2s, color 0.2s; }
  .state-label { display: flex; align-items: center; gap: 12px; font-size: 34px;
    font-weight: 600; line-height: 1.2; letter-spacing: -0.02em; }
  .state-label::before { content: ''; flex: none; width: 14px; height: 14px;
    border-radius: 50%; background: #c8c8c8; }
  .state-task { min-height: 2.8em; overflow-wrap: anywhere; }
  .state-phase { color: var(--muted); font-variant-numeric: tabular-nums; }
  .state.is-running { background: var(--text); color: #fff; }
  .state.is-running .state-phase { color: rgba(255, 255, 255, 0.7); }
  .state.is-running .state-label::before { background: #4ade80;
    box-shadow: 0 0 14px #4ade80; animation: pulse 1.4s ease-in-out infinite; }
  .state.is-offline { background: #fbeaea; color: #7a1f1f; }
  .state.is-offline .state-phase { color: #a04848; }
  .state.is-offline .state-label::before { background: #d94848; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  @media (prefers-reduced-motion: reduce) {
    .state-label::before { animation: none !important; }
    * { transition: none !important; }
  }
  .message { margin: -8px 0 0; color: #7a1f1f; overflow-wrap: anywhere; }

  .history { margin: 0; padding: 0; list-style: none; max-height: 9.5em;
    overflow-y: auto; color: var(--muted); }
  .history:empty { display: none; }
  .history li { padding: 3px 0; cursor: pointer; overflow-wrap: anywhere; }
  .history li:hover { color: var(--text); }
  .history li.is-active { color: var(--text); font-weight: 500; }

  .prompt { display: flex; flex-direction: column; gap: 8px; }
  .prompt textarea { width: 100%; min-height: 3.6em; padding: 10px 12px;
    font: inherit; line-height: 1.4; resize: none; overflow: hidden;
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    background: var(--bg); color: var(--text); }
  .prompt textarea:focus { outline: none; border-color: var(--text); }
  .note { margin: -12px 0 0; min-height: 1.4em; color: var(--muted);
    overflow-wrap: anywhere; }
  .note[data-kind=error] { color: #a04848; }

  .btn { padding: 10px 16px; border: 1px solid transparent; font: inherit;
    font-weight: 500; border-radius: var(--radius-sm); background: var(--surface);
    color: var(--text); cursor: pointer;
    transition: transform 0.08s, background 0.12s; }
  .btn:hover:not(:disabled) { background: var(--surface-hover); }
  .btn:active:not(:disabled) { transform: scale(0.96); }
  .btn-primary { background: var(--text); color: #fff; }
  .btn-primary:hover:not(:disabled) { background: #333; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .controls { margin-top: auto; display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; }
  .btn-big { padding: 18px; font-size: 18px; }
  #rest { grid-column: 1 / -1; }
  .hint { margin: -8px 0 0; color: var(--muted); font-size: 12px; }

  @media (min-width: 1800px) {
    body { font-size: 17px; }
    .stage { grid-template-columns: minmax(0, 1fr) 480px; }
    .panel h1 { font-size: 36px; }
    .state-label { font-size: 42px; }
    .btn-big { padding: 22px; font-size: 22px; }
  }
  @media (max-width: 1100px) {
    .stage { grid-template-columns: minmax(0, 1fr) 320px; gap: 16px;
      padding: 16px 20px 24px; }
    .panel { padding: 20px; }
    .feeds { grid-template-columns: 1fr 1fr; }
    .feeds .feed:first-child { grid-row: auto; grid-column: 1 / -1;
      aspect-ratio: 4 / 3; }
    .feeds.single .readout { grid-column: 1 / -1; }
  }
  @media (max-width: 720px) {
    .stage { grid-template-columns: minmax(0, 1fr); padding: 16px; }
    .feeds .feed:first-child { grid-column: auto; }
    .feeds .readout { grid-column: 1 / -1; aspect-ratio: auto; }
    .panel h1 { font-size: 24px; }
    .state-label { font-size: 28px; }
  }
</style>
</head>
<body>
<main class="stage">
  <section id="feeds" class="feeds" aria-label="Camera views as the model gets them">
    <div class="readout">
      <dl>
        <div><dt>Chunk</dt><dd id="chunk">-</dd></div>
        <div><dt>Round trip</dt><dd id="rtt">-</dd></div>
        <div><dt>Model time</dt><dd id="policy">-</dd></div>
      </dl>
      <dl id="state" title="Joints in degrees, gripper 0 to 100"></dl>
    </div>
  </section>
  <aside class="panel">
    <div id="dry" class="pill" hidden>Dry run: the arm does not move</div>
    <div id="card" class="state">
      <div id="label" class="state-label" role="status">Connecting</div>
      <div id="inuse" class="state-task"></div>
      <div id="phase" class="state-phase"></div>
    </div>
    <p id="message" class="message" hidden></p>
    <h1>What should the robot do?</h1>
    <ul id="history" class="history" aria-label="Earlier prompts"></ul>
    <form id="form" class="prompt">
      <textarea id="prompt" rows="2" maxlength="500" spellcheck="false"
                autocomplete="off" aria-label="New prompt"
                placeholder="Type a new prompt"></textarea>
      <button type="submit" id="apply" class="btn btn-primary" disabled>Set
        prompt</button>
    </form>
    <p id="note" class="note"></p>
    <div class="controls">
      <button type="button" id="start" class="btn btn-primary btn-big"
              disabled>Start</button>
      <button type="button" id="stop" class="btn btn-primary btn-big" disabled
              title="Ends the session and holds the arm in place">Stop</button>
      <button type="button" id="rest" class="btn" disabled
              title="Folds the arm to the rest pose">Return to rest</button>
    </div>
    <p class="hint">Stop holds the arm in place with the motors on. Ctrl-C in the
      terminal ends the client and returns the arm to rest.</p>
  </aside>
</main>
<script>
  const $ = (id) => document.getElementById(id);
  const JOINTS = ['Pan', 'Lift', 'Elbow', 'Wrist flex', 'Wrist roll', 'Gripper'];
  const LABELS = {idle: 'Idle', running: 'Running', moving: 'Moving'};
  const feeds = [], cells = [];
  let built = false, online = false, recent = [], shown = '';
  const post = (path, body) => fetch(path, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
  function build(s) {
    $('feeds').classList.toggle('single', s.views.length < 2);
    for (const view of s.views) {
      const fig = document.createElement('figure');
      fig.className = 'feed';
      const img = document.createElement('img');
      img.alt = cap(view) + ' camera';
      const label = document.createElement('figcaption');
      label.textContent = cap(view);
      const wait = document.createElement('div');
      wait.className = 'feed-status';
      img.addEventListener('load', () => {
        img.hidden = false;
        wait.textContent = '';
      });
      fig.append(img, label, wait);
      $('feeds').lastElementChild.before(fig);
      feeds.push({view, img, wait});
    }
    for (const name of JOINTS) {
      const row = document.createElement('div');
      const dt = document.createElement('dt');
      dt.textContent = name;
      const dd = document.createElement('dd');
      dd.textContent = '-';
      row.append(dt, dd);
      $('state').append(row);
      cells.push(dd);
    }
    try { recent = JSON.parse(localStorage.getItem('so101-prompts')) || []; }
    catch (e) { recent = []; }
    if (!Array.isArray(recent)) recent = [];
    built = true;
  }
  function connect(up) {  // the streams end with the client: open them again
    online = up;
    for (const f of feeds) {
      f.img.hidden = true;
      f.wait.textContent = up ? 'No frame yet' : 'No signal';
      if (up) f.img.src = '/stream/' + f.view + '.mjpg?' + Date.now();
      else f.img.removeAttribute('src');
    }
  }
  function history(current) {  // every prompt so far, the current one marked
    if (recent[0] !== current) {
      recent = [current, ...recent.filter((text) => text !== current)];
      try { localStorage.setItem('so101-prompts', JSON.stringify(recent)); }
      catch (e) {}
    }
    const key = JSON.stringify(recent);
    if (key === shown) return;
    shown = key;
    $('history').replaceChildren(...recent.map((text) => {
      const li = document.createElement('li');
      li.className = text === current ? 'is-active' : '';
      li.textContent = text;
      li.tabIndex = 0;
      const pick = () => { $('prompt').value = text; fit(); $('prompt').focus(); };
      li.addEventListener('click', pick);
      li.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') pick(); });
      return li;
    }));
  }
  async function setPrompt(text) {
    const r = await post('/prompt', {prompt: text});
    const body = await r.json();
    $('note').dataset.kind = r.ok ? '' : 'error';
    $('note').textContent = r.ok ? '' : cap(body.error);
    if (r.ok) { $('prompt').value = ''; fit(); }
    tick();
  }
  async function tick() {
    let s;
    try {
      s = await (await fetch('/status', {cache: 'no-store'})).json();
    } catch (e) {
      if (online || !built) connect(false);
      $('card').className = 'state is-offline';
      $('label').textContent = 'Offline';
      $('inuse').textContent = 'The client is not running.';
      $('phase').textContent = 'Start it again in the terminal.';
      for (const id of ['start', 'stop', 'rest', 'apply']) $(id).disabled = true;
      return;
    }
    if (!built) build(s);
    if (!online) connect(true);
    const mode = s.mode || 'idle';
    $('card').className = 'state' + (mode === 'idle' ? '' : ' is-running');
    $('label').textContent = LABELS[mode] || cap(mode);
    $('inuse').textContent = s.effective_prompt || s.prompt;
    $('phase').textContent = cap(s.phase) + ', ' + Math.round(s.phase_s) + ' s';
    $('message').textContent = s.message ? cap(s.message) : '';
    $('message').hidden = !s.message;
    $('dry').hidden = !s.dry_run;
    history(s.prompt);
    if (s.effective_prompt && s.prompt !== s.effective_prompt) {
      $('note').dataset.kind = '';
      $('note').textContent = 'From the next chunk: ' + s.prompt;
    } else if ($('note').dataset.kind !== 'error') {
      $('note').textContent = '';
    }
    $('chunk').textContent = s.chunk === undefined ? '-'
      : s.chunk + (s.rows ? ', ' + s.rows + ' rows' : '');
    $('rtt').textContent = s.rtt_ms === undefined ? '-' : s.rtt_ms + ' ms';
    $('policy').textContent = s.policy_ms === undefined ? '-' : s.policy_ms + ' ms';
    (s.state || []).slice(0, cells.length).forEach((value, i) => {
      cells[i].textContent = Number(value).toFixed(1) + (i < 5 ? '°' : '');
    });
    $('start').disabled = mode !== 'idle';
    $('stop').disabled = mode === 'idle';
    $('rest').disabled = mode === 'moving';
    $('apply').disabled = false;
  }
  function fit() {  // the field is as tall as its text
    $('prompt').style.height = 'auto';
    $('prompt').style.height = ($('prompt').scrollHeight + 2) + 'px';
  }
  $('prompt').addEventListener('input', fit);
  $('prompt').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      $('form').requestSubmit();
    }
  });
  $('form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    setPrompt($('prompt').value);
  });
  for (const id of ['start', 'stop', 'rest']) {
    $(id).addEventListener('click', async () => {
      $(id).disabled = true;
      await post('/' + id, {});
      tick();
    });
  }
  setInterval(tick, 500);
  tick();
</script>
</body>
</html>
"""

LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})


class Ui:
    """The page of the client: the live views, the prompt, the state and the
    buttons Start, Stop and Return to rest.

    An HTTP server in a daemon thread, standard library only. The routes:

    * ``/``: the page.
    * ``/status``: the state of the run as JSON.
    * ``/stream/<view>.mjpg``: the latest frame of a view as an MJPEG stream.
    * ``/frame/<view>.jpg``: the latest frame as one JPEG.
    * ``POST /prompt`` (JSON ``{"prompt": str}``): the prompt of the next
      chunk request.
    * ``POST /start``, ``POST /stop``, ``POST /rest``: the buttons
      (:meth:`request`). No route ends the client.

    The frames are the fitted frames that the model gets
    (:func:`read_entry`); :meth:`watch` also reads the cameras between the
    entries, so the views stay live while the client waits for the server.
    The page has no password. The server answers only a request whose
    ``Host`` is a loopback name or the address of ``--ui-host`` (no DNS
    rebinding; a bind on ``0.0.0.0`` answers the loopback names only); a
    ``POST`` must be JSON from the page's own origin (no cross-site request).
    """

    def __init__(self, host: str, port: int, prompt: str, views: list[str]) -> None:
        self.views = list(views)
        self.lock = threading.Lock()
        self.prompt = prompt
        self.status: dict[str, Any] = {"phase": "starting"}
        self.since = time.monotonic()
        self.frames: dict[str, tuple[int, np.ndarray]] = {}
        self.jpegs: dict[str, tuple[int, bytes]] = {}
        self.wanted: str | None = None  # the button that the client has not taken
        self.interrupt = threading.Event()  # stop / rest: end what runs
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.httpd = ThreadingHTTPServer((host, port), _handler(self, host))
        self.httpd.daemon_threads = True
        self.host, self.port = host, int(self.httpd.server_address[1])
        threading.Thread(
            target=self.httpd.serve_forever, name="so101-ui", daemon=True
        ).start()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def publish(self, view: str, frame: np.ndarray) -> None:
        """Store the latest frame of ``view`` (RGB uint8). No encoding here."""
        with self.lock:
            self.frames[view] = (self.frames.get(view, (0, None))[0] + 1, frame)

    def jpeg(self, view: str) -> tuple[int, bytes] | None:
        """The latest frame of ``view`` as (sequence number, JPEG), or None."""
        with self.lock:
            cur, done = self.frames.get(view), self.jpegs.get(view)
        if cur is None:
            return None
        if done is not None and done[0] == cur[0]:
            return done
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(cur[1]).save(buf, format="JPEG", quality=80)
        out = (cur[0], buf.getvalue())
        with self.lock:
            self.jpegs[view] = out
        return out

    def set_prompt(self, text: object) -> str:
        """Set the prompt of the next chunk request; return it normalized."""
        if not isinstance(text, str):
            raise ValueError("the prompt must be a string")
        new = " ".join(text.split())
        if not new:
            raise ValueError("the prompt is empty")
        if not new.isprintable():
            raise ValueError("the prompt has a control character")
        if len(new) > MAX_PROMPT_CHARS:
            raise ValueError(f"the prompt is longer than {MAX_PROMPT_CHARS} characters")
        with self.lock:
            self.prompt = new
        return new

    def get_prompt(self) -> str:
        with self.lock:
            return self.prompt

    def update(self, **values: Any) -> None:
        """Set values of the status. A new ``phase`` restarts its clock."""
        with self.lock:
            if "phase" in values and values["phase"] != self.status.get("phase"):
                self.since = time.monotonic()
            self.status.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            out = dict(self.status)
            out["prompt"] = self.prompt
            out["phase_s"] = round(time.monotonic() - self.since, 1)
        out["views"] = self.views
        return out

    def request(self, command: str) -> bool:
        """A button of the page: ``start``, ``stop`` or ``rest``.

        ``start`` counts only while the client is idle, ``rest`` not during
        a rest move; the return value says whether the request counts.
        ``stop`` and ``rest`` end the session or the rest move that runs
        (:attr:`interrupt`). :meth:`take` gives the request to the client.
        """
        with self.lock:
            mode = self.status.get("mode", "idle")
            if (command == "start" and mode != "idle") or (
                command == "rest" and mode == "moving"
            ):
                return False
            self.wanted = command
        if command != "start":
            self.interrupt.set()
        self.wake.set()
        return True

    def take(self) -> str | None:
        """The last request of the page that the client has not taken, or None."""
        with self.lock:
            command, self.wanted = self.wanted, None
            self.wake.clear()
        return command

    def wait_command(self, idle: Callable[[], Any] | None = None) -> str:
        """Wait for a request of the page. ``idle`` runs 5 times a second in
        the wait (the readings of the page)."""
        while True:
            command = self.take()
            if command is not None:
                return command
            if idle is not None:
                idle()
            self.wake.wait(0.2)

    def watch(self, robot: Any, hz: float = 15.0) -> None:
        """Publish the frames of the robot's cameras at ``hz`` from a thread.

        LeRobot cameras read in their own thread, so this does not touch the
        motor bus. A robot without cameras (``--fake-robot``) is left alone.
        """
        cams = getattr(robot, "cameras", None)
        if not isinstance(cams, dict):
            return
        readers = {
            v: getattr(cams[v], "read_latest", None)
            or getattr(cams[v], "async_read", None)
            for v in self.views
            if v in cams
        }

        revived = [0.0]  # the time of the last try to connect a camera again

        def loop() -> None:
            while not self.closed.wait(1.0 / hz):
                for view, reader in readers.items():
                    if reader is None:
                        continue
                    try:
                        frame = np.asarray(reader(), np.uint8)
                    except RuntimeError:  # its reader has ended
                        if time.monotonic() - revived[0] > 3.0:
                            revived[0] = time.monotonic()
                            revive_cameras(robot)
                        continue
                    except Exception:
                        continue  # a late frame, a camera that reconnects
                    self.publish(view, fit_frame(frame))

        threading.Thread(target=loop, name="so101-ui-cameras", daemon=True).start()

    def close(self) -> None:
        self.closed.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def _handler(ui: Ui, bind_host: str) -> type[BaseHTTPRequestHandler]:
    """The request handler class of :class:`Ui`."""
    allowed = LOOPBACK_NAMES | ({bind_host} - {"0.0.0.0", "::", ""})

    class Handler(BaseHTTPRequestHandler):
        server_version = "so101-client"
        timeout = 15  # seconds: a request that sends no body does not hold a thread

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: object) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _host_ok(self) -> bool:
            host = self.headers.get("Host") or ""
            name = host[: host.index("]") + 1] if host.startswith("[") else host
            return name.split(":")[0] in allowed or name in allowed

        def do_GET(self) -> None:
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
            path = self.path.split("?", 1)[0]
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status":
                return self._json(200, ui.snapshot())
            for prefix, suffix in (("/frame/", ".jpg"), ("/stream/", ".mjpg")):
                if path.startswith(prefix) and path.endswith(suffix):
                    view = path[len(prefix) : -len(suffix)]
                    if view in ui.views:
                        return (
                            self._frame(view)
                            if suffix == ".jpg"
                            else self._stream(view)
                        )
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _frame(self, view: str) -> None:
            got = ui.jpeg(view)
            if got is None:
                return self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no frame yet"}
                )
            self._send(200, got[1], "image/jpeg")

        def _stream(self, view: str) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            try:
                while not ui.closed.is_set():
                    got = ui.jpeg(view)
                    if got is not None and got[0] != last:
                        last = got[0]
                        head = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(got[1])}\r\n\r\n"
                        self.wfile.write(head.encode() + got[1] + b"\r\n")
                        self.wfile.flush()
                    ui.closed.wait(1.0 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden origin"})
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if ctype != "application/json":
                return self._json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "JSON only"}
                )
            try:
                size = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(size)) if 0 < size <= 4096 else None
            except ValueError:
                body = None
            if not isinstance(body, dict):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "a JSON object"})
            path = self.path.split("?", 1)[0]
            if path == "/prompt":
                try:
                    return self._json(
                        200, {"prompt": ui.set_prompt(body.get("prompt"))}
                    )
                except ValueError as e:
                    return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            if path in ("/start", "/stop", "/rest"):
                if ui.request(path[1:]):
                    return self._json(200, {"requested": path[1:]})
                return self._json(HTTPStatus.CONFLICT, {"error": "the client is busy"})
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


class Waiting:
    """A line every ``every`` seconds until :meth:`done`: a cold server takes
    minutes to answer, and the client prints nothing else in that time."""

    def __init__(self, what: str, every: float = 30.0) -> None:
        self._done = threading.Event()
        start = time.monotonic()

        def loop() -> None:
            while not self._done.wait(every):
                say(f"[client] still {what}: {time.monotonic() - start:.0f} s")

        threading.Thread(target=loop, name="so101-waiting", daemon=True).start()

    def done(self) -> None:
        self._done.set()


@contextlib.contextmanager
def signals_interrupt() -> Iterator[None]:
    """SIGTERM and SIGHUP end the run as Ctrl-C does (a kill, a closed
    terminal), so the arm returns to the rest pose. Main thread only."""

    def handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(ValueError):  # not the main thread
                previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


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
        "--max-chunks",
        type=int,
        default=0,
        help="end the session after N chunks (0 = forever)",
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
    ap.add_argument("--no-rest-move", action="store_true",
                    help="no move to the rest pose at the start and at the end")  # fmt: skip
    ap.add_argument("--rest-pose", type=rest_pose_arg, default=None,
                    help="the rest pose, 'pan,lift,elbow,wrist_flex,wrist_roll' in "
                         "degrees (default 0,-100,95,70,0)")  # fmt: skip
    ap.add_argument("--ui-port", type=int, default=None,
                    help="serve the page (views, prompt, buttons) on this port; 0 = a free port")  # fmt: skip
    ap.add_argument("--autostart", action="store_true",
                    help="with the page: start the first session without the Start button")  # fmt: skip
    ap.add_argument("--ui-host", default="127.0.0.1",
                    help="address of the page; it has no password, keep a loopback address")  # fmt: skip
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


class SessionError(Exception):
    """The server refused the session, answered with an error or went silent."""


def handshake(ws: Any, stop: Callable[[], bool] | None = None) -> dict:
    """Wait for the ``ready`` message; ``stop`` (asked twice a second) ends
    the wait with :class:`StopRun`."""
    while True:
        if stop is None:
            raw = ws.recv()
        else:
            try:
                raw = ws.recv(timeout=0.5)
            except TimeoutError:
                if stop():
                    raise StopRun from None
                continue
        m = loads(raw)
        st = _s(m, "status")
        st = st.decode() if isinstance(st, bytes) else st
        if st == "ready":
            return _s(m, "meta")
        if st == "error":
            raise SessionError(f"server refused the session: {_s(m, 'error')}")
        print(f"[server] {st}: {_s(m, 'message')}", flush=True)


def recv_reply(ws: Any, timeout: float, stop: Callable[[], bool] | None = None) -> Any:
    """The answer to a chunk request. No answer in ``timeout`` seconds raises
    :class:`SessionError`; ``stop`` (asked 4 times a second) raises
    :class:`StopRun`."""
    gone = f"no answer in {timeout:.0f} s (connection lost?): holding and stopping"
    if stop is None:
        try:
            return loads(ws.recv(timeout=timeout))
        except TimeoutError:
            raise SessionError(gone) from None
    deadline = time.monotonic() + timeout
    while True:
        try:
            return loads(ws.recv(timeout=0.25))
        except TimeoutError:
            if stop():
                raise StopRun from None
            if time.monotonic() >= deadline:
                raise SessionError(gone) from None


def _text(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def play_chunk(
    a: argparse.Namespace,
    robot: Any,
    views: list[str],
    hist: collections.deque,
    anchor: dict[str, Any],
    targets: np.ndarray,
    last_cmd: np.ndarray | None,
    chunk: int,
    rtt: float,
    policy_ms: float,
    detail: str,
    log: Any,
    record: dict[str, Any],
    ui: Ui | None = None,
) -> np.ndarray | None:
    """Clamp, print, log and play the ``targets`` of chunk ``chunk``.

    ``anchor`` is the entry of the request, ``last_cmd`` the last command of
    the chunk before (None at the start). ``rtt`` and ``policy_ms`` are the
    round trip and the model time, ``detail`` the transport's part of the
    line, ``record`` its keys of the log record. ``ui`` gets the numbers of
    the chunk and the frames of its entries; its stop and rest requests end
    the chunk before the next row (:class:`StopRun`). Return the last
    command of the chunk, or None in a dry run (the arm did not move).
    """
    if not np.isfinite(targets).all() or targets.shape[1:] != (6,):
        raise SessionError(f"bad chunk {targets.shape}")
    sub = max(1, round(a.control_hz / FPS))
    period = 1.0 / (FPS * sub)
    start = anchor["state"] if last_cmd is None else last_cmd
    safe = clamp_chunk(targets, start, a.max_step_deg, a.max_grip_step)
    print(f"chunk {chunk}: {len(safe)} rows, rtt {rtt:.0f} ms "
          f"(policy {policy_ms:.0f} ms), {detail}, stack {len(hist) + 1}, clamped "
          f"{float(np.abs(safe - targets).max()):.2f}", flush=True)  # fmt: skip
    if ui is not None:
        ui.update(phase="dry run" if a.dry_run else "running", rows=len(safe),
                  rtt_ms=round(rtt), policy_ms=round(policy_ms))  # fmt: skip
    if a.dry_run:
        for i in (0, len(safe) // 2, len(safe) - 1):
            cols = " ".join(
                f"{j}={x:7.2f}" for j, x in zip(JOINTS, safe[i], strict=True)
            )
            print(f"   row {i:2d}: {cols}")
    if log is not None:
        log.write(json.dumps({"chunk": chunk, "rtt_ms": rtt, "anchor": anchor["state"].tolist(),
                              "targets": targets.tolist(), "sent": safe.tolist(), **record}) + "\n")  # fmt: skip
        log.flush()
    prev = play_rows(robot, safe, start, anchor, hist, views, sub, period,
                     dry=a.dry_run, paced=not a.fake_robot, ui=ui,
                     stop=None if ui is None else ui.interrupt.is_set)  # fmt: skip
    return prev if not a.dry_run else None


def run_session(
    a: argparse.Namespace,
    robot: Any,
    views: list[str],
    log: Any,
    ui: Ui | None = None,
) -> None:
    """The session of a praxis serve endpoint (WebSocket): the handshake,
    then request, play and log chunk after chunk (:func:`play_chunk`). It
    returns after ``--max-chunks`` chunks. A problem of the server raises
    :class:`SessionError`; a stop or rest request of the page (``ui``) raises
    :class:`StopRun`."""
    from websockets.sync.client import connect

    stop = ui.interrupt.is_set if ui is not None else None
    prompt = a.prompt
    last_cmd: np.ndarray | None = None
    if ui is not None:
        ui.update(phase="waiting for the server")
    wait = Waiting("waiting for the server (a cold start takes minutes)")
    try:
        with connect(session_url(a, len(views)), additional_headers=auth_headers(a),
                     max_size=16 * 1024 * 1024, open_timeout=a.connect_timeout,
                     close_timeout=5) as ws:  # fmt: skip
            meta = handshake(ws, stop)
            wait.done()
            print(f"[server] ready: {_s(meta, 'arm')} ({_s(meta, 'embodiment')}), views "
                  f"{_s(meta, 'views')}, steps {_s(meta, 'num_steps')}, cfg_video "
                  f"{_s(meta, 'cfg_video_scale')}, context {_s(meta, 'max_context_chunks')}, "
                  f"grip {_s(meta, 'grip_range')}", flush=True)  # fmt: skip
            hist: collections.deque = collections.deque(
                maxlen=max(0, int(_s(meta, "stack_entries")) - 1)
            )
            chunk = 0
            while not a.max_chunks or chunk < a.max_chunks:
                if stop is not None and stop():
                    raise StopRun
                if ui is not None and ui.get_prompt() != prompt:
                    prompt = ui.get_prompt()
                    print(f"[client] prompt: {prompt!r}", flush=True)
                anchor = read_entry(robot, views, ui)
                req = build_request([*hist, anchor], views, prompt, a.jpeg_quality)
                if ui is not None:
                    ui.update(phase="waiting for the chunk", chunk=chunk)
                t0 = time.monotonic()
                ws.send(req)
                rep = recv_reply(ws, a.recv_timeout, stop)
                rtt = (time.monotonic() - t0) * 1e3
                err = _s(rep, "error")
                if err:
                    raise SessionError(f"server error: {_text(err)}")
                targets = np.asarray(_s(rep, "actions"), np.float64)
                timing = _s(rep, "server_timing") or {}
                used = _s(rep, "effective_prompt")
                if ui is not None:
                    ui.update(effective_prompt=_text(used) if used else prompt)
                record = {"prompt": prompt,
                          "server_timing": {k.decode() if isinstance(k, bytes) else k: v
                                            for k, v in dict(timing).items()}}  # fmt: skip
                last_cmd = play_chunk(
                    a, robot, views, hist, anchor, targets, last_cmd, chunk, rtt,
                    float(_s(timing, "policy_ms") or 0),
                    f"request {len(req) / 1e6:.2f} MB", log, record, ui=ui,
                )  # fmt: skip
                chunk += 1
    finally:
        wait.done()


def hold(robot: Any, dry: bool) -> None:
    """Command the reading: the arm stays where it is, with the torque on."""
    send(robot, read_entry(robot, [])["state"], dry)


def serve_page(
    a: argparse.Namespace,
    robot: Any,
    views: list[str],
    ui: Ui,
    rest: np.ndarray,
    hz: float,
    log: Any,
) -> None:
    """The client under the page, until Ctrl-C.

    The client waits for a button. ``Start`` runs a session
    (:func:`run_session`, a new connection each time, so a new context). The
    session ends on ``Stop``, on ``Return to rest``, after ``--max-chunks``
    chunks or on a problem of the server or the connection; the arm then
    holds its position and the page shows why. ``Return to rest`` moves the
    arm to the rest pose (:func:`move_to_rest`); ``Stop`` ends that move too.
    No button ends the client. ``--autostart`` starts the first session
    without the button.
    """
    command = "start" if a.autostart else None
    ui.update(mode="idle", phase="ready: press Start")
    while True:
        if command is None:
            command = ui.wait_command(lambda: read_entry(robot, [], ui))
        ui.interrupt.clear()
        if command == "start":
            ui.update(mode="running", message="")
            note = ""
            try:
                run_session(a, robot, views, log, ui)
                note = f"the session ended after {a.max_chunks} chunks (--max-chunks)"
            except StopRun:
                pass
            except SessionError as e:
                note = str(e)
            except Exception as e:  # the connection or the transport of the session
                note = f"connection: {type(e).__name__}: {e}"
            if note:
                say(f"[client] {note}")
            hold(robot, a.dry_run)
            ui.update(mode="idle", phase="stopped: holding", message=note)
        elif command == "rest" and a.dry_run:
            ui.update(message="dry run: the arm does not move")
        elif command == "rest":
            ui.update(mode="moving", phase="returning to the rest pose", message="")
            try:
                far = rest_check(
                    move_to_rest(robot, rest, hz, stop=ui.interrupt.is_set), rest=rest
                )
                note = "the rest move stopped short: " + ", ".join(far) if far else ""
                ui.update(mode="idle", phase="at rest", message=note)
            except StopRun:
                hold(robot, False)
                ui.update(mode="idle", phase="stopped: holding")
        # The request that ended the session or the move. A stop is done.
        command = ui.take()
        if command == "stop":
            command = None


def shutdown(
    a: argparse.Namespace,
    robot: Any,
    rest: np.ndarray,
    park: bool,
    hz: float,
    ui: Ui | None,
    log: Any,
) -> None:
    """End the client: hold, return to the rest pose (``park``), disconnect.
    LeRobot releases the torque on disconnect unless ``--hold-torque-on-exit``."""
    try:
        if not a.dry_run:
            hold(robot, False)
            if park:
                if ui is not None:
                    ui.update(mode="moving", phase="returning to the rest pose")
                say(
                    "[client] returning to the rest pose (Ctrl-C releases the arm here)"
                )
                try:
                    move_to_rest(robot, rest, hz)
                except KeyboardInterrupt:
                    say("\n[client] rest move interrupted: the arm stays where it is")
                except Exception as e:
                    say(f"[client] rest move failed: {e}")
    finally:
        try:
            robot.disconnect()
        finally:
            if log is not None:
                log.close()
            if ui is not None:
                ui.update(mode="moving", phase="the client ended")
                ui.close()


def main(argv: list[str] | None = None) -> None:
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
    ui = None
    if a.ui_port is not None:
        try:
            ui = Ui(a.ui_host, a.ui_port, a.prompt, views)
        except OSError:  # a busy port: the torque must not stay on
            robot.disconnect()
            raise
        ui.update(dry_run=bool(a.dry_run), mode="moving")
        ui.watch(robot)
        print(f"[client] page: {ui.url}", flush=True)
    rest = rest_vector(a)
    hz = FPS * max(1, round(a.control_hz / FPS))
    # The rest moves need a live arm. The end move is dropped when the start
    # pose is refused: the rest value is then not a pose to push toward.
    park = not a.dry_run and not a.no_rest_move
    log = None
    with signals_interrupt():
        try:
            start_state = None  # None: Stop on the page ended the start move
            if park:
                if ui is not None:
                    ui.update(phase="moving to the rest pose")
                print("[client] moving to the rest pose", flush=True)
                try:
                    start_state = move_to_rest(
                        robot,
                        rest,
                        hz,
                        stop=None if ui is None else ui.interrupt.is_set,
                    )
                except KeyboardInterrupt:
                    park = False  # Ctrl-C in the start move stops the arm
                    raise
                except StopRun:
                    hold(robot, False)
                    ui.take()
                    a.autostart = False  # a Stop in the start move cancels it too
            else:
                start_state = read_entry(robot, [])["state"]
            far = []
            if start_state is not None:  # after a move: against its target
                far = rest_check(start_state, rest=rest if park else None)
            if far:
                msg = (
                    "the arm is not at the folded rest pose of training: "
                    + ", ".join(far)
                )
                if not a.no_rest_check:
                    calibration = (
                        "the calibration (use_degrees=True, 0 = the mid-range)"
                    )
                    if park:
                        fix = (
                            "The rest move stopped short: check that nothing blocks "
                            f"the arm, and {calibration}"
                        )
                    else:
                        fix = f"Move it to the rest pose, or check {calibration}"
                    park = False
                    sys.exit(f"{msg}. {fix}, or pass --no-rest-check")
                print(f"WARNING: {msg}", flush=True)
            # inside the try: a log path that cannot be written still ends with
            # the disconnect of the follower (LeRobot's torque release)
            log = open(a.log_jsonl, "a") if a.log_jsonl else None
            if ui is None:
                try:
                    run_session(a, robot, views, log)
                except SessionError as e:
                    sys.exit(str(e))
            else:
                print(
                    "[client] the page has the buttons; Ctrl-C ends the client",
                    flush=True,
                )
                serve_page(a, robot, views, ui, rest, hz, log)
        except KeyboardInterrupt:
            if ui is not None and ui.snapshot().get("mode") == "moving":
                park = False  # Ctrl-C in a rest move of the page stops the arm
            say("\nCtrl-C: holding the current position and stopping")
        finally:
            shutdown(a, robot, rest, park, hz, ui, log)


if __name__ == "__main__":
    main()
