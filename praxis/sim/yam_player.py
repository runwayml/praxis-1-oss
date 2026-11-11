"""The client side of a bimanual model session: the trajectory player and its smoothing.

On the robot, the client runs the policy server's chunks with a trajectory
player: each command row is due at its time on the 15 Hz grid, the player
holds the newest due row, and a new chunk replaces the plan from its own
start on. Under real-time chunking (``schedule: rtc``) the client sends a
sample every tick with the plan it runs meanwhile, and the server answers
every ``rtc_stride`` rows with a chunk that it has joined to that plan
(the junction rule of the server). The sim does the same on its own clock:
:class:`TrajectoryPlayer` is the player, and :class:`RealTimeChunking` runs
the real-time rules of the server on the session of the episode, with each
answer arriving a fixed ``latency_s`` after its tick.

:class:`PoseSmoother` smooths the poses that the arms execute with a
Savitzky-Golay filter (``player.smoothing: savgol``, window 7 rows, order 2 by
default), as the hosted demo does. The window is centred on the running row,
over the rows already played and the plan rows still to come, so it adds no
lag; where the plan ends the fit is one-sided. Only execution changes: the
plan, which the junction rule of the next chunk reads, keeps the session's
rows, as the plan that a client sends back to the server does. The grips are
not smoothed.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from praxis import bimanual
from praxis.sim.robot import mat_to_quat, pose12_to_pos_rot, quat_to_mat

SMOOTHING_MODES: tuple[str, ...] = ("savgol", "none")
NS = 1_000_000_000
# The rows the player keeps as history for the smoothing window.
PLAYED_ROWS = 16
# Rows the running row may be held with no plan after it before the history resets.
STALL_ROWS = 1.5
# The due times and the tick times round to whole ns on their own: a row due
# within this of a tick is due at that tick.
TOLERANCE_NS = 1_000


@dataclass
class YamPlayerConfig:
    """The trajectory player of the client: the smoothing of the executed poses.

    ``smoothing`` is ``savgol`` (Savitzky-Golay over ``window`` rows, polynomial
    ``order``) or ``none`` (play the server's rows as they are).
    """

    smoothing: str = "savgol"
    window: int = 7
    order: int = 2

    def __post_init__(self) -> None:
        if self.smoothing not in SMOOTHING_MODES:
            raise ValueError(
                f"player.smoothing must be one of {SMOOTHING_MODES}, got {self.smoothing!r}"
            )
        if int(self.window) < 1 or int(self.window) % 2 == 0:
            raise ValueError(
                f"player.window must be an odd number of rows, got {self.window}"
            )
        if not 0 <= int(self.order) < int(self.window):
            raise ValueError(
                f"player.order must be 0..window-1, got {self.order} (window {self.window})"
            )

    def smoother(self) -> PoseSmoother | None:
        """Return the smoother of the config, or None for ``smoothing: none``."""
        if self.smoothing == "none":
            return None
        return PoseSmoother(int(self.window), int(self.order))


def sg_weights(left: int, right: int, order: int) -> np.ndarray:
    """Return the weights of the samples at offsets ``-left..right`` that evaluate the
    least-squares polynomial of degree ``order`` (capped by the samples) at offset 0."""
    t = np.arange(-left, right + 1, dtype=np.float64)
    p = min(int(order), len(t) - 1)
    vander = t[:, None] ** np.arange(p + 1)[None, :]
    # Row 0 of the pseudo-inverse: the constant term of the fit.
    return np.linalg.pinv(vander)[0]


class PoseSmoother:
    """Savitzky-Golay smoothing of one arm's 12-D poses (xyz + row-major R) around a row."""

    def __init__(self, window: int = 7, order: int = 2) -> None:
        self.window = int(window)
        self.order = int(order)
        self._weights: dict[tuple[int, int], np.ndarray] = {}

    @property
    def half(self) -> int:
        return self.window // 2

    def smooth(self, poses: list[np.ndarray], center: int) -> np.ndarray:
        """Return the smoothed pose of ``poses[center]``.

        Positions take the weights; rotations take them on sign-aligned
        quaternions (renormalized), which matches averaging rotation vectors
        for the few-degree spread of a window. Too few rows for the order:
        the centre row.
        """
        left = min(center, self.half)
        right = min(len(poses) - 1 - center, self.half)
        out = np.asarray(poses[center], np.float64).reshape(12).copy()
        if left + right < self.order:
            return out
        key = (left, right)
        if key not in self._weights:
            self._weights[key] = sg_weights(left, right, self.order)
        w = self._weights[key]
        rows = [
            np.asarray(p, np.float64).reshape(12)
            for p in poses[center - left : center + right + 1]
        ]
        out[:3] = sum(wi * r[:3] for wi, r in zip(w, rows, strict=True))
        q0 = mat_to_quat(pose12_to_pos_rot(out)[1])
        q = np.zeros(4)
        for wi, r in zip(w, rows, strict=True):
            qi = mat_to_quat(pose12_to_pos_rot(r)[1])
            q += wi * (qi if float(qi @ q0) >= 0 else -qi)
        n = float(np.linalg.norm(q))
        if n > 1e-9:
            out[3:] = quat_to_mat(q / n).reshape(9)
        return out


@dataclass
class _Waypoint:
    row: dict
    due_ns: int


@dataclass
class TrajectoryPlayer:
    """The plan of the client and the command of each control tick, on one ns clock.

    ``merge`` puts a chunk into the plan: the first chunk starts when it
    arrives, a later one at its observation (``merge.start: observation`` of
    the server's recipe), and it replaces the plan rows from its start on.
    ``command_at`` returns the newest due row, its poses smoothed when the
    player has a smoother. ``action_prefix`` is the plan the client sends
    with a sample.
    """

    fps: float
    smoother: PoseSmoother | None = None
    plan: list[_Waypoint] = field(default_factory=list)
    current: _Waypoint | None = None
    played: list[_Waypoint] = field(default_factory=list)
    previous: tuple[int, int] | None = None  # (start_ns, end_ns) of the last chunk
    _current_replaced: bool = False
    _stalled: bool = False

    @property
    def row_ns(self) -> float:
        return NS / float(self.fps)

    def _due(self, start_ns: int, i: int) -> int:
        return start_ns + round(i * self.row_ns)

    def merge(self, chunk: list[dict], obs_ns: int, now_ns: int) -> None:
        """Put the answer ``chunk`` to the sample of ``obs_ns`` into the plan at ``now_ns``."""
        start = now_ns if self.previous is None else obs_ns
        end = start + round(len(chunk) * self.row_ns)
        incoming = [_Waypoint(r, self._due(start, i)) for i, r in enumerate(chunk)]
        self.previous = (start, end)
        if start <= now_ns:
            self._advance(now_ns)
            self.plan = []
            dues = [w.due_ns for w in incoming]
            running = max(0, bisect.bisect_right(dues, now_ns + TOLERANCE_NS) - 1)
            incoming = incoming[running:]
            # The running row of the chunk is the row in the slot of the running
            # command (real-time chunking): that command is not history then.
            self._current_replaced = (
                self.current is not None
                and incoming[0].due_ns - self.current.due_ns < self.row_ns / 2
            )
        while self.plan and self.plan[-1].due_ns >= start:
            self.plan.pop()
        self.plan.extend(incoming)

    def _advance(self, now_ns: int) -> None:
        held = 0 if self.current is None else now_ns - self.current.due_ns
        if not self.plan and held > STALL_ROWS * self.row_ns:
            self._stalled = True
        while self.plan and self.plan[0].due_ns <= now_ns + TOLERANCE_NS:
            if self._stalled:
                self.played = []
            elif self.current is not None and not self._current_replaced:
                self.played.append(self.current)
            self._stalled = False
            self._current_replaced = False
            self.current = self.plan.pop(0)
        del self.played[:-PLAYED_ROWS]

    def command_at(self, now_ns: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the poses ``(2, 12)`` and grips ``(2,)`` of the tick at ``now_ns``.

        None before the first chunk arrives.
        """
        self._advance(now_ns)
        if self.current is None:
            return None
        poses, grips = _row_arrays(self.current.row)
        if self.smoother is None:
            return poses, grips
        half = self.smoother.half
        past = self.played[-half:] if half else []
        window = [*past, self.current, *self.plan[:half]]
        rows = [_row_arrays(w.row)[0] for w in window]
        for a in range(poses.shape[0]):
            poses[a] = self.smoother.smooth([r[a] for r in rows], len(past))
        return poses, grips

    def action_prefix(self, obs_ns: int) -> dict:
        """Return the ``action_prefix`` of a sample at ``obs_ns`` (positronic's
        ``InferenceSchedule``): the remaining plan from the row running then."""
        at = obs_ns + TOLERANCE_NS
        dues = [w.due_ns for w in self.plan]
        running0 = bisect.bisect_right(dues, at) - 1
        if running0 >= 0:
            remaining_plan = self.plan[running0:]
        elif self.current is not None:
            remaining_plan = [self.current, *self.plan]
        else:
            remaining_plan = list(self.plan)
        start, end = self.previous if self.previous is not None else (obs_ns, obs_ns)
        dues = [w.due_ns for w in remaining_plan]
        running = max(0, bisect.bisect_right(dues, at) - 1)
        remaining = remaining_plan[running:] if at < end else []
        due = remaining[0].due_ns if remaining else end
        return {
            "actions": [w.row for w in remaining],
            "chunk_offset_sec": round((due - start) * self.fps / NS) / self.fps,
            "observation_offset_sec": (due - obs_ns) / NS,
        }


def _row_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the poses ``(2, 12)`` and grips ``(2,)`` (float64, left, right) of one row."""
    poses = np.stack(
        [
            np.asarray(row[f"robot_command.{s}"]["pose"], np.float64).reshape(12)
            for s in bimanual.SIDES
        ]
    )
    grips = np.asarray(
        [float(row[f"target_grip.{s}"]) for s in bimanual.SIDES], np.float64
    )
    return poses, grips


# ---------------------------------------------------------------------------
# Real-time chunking on the sim clock
#
# The sim does not import the policy server (a release without the server
# deletes praxis/serve), so this is a copy of the rules of its real-time
# sessions: the prediction grid (praxis.serve.periodic.PredictGrid), the delay
# estimate (praxis.serve.rtc.DelayEstimate) and the junction rule
# (praxis.serve.rtc: prefix_rows, rebase_into_prefix, fade_velocity,
# grip_crossfade; praxis.bimanual.clamp_reach). tests/serve/test_sim_rtc_parity.py
# checks the copy against the server code, bit for bit.
# ---------------------------------------------------------------------------

ROW_WIDTH = 13  # xyz3 | R9 row-major | grip1 per arm


@dataclass(frozen=True)
class JunctionConfig:
    """The junction settings of a real-time session: the defaults of the server
    (``praxis.serve.rtc.RTCConfig`` on the bimanual rig)."""

    stride: int = 12
    rebase_decay: int = 24
    vel_fade: int = 6
    grip_rows: int = 16
    reach_max_m: tuple[float, ...] = (bimanual.REACH_MAX_M, bimanual.REACH_MAX_M)

    def max_delay_sec(self, fps: float) -> float:
        """Return the cap of the delay estimate: the rows a 24-row chunk has left after the stride."""
        return max(1, 24 - int(self.stride)) / float(fps)


def rows_to_array(rows: list[dict]) -> np.ndarray:
    """Return command rows as ``(T, 2, 13)`` float64 (left, right)."""
    out = np.zeros((len(rows), len(bimanual.SIDES), ROW_WIDTH), np.float64)
    for t, row in enumerate(rows):
        for a, s in enumerate(bimanual.SIDES):
            out[t, a, :12] = np.asarray(
                row[f"robot_command.{s}"]["pose"], np.float64
            ).reshape(12)
            out[t, a, 12] = float(row[f"target_grip.{s}"])
    return out


def write_rows(rows: list[dict], arr: np.ndarray) -> list[dict]:
    """Return copies of ``rows`` with the pose and grip of each arm from ``arr``."""
    out: list[dict] = []
    for t, row in enumerate(rows):
        new = dict(row)
        for a, s in enumerate(bimanual.SIDES):
            cmd = dict(new[f"robot_command.{s}"])
            cmd["pose"] = arr[t, a, :12].astype(np.float32)
            new[f"robot_command.{s}"] = cmd
            new[f"target_grip.{s}"] = float(arr[t, a, 12])
        out.append(new)
    return out


def prefix_rows(estimated_sec: float, fps: float, available: int) -> int:
    """Return the plan rows that run before the answer: rows ``k`` with ``k / fps``
    before ``estimated_sec`` (on the ns grid of the client), at most ``available``."""
    end_ns = round(float(estimated_sec) * 1e9)
    n = 0
    while n < int(available) and round(n * 1e9 / float(fps)) < end_ns:
        n += 1
    return n


def _rot(row: np.ndarray) -> np.ndarray:
    return row[3:12].reshape(3, 3)


def _ramp(i: int, steps: int) -> float:
    """Return the cosine ramp ``(1 - cos(pi (i + 1) / (steps + 1))) / 2`` in (0, 1)."""
    return 0.5 * (1.0 - math.cos(math.pi * (i + 1) / (steps + 1)))


def rebase_into_prefix(
    new: np.ndarray, prefix: np.ndarray, decay: int = 0, vel_fade: int = 0
) -> np.ndarray:
    """Join the model rows ``new`` to the prefix that runs meanwhile.

    Rows ``k < P`` are the prefix. From row ``P`` on, the model rows carry the
    offset between the prefix end and the model at row ``P - 1`` (xyz, and the
    rotation from the left), handed back to zero over ``decay`` rows; then
    ``vel_fade`` fades the velocity of the seam (:func:`fade_velocity`).
    """
    out = np.array(new, np.float64, copy=True)
    p = min(len(prefix), len(new))
    if p == 0:
        return out
    out[:p] = prefix[:p]
    k = p - 1
    last = len(new) - 1
    for a in range(new.shape[1]):
        end = prefix[k, a]
        dx = end[:3] - new[k, a, :3]
        rv = Rotation.from_matrix(_rot(end) @ _rot(new[k, a]).T).as_rotvec()
        for i in range(len(new) - p):
            src = min(k + 1 + i, last)
            w = 1.0 if decay <= 0 else (1.0 - _ramp(i, decay) if i < decay else 0.0)
            out[p + i, a, :3] = new[src, a, :3] + w * dx
            r_off = Rotation.from_rotvec(w * rv).as_matrix()
            out[p + i, a, 3:12] = (r_off @ _rot(new[src, a])).reshape(-1)
            out[p + i, a, 12] = new[src, a, 12]
    if int(vel_fade) > 0:
        fade_velocity(out, prefix[:p], int(vel_fade), int(decay))
    return out


def fade_velocity(out: np.ndarray, prefix: np.ndarray, rows: int, decay: int) -> None:
    """Fade the xyz velocity from the prefix end's to the rule's over ``rows`` rows, in place."""
    p = len(prefix)
    if p < 2 or rows <= 0:
        return
    vr = min(2, p - 1)
    back = max(int(decay) - rows, 1)
    for a in range(out.shape[1]):
        v_end = (prefix[p - 1, a, :3] - prefix[p - 1 - vr, a, :3]) / vr
        base = out[p:, a, :3].copy()
        prev = out[p - 1, a, :3].copy()
        acc = np.zeros(3)
        for i in range(len(base)):
            step = base[i] - (base[i - 1] if i else prev)
            if i < rows:
                acc += (1.0 - _ramp(i, rows)) * (v_end - step)
                out[p + i, a, :3] = base[i] + acc
                continue
            j = i - rows
            w = 1.0 if decay <= 0 else (1.0 - _ramp(j, back) if j < back else 0.0)
            out[p + i, a, :3] = base[i] + w * acc


def grip_crossfade(
    out: np.ndarray, p: int, continuation: np.ndarray, rows: int
) -> None:
    """Fade the grips of rows ``p, p + 1, ...`` from the plan's continuation to the model's, in place."""
    if p <= 0 or p >= len(out):
        return
    cont = continuation if len(continuation) else None
    for i in range(min(int(rows), len(out) - p)):
        old = out[p - 1, :, 12] if cont is None else cont[min(i, len(cont) - 1), :, 12]
        out[p + i, :, 12] = old + _ramp(i, int(rows)) * (out[p + i, :, 12] - old)


@dataclass
class _InFlight:
    obs_ns: int
    arrive_ns: int
    rows: list[dict]


class RealTimeChunking:
    """Real-time chunking of the policy server, on the sim clock.

    ``step(stack, executed_rows)`` runs one prediction of the session on the
    frame stack of the tick and returns its command rows. The first
    prediction runs at tick 0 and plays from its arrival; the next one runs
    ``stride`` rows after that, then one every ``stride`` rows. Each answer
    arrives ``latency_s`` after its tick, whatever the model took on the GPU,
    so a run does not depend on the speed of the machine. Each chunk is
    joined to the plan that runs meanwhile, as the server does: the first
    ``P`` rows (those that run before the answer arrives) repeat the plan,
    the rest continue from its end. ``hold`` is the command before the first
    chunk arrives.
    """

    def __init__(
        self,
        step: Callable[[Callable[[], dict], int], list[dict]],
        player: TrajectoryPlayer,
        latency_s: float,
        hold: tuple[np.ndarray, np.ndarray],
        junction: JunctionConfig | None = None,
    ) -> None:
        self.junction = junction or JunctionConfig()
        cap = self.junction.max_delay_sec(player.fps)
        if not (math.isfinite(latency_s) and 0 <= latency_s <= cap):
            raise ValueError(
                f"session.latency_s must be 0..{cap:g} s (the rows a chunk has left "
                f"after the stride), got {latency_s}"
            )
        self.step = step
        self.player = player
        self.latency_s = float(latency_s)
        self.latency_ns = round(self.latency_s * NS)
        self.hold = hold
        self.in_flight: list[_InFlight] = []
        self.next_index: int | None = None
        self.last_index: int | None = None
        self.predictions = 0
        self.prefix_rows: list[int] = []

    def _deliver(self, now_ns: int) -> None:
        while self.in_flight and self.in_flight[0].arrive_ns <= now_ns:
            call = self.in_flight.pop(0)
            self.player.merge(call.rows, call.obs_ns, call.arrive_ns)

    def _join(self, rows: list[dict], now_ns: int, delay_sec: float) -> list[dict]:
        new = rows_to_array(rows)
        plan_rows = self.player.action_prefix(now_ns)["actions"]
        plan = rows_to_array(plan_rows) if plan_rows else new[:0]
        j = self.junction
        p = min(prefix_rows(delay_sec, self.player.fps, len(plan)), len(new))
        self.prefix_rows.append(p)
        out = rebase_into_prefix(new, plan[:p], j.rebase_decay, j.vel_fade)
        if p:
            grip_crossfade(
                out, p, plan[p:] if p < len(plan) else plan[-1:], j.grip_rows
            )
        if j.reach_max_m:
            bimanual.clamp_reach(out, p, j.reach_max_m)
        return write_rows(rows, out)

    def tick(
        self, index: int, stack: Callable[[], dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run tick ``index``; ``stack()`` returns the frame stack of the tick."""
        now = round(index * self.player.row_ns)
        self._deliver(now)
        if self.next_index is None or index >= self.next_index:
            first = self.last_index is None
            executed = 0 if first else index - int(self.last_index)
            rows = self.step(stack, executed)
            # The server's delay estimate: no duration before the first answer.
            delay = (
                0.0
                if first
                else min(self.latency_s, self.junction.max_delay_sec(self.player.fps))
            )
            self.in_flight.append(
                _InFlight(now, now + self.latency_ns, self._join(rows, now, delay))
            )
            stride = int(self.junction.stride)
            # The first chunk plays from its arrival, the next ones from their sample.
            wait = round(self.latency_s * self.player.fps) if first else 0
            self.next_index = index + stride + wait
            self.last_index = index
            self.predictions += 1
            self._deliver(now)
        cmd = self.player.command_at(now)
        return self.hold if cmd is None else cmd
