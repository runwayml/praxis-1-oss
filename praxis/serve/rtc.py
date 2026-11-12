"""Real-time chunking: the action_prefix plan wire and the junction rule.

Under real-time chunking the robot asks for the next chunk while it still
runs the current one. Row ``k`` of an answer is due at the observation time
plus ``k / fps``. The client sends the plan it runs meanwhile as the
``action_prefix`` mapping of the observation (positronic's
``InferenceSchedule``,
https://github.com/Positronic-Robotics/positronic/pull/1036):

* ``actions``: the remaining command rows of the plan, from the row that runs
  at the observation;
* ``chunk_offset_sec``: the offset of that row in its chunk;
* ``observation_offset_sec``: the due time of that row minus the observation
  time (zero or negative).

Plan row ``k`` and answer row ``k`` are due less than one row apart, so the
server treats them as the same slot. The robot runs the first ``P`` rows of
the plan before the answer arrives, where ``P`` covers the delay estimate.
The answer therefore keeps them, and joins the model's rows to the end of
this prefix (:func:`rebase_into_prefix`):

* rows ``k < P`` repeat the prefix;
* from row ``P`` on, the model's motion continues from the prefix end: the
  offset between the prefix end and the model at row ``P - 1`` is added to
  the model rows, and handed back to zero over ``rebase_decay`` rows;
* ``vel_fade`` > 0: over that many rows the velocity fades from the velocity
  of the prefix end to the velocity of the rule, so the seam turns instead
  of switching direction;
* the grippers fade from the continuation of the plan to the model's grips
  over ``grip_rows`` rows (``grip="cont"``), or follow the model
  (``grip="model"``);
* on the bimanual rig, the YAM reach clamp moves targets past a reach
  radius back onto it (:func:`praxis.bimanual.clamp_reach`).

On the single-arm rig the rows are in the frame of the client, the frame the
robot runs, and every row keeps the control law of the session
(``droid_impedance``), the kept prefix rows included.

The rule never changes the sampler. The plan is anchored on the measured
pose at the observation, and the rebase puts no measured-pose term into the
output, so the junctions do not feed back into an oscillation.

Rows are arrays ``(T, arms, 13)``: ``xyz3 | R9 row-major | grip1`` per arm,
the layout of the command dicts.

On the so101 rig the rows are ``(T, 6)`` joint targets (degrees x 5,
``gripper.pos``; :mod:`praxis.so101`). The junction is the same rule with each
joint angle as a linear channel and the gripper as the grip
(:func:`rebase_joints`, :func:`grip_crossfade_joints`): the embedding
:func:`joints_as_arms` puts joint ``j`` on the x axis of arm ``j`` with the
identity rotation, so the pose rule runs unchanged. The reach clamp is a pose
rule and does not apply to joint targets.
"""

from __future__ import annotations

import collections.abc as cabc
import dataclasses
import math

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_PREFIX = "action_prefix"
PREFIX_ACTIONS = "actions"
CHUNK_OFFSET_SEC = "chunk_offset_sec"
OBSERVATION_OFFSET_SEC = "observation_offset_sec"
# The delay estimate is the longest of the last 5 prediction durations.
DELAY_WINDOW = 5
GRIPS: tuple[str, ...] = ("cont", "model")
DEFAULT_GRIP_ROWS = 16
MAX_GRIP_ROWS = 24
MAX_VEL_FADE = 24
# The velocity of the prefix end is the mean of its last two steps.
VEL_FADE_ROWS = 2
ROW_WIDTH = 13
# The chunk rows of the model after the cold start, and of the cold start.
CHUNK_ROWS = 24
FIRST_CHUNK_ROWS = 21


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """The junction settings of a real-time chunking session.

    ``stride`` is the number of rows between two predictions. It must be
    below the 21 rows of the cold-start chunk, so a chunk outlasts the next
    call. The delay estimate is capped at the rows that a 24-row chunk has
    left after the stride (:meth:`max_delay_sec`). During a 21-row chunk the
    plan of the client holds ``21 - stride`` rows, and the kept prefix stops
    at its end (:func:`prefix_rows`), below the cap.

    ``reach_max_m`` holds one reach radius per arm (left, right) of the
    bimanual rig, or is empty for no clamp.

    The defaults are the junction rule of our robot runs with real-time
    chunking: stride 12, rebase decay 24, velocity fade 6, grip continuation
    over 16 rows, and the reach clamp on the bimanual rig.
    """

    stride: int = 12
    rebase_decay: int = 24
    vel_fade: int = 6
    grip: str = "cont"
    grip_rows: int = DEFAULT_GRIP_ROWS
    reach_max_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= int(self.stride) < FIRST_CHUNK_ROWS:
            raise ValueError(
                f"rtc_stride must be 1..{FIRST_CHUNK_ROWS - 1} rows (the cold-start "
                f"chunk must outlast the call stride), got {self.stride}"
            )
        if int(self.rebase_decay) < 0:
            raise ValueError(f"rtc_rebase_decay must be >= 0, got {self.rebase_decay}")
        if not 0 <= int(self.vel_fade) <= MAX_VEL_FADE:
            raise ValueError(
                f"rtc_vel_fade must be 0..{MAX_VEL_FADE} rows, got {self.vel_fade}"
            )
        if int(self.vel_fade) and 0 < int(self.rebase_decay) <= int(self.vel_fade):
            # The fade's offset goes back over the decay rows after the fade.
            raise ValueError(
                f"rtc_rebase_decay ({self.rebase_decay}) must be 0 or more than "
                f"rtc_vel_fade ({self.vel_fade})"
            )
        if self.grip not in GRIPS:
            raise ValueError(f"rtc_grip must be one of {GRIPS}, got {self.grip!r}")
        if not 0 <= int(self.grip_rows) <= MAX_GRIP_ROWS:
            raise ValueError(
                f"rtc_grip_rows must be 0..{MAX_GRIP_ROWS}, got {self.grip_rows}"
            )

    def max_delay_sec(self, fps: float) -> float:
        """Return the cap of the delay estimate: the rows that a 24-row chunk has
        left after the stride. The rows of the plan bound the prefix of a
        21-row chunk."""
        return max(1, CHUNK_ROWS - int(self.stride)) / float(fps)

    def meta(self, fps: float) -> dict[str, object]:
        return {
            "stride": int(self.stride),
            "max_delay_sec": self.max_delay_sec(fps),
            "rebase_decay": int(self.rebase_decay),
            "vel_fade": int(self.vel_fade),
            "grip": self.grip,
            "grip_rows": int(self.grip_rows),
            "reach_max_m": list(self.reach_max_m),
        }


def channels(sides: cabc.Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return the ``(robot_command key, target_grip key)`` of each arm.

    No sides (the single-arm rig) gives the one arm ``robot_command`` and
    ``target_grip``.
    """
    if not sides:
        return (("robot_command", "target_grip"),)
    return tuple((f"robot_command.{s}", f"target_grip.{s}") for s in sides)


def rows_to_array(
    rows: cabc.Sequence[cabc.Mapping], chans: tuple[tuple[str, str], ...]
) -> np.ndarray:
    """Convert command rows to ``(T, arms, 13)`` float64.

    A row that is not a ``cartesian_pos`` command with a finite pose and
    grip raises ``ValueError``.
    """
    out = np.zeros((len(rows), len(chans), ROW_WIDTH), np.float64)
    for t, row in enumerate(rows):
        if not isinstance(row, cabc.Mapping):
            raise ValueError(f"{ACTION_PREFIX} rows must be command mappings")
        for a, (ck, gk) in enumerate(chans):
            cmd = row.get(ck)
            if not isinstance(cmd, cabc.Mapping) or cmd.get("type") != "cartesian_pos":
                raise ValueError(
                    f"{ck}: expected a cartesian_pos command, got {cmd!r:.80}"
                )
            pose = np.asarray(cmd["pose"], np.float64).reshape(-1)
            if pose.shape != (12,) or not np.all(np.isfinite(pose)):
                raise ValueError(f"{ck}: pose must be a finite xyz + row-major R (12,)")
            grip = np.asarray(row.get(gk), np.float64).reshape(-1)
            if grip.shape != (1,) or not np.isfinite(grip[0]):
                raise ValueError(f"{gk}: expected one finite gripper target")
            out[t, a, :12] = pose
            out[t, a, 12] = grip[0]
    return out


def write_rows(
    rows: cabc.Sequence[cabc.Mapping],
    arr: np.ndarray,
    chans: tuple[tuple[str, str], ...],
) -> list[dict]:
    """Return copies of ``rows`` with the pose and grip of each arm from ``arr``.

    The other keys of each row and command stay.
    """
    out: list[dict] = []
    for t, row in enumerate(rows):
        new = dict(row)
        for a, (ck, gk) in enumerate(chans):
            cmd = dict(new[ck])
            cmd["pose"] = arr[t, a, :12].astype(np.float32)
            new[ck] = cmd
            new[gk] = float(arr[t, a, 12])
        out.append(new)
    return out


def strip_timestamps(rows: cabc.Sequence[dict]) -> list[dict]:
    """Return the rows without their ``timestamp``.

    Protocol v2 forwards every key of a row to the robot as a command
    channel. The schedule of the client times the rows.
    """
    return [{k: v for k, v in row.items() if k != "timestamp"} for row in rows]


def _seconds(value: object, name: str, *, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"{name} must be a number of seconds, got {value!r:.40}")
    out = float(value)
    if not math.isfinite(out) or (out < 0 and not signed):
        raise ValueError(f"{name} must be finite seconds, got {out}")
    return out


def read_plan(
    obs: cabc.Mapping,
    chans: tuple[tuple[str, str], ...],
    fps: float,
    max_rows: int = CHUNK_ROWS,
) -> np.ndarray | None:
    """Return the plan of the observation's ``action_prefix`` as ``(T, arms, 13)``.

    Returns None without an ``action_prefix`` (the first call of an episode,
    a replan). The sampled prefix list of older clients raises ``ValueError``:
    the server reads the plan mapping only. A ``chunk_offset_sec`` off the row
    grid raises ``ValueError``.
    """
    rows = read_plan_rows(obs, fps, max_rows)
    return None if rows is None else rows_to_array(rows, chans)


def read_plan_rows(
    obs: cabc.Mapping, fps: float, max_rows: int = CHUNK_ROWS
) -> list | None:
    """Return the command rows of the observation's ``action_prefix`` plan.

    The checks of :func:`read_plan` without the row layout: the plan mapping,
    at most ``max_rows`` rows, the offsets, ``chunk_offset_sec`` on the row
    grid. None without an ``action_prefix``.
    """
    raw = obs.get(ACTION_PREFIX)
    if raw is None:
        return None
    if not isinstance(raw, cabc.Mapping):
        raise ValueError(
            f"{ACTION_PREFIX} must be the plan mapping ({PREFIX_ACTIONS}, "
            f"{CHUNK_OFFSET_SEC}, {OBSERVATION_OFFSET_SEC}); a sampled prefix list "
            "is not served"
        )
    actions = raw.get(PREFIX_ACTIONS)
    if not isinstance(actions, list | tuple):
        raise ValueError(f"{ACTION_PREFIX}.{PREFIX_ACTIONS} must be a list of commands")
    if len(actions) > max_rows:
        raise ValueError(
            f"{ACTION_PREFIX}.{PREFIX_ACTIONS} holds {len(actions)} rows, more than "
            "a chunk"
        )
    offset = _seconds(raw.get(CHUNK_OFFSET_SEC), CHUNK_OFFSET_SEC)
    _seconds(raw.get(OBSERVATION_OFFSET_SEC), OBSERVATION_OFFSET_SEC, signed=True)
    if not math.isclose(offset * float(fps), round(offset * float(fps)), abs_tol=1e-6):
        raise ValueError(
            f"{CHUNK_OFFSET_SEC}={offset} is not a row of the chunk at {float(fps):g} fps"
        )
    return list(actions)


def prefix_rows(estimated_sec: float, fps: float, available: int) -> int:
    """Return the plan rows that run before the answer: rows ``k`` with ``k / fps``
    before ``estimated_sec`` (on the ns grid of the client), at most ``available``."""
    end_ns = round(float(estimated_sec) * 1e9)
    n = 0
    while n < int(available) and round(n * 1e9 / float(fps)) < end_ns:
        n += 1
    return n


@dataclasses.dataclass
class DelayEstimate:
    """The delay estimate of a session: the longest of the last
    :data:`DELAY_WINDOW` prediction durations, 0 before the first."""

    durations: list[float] = dataclasses.field(default_factory=list)

    def update(self, elapsed: cabc.Iterable[float]) -> float:
        self.durations = [*self.durations, *map(float, elapsed)][-DELAY_WINDOW:]
        return max(self.durations, default=0.0)

    def reset(self) -> None:
        self.durations = []


def _rot(row: np.ndarray) -> np.ndarray:
    return row[3:12].reshape(3, 3)


def _ramp(i: int, steps: int) -> float:
    """Return the cosine ramp ``(1 - cos(pi (i + 1) / (steps + 1))) / 2`` in (0, 1)."""
    return 0.5 * (1.0 - math.cos(math.pi * (i + 1) / (steps + 1)))


def rebase_into_prefix(
    new: np.ndarray, prefix: np.ndarray, decay: int = 0, vel_fade: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """Join the model rows ``new`` to the prefix the robot runs meanwhile.

    Rows ``k < P`` are the prefix. Row ``P + i`` is the model row ``P + i``
    plus the junction offset: the xyz offset of the prefix end from the model
    at row ``P - 1``, and the rotation offset applied from the left. With
    ``decay`` > 0 the offset goes back to zero over ``decay`` rows (weight
    ``1 - ramp(i, decay)``). ``vel_fade`` > 0 then fades the velocity
    (:func:`fade_velocity`). The grips of rows from ``P`` are the model's.

    Returns the rows and the offset at row ``P - 1`` (the largest over the
    arms, cm and degrees): the jump that a plain switch to the model makes.
    """
    out = np.array(new, np.float64, copy=True)
    p = min(len(prefix), len(new))
    stats = {"prefix_rows": float(p), "junction_cm": 0.0, "junction_deg": 0.0}
    if p == 0:
        return out, stats
    out[:p] = prefix[:p]
    k = p - 1
    last = len(new) - 1
    for a in range(new.shape[1]):
        end = prefix[k, a]
        dx = end[:3] - new[k, a, :3]
        rv = Rotation.from_matrix(_rot(end) @ _rot(new[k, a]).T).as_rotvec()
        stats["junction_cm"] = max(
            stats["junction_cm"], float(np.linalg.norm(dx)) * 100
        )
        deg = math.degrees(float(np.linalg.norm(rv)))
        stats["junction_deg"] = max(stats["junction_deg"], deg)
        for i in range(len(new) - p):
            src = min(k + 1 + i, last)
            w = 1.0 if decay <= 0 else (1.0 - _ramp(i, decay) if i < decay else 0.0)
            out[p + i, a, :3] = new[src, a, :3] + w * dx
            r_off = Rotation.from_rotvec(w * rv).as_matrix()
            out[p + i, a, 3:12] = (r_off @ _rot(new[src, a])).reshape(-1)
            out[p + i, a, 12] = new[src, a, 12]
    if int(vel_fade) > 0:
        fade_velocity(out, prefix[:p], int(vel_fade), int(decay))
    return out, stats


def fade_velocity(out: np.ndarray, prefix: np.ndarray, rows: int, decay: int) -> None:
    """Fade the xyz velocity from the prefix end's to the rule's, in place.

    From row ``P = len(prefix)``: the step into row ``P + i`` is the step of
    the rule plus ``1 - ramp(i, rows)`` of the difference between the
    velocity of the prefix end (the mean of its last :data:`VEL_FADE_ROWS`
    steps) and that step, for ``i < rows``. The position that this builds goes
    back to zero over the ``decay - rows`` rows after the fade (kept for good
    with ``decay`` 0). Rotations and grips stay. A prefix below 2 rows leaves
    the rows unchanged.
    """
    p = len(prefix)
    if p < 2 or rows <= 0:
        return
    vr = min(VEL_FADE_ROWS, p - 1)
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
    out: np.ndarray, p: int, continuation: np.ndarray | None, rows: int
) -> None:
    """Fade the grips of rows ``p, p + 1, ...`` from the plan to the model, in place.

    Over ``rows`` rows the grip goes from the plan's continuation (the plan
    rows due at the same slots, its last row held) to the grip of the rule
    with the cosine ramp. Without a continuation the grip of the prefix end
    is held. No prefix (``p`` 0) or ``rows`` 0 leaves the grips unchanged.
    """
    if p <= 0 or p >= len(out):
        return
    cont = continuation if continuation is not None and len(continuation) else None
    for i in range(min(int(rows), len(out) - p)):
        old = out[p - 1, :, 12] if cont is None else cont[min(i, len(cont) - 1), :, 12]
        out[p + i, :, 12] = old + _ramp(i, int(rows)) * (out[p + i, :, 12] - old)


# ---------------------------------------------------------------------------
# Joint rows (the so101 rig)
# ---------------------------------------------------------------------------


def joint_rows_to_array(
    rows: cabc.Sequence[cabc.Mapping], key: str, dim: int
) -> np.ndarray:
    """Convert joint command rows to ``(T, dim)`` float64.

    A row without a finite ``key`` of ``dim`` values raises ``ValueError``.
    """
    out = np.zeros((len(rows), dim), np.float64)
    for t, row in enumerate(rows):
        if not isinstance(row, cabc.Mapping):
            raise ValueError(f"{ACTION_PREFIX} rows must be command mappings")
        q = np.asarray(row.get(key), np.float64).reshape(-1)
        if q.shape != (dim,) or not np.all(np.isfinite(q)):
            raise ValueError(f"{key}: expected {dim} finite joint targets")
        out[t] = q
    return out


def write_joint_rows(
    rows: cabc.Sequence[cabc.Mapping], arr: np.ndarray, key: str
) -> list[dict]:
    """Return copies of ``rows`` with ``key`` from ``arr`` (float32). The other
    keys of each row stay."""
    return [{**row, key: arr[t].astype(np.float32)} for t, row in enumerate(rows)]


def joints_as_arms(q: np.ndarray) -> np.ndarray:
    """Embed joint rows ``(T, D)`` as pose rows ``(T, D - 1, 13)``: joint ``j``
    on the x axis of arm ``j``, the identity rotation, the last column (the
    gripper) as the grip of every arm."""
    q = np.asarray(q, np.float64)
    t, d = q.shape
    out = np.zeros((t, d - 1, ROW_WIDTH), np.float64)
    out[:, :, 0] = q[:, :-1]
    out[:, :, 3:12] = np.eye(3).reshape(-1)
    out[:, :, 12] = q[:, -1:]
    return out


def arms_as_joints(rows: np.ndarray) -> np.ndarray:
    """Inverse of :func:`joints_as_arms`: ``(T, D - 1, 13)`` -> ``(T, D)``."""
    return np.concatenate([rows[:, :, 0], rows[:, :1, 12]], axis=1)


def rebase_joints(
    new: np.ndarray, prefix: np.ndarray, decay: int = 0, vel_fade: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """Join the model's joint rows ``new`` ``(T, D)`` to the prefix the robot runs.

    :func:`rebase_into_prefix` on :func:`joints_as_arms`: rows ``k < P`` are
    the prefix; from row ``P`` the offset of the prefix end from the model at
    row ``P - 1`` is added to each joint and handed back over ``decay`` rows,
    and the joint velocity fades over ``vel_fade`` rows. The grips of rows
    from ``P`` are the model's. Returns the rows and the largest offset over
    the joints at row ``P - 1`` (``junction_deg``, the units of the rows).
    """
    out, stats = rebase_into_prefix(
        joints_as_arms(new), joints_as_arms(prefix), decay, vel_fade
    )
    return arms_as_joints(out), {
        "prefix_rows": stats["prefix_rows"],
        "junction_deg": stats["junction_cm"] / 100.0,
    }


def grip_crossfade_joints(
    out: np.ndarray, p: int, continuation: np.ndarray | None, rows: int
) -> None:
    """:func:`grip_crossfade` on the last column of joint rows ``(T, D)``, in place."""
    arms = joints_as_arms(out)
    cont = None if continuation is None else joints_as_arms(continuation)
    grip_crossfade(arms, p, cont, rows)
    out[:, -1] = arms[:, 0, 12]
