"""A scripted bimanual oracle for the tasks of :mod:`praxis_sim.yam_tasks`.

The oracle reads the object poses from the simulation. Each arm runs its own
program: a list of steps (pick, place, turn, wait, home). A step is a Python
generator that yields one target per 15 Hz tick: the grasp-site position, the
grasp-site rotation and the grip. The steps read the live object poses when
they start, so an object that moved is still found. Shared regions (a bowl
in the middle) have a lock, so only one arm is in the region at a time.

The oracle proves that a task is solvable with the arms, the grippers and the
time limit of the rig. It sends the same command rows as the model: absolute
grasp-site poses ``[xyz, R row-major]`` and grips (0 = open, 1 = closed).

Grasp rotation: the tool axis (grasp-site z) points down, tilted
``pitch_deg`` away from the arm base (the real episodes grasp with a tilt of
about 20 deg). The fingers open along the grasp-site y axis, which stays
horizontal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from praxis_sim.robot import pos_rot_to_pose12
from praxis_sim.scene import CONTROL_HZ
from praxis_sim.scripted import _slerp, min_jerk
from praxis_sim.yam_scene import MOUNTS, SIDES

if TYPE_CHECKING:
    from praxis_sim.yam_env import YamEnv

Target = tuple[np.ndarray, np.ndarray, float]
Step = Callable[["ArmState"], Generator[Target, None, None]]


@dataclass
class OracleConfig:
    """Settings of the scripted oracle."""

    speed: float = 0.3  # metres per second of the free moves
    min_move_s: float = 0.6
    approach: float = 0.08  # metres above the grasp point
    lift: float = 0.1  # metres
    pitch_deg: float = 20.0  # tool tilt from straight down
    close_s: float = 0.5
    open_s: float = 0.4
    grip_rate: float = 0.2  # grip change per tick
    descend_s: float = 0.8


@dataclass
class ArmState:
    """The last target of one arm."""

    pos: np.ndarray
    rot: np.ndarray
    grip: float = 0.0
    held_offset: np.ndarray | None = None  # object origin in the grasp-site frame


def flat(v: np.ndarray) -> np.ndarray:
    """Return the unit horizontal part of ``v``."""
    w = np.array([v[0], v[1], 0.0])
    n = float(np.linalg.norm(w))
    return w / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def tool_rotation(finger: np.ndarray, tilt_dir: np.ndarray, pitch: float) -> np.ndarray:
    """Return the grasp-site rotation: fingers along ``finger``, tool tilted ``pitch``.

    ``finger`` is made horizontal. The tool tilts in the vertical plane
    orthogonal to the fingers, to the side of ``tilt_dir``.
    """
    y = flat(finger)
    h = np.cross(np.array([0.0, 0.0, 1.0]), y)  # horizontal, orthogonal to y
    if np.dot(h, tilt_dir) < 0:
        h = -h
    t = math.cos(pitch) * np.array([0.0, 0.0, -1.0]) + math.sin(pitch) * h
    t /= np.linalg.norm(t)
    x = np.cross(y, t)
    return np.stack([x, y, t], axis=1)


class Plan:
    """The programs of both arms and the locks of the shared regions."""

    def __init__(self, env: YamEnv, cfg: OracleConfig | None = None) -> None:
        self.env = env
        self.cfg = cfg or OracleConfig()
        self.steps: dict[str, list[Step]] = {s: [] for s in SIDES}
        self.locks: dict[str, str] = {}
        self.state: dict[str, ArmState] = {}
        self.home_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s, arm in env.arms.items():
            p, r = arm.site_pose()
            self.state[s] = ArmState(p, r)
            self.home_pose[s] = (p, r)
        self._runs: dict[str, Iterator[Target]] = {}

    # -- helpers -------------------------------------------------------------

    @property
    def data(self) -> mujoco.MjData:
        return self.env.data

    def lock_free(self, name: str) -> bool:
        return name not in self.locks

    def base(self, side: str) -> np.ndarray:
        return np.asarray(MOUNTS[side], dtype=np.float64)

    def grasp_rot(
        self,
        side: str,
        point: np.ndarray,
        finger: np.ndarray | None,
        pitch_deg: float | None = None,
        tilt: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the grasp rotation at ``point`` (the sign of the fingers nearest to home)."""
        pitch = math.radians(self.cfg.pitch_deg if pitch_deg is None else pitch_deg)
        out = flat(point - self.base(side))
        home_rot = self.home_pose[side][1]
        if finger is None:
            finger = np.array([-out[1], out[0], 0.0])
        tilt_dir = out if tilt is None else tilt
        best = None
        for sgn in (1.0, -1.0):
            r = tool_rotation(sgn * finger, tilt_dir, pitch)
            score = float(
                np.dot(r[:, 0], home_rot[:, 0]) + np.dot(r[:, 1], home_rot[:, 1])
            )
            if best is None or score > best[0]:
                best = (score, r)
        assert best is not None
        return best[1]

    def move(
        self,
        st: ArmState,
        pos: np.ndarray,
        rot: np.ndarray | None = None,
        grip: float | None = None,
        seconds: float | None = None,
    ) -> Generator[Target, None, None]:
        """Move to a pose along a min-jerk line (rotation by slerp)."""
        c = self.cfg
        rot = st.rot if rot is None else rot
        grip = st.grip if grip is None else grip
        if seconds is None:
            dist = float(np.linalg.norm(pos - st.pos))
            ang = math.acos(float(np.clip((np.trace(rot @ st.rot.T) - 1) / 2, -1, 1)))
            seconds = max(c.min_move_s, dist / c.speed, ang / 1.6)
        n = max(1, round(seconds * CONTROL_HZ))
        p0, r0, g0 = st.pos.copy(), st.rot.copy(), st.grip
        s = min_jerk(np.arange(1, n + 1) / n)
        for k in range(n):
            st.pos = p0 + (pos - p0) * s[k]
            st.rot = _slerp(r0, rot, float(s[k]))
            st.grip = float(
                g0 + np.clip(grip - g0, -c.grip_rate * (k + 1), c.grip_rate * (k + 1))
            )
            yield st.pos.copy(), st.rot.copy(), st.grip

    def hold(
        self, st: ArmState, seconds: float, grip: float | None = None
    ) -> Generator[Target, None, None]:
        """Stay at the pose for ``seconds``, ramping the grip."""
        yield from self.move(st, st.pos, st.rot, grip, seconds)

    def _site_and_item(
        self, side: str, item: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, r = self.env.arms[side].site_pose()
        return p, r, self.env.task.pos(self.data, item)

    def reachable(self, side: str, pos: np.ndarray, rot: np.ndarray) -> bool:
        """Return ``True`` when the inverse kinematics reaches the pose from home."""
        arm = self.env.arms[side]
        return not arm.ik.solve(pos, rot, arm.home).miss

    # -- steps ---------------------------------------------------------------

    def pick(
        self,
        side: str,
        item: str,
        point: np.ndarray,
        finger_axis_perp: np.ndarray | None = None,
        avoid_axis: np.ndarray | None = None,
        live: bool = False,
        lock: str | None = None,
        pitch_deg: float | None = None,
        lift: float | None = None,
        tilt: np.ndarray | None = None,
    ) -> None:
        """Grasp ``item`` at ``point`` (fingers across ``finger_axis_perp``) and lift it."""
        c = self.cfg
        task = self.env.task
        offset = np.asarray(point, dtype=np.float64) - task.pos(self.data, item)
        axis = finger_axis_perp if finger_axis_perp is not None else avoid_axis
        axis_local = None
        if axis is not None:
            axis_local = task.rot(self.data, item).T @ np.asarray(
                axis, dtype=np.float64
            )

        def step(st: ArmState) -> Generator[Target, None, None]:
            if lock is not None:
                while not self.lock_free(lock):
                    yield from self.hold(st, 1.0 / CONTROL_HZ)
                self.locks[lock] = side
            p_item = task.pos(self.data, item)
            pt = p_item + offset if live else np.asarray(point, dtype=np.float64)
            finger = None
            if axis_local is not None:
                a = task.rot(self.data, item) @ axis_local
                finger = np.array([-a[1], a[0], 0.0])
            rot = self.grasp_rot(side, pt, finger, pitch_deg, tilt)
            above = pt + np.array([0.0, 0.0, c.approach])
            yield from self.move(st, above, rot, 0.0)
            yield from self.move(st, pt, rot, 0.0, c.descend_s)
            yield from self.hold(st, c.close_s, 1.0)
            p, r, it = self._site_and_item(side, item)
            st.held_offset = r.T @ (it - p)
            up = st.pos + np.array([0.0, 0.0, c.lift if lift is None else lift])
            yield from self.move(st, up, rot, 1.0, max(0.6, (lift or c.lift) / 0.15))
            if lock is not None:
                self.locks.pop(lock, None)

        self.steps[side].append(step)

    def place(
        self,
        side: str,
        item: str,
        target: Callable[[], np.ndarray],
        release: bool = True,
        yaw: Callable[[], float] | None = None,
        lock: str | None = None,
        carry_z: float | None = None,
        descend: bool = True,
    ) -> None:
        """Carry the held ``item`` so that its origin goes to ``target()``, then let go.

        ``yaw`` sets the yaw of the finger axis at the target (the item yaw
        follows). ``descend=False`` drops the item from above the target
        (``target`` is the drop point).
        """
        c = self.cfg

        def step(st: ArmState) -> Generator[Target, None, None]:
            if lock is not None:
                while not self.lock_free(lock):
                    yield from self.hold(st, 1.0 / CONTROL_HZ)
                self.locks[lock] = side
            off = st.held_offset if st.held_offset is not None else np.zeros(3)
            tgt = target()
            rot = st.rot
            if yaw is not None:
                y = yaw()
                finger = np.array([-math.sin(y), math.cos(y), 0.0])
                rot = self.grasp_rot(side, tgt, finger, None)
            site = tgt - rot @ off
            z_over = max(site[2] + 0.06, carry_z if carry_z is not None else 0.0)
            over = np.array([site[0], site[1], z_over])
            if carry_z is not None and st.pos[2] < z_over:
                # Climb before the carry so the held item clears the container walls.
                lift = np.array([st.pos[0], st.pos[1], z_over])
                yield from self.move(st, lift, st.rot, 1.0)
            yield from self.move(st, over, rot, 1.0)
            if descend:
                tgt = target()
                site = tgt - rot @ off
                yield from self.move(st, site, rot, 1.0, c.descend_s)
            if release:
                yield from self.hold(st, c.open_s, 0.0)
                st.held_offset = None
                back = st.pos - st.rot[:, 2] * 0.03 + np.array([0.0, 0.0, 0.07])
                yield from self.move(st, back, st.rot, 0.0, 0.6)
            if lock is not None:
                self.locks.pop(lock, None)

        self.steps[side].append(step)

    def turn_place(
        self, side: str, item: str, direction: np.ndarray, below: float
    ) -> None:
        """Turn the held item 90 deg about the horizontal axis ``direction x z``, set it down.

        The turn takes ``direction`` (horizontal) to +z: a mug that lies with
        its opening along ``direction`` ends up right side up. ``below`` is
        the distance from the grasp site down to the bottom of the item after
        the turn. The item is let go 5 mm above the table.
        """
        c = self.cfg

        def step(st: ArmState) -> Generator[Target, None, None]:
            d = flat(direction)
            axis = np.cross(d, np.array([0.0, 0.0, 1.0]))
            turn = np.zeros(9)
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, axis, math.pi / 2)
            mujoco.mju_quat2Mat(turn, q)
            rot = turn.reshape(3, 3) @ st.rot
            up = st.pos + np.array([0.0, 0.0, 0.02])
            yield from self.move(st, up, rot, 1.0, 1.8)
            down = np.array([st.pos[0], st.pos[1], self.env.task.z0 + below + 0.005])
            yield from self.move(st, down, rot, 1.0, 1.0)
            yield from self.hold(st, c.open_s + 0.2, 0.0)
            back = st.pos - rot[:, 2] * 0.06 + np.array([0.0, 0.0, 0.05])
            yield from self.move(st, back, rot, 0.0, 0.8)

        self.steps[side].append(step)

    def wait(
        self, side: str, cond: Callable[[], bool], timeout_s: float = 30.0
    ) -> None:
        """Hold until ``cond()`` is true."""

        def step(st: ArmState) -> Generator[Target, None, None]:
            n = 0
            while not cond() and n < timeout_s * CONTROL_HZ:
                n += 1
                yield from self.hold(st, 1.0 / CONTROL_HZ)

        self.steps[side].append(step)

    def home(self, side: str) -> None:
        """Go back to the start pose with the gripper open."""

        def step(st: ArmState) -> Generator[Target, None, None]:
            p, r = self.home_pose[side]
            yield from self.move(st, p, r, 0.0)

        self.steps[side].append(step)

    # -- execution -----------------------------------------------------------

    def _program(self, side: str) -> Iterator[Target]:
        st = self.state[side]
        for step in self.steps[side]:
            yield from step(st)
        while True:
            yield st.pos.copy(), st.rot.copy(), st.grip

    def next_row(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the next command row: poses ``(2, 12)`` and grips ``(2,)`` (left, right)."""
        if not self._runs:
            self._runs = {s: self._program(s) for s in SIDES}
        poses = np.zeros((2, 12), dtype=np.float32)
        grips = np.zeros(2, dtype=np.float32)
        for k, s in enumerate(SIDES):
            p, r, g = next(self._runs[s])
            poses[k] = pos_rot_to_pose12(p, r)
            grips[k] = g
        return poses, grips


@dataclass
class YamOracle:
    """The scripted oracle as a policy: plan at reset, one row per call."""

    cfg: OracleConfig = field(default_factory=OracleConfig)
    name: str = "oracle"
    plan: Plan | None = None

    def reset(self, env: YamEnv) -> None:
        self.plan = Plan(env, self.cfg)
        task = env.task
        task.live = env.data  # live poses for the task callbacks
        task.oracle(self.plan, env.data)

    def next_row(self) -> tuple[np.ndarray, np.ndarray]:
        if self.plan is None:
            raise RuntimeError("call reset before next_row")
        return self.plan.next_row()


def run_oracle(
    env: YamEnv,
    seed: int,
    cfg: OracleConfig | None = None,
    stop_on_success: bool = True,
    after_success_s: float = 0.5,
    on_tick: Callable[[YamEnv], None] | None = None,
) -> dict[str, float | int | bool | None]:
    """Run one oracle episode (no rendering needed) and return the episode summary."""
    env.reset(seed)
    oracle = YamOracle(cfg or OracleConfig())
    oracle.reset(env)
    linger = round(after_success_s * CONTROL_HZ)
    success_at = None
    while env.tick_count < env.max_ticks:
        poses, grips = oracle.next_row()
        env.step(poses, grips)
        if on_tick is not None:
            on_tick(env)
        if env.task.status.success and success_at is None:
            success_at = env.tick_count
        if (
            stop_on_success
            and success_at is not None
            and env.tick_count - success_at >= linger
        ):
            break
    return env.summary()
