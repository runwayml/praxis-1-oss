"""The bimanual YAM environment: two arms, one 15 Hz step per command row, observe.

Use it like this::

    env = YamEnv(YamEnvConfig(task=YamTaskConfig(name="screwdriver_in_bin")))
    obs = env.reset(seed=0)
    for poses, grips in rows:            # poses (2, 12), grips (2,): left, right
        obs = env.step(poses, grips)     # one row = one tick of 1/15 s
    env.close()

Policy convention of each arm (the bimanual checkpoint, see
:mod:`praxis.bimanual`):

* The pose is the ``grasp_site`` pose ``[x, y, z, qw, qx, qy, qz]`` in the
  world frame of :mod:`praxis_sim.yam_scene` (the frame of the training
  data). A command pose is ``[xyz, R row-major 9]``.
* The grip is 0 when the gripper is open and 1 when it is closed. The
  observed grip is ``GRIP_OBS_MAX * (1 - finger / FINGER_OPEN)``: an empty
  closed gripper reads 0.95, as on the real rig. A commanded grip ``g`` sets
  the finger target ``FINGER_OPEN * (1 - g / GRIP_OBS_MAX)``; ``g = 1``
  squeezes 2.5 mm past closed.

One tick solves the inverse kinematics of the row of each arm once (damped
least squares, :class:`praxis_sim.robot.DlsIk`). Then the joint targets of
both arms ramp from the previous solution to the new one over the physics
substeps, with a velocity feed-forward, as in :class:`praxis_sim.robot.SimArm`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from praxis_sim.robot import (
    Cameras,
    DlsIk,
    IkConfig,
    IkResult,
    pose7,
    pose12_to_pos_rot,
)
from praxis_sim.scene import CONTROL_HZ
from praxis_sim.tasks import TaskStatus
from praxis_sim.yam_scene import (
    ARM_JOINT_NAMES,
    FINGER_OPEN,
    GRIP_OBS_MAX,
    HOME,
    MODEL_VIEWS,
    SIDES,
    TABLE_GEOM,
    YamSceneConfig,
    build_model,
    joint,
)
from praxis_sim.yam_tasks import YamTask, YamTaskConfig, make_yam_task

FINGER_CTRL_MIN = -0.005  # metres past closed (squeeze)
PLACE_TRIES = 50  # start-state draws until no object overlaps another geom


@dataclass
class YamEnvConfig:
    """Settings of the bimanual environment."""

    scene: YamSceneConfig = field(default_factory=YamSceneConfig)
    task: YamTaskConfig = field(default_factory=YamTaskConfig)
    ik: IkConfig = field(default_factory=lambda: IkConfig(null_gain=0.02, max_iters=80))
    home_noise: float = 0.03  # uniform joint noise at reset, radians
    settle_s: float = 1.0  # physics time before the first observation
    feedforward: bool = True
    render: bool = True


@dataclass
class YamObservation:
    """One observation of the bimanual environment."""

    ee_pose: dict[str, np.ndarray]  # side -> (7,) float32 [xyz, qw, qx, qy, qz]
    grip: dict[str, float]  # side -> 0 = open, 1 = closed
    views: dict[str, np.ndarray]  # top, wrist_left, wrist_right (uint8 HxWx3)
    tick: int
    status: TaskStatus
    ik: dict[str, IkResult | None] = field(default_factory=dict)


class YamArm:
    """One simulated YAM arm: state readout, inverse kinematics and gripper map."""

    def __init__(
        self,
        side: str,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ik: IkConfig | None = None,
    ) -> None:
        self.side = side
        self.model = model
        self.data = data
        joints = tuple(joint(side, j) for j in ARM_JOINT_NAMES)
        self.home = np.asarray(HOME[side], dtype=np.float64)
        self.ik = DlsIk(model, joint(side, "grasp_site"), self.home, ik, joints)
        self.qpos_adr = self.ik.qpos_adr
        self.dof_adr = self.ik.dof_adr
        self.act_ids = np.array([model.actuator(j).id for j in joints])
        self.grip_act = model.actuator(joint(side, "gripper")).id
        self.finger_adr = [
            model.joint(joint(side, f)).qposadr[0]
            for f in ("left_finger", "right_finger")
        ]
        self.finger_dof = [
            model.joint(joint(side, f)).dofadr[0]
            for f in ("left_finger", "right_finger")
        ]
        kp = model.actuator_gainprm[self.act_ids, 0]
        kv = -model.actuator_biasprm[self.act_ids, 2]
        self.kv_over_kp = kv / kp
        self.site = model.site(joint(side, "grasp_site")).id
        self.q_cmd = self.home.copy()
        self.grip_cmd = 0.0
        self.ik_misses = 0
        self.bad_rows = 0

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the world pose of the grasp site."""
        d = self.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def ee_pose(self) -> np.ndarray:
        """Return the grasp-site pose ``[xyz, qw, qx, qy, qz]`` (float32)."""
        return pose7(*self.site_pose())

    def finger(self) -> float:
        """Return the finger opening (metres from closed, one finger)."""
        return float(self.data.qpos[self.finger_adr[0]])

    def grip(self) -> float:
        """Return the observed grip: 0 = open, 0.95 = closed without an object."""
        g = GRIP_OBS_MAX * (1.0 - self.finger() / FINGER_OPEN)
        return float(np.clip(g, 0.0, 1.0))

    @staticmethod
    def finger_target(grip: float) -> float:
        """Return the finger target of a commanded grip (see the module doc)."""
        f = FINGER_OPEN * (1.0 - float(grip) / GRIP_OBS_MAX)
        return float(np.clip(f, FINGER_CTRL_MIN, FINGER_OPEN))

    def reset(self, q: np.ndarray) -> None:
        """Put the arm at the joints ``q``, at rest, gripper open."""
        d = self.data
        d.qpos[self.qpos_adr] = q
        d.qvel[self.dof_adr] = 0.0
        for adr, dof in zip(self.finger_adr, self.finger_dof, strict=True):
            d.qpos[adr] = FINGER_OPEN
            d.qvel[dof] = 0.0
        d.ctrl[self.act_ids] = q
        d.ctrl[self.grip_act] = FINGER_OPEN
        self.q_cmd = np.asarray(q, dtype=np.float64).copy()
        self.grip_cmd = 0.0
        self.ik_misses = 0
        self.bad_rows = 0

    def plan(
        self, pose12: np.ndarray, grip: float
    ) -> tuple[np.ndarray, IkResult | None]:
        """Solve one command row. Return the new joint target and the IK result."""
        pose12 = np.asarray(pose12, dtype=np.float64).reshape(12)
        g = float(grip)
        if math.isfinite(g):
            self.grip_cmd = float(np.clip(g, 0.0, 1.0))
        if not np.all(np.isfinite(pose12)):
            self.bad_rows += 1
            return self.q_cmd.copy(), None
        pos, rot = pose12_to_pos_rot(pose12)
        result = self.ik.solve(pos, rot, self.q_cmd)
        if result.miss:
            # The solver can stop in a local minimum at a joint limit. Try
            # again from the home posture and keep the better solution.
            again = self.ik.solve(pos, rot, self.home)
            if (
                again.pos_err + 0.1 * again.rot_err
                < result.pos_err + 0.1 * result.rot_err
            ):
                result = again
        self.ik_misses += int(result.miss)
        return result.q, result


class YamEnv:
    """The simulated bimanual rig and one task.

    The renderer of an env works only on the thread that created the env, and
    :meth:`reset` writes the body positions of the model. Use one env per
    thread and per session; separate envs on separate threads render at the
    same time.
    """

    def __init__(self, cfg: YamEnvConfig | None = None) -> None:
        self.cfg = cfg or YamEnvConfig()
        self.task: YamTask = make_yam_task(self.cfg.task, self.cfg.scene.table_top_z)
        self.model = build_model(self.cfg.scene, self.task.build)
        self.data = mujoco.MjData(self.model)
        self.arms = {s: YamArm(s, self.model, self.data, self.cfg.ik) for s in SIDES}
        self.task.bind(self.model, self.arms)
        self.substeps = round(1.0 / (CONTROL_HZ * self.model.opt.timestep))
        table = self.model.geom(TABLE_GEOM)
        center, half = table.pos[:2].copy(), table.size[:2].copy()
        self.table_xy = (center - half, center + half)
        self.cameras: Cameras | None = None
        if self.cfg.render:
            self.cameras = Cameras(self.model, self.cfg.scene, MODEL_VIEWS)
        self.tick_count = 0
        self.timing = {"physics_s": 0.0, "render_s": 0.0, "ticks": 0}

    @property
    def prompt(self) -> str:
        """Return the prompt of the task."""
        return self.task.prompt

    @property
    def max_ticks(self) -> int:
        """Return the time limit of an episode in ticks."""
        return self.task.max_ticks

    def reset(self, seed: int = 0) -> YamObservation:
        """Start a new episode. The seed sets the object placement and the joint noise."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        for side, arm in self.arms.items():
            q = np.asarray(HOME[side], dtype=np.float64)
            if self.cfg.home_noise > 0:
                q = q + rng.uniform(-self.cfg.home_noise, self.cfg.home_noise, 6)
            arm.reset(q)
        for _ in range(PLACE_TRIES):
            self.task.place(rng, self.data)
            mujoco.mj_forward(self.model, self.data)
            if not self.start_overlap():
                break
        self.settle(self.cfg.settle_s)
        self.task.start(self.model, self.data)
        self.tick_count = 0
        return self.observe()

    def start_overlap(self, depth: float = 0.002) -> bool:
        """Return ``True`` when an object of the task goes into another geom.

        Contacts with the table count only when they are deeper than ``depth``
        too. :meth:`reset` draws the start state again (same generator) until
        no object overlaps.
        """
        d = self.data
        items = set()
        for it in self.task.items.values():
            items |= it.geoms
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in items or c.geom2 in items) and c.dist < -depth:
                return True
        return False

    def settle(self, seconds: float) -> None:
        """Run the physics for ``seconds`` with the present targets."""
        for arm in self.arms.values():
            self.data.ctrl[arm.act_ids] = arm.q_cmd
            self.data.ctrl[arm.grip_act] = arm.finger_target(arm.grip_cmd)
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def step(self, poses: np.ndarray, grips: np.ndarray) -> YamObservation:
        """Execute one command row of both arms (left, right) for one tick, then observe."""
        t0 = time.monotonic()
        poses = np.asarray(poses, dtype=np.float64).reshape(2, 12)
        grips = np.asarray(grips, dtype=np.float64).reshape(2)
        prev: dict[str, np.ndarray] = {}
        new: dict[str, np.ndarray] = {}
        results: dict[str, IkResult | None] = {}
        for k, (side, arm) in enumerate(self.arms.items()):
            prev[side] = arm.q_cmd.copy()
            new[side], results[side] = arm.plan(poses[k], float(grips[k]))
        self._run_substeps(prev, new)
        for side, arm in self.arms.items():
            arm.q_cmd = new[side]
        self.tick_count += 1
        self.task.update(self.data, self.table_xy)
        self.timing["physics_s"] += time.monotonic() - t0
        self.timing["ticks"] += 1
        obs = self.observe()
        obs.ik = results
        return obs

    def _run_substeps(
        self, prev: dict[str, np.ndarray], new: dict[str, np.ndarray]
    ) -> None:
        m, d, n = self.model, self.data, self.substeps
        ff: dict[str, np.ndarray] = {}
        for side, arm in self.arms.items():
            dq = new[side] - prev[side]
            ff[side] = (
                arm.kv_over_kp * dq * CONTROL_HZ
                if self.cfg.feedforward
                else np.zeros(6)
            )
            d.ctrl[arm.grip_act] = arm.finger_target(arm.grip_cmd)
        for i in range(n):
            s = (i + 1) / n
            for side, arm in self.arms.items():
                d.ctrl[arm.act_ids] = (
                    prev[side] + (new[side] - prev[side]) * s + ff[side]
                )
            mujoco.mj_step(m, d)
        if self.cfg.feedforward:
            for side, arm in self.arms.items():
                d.ctrl[arm.act_ids] = new[side]

    def observe(self) -> YamObservation:
        """Return the present observation (renders the views when enabled)."""
        views: dict[str, np.ndarray] = {}
        if self.cameras is not None:
            t0 = time.monotonic()
            views = self.cameras.render(self.data, overview=False)
            self.timing["render_s"] += time.monotonic() - t0
        return YamObservation(
            ee_pose={s: a.ee_pose() for s, a in self.arms.items()},
            grip={s: a.grip() for s, a in self.arms.items()},
            views=views,
            tick=self.tick_count,
            status=self.task.status,
        )

    def done(self) -> bool:
        """Return ``True`` at the time limit."""
        return self.tick_count >= self.max_ticks

    def summary(self) -> dict[str, float | int | bool | None]:
        """Return the episode result: success, progress metrics and counters."""
        s = self.task.status
        out: dict[str, float | int | bool | None] = {
            "success": s.success,
            "success_tick": s.success_tick,
            "ticks": s.ticks,
            "object_off_table": s.failed,
            "ik_misses": sum(a.ik_misses for a in self.arms.values()),
            "bad_rows": sum(a.bad_rows for a in self.arms.values()),
        }
        out.update({k: float(v) for k, v in s.metrics.items()})
        return out

    def close(self) -> None:
        """Free the renderers."""
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None

    def __enter__(self) -> YamEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
