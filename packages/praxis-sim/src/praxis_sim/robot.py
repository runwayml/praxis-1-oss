"""Frames, inverse kinematics, gripper map and the 15 Hz control tick of the sim arm.

Policy convention (the observation and command contract of
:mod:`praxis.single_arm` (copied in :mod:`praxis_sim.pose`)):

* The pose is ``[x, y, z, qw, qx, qy, qz]`` in metres, in the robot base
  frame. The quaternion is scalar first.
* The frame is the ``ee`` site of :mod:`praxis_sim.scene`: origin at the
  Franka flange, rotation ``flange @ Rz(-45 deg)``. :class:`FrameConfig`
  can move this frame (a yaw about the tool axis, a point along the tool
  axis, a base offset) to test other conventions.
* The grip is 0 when the gripper is open and 1 when it is closed. The
  observed grip is ``1 - width / 0.085``, with ``width`` the distance between
  the finger pads (the 2F-85 opens 85 mm).

One control tick holds one command row for ``1 / 15`` s. The tick solves the
inverse kinematics of the row once. Then each physics substep moves the joint
position targets along a straight line from the previous solution to the new
one, with a velocity feed-forward. The gripper target is ``255 * grip``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from praxis_sim import gl
from praxis_sim.scene import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    CONTROL_HZ,
    EE_SITE,
    FLANGE_SITE,
    GRIP_ACTUATOR,
    GRIPPER_PREFIX,
    MODEL_VIEWS,
    PINCH_SITE,
    mat_to_quat,
    rot_z,
)

logger = logging.getLogger(__name__)

# The reset joints of the DROID platform (public DROID code), radians.
DROID_HOME: tuple[float, ...] = (
    0.0,
    -math.pi / 5,
    0.0,
    -4 * math.pi / 5,
    0.0,
    3 * math.pi / 5,
    0.0,
)
GRIP_OPEN_WIDTH = 0.085  # metres between the pads of an open 2F-85
GRIP_CTRL_MAX = 255.0
LEFT_PAD = GRIPPER_PREFIX + "left_pad1"
RIGHT_PAD = GRIPPER_PREFIX + "right_pad1"

# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------


@dataclass
class FrameConfig:
    """The policy frame relative to the ``ee`` site of the scene.

    ``yaw_offset_deg`` turns the policy frame about the tool axis.
    ``tcp_offset`` moves the reported point along the tool axis (metres,
    0 = the flange). ``base_offset`` is the origin of the policy base frame
    in the world frame. The defaults give the frame of the DROID data.
    """

    yaw_offset_deg: float = 0.0
    tcp_offset: float = 0.0
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert one wxyz quaternion to a 3x3 rotation matrix (float64)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def site_to_policy(
    pos: np.ndarray, rot: np.ndarray, frame: FrameConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the ``ee`` site pose (world frame) to the policy pose."""
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    p = (
        np.asarray(pos, dtype=np.float64)
        + rot[:, 2] * frame.tcp_offset
        - np.asarray(frame.base_offset, dtype=np.float64)
    )
    r = rot @ rot_z(frame.yaw_offset_deg)
    return p, r


def policy_to_site(
    pos: np.ndarray, rot: np.ndarray, frame: FrameConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a policy pose to the ``ee`` site pose (world frame). The inverse of :func:`site_to_policy`."""
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3) @ rot_z(-frame.yaw_offset_deg)
    p = (
        np.asarray(pos, dtype=np.float64)
        + np.asarray(frame.base_offset, dtype=np.float64)
        - r[:, 2] * frame.tcp_offset
    )
    return p, r


def pose7(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Return ``[xyz, qw, qx, qy, qz]`` (float32) of a position and a rotation."""
    return np.concatenate([pos, mat_to_quat(rot)]).astype(np.float32)


def pose7_to_pos_rot(p7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[xyz, quat_wxyz]`` into a position and a 3x3 rotation."""
    p7 = np.asarray(p7, dtype=np.float64).reshape(7)
    return p7[:3].copy(), quat_to_mat(p7[3:7])


def pose12_to_pos_rot(p12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a command pose ``[xyz, R row-major 9]`` into a position and a rotation.

    The rotation is made orthonormal (SVD), because the command is float32.
    """
    p12 = np.asarray(p12, dtype=np.float64).reshape(12)
    u, _, vt = np.linalg.svd(p12[3:].reshape(3, 3))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return p12[:3].copy(), r


def pos_rot_to_pose12(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Return the command pose ``[xyz, R row-major 9]`` (float32)."""
    return np.concatenate(
        [np.asarray(pos).reshape(3), np.asarray(rot).reshape(9)]
    ).astype(np.float32)


def rotation_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector (world frame) of ``r_target @ r_current^T``."""
    q = mat_to_quat(r_target @ r_current.T)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Return the angle between two rotations in degrees."""
    c = (np.trace(r_a @ r_b.T) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


@dataclass
class IkConfig:
    """Settings of the damped least-squares inverse kinematics."""

    damping: float = 0.01  # lambda of (J J^T + lambda^2 I)
    rot_weight: float = 0.5  # metres of error per radian of error
    max_iters: int = 60
    tol_pos: float = 1e-4  # metres
    tol_rot: float = 1e-3  # radians
    max_step: float = 0.2  # largest joint change per iteration, radians
    null_gain: float = 0.05  # pull to the rest posture in the null space
    limit_margin: float = 0.01  # radians kept free at each joint limit
    miss_pos: float = 0.01  # a row with a larger final error is a miss (metres)
    miss_rot_deg: float = 5.0  # a row with a larger final error is a miss (degrees)


@dataclass
class IkResult:
    """Result of one inverse kinematics solve."""

    q: np.ndarray
    pos_err: float
    rot_err: float
    iters: int
    converged: bool
    miss: bool


class DlsIk:
    """Damped least-squares inverse kinematics of one site on the arm joints.

    ``joints`` names the joints that the solver moves (default: the 7 joints
    of the Panda arm).

    Each iteration does these steps:

    * Solve ``dq = J^T (J J^T + lambda^2 I)^-1 e`` for the 6-D error
      ``e = [dp, w * drot]``.
    * Add a pull to the rest posture in the null space of ``J``.
    * Limit the largest joint change.
    * Clip ``q`` to the joint limits less a margin.

    The solver uses its own ``MjData``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        q_rest: np.ndarray,
        cfg: IkConfig | None = None,
        joints: tuple[str, ...] = ARM_JOINTS,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.cfg = cfg or IkConfig()
        self.site_id = model.site(site).id
        self.qpos_adr = np.array([model.joint(j).qposadr[0] for j in joints])
        self.dof_adr = np.array([model.joint(j).dofadr[0] for j in joints])
        ranges = np.array([model.joint(j).range for j in joints])
        self.lo = ranges[:, 0] + self.cfg.limit_margin
        self.hi = ranges[:, 1] - self.cfg.limit_margin
        self.q_rest = np.asarray(q_rest, dtype=np.float64).copy()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the site position and rotation at the arm joints ``q``."""
        d = self.data
        d.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)  # mj_jacSite reads cdof
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(
            3, 3
        ).copy()

    def solve(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray) -> IkResult:
        """Solve for the site pose ``(pos, rot)``, starting from ``q_init``."""
        c = self.cfg
        q = np.clip(np.asarray(q_init, dtype=np.float64), self.lo, self.hi)
        eye6 = np.eye(6)
        eye7 = np.eye(len(q))
        pos_err = rot_err = math.inf
        iters = 0
        while iters < c.max_iters:
            iters += 1
            p, r = self.forward(q)
            dp = pos - p
            dr = rotation_error(rot, r)
            pos_err, rot_err = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if pos_err < c.tol_pos and rot_err < c.tol_rot:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site_id)
            jac = np.vstack(
                [self._jp[:, self.dof_adr], c.rot_weight * self._jr[:, self.dof_adr]]
            )
            err = np.concatenate([dp, c.rot_weight * dr])
            j_pinv = jac.T @ np.linalg.inv(jac @ jac.T + c.damping**2 * eye6)
            dq = j_pinv @ err + (eye7 - j_pinv @ jac) @ (
                c.null_gain * (self.q_rest - q)
            )
            largest = float(np.abs(dq).max())
            if largest > c.max_step:
                dq *= c.max_step / largest
            q = np.clip(q + dq, self.lo, self.hi)
        else:
            p, r = self.forward(q)
            pos_err = float(np.linalg.norm(pos - p))
            rot_err = float(np.linalg.norm(rotation_error(rot, r)))
        converged = pos_err < c.tol_pos and rot_err < c.tol_rot
        miss = pos_err > c.miss_pos or math.degrees(rot_err) > c.miss_rot_deg
        return IkResult(q, pos_err, rot_err, iters, converged, miss)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


class ViewSizes(Protocol):
    """The image sizes that :class:`Cameras` reads from a scene config."""

    image_width: int
    image_height: int
    overview_width: int
    overview_height: int


class Cameras:
    """Offscreen renderers of the scene cameras.

    The model views (default: exterior and wrist) use one renderer at the
    model image size. The overview view uses a second renderer at its own
    size. ``cfg`` needs ``image_height``, ``image_width``, ``overview_height``
    and ``overview_width``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: ViewSizes,
        views: tuple[str, ...] = MODEL_VIEWS,
    ) -> None:
        self.views = views
        self.model_views = mujoco.Renderer(model, cfg.image_height, cfg.image_width)
        self.renderer = gl.renderer_name()
        if gl.is_software_renderer(self.renderer):
            logger.warning(
                "MuJoCo renders on the CPU (%s). This is slow. See "
                "praxis_sim.gl for the EGL setup of an NVIDIA GPU.",
                self.renderer,
            )
        else:
            logger.info("MuJoCo renders with %s", self.renderer)
        self.overview: mujoco.Renderer | None = None
        if cfg.overview_width > 0 and cfg.overview_height > 0:
            self.overview = mujoco.Renderer(
                model, cfg.overview_height, cfg.overview_width
            )

    def render(
        self, data: mujoco.MjData, overview: bool = True
    ) -> dict[str, np.ndarray]:
        """Render the model views (and the overview view) as uint8 RGB images."""
        out: dict[str, np.ndarray] = {}
        for name in self.views:
            self.model_views.update_scene(data, camera=name)
            out[name] = self.model_views.render()
        if overview and self.overview is not None:
            self.overview.update_scene(data, camera="overview")
            out["overview"] = self.overview.render()
        return out

    def close(self) -> None:
        """Free the OpenGL resources."""
        self.model_views.close()
        if self.overview is not None:
            self.overview.close()
            self.overview = None


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------


@dataclass
class RobotState:
    """The state of the arm in the policy convention."""

    ee_pose: np.ndarray  # (7,) float32 [xyz, qw, qx, qy, qz]
    grip: float  # 0 = open, 1 = closed
    pinch: np.ndarray  # (3,) world position of the point between the pads
    q: np.ndarray = field(default_factory=lambda: np.zeros(7))


class SimArm:
    """The simulated arm and gripper: state readout, reset and one control tick."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        frame: FrameConfig | None = None,
        ik: IkConfig | None = None,
        home: tuple[float, ...] = DROID_HOME,
        feedforward: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.frame = frame or FrameConfig()
        self.home = np.asarray(home, dtype=np.float64)
        self.ik = DlsIk(model, EE_SITE, self.home, ik)
        self.feedforward = feedforward
        self.substeps = round(1.0 / (CONTROL_HZ * model.opt.timestep))
        self.qpos_adr = self.ik.qpos_adr
        self.dof_adr = self.ik.dof_adr
        self.act_ids = np.array([model.actuator(a).id for a in ARM_ACTUATORS])
        self.grip_act = model.actuator(GRIP_ACTUATOR).id
        kp = model.actuator_gainprm[self.act_ids, 0]
        kv = -model.actuator_biasprm[self.act_ids, 2]
        self.kv_over_kp = kv / kp
        self.ee_site = model.site(EE_SITE).id
        self.flange_site = model.site(FLANGE_SITE).id
        self.pinch_site = model.site(PINCH_SITE).id
        self.left_pad = model.geom(LEFT_PAD).id
        self.right_pad = model.geom(RIGHT_PAD).id
        self.pad_thickness = float(
            model.geom_size[self.left_pad][1] + model.geom_size[self.right_pad][1]
        )
        self.q_cmd = self.home.copy()
        self.grip_cmd = 0.0
        self.ik_misses = 0
        self.bad_rows = 0

    # -- state -------------------------------------------------------------

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the world pose of the ``ee`` site."""
        d = self.data
        return d.site_xpos[self.ee_site].copy(), d.site_xmat[self.ee_site].reshape(
            3, 3
        ).copy()

    def ee_pose(self) -> np.ndarray:
        """Return the end-effector pose in the policy convention (float32, 7)."""
        pos, rot = site_to_policy(*self.site_pose(), self.frame)
        return pose7(pos, rot)

    def grip_width(self) -> float:
        """Return the distance between the inner faces of the finger pads (metres)."""
        d = self.data
        axis = d.site_xmat[self.flange_site].reshape(3, 3)[:, 1]
        gap = abs(
            float(
                np.dot(d.geom_xpos[self.right_pad] - d.geom_xpos[self.left_pad], axis)
            )
        )
        return max(gap - self.pad_thickness, 0.0)

    def grip(self) -> float:
        """Return the observed grip: 0 = open, 1 = closed (DROID convention)."""
        return float(np.clip(1.0 - self.grip_width() / GRIP_OPEN_WIDTH, 0.0, 1.0))

    def pinch(self) -> np.ndarray:
        """Return the world position of the point between the finger pads."""
        return self.data.site_xpos[self.pinch_site].copy()

    def state(self) -> RobotState:
        """Return the arm state in the policy convention."""
        return RobotState(
            ee_pose=self.ee_pose(),
            grip=self.grip(),
            pinch=self.pinch(),
            q=self.data.qpos[self.qpos_adr].copy(),
        )

    # -- control -----------------------------------------------------------

    def reset(self, q: np.ndarray | None = None) -> None:
        """Put the arm at the joints ``q`` (default: home), at rest, gripper open."""
        q = self.home if q is None else np.asarray(q, dtype=np.float64)
        d = self.data
        d.qpos[self.qpos_adr] = q
        d.qvel[self.dof_adr] = 0.0
        d.ctrl[self.act_ids] = q
        d.ctrl[self.grip_act] = 0.0
        self.q_cmd = q.copy()
        self.grip_cmd = 0.0
        self.ik_misses = 0
        self.bad_rows = 0

    def joints_for_policy_pose(
        self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray | None = None
    ) -> IkResult:
        """Solve the inverse kinematics of a pose in the policy convention."""
        site_pos, site_rot = policy_to_site(pos, rot, self.frame)
        return self.ik.solve(
            site_pos, site_rot, self.q_cmd if q_init is None else q_init
        )

    def tick(self, pose12: np.ndarray, grip: float) -> IkResult | None:
        """Hold one command row for one 15 Hz tick.

        ``pose12`` is ``[xyz, R row-major]`` in the policy convention. A row
        that is not finite keeps the previous arm target. A grip that is not
        finite keeps the previous grip target. The grip is clipped to
        ``[0, 1]``. When the solver cannot reach a row, the arm goes to the
        nearest solution that the solver found, and the row counts as a miss.
        Returns the inverse kinematics result (``None`` for a row that is not
        finite).
        """
        pose12 = np.asarray(pose12, dtype=np.float64).reshape(12)
        result: IkResult | None = None
        q_prev = self.q_cmd.copy()
        if np.all(np.isfinite(pose12)):
            pos, rot = pose12_to_pos_rot(pose12)
            result = self.joints_for_policy_pose(pos, rot)
            self.ik_misses += int(result.miss)
            q_new = result.q
        else:
            self.bad_rows += 1
            q_new = q_prev
        g = float(grip)
        if math.isfinite(g):
            self.grip_cmd = float(np.clip(g, 0.0, 1.0))
        self.run_substeps(q_prev, q_new)
        self.q_cmd = q_new
        return result

    def run_substeps(self, q_prev: np.ndarray, q_new: np.ndarray) -> None:
        """Run one tick of physics. The joint targets ramp from ``q_prev`` to ``q_new``."""
        m, d = self.model, self.data
        n = self.substeps
        dq = q_new - q_prev
        ff = self.kv_over_kp * dq * CONTROL_HZ if self.feedforward else 0.0
        d.ctrl[self.grip_act] = GRIP_CTRL_MAX * self.grip_cmd
        for i in range(n):
            d.ctrl[self.act_ids] = q_prev + dq * ((i + 1) / n) + ff
            mujoco.mj_step(m, d)
        if self.feedforward:
            d.ctrl[self.act_ids] = q_new

    def settle(self, seconds: float) -> None:
        """Run the physics for ``seconds`` with the present targets."""
        steps = round(seconds / self.model.opt.timestep)
        self.data.ctrl[self.act_ids] = self.q_cmd
        self.data.ctrl[self.grip_act] = GRIP_CTRL_MAX * self.grip_cmd
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
