"""A scripted pick policy that sends the same commands as the model.

Use it to check the environment without model weights. Each call of
:meth:`ScriptedPick.step` returns one chunk of command dicts, in the format
of ``StepResult.commands`` of :mod:`praxis.pipeline`:

* absolute end-effector poses ``[xyz, R row-major]`` and a target grip,
* one row per 15 Hz tick,
* 21 rows in the first chunk and 24 rows in each chunk after it.

The rows are relative to the arm pose at the start of the chunk, as the
model actions are. :func:`rig_commands` composes them with that pose in the
frame of the rig (the ``ee`` site of :mod:`praxis.sim.scene`) and gives the
command dicts of :mod:`praxis.single_arm`. The model session converts its own
pose frame to the rig frame itself, so the scripted rows need no conversion.

The policy reads the cube pose from the simulation (privileged data). It
plans a path of the pinch point: above the cube, down to the cube, close the
gripper, lift, hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from praxis import single_arm
from praxis.robot_io import rot6d, rot6d_to_matrix, unit_quat_wxyz_to_matrix
from praxis.sim.env import Observation, SimEnv
from praxis.sim.robot import pose7_to_pos_rot, site_to_policy
from praxis.sim.scene import CONTROL_HZ, PINCH_OFFSET

FIRST_CHUNK_ROWS = 21
CHUNK_ROWS = 24


@dataclass
class ScriptedConfig:
    """Settings of the scripted pick."""

    approach_height: float = 0.12  # metres above the grasp point
    grasp_depth: float = 0.0  # metres of the pinch point below the cube centre
    lift_height: float = 0.15  # metres
    speed: float = 0.15  # metres per second of the free moves
    descend_s: float = 1.2
    close_s: float = 0.8
    lift_s: float = 1.5
    grip_rate: float = 0.25  # grip change per tick
    control_mode: str = "droid_impedance"


def rig_commands(
    rel: np.ndarray,
    pose8_now: np.ndarray,
    dt: float = 1.0 / CONTROL_HZ,
    mode: dict[str, object] | None = None,
) -> list[dict]:
    """Convert ``(T, 10)`` relative rows in the rig frame to command dicts.

    Each row is ``[xyz3, rot6d6, grip1]`` relative to the pose ``pose8_now``
    (``[xyz3, quat_wxyz4, grip1]``): ``xyz_abs = xyz_rel + anchor_xyz`` in
    float64 and ``R_abs = R_rel @ R_anchor``. The gripper is absolute. The
    dicts have the format of :mod:`praxis.single_arm`: the pose
    ``[xyz3, R_flat9]`` float32 with a row-major rotation, the target grip,
    the ``timestamp`` offset ``t * dt``, and ``mode`` in every command when it
    is set.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < 10:
        raise ValueError(f"rel rows must be (T, >=10), got {rel.shape}")
    anchor = np.asarray(pose8_now)
    anchor_xyz = anchor[:3].astype(np.float64)
    r_anchor = unit_quat_wxyz_to_matrix(anchor[3:7])
    out: list[dict] = []
    for t, r in enumerate(rel):
        xyz_abs = r[:3].astype(np.float64) + anchor_xyz
        r_abs = rot6d_to_matrix(r[3:9]) @ r_anchor
        pose12 = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
        cmd: dict[str, object] = {"type": "cartesian_pos", "pose": pose12}
        if mode is not None:
            cmd["mode"] = mode
        out.append(
            {
                "robot_command": cmd,
                "target_grip": float(r[9]),
                "timestamp": float(t * dt),
            }
        )
    return out


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Return the minimum-jerk profile ``10 s^3 - 15 s^4 + 6 s^5`` on ``[0, 1]``."""
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def down_rotation(finger_yaw: float) -> np.ndarray:
    """Return the flange rotation with the tool axis down and the fingers along ``finger_yaw``.

    The fingers move along the flange y axis. ``finger_yaw`` is the angle of
    that axis about the base z axis, radians.
    """
    c, s = math.cos(finger_yaw), math.sin(finger_yaw)
    y = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def wrap(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


class ScriptedPick:
    """A scripted pick-and-lift of the task cube, in the command format of the model."""

    def __init__(self, cfg: ScriptedConfig | None = None) -> None:
        self.cfg = cfg or ScriptedConfig()
        self.mode = single_arm.control_mode_payload(self.cfg.control_mode)
        self.plan_pos: np.ndarray = np.zeros((0, 3))
        self.plan_rot: np.ndarray = np.zeros((0, 3, 3))
        self.plan_grip: np.ndarray = np.zeros(0)
        self.cursor = 0
        self.calls = 0

    def reset(self, env: SimEnv) -> None:
        """Plan the episode from the present arm pose and the oracle steps of the task."""
        c = self.cfg
        site_pos, site_rot = env.arm.site_pose()
        flange_rot = env.data.site_xmat[env.arm.flange_site].reshape(3, 3)
        site_from_flange = flange_rot.T @ site_rot
        press = getattr(env.task, "press_depth", None)
        steps = env.task.oracle_steps(env.data)
        segments: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]
        ]
        segments = []
        now_pos, now_rot = site_pos, site_rot
        now_yaw = math.atan2(flange_rot[1, 1], flange_rot[0, 1])
        for step in steps:
            # Fingers across two opposite faces of the object: the face normals
            # are at yaw + k * 90 deg. Use the one nearest to the present yaw.
            yaw = min(
                (step.yaw + k * math.pi / 2 for k in range(-4, 5)),
                key=lambda a: abs(wrap(a - now_yaw)),
            )
            yaw = now_yaw + wrap(yaw - now_yaw)
            grasp_flange_rot = down_rotation(yaw)
            grasp_rot = grasp_flange_rot @ site_from_flange
            grasp = step.pos + np.array([0.0, 0.0, step.grasp_dz - c.grasp_depth])
            approach = c.approach_height if step.approach is None else step.approach

            def site_at(
                pinch: np.ndarray, r: np.ndarray = grasp_flange_rot
            ) -> np.ndarray:
                return pinch - r[:, 2] * PINCH_OFFSET

            above = site_at(grasp + np.array([0.0, 0.0, approach]))
            at = site_at(grasp)
            lifted = site_at(grasp + np.array([0.0, 0.0, c.lift_height]))
            move_s = max(1.0, float(np.linalg.norm(above - now_pos)) / c.speed)
            if press is not None:
                # A press: close the fingers on the way, push the closed fingers
                # into the top of the target, then move up.
                top = step.pos + np.array([0.0, 0.0, 0.0075])
                down = site_at(top - np.array([0.0, 0.0, press]))
                segments += [
                    (now_pos, above, now_rot, grasp_rot, move_s, 1.0),
                    (above, down, grasp_rot, grasp_rot, c.descend_s * 1.5, 1.0),
                    (down, down, grasp_rot, grasp_rot, c.close_s, 1.0),
                    (down, above, grasp_rot, grasp_rot, c.lift_s, 1.0),
                ]
                now_pos, now_rot, now_yaw = above, grasp_rot, yaw
                continue
            segments += [
                (now_pos, above, now_rot, grasp_rot, move_s, 0.0),
                (above, at, grasp_rot, grasp_rot, c.descend_s, 0.0),
                (at, at, grasp_rot, grasp_rot, c.close_s, 1.0),
                (at, lifted, grasp_rot, grasp_rot, c.lift_s, 1.0),
            ]
            now_pos, now_rot, now_yaw = lifted, grasp_rot, yaw
            if step.target is None:
                continue
            # Carry the object above the target, put it down, open, move up.
            drop = step.target + np.array(
                [0.0, 0.0, step.grasp_dz - c.grasp_depth + 0.004]
            )
            over = site_at(drop + np.array([0.0, 0.0, c.lift_height]))
            down = site_at(drop)
            place_rot = grasp_rot
            if step.target_yaw is not None:
                # Keep the object yaw relative to the fingers; turn the fingers
                # by the smallest angle that gives the object the target yaw.
                turn = min(
                    (
                        step.target_yaw - step.yaw + k * math.pi / 2
                        for k in range(-4, 5)
                    ),
                    key=lambda a: abs(wrap(a)),
                )
                yaw = yaw + wrap(turn)
                place_rot = down_rotation(yaw) @ site_from_flange
            carry_s = max(1.0, float(np.linalg.norm(over - lifted)) / c.speed)
            place_s = c.descend_s if step.place_s is None else step.place_s
            segments += [
                (lifted, over, grasp_rot, place_rot, carry_s, 1.0),
                (over, down, place_rot, place_rot, place_s, 1.0),
                (down, down, place_rot, place_rot, c.close_s, 0.0),
                (down, over, place_rot, place_rot, c.lift_s, 0.0),
            ]
            now_pos, now_rot, now_yaw = over, place_rot, yaw
        pos_rows: list[np.ndarray] = []
        rot_rows: list[np.ndarray] = []
        grip_rows: list[float] = []
        grip = 0.0
        for p0, p1, r0, r1, seconds, g in segments:
            n = max(1, round(seconds * CONTROL_HZ))
            s = min_jerk(np.arange(1, n + 1) / n)
            for k in range(n):
                pos_rows.append(p0 + (p1 - p0) * s[k])
                rot_rows.append(_slerp(r0, r1, float(s[k])))
                grip += float(np.clip(g - grip, -c.grip_rate, c.grip_rate))
                grip_rows.append(grip)
        frame = env.arm.frame
        policy = [
            site_to_policy(p, r, frame) for p, r in zip(pos_rows, rot_rows, strict=True)
        ]
        self.plan_pos = np.stack([p for p, _ in policy])
        self.plan_rot = np.stack([r for _, r in policy])
        self.plan_grip = np.asarray(grip_rows)
        self.cursor = 0
        self.calls = 0

    def _row(self, i: int) -> tuple[np.ndarray, np.ndarray, float]:
        i = min(i, len(self.plan_grip) - 1)
        return self.plan_pos[i], self.plan_rot[i], float(self.plan_grip[i])

    def step(self, obs: Observation) -> list[dict]:
        """Return the command dicts of the next chunk (21 rows first, then 24)."""
        rows = FIRST_CHUNK_ROWS if self.calls == 0 else CHUNK_ROWS
        anchor_pos, anchor_rot = pose7_to_pos_rot(obs.ee_pose)
        rel = np.zeros((rows, 10), dtype=np.float32)
        for t in range(rows):
            pos, rot, grip = self._row(self.cursor + t)
            rel[t, :3] = pos - anchor_pos
            rel[t, 3:9] = rot6d(rot @ anchor_rot.T)
            rel[t, 9] = grip
        pose8_now = np.concatenate([obs.ee_pose, [obs.grip]]).astype(np.float32)
        self.cursor += rows
        self.calls += 1
        return rig_commands(rel, pose8_now, 1.0 / CONTROL_HZ, self.mode)


def _slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    """Interpolate two rotations along the shortest arc (``s`` in ``[0, 1]``)."""
    rel = r1 @ r0.T
    angle = math.acos(float(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-9:
        return r1.copy() if s >= 1 else r0.copy()
    axis = np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # angle near pi: take the axis from the symmetric part
        w, v = np.linalg.eigh((rel + rel.T) / 2)
        axis = v[:, np.argmax(w)]
    else:
        axis = axis / norm
    a = angle * s
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    turn = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)
    return turn @ r0


def run_episode(
    env: SimEnv,
    policy: ScriptedPick,
    seed: int,
    max_ticks: int | None = None,
    stop_on_success: bool = False,
) -> dict[str, float | int | bool | None]:
    """Run one episode of the scripted policy and return the episode summary.

    The function executes every row of each chunk, as the model loop does.
    """
    obs = env.reset(seed)
    policy.reset(env)
    limit = env.max_ticks if max_ticks is None else int(max_ticks)
    while env.tick_count < limit:
        commands = policy.step(obs)
        poses, grips = single_arm.wire_to_arrays(commands)
        for pose12, grip in zip(poses, grips, strict=True):
            obs = env.step(pose12, float(grip))
            if env.tick_count >= limit:
                break
        if stop_on_success and env.task.status.success:
            break
    return env.summary()
