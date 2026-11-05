"""Pure numpy pieces of the praxis <-> RoboLab stock-runner bridge (no isaaclab, no positronic).

RoboLab's official runner calls `InferenceClient.infer(obs, instruction, env_id=...)`
once per 15 Hz control step and steps the env with the returned 8-dim action. With
RoboLab's AbsIK registration (`auto_register_droid_abs_ik_envs`) that action is
`DroidIKActionCfg`'s layout:

    [x, y, z, qw, qx, qy, qz, grip]

* position: base_link (the Robotiq mount, `body_name="base_link"`, zero body
  offset) in the ROBOT ROOT frame. RoboLab's robot root sits at the env origin
  with identity rotation, so the root frame IS the env-local frame the
  `eef_pos` observation reports;
* rotation: base_link's orientation. `DroidIKActionCfg` tracks base_link (no
  body_offset rotation), so a target given in RoboLab's `eef_frame`
  (= base_link @ `EEF_OFFSET_ROT`, zero translation) is un-offset first:
  `q_base = q_eef (x) EEF_OFFSET_ROT^-1` (`examples/run_abs_ik_demo.py`);
* grip: `BinaryJointPositionZeroToOneAction`, closed when > 0.5.

Positronic's RoboLab env server (`positronic/simulator/robolab/env.py`) gets the
same command as a wire `cartesian` pose `[t(3), R(9) row-major]` in `eef_frame`
coordinates and does exactly this conversion before its own IK solve
(`_solve_ik`: `target_quat_w = quat_mul(target_quat, quat_inv(eef_offset_rot))`,
`target_pos_w = target_pos + env_origins`). `wire_to_absik_action` is that
conversion. A wire `hold` (no command yet: the policy's first chunk is still being
computed, which in a paused-sim loop never happens) holds the measured base_link
pose.

Frames, all FK-measured (`positronic_cfg/ee_frame_wire.py`): RoboLab's `eef_frame`
= flange @ [Rz(+90 deg) | +18.17 mm]; the policy server
expects the FR3 end_effector = flange @ [Rz(-45 deg) | +103.4 mm]. That client-side
remap stays in `ee_frame_policy_cfg.EeFrameChange` (run in the sidecar), so
the wire pose reaching this module is already back in RoboLab's eef frame.
"""

from collections.abc import Mapping

import numpy as np

# RoboLab's `EEF_OFFSET_ROT` (robolab/robots/droid.py @ 7d45d74), wxyz: the
# eef_frame's rotation relative to base_link. Duplicated (not imported) so this
# module stays importable without isaaclab.
EEF_OFFSET_ROT = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)
EEF_OFFSET_POS = np.zeros(3, dtype=np.float64)

ACTION_DIM = 8  # DroidIKActionCfg: arm_action (7, absolute pose) + finger_joint (1)

# The wire command tags `positronic.simulator.env_server.protocol` defines.
CARTESIAN = "cartesian"
HOLD = "hold"


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product `q1 (x) q2`, both wxyz."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a quaternion (wxyz), not assumed unit."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion of a 3x3 rotation (Shepperd), sign-canonical (w >= 0).

    The sign is irrelevant downstream: IsaacLab's IK pose error goes through
    `axis_angle_from_quat`, which flips a negative-w error quaternion.
    """
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    quat = np.asarray(q, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return -quat if quat[0] < 0 else quat


def eef_to_base_quat(q_eef: np.ndarray) -> np.ndarray:
    """A target orientation in RoboLab's eef_frame -> base_link's (what DroidIK tracks)."""
    return quat_mul(q_eef, quat_inv(EEF_OFFSET_ROT))


def base_to_eef_quat(q_base: np.ndarray) -> np.ndarray:
    """base_link's orientation -> RoboLab's eef_frame (the `eef_quat` obs term)."""
    return quat_mul(q_base, EEF_OFFSET_ROT)


def wire_to_absik_action(
    command: Mapping[str, object],
    grip: float,
    held_base_pose7: np.ndarray,
) -> np.ndarray:
    """One held wire action -> the 8-dim `DroidIKActionCfg` action.

    `command` is the sidecar's wire command (`praxis_sidecar.wire_command`): `{'type':
    'cartesian', 'pose': [t(3), R(9) row-major]}` in RoboLab's eef frame
    (env-local position, world rotation), or `{'type': 'hold'}`. `grip` is the
    wire grip in [0, 1] (1 = closed); RoboLab's binary finger term thresholds it
    at 0.5 itself, as Positronic's env server forwards it. `held_base_pose7`
    (`[xyz, wxyz]`, base_link, env-local) is what a `hold` re-commands.
    """
    kind = command.get("type")
    action = np.zeros(ACTION_DIM, dtype=np.float64)
    if kind == CARTESIAN:
        pose = np.asarray(command["pose"], dtype=np.float64).reshape(-1)
        if pose.shape != (12,):
            raise ValueError(f"cartesian wire pose must be (12,), got {pose.shape}")
        q_eef = matrix_to_quat(pose[3:].reshape(3, 3))
        # eef_frame and base_link share their origin (EEF_OFFSET_POS = 0).
        action[:3] = pose[:3] - quat_to_matrix(eef_to_base_quat(q_eef)) @ EEF_OFFSET_POS
        action[3:7] = eef_to_base_quat(q_eef)
    elif kind == HOLD:
        held = np.asarray(held_base_pose7, dtype=np.float64).reshape(7)
        action[:7] = held
    else:
        raise ValueError(f"unsupported wire command type {kind!r} for DroidIKActionCfg")
    action[7] = float(grip)
    return action
