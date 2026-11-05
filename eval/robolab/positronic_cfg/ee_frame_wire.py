"""Constant EE tool-frame change between a sim's wire EE frame and a policy's EE frame.

A policy server that applies no EE-convention remap consumes the wire
``robot_state.ee_pose`` as if it were its TRAINED EE frame and emits
``robot_command`` poses in that same frame. When the sim's wire frame differs from the
trained one by a constant tool offset, the proprio it sees is off-distribution (the
anchor-relative ACTIONS are unaffected: a right-multiplied constant cancels in
``R_t @ R_anchor^T``). This adapter fixes that client-side, both ways:

  obs:      T_model = T_wire  @ M        (``robot_state.ee_pose``, xyz + wxyz)
  commands: T_wire  = T_model @ M^-1     (``robot_command`` Cartesian poses)

``M`` is a TOOL-frame (right-multiplied) rigid offset, composed from each frame's
offset relative to the bare Franka flange (the canonical basis):
``T_frame = T_flange @ tool_offset(...)``.

Frames (all FK-measured):

* ``ROBOLAB_EEF``: RoboLab's wire ``eef_frame`` = flange @ [Rz(+90 deg) | (0, 0,
  +18.17 mm)] (DH-flange FK vs recorded wire eef, residual 0.000 deg / 0.00 mm).
* ``FR3_END_EFFECTOR``: the frame the policy server expects from
  ``descriptor="remote.robolab.droid"`` = flange @ [Rz(-45 deg)
  | (0, 0, +103.4 mm)]. That server converts it to the flange and the checkpoint's own
  model turn itself, and refuses any request > 5 mm off the flange FK of
  ``robot_state.q``, so a client in front of it maps the wire to this frame and applies
  no model turn of its own (``ROBOLAB_TO_FR3_EE``).

This module is the pure mapping: numpy only, no positronic or configuronic imports.
``ee_frame_policy_cfg.EeFrameChange`` (inside positronic's venv) applies it.
"""

from collections.abc import Mapping

import numpy as np

EE_POSE = "robot_state.ee_pose"
ROBOT_COMMAND = "robot_command"


def rz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_offset(rot_z_deg: float, z_offset_m: float) -> np.ndarray:
    """4x4 ``[Rz(rot_z_deg) | (0, 0, z_offset_m)]``: T_frame = T_flange @ this."""
    out = np.eye(4)
    out[:3, :3] = rz(rot_z_deg)
    out[2, 3] = z_offset_m
    return out


def inverse(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    rot = transform[:3, :3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ transform[:3, 3]
    return out


def frame_change(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """``M`` with ``T_dst = T_src @ M`` for two flange-relative tool offsets."""
    return inverse(src) @ dst


ROBOLAB_EEF = tool_offset(90.0, 0.01817)
FR3_END_EFFECTOR = tool_offset(-45.0, 0.1034)
# = [Rz(-135 deg) | (0, 0, +85.23 mm)]
ROBOLAB_TO_FR3_EE = frame_change(ROBOLAB_EEF, FR3_END_EFFECTOR)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Unit ``[w, x, y, z]`` (Shepperd), sign-canonicalised to ``w >= 0``."""
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


def apply_rt(
    translation: np.ndarray, rotation: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``[R | t] @ offset`` -> (t', R')."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return t + r @ offset[:3, 3], r @ offset[:3, :3]


def remap_pose7(pose: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """xyz + wxyz pose(s) ``(..., 7)`` right-multiplied by ``offset``; dtype kept."""
    arr = np.asarray(pose)
    if arr.shape[-1] != 7:
        raise ValueError(f"expected (..., 7) xyz+wxyz poses, got {arr.shape}")
    flat = arr.reshape(-1, 7).astype(np.float64)
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        t, r = apply_rt(row[:3], quat_wxyz_to_matrix(row[3:]), offset)
        out[i, :3], out[i, 3:] = t, matrix_to_quat_wxyz(r)
    dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
    return out.reshape(arr.shape).astype(dtype)


def obs_to_model(obs: Mapping[str, object], offset: np.ndarray) -> dict[str, object]:
    """``obs`` with ``robot_state.ee_pose`` re-expressed in the model's EE frame."""
    if EE_POSE not in obs:
        raise KeyError(f"obs missing {EE_POSE!r}; got keys={sorted(obs)}")
    out = dict(obs)
    out[EE_POSE] = remap_pose7(np.asarray(obs[EE_POSE]), offset)
    return out
