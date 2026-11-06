"""Pose math and command helpers of the simulation, without torch.

Copies of the functions of ``praxis.robot_io``, ``praxis.single_arm`` and
``praxis.train.data`` that the simulation uses (the scripted policy, the Gymnasium
wrapper), so that the package installs without the model's dependencies.
``tests/test_sim_pose_copies.py`` of the praxis repository checks every copy against
its original, bit for bit (``DROID_IMPEDANCE`` included: a change of the table in
``praxis.single_arm`` has to land here too). Keep them in sync.
``matrix_to_euler_xyz`` and ``pose8_to_droid7`` are new (the DROID pose of the
Gymnasium wrapper); ``euler_xyz_to_matrix`` composes two of the copies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CONTROL_MODES",
    "DROID_IMPEDANCE",
    "FLANGE_TURN",
    "FR3_HAND",
    "NO_TURN",
    "TURNED_FLANGE",
    "RigFrame",
    "control_mode_payload",
    "euler_xyz_to_matrix",
    "euler_xyz_to_quat_xyzw",
    "matrix_to_euler_xyz",
    "matrix_to_quat_xyzw",
    "pose8_to_droid7",
    "pose8_to_model",
    "pose_to_rig",
    "quat_wxyz_to_matrix",
    "quat_xyzw_to_matrix",
    "rot6d",
    "rot6d_to_matrix",
    "unit_quat_wxyz_to_matrix",
    "wire_to_arrays",
]

# --- praxis.robot_io


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` matrices.

    The quaternion is normalized with a 1e-8 guard. This is the convention of
    the proprio and of the training targets.
    """
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(*q.shape[:-1], 3, 3)


def unit_quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert one wxyz quaternion to a float64 3x3 matrix.

    This is the convention of the command path. The quaternion is normalized
    exactly (no guard) and the matrix uses the ``x2 - y2 - z2 + w2`` diagonal
    form. The scalar operation order is fixed, so the result is bitwise
    reproducible.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = (float(v) for v in q)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Found zero norm quaternions in `quat`.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array(
        [
            [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ],
        dtype=np.float64,
    )


def rot6d(r: np.ndarray) -> np.ndarray:
    """Return the first two columns of ``(..., 3, 3)`` matrices as ``(..., 6)``."""
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Convert one rot6d ``[c0(3), c1(3)]`` to a float64 3x3 matrix.

    The two columns are made orthonormal with Gram-Schmidt. The third column
    is their cross product.
    """
    a0 = r6[:3].astype(np.float64)
    a1 = r6[3:6].astype(np.float64)
    c0 = a0 / (np.linalg.norm(a0) + 1e-8)
    a1 = a1 - np.dot(c0, a1) * c0
    c1 = a1 / (np.linalg.norm(a1) + 1e-8)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1)


# --- praxis.single_arm: client frames

_C, _S = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
FLANGE_TURN = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])

NO_TURN = np.eye(3)


@dataclass(frozen=True)
class RigFrame:
    """The frame of the poses of one client, relative to the Franka flange.

    ``tool_offset`` is the distance of the pose origin from the flange along
    the flange z axis (metres). ``turn`` is the turn of the client rotation:
    ``R_client = R_flange @ turn``. ``base_offset`` is the origin of the robot
    base in the coordinates of the client (metres).
    """

    tool_offset: float
    turn: tuple[tuple[float, ...], ...]
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _rows(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in m)


TURNED_FLANGE = RigFrame(0.0, _rows(FLANGE_TURN))
FR3_HAND = RigFrame(0.1034, _rows(FLANGE_TURN))


# --- praxis.single_arm: control modes

DROID_IMPEDANCE: dict[str, object] = {
    "type": "impedance",
    "kq": [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    "kqd": [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    "kx": [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
    "kxd": [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
}

CONTROL_MODES: dict[str, dict[str, object] | None] = {
    "none": None,
    "droid_impedance": DROID_IMPEDANCE,
}


def control_mode_payload(control_mode: str) -> dict[str, object] | None:
    """Return the ``robot_command["mode"]`` value of a control mode (a copy).

    ``"none"`` gives ``None``. An unknown name raises ``ValueError``.
    """
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"Unknown control_mode {control_mode!r}. The known modes are "
            f"{sorted(CONTROL_MODES)}."
        )
    payload = CONTROL_MODES[control_mode]
    return None if payload is None else copy.deepcopy(payload)


# --- praxis.single_arm: quaternion conversions with the bits of scipy, and the frame conversions


def _quat_xyzw_unit(q: np.ndarray) -> np.ndarray:
    """Normalize ``(..., 4)`` xyzw quaternions in float64 (scipy's order of ops)."""
    q = np.array(q, dtype=np.float64, copy=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return q / np.sqrt(x * x + y * y + z * z + w * w)[..., None]


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` xyzw quaternions to ``(..., 3, 3)`` float64 matrices.

    The quaternion is normalized exactly. The operations are those of
    ``scipy.spatial.transform.Rotation.from_quat(q).as_matrix()``, so the
    result has the same bits.
    """
    q = _quat_xyzw_unit(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    m = np.empty((*q.shape[:-1], 3, 3))
    m[..., 0, 0] = x2 - y2 - z2 + w2
    m[..., 1, 0] = 2 * (xy + zw)
    m[..., 2, 0] = 2 * (xz - yw)
    m[..., 0, 1] = 2 * (xy - zw)
    m[..., 1, 1] = -x2 + y2 - z2 + w2
    m[..., 2, 1] = 2 * (yz + xw)
    m[..., 0, 2] = 2 * (xz + yw)
    m[..., 1, 2] = 2 * (yz - xw)
    m[..., 2, 2] = -x2 - y2 + z2 + w2
    return m


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Convert ``(..., 3, 3)`` rotation matrices to ``(..., 4)`` xyzw quaternions.

    The operations are those of
    ``scipy.spatial.transform.Rotation.from_matrix(m).as_quat()`` (the
    largest-component branch, then an exact normalization), so the result
    has the same bits. The sign is not made canonical.
    """
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for n, c in enumerate(flat):
        dec = [c[0, 0], c[1, 1], c[2, 2], c[0, 0] + c[1, 1] + c[2, 2]]
        choice = int(np.argmax(dec))
        q = np.empty(4)
        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3
            q[i] = 1 - dec[3] + 2 * c[i, i]
            q[j] = c[j, i] + c[i, j]
            q[k] = c[k, i] + c[i, k]
            q[3] = c[k, j] - c[j, k]
        else:
            q[0] = c[2, 1] - c[1, 2]
            q[1] = c[0, 2] - c[2, 0]
            q[2] = c[1, 0] - c[0, 1]
            q[3] = 1 + dec[3]
        out[n] = q
    return _quat_xyzw_unit(out).reshape(*m.shape[:-2], 4)


def pose8_to_model(pose8: np.ndarray, frame: RigFrame = TURNED_FLANGE) -> np.ndarray:
    """Convert client pose8 rows ``(..., 8)`` in ``frame`` to the flange frame of the model.

    The position becomes ``xyz - base_offset - tool_offset * z_client`` (the
    flange position in base coordinates). The rotation becomes
    ``R_client @ turn^T`` (the flange rotation). The quaternion comes back
    wxyz with the sign of :func:`matrix_to_quat_xyzw`. The gripper is clipped
    to [0, 1]. The result is float32. With ``TURNED_FLANGE`` the offsets are
    zero and keep the signed zeros of the original implementation.
    """
    out = np.array(pose8, dtype=np.float64, copy=True)
    rot = quat_xyzw_to_matrix(out[..., [4, 5, 6, 3]])
    base = np.asarray(frame.base_offset, dtype=np.float64)
    out[..., :3] = out[..., :3] - base - frame.tool_offset * rot[..., :, 2]
    turn = np.asarray(frame.turn).T @ NO_TURN
    out[..., 3:7] = matrix_to_quat_xyzw(rot @ turn)[..., [3, 0, 1, 2]]
    out[..., 7] = np.clip(out[..., 7], 0.0, 1.0)
    return out.astype(np.float32)


def pose_to_rig(
    xyz: np.ndarray, rot: np.ndarray, frame: RigFrame = TURNED_FLANGE
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a flange-frame target to ``frame``, the frame of the client.

    ``xyz (3,)`` and ``rot (3, 3)`` are float64. The rotation becomes
    ``rot @ turn``. The position becomes
    ``xyz + tool_offset * z_flange + base_offset``. The flange and the client
    frame share the z axis.
    """
    xyz = xyz + frame.tool_offset * rot[:, 2] + np.asarray(frame.base_offset)
    return xyz, rot @ NO_TURN.T @ np.asarray(frame.turn)


def wire_to_arrays(wire: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack command dicts to ``(T, 12)`` poses and ``(T,)`` grips (float32)."""
    poses = np.stack([np.asarray(r["robot_command"]["pose"]) for r in wire])
    grips = np.asarray([r["target_grip"] for r in wire], dtype=np.float32)
    return poses.astype(np.float32), grips


# --- Extrinsic x-y-z Euler angles (the DROID convention), with the bits of scipy's conversions.


def quat_mul_xyzw(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the products ``p * q`` of ``(N, 4)`` xyzw quaternions (scipy's order of operations)."""
    c0 = p[:, 1] * q[:, 2] - p[:, 2] * q[:, 1]
    c1 = p[:, 2] * q[:, 0] - p[:, 0] * q[:, 2]
    c2 = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    out = np.empty_like(p)
    out[:, 0] = p[:, 3] * q[:, 0] + q[:, 3] * p[:, 0] + c0
    out[:, 1] = p[:, 3] * q[:, 1] + q[:, 3] * p[:, 1] + c1
    out[:, 2] = p[:, 3] * q[:, 2] + q[:, 3] * p[:, 2] + c2
    out[:, 3] = p[:, 3] * q[:, 3] - p[:, 0] * q[:, 0] - p[:, 1] * q[:, 1]
    out[:, 3] -= p[:, 2] * q[:, 2]
    return out


def euler_xyz_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` extrinsic x-y-z Euler angles to ``(N, 4)`` xyzw quaternions
    (the operations of ``Rotation.from_euler("xyz", rpy)``)."""
    e = np.asarray(rpy, dtype=np.float64).reshape(-1, 3)
    q = np.zeros((e.shape[0], 4))
    for axis in range(3):
        half = e[:, axis] / 2
        elem = np.zeros((e.shape[0], 4))
        elem[:, axis] = np.sin(half)
        elem[:, 3] = np.cos(half)
        q = elem if axis == 0 else quat_mul_xyzw(elem, q)
    return q


def quat_xyzw_to_matrix_raw(q: np.ndarray) -> np.ndarray:
    """``(N, 4)`` xyzw quaternions to ``(N, 3, 3)`` matrices without normalization
    (``Rotation.as_matrix`` of a rotation from ``from_euler``)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = x2 - y2 - z2 + w2
    m[:, 1, 0] = 2 * (xy + zw)
    m[:, 2, 0] = 2 * (xz - yw)
    m[:, 0, 1] = 2 * (xy - zw)
    m[:, 1, 1] = -x2 + y2 - z2 + w2
    m[:, 2, 1] = 2 * (yz + xw)
    m[:, 0, 2] = 2 * (xz + yw)
    m[:, 1, 2] = 2 * (yz - xw)
    m[:, 2, 2] = -x2 - y2 + z2 + w2
    return m


def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` extrinsic x-y-z Euler angles to ``(N, 3, 3)`` rotation matrices."""
    return quat_xyzw_to_matrix_raw(euler_xyz_to_quat_xyzw(rpy))


def matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3, 3)`` rotation matrices to ``(N, 3)`` extrinsic x-y-z Euler angles
    (``roll = atan2(R21, R22)``, ``pitch = asin(-R20)``, ``yaw = atan2(R10, R00)``; near a right
    angle of pitch the roll is 0 and the yaw absorbs the turn)."""
    r = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
    pitch = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    regular = np.abs(cos_pitch) > 1e-6
    roll = np.where(regular, np.arctan2(r[:, 2, 1], r[:, 2, 2]), 0.0)
    yaw = np.where(
        regular, np.arctan2(r[:, 1, 0], r[:, 0, 0]), np.arctan2(-r[:, 0, 1], r[:, 1, 1])
    )
    return np.stack([roll, pitch, yaw], axis=-1)


def pose8_to_droid7(pose8: np.ndarray) -> np.ndarray:
    """Convert ``(..., 8)`` pose8 rows to float32 ``[xyz, roll, pitch, yaw, grip]`` rows."""
    a = np.asarray(pose8, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    rot = np.stack([unit_quat_wxyz_to_matrix(q) for q in flat[:, 3:7]])
    rpy = matrix_to_euler_xyz(rot)
    out = np.concatenate([flat[:, :3], rpy, flat[:, 7:8]], axis=1).astype(np.float32)
    return out.reshape(*a.shape[:-1], 7)
