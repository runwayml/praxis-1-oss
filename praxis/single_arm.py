"""Robot input and output glue for the single-arm rig. No neural networks.

Observation contract (rig side):

* ``robot_state.ee_pose``: end-effector pose, 7 values
  ``[x, y, z, qw, qx, qy, qz]`` (quaternion scalar first), metres.
* ``grip``: absolute gripper value, 0 = open.
* ``image.exterior`` and ``image.wrist``: RGB uint8. Both views are needed.
* Each value can carry a leading time axis (a frame stack). The current
  frame is the last row.

Model space:

* pose8 = ``[xyz3, quat_wxyz4, grip1]``.
* 10-D = ``[xyz3, rot6d6, grip1]``. rot6d holds the first two columns of the
  rotation matrix.
* Actions are relative to the arm pose at the start of the chunk
  (``xyz - anchor_xyz``, ``R @ R_anchor^T``). The gripper is absolute.
* Actions are q99-normalized on the xyz and grip dims. rot6d stays identity.
  Proprio is mean/std-normalized on all 10 dims.
* The decode clips the normalized xyz and grip dims to [-1, 1], the range of
  the training targets. After the inverse q99 map, it clips the gripper to
  [0, 1].
* The canvas is the tile canvas. Each view gets its own tile (256x416 by
  default): a centre crop to the tile aspect, one bilinear resize, a round to
  uint8. The exterior tile and the wrist tile go side by side (256x832).
* The model works in the flange frame without the turn: the Franka flange
  position in robot-base coordinates and the flange rotation. Each client
  reports its pose in its own frame (:data:`RIG_FRAMES`, keyed by the
  ``descriptor`` of the observation). The FR3 rig and RoboLab report the
  Franka hand frame: 103.4 mm along the flange z, rotation
  ``R_flange @ Rz(-45 deg)``. The session converts the poses of the client to
  the flange frame and the commands back to the frame of the client. With
  ``robot_state.q`` in the observation, it checks the converted pose against
  the flange forward kinematics of the joints.

Command contract (rig side): one dict per action row at 15 Hz. The dict holds
these keys:

* ``robot_command``: ``{"type": "cartesian_pos", "pose": [xyz3, R_flat9]}``.
  The rotation is row-major. With a control mode, it also holds ``mode``, the
  control law of the arm.
* ``target_grip``: the absolute gripper value.
* ``timestamp``: the time offset of the row in seconds.

Chunk 0 of a context window gives 21 rows. Every later chunk gives 24 rows.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from praxis.robot_io import (
    ACTION_CHUNK,
    ASPECT_TOL,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    aspect_crop,
    clip_grip,
    denormalize_rows,
    latent_frames,
    pose8_to_10d,
    proprio_normalize,
    rot6d_to_matrix,
    text_encoder_prompt,
    unit_quat_wxyz_to_matrix,
)

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

VIEWS: tuple[str, ...] = ("exterior", "wrist")
# The rig scales each view to fit this size (height, width) before it sends
# it. The session then makes the tiles.
CANVAS_H = 288
CANVAS_W = 1024
ACTION_DIM = 10
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3

# The tile canvas: one tile (height, width) per view, side by side.
TILE_SIZE: tuple[int, int] = (256, 416)
# Frames per resize call of the tile canvas (bounds the float32 memory).
TILE_CHUNK_FRAMES = 32

_C, _S = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
# Rz(-45 deg): the turn of the rig frame about the flange z axis.
FLANGE_TURN = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
# The turn of the model frame (the flange frame): none.
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


# The flange origin with the rotation R_flange @ Rz(-45 deg): the ``ee`` site
# of the sim (praxis_sim.scene, packages/praxis-sim).
TURNED_FLANGE = RigFrame(0.0, _rows(FLANGE_TURN))
# The FR3 ``end_effector`` (the Franka hand frame): 103.4 mm along the flange z.
FR3_HAND = RigFrame(0.1034, _rows(FLANGE_TURN))
# The pose frame of each client, keyed by the ``descriptor`` of the
# observation. The FR3 rig sends no descriptor. Measured with the flange
# forward kinematics of the joints on 1,570 recorded requests.
RIG_FRAMES: dict[str, RigFrame] = {
    "": FR3_HAND,
    "remote.robolab.droid": FR3_HAND,
    # The 2F-85 grasp site, 155 mm along the flange z, in world coordinates
    # with the robot base 0.58 m up.
    "remote.molmo_spaces.droid": RigFrame(0.155, _rows(NO_TURN), (0.0, 0.0, 0.58)),
    "praxis.sim": TURNED_FLANGE,
    # A client that reports the flange itself is RigFrame(0.0, _rows(NO_TURN)).
}
# A converted pose must be this close to the flange forward kinematics (metres).
FK_TOLERANCE = 0.005

# Franka modified DH parameters (a, d, alpha) of joints 1 to 7. The flange is
# 0.107 m past joint 7 along its z axis.
FRANKA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FRANKA_FLANGE_D = 0.107

# Dims of a 10-D row that use the q99 normalization (xyz and grip).
Q99_DIMS = np.array([0, 1, 2, 9])
# The dim of a 10-D row that holds the gripper.
GRIP_DIMS = np.array([9])

POSE_KEYS: tuple[str, ...] = ("ee_pose", "robot_state.ee_pose")
GRIP_KEYS: tuple[str, ...] = ("grip", "robot_state.grip", "gripper")
VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "exterior": ("exterior_image", "image.exterior", "observation.images.side"),
    "wrist": ("wrist_image", "image.wrist", "observation.images.left"),
}

# ---------------------------------------------------------------------------
# Control modes
# ---------------------------------------------------------------------------

# The software impedance law of the DROID demonstrations. kq and kqd are the
# joint gains (7 joints). kx and kxd are the Cartesian gains (3 translation,
# then 3 rotation).
DROID_IMPEDANCE: dict[str, object] = {
    "type": "impedance",
    "kq": [40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0],
    "kqd": [4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0],
    "kx": [750.0, 750.0, 750.0, 15.0, 15.0, 15.0],
    "kxd": [37.0, 37.0, 37.0, 2.0, 2.0, 2.0],
}

# "none" adds no ``mode`` to the commands. The arm then keeps its own law.
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


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def canvas_size(config: dict) -> tuple[int, int]:
    """Return the canvas ``(height, width)`` of a single-arm checkpoint config.

    The config must declare the tile canvas of the views ``exterior`` and
    ``wrist``. The result is ``(tile_height, 2 * tile_width)``. Another canvas
    raises ``ValueError``.
    """
    video = config.get("video", {})
    views = tuple(video.get("views", VIEWS))
    canvas = video.get("canvas")
    if views != VIEWS or canvas != "tiles":
        raise ValueError(
            f"The single-arm rig supports the tile canvas of {VIEWS}. The config "
            f"has views={views} canvas={canvas!r}. This canvas is not "
            "supported; use the single-arm checkpoint."
        )
    th = int(video.get("tile_height", TILE_SIZE[0]))
    tw = int(video.get("tile_width", TILE_SIZE[1]))
    return th, len(VIEWS) * tw


def q99_dims(config: dict) -> np.ndarray:
    """Return the q99-normalized action dims of a single-arm checkpoint config."""
    norm = config.get("normalization", {})
    return np.asarray(norm.get("q99_normed_dims", Q99_DIMS), dtype=np.int64)


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def _pick(obs: dict, candidates: tuple[str, ...]) -> object:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"obs missing any of {candidates}; got keys={list(obs)}")


def extract_pose8(obs: dict) -> np.ndarray:
    """Return the current pose8 ``(8,)``: the last row of the pose and grip stacks.

    The grip can be a scalar, ``(1,)``, a stack ``(T,)`` or a stack
    ``(T, 1)``. The current grip is the last value of a stack.
    """
    ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
    if ee.ndim >= 2:
        ee = ee[-1]
    ee = ee.ravel()
    if ee.shape[0] < 7:
        raise ValueError(f"ee_pose must be (>=7,), got {ee.shape}")
    grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    if grip.ndim >= 2:
        grip = grip[-1]
    elif grip.ndim == 1 and grip.shape[0] > 1:
        grip = grip[-1:]
    grip = grip.ravel()
    return np.concatenate([ee[:7], grip[:1]]).astype(np.float32)


def pose8_history(obs: dict, need: int) -> np.ndarray | None:
    """Return the ``(need, 8)`` pose8 rows of the executed frames, or ``None``.

    The rows are stack rows ``[-(need + 1):-1]`` (the current row is not
    included). ``None`` means that the observation has no pose stack or that
    the stack is too short.
    """
    try:
        ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
        grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    except KeyError:
        return None
    if ee.ndim != 2 or ee.shape[0] < need + 1 or ee.shape[1] < 7:
        return None
    g = grip.reshape(grip.shape[0], -1) if grip.ndim >= 1 else None
    if g is None or g.shape[0] < need + 1:
        return None
    rows = np.concatenate([ee[-(need + 1) : -1, :7], g[-(need + 1) : -1, :1]], axis=1)
    return rows.astype(np.float32)


def latest_frame(obs: dict, view: str) -> np.ndarray:
    """Return the current frame ``(H, W, 3)`` uint8 of one view.

    A missing view raises ``KeyError``.
    """
    img = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES[view]))
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be (H,W,3), got {img.shape}")
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def _tile_view(
    frames: np.ndarray, tile_h: int, tile_w: int, tol: float
) -> torch.Tensor:
    """Return the ``(T, 3, tile_h, tile_w)`` uint8 tile of one view.

    ``frames`` is ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8. The frames get a
    centre crop to the tile aspect (:func:`praxis.robot_io.aspect_crop`, no
    crop when the aspects agree within ``tol``), one bilinear resize without
    antialias in float32, then a round and a clamp to uint8.
    """
    a = np.asarray(frames, np.uint8)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {a.shape}")
    fr = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    cropped = aspect_crop(fr, tile_h, tile_w, tol)
    out = torch.cat(
        [
            torch.nn.functional.interpolate(
                chunk.float(),
                size=(tile_h, tile_w),
                mode="bilinear",
                align_corners=False,
            )
            for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
        ]
    )
    return out.round().clamp(0, 255).to(torch.uint8)


def tile_clip(
    exterior: np.ndarray,
    wrist: np.ndarray | None,
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    tol: float = ASPECT_TOL,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert frames to a ``(1, 3, T, tile_h, 2 * tile_w)`` clip of the tile canvas.

    ``exterior`` and ``wrist`` are ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8
    frames with the same ``T``. Each view becomes one uint8 tile
    (:func:`_tile_view`). The exterior tile goes left and the wrist tile
    right. The clip is float32 with integer values in [0, 255]. The tile
    canvas needs both views: ``wrist=None`` raises ``ValueError``.
    """
    if wrist is None:
        raise ValueError("The tile canvas needs the exterior and the wrist view.")
    tiles = [_tile_view(v, tile_h, tile_w, tol) for v in (exterior, wrist)]
    if tiles[0].shape[0] != tiles[1].shape[0]:
        raise ValueError(
            f"exterior has {tiles[0].shape[0]} frames, wrist has {tiles[1].shape[0]}"
        )
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def _tile_args(
    tile_size: tuple[int, int] | None, aspect_tol: float | None
) -> tuple[int, int, float]:
    th, tw = TILE_SIZE if tile_size is None else tile_size
    return int(th), int(tw), ASPECT_TOL if aspect_tol is None else float(aspect_tol)


def current_clip(
    obs: dict,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
) -> torch.Tensor:
    """Return the tile clip ``(1, 3, 1, H, W)`` of the current frames.

    See :func:`tile_clip`. A missing wrist frame raises ``ValueError``.
    """
    exterior = latest_frame(obs, "exterior")
    try:
        wrist = latest_frame(obs, "wrist")
    except KeyError:
        wrist = None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    return tile_clip(
        exterior[None], None if wrist is None else wrist[None], th, tw, tol, device
    )


def executed_clip(
    obs: dict,
    need: int,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
) -> tuple[torch.Tensor, np.ndarray | None] | None:
    """Return the clip and the pose8 rows of the ``need`` executed raws, or ``None``.

    The frames are stack rows ``[-(need + 1):-1]``. ``None`` means that the
    exterior stack or the wrist stack is missing or too short. The pose rows
    are ``None`` when the observation has no pose stack. They are in the frame
    of the rig.
    """
    ext = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["exterior"]))
    if ext.ndim != 4 or ext.shape[0] < need + 1:
        return None
    ext_chunk = ext[-(need + 1) : -1].astype(np.uint8)
    wrist_chunk = None
    try:
        wrist = np.asarray(_pick(obs, VIEW_KEY_CANDIDATES["wrist"]))
        if wrist.ndim == 4 and wrist.shape[0] >= need + 1:
            wrist_chunk = wrist[-(need + 1) : -1].astype(np.uint8)
    except KeyError:
        wrist_chunk = None
    if wrist_chunk is None:
        return None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    clip = tile_clip(ext_chunk, wrist_chunk, th, tw, tol, device)
    return clip, pose8_history(obs, need)


# ---------------------------------------------------------------------------
# Pose frames
# ---------------------------------------------------------------------------


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


def franka_flange_pose(q: np.ndarray) -> np.ndarray:
    """Return the flange pose ``(4, 4)`` in the robot base frame for the joints ``q (7,)``."""
    m = np.eye(4)
    for (a, d, alpha), theta in zip(
        FRANKA_DH, np.asarray(q, dtype=np.float64), strict=True
    ):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        m = m @ np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    m[:3, 3] += FRANKA_FLANGE_D * m[:3, 2]
    return m


def rig_frame(obs: dict) -> RigFrame:
    """Return the pose frame of the client of ``obs`` (its ``descriptor``).

    An unknown descriptor raises ``ValueError``.
    """
    descriptor = obs.get("descriptor") or ""
    if isinstance(descriptor, bytes):
        descriptor = descriptor.decode("utf-8", "replace")
    frame = RIG_FRAMES.get(str(descriptor))
    if frame is None:
        raise ValueError(
            f"The client {descriptor!r} reports poses in an unknown frame. The "
            f"known clients are {sorted(RIG_FRAMES)}."
        )
    return frame


def check_flange(obs: dict, pose8_model: np.ndarray) -> None:
    """Check a converted pose against the flange forward kinematics.

    The check runs when ``obs`` holds ``robot_state.q`` (the current row of a
    stack). A non-finite pose, joints that are not 7 numbers, or a position
    more than :data:`FK_TOLERANCE` from the flange raise ``ValueError``.
    """
    if "robot_state.q" not in obs:
        return
    q = np.asarray(obs["robot_state.q"])
    if q.ndim == 2:
        q = q[-1]
    if q.shape != (7,) or q.dtype.kind not in "iuf":
        raise ValueError(f"robot_state.q must be 7 joint angles, got {q.shape}")
    xyz = np.asarray(pose8_model[:3], dtype=np.float64)
    err = float(np.linalg.norm(xyz - franka_flange_pose(q)[:3, 3]))
    if not np.isfinite(err) or err > FK_TOLERANCE:
        raise ValueError(
            f"The converted ee_pose is {err * 1e3:.1f} mm from the flange forward "
            "kinematics of robot_state.q. The client frame is not the frame of "
            "its descriptor in RIG_FRAMES."
        )


# ---------------------------------------------------------------------------
# Proprio
# ---------------------------------------------------------------------------


def proprio_rows(pose8: np.ndarray, stats: NormStats) -> np.ndarray:
    """Return the normalized absolute 10-D proprio of pose8 rows (float32)."""
    return proprio_normalize(pose8_to_10d(pose8), stats.proprio_mean, stats.proprio_std)


def proprio_tensors(
    poses: np.ndarray, stats: NormStats, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the expert proprio inputs of the boundary poses (float32).

    ``poses`` holds the ``K`` boundary pose8 rows, current pose last. The
    function returns the current proprio ``(1, 1, 10)`` and the proprio of
    all rows ``(1, K, 10)``.
    """
    p = proprio_rows(poses, stats)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    last = (
        torch.from_numpy(np.ascontiguousarray(p[-1:]))
        .reshape(1, 1, -1)
        .to(device, torch.float32)
    )
    every = (
        torch.from_numpy(np.ascontiguousarray(p))
        .reshape(1, p.shape[0], -1)
        .to(device, torch.float32)
    )
    return last, every


# ---------------------------------------------------------------------------
# Action tokens -> rows -> commands
# ---------------------------------------------------------------------------


def rel10_to_wire(
    rel: np.ndarray,
    pose8_now: np.ndarray,
    dt: float = 1.0 / FPS,
    mode: dict[str, object] | None = None,
    frame: RigFrame = TURNED_FLANGE,
) -> list[dict]:
    """Convert denormalized ``(T, 10)`` relative actions to command dicts.

    ``xyz_abs = xyz_rel + anchor_xyz`` in float64 and
    ``R_abs = R_rel @ R_anchor``. The anchor is the current pose
    ``pose8_now`` in the flange frame of the model. The target then goes to
    ``frame``, the frame of the client (:func:`pose_to_rig`). The gripper is absolute. The pose is
    ``[xyz3, R_flat9]`` float32 with a row-major rotation. Each row carries a
    ``timestamp`` offset ``t * dt``. A ``mode`` goes into every command.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    anchor = np.asarray(pose8_now)
    anchor_xyz = anchor[:3].astype(np.float64)
    r_anchor = unit_quat_wxyz_to_matrix(anchor[3:7])
    out: list[dict] = []
    for t, r in enumerate(rel):
        xyz_abs = r[:3].astype(np.float64) + anchor_xyz
        r_abs = rot6d_to_matrix(r[3:9]) @ r_anchor
        xyz_abs, r_abs = pose_to_rig(xyz_abs, r_abs, frame)
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


def decode_actions(
    pred_action: torch.Tensor,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized rows and the relative actions of the commands of a chunk.

    The function unpacks the ``(1, L, 4 * 10)`` tokens to 21 rows at depth 0
    and 24 rows after (:func:`praxis.robot_io.action_rows`). These normalized
    rows have no clip. The relative actions come from these steps:

    1. Clip ``dims`` to [-1, 1], the range of the training targets.
    2. Apply the inverse q99 map on ``dims``
       (:func:`praxis.robot_io.denormalize_rows`).
    3. Clip the gripper to [0, 1] (:func:`praxis.robot_io.clip_grip`).
    """
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, ACTION_DIM)
    return rows, clip_grip(denormalize_rows(rows, stats, dims), GRIP_DIMS)


def actions_to_wire(
    pred_action: torch.Tensor,
    pose8_now: np.ndarray,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
    fps: float = FPS,
    mode: dict[str, object] | None = None,
) -> list[dict]:
    """Convert predicted normalized action tokens to the command dicts.

    Steps: unpack, clip and denormalize the rows (:func:`decode_actions`),
    then compose them with the current pose ``pose8_now`` in the flange frame
    (:func:`rel10_to_wire`). Gives 21 rows at depth 0 and 24 rows after.
    """
    _rows, rel = decode_actions(pred_action, stats, depth, dims)
    return rel10_to_wire(rel, pose8_now, 1.0 / max(fps, 1e-6), mode)


def wire_to_arrays(wire: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack command dicts to ``(T, 12)`` poses and ``(T,)`` grips (float32)."""
    poses = np.stack([np.asarray(r["robot_command"]["pose"]) for r in wire])
    grips = np.asarray([r["target_grip"] for r in wire], dtype=np.float32)
    return poses.astype(np.float32), grips


# ---------------------------------------------------------------------------
# Serve session state (model free)
# ---------------------------------------------------------------------------


@dataclass
class SingleArmChunkInputs:
    """Model inputs of one served chunk of the single-arm rig.

    ``anchor`` is the 1-latent window anchor and ``cleans`` holds ``depth``
    clean chunk latents (oldest first). ``boundary_poses`` holds the
    ``depth + 1`` pose8 rows at the chunk starts of the window (current pose
    last). ``boundary_poses`` and ``pose8_now`` are in the flange frame of
    the model. ``proprio`` is the normalized current pose ``(1, 1, 10)``
    float32. ``proprio_all`` is the normalized proprio of all boundary poses
    ``(1, depth + 1, 10)`` float32. ``prompt`` is the string for the text
    encoder.
    """

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    boundary_poses: np.ndarray
    proprio: torch.Tensor
    proprio_all: torch.Tensor
    depth: int
    pose8_now: np.ndarray
    chunk_index: int
    prompt: str

    @property
    def refs(self) -> list[torch.Tensor]:
        """Return the reference latents ``[anchor, *cleans]``."""
        return [self.anchor, *self.cleans]

    @property
    def video_shape(self) -> tuple[int, ...]:
        """Return the shape of the noised video chunk (6 latent frames)."""
        shape = list(self.anchor.shape)
        shape[-3] = latent_frames(ACTION_CHUNK)
        return tuple(shape)


class SingleArmServeSession:
    """Observed-context state machine of one single-arm rig session.

    Each call of :meth:`prepare` does these steps:

    1. Apply a prompt override (``obs["task"]`` or ``obs["prompt"]``).
    2. Append the executed frames of the previous chunk to the stream.
    3. Encode the current frame.
    4. Build the window ``[anchor | d cleans]`` and the proprio.

    After the sampler ran, :meth:`finish` converts the actions to commands.
    The sampler uses the seed ``base_seed + chunk_index`` for a chunk.

    The prompt goes to the text encoder verbatim. An override loses its
    leading and trailing white space.

    ``encode`` maps a prepared clip ``(1, 3, T, H, W)`` in [0, 255] to video
    latents ``(1, C, T', H', W')``. The clips are tile clips
    (:func:`tile_clip`, ``tile_size`` and ``aspect_tol`` default to 256x416
    and 0.03). A missing current wrist frame raises ``ValueError``. An
    exterior or wrist stack that is missing or too short for the executed
    chunk restarts the session (a cold start at depth 0).

    ``exec_horizon`` cuts the commands of a chunk to that many rows. A value
    below 24 moves the next observation off the chunk grid of the context
    window. The session then serves every call as a cold start.
    ``max_context_chunks=0`` has the same effect.

    The session converts every pose of the client to the flange frame of the
    model (:func:`pose8_to_model`) and every command back
    (:meth:`rel_to_wire`). The ``descriptor`` of each observation names the
    frame of the client (:func:`rig_frame`). With ``robot_state.q``, the
    session checks the current pose against the flange forward kinematics
    (:func:`check_flange`).
    """

    def __init__(
        self,
        encode: Callable[[torch.Tensor], torch.Tensor],
        stats: NormStats,
        device: torch.device | str,
        prompt: str = "",
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        fps: float = FPS,
        dims: np.ndarray = Q99_DIMS,
        control_mode: str = "none",
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] | None = None,
        aspect_tol: float | None = None,
    ):
        self.tile_size = (
            None if tile_size is None else (int(tile_size[0]), int(tile_size[1]))
        )
        self.aspect_tol = None if aspect_tol is None else float(aspect_tol)
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
        self.control_mode = str(control_mode)
        self.mode = control_mode_payload(self.control_mode)
        self.exec_horizon = None if exec_horizon is None else int(exec_horizon)
        self.max_context_chunks = max(
            0, min(int(max_context_chunks), int(n_chunks) - 1)
        )
        self.stateless = self.max_context_chunks == 0 or (
            self.exec_horizon is not None and self.exec_horizon < ACTION_CHUNK
        )
        self.stream = ContiguousStreamContext(
            action_chunk=ACTION_CHUNK,
            max_context_chunks=max(self.max_context_chunks, 1),
        )
        self.chunk_index = 0
        self.restart_count = 0
        self._boundary_latents: list[torch.Tensor] = []
        self._boundary_poses: list[np.ndarray] = []
        self._cap = self.max_context_chunks + 1
        self.frame = TURNED_FLANGE

    def reset(self) -> None:
        """Start a new episode (chunk counter back to 0, context dropped)."""
        self.chunk_index = 0
        self.restart()

    def restart(self) -> None:
        """Drop the context. The next call is a cold start at depth 0."""
        self.restart_count += 1
        self._boundary_latents = []
        self._boundary_poses = []
        self.stream.reset()

    def prepare(self, obs: dict) -> SingleArmChunkInputs:
        """Update the context from ``obs`` and return the inputs of this chunk."""
        if bool(obs.get("reset_context") or obs.get("__reset_context__")):
            self.restart()
        override = obs.get("task") or obs.get("prompt")
        if override:
            if isinstance(override, bytes):
                override = override.decode("utf-8", "replace")
            new = str(override).strip()
            if new and new != self.prompt:
                self.prompt = new
        frame = rig_frame(obs)
        pose8 = pose8_to_model(extract_pose8(obs), frame)
        check_flange(obs, pose8)
        cur_clip = current_clip(obs, self.device, self.tile_size, self.aspect_tol)
        if self.stateless:
            self.restart()
        elif self._boundary_latents:
            need = self.stream.expected_next_rows
            got = executed_clip(obs, need, self.device, self.tile_size, self.aspect_tol)
            if got is None:
                self.restart()
            else:
                clip, rows = got
                if rows is not None:
                    rows = pose8_to_model(rows, frame)
                self.stream.append_executed(clip, rows)
        cur = self.encode(cur_clip)
        self._boundary_latents = [*self._boundary_latents, cur.detach()][-self._cap :]
        pose = np.asarray(pose8, dtype=np.float32)
        self._boundary_poses = [*self._boundary_poses, pose][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        d = len(cleans)
        af = self.stream.anchor_frame()
        anchor = self._boundary_latents[-1] if af is None else self.encode(af)
        bp = self.stream.boundary_poses() if d else None
        if d == 0:
            poses = pose8[None].astype(np.float32)
        elif bp is not None:
            poses = np.concatenate([bp, pose8[None]], axis=0).astype(np.float32)
        else:
            n = min(d + 1, len(self._boundary_poses))
            poses = np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)
        proprio, proprio_all = proprio_tensors(poses, self.stats, self.device)
        # The commands go to the frame of this observation once it is accepted.
        self.frame = frame
        return SingleArmChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=proprio,
            proprio_all=proprio_all,
            depth=d,
            pose8_now=pose8,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def decode_actions(
        self, inputs: SingleArmChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized rows and the relative actions of the commands.

        See :func:`decode_actions`. The normalized rows have no clip.
        """
        return decode_actions(pred_action, self.stats, inputs.depth, self.dims)

    def rel_to_wire(self, rel: np.ndarray, pose8_now: np.ndarray) -> list[dict]:
        """Convert denormalized ``(T, 10)`` relative actions to commands.

        ``pose8_now`` is the anchor pose in the flange frame of the model
        (``inputs.pose8_now``). The commands carry the timestamps of the
        session fps and its control mode, in the frame of the client of the
        last observation (:func:`rel10_to_wire`).
        """
        dt = 1.0 / max(self.fps, 1e-6)
        return rel10_to_wire(rel, pose8_now, dt, self.mode, self.frame)

    def finish(
        self, inputs: SingleArmChunkInputs, pred_action: torch.Tensor
    ) -> list[dict]:
        """Convert the sampled actions of this chunk to commands and advance.

        With ``exec_horizon`` set, only the first ``exec_horizon`` commands
        come back.
        """
        _rows, rel = self.decode_actions(inputs, pred_action)
        wire = self.rel_to_wire(rel, inputs.pose8_now)
        self.chunk_index += 1
        return wire if self.exec_horizon is None else wire[: self.exec_horizon]
