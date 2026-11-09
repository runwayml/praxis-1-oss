"""The MuJoCo model of the bimanual YAM rig, built with ``mujoco.MjSpec``.

The rig has two I2RT YAM arms with the ``linear_4310`` parallel gripper
(MuJoCo Menagerie ``i2rt_yam/yam_linear_4310.xml``, see
:mod:`praxis_sim.assets`), a table, a top camera above the arms and one
camera on each wrist.

The world frame is the frame of the poses of the bimanual checkpoint (the
frame of ``observation.state.ee_pose16`` in the converted ABC-130k data):

* z is up. x points away from the arm bases, over the table. +y is the left
  of the robot (the side of the left arm).
* The arm bases are at ``LEFT_MOUNT`` and ``RIGHT_MOUNT``, with the identity
  rotation. Forward kinematics of the recorded joints of the converted
  episodes through this model gives the recorded grasp-site poses. The
  public ABC station model (``yam_bimanual_empty.xml`` of
  github.com/amazon-far/abc) has the
  same arm chain, with its mounts at (0.2525, +-0.31, 0.75). Its frame is
  this frame plus (-0.0475, -+0.005, +0.439) m.
* The table top is at the height of the arm bases (``TABLE_TOP_Z``).

Names of a side ``s`` in ``("left", "right")``: joints ``s/joint1`` ..
``s/joint6``, finger joints ``s/left_finger`` and ``s/right_finger``,
actuators ``s/joint1`` .. ``s/joint6`` and ``s/gripper``, the policy site
``s/grasp_site`` and the camera ``wrist_<s>``. The top camera is ``top``.

The policy pose of an arm is the pose of its ``grasp_site`` (between the
finger pads): ``[x, y, z, qw, qx, qy, qz]``. The finger joints are 0 when
the gripper is closed and 0.0475 m when it is open. The grip of the policy is
0 when open and 1 when closed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from praxis_sim import assets
from praxis_sim.scene import CONTROL_HZ, camera_quat

SIDES: tuple[str, ...] = ("left", "right")
LEFT_MOUNT = (0.3, 0.305, 0.311)
RIGHT_MOUNT = (0.3, -0.305, 0.311)
MOUNTS = {"left": LEFT_MOUNT, "right": RIGHT_MOUNT}
TABLE_TOP_Z = 0.311
# Table extent in x and y (the play table of the public station model).
TABLE_X = (0.35, 0.945)
TABLE_Y = (-0.645, 0.645)
ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 7))
FINGER_OPEN = 0.0475  # finger joint at the open end stop, metres
# The observed grip of a closed gripper without an object. The real gripper
# reads about 0.95 at its end stop (converted ABC-130k data: commanded grip
# 1.0 gives an observed median of 0.945).
GRIP_OBS_MAX = 0.95
TOP_CAMERA = "top"
MODEL_VIEWS: tuple[str, ...] = ("top", "wrist_left", "wrist_right")
TABLE_GEOM = "table"
WALL_H = 0.9  # height of the cell walls above the table top
TEXTURE_KEYS: tuple[str, ...] = ("table", "wall_back", "wall_left", "wall_right")
PROFILE_KEYS = frozenset(
    {
        "name",
        "table_top_z",
        "table_x",
        "table_y",
        "wall_h",
        "side_wall_x0",
        "top",
        "wrist",
        "textures",
        "lighting",
    }
)
CAMERA_KEYS = frozenset({"pos", "forward", "right", "intrinsics"})
LIGHT_KEYS = frozenset(
    {"ambient", "head_diffuse", "key", "fill", "table_emission", "wall_emission"}
)


def load_rig_profile(path: str | Path) -> dict:
    """Read a rig profile: a JSON file that describes one real YAM cell.

    A profile replaces the generic cell of the scene. All keys are optional,
    a camera block needs all four of its keys, and unknown keys raise
    ``ValueError``. The only metadata key is ``name``; remove any other
    metadata first:

    * ``name`` (str): a label of the cell; the scene ignores it;
    * ``table_top_z`` (m), ``table_x`` and ``table_y`` (``[min, max]``, m),
      ``wall_h`` (m), ``side_wall_x0`` (m): the table and the walls;
    * ``top``: the top camera (``pos``, ``forward``, ``right``, ``intrinsics``,
      as :class:`TopCameraConfig`);
    * ``wrist``: ``{"left": ..., "right": ...}``, one wrist camera per arm (as
      :class:`WristCameraConfig`);
    * ``textures``: an image per surface (keys of :data:`TEXTURE_KEYS`), paths
      relative to the profile file, projected on the table and the walls;
    * ``lighting``: ``ambient``, ``head_diffuse``, ``key``, ``fill``,
      ``table_emission``, ``wall_emission``.
    """
    path = Path(path)
    prof = json.loads(path.read_text())
    if not isinstance(prof, dict):
        raise ValueError(f"{path}: a rig profile is a JSON object")
    unknown = sorted(set(prof) - PROFILE_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown keys {unknown}; the keys are {sorted(PROFILE_KEYS)}"
        )
    if not isinstance(prof.get("name", ""), str):
        raise ValueError(f"{path}: name is a string, got {prof['name']!r}")
    cameras = {"top": prof["top"]} if "top" in prof else {}
    wrist = prof.get("wrist", {})
    if not isinstance(wrist, dict) or set(wrist) - set(SIDES):
        raise ValueError(f"{path}: wrist holds one camera per side of {SIDES}")
    cameras |= {f"wrist.{side}": cam for side, cam in wrist.items()}
    for name, cam in cameras.items():
        if not isinstance(cam, dict) or set(cam) != CAMERA_KEYS:
            got = sorted(cam) if isinstance(cam, dict) else cam
            raise ValueError(
                f"{path}: the {name} camera needs the keys {sorted(CAMERA_KEYS)}, "
                f"got {got}"
            )
    light = prof.get("lighting", {})
    if not isinstance(light, dict) or set(light) - LIGHT_KEYS:
        raise ValueError(
            f"{path}: lighting takes the keys {sorted(LIGHT_KEYS)}, got {light}"
        )
    tex = prof.get("textures", {})
    if tex and set(tex) != set(TEXTURE_KEYS):
        raise ValueError(
            f"{path}: textures need the keys {TEXTURE_KEYS}, got {sorted(tex)}"
        )
    prof["textures"] = {k: str(path.resolve().parent / v) for k, v in tex.items()}
    missing = [f for f in prof["textures"].values() if not Path(f).is_file()]
    if missing:
        raise ValueError(f"{path}: these texture files do not exist: {missing}")
    return prof


# The median start joints of the converted ABC-130k ZED-X episodes, radians.
HOME: dict[str, tuple[float, ...]] = {
    "left": (-0.029, 1.549, 0.362, 0.929, -0.162, -0.021),
    "right": (0.032, 1.537, 0.35, 0.953, 0.113, 0.03),
}


def joint(side: str, name: str) -> str:
    """Return the full name of the joint (or actuator) ``name`` of one arm."""
    return f"{side}/{name}"


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------


@dataclass
class Intrinsics:
    """Pinhole intrinsics at the full sensor resolution (pixels).

    The ZED-X cameras of the rig record 1920x1200 rectified frames (zero
    distortion). The values are typical values of the camera calibration
    messages of the ABC-130k episodes (they change a little per station).
    """

    width: int = 1920
    height: int = 1200
    fx: float = 750.0
    fy: float = 750.0
    cx: float = 985.0
    cy: float = 620.0


@dataclass
class TopCameraConfig:
    """The fixed top camera, in the world frame.

    ``forward`` is the view direction and ``right`` the image right. The
    calibration (see the note below the class) found them.
    """

    pos: tuple[float, float, float] = (0.2322, 0.037, 1.2015)
    forward: tuple[float, float, float] = (0.555, 0.02, -0.832)
    right: tuple[float, float, float] = (-0.003, -1.0, -0.027)
    intrinsics: Intrinsics = field(default_factory=Intrinsics)


@dataclass
class WristCameraConfig:
    """The camera on each wrist, in the frame of the ``grasp_site`` of its arm.

    ``forward`` is the view direction and ``right`` the image right, both in
    the grasp-site frame (z = the tool axis out of the gripper, y = the axis
    of the fingers). Both arms use the same values.
    """

    pos: tuple[float, float, float] = (-0.0829, 0.0, -0.0411)
    forward: tuple[float, float, float] = (0.459, 0.0, 0.889)
    right: tuple[float, float, float] = (0.0, -1.0, 0.0)
    intrinsics: Intrinsics = field(
        default_factory=lambda: Intrinsics(fx=672.0, fy=720.0, cx=935.0, cy=615.0)
    )


# How the camera constants were found. The intrinsics are typical values of
# the ZED-X calibration messages of the ABC-130k episodes. The poses come
# from a fit of rendered silhouettes to real frames of converted ABC-130k
# episodes: both arms are rendered at the recorded joints, and the camera
# pose minimizes the chamfer distance between the rendered arm (or finger)
# contours and the arm (or finger) masks of the real frames.
#
# * Top camera: 8 stations, 12 to 16 frames each. The pose is the mean of the
#   8 per-station poses. Contour error of the mean pose: 18 px at 1920x1200
#   (12 to 27 px per station). Per-station spread: 1.5 / 2.8 / 1.1 cm (x, y,
#   z), 1 to 4 deg. The camera looks 34 deg from straight down.
# * Wrist cameras: finger masks at 4 gripper openings, 4 stations. Left and
#   right fit within 5 mm and 5 deg of each other, with the same (not
#   mirrored) mount. The constants are the mean of both, symmetric in y.
#   Contour error of the fingers: 8 px. The start point was the camera of the
#   Menagerie model (I2RT bracket), which is 2 cm from the fit.


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class YamSceneConfig:
    """Settings of the bimanual scene."""

    assets_dir: str | None = None  # cache directory (None = the default cache)
    asset_source: str | None = None  # a local Menagerie copy at the pinned commit
    download: bool = True
    substeps: int = 34  # physics steps per 15 Hz tick
    image_width: int = 416  # each model view (the size that the rig client sends)
    image_height: int = 260
    overview_width: int = 0  # the bimanual scene has no overview camera
    overview_height: int = 0
    shadow_size: int = 2048
    msaa_samples: int = 4
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.52, 1.0)
    wall_rgba: tuple[float, float, float, float] = (0.86, 0.82, 0.84, 1.0)
    top: TopCameraConfig = field(default_factory=TopCameraConfig)
    wrist: WristCameraConfig = field(default_factory=WristCameraConfig)
    # A rig profile (:func:`load_rig_profile`): the table, cameras, textures and
    # lighting of one real cell. None = the generic cell of this module.
    rig_profile: str | None = None
    rig: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.rig = load_rig_profile(self.rig_profile) if self.rig_profile else {}
        top = self.rig.get("top")
        if top is not None:
            self.top = TopCameraConfig(
                pos=tuple(top["pos"]),
                forward=tuple(top["forward"]),
                right=tuple(top["right"]),
                intrinsics=Intrinsics(**top["intrinsics"]),
            )

    @property
    def table_top_z(self) -> float:
        """Height of the table top in the world frame, metres."""
        return float(self.rig.get("table_top_z", TABLE_TOP_Z))

    @property
    def table_x(self) -> tuple[float, float]:
        x = self.rig.get("table_x", TABLE_X)
        return (float(x[0]), float(x[1]))

    @property
    def table_y(self) -> tuple[float, float]:
        y = self.rig.get("table_y", TABLE_Y)
        return (float(y[0]), float(y[1]))

    def wrist_camera(self, side: str) -> WristCameraConfig:
        """Return the wrist camera of one arm (the profile's, else ``wrist``)."""
        w = self.rig.get("wrist", {}).get(side)
        if w is None:
            return self.wrist
        return WristCameraConfig(
            pos=tuple(w["pos"]),
            forward=tuple(w["forward"]),
            right=tuple(w["right"]),
            intrinsics=Intrinsics(**w["intrinsics"]),
        )

    def lighting(self, key: str, plain: float, textured: float) -> float:
        """Return a light value of the profile, else the default of the cell."""
        default = textured if self.rig.get("textures") else plain
        return float(self.rig.get("lighting", {}).get(key, default))

    @property
    def timestep(self) -> float:
        """Physics timestep in seconds."""
        return 1.0 / (CONTROL_HZ * int(self.substeps))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _arm_xml(cfg: YamSceneConfig) -> Path:
    root = assets.ensure_assets(
        cache_dir=cfg.assets_dir, source=cfg.asset_source, download=cfg.download
    )
    return root / assets.MODELS["yam"].xml


def _camera_params(
    intr: Intrinsics,
) -> dict[str, list[float] | list[int]]:
    """Return the MjSpec camera fields of pinhole intrinsics.

    MuJoCo puts the principal point at ``principalpixel`` from the image
    centre, x to the right and y up, in pixels of ``resolution``.
    """
    return {
        "resolution": [int(intr.width), int(intr.height)],
        "focal_pixel": [float(intr.fx), float(intr.fy)],
        "principal_pixel": [
            float(intr.width / 2 - intr.cx),
            float(intr.height / 2 - intr.cy),
        ],
        # Any sensor size works with pixel intrinsics; keep the aspect.
        "sensor_size": [0.01, 0.01 * intr.height / intr.width],
    }


def _add_camera(
    body: mujoco.MjsBody,
    name: str,
    pos: np.ndarray,
    quat: np.ndarray,
    intr: Intrinsics,
) -> None:
    cam = body.add_camera(name=name, pos=pos.tolist(), quat=quat.tolist())
    for key, value in _camera_params(intr).items():
        setattr(cam, key, value)


def _set_options(spec: mujoco.MjSpec, cfg: YamSceneConfig) -> None:
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = int(max(cfg.image_width, 640))
    spec.visual.global_.offheight = int(max(cfg.image_height, 480))
    spec.visual.quality.shadowsize = int(cfg.shadow_size)
    spec.visual.quality.offsamples = int(cfg.msaa_samples)
    spec.visual.map.znear = 0.003
    amb = cfg.lighting("ambient", 0.42, 0.25)
    hdif = cfg.lighting("head_diffuse", 0.15, 0.1)
    spec.visual.headlight.ambient = [amb, amb, amb]
    spec.visual.headlight.diffuse = [hdif, hdif, hdif]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]


def _add_arm(
    spec: mujoco.MjSpec, side: str, arm_xml: Path, cfg: YamSceneConfig
) -> None:
    """Attach one arm at its mount and add its wrist camera."""
    arm = mujoco.MjSpec.from_file(str(arm_xml))
    for light in list(arm.lights):
        arm.delete(light)
    # The rig camera replaces the camera of the Menagerie model (same mount).
    for cam in list(arm.cameras):
        arm.delete(cam)
    frame = spec.worldbody.add_frame(pos=list(MOUNTS[side]))
    spec.attach(arm, prefix=f"{side}/", frame=frame)
    # The grasp site is (0, 0, 0.1347) in the link_6 frame, turned -90 deg
    # about z. Put the wrist camera in the link_6 frame.
    link6 = spec.body(f"{side}/link_6")
    site = spec.site(f"{side}/grasp_site")
    q_site = np.asarray(site.quat, dtype=np.float64)  # not normalized in the XML
    r_site = np.zeros(9)
    mujoco.mju_quat2Mat(r_site, q_site / np.linalg.norm(q_site))
    r_site = r_site.reshape(3, 3)
    w = cfg.wrist_camera(side)
    forward = r_site @ np.asarray(w.forward, dtype=np.float64)
    right = r_site @ np.asarray(w.right, dtype=np.float64)
    pos = np.asarray(site.pos, dtype=np.float64) + r_site @ np.asarray(
        w.pos, dtype=np.float64
    )
    _add_camera(link6, f"wrist_{side}", pos, camera_quat(forward, right), w.intrinsics)


def _add_quad(
    spec: mujoco.MjSpec,
    name: str,
    corner: Sequence[float],
    eu: Sequence[float],
    ev: Sequence[float],
    material: str,
) -> None:
    """A visual-only textured quad corner + a * eu + b * ev, texcoord (a, b) -> texture
    (column a, row 1 - b): a projected texture of a rig profile."""
    c, u, v = (np.asarray(x, dtype=np.float64) for x in (corner, eu, ev))
    n = np.cross(u, v)
    n = 0.0005 * n / np.linalg.norm(n)
    verts = [c, c + u, c + u + v, c + v]
    verts += [p - n for p in verts]  # a 0.5 mm slab, so the mesh has a volume
    mesh = spec.add_mesh(name=f"{name}_mesh")
    mesh.uservert = np.concatenate(verts).tolist()
    mesh.usertexcoord = [0, 0, 1, 0, 1, 1, 0, 1] * 2
    mesh.userface = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6]
    mesh.userfacetexcoord = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6]
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_mesh",
        material=material,
        contype=0,
        conaffinity=0,
        group=1,
        density=0,
    )


def _add_room(spec: mujoco.MjSpec, cfg: YamSceneConfig) -> None:
    """Add the floor, the station base under the arms, the table, walls and lights."""
    world = spec.worldbody
    top = cfg.table_top_z
    floor_z = top - 0.75
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.9, 0.9, 0.9],
        rgb2=[0.7, 0.7, 0.72],
        width=256,
        height=1536,
    )
    rgb = list(cfg.table_rgba[:3])
    spec.add_texture(
        name="table_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mujoco.mjtMark.mjMARK_RANDOM,
        random=0.04,
        rgb1=rgb,
        rgb2=rgb,
        markrgb=[min(1.0, c + 0.12) for c in rgb],
        width=512,
        height=512,
    )
    tex = cfg.rig.get("textures", {})
    lit = cfg.rig.get("lighting", {})
    if tex:
        # Images of the real cell projected on the table and the walls (unlit
        # walls, a mostly emissive table so that the shadows still darken it).
        for key in TEXTURE_KEYS:
            spec.add_texture(
                name=f"rig_{key}",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=tex[key],
            )
        spec.add_material(
            name="table",
            textures=["", "rig_table"],
            texrepeat=[1, 1],
            texuniform=False,
            emission=float(lit.get("table_emission", 0.0)),
            specular=0.05,
        )
        for key in ("wall_back", "wall_left", "wall_right"):
            spec.add_material(
                name=key,
                textures=["", f"rig_{key}"],
                texrepeat=[1, 1],
                texuniform=False,
                emission=float(lit.get("wall_emission", 0.6)),
                specular=0.0,
                rgba=[1, 1, 1, 1],
            )
    else:
        spec.add_material(
            name="table", textures=["", "table_tex"], texrepeat=[3, 6], specular=0.15
        )
    spec.add_material(name="wall", rgba=list(cfg.wall_rgba))
    spec.add_material(name="rail", rgba=[0.75, 0.76, 0.78, 1.0], specular=0.4)
    spec.add_material(name="floor", rgba=[0.5, 0.5, 0.52, 1.0])
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[4.0, 4.0, 0.05],
        pos=[0.0, 0.0, floor_z],
        material="floor",
    )
    x0, x1 = cfg.table_x
    y0, y1 = cfg.table_y
    world.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[(x1 - x0) / 2, (y1 - y0) / 2, 0.02],
        pos=[(x0 + x1) / 2, (y0 + y1) / 2, top - 0.02],
        material="table",
        friction=[1.0, 0.005, 0.0001],
        group=3 if tex else 0,
    )
    if tex:
        # texture rows run from the far (back) edge; texcoord b = 0 at -y, a along +x
        _add_quad(
            spec,
            "table_vis",
            [x0, y0, top + 0.0002],
            [x1 - x0, 0, 0],
            [0, y1 - y0, 0],
            "table",
        )
    # The station base under the arm mounts (front rail), up to the table top.
    world.add_geom(
        name="station_base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.08, (y1 - y0) / 2, (top - floor_z) / 2],
        pos=[x0 - 0.08, 0.0, (top + floor_z) / 2],
        material="rail",
    )
    wall_h = float(cfg.rig.get("wall_h", WALL_H))
    for name, pos, size in (
        ("wall_back", [x1 + 0.005, 0.0, top + wall_h / 2], [0.005, y1, wall_h / 2]),
        (
            "wall_left",
            [(x0 + x1) / 2 - 0.1, y1 + 0.005, top + wall_h / 2],
            [(x1 - x0) / 2 + 0.1, 0.005, wall_h / 2],
        ),
        (
            "wall_right",
            [(x0 + x1) / 2 - 0.1, y0 - 0.005, top + wall_h / 2],
            [(x1 - x0) / 2 + 0.1, 0.005, wall_h / 2],
        ),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            material="wall",
            contype=0,
            conaffinity=0,
            group=3 if tex else 0,
        )
    if tex:
        H = wall_h
        sx0 = float(cfg.rig.get("side_wall_x0", x0 - 0.1))
        _add_quad(
            spec,
            "wall_back_vis",
            [x1, y0, top],
            [0, y1 - y0, 0],
            [0, 0, H],
            "wall_back",
        )
        _add_quad(
            spec,
            "wall_left_vis",
            [sx0, y1, top],
            [x1 - sx0, 0, 0],
            [0, 0, H],
            "wall_left",
        )
        _add_quad(
            spec,
            "wall_right_vis",
            [sx0, y0, top],
            [x1 - sx0, 0, 0],
            [0, 0, H],
            "wall_right",
        )
    # Corner posts of the cell frame (visual only).
    for k, (px, py) in enumerate(((x1, y1), (x1, y0), (x0 - 0.1, y1), (x0 - 0.1, y0))):
        world.add_geom(
            name=f"post{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.015, 0.015, wall_h / 2],
            pos=[px, py, top + wall_h / 2],
            material="rail",
            contype=0,
            conaffinity=0,
        )
    world.add_light(
        name="key",
        pos=[0.65, 0.0, top + 1.4],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        cutoff=70,
        exponent=2,
        diffuse=[cfg.lighting("key", 0.5, 0.45)] * 3,
        specular=[0.08, 0.08, 0.08],
        castshadow=cfg.shadow_size > 0,
        softness=0.8,
    )
    world.add_light(
        name="fill",
        pos=[0.0, 0.0, 2.0],
        dir=[0.3, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[cfg.lighting("fill", 0.3, 0.2)] * 3,
        specular=[0.0, 0.0, 0.0],
        castshadow=False,
    )


def _add_cameras(spec: mujoco.MjSpec, cfg: YamSceneConfig) -> None:
    t = cfg.top
    _add_camera(
        spec.worldbody,
        TOP_CAMERA,
        np.asarray(t.pos, dtype=np.float64),
        camera_quat(np.asarray(t.forward), np.asarray(t.right)),
        t.intrinsics,
    )


def build_spec(
    cfg: YamSceneConfig,
    build: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjSpec:
    """Build the ``MjSpec`` of the rig. ``build`` adds the bodies of a task."""
    arm_xml = _arm_xml(cfg)
    spec = mujoco.MjSpec()
    spec.modelname = "praxis_yam_bimanual"
    spec.compiler.degree = False
    _set_options(spec, cfg)
    for side in SIDES:
        _add_arm(spec, side, arm_xml, cfg)
    _add_room(spec, cfg)
    if build is not None:
        build(spec)
    _add_cameras(spec, cfg)
    return spec


def build_model(
    cfg: YamSceneConfig,
    build: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjModel:
    """Build and compile the model of the rig and the task bodies."""
    return build_spec(cfg, build).compile()
