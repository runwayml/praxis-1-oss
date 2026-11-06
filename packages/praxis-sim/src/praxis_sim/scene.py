"""The MuJoCo model of the simulated single-arm rig, built with ``mujoco.MjSpec``.

The scene has these parts:

* A Franka Emika Panda arm without its hand, and a Robotiq 2F-85 gripper on
  the flange (MuJoCo Menagerie models, see :mod:`praxis_sim.assets`). The
  robot base is at the world origin, so the world frame is the base frame of
  the policy.
* A table with its top near the height of the robot base, a floor and walls.
* The objects of the task (boxes with a free joint).
* Two lights and an ambient headlight.
* Three cameras: ``exterior`` (a fixed third-person view), ``wrist`` (a camera
  on the gripper) and ``overview`` (a clean view for people).

Frames (all on the arm body ``link7``):

* ``flange``: the Franka flange (the Denavit-Hartenberg frame 7, moved
  0.107 m along its z axis). z points out of the flange to the gripper.
* ``ee``: the end-effector frame of the policy. Its origin is at the flange
  and its rotation is ``flange @ Rz(-45 deg)``. At the home joints, its x
  axis points 45 deg to the left of the base x axis.
* The gripper fingers open and close along the flange y axis.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from praxis_sim import assets

# ---------------------------------------------------------------------------
# Names and constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 15.0
ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS: tuple[str, ...] = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_PREFIX = "g/"
GRIP_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"
GRIP_JOINT = GRIPPER_PREFIX + "right_driver_joint"
PAD_GEOMS: tuple[str, ...] = tuple(
    GRIPPER_PREFIX + f"{side}_pad{k}" for side in ("left", "right") for k in (1, 2)
)
FLANGE_BODY = "link7"
FLANGE_OFFSET = 0.107  # metres from the link7 origin along its z axis
EE_SITE = "ee"
FLANGE_SITE = "flange"
PINCH_SITE = "pinch"
TABLE_GEOM = "table"
CAMERAS: tuple[str, ...] = ("exterior", "wrist", "overview")
MODEL_VIEWS: tuple[str, ...] = ("exterior", "wrist")

# The ee frame is the flange frame turned -45 deg about the flange z axis.
EE_YAW_IN_FLANGE_DEG = -45.0
# The gripper base turns this angle about the flange z axis so that the
# fingers move along the flange y axis. The Menagerie attachment body is the
# flange turned +135 deg, and the gripper moves its fingers along the x axis
# of its mount.
GRIPPER_YAW_IN_FLANGE_DEG = 90.0
# The pinch point (between the finger pads) on the flange z axis, metres.
# It is the Menagerie pinch site of the gripper (0.145 m from the gripper
# base) plus the mount offsets (0.007 + 0.0038 m).
PINCH_OFFSET = 0.1558


def rot_z(deg: float) -> np.ndarray:
    """Return the 3x3 rotation of ``deg`` degrees about z."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion (float64)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(r, dtype=np.float64).reshape(9))
    return q


def camera_quat(forward: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the wxyz quaternion of a camera from its view and right directions.

    A MuJoCo camera looks along its -z axis. Its x axis points to the right of
    the image and its y axis points up. ``right`` is made orthogonal to
    ``forward``.
    """
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    r = np.asarray(right, dtype=np.float64)
    r = r - np.dot(r, f) * f
    r = r / np.linalg.norm(r)
    up = np.cross(r, f)
    return mat_to_quat(np.stack([r, up, -f], axis=1))


def look_at_quat(
    pos: tuple[float, float, float], target: tuple[float, float, float]
) -> np.ndarray:
    """Return the quaternion of a camera at ``pos`` that looks at ``target``.

    The image horizon stays level (world z is up in the image).
    """
    f = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    right = np.cross(f, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("The camera must not look straight up or down.")
    return camera_quat(f, right)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class WorldCamera:
    """A fixed camera. Position and target in the base frame, metres."""

    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float  # vertical field of view, degrees


@dataclass
class WristCamera:
    """The camera on the gripper. Position in the flange frame, metres.

    The camera is on the flange -x side. It looks along the flange z axis,
    turned ``tilt_deg`` towards the z axis. The image right points along the
    flange -y axis (the finger line), so the fingers show at the bottom.
    """

    pos: tuple[float, float, float] = (-0.08, 0.03, 0.03)
    tilt_deg: float = 20.0
    fovy: float = 62.0


@dataclass
class ObjectSpec:
    """A box with a free joint. ``half_size`` in metres, ``mass`` in kg."""

    name: str
    half_size: tuple[float, float, float] = (0.025, 0.025, 0.025)
    rgba: tuple[float, float, float, float] = (0.85, 0.1, 0.1, 1.0)
    mass: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)


@dataclass
class StaticBox:
    """A box without a joint (a fixture of a task). Position in the base frame."""

    name: str
    pos: tuple[float, float, float]
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.2, 0.35, 0.75, 1.0)
    collide: bool = True


@dataclass
class SceneConfig:
    """Settings of the scene. The YAML ``scene`` block fills it."""

    assets_dir: str | None = None  # cache directory (None = the default cache)
    asset_source: str | None = None  # a local Menagerie copy at the pinned commit
    download: bool = True  # download the missing model files
    substeps: int = 34  # physics steps per 15 Hz tick (timestep 1 / (15 * N))
    gravcomp: bool = True  # gravity compensation of the arm and the gripper
    table_top_z: float = 0.0  # height of the table top above the robot base
    table_center: tuple[float, float] = (0.6, 0.0)
    table_size: tuple[float, float] = (0.9, 1.4)  # x, y extent, metres
    table_rgba: tuple[float, float, float, float] = (0.72, 0.55, 0.38, 1.0)
    image_width: int = 512  # width of the exterior and wrist images
    image_height: int = 288  # height of the exterior and wrist images
    overview_width: int = 640
    overview_height: int = 360
    shadow_size: int = 2048  # shadow map size (0 turns the shadows off)
    # Multisample anti-aliasing samples. With 0, two renders of one state are
    # equal. With 4, some edge pixels change between two renders on a GPU.
    msaa_samples: int = 4
    exterior: WorldCamera = field(
        default_factory=lambda: WorldCamera(
            pos=(0.15, 0.68, 0.48), target=(0.58, 0.0, 0.16), fovy=70.0
        )
    )
    wrist: WristCamera = field(default_factory=WristCamera)
    overview: WorldCamera = field(
        default_factory=lambda: WorldCamera(
            pos=(1.45, -0.85, 0.75), target=(0.45, 0.0, 0.12), fovy=42.0
        )
    )

    @property
    def timestep(self) -> float:
        """Physics timestep in seconds."""
        return 1.0 / (CONTROL_HZ * int(self.substeps))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _model_files(cfg: SceneConfig) -> tuple[Path, Path]:
    """Return the paths of the arm and gripper XML files (download when needed)."""
    root = assets.ensure_assets(
        cache_dir=cfg.assets_dir, source=cfg.asset_source, download=cfg.download
    )
    return root / assets.MODELS["panda"].xml, root / assets.MODELS["robotiq_2f85"].xml


def _set_options(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Set the physics and visual options of the scene."""
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    width = max(cfg.image_width, cfg.overview_width, 640)
    height = max(cfg.image_height, cfg.overview_height, 480)
    spec.visual.global_.offwidth = int(width)
    spec.visual.global_.offheight = int(height)
    spec.visual.quality.shadowsize = int(cfg.shadow_size)
    spec.visual.quality.offsamples = int(cfg.msaa_samples)
    spec.visual.map.znear = 0.005
    spec.visual.headlight.ambient = [0.36, 0.36, 0.36]
    spec.visual.headlight.diffuse = [0.12, 0.12, 0.12]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]


def _add_robot_frames(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add the flange, ee and pinch sites and the wrist camera to ``link7``."""
    link7 = spec.body(FLANGE_BODY)
    flange = np.array([0.0, 0.0, FLANGE_OFFSET])
    link7.add_site(name=FLANGE_SITE, pos=flange.tolist(), size=[0.004, 0, 0], group=5)
    link7.add_site(
        name=EE_SITE,
        pos=flange.tolist(),
        quat=mat_to_quat(rot_z(EE_YAW_IN_FLANGE_DEG)).tolist(),
        size=[0.004, 0, 0],
        group=5,
    )
    link7.add_site(
        name=PINCH_SITE,
        pos=[0.0, 0.0, FLANGE_OFFSET + PINCH_OFFSET],
        size=[0.004, 0, 0],
        group=5,
    )
    w = cfg.wrist
    tilt = math.radians(w.tilt_deg)
    forward = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    right = np.array([0.0, -1.0, 0.0])
    link7.add_camera(
        name="wrist",
        pos=(flange + np.asarray(w.pos, dtype=np.float64)).tolist(),
        quat=camera_quat(forward, right).tolist(),
        fovy=float(w.fovy),
    )


def _add_gripper(spec: mujoco.MjSpec, gripper_xml: Path) -> None:
    """Attach the gripper to the flange, turned so the fingers move along flange y."""
    mount = spec.body(FLANGE_BODY).add_site(
        name="gripper_mount",
        pos=[0.0, 0.0, FLANGE_OFFSET],
        quat=mat_to_quat(rot_z(GRIPPER_YAW_IN_FLANGE_DEG)).tolist(),
    )
    gripper = mujoco.MjSpec.from_file(str(gripper_xml))
    spec.attach(gripper, prefix=GRIPPER_PREFIX, site=mount)


def _add_room(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add the floor, the walls, the robot stand, the table and the lights."""
    world = spec.worldbody
    top = float(cfg.table_top_z)
    floor_z = top - 0.76
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.86, 0.87, 0.88],
        rgb2=[0.62, 0.63, 0.65],
        width=256,
        height=1536,
    )
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.45, 0.45, 0.47],
        rgb2=[0.40, 0.40, 0.42],
        width=512,
        height=512,
    )
    spec.add_material(name="floor", textures=["", "floor_tex"], texrepeat=[8, 8])
    spec.add_material(name="table", rgba=list(cfg.table_rgba), specular=0.1)
    spec.add_material(name="stand", rgba=[0.22, 0.22, 0.24, 1.0])
    spec.add_material(name="wall", rgba=[0.84, 0.83, 0.80, 1.0])
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.05],
        pos=[0.0, 0.0, floor_z],
        material="floor",
    )
    for name, pos, size in (
        ("wall_back", [1.9, 0.0, floor_z + 1.2], [0.02, 2.5, 1.2]),
        ("wall_left", [0.4, 1.6, floor_z + 1.2], [2.5, 0.02, 1.2]),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            material="wall",
            contype=0,
            conaffinity=0,
        )
    stand_h = (0.0 - floor_z) / 2
    world.add_geom(
        name="stand",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.14, 0.14, stand_h],
        pos=[-0.05, 0.0, floor_z + stand_h],
        material="stand",
    )
    cx, cy = cfg.table_center
    sx, sy = cfg.table_size[0] / 2, cfg.table_size[1] / 2
    world.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[sx, sy, 0.02],
        pos=[cx, cy, top - 0.02],
        material="table",
        friction=[1.0, 0.005, 0.0001],
    )
    leg_h = (top - 0.04 - floor_z) / 2
    for i, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        world.add_geom(
            name=f"table_leg{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, leg_h],
            pos=[cx + dx * (sx - 0.05), cy + dy * (sy - 0.05), floor_z + leg_h],
            material="table",
            contype=0,
            conaffinity=0,
        )
    # The key light is low and in front of the robot. Thus the arm shadow
    # falls behind the gripper, not on the table in the wrist view.
    world.add_light(
        name="key",
        pos=[2.0, -0.4, 1.3 + top],
        dir=[-0.7, 0.15, -0.7],
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        cutoff=45,
        exponent=1,
        diffuse=[0.55, 0.55, 0.53],
        specular=[0.1, 0.1, 0.1],
        castshadow=cfg.shadow_size > 0,
        softness=0.6,
    )
    world.add_light(
        name="fill",
        pos=[0.0, 0.0, 2.0],
        dir=[0.2, 0.3, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.42, 0.42, 0.43],
        specular=[0.0, 0.0, 0.0],
        castshadow=False,
    )


def _add_objects(
    spec: mujoco.MjSpec, objects: list[ObjectSpec], fixtures: list[StaticBox]
) -> None:
    """Add the free objects and the fixed boxes of the task."""
    world = spec.worldbody
    for k, obj in enumerate(objects):
        body = world.add_body(name=obj.name, pos=[0.4, -0.3 + 0.15 * k, 0.5])
        body.add_freejoint(name=obj.name + "_joint")
        body.add_geom(
            name=obj.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(obj.half_size),
            rgba=list(obj.rgba),
            mass=float(obj.mass),
            friction=list(obj.friction),
            condim=4,
        )
    for box in fixtures:
        world.add_geom(
            name=box.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(box.half_size),
            pos=list(box.pos),
            rgba=list(box.rgba),
            contype=1 if box.collide else 0,
            conaffinity=1 if box.collide else 0,
        )


def _add_world_cameras(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """Add the exterior and overview cameras."""
    for name, cam in (("exterior", cfg.exterior), ("overview", cfg.overview)):
        spec.worldbody.add_camera(
            name=name,
            pos=list(cam.pos),
            quat=look_at_quat(cam.pos, cam.target).tolist(),
            fovy=float(cam.fovy),
        )


def build_spec(
    cfg: SceneConfig,
    objects: list[ObjectSpec] | None = None,
    fixtures: list[StaticBox] | None = None,
    build: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjSpec:
    """Build the ``MjSpec`` of the scene. Downloads the model files when needed."""
    arm_xml, gripper_xml = _model_files(cfg)
    spec = mujoco.MjSpec.from_file(str(arm_xml))
    spec.modelname = "praxis_single_arm"
    for light in list(spec.lights):
        spec.delete(light)
    _set_options(spec, cfg)
    _add_robot_frames(spec, cfg)
    _add_gripper(spec, gripper_xml)
    _add_room(spec, cfg)
    _add_objects(spec, objects or [], fixtures or [])
    if build is not None:
        build(spec)
    _add_world_cameras(spec, cfg)
    if cfg.gravcomp:
        names = {o.name for o in objects or []}
        for body in spec.bodies:
            if body.name not in names and body.name != "world":
                body.gravcomp = 1.0
    return spec


def build_model(
    cfg: SceneConfig,
    objects: list[ObjectSpec] | None = None,
    fixtures: list[StaticBox] | None = None,
    build: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjModel:
    """Build and compile the scene model. ``build`` adds jointed task bodies."""
    return build_spec(cfg, objects, fixtures, build).compile()
