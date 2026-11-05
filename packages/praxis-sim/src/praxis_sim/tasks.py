"""The task registry of the simulation: prompt, objects, seeded placement, success.

A task gives:

* the default prompt of the policy,
* the objects and fixtures that the scene adds,
* a seeded placement of the objects at each reset,
* a per-tick update that measures progress and latches success,
* a time limit.

Tasks:

* ``lift_cube``: "Pick up the red cube." Success: the cube centre is 5 cm or
  more above its rest height for 0.25 s or more. A cube that falls from 5 cm
  touches the table after 0.10 s (``sqrt(2 * 0.05 / 9.81)``). Thus a cube
  that the arm hits cannot pass.
* ``cube_on_plate``: "Put the red cube on the blue plate." Success: the cube
  is on the plate, the gripper does not touch it, for 0.25 s or more.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from praxis_sim.scene import CONTROL_HZ, PAD_GEOMS, ObjectSpec, StaticBox

CUBE = "red_cube"
PLATE = "blue_plate"


def yaw_quat(yaw: float) -> np.ndarray:
    """Return the wxyz quaternion of a turn of ``yaw`` radians about z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


@dataclass
class TaskConfig:
    """Settings of a task. The YAML ``task`` block fills it."""

    name: str = "lift_cube"
    prompt: str | None = None  # None = the default prompt of the task
    time_limit_s: float | None = None  # None = the default time limit of the task
    cube_size: float = 0.05  # edge length of the cube, metres
    cube_mass: float = 0.1  # kg
    # Ranges of the seeded placement: x, y in the base frame (metres) and yaw.
    cube_x: tuple[float, float] = (0.45, 0.62)
    cube_y: tuple[float, float] = (-0.15, 0.15)
    cube_yaw_deg: tuple[float, float] = (-45.0, 45.0)
    lift_height: float = 0.05  # metres above the rest height
    hold_s: float = 0.25
    table_top_z: float = 0.0


@dataclass
class PickPlace:
    """One pick-and-place step of the scripted oracle (privileged object poses)."""

    pos: np.ndarray  # centre of the object to pick, world frame
    yaw: float  # yaw of the object: the fingers close across two opposite faces
    target: np.ndarray | None = None  # object centre at the place point (None = lift)
    grasp_dz: float = 0.0  # grasp point above the object centre, metres
    place_s: float | None = (
        None  # time of the descent to the place point (None = default)
    )
    approach: float | None = None  # height of the approach above the grasp point
    # Object yaw at the place point (None = keep the grasp yaw). The oracle
    # turns the gripper on the carry. Square objects: any multiple of 90 deg.
    target_yaw: float | None = None


@dataclass
class TaskStatus:
    """Progress of one episode, updated at each tick."""

    ticks: int = 0
    success: bool = False
    success_tick: int | None = None
    hold_ticks: int = 0
    failed: bool = False  # the object left the table
    metrics: dict[str, float] = field(default_factory=dict)


class Task:
    """The base class of a task. One instance holds the state of one episode."""

    name = "task"
    default_prompt = ""
    default_time_limit_s = 10.0

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg
        self.status = TaskStatus()
        self._ids: dict[str, int] = {}

    # -- scene ---------------------------------------------------------------

    @property
    def prompt(self) -> str:
        """Return the prompt of the policy."""
        return self.cfg.prompt if self.cfg.prompt else self.default_prompt

    @property
    def time_limit_s(self) -> float:
        """Return the time limit of an episode in seconds."""
        return float(self.cfg.time_limit_s or self.default_time_limit_s)

    @property
    def max_ticks(self) -> int:
        """Return the time limit of an episode in 15 Hz ticks."""
        return round(self.time_limit_s * CONTROL_HZ)

    @property
    def hold_ticks(self) -> int:
        """Return the number of ticks that a success condition must hold."""
        return max(1, math.ceil(self.cfg.hold_s * CONTROL_HZ - 1e-9))

    def cube_spec(self) -> ObjectSpec:
        """Return the red cube of the task."""
        h = self.cfg.cube_size / 2
        return ObjectSpec(
            name=CUBE,
            half_size=(h, h, h),
            rgba=(0.85, 0.1, 0.1, 1.0),
            mass=self.cfg.cube_mass,
        )

    def objects(self) -> list[ObjectSpec]:
        """Return the free objects that the scene must add."""
        return [self.cube_spec()]

    def fixtures(self) -> list[StaticBox]:
        """Return the fixed boxes that the scene must add."""
        return []

    # -- episode -------------------------------------------------------------

    def bind(self, model: mujoco.MjModel) -> None:
        """Find the ids of the task objects in a compiled model."""
        self._ids = {
            "cube_body": model.body(CUBE).id,
            "cube_geom": model.geom(CUBE).id,
            "cube_qpos": model.joint(CUBE + "_joint").qposadr[0],
            "cube_dof": model.joint(CUBE + "_joint").dofadr[0],
        }
        self._pad_ids = {model.geom(name).id: name for name in PAD_GEOMS}

    def bind_free(self, model: mujoco.MjModel, key: str, name: str) -> None:
        """Find the ids of the free object ``name`` and store them under ``key``."""
        self._ids[key + "_body"] = model.body(name).id
        self._ids[key + "_geom"] = model.geom(name).id
        self._ids[key + "_qpos"] = model.joint(name + "_joint").qposadr[0]
        self._ids[key + "_dof"] = model.joint(name + "_joint").dofadr[0]

    def free_pos(self, data: mujoco.MjData, key: str) -> np.ndarray:
        """Return the world position of the free object stored under ``key``."""
        return data.xpos[self._ids[key + "_body"]].copy()

    def free_yaw(self, data: mujoco.MjData, key: str) -> float:
        """Return the yaw about z of the free object stored under ``key``, radians."""
        r = data.xmat[self._ids[key + "_body"]].reshape(3, 3)
        return math.atan2(r[1, 0], r[0, 0])

    def _set_free_pose(
        self, data: mujoco.MjData, name: str, pos: np.ndarray, yaw: float
    ) -> None:
        adr, dof = self._ids[name + "_qpos"], self._ids[name + "_dof"]
        data.qpos[adr : adr + 3] = pos
        data.qpos[adr + 3 : adr + 7] = yaw_quat(yaw)
        data.qvel[dof : dof + 6] = 0.0

    def sample_cube(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Draw the cube position (at rest on the table) and yaw."""
        c = self.cfg
        x = rng.uniform(*c.cube_x)
        y = rng.uniform(*c.cube_y)
        yaw = math.radians(rng.uniform(*c.cube_yaw_deg))
        z = c.table_top_z + c.cube_size / 2 + 1e-4
        return np.array([x, y, z]), yaw

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        """Put the objects at their seeded start poses."""
        pos, yaw = self.sample_cube(rng)
        self._set_free_pose(data, "cube", pos, yaw)

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Start an episode after the reset and the settle time."""
        self.status = TaskStatus()
        self.rest_z = self.cube_pos(data)[2]
        self.status.metrics = {
            "min_reach_m": math.inf,
            "max_lift_m": 0.0,
            "max_grip": 0.0,
            "grasp_ticks": 0.0,
        }

    def cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Return the world position of the cube centre."""
        return data.xpos[self._ids["cube_body"]].copy()

    def cube_yaw(self, data: mujoco.MjData) -> float:
        """Return the yaw of the cube about z, radians."""
        r = data.xmat[self._ids["cube_body"]].reshape(3, 3)
        return math.atan2(r[1, 0], r[0, 0])

    def pad_contacts(self, data: mujoco.MjData, geom: int) -> tuple[bool, bool]:
        """Return (left pad touches ``geom``, right pad touches ``geom``)."""
        left = right = False
        for i in range(data.ncon):
            con = data.contact[i]
            pair = (con.geom1, con.geom2)
            if geom not in pair:
                continue
            other = pair[1] if pair[0] == geom else pair[0]
            name = self._pad_ids.get(other)
            if name is None:
                continue
            if "left" in name:
                left = True
            else:
                right = True
        return left, right

    def on_table(self, pos: np.ndarray, table: tuple[np.ndarray, np.ndarray]) -> bool:
        """Return ``True`` when ``pos`` is above the table top and not below it."""
        lo, hi = table
        inside = bool(np.all(pos[:2] >= lo) and np.all(pos[:2] <= hi))
        return inside and pos[2] > self.cfg.table_top_z - 0.02

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        """Return ``True`` when the success condition is true at this tick."""
        raise NotImplementedError

    def place_target(self) -> np.ndarray | None:
        """Return the point where the cube must go down, or ``None`` (no place step)."""
        return None

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        """Return the pick-and-place steps of the scripted oracle, in order."""
        return [
            PickPlace(self.cube_pos(data), self.cube_yaw(data), self.place_target())
        ]

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add bodies with joints (for example a button) to the scene spec."""

    def progress(self, data: mujoco.MjData) -> float:
        """Return the progress of the task in [0, 1] at this tick (1 = done)."""
        lift = (self.cube_pos(data)[2] - self.rest_z) / self.cfg.lift_height
        return float(np.clip(lift, 0.0, 1.0))

    def update(
        self,
        data: mujoco.MjData,
        pinch: np.ndarray,
        grip: float,
        table: tuple[np.ndarray, np.ndarray],
    ) -> TaskStatus:
        """Measure one tick and latch success after the hold time."""
        s = self.status
        s.ticks += 1
        cube = self.cube_pos(data)
        m = s.metrics
        m["min_reach_m"] = min(m["min_reach_m"], float(np.linalg.norm(cube - pinch)))
        m["max_lift_m"] = max(m["max_lift_m"], float(cube[2] - self.rest_z))
        m["max_grip"] = max(m["max_grip"], float(grip))
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        m["grasp_ticks"] += float(left and right)
        if (
            not self.on_table(cube, table)
            and cube[2] - self.rest_z < self.cfg.lift_height
        ):
            s.failed = True
        if self.condition(data, grip):
            s.hold_ticks += 1
        else:
            s.hold_ticks = 0
        if not s.success and s.hold_ticks >= self.hold_ticks:
            s.success = True
            s.success_tick = s.ticks
        m["progress"] = (
            1.0 if s.success else max(m.get("progress", 0.0), self.progress(data))
        )
        return s


class LiftCube(Task):
    """Pick up the red cube and hold it 5 cm or more above its rest height."""

    name = "lift_cube"
    default_prompt = "Pick up the red cube."
    default_time_limit_s = 11.0  # 165 ticks: one chunk of 21 rows and six of 24

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return bool(self.cube_pos(data)[2] - self.rest_z >= self.cfg.lift_height)


class CubeOnPlate(Task):
    """Put the red cube on a thin blue plate and let it go."""

    name = "cube_on_plate"
    default_prompt = "Put the red cube on the blue plate."
    default_time_limit_s = 16.0  # 237 ticks: one chunk of 21 rows and nine of 24
    plate_half = (0.07, 0.07, 0.006)
    min_distance = 0.18  # metres between the cube and the plate centre

    def fixtures(self) -> list[StaticBox]:
        # The plate is a fixed box. ``place`` moves it at each reset through
        # the geom position of the compiled model.
        return [
            StaticBox(
                name=PLATE,
                pos=(0.55, 0.2, self.cfg.table_top_z + self.plate_half[2]),
                half_size=self.plate_half,
                rgba=(0.15, 0.3, 0.8, 1.0),
            )
        ]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self._model = model
        self._ids["plate_geom"] = model.geom(PLATE).id

    def plate_pos(self) -> np.ndarray:
        """Return the centre of the plate top in the world frame."""
        p = self._model.geom_pos[self._ids["plate_geom"]].copy()
        p[2] += self.plate_half[2]
        return p

    def place_target(self) -> np.ndarray | None:
        """Return the cube centre on the plate."""
        return self.plate_pos() + np.array([0.0, 0.0, self.cfg.cube_size / 2])

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        c = self.cfg
        for _ in range(100):
            pos, yaw = self.sample_cube(rng)
            plate = np.array(
                [
                    rng.uniform(*c.cube_x),
                    rng.uniform(c.cube_y[0] - 0.05, c.cube_y[1] + 0.05),
                ]
            )
            if np.linalg.norm(plate - pos[:2]) >= self.min_distance:
                break
        self._model.geom_pos[self._ids["plate_geom"]] = [
            plate[0],
            plate[1],
            c.table_top_z + self.plate_half[2],
        ]
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def progress(self, data: mujoco.MjData) -> float:
        d0 = float(np.linalg.norm(self.start_xy - self.plate_pos()[:2]))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - self.plate_pos()[:2]))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        plate = self.plate_pos()
        h = self.cfg.cube_size / 2
        inside = bool(
            np.all(np.abs(cube[:2] - plate[:2]) <= np.array(self.plate_half[:2]))
        )
        resting = abs(cube[2] - (plate[2] + h)) < 0.01
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return inside and resting and not (left or right)


TASKS: dict[str, Callable[[TaskConfig], Task]] = {
    LiftCube.name: LiftCube,
    CubeOnPlate.name: CubeOnPlate,
}


def _register_suite() -> None:
    from praxis_sim.suite import SUITE_TASKS
    from praxis_sim.suite_hard import HARD_TASKS
    from praxis_sim.suite_insert import INSERT_TASKS
    from praxis_sim.suite_multi import MULTI_TASKS
    from praxis_sim.suite_place import PLACE_TASKS

    TASKS.update(SUITE_TASKS)
    TASKS.update(PLACE_TASKS)
    TASKS.update(INSERT_TASKS)
    TASKS.update(HARD_TASKS)
    TASKS.update(MULTI_TASKS)


def make_task(cfg: TaskConfig) -> Task:
    """Return a new task instance from the registry."""
    _register_suite()
    if cfg.name not in TASKS:
        raise ValueError(
            f"Unknown task {cfg.name!r}. The known tasks are {sorted(TASKS)}."
        )
    return TASKS[cfg.name](cfg)
