"""Placement tasks with fixtures: a cube in a tray, a cube on one of several colour squares.

* ``cube_in_tray``: "Put the red cube in the tray." A square open tray (a
  floor plate and four low walls on one fixed body) is on the table. Success:
  the cube rests on the tray floor near the tray centre, the finger pads do
  not touch it. The inside side of the tray (8 cm) is the difficulty knob.
* ``cube_on_color``: "Put the red cube on the <colour> square." Three flat
  colour squares (blue, yellow, purple, 6 cm) are on the table. The seed chooses
  the target colour and the layout. Success: the cube rests on the target
  square. The cube at rest on another square fails the episode and gets no
  progress.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from praxis.sim.scene import StaticBox
from praxis.sim.tasks import PickPlace, Task, TaskConfig, TaskStatus

TRAY = "tray"
SQUARE_COLOURS: dict[str, tuple[float, float, float, float]] = {
    "blue": (0.15, 0.3, 0.85, 1.0),
    "yellow": (0.95, 0.82, 0.1, 1.0),
    "purple": (0.55, 0.18, 0.75, 1.0),
}
# Progress of an episode without success stays below 1.
PROGRESS_CAP = 0.95


def _rect_distance(p: np.ndarray, centre: np.ndarray, half: float) -> float:
    """Return the distance in xy from ``p`` to an axis-aligned square."""
    d = np.maximum(np.abs(p[:2] - centre[:2]) - half, 0.0)
    return float(np.linalg.norm(d))


class CubeInTray(Task):
    """Put the red cube in a square open tray and let it go."""

    name = "cube_in_tray"
    default_prompt = "Put the red cube in the tray."
    default_time_limit_s = 14.2  # 213 ticks: one chunk of 21 rows and eight of 24
    # Inside side of the tray, metres: the difficulty knob (16, 12, 10 or 8 cm).
    # With 12 cm and 16 s the checkpoint of the suite solves 7 of 8 seeds.
    inside = 0.08
    wall_height = 0.03  # above the floor
    wall_thickness = 0.006
    floor_thickness = 0.004
    rgba = (0.28, 0.29, 0.32, 1.0)
    min_distance = 0.21  # metres between the cube and the tray centre
    tray_x = (0.47, 0.62)
    tray_y = (-0.18, 0.18)

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add the tray: a floor plate and four wall boxes on one fixed body."""
        s, t, h = self.inside / 2, self.wall_thickness, self.wall_height
        f = self.floor_thickness
        body = spec.worldbody.add_body(name=TRAY, pos=[0.55, 0.15, 0.0])
        body.add_geom(
            name=f"{TRAY}_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[s + t, s + t, f / 2],
            pos=[0, 0, f / 2],
            rgba=list(self.rgba),
        )
        z = f + h / 2
        walls = (
            ("px", [s + t / 2, 0.0, z], [t / 2, s + t, h / 2]),
            ("nx", [-s - t / 2, 0.0, z], [t / 2, s + t, h / 2]),
            ("py", [0.0, s + t / 2, z], [s, t / 2, h / 2]),
            ("ny", [0.0, -s - t / 2, z], [s, t / 2, h / 2]),
        )
        for tag, pos, size in walls:
            body.add_geom(
                name=f"{TRAY}_wall_{tag}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                rgba=list(self.rgba),
            )

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self._model = model
        self._tray_body = model.body(TRAY).id
        self.tray_xy = np.array([0.55, 0.15])

    def move_tray(self, xy: np.ndarray) -> None:
        """Put the tray at ``xy`` (base frame), on the table."""
        self._model.body_pos[self._tray_body] = [xy[0], xy[1], self.cfg.table_top_z]
        self.tray_xy = np.asarray(xy, dtype=np.float64).copy()

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        for _ in range(200):
            pos, yaw = self.sample_cube(rng)
            tray = np.array([rng.uniform(*self.tray_x), rng.uniform(*self.tray_y)])
            if np.linalg.norm(tray - pos[:2]) >= self.min_distance:
                break
        self.move_tray(tray)
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def rest_height(self) -> float:
        """Return the height of the cube centre at rest on the tray floor."""
        return self.cfg.table_top_z + self.floor_thickness + self.cfg.cube_size / 2

    def place_target(self) -> np.ndarray | None:
        return np.array([self.tray_xy[0], self.tray_xy[1], self.rest_height()])

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        steps = super().oracle_steps(data)
        steps[0].target_yaw = 0.0  # cube faces along the tray walls
        return steps

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        near = np.abs(cube[:2] - self.tray_xy) <= self.inside / 2 - 0.02
        resting = abs(cube[2] - self.rest_height()) < 0.006
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return bool(np.all(near) and resting and not (left or right))

    def progress(self, data: mujoco.MjData) -> float:
        d0 = float(np.linalg.norm(self.start_xy - self.tray_xy))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - self.tray_xy))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, PROGRESS_CAP))


class CubeOnColor(Task):
    """Put the red cube on the colour square that the seed names."""

    name = "cube_on_color"
    # Difficulty knobs: the square size, the gaps, the target distance and the
    # time limit. With 8 cm squares, free gaps and 16 s the checkpoint of the
    # suite solves 8 of 8 seeds.
    default_time_limit_s = 14.2  # 213 ticks: one chunk of 21 rows and eight of 24
    half = 0.03  # half side of a square, metres
    thickness = 0.004
    min_gap = 0.03  # metres between the edges of two squares
    max_gap: float | None = 0.04  # largest edge gap from the target to a distractor
    cube_gap = 0.045  # metres between the cube and the nearest square edge
    min_distance = 0.18  # metres between the cube and the target centre
    tolerance = 0.03  # cube centre to target centre in x and y, metres
    square_x = (0.44, 0.65)
    square_y = (-0.2, 0.2)

    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.colours = list(SQUARE_COLOURS)
        self.target = 0
        self.wrong = False
        self.wrong_ticks = 0

    @property
    def colour(self) -> str:
        """Return the name of the target colour."""
        return self.colours[self.target]

    @property
    def prompt(self) -> str:
        if self.cfg.prompt:
            return self.cfg.prompt
        return f"Put the red cube on the {self.colour} square."

    def fixtures(self) -> list[StaticBox]:
        # ``place`` moves the squares at each reset through the geom positions.
        z = self.cfg.table_top_z + self.thickness / 2
        return [
            StaticBox(
                name=f"{colour}_square",
                pos=(0.55, -0.15 + 0.15 * k, z),
                half_size=(self.half, self.half, self.thickness / 2),
                rgba=rgba,
            )
            for k, (colour, rgba) in enumerate(SQUARE_COLOURS.items())
        ]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self._model = model
        self._square_geoms = [model.geom(f"{c}_square").id for c in self.colours]

    def square_xy(self, k: int) -> np.ndarray:
        """Return the centre of square ``k`` in xy."""
        return self._model.geom_pos[self._square_geoms[k]][:2].copy()

    def _layout_ok(self, cube: np.ndarray, xy: list[np.ndarray]) -> bool:
        reach = self.cube_gap + self.cfg.cube_size / np.sqrt(2)
        if any(_rect_distance(cube, p, self.half) < reach for p in xy):
            return False
        if np.linalg.norm(xy[self.target] - cube[:2]) < self.min_distance:
            return False
        gaps = {}
        for i in range(3):
            for j in range(i + 1, 3):
                gap = float(np.max(np.abs(xy[i] - xy[j]))) - 2 * self.half
                if gap < self.min_gap:
                    return False
                gaps[i, j] = gaps[j, i] = gap
        if self.max_gap is not None:
            near = min(g for (i, _), g in gaps.items() if i == self.target)
            if near > self.max_gap:
                return False
        return True

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        self.target = int(rng.integers(3))
        for _ in range(20000):
            pos, yaw = self.sample_cube(rng)
            xy = [
                np.array([rng.uniform(*self.square_x), rng.uniform(*self.square_y)])
                for _ in range(3)
            ]
            if self._layout_ok(pos, xy):
                break
        z = self.cfg.table_top_z + self.thickness / 2
        for g, p in zip(self._square_geoms, xy, strict=True):
            self._model.geom_pos[g] = [p[0], p[1], z]
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().start(model, data)
        self.wrong = False
        self.wrong_ticks = 0
        self.status.metrics["wrong_square"] = 0.0

    def rest_height(self) -> float:
        """Return the height of the cube centre at rest on a square."""
        return self.cfg.table_top_z + self.thickness + self.cfg.cube_size / 2

    def place_target(self) -> np.ndarray | None:
        p = self.square_xy(self.target)
        return np.array([p[0], p[1], self.rest_height()])

    def free(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when no finger pad touches the cube."""
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return not (left or right)

    def on_square(self, data: mujoco.MjData, k: int, tol: float) -> bool:
        """Return ``True`` when the cube rests on square ``k`` within ``tol``."""
        cube = self.cube_pos(data)
        near = np.all(np.abs(cube[:2] - self.square_xy(k)) <= tol)
        resting = abs(cube[2] - self.rest_height()) < 0.005
        return bool(near and resting and self.free(data))

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return not self.wrong and self.on_square(data, self.target, self.tolerance)

    def progress(self, data: mujoco.MjData) -> float:
        if self.wrong:
            return 0.0
        goal = self.square_xy(self.target)
        d0 = float(np.linalg.norm(self.start_xy - goal))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - goal))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, PROGRESS_CAP))

    def update(
        self,
        data: mujoco.MjData,
        pinch: np.ndarray,
        grip: float,
        table: tuple[np.ndarray, np.ndarray],
    ) -> TaskStatus:
        """Measure one tick. The cube at rest on another square fails the episode."""
        s = super().update(data, pinch, grip, table)
        others = [k for k in range(3) if k != self.target]
        if not s.success and any(self.on_square(data, k, self.half) for k in others):
            self.wrong_ticks += 1
        else:
            self.wrong_ticks = 0
        if self.wrong_ticks >= self.hold_ticks:
            self.wrong = True
            s.failed = True
            s.metrics["wrong_square"] = 1.0
        if self.wrong:
            s.metrics["progress"] = 0.0
        return s


PLACE_TASKS: dict[str, Callable[[TaskConfig], Task]] = {
    CubeInTray.name: CubeInTray,
    CubeOnColor.name: CubeOnColor,
}
