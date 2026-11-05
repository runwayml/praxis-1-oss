"""Tasks of the DROID task families, for an in-distribution evaluation suite.

The tasks cover the most frequent instruction families of the DROID data:
put an object in a container, put it on a surface, take it out of a
container, move it to the left or the right, stack it, and press a button.
All objects are MuJoCo primitives. A container is a disc and a ring of wall
boxes, because MuJoCo collides only convex shapes.

Each task gives a seeded start state, a prompt in the style of the DROID
instructions, a physics success check and a dense progress value in [0, 1]
(``metrics["progress"]``). "Left" is +y of the robot base, as in DROID.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from praxis_sim.scene import ObjectSpec, StaticBox
from praxis_sim.tasks import Task, TaskConfig, TaskStatus

BOWL = "bowl"
TARGET = "green_block"
BUTTON = "button"
WALLS = 16


class _Container(Task):
    """A task with a container on the table (a bowl): one body without a joint."""

    radius = 0.08
    wall_height = 0.04
    rgba = (0.92, 0.92, 0.88, 1.0)
    wall_thickness = 0.006

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add the bowl: a base disc and a ring of wall boxes on one fixed body."""
        body = spec.worldbody.add_body(name=BOWL, pos=[0.55, 0.15, 0.0])
        body.add_geom(
            name=f"{BOWL}_base",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.radius + self.wall_thickness / 2, 0.003, 0],
            pos=[0, 0, 0.003],
            rgba=list(self.rgba),
        )
        seg = self.radius * math.tan(math.pi / WALLS) * 1.05
        for k in range(WALLS):
            a = 2 * math.pi * k / WALLS
            body.add_geom(
                name=f"{BOWL}_wall{k:02d}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_thickness / 2, seg, self.wall_height / 2],
                pos=[
                    self.radius * math.cos(a),
                    self.radius * math.sin(a),
                    self.wall_height / 2,
                ],
                quat=[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)],
                rgba=list(self.rgba),
            )

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self._model = model
        self._bowl_body = model.body(BOWL).id
        self.bowl_xy = np.array([0.55, 0.15])

    def move_bowl(self, xy: np.ndarray) -> None:
        """Put the container at ``xy`` (base frame), on the table."""
        self._model.body_pos[self._bowl_body] = [xy[0], xy[1], self.cfg.table_top_z]
        self.bowl_xy = np.asarray(xy, dtype=np.float64).copy()

    def in_bowl(self, pos: np.ndarray) -> bool:
        """Return ``True`` when ``pos`` is inside the container walls."""
        inner = self.radius - self.cfg.cube_size / 2
        return bool(np.linalg.norm(pos[:2] - self.bowl_xy) < inner)

    def free(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when no finger pad touches the cube."""
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return not (left or right)


class BlockInBowl(_Container):
    """Put the red block in the bowl and let it go."""

    name = "block_in_bowl"
    default_prompt = "Put the red block in the bowl."
    default_time_limit_s = 20.6  # 309 ticks: one chunk of 21 rows and twelve of 24

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        c = self.cfg
        for _ in range(200):
            pos, yaw = self.sample_cube(rng)
            bowl = np.array([rng.uniform(0.48, 0.62), rng.uniform(-0.18, 0.18)])
            if np.linalg.norm(bowl - pos[:2]) >= 0.2:
                break
        self.move_bowl(bowl)
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()
        _ = c

    def place_target(self) -> np.ndarray | None:
        z = self.cfg.table_top_z + 0.006 + self.cfg.cube_size / 2
        return np.array([self.bowl_xy[0], self.bowl_xy[1], z])

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        low = cube[2] - self.rest_z < 0.03
        return self.in_bowl(cube) and low and self.free(data)

    def progress(self, data: mujoco.MjData) -> float:
        d0 = float(np.linalg.norm(self.start_xy - self.bowl_xy))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - self.bowl_xy))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))


class BlockOutOfBowl(_Container):
    """Take the red block out of the bowl and put it on the table."""

    name = "block_out_of_bowl"
    default_prompt = "Remove the red block from the bowl."
    default_time_limit_s = 20.6
    out_distance = 0.13  # metres from the bowl centre

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        bowl = np.array([rng.uniform(0.48, 0.60), rng.uniform(-0.12, 0.12)])
        self.move_bowl(bowl)
        off = rng.uniform(-0.015, 0.015, size=2)
        z = self.cfg.table_top_z + 0.006 + self.cfg.cube_size / 2 + 1e-4
        yaw = math.radians(rng.uniform(-45, 45))
        pos = np.array([bowl[0] + off[0], bowl[1] + off[1], z])
        self._set_free_pose(data, "cube", pos, yaw)
        side = 1.0 if bowl[1] <= 0 else -1.0
        self.out_xy = bowl + np.array([0.0, side * 0.17])

    def place_target(self) -> np.ndarray | None:
        z = self.cfg.table_top_z + self.cfg.cube_size / 2
        return np.array([self.out_xy[0], self.out_xy[1], z])

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        far = np.linalg.norm(cube[:2] - self.bowl_xy) > self.out_distance
        on_table = abs(cube[2] - (self.cfg.table_top_z + self.cfg.cube_size / 2)) < 0.01
        return bool(far and on_table and self.free(data))

    def progress(self, data: mujoco.MjData) -> float:
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - self.bowl_xy))
        return float(np.clip(d / self.out_distance, 0.0, 1.0))


class MoveBlock(Task):
    """Move the red block 12 cm or more to the left or the right (seed chooses)."""

    name = "move_block"
    default_time_limit_s = 16.0  # 237 ticks
    distance = 0.12

    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.direction = 1.0

    @property
    def prompt(self) -> str:
        if self.cfg.prompt:
            return self.cfg.prompt
        side = "left" if self.direction > 0 else "right"
        return f"Move the red block to the {side}."

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        self.direction = 1.0 if rng.integers(2) == 0 else -1.0
        pos, yaw = self.sample_cube(rng)
        pos[0] = rng.uniform(0.48, 0.60)
        pos[1] = rng.uniform(-0.06, 0.06)
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def place_target(self) -> np.ndarray | None:
        z = self.cfg.table_top_z + self.cfg.cube_size / 2
        xy = self.start_xy + np.array([0.0, self.direction * (self.distance + 0.03)])
        return np.array([xy[0], xy[1], z])

    def moved(self, data: mujoco.MjData) -> float:
        return float(self.direction * (self.cube_pos(data)[1] - self.start_xy[1]))

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        on_table = abs(cube[2] - (self.cfg.table_top_z + self.cfg.cube_size / 2)) < 0.01
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return self.moved(data) >= self.distance and on_table and not (left or right)

    def progress(self, data: mujoco.MjData) -> float:
        return float(np.clip(self.moved(data) / self.distance, 0.0, 1.0))


class StackBlock(Task):
    """Put the red block on top of the green block."""

    name = "stack_block"
    default_prompt = "Put the red block on top of the green block."
    default_time_limit_s = 20.6
    target_half = 0.03

    def objects(self) -> list[ObjectSpec]:
        h = self.target_half
        return [
            self.cube_spec(),
            ObjectSpec(
                name=TARGET, half_size=(h, h, h), rgba=(0.15, 0.6, 0.2, 1.0), mass=0.4
            ),
        ]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self._ids["target_body"] = model.body(TARGET).id
        self._ids["target_qpos"] = model.joint(TARGET + "_joint").qposadr[0]
        self._ids["target_dof"] = model.joint(TARGET + "_joint").dofadr[0]

    def target_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self._ids["target_body"]].copy()

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        for _ in range(200):
            pos, yaw = self.sample_cube(rng)
            t = np.array([rng.uniform(0.48, 0.62), rng.uniform(-0.15, 0.15)])
            if np.linalg.norm(t - pos[:2]) >= 0.15:
                break
        z = self.cfg.table_top_z + self.target_half + 1e-4
        self._set_free_pose(
            data, "target", np.array([t[0], t[1], z]), rng.uniform(-0.6, 0.6)
        )
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()
        self.target_xy = t

    def place_target(self) -> np.ndarray | None:
        z = self.cfg.table_top_z + 2 * self.target_half + self.cfg.cube_size / 2
        return np.array([self.target_xy[0], self.target_xy[1], z])

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube, tgt = self.cube_pos(data), self.target_pos(data)
        above = (
            abs(cube[2] - (tgt[2] + self.target_half + self.cfg.cube_size / 2)) < 0.012
        )
        centred = np.linalg.norm(cube[:2] - tgt[:2]) < self.target_half
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return bool(above and centred and not (left or right))

    def progress(self, data: mujoco.MjData) -> float:
        tgt = self.target_pos(data)
        d0 = float(np.linalg.norm(self.start_xy - self.target_xy))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - tgt[:2]))
        reach = float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))
        lift = float(
            np.clip(
                (self.cube_pos(data)[2] - self.rest_z) / (2 * self.target_half), 0, 1
            )
        )
        return (
            0.5 * reach + 0.5 * min(reach, lift)
            if d > self.target_half
            else 0.5 + 0.5 * lift
        )


class PressButton(Task):
    """Press the red button (a spring cap on a fixed base) down by 6 mm or more."""

    name = "press_button"
    default_prompt = "Press the red button."
    default_time_limit_s = 11.0
    travel = 0.006
    press_depth = 0.010  # how far the scripted oracle pushes below the cap top

    def objects(self) -> list[ObjectSpec]:
        return []

    def fixtures(self) -> list[StaticBox]:
        return [
            StaticBox(
                name=f"{BUTTON}_base",
                pos=(0.55, 0.0, 0.015),
                half_size=(0.04, 0.04, 0.015),
                rgba=(0.35, 0.35, 0.35, 1.0),
            )
        ]

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add the spring cap of the button (a slide joint along z)."""
        body = spec.worldbody.add_body(name=BUTTON, pos=[0.55, 0.0, 0.03])
        body.add_joint(
            name=BUTTON + "_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            range=[-0.012, 0.0],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=300.0,
            damping=3.0,
        )
        body.add_geom(
            name=BUTTON,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.02, 0.0075, 0],
            pos=[0, 0, 0.0075],
            rgba=[0.85, 0.1, 0.1, 1.0],
            mass=0.02,
            condim=4,
        )

    def bind(self, model: mujoco.MjModel) -> None:
        self._ids = {
            "button_body": model.body(BUTTON).id,
            "button_geom": model.geom(BUTTON).id,
            "button_qpos": model.joint(BUTTON + "_joint").qposadr[0],
            "base_geom": model.geom(f"{BUTTON}_base").id,
        }
        from praxis_sim.scene import PAD_GEOMS

        self._pad_ids = {model.geom(name).id: name for name in PAD_GEOMS}
        self._model = model

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        xy = np.array([rng.uniform(0.48, 0.62), rng.uniform(-0.15, 0.15)])
        z0 = self.cfg.table_top_z
        self._model.geom_pos[self._ids["base_geom"]] = [xy[0], xy[1], z0 + 0.015]
        self._model.body_pos[self._ids["button_body"]] = [xy[0], xy[1], z0 + 0.03]
        data.qpos[self._ids["button_qpos"]] = 0.0
        self.button_xy = xy

    def cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self._ids["button_body"]].copy() + np.array([0, 0, 0.015])

    def cube_yaw(self, data: mujoco.MjData) -> float:
        return 0.0

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().start(model, data)
        self.status.metrics["max_press_m"] = 0.0

    def depth(self, data: mujoco.MjData) -> float:
        return float(-data.qpos[self._ids["button_qpos"]])

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return self.depth(data) >= self.travel

    def progress(self, data: mujoco.MjData) -> float:
        return float(
            np.clip(self.status.metrics.get("max_press_m", 0.0) / self.travel, 0, 1)
        )

    def update(
        self,
        data: mujoco.MjData,
        pinch: np.ndarray,
        grip: float,
        table: tuple[np.ndarray, np.ndarray],
    ) -> TaskStatus:
        """Measure one tick. The button has no cube: track the press depth."""
        s = self.status
        s.ticks += 1
        m = s.metrics
        m["min_reach_m"] = min(
            m["min_reach_m"], float(np.linalg.norm(self.cube_pos(data) - pinch))
        )
        m["max_grip"] = max(m["max_grip"], float(grip))
        m["max_press_m"] = max(m["max_press_m"], self.depth(data))
        if self.condition(data, grip):
            s.hold_ticks += 1
        else:
            s.hold_ticks = 0
        if not s.success and s.hold_ticks >= 1:
            s.success, s.success_tick = True, s.ticks
        m["progress"] = max(m.get("progress", 0.0), self.progress(data))
        return s


SUITE_TASKS = {
    t.name: t for t in (BlockInBowl, BlockOutOfBowl, MoveBlock, StackBlock, PressButton)
}
