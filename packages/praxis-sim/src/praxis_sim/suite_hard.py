"""Harder versions of the DROID suite tasks: distractor objects, a named target colour.

Each task puts 5 cm blocks of distinct colours on the table. The seed picks
the target colour and the prompt names it. The other blocks are
distractors. A moved distractor gives no progress. A distractor that meets
the goal (lifted, in the bowl, stacked) fails the episode: the task latches
``failed`` and can no longer succeed.

Tasks:

* ``pick_color_block``: "Pick up the <colour> block." Success: the target
  block is 5 cm or more above its rest height for 0.25 s or more.
* ``color_block_in_bowl``: "Put the <colour> block in the bowl." Success:
  the target block rests in the bowl and no finger pad touches it.
* ``stack_color_blocks``: "Put the <colour> block on the <colour> block."
  Success: the target block rests on the base block, centred, and no finger
  pad touches it.

The base class metrics (reach, lift, grasp, object off the table) and the
scripted oracle use the target block: ``place`` binds the target block to
the ids of the cube.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np

from praxis_sim.scene import ObjectSpec
from praxis_sim.suite import _Container
from praxis_sim.tasks import CUBE, PickPlace, Task, TaskConfig, TaskStatus

# Colours that stay distinct on the wood table in both camera views.
COLOURS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.1, 0.1, 1.0),
    "green": (0.15, 0.6, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.8, 1.0),
    "yellow": (0.95, 0.85, 0.1, 1.0),
    "purple": (0.55, 0.2, 0.75, 1.0),
}


def block_name(colour: str) -> str:
    """Return the body name of the block of ``colour`` (red is the base cube)."""
    return CUBE if colour == "red" else f"{colour}_block"


class _ColourBlocks(Task):
    """Blocks of distinct colours on the table. The seed picks the target colour."""

    colours: tuple[str, ...] = ("red", "green", "blue", "yellow")
    spacing = 0.10  # minimum distance between two block centres, metres
    block_x = (0.44, 0.64)
    block_y = (-0.2, 0.2)

    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.target = self.colours[0]
        self.wrong = False
        self.rest: dict[str, float] = {}

    def objects(self) -> list[ObjectSpec]:
        h = self.cfg.cube_size / 2
        return [
            ObjectSpec(
                name=block_name(c),
                half_size=(h, h, h),
                rgba=COLOURS[c],
                mass=self.cfg.cube_mass,
            )
            for c in self.colours
        ]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        for c in self.colours:
            self.bind_free(model, c, block_name(c))

    def set_target(self, colour: str) -> None:
        """Bind the block of ``colour`` to the ids of the cube (metrics, oracle)."""
        self.target = colour
        for part in ("body", "geom", "qpos", "dof"):
            self._ids[f"cube_{part}"] = self._ids[f"{colour}_{part}"]

    def draw_target(self, rng: np.random.Generator) -> None:
        """Pick the target colour: seeds 0-7 cover up to five colours."""
        self.set_target(self.colours[int(rng.uniform() * len(self.colours))])

    def others(self) -> list[str]:
        """Return the colours of the distractor blocks."""
        return [c for c in self.colours if c != self.target]

    def sample_blocks(
        self, rng: np.random.Generator, keep_out: list[tuple[np.ndarray, float]]
    ) -> dict[str, np.ndarray] | None:
        """Draw the block centres (xy), apart and clear of the keep-out discs.

        Return ``None`` when a block finds no free spot.
        """
        out: dict[str, np.ndarray] = {}
        for c in self.colours:
            for _ in range(200):
                xy = np.array([rng.uniform(*self.block_x), rng.uniform(*self.block_y)])
                apart = all(
                    np.linalg.norm(xy - p) >= self.spacing for p in out.values()
                )
                clear = all(np.linalg.norm(xy - p) >= r for p, r in keep_out)
                if apart and clear:
                    out[c] = xy
                    break
            else:
                return None
        return out

    def put_blocks(
        self, rng: np.random.Generator, data: mujoco.MjData, xy: dict[str, np.ndarray]
    ) -> None:
        """Put the blocks at rest on the table at ``xy`` with seeded yaws."""
        z = self.cfg.table_top_z + self.cfg.cube_size / 2 + 1e-4
        for c, p in xy.items():
            yaw = math.radians(rng.uniform(*self.cfg.cube_yaw_deg))
            self._set_free_pose(data, c, np.array([p[0], p[1], z]), yaw)
        self.start_xy = xy[self.target].copy()

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().start(model, data)
        self.wrong = False
        self.rest = {c: float(self.free_pos(data, c)[2]) for c in self.colours}
        self.status.metrics["wrong_lift_m"] = 0.0
        self.status.metrics["wrong_block"] = 0.0

    def lift(self, data: mujoco.MjData, colour: str) -> float:
        """Return the height of the block of ``colour`` above its rest height."""
        return float(self.free_pos(data, colour)[2] - self.rest[colour])

    def free_of(self, data: mujoco.MjData, colour: str) -> bool:
        """Return ``True`` when no finger pad touches the block of ``colour``."""
        left, right = self.pad_contacts(data, self._ids[f"{colour}_geom"])
        return not (left or right)

    def wrong_done(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when a distractor meets the goal of the task."""
        raise NotImplementedError

    def goal(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when the target block meets the goal of the task."""
        raise NotImplementedError

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return not self.wrong and self.goal(data)

    def update(
        self,
        data: mujoco.MjData,
        pinch: np.ndarray,
        grip: float,
        table: tuple[np.ndarray, np.ndarray],
    ) -> TaskStatus:
        """Latch a failure when a distractor meets the goal, then measure the tick."""
        m = self.status.metrics
        m["wrong_lift_m"] = max(
            m["wrong_lift_m"], max(self.lift(data, c) for c in self.others())
        )
        if not self.wrong and not self.status.success and self.wrong_done(data):
            self.wrong = True
            self.status.failed = True
            m["wrong_block"] = 1.0
        return super().update(data, pinch, grip, table)


class PickColorBlock(_ColourBlocks):
    """Pick up the block of the named colour among distractor blocks."""

    name = "pick_color_block"
    default_time_limit_s = 7.8  # 117 ticks: one chunk of 21 rows and four of 24
    colours = ("red", "green", "blue", "yellow", "purple")
    spacing = 0.085

    @property
    def prompt(self) -> str:
        if self.cfg.prompt:
            return self.cfg.prompt
        return f"Pick up the {self.target} block."

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        self.draw_target(rng)
        xy = None
        while xy is None:
            xy = self.sample_blocks(rng, [])
        self.put_blocks(rng, data, xy)

    def goal(self, data: mujoco.MjData) -> bool:
        return self.lift(data, self.target) >= self.cfg.lift_height

    def wrong_done(self, data: mujoco.MjData) -> bool:
        return any(self.lift(data, c) >= self.cfg.lift_height for c in self.others())

    def progress(self, data: mujoco.MjData) -> float:
        if self.wrong:
            return 0.0
        lift = self.lift(data, self.target) / self.cfg.lift_height
        return float(np.clip(lift, 0.0, 1.0))


class ColorBlockInBowl(_ColourBlocks, _Container):
    """Put the block of the named colour in the bowl; distractor blocks around."""

    name = "color_block_in_bowl"
    default_time_limit_s = 11.0  # 165 ticks: one chunk of 21 rows and six of 24
    colours = ("red", "green", "blue", "yellow", "purple")
    spacing = 0.085
    radius = 0.06  # a smaller bowl than in the base suite (0.08)
    bowl_clear = 0.12  # minimum distance between the bowl and a block centre
    target_far = 0.18  # minimum distance between the bowl and the target block

    @property
    def prompt(self) -> str:
        if self.cfg.prompt:
            return self.cfg.prompt
        return f"Put the {self.target} block in the bowl."

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        self.draw_target(rng)
        while True:
            bowl = np.array([rng.uniform(0.48, 0.62), rng.uniform(-0.17, 0.17)])
            xy = self.sample_blocks(rng, [(bowl, self.bowl_clear)])
            if xy is not None and np.linalg.norm(xy[self.target] - bowl) >= (
                self.target_far
            ):
                break
        self.move_bowl(bowl)
        self.put_blocks(rng, data, xy)

    def place_target(self) -> np.ndarray | None:
        z = self.cfg.table_top_z + 0.006 + self.cfg.cube_size / 2
        return np.array([self.bowl_xy[0], self.bowl_xy[1], z])

    def block_in_bowl(self, data: mujoco.MjData, colour: str) -> bool:
        """Return ``True`` when the block of ``colour`` rests in the bowl, let go."""
        pos = self.free_pos(data, colour)
        low = pos[2] - self.rest[colour] < 0.03
        return self.in_bowl(pos) and low and self.free_of(data, colour)

    def goal(self, data: mujoco.MjData) -> bool:
        return self.block_in_bowl(data, self.target)

    def wrong_done(self, data: mujoco.MjData) -> bool:
        return any(self.block_in_bowl(data, c) for c in self.others())

    def progress(self, data: mujoco.MjData) -> float:
        if self.wrong:
            return 0.0
        d0 = float(np.linalg.norm(self.start_xy - self.bowl_xy))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - self.bowl_xy))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))


class StackColorBlocks(_ColourBlocks):
    """Put the block of one named colour on the block of another named colour."""

    name = "stack_color_blocks"
    default_time_limit_s = 17.4  # 261 ticks: one chunk of 21 rows and ten of 24
    colours = ("red", "green", "blue", "yellow", "purple")
    spacing = 0.085

    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__(cfg)
        self.base = self.colours[1]

    @property
    def prompt(self) -> str:
        if self.cfg.prompt:
            return self.cfg.prompt
        return f"Put the {self.target} block on the {self.base} block."

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        self.draw_target(rng)
        others = self.others()
        self.base = others[int(rng.uniform() * len(others))]
        xy = None
        while xy is None:
            xy = self.sample_blocks(rng, [])
        self.put_blocks(rng, data, xy)
        self.base_xy = xy[self.base].copy()

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        base = self.free_pos(data, self.base)
        top = base + np.array([0.0, 0.0, self.cfg.cube_size])
        return [
            PickPlace(
                self.cube_pos(data),
                self.cube_yaw(data),
                top,
                target_yaw=self.free_yaw(data, self.base),
            )
        ]

    def stacked(self, data: mujoco.MjData, top: str, base: str) -> bool:
        """Return ``True`` when block ``top`` rests on block ``base``, let go."""
        a, b = self.free_pos(data, top), self.free_pos(data, base)
        above = abs(a[2] - (b[2] + self.cfg.cube_size)) < 0.012
        centred = np.linalg.norm(a[:2] - b[:2]) < self.cfg.cube_size / 2
        return bool(above and centred and self.free_of(data, top))

    def goal(self, data: mujoco.MjData) -> bool:
        return self.stacked(data, self.target, self.base)

    def wrong_done(self, data: mujoco.MjData) -> bool:
        return any(
            self.stacked(data, a, b)
            for a in self.colours
            for b in self.colours
            if a != b and (a, b) != (self.target, self.base)
        )

    def progress(self, data: mujoco.MjData) -> float:
        if self.wrong:
            return 0.0
        base = self.free_pos(data, self.base)
        h = self.cfg.cube_size / 2
        d0 = float(np.linalg.norm(self.start_xy - self.base_xy))
        d = float(np.linalg.norm(self.cube_pos(data)[:2] - base[:2]))
        reach = float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))
        lift = float(np.clip(self.lift(data, self.target) / (2 * h), 0.0, 1.0))
        # 0.9 at most before success: a block held on the base is not done.
        if d > h:
            return 0.45 * reach + 0.45 * min(reach, lift)
        return 0.5 + 0.4 * lift


HARD_TASKS: dict[str, Callable[[TaskConfig], Task]] = {
    t.name: t for t in (PickColorBlock, ColorBlockInBowl, StackColorBlocks)
}
