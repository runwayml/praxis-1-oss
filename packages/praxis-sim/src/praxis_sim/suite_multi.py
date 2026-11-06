"""Harder versions of the DROID suite tasks: smaller targets, wider starts, two steps.

The tasks keep the objects, colours and prompts of the suite tasks:

* ``two_blocks_in_bowl``: put the red and the green block in a bowl (two
  pick-and-place steps in one episode).
* ``block_in_small_bowl``: ``block_in_bowl`` with a smaller bowl and wider
  start ranges.
* ``cube_on_small_plate``: ``cube_on_plate`` with a smaller plate and wider
  start ranges.
* ``stack_small_block``: ``stack_block`` with a smaller base block and a
  tighter centring rule.
* ``stack_three_blocks``: stack three blocks into one tower, in any order
  (two pick-and-place steps).

Progress is capped below 1 until the success latches.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np

from praxis_sim.scene import ObjectSpec
from praxis_sim.suite import BlockInBowl, StackBlock, _Container
from praxis_sim.tasks import CubeOnPlate, PickPlace, Task, TaskConfig

GREEN = "green_block"
BLUE = "blue_block"
MAX_PROGRESS = 0.95  # the progress before the success latches


def _sample_xy(
    rng: np.random.Generator, x: tuple[float, float], y: tuple[float, float]
) -> np.ndarray:
    """Draw a point of the table in the box ``x`` by ``y`` (base frame)."""
    return np.array([rng.uniform(*x), rng.uniform(*y)])


class BlockInSmallBowl(BlockInBowl):
    """Put the red block in a small bowl. The block and the bowl start in wider ranges."""

    name = "block_in_small_bowl"
    default_time_limit_s = 14.2  # 213 ticks (the suite task: 20.6 s)
    radius = 0.045  # radius of the bowl walls, metres (the suite bowl: 0.08)
    floor_tol = 0.012  # metres of the block centre above its rest height on the floor
    cube_x = (0.43, 0.65)
    cube_y = (-0.2, 0.2)
    bowl_x = (0.46, 0.64)
    bowl_y = (-0.21, 0.21)
    min_distance = 0.18  # metres between the block and the bowl centre

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        pos, yaw = self.sample_cube(rng)
        bowl = _sample_xy(rng, self.bowl_x, self.bowl_y)
        for _ in range(200):
            pos[:2] = _sample_xy(rng, self.cube_x, self.cube_y)
            bowl = _sample_xy(rng, self.bowl_x, self.bowl_y)
            if np.linalg.norm(bowl - pos[:2]) >= self.min_distance:
                break
        self.move_bowl(bowl)
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        # The block must lie flat on the bowl floor, not lean on the rim.
        cube = self.cube_pos(data)
        floor = self.cfg.table_top_z + 0.006 + self.cfg.cube_size / 2
        flat = abs(cube[2] - floor) < self.floor_tol
        return self.in_bowl(cube) and flat and self.free(data)

    def progress(self, data: mujoco.MjData) -> float:
        return min(super().progress(data), MAX_PROGRESS)


class CubeOnSmallPlate(CubeOnPlate):
    """Put the red cube on a small blue plate. The cube and the plate start in wider ranges."""

    name = "cube_on_small_plate"
    default_time_limit_s = 14.2  # 213 ticks: one chunk of 21 rows and eight of 24
    plate_half = (0.035, 0.035, 0.006)  # the suite plate: 0.07 x 0.07
    cube_x = (0.43, 0.65)
    cube_y = (-0.2, 0.2)
    plate_x = (0.45, 0.64)
    plate_y = (-0.21, 0.21)
    min_distance = 0.18
    edge_margin = 0.01  # the cube centre is this far inside the plate edges, metres

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        pos, yaw = self.sample_cube(rng)
        plate = _sample_xy(rng, self.plate_x, self.plate_y)
        for _ in range(200):
            pos[:2] = _sample_xy(rng, self.cube_x, self.cube_y)
            plate = _sample_xy(rng, self.plate_x, self.plate_y)
            if np.linalg.norm(plate - pos[:2]) >= self.min_distance:
                break
        self._model.geom_pos[self._ids["plate_geom"]] = [
            plate[0],
            plate[1],
            self.cfg.table_top_z + self.plate_half[2],
        ]
        self._set_free_pose(data, "cube", pos, yaw)
        self.start_xy = pos[:2].copy()

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube = self.cube_pos(data)
        plate = self.plate_pos()
        inner = np.array(self.plate_half[:2]) - self.edge_margin
        inside = bool(np.all(np.abs(cube[:2] - plate[:2]) <= inner))
        resting = abs(cube[2] - (plate[2] + self.cfg.cube_size / 2)) < 0.01
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return inside and resting and not (left or right)

    def progress(self, data: mujoco.MjData) -> float:
        return min(super().progress(data), MAX_PROGRESS)


class StackSmallBlock(StackBlock):
    """Put the red block on top of a small green block, near its centre."""

    name = "stack_small_block"
    target_half = 0.02  # half size of the green block, metres (the suite block: 0.03)
    centre_tol = 0.015  # metres between the two block centres in the table plane

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        cube, tgt = self.cube_pos(data), self.target_pos(data)
        top = tgt[2] + self.target_half + self.cfg.cube_size / 2
        above = abs(cube[2] - top) < 0.012
        centred = np.linalg.norm(cube[:2] - tgt[:2]) < self.centre_tol
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        return bool(above and centred and not (left or right))

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().start(model, data)
        self.target_rest_z = self.target_pos(data)[2]

    def progress(self, data: mujoco.MjData) -> float:
        # No credit while the green block is off the table: the red block goes
        # on top of the green block, not under it.
        if self.target_pos(data)[2] - self.target_rest_z > 0.01:
            return 0.0
        return min(super().progress(data), MAX_PROGRESS)


class TwoBlocksInBowl(_Container):
    """Put the red block and the green block in one bowl (two steps)."""

    name = "two_blocks_in_bowl"
    # One sentence that names both blocks, as the DROID instructions do. The
    # success rule takes the blocks in any order, so the prompt gives no order.
    default_prompt = "Put the red and the green block in the bowl."
    default_time_limit_s = 27.0  # 405 ticks: one chunk of 21 rows and sixteen of 24
    radius = 0.08  # the suite bowl; two 5 cm blocks fit side by side
    slot = 0.038  # metres from the bowl centre to each place point of the oracle
    block_x = (0.43, 0.65)
    block_y = (-0.21, 0.21)
    block_yaw_deg = (-40.0, 40.0)
    bowl_x = (0.48, 0.62)
    bowl_y = (-0.17, 0.17)
    bowl_distance = 0.19  # metres between each block and the bowl centre
    block_distance = 0.12  # metres between the two blocks

    def objects(self) -> list[ObjectSpec]:
        h = self.cfg.cube_size / 2
        green = ObjectSpec(
            name=GREEN,
            half_size=(h, h, h),
            rgba=(0.15, 0.6, 0.2, 1.0),
            mass=self.cfg.cube_mass,
        )
        return [self.cube_spec(), green]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self.bind_free(model, "green", GREEN)

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        bowl = _sample_xy(rng, self.bowl_x, self.bowl_y)
        red = green = _sample_xy(rng, self.block_x, self.block_y)
        for _ in range(500):
            bowl = _sample_xy(rng, self.bowl_x, self.bowl_y)
            red = _sample_xy(rng, self.block_x, self.block_y)
            green = _sample_xy(rng, self.block_x, self.block_y)
            if (
                np.linalg.norm(red - bowl) >= self.bowl_distance
                and np.linalg.norm(green - bowl) >= self.bowl_distance
                and np.linalg.norm(red - green) >= self.block_distance
            ):
                break
        self.move_bowl(bowl)
        z = self.cfg.table_top_z + self.cfg.cube_size / 2 + 1e-4
        for key, xy in (("cube", red), ("green", green)):
            yaw = math.radians(rng.uniform(*self.block_yaw_deg))
            self._set_free_pose(data, key, np.array([xy[0], xy[1], z]), yaw)
        self.start_xy = {"cube": red.copy(), "green": green.copy()}

    def block_pos(self, data: mujoco.MjData, key: str) -> np.ndarray:
        """Return the centre of the red (``"cube"``) or the green block."""
        return self.free_pos(data, key)

    def placed(self, data: mujoco.MjData, key: str) -> bool:
        """Return ``True`` when the block rests on the bowl floor and no pad touches it."""
        pos = self.block_pos(data, key)
        low = pos[2] - (self.cfg.table_top_z + self.cfg.cube_size / 2) < 0.03
        left, right = self.pad_contacts(data, self._ids[key + "_geom"])
        return self.in_bowl(pos) and low and not (left or right)

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return self.placed(data, "cube") and self.placed(data, "green")

    def reach(self, data: mujoco.MjData, key: str) -> float:
        """Return how far the block has come to the bowl centre, in [0, 1]."""
        d0 = float(np.linalg.norm(self.start_xy[key] - self.bowl_xy))
        d = float(np.linalg.norm(self.block_pos(data, key)[:2] - self.bowl_xy))
        return float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))

    def progress(self, data: mujoco.MjData) -> float:
        """0.5 per block in the bowl, plus the distance term of the next block."""
        done = [k for k in ("cube", "green") if self.placed(data, k)]
        todo = [k for k in ("cube", "green") if k not in done]
        if not todo:
            return MAX_PROGRESS
        dense = max(self.reach(data, k) for k in todo)
        return min(0.5 * len(done) + 0.45 * dense, MAX_PROGRESS)

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        # The place points are side by side along x. The oracle turns each
        # block square to the table axes (target yaw 0), so that the fingers
        # stay along y and open clear of the other block.
        z = self.cfg.table_top_z + 0.006 + self.cfg.cube_size / 2
        slots = [
            np.array([self.bowl_xy[0] + s * self.slot, self.bowl_xy[1], z])
            for s in (-1.0, 1.0)
        ]
        red = self.block_pos(data, "cube")
        green = self.block_pos(data, "green")
        if abs(red[0] - slots[0][0]) > abs(red[0] - slots[1][0]):
            slots.reverse()
        return [
            PickPlace(red, self.free_yaw(data, "cube"), slots[0], target_yaw=0.0),
            PickPlace(green, self.free_yaw(data, "green"), slots[1], target_yaw=0.0),
        ]


class StackThreeBlocks(Task):
    """Stack the red, the green and the blue block into one tower, in any order."""

    name = "stack_three_blocks"
    # The prompt names no order: any tower of the three blocks is a success.
    default_prompt = "Stack the blocks."
    default_time_limit_s = 31.8  # 477 ticks: one chunk of 21 rows and nineteen of 24
    keys = ("cube", "green", "blue")
    block_x = (0.47, 0.61)
    block_y = (-0.14, 0.14)
    block_distance = 0.1  # metres between any two blocks at the start
    centre_tol = 0.025  # metres between the centres of two stacked blocks

    def objects(self) -> list[ObjectSpec]:
        h = self.cfg.cube_size / 2
        m = self.cfg.cube_mass
        return [
            self.cube_spec(),
            ObjectSpec(GREEN, (h, h, h), (0.15, 0.6, 0.2, 1.0), m),
            ObjectSpec(BLUE, (h, h, h), (0.15, 0.3, 0.8, 1.0), m),
        ]

    def bind(self, model: mujoco.MjModel) -> None:
        super().bind(model)
        self.bind_free(model, "green", GREEN)
        self.bind_free(model, "blue", BLUE)

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        pts = [_sample_xy(rng, self.block_x, self.block_y) for _ in self.keys]
        for _ in range(500):
            pts = [_sample_xy(rng, self.block_x, self.block_y) for _ in self.keys]
            gaps = [
                np.linalg.norm(pts[i] - pts[j])
                for i in range(3)
                for j in range(i + 1, 3)
            ]
            if min(gaps) >= self.block_distance:
                break
        z = self.cfg.table_top_z + self.cfg.cube_size / 2 + 1e-4
        for key, xy in zip(self.keys, pts, strict=True):
            yaw = math.radians(rng.uniform(-40.0, 40.0))
            self._set_free_pose(data, key, np.array([xy[0], xy[1], z]), yaw)

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().start(model, data)
        self.start_pos = {k: self.free_pos(data, k) for k in self.keys}

    def on_top(self, data: mujoco.MjData, upper: str, lower: str) -> bool:
        """Return ``True`` when ``upper`` rests on ``lower``, centred, no pad on it."""
        a, b = self.free_pos(data, upper), self.free_pos(data, lower)
        level = abs(a[2] - b[2] - self.cfg.cube_size) < 0.012
        centred = np.linalg.norm(a[:2] - b[:2]) < self.centre_tol
        left, right = self.pad_contacts(data, self._ids[upper + "_geom"])
        return bool(level and centred and not (left or right))

    def tower(self, data: mujoco.MjData) -> list[str]:
        """Return the tallest tower, bottom first: blocks that rest on one another."""
        h = self.cfg.cube_size / 2
        best: list[str] = []
        for base in self.keys:
            if abs(self.free_pos(data, base)[2] - (self.cfg.table_top_z + h)) > 0.01:
                continue
            stack = [base]
            for _ in range(2):
                nxt = [
                    k
                    for k in self.keys
                    if k not in stack and self.on_top(data, k, stack[-1])
                ]
                if not nxt:
                    break
                stack.append(nxt[0])
            if len(stack) > len(best):
                best = stack
        return best

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        return len(self.tower(data)) == 3

    def progress(self, data: mujoco.MjData) -> float:
        """0.45 per block on the tower, plus a dense carry term for the next block."""
        tower = self.tower(data)
        if len(tower) == 3:
            return MAX_PROGRESS
        size = self.cfg.cube_size
        tops = [tower[-1]] if tower else list(self.keys)
        dense = 0.0
        for top in tops:
            t = self.free_pos(data, top)
            for k in self.keys:
                if k == top or k in tower:
                    continue
                p = self.free_pos(data, k)
                d0 = float(np.linalg.norm(self.start_pos[k][:2] - t[:2]))
                d = float(np.linalg.norm(p[:2] - t[:2]))
                reach = float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))
                lift = float(np.clip((p[2] - self.start_pos[k][2]) / size, 0.0, 1.0))
                dense = max(dense, min(reach, lift))
        stacked = max(len(tower) - 1, 0)
        return min(0.45 * stacked + 0.45 * dense, MAX_PROGRESS)

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        # The base is the block nearest to the other two. The oracle turns
        # each block to the yaw of the base, so the tower faces line up.
        pos = {k: self.free_pos(data, k) for k in self.keys}
        base = min(
            self.keys,
            key=lambda b: sum(np.linalg.norm(pos[b][:2] - pos[k][:2]) for k in pos),
        )
        rest = sorted(
            (k for k in self.keys if k != base),
            key=lambda k: np.linalg.norm(pos[k][:2] - pos[base][:2]),
        )
        yaw = self.free_yaw(data, base)
        size = self.cfg.cube_size
        steps = []
        for level, k in enumerate(rest, start=1):
            target = pos[base] + np.array([0.0, 0.0, level * size])
            steps.append(
                PickPlace(pos[k], self.free_yaw(data, k), target, target_yaw=yaw)
            )
        return steps


MULTI_TASKS: dict[str, Callable[[TaskConfig], Task]] = {
    t.name: t
    for t in (
        TwoBlocksInBowl,
        BlockInSmallBowl,
        CubeOnSmallPlate,
        StackSmallBlock,
        StackThreeBlocks,
    )
}
