"""Insertion tasks: a square peg in a square hole at a tight clearance.

The peg is a free blue box (3 x 3 x 6 cm) that stands upright on the table.
The board is a fixed dark grey body (10 x 10 cm, 2.5 cm high) of four boxes
around a square hole that goes down to the table. The side of the hole is the
peg side plus two times the clearance. The clearance is the difficulty knob:
one task for each clearance, with one prompt for all.

Success: the peg bottom is 1.5 cm or more below the board top, the peg centre
is within 1 cm of the hole centre in x and y, the peg tilts less than 10 deg,
and the gripper has let it go, for 0.25 s or more.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np

from praxis.sim.scene import PAD_GEOMS, PINCH_SITE, ObjectSpec
from praxis.sim.tasks import PickPlace, Task, TaskConfig, yaw_quat

PEG = "blue_peg"
BOARD = "board"


class PegInHole(Task):
    """Put a free square peg in the square hole of a fixed board."""

    default_prompt = "Put the blue peg in the square hole."
    default_time_limit_s = 25.4  # 381 ticks: one chunk of 21 rows and 15 of 24
    clearance = 0.004  # metres between the peg and the hole wall, per side
    peg_half = (0.015, 0.015, 0.03)
    peg_mass = 0.043  # kg, a density of about 800 kg/m^3
    board_half = 0.05  # half the side of the square board, metres
    board_height = 0.025
    board_rgba = (0.22, 0.22, 0.24, 1.0)
    board_yaw_deg = (0.0, 0.0)  # range of the seeded board yaw
    depth = 0.015  # how far the peg bottom must go below the board top
    centre_tol = 0.01  # peg centre to hole centre, in x and in y
    max_tilt_deg = 10.0
    release_distance = 0.05  # the pinch point this far from the peg = released
    min_edge_gap = 0.06  # peg centre to the board edge, at the start
    min_distance = 0.15  # peg centre to the hole centre, at the start
    peg_yaw_deg = (-20.0, 20.0)  # range of the peg yaw relative to the board

    @property
    def hole_half(self) -> float:
        """Return half the side of the square hole, metres."""
        return self.peg_half[0] + self.clearance

    def objects(self) -> list[ObjectSpec]:
        return [
            ObjectSpec(
                name=PEG,
                half_size=self.peg_half,
                rgba=(0.12, 0.3, 0.85, 1.0),
                mass=self.peg_mass,
            )
        ]

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add the board: four boxes around the hole on one fixed body."""
        body = spec.worldbody.add_body(name=BOARD, pos=[0.55, 0.15, 0.0])
        b, h, z = self.board_half, self.hole_half, self.board_height / 2
        side = (b - h) / 2  # half the width of a wall
        walls = (
            ("front", [b - side, 0.0], [side, b]),
            ("back", [-(b - side), 0.0], [side, b]),
            ("left", [0.0, b - side], [h, side]),
            ("right", [0.0, -(b - side)], [h, side]),
        )
        for name, xy, size in walls:
            body.add_geom(
                name=f"{BOARD}_{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[size[0], size[1], z],
                pos=[xy[0], xy[1], z],
                rgba=list(self.board_rgba),
            )

    def bind(self, model: mujoco.MjModel) -> None:
        self._ids = {}
        self.bind_free(model, "cube", PEG)
        self._pad_ids = {model.geom(name).id: name for name in PAD_GEOMS}
        self._model = model
        self._board_body = model.body(BOARD).id
        self._pinch_site = model.site(PINCH_SITE).id
        self.hole_xy = np.array([0.55, 0.15])
        self.board_yaw = 0.0

    def move_board(self, xy: np.ndarray, yaw: float) -> None:
        """Put the board at ``xy`` (base frame) with the yaw ``yaw``, on the table."""
        self._model.body_pos[self._board_body] = [xy[0], xy[1], self.cfg.table_top_z]
        self._model.body_quat[self._board_body] = yaw_quat(yaw)
        self.hole_xy = np.asarray(xy, dtype=np.float64).copy()
        self.board_yaw = float(yaw)

    def to_board(self, xy: np.ndarray) -> np.ndarray:
        """Return ``xy`` (base frame) in the frame of the board."""
        c, s = math.cos(self.board_yaw), math.sin(self.board_yaw)
        d = np.asarray(xy[:2], dtype=np.float64) - self.hole_xy
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def edge_gap(self, xy: np.ndarray) -> float:
        """Return the distance from ``xy`` to the board square, metres."""
        d = np.maximum(np.abs(self.to_board(xy)) - self.board_half, 0.0)
        return float(np.linalg.norm(d))

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        c = self.cfg
        z = c.table_top_z + self.peg_half[2] + 1e-4
        for _ in range(200):
            board = np.array([rng.uniform(0.48, 0.62), rng.uniform(-0.15, 0.15)])
            yaw = math.radians(rng.uniform(*self.board_yaw_deg))
            self.move_board(board, yaw)
            peg = np.array([rng.uniform(*c.cube_x), rng.uniform(*c.cube_y)])
            far = np.linalg.norm(peg - board) >= self.min_distance
            if far and self.edge_gap(peg) >= self.min_edge_gap:
                break
        peg_yaw = self.board_yaw + math.radians(rng.uniform(*self.peg_yaw_deg))
        self._set_free_pose(data, "cube", np.array([peg[0], peg[1], z]), peg_yaw)
        self.start_xy = peg.copy()

    def board_top(self) -> float:
        """Return the height of the board top, metres."""
        return self.cfg.table_top_z + self.board_height

    def peg_bottom(self, data: mujoco.MjData) -> float:
        """Return the height of the lowest point of the peg, metres."""
        r = data.xmat[self._ids["cube_body"]].reshape(3, 3)
        down = float(np.abs(r[2]) @ np.asarray(self.peg_half))
        return float(self.cube_pos(data)[2]) - down

    def tilt_deg(self, data: mujoco.MjData) -> float:
        """Return the angle between the peg axis and the vertical, degrees."""
        r = data.xmat[self._ids["cube_body"]].reshape(3, 3)
        return math.degrees(math.acos(float(np.clip(abs(r[2, 2]), 0.0, 1.0))))

    def released(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when no pad touches the peg or the pinch point is far."""
        left, right = self.pad_contacts(data, self._ids["cube_geom"])
        pinch = data.site_xpos[self._pinch_site]
        far = np.linalg.norm(pinch - self.cube_pos(data)) >= self.release_distance
        return bool(not (left or right) or far)

    def condition(self, data: mujoco.MjData, grip: float) -> bool:
        off = np.abs(self.to_board(self.cube_pos(data)[:2]))
        centred = bool(np.all(off <= self.centre_tol))
        deep = self.peg_bottom(data) <= self.board_top() - self.depth
        upright = self.tilt_deg(data) < self.max_tilt_deg
        return centred and deep and upright and self.released(data)

    def progress(self, data: mujoco.MjData) -> float:
        """Return a mix of the carry to the hole, the lift and the insertion depth.

        Away from the hole: 0.35 for the carry and 0.35 for the carry with
        the peg upright and high enough to clear the board (0 at the start).
        Upright over the hole: 0.7 plus 0.25 for the depth. A peg that has
        fallen over gets the carry part only. The base update gives 1 at
        success.
        """
        peg = self.cube_pos(data)
        d0 = float(np.linalg.norm(self.start_xy - self.hole_xy))
        d = float(np.linalg.norm(peg[:2] - self.hole_xy))
        reach = float(np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0))
        if self.tilt_deg(data) >= self.max_tilt_deg:
            return 0.35 * reach
        bottom = self.peg_bottom(data)
        lift = (bottom - self.cfg.table_top_z) / self.board_height
        lift = float(np.clip(lift, 0.0, 1.0))
        if np.all(np.abs(self.to_board(peg[:2])) <= self.centre_tol):
            depth = (self.board_top() - bottom) / self.depth
            return 0.7 + 0.25 * float(np.clip(depth, 0.0, 1.0))
        return 0.35 * reach + 0.35 * min(reach, lift)

    def oracle_steps(self, data: mujoco.MjData) -> list[PickPlace]:
        """Grasp the peg 2 cm below its top, turn it to the board, lower it slowly."""
        h = self.peg_half[2]
        # The oracle stops 4 mm above the target: the peg drops to the table.
        target = np.array([self.hole_xy[0], self.hole_xy[1], self.cfg.table_top_z + h])
        return [
            PickPlace(
                self.cube_pos(data),
                self.cube_yaw(data),
                target,
                grasp_dz=h - 0.02,
                place_s=2.5,
                target_yaw=self.board_yaw,
            )
        ]


def _peg_task(mm: int) -> type[PegInHole]:
    """Return the peg-in-hole task class with a clearance of ``mm`` mm per side."""
    return type(
        f"PegInHole{mm}mm",
        (PegInHole,),
        {
            "name": f"peg_in_hole_{mm}mm",
            "clearance": mm / 1000,
            "__doc__": f"Put the blue peg in a square hole with {mm} mm per side.",
        },
    )


INSERT_TASKS: dict[str, Callable[[TaskConfig], Task]] = {
    t.name: t for t in (_peg_task(6), _peg_task(4), _peg_task(2))
}
