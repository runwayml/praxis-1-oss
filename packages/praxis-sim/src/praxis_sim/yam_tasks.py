"""Tasks of the bimanual YAM rig: tabletop versions of ABC-130k fine-tune tasks.

Each task gives:

* the prompt: the caption of the training data for a pick-and-place task
  (``caption``), else the ABC-130k task name with spaces,
* the bodies of the task (MuJoCo primitives: free objects and fixed
  containers),
* a seeded start state (object poses), which is never a success,
* a per-tick update: success after a hold time, and a dense progress value
  ``metrics["progress"]`` in [0, 1] (the maximum over the episode, 1 at
  success),
* the plan of a scripted bimanual oracle (:mod:`praxis_sim.yam_scripted`)
  that solves the task. The oracle reads the object poses from the
  simulation.

All containers are bodies without a joint: a base and wall boxes (MuJoCo
collides convex shapes only). ``place`` moves them through ``model.body_pos``.
Positions are in the world frame of :mod:`praxis_sim.yam_scene`; +y is the
side of the left arm.

Progress of one object that must go to a target region::

    p = 0.15 * approach + 0.15 * lifted + 0.7 * transport

``approach`` is 1 - d / d0 of the nearest grasp site to the object (d0 = the
start distance), ``lifted`` is 1 once a gripper holds the object 2 cm or more
above its start height, ``transport`` is 1 - dxy / dxy0 of the object to the
target. An object in its region counts 1. Each term keeps its maximum over
the episode. The task progress is the mean over the objects, at most 0.95
before success and 1 at success.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import mujoco
import numpy as np

from praxis_sim.scene import CONTROL_HZ
from praxis_sim.tasks import TaskStatus, yaw_quat
from praxis_sim.yam_scene import SIDES, TABLE_TOP_Z

if TYPE_CHECKING:
    from praxis_sim.yam_env import YamArm
    from praxis_sim.yam_scripted import Plan

RING_WALLS = 16
PROGRESS_CAP = 0.95  # the largest progress before success
BAG_CLEARANCE = 0.07  # metres from an item centre to the outside of the bag walls
BAG_PLACE_TRIES = 100  # item draws until every item clears the bag (else the best)

# ---------------------------------------------------------------------------
# Settings and the base class
# ---------------------------------------------------------------------------


@dataclass
class YamTaskConfig:
    """Settings of a bimanual task."""

    name: str = "screwdriver_in_bin"
    prompt: str | None = None  # None = the training text of the task
    time_limit_s: float | None = None  # None = the default of the task
    hold_s: float = 0.5  # time that the success condition must hold


@dataclass
class Item:
    """One free object of a task: its body, geoms and progress terms."""

    name: str
    body: int = -1
    qpos: int = -1
    dof: int = -1
    geoms: frozenset[int] = frozenset()
    start: np.ndarray | None = None
    d0: float = 1.0
    approach: float = 0.0
    lifted: float = 0.0
    transport: float = 0.0
    done: float = 0.0


def _box(
    body: mujoco.MjsBody,
    name: str,
    pos: list[float],
    size: list[float],
    rgba: tuple[float, ...],
    quat: list[float] | None = None,
    mass: float | None = None,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
) -> None:
    g = body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        rgba=list(rgba),
        friction=list(friction),
        condim=4,
    )
    if quat is not None:
        g.quat = quat
    if mass is not None:
        g.mass = mass


def add_tray(
    spec: mujoco.MjSpec,
    name: str,
    inner: tuple[float, float],
    height: float,
    rgba: tuple[float, ...],
    wall: float = 0.006,
    z0: float = TABLE_TOP_Z,
) -> None:
    """Add an open box (base and four walls) without a joint."""
    body = spec.worldbody.add_body(name=name, pos=[0.7, 0.0, z0])
    hx, hy = inner[0] / 2, inner[1] / 2
    _box(body, f"{name}_base", [0, 0, 0.003], [hx + wall, hy + wall, 0.003], rgba)
    h = height / 2
    for k, (px, py, sx, sy) in enumerate(
        (
            (hx + wall / 2, 0.0, wall / 2, hy + wall),
            (-hx - wall / 2, 0.0, wall / 2, hy + wall),
            (0.0, hy + wall / 2, hx, wall / 2),
            (0.0, -hy - wall / 2, hx, wall / 2),
        )
    ):
        _box(body, f"{name}_wall{k}", [px, py, h], [sx, sy, h], rgba)


def add_ring(
    spec: mujoco.MjSpec,
    name: str,
    radius: float,
    height: float,
    rgba: tuple[float, ...],
    wall: float = 0.006,
    flare: float = 0.0,
    z0: float = TABLE_TOP_Z,
) -> None:
    """Add a bowl: a base disc and a ring of wall boxes, without a joint.

    ``flare`` tilts the walls outwards (radians).
    """
    body = spec.worldbody.add_body(name=name, pos=[0.7, 0.0, z0])
    body.add_geom(
        name=f"{name}_base",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius + wall / 2, 0.003, 0],
        pos=[0, 0, 0.003],
        rgba=list(rgba),
    )
    seg = radius * math.tan(math.pi / RING_WALLS) * 1.08
    for k in range(RING_WALLS):
        a = 2 * math.pi * k / RING_WALLS
        r = radius + math.sin(flare) * height / 2
        qz = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        qy = np.array([math.cos(-flare / 2), 0.0, math.sin(-flare / 2), 0.0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qz, qy)
        _box(
            body,
            f"{name}_wall{k:02d}",
            [r * math.cos(a), r * math.sin(a), height / 2],
            [wall / 2, seg + math.sin(flare) * 0.01, height / 2],
            rgba,
            quat=q.tolist(),
        )


class YamTask:
    """The base class of a bimanual task. One instance holds one episode."""

    name = "task"
    task_name = ""  # the ABC-130k task name
    caption = ""  # the prompt in the caption grammar of the training data, if any
    default_time_limit_s = 30.0
    item_names: tuple[str, ...] = ()

    def __init__(self, cfg: YamTaskConfig, table_top_z: float = TABLE_TOP_Z) -> None:
        self.cfg = cfg
        self.z0 = float(table_top_z)  # height of the table top, metres
        self.status = TaskStatus()
        self.items: dict[str, Item] = {}
        self.model: mujoco.MjModel | None = None
        self.arms: dict[str, YamArm] = {}
        # The data of the episode, for the live object poses of the oracle.
        self.live: mujoco.MjData | None = None

    # -- text and time -------------------------------------------------------

    @property
    def default_prompt(self) -> str:
        """Return the caption of the task, else the task name with spaces."""
        return self.caption or self.task_name.replace("_", " ")

    @property
    def prompt(self) -> str:
        """Return the prompt of the policy."""
        return self.default_prompt if self.cfg.prompt is None else self.cfg.prompt

    @property
    def time_limit_s(self) -> float:
        """Return the time limit of an episode in seconds."""
        return float(self.cfg.time_limit_s or self.default_time_limit_s)

    @property
    def max_ticks(self) -> int:
        """Return the time limit in 15 Hz ticks."""
        return round(self.time_limit_s * CONTROL_HZ)

    @property
    def hold_ticks(self) -> int:
        """Return the ticks that the success condition must hold."""
        return max(1, math.ceil(self.cfg.hold_s * CONTROL_HZ - 1e-9))

    def describe(self) -> str:
        """Return the success rule in one sentence."""
        return ""

    # -- scene ---------------------------------------------------------------

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add the bodies of the task to the scene spec."""

    def add_item(
        self,
        spec: mujoco.MjSpec,
        name: str,
        geoms: list[dict],
    ) -> None:
        """Add a free object made of geoms (MjSpec geom fields per dict)."""
        body = spec.worldbody.add_body(name=name, pos=[0.6, 0.0, self.z0 + 0.1])
        body.add_freejoint(name=f"{name}_joint")
        for k, g in enumerate(geoms):
            fields = {"condim": 4, "friction": [1.0, 0.005, 0.0001], **g}
            body.add_geom(name=f"{name}_g{k}", **fields)

    def bind(self, model: mujoco.MjModel, arms: dict[str, YamArm]) -> None:
        """Find the ids of the task bodies in the compiled model."""
        self.model = model
        self.arms = arms
        self.items = {}
        for n in self.item_names:
            body = model.body(n).id
            jnt = model.joint(f"{n}_joint")
            geoms = frozenset(
                g for g in range(model.ngeom) if model.geom_bodyid[g] == body
            )
            self.items[n] = Item(
                n, body, int(jnt.qposadr[0]), int(jnt.dofadr[0]), geoms
            )
        self._finger_sets = {
            s: (
                self._subtree_geoms(model, f"{s}/left_finger"),
                self._subtree_geoms(model, f"{s}/right_finger"),
            )
            for s in SIDES
        }

    @staticmethod
    def _subtree_geoms(model: mujoco.MjModel, body_name: str) -> frozenset[int]:
        root = model.body(body_name).id
        out = set()
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            while b > 0:
                if b == root:
                    out.add(g)
                    break
                b = int(model.body_parentid[b])
        return frozenset(out)

    def move_static(self, name: str, xy: np.ndarray, yaw: float = 0.0) -> None:
        """Put a body without a joint at ``xy`` on the table, turned ``yaw``."""
        assert self.model is not None
        b = self.model.body(name).id
        self.model.body_pos[b] = [xy[0], xy[1], self.z0]
        self.model.body_quat[b] = yaw_quat(yaw)

    def set_pose(
        self, data: mujoco.MjData, name: str, pos: np.ndarray, quat: np.ndarray
    ) -> None:
        """Set the free joint of an object, at rest."""
        it = self.items[name]
        data.qpos[it.qpos : it.qpos + 3] = pos
        data.qpos[it.qpos + 3 : it.qpos + 7] = quat
        data.qvel[it.dof : it.dof + 6] = 0.0

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        """Put the objects at their seeded start poses."""
        raise NotImplementedError

    # -- state ---------------------------------------------------------------

    def pos(self, data: mujoco.MjData, name: str) -> np.ndarray:
        """Return the world position of an object (its body origin)."""
        return data.xpos[self.items[name].body].copy()

    def rot(self, data: mujoco.MjData, name: str) -> np.ndarray:
        """Return the world rotation of an object."""
        return data.xmat[self.items[name].body].reshape(3, 3).copy()

    def yaw(self, data: mujoco.MjData, name: str) -> float:
        """Return the yaw of the object x axis about z, radians."""
        r = self.rot(data, name)
        return math.atan2(r[1, 0], r[0, 0])

    def contacts(self, data: mujoco.MjData, name: str) -> dict[str, tuple[bool, bool]]:
        """Return per arm (left finger touches, right finger touches) the object."""
        geoms = self.items[name].geoms
        out = {s: [False, False] for s in SIDES}
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a in geoms:
                other = b
            elif b in geoms:
                other = a
            else:
                continue
            for s, (lf, rf) in self._finger_sets.items():
                if other in lf:
                    out[s][0] = True
                elif other in rf:
                    out[s][1] = True
        return {s: (v[0], v[1]) for s, v in out.items()}

    def grasped(self, data: mujoco.MjData, name: str) -> bool:
        """Return ``True`` when both fingers of one arm touch the object."""
        return any(a and b for a, b in self.contacts(data, name).values())

    def pinches(self) -> list[np.ndarray]:
        """Return the grasp-site positions of both arms."""
        return [a.site_pose()[0] for a in self.arms.values()]

    def resting(self, data: mujoco.MjData, name: str, tol: float = 0.05) -> bool:
        """Return ``True`` when the object moves slower than ``tol`` m/s."""
        it = self.items[name]
        return bool(np.linalg.norm(data.qvel[it.dof : it.dof + 3]) < tol)

    # -- episode -------------------------------------------------------------

    def start(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Start the episode after the reset and the settle time."""
        self.status = TaskStatus()
        self.status.metrics = {"progress": 0.0, "max_grip": 0.0}
        pinch = self.pinches()
        for n, it in self.items.items():
            it.start = self.pos(data, n)
            it.d0 = max(min(float(np.linalg.norm(p - it.start)) for p in pinch), 0.05)
            it.approach = it.lifted = it.transport = it.done = 0.0
        self.on_start(data)

    def on_start(self, data: mujoco.MjData) -> None:
        """Hook of a task after ``start`` (targets of the progress terms)."""

    def target_xy(self, name: str) -> np.ndarray | None:
        """Return the xy target of an object for the transport term, or ``None``."""
        return None

    def item_done(self, data: mujoco.MjData, name: str) -> bool:
        """Return ``True`` when one object is in its target region."""
        return False

    def condition(self, data: mujoco.MjData) -> bool:
        """Return ``True`` when the success condition is true at this tick."""
        return all(
            self.item_done(data, n) and not self.grasped(data, n) for n in self.items
        )

    def update_item(self, data: mujoco.MjData, name: str) -> None:
        """Update the progress terms of one object."""
        it = self.items[name]
        p = self.pos(data, name)
        assert it.start is not None
        d = min(float(np.linalg.norm(q - p)) for q in self.pinches())
        it.approach = max(it.approach, float(np.clip(1 - d / it.d0, 0, 1)))
        if p[2] - it.start[2] > 0.02 and self.grasped(data, name):
            it.lifted = 1.0
        tgt = self.target_xy(name)
        if tgt is not None:
            d0 = max(float(np.linalg.norm(it.start[:2] - tgt)), 0.05)
            dt = float(np.linalg.norm(p[:2] - tgt))
            it.transport = max(it.transport, float(np.clip(1 - dt / d0, 0, 1)))
        if self.item_done(data, name) and not self.grasped(data, name):
            it.done = 1.0

    def item_progress(self, it: Item) -> float:
        """Return the progress of one object in [0, 1]."""
        if it.done:
            return 1.0
        return 0.15 * it.approach + 0.15 * it.lifted + 0.7 * it.transport

    def progress(self, data: mujoco.MjData) -> float:
        """Return the progress at this tick in [0, 1] (the mean over the objects)."""
        if not self.items:
            return 0.0
        return float(np.mean([self.item_progress(it) for it in self.items.values()]))

    def update(
        self, data: mujoco.MjData, table: tuple[np.ndarray, np.ndarray]
    ) -> TaskStatus:
        """Measure one tick and latch success after the hold time."""
        s = self.status
        s.ticks += 1
        lo, hi = table
        for n in self.items:
            self.update_item(data, n)
            p = self.pos(data, n)
            inside = bool(np.all(p[:2] >= lo - 0.02) and np.all(p[:2] <= hi + 0.02))
            if not inside or p[2] < self.z0 - 0.05:
                s.failed = True
        m = s.metrics
        m["max_grip"] = max(
            m.get("max_grip", 0.0), max(a.grip() for a in self.arms.values())
        )
        s.hold_ticks = s.hold_ticks + 1 if self.condition(data) else 0
        if not s.success and s.hold_ticks >= self.hold_ticks:
            s.success, s.success_tick = True, s.ticks
        # Progress reaches 1 only at success.
        part = min(self.progress(data), PROGRESS_CAP)
        m["progress"] = 1.0 if s.success else max(m["progress"], part)
        for n, it in self.items.items():
            m[f"{n}_progress"] = self.item_progress(it)
        return s

    # -- oracle --------------------------------------------------------------

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        """Fill the plan of the scripted oracle (one step list per arm)."""
        raise NotImplementedError


def side_of(y: float) -> str:
    """Return the arm on the side of ``y`` (+y = left)."""
    return "left" if y >= 0 else "right"


def side_range(sign: float, near: float, far: float) -> tuple[float, float]:
    """Return the y range ``[near, far]`` on the side ``sign`` (+1 = left), ordered."""
    a, b = sign * near, sign * far
    return (min(a, b), max(a, b))


def sample_apart(
    rng: np.random.Generator,
    boxes: list[tuple[tuple[float, float], tuple[float, float]]],
    min_dist: float,
    taken: list[np.ndarray] | None = None,
    tries: int = 400,
) -> list[np.ndarray]:
    """Draw one xy per box ``((x0, x1), (y0, y1))``, all ``min_dist`` apart."""
    taken = list(taken or [])
    out: list[np.ndarray] = []
    for (x0, x1), (y0, y1) in boxes:
        best = None
        for _ in range(tries):
            p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if all(np.linalg.norm(p - q) >= min_dist for q in taken + out):
                best = p
                break
        if best is None:
            best = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        out.append(best)
    return out


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class _InContainer(YamTask):
    """Objects that go into one container (a tray or a bowl)."""

    container = "bin"
    inner: tuple[float, float] = (0.2, 0.14)  # tray inner size (x, y), metres
    rim = 0.05  # container height, metres
    round_container = False
    radius = 0.1  # bowl radius

    def container_xy(self) -> np.ndarray:
        assert self.model is not None
        return self.model.body_pos[self.model.body(self.container).id][:2].copy()

    def container_yaw(self) -> float:
        assert self.model is not None
        q = self.model.body_quat[self.model.body(self.container).id]
        return 2 * math.atan2(q[3], q[0])

    def target_xy(self, name: str) -> np.ndarray | None:
        return self.container_xy()

    def in_container(self, p: np.ndarray, margin: float = 0.01) -> bool:
        c = self.container_xy()
        if p[2] > self.z0 + self.rim + 0.03:
            return False
        if self.round_container:
            return bool(np.linalg.norm(p[:2] - c) < self.radius - margin)
        a = -self.container_yaw()
        d = p[:2] - c
        local = np.array(
            [
                math.cos(a) * d[0] - math.sin(a) * d[1],
                math.sin(a) * d[0] + math.cos(a) * d[1],
            ]
        )
        half = np.array(self.inner) / 2 - margin
        return bool(np.all(np.abs(local) <= half))

    def item_done(self, data: mujoco.MjData, name: str) -> bool:
        return self.in_container(self.pos(data, name)) and self.resting(data, name)


class ScrewdriverInBin(_InContainer):
    """Put the screwdriver (a handle and a metal shaft) in the open bin."""

    name = "screwdriver_in_bin"
    task_name = "put_the_screwdriver_in_the_bin"
    caption = (
        "Pick up the screwdriver from the center of the table, "
        "then place the screwdriver in the bin"
    )
    default_time_limit_s = 20.0
    item_names = ("screwdriver",)
    inner = (0.22, 0.15)
    rim = 0.05

    def describe(self) -> str:
        return (
            "The screwdriver centre is inside the bin walls, at rest, and no "
            "gripper holds it, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        add_tray(
            spec,
            self.container,
            self.inner,
            self.rim,
            (0.95, 0.8, 0.1, 1.0),
            z0=self.z0,
        )
        # The body origin is the centre of the handle. The tool points along +x.
        cyl = mujoco.mjtGeom.mjGEOM_CAPSULE
        self.add_item(
            spec,
            "screwdriver",
            [
                {
                    "type": cyl,
                    "size": [0.014, 0.04, 0],
                    "quat": [0.7071, 0, 0.7071, 0],
                    "rgba": [0.8, 0.12, 0.1, 1.0],
                    "mass": 0.06,
                },
                {
                    "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                    "size": [0.0035, 0.045, 0],
                    "pos": [0.098, 0, -0.008],
                    "quat": [0.7071, 0, 0.7071, 0],
                    "rgba": [0.75, 0.75, 0.78, 1.0],
                    "mass": 0.02,
                },
            ],
        )

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        side = 1.0 if rng.integers(2) == 0 else -1.0
        bin_xy = np.array([rng.uniform(0.64, 0.72), rng.uniform(-0.07, 0.07)])
        self.move_static(self.container, bin_xy, rng.uniform(-0.25, 0.25))
        xy = sample_apart(
            rng,
            [((0.47, 0.6), side_range(side, 0.1, 0.24))],
            0.0,
        )[0]
        yaw = rng.uniform(-math.pi, math.pi)
        self.set_pose(
            data,
            "screwdriver",
            np.array([xy[0], xy[1], self.z0 + 0.0145]),
            yaw_quat(yaw),
        )

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        p = self.pos(data, "screwdriver")
        arm = side_of(p[1])
        axis = self.rot(data, "screwdriver")[:, 0]
        plan.pick(arm, "screwdriver", p, finger_axis_perp=axis)
        plan.place(arm, "screwdriver", lambda: self.drop_point(0.06), release=True)
        plan.home(arm)

    def drop_point(self, height: float) -> np.ndarray:
        c = self.container_xy()
        return np.array([c[0], c[1], self.z0 + height])


class FruitInBowl(_InContainer):
    """Put the two fake fruits (one on each side) in the fruit bowl."""

    name = "fruit_in_bowl"
    task_name = "place_and_organize_the_fake_fruits_in_the_fruit_bowl"
    default_time_limit_s = 32.0
    item_names = ("orange", "lemon")
    container = "fruit_bowl"
    round_container = True
    radius = 0.1
    rim = 0.045

    def describe(self) -> str:
        return (
            "Both fruits are inside the bowl rim, at rest, and no gripper "
            "holds them, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        add_ring(
            spec,
            self.container,
            self.radius,
            self.rim,
            (0.45, 0.75, 0.4, 1.0),
            flare=0.35,
            z0=self.z0,
        )
        roll = {"condim": 6, "friction": [1.0, 0.01, 0.002]}
        sph = mujoco.mjtGeom.mjGEOM_SPHERE
        self.add_item(
            spec,
            "orange",
            [
                {
                    "type": sph,
                    "size": [0.03, 0, 0],
                    "rgba": [0.98, 0.5, 0.05, 1],
                    "mass": 0.05,
                    **roll,
                }
            ],
        )
        self.add_item(
            spec,
            "lemon",
            [
                {
                    "type": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    "size": [0.036, 0.027, 0.027],
                    "rgba": [0.98, 0.9, 0.2, 1],
                    "mass": 0.05,
                    **roll,
                }
            ],
        )

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        bowl = np.array([rng.uniform(0.64, 0.72), rng.uniform(-0.05, 0.05)])
        self.move_static(self.container, bowl)
        sides = [1.0, -1.0]
        rng.shuffle(sides)
        boxes = [((0.45, 0.6), side_range(s, 0.1, 0.27)) for s in sides]
        xys = sample_apart(
            rng,
            boxes,
            0.09,
            taken=[bowl, bowl + np.array([0, 0.06]), bowl - np.array([0, 0.06])],
        )
        z = {"orange": 0.03, "lemon": 0.027}
        for n, xy in zip(self.item_names, xys, strict=True):
            self.set_pose(
                data,
                n,
                np.array([xy[0], xy[1], self.z0 + z[n] + 1e-3]),
                yaw_quat(rng.uniform(-3, 3)),
            )

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        per = {s: [] for s in SIDES}
        for n in self.item_names:
            p = self.pos(data, n)
            per[side_of(p[1])].append((abs(p[1]), n))
        for side, items in per.items():
            sign = 1.0 if side == "left" else -1.0
            for k, (_, n) in enumerate(sorted(items)):
                plan.pick(side, n, self.pos(data, n), live=True)
                off = np.array([0.03 * (k % 2) - 0.015, sign * 0.035])
                plan.place(
                    side, n, lambda o=off: self.drop(o), release=True, lock="bowl"
                )
            plan.home(side)

    def drop(self, off: np.ndarray) -> np.ndarray:
        c = self.container_xy() + off
        return np.array([c[0], c[1], self.z0 + 0.06])


class BottlesInBin(_InContainer):
    """Throw the two plastic bottles (lying cylinders) in the tall bin."""

    name = "bottles_in_bin"
    task_name = "throw_the_plastic_bottles_in_the_bin"
    default_time_limit_s = 36.0
    item_names = ("bottle_a", "bottle_b")
    container = "trash_bin"
    inner = (0.17, 0.2)
    rim = 0.11

    def describe(self) -> str:
        return (
            "Both bottles are inside the bin walls, below the rim, at rest, "
            "and no gripper holds them, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        add_tray(
            spec,
            self.container,
            self.inner,
            self.rim,
            (0.35, 0.75, 0.2, 1.0),
            z0=self.z0,
        )
        q = [0.7071, 0, 0.7071, 0]  # the bottle axis along the body x axis
        # Rolling friction: a soft plastic bottle does not roll far.
        roll = {"condim": 6, "friction": [1.0, 0.02, 0.01]}
        for n, cap in (
            ("bottle_a", (0.1, 0.3, 0.85, 1)),
            ("bottle_b", (0.85, 0.15, 0.1, 1)),
        ):
            self.add_item(
                spec,
                n,
                [
                    {
                        "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                        "size": [0.03, 0.075, 0],
                        "quat": q,
                        "rgba": [0.78, 0.88, 0.95, 0.75],
                        "mass": 0.03,
                        **roll,
                    },
                    {
                        "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                        "size": [0.0302, 0.02, 0],
                        "quat": q,
                        "rgba": list(cap),
                        "mass": 0.002,
                        **roll,
                    },
                    {
                        "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                        "size": [0.013, 0.012, 0],
                        "pos": [0.087, 0, 0],
                        "quat": q,
                        "rgba": list(cap),
                        "mass": 0.003,
                        **roll,
                    },
                ],
            )

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        bin_xy = np.array([rng.uniform(0.68, 0.73), rng.uniform(-0.04, 0.04)])
        self.move_static(self.container, bin_xy)
        first = 1.0 if rng.integers(2) == 0 else -1.0
        boxes = [((0.47, 0.6), side_range(s, 0.08, 0.24)) for s in (first, -first)]
        xys = sample_apart(rng, boxes, 0.15)
        for n, xy in zip(self.item_names, xys, strict=True):
            yaw = rng.uniform(-0.6, 0.6) + (math.pi / 2 if rng.integers(2) else 0.0)
            self.set_pose(
                data, n, np.array([xy[0], xy[1], self.z0 + 0.031]), yaw_quat(yaw)
            )

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        for n in self.item_names:
            p = self.pos(data, n)
            side = side_of(p[1])
            sign = 1.0 if side == "left" else -1.0
            plan.pick(side, n, p, finger_axis_perp=self.rot(data, n)[:, 0], live=True)
            plan.place(
                side,
                n,
                lambda s=sign: self.drop(s),
                release=True,
                lock="bin",
                carry_z=self.z0 + self.rim + 0.1,
                descend=False,
            )
            plan.home(side)

    def drop(self, sign: float) -> np.ndarray:
        c = self.container_xy()
        return np.array([c[0] - 0.02, c[1] + sign * 0.03, self.z0 + self.rim + 0.07])


class CupByCoaster(YamTask):
    """Put the cup upright on the coaster."""

    name = "cup_by_coaster"
    task_name = "place_the_cup_by_the_coaster"
    default_time_limit_s = 20.0
    item_names = ("cup",)
    radius = 0.034
    height = 0.085
    coaster_r = 0.05

    def describe(self) -> str:
        return (
            "The cup stands upright (tilt < 15 deg) with its centre within "
            "6 cm of the coaster centre, at rest, not held, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        body = spec.worldbody.add_body(name="coaster", pos=[0.6, 0, self.z0])
        body.add_geom(
            name="coaster_disc",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.coaster_r, 0.003, 0],
            pos=[0, 0, 0.003],
            rgba=[0.55, 0.36, 0.2, 1],
        )
        # The cup: a solid cylinder with a dark disc on top (the opening).
        h = self.height / 2
        self.add_item(
            spec,
            "cup",
            [
                {
                    "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                    "size": [self.radius, h, 0],
                    "pos": [0, 0, h],
                    "rgba": [0.95, 0.95, 0.92, 1],
                    "mass": 0.12,
                },
                {
                    "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                    "size": [self.radius - 0.004, 0.0005, 0],
                    "pos": [0, 0, self.height + 0.0005],
                    "rgba": [0.3, 0.25, 0.22, 1],
                    "mass": 0.001,
                    "contype": 0,
                    "conaffinity": 0,
                },
                {
                    "type": mujoco.mjtGeom.mjGEOM_BOX,
                    "size": [0.006, 0.004, 0.026],
                    "pos": [self.radius + 0.012, 0, h + 0.004],
                    "rgba": [0.95, 0.95, 0.92, 1],
                    "mass": 0.005,
                },
            ],
        )

    def bind(self, model: mujoco.MjModel, arms: dict[str, YamArm]) -> None:
        super().bind(model, arms)
        self._coaster = model.body("coaster").id

    def coaster_xy(self) -> np.ndarray:
        assert self.model is not None
        return self.model.body_pos[self._coaster][:2].copy()

    def target_xy(self, name: str) -> np.ndarray | None:
        return self.coaster_xy()

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        # One arm carries the cup: the coaster is on the side of the cup, or
        # less than 5 cm past the middle.
        for _ in range(200):
            cup = np.array([rng.uniform(0.47, 0.62), rng.uniform(-0.22, 0.22)])
            co = np.array([rng.uniform(0.5, 0.72), rng.uniform(-0.2, 0.2)])
            same_side = math.copysign(1.0, cup[1]) * co[1] > -0.05
            if np.linalg.norm(cup - co) > 0.17 and abs(cup[1]) > 0.06 and same_side:
                break
        self.move_static("coaster", co)
        self.set_pose(
            data,
            "cup",
            np.array([cup[0], cup[1], self.z0 + 1e-3]),
            yaw_quat(rng.uniform(-3, 3)),
        )

    def upright(self, data: mujoco.MjData) -> float:
        return float(self.rot(data, "cup")[2, 2])

    def item_done(self, data: mujoco.MjData, name: str) -> bool:
        p = self.pos(data, "cup")
        near = np.linalg.norm(p[:2] - self.coaster_xy()) < 0.06
        return bool(
            near
            and self.upright(data) > math.cos(math.radians(15))
            and p[2] < self.z0 + 0.012
            and self.resting(data, "cup")
        )

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        p = self.pos(data, "cup")
        side = side_of(p[1])
        grasp = p + np.array([0, 0, 0.055])
        plan.pick(side, "cup", grasp, avoid_axis=self.rot(data, "cup")[:, 0])
        plan.place(
            side,
            "cup",
            lambda: np.array([*self.coaster_xy(), self.z0 + 0.008]),
            release=True,
        )
        plan.home(side)


class BlockTower(YamTask):
    """Stack the wood block on the base block (a tower of two)."""

    name = "block_tower"
    task_name = "build_the_wood_block_tower"
    default_time_limit_s = 24.0
    item_names = ("block_base", "block_top")
    half = (0.035, 0.0175, 0.0175)

    def describe(self) -> str:
        return (
            "The two blocks form a tower: the top block rests on the base block "
            "(centres within 2.5 cm in xy, one block height apart), not held, "
            "for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        wood = ((0.86, 0.7, 0.48, 1), (0.8, 0.62, 0.4, 1), (0.9, 0.76, 0.55, 1))
        for n, c in zip(self.item_names, wood[: len(self.item_names)], strict=True):
            self.add_item(
                spec,
                n,
                [
                    {
                        "type": mujoco.mjtGeom.mjGEOM_BOX,
                        "size": list(self.half),
                        "rgba": list(c),
                        "mass": 0.04,
                    }
                ],
            )

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        side = 1.0 if rng.integers(2) == 0 else -1.0  # the side of the top block
        base, top = sample_apart(
            rng,
            [
                ((0.54, 0.64), side_range(-side, -0.02, 0.08)),
                ((0.46, 0.6), side_range(side, 0.1, 0.22)),
            ],
            0.12,
        )
        z = self.z0 + self.half[2] + 1e-4
        for n, xy in (("block_base", base), ("block_top", top)):
            self.set_pose(
                data, n, np.array([xy[0], xy[1], z]), yaw_quat(rng.uniform(-1.2, 1.2))
            )

    def on_start(self, data: mujoco.MjData) -> None:
        self.level_best = 0.0

    def on_top(self, data: mujoco.MjData, upper: str, lower: str) -> bool:
        a, b = self.pos(data, upper), self.pos(data, lower)
        dz = a[2] - b[2]
        return bool(
            np.linalg.norm(a[:2] - b[:2]) < 0.025 and abs(dz - 2 * self.half[2]) < 0.008
        )

    def tower(self, data: mujoco.MjData) -> list[str]:
        """Return the blocks of the highest stack, bottom first."""
        best: list[str] = []
        names = list(self.item_names)
        for bottom in names:
            p = self.pos(data, bottom)
            if p[2] > self.z0 + self.half[2] + 0.006:
                continue
            stack = [bottom]
            while True:
                nxt = [
                    n
                    for n in names
                    if n not in stack and self.on_top(data, n, stack[-1])
                ]
                if not nxt:
                    break
                stack.append(nxt[0])
            if len(stack) > len(best):
                best = stack
        return best

    def condition(self, data: mujoco.MjData) -> bool:
        st = self.tower(data)
        return (
            len(st) == len(self.item_names)
            and not any(self.grasped(data, n) for n in st)
            and all(self.resting(data, n) for n in st)
        )

    def progress(self, data: mujoco.MjData) -> float:
        st = self.tower(data)
        level = max(len(st) - 1, 0)  # blocks on the tower above the bottom one
        if level >= len(self.item_names) - 1 and not any(
            self.grasped(data, n) for n in st
        ):
            return 1.0
        top = self.pos(data, st[-1]) if st else self.pos(data, "block_base")
        part = 0.0
        for n in self.item_names:
            if n in st:
                continue
            it = self.items[n]
            assert it.start is not None
            p = self.pos(data, n)
            d0 = max(float(np.linalg.norm(it.start[:2] - top[:2])), 0.05)
            tr = float(np.clip(1 - np.linalg.norm(p[:2] - top[:2]) / d0, 0, 1))
            lift = float(
                np.clip((p[2] - it.start[2]) / (2 * self.half[2] + 0.01), 0, 1)
            )
            d = min(float(np.linalg.norm(q - p)) for q in self.pinches())
            app = float(np.clip(1 - d / it.d0, 0, 1))
            part = max(part, 0.15 * app + 0.15 * lift + 0.6 * min(tr, 0.5 + 0.5 * lift))
        steps = len(self.item_names) - 1
        self.level_best = max(self.level_best, (level + min(part, 0.95)) / steps)
        return float(self.level_best)

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        h = 2 * self.half[2]
        p = self.pos(data, "block_top")
        side = side_of(p[1])
        plan.pick(
            side,
            "block_top",
            p,
            finger_axis_perp=self.rot(data, "block_top")[:, 0],
        )
        plan.place(
            side,
            "block_top",
            lambda: self.pos_live("block_base") + np.array([0, 0, h + 0.004]),
            release=True,
            yaw=lambda: self.yaw_live("block_base"),
        )
        plan.home(side)

    def pos_live(self, name: str) -> np.ndarray:
        assert self.live is not None
        return self.pos(self.live, name)

    def yaw_live(self, name: str) -> float:
        assert self.live is not None
        return self.yaw(self.live, name)


class MugUpright(YamTask):
    """Turn the mug, which lies on its side, right side up."""

    name = "mug_upright"
    task_name = "turn_the_mug_right_side_up"
    default_time_limit_s = 24.0
    item_names = ("mug",)
    radius = 0.034
    height = 0.085

    def describe(self) -> str:
        return (
            "The mug stands on its base with the opening up (axis within 20 "
            "deg of vertical), at rest, not held, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        # Body frame: z = the mug axis, from the base (z = 0) to the opening.
        r, h = self.radius, self.height
        col = [0.25, 0.45, 0.75, 1]
        geoms: list[dict] = [
            {
                "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "size": [r, 0.004, 0],
                "pos": [0, 0, 0.004],
                "rgba": col,
                "mass": 0.03,
            },
        ]
        n = 12
        seg = r * math.tan(math.pi / n) * 1.1
        for k in range(n):
            a = 2 * math.pi * k / n
            geoms.append(
                {
                    "type": mujoco.mjtGeom.mjGEOM_BOX,
                    "size": [0.003, seg, h / 2],
                    "pos": [
                        (r - 0.003) * math.cos(a),
                        (r - 0.003) * math.sin(a),
                        h / 2,
                    ],
                    "quat": [math.cos(a / 2), 0, 0, math.sin(a / 2)],
                    "rgba": col,
                    "mass": 0.008,
                }
            )
        geoms.append(
            {
                "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "size": [r - 0.006, 0.0008, 0],
                "pos": [0, 0, 0.0088],
                "rgba": [0.92, 0.92, 0.9, 1],
                "mass": 0.001,
                "contype": 0,
                "conaffinity": 0,
            }
        )
        geoms.append(
            {
                "type": mujoco.mjtGeom.mjGEOM_BOX,
                "size": [0.005, 0.004, 0.024],
                "pos": [r + 0.016, 0, h / 2],
                "rgba": col,
                "mass": 0.006,
            }
        )
        for z in (h / 2 - 0.02, h / 2 + 0.02):
            geoms.append(
                {
                    "type": mujoco.mjtGeom.mjGEOM_BOX,
                    "size": [0.009, 0.004, 0.004],
                    "pos": [r + 0.007, 0, z],
                    "rgba": col,
                    "mass": 0.002,
                }
            )
        self.add_item(
            spec,
            "mug",
            [{**g, "condim": 6, "friction": [1.0, 0.01, 0.002]} for g in geoms],
        )

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        sign = 1.0 if rng.integers(2) == 0 else -1.0
        xy = np.array([rng.uniform(0.5, 0.6), sign * rng.uniform(0.1, 0.18)])
        # The opening points along (cos psi, sin psi, 0), forward and away from
        # the middle (a 90 deg turn of this mug is in reach of the arm).
        psi = sign * math.radians(rng.uniform(20.0, 45.0))
        d = np.array([math.cos(psi), math.sin(psi), 0.0])
        # Turn the body z axis onto d, and roll the handle down to the table.
        roll = math.pi / 2 + rng.uniform(-0.1, 0.1)
        q_axis = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            q_axis, np.array([-math.sin(psi), math.cos(psi), 0.0]), math.pi / 2
        )
        q_roll = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_roll, np.array([0.0, 0.0, 1.0]), roll)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_axis, q_roll)
        pos = (
            np.array([xy[0], xy[1], self.z0 + self.radius + 0.002])
            - d * self.height / 2
        )
        self.set_pose(data, "mug", pos, q)

    def axis(self, data: mujoco.MjData) -> np.ndarray:
        return self.rot(data, "mug")[:, 2]

    def center(self, data: mujoco.MjData) -> np.ndarray:
        return self.pos(data, "mug") + self.axis(data) * self.height / 2

    def item_done(self, data: mujoco.MjData, name: str) -> bool:
        up = self.axis(data)[2] > math.cos(math.radians(20))
        low = self.pos(data, "mug")[2] < self.z0 + 0.012
        return bool(up and low and self.resting(data, "mug"))

    def update_item(self, data: mujoco.MjData, name: str) -> None:
        # The transport term of the mug is its uprightness.
        super().update_item(data, name)
        it = self.items[name]
        it.transport = max(it.transport, float(np.clip(self.axis(data)[2], 0.0, 1.0)))

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        c = self.center(data)
        side = side_of(c[1])
        d = self.axis(data).copy()
        d[2] = 0.0
        d /= np.linalg.norm(d)
        # Grasp across the middle, the tool tilted towards the base of the
        # mug. The 90 deg turn then ends with the tool tilted the other way
        # (a level tool at the table is out of reach). Take the first tilt for
        # which the grasp, the turn and the set-down poses are in reach.
        from praxis_sim.yam_scripted import tool_rotation

        pitch = 45.0
        axis = np.cross(d, np.array([0.0, 0.0, 1.0]))
        turn = np.zeros(9)
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, axis, math.pi / 2)
        mujoco.mju_quat2Mat(turn, q)
        for cand in (45.0, 35.0, 55.0, 25.0, 65.0):
            g = tool_rotation(np.array([-d[1], d[0], 0.0]), -d, math.radians(cand))
            after = turn.reshape(3, 3) @ g
            down = np.array([c[0], c[1], self.z0 + self.height / 2 + 0.005])
            poses = (
                (c, g),
                (c + np.array([0, 0, 0.08]), g),
                (c + np.array([0, 0, 0.12]), after),
                (down, after),
            )
            if all(plan.reachable(side, p, r) for p, r in poses):
                pitch = cand
                break
        plan.pick(
            side,
            "mug",
            c,
            finger_axis_perp=d,
            pitch_deg=pitch,
            lift=0.1,
            tilt=-d,
            live=True,
        )
        plan.turn_place(side, "mug", d, self.height / 2)
        plan.home(side)


class LegoSort(YamTask):
    """Sort the red and the blue brick into the container of their colour."""

    name = "lego_sort"
    task_name = "sort_the_legos_into_containers_by_color"
    default_time_limit_s = 30.0
    item_names = ("brick_red", "brick_blue")
    half = (0.02, 0.016, 0.012)
    inner = (0.12, 0.12)
    rim = 0.035
    colors: ClassVar[dict[str, tuple[float, ...]]] = {
        "red": (0.85, 0.12, 0.1, 1.0),
        "blue": (0.15, 0.35, 0.85, 1.0),
    }

    def describe(self) -> str:
        return (
            "Each brick is inside the container of its colour, at rest, "
            "not held, for 0.5 s."
        )

    def build(self, spec: mujoco.MjSpec) -> None:
        for c, rgba in self.colors.items():
            tray = (*(0.55 + 0.45 * v for v in rgba[:3]), 1.0)
            add_tray(spec, f"tray_{c}", self.inner, self.rim, tray, z0=self.z0)
            self.add_item(
                spec,
                f"brick_{c}",
                [
                    {
                        "type": mujoco.mjtGeom.mjGEOM_BOX,
                        "size": list(self.half),
                        "rgba": list(rgba),
                        "mass": 0.02,
                    }
                ],
            )

    def tray_xy(self, color: str) -> np.ndarray:
        assert self.model is not None
        return self.model.body_pos[self.model.body(f"tray_{color}").id][:2].copy()

    def target_xy(self, name: str) -> np.ndarray | None:
        return self.tray_xy(name.split("_")[1])

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        red_left = bool(rng.integers(2) == 0)
        for c, s in (
            ("red", 1.0 if red_left else -1.0),
            ("blue", -1.0 if red_left else 1.0),
        ):
            self.move_static(
                f"tray_{c}",
                np.array([rng.uniform(0.66, 0.74), s * rng.uniform(0.13, 0.2)]),
            )
        a, b = sample_apart(
            rng, [((0.46, 0.58), (0.03, 0.13)), ((0.46, 0.58), (-0.13, -0.03))], 0.08
        )
        names = list(self.item_names)
        rng.shuffle(names)
        for n, xy in zip(names, (a, b), strict=True):
            self.set_pose(
                data,
                n,
                np.array([xy[0], xy[1], self.z0 + self.half[2] + 1e-4]),
                yaw_quat(rng.uniform(-1.5, 1.5)),
            )

    def item_done(self, data: mujoco.MjData, name: str) -> bool:
        p = self.pos(data, name)
        c = self.tray_xy(name.split("_")[1])
        inside = bool(np.all(np.abs(p[:2] - c) <= np.array(self.inner) / 2 - 0.008))
        return inside and p[2] < self.z0 + self.rim and self.resting(data, name)

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        jobs = []
        for n in self.item_names:
            tgt = self.tray_xy(n.split("_")[1])
            jobs.append((side_of(tgt[1]), n, tgt))
        jobs.sort(key=lambda j: j[0] != "left")
        for k, (side, n, tgt) in enumerate(jobs):
            if k == 1:
                plan.wait(side, lambda: plan.lock_free("picks"))
            plan.pick(
                side,
                n,
                self.pos(data, n),
                finger_axis_perp=self.rot(data, n)[:, 0],
                live=True,
                lock="picks",
            )
            plan.place(
                side,
                n,
                lambda t=tgt: np.array([t[0], t[1], self.z0 + 0.04]),
                release=True,
            )
            plan.home(side)


class _IntoBag(_InContainer):
    """One item into the paper bag at the back centre of the table.

    The bag is a tall open box (kraft paper colour, a rope handle on each side). Success: the item (any one of ``item_names``
    for ``any_item``) is inside the bag below the rim, at rest, not held, for 0.5 s.
    """

    container = "bag"
    inner = (0.12, 0.2)
    rim = 0.2
    any_item = False
    item_r = 0.03

    def describe(self) -> str:
        what = "One item" if self.any_item else "The item"
        return f"{what} is inside the bag walls, below the rim, at rest, not held, for 0.5 s."

    def build_bag(self, spec: mujoco.MjSpec) -> None:
        add_tray(
            spec,
            self.container,
            self.inner,
            self.rim,
            (0.62, 0.45, 0.28, 1.0),
            wall=0.004,
            z0=self.z0,
        )
        b = spec.body(self.container)
        for sgn in (1.0, -1.0):
            _box(
                b,
                f"bag_handle{int(sgn > 0)}",
                [0, sgn * (self.inner[1] / 2 + 0.002), self.rim + 0.03],
                [0.004, 0.003, 0.03],
                (0.85, 0.8, 0.7, 1.0),
            )

    def condition(self, data: mujoco.MjData) -> bool:
        done = [
            self.item_done(data, n) and not self.grasped(data, n)
            for n in self.item_names
        ]
        return any(done) if self.any_item else all(done)

    def progress(self, data: mujoco.MjData) -> float:
        vals = [self.item_progress(it) for it in self.items.values()]
        return float(max(vals) if self.any_item else np.mean(vals))

    def bag_gap(self, xy: np.ndarray, bag_xy: np.ndarray) -> float:
        """Return the xy distance from ``xy`` to the outside of the bag walls."""
        half = np.array(self.inner) / 2 + 0.004
        d = np.maximum(np.abs(np.asarray(xy) - bag_xy) - half, 0.0)
        return float(np.hypot(d[0], d[1]))

    def place(self, rng: np.random.Generator, data: mujoco.MjData) -> None:
        bag_xy = np.array([rng.uniform(0.66, 0.72), rng.uniform(-0.03, 0.03)])
        self.move_static(self.container, bag_xy)
        n = len(self.item_names)
        sides = [1.0 if (k + rng.integers(2)) % 2 else -1.0 for k in range(n)]
        boxes = [((0.42, 0.6), side_range(sg, 0.06, 0.26)) for sg in sides]
        # A gripper cannot reach down to an item next to the tall bag wall, so
        # the items are drawn again (same generator) until all clear the bag.
        # If no draw clears it, the draw with the largest smallest gap stays.
        best, best_gap = None, -1.0
        for _ in range(BAG_PLACE_TRIES):
            draw = sample_apart(rng, boxes, 0.08)
            gap = min(self.bag_gap(xy, bag_xy) for xy in draw)
            if gap > best_gap:
                best, best_gap = draw, gap
            if gap >= BAG_CLEARANCE:
                break
        xys = best
        for name, xy in zip(self.item_names, xys, strict=True):
            self.set_pose(
                data,
                name,
                np.array([xy[0], xy[1], self.z0 + self.item_r + 0.002]),
                yaw_quat(rng.uniform(-math.pi, math.pi)),
            )

    def oracle(self, plan: Plan, data: mujoco.MjData) -> None:
        names = self.item_names[:1] if self.any_item else self.item_names
        for n in names:
            p = self.pos(data, n)
            side = side_of(p[1])
            plan.pick(side, n, p, finger_axis_perp=self.rot(data, n)[:, 0], live=True)
            c = self.container_xy()
            plan.place(
                side,
                n,
                lambda c=c: np.array([c[0], c[1], self.z0 + self.rim + 0.08]),
                release=True,
                lock="bin",
                carry_z=self.z0 + self.rim + 0.12,
                descend=False,
            )
            plan.home(side)


class DiceInBag(_IntoBag):
    """Foam dice on the table: the first die in the bag ends the episode."""

    name = "dice_in_bag"
    task_name = "put_the_foam_stress_ball_in_the_bag"
    caption = (
        "Pick up the foam dice stress ball from the center of the table, "
        "then place the foam dice stress ball in the bag"
    )
    default_time_limit_s = 60.0
    item_names = (
        "die_red_a",
        "die_red_b",
        "die_black",
        "die_white",
        "die_red_c",
        "die_black_b",
    )
    any_item = True
    item_r = 0.0175

    def build(self, spec: mujoco.MjSpec) -> None:
        self.build_bag(spec)
        cols = {
            "red": (0.85, 0.1, 0.1, 1),
            "black": (0.08, 0.08, 0.08, 1),
            "white": (0.92, 0.92, 0.9, 1),
        }
        for n in self.item_names:
            c = cols[n.split("_")[1]]
            self.add_item(
                spec,
                n,
                [
                    {
                        "type": mujoco.mjtGeom.mjGEOM_BOX,
                        "size": [0.0175, 0.0175, 0.0175],
                        "rgba": list(c),
                        "mass": 0.01,
                        "friction": [1.2, 0.01, 0.002],
                    }
                ],
            )


class _OneItemInBag(_IntoBag):
    geoms: ClassVar[list[dict]] = []

    def build(self, spec: mujoco.MjSpec) -> None:
        self.build_bag(spec)
        self.add_item(spec, self.item_names[0], self.geoms)


class TennisBallInBag(_OneItemInBag):
    name = "tennis_ball_in_bag"
    task_name = "put_the_tennis_ball_in_the_bag"
    caption = (
        "Pick up the tennis ball from the center of the table, "
        "then place the tennis ball in the bag"
    )
    default_time_limit_s = 60.0
    item_names = ("tennis_ball",)
    item_r = 0.033
    geoms: ClassVar[list[dict]] = [
        {
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "size": [0.033, 0, 0],
            "rgba": [0.8, 0.9, 0.2, 1],
            "mass": 0.057,
            "condim": 6,
            "friction": [1.0, 0.02, 0.01],
        }
    ]


class AppleInBag(_OneItemInBag):
    name = "apple_in_bag"
    task_name = "put_the_fake_apple_in_the_bag"
    caption = (
        "Pick up the fake apple from the center of the table, "
        "then place the fake apple in the bag"
    )
    default_time_limit_s = 60.0
    item_names = ("apple",)
    item_r = 0.036
    geoms: ClassVar[list[dict]] = [
        {
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "size": [0.036, 0, 0],
            "rgba": [0.8, 0.08, 0.08, 1],
            "mass": 0.05,
            "condim": 6,
            "friction": [1.0, 0.02, 0.01],
        }
    ]


class BellInBag(_OneItemInBag):
    """A desk bell: black base disc + chrome dome + plunger."""

    name = "bell_in_bag"
    task_name = "put_the_bell_in_the_bag"
    caption = (
        "Pick up the bell from the center of the table, then place the bell in the bag"
    )
    default_time_limit_s = 60.0
    item_names = ("bell",)
    item_r = 0.008
    geoms: ClassVar[list[dict]] = [
        {
            "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "size": [0.042, 0.008, 0],
            "rgba": [0.1, 0.1, 0.1, 1],
            "mass": 0.08,
        },
        {
            "type": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "size": [0.036, 0.036, 0.026],
            "pos": [0, 0, 0.022],  # the dome bottom 4 mm above the base bottom
            "rgba": [0.8, 0.8, 0.82, 1],
            "mass": 0.04,
        },
        {
            "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "size": [0.004, 0.008, 0],
            "pos": [0, 0, 0.054],
            "rgba": [0.1, 0.1, 0.1, 1],
            "mass": 0.005,
        },
    ]


YAM_TASKS: dict[str, type[YamTask]] = {
    t.name: t
    for t in (
        ScrewdriverInBin,
        FruitInBowl,
        BottlesInBin,
        CupByCoaster,
        BlockTower,
        MugUpright,
        LegoSort,
        DiceInBag,
        TennisBallInBag,
        AppleInBag,
        BellInBag,
    )
}


def make_yam_task(cfg: YamTaskConfig, table_top_z: float = TABLE_TOP_Z) -> YamTask:
    """Return a new task instance of the bimanual rig."""
    if cfg.name not in YAM_TASKS:
        raise ValueError(
            f"Unknown task {cfg.name!r}. The known tasks are {sorted(YAM_TASKS)}."
        )
    return YAM_TASKS[cfg.name](cfg, table_top_z)
