"""The simulation environment: reset, one 15 Hz step per command row, observe.

Use it like this::

    env = SimEnv(EnvConfig())
    obs = env.reset(seed=0)
    for pose12, grip in rows:          # command rows of the policy
        obs = env.step(pose12, grip)   # one row = one tick of 1/15 s
    env.close()

``obs.exterior`` and ``obs.wrist`` are the two model views (uint8 RGB).
``obs.ee_pose`` and ``obs.grip`` are in the policy convention of
:mod:`praxis.sim.robot`. ``obs.overview`` is the view for people.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from praxis.sim.robot import (
    DROID_HOME,
    Cameras,
    FrameConfig,
    IkConfig,
    IkResult,
    SimArm,
)
from praxis.sim.scene import TABLE_GEOM, SceneConfig, build_model
from praxis.sim.tasks import Task, TaskConfig, TaskStatus, make_task


@dataclass
class EnvConfig:
    """Settings of the environment. The YAML ``scene``, ``robot`` and ``task`` blocks fill it."""

    scene: SceneConfig = field(default_factory=SceneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    ik: IkConfig = field(default_factory=IkConfig)
    home: tuple[float, ...] = DROID_HOME  # arm joints at reset, radians
    home_noise: float = 0.0  # uniform joint noise at reset, radians
    settle_s: float = 0.5  # physics time before the first observation
    feedforward: bool = True  # joint velocity feed-forward in each tick
    render: bool = True  # render the camera views
    render_overview: bool = True  # also render the overview view


@dataclass
class Observation:
    """One observation of the environment."""

    ee_pose: np.ndarray  # (7,) float32 [xyz, qw, qx, qy, qz], policy convention
    grip: float  # 0 = open, 1 = closed
    exterior: np.ndarray | None  # (H, W, 3) uint8
    wrist: np.ndarray | None  # (H, W, 3) uint8
    overview: np.ndarray | None  # (H, W, 3) uint8
    tick: int
    status: TaskStatus
    ik: IkResult | None = None


class SimEnv:
    """The simulated single-arm rig and one task."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.cfg.task.table_top_z = self.cfg.scene.table_top_z
        self.task: Task = make_task(self.cfg.task)
        self.model = build_model(
            self.cfg.scene, self.task.objects(), self.task.fixtures(), self.task.build
        )
        self.data = mujoco.MjData(self.model)
        self.task.bind(self.model)
        self.arm = SimArm(
            self.model,
            self.data,
            frame=self.cfg.frame,
            ik=self.cfg.ik,
            home=self.cfg.home,
            feedforward=self.cfg.feedforward,
        )
        table = self.model.geom(TABLE_GEOM)
        center, half = table.pos[:2].copy(), table.size[:2].copy()
        self.table_xy = (center - half, center + half)
        self.cameras: Cameras | None = (
            Cameras(self.model, self.cfg.scene) if self.cfg.render else None
        )
        self.tick_count = 0
        self.timing = {"physics_s": 0.0, "render_s": 0.0, "ticks": 0}

    @property
    def prompt(self) -> str:
        """Return the prompt of the task."""
        return self.task.prompt

    @property
    def max_ticks(self) -> int:
        """Return the time limit of an episode in ticks."""
        return self.task.max_ticks

    def reset(self, seed: int = 0) -> Observation:
        """Start a new episode. The seed sets the object placement and the joint noise."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        home = np.asarray(self.cfg.home, dtype=np.float64)
        if self.cfg.home_noise > 0:
            home = home + rng.uniform(-self.cfg.home_noise, self.cfg.home_noise, size=7)
        self.arm.reset(home)
        self.task.place(rng, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.arm.settle(self.cfg.settle_s)
        self.task.start(self.model, self.data)
        self.tick_count = 0
        return self.observe()

    def step(self, pose12: np.ndarray, grip: float) -> Observation:
        """Execute one command row for one tick, then observe."""
        t0 = time.monotonic()
        result = self.arm.tick(pose12, grip)
        self.tick_count += 1
        self.task.update(self.data, self.arm.pinch(), self.arm.grip(), self.table_xy)
        self.timing["physics_s"] += time.monotonic() - t0
        self.timing["ticks"] += 1
        obs = self.observe()
        obs.ik = result
        return obs

    def observe(self) -> Observation:
        """Return the present observation (renders the views when enabled)."""
        views: dict[str, np.ndarray] = {}
        if self.cameras is not None:
            t0 = time.monotonic()
            views = self.cameras.render(self.data, overview=self.cfg.render_overview)
            self.timing["render_s"] += time.monotonic() - t0
        return Observation(
            ee_pose=self.arm.ee_pose(),
            grip=self.arm.grip(),
            exterior=views.get("exterior"),
            wrist=views.get("wrist"),
            overview=views.get("overview"),
            tick=self.tick_count,
            status=self.task.status,
        )

    def done(self) -> bool:
        """Return ``True`` at the time limit."""
        return self.tick_count >= self.max_ticks

    def summary(self) -> dict[str, float | int | bool | None]:
        """Return the episode result: success, progress metrics and counters."""
        s = self.task.status
        out: dict[str, float | int | bool | None] = {
            "success": s.success,
            "success_tick": s.success_tick,
            "ticks": s.ticks,
            "object_off_table": s.failed,
            "ik_misses": self.arm.ik_misses,
            "bad_rows": self.arm.bad_rows,
        }
        out.update({k: float(v) for k, v in s.metrics.items()})
        return out

    def close(self) -> None:
        """Free the renderers."""
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None

    def __enter__(self) -> SimEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
