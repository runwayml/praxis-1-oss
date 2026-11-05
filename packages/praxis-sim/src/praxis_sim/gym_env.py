"""Gymnasium wrapper of the praxis MuJoCo single-arm rig and its evaluation suites.

One ``PraxisSimGymEnv`` is one task (``lift_cube``, ``block_in_bowl``, ...) on the
DROID-like rig of the praxis repository: a Franka arm with a Robotiq 2F-85
gripper, an exterior camera and a wrist camera rendered at 512 x 288, 15 Hz
control. The reset seed is the scene seed: it places the objects (and picks the
target color or direction of the prompted tasks). The observation holds the two
camera views, the pose ``[x, y, z, roll, pitch, yaw, gripper]`` of the Franka
flange and the scene seed; the action is the same seven values, an absolute
target. Success is the task's physics check; an episode ends ``after_success_s``
after success or at the task time limit.

``create_praxis_sim_envs`` builds the ``{suite: {task_index: VectorEnv}}`` layout
of the fixed evaluation suites (``droid``, ``droid_guard``, ``hard``), the same
episodes as ``scripts/run_sim_eval.py`` of the praxis repository: scene seeds
``0 .. N - 1`` are the LeRobot seeds ``start_seed .. start_seed + N - 1``, and
the policy's sampler seed follows ``seed + 1000 * scene_seed + k`` through the
``scene_seed`` observation. The package has no LeRobot dependency: LeRobot's
``PraxisSimEnv`` config calls :func:`create_praxis_sim_envs`.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import RobotConfig
from .env import EnvConfig as SimEnvConfig
from .env import Observation, SimEnv
from .eval_suites import SUITES
from .pose import (
    TURNED_FLANGE,
    euler_xyz_to_matrix,
    pose8_to_droid7,
    pose8_to_model,
    pose_to_rig,
)
from .scene import CONTROL_HZ, SceneConfig
from .tasks import TASKS, TaskConfig, _register_suite, make_task

__all__ = ["PraxisSimGymEnv", "create_praxis_sim_envs", "resolve_tasks"]

# The suite tasks register lazily in praxis; the wrapper needs the full registry at import.
_register_suite()

STATE_DIM = 7
SCENE_SEED_KEY = "scene_seed"


def resolve_tasks(task: str) -> tuple[str, list[str]]:
    """Return ``(suite name, task names)`` for a suite name or a comma list of task names."""
    name = task.strip()
    if name in SUITES:
        return name, list(SUITES[name].tasks)
    names = [t.strip() for t in name.split(",") if t.strip()]
    unknown = [t for t in names if t not in TASKS]
    if unknown or not names:
        raise ValueError(
            f"unknown praxis sim task(s) {unknown or name!r}; suites: {sorted(SUITES)}, tasks: {sorted(TASKS)}"
        )
    return "praxis_sim", names


class PraxisSimGymEnv(gym.Env):
    """One praxis sim task as a Gymnasium environment (see the module docstring)."""

    # An int: the video writer of lerobot-eval passes it to PyAV as the stream rate.
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": round(CONTROL_HZ),
    }

    def __init__(
        self,
        task: str,
        *,
        scene: SceneConfig | None = None,
        robot: RobotConfig | None = None,
        prompt: str | None = None,
        time_limit_s: float | None = None,
        after_success_s: float = 1.0,
        render_overview: bool = True,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(
                f"unknown praxis sim task {task!r}; known tasks: {sorted(TASKS)}"
            )
        self.task_name = task
        self.render_mode = render_mode
        self.render_overview = render_overview
        self._scene = scene or SceneConfig()
        robot = robot or RobotConfig()
        self._task_cfg = TaskConfig(name=task, prompt=prompt, time_limit_s=time_limit_s)
        self._sim_cfg = SimEnvConfig(
            scene=self._scene,
            task=self._task_cfg,
            frame=robot.frame,
            ik=robot.ik,
            home=robot.home,
            home_noise=robot.home_noise,
            settle_s=robot.settle_s,
            feedforward=robot.feedforward,
            render=True,
            render_overview=render_overview,
        )
        probe = make_task(replace(self._task_cfg, table_top_z=self._scene.table_top_z))
        # The prompt of the current episode. Some tasks choose a direction or a color per scene
        # seed ("Move the red block to the left."), so ``reset`` updates it from the placed task.
        self.task_description: str = probe.prompt
        self._time_limit_ticks = int(probe.max_ticks)
        self._linger_ticks = round(after_success_s * CONTROL_HZ)
        # The rollout runs at most this many steps: the time limit plus the time after success.
        self._max_episode_steps = self._time_limit_ticks + self._linger_ticks

        self._env: SimEnv | None = None
        self._obs: Observation | None = None
        self._scene_seed = 0
        self._ticks = 0
        self._success_tick: int | None = None

        h, w = self._scene.image_height, self._scene.image_width
        image = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"exterior": image, "wrist": image}),
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
                SCENE_SEED_KEY: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        return self.task_name

    def _ensure_env(self) -> SimEnv:
        if self._env is None:
            self._env = SimEnv(self._sim_cfg)
        return self._env

    def _format(self, obs: Observation) -> dict[str, Any]:
        if obs.exterior is None or obs.wrist is None:
            raise RuntimeError(
                "the praxis sim did not render the exterior and wrist views"
            )
        # The sim reports the ``ee`` site (the flange turned -45 deg); LeRobot carries the flange.
        pose8_sim = np.concatenate(
            [np.asarray(obs.ee_pose, dtype=np.float32), [np.float32(obs.grip)]]
        )
        pose8 = pose8_to_model(pose8_sim, TURNED_FLANGE)
        return {
            "pixels": {"exterior": obs.exterior, "wrist": obs.wrist},
            "agent_pos": pose8_to_droid7(pose8).astype(np.float32),
            SCENE_SEED_KEY: np.array([self._scene_seed], dtype=np.float32),
        }

    def _info(self) -> dict[str, Any]:
        env = self._ensure_env()
        status = env.task.status
        return {
            "task": self.task_name,
            "prompt": self.task_description,
            "scene_seed": self._scene_seed,
            "is_success": bool(status.success),
            # -1 before success: vector envs stack info values into typed arrays, so no None here.
            "success_tick": -1
            if status.success_tick is None
            else int(status.success_tick),
            "progress": float(
                status.metrics.get("progress", 1.0 if status.success else 0.0)
            ),
            "ticks": self._ticks,
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None, **kwargs: Any
    ):
        super().reset(seed=seed)
        env = self._ensure_env()
        self._scene_seed = int(seed) if seed is not None else int(self._scene_seed)
        self._obs = env.reset(self._scene_seed)
        self.task_description = env.prompt
        self._ticks = 0
        self._success_tick = None
        return self._format(self._obs), self._info()

    def step(self, action: np.ndarray):
        env = self._ensure_env()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != STATE_DIM:
            raise ValueError(
                f"expected an action of {STATE_DIM} values [xyz, rpy, grip], got {action.shape}"
            )
        rot = euler_xyz_to_matrix(action[None, 3:6])[0]
        xyz, rot = pose_to_rig(action[:3], rot, TURNED_FLANGE)
        pose12 = np.concatenate([xyz, rot.reshape(-1)]).astype(np.float32)
        self._obs = env.step(pose12, float(action[6]))
        self._ticks += 1
        status = env.task.status
        if status.success and self._success_tick is None:
            self._success_tick = self._ticks
        info = self._info()
        reward = float(info["progress"])
        terminated = (
            self._success_tick is not None
            and self._ticks >= self._success_tick + self._linger_ticks
        )
        truncated = not terminated and self._ticks >= self._time_limit_ticks
        return self._format(self._obs), reward, bool(terminated), bool(truncated), info

    def render(self):
        if self._obs is None:
            return None
        frame = (
            self._obs.overview
            if self.render_overview and self._obs.overview is not None
            else self._obs.exterior
        )
        return None if frame is None else np.asarray(frame)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def _build_env(
    wrap: Callable[[gym.Env], gym.Env] | None = None, **kwargs: Any
) -> gym.Env:
    """Build one sim env, through ``wrap`` when given. A module-level function, so async vector envs can pickle it."""
    env: gym.Env = PraxisSimGymEnv(**kwargs)
    return env if wrap is None else wrap(env)


def _make_env_fns(n_envs: int, **kwargs: Any) -> list[Callable[[], gym.Env]]:
    return [functools.partial(_build_env, **kwargs) for _ in range(n_envs)]


def create_praxis_sim_envs(
    task: str,
    n_envs: int,
    *,
    task_ids: Sequence[int] | None = None,
    env_cls: Callable[..., Any] | None = None,
    wrap: Callable[[gym.Env], gym.Env] | None = None,
    **env_kwargs: Any,
) -> dict[str, dict[int, Any]]:
    """Create ``{suite: {task_index: VectorEnv}}`` for a suite name or a comma list of tasks.

    ``task_ids`` restricts the suite to the tasks at those indices (the order of
    ``SUITES[name].tasks``). ``wrap`` is applied to every environment (LeRobot passes its
    freeze-after-episode-end wrapper). ``env_kwargs`` go to :class:`PraxisSimGymEnv`.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    suite, names = resolve_tasks(task)
    selected = (
        list(range(len(names))) if task_ids is None else [int(i) for i in task_ids]
    )
    bad = [i for i in selected if not 0 <= i < len(names)]
    if bad:
        raise ValueError(
            f"task_ids {bad} are outside the {len(names)} tasks of {task!r}"
        )
    env_cls = env_cls or gym.vector.SyncVectorEnv
    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for index in selected:
        fns = _make_env_fns(n_envs, wrap=wrap, task=names[index], **env_kwargs)
        if env_cls is gym.vector.AsyncVectorEnv:
            vec = env_cls(
                fns,
                context="forkserver",
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
        else:
            vec = env_cls(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        out[suite][index] = vec
    return {suite: dict(tasks) for suite, tasks in out.items()}
