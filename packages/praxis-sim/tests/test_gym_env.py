"""The Gymnasium wrapper and the comparison tool of praxis-sim."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from praxis_sim.compare import compare, format_report, load_lerobot_run, load_praxis_run
from praxis_sim.eval_suites import SUITES
from praxis_sim.gym_env import PraxisSimGymEnv, create_praxis_sim_envs, resolve_tasks
from praxis_sim.tasks import TASKS

needs_renderer = pytest.mark.skipif(
    not os.environ.get("MUJOCO_GL"),
    reason="needs an offscreen MuJoCo renderer (MUJOCO_GL=egl)",
)


def test_resolve_tasks() -> None:
    suite, names = resolve_tasks("droid_guard")
    assert suite == "droid_guard" and names == list(SUITES["droid_guard"].tasks)
    assert set(names) <= set(TASKS)
    assert resolve_tasks("lift_cube, block_in_bowl") == (
        "praxis_sim",
        ["lift_cube", "block_in_bowl"],
    )
    with pytest.raises(ValueError, match="unknown"):
        resolve_tasks("no_such_task")


def test_compare_pairs_episodes(tmp_path: Path) -> None:
    run = tmp_path / "praxis" / "suite" / "lift_cube"
    run.mkdir(parents=True)
    for seed, success in enumerate([True, True, False]):
        (run / f"episode_{seed:03d}.json").write_text(
            json.dumps(
                {
                    "task": "lift_cube",
                    "seed": seed,
                    "sampler_seed": 0,
                    "success": success,
                    "progress": 1.0,
                }
            )
        )
    eval_info = {
        "per_task": [
            {
                "task_group": "droid_guard",
                "task_id": 0,
                "metrics": {
                    "successes": [True, False, True],
                    "max_rewards": [1.0, 0.4, 1.0],
                    "sum_rewards": [3.0, 1.0, 2.0],
                },
            }
        ]
    }
    (tmp_path / "eval_info.json").write_text(json.dumps(eval_info))
    result = compare(
        load_praxis_run(tmp_path / "praxis"),
        load_lerobot_run(tmp_path / "eval_info.json"),
    )
    task = result["tasks"]["lift_cube"]
    assert task["n"] == 3 and task["a"] == 2 and task["b"] == 2
    assert task["won"] == [2] and task["lost"] == [1]
    assert result["total"]["mcnemar_p"] == 1.0
    assert "lerobot - praxis" in format_report(result)


@needs_renderer
def test_gym_env_reset_step_and_hold() -> None:
    env = PraxisSimGymEnv("block_in_bowl")
    try:
        assert env.task_description == "Put the red block in the bowl."
        assert env.metadata["render_fps"] == 15 and isinstance(
            env.metadata["render_fps"], int
        )
        assert env._max_episode_steps == round(20.6 * 15) + 15
        obs, info = env.reset(seed=3)
        assert (
            obs["pixels"]["exterior"].shape == (288, 512, 3)
            and obs["pixels"]["exterior"].dtype == np.uint8
        )
        assert obs["agent_pos"].shape == (7,) and obs["scene_seed"].tolist() == [3.0]
        assert info["is_success"] is False and info["scene_seed"] == 3
        hold = obs["agent_pos"].copy()
        for _ in range(5):
            obs, _reward, terminated, truncated, info = env.step(hold)
        assert not terminated and not truncated
        assert (
            np.linalg.norm(obs["agent_pos"][:3] - hold[:3]) < 1e-3
        )  # the arm holds its pose
        frame = env.render()
        assert frame is not None and frame.shape == (360, 640, 3)
    finally:
        env.close()


@needs_renderer
def test_prompt_follows_the_scene_seed() -> None:
    """``move_block`` picks its direction per scene seed; the prompt of the episode must follow."""
    env = PraxisSimGymEnv("move_block")
    try:
        prompts = {}
        for seed in range(4):
            _, info = env.reset(seed=seed)
            assert env.task_description == env._env.task.prompt == info["prompt"]
            prompts[seed] = env.task_description
        assert set(prompts.values()) == {
            "Move the red block to the left.",
            "Move the red block to the right.",
        }
    finally:
        env.close()


@needs_renderer
def test_create_envs_wraps_and_lays_out_the_suite() -> None:
    seen: list[str] = []

    def wrap(env):
        seen.append(env.unwrapped.task)
        return env

    envs = create_praxis_sim_envs("lift_cube,block_in_bowl", 1, wrap=wrap)
    try:
        assert list(envs) == ["praxis_sim"] and sorted(envs["praxis_sim"]) == [0, 1]
        assert seen == ["lift_cube", "block_in_bowl"]
    finally:
        for vec in envs["praxis_sim"].values():
            vec.close()
