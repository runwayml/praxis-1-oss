"""Unit tests for the sim tasks and the scripted pick policy (CPU only).

The tests need the ``mujoco`` package and the pinned Menagerie files. Without
the files and without a network, the tests skip.
"""

from __future__ import annotations

import numpy as np
import pytest

# praxis.sim sets the OpenGL backend. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis import single_arm
from praxis.sim import assets
from praxis.sim.env import EnvConfig, SimEnv
from praxis.sim.robot import pos_rot_to_pose12, pose7_to_pos_rot
from praxis.sim.scripted import ScriptedPick, run_episode
from praxis.sim.tasks import TASKS, TaskConfig, make_task

LIFT_SEEDS = range(5)


def _env(task: str) -> SimEnv:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"The pinned Menagerie files are not available: {exc}")
    return SimEnv(EnvConfig(task=TaskConfig(name=task), render=False))


@pytest.fixture(scope="module")
def lift_env() -> SimEnv:
    sim = _env("lift_cube")
    yield sim
    sim.close()


def test_registry() -> None:
    assert {"lift_cube", "cube_on_plate"} <= set(TASKS)
    task = make_task(TaskConfig())
    assert task.prompt == "Pick up the red cube."
    assert task.max_ticks == 165 and task.hold_ticks == 4
    custom = make_task(TaskConfig(prompt="Lift the cube.", time_limit_s=4.0))
    assert custom.prompt == "Lift the cube." and custom.max_ticks == 60
    with pytest.raises(ValueError):
        make_task(TaskConfig(name="bogus"))


def test_placement_is_seeded(lift_env: SimEnv) -> None:
    cfg = lift_env.cfg.task
    poses = []
    for seed in (0, 0, 1, 2):
        lift_env.reset(seed)
        poses.append(lift_env.task.cube_pos(lift_env.data))
    np.testing.assert_array_equal(poses[0], poses[1])
    assert np.linalg.norm(poses[0] - poses[2]) > 1e-3
    for p in poses:
        assert cfg.cube_x[0] <= p[0] <= cfg.cube_x[1]
        assert cfg.cube_y[0] <= p[1] <= cfg.cube_y[1]
        assert p[2] == pytest.approx(cfg.cube_size / 2, abs=2e-3)  # at rest


def test_start_is_not_a_success(lift_env: SimEnv) -> None:
    obs = lift_env.reset(seed=3)
    hold = pos_rot_to_pose12(*pose7_to_pos_rot(obs.ee_pose))
    for _ in range(30):
        lift_env.step(hold, 0.0)
    summary = lift_env.summary()
    assert not summary["success"] and summary["ticks"] == 30
    assert abs(float(summary["max_lift_m"])) < 2e-3
    assert summary["ik_misses"] == 0


def test_a_dropped_cube_does_not_pass(lift_env: SimEnv) -> None:
    obs = lift_env.reset(seed=4)
    hold = pos_rot_to_pose12(*pose7_to_pos_rot(obs.ee_pose))
    adr = lift_env.model.joint("red_cube_joint").qposadr[0]
    lift_env.data.qpos[adr + 2] += 0.08  # 8 cm above the rest height, then free fall
    for _ in range(30):
        lift_env.step(hold, 0.0)
    status = lift_env.task.status
    assert status.metrics["max_lift_m"] > 0.05
    assert not status.success


def test_scripted_commands_have_the_model_format(lift_env: SimEnv) -> None:
    obs = lift_env.reset(seed=0)
    policy = ScriptedPick()
    policy.reset(lift_env)
    for rows in (21, 24):
        commands = policy.step(obs)
        assert len(commands) == rows
        for t, row in enumerate(commands):
            cmd = row["robot_command"]
            assert cmd["type"] == "cartesian_pos"
            assert cmd["mode"] == single_arm.DROID_IMPEDANCE
            assert cmd["pose"].dtype == np.float32 and cmd["pose"].shape == (12,)
            assert row["timestamp"] == pytest.approx(t / 15)
            assert 0.0 <= row["target_grip"] <= 1.0
        poses, grips = single_arm.wire_to_arrays(commands)
        for pose12, grip in zip(poses, grips, strict=True):
            obs = lift_env.step(pose12, float(grip))


def test_scripted_pick_lifts_the_cube(lift_env: SimEnv) -> None:
    policy = ScriptedPick()
    for seed in LIFT_SEEDS:
        summary = run_episode(lift_env, policy, seed)
        assert summary["success"], (seed, summary)
        assert summary["ik_misses"] == 0
        assert float(summary["max_lift_m"]) > 0.1
        assert float(summary["grasp_ticks"]) > 30
        assert 0.3 < float(summary["max_grip"]) < 0.6  # closed on a 5 cm cube


def test_scripted_cube_on_plate() -> None:
    sim = _env("cube_on_plate")
    try:
        policy = ScriptedPick()
        for seed in range(2):
            summary = run_episode(sim, policy, seed)
            assert summary["success"], (seed, summary)
    finally:
        sim.close()
