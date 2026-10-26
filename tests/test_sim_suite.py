"""Tests of the DROID task suite (CPU, no rendering, no model)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from praxis.sim.env import EnvConfig, SimEnv
from praxis.sim.loop import ScriptedPolicy, run_episode
from praxis.sim.suite import MoveBlock
from praxis.sim.suite_hard import ColorBlockInBowl, StackColorBlocks, _ColourBlocks
from praxis.sim.suite_insert import PegInHole
from praxis.sim.suite_multi import StackSmallBlock, StackThreeBlocks, TwoBlocksInBowl
from praxis.sim.suite_place import CubeInTray, CubeOnColor
from praxis.sim.tasks import TaskConfig, make_task

SUITE = [
    "lift_cube",
    "cube_on_plate",
    "block_in_bowl",
    "block_out_of_bowl",
    "move_block",
    "stack_block",
    "press_button",
]


@pytest.mark.parametrize("name", SUITE)
def test_oracle_solves_the_task(name: str) -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        for seed in (0, 1, 2):
            r = run_episode(env, ScriptedPolicy(), seed, after_success_s=0.3)
            assert r.success, (name, seed, r.metrics)
            assert r.metrics["progress"] == pytest.approx(1.0)
    finally:
        env.close()


@pytest.mark.parametrize("name", SUITE)
def test_start_state_is_not_a_success(name: str) -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        for seed in (0, 1, 2, 3):
            env.reset(seed)
            assert not env.task.status.success
            assert env.task.progress(env.data) < 0.5, (name, seed)
    finally:
        env.close()


def test_move_block_prompt_follows_the_seed_direction() -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name="move_block"), render=False))
    try:
        sides = set()
        for seed in range(8):
            env.reset(seed)
            sides.add(env.prompt)
            assert isinstance(env.task, MoveBlock)
            want = "left" if env.task.direction > 0 else "right"
            assert want in env.prompt
        assert len(sides) == 2
    finally:
        env.close()


def test_unknown_task_names_the_suite() -> None:
    with pytest.raises(ValueError, match="block_in_bowl"):
        make_task(TaskConfig(name="nope"))


# -- The harder tasks (placement with fixtures, insertion, harder versions) --


def _hard_tasks() -> list[str]:
    from praxis.sim.suite_hard import HARD_TASKS
    from praxis.sim.suite_insert import INSERT_TASKS
    from praxis.sim.suite_multi import MULTI_TASKS
    from praxis.sim.suite_place import PLACE_TASKS

    return sorted({**PLACE_TASKS, **INSERT_TASKS, **HARD_TASKS, **MULTI_TASKS})


HARD = _hard_tasks()


@pytest.mark.parametrize("name", HARD)
def test_oracle_solves_the_hard_task(name: str) -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        wins = 0
        for seed in range(10):
            r = run_episode(env, ScriptedPolicy(), seed, after_success_s=0.3)
            wins += bool(r.success)
            if r.success:
                assert r.metrics["progress"] == pytest.approx(1.0)
        assert wins >= 9, (name, wins)
    finally:
        env.close()


@pytest.mark.parametrize("name", HARD)
def test_hard_start_state_is_not_a_success(name: str) -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        for seed in range(8):
            env.reset(seed)
            assert not env.task.status.success
            p = env.task.progress(env.data)
            assert 0.0 <= p < 0.5, (name, seed, p)
            assert env.prompt[0].isupper() and env.prompt.endswith("."), env.prompt
    finally:
        env.close()


# -- place tasks --


def _colour_task(env: SimEnv) -> CubeOnColor:
    assert isinstance(env.task, CubeOnColor)
    return env.task


def test_cube_on_color_prompt_follows_the_seed_colour() -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name="cube_on_color"), render=False))
    try:
        colours = set()
        for seed in range(12):
            env.reset(seed)
            colour = _colour_task(env).colour
            colours.add(colour)
            assert env.prompt == f"Put the red cube on the {colour} square."
        assert colours == {"blue", "yellow", "purple"}
    finally:
        env.close()


def _rest_cube_on(env: SimEnv, xy: np.ndarray, z: float, ticks: int) -> None:
    """Put the cube at rest at ``xy``, height ``z``, and measure ``ticks`` ticks."""
    import mujoco

    task = env.task
    adr = task._ids["cube_qpos"]
    env.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
    env.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    for _ in range(ticks):
        task.update(env.data, env.arm.pinch(), env.arm.grip(), env.table_xy)


def test_cube_on_a_wrong_square_fails_without_progress() -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name="cube_on_color"), render=False))
    try:
        env.reset(3)
        task = _colour_task(env)
        wrong = (task.target + 1) % 3
        _rest_cube_on(env, task.square_xy(wrong), task.rest_height(), 10)
        assert task.status.failed and not task.status.success
        assert task.status.metrics["progress"] == 0.0
        # Moving it to the target afterwards does not count.
        _rest_cube_on(env, task.square_xy(task.target), task.rest_height(), 10)
        assert not task.status.success
        env.reset(3)
        _rest_cube_on(env, task.square_xy(task.target), task.rest_height(), 10)
        assert task.status.success and not task.status.failed
    finally:
        env.close()


def test_cube_in_tray_needs_the_floor_near_the_centre() -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name="cube_in_tray"), render=False))
    try:
        env.reset(2)
        task = env.task
        assert isinstance(task, CubeInTray)
        z = task.rest_height()
        _rest_cube_on(env, task.tray_xy, z + 0.01, 10)
        assert not task.status.success  # not resting on the floor
        assert task.status.metrics["progress"] < 1.0
        _rest_cube_on(env, task.tray_xy, z, 10)
        assert task.status.success
    finally:
        env.close()


# -- insert tasks --


def _peg_env(name: str) -> SimEnv:
    return SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))


def _peg_task(env: SimEnv) -> PegInHole:
    assert isinstance(env.task, PegInHole)
    return env.task


@pytest.mark.parametrize(
    "name", ["peg_in_hole_6mm", "peg_in_hole_4mm", "peg_in_hole_2mm"]
)
def test_peg_in_the_hole_is_a_success_and_on_the_board_is_not(name: str) -> None:
    import mujoco
    import numpy as np

    env = _peg_env(name)
    try:
        env.reset(0)
        task = _peg_task(env)
        h = task.peg_half[2]
        hole = task.hole_xy
        # Upright in the hole, on the table, released: a success.
        task._set_free_pose(
            env.data, "cube", np.array([hole[0], hole[1], h + 1e-4]), task.board_yaw
        )
        mujoco.mj_forward(env.model, env.data)
        assert task.condition(env.data, 0.0)
        assert task.progress(env.data) == pytest.approx(0.95)
        # On top of the board, next to the hole: not a success, low progress.
        off = np.array([hole[0] + 0.035, hole[1], task.board_height + h + 1e-4])
        task._set_free_pose(env.data, "cube", off, task.board_yaw)
        mujoco.mj_forward(env.model, env.data)
        assert not task.condition(env.data, 0.0)
        assert task.progress(env.data) < 0.7
    finally:
        env.close()


def test_peg_hole_side_follows_the_clearance() -> None:
    sides = {}
    for name in ("peg_in_hole_6mm", "peg_in_hole_4mm", "peg_in_hole_2mm"):
        env = _peg_env(name)
        try:
            task = _peg_task(env)
            sides[name] = 2 * task.hole_half
            # The peg starts clear of the board.
            for seed in range(8):
                env.reset(seed)
                assert task.edge_gap(task.cube_pos(env.data)) >= 0.059
        finally:
            env.close()
    assert sides["peg_in_hole_6mm"] == pytest.approx(0.042)
    assert sides["peg_in_hole_4mm"] == pytest.approx(0.038)
    assert sides["peg_in_hole_2mm"] == pytest.approx(0.034)


def test_a_fallen_peg_gets_the_carry_progress_only() -> None:
    import math

    import mujoco

    env = _peg_env("peg_in_hole_4mm")
    try:
        env.reset(1)
        task = _peg_task(env)
        hole = task.hole_xy
        # The peg lies on its side on the board, over the hole.
        a = math.pi / 2
        adr = task._ids["cube_qpos"]
        env.data.qpos[adr : adr + 3] = [hole[0], hole[1], task.board_height + 0.016]
        env.data.qpos[adr + 3 : adr + 7] = [math.cos(a / 2), math.sin(a / 2), 0, 0]
        mujoco.mj_forward(env.model, env.data)
        assert not task.condition(env.data, 0.0)
        assert task.progress(env.data) <= 0.35 + 1e-9
    finally:
        env.close()


# -- hard tasks --

COLOUR_TASKS = ["pick_color_block", "color_block_in_bowl", "stack_color_blocks"]


def _block_task(env: SimEnv) -> _ColourBlocks:
    task = env.task
    assert isinstance(task, _ColourBlocks)
    return task


@pytest.mark.parametrize("name", COLOUR_TASKS)
def test_colour_prompt_follows_the_seed_target(name: str) -> None:
    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        seen = set()
        for seed in range(12):
            env.reset(seed)
            task = _block_task(env)
            seen.add(task.target)
            assert f"the {task.target} block" in env.prompt
            assert task.cube_pos(env.data) == pytest.approx(
                task.free_pos(env.data, task.target)
            )
            if isinstance(task, StackColorBlocks):
                assert task.base != task.target
                assert env.prompt.endswith(f"on the {task.base} block.")
        assert seen == set(_block_task(env).colours), seen
    finally:
        env.close()


@pytest.mark.parametrize("name", COLOUR_TASKS)
def test_colour_blocks_start_apart_and_in_reach(name: str) -> None:
    import numpy as np

    env = SimEnv(EnvConfig(task=TaskConfig(name=name), render=False))
    try:
        for seed in range(8):
            env.reset(seed)
            task = _block_task(env)
            xy = [task.free_pos(env.data, c)[:2] for c in task.colours]
            for i, a in enumerate(xy):
                assert 0.42 <= a[0] <= 0.66 and abs(a[1]) <= 0.21, (seed, a)
                for b in xy[i + 1 :]:
                    assert np.linalg.norm(a - b) >= task.spacing - 0.01, seed
                if isinstance(task, ColorBlockInBowl):
                    assert np.linalg.norm(a - task.bowl_xy) >= task.bowl_clear - 0.01
    finally:
        env.close()


def test_lifting_a_distractor_fails_pick_color_block() -> None:
    import mujoco
    import numpy as np

    env = SimEnv(EnvConfig(task=TaskConfig(name="pick_color_block"), render=False))
    try:
        env.reset(0)
        task = _block_task(env)
        other = task.others()[0]
        adr = task._ids[f"{other}_qpos"]
        env.data.qpos[adr + 2] += 0.1
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        far = np.array([0.0, 0.0, 1.0])
        task.update(env.data, far, 0.0, env.table_xy)
        assert task.status.failed and task.wrong
        assert task.status.metrics["progress"] == 0.0
        assert not task.condition(env.data, 0.0)
    finally:
        env.close()


# -- multi tasks --


def _settle(env: SimEnv, seconds: float = 0.5) -> None:
    import mujoco

    for _ in range(round(seconds / env.model.opt.timestep)):
        mujoco.mj_step(env.model, env.data)


def test_two_blocks_in_bowl_progress_counts_each_block() -> None:
    import numpy as np

    env = SimEnv(EnvConfig(task=TaskConfig(name="two_blocks_in_bowl"), render=False))
    try:
        env.reset(0)
        task = env.task
        assert isinstance(task, TwoBlocksInBowl)
        bowl = task.bowl_xy
        z = 0.006 + 0.025 + 1e-3
        task._set_free_pose(
            env.data, "cube", np.array([bowl[0] - 0.035, bowl[1], z]), 0.0
        )
        _settle(env)
        assert task.placed(env.data, "cube") and not task.condition(env.data, 0.0)
        assert 0.5 <= task.progress(env.data) < 0.95
        task._set_free_pose(
            env.data, "green", np.array([bowl[0] + 0.035, bowl[1], z]), 0.0
        )
        _settle(env)
        assert task.condition(env.data, 0.0)
    finally:
        env.close()


def test_smaller_targets_than_the_suite() -> None:
    from praxis.sim.suite import BlockInBowl, StackBlock
    from praxis.sim.suite_multi import (
        BlockInSmallBowl,
        CubeOnSmallPlate,
        StackSmallBlock,
    )
    from praxis.sim.tasks import CubeOnPlate

    assert BlockInSmallBowl.radius < BlockInBowl.radius
    assert CubeOnSmallPlate.plate_half[0] < CubeOnPlate.plate_half[0]
    assert StackSmallBlock.target_half < StackBlock.target_half


def test_stack_small_block_gives_no_progress_for_the_wrong_order() -> None:
    import numpy as np

    env = SimEnv(EnvConfig(task=TaskConfig(name="stack_small_block"), render=False))
    try:
        env.reset(0)
        task = env.task
        assert isinstance(task, StackSmallBlock)
        red = task.cube_pos(env.data)
        top = np.array([red[0], red[1], red[2] + 0.025 + 0.02 + 1e-3])
        task._set_free_pose(env.data, "target", top, 0.0)
        _settle(env)
        assert task.progress(env.data) == 0.0
    finally:
        env.close()


def test_stack_three_blocks_counts_the_tower() -> None:
    import numpy as np

    env = SimEnv(EnvConfig(task=TaskConfig(name="stack_three_blocks"), render=False))
    try:
        env.reset(0)
        task = env.task
        assert isinstance(task, StackThreeBlocks)
        base = task.free_pos(env.data, "blue")
        task._set_free_pose(
            env.data,
            "cube",
            base + np.array([0.0, 0.0, 0.051]),
            task.free_yaw(env.data, "blue"),
        )
        _settle(env)
        assert task.tower(env.data) == ["blue", "cube"]
        assert 0.45 <= task.progress(env.data) < 0.95
        assert not task.condition(env.data, 0.0)
        task._set_free_pose(
            env.data,
            "green",
            base + np.array([0.0, 0.0, 0.102]),
            task.free_yaw(env.data, "blue"),
        )
        _settle(env)
        assert task.condition(env.data, 0.0)
    finally:
        env.close()
