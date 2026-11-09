"""Tests of the bimanual YAM sim (CPU, no model weights).

The tests need the ``mujoco`` package and the pinned Menagerie files (cache or
download). Without them, the tests skip. Only the rendering tests need an
offscreen OpenGL context (``MUJOCO_GL=egl``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis import bimanual
from praxis.robot_io import NormStats
from praxis.sim import assets
from praxis.sim.yam_config import load_yam_config, render_gpu
from praxis.sim.yam_env import PLACE_TRIES, YamEnv, YamEnvConfig
from praxis.sim.yam_loop import (
    YamHistory,
    YamHoldPolicy,
    YamModelPolicy,
    YamSessionConfig,
    run_yam_episode,
)
from praxis.sim.yam_scene import TABLE_TOP_Z, YamSceneConfig, load_rig_profile
from praxis.sim.yam_scripted import run_oracle
from praxis.sim.yam_tasks import BAG_CLEARANCE, YAM_TASKS, YamTaskConfig

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
TASKS = sorted(YAM_TASKS)
SMALL = YamSceneConfig(image_width=96, image_height=60, shadow_size=0, msaa_samples=0)


def _assets_or_skip() -> None:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"robot model files not available: {exc}")


def _env(
    name: str, render: bool = False, scene: YamSceneConfig | None = None
) -> YamEnv:
    _assets_or_skip()
    cfg = YamEnvConfig(scene=scene or SMALL, task=YamTaskConfig(name=name))
    cfg.render = render
    if render:
        import mujoco

        probe = mujoco.MjModel.from_xml_string("<mujoco/>")
        try:
            mujoco.Renderer(probe, height=16, width=16).close()
        except Exception as exc:  # no OpenGL backend on this machine
            pytest.skip(f"MuJoCo cannot make an offscreen OpenGL context: {exc}")
    return YamEnv(cfg)


@pytest.mark.parametrize("name", TASKS)
def test_oracle_solves_the_task(name: str) -> None:
    env = _env(name)
    try:
        summary = run_oracle(env, seed=0)
        assert summary["success"], summary
        start = env.reset(0)
        assert not start.status.success
    finally:
        env.close()


def test_seed_sets_the_start_state() -> None:
    env = _env("lego_sort")
    try:
        a = env.reset(3).ee_pose["left"].copy()
        qa = env.data.qpos.copy()
        b = env.reset(3).ee_pose["left"].copy()
        qb = env.data.qpos.copy()
        env.reset(4)
        assert np.array_equal(a, b) and np.array_equal(qa, qb)
        assert not np.array_equal(qa, env.data.qpos)
    finally:
        env.close()


@pytest.mark.parametrize("name", TASKS)
def test_start_state_has_no_overlap(name: str) -> None:
    """Each reset finds a start state without overlaps well within the tries."""
    env = _env(name)
    checks: list[bool] = []
    check = env.start_overlap

    def counted() -> bool:
        checks.append(check())
        return checks[-1]

    env.start_overlap = counted
    try:
        for seed in range(5):
            checks.clear()
            env.reset(seed)
            assert not checks[-1] and len(checks) < PLACE_TRIES // 5, (seed, checks)
    finally:
        env.close()


@pytest.mark.parametrize(
    "name", ["apple_in_bag", "tennis_ball_in_bag", "dice_in_bag", "bell_in_bag"]
)
def test_bag_items_start_clear_of_the_bag(name: str) -> None:
    import mujoco

    env = _env(name)
    task = env.task
    try:
        for seed in range(40):
            task.place(np.random.default_rng(seed), env.data)
            mujoco.mj_forward(env.model, env.data)
            bag = task.container_xy()
            for item in task.item_names:
                gap = task.bag_gap(task.pos(env.data, item)[:2], bag)
                assert gap >= BAG_CLEARANCE, (seed, item, gap)
    finally:
        env.close()


@pytest.mark.parametrize("seed", [2, 35])
def test_oracle_drops_the_bell_to_rest(seed: int) -> None:
    """The dropped bell settles on its base in the bag (it rocked on its dome)."""
    env = _env("bell_in_bag")
    try:
        assert run_oracle(env, seed=seed)["success"]
    finally:
        env.close()


def test_hold_policy_keeps_the_arms_still() -> None:
    env = _env("screwdriver_in_bin")
    try:
        start = env.reset(0)
        res = run_yam_episode(env, YamHoldPolicy(), 0, max_ticks=45)
        assert res.ticks == 45 and not res.success
        for side in ("left", "right"):
            drift = np.linalg.norm(
                env.observe().ee_pose[side][:3] - start.ee_pose[side][:3]
            )
            assert drift < 0.01
    finally:
        env.close()


def test_task_prompts() -> None:
    task = YAM_TASKS["screwdriver_in_bin"](YamTaskConfig())
    assert task.prompt == (
        "Pick up the screwdriver from the center of the table, "
        "then place the screwdriver in the bin"
    )
    for task_cls in YAM_TASKS.values():
        # The training captions have no closing period.
        assert not task_cls(YamTaskConfig()).prompt.endswith(".")
    assert YAM_TASKS["block_tower"](YamTaskConfig()).prompt == (
        "build the wood block tower"
    )
    assert YAM_TASKS["block_tower"](YamTaskConfig(prompt="x")).prompt == "x"


def test_rig_profile_sets_the_table_and_the_cameras(tmp_path: Path) -> None:
    """A profile moves the table and the cameras, and the tasks follow the table."""
    top = {
        "pos": [0.25, 0.0, 1.3],
        "forward": [0.5, 0.0, -0.866],
        "right": [0.0, -1.0, 0.0],
        "intrinsics": {
            "width": 1920,
            "height": 1200,
            "fx": 760.0,
            "fy": 760.0,
            "cx": 960.0,
            "cy": 600.0,
        },
    }
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"table_top_z": 0.29, "top": top, "wall_h": 1.5}))
    scene = YamSceneConfig(
        image_width=96, image_height=60, shadow_size=0, rig_profile=str(path)
    )
    assert scene.table_top_z == 0.29 and scene.top.pos == (0.25, 0.0, 1.3)
    assert scene.wrist_camera("left") == scene.wrist
    env = _env("tennis_ball_in_bag", scene=scene)
    try:
        assert env.task.z0 == 0.29
        table = env.model.geom("table")
        assert table.pos[2] + table.size[2] == pytest.approx(0.29)
        assert run_oracle(env, seed=0)["success"]
    finally:
        env.close()
    default = _env("tennis_ball_in_bag")
    try:
        assert default.task.z0 == TABLE_TOP_Z
    finally:
        default.close()


def test_rig_profile_needs_all_the_textures(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"textures": {"table": "table.png"}}))
    with pytest.raises(ValueError, match="textures"):
        load_rig_profile(path)


def _camera() -> dict:
    return {
        "pos": [0.25, 0.0, 1.3],
        "forward": [0.5, 0.0, -0.866],
        "right": [0.0, -1.0, 0.0],
        "intrinsics": {"width": 1920, "height": 1200, "fx": 760.0, "fy": 760.0},
    }


@pytest.mark.parametrize(
    ("profile", "match"),
    [
        ({"table_top": 0.25}, "unknown keys"),
        ({"lighting": {"lightning": 0.5}}, "lighting"),
        ({"wrist": {"middle": _camera()}}, "wrist"),
        ({"top": {k: v for k, v in _camera().items() if k != "intrinsics"}}, "top"),
        ({"wrist": {"left": {"pos": [0, 0, 1]}}}, "wrist.left"),
        (
            {
                "textures": {
                    k: f"{k}.png"
                    for k in ("table", "wall_back", "wall_left", "wall_right")
                }
            },
            "do not exist",
        ),
        ({"name": {"fit": "notes"}}, "name"),
        ({"name": "cell", "provenance": {"tool": "fit"}}, "unknown keys"),
    ],
)
def test_rig_profile_rejects_bad_keys(
    tmp_path: Path, profile: dict, match: str
) -> None:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match=match):
        load_rig_profile(path)


def test_render_gpu_avoids_the_model_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert render_gpu(None, "cuda") == 5
    assert render_gpu(None, "cuda:1") == 3
    assert render_gpu(None, None) is None  # no model: the default
    assert render_gpu(2, "cuda") == 2  # a set device stays
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert render_gpu(None, "cuda") is None  # one GPU: the default
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert render_gpu(None, "cuda") is None  # the user's choice stays


def test_loop_stops_on_black_views(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run of black model views stops the episode with an error."""
    env = _env("cup_by_coaster")
    observe = env.observe

    def black():
        obs = observe()
        obs.views = {v: np.zeros((8, 8, 3), np.uint8) for v in ("top", "wrist_left")}
        return obs

    monkeypatch.setattr(env, "observe", black)
    try:
        with pytest.raises(RuntimeError, match="black or one flat color"):
            run_yam_episode(env, YamHoldPolicy(), 0, max_ticks=45)
    finally:
        env.close()


def test_rig_profile_takes_a_name(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"name": "cell 3", "table_top_z": 0.29}))
    assert load_rig_profile(path)["table_top_z"] == 0.29
    assert YamSceneConfig(rig_profile=str(path)).table_top_z == 0.29


def test_model_policy_refuses_a_single_arm_checkpoint(tmp_path: Path) -> None:
    """The embodiment of config.json is checked before any weight loads."""
    from tiny_model import tiny_config

    from praxis.sim.yam_config import make_yam_policy

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(tiny_config()))
    (tmp_path / "vae.safetensors").touch()
    (tmp_path / "text").mkdir()
    cfg = load_yam_config(
        ROOT / "configs" / "sim_yam.yaml",
        [
            f"model.checkpoint={ckpt}",
            f"model.vae_weights={tmp_path / 'vae.safetensors'}",
            f"model.text_encoder={tmp_path / 'text'}",
        ],
    )
    with pytest.raises(SystemExit, match="'single_arm' checkpoint"):
        make_yam_policy("model", cfg)


def test_sim_yam_config_loads() -> None:
    cfg = load_yam_config(ROOT / "configs" / "sim_yam.yaml")
    assert cfg.session.max_context_chunks == 1 and cfg.session.exec_horizon == 24
    assert cfg.sampler.t_shift == 1.0 and cfg.sampler.cfg_video_scale == 1.0
    assert (cfg.scene.image_width, cfg.scene.image_height) == (416, 260)
    assert cfg.scene.rig_profile is None


# ---------------------------------------------------------------------------
# The model path with a stand-in sampler
# ---------------------------------------------------------------------------


def _encode(clip: torch.Tensor) -> torch.Tensor:
    t = clip.shape[2]
    return torch.zeros(
        1, 48, 1 + (t - 1) // 4, clip.shape[3] // 16, clip.shape[4] // 16
    )


class _StandInSession:
    """The real bimanual session state. The sampler gives rows that hold the pose."""

    def __init__(self, prompt: str, max_context_chunks: int, exec_horizon: int) -> None:
        cfg = json.loads((DATA / "praxis-1-yam.json").read_text())
        n = cfg["normalization"]
        self.state = bimanual.BimanualServeSession(
            encode=_encode,
            stats=NormStats(
                action_q01=np.asarray(n["action_q01"], np.float32),
                action_q99=np.asarray(n["action_q99"], np.float32),
                proprio_mean=np.asarray(n["proprio_mean"], np.float32),
                proprio_std=np.asarray(n["proprio_std"], np.float32),
            ),
            device="cpu",
            prompt=prompt,
            max_context_chunks=max_context_chunks,
            exec_horizon=exec_horizon,
            dims=bimanual.q99_dims(cfg),
        )
        self.stacks: list[int] = []

    def step(self, obs: dict) -> SimpleNamespace:
        self.stacks.append(int(obs["image.exterior"].shape[0]))
        inputs = self.state.prepare(obs)
        commands = self.state.finish(inputs, torch.zeros(1, 6, 80))
        return SimpleNamespace(
            commands=commands, depth=inputs.depth, seed=0, prompt=inputs.prompt
        )


class _StandInPipe:
    embodiment = "bimanual"
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.sessions: list[_StandInSession] = []
        self.prompts: list[str] = []

    def session(
        self, prompt: str, seed: int, max_context_chunks: int, exec_horizon: int
    ):
        self.prompts.append(prompt)
        s = _StandInSession(prompt, max_context_chunks, exec_horizon)
        self.sessions.append(s)
        return s


def test_model_policy_drives_the_rig_through_the_session() -> None:
    """The sim views, poses and grips go through the bimanual session of the server."""
    env = _env("screwdriver_in_bin", render=True, scene=YamSceneConfig(shadow_size=0))
    pipe = _StandInPipe()
    try:
        policy = YamModelPolicy(
            pipe,  # ty: ignore[invalid-argument-type]
            env.prompt,
            YamSessionConfig(),
        )
        res = run_yam_episode(env, policy, 0, max_ticks=21 + 24 + 24)
        assert res.calls == 3 and res.ticks == 69
        assert pipe.prompts == [env.prompt]
        # The start frame, then the start + 21 rows, then the last 25 frames.
        assert pipe.sessions[0].stacks == [1, 22, 25]
    finally:
        env.close()


def test_model_policy_refuses_the_single_arm_checkpoint() -> None:
    pipe = SimpleNamespace(embodiment="single_arm")
    with pytest.raises(ValueError, match="bimanual"):
        YamModelPolicy(pipe, "")  # ty: ignore[invalid-argument-type]


def test_history_gives_the_bimanual_observation_format() -> None:
    env = _env("cup_by_coaster", render=True)
    try:
        hist = YamHistory()
        hist.push(env.reset(0))
        obs = hist.obs_dict()
        views = bimanual.view_stacks(obs)
        assert all(v.shape == (1, 60, 96, 3) for v in views.values())
        assert bimanual.extract_pose16(obs).shape == (16,)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_sim_yam", ROOT / "scripts" / "run_sim_yam.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_with_the_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assets_or_skip()
    out = tmp_path / "out"
    argv = [
        "run_sim_yam.py",
        "--config",
        str(ROOT / "configs" / "sim_yam.yaml"),
        "--policy",
        "oracle",
        "--out",
        str(out),
        "--units",
        "block_tower:0,dice_in_bag:1",
        "--no-video",
    ]
    monkeypatch.setattr("sys.argv", argv)
    _runner().main()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["all"]["episodes"] == 2 and summary["all"]["successes"] == 2
    assert (out / "dice_in_bag" / "episode_001.json").exists()
    # The same units with other settings do not reuse the episodes.
    for change in (
        ["--prompt", "build the tower"],
        ["--set", "episodes.time_scale=0.1"],
        ["--set", "scene.image_width=64"],
    ):
        monkeypatch.setattr("sys.argv", [*argv, *change])
        with pytest.raises(SystemExit, match="other settings"):
            _runner().main()


def test_runner_refuses_a_malformed_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = str(ROOT / "configs" / "sim_yam.yaml")
    argv = ["run_sim_yam.py", "--config", config, "--policy", "oracle"]
    argv += ["--out", str(tmp_path)]
    monkeypatch.setattr("sys.argv", [*argv, "--units", "block_tower"])
    with pytest.raises(SystemExit, match="task:seed"):
        _runner().main()
