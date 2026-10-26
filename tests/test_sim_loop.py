"""Unit tests for the closed-loop sim loop (CPU only, no weights).

The tests need the ``mujoco`` package and the pinned Menagerie files. The
tests that render also need an offscreen OpenGL context. Without them, the
tests skip.

A stand-in pipeline drives the real single-arm session of
:mod:`praxis.single_arm` with a small encoder, the tile canvas and the flange
pose frame of the checkpoint. Thus the tests check that the loop gives the
session the frame stack that it expects. The context depth goes 0, 1, 2, 3
and the session never restarts.
"""

from __future__ import annotations

import http.client
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

# praxis.sim sets the OpenGL backend. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis import single_arm
from praxis.robot_io import NormStats
from praxis.sim import assets
from praxis.sim.config import SessionConfig, SimConfig, from_dict
from praxis.sim.env import EnvConfig, Observation, SimEnv
from praxis.sim.loop import (
    STACK_ROWS,
    EpisodeResult,
    FrameCheck,
    History,
    HoldPolicy,
    ModelPolicy,
    ScriptedPolicy,
    bad_views,
    check_model_paths,
    run,
    run_episode,
)
from praxis.sim.scene import SceneConfig
from praxis.sim.tasks import TaskConfig, TaskStatus
from praxis.sim.viewer import Viewer

if TYPE_CHECKING:
    from praxis.pipeline import PraxisPipeline

SMALL_SCENE = {
    "image_width": 128,
    "image_height": 72,
    "overview_width": 128,
    "overview_height": 72,
    "shadow_size": 0,
    "msaa_samples": 0,
}


def _assets_or_skip() -> None:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"The pinned Menagerie files are not available: {exc}")


def _render_env_or_skip(task: str = "lift_cube") -> SimEnv:
    _assets_or_skip()
    cfg = EnvConfig(
        scene=from_dict(SceneConfig, SMALL_SCENE), task=TaskConfig(name=task)
    )
    try:
        return SimEnv(cfg)
    except Exception as exc:  # no OpenGL backend on this machine
        pytest.skip(f"MuJoCo cannot make an offscreen OpenGL context: {exc}")


# ---------------------------------------------------------------------------
# A stand-in pipeline: the real single-arm session, a small encoder, no model
# ---------------------------------------------------------------------------


def _encode(clip: torch.Tensor) -> torch.Tensor:
    """Give one latent for the first raw frame and one for each 4 raw frames after it."""
    t = clip.shape[2]
    idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
    return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 32, 32))


def _stats() -> NormStats:
    lo = np.full(10, -0.2, dtype=np.float32)
    hi = np.full(10, 0.2, dtype=np.float32)
    lo[9], hi[9] = 0.0, 1.0
    return NormStats(
        action_q01=lo,
        action_q99=hi,
        proprio_mean=np.zeros(10, dtype=np.float32),
        proprio_std=np.ones(10, dtype=np.float32),
    )


class _StandInSession:
    """Real session state. The sampler gives rows that hold the pose, gripper open."""

    def __init__(
        self,
        prompt: str,
        seed: int,
        max_context_chunks: int,
        control_mode: str,
        exec_horizon: int | None,
    ) -> None:
        self.seed = seed
        self.stacks: list[int] = []
        self.state = single_arm.SingleArmServeSession(
            encode=_encode,
            stats=_stats(),
            device="cpu",
            prompt=prompt,
            max_context_chunks=max_context_chunks,
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            tile_size=(64, 64),
        )

    def step(self, obs: dict) -> SimpleNamespace:
        self.stacks.append(int(obs["image.exterior"].shape[0]))
        inputs = self.state.prepare(obs)
        rows = np.zeros((6, 4, 10), dtype=np.float32)
        rows[..., 3] = 1.0  # rot6d of the identity
        rows[..., 7] = 1.0
        rows[..., 9] = -1.0  # normalized grip -1 = open
        actions = torch.from_numpy(rows.reshape(1, 6, 40))
        commands = self.state.finish(inputs, actions)
        return SimpleNamespace(
            commands=commands, depth=inputs.depth, seed=self.seed + inputs.chunk_index
        )

    def decode_video(self, res: SimpleNamespace) -> torch.Tensor:
        return torch.full((len(res.commands), 3, 256, 832), 90, dtype=torch.uint8)


class _StandInPipe:
    embodiment = "single_arm"
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.sessions: list[_StandInSession] = []

    def session(
        self,
        prompt: str = "",
        seed: int = 0,
        max_context_chunks: int = 3,
        control_mode: str = "none",
        exec_horizon: int | None = None,
    ) -> _StandInSession:
        s = _StandInSession(
            prompt, seed, max_context_chunks, control_mode, exec_horizon
        )
        self.sessions.append(s)
        return s


def _metric(result: EpisodeResult, key: str) -> float:
    value = result.metrics[key]
    assert value is not None
    return float(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_history_keeps_the_last_rows() -> None:
    hist = History()
    for k in range(30):
        hist.push(
            Observation(
                ee_pose=np.full(7, k, dtype=np.float32),
                grip=k / 30,
                exterior=np.full((4, 8, 3), k, dtype=np.uint8),
                wrist=np.full((4, 8, 3), k, dtype=np.uint8),
                overview=None,
                tick=k,
                status=TaskStatus(),
            )
        )
    obs = hist.obs_dict()
    assert len(hist) == STACK_ROWS == 25
    assert obs["image.exterior"].shape == (25, 4, 8, 3)
    assert obs["robot_state.ee_pose"].shape == (25, 7)
    assert obs["grip"].shape == (25, 1)
    assert obs["image.wrist"][0, 0, 0, 0] == 5 and obs["image.wrist"][-1, 0, 0, 0] == 29
    assert single_arm.extract_pose8(obs)[0] == 29


def test_scripted_loop_without_rendering() -> None:
    _assets_or_skip()
    env = SimEnv(EnvConfig(task=TaskConfig(name="lift_cube"), render=False))
    try:
        policy = ScriptedPolicy()
        for seed in (0, 1):
            result = run_episode(env, policy, seed, after_success_s=0.5)
            assert result.success, result
            assert result.call_rows[0] == 21 and set(result.call_rows[1:]) == {24}
            assert result.success_tick is not None
            assert result.ticks == result.success_tick + 8
            assert result.tracking["track_pos_mm_p95"] < 2.0
            assert result.tracking["track_rot_deg_max"] < 1.0
            assert _metric(result, "max_lift_m") > 0.05
            assert result.to_json()["success_time_s"] == pytest.approx(
                result.success_tick / 15
            )
    finally:
        env.close()


def test_hold_policy_keeps_the_pose() -> None:
    _assets_or_skip()
    env = SimEnv(EnvConfig(render=False))
    try:
        result = run_episode(env, HoldPolicy(), seed=0, max_ticks=45)
        assert not result.success and result.ticks == 45 and result.calls == 2
        assert result.tracking["track_pos_mm_max"] < 1.0
        assert abs(_metric(result, "max_lift_m")) < 2e-3
    finally:
        env.close()


@pytest.mark.parametrize(
    ("max_context_chunks", "depths"),
    [
        (None, [0, 1, 1, 1]),  # the default: 1 context chunk
        (3, [0, 1, 2, 3]),  # the context grows to 3 chunks
    ],
)
def test_model_loop_feeds_the_session_stack(
    max_context_chunks: int | None, depths: list[int]
) -> None:
    env = _render_env_or_skip()
    pipe = _StandInPipe()
    session_cfg = (
        SessionConfig()
        if max_context_chunks is None
        else SessionConfig(max_context_chunks=max_context_chunks)
    )
    try:
        policy = ModelPolicy(
            cast("PraxisPipeline", pipe), "Pick up the red cube.", session_cfg
        )
        result = run_episode(env, policy, seed=2, max_ticks=21 + 3 * 24)
    finally:
        env.close()
    session = pipe.sessions[-1]
    assert result.call_rows == [21, 24, 24, 24]
    assert result.call_depths == depths  # no cold starts
    assert result.restarts == 0 and session.state.restart_count == 0
    assert session.stacks == [1, 22, 25, 25]
    assert session.seed == 2000  # session.seed + seed_stride * episode seed
    assert result.ticks == 93 and len(result.decode_s) == 4
    assert result.tracking["track_pos_mm_max"] < 1.0


def test_run_writes_outputs_and_serves_the_video(tmp_path: Path) -> None:
    env = _render_env_or_skip()
    env.close()
    pytest.importorskip("imageio_ffmpeg")
    cfg = from_dict(
        SimConfig,
        {
            "policy": {"type": "scripted"},
            "episodes": {"seeds": [3], "after_success_s": 0.2},
            "scene": SMALL_SCENE,
            "output": {"dir": str(tmp_path / "run")},
            "viewer": {"host": "127.0.0.1", "port": 0},
        },
    )
    viewer = Viewer(host="127.0.0.1", port=0)
    viewer.start()
    try:
        result = run(cfg, viewer=viewer, log=lambda _: None)
        conn = http.client.HTTPConnection("127.0.0.1", viewer.port, timeout=5)
        conn.request("GET", "/status")
        status = json.loads(conn.getresponse().read())
        conn.request("GET", "/video")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
    finally:
        viewer.stop()
    assert result.success_rate == 1.0
    out = tmp_path / "run"
    for name in ("config.yaml", "summary.json", "episode_003.json", "episode_003.npz"):
        assert (out / name).is_file(), name
    video = out / "episode_003.mp4"
    assert video.is_file() and video.stat().st_size > 1000
    assert status["video"] and status["status"]["phase"] == "episode done"
    assert resp.status == 200 and body[4:8] == b"ftyp"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["successes"] == 1 and summary["per_episode"][0]["seed"] == 3
    arrays = np.load(out / "episode_003.npz")
    assert arrays["ee_pose"].shape == (result.episodes[0].ticks, 7)
    assert arrays["cmd_pose"].shape == (result.episodes[0].ticks, 12)


def test_frame_check_warns_then_stops() -> None:
    good = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    black = np.zeros((8, 8, 3), dtype=np.uint8)

    def obs(exterior: np.ndarray, wrist: np.ndarray, tick: int) -> Observation:
        return Observation(
            ee_pose=np.zeros(7),
            grip=0.0,
            exterior=exterior,
            wrist=wrist,
            overview=None,
            tick=tick,
            status=TaskStatus(),
        )

    assert bad_views(obs(good, good, 0)) == []
    assert bad_views(obs(good, black, 0)) == ["wrist"]
    check = FrameCheck(max_bad_ticks=3)
    check(obs(black, good, 1))
    check(obs(good, good, 2))
    check(obs(black, good, 3))
    check(obs(black, good, 4))
    assert check.bad_ticks == 3
    with pytest.raises(RuntimeError, match="exterior view was black"):
        check(obs(black, good, 5))


def test_run_checks_the_model_paths_first(tmp_path: Path) -> None:
    cfg = from_dict(
        SimConfig,
        {
            "policy": {"type": "model"},
            "model": {"checkpoint": str(tmp_path / "missing")},
            "output": {"dir": str(tmp_path / "out")},
        },
    )
    with pytest.raises(FileNotFoundError, match=r"model\.checkpoint"):
        check_model_paths(cfg)
    assert not (tmp_path / "out").exists()
