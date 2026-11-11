"""Tests of the client side of the YAM sim: the trajectory player, its smoothing and
real-time chunking on the sim clock (CPU, no model weights)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("praxis_sim")  # the `sim` extra
pytest.importorskip("scipy")  # praxis.serve.rtc

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis.model import SamplerSettings
from praxis.robot_io import NormStats
from praxis.sim import assets
from praxis.sim.yam_env import YamEnv, YamEnvConfig
from praxis.sim.yam_loop import YamModelPolicy, YamSessionConfig, run_yam_episode
from praxis.sim.yam_player import (
    NS,
    PoseSmoother,
    RealTimeChunking,
    TrajectoryPlayer,
    YamPlayerConfig,
    sg_weights,
)
from praxis.sim.yam_scene import YamSceneConfig
from praxis.sim.yam_tasks import YamTaskConfig

ROOT = Path(__file__).resolve().parents[1]
YAM = json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
FPS = 15.0
ROW_NS = NS / FPS


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(xyz: np.ndarray | list[float], deg: float = 0.0) -> np.ndarray:
    return np.concatenate([np.asarray(xyz, np.float64), _rot_z(deg).reshape(9)])


def _row(
    xyz_left: np.ndarray, xyz_right: np.ndarray | None = None, grip: float = 0.0
) -> dict:
    right = xyz_left + np.array([0.0, -0.6, 0.0]) if xyz_right is None else xyz_right
    return {
        "robot_command.left": {
            "type": "cartesian_pos",
            "pose": _pose(xyz_left).astype(np.float32),
        },
        "robot_command.right": {
            "type": "cartesian_pos",
            "pose": _pose(right).astype(np.float32),
        },
        "target_grip.left": grip,
        "target_grip.right": grip,
    }


def _line(n: int, x0: float = 0.4, step: float = 0.01) -> list[dict]:
    return [_row(np.array([x0 + step * i, 0.3, 0.3])) for i in range(n)]


# ---------------------------------------------------------------------------
# The smoothing
# ---------------------------------------------------------------------------


def test_sg_weights_are_the_textbook_ones() -> None:
    # A wrong fit would bias every executed pose.
    np.testing.assert_allclose(
        sg_weights(3, 3, 2) * 21, [-2, 3, 6, 7, 6, 3, -2], atol=1e-9
    )
    assert sg_weights(3, 1, 2).sum() == pytest.approx(1.0)


def test_smoother_passes_a_quadratic_reach_and_halves_jitter() -> None:
    sm = PoseSmoother(7, 2)
    t = np.arange(20, dtype=np.float64)
    quad = [_pose([0.4 + 0.002 * k + 0.0003 * k * k, 0.1, 0.3], 2.0 * k) for k in t]
    # Centred and at the end of the plan (one-sided): no lag, no overshoot.
    for c in (10, 19):
        out = sm.smooth(quad, c)
        np.testing.assert_allclose(out[:3], quad[c][:3], atol=1e-9)
        np.testing.assert_allclose(out[3:], quad[c][3:], atol=1e-6)
    rng = np.random.default_rng(0)
    noisy = [_pose([0.4 + 0.01 * k + rng.normal(0, 0.003), 0.1, 0.3]) for k in t]
    raw = np.array([p[0] for p in noisy])
    smooth = np.array([sm.smooth(noisy, c)[0] for c in range(len(noisy))])
    jitter = lambda x: np.abs(np.diff(x, 2)).mean()
    assert jitter(smooth) < 0.5 * jitter(raw)


def test_smoothed_rotations_stay_proper() -> None:
    sm = PoseSmoother(7, 2)
    rng = np.random.default_rng(1)
    poses = [_pose([0.4, 0.1, 0.3], 3.0 * k + rng.normal(0, 2.0)) for k in range(9)]
    r = sm.smooth(poses, 4)[3:].reshape(3, 3)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)
    # Close to the steady turn at the centre (12 deg).
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    assert abs(yaw - 12.0) < 2.5


def test_player_config_refuses_bad_values() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        YamPlayerConfig(smoothing="ema")
    with pytest.raises(ValueError, match="odd"):
        YamPlayerConfig(window=6)
    with pytest.raises(ValueError, match="order"):
        YamPlayerConfig(window=3, order=3)
    assert YamPlayerConfig(smoothing="none").smoother() is None


# ---------------------------------------------------------------------------
# The trajectory player
# ---------------------------------------------------------------------------


def _ns(tick: float) -> int:
    return round(tick * ROW_NS)


def test_player_plays_a_chunk_row_by_row_and_holds_the_last() -> None:
    player = TrajectoryPlayer(FPS)
    assert player.command_at(0) is None
    chunk = _line(4)
    player.merge(chunk, 0, 0)
    cmds = [player.command_at(_ns(k)) for k in range(6)]
    xs = [c[0][0, 0] for c in cmds if c is not None]
    np.testing.assert_allclose(xs, [0.40, 0.41, 0.42, 0.43, 0.43, 0.43], atol=1e-6)


def test_chunk_boundaries_keep_the_played_row_and_smooth_across() -> None:
    """A chunked call at the next tick keeps the last row of the chunk before in
    the history, so the window is centred across the junction."""
    player = TrajectoryPlayer(FPS, PoseSmoother(7, 2))
    first_chunk = _line(24)
    player.merge(first_chunk, 0, 0)
    for k in range(24):
        player.command_at(_ns(k))
    # The next chunk starts 3 cm on (1 cm of reach plus a 2 cm jump).
    player.merge(_line(24, x0=0.66), _ns(24), _ns(24))
    cmd = player.command_at(_ns(24))
    assert cmd is not None
    assert player.played[-1].row is first_chunk[-1]
    assert len(player.played) == 16
    # The smoothed row takes part of the jump; the raw row would step 3 cm.
    assert 0.0 < cmd[0][0, 0] - 0.63 < 0.025


def test_rtc_merge_replaces_the_running_row_and_keeps_the_prefix() -> None:
    player = TrajectoryPlayer(FPS)
    cold = _line(21)
    player.merge(cold, 0, _ns(4))  # the cold start plays from its arrival
    for k in range(4, 17):
        player.command_at(_ns(k))
    prefix = player.action_prefix(_ns(16))
    # The rows from the one due at the sample on, on the row grid of the chunk.
    assert [id(r) for r in prefix["actions"]] == [id(r) for r in cold[12:]]
    assert prefix["chunk_offset_sec"] == pytest.approx(12 / FPS)
    assert prefix["observation_offset_sec"] == pytest.approx(0.0)
    # The answer to the sample at tick 16 arrives at tick 19.75. Its rows are
    # due from the sample on; its row 3 (a prefix row) takes the running slot.
    answer = prefix["actions"][:4] + _line(20, x0=0.7)
    player.merge(answer, _ns(16), _ns(19.75))
    cmd = player.command_at(_ns(20))
    assert cmd is not None
    assert cmd[0][0, 0] == pytest.approx(0.7, abs=1e-6)
    # Each slot is in the history once.
    assert [w.row for w in player.played[-4:]] == cold[12:16]
    assert len({id(w.row) for w in player.played}) == len(player.played)


def test_action_prefix_before_a_plan_is_empty() -> None:
    p = TrajectoryPlayer(FPS).action_prefix(0)
    assert p == {"actions": [], "chunk_offset_sec": 0.0, "observation_offset_sec": 0.0}


def test_smoothing_leaves_the_prefix_rows_of_the_server() -> None:
    """The server matches the prefix against its own rows, so the player sends them unchanged."""
    player = TrajectoryPlayer(FPS, PoseSmoother(7, 2))
    rng = np.random.default_rng(2)
    chunk = [
        _row(np.array([0.4 + 0.01 * i + rng.normal(0, 0.003), 0.3, 0.3]))
        for i in range(21)
    ]
    player.merge(chunk, 0, 0)
    for k in range(5):
        player.command_at(_ns(k))
    prefix = player.action_prefix(_ns(4))
    assert all(a is b for a, b in zip(prefix["actions"], chunk[4:], strict=True))


# ---------------------------------------------------------------------------
# Real-time chunking on the sim clock
# ---------------------------------------------------------------------------


def test_rtc_predicts_on_the_grid_of_the_server_and_holds_until_the_first_answer() -> (
    None
):
    calls: list[tuple[int, int]] = []

    def step(stack, executed: int) -> list[dict]:
        calls.append((stack()["tick"], executed))
        return _line(21 if not calls[:-1] else 24)

    hold = (np.zeros((2, 12)), np.zeros(2))
    rtc = RealTimeChunking(step, TrajectoryPlayer(FPS), 0.25, hold)
    cmds = [rtc.tick(k, lambda k=k: {"tick": k}) for k in range(42)]
    # The cold start at tick 0 arrives at 0.25 s (tick 3.75): the arms hold
    # until tick 4. The next prediction is a stride after the first chunk
    # starts (12 + 4 rows), then one every 12 rows.
    assert all(c is hold for c in cmds[:4]) and cmds[4] is not hold
    assert calls == [(0, 0), (16, 16), (28, 12), (40, 12)]
    # The 4 plan rows that run during a 0.25 s call repeat the plan.
    assert rtc.prefix_rows == [0, 4, 4, 4]


def test_rtc_joins_each_chunk_to_the_plan_without_a_jump() -> None:
    def step(stack, executed: int) -> list[dict]:
        # Each prediction restarts 3 cm off the plan: the junction hands it back.
        return _line(24, x0=0.43 if executed else 0.4)

    rtc = RealTimeChunking(
        step, TrajectoryPlayer(FPS), 0.25, (np.zeros((2, 12)), np.zeros(2))
    )
    xs = np.array([rtc.tick(k, dict)[0][0, 0] for k in range(60)])[4:]
    assert np.abs(np.diff(xs)).max() < 0.02


def test_rtc_refuses_a_latency_off_the_chunk() -> None:
    hold = (np.zeros((2, 12)), np.zeros(2))
    for bad in (-0.1, 0.9, float("nan")):
        with pytest.raises(ValueError, match="latency"):
            RealTimeChunking(lambda s, e: [], TrajectoryPlayer(FPS), bad, hold)
    with pytest.raises(ValueError, match="schedule"):
        YamSessionConfig(schedule="periodic")


# ---------------------------------------------------------------------------
# The model policy on the real-time session of the server code
# ---------------------------------------------------------------------------


class _Policy:
    """Stand-in networks: a slow reach plus row noise, from the seed."""

    def __init__(self, config: dict) -> None:
        self.config = config

    def sample_window_joint(
        self,
        text,
        text_uncond,
        anchor,
        cleans,
        proprio_all,
        seed,
        settings,
        trace=None,
        skip_last_video=None,
    ):
        generator = torch.Generator().manual_seed(int(seed))
        dim = 4 * int(self.config["action_space"]["dim"])
        actions = 0.02 * torch.randn((1, 6, dim), generator=generator)
        return torch.zeros(1, 1, 6, 1, 1), actions


class _Encoder:
    norm_block_c = None

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 16, 16))


class _Pipeline:
    embodiment = "bimanual"
    device = torch.device("cpu")

    def __init__(self) -> None:
        from praxis.pipeline import PraxisPipeline

        self._session = PraxisPipeline.session
        self.policy = _Policy(YAM)
        self.vae = _Encoder()
        self.sampler = SamplerSettings(num_steps=2, t_shift=1.0, cfg_video_scale=1.0)
        n = YAM["normalization"]
        self.stats = NormStats(
            action_q01=np.asarray(n["action_q01"], np.float32),
            action_q99=np.asarray(n["action_q99"], np.float32),
            proprio_mean=np.asarray(n["proprio_mean"], np.float32),
            proprio_std=np.asarray(n["proprio_std"], np.float32),
        )
        self.prompts: list[str] = []

    def encode_text(self, prompt: str) -> torch.Tensor:
        self.prompts.append(prompt)
        return torch.zeros(1, 4, 8)

    def encode_uncond_text(self) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def session(self, **kwargs: object):
        return self._session(self, **kwargs)  # ty: ignore[invalid-argument-type]


def _env(name: str) -> YamEnv:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"robot model files not available: {exc}")
    scene = YamSceneConfig(
        image_width=416, image_height=260, shadow_size=0, msaa_samples=0
    )
    try:
        return YamEnv(YamEnvConfig(scene=scene, task=YamTaskConfig(name=name)))
    except Exception as exc:  # no offscreen OpenGL context
        pytest.skip(f"no renderer: {exc}")


@pytest.mark.parametrize("smoothing", ["savgol", "none"])
def test_model_policy_runs_the_real_time_session_of_the_server(smoothing: str) -> None:
    """The default schedule predicts every 12 rows after the cold start, keeps
    the 4 plan rows that run during a 0.25 s call and never stops the arms."""
    env = _env("dice_in_bag")
    pipe = _Pipeline()
    try:
        policy = YamModelPolicy(
            pipe,  # ty: ignore[invalid-argument-type]
            session=YamSessionConfig(),
            player=YamPlayerConfig(smoothing=smoothing),
        )
        res = run_yam_episode(env, policy, 0, max_ticks=60)
        assert res.ticks == 60 and res.metrics["bad_rows"] == 0
        # Predictions at samples 0, 16, 28, 40 and 52 (0.25 s = 4 rows of wait).
        assert res.calls == 5
        assert res.model_prompt == env.prompt
        assert pipe.prompts[0] == env.prompt
    finally:
        env.close()


def test_model_policy_chunked_keeps_the_rows_without_smoothing() -> None:
    """``schedule: chunked`` with ``smoothing: none`` plays the rows of the session as they are."""
    env = _env("dice_in_bag")
    try:
        policy = YamModelPolicy(
            _Pipeline(),  # ty: ignore[invalid-argument-type]
            session=YamSessionConfig(schedule="chunked"),
            player=YamPlayerConfig(smoothing="none"),
        )
        res = run_yam_episode(env, policy, 0, max_ticks=45)
        assert res.calls == 2 and res.ticks == 45
    finally:
        env.close()


def test_model_policy_refuses_the_video_decode_under_rtc() -> None:
    with pytest.raises(ValueError, match="chunked"):
        YamModelPolicy(
            _Pipeline(),  # ty: ignore[invalid-argument-type]
            decode_video=True,
        )
