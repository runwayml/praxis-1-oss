"""The real-time chunking of the YAM sim is the one of the server, bit for bit.

The sim does not import the policy server (a release without the server
deletes it), so ``praxis/sim/yam_player.py`` carries a copy of the real-time
rules of the server: the prediction grid, the delay estimate and the junction
rule. These tests pin the copy to the server code: the junction functions on
random rows, and whole sim episodes against a client that drives the
server's real-time session (``ServedPolicy``) with the periodic protocol on
the sim clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
pytest.importorskip("praxis_sim")  # the `sim` extra

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis import bimanual
from praxis.model import SamplerSettings
from praxis.robot_io import NormStats
from praxis.serve import rtc as server_rtc
from praxis.serve.policy import PolicyFactory, ServedPolicy
from praxis.sim import assets
from praxis.sim import yam_player as sim_rtc
from praxis.sim.robot import pos_rot_to_pose12, pose7_to_pos_rot, quat_to_mat
from praxis.sim.yam_env import YamEnv, YamEnvConfig
from praxis.sim.yam_loop import (
    VIEW_KEYS,
    YamHistory,
    YamModelPolicy,
    YamPolicyOutput,
    YamSessionConfig,
    run_yam_episode,
)
from praxis.sim.yam_player import NS, TrajectoryPlayer, YamPlayerConfig
from praxis.sim.yam_scene import YamSceneConfig
from praxis.sim.yam_tasks import YamTaskConfig

ROOT = Path(__file__).resolve().parents[2]
YAM = json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
FPS = 15.0


def _rows(rng: np.random.Generator, n: int) -> np.ndarray:
    out = np.zeros((n, 2, 13))
    for a in range(2):
        base = np.array([0.5, 0.3 - 0.6 * a, 0.3])
        xyz = np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0) + base
        out[:, a, :3] = xyz
        for t in range(n):
            q = rng.normal(0, 1, 4)
            q /= np.linalg.norm(q)
            out[t, a, 3:12] = quat_to_mat(q).reshape(9).astype(np.float32)
        out[:, a, 12] = rng.uniform(0, 1, n)
    return out


@pytest.mark.parametrize("p", [0, 1, 2, 4, 9, 12])
def test_the_junction_rule_is_the_one_of_the_server(p: int) -> None:
    rng = np.random.default_rng(p)
    new, plan = _rows(rng, 24), _rows(rng, 21)
    for decay, fade in ((24, 6), (0, 0), (8, 0), (24, 0)):
        ours = sim_rtc.rebase_into_prefix(new, plan[:p], decay, fade)
        theirs, _ = server_rtc.rebase_into_prefix(new, plan[:p], decay, fade)
        np.testing.assert_array_equal(ours, theirs)
    cont = plan[p:] if p < len(plan) else plan[-1:]
    a, b = ours.copy(), theirs.copy()
    sim_rtc.grip_crossfade(a, p, cont, 16)
    server_rtc.grip_crossfade(b, p, cont, 16)
    np.testing.assert_array_equal(a, b)
    assert sim_rtc.JunctionConfig() == sim_rtc.JunctionConfig(
        **{
            k: v
            for k, v in server_rtc.RTCConfig(
                reach_max_m=(bimanual.REACH_MAX_M, bimanual.REACH_MAX_M)
            ).__dict__.items()
            if k != "grip"
        }
    )


def test_prefix_rows_and_the_delay_cap_are_the_ones_of_the_server() -> None:
    for d in np.linspace(0, 1.2, 97):
        assert sim_rtc.prefix_rows(d, FPS, 21) == server_rtc.prefix_rows(d, FPS, 21)
    assert sim_rtc.JunctionConfig().max_delay_sec(
        FPS
    ) == server_rtc.RTCConfig().max_delay_sec(FPS)


# ---------------------------------------------------------------------------
# Whole episodes: the sim against the server's real-time session
# ---------------------------------------------------------------------------


class _Policy:
    def __init__(self, config: dict) -> None:
        self.config = config

    def sample_window_joint(
        self, text, text_uncond, anchor, cleans, proprio_all, seed, settings,
        trace=None, skip_last_video=None,
    ):  # fmt: skip
        generator = torch.Generator().manual_seed(int(seed))
        dim = 4 * int(self.config["action_space"]["dim"])
        actions = 0.05 * torch.randn((1, 6, dim), generator=generator)
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

    def encode_text(self, prompt: str) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def encode_uncond_text(self) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def session(self, **kwargs: object):
        return self._session(self, **kwargs)  # ty: ignore[invalid-argument-type]


class _ServerClient:
    """The periodic client of the robot on the sim clock, against ``ServedPolicy``.

    Each tick it delivers the answers due by then into the player, sends the
    sample (frames, poses, grips, ``sample_index``, ``obs_time_ns``,
    ``call_id``, the durations of the answered calls, ``action_prefix``) and
    plays the command of the tick. Every answer arrives ``latency_s`` after
    its sample.
    """

    name = "server"

    def __init__(self, latency_s: float, player: YamPlayerConfig) -> None:
        self.factory = PolicyFactory(
            _Pipeline(),  # ty: ignore[invalid-argument-type]
            {
                "exec_horizon": 24,
                "control_mode": "none",
                "max_context_chunks": 1,
                "precompute": False,
            },
            "/ckpt/praxis-1-yam",
        )
        self.latency_ns = round(latency_s * NS)
        self.player_cfg = player
        self.served: ServedPolicy | None = None

    def reset(self, env: YamEnv, seed: int) -> None:
        self.served = self.factory(
            {"schedule": "rtc", "prompt": env.prompt, "sample_seed": seed * 1000}
        )
        self.player = TrajectoryPlayer(FPS, self.player_cfg.smoother())
        self.in_flight: list[tuple[int, int, int, list[dict] | None]] = []
        self.timings: list[dict] = []

    def _deliver(self, now: int) -> None:
        while self.in_flight and self.in_flight[0][2] <= now:
            call_id, obs_ns, arrive, rows = self.in_flight.pop(0)
            self.timings.append(
                {"call_id": call_id, "elapsed_sec": (arrive - obs_ns) / NS}
            )
            if rows is not None:
                self.player.merge(rows, obs_ns, arrive)

    def act(self, history: YamHistory) -> YamPolicyOutput:
        served = self.served
        assert served is not None
        start = history.latest
        hold = (
            np.stack(
                [
                    pos_rot_to_pose12(*pose7_to_pos_rot(start.ee_pose[s]))
                    for s in bimanual.SIDES
                ]
            ).astype(np.float64),
            np.zeros(2),
        )

        def stream():
            while True:
                obs = history.latest
                now = round(obs.tick * self.player.row_ns)
                self._deliver(now)
                sample: dict = {key: obs.views[view] for view, key in VIEW_KEYS.items()}
                for s in bimanual.SIDES:
                    sample[f"robot_state.{s}.ee_pose"] = np.asarray(
                        obs.ee_pose[s], np.float32
                    )
                    sample[f"grip.{s}"] = float(obs.grip[s])
                sample.update(
                    sample_index=obs.tick, obs_time_ns=now, call_id=obs.tick,
                    call_timings=self.timings, action_prefix=self.player.action_prefix(now),
                )  # fmt: skip
                self.timings = []
                rows = served.select_action(sample)
                self.in_flight.append((obs.tick, now, now + self.latency_ns, rows))
                self._deliver(now)
                cmd = self.player.command_at(now)
                yield hold if cmd is None else cmd

        empty = np.zeros((0, 2, 12), np.float32)
        return YamPolicyOutput(empty, np.zeros((0, 2), np.float32), {}, stream=stream())


def _commands(policy, ticks: int) -> np.ndarray:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"robot model files not available: {exc}")
    scene = YamSceneConfig(
        image_width=416, image_height=260, shadow_size=0, msaa_samples=0
    )
    try:
        env = YamEnv(YamEnvConfig(scene=scene, task=YamTaskConfig(name="dice_in_bag")))
    except Exception as exc:  # no offscreen OpenGL context
        pytest.skip(f"no renderer: {exc}")
    log: list[np.ndarray] = []
    step = env.step

    def logged(poses, grips):
        log.append(
            np.concatenate(
                [
                    np.asarray(poses, np.float64).reshape(-1),
                    np.asarray(grips, np.float64),
                ]
            )
        )
        return step(poses, grips)

    env.step = logged
    try:
        run_yam_episode(env, policy, 0, max_ticks=ticks)
    finally:
        env.close()
    return np.stack(log)


@pytest.mark.parametrize("smoothing", ["none", "savgol"])
@pytest.mark.parametrize("latency_s", [0.25, 0.6])
def test_a_sim_episode_runs_the_real_time_session_of_the_server(
    smoothing: str, latency_s: float
) -> None:
    player = YamPlayerConfig(smoothing=smoothing)
    ours = _commands(
        YamModelPolicy(
            _Pipeline(),  # ty: ignore[invalid-argument-type]
            session=YamSessionConfig(latency_s=latency_s),
            player=player,
        ),
        66,
    )
    theirs = _commands(_ServerClient(latency_s, player), 66)
    np.testing.assert_array_equal(ours, theirs)
