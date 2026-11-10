"""Real-time chunking over the periodic flow (CPU only, no weights).

The units: the sample history, the prediction grid, the call timings, the
client recipe, the junction rule, the reach clamp and the stream appends off
the chunk grid. The session tests drive a bimanual session with the stand-in
networks of ``test_serve_protocol`` the way the periodic recipe of a
Positronic client does: one sample every 1/30 s with the current frames
only. The rows that each prediction serves and the frames that its window
encodes must be the ones of the chunked frame stack on the same calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from praxis import bimanual
from praxis.model import SamplerSettings
from praxis.robot_io import ContiguousStreamContext, NormStats
from praxis.serve import params, rtc
from praxis.serve import periodic as per
from praxis.serve.policy import PolicyFactory, ServedPolicy

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
YAM = json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
FPS = 15.0
SAMPLE_FPS = 30.0


# ---------------------------------------------------------------------------
# Sample history, grid, call timings, client recipe
# ---------------------------------------------------------------------------


def _sample(i: int) -> dict:
    return {
        "img": np.full((2, 2, 3), i % 256, np.uint8),
        "pose": np.array([i], np.float32),
    }


def test_history_rebuilds_the_training_grid_from_every_second_sample() -> None:
    """At twice the action rate, row ``b`` back is sample ``n - 2 b``. Rows
    before the first sample are left out (``pad_start=False``)."""
    h = per.SampleHistory(keys=("img", "pose"), history_rows=4, ratio=2)
    for i in range(20):
        h.append(_sample(i), i)
    out = h.stacked({"img": None, "other": 7}, 19)
    assert out["pose"][:, 0].tolist() == [11, 13, 15, 17, 19]
    assert out["other"] == 7
    early = per.SampleHistory(keys=("img", "pose"), history_rows=4, ratio=2)
    for i in range(3):
        early.append(_sample(i), i)
    assert early.stacked({}, 2)["pose"][:, 0].tolist() == [0, 2]


def test_history_takes_the_sample_before_a_skipped_grid_position() -> None:
    h = per.SampleHistory(keys=("pose",), history_rows=3, ratio=2)
    for i in (0, 1, 2, 3, 6, 7, 8):  # 4 and 5 skipped
        h.append(_sample(i), i)
    assert h.stacked({}, 8)["pose"][:, 0].tolist() == [2, 3, 6, 8]


def test_history_refuses_samples_out_of_order_and_without_the_stack_keys() -> None:
    h = per.SampleHistory(keys=("img", "pose"), history_rows=2, ratio=2)
    h.append(_sample(3), 3)
    with pytest.raises(ValueError, match="must increase"):
        h.append(_sample(3), 3)
    with pytest.raises(ValueError, match="stack keys"):
        h.append({"img": np.zeros(1)}, 4)


def test_history_keeps_a_bounded_window_and_drops_before_a_jump() -> None:
    h = per.SampleHistory(keys=("pose",), history_rows=4, ratio=2, capacity=12)
    for i in range(100):
        h.append(_sample(i), i)
    assert len(h) == 12 and h.newest_index == 99
    h.drop_before(95)
    assert len(h) == 5 and h.at(94) is None and h.at(95) is not None


def test_grid_times_calls_as_the_real_time_schedule_does() -> None:
    """The first chunk starts at its answer: the next prediction is ``stride``
    rows after the reported duration. Later chunks start at their observation."""
    g = per.PredictGrid(ratio=2, sample_fps=SAMPLE_FPS)
    assert g.due(0)
    g.predicted(0, 12, call_id=0)
    assert not g.due(24)  # waiting for the duration of the first answer
    g.answered(0, 0.3)  # 9 samples at 30 Hz
    assert g.next_index == 0 + 24 + 9 and g.last_index == 0
    assert not g.due(32) and g.due(33)
    g.predicted(33, 12, call_id=9)
    assert g.next_index == 57


def test_grid_predicts_without_a_duration_report_after_the_wait() -> None:
    g = per.PredictGrid(ratio=2, sample_fps=SAMPLE_FPS)
    g.predicted(0, 12, call_id=0)
    last_wait = 24 + round(per.MAX_WAIT_SEC * SAMPLE_FPS) - 1
    assert not g.due(last_wait) and g.due(last_wait + 1)
    g.predicted(last_wait + 1, 12, call_id=55)
    assert g.next_index == last_wait + 1 + 24
    assert not g.due(last_wait + 24) and g.due(last_wait + 25)


def test_call_timings_keep_the_prediction_calls_only() -> None:
    t = per.CallTimings()
    t.predicted.update({0, 5})
    obs = {
        per.CALL_TIMINGS: [{"call_id": i, "elapsed_sec": 0.01 * i} for i in range(7)]
    }
    assert t.collect(obs) == [(0, 0.0), (5, 0.05)]
    assert t.take() == [0.0, 0.05] and t.take() == []
    # Each call is reported once.
    assert t.collect({per.CALL_TIMINGS: [{"call_id": 0, "elapsed_sec": 1.0}]}) == []
    with pytest.raises(ValueError, match="list"):
        t.collect({per.CALL_TIMINGS: 3})


@pytest.mark.parametrize("call_id", [[1], {"id": 1}, "1", 1.0, None, True])
def test_call_timings_skip_a_malformed_call_id(call_id: object) -> None:
    """A ``call_id`` that is not an integer is skipped like an unknown one.
    An unhashable one (``[1]``) raises no TypeError. ``True`` is a bool, not
    id 1, so the real ``1`` entry next to it is the one used."""
    t = per.CallTimings()
    t.predicted.add(1)
    obs = {
        per.CALL_TIMINGS: [
            {"call_id": call_id, "elapsed_sec": 0.2},
            {"call_id": 1, "elapsed_sec": 0.3},
        ]
    }
    assert t.collect(obs) == [(1, 0.3)]
    assert t.predicted == set() and t.take() == [0.3]


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"sample_fps": 20.0}, "whole multiple"),
        ({"sample_fps": 7.5}, "whole multiple"),
        ({"sample_fps": 90.0}, "sample_fps"),
        ({"max_in_flight": 0}, "max_in_flight"),
        ({"max_in_flight": 17}, "max_in_flight"),
    ],
)
def test_periodic_config_refuses_bad_values(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        per.PeriodicConfig(**kw).ratio(FPS)


def test_client_recipe_replaces_the_stack_and_the_schedule() -> None:
    base = params.local_stack(bimanual.STACK_KEYS, 24, FPS, 416, 416, True)
    spec = per.client_recipe(base, fps=FPS, sample_fps=SAMPLE_FPS)
    assert [e["name"] for e in spec["seq"]] == [
        "stop_on_fault",
        "trajectory_player",
        "periodic",
        "inference_schedule",
        "restrict_image_size",
    ]
    assert all("version" in e for e in spec["seq"])
    assert spec["seq"][0]["version"] == 2
    assert spec["seq"][1]["args"] == {"plan_key": "plan", "replan_on_error": True}
    assert spec["seq"][2]["args"] == {"fps": 30.0, "obs_time_key": "obs_time_ns"}
    sched = spec["seq"][3]["args"]
    assert sched["request_timing"] == {"reference": "every", "seconds": 1 / 30.0}
    assert sched["merge"] == {"start": "observation", "horizon_sec": None}
    assert sched["latency_estimate"] == 0.0 and sched["fps"] == FPS


# ---------------------------------------------------------------------------
# The junction rule and the reach clamp
# ---------------------------------------------------------------------------


def _line(x0: float, n: int, step: float = 0.01) -> np.ndarray:
    """(n, 1, 13) rows moving along x by ``step`` per row, identity rotation."""
    out = np.zeros((n, 1, 13))
    out[:, 0, 0] = x0 + step * np.arange(n)
    out[:, 0, 3:12] = np.eye(3).reshape(-1)
    return out


def _loop_root(prefix_rows: int = 11, stride: int = 12) -> tuple[float, float]:
    """The junction loop under a model that plans from the measured pose.

    The plan of call ``k`` starts at the pose of a lagless arm at call ``k``
    plus a random offset and moves at a random speed. At the next call the
    robot runs row ``stride`` of this answer, and the next prefix is rows
    ``stride ..`` of it. Returns ``a`` of ``d_{k+1} = a d_k`` over the moves
    per call, and the median junction step in cm.
    """
    rng = np.random.default_rng(0)
    x, prev = 0.0, _line(0.0, 24, step=0.0)
    xs, jumps = [x], []
    for _ in range(300):
        plan = _line(x + rng.normal(0, 0.03), 24, step=rng.normal(0, 0.003))
        prefix = prev[stride : stride + prefix_rows]
        out, _ = rtc.rebase_into_prefix(plan, prefix)
        jumps.append(abs(out[prefix_rows, 0, 0] - out[prefix_rows - 1, 0, 0]))
        x = float(out[stride, 0, 0])
        xs.append(x)
        prev = out
    d = np.diff(xs)[20:]
    d = d - d.mean()
    a = float((d[1:] * d[:-1]).sum() / (d[:-1] ** 2).sum())
    return a, float(np.median(jumps) * 100)


def test_rebase_leaves_no_junction_loop_and_no_jump() -> None:
    """The measured pose never enters the output, so the moves per call do
    not alternate. A crossfade toward the plan gives ``a`` near -0.8."""
    a, jump = _loop_root()
    assert abs(a) < 0.25 and jump < 0.5, (a, jump)


def test_rebase_decay_hands_the_offset_back() -> None:
    prefix = _line(0.0, 5)
    new = _line(-0.08, 24, step=0.02)
    dx = prefix[4, 0, 0] - new[4, 0, 0]
    for d in (6, 12, 24):
        out, stats = rtc.rebase_into_prefix(new, prefix, d)
        w = [1.0 - rtc._ramp(i, d) if i < d else 0.0 for i in range(19)]
        np.testing.assert_allclose(
            out[5:, 0, 0], new[5:, 0, 0] + np.asarray(w) * dx, atol=1e-12
        )
        np.testing.assert_allclose(out[5 + d :], new[5 + d :], atol=1e-12)
        np.testing.assert_array_equal(out[:5], prefix)
        assert stats["junction_cm"] == pytest.approx(abs(dx) * 100)
    held, _ = rtc.rebase_into_prefix(new, prefix, 0)
    np.testing.assert_allclose(held[5:, 0, 0], new[5:, 0, 0] + dx, atol=1e-12)


def test_vel_fade_turns_the_seam_and_hands_the_position_back() -> None:
    prefix = _line(0.0, 6)  # +x at 1 cm per row
    new = _line(0.0, 24, step=0.0)
    new[:, 0, 1] = 0.03 - 0.01 * np.arange(24)  # the model moves -y
    plain, _ = rtc.rebase_into_prefix(new, prefix, 24)
    out, _ = rtc.rebase_into_prefix(new, prefix, 24, 6)
    np.testing.assert_allclose(out[:6], prefix)
    # The first served step mostly continues the prefix (+x), not the model (-y).
    step = out[6, 0, :3] - out[5, 0, :3]
    assert step[0] > 0.7 * 0.01 and abs(step[1]) < 0.3 * 0.01
    # After the decay the rows are the rule's own. Rotations and grips stay.
    np.testing.assert_allclose(out[6 + 24 - 6 :, 0, :3], plain[6 + 24 - 6 :, 0, :3])
    np.testing.assert_allclose(out[:, :, 3:], plain[:, :, 3:])


def test_rotation_offset_is_applied_from_the_left_and_decays() -> None:
    prefix = _line(0.0, 4)
    new = _line(0.0, 24)
    turn = np.array(
        [
            [math.cos(0.2), -math.sin(0.2), 0.0],
            [math.sin(0.2), math.cos(0.2), 0.0],
            [0, 0, 1],
        ]
    )
    prefix[:, 0, 3:12] = turn.reshape(-1)
    out, stats = rtc.rebase_into_prefix(new, prefix, 8)
    assert stats["junction_deg"] == pytest.approx(math.degrees(0.2))
    np.testing.assert_allclose(out[4, 0, 3:12].reshape(3, 3), turn, atol=0.03)
    np.testing.assert_allclose(out[4 + 8 :, 0, 3:12], new[4 + 8 :, 0, 3:12], atol=1e-12)


def test_grip_crossfade_fades_from_the_plan_to_the_model() -> None:
    new = np.zeros((24, 2, 13))
    new[4:, 0, 12] = 0.9
    cont = np.zeros((12, 2, 13))
    cont[:, 0, 12] = 0.1
    out = new.copy()
    rtc.grip_crossfade(out, 4, cont, 6)
    want = [0.1 + rtc._ramp(i, 6) * (0.9 - 0.1) for i in range(6)]
    np.testing.assert_allclose(out[4:10, 0, 12], want)
    np.testing.assert_array_equal(out[10:, 0, 12], new[10:, 0, 12])
    np.testing.assert_array_equal(out[:4], new[:4])
    for p, n in ((0, 6), (4, 0)):
        same = new.copy()
        rtc.grip_crossfade(same, p, cont, n)
        np.testing.assert_array_equal(same, new)


def _yam_rows(xyz_right: np.ndarray, tool_z: tuple[float, ...]) -> np.ndarray:
    t = len(xyz_right)
    out = np.zeros((t, 2, 13))
    z = np.asarray(tool_z, np.float64) / np.linalg.norm(tool_z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    r = np.stack([x, np.cross(z, x), z], axis=1)
    for a in range(2):
        out[:, a, 3:12] = r.reshape(-1)
    out[:, 0, :3] = (0.45, 0.305, 0.40)
    out[:, 1, :3] = xyz_right
    return out


def test_reach_clamp_slides_far_targets_onto_the_radius_only() -> None:
    tool_z = (0.8, 0.0, -0.6)
    xyz = np.array([[0.70, -0.20, 0.40], [1.01, -0.10, 0.36], [1.03, -0.12, 0.40]])
    rows = _yam_rows(xyz, tool_z)
    before = rows.copy()
    assert bimanual.clamp_reach(rows, 0, (0.595, 0.595)) == 2
    np.testing.assert_array_equal(rows[0], before[0])
    np.testing.assert_array_equal(rows[:, 0], before[:, 0])
    np.testing.assert_array_equal(rows[:, :, 3:], before[:, :, 3:])
    z = np.asarray(tool_z) / np.linalg.norm(tool_z)
    base = np.asarray(bimanual.ARM_MOUNTS_M[1])
    for t in (1, 2):
        flange = rows[t, 1, :3] - bimanual.TCP_FROM_FLANGE_M * z
        rel = flange - base
        az = math.atan2(rel[1], rel[0])
        shoulder = base + np.array([0.02 * math.cos(az), 0.02 * math.sin(az), 0.104])
        assert np.linalg.norm(flange - shoulder) == pytest.approx(0.595, abs=1e-9)
    again = before.copy()
    assert bimanual.clamp_reach(again, 2, (0.595, 0.595)) == 1
    np.testing.assert_array_equal(again[:2], before[:2])


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"stride": 21}, "rtc_stride"),
        ({"rebase_decay": -1}, "rtc_rebase_decay"),
        ({"vel_fade": 25}, "rtc_vel_fade"),
        ({"rebase_decay": 6, "vel_fade": 6}, "more than rtc_vel_fade"),
        ({"grip": "hold"}, "rtc_grip"),
        ({"grip_rows": 30}, "rtc_grip_rows"),
    ],
)
def test_rtc_settings_validate(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rtc.RTCConfig(**kw)


def test_rtc_settings_parse() -> None:
    assert params.parse_reach("auto", 0.595) == (0.595, 0.595)
    assert params.parse_reach("off", 0.595) == ()
    assert params.parse_reach("0.6,0.55", 0.595) == (0.6, 0.55)
    with pytest.raises(ValueError, match="rtc_reach_max_m"):
        params.parse_reach("2.0", 0.595)
    assert rtc.prefix_rows(0.0, FPS, 20) == 0
    assert rtc.prefix_rows(0.2, FPS, 20) == 3  # rows at 0, 66.7 and 133.3 ms
    assert rtc.prefix_rows(5.0, FPS, 7) == 7


def test_read_plan_reads_the_plan_wire_and_refuses_the_sampled_list() -> None:
    chans = rtc.channels(bimanual.SIDES)
    row = {
        "robot_command.left": {
            "type": "cartesian_pos",
            "pose": np.r_[0.1, 0.2, 0.3, np.eye(3).ravel()],
        },
        "robot_command.right": {
            "type": "cartesian_pos",
            "pose": np.r_[0.4, 0.5, 0.6, np.eye(3).ravel()],
        },
        "target_grip.left": 0.25,
        "target_grip.right": 0.75,
    }
    prefix = {
        "actions": [row, row],
        "chunk_offset_sec": 4 / FPS,
        "observation_offset_sec": -0.01,
    }
    plan = rtc.read_plan({"action_prefix": prefix}, chans, FPS)
    assert plan is not None and plan.shape == (2, 2, 13)
    assert plan[0, 1, 0] == pytest.approx(0.4) and plan[1, 0, 12] == 0.25
    assert rtc.read_plan({}, chans, FPS) is None
    with pytest.raises(ValueError, match="sampled prefix list"):
        rtc.read_plan({"action_prefix": [row]}, chans, FPS)
    with pytest.raises(ValueError, match="not a row"):
        rtc.read_plan(
            {"action_prefix": {**prefix, "chunk_offset_sec": 0.01}}, chans, FPS
        )


# ---------------------------------------------------------------------------
# The stream off the chunk grid
# ---------------------------------------------------------------------------


def _frames(start: int, n: int) -> torch.Tensor:
    """(1, 3, n, 2, 2) frames whose pixels hold their raw index."""
    idx = torch.arange(start, start + n, dtype=torch.float32)
    return idx.view(1, 1, n, 1, 1).expand(1, 3, n, 2, 2).clone()


def _raw_ids(clip: torch.Tensor) -> list[int]:
    return [int(v) for v in clip[0, 0, :, 0, 0]]


def test_stream_off_the_grid_serves_the_newest_window() -> None:
    """Appends of any count keep the newest raws; the window ends right before
    the current frame. On the chunk grid both rules give the same window."""
    seen: list[list[int]] = []

    def encode(clip: torch.Tensor) -> torch.Tensor:
        seen.append(_raw_ids(clip))
        return torch.zeros(1, 1, 1 + (clip.shape[2] - 1) // 4, 1, 1)

    s = ContiguousStreamContext(max_context_chunks=3)
    s.append_executed(_frames(0, 30), strict=False)
    assert s.depth == 1 and s.window_start == 9
    s.cleans(encode)
    assert seen[-1] == list(range(9, 30))
    anchor = s.anchor_frame()
    assert anchor is not None and _raw_ids(anchor) == [9]
    s.append_executed(_frames(30, 24), strict=False)
    assert s.depth == 2 and s.window_start == 9
    s.append_executed(_frames(54, 24), strict=False)
    assert s.n_raws == s.max_raws == 69 and s.window_start == 0
    s.cleans(encode)
    assert seen[-1] == list(range(9, 78))
    small = ContiguousStreamContext(max_context_chunks=1)
    small.append_executed(_frames(0, 12), strict=False)
    assert small.depth == 0 and small.anchor_frame() is None
    small.append_executed(_frames(12, 12), strict=False)
    assert (
        small.n_raws == 21 and small.cleans(encode) and seen[-1] == list(range(3, 24))
    )
    with pytest.raises(ValueError, match="expected 24"):
        s.append_executed(_frames(78, 12))
    with pytest.raises(ValueError, match="at least one raw"):
        s.append_executed(_frames(78, 0), strict=False)
    grid = ContiguousStreamContext(max_context_chunks=3)
    grid.append_executed(_frames(0, 21))
    grid.append_executed(_frames(21, 24))
    assert grid.window_start == 0 and grid.boundary_offsets() == [0, 21]


# ---------------------------------------------------------------------------
# A bimanual session driven like the periodic recipe
# ---------------------------------------------------------------------------


class _Policy:
    """Stand-in networks: the actions follow the anchor, the proprio and the noise."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.calls: list[dict] = []

    def sample_window_joint(
        self, text, text_uncond, anchor, cleans, proprio_all, seed, settings, trace=None
    ):
        self.calls.append({"depth": len(cleans), "seed": int(seed)})
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn((1, 6, 80), generator=generator)
        proprio = float(proprio_all[..., -1, :].sum())
        actions = 0.1 * noise + 0.001 * float(anchor.mean()) + 0.001 * proprio
        return torch.zeros(1, 1, 6, 1, 1), actions


class _Encoder:
    """A causal stand-in for the VAE that records the raw index of each frame."""

    norm_block_c = None

    def __init__(self) -> None:
        self.raws: list[list[int]] = []

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        self.raws.append([int(v) for v in clip[0, 2, :, 0, 0].round()])
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 16, 16))


class _Pipeline:
    def __init__(self) -> None:
        from praxis.pipeline import PraxisPipeline

        self._session = PraxisPipeline.session
        self.device = torch.device("cpu")
        self.policy = _Policy(YAM)
        self.vae = _Encoder()
        self.sampler = SamplerSettings(num_steps=2, t_shift=1.0, cfg_video_scale=1.0)
        self.embodiment = "bimanual"
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


def _factory() -> PolicyFactory:
    return PolicyFactory(
        _Pipeline(),  # ty: ignore[invalid-argument-type]
        {"exec_horizon": 24, "control_mode": "none", "max_context_chunks": 1},
        "/ckpt/praxis-1-yam",
    )


def _frame(raw: float, h: int = 260, w: int = 416) -> np.ndarray:
    """A view whose blue channel holds the raw index (the encoder reads it)."""
    f = np.zeros((h, w, 3), np.uint8)
    f[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    f[..., 2] = int(raw) % 256
    return f


def _pose(t: float, side: int) -> np.ndarray:
    return np.array(
        [
            0.45 + 0.05 * math.sin(t),
            0.3 - 0.6 * side + 0.02 * t,
            0.4,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        np.float32,
    )


def _obs(raws: list[float]) -> dict:
    """The observation at the action raws ``raws`` (one row per raw, current last)."""
    t = np.asarray(raws) / FPS
    obs: dict = {}
    for view in bimanual.VIEWS:
        obs[bimanual.VIEW_KEY_CANDIDATES[view][0]] = np.stack([_frame(r) for r in raws])
    for j, side in enumerate(bimanual.SIDES):
        obs[bimanual.pose_keys(side)[0]] = np.stack([_pose(x, j) for x in t])
        obs[bimanual.grip_keys(side)[0]] = np.full(len(raws), 0.1 * j, np.float32)
    return obs


def _current(obs: dict) -> dict:
    return {k: v[-1] for k, v in obs.items()}


def _plan_context(chunk: list[dict] | None, start: float, t: float) -> dict:
    """The ``action_prefix`` of the client at time ``t`` for the plan ``chunk`` started at ``start``."""
    if chunk is None:
        return {}
    running = min(int(np.floor((t - start) * FPS + 1e-9)), len(chunk))
    return {
        "action_prefix": {
            "actions": chunk[running:],
            "chunk_offset_sec": running / FPS,
            "observation_offset_sec": start + running / FPS - t,
        }
    }


def _drive(
    policy: ServedPolicy,
    delays: list[float],
    seconds: float,
    late: dict[int, int] | None = None,
) -> tuple[list[list[dict]], list[int], list[int]]:
    """Drive a session as the periodic recipe does, one call in flight.

    ``delays[k]`` is the duration of prediction ``k``. ``late`` maps a sample
    index to the index after which it arrives (out of order). Returns the
    answered chunks, their sample indices and the sample indices that
    answered None.
    """
    chunk: list[dict] | None = None
    start = 0.0
    unreported: list[dict] = []
    answers: list[list[dict]] = []
    at: list[int] = []
    nones: list[int] = []
    order = list(range(round(seconds * SAMPLE_FPS)))
    for n, after in (late or {}).items():
        order.remove(n)
        order.insert(order.index(after) + 1, n)
    k = 0
    for n in order:
        if k >= len(delays):
            break
        t = n / SAMPLE_FPS
        obs = _current(_obs([t * FPS]))
        obs.update(_plan_context(chunk, start, t))
        obs.update(
            {
                per.SAMPLE_INDEX: n,
                per.OBS_TIME_NS: round(t * 1e9),
                per.CALL_ID: n,
                per.CALL_TIMINGS: unreported,
            }
        )
        rows = policy.select_action(obs)
        elapsed = 0.004 if rows is None else delays[k]
        unreported = [{per.CALL_ID: n, per.ELAPSED_SEC: elapsed}]
        if rows is None:
            nones.append(n)
            continue
        answers.append(rows)
        at.append(n)
        start = t + elapsed if chunk is None else t
        chunk = rows
        k += 1
    return answers, at, nones


DELAYS = [0.30, 0.27, 0.41, 0.33, 0.29, 0.31, 0.36, 0.30]


def test_a_real_time_session_declares_the_periodic_recipe() -> None:
    policy = _factory()({"schedule": "rtc"})
    meta = policy.meta
    names = [e["name"] for e in policy.local_stack["seq"]]
    assert names == [
        "stop_on_fault",
        "trajectory_player",
        "periodic",
        "inference_schedule",
        "restrict_image_size",
    ]
    assert policy.protocol_version == 2 and policy.max_in_flight == 8
    assert meta["schedule"] == "rtc"
    assert meta["periodic"] == {"sample_fps": 30.0, "max_in_flight": 8}
    assert meta["rtc"] == {
        "stride": 12,
        "max_delay_sec": 12 / FPS,
        "rebase_decay": 24,
        "vel_fade": 6,
        "grip": "cont",
        "grip_rows": 16,
        "reach_max_m": [0.595, 0.595],
    }
    chunked = _factory()({})
    assert chunked.protocol_version == 1 and chunked.max_in_flight == 1
    assert "rtc" not in chunked.meta and "schedule" not in chunked.meta


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schedule": "fast"}, "schedule"),
        ({"rtc_stride": 8}, "schedule=rtc"),
        ({"schedule": "rtc", "context_mode": "stateless"}, "observed"),
        ({"schedule": "rtc", "max_context_chunks": 0}, "observed"),
        ({"schedule": "rtc", "exec_horizon": 12}, "exec_horizon"),
        ({"schedule": "rtc", "sample_fps": 20}, "whole multiple"),
        ({"schedule": "rtc", "rtc_reach_max_m": "5"}, "rtc_reach_max_m"),
    ],
)
def test_a_real_time_session_refuses_bad_settings(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _factory()(overrides)


def test_a_real_time_session_predicts_on_the_real_time_grid() -> None:
    """The first prediction at once. The next one 12 rows (24 samples) plus the
    first duration (0.30 s, 9 samples) later. Then every 24 samples. Every
    other sample answers None, and each answer has no timestamps."""
    policy = _factory()({"schedule": "rtc", "sample_seed": 0})
    answers, at, nones = _drive(policy, DELAYS, 9.0)
    assert len(answers) == len(DELAYS)
    gaps = np.diff(at).tolist()
    assert gaps[0] == 24 + 9 and set(gaps[1:]) == {24}
    assert len(nones) > 10 * len(answers)
    assert all("timestamp" not in row for rows in answers for row in rows)
    # The second prediction has 16 rows of history, less than the 21 of a
    # chunk: a cold start again. From the third one the context holds a chunk.
    assert [len(r) for r in answers] == [21, 21, *[24] * (len(answers) - 2)]


def test_the_window_of_each_prediction_is_the_chunked_stack_of_that_call() -> None:
    """The raws that each prediction encodes are the stack rows that the
    chunked recipe would send on the same call, ending right before the
    current frame: 21 raws, one every 1/15 s."""
    factory = _factory()
    policy = factory({"schedule": "rtc", "sample_seed": 0})
    _answers, at, _ = _drive(policy, DELAYS[:6], 6.0)
    vae = factory.pipe.vae
    assert isinstance(vae, _Encoder)
    windows = [r for r in vae.raws if len(r) > 1]
    # The first two predictions are cold starts (0 and 16 rows of history).
    # Each later one encodes the 21 stack rows before its current frame.
    want = [list(range(n // 2 - 21, n // 2)) for n in at[2:]]
    assert windows == want and len(windows) == len(at) - 2


def test_the_junction_keeps_the_plan_until_the_answer_arrives() -> None:
    """After the first prediction the answer repeats the rows of the plan that
    run before it arrives (the delay estimate), then the model takes over."""
    policy = _factory()({"schedule": "rtc", "sample_seed": 0, "rtc_reach_max_m": "off"})
    answers, at, _ = _drive(policy, DELAYS[:4], 5.0)
    chans = rtc.channels(bimanual.SIDES)
    first, second = (rtc.rows_to_array(a, chans) for a in answers[:2])
    # The second call runs at sample at[1]: the first chunk started at its
    # answer (0.30 s), so row (t - 0.30) * 15 runs.
    t = at[1] / SAMPLE_FPS
    running = int(np.floor((t - 0.30) * FPS + 1e-9))
    p = rtc.prefix_rows(0.30, FPS, len(first) - running)
    assert p == 5
    np.testing.assert_allclose(second[:p], first[running : running + p], atol=1e-6)


def test_a_sample_after_a_later_one_answers_none_and_keeps_its_timings() -> None:
    """The workers of the client race to send: sample 1 can arrive before
    sample 0. The late sample answers None, and its call timings still anchor
    the grid."""
    policy = _factory()({"schedule": "rtc", "sample_seed": 0})
    first = _current(_obs([0.5]))
    first.update(
        {
            per.SAMPLE_INDEX: 1,
            per.OBS_TIME_NS: round(1e9 / 30),
            per.CALL_ID: 1,
            per.CALL_TIMINGS: [],
        }
    )
    assert policy.select_action(first) is not None
    late = _current(_obs([0.0]))
    late.update(
        {
            per.SAMPLE_INDEX: 0,
            per.OBS_TIME_NS: 0,
            per.CALL_ID: 0,
            per.CALL_TIMINGS: [{per.CALL_ID: 1, per.ELAPSED_SEC: 0.3}],
        }
    )
    assert policy.select_action(late) is None
    assert policy._history is not None and policy._grid is not None
    assert policy._history.newest_index == 1 and len(policy._history) == 1
    assert policy._grid.next_index == 1 + 24 + 9
    # A late reset drops the samples before it, and the next prediction restarts.
    late_reset = {
        **late,
        per.SAMPLE_INDEX: 1,
        "reset_context": True,
        per.CALL_TIMINGS: [],
    }
    assert policy.select_action(late_reset) is None
    nxt = _current(_obs([1.0]))
    nxt.update(
        {
            per.SAMPLE_INDEX: 2,
            per.OBS_TIME_NS: round(2e9 / 30),
            per.CALL_ID: 2,
            per.CALL_TIMINGS: [],
        }
    )
    restarts = policy.session.state.restart_count
    assert policy.select_action(nxt) is not None
    assert policy.session.state.restart_count == restarts + 1


def test_a_session_with_late_samples_serves_the_same_rows() -> None:
    """Samples that arrive out of order and answer None change no answer."""
    a = _factory()({"schedule": "rtc", "sample_seed": 0})
    b = _factory()({"schedule": "rtc", "sample_seed": 0})
    ref, at_a, _ = _drive(a, DELAYS[:5], 6.0)
    late = {n: n + 1 for n in (3, 40, 61, 90) if n not in at_a and n + 1 not in at_a}
    got, at_b, _ = _drive(b, DELAYS[:5], 6.0, late=late)
    assert at_a == at_b
    for x, y in zip(ref, got, strict=True):
        assert [r.keys() for r in x] == [r.keys() for r in y]
        np.testing.assert_array_equal(
            rtc.rows_to_array(x, rtc.channels(bimanual.SIDES)),
            rtc.rows_to_array(y, rtc.channels(bimanual.SIDES)),
        )


def test_a_replan_predicts_on_its_sample() -> None:
    policy = _factory()({"schedule": "rtc", "sample_seed": 0})
    _drive(policy, DELAYS[:3], 2.5)
    grid = policy._grid
    assert grid is not None and grid.next_index is not None
    n = grid.next_index - 5
    obs = _current(_obs([n / 2]))
    obs.update(
        {
            per.SAMPLE_INDEX: n,
            per.OBS_TIME_NS: round(n / SAMPLE_FPS * 1e9),
            per.CALL_ID: 999,
            per.CALL_TIMINGS: [],
            per.REPLAN_NOW: True,
        }
    )
    assert policy.select_action(obs) is not None and grid.last_index == n


def test_a_malformed_call_id_in_the_timings_keeps_the_session() -> None:
    """A sample whose ``call_timings`` holds an unhashable ``call_id`` answers
    as usual. The session goes on and predicts on its grid."""
    policy = _factory()({"schedule": "rtc", "sample_seed": 0})
    _drive(policy, DELAYS[:2], 2.0)
    grid = policy._grid
    assert grid is not None and grid.next_index is not None
    due = grid.next_index
    for n in (due - 1, due):
        obs = _current(_obs([n / 2]))
        obs.update(
            {
                per.SAMPLE_INDEX: n,
                per.OBS_TIME_NS: round(n / SAMPLE_FPS * 1e9),
                per.CALL_ID: n,
                per.CALL_TIMINGS: [{per.CALL_ID: [1], per.ELAPSED_SEC: 0.2}],
            }
        )
        rows = policy.select_action(obs)
        assert (rows is None) == (n < due)
    assert grid.last_index == due


def test_the_real_time_helpers_refuse_a_chunked_session() -> None:
    """``_sample`` and ``_join`` raise, not assert, so the guard holds under ``-O``."""
    policy = _factory()({})
    with pytest.raises(RuntimeError, match="_sample serves a real-time session"):
        policy._sample({per.SAMPLE_INDEX: 0})
    with pytest.raises(RuntimeError, match="_join serves a real-time session"):
        policy._join([], {}, 0.0)


def test_close_drops_the_history_and_the_grid() -> None:
    policy = _factory()({"schedule": "rtc", "sample_seed": 0})
    _drive(policy, DELAYS[:3], 2.5)
    assert policy._history is not None and len(policy._history) > 0
    policy.close()
    assert len(policy._history) == 0
    assert policy._grid is not None and policy._grid.next_index is None


# ---------------------------------------------------------------------------
# Protocol v2 over a real WebSocket
# ---------------------------------------------------------------------------


def _v2_server():
    pytest.importorskip("uvicorn")
    pytest.importorskip("websockets")
    import socket
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import uvicorn

    from praxis.serve.server import InferenceServer

    executor = ThreadPoolExecutor(max_workers=1)
    app = InferenceServer({"default": _factory()}, executor)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    config = uvicorn.Config(app.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    return f"127.0.0.1:{port}", srv, thread, executor


def _recv_v2(ws: object) -> dict:
    """Read one binary frame of the server and decode it."""
    from praxis.serve import wire

    data = ws.recv(timeout=10)  # ty: ignore[unresolved-attribute]
    assert isinstance(data, bytes)
    return wire.deserialise(data)


def test_protocol_v2_session_over_a_websocket() -> None:
    """A real-time session: the ready message declares protocol 2, the session
    id and max_in_flight; requests carry the envelope; samples answer None or
    rows with the timing block; the plan arrives as robot commands in the
    ``__cmd__`` envelope; end_session is acknowledged."""
    from websockets.sync.client import connect

    from praxis.serve import wire

    addr, srv, thread, executor = _v2_server()
    try:
        with connect(f"ws://{addr}/api/v1/session?schedule=rtc&sample_seed=0") as ws:
            assert _recv_v2(ws)["status"] == "loading"
            ready = _recv_v2(ws)
            assert ready["status"] == "ready" and ready["protocol_version"] == 2
            assert ready["max_in_flight"] == 8 and len(ready["session_id"]) == 32
            sid = ready["session_id"]
            first = None
            for n in range(3):
                obs = _current(_obs([n / 2]))
                obs.update(
                    {
                        per.SAMPLE_INDEX: n,
                        per.OBS_TIME_NS: n,
                        per.CALL_ID: n,
                        per.CALL_TIMINGS: [],
                    }
                )
                ws.send(wire.serialise({"session_id": sid, "observation": obs}))
                answer = _recv_v2(ws)
                assert set(answer["timing"]) == {
                    "served_ms",
                    "decode_ms",
                    "infer_ms",
                    "queued_ms",
                }
                if n == 0:
                    first = answer["result"]
                    assert len(first) == 21 and "timestamp" not in first[0]
                else:
                    assert answer["result"] is None
            assert first is not None
            # A replan with the plan as the client packs it: each command in
            # the ``__cmd__`` envelope.
            plan = [
                {
                    k: ({b"__cmd__": v} if k.startswith("robot_command") else v)
                    for k, v in row.items()
                }
                for row in first[2:]
            ]
            obs = _current(_obs([1.5]))
            obs.update(
                {
                    per.SAMPLE_INDEX: 3,
                    per.OBS_TIME_NS: 3,
                    per.CALL_ID: 3,
                    per.CALL_TIMINGS: [{per.CALL_ID: 0, per.ELAPSED_SEC: 0.2}],
                    per.REPLAN_NOW: True,
                    "action_prefix": {
                        "actions": plan,
                        "chunk_offset_sec": 2 / FPS,
                        "observation_offset_sec": 0.0,
                    },
                }
            )
            ws.send(wire.serialise({"session_id": sid, "observation": obs}))
            answer = _recv_v2(ws)
            rows = answer["result"]
            assert rows is not None
            chans = rtc.channels(bimanual.SIDES)
            # The delay estimate (0.2 s) keeps 3 rows of the plan.
            np.testing.assert_allclose(
                rtc.rows_to_array(rows, chans)[:3],
                rtc.rows_to_array(first[2:5], chans),
                atol=1e-6,
            )
            ws.send(wire.serialise({"session_id": "other", "observation": obs}))
            err = _recv_v2(ws)
            assert err["status"] == "error" and "session ID" in err["error"]
        with connect(f"ws://{addr}/api/v1/session?schedule=rtc") as ws:
            ws.recv(timeout=10)
            sid = _recv_v2(ws)["session_id"]
            ws.send(wire.serialise({"session_id": sid, "end_session": True}))
            assert _recv_v2(ws) == {
                "session_id": sid,
                "end_session": True,
            }
        with connect(f"ws://{addr}/api/v1/session") as ws:
            ws.recv(timeout=10)
            ready = _recv_v2(ws)
            assert list(ready) == ["status", "meta"]
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
        executor.shutdown()
