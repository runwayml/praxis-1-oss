"""The so101 rig (praxis/so101.py) without a model (CPU only).

The golden values come from the original serve and training code of the
SO-101 fine-tune (its unit conversions and the fit-416 canvas of its data
loader), run once on the inputs of each test. The session tests
compare the streaming session with the first praxis implementation, which
rebuilt the context from the whole window of each observation
(``_rebuild_inputs`` below): the same inputs give the same bits.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from praxis import bimanual, single_arm, so101
from praxis.checkpoint import validate_config
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline, Session
from praxis.robot_io import (
    ContiguousStreamContext,
    NormStats,
    action_rows,
    clip_normalized,
    pack_action_tokens,
    proprio_normalize,
    q99_denormalize,
    q99_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "tests" / "data" / "praxis-1-so101.json").read_text())

GRIP = so101.GripRange(3.84885, 37.2988)
STATE = np.array(
    [
        [10.0, -100.0, 95.0, 70.0, -30.0, 2.0],
        [-45.5, 12.25, 0.0, -90.0, 150.0, 40.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 60.0],
    ]
)
# The original LeRobot-to-model conversion of STATE, GripRange(3.84885, 37.2988)
TRAIN_ROWS = np.array(
    [
        [0.174532925199, -1.745329251994, 1.658062789395, 1.221730476396,
         -0.523598775598, 1.0],
        [-0.794124809657, 0.213802833369, 0.0, -1.570796326795, 2.617993877991,
         0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)  # fmt: skip
# The original model-to-LeRobot conversion of TRAIN_ROWS: the gripper clipped into [closed, open]
BACK = np.array(
    [
        [10.0, -100.0, 95.0, 70.0, -30.0, 3.84885],
        [-45.5, 12.25, 0.0, -90.0, 150.0, 37.2988],
        [0.0, 0.0, 0.0, 0.0, 0.0, 37.2988],
    ]
)
# The original rel_to_abs(REL, TRAIN_ROWS[0])
REL = np.array([[0.1, -0.2, 0.0, 0.05, 0.3, 0.9], [0, 0, 0, 0, 0, 1.3]])
ABS = np.array(
    [
        [0.274532925199, -1.945329251994, 1.658062789395, 1.271730476396,
         -0.223598775598, 0.9],
        [0.174532925199, -1.745329251994, 1.658062789395, 1.221730476396,
         -0.523598775598, 1.0],
    ]
)  # fmt: skip

STATS = NormStats(
    action_q01=np.asarray(CONFIG["normalization"]["action_q01"], np.float32),
    action_q99=np.asarray(CONFIG["normalization"]["action_q99"], np.float32),
    proprio_mean=np.asarray(CONFIG["normalization"]["proprio_mean"], np.float32),
    proprio_std=np.asarray(CONFIG["normalization"]["proprio_std"], np.float32),
)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_units_match_the_training_conversion() -> None:
    rows = so101.lerobot_to_model(STATE, GRIP)
    np.testing.assert_allclose(rows, TRAIN_ROWS, atol=1e-11)
    np.testing.assert_allclose(so101.model_to_lerobot(rows, GRIP), BACK, atol=1e-8)
    np.testing.assert_allclose(so101.rel_to_abs(REL, TRAIN_ROWS[0]), ABS, atol=1e-11)


def test_units_round_trip_inside_the_grip_range() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.uniform(-150, 150, (50, 5)), rng.uniform(GRIP.closed, GRIP.open, (50, 1))],
        axis=1,
    )
    back = so101.model_to_lerobot(so101.lerobot_to_model(x, GRIP), GRIP)
    np.testing.assert_allclose(back, x, atol=1e-9)
    anchor = so101.lerobot_to_model(x[0], GRIP)
    rows = so101.lerobot_to_model(x, GRIP)
    np.testing.assert_allclose(
        so101.rel_to_abs(so101.abs_to_rel(rows, anchor), anchor), rows, atol=1e-12
    )


def test_a_grip_range_may_read_high_when_closed() -> None:
    """``closed > open`` is a follower whose reading grows as it closes: the
    closure still runs from 0 (open) to 1 (closed)."""
    grip = so101.GripRange(closed=40.0, open=5.0)
    state = np.array(
        [[0, 0, 0, 0, 0, 40.0], [0, 0, 0, 0, 0, 5.0], [0, 0, 0, 0, 0, 22.5]]
    )
    assert so101.lerobot_to_model(state, grip)[:, 5].tolist() == [1.0, 0.0, 0.5]
    rows = np.zeros((2, 6))
    rows[:, 5] = (1.0, 0.0)
    assert so101.model_to_lerobot(rows, grip)[:, 5].tolist() == [40.0, 5.0]
    assert grip.as_dict() == {"closed": 40.0, "open": 5.0}


def test_grip_range_refuses_a_narrow_range() -> None:
    with pytest.raises(ValueError, match="must differ"):
        so101.GripRange(10.0, 10.5)
    with pytest.raises(ValueError, match="non-finite"):
        so101.GripRange(float("nan"), 10.0)


def test_state_refuses_raw_ticks() -> None:
    obs = {"state": np.full((1, 6), 2048.0)}
    with pytest.raises(ValueError, match="use_degrees"):
        so101.state_stack(obs)
    with pytest.raises(ValueError, match="missing state"):
        so101.state_stack({})
    with pytest.raises(ValueError, match=r"expected \(T, 6\)"):
        so101.state_stack({"state": np.zeros((3, 5))})


def test_views_are_one_or_two() -> None:
    assert so101.check_views(1) == 1 and so101.check_views(np.int64(2)) == 2
    for bad in (0, 3, True, "2", 1.0):
        with pytest.raises(ValueError, match="views must be 1 or 2"):
            so101.check_views(bad)
    assert so101.stack_keys(1) == ("image.exterior", "state")
    assert so101.stack_keys(2) == so101.STACK_KEYS


# ---------------------------------------------------------------------------
# Canvas and config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sizes", "want"),
    [
        ([(480, 640), (480, 640)], (288, 832)),
        ([(480, 640)], (288, 416)),
        ([(312, 416), (312, 416)], (288, 832)),
        # The pogo example of the training fit rule: 416x224 + 288x224.
        ([(720, 1280), (480, 640)], (224, 704)),
    ],
)
def test_canvas_shapes(sizes: list[tuple[int, int]], want: tuple[int, int]) -> None:
    stacks = [np.zeros((3, h, w, 3), np.uint8) for h, w in sizes]
    clip = so101.canvas_clip(stacks)
    assert tuple(clip.shape) == (1, 3, 3, *want)
    assert so101.canvas_hw(sizes) == want


def test_canvas_matches_the_training_fit_rule() -> None:
    """Checksums of the original fit-416 canvas on the same frames (288x832 uint8)."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (2, 480, 640, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (2, 360, 480, 3), dtype=np.uint8)
    clip = so101.canvas_clip([a, b])[0].permute(1, 0, 2, 3).long()
    assert tuple(clip.shape) == (2, 3, 288, 832)
    assert int(clip.sum()) == 183272457
    assert int((clip * torch.arange(clip.shape[-1])).sum() % 1000003) == 596040


def test_canvas_refuses_mixed_counts_and_small_frames() -> None:
    with pytest.raises(ValueError, match="different frame counts"):
        so101.canvas_clip(
            [np.zeros((2, 48, 64, 3), np.uint8), np.zeros((3, 48, 64, 3), np.uint8)]
        )
    with pytest.raises(ValueError, match="under 32 px"):
        so101.view_stacks({"image.exterior": np.zeros((1, 16, 64, 3), np.uint8)}, 1)
    with pytest.raises(ValueError, match="uint8"):
        so101.view_stacks({"image.exterior": np.zeros((1, 48, 64, 3), np.float32)}, 1)


def test_config_helpers_read_the_checkpoint() -> None:
    validate_config(CONFIG)
    assert so101.canvas_size(CONFIG) == (288, 832)
    assert so101.canvas_size(CONFIG, 1) == (288, 416)
    assert so101.q99_dims(CONFIG).tolist() == list(range(6))
    bad = copy.deepcopy(CONFIG)
    bad["video"]["views"] = ["exterior"]
    with pytest.raises(ValueError, match="tile canvas"):
        so101.canvas_size(bad)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class RecordingEncoder:
    """A causal stand-in of the VAE: latent 0 of frame 0, then one per 4
    frames; 16x16 average pooling. It records the clips."""

    def __init__(self) -> None:
        self.clips: list[torch.Tensor] = []

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        self.clips.append(clip)
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 16, 16))


def _obs(raws: list[int], n_views: int = 2) -> dict:
    """The observation at the executed raws ``raws`` (one entry per raw,
    current last): the raw index sets the state and the frames."""
    t = np.asarray(raws, np.float64)
    state = STATE[0] + 0.1 * t[:, None] * np.array([1, 0.5, -0.5, 0.2, 0.1, 0.0])
    state[:, 5] = 5.0 + (t % 30)
    obs: dict = {"state": state}
    for i, v in enumerate(so101.VIEWS[:n_views]):
        frames = np.zeros((len(raws), 48, 64, 3), np.uint8)
        frames[..., 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, None, :]
        frames[..., 1] = (37 * t.astype(np.int64) + 61 * i)[:, None, None] % 256
        frames[..., 2] = t.astype(np.int64)[:, None, None] % 256
        obs[so101.VIEW_KEYS[v]] = frames
    return obs


def _stack(current: int, n: int, n_views: int = 2) -> dict:
    """The stack of ``n`` entries that ends at the raw ``current``."""
    return _obs(list(range(current - n + 1, current + 1)), n_views)


def _session(
    max_context_chunks: int = 1, n_views: int = 2, **kw: object
) -> so101.So101ServeSession:
    return so101.So101ServeSession(
        encode=RecordingEncoder(),
        stats=STATS,
        device="cpu",
        max_context_chunks=max_context_chunks,
        n_views=n_views,
        grip=GRIP,
        **kw,  # ty: ignore[invalid-argument-type]
    )


def _encoder(s: so101.So101ServeSession) -> RecordingEncoder:
    enc = s.encode
    assert isinstance(enc, RecordingEncoder)
    return enc


def test_a_first_call_is_a_cold_start_on_the_current_frame() -> None:
    s = _session(1)
    obs = _stack(24, 25)
    inputs = s.prepare(obs)
    assert inputs.depth == 0 and inputs.cleans == []
    rows = so101.lerobot_to_model(obs["state"], GRIP)
    np.testing.assert_allclose(inputs.pose_now, rows[-1])
    assert len(_encoder(s).clips) == 1 and _encoder(s).clips[0].shape[2] == 1
    np.testing.assert_allclose(inputs.boundary_poses, rows[-1:], rtol=1e-6)
    want = proprio_normalize(
        rows[-1:].astype(np.float32), STATS.proprio_mean, STATS.proprio_std
    )
    np.testing.assert_array_equal(inputs.proprio_all[0].numpy(), want)
    assert tuple(inputs.proprio.shape) == (1, 1, 6)
    assert inputs.canvas_hw == (32, 128)


def test_the_executed_rows_build_the_context_and_slide_the_window() -> None:
    """After the 21-row first chunk the client sends 22 entries; the stream
    takes the 21 before the current one. After a 24-row chunk it sends 25
    and the window slides to the newest 21 raws."""
    s = _session(1)
    s.finish(s.prepare(_stack(0, 1)), torch.zeros(1, 6, 24))
    second = s.prepare(_stack(21, 22))
    assert second.depth == 1 and len(second.cleans) == 1
    enc = _encoder(s)
    window, first = enc.clips[1:]
    assert window.shape[2] == 21 and first.shape[2] == 1
    assert torch.equal(window, so101.canvas_clip(so101.view_stacks(_stack(20, 21), 2)))
    assert torch.equal(first, window[:, :, :1])
    rows = so101.lerobot_to_model(_stack(21, 22)["state"], GRIP)
    np.testing.assert_allclose(second.boundary_poses, rows[[0, -1]], rtol=1e-6)
    assert tuple(second.proprio_all.shape) == (1, 2, 6)
    s.finish(second, torch.zeros(1, 6, 24))
    third = s.prepare(_stack(45, 25))
    assert third.depth == 1 and s.stream.n_raws == 21
    window = enc.clips[-2]
    assert torch.equal(window, so101.canvas_clip(so101.view_stacks(_stack(44, 21), 2)))
    rows = so101.lerobot_to_model(_stack(45, 25)["state"], GRIP)
    np.testing.assert_allclose(third.boundary_poses, rows[[3, -1]], rtol=1e-6)


def test_a_deeper_context_fills_one_chunk_per_call() -> None:
    s = _session(3)
    depths = []
    for current, n in ((0, 1), (21, 22), (45, 25), (69, 25), (93, 25)):
        inputs = s.prepare(_stack(current, n))
        depths.append(inputs.depth)
        s.finish(inputs, torch.zeros(1, 6, 24))
    assert depths == [0, 1, 2, 3, 3]
    assert s.stream.n_raws == 69 and tuple(inputs.proprio_all.shape) == (1, 4, 6)


def test_a_short_stack_restarts_the_session() -> None:
    s = _session(1)
    s.finish(s.prepare(_stack(0, 1)), torch.zeros(1, 6, 24))
    again = s.prepare(_stack(21, 3))
    assert again.depth == 0 and s.restart_count == 1
    s.finish(again, torch.zeros(1, 6, 24))
    assert s.prepare(_stack(45, 25)).depth == 1


def test_reset_context_and_stateless_sessions_cold_start() -> None:
    s = _session(1)
    s.finish(s.prepare(_stack(0, 1)), torch.zeros(1, 6, 24))
    assert s.prepare({**_stack(21, 22), "reset_context": True}).depth == 0
    stateless = _session(1, exec_horizon=12)
    assert stateless.stateless
    stateless.finish(stateless.prepare(_stack(0, 1)), torch.zeros(1, 6, 24))
    assert stateless.prepare(_stack(12, 13)).depth == 0
    assert (
        len(stateless.finish(stateless.prepare(_stack(0, 1)), torch.zeros(1, 6, 24)))
        == 12
    )
    assert _session(0).stateless


def test_executed_rows_append_off_the_chunk_grid() -> None:
    """Real-time chunking: an empty stream takes the whole history, then each
    call appends its stride; a stride past the history restarts."""
    s = _session(1)
    s.prepare(_stack(16, 17), executed_rows=0)
    assert s.stream.n_raws == 16 and s.stream.depth == 0
    inputs = s.prepare(_stack(28, 25), executed_rows=12)
    assert s.stream.n_raws == 21 and inputs.depth == 1
    window = _encoder(s).clips[-2]
    assert torch.equal(window, so101.canvas_clip(so101.view_stacks(_stack(27, 21), 2)))
    s.prepare(_stack(80, 25), executed_rows=52)
    assert s.restart_count == 1 and s.stream.n_raws == 21


def test_one_view_session_reads_the_exterior_only() -> None:
    s = _session(1, n_views=1)
    inputs = s.prepare(_stack(0, 1, n_views=1))
    assert inputs.canvas_hw == (32, 64) and s.stack_keys == ("image.exterior", "state")
    assert s.canvas_size == (288, 416)
    with pytest.raises(KeyError, match="image.wrist"):
        _session(1).prepare(_stack(0, 1, n_views=1))
    with pytest.raises(ValueError, match="stack lengths differ"):
        _session(1).prepare({**_stack(5, 6), "state": _stack(5, 3)["state"]})


def test_the_session_checks_its_canvas_settings() -> None:
    with pytest.raises(ValueError, match="antialias"):
        _session(1, antialias=False)
    with pytest.raises(ValueError, match="fit rule gives 288x416"):
        _session(1, tile_size=(256, 416))
    with pytest.raises(ValueError, match="control_mode"):
        _session(1, control_mode="droid_impedance")
    with pytest.raises(ValueError, match="views must be 1 or 2"):
        _session(1, n_views=3)


def test_decode_matches_the_training_codec() -> None:
    """Pack known relative rows as training does, decode them back."""
    s = _session(1)
    for current, n, first in ((0, 1, True), (21, 22, False)):
        inputs = s.prepare(_stack(current, n))
        t = 21 if first else 24
        rng = np.random.default_rng(n)
        rel = np.concatenate(
            [rng.uniform(-0.5, 0.5, (t, 5)), rng.uniform(0, 1, (t, 1))], axis=1
        ).astype(np.float32)
        norm = q99_normalize(rel, STATS.action_q01, STATS.action_q99, so101.Q99_DIMS)
        tokens = torch.from_numpy(pack_action_tokens(norm, first))[None]
        before = s.chunk_index
        wire = s.finish(inputs, tokens)
        assert s.chunk_index == before + 1
        assert len(wire) == t
        targets = so101.wire_to_targets(wire)
        want = so101.model_to_lerobot(so101.rel_to_abs(rel, inputs.pose_now), GRIP)
        np.testing.assert_allclose(targets, want, atol=2e-3)
        assert [r["timestamp"] for r in wire[:3]] == [0.0, 1 / 15, 2 / 15]


def test_decode_clips_out_of_range_tokens() -> None:
    s = _session(1)
    s.finish(s.prepare(_stack(0, 1)), torch.zeros(1, 6, 24))
    inputs = s.prepare(_stack(21, 22))
    tokens = torch.full((1, 6, 24), 5.0)
    targets = so101.wire_to_targets(s.finish(inputs, tokens))
    q = np.degrees(inputs.pose_now[:5] + STATS.action_q99[:5])
    np.testing.assert_allclose(targets[:, :5], np.broadcast_to(q, (24, 5)), rtol=1e-5)
    np.testing.assert_allclose(targets[:, 5], GRIP.closed, rtol=1e-6)


def test_prompt_override() -> None:
    s = _session(1)
    obs = _stack(0, 1)
    obs["task"] = b"  Put the cube in the box.  "
    assert s.prepare(obs).prompt == "Put the cube in the box."
    assert s.prepare(_stack(21, 22)).prompt == "Put the cube in the box."
    assert _session(1).prepare(_stack(0, 1)).prompt == " "


def test_warmup_obs_runs_through_the_session() -> None:
    for views in (2, 1):
        s = _session(1, n_views=views)
        for step in range(2):
            inputs = s.prepare(so101.warmup_obs(step, views=views))
            s.finish(inputs, torch.zeros(1, 6, 24))
        assert inputs.depth == 1 and inputs.canvas_hw == (288, 416 * views)


# ---------------------------------------------------------------------------
# The same bits as the stack-rebuild implementation
# ---------------------------------------------------------------------------


def _rebuild_inputs(
    obs: dict, encode: RecordingEncoder, n_views: int, max_context_chunks: int
) -> tuple[torch.Tensor, list[torch.Tensor], np.ndarray, torch.Tensor, int]:
    """The first implementation of the so101 session: the context rebuilt from
    the whole window of each observation (its
    ``So101ServeSession.prepare``). Returns the anchor, the cleans, the
    boundary poses, the proprio and the depth."""
    state = so101.state_stack(obs)
    stacks = so101.view_stacks(obs, n_views)
    rows = so101.lerobot_to_model(state, GRIP)
    n = len(state)
    stream = ContiguousStreamContext(
        action_chunk=24, max_context_chunks=max(max_context_chunks, 1)
    )
    hist = n - 1

    def window_raws(depth: int) -> int:
        return 0 if depth <= 0 else 21 + 24 * (depth - 1)

    if max_context_chunks and hist >= window_raws(1):
        take = min(hist, window_raws(max_context_chunks))
        clip = so101.canvas_clip([s[n - 1 - take : n - 1] for s in stacks])
        stream.append_executed(clip, rows[n - 1 - take : n - 1], strict=False)
    d = stream.depth
    cleans = stream.cleans(encode)
    af = stream.anchor_frame()
    if af is None:
        af = so101.canvas_clip([s[-1:] for s in stacks])
    anchor = encode(af)
    bp = stream.boundary_poses() if d else None
    poses = rows[-1:] if bp is None else np.concatenate([bp, rows[-1:]], axis=0)
    p = proprio_normalize(
        poses.astype(np.float32), STATS.proprio_mean, STATS.proprio_std
    )
    every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, len(p), -1)
    return anchor, cleans, poses.astype(np.float32), every.float(), d


@pytest.mark.parametrize(("max_context_chunks", "n_views"), [(1, 2), (1, 1), (3, 2)])
def test_chunked_answers_equal_the_stack_rebuild_implementation(
    max_context_chunks: int, n_views: int
) -> None:
    """On the chunk grid the streaming session and the stack-rebuild
    implementation see the same observations and give the same model inputs,
    bit for bit, and the same commands for the same tokens."""
    s = _session(max_context_chunks, n_views)
    # The client keeps the executed chunk (24 rows) and the current entry, or
    # the whole window of a deeper context, as the stack-rebuild client did.
    window = max(24, 21 + 24 * (max_context_chunks - 1)) + 1
    current, n = 0, 1
    depths = []
    for k in range(6):
        obs = _stack(current, n, n_views)
        inputs = s.prepare(obs)
        anchor, cleans, poses, proprio, depth = _rebuild_inputs(
            obs, RecordingEncoder(), n_views, max_context_chunks
        )
        assert inputs.depth == depth and len(inputs.cleans) == len(cleans)
        assert torch.equal(inputs.anchor, anchor)
        assert all(
            torch.equal(a, b) for a, b in zip(inputs.cleans, cleans, strict=True)
        )
        np.testing.assert_array_equal(inputs.boundary_poses, poses)
        assert torch.equal(inputs.proprio_all, proprio)
        rng = np.random.default_rng(k)
        tokens = torch.from_numpy(rng.uniform(-1.5, 1.5, (1, 6, 24)).astype(np.float32))
        wire = s.finish(inputs, tokens)
        rows = action_rows(tokens, depth, 4, 6)
        rel = q99_denormalize(
            clip_normalized(rows, so101.Q99_DIMS),
            STATS.action_q01,
            STATS.action_q99,
            dims=so101.Q99_DIMS,
        )
        rel[:, 5] = np.clip(rel[:, 5], 0.0, 1.0)
        pose_now = so101.lerobot_to_model(obs["state"], GRIP)[-1]
        want = so101.model_to_lerobot(so101.rel_to_abs(rel, pose_now), GRIP)
        np.testing.assert_array_equal(
            so101.wire_to_targets(wire), want.astype(np.float32)
        )
        depths.append(depth)
        current += len(wire)
        n = min(n + len(wire), window)
    assert depths == [0, *[min(k, max_context_chunks) for k in range(1, 6)]]


# ---------------------------------------------------------------------------
# Rig selection by embodiment
# ---------------------------------------------------------------------------


class _Pipe:
    """The attributes of a pipeline that :class:`Session` reads."""

    def __init__(self, config: dict) -> None:
        self.policy = type("P", (), {"config": config})()
        self.vae = type("V", (), {"encode": staticmethod(RecordingEncoder())})()
        self.stats = STATS
        self.device = torch.device("cpu")
        self.sampler = SamplerSettings()


@pytest.mark.parametrize(
    ("config", "cls", "mode"),
    [
        ("praxis-1-so101.json", so101.So101ServeSession, "none"),
        ("praxis-1-yam.json", bimanual.BimanualServeSession, "none"),
        ("praxis-1-droid.json", single_arm.SingleArmServeSession, "droid_impedance"),
    ],
)
def test_session_picks_the_rig_of_the_config(config: str, cls: type, mode: str) -> None:
    cfg = json.loads((ROOT / "tests" / "data" / config).read_text())
    validate_config(cfg)
    session = Session(_Pipe(cfg))  # ty: ignore[invalid-argument-type]
    assert type(session.state) is cls
    assert session.state.control_mode == mode


def test_rig_options_reach_the_session_of_their_rig_only() -> None:
    pipe: PraxisPipeline = _Pipe(CONFIG)  # ty: ignore[invalid-assignment]
    session = PraxisPipeline.session(pipe, rig_options={"n_views": 1, "grip": GRIP})
    state = session.state
    assert isinstance(state, so101.So101ServeSession)
    assert state.n_views == 1 and state.grip == GRIP
    assert state.tile_size == (288, 416)
    yam: PraxisPipeline = _Pipe(  # ty: ignore[invalid-assignment]
        json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
    )
    with pytest.raises(
        ValueError, match=r"\['n_views'\] are not settings of the bimanual"
    ):
        Session(yam, rig_options={"n_views": 1})
    with pytest.raises(ValueError, match=r"\['prompt'\] are not settings of the so101"):
        Session(pipe, rig_options={"prompt": "x"})
