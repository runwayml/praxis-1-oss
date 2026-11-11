"""Unit tests for the single-arm robot input and output (CPU only, no weights)."""

import numpy as np
import pytest
import torch

from praxis import single_arm as sa
from praxis.robot_io import (
    NormStats,
    pose8_to_10d,
    proprio_normalize,
    q99_denormalize,
    unit_quat_wxyz_to_matrix,
)

CONFIG = {
    "embodiment": "single_arm",
    "video": {
        "height": 256,
        "width": 832,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "tile_height": 256,
        "tile_width": 416,
    },
    "normalization": {"q99_normed_dims": [0, 1, 2, 9]},
}
# A small tile for the session tests.
TILE = (32, 48)
# The frame of an observation without a descriptor: the FR3 rig.
RIG = sa.RIG_FRAMES[""]


def _stats(dim: int = 10) -> NormStats:
    rng = np.random.default_rng(3)
    return NormStats(
        action_q01=np.full(dim, -0.2, dtype=np.float32),
        action_q99=np.full(dim, 0.3, dtype=np.float32),
        proprio_mean=rng.normal(size=dim).astype(np.float32),
        proprio_std=rng.uniform(0.1, 1.0, size=dim).astype(np.float32),
    )


def _pose8(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xyz = rng.normal(size=(n, 3)) * 0.2
    return np.concatenate([xyz, q, rng.uniform(size=(n, 1))], axis=1).astype(np.float32)


def _obs(frames: dict, pose8: np.ndarray, lo: int, hi: int) -> dict:
    return {
        "image.exterior": frames["exterior"][lo:hi],
        "image.wrist": frames["wrist"][lo:hi],
        "robot_state.ee_pose": pose8[lo:hi, :7],
        "grip": pose8[lo:hi, 7:8],
    }


def _encode(clip: torch.Tensor) -> torch.Tensor:
    """A stand-in encoder: one latent per 4 raws after the first raw."""
    t = clip.shape[2]
    idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
    return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 32, 32))


def test_config_helpers() -> None:
    assert sa.canvas_size(CONFIG) == (256, 832)
    assert sa.q99_dims(CONFIG).tolist() == [0, 1, 2, 9]
    bad = {"video": {"views": ["top_left", "wrist_left", "wrist_right"]}}
    with pytest.raises(ValueError):
        sa.canvas_size(bad)


def test_control_modes() -> None:
    assert sa.control_mode_payload("none") is None
    payload = sa.control_mode_payload("droid_impedance")
    assert payload == sa.DROID_IMPEDANCE
    assert payload is not sa.DROID_IMPEDANCE
    with pytest.raises(ValueError):
        sa.control_mode_payload("bogus")


def test_extract_pose8_takes_the_last_row() -> None:
    rng = np.random.default_rng(0)
    p = _pose8(rng, 25)
    want = p[-1]
    for grip in (p[:, 7], p[:, 7:8]):
        obs = {"robot_state.ee_pose": p[:, :7], "grip": grip}
        assert np.array_equal(sa.extract_pose8(obs), want)
    single = {"ee_pose": p[-1, :7], "gripper": np.float32(p[-1, 7])}
    assert np.array_equal(sa.extract_pose8(single), want)


def test_extract_pose8_checks_the_pose_before_the_grip() -> None:
    obs = {"robot_state.ee_pose": np.zeros(3, np.float32)}
    with pytest.raises(ValueError, match=r"ee_pose must be \(>=7,\), got \(3,\)"):
        sa.extract_pose8(obs)


def test_pose8_history_rows() -> None:
    rng = np.random.default_rng(1)
    p = _pose8(rng, 25)
    obs = {"robot_state.ee_pose": p[:, :7], "grip": p[:, 7:8]}
    h24, h21 = sa.pose8_history(obs, 24), sa.pose8_history(obs, 21)
    assert h24 is not None and np.array_equal(h24, p[:24])
    assert h21 is not None and np.array_equal(h21, p[3:24])
    assert sa.pose8_history(obs, 25) is None
    assert sa.pose8_history({}, 21) is None


def test_proprio_tensors() -> None:
    rng = np.random.default_rng(2)
    stats = _stats()
    poses = _pose8(rng, 3)
    last, every = sa.proprio_tensors(poses, stats, "cpu")
    want = proprio_normalize(pose8_to_10d(poses), stats.proprio_mean, stats.proprio_std)
    assert last.shape == (1, 1, 10) and every.shape == (1, 3, 10)
    assert torch.equal(every[0], torch.from_numpy(want))
    assert torch.equal(last[0, 0], every[0, -1])


def test_relative_pose_then_wire_gives_absolute_pose() -> None:
    """A pose relative to the anchor in the model frame maps back to the rig pose."""
    rng = np.random.default_rng(4)
    anchor, target = _pose8(rng, 2)
    anchor_m, target_m = sa.pose8_to_model(anchor), sa.pose8_to_model(target)
    rel = pose8_to_10d(target_m[None], anchor8=anchor_m[None])
    mode = sa.control_mode_payload("droid_impedance")
    wire = sa.rel10_to_wire(rel, anchor_m, mode=mode)
    pose = np.asarray(wire[0]["robot_command"]["pose"], dtype=np.float64)
    assert np.allclose(pose[:3], target[:3], atol=1e-5)
    r_target = unit_quat_wxyz_to_matrix(target[3:7])
    assert np.allclose(pose[3:].reshape(3, 3), r_target, atol=1e-4)
    assert wire[0]["target_grip"] == float(target[7])
    assert wire[0]["robot_command"]["type"] == "cartesian_pos"
    assert wire[0]["robot_command"]["mode"] == sa.DROID_IMPEDANCE
    assert wire[0]["timestamp"] == 0.0


def test_actions_to_wire_rows_and_horizon() -> None:
    stats = _stats()
    tokens = torch.zeros(1, 6, 40)
    pose = np.array([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    assert len(sa.actions_to_wire(tokens, pose, stats, depth=0)) == 21
    wire = sa.actions_to_wire(tokens, pose, stats, depth=2)
    assert len(wire) == 24
    assert "mode" not in wire[0]["robot_command"]
    assert wire[1]["timestamp"] == pytest.approx(1.0 / 15.0)
    rows, rel = sa.decode_actions(tokens, stats, depth=2)
    assert rows.shape == rel.shape == (24, 10)
    assert np.allclose(rel[:, [0, 1, 2, 9]], 0.05)
    assert np.array_equal(rel[:, 3:9], rows[:, 3:9])


def _session(
    control_mode: str = "none", exec_horizon: int | None = None
) -> sa.SingleArmServeSession:
    return sa.SingleArmServeSession(
        encode=_encode,
        stats=_stats(),
        device="cpu",
        prompt="Move the cloth",
        control_mode=control_mode,
        exec_horizon=exec_horizon,
        tile_size=TILE,
    )


def test_session_depth_ramp_and_window_slide() -> None:
    rng = np.random.default_rng(5)
    n = 142
    frames = {
        v: rng.integers(0, 256, size=(n, 32, 48, 3), dtype=np.uint8)
        for v in ("exterior", "wrist")
    }
    p = _pose8(rng, n)
    sess = _session(control_mode="droid_impedance", exec_horizon=24)
    depths, rows = [], []
    for k, f in enumerate((0, 21, 45, 69, 93, 117)):
        lo = f if k == 0 else max(0, f - 24)
        inputs = sess.prepare(_obs(frames, p, lo, f + 1))
        assert inputs.proprio_all.shape == (1, inputs.depth + 1, 10)
        assert inputs.boundary_poses.shape == (inputs.depth + 1, 8)
        assert np.array_equal(inputs.boundary_poses[-1], sa.pose8_to_model(p[f], RIG))
        assert inputs.chunk_index == k
        assert inputs.prompt == "Move the cloth"
        depths.append(inputs.depth)
        rows.append(len(sess.finish(inputs, torch.zeros(1, 6, 40))))
    assert depths == [0, 1, 2, 3, 3, 3]
    assert rows == [21, 24, 24, 24, 24, 24]
    # After two slides the window starts at raw 48: boundaries 48, 69, 93, 117.
    want = sa.pose8_to_model(p[[48, 69, 93, 117]], RIG)
    assert np.array_equal(inputs.boundary_poses, want)
    assert sess.restart_count == 0


def test_session_short_stack_restarts() -> None:
    rng = np.random.default_rng(6)
    frames = {
        v: rng.integers(0, 256, size=(50, 32, 48, 3), dtype=np.uint8)
        for v in ("exterior", "wrist")
    }
    p = _pose8(rng, 50)
    sess = _session()
    sess.finish(sess.prepare(_obs(frames, p, 0, 1)), torch.zeros(1, 6, 40))
    inputs = sess.prepare(_obs(frames, p, 15, 22))
    assert inputs.depth == 0 and sess.restart_count == 1


def test_short_exec_horizon_serves_cold_starts() -> None:
    rng = np.random.default_rng(7)
    frames = {
        v: rng.integers(0, 256, size=(40, 32, 48, 3), dtype=np.uint8)
        for v in ("exterior", "wrist")
    }
    p = _pose8(rng, 40)
    sess = _session(exec_horizon=10)
    assert sess.stateless
    for k, f in enumerate((0, 10, 20, 30)):
        inputs = sess.prepare(_obs(frames, p, max(0, f - 24), f + 1))
        assert inputs.depth == 0
        assert len(sess.finish(inputs, torch.zeros(1, 6, 40))) == 10
        assert inputs.chunk_index == k


@pytest.mark.parametrize("dedup", [False, True])
def test_lazy_current_clip_preserves_inputs_and_skips_unused_builds(monkeypatch, dedup):
    rng = np.random.default_rng(31)
    frames = {
        view: rng.integers(0, 256, (118, 32, 48, 3), dtype=np.uint8)
        for view in ("exterior", "wrist")
    }
    poses = _pose8(rng, 118)
    default = sa.LAZY_CURRENT_CLIP
    original = sa.current_clip
    built = []

    def current(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sa, "current_clip", current)
    runs, counts = [], []
    for lazy in (False, default):
        monkeypatch.setattr(sa, "LAZY_CURRENT_CLIP", lazy)
        sess = _session(exec_horizon=24)
        sess.dedup_encodes = dedup
        built.clear()
        outputs = []
        for k, frame in enumerate((0, 21, 45, 69, 93, 117)):
            lo = frame if k == 0 else max(0, frame - 24)
            inputs = sess.prepare(_obs(frames, poses, lo, frame + 1))
            commands = sess.finish(inputs, torch.zeros(1, 6, 40))
            outputs.append((inputs, sa.wire_to_arrays(commands)))
        counts.append(len(built))
        runs.append(outputs)
    assert counts == [6, 1 if dedup else 6]
    for (actual, wire), (expected, reference_wire) in zip(*runs, strict=True):
        for field in ("anchor", "proprio", "proprio_all"):
            a, b = getattr(actual, field), getattr(expected, field)
            assert torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
        for a, b in zip(actual.cleans, expected.cleans, strict=True):
            assert torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
        for field in ("pose8_now", "boundary_poses"):
            assert (
                getattr(actual, field).tobytes() == getattr(expected, field).tobytes()
            )
        for field in ("depth", "chunk_index", "prompt"):
            assert getattr(actual, field) == getattr(expected, field)
        assert all(
            a.tobytes() == b.tobytes()
            for a, b in zip(wire, reference_wire, strict=True)
        )


@pytest.mark.parametrize("fallback", ["reset", "short_stack", "stateless", "dedup_off"])
def test_lazy_current_clip_builds_frames_when_encoding_is_required(
    monkeypatch, fallback
):
    rng = np.random.default_rng(32)
    frames = {
        view: rng.integers(0, 256, (22, 32, 48, 3), dtype=np.uint8)
        for view in ("exterior", "wrist")
    }
    poses = _pose8(rng, 22)
    original = sa.current_clip
    built = []

    def current(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sa, "current_clip", current)
    sess = _session(exec_horizon=10 if fallback == "stateless" else 24)
    sess.dedup_encodes = fallback != "dedup_off"
    sess.finish(sess.prepare(_obs(frames, poses, 0, 1)), torch.zeros(1, 6, 40))
    obs = _obs(frames, poses, 18 if fallback == "short_stack" else 0, 22)
    if fallback == "reset":
        obs["reset_context"] = True
    inputs = sess.prepare(obs)
    assert len(built) == 2
    if fallback != "dedup_off":
        assert inputs.depth == 0


# ---------------------------------------------------------------------------
# Clamp of the decoded actions
# ---------------------------------------------------------------------------


def _clamp_stats() -> NormStats:
    """Stats like the stats of the DROID checkpoint: the grip range is (0, 1)."""
    q01 = np.linspace(-0.2, -0.15, 10).astype(np.float32)
    q99 = np.linspace(0.18, 0.2, 10).astype(np.float32)
    q01[9], q99[9] = 0.0, 1.0
    return NormStats(
        action_q01=q01,
        action_q99=q99,
        proprio_mean=np.zeros(10, np.float32),
        proprio_std=np.ones(10, np.float32),
    )


def _rows10(rng: np.random.Generator) -> np.ndarray:
    rows = rng.uniform(-1.0, 1.0, size=(24, 10)).astype(np.float32)
    rows[:, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return rows


@pytest.mark.parametrize("value", [1.5, 10.0])
def test_decode_clamps_out_of_range_actions(value: float) -> None:
    """Normalized values outside [-1, 1] give commands in [q01, q99]."""
    stats = _clamp_stats()
    dims = sa.q99_dims(CONFIG)
    rows = _rows10(np.random.default_rng(8))
    rows[0::2, dims], rows[1::2, dims] = value, -value
    tokens = torch.from_numpy(rows.reshape(1, 6, 40).copy())
    norm, rel = sa.decode_actions(tokens, stats, depth=1, dims=dims)
    assert np.array_equal(norm, rows)  # the normalized rows have no clip
    edge = rows.copy()
    edge[0::2, dims], edge[1::2, dims] = 1.0, -1.0
    want = q99_denormalize(edge, stats.action_q01, stats.action_q99, dims)
    assert rel.tobytes() == want.tobytes()
    anchor = np.array([0.5, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.4], np.float32)
    wire = sa.actions_to_wire(tokens, anchor, stats, depth=1, dims=dims)
    poses, grips = sa.wire_to_arrays(wire)
    xyz = poses[:, :3].astype(np.float64) - anchor[:3]
    assert np.all(xyz >= stats.action_q01[:3] - 1e-6)
    assert np.all(xyz <= stats.action_q99[:3] + 1e-6)
    assert grips[0::2].tolist() == [1.0] * 12 and grips[1::2].tolist() == [0.0] * 12


def test_decode_clamps_the_grip_to_its_range() -> None:
    """A grip stat outside [0, 1] gives a grip command in [0, 1]."""
    stats = _clamp_stats()
    stats.action_q01[9], stats.action_q99[9] = -0.5, 1.5
    rows = _rows10(np.random.default_rng(9))
    rows[:12, 9], rows[12:, 9] = -1.0, 1.0
    _norm, rel = sa.decode_actions(torch.from_numpy(rows.reshape(1, 6, 40)), stats, 1)
    assert np.all(rel[:12, 9] == 0.0) and np.all(rel[12:, 9] == 1.0)


def test_decode_keeps_in_range_actions_bit_equal() -> None:
    """For rows in [-1, 1], the clamp changes no bit of the commands."""
    stats = _clamp_stats()
    rng = np.random.default_rng(10)
    mode = sa.control_mode_payload("droid_impedance")
    for depth in (0, 1):
        rows = _rows10(rng)
        rows[:3, sa.Q99_DIMS] = [[-1.0] * 4, [1.0] * 4, [-0.0] * 4]
        tokens = torch.from_numpy(rows.reshape(1, 6, 40).copy())
        norm, rel = sa.decode_actions(tokens, stats, depth)
        old = q99_denormalize(norm, stats.action_q01, stats.action_q99, sa.Q99_DIMS)
        assert rel.dtype == old.dtype and rel.tobytes() == old.tobytes()
        anchor = _pose8(rng, 1)[0]
        new_wire = sa.actions_to_wire(tokens, anchor, stats, depth, mode=mode)
        old_wire = sa.rel10_to_wire(old, anchor, mode=mode)
        for a, b in zip(new_wire, old_wire, strict=True):
            assert a["robot_command"]["pose"].tobytes() == (
                b["robot_command"]["pose"].tobytes()
            )
            assert a["target_grip"] == b["target_grip"]
            assert a["timestamp"] == b["timestamp"]


def _q(stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    return stats.action_q01, stats.action_q99


def test_decode_clips_only_the_q99_dims_of_the_config() -> None:
    """The clip to [-1, 1] uses the q99 dims of the config, as the inverse map does.

    A dim without the q99 map gets no clip. The gripper always gets the clip
    to [0, 1].
    """
    stats = _clamp_stats()
    dims = sa.q99_dims({"normalization": {"q99_normed_dims": [0, 1, 9]}})
    rows = _rows10(np.random.default_rng(11))
    rows[:, [0, 1, 2, 9]] = 1.5
    tokens = torch.from_numpy(rows.reshape(1, 6, 40).copy())
    _norm, rel = sa.decode_actions(tokens, stats, depth=1, dims=dims)
    assert np.all(rel[:, 2] == np.float32(1.5))
    edge = q99_denormalize(np.ones((1, 10), np.float32), *_q(stats), dims)
    assert np.all(rel[:, [0, 1]] == edge[0, [0, 1]])
    assert np.all(rel[:, 9] == 1.0)


def _views(seed: int, h: int = 288, w: int = 512) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ext = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    return ext, wrist


def test_tile_clip_antialias_off_is_the_plain_bilinear_resize() -> None:
    """``antialias=False`` (the default) is byte-identical to the resize before the flag."""
    ext, wrist = _views(0)
    clip = sa.tile_clip(ext, wrist, 224, 416, 0.03)
    assert torch.equal(clip, sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=False))
    fr = torch.from_numpy(ext).permute(0, 3, 1, 2)
    want = (
        torch.nn.functional.interpolate(
            sa.aspect_crop(fr, 224, 416, 0.03).float(),
            size=(224, 416),
            mode="bilinear",
            align_corners=False,
        )
        .round()
        .clamp(0, 255)
    )
    assert torch.equal(clip[0, :, :, :, :416].permute(1, 0, 2, 3), want)


def test_tile_clip_antialias_on_is_the_fit_rule() -> None:
    """``antialias=True`` crops to the native aspect, resizes to 234x416, trims to 224x416."""
    ext, wrist = _views(1)
    on = sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=True)
    off = sa.tile_clip(ext, wrist, 224, 416, 0.03, antialias=False)
    assert not torch.equal(on, off)
    assert sa.fit_resize_size(416) == (234, 416)
    fr = torch.from_numpy(wrist).permute(0, 3, 1, 2)
    fit = (
        torch.nn.functional.interpolate(
            sa.aspect_crop(fr, 720, 1280, 0.03).float(),
            size=(234, 416),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
    )
    want = fit[..., 5:229, :]
    assert torch.equal(on[0, :, :, :, 416:].permute(1, 0, 2, 3), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize("size", [(288, 512), (234, 416)])
def test_tile_u8_pinned_is_the_stock_clip(monkeypatch, antialias, size) -> None:
    """The pinned uint8 canvas gives the stock clip (values, dtype, strides), also on reuse."""
    rng = np.random.default_rng(4)
    ext = rng.integers(0, 256, (3, *size, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (3, *size, 3), dtype=np.uint8)
    monkeypatch.setattr(sa, "TILE_U8_PINNED", False)
    want = sa.tile_clip(ext, wrist, 224, 416, 0.03, "cuda", antialias)
    monkeypatch.setattr(sa, "TILE_U8_PINNED", True)
    for _ in range(2):
        got = sa.tile_clip(ext, wrist, 224, 416, 0.03, "cuda", antialias)
        assert got.dtype == want.dtype and got.stride() == want.stride()
        assert torch.equal(got, want)


def test_tile_clip_antialias_on_does_not_resize_a_fit_size_input() -> None:
    """A 416x234 view (what the client sends) is only trimmed to 416x224."""
    rng = np.random.default_rng(3)
    ext = rng.integers(0, 256, (2, 234, 416, 3), dtype=np.uint8)
    clip = sa.tile_clip(ext, ext, 224, 416, 0.03, antialias=True)
    want = torch.from_numpy(ext[:, 5:229]).permute(0, 3, 1, 2).float()
    assert torch.equal(clip[0, :, :, :, :416].permute(1, 0, 2, 3), want)
