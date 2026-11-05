"""The bimanual (YAM) rig: config, canvas, proprio, actions and the session.

CPU only, no weights. A small encoder stands in for the VAE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from praxis import bimanual as bm
from praxis.checkpoint import CheckpointError, model_variant, validate_config
from praxis.robot_io import NormStats, pose8_to_10d, rot6d

DATA = Path(__file__).resolve().parent / "data"
YAM = json.loads((DATA / "praxis-1-yam.json").read_text())


def _with(config: dict, path: str, value: object) -> dict:
    out = copy.deepcopy(config)
    node = out
    *head, last = path.split(".")
    for key in head:
        node = node[key]
    node[last] = value
    return out


def _stats() -> NormStats:
    n = YAM["normalization"]
    return NormStats(
        action_q01=np.asarray(n["action_q01"], np.float32),
        action_q99=np.asarray(n["action_q99"], np.float32),
        proprio_mean=np.asarray(n["proprio_mean"], np.float32),
        proprio_std=np.asarray(n["proprio_std"], np.float32),
    )


def _quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_yam_config_validates() -> None:
    validate_config(YAM)
    variant = model_variant(YAM)
    assert variant.tile_size == (256, 416) and variant.antialias is True
    assert variant.pose_frame == "rig"
    # A config without the field takes the frame of its rig.
    assert (
        model_variant({k: v for k, v in YAM.items() if k != "pose_frame"}).pose_frame
        == "rig"
    )
    assert bm.canvas_size(YAM) == (256, 1248)
    assert bm.q99_dims(YAM).tolist() == bm.Q99_DIMS.tolist()


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("video.views", ["exterior", "wrist_left", "wrist_right"], "views"),
        ("action_expert.action_patch_dim", 40, "action_patch_dim"),
        ("action_expert.proprio_patch_dim", 10, "proprio_patch_dim"),
        ("action_space.dim", 10, "action_space.dim"),
        ("format_version", 1, "format_version"),
    ],
)
def test_bad_yam_config_is_refused(path: str, value: object, match: str) -> None:
    config = _with(YAM, "video.tile_bounds_rows_cols", None)
    with pytest.raises(CheckpointError, match=match):
        validate_config(_with(config, path, value))


@pytest.mark.parametrize(
    ("embodiment", "frame", "ok"),
    [
        ("bimanual", "rig", True),
        ("bimanual", "flange", False),
        ("single_arm", "rig", False),
    ],
)
def test_pose_frame_follows_the_rig(embodiment: str, frame: str, ok: bool) -> None:
    config = _with(YAM, "pose_frame", frame)
    if embodiment == "single_arm":
        config = _with(config, "embodiment", "single_arm")
        config = _with(config, "video.views", ["exterior", "wrist"])
        config = _with(config, "video.width", 832)
        config = _with(config, "video.tile_bounds_rows_cols", None)
        config = _with(config, "video.antialias", False)
        config = _with(config, "action_space.dim", None)
    if ok:
        validate_config(config)
        assert model_variant(config).pose_frame == frame
    else:
        with pytest.raises(CheckpointError, match="pose_frame"):
            validate_config(config)


def test_antialias_tile_bound_follows_the_rig() -> None:
    """A 256-tall antialiased tile is cut from a 1200x1920 view (260 px), not
    from a 720x1280 one (234 px)."""
    single = _with(YAM, "embodiment", "single_arm")
    single = _with(single, "video.views", ["exterior", "wrist"])
    single = _with(single, "video.width", 832)
    single = _with(single, "video.tile_bounds_rows_cols", None)
    with pytest.raises(CheckpointError, match="no taller than 234"):
        model_variant(single)
    model_variant(YAM)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def _views(h: int, w: int, t: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {v: rng.integers(0, 256, (t, h, w, 3), dtype=np.uint8) for v in bm.VIEWS}


def test_fit_rule_sizes() -> None:
    assert bm.fit_resize_size(1200, 1920) == (260, 416)
    resize, trimmed = bm.fit_row_sizes([(1200, 1920)] * 3)
    assert resize == [(260, 416)] * 3 and trimmed == [(256, 416)] * 3
    # The size the client sends is already the resize size.
    assert bm.fit_row_sizes([(260, 416)] * 3) == (resize, trimmed)


def test_canvas_of_native_and_restricted_views() -> None:
    clip = bm.canvas_clip(_views(1200, 1920, t=2))
    assert clip.shape == (1, 3, 2, 256, 1248) and clip.dtype == torch.float32
    small = _views(260, 416)
    clip = bm.canvas_clip(small)
    # A 260x416 view is not resized: the tile is its centre trim (2 rows each
    # side), the views side by side.
    for i, v in enumerate(bm.VIEWS):
        want = torch.from_numpy(small[v][0, 2:258]).permute(2, 0, 1).float()
        assert torch.equal(clip[0, :, 0, :, 416 * i : 416 * (i + 1)], want)


def test_canvas_refuses_views_at_another_aspect() -> None:
    with pytest.raises(ValueError, match="native aspect"):
        bm.canvas_clip(_views(234, 416))


def test_views_must_be_uint8() -> None:
    obs = {bm.VIEW_KEY_CANDIDATES[k][0]: v for k, v in _views(260, 416).items()}
    obs["image.wrist_left"] = obs["image.wrist_left"].astype(np.float32) / 255
    with pytest.raises(ValueError, match="uint8"):
        bm.view_stacks(obs)


def test_serve_config_documents_a_caption_prompt() -> None:
    """The checkpoint was trained on window captions; the serve config and the
    README show the same caption-form prompt to send."""
    root = DATA.parents[1]
    example = (
        '"Pick up the pen from the center of the table, then place the pen in the bag"'
    )
    serve = " ".join(
        (root / "configs" / "serve_yam.yaml").read_text().replace("#", " ").split()
    )
    readme = " ".join((root / "README.md").read_text().split())
    assert example in serve and example in readme
    assert "the bag." not in serve


def test_canvas_needs_equal_frame_counts() -> None:
    views = _views(260, 416, t=2)
    views["wrist_right"] = views["wrist_right"][:1]
    with pytest.raises(ValueError, match="frame counts"):
        bm.canvas_clip(views)


# ---------------------------------------------------------------------------
# Poses, proprio and actions
# ---------------------------------------------------------------------------


def _pose8(xyz: list[float], quat: np.ndarray, grip: float) -> np.ndarray:
    return np.concatenate([xyz, quat, [grip]]).astype(np.float32)


LEFT = _pose8([0.4, 0.2, 0.3], _quat(np.array([0.0, 1.0, 0.2]), 0.7), 0.25)
RIGHT = _pose8([0.5, -0.2, 0.35], _quat(np.array([1.0, 0.0, 0.3]), -0.4), 1.2)


def _obs(t: int = 1) -> dict:
    obs: dict = {}
    for side, pose in (("left", LEFT), ("right", RIGHT)):
        obs[f"robot_state.{side}.ee_pose"] = np.tile(pose[:7], (t, 1))
        obs[f"grip.{side}"] = np.full(t, pose[7], np.float32)
    for k, v in _views(260, 416, t=t).items():
        obs[bm.VIEW_KEY_CANDIDATES[k][0]] = v
    return obs


def test_pose16_and_proprio_per_arm() -> None:
    pose16 = bm.extract_pose16(_obs(3))
    assert np.array_equal(pose16, np.concatenate([LEFT, RIGHT]))
    p20 = bm.pose16_to_20d(pose16)
    assert np.array_equal(p20[:10], pose8_to_10d(LEFT))
    assert np.array_equal(p20[10:], pose8_to_10d(RIGHT))
    # The proprio keeps the gripper of the rig (1.2): no clip.
    assert p20[19] == pytest.approx(1.2)
    proprio, every = bm.proprio_tensors(pose16[None], _stats(), "cpu")
    assert proprio.shape == (1, 1, 20) and every.shape == (1, 1, 20)


def test_pose16_history_needs_real_rows() -> None:
    obs = _obs(25)
    assert bm.pose16_history(obs, 24).shape == (24, 16)
    obs["grip.left"] = obs["grip.left"][-1:]
    assert bm.pose16_history(obs, 24) is None


def test_bad_quaternion_is_refused() -> None:
    obs = _obs()
    obs["robot_state.left.ee_pose"] = obs["robot_state.left.ee_pose"] * 2
    with pytest.raises(ValueError, match="quaternion"):
        bm.extract_pose16(obs)


def test_rel20_to_wire_recovers_each_arm_target() -> None:
    """Targets made relative to each arm's anchor come back on the wire."""
    pose16 = np.concatenate([LEFT, RIGHT])
    target = np.concatenate(
        [
            _pose8([0.45, 0.1, 0.32], _quat(np.array([0.3, 1.0, 0.0]), 0.9), 0.8),
            _pose8([0.55, -0.3, 0.3], _quat(np.array([1.0, 0.2, 0.0]), -0.1), 0.0),
        ]
    )
    rel = bm.pose16_to_20d(target[None], pose16[None])
    wire = bm.rel20_to_wire(rel, pose16, dt=0.1)
    arrays = bm.wire_to_arrays(wire)
    for i, side in enumerate(bm.SIDES):
        poses, grips = arrays[side]
        t8 = target[8 * i : 8 * i + 8]
        assert np.allclose(poses[0, :3], t8[:3], atol=1e-6)
        r = poses[0, 3:].reshape(3, 3).astype(np.float64)
        want = pose8_to_10d(t8)[3:9]
        assert np.allclose(rot6d(r), want, atol=1e-5)
        assert grips[0] == pytest.approx(t8[7])
    assert set(wire[0]) == {
        "robot_command.left",
        "robot_command.right",
        "target_grip.left",
        "target_grip.right",
        "timestamp",
    }


def test_decode_actions_clips_and_unpacks() -> None:
    stats = _stats()
    tokens = torch.full((1, 6, 80), 3.0)  # far outside [-1, 1]
    rows, rel = bm.decode_actions(tokens, stats, depth=0)
    assert rows.shape == (21, 20) and rows.max() == 3.0
    q99 = stats.action_q99
    assert np.allclose(rel[:, [0, 1, 2, 10, 11, 12]], q99[[0, 1, 2, 10, 11, 12]])
    assert np.all(rel[:, bm.GRIP_DIMS] <= 1.0)
    rows, _rel = bm.decode_actions(tokens, stats, depth=1)
    assert rows.shape == (24, 20)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def _encode(clip: torch.Tensor) -> torch.Tensor:
    """A causal stand-in for the VAE: 1 + (T - 1) // 4 latent frames."""
    t = clip.shape[2]
    n = 1 + (t - 1) // 4
    return torch.zeros(1, 48, n, clip.shape[3] // 16, clip.shape[4] // 16)


def _session(**kw: object) -> bm.BimanualServeSession:
    return bm.BimanualServeSession(
        encode=_encode,
        stats=_stats(),
        device="cpu",
        max_context_chunks=1,
        dims=bm.q99_dims(YAM),
        **kw,  # ty: ignore[invalid-argument-type]
    )


def test_session_builds_context_from_the_executed_frames() -> None:
    sess = _session()
    first = sess.prepare(_obs(25))
    assert first.depth == 0 and first.proprio_all.shape == (1, 1, 20)
    assert first.anchor.shape == (1, 48, 1, 16, 78)
    wire = sess.finish(first, torch.zeros(1, 6, 80))
    assert len(wire) == 21
    second = sess.prepare(_obs(25))
    assert second.depth == 1 and len(second.cleans) == 1
    assert second.proprio_all.shape == (1, 2, 20)
    assert sess.finish(second, torch.zeros(1, 6, 80)).__len__() == 24


def test_session_restarts_on_a_short_stack() -> None:
    sess = _session()
    sess.finish(sess.prepare(_obs(25)), torch.zeros(1, 6, 80))
    again = sess.prepare(_obs(3))
    assert again.depth == 0 and sess.restart_count == 1


def test_session_takes_a_prompt_override() -> None:
    sess = _session(prompt="old")
    inputs = sess.prepare({**_obs(), "prompt": "  Pick up the pen.  "})
    assert inputs.prompt == "Pick up the pen." and sess.prompt == "Pick up the pen."


def test_session_refuses_a_control_mode() -> None:
    with pytest.raises(ValueError, match="control_mode"):
        _session(control_mode="droid_impedance")


def test_warmup_obs_runs_through_the_session() -> None:
    sess = _session()
    for step in range(2):
        inputs = sess.prepare(bm.warmup_obs(step))
        sess.finish(inputs, torch.zeros(1, 6, 80))
    assert inputs.depth == 1
