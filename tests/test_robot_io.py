"""Unit tests for the robot input and output helpers (CPU only)."""

import numpy as np
import torch

from praxis.robot_io import (
    NormStats,
    action_rows,
    clip_grip,
    clip_normalized,
    pack_action_tokens,
    pose8_to_10d,
    q99_denormalize,
    q99_normalize,
    rot6d_to_matrix,
    unit_quat_wxyz_to_matrix,
)

# The xyz and grip dims of a 10-D row.
Q99_DIMS = np.array([0, 1, 2, 9])


def test_rotation_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = unit_quat_wxyz_to_matrix(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        r6 = np.concatenate([r[:, 0], r[:, 1]])
        assert np.allclose(rot6d_to_matrix(r6), r, atol=1e-6)


def test_relative_pose_is_relative_to_the_anchor() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(2, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pose = np.concatenate([rng.normal(size=(2, 3)), q, [[0.2], [0.7]]], axis=1)
    rel = pose8_to_10d(pose[1:], anchor8=pose[:1])[0]
    assert np.allclose(rel[:3], pose[1, :3] - pose[0, :3], atol=1e-6)
    r_rel = rot6d_to_matrix(rel[3:9])
    r_abs = r_rel @ unit_quat_wxyz_to_matrix(pose[0, 3:7])
    assert np.allclose(r_abs, unit_quat_wxyz_to_matrix(pose[1, 3:7]), atol=1e-5)
    assert rel[9] == np.float32(0.7)


def test_action_rows_counts() -> None:
    tokens = torch.arange(6 * 40, dtype=torch.float32).reshape(1, 6, 40)
    assert action_rows(tokens, depth=0, frames_per_token=4, action_dim=10).shape == (
        21,
        10,
    )
    assert action_rows(tokens, depth=2, frames_per_token=4, action_dim=10).shape == (
        24,
        10,
    )


def test_pack_action_tokens_inverts_action_rows() -> None:
    rows = np.random.default_rng(2).normal(size=(24, 10)).astype(np.float32)
    later = torch.from_numpy(pack_action_tokens(rows, first_chunk=False))[None]
    assert np.array_equal(action_rows(later, 1, 4, 10), rows)
    first = torch.from_numpy(pack_action_tokens(rows[:21], first_chunk=True))[None]
    assert first.shape == (1, 6, 40)
    assert np.array_equal(action_rows(first, 0, 4, 10), rows[:21])


def test_q99_maps_minus_one_and_one_to_q01_and_q99() -> None:
    stats = NormStats(
        action_q01=np.full(10, -0.2, dtype=np.float32),
        action_q99=np.full(10, 0.3, dtype=np.float32),
        proprio_mean=np.zeros(10, dtype=np.float32),
        proprio_std=np.ones(10, dtype=np.float32),
    )
    q01, q99 = stats.action_q01, stats.action_q99
    lo = q99_denormalize(np.full((1, 10), -1.0, np.float32), q01, q99, Q99_DIMS)
    hi = q99_denormalize(np.full((1, 10), 1.0, np.float32), q01, q99, Q99_DIMS)
    assert np.allclose(lo[0, Q99_DIMS], -0.2) and np.allclose(hi[0, Q99_DIMS], 0.3)
    other = np.setdiff1d(np.arange(10), Q99_DIMS)
    assert np.allclose(lo[0, other], -1.0) and np.allclose(hi[0, other], 1.0)
    back = q99_normalize(hi, q01, q99, Q99_DIMS)
    assert np.allclose(back[0, Q99_DIMS], 1.0, atol=1e-6)
    assert np.array_equal(back[0, other], hi[0, other])


def test_clip_helpers_keep_values_in_range() -> None:
    x = np.array([[-0.0, 0.0, -1.0, 1.0, 0.25, -3.0, 3.0, np.nan]], np.float32)
    dims = np.arange(8)
    got = clip_normalized(x, dims)
    want = np.array([[-0.0, 0.0, -1.0, 1.0, 0.25, -1.0, 1.0, np.nan]], np.float32)
    assert got.tobytes() == want.tobytes()
    assert x[0, 5] == -3.0  # the input does not change
    g = clip_grip(np.array([[-0.5, 0.0, 0.5, 1.0, 2.0]], np.float32), dims[:5])
    assert g.tolist() == [[0.0, 0.0, 0.5, 1.0, 1.0]]
