"""Unit tests for the sim arm: frames, inverse kinematics, gripper, ticks (CPU only).

The tests need the ``mujoco`` package and the pinned Menagerie files. Without
the files and without a network, the tests skip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("praxis_sim")  # the `sim` extra

# praxis.sim sets the OpenGL backend. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis.sim import assets
from praxis.sim.env import EnvConfig, SimEnv
from praxis.sim.robot import (
    FrameConfig,
    policy_to_site,
    pos_rot_to_pose12,
    pose7_to_pos_rot,
    pose12_to_pos_rot,
    quat_to_mat,
    rot_z,
    rotation_angle_deg,
    site_to_policy,
)

# Tracking limits of the figure-eight test (end of each tick). Measured:
# p95 0.10 mm, max 0.58 mm, rotation max 0.07 deg.
TRACK_P95_M = 0.5e-3
TRACK_MAX_M = 2e-3
TRACK_ROT_MAX_DEG = 0.5


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_mat(q / np.linalg.norm(q))


@pytest.fixture(scope="module")
def env() -> SimEnv:
    try:
        assets.ensure_assets()
    except assets.AssetError as exc:
        pytest.skip(f"The pinned Menagerie files are not available: {exc}")
    sim = SimEnv(EnvConfig(render=False))
    yield sim
    sim.close()


def test_frame_conversion_round_trip() -> None:
    rng = np.random.default_rng(0)
    frames = [
        FrameConfig(),
        FrameConfig(yaw_offset_deg=-45.0, tcp_offset=0.16, base_offset=(0.0, 0.0, 0.3)),
        FrameConfig(
            yaw_offset_deg=90.0, tcp_offset=-0.02, base_offset=(0.1, -0.2, 0.0)
        ),
    ]
    for frame in frames:
        for _ in range(20):
            p, r = rng.normal(size=3), _random_rotation(rng)
            pp, rp = site_to_policy(p, r, frame)
            p2, r2 = policy_to_site(pp, rp, frame)
            np.testing.assert_allclose(p2, p, atol=1e-12)
            np.testing.assert_allclose(r2, r, atol=1e-12)
    p, r = rng.normal(size=3), _random_rotation(rng)
    pp, rp = site_to_policy(p, r, FrameConfig())
    np.testing.assert_array_equal(pp, p)
    np.testing.assert_allclose(rp, r, atol=1e-15)
    # The yaw offset turns the frame about its own z axis.
    _, ry = site_to_policy(p, r, FrameConfig(yaw_offset_deg=30.0))
    np.testing.assert_allclose(ry, r @ rot_z(30.0), atol=1e-12)


def test_pose_formats_round_trip() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        p, r = rng.normal(size=3), _random_rotation(rng)
        p12 = pos_rot_to_pose12(p, r)
        assert p12.dtype == np.float32 and p12.shape == (12,)
        p2, r2 = pose12_to_pos_rot(p12)
        np.testing.assert_allclose(p2, p, atol=1e-6)
        assert rotation_angle_deg(r2, r) < 1e-4
        np.testing.assert_allclose(r2 @ r2.T, np.eye(3), atol=1e-12)


def test_ee_pose_is_the_policy_convention(env: SimEnv) -> None:
    obs = env.reset(seed=0)
    assert obs.ee_pose.dtype == np.float32 and obs.ee_pose.shape == (7,)
    assert np.linalg.norm(obs.ee_pose[3:]) == pytest.approx(1.0, abs=1e-6)
    site_p, site_r = env.arm.site_pose()
    p, r = pose7_to_pos_rot(obs.ee_pose)
    np.testing.assert_allclose(p, site_p, atol=1e-6)
    assert rotation_angle_deg(r, site_r) < 1e-3
    # The start pose is in the range of the DROID data: tool down, yaw near +45.
    assert 0.30 < p[0] < 0.45 and abs(p[1]) < 0.05 and 0.40 < p[2] < 0.55
    assert r[2, 2] < -0.99
    assert math.degrees(math.atan2(r[1, 0], r[0, 0])) == pytest.approx(45.0, abs=2.0)
    assert obs.grip < 0.02


def test_ik_converges_on_reachable_targets(env: SimEnv) -> None:
    ik = env.arm.ik
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = np.clip(env.arm.home + rng.uniform(-0.4, 0.4, size=7), ik.lo, ik.hi)
        p, r = ik.forward(q)
        res = ik.solve(p, r, env.arm.home)
        assert res.converged and not res.miss
        assert res.pos_err < 1e-4 and math.degrees(res.rot_err) < 0.06
        assert np.all(res.q >= ik.lo) and np.all(res.q <= ik.hi)


def test_ik_keeps_joint_limits_on_a_far_target(env: SimEnv) -> None:
    ik = env.arm.ik
    p, r = ik.forward(env.arm.home)
    res = ik.solve(p + np.array([1.5, 0.0, 0.0]), r, env.arm.home)
    assert res.miss and not res.converged
    assert np.all(res.q >= ik.lo) and np.all(res.q <= ik.hi)


def test_ik_tracks_a_target_sequence(env: SimEnv) -> None:
    obs = env.reset(seed=0)
    p0, r0 = pose7_to_pos_rot(obs.ee_pose)
    n = 240

    def target(k: int) -> tuple[np.ndarray, np.ndarray]:
        a = 2 * math.pi * k / n
        p = np.array(
            [
                0.50 + 0.10 * math.sin(a),
                0.15 * math.sin(2 * a),
                0.35 + 0.10 * math.cos(a),
            ]
        )
        tilt = math.radians(10) * math.sin(a)
        c, s = math.cos(tilt), math.sin(tilt)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return p, rot_z(30 * math.sin(a)) @ r0 @ rx

    p_start, r_start = target(0)
    for k in range(30):  # move to the start of the path
        s = (k + 1) / 30
        env.step(
            pos_rot_to_pose12(p0 + (p_start - p0) * s, r_start if s == 1 else r0), 0.0
        )
    env.arm.settle(0.3)
    pos_err, rot_err = [], []
    for k in range(n):
        p, r = target(k)
        obs = env.step(pos_rot_to_pose12(p, r), 0.0)
        assert obs.ik is not None and not obs.ik.miss
        q, rq = pose7_to_pos_rot(obs.ee_pose)
        pos_err.append(np.linalg.norm(q - p))
        rot_err.append(rotation_angle_deg(rq, r))
    pos_arr, rot_arr = np.asarray(pos_err), np.asarray(rot_err)
    assert np.percentile(pos_arr, 95) < TRACK_P95_M
    assert pos_arr.max() < TRACK_MAX_M
    assert rot_arr.max() < TRACK_ROT_MAX_DEG


def test_gripper_map_and_bad_rows(env: SimEnv) -> None:
    obs = env.reset(seed=0)
    hold = pos_rot_to_pose12(*pose7_to_pos_rot(obs.ee_pose))
    for cmd, want in ((1.0, 0.99), (0.5, 0.48), (0.0, 0.0), (-0.3, 0.0)):
        for _ in range(8):
            obs = env.step(hold, cmd)
        assert obs.grip == pytest.approx(want, abs=0.05)
    for _ in range(8):
        obs = env.step(hold, 1.0)
    closed = obs.grip
    obs = env.step(hold, float("nan"))  # keeps the previous grip target
    assert env.arm.grip_cmd == 1.0 and obs.grip == pytest.approx(closed, abs=0.02)
    before = obs.ee_pose.copy()
    bad = hold.copy()
    bad[0] = np.nan
    obs = env.step(bad, 1.0)  # keeps the previous arm target
    assert obs.ik is None and env.arm.bad_rows == 1
    np.testing.assert_allclose(obs.ee_pose[:3], before[:3], atol=1e-3)
