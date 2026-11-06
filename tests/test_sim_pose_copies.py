"""The pose helpers of ``praxis_sim.pose`` are copies of the core functions: equal, bit for bit."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from praxis_sim import pose

from praxis import robot_io, single_arm


def _quats(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_rotation_helpers_match_the_core() -> None:
    q = _quats(200, 0).astype(np.float32)
    for row in q:
        assert np.array_equal(
            pose.unit_quat_wxyz_to_matrix(row), robot_io.unit_quat_wxyz_to_matrix(row)
        )
        assert np.array_equal(
            pose.quat_wxyz_to_matrix(row), robot_io.quat_wxyz_to_matrix(row)
        )
    m = np.stack([robot_io.unit_quat_wxyz_to_matrix(row) for row in q])
    for rot in m:
        assert np.array_equal(pose.rot6d(rot), robot_io.rot6d(rot))
        r6 = robot_io.rot6d(rot)
        assert np.array_equal(pose.rot6d_to_matrix(r6), robot_io.rot6d_to_matrix(r6))
    xyzw = _quats(200, 1)
    assert np.array_equal(
        pose.quat_xyzw_to_matrix(xyzw), single_arm.quat_xyzw_to_matrix(xyzw)
    )
    assert np.array_equal(
        pose.matrix_to_quat_xyzw(m), single_arm.matrix_to_quat_xyzw(m)
    )


def test_frame_conversions_match_the_core() -> None:
    rng = np.random.default_rng(2)
    pose8 = np.concatenate(
        [rng.normal(size=(50, 3)), _quats(50, 3), rng.uniform(-0.2, 1.2, size=(50, 1))],
        axis=1,
    )
    for frame_name in ("TURNED_FLANGE", "FR3_HAND"):
        ours, core = getattr(pose, frame_name), getattr(single_arm, frame_name)
        assert dataclasses.astuple(ours) == dataclasses.astuple(core)
        assert np.array_equal(
            pose.pose8_to_model(pose8, ours), single_arm.pose8_to_model(pose8, core)
        )
        xyz, rot = rng.normal(size=3), robot_io.unit_quat_wxyz_to_matrix(pose8[0, 3:7])
        a, b = pose.pose_to_rig(xyz, rot, ours), single_arm.pose_to_rig(xyz, rot, core)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert pose.control_mode_payload(
        "droid_impedance"
    ) == single_arm.control_mode_payload("droid_impedance")
    assert pose.DROID_IMPEDANCE == single_arm.DROID_IMPEDANCE
    assert pose.control_mode_payload("none") is None
    rel = np.tile(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], np.float32), (3, 1))
    wire = single_arm.rel10_to_wire(rel, pose8[0])
    a, b = pose.wire_to_arrays(wire), single_arm.wire_to_arrays(wire)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_euler_helpers_match_the_training_data_module() -> None:
    """The Euler and quaternion helpers copy ``praxis.train.data`` (the DROID pose code)."""
    pytest.importorskip("av")  # praxis.train.data needs the `train` extra
    pytest.importorskip("pyarrow")
    from praxis.train import data

    rng = np.random.default_rng(4)
    rpy = rng.uniform(-np.pi, np.pi, size=(200, 3))
    assert np.array_equal(
        pose.euler_xyz_to_quat_xyzw(rpy), data.euler_xyz_to_quat_xyzw(rpy)
    )
    p, q = _quats(200, 5), _quats(200, 6)
    assert np.array_equal(pose.quat_mul_xyzw(p, q), data._quat_mul_xyzw(p, q))
    assert np.array_equal(
        pose.quat_xyzw_to_matrix_raw(p), data.quat_xyzw_to_matrix_raw(p)
    )
    # The composition the Gymnasium wrapper uses, and its inverse (new code): a round trip
    # of the DROID pose through pose8 keeps the Euler angles to float precision.
    rot = pose.euler_xyz_to_matrix(rpy)
    assert np.array_equal(
        rot, data.quat_xyzw_to_matrix_raw(data.euler_xyz_to_quat_xyzw(rpy))
    )
    back = pose.matrix_to_euler_xyz(rot)
    assert np.allclose(pose.euler_xyz_to_matrix(back), rot, atol=1e-12)
