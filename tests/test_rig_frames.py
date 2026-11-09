"""The pose frames of the clients (CPU only, no weights).

``tests/data/rig_requests.npz`` holds 1,570 recorded Positronic requests: the
client index (0 = the FR3 rig, 1 = RoboLab, 2 = MolmoSpaces), the joints
``q (7,)`` and the current pose8 of each request. The converted pose of each
request must be the flange forward kinematics of its joints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from praxis import single_arm as sa
from praxis.robot_io import unit_quat_wxyz_to_matrix

REQUESTS = Path(__file__).parent / "data" / "rig_requests.npz"
CLIENTS = ("", "remote.robolab.droid", "remote.molmo_spaces.droid")


def _requests() -> dict[str, np.ndarray]:
    with np.load(REQUESTS) as npz:
        return dict(npz)


def _flange(q: np.ndarray) -> np.ndarray:
    return np.stack([sa.franka_flange_pose(row) for row in q])


@pytest.mark.parametrize("client", range(len(CLIENTS)))
def test_recorded_requests_convert_to_the_flange(client: int) -> None:
    """The converted pose is the flange FK of the joints: within 1 mm and 2e-3 per rotation entry."""
    rec = _requests()
    rows = rec["client"] == client
    frame = sa.RIG_FRAMES[CLIENTS[client]]
    model = sa.pose8_to_model(rec["pose8"][rows], frame)
    fk = _flange(rec["q"][rows])
    err = np.linalg.norm(model[:, :3].astype(np.float64) - fk[:, :3, 3], axis=1)
    assert err.max() < 1e-3, err.max()
    rot = np.stack([unit_quat_wxyz_to_matrix(r[3:7]) for r in model])
    assert np.abs(rot - fk[:, :3, :3]).max() < 2e-3
    # The rig frame without the tool offset is 103.4 mm off on the FR3 rig.
    if client == 0:
        old = sa.pose8_to_model(rec["pose8"][rows])
        err = np.linalg.norm(old[:, :3].astype(np.float64) - fk[:, :3, 3], axis=1)
        assert 0.103 < err.min() and err.max() < 0.104


@pytest.mark.parametrize("client", range(len(CLIENTS)))
def test_flange_targets_go_back_to_the_client_pose(client: int) -> None:
    """A flange target in the model frame becomes the pose that the client reports."""
    rec = _requests()
    frame = sa.RIG_FRAMES[CLIENTS[client]]
    for pose8 in rec["pose8"][rec["client"] == client][:50]:
        model = sa.pose8_to_model(pose8, frame).astype(np.float64)
        xyz, rot = sa.pose_to_rig(
            model[:3], unit_quat_wxyz_to_matrix(model[3:7]), frame
        )
        assert np.abs(xyz - pose8[:3]).max() < 1e-6
        want = unit_quat_wxyz_to_matrix(pose8[3:7])
        assert np.abs(rot - want).max() < 1e-6


def test_session_converts_the_rig_pose_and_checks_the_joints() -> None:
    rec = _requests()
    i = int(np.flatnonzero(rec["client"] == 0)[0])  # an FR3 rig request
    pose8, q = rec["pose8"][i], rec["q"][i]
    obs = {
        "robot_state.ee_pose": pose8[None, :7],
        "grip": pose8[None, 7:],
        "robot_state.q": q,
    }
    frame = sa.rig_frame(obs)
    assert frame is sa.FR3_HAND
    model = sa.pose8_to_model(pose8, frame)
    sa.check_flange(obs, model)
    sa.check_flange({**obs, "robot_state.q": q[None]}, model)  # a stack
    with pytest.raises(ValueError, match="mm from the flange"):
        sa.check_flange(obs, sa.pose8_to_model(pose8))
    with pytest.raises(ValueError, match="7 joint angles"):
        sa.check_flange({"robot_state.q": q[:6]}, model)
    with pytest.raises(ValueError, match="mm from the flange"):
        sa.check_flange(obs, np.full(8, np.nan, np.float32))
    sa.check_flange({}, model)  # no joints: no check


def test_unknown_client_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown frame"):
        sa.rig_frame({"descriptor": "remote.other"})
    assert sa.rig_frame({"descriptor": b"remote.robolab.droid"}) is sa.FR3_HAND
    assert sa.rig_frame({"descriptor": "praxis.sim"}) is sa.TURNED_FLANGE


def test_turned_flange_keeps_the_earlier_conversion() -> None:
    """The sim frame (no offsets) keeps the bits of the earlier conversion."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64, 4))
    pose8 = np.concatenate(
        [
            rng.normal(size=(64, 3)),
            q / np.linalg.norm(q, axis=1, keepdims=True),
            rng.uniform(size=(64, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    rot = sa.quat_xyzw_to_matrix(pose8[:, [4, 5, 6, 3]].astype(np.float64))
    want = pose8.astype(np.float64).copy()
    want[:, 3:7] = sa.matrix_to_quat_xyzw(rot @ sa.FLANGE_TURN.T)[:, [3, 0, 1, 2]]
    want[:, 7] = np.clip(want[:, 7], 0.0, 1.0)
    assert np.array_equal(sa.pose8_to_model(pose8), want.astype(np.float32))


def test_a_refused_observation_keeps_the_frame_of_the_session() -> None:
    """A refused observation does not change the frame of the next commands."""
    from praxis.robot_io import NormStats

    rec = _requests()
    i = int(np.flatnonzero(rec["client"] == 0)[0])
    pose8 = rec["pose8"][i]
    sess = sa.SingleArmServeSession(
        encode=lambda clip: clip,
        stats=NormStats(*(np.zeros(10, np.float32) for _ in range(4))),
        device="cpu",
    )
    sess.frame = sa.FR3_HAND
    obs = {
        "robot_state.ee_pose": pose8[None, :7],
        "grip": pose8[None, 7:],
        "robot_state.q": rec["q"][i],
        "descriptor": "remote.molmo_spaces.droid",
    }
    with pytest.raises(ValueError, match="mm from the flange"):
        sess.prepare(obs)
    assert sess.frame is sa.FR3_HAND


def test_flange_descriptor_is_the_model_frame() -> None:
    """The ``"flange"`` client (a rig fine-tuned from the base) reports the pose
    of the model frame: the conversion is the identity both ways."""
    frame = sa.rig_frame({"descriptor": "flange"})
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pose8 = np.concatenate(
        [rng.uniform(-0.8, 0.8, (20, 3)), q, rng.uniform(0, 1, (20, 1))], axis=1
    ).astype(np.float32)
    model = sa.pose8_to_model(pose8, frame)
    np.testing.assert_allclose(model[:, :3], pose8[:, :3], atol=1e-7)
    for row, want in zip(model, pose8, strict=True):
        got = unit_quat_wxyz_to_matrix(row[3:7].astype(np.float64))
        np.testing.assert_allclose(
            got, unit_quat_wxyz_to_matrix(want[3:7].astype(np.float64)), atol=1e-6
        )
        xyz, rot = sa.pose_to_rig(row[:3].astype(np.float64), got, frame)
        np.testing.assert_allclose(xyz, want[:3], atol=1e-6)
        np.testing.assert_allclose(rot, got, atol=1e-12)
