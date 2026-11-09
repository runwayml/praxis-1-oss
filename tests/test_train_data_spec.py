"""Tests of the dataset spec of other single-arm rigs (CPU, no weights, no network)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("av")

from test_train_data_droid import (  # noqa: E402
    CONFIG,
    EPISODES,
    _camera,
    _rot,
    _streams,
    _write_video,
    make_droid_tree,
)

from praxis.single_arm import matrix_to_quat_xyzw  # noqa: E402
from praxis.train import data  # noqa: E402

GRIP_CLOSED = 0.08  # metres: the raw gripper of the other rig
VIEWS = ("observation.images.front", "observation.images.hand")


def _quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    return np.stack([matrix_to_quat_xyzw(_rot(r)) for r in rpy])


@pytest.fixture(scope="module")
def droid_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The DROID tree of ``test_train_data_droid``."""
    return make_droid_tree(tmp_path_factory.mktemp("droid"))


@pytest.fixture(scope="module")
def quat_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The poses of the DROID fixture in another layout: one ``state`` and one
    ``action`` column ``[xyz, quat_wxyz, grip]``, the gripper in metres, other
    video keys and no success column."""
    root = tmp_path_factory.mktemp("quat")
    features = {k: {} for k in (*VIEWS, "observation.state", "action", "timestamp")}
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info))
    lines = []
    for index in (1, 2):
        n, task = EPISODES[index][0], EPISODES[index][1]
        lines.append(json.dumps({"episode_index": index, "tasks": [task], "length": n}))
        s = _streams(n)

        def rows(pose6: np.ndarray, grip: np.ndarray) -> list[list[float]]:
            q = _quat_xyzw(pose6[:, 3:6])[:, [3, 0, 1, 2]]  # wxyz
            g = np.clip(grip, 0.0, 1.0) * GRIP_CLOSED
            return np.concatenate([pose6[:, :3], q, g[:, None]], 1).tolist()

        table = pa.table(
            {
                "observation.state": pa.array(
                    rows(s["obs"], s["obs_grip"]), pa.list_(pa.float64())
                ),
                "action": pa.array(
                    rows(s["cmd"], s["cmd_grip"]), pa.list_(pa.float64())
                ),
                "timestamp": np.arange(n) / 15.0,
            }
        )
        path = root / f"data/chunk-000/episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for view, key in enumerate(VIEWS):
            _write_video(
                root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4",
                _camera(index, n, view),
                rate=15,
            )
    (root / "meta/episodes.jsonl").write_text("\n".join(lines) + "\n")
    return root


QUAT_SPEC = data.DatasetSpec(
    views=VIEWS,
    observed_pose="observation.state[0:7]",
    observed_gripper="observation.state[7]",
    commanded_pose="action[0:7]",
    commanded_gripper="action[7]",
    rotation="quat_wxyz",
    gripper_range=(0.0, GRIP_CLOSED),
    success=None,
)


def test_column_ref_parse() -> None:
    p = data.ColumnRef.parse
    assert p("a.b") == data.ColumnRef("a.b")
    assert p("action[7]") == data.ColumnRef("action", 7, 8)
    assert p("action[0:7]") == data.ColumnRef("action", 0, 7)
    assert p("action[3:]") == data.ColumnRef("action", 3, None)
    with pytest.raises(ValueError):
        p("action[")


def test_spec_defaults_are_droid() -> None:
    spec = data.DatasetSpec()
    assert spec.views == data.VIEWS
    assert spec.columns() == [*data.OBSERVED, *data.COMMANDED, data.SUCCESS]
    with pytest.raises(ValueError):
        data.DatasetSpec(rotation="euler_zyx")
    with pytest.raises(ValueError):
        data.DatasetSpec(views=("a",))
    with pytest.raises(ValueError):
        data.DatasetSpec(gripper_range=(1.0, 1.0))


def test_rig_pose8_euler_is_droid_pose8() -> None:
    rng = np.random.default_rng(0)
    pose6 = rng.uniform(-3, 3, size=(50, 6))
    grip = rng.uniform(-0.2, 1.2, size=50)
    assert np.array_equal(data.rig_pose8(pose6, grip), data.droid_pose8(pose6, grip))


@pytest.mark.parametrize("rotation", ["quat_xyzw", "quat_wxyz"])
def test_rig_pose8_quaternions_match_euler(rotation: str) -> None:
    rng = np.random.default_rng(1)
    pose6 = rng.uniform(-3, 3, size=(50, 6))
    q = _quat_xyzw(pose6[:, 3:6])
    q = -q if rotation == "quat_wxyz" else q  # the sign of the input does not matter
    if rotation == "quat_wxyz":
        q = q[:, [3, 0, 1, 2]]
    grip = rng.uniform(0, 2, size=50)
    got = data.rig_pose8(np.concatenate([pose6[:, :3], q], 1), grip, rotation, (0, 2))
    ref = data.droid_pose8(pose6, grip / 2)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_quat_dataset_gives_the_droid_windows(
    droid_root: Path, quat_root: Path
) -> None:
    """The same episodes in another layout give the same windows (within the
    float32 poses of the DROID parquet files)."""
    droid = data.DroidWindowDataset(str(droid_root), CONFIG)
    other = data.DroidWindowDataset(str(quat_root), CONFIG, spec=QUAT_SPEC)
    assert other.episodes == [1, 2]
    for index, anchor in ((1, 0), (2, 25)):
        a_ref, p_ref = droid.window_poses(droid.load_episode(index), anchor)
        a, p = other.window_poses(other.load_episode(index), anchor)
        np.testing.assert_allclose(a, a_ref, atol=2e-6)
        np.testing.assert_allclose(p, p_ref, atol=2e-6)
    w = other.load_window(other.load_episode(2), 25)
    assert w.frames.shape[0] == droid.window_frames
    np.testing.assert_allclose(w.actions_rel, a, atol=0)


def test_missing_spec_columns_are_reported(quat_root: Path) -> None:
    with pytest.raises(ValueError, match="no feature"):
        data.DroidWindowDataset(str(quat_root), CONFIG)  # the DROID columns


@pytest.mark.parametrize("rotation", ["quat_xyzw", "quat_wxyz"])
def test_rig_pose8_refuses_a_zero_quaternion(rotation: str) -> None:
    """A zero quaternion has no rotation: the episode leaves the pool (a
    ValueError of load_episode) instead of sending NaN into the windows."""
    pose = np.zeros((3, 7))
    pose[:, 3 if rotation == "quat_wxyz" else 6] = 1.0  # identity rotations
    pose[1, 3:7] = 0.0
    with pytest.raises(ValueError, match="zero quaternion"):
        data.rig_pose8(pose, np.zeros(3), rotation)


def test_spec_widths_are_checked_against_the_rotation() -> None:
    features = {
        "observation.state": {"shape": [8]},
        "action": {"shape": [8]},
        "observation.state.cartesian_position": {"shape": [6]},
    }
    QUAT_SPEC.check_widths(features)  # [xyz, quat_wxyz] slices and one gripper value
    wrong = replace(QUAT_SPEC, rotation="euler_xyz")
    with pytest.raises(ValueError, match=r"observation\.state"):
        wrong.check_widths(features)
    sliced = replace(QUAT_SPEC, commanded_pose="action[0:6]")
    with pytest.raises(ValueError, match=r"action\[0:6\]"):
        sliced.check_widths(features)
    # the DROID defaults: a 6-value euler pose column and a gripper column
    droid = data.DatasetSpec()
    droid.check_widths(
        {
            droid.observed_pose: {"shape": [6]},
            droid.commanded_pose: {"shape": [6]},
            droid.observed_gripper: {"shape": [1]},
            droid.commanded_gripper: {"shape": [1]},
        }
    )
    droid.check_widths({})  # no shapes: nothing to check
