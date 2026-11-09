"""Unit tests for the DROID training windows (CPU only, no weights, no network)."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

av = pytest.importorskip("av")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from praxis.robot_io import contiguous_chunk_geometry  # noqa: E402
from praxis.single_arm import (  # noqa: E402
    FLANGE_TURN,
    matrix_to_quat_xyzw,
    pose8_to_model,
    rel10_to_wire,
    tile_clip,
)
from praxis.train import data  # noqa: E402

TILE = (8, 12)
CONFIG = {
    "action_space": {"dim": 10},
    "chunking": {
        "n_chunks": 4,
        "action_chunk": 24,
        "grid_hz": 15.0,
        "window_frames": 93,
        "chunk_geometry": [[0, 21], [21, 24], [45, 24], [69, 24]],
    },
    "video": {
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "tile_height": TILE[0],
        "tile_width": TILE[1],
        "aspect_tol": 0.03,
    },
    "pose_frame": "flange",
}
CAM_H, CAM_W = 12, 16
GEOMETRY = contiguous_chunk_geometry(24, 4)
# episode -> (frames, task, successful, timestamp shift, wrist frames)
# Wrist frames: None = all frames, 0 = no wrist video file.
EPISODES = {
    0: (110, "wipe the table", True, 0, None),
    1: (110, "push the block", True, 0, None),
    2: (140, "open the drawer", True, 0, None),
    3: (110, "close the drawer", False, 0, None),
    4: (110, "", True, 0, None),
    5: (101, "stack the cups", True, 0, None),
    6: (110, "pick the pen", True, 0, 50),
    7: (110, "fold the towel", True, 1, None),
    8: (110, "pour the water", True, 0, 0),
    50: (110, "  move the cup ", True, 0, None),
}
TRAIN_POOL = [1, 2, 3, 6, 7]


def _rot(rpy: np.ndarray) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) from elementary rotations (test reference)."""
    a, b, c = (float(v) for v in rpy)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _streams(n: int) -> dict[str, np.ndarray]:
    """Observed and commanded poses: x moves 2 mm per frame."""
    t = np.arange(n, dtype=np.float64)
    obs = np.stack(
        [
            0.4 + 0.002 * t,
            0.05 * np.sin(t / 20),
            0.3 + 0.001 * t,
            3.0 + 0.01 * t,
            -0.2 + 0.004 * t,
            0.4 - 0.012 * t,
        ],
        axis=1,
    )
    cmd = obs + np.array([0.004, -0.002, 0.003, 0.02, -0.01, 0.03])
    return {
        "obs": obs,
        "cmd": cmd,
        "obs_grip": np.linspace(0.0, 0.8, n),
        "cmd_grip": np.linspace(-0.2, 1.2, n),
    }


def _camera(index: int, n: int, view: int) -> np.ndarray:
    """Frame t of a camera is the value ``2 t + 7 view`` with the episode in channel 2."""
    img = np.zeros((n, CAM_H, CAM_W, 3), dtype=np.uint8)
    img[..., 0] = ((2 * np.arange(n) + 7 * view) % 256)[:, None, None]
    img[..., 1] = 255 - img[..., 0]
    img[..., 2] = index
    return img


def _write_video(path: Path, frames: np.ndarray, rate: int) -> None:
    """Write a lossless H.264 video with a key frame every 4 frames.

    ``rate`` is the time stamp rate. The DROID raw files have 60 fps time
    stamps and one frame per 15 Hz step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        options = {"qp": "0", "keyint_min": "4", "sc_threshold": "0"}
        stream = container.add_stream("libx264rgb", rate=rate, options=options)
        stream.width, stream.height, stream.pix_fmt = CAM_W, CAM_H, "rgb24"
        stream.codec_context.gop_size = 4
        stream.codec_context.max_b_frames = 0
        for img in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A LeRobot v2.1 directory with the episodes of ``EPISODES``."""
    return make_droid_tree(tmp_path_factory.mktemp("droid"))


def make_droid_tree(root: Path) -> Path:
    """Write a LeRobot v2.1 directory with the episodes of ``EPISODES`` to ``root``."""
    features = {k: {} for k in (*data.VIEWS, *data.OBSERVED, *data.COMMANDED)}
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {**features, data.SUCCESS: {}, data.TIMESTAMP: {}},
    }
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info))
    lines = []
    for index, (n, task, ok, shift, n_wrist) in EPISODES.items():
        lines.append(json.dumps({"episode_index": index, "tasks": [task], "length": n}))
        s = _streams(n)
        table = pa.table(
            {
                data.OBSERVED[0]: pa.array(s["obs"].tolist(), pa.list_(pa.float32())),
                data.OBSERVED[1]: s["obs_grip"],
                data.COMMANDED[0]: pa.array(s["cmd"].tolist(), pa.list_(pa.float32())),
                data.COMMANDED[1]: s["cmd_grip"],
                data.SUCCESS: [ok] * n,
                data.TIMESTAMP: (np.arange(n) + shift) / 15.0,
                "frame_index": np.arange(n),
            }
        )
        path = root / f"data/chunk-000/episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for view, key in enumerate(data.VIEWS):
            frames = _camera(index, n, view)
            if view == 1 and n_wrist == 0:
                continue
            if view == 1 and n_wrist is not None:
                frames = frames[:n_wrist]
            _write_video(
                root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4",
                frames,
                rate=60 if index == 2 else 15,
            )
    (root / "meta/episodes.jsonl").write_text("\n".join(lines) + "\n")
    return root


def _float32_pose(x: np.ndarray) -> np.ndarray:
    """The poses as the parquet file keeps them (float32)."""
    return x.astype(np.float32).astype(np.float64)


def _draws(ds: data.DroidWindowDataset, n: int) -> list[data.TrainingWindow]:
    it = iter(ds)
    return [next(it) for _ in range(n)]


def test_droid_pose8_matches_reference_quaternions() -> None:
    rpy = np.array(
        [[3.0, -0.2, 0.4], [-3.1, 0.05, 2.9], [0.1, 1.4, -1.0], [3.14159, 0.0, 0.0]]
    )
    # Reference values: scipy Rotation.from_matrix(Rotation.from_euler("xyz",
    # rpy).as_matrix()).as_quat(), wxyz, float32. The flange frame: no turn.
    expected = np.array(
        [
            [4.91966642e-02, 9.74130511e-01, 1.90260455e-01, 1.11581400e-01],
            [2.23048218e-02, 1.20955095e-01, 9.92125571e-01, -2.36484483e-02],
            [6.54937029e-01, 3.42015058e-01, 5.46321034e-01, -3.94482553e-01],
            [1.32679486e-06, 1.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
        dtype=np.float32,
    )
    xyz = np.array([[0.1, 0.2, 0.3]] * 4)
    pose8 = data.droid_pose8(
        np.concatenate([xyz, rpy], axis=1), np.array([-1, 0, 0.5, 2])
    )
    assert pose8.dtype == np.float32 and pose8.shape == (4, 8)
    np.testing.assert_allclose(pose8[:, 3:7], expected, atol=1e-7)
    np.testing.assert_array_equal(pose8[:, :3], xyz.astype(np.float32))
    np.testing.assert_array_equal(pose8[:, 7], [0.0, 0.0, 0.5, 1.0])


def test_euler_conversion_matches_the_elementary_rotations() -> None:
    rng = np.random.default_rng(0)
    rpy = rng.uniform(-np.pi, np.pi, size=(64, 3))
    m = data.quat_xyzw_to_matrix_raw(data.euler_xyz_to_quat_xyzw(rpy))
    for i in range(len(rpy)):
        np.testing.assert_allclose(m[i], _rot(rpy[i]), atol=1e-12)


def test_droid_pose8_has_the_bits_of_scipy() -> None:
    transform = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(1)
    rpy = rng.uniform(-np.pi, np.pi, size=(4096, 3)).astype(np.float32)
    rpy[:64] = rng.normal(size=(64, 3)) * 1e-3
    rot = transform.Rotation.from_euler("xyz", rpy.astype(np.float64))
    np.testing.assert_array_equal(data.euler_xyz_to_quat_xyzw(rpy), rot.as_quat())
    want = transform.Rotation.from_matrix(rot.as_matrix()).as_quat()[:, [3, 0, 1, 2]]
    pose8 = data.droid_pose8(np.concatenate([np.zeros_like(rpy), rpy], 1), rpy[:, 0])
    np.testing.assert_array_equal(pose8[:, 3:7], want.astype(np.float32))


def test_droid_pose8_is_the_serve_conversion_of_the_rig_pose() -> None:
    """The rig reports R_flange @ Rz(-45 deg). The serve path converts it back
    with ``pose8_to_model``. Both give the same flange pose."""
    rng = np.random.default_rng(2)
    rpy = rng.uniform(-np.pi, np.pi, size=(256, 3))
    pose6 = np.concatenate([rng.uniform(-1, 1, size=(256, 3)), rpy], axis=1)
    grip = rng.uniform(-0.5, 1.5, size=256)
    rig_rot = np.stack([_rot(r) for r in rpy]) @ FLANGE_TURN
    rig = np.concatenate(
        [pose6[:, :3], matrix_to_quat_xyzw(rig_rot)[:, [3, 0, 1, 2]], grip[:, None]],
        axis=1,
    )
    ours = data.droid_pose8(pose6, grip)
    serve = pose8_to_model(rig)
    np.testing.assert_array_equal(ours[:, [0, 1, 2, 7]], serve[:, [0, 1, 2, 7]])
    # Same rotation (a quaternion and its negative are the same rotation).
    dots = np.abs((ours[:, 3:7].astype(np.float64) * serve[:, 3:7]).sum(axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-6)


def test_window_fits_and_anchor_draw() -> None:
    # 6.2 s window + 0.5 s margin: 102 frames or more.
    assert data.window_fits(102, 93, 15.0) and not data.window_fits(101, 93, 15.0)
    assert data.episode_duration(121, 15.0) == pytest.approx(8.0 + 1 / 60)
    rng = np.random.default_rng(0)
    n = 110
    draws = np.array([data.draw_anchor(n, 93, 15.0, rng) for _ in range(40000)])
    values, counts = np.unique(draws, return_counts=True)
    # Uniform over 0 .. n - 95, a quarter weight at n - 94, never n - 93.
    assert values.tolist() == list(range(n - 93))
    full = counts[:-1] / len(draws)
    np.testing.assert_allclose(full, 1 / 16.25, rtol=0.1)
    assert counts[-1] / len(draws) == pytest.approx(0.25 / 16.25, rel=0.2)
    with pytest.raises(ValueError):
        data.draw_anchor(101, 93, 15.0, rng)
    # The shortest episode: start in [0, 0.55 s], first frames 0 .. 8.
    short = {data.draw_anchor(102, 93, 15.0, rng) for _ in range(2000)}
    assert short == set(range(9))


def test_pool_split_and_ranks(root: Path) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG)
    # Episode 8 has no wrist video file, so it is not in the pool. Episode 5
    # is too short and episode 4 has no task.
    assert ds.episodes == TRAIN_POOL
    assert ds.window_frames == 93 and ds.tile == TILE
    val = data.DroidWindowDataset(str(root), CONFIG, split="val")
    assert val.episodes == [0, 50] and val.prompts[50] == "move the cup"
    shards = [
        data.DroidWindowDataset(str(root), CONFIG, rank=r, world_size=2).episodes
        for r in range(2)
    ]
    assert shards == [[1, 3, 7], [2, 6]]
    assert data.DroidWindowDataset(str(root), CONFIG, max_episodes=2).episodes == [1, 2]
    # A list of episodes replaces the split and keeps its order.
    listed = data.DroidWindowDataset(str(root), CONFIG, episodes=[50, 2, 1])
    assert listed.episodes == [50, 2, 1]
    sharded = [
        data.DroidWindowDataset(
            str(root), CONFIG, episodes=[50, 2, 1], max_episodes=2, rank=r, world_size=2
        ).episodes
        for r in range(2)
    ]
    assert sharded == [[50], [2]]
    with pytest.raises(ValueError, match="more than once"):
        data.DroidWindowDataset(str(root), CONFIG, episodes=[50, 2, 50])
    for unusable in (4, 5, 8, 999):  # no task, too short, no wrist file, absent
        with pytest.raises(ValueError, match="not usable"):
            data.DroidWindowDataset(str(root), CONFIG, episodes=[1, unusable])
    with pytest.raises(ValueError):
        data.DroidWindowDataset(str(root), CONFIG, split="test")
    with pytest.raises(ValueError):
        data.DroidWindowDataset(str(root), {**CONFIG, "action_space": {"dim": 20}})
    with pytest.raises(ValueError, match="pose_frame"):
        data.DroidWindowDataset(str(root), {**CONFIG, "pose_frame": "flange_turn_-45"})
    chunking = dict(CONFIG["chunking"])  # ty: ignore[no-matching-overload]
    bad = {**CONFIG, "chunking": {**chunking, "chunk_geometry": [[0, 24]]}}
    with pytest.raises(ValueError, match="chunk_geometry"):
        data.DroidWindowDataset(str(root), bad)


def test_episode_checks(root: Path) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG)
    episode = ds.load_episode(1)
    assert episode.prompt == "push the block"
    assert episode.observed.shape == episode.commanded.shape == (110, 8)
    with pytest.raises(ValueError, match="not successful"):
        ds.load_episode(3)
    with pytest.raises(ValueError, match="time"):
        ds.load_episode(7)
    with pytest.raises(ValueError, match="decoded"):
        ds.load_window(ds.load_episode(6), 0)


def test_anchor_range(root: Path) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG)
    seen: dict[int, set[int]] = {1: set(), 2: set()}
    for w in _draws(ds, 200):
        seen[w.meta["episode"]].add(w.meta["anchor"])
    assert seen[1] <= set(range(17)) and len(seen[1]) > 12
    assert seen[2] <= set(range(47)) and len(seen[2]) > 30


def test_window_geometry_frames_actions_proprio(root: Path) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG)
    w = ds.load_window(ds.load_episode(2), 25)
    assert w.meta == {"episode": 2, "anchor": 25, "source": str(root)}
    assert w.prompt == "open the drawer"
    assert w.frames.shape == (93, 3, 8, 24) and w.frames.dtype == torch.uint8
    assert w.actions_rel.shape == (93, 10) and w.actions_rel.dtype == np.float32
    assert w.proprio_abs.shape == (4, 10) and w.proprio_abs.dtype == np.float32

    # Episode 2 has 60 fps time stamps: frame i is the i-th decoded frame.
    ext, wrist = (_camera(2, 140, v)[25:118] for v in (0, 1))
    canvas = tile_clip(ext, wrist, *TILE, 0.03)[0].permute(1, 0, 2, 3)
    assert torch.equal(w.frames, canvas.to(torch.uint8))
    assert w.frames[:, 0, 4, 6].tolist() == [2 * t for t in range(25, 118)]

    s = _streams(140)
    obs, cmd = _float32_pose(s["obs"]), _float32_pose(s["cmd"])
    for k, (off, ln) in enumerate(GEOMETRY):
        sk = 25 + off
        r_obs = _rot(obs[sk, 3:])
        p = w.proprio_abs[k]
        np.testing.assert_allclose(p[:3], obs[sk, :3], atol=1e-6)
        np.testing.assert_allclose(
            p[3:9], np.concatenate([r_obs[:, 0], r_obs[:, 1]]), atol=1e-6
        )
        np.testing.assert_allclose(p[9], np.clip(s["obs_grip"][sk], 0, 1), atol=1e-6)
        for t in range(sk, sk + ln):
            row = w.actions_rel[off + t - sk]
            r_rel = _rot(cmd[t, 3:]) @ r_obs.T
            np.testing.assert_allclose(row[:3], cmd[t, :3] - obs[sk, :3], atol=1e-6)
            np.testing.assert_allclose(
                row[3:9], np.concatenate([r_rel[:, 0], r_rel[:, 1]]), atol=1e-6
            )
            np.testing.assert_allclose(
                row[9], np.clip(s["cmd_grip"][t], 0, 1), atol=1e-6
            )


def test_targets_are_the_inverse_of_the_serve_decode(root: Path) -> None:
    """``rel10_to_wire`` of the action rows of a chunk, with the observed pose
    at the chunk start as the anchor, gives the commanded poses in the frame
    of the rig (R_flange @ Rz(-45 deg))."""
    ds = data.DroidWindowDataset(str(root), CONFIG)
    episode = ds.load_episode(2)
    w = ds.load_window(episode, 25)
    s = _streams(140)
    cmd = _float32_pose(s["cmd"])
    for off, ln in GEOMETRY:
        sk = 25 + off
        wire = rel10_to_wire(w.actions_rel[off : off + ln], episode.observed[sk])
        for t, row in enumerate(wire):
            pose = np.asarray(row["robot_command"]["pose"], dtype=np.float64)
            np.testing.assert_allclose(pose[:3], cmd[sk + t, :3], atol=1e-6)
            want = _rot(cmd[sk + t, 3:]) @ FLANGE_TURN
            np.testing.assert_allclose(pose[3:].reshape(3, 3), want, atol=1e-5)
            grip = np.clip(s["cmd_grip"][sk + t], 0, 1)
            assert row["target_grip"] == pytest.approx(grip, abs=1e-6)


def test_iteration_is_seeded_and_skips_bad_episodes(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG, seed=3)
    first = [(w.meta["episode"], w.meta["anchor"]) for w in _draws(ds, 12)]
    again = [(w.meta["episode"], w.meta["anchor"]) for w in _draws(ds, 12)]
    assert first == again
    assert {e for e, _ in first} <= {1, 2}
    other = data.DroidWindowDataset(str(root), CONFIG, seed=4)
    assert [(w.meta["episode"], w.meta["anchor"]) for w in _draws(other, 12)] != first
    monkeypatch.setattr(data, "get_worker_info", lambda: SimpleNamespace(id=1))
    worker1 = [(w.meta["episode"], w.meta["anchor"]) for w in _draws(ds, 12)]
    assert worker1 != first


def test_hub_files_download_on_first_use(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    def download(repo: str, rel: str, **kwargs: object) -> str:
        assert repo == "cadene/droid_1.0.1" and kwargs["repo_type"] == "dataset"
        calls.append((rel, kwargs["revision"]))
        return str(root / rel)

    info = SimpleNamespace(dataset_info=lambda repo: SimpleNamespace(sha="abc123"))
    monkeypatch.setattr(data, "HfApi", lambda: info)
    monkeypatch.setattr(data, "hf_hub_download", download)
    ds = data.DroidWindowDataset("cadene/droid_1.0.1", CONFIG, max_episodes=1)
    assert calls == [("meta/info.json", "abc123"), ("meta/episodes.jsonl", "abc123")]
    calls.clear()
    w = next(iter(ds))
    assert w.meta["episode"] == 1 and w.meta["source"] == "cadene/droid_1.0.1"
    assert [rel for rel, _ in calls] == [
        "data/chunk-000/episode_000001.parquet",
        f"videos/chunk-000/{data.VIEWS[0]}/episode_000001.mp4",
        f"videos/chunk-000/{data.VIEWS[1]}/episode_000001.mp4",
    ]


class _Hub:
    """A stand-in for the hub over the files of ``root``.

    A download first raises the errors of ``errors``. A file that is not in
    ``root`` gives the 404 error of the hub.
    """

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.errors: list[Exception] = []
        self.calls: list[str] = []
        self.waits: list[float] = []
        info = SimpleNamespace(dataset_info=lambda repo: SimpleNamespace(sha="abc"))
        monkeypatch.setattr(data, "HfApi", lambda: info)
        monkeypatch.setattr(data, "hf_hub_download", self.download)
        monkeypatch.setattr(data.time, "sleep", self.waits.append)

    def download(self, repo: str, rel: str, **kwargs: object) -> str:
        self.calls.append(rel)
        if self.errors:
            raise self.errors.pop(0)
        if not (self.root / rel).is_file():
            response = httpx.Response(404, request=httpx.Request("HEAD", rel))
            raise RemoteEntryNotFoundError("404", response=response)
        return str(self.root / rel)


def test_hub_errors_get_new_tries(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = _Hub(root, monkeypatch)
    ds = data.DroidWindowDataset("cadene/droid_1.0.1", CONFIG)
    # Hub mode does not look for the episode files when it starts.
    assert ds.episodes == [*TRAIN_POOL, 8]
    hub.calls.clear()
    response = httpx.Response(429, request=httpx.Request("HEAD", "x"))
    hub.errors = [httpx.ConnectError("down"), HfHubHTTPError("429", response=response)]
    assert ds._file("meta/info.json") == root / "meta/info.json"
    assert hub.calls == ["meta/info.json"] * 3 and hub.waits == [1.0, 2.0]

    monkeypatch.setattr(data, "HUB_TRIES", 3)
    hub.waits.clear()
    hub.errors = [httpx.ConnectError("down") for _ in range(3)]
    ds.episodes = [1]
    # The iteration stops. A network error does not remove the episode.
    with pytest.raises(RuntimeError, match="in 3 tries"):
        next(iter(ds))
    assert hub.waits == [1.0, 2.0]


def test_hub_episode_without_a_file_leaves_the_pool(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = _Hub(root, monkeypatch)
    ds = data.DroidWindowDataset("cadene/droid_1.0.1", CONFIG)
    ds.episodes = [8, 1]
    hub.calls.clear()
    assert {w.meta["episode"] for w in _draws(ds, 8)} == {1}
    # Episode 8 has no wrist video. It leaves the pool after one try.
    wrist = f"videos/chunk-000/{data.VIEWS[1]}/episode_000008.mp4"
    assert hub.calls.count(wrist) == 1 and hub.waits == []


# Recorded with the window start rule of the checkpoint recipe: the
# (episode, anchor) of the first windows of two default sequences.
RECORDED_ORDER = {
    (3, 0, 1): [
        (1, 3),
        (1, 9),
        (1, 7),
        (2, 7),
        (2, 5),
        (1, 8),
        (1, 6),
        (2, 34),
        (2, 44),
        (2, 29),
        (1, 11),
        (2, 0),
    ],
    (5, 1, 2): [(2, 21), (2, 37), (2, 24), (2, 23), (2, 9), (2, 3), (2, 22), (2, 45)],
}


def test_default_window_order_matches_the_recorded_order(root: Path) -> None:
    for (seed, rank, world), want in RECORDED_ORDER.items():
        ds = data.DroidWindowDataset(
            str(root), CONFIG, seed=seed, rank=rank, world_size=world
        )
        got = [(w.meta["episode"], w.meta["anchor"]) for w in _draws(ds, len(want))]
        assert got == want


def test_fixed_windows_are_the_first_windows(root: Path) -> None:
    ds = data.DroidWindowDataset(str(root), CONFIG, seed=3)
    fixed = data.fixed_windows(ds, 5)
    first = _draws(ds, 5)
    assert [(w.meta["episode"], w.meta["anchor"]) for w in fixed] == RECORDED_ORDER[
        (3, 0, 1)
    ][:5]
    for a, b in zip(fixed, first, strict=True):
        assert torch.equal(a.frames, b.frames) and a.prompt == b.prompt
        np.testing.assert_array_equal(a.actions_rel, b.actions_rel)
        np.testing.assert_array_equal(a.proprio_abs, b.proprio_abs)
    with pytest.raises(ValueError):
        data.fixed_windows(ds, 0)


def test_overfit_batches_read_files_only_once(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overfit mode of the script draws the windows of the full pool on
    rank 0 once. Later micro-steps read no file and decode no video."""
    import importlib.util

    import torch.distributed as dist

    from praxis.train.objective import EncodedWindow

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "finetune_script", repo / "scripts" / "finetune.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    reads: list[str] = []
    real_file = data.DroidWindowDataset._file

    def counted(self: data.DroidWindowDataset, rel: str) -> Path:
        reads.append(rel)
        return real_file(self, rel)

    monkeypatch.setattr(data.DroidWindowDataset, "_file", counted)
    seen: list[tuple[int, int]] = []

    def encode(
        window: data.TrainingWindow,
        vae: object,
        text: object,
        config: dict,
        device: torch.device,
        window_id: int | None = None,
    ) -> EncodedWindow:
        seen.append((window.meta["episode"], window.meta["anchor"]))
        z = torch.zeros(1, 1)
        return EncodedWindow(z, z, [z], z, z, window_id=window_id)

    monkeypatch.setattr(script, "encode_window", encode)

    class FakeTrainer:
        rank = 0
        index = 0

        def window_index(self) -> int:
            return self.index

    settings = script.TrainSettings(seed=0)
    d = {"source": str(root), "split": "train", "overfit_windows": 4}
    trainer = FakeTrainer()
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        next_batch = script._overfit_batches(
            d, CONFIG, settings, trainer, None, None, torch.device("cpu")
        )
    finally:
        dist.destroy_process_group()
    ref = data.DroidWindowDataset(str(root), CONFIG, seed=0)
    assert seen == [(w.meta["episode"], w.meta["anchor"]) for w in _draws(ref, 4)]
    n_reads = len(reads)
    assert n_reads > 0
    ids = []
    for i in range(11):
        trainer.index = i
        ids.append(next_batch().window_id)
    assert ids == [i % 4 for i in range(11)]
    assert len(reads) == n_reads and len(seen) == 4


def test_window_frames_follow_video_antialias(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``video.antialias`` sets the tile resize of the training windows like inference."""
    import copy

    rng = np.random.default_rng(0)
    views = [rng.integers(0, 256, (93, 72, 96, 3), dtype=np.uint8) for _ in range(2)]
    reads = iter(views * 3)
    monkeypatch.setattr(data, "read_frames", lambda *a, **k: next(reads))

    def frames(antialias: bool | None) -> torch.Tensor:
        cfg = copy.deepcopy(CONFIG)
        if antialias is not None:
            cfg["video"]["antialias"] = antialias
        if antialias:
            cfg["video"].update(tile_height=224, tile_width=416)
        ds = data.DroidWindowDataset(str(root), cfg)
        return ds.load_window(ds.load_episode(2), 25).frames

    plain = tile_clip(*views, *TILE, 0.03)[0].permute(1, 0, 2, 3).to(torch.uint8)
    # The fit rule of an antialiased canvas: the praxis-1-droid tile, from the config.
    smooth = tile_clip(*views, 224, 416, 0.03, antialias=True)[0].permute(1, 0, 2, 3)
    smooth = smooth.to(torch.uint8)
    assert not torch.equal(plain, smooth)
    assert torch.equal(frames(None), plain)
    assert torch.equal(frames(False), plain)
    assert torch.equal(frames(True), smooth)
