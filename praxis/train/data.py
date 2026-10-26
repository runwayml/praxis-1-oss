"""Training windows of the single-arm checkpoint from the DROID dataset.

The dataset gives endless random training windows. The recipe matches the
DROID training data of the checkpoint:

* Episodes: the successful episodes with a language instruction and at least
  6.7 s of video, the 6.2 s window plus a margin of 0.5 s. At 15 Hz this is
  102 frames or more.
* Split: the episodes with ``episode_index % 50 == 0`` are the ``"val"``
  split. All other episodes are the ``"train"`` split.
* Window: 93 consecutive frames at 15 Hz in 4 chunks of 21, 24, 24 and 24
  frames (``chunking.chunk_geometry`` of the config).
* Start: the start time is uniform in ``[0, D - 6.2 s]``. ``D`` is the
  duration of the episode video, ``(T - 1) / 15 + 1 / 60`` s for ``T``
  frames (the last frame keeps the 1/60 s of the 60 fps raw file). The start
  then snaps down to the frame grid (:func:`draw_anchor`). Thus the first
  frame ``s`` is uniform over ``0 .. T - 95``, ``s = T - 94`` has a quarter
  of that weight, and no window holds the last frame. There is no idle trim.
* Frames: frame ``i`` of a video is the ``i``-th decoded frame
  (:func:`read_frames`). The raw DROID MP4 files have 60 fps time stamps and
  one frame per 15 Hz step.
* Canvas: each camera gets its own tile with the code of the serve path
  (:func:`praxis.single_arm.tile_clip`): a centre crop to the tile aspect
  (1280x720 becomes 1170x720), one bilinear resize without antialias to
  256x416 and a round to uint8. The exterior camera 1 goes left and the
  wrist camera right (256x832).
* Poses: DROID gives the flange pose ``[x, y, z, roll, pitch, yaw]`` with
  extrinsic x-y-z Euler angles. The model uses the flange frame without a
  turn (``pose_frame: "flange"``). :func:`droid_pose8` makes the
  quaternion with the operations of ``scipy.spatial.transform.Rotation``
  (``from_euler``, ``as_matrix``, ``from_matrix``, ``as_quat``), so the bits
  are those of the original data code. The gripper is clipped to [0, 1].
* Actions: the commanded pose of each frame relative to the observed pose
  at the first frame of its chunk (``xyz - xyz_anchor``,
  ``R @ R_anchor^T``), the gripper absolute. The proprio is the observed
  absolute pose at each chunk start. This is the inverse of the decode of
  the serve path (:func:`praxis.single_arm.rel10_to_wire`).
* Prompt: the language instruction of the episode (``tasks[0]`` in
  ``meta/episodes.jsonl``) without leading and trailing white space.

The windows are not normalized. ``praxis.train.objective.encode_window``
normalizes them with the statistics of the config. The text dropout and the
proprio dropout belong to the objective.

Differences from the original data code:

* Poses: the parquet files keep the poses in float32. The original data code
  read float64 poses. They differ by less than 2e-7.
* Split: the checkpoint was trained on most of the ``"val"`` episodes of
  this split. Thus a loss of the checkpoint on the ``"val"`` split is not a
  held-out loss.
* Order: each draw picks an episode uniformly (with replacement).

Source: a LeRobot v2.x directory or the Hugging Face dataset id
``cadene/droid_1.0.1``. The recipe needs the 1280x720 videos. The Hugging Face
copy has 320x180 videos: its canvas is an upscale with less detail. For the
recipe, keep the parquet files and ``meta/`` of the copy and put the raw
DROID MP4 file of each camera (``recordings/MP4/<serial>.mp4`` of the raw
release) at the video path of the episode, for example as a link. The section
"Data" of the README tells how to make this tree.

Use ``DataLoader(dataset, batch_size=None)``, because the dataset gives one
window per item. Each call of ``__iter__`` starts the same sequence again.
Use a new ``seed`` to get new windows, for example after a resume.
:func:`fixed_windows` gives the first windows of the sequence for the overfit
check.

With a dataset id, the dataset gets the files of an episode from the hub
when it draws the episode for the first time. The hub limits the request
rate. When a request fails, the dataset waits and tries again (``_file``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from av.error import FFmpegError
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from torch.utils.data import IterableDataset, get_worker_info

from praxis.robot_io import ASPECT_TOL, contiguous_chunk_geometry, pose8_to_10d
from praxis.single_arm import canvas_size, matrix_to_quat_xyzw, tile_clip

log = logging.getLogger(__name__)

# Video keys of the two canvas views, left to right.
VIEWS: tuple[str, ...] = (
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
)
# (pose, gripper) columns of the proprio and of the actions.
OBSERVED: tuple[str, str] = (
    "observation.state.cartesian_position",
    "observation.state.gripper_position",
)
COMMANDED: tuple[str, str] = ("action.cartesian_position", "action.gripper_position")
SUCCESS = "is_episode_successful"
TIMESTAMP = "timestamp"
ACTION_DIM = 10

# Every VAL_EVERY-th episode (by episode_index) is a validation episode.
VAL_EVERY = 50

# An episode needs this much video beyond the window (s).
MARGIN_S = 0.5
# The last frame of an episode video lasts one step of the 60 fps raw file (s).
LAST_FRAME_S = 1.0 / 60.0
# Tolerance of the time comparisons (s) and of the snap to the frame grid (frames).
TIME_EPS = 1e-9

# Row i of an episode must have the timestamp i / fps within this tolerance (s).
TIMESTAMP_TOL = 1e-3
# The iterator stops after this many failed episodes in sequence.
MAX_FAILURES = 20
# Tries of a hub download, and the longest wait between two tries (s).
HUB_TRIES = 12
HUB_MAX_WAIT = 120.0


@dataclass
class TrainingWindow:
    """One training window: the canvas frames, the prompt, the actions and the proprio.

    ``frames`` is ``(T, 3, H, W)`` uint8: the tile canvas of each window
    frame, as the video encoder gets it. ``actions_rel`` is ``(T, A)``: the
    commanded pose of each frame relative to the observed pose at the start
    of its chunk. ``proprio_abs`` is ``(K, A)``: the observed absolute pose at
    each chunk start. Both are float32 and not normalized. ``meta`` holds
    ``episode``, ``anchor`` (the first frame) and ``source``.
    """

    frames: torch.Tensor
    prompt: str
    actions_rel: np.ndarray
    proprio_abs: np.ndarray
    meta: dict


@dataclass
class DroidEpisode:
    """The pose streams of one episode as pose8 rows ``[xyz3, quat_wxyz4, grip1]``."""

    index: int
    prompt: str
    observed: np.ndarray  # (T, 8) float32
    commanded: np.ndarray  # (T, 8) float32


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


def _quat_mul_xyzw(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the products ``p * q`` of ``(N, 4)`` xyzw quaternions (scipy's order of ops)."""
    c0 = p[:, 1] * q[:, 2] - p[:, 2] * q[:, 1]
    c1 = p[:, 2] * q[:, 0] - p[:, 0] * q[:, 2]
    c2 = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    out = np.empty_like(p)
    out[:, 0] = p[:, 3] * q[:, 0] + q[:, 3] * p[:, 0] + c0
    out[:, 1] = p[:, 3] * q[:, 1] + q[:, 3] * p[:, 1] + c1
    out[:, 2] = p[:, 3] * q[:, 2] + q[:, 3] * p[:, 2] + c2
    out[:, 3] = p[:, 3] * q[:, 3] - p[:, 0] * q[:, 0] - p[:, 1] * q[:, 1]
    out[:, 3] -= p[:, 2] * q[:, 2]
    return out


def euler_xyz_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3)`` extrinsic x-y-z Euler angles to ``(N, 4)`` xyzw quaternions.

    The rotation is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The operations are
    those of ``scipy.spatial.transform.Rotation.from_euler("xyz", rpy)``, so
    the result has the same bits. The result is not normalized again.
    """
    e = np.asarray(rpy, dtype=np.float64).reshape(-1, 3)
    q = np.zeros((e.shape[0], 4))
    for axis in range(3):
        half = e[:, axis] / 2
        elem = np.zeros((e.shape[0], 4))
        elem[:, axis] = np.sin(half)
        elem[:, 3] = np.cos(half)
        q = elem if axis == 0 else _quat_mul_xyzw(elem, q)
    return q


def quat_xyzw_to_matrix_raw(q: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` xyzw quaternions to ``(N, 3, 3)`` matrices, no normalization.

    The operations are those of ``Rotation.as_matrix`` of a rotation from
    ``from_euler`` (scipy keeps that quaternion as it is).
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = x2 - y2 - z2 + w2
    m[:, 1, 0] = 2 * (xy + zw)
    m[:, 2, 0] = 2 * (xz - yw)
    m[:, 0, 1] = 2 * (xy - zw)
    m[:, 1, 1] = -x2 + y2 - z2 + w2
    m[:, 2, 1] = 2 * (yz + xw)
    m[:, 0, 2] = 2 * (xz + yw)
    m[:, 1, 2] = 2 * (yz - xw)
    m[:, 2, 2] = -x2 - y2 + z2 + w2
    return m


def droid_pose8(cartesian: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert DROID poses to pose8 rows ``[xyz3, quat_wxyz4, grip1]`` (float32).

    ``cartesian`` is ``(T, 6)``: the flange pose ``[x, y, z, roll, pitch,
    yaw]`` with extrinsic x-y-z Euler angles. ``gripper`` is ``(T,)``. The
    pose stays in the flange frame (``pose_frame: "flange"``). The quaternion
    comes from the Euler angles, the matrix and back
    (:func:`praxis.single_arm.matrix_to_quat_xyzw`), with the operations and
    the sign of ``scipy.spatial.transform``. The function clips the gripper
    to [0, 1].
    """
    pose6 = np.asarray(cartesian, dtype=np.float64).reshape(-1, 6)
    rot = quat_xyzw_to_matrix_raw(euler_xyz_to_quat_xyzw(pose6[:, 3:6]))
    quat = matrix_to_quat_xyzw(rot)[:, [3, 0, 1, 2]]
    grip = np.clip(np.asarray(gripper, dtype=np.float64).reshape(-1), 0.0, 1.0)
    return np.concatenate([pose6[:, :3], quat, grip[:, None]], axis=1).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Window start
# ---------------------------------------------------------------------------


def episode_duration(n_frames: int, fps: float) -> float:
    """Return the video duration (s) of an episode with ``n_frames`` frames.

    The frames are ``1 / fps`` apart, and the last frame lasts
    :data:`LAST_FRAME_S`.
    """
    return (int(n_frames) - 1) / float(fps) + LAST_FRAME_S


def window_fits(n_frames: int, window_frames: int, fps: float) -> bool:
    """Return True when an episode holds a window plus :data:`MARGIN_S` of video."""
    need = window_frames / float(fps) + MARGIN_S
    return episode_duration(n_frames, fps) >= need - TIME_EPS


def draw_anchor(
    n_frames: int, window_frames: int, fps: float, rng: np.random.Generator
) -> int:
    """Draw the first frame of a window of an episode with ``n_frames`` frames.

    The start time is uniform in ``[0, duration - window]`` (s). It then
    snaps down to the frame grid, and the window stays inside the episode.
    Raises ``ValueError`` when the episode does not hold the window plus
    :data:`MARGIN_S`.
    """
    if not window_fits(n_frames, window_frames, fps):
        raise ValueError(f"{n_frames} frames do not hold a window of {window_frames}")
    span = max(episode_duration(n_frames, fps) - window_frames / float(fps), 0.0)
    start = float(rng.uniform(0.0, span))
    anchor = math.floor(start * float(fps) + TIME_EPS)
    return int(min(max(anchor, 0), int(n_frames) - int(window_frames)))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def read_frames(path: Path, start: int, stop: int) -> np.ndarray:
    """Decode the frames ``[start, stop)`` of a video file as ``(T, H, W, 3)`` uint8 RGB.

    Frame ``i`` is the ``i``-th decoded frame (decode order). This works for
    videos with one frame per step at any container rate (the Hugging Face
    copy stores 15 fps, the DROID raw MP4 files 50-60 fps time stamps). The
    decoder reads from the start of the file, because many raw files have one
    key frame only. Raises ``ValueError`` when a frame of the range is missing.
    """
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i >= stop:
                break
            if i >= start:
                out.append(frame.to_ndarray(format="rgb24"))
    if len(out) != stop - start:
        raise ValueError(f"{path}: {len(out)} of the frames [{start}, {stop}) decoded")
    return np.stack(out)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _column(table: pa.Table, name: str, width: int) -> np.ndarray:
    """Return a numeric column as float64 ``(T, width)``. Bad values raise ``ValueError``."""
    x = np.asarray(table.column(name).to_pylist(), dtype=np.float64)
    x = x.reshape(x.shape[0], -1)
    if x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError(f"column {name} has the shape {x.shape} or bad values")
    return x


class DroidWindowDataset(IterableDataset):
    """Endless random training windows of DROID episodes.

    ``source`` is the Hugging Face dataset id ``"cadene/droid_1.0.1"`` or the
    directory of a copy in the LeRobot v2.x layout. For a dataset id, the
    dataset gets ``meta/info.json`` and ``meta/episodes.jsonl`` when it
    starts. It gets the parquet file and the two video files of an episode
    when it draws the episode for the first time. ``cache_dir`` is the Hugging
    Face cache directory. None selects the default cache.

    ``config`` is the checkpoint config (``config.json``). It gives the chunk
    geometry, the frame rate, the tile size and the action dim.

    The episode pool holds the episodes of ``split`` with a task and enough
    frames for a window plus :data:`MARGIN_S` (:func:`window_fits`). For a directory, the pool holds only the episodes with all three
    files: the parquet file and the two videos. Thus a part of the dataset
    is sufficient. ``max_episodes`` keeps the first episodes of the pool.
    Rank ``rank`` of ``world_size`` takes every ``world_size``-th episode of
    the pool. The random draws of a worker depend on ``(seed, rank, worker
    id)``.

    Each window draw does these steps:

    1. Draw an episode from the pool.
    2. Read its poses and do the episode checks (:meth:`load_episode`).
    3. Draw the first frame of the window (:func:`draw_anchor`).
    4. Decode the frames of the two cameras and make the window
       (:meth:`load_window`).

    When a step fails, the episode leaves the pool and the draw starts again.
    A hub error does not remove an episode, except a missing file
    (:meth:`_file`).
    """

    def __init__(
        self,
        source: str,
        config: dict,
        *,
        split: str = "train",
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        cache_dir: str | None = None,
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in [0, {world_size})")
        if int(config["action_space"]["dim"]) != ACTION_DIM:
            raise ValueError(f"DROID needs a {ACTION_DIM}-D action space")
        chunking = config["chunking"]
        self.action_chunk = int(chunking["action_chunk"])
        n_chunks = int(chunking["n_chunks"])
        self.geometry = contiguous_chunk_geometry(self.action_chunk, n_chunks)
        declared = chunking.get("chunk_geometry")
        if declared is not None and [tuple(g) for g in declared] != self.geometry:
            raise ValueError(
                f"chunking.chunk_geometry {declared} is not the contiguous "
                f"geometry {self.geometry}"
            )
        self.window_frames = sum(self.geometry[-1])
        if int(chunking.get("window_frames", self.window_frames)) != self.window_frames:
            raise ValueError(f"chunking.window_frames is not {self.window_frames}")
        self.fps = float(chunking["grid_hz"])
        tile_h, width = canvas_size(config)
        self.tile = (tile_h, width // len(VIEWS))
        self.aspect_tol = float(config["video"].get("aspect_tol", ASPECT_TOL))
        if config.get("pose_frame") != "flange":
            raise ValueError("DROID windows need the config pose_frame 'flange'")
        self.source = str(source)
        self.seed, self.rank = int(seed), int(rank)
        self.cache_dir = cache_dir
        self.root = Path(source) if Path(source).is_dir() else None
        self.revision = None
        if self.root is None:
            # A fixed commit lets the cache serve the files without a request.
            self.revision = HfApi().dataset_info(self.source).sha

        info = json.loads(self._file("meta/info.json").read_text())
        if not str(info["codebase_version"]).startswith("v2"):
            raise ValueError(f"{source} is not a LeRobot v2.x dataset")
        if float(info["fps"]) != self.fps:
            raise ValueError(f"{source} has {info['fps']} fps, the model {self.fps}")
        if missing := {*VIEWS, *OBSERVED, *COMMANDED, SUCCESS, TIMESTAMP} - set(
            info["features"]
        ):
            raise ValueError(f"{source} has no feature {sorted(missing)}")
        self.data_path = str(info["data_path"])
        self.video_path = str(info["video_path"])
        self.chunks_size = int(info["chunks_size"])

        pool: list[tuple[int, str]] = []
        for line in self._file("meta/episodes.jsonl").read_text().splitlines():
            ep = json.loads(line)
            index = int(ep["episode_index"])
            task = str(ep["tasks"][0]).strip() if ep["tasks"] else ""
            if (index % VAL_EVERY == 0) != (split == "val"):
                continue
            if not task or not window_fits(ep["length"], self.window_frames, self.fps):
                continue
            if self.root is not None and not all(
                (self.root / rel).is_file() for rel in self._files(index)
            ):
                continue
            pool.append((index, task))
        if max_episodes is not None:
            pool = pool[: int(max_episodes)]
        self.prompts = dict(pool[rank::world_size])
        self.episodes = list(self.prompts)
        if not self.episodes:
            raise ValueError(f"rank {rank} has no {split} episode of {source}")
        log.info(
            "%s %s: %d episodes in the pool, %d on rank %d",
            source,
            split,
            len(pool),
            len(self.episodes),
            rank,
        )

    def _data(self, index: int) -> str:
        return self.data_path.format(
            episode_chunk=index // self.chunks_size, episode_index=index
        )

    def _video(self, index: int, view: str) -> str:
        return self.video_path.format(
            episode_chunk=index // self.chunks_size, video_key=view, episode_index=index
        )

    def _files(self, index: int) -> list[str]:
        """Return the parquet file and the two video files of an episode."""
        return [self._data(index), *(self._video(index, v) for v in VIEWS)]

    def _file(self, rel: str) -> Path:
        """Return the local path of a dataset file. Download the file when necessary.

        A file that is not on the hub raises ``FileNotFoundError``. Thus its
        episode leaves the pool. For other errors (network errors, the rate
        limit, server errors), the function waits and tries the download
        again. After ``HUB_TRIES`` tries it raises ``RuntimeError``, and the
        iteration stops.
        """
        if self.root is not None:
            return self.root / rel
        for attempt in range(HUB_TRIES):
            try:
                path = hf_hub_download(
                    self.source,
                    rel,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                )
                return Path(path)
            except RemoteEntryNotFoundError as err:
                raise FileNotFoundError(f"{rel} is not in {self.source}") from err
            except Exception as err:
                # The hub client can raise errors that are not an OSError,
                # for example network errors.
                if attempt + 1 == HUB_TRIES:
                    raise RuntimeError(
                        f"cannot get {rel} of {self.source} in {HUB_TRIES} tries"
                    ) from err
                wait = min(2.0**attempt, HUB_MAX_WAIT)
                log.warning("Cannot get %s (%s). Next try in %.0f s.", rel, err, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def load_episode(self, index: int) -> DroidEpisode:
        """Read the pose streams of one episode and do the episode checks.

        Raises ``ValueError`` in these cases:

        * The episode is not successful.
        * Row ``i`` does not have the timestamp ``i / fps``.
        * A pose has bad values.
        """
        table = pq.read_table(
            self._file(self._data(index)),
            columns=[*OBSERVED, *COMMANDED, SUCCESS, TIMESTAMP],
        )
        if not all(table.column(SUCCESS).to_pylist()):
            raise ValueError(f"episode {index} is not successful")
        t = _column(table, TIMESTAMP, 1)[:, 0]
        if np.abs(t - np.arange(t.shape[0]) / self.fps).max() > TIMESTAMP_TOL:
            raise ValueError(f"episode {index}: row i is not at the time i / fps")
        return DroidEpisode(
            index=index,
            prompt=self.prompts[index],
            observed=droid_pose8(
                _column(table, OBSERVED[0], 6), _column(table, OBSERVED[1], 1)
            ),
            commanded=droid_pose8(
                _column(table, COMMANDED[0], 6), _column(table, COMMANDED[1], 1)
            ),
        )

    def load_window(self, episode: DroidEpisode, anchor: int) -> TrainingWindow:
        """Build the window of ``episode`` that starts at frame ``anchor``."""
        stop = anchor + self.window_frames
        if anchor < 0 or stop > episode.observed.shape[0]:
            raise ValueError(f"episode {episode.index}: no window at frame {anchor}")
        ext, wrist = (
            read_frames(self._file(self._video(episode.index, v)), anchor, stop)
            for v in VIEWS
        )
        clip = tile_clip(ext, wrist, *self.tile, self.aspect_tol)
        obs, cmd = episode.observed, episode.commanded
        starts = [anchor + off for off, _ in self.geometry]
        actions = [
            pose8_to_10d(cmd[sk : sk + ln], anchor8=obs[sk])
            for sk, (_, ln) in zip(starts, self.geometry, strict=True)
        ]
        return TrainingWindow(
            frames=clip[0].permute(1, 0, 2, 3).to(torch.uint8).contiguous(),
            prompt=episode.prompt,
            actions_rel=np.concatenate(actions, axis=0),
            proprio_abs=pose8_to_10d(obs[starts]),
            meta={"episode": episode.index, "anchor": anchor, "source": self.source},
        )

    def __iter__(self) -> Iterator[TrainingWindow]:
        info = get_worker_info()
        worker = 0 if info is None else int(info.id)
        rng = np.random.default_rng((self.seed, self.rank, worker))
        pool = list(self.episodes)
        while True:
            for _ in range(MAX_FAILURES):
                index = pool[int(rng.integers(len(pool)))]
                try:
                    episode = self.load_episode(index)
                    n = episode.observed.shape[0]
                    anchor = draw_anchor(n, self.window_frames, self.fps, rng)
                    window = self.load_window(episode, anchor)
                    break
                except (OSError, ValueError, FFmpegError) as err:
                    log.warning("Episode %d leaves the pool: %s", index, err)
                    pool.remove(index)
                    if not pool:
                        raise RuntimeError("no episode is left in the pool") from err
            else:
                raise RuntimeError(f"{MAX_FAILURES} episodes in sequence failed")
            yield window


def fixed_windows(dataset: DroidWindowDataset, n: int) -> list[TrainingWindow]:
    """Return the first ``n`` windows of ``iter(dataset)`` (the overfit check).

    The function reads the files of these windows once. The caller keeps the
    windows and uses them again and again. Call it outside a ``DataLoader``:
    the windows then come from the generator of worker 0.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    it = iter(dataset)
    return [next(it) for _ in range(n)]
