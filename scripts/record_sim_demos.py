"""Record scripted-oracle demonstrations of a simulation task as a LeRobot v2.1 dataset.

The dataset has the layout and the columns of ``cadene/droid_1.0.1`` that
``praxis.train.data.DroidWindowDataset`` reads with its default ``data.spec``,
so ``configs/finetune_droid_lora.yaml`` trains on it with only
``data.source`` changed::

    MUJOCO_GL=egl python scripts/record_sim_demos.py --task two_blocks_in_bowl \\
        --episodes 300 --first-seed 1000 --out data/sim_two_blocks_in_bowl

    torchrun --nproc_per_node 1 scripts/finetune.py \\
        --config configs/finetune_droid_lora.yaml \\
        --set data.source=data/sim_two_blocks_in_bowl

The loop of one episode is the loop of ``scripts/run_sim_eval.py`` with the
scripted oracle as the policy: reset the scene with the seed, call the
oracle for 21 rows, then for 24 rows per call, execute one row per 15 Hz
tick and render the two views after each tick. The episode stops 1 s after
the first success or at the time limit of the task.

Rows. Row ``i`` holds the observation the policy saw before tick ``i`` and
the command it sent for tick ``i``:

* ``observation.images.exterior_1_left``, ``observation.images.wrist_left``:
  the two model views as rendered (512x288, 16:9), frame ``i`` = row ``i``,
  15 fps H.264 (``yuv420p``, CRF ``--crf``).
* ``observation.state.cartesian_position`` and ``.gripper_position``: the
  measured pose and grip (0 open, 1 closed).
* ``action.cartesian_position`` and ``.gripper_position``: the commanded
  target pose and grip of the row.
* ``is_episode_successful`` (true: only successful episodes are written),
  and at least 102 rows (a fast episode holds its final oracle pose after
  the success until it has 102 rows; see :func:`record_episode`),
  ``timestamp`` = ``i / 15``, ``frame_index``, ``episode_index``, ``index``
  and ``task_index``.

Poses. The sim reports the ``ee`` site (``RIG_FRAMES["praxis.sim"]``: the
Franka flange turned -45 deg about its z axis). The model trains and serves
in the flange frame with no turn (``pose_frame: "flange"``). The script
converts every observed and commanded pose with
:func:`praxis.single_arm.pose8_to_model` before it writes it, as the serve
session converts the observations of the sim. The rotation is stored as
extrinsic x-y-z Euler angles (radians), the DROID convention; the data code
converts them back to the same quaternion (``rig_pose8``) to float32
precision.

Seeds. Scene seed ``first_seed + k`` places the objects of episode ``k``.
``praxis/sim/EVAL.md`` keeps scene seeds 0 to 15 for the evaluation, so the
script refuses seeds below 1000 unless ``--allow-eval-seeds`` is set. A
failed oracle episode is skipped and the next seed is tried, so the seeds
of the dataset are not always contiguous: ``meta/episodes.jsonl`` keeps the
seed of each episode.

Variation (``--vary``). Identical demonstrations teach a policy less than
varied ones. With ``--vary`` each episode draws its oracle settings from
``(seed, "vary")``: the free-move speed in ``--speed``, the approach height in
``--approach``, and the descend, close and lift durations scaled by a factor
in ``--timing``. The oracle stays the task oracle; only successful episodes
are kept.

The script needs the ``sim`` and ``train`` extras (MuJoCo, PyAV, pyarrow)
and an offscreen OpenGL context (``MUJOCO_GL=egl`` on a GPU machine). It
needs no model weights. ``--workers N`` records in N processes (one scene
each); the episodes are numbered in seed order at the end, so the result
does not depend on N.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401
from praxis.robot_io import quat_wxyz_to_matrix
from praxis.sim.config import load_config
from praxis.sim.env import Observation, SimEnv
from praxis.sim.scene import CONTROL_HZ
from praxis.sim.scripted import ScriptedConfig, ScriptedPick
from praxis.single_arm import (
    RIG_FRAMES,
    matrix_to_quat_xyzw,
    pose8_to_model,
    wire_to_arrays,
)

FPS = int(CONTROL_HZ)
CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
EXTERIOR = "observation.images.exterior_1_left"
WRIST = "observation.images.wrist_left"
OBSERVED_POSE = "observation.state.cartesian_position"
OBSERVED_GRIP = "observation.state.gripper_position"
COMMANDED_POSE = "action.cartesian_position"
COMMANDED_GRIP = "action.gripper_position"
SUCCESS = "is_episode_successful"
# The eval suites keep scene seeds 0 to 15 (praxis/sim/EVAL.md).
FIRST_TRAIN_SEED = 1000
# The training windows need 6.2 s plus a 0.5 s margin: 102 rows at 15 Hz.
MIN_ROWS = 102
# Oracle failures in a row that stop the run (a broken task or scene).
MAX_FAILURES_IN_A_ROW = 20
AXES = ["x", "y", "z", "roll", "pitch", "yaw"]


@dataclass(frozen=True)
class Variation:
    """The ranges of the oracle settings that ``--vary`` draws per episode."""

    speed: tuple[float, float] = (0.11, 0.19)  # metres per second
    approach: tuple[float, float] = (0.09, 0.15)  # metres above the grasp
    timing: tuple[float, float] = (0.8, 1.25)  # factor of descend, close, lift


@dataclass
class Episode:
    """The rows of one recorded episode, before the episode index is known."""

    seed: int
    prompt: str
    observed: np.ndarray  # (T, 8) pose8 in the model frame
    commanded: np.ndarray  # (T, 8) pose8 in the model frame
    exterior: np.ndarray  # (T, H, W, 3) uint8
    wrist: np.ndarray  # (T, H, W, 3) uint8
    success_tick: int | None
    oracle: dict[str, float]


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


def matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    """Return the extrinsic x-y-z Euler angles ``(N, 3)`` of ``(N, 3, 3)`` rotations.

    ``rot = Rz(yaw) @ Ry(pitch) @ Rx(roll)``, the DROID convention that
    :func:`praxis.train.data.euler_xyz_to_quat_xyzw` reads. Pitch is in
    ``[-pi/2, pi/2]``. Near the gimbal lock (``|pitch| = pi/2``) the roll is
    set to 0 and the yaw takes the whole turn about z.
    """
    r = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
    sin_pitch = np.clip(-r[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.sqrt(np.maximum(1.0 - sin_pitch**2, 0.0))
    lock = cos_pitch < 1e-9
    roll = np.where(lock, 0.0, np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    yaw = np.where(
        lock,
        np.arctan2(-r[:, 0, 1], r[:, 1, 1]),
        np.arctan2(r[:, 1, 0], r[:, 0, 0]),
    )
    return np.stack([roll, pitch, yaw], axis=1)


def pose8_to_droid(pose8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split pose8 rows ``(T, 8)`` into DROID cartesian ``(T, 6)`` and gripper ``(T,)`` (float32)."""
    p = np.asarray(pose8, dtype=np.float64).reshape(-1, 8)
    euler = matrix_to_euler_xyz(quat_wxyz_to_matrix(p[:, 3:7]))
    cart = np.concatenate([p[:, :3], euler], axis=1).astype(np.float32)
    return cart, p[:, 7].astype(np.float32)


def obs_pose8(obs: Observation) -> np.ndarray:
    """The observed pose8 ``[xyz, quat_wxyz, grip]`` of the sim, in the rig frame."""
    return np.concatenate([obs.ee_pose, [obs.grip]]).astype(np.float32)


def pose12_to_pose8(pose12: np.ndarray, grip: float) -> np.ndarray:
    """A commanded ``[xyz3, R_flat9]`` pose and grip as pose8 (rig frame)."""
    p = np.asarray(pose12, dtype=np.float64).reshape(12)
    quat = matrix_to_quat_xyzw(p[3:].reshape(1, 3, 3))[0, [3, 0, 1, 2]]
    return np.concatenate([p[:3], quat, [grip]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def oracle_config(seed: int, vary: Variation | None) -> ScriptedConfig:
    """The oracle settings of the episode with scene seed ``seed``."""
    cfg = ScriptedConfig()
    if vary is None:
        return cfg
    rng = np.random.default_rng([seed, 0x7661_7279])  # (seed, "vary")
    k = float(rng.uniform(*vary.timing))
    return dataclasses.replace(
        cfg,
        speed=float(rng.uniform(*vary.speed)),
        approach_height=float(rng.uniform(*vary.approach)),
        descend_s=cfg.descend_s * k,
        close_s=cfg.close_s * k,
        lift_s=cfg.lift_s * k,
    )


def record_episode(
    env: SimEnv,
    seed: int,
    oracle: ScriptedConfig,
    after_success_s: float = 1.0,
    min_rows: int = MIN_ROWS,
) -> Episode | None:
    """Run the oracle on scene seed ``seed`` and return its rows, or None on a failure.

    The loop is the loop of :func:`praxis.sim.loop.run_episode` with the
    scripted policy. Row ``i`` pairs the observation before tick ``i`` with
    the command of tick ``i``. The rows end at the last executed command:
    ``after_success_s`` after the first success and at least ``min_rows``
    rows, or at the time limit. A fast episode thus holds the last oracle
    row (the arm keeps the final pose) until it has ``min_rows`` rows, the
    length that a training window needs.
    """
    obs = env.reset(seed)
    pick = ScriptedPick(oracle)
    pick.reset(env)
    frame = RIG_FRAMES["praxis.sim"]
    linger = round(after_success_s * CONTROL_HZ)
    observed: list[np.ndarray] = []
    commanded: list[np.ndarray] = []
    exterior: list[np.ndarray] = []
    wrist: list[np.ndarray] = []
    success_at: int | None = None
    done = False
    while not done and env.tick_count < env.max_ticks:
        poses, grips = wire_to_arrays(pick.step(obs))
        for pose12, grip in zip(poses, grips, strict=True):
            if obs.exterior is None or obs.wrist is None:
                raise RuntimeError("the recorder needs the rendered views")
            observed.append(obs_pose8(obs))
            commanded.append(pose12_to_pose8(pose12, float(grip)))
            exterior.append(obs.exterior)
            wrist.append(obs.wrist)
            obs = env.step(pose12, float(grip))
            if env.task.status.success and success_at is None:
                success_at = env.tick_count
            if env.tick_count >= env.max_ticks or (
                success_at is not None
                and env.tick_count - success_at >= linger
                and len(observed) >= min_rows
            ):
                done = True
                break
    if success_at is None:
        return None
    return Episode(
        seed=seed,
        prompt=env.prompt,
        observed=pose8_to_model(np.stack(observed), frame),
        commanded=pose8_to_model(np.stack(commanded), frame),
        exterior=np.stack(exterior),
        wrist=np.stack(wrist),
        success_tick=success_at,
        oracle={
            "speed": oracle.speed,
            "approach_height": oracle.approach_height,
            "descend_s": oracle.descend_s,
            "close_s": oracle.close_s,
            "lift_s": oracle.lift_s,
        },
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_video(path: Path, frames: np.ndarray, crf: int) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames as a 15 fps H.264 MP4 (one frame per row)."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS, options={"crf": str(crf)})
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        for img in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(img, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_episode(
    root: Path, index: int, task_index: int, first_row: int, ep: Episode, crf: int
) -> None:
    """Write the parquet file and the two videos of one episode."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = ep.observed.shape[0]
    obs_cart, obs_grip = pose8_to_droid(ep.observed)
    cmd_cart, cmd_grip = pose8_to_droid(ep.commanded)

    def vec(x: np.ndarray) -> pa.Array:
        return pa.array(x.tolist(), pa.list_(pa.float32(), x.shape[1]))

    def one(x: np.ndarray) -> pa.Array:
        return pa.array(x.reshape(-1, 1).tolist(), pa.list_(pa.float32(), 1))

    table = pa.table(
        {
            OBSERVED_POSE: vec(obs_cart),
            OBSERVED_GRIP: one(obs_grip),
            COMMANDED_POSE: vec(cmd_cart),
            COMMANDED_GRIP: one(cmd_grip),
            SUCCESS: pa.array([True] * n),
            "language_instruction": pa.array([ep.prompt] * n),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, index, dtype=np.int64)),
            "index": pa.array(np.arange(first_row, first_row + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        }
    )
    chunk = index // CHUNKS_SIZE
    path = root / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    for key, frames in ((EXTERIOR, ep.exterior), (WRIST, ep.wrist)):
        rel = VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=index)
        write_video(root / rel, frames, crf)


def info_json(
    n_episodes: int, n_frames: int, n_tasks: int, hw: tuple[int, int]
) -> dict:
    """``meta/info.json`` in the layout of ``cadene/droid_1.0.1``."""
    h, w = hw
    video = {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.fps": float(FPS),
            "video.height": h,
            "video.width": w,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    def vec(n: int, axes: list[str]) -> dict:
        return {"dtype": "float32", "shape": [n], "names": {"axes": axes}}

    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "Franka",
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": n_tasks,
        "total_videos": 2 * n_episodes,
        "total_chunks": max(1, math.ceil(n_episodes / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            EXTERIOR: video,
            WRIST: video,
            OBSERVED_POSE: vec(6, AXES),
            OBSERVED_GRIP: vec(1, ["gripper"]),
            COMMANDED_POSE: vec(6, AXES),
            COMMANDED_GRIP: vec(1, ["gripper"]),
            SUCCESS: {"dtype": "bool", "shape": [1], "names": None},
            "language_instruction": {"dtype": "string", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


DATASET_CARD = """\
---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- praxis-1
- simulation
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# {name}

Scripted-oracle demonstrations of the praxis-sim task `{task}` ("{prompt}") on
the simulated DROID rig (Franka Panda + Robotiq 2F-85, an exterior and a wrist
camera at 512x288, 15 Hz). Made with `scripts/record_sim_demos.py` of
praxis-1 for the LoRA tutorial of
`praxis-1-droid` (`praxis/train/README.md`).

- {episodes} successful episodes, {frames} frames, from scene seeds {seeds}
  (the lowest successful seeds; `meta/episodes.jsonl` keeps the seed and the
  oracle settings of each episode). Scene seeds 0-15 are the evaluation seeds
  of `praxis/sim/EVAL.md` and are not in the data.
- LeRobot v2.1 in the layout and the column names of `cadene/droid_1.0.1`:
  `observation.images.exterior_1_left`, `observation.images.wrist_left`,
  `observation.state.cartesian_position` / `.gripper_position` (measured),
  `action.cartesian_position` / `.gripper_position` (the commanded target of
  the row), `is_episode_successful`, `timestamp`.
- Poses: the Franka flange in the robot base frame, metres, extrinsic x-y-z
  Euler angles in radians (the DROID convention, the `flange` pose frame of
  praxis-1). Gripper 0 = open, 1 = closed.
- Oracle variation: {vary}.

Regenerate:

```bash
{command}
```
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """The work of one recorder process."""

    task: str
    seeds: tuple[int, ...]
    config: str
    sets: tuple[str, ...]
    vary: Variation | None
    stage: str
    crf: int
    want: int


def _record_worker(job: Job, queue: mp.Queue) -> None:
    """Record the seeds of ``job`` and report each result on ``queue``."""
    try:
        _record_seeds(job, queue)
    except BaseException:
        queue.put(("error", -1, 0, 0.0))
        raise


def _record_seeds(job: Job, queue: mp.Queue) -> None:
    cfg = load_config(
        job.config,
        ["viewer.enabled=false", "policy.type=scripted", *job.sets],
    )
    cfg.task.name = job.task
    cfg.task.prompt = None
    cfg.task.time_limit_s = None
    stage = Path(job.stage)
    failures = 0
    with SimEnv(cfg.env_config()) as env:
        for seed in job.seeds:
            if staged_below(stage, seed) >= job.want:
                break
            t0 = time.monotonic()
            ep = record_episode(env, seed, oracle_config(seed, job.vary))
            wall = time.monotonic() - t0
            if ep is None or ep.observed.shape[0] < MIN_ROWS:
                failures += 1
                rows = 0 if ep is None else int(ep.observed.shape[0])
                kind = "fail" if ep is None else "short"
                queue.put((kind, seed, rows, wall))
                if failures >= MAX_FAILURES_IN_A_ROW:
                    queue.put(("abort", seed, 0, wall))
                    return
                continue
            failures = 0
            # Stage the episode under its seed; the main process numbers them.
            tmp = stage / f"seed_{seed:07d}"
            write_episode(tmp, 0, 0, 0, ep, job.crf)
            meta = {
                "seed": seed,
                "prompt": ep.prompt,
                "length": int(ep.observed.shape[0]),
                "success_tick": ep.success_tick,
                "oracle": ep.oracle,
                "record_s": round(wall, 2),
            }
            (tmp / "episode.json").write_text(json.dumps(meta))
            queue.put(("ok", seed, meta["length"], wall))
    queue.put(("done", -1, 0, 0.0))


def staged_below(stage: Path, seed: int) -> int:
    """The number of staged episodes with a scene seed below ``seed``.

    A worker stops at the first seed with ``want`` staged episodes below it.
    Thus every seed below the ``want``-th successful seed is tried by its
    worker, and the kept episodes are the ``want`` lowest successful seeds
    for any number of workers.
    """
    return sum(
        1 for p in stage.glob("seed_*/episode.json") if int(p.parent.name[5:]) < seed
    )


def _stride(seeds: Sequence[int], workers: int) -> Iterator[tuple[int, ...]]:
    for w in range(workers):
        yield tuple(seeds[w::workers])


def assemble(stage: Path, out: Path, want: int, card: dict[str, str]) -> dict:
    """Number the staged episodes in seed order and write the dataset to ``out``."""
    staged = sorted(stage.glob("seed_*/episode.json"))[:want]
    tasks: dict[str, int] = {}
    episodes = []
    first_row = 0
    hw: tuple[int, int] | None = None
    for index, meta_path in enumerate(staged):
        meta = json.loads(meta_path.read_text())
        src = meta_path.parent
        task_index = tasks.setdefault(meta["prompt"], len(tasks))
        chunk = index // CHUNKS_SIZE
        _relocate(src, out, index, chunk, task_index, first_row)
        if hw is None:
            hw = _video_hw(
                out
                / VIDEO_PATH.format(
                    episode_chunk=chunk, video_key=EXTERIOR, episode_index=index
                )
            )
        episodes.append(
            {
                "episode_index": index,
                "tasks": [meta["prompt"]],
                "length": meta["length"],
                "scene_seed": meta["seed"],
                "success_tick": meta["success_tick"],
                "oracle": meta["oracle"],
            }
        )
        first_row += meta["length"]
    if not episodes or hw is None:
        raise RuntimeError("no episode was recorded")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    info = info_json(len(episodes), first_row, len(tasks), hw)
    (out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes)
    )
    (out / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()
        )
    )
    seeds = [e["scene_seed"] for e in episodes]
    (out / "README.md").write_text(
        DATASET_CARD.format(
            episodes=len(episodes),
            frames=first_row,
            seeds=f"{min(seeds)}-{max(seeds)}",
            prompt=episodes[0]["tasks"][0],
            **card,
        )
    )
    return {
        "episodes": len(episodes),
        "frames": first_row,
        "seeds": [min(seeds), max(seeds)],
    }


def _relocate(
    src: Path, out: Path, index: int, chunk: int, task_index: int, first_row: int
) -> None:
    """Move one staged episode to its index, rewriting its index columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src / DATA_PATH.format(episode_chunk=0, episode_index=0))
    n = table.num_rows
    table = table.set_column(
        table.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array(np.full(n, index, dtype=np.int64)),
    )
    table = table.set_column(
        table.schema.get_field_index("index"),
        "index",
        pa.array(np.arange(first_row, first_row + n, dtype=np.int64)),
    )
    table = table.set_column(
        table.schema.get_field_index("task_index"),
        "task_index",
        pa.array(np.full(n, task_index, dtype=np.int64)),
    )
    dst = out / DATA_PATH.format(episode_chunk=chunk, episode_index=index)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    for key in (EXTERIOR, WRIST):
        v = out / VIDEO_PATH.format(
            episode_chunk=chunk, video_key=key, episode_index=index
        )
        v.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(
            str(
                src / VIDEO_PATH.format(episode_chunk=0, video_key=key, episode_index=0)
            ),
            v,
        )


def _video_hw(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return int(s.height), int(s.width)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, help="a praxis-sim task with an oracle")
    ap.add_argument(
        "--episodes", type=int, required=True, help="successful episodes to keep"
    )
    ap.add_argument("--first-seed", type=int, default=FIRST_TRAIN_SEED)
    ap.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="seeds to try (default: 2 x --episodes)",
    )
    ap.add_argument(
        "--out", required=True, help="the LeRobot tree to write (must not exist)"
    )
    ap.add_argument("--config", default="configs/sim_droid.yaml")
    ap.add_argument(
        "--set", action="append", default=[], help="key.sub=value of the sim config"
    )
    ap.add_argument(
        "--vary", action="store_true", help="draw the oracle settings per episode"
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality of the videos")
    ap.add_argument("--allow-eval-seeds", action="store_true")
    args = ap.parse_args(argv)

    if args.first_seed < FIRST_TRAIN_SEED and not args.allow_eval_seeds:
        raise SystemExit(
            f"scene seeds below {FIRST_TRAIN_SEED} are kept for the evaluation "
            "(praxis/sim/EVAL.md); use --allow-eval-seeds to record them anyway"
        )
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    stage = out.with_name(out.name + ".staging")
    if stage.exists():
        raise SystemExit(
            f"{stage} exists (left by an earlier run); delete it to record again"
        )
    stage.mkdir(parents=True)
    n_seeds = args.max_seeds or 2 * args.episodes
    seeds = list(range(args.first_seed, args.first_seed + n_seeds))
    vary = Variation() if args.vary else None
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_record_worker,
            args=(
                Job(
                    args.task,
                    part,
                    args.config,
                    tuple(args.set),
                    vary,
                    str(stage),
                    args.crf,
                    args.episodes,
                ),
                queue,
            ),
        )
        for part in _stride(seeds, args.workers)
    ]
    for p in procs:
        p.start()
    done = ok = fail = 0
    while done < len(procs):
        kind, seed, rows, wall = queue.get()
        if kind == "done":
            done += 1
        elif kind in ("abort", "error"):
            for p in procs:
                p.terminate()
            raise SystemExit(
                "a recorder process failed (see its traceback above)"
                if kind == "error"
                else f"{MAX_FAILURES_IN_A_ROW} oracle failures in a row (last seed {seed})"
            )
        else:
            ok += kind == "ok"
            fail += kind in ("fail", "short")
            print(
                f"seed {seed}: {kind} rows {rows} {wall:.1f} s  [{ok} ok, {fail} failed]",
                flush=True,
            )
    for p in procs:
        p.join()
    given = list(sys.argv[1:] if argv is None else argv)
    if "--out" in given:
        given[given.index("--out") + 1] = out.name
    command = "MUJOCO_GL=egl python scripts/record_sim_demos.py " + " ".join(given)
    card = {
        "name": out.name,
        "task": args.task,
        "vary": "on: " + json.dumps(dataclasses.asdict(Variation())) if vary else "off",
        "command": command,
    }
    summary = assemble(stage, out, args.episodes, card)
    shutil.rmtree(stage)
    summary.update(
        tried=ok + fail,
        failed=fail,
        wall_s=round(time.monotonic() - t0, 1),
        command=command,
    )
    (out / "meta/record_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["episodes"] < args.episodes:
        raise SystemExit(
            f"only {summary['episodes']} of {args.episodes} episodes recorded"
        )


if __name__ == "__main__":
    main()
