"""scripts/record_sim_demos.py writes a tree that the DROID window dataset reads (CPU, no weights).

The episodes are synthetic (no scene, no rendering): the test checks the
writer, the assembly and the read-back by the DROID window dataset.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("praxis_sim")  # the `sim` extra

# praxis_sim sets the OpenGL backend of MuJoCo. It must load before mujoco, or a
# later rendering test aborts in the same process.
import praxis_sim  # noqa: F401

pytest.importorskip("mujoco")
pytest.importorskip("av")
pytest.importorskip("pyarrow")

from praxis.robot_io import pose8_to_10d
from praxis.single_arm import RIG_FRAMES, pose8_to_model
from praxis.train import data

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "record_sim_demos", ROOT / "scripts" / "record_sim_demos.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclasses of the script look their module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rec = _script()

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
        "tile_height": 8,
        "tile_width": 14,
        "aspect_tol": 0.03,
    },
    "pose_frame": "flange",
}


def _rot(rpy: np.ndarray) -> np.ndarray:
    a, b, c = (float(v) for v in rpy)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_euler_round_trip_matches_the_data_code() -> None:
    rng = np.random.default_rng(0)
    rpy = np.stack(
        [
            rng.uniform(-np.pi, np.pi, 64),
            rng.uniform(-1.4, 1.4, 64),
            rng.uniform(-np.pi, np.pi, 64),
        ],
        axis=1,
    )
    mats = np.stack([_rot(r) for r in rpy])
    np.testing.assert_allclose(rec.matrix_to_euler_xyz(mats), rpy, atol=1e-9)
    # The tool-down pose of the sim rig: pitch near +-pi, the DROID data range.
    down = np.diag([1.0, -1.0, -1.0])
    back = rec.matrix_to_euler_xyz(down[None])
    np.testing.assert_allclose(_rot(back[0]), down, atol=1e-12)


def _episode(n: int, seed: int) -> object:
    """A synthetic episode in the rig frame of the sim, converted as the recorder does."""
    t = np.arange(n, dtype=np.float64)
    rig = np.zeros((n, 8))
    rig[:, 0] = 0.4 + 0.002 * t
    rig[:, 1] = 0.05 * np.sin(t / 20)
    rig[:, 2] = 0.3
    # Tool down, turned by a yaw that changes slowly.
    for i, yaw in enumerate(0.3 * np.sin(t / 30)):
        m = _rot(np.array([np.pi, 0.0, yaw]))
        rig[i, 3:7] = rec.matrix_to_quat_xyzw(m[None])[0, [3, 0, 1, 2]]
    rig[:, 7] = (t > n / 2).astype(np.float64)
    cmd = np.roll(rig, -1, axis=0)
    cmd[-1] = rig[-1]
    frame = RIG_FRAMES["praxis.sim"]
    rng = np.random.default_rng(seed)
    return rec.Episode(
        seed=seed,
        prompt="Put the red block in the bowl.",
        observed=pose8_to_model(rig.astype(np.float32), frame),
        commanded=pose8_to_model(cmd.astype(np.float32), frame),
        exterior=rng.integers(0, 256, (n, 18, 32, 3), dtype=np.uint8),
        wrist=rng.integers(0, 256, (n, 18, 32, 3), dtype=np.uint8),
        success_tick=n - 15,
        oracle={"speed": 0.15},
    )


def test_assembled_tree_reads_as_droid_windows(tmp_path: Path) -> None:
    stage = tmp_path / "out.staging"
    episodes = {1003: _episode(110, 1003), 1000: _episode(102, 1000)}
    for seed, ep in episodes.items():
        tmp = stage / f"seed_{seed:07d}"
        rec.write_episode(tmp, 0, 0, 0, ep, crf=0)
        meta = {
            "seed": seed,
            "prompt": ep.prompt,
            "length": ep.observed.shape[0],
            "success_tick": ep.success_tick,
            "oracle": ep.oracle,
        }
        (tmp / "episode.json").write_text(json.dumps(meta))
    out = tmp_path / "out"
    card = {"name": "out", "task": "block_in_bowl", "vary": "off", "command": "x"}
    summary = rec.assemble(stage, out, 2, card)
    assert summary == {"episodes": 2, "frames": 212, "seeds": [1000, 1003]}
    lines = (out / "meta/episodes.jsonl").read_text().splitlines()
    eps = [json.loads(line) for line in lines]
    # Numbered in seed order, whatever the order of the workers.
    assert [e["scene_seed"] for e in eps] == [1000, 1003]
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["fps"] == 15
    assert info["features"][data.VIEWS[0]]["shape"] == [18, 32, 3]
    assert (out / "README.md").is_file()

    # Episode 0 is in the "val" split (index % val_every == 0), episode 1 in "train".
    splits = {
        s: data.DroidWindowDataset(str(out), CONFIG, split=s) for s in ("val", "train")
    }
    assert splits["val"].episodes == [0] and splits["train"].episodes == [1]
    for e in eps:
        ds = splits["val" if e["episode_index"] == 0 else "train"]
        ep = ds.load_episode(e["episode_index"])
        src = episodes[e["scene_seed"]]
        # The stored poses come back in the flange frame of the model.
        np.testing.assert_allclose(
            pose8_to_10d(ep.observed), pose8_to_10d(src.observed), atol=2e-6
        )
        np.testing.assert_allclose(
            pose8_to_10d(ep.commanded), pose8_to_10d(src.commanded), atol=2e-6
        )
        win = ds.load_window(ep, 0)
        assert win.frames.shape == (93, 3, 8, 28)
        assert win.prompt == "Put the red block in the bowl."


def test_eval_seeds_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="kept for the evaluation"):
        rec.main(
            [
                "--task",
                "lift_cube",
                "--episodes",
                "1",
                "--first-seed",
                "0",
                "--out",
                str(tmp_path / "x"),
            ]
        )


def test_a_leftover_staging_dir_is_refused(tmp_path: Path) -> None:
    old = tmp_path / "x.staging" / "seed_0002000"
    old.mkdir(parents=True)
    (old / "episode.json").write_text(json.dumps({"seed": 2000, "prompt": "old"}))
    with pytest.raises(SystemExit, match="left by an earlier run"):
        rec.main(
            ["--task", "lift_cube", "--episodes", "1", "--out", str(tmp_path / "x")]
        )
    assert not (tmp_path / "x").exists()


def test_a_failing_worker_stops_the_recorder(tmp_path: Path) -> None:
    # The worker raises on the unknown task; the parent must exit, not wait on
    # the queue for a "done" that never comes.
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "record_sim_demos.py"),
            "--task",
            "no_such_task",
            "--episodes",
            "1",
            "--out",
            str(tmp_path / "x"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode != 0
    assert "a recorder process failed" in run.stderr
    assert not (tmp_path / "x").exists()


def test_variation_is_seeded_and_in_range() -> None:
    vary = rec.Variation()
    a, b = rec.oracle_config(1234, vary), rec.oracle_config(1234, vary)
    assert a == b
    assert vary.speed[0] <= a.speed <= vary.speed[1]
    assert vary.approach[0] <= a.approach_height <= vary.approach[1]
    assert rec.oracle_config(1234, None) == rec.ScriptedConfig()
