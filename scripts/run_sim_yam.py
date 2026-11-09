"""Run the bimanual YAM sim: a checkpoint (or the scripted oracle) on (task, seed) units.

Usage:
    # The scripted oracle (no GPU, no weights): checks that each task is solvable.
    python scripts/run_sim_yam.py --config configs/sim_yam.yaml --policy oracle \\
        --out outputs/sim_yam_oracle --seeds 20 --no-video

    # The bimanual checkpoint on 2 seeds per task (one GPU).
    MUJOCO_GL=egl python scripts/run_sim_yam.py --config configs/sim_yam.yaml \\
        --checkpoint checkpoints/praxis-1-yam --out outputs/sim_yam --seeds 2

    # A real cell: a rig profile sets the table, the cameras, textures and lights.
    MUJOCO_GL=egl python scripts/run_sim_yam.py --config configs/sim_yam.yaml \\
        --rig-profile path/to/cell.json --out outputs/sim_yam_cell --seeds 2

``--prompt`` replaces the prompt of every task (the success check of the task
still runs). ``--units task:seed,task:seed,...`` selects single episodes instead of
``--seeds N`` (seeds 0 .. N-1 of every task, or of ``--tasks``). The model
loads once per process. Each episode writes ``<out>/<task>/episode_<seed>.json`` (seed zero-padded, ``episode_007``)
and ``.mp4`` (top | wrist left | wrist right, with a status band) and appends
one line to ``<out>/episodes.jsonl``. A rerun skips the episodes that exist.
At the end the script prints the success rate and the mean progress per task
and writes ``<out>/summary.json``.

The script needs the ``sim`` extra: ``pip install -e ".[sim]"``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401
from praxis.sim.assets import AssetError
from praxis.sim.config import to_dict
from praxis.sim.loop import select_egl_device
from praxis.sim.video import VideoWriter
from praxis.sim.yam_config import (
    YAM_POLICIES,
    load_yam_config,
    make_yam_policy,
    render_gpu,
)
from praxis.sim.yam_env import YamEnv, YamEnvConfig
from praxis.sim.yam_loop import run_yam_episode
from praxis.sim.yam_scene import load_rig_profile
from praxis.sim.yam_tasks import YAM_TASKS, YamTaskConfig


def parse_units(args: argparse.Namespace) -> list[tuple[str, int]]:
    """Return the (task, seed) units of the command line."""
    if args.units:
        units = []
        for item in args.units.split(","):
            task, sep, seed = item.partition(":")
            if not sep or not seed.strip().isdigit():
                raise SystemExit(f"--units items are task:seed, got {item!r}")
            units.append((task.strip(), int(seed)))
    else:
        tasks = args.tasks.split(",") if args.tasks else sorted(YAM_TASKS)
        units = [(t, s) for s in range(args.seeds) for t in tasks]
    unknown = sorted({t for t, _ in units} - set(YAM_TASKS))
    if unknown:
        raise SystemExit(f"unknown tasks {unknown}; the tasks are {sorted(YAM_TASKS)}")
    return units


def file_sha256(path: str | None) -> str | None:
    """Return the SHA-256 of a file, or None for no file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def rig_profile_key(path: str | None) -> dict[str, str | None] | None:
    """Return the SHA-256 of a rig profile and of each of its textures."""
    if not path:
        return None
    prof = load_rig_profile(path)
    key = {"profile": file_sha256(path)}
    key |= {k: file_sha256(f) for k, f in sorted(prof["textures"].items())}
    return key


def summarize(records: list[dict]) -> dict:
    """Return the success rate and the mean progress per task and over all episodes."""
    out: dict[str, dict] = {}
    for task in sorted({r["task"] for r in records}):
        rs = [r for r in records if r["task"] == task]
        out[task] = {
            "episodes": len(rs),
            "successes": sum(r["success"] for r in rs),
            "success_rate": sum(r["success"] for r in rs) / len(rs),
            "mean_progress": sum(r["progress"] for r in rs) / len(rs),
        }
    n = len(records)
    out["all"] = {
        "episodes": n,
        "successes": sum(r["success"] for r in records),
        "success_rate": sum(r["success"] for r in records) / n if n else 0.0,
        "mean_progress": sum(r["progress"] for r in records) / n if n else 0.0,
    }
    return out


def main() -> None:
    try:
        run()
    except AssetError as e:
        raise SystemExit(
            f"{e}\nCopy the assets with --set scene.asset_source=DIR, or allow "
            "the download with --set scene.download=true."
        ) from e


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/sim_yam.yaml")
    ap.add_argument("--policy", default="model", choices=YAM_POLICIES)
    ap.add_argument("--checkpoint", default=None, help="replaces model.checkpoint")
    ap.add_argument("--rig-profile", default=None, help="replaces scene.rig_profile")
    ap.add_argument("--prompt", default=None, help="replaces the prompt of every task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default=None, help="task:seed,task:seed,...")
    ap.add_argument("--seeds", type=int, default=2, help="seeds 0..N-1 per task")
    ap.add_argument("--tasks", default=None, help="comma list (default: all tasks)")
    ap.add_argument(
        "--sampler-seed", type=int, default=None, help="replaces session.seed"
    )
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="key.sub=value")
    args = ap.parse_args()
    cfg = load_yam_config(args.config, args.set)
    if args.checkpoint:
        cfg.model.checkpoint = args.checkpoint
    if args.rig_profile:
        cfg.scene.rig_profile = args.rig_profile
    if args.sampler_seed is not None:
        cfg.session.seed = args.sampler_seed
    units = parse_units(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = cfg.output.video and not args.no_video
    if args.policy == "model" or video:
        model_device = cfg.model.device if args.policy == "model" else None
        select_egl_device(render_gpu(cfg.render.egl_device, model_device))
    t0 = time.monotonic()
    policy = make_yam_policy(args.policy, cfg)
    print(f"policy {policy.name} ready in {time.monotonic() - t0:.0f} s", flush=True)
    scene = cfg.scene_config()
    render = args.policy == "model" or video
    envs: dict[str, YamEnv] = {}
    records: list[dict] = []
    log_path = out / "episodes.jsonl"
    if log_path.exists():
        records = [
            json.loads(x) for x in log_path.read_text().splitlines() if x.strip()
        ]
    # The settings that change a result. A rerun into the same directory resumes
    # only with the same settings.
    run_key = {
        "policy": args.policy,
        "model": to_dict(cfg.model) if args.policy == "model" else None,
        "prompt": args.prompt,
        "sampler": to_dict(cfg.sampler),
        "session": to_dict(cfg.session),
        "policy_settings": to_dict(cfg.policy),
        "episodes": to_dict(cfg.episodes),
        "scene": {
            k: v
            for k, v in to_dict(cfg.scene).items()
            if k not in ("assets_dir", "asset_source", "download")
        },
        "rig_profile_sha256": rig_profile_key(cfg.scene.rig_profile),
    }
    if any(r.get("run_key") != run_key for r in records):
        raise SystemExit(
            f"{log_path} holds episodes of other settings. Use a new --out directory."
        )
    done = {(r["task"], r["seed"]) for r in records}
    try:
        with log_path.open("a") as log:
            for task, seed in units:
                if (task, seed) in done:
                    continue
                if task not in envs:
                    env = YamEnv(
                        YamEnvConfig(
                            scene=scene,
                            task=YamTaskConfig(name=task, prompt=args.prompt),
                            render=render,
                        )
                    )
                    env.task.cfg.time_limit_s = (
                        env.task.time_limit_s * cfg.episodes.time_scale
                    )
                    envs[task] = env
                env = envs[task]
                stem = out / task / f"episode_{seed:03d}"
                stem.parent.mkdir(parents=True, exist_ok=True)
                writer = (
                    VideoWriter(
                        stem.with_suffix(".mp4"), fps=15, crf=cfg.output.video_crf
                    )
                    if video
                    else None
                )
                try:
                    r = run_yam_episode(
                        env,
                        policy,
                        seed,
                        stop_on_success=cfg.episodes.stop_on_success,
                        after_success_s=cfg.episodes.after_success_s,
                        video=writer,
                    )
                finally:
                    if writer is not None:
                        writer.close()
                rec = {
                    "run_key": run_key,
                    "task": task,
                    "seed": seed,
                    "prompt": r.prompt,
                    "model_prompt": r.model_prompt,
                    "policy": r.policy,
                    "success": r.success,
                    "progress": round(r.progress, 4),
                    "ticks": r.ticks,
                    "max_ticks": env.max_ticks,
                    "calls": r.calls,
                    "success_tick": r.success_tick,
                    "ik_misses": r.metrics.get("ik_misses"),
                    "object_off_table": r.metrics.get("object_off_table"),
                    "bad_view_ticks": r.bad_view_ticks,
                    "mean_call_s": round(sum(r.call_s) / max(len(r.call_s), 1), 3),
                    "wall_s": round(r.wall_s, 1),
                    "sampler_seed": cfg.session.seed,
                    "checkpoint": cfg.model.checkpoint
                    if args.policy == "model"
                    else None,
                    "video": str(stem.with_suffix(".mp4")) if video else None,
                }
                stem.with_suffix(".json").write_text(json.dumps(r.to_json(), indent=1))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                records.append(rec)
                done.add((task, seed))
                print(
                    f"{task} seed {seed}: success {r.success}, progress {r.progress:.2f}, "
                    f"ticks {r.ticks}, calls {r.calls}",
                    flush=True,
                )
    finally:
        for env in envs.values():
            env.close()
    summary = summarize([r for r in records if (r["task"], r["seed"]) in set(units)])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for task, s in summary.items():
        print(
            f"{task:20s} success {s['successes']}/{s['episodes']} "
            f"({100 * s['success_rate']:.0f}%), mean progress {s['mean_progress']:.2f}"
        )


if __name__ == "__main__":
    main()
