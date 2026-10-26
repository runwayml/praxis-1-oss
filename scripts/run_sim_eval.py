"""Run the single-arm checkpoint on a fixed evaluation suite of the simulation.

Usage:
    # The harder suite (352 episodes) on GPU 0, sampler seed 0 and then 1.
    MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \\
        --checkpoint checkpoints/praxis-1-droid --out outputs/eval/start

    # Two GPUs: one sampler seed on each GPU, into the same output directory.
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \\
        --checkpoint CKPT --out outputs/eval/start --sampler-seeds 0 &
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \\
        --checkpoint CKPT --out outputs/eval/start --sampler-seeds 1 &

    # The scripted oracle (no GPU, no weights): checks that each task is solvable.
    python scripts/run_sim_eval.py --suite hard --policy scripted \\
        --out outputs/eval/oracle --no-video

The suites are in ``praxis/sim/eval_suites.py``: ``droid`` (7 tasks, 224
episodes), ``droid_guard`` (7 tasks, 56 episodes) and ``hard`` (11 tasks, 352
episodes). Each episode writes ``<out>/<suite dir>/<task>/episode_<seed>.json``
and ``.mp4``. The suite dir is ``suite`` for sampler seed 0 and ``suite_s<k>``
for sampler seed ``k``. One line per episode goes to
``<out>/<suite dir>/episodes.jsonl``. A rerun skips the episodes that exist,
so a stopped run continues. ``--shard I/N`` runs every N-th unit, for more
processes per sampler seed. At the end the script writes
``<out>/summary.json``. ``scripts/compare_sim_eval.py`` compares two outputs.

The script uses ``configs/sim_droid.yaml`` with the default prompt and time
limit of each task, and without the predicted video. See
``praxis/sim/EVAL.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401
from praxis.sim.config import load_config
from praxis.sim.env import SimEnv
from praxis.sim.eval_suites import SUITES, sub_dir
from praxis.sim.loop import ModelPolicy, make_policy, run_episode, select_egl_device
from praxis.sim.scene import CONTROL_HZ
from praxis.sim.video import VideoWriter


def finite(value: object) -> float | None:
    """Return ``value`` as a float, or None when it is missing or not finite."""
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def read_records(root: Path) -> list[dict]:
    """Return the episode records of every sampler seed under ``root``."""
    out = []
    for path in sorted(root.glob("suite*/*/episode_*.json")):
        out.append(json.loads(path.read_text()))
    return out


def summarize(records: list[dict], tasks: tuple[str, ...]) -> dict:
    """Return the successes and the mean progress per task and in total."""
    out: dict[str, dict] = {}
    for task in tasks:
        rs = [r for r in records if r["task"] == task]
        if rs:
            out[task] = {
                "episodes": len(rs),
                "successes": sum(r["success"] for r in rs),
                "mean_progress": sum(r["progress"] for r in rs) / len(rs),
            }
    n = sum(v["episodes"] for v in out.values())
    out["all"] = {
        "episodes": n,
        "successes": sum(v["successes"] for v in out.values()),
        "success_rate": sum(v["successes"] for v in out.values()) / n if n else 0.0,
        # The mean over the tasks, so each task has the same weight.
        "mean_progress": (
            sum(v["mean_progress"] for v in out.values()) / len(out) if out else 0.0
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--config", default="configs/sim_droid.yaml")
    ap.add_argument("--policy", default="model", choices=("model", "scripted"))
    ap.add_argument("--checkpoint", default=None, help="replaces model.checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sampler-seeds",
        default=None,
        help="comma list (default: the sampler seeds of the suite)",
    )
    ap.add_argument("--tasks", default=None, help="comma list (default: all tasks)")
    ap.add_argument("--shard", default="0/1", help="I/N: every N-th unit from I")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="key.sub=value")
    args = ap.parse_args()

    suite = SUITES[args.suite]
    sets = [
        "policy.decode_video=false",
        "viewer.enabled=false",
        f"policy.type={args.policy}",
        *args.set,
    ]
    if args.checkpoint:
        sets.append(f"model.checkpoint={args.checkpoint}")
    cfg = load_config(args.config, sets)
    seeds = (
        [int(s) for s in args.sampler_seeds.split(",")]
        if args.sampler_seeds
        else list(suite.sampler_seeds)
    )
    tasks = tuple(args.tasks.split(",")) if args.tasks else suite.tasks
    unknown = sorted(set(tasks) - set(suite.tasks))
    if unknown:
        raise SystemExit(f"tasks {unknown} are not in suite {suite.name!r}")
    shard, shards = (int(x) for x in args.shard.split("/"))
    out = Path(args.out)
    video = cfg.output.video and not args.no_video
    if args.policy == "model" or video:
        select_egl_device(cfg.render.egl_device)
    t0 = time.monotonic()
    policy = make_policy(cfg, "")
    print(f"policy {policy.name} ready in {time.monotonic() - t0:.0f} s", flush=True)
    envs: dict[str, SimEnv] = {}
    for sampler_seed in seeds:
        # The model policy reads cfg.session at each reset.
        cfg.session.seed = sampler_seed
        units = [u for u in suite.units(sampler_seed) if u[0] in tasks]
        units = units[shard::shards]
        root = out / sub_dir(sampler_seed)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "episodes.jsonl").open("a") as log:
            for task, seed in units:
                stem = root / task / f"episode_{seed:03d}"
                if stem.with_suffix(".json").exists():
                    continue
                if task not in envs:
                    cfg.task.name = task
                    cfg.task.prompt = None
                    cfg.task.time_limit_s = None
                    envs[task] = SimEnv(cfg.env_config())
                env = envs[task]
                env.reset(seed)
                prompt = env.prompt
                if isinstance(policy, ModelPolicy):
                    policy.prompt = prompt
                stem.parent.mkdir(parents=True, exist_ok=True)
                writer = (
                    VideoWriter(
                        stem.with_suffix(".mp4"),
                        fps=CONTROL_HZ,
                        crf=cfg.output.video_crf,
                    )
                    if video
                    else None
                )
                try:
                    r = run_episode(
                        env,
                        policy,
                        seed,
                        prompt=prompt,
                        stop_on_success=True,
                        after_success_s=1.0,
                        video=writer,
                    )
                finally:
                    if writer is not None:
                        writer.close()
                rec = {
                    "suite": suite.name,
                    "task": task,
                    "seed": seed,
                    "sampler_seed": sampler_seed,
                    "prompt": prompt,
                    "policy": policy.name,
                    "checkpoint": cfg.model.checkpoint
                    if args.policy == "model"
                    else None,
                    "success": bool(r.success),
                    "progress": float(r.metrics.get("progress") or 0.0),
                    "min_reach_m": finite(r.metrics.get("min_reach_m")),
                    "success_tick": r.success_tick,
                    "ticks": r.ticks,
                    "calls": r.calls,
                    "bad_view_ticks": r.bad_view_ticks,
                    "mean_call_s": round(r.mean_call_s, 3),
                    "wall_s": round(r.wall_s, 1),
                }
                stem.with_suffix(".json").write_text(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(
                    f"{task} seed {seed} sampler {sampler_seed}: success "
                    f"{rec['success']}, progress {rec['progress']:.2f}, "
                    f"ticks {r.ticks}, calls {r.calls}",
                    flush=True,
                )
    for env in envs.values():
        env.close()
    policy.close()
    summary = summarize(read_records(out), suite.tasks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for task, s in summary.items():
        print(
            f"{task:20s} success {s['successes']}/{s['episodes']}, "
            f"mean progress {s['mean_progress']:.2f}"
        )


if __name__ == "__main__":
    main()
