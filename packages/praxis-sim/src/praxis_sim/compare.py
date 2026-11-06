"""Compare a ``lerobot-eval`` run of the praxis sim with a praxis ``run_sim_eval.py`` run.

Both runs are sets of episodes keyed by ``(task, scene seed)``. The LeRobot run is
the ``eval_info.json`` that ``lerobot-eval`` writes; the praxis
run is the output directory of ``scripts/run_sim_eval.py`` (``suite*/<task>/episode_NNN.json``).
For each task and in total the script gives the successes of both runs, the
episodes the LeRobot run won (praxis failed, LeRobot succeeded) and lost, the exact
McNemar p, and for the total the mean difference with a 95 % bootstrap interval.

    python -m praxis_sim.compare outputs/eval/lerobot_guard/eval_info.json \\
        /path/to/praxis/outputs/eval/guard --sampler-seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_suites import SUITES, mcnemar_p, paired_difference, sub_dir

__all__ = ["compare", "load_lerobot_run", "load_praxis_run"]

Key = tuple[str, int]


def load_lerobot_run(eval_info: str | Path, start_seed: int = 0) -> dict[Key, dict]:
    """Return ``{(task, scene seed): {"success", "max_reward"}}`` of a ``lerobot-eval`` run.

    ``lerobot-eval`` seeds the episodes of a task ``start_seed, start_seed + 1, ...`` in
    order, so episode ``i`` of a task has the scene seed ``start_seed + i``.
    """
    info = json.loads(Path(eval_info).read_text())
    out: dict[Key, dict] = {}
    for task in info.get("per_task", []):
        group, task_id = task["task_group"], int(task["task_id"])
        if group in SUITES:
            name = SUITES[group].tasks[task_id]
        else:
            name = task.get("task_name") or f"{group}/{task_id}"
        metrics = task["metrics"]
        successes = metrics.get("successes") or []
        max_rewards = metrics.get("max_rewards") or [float("nan")] * len(successes)
        for index, (success, max_reward) in enumerate(
            zip(successes, max_rewards, strict=True)
        ):
            out[(name, start_seed + index)] = {
                "success": bool(success),
                "max_reward": float(max_reward),
            }
    if not out:
        raise ValueError(f"{eval_info}: no per-episode records")
    return out


def load_praxis_run(root: str | Path, sampler_seed: int = 0) -> dict[Key, dict]:
    """Return ``{(task, scene seed): {"success", "progress"}}`` of a praxis ``run_sim_eval.py`` run."""
    directory = Path(root) / sub_dir(sampler_seed)
    out: dict[Key, dict] = {}
    for path in sorted(directory.glob("*/episode_*.json")):
        r = json.loads(path.read_text())
        out[(r["task"], int(r["seed"]))] = {
            "success": bool(r["success"]),
            "progress": float(r.get("progress", 0.0)),
        }
    if not out:
        raise ValueError(f"{directory}: no episode records (<task>/episode_NNN.json)")
    return out


def compare(praxis: dict[Key, dict], lerobot: dict[Key, dict]) -> dict:
    """Pair the episodes and return per-task and total statistics (``B`` = LeRobot vs ``A`` = praxis)."""
    keys = sorted(set(praxis) & set(lerobot))
    if not keys:
        raise ValueError("the two runs share no (task, scene seed) episode")
    tasks: dict[str, dict] = {}
    for task, seed in keys:
        a, b = praxis[(task, seed)]["success"], lerobot[(task, seed)]["success"]
        t = tasks.setdefault(task, {"n": 0, "a": 0, "b": 0, "won": [], "lost": []})
        t["n"] += 1
        t["a"] += int(a)
        t["b"] += int(b)
        if b and not a:
            t["won"].append(seed)
        if a and not b:
            t["lost"].append(seed)
    for t in tasks.values():
        t["mcnemar_p"] = mcnemar_p(len(t["won"]), len(t["lost"]))
    a_all = {
        (task, seed, 0): bool(praxis[(task, seed)]["success"]) for task, seed in keys
    }
    b_all = {
        (task, seed, 0): bool(lerobot[(task, seed)]["success"]) for task, seed in keys
    }
    diff = paired_difference(b_all, a_all)  # lerobot - praxis
    level = diff["lo"] <= 0.0 <= diff["hi"]
    won = sum(len(t["won"]) for t in tasks.values())
    lost = sum(len(t["lost"]) for t in tasks.values())
    total = {
        "n": len(keys),
        "a": sum(a_all.values()),
        "b": sum(b_all.values()),
        "won": won,
        "lost": lost,
        "only_praxis": len(set(praxis) - set(lerobot)),
        "only_lerobot": len(set(lerobot) - set(praxis)),
        "diff": diff["diff"],
        "lo": diff["lo"],
        "hi": diff["hi"],
        "verdict": "level (the interval includes 0)"
        if level
        else ("clearly above" if diff["lo"] > 0 else "clearly below"),
    }
    total["mcnemar_p"] = mcnemar_p(won, lost)
    return {"tasks": tasks, "total": total}


def format_report(result: dict) -> str:
    lines = [
        f"{'task':22s} {'praxis':>7s} {'lerobot':>8s} {'won':>4s} {'lost':>5s} {'McNemar p':>10s}"
    ]
    for name, t in result["tasks"].items():
        lines.append(
            f"{name:22s} {t['a']:>3d}/{t['n']:<3d} {t['b']:>4d}/{t['n']:<3d} {len(t['won']):>4d} {len(t['lost']):>5d} {t['mcnemar_p']:>10.4f}"
        )
    tot = result["total"]
    lines.append("")
    lines.append(
        f"total: praxis {tot['a']}/{tot['n']} ({100 * tot['a'] / tot['n']:.1f} %), "
        f"lerobot {tot['b']}/{tot['n']} ({100 * tot['b'] / tot['n']:.1f} %)"
    )
    lo, hi = 100 * tot["lo"], 100 * tot["hi"]
    lines.append(
        f"lerobot - praxis: {100 * tot['diff']:+.1f} points, 95 % interval [{lo:+.1f}, {hi:+.1f}], {tot['verdict']}"
    )
    lines.append(
        f"won {tot['won']}, lost {tot['lost']}, McNemar p {tot['mcnemar_p']:.4f}"
    )
    if tot["only_praxis"] or tot["only_lerobot"]:
        lines.append(
            f"unpaired episodes: {tot['only_praxis']} only in praxis, {tot['only_lerobot']} only in lerobot"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("eval_info", help="eval_info.json of lerobot-eval")
    parser.add_argument(
        "praxis_run", help="output directory of praxis scripts/run_sim_eval.py"
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=0,
        help="sampler seed of the praxis run to pair with",
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="the --seed of the lerobot-eval run"
    )
    parser.add_argument("--json", default=None, help="also write the result here")
    args = parser.parse_args()
    result = compare(
        load_praxis_run(args.praxis_run, args.sampler_seed),
        load_lerobot_run(args.eval_info, args.start_seed),
    )
    print(format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
