"""Compare two runs of ``scripts/run_sim_eval.py`` on the same episodes.

Usage:
    python scripts/compare_sim_eval.py outputs/eval/start outputs/eval/step600
    python scripts/compare_sim_eval.py A B --json compare.json

The script pairs the episodes by ``(task, scene seed, sampler seed)``. For the
success of B against A it gives, per task and in total: the successes of each
run, the wins (A fails and B succeeds), the losses (A succeeds and B fails),
and the exact McNemar p. The total also has the mean difference with a 95 %
bootstrap interval. "clearly above" or "clearly below" means that the interval
does not include 0. The mean progress of each run is also shown. Episodes that
are in only one run are not used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.sim.eval_suites import mcnemar_p, paired_difference

Key = tuple[str, int, int]


def load(root: Path) -> dict[Key, dict]:
    """Return the episode records of a run, by (task, scene seed, sampler seed)."""
    out: dict[Key, dict] = {}
    for path in sorted(root.glob("suite*/*/episode_*.json")):
        r = json.loads(path.read_text())
        out[(r["task"], int(r["seed"]), int(r["sampler_seed"]))] = r
    if not out:
        raise SystemExit(f"{root}: no episode records (suite*/<task>/episode_*.json)")
    return out


def verdict(lo: float, hi: float) -> str:
    """Return the reading of a 95 % interval of B - A."""
    if lo > 0:
        return "clearly above"
    if hi < 0:
        return "clearly below"
    return "level (the interval includes 0)"


def compare(a: dict[Key, dict], b: dict[Key, dict]) -> dict:
    """Return the per-task and the total comparison of B against A."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("the two runs have no common episodes")
    tasks = sorted({k[0] for k in keys})
    per: dict[str, dict] = {}
    for t in tasks:
        ks = [k for k in keys if k[0] == t]
        wins = sum(1 for k in ks if b[k]["success"] and not a[k]["success"])
        losses = sum(1 for k in ks if a[k]["success"] and not b[k]["success"])
        per[t] = {
            "episodes": len(ks),
            "a": sum(bool(a[k]["success"]) for k in ks),
            "b": sum(bool(b[k]["success"]) for k in ks),
            "wins": wins,
            "losses": losses,
            "mcnemar_p": mcnemar_p(wins, losses),
            "a_progress": sum(a[k]["progress"] for k in ks) / len(ks),
            "b_progress": sum(b[k]["progress"] for k in ks) / len(ks),
        }
    sa = {k: bool(a[k]["success"]) for k in keys}
    sb = {k: bool(b[k]["success"]) for k in keys}
    diff = paired_difference(sb, sa)
    total: dict[str, float | int | str] = {
        **diff,
        "a": sum(sa.values()),
        "b": sum(sb.values()),
        "verdict": verdict(float(diff["lo"]), float(diff["hi"])),
        "only_a": len(set(a) - set(b)),
        "only_b": len(set(b) - set(a)),
    }
    return {"tasks": per, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", help="the baseline run, for example the start checkpoint")
    ap.add_argument("b", help="the run to compare with A")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()
    res = compare(load(Path(args.a)), load(Path(args.b)))
    print(
        f"{'task':22s} {'A':>5s} {'B':>5s} {'won':>5s} {'lost':>5s} "
        f"{'McNemar p':>10s} {'progress A':>11s} {'progress B':>11s}"
    )
    for t, r in res["tasks"].items():
        n = r["episodes"]
        print(
            f"{t:22s} {r['a']:>2d}/{n:<2d} {r['b']:>2d}/{n:<2d} {r['wins']:>5d} "
            f"{r['losses']:>5d} {r['mcnemar_p']:>10.4f} {r['a_progress']:>11.3f} "
            f"{r['b_progress']:>11.3f}"
        )
    t = res["total"]
    n = t["n"]
    print(
        f"\ntotal: A {t['a']}/{n} ({100 * t['a'] / n:.1f} %), "
        f"B {t['b']}/{n} ({100 * t['b'] / n:.1f} %)\n"
        f"B - A: {100 * t['diff']:+.1f} points, 95 % interval "
        f"[{100 * t['lo']:+.1f}, {100 * t['hi']:+.1f}], {t['verdict']}\n"
        f"won {t['wins']}, lost {t['losses']}, McNemar p {t['mcnemar_p']:.4f}"
    )
    if t["only_a"] or t["only_b"]:
        print(f"not paired: {t['only_a']} episodes only in A, {t['only_b']} only in B")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
