"""Judge and score the videos of the autoregressive eval, for any model.

Usage:
    # 1. Judge: one <id>.mp4 per rollout of the set in RUN (189 frames, 15 fps,
    #    832x224 or 1248x256). Writes <id>.verdict.json, and with --tts
    #    <id>.tts.json for the fully adherent clips. Existing answers are kept;
    #    an answer without the scored fields is refused and not written.
    ASTRA_BASE_URL=... ASTRA_API_KEY=... \\
    python scripts/score_autoregressive_eval.py judge --set droid --run outputs/ar/mine [--tts]

    # 2. Score: full adherence of each run with its 95 % interval, and the
    #    paired difference against the first run.
    python scripts/score_autoregressive_eval.py score --set droid \\
        outputs/ar/praxis-1-droid outputs/ar/mine [--cutoff 11.894] [--out summary.json]

``--set-dir`` reads the set from a local copy instead of the pinned
Hugging Face revision. A run must hold a verdict for every rollout of the
set, else the score stops. With ``--cutoff`` every fully adherent rollout
needs a ``<id>.tts.json``. The run folders need distinct names. See
``praxis/eval/autoregressive/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.eval.autoregressive import judge as J
from praxis.eval.autoregressive import score as S
from praxis.eval.autoregressive import sets


def _run_file(run: Path, rollout_id: str, suffix: str) -> Path:
    """Return ``run/<rollout_id><suffix>``; refuse an id that leaves ``run``.

    The ids come from the verified set, but a local copy (``--set-dir``) can
    hold any ``rollouts.jsonl``, so an id with a path separator or ``..`` is
    refused before any file is read or written.
    """
    name = f"{rollout_id}{suffix}"
    if Path(name).name != name or name.startswith("."):
        raise ValueError(f"rollout id {rollout_id!r} is not a plain file name")
    path = (run / name).resolve()
    if path.parent != run.resolve():
        raise ValueError(f"rollout id {rollout_id!r} leaves the run folder {run}")
    return path


def _judge_one(es: sets.EvalSet, run: Path, ro: dict, tts: bool, model: str) -> str:
    mp4 = _run_file(run, ro["id"], ".mp4")
    scene = es.scenes[ro["scene"]]
    frames = fps = None
    out = _run_file(run, ro["id"], ".verdict.json")
    if not out.exists():
        frames, fps = J.read_video(mp4)
        v = J.call(J.request(ro, scene, es.spec.name, frames, fps, model))
        S.full_adherence(v)  # refuses an answer without the three booleans
        out.write_text(json.dumps(v, indent=1))
    v = json.loads(out.read_text())
    tts_out = _run_file(run, ro["id"], ".tts.json")
    if tts and S.full_adherence(v) and not tts_out.exists():
        if frames is None:
            frames, fps = J.read_video(mp4)
        if fps is None:
            raise ValueError(f"{mp4} has no frame rate")
        t = J.call(J.tts_request(ro, scene, es.spec.name, frames, fps, model))
        S.success_time(t)  # refuses an answer without a valid success_time_s
        tts_out.write_text(json.dumps(t, indent=1))
    return f"{ro['id']}: full adherence {int(S.full_adherence(v))}"


def judge(args: argparse.Namespace) -> None:
    es = sets.load_set(args.set, args.set_dir)
    run = Path(args.run)
    missing = [
        r["id"] for r in es.rollouts if not _run_file(run, r["id"], ".mp4").exists()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} videos missing in {run}, e.g. {missing[:3]}")
    model = os.environ.get("ASTRA_MODEL", J.DEFAULT_MODEL)
    fails = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_judge_one, es, run, r, args.tts, model) for r in es.rollouts]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                fails += 1
                print(f"FAIL {e}", flush=True)
    if fails:
        raise SystemExit(f"{fails} rollouts failed; run the command again")


def score(args: argparse.Namespace) -> None:
    names = [Path(r).name for r in args.runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"two run folders have the same name: {names}")
    es = sets.load_set(args.set, args.set_dir)
    ids = {r["id"] for r in es.rollouts}
    runs = {Path(r).name: S.load_verdicts(r) for r in args.runs}
    tts = {Path(r).name: S.load_verdicts(r, "tts") for r in args.runs}
    out = S.summarize(
        runs,
        reference=Path(args.runs[0]).name,
        tts=tts if args.cutoff is not None else None,
        cutoff_s=args.cutoff,
        expected=ids,
    )
    out["set"] = args.set
    what = "full adherence" + (
        f" within {args.cutoff} s" if args.cutoff is not None else ""
    )
    for name, r in out["runs"].items():
        line = f"{name}: {what} {r['mean']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]"
        p = r.get(f"paired_vs_{Path(args.runs[0]).name}")
        if p and p.get("n_prompts"):
            line += (
                f", paired {p['delta']:+.1f} [{p['lo']:+.1f}, {p['hi']:+.1f}]"
                f" {S.p_text(p['p_perm'])}"
            )
        print(line)
    if args.out:
        dest = Path(args.out).resolve()
        if dest.is_dir():
            raise SystemExit(f"--out {args.out} is a folder; give a file path")
        dest.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--set", required=True, choices=sorted(sets.SETS))
    j.add_argument("--run", required=True)
    j.add_argument("--set-dir", default=None)
    j.add_argument("--tts", action="store_true", help="also time the successes")
    j.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--set", required=True, choices=sorted(sets.SETS))
    s.add_argument("runs", nargs="+", help="run folders; the first is the reference")
    s.add_argument("--set-dir", default=None)
    s.add_argument("--cutoff", type=float, default=None, help="seconds, e.g. 11.894")
    s.add_argument("--out", default=None)
    args = ap.parse_args()
    judge(args) if args.cmd == "judge" else score(args)


if __name__ == "__main__":
    main()
