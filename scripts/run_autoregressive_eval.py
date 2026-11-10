"""Roll out Praxis-1 on a set of the autoregressive eval (video only).

Usage:
    python scripts/run_autoregressive_eval.py --config configs/autoregressive_droid.yaml \\
        --out outputs/ar/praxis-1-droid [--set-dir DIR] [--ids ID ...] [--limit N] \\
        [--shard I/N] [--set key=value ...]

For each rollout of the set the script writes ``<out>/<id>.mp4`` (189
frames, 15 fps) and then ``<out>/<id>.json`` (the settings and the chunk
log). A rollout with a json is skipped, so a run resumes where it stopped.
``--shard I/N`` takes every N-th rollout from the I-th (one process per GPU).
Then judge and score the folder with ``scripts/score_autoregressive_eval.py``.
See ``praxis/eval/autoregressive/README.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.eval.autoregressive import rollout as R
from praxis.eval.autoregressive import sets
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.text_encoder import TextEncoder
from praxis.vae import Autoencoder
from scripts.run_inference import load_config


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set-dir", default=None, help="local copy of the set")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    m, r = cfg["model"], cfg["rollout"]
    settings = SamplerSettings.from_config(cfg["sampler"])
    es = sets.load_set(r["set"], args.set_dir)
    todo = es.rollouts
    if args.ids:
        todo = [x for x in todo if x["id"] in set(args.ids)]
    i, n = map(int, args.shard.split("/"))
    todo = todo[i::n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [x for x in todo if not (out / f"{x['id']}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} rollouts to run in {out}", flush=True)
    if not todo:
        return
    dev = m.get("device", "cuda")
    policy = PraxisPolicy.from_pretrained(m["checkpoint"], device=dev)
    vae = Autoencoder.from_pretrained(m["vae_weights"], device=dev)
    text = TextEncoder(m["text_encoder"], device=dev)
    negative = text.encode([r["negative_prompt"]])[0]
    for ro in todo:
        t0 = time.monotonic()
        frame = R.first_frame_clip(sets.praxis_canvas(es, ro["scene"]), policy.config)
        roll = R.VideoRollout(
            policy,
            vae.encode,
            vae.decode_unit,
            frame,
            seed=int(ro["seed"]),
            max_chunks=int(r["max_chunks"]),
            settings=settings,
        )
        prompt = text.encode([ro["prompt"]])[0]
        for _ in range(int(r["chunks"])):
            roll.step(prompt, negative)
        R.write_mp4(roll.video(), out / f"{ro['id']}.mp4")
        rec = {
            "id": ro["id"],
            "set": r["set"],
            "prompt": ro["prompt"],
            "seed": ro["seed"],
            "checkpoint": str(m["checkpoint"]),
            "sampler": dataclasses.asdict(settings),
            "rollout": r,
            "frames": list(roll.video().shape),
            "chunk_log": roll.log,
            "seconds": round(time.monotonic() - t0, 1),
            "torch": torch.__version__,
        }
        (out / f"{ro['id']}.json").write_text(json.dumps(rec, indent=1))
        print(f"{ro['id']}: {rec['frames']} in {rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
