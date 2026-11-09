"""Compute the normalization statistics of your dataset for the base checkpoint.

Usage:
    python scripts/compute_norm_stats.py --config configs/finetune_base.yaml [--set key=value ...]

The script reads the run config of the fine-tune: ``model.base`` (the base
checkpoint), the ``rig`` section (the tile canvas) and the ``data`` section
(the dataset and its columns, ``data.spec``). It draws ``rig.stats_windows``
random windows of the ``train`` split, as the fine-tune draws them, without
the video. It writes the statistics to ``rig.stats`` as JSON
(``praxis.train.norm_stats``). ``scripts/init_heads.py`` reads that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.checkpoint import load_base_config, validate_config
from praxis.rig import single_arm_config
from praxis.train.data import DatasetSpec, DroidWindowDataset
from praxis.train.norm_stats import norm_stats, window_poses_sample
from praxis.train.run_config import load_run_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_run_config(args.config, args.set)
    m, rig, d = cfg["model"], cfg["rig"], cfg["data"]

    config = single_arm_config(
        load_base_config(m["base"]),
        tile_height=int(rig["tile_height"]),
        tile_width=int(rig["tile_width"]),
        antialias=bool(rig["antialias"]),
    )
    validate_config(config)  # the rig fields, before the stats run
    spec = DatasetSpec(**d.get("spec") or {})
    dataset = DroidWindowDataset(
        d["source"],
        config,
        split="train",
        cache_dir=d.get("cache_dir"),
        max_episodes=d.get("max_episodes"),
        spec=spec,
    )
    n = int(rig.get("stats_windows", 20_000))
    actions, proprio = window_poses_sample(dataset, n, seed=int(rig.get("seed", 0)))
    stats = norm_stats(actions, proprio)
    out = Path(rig["stats"])
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **stats,
        "windows": n,
        "episodes": len(dataset.episodes),
        "source": str(d["source"]),
        "spec": asdict(spec),
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out} ({n} windows of {len(dataset.episodes)} episodes)")
    for key in ("action_q01", "action_q99", "proprio_mean", "proprio_std"):
        print(f"{key:13s} " + " ".join(f"{x:+.4f}" for x in stats[key]))


if __name__ == "__main__":
    main()
