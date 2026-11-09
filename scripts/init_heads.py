"""Make a single-arm checkpoint of your rig from the base checkpoint.

Usage:
    python scripts/init_heads.py --config configs/finetune_base.yaml [--set key=value ...]

The script reads the run config of the fine-tune:

* ``model.base``: the base checkpoint (``"embodiment": "base"``).
* ``rig.stats``: the statistics of ``scripts/compute_norm_stats.py``.
* ``rig.tile_height``, ``rig.tile_width``, ``rig.antialias``: the tile
  canvas of the two cameras.
* ``rig.head_seed``: the seed of the new head tensors.
* ``rig.heads`` (optional): a safetensors file with the six head tensors
  (``praxis.checkpoint.BASE_HEAD_TENSORS``) instead of new random tensors.

It writes the checkpoint to ``model.checkpoint``. ``scripts/finetune.py``
starts from that directory. The script needs CPU memory for the full
weights (about 55 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.checkpoint import weight_map
from praxis.rig import init_heads, normalization_from_stats
from praxis.train.run_config import load_run_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_run_config(args.config, args.set)
    m, rig = cfg["model"], cfg["rig"]

    if Path(m["checkpoint"]).exists():  # before the 55 GB load of the base
        raise SystemExit(f"{m['checkpoint']} exists")
    stats = json.loads(Path(rig["stats"]).read_text())
    norm = normalization_from_stats(stats, domain=str(rig.get("name", "custom")))
    heads = load_file(rig["heads"]) if rig.get("heads") else None
    out = Path(m["checkpoint"])
    config = init_heads(
        m["base"],
        out,
        tile_height=int(rig["tile_height"]),
        tile_width=int(rig["tile_width"]),
        antialias=bool(rig["antialias"]),
        normalization=norm,
        seed=int(rig.get("head_seed", 0)),
        heads=heads,
    )
    video = config["video"]
    ex = config["action_expert"]
    print(f"wrote {out}")
    print(
        f"  canvas {video['height']}x{video['width']} (antialias {video['antialias']}), "
        f"action_patch_dim {ex['action_patch_dim']}, "
        f"proprio_patch_dim {ex['proprio_patch_dim']}, "
        f"heads {'from ' + rig['heads'] if heads else 'new (seed ' + str(rig.get('head_seed', 0)) + ')'}"
    )
    # The load receipt: only the six head tensors are not from the base.
    base_names, out_names = set(weight_map(m["base"])), set(weight_map(out))
    print(f"  {len(base_names & out_names)} tensors from the base {m['base']}")
    print(f"  {len(out_names - base_names)} new tensors:")
    for name in sorted(out_names - base_names):
        print(f"    {name}")


if __name__ == "__main__":
    main()
