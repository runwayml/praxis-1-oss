"""Measure the oracle and the deployment action error on fixed DROID windows.

Usage:
    python scripts/run_action_error_eval.py --checkpoint checkpoints/praxis-1-droid \\
        --vae weights/Wan2.2_VAE.pth --text-encoder weights/Qwen3.5-2B \\
        --data droid_hires --out outputs/action_error/start

The episodes are the 19 DROID 1.0.1 val episodes of
``praxis/eval/droid_val_episodes.json`` (the ``"val"`` split of the
fine-tune data, see the docstring of ``praxis/eval/offline_action_error.py``).
Each episode gives ``--windows-per-episode`` windows of 93 frames (default 7,
133 windows). The start range of the episode splits into that many equal
parts, and each part gets one random start from the generator of
``(--seed, episode)``. Thus a run gives the same windows for every
checkpoint, and the windows spread over each episode. The episodes have 135
to 629 frames.

``--data`` is required: the DROID tree of ``praxis/train/README.md`` (the
raw 1280x720 videos). The Hugging Face id ``cadene/droid_1.0.1`` also works,
but its videos have 320x180 frames, so the canvas is an upscale: use it only
for a quick test. ``summary.json`` records the source and, for a dataset id,
its revision. A revision other than the one of the episode list gives a
warning, because the window starts depend on the episode lengths.

``--anchor-frame N`` replaces the random windows by one window per episode
that starts at frame ``N`` (19 windows). ``--anchor-frame 24`` is the window
of the parity checks of the original implementation.

The sampler is the eval sampler of the original implementation by default: 5
Euler steps, flow shift 1.0, no text guidance. It is not the recommended
serve recipe, so the deployment leg then measures the eval sampler, not the
served policy. ``--sampler-config configs/serve_droid.yaml`` reads the
``sampler`` section of that YAML file the way the policy server does
(:meth:`praxis.model.SamplerSettings.from_config`), for example the served
2 steps, flow shift 12.5 and video guidance 4.0. The context stays the
teacher-forced window of training (chunk ``k`` reads the ``k`` recorded
chunks before it). ``summary.json`` records the sampler of the run.

The script needs one GPU. Each window runs two sampler calls (the oracle and
the deployment leg). It writes:

* ``<out>/windows.jsonl``: one line per window and leg, with the episode,
  the first frame, the prompt and the metrics of each chunk;
* ``<out>/actions/w<NNN>.npz``: the predicted rows of both legs and the
  recorded rows (``(93, 10)`` each);
* ``<out>/summary.json``: the settings, the window starts and the mean and
  standard error of each metric (:func:`praxis.eval.offline_action_error.summarize`).

``--out`` must be a new directory.

See the docstring of ``praxis/eval/offline_action_error.py`` for the metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.checkpoint import model_variant
from praxis.eval import offline_action_error as ae
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.text_encoder import TextEncoder
from praxis.train.data import DroidWindowDataset
from praxis.train.objective import encode_window
from praxis.vae import Autoencoder

#: Windows per episode: 7 x 19 episodes = 133 windows. The 7 starts spread
#: over the start, the middle and the end of each episode.
WINDOWS_PER_EPISODE = 7


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vae", required=True, help="Wan2.2_VAE.pth")
    ap.add_argument("--text-encoder", required=True, help="Qwen3.5-2B directory")
    ap.add_argument("--data", required=True, help="DROID tree or dataset id")
    ap.add_argument("--out", required=True)
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument("--windows-per-episode", type=int, default=WINDOWS_PER_EPISODE)
    choice.add_argument(
        "--anchor-frame",
        type=int,
        default=None,
        help="one window per episode, from this frame",
    )
    ap.add_argument(
        "--sampler-config",
        default=None,
        help="YAML file whose sampler section replaces the eval sampler",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    spec = ae.load_windows_file()
    settings = ae.EVAL_SAMPLER
    if args.sampler_config is not None:
        cfg = yaml.safe_load(Path(args.sampler_config).read_text())
        settings = SamplerSettings.from_config(cfg["sampler"])
    per_episode = 1 if args.anchor_frame is not None else args.windows_per_episode
    n_windows = len(spec["episodes"]) * per_episode
    if n_windows > ae.SEED_STRIDE:
        raise SystemExit(f"{n_windows} windows; a run holds at most {ae.SEED_STRIDE}")

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; give a new --out")
    policy = PraxisPolicy.from_pretrained(args.checkpoint, device=args.device)
    config = policy.config
    vae = Autoencoder.from_pretrained(args.vae, device=args.device)
    text = TextEncoder(args.text_encoder, device=args.device)
    data = DroidWindowDataset(args.data, config, episodes=list(spec["episodes"]))
    if data.revision not in (None, spec["revision"]):
        print(
            f"warning: {data.source} is at the revision {data.revision}, the "
            f"episode list at {spec['revision']}: the windows can differ",
            flush=True,
        )
    window_frames = data.window_frames
    uncond = None
    if settings.guided:
        uncond = text.encode([model_variant(config).blank_caption])[0]
    (out / "actions").mkdir(parents=True)

    records: list[dict] = []
    starts: list[dict] = []
    index = 0
    for episode_index in spec["episodes"]:
        episode = data.load_episode(int(episode_index))
        if args.anchor_frame is not None:
            anchors = [args.anchor_frame]
        else:
            anchors = ae.stratified_anchors(
                episode.observed.shape[0],
                window_frames,
                data.fps,
                args.windows_per_episode,
                ae.episode_rng(args.seed, int(episode_index)),
            )
        for anchor in anchors:
            t0 = time.monotonic()
            window = data.load_window(episode, anchor)
            encoded = encode_window(window, vae, text, config, args.device)
            gt = ae.gt_chunks(window, config)
            recs = ae.evaluate_window(
                policy,
                encoded,
                gt,
                index,
                args.seed,
                settings,
                text_uncond=uncond,
            )
            np.savez_compressed(
                out / "actions" / f"w{index:03d}.npz",
                gt_rows=np.concatenate(gt),
                **{f"pred_rows_{r['leg']}": r.pop("pred_rows") for r in recs},
            )
            meta = {"episode": int(episode_index), "anchor": anchor}
            meta["prompt"] = window.prompt
            for r in recs:
                r.update(meta)
                records.append(r)
            with (out / "windows.jsonl").open("a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            starts.append({"window": index, **meta})
            print(
                f"window {index:3d} episode {episode_index} frame {anchor}: "
                + " | ".join(
                    f"{r['leg']} "
                    + " ".join(
                        f"{r[f'endpoint_err_cm_chunk{k}']:.2f}" for k in range(4)
                    )
                    for r in recs
                )
                + f" cm ({time.monotonic() - t0:.0f} s)",
                flush=True,
            )
            index += 1
            del window, encoded

    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "revision": data.revision,
        "episode_list_revision": spec["revision"],
        "episodes": len(spec["episodes"]),
        "windows_per_episode": per_episode,
        "anchor_frame": args.anchor_frame,
        "windows": starts,
        "seed": args.seed,
        "environment": {
            "gpu": (
                torch.cuda.get_device_name(policy.device)
                if policy.device.type == "cuda"
                else None
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "sampler": {
            "source": args.sampler_config or "eval sampler (EVAL_SAMPLER)",
            **dataclasses.asdict(settings),
        },
        "context": "teacher-forced: chunk k reads the k recorded chunks before it",
        "metrics": ae.summarize(records),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    m = summary["metrics"]
    for leg in ae.LEGS:
        e = m[f"{leg}/endpoint_err_cm_chunks123_mean"]
        print(
            f"{leg}: endpoint error, chunks 1-3: {e['mean']:.2f} +- {e['sem']:.2f} cm"
        )


if __name__ == "__main__":
    main()
