"""Fine-tune a praxis checkpoint with FSDP on one or more GPUs.

Usage:
    torchrun --nproc_per_node N scripts/finetune.py --config configs/finetune_base.yaml [--set key=value ...]
    torchrun --nproc_per_node N scripts/finetune.py --config configs/finetune_base.yaml --resume output/step-000500

Each GPU holds a shard of the policy, a full copy of the frozen autoencoder
and a full copy of the frozen text encoder. Each GPU draws its own training
windows. One optimizer step uses ``N * train.grad_accum`` windows.

The script writes ``output.dir/metrics.jsonl`` and one save every
``train.save_every`` steps, ``output.dir/step-NNNNNN``:

* ``model/``: a praxis checkpoint. Use it with ``PraxisPolicy.from_pretrained``
  and the inference scripts.
* ``optimizer/`` and ``trainer_state.json``: the state to continue the run.
  Give the save directory with ``--resume`` (or ``output.resume``).

LoRA: with ``lora.rank`` > 0 (``configs/finetune_droid_lora.yaml``), the
weights of the checkpoint stay frozen and only low-rank adapters of the
``lora.targets`` layers train (``praxis.train.lora``). A save then holds the
merged weights in ``model/`` (a full praxis checkpoint, as for a full
fine-tune) and the trainable tensors in ``trainable.safetensors``. A LoRA run
resumes from the start checkpoint and these tensors.

``train.save: false`` turns off all saves. The overfit check
(``praxis/train/README.md``, "Overfit check") uses it.

Overfit check: with ``data.overfit_windows: N``, rank 0 draws the first N
windows of the ``data.split`` sequence and encodes them once. The draw uses
the data seed of step 0 and all episodes of the split. All ranks get the
encoded windows and use them again and again
(``praxis.train.trainer.cycle_windows``). The run reads no more files after
this. With ``objective.fixed_draws: true``, each window also gets the same
random values each time.

The script turns on the expandable segments of the PyTorch CUDA allocator.
To use other allocator settings, set ``PYTORCH_CUDA_ALLOC_CONF`` before the
start. The text length changes from window to window. Without expandable
segments the GPU memory can fragment, and a run can stop when the memory is
full.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis.model import PraxisPolicy
from praxis.text_encoder import TextEncoder
from praxis.train.data import DatasetSpec, DroidWindowDataset, fixed_windows
from praxis.train.lora import LoRASettings, apply_lora, lora_parameters
from praxis.train.moe import update_expert_biases
from praxis.train.objective import (
    EncodedWindow,
    FinetuneModel,
    ObjectiveSettings,
    encode_window,
)
from praxis.train.run_config import load_run_config
from praxis.train.trainer import (
    TRAINABLE_NAME,
    TextCache,
    Trainer,
    TrainSettings,
    broadcast_windows,
    cycle_windows,
    init_distributed,
)
from praxis.vae import Autoencoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--resume", default=None, help="a save directory to continue from")
    args = ap.parse_args()
    cfg = load_run_config(args.config, args.set)
    if args.resume:
        cfg["output"]["resume"] = args.resume
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rank, world, device = init_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    m, d, o, out = cfg["model"], cfg["data"], cfg["objective"], cfg["output"]
    settings = TrainSettings(**cfg["train"])
    objective = ObjectiveSettings(**o)
    resume = out.get("resume")
    overfit = d.get("overfit_windows")
    if overfit is not None and int(overfit) < 1:
        raise ValueError(f"data.overfit_windows must be >= 1, got {overfit}")
    if objective.fixed_draws and overfit is None:
        raise ValueError("objective.fixed_draws needs data.overfit_windows")
    # Each rank maps the full weights from the safetensors files. The page cache
    # holds one copy of the files for all ranks. FSDP then copies the shard of
    # each rank to its GPU.
    lora = LoRASettings(**cfg.get("lora", {}))
    # A LoRA run resumes from the start checkpoint: model/ of a LoRA save
    # holds merged weights, and the adapters come from trainable.safetensors.
    start = (
        Path(resume) / "model" if resume and not lora.enabled else Path(m["checkpoint"])
    )
    policy = PraxisPolicy.from_pretrained(start, device="cpu")
    if lora.enabled:
        wrapped = apply_lora(policy, lora)
        if resume:
            saved = load_file(str(Path(resume) / TRAINABLE_NAME))
            unknown = policy.load_state_dict(saved, strict=False).unexpected_keys
            if unknown:
                raise ValueError(f"{TRAINABLE_NAME} has unknown tensors: {unknown[:5]}")
        n_lora = sum(p.numel() for _, p in lora_parameters(policy))
        n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        if rank == 0:
            print(
                f"lora: rank {lora.rank}, alpha {lora.alpha}, {len(wrapped)} layers, "
                f"{n_lora / 1e6:.1f} M adapter weights, {n_train / 1e6:.1f} M trainable",
                flush=True,
            )
    model = FinetuneModel(policy, objective)
    trainer = Trainer(
        model,
        model.blocks(),
        settings,
        device,
        post_step=lambda mod: update_expert_biases(mod, objective.moe_balance_speed),
        fixed_draws=objective.fixed_draws,
        # the head-only steps keep the routing biases: only clear their counts,
        # so the first bias update sees the counts of one step
        frozen_step=lambda mod: update_expert_biases(mod, 0.0),
    )
    if resume:
        trainer.load(resume)
    config = policy.config

    block_c = m.get("vae_norm_block_c")
    vae = Autoencoder.from_pretrained(
        m["vae_weights"],
        device=device,
        norm_block_c={int(k): int(v) for k, v in block_c.items()} if block_c else None,
    )
    text = TextCache(
        TextEncoder(m["text_encoder"], device=device), d["text_cache_size"]
    )
    # The text dropout uses the encoded blank caption of the checkpoint.
    model.set_text_uncond(text.encode([blank_caption(config)])[0])
    if overfit is not None:
        next_batch = _overfit_batches(d, config, settings, trainer, vae, text, device)
    else:
        # A resumed run gets new windows: the data seed depends on the start step.
        dataset = DroidWindowDataset(
            d["source"],
            config,
            split=d.get("split", "train"),
            seed=settings.seed * 1_000_003 + trainer.step,
            rank=rank,
            world_size=world,
            cache_dir=d.get("cache_dir"),
            max_episodes=d.get("max_episodes"),
            spec=DatasetSpec(**d.get("spec") or {}),
        )
        workers = int(d["num_workers"])
        loader = DataLoader(dataset, batch_size=None, num_workers=workers)
        windows = iter(loader)

        def next_batch() -> EncodedWindow:
            return encode_window(next(windows), vae, text, config, device)

    if rank == 0:
        names = ", ".join(
            f"{f.name}={getattr(settings, f.name)}" for f in fields(settings)
        )
        print(f"world {world}, {world * settings.grad_accum} windows per step, {names}")
    trainer.fit(next_batch, out["dir"])
    dist.destroy_process_group()


def blank_caption(config: dict) -> str:
    """The blank caption of the unconditional text of the checkpoint config."""
    text_cfg = config.get("text_encoder", {})
    if text_cfg.get("uncond") != "blank_caption":
        raise ValueError("the checkpoint config has no blank caption (text_encoder)")
    return str(text_cfg["blank_caption"])


def _overfit_batches(
    d: dict,
    config: dict,
    settings: TrainSettings,
    trainer: Trainer,
    vae: Autoencoder,
    text: TextCache,
    device: torch.device,
) -> Callable[[], EncodedWindow]:
    """Return the ``next_batch`` function of the overfit check.

    Rank 0 draws the first ``data.overfit_windows`` windows of the split with
    the data seed of step 0 and the full episode pool (not the part of rank
    0). It encodes each window once. Then all ranks get the same bits of the
    encoded windows. Window i gets the ``window_id`` i.
    """
    encoded: list[EncodedWindow] | None = None
    if trainer.rank == 0:
        dataset = DroidWindowDataset(
            d["source"],
            config,
            split=d.get("split", "train"),
            seed=settings.seed * 1_000_003,
            cache_dir=d.get("cache_dir"),
            max_episodes=d.get("max_episodes"),
            spec=DatasetSpec(**d.get("spec") or {}),
        )
        raw = fixed_windows(dataset, int(d["overfit_windows"]))
        picks = [(w.meta["episode"], w.meta["anchor"]) for w in raw]
        print(f"overfit windows (episode, anchor): {picks}", flush=True)
        encoded = [
            encode_window(w, vae, text, config, device, window_id=i)
            for i, w in enumerate(raw)
        ]
    windows = broadcast_windows(encoded, device)
    return cycle_windows(trainer, windows)


if __name__ == "__main__":
    main()
