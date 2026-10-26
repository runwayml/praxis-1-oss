"""Training loop for the fine-tuning of a praxis checkpoint on several GPUs.

The loop uses PyTorch FSDP2 (``torch.distributed.fsdp.fully_shard``) and runs
under ``torchrun``. In each optimizer step, each rank does these steps:

1. Get ``grad_accum`` encoded windows, one window for each micro-step.
2. Run the model on each window and add the gradients. The trainer divides
   the loss of each micro-step by ``grad_accum``.
3. Do one AdamW step. Then call ``post_step`` (the update of the MoE routing
   biases).

Sharding: each module of ``blocks`` is one FSDP unit. The root module holds
all other parameters. The parameters keep their dtype (bfloat16), and the
mixed precision policy casts no inputs. Thus the float32 tensors of the model
(the rotary tables, the actions, the proprio) stay float32. The gradient
reduce-scatter runs in ``reduce_dtype`` (float32 by default). The sharded
gradients keep the parameter dtype. The float32 routing biases
(``*.ffn.expert_bias``) are buffers. FSDP does not shard them, and they stay
float32.

Memory for P parameters on N GPUs: each GPU holds 8 bytes per parameter,
divided by N. These are 2 bytes of parameters, 2 bytes of gradients and 4
bytes of AdamW moments. For the 27.2B parameters of the praxis checkpoint,
this is 218 GB. Thus each GPU holds 54 GB on 4 GPUs and 27 GB on 8 GPUs. The
activations, the FSDP buffers, the autoencoder and the text encoder come in
addition.

Gradient accumulation: by default, FSDP reduce-scatters the gradients after
each micro-step and adds them to the sharded bfloat16 gradients. Each add
rounds to bfloat16. With
``no_sync``, FSDP keeps the full float32 gradients of all parameters on each
GPU until the last micro-step. This needs 4 bytes per parameter on each GPU
(109 GB for the praxis checkpoint). Use ``no_sync`` only for small models.

Seeds: the noise generator of a micro-step depends on the seed, the step and
the window index in the step (``micro * world_size + rank``). The stochastic
rounding of the optimizer depends on the seed, the step and the parameter.
Thus a resumed run gives the same noise. The noise and the rounding do not
depend on the number of GPUs. With ``fixed_draws``, the noise generator
depends only on the seed and the ``window_id`` of the window. Thus each use
of a window gets the same random values (for the overfit check).
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import logging
import math
import os
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from praxis.checkpoint import save_checkpoint
from praxis.model import PraxisPolicy

if TYPE_CHECKING:
    from praxis.text_encoder import TextEncoder
    from praxis.train.objective import EncodedWindow

log = logging.getLogger(__name__)

# The stochastic rounding noise of a parameter comes in blocks of this many
# elements of the flat parameter. Each block has its own generator seed.
SR_BLOCK = 1 << 24

REDUCE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

# The second key of the noise seed of a window with fixed draws, in the place
# of the step. ``SeedSequence`` cuts each key into 32-bit words, so a larger
# key could give the words of a default seed. With this key, the seeds of the
# two kinds differ for all steps below 2**32 - 1.
FIXED_DRAWS_KEY = (1 << 32) - 1


def derive_seed(*keys: int) -> int:
    """Return a 64-bit seed that depends on all ``keys`` (integers >= 0).

    A CPU ``torch.Generator`` uses only the low 32 bits of a seed.
    """
    hi, lo = np.random.SeedSequence([int(k) for k in keys]).generate_state(2)
    return (int(hi) << 32) | int(lo)


@dataclass
class TrainSettings:
    """Settings of the training loop (the ``train`` section of the run config).

    One optimizer step uses ``world_size * grad_accum`` windows. The learning
    rate rises linearly for ``warmup_steps`` steps. After the warmup it is flat
    (``lr_schedule`` "flat") or falls on a cosine curve from ``lr`` to
    ``min_lr`` at the last step (``lr_schedule`` "cosine").
    ``clip_grad_norm`` None turns the gradient clipping off. ``save`` false
    turns off all saves, also the save after the last step. The defaults of
    ``steps``, ``lr`` and ``save_every`` are the values of
    ``configs/finetune_droid.yaml``.
    """

    steps: int = 1_200
    grad_accum: int = 1
    lr: float = 3e-6
    warmup_steps: int = 0
    lr_schedule: str = "flat"
    min_lr: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4
    eps: float = 1e-8
    clip_grad_norm: float | None = None
    activation_checkpointing: bool = True
    no_sync: bool = False
    reduce_dtype: str = "float32"
    seed: int = 0
    log_every: int = 1
    save_every: int = 100
    save: bool = True

    def __post_init__(self) -> None:
        # A value from the command line can be a string ("1e-4" is a string
        # in YAML). Convert each value to the type of its field.
        for name in ("lr", "min_lr", "weight_decay", "eps"):
            setattr(self, name, float(getattr(self, name)))
        for name in ("steps", "grad_accum", "warmup_steps", "seed"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("log_every", "save_every"):
            setattr(self, name, int(getattr(self, name)))
        if self.clip_grad_norm is not None:
            self.clip_grad_norm = float(self.clip_grad_norm)
        self.betas = (float(self.betas[0]), float(self.betas[1]))
        for name in ("grad_accum", "log_every", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.lr_schedule not in ("flat", "cosine"):
            raise ValueError(
                f"lr_schedule must be flat or cosine, got {self.lr_schedule!r}"
            )
        if self.reduce_dtype not in REDUCE_DTYPES:
            raise ValueError(f"reduce_dtype must be one of {sorted(REDUCE_DTYPES)}")
        if not isinstance(self.save, bool):
            raise ValueError(f"save must be true or false, got {self.save!r}")

    def lr_at(self, step: int) -> float:
        """Return the learning rate of step ``step`` (0-based)."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if self.lr_schedule == "flat":
            return self.lr
        span = max(self.steps - 1 - self.warmup_steps, 1)
        frac = min(max(step - self.warmup_steps, 0) / span, 1.0)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * frac)
        )


def init_distributed() -> tuple[int, int, torch.device]:
    """Start the process group. Return the rank, the world size and the device.

    Under ``torchrun`` the environment gives the rank and the world size.
    Without ``torchrun`` the process is the only rank. The backend is NCCL on
    CUDA and gloo on the CPU.
    """
    timeout = datetime.timedelta(minutes=30)
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if "WORLD_SIZE" in os.environ:
        device_id = device if device.type == "cuda" else None
        dist.init_process_group(backend, timeout=timeout, device_id=device_id)
    else:
        dist.init_process_group(
            backend, store=dist.HashStore(), rank=0, world_size=1, timeout=timeout
        )
    return dist.get_rank(), dist.get_world_size(), device


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _local_flat(t: Tensor) -> tuple[Tensor, int]:
    """Return the local elements of ``t`` as a flat view, and the global flat
    index of the first local element.

    A DTensor of FSDP has dim-0 shards. Rank r of N holds the rows
    ``[r * c, (r + 1) * c)``, with ``c = ceil(rows / N)``.
    """
    if not isinstance(t, DTensor):
        return t.view(-1), 0
    local = t.to_local()
    mesh = t.device_mesh
    if mesh.ndim != 1 or not t.placements[0].is_shard(0):
        raise ValueError(f"expected a dim-0 shard on a 1-D mesh, got {t.placements}")
    rows = t.shape[0]
    per_rank = -(-rows // mesh.size())
    start = min(mesh.get_local_rank() * per_rank, rows)
    if local.shape[0] != min(per_rank, rows - start):
        raise ValueError(f"unexpected shard of {local.shape[0]} rows at row {start}")
    return local.view(-1), start * math.prod(t.shape[1:])


def round_stochastic_(x: Tensor, noise: Tensor) -> Tensor:
    """Round the float32 values of ``x`` to bfloat16 values at random, in place.

    ``noise`` holds int32 values in ``[0, 2**16)``. The function adds the
    noise to the low 16 bits of each value and then clears these bits. A value
    thus moves to the next bfloat16 value away from zero with the probability
    ``low bits / 2**16``. The expected value of the result is ``x``. The
    function keeps the sign and keeps NaN values. The result is exact in
    bfloat16.
    """
    nan = x.isnan()
    bits = x.view(torch.int32)
    bits.add_(noise).bitwise_and_(-(1 << 16))
    return x.masked_fill_(nan, float("nan"))


class AdamW(torch.optim.Optimizer):
    """AdamW for bfloat16 parameters, with stochastic rounding.

    The update runs in float32 on each parameter:

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        p = p * (1 - lr * wd) - lr / (1 - beta1**t) * m / (sqrt(v / (1 - beta2**t)) + eps)

    The optimizer keeps ``m`` and ``v`` in the dtype of the parameter. A
    bfloat16 parameter gets the new value with stochastic rounding
    (``round_stochastic_``). A plain rounding would lose most updates. A
    bfloat16 value has 8 significant bits. An update of ``lr = 5e-5`` is much
    smaller than the spacing of the values.

    The decay factor ``1 - lr * wd`` is a float32 value. When ``lr * wd`` is
    below about 3e-8, the factor is 1.0 and the weight decay has no effect.
    This is the case for ``lr`` 5e-5 or less with ``wd`` 1e-4, for example
    for the default config (3e-6 * 1e-4).

    The noise of element ``e`` of parameter ``i`` in step ``t`` comes from a
    generator with the seed ``derive_seed(seed, t, i, e // SR_BLOCK)``. Thus
    the noise does not depend on the sharding of the parameter. Parameter
    ``i`` is the position in the parameter groups, in order.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 5e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.seed = int(seed)
        self._generators: dict[torch.device, torch.Generator] = {}
        # All parameters get their state now, also the parameters that never
        # get a gradient. Thus every save holds the same state keys.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # ty: ignore[invalid-method-override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, index, group)
                index += 1
        return loss

    def _update(self, p: Tensor, index: int, group: dict) -> None:
        state = self.state[p]
        state["step"] += 1
        step = int(state["step"].item())
        lr, eps, wd = float(group["lr"]), float(group["eps"]), group["weight_decay"]
        beta1, beta2 = group["betas"]
        step_size = lr / (1 - beta1**step)
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)
        assert p.grad is not None
        param, offset = _local_flat(p)
        grad, _ = _local_flat(p.grad)
        exp_avg, _ = _local_flat(state["exp_avg"])
        exp_avg_sq, _ = _local_flat(state["exp_avg_sq"])
        n, total = param.numel(), p.numel()
        if n == 0:
            return
        block = min(SR_BLOCK, total)
        for j in range(offset // block, (offset + n - 1) // block + 1):
            lo, hi = max(offset, j * block), min(offset + n, (j + 1) * block)
            sl = slice(lo - offset, hi - offset)
            g = grad[sl].float()
            m = exp_avg[sl].float().lerp(g, 1 - beta1)
            v = exp_avg_sq[sl].float().mul(beta2).addcmul_(g, g, value=1 - beta2)
            exp_avg[sl].copy_(m)
            exp_avg_sq[sl].copy_(v)
            denom = v.sqrt_().div_(bias_correction2_sqrt).add_(eps)
            new = (
                param[sl].float().mul(1 - lr * wd).addcdiv_(m, denom, value=-step_size)
            )
            if param.dtype == torch.bfloat16:
                gen = self._generator(param.device)
                gen.manual_seed(derive_seed(self.seed, step, index, j))
                size = min(block, total - j * block)
                noise = torch.randint(
                    0,
                    1 << 16,
                    (size,),
                    generator=gen,
                    device=param.device,
                    dtype=torch.int32,
                )
                round_stochastic_(new, noise[lo - j * block : hi - j * block])
            param[sl].copy_(new)

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]


# ---------------------------------------------------------------------------
# Text cache
# ---------------------------------------------------------------------------


class TextCache:
    """Keep the text embeddings of the most recent prompts.

    The class has the ``encode`` method of ``praxis.text_encoder.TextEncoder``.
    It keeps up to ``size`` embeddings and drops the least recently used one
    first.
    """

    def __init__(self, encoder: TextEncoder, size: int = 1024) -> None:
        self.encoder = encoder
        self.size = int(size)
        self._cache: OrderedDict[str, Tensor] = OrderedDict()

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        out: list[Tensor] = []
        for prompt in prompts:
            emb = self._cache.pop(prompt, None)
            if emb is None:
                emb = self.encoder.encode([prompt])[0]
            self._cache[prompt] = emb
            out.append(emb)
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def window_tokens(batch: EncodedWindow) -> int:
    """Return the token count of one encoded window: the text tokens, one
    token per position of the anchor and chunk latents, and the action tokens."""
    video = sum(int(math.prod(x.shape[2:])) for x in [batch.anchor, *batch.chunks])
    return int(batch.text.shape[1]) + video + int(batch.actions.shape[1])


def _checkpoint_forward(module: nn.Module) -> None:
    """Recompute the forward pass of ``module`` in the backward pass.

    The forward method of the module runs under
    ``torch.utils.checkpoint.checkpoint`` (``use_reentrant=False``). The
    parameter names do not change. The recompute restores the global RNG
    states. Thus code in the module must not draw from other generators.
    """
    module.forward = functools.partial(checkpoint, module.forward, use_reentrant=False)


def _find_policy(model: nn.Module) -> tuple[str, PraxisPolicy]:
    """Return the name prefix and the ``PraxisPolicy`` module in ``model``."""
    found = [(n, m) for n, m in model.named_modules() if isinstance(m, PraxisPolicy)]
    if len(found) != 1:
        raise ValueError(f"the model must hold one PraxisPolicy, found {len(found)}")
    name, policy = found[0]
    assert isinstance(policy, PraxisPolicy)
    return (f"{name}." if name else ""), policy


class Trainer:
    """FSDP2 training of a model that holds one ``PraxisPolicy``.

    ``model(batch, generator)`` must return ``(loss, metrics)``: the scalar
    loss of one window and a dict of scalar tensors. ``blocks`` are the FSDP
    units and the activation checkpointing units (``FinetuneModel.blocks()``).
    The constructor shards the model on all ranks of the default process
    group. Call it on every rank with the same model and the same weights.
    The model can be on the CPU: FSDP moves each unit to ``device``. The
    optimizer trains the parameters with ``requires_grad``. If no parameter
    has it (a policy from ``from_pretrained``), the trainer sets it on all
    parameters.

    ``post_step(model)`` runs after each optimizer step, for example the
    update of the MoE routing biases.

    ``fixed_draws`` true seeds the noise generator of a window from
    ``(settings.seed, FIXED_DRAWS_KEY, batch.window_id)``. Each window from
    ``next_batch`` must then have a ``window_id``. False (the default) seeds
    it from ``(settings.seed, step, micro * world + rank)``.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: Sequence[nn.Module],
        settings: TrainSettings,
        device: torch.device,
        post_step: Callable[[nn.Module], None] | None = None,
        fixed_draws: bool = False,
    ) -> None:
        self.settings = settings
        self.device = device
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.post_step = post_step
        self.fixed_draws = bool(fixed_draws)
        self.step = 0
        self.micro = 0
        self.policy_prefix, policy = _find_policy(model)
        self.config = policy.config
        # The save writes only the policy. Check this on every rank now: a
        # failure in the save on rank 0 would block the other ranks.
        if other := [
            k for k in model.state_dict() if not k.startswith(self.policy_prefix)
        ]:
            raise ValueError(f"the model holds tensors outside the policy: {other}")
        torch.manual_seed(derive_seed(settings.seed, self.rank))
        if not any(p.requires_grad for p in model.parameters()):
            model.requires_grad_(True)
        model.train()
        mesh = init_device_mesh(device.type, (self.world,))
        mp = MixedPrecisionPolicy(reduce_dtype=REDUCE_DTYPES[settings.reduce_dtype])
        for block in blocks:
            if settings.activation_checkpointing:
                _checkpoint_forward(block)
            fully_shard(block, mesh=mesh, mp_policy=mp)
        self.fsdp = fully_shard(model, mesh=mesh, mp_policy=mp)
        self.model = model
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=settings.lr,
            betas=settings.betas,
            eps=settings.eps,
            weight_decay=settings.weight_decay,
            seed=settings.seed,
        )

    def window_index(self) -> int:
        """Return the global index of the window of the current micro-step.

        The index is ``(step * grad_accum + micro) * world + rank``. Each
        window of a run has its own index. ``next_batch`` can call this
        method.
        """
        return (self.step * self.settings.grad_accum + self.micro) * self.world + (
            self.rank
        )

    def train_step(self, next_batch: Callable[[], EncodedWindow]) -> dict[str, float]:
        """Run one optimizer step. ``next_batch()`` gives the window of a micro-step.

        Returns the metrics of the step. The losses and the model metrics are
        means over the micro-steps and the ranks. ``tokens`` and ``windows``
        are sums over the ranks.
        """
        s, n = self.settings, self.settings.grad_accum
        t_start = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        data_time = 0.0
        tokens = 0
        sums: dict[str, Tensor] = {}
        for micro in range(n):
            self.micro = micro
            if s.no_sync:
                self.fsdp.set_requires_gradient_sync(micro == n - 1)
            t0 = time.monotonic()
            batch = next_batch()
            data_time += time.monotonic() - t0
            tokens += window_tokens(batch)
            if self.fixed_draws:
                if batch.window_id is None:
                    raise ValueError("fixed_draws needs a window_id on each window")
                seed = derive_seed(s.seed, FIXED_DRAWS_KEY, batch.window_id)
            else:
                seed = derive_seed(s.seed, self.step, micro * self.world + self.rank)
            loss, metrics = self.model(batch, torch.Generator().manual_seed(seed))
            (loss / n).backward()
            for key, value in {"loss": loss, **metrics}.items():
                part = value.detach().float().reshape(()) / n
                sums[key] = sums[key] + part if key in sums else part
        params = [p for p in self.model.parameters() if p.grad is not None]
        grads = [g for p in params if (g := p.grad) is not None]
        norm = torch.nn.utils.get_total_norm(grads)
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        if not torch.isfinite(norm):
            raise FloatingPointError(f"step {self.step}: the gradient norm is {norm}")
        if s.clip_grad_norm is not None:
            torch.nn.utils.clip_grads_with_norm_(params, s.clip_grad_norm, norm)
        lr = s.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.post_step is not None:
            self.post_step(self.model)
        self.step += 1
        self.micro = 0

        keys = sorted(sums)
        means = torch.stack([sums[k].to(self.device) for k in keys])
        dist.all_reduce(means)
        means /= self.world
        counts = torch.tensor([float(tokens), float(n)], device=self.device)
        dist.all_reduce(counts)
        peak = 0.0
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30
        maxima = torch.tensor([data_time, peak], device=self.device)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        step_time = time.monotonic() - t_start
        out = dict(zip(keys, means.tolist(), strict=True))
        out.update(
            step=float(self.step),
            lr=lr,
            grad_norm=float(norm),
            tokens=counts[0].item(),
            windows=counts[1].item(),
            step_time=step_time,
            data_time=maxima[0].item(),
            tokens_per_s=counts[0].item() / step_time,
            peak_mem_gib=maxima[1].item(),
        )
        return out

    def save(self, out_dir: str | Path) -> Path:
        """Save the state after the current step to ``out_dir/step-NNNNNN``.

        * ``model/``: the policy weights as a praxis checkpoint. The config is
          the config of the start checkpoint. ``PraxisPolicy.from_pretrained``
          loads it.
        * ``optimizer/``: the sharded optimizer state (torch distributed
          checkpoint).
        * ``trainer_state.json``: the step.

        The files go into ``step-NNNNNN.tmp``. When all files are complete,
        rank 0 renames the directory. A save that stops before the end leaves
        only the ``.tmp`` directory, and the next save at that step removes it.
        Rank 0 holds the full weights in CPU memory during the save.
        """
        path = Path(out_dir) / f"step-{self.step:06d}"
        tmp = path.with_name(path.name + ".tmp")
        if path.exists():
            raise FileExistsError(f"{path} exists")
        if self.rank == 0 and tmp.exists():
            shutil.rmtree(tmp)
        if self.device.type == "cuda":
            # The optimizer save sends objects between ranks. NCCL then
            # allocates its own GPU buffers, outside the PyTorch allocator.
            # Give the cached blocks of the allocator back to the GPU first.
            torch.cuda.empty_cache()
        dist.barrier()
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state = get_model_state_dict(self.model, options=options)
        if self.rank == 0:
            cut = len(self.policy_prefix)
            weights = {k[cut:]: v for k, v in state.items() if isinstance(v, Tensor)}
            save_checkpoint(tmp / "model", weights, self.config)
        del state
        optim = get_optimizer_state_dict(self.model, self.optimizer)
        dcp.save({"optimizer": optim}, checkpoint_id=tmp / "optimizer")
        if self.rank == 0:
            meta = {"step": self.step, "settings": asdict(self.settings)}
            (tmp / "trainer_state.json").write_text(json.dumps(meta, indent=2))
        dist.barrier()
        if self.rank == 0:
            tmp.rename(path)
        dist.barrier()
        return path

    def load(self, path: str | Path) -> None:
        """Load the optimizer state and the step of a save of ``save``.

        The weights do not come from this call: build the model from
        ``path/model`` first. The optimizer settings (betas, eps, weight
        decay) come from ``settings``, not from the save.
        """
        path = Path(path)
        state = {"optimizer": get_optimizer_state_dict(self.model, self.optimizer)}
        dcp.load(state, checkpoint_id=path / "optimizer")
        set_optimizer_state_dict(self.model, self.optimizer, state["optimizer"])
        s = self.settings
        for group in self.optimizer.param_groups:
            group.update(betas=s.betas, eps=s.eps, weight_decay=s.weight_decay)
        self.step = int(json.loads((path / "trainer_state.json").read_text())["step"])

    def fit(self, next_batch: Callable[[], EncodedWindow], out_dir: str | Path) -> None:
        """Train until ``settings.steps`` steps. Log and save on the way.

        Rank 0 prints the metrics and appends them to ``out_dir/metrics.jsonl``.
        The loop saves every ``save_every`` steps and after the last step.
        With ``settings.save`` false, the loop does not save.
        """
        s, out = self.settings, Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        later = [d.name for d in out.glob("step-*") if _save_step(d.name) > self.step]
        if later and s.save:  # a save at one of these steps would fail
            raise FileExistsError(f"{out} holds saves of later steps: {sorted(later)}")
        while self.step < s.steps:
            metrics = self.train_step(next_batch)
            if self.rank == 0 and (self.step % s.log_every == 0 or self.step == 1):
                print(_format(metrics), flush=True)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
            if not s.save:
                continue
            if self.step % s.save_every == 0 or self.step == s.steps:
                path = self.save(out)
                if self.rank == 0:
                    print(f"saved {path}", flush=True)


def cycle_windows(
    trainer: Trainer, windows: Sequence[EncodedWindow]
) -> Callable[[], EncodedWindow]:
    """Return a ``next_batch`` function that serves a fixed set of windows.

    The window of rank r at micro-step k of step s is window
    ``((s * grad_accum + k) * world + r) % len(windows)``. Thus the run uses
    all windows in turn on any number of GPUs. Give each window its position
    in the list as its ``window_id``.
    """
    if not windows:
        raise ValueError("cycle_windows needs at least one window")
    for i, w in enumerate(windows):
        if w.window_id != i:
            raise ValueError(f"window {i} has the window_id {w.window_id}")

    def next_batch() -> EncodedWindow:
        return windows[trainer.window_index() % len(windows)]

    return next_batch


def broadcast_windows(
    windows: list[EncodedWindow] | None, device: torch.device, src: int = 0
) -> list[EncodedWindow]:
    """Send the windows of rank ``src`` to all ranks. Return them on ``device``.

    Rank ``src`` gives the list. The other ranks give None. The tensors go
    through the CPU, so all ranks get the same bits.
    """
    rank = dist.get_rank()
    if (windows is None) != (rank != src):
        raise ValueError("only rank src gives the windows")
    box: list[object] = [None]
    if rank == src:
        assert windows is not None
        box[0] = [_window_to(w, torch.device("cpu")) for w in windows]
    dist.broadcast_object_list(box, src=src, device=device)
    got = box[0]
    assert isinstance(got, list)
    return [_window_to(w, device) for w in got]


def _window_to(w: EncodedWindow, device: torch.device) -> EncodedWindow:
    """Return a copy of ``w`` with all tensors on ``device`` (same dtypes)."""
    return dataclasses.replace(
        w,
        text=w.text.to(device),
        anchor=w.anchor.to(device),
        chunks=[c.to(device) for c in w.chunks],
        actions=w.actions.to(device),
        proprio=w.proprio.to(device),
    )


def _save_step(name: str) -> int:
    """The step of a complete save ``step-NNNNNN``. -1 for other names."""
    digits = name.removeprefix("step-")
    return int(digits) if digits.isdigit() else -1


def _format(m: dict[str, float]) -> str:
    """One log line of the step metrics."""
    head = (
        f"step {int(m['step'])} loss {m['loss']:.4f} grad_norm {m['grad_norm']:.3f} "
        f"lr {m['lr']:.2e} {m['step_time']:.2f} s/step (data {m['data_time']:.2f} s) "
        f"{m['tokens_per_s']:.0f} tok/s mem {m['peak_mem_gib']:.1f} GiB"
    )
    fixed = {"step", "loss", "grad_norm", "lr", "step_time", "data_time"}
    fixed |= {"tokens_per_s", "peak_mem_gib", "tokens", "windows"}
    rest = " ".join(f"{k} {v:.4g}" for k, v in m.items() if k not in fixed)
    return f"{head} {rest}".rstrip()
