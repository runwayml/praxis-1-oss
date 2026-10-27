"""Low-rank adapters (LoRA) for the fine-tune.

A LoRA fine-tune keeps the weights of the checkpoint frozen and trains a
low-rank update ``delta_W = (alpha / rank) * B @ A`` of selected
``nn.Linear`` layers. ``A`` is ``(rank, in)`` and ``B`` is ``(out, rank)``.
``B`` starts at zero, so at step 0 the model is the checkpoint, bit for bit.

``apply_lora`` replaces each target layer by a ``LoRALinear`` that holds the
frozen layer and the two adapter matrices. The parameter names of the frozen
layer change from ``<name>.weight`` to ``<name>.base.weight``.
``merged_state_dict`` gives the weights of the policy with each update added
to its layer, under the names of the checkpoint. The trainer saves this state
dict, so a LoRA save is a full praxis checkpoint that
``PraxisPolicy.from_pretrained``, the pipeline, the server and the
simulation load as they load any other checkpoint.

The targets are name patterns (``fnmatch``) of ``nn.Linear`` modules of the
policy, for example ``action_expert.blocks.*.to_q``. ``LORA_TARGETS`` lists
named groups.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = [
    "LORA_TARGETS",
    "LoRALinear",
    "LoRASettings",
    "apply_lora",
    "lora_parameters",
    "merged_state_dict",
]

#: Named target groups. A group name in ``LoRASettings.targets`` stands for
#: its patterns.
LORA_TARGETS: dict[str, tuple[str, ...]] = {
    # The attention and MLP projections of the action expert.
    "expert": (
        "action_expert.blocks.*.to_q",
        "action_expert.blocks.*.to_kv",
        "action_expert.blocks.*.to_out",
        "action_expert.blocks.*.mlp_in",
        "action_expert.blocks.*.mlp_out",
    ),
    # The attention output projections of both backbone towers.
    "backbone_attention": (
        "backbone.encoder.core.blocks.*.mha.out_proj",
        "backbone.decoder.core.blocks.*.mha.out_proj",
    ),
}


@dataclass
class LoRASettings:
    """The ``lora`` section of the run config.

    ``rank`` 0 turns LoRA off (a full fine-tune). ``targets`` holds group
    names of ``LORA_TARGETS`` or ``fnmatch`` patterns of ``nn.Linear`` module
    names in the policy. ``train_also`` holds ``fnmatch`` patterns of
    parameter names of the policy that train in full next to the adapters
    (for example ``action_expert.head.*``); every other weight is frozen.
    """

    rank: int = 0
    alpha: float = 16.0
    dropout: float = 0.0
    targets: list[str] = field(default_factory=lambda: ["expert"])
    train_also: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        self.alpha = float(self.alpha)
        self.dropout = float(self.dropout)
        self.targets = [str(t) for t in self.targets]
        self.train_also = [str(t) for t in self.train_also]
        if self.rank < 0:
            raise ValueError(f"lora.rank must be >= 0, got {self.rank}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"lora.dropout must be in [0, 1), got {self.dropout}")
        if self.rank > 0 and not self.targets:
            raise ValueError("lora.targets is empty")

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    def patterns(self) -> list[str]:
        """Return the target patterns, with each group name expanded."""
        out: list[str] = []
        for t in self.targets:
            out.extend(LORA_TARGETS.get(t, (t,)))
        return out


class LoRALinear(nn.Module):
    """An ``nn.Linear`` with a trainable low-rank update.

    ``forward(x) = base(x) + scale * B(A(dropout(x)))`` with
    ``scale = alpha / rank``. The base layer is frozen. ``A`` gets the
    Kaiming-uniform start of ``nn.Linear``; ``B`` starts at zero.
    """

    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        w = base.weight
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=w.dtype, device=w.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=w.dtype, device=w.device)
        )
        if w.device.type != "meta":
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        base.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        delta = (self.dropout(x) @ self.lora_a.t()) @ self.lora_b.t()
        return out + self.scale * delta

    def delta_weight(self) -> Tensor:
        """Return ``scale * B @ A`` in float32, with the shape of the weight."""
        return self.scale * (self.lora_b.float() @ self.lora_a.float())


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def apply_lora(policy: nn.Module, settings: LoRASettings) -> list[str]:
    """Freeze ``policy`` and wrap its target layers in ``LoRALinear``.

    After the call only the adapter matrices and the parameters that match
    ``settings.train_also`` have ``requires_grad``. Returns the names of the
    wrapped modules. Raises ``ValueError`` when no module matches a pattern,
    or when a pattern matches a module that is not an ``nn.Linear``.
    """
    if not settings.enabled:
        raise ValueError("lora.rank is 0: LoRA is off")
    patterns = settings.patterns()
    policy.requires_grad_(False)
    named = dict(policy.named_modules())
    targets = [n for n in named if n and _matches(n, patterns)]
    for p in patterns:
        if not any(fnmatch.fnmatchcase(n, p) for n in targets):
            raise ValueError(f"lora target {p!r} matches no module")
    bad = [n for n in targets if not isinstance(named[n], nn.Linear)]
    if bad:
        raise ValueError(f"lora targets must be nn.Linear modules: {bad[:5]}")
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = named[parent_name] if parent_name else policy
        base = named[name]
        assert isinstance(base, nn.Linear)
        setattr(
            parent,
            leaf,
            LoRALinear(base, settings.rank, settings.alpha, settings.dropout),
        )
    if settings.train_also:
        found = False
        for pname, param in policy.named_parameters():
            if _matches(pname, settings.train_also) and ".lora_" not in pname:
                param.requires_grad_(True)
                found = True
        if not found:
            raise ValueError(f"lora.train_also {settings.train_also} matches nothing")
    return targets


def lora_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return the named adapter parameters of ``module``."""
    return [
        (n, p)
        for n, p in module.named_parameters()
        if n.rsplit(".", 1)[-1].startswith("lora_")
    ]


def merged_state_dict(
    state: Mapping[str, Tensor], lora_scales: Mapping[str, float]
) -> dict[str, Tensor]:
    """Fold the adapters of a full state dict into the base weights.

    ``state`` is a state dict of a policy after ``apply_lora`` (full tensors,
    for example on the CPU). ``lora_scales`` maps each wrapped module name to
    its ``alpha / rank``. The result has the names of the checkpoint:
    ``<m>.base.weight`` becomes ``<m>.weight`` with ``scale * B @ A`` added in
    float32 and rounded to the dtype of the weight; ``<m>.base.bias`` becomes
    ``<m>.bias``; the adapter tensors are dropped. Other tensors are unchanged.
    """
    out: dict[str, Tensor] = {}
    for name, value in state.items():
        module, _, leaf = name.rpartition(".")
        if leaf in ("lora_a", "lora_b"):
            continue
        if module.endswith(".base") and module[: -len(".base")] in lora_scales:
            m = module[: -len(".base")]
            if leaf == "weight":
                a, b = state[f"{m}.lora_a"], state[f"{m}.lora_b"]
                delta = lora_scales[m] * (b.float() @ a.float())
                value = (value.float() + delta).to(value.dtype)
            out[f"{m}.{leaf}"] = value
        else:
            out[name] = value
    return out
