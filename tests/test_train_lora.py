"""LoRA adapters of the fine-tune (CPU only, no weights)."""

from __future__ import annotations

import pytest
import torch
from tiny_model import tiny_policy
from torch import nn

from praxis.train.lora import (
    LORA_TARGETS,
    LoRALinear,
    LoRASettings,
    apply_lora,
    lora_parameters,
    merged_state_dict,
)


def test_settings() -> None:
    assert not LoRASettings().enabled
    s = LoRASettings(rank="8", alpha="16", targets=["expert", "x.*.y"])  # ty: ignore[invalid-argument-type]
    assert s.enabled and s.rank == 8 and s.alpha == 16.0
    assert s.patterns() == [*LORA_TARGETS["expert"], "x.*.y"]
    with pytest.raises(ValueError):
        LoRASettings(rank=-1)
    with pytest.raises(ValueError):
        LoRASettings(rank=4, dropout=1.0)
    with pytest.raises(ValueError):
        LoRASettings(rank=4, targets=[])


def test_lora_linear_starts_as_the_base_layer() -> None:
    torch.manual_seed(0)
    base = nn.Linear(12, 7)
    x = torch.randn(3, 12)
    ref = base(x)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    assert torch.equal(lora(x), ref)  # B = 0
    assert not base.weight.requires_grad and lora.lora_a.requires_grad
    assert lora.scale == 2.0


def test_lora_linear_is_base_plus_the_low_rank_update() -> None:
    torch.manual_seed(1)
    base = nn.Linear(12, 7)
    lora = LoRALinear(base, rank=3, alpha=6.0)
    with torch.no_grad():
        lora.lora_b.normal_()
    x = torch.randn(5, 12)
    w = base.weight + lora.delta_weight()
    torch.testing.assert_close(lora(x), x @ w.t() + base.bias, rtol=1e-5, atol=1e-5)


def test_apply_lora_freezes_all_but_the_adapters() -> None:
    policy = tiny_policy(dtype=torch.float32)
    wrapped = apply_lora(policy, LoRASettings(rank=2, targets=["expert"]))
    n_blocks = len(policy.action_expert.blocks)
    assert len(wrapped) == 5 * n_blocks
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)
    assert len(lora_parameters(policy)) == 2 * len(wrapped)
    assert isinstance(policy.action_expert.blocks[0].to_q, LoRALinear)


def test_apply_lora_train_also_and_errors() -> None:
    policy = tiny_policy(dtype=torch.float32)
    apply_lora(
        policy,
        LoRASettings(rank=2, targets=["expert"], train_also=["action_expert.head.*"]),
    )
    head = policy.action_expert.head
    assert head.weight.requires_grad and head.bias.requires_grad
    with pytest.raises(ValueError, match="matches no module"):
        apply_lora(tiny_policy(), LoRASettings(rank=2, targets=["no.such.layer"]))
    with pytest.raises(ValueError, match=r"nn\.Linear"):
        apply_lora(
            tiny_policy(), LoRASettings(rank=2, targets=["action_expert.blocks.0"])
        )


def test_lora_policy_output_at_step_0_is_the_base() -> None:
    """With B = 0 every wrapped layer returns the bits of the base layer."""
    policy = tiny_policy(dtype=torch.float32)
    blk = policy.action_expert.blocks[0]
    x = torch.randn(2, 5, policy.config["action_expert"]["dim"])
    ref = {n: getattr(blk, n)(x) for n in ("to_q", "mlp_in")}
    apply_lora(policy, LoRASettings(rank=4, targets=["expert"]))
    for n, r in ref.items():
        assert torch.equal(getattr(blk, n)(x), r), n


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merged_state_dict_has_the_checkpoint_names(dtype: torch.dtype) -> None:
    policy = tiny_policy(dtype=dtype)
    names = set(policy.state_dict())
    s = LoRASettings(rank=2, alpha=4.0, targets=["expert"])
    wrapped = apply_lora(policy, s)
    with torch.no_grad():
        for _, p in lora_parameters(policy):
            p.normal_(0.0, 0.1)
    scales = {n: s.alpha / s.rank for n in wrapped}
    state = {k: v.detach() for k, v in policy.state_dict().items()}
    merged = merged_state_dict(state, scales)
    assert set(merged) == names
    m = wrapped[0]
    mod = policy.get_submodule(m)
    assert isinstance(mod, LoRALinear)
    want = (mod.base.weight.float() + mod.delta_weight()).to(dtype)
    assert torch.equal(merged[f"{m}.weight"], want)
    assert merged[f"{m}.weight"].dtype == dtype
    # A layer outside the targets keeps its bytes.
    assert torch.equal(
        merged["action_expert.head.weight"], state["action_expert.head.weight"]
    )


def test_merged_policy_matches_the_lora_policy() -> None:
    """A policy that loads the merged weights gives the output of the LoRA policy."""
    from praxis.model import PraxisPolicy

    policy = tiny_policy(dtype=torch.float32)
    s = LoRASettings(rank=3, alpha=6.0, targets=["expert"])
    wrapped = apply_lora(policy, s)
    with torch.no_grad():
        for _, p in lora_parameters(policy):
            p.normal_(0.0, 0.05)
    merged = merged_state_dict(
        {k: v.detach() for k, v in policy.state_dict().items()},
        {n: s.alpha / s.rank for n in wrapped},
    )
    plain = PraxisPolicy.from_config(policy.config)
    plain.load_state_dict(merged)
    plain.float()
    name = "action_expert.blocks.1.mlp_in"
    lora_mlp, plain_mlp = policy.get_submodule(name), plain.get_submodule(name)
    x = torch.randn(2, 4, policy.config["action_expert"]["dim"])
    torch.testing.assert_close(lora_mlp(x), plain_mlp(x), rtol=1e-5, atol=1e-5)
