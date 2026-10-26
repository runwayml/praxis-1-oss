"""A small random praxis policy for the CPU tests of the training code.

The config has the layout of the single-arm checkpoint: the
chunk-causal backbone, the trunk-rope expert, chunked MoE dispatch and the
encoded blank caption as the unconditional text. The dims are small.
"""

from __future__ import annotations

import torch

from praxis.backbone import TEXT, VIDEO
from praxis.checkpoint import EXPERT_TRUNK_ROPE, LAYOUT_STAMP_CHUNK_CAUSAL
from praxis.model import PraxisPolicy

#: The latent grid of the tiny windows: (h, w) of the encoder tokens.
TINY_HW = (2, 4)


def tiny_config() -> dict:
    """A praxis config with small dims. The dims are multiples of 8, so that
    the grouped GEMM of the experts runs on the CPU."""
    attention = {"num_heads": 4, "num_kv_heads": 2, "head_dim": 16}
    attention["flex_block_size"] = 128
    moe = {"num_routed_experts": 4, "top_k_routed": 2, "expert_hidden": 32}
    moe["routed_scaling_factor"] = 2.5
    moe["dispatch_chunk_size"] = 40
    projections = {TEXT: {"c_in": 32}, VIDEO: {"c_in": 192}}
    backbone = {
        "layout": "chunk_causal",
        "dim": 64,
        "timestep_dim": 16,
        "num_registers": 4,
        "modulation_dtype": "mixed",
        "dense_ffn_hidden": 96,
        "attention": attention,
        "rope": {
            "effective_dim": 8,
            "base": 26_000_000,
            "efficient_cache_max": 10**6,
            "ref_placement": "group",
        },
        "moe": moe,
        "encoder": {
            "depth": 3,
            "dense_blocks": [0],
            "tap_blocks": [0, 2],
            "input_projections": projections,
            "positive_zero_residual_add": True,
        },
        "decoder": {
            "depth": 2,
            "dense_blocks": [0],
            "modulation_dim": 64,
            "unpatch_factor": [1, 2, 2],
            "input_projections": {VIDEO: {"c_in": 48}},
            "positive_zero_residual_add": True,
        },
    }
    expert = {
        "dim": 64,
        "num_layers": 4,
        "num_heads": 1,
        "head_dim": 64,
        "trunk_dim": 64,
        "action_patch_dim": 40,
        "proprio_patch_dim": 10,
        "num_future_per_chunk": 2,
        "max_chunks": 16,
        "max_action_tokens": 64,
        "tap_norm": True,
        "clean_only_conditioning": False,
        "rope_cross": False,
        "rope_self": False,
        "rope_base": 100.0,
        "absolute_positions": True,
        "action_pos_table": False,
        "future_chunk_offset": "zero",
        "rope": dict(EXPERT_TRUNK_ROPE),
        "global_order": ["register", "anchor", "text"],
        "attention": "sdpa",
        "head_plus_zero": True,
        "layout_stamp": LAYOUT_STAMP_CHUNK_CAUSAL,
    }
    return {
        "format_version": 2,
        "model_type": "praxis",
        "embodiment": "single_arm",
        "backbone": backbone,
        "action_expert": expert,
        "action_space": {"dim": 10},
        "chunking": {"n_chunks": 4, "latents_per_chunk": 6, "action_chunk": 24},
        "text_encoder": {"uncond": "blank_caption", "blank_caption": " "},
    }


def tiny_policy(seed: int = 0, dtype: torch.dtype = torch.bfloat16) -> PraxisPolicy:
    """A random policy in ``dtype``. The routing biases stay float32."""
    torch.manual_seed(seed)
    policy = PraxisPolicy.from_config(tiny_config())
    with torch.no_grad():
        for p in policy.parameters():
            p.normal_(0.0, 0.05)
        for name, buf in policy.named_buffers():
            if name.endswith("expert_bias"):
                buf.normal_(0.0, 0.01)
            elif name.endswith(("freqs", "phases")):
                buf.uniform_(-1.0, 1.0)
    return policy.to(dtype)
