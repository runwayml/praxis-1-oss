"""The config checks of the checkpoint loader (CPU only, no weights).

The config of the single-arm checkpoint loads. A config of another rig and
a single-arm config of another model variant are refused.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from praxis.checkpoint import (
    OTHER_RIG_REFUSED,
    OTHER_VARIANT_REFUSED,
    CheckpointError,
    load_config,
    model_variant,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
# The checkpoints (``PRAXIS_CHECKPOINT_DIR`` replaces the directory).
CHECKPOINTS = Path(os.environ.get("PRAXIS_CHECKPOINT_DIR", ROOT / "checkpoints"))
REFUSED = OTHER_VARIANT_REFUSED.split("(")[0].strip()
CHECKPOINT_CONFIG = Path(__file__).resolve().parent / "data" / "praxis-1-droid.json"

# The variant fields of a config of another rig (format 1, no embodiment).
OTHER_RIG: dict = {
    "format_version": 1,
    "model_type": "praxis",
    "backbone": {"rope": {"n_chunks": 4}},
    "action_expert": {
        "rope_cross": True,
        "rope_self": True,
        "rope_base": 100.0,
        "absolute_positions": False,
    },
    "chunking": {"n_chunks": 4, "context_format": "tv"},
    "video": {
        "height": 256,
        "width": 1248,
        "views": ["top_left", "wrist_left", "wrist_right"],
        "layout": "hstack",
        "keep_aspect": True,
    },
    "action_space": {"dim": 20},
}

# The variant fields of the single-arm checkpoint config.
SINGLE_ARM: dict = {
    "format_version": 2,
    "model_type": "praxis",
    "embodiment": "single_arm",
    "backbone": {
        "attention": {"backend": "flash"},
        "rope": {"n_chunks": None, "ref_placement": "group"},
        "moe": {"dispatch_chunk_size": 16384},
        "layout": "chunk_causal",
    },
    "action_expert": {
        "head_dim": 64,
        "rope_cross": False,
        "rope_self": False,
        "absolute_positions": True,
        "action_pos_table": False,
        "future_chunk_offset": "zero",
        "rope": {
            "kind": "trunk",
            "dim": 64,
            "effective_dim": 64,
            "base": 26000000,
            "table_size": 1000000,
            "order": "before_norm",
        },
        "global_order": ["register", "anchor", "text"],
        "attention": "flex_dynamic",
        "flex_kernel_options": {
            "fwd_BLOCK_M": 32,
            "fwd_BLOCK_N": 128,
            "num_warps": 4,
            "num_stages": 3,
        },
        "head_plus_zero": True,
        "layout_stamp": 20260924,
    },
    "video": {
        "height": 256,
        "width": 832,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "keep_aspect": True,
        "tile_height": 256,
        "tile_width": 416,
        "aspect_tol": 0.03,
        "vae_encode_impl": "megakernel",
        "vae_megakernel_chunk_frames": 16,
    },
    "text_encoder": {"uncond": "blank_caption", "blank_caption": " "},
    "pose_frame": "flange",
}


def _with(base: dict, path: str, value: object) -> dict:
    out = copy.deepcopy(base)
    node = out
    *head, last = path.split(".")
    for key in head:
        node = node[key]
    node[last] = value
    return out


def test_single_arm_config_loads() -> None:
    validate_config(SINGLE_ARM)
    variant = model_variant(SINGLE_ARM)
    kwargs = variant.expert_kwargs()
    assert set(kwargs) == {
        "rope",
        "global_order",
        "attention",
        "flex_kernel_options",
        "head_plus_zero",
        "layout_stamp",
    }
    assert variant.backbone_layout == "chunk_causal" and variant.canvas == "tiles"
    assert variant.tile_size == (256, 416) and variant.pose_frame == "flange"
    assert variant.text_uncond == "blank_caption" and variant.blank_caption == " "


def test_checkpoint_config_validates() -> None:
    """The config.json of the checkpoint (a copy in tests/data) loads unchanged."""
    config = json.loads(CHECKPOINT_CONFIG.read_text())
    assert config["embodiment"] == "single_arm" and config["format_version"] == 2
    validate_config(config)
    variant = model_variant(config)
    assert variant.attention_backend == "flash"
    assert variant.moe_dispatch_chunk_size == 16384
    assert variant.vae_encode_impl == "megakernel"
    assert variant.vae_megakernel_chunk_frames == 16
    triton = _with(config, "backbone.attention.backend", "triton")
    validate_config(triton)
    assert model_variant(triton).attention_backend == "triton"


@pytest.mark.parametrize(
    "config",
    [
        OTHER_RIG,
        _with(OTHER_RIG, "embodiment", "two_arm"),
        {k: v for k, v in SINGLE_ARM.items() if k != "embodiment"},
        _with(SINGLE_ARM, "embodiment", "two_arm"),
    ],
)
def test_other_rig_config_is_refused(config: dict) -> None:
    with pytest.raises(CheckpointError) as err:
        validate_config(config)
    assert str(err.value) == OTHER_RIG_REFUSED
    assert '"embodiment": "single_arm"' in str(err.value)
    assert "single-arm checkpoint" in str(err.value)
    if "embodiment" not in config and config["format_version"] == 1:
        with pytest.raises(CheckpointError, match="single_arm"):
            model_variant(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("format_version", 1),
        ("backbone.layout", "serve"),
        ("video.canvas", "joint"),
        ("pose_frame", "flange_turn_-45"),
        ("action_expert.action_pos_table", True),
        ("action_expert.rope", None),
    ],
)
def test_first_single_arm_config_is_refused(path: str, value: object) -> None:
    with pytest.raises(CheckpointError, match=REFUSED):
        validate_config(_with(SINGLE_ARM, path, value))


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("action_expert.absolute_positions", False, "single_arm"),
        ("action_expert.future_chunk_offset", None, REFUSED),
        ("action_expert.future_chunk_offset", "serve", "future_chunk_offset"),
        ("action_expert.rope_cross", True, "rope_cross"),
        ("action_expert.global_order", ["register", "text", "anchor"], "global"),
        ("action_expert.layout_stamp", 1, "layout_stamp"),
        ("backbone.rope.ref_placement", None, "ref_placement"),
        ("backbone.rope.n_chunks", 4, "n_chunks"),
        ("video.width", 1248, "canvas"),
        ("video.vae_encode_impl", "eager", "vae_encode_impl"),
        ("text_encoder.uncond", "zero_embedding", "uncond"),
        ("format_version", 3, "format_version"),
    ],
)
def test_other_variants_are_refused(path: str, value: object, match: str) -> None:
    with pytest.raises(CheckpointError, match=match):
        validate_config(_with(SINGLE_ARM, path, value))


def test_checkpoint_dir_config() -> None:
    """A local copy of the checkpoint has the config of tests/data."""
    path = CHECKPOINTS / "praxis-1-droid"
    if not (path / "config.json").is_file():
        pytest.skip(f"{path} is not available")
    assert load_config(path) == json.loads(CHECKPOINT_CONFIG.read_text())
