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
# The so101 checkpoint (joint space, 6-D head, 480x640 views): a copy of its
# config.json.
SO101_CONFIG = Path(__file__).resolve().parent / "data" / "praxis-1-so101.json"

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


def _tall_tiles(config: dict) -> dict:
    """The same model with 256x416 tiles and a resize without antialias."""
    out = _with(config, "video.antialias", False)
    out = _with(out, "video.height", 256)
    out = _with(out, "video.tile_height", 256)
    out = _with(out, "video.tile_bounds_rows_cols.exterior", [0, 256, 0, 416])
    return _with(out, "video.tile_bounds_rows_cols.wrist", [0, 256, 416, 832])


@pytest.mark.parametrize(
    ("tall", "tile_size", "antialias"),
    [(False, (224, 416), True), (True, (256, 416), False)],
)
def test_checkpoint_config_validates(
    tall: bool, tile_size: tuple[int, int], antialias: bool
) -> None:
    """The config.json of the checkpoint (a copy in tests/data) loads unchanged.

    The second case changes it to 256x416 tiles without antialias.
    """
    config = json.loads(CHECKPOINT_CONFIG.read_text())
    if tall:
        config = _tall_tiles(config)
    assert config["embodiment"] == "single_arm" and config["format_version"] == 2
    validate_config(config)
    variant = model_variant(config)
    assert variant.tile_size == tile_size and variant.antialias is antialias
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
    assert '"embodiment": "single_arm", "bimanual" or "so101"' in str(err.value)
    assert "the single-arm, the bimanual or an so101 checkpoint" in str(err.value)
    if "embodiment" not in config and config["format_version"] == 1:
        with pytest.raises(CheckpointError, match="single_arm"):
            model_variant(config)


def test_so101_config_validates() -> None:
    """The config.json of the so101 checkpoint (a copy in tests/data) loads
    unchanged: the ``so101`` rig, ``pose_frame`` ``joints``, a 6-D head and
    288x416 tiles (the fit of a 480x640 view) of the views exterior, wrist."""
    config = json.loads(SO101_CONFIG.read_text())
    assert config["embodiment"] == "so101" and config["format_version"] == 2
    validate_config(config)
    variant = model_variant(config)
    assert variant.tile_size == (288, 416) and variant.antialias is True
    assert variant.pose_frame == "joints" and variant.canvas == "tiles"
    assert variant.attention_backend == "flash"
    assert config["action_space"]["dim"] == 6
    assert config["video"]["views"] == ["exterior", "wrist"]
    assert config["normalization"]["q99_normed_dims"] == [0, 1, 2, 3, 4, 5]
    # A config without the field takes the frame of its rig.
    without = {k: v for k, v in config.items() if k != "pose_frame"}
    assert model_variant(without).pose_frame == "joints"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("pose_frame", "rig", "pose_frame"),
        ("pose_frame", "flange", "pose_frame"),
        ("video.views", ["exterior"], "views"),
        ("action_space.dim", 10, "action_space.dim"),
    ],
)
def test_bad_so101_config_is_refused(path: str, value: object, match: str) -> None:
    config = _with(json.loads(SO101_CONFIG.read_text()), path, value)
    config = _with(config, "video.tile_bounds_rows_cols", None)
    with pytest.raises(CheckpointError, match=match):
        validate_config(config)


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


def test_video_antialias_is_read() -> None:
    """``video.antialias`` sets the resize of the tiles; absent means off."""
    assert model_variant(SINGLE_ARM).antialias is False
    assert model_variant(_with(SINGLE_ARM, "video.antialias", False)).antialias is False
    antialiased = _with(
        _with(_with(SINGLE_ARM, "video.height", 224), "video.tile_height", 224),
        "video.antialias",
        True,
    )
    assert model_variant(antialiased).antialias is True
    with pytest.raises(CheckpointError, match=r"video\.antialias"):
        model_variant(_with(SINGLE_ARM, "video.antialias", "yes"))


def test_antialias_refuses_a_tile_taller_than_the_fit() -> None:
    """The fit rule trims 234x416 to the tile, so a 256x416 tile cannot use it."""
    with pytest.raises(CheckpointError, match="no taller than 234"):
        model_variant(_with(SINGLE_ARM, "video.antialias", True))
