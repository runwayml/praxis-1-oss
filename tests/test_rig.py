"""Tests of the base checkpoint and of scripts/init_heads.py (CPU, no weights)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tiny_model import tiny_config, tiny_policy

from praxis.checkpoint import (
    BASE_HEAD_TENSORS,
    BASE_REFUSED,
    CheckpointError,
    load_base_config,
    load_config,
    save_checkpoint,
    validate_base_config,
)
from praxis.model import PraxisPolicy
from praxis.rig import (
    HEAD_INIT_STD,
    init_head_tensors,
    init_heads,
    normalization_from_stats,
    single_arm_config,
)
from praxis.train.norm_stats import norm_stats

DROID = Path(__file__).parent / "data" / "praxis-1-droid.json"
BASE = Path(__file__).parent / "data" / "praxis-1-base.json"


def strip_rig(config: dict) -> dict:
    """The base config of a single-arm config: the rig fields removed."""
    base = copy.deepcopy(config)
    base["embodiment"] = "base"
    for key in ("action_space", "normalization", "pose_frame"):
        base.pop(key, None)
    for key in ("action_patch_dim", "proprio_patch_dim"):
        base["action_expert"].pop(key, None)
    for key in (
        "height",
        "width",
        "views",
        "tile_bounds_rows_cols",
        "tile_height",
        "tile_width",
        "antialias",
    ):
        base.get("video", {}).pop(key, None)
    return base


def test_base_config_of_droid_is_valid_and_refused_by_load_config(
    tmp_path: Path,
) -> None:
    base = strip_rig(json.loads(DROID.read_text()))
    validate_base_config(base)
    (tmp_path / "config.json").write_text(json.dumps(base))
    assert load_base_config(tmp_path) == base
    with pytest.raises(CheckpointError) as err:
        load_config(tmp_path)
    assert str(err.value) == BASE_REFUSED


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(action_space={"dim": 10}),
        lambda c: c["action_expert"].update(action_patch_dim=40),
        lambda c: c["video"].update(tile_height=224),
        lambda c: c.update(embodiment="single_arm"),
        lambda c: c.update(format_version=1),
    ],
)
def test_base_config_refuses_rig_fields(change: object) -> None:
    base = strip_rig(json.loads(DROID.read_text()))
    change(base)  # ty: ignore[call-non-callable]
    with pytest.raises(CheckpointError):
        validate_base_config(base)


def test_single_arm_config_rebuilds_the_droid_config() -> None:
    """Base + the rig of praxis-1-droid gives the config of praxis-1-droid."""
    droid = json.loads(DROID.read_text())
    norm = normalization_from_stats(droid["normalization"], domain="droid")
    video = droid["video"]
    rebuilt = single_arm_config(
        strip_rig(droid),
        tile_height=video["tile_height"],
        tile_width=video["tile_width"],
        antialias=video["antialias"],
        normalization=norm,
    )
    assert rebuilt == droid


def test_normalization_from_stats_checks_the_vectors() -> None:
    stats = {k: [0.0] * 10 for k in ("action_q01", "proprio_mean")}
    stats |= {k: [1.0] * 10 for k in ("action_q99", "proprio_std")}
    assert normalization_from_stats(stats, "x")["q99_normed_dims"] == [0, 1, 2, 9]
    with pytest.raises(ValueError):
        normalization_from_stats({**stats, "action_q01": [0.0] * 9}, "x")
    with pytest.raises(ValueError):
        normalization_from_stats({**stats, "proprio_std": [float("nan")] * 10}, "x")


def test_norm_stats_quantiles_and_guards() -> None:
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(10_000, 10))
    actions[:, 9] = 0.5  # a gripper that never moves
    proprio = rng.normal(2.0, 3.0, size=(4_000, 10))
    proprio[:, 3] = 1.0  # a constant proprio dim
    s = norm_stats(actions, proprio)
    assert s["action_q01"][0] == pytest.approx(-2.33, abs=0.1)
    assert s["action_q99"][0] == pytest.approx(2.33, abs=0.1)
    # the gripper is pinned to [0, 1], not a quantile
    assert (s["action_q01"][9], s["action_q99"][9]) == (0.0, 1.0)
    rare = actions.copy()
    rare[:, 9] = 0.0
    rare[:50, 9] = 1.0  # closed in 0.5 % of the rows
    assert norm_stats(rare, proprio)["action_q99"][9] == 1.0
    flat = actions.copy()
    flat[:, 2] = 0.3  # an xyz dim that never moves
    s2 = norm_stats(flat, proprio)
    assert s2["action_q99"][2] == s2["action_q01"][2] + 1.0
    # the constant value sits at the centre of the range: it normalizes to 0
    assert (s2["action_q01"][2], s2["action_q99"][2]) == pytest.approx((-0.2, 0.8))
    assert s["proprio_mean"][0] == pytest.approx(2.0, abs=0.2)
    assert s["proprio_std"][0] == pytest.approx(3.0, abs=0.2)
    assert s["proprio_std"][3] > 0


def _tiny_single_arm() -> dict:
    cfg = tiny_config()
    cfg["chunking"].update(action_frames_per_token=4, proprio_frames=1)
    return cfg


def _write_tiny_base(path: Path) -> dict[str, torch.Tensor]:
    """Save the tiny policy without its heads as a base checkpoint."""
    weights = tiny_policy(seed=3).state_dict()
    base = {k: v for k, v in weights.items() if k not in BASE_HEAD_TENSORS}
    save_checkpoint(path, base, strip_rig(_tiny_single_arm()))
    return base


def _norm() -> dict:
    stats = {k: [0.0] * 10 for k in ("action_q01", "proprio_mean")}
    stats |= {k: [1.0] * 10 for k in ("action_q99", "proprio_std")}
    return normalization_from_stats(stats, "tiny")


def test_init_heads_writes_a_checkpoint_that_loads(tmp_path: Path) -> None:
    base = _write_tiny_base(tmp_path / "base")
    cfg = init_heads(
        tmp_path / "base",
        tmp_path / "rig",
        tile_height=224,
        tile_width=416,
        antialias=True,
        normalization=_norm(),
        seed=7,
    )
    assert cfg["embodiment"] == "single_arm" and "heads" not in cfg
    assert load_config(tmp_path / "rig") == cfg
    policy = PraxisPolicy.from_pretrained(tmp_path / "rig", device="cpu")
    state = policy.state_dict()
    for name, t in base.items():
        assert torch.equal(state[name], t), name
    head = state["action_expert.head.weight"]
    assert head.shape == (40, 64) and head.dtype == torch.bfloat16
    assert float(head.float().std()) == pytest.approx(HEAD_INIT_STD, rel=0.2)
    assert not state["action_expert.head.bias"].any()
    assert state["action_expert.embed_proprio.weight"].shape == (64, 10)
    assert state["action_expert.action_encoder.w1.weight"].shape == (64, 40)
    # The same seed gives the same heads.
    again = init_head_tensors(cfg, seed=7)
    for name in BASE_HEAD_TENSORS:
        assert torch.equal(state[name], again[name]), name


def test_init_head_tensors_keeps_the_global_rng() -> None:
    """The head init draws from its own seed and leaves the global generators
    (CPU, and CUDA when present) where they were."""
    cfg = single_arm_config(
        strip_rig(_tiny_single_arm()), tile_height=224, tile_width=416, antialias=True
    )
    cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    want = (torch.randn(4), torch.randn(4, device="cuda") if cuda else None)
    torch.manual_seed(1234)
    init_head_tensors(cfg, seed=0)
    got = (torch.randn(4), torch.randn(4, device="cuda") if cuda else None)
    assert torch.equal(want[0], got[0])
    if cuda:
        assert torch.equal(want[1], got[1])


def test_init_heads_takes_given_heads_and_refuses_bad_input(tmp_path: Path) -> None:
    _write_tiny_base(tmp_path / "base")
    args = dict(tile_height=224, tile_width=416, antialias=True, normalization=_norm())
    given = {
        k: torch.full_like(v, 0.5)
        for k, v in init_head_tensors(
            single_arm_config(strip_rig(_tiny_single_arm()), **args), 0
        ).items()
    }
    init_heads(tmp_path / "base", tmp_path / "rig", heads=given, **args)
    state = PraxisPolicy.from_pretrained(tmp_path / "rig", device="cpu").state_dict()
    assert all(bool((state[k] == 0.5).all()) for k in BASE_HEAD_TENSORS)
    with pytest.raises(CheckpointError):  # not all six tensors
        init_heads(
            tmp_path / "base",
            tmp_path / "rig2",
            heads={k: given[k] for k in BASE_HEAD_TENSORS[:2]},
            **args,
        )
    with pytest.raises(CheckpointError):  # the rig checkpoint is not a base
        init_heads(tmp_path / "rig", tmp_path / "rig3", **args)
    with pytest.raises(CheckpointError):  # the output exists
        init_heads(tmp_path / "base", tmp_path / "rig", **args)


def test_the_base_checkpoint_config() -> None:
    """The config of praxis-1-base is a valid base config. With the rig of
    praxis-1-droid it gives the model fields of praxis-1-droid."""
    base = json.loads(BASE.read_text())
    validate_base_config(base)
    droid = json.loads(DROID.read_text())
    video = droid["video"]
    rig = single_arm_config(
        base,
        tile_height=video["tile_height"],
        tile_width=video["tile_width"],
        antialias=video["antialias"],
        normalization=normalization_from_stats(droid["normalization"], "droid"),
    )
    rig.pop("source")
    assert rig == droid


@pytest.mark.parametrize(("th", "tw"), [(250, 416), (256, 400), (0, 416)])
def test_tile_sides_are_multiples_of_the_patch(th: int, tw: int) -> None:
    with pytest.raises(ValueError, match="multiples of 32"):
        single_arm_config(
            strip_rig(_tiny_single_arm()),
            tile_height=th,
            tile_width=tw,
            antialias=False,
        )
