"""The configs of the single-arm DROID checkpoint hold its recipe (CPU only).

The single-arm checkpoint is the only checkpoint of praxis. The recipe is
its recommended sampling settings: 2 Euler steps, the flow
shift 12.5, text guidance 4.0 on the video and none on the actions, the
``window_joint`` layout and 1 context chunk. The README ("Recommended settings") gives the same values. When the
recipe changes, change the configs, this test and the README together.
``tests/serve/test_serve_single_arm.py`` checks the serve config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from praxis.model import SamplerSettings

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = "inference_droid.yaml"
SIM = "sim_droid.yaml"

# The recipe: config block -> key -> value.
RECIPE: dict[str, dict[str, object]] = {
    "model": {
        "checkpoint": "checkpoints/praxis-1-droid",
        "vae_weights": "weights/Wan2.2_VAE.pth",
        "text_encoder": "weights/Qwen3.5-2B",
    },
    "sampler": {
        "num_steps": 2,
        "t_shift": 12.5,
        "cfg_scale": 1.0,
        "cfg_video_scale": 4.0,
        "cfg_batched": False,
        "layout": "window_joint",
        "solver": "euler",
        "action_steps": None,
    },
    "session": {
        "exec_horizon": 24,
        "control_mode": "droid_impedance",
        "max_context_chunks": 1,
        "seed": 0,
    },
}


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


@pytest.mark.parametrize("name", [INFERENCE, SIM])
def test_config_has_the_recipe(name: str) -> None:
    cfg = _load(name)
    for block, values in RECIPE.items():
        got = {key: cfg[block].get(key) for key in values}
        assert got == values, f"{name}: {block}"
    assert set(cfg["sampler"]) == set(RECIPE["sampler"]), name


@pytest.mark.parametrize("name", [INFERENCE, SIM])
def test_sampler_block_makes_the_sampler_settings(name: str) -> None:
    settings = SamplerSettings(**_load(name)["sampler"])
    assert settings.layout == "window_joint" and settings.guided


def test_sim_config_renders_the_rig_views() -> None:
    """The sim renders each view at the rig size. The pipeline makes the tiles."""
    scene = _load(SIM)["scene"]
    assert (scene["image_height"], scene["image_width"]) == (288, 512)


@pytest.mark.parametrize("name", [INFERENCE, SIM])
def test_paths_are_relative(name: str) -> None:
    cfg = _load(name)
    paths = [cfg["model"][k] for k in ("checkpoint", "vae_weights", "text_encoder")]
    paths.append(cfg["output"]["dir"])
    if "input" in cfg:
        paths.append(cfg["input"]["episodes_dir"])
    for path in paths:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts, path
