"""Unit tests for the settings of a sim run (CPU only, no weights).

The tests need the ``mujoco`` package, because the settings use the scene,
robot and task dataclasses.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

pytest.importorskip("praxis_sim")  # the `sim` extra

# praxis.sim sets the OpenGL backend. It must load before mujoco.
import praxis.sim  # noqa: F401

pytest.importorskip("mujoco")

from praxis.sim.config import (
    ModelConfig,
    SamplerConfig,
    SessionConfig,
    SimConfig,
    from_dict,
    load_config,
    read_yaml,
    to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
DROID = ROOT / "configs" / "sim_droid.yaml"
SCRIPTED = ROOT / "configs" / "sim_scripted.yaml"


def test_droid_config_has_the_serve_settings() -> None:
    """The sim config samples as the inference config of the checkpoint."""
    cfg = load_config(DROID)
    serve = yaml.safe_load((ROOT / "configs" / "inference_droid.yaml").read_text())
    assert dataclasses.asdict(cfg.sampler) == serve["sampler"]
    for key, value in serve["session"].items():
        assert getattr(cfg.session, key) == value, key
    for key, value in serve["model"].items():
        assert getattr(cfg.model, key) == value, key
    assert cfg.policy.type == "model"
    assert cfg.task.name == "lift_cube" and cfg.task.prompt
    assert cfg.task.cube_x == (0.45, 0.62)
    assert cfg.scene.exterior.pos == (0.15, 0.68, 0.48)
    assert (cfg.scene.image_height, cfg.scene.image_width) == (288, 512)
    assert cfg.viewer.host == "127.0.0.1"
    assert cfg.needs_images


def test_code_defaults_are_the_droid_config() -> None:
    """A block that is not in the YAML file gets the DROID settings."""
    cfg = yaml.safe_load(DROID.read_text())
    assert dataclasses.asdict(SamplerConfig()) == cfg["sampler"]
    assert dataclasses.asdict(SessionConfig()) == cfg["session"]
    model = dataclasses.asdict(ModelConfig())
    # Inference precision is opt-in; recommended YAML settings stay unchanged.
    assert model.pop("routed_precision", "bf16") == "bf16"
    assert model.pop("nvfp4_cache", None) is None
    assert model == cfg["model"]
    assert (SamplerConfig().layout, SessionConfig().max_context_chunks) == (
        "window_joint",
        1,
    )


def test_scripted_config_needs_no_weights() -> None:
    cfg = load_config(SCRIPTED)
    assert cfg.policy.type == "scripted"
    assert cfg.task.prompt is None and cfg.task.time_limit_s is None
    env_cfg = cfg.env_config()
    assert env_cfg.render and env_cfg.task is cfg.task


def test_overrides_and_number_strings() -> None:
    cfg = load_config(
        DROID,
        [
            "task.prompt=Lift the cube.",
            "episodes.seeds=[4, 5]",
            "sampler.num_steps=4",
            "robot.ik.tol_pos=1e-3",  # YAML reads 1e-3 as a string
            "sampler.cfg_scale=2",  # an int for a float field
            "viewer.keep_open_s=null",
            "scene.exterior.fovy=65",
            "output.dir=/tmp/x",
        ],
    )
    assert cfg.task.prompt == "Lift the cube."
    assert cfg.episodes.seeds == [4, 5]
    assert cfg.sampler.num_steps == 4
    assert cfg.robot.ik.tol_pos == pytest.approx(1e-3)
    assert isinstance(cfg.sampler.cfg_scale, float) and cfg.sampler.cfg_scale == 2.0
    assert cfg.viewer.keep_open_s is None
    assert cfg.scene.exterior.fovy == 65.0
    assert cfg.output.dir == "/tmp/x"
    raw = read_yaml(DROID, ["a.b.c=1"])
    assert raw["a"] == {"b": {"c": 1}}


def test_bad_values_raise() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        load_config(DROID, ["sampler.num_step=4"])
    with pytest.raises(ValueError, match="integer"):
        load_config(DROID, ["sampler.num_steps=4.5"])
    with pytest.raises(ValueError, match="true or false"):
        load_config(DROID, ["sampler.cfg_batched=1"])
    with pytest.raises(ValueError, match="3 values"):
        load_config(DROID, ["scene.exterior.pos=[1, 2]"])
    with pytest.raises(ValueError, match=r"policy\.type"):
        load_config(DROID, ["policy.type=bogus"])
    with pytest.raises(ValueError, match="key=value"):
        load_config(DROID, ["sampler.num_steps"])


def test_round_trip(tmp_path: Path) -> None:
    cfg = load_config(DROID, ["model.vae_norm_block_c={160: 64, 640: 128}"])
    assert cfg.model.vae_norm_block_c == {160: 64, 640: 128}
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    again = load_config(path)
    assert again == cfg
    assert from_dict(SimConfig, {}) == SimConfig()
