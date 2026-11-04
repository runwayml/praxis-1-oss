"""The serve config holds the recommended sampling settings (CPU only).

``praxis/serve/README.md`` gives the same values. The deployed server reads
``configs/serve_droid.yaml``, so a change of these values changes the served answers.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# The recommended settings: config block -> key -> value.
RECOMMENDED: dict[str, dict[str, object]] = {
    "model": {"embodiment": "single_arm"},
    "sampler": {
        "num_steps": 2,
        "cfg_scale": 1.0,
        "cfg_video_scale": 4.0,
        "t_shift": 12.5,
        "layout": "window_joint",
        "solver": "euler",
        "action_steps": None,
    },
    "session": {
        "context_mode": "observed",
        "max_context_chunks": 1,
        "carry_mode": "clean",
        "exec_horizon": 24,
        "control_mode": "droid_impedance",
    },
}


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_serve_config_has_the_recommended_settings() -> None:
    name = "serve_droid.yaml"
    cfg = _load(name)
    for block, values in RECOMMENDED.items():
        got = {key: cfg[block].get(key) for key in values}
        assert got == values, f"{name}: {block}"


def test_serve_and_offline_configs_sample_the_same() -> None:
    """``run_inference.py`` replays a session with the settings of the server."""
    a, b = _load("serve_droid.yaml"), _load("inference_droid.yaml")
    defaults = {"cfg_scale": 1.0, "cfg_video_scale": 4.0, "layout": "window_joint"}
    for key in ("num_steps", "t_shift", "solver", "action_steps", *defaults):
        default = defaults.get(key)
        assert a["sampler"].get(key, default) == b["sampler"].get(key, default), key
    assert a["session"]["max_context_chunks"] == b["session"]["max_context_chunks"]
    assert a["session"].get("exec_horizon") == b["session"].get("exec_horizon")
