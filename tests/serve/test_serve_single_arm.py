"""The server with the single-arm DROID checkpoint (CPU only, no weights).

The server serves the single-arm checkpoint. A stand-in replaces the
pipeline. The tests check the serve config of the checkpoint, the image size
of the rig, the sampler of a session and the keys of the handshake meta.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from praxis.model import SamplerSettings
from praxis.serve.policy import PolicyFactory

ROOT = Path(__file__).resolve().parents[2]

# A single-arm config of the checkpoint (the fields the server reads).
SINGLE_ARM: dict = {
    "format_version": 2,
    "embodiment": "single_arm",
    "chunking": {"n_chunks": 4, "grid_hz": 15.0},
    "video": {
        "height": 256,
        "width": 832,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "keep_aspect": True,
        "tile_height": 256,
        "tile_width": 416,
        "aspect_tol": 0.03,
    },
    "text_encoder": {"uncond": "blank_caption", "blank_caption": " "},
    "action_expert": {
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
    },
    "backbone": {
        "layout": "chunk_causal",
        "rope": {
            "chunked_video_layout": True,
            "n_tiles": None,
            "n_chunks": None,
            "ref_placement": "group",
        },
    },
    "pose_frame": "flange",
}
VARIANT_KEYS = [
    "sampler_layout",
    "backbone_layout",
    "canvas",
    "tile_size",
    "pose_frame",
    "text_uncond",
]


class _State:
    prompt = ""


class _Session:
    def __init__(self) -> None:
        self.state = _State()


class _Policy:
    def __init__(self, config: dict) -> None:
        self.config = config


class _Pipeline:
    """The attributes of :class:`praxis.pipeline.PraxisPipeline` that the
    factory reads. A session is a stub. ``sessions`` keeps the arguments of
    each session."""

    def __init__(self, config: dict, sampler: SamplerSettings) -> None:
        self.policy = _Policy(config)
        self.sampler = sampler
        self.embodiment = str(config.get("embodiment"))
        self.sessions: list[dict[str, object]] = []

    def session(self, **kwargs: object) -> _Session:
        self.sessions.append(kwargs)
        return _Session()


def _serve_config(name: str = "serve_droid.yaml") -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def _factory(
    config: dict,
    sampler: SamplerSettings | None = None,
    session: dict | None = None,
) -> tuple[PolicyFactory, _Pipeline]:
    cfg = _serve_config()
    if sampler is None:
        sampler = SamplerSettings(**cfg["sampler"])
    pipe = _Pipeline(copy.deepcopy(config), sampler)
    factory = PolicyFactory(
        pipe,  # ty: ignore[invalid-argument-type]
        dict(cfg["session"]) if session is None else session,
        "/ckpt/praxis-1-droid",
    )
    return factory, pipe


def test_serve_config_has_the_recipe() -> None:
    cfg = _serve_config()
    assert cfg["model"]["embodiment"] == "single_arm"
    assert cfg["model"]["checkpoint"] == "checkpoints/praxis-1-droid"
    assert cfg["sampler"] == {
        "num_steps": 2,
        "cfg_scale": 1.0,
        "cfg_video_scale": 4.0,
        "cfg_batched": False,
        "layout": "window_joint",
        "t_shift": 12.5,
        "solver": "euler",
        "action_steps": None,
    }
    assert cfg["session"] == {
        "exec_horizon": 24,
        "control_mode": "droid_impedance",
        "context_mode": "observed",
        "max_context_chunks": 1,
        "carry_mode": "clean",
        "prompt": "",
        "schedule": "chunked",
    }
    for key in ("checkpoint", "vae_weights", "text_encoder"):
        assert not Path(cfg["model"][key]).is_absolute(), key


@pytest.mark.parametrize("offline", ["inference_droid.yaml", "sim_droid.yaml"])
def test_serve_and_offline_configs_sample_the_same(offline: str) -> None:
    serve = _serve_config()
    other = yaml.safe_load((ROOT / "configs" / offline).read_text())
    assert serve["sampler"] == other["sampler"]
    for key in ("exec_horizon", "control_mode", "max_context_chunks"):
        assert serve["session"][key] == other["session"][key], key
    for key in ("checkpoint", "vae_weights", "text_encoder"):
        assert serve["model"][key] == other["model"][key], key


def test_tile_canvas_keeps_the_rig_image_size() -> None:
    """Without antialias the rig sends the views at 1024x288. The server makes the tiles."""
    factory = _factory(SINGLE_ARM)[0]
    assert factory.restrict == (1024, 288) and factory.restrict_first is False
    assert factory.stack_keys == (
        "image.wrist",
        "image.exterior",
        "robot_state.ee_pose",
        "grip",
    )
    assert factory.fps == 15.0 and factory.n_chunks == 4


def test_meta_adds_the_variant() -> None:
    meta = _factory(SINGLE_ARM)[0]({}).meta
    keys = list(meta)
    at = keys.index("tile_embedding") + 1
    assert keys[at : at + len(VARIANT_KEYS)] == VARIANT_KEYS
    assert {k: meta[k] for k in VARIANT_KEYS} == {
        "sampler_layout": "window_joint",
        "backbone_layout": "chunk_causal",
        "canvas": "tiles",
        "tile_size": [256, 416],
        "pose_frame": "flange",
        "text_uncond": "blank_caption",
    }
    assert meta["max_context_chunks"] == 1 and meta["num_evals"] == 2
    assert meta["t_shift"] == 12.5 and meta["cfg_batched"] is False
    # The serve config's default is the chunked schedule (a chunked session states no
    # schedule key); real-time chunking is opt-in.
    assert "schedule" not in meta
    assert _factory(SINGLE_ARM)[0]({"schedule": "rtc"}).meta["schedule"] == "rtc"
    local_stack = meta["local_stack"]
    assert isinstance(local_stack, dict)
    # The chunked recipe of main: its entries carry no version.
    assert local_stack["seq"][-1] == {
        "name": "restrict_image_size",
        "args": {"width": 1024, "height": 288},
    }


def test_meta_names_the_single_arm_rig() -> None:
    meta = _factory(SINGLE_ARM)[0]({"sample_seed": 0}).meta
    assert {k: meta[k] for k in ("arm", "checkpoint")} == {
        "arm": "praxis-1-droid",
        "checkpoint": "/ckpt/praxis-1-droid",
    }
    assert {
        k: meta[k]
        for k in (
            "expert_rope",
            "expert_rope_self",
            "expert_absolute_positions",
            "rope_layout",
            "rope_n_tiles",
            "tile_embedding",
        )
    } == {
        "expert_rope": False,
        "expert_rope_self": False,
        "expert_absolute_positions": True,
        "rope_layout": "contiguous",
        "rope_n_tiles": None,
        "tile_embedding": False,
    }
    assert meta["embodiment"] == "franka_ee"
    assert meta["control_mode"] == "droid_impedance"
    assert (
        meta["action_layout"] == "ee-10d [xyz3 rot6d6 grip1], rel-to-anchor, grip abs"
    )
    assert meta["wire_commands"] == ["robot_command", "target_grip"]
    assert meta["exec_horizon"] == 24 and meta["chunk_rows"] == 24
    assert meta["prompt_style"] == "caption" and meta["prompt_rewritten"] is None
    assert meta["wan_pattern"] == "contiguous-window-144444"
    assert meta["pad_start"] is False and meta["prompt_overridable"] is True
    assert meta["sample_seed_base"] == 0 and meta["seed_source"] == "session_param"


@pytest.mark.parametrize("embodiment", ["two_arm", "other"])
def test_factory_refuses_a_checkpoint_of_another_rig(embodiment: str) -> None:
    with pytest.raises(ValueError, match="single-arm and the bimanual checkpoints"):
        _factory({**SINGLE_ARM, "embodiment": embodiment})


def test_window_joint_session_drops_action_steps() -> None:
    factory, pipe = _factory(SINGLE_ARM)
    factory({"action_steps": 4, "num_steps": 2})
    sampler = pipe.sessions[-1]["sampler"]
    assert isinstance(sampler, SamplerSettings)
    assert sampler.layout == "window_joint" and sampler.action_steps is None
    assert pipe.sessions[-1]["max_context_chunks"] == 1


def test_session_gets_the_sampler_and_settings_of_the_session() -> None:
    """The session parameters replace the boot values for one session only."""
    factory, pipe = _factory(SINGLE_ARM)
    boot = pipe.sampler
    factory(
        {
            "sample_seed": 7,
            "num_steps": 3,
            "t_shift": 1.0,
            "cfg_scale": 3.0,
            "cfg_video_scale": 2.0,
            "cfg_batched": "on",
            "max_context_chunks": 1,
            "control_mode": "none",
            "prompt": "pick the cube",
        }
    )
    kwargs = pipe.sessions[-1]
    assert list(kwargs) == [
        "prompt",
        "seed",
        "max_context_chunks",
        "control_mode",
        "exec_horizon",
        "sampler",
    ]
    assert kwargs["prompt"] == "pick the cube" and kwargs["seed"] == 7
    assert kwargs["max_context_chunks"] == 1 and kwargs["control_mode"] == "none"
    assert kwargs["exec_horizon"] is None  # the server cuts the rows itself
    assert kwargs["sampler"] == SamplerSettings(
        num_steps=3,
        t_shift=1.0,
        solver="euler",
        action_steps=None,
        layout="window_joint",
        cfg_scale=3.0,
        cfg_video_scale=2.0,
        cfg_batched=True,
    )
    assert pipe.sampler is boot and boot.num_steps == 2  # the boot values stay
    factory({})
    assert pipe.sessions[-1]["sampler"] == boot
    assert pipe.sessions[-1]["control_mode"] == "droid_impedance"
    # max_context_chunks is clipped to n_chunks - 1.
    factory({"max_context_chunks": 9})
    assert pipe.sessions[-1]["max_context_chunks"] == 3


def test_antialiased_canvas_declares_the_fit_serve_stack() -> None:
    """An antialiased 224x416 canvas (praxis-1-droid) asks the client for 416x234 views.

    The stack is the handshake of the praxis-1-droid serve:
    ``restrict_image_size`` right after ``stop_on_fault``, then the frame stack
    with every past frame half a frame period after its tick, then the schedule.
    """
    config = copy.deepcopy(SINGLE_ARM)
    video = config["video"]
    video.update(antialias=True, height=224, tile_height=224)
    if "tile_bounds_rows_cols" in video:
        video["tile_bounds_rows_cols"] = {
            "exterior": [0, 224, 0, 416],
            "wrist": [0, 224, 416, 832],
        }
    factory = _factory(config)[0]
    assert factory.restrict == (416, 234) and factory.restrict_first is True
    seq = factory({"schedule": "chunked"}).meta["local_stack"]["seq"]
    assert [e["name"] for e in seq] == [
        "stop_on_fault",
        "restrict_image_size",
        "temporal_stack",
        "chunked_schedule",
    ]
    # The default is that chunked stack; schedule=rtc declares the periodic recipe,
    # the restrict in its place.
    assert factory({}).meta["local_stack"]["seq"] == seq
    rtc_seq = factory({"schedule": "rtc"}).meta["local_stack"]["seq"]
    assert [e["name"] for e in rtc_seq] == [
        "stop_on_fault",
        "restrict_image_size",
        "trajectory_player",
        "periodic",
        "inference_schedule",
    ]
    assert seq[1]["args"] == {"width": 416, "height": 234}
    offsets = seq[2]["args"]["offsets_sec"]
    assert len(offsets) == 25 and offsets[-1] == 0.0
    assert offsets[0] == -24 / 15 + 0.5 / 15 and offsets[-2] == -1 / 15 + 0.5 / 15


def test_factory_serves_the_bimanual_rig() -> None:
    """The bimanual checkpoint: the yam_ee2 wire, its stack keys and a
    416x416 restrict box ahead of the frame stack."""
    import json

    from praxis import bimanual

    yam = json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
    cfg = _serve_config("serve_yam.yaml")
    assert cfg["model"]["embodiment"] == "bimanual"
    pipe = _Pipeline(yam, SamplerSettings(**cfg["sampler"]))
    factory = PolicyFactory(
        pipe,  # ty: ignore[invalid-argument-type]
        dict(cfg["session"]),
        "/ckpt/praxis-1-yam",
    )
    assert factory.stack_keys == bimanual.STACK_KEYS
    assert factory.restrict == (416, 416) and factory.restrict_first is False
    assert factory.half_frame is True
    meta = factory({}).meta
    assert meta["embodiment"] == "yam_ee2" and meta["control_mode"] == "none"
    assert meta["wire_commands"] == [
        "robot_command.left",
        "robot_command.right",
        "target_grip.left",
        "target_grip.right",
    ]
    assert pipe.sessions[-1]["control_mode"] == "none"
    assert "schedule" not in meta  # the chunked default
    assert factory({"schedule": "rtc"}).meta["schedule"] == "rtc"
    seq = factory({"schedule": "chunked"}).meta["local_stack"]["seq"]
    # The YAM client order: the stack, then the restrict; half-frame offsets.
    assert [s["name"] for s in seq] == [
        "stop_on_fault",
        "temporal_stack",
        "chunked_schedule",
        "restrict_image_size",
    ]
    offsets = seq[1]["args"]["offsets_sec"]
    assert offsets[-1] == 0.0 and abs(offsets[0] + 23.5 / 15) < 1e-9
    with pytest.raises(ValueError, match="control_mode"):
        factory({"control_mode": "droid_impedance"})
