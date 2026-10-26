"""Protocol tests of the policy server (CPU only, no weights).

The tests run the real server app under uvicorn on a free local port. A small
stand-in replaces the networks of the pipeline. The session logic, the codec
and the command format are the real code. A test client speaks the protocol
over a real WebSocket connection. The server serves the single-arm
checkpoint.
"""

from __future__ import annotations

import io
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# The server tests need the [serve] extra.
pytest.importorskip("fastapi")
pytest.importorskip("msgpack")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import msgpack
import uvicorn
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline, Session
from praxis.robot_io import NormStats
from praxis.serve import params, wire
from praxis.serve import server as server_module
from praxis.serve.policy import PolicyFactory, served_code_version
from praxis.serve.server import InferenceServer

ROOT = Path(__file__).resolve().parents[2]

# The single-arm config: the variant of the checkpoint, small tiles.
SINGLE_ARM = {
    "format_version": 2,
    "embodiment": "single_arm",
    "chunking": {"n_chunks": 4, "grid_hz": 15.0},
    "video": {
        "height": 32,
        "width": 96,
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "keep_aspect": True,
        "tile_height": 32,
        "tile_width": 48,
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


class FakePolicy:
    """Stand-in for the networks: the actions depend on the inputs and the noise.

    Almost all of the actions are in [-1, 1], the range of the training
    targets, so the clamp of the decode does not hide the smoothing. The
    checkpoint samples with ``sample_window_joint``.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.calls: list[dict] = []

    def sample_window_joint(
        self, text, text_uncond, anchor, cleans, proprio_all, seed, settings, trace=None
    ):
        self.calls.append({"depth": len(cleans), "settings": settings})
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn((1, 6, 40), generator=generator)
        proprio = float(proprio_all[..., -1, :].sum())
        actions = 0.1 * noise + 0.001 * float(anchor.mean()) + 0.001 * proprio
        return torch.zeros(1, 1, 6, 1, 1), actions


class FakeEncoder:
    norm_block_c: dict[int, int] | None = None  # read by the boot log

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 16, 16))


class FakePipeline:
    """The attributes of :class:`praxis.pipeline.PraxisPipeline` that a session reads."""

    def __init__(self, config: dict, sampler: SamplerSettings) -> None:
        self.device = torch.device("cpu")
        self.policy = FakePolicy(config)
        self.vae = FakeEncoder()
        self.sampler = sampler
        self.embodiment = str(config.get("embodiment"))
        dim = 10
        rng = np.random.default_rng(0)
        self.stats = NormStats(
            action_q01=np.full(dim, -0.2, np.float32),
            action_q99=np.full(dim, 0.3, np.float32),
            proprio_mean=rng.normal(size=dim).astype(np.float32),
            proprio_std=rng.uniform(0.5, 1.0, size=dim).astype(np.float32),
        )

    def encode_text(self, prompt: str) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def encode_uncond_text(self) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def session(self, **kwargs: object) -> Session:
        # The session of the real pipeline, with its defaults.
        return PraxisPipeline.session(self, **kwargs)  # ty: ignore[invalid-argument-type]


def _single_arm_factory(
    checkpoint: str = "/ckpt/droid", **session: object
) -> PolicyFactory:
    # Every field set: the test values do not follow the class defaults.
    sampler = SamplerSettings(
        num_steps=5,
        t_shift=1.0,
        solver="euler",
        action_steps=None,
        layout="window_joint",
        cfg_scale=3.0,
        cfg_video_scale=1.0,
        cfg_batched=True,
    )
    defaults = {
        "exec_horizon": 24,
        "control_mode": "droid_impedance",
        "max_context_chunks": 3,
        **session,
    }
    pipe = FakePipeline(SINGLE_ARM, sampler)
    return PolicyFactory(pipe, defaults, checkpoint)  # ty: ignore[invalid-argument-type]


def _pipe(factory: PolicyFactory) -> FakePipeline:
    """Return the stand-in pipeline of a factory."""
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    return pipe


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def server() -> Iterator[tuple[str, InferenceServer]]:
    """Serve two single-arm models on a local port.

    The second model is a fine-tuned copy with other boot defaults. It checks
    the model path of the session URL.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    finetuned = _single_arm_factory(
        "/ckpt/droid-ft", exec_horizon=None, control_mode="none"
    )
    app = InferenceServer(
        {"default": _single_arm_factory(), "droid/ft": finetuned}, executor
    )
    port = _free_port()
    config = uvicorn.Config(
        app.app, host="127.0.0.1", port=port, log_level="warning", ws_max_size=1 << 28
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    yield f"127.0.0.1:{port}", app
    srv.should_exit = True
    thread.join(timeout=10)
    executor.shutdown()


def _recv(ws: Connection) -> dict:
    return wire.deserialise(ws.recv(timeout=10))


def _open(addr: str, path: str = "/api/v1/session") -> tuple[Connection, dict]:
    ws = connect(f"ws://{addr}{path}", open_timeout=10)
    loading = _recv(ws)
    assert loading == {"status": "loading", "message": "Loading policy..."}
    return ws, _recv(ws)


def _closed(ws: Connection) -> tuple[int, str]:
    with pytest.raises(ConnectionClosed) as info:
        ws.recv(timeout=10)
    assert info.value.rcvd is not None
    return info.value.rcvd.code, info.value.rcvd.reason


def _refused(addr: str, path: str) -> tuple[list[dict], int, str]:
    with connect(f"ws://{addr}{path}", open_timeout=10) as ws:
        frames = []
        while True:
            try:
                frames.append(_recv(ws))
            except ConnectionClosed as exc:
                assert exc.rcvd is not None
                return frames, exc.rcvd.code, exc.rcvd.reason


def _droid_obs(t: int, seed: int = 0, grip_2d: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(t, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pose = np.concatenate([rng.normal(size=(t, 3)) * 0.1, q], axis=1).astype(np.float32)
    grip = rng.uniform(size=(t, 1)).astype(np.float32)
    return {
        "image.exterior": rng.integers(0, 255, (t, 64, 64, 3), dtype=np.uint8),
        "image.wrist": rng.integers(0, 255, (t, 64, 64, 3), dtype=np.uint8),
        "robot_state.ee_pose": pose,
        "grip": grip if grip_2d else grip[:, 0],
        "task": "Move the cloth",
    }


def _jpeg(frames: np.ndarray) -> dict:
    bufs = []
    for frame in frames.reshape(-1, *frames.shape[-3:]):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        bufs.append(buf.getvalue())
    return {b"__jpeg__": True, b"frames": bufs, b"ndim": frames.ndim}


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_numpy_round_trip() -> None:
    value = {
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
        "b": np.float32(1.5),
        "c": [np.int16(3), (1, 2)],
        "d": "text",
    }
    out = wire.deserialise(wire.serialise(value))
    assert out["a"].dtype == np.float32 and np.array_equal(out["a"], value["a"])
    assert not out["a"].flags.writeable
    assert out["b"] == 1.5 and isinstance(out["b"], np.float32)
    assert out["c"] == [np.int16(3), [1, 2]] and out["d"] == "text"


def test_numpy_markers_are_the_positronic_form() -> None:
    packed = wire.serialise({"x": np.array([1, 2], np.uint8)})
    raw = msgpack.unpackb(packed)
    assert raw == {
        "x": {
            b"__ndarray__": True,
            b"data": b"\x01\x02",
            b"dtype": "|u1",
            b"shape": [2],
        }
    }


@pytest.mark.parametrize("dtype", ["O", "c8", "V8"])
def test_unsafe_dtypes_are_refused(dtype: str) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        wire.serialise(np.zeros(2, dtype=dtype))
    marker = {b"__ndarray__": True, b"data": b"\0" * 16, b"dtype": dtype, b"shape": [2]}
    with pytest.raises(ValueError, match="Unsupported dtype"):
        wire.deserialise(msgpack.packb(marker))
    generic = {b"__npgeneric__": True, b"data": 0, b"dtype": dtype}
    with pytest.raises(ValueError, match="Unsupported dtype"):
        wire.deserialise(msgpack.packb(generic))


def test_bad_shape_is_refused() -> None:
    marker = {b"__ndarray__": True, b"data": b"\0" * 6, b"dtype": "<f4", b"shape": [2]}
    with pytest.raises(ValueError):
        wire.deserialise(msgpack.packb(marker))


def test_jpeg_stacks_and_quality() -> None:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (3, 16, 24, 3), dtype=np.uint8)
    obs, stats = wire.deserialise_request(
        msgpack.packb({"s": _jpeg(frames), "one": _jpeg(frames[0])})
    )
    assert obs["s"].shape == (3, 16, 24, 3) and obs["s"].dtype == np.uint8
    assert obs["one"].shape == (16, 24, 3)
    assert stats.codecs == {"jpeg"} and stats.jpeg_quality == 90
    _, raw = wire.deserialise_request(wire.serialise({"s": frames}))
    assert raw.codecs == set() and raw.jpeg_quality is None


def test_h264_stack() -> None:
    av = pytest.importorskip("av")
    t, h, w = 5, 21, 33
    frames = np.zeros((t, h, w, 3), np.uint8)
    frames[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)
    frames[..., 1] = (np.arange(t) * 40)[:, None, None]
    padded = np.pad(frames, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="edge")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="h264") as out:
        stream = out.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = w + 1, h + 1, "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency"}
        stream.codec_context.gop_size = t
        stream.codec_context.max_b_frames = 0
        for f in padded:
            for p in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)
    marker = {
        b"__h264__": True,
        b"data": buf.getvalue(),
        b"shape": [t, h, w, 3],
        b"ndim": 4,
    }
    obs, stats = wire.deserialise_request(msgpack.packb({"v": marker}))
    assert obs["v"].shape == (t, h, w, 3) and stats.codecs == {"h264"}
    assert np.abs(obs["v"].astype(int) - frames).mean() < 8
    marker[b"shape"] = [t + 1, h, w, 3]
    with pytest.raises(ValueError, match="expected 6"):
        wire.deserialise(msgpack.packb({"v": marker}))


def test_image_codecs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert wire.image_codecs() == ["h264", "jpeg"]
    monkeypatch.setenv("PRAXIS_IMAGE_CODECS", " JPEG ")
    assert wire.image_codecs() == ["jpeg"]
    monkeypatch.setenv("PRAXIS_IMAGE_CODECS", "png")
    with pytest.raises(ValueError):
        wire.image_codecs()
    assert wire.compress_images()
    monkeypatch.setenv("PRAXIS_COMPRESS_IMAGES", "0")
    assert not wire.compress_images()


# ---------------------------------------------------------------------------
# Session parameters
# ---------------------------------------------------------------------------


def test_param_literals() -> None:
    raw = {
        "num_steps": "5",
        "cfg_scale": "3.0",
        "cfg_batched": "true",
        "exec_horizon": "null",
        "prompt": "pick the cube",
        "stack_keys": "a, b,",
        "rollout_id": "ep00",
    }
    assert params.parse_session_params(raw) == {
        "num_steps": 5,
        "cfg_scale": 3.0,
        "cfg_batched": True,
        "exec_horizon": None,
        "prompt": "pick the cube",
        "stack_keys": ("a", "b"),
    }
    assert params.parse_session_params({"stack_keys": '["x", "y"]'}) == {
        "stack_keys": ("x", "y")
    }
    with pytest.raises(ValueError, match=r"Unknown session params \['bogus'\]"):
        params.parse_session_params({"bogus": "1"})
    assert len(params.SESSION_PARAMS) == 28


def test_rollout_ids_and_seeds() -> None:
    assert params.resolve_rollout_id("ep-00.a_b") == ("ep-00.a_b", "client_supplied")
    for bad in ("../x", ".", "a/b", "-a", "x" * 129):
        rid, source = params.resolve_rollout_id(bad)
        assert source == "bridge_generated_after_reject" and rid.startswith("praxis-")
    rid, source = params.resolve_rollout_id(None)
    assert source == "bridge_generated" and len(rid) == len("praxis-") + 12
    assert params.resolve_sample_seed(3.7, "9") == (3, "session_param")
    assert params.resolve_sample_seed("null", "9") == (9, "env")
    base, source = params.resolve_sample_seed(None, None)
    assert source == "rotated" and 0 <= base < 2**31


def test_local_stack_offsets() -> None:
    spec = params.local_stack(("a",), 24, 15.0, 1024, 288, True)
    names = [e["name"] for e in spec["seq"]]
    assert names == [
        "stop_on_fault",
        "temporal_stack",
        "chunked_schedule",
        "restrict_image_size",
    ]
    offsets = spec["seq"][1]["args"]["offsets_sec"]
    assert len(offsets) == 25 and offsets[0] == -1.6 and offsets[-1] == 0.0
    assert offsets[1] == -1.5333333333333334
    stateless = params.local_stack(("a",), 0, 15.0, 416, 256, False)
    assert [e["name"] for e in stateless["seq"]] == [
        "chunked_schedule",
        "restrict_image_size",
    ]


# ---------------------------------------------------------------------------
# Handshake and calls over a real WebSocket
# ---------------------------------------------------------------------------

META_KEYS = [
    "policy", "code_version", "positronic_wire_ref", "arm", "checkpoint",
    "expert_rope", "expert_rope_self", "expert_absolute_positions", "rope_layout",
    "rope_n_tiles", "tile_embedding", "sampler_layout", "backbone_layout", "canvas",
    "tile_size", "pose_frame", "text_uncond", "exec_horizon", "num_steps", "t_shift",
    "solver", "num_evals", "cfg_scale", "cfg_video_scale", "cfg_batched",
    "context_mode", "max_context_chunks", "carry_mode", "record_denoised",
    "compile", "skip_video_decoder", "action_steps", "embodiment", "control_mode",
    "action_layout", "wire_commands", "wan_pattern", "chunk_rows", "prompt",
    "prompt_overridable", "pad_start", "prompt_style", "prompt_rewritten",
    "sample_seed_base", "seed_source", "local_stack", "compress_images",
    "image_codecs", "rollout_id", "rollout_id_source", "session_overrides",
]  # fmt: skip


def test_handshake_meta(server: tuple[str, InferenceServer]) -> None:
    addr, _ = server
    ws, ready = _open(addr, "/api/v1/session?rollout_id=ep00&sample_seed=0")
    ws.close()
    assert list(ready) == ["status", "meta"] and ready["status"] == "ready"
    meta = ready["meta"]
    assert list(meta) == META_KEYS
    assert isinstance(meta.pop("code_version"), str)
    offsets = params.frame_offsets_sec(24, 1, 15.0)
    assert meta == {
        "policy": "praxis",
        "positronic_wire_ref": "d5f2a47",
        "arm": "droid",
        "checkpoint": "/ckpt/droid",
        "expert_rope": False,
        "expert_rope_self": False,
        "expert_absolute_positions": True,
        "rope_layout": "contiguous",
        "rope_n_tiles": None,
        "tile_embedding": False,
        "sampler_layout": "window_joint",
        "backbone_layout": "chunk_causal",
        "canvas": "tiles",
        "tile_size": [32, 48],
        "pose_frame": "flange",
        "text_uncond": "blank_caption",
        "exec_horizon": 24,
        "num_steps": 5,
        "t_shift": 1.0,
        "solver": "euler",
        "num_evals": 5,
        "cfg_scale": 3.0,
        "cfg_video_scale": 1.0,
        "cfg_batched": True,
        "context_mode": "observed",
        "max_context_chunks": 3,
        "carry_mode": "clean",
        "record_denoised": None,
        "compile": None,
        "skip_video_decoder": None,
        "action_steps": None,
        "embodiment": "franka_ee",
        "control_mode": "droid_impedance",
        "action_layout": "ee-10d [xyz3 rot6d6 grip1], rel-to-anchor, grip abs",
        "wire_commands": ["robot_command", "target_grip"],
        "wan_pattern": "contiguous-window-144444",
        "chunk_rows": 24,
        "prompt": "",
        "prompt_overridable": True,
        "pad_start": False,
        "prompt_style": "caption",
        "prompt_rewritten": None,
        "sample_seed_base": 0,
        "seed_source": "session_param",
        # The rig sends the views at 1024x288. The server makes the tiles.
        "local_stack": {
            "seq": [
                {"name": "stop_on_fault"},
                {
                    "name": "temporal_stack",
                    "args": {
                        "keys": [
                            "image.wrist",
                            "image.exterior",
                            "robot_state.ee_pose",
                            "grip",
                        ],
                        "offsets_sec": offsets,
                        "pad_start": False,
                    },
                },
                {"name": "chunked_schedule"},
                {
                    "name": "restrict_image_size",
                    "args": {"width": 1024, "height": 288},
                },
            ]
        },
        "compress_images": True,
        "image_codecs": ["h264", "jpeg"],
        "rollout_id": "ep00",
        "rollout_id_source": "client_supplied",
        "session_overrides": {"sample_seed": "0"},
    }


def test_paths_and_models(server: tuple[str, InferenceServer]) -> None:
    addr, _ = server
    for path in (
        "/api/v1/session/",
        "/api/v1/session/default",
        "/api/v1/session/droid/ft",
    ):
        ws, ready = _open(addr, path)
        ws.close()
        assert ready["status"] == "ready"
    frames, code, reason = _refused(addr, "/api/v1/session/nope")
    assert frames == [
        {
            "status": "error",
            "error": "Unknown model 'nope'; this endpoint serves ['default', "
            "'droid/ft'] (see GET /api/v1/models).",
        }
    ]
    assert code == 1008 and reason == frames[0]["error"][:100]
    with urllib.request.urlopen(f"http://{addr}/api/v1/models", timeout=10) as r:
        assert json.load(r) == {"models": ["default", "droid/ft"]}
    keepalive = urllib.request.Request(f"http://{addr}/api/v1/keepalive", method="POST")
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(keepalive, timeout=10)
    assert info.value.code == 404
    with urllib.request.urlopen(f"http://{addr}/openapi.json", timeout=10) as r:
        models = json.load(r)["paths"]["/api/v1/models"]["get"]
    assert models["operationId"] == "get_models_api_v1_models_get"
    assert models["summary"] == "Get Models"


def test_calls_follow_the_context(server: tuple[str, InferenceServer]) -> None:
    """Rows 21, 24, 24; a short stack and reset_context restart the context."""
    addr, app = server
    pipe = _pipe(app.registry["default"])
    ws, _ = _open(addr, "/api/v1/session?sample_seed=7")
    rows = []
    for k, t in enumerate((1, 22, 25, 3, 25)):
        ws.send(wire.serialise(_droid_obs(t, seed=k)))
        out = _recv(ws)
        assert list(out) == [
            "result",
            "effective_prompt",
            "prompt_original",
            "server_timing",
        ]
        rows.append(len(out["result"]))
    ws.close()
    assert rows == [21, 24, 24, 21, 24]
    assert [c["depth"] for c in pipe.policy.calls] == [0, 1, 2, 0, 1]
    assert (
        out["effective_prompt"] == "Move the cloth" and out["prompt_original"] is None
    )
    row = out["result"][0]
    assert list(row) == ["robot_command", "target_grip", "timestamp"]
    assert list(row["robot_command"]) == ["type", "pose", "mode"]
    assert row["robot_command"]["pose"].dtype == np.float32
    assert row["robot_command"]["pose"].shape == (12,)
    assert isinstance(row["target_grip"], float) and row["timestamp"] == 0.0
    timing = out["server_timing"]
    assert list(timing) == [
        "request_bytes", "deserialize_ms", "jpeg_decode_ms", "video_decode_ms",
        "image_codec", "wire_jpeg_quality_est", "recv_iso", "policy_ms", "cfg_scale",
        "cfg_video_scale", "cfg_batched", "t_shift", "solver", "num_evals", "send_iso",
    ]  # fmt: skip
    assert timing["image_codec"] == "raw" and timing["num_evals"] == 5

    pipe.policy.calls.clear()
    ws, _ = _open(addr, "/api/v1/session?sample_seed=7")
    ws.send(wire.serialise(_droid_obs(1)))
    _recv(ws)
    obs = _droid_obs(22, seed=1)
    obs["reset_context"] = True
    ws.send(wire.serialise(obs))
    assert len(_recv(ws)["result"]) == 21
    ws.close()
    assert [c["depth"] for c in pipe.policy.calls] == [0, 0]


def test_same_seed_same_answer_and_settings(
    server: tuple[str, InferenceServer],
) -> None:
    addr, app = server
    pipe = _pipe(app.registry["default"])
    answers = []
    for query in (
        "sample_seed=3",
        "sample_seed=3",
        "sample_seed=4&num_steps=2&cfg_scale=1.0",
    ):
        ws, _ = _open(addr, f"/api/v1/session?{query}")
        ws.send(wire.serialise(_droid_obs(1)))
        answers.append(_recv(ws)["result"])
        ws.close()
    poses = [np.stack([r["robot_command"]["pose"] for r in a]) for a in answers]
    assert np.array_equal(poses[0], poses[1]) and not np.array_equal(poses[0], poses[2])
    settings = [c["settings"] for c in pipe.policy.calls]
    assert settings[0].num_steps == 5 and settings[0].cfg_scale == 3.0
    assert settings[2].num_steps == 2 and settings[2].cfg_scale == 1.0
    assert pipe.sampler.num_steps == 5  # the boot settings do not change


def test_grip_stack_shapes_read_the_newest_value(
    server: tuple[str, InferenceServer],
) -> None:
    """A (T,) grip stack and a (T, 1) grip stack give the same answer."""
    addr, _ = server
    answers = []
    for grip_2d in (True, False):
        ws, _ = _open(addr, "/api/v1/session?sample_seed=1")
        ws.send(wire.serialise(_droid_obs(1)))
        _recv(ws)
        ws.send(wire.serialise(_droid_obs(22, seed=5, grip_2d=grip_2d)))
        answers.append(_recv(ws)["result"])
        ws.close()
    assert wire.serialise(answers[0]) == wire.serialise(answers[1])


def test_jpeg_session_and_horizon(server: tuple[str, InferenceServer]) -> None:
    addr, _ = server
    ws, _ = _open(addr, "/api/v1/session?exec_horizon=12&return_server_timing=false")
    obs = _droid_obs(1)
    obs["image.exterior"] = _jpeg(obs["image.exterior"])
    ws.send(msgpack.packb(obs, default=wire.pack_numpy))
    out = _recv(ws)
    ws.close()
    assert len(out["result"]) == 12 and "server_timing" not in out


def test_second_model_session(server: tuple[str, InferenceServer]) -> None:
    """A model path gives the session the boot defaults of that model."""
    addr, app = server
    pipe = _pipe(app.registry["droid/ft"])
    ws, ready = _open(addr, "/api/v1/session/droid/ft?sample_seed=0")
    meta = ready["meta"]
    assert meta["checkpoint"] == "/ckpt/droid-ft" and meta["arm"] == "droid-ft"
    assert meta["exec_horizon"] is None and meta["control_mode"] == "none"
    assert meta["embodiment"] == "franka_ee"
    rows = []
    for k, t in enumerate((1, 22, 25)):
        ws.send(wire.serialise(_droid_obs(t, seed=k)))
        out = _recv(ws)
        rows.append(len(out["result"]))
    ws.close()
    assert rows == [21, 24, 24]
    assert [c["depth"] for c in pipe.policy.calls] == [0, 1, 2]
    # control_mode none: the commands carry no mode.
    assert list(out["result"][0]["robot_command"]) == ["type", "pose"]


@pytest.mark.parametrize("prompt_style", ["caption", "verbatim"])
def test_prompt_goes_to_the_model_verbatim(
    server: tuple[str, InferenceServer], prompt_style: str
) -> None:
    """The task of an observation replaces the prompt. No style rewrites it."""
    addr, _ = server
    query = f"sample_seed=0&prompt=put%20the%20ball%20in%20the%20bag&prompt_style={prompt_style}"
    ws, ready = _open(addr, f"/api/v1/session?{query}")
    meta = ready["meta"]
    assert meta["prompt"] == "put the ball in the bag"
    assert meta["prompt_style"] == prompt_style and meta["prompt_rewritten"] is None
    obs = _droid_obs(1)
    del obs["task"]
    ws.send(wire.serialise(obs))
    out = _recv(ws)
    assert out["effective_prompt"] == "put the ball in the bag"
    assert out["prompt_original"] is None
    obs = _droid_obs(22, seed=1)
    obs["task"] = b"  pick the cube \n"
    ws.send(wire.serialise(obs))
    out = _recv(ws)
    ws.close()
    assert out["effective_prompt"] == "pick the cube"
    assert out["prompt_original"] is None


def test_stateless_session_drops_the_temporal_stack(
    server: tuple[str, InferenceServer],
) -> None:
    addr, app = server
    pipe = _pipe(app.registry["default"])
    pipe.policy.calls.clear()
    ws, ready = _open(addr, "/api/v1/session?context_mode=stateless")
    assert ready["meta"]["context_mode"] == "stateless"
    names = [e["name"] for e in ready["meta"]["local_stack"]["seq"]]
    assert names == ["stop_on_fault", "chunked_schedule", "restrict_image_size"]
    for k, t in enumerate((1, 22)):
        ws.send(wire.serialise(_droid_obs(t, seed=k)))
        assert len(_recv(ws)["result"]) == 21
    ws.close()
    assert [c["depth"] for c in pipe.policy.calls] == [0, 0]


# ---------------------------------------------------------------------------
# Errors and close codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "loading", "text"),
    [
        ("bogus=1", False, "Unknown session params ['bogus']"),
        ("pad_start=true", True, "pad_start=true is not served"),
        ("context_mode=rolling", True, "context_mode 'rolling' is not served"),
        ("context_mode=other", True, "context_mode 'other' not in"),
        ("history_frames=12", True, "history_frames=12 is not served"),
        ("cfg_batched=maybe", True, "expected on|off, got 'maybe'"),
        ("solver=rk4", True, "solver must be one of"),
        ("control_mode=bogus", True, "Unknown control_mode 'bogus'"),
        ("num_steps=%5B1%5D", True, "int() argument"),
        ("sample_seed=abc", True, "invalid literal for int()"),
        ("return_server_timing=maybe", True, "expected on|off, got 'maybe'"),
    ],
)
def test_bad_params_are_refused(
    server: tuple[str, InferenceServer], query: str, loading: bool, text: str
) -> None:
    addr, _ = server
    frames, code, reason = _refused(addr, f"/api/v1/session?{query}")
    if loading:
        assert frames[0] == {"status": "loading", "message": "Loading policy..."}
        frames = frames[1:]
    assert len(frames) == 1 and frames[0]["status"] == "error"
    assert text in frames[0]["error"]
    assert code == 1008 and reason == frames[0]["error"][:100]


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("sample_seed=1180591620717411303424", "Integer value out of range"),
        ("exec_horizon=-1180591620717411303424", "Integer value out of range"),
        ("prompt=%22%5Cud800%22", "surrogates not allowed"),
    ],
)
def test_unpackable_values_are_refused(
    server: tuple[str, InferenceServer], query: str, text: str
) -> None:
    """A value that msgpack cannot pack gives an error frame, not a hang."""
    addr, _ = server
    frames, code, _ = _refused(addr, f"/api/v1/session?{query}")
    assert frames[0]["status"] == "loading" and len(frames) == 2
    assert frames[1]["status"] == "error" and text in frames[1]["error"]
    assert code == 1008


def test_matching_derived_params_are_accepted(
    server: tuple[str, InferenceServer],
) -> None:
    addr, _ = server
    query = "history_frames=24&stride=1&stack_keys=image.wrist,image.exterior,robot_state.ee_pose,grip&pad_start=false"
    ws, ready = _open(addr, f"/api/v1/session?{query}")
    ws.close()
    assert ready["status"] == "ready"


def test_call_errors(server: tuple[str, InferenceServer]) -> None:
    addr, _ = server
    ws, _ = _open(addr)
    ws.send(b"\xc1")
    assert list(_recv(ws)) == ["error"]
    assert _closed(ws)[0] == 1011

    ws, _ = _open(addr)
    ws.send(wire.serialise({"image.exterior": np.zeros((1, 8, 8, 3), np.uint8)}))
    assert _recv(ws)["error"] == (
        "\"obs missing any of ('ee_pose', 'robot_state.ee_pose'); "
        "got keys=['image.exterior']\""
    )
    assert _closed(ws)[0] == 1011

    ws, _ = _open(addr)
    ws.send(wire.serialise({"robot_state.ee_pose": np.zeros(3, np.float32)}))
    assert _recv(ws)["error"] == "ee_pose must be (>=7,), got (3,)"
    assert _closed(ws)[0] == 1011

    ws, _ = _open(addr)
    ws.send("hello")
    assert _recv(ws)["status"] == "error"
    assert _closed(ws)[0] == 1008


def test_chunk_budget(server: tuple[str, InferenceServer]) -> None:
    addr, _ = server
    ws, _ = _open(addr, "/api/v1/session?max_chunks_per_episode=1")
    ws.send(wire.serialise(_droid_obs(1)))
    assert len(_recv(ws)["result"]) == 21
    ws.send(wire.serialise(_droid_obs(22)))
    assert _recv(ws) == {"error": "max_chunks_per_episode=1 reached"}
    ws.close()


def test_open_sessions_keep_their_own_context(
    server: tuple[str, InferenceServer],
) -> None:
    addr, app = server
    pipe = _pipe(app.registry["default"])
    pipe.policy.calls.clear()
    a, _ = _open(addr, "/api/v1/session?sample_seed=0")
    b, _ = _open(addr, "/api/v1/session?sample_seed=0")
    for ws, t in ((a, 1), (b, 1), (a, 22), (b, 1)):
        ws.send(wire.serialise(_droid_obs(t)))
        _recv(ws)
    a.close()
    b.close()
    assert [c["depth"] for c in pipe.policy.calls] == [0, 0, 1, 0]


def test_boot_rejects_bad_defaults() -> None:
    with pytest.raises(ValueError, match="Unknown session settings"):
        _single_arm_factory(num_steps=3)
    factory = _single_arm_factory(context_mode="rolling")
    with pytest.raises(ValueError, match="rolling"):
        factory({})


def test_boot_rejects_a_checkpoint_that_is_not_single_arm() -> None:
    sampler = SamplerSettings(num_steps=2, layout="window_joint")
    pipe = FakePipeline({**SINGLE_ARM, "embodiment": "two_arm"}, sampler)
    with pytest.raises(ValueError, match="single-arm checkpoint only"):
        PolicyFactory(pipe, {}, "/ckpt/other")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("config", ["serve_droid.yaml", "serve_droid_k.yaml"])
@pytest.mark.parametrize(
    ("embodiment", "overrides", "error"),
    [
        ("single_arm", [], None),
        ("two_arm", [], "Serve the single-arm checkpoint"),
        ("single_arm", ["model.embodiment=two_arm"], "Serve the single-arm"),
        ("two_arm", ["model.embodiment=two_arm"], "single-arm checkpoint only"),
    ],
)
def test_boot_checks_the_rig_of_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    config: str,
    embodiment: str,
    overrides: list[str],
    error: str | None,
) -> None:
    sessions: list[dict] = []

    def pipeline(**kwargs: object) -> FakePipeline:
        sampler = kwargs["sampler"]
        assert isinstance(sampler, SamplerSettings)
        pipe = FakePipeline({**SINGLE_ARM, "embodiment": embodiment}, sampler)
        open_session = pipe.session

        def session(**kw: object) -> Session:
            sessions.append(kw)
            return open_session(**kw)

        pipe.session = session  # ty: ignore[invalid-assignment]
        return pipe

    monkeypatch.setattr(server_module, "PraxisPipeline", pipeline)
    cfg = server_module.load_config(
        ROOT / "configs" / config, ["model.device=cpu", *overrides]
    )
    if error is None:
        factory = server_module.build_factory(cfg)
        assert factory.pipe.embodiment == "single_arm"
        assert factory.pipe.sampler == SamplerSettings(**cfg["sampler"])
        # The boot opens one session with the defaults of the config.
        assert sessions[0]["control_mode"] == "droid_impedance"
    else:
        with pytest.raises(ValueError, match=error):
            server_module.build_factory(cfg)


def test_warmup_runs_each_context_depth() -> None:
    from praxis.serve.policy import warmup_obs

    factory = _single_arm_factory()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        server_module.warmup(factory, executor)
    finally:
        executor.shutdown()
    calls = _pipe(factory).policy.calls
    assert [c["depth"] for c in calls] == [0, 1, 2, 3]
    assert all(c["settings"].num_steps == 2 for c in calls)
    obs = warmup_obs(0)
    assert list(obs) == ["image.exterior", "image.wrist", "robot_state.ee_pose", "grip"]
    assert obs["image.exterior"].shape == (25, 288, 512, 3)


def test_code_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PRAXIS_SERVED_COMMIT", "abc")
    assert served_code_version() == "abc"
    monkeypatch.delenv("PRAXIS_SERVED_COMMIT")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    try:
        subprocess.run([*git, "init", "-q"], check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    tracked = tmp_path / "a.txt"
    tracked.write_text("1")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    sha = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert params.git_commit(tmp_path) == sha
    tracked.write_text("2")
    assert params.git_commit(tmp_path) == sha + "-dirty"
    # A directory inside another repository is not a praxis checkout.
    (tmp_path / "site-packages").mkdir()
    assert params.git_commit(tmp_path / "site-packages") == "unknown"


def test_smoothing_filters_xyz_and_grip_only() -> None:
    from scipy.signal import savgol_filter

    factory = _single_arm_factory()
    plain = factory({"sample_seed": 3}).select_action(_droid_obs(1))
    query = {"sample_seed": 3, "smoothing": "sg", "sg_window": 5, "sg_order": 2}
    smooth = factory(query).select_action(_droid_obs(1))
    assert len(plain) == len(smooth) == 21
    grips = np.asarray([r["target_grip"] for r in plain], np.float32)
    want = savgol_filter(grips, 5, 2).astype(np.float32)
    assert np.array_equal(
        np.asarray([r["target_grip"] for r in smooth], np.float32), want
    )
    for a, b in zip(plain, smooth, strict=True):
        assert np.array_equal(
            a["robot_command"]["pose"][3:], b["robot_command"]["pose"][3:]
        )


class OutOfRangePolicy(FakePolicy):
    """Stand-in that gives normalized actions far outside [-1, 1].

    Each xyz dim holds one value for all the rows of the chunk. The grip
    jumps from ``-value`` to ``+value`` in the middle of the chunk, so the
    smoothing filter goes past the end stops of the gripper.
    """

    def __init__(self, config: dict, value: float) -> None:
        super().__init__(config)
        self.value = value

    def sample_window_joint(
        self, text, text_uncond, anchor, cleans, proprio_all, seed, settings, trace=None
    ):
        self.calls.append({"depth": len(cleans), "settings": settings})
        rows = np.zeros((24, 10), np.float32)
        rows[:, :3] = self.value * np.array([1.0, -1.0, 1.0])
        rows[:, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        rows[:12, 9], rows[12:, 9] = -self.value, self.value
        actions = torch.from_numpy(rows.reshape(1, 6, 40))
        return torch.zeros(1, 1, 6, 1, 1), actions


def _grips(rows: list[dict]) -> np.ndarray:
    return np.asarray([r["target_grip"] for r in rows], np.float32)


@pytest.mark.parametrize("smoothing", [None, "savgol"])
@pytest.mark.parametrize("value", [1.5, 10.0])
def test_out_of_range_actions_give_commands_in_the_training_range(
    smoothing: str | None, value: float
) -> None:
    """The decode clamps the actions, with and without smoothing.

    The xyz of each command is in [q01, q99] around the current pose and each
    grip is in [0, 1]. The xyz is constant over the chunk, thus the filter
    does not change it (the decode does not clip the xyz after the filter).
    """
    from scipy.signal import savgol_filter

    factory = _single_arm_factory()
    pipe = _pipe(factory)
    pipe.policy = OutOfRangePolicy(pipe.policy.config, value)
    stats = pipe.stats
    stats.action_q01[9], stats.action_q99[9] = 0.0, 1.0
    query: dict[str, object] = {"sample_seed": 0}
    if smoothing:
        query.update({"smoothing": smoothing, "sg_window": 9, "sg_order": 3})
    obs = _droid_obs(1)
    out = factory(query).select_action(obs)
    assert len(out) == 21  # depth 0
    anchor = obs["robot_state.ee_pose"][-1]
    lo = stats.action_q01[:3].astype(np.float64)
    hi = stats.action_q99[:3].astype(np.float64)
    for row in out:
        xyz = row["robot_command"]["pose"][:3].astype(np.float64) - anchor[:3]
        assert np.all(xyz >= lo - 1e-6) and np.all(xyz <= hi + 1e-6)
        assert 0.0 <= row["target_grip"] <= 1.0
    grips = _grips(out)
    assert grips.min() == 0.0 and grips.max() == 1.0
    between = (grips > 0.0) & (grips < 1.0)
    assert between.any() if smoothing else not between.any()
    if smoothing:
        # Without the clip after the filter, the grips go past 0 and 1.
        steps = np.asarray([0.0] * (len(out) - 12) + [1.0] * 12)
        rings = savgol_filter(steps, 9, 3)
        assert rings.min() < 0.0 and rings.max() > 1.0


def test_step_result_holds_raw_rows_and_clamped_actions() -> None:
    """``action_rows_norm`` has no clip. ``action_rel`` gives the commands."""
    from praxis import robot_io, single_arm

    factory = _single_arm_factory()
    pipe = _pipe(factory)
    pipe.policy = OutOfRangePolicy(pipe.policy.config, 1.5)
    stats = pipe.stats
    session = factory({"sample_seed": 0}).session
    res = session.step(_droid_obs(1))
    dims = single_arm.Q99_DIMS
    assert np.abs(res.action_rows_norm[:, dims]).max() == 1.5
    want = robot_io.denormalize_rows(res.action_rows_norm, stats, dims)
    want = robot_io.clip_grip(want, single_arm.GRIP_DIMS)
    assert res.action_rel.tobytes() == want.tobytes()
    state = session.state
    assert isinstance(state, single_arm.SingleArmServeSession)
    # The fps, the control mode and the pose frame of the session.
    wire = state.rel_to_wire(res.action_rel, res.inputs["pose8_now"])
    assert wire[0]["robot_command"]["mode"] == single_arm.DROID_IMPEDANCE
    poses, grips = single_arm.wire_to_arrays(wire)
    got_poses, got_grips = single_arm.wire_to_arrays(res.commands)
    assert np.array_equal(poses, got_poses) and np.array_equal(grips, got_grips)


def test_smoothing_filters_the_grips_before_the_grip_clip() -> None:
    """The filter runs on the grips of the inverse q99 map, with no grip clip.

    Grip stats of (-0.5, 1.5) put the grips outside [0, 1] before the filter.
    """
    from scipy.signal import savgol_filter

    factory = _single_arm_factory()
    pipe = _pipe(factory)
    pipe.policy = OutOfRangePolicy(pipe.policy.config, 1.0)
    stats = pipe.stats
    stats.action_q01[9], stats.action_q99[9] = -0.5, 1.5
    query = {"sample_seed": 0, "smoothing": "savgol", "sg_window": 9, "sg_order": 3}
    out = factory(query).select_action(_droid_obs(1))
    raw = np.asarray([-0.5] * (len(out) - 12) + [1.5] * 12, np.float32)
    want = np.clip(savgol_filter(raw, 9, 3), 0.0, 1.0).astype(np.float32)
    assert np.allclose(_grips(out), want, rtol=0, atol=1e-6)


@pytest.mark.parametrize(("window", "order"), [(31, 3), (3, 3), (5, 7)])
def test_smoothing_keeps_a_chunk_it_cannot_filter(window: int, order: int) -> None:
    """A window longer than the chunk, or not above the order, keeps the rows."""
    factory = _single_arm_factory()
    plain = factory({"sample_seed": 3}).select_action(_droid_obs(1))
    query = {
        "sample_seed": 3,
        "smoothing": "sg",
        "sg_window": window,
        "sg_order": order,
    }
    smooth = factory(query).select_action(_droid_obs(1))
    assert wire.serialise(plain) == wire.serialise(smooth)
