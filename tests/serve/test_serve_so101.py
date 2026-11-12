"""The policy server on an so101 checkpoint (CPU only, no weights).

The real server app runs under uvicorn on a free local port, with a stand-in
for the networks. The client speaks the protocol of the LeRobot client
(``eval/so101/so101_client.py``): the query keys ``views``, ``grip_closed``,
``grip_open``, ``state`` and the JPEG image markers in, ``actions`` ``(R, 6)``
out. The real-time sessions run as the periodic recipe of a Positronic client
does (``test_serve_rtc._drive``), on the stand-in networks and on the tiny
real model of ``tests/tiny_model.py``.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("msgpack")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")
pytest.importorskip("scipy")

import msgpack
import uvicorn
from test_serve_precompute import _PlanPolicy
from test_serve_rtc import DELAYS, FPS, SAMPLE_FPS, _drive, _Encoder, _Pipeline
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from praxis import so101
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline, Session
from praxis.robot_io import NormStats
from praxis.serve import rtc, wire
from praxis.serve import server as server_module
from praxis.serve.policy import PolicyFactory, ServedPolicy
from praxis.serve.server import InferenceServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))  # tiny_model, as the tests of tests/ import it
SO101 = json.loads((ROOT / "tests" / "data" / "praxis-1-so101.json").read_text())
YAM = json.loads((ROOT / "tests" / "data" / "praxis-1-yam.json").read_text())
REST = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])


class FakePolicy:
    """Stand-in for the networks: small tokens of the so101 head (6 x 24)."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.calls: list[dict] = []

    def sample_window_joint(
        self,
        text,
        text_uncond,
        anchor,
        cleans,
        proprio_all,
        seed,
        settings,
        trace=None,
        skip_last_video=None,
        ctx_cache=False,
        overlap_expert=False,
        pair_cfg=False,
    ):
        self.calls.append(
            {"depth": len(cleans), "settings": settings, "hw": anchor.shape[-2:]}
        )
        generator = torch.Generator().manual_seed(int(seed))
        return None, 0.1 * torch.randn((1, 6, 24), generator=generator)


class FakeEncoder:
    norm_block_c: dict[int, int] | None = None

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        return torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 16, 16))


class FakePipeline:
    def __init__(self, config: dict, sampler: SamplerSettings) -> None:
        self.device = torch.device("cpu")
        self.policy = FakePolicy(config)
        self.vae = FakeEncoder()
        self.sampler = sampler
        self.embodiment = str(config["embodiment"])
        n = config["normalization"]
        self.stats = NormStats(
            **{k: np.asarray(n[k], np.float32) for k in NormStats.__annotations__}
        )

    def encode_text(self, prompt: str) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def encode_uncond_text(self) -> torch.Tensor:
        return torch.zeros(1, 4, 8)

    def session(self, **kwargs: object) -> Session:
        return PraxisPipeline.session(
            self,  # ty: ignore[invalid-argument-type]
            **kwargs,  # ty: ignore[invalid-argument-type]
        )


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "serve_so101.yaml").read_text())


def _factory(config: dict = SO101, **session: object) -> PolicyFactory:
    cfg = _config()
    pipe = FakePipeline(config, SamplerSettings.from_config(cfg["sampler"]))
    return PolicyFactory(
        pipe,  # ty: ignore[invalid-argument-type]
        {**cfg["session"], **session},
        "/ckpt/praxis-1-so101-nvfp4",
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def server() -> Iterator[tuple[str, PolicyFactory]]:
    executor = ThreadPoolExecutor(max_workers=1)
    factory = _factory()
    app = InferenceServer({"default": factory}, executor)
    port = _free_port()
    config = uvicorn.Config(app.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    yield f"127.0.0.1:{port}", factory
    srv.should_exit = True
    thread.join(timeout=10)
    executor.shutdown()


def _jpeg(frames: np.ndarray) -> dict:
    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return {b"__jpeg__": True, b"frames": out, b"ndim": 4}


def _request(n: int, views: int, seed: int = 0, **extra: object) -> bytes:
    """A request as the LeRobot client packs it: the stack of ``n`` entries."""
    rng = np.random.default_rng(seed)
    state = REST + rng.normal(0, 1, (n, 6))
    msg: dict = {
        "state": state.astype(np.float32),
        "task": "Pick up the red cube and put it in the box.",
    }
    for v in ("exterior", "wrist")[:views]:
        msg[f"image.{v}"] = _jpeg(rng.integers(0, 255, (n, 312, 416, 3), np.uint8))
    msg.update(extra)
    return msgpack.packb(msg, default=wire.pack_numpy)


def _url(addr: str, **query: object) -> str:
    q = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in query.items()}
    return f"ws://{addr}/api/v1/session?{urlencode(q)}"


def _recv(ws: Connection, timeout: float = 30.0) -> dict:
    raw = ws.recv(timeout=timeout)
    assert isinstance(raw, bytes)
    return wire.deserialise(raw)


def _handshake(ws: Connection) -> dict:
    while True:
        m = _recv(ws, 10.0)
        if m.get("status") == "ready":
            return m["meta"]
        assert m.get("status") == "loading", m


# ---------------------------------------------------------------------------
# Chunked sessions over a WebSocket
# ---------------------------------------------------------------------------


def test_the_lerobot_client_session(server: tuple[str, PolicyFactory]) -> None:
    addr, factory = server
    url = _url(
        addr,
        views=2,
        prompt="Pick up the red cube and put it in the box.",
        grip_closed=3.8,
        grip_open=37.3,
        sample_seed=0,
    )
    with connect(url, open_timeout=10) as ws:
        meta = _handshake(ws)
        assert meta["embodiment"] == "so101_joints" and meta["views"] == 2
        assert meta["wire_commands"] == ["joints"] and meta["stack_entries"] == 25
        assert meta["grip_range"] == {"closed": 3.8, "open": 37.3}
        assert meta["view_order"] == ["image.exterior", "image.wrist"]
        assert meta["canvas_native"] == [288, 832] and meta["pose_frame"] == "joints"
        stack = [e for e in meta["local_stack"]["seq"] if e["name"] == "temporal_stack"]
        assert stack[0]["args"]["keys"] == ["image.exterior", "image.wrist", "state"]
        assert len(stack[0]["args"]["offsets_sec"]) == 25
        names = [e["name"] for e in meta["local_stack"]["seq"]]
        assert names == [
            "stop_on_fault",
            "restrict_image_size",
            "temporal_stack",
            "chunked_schedule",
        ]
        for n, rows in ((1, 21), (22, 24), (25, 24)):
            ws.send(_request(n, 2))
            rep = _recv(ws)
            assert "error" not in rep, rep
            actions = np.asarray(rep["actions"])
            assert actions.shape == (rows, 6) and np.isfinite(actions).all()
            assert actions.dtype == np.float32
            assert len(rep["result"]) == rows and "joints" in rep["result"][0]
            np.testing.assert_array_equal(
                actions, np.stack([r["joints"] for r in rep["result"]])
            )
            assert rep["joint_names"] == list(so101.JOINT_NAMES)
            assert 3.8 <= actions[:, 5].min() and actions[:, 5].max() <= 37.3
            np.testing.assert_allclose(rep["timestamps"][:2], [0.0, 1 / 15])
            assert rep["server_timing"]["policy_ms"] >= 0
            assert (
                rep["effective_prompt"] == "Pick up the red cube and put it in the box."
            )
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    assert [c["depth"] for c in pipe.policy.calls] == [0, 1, 1]
    # 416x312 frames: 288x832 canvas -> 18x52 latents.
    assert tuple(pipe.policy.calls[-1]["hw"]) == (18, 52)


def test_one_view_session(server: tuple[str, PolicyFactory]) -> None:
    addr, factory = server
    with connect(_url(addr, views=1), open_timeout=10) as ws:
        meta = _handshake(ws)
        assert meta["views"] == 1 and meta["view_order"] == ["image.exterior"]
        stack = [e for e in meta["local_stack"]["seq"] if e["name"] == "temporal_stack"]
        assert stack[0]["args"]["keys"] == ["image.exterior", "state"]
        ws.send(_request(1, 1))
        assert np.asarray(_recv(ws)["actions"]).shape == (21, 6)
        ws.send(_request(22, 1))
        assert np.asarray(_recv(ws)["actions"]).shape == (24, 6)
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    assert tuple(pipe.policy.calls[-1]["hw"]) == (18, 26)


def test_a_short_stack_is_a_cold_start_and_a_bad_state_an_error(
    server: tuple[str, PolicyFactory],
) -> None:
    addr, factory = server
    with connect(_url(addr, views=2), open_timeout=10) as ws:
        _handshake(ws)
        ws.send(_request(1, 2))
        _recv(ws)
        ws.send(_request(5, 2))  # fewer than the 21 executed rows: a restart
        assert np.asarray(_recv(ws)["actions"]).shape == (21, 6)
        ws.send(_request(22, 2, state=np.full((22, 6), 2048.0, np.float32)))
        assert "use_degrees" in _recv(ws)["error"]
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    assert [c["depth"] for c in pipe.policy.calls] == [0, 0]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"views": 3}, "views must be 1 or 2"),
        ({"views": "2"}, "views must be 1 or 2"),
        ({"grip_closed": 10.0, "grip_open": 10.5}, "must differ"),
        ({"grip_closed": "low"}, "grip_closed must be a number"),
        ({"schedule": "rtc", "rtc_reach_max_m": "0.6"}, "so101 rig has none"),
        ({"control_mode": "droid_impedance"}, "control_mode"),
    ],
)
def test_the_session_parameters_of_the_rig_are_checked(
    overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _factory()(overrides)


def test_the_rig_parameters_belong_to_the_so101_rig() -> None:
    yam = PolicyFactory(
        FakePipeline(YAM, SamplerSettings()),  # ty: ignore[invalid-argument-type]
        {"control_mode": "none"},
        "/ckpt/praxis-1-yam",
    )
    with pytest.raises(
        ValueError,
        match=r"\['grip_open', 'views'\] are session parameters of the so101",
    ):
        yam({"views": 1, "grip_open": 40.0})
    assert yam({}).rig_options == {} and yam({}).stack_keys == yam.stack_keys


def test_a_stateless_session_cold_starts_each_call() -> None:
    policy = _factory()({"exec_horizon": 12, "views": 1})
    assert policy.context_mode == "stateless" and policy.history_frames == 0
    assert policy.meta["stack_entries"] == 1
    for step in range(3):
        rows = policy.select_action(so101.warmup_obs(step, 25, views=1))
        assert rows is not None and len(rows) == 12
    pipe = policy.factory.pipe
    assert isinstance(pipe, FakePipeline)
    assert [c["depth"] for c in pipe.policy.calls] == [0, 0, 0]


def test_a_grip_reading_high_when_closed_serves_targets_in_its_range() -> None:
    policy = _factory()({"grip_closed": 40.0, "grip_open": 5.0, "sample_seed": 0})
    rows = policy.select_action(so101.warmup_obs(0))
    assert rows is not None
    grip = so101.wire_to_targets(rows)[:, 5]
    assert 5.0 <= grip.min() and grip.max() <= 40.0
    assert policy.meta["grip_range"] == {"closed": 40.0, "open": 5.0}


def test_warmup_covers_both_canvases() -> None:
    factory = _factory()
    executor = ThreadPoolExecutor(max_workers=1)
    server_module.warmup(factory, executor)
    executor.shutdown()
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    calls = [(c["depth"], tuple(c["hw"])) for c in pipe.policy.calls]
    assert calls == [(0, (18, 52)), (1, (18, 52)), (0, (18, 26)), (1, (18, 26))]


def test_serve_config_is_the_so101_preset() -> None:
    cfg = _config()
    assert cfg["model"]["embodiment"] == "so101"
    assert cfg["model"]["routed_precision"] == "nvfp4"
    assert cfg["model"]["nvfp4_cache"] is None
    s = cfg["sampler"]
    assert (s["num_steps"], s["t_shift"], s["cfg_scale"], s["cfg_video_scale"]) == (
        4,
        12.5,
        1.0,
        4.0,
    )
    SamplerSettings.from_config(s)
    session = cfg["session"]
    assert session["max_context_chunks"] == 1 and session["exec_horizon"] == 24
    assert session["skip_video_decoder"] is True and session["prompt"] == ""
    assert session["schedule"] == "chunked" and session["control_mode"] == "none"
    # The bf16 configs keep the default precision.
    for name in ("serve_droid.yaml", "serve_yam.yaml"):
        other = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert other["model"].get("routed_precision", "bf16") == "bf16"


def test_a_real_time_session_over_the_websocket_answers_joint_rows(
    server: tuple[str, PolicyFactory],
) -> None:
    """``schedule=rtc`` through the server: protocol v2 samples with the
    current frames and state, the first sample predicts 21 joint rows without
    timestamps, and the answer carries the rig keys ``actions`` and
    ``joint_names`` (no ``timestamps``) next to ``result``."""
    addr, _factory = server
    url = _url(addr, views=1, schedule="rtc", sample_seed=0, prompt="Pick up.")
    rng = np.random.default_rng(0)
    with connect(url, open_timeout=10) as ws:
        ready = None
        while ready is None:
            m = _recv(ws, 10.0)
            ready = m if m.get("status") == "ready" else None
        assert ready["protocol_version"] == 2
        sid = ready["session_id"]
        predicted = []
        for n in range(3):
            frame = _jpeg(rng.integers(0, 255, (1, 312, 416, 3), np.uint8))
            frame[b"ndim"] = 3
            obs = {
                "state": (REST + rng.normal(0, 1, 6)).astype(np.float32),
                "image.exterior": frame,
                "sample_index": n,
                "obs_time_ns": n * 10**9 // 15,
                "call_id": n,
                "call_timings": [],
            }
            ws.send(msgpack.packb({"session_id": sid, "observation": obs}, default=wire.pack_numpy))  # fmt: skip
            rep = _recv(ws)
            assert "error" not in rep, rep
            if rep["result"] is not None:
                predicted.append(rep)
        assert len(predicted) == 1
        rep = predicted[0]
        assert len(rep["result"]) == 21 and set(rep["result"][0]) == {"joints"}
        actions = np.asarray(rep["actions"])
        assert actions.shape == (21, 6) and np.isfinite(actions).all()
        assert rep["joint_names"] == list(so101.JOINT_NAMES)
        assert "timestamps" not in rep
        assert set(rep["timing"]) == {"served_ms", "decode_ms", "infer_ms", "queued_ms"}  # fmt: skip


# ---------------------------------------------------------------------------
# The joint-space junction
# ---------------------------------------------------------------------------


def _joint_line(q0: float, n: int, step: float = 1.0, grip: float = 10.0) -> np.ndarray:
    out = np.zeros((n, 6))
    out[:, 0] = q0 + step * np.arange(n)
    out[:, 1] = -100.0
    out[:, 5] = grip
    return out


def test_rebase_joints_is_the_pose_rule_on_each_joint() -> None:
    """Rows ``k < P`` are the prefix; the offset of the prefix end is handed
    back over ``decay`` rows; the grip is the model's. The same numbers as a
    direct computation of the rule."""
    prefix = _joint_line(0.0, 5)
    new = _joint_line(-8.0, 24, step=2.0, grip=30.0)
    out, stats = rtc.rebase_joints(new, prefix, decay=12)
    dq = prefix[4, 0] - new[4, 0]
    assert stats["junction_deg"] == pytest.approx(abs(dq)) and stats["prefix_rows"] == 5
    np.testing.assert_array_equal(out[:5], prefix)
    w = np.array([1.0 - rtc._ramp(i, 12) if i < 12 else 0.0 for i in range(19)])
    np.testing.assert_allclose(out[5:, 0], new[5:, 0] + w * dq, atol=1e-12)
    np.testing.assert_allclose(out[5:, 1], new[5:, 1], atol=1e-12)
    np.testing.assert_array_equal(out[5:, 5], new[5:, 5])
    held, _ = rtc.rebase_joints(new, prefix, decay=0)
    np.testing.assert_allclose(held[5:, 0], new[5:, 0] + dq, atol=1e-12)
    # No prefix: the model's rows.
    same, stats = rtc.rebase_joints(new, new[:0])
    np.testing.assert_array_equal(same, new)
    assert stats["junction_deg"] == 0.0


def test_vel_fade_and_grip_crossfade_on_joint_rows() -> None:
    prefix = _joint_line(0.0, 6, step=1.0)  # +1 deg per row on joint 0
    new = _joint_line(5.0, 24, step=0.0)
    new[:, 2] = 95.0 - 1.0 * np.arange(24)  # the model moves joint 2
    plain, _ = rtc.rebase_joints(new, prefix, 24)
    out, _ = rtc.rebase_joints(new, prefix, 24, 6)
    np.testing.assert_allclose(out[:6], prefix)
    step = out[6] - out[5]
    # The first served step mostly continues the prefix (joint 0), not the model (joint 2).
    assert step[0] > 0.7 and abs(step[2]) < 0.3
    np.testing.assert_allclose(out[6 + 24 - 6 :], plain[6 + 24 - 6 :])
    grips = np.zeros((24, 6))
    grips[4:, 5] = 90.0
    cont = np.zeros((12, 6))
    cont[:, 5] = 10.0
    faded = grips.copy()
    rtc.grip_crossfade_joints(faded, 4, cont, 6)
    want = [10.0 + rtc._ramp(i, 6) * 80.0 for i in range(6)]
    np.testing.assert_allclose(faded[4:10, 5], want)
    np.testing.assert_array_equal(faded[10:], grips[10:])
    np.testing.assert_array_equal(faded[:, :5], grips[:, :5])


def test_joint_rows_read_and_write_the_wire() -> None:
    rows = [{"joints": np.arange(6, dtype=np.float32) + t, "x": t} for t in range(3)]
    arr = rtc.joint_rows_to_array(rows, "joints", 6)
    assert arr.shape == (3, 6) and arr.dtype == np.float64 and arr[2, 0] == 2.0
    back = rtc.write_joint_rows(rows, arr + 1.0, "joints")
    assert back[1]["x"] == 1 and back[1]["joints"].dtype == np.float32
    np.testing.assert_array_equal(back[1]["joints"], np.arange(6) + 2.0)
    with pytest.raises(ValueError, match="6 finite joint targets"):
        rtc.joint_rows_to_array([{"joints": np.zeros(5)}], "joints", 6)
    with pytest.raises(ValueError, match="command mappings"):
        rtc.joint_rows_to_array([3], "joints", 6)  # ty: ignore[invalid-argument-type]
    plan = {
        "actions": rows,
        "chunk_offset_sec": 2 / FPS,
        "observation_offset_sec": -0.01,
    }
    assert rtc.read_plan_rows({"action_prefix": plan}, FPS) == rows
    assert rtc.read_plan_rows({}, FPS) is None
    with pytest.raises(ValueError, match="sampled prefix list"):
        rtc.read_plan_rows({"action_prefix": rows}, FPS)
    embedded = rtc.joints_as_arms(arr)
    assert embedded.shape == (3, 5, 13)
    np.testing.assert_array_equal(rtc.arms_as_joints(embedded), arr)


# ---------------------------------------------------------------------------
# Real-time sessions on the stand-in networks
# ---------------------------------------------------------------------------


def _frame(raw: float, h: int = 48, w: int = 64) -> np.ndarray:
    """A view whose blue channel holds the raw index (``_Encoder`` reads it)."""
    f = np.zeros((h, w, 3), np.uint8)
    f[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    f[..., 2] = int(raw) % 256
    return f


def _state(raw: float) -> np.ndarray:
    q = REST + 0.2 * np.sin(raw / 15.0 + np.arange(6))
    q[5] = 5.0 + raw % 30
    return q.astype(np.float32)


def _sample(raw: float, views: int = 2) -> dict:
    """The current observation of the LeRobot rig at the action raw ``raw``."""
    obs: dict = {"state": _state(raw)}
    for v in so101.VIEWS[:views]:
        obs[so101.VIEW_KEYS[v]] = _frame(raw)
    return obs


def _rtc_factory(**session: object) -> PolicyFactory:
    pipe = _Pipeline(SO101)
    return PolicyFactory(
        pipe,  # ty: ignore[invalid-argument-type]
        {
            "exec_horizon": 24,
            "control_mode": "none",
            "max_context_chunks": 1,
            "schedule": "chunked",
            **session,
        },
        "/ckpt/praxis-1-so101-nvfp4",
    )


def test_a_real_time_so101_session_declares_the_periodic_recipe() -> None:
    policy = _rtc_factory()({"schedule": "rtc", "views": 1})
    meta = policy.meta
    names = [e["name"] for e in policy.local_stack["seq"]]
    assert names == [
        "stop_on_fault",
        "restrict_image_size",
        "trajectory_player",
        "periodic",
        "inference_schedule",
    ]
    assert policy.protocol_version == 2 and policy.max_in_flight == 16
    assert meta["schedule"] == "rtc" and meta["rtc"]["reach_max_m"] == []
    assert meta["rtc"]["stride"] == 12 and meta["views"] == 1
    assert policy.stack_keys == ("image.exterior", "state")
    assert policy._history is not None
    assert policy._history.keys == ("image.exterior", "state")


def test_a_real_time_so101_session_predicts_on_the_real_time_grid() -> None:
    """The first prediction at once, the next one 12 rows plus the first
    duration later, then every 12 rows; the answers are joint rows without
    timestamps: 21 rows while the context has no chunk, then 24."""
    policy = _rtc_factory()(
        {"schedule": "rtc", "sample_seed": 0, "sample_fps": 30, "precompute": "off"}
    )
    answers, at, nones = _drive(policy, DELAYS, 9.0, sample=_sample)
    assert len(answers) == len(DELAYS)
    gaps = np.diff(at).tolist()
    assert gaps[0] == 24 + 9 and set(gaps[1:]) == {24}
    assert len(nones) > 10 * len(answers)
    assert all(set(row) == {"joints"} for rows in answers for row in rows)
    assert [len(r) for r in answers] == [21, 21, *[24] * (len(answers) - 2)]
    for rows in answers:
        q = so101.wire_to_targets(rows)
        assert np.isfinite(q).all() and q.dtype == np.float32
        assert (q[:, 5] >= so101.DEFAULT_GRIP_CLOSED - 1e-3).all()
        assert (q[:, 5] <= so101.DEFAULT_GRIP_OPEN + 1e-3).all()
    assert policy.timing["rtc_prefix_rows"] > 0 and "rtc_junction_deg" in policy.timing


def test_the_window_of_each_so101_prediction_is_the_chunked_stack() -> None:
    """Each prediction after the cold starts encodes the 21 stack rows before
    its current frame, one every 1/15 s, as a chunked call would."""
    factory = _rtc_factory()
    policy = factory(
        {"schedule": "rtc", "sample_seed": 0, "sample_fps": 30, "precompute": "off"}
    )
    _answers, at, _ = _drive(policy, DELAYS[:6], 6.0, sample=_sample)
    vae = factory.pipe.vae
    assert isinstance(vae, _Encoder)
    windows = [r for r in vae.raws if len(r) > 1]
    want = [list(range(n // 2 - 21, n // 2)) for n in at[2:]]
    assert windows == want and len(windows) == len(at) - 2


def test_the_so101_junction_keeps_the_plan_until_the_answer_arrives() -> None:
    """After the first prediction the answer repeats the joint rows of the plan
    that run before it arrives (the delay estimate), then the model takes over
    with the offset handed back."""
    policy = _rtc_factory()(
        {"schedule": "rtc", "sample_seed": 0, "sample_fps": 30, "precompute": "off"}
    )
    answers, at, _ = _drive(policy, DELAYS[:4], 5.0, sample=_sample)
    first, second = (so101.wire_to_targets(a).astype(np.float64) for a in answers[:2])
    # The second call runs at sample at[1]: the first chunk started at its
    # answer (0.30 s), so row (t - 0.30) * 15 runs.
    t = at[1] / SAMPLE_FPS
    running = int(np.floor((t - 0.30) * FPS + 1e-9))
    p = rtc.prefix_rows(0.30, FPS, len(first) - running)
    assert p == 5
    np.testing.assert_allclose(second[:p], first[running : running + p], atol=1e-5)
    # After the prefix the model's rows continue from its end (no jump).
    assert (
        np.abs(second[p] - second[p - 1]).max()
        < np.abs(first[p] - second[p - 1]).max() + 1.0
    )


def test_the_precompute_keeps_the_bits_of_an_so101_real_time_session() -> None:
    """A session with the precompute (the default) gives the rows of a session
    without it, and reports its plans as hits once the context has a chunk."""
    runs = {}
    for pre in ("on", "off"):
        factory = _rtc_factory()
        factory.pipe.policy = _PlanPolicy(SO101)  # ty: ignore[unresolved-attribute]
        policy = factory(
            {"schedule": "rtc", "sample_seed": 0, "sample_fps": 15, "precompute": pre}
        )
        plans: list[object] = []

        def sample(
            raw: float, policy: ServedPolicy = policy, plans: list = plans
        ) -> dict:
            plans.append(policy.timing.get("rtc_plan"))
            return _sample(raw)

        answers, _at, _ = _drive(policy, DELAYS[:6], 6.0, sample=sample, sample_fps=15)
        runs[pre] = (answers, [p for p in plans if p is not None])
    on, off = runs["on"][0], runs["off"][0]
    assert len(on) == len(off) == 6
    for a, b in zip(on, off, strict=True):
        np.testing.assert_array_equal(
            so101.wire_to_targets(a), so101.wire_to_targets(b)
        )
    assert "hit" in runs["on"][1] and runs["off"][1] == []


# ---------------------------------------------------------------------------
# The tiny real model
# ---------------------------------------------------------------------------


def _tiny_so101_config() -> dict:
    """The tiny single-arm config of ``tests/tiny_model.py`` as an so101
    checkpoint: the 6-D head, the joints pose frame, the fit-rule canvas."""
    from tiny_model import tiny_config

    cfg = tiny_config()
    cfg["embodiment"] = "so101"
    cfg["action_space"] = {"dim": 6}
    cfg["pose_frame"] = "joints"
    cfg["action_expert"]["action_patch_dim"] = 24
    cfg["action_expert"]["proprio_patch_dim"] = 6
    cfg["chunking"].update(grid_hz=15.0)
    cfg["video"] = {
        "views": ["exterior", "wrist"],
        "canvas": "tiles",
        "keep_aspect": True,
        "antialias": True,
        "tile_height": 288,
        "tile_width": 416,
        "height": 288,
        "width": 832,
    }
    cfg["normalization"] = SO101["normalization"]
    return cfg


class _TinyEncoder:
    """A stand-in VAE with the latent layout of the tiny model: 192 channels
    at a 16x stride, causal over time."""

    norm_block_c = None

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        pooled = torch.nn.functional.avg_pool3d(
            clip[:, :, idx], kernel_size=(1, 16, 16)
        )
        return (pooled / 255.0 - 0.5).repeat(1, 64, 1, 1, 1).to(torch.bfloat16)


class _TinyPipeline:
    def __init__(self) -> None:
        from tiny_model import tiny_policy

        config = _tiny_so101_config()
        torch.manual_seed(0)
        policy = tiny_policy()
        from praxis.model import PraxisPolicy

        self.policy = PraxisPolicy.from_config(config)
        with torch.no_grad():
            for p in self.policy.parameters():
                p.normal_(0.0, 0.05)
            for name, buf in self.policy.named_buffers():
                if name.endswith("expert_bias"):
                    buf.normal_(0.0, 0.01)
                elif name.endswith(("freqs", "phases")):
                    buf.uniform_(-1.0, 1.0)
        self.policy = self.policy.to(torch.bfloat16)
        del policy
        self.device = torch.device("cpu")
        self.vae = _TinyEncoder()
        self.sampler = SamplerSettings(num_steps=2, t_shift=12.5, cfg_video_scale=4.0)
        self.embodiment = "so101"
        n = config["normalization"]
        self.stats = NormStats(
            **{k: np.asarray(n[k], np.float32) for k in NormStats.__annotations__}
        )
        g = torch.Generator().manual_seed(1)
        self._text = torch.randn(1, 5, 32, generator=g).to(torch.bfloat16)
        self._uncond = torch.randn(1, 1, 32, generator=g).to(torch.bfloat16)

    def encode_text(self, prompt: str) -> torch.Tensor:
        return self._text

    def encode_uncond_text(self) -> torch.Tensor:
        return self._uncond

    def session(self, **kwargs: object) -> Session:
        return PraxisPipeline.session(
            self,  # ty: ignore[invalid-argument-type]
            **kwargs,  # ty: ignore[invalid-argument-type]
        )


def _tiny_sample(raw: float) -> dict:
    """A 1-view observation of 32x64 frames: the latent grid of the tiny model."""
    return {"state": _state(raw), "image.exterior": _frame(raw, 32, 64)}


def test_chunked_and_real_time_so101_sessions_run_on_the_tiny_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real (tiny) backbone and expert with the 6-D head sample chunked and
    real-time so101 sessions through the server's policy: the actions are
    finite joint targets, the real-time junction keeps the plan prefix, and
    the precompute reports a hit."""
    from torch.nn.attention.flex_attention import flex_attention

    import praxis.layers

    # The compiled CPU flex kernel of torch 2.13 fails to build for some masks.
    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    factory = PolicyFactory(
        _TinyPipeline(),  # ty: ignore[invalid-argument-type]
        {"exec_horizon": 24, "control_mode": "none", "max_context_chunks": 1},
        "/ckpt/tiny-so101",
    )
    chunked = factory({"views": 1, "sample_seed": 0})
    stacks = [[0], list(range(0, 22)), list(range(21, 46))]
    for raws, rows in zip(stacks, (21, 24, 24), strict=True):
        obs = {
            k: np.stack([_tiny_sample(r)[k] for r in raws])
            for k in ("state", "image.exterior")
        }
        out = chunked.select_action(obs)
        assert out is not None and len(out) == rows
        q = so101.wire_to_targets(out)
        assert np.isfinite(q).all() and (np.abs(q[:, :5]) < 400).all()
    assert chunked.session.state.stream.depth == 1
    chunked.close()
    policy = factory({"views": 1, "schedule": "rtc", "sample_seed": 0})
    plans: list[object] = []

    def sample(raw: float) -> dict:
        plans.append(policy.timing.get("rtc_plan"))
        return _tiny_sample(raw)

    answers, at, _ = _drive(policy, DELAYS[:4], 4.0, sample=sample, sample_fps=15)
    assert [len(a) for a in answers] == [21, 21, 24, 24]
    first, second = (so101.wire_to_targets(a).astype(np.float64) for a in answers[:2])
    t = at[1] / 15.0
    running = int(np.floor((t - 0.30) * FPS + 1e-9))
    p = rtc.prefix_rows(0.30, FPS, len(first) - running)
    assert p == 5
    np.testing.assert_allclose(second[:p], first[running : running + p], atol=1e-5)
    assert "hit" in plans
    assert all(np.isfinite(so101.wire_to_targets(a)).all() for a in answers)


# ---------------------------------------------------------------------------
# The LeRobot client against the server
# ---------------------------------------------------------------------------


def test_the_lerobot_client_runs_a_fake_robot_against_the_server(
    server: tuple[str, PolicyFactory], tmp_path: Path
) -> None:
    """eval/so101/so101_client.py --fake-robot: the stack grows with the
    executed rows (1, then 22, then 25 entries), the chunks are 21, 24, 24
    rows, the context depth 0, 1, 1."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "so101_client", ROOT / "eval" / "so101" / "so101_client.py"
    )
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    addr, factory = server
    log = tmp_path / "log.jsonl"
    client.main(
        [
            "--url", f"ws://{addr}/api/v1/session", "--fake-robot", "--wrist-cam", "1",
            "--prompt", "Pick up the cube.", "--max-chunks", "3", "--log-jsonl", str(log),
            "--grip-closed", "3.8", "--grip-open", "37.3",
        ]
    )  # fmt: skip
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [len(r["targets"]) for r in records] == [21, 24, 24]
    assert all(3.8 <= t[5] <= 37.3 for r in records for t in r["targets"])
    assert records[0]["anchor"] == [0.0, -100.0, 95.0, 70.0, 0.0, 2.0]
    pipe = factory.pipe
    assert isinstance(pipe, FakePipeline)
    assert [c["depth"] for c in pipe.policy.calls] == [0, 1, 1]
    assert tuple(pipe.policy.calls[-1]["hw"]) == (18, 52)
