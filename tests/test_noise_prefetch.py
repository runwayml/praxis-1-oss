"""The noise draw ahead of the call (praxis.noise_prefetch), CPU only.

The noise of a call is a pure function of its seed, salt and shape. A draw
made ahead of the call must give the same bytes, and a call that finds no
draw of its key must draw itself. The session tests run a call sequence with
resets, a restart, a seed change and a failed call, with and without the
prefetch, and compare the noise of every call and its outputs bitwise. Real
sessions (``Session.__init__``) read the default from the device and
``PRAXIS_NOISE_PREFETCH``, and keep their bits through a real episode reset.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from test_profiling import _pipe
from test_profiling import _script as _bench
from test_sampler_window_joint import (
    A_DIM,
    LAT,
    P_DIM,
    Calls,
    StubInputs,
    _inputs,
    _policy,
    _sha,
    _stub_expert,
)
from torch import Tensor

import praxis.model as model_mod
import praxis.noise_prefetch as np_mod
from praxis.layers import to_device_async
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.noise_prefetch import NoisePrefetch, draw_window_noise
from praxis.pipeline import PraxisPipeline, Session, noise_prefetch_default
from praxis.profiling import synthetic_obs
from praxis.sampler import ACTION_SALT, VIDEO_SALT, seeded_normal
from praxis.single_arm import SingleArmChunkInputs


def _settle(cache: NoisePrefetch | None) -> None:
    """Wait for the pending draw, as the GPU time of a call would."""
    if cache is not None and cache.future is not None:
        cache.future.exception(timeout=30)


# The DROID window shapes (canvas latents 8x26) at depth 0, 1 and 3.
DROID = [((1, 192, 6 * kk, 8, 26), (6 * kk, 40)) for kk in (1, 2, 4)]


@pytest.mark.parametrize(("video_shape", "action_shape"), DROID)
@pytest.mark.parametrize("seed", [0, 7, 2**31 - 1, 2**40 + 3])
def test_draw_has_the_bytes_of_the_call(
    seed: int, video_shape: tuple[int, ...], action_shape: tuple[int, ...]
) -> None:
    noise_v, noise_a = draw_window_noise(seed, video_shape, action_shape)
    want_v = seeded_normal(video_shape, seed, VIDEO_SALT)
    want_a = seeded_normal(action_shape, seed, ACTION_SALT)
    for got, want in ((noise_v, want_v), (noise_a, want_a)):
        assert got.dtype == want.dtype == torch.float32
        assert got.device == want.device
        assert got.shape == want.shape and got.stride() == want.stride()
        assert _sha(got) == _sha(want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory, CUDA copy")
def test_a_pinned_draw_reaches_the_gpu_with_the_bytes_of_the_call() -> None:
    vs, acts = DROID[2]
    cache = NoisePrefetch(pin=True)
    cache.start(11, vs, acts)
    _settle(cache)
    noise = cache.take(11, vs, acts)
    assert noise is not None and all(t.is_pinned() for t in noise)
    dev = torch.device("cuda")
    got = [to_device_async(t, dev) for t in noise]
    want = [
        seeded_normal(vs, 11, VIDEO_SALT).to(dev),
        seeded_normal(acts, 11, ACTION_SALT).to(dev),
    ]
    torch.cuda.synchronize()
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g.view(torch.int32), w.view(torch.int32))


def test_take_needs_the_key_and_serves_one_call() -> None:
    vs, acts = DROID[1]
    cache = NoisePrefetch()
    cache.start(100, vs, acts)
    # Another seed or shape: the draw is dropped, the call draws itself.
    assert cache.take(101, vs, acts) is None
    assert cache.take(100, vs, acts) is None
    cache.start(100, vs, acts)
    assert cache.take(100, DROID[2][0], acts) is None
    cache.start(100, vs, acts)
    assert cache.take(100, vs, DROID[2][1]) is None
    cache.start(100, vs, acts)
    _settle(cache)
    noise = cache.take(100, vs, acts)
    assert noise is not None
    assert torch.equal(noise[0], seeded_normal(vs, 100, VIDEO_SALT))
    assert torch.equal(noise[1], seeded_normal(acts, 100, ACTION_SALT))
    # A draw serves one call.
    assert cache.take(100, vs, acts) is None
    assert (cache.hits, cache.misses) == (1, 5)
    # A newer start replaces the older draw.
    cache.start(5, vs, acts)
    cache.start(6, vs, acts)
    assert cache.take(5, vs, acts) is None
    cache.start(6, vs, acts)
    _settle(cache)
    noise = cache.take(6, vs, acts)
    assert noise is not None and torch.equal(noise[0], seeded_normal(vs, 6, VIDEO_SALT))


def test_a_queued_draw_is_cancelled_not_waited_for() -> None:
    """A draw that the worker has not started gives None at once."""
    release = threading.Event()
    blocker = np_mod._EXECUTOR.submit(release.wait, 30)
    try:
        cache = NoisePrefetch()
        vs, acts = DROID[0]
        cache.start(3, vs, acts)
        future = cache.future
        assert cache.take(3, vs, acts) is None
        assert future is not None and future.cancelled()
    finally:
        release.set()
        blocker.result(timeout=30)


def test_take_waits_for_a_running_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draw that the worker is running when the call takes it is waited for."""
    started, release = threading.Event(), threading.Event()
    real = np_mod.draw_window_noise

    def slow(*args: object) -> tuple[Tensor, Tensor]:
        started.set()
        assert release.wait(30)
        return real(*args)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(np_mod, "draw_window_noise", slow)
    cache = NoisePrefetch()
    vs, acts = DROID[1]
    cache.start(21, vs, acts)
    assert started.wait(30)
    future = cache.future
    assert future is not None and future.running() and not future.done()
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        noise = cache.take(21, vs, acts)
    finally:
        release.set()
        timer.join()
    assert noise is not None and (cache.hits, cache.misses) == (1, 0)
    assert _sha(noise[0]) == _sha(seeded_normal(vs, 21, VIDEO_SALT))
    assert _sha(noise[1]) == _sha(seeded_normal(acts, 21, ACTION_SALT))


def test_a_worker_error_gives_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("no pinned memory")

    monkeypatch.setattr(np_mod, "draw_window_noise", fail)
    cache = NoisePrefetch()
    vs, acts = DROID[0]
    cache.start(3, vs, acts)
    _settle(cache)
    assert cache.take(3, vs, acts) is None
    assert cache.misses == 1


def test_the_draw_does_not_touch_the_global_generator() -> None:
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    cache = NoisePrefetch()
    vs, acts = DROID[1]
    cache.start(9, vs, acts)
    _settle(cache)
    assert cache.take(9, vs, acts) is not None
    assert torch.equal(torch.get_rng_state(), state)


# ---------------------------------------------------------------------------
# A session call sequence with and without the prefetch
# ---------------------------------------------------------------------------


@dataclass
class RigState:
    """The fields of a rig session that the prefetch reads."""

    max_context_chunks: int
    stateless: bool = False


class Boom(RuntimeError):
    pass


# (chunk_index, depth, event): the event happens before the call.
SCRIPT: list[tuple[int, int, str]] = [
    (0, 0, ""),
    (1, 1, ""),
    (2, 2, ""),
    (3, 2, ""),
    (0, 0, "reset"),
    (1, 1, ""),
    (2, 2, "fail"),  # the call raises after it took its noise
    (2, 2, ""),  # the retry of the failed call
    (3, 2, ""),
    (4, 2, "seed"),  # the session seed changes
    (5, 0, ""),  # a restart: depth 0 in the middle of the episode
    (6, 1, ""),
    (7, 2, ""),
]
# The calls of SCRIPT whose noise was drawn ahead of the call (the failed
# call took its draw; its retry draws itself).
HITS = {1, 2, 3, 5, 6, 8, 11, 12}


def _run_script(monkeypatch: pytest.MonkeyPatch, prefetch: bool) -> dict:
    calls = Calls()
    stub = _stub_expert(calls)
    fail = {"on": False}

    def expert(*args: object, **kw: object) -> Tensor:
        if fail["on"]:
            raise Boom("expert failed")
        return stub(*args, **kw)

    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", expert)
    copied: list[Tensor] = []

    def record(t: Tensor, device: torch.device) -> Tensor:
        # The noise of a call is the first two tensors that it copies.
        copied.append(t.clone())
        return t.to(device)

    monkeypatch.setattr(model_mod, "to_device_async", record)
    policy = _policy(calls)
    settings = SamplerSettings(layout="window_joint", num_steps=2, cfg_video_scale=4.0)
    by_depth = {d: _inputs(d) for d in range(3)}

    @dataclass
    class Pipe:
        policy: PraxisPolicy

        def encode_text(self, prompt: str) -> Tensor:
            return by_depth[0]["text"]

        def encode_uncond_text(self) -> Tensor:
            return by_depth[0]["text_uncond"]

    session = object.__new__(Session)
    session.pipe = cast(PraxisPipeline, Pipe(policy))
    session.sampler = settings
    session.seed = 40
    session.skip_last_video = False
    session.ctx_cache = True
    session.overlap_expert = False
    session.state = cast(object, RigState(max_context_chunks=2))
    session.noise_prefetch = prefetch
    noise: list[tuple[Tensor, Tensor]] = []
    outputs: list[tuple[Tensor | None, Tensor] | None] = []
    seeds: list[int] = []
    for ci, depth, event in SCRIPT:
        if event == "seed":
            session.seed = 977
        inp = by_depth[depth]
        chunk = StubInputs(
            prompt="p",
            anchor=inp["anchor"],
            cleans=inp["cleans"],
            proprio=torch.zeros(1, 1, P_DIM),
            proprio_all=inp["proprio_all"],
            chunk_index=ci,
        )
        copied.clear()
        fail["on"] = event == "fail"
        seeds.append(session.seed + ci)
        try:
            video, actions, _ = session._sample(cast(SingleArmChunkInputs, chunk))
            outputs.append((video, actions))
        except Boom:
            outputs.append(None)
        _settle(getattr(session, "_noise", None))
        noise.append((copied[0], copied[1]))
    cache = getattr(session, "_noise", None)
    return {
        "noise": noise,
        "outputs": outputs,
        "seeds": seeds,
        "hits": None if cache is None else cache.hits,
        "misses": None if cache is None else cache.misses,
    }


def test_a_session_gives_the_noise_and_outputs_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _run_script(monkeypatch, prefetch=False)
    fast = _run_script(monkeypatch, prefetch=True)
    assert base["hits"] is None
    assert fast["hits"] == len(HITS)
    assert fast["misses"] == len(SCRIPT) - len(HITS)
    assert fast["seeds"] == base["seeds"]
    for i, (ci, depth, _event) in enumerate(SCRIPT):
        seed = base["seeds"][i]
        kk = depth + 1
        want_v = seeded_normal((1, 192, kk * LAT, 2, 3), seed, VIDEO_SALT)
        want_a = seeded_normal((kk * LAT, A_DIM), seed, ACTION_SALT)
        for run in (base, fast):
            got_v, got_a = run["noise"][i]
            assert _sha(got_v) == _sha(want_v), (i, ci)
            assert _sha(got_a) == _sha(want_a), (i, ci)
        b, f = base["outputs"][i], fast["outputs"][i]
        assert (b is None) == (f is None) == (SCRIPT[i][2] == "fail")
        if b is not None and f is not None:
            assert b[0] is not None and f[0] is not None
            assert _sha(b[0]) == _sha(f[0]) and _sha(b[1]) == _sha(f[1])


def test_a_stateless_session_prefetches_depth_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(0)

    @dataclass
    class Pipe:
        policy: PraxisPolicy

        def encode_text(self, prompt: str) -> Tensor:
            return inp["text"]

        def encode_uncond_text(self) -> Tensor:
            return inp["text_uncond"]

    session = object.__new__(Session)
    session.pipe = cast(PraxisPipeline, Pipe(policy))
    session.sampler = SamplerSettings(layout="window_joint", num_steps=2)
    session.seed = 0
    session.skip_last_video = False
    session.ctx_cache = True
    session.overlap_expert = False
    session.state = cast(object, RigState(max_context_chunks=0, stateless=True))
    session.noise_prefetch = True
    for ci in range(3):
        chunk = StubInputs(
            prompt="p",
            anchor=inp["anchor"],
            cleans=[],
            proprio=torch.zeros(1, 1, P_DIM),
            proprio_all=inp["proprio_all"],
            chunk_index=ci,
        )
        session._sample(cast(SingleArmChunkInputs, chunk))
        _settle(session._noise)
    assert session._noise is not None and session._noise.hits == 2
    # Turning the flag off drops the draw.
    session.noise_prefetch = False
    assert session._noise_prefetch() is None and session._noise is None


def test_sessions_built_without_init_keep_the_prefetch_off() -> None:
    assert Session.noise_prefetch is False


# ---------------------------------------------------------------------------
# Real sessions (Session.__init__) and the benchmark switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device", ["cpu", torch.device("cpu"), "cuda", torch.device("cuda")]
)
@pytest.mark.parametrize("env", [None, "0", "1"])
def test_a_real_session_reads_the_device_and_the_env(
    monkeypatch: pytest.MonkeyPatch, device: str | torch.device, env: str | None
) -> None:
    """On by default for a CUDA device; PRAXIS_NOISE_PREFETCH=0 turns it off.

    The device may be a string (as in stand-in pipelines) or a torch.device.
    No CUDA device is needed: the session only reads the device type.
    """
    if env is None:
        monkeypatch.delenv("PRAXIS_NOISE_PREFETCH", raising=False)
    else:
        monkeypatch.setenv("PRAXIS_NOISE_PREFETCH", env)
    want = torch.device(device).type == "cuda" and env != "0"
    pipe = _pipe(4.0)
    pipe.device = device  # ty: ignore[invalid-assignment]
    session = Session(pipe, max_context_chunks=1)
    assert session.noise_prefetch is want
    assert noise_prefetch_default(device) is want
    assert session._noise is None


def _episode(
    monkeypatch: pytest.MonkeyPatch, prefetch: bool
) -> tuple[list[tuple], NoisePrefetch | None]:
    """Run calls 0-2, a real episode reset (``state.reset``), then calls 0-1."""
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(Calls()))
    pipe = _pipe(4.0)
    session = pipe.session(prompt="p", seed=50, max_context_chunks=1)
    session.noise_prefetch = prefetch
    outputs = []
    for i, k in enumerate([0, 1, 2, None, 0, 1]):
        if k is None:
            session.state.reset()
            continue
        res = session.step(synthetic_obs(i, 1 if k == 0 else 25, 234, 416))
        _settle(getattr(session, "_noise", None))
        outputs.append(
            (
                res.seed,
                res.depth,
                _sha(res.pred_action_norm),
                None if res.pred_video_latent is None else _sha(res.pred_video_latent),
                res.commands,
            )
        )
    return outputs, getattr(session, "_noise", None)


def test_a_real_reset_keeps_the_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    base, off = _episode(monkeypatch, prefetch=False)
    fast, cache = _episode(monkeypatch, prefetch=True)
    assert off is None and cache is not None
    assert [(o[0], o[1]) for o in fast] == [(50, 0), (51, 1), (52, 1), (50, 0), (51, 1)]
    assert repr(fast) == repr(base)
    # The first call and the first call after the reset (seed 50, not the
    # drawn 53) draw themselves; the other calls take the draw.
    assert (cache.hits, cache.misses) == (3, 2)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (None, None),
        (True, True),
        (False, False),
        ("on", True),
        ("off", False),
        (1, True),
        (0, False),
        ("yes", True),
        ("no", False),
    ],
)
def test_the_benchmark_switch_is_on_or_off(value: object, want: bool | None) -> None:
    session = {} if value is None else {"noise_prefetch": value}
    assert _bench().noise_prefetch_setting(session) is want


@pytest.mark.parametrize("value", ["auto", "default", "", "maybe", 2])
def test_the_benchmark_switch_refuses_other_values(value: object) -> None:
    with pytest.raises(SystemExit, match="noise_prefetch"):
        _bench().noise_prefetch_setting({"noise_prefetch": value})


@pytest.mark.parametrize("env", ["0", "1"])
def test_the_benchmark_keeps_the_env_switch(
    monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    """The benchmark reports and keeps the session default of the env."""
    monkeypatch.setenv("PRAXIS_NOISE_PREFETCH", env)
    from praxis import layers

    monkeypatch.setattr(layers, "FUSED_ROPE", layers.FUSED_ROPE)
    monkeypatch.setattr(layers, "CACHE_DEVICE_TABLES", layers.CACHE_DEVICE_TABLES)
    pipe = _pipe(4.0)
    pipe.device = torch.device("cuda")
    _bench().configure_optimizations(pipe, True)
    from praxis.pipeline import noise_prefetch_default

    assert noise_prefetch_default(pipe.device) is (env == "1")
    assert pipe.session(max_context_chunks=1).noise_prefetch is (env == "1")


def test_the_benchmark_counts_the_draws_and_keeps_the_bits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(Calls()))
    bench = _bench()
    pipe = _pipe(4.0)
    base = {"max_context_chunks": 1, "exec_horizon": 24, "skip_video_decoder": "off"}
    counts = {}
    for value in ("off", "on"):
        counts[value] = {"hits": 0, "misses": 0}
        bench.run(
            pipe,
            {**base, "noise_prefetch": value},
            2,
            3,
            3,
            save_outputs=tmp_path / value,
            noise_counts=counts[value],
        )
    assert counts["off"] == {"hits": 0, "misses": 0}
    # Three sessions (two staged, one plain) of three calls: the first call
    # of each session draws itself.
    assert counts["on"] == {"hits": 6, "misses": 3}
    names = sorted(p.name for p in (tmp_path / "on").glob("*.pt"))
    assert names and names == sorted(p.name for p in (tmp_path / "off").glob("*.pt"))
    for name in names:
        got = torch.load(tmp_path / "on" / name, weights_only=False)
        want = torch.load(tmp_path / "off" / name, weights_only=False)
        for key in ("actions", "video"):
            assert torch.equal(got[key], want[key])
        assert repr(got["commands"]) == repr(want["commands"])
