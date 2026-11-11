"""Tests of the stage timers and the local benchmark (CPU only, no weights).

A stand-in pipeline runs the real session and the real window sampler with
stub networks, so the timers see the calls of a real policy call.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_sampler_window_joint import Calls, StubBackbone, _stub_expert

from praxis import profiling
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.pipeline import PraxisPipeline, Session, StepResult
from praxis.robot_io import NormStats

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "tests" / "data" / "praxis-1-droid.json").read_text())


def _script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_inference", ROOT / "scripts" / "benchmark_inference.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize() -> None:
    s = profiling.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s == {"n": 5, "p50": 3.0, "p90": pytest.approx(4.6), "max": 5.0}
    assert profiling.summarize([]) == {"n": 0}
    out = profiling.summarize_calls([{"a": 1.0}, {"a": 3.0, "b": 2.0}], ["a", "b"])
    assert out == {
        "a": {"n": 2, "p50": 2.0, "p90": pytest.approx(2.8), "max": 3.0},
        "b": {"n": 1, "p50": 2.0, "p90": 2.0, "max": 2.0},
    }


def test_stage_timer_adds_the_stages_of_one_call() -> None:
    syncs: list[int] = []
    timer = profiling.StageTimer(sync=lambda: syncs.append(1))
    double = timer.wrap("backbone", lambda x: 2 * x)
    assert double(3) == 6 and timer.calls == [] and syncs == []  # outside a call
    with timer.call() as rec:
        assert double(1) == 2 and double(2) == 4
    (call,) = timer.calls
    assert call is not rec and call["backbone_calls"] == 2
    assert call["text_encode"] == call["vae_encode"] == call["expert"] == 0.0
    assert call["total"] >= call["backbone"] >= 0.0
    assert len(syncs) == 2 + 2 * 2  # the call, and each wrapped call


def test_each_timed_stage_is_an_nvtx_range() -> None:
    ranges: list[str] = []

    @contextmanager
    def nvtx(name: str) -> Iterator[None]:
        ranges.append(name)
        yield

    timer = profiling.StageTimer(sync=lambda: None, nvtx=nvtx)
    expert = timer.wrap("expert", lambda x: x + 1)
    assert expert(1) == 2 and ranges == []  # outside a call
    with timer.call():
        assert expert(1) == 2
    assert ranges == ["expert"]


def _pipe(cfg_video_scale: float) -> PraxisPipeline:
    """A pipeline with the real session and sampler, and stub networks."""
    calls = Calls()
    policy = PraxisPolicy(
        StubBackbone(calls),  # ty: ignore[invalid-argument-type]
        torch.nn.Module(),  # ty: ignore[invalid-argument-type]
        CONFIG,
    )

    def encode_video(clip: torch.Tensor) -> torch.Tensor:
        t = clip.shape[2]
        idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
        x = torch.nn.functional.avg_pool3d(clip[:, :, idx], kernel_size=(1, 112, 139))
        return x.repeat(1, 64, 1, 1, 1).to(torch.bfloat16)

    pipe = object.__new__(PraxisPipeline)
    pipe.device = torch.device("cpu")
    pipe.policy = policy
    pipe.vae = SimpleNamespace(encode=encode_video)  # ty: ignore[invalid-assignment]
    pipe.text = SimpleNamespace(  # ty: ignore[invalid-assignment]
        encode=lambda prompts: [torch.ones(1, 3, 2048, dtype=torch.bfloat16)]
    )
    pipe.sampler = SamplerSettings(cfg_video_scale=cfg_video_scale)
    pipe.embodiment = "single_arm"
    pipe.variant = profiling.model_module.model_variant(CONFIG)
    n = CONFIG["normalization"]
    pipe.stats = NormStats(
        *(
            torch.tensor(n[k]).numpy()
            for k in ("action_q01", "action_q99", "proprio_mean", "proprio_std")
        )
    )
    pipe._text_cache = {}
    return pipe


@pytest.fixture
def stub_expert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        profiling.model_module, "run_expert_on_taps_k", _stub_expert(Calls())
    )


@pytest.mark.parametrize(("cfg_video", "passes"), [(4.0, 2), (1.0, 1)])
def test_benchmark_times_each_stage_of_the_real_session(
    stub_expert: None, cfg_video: float, passes: int
) -> None:
    bench = _script()
    pipe = _pipe(cfg_video)
    session = {"max_context_chunks": 1, "exec_horizon": 24, "seed": 0}
    records = bench.run(pipe, session, sessions=2, calls=3, warmup_calls=3)
    # Two staged sessions (the first one is the warm-up), then the plain one.
    assert [
        (r["session"], r["call"], r["depth"], r["warmup"], r["plain"]) for r in records
    ] == [
        (0, 0, 0, True, False),
        (0, 1, 1, True, False),
        (0, 2, 1, True, False),
        (1, 0, 0, False, False),
        (1, 1, 1, False, False),
        (1, 2, 1, False, False),
        (2, 0, 0, False, True),
        (2, 1, 1, False, True),
        (2, 2, 1, False, True),
    ]
    assert not any(r["traced"] for r in records)
    assert [r["rows"] for r in records] == [21, 24, 24] * 3
    staged = [r for r in records if not r["plain"]]
    for r in staged:
        # num_steps 2: one backbone call per pass and step, but the last step
        # skips its video, so it runs no unguided pass (it only guides the
        # video), and an expert call for each backbone call at most (a sampler
        # may skip the expert of a pass whose actions it does not need).
        assert r["backbone_calls"] == passes + 1
        assert 2 <= r["expert_calls"] <= r["backbone_calls"]
        assert r["vae_encode_calls"] == (1 if r["depth"] == 0 else 3)
        assert r["total"] >= r["backbone"] + r["expert"] + r["vae_encode"]
    # The plain calls have no stage times.
    assert all(set(r) >= {"total"} and "backbone" not in r for r in records[6:])
    # The text cache holds the prompts after the first call of the run.
    assert records[0]["text_encode_calls"] == passes
    assert all(r["text_encode_calls"] == 0 for r in staged[1:])
    out = bench.summary(records)
    # total: the plain session; total_staged and the stages: the staged one.
    assert out["first"]["total"]["n"] == 1 and out["later"]["total"]["n"] == 2
    assert out["later"]["total_staged"]["n"] == out["later"]["backbone"]["n"] == 2
    assert "backbone" in bench.format_summary(out)
    s = bench.settings(pipe, session)
    assert s["cfg_mode"] == ("sequential" if passes == 2 else "off")
    assert s["passes_per_step"] == passes
    assert s["last_step_passes"] == 1
    # The wrappers are gone after the run.
    assert "forward_chunk_causal" not in vars(pipe.policy.backbone)
    assert "encode" in vars(pipe.vae)


@pytest.mark.parametrize(
    ("value", "skip"),
    [
        *[(None, True), ("auto", True), (True, True), ("on", True), (1, True)],
        *[(False, False), ("off", False), ("none", False), ("", False), (0, False)],
    ],
)
def test_benchmark_skips_the_last_video_decoder_as_the_server(
    stub_expert: None, monkeypatch: pytest.MonkeyPatch, value: object, skip: bool
) -> None:
    bench = _script()
    pipe = _pipe(1.0)
    session = {"max_context_chunks": 1, "exec_horizon": 24}
    if value is not None:
        session["skip_video_decoder"] = value
    results = []
    step = Session.step

    def recorded(self: Session, obs: dict) -> StepResult:
        results.append(step(self, obs))
        return results[-1]

    monkeypatch.setattr(Session, "step", recorded)
    bench.run(pipe, session, sessions=1, calls=2, warmup_calls=2)
    # A skipped call returns no predicted video.
    assert len(results) == 4
    assert all((r.pred_video_latent is None) is skip for r in results)
    trunk = pipe.policy.backbone.calls.trunk  # ty: ignore[unresolved-attribute]
    # num_steps 2, one pass per step: the second pass of each call is the last step.
    assert [c["decode"] for c in trunk] == [True, not skip] * 4
    assert bench.settings(pipe, session)["skip_video_decoder"] is skip


def test_the_trace_session_is_extra_and_not_counted(
    stub_expert: None, tmp_path: Path
) -> None:
    bench = _script()
    session = {"max_context_chunks": 1, "exec_horizon": 24}
    records = bench.run(_pipe(4.0), session, 2, 2, 2, trace_dir=tmp_path)
    assert [(r["session"], r["warmup"], r["plain"], r["traced"]) for r in records] == [
        (0, True, False, False),
        (0, True, False, False),
        (1, False, False, False),
        (1, False, False, False),
        (2, False, True, False),
        (2, False, True, False),
        (3, False, False, True),
        (3, False, False, True),
    ]
    out = bench.summary(records)
    assert out["later"]["total"]["n"] == out["later"]["total_staged"]["n"] == 1
    assert (tmp_path / "trace.json").stat().st_size > 0
    assert "backbone" in (tmp_path / "kernels.txt").read_text()


def test_benchmark_reads_the_serve_config() -> None:
    bench = _script()
    cfg = bench.load_config(str(ROOT / "configs" / "serve_droid.yaml"), [])
    calls: list[dict] = []

    class Pipe:
        def __init__(self, **kw: object) -> None:
            calls.append(kw)

    original = bench.build_pipeline.__globals__["PraxisPipeline"]
    bench.build_pipeline.__globals__["PraxisPipeline"] = Pipe
    try:
        bench.build_pipeline(cfg)
    finally:
        bench.build_pipeline.__globals__["PraxisPipeline"] = original
    (kw,) = calls
    assert kw["sampler"] == SamplerSettings(**cfg["sampler"])
    assert kw["checkpoint"] == "checkpoints/praxis-1-droid"


def test_benchmark_refuses_a_bimanual_checkpoint_before_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _script()
    single, bimanual = tmp_path / "single", tmp_path / "bimanual"
    for path, name in ((single, "droid"), (bimanual, "yam")):
        path.mkdir()
        config = (ROOT / "tests" / "data" / f"praxis-1-{name}.json").read_text()
        (path / "config.json").write_text(config)
    bench.check_single_arm(single)
    with pytest.raises(SystemExit, match="'bimanual' rig"):
        bench.check_single_arm(bimanual)

    def no_load(cfg: dict) -> None:
        raise AssertionError("the weights load before the embodiment check")

    monkeypatch.setattr(bench, "build_pipeline", no_load)
    argv = ["benchmark_inference.py", "--set", f"model.checkpoint={bimanual}"]
    argv += ["--out", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="'bimanual' rig"):
        bench.main()
    assert not (tmp_path / "out").exists()


def test_synthetic_obs_has_the_client_shapes() -> None:
    obs = profiling.synthetic_obs(2, 25, 234, 416)
    assert obs["image.exterior"].shape == (25, 234, 416, 3)
    assert obs["image.wrist"].dtype == np.uint8
    assert obs["robot_state.ee_pose"].shape == (25, 7)
    assert obs["grip"].shape == (25,)


def test_environment_reports_the_routed_precision() -> None:
    bench = _script()
    pipe = _pipe(1.0)
    assert bench.environment(pipe)["routed_precision"] == "bf16"
    pipe.policy.routed_quantization = {"format": "nvfp4"}
    assert bench.environment(pipe)["routed_precision"] == "nvfp4"
