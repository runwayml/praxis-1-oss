"""CPU tests of the offline action error (``praxis/eval/offline_action_error.py``)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_sampler_window_joint import (
    LAT,
    Calls,
    _inputs,
    _policy,
    _reference_loop,
    _stub_expert,
)

import praxis.model as model_mod
from praxis.eval import offline_action_error as ae
from praxis.model import SamplerSettings
from praxis.robot_io import pack_action_tokens, q99_normalize
from praxis.sampler import seeded_normal, window_time_grid

DATA = Path(__file__).parent / "data"
CONFIG = json.loads((DATA / "praxis-1-droid.json").read_text())
# praxis.train.data (the windows of the fine-tune data) needs the `train` extra.
needs_train = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("av", "pyarrow")),
    reason="needs the `train` extra",
)


def _rel_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """Plausible relative rows: small xyz, a rotation near identity, a grip."""
    rows = np.zeros((n, 10), dtype=np.float32)
    rows[:, :3] = rng.normal(0, 0.03, (n, 3))
    rows[:, 3:9] = [1, 0, 0, 0, 1, 0]
    rows[:, 3:9] += rng.normal(0, 0.01, (n, 6))
    rows[:, 9] = rng.uniform(0, 1, n)
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_chunk_metrics_values() -> None:
    gt = np.zeros((24, 10), dtype=np.float32)
    gt[:, 3:9] = [1, 0, 0, 0, 1, 0]
    gt[-1, :3] = [0.03, 0.04, 0.0]  # 5 cm of motion
    pred = gt.copy()
    pred[-1, :3] = [0.03, 0.04, 0.02]  # 2 cm off in z
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    pred[-1, 3:9] = [c, s, 0, -s, c, 0]  # 30 deg about z
    pred[:, 9] = 0.25
    m = ae.chunk_metrics(pred, gt)
    assert m["endpoint_err_cm"] == pytest.approx(2.0, abs=1e-5)
    assert m["gt_motion_cm"] == pytest.approx(5.0, abs=1e-5)
    assert m["rot_endpoint_deg"] == pytest.approx(30.0, abs=1e-4)
    assert m["grip_mae"] == pytest.approx(0.25)
    assert ae.chunk_metrics(gt, gt)["endpoint_err_cm"] == 0.0


def test_degenerate_rotation_is_nan() -> None:
    good = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    assert math.isnan(ae.geodesic_deg(np.zeros(6), good))
    assert math.isnan(ae.geodesic_deg(np.array([1, 0, 0, 2, 0, 0.0]), good))
    assert ae.geodesic_deg(good, good) == pytest.approx(0.0, abs=1e-6)


def test_window_metrics_keys_and_summary() -> None:
    rng = np.random.default_rng(0)
    gt = ae.split_chunks(_rel_rows(rng, 93), CONFIG)
    assert [len(c) for c in gt] == [21, 24, 24, 24]
    records = []
    for w in range(3):
        for leg, scale in (("oracle", 0.001), ("deployment", 0.01)):
            pred = [c + rng.normal(0, scale, c.shape).astype(np.float32) for c in gt]
            records.append({"window": w, "leg": leg, **ae.window_metrics(pred, gt)})
    keys = {f"{m}_chunk{k}" for m in ae.CHUNK_METRICS for k in range(4)}
    assert keys <= set(records[0])
    s = ae.summarize(records)
    row = [r["endpoint_err_cm_chunk2"] for r in records if r["leg"] == "oracle"]
    assert s["oracle/endpoint_err_cm_chunk2"]["mean"] == pytest.approx(np.mean(row))
    assert s["oracle/endpoint_err_cm_chunk2"]["sem"] == pytest.approx(
        np.std(row, ddof=1) / math.sqrt(3)
    )
    later = [
        np.mean([r[f"endpoint_err_cm_chunk{k}"] for k in (1, 2, 3)])
        for r in records
        if r["leg"] == "deployment"
    ]
    got = s["deployment/endpoint_err_cm_chunks123_mean"]
    assert got["mean"] == pytest.approx(np.mean(later)) and got["n"] == 3
    assert (
        s["deployment/endpoint_err_cm_chunks123_mean"]["mean"]
        > s["oracle/endpoint_err_cm_chunks123_mean"]["mean"]
    )


def test_decode_chunks_inverts_the_training_codec() -> None:
    """Tokens of the training codec decode to the rows (inside the q99 range)."""
    rng = np.random.default_rng(1)
    rows = _rel_rows(rng, 93)
    norm = CONFIG["normalization"]
    q01 = np.asarray(norm["action_q01"], np.float32)
    q99 = np.asarray(norm["action_q99"], np.float32)
    dims = np.asarray(norm["q99_normed_dims"])
    rows[:, dims] = np.clip(rows[:, dims], q01[dims], q99[dims])
    n = q99_normalize(rows, q01, q99, dims)
    tokens = np.concatenate(
        [
            pack_action_tokens(c, first_chunk=k == 0)
            for k, c in enumerate(ae.split_chunks(n, CONFIG))
        ]
    )
    got = ae.decode_chunks(torch.from_numpy(tokens)[None], CONFIG)
    want = ae.split_chunks(rows, CONFIG)
    assert [len(g) for g in got] == [21, 24, 24, 24]
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, atol=1e-6)


def test_decode_keeps_the_first_row_of_the_anchor_token() -> None:
    tokens = torch.zeros(1, 24, 40)
    tokens[0, 0, 9::10] = torch.tensor([-0.5, 0.1, 0.2, 0.3])  # grip of 4 sub-rows
    tokens[0, 1, 9::10] = 0.9
    got = ae.decode_chunks(tokens, CONFIG)[0][:, 9]
    assert len(got) == 21
    assert got[0] == pytest.approx(0.25) and got[1] == pytest.approx(0.95)


def test_decode_does_not_clip_the_prediction() -> None:
    tokens = torch.zeros(1, 24, 40)
    tokens[0, 6:, 0] = 1.5  # x of chunks 1-3, outside [-1, 1]
    got = ae.decode_chunks(tokens, CONFIG)
    q01, q99 = (
        CONFIG["normalization"]["action_q01"][0],
        CONFIG["normalization"]["action_q99"][0],
    )
    assert got[1][0, 0] == pytest.approx((2.5 / 2) * (q99 - q01) + q01, rel=1e-6)
    assert got[1][0, 0] > q99


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@needs_train
def test_windows_file() -> None:
    spec = ae.load_windows_file()
    assert spec["source"] == "cadene/droid_1.0.1"
    assert len(spec["episodes"]) == len(set(spec["episodes"])) == 19
    assert spec["episodes"] == sorted(spec["episodes"])
    # Every episode is in the val split of the fine-tune data.
    from praxis.train.data import VAL_EVERY

    assert all(e % VAL_EVERY == 0 for e in spec["episodes"])


@pytest.mark.parametrize("n_frames", [102, 124, 135, 200, 536, 629])
@pytest.mark.parametrize("n", [4, 7])
@needs_train
def test_stratified_anchors_spread_and_reproduce(n_frames: int, n: int) -> None:
    a = ae.stratified_anchors(n_frames, 93, 15.0, n, ae.episode_rng(0, 7))
    assert a == ae.stratified_anchors(n_frames, 93, 15.0, n, ae.episode_rng(0, 7))
    assert len(a) == n and a == sorted(a)
    assert all(0 <= x <= n_frames - 93 for x in a)
    last = n_frames - 93
    if last >= 4 * n:  # each part holds its own frames
        for j, x in enumerate(a):
            assert j * last / n - 1 <= x <= (j + 1) * last / n + 1
    other = ae.stratified_anchors(n_frames, 93, 15.0, n, ae.episode_rng(1, 7))
    if last >= 4 * n:
        assert other != a


@needs_train
def test_stratified_anchors_refusals() -> None:
    with pytest.raises(ValueError, match="n_windows"):
        ae.stratified_anchors(200, 93, 15.0, 0, ae.episode_rng(0, 0))
    with pytest.raises(ValueError, match="do not hold"):
        ae.stratified_anchors(95, 93, 15.0, 2, ae.episode_rng(0, 0))


def test_window_seed() -> None:
    assert ae.window_seed(0, 5) == 5
    assert ae.window_seed(2, 31) == 2031
    assert ae.window_seed(0, ae.SEED_STRIDE - 1) == ae.SEED_STRIDE - 1
    for window in (-1, ae.SEED_STRIDE):
        with pytest.raises(ValueError, match="outside"):
            ae.window_seed(1, window)


# ---------------------------------------------------------------------------
# The oracle leg of the sampler
# ---------------------------------------------------------------------------


def _oracle_reference(inp: dict, x0: torch.Tensor, seed: int, steps: int) -> tuple:
    """The oracle loop written out: the video state is re-pinned to the noised
    clean video before each step, and only the actions integrate."""
    kk = len(inp["cleans"]) + 1
    calls = Calls()
    run = _stub_expert(calls)
    from test_sampler_window_joint import MD, StubBackbone

    backbone = StubBackbone(calls)
    clean = torch.cat(inp["cleans"], 2) if inp["cleans"] else None
    noise_v = seeded_normal(tuple(x0.shape), seed, 0)
    xa = seeded_normal((kk * LAT, 40), seed, 1)
    ts = window_time_grid(steps, 1.0)
    for step in range(steps):
        t_cur, t_next = ts[step], ts[step + 1]
        xv = (1 - t_cur) * x0.float() + t_cur * noise_v
        t = torch.full((1, kk), float(t_cur), dtype=MD)
        out = backbone.forward_chunk_causal(
            inp["text"], inp["anchor"], clean, xv.to(MD), t
        )
        tp = out.taps
        a = run(
            None,
            tp.taps,
            tp.embeddings,
            tp.rope_ids,
            noised_actions=xa[None],
            action_t_per_chunk=torch.full((kk,), float(t_cur))[None],
            proprio_per_chunk=inp["proprio_all"],
        ).float()[0]
        xa = xa + (t_next - t_cur) * a
    return calls, xa


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("depth", [0, 3])
def test_oracle_video_pins_the_video(
    monkeypatch: pytest.MonkeyPatch, depth: int, dtype: torch.dtype
) -> None:
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(depth)
    kk = depth + 1
    g = torch.Generator().manual_seed(99)
    x0 = torch.randn(1, 192, kk * LAT, 2, 3, generator=g).to(dtype)
    settings = SamplerSettings(num_steps=5, t_shift=1.0, cfg_video_scale=1.0)
    video, actions = policy.sample_window_joint(
        seed=7, settings=settings, oracle_video=x0, return_window=True, **inp
    )
    ref_calls, want_a = _oracle_reference(inp, x0, seed=7, steps=5)
    assert torch.equal(actions[0], want_a)
    assert torch.equal(video, x0.float())
    for got, want in zip(calls.trunk, ref_calls.trunk, strict=True):
        assert torch.equal(got["noised"], want["noised"])
    # The pinned state differs from the joint run.
    calls2 = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls2))
    _, joint = _policy(calls2).sample_window_joint(
        seed=7, settings=settings, return_window=True, **inp
    )
    assert not torch.equal(joint, actions)


def test_window_return_holds_the_served_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """``return_window`` returns all chunks; its last chunk is the default output."""
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(2)
    settings = SamplerSettings(num_steps=2, t_shift=8.0, cfg_video_scale=1.0)
    v_last, a_last = policy.sample_window_joint(seed=3, settings=settings, **inp)
    v_all, a_all = policy.sample_window_joint(
        seed=3, settings=settings, return_window=True, **inp
    )
    assert torch.equal(v_all[:, :, 2 * LAT :], v_last)
    assert torch.equal(a_all[:, 2 * LAT :], a_last)
    want_v, want_a = _reference_loop(inp, 3, 2, 8.0, 1.0, 1.0)
    assert torch.equal(v_last, want_v) and torch.equal(a_last, want_a)


def test_oracle_video_shape_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    inp = _inputs(1)
    with pytest.raises(ValueError, match="oracle_video"):
        _policy(calls).sample_window_joint(
            seed=0,
            settings=SamplerSettings(cfg_video_scale=1.0),
            oracle_video=torch.zeros(1, 192, LAT, 2, 3),
            **inp,
        )


# ---------------------------------------------------------------------------
# Parity with the original implementation
# ---------------------------------------------------------------------------


def test_metrics_equal_the_original_implementation() -> None:
    """The per-window values of the original implementation.

    ``action_error_reference.npz`` holds the recorded rows of 4 of the val
    windows (frame 24 of the episodes ``episodes`` of the JSON file, not
    clipped), built by the window builder of the original implementation, and per
    leg a predicted ``(93, 10)`` array: the recorded rows plus seeded noise
    (the leg sets its scale), so that each metric has a nonzero value.
    ``action_error_reference.json`` holds the values that the metric function
    of the original implementation computed from them:

    * bitwise equal: ``endpoint_err_cm`` and ``gt_motion_cm`` of chunks 0 to
      3, ``rot_endpoint_deg`` of chunk 0;
    * within 1e-7: the gripper error over the 93 rows (``grip_mae_rows``),
      the row-weighted mean of ``grip_mae`` of the 4 chunks. The original
      implementation computed it in one float32 mean over the rows.

    The original implementation did not compute ``rot_endpoint_deg`` of chunks 1 to
    3 or ``grip_mae`` per chunk. ``test_chunk_metrics_values`` covers them.
    """
    ref = json.loads((DATA / "action_error_reference.json").read_text())
    with np.load(DATA / "action_error_reference.npz") as z:
        arrays = {k: z[k] for k in z.files}
    for w in ref["windows"]:
        gt = ae.split_chunks(arrays[f"w{w:02d}_gt_raw"], CONFIG)
        for leg in ae.LEGS:
            pred = ae.split_chunks(arrays[f"w{w:02d}_{leg}_pred"], CONFIG)
            got = ae.window_metrics(pred, gt)
            want = ref["values"][f"w{w:02d}_{leg}"]
            for name in ("endpoint_err_cm", "gt_motion_cm"):
                for k in range(4):
                    key = f"{name}_chunk{k}"
                    assert got[key] == want[key], (w, leg, key)
            assert (
                got["rot_endpoint_deg_chunk0"] == (want["rot_endpoint_deg_chunk0"])
            ), (w, leg)
            rows = [len(p) for p in pred]
            grip = sum(got[f"grip_mae_chunk{k}"] * n for k, n in enumerate(rows))
            assert grip / sum(rows) == pytest.approx(want["grip_mae_rows"], abs=1e-7)


# ---------------------------------------------------------------------------
# The sampler of the deployment leg
# ---------------------------------------------------------------------------


def test_serve_sampler_from_the_serve_config() -> None:
    """``from_config`` of the serve YAML gives the recommended serve recipe.

    The policy server builds its sampler with the same ``from_config``.
    """
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "serve_droid.yaml").read_text())
    got = SamplerSettings.from_config(cfg["sampler"])
    assert got == SamplerSettings()
    assert (got.num_steps, got.t_shift, got.cfg_scale, got.cfg_video_scale) == (
        2,
        12.5,
        1.0,
        4.0,
    )
    assert got.guided and not ae.EVAL_SAMPLER.guided


def test_sample_leg_with_the_guided_serve_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guided sampler runs with the blank caption and refuses without it."""
    from praxis.train.objective import EncodedWindow

    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    policy = _policy(calls)
    inp = _inputs(3)
    g = torch.Generator().manual_seed(5)
    last = torch.randn(1, *inp["anchor"].shape[1:2], LAT, 2, 3, generator=g)
    window = EncodedWindow(
        text=inp["text"],
        anchor=inp["anchor"],
        chunks=[*inp["cleans"], last.to(inp["anchor"].dtype)],
        actions=torch.zeros(1, 4 * LAT, 40),
        proprio=inp["proprio_all"],
    )
    settings = SamplerSettings()
    with pytest.raises(ValueError, match="blank caption"):
        ae.sample_leg(policy, window, "deployment", 3, settings)
    got = ae.sample_leg(
        policy, window, "deployment", 3, settings, text_uncond=inp["text_uncond"]
    )
    _, want = policy.sample_window_joint(
        seed=3, settings=settings, return_window=True, **inp
    )
    assert torch.equal(got, want)
    _, unguided = policy.sample_window_joint(
        seed=3, settings=ae.EVAL_SAMPLER, return_window=True, **inp
    )
    assert not torch.equal(got, unguided)
