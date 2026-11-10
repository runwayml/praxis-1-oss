"""CPU tests of the autoregressive eval (``praxis/eval/autoregressive``).

The fixtures come from the published results:

* ``autoregressive_verdicts.json``: the judge fields of the published runs
  (target, destination, action, and the DROID success time) per rollout.
  On DROID (104 rollouts) the baselines were given the second exterior view
  of the set (``frames_ext2/``). The scorer must give the published numbers
  from them.
* ``autoregressive_judge_requests.json``: for 37 rollouts of the three sets
  (every feature and level, and 3 time-to-success requests), the SHA-256 of
  the request that the published judge built on seeded synthetic frames.
  The judge of this package must build the same request, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_sampler_window_joint import (
    Calls,
    _inputs,
    _policy,
    _reference_loop,
    _stub_expert,
)

import praxis.model as model_mod
from praxis.eval.autoregressive import judge as J
from praxis.eval.autoregressive import score as S
from praxis.eval.autoregressive import sets as sets_mod
from praxis.model import SamplerSettings

DATA = Path(__file__).parent / "data"
VERDICTS = json.loads((DATA / "autoregressive_verdicts.json").read_text())["sets"]
REQUESTS = json.loads((DATA / "autoregressive_judge_requests.json").read_text())


def _verdicts(set_name: str, run: str) -> dict[str, dict]:
    return {
        rid: dict(zip(S.FIELDS, map(bool, row[:3]), strict=True))
        for rid, row in VERDICTS[set_name][run].items()
    }


def _tts(set_name: str, run: str) -> dict[str, dict]:
    return {
        rid: {"success_time_s": row[3]}
        for rid, row in VERDICTS[set_name][run].items()
        if len(row) > 3
    }


# ---------------------------------------------------------------------------
# Scores (parity check 1)
# ---------------------------------------------------------------------------

DROID_BASELINES = ("flux3_action", "dreamzero", "cosmos3_nano", "openwam_alpha")


@pytest.mark.parametrize(
    ("set_name", "run", "want"),
    [
        ("droid", "praxis-1-droid", (52.9, 38.6, 67.1)),
        ("droid", "flux3_action", (48.6, 35.0, 62.9)),
        ("droid", "dreamzero", (42.1, 30.0, 55.0)),
        ("droid", "cosmos3_nano", (22.9, 11.4, 35.7)),
        ("droid", "openwam_alpha", (1.4, 0.0, 4.3)),
    ],
)
def test_full_adherence_gives_the_published_numbers(
    set_name: str, run: str, want: tuple[float, float, float]
) -> None:
    lv = S.level(S.run_values(_verdicts(set_name, run)))
    assert (round(lv["mean"], 1), round(lv["lo"], 1), round(lv["hi"], 1)) == want


def test_droid_paired_deltas() -> None:
    runs = {r: _verdicts("droid", r) for r in ("praxis-1-droid", *DROID_BASELINES)}
    out = S.summarize(runs, reference="praxis-1-droid")["runs"]
    got = {
        r: (round(out[r]["paired_vs_praxis-1-droid"]["delta"], 1),
            out[r]["paired_vs_praxis-1-droid"]["p_perm"])
        for r in DROID_BASELINES
    }  # fmt: skip
    assert got == {
        "flux3_action": (-4.3, 0.6121),
        "dreamzero": (-10.7, 0.2205),
        "cosmos3_nano": (-30.0, 0.0002),
        "openwam_alpha": (-51.4, 0.0),
    }
    assert out["flux3_action"]["paired_vs_praxis-1-droid"]["n_pairs"] == 104


def test_droid_within_the_cutoff() -> None:
    names = ("praxis-1-droid", *DROID_BASELINES)
    runs = {r: _verdicts("droid", r) for r in names}
    tts = {r: _tts("droid", r) for r in names}
    times = {
        r: [tts[r][rid]["success_time_s"] for rid, v in runs[r].items()
            if S.full_adherence(v) and rid in tts[r]]
        for r in ("flux3_action", "dreamzero", "cosmos3_nano")
    }  # fmt: skip
    assert S.fastest_p95(times) == pytest.approx(S.DROID_CUTOFF_S)
    out = S.summarize(
        runs, reference="praxis-1-droid", tts=tts, cutoff_s=S.DROID_CUTOFF_S
    )["runs"]
    got = {r: tuple(round(out[r][k], 1) for k in ("mean", "lo", "hi")) for r in names}
    assert got == {
        "praxis-1-droid": (46.4, 31.4, 61.4),
        "flux3_action": (42.1, 27.9, 57.1),
        "dreamzero": (37.9, 25.7, 51.4),
        "cosmos3_nano": (17.9, 7.1, 30.7),
        "openwam_alpha": (0.0, 0.0, 0.0),
    }
    paired = {
        r: (round(out[r]["paired_vs_praxis-1-droid"]["delta"], 1),
            out[r]["paired_vs_praxis-1-droid"]["p_perm"])
        for r in DROID_BASELINES
    }  # fmt: skip
    assert paired == {
        "flux3_action": (-4.3, 0.641),
        "dreamzero": (-8.6, 0.3734),
        "cosmos3_nano": (-28.6, 0.0017),
        "openwam_alpha": (-46.4, 0.0),
    }


def test_summarize_refuses_an_incomplete_run() -> None:
    v = _verdicts("droid", "praxis-1-droid")
    expected = set(v)
    v.pop(next(iter(v)))
    with pytest.raises(ValueError, match="missing"):
        S.summarize({"run": v}, expected=expected)


def test_split_id() -> None:
    assert S.split_id("co_e0_AUTOLab_2023-10-27_2001__p0__s3") == (
        "co_e0_AUTOLab_2023-10-27_2001__p0",
        3,
    )
    with pytest.raises(ValueError):
        S.split_id("co_e0__s3")


# ---------------------------------------------------------------------------
# Judge requests (parity check 2)
# ---------------------------------------------------------------------------


def synthetic_frames(w: int, h: int, seed: int, n: int = 189) -> list[np.ndarray]:
    """The seeded frames of the request fixtures."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    out = []
    for t in range(n):
        base = np.stack(
            [
                (x * 255 // w + 3 * t) % 256,
                (y * 255 // h + 5 * t) % 256,
                (x + y + 7 * t) % 256,
            ],
            -1,
        )
        noise = rng.integers(-20, 21, (h, w, 3))
        out.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    return out


def _canonical(req: dict) -> str:
    m = req["messages"]
    d = {
        "system": m[0]["content"],
        "content": m[1]["content"],
        "model": req["model"],
        "max_completion_tokens": req["max_completion_tokens"],
        "temperature": req["temperature"],
        "response_format": req["response_format"],
    }
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _font_is_the_published_one() -> bool:
    p = Path(J.FONT_PATHS[0])
    return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == J.FONT_SHA256


@pytest.mark.skipif(
    not _font_is_the_published_one(),
    reason="the tile labels need DejaVuSans-Bold.ttf of fonts-dejavu-core 2.37",
)
@pytest.mark.parametrize(
    "case", REQUESTS, ids=lambda c: f"{c['set']}-{c['kind']}-{c['rollout']['id']}"
)
def test_judge_request_equals_the_published_one(case: dict) -> None:
    frames = synthetic_frames(case["width"], case["height"], case["frames_seed"])
    build = J.tts_request if case["kind"] == "tts" else J.request
    req = build(case["rollout"], case["scene"], case["set"], frames, 15.0)
    assert _canonical(req) == case["request_sha256"]


def test_judge_prompts_are_the_published_ones() -> None:
    def sha(name: str) -> str:
        return hashlib.sha256((J.HERE / name).read_bytes()).hexdigest()

    assert sha("judge_system_droid.txt").startswith("c11a6b2a5dfc")


class _Reply:
    def __init__(self, text: str) -> None:
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": text})()})()
        ]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()


class _Api:
    def __init__(self, fail: int) -> None:
        self.fail, self.calls = fail, []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw: object) -> _Reply:
        self.calls.append(kw)
        if len(self.calls) <= self.fail:
            raise RuntimeError("response_format json_schema is not supported")
        return _Reply('{"target_object_acted_on": true}')


def test_call_retries_and_falls_back_to_json_object() -> None:
    api = _Api(fail=1)
    req = {"model": "m", "messages": [], "response_format": {"type": "json_schema"}}
    out = J.call(req, api=api, sleep=lambda s: None)
    assert out["target_object_acted_on"] is True and out["_judge"]["attempts"] == 2
    assert api.calls[1]["response_format"] == {"type": "json_object"}
    assert req["response_format"] == {"type": "json_schema"}


def test_call_raises_after_the_retries() -> None:
    with pytest.raises(RuntimeError, match="judge failed"):
        J.call(
            {"model": "m", "messages": []},
            api=_Api(fail=9),
            retries=2,
            sleep=lambda s: None,
        )


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------


def _fake_set(tmp: Path) -> sets_mod.SetSpec:
    """A one-scene, one-rollout set folder and its spec."""
    (tmp / "frames").mkdir(parents=True)
    Image.fromarray(np.full((4, 8, 3), 7, np.uint8)).save(tmp / "frames" / "a.png")
    png = sets_mod.sha256_file(tmp / "frames" / "a.png")
    (tmp / "frames_ext2").mkdir()
    Image.fromarray(np.full((4, 4, 3), 5, np.uint8)).save(tmp / "frames_ext2" / "a.png")
    ext2 = sets_mod.sha256_file(tmp / "frames_ext2" / "a.png")
    scene = {
        "scene": "a",
        "frame": "frames/a.png",
        "sha256": png,
        "frame_ext2": "frames_ext2/a.png",
        "sha256_ext2": ext2,
    }
    (tmp / "scenes.jsonl").write_text(json.dumps(scene) + "\n")
    (tmp / "rollouts.jsonl").write_text(
        json.dumps({"id": "a__p0__s0", "scene": "a"}) + "\n"
    )
    return sets_mod.SetSpec(
        name="fake",
        rig="single_arm",
        views=("exterior", "wrist"),
        judge_views=("exterior", "wrist"),
        tile_wh=(4, 4),
        praxis_tile_wh=(4, 4),
        judge="droid",
        rollouts_sha256=sets_mod.sha256_file(tmp / "rollouts.jsonl"),
        scenes_sha256=sets_mod.sha256_file(tmp / "scenes.jsonl"),
        n_rollouts=1,
        n_scenes=1,
    )


def test_load_set_verifies_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _fake_set(tmp_path / "fake")
    monkeypatch.setitem(sets_mod.SETS, "fake", spec)
    es = sets_mod.load_set("fake", tmp_path / "fake")
    assert es.first_frame("a").shape == (4, 8, 3)
    second = es.second_exterior("a")
    assert second is not None and second.shape == (4, 4, 3)
    # A changed second exterior view is refused.
    Image.fromarray(np.full((4, 4, 3), 6, np.uint8)).save(
        tmp_path / "fake" / "frames_ext2" / "a.png"
    )
    with pytest.raises(sets_mod.SetError, match="SHA-256"):
        sets_mod.load_set("fake", tmp_path / "fake")
    shutil.rmtree(tmp_path / "fake")
    spec = _fake_set(tmp_path / "fake")
    monkeypatch.setitem(sets_mod.SETS, "fake", spec)
    # A changed frame is refused.
    Image.fromarray(np.full((4, 8, 3), 8, np.uint8)).save(
        tmp_path / "fake" / "frames" / "a.png"
    )
    with pytest.raises(sets_mod.SetError, match="SHA-256"):
        sets_mod.load_set("fake", tmp_path / "fake")
    # A changed rollout list is refused.
    shutil.rmtree(tmp_path / "fake")
    spec = _fake_set(tmp_path / "fake")
    with (tmp_path / "fake" / "rollouts.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(sets_mod.SetError, match="rollouts.jsonl"):
        sets_mod.load_set("fake", tmp_path / "fake")
    with pytest.raises(sets_mod.SetError, match="unknown set"):
        sets_mod.load_set("nope", tmp_path)


def test_set_specs_match_the_fixtures() -> None:
    for name, spec in sets_mod.SETS.items():
        assert len(VERDICTS[name][next(iter(VERDICTS[name]))]) == spec.n_rollouts
        assert spec.canvas_wh[0] == spec.tile_wh[0] * len(spec.views)


# ---------------------------------------------------------------------------
# The sampler with a negative prompt
# ---------------------------------------------------------------------------


def test_negative_prompt_goes_to_the_unguided_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative prompt of another length replaces the blank caption as is.

    The default call (the blank caption) is unchanged: the same reference
    loop gives both.
    """
    calls = Calls()
    monkeypatch.setattr(model_mod, "run_expert_on_taps_k", _stub_expert(calls))
    inp = _inputs(2)
    neg = torch.randn(1, 9, 2048, generator=torch.Generator().manual_seed(3)).bfloat16()
    settings = SamplerSettings(
        num_steps=2, t_shift=8.0, cfg_scale=1.0, cfg_video_scale=3.0
    )
    for uncond in (inp["text_uncond"], neg):
        calls.trunk.clear()
        got = _policy(calls).sample_window_joint(
            seed=4, settings=settings, **{**inp, "text_uncond": uncond}
        )
        want = _reference_loop({**inp, "text_uncond": uncond}, 4, 2, 8.0, 1.0, 3.0)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
        assert [c["text"] is uncond for c in calls.trunk] == [False, True] * 2


# ---------------------------------------------------------------------------
# The video decode in [-1, 1]
# ---------------------------------------------------------------------------


def test_decode_is_decode_unit_in_uint8() -> None:
    from praxis.vae import Autoencoder

    torch.manual_seed(0)
    vae = Autoencoder().float()
    for p in vae.parameters():
        p.data.normal_(0, 0.02)
    latent = torch.randn(1, 192, 2, 2, 2)
    unit = vae.decode_unit(latent)
    assert unit.dtype == torch.float32 and unit.shape == (1, 3, 5, 64, 64)
    want = unit[0].permute(1, 0, 2, 3).float().mul(0.5).add(0.5).clamp(0, 1)
    assert torch.equal(vae.decode(latent), (want * 255).to(torch.uint8))


# ---------------------------------------------------------------------------
# The rollout (tiny model)
# ---------------------------------------------------------------------------


def test_history_window_geometry() -> None:
    from praxis.eval.autoregressive.rollout import history_window

    assert history_window(1, True, 4) == (0, 0)
    got = [history_window(21 + 24 * j, False, 4) for j in range(7)]
    assert got == [(0, 1), (0, 2), (0, 3), (24, 3), (48, 3), (72, 3), (96, 3)]
    for n in (21 + 24 * j for j in range(7)):
        w, k = history_window(n, False, 4)
        assert n - w == 24 * k - 3  # the window encodes to 6 k latents


def test_first_frame_clip_keeps_the_canvas() -> None:
    from praxis.eval.autoregressive.rollout import first_frame_clip

    rng = np.random.default_rng(0)
    for name, shape in (
        ("praxis-1-droid", (224, 832, 3)),
        ("praxis-1-yam", (256, 1248, 3)),
    ):
        config = json.loads((DATA / f"{name}.json").read_text())
        frame = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(first_frame_clip(frame, config), frame)
        with pytest.raises(ValueError, match="canvas"):
            first_frame_clip(frame[:, :-32], config)


def test_rollout_seeds_frames_and_negative_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5 chunks of the tiny model: seeds, depths, text of the unguided pass, frames."""
    from tiny_model import TINY_HW, tiny_policy
    from torch.nn.attention.flex_attention import flex_attention

    import praxis.layers
    from praxis.eval.autoregressive import rollout as R

    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    policy = tiny_policy()
    h, w = TINY_HW
    H, W = 32 * h, 32 * w

    def encode(clip: torch.Tensor) -> torch.Tensor:
        t = (clip.shape[2] - 1) // 4 + 1
        m = clip.float().mean() / 255
        return torch.full((1, 192, t, h, w), float(m)).bfloat16()

    def decode_unit(lat: torch.Tensor) -> torch.Tensor:
        t = 1 + 4 * (lat.shape[2] - 1)
        x = torch.tanh(lat.float().mean(dim=(1, 3, 4)))[:, None, :, None, None]
        x = x.repeat_interleave(4, 2)[:, :, 3:] if lat.shape[2] > 1 else x
        return x[:, :, -t:].expand(1, 3, t, H, W)

    seen: list[dict] = []
    real = policy.sample_window_joint

    def spy(**kw: object) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(policy, "sample_window_joint", spy)
    g = torch.Generator().manual_seed(0)
    text = torch.randn(1, 5, 32, generator=g).bfloat16()
    neg = torch.randn(1, 7, 32, generator=g).bfloat16()
    frame = np.full((H, W, 3), 100, np.uint8)
    settings = SamplerSettings(
        num_steps=1, t_shift=8.0, cfg_scale=1.0, cfg_video_scale=3.0
    )
    roll = R.VideoRollout(policy, encode, decode_unit, frame, seed=7, settings=settings)
    for _ in range(5):
        roll.step(text, neg)
    assert [kw["seed"] for kw in seen] == [7000, 7001, 7002, 7003, 7004]
    assert [len(kw["cleans"]) for kw in seen] == [0, 1, 2, 3, 3]
    assert all(kw["text_uncond"] is neg and kw["text"] is text for kw in seen)
    assert [c["window_start"] for c in roll.log] == [0, 0, 0, 0, 24]
    assert roll.video().shape == (21 + 4 * 24, H, W, 3)
    assert roll.video().dtype == np.uint8


def test_eval_configs_hold_the_published_recipe() -> None:
    import yaml

    from praxis.eval.autoregressive import rollout as R

    for name in ("autoregressive_droid.yaml",):
        cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / name).read_text())
        assert SamplerSettings.from_config(cfg["sampler"]) == R.EVAL_SAMPLER
        r = cfg["rollout"]
        assert (r["chunks"], r["max_chunks"]) == (R.CHUNKS, R.MAX_CHUNKS)
        assert r["negative_prompt"] == R.NEGATIVE_PROMPT
        assert r["set"] in sets_mod.SETS


def test_to_uint8_rounds_like_the_reference() -> None:
    from praxis.eval.autoregressive.rollout import to_uint8

    x = (
        torch.tensor([-1.0, -0.5, 0.0, 0.00393, 1.0, 2.0])
        .view(1, 1, 6, 1, 1)
        .expand(1, 3, 6, 1, 1)
    )
    assert to_uint8(x)[:, 0, 0, 0].tolist() == [0, 64, 128, 128, 255, 255]


# ---------------------------------------------------------------------------
# Review fixes: invalid answers, missing times, input format, retries
# ---------------------------------------------------------------------------


def test_full_adherence_refuses_a_verdict_without_the_booleans() -> None:
    ok = dict.fromkeys(S.FIELDS, True)
    assert S.full_adherence(ok) == 1.0
    for bad in (
        dict.fromkeys(S.FIELDS, "false"),
        {"target_object_acted_on": True},
        {**ok, "action_correct": None},
        {**ok, "action_correct": 1},
    ):
        with pytest.raises(ValueError, match="booleans"):
            S.full_adherence(bad)


def test_cutoff_refuses_a_missing_time_to_success_answer() -> None:
    ok = dict.fromkeys(S.FIELDS, True)
    no = {**ok, "action_correct": False}
    runs = {"run": {"a__p0__s0": ok, "a__p0__s1": no}}
    with pytest.raises(ValueError, match="a__p0__s0.*time-to-success"):
        S.summarize(runs, tts={"run": {}}, cutoff_s=9.8)
    with pytest.raises(ValueError, match="time-to-success"):
        S.within(ok, {"grasp_time_s": 1.0}, 9.8)
    with pytest.raises(ValueError, match="success_time_s"):
        S.within(ok, {"success_time_s": "3.0"}, 9.8)
    for bad in (-1.0, float("nan"), float("-inf"), float("inf")):
        with pytest.raises(ValueError, match="success_time_s"):
            S.within(ok, {"success_time_s": bad}, 9.8)
    assert S.success_time({"success_time_s": 0}) == 0
    assert S.success_time({"success_time_s": 3.2}) == 3.2
    # A failed rollout needs no answer; an explicit null is "never visibly done".
    tts = {"run": {"a__p0__s0": {"success_time_s": None}}}
    assert S.run_values(runs["run"], tts["run"], 9.8) == {
        "a__p0__s0": 0.0,
        "a__p0__s1": 0.0,
    }
    tts = {"run": {"a__p0__s0": {"success_time_s": 4.0}}}
    assert S.run_values(runs["run"], tts["run"], 9.8)["a__p0__s0"] == 1.0


def test_p_text_reports_the_resolution_of_the_sign_flips() -> None:
    assert S.p_text(0.0) == "p<=0.0001 (0 of 10000 sign flips)"
    assert S.p_text(0.6121) == "p=0.612"


def _cli():
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / "score_autoregressive_eval.py"
    spec = importlib.util.spec_from_file_location("score_autoregressive_eval", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli_set() -> sets_mod.EvalSet:
    ro = {"id": "a__p0__s0", "scene": "a", "prompt": "put the cup in the bowl"}
    return sets_mod.EvalSet(sets_mod.SETS["droid"], Path("."), {"a": {}}, [ro])


@pytest.mark.parametrize(
    "answer", [{}, dict.fromkeys(S.FIELDS, "false"), {"target_object_acted_on": True}]
)
def test_judge_cli_does_not_write_an_invalid_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: dict
) -> None:
    cli = _cli()
    monkeypatch.setattr(cli.J, "read_video", lambda p: ([], 15.0))
    monkeypatch.setattr(cli.J, "request", lambda *a: {})
    monkeypatch.setattr(cli.J, "call", lambda req: dict(answer))
    es = _cli_set()
    with pytest.raises(ValueError):
        cli._judge_one(es, tmp_path, es.rollouts[0], False, "m")
    assert not (tmp_path / "a__p0__s0.verdict.json").exists()
    monkeypatch.setattr(cli.J, "call", lambda req: dict.fromkeys(S.FIELDS, True))
    cli._judge_one(es, tmp_path, es.rollouts[0], False, "m")
    assert (tmp_path / "a__p0__s0.verdict.json").exists()


def test_score_cli_refuses_two_runs_with_the_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    cli = _cli()
    monkeypatch.setattr(cli.sets, "load_set", lambda *a: _cli_set())
    for d in ("a/run", "b/run"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "a__p0__s0.verdict.json").write_text(
            json.dumps(dict.fromkeys(S.FIELDS, True))
        )
    args = argparse.Namespace(
        set="droid",
        set_dir=None,
        runs=[str(tmp_path / "a/run"), str(tmp_path / "b/run")],
        cutoff=None,
        out=None,
    )
    with pytest.raises(SystemExit, match="same name"):
        cli.score(args)


def test_judge_reads_the_views_and_tiles_of_the_set_spec() -> None:
    for name, spec in sets_mod.SETS.items():
        assert J._cameras(name).count("camera ") == len(spec.judge_views)
        assert (
            J.system_prompt(name)
            == (J.HERE / f"judge_system_{spec.judge}.txt").read_text()
        )
    assert sets_mod.SETS["droid"].judge_canvas_wh == (832, 224)


@pytest.mark.parametrize(
    ("w", "h", "n", "fps", "match"),
    [
        (900, 176, 189, 15.0, "900x176"),
        (1248, 256, 189, 15.0, "1248x256"),
        (832, 224, 100, 15.0, "100 frames"),
        (832, 224, 189, 30.0, "30.0 fps"),
    ],
)
def test_judge_refuses_a_clip_of_another_format(
    w: int, h: int, n: int, fps: float, match: str
) -> None:
    frames = [np.zeros((h, w, 3), np.uint8)] * n
    ro = {"id": "a__p0__s0", "prompt": "put the cup in the bowl"}
    with pytest.raises(ValueError, match=match):
        J.request(ro, {}, "droid", frames, fps)
    with pytest.raises(ValueError, match=match):
        J.tts_request(ro, {}, "droid", frames, fps)


def test_judge_refuses_frames_of_different_sizes() -> None:
    frames = [np.zeros((224, 832, 3), np.uint8)] * 188
    frames.append(np.zeros((224, 824, 3), np.uint8))
    with pytest.raises(ValueError, match="differ in size"):
        J.judged_frames(frames, 15.0, "droid")


class _StatusError(Exception):
    def __init__(self, status: int, text: str = "error") -> None:
        super().__init__(text)
        self.status_code = status


class _FailApi(_Api):
    def __init__(self, exc: Exception) -> None:
        super().__init__(fail=99)
        self.exc = exc

    def create(self, **kw: object) -> _Reply:
        self.calls.append(kw)
        raise self.exc


@pytest.mark.parametrize("status", [401, 403, 404, 400])
def test_call_does_not_retry_an_auth_or_model_error(status: int) -> None:
    api = _FailApi(_StatusError(status, "invalid api key"))
    req = {"model": "m", "messages": [], "response_format": {"type": "json_schema"}}
    with pytest.raises(RuntimeError, match=f"without a retry \\(HTTP {status}"):
        J.call(req, api=api, sleep=lambda s: None)
    assert len(api.calls) == 1


def test_call_retries_a_server_error_and_falls_back_only_on_a_schema_refusal() -> None:
    api = _FailApi(_StatusError(500, "response_format upstream timeout"))
    req = {"model": "m", "messages": [], "response_format": {"type": "json_schema"}}
    with pytest.raises(RuntimeError, match="judge failed"):
        J.call(req, api=api, retries=3, sleep=lambda s: None)
    assert len(api.calls) == 3
    assert all(c["response_format"] == {"type": "json_schema"} for c in api.calls)
    api = _FailApi(_StatusError(400, "response_format json_schema is not supported"))
    with pytest.raises(RuntimeError, match="without a retry"):
        J.call(req, api=api, retries=3, sleep=lambda s: None)
    # The 400 refusal of the schema switches to json_object once; the second
    # 400 (now with json_object) stops the call.
    assert [c["response_format"]["type"] for c in api.calls] == [
        "json_schema",
        "json_object",
    ]


# ---------------------------------------------------------------------------
# The set-generic code that no released set reaches yet (the feature section
# and the per-tile resize of a later bimanual release), on a synthetic set
# ---------------------------------------------------------------------------


def _bimanual_spec() -> sets_mod.SetSpec:
    return sets_mod.SetSpec(
        name="synthetic_bimanual",
        rig="bimanual",
        views=("top", "left wrist", "right wrist"),
        judge_views=("top", "left wrist", "right wrist"),
        tile_wh=(320, 176),
        praxis_tile_wh=(416, 256),
        judge="yam",
        rollouts_sha256="",
        scenes_sha256="",
        n_rollouts=1,
        n_scenes=1,
    )


def test_feature_section_of_a_count_and_a_spatial_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sets_mod.SETS, "synthetic_bimanual", _bimanual_spec())
    count = {
        "target_object": "red_cube",
        "destination": "bowl",
        "action": "place_in",
        "feature": "count",
        "level": "hard",
        "required_count": 2,
        "count_mode": "exact",
        "n_present": 3,
        "target_instances": ["red_cube_1", "red_cube_2", "red_cube_3"],
    }
    spec = J.adherence_spec(count, "synthetic_bimanual")
    assert spec["required_count"] == 2 and spec["distractors"] == []
    assert spec["distance_confounded"] is False
    text = J.feature_text(spec, {})
    assert "EXACTLY 2 red cube item(s) in/on bowl" in text
    assert "red_cube_1, red_cube_2, red_cube_3" in text
    spatial = {
        "target_object": "cup_left",
        "destination": "plate",
        "action": "place_on",
        "feature": "spatial",
        "level": "medium",
        "relation": "left of the bowl",
        "relation_applies_to": "object",
    }
    scene = {
        "twins": ["cup_left", "cup_right"],
        "top_center_px": {"cup_left": [100, 120], "cup_right": [300, 118]},
        "objects": [{"name": "cup_left", "rests_in": "table", "state": None}],
    }
    spec = J.adherence_spec(spatial, "synthetic_bimanual")
    assert spec["distance_confounded"] is True
    text = J.feature_text(spec, scene)
    assert "cup_left at [100, 120] (resting table)" in text
    assert "cup_right at [300, 118]" in text
    assert "distance-confounded" in text
    # A set without the feature section keeps the adherence targets only.
    assert "feature" not in J.adherence_spec(spatial, "droid")


def test_judge_resizes_a_native_canvas_per_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _bimanual_spec()
    monkeypatch.setitem(sets_mod.SETS, "synthetic_bimanual", spec)
    native = synthetic_frames(960, 176, 3)
    resized = sets_mod.resize_tiles(native[0], spec)
    assert resized.shape == (256, 1248, 3)
    want = Image.fromarray(np.ascontiguousarray(native[0][:, 320:640])).resize(
        (416, 256), Image.Resampling.BICUBIC
    )
    assert np.array_equal(resized[:, 416:832], np.asarray(want))
    out = J.judged_frames(native, 15.0, "synthetic_bimanual")
    assert np.array_equal(out[0], resized)
    kept = [np.zeros((256, 1248, 3), np.uint8)] * 189
    assert J.judged_frames(kept, 15.0, "synthetic_bimanual") is kept
