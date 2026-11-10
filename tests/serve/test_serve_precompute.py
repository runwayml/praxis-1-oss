"""The overlap precompute of a real-time session (CPU only, no weights).

A real-time session with ``precompute`` (on by default) encodes the window
of its next prediction and runs the trunk passes on the samples before it
(:mod:`praxis.serve.precompute`). The stand-in policy splits its sampler the
same way. Each session here runs next to a session without the precompute
on the same samples: every answer must have the same bits, and each
prediction reports how its trunk passes ran (``rtc_plan``).
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import yaml
from test_serve_rtc import (
    DELAYS,
    DROID,
    YAM,
    _current,
    _drive,
    _droid_factory,
    _droid_sample,
    _Encoder,
    _factory,
    _obs,
    _Policy,
)
from test_serve_rtc import _Pipeline as _RtcPipeline

from praxis import bimanual
from praxis.model import TrunkPlan
from praxis.pipeline import PraxisPipeline, StepResult
from praxis.serve import periodic as per
from praxis.serve import rtc
from praxis.serve.policy import SESSION_DEFAULTS, PolicyFactory, ServedPolicy

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]


class _Pipeline(_RtcPipeline):
    """The stand-in pipeline with a text encode that reads the prompt."""

    def encode_text(self, prompt: str) -> torch.Tensor:
        return torch.full((1, 4, 8), float(len(prompt)))


class _PlanPolicy(_Policy):
    """The stand-in networks with the sampler split in its trunk and expert parts.

    The actions read the anchor, the cleans, the proprio and the seed, so a
    wrong window or a wrong seed changes them.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.plans: list[int] = []
        self.plan_skips: list[bool] = []
        self.finishes = 0

    def _actions(self, anchor, cleans, proprio_all, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(int(seed))
        dim = 4 * int(self.config["action_space"]["dim"])
        noise = torch.randn((1, 6, dim), generator=generator)
        ctx = sum(float(z.mean()) for z in cleans)
        proprio = float(proprio_all[..., -1, :].sum())
        return 0.1 * noise + 0.001 * (float(anchor.mean()) + ctx + proprio)

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
    ):
        self.calls.append(
            {"depth": len(cleans), "seed": int(seed), "skip": skip_last_video}
        )
        actions = self._actions(anchor, cleans, proprio_all, seed)
        return None if skip_last_video else torch.zeros(1, 1, 6, 1, 1), actions

    def plan_window_joint(
        self, text, text_uncond, anchor, cleans, seed, settings, skip_last_video=False
    ):
        self.plans.append(int(seed))
        self.plan_skips.append(bool(skip_last_video))
        return TrunkPlan(
            text=text,
            text_uncond=text_uncond,
            anchor=anchor,
            cleans=list(cleans),
            seed=int(seed),
            settings=settings,
            taps=[],
            taps_uncond=[],
            video=None if skip_last_video else torch.zeros(1, 1, 6, 1, 1),
        )

    def finish_window_joint(self, plan: TrunkPlan, proprio_all):
        self.finishes += 1
        self.calls.append(
            {"depth": len(plan.cleans), "seed": plan.seed, "skip": plan.video is None}
        )
        return plan.video, self._actions(
            plan.anchor, plan.cleans, proprio_all, plan.seed
        )


def _with_plans(factory: PolicyFactory) -> PolicyFactory:
    """The factory on the split stand-in and a text encode that reads the prompt."""
    config = factory.pipe.policy.config
    pipe = _Pipeline(config)
    pipe.policy = _PlanPolicy(config)
    factory.pipe = cast(PraxisPipeline, pipe)
    return factory


def _send(policy: ServedPolicy, n: int, **extra: object) -> list[dict] | None:
    """Send the 15 Hz sample ``n`` of the bimanual rig."""
    obs = _current(_obs([n]))
    obs.update(
        {
            per.SAMPLE_INDEX: n,
            per.OBS_TIME_NS: round(n / 15.0 * 1e9),
            per.CALL_ID: 1000 + n,
            per.CALL_TIMINGS: [],
            **extra,
        }
    )
    return policy.select_action(obs)


class _Recorder:
    """A session for ``_drive`` that keeps the plan keys and the encodes of each prediction."""

    def __init__(self, policy: ServedPolicy) -> None:
        self.policy = policy
        self.plans: list[str | None] = []
        self.encodes: list[list[list[int]]] = []

    def select_action(self, obs: dict) -> list[dict] | None:
        vae = self.policy.factory.pipe.vae
        assert isinstance(vae, _Encoder)
        n = len(vae.raws)
        rows = self.policy.select_action(obs)
        if rows is not None:
            plan = self.policy.timing.get("rtc_plan")
            self.plans.append(None if plan is None else str(plan))
            self.encodes.append(vae.raws[n:])
        return rows

    def as_policy(self) -> ServedPolicy:
        """This recorder in place of a session (``_drive`` calls ``select_action`` only)."""
        return cast(ServedPolicy, self)


def _pair(
    make: Callable[[], PolicyFactory] = _factory, **query: object
) -> tuple[ServedPolicy, ServedPolicy]:
    """A session without and one with the precompute, on two factories."""
    base = {"schedule": "rtc", "sample_seed": 0, **query}
    a = _with_plans(make())({**base, "precompute": "off"})
    b = _with_plans(make())({**base, "precompute": "on"})
    return a, b


def _same_rows(x: list[list[dict]], y: list[list[dict]], sides: tuple) -> None:
    assert len(x) == len(y)
    chans = rtc.channels(sides)
    for a, b in zip(x, y, strict=True):
        np.testing.assert_array_equal(
            rtc.rows_to_array(a, chans), rtc.rows_to_array(b, chans)
        )


@pytest.mark.parametrize("sample_fps", [15.0, 30.0])
def test_the_precompute_serves_the_same_rows(sample_fps: float) -> None:
    """From the third prediction on (the first with a context chunk) the
    trunk passes run ahead of the prediction, and the prediction encodes only
    its current frame. The rows keep their bits."""
    a, b = _pair(sample_fps=sample_fps)
    rec = _Recorder(b)
    want, at_a, _ = _drive(a, DELAYS, 9.0, sample_fps=sample_fps)
    got, at_b, _ = _drive(rec.as_policy(), DELAYS, 9.0, sample_fps=sample_fps)
    assert at_a == at_b and len(got) == len(DELAYS)
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans == ["none", "none", *["hit"] * (len(DELAYS) - 2)]
    # A hit encodes the current frame only: the window and the anchor latent
    # came from the samples before it.
    ratio = round(sample_fps / 15.0)
    for n, enc in zip(at_b[2:], rec.encodes[2:], strict=True):
        assert enc == [[round(n / ratio)]]
    policy = b.factory.pipe.policy
    assert isinstance(policy, _PlanPolicy)
    assert policy.finishes == len(DELAYS) - 2
    assert b.session.memo is not None and b.session.memo._kept == []
    assert not b.session.has_plan


def test_the_window_is_encoded_on_the_samples_before_the_prediction() -> None:
    """The stand-in encode has no fused encoder, so the window goes to it in
    one call on the sample of its last row, one row before the prediction."""
    factory = _with_plans(_factory())
    policy = factory({"schedule": "rtc", "sample_seed": 0, "precompute": "on"})
    _answers, at, nones = _drive(policy, DELAYS[:5], 6.0, sample_fps=15.0)
    vae = factory.pipe.vae
    assert isinstance(vae, _Encoder)
    windows = [r for r in vae.raws if len(r) > 1]
    assert windows == [list(range(n - 21, n)) for n in at[2:]]
    assert all(n - 1 in nones for n in at[2:])


def test_a_prompt_change_at_the_prediction_misses_and_falls_back() -> None:
    """The plan ran with the prompt of the samples before the prediction. A
    prediction that brings another prompt samples without the plan."""
    a, _ = _pair()
    _, at, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    change = at[3]
    a2, b2 = _pair()
    rec = _Recorder(b2)

    def sample(raw: float) -> dict:
        obs = _current(_obs([raw]))
        obs["task"] = "fold the towel" if raw >= change else "pick up the pen"
        return obs

    want, at_a, _ = _drive(a2, DELAYS[:6], 6.0, sample=sample, sample_fps=15.0)
    got, at_b, _ = _drive(
        rec.as_policy(), DELAYS[:6], 6.0, sample=sample, sample_fps=15.0
    )
    assert at_a == at_b == at
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans == ["none", "none", "hit", "miss", "hit", "hit"]


@pytest.mark.parametrize("skip", ["on", "off"])
def test_a_miss_falls_back_with_the_skip_of_the_session(skip: str) -> None:
    """The plan and the stock sampler of a miss both take the session's
    skip_video_decoder: a miss returns no video under the skip, as a hit
    does, and a video without it, as a hit does."""
    a, _ = _pair()
    _, at, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    change = at[3]
    _, b = _pair(skip_video_decoder=skip)
    rec = _Recorder(b)
    videos: list[bool] = []
    step = b.session.step

    def step_and_record(*args: object, **kwargs: object) -> StepResult:
        res = step(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        videos.append(res.pred_video_latent is not None)
        return res

    setattr(b.session, "step", step_and_record)  # noqa: B010

    def sample(raw: float) -> dict:
        obs = _current(_obs([raw]))
        obs["task"] = "fold the towel" if raw >= change else "pick up the pen"
        return obs

    _drive(rec.as_policy(), DELAYS[:6], 6.0, sample=sample, sample_fps=15.0)
    assert rec.plans == ["none", "none", "hit", "miss", "hit", "hit"]
    policy = b.factory.pipe.policy
    assert isinstance(policy, _PlanPolicy)
    want = skip == "on"
    assert policy.plan_skips and set(policy.plan_skips) == {want}
    assert [c["skip"] for c in policy.calls] == [want] * 6
    # Cold starts, hits and the miss: the same video presence on every call.
    assert videos == [not want] * 6


def test_a_changed_window_misses_and_falls_back() -> None:
    """Window frames that differ from the call's (here: the precompute reads
    other canvas bits) give no stored latent and no plan."""
    a, b = _pair()
    rec = _Recorder(b)
    state = b.session.state
    real = state.frame_clip

    def shifted(row: dict) -> torch.Tensor:
        return real(row) + 1.0

    setattr(state, "frame_clip", shifted)  # noqa: B010
    want, _, _ = _drive(a, DELAYS[:5], 6.0, sample_fps=15.0)
    got, _, _ = _drive(rec.as_policy(), DELAYS[:5], 6.0, sample_fps=15.0)
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans[2:] == ["miss"] * 3
    assert b.session.memo_hits == 0


def test_a_skipped_prediction_sample_drops_the_stale_plan() -> None:
    """The client skipped the sample of the planned prediction: the next
    sample predicts on a window one row later, so the plan is dropped."""
    a, _ = _pair()
    _, at, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    skip = at[3]
    a2, b2 = _pair()
    rec = _Recorder(b2)
    want, at_a, _ = _drive_skipping(a2, DELAYS[:6], 6.0, {skip})
    got, at_b, _ = _drive_skipping(rec.as_policy(), DELAYS[:6], 6.0, {skip})
    assert at_a == at_b and skip + 1 in at_b
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans[at_b.index(skip + 1)] == "stale"
    assert rec.plans.count("hit") >= 2


def test_a_skipped_last_row_plans_in_the_prediction() -> None:
    """The client skipped the sample of the last window row: the window takes
    the sample before it, as the stack of the call does, and the prediction
    runs the trunk passes itself."""
    a, _ = _pair()
    _, at, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    skip = at[3] - 1
    a2, b2 = _pair()
    rec = _Recorder(b2)
    want, at_a, _ = _drive_skipping(a2, DELAYS[:6], 6.0, {skip})
    got, at_b, _ = _drive_skipping(rec.as_policy(), DELAYS[:6], 6.0, {skip})
    assert at_a == at_b
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans[3] == "late"


def test_a_replan_and_a_reset_drop_the_work() -> None:
    """A replan predicts on its sample, not the planned one: the work is
    stale. A reset restarts the context: the work is dropped."""
    a, b = _pair()
    for policy in (a, b):
        _drive(policy, DELAYS[:3], 2.5, sample_fps=15.0)
    grid, ahead = b._grid, b._ahead
    assert grid is not None and grid.next_index is not None and ahead is not None
    last = b._history.newest_index if b._history is not None else None
    assert last is not None
    n = grid.next_index - 3
    for m in range(last + 1, n):
        assert _send(a, m) is None and _send(b, m) is None
    assert ahead.target == grid.next_index and ahead.planned is None
    want = _send(a, n, **{per.REPLAN_NOW: True})
    got = _send(b, n, **{per.REPLAN_NOW: True})
    assert want is not None and got is not None
    _same_rows([want], [got], bimanual.SIDES)
    assert b.timing["rtc_plan"] == "stale" and ahead.target is None
    assert not b.session.has_plan
    # After a replan the grid waits for the duration of its call, as after
    # the first prediction; the work starts once the client reports it.
    report = {per.CALL_TIMINGS: [{per.CALL_ID: 1000 + n, per.ELAPSED_SEC: 0.3}]}
    assert _send(a, n + 1, **report) is None and _send(b, n + 1, **report) is None
    assert ahead.target == grid.next_index == n + 12 + round(0.3 * 15)
    for m in range(n + 2, n + 12):
        _send(a, m)
        _send(b, m)
    assert ahead.target is not None and ahead.planned is None
    m = n + 12
    want = _send(a, m, reset_context=True)
    got = _send(b, m, reset_context=True)
    assert want is not None and got is not None
    _same_rows([want], [got], bimanual.SIDES)
    assert ahead.target is None and not b.session.has_plan
    _send(b, m + 1)
    b.close()
    assert ahead.target is None and not b.session.has_plan


def test_a_single_arm_session_serves_the_same_rows() -> None:
    a, b = _pair(_droid_factory)
    rec = _Recorder(b)
    want, _, _ = _drive(a, DELAYS[:6], 6.0, sample=_droid_sample, sample_fps=15.0)
    got, _, _ = _drive(
        rec.as_policy(), DELAYS[:6], 6.0, sample=_droid_sample, sample_fps=15.0
    )
    _same_rows(want, got, ())
    assert rec.plans == ["none", "none", "hit", "hit", "hit", "hit"]
    assert DROID["embodiment"] == "single_arm" and YAM["embodiment"] == "bimanual"


def test_the_precompute_is_on_by_default_for_real_time_sessions() -> None:
    """The default is on in SESSION_DEFAULTS and in both serve configs. It
    applies to a real-time session with an observed context, and nowhere else."""
    assert SESSION_DEFAULTS["precompute"] is True
    for name in ("serve_yam.yaml", "serve_droid.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
        assert cfg["session"]["precompute"] is True, name
        assert cfg["session"]["schedule"] == "chunked", name
    factory = _factory()
    rt = factory({"schedule": "rtc"})
    assert rt.precompute and rt.meta["precompute"] is True
    assert rt._ahead is not None and rt.session.memo is not None
    # A chunked session (the default schedule) is unaffected: no precompute,
    # no memo, no precompute key in its meta.
    for chunked in (factory({}), factory({"schedule": "chunked"})):
        assert not chunked.precompute and chunked._ahead is None
        assert chunked.session.memo is None and "precompute" not in chunked.meta


def test_precompute_off_in_the_config_or_the_session_turns_it_off() -> None:
    base = _factory()
    off_boot = PolicyFactory(
        base.pipe, {**base.defaults, "precompute": False}, "/ckpt/praxis-1-yam"
    )
    for policy in (
        off_boot({"schedule": "rtc"}),
        base({"schedule": "rtc", "precompute": "off"}),
    ):
        assert not policy.precompute and policy._ahead is None
        assert policy.session.memo is None and policy.meta["precompute"] is False
    # A deploy that turns it off still serves an explicit precompute=on.
    assert off_boot({"schedule": "rtc", "precompute": "on"}).precompute is True
    # precompute=off asks for nothing, so a chunked session takes it.
    assert not base({"precompute": "off"}).precompute


def test_an_explicit_precompute_on_needs_the_real_time_schedule() -> None:
    """The default on applies silently where it can. An explicit on that
    cannot apply is refused, as the other real-time keys are."""
    factory = _factory()
    with pytest.raises(ValueError, match="schedule=rtc"):
        factory({"precompute": "on"})
    with pytest.raises(ValueError, match="schedule=rtc"):
        factory({"schedule": "chunked", "precompute": "on"})
    # A boot default of rtc: a session that asks for a stateless context gets
    # the chunked schedule, and the default precompute does nothing there.
    boot = PolicyFactory(
        factory.pipe, {**factory.defaults, "schedule": "rtc"}, "/ckpt/praxis-1-yam"
    )
    debug = boot({"context_mode": "stateless"})
    assert debug.schedule == "chunked" and not debug.precompute
    assert boot({}).precompute is True


def test_a_default_session_plans_and_an_off_session_does_not() -> None:
    """The default rtc session plans its predictions ahead; a precompute=off
    session runs the stock path. The rows of the two have the same bits."""
    on = _with_plans(_factory())({"schedule": "rtc", "sample_seed": 0})
    off = _with_plans(_factory())(
        {"schedule": "rtc", "sample_seed": 0, "precompute": "off"}
    )
    rec_on, rec_off = _Recorder(on), _Recorder(off)
    got, at_on, _ = _drive(rec_on.as_policy(), DELAYS, 9.0, sample_fps=15.0)
    want, at_off, _ = _drive(rec_off.as_policy(), DELAYS, 9.0, sample_fps=15.0)
    assert at_on == at_off
    _same_rows(want, got, bimanual.SIDES)
    assert rec_on.plans == ["none", "none", *["hit"] * (len(DELAYS) - 2)]
    assert rec_off.plans == [None] * len(DELAYS)
    on_policy, off_policy = on.factory.pipe.policy, off.factory.pipe.policy
    assert isinstance(on_policy, _PlanPolicy) and isinstance(off_policy, _PlanPolicy)
    assert len(on_policy.plans) == len(DELAYS) - 2 and off_policy.plans == []
    assert off_policy.finishes == 0


@pytest.mark.parametrize("sample_fps", [15.0, 30.0])
def test_a_three_chunk_context_keeps_the_precompute(sample_fps: float) -> None:
    """At max_context_chunks=3 the window of a prediction is 69 rows, more than
    the frame stack. The history keeps them, so every warm prediction at
    depth 3 still plans ahead, and the rows keep their bits."""
    delays = DELAYS * 2
    a, b = _pair(max_context_chunks=3, sample_fps=sample_fps)
    rec = _Recorder(b)
    want, at_a, _ = _drive(a, delays, 20.0, sample_fps=sample_fps)
    got, at_b, _ = _drive(rec.as_policy(), delays, 20.0, sample_fps=sample_fps)
    assert at_a == at_b and len(got) == len(delays)
    _same_rows(want, got, bimanual.SIDES)
    policy = b.factory.pipe.policy
    assert isinstance(policy, _PlanPolicy)
    depths = [c["depth"] for c in policy.calls]
    assert depths.count(3) >= 6
    assert rec.plans == ["none", "none", *["hit"] * (len(delays) - 2)]


def test_a_miss_frees_the_plan_before_the_fallback() -> None:
    """On a miss the session drops its plan before the stock sampler runs,
    so the fallback does not hold the stale taps."""
    a, _ = _pair()
    _, at, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    change = at[3]
    _, b = _pair()
    policy = b.factory.pipe.policy
    assert isinstance(policy, _PlanPolicy)
    held: list[bool] = []
    plans: list[weakref.ref] = []
    plan_window = policy.plan_window_joint
    stock = policy.sample_window_joint

    def plan_and_watch(*args: object, **kwargs: object) -> TrunkPlan:
        plan = plan_window(*args, **kwargs)
        plans.append(weakref.ref(plan))
        return plan

    def sample_and_check(*args: object, **kwargs: object):
        gc.collect()
        held.append(any(ref() is not None for ref in plans))
        return stock(*args, **kwargs)

    setattr(policy, "plan_window_joint", plan_and_watch)  # noqa: B010
    setattr(policy, "sample_window_joint", sample_and_check)  # noqa: B010

    def sample(raw: float) -> dict:
        obs = _current(_obs([raw]))
        obs["task"] = "fold the towel" if raw >= change else "pick up the pen"
        return obs

    rec = _Recorder(b)
    _drive(rec.as_policy(), DELAYS[:6], 6.0, sample=sample, sample_fps=15.0)
    assert rec.plans[3] == "miss"
    # The stock sampler ran on the two cold starts and on the miss; no plan
    # was alive on any of them.
    assert len(held) == 3 and not any(held)


def test_a_batched_guidance_session_does_not_plan() -> None:
    """The plan runs the guidance passes one after the other. A session with
    cfg_batched=on makes no plan: its predictions report none and run the
    stock sampler, with the rows of a session without the precompute. The
    window encode still runs ahead: a prediction encodes its current frame
    only."""
    a, b = _pair(cfg_video_scale=4.0, cfg_batched="on")
    assert b.precompute and b.session.sampler.cfg_batched
    rec = _Recorder(b)
    want, _, _ = _drive(a, DELAYS[:6], 6.0, sample_fps=15.0)
    got, at, _ = _drive(rec.as_policy(), DELAYS[:6], 6.0, sample_fps=15.0)
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans == ["none"] * 6
    policy = b.factory.pipe.policy
    assert isinstance(policy, _PlanPolicy)
    assert policy.plans == [] and policy.finishes == 0
    assert all(enc == [[n]] for n, enc in zip(at[2:], rec.encodes[2:], strict=True))


def _drive_skipping(
    policy: ServedPolicy, delays: list[float], seconds: float, skipped: set[int]
) -> tuple[list[list[dict]], list[int], list[int]]:
    """``_drive`` at 15 Hz with the samples ``skipped`` never sent."""

    class _Skip:
        def select_action(self, obs: dict) -> list[dict] | None:
            if obs[per.SAMPLE_INDEX] in skipped:
                return None
            return policy.select_action(obs)

    return _drive(cast(ServedPolicy, _Skip()), delays, seconds, sample_fps=15.0)


def test_a_prediction_off_the_row_grid_has_no_plan() -> None:
    """At 30 Hz a skipped prediction sample puts the next prediction off the
    row grid: the context restarts, so the precompute makes no plan for it.
    The rows equal those of a session without the precompute, and the
    predictions after it plan again."""
    a, _ = _pair(sample_fps=30)
    _, at, _ = _drive(a, DELAYS, 9.0)
    skip = at[3]
    a2, b2 = _pair(sample_fps=30)
    rec = _Recorder(b2)

    class _Skip:
        def __init__(self, policy: ServedPolicy) -> None:
            self.policy = policy

        def select_action(self, obs: dict) -> list[dict] | None:
            if obs[per.SAMPLE_INDEX] == skip:
                return None
            return self.policy.select_action(obs)

    want, at_a, _ = _drive(cast(ServedPolicy, _Skip(a2)), DELAYS, 9.0)
    got, at_b, _ = _drive(cast(ServedPolicy, _Skip(rec.as_policy())), DELAYS, 9.0)
    assert at_a == at_b and at_b[3] == skip + 1
    _same_rows(want, got, bimanual.SIDES)
    assert rec.plans[3] == "stale"
    assert rec.plans[4:] == ["hit"] * (len(DELAYS) - 4)
