"""Scores of the autoregressive eval: full adherence, intervals, paired differences.

A verdict is the answer of the judge for one rollout
(:mod:`praxis.eval.autoregressive.judge`). A rollout is **fully adherent**
when ``target_object_acted_on``, ``target_destination_used`` and
``action_correct`` are all true (:func:`full_adherence`).

The statistics:

* A prompt is a rollout id without its seed (``<scene>__p<k>``). The level
  of a run is the mean over prompts of the mean over the seeds of the
  prompt, in percent.
* The 95 % interval is the 2.5 and 97.5 percentiles of 10,000 bootstrap
  means over prompts (``numpy.random.default_rng(0)``).
* Two runs compare paired on the same ``(prompt, seed)``: the per-prompt
  mean of the differences, its bootstrap interval as above, and a sign-flip
  permutation p value over the prompts (10,000 draws,
  ``numpy.random.default_rng(1)``). A p value of 0 means 0 of the 10,000
  flips; :func:`p_text` prints it as ``p<=0.0001``.

The optional time cut-off counts a rollout only if it is fully adherent and
its ``success_time_s`` (the time-to-success judge) is at most the cut-off.
Every fully adherent rollout needs a time-to-success answer.
The cut-off of the published DROID results is :data:`DROID_CUTOFF_S`
(11.894 s): the time by which the fastest baseline completes 95 % of its
successes (:func:`fastest_p95`).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

N_BOOT = 10_000
DROID_CUTOFF_S = 11.894
KEY_RE = re.compile(r"^(.+?__p\d+)__s(\d+)$")
FIELDS = ("target_object_acted_on", "target_destination_used", "action_correct")


def split_id(rollout_id: str) -> tuple[str, int]:
    """Return ``(prompt, seed)`` of ``<scene>__p<k>__s<seed>``."""
    m = KEY_RE.match(rollout_id)
    if m is None:
        raise ValueError(f"rollout id {rollout_id!r} is not <scene>__p<k>__s<seed>")
    return m.group(1), int(m.group(2))


def full_adherence(verdict: Mapping) -> float:
    """1.0 if the three fields of :data:`FIELDS` are true, else 0.0.

    A verdict without the three booleans raises ``ValueError``. Thus a
    string ``"false"`` or a missing field is not scored.
    """
    if any(not isinstance(verdict.get(k), bool) for k in FIELDS):
        raise ValueError(f"a verdict needs the booleans {FIELDS}: {dict(verdict)}")
    return float(all(verdict[k] for k in FIELDS))


def success_time(tts: Mapping | None) -> float | None:
    """The ``success_time_s`` of a time-to-success answer: a time >= 0 or None.

    None means that the judge saw no visible success. A missing answer, a
    missing field, a value of another type or a negative, NaN or infinite
    value raises ``ValueError``.
    """
    if tts is None or "success_time_s" not in tts:
        raise ValueError(
            "a fully adherent rollout has no time-to-success answer "
            "(run the judge with --tts)"
        )
    t = tts["success_time_s"]
    if t is not None and (
        isinstance(t, bool) or not isinstance(t, int | float) or not 0 <= t < 1e9
    ):
        raise ValueError(f"success_time_s must be a time >= 0 or null: {t!r}")
    return t


def within(verdict: Mapping, tts: Mapping | None, cutoff_s: float) -> float:
    """1.0 if fully adherent and done by ``cutoff_s`` (time-to-success judge).

    A fully adherent rollout without a valid time-to-success answer raises
    ``ValueError`` (:func:`success_time`).
    """
    if not full_adherence(verdict):
        return 0.0
    t = success_time(tts)
    return float(t is not None and t <= cutoff_s)


def p_text(p: float, n: int = N_BOOT) -> str:
    """The sign-flip p value as printed.

    0 of ``n`` flips prints ``p<=0.0001 (0 of 10000 sign flips)``: the Monte
    Carlo resolution is ``1 / n``, not 0.
    """
    if p == 0:
        return f"p<={1 / n:g} (0 of {n} sign flips)"
    return f"p={p:.3g}"


def boot(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Return the mean and the 95 % bootstrap interval of ``x``, in percent."""
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return (
        float(100 * x.mean()),
        float(100 * np.percentile(bs, 2.5)),
        float(100 * np.percentile(bs, 97.5)),
    )


def signflip(d: np.ndarray, seed: int = 1) -> float:
    """Two-sided sign-flip permutation p value of the mean of ``d``."""
    if len(d) == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], (N_BOOT, len(d)))
    return float((np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def per_prompt(values: Mapping[str, float]) -> np.ndarray:
    """Mean over the seeds of each prompt, prompts in the order of the sorted ids."""
    d: dict[str, list[float]] = defaultdict(list)
    for rid, v in sorted(values.items()):
        d[split_id(rid)[0]].append(v)
    return np.array([np.mean(v) for v in d.values()])


def level(values: Mapping[str, float]) -> dict:
    """The level of one run: ``values`` maps rollout id -> 0.0 / 1.0."""
    x = per_prompt(values)
    mean, lo, hi = boot(x)
    return {"mean": mean, "lo": lo, "hi": hi, "n_prompts": len(x), "n": len(values)}


def paired(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    """``a - b`` paired on the rollout ids that both runs have."""
    d: dict[str, list[float]] = defaultdict(list)
    for rid, va in sorted(a.items()):
        if rid in b:
            d[split_id(rid)[0]].append(va - b[rid])
    x = np.array([np.mean(v) for v in d.values()])
    if len(x) == 0:
        return {"n_prompts": 0}
    mean, lo, hi = boot(x)
    return {
        "delta": mean,
        "lo": lo,
        "hi": hi,
        "p_perm": signflip(x),
        "n_prompts": len(x),
        "n_pairs": sum(len(v) for v in d.values()),
    }


def fastest_p95(success_times: Mapping[str, list[float | None]]) -> float:
    """The smallest 95th percentile of the success times over the models.

    A time of ``None`` (the judge saw no visible success) is dropped.
    """
    p95 = []
    for times in success_times.values():
        t = [x for x in times if x is not None]
        if t:
            p95.append(float(np.percentile(t, 95)))
    return min(p95)


def load_verdicts(folder: str | Path, suffix: str = "verdict") -> dict[str, dict]:
    """Read ``<id>.<suffix>.json`` files of a run folder: rollout id -> verdict."""
    out = {}
    for p in sorted(Path(folder).glob(f"*.{suffix}.json")):
        out[p.name[: -len(f".{suffix}.json")]] = json.loads(p.read_text())
    return out


def run_values(
    verdicts: Mapping[str, dict],
    tts: Mapping[str, dict] | None = None,
    cutoff_s: float | None = None,
) -> dict[str, float]:
    """Rollout id -> 1.0 / 0.0 (full adherence, or within the cut-off).

    An invalid verdict or a missing time-to-success answer raises
    ``ValueError`` with the rollout id.
    """
    tts = tts or {}
    out = {}
    for rid, v in verdicts.items():
        try:
            if cutoff_s is None:
                out[rid] = full_adherence(v)
            else:
                out[rid] = within(v, tts.get(rid), cutoff_s)
        except ValueError as e:
            raise ValueError(f"{rid}: {e}") from e
    return out


def summarize(
    runs: Mapping[str, Mapping[str, dict]],
    reference: str | None = None,
    tts: Mapping[str, Mapping[str, dict]] | None = None,
    cutoff_s: float | None = None,
    expected: set[str] | None = None,
) -> dict:
    """The level of each run and, with ``reference``, its paired difference.

    ``runs`` maps a run name to its verdicts. ``expected`` holds the rollout
    ids of the set: a run that misses one of them raises ``ValueError``.
    """
    out: dict = {"cutoff_s": cutoff_s, "runs": {}}
    vals: dict[str, dict[str, float]] = {}
    for name, verdicts in runs.items():
        if expected is not None and set(verdicts) != expected:
            missing = sorted(expected - set(verdicts))[:3]
            extra = sorted(set(verdicts) - expected)[:3]
            raise ValueError(f"{name}: missing {missing}..., unknown {extra}...")
        vals[name] = run_values(verdicts, (tts or {}).get(name), cutoff_s)
        out["runs"][name] = level(vals[name])
    if reference is not None:
        for name in runs:
            if name != reference:
                out["runs"][name]["paired_vs_" + reference] = paired(
                    vals[name], vals[reference]
                )
    return out
