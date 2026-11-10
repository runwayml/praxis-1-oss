"""The periodic flow: the client samples, the server keeps the history.

A Positronic client can run the recipe ``trajectory_player | periodic(fps) |
inference_schedule`` with a call on every sample and no ``temporal_stack``
(https://github.com/Positronic-Robotics/positronic/pull/1139).
It then sends each sampled observation as an ordinary call that carries only
the current frames and state, with the grid index ``sample_index`` and the
sample time ``obs_time_ns``. The server keeps the history and decides when to
predict. It answers None on the other samples. Calls overlap, up to the
``max_in_flight`` that the ready message of the server declares
(https://github.com/Positronic-Robotics/positronic/pull/1153).

The session samples at ``ratio`` times the action rate (15 Hz by default, the
action rate: ``ratio`` 1). The history and the prediction grid work on the
sample index, not on the clock:

* :class:`SampleHistory` rebuilds, for a prediction at sample ``n``, the frame
  stack that the chunked recipe declares: row ``b`` rows back is the latest
  sample at or before index ``n - ratio * b``. On a grid without skips these
  are the samples of the 15 Hz frame grid of training. A late tick that
  skipped grid positions falls back to the sample before. A pick by time would
  move a pick to the next frame whenever the sampler jitters by a few ms.
* :class:`PredictGrid` predicts on the first sample, then ``ratio * stride``
  samples after the previous prediction, or on the first sample past that
  index when it was skipped. A replan predicts on its sample.
* :class:`CallTimings` keeps the durations of the prediction calls for the
  delay estimate. The client reports every call, and the None answers take a
  few ms, so an estimate over all of them runs short.

The client sends from several worker threads, so a sample can arrive after a
later one. It answers None and changes the history only for a reset (see
:meth:`SampleHistory.drop_before`).
"""

from __future__ import annotations

import collections.abc as cabc
import copy
import math
from dataclasses import dataclass, field

import numpy as np

# The observation keys of the Positronic recipe.
OBS_TIME_NS = "obs_time_ns"
SAMPLE_INDEX = "sample_index"
PLAN = "plan"
REPLAN_NOW = "replan_now"
CALL_ID = "call_id"
CALL_TIMINGS = "call_timings"
ELAPSED_SEC = "elapsed_sec"

# The action rate: each sample is a row of the window, the 15 Hz frame grid of
# training. A prediction takes about 0.63 to 0.69 s on an H100 with real
# weights, and the samples queue behind it on the one inference thread. At
# 30 Hz with 8 calls in flight the client skipped about half the samples
# during each prediction, and about half the rows of each window fell back to
# an older sample. At 15 Hz with 16 in flight it skipped none.
DEFAULT_SAMPLE_FPS = 15.0
# The history keeps ``ratio`` samples per window row, each with its views.
MAX_SAMPLE_FPS = 60.0
# The calls in flight: 16 samples at 15 Hz cover a prediction of about 1 s.
DEFAULT_MAX_IN_FLIGHT = 16
MAX_IN_FLIGHT_RANGE = (1, 16)
# How long past the end of the first chunk the grid waits for the duration of
# the first prediction before it predicts anyway.
MAX_WAIT_SEC = 1.0


@dataclass(frozen=True)
class PeriodicConfig:
    """The sampling rate (a whole multiple of the action rate) and the call limit."""

    sample_fps: float = DEFAULT_SAMPLE_FPS
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT

    def __post_init__(self) -> None:
        fps = float(self.sample_fps)
        if not math.isfinite(fps) or not 0 < fps <= MAX_SAMPLE_FPS:
            raise ValueError(
                f"sample_fps must be in (0, {MAX_SAMPLE_FPS:g}], got {self.sample_fps}"
            )
        lo, hi = MAX_IN_FLIGHT_RANGE
        if not lo <= int(self.max_in_flight) <= hi:
            raise ValueError(
                f"max_in_flight must be {lo}..{hi}, got {self.max_in_flight}"
            )

    def ratio(self, fps: float) -> int:
        """Return the samples per action row. ``sample_fps`` must be a whole multiple of ``fps``."""
        r = float(self.sample_fps) / float(fps)
        if r < 1 or not math.isclose(r, round(r), abs_tol=1e-6):
            raise ValueError(
                f"sample_fps={self.sample_fps:g} must be a whole multiple of the "
                f"action rate {float(fps):g}"
            )
        return round(r)


def history_capacity(rows: int, ratio: int) -> int:
    """Return the samples that a history keeps for ``rows`` rows back: plus a chunk of slack."""
    return int(ratio) * (int(rows) + 24) + 4


@dataclass
class SampleHistory:
    """The stack keys of the recent samples by sample index, and the stack rule."""

    keys: tuple[str, ...]
    # The window rows before the current frame.
    history_rows: int
    ratio: int
    # The samples kept: the span of the stack plus a chunk of slack.
    capacity: int = 0
    _indices: list[int] = field(default_factory=list)
    _values: list[dict[str, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            self.capacity = history_capacity(self.history_rows, self.ratio)

    def reset(self) -> None:
        self._indices.clear()
        self._values.clear()

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def newest_index(self) -> int | None:
        return self._indices[-1] if self._indices else None

    def append(self, obs: cabc.Mapping, index: int) -> None:
        """Record the stack keys of one sample. The index must increase."""
        if self._indices and index <= self._indices[-1]:
            raise ValueError(
                f"{SAMPLE_INDEX} must increase: {index} after {self._indices[-1]}"
            )
        missing = [k for k in self.keys if k not in obs]
        if missing:
            raise ValueError(f"a periodic sample needs {missing} (the stack keys)")
        self._indices.append(int(index))
        self._values.append({k: np.asarray(obs[k]) for k in self.keys})
        if len(self._indices) > self.capacity:
            del self._indices[: -self.capacity]
            del self._values[: -self.capacity]

    def drop_before(self, index: int) -> None:
        """Forget the samples before ``index``: a jump that the client signalled at ``index``."""
        i = int(np.searchsorted(self._indices, index, side="left"))
        del self._indices[:i]
        del self._values[:i]

    def at(self, index: int) -> dict[str, np.ndarray] | None:
        """Return the latest sample at or before ``index``, or None."""
        i = int(np.searchsorted(self._indices, index, side="right")) - 1
        return None if i < 0 else self._values[i]

    def picks(self, index: int) -> list[dict[str, np.ndarray]]:
        """Return the sample of each stack row at sample ``index``, oldest first.

        Rows before the first kept sample are left out, as the stack of the
        chunked recipe (``pad_start=False``) leaves them out.
        """
        rows = [
            self.at(index - self.ratio * b) for b in range(self.history_rows, -1, -1)
        ]
        return [r for r in rows if r is not None]

    def stacked(self, obs: cabc.Mapping, index: int) -> dict:
        """Return ``obs`` with every stack key replaced by its ``(T, ...)`` stack at ``index``."""
        picked = self.picks(index)
        if not picked:
            raise ValueError("no sample recorded yet")
        out = dict(obs)
        for k in self.keys:
            out[k] = np.stack([r[k] for r in picked])
        return out


@dataclass
class PredictGrid:
    """Which samples predict, as Positronic's real-time schedule times its calls.

    The first prediction after a reset runs at once. Its chunk starts when its
    answer is read, so the next prediction is ``rows`` rows after that. Every
    later chunk starts at its observation, so the next prediction is ``rows``
    rows after it. A row is ``ratio`` samples.

    The read time of the first answer is the duration of its call, which the
    client reports on a later call (``call_timings``, :meth:`answered`). The
    samples wait until it arrives. Without a report, the grid predicts
    :data:`MAX_WAIT_SEC` after the first chunk would have ended, and times the
    next predictions from that one.
    """

    ratio: int
    sample_fps: float
    next_index: int | None = None
    last_index: int | None = None
    # (index, rows, call_id) of a first prediction that waits for its duration.
    _waiting: tuple[int, int, int] | None = None

    def reset(self) -> None:
        self.next_index = None
        self.last_index = None
        self._waiting = None

    def due(self, index: int) -> bool:
        if self._waiting is not None:
            first, rows, _ = self._waiting
            wait = round(MAX_WAIT_SEC * self.sample_fps)
            if index < first + self.ratio * rows + wait:
                return False
            self._waiting = None
            self.next_index = int(index)
            return True
        return self.next_index is None or index >= self.next_index

    def predicted(self, index: int, rows: int, call_id: int | None) -> None:
        """Record a prediction at sample ``index`` that the client plays for ``rows`` rows."""
        rows = max(int(rows), 1)
        first = self.next_index is None
        self.last_index = int(index)
        if first and call_id is not None:
            self._waiting = (int(index), rows, int(call_id))
            return
        self.next_index = int(index) + self.ratio * rows

    def answered(self, call_id: int, elapsed_sec: float) -> None:
        """Take the duration of a completed call. The first prediction's anchors the grid."""
        if self._waiting is None or int(call_id) != self._waiting[2]:
            return
        index, rows, _ = self._waiting
        self._waiting = None
        wait = round(float(elapsed_sec) * self.sample_fps)
        self.next_index = index + self.ratio * rows + wait


@dataclass
class CallTimings:
    """The reported durations of the prediction calls of a session.

    Each prediction call is reported once, on a later call. :meth:`take`
    hands the durations reported since the last prediction to the next one.
    """

    predicted: set[int] = field(default_factory=set)
    _reported: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.predicted.clear()
        self._reported.clear()

    def collect(self, obs: cabc.Mapping) -> list[tuple[int, float]]:
        """Return ``(call_id, elapsed_sec)`` of the prediction calls that this sample reports."""
        timings = obs.get(CALL_TIMINGS)
        if timings is None:
            return []
        if not isinstance(timings, list | tuple):
            raise ValueError(
                f"{CALL_TIMINGS} must be a list of {{call_id, elapsed_sec}}"
            )
        out: list[tuple[int, float]] = []
        for t in timings:
            if not isinstance(t, cabc.Mapping):
                raise ValueError(f"{CALL_TIMINGS} entries must be mappings")
            call_id = t.get(CALL_ID)
            if (
                isinstance(call_id, bool)
                or not isinstance(call_id, int | np.integer)
                or call_id not in self.predicted
            ):
                continue
            elapsed = t.get(ELAPSED_SEC)
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, int | float | np.number)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
            ):
                raise ValueError(f"{CALL_TIMINGS}.{ELAPSED_SEC} must be finite seconds")
            self.predicted.discard(call_id)
            out.append((int(call_id), float(elapsed)))
        self._reported.extend(e for _, e in out)
        return out

    def take(self) -> list[float]:
        """Return the durations reported since the last call of :meth:`take`."""
        out, self._reported = self._reported, []
        return out


def client_recipe(base: dict, *, fps: float, sample_fps: float) -> dict:
    """Return ``base`` (the declared ``seq`` of the chunked recipe) as the periodic recipe.

    ``temporal_stack`` goes. ``chunked_schedule`` becomes
    ``trajectory_player | periodic | inference_schedule`` with a call on
    every sample. Each prediction starts at its observation and the plan goes
    out with each call (``plan_key``). The server times the predictions, so
    the latency estimate of the client is a constant 0. The other entries keep
    their place, so ``stop_on_fault`` stays outermost and each sample is
    resized once. Every entry carries its version: protocol v2 reads an entry
    without one as version 1.
    """
    versions = {"stop_on_fault": 2, "restrict_image_size": 1}
    out: list[dict] = []
    for entry in base["seq"]:
        name = entry.get("name")
        if name == "temporal_stack":
            continue
        if name == "chunked_schedule":
            out += [
                {
                    "name": "trajectory_player",
                    "version": 1,
                    "args": {"plan_key": PLAN, "replan_on_error": True},
                },
                {
                    "name": "periodic",
                    "version": 1,
                    "args": {"fps": float(sample_fps), "obs_time_key": OBS_TIME_NS},
                },
                {
                    "name": "inference_schedule",
                    "version": 1,
                    "args": {
                        "fps": float(fps),
                        "obs_time_key": OBS_TIME_NS,
                        "request_timing": {
                            "reference": "every",
                            "seconds": 1.0 / float(sample_fps),
                        },
                        "merge": {"start": "observation", "horizon_sec": None},
                        "latency_estimate": 0.0,
                    },
                },
            ]
            continue
        entry = copy.deepcopy(entry)
        if name in versions:
            entry.setdefault("version", versions[name])
        out.append(entry)
    return {"seq": out}
