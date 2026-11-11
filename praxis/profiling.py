"""Stage timers for one policy call of the praxis pipeline.

:class:`StageTimer` wraps the callables of a loaded
:class:`praxis.pipeline.PraxisPipeline` in place and measures the wall time
of each call, with a device synchronize before and after it, so each time
holds the GPU work of that stage. The stages of one call are:

* ``text_encode``: the text encoder. Its cache holds each prompt after the
  first call, so this stage is 0 in the later calls of a session.
* ``vae_encode``: the video encodes of the canvas (the current frame, and
  with context the executed frames and the window anchor).
* ``backbone``: one call of the video backbone (one per sampler pass).
* ``expert``: one call of the action expert on the taps of the backbone.
* ``total``: the whole ``Session.step`` call, with the canvas, the pose
  conversions and the command decode.

The times are wall times with the device synchronized, not kernel times,
and the synchronize calls serialize the CPU and the GPU. The sum of the
stages is lower than ``total``. The rest is the work between the stages: the
canvas and the pose conversions, the copies to the GPU, the guidance combine
and the Euler updates of the sampler, and the command decode.

Each timed stage is a named range for ``torch.profiler`` and, when CUDA is
available, an NVTX range, so Nsight Systems shows the stages with
``nsys profile -t cuda,nvtx``.
:func:`summarize` gives the median (p50), the p90 and the max.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from praxis import model as model_module

if TYPE_CHECKING:
    from praxis.pipeline import PraxisPipeline

F = TypeVar("F", bound=Callable)

STAGES: tuple[str, ...] = ("text_encode", "vae_encode", "backbone", "expert")


def synthetic_obs(step: int, stack: int, height: int, width: int) -> dict:
    """Return a synthetic single-arm observation: ``stack`` frames per view.

    The frames are colour gradients of ``height`` x ``width``. The pose moves
    a little with ``step``. The shapes and dtypes are those of a Positronic
    client request with raw images.
    """
    t = step * 24 + np.arange(stack)
    x = np.zeros((stack, height, width, 3), np.uint8)
    x[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, None, :]
    x[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[None, :, None]
    x[..., 2] = (t * 37 % 255)[:, None, None]
    pose = np.zeros((stack, 7), np.float32)
    pose[:, 0], pose[:, 2], pose[:, 3] = 0.4 + 0.0005 * t, 0.4, 1.0
    return {
        "image.exterior": x,
        "image.wrist": x.copy(),
        "robot_state.ee_pose": pose,
        "grip": np.zeros(stack, np.float32),
    }


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Return the count, the median, the p90 and the max of ``values``."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
    }


def summarize_calls(
    calls: Iterable[dict[str, float]], keys: Iterable[str]
) -> dict[str, dict[str, float | int]]:
    """Summarize ``keys`` of the records ``calls``."""
    calls = list(calls)
    return {k: summarize(c[k] for c in calls if k in c) for k in keys}


class StageTimer:
    """Collect the time of each stage of each policy call, in milliseconds.

    ``sync`` waits for the device. The default synchronizes CUDA when it is
    available and does nothing on the CPU. ``nvtx`` opens a named range for
    Nsight Systems. The default is ``torch.cuda.nvtx.range`` when CUDA is
    available, else no range.
    """

    def __init__(
        self,
        sync: Callable[[], None] | None = None,
        nvtx: Callable[[str], AbstractContextManager[object]] | None = None,
    ) -> None:
        cuda = torch.cuda.is_available()
        if sync is None:
            sync = torch.cuda.synchronize if cuda else (lambda: None)
        if nvtx is None:
            nvtx = torch.cuda.nvtx.range if cuda else (lambda _name: nullcontext())
        self.sync = sync
        self.nvtx = nvtx
        self.calls: list[dict[str, float]] = []
        self._current: dict[str, float] | None = None
        self._active_stages: set[str] = set()

    def wrap(self, stage: str, fn: F) -> F:
        """Return ``fn`` with its time added to ``stage`` of the current call."""

        def timed(*args: object, **kwargs: object) -> object:
            if self._current is None or stage in self._active_stages:
                return fn(*args, **kwargs)
            self._active_stages.add(stage)
            try:
                self.sync()
                t0 = time.monotonic()
                with torch.profiler.record_function(stage), self.nvtx(stage):
                    out = fn(*args, **kwargs)
                self.sync()
                self._current[stage] += (time.monotonic() - t0) * 1e3
                self._current[f"{stage}_calls"] += 1
                return out
            finally:
                self._active_stages.remove(stage)

        return timed  # ty: ignore[invalid-return-type]

    @contextmanager
    def call(self) -> Iterator[dict[str, float]]:
        """Time one policy call. The stage times of the call go to ``calls``."""
        record: dict[str, float] = defaultdict(float)
        for stage in STAGES:
            record[stage] = record[f"{stage}_calls"] = 0.0
        self._current = record
        self.sync()
        t0 = time.monotonic()
        try:
            yield record
            self.sync()
            record["total"] = (time.monotonic() - t0) * 1e3
            self.calls.append(dict(record))
        finally:
            self._current = None


@contextmanager
def instrument(pipe: PraxisPipeline, timer: StageTimer) -> Iterator[StageTimer]:
    """Time the stages of a loaded pipeline while the context is open.

    The wrappers replace the text encode, the video encode, the backbone
    call and the expert call of the pipeline. They change no value: each
    wrapper returns the output of the original call. Sessions opened inside
    the context use the timed video encode.
    """
    targets = (
        (pipe.text, "encode", "text_encode"),
        (pipe.vae, "encode", "vae_encode"),
        (pipe.policy.backbone, "forward_chunk_causal", "backbone"),
    )
    # Expert overlap calls the towers directly. The stage guard prevents
    # counting them twice when the ordinary backbone entry point calls them.
    for name in ("encoder", "decoder"):
        tower = getattr(pipe.policy.backbone, name, None)
        if tower is not None and hasattr(tower, "forward_chunk_causal"):
            targets += ((tower, "forward_chunk_causal", "backbone"),)
        if tower is not None and hasattr(tower, "forward_chunk_causal_pair"):
            targets += ((tower, "forward_chunk_causal_pair", "backbone"),)
    # An instance attribute shadows a method until the context closes. An
    # attribute that was an instance attribute already gets its value back.
    saved = [vars(obj).get(name) for obj, name, _stage in targets]
    for obj, name, stage in targets:
        object.__setattr__(obj, name, timer.wrap(stage, getattr(obj, name)))
    expert = model_module.run_expert_on_taps_k
    model_module.run_expert_on_taps_k = timer.wrap("expert", expert)
    try:
        yield timer
    finally:
        for (obj, name, _stage), old in zip(targets, saved, strict=True):
            if old is None:
                object.__delattr__(obj, name)
            else:
                object.__setattr__(obj, name, old)
        model_module.run_expert_on_taps_k = expert
