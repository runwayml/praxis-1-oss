"""The overlap precompute of a real-time session (``precompute``, on by default).

A prediction encodes its window (the frames that the robot ran since the
context started, one per row), then runs the trunk passes of both Euler
steps and the expert passes. The window rows are samples that arrive before
the prediction, and no trunk pass reads the actions or the proprio
(:meth:`praxis.model.PraxisPolicy.plan_window_joint`). So a session with
``precompute`` does this work on the samples before the prediction, on the
inference thread:

* After a prediction the next one is due at ``PredictGrid.next_index``. Row
  ``b`` of its window is the sample ``SampleHistory.at(next - ratio * b)``,
  the row that the frame stack of the prediction holds.
* Each row goes into a streamed encode of the window (:class:`StreamEncode`)
  as soon as its sample is in. The window start, the anchor frame, is also
  encoded alone, as the call does.
* When the last row is in, one row before the prediction, the session runs
  the trunk passes on the window (:meth:`praxis.pipeline.Session.plan`).
  When that row never arrives (the client skipped its sample), the
  prediction finishes the window and runs the trunk passes itself.
* With ``skip_video_decoder=on`` the plan runs the last step without the
  video decoder: the server returns no video, and the actions read only the
  encoder taps, so they keep their bits.
* A session with ``cfg_batched=on`` makes no plan: the plan runs the guided
  and the unguided passes one after the other. The window encode still runs
  ahead. Its predictions report ``none`` and sample the trunk passes in the
  call.
* The work is for one prediction. A prediction at another sample (a replan,
  a skipped prediction sample), a reset or a replan drops it. So the session
  holds at most one window and one plan.
* The prediction checks each value before it uses it: a latent only for a
  clip equal to the streamed one, the trunk passes only when the anchor, the
  cleans, the text, the seed and the settings are equal
  (:meth:`praxis.model.TrunkPlan.matches`). Otherwise it computes them as
  without the precompute.

The answers keep their bits. The streamed encode runs the chunks of
``Autoencoder.encode`` in the same order on the same buffers, and the plan
runs the backbone calls of ``sample_window_joint`` on the same inputs. The
trunk passes need the last window row, so only the work before that row
(the encode) leaves the prediction. The trunk passes start one row before
the prediction, and the prediction waits for the rest of them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from praxis.pipeline import Session
from praxis.serve.periodic import SampleHistory
from praxis.vae import StreamEncode

logger = logging.getLogger(__name__)


class _BufferedEncode:
    """The encode of a window without a fused encoder: one call once all frames are in."""

    def __init__(self, encode: Callable[[Tensor], Tensor], n_frames: int) -> None:
        self.encode = encode
        self.n = int(n_frames)
        self.frames: list[Tensor] = []

    @property
    def done(self) -> bool:
        return len(self.frames) == self.n

    def add(self, frame: Tensor) -> None:
        self.frames.append(frame)

    def latent(self) -> Tensor:
        return self.encode(torch.cat(self.frames, dim=2))


def prompt_of(obs: dict, current: str) -> str:
    """Return the prompt that a call on ``obs`` uses: its override, else ``current``."""
    override = obs.get("task") or obs.get("prompt")
    if override:
        if isinstance(override, bytes):
            override = override.decode("utf-8", "replace")
        new = str(override).strip()
        if new:
            return new
    return current


class NextWindow:
    """The window encode and the trunk passes of the next prediction of a session."""

    def __init__(
        self,
        session: Session,
        history: SampleHistory,
        ratio: int,
        skip_last_video: bool = False,
    ) -> None:
        if session.memo is None:
            raise ValueError("the precompute needs a session opened with precompute")
        self.session = session
        self.history = history
        self.ratio = int(ratio)
        self.skip_last_video = bool(skip_last_video)
        # The sample index of the prediction that the work is for, and how
        # its plan was made: "ahead" (on a sample before the prediction) or
        # "late" (in the prediction).
        self.target: int | None = None
        self.planned: str | None = None
        self._skip = False
        self._depth = 0
        self._next_row = 0
        self._frames: list[Tensor] = []
        self._encode: StreamEncode | _BufferedEncode | None = None
        self._anchor: Tensor | None = None
        self._events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None

    def drop(self) -> None:
        """Drop the work for the next prediction, and the plan of the session."""
        self.target = None
        self.planned = None
        self._skip = False
        self._depth = 0
        self._next_row = 0
        self._frames = []
        self._encode = None
        self._anchor = None
        self._events = None
        self.session.drop_plan()

    def plan_gpu_ms(self) -> float | None:
        """Return the GPU time of the trunk passes of the plan, once they ran."""
        if self._events is None:
            return None
        start, end = self._events
        end.synchronize()
        return round(start.elapsed_time(end), 1)

    def feed(
        self,
        obs: dict,
        next_index: int | None,
        last_index: int | None,
        pending_reset: bool,
    ) -> None:
        """Take the sample just recorded (not a prediction) and run the work it completes.

        ``next_index`` is the sample of the next prediction and ``last_index``
        that of the last one. A failure skips the precompute of this
        prediction: the prediction then computes everything itself.
        """
        if next_index is None or last_index is None or pending_reset:
            if self.target is not None:
                self.drop()
            return
        if next_index != self.target:
            self.drop()
            self.target = int(next_index)
            self._guarded(self._start, int(next_index) - int(last_index))
        self._guarded(self._advance, obs, "ahead")

    def before_prediction(self, obs: dict, index: int) -> str | None:
        """Prepare the plan of the prediction at sample ``index`` (just recorded).

        Work for another sample is dropped: returns ``"stale"`` if it had
        started. A window that is not complete is finished now, and its
        trunk passes run now (``"late"``). Returns None otherwise.
        """
        if self.target is None:
            return None
        if self.target != index:
            self.drop()
            return "stale"
        self._guarded(self._advance, obs, "late")
        return None

    def _guarded(self, fn: Callable[..., None], *args: object) -> None:
        if self._skip or self.planned is not None:
            return
        try:
            fn(*args)
        except Exception:
            logger.warning(
                "precompute of sample %s skipped", self.target, exc_info=True
            )
            target = self.target
            self.drop()
            self.target = target
            self._skip = True

    def _start(self, samples: int) -> None:
        """Set the window of the next prediction, ``samples`` after the last one."""
        state = self.session.state
        stream = state.stream
        rows = samples // self.ratio
        # A stream that is empty or past its stack refills from the stack of
        # the call, and a stateless call has no window: no precompute.
        if state.stateless or stream.n_raws == 0 or rows > self.history.history_rows:
            self._skip = True
            return
        self._depth = stream.depth_after(rows)
        if self._depth == 0:
            self._skip = True
            return
        n = stream.window_raws(self._depth)
        self._next_row = n
        vae = self.session.pipe.vae
        fused = getattr(vae, "fused_encoder", None)
        if fused is None:
            assert self.session.memo is not None
            self._encode = _BufferedEncode(self.session.memo.encode, n)
        else:
            self._encode = StreamEncode(fused(), n)

    def _advance(self, obs: dict, when: str) -> None:
        """Encode the window rows that are in; plan when the last one is."""
        memo = self.session.memo
        assert memo is not None and self.target is not None and self._encode
        newest = self.history.newest_index
        while (
            self._next_row >= 1
            and newest is not None
            and newest >= self.target - self.ratio * self._next_row
        ):
            row = self.history.at(self.target - self.ratio * self._next_row)
            if row is None:
                self._skip = True
                return
            frame = self.session.state.frame_clip(row)
            if not self._frames:
                self._anchor = memo.store(frame)
            self._frames.append(frame)
            self._encode.add(frame)
            self._next_row -= 1
        if self._next_row == 0:
            self._plan(obs, when)

    def _plan(self, obs: dict, when: str) -> None:
        memo = self.session.memo
        assert memo is not None and self._encode is not None
        assert self._anchor is not None
        latent = self._encode.latent()
        memo.put(torch.cat(self._frames, dim=2), latent)
        self._frames = []
        self._encode = None
        n = self.session.state.stream.lat_per_chunk
        cleans = [latent.narrow(-3, k * n, n).contiguous() for k in range(self._depth)]
        cuda = latent.device.type == "cuda"
        if cuda:
            self._events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._events[0].record()
        state = self.session.state
        self.session.plan(
            self._anchor, cleans, prompt_of(obs, state.prompt), self.skip_last_video
        )
        if self._events is not None:
            self._events[1].record()
        self.planned = when
