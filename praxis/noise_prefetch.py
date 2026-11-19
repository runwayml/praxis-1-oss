"""Draw the sampling noise of the next policy call ahead of the call.

Each ``window_joint`` call draws its initial noise on the CPU with
:func:`praxis.sampler.seeded_normal`: the video noise ``(1, 192, 6 K', h, w)``
and the action noise ``(6 K', A)`` of the seed ``base + chunk_index``. For the
DROID canvas (``h, w = 8, 26``) the video noise holds 239,616 float32 values
per window chunk, 0.96 MB, drawn by one CPU generator on the call's thread.
The call then copies them to pinned memory before its non-blocking copy to
the GPU (:func:`praxis.layers.to_device_async`).

Every draw seeds a fresh ``torch.Generator`` from ``(seed, salt)``. So the
noise is a pure function of the seed, the salt and the shape. No generator
state carries from one call to the next, and the global torch generator is
not read. A draw made ahead of the call on another thread therefore gives the
bytes of the draw in the call, whatever happened in between (a reset, a
failed call, a seed change).

:class:`NoisePrefetch` keeps one speculative draw per session. The session
starts it after the GPU work of a call is queued (``Session._sample``), with
the seed and the window depth that the next call is expected to have. The
next call takes it only when its seed and both shapes equal the key of the
draw. Otherwise it drops the draw and draws in the call, as without the
prefetch.

Only calls that sample through ``PraxisPolicy.sample_window_joint`` take a
draw or start one. With the shipped serve configs (``schedule: chunked``)
that is every call. A real-time chunking session (``schedule: rtc``, opt-in)
whose call finishes a plan made ahead of it (``finish_window_joint``) draws
no noise in the call and starts no draw; its next call draws itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from torch import Tensor

from praxis.sampler import ACTION_SALT, VIDEO_SALT, seeded_normal

log = logging.getLogger(__name__)

#: One worker for all sessions of the process. It runs only CPU work (the
#: generator draw and the copy into pinned memory). ``torch.randn`` releases
#: the GIL while it fills the tensor.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="praxis-noise")

NoiseKey = tuple[int, tuple[int, ...], tuple[int, ...]]


def draw_window_noise(
    seed: int,
    video_shape: Sequence[int],
    action_shape: Sequence[int],
    pin: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return the video and the action noise of one ``window_joint`` call.

    The values come from :func:`praxis.sampler.seeded_normal` with
    :data:`VIDEO_SALT` and :data:`ACTION_SALT`, as in the call. ``pin`` copies
    them into page-locked memory (exact copy). The values, dtype (float32),
    shape and strides stay those of the call's own draw.
    """
    noise_v = seeded_normal(video_shape, seed, VIDEO_SALT)
    noise_a = seeded_normal(action_shape, seed, ACTION_SALT)
    if pin:
        noise_v, noise_a = noise_v.pin_memory(), noise_a.pin_memory()
    return noise_v, noise_a


class NoisePrefetch:
    """One speculative noise draw of the next call of one session.

    ``start`` submits the draw of a key to the worker and drops the previous
    one. ``take`` returns the draw of a key, or None when the session holds
    no draw of that key (the call then draws itself). A taken draw is gone:
    each draw serves one call.
    """

    def __init__(self, pin: bool = False) -> None:
        self.pin = bool(pin)
        self.key: NoiseKey | None = None
        self.future: Future | None = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        seed: int, video_shape: Sequence[int], action_shape: Sequence[int]
    ) -> NoiseKey:
        return (
            int(seed),
            tuple(int(s) for s in video_shape),
            tuple(int(s) for s in action_shape),
        )

    def start(
        self, seed: int, video_shape: Sequence[int], action_shape: Sequence[int]
    ) -> None:
        """Draw the noise of ``seed`` and these shapes on the worker."""
        self.drop()
        self.key = self.make_key(seed, video_shape, action_shape)
        self.future = _EXECUTOR.submit(draw_window_noise, *self.key, self.pin)

    def drop(self) -> None:
        """Forget the pending draw (a queued draw is cancelled)."""
        if self.future is not None:
            self.future.cancel()
        self.key = None
        self.future = None

    def take(
        self, seed: int, video_shape: Sequence[int], action_shape: Sequence[int]
    ) -> tuple[Tensor, Tensor] | None:
        """Return the draw of this key, or None.

        A draw of another key is dropped. A draw that the worker has not
        started is cancelled and gives None: the call draws itself, no later
        than it would without the prefetch. A draw that is running is waited
        for: it ends before a new draw would. A worker error gives None and a
        warning.
        """
        key = self.make_key(seed, video_shape, action_shape)
        future = self.future
        if future is None or self.key != key:
            self.drop()
            self.misses += 1
            return None
        self.key = None
        self.future = None
        if future.cancel():
            self.misses += 1
            return None
        try:
            noise = future.result()
        except Exception:
            log.warning("noise prefetch failed; drawing in the call", exc_info=True)
            self.misses += 1
            return None
        self.hits += 1
        return noise
