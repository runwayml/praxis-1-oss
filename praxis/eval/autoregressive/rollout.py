"""The video-only self-rollout of Praxis-1 for the autoregressive eval.

The rollout starts from one real first frame and a prompt. Each chunk call
samples the next video chunk with
:meth:`praxis.model.PraxisPolicy.sample_window_joint`, the sampler of the
policy. The history of a call is the model's own decoded 15 fps frames, encoded
again on every call. No ground-truth frame or action enters after frame 0.

One call (:meth:`VideoRollout.step`):

1. The history window (:func:`history_window`). The first call reads frame
   0 only (``k = 0``). Later calls read ``k = min(max_chunks - 1, (n + 3) //
   24)`` chunks of the ``n`` frames so far, from frame ``w = n - 24 k + 3``.
   The window holds ``24 k - 3`` frames, so it encodes to ``6 k`` latents:
   the anchor latent and ``k`` clean chunks.
2. The encode: the anchor is the encode of frame ``w`` alone, the clean
   chunks are the 6-latent slices of the encode of the window.
3. The sample: ``sample_window_joint`` at depth ``k``, seed
   ``1000 * seed + chunk`` (:func:`chunk_seed`), with the encoded negative
   prompt as the text of the unguided pass. The eval sampler guides the
   video only (``cfg_video_scale`` 3.0, ``cfg_scale`` 1.0).
4. The decode of ``[clean chunks | new chunk]``, rounded to uint8 by
   ``round((x + 1) * 127.5)``. The call keeps the last 21 frames (first
   call; they replace frame 0) or the last 24 frames.

8 calls give 21 + 7 x 24 = 189 frames, 12.6 s at 15 fps.

The video does not depend on the proprio or on the actions: the backbone
reads no action input, and only the action expert reads the backbone
(``sample_window_joint``, ``velocity``). The rollout therefore passes zero
proprio rows and drops the sampled actions.

The noise of the video of a call is ``seeded_normal(shape, 1000 * seed +
chunk, VIDEO_SALT)``, the scheme of the reference rollouts: same generator,
same seed, same salt, same shape. The reference sampled the new chunk with a
KV cache over the context; this rollout runs the full window. Under the
chunk-causal masks the two are the same function, but the attention kernels
differ, so the outputs are not bitwise equal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from praxis import bimanual, single_arm
from praxis.model import PraxisPolicy, SamplerSettings

CHUNK_ROWS = 24
FIRST_CHUNK_ROWS = 21
CHUNKS = 8
MAX_CHUNKS = 4
SEED_STRIDE = 1000
NEGATIVE_PROMPT = (
    "The robot does not move. Static scene, blurry, flickering, objects morphing, "
    "duplicated objects."
)
#: The sampler of the eval: 10 Euler steps, flow shift 8, video guidance 3.
EVAL_SAMPLER = SamplerSettings(
    num_steps=10, t_shift=8.0, cfg_scale=1.0, cfg_video_scale=3.0
)


def chunk_seed(seed: int, chunk: int) -> int:
    return SEED_STRIDE * int(seed) + int(chunk)


def history_window(n_frames: int, first_call: bool, max_chunks: int) -> tuple[int, int]:
    """Return ``(w, k)``: the first frame of the window and its clean chunks."""
    if first_call:
        return 0, 0
    k = min(max_chunks - 1, (n_frames + 3) // CHUNK_ROWS)
    w = n_frames - CHUNK_ROWS * k + 3
    if w < 0 or k < 1:
        raise ValueError(f"no history window for {n_frames} frames")
    return w, k


def first_frame_clip(frame: np.ndarray, config: dict) -> np.ndarray:
    """Rebuild the first-frame canvas with the tile code of the rig.

    ``frame`` is the canvas of the set at the tile size of the checkpoint.
    Each view goes through the tile rule of its rig
    (:func:`praxis.single_arm.tile_clip`, :func:`praxis.bimanual.canvas_clip`).
    At the tile size the rule keeps every pixel. A canvas of another size, or
    one that the rule changes, raises ``ValueError``.
    """
    bim = config.get("embodiment") == "bimanual"
    th, cw = (bimanual if bim else single_arm).canvas_size(config)
    if frame.shape != (th, cw, 3):
        raise ValueError(f"canvas {frame.shape}, the checkpoint needs {(th, cw, 3)}")
    views = bimanual.VIEWS if bim else single_arm.VIEWS
    tw = cw // len(views)
    tiles = {v: frame[None, :, i * tw : (i + 1) * tw] for i, v in enumerate(views)}
    if bim:
        clip = bimanual.canvas_clip(tiles, th, tw)
    else:
        clip = single_arm.tile_clip(tiles["exterior"], tiles["wrist"], th, tw)
    out = clip[0, :, 0].permute(1, 2, 0).round().to(torch.uint8).numpy()
    if not np.array_equal(out, frame):
        raise ValueError("the tile rule of the rig changes this canvas")
    return out


def to_uint8(video: Tensor) -> np.ndarray:
    """``(1, 3, T, H, W)`` in [-1, 1] -> ``(T, H, W, 3)`` uint8, ``round((x + 1) * 127.5)``."""
    v = (video[0].float().clamp(-1, 1) + 1) * 127.5
    return v.permute(1, 2, 3, 0).round().to(torch.uint8).cpu().numpy()


def frames_clip(frames: np.ndarray, device: torch.device | str) -> Tensor:
    """``(T, H, W, 3)`` uint8 -> ``(1, 3, T, H, W)`` float32 in [0, 255]."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2)[None]
    return t.to(device).float()


@dataclass
class VideoRollout:
    """The rollout of one first frame.

    ``encode`` maps a clip ``(1, 3, T, H, W)`` in [0, 255] to the policy
    latents, ``decode_unit`` maps latents to ``(1, 3, T', H, W)`` in [-1, 1]
    (:class:`praxis.vae.Autoencoder`). ``text`` and ``negative`` are the
    encoded prompt and negative prompt.
    """

    policy: PraxisPolicy
    encode: Callable[[Tensor], Tensor]
    decode_unit: Callable[[Tensor], Tensor]
    first_frame: np.ndarray
    seed: int = 0
    max_chunks: int = MAX_CHUNKS
    settings: SamplerSettings = field(default_factory=lambda: EVAL_SAMPLER)
    frames: list[np.ndarray] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.frames = [np.ascontiguousarray(self.first_frame)]

    @torch.no_grad()
    def step(self, text: Tensor, negative: Tensor | None) -> np.ndarray:
        """Sample one chunk and return its new frames."""
        policy = self.policy
        chunk = len(self.log)
        w, k = history_window(len(self.frames), chunk == 0, self.max_chunks)
        window = np.stack(self.frames[w : len(self.frames)] if k else self.frames[:1])
        dev = policy.device
        anchor = self.encode(frames_clip(window[:1], dev))
        cleans: list[Tensor] = []
        if k:
            lat = self.encode(frames_clip(window, dev))
            n = int(policy.config["chunking"]["latents_per_chunk"])
            cleans = [lat.narrow(2, j * n, n).contiguous() for j in range(k)]
        p_dim = int(policy.config["action_expert"]["proprio_patch_dim"])
        video, _ = policy.sample_window_joint(
            text=text,
            text_uncond=negative,
            anchor=anchor,
            cleans=cleans,
            proprio_all=torch.zeros(1, k + 1, p_dim),
            seed=chunk_seed(self.seed, chunk),
            settings=self.settings,
        )
        full = torch.cat([*(c.float() for c in cleans), video.to(dev).float()], dim=2)
        img = to_uint8(self.decode_unit(full))
        n_new = FIRST_CHUNK_ROWS if k == 0 else CHUNK_ROWS
        new = img[-n_new:]
        if chunk == 0:
            self.frames = list(new)
        else:
            self.frames.extend(list(new))
        self.log.append({"chunk": chunk, "k_hist": k, "window_start": w, "new": n_new})
        return new

    def video(self) -> np.ndarray:
        return np.stack(self.frames)


def write_mp4(frames: np.ndarray, path: str | Path, fps: int = 15) -> None:
    """H.264 mp4 (libx264, CRF 18, yuv420p), the encode of the reference videos."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape
    with av.open(str(path), mode="w") as out:
        st = out.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
        st.options = {"crf": "18"}
        for f in frames:
            fr = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(f[: st.height, : st.width]), format="rgb24"
            )
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
