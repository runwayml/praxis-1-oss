"""Offline action error of the single-arm checkpoint on fixed DROID windows.

For each window, one teacher-forced sampler call predicts the actions of all
4 chunks. The call has the layout of training: the 4 chunks are noised at
the same time, and chunk ``k`` reads the clean recorded chunks before it
(:meth:`praxis.model.PraxisPolicy.sample_window_joint` at depth 3 with
``return_window``). Each window runs two legs with the same seed:

* ``oracle``: before each sampler step the video state is the noised
  recorded video, ``(1 - t) * x0 + t * noise`` (``oracle_video``). Only the
  actions integrate, so the actions read the true future video. This
  measures the action readout alone.
* ``deployment``: the video and the actions integrate jointly from noise, as
  at inference. ``deployment - oracle`` is the cost of the imperfect video
  prediction.

By default the sampler is the sampler of the original implementation: 5 Euler
steps, the flow shift 1.0 and no text guidance (:data:`EVAL_SAMPLER`). This
is not the recommended serve recipe (2 steps, flow shift 12.5, video
guidance 4.0, ``configs/serve_droid.yaml``), so by default the
``deployment`` leg measures the eval sampler, not the served policy.
``settings`` selects another sampler, for example the serve recipe
(:meth:`praxis.model.SamplerSettings.from_config` of the ``sampler``
section of ``configs/serve_droid.yaml``). A guided sampler needs
``text_uncond``, the encoded blank caption of the checkpoint config. The
context stays the teacher-forced window of training in every case: chunk
``k`` reads the ``k`` clean chunks before it, while a served session keeps
``max_context_chunks`` (1 in the serve recipe) executed chunks. Window ``i`` samples with the seed
``1000 * seed + i`` (:func:`window_seed`), so a run holds at most 1000
windows.

Metrics of a chunk (:func:`chunk_metrics`), from the decoded rows: each row
is relative to the observed pose at the start of its chunk, chunk 0 has 21
rows and chunks 1 to 3 have 24 rows.

* ``endpoint_err_cm``: the xyz distance between the predicted and the
  recorded row at the last row of the chunk, in cm (the headline).
* ``rot_endpoint_deg``: the geodesic angle between the predicted and the
  recorded relative rotation at the last row.
* ``grip_mae``: the mean absolute gripper error over the rows of the chunk.
* ``gt_motion_cm``: the recorded xyz motion of the chunk, from its first to
  its last row. The endpoint error must be read against it.

The ground truth is the recorded relative rows, not clipped. The prediction
is decoded with the inverse q99 map of the checkpoint statistics, not clipped
(:func:`decode_chunks`). :func:`summarize` gives the mean and the standard
error over the windows, per chunk and for the mean of chunks 1 to 3. Chunk
0 has no clean history.

The episodes (``droid_val_episodes.json``) are 19 successful DROID 1.0.1
episodes of the ``"val"`` split of
:class:`praxis.train.data.DroidWindowDataset` (``episode_index % 50 == 0``),
so a fine-tune on the default split (the default ``data.spec``) has
not trained on them. They are the val episodes that the training data of the
checkpoint also held out, less episode 50050: the original implementation cannot
start a window at its frame 24, the window of the parity checks. The list
holds every such episode, in ``episode_index`` order. The episodes have 135
to 629 frames.

Each episode gives a fixed number of 93-frame windows that spread over the
episode (:func:`stratified_anchors`, one seeded generator per episode,
:func:`episode_rng`). The prompt is the language instruction of the episode.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from praxis.model import PraxisPolicy, SamplerSettings
from praxis.robot_io import action_rows, contiguous_chunk_geometry, q99_denormalize

if TYPE_CHECKING:
    from praxis.train.data import TrainingWindow
    from praxis.train.objective import EncodedWindow

WINDOWS_FILE = Path(__file__).with_name("droid_val_episodes.json")
LEGS = ("deployment", "oracle")
CHUNK_METRICS = ("endpoint_err_cm", "rot_endpoint_deg", "grip_mae", "gt_motion_cm")
#: The sampler of the evaluation: 5 Euler steps, flow shift 1.0, no guidance.
EVAL_SAMPLER = SamplerSettings(
    num_steps=5, t_shift=1.0, cfg_scale=1.0, cfg_video_scale=1.0
)
#: Window ``i`` samples with the seed ``SEED_STRIDE * seed + i``.
SEED_STRIDE = 1000
XYZ = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9


def load_windows_file(path: str | Path = WINDOWS_FILE) -> dict:
    """Return the episode list: ``source``, ``revision`` and ``episodes``."""
    return json.loads(Path(path).read_text())


def stratified_anchors(
    n_frames: int,
    window_frames: int,
    fps: float,
    n_windows: int,
    rng: np.random.Generator,
) -> list[int]:
    """Draw ``n_windows`` first frames that spread over one episode.

    The start range ``[0, duration - window]`` of
    :func:`praxis.train.data.draw_anchor` splits into ``n_windows`` equal
    parts, and each part gets one uniform start, snapped down to the frame
    grid. The windows thus cover the start, the middle and the end of the
    episode. ``rng`` is the generator of the episode.
    """
    from praxis.train.data import TIME_EPS, episode_duration, window_fits

    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if not window_fits(n_frames, window_frames, fps):
        raise ValueError(f"{n_frames} frames do not hold a window of {window_frames}")
    span = max(episode_duration(n_frames, fps) - window_frames / float(fps), 0.0)
    out = []
    for j in range(n_windows):
        start = float(rng.uniform(j * span / n_windows, (j + 1) * span / n_windows))
        anchor = math.floor(start * float(fps) + TIME_EPS)
        out.append(int(min(max(anchor, 0), int(n_frames) - int(window_frames))))
    return out


def episode_rng(seed: int, episode: int) -> np.random.Generator:
    """Return the generator of the window starts of one episode."""
    return np.random.default_rng((int(seed), int(episode)))


def window_seed(seed: int, window: int) -> int:
    """Return the sampler seed of window ``window``.

    ``window`` must be in ``[0, SEED_STRIDE)``, so that no two windows of
    two seeds share a sampler seed.
    """
    if not 0 <= int(window) < SEED_STRIDE:
        raise ValueError(f"window {window} is outside [0, {SEED_STRIDE})")
    return SEED_STRIDE * int(seed) + int(window)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def split_chunks(rows: np.ndarray, config: dict) -> list[np.ndarray]:
    """Split the ``(T, A)`` rows of a window into its chunks (21, 24, 24, 24)."""
    ch = config["chunking"]
    geom = contiguous_chunk_geometry(int(ch["action_chunk"]), int(ch["n_chunks"]))
    if rows.shape[0] != sum(geom[-1]):
        raise ValueError(f"{rows.shape[0]} rows for the geometry {geom}")
    return [rows[off : off + ln] for off, ln in geom]


def decode_chunks(tokens: Tensor, config: dict) -> list[np.ndarray]:
    """Decode the ``(1, L K, 4 A)`` tokens of a window to the rows of each chunk.

    :func:`praxis.robot_io.action_rows` unpacks each chunk: chunk 0 keeps the
    first row of its anchor token.
    The inverse q99 map of the checkpoint statistics on ``q99_normed_dims``,
    without the clip to [-1, 1] of the serve path, so the score sees the
    sampled value. The other dims stay as sampled (rot6d).
    """
    norm, ch = config["normalization"], config["chunking"]
    lat = int(ch["latents_per_chunk"])
    per_token = int(ch["action_frames_per_token"])
    dim = int(config["action_space"]["dim"])
    tok = tokens.detach().float().cpu()
    n_chunks = tok.shape[1] // lat
    q01 = np.asarray(norm["action_q01"], dtype=np.float32)
    q99 = np.asarray(norm["action_q99"], dtype=np.float32)
    dims = np.asarray(norm["q99_normed_dims"], dtype=np.int64)
    return [
        q99_denormalize(
            action_rows(tok[:, k * lat : (k + 1) * lat], k, per_token, dim),
            q01,
            q99,
            dims,
        )
        for k in range(n_chunks)
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def so3_from_rot6d(v6: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the rotation of one rot6d ``[c0, c1]`` and whether it is valid.

    Gram-Schmidt on the two columns in float64, the third column is their
    cross product. The norms reduce over the last axis, as in the original
    implementation: ``np.linalg.norm`` without an axis takes another summation
    path and can differ in the last bit. A degenerate or non-finite input is not valid.
    """
    v = np.asarray(v6, dtype=np.float64)
    a1, a2 = v[0:3], v[3:6]
    n1 = np.linalg.norm(a1, axis=-1)
    b1 = a1 / (n1 if n1 > 1e-8 else 1.0)
    u2 = a2 - (b1 * a2).sum() * b1
    n2 = np.linalg.norm(u2, axis=-1)
    b2 = u2 / (n2 if n2 > 1e-8 else 1.0)
    r = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
    orth = np.abs(r.T @ r - np.eye(3)).max()
    valid = bool(
        np.isfinite(v).all()
        and n1 > 1e-8
        and n2 > 1e-8
        and orth < 1e-5
        and abs(np.linalg.det(r) - 1.0) < 1e-5
    )
    return r, valid


def geodesic_deg(pred6: np.ndarray, gt6: np.ndarray) -> float:
    """Return the angle between two rot6d rotations in degrees (NaN if invalid)."""
    rp, vp = so3_from_rot6d(pred6)
    rg, vg = so3_from_rot6d(gt6)
    if not (vp and vg):
        return float("nan")
    tr = float(np.einsum("ij,ij->", rp, rg))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def chunk_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Return the metrics of one chunk from its ``(rows, 10)`` rows."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} differ")
    return {
        "endpoint_err_cm": float(np.linalg.norm(pred[-1, XYZ] - gt[-1, XYZ]) * 100),
        "rot_endpoint_deg": geodesic_deg(pred[-1, ROT6D], gt[-1, ROT6D]),
        "grip_mae": float(np.abs(pred[:, GRIP] - gt[:, GRIP]).mean()),
        "gt_motion_cm": float(np.linalg.norm(gt[-1, XYZ] - gt[0, XYZ]) * 100),
    }


def window_metrics(
    pred: Sequence[np.ndarray], gt: Sequence[np.ndarray]
) -> dict[str, float]:
    """Return ``<metric>_chunk<k>`` for each chunk of one window."""
    if len(pred) != len(gt):
        raise ValueError(f"{len(pred)} predicted vs {len(gt)} recorded chunks")
    out: dict[str, float] = {}
    for k, (p, g) in enumerate(zip(pred, gt, strict=True)):
        for name, value in chunk_metrics(p, g).items():
            out[f"{name}_chunk{k}"] = value
    return out


def _mean_sem(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return {
        "mean": float(v.mean()) if len(v) else float("nan"),
        "sem": sem,
        "n": len(v),
    }


def summarize(records: Sequence[dict], n_chunks: int = 4) -> dict[str, dict]:
    """Return ``<leg>/<metric>_chunk<k>`` and ``<leg>/<metric>_chunks123_mean``.

    Each value holds the mean, the standard error and the number of windows
    with a finite value. ``_chunks123_mean`` first averages chunks 1 to 3 in
    each window. ``records`` are the per-window rows of
    :func:`evaluate_window`.
    """
    out: dict[str, dict] = {}
    for leg in sorted({r["leg"] for r in records}):
        rows = [r for r in records if r["leg"] == leg]
        for name in CHUNK_METRICS:
            for k in range(n_chunks):
                out[f"{leg}/{name}_chunk{k}"] = _mean_sem(
                    [r[f"{name}_chunk{k}"] for r in rows]
                )
            if n_chunks > 1:
                later = [
                    float(np.mean([r[f"{name}_chunk{k}"] for k in range(1, n_chunks)]))
                    for r in rows
                ]
                out[f"{leg}/{name}_chunks123_mean"] = _mean_sem(later)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def gt_chunks(window: TrainingWindow, config: dict) -> list[np.ndarray]:
    """Return the recorded relative rows of each chunk of a DROID window."""
    return split_chunks(np.asarray(window.actions_rel, dtype=np.float32), config)


def sample_leg(
    policy: PraxisPolicy,
    window: EncodedWindow,
    leg: str,
    seed: int,
    settings: SamplerSettings = EVAL_SAMPLER,
    oracle_latents: Tensor | None = None,
    text_uncond: Tensor | None = None,
) -> Tensor:
    """Run one leg on one window. Return the action tokens ``(1, L K, 4 A)``.

    ``text_uncond`` is the encoded blank caption of the unguided pass. A
    guided ``settings`` needs it.

    The oracle leg pins the video to ``oracle_latents`` ``(1, 192, 6 K, h, w)``,
    by default the window chunks. The praxis video encoder gives bfloat16
    latents, so the default pins the same values that the trunk reads. A
    caller with float32 latents of another encoder passes them here.
    """
    if leg not in LEGS:
        raise ValueError(f"leg must be one of {LEGS}, got {leg!r}")
    _, actions = policy.sample_window_joint(
        text=window.text,
        text_uncond=text_uncond,
        anchor=window.anchor,
        cleans=window.chunks[:-1],
        proprio_all=window.proprio,
        seed=seed,
        settings=settings,
        oracle_video=(
            (
                torch.cat(window.chunks, dim=2)
                if oracle_latents is None
                else oracle_latents
            )
            if leg == "oracle"
            else None
        ),
        return_window=True,
    )
    return actions


def evaluate_window(
    policy: PraxisPolicy,
    window: EncodedWindow,
    gt_rows: Sequence[np.ndarray],
    index: int,
    seed: int = 0,
    settings: SamplerSettings = EVAL_SAMPLER,
    text_uncond: Tensor | None = None,
) -> list[dict]:
    """Run both legs on one window. ``window`` is the output of
    :func:`praxis.train.objective.encode_window` and ``gt_rows`` the recorded
    rows of each chunk (:func:`gt_chunks`). ``settings`` and ``text_uncond``
    go to :func:`sample_leg`. Return one record per leg with its metrics and
    the decoded predicted rows (``pred_rows``, ``(T, A)``)."""
    out = []
    for leg in LEGS:
        tokens = sample_leg(
            policy,
            window,
            leg,
            window_seed(seed, index),
            settings,
            text_uncond=text_uncond,
        )
        pred = decode_chunks(tokens, policy.config)
        out.append(
            {
                "window": index,
                "leg": leg,
                **window_metrics(pred, gt_rows),
                "pred_rows": np.concatenate(pred),
            }
        )
    return out
