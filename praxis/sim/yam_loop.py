"""The closed loop of the bimanual YAM sim: observe, call the policy, execute, repeat.

One episode:

1. Reset the task with the episode seed and observe.
2. Call the policy with the frame stack of the last 25 ticks (the observation
   format of :mod:`praxis.bimanual`). It returns one chunk of command rows
   (21 rows for the first call of a context window, 24 after).
3. Execute each row for one 15 Hz tick, push the observation to the stack.
4. Go back to step 2. Stop at success (plus ``after_success_s``) or at the
   time limit.

The loop checks each rendered model view: after a run of black or flat views
(an EGL renderer that shares a GPU with CUDA work can return them without an
error) it stops with an error, as the single-arm loop does.

Policies:

* :class:`YamModelPolicy`: the bimanual praxis checkpoint through
  :class:`praxis.pipeline.PraxisPipeline`, with the serve defaults of the
  bimanual rig (``configs/serve_yam.yaml``: 2 Euler steps of the
  ``window_joint`` layout, flow shift 1, no guidance, 1 context chunk, 24
  rows per call). The prompt goes to the model verbatim, as on the server.
* :class:`YamOraclePolicy`: the scripted oracle of :mod:`praxis.sim.yam_scripted`.
* :class:`YamHoldPolicy`: hold the start poses with the grippers open.

The model views are 416x260 by default (``configs/sim_yam.yaml``), the size
that the client of the rig sends. The pipeline fits each view to a 256x416
tile.

With ``decode_video``, :class:`YamModelPolicy` also decodes the video that
the model predicts for each call (one 256x1248 canvas per command row: top |
wrist left | wrist right). The human view then shows it under the rendered
views.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from praxis import bimanual
from praxis.sim.loop import (
    CHUNK_ROWS,
    FIRST_CHUNK_ROWS,
    STACK_ROWS,
    FrameCheck,
    _sync,
    bad_view_names,
)
from praxis.sim.robot import pos_rot_to_pose12, pose7_to_pos_rot
from praxis.sim.scene import CONTROL_HZ
from praxis.sim.video import VideoWriter, compose_panels, overlay_text
from praxis.sim.yam_env import YamEnv, YamObservation
from praxis.sim.yam_scene import MODEL_VIEWS, SIDES
from praxis.sim.yam_scripted import OracleConfig, YamOracle

if TYPE_CHECKING:
    from praxis.model import SamplerSettings
    from praxis.pipeline import PraxisPipeline

# Sim view -> the first observation key of the bimanual view in the same slot.
VIEW_KEYS = {
    sim: bimanual.VIEW_KEY_CANDIDATES[view][0]
    for sim, view in zip(MODEL_VIEWS, bimanual.VIEWS, strict=True)
}


class YamHistory:
    """The last 25 observations: the frame stack of the policy input."""

    def __init__(self, rows: int = STACK_ROWS) -> None:
        self.rows: deque[YamObservation] = deque(maxlen=rows)

    def push(self, obs: YamObservation) -> None:
        self.rows.append(obs)

    @property
    def latest(self) -> YamObservation:
        return self.rows[-1]

    def __len__(self) -> int:
        return len(self.rows)

    def obs_dict(self) -> dict[str, np.ndarray]:
        """Return the stack in the bimanual observation format of :mod:`praxis.bimanual`."""
        if not self.rows:
            raise ValueError("the history is empty")
        out: dict[str, np.ndarray] = {}
        for view, key in VIEW_KEYS.items():
            if any(view not in o.views for o in self.rows):
                raise ValueError("the model policy needs the rendered views")
            out[key] = np.stack([o.views[view] for o in self.rows])
        for s in SIDES:
            out[f"robot_state.{s}.ee_pose"] = np.stack(
                [o.ee_pose[s] for o in self.rows]
            ).astype(np.float32)
            out[f"grip.{s}"] = np.asarray(
                [o.grip[s] for o in self.rows], dtype=np.float32
            )
        return out


@dataclass
class YamPolicyOutput:
    """One policy call: poses ``(T, 2, 12)``, grips ``(T, 2)`` (left, right).

    ``stream`` (optional) gives the rows one at a time instead: the loop asks
    for the next row only after it executed the previous one, so a
    closed-loop policy (the scripted oracle) reads the present state.
    ``predicted`` (optional) is the predicted video of the call: ``(T, H, W, 3)``
    uint8, one canvas per row.
    """

    poses: np.ndarray
    grips: np.ndarray
    info: dict[str, float | int | str] = field(default_factory=dict)
    stream: Iterator[tuple[np.ndarray, np.ndarray]] | None = None
    predicted: np.ndarray | None = None

    def rows(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Return the rows of the chunk in execution order."""
        if self.stream is not None:
            return self.stream
        return zip(self.poses, self.grips, strict=True)


class YamPolicy(Protocol):
    name: str

    def reset(self, env: YamEnv, seed: int) -> None: ...

    def act(self, history: YamHistory) -> YamPolicyOutput: ...


class YamHoldPolicy:
    """Hold the start poses with the grippers open."""

    name = "hold"

    def __init__(self) -> None:
        self.calls = 0
        self.start: YamObservation | None = None

    def reset(self, env: YamEnv, seed: int) -> None:
        self.calls = 0
        self.start = None

    def act(self, history: YamHistory) -> YamPolicyOutput:
        if self.start is None:
            self.start = history.latest
        rows = FIRST_CHUNK_ROWS if self.calls == 0 else CHUNK_ROWS
        self.calls += 1
        pose = np.stack(
            [pos_rot_to_pose12(*pose7_to_pos_rot(self.start.ee_pose[s])) for s in SIDES]
        )
        return YamPolicyOutput(
            np.repeat(pose[None], rows, axis=0), np.zeros((rows, 2), np.float32)
        )


class YamOraclePolicy:
    """The scripted oracle as a chunked policy (21 rows, then 24 per call)."""

    name = "oracle"

    def __init__(self, cfg: OracleConfig | None = None) -> None:
        self.oracle = YamOracle(cfg or OracleConfig())
        self.calls = 0

    def reset(self, env: YamEnv, seed: int) -> None:
        self.oracle.reset(env)
        self.calls = 0

    def act(self, history: YamHistory) -> YamPolicyOutput:
        rows = FIRST_CHUNK_ROWS if self.calls == 0 else CHUNK_ROWS
        self.calls += 1

        def stream() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for _ in range(rows):
                yield self.oracle.next_row()

        empty = np.zeros((0, 2, 12), np.float32)
        return YamPolicyOutput(empty, np.zeros((0, 2), np.float32), stream=stream())


@dataclass
class YamSessionConfig:
    """Session settings of the bimanual checkpoint (``configs/serve_yam.yaml``)."""

    max_context_chunks: int = 1
    exec_horizon: int = 24  # command rows per call
    seed: int = 0  # chunk k of the episode with seed s: seed + seed_stride * s + k
    seed_stride: int = 1000


class YamModelPolicy:
    """The bimanual praxis checkpoint in closed loop: one session per episode.

    The session is the one of the policy server (:meth:`PraxisPipeline.session`),
    so the sim and the server run the same code path. With ``decode_video``,
    each call also decodes the predicted video. The decode does not change the
    commands: the sampler seeds and the inputs stay the same. ``prompt`` None
    sends the prompt of the env of each episode; a string replaces it.

    One policy serves one episode at a time: it holds the session of the
    current episode.
    """

    name = "model"

    def __init__(
        self,
        pipe: PraxisPipeline,
        prompt: str | None = None,
        session: YamSessionConfig | None = None,
        decode_video: bool = False,
    ) -> None:
        if pipe.embodiment != "bimanual":
            raise ValueError(
                f"the YAM sim drives the bimanual rig, the checkpoint is {pipe.embodiment!r}"
            )
        self.pipe = pipe
        self.prompt = prompt
        self.cfg = session or YamSessionConfig()
        self.decode_video = decode_video
        self.session = None

    @classmethod
    def load(
        cls,
        checkpoint: str,
        vae_weights: str,
        text_encoder: str,
        prompt: str | None = None,
        session: YamSessionConfig | None = None,
        device: str = "cuda",
        sampler: SamplerSettings | None = None,
        decode_video: bool = False,
    ) -> YamModelPolicy:
        """Load the checkpoint. ``sampler`` None = the serve defaults of the bimanual rig."""
        from praxis.model import SamplerSettings
        from praxis.pipeline import PraxisPipeline

        pipe = PraxisPipeline(
            checkpoint=checkpoint,
            vae_weights=vae_weights,
            text_encoder=text_encoder,
            device=device,
            sampler=sampler or SamplerSettings(t_shift=1.0, cfg_video_scale=1.0),
        )
        return cls(pipe, prompt, session, decode_video)

    def reset(self, env: YamEnv, seed: int) -> None:
        c = self.cfg
        self.session = self.pipe.session(
            prompt=env.prompt if self.prompt is None else self.prompt,
            seed=int(c.seed) + int(c.seed_stride) * int(seed),
            max_context_chunks=c.max_context_chunks,
            exec_horizon=c.exec_horizon,
        )

    def act(self, history: YamHistory) -> YamPolicyOutput:
        if self.session is None:
            raise RuntimeError("call reset before act")
        t0 = time.monotonic()
        res = self.session.step(history.obs_dict())
        _sync(self.pipe)
        arrays = bimanual.wire_to_arrays(res.commands)
        poses = np.stack([arrays[s][0] for s in SIDES], axis=1)
        grips = np.stack([arrays[s][1] for s in SIDES], axis=1)
        info: dict[str, float | int | str] = {
            "step_s": time.monotonic() - t0,
            "depth": int(res.depth),
            "seed": int(res.seed),
            "prompt": str(res.prompt),
        }
        predicted = None
        if self.decode_video:
            t1 = time.monotonic()
            frames = self.session.decode_video(res)
            predicted = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            info["decode_s"] = time.monotonic() - t1
        return YamPolicyOutput(poses, grips, info, predicted=predicted)


@dataclass
class YamEpisodeResult:
    """The result of one episode."""

    task: str
    seed: int
    prompt: str
    policy: str
    success: bool
    success_tick: int | None
    progress: float
    ticks: int
    calls: int
    call_s: list[float]
    metrics: dict[str, float | int | bool | None]
    wall_s: float
    model_prompt: str = ""
    bad_view_ticks: int = 0  # ticks with a black or flat model view

    def to_json(self) -> dict:
        out = asdict(self)
        out["metrics"] = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in self.metrics.items()
        }
        return out


def predicted_row(predicted: np.ndarray | None, height: int, width: int) -> np.ndarray:
    """Return the predicted canvas as a ``(height, width)`` row of the human view.

    The canvas is three 256x416 tiles (top | wrist left | wrist right). Each
    tile shows the centre crop of a rendered view at the tile aspect, so the
    canvas gets the width of the rendered row and a dark band above and below.
    ``None`` (before the first call) gives a dark row.
    """
    row = np.full((height, width, 3), 16, dtype=np.uint8)
    if predicted is None:
        return row
    h = min(height, max(1, round(predicted.shape[0] * width / predicted.shape[1])))
    pred = compose_panels([predicted], height=h, labels=["predicted by the model"])
    pred = pred[:, :width]
    y0 = (height - h) // 2
    row[y0 : y0 + h, : pred.shape[1]] = pred
    return row


def human_frame(
    env: YamEnv,
    obs: YamObservation,
    call: int,
    predicted: np.ndarray | None = None,
    predicted_slot: bool = False,
) -> np.ndarray:
    """Return the video frame of one tick: top | wrist left | wrist right, and a status band.

    With ``predicted_slot``, a second row shows the predicted canvas of the
    model for the same tick (dark before the first call), so all the frames
    of a video have the same size.
    """
    v = obs.views
    frame = compose_panels(
        [v["top"], v["wrist_left"], v["wrist_right"]],
        labels=["top", "wrist left", "wrist right"],
    )
    if predicted_slot:
        h, w = frame.shape[:2]
        frame = np.concatenate([frame, predicted_row(predicted, h, w)], axis=0)
    s = env.task.status
    line = (
        f"t {env.tick_count / CONTROL_HZ:5.1f} s   call {call}   "
        f"{'SUCCESS' if s.success else 'no success'}   "
        f"progress {s.metrics.get('progress', 0.0):.2f}"
    )
    return overlay_text(frame, [line, env.prompt])


def run_yam_episode(
    env: YamEnv,
    policy: YamPolicy,
    seed: int,
    *,
    stop_on_success: bool = True,
    after_success_s: float = 1.0,
    video: VideoWriter | None = None,
    max_ticks: int | None = None,
) -> YamEpisodeResult:
    """Run one closed-loop episode and return its result."""
    wall0 = time.monotonic()
    limit = env.max_ticks if max_ticks is None else int(max_ticks)
    linger = round(after_success_s * CONTROL_HZ)
    history = YamHistory()
    obs = env.reset(seed)
    frame_check = FrameCheck()
    frame_check.check(bad_view_names(obs.views), obs.tick)
    history.push(obs)
    policy.reset(env, seed)
    call_s: list[float] = []
    model_prompt = ""
    predicted_slot = bool(getattr(policy, "decode_video", False))

    def show(predicted: np.ndarray | None) -> None:
        if video is not None:
            video.write(
                human_frame(
                    env, obs, max(len(call_s) - 1, 0), predicted, predicted_slot
                )
            )

    show(None)
    success_at: int | None = None
    done = False
    while not done and env.tick_count < limit:
        t0 = time.monotonic()
        out = policy.act(history)
        call_s.append(float(out.info.get("step_s", time.monotonic() - t0)))
        model_prompt = str(out.info.get("prompt", model_prompt))
        if out.stream is None and len(out.poses) == 0:
            raise RuntimeError(f"the {policy.name} policy returned no commands")
        predicted = out.predicted
        for i, (poses, grips) in enumerate(out.rows()):
            obs = env.step(poses, grips)
            frame_check.check(bad_view_names(obs.views), obs.tick)
            history.push(obs)
            row_frame = None
            if predicted is not None and len(predicted):
                row_frame = predicted[min(i, len(predicted) - 1)]
            show(row_frame)
            if env.task.status.success and success_at is None:
                success_at = env.tick_count
            if env.tick_count >= limit or (
                stop_on_success
                and success_at is not None
                and env.tick_count - success_at >= linger
            ):
                done = True
                break
    summary = env.summary()
    return YamEpisodeResult(
        task=env.task.name,
        seed=int(seed),
        prompt=env.prompt,
        policy=policy.name,
        success=bool(summary["success"]),
        success_tick=env.task.status.success_tick,
        progress=float(summary.get("progress") or 0.0),
        ticks=env.tick_count,
        calls=len(call_s),
        call_s=call_s,
        metrics=summary,
        wall_s=time.monotonic() - wall0,
        model_prompt=model_prompt,
        bad_view_ticks=frame_check.bad_ticks,
    )
