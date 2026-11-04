"""The closed-loop sampling loop of the sim: observe, call the policy, execute, repeat.

One episode does these steps:

1. Reset the task with the episode seed and observe.
2. Call the policy. It returns one chunk of command rows (21 rows for the
   first call of a context window, 24 rows after).
3. Execute each row for one 15 Hz tick. After each tick, render the views,
   add the observation to the frame history, write the video frame and send
   it to the live view.
4. Go back to step 2. Stop at success (plus ``after_success_s``) or at the
   time limit.

The loop executes all the rows of a chunk before the next call. The model
session then gets the 24 executed frames plus the current frame, and it keeps
its context. A skipped row makes the next call a cold start.

Policies:

* ``model``: the praxis checkpoint through :class:`praxis.pipeline.PraxisPipeline`.
* ``scripted``: :class:`praxis.sim.scripted.ScriptedPick`, which reads the
  cube pose from the simulation. Use it to check the environment.
* ``hold``: hold the start pose with the gripper open.

Outputs per episode: ``episode_<seed>.mp4`` (the human view),
``episode_<seed>.json`` (the result) and ``episode_<seed>.npz`` (the poses
and commands of each tick). ``summary.json`` holds the result of all the
episodes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import yaml

from praxis import single_arm
from praxis.sim import gl
from praxis.sim.config import SessionConfig, SimConfig, to_dict
from praxis.sim.env import Observation, SimEnv
from praxis.sim.robot import pose7_to_pos_rot, pose12_to_pos_rot, rotation_angle_deg
from praxis.sim.scene import CONTROL_HZ
from praxis.sim.scripted import ScriptedConfig, ScriptedPick, rig_commands
from praxis.sim.video import VideoWriter, compose_panels, overlay_text, status_lines
from praxis.sim.viewer import Viewer

if TYPE_CHECKING:
    from praxis.pipeline import PraxisPipeline

logger = logging.getLogger(__name__)

STACK_ROWS = 25  # 24 executed rows + the current row
FIRST_CHUNK_ROWS = 21
CHUNK_ROWS = 24
# The size of the dark placeholder before the first prediction when the policy
# has no canvas of its own (the canvas of the 256x416 tiles).
PREDICTED_HW = (256, 832)
BAD_VIEW_MEAN = 2.0  # a model view with a lower mean pixel value is black
BAD_VIEW_STD = 1.0  # a model view with a lower pixel std is one flat color
BAD_VIEW_TICKS = 15  # stop after this many ticks in a row with a bad view

# ---------------------------------------------------------------------------
# Frame check
# ---------------------------------------------------------------------------


def bad_views(obs: Observation) -> list[str]:
    """Return the names of the model views that are black or one flat color.

    On some GPUs, an EGL renderer that shares the GPU with CUDA work returns
    black or corrupt frames and gives no error.
    """
    bad = []
    for name in ("exterior", "wrist"):
        view = getattr(obs, name)
        if view is None:
            continue
        pixels = np.asarray(view, dtype=np.float32)
        if pixels.mean() < BAD_VIEW_MEAN or pixels.std() < BAD_VIEW_STD:
            bad.append(name)
    return bad


class FrameCheck:
    """Warn at the first bad model view, and stop after a run of bad views."""

    def __init__(self, max_bad_ticks: int = BAD_VIEW_TICKS) -> None:
        self.max_bad_ticks = max_bad_ticks
        self.bad_ticks = 0
        self.run = 0

    def __call__(self, obs: Observation) -> None:
        bad = bad_views(obs)
        if not bad:
            self.run = 0
            return
        if self.bad_ticks == 0:
            logger.warning(
                "tick %d: the %s view is black or one flat color. Set "
                "render.egl_device to a GPU that does not run the model.",
                obs.tick,
                " and ".join(bad),
            )
        self.bad_ticks += 1
        self.run += 1
        if self.run >= self.max_bad_ticks:
            raise RuntimeError(
                f"the {' and '.join(bad)} view was black or one flat color for "
                f"{self.run} ticks in a row (last tick {obs.tick}). The renderer "
                "gives bad frames. Set render.egl_device to a GPU that does "
                "not run the model."
            )


# ---------------------------------------------------------------------------
# Frame history
# ---------------------------------------------------------------------------


class History:
    """The last 25 observations, the input stack of the model.

    Push one observation per 15 Hz tick, and the start observation before
    the first call. Then the stack holds the executed rows of the last chunk
    and the current row last.
    """

    def __init__(self, rows: int = STACK_ROWS) -> None:
        self.rows: deque[Observation] = deque(maxlen=rows)

    def clear(self) -> None:
        """Remove all observations."""
        self.rows.clear()

    def push(self, obs: Observation) -> None:
        """Add the observation of one tick."""
        self.rows.append(obs)

    @property
    def latest(self) -> Observation:
        """Return the current observation."""
        return self.rows[-1]

    def __len__(self) -> int:
        return len(self.rows)

    def obs_dict(self) -> dict[str, np.ndarray]:
        """Return the stack in the observation format of :mod:`praxis.single_arm`."""
        if not self.rows:
            raise ValueError("the history is empty")
        exterior = [o.exterior for o in self.rows if o.exterior is not None]
        wrist = [o.wrist for o in self.rows if o.wrist is not None]
        if len(exterior) != len(self.rows) or len(wrist) != len(self.rows):
            raise ValueError(
                "the model policy needs the rendered exterior and wrist views"
            )
        return {
            "image.exterior": np.stack(exterior),
            "image.wrist": np.stack(wrist),
            "robot_state.ee_pose": np.stack([o.ee_pose for o in self.rows]).astype(
                np.float32
            ),
            "grip": np.asarray([[o.grip] for o in self.rows], dtype=np.float32),
            # The sim reports the ee site: the flange, turned -45 deg.
            "descriptor": "praxis.sim",
        }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass
class PolicyOutput:
    """The output of one policy call."""

    commands: list[dict]  # the command dicts of :mod:`praxis.single_arm`
    predicted: np.ndarray | None = None  # (T, H, W, 3) uint8, one frame per row
    info: dict[str, float | int | str] = field(default_factory=dict)


class Policy(Protocol):
    """A policy of the loop: one chunk of command rows per call."""

    name: str

    def reset(self, env: SimEnv, seed: int) -> None:
        """Start a new episode. The environment is already reset."""
        ...

    def act(self, history: History) -> PolicyOutput:
        """Return the commands of the next chunk."""
        ...

    def close(self) -> None:
        """Free the resources of the policy."""
        ...


def _hold_rows(obs: Observation, rows: int, mode: dict | None) -> list[dict]:
    """Return ``rows`` command rows that hold the pose of ``obs``, gripper open."""
    rel = np.zeros((rows, single_arm.ACTION_DIM), dtype=np.float32)
    rel[:, 3] = 1.0  # rot6d of the identity: columns (1, 0, 0) and (0, 1, 0)
    rel[:, 7] = 1.0
    pose8 = np.concatenate([obs.ee_pose, [obs.grip]]).astype(np.float32)
    return rig_commands(rel, pose8, 1.0 / CONTROL_HZ, mode)


class HoldPolicy:
    """Hold the start pose with the gripper open (a check of the loop and the arm)."""

    name = "hold"

    def __init__(self, control_mode: str = "droid_impedance") -> None:
        self.mode = single_arm.control_mode_payload(control_mode)
        self.calls = 0
        self.start: Observation | None = None

    def reset(self, env: SimEnv, seed: int) -> None:
        self.calls = 0
        self.start = None

    def act(self, history: History) -> PolicyOutput:
        if self.start is None:
            self.start = history.latest
        rows = FIRST_CHUNK_ROWS if self.calls == 0 else CHUNK_ROWS
        self.calls += 1
        return PolicyOutput(_hold_rows(self.start, rows, self.mode))

    def close(self) -> None:
        pass


class ScriptedPolicy:
    """The scripted pick of :mod:`praxis.sim.scripted` (reads the cube pose)."""

    name = "scripted"

    def __init__(self, cfg: ScriptedConfig | None = None) -> None:
        self.pick = ScriptedPick(cfg)

    def reset(self, env: SimEnv, seed: int) -> None:
        self.pick.reset(env)

    def act(self, history: History) -> PolicyOutput:
        return PolicyOutput(self.pick.step(history.latest))

    def close(self) -> None:
        pass


class ModelPolicy:
    """The single-arm praxis checkpoint in closed loop: one model session per episode.

    The session builds the tile canvas and converts the poses between the rig
    frame of the sim and the pose frame of the checkpoint.

    The session of an episode with seed ``s`` samples chunk ``k`` with the
    seed ``session.seed + session.seed_stride * s + k``.
    """

    name = "model"

    def __init__(
        self,
        pipe: PraxisPipeline,
        prompt: str,
        session: SessionConfig | None = None,
        decode_video: bool = True,
    ) -> None:
        if pipe.embodiment != "single_arm":
            raise ValueError(
                f"the sim drives the single-arm rig, the checkpoint is {pipe.embodiment!r}"
            )
        self.pipe = pipe
        self.prompt = prompt
        self.cfg = session or SessionConfig()
        self.decode_video = decode_video
        self.session = None
        self.calls = 0

    @property
    def canvas_hw(self) -> tuple[int, int]:
        """The ``(height, width)`` of the predicted canvas of the checkpoint."""
        return single_arm.canvas_size(self.pipe.policy.config)

    @classmethod
    def from_config(cls, cfg: SimConfig, prompt: str) -> ModelPolicy:
        """Load the checkpoint, the autoencoder and the text encoder of ``cfg``."""
        from praxis.model import SamplerSettings
        from praxis.pipeline import PraxisPipeline

        m, s = cfg.model, cfg.sampler
        pipe = PraxisPipeline(
            checkpoint=m.checkpoint,
            vae_weights=m.vae_weights,
            text_encoder=m.text_encoder,
            device=m.device,
            vae_norm_block_c=m.vae_norm_block_c or None,
            sampler=SamplerSettings(
                num_steps=s.num_steps,
                t_shift=s.t_shift,
                solver=s.solver,
                action_steps=s.action_steps,
                layout=s.layout,
                cfg_scale=s.cfg_scale,
                cfg_video_scale=s.cfg_video_scale,
                cfg_batched=s.cfg_batched,
            ),
        )
        return cls(pipe, prompt, cfg.session, cfg.policy.decode_video)

    def reset(self, env: SimEnv, seed: int) -> None:
        c = self.cfg
        self.session = self.pipe.session(
            prompt=self.prompt,
            seed=int(c.seed) + int(c.seed_stride) * int(seed),
            max_context_chunks=c.max_context_chunks,
            control_mode=c.control_mode,
            exec_horizon=c.exec_horizon,
        )
        self.calls = 0

    def act(self, history: History) -> PolicyOutput:
        if self.session is None:
            raise RuntimeError("call reset before act")
        obs = history.obs_dict()
        t0 = time.monotonic()
        res = self.session.step(obs)
        _sync(self.pipe)
        step_s = time.monotonic() - t0
        info: dict[str, float | int | str] = {
            "step_s": step_s,
            "depth": int(res.depth),
            "seed": int(res.seed),
            "rows": len(res.commands),
            "restarts": int(getattr(self.session.state, "restart_count", 0)),
        }
        predicted = None
        if self.decode_video:
            t1 = time.monotonic()
            frames = self.session.decode_video(res)
            predicted = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            info["decode_s"] = time.monotonic() - t1
        self.calls += 1
        return PolicyOutput(res.commands, predicted, info)

    def close(self) -> None:
        self.session = None


def _sync(pipe: object) -> None:
    """Wait for the GPU work of the pipeline (for the call times)."""
    device = getattr(pipe, "device", None)
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def make_policy(cfg: SimConfig, prompt: str) -> Policy:
    """Return the policy of ``cfg.policy.type``."""
    kind = cfg.policy.type
    if kind == "model":
        return ModelPolicy.from_config(cfg, prompt)
    if kind == "scripted":
        return ScriptedPolicy(cfg.policy.scripted)
    if kind == "hold":
        return HoldPolicy(cfg.session.control_mode)
    raise ValueError(f"unknown policy type {kind!r}")


# ---------------------------------------------------------------------------
# Human view
# ---------------------------------------------------------------------------


def human_frame(
    obs: Observation,
    predicted: np.ndarray | None,
    lines: list[str],
    predicted_slot: bool = False,
    predicted_hw: tuple[int, int] = PREDICTED_HW,
) -> np.ndarray | None:
    """Return the video frame of one tick, or ``None`` without rendered views.

    Top row: ``[overview | exterior | wrist]``, the rendered views. Bottom
    row (with ``predicted_slot``): the predicted ``[exterior | wrist]`` canvas
    of the model under the rendered views. Before the first prediction, the
    bottom row is dark (``predicted_hw``, the canvas of the checkpoint), so
    all the frames of a video have the same size. The status lines go on a
    band at the bottom.
    """
    if predicted is None and predicted_slot:
        predicted = np.full((*predicted_hw, 3), 16, dtype=np.uint8)
    views = [obs.overview, obs.exterior, obs.wrist]
    names = ["overview", "exterior (model input)", "wrist (model input)"]
    panels = [v for v in views if v is not None]
    labels = [n for v, n in zip(views, names, strict=True) if v is not None]
    if not panels:
        return None
    h = panels[-1].shape[0]
    frame = compose_panels(panels, height=h, labels=labels)
    if predicted is not None:
        pred = compose_panels([predicted], height=h, labels=["predicted by the model"])
        width = frame.shape[1]
        if pred.shape[1] > width:
            pad = np.full((h, pred.shape[1] - width, 3), 16, dtype=np.uint8)
            frame = np.concatenate([frame, pad], axis=1)
            width = frame.shape[1]
        left = np.full((h, width - pred.shape[1], 3), 16, dtype=np.uint8)
        frame = np.concatenate([frame, np.concatenate([left, pred], axis=1)], axis=0)
    return overlay_text(frame, lines)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """The result of one episode."""

    seed: int
    policy: str
    task: str
    prompt: str
    success: bool
    success_tick: int | None
    ticks: int
    calls: int
    call_rows: list[int]
    call_depths: list[int]
    call_s: list[float]
    decode_s: list[float]
    restarts: int
    bad_view_ticks: int
    metrics: dict[str, float | int | bool | None]
    tracking: dict[str, float]
    start_ee_pose: list[float]
    start_grip: float
    wall_s: float
    video: str | None = None
    arrays: str | None = None

    @property
    def mean_call_s(self) -> float:
        """Return the mean time of one policy call in seconds."""
        return float(np.mean(self.call_s)) if self.call_s else 0.0

    def to_json(self) -> dict:
        """Return the result as JSON types."""
        out = asdict(self)
        out["mean_call_s"] = self.mean_call_s
        out["success_time_s"] = (
            None if self.success_tick is None else self.success_tick / CONTROL_HZ
        )
        return _finite_dict(out)


def _finite(value: object) -> object:
    """Return ``value`` with NaN and inf floats as ``None`` (strict JSON)."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _finite_dict(value: dict) -> dict:
    """Return :func:`_finite` of a mapping."""
    out = _finite(value)
    assert isinstance(out, dict)
    return out


def _stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(arr.max()),
    }


class _Recorder:
    """Per-tick arrays and tracking errors of one episode."""

    def __init__(self) -> None:
        self.ee_pose: list[np.ndarray] = []
        self.grip: list[float] = []
        self.cmd_pose: list[np.ndarray] = []
        self.cmd_grip: list[float] = []
        self.cube: list[np.ndarray] = []
        self.call: list[int] = []
        self.pos_err_mm: list[float] = []
        self.rot_err_deg: list[float] = []
        self.ik_pos_mm: list[float] = []

    def add(
        self,
        obs: Observation,
        pose12: np.ndarray,
        grip: float,
        cube: np.ndarray,
        call: int,
    ) -> None:
        self.ee_pose.append(obs.ee_pose.copy())
        self.grip.append(float(obs.grip))
        self.cmd_pose.append(np.asarray(pose12, dtype=np.float32).copy())
        self.cmd_grip.append(float(grip))
        self.cube.append(np.asarray(cube, dtype=np.float32))
        self.call.append(call)
        if np.all(np.isfinite(pose12)):
            p_cmd, r_cmd = pose12_to_pos_rot(pose12)
            p_obs, r_obs = pose7_to_pos_rot(obs.ee_pose)
            self.pos_err_mm.append(1e3 * float(np.linalg.norm(p_obs - p_cmd)))
            self.rot_err_deg.append(rotation_angle_deg(r_obs, r_cmd))
        if obs.ik is not None:
            self.ik_pos_mm.append(1e3 * obs.ik.pos_err)

    def tracking(self) -> dict[str, float]:
        out = {
            **_stats("track_pos_mm", self.pos_err_mm),
            **_stats("track_rot_deg", self.rot_err_deg),
            **_stats("ik_residual_mm", self.ik_pos_mm),
        }
        if len(self.cmd_pose) > 1:
            xyz = np.stack(self.cmd_pose)[:, :3]
            out["cmd_step_mm_mean"] = 1e3 * float(
                np.linalg.norm(np.diff(xyz, axis=0), axis=1).mean()
            )
        return out

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            ee_pose=np.asarray(self.ee_pose, dtype=np.float32).reshape(-1, 7),
            grip=np.asarray(self.grip, dtype=np.float32),
            cmd_pose=np.asarray(self.cmd_pose, dtype=np.float32).reshape(-1, 12),
            cmd_grip=np.asarray(self.cmd_grip, dtype=np.float32),
            cube_pos=np.asarray(self.cube, dtype=np.float32).reshape(-1, 3),
            call=np.asarray(self.call, dtype=np.int32),
        )


def run_episode(
    env: SimEnv,
    policy: Policy,
    seed: int,
    *,
    prompt: str = "",
    stop_on_success: bool = True,
    after_success_s: float = 1.0,
    max_ticks: int | None = None,
    video: VideoWriter | None = None,
    viewer: Viewer | None = None,
    pace_realtime: bool = False,
    arrays_path: Path | None = None,
    status_extra: dict[str, object] | None = None,
) -> EpisodeResult:
    """Run one closed-loop episode and return its result.

    The loop calls the policy, executes every row of the chunk (one row per
    tick), and calls the policy again. It stops at the time limit, or
    ``after_success_s`` after success when ``stop_on_success`` is set.
    """
    wall0 = time.monotonic()
    limit = env.max_ticks if max_ticks is None else int(max_ticks)
    linger = round(after_success_s * CONTROL_HZ)
    history = History()
    rec = _Recorder()
    obs = env.reset(seed)
    frame_check = FrameCheck()
    frame_check(obs)
    start_pose, start_grip = obs.ee_pose.copy(), float(obs.grip)
    history.push(obs)
    policy.reset(env, seed)
    call_rows: list[int] = []
    call_depths: list[int] = []
    call_s: list[float] = []
    decode_s: list[float] = []
    restarts = 0
    predicted: np.ndarray | None = None
    extra = dict(status_extra or {})
    predicted_slot = bool(getattr(policy, "decode_video", False))
    ph, pw = getattr(policy, "canvas_hw", PREDICTED_HW)
    predicted_hw = (int(ph), int(pw))

    def status(phase: str) -> dict[str, object]:
        s = env.task.status
        return {
            **extra,
            "phase": phase,
            "seed": seed,
            "tick": env.tick_count,
            "time_s": env.tick_count / CONTROL_HZ,
            "calls": len(call_s),
            "success": s.success,
            "max_lift_m": s.metrics.get("max_lift_m", 0.0),
            "min_reach_m": s.metrics.get("min_reach_m", math.inf),
            "grip": obs.grip,
        }

    def show(row_frame: np.ndarray | None, phase: str) -> None:
        lines = status_lines(
            env.tick_count / CONTROL_HZ,
            call_index=max(len(call_s) - 1, 0),
            prompt=prompt or None,
            success=env.task.status.success,
            extra={"seed": seed, "grip": obs.grip},
        )
        frame = human_frame(obs, row_frame, lines, predicted_slot, predicted_hw)
        if frame is None:
            return
        if video is not None:
            video.write(frame)
        if viewer is not None:
            viewer.publish(frame, status(phase))

    show(None, "start")
    success_at: int | None = None
    done = False
    while not done and env.tick_count < limit:
        if viewer is not None:
            viewer.set_status(status(f"policy call {len(call_s)}"))
        t0 = time.monotonic()
        out = policy.act(history)
        call_s.append(float(out.info.get("step_s", time.monotonic() - t0)))
        if "decode_s" in out.info:
            decode_s.append(float(out.info["decode_s"]))
        restarts = int(out.info.get("restarts", restarts))
        call_depths.append(int(out.info.get("depth", -1)))
        call_rows.append(len(out.commands))
        if not out.commands:
            raise RuntimeError(f"the {policy.name} policy returned no commands")
        predicted = out.predicted
        poses, grips = single_arm.wire_to_arrays(out.commands)
        for i, (pose12, grip) in enumerate(zip(poses, grips, strict=True)):
            tick0 = time.monotonic()
            obs = env.step(pose12, float(grip))
            frame_check(obs)
            history.push(obs)
            rec.add(
                obs, pose12, float(grip), env.task.cube_pos(env.data), len(call_s) - 1
            )
            row_frame = None
            if predicted is not None and len(predicted):
                row_frame = predicted[min(i, len(predicted) - 1)]
            show(row_frame, "running")
            if env.task.status.success and success_at is None:
                success_at = env.tick_count
            if pace_realtime:
                delay = 1.0 / CONTROL_HZ - (time.monotonic() - tick0)
                if delay > 0:
                    time.sleep(delay)
            if env.tick_count >= limit or (
                stop_on_success
                and success_at is not None
                and env.tick_count - success_at >= linger
            ):
                done = True
                break
    summary = env.summary()
    if arrays_path is not None:
        rec.save(arrays_path)
    result = EpisodeResult(
        seed=int(seed),
        policy=policy.name,
        task=env.task.name,
        prompt=prompt,
        success=bool(summary["success"]),
        success_tick=env.task.status.success_tick,
        ticks=env.tick_count,
        calls=len(call_s),
        call_rows=call_rows,
        call_depths=call_depths,
        call_s=call_s,
        decode_s=decode_s,
        restarts=restarts,
        bad_view_ticks=frame_check.bad_ticks,
        metrics=summary,
        tracking=rec.tracking(),
        start_ee_pose=[float(v) for v in start_pose],
        start_grip=start_grip,
        wall_s=time.monotonic() - wall0,
        arrays=None if arrays_path is None else str(arrays_path),
    )
    if viewer is not None:
        viewer.set_status(status("episode done"))
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """The result of all the episodes of a run."""

    out_dir: Path
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        """Return the fraction of episodes with success."""
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    def to_json(self) -> dict:
        """Return the summary as JSON types."""
        eps = self.episodes
        calls = [s for e in eps for s in e.call_s]
        return _finite_dict(
            {
                "episodes": len(eps),
                "successes": sum(e.success for e in eps),
                "success_rate": self.success_rate,
                "mean_ticks": float(np.mean([e.ticks for e in eps])) if eps else 0.0,
                "mean_call_s": float(np.mean(calls)) if calls else 0.0,
                "per_episode": [
                    {
                        "seed": e.seed,
                        "success": e.success,
                        "success_tick": e.success_tick,
                        "ticks": e.ticks,
                        "calls": e.calls,
                        "max_lift_m": e.metrics.get("max_lift_m"),
                        "min_reach_m": e.metrics.get("min_reach_m"),
                        "max_grip": e.metrics.get("max_grip"),
                        "video": e.video,
                    }
                    for e in eps
                ],
            }
        )


def check_model_paths(cfg: SimConfig) -> None:
    """Raise ``FileNotFoundError`` when a model path of ``cfg`` does not exist."""
    m = cfg.model
    missing = [
        f"model.{key}: {path}"
        for key, path in (
            ("checkpoint", m.checkpoint),
            ("vae_weights", m.vae_weights),
            ("text_encoder", m.text_encoder),
        )
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "these model paths do not exist (relative paths start at the current "
            "directory; change them in the config or with --set):\n  "
            + "\n  ".join(missing)
        )


def select_egl_device(egl_device: int | None) -> None:
    """Select the GPU of the EGL renderer. Call it before the first renderer.

    ``None`` keeps ``MUJOCO_EGL_DEVICE_ID`` when it is set. Else it selects
    the first GPU of ``CUDA_VISIBLE_DEVICES`` when that is a list of
    numbers, because the EGL device index counts all physical GPUs.
    """
    if egl_device is None:
        if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
            return
        first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if not first.isdigit():
            return
        egl_device = int(first)
    gl.prepare(egl_device=egl_device)


def run(
    cfg: SimConfig,
    viewer: Viewer | None = None,
    policy: Policy | None = None,
    log: Callable[[str], None] = print,
) -> RunResult:
    """Run all the episodes of ``cfg`` and write the outputs.

    ``policy`` replaces the policy of ``cfg.policy.type`` (for tests).
    """
    if cfg.policy.type == "model":
        check_model_paths(cfg)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    select_egl_device(cfg.render.egl_device)
    env = SimEnv(cfg.env_config())
    prompt = env.prompt
    episodes: list[EpisodeResult] = []
    own_policy = policy is None
    try:
        if policy is None:
            log(f"loading the {cfg.policy.type} policy")
            t0 = time.monotonic()
            policy = make_policy(cfg, prompt)
            log(f"policy ready in {time.monotonic() - t0:.1f} s")
        renderer = env.cameras.renderer if env.cameras is not None else "off"
        log(f"task {env.task.name}, prompt {prompt!r}, renderer {renderer}")
        for k, seed in enumerate(cfg.episodes.seeds):
            stem = out_dir / f"episode_{seed:03d}"
            writer = (
                VideoWriter(
                    stem.with_suffix(".mp4"), fps=CONTROL_HZ, crf=cfg.output.video_crf
                )
                if cfg.output.video and env.cameras is not None
                else None
            )
            try:
                result = run_episode(
                    env,
                    policy,
                    seed,
                    prompt=prompt,
                    stop_on_success=cfg.episodes.stop_on_success,
                    after_success_s=cfg.episodes.after_success_s,
                    video=writer,
                    viewer=viewer,
                    pace_realtime=cfg.viewer.pace_realtime and viewer is not None,
                    arrays_path=stem.with_suffix(".npz")
                    if cfg.output.save_arrays
                    else None,
                    status_extra={"episode": f"{k + 1}/{len(cfg.episodes.seeds)}"},
                )
            finally:
                video_path = writer.close() if writer is not None else None
            result.video = None if video_path is None else str(video_path)
            stem.with_suffix(".json").write_text(json.dumps(result.to_json(), indent=1))
            episodes.append(result)
            if viewer is not None and video_path is not None:
                viewer.set_video(video_path)
            log(
                f"episode seed {seed}: success {result.success}, ticks {result.ticks}, "
                f"calls {result.calls}, mean call {result.mean_call_s:.2f} s, "
                f"max lift {float(result.metrics.get('max_lift_m') or 0.0):.3f} m, "
                f"min reach {float(result.metrics.get('min_reach_m') or 0.0):.3f} m"
            )
    finally:
        if own_policy and policy is not None:
            policy.close()
        env.close()
    run_result = RunResult(out_dir, episodes)
    (out_dir / "summary.json").write_text(json.dumps(run_result.to_json(), indent=1))
    return run_result
