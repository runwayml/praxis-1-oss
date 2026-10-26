"""The served policy: session settings, handshake meta and one call per chunk.

A :class:`PolicyFactory` holds the loaded pipeline and the boot settings. It
makes one :class:`ServedPolicy` per WebSocket session from the session
parameters. A :class:`ServedPolicy` wraps one :class:`praxis.pipeline.Session`
and adds the session rules of the server:

* ``context_mode``: ``observed`` builds the context from the executed frames
  that the client sends as a frame stack. ``stateless`` makes each call a
  cold start. ``exec_horizon`` below 24 and ``max_context_chunks=0`` also
  make the session stateless.
* ``exec_horizon``: the number of command rows of each answer.
* ``max_chunks_per_episode``: the number of calls before the session ends.
* The seed plan: chunk ``k`` samples with the seed ``base + k``.
* The sampler settings of the session (``num_steps``, ``cfg_scale`` and the
  other sampler parameters).

The server serves the single-arm checkpoint. The handshake meta
holds the variant of the checkpoint (:func:`praxis.checkpoint.model_variant`).

The checks of the session parameters run in a fixed order. The first bad
value gives the error message of the session.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import numpy as np

from praxis import robot_io, single_arm
from praxis.checkpoint import model_variant
from praxis.pipeline import PraxisPipeline, StepResult
from praxis.sampler import action_substep_grid, check_guidance, sampler_num_evals
from praxis.serve import params
from praxis.serve.wire import compress_images, image_codecs

logger = logging.getLogger(__name__)

CONTEXT_MODES: tuple[str, ...] = ("observed", "stateless")
CARRY_MODES: tuple[str, ...] = ("clean", "zeros")
PROMPT_STYLES: tuple[str, ...] = ("caption", "verbatim")

# The boot values of the session settings that are not sampler settings. The
# ``session`` block of the config replaces them. A session parameter replaces
# both for one session.
SESSION_DEFAULTS: dict = {
    "exec_horizon": None,
    "context_mode": "observed",
    "max_context_chunks": 1,
    "carry_mode": "clean",
    "prompt": "",
    "prompt_style": "caption",
    "control_mode": "none",
    "smoothing": None,
    "sg_window": 9,
    "sg_order": 3,
    "max_chunks_per_episode": None,
}

# The frame stack keys of the rig, in the declared order.
STACK_KEYS: tuple[str, ...] = (
    "image.wrist",
    "image.exterior",
    "robot_state.ee_pose",
    "grip",
)


class ChunkBudgetExhausted(RuntimeError):
    """The session reached its ``max_chunks_per_episode``."""


def served_code_version() -> str:
    """Return the code version for the handshake.

    ``PRAXIS_SERVED_COMMIT`` sets it. Else it is the git commit of the praxis
    checkout, with ``-dirty`` for changes that are not committed. A praxis
    package outside its own checkout (for example in site-packages) gives
    ``unknown``.
    """
    stamped = os.environ.get("PRAXIS_SERVED_COMMIT")
    if stamped:
        return stamped
    return params.git_commit(Path(__file__).resolve().parents[2])


class PolicyFactory:
    """Make one :class:`ServedPolicy` per session from the session parameters.

    ``session`` holds the boot values of :data:`SESSION_DEFAULTS` (the
    ``session`` block of the config). The sampler settings of the pipeline
    are the boot values of the sampler parameters.
    """

    def __init__(self, pipe: PraxisPipeline, session: dict, checkpoint: str) -> None:
        unknown = sorted(set(session) - set(SESSION_DEFAULTS))
        if unknown:
            raise ValueError(
                f"Unknown session settings {unknown}. The known settings are "
                f"{sorted(SESSION_DEFAULTS)}."
            )
        self.pipe = pipe
        self.defaults = {**SESSION_DEFAULTS, **session}
        self.checkpoint = checkpoint
        self.arm = Path(checkpoint).name
        self.code_version = served_code_version()
        image_codecs()  # a bad PRAXIS_IMAGE_CODECS fails the boot
        config = pipe.policy.config
        if pipe.embodiment != "single_arm":
            raise ValueError(
                f"the checkpoint {checkpoint} is for the {pipe.embodiment!r} rig. "
                "The server serves the single-arm checkpoint only."
            )
        variant = model_variant(config)
        self.stack_keys = STACK_KEYS
        # The rig sends the views at 1024x288. The server makes the tiles.
        self.restrict = (single_arm.CANVAS_W, single_arm.CANVAS_H)
        self.fps = float(config["chunking"].get("grid_hz", robot_io.FPS))
        self.n_chunks = int(config["chunking"].get("n_chunks", single_arm.N_CHUNKS))
        expert, rope = config["action_expert"], config["backbone"]["rope"]
        self.provenance = {
            "expert_rope": bool(expert["rope_cross"]),
            "expert_rope_self": bool(expert["rope_self"]),
            "expert_absolute_positions": bool(expert["absolute_positions"]),
            "rope_layout": "contiguous"
            if rope.get("chunked_video_layout")
            else "legacy",
            "rope_n_tiles": rope.get("n_tiles"),
            "tile_embedding": False,
        }
        # The variant of the checkpoint.
        self.variant_meta: dict[str, object] = {
            "sampler_layout": pipe.sampler.layout,
            "backbone_layout": variant.backbone_layout,
            "canvas": variant.canvas,
            "tile_size": None if variant.tile_size is None else list(variant.tile_size),
            "pose_frame": variant.pose_frame,
            "text_uncond": variant.text_uncond,
        }

    def __call__(self, overrides: dict) -> ServedPolicy:
        return ServedPolicy(self, dict(overrides))


class ServedPolicy:
    """The policy of one robot session.

    The constructor checks the session parameters ``o``. A bad value raises
    ``ValueError``, or ``TypeError`` for a value of the wrong type.
    """

    def __init__(self, factory: PolicyFactory, o: dict) -> None:
        d = factory.defaults
        s = factory.pipe.sampler
        self.factory = factory
        if "pad_start" in o and params.parse_on_off(o["pad_start"]) is True:
            raise ValueError(
                "pad_start=true is not served: the rig TemporalStack is hardcoded "
                "pad_start=False (padded leading frames were never trained on; a "
                "short stack restarts into the trained cold start). Drop the param."
            )
        eh = o.get("exec_horizon", d["exec_horizon"])
        self.exec_horizon = None if eh is None else int(eh)
        self.num_steps = int(o.get("num_steps", s.num_steps))
        self.cfg_scale = float(o.get("cfg_scale", s.cfg_scale))
        context_mode = str(o.get("context_mode", d["context_mode"]))
        max_context_chunks = int(o.get("max_context_chunks", d["max_context_chunks"]))
        self.carry_mode = str(o.get("carry_mode", d["carry_mode"]))
        self.smoothing = o.get("smoothing", d["smoothing"])
        self.sg_window = int(o.get("sg_window", d["sg_window"]))
        self.sg_order = int(o.get("sg_order", d["sg_order"]))
        self.max_chunks = o.get("max_chunks_per_episode", d["max_chunks_per_episode"])
        prompt = str(o.get("prompt", d["prompt"]))
        self.stop_on_fault = bool(o.get("declare_stop_on_fault", True))
        record = o.get("record_denoised")
        self.compile = params.parse_on_off(o.get("compile"))
        self.skip_video_decoder = params.parse_skip_mode(o.get("skip_video_decoder"))
        self.action_steps = params.parse_opt_int(o.get("action_steps", s.action_steps))
        self.cfg_video_scale = float(o.get("cfg_video_scale", s.cfg_video_scale))
        self.cfg_batched = (
            params.parse_on_off(o.get("cfg_batched", s.cfg_batched)) is not False
        )
        self.t_shift = float(o.get("t_shift", s.t_shift))
        self.solver = str(o.get("solver", s.solver))
        self.prompt_style = str(o.get("prompt_style", d["prompt_style"]))
        self.control_mode = str(o.get("control_mode", d["control_mode"]))

        if context_mode == "rolling":
            raise ValueError(
                "context_mode 'rolling' is not served: the model trains on real "
                "frames only, so its context cannot hold its own predictions. Use "
                "context_mode='observed' ('stateless' is available for debugging)."
            )
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode {context_mode!r} not in {CONTEXT_MODES}")
        if self.carry_mode not in CARRY_MODES:
            raise ValueError(f"carry_mode {self.carry_mode!r} not in {CARRY_MODES}")
        check_guidance(self.cfg_scale, self.cfg_video_scale)
        self.max_context_chunks = max(0, min(max_context_chunks, factory.n_chunks - 1))
        if self.max_context_chunks == 0:
            context_mode = "stateless"
        if self.prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"prompt_style {self.prompt_style!r} not in {PROMPT_STYLES}"
            )
        if self.control_mode not in single_arm.CONTROL_MODES:
            raise ValueError(
                f"Unknown control_mode {self.control_mode!r}; known modes are "
                f"{sorted(single_arm.CONTROL_MODES)}"
            )
        if (
            context_mode == "observed"
            and self.exec_horizon is not None
            and self.exec_horizon < robot_io.ACTION_CHUNK
        ):
            logger.warning(
                "exec_horizon=%d < %d: the next observation is off the chunk grid. "
                "Each call of the session is a cold start (stateless).",
                self.exec_horizon,
                robot_io.ACTION_CHUNK,
            )
            context_mode = "stateless"
        self.context_mode = context_mode
        self.record_denoised = None if record is None else bool(record)
        action_substep_grid(self.num_steps, self.action_steps, self.t_shift)
        self.num_evals = sampler_num_evals(
            self.num_steps, self.solver, self.action_steps
        )
        served = {
            "history_frames": self.history_frames,
            "stride": 1,
            "stack_keys": factory.stack_keys,
        }
        for key, actual in served.items():
            if key not in o:
                continue
            want = tuple(o[key]) if key == "stack_keys" else int(o[key])
            if want != actual:
                raise ValueError(
                    f"{key}={want!r} is not served: the rig TemporalStack's {key} is "
                    f"DERIVED by the server and is {actual!r} for this session "
                    "(history_frames/stride follow context_mode; stack_keys follows "
                    "the embodiment). Drop the param rather than have it validated "
                    "and then ignored."
                )
        self.seed_base, self.seed_source = params.resolve_sample_seed(
            o.get("sample_seed"), os.environ.get("PRAXIS_SAMPLE_SEED")
        )
        timing = params.parse_on_off(o.get("return_server_timing", True))
        self.return_timing = True if timing is None else timing
        # The window_joint sampler has no expert-only sub-steps.
        no_substeps = s.layout == "window_joint"
        if no_substeps and self.action_steps not in (None, self.num_steps):
            logger.warning(
                "The %s layout has no expert-only sub-steps. "
                "The session ignores action_steps=%d.",
                s.layout,
                self.action_steps,
            )
        sampler = dataclasses.replace(
            s,
            num_steps=self.num_steps,
            t_shift=self.t_shift,
            solver=self.solver,
            action_steps=None if no_substeps else self.action_steps,
            cfg_scale=self.cfg_scale,
            cfg_video_scale=self.cfg_video_scale,
            cfg_batched=self.cfg_batched,
        )
        self.session = factory.pipe.session(
            prompt=prompt,
            seed=self.seed_base,
            max_context_chunks=self.max_context_chunks,
            control_mode=self.control_mode,
            # The server cuts the rows to its exec_horizon (select_action).
            exec_horizon=None,
            sampler=sampler,
        )

    @property
    def history_frames(self) -> int:
        """Return the frames of the executed chunk that each observation carries."""
        return robot_io.ACTION_CHUNK if self.context_mode == "observed" else 0

    @property
    def effective_prompt(self) -> str:
        """Return the live prompt of the session (the last override, verbatim)."""
        return self.session.state.prompt

    @property
    def timing(self) -> dict[str, object]:
        """Return the sampler values of the ``server_timing`` block."""
        return {
            "cfg_scale": self.cfg_scale,
            "cfg_video_scale": self.cfg_video_scale,
            "cfg_batched": self.cfg_batched,
            "t_shift": self.t_shift,
            "solver": self.solver,
            "num_evals": self.num_evals,
        }

    @property
    def meta(self) -> dict[str, object]:
        """Return the ``meta`` value of the handshake, without the rollout keys."""
        f = self.factory
        return {
            "policy": "praxis",
            "code_version": f.code_version,
            "positronic_wire_ref": params.POSITRONIC_WIRE_REF,
            "arm": f.arm,
            "checkpoint": f.checkpoint,
            **f.provenance,
            **f.variant_meta,
            "exec_horizon": self.exec_horizon,
            "num_steps": self.num_steps,
            "t_shift": self.t_shift,
            "solver": self.solver,
            "num_evals": self.num_evals,
            "cfg_scale": self.cfg_scale,
            "cfg_video_scale": self.cfg_video_scale,
            "cfg_batched": self.cfg_batched,
            "context_mode": self.context_mode,
            "max_context_chunks": self.max_context_chunks,
            "carry_mode": self.carry_mode,
            "record_denoised": self.record_denoised,
            "compile": self.compile,
            "skip_video_decoder": self.skip_video_decoder,
            "action_steps": self.action_steps,
            "embodiment": "franka_ee",
            "control_mode": self.control_mode,
            "action_layout": "ee-10d [xyz3 rot6d6 grip1], rel-to-anchor, grip abs",
            "wire_commands": ["robot_command", "target_grip"],
            "wan_pattern": "contiguous-window-144444",
            "chunk_rows": robot_io.ACTION_CHUNK,
            "prompt": self.session.state.prompt,
            "prompt_overridable": True,
            "pad_start": False,
            # The prompt goes to the text encoder verbatim, whatever the
            # prompt_style. The two keys stay for the clients.
            "prompt_style": self.prompt_style,
            "prompt_rewritten": None,
            "sample_seed_base": self.seed_base,
            "seed_source": self.seed_source,
            "local_stack": params.local_stack(
                f.stack_keys,
                self.history_frames,
                f.fps,
                *f.restrict,
                self.stop_on_fault,
            ),
            "compress_images": compress_images(),
            "image_codecs": image_codecs(),
        }

    def select_action(self, obs: dict) -> list[dict]:
        """Run one policy call. Return the command rows of the chunk."""
        state = self.session.state
        if self.max_chunks is not None and state.chunk_index >= self.max_chunks:
            raise ChunkBudgetExhausted(
                f"max_chunks_per_episode={self.max_chunks} reached"
            )
        if self.context_mode == "stateless":
            state.restart()
        res = self.session.step(obs)
        wire = res.commands
        if self.smoothing in ("savgol", "sg"):
            wire = self._smoothed(res)
        return wire if self.exec_horizon is None else wire[: self.exec_horizon]

    def _smoothed(self, res: StepResult) -> list[dict]:
        """Return the commands of Savitzky-Golay smoothed relative actions.

        The input of the filter is the normalized rows with the clip to
        [-1, 1] and the inverse q99 map (:func:`praxis.robot_io.denormalize_rows`).
        The filter runs over the time axis of the xyz and gripper dims. The
        rotation dims stay unchanged. The filter can go past the end stops of
        the gripper, thus the grippers get a clip to [0, 1] after the filter.
        A chunk shorter than ``sg_window`` stays unchanged.
        """
        from scipy.signal import savgol_filter

        state = self.session.state
        assert isinstance(state, single_arm.SingleArmServeSession)
        rows = res.action_rows_norm
        if rows.shape[0] < self.sg_window or self.sg_window <= self.sg_order:
            return res.commands
        rel = robot_io.denormalize_rows(rows, state.stats, state.dims)
        out = rel.copy()
        for dim in single_arm.Q99_DIMS:
            if dim < rel.shape[1]:
                out[:, dim] = savgol_filter(rel[:, dim], self.sg_window, self.sg_order)
        out = robot_io.clip_grip(out, single_arm.GRIP_DIMS)
        # The fps, the control mode and the pose frame of the session.
        return state.rel_to_wire(out, res.inputs["pose8_now"])

    def close(self) -> None:
        """Drop the context of the session."""
        self.session.state.reset()


def warmup_obs(step: int, stack: int = 25) -> dict:
    """Return a synthetic observation with a full frame stack for the boot warmup."""
    h, w = 288, 512
    t = step * robot_io.ACTION_CHUNK + np.arange(stack)

    def frames(phase: int) -> np.ndarray:
        x = np.zeros((stack, h, w, 3), np.uint8)
        x[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, None, :]
        x[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[None, :, None]
        x[..., 2] = ((t * 37 + phase * 61) % 255)[:, None, None]
        return x

    def poses(y: float) -> np.ndarray:
        p = np.zeros((stack, 7), np.float32)
        p[:, 0], p[:, 1], p[:, 2], p[:, 3] = 0.4 + 0.0005 * t, y, 0.4, 1.0
        return p

    ext = frames(0)
    return {
        "image.exterior": ext,
        "image.wrist": ext.copy(),
        "robot_state.ee_pose": poses(0.0),
        "grip": np.zeros(stack, np.float32),
    }
