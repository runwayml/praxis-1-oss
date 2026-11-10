"""The full praxis inference pipeline: observations in, robot commands out.

One ``PraxisPipeline`` holds the policy, the autoencoder and the text encoder.
One ``Session`` holds the context of one robot episode. Each call of
``Session.step`` takes one observation, builds the model inputs (canvas,
latents of the executed context, proprio, prompt), samples one chunk and
returns the robot commands of that chunk, with the predicted video latent and
the normalized actions.

The pipeline runs the single-arm checkpoint (``"embodiment":
"single_arm"``) on the single-arm rig (:mod:`praxis.single_arm`) and the
bimanual checkpoint (``"embodiment": "bimanual"``) on the bimanual rig
(:mod:`praxis.bimanual`), with the ``window_joint`` sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from praxis import bimanual, single_arm
from praxis.checkpoint import model_variant
from praxis.model import PraxisPolicy, SamplerSettings, TrunkPlan
from praxis.robot_io import NormStats, text_encoder_prompt
from praxis.text_encoder import TextEncoder
from praxis.vae import Autoencoder


@dataclass
class StepResult:
    """Outputs of one policy call.

    ``action_rows_norm`` holds the executed rows of the sampler output in the
    normalized space, with no clip. ``action_rel`` holds the relative actions
    of ``commands``: the normalized xyz and grip dims get a clip to [-1, 1]
    before the inverse q99 map, and the grippers get a clip to [0, 1] after
    it. For rows in these ranges, the clips change no value.
    """

    commands: list[dict]
    action_rows_norm: np.ndarray
    action_rel: np.ndarray
    pred_video_latent: Tensor | None
    pred_action_norm: Tensor
    depth: int
    seed: int
    prompt: str
    inputs: dict = field(default_factory=dict)


class PraxisPipeline:
    """Load all model parts once and open sessions on them."""

    def __init__(
        self,
        checkpoint: str | Path,
        vae_weights: str | Path,
        text_encoder: str | Path,
        device: str | torch.device = "cuda",
        sampler: SamplerSettings | None = None,
        vae_norm_block_c: dict[int, int] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.policy = PraxisPolicy.from_pretrained(checkpoint, device=self.device)
        self.vae = Autoencoder.from_pretrained(
            vae_weights, device=self.device, norm_block_c=vae_norm_block_c
        )
        self.text = TextEncoder(text_encoder, device=self.device)
        self.sampler = sampler or SamplerSettings()
        # The loader refuses every config but the single-arm and bimanual ones.
        self.embodiment = str(self.policy.config["embodiment"])
        self.variant = model_variant(self.policy.config)
        norm = self.policy.config["normalization"]
        self.stats = NormStats(
            action_q01=np.asarray(norm["action_q01"], dtype=np.float32),
            action_q99=np.asarray(norm["action_q99"], dtype=np.float32),
            proprio_mean=np.asarray(norm["proprio_mean"], dtype=np.float32),
            proprio_std=np.asarray(norm["proprio_std"], dtype=np.float32),
        )
        self._text_cache: dict[str, Tensor] = {}

    def encode_text(self, prompt: str) -> Tensor:
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self.text.encode([prompt])[0]
        return self._text_cache[prompt]

    def encode_uncond_text(self) -> Tensor:
        """Return the encoded blank caption of the unguided pass."""
        return self.encode_text(self.variant.blank_caption)

    def session(
        self,
        prompt: str = "",
        seed: int = 0,
        max_context_chunks: int = 1,
        control_mode: str | None = None,
        exec_horizon: int | None = 24,
        sampler: SamplerSettings | None = None,
        skip_last_video: bool = False,
        precompute: bool = False,
    ) -> Session:
        """Open a session for one robot episode.

        The defaults are the recommended settings of the checkpoint.
        ``control_mode`` defaults to ``droid_impedance`` on the single-arm rig
        and ``none`` on the bimanual rig.
        ``sampler`` replaces the sampler settings of the pipeline for this
        session.
        ``skip_last_video`` skips the video decoder on the last Euler step of
        each call: the commands keep their bits, and the call returns no
        predicted video (:meth:`Session.decode_video` refuses it).
        ``precompute`` lets the session take the latents and the trunk passes
        of its next call made ahead of it (:meth:`Session.plan`).
        """
        return Session(
            self,
            prompt,
            seed,
            max_context_chunks,
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            sampler=sampler,
            skip_last_video=skip_last_video,
            precompute=precompute,
        )


class EncodeMemo:
    """The encode of a rig session, with latents of the next call made ahead.

    A real-time session with ``precompute`` encodes the window and the anchor
    frame of its next prediction while the samples arrive
    (:mod:`praxis.serve.precompute`), and keeps each latent here with its
    clip (:meth:`put`, :meth:`store`). A call gets a kept latent only for a
    clip equal to the kept clip (same shape, dtype and device,
    ``torch.equal``). Any other clip is encoded.
    """

    def __init__(self, encode: Callable[[Tensor], Tensor]) -> None:
        self.encode = encode
        self.hits = 0
        self._kept: list[tuple[Tensor, Tensor]] = []

    def __call__(self, clip: Tensor) -> Tensor:
        for kept, latent in self._kept:
            if (
                kept.shape == clip.shape
                and kept.dtype == clip.dtype
                and kept.device == clip.device
                and torch.equal(kept, clip)
            ):
                self.hits += 1
                return latent
        return self.encode(clip)

    def put(self, clip: Tensor, latent: Tensor) -> None:
        """Keep ``latent``, the encode of ``clip``, for the next call."""
        self._kept.append((clip, latent))

    def store(self, clip: Tensor) -> Tensor:
        """Encode ``clip`` now and keep its latent for the next call."""
        latent = self.encode(clip)
        self.put(clip, latent)
        return latent

    def clear(self) -> None:
        self._kept = []
        self.hits = 0


class Session:
    """The context state of one robot episode."""

    def __init__(
        self,
        pipeline: PraxisPipeline,
        prompt: str = "",
        seed: int = 0,
        max_context_chunks: int = 1,
        control_mode: str | None = None,
        exec_horizon: int | None = 24,
        sampler: SamplerSettings | None = None,
        skip_last_video: bool = False,
        precompute: bool = False,
    ) -> None:
        self.pipe = pipeline
        self.sampler = sampler or pipeline.sampler
        self.skip_last_video = bool(skip_last_video)
        self.seed = int(seed)
        variant = model_variant(pipeline.policy.config)
        config = pipeline.policy.config
        # The checkpoint config sets the tiles of the canvas. The canvas and
        # the pose frame of each rig are fixed.
        rig = bimanual if config.get("embodiment") == "bimanual" else single_arm
        if control_mode is None:
            control_mode = "none" if rig is bimanual else "droid_impedance"
        session_cls = (
            bimanual.BimanualServeSession
            if rig is bimanual
            else single_arm.SingleArmServeSession
        )
        self.state = session_cls(
            encode=pipeline.vae.encode,
            stats=pipeline.stats,
            device=pipeline.device,
            prompt=prompt,
            max_context_chunks=max_context_chunks,
            fps=float(config["chunking"].get("grid_hz", rig.FPS)),
            dims=rig.q99_dims(config),
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            n_chunks=int(config["chunking"].get("n_chunks", rig.N_CHUNKS)),
            tile_size=variant.tile_size,
            aspect_tol=variant.aspect_tol,
            antialias=variant.antialias,
        )
        # The trunk passes of the next call, run ahead of it (plan), and
        # the encode that serves its stored latents.
        self._plan: TrunkPlan | None = None
        self.memo: EncodeMemo | None = None
        # "hit" when the last call used its plan, "miss" when its inputs
        # differed from the plan's, None without a plan. memo_hits counts the
        # latents of the last call that came from the memo.
        self.plan_result: str | None = None
        self.memo_hits = 0
        if precompute:
            self.memo = EncodeMemo(self.state.encode)
            self.state.encode = self.memo

    @property
    def has_plan(self) -> bool:
        return self._plan is not None

    def drop_plan(self) -> None:
        """Drop the plan of the next call and the latents kept for it."""
        self._plan = None
        if self.memo is not None:
            self.memo.clear()

    @torch.no_grad()
    def plan(
        self,
        anchor: Tensor,
        cleans: list[Tensor],
        prompt: str,
        skip_last_video: bool = False,
    ) -> None:
        """Run the trunk passes of the next call now, on ``anchor`` and ``cleans``.

        Needs a session opened with ``precompute=True``. The plan samples
        with ``prompt`` and the seed and the settings of the next call
        (:meth:`PraxisPolicy.plan_window_joint`). The next :meth:`step` uses
        it only when all the trunk inputs of the call equal those of the plan
        (:meth:`TrunkPlan.matches`); it then runs only the expert passes.
        Otherwise the call samples as without a plan. One plan is kept at a
        time, and every call drops it. ``skip_last_video`` skips the video
        decoder of the last step: the call then returns no video. A session
        with ``cfg_batched`` makes no plan: the plan runs the guidance passes
        one after the other.
        """
        if self.memo is None:
            raise RuntimeError("plan needs a session opened with precompute")
        if self.sampler.cfg_batched:
            # The plan runs the sequential passes: under cfg_batched a hit would
            # not give the bits of the batched fallback. Sample in the call.
            self._plan = None
            return
        text = self.pipe.encode_text(text_encoder_prompt(prompt))
        uncond = self.pipe.encode_uncond_text() if self.sampler.guided else None
        self._plan = self.pipe.policy.plan_window_joint(
            text=text,
            text_uncond=uncond,
            anchor=anchor,
            cleans=cleans,
            seed=self.seed + self.state.chunk_index,
            settings=self.sampler,
            skip_last_video=skip_last_video,
        )

    @torch.no_grad()
    def step(self, obs: dict, executed_rows: int | None = None) -> StepResult:
        """Run one policy call on ``obs`` and return the commands of the chunk.

        ``executed_rows`` (real-time chunking) is the number of rows that the
        robot ran since the previous call, off the chunk grid (the
        ``prepare`` of the rig session).
        """
        state = self.state
        if self.memo is None:
            inputs = state.prepare(obs, executed_rows)
            video, actions, text = self._sample(inputs)
        else:
            self.plan_result = None
            try:
                inputs = state.prepare(obs, executed_rows)
                self.memo_hits = self.memo.hits
                video, actions, text = self._sample(inputs)
            finally:
                self.drop_plan()
        commands = state.finish(inputs, actions)
        rows, rel = state.decode_actions(inputs, actions)
        return StepResult(
            commands=commands,
            action_rows_norm=rows,
            action_rel=rel,
            pred_video_latent=video,
            pred_action_norm=actions,
            depth=inputs.depth,
            seed=self.seed + inputs.chunk_index,
            prompt=inputs.prompt,
            inputs={
                "text_embedding": text,
                "anchor": inputs.anchor,
                "cleans": inputs.cleans,
                "proprio": inputs.proprio,
                "proprio_all": inputs.proprio_all,
                "boundary_poses": inputs.boundary_poses,
                "pose_now": inputs.pose_now,
            },
        )

    def _sample(
        self,
        inputs: single_arm.SingleArmChunkInputs | bimanual.BimanualChunkInputs,
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        """Sample one chunk. Return the video latent, the actions and the text.

        The video latent is None when the session skips the video of the last
        step (``skip_last_video``).

        A plan of this call (:meth:`plan`) whose inputs equal the call's, made
        with the session's ``skip_last_video``, gives the trunk passes. The
        call then runs only the expert passes. A miss samples as without the
        plan, with the same ``skip_last_video``.
        """
        text = self.pipe.encode_text(inputs.prompt)
        uncond = self.pipe.encode_uncond_text() if self.sampler.guided else None
        plan = getattr(self, "_plan", None)
        if plan is not None:
            self._plan = None
            seed = self.seed + inputs.chunk_index
            if plan.matches(
                text, uncond, inputs.anchor, inputs.cleans, seed, self.sampler
            ):
                self.plan_result = "hit"
                video, actions = self.pipe.policy.finish_window_joint(
                    plan, inputs.proprio_all
                )
                return video, actions, text
            self.plan_result = "miss"
            # The fallback must not keep the taps of the stale plan alive.
            del plan
        video, actions = self.pipe.policy.sample_window_joint(
            text=text,
            text_uncond=uncond,
            anchor=inputs.anchor,
            cleans=inputs.cleans,
            proprio_all=inputs.proprio_all,
            seed=self.seed + inputs.chunk_index,
            settings=self.sampler,
            # Always the session's value, so that False overrides the policy
            # flag. The call with False gives the bits of the stock call.
            skip_last_video=bool(getattr(self, "skip_last_video", False)),
        )
        return video, actions, text

    @torch.no_grad()
    def decode_video(self, result: StepResult) -> Tensor:
        """Decode the predicted video of one call: one frame per command row.

        The decoder is causal over latents. Without context, the first latent
        of the chunk decodes to one frame, so the chunk gives 21 frames. With
        the clean context latents in front, the chunk gives 24 frames that
        cover the same raw frames as the 24 command rows. At depth 0 the chunk
        owns the window anchor, so both decodes give the same 21 frames.

        With ``exec_horizon`` set, the frames of the cut rows are dropped too.

        Returns uint8 frames ``(len(result.commands), 3, H, W)``.
        """
        latent = result.pred_video_latent
        if latent is None:
            raise ValueError(
                "this call skipped the predicted video (skip_last_video); open the"
                " session with skip_last_video=False to decode it"
            )
        cleans = result.inputs["cleans"]
        if cleans:
            dtype = cleans[0].dtype
            latent = torch.cat([*cleans, latent.to(cleans[0].device, dtype)], dim=2)
        frames = self.pipe.vae.decode(latent)
        return frames[-len(result.action_rows_norm) :][: len(result.commands)]
