"""The full praxis inference pipeline: observations in, robot commands out.

One ``PraxisPipeline`` holds the policy, the autoencoder and the text encoder.
One ``Session`` holds the context of one robot episode. Each call of
``Session.step`` takes one observation, builds the model inputs (canvas,
latents of the executed context, proprio, prompt), samples one chunk and
returns the robot commands of that chunk, with the predicted video latent and
the normalized actions.

The pipeline runs the single-arm checkpoint (``"embodiment":
"single_arm"``) on the single-arm rig (:mod:`praxis.single_arm`), with the
``window_joint`` sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from praxis import single_arm
from praxis.checkpoint import model_variant
from praxis.model import PraxisPolicy, SamplerSettings
from praxis.robot_io import NormStats
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
    pred_video_latent: Tensor
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
        # The loader refuses every config but the single-arm one.
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
        control_mode: str = "droid_impedance",
        exec_horizon: int | None = 24,
        sampler: SamplerSettings | None = None,
    ) -> Session:
        """Open a session for one robot episode.

        The defaults are the recommended settings of the checkpoint.
        ``sampler`` replaces the sampler settings of the pipeline for this
        session.
        """
        return Session(
            self,
            prompt,
            seed,
            max_context_chunks,
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            sampler=sampler,
        )


class Session:
    """The context state of one robot episode."""

    def __init__(
        self,
        pipeline: PraxisPipeline,
        prompt: str = "",
        seed: int = 0,
        max_context_chunks: int = 1,
        control_mode: str = "droid_impedance",
        exec_horizon: int | None = 24,
        sampler: SamplerSettings | None = None,
    ) -> None:
        self.pipe = pipeline
        self.sampler = sampler or pipeline.sampler
        self.seed = int(seed)
        variant = model_variant(pipeline.policy.config)
        config = pipeline.policy.config
        # The checkpoint config sets the tiles of the canvas. The canvas and
        # the pose frame of the single-arm rig are fixed.
        self.state = single_arm.SingleArmServeSession(
            encode=pipeline.vae.encode,
            stats=pipeline.stats,
            device=pipeline.device,
            prompt=prompt,
            max_context_chunks=max_context_chunks,
            fps=float(config["chunking"].get("grid_hz", single_arm.FPS)),
            dims=single_arm.q99_dims(config),
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            n_chunks=int(config["chunking"].get("n_chunks", single_arm.N_CHUNKS)),
            tile_size=variant.tile_size,
            aspect_tol=variant.aspect_tol,
            antialias=variant.antialias,
        )

    @torch.no_grad()
    def step(self, obs: dict) -> StepResult:
        """Run one policy call on ``obs`` and return the commands of the chunk."""
        state = self.state
        inputs = state.prepare(obs)
        video, actions, text = self._sample(inputs)
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
                "pose8_now": inputs.pose8_now,
            },
        )

    def _sample(
        self, inputs: single_arm.SingleArmChunkInputs
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample one chunk. Return the video latent, the actions and the text."""
        text = self.pipe.encode_text(inputs.prompt)
        uncond = self.pipe.encode_uncond_text() if self.sampler.guided else None
        video, actions = self.pipe.policy.sample_window_joint(
            text=text,
            text_uncond=uncond,
            anchor=inputs.anchor,
            cleans=inputs.cleans,
            proprio_all=inputs.proprio_all,
            seed=self.seed + inputs.chunk_index,
            settings=self.sampler,
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
        cleans = result.inputs["cleans"]
        if cleans:
            dtype = cleans[0].dtype
            latent = torch.cat([*cleans, latent.to(cleans[0].device, dtype)], dim=2)
        frames = self.pipe.vae.decode(latent)
        return frames[-len(result.action_rows_norm) :][: len(result.commands)]
