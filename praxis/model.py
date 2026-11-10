"""The praxis policy: video backbone, action expert, autoencoder, text encoder.

One policy call denoises one chunk of the future: 6 video latents and 6 action
tokens (24 action rows at 15 Hz). The inputs are the task text, the window
anchor latent, up to three clean chunk latents of the executed past, and the
current arm pose. The backbone denoises the video. Each backbone step also
gives the hidden states ("taps") that the action expert reads to denoise the
actions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from praxis.action_expert import ActionExpert, run_expert_on_taps_k
from praxis.backbone import VideoBackbone, build_backbone
from praxis.checkpoint import load_checkpoint, load_config, model_variant
from praxis.layers import to_device_async
from praxis.sampler import (
    ACTION_SALT,
    VIDEO_SALT,
    check_guidance,
    guidance_combine,
    seeded_normal,
    window_time_grid,
)

LAYOUTS = ("window_joint",)

TraceFn = Callable[[str, Tensor], None]
"""Optional probe ``(name, tensor)`` of the window joint sampler."""


class ChunkCausalTaps(Protocol):
    """The encoder features of one chunk-causal backbone call.

    ``global_spans`` maps ``register``, ``anchor`` and ``text`` to their
    ``(start, length)`` token range. ``noised_span`` and ``clean_span`` are
    the ranges of the noised window and of the clean copies (None for a
    window of one chunk). ``rope_ids`` are the ``(1, S)`` rotary positions of
    the encoder sequence.
    """

    taps: list[Tensor]
    embeddings: Tensor
    rope_ids: Tensor
    global_spans: Mapping[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    frames_per_chunk: int
    tokens_per_frame: int


class ChunkCausalOutput(Protocol):
    velocity: Tensor | None
    taps: ChunkCausalTaps


class ChunkCausalBackbone(Protocol):
    """A backbone with the chunk-causal sequence of the checkpoint."""

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
    ) -> ChunkCausalOutput: ...


@dataclass
class SamplerSettings:
    """Sampler settings of one policy call.

    The defaults are the recommended settings of the checkpoint.

    ``layout`` is ``window_joint``, the sampler of the checkpoint
    (chunk-causal backbone). The window holds the ``d + 1`` chunks up to the
    served one, all noised at the same time and integrated jointly, video and
    actions. It needs the ``euler`` solver and no expert-only sub-steps. See
    :meth:`PraxisPolicy.sample_window_joint`.

    ``cfg_scale`` and ``cfg_video_scale`` are the text guidance scales of the
    action velocity and of the video velocity. 1.0 turns guidance off. The
    guided and the unguided pass run one after the other, and the unguided
    pass uses the encoded blank caption of the checkpoint config with the
    same video and proprio inputs. ``cfg_batched`` has no effect.
    """

    num_steps: int = 2
    t_shift: float = 12.5
    solver: str = "euler"
    action_steps: int | None = None
    layout: str = "window_joint"
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0
    cfg_batched: bool = False

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        check_guidance(self.cfg_scale, self.cfg_video_scale)
        if self.solver != "euler":
            raise ValueError("layout='window_joint' needs solver='euler'")
        if self.action_steps not in (None, self.num_steps):
            raise ValueError("layout='window_joint' has no expert-only sub-steps")

    @property
    def guided(self) -> bool:
        """True if text guidance is on (a scale is not 1.0)."""
        return check_guidance(self.cfg_scale, self.cfg_video_scale)

    @classmethod
    def from_config(cls, s: dict) -> SamplerSettings:
        """Return the settings of the ``sampler`` section of a YAML config.

        The policy server builds its sampler with this function
        (:func:`praxis.serve.server.build_factory`).
        """
        return cls(
            num_steps=int(s["num_steps"]),
            t_shift=float(s["t_shift"]),
            solver=str(s["solver"]),
            action_steps=s.get("action_steps"),
            layout=str(s.get("layout", "window_joint")),
            cfg_scale=float(s.get("cfg_scale", 1.0)),
            cfg_video_scale=float(s.get("cfg_video_scale", 4.0)),
            cfg_batched=bool(s.get("cfg_batched", False)),
        )


@dataclass
class TrunkPlan:
    """The trunk passes of one ``window_joint`` call, run ahead of the call.

    :meth:`PraxisPolicy.plan_window_joint` makes it and
    :meth:`PraxisPolicy.finish_window_joint` runs the expert passes on it.
    The inputs fields hold the tensors that the trunk read. ``taps[s]`` holds
    the taps of the guided pass of step ``s``, ``taps_uncond[s]`` those of the
    unguided pass (None when the action scale is 1.0). ``video`` is the video
    latent of the last chunk, float32, or None for a plan that skipped the
    video decoder of the last step.
    """

    text: Tensor
    text_uncond: Tensor | None
    anchor: Tensor
    cleans: list[Tensor]
    seed: int
    settings: SamplerSettings
    taps: list[ChunkCausalTaps]
    taps_uncond: list[ChunkCausalTaps | None]
    video: Tensor | None

    def matches(
        self,
        text: Tensor,
        text_uncond: Tensor | None,
        anchor: Tensor,
        cleans: list[Tensor],
        seed: int,
        settings: SamplerSettings,
    ) -> bool:
        """Return whether the trunk read exactly these inputs (``torch.equal``)."""
        if int(seed) != self.seed or settings != self.settings:
            return False
        if len(cleans) != len(self.cleans):
            return False
        if (text_uncond is None) != (self.text_uncond is None):
            return False
        pairs = [(text, self.text), (anchor, self.anchor)]
        pairs += list(zip(cleans, self.cleans, strict=True))
        if text_uncond is not None and self.text_uncond is not None:
            pairs.append((text_uncond, self.text_uncond))
        return all(_same_tensor(a, b) for a, b in pairs)


def _same_tensor(a: Tensor, b: Tensor) -> bool:
    """Return whether two tensors have the same shape, dtype, device and values."""
    if a is b:
        return True
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    return bool(torch.equal(a, b))


class PraxisPolicy(nn.Module):
    """Video backbone plus action expert, loaded from a praxis checkpoint."""

    def __init__(
        self, backbone: VideoBackbone, expert: ActionExpert, config: dict
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_expert = expert
        self.config = config
        # True: the last Euler step of sample_window_joint moves only the actions
        # (no video decoder on that step); the sampler then returns no video.
        self.skip_last_video = False

    @classmethod
    def from_config(cls, config: dict) -> PraxisPolicy:
        """Build the model from a checkpoint config, with uninitialized weights.

        The modules go to the current default device. Use
        ``with torch.device("meta")`` to build the model without memory.
        """
        ex = cast(dict, config["action_expert"])
        backbone = build_backbone(cast(dict, config["backbone"]))
        # The variant fields of the trunk-rope expert (rope, global order,
        # attention, head_plus_zero, ...).
        variant_kwargs = model_variant(config).expert_kwargs()
        expert = ActionExpert(
            action_patch_dim=ex["action_patch_dim"],
            proprio_patch_dim=ex["proprio_patch_dim"],
            trunk_dim=ex["trunk_dim"],
            dim=ex["dim"],
            num_layers=ex["num_layers"],
            num_heads=ex["num_heads"],
            head_dim=ex["head_dim"],
            num_future_per_chunk=ex["num_future_per_chunk"],
            max_chunks=ex["max_chunks"],
            max_action_tokens=ex["max_action_tokens"],
            tap_norm=ex["tap_norm"],
            clean_only_conditioning=ex["clean_only_conditioning"],
            **variant_kwargs,
        )
        return cls(backbone, expert, config)

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: str | torch.device = "cuda"
    ) -> PraxisPolicy:
        """Build the model from ``config.json`` and load the weights strictly."""
        with torch.device("meta"):
            model = cls.from_config(load_config(path))
        load_checkpoint(model, path, device=str(device), assign=True)
        return model.eval().requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.encoder.registers.dtype

    @property
    def device(self) -> torch.device:
        return self.backbone.encoder.registers.device

    @torch.no_grad()
    def sample_window_joint(
        self,
        text: Tensor,
        text_uncond: Tensor | None,
        anchor: Tensor,
        cleans: list[Tensor],
        proprio_all: Tensor,
        seed: int,
        settings: SamplerSettings,
        trace: TraceFn | None = None,
        oracle_video: Tensor | None = None,
        return_window: bool = False,
        skip_last_video: bool | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        """Denoise the window of the chunk at depth ``d = len(cleans)`` jointly.

        The sampler of the checkpoint (layout ``window_joint``). The
        window holds ``K' = d + 1`` chunks: the clean copies of the ``d``
        context chunks and ``K'`` noised chunks, all at the same time ``t``.
        Each step makes one backbone call on the whole window and one expert
        call on its taps, and moves the video and the actions of every chunk
        with one Euler step. Only the last chunk is returned. The noised
        history chunks start from their own noise and are dropped.

        * ``text`` is ``(1, S, 2048)``. ``text_uncond`` is the encoded blank
          caption of the unguided pass. Guidance needs it. The unguided pass
          keeps every other input.
        * ``anchor`` is ``(1, 192, 1, h, w)``, each clean chunk
          ``(1, 192, 6, h, w)``. ``proprio_all`` holds the normalized boundary
          pose of each chunk ``(1, K', P)`` float32, current pose last.
        * The noise is float32 from :func:`praxis.sampler.seeded_normal` on the
          CPU: the video ``(1, 192, 6 K', h, w)`` with :data:`VIDEO_SALT` and
          the actions ``(6 K', A)`` with :data:`ACTION_SALT`.
        * The grid is :func:`praxis.sampler.window_time_grid` (float64). The
          states stay float32 and enter the models in the model dtype. The
          step is ``x + (t_next - t_cur) * v`` with a Python float step.
        * ``trace`` receives the intermediate tensors by name.
        * ``oracle_video`` (offline evaluation, ``praxis.eval``) holds the
          clean latents of all ``K'`` window chunks ``(1, 192, 6 K', h, w)``.
          Before each step the video state is set to
          ``(1 - t) * oracle_video + t * noise`` in float32 on the CPU, with
          the video noise of the call. Only the actions integrate: the
          actions read the true future video. The returned video is then
          ``oracle_video``.
        * ``return_window`` returns all ``K'`` chunks instead of the last.
        * ``skip_last_video`` (default None: ``self.skip_last_video``, which
          is False unless set): the last step moves only
          the actions. Its guided pass runs without the video decoder, and its
          unguided pass runs only for action guidance, also without the
          decoder. The action expert reads only the encoder taps, and the
          guidance combine at scale 1.0 is the identity, so the actions keep
          their bits. The video comes back as None. A ``trace`` call runs all
          passes.

        Returns the video latent of the last chunk ``(1, 192, 6, h, w)`` (None
        with ``skip_last_video``) and its normalized action tokens
        ``(1, 6, A)``, float32. With ``return_window`` the shapes are
        ``(1, 192, 6 K', h, w)`` and ``(1, 6 K', A)``.
        """
        if settings.layout != "window_joint":
            raise ValueError(f"layout {settings.layout!r} is not 'window_joint'")
        guided = settings.guided
        if guided and text_uncond is None:
            raise ValueError("text guidance needs the encoded blank caption")
        dev, md = self.device, self.dtype
        lat = int(self.config["chunking"]["latents_per_chunk"])
        max_chunks = int(self.config["chunking"]["n_chunks"])
        d = len(cleans)
        kk = d + 1
        if kk > max_chunks:
            raise ValueError(f"depth {d} is out of range for {max_chunks} chunks")
        pro = proprio_all if proprio_all.ndim == 3 else proprio_all[None]
        if pro.shape[:2] != (1, kk):
            raise ValueError(f"proprio_all {tuple(pro.shape)} for {kk} chunks")
        a_dim = int(self.config["action_expert"]["action_patch_dim"])
        c, h, w = anchor.shape[1], anchor.shape[3], anchor.shape[4]
        if oracle_video is not None and tuple(oracle_video.shape) != (
            1,
            c,
            kk * lat,
            h,
            w,
        ):
            raise ValueError(
                f"oracle_video {tuple(oracle_video.shape)} for {kk} chunks"
            )

        def probe(name: str, t: Tensor) -> None:
            if trace is not None:
                trace(name, t)

        noise_v = seeded_normal((1, c, kk * lat, h, w), seed, VIDEO_SALT)
        probe("noise.salt0", noise_v)
        noise_a = seeded_normal((kk * lat, a_dim), seed, ACTION_SALT)
        probe("noise.salt1", noise_a)
        ts = window_time_grid(settings.num_steps, settings.t_shift)
        probe("sampler.time_grid", torch.tensor(ts, dtype=torch.float64))
        xv = to_device_async(noise_v, dev)
        xa = to_device_async(noise_a, dev)
        anchor_m = anchor.to(dev, md)
        clean = torch.cat([z.to(dev, md) for z in cleans], dim=2) if d else None
        pro = pro.to(dev, torch.float32)
        text_c = text.to(dev, md)
        text_u = None if text_uncond is None else text_uncond.to(dev, md)
        backbone = cast(ChunkCausalBackbone, self.backbone)

        def velocity(
            text_in: Tensor,
            x_v: Tensor,
            x_a: Tensor,
            t_cur: float,
            decode: bool,
            act: bool = True,
        ) -> tuple[Tensor | None, Tensor | None]:
            probe("velocity.in.xt_v", x_v)
            probe("velocity.in.xt_a", x_a)
            probe("velocity.in.t_cur", torch.tensor(t_cur, dtype=torch.float64))
            t_trunk = torch.full((1, kk), t_cur, dtype=md, device=dev)
            out = backbone.forward_chunk_causal(
                text_in, anchor_m, clean, x_v.to(md), t_trunk, decode=decode
            )
            v = None if out.velocity is None else out.velocity.float()
            if v is not None:
                probe("velocity.out.video", v)
            if not act:
                return v, None
            taps = out.taps
            a = run_expert_on_taps_k(
                self.action_expert,
                taps.taps,
                taps.embeddings,
                taps.rope_ids,
                global_spans=taps.global_spans,
                noised_span=taps.noised_span,
                clean_span=taps.clean_span,
                n_chunks=kk,
                frames_per_chunk=taps.frames_per_chunk,
                tokens_per_frame=taps.tokens_per_frame,
                noised_actions=x_a[None],
                action_t_per_chunk=torch.full((1, kk), t_cur, device=dev),
                proprio_per_chunk=pro,
            ).float()[0]
            probe("velocity.out.action", a)
            return v, a

        # The unguided pass computes only what a guidance scale reads: its
        # video with cfg_video_scale != 1, its actions with cfg_scale != 1.
        # guidance_combine returns the guided velocity at scale 1.0, so a
        # skipped part changes no bit. A trace keeps both parts.
        decode_uncond = trace is not None or float(settings.cfg_video_scale) != 1.0
        act_uncond = trace is not None or float(settings.cfg_scale) != 1.0
        skip = self.skip_last_video if skip_last_video is None else skip_last_video
        skip_video = trace is None and bool(skip)
        x0_v = None if oracle_video is None else oracle_video.float().cpu()
        for step in range(settings.num_steps):
            t_cur, t_next = ts[step], ts[step + 1]
            # the last step's video moves only the returned latent: skip its decoder
            video = not (skip_video and step == settings.num_steps - 1)
            if x0_v is not None:
                xv = ((1 - t_cur) * x0_v + t_cur * noise_v).to(dev)
            probe("pass.marker", torch.tensor([step, 0]))
            v, a = velocity(text_c, xv, xa, t_cur, video)
            if a is None or (video and v is None):
                raise RuntimeError(
                    f"the trunk pass of step {step} returned no "
                    f"{'actions' if a is None else 'video velocity'}"
                )
            if guided and (video or act_uncond):
                if text_u is None:
                    raise RuntimeError("guided sampling needs text_uncond")
                probe("pass.marker", torch.tensor([step, 1]))
                v_u, a_u = velocity(
                    text_u, xv, xa, t_cur, video and decode_uncond, act=act_uncond
                )
                for kind, cond, unc, scale in (
                    ("video", v, v_u, settings.cfg_video_scale),
                    ("action", a, a_u, settings.cfg_scale),
                ):
                    if unc is None or cond is None:  # a part not needed
                        continue
                    out = guidance_combine(cond, unc, scale)
                    probe(f"cfg_combine.{kind}.cond", cond)
                    probe(f"cfg_combine.{kind}.uncond", unc)
                    probe(f"cfg_combine.{kind}.out", out)
                    if kind == "video":
                        v = out
                    else:
                        a = out
            dt = t_next - t_cur
            if video and v is not None:
                xv = xv + dt * v
            xa = xa + dt * a
        if x0_v is not None:
            xv = x0_v.to(dev)
        probe("out.video", xv)
        probe("out.action", xa)
        if return_window:
            return (None if skip_video else xv), xa[None]
        video_last = None if skip_video else xv[:, :, d * lat :].contiguous()
        actions = xa[d * lat :].contiguous()
        if video_last is not None:
            probe("out.video_last", video_last)
        probe("out.action_last", actions)
        return video_last, actions[None]

    @torch.no_grad()
    def plan_window_joint(
        self,
        text: Tensor,
        text_uncond: Tensor | None,
        anchor: Tensor,
        cleans: list[Tensor],
        seed: int,
        settings: SamplerSettings,
        skip_last_video: bool = False,
    ) -> TrunkPlan:
        """Run the trunk passes of :meth:`sample_window_joint` ahead of the call.

        No trunk pass reads the actions or the proprio: the video Euler state
        moves with the video velocity only. So every backbone call of a
        ``window_joint`` call can run once the window frames are known, before
        the current pose arrives. This runs them, in the order and with the
        inputs of :meth:`sample_window_joint`, and keeps the taps of each pass
        that the expert reads. :meth:`finish_window_joint` then runs the
        expert passes. Together they give the bits of
        :meth:`sample_window_joint`: the same calls on the same inputs, with
        only the order between the backbone and the expert changed. A trace
        is not supported.

        ``skip_last_video`` runs the last step without the video decoder: the
        last step moves only the actions, which read only the encoder taps.
        The actions keep their bits, and the plan has no video (None).

        ``settings.cfg_batched`` raises ``ValueError``: the plan runs the
        guided and the unguided passes one after the other.
        """
        if settings.layout != "window_joint":
            raise ValueError(f"layout {settings.layout!r} is not 'window_joint'")
        if settings.cfg_batched:
            raise ValueError(
                "plan_window_joint runs the guidance passes one after the other; "
                "it does not plan cfg_batched"
            )
        guided = settings.guided
        if guided and text_uncond is None:
            raise ValueError("text guidance needs the encoded blank caption")
        dev, md = self.device, self.dtype
        lat = int(self.config["chunking"]["latents_per_chunk"])
        d = len(cleans)
        kk = d + 1
        if kk > int(self.config["chunking"]["n_chunks"]):
            raise ValueError(f"depth {d} is out of range")
        c, h, w = anchor.shape[1], anchor.shape[3], anchor.shape[4]
        noise_v = seeded_normal((1, c, kk * lat, h, w), seed, VIDEO_SALT)
        ts = window_time_grid(settings.num_steps, settings.t_shift)
        xv = to_device_async(noise_v, dev)
        anchor_m = anchor.to(dev, md)
        clean = torch.cat([z.to(dev, md) for z in cleans], dim=2) if d else None
        text_c = text.to(dev, md)
        text_u = None if text_uncond is None else text_uncond.to(dev, md)
        backbone = cast(ChunkCausalBackbone, self.backbone)
        # The passes of sample_window_joint without a trace.
        decode_uncond = float(settings.cfg_video_scale) != 1.0
        act_uncond = float(settings.cfg_scale) != 1.0
        taps: list[ChunkCausalTaps] = []
        taps_uncond: list[ChunkCausalTaps | None] = []
        for step in range(settings.num_steps):
            t_cur, t_next = ts[step], ts[step + 1]
            video = not (skip_last_video and step == settings.num_steps - 1)
            t_trunk = torch.full((1, kk), t_cur, dtype=md, device=dev)
            out = backbone.forward_chunk_causal(
                text_c, anchor_m, clean, xv.to(md), t_trunk, decode=video
            )
            if video and out.velocity is None:
                raise RuntimeError(f"the trunk pass of step {step} gave no video")
            v = None if out.velocity is None else out.velocity.float()
            taps.append(out.taps)
            unc_taps = None
            if guided and (video or act_uncond):
                assert text_u is not None
                t_trunk = torch.full((1, kk), t_cur, dtype=md, device=dev)
                out_u = backbone.forward_chunk_causal(
                    text_u,
                    anchor_m,
                    clean,
                    xv.to(md),
                    t_trunk,
                    decode=video and decode_uncond,
                )
                if v is not None and out_u.velocity is not None:
                    v = guidance_combine(
                        v, out_u.velocity.float(), settings.cfg_video_scale
                    )
                if act_uncond:
                    unc_taps = out_u.taps
            taps_uncond.append(unc_taps)
            if v is not None:
                xv = xv + (t_next - t_cur) * v
        # Copies of the inputs that matches() checks: a caller that changes
        # its tensors or its settings after the plan does not change the plan.
        return TrunkPlan(
            text=text.clone(),
            text_uncond=None if text_uncond is None else text_uncond.clone(),
            anchor=anchor.clone(),
            cleans=[z.clone() for z in cleans],
            seed=int(seed),
            settings=dataclasses.replace(settings),
            taps=taps,
            taps_uncond=taps_uncond,
            video=None if skip_last_video else xv[:, :, d * lat :].contiguous(),
        )

    @torch.no_grad()
    def finish_window_joint(
        self, plan: TrunkPlan, proprio_all: Tensor
    ) -> tuple[Tensor | None, Tensor]:
        """Run the expert passes of a :class:`TrunkPlan` on ``proprio_all``.

        Returns the video latent (None if the plan skipped it) and the actions
        of :meth:`sample_window_joint` on the inputs of the plan.
        """
        settings = plan.settings
        dev = self.device
        lat = int(self.config["chunking"]["latents_per_chunk"])
        d = len(plan.cleans)
        kk = d + 1
        pro = proprio_all if proprio_all.ndim == 3 else proprio_all[None]
        if pro.shape[:2] != (1, kk):
            raise ValueError(f"proprio_all {tuple(pro.shape)} for {kk} chunks")
        a_dim = int(self.config["action_expert"]["action_patch_dim"])
        noise_a = seeded_normal((kk * lat, a_dim), plan.seed, ACTION_SALT)
        ts = window_time_grid(settings.num_steps, settings.t_shift)
        xa = to_device_async(noise_a, dev)
        pro = pro.to(dev, torch.float32)

        def expert(taps: ChunkCausalTaps, x_a: Tensor, t_cur: float) -> Tensor:
            return run_expert_on_taps_k(
                self.action_expert,
                taps.taps,
                taps.embeddings,
                taps.rope_ids,
                global_spans=taps.global_spans,
                noised_span=taps.noised_span,
                clean_span=taps.clean_span,
                n_chunks=kk,
                frames_per_chunk=taps.frames_per_chunk,
                tokens_per_frame=taps.tokens_per_frame,
                noised_actions=x_a[None],
                action_t_per_chunk=torch.full((1, kk), t_cur, device=dev),
                proprio_per_chunk=pro,
            ).float()[0]

        for step in range(settings.num_steps):
            t_cur, t_next = ts[step], ts[step + 1]
            a = expert(plan.taps[step], xa, t_cur)
            unc_taps = plan.taps_uncond[step]
            if unc_taps is not None:
                a = guidance_combine(a, expert(unc_taps, xa, t_cur), settings.cfg_scale)
            xa = xa + (t_next - t_cur) * a
        return plan.video, xa[d * lat :].contiguous()[None]
