"""Settings of a bimanual YAM sim run, read from YAML.

``configs/sim_yam.yaml`` holds the settings of a run of
``scripts/run_sim_yam.py``.

Blocks:

* ``model``: the checkpoint and the weights (as ``configs/serve_yam.yaml``).
* ``sampler``: the sampler of each call. The defaults are the serve defaults
  of the bimanual rig: 2 Euler steps of the ``window_joint`` layout, flow
  shift 1, no text guidance.
* ``session``: the schedule (``rtc``, real-time chunking as on the robot,
  or ``chunked``), the latency of a real-time call, context chunks, rows per
  chunked call and the sampler seed.
* ``player``: the smoothing of the executed poses (Savitzky-Golay, 7 rows,
  order 2, as the robot client and the hosted demo play them).
* ``policy``: the predicted video of the model policy.
* ``episodes``: the stop rule and a scale of the time limits.
* ``scene``: image size, shadows, anti-aliasing, the asset cache and an
  optional rig profile (:func:`praxis.sim.yam_scene.load_rig_profile`).
* ``render``: the EGL device of the renderer.
* ``output``: the MP4 per episode and its quality.

The episode seed (the ``task:seed`` units of ``scripts/run_sim_yam.py``) sets
the start state. ``session.seed`` is the sampler seed (``--sampler-seed``).
Chunk ``k`` of the episode with seed ``s`` samples with the seed
``session.seed + session.seed_stride * s + k``.

``--set key.sub=value`` items change single values, as in
:func:`praxis.sim.config.read_yaml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from praxis.sim.config import from_dict, read_yaml
from praxis.sim.yam_loop import (
    YamHoldPolicy,
    YamModelPolicy,
    YamOraclePolicy,
    YamPolicy,
    YamSessionConfig,
)
from praxis.sim.yam_player import YamPlayerConfig
from praxis.sim.yam_scene import YamSceneConfig

YAM_POLICIES: tuple[str, ...] = ("model", "oracle", "hold")


@dataclass
class YamModelConfig:
    """Paths of the weights."""

    checkpoint: str = "checkpoints/praxis-1-yam"
    vae_weights: str = "weights/Wan2.2_VAE.pth"
    text_encoder: str = "weights/Qwen3.5-2B"
    device: str = "cuda"


@dataclass
class YamSamplerConfig:
    """Sampler settings of each policy call (the serve defaults of the bimanual rig)."""

    num_steps: int = 2
    cfg_scale: float = 1.0
    cfg_video_scale: float = 1.0
    t_shift: float = 1.0


@dataclass
class YamPolicySettings:
    """``decode_video``: the model policy also decodes the predicted video of
    each call for the MP4. The decode takes GPU time but does not change the
    commands."""

    decode_video: bool = False


@dataclass
class YamEpisodesConfig:
    """The stop rule of the episodes."""

    stop_on_success: bool = True
    after_success_s: float = 1.0
    time_scale: float = 1.0  # multiplies the time limit of each task


@dataclass
class YamSceneSettings:
    """The scene settings that a run can change."""

    assets_dir: str | None = None
    asset_source: str | None = None
    download: bool = True
    image_width: int = 416
    image_height: int = 260
    shadow_size: int = 2048
    msaa_samples: int = 4
    # JSON file of one real cell (None = the generic cell)
    rig_profile: str | None = None


@dataclass
class YamRenderConfig:
    """``egl_device``: the physical GPU of the renderer.

    None = a ``CUDA_VISIBLE_DEVICES`` GPU that does not run the model when
    there is one (see :func:`render_gpu`), else the model's GPU.
    """

    egl_device: int | None = None


def render_gpu(egl_device: int | None, model_device: str | None) -> int | None:
    """Return the physical GPU of the EGL renderer.

    An EGL renderer that shares the GPU with CUDA work can return black frames
    without an error. With ``egl_device`` None and a model on
    ``model_device`` (``cuda`` or ``cuda:N``), the renderer takes the first
    ``CUDA_VISIBLE_DEVICES`` GPU that is not the model's. None keeps the
    default of :func:`praxis.sim.loop.select_egl_device` (one visible GPU, or
    no model).
    """
    if egl_device is not None or model_device is None:
        return egl_device
    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return None
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(visible) < 2 or not all(v.isdigit() for v in visible):
        return None
    _, _, index = model_device.partition(":")
    model_gpu = (
        visible[int(index)]
        if index.isdigit() and int(index) < len(visible)
        else visible[0]
    )
    return next(int(v) for v in visible if v != model_gpu)


@dataclass
class YamOutputConfig:
    """The outputs of a run (the directory is ``--out``)."""

    video: bool = True
    video_crf: int = 23


@dataclass
class YamSuiteConfig:
    """All the settings of one suite run."""

    model: YamModelConfig = field(default_factory=YamModelConfig)
    sampler: YamSamplerConfig = field(default_factory=YamSamplerConfig)
    session: YamSessionConfig = field(default_factory=YamSessionConfig)
    player: YamPlayerConfig = field(default_factory=YamPlayerConfig)
    policy: YamPolicySettings = field(default_factory=YamPolicySettings)
    episodes: YamEpisodesConfig = field(default_factory=YamEpisodesConfig)
    scene: YamSceneSettings = field(default_factory=YamSceneSettings)
    render: YamRenderConfig = field(default_factory=YamRenderConfig)
    output: YamOutputConfig = field(default_factory=YamOutputConfig)

    def scene_config(self) -> YamSceneConfig:
        """Return the scene settings of the environment."""
        sc = self.scene
        return YamSceneConfig(
            assets_dir=sc.assets_dir,
            asset_source=sc.asset_source,
            download=sc.download,
            image_width=sc.image_width,
            image_height=sc.image_height,
            shadow_size=sc.shadow_size,
            msaa_samples=sc.msaa_samples,
            rig_profile=sc.rig_profile,
        )


def load_yam_config(
    path: str | Path, overrides: list[str] | None = None
) -> YamSuiteConfig:
    """Read a YAML file, apply the overrides and return the settings."""
    return from_dict(YamSuiteConfig, read_yaml(path, overrides))


def make_yam_policy(kind: str, cfg: YamSuiteConfig) -> YamPolicy:
    """Return the policy ``kind`` (``model``, ``oracle`` or ``hold``) of ``cfg``.

    The model policy loads the weights of ``cfg.model``. Raise ``SystemExit``
    when a model path does not exist or the checkpoint is not of the bimanual
    rig.
    """
    if kind == "oracle":
        return YamOraclePolicy()
    if kind == "hold":
        return YamHoldPolicy()
    if kind != "model":
        raise ValueError(f"unknown policy {kind!r}; the policies are {YAM_POLICIES}")
    from praxis.model import SamplerSettings

    m, s = cfg.model, cfg.sampler
    missing = [
        p for p in (m.checkpoint, m.vae_weights, m.text_encoder) if not Path(p).exists()
    ]
    if missing:
        raise SystemExit(
            f"these model paths do not exist: {missing}. Relative paths start at "
            "the current directory. Change them in the config or with --set."
        )
    from praxis.checkpoint import load_config

    embodiment = load_config(m.checkpoint)["embodiment"]
    if embodiment != "bimanual":
        raise SystemExit(
            f"{m.checkpoint} is a {embodiment!r} checkpoint; the YAM sim runs the "
            "bimanual checkpoint (praxis-1-yam)."
        )
    return YamModelPolicy.load(
        m.checkpoint,
        m.vae_weights,
        m.text_encoder,
        session=cfg.session,
        device=m.device,
        sampler=SamplerSettings(
            num_steps=s.num_steps,
            cfg_scale=s.cfg_scale,
            cfg_video_scale=s.cfg_video_scale,
            t_shift=s.t_shift,
        ),
        decode_video=cfg.policy.decode_video,
        player=cfg.player,
    )
