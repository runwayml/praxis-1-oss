"""The settings of a sim run: one dataclass per YAML block, and the YAML loader.

A run reads one YAML file. ``--set key.sub=value`` items change single values
(the same convention as ``scripts/run_inference.py``)::

    cfg = load_config("configs/sim_droid.yaml", ["task.prompt=Lift the cube."])

The YAML blocks:

* ``model``, ``sampler``, ``session``: the checkpoint and the serve settings
  of the policy (the same keys as ``configs/inference_droid.yaml``).
* ``policy``: ``model``, ``scripted`` or ``hold``.
* ``task``: the task name, the prompt, the time limit and the object
  placement (:class:`praxis_sim.tasks.TaskConfig`).
* ``episodes``: the seeds and the stop rule.
* ``scene``: images, cameras, lights and physics
  (:class:`praxis_sim.scene.SceneConfig`).
* ``robot``: the policy frame, the inverse kinematics and the reset pose.
* ``render``: the EGL device of the renderer.
* ``output``: the output directory, the MP4 and the arrays.
* ``viewer``: the live view in a web browser.

The loader converts each value to the type of its field. A key that the
dataclass does not have causes a ``ValueError``. YAML reads ``1e-3`` (no
dot) as a string. The loader converts such strings to numbers for the
number fields.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from praxis_sim.env import EnvConfig
from praxis_sim.robot import DROID_HOME, FrameConfig, IkConfig
from praxis_sim.scene import SceneConfig
from praxis_sim.scripted import ScriptedConfig
from praxis_sim.tasks import TaskConfig

POLICIES: tuple[str, ...] = ("model", "scripted", "hold")


@dataclass
class ModelConfig:
    """Paths of the weights. The YAML ``model`` block fills it."""

    checkpoint: str = "checkpoints/praxis-1-droid"
    vae_weights: str = "weights/Wan2.2_VAE.pth"
    text_encoder: str = "weights/Qwen3.5-2B"
    device: str = "cuda"
    vae_norm_block_c: dict[int, int] | None = None  # None = the default table
    routed_precision: str = "bf16"
    nvfp4_cache: str | None = None
    action_cuda_graphs: bool = True
    nvfp4_moe_cuda_graphs: bool = True


@dataclass
class SamplerConfig:
    """Sampler settings of each policy call. The YAML ``sampler`` block fills it.

    The defaults are the recommended settings of the single-arm checkpoint.
    They are 2 Euler steps of the ``window_joint`` layout with the flow shift
    12.5, text guidance 4.0 on the video and none on the actions (scale 1.0).
    """

    num_steps: int = 2
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0
    cfg_batched: bool = False
    layout: str = "window_joint"
    t_shift: float = 12.5
    solver: str = "euler"
    action_steps: int | None = None


@dataclass
class SessionConfig:
    """Session settings of the policy. The YAML ``session`` block fills it.

    Chunk ``k`` of the episode with seed ``s`` samples with the seed
    ``seed + seed_stride * s + k``. The defaults are the recommended settings
    of the single-arm checkpoint. They are 24 command rows per call and 1
    context chunk.
    """

    exec_horizon: int | None = 24
    control_mode: str = "droid_impedance"
    max_context_chunks: int = 1
    seed: int = 0
    seed_stride: int = 1000


@dataclass
class PolicyConfig:
    """The policy that drives the arm. The YAML ``policy`` block fills it."""

    type: str = "model"  # model | scripted | hold
    decode_video: bool = True  # model only: show the predicted video
    scripted: ScriptedConfig = field(default_factory=ScriptedConfig)


@dataclass
class EpisodesConfig:
    """The episodes of a run. The YAML ``episodes`` block fills it."""

    seeds: list[int] = field(default_factory=lambda: [0])
    stop_on_success: bool = True
    after_success_s: float = 1.0  # time that the run continues after success


@dataclass
class RobotConfig:
    """The arm settings. The YAML ``robot`` block fills it."""

    frame: FrameConfig = field(default_factory=FrameConfig)
    ik: IkConfig = field(default_factory=IkConfig)
    home: tuple[float, ...] = DROID_HOME  # arm joints at reset, radians
    home_noise: float = 0.0  # uniform joint noise at reset, radians
    settle_s: float = 0.5  # physics time before the first observation
    feedforward: bool = True  # joint velocity feed-forward in each tick


@dataclass
class RenderConfig:
    """The renderer settings. The YAML ``render`` block fills it.

    ``egl_device`` is the physical GPU index of the EGL renderer. ``None``
    selects the first GPU of ``CUDA_VISIBLE_DEVICES``.
    """

    egl_device: int | None = None


@dataclass
class OutputConfig:
    """The outputs of a run. The YAML ``output`` block fills it."""

    dir: str = "outputs/sim"
    video: bool = True  # write one MP4 per episode
    video_crf: int = 20  # H.264 quality (lower is better)
    save_arrays: bool = True  # write the poses and commands of each tick (.npz)


@dataclass
class ViewerConfig:
    """The live view in a web browser. The YAML ``viewer`` block fills it.

    ``keep_open_s`` is the time that the viewer stays open after the run.
    ``None`` keeps it open until Ctrl-C. ``0`` stops it at once.
    """

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    max_fps: float = 15.0
    jpeg_quality: int = 80
    keep_open_s: float | None = 600.0
    pace_realtime: bool = False  # with a viewer: at most 15 ticks per second


@dataclass
class SimConfig:
    """All the settings of one sim run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    episodes: EpisodesConfig = field(default_factory=EpisodesConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    def __post_init__(self) -> None:
        if self.policy.type not in POLICIES:
            raise ValueError(
                f"policy.type must be one of {POLICIES}, got {self.policy.type!r}"
            )
        if not self.episodes.seeds:
            raise ValueError("episodes.seeds must hold one seed or more")

    @property
    def needs_images(self) -> bool:
        """Return ``True`` when the run must render the camera views."""
        return self.policy.type == "model" or self.output.video or self.viewer.enabled

    def env_config(self) -> EnvConfig:
        """Return the environment settings of the run."""
        r = self.robot
        return EnvConfig(
            scene=self.scene,
            task=self.task,
            frame=r.frame,
            ik=r.ik,
            home=tuple(r.home),
            home_noise=r.home_noise,
            settle_s=r.settle_s,
            feedforward=r.feedforward,
            render=self.needs_images,
            render_overview=self.output.video or self.viewer.enabled,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def read_yaml(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Read a YAML file and apply ``key.sub=value`` overrides.

    YAML parses each override value, so ``[0, 1]`` gives a list and ``null``
    gives ``None``.
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"an override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            child = node.get(p)
            if not isinstance(child, dict):
                child = {}
                node[p] = child
            node = child
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> SimConfig:
    """Read a YAML file, apply the overrides and return the settings."""
    return from_dict(SimConfig, read_yaml(path, overrides))


def to_dict(cfg: object) -> dict:
    """Return the settings as plain YAML types (tuples become lists)."""

    def plain(value: object) -> object:
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [plain(v) for v in value]
        return value

    if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError(f"to_dict needs a dataclass instance, got {type(cfg)}")
    out = plain(dataclasses.asdict(cfg))
    assert isinstance(out, dict)
    return out


T = typing.TypeVar("T")


def from_dict(cls: type[T], data: dict | None, path: str = "") -> T:
    """Build the dataclass ``cls`` from a mapping. Missing keys keep their defaults."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'config'} must be a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"from_dict needs a dataclass type, got {cls!r}")
    fields = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in {path or 'the config'}. "
            f"The known keys are {sorted(fields)}."
        )
    kwargs = {
        name: _coerce(hints[name], value, f"{path}.{name}" if path else name)
        for name, value in data.items()
    }
    return cls(**kwargs)


def _coerce(hint: object, value: object, path: str) -> object:
    """Convert one YAML value to the type ``hint``. Raise ``ValueError`` on a mismatch."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError("; ".join(errors))
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping, got {value!r}")
        return from_dict(hint, value, path)
    if hint is typing.Any:
        return value
    if value is None:
        raise ValueError(f"{path} must not be null")
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{path} must be true or false, got {value!r}")
    if hint is int:
        return _to_int(value, path)
    if hint is float:
        return _to_float(value, path)
    if hint is str:
        if isinstance(value, str):
            return value
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
        raise ValueError(f"{path} must be a string, got {value!r}")
    if origin is tuple:
        seq = _as_list(value, path)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(seq))
        if len(seq) != len(args):
            raise ValueError(f"{path} must hold {len(args)} values, got {len(seq)}")
        return tuple(
            _coerce(a, v, f"{path}[{i}]")
            for i, (a, v) in enumerate(zip(args, seq, strict=True))
        )
    if origin is list:
        seq = value if isinstance(value, list | tuple) else [value]
        return [_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(seq)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping, got {value!r}")
        return {
            _coerce(args[0], k, f"{path} key"): _coerce(args[1], v, f"{path}[{k}]")
            for k, v in value.items()
        }
    raise ValueError(f"{path}: the loader does not know the type {hint!r}")


def _as_list(value: object, path: str) -> list:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{path} must be a list, got {value!r}")
    return list(value)


def _to_float(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a number, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"{path} must be a number, got {value!r}")


def _to_int(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    number = _to_float(value, path)
    if not number.is_integer():
        raise ValueError(f"{path} must be an integer, got {value!r}")
    return int(number)
