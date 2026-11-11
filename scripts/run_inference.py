"""Run praxis on recorded episodes and save the commands, actions and video.

Usage:
    python scripts/run_inference.py --config configs/inference_droid.yaml [--set key=value ...]

An episode directory holds:

* ``meta.json``: ``{"prompt": str, "fps": 15, "num_frames": N, "views":
  ["exterior", "wrist"]}``.
* ``frames/<view>/<index:03d>.png`` for each view in ``meta.json``.
* ``observed_pose8.npy``: ``(N, 8)`` float32 observed arm poses,
  ``[xyz3 quat_wxyz4 grip1]``.

The script replays each episode as one robot session: one policy call at each
frame of ``input.chunk_starts``. After call 0, each observation holds the
executed frames of the previous chunk as a frame stack.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from praxis import single_arm
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline, StepResult

SINGLE_ARM_VIEW_KEYS = {"exterior": "image.exterior", "wrist": "image.wrist"}


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def build_pipeline(cfg: dict) -> PraxisPipeline:
    """Load the pipeline with the ``model`` and ``sampler`` settings of a config."""
    m, s = cfg["model"], cfg["sampler"]
    return PraxisPipeline(
        checkpoint=m["checkpoint"],
        vae_weights=m["vae_weights"],
        text_encoder=m["text_encoder"],
        device=m.get("device", "cuda"),
        decoder_attention_backend=m.get("decoder_attention_backend"),
        encoder_attention_backend=m.get("encoder_attention_backend"),
        vae_norm_block_c=(
            {int(k): int(v) for k, v in m["vae_norm_block_c"].items()}
            if m.get("vae_norm_block_c")
            else None
        ),
        routed_precision=str(m.get("routed_precision", "bf16")),
        nvfp4_cache=m.get("nvfp4_cache"),
        action_cuda_graphs=bool(m.get("action_cuda_graphs", True)),
        nvfp4_moe_cuda_graphs=bool(m.get("nvfp4_moe_cuda_graphs", True)),
        sampler=SamplerSettings(
            num_steps=int(s["num_steps"]),
            t_shift=float(s["t_shift"]),
            solver=str(s["solver"]),
            action_steps=s.get("action_steps"),
            layout=str(s.get("layout", "window_joint")),
            cfg_scale=float(s.get("cfg_scale", 1.0)),
            cfg_video_scale=float(s.get("cfg_video_scale", 4.0)),
            cfg_batched=bool(s.get("cfg_batched", False)),
        ),
    )


def _load_frames(path: Path, views: list[str] | dict[str, str]) -> dict:
    frames = {}
    for view in views:
        files = sorted((path / "frames" / view).glob("*.png"))
        frames[view] = np.stack(
            [np.asarray(Image.open(f).convert("RGB")) for f in files]
        )
    return frames


def load_single_arm_episode(path: Path) -> dict:
    """Load an episode of the single-arm rig. The views come from ``meta.json``."""
    meta = json.loads((path / "meta.json").read_text())
    views = list(meta.get("views", single_arm.VIEWS))
    unknown = sorted(set(views) - set(SINGLE_ARM_VIEW_KEYS))
    if unknown or "exterior" not in views:
        raise ValueError(f"{path}: views must be exterior (+ wrist), got {views}")
    frames = _load_frames(path, views)
    pose8 = np.load(path / "observed_pose8.npy").astype(np.float32)
    return {"name": path.name, "meta": meta, "frames": frames, "pose8": pose8}


def build_single_arm_obs(ep: dict, call: int, frame: int, stack_frames: int) -> dict:
    """The observation at ``frame``: a frame stack after call 0, one frame at
    call 0. The grip stack is ``(T, 1)``."""
    lo = frame if call == 0 else max(0, frame - (stack_frames - 1))
    sl = slice(lo, frame + 1)
    p = ep["pose8"]
    obs: dict = {
        SINGLE_ARM_VIEW_KEYS[view]: frames[sl] for view, frames in ep["frames"].items()
    }
    obs["robot_state.ee_pose"] = np.ascontiguousarray(p[sl, 0:7])
    obs["grip"] = np.ascontiguousarray(p[sl, 7:8])
    obs["task"] = ep["meta"]["prompt"]
    return obs


def call_record(
    res: StepResult, encodes: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict:
    """Collect the outputs of one call for ``call<k>.pt``.

    ``encodes`` holds the ``(clip, latent)`` pairs of the video encodes of
    the call. The record keeps them.
    """
    rec: dict = {
        "depth": res.depth,
        "seed": res.seed,
        "prompt": res.prompt,
        "pred_video_latent": res.pred_video_latent.cpu(),
        "pred_action_norm": res.pred_action_norm.cpu(),
        "action_rows_norm": torch.from_numpy(res.action_rows_norm),
    }
    rec["action_rel10"] = torch.from_numpy(res.action_rel)
    poses, _ = single_arm.wire_to_arrays(res.commands)
    rec["command_pose"] = torch.from_numpy(poses)
    rec["command_grip"] = torch.tensor(
        [c["target_grip"] for c in res.commands], dtype=torch.float64
    )
    rec["command_timestamp"] = torch.tensor(
        [c["timestamp"] for c in res.commands], dtype=torch.float64
    )
    modes = [c["robot_command"].get("mode") for c in res.commands]
    rec["command_mode"] = modes[0] if modes else None
    inp = res.inputs
    rec["inputs"] = {
        "latent_anchor": inp["anchor"].cpu(),
        "latent_cleans": [c.cpu() for c in inp["cleans"]],
        "proprio": inp["proprio"].cpu(),
        "proprio_all": inp["proprio_all"].cpu(),
        "boundary_poses8": torch.from_numpy(inp["boundary_poses"].copy()),
        "pose8_now": torch.from_numpy(inp["pose_now"].copy()),
        "text_embedding": inp["text_embedding"].cpu(),
    }
    # The session encodes the current frame first. With context it then
    # encodes the window prefix and the window start frame.
    names = ["boundary", "context", "anchor"] if res.depth > 0 else ["boundary"]
    for name, (clip, latent) in zip(names, encodes, strict=True):
        rec["inputs"][f"vae_in_{name}"] = clip
        rec["inputs"][f"latent_{name}"] = latent
    return rec


class EncodeRecorder:
    """Wrap a video encoder and keep CPU copies of the clips and latents."""

    def __init__(self, encode: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.encode = encode
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        latent = self.encode(clip)
        self.calls.append((compact_clip(clip), latent.detach().cpu()))
        return latent

    def take(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the calls since the last ``take`` and forget them."""
        out, self.calls = self.calls, []
        return out


def compact_clip(clip: torch.Tensor) -> torch.Tensor:
    """Return a CPU copy of a clip: uint8 when all values are integers, else float32."""
    c = clip.detach().cpu().contiguous()
    u = c.round().clamp(0, 255).to(torch.uint8)
    return u if torch.equal(u.to(c.dtype), c) else c


def save_video(path: Path, frames: torch.Tensor, fps: float) -> None:
    """Save uint8 frames ``(T, 3, H, W)`` as ``.npy`` and, when possible, ``.mp4``."""
    arr = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    np.save(path.with_suffix(".npy"), arr)
    try:
        import imageio.v3 as iio

        iio.imwrite(path.with_suffix(".mp4"), arr, fps=fps, codec="libx264")
    except Exception as exc:  # mp4 export is optional
        print(f"mp4 export skipped for {path.name}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set", action="append", default=[], help="override, e.g. sampler.num_steps=4"
    )
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    pipe = build_pipeline(cfg)
    root = Path(cfg["input"]["episodes_dir"])
    names = cfg["input"]["episodes"]
    dirs = (
        sorted(p for p in root.iterdir() if p.is_dir())
        if names == "all"
        else [root / n for n in names]
    )
    out_root = Path(cfg["output"]["dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sess_cfg = cfg["session"]
    stack = int(cfg["input"]["stack_frames"])
    for ep_dir in dirs:
        ep = load_single_arm_episode(ep_dir)
        sess = pipe.session(
            prompt=ep["meta"]["prompt"],
            seed=int(sess_cfg["seed"]),
            max_context_chunks=int(sess_cfg["max_context_chunks"]),
            control_mode=str(sess_cfg.get("control_mode", "none")),
            exec_horizon=sess_cfg.get("exec_horizon"),
        )
        recorder = EncodeRecorder(sess.state.encode)
        sess.state.encode = recorder
        out = out_root / ep["name"]
        out.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(cfg["input"]["chunk_starts"]):
            obs = build_single_arm_obs(ep, k, int(frame), stack)
            t0 = time.monotonic()
            res = sess.step(obs)
            torch.cuda.synchronize()
            dt = time.monotonic() - t0
            rec = call_record(res, recorder.take())
            rec["seconds"] = dt
            if cfg["output"].get("save_video", True):
                # One frame per command row: the chunk decoded after its context.
                frames = sess.decode_video(res)
                rec["pred_frames"] = frames.cpu()
                # The chunk decoded alone (21 frames; frame 0 is a key frame).
                rec["pred_frames_standalone"] = pipe.vae.decode(
                    res.pred_video_latent
                ).cpu()
                save_video(
                    out / f"call{k}_video", frames, float(ep["meta"].get("fps", 15))
                )
            torch.save(rec, out / f"call{k}.pt")
            print(
                f"{ep['name']} call {k}: depth {res.depth}, {len(res.commands)} rows, {dt * 1e3:.0f} ms"
            )


if __name__ == "__main__":
    main()
