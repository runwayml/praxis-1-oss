"""Start the praxis policy server on this host.

Usage:
    python -m praxis.serve --config configs/serve_droid.yaml [--host 0.0.0.0] [--port 8000]

The config holds the model paths and the default value of each session
parameter. A flag replaces one value of the config. ``--set key=value``
replaces any value, for example
``--set "model.vae_norm_block_c={160: 64, 320: 64, 640: 128}"``.
A Positronic client connects to ``ws://<host>:<port>/api/v1/session``.
"""

from __future__ import annotations

import argparse
import logging

from praxis.serve.server import build_server, load_config, serve

# CLI flag -> config key. A flag that is not set keeps the config value.
FLAGS: dict[str, tuple[str, type]] = {
    "decoder_attention_backend": ("model.decoder_attention_backend", str),
    "encoder_attention_backend": ("model.encoder_attention_backend", str),
    "checkpoint": ("model.checkpoint", str),
    "exec_horizon": ("session.exec_horizon", int),
    "num_steps": ("sampler.num_steps", int),
    "cfg_scale": ("sampler.cfg_scale", float),
    "cfg_video_scale": ("sampler.cfg_video_scale", float),
    "t_shift": ("sampler.t_shift", float),
    "control_mode": ("session.control_mode", str),
    "solver": ("sampler.solver", str),
    "context_mode": ("session.context_mode", str),
    "max_context_chunks": ("session.max_context_chunks", int),
    "carry_mode": ("session.carry_mode", str),
    "prompt": ("session.prompt", str),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="serve config (YAML)")
    ap.add_argument(
        "--set", action="append", default=[], help="override, e.g. sampler.num_steps=4"
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    for name, (key, kind) in FLAGS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=kind, help=f"sets {key}")
    args = ap.parse_args()
    # force: an imported package can install a root handler before this line.
    logging.basicConfig(level=logging.INFO, force=True)
    cfg = load_config(args.config, args.set)
    for name, (key, _kind) in FLAGS.items():
        value = getattr(args, name)
        if value is not None:
            section, field = key.split(".")
            cfg.setdefault(section, {})[field] = value
    server = build_server(cfg)
    serve(server.app, args.host, args.port)


if __name__ == "__main__":
    main()
