"""Run the praxis policy in closed loop on a MuJoCo task and watch it in a browser.

Usage:
    python scripts/run_sim.py --config configs/sim_droid.yaml [--set key=value ...]

The YAML file holds all the settings. Examples: the checkpoint, the sampler,
the task, the prompt, the seeds, the episode length, the cameras, the outputs
and the viewer. ``--set`` changes one value, for example ``--set task.prompt="Lift
the red cube."`` or ``--set episodes.seeds=[0,1,2]``.

The script prints the URL of the live view and the output paths. The live
view shows the rendered views (and the predicted video of the model) while
the loop runs. After the run, the same page plays the MP4 of the last
episode. The viewer stays open for ``viewer.keep_open_s`` seconds, or until
Ctrl-C.

The script needs the ``sim`` extra: ``pip install -e ".[sim]"``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# praxis.sim sets the OpenGL backend of MuJoCo. It must load before mujoco.
import praxis.sim  # noqa: F401
from praxis.sim.config import load_config
from praxis.sim.loop import run
from praxis.sim.viewer import Viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="override, e.g. task.prompt='Lift the red cube.'",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config, args.set)

    viewer: Viewer | None = None
    if cfg.viewer.enabled:
        viewer = Viewer(
            host=cfg.viewer.host,
            port=cfg.viewer.port,
            jpeg_quality=cfg.viewer.jpeg_quality,
            max_fps=cfg.viewer.max_fps,
            title=f"praxis sim: {cfg.task.name}",
        )
        url = viewer.start()
        print(f"live view: {url}")
        if cfg.viewer.host in ("127.0.0.1", "localhost"):
            print(
                "on a remote machine, forward the port first: "
                f"ssh -L {viewer.port}:127.0.0.1:{viewer.port} <user>@<host>"
            )
    try:
        result = run(cfg, viewer=viewer)
        print(
            f"success {sum(e.success for e in result.episodes)}/{len(result.episodes)} "
            f"episodes"
        )
        print(f"outputs: {result.out_dir.resolve()}")
        for ep in result.episodes:
            if ep.video:
                print(f"  video seed {ep.seed}: {Path(ep.video).resolve()}")
        print(f"  summary: {(result.out_dir / 'summary.json').resolve()}")
        if viewer is not None:
            keep = cfg.viewer.keep_open_s
            if keep is None:
                print(f"the viewer stays open at {viewer.url}. Press Ctrl-C to stop.")
                while True:
                    time.sleep(1.0)
            elif keep > 0:
                print(
                    f"the viewer stays open at {viewer.url} for {keep:.0f} s. "
                    "Press Ctrl-C to stop."
                )
                time.sleep(keep)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if viewer is not None:
            viewer.stop()


if __name__ == "__main__":
    main()
