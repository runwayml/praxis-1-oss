# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified from RoboLab's `policies/cosmos3/run.py` (7d45d74) to run the praxis policy server.
"""Evaluate our policy server through RoboLab's official runner (`run_evaluation`).

Differences from `policies/cosmos3/run.py`, all about WHAT is registered and which
client is built; the episode loop is RoboLab's, untouched:

* tasks are registered with RoboLab's own AbsIK registration
  (`auto_register_droid_abs_ik_envs`: `DroidIKActionCfg`, absolute base_link pose
  IK), because the policy emits end-effector poses; the env names gain no postfix, so
  `--task GrabAFruitTask` resolves exactly as with the joint-position registration;
* the cameras are RoboLab's WRIST_LEFT preset (over_shoulder_left + wrist, 1280x720),
  the two views the policy uses;
* the client is `PraxisRemoteClient` (positronic remote protocol, positronic's client
  stack in a sidecar).

Run from the RoboLab checkout (cwd = its root, as RoboLab's docs run
`policies/<policy>/run.py`), with this directory on PYTHONPATH:

    python <this dir>/run_praxis.py --headless --task GrabAFruitTask ... \\
        --num-envs 1 --num-runs 5 --policy-url HOST:PORT/api/v1/session?... \\
        --output-folder-name <name>
"""

import argparse
import sys
import traceback

import cv2  # noqa: F401  # Must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "praxis"

parser = argparse.ArgumentParser(description="Evaluate the praxis policy backend.")
parser.add_argument(
    "--policy-url",
    required=True,
    help="positronic session URL of the policy server (host:port/api/v1/session?...).",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.num_envs != 1:
    parser.error("run one env per process (--num-envs 1); see the README")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from praxis_client import PraxisRemoteClient  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import (  # noqa: E402
    auto_register_droid_abs_ik_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa: E402

auto_register_droid_abs_ik_envs(task=args_cli.task, cameras=WRIST_LEFT)


def make_client(args: argparse.Namespace) -> PraxisRemoteClient:
    if args.num_envs != 1:
        raise ValueError("Run one env per process (--num-envs 1); see the README.")
    return PraxisRemoteClient(policy_url=args.policy_url)


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
