"""Fake-data example client for a praxis RoboLab endpoint.

Drives `PraxisRemoteClient` exactly as RoboLab's `robolab/eval/episode.py` does (one
`infer(obs, instruction, env_id=0)` per 15 Hz control step, then `reset()`), on synthetic
RoboLab-shaped observations: RoboLab's DROID reset joints and eef pose, uniform-noise
1280x720 exterior + wrist frames. No Isaac, no RoboLab env: only RoboLab's
`InferenceClient` base class and positronic (the sidecar) are needed.

    # endpoint + proxy auth (Modal: https://<workspace>--<app>-<class>-web.modal.run);
    # the headers ride the environment only, never argv or a file in a log.
    export PRAXIS_POLICY_HEADERS='{"Modal-Key": "<token-id>", "Modal-Secret": "<token-secret>"}'
    # sidecar_env.json: the sidecar's env, with positronic_cfg/ on PYTHONPATH (README, Setup)
    PRAXIS_SIDECAR_ENV_FILE=<abs path>/sidecar_env.json \\
    PRAXIS_SIDECAR_PYTHON=<positronic venv python> PRAXIS_POSITRONIC_DIR=<positronic tree> \\
    PYTHONPATH=<this dir>:<robolab checkout> \\
      python example_client.py --policy-url https://...modal.run --steps 48

Logs each policy call (tick, chunk length, latency) to --calls-log and prints the 8-dim
DroidIK actions' shape/range. With 24-action chunks at 15 Hz, 48 steps make three
calls (ticks 2, 23 and 47).
"""

import argparse
import time
from pathlib import Path

import numpy as np

# RoboLab's DROID reset (robolab/robots/droid.py init_state) and its eef_frame / base_link
# poses at that reset (recorded from a live env; eef = base_link (x) EEF_OFFSET_ROT).
RESET_Q = np.array(
    [0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0], np.float32
)
RESET_EEF_POS = np.array([0.3597, 0.0, 0.4718], np.float32)
RESET_EEF_QUAT = np.array([0.0, -0.70710677, 0.70710677, 0.0], np.float32)
# The client runs the chunked schedule: RoboLab pauses the sim during each call,
# so the robot never runs a chunk while the next one is computed.
QUERY = (
    "schedule=chunked&declare_stop_on_fault=false&record_denoised=false"
    "&max_context_chunks=1&t_shift=12.5"
)


def fake_obs(rng: np.random.Generator) -> dict:
    """One RoboLab-shaped batched observation (env axis first, num_envs = 1)."""
    import praxis_stock_wire as wire

    base_quat = wire.eef_to_base_quat(RESET_EEF_QUAT.astype(np.float64)).astype(
        np.float32
    )
    frame = lambda: rng.integers(0, 256, (1, 720, 1280, 3), dtype=np.uint8)
    return {
        "proprio_obs": {
            "arm_joint_pos": RESET_Q[None],
            "gripper_pos": np.zeros((1, 1), np.float32),
            "ee_pos": RESET_EEF_POS[None],
            "ee_quat": base_quat[None],
            "eef_pos": RESET_EEF_POS[None],
            "eef_quat": RESET_EEF_QUAT[None],
        },
        "image_obs": {"over_shoulder_left_camera": frame(), "wrist_cam": frame()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fake-data example client for a praxis RoboLab endpoint."
    )
    parser.add_argument(
        "--policy-url",
        required=True,
        help="wss:// or https:// endpoint (ws://, http:// or host:port without TLS)",
    )
    parser.add_argument("--query", default=QUERY)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--instruction", default="Pick up a fruit")
    parser.add_argument("--calls-log", default="/tmp/praxis_example_calls.jsonl")
    args = parser.parse_args()

    import praxis_client

    url = f"{args.policy_url.rstrip('/')}/api/v1/session?{args.query}"
    rng = np.random.default_rng(0)
    client = praxis_client.PraxisRemoteClient(
        policy_url=url, calls_log=Path(args.calls_log)
    )
    actions = []
    started = time.monotonic()
    for _ in range(args.steps):
        actions.append(
            client.infer(fake_obs(rng), args.instruction, env_id=0)["action"]
        )
    client.reset()
    arr = np.stack(actions)
    print(
        f"{args.steps} steps in {time.monotonic() - started:.1f} s; actions {arr.shape} {arr.dtype}"
    )
    print(f"  position range {arr[:, :3].min(0)} .. {arr[:, :3].max(0)}")
    print(
        f"  |quat| {np.linalg.norm(arr[:, 3:7], axis=1).round(4).min()}..{np.linalg.norm(arr[:, 3:7], axis=1).round(4).max()}"
    )
    print(f"  grip values {np.unique(arr[:, 7].round(3))[:6]}")
    print(f"  calls log: {args.calls_log}")


if __name__ == "__main__":
    main()
