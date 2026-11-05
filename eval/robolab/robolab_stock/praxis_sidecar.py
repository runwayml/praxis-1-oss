"""Positronic's client half, driven one RoboLab control step at a time.

Runs INSIDE positronic's own venv (public positronic, the commit the README pins) with
`eval/robolab/positronic_cfg` on PYTHONPATH. Spawned by `praxis_client.PraxisRemoteClient`
(RoboLab's Isaac interpreter), it speaks length-prefixed frames of positronic's
env-server codec (`positronic.simulator.env_server.protocol`) on its stdin/stdout.

Per RoboLab step it does what one round of a positronic sim harness run does, on a
virtual clock that advances one control period per step:

* the raw RoboLab payload goes through positronic's `RobolabAdapter.observations` and
  the remote embodiment's serializers (`expand_suffixed`), plus `wall_time_ns`,
  `obs_time_ns`, the task and the `descriptor`. The end-effector pose stays in
  RoboLab's `eef_frame` (`EefFrameRobolabAdapter`);
* the policy is `ee_frame_policy_cfg.robolab_to_fr3_ee`: the client EE frame change in
  front of positronic's `RemotePolicy`, which builds the client stack the server
  declares (for example: image resize, temporal stack, chunked schedule) and
  JPEG-encodes the frames the server asks for. It runs on positronic's `Executor` in
  simulation without charged inference time: a call that is due runs while the clock
  stands still, and the policy reads its answer in the same step, as positronic's
  harness does;
* the commands the policy emits in a step update the held command, and the held
  command goes out as the wire action `{'command': {...}, 'grip': g}`, the action the
  env server would receive.

Virtual clock: a trial begins at step 0, the env publishes frame 0 at step 1, and the
first policy call runs at step 2 on frame 0. Step k is at `k * round(control_dt * 1e9)`
ns.

What a harness run sends and this does not: `robot_state.dq` is zeros (RoboLab's
observation has no joint velocities; the server never reads them),
`robot_state.status` is left out, and the trial bookkeeping keys (`eval.task`,
`eval.trial_index`, ...) are absent (RoboLab's runner does not hand them to a client).

usage: python praxis_sidecar.py --url HOST:PORT/api/v1/session?...  (stdin/stdout frames)
"""

import argparse
import logging
import os
import struct
import sys
import time
import traceback
from contextlib import nullcontext
from typing import Any

import ee_frame_policy_cfg as cfgmod
import ee_frame_wire as ee
import numpy as np
from positronic import geom, keys
from positronic.dataset.serializers import expand_suffixed
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy import keys as policy_keys
from positronic.policy.executor import Executor, WaitStatus
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.proxy import (
    RemoteEnvControlSystem,
    remote_embodiment,
)
from positronic.simulator.robolab.adapter import RobolabAdapter
from positronic.utils import flatten_dict, frozen_view

DESCRIPTOR = "remote.robolab.droid"
# positronic's stock RoboLab eval camera_dict.
CAMERA_DICT = {
    keys.EXTERIOR_IMAGE: "over_shoulder_left_camera",
    keys.WRIST_IMAGE: "wrist_cam",
}
# RoboLab's control period, computed like env.py's `sim.dt * decimation` for the
# DROID registrations (dt = 1 / (60 * 2), decimation = 8).
CONTROL_DT = (1 / (60 * 2)) * 8
TICK_NS = max(1, round(CONTROL_DT * 1e9))
# How long one wait for a policy answer blocks before it checks again.
_ANSWER_POLL_SEC = 1.0
_ROTATION_MATRIX = geom.Rotation.Representation.ROTATION_MATRIX

# Franka modified-DH (a, d, alpha); flange 0.107 m past joint 7 (numpy-only copy of
# the server's rig-frame FK).
_FRANKA_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
_FLANGE_D = 0.107
_FR3_TOOL_OFFSET_M = 0.1034


def franka_flange_xyz(q: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    for (a, d, alpha), theta in zip(_FRANKA_DH, np.asarray(q, np.float64), strict=True):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        m = m @ np.array(
            [
                [ct, -st, 0.0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return m[:3, 3] + _FLANGE_D * m[:3, 2]


def fk_error_mm(wire_pose7: np.ndarray, q: np.ndarray) -> float:
    """How far the pose the server will see (wire eef -> FR3 end_effector -> flange)
    lands from the flange FK of `q`; the server refuses > 5 mm."""
    fr3 = ee.remap_pose7(np.asarray(wire_pose7, np.float64), ee.ROBOLAB_TO_FR3_EE)
    rot = ee.quat_wxyz_to_matrix(fr3[3:])
    flange = fr3[:3] - _FR3_TOOL_OFFSET_M * rot[:, 2]
    return float(np.linalg.norm(flange - franka_flange_xyz(q)) * 1e3)


class EefFrameRobolabAdapter(RobolabAdapter):
    """`RobolabAdapter` that reports the end-effector pose in RoboLab's `eef_frame`,
    the frame the measured EE frame change (`ee_frame_wire.ROBOLAB_TO_FR3_EE`) starts
    from, rather than at positronic's gripper site."""

    def _measure(
        self, env_pose: geom.Transform3D, arm: str | None = None
    ) -> geom.Transform3D:
        self._measured[keys.arm_channel(keys.ROBOT_COMMAND, arm)] = env_pose
        return env_pose


def wire_command(command: Any) -> dict[str, Any]:
    """The held arm command as the env server's wire payload."""
    if command is None:
        return {"type": protocol.HOLD}
    if isinstance(command, roboarm_command.CartesianPosition):
        return {
            "type": protocol.CARTESIAN,
            "pose": command.pose.as_vector(_ROTATION_MATRIX),
        }
    raise ValueError(f"no wire encoding for robot_command {type(command).__name__}")


def _static_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_static_value(item) for item in value if item is not None]
    if isinstance(value, dict):
        return {str(k): _static_value(v) for k, v in value.items() if v is not None}
    return str(value)


def drop_none(meta: dict[str, Any]) -> dict[str, Any]:
    """``meta`` with None values dropped and anything else non-JSON stringified.

    The policy server publishes every server knob in its session metadata, unset ones
    as None (record_denoised, compile, ...), and an episode writer accepts only JSON
    numbers, strings and containers.
    """
    return {
        key: _static_value(value) for key, value in meta.items() if value is not None
    }


def session_meta(runtime: Executor) -> dict[str, Any]:
    """The policy meta the episode records: the client type and the server's metadata."""
    server = runtime.metadata_snapshot().get(policy_keys.SERVER, {})
    return drop_none(flatten_dict({"type": "remote", "server": server}))


class EnvSession:
    """One RoboLab env's trial: its own policy run, held command and virtual clock."""

    def __init__(self, policy: Any, embodiment: Any, instruction: str):
        self.policy = policy
        self.embodiment = embodiment
        self.adapter = EefFrameRobolabAdapter(CAMERA_DICT)
        self.task = instruction
        self.now_ns = 0
        self.runtime = Executor(
            self._clock, simulated=True, charge_inference_time=False
        )
        # Opens the server session.
        self.run = self.runtime.start(policy)
        self.meta = session_meta(self.runtime)
        self.command: Any = None
        self.grip = 0.0
        # Step 1 is the env publishing frame 0; step 2 calls the policy on it.
        self.tick = 1

    def _clock(self) -> int:
        return self.now_ns

    def _build_obs(self, latest: dict[str, Any]) -> dict[str, Any]:
        """The observation a harness builds from the values the env just emitted."""
        inputs: dict[str, Any] = {}
        for name, obs in self.embodiment.observations.items():
            value = latest[name]
            if obs.serializer is not None:
                value = obs.serializer(value)
            for full_name, v in expand_suffixed(name, value):
                if v is not None and full_name != keys.ROBOT_STATUS:
                    inputs[full_name] = v
        inputs["wall_time_ns"] = time.time_ns()
        inputs["obs_time_ns"] = self.now_ns
        inputs[keys.TASK] = self.task
        inputs[keys.DESCRIPTOR] = self.embodiment.descriptor
        return inputs

    def _hold(self, commands: dict[str, Any]) -> None:
        if keys.ROBOT_COMMAND in commands:
            self.command = commands[keys.ROBOT_COMMAND]
        if keys.TARGET_GRIP in commands:
            self.grip = float(commands[keys.TARGET_GRIP])

    def step(self, raw: dict[str, Any]) -> dict[str, Any]:
        self.tick += 1
        self.now_ns = self.tick * TICK_NS
        obs = frozen_view(self._build_obs(self.adapter.observations(raw)))
        calls = len(self.policy.remote.calls)
        started = time.monotonic()
        self.runtime.start_tick()
        self._hold(self.run.send(obs).commands)
        # A due call runs while the clock stands still; the policy reads its answer
        # in this same step.
        while True:
            waited = self.runtime.wait(_ANSWER_POLL_SEC)
            # `wait` can report CAN_ADVANCE for a call that finished after its own
            # check; such an answer is read in this step too.
            ready = waited.status is WaitStatus.ANSWERS_READY or (
                waited.status is WaitStatus.CAN_ADVANCE
                and self.runtime.take_completed()
            )
            if ready:
                self._hold(self.run.send(obs).commands)
            elif waited.status is WaitStatus.CAN_ADVANCE:
                break
        latency = time.monotonic() - started
        inferred = len(self.policy.remote.calls) > calls
        if inferred and self.policy.remote.calls[-1][0] == 0:
            # An empty chunk cancels the held pose: the arm holds where it is.
            self.command = None
        wire = wire_command(self.command)
        out: dict[str, Any] = {
            "command": wire,
            "grip": self.grip,
            "inferred": inferred,
            "tick": self.tick,
        }
        if out["inferred"]:
            out["n_actions"] = self.policy.remote.calls[-1][0]
            out["latency_s"] = latency
            pose7 = np.concatenate(
                [
                    np.asarray(raw["eef_pos"], np.float64),
                    np.asarray(raw["eef_quat"], np.float64),
                ]
            )
            out["fk_err_mm"] = fk_error_mm(pose7, raw["joint_pos"])
            if wire["type"] == protocol.CARTESIAN:
                first = np.asarray(wire["pose"], np.float64)[:3]
                out["first_cmd_dist_mm"] = float(
                    np.linalg.norm(first - pose7[:3]) * 1e3
                )
        return out

    def close(self) -> None:
        # Drain the calls before closing the run that owns the server session.
        self.runtime.close()
        self.run.close()


class Sidecar:
    """The policy object, shared by one `EnvSession` per RoboLab env id."""

    def __init__(self, url: str):
        self.policy = cfgmod.robolab_to_fr3_ee(url)
        # The remote embodiment a harness runs (its serializers, channel names and
        # descriptor); the proxy is never connected, it only owns the pimm ports.
        proxy = RemoteEnvControlSystem(
            RobolabAdapter(CAMERA_DICT), nullcontext(("unused", 0))
        )
        self.embodiment = remote_embodiment(proxy, CAMERA_DICT, descriptor=DESCRIPTOR)
        self.envs: dict[int, EnvSession] = {}

    def warmup(self) -> dict[str, Any]:
        # Open and close one session, as positronic's eval warms up its policy.
        runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
        run = runtime.start(self.policy)
        try:
            return session_meta(runtime)
        finally:
            runtime.close()
            run.close()

    def reset(self, env_id: int, instruction: str) -> dict[str, Any]:
        self.end(env_id)
        env = EnvSession(self.policy, self.embodiment, instruction)
        self.envs[env_id] = env
        return env.meta

    def step(self, env_id: int, raw: dict[str, Any]) -> dict[str, Any]:
        return self.envs[env_id].step(raw)

    def end(self, env_id: int | None) -> None:
        ids = list(self.envs) if env_id is None else [env_id]
        for eid in ids:
            env = self.envs.pop(eid, None)
            if env is not None:
                env.close()

    def close(self) -> None:
        self.end(None)


def _plain(value: Any) -> Any:
    """Handshake meta as plain data for the frame codec."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, np.ndarray | np.generic):
        return np.asarray(value).tolist()
    return str(value)


def _read_frame(stream) -> dict[str, Any] | None:
    header = stream.read(8)
    if len(header) < 8:
        return None
    (size,) = struct.unpack("<Q", header)
    payload = stream.read(size)
    return protocol.decode(payload)


def _write_frame(stream, obj: dict[str, Any]) -> None:
    payload = protocol.encode(obj)
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    # The frames own the real stdout; anything else printed goes to stderr.
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    # Positronic logs its connect retries (a cold server can take minutes) at INFO.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    inp = os.fdopen(os.dup(0), "rb")
    sidecar = Sidecar(args.url)
    try:
        while True:
            msg = _read_frame(inp)
            if msg is None:
                return 0
            try:
                cmd = msg["cmd"]
                if cmd == "warmup":
                    reply = {"meta": _plain(sidecar.warmup())}
                elif cmd == "reset":
                    meta = sidecar.reset(int(msg["env_id"]), str(msg["instruction"]))
                    reply = {"meta": _plain(meta)}
                elif cmd == "step":
                    reply = sidecar.step(int(msg["env_id"]), msg["raw"])
                elif cmd == "end":
                    env_id = msg.get("env_id")
                    sidecar.end(None if env_id is None else int(env_id))
                    reply = {}
                elif cmd == "close":
                    _write_frame(out, {})
                    return 0
                else:
                    raise ValueError(f"unknown sidecar command {cmd!r}")
            except Exception as error:
                reply = {"error": f"{error!r}\n{traceback.format_exc()}"}
            _write_frame(out, reply)
    finally:
        sidecar.close()


if __name__ == "__main__":
    raise SystemExit(main())
