"""A praxis `InferenceClient` for RoboLab's OFFICIAL runner (`robolab.eval.runner`).

Runs in RoboLab's own Isaac interpreter; imports no positronic. The episode loop
(`robolab/eval/episode.py`, unmodified) calls `infer(obs, instruction, env_id=...)`
once per control step with the sim paused, and steps the env with the returned
8-dim `DroidIKActionCfg` action. This client overrides `infer` entirely (the
`InferenceClient` contract's level 3: "your flow isn't query-then-step-chunk"):

1. `_extract_observation` cuts RoboLab's batched obs into exactly the raw payload
   Positronic's RoboLab env server emits per tick (`positronic/simulator/robolab/env.py`
   `_observe`) — `joint_pos`, `eef_pos`/`eef_quat` (the `eef_frame` obs
   terms), `grip`, the 1280x720 `over_shoulder_left_camera` + `wrist_cam` renders —
   except `joint_vel`, which RoboLab's obs does not carry (zeros; the server never
   reads it).
2. A sidecar process in positronic's own venv (`praxis_sidecar.py`) runs positronic's
   client half on it: the RoboLab adapter, the observation build and the policy
   (`robolab_to_fr3_ee`: the client EE frame change in front of positronic's
   `RemotePolicy` and the server-declared client stack). It returns the held wire
   action `{'command': ..., 'grip': g}` the env server would receive this tick.
3. `praxis_stock_wire.wire_to_absik_action` turns it into the DroidIK action
   (`[pos, q_eef (x) EEF_OFFSET_ROT^-1, grip]`, positronic env.py's own conversion).

One sidecar per process, shared by the per-task clients `run_evaluation` builds;
one positronic session per RoboLab env id, opened on its first `infer` and closed on
`reset` (the episode loop resets the client after every run).

The server's pose-check refusal (a `Server error` reply saying the converted ee_pose is
too far from the flange forward kinematics of robot_state.q, as after a PhysX blow-up
throws the articulation) faults that env: the refusal and the env's last steps are logged, its session is
closed, and from then on it is held at its measured base_link pose with its last
grip until RoboLab's runner resets it. The episode then ends by RoboLab's own success
check or timeout and is scored like any other (a physics blow-up counts as a
failure, never discarded or re-run). Any other error (another server error, the
sidecar, the transport) propagates and stops the run.
"""

import atexit
import contextlib
import importlib.util
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import praxis_stock_wire as wire
from robolab.eval.base_client import InferenceClient

_HERE = Path(__file__).resolve().parent
SIDECAR_SCRIPT = _HERE / "praxis_sidecar.py"

# The live frame sanity bound at each episode's first step: base_link (`ee_*` obs)
# composed with EEF_OFFSET_ROT must be the `eef_*` obs (the FrameTransformer offset).
_FRAME_TOL_DEG = 0.5
_FRAME_TOL_MM = 0.5
# A server error crosses the sidecar as `RuntimeError: Server error: ...` (positronic's
# `InferenceSession.infer`). Only the server's pose-input checks fault the env: a pose
# far from the joints' forward kinematics, or a non-finite pose or gripper value (both
# what a physics blow-up produces). Any other server error stops the run.
_SERVE_REFUSAL = "Server error:"
_POSE_REFUSALS = (
    "converted ee_pose",
    "ee_pose has non-finite values",
    "grip has non-finite values",
    "ee_pose/grip history has non-finite values",
)
_FAULT_HISTORY = 30


def _load_protocol(positronic_dir: Path) -> Any:
    """positronic's env-server wire codec, loaded by path (positronic-free module)."""
    path = positronic_dir / "positronic" / "simulator" / "env_server" / "protocol.py"
    spec = importlib.util.spec_from_file_location("_praxis_wire_protocol", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no positronic wire protocol at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _np(value: Any) -> np.ndarray:
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


class ServeRefusal(RuntimeError):
    """The policy server refused this env's request (its explicit `Server error`)."""


class Sidecar:
    """The positronic sidecar subprocess: length-prefixed protocol frames on its pipes."""

    def __init__(
        self,
        policy_url: str,
        *,
        python: str,
        positronic_dir: Path,
        env: dict[str, str],
        log_path: Path,
        script: Path = SIDECAR_SCRIPT,
    ):
        self._protocol = _load_protocol(positronic_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("ab")
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [python, str(script), "--url", policy_url],
            cwd=str(positronic_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
        )

    def request(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            payload = self._protocol.encode(msg)
            try:
                self.proc.stdin.write(struct.pack("<Q", len(payload)) + payload)
                self.proc.stdin.flush()
                header = self.proc.stdout.read(8)
            except BrokenPipeError:
                header = b""
            if len(header) < 8:
                raise RuntimeError(
                    f"praxis sidecar exited (code {self.proc.poll()}); see {self._log.name}"
                )
            (size,) = struct.unpack("<Q", header)
            reply = self._protocol.decode(self.proc.stdout.read(size))
        if "error" in reply:
            error = str(reply["error"])
            first = error.splitlines()[0] if error else error
            if _SERVE_REFUSAL in first and any(r in first for r in _POSE_REFUSALS):
                raise ServeRefusal(first)
            raise RuntimeError(f"praxis sidecar: {error}")
        return reply

    def close(self) -> None:
        if self.proc.poll() is None:
            with contextlib.suppress(Exception):  # best effort at exit
                self.request({"cmd": "close"})
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


_SHARED: dict[str, Sidecar] = {}


def shared_sidecar(policy_url: str) -> Sidecar:
    """The process-wide sidecar (configured from the environment), started once."""
    if policy_url not in _SHARED:
        # The sidecar inherits this process's environment (proxy and TLS settings
        # included) minus its Python paths; the optional env file overrides entries.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "VIRTUAL_ENV")
        }
        env_file = os.environ.get("PRAXIS_SIDECAR_ENV_FILE")
        if env_file:
            env.update(json.loads(Path(env_file).read_text()))
        # The sidecar imports this client's positronic_cfg modules.
        env.setdefault("PYTHONPATH", str(_HERE.parent / "positronic_cfg"))
        # Proxy-auth headers (e.g. Modal-Key / Modal-Secret) ride the process env only:
        # `ee_frame_policy_cfg` reads PRAXIS_POLICY_HEADERS itself.
        if os.environ.get("PRAXIS_POLICY_HEADERS"):
            env["PRAXIS_POLICY_HEADERS"] = os.environ["PRAXIS_POLICY_HEADERS"]
        missing = [
            name
            for name in ("PRAXIS_SIDECAR_PYTHON", "PRAXIS_POSITRONIC_DIR")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"set {', '.join(missing)} (see eval/robolab/README.md, "
                "Environment variables)"
            )
        sidecar = Sidecar(
            policy_url,
            python=os.environ["PRAXIS_SIDECAR_PYTHON"],
            positronic_dir=Path(os.environ["PRAXIS_POSITRONIC_DIR"]),
            env=env,
            log_path=Path(
                os.environ.get(
                    "PRAXIS_SIDECAR_LOG", f"/tmp/praxis_sidecar.{os.getpid()}.log"
                )
            ),
        )
        meta = sidecar.request({"cmd": "warmup"})["meta"]
        print(
            f"[PraxisRemoteClient] sidecar up; policy meta keys: {len(meta)}",
            flush=True,
        )
        atexit.register(sidecar.close)
        _SHARED[policy_url] = sidecar
    return _SHARED[policy_url]


class PraxisRemoteClient(InferenceClient):
    """Our policy server, behind RoboLab's official episode loop (see the module doc)."""

    def __init__(self, policy_url: str, calls_log: Path | None = None):
        super().__init__()
        self._sidecar = shared_sidecar(policy_url)
        self._active: dict[int, str] = {}
        self._calls_log = calls_log or Path(
            os.environ.get(
                "PRAXIS_CALLS_LOG", f"/tmp/praxis_client_calls.{os.getpid()}.jsonl"
            )
        )
        self._handshake_path = self._calls_log.with_name(
            f"{self._calls_log.stem}.handshake.json"
        )
        self._handshake_written = False
        self._calls_log.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        # env id -> the action it is held at after a server refusal (until reset)
        self._faulted: dict[int, np.ndarray] = {}
        # env id -> its recent (step, action, base pose, joints) rows, for the fault log
        self._history: dict[int, list[dict]] = {}
        self._step: dict[int, int] = {}

    # ---- the InferenceClient hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        """One env's rows as the raw payload positronic's RoboLab env server emits."""
        proprio = raw_obs["proprio_obs"]
        images = raw_obs["image_obs"]
        return {
            "joint_pos": _np(proprio["arm_joint_pos"][env_id]),
            # RoboLab's proprio group has no joint velocities (positronic's env.py
            # reads them off the articulation); the policy server never reads them.
            "joint_vel": np.zeros(7, dtype=np.float32),
            "eef_pos": _np(proprio["eef_pos"][env_id]),
            "eef_quat": _np(proprio["eef_quat"][env_id]),
            "grip": float(proprio["gripper_pos"][env_id, 0]),
            "over_shoulder_left_camera": _np(
                images["over_shoulder_left_camera"][env_id]
            ),
            "wrist_cam": _np(images["wrist_cam"][env_id]),
            # The env server's privileged ground truth (never fed to the policy).
            "subtask": np.zeros(4, dtype=np.float32),
            # base_link (the DroidIK body): what a `hold` re-commands, and the frame check.
            "_base_pose7": np.concatenate(
                [_np(proprio["ee_pos"][env_id]), _np(proprio["ee_quat"][env_id])]
            ).astype(np.float64),
            "_env_id": env_id,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        raw = {k: v for k, v in extracted_obs.items() if not k.startswith("_")}
        return {"cmd": "step", "env_id": int(extracted_obs["_env_id"]), "raw": raw}

    def _query_server(self, request: dict) -> dict:
        return self._sidecar.request(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        raise NotImplementedError(
            "PraxisRemoteClient.infer converts the wire action itself"
        )

    # ---- the per-step flow -------------------------------------------------------

    def _begin(self, env_id: int, instruction: str, extracted: dict) -> None:
        meta = self._sidecar.request(
            {"cmd": "reset", "env_id": env_id, "instruction": instruction}
        )["meta"]
        self._active[env_id] = instruction
        if not self._handshake_written:
            self._handshake_path.write_text(json.dumps(meta, indent=1, default=str))
            self._handshake_written = True
        base = extracted["_base_pose7"]
        eef_from_base = wire.base_to_eef_quat(base[3:])
        frame_deg = quat_angle_deg(
            eef_from_base, np.asarray(extracted["eef_quat"], np.float64)
        )
        frame_mm = float(np.linalg.norm(base[:3] - extracted["eef_pos"]) * 1e3)
        self._write_call(
            {
                "event": "begin",
                "env_id": env_id,
                "instruction": instruction,
                "episode": self._episode,
                "frame_check_deg": frame_deg,
                "frame_check_mm": frame_mm,
            }
        )
        if frame_deg > _FRAME_TOL_DEG or frame_mm > _FRAME_TOL_MM:
            raise RuntimeError(
                f"eef_frame != base_link (x) EEF_OFFSET_ROT on the live obs: "
                f"{frame_deg:.3f} deg / {frame_mm:.3f} mm"
            )

    def _write_call(self, record: dict) -> None:
        with self._calls_log.open("a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")

    def _fault(self, env_id: int, extracted: dict, error: ServeRefusal) -> np.ndarray:
        last = self._history.get(env_id, [])
        grip = float(last[-1]["action"][7]) if last else 0.0
        pose = np.asarray(extracted["_base_pose7"], np.float64)
        if not np.all(np.isfinite(pose)):
            # A physics blow-up can make the measured pose non-finite, also on the
            # steps before the refused call: hold the last finite one instead.
            for row in reversed(last):
                if np.all(np.isfinite(row["base_pose7"])):
                    pose = np.asarray(row["base_pose7"], np.float64)
                    break
            else:
                raise RuntimeError(
                    f"env {env_id}: the server refused the pose and no finite pose is "
                    f"in the last {len(last)} steps to hold: {error}"
                ) from error
        hold = np.concatenate([pose, [grip]])
        self._faulted[env_id] = hold
        self._write_call(
            {
                "event": "fault",
                "env_id": env_id,
                "episode": self._episode,
                "step": self._step.get(env_id),
                "error": str(error)[:500],
                "joint_pos": np.asarray(extracted["joint_pos"]).tolist(),
                "eef_pos": np.asarray(extracted["eef_pos"]).tolist(),
                "base_pose7": np.asarray(extracted["_base_pose7"]).tolist(),
                "history": last[-_FAULT_HISTORY:],
            }
        )
        print(
            f"[PraxisRemoteClient] env {env_id} FAULTED at step {self._step.get(env_id)}: "
            f"{error}; holding until RoboLab resets it",
            flush=True,
        )
        # The server's session is spent: close it (a new one opens after the reset).
        with contextlib.suppress(Exception):
            self._sidecar.request({"cmd": "end", "env_id": env_id})
        return hold

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)
        self._step[env_id] = self._step.get(env_id, -1) + 1
        if env_id in self._faulted:
            return {"action": self._faulted[env_id].astype(np.float32), "viz": None}
        if env_id not in self._active:
            self._begin(env_id, instruction, extracted)
        started = time.monotonic()
        try:
            reply = self._query_server(self._pack_request(extracted, instruction))
        except ServeRefusal as error:
            if env_id not in self._history:
                # A pose refusal of the episode's first call comes at the reset pose,
                # so it is a misconfiguration (wrong frame or descriptor), not physics.
                raise RuntimeError(
                    f"the server refused the first call of the episode: {error}"
                ) from error
            hold = self._fault(env_id, extracted, error)
            return {"action": hold.astype(np.float32), "viz": None}
        action = wire.wire_to_absik_action(
            reply["command"], reply["grip"], extracted["_base_pose7"]
        )
        rows = self._history.setdefault(env_id, [])
        rows.append(
            {
                "step": self._step[env_id],
                "action": action.tolist(),
                "joint_pos": np.asarray(extracted["joint_pos"]).tolist(),
                "base_pose7": np.asarray(extracted["_base_pose7"]).tolist(),
            }
        )
        del rows[:-_FAULT_HISTORY]
        if reply.get("inferred"):
            self._write_call(
                {
                    "event": "infer",
                    "env_id": env_id,
                    "instruction": instruction,
                    "episode": self._episode,
                    "tick": reply.get("tick"),
                    "n_actions": reply.get("n_actions"),
                    "latency_s": reply.get("latency_s"),
                    "client_s": time.monotonic() - started,
                    "fk_err_mm": reply.get("fk_err_mm"),
                    "first_cmd_dist_mm": reply.get("first_cmd_dist_mm"),
                }
            )
        return {"action": action.astype(np.float32), "viz": None}

    def reset(self, *, env_id: int | None = None) -> None:
        ids = list(self._active) if env_id is None else [env_id]
        try:
            for eid in ids:
                if eid in self._active and eid not in self._faulted:
                    self._sidecar.request({"cmd": "end", "env_id": eid})
        finally:
            # RoboLab's episode loop logs and swallows a failed reset, so the per-env
            # state is cleared even when a session close fails.
            for eid in ids:
                self._active.pop(eid, None)
            for store in (self._faulted, self._history, self._step):
                if env_id is None:
                    store.clear()
                else:
                    store.pop(env_id, None)
            self._episode += 1
            super().reset(env_id=env_id)
