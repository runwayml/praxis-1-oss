"""EE tool-frame adapter config, loaded INSIDE positronic's venv.

Puts a constant end-effector frame change in front of positronic's ``RemotePolicy``, so
a server that applies no EE-convention remap sees the sim's EE pose in the frame it
expects, and its Cartesian commands go back to the sim in the sim's wire frame
(``ee_frame_wire`` has the math and the measured frames):

* obs: ``robot_state.ee_pose`` -> ``T @ M`` (per control tick, before the server-declared
  temporal stack stacks it: the frame change sits strictly outside the client stack);
* commands: every ``robot_command`` ``CartesianPosition`` pose -> ``T @ M^-1``;
  ``CartesianDelta`` is refused, pose-less commands pass through.

``robolab_to_fr3_ee``: RoboLab's ``eef_frame`` (flange @ [Rz(+90 deg) | +18.17 mm])
-> the FR3 end_effector frame the policy
server expects from ``descriptor="remote.robolab.droid"`` (flange @ [Rz(-45 deg) |
+103.4 mm]), M = [Rz(-135 deg) | (0, 0, +85.23 mm)]. That server converts to the flange
and the checkpoint's model turn itself (and FK-checks every request), so this is the
only client-side change in front of it.

The frame change runs on the client, not as a ``change_ee_frame`` entry that the server
declares, so a server that declares no frame change gets the frame it expects.

Used by ``robolab_stock/praxis_sidecar.py`` with this directory on PYTHONPATH. This
module is imported inside the positronic uv environment, never by RoboLab's interpreter.
"""

import json
import os
import time
import urllib.parse
from typing import Any

import ee_frame_wire as ee
import numpy as np
from positronic import geom
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.remote import RemotePolicy
from positronic.policy.sequential import Sequential
from positronic_wire.wire import SESSION_PATH, HostPortAddress

# Fields the public client schedule adds to every request. A protocol v1 server reads
# neither, so they stay off the wire.
_SCHEDULE_FIELDS = (policy_keys.CALL_ID, policy_keys.CALL_TIMINGS)


def _command_to_wire(command: Any, offset_inv: np.ndarray) -> Any:
    """A positronic command with its pose mapped back into the sim's wire frame."""
    if isinstance(command, roboarm_command.CartesianPosition):
        pose7 = np.concatenate(
            [
                np.asarray(command.pose.translation, dtype=np.float64).reshape(3),
                np.asarray(command.pose.rotation.as_quat, dtype=np.float64).reshape(4),
            ]
        )
        out = ee.remap_pose7(pose7, offset_inv)
        return roboarm_command.CartesianPosition(
            pose=geom.Transform3D(
                translation=out[:3], rotation=geom.Rotation.from_quat(out[3:])
            )
        )
    if isinstance(command, roboarm_command.CartesianDelta):
        raise ValueError("ee frame adapter: cartesian_delta commands are unsupported")
    return command


def wire_address(url: str) -> tuple[str, HostPortAddress]:
    """The positronic wire and session address for a policy URL.

    Accepted forms: ``host``, ``host:port`` and ``scheme://host[:port][/api/v1/session[/<model_id>]]``,
    each with an optional ``?query``. ``https``/``wss`` use TLS, the port defaults to 443 for
    TLS and 80 otherwise, and the query reaches the server as written.
    """
    split = urllib.parse.urlsplit(url if "://" in url else f"//{url}")
    if split.scheme not in ("", "http", "ws", "https", "wss"):
        raise ValueError(f"Unsupported scheme {split.scheme!r} in {url!r}")
    if not split.hostname:
        raise ValueError(f"No host in {url!r}")
    secure = split.scheme in ("https", "wss")
    path = split.path
    if path.rstrip("/") in ("", SESSION_PATH):
        path = SESSION_PATH
    elif not path.startswith(f"{SESSION_PATH}/"):
        raise ValueError(
            f"Unexpected path {path!r} in {url!r}; expected {SESSION_PATH}[/<model_id>]"
        )
    port = split.port if split.port is not None else (443 if secure else 80)
    name = "websocket_tls" if secure else "websocket"
    return name, HostPortAddress(split.hostname, port, path, split.query)


class _CallLoggingSession:
    """An inference session that drops ``_SCHEDULE_FIELDS`` from each request and logs
    each call (rows answered, round-trip seconds) to ``calls``."""

    def __init__(self, inner: Any, calls: list[tuple[int, float]]):
        self._inner = inner
        self._calls = calls

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def infer(self, obs: dict[str, Any]) -> Any:
        sent = {k: v for k, v in obs.items() if k not in _SCHEDULE_FIELDS}
        started = time.monotonic()
        result = self._inner.infer(sent)
        if result is not None:  # a None answer keeps the current trajectory
            rows = len(result) if isinstance(result, list) else 1
            self._calls.append((rows, time.monotonic() - started))
        return result


class LoggedRemotePolicy(RemotePolicy):
    """``RemotePolicy`` on a URL, whose sessions go through ``_CallLoggingSession``.

    ``calls`` holds one ``(rows, seconds)`` entry per answered policy call.
    """

    def __init__(self, url: str, *, headers: dict[str, str] | None = None):
        name, address = wire_address(url)
        super().__init__(name, address, headers=headers)
        self.calls: list[tuple[int, float]] = []
        new_session = self._client.new_session
        self._client.new_session = lambda: _CallLoggingSession(
            new_session(), self.calls
        )


class EeFrameChange(Policy):
    """Re-express the EE pose in the policy's frame, and its commands in the sim's."""

    def __init__(self, offset: np.ndarray):
        self._offset = np.asarray(offset, dtype=np.float64).reshape(4, 4)
        self._offset_inv = ee.inverse(self._offset)

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            try:
                step = inner.send(ee.obs_to_model(obs, self._offset))
            except StopIteration as end:
                return end.value
            commands = dict(step.commands)
            if ee.ROBOT_COMMAND in commands:
                commands[ee.ROBOT_COMMAND] = _command_to_wire(
                    commands[ee.ROBOT_COMMAND], self._offset_inv
                )
            obs = yield Step(commands, step.resume_at_ns, outputs=step.outputs)


def policy_headers() -> dict[str, str] | None:
    """``PRAXIS_POLICY_HEADERS``: a JSON object of HTTP headers (proxy auth), or unset.

    The headers travel by environment variable rather than argv, which would leak the
    secret into process lists.
    """
    value = os.environ.get("PRAXIS_POLICY_HEADERS")
    if not value:
        return None
    headers = json.loads(value)
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ValueError(
            "PRAXIS_POLICY_HEADERS must be a JSON object of string headers"
        )
    return headers


class EeFramePolicy(Sequential):
    """``EeFrameChange(offset)`` in front of ``LoggedRemotePolicy(url)``; ``remote`` is the
    remote policy, whose ``calls`` log the answered policy calls."""

    def __init__(self, offset: np.ndarray, url: str):
        self.remote = LoggedRemotePolicy(url, headers=policy_headers())
        super().__init__(EeFrameChange(offset), self.remote)


def robolab_to_fr3_ee(url: str) -> EeFramePolicy:
    """The wire eef -> the FR3 end_effector frame the policy server expects from
    descriptor "remote.robolab.droid"; the server does the rest."""
    return EeFramePolicy(ee.ROBOLAB_TO_FR3_EE, url)
