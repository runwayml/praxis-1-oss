"""The RoboLab client holds the arm on the server's pose refusals and on no other error."""

import ast
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from praxis import single_arm

STOCK = Path(__file__).resolve().parents[1] / "eval/robolab/robolab_stock"


def _client_constant(name: str) -> object:
    """A constant of the client module, read without importing RoboLab."""
    for node in ast.parse((STOCK / "praxis_client.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in praxis_client.py")


def test_pose_check_refusal_text_matches_the_client() -> None:
    q = np.zeros(7)
    far = single_arm.franka_flange_pose(q)[:3, 3] + np.array([0.1, 0.0, 0.0])
    pose8 = np.concatenate([far, [1.0, 0.0, 0.0, 0.0], [0.0]])
    with pytest.raises(ValueError) as refusal:
        single_arm.check_flange({"robot_state.q": q}, pose8)
    assert any(r in str(refusal.value) for r in _client_constant("_POSE_REFUSALS"))


@pytest.fixture
def client_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """praxis_client with a stand-in for RoboLab's InferenceClient base class."""

    class InferenceClient:
        def __init__(self) -> None:
            pass

        def reset(self, env_id: int | None = None) -> None:
            pass

    base = types.ModuleType("robolab.eval.base_client")
    base.InferenceClient = InferenceClient
    for name in ("robolab", "robolab.eval"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "robolab.eval.base_client", base)
    monkeypatch.syspath_prepend(str(STOCK))
    monkeypatch.delitem(sys.modules, "praxis_client", raising=False)
    import praxis_client

    return praxis_client


def _sidecar_replying(module: types.ModuleType, error: str) -> object:
    """A Sidecar whose pipe answers every request with the server error ``error``."""

    class Protocol:
        @staticmethod
        def encode(msg: dict) -> bytes:
            return b""

        @staticmethod
        def decode(payload: bytes) -> dict:
            return {"error": f"RuntimeError('Server error: {error}')\ntraceback"}

    class Pipe:
        def write(self, data: bytes) -> None:
            pass

        def flush(self) -> None:
            pass

        def read(self, n: int) -> bytes:
            return b"\x00" * n

    sidecar = module.Sidecar.__new__(module.Sidecar)
    sidecar._protocol = Protocol
    sidecar._lock = threading.Lock()
    sidecar.proc = types.SimpleNamespace(stdin=Pipe(), stdout=Pipe(), poll=lambda: None)
    sidecar._log = types.SimpleNamespace(name="sidecar.log")
    return sidecar


# Pose refusals of a deployed policy server, as they follow the client's "Server error: " prefix.
POSE_REFUSALS = [
    # Observed verbatim: two refusals of a RoboLab run, after the simulated
    # arm blew up.
    "client 'remote.robolab.droid': converted ee_pose is 42.3 mm off the flange FK of robot_state.q",
    "client 'remote.robolab.droid': converted ee_pose is 719.2 mm off the flange FK of robot_state.q",
    # From the server source only (never observed): the non-finite checks, which run
    # before the forward-kinematics check; two of them carry no descriptor prefix.
    "client 'remote.robolab.droid': ee_pose has non-finite values",
    "grip has non-finite values",
    "ee_pose/grip history has non-finite values",
]


@pytest.mark.parametrize("error", POSE_REFUSALS)
def test_pose_refusals_fault_the_env(
    client_module: types.ModuleType, error: str
) -> None:
    with pytest.raises(client_module.ServeRefusal):
        _sidecar_replying(client_module, error).request({"cmd": "step"})


@pytest.mark.parametrize(
    "error",
    [
        "CUDA out of memory",
        "max_chunks_per_episode reached",
        # Output-side checks are not pose-input refusals.
        "action chunk has non-finite values",
        "sampler produced non-finite actions",
    ],
)
def test_other_server_errors_stop_the_run(
    client_module: types.ModuleType, error: str
) -> None:
    with pytest.raises(RuntimeError) as raised:
        _sidecar_replying(client_module, error).request({"cmd": "step"})
    assert not isinstance(raised.value, client_module.ServeRefusal)


def _client(module: types.ModuleType, sidecar: object) -> object:
    client = module.PraxisRemoteClient.__new__(module.PraxisRemoteClient)
    module.InferenceClient.__init__(client)
    client._sidecar = sidecar
    client._active, client._faulted, client._history, client._step = {}, {}, {}, {}
    client._episode = 0
    client._calls_log = Path("/dev/null")
    return client


def _obs() -> dict:
    eef_quat = np.array([0.0, -0.70710677, 0.70710677, 0.0], np.float32)
    proprio = {
        "arm_joint_pos": np.zeros((1, 7), np.float32),
        "gripper_pos": np.zeros((1, 1), np.float32),
        "ee_pos": np.array([[0.36, 0.0, 0.47]], np.float32),
        "ee_quat": np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        "eef_pos": np.array([[0.36, 0.0, 0.47]], np.float32),
        "eef_quat": eef_quat[None],
    }
    frame = np.zeros((1, 4, 4, 3), np.uint8)
    return {
        "proprio_obs": proprio,
        "image_obs": {"over_shoulder_left_camera": frame, "wrist_cam": frame},
    }


def test_refusal_on_the_first_call_stops_the_run(
    client_module: types.ModuleType,
) -> None:
    client = _client(client_module, None)
    client._active[0] = "task"
    client._query_server = lambda request: (_ for _ in ()).throw(
        client_module.ServeRefusal("Server error: ee_pose has non-finite values")
    )
    with pytest.raises(RuntimeError, match="first call"):
        client.infer(_obs(), "task", env_id=0)


def test_refusal_after_the_first_call_holds_the_last_finite_pose(
    client_module: types.ModuleType,
) -> None:
    client = _client(client_module, types.SimpleNamespace(request=lambda msg: {}))
    client._active[0] = "task"
    last = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    client._history[0] = [{"action": [0.0] * 7 + [0.4], "base_pose7": last}]
    client._query_server = lambda request: (_ for _ in ()).throw(
        client_module.ServeRefusal("Server error: ee_pose has non-finite values")
    )
    obs = _obs()
    obs["proprio_obs"]["ee_pos"][:] = np.nan
    action = client.infer(obs, "task", env_id=0)["action"]
    np.testing.assert_array_equal(action, np.array([*last, 0.4], np.float32))


def test_refusal_holds_the_last_finite_pose_after_a_blow_up_between_calls(
    client_module: types.ModuleType,
) -> None:
    """The pose turns non-finite on steps without a policy call, before the refusal."""
    client = _client(client_module, types.SimpleNamespace(request=lambda msg: {}))
    client._active[0] = "task"
    finite = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    nan = [float("nan")] * 3 + [1.0, 0.0, 0.0, 0.0]
    client._history[0] = [
        {"action": [0.0] * 7 + [0.4], "base_pose7": finite},
        {"action": [0.0] * 7 + [0.4], "base_pose7": nan},
        {"action": [0.0] * 7 + [0.4], "base_pose7": nan},
    ]
    client._query_server = lambda request: (_ for _ in ()).throw(
        client_module.ServeRefusal("Server error: ee_pose has non-finite values")
    )
    obs = _obs()
    obs["proprio_obs"]["ee_pos"][:] = np.nan
    action = client.infer(obs, "task", env_id=0)["action"]
    np.testing.assert_array_equal(action, np.array([*finite, 0.4], np.float32))


def test_refusal_with_no_finite_pose_to_hold_stops_the_run(
    client_module: types.ModuleType,
) -> None:
    client = _client(client_module, types.SimpleNamespace(request=lambda msg: {}))
    client._active[0] = "task"
    nan = [float("nan")] * 3 + [1.0, 0.0, 0.0, 0.0]
    client._history[0] = [{"action": [0.0] * 7 + [0.4], "base_pose7": nan}]
    client._query_server = lambda request: (_ for _ in ()).throw(
        client_module.ServeRefusal("Server error: ee_pose has non-finite values")
    )
    obs = _obs()
    obs["proprio_obs"]["ee_pos"][:] = np.nan
    with pytest.raises(RuntimeError, match="no finite pose"):
        client.infer(obs, "task", env_id=0)


def _cfg_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """ee_frame_policy_cfg with stand-ins for the positronic modules it imports."""
    cfg_dir = STOCK.parent / "positronic_cfg"
    stubs: dict[str, dict[str, object]] = {
        "positronic": {"geom": types.SimpleNamespace()},
        "positronic.drivers": {},
        "positronic.drivers.roboarm": {"command": types.SimpleNamespace()},
        "positronic.policy": {
            "keys": types.SimpleNamespace(
                CALL_ID="call_id", CALL_TIMINGS="call_timings"
            )
        },
        "positronic.policy.base": {
            "Policy": object,
            "PolicyRun": object,
            "Runtime": object,
            "Step": object,
        },
        "positronic.policy.remote": {"RemotePolicy": object},
        "positronic.policy.sequential": {"Sequential": object},
        "positronic_wire": {},
        "positronic_wire.wire": {
            "SESSION_PATH": "/api/v1/session",
            "HostPortAddress": lambda *a: a,
        },
    }
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.syspath_prepend(str(cfg_dir))
    monkeypatch.delitem(sys.modules, "ee_frame_policy_cfg", raising=False)
    import ee_frame_policy_cfg

    return ee_frame_policy_cfg


@pytest.mark.parametrize(
    ("url", "query"),
    [
        ("host:8000", ""),
        (
            "wss://h/api/v1/session?max_context_chunks=1",
            "max_context_chunks=1&schedule=chunked",
        ),
        ("ws://h:80/api/v1/session?schedule=rtc&x=1", "schedule=rtc&x=1"),
        ("https://h/api/v1/session/m?schedule=chunked", "schedule=chunked"),
    ],
)
def test_the_client_asks_for_the_chunked_schedule(
    monkeypatch: pytest.MonkeyPatch, url: str, query: str
) -> None:
    """RoboLab pauses the sim during a call, so the client asks for the chunked
    schedule unless the URL sets one. A deploy can set real-time chunking as its
    default. An empty query goes as written."""
    cfg = _cfg_module(monkeypatch)
    _name, address = cfg.wire_address(url)
    assert address[3] == query


@pytest.mark.parametrize("url", ["host:8000", "wss://h/api/v1/session?"])
def test_an_empty_query_stays_empty(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """``--query ""`` reaches a server that accepts no session settings as written:
    a server built before the ``schedule`` key refuses ``schedule=chunked``."""
    cfg = _cfg_module(monkeypatch)
    assert cfg.with_chunked_schedule("") == ""
    _name, address = cfg.wire_address(url)
    assert address[3] == ""


def test_the_example_query_asks_for_the_chunked_schedule() -> None:
    tree = ast.parse((STOCK / "example_client.py").read_text())
    query = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "QUERY" for t in node.targets)
    )
    assert "schedule=chunked" in query.split("&")
