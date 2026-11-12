"""The LeRobot client of the so101 rig (eval/so101/so101_client.py), without a
robot or a server. The module is loaded by path, as RoboLab's client is."""

from __future__ import annotations

import collections
import importlib.util
import json
import types
from pathlib import Path

import msgpack
import numpy as np
import pytest

pytest.importorskip("PIL")

CLIENT = Path(__file__).resolve().parents[2] / "eval" / "so101" / "so101_client.py"


@pytest.fixture(scope="module")
def client() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("so101_client", CLIENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_client_imports_no_praxis(client: types.ModuleType) -> None:
    text = CLIENT.read_text()
    assert "import praxis" not in text and "from praxis" not in text
    assert client.FPS == 15.0 and client.FIT_SIDE == 416


def test_frames_fit_the_416_box_and_never_upscale(client: types.ModuleType) -> None:
    big = np.zeros((480, 640, 3), np.uint8)
    assert client.fit_frame(big).shape == (312, 416, 3)
    tall = np.zeros((1280, 720, 3), np.uint8)
    assert client.fit_frame(tall).shape == (416, 234, 3)
    small = np.zeros((240, 320, 3), np.uint8)
    assert client.fit_frame(small) is small


def test_the_request_is_the_stack_of_the_server(client: types.ModuleType) -> None:
    rng = np.random.default_rng(0)
    entries = [
        {
            "state": np.arange(6, dtype=np.float64) + t,
            "exterior": rng.integers(0, 255, (312, 416, 3), dtype=np.uint8),
            "wrist": rng.integers(0, 255, (312, 416, 3), dtype=np.uint8),
        }
        for t in range(3)
    ]
    raw = client.build_request(entries, ["exterior", "wrist"], "Pick up the cube.", 90)
    msg = msgpack.unpackb(raw, strict_map_key=False)
    assert msg["task"] == "Pick up the cube."
    state = msg["state"]
    assert state[b"shape"] == [3, 6] and np.dtype(state[b"dtype"]) == np.float32
    np.testing.assert_array_equal(
        np.frombuffer(state[b"data"], np.float32).reshape(3, 6)[2], np.arange(6) + 2
    )
    for view in ("image.exterior", "image.wrist"):
        assert msg[view][b"__jpeg__"] is True and len(msg[view][b"frames"]) == 3
    one = msgpack.unpackb(
        client.build_request(entries, ["exterior"], "x", 90), strict_map_key=False
    )
    assert "image.wrist" not in one
    assert len(raw) <= client.MAX_MESSAGE_BYTES


def test_the_clamp_is_off_by_default_and_limits_each_row(
    client: types.ModuleType,
) -> None:
    start = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])
    targets = np.array(
        [[30.0, -100.0, 95.0, 70.0, 0.0, 120.0], [60.0, -100.0, 95.0, 70.0, 0.0, -5.0]]
    )
    off = client.clamp_chunk(targets, start, None, None)
    np.testing.assert_array_equal(off[:, :5], targets[:, :5])
    assert off[:, 5].tolist() == [100.0, 0.0]  # gripper.pos stays in 0..100
    on = client.clamp_chunk(targets, start, 4.0, 8.0)
    assert on[:, 0].tolist() == [4.0, 8.0] and on[:, 5].tolist() == [10.0, 2.0]
    joints_only = client.clamp_chunk(targets, start, 4.0, None)
    assert joints_only[:, 0].tolist() == [4.0, 8.0] and joints_only[0, 5] == 100.0


def test_the_control_loop_interpolates_the_15_hz_rows(client: types.ModuleType) -> None:
    prev = np.zeros(6)
    row = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    two = client.interpolate_rows(prev, row, 2)  # 30 Hz: 2 commands per row
    np.testing.assert_allclose(two, [row / 2, row])
    one = client.interpolate_rows(prev, row, 1)  # 15 Hz: the row itself
    np.testing.assert_allclose(one, [row])


class _Clock:
    """A fake ``time``: ``sleep`` advances ``monotonic``, nothing else does."""

    def __init__(self) -> None:
        self.t = 100.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0
        self.t += seconds


class _StallingRobot:
    """A fake robot whose camera read stalls once, on one row of the chunk.

    It records when each command went out and when each observation was
    read, on the fake clock.
    """

    def __init__(
        self, clock: _Clock, joints: tuple[str, ...], stall_on_read: int, stall_s: float
    ) -> None:
        self.clock, self.joints = clock, joints
        self.stall_on_read, self.stall_s = stall_on_read, stall_s
        self.q = np.zeros(6)
        self.reads: list[float] = []
        self.sends: list[float] = []

    def get_observation(self) -> dict:
        self.reads.append(self.clock.t)
        if len(self.reads) == self.stall_on_read:
            self.clock.t += self.stall_s  # a USB or camera hiccup, a GC pause
        return {f"{j}.pos": float(v) for j, v in zip(self.joints, self.q, strict=True)}

    def send_action(self, action: dict) -> dict:
        self.sends.append(self.clock.t)
        self.q = np.array([action[f"{j}.pos"] for j in self.joints])
        return action


def test_a_stalled_read_slows_the_rows_down_instead_of_bursting_them(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a stall of 3 row periods inside the camera read of row 2, the
    overdue deadline slips: every command still goes out one control period
    after the previous one, and the stack entries stay one row period apart.
    Without the slip, the commands due during the stall go out back to back."""
    clock = _Clock()
    monkeypatch.setattr(client, "time", clock)
    sub, period = 2, 1.0 / (client.FPS * 2)  # 30 Hz commands on 15 Hz rows
    row_period = sub * period
    stall = 3 * row_period
    # the first read is the anchor (done by the caller), the second is row 1
    robot = _StallingRobot(clock, client.JOINTS, stall_on_read=2, stall_s=stall)
    rows = np.cumsum(np.ones((8, 6)), axis=0)  # 8 rows, 1 deg per row per joint
    start = np.zeros(6)
    anchor = {"state": start}
    hist: collections.deque = collections.deque()

    last = client.play_rows(
        robot, rows, start, anchor, hist, [], sub, period, dry=False, paced=True
    )

    np.testing.assert_array_equal(last, rows[-1])
    assert len(hist) == len(rows) and hist[0] is anchor
    assert len(robot.sends) == len(rows) * sub
    gaps = np.diff(robot.sends)
    # no back-to-back commands: each goes out one control period after the
    # previous one, except across the stall, which is longer, not shorter
    assert gaps.min() >= period - 1e-9, gaps
    assert np.isclose(gaps, period).sum() == len(gaps) - 1
    assert gaps.max() > stall
    # one stack entry per executed row, each read as its row starts, so the
    # entries stay on the row grid: never closer than one row period
    row_gaps = np.diff(robot.reads)
    assert len(robot.reads) == len(rows) - 1
    assert row_gaps.min() >= row_period - 1e-9, row_gaps
    assert np.isclose(row_gaps, row_period).sum() == len(row_gaps) - 1
    # the stall cost exactly one stall: the chunk took its nominal time plus it
    assert np.isclose(
        robot.sends[-1] - robot.sends[0], (len(rows) * sub - 1) * period + stall
    )


def test_the_rest_pose_check_names_the_joints_that_are_far(
    client: types.ModuleType,
) -> None:
    rest = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])
    assert client.rest_check(rest) == []
    assert client.rest_check(rest + np.array([40.0, 0, 0, 0, 0, 0])) == []
    far = client.rest_check(np.array([0.0, -60.0, 95.0, 70.0, 0.0, 2.0]))
    assert len(far) == 1 and far[0].startswith("shoulder_lift.pos -60.0 deg")
    both = client.rest_check(np.zeros(6))
    assert len(both) == 2 and "elbow_flex.pos 0.0 deg (rest 95)" in both[1]


def test_the_session_url_uses_the_server_parameter_names(
    client: types.ModuleType,
) -> None:
    a = client.parse_args(
        ["--url", "ws://h:8000/api/v1/session", "--prompt", "Pick up the cube",
         "--grip-closed", "3.8", "--grip-open", "37.3", "--max-context-chunks", "2",
         "--session", '{"num_steps": 4, "sample_seed": 0}', "--fake-robot"]
    )  # fmt: skip
    assert a.dry_run and a.control_hz == 30.0 and a.max_step_deg is None
    assert not a.hold_torque_on_exit and a.max_relative_target is None
    url = client.session_url(a, 1)
    query = url.split("?", 1)[1]
    assert "views=1" in query and "prompt=Pick+up+the+cube" in query
    assert "grip_closed=3.8" in query and "grip_open=37.3" in query
    assert "max_context_chunks=2" in query and "num_steps=4" in query
    assert "sample_seed=0" in query and "exec_horizon" not in query
    for alias in ("cfg", "cfg_video", "shift", "history_chunks"):
        assert f"{alias}=" not in query
    with pytest.raises(SystemExit):
        client.parse_args(["--url", "ws://h/api/v1/session", "--fake-robot"])
    with pytest.raises(SystemExit):
        client.parse_args(["--url", "ws://h/x", "--prompt", "p", "--control-hz", "10"])
    with pytest.raises(SystemExit):
        client.parse_args(
            [
                "--url",
                "ws://h/x",
                "--prompt",
                "p",
                "--grip-closed",
                "5",
                "--grip-open",
                "5.5",
            ]
        )
    measure = client.parse_args(["--url", "ws://h/x", "--measure-grip"])
    assert measure.prompt is None


def test_the_fake_robot_follows_the_commands_and_measures_its_grip(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = client.FakeRobot(["exterior"])
    entry = client.read_entry(robot, ["exterior"])
    assert entry["exterior"].shape == (312, 416, 3)
    np.testing.assert_array_equal(entry["state"], [0.0, -100.0, 95.0, 70.0, 0.0, 2.0])
    client.send(robot, np.arange(6, dtype=np.float64), dry=False)
    np.testing.assert_array_equal(client.read_entry(robot, [])["state"], np.arange(6))
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert client.measure_grip(robot) == (5.0, 5.0) and robot.torque is False


def test_the_headers_come_from_a_file_only(
    client: types.ModuleType, tmp_path: Path
) -> None:
    a = client.parse_args(["--url", "ws://h/x", "--prompt", "p"])
    assert client.auth_headers(a) == {}
    headers = tmp_path / "headers.json"
    headers.write_text(json.dumps({"Modal-Key": "k", "Modal-Secret": "s"}))
    a = client.parse_args(
        ["--url", "ws://h/x", "--prompt", "p", "--headers-file", str(headers)]
    )
    assert client.auth_headers(a) == {"Modal-Key": "k", "Modal-Secret": "s"}
