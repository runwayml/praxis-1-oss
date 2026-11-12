"""The LeRobot client of the so101 rig (demo/so101/so101_client.py), without a
robot or a server. The module is loaded by path, as RoboLab's client is."""

from __future__ import annotations

import collections
import http.client
import importlib.util
import json
import types
from pathlib import Path

import msgpack
import numpy as np
import pytest

pytest.importorskip("PIL")

CLIENT = Path(__file__).resolve().parents[2] / "demo" / "so101" / "so101_client.py"


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


# ---------------------------------------------------------------------------
# The rest moves
# ---------------------------------------------------------------------------

REST = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])


class _Arm:
    """A fake arm that records each command. ``follows`` False is an arm that
    does not move (blocked, or a calibration that puts the rest value past a
    stop). ``frames`` gives it one camera per name."""

    def __init__(
        self,
        joints: tuple[str, ...],
        q: np.ndarray,
        follows: bool = True,
        frames: tuple[str, ...] = (),
    ) -> None:
        self.joints, self.q = joints, np.array(q, np.float64)
        self.follows, self.frames = follows, frames
        self.sent: list[np.ndarray] = []
        self.read_before: list[np.ndarray] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self) -> dict:
        obs: dict = {
            f"{j}.pos": float(v) for j, v in zip(self.joints, self.q, strict=True)
        }
        for name in self.frames:
            obs[name] = np.zeros((480, 640, 3), np.uint8)
        return obs

    def send_action(self, action: dict) -> dict:
        self.read_before.append(self.q.copy())
        cmd = np.array([action[f"{j}.pos"] for j in self.joints])
        self.sent.append(cmd)
        if self.follows:
            self.q = cmd
        return action


def test_the_rest_move_folds_then_turns_then_sets_the_wrist(
    client: types.ModuleType,
) -> None:
    """From a reach pose the arm folds (shoulder lift, elbow), then turns
    (pan), then sets the wrist and the gripper. No command leads the reading
    by more than REST_LEAD, and the last reading is the rest pose."""
    start = np.array([40.0, 20.0, -30.0, 10.0, 30.0, 60.0])
    arm = _Arm(client.JOINTS, start)
    last = client.move_to_rest(arm, REST, paced=False)
    np.testing.assert_allclose(last, REST)
    sent, before = np.array(arm.sent), np.array(arm.read_before)
    assert np.abs(sent - before).max() <= client.REST_LEAD + 1e-9
    moved = np.abs(sent - start) > 1e-9  # (command, joint): the joint left the start
    first = [int(np.argmax(moved[:, j])) for j in range(6)]
    fold_done = max(
        int(np.argmax(np.isclose(sent[:, j], REST[j]))) for j in (1, 2)
    )  # the command that puts shoulder lift and elbow at rest
    pan_done = int(np.argmax(np.isclose(sent[:, 0], REST[0])))
    assert first[1] == first[2] == 0  # the fold starts at once, both joints together
    assert first[0] > fold_done  # the pan waits for the fold
    assert min(first[3:]) > pan_done  # the wrist and the gripper wait for the pan
    # the joints of a phase arrive together: the same fraction of each distance
    np.testing.assert_allclose(
        (sent[0, 1] - start[1]) / (REST[1] - start[1]),
        (sent[0, 2] - start[2]) / (REST[2] - start[2]),
    )
    # no command moves a joint faster than REST_SPEED, the last one included
    steps = np.abs(np.diff(np.vstack([start, sent]), axis=0))
    assert steps.max() <= client.REST_SPEED / 30.0 + 1e-9


def test_the_rest_move_ends_the_push_on_a_blocked_arm(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An arm that does not follow gets a push of at most REST_LEAD for
    REST_STALL_S per phase; then the command goes back to the reading. The
    pan phase is skipped after a fold that stopped short. The rest check
    names the joints that stayed far."""
    clock = _Clock()
    monkeypatch.setattr(client, "time", clock)
    start = np.array([40.0, 20.0, -30.0, 10.0, 30.0, 60.0])
    arm = _Arm(client.JOINTS, start, follows=False)
    t0 = clock.t
    last = client.move_to_rest(arm, REST, hz=30.0)
    np.testing.assert_array_equal(last, start)
    sent = np.array(arm.sent)
    assert np.abs(sent - start).max() <= client.REST_LEAD + 1e-9
    np.testing.assert_array_equal(sent[-1], start)  # no push is left at the end
    assert not np.any(sent[:, 0] != start[0])  # no pan of an arm that is not folded
    stall = 2 * client.REST_STALL_S  # the fold, then the wrist and the gripper
    assert stall < clock.t - t0 < stall + 0.5
    assert len(client.rest_check(last)) == 2


def test_the_rest_pose_flag_and_the_new_defaults(client: types.ModuleType) -> None:
    base = ["--url", "ws://h/x", "--prompt", "p"]
    a = client.parse_args(base)
    assert a.ui_port is None and a.ui_host == "127.0.0.1"
    assert not a.no_rest_move and a.rest_pose is None
    np.testing.assert_array_equal(
        client.rest_vector(a), [0.0, -100.0, 95.0, 70.0, 0.0, 1.4]
    )
    a = client.parse_args(
        [*base, "--rest-pose", "10,-95,90,60,5", "--grip-closed", "3.5"]
    )
    np.testing.assert_array_equal(
        client.rest_vector(a), [10.0, -95.0, 90.0, 60.0, 5.0, 3.5]
    )
    for bad in ("0,-100,95,70", "0,-100,95,70,0,2", "a,b,c,d,e", "0,-100,95,70,400"):
        with pytest.raises(SystemExit):
            client.parse_args([*base, "--rest-pose", bad])


def test_a_stop_request_ends_a_chunk_before_the_next_row(
    client: types.ModuleType,
) -> None:
    arm = _Arm(client.JOINTS, np.zeros(6))
    rows = np.cumsum(np.ones((8, 6)), axis=0)
    hist: collections.deque = collections.deque()
    asked = iter([False, False, False, True])
    with pytest.raises(client.StopRun):
        client.play_rows(
            arm, rows, np.zeros(6), {"state": np.zeros(6)}, hist, [], 2, 1 / 30,
            dry=False, paced=False, stop=lambda: next(asked),
        )  # fmt: skip
    assert len(hist) == 3 and len(arm.sent) == 3 * 2
    np.testing.assert_array_equal(arm.q, rows[2])


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _request(
    port: int,
    method: str,
    path: str,
    body: object = None,
    headers: dict[str, str] | None = None,
    host: str | None = None,
) -> tuple[int, bytes]:
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=host is not None)
        if host is not None:
            conn.putheader("Host", host)
        raw = b"" if body is None else json.dumps(body).encode()
        send = {"Content-Type": "application/json"} if body is not None else {}
        send.update(headers or {})
        for k, v in send.items():
            conn.putheader(k, v)
        conn.putheader("Content-Length", str(len(raw)))
        conn.endheaders(raw)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def test_the_page_serves_the_views_and_takes_a_prompt(
    client: types.ModuleType,
) -> None:
    ui = client.Ui("127.0.0.1", 0, "Pick up the cube.", ["exterior", "wrist"])
    try:
        port = ui.port
        status, body = _request(port, "GET", "/status")
        got = json.loads(body)
        assert status == 200 and got["prompt"] == "Pick up the cube."
        assert got["views"] == ["exterior", "wrist"] and got["phase"] == "starting"
        status, body = _request(port, "GET", "/")
        assert status == 200 and b"SO-101 client" in body
        # one document: nothing in it comes from another host
        assert not any(x in body for x in (b"http:", b"https:", b"@import", b"url("))
        for name in ("start", "stop", "rest", "prompt"):
            assert f'id="{name}"'.encode() in body
        # a view has no frame until the client reads one
        assert _request(port, "GET", "/frame/exterior.jpg")[0] == 503
        ui.publish("exterior", np.full((312, 416, 3), 90, np.uint8))
        status, body = _request(port, "GET", "/frame/exterior.jpg")
        assert status == 200 and body[:2] == b"\xff\xd8"
        assert _request(port, "GET", "/frame/overhead.jpg")[0] == 404
        # the prompt: normalized, refused when empty or too long
        status, body = _request(
            port, "POST", "/prompt", {"prompt": "  Put the cube\n in the box. "}
        )
        assert status == 200
        assert json.loads(body) == {"prompt": "Put the cube in the box."}
        assert ui.get_prompt() == "Put the cube in the box."
        assert _request(port, "POST", "/prompt", {"prompt": "  "})[0] == 400
        too_long = "x" * (client.MAX_PROMPT_CHARS + 1)
        assert _request(port, "POST", "/prompt", {"prompt": too_long})[0] == 400
        assert _request(port, "POST", "/prompt", {"prompt": 3})[0] == 400
        assert ui.get_prompt() == "Put the cube in the box."
        # another site can neither read the page nor post to it
        assert _request(port, "GET", "/status", host="evil.example")[0] == 403
        form = {"Content-Type": "application/x-www-form-urlencoded"}
        assert _request(port, "POST", "/stop", {}, headers=form)[0] == 415
        cross = {"Origin": "http://evil.example"}
        assert _request(port, "POST", "/stop", {}, headers=cross)[0] == 403
        assert ui.take() is None and not ui.interrupt.is_set()
        # the buttons: stop and rest end what runs, start counts only when idle
        own = {"Origin": f"http://127.0.0.1:{port}"}
        assert _request(port, "POST", "/stop", {}, headers=own)[0] == 200
        assert ui.interrupt.is_set() and ui.take() == "stop" and ui.take() is None
        assert _request(port, "POST", "/start", {})[0] == 200
        assert ui.take() == "start"
        ui.update(mode="running")
        assert _request(port, "POST", "/start", {})[0] == 409 and ui.take() is None
        assert _request(port, "POST", "/rest", {})[0] == 200 and ui.take() == "rest"
        ui.update(mode="moving")
        assert _request(port, "POST", "/rest", {})[0] == 409
        assert _request(port, "POST", "/quit", {})[0] == 404  # no route ends the client
    finally:
        ui.close()


def test_the_stream_is_mjpeg_of_the_latest_frame(client: types.ModuleType) -> None:
    import http.client

    ui = client.Ui("127.0.0.1", 0, "p", ["exterior"])
    try:
        ui.publish("exterior", np.zeros((312, 416, 3), np.uint8))
        conn = http.client.HTTPConnection("127.0.0.1", ui.port, timeout=5)
        conn.request("GET", "/stream/exterior.mjpg")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type").startswith("multipart/x-mixed-replace")
        head = resp.read(120)
        assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        assert b"\xff\xd8" in head
        conn.close()
    finally:
        ui.close()


# ---------------------------------------------------------------------------
# A run on a fake arm and a fake server: rest moves, prompt switch, stop
# ---------------------------------------------------------------------------


class _Server:
    """A fake praxis server: each connection gets ``ready``, then one chunk
    per request that lifts the arm 1 deg per row. ``on_request`` runs on each
    request. Request number ``fail_at`` gets an error answer."""

    def __init__(
        self, client: types.ModuleType, on_request: object = None, fail_at: int = 0
    ) -> None:
        self.client, self.on_request, self.fail_at = client, on_request, fail_at
        self.tasks: list[str] = []
        self.anchors: list[np.ndarray] = []
        self.inbox: list[bytes] = []

    def __enter__(self) -> _Server:
        ready = {"status": "ready", "meta": {"stack_entries": 25}}
        self.inbox = [self.client.dumps(ready)]
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def send(self, raw: bytes) -> None:
        msg = self.client.loads(raw)
        state = np.asarray(msg["state"], np.float64)[-1]
        self.tasks.append(msg["task"])
        self.anchors.append(state)
        if self.on_request is not None:
            self.on_request(len(self.tasks))
        if len(self.tasks) == self.fail_at:
            self.inbox.append(self.client.dumps({"error": "boom"}))
            return
        rows = np.tile(state, (24, 1))
        rows[:, 1] += np.arange(1, 25)
        self.inbox.append(
            self.client.dumps(
                {
                    "actions": rows.astype(np.float32),
                    "effective_prompt": msg["task"],
                    "server_timing": {"policy_ms": 5.0},
                }
            )
        )

    def recv(self, timeout: float | None = None) -> bytes:
        if not self.inbox:
            raise TimeoutError
        return self.inbox.pop(0)


def _run(
    client: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    arm: _Arm,
    server: _Server,
    args: list[str],
    pages: list | None = None,
    script: list[str] | None = None,
    waits: list | None = None,
) -> None:
    """``client.main`` on ``arm`` and ``server`` with a fake clock. ``pages``
    gets the page of the run as soon as the run makes it. Under the page the
    buttons come from ``script``, one per wait of the client; Ctrl-C comes
    when the script is over. ``waits`` gets the status of the page and the
    pose of the arm at each wait."""
    wsc = pytest.importorskip("websockets.sync.client")
    monkeypatch.setattr(wsc, "connect", lambda *a, **k: server)
    monkeypatch.setattr(client, "make_robot", lambda a, cams: arm)
    monkeypatch.setattr(client, "time", _Clock())
    made = [] if pages is None else pages
    base = client.Ui

    class Page(base):
        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)
            made.append(self)

        def wait_command(self, idle: object = None) -> str:
            if idle is not None:
                idle()
            if waits is not None:
                waits.append((self.snapshot(), arm.q.copy()))
            if not script:
                raise KeyboardInterrupt
            return script.pop(0)

    monkeypatch.setattr(client, "Ui", Page)
    client.main(
        ["--url", "ws://h/api/v1/session", "--prompt", "Pick up the cube.", *args]
    )


def test_a_run_starts_and_ends_at_the_rest_pose(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live run from a reach pose: the arm goes to the rest pose before the
    first request, plays the chunks, returns to the rest pose and disconnects."""
    arm = _Arm(
        client.JOINTS, [40.0, 20.0, -30.0, 10.0, 30.0, 60.0], frames=("exterior",)
    )
    server = _Server(client)
    log = tmp_path / "log.jsonl"
    rest = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 3.8])
    _run(client, monkeypatch, arm, server,
         ["--max-chunks", "2", "--grip-closed", "3.8", "--grip-open", "37.3",
          "--log-jsonl", str(log)])  # fmt: skip
    np.testing.assert_allclose(server.anchors[0], rest, atol=1e-5)
    assert server.anchors[1][1] == pytest.approx(rest[1] + 24)  # the chunk was played
    np.testing.assert_allclose(arm.q, rest)  # back at rest
    assert arm.connected is False
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["prompt"] for r in records] == ["Pick up the cube."] * 2


def test_no_rest_move_keeps_the_start_check_and_ends_in_place(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = _Arm(
        client.JOINTS, [40.0, 20.0, -30.0, 10.0, 30.0, 60.0], frames=("exterior",)
    )
    with pytest.raises(SystemExit, match="not at the folded rest pose"):
        _run(client, monkeypatch, arm, _Server(client), ["--no-rest-move"])
    assert arm.connected is False and len(arm.sent) == 1  # the hold of the end only
    arm = _Arm(client.JOINTS, [5.0, -98.0, 93.0, 2.0, 0.0, 3.0], frames=("exterior",))
    server = _Server(client)
    _run(client, monkeypatch, arm, server, ["--no-rest-move", "--max-chunks", "1"])
    np.testing.assert_allclose(server.anchors[0], [5.0, -98.0, 93.0, 2.0, 0.0, 3.0])
    assert arm.q[1] == pytest.approx(-98.0 + 24)  # where the chunk ended


def test_a_blocked_start_stops_the_client_without_a_second_push(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = _Arm(client.JOINTS, [40.0, 20.0, -30.0, 10.0, 30.0, 60.0],
               follows=False, frames=("exterior",))  # fmt: skip
    server = _Server(client)
    with pytest.raises(SystemExit, match="The rest move stopped short"):
        _run(client, monkeypatch, arm, server, [])
    assert server.tasks == [] and arm.connected is False
    # the start move pushed (3 phases); the end only holds the reading
    np.testing.assert_array_equal(arm.sent[-1], arm.q)
    pushes = [s for s in arm.sent if np.abs(s - arm.q).max() > 1e-9]
    assert pushes and np.abs(np.array(pushes) - arm.q).max() <= client.REST_LEAD + 1e-9


def test_the_buttons_start_stop_and_return_to_rest(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client under the page waits for Start. Stop ends the session
    inside a chunk and the arm holds. Return to rest folds the arm. A second
    Start is a new session with the prompt of the page, and Return to rest
    ends it. No button ends the client: Ctrl-C does."""
    old, new = "Pick up the cube.", "Put the cube in the box."
    rest = [0.0, -100.0, 95.0, 70.0, 0.0, 2.0]
    arm = _Arm(client.JOINTS, rest, frames=("exterior",))
    pages: list = []
    seen: list[dict] = []  # the status of the page at each request
    waits: list = []

    def on_request(n: int) -> None:
        seen.append(pages[0].snapshot())
        if n == 1:
            pages[0].set_prompt(new)
        if n == 2:
            assert not pages[0].request("start")  # a session runs already
        if n == 3:
            assert pages[0].request("stop")
        if n == 5:
            assert pages[0].request("rest")

    server = _Server(client, on_request)
    _run(client, monkeypatch, arm, server, ["--ui-port", "0", "--grip-closed", "2"],
         pages, script=["start", "rest", "start"], waits=waits)  # fmt: skip
    assert server.tasks == [old, new, new, new, new]
    # at request 2 the page still shows the prompt of the answer to request 1
    assert seen[1]["effective_prompt"] == old and seen[1]["prompt"] == new
    assert seen[2]["effective_prompt"] == new and seen[2]["mode"] == "running"
    assert seen[2]["phase"] == "waiting for the chunk" and seen[2]["chunk"] == 2
    assert seen[2]["rows"] == 24 and seen[2]["policy_ms"] == 5
    assert seen[2]["views"] == ["exterior"] and len(seen[2]["state"]) == 6
    # wait 1: before Start. wait 2: after Stop, where chunk 1 ended.
    # wait 3 and 4: after a Return to rest.
    assert [w[0]["mode"] for w in waits] == ["idle"] * 4
    assert [w[0]["phase"] for w in waits] == [
        "ready: press Start", "stopped: holding", "at rest", "at rest",
    ]  # fmt: skip
    np.testing.assert_allclose(waits[0][1], rest)
    assert waits[1][1][1] == pytest.approx(-100.0 + 48)  # no row of chunk 2 ran
    np.testing.assert_allclose(waits[2][1], rest)
    np.testing.assert_allclose(waits[3][1], rest)
    # the second session starts from the rest pose and plays one chunk
    assert server.anchors[3][1] == pytest.approx(-100.0)
    assert server.anchors[4][1] == pytest.approx(-100.0 + 24)
    np.testing.assert_allclose(arm.q, rest)
    assert arm.connected is False and pages[0].closed.is_set()


def test_the_end_of_a_session_does_not_end_the_page(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the page, ``--max-chunks`` and an error of the server end the
    session: the arm holds, the page says why and waits for a button.
    Without the page the same error ends the client."""
    rest = [0.0, -100.0, 95.0, 70.0, 0.0, 2.0]
    arm = _Arm(client.JOINTS, rest, frames=("exterior",))
    server = _Server(client, fail_at=2)
    waits: list = []
    _run(client, monkeypatch, arm, server,
         ["--ui-port", "0", "--autostart", "--max-chunks", "1", "--grip-closed", "2"],
         script=["start"], waits=waits)  # fmt: skip
    assert len(server.tasks) == 2 and [w[0]["mode"] for w in waits] == ["idle"] * 2
    assert waits[0][0]["message"] == "the session ended after 1 chunks (--max-chunks)"
    assert waits[0][1][1] == pytest.approx(-100.0 + 24)  # holding where it ended
    assert waits[1][0]["message"] == "server error: boom"
    np.testing.assert_allclose(arm.q, rest)  # Ctrl-C returned it to rest
    arm = _Arm(client.JOINTS, rest, frames=("exterior",))
    with pytest.raises(SystemExit, match="server error: boom"):
        _run(
            client, monkeypatch, arm, _Server(client, fail_at=1), ["--grip-closed", "2"]
        )
    assert arm.connected is False


def test_stop_ends_a_rest_move(client: types.ModuleType) -> None:
    arm = _Arm(client.JOINTS, [40.0, 20.0, -30.0, 10.0, 30.0, 60.0])
    asked = iter([False] * 10 + [True])
    with pytest.raises(client.StopRun):
        client.move_to_rest(arm, REST, paced=False, stop=lambda: next(asked))
    assert len(arm.sent) == 10 and -100.0 < arm.q[1] < 20.0  # part of the fold


def test_ctrl_c_in_the_start_move_stops_the_arm(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C during the move to the rest pose ends the move: the client holds
    the reading, does not start the end move, and disconnects."""

    class Interrupted(_Arm):
        fired = False

        def send_action(self, action: dict) -> dict:
            if len(self.sent) == 5 and not self.fired:
                self.fired = True
                raise KeyboardInterrupt
            return super().send_action(action)

    start = [40.0, 20.0, -30.0, 10.0, 30.0, 60.0]
    arm = Interrupted(client.JOINTS, start, frames=("exterior",))
    server = _Server(client)
    _run(client, monkeypatch, arm, server, [])
    assert server.tasks == [] and arm.connected is False
    assert len(arm.sent) == 6  # 5 commands of the move, then the hold
    np.testing.assert_array_equal(arm.sent[-1], arm.sent[-2])
    assert abs(arm.q[1] - start[1]) < 5 * client.REST_SPEED / 30 + 1e-9


class _BlockedElbow(_Arm):
    """An arm whose elbow stops at 70 deg while ``blocks(q)`` holds: the
    gripper of a flexed wrist in the way, or anything that a later phase
    clears."""

    def __init__(self, *args: object, blocks: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = blocks

    def send_action(self, action: dict) -> dict:
        before = self.q.copy()
        out = super().send_action(action)
        if self.blocks(before):
            self.q[2] = min(self.q[2], max(before[2], 70.0))
        return out


def test_a_wrist_flexed_past_rest_unflexes_before_the_fold(
    client: types.ModuleType,
) -> None:
    """The reach poses of a run end with the wrist flexed past its rest value
    (100 deg against 70). The gripper then blocks the elbow, so the wrist
    goes to its rest value first and the fold is not pushed against it."""
    start = np.array([10.0, 20.0, -30.0, 100.0, 0.0, 50.0])
    phases = client.rest_phases(start, REST)
    assert phases[0] == ("wrist_flex",) and tuple(phases[1:]) == client.REST_PHASES
    # a wrist short of the rest value, or on the other side of 0, stays last
    for wrist in (10.0, 69.0, 71.0, -40.0, 0.0):
        q = start.copy()
        q[3] = wrist
        assert tuple(client.rest_phases(q, REST)) == client.REST_PHASES
    arm = _BlockedElbow(client.JOINTS, start, blocks=lambda q: q[3] > 75.0)
    last = client.move_to_rest(arm, REST, paced=False)
    np.testing.assert_allclose(last, REST)
    sent = np.array(arm.sent)
    moved = np.abs(sent - start) > 1e-9
    wrist_done = int(np.argmax(np.isclose(sent[:, 3], REST[3])))
    assert int(np.argmax(moved[:, 3])) == 0  # the wrist moves first
    assert int(np.argmax(moved[:, 2])) > wrist_done  # the fold waits for it
    # the fold was never pushed against the gripper
    lead = np.abs(sent - np.array(arm.read_before)).max()
    assert lead <= client.REST_ARRIVE + 1e-9 < client.REST_LEAD


def test_a_fold_that_a_later_phase_frees_gets_a_second_pass(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The elbow is blocked while the gripper is open. The first pass stops
    the fold short (a bounded push, one stall), leaves the pan where it is
    and closes the gripper; the second pass finishes the fold, and only then
    the arm turns."""
    clock = _Clock()
    monkeypatch.setattr(client, "time", clock)
    lines: list[str] = []
    monkeypatch.setattr(client, "say", lines.append)
    start = np.array([40.0, 20.0, -30.0, 10.0, 30.0, 60.0])
    arm = _BlockedElbow(client.JOINTS, start, blocks=lambda q: q[5] > 10.0)
    last = client.move_to_rest(arm, REST, hz=30.0)
    np.testing.assert_allclose(last, REST)
    sent, before = np.array(arm.sent), np.array(arm.read_before)
    assert np.abs(sent - before).max() <= client.REST_LEAD + 1e-9
    assert len(lines) == 2 and "elbow_flex at 70.0 (rest 95.0)" in lines[0]
    assert "the pan stays" in lines[1]
    turning = sent[:, 0] != start[0]  # the commands that turn the arm
    assert turning.any() and before[turning, 2].min() > REST[2] - client.REST_ARRIVE


def test_a_late_camera_frame_does_not_stop_the_joint_readings(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LeRobot's get_observation raises on a late camera frame. The readings
    without views (the hold, the rest moves) come from the motor bus alone;
    an entry with views ends the session after the wait, not the client."""

    class Bus:
        def sync_read(self, name: str, **_: object) -> dict[str, float]:
            assert name == "Present_Position"
            return {j: float(i) for i, j in enumerate(client.JOINTS)}

    class Robot:
        bus = Bus()

        def get_observation(self) -> dict:
            raise TimeoutError("latest frame is too old")

    state = client.read_entry(Robot(), [])["state"]
    np.testing.assert_array_equal(state, np.arange(6.0))
    monkeypatch.setattr(client, "FRAME_WAIT_S", 0.0)
    with pytest.raises(client.SessionError, match="a camera sends no frame"):
        client.read_entry(Robot(), ["exterior"])


def test_a_camera_whose_reader_ended_is_connected_again(
    client: types.ModuleType,
) -> None:
    """LeRobot's camera ends its read thread after a row of failed reads and
    every later read raises. The client connects the camera again and the
    entry comes."""

    class Thread:
        alive = False

        def is_alive(self) -> bool:
            return self.alive

    class Cam:
        def __init__(self) -> None:
            self.thread, self.connects = Thread(), 0

        def disconnect(self) -> None:
            pass

        def connect(self) -> None:
            self.connects += 1
            self.thread.alive = True

    class Robot:
        def __init__(self) -> None:
            self.cameras = {"exterior": Cam()}

        def get_observation(self) -> dict:
            if not self.cameras["exterior"].thread.is_alive():
                raise RuntimeError("OpenCVCamera(1) read thread is not running.")
            obs = {f"{j}.pos": 0.0 for j in client.JOINTS}
            return {**obs, "exterior": np.zeros((480, 640, 3), np.uint8)}

    robot = Robot()
    entry = client.read_entry(robot, ["exterior"])
    assert entry["exterior"].shape == (312, 416, 3)
    assert robot.cameras["exterior"].connects == 1


def test_the_rest_check_uses_the_target_of_the_move(client: types.ModuleType) -> None:
    """With ``--rest-pose`` the check after the move is against that pose."""
    pose = np.array([0.0, -60.0, 60.0, 70.0, 0.0, 2.0])
    assert len(client.rest_check(pose)) == 2  # far from the training pose
    assert client.rest_check(pose, rest=pose) == []
    off = pose + np.array([0.0, 30.0, 0.0, 0.0, 0.0, 0.0])
    assert len(client.rest_check(off, rest=pose)) == 1


def test_the_page_answers_its_own_names_on_any_bind(client: types.ModuleType) -> None:
    """A bind on every interface still refuses a rebound Host: the page
    answers the loopback names (and the address of a real ``--ui-host``)."""
    ui = client.Ui("0.0.0.0", 0, "Pick up the cube.", ["exterior"])
    try:
        ui.update(mode="idle")

        def post(host: str) -> int:
            conn = http.client.HTTPConnection("127.0.0.1", ui.port, timeout=5)
            try:
                headers = {"Host": host, "Origin": f"http://{host}",
                           "Content-Type": "application/json"}  # fmt: skip
                conn.request("POST", "/start", body=b"{}", headers=headers)
                return conn.getresponse().status
            finally:
                conn.close()

        assert post(f"evil.example:{ui.port}") == 403
        assert ui.take() is None
        assert post(f"127.0.0.1:{ui.port}") == 200
        assert ui.take() == "start"
        # a control character is not a prompt
        with pytest.raises(ValueError, match="control character"):
            ui.set_prompt("Pick up\x00 the cube.")
    finally:
        ui.close()


def test_sigterm_ends_the_client_as_ctrl_c(client: types.ModuleType) -> None:
    import os
    import signal

    with pytest.raises(KeyboardInterrupt), client.signals_interrupt():
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(100):  # the handler runs between two bytecodes
            pass
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_a_stop_in_the_start_move_cancels_autostart(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``--autostart``, Stop on the page during the start move holds the
    arm and no session starts."""
    pages: list = []

    class Stopped(_Arm):
        def send_action(self, action: dict) -> dict:
            if len(self.sent) == 5:
                pages[0].request("stop")
            return super().send_action(action)

    arm = Stopped(client.JOINTS, [40.0, 20.0, -30.0, 10.0, 30.0, 60.0],
                  frames=("exterior",))  # fmt: skip
    server = _Server(client)
    _run(client, monkeypatch, arm, server, ["--ui-port", "0", "--autostart"],
         pages=pages, script=[])  # fmt: skip
    assert server.tasks == []  # the policy never ran


def test_ctrl_c_in_a_rest_move_of_the_page_stops_the_arm(
    client: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C while Return to rest moves the arm ends that move: the client
    holds and disconnects, with no second rest move."""

    class Interrupted(_Arm):
        armed = False

        def send_action(self, action: dict) -> dict:
            if self.armed and self.q[1] < -85.0:  # on the way back to rest
                self.armed = False
                self.stopped_at = len(self.sent)
                raise KeyboardInterrupt
            out = super().send_action(action)
            self.armed = self.armed or self.q[1] > -80.0  # the session lifted the arm
            return out

    arm = Interrupted(client.JOINTS, REST.copy(), frames=("exterior",))
    server = _Server(client)
    _run(client, monkeypatch, arm, server,
         ["--ui-port", "0", "--max-chunks", "1", "--grip-closed", "2"],
         script=["start", "rest"])  # fmt: skip
    assert len(server.tasks) == 1 and arm.connected is False
    assert len(arm.sent) == arm.stopped_at + 1  # only the hold after Ctrl-C
    assert arm.q[1] > REST[1] + 5.0  # not at rest: the move did not run again
