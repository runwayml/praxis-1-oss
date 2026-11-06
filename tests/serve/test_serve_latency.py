"""Tests of the latency client (CPU only, no weights).

The end-to-end test runs the client against the server of the protocol tests:
the real server app with a stand-in for the networks.
"""

from __future__ import annotations

import json

import pytest

# The latency client needs the [serve] extra.
pytest.importorskip("msgpack")
pytest.importorskip("websockets")

from test_serve_protocol import server  # noqa: F401  (the fixture)

from praxis.serve import latency, params, wire
from praxis.serve.server import InferenceServer


def test_summarize() -> None:
    s = latency.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s == {"n": 5, "p50": 3.0, "p90": pytest.approx(4.6), "max": 5.0}
    assert latency.summarize([]) == {"n": 0}


def test_summary_skips_the_warmup_and_splits_first_calls() -> None:
    calls = [
        {"call": 0, "warmup": True, "round_trip_ms": 9000.0},
        {"call": 1, "warmup": False, "round_trip_ms": 10.0, "policy_ms": 8.0},
        {"call": 2, "warmup": False, "round_trip_ms": 30.0, "policy_ms": 20.0},
        {"call": 0, "warmup": False, "round_trip_ms": 7.0},
    ]
    out = latency.summarize_calls(calls)
    assert out["first"]["round_trip_ms"] == {"n": 1, "p50": 7.0, "p90": 7.0, "max": 7.0}
    assert out["later"]["round_trip_ms"]["n"] == 2
    assert out["later"]["round_trip_ms"]["max"] == 30.0
    assert out["later"]["policy_ms"]["p50"] == 14.0
    assert out["first"]["policy_ms"] == {"n": 0}
    text = latency.format_summary(out)
    assert "9000" not in text and "policy_ms" in text


def test_partial_server_timing_is_kept() -> None:
    """A server may send only a part of the server_timing block."""
    rec = latency.timing_record(100.0, {"deserialize_ms": 3.0, "policy_ms": 80.0})
    assert rec == {"deserialize_ms": 3.0, "policy_ms": 80.0, "other_ms": 17.0}
    assert latency.timing_record(100.0, {"policy_ms": 80.0}) == {"policy_ms": 80.0}
    assert latency.timing_record(100.0, None) == {}


def test_stack_layout_reads_the_handshake() -> None:
    stack = params.local_stack(("image.wrist",), 24, 15.0, 416, 234, True, True)
    assert latency.stack_layout({"local_stack": stack}) == (25, 416, 234)
    stateless = params.local_stack(("image.wrist",), 0, 15.0, 1024, 288, False)
    assert latency.stack_layout({"local_stack": stateless}) == (1, 1024, 288)


def test_observation_is_a_jpeg_request_within_the_handshake_size() -> None:
    obs = latency.encode_images(latency.raw_observation(1, 25), 416, 234)
    data = wire.serialise(obs)
    decoded, stats = wire.deserialise_request(data)
    assert decoded["image.exterior"].shape == (25, 234, 416, 3)
    assert decoded["robot_state.ee_pose"].shape == (25, 7)
    assert decoded["task"] == latency.PROMPT
    assert stats.codecs == {"jpeg"} and stats.jpeg_quality == 90
    again = latency.encode_images(latency.raw_observation(1, 25), 416, 234)
    assert data == wire.serialise(again)
    # A bound larger than the frame keeps its size: no upscale.
    big = wire.deserialise(
        wire.serialise(latency.encode_images(latency.raw_observation(0, 1), 1024, 288))
    )
    assert big["image.wrist"].shape == (1, 288, 512, 3)
    assert latency.raw_observation(0, 1, "Pick up the cube.")["task"] == (
        "Pick up the cube."
    )


def test_run_against_a_local_server(
    server: tuple[str, InferenceServer],  # noqa: F811
) -> None:
    addr, _ = server
    url = f"ws://{addr}/api/v1/session?sample_seed=0"
    meta, records = latency.run(url, {}, sessions=2, calls=3, warmup_calls=1)
    assert meta["checkpoint"] == "/ckpt/droid"
    assert [(r["session"], r["call"], r["warmup"]) for r in records] == [
        (0, 0, True),
        (0, 1, False),
        (0, 2, False),
        (1, 0, False),
        (1, 1, False),
        (1, 2, False),
    ]
    assert [r["rows"] for r in records] == [21, 24, 24, 21, 24, 24]
    assert {r["effective_prompt"] for r in records} == {latency.PROMPT}
    for r in records:
        assert r["round_trip_ms"] >= r["policy_ms"] >= 0.0
        assert r["other_ms"] == pytest.approx(
            r["round_trip_ms"] - r["deserialize_ms"] - r["policy_ms"]
        )
    summary = latency.summarize_calls(records)
    assert summary["first"]["round_trip_ms"]["n"] == 1
    assert summary["later"]["policy_ms"]["n"] == 4


def test_a_failed_call_keeps_the_records_before_it(
    server: tuple[str, InferenceServer],  # noqa: F811
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    addr, _ = server
    url = f"ws://{addr}/api/v1/session"
    query = "sample_seed=0&max_chunks_per_episode=2"
    meta, records = latency.run(
        f"{url}?{query}", {}, sessions=2, calls=4, warmup_calls=1
    )
    assert meta["checkpoint"] == "/ckpt/droid"
    assert [(r["session"], r["call"], "error" in r) for r in records] == [
        (0, 0, False),
        (0, 1, False),
        (0, 2, True),
    ]
    assert latency.summarize_calls(records)["later"]["round_trip_ms"]["n"] == 1

    out = tmp_path / "latency.jsonl"
    argv = ["--url", url, "--query", query, "--sessions", "2", "--calls", "4"]
    argv += ["--warmup-calls", "1", "--out", str(out)]
    with pytest.raises(SystemExit, match="stopped at session 0 call 2"):
        latency.main(argv)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["call"] for r in lines] == [0, 1, 2] and "error" in lines[-1]
    assert "round_trip_ms" in capsys.readouterr().out  # the summary is printed


def test_a_refused_session_writes_out_and_exits(
    server: tuple[str, InferenceServer],  # noqa: F811
    tmp_path,
) -> None:
    addr, _ = server
    out = tmp_path / "latency.jsonl"
    argv = ["--url", f"ws://{addr}/api/v1/session/nope", "--out", str(out)]
    with pytest.raises(SystemExit, match="refused the session"):
        latency.main(argv)
    (rec,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert rec["session"] == 0 and rec["call"] == 0
    assert "refused the session" in rec["error"]


def test_a_client_bug_is_not_recorded_as_a_failed_call(
    server: tuple[str, InferenceServer],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addr, _ = server

    def broken(*args: object) -> dict:
        raise AssertionError("client bug")

    monkeypatch.setattr(latency, "raw_observation", broken)
    url = f"ws://{addr}/api/v1/session?sample_seed=0"
    with pytest.raises(AssertionError, match="client bug"):
        latency.run(url, {}, sessions=1, calls=2, warmup_calls=0)


def test_an_unreachable_server_is_recorded() -> None:
    meta, records = latency.run(
        "ws://127.0.0.1:9/api/v1/session", {}, 1, 2, 0, timeout=5.0
    )
    assert meta == {}
    (rec,) = records
    assert rec["call"] == 0 and rec["session"] == 0 and rec["error"]
