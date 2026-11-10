"""Measure the end-to-end latency of a running praxis policy server.

This client times what a robot sees: the request, the network and the
server. To time the model alone on a local GPU, with no server, use
``scripts/benchmark_inference.py``.

Usage:
    python -m praxis.serve.latency --url ws://localhost:8000/api/v1/session
    python -m praxis.serve.latency \\
        --url wss://<workspace>--praxis-serve-policyserver-web.modal.run/api/v1/session \\
        --headers-file ~/.config/praxis/modal_headers.json --out outputs/latency.jsonl

The client sends the calls of a Positronic client on the chunk grid, one after
the other with no robot time between them. It runs the chunked schedule: the
default ``--query`` holds ``schedule=chunked``, so a deploy whose config sets
real-time chunking (periodic samples on protocol v2) is timed chunked too.
Each session sends a first observation with one frame, then observations with
the frame stack of the handshake (``local_stack``). Each view is scaled down to
fit the ``restrict_image_size`` of the handshake and goes as JPEG frames of
quality 90. The frames are synthetic (a seeded noise pattern): a full request of
2 x 25 frames at 416x234 is about 1.2 MB, in the range of the recorded client
requests (see "Request size on Modal" in ``praxis/serve/README.md``).

Each call gives these times, in milliseconds:

* ``encode_ms``: the JPEG encode and the msgpack pack of the request. The
  synthetic frames are made before this timer starts.
* ``round_trip_ms``: from the send of the request to the decoded answer.
* ``deserialize_ms``: the request decode of the server, with
  ``jpeg_decode_ms`` in it (from the ``server_timing`` block of the answer).
* ``policy_ms``: the server time around the policy call: the wait for its
  worker thread, the canvas and the VAE encode, the text encode of a new
  prompt, the sampler and the command decode. It is wall time, not GPU time.
* ``other_ms``: ``round_trip_ms - deserialize_ms - policy_ms``: the network,
  the answer encode and decode, and the work outside the server timers.

The connection and the handshake are not timed. A refused session or a
failed call stops the run: the client still writes the records so far to
``--out`` and prints their summary, then exits with the error. The first ``--warmup-calls``
calls of the run are not counted; the default is the first session, so the
compile of each context depth stays out of the numbers. The summary gives the median (p50), the
p90 and the max of each time, for the first call of each session and for
the later calls. With the observed context of the serve config these give
21 and 24 command rows.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from praxis.serve import wire
from praxis.serve.policy import warmup_obs

STAGES: tuple[str, ...] = (
    "encode_ms",
    "round_trip_ms",
    "deserialize_ms",
    "jpeg_decode_ms",
    "policy_ms",
    "other_ms",
)
IMAGE_KEYS: tuple[str, ...] = ("image.exterior", "image.wrist")
PROMPT = "Put the red block in the bowl."


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Return the count, the median, the p90 and the max of ``values``."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": float(x.max()),
    }


def summarize_calls(calls: Sequence[dict]) -> dict[str, dict[str, dict]]:
    """Summarize each stage of the counted calls, per group (first, later)."""
    counted = [c for c in calls if not (c["warmup"] or "error" in c)]
    groups = {
        "first": [c for c in counted if c["call"] == 0],
        "later": [c for c in counted if c["call"] > 0],
    }
    return {
        name: {s: summarize(c[s] for c in group if s in c) for s in STAGES}
        for name, group in groups.items()
    }


def stack_layout(meta: dict) -> tuple[int, int, int]:
    """Return the frames of a full stack and the image width and height of the handshake."""
    frames, width, height = 1, 0, 0
    for entry in meta["local_stack"]["seq"]:
        args = entry.get("args", {})
        if entry["name"] == "temporal_stack":
            frames = len(args["offsets_sec"])
        elif entry["name"] == "restrict_image_size":
            width, height = int(args["width"]), int(args["height"])
    return frames, width, height


def _jpeg(frames: np.ndarray, width: int, height: int, quality: int) -> dict:
    bufs = []
    for frame in frames:
        im = Image.fromarray(frame)
        # restrict_image_size bounds the view: scale down, keep the aspect.
        scale = min(1.0, width / im.width, height / im.height) if width else 1.0
        if scale < 1.0:
            size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(size, Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        bufs.append(buf.getvalue())
    return {b"__jpeg__": True, b"frames": bufs, b"ndim": 4}


def raw_observation(step: int, frames: int, prompt: str | None = None) -> dict:
    """Return the synthetic observation of call ``step``: ``frames`` frames per view.

    The task is ``prompt`` (the prompt of the session), else :data:`PROMPT`.
    """
    obs = warmup_obs(step, stack=frames)
    rng = np.random.default_rng(step)
    for key in IMAGE_KEYS:
        noisy = obs[key] + rng.normal(0.0, 12.0, obs[key].shape)
        obs[key] = np.clip(noisy, 0, 255).astype(np.uint8)
    obs["task"] = prompt or PROMPT
    return obs


def encode_images(obs: dict, width: int, height: int, quality: int = 90) -> dict:
    """Return ``obs`` with each view as JPEG frames within ``width`` x ``height``."""
    return {
        k: _jpeg(v, width, height, quality) if k in IMAGE_KEYS else v
        for k, v in obs.items()
    }


class ServerError(RuntimeError):
    """The server refused the session or answered a call with an error."""


def _recv(ws: Connection, timeout: float) -> dict:
    return wire.deserialise(ws.recv(timeout=timeout))


def timing_record(round_trip_ms: float, timing: dict | None) -> dict[str, float]:
    """Return the server times of one answer and the rest of its round trip.

    Other servers can send a part of the ``server_timing`` block (for example
    no ``jpeg_decode_ms``). The missing keys stay out of the record, and the
    summary skips them.
    """
    rec = {
        key: float(timing[key])
        for key in ("deserialize_ms", "jpeg_decode_ms", "policy_ms")
        if timing and timing.get(key) is not None
    }
    if "deserialize_ms" in rec and "policy_ms" in rec:
        rec["other_ms"] = round_trip_ms - rec["deserialize_ms"] - rec["policy_ms"]
    return rec


def run_session(
    url: str, headers: dict[str, str], calls: int, timeout: float = 300.0
) -> tuple[dict, list[dict]]:
    """Run one session of ``calls`` calls. Return the handshake meta and the call records.

    A refused session, a failed call or a lost connection ends the session.
    Its record holds the call index and the ``error``, and the records before
    it are kept. Other errors (a bug of the client) propagate.
    """
    meta: dict = {}
    records: list[dict] = []
    k = 0
    try:
        with connect(
            url,
            additional_headers=headers,
            open_timeout=timeout,
            max_size=None,
            compression=None,
        ) as ws:
            while True:
                status = _recv(ws, timeout)
                if status.get("status") == "error":
                    raise ServerError(f"the server refused the session: {status}")
                if status.get("status") == "ready":
                    meta = status["meta"]
                    break
            stack, width, height = stack_layout(meta)
            for k in range(calls):
                obs = raw_observation(k, 1 if k == 0 else stack, meta.get("prompt"))
                t0 = time.monotonic()
                data = wire.serialise(encode_images(obs, width, height))
                t1 = time.monotonic()
                ws.send(data)
                answer = _recv(ws, timeout)
                t2 = time.monotonic()
                if "result" not in answer:
                    raise ServerError(f"call {k} failed: {answer}")
                rec: dict = {
                    "call": k,
                    "rows": len(answer["result"]),
                    "request_bytes": len(data),
                    "effective_prompt": answer.get("effective_prompt"),
                    "encode_ms": (t1 - t0) * 1e3,
                    "round_trip_ms": (t2 - t1) * 1e3,
                }
                timing = answer.get("server_timing")
                rec.update(timing_record(rec["round_trip_ms"], timing))
                records.append(rec)
    except (ServerError, WebSocketException, OSError) as exc:
        records.append({"call": k, "error": f"{type(exc).__name__}: {exc}"})
    return meta, records


def run(
    url: str,
    headers: dict[str, str],
    sessions: int,
    calls: int,
    warmup_calls: int,
    timeout: float = 300.0,
) -> tuple[dict, list[dict]]:
    """Run ``sessions`` sessions. Return the first handshake meta and all call records.

    The run stops at the first failed call (a record with an ``error``) and
    returns the records up to it.
    """
    meta: dict = {}
    records: list[dict] = []
    for s in range(sessions):
        m, recs = run_session(url, headers, calls, timeout)
        meta = meta or m
        for rec in recs:
            rec["session"] = s
            rec["warmup"] = len(records) < warmup_calls
            records.append(rec)
        if records and "error" in records[-1]:
            break
    return meta, records


def format_summary(summary: dict[str, dict[str, dict]]) -> str:
    """Return the summary as a text table, in milliseconds."""
    lines = [f"{'calls':<7}{'stage':<16}{'n':>5}{'p50':>9}{'p90':>9}{'max':>9}"]
    for group, stages in summary.items():
        for stage, s in stages.items():
            if not s.get("n"):
                continue
            lines.append(
                f"{group:<7}{stage:<16}{s['n']:>5}"
                f"{s['p50']:>9.1f}{s['p90']:>9.1f}{s['max']:>9.1f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--url", required=True, help="ws:// or wss:// session URL")
    ap.add_argument(
        "--query",
        default="sample_seed=0&schedule=chunked",
        help="session parameters, for example 'sample_seed=0&cfg_video_scale=1.0'",
    )
    ap.add_argument(
        "--headers-file",
        type=Path,
        help="JSON file with the headers, for example the Modal-Key and Modal-Secret",
    )
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--calls", type=int, default=8, help="calls per session")
    ap.add_argument(
        "--warmup-calls",
        type=int,
        default=None,
        help="first calls that are not counted (default: one session)",
    )
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per wait")
    ap.add_argument("--out", type=Path, help="write one JSON line per call")
    args = ap.parse_args(argv)
    headers = json.loads(args.headers_file.read_text()) if args.headers_file else {}
    url = args.url + (
        ("&" if "?" in args.url else "?") + args.query if args.query else ""
    )
    # One session warms each context depth of the server, as in
    # scripts/benchmark_inference.py (a server can skip its startup warm-up).
    warmup = args.calls if args.warmup_calls is None else args.warmup_calls
    if warmup >= args.sessions * args.calls:
        ap.error("--warmup-calls leaves no counted call")
    meta, records = run(url, headers, args.sessions, args.calls, warmup, args.timeout)
    settings = ("checkpoint", "code_version", "num_steps", "cfg_scale")
    settings += ("cfg_video_scale", "t_shift", "max_context_chunks", "exec_horizon")
    print("server:", {k: meta.get(k) for k in settings})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(format_summary(summarize_calls(records)))
    failed = [r for r in records if "error" in r]
    if failed:
        r = failed[0]
        raise SystemExit(
            f"stopped at session {r['session']} call {r['call']}: {r['error']}"
        )


if __name__ == "__main__":
    main()
