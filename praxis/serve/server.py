"""WebSocket policy server of praxis. It speaks the Positronic offboard protocol.

A chunked session speaks protocol v1. A real-time session (``schedule=rtc``,
the periodic flow) speaks protocol v2, which the periodic recipe of the
client needs.

Endpoints:

* ``ws://<host>:<port>/api/v1/session`` and ``/api/v1/session/<model_id>``:
  one robot session per connection. The query holds the session parameters
  (:mod:`praxis.serve.params`).
* ``GET /api/v1/models``: ``{"models": ["default"]}``.

All frames are binary msgpack frames (:mod:`praxis.serve.wire`). A session
goes through these steps:

1. The server sends ``{"status": "loading", "message": "Loading policy..."}``.
2. The server sends ``{"status": "ready", "meta": {...}}``. The meta holds the
   session settings and the rig-side pipeline (``local_stack``).
3. For each observation, the server sends ``{"result": [rows],
   "effective_prompt": str, "prompt_original": None, "server_timing":
   {...}}``. The prompt goes to the model verbatim, so ``prompt_original``
   (the prompt before a rewrite) is always None.

Protocol v2 (a real-time session) differs in these steps:

* The ready message also holds ``"protocol_version": 2``, a ``session_id``
  and ``max_in_flight``, the calls that the client may have in flight
  (https://github.com/Positronic-Robotics/positronic/pull/1153).
* Each request is ``{"session_id": id, "observation": {...}}``. The request
  ``{"session_id": id, "end_session": True}`` ends the session. The server
  answers it with the same message after the context is dropped.
* ``result`` is None for a sample that does not predict. Each answer also
  holds ``timing``, the served, decode, inference and queue times in ms.
* An error answer holds ``"status": "error"``, so the client closes too.

Errors:

* A bad session parameter or an unknown model: ``{"status": "error",
  "error": msg}``, then the close code 1008.
* A failed policy call: ``{"error": msg}``, then the close code 1011.
* ``max_chunks_per_episode`` reached: ``{"error": msg}``. The server stops
  reading the session.

All policy calls of all sessions run one after the other on one worker
thread. The event loop stays free for the pings of the clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket

from praxis.layers import Attention
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline
from praxis.serve import params
from praxis.serve.policy import (
    ChunkBudgetExhausted,
    PolicyFactory,
    ServedPolicy,
    warmup_obs,
)
from praxis.serve.wire import deserialise_request, serialise

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class InferenceServer:
    """The FastAPI app of the policy server.

    ``registry`` maps a model id to its policy factory. The first entry is
    the default model. ``executor`` runs the policy calls.
    """

    def __init__(
        self, registry: dict[str, PolicyFactory], executor: ThreadPoolExecutor
    ) -> None:
        if not registry:
            raise ValueError("the registry must hold at least one policy")
        self.registry = registry
        self.default_key = next(iter(registry))
        self.executor = executor
        self.app = FastAPI()
        self.app.get("/api/v1/models")(self.get_models)
        self.app.websocket("/api/v1/session")(self.session)
        self.app.websocket("/api/v1/session/{model_id:path}")(self.session)

    async def get_models(self) -> dict:
        return {"models": list(self.registry)}

    async def session(self, websocket: WebSocket, model_id: str | None = None) -> None:
        """Serve one robot session."""
        await websocket.accept()
        try:
            overrides = params.parse_session_params(websocket.query_params)
        except ValueError as exc:
            await _refuse(websocket, str(exc))
            return
        rollout_id, source = params.resolve_rollout_id(
            websocket.query_params.get("rollout_id")
        )
        logger.info(
            "session open: client=%s model=%s rollout_id=%s overrides=%s",
            websocket.client,
            model_id or self.default_key,
            rollout_id,
            overrides,
        )
        factory = self.registry.get(model_id or self.default_key)
        if factory is None:
            await _refuse(
                websocket,
                f"Unknown model {model_id!r}; this endpoint serves "
                f"{sorted(self.registry)} (see GET /api/v1/models).",
            )
            return
        policy: ServedPolicy | None = None
        try:
            await websocket.send_bytes(
                serialise({"status": "loading", "message": "Loading policy..."})
            )
            try:
                policy = factory(overrides)
                meta = {
                    **policy.meta,
                    "rollout_id": rollout_id,
                    "rollout_id_source": source,
                    "session_overrides": {k: str(v) for k, v in overrides.items()},
                }
                ready_msg: dict[str, object] = {"status": "ready", "meta": meta}
                session_id = None
                if policy.protocol_version == 2:
                    session_id = secrets.token_hex(16)
                    ready_msg["protocol_version"] = 2
                    ready_msg["session_id"] = session_id
                    ready_msg["max_in_flight"] = policy.max_in_flight
                # A session value that msgpack cannot pack fails here.
                ready = serialise(ready_msg)
            except Exception as exc:  # any bad session value refuses the session
                logger.warning("session refused: %s", exc)
                await _refuse(websocket, str(exc))
                return
            await websocket.send_bytes(ready)
            await self._calls(websocket, policy, rollout_id, session_id)
        except Exception as exc:  # the client left, or a send failed
            logger.info("session closed: %s", exc)
        finally:
            if policy is not None:
                policy.close()

    async def _calls(
        self,
        websocket: WebSocket,
        policy: ServedPolicy,
        rollout_id: str,
        session_id: str | None = None,
    ) -> None:
        """Answer the observations of one session until it ends.

        ``session_id`` is the id of a protocol v2 session, None for v1.
        """
        loop = asyncio.get_running_loop()
        v2 = session_id is not None
        n_calls = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data is None:
                await _refuse(
                    websocket,
                    "Text frames are not served. Send each observation as one "
                    "binary msgpack frame.",
                )
                return
            t_recv = time.monotonic()
            recv_iso = _iso_now()
            try:
                obs, stats = deserialise_request(data)
                if v2:
                    if not isinstance(obs, dict) or obs.get("session_id") != session_id:
                        raise ValueError(
                            "The session ID does not belong to this connection"
                        )
                    if obs.get("end_session") is True:
                        await loop.run_in_executor(self.executor, policy.close)
                        await websocket.send_bytes(
                            serialise({"session_id": session_id, "end_session": True})
                        )
                        return
                    obs = obs.get("observation")
                    if not isinstance(obs, dict):
                        raise ValueError("a v2 request carries an observation dict")
                t_des = time.monotonic()
                codec = "+".join(sorted(stats.codecs)) if stats.codecs else "raw"
                timing: dict[str, object] = {
                    "request_bytes": len(data),
                    "deserialize_ms": round((t_des - t_recv) * 1e3, 1),
                    "jpeg_decode_ms": round(stats.jpeg_ms, 1),
                    "video_decode_ms": round(stats.video_ms, 1),
                    "image_codec": codec,
                    "wire_jpeg_quality_est": stats.jpeg_quality,
                    "recv_iso": recv_iso,
                }
                if n_calls == 0:
                    logger.info(
                        "rollout %s: image codec %s, JPEG quality %s, %d bytes",
                        rollout_id,
                        codec,
                        stats.jpeg_quality,
                        len(data),
                    )
                n_calls += 1
                t_queue = time.monotonic()
                rows, t_call = await loop.run_in_executor(
                    self.executor, _timed, policy.select_action, obs
                )
                t_pol = time.monotonic()
                response: dict[str, object] = {
                    "result": rows,
                    "effective_prompt": policy.effective_prompt,
                    "prompt_original": None,
                }
                timing["policy_ms"] = round((t_pol - t_des) * 1e3, 1)
                timing.update(policy.timing)
                timing["send_iso"] = _iso_now()
                if policy.return_timing:
                    response["server_timing"] = timing
                if v2:
                    # The timing block of the protocol: numbers only.
                    response["timing"] = {
                        "served_ms": round((t_pol - t_recv) * 1e3, 1),
                        "decode_ms": round((t_des - t_recv) * 1e3, 1),
                        "infer_ms": round((t_pol - t_des) * 1e3, 1),
                        "queued_ms": round((t_call - t_queue) * 1e3, 1),
                    }
                await websocket.send_bytes(serialise(response))
                if rows is not None:
                    junction = {k: v for k, v in timing.items() if k.startswith("rtc_")}
                    logger.info(
                        "rollout %s call %d: %d rows, policy %.1f ms%s",
                        rollout_id,
                        n_calls,
                        len(rows),
                        timing["policy_ms"],
                        f", {junction}" if junction else "",
                    )
            except ChunkBudgetExhausted as exc:
                logger.info("rollout %s: %s", rollout_id, exc)
                with contextlib.suppress(Exception):
                    await websocket.send_bytes(serialise(_error(str(exc), v2)))
                return
            except Exception as exc:
                # The call can fail after it changed the context, so the
                # session cannot continue. The next session starts cold.
                logger.exception("rollout %s: the policy call failed", rollout_id)
                await websocket.send_bytes(serialise(_error(str(exc), v2)))
                with contextlib.suppress(Exception):
                    await websocket.close(code=1011)
                return


def _timed(
    function: Callable[[dict], list[dict] | None], obs: dict
) -> tuple[list[dict] | None, float]:
    """Run ``function(obs)`` on the worker. Return its result and its start time."""
    t_call = time.monotonic()
    return function(obs), t_call


def _error(reason: str, v2: bool) -> dict[str, object]:
    """Return the error answer of a call. A v2 client closes on ``status`` error."""
    return {"status": "error", "error": reason} if v2 else {"error": reason}


async def _refuse(websocket: WebSocket, reason: str) -> None:
    """Send the error status, then close the session with the code 1008."""
    await websocket.send_bytes(serialise({"status": "error", "error": reason}))
    await websocket.close(code=1008, reason=reason[:100])


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Read a YAML config and apply ``key.sub=value`` overrides (YAML values)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def build_factory(cfg: dict) -> PolicyFactory:
    """Load the model of a serve config and return its policy factory.

    The server serves the single-arm and the bimanual checkpoints.
    ``model.embodiment`` of the config must match the checkpoint, else the
    boot fails.
    """
    m, s = cfg["model"], cfg["sampler"]
    block_c = m.get("vae_norm_block_c")
    pipe = PraxisPipeline(
        checkpoint=m["checkpoint"],
        vae_weights=m["vae_weights"],
        text_encoder=m["text_encoder"],
        device=m.get("device", "cuda"),
        decoder_attention_backend=m.get("decoder_attention_backend"),
        encoder_attention_backend=m.get("encoder_attention_backend"),
        vae_norm_block_c={int(k): int(v) for k, v in block_c.items()}
        if block_c
        else None,
        routed_precision=str(m.get("routed_precision", "bf16")),
        nvfp4_cache=m.get("nvfp4_cache"),
        action_cuda_graphs=bool(m.get("action_cuda_graphs", True)),
        nvfp4_moe_cuda_graphs=bool(m.get("nvfp4_moe_cuda_graphs", True)),
        sampler=SamplerSettings.from_config(s),
    )
    if m.get("embodiment") != pipe.embodiment:
        raise ValueError(
            f"model.embodiment of the serve config is {m.get('embodiment')!r}, but "
            f"the checkpoint {m['checkpoint']} is for the {pipe.embodiment!r} rig. "
            "Serve the single-arm checkpoint with configs/serve_droid.yaml and "
            "the bimanual checkpoint with configs/serve_yam.yaml."
        )
    factory = PolicyFactory(pipe, dict(cfg.get("session") or {}), str(m["checkpoint"]))
    factory({}).close()  # a bad default fails the boot, not each session
    modules = pipe.policy.modules() if isinstance(pipe.policy, torch.nn.Module) else ()
    attention = next((a for a in modules if isinstance(a, Attention)), None)
    logger.info(
        "loaded %s (%s rig), VAE norm BLOCK_C %s, sampler %s, routed experts %s,"
        " attention %s",
        m["checkpoint"],
        pipe.embodiment,
        pipe.vae.norm_block_c,
        pipe.sampler,
        getattr(pipe.policy, "routed_quantization", {}).get("format", "bf16"),
        attention.effective_backend(pipe.device) if attention else None,
    )
    return factory


def warmup(factory: PolicyFactory, executor: ThreadPoolExecutor) -> None:
    """Run one session over the context depths 0 to ``max_context_chunks``.

    The warmup compiles the attention kernels of each depth before the first
    robot call. It uses 2 sampler steps and synthetic observations. Each
    later session starts with a new context. A real-time session samples the
    same window shapes, so the chunked warmup covers it too.
    """
    t0 = time.monotonic()
    depths = int(factory.defaults["max_context_chunks"]) + 1
    policy = factory(
        {
            "schedule": "chunked",
            "context_mode": "observed",
            "num_steps": 2,
            "action_steps": None,
            "max_context_chunks": depths - 1,
        }
    )
    for step in range(depths):
        obs = warmup_obs(step, embodiment=factory.embodiment)
        executor.submit(policy.select_action, obs).result()
    policy.close()
    logger.info("warmup: %d depths in %.1f s", depths, time.monotonic() - t0)


def build_server(cfg: dict) -> InferenceServer:
    """Load the model, run the warmup and return the server.

    Set ``PRAXIS_SKIP_WARMUP=1`` to skip the warmup.
    """
    factory = build_factory(cfg)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="praxis-infer")
    if os.environ.get("PRAXIS_SKIP_WARMUP", "0") != "1":
        warmup(factory, executor)
    return InferenceServer({"default": factory}, executor)


def serve(app: FastAPI, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve ``app`` with uvicorn until the process stops.

    The server pings each client every 120 s, so a long first call does not
    close the connection. A request frame can be up to 256 MiB.
    """
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ws_ping_interval=120.0,
        ws_ping_timeout=300.0,
        ws_max_size=256 * 1024 * 1024,
    )
