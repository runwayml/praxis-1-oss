"""WebSocket policy server of praxis. It speaks the Positronic offboard protocol v1.

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
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket

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
                # A session value that msgpack cannot pack fails here.
                ready = serialise({"status": "ready", "meta": meta})
            except Exception as exc:  # any bad session value refuses the session
                logger.warning("session refused: %s", exc)
                await _refuse(websocket, str(exc))
                return
            await websocket.send_bytes(ready)
            await self._calls(websocket, policy, rollout_id)
        except Exception as exc:  # the client left, or a send failed
            logger.info("session closed: %s", exc)
        finally:
            if policy is not None:
                policy.close()

    async def _calls(
        self, websocket: WebSocket, policy: ServedPolicy, rollout_id: str
    ) -> None:
        """Answer the observations of one session until it ends."""
        loop = asyncio.get_running_loop()
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
                rows = await loop.run_in_executor(
                    self.executor, policy.select_action, obs
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
                await websocket.send_bytes(serialise(response))
                logger.info(
                    "rollout %s call %d: %d rows, policy %.1f ms",
                    rollout_id,
                    n_calls,
                    len(rows),
                    timing["policy_ms"],
                )
            except ChunkBudgetExhausted as exc:
                logger.info("rollout %s: %s", rollout_id, exc)
                with contextlib.suppress(Exception):
                    await websocket.send_bytes(serialise({"error": str(exc)}))
                return
            except Exception as exc:
                # The call can fail after it changed the context, so the
                # session cannot continue. The next session starts cold.
                logger.exception("rollout %s: the policy call failed", rollout_id)
                await websocket.send_bytes(serialise({"error": str(exc)}))
                with contextlib.suppress(Exception):
                    await websocket.close(code=1011)
                return


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

    The server serves the single-arm checkpoint. ``model.embodiment``
    of the config must be ``single_arm`` and match the checkpoint, else the
    boot fails.
    """
    m, s = cfg["model"], cfg["sampler"]
    block_c = m.get("vae_norm_block_c")
    pipe = PraxisPipeline(
        checkpoint=m["checkpoint"],
        vae_weights=m["vae_weights"],
        text_encoder=m["text_encoder"],
        device=m.get("device", "cuda"),
        vae_norm_block_c={int(k): int(v) for k, v in block_c.items()}
        if block_c
        else None,
        sampler=SamplerSettings(
            num_steps=int(s["num_steps"]),
            t_shift=float(s["t_shift"]),
            solver=str(s["solver"]),
            action_steps=s.get("action_steps"),
            layout=str(s.get("layout", "window_joint")),
            cfg_scale=float(s.get("cfg_scale", 1.0)),
            cfg_video_scale=float(s.get("cfg_video_scale", 4.0)),
            cfg_batched=bool(s.get("cfg_batched", False)),
        ),
    )
    if m.get("embodiment") != pipe.embodiment:
        raise ValueError(
            f"model.embodiment of the serve config is {m.get('embodiment')!r}, but "
            f"the checkpoint {m['checkpoint']} is for the {pipe.embodiment!r} rig. "
            "Serve the single-arm checkpoint with configs/serve_droid.yaml."
        )
    factory = PolicyFactory(pipe, dict(cfg.get("session") or {}), str(m["checkpoint"]))
    factory({}).close()  # a bad default fails the boot, not each session
    logger.info(
        "loaded %s (%s rig), VAE norm BLOCK_C %s, sampler %s",
        m["checkpoint"],
        pipe.embodiment,
        pipe.vae.norm_block_c,
        pipe.sampler,
    )
    return factory


def warmup(factory: PolicyFactory, executor: ThreadPoolExecutor) -> None:
    """Run one session over the context depths 0 to ``max_context_chunks``.

    The warmup compiles the attention kernels of each depth before the first
    robot call. It uses 2 sampler steps and synthetic observations. Each
    later session starts with a new context.
    """
    t0 = time.monotonic()
    depths = int(factory.defaults["max_context_chunks"]) + 1
    policy = factory(
        {
            "context_mode": "observed",
            "num_steps": 2,
            "action_steps": None,
            "max_context_chunks": depths - 1,
        }
    )
    for step in range(depths):
        obs = warmup_obs(step)
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
