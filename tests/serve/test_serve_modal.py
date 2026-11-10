"""Tests of the Modal app (CPU only, no weights, no Modal credentials).

The tests import praxis/serve/modal_app.py and run the boot of the app in
this process with ``web.local()``. A stand-in replaces the networks of the
pipeline. Nothing connects to Modal.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

# The Modal tests need the [serve] extra.
pytest.importorskip("modal")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import uvicorn
from fastapi import FastAPI
from test_serve_protocol import (
    SINGLE_ARM,
    FakePipeline,
    _droid_obs,
    _free_port,
    _open,
    _recv,
)
from websockets.sync.client import connect

from praxis.serve import modal_app, server, wire

ROOT = Path(__file__).resolve().parents[2]


def test_import_defines_the_app_and_needs_no_credentials() -> None:
    assert modal_app.app.name == "praxis-serve"
    assert modal_app.CONTAINER_ENV["PRAXIS_SERVED_COMMIT"]
    assert all(k.startswith("PRAXIS_") for k in modal_app.CONTAINER_ENV)


def test_the_image_is_the_example_dockerfile() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    assert "WORKDIR " + str(modal_app.IMAGE_REPO) in dockerfile
    assert '"python", "-m", "praxis.serve"' in dockerfile
    rules = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rules[0] == "*"
    assert {"!pyproject.toml", "!README.md", "!praxis", "!configs"} <= set(rules)


def test_the_image_has_the_flash_attention_package() -> None:
    """The single-arm checkpoint asks for the flash attention backend.

    Its first policy call fails without ``flash-attn-4``. The install resolves
    the torch dependencies again, so it must come before the cuDNN pin.
    """
    lines = (ROOT / "docker" / "Dockerfile").read_text().splitlines()
    runs = [i for i, line in enumerate(lines) if line.startswith("RUN ")]
    flash = [i for i in runs if "flash-attn-4" in lines[i]]
    cudnn = [i for i in runs if "nvidia-cudnn-cu13==" in lines[i]]
    assert len(flash) == 1 and len(cudnn) == 1
    assert "flash-attn-4[cu13]==" in lines[flash[0]]
    assert flash[0] < cudnn[0]


def test_web_local_serves_the_server_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``web.local()`` runs the boot and returns the app of ``python -m praxis.serve``."""
    paths: dict[str, str] = {}

    def pipeline(**kwargs: object) -> FakePipeline:
        paths.update(
            {k: str(kwargs[k]) for k in ("checkpoint", "vae_weights", "text_encoder")}
        )
        return FakePipeline(SINGLE_ARM, kwargs["sampler"])  # ty: ignore[invalid-argument-type]

    # The serve config of the image, on the CPU.
    cfg = server.load_config(
        ROOT / "configs" / "serve_droid.yaml", ["model.device=cpu"]
    )
    config = tmp_path / "configs" / "serve_test.yaml"
    config.parent.mkdir()
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(server, "PraxisPipeline", pipeline)
    monkeypatch.setattr(modal_app, "IMAGE_REPO", tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG", "configs/serve_test.yaml")
    # A leading "/" is the root of the volume, as in "modal volume put".
    monkeypatch.setenv("PRAXIS_CHECKPOINT", "/checkpoints/other")
    monkeypatch.setenv("PRAXIS_SKIP_WARMUP", "1")

    app = modal_app.PolicyServer().web.local()
    assert type(app) is FastAPI
    assert paths == {
        "checkpoint": "/data/checkpoints/other",
        "vae_weights": "/data/weights/Wan2.2_VAE.pth",
        "text_encoder": "/data/weights/Qwen3.5-2B",
    }

    port = _free_port()
    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    try:
        addr = f"127.0.0.1:{port}"
        ws, ready = _open(addr, "/api/v1/session?sample_seed=0&schedule=chunked")
        with ws:
            assert ready["meta"]["checkpoint"] == "/data/checkpoints/other"
            ws.send(wire.serialise(_droid_obs(1)))
            assert len(_recv(ws)["result"]) == 21
        with connect(f"ws://{addr}/api/v1/session/nope", open_timeout=10) as ws:
            assert _recv(ws)["status"] == "error"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
