"""Modal app of the praxis policy server.

The app serves the FastAPI app of ``python -m praxis.serve`` on one Modal GPU,
with the single-arm checkpoint. The Modal edge requires a proxy auth
token on each connection. A client sends the token in the headers
``Modal-Key`` and ``Modal-Secret``.

Deploy with ``modal>=1.2`` and your own Modal credentials (``modal token
set``), from the root of the repository:

    modal volume create praxis-weights
    modal volume put praxis-weights checkpoints/praxis-1-droid /checkpoints/praxis-1-droid
    modal volume put praxis-weights weights /weights
    modal deploy praxis/serve/modal_app.py

The serve config of the single-arm checkpoint is ``configs/serve_droid.yaml``
(the default). Its trunk uses the ``flash`` attention backend; the image
installs ``flash-attn-4`` for it.

Modal builds the image from ``docker/Dockerfile``. The volume holds the
``checkpoints/`` and ``weights/`` directories of the repository layout. The
model paths of the config and ``PRAXIS_CHECKPOINT`` are relative to the root
of the volume. A leading ``/`` also means the root of the volume.

These environment variables of ``modal deploy`` change the deployment:

* ``PRAXIS_CONFIG``: the serve config, a path in the repository. The default
  is ``configs/serve_droid.yaml``.
* ``PRAXIS_CHECKPOINT``: the checkpoint directory in the volume. The default
  is ``model.checkpoint`` of the config, for example a fine-tuned copy of
  the checkpoint. A checkpoint that is not single-arm fails the boot.
* ``PRAXIS_MODAL_APP_NAME``: the app name. The default is ``praxis-serve``.
  Deploy each checkpoint as a separate app.
* ``PRAXIS_MODAL_GPU``: the GPU type. The default is ``H100``. Modal can run
  an ``H100`` request on an ``H200``. ``H100!`` prevents this.
* ``PRAXIS_MODAL_VOLUME``: the volume with the weights. The default is
  ``praxis-weights``.
* ``PRAXIS_MODAL_SCALEDOWN_S``: the idle time in seconds before Modal stops
  the container. The default is 1800.

The app copies all ``PRAXIS_*`` variables into the container, for example
``PRAXIS_SAMPLE_SEED`` and ``PRAXIS_SKIP_WARMUP`` of the server.

The app has no ``@modal.concurrent`` setting. Modal then gives a container
one connection at a time.

The Modal edge accepts WebSocket messages of up to 2 MiB. With the JPEG
quality 90 of the Positronic client, the largest of 128 recorded single-arm
requests was 1.74 MiB (see ``praxis/serve/README.md``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from praxis.serve.params import git_commit

if TYPE_CHECKING:
    from fastapi import FastAPI

# The repository root on the deploy machine.
REPO = Path(__file__).resolve().parent.parent.parent
# The repository in the image (docker/Dockerfile) and the mount point of the
# weights volume.
IMAGE_REPO = Path("/opt/praxis")
WEIGHTS = Path("/data")

# The environment of the container: the PRAXIS_* variables of the deploy. The
# image holds no git history, so the deploy machine sets the commit.
CONTAINER_ENV: dict[str, str | None] = {
    k: v for k, v in os.environ.items() if k.startswith("PRAXIS_")
}
if modal.is_local():
    CONTAINER_ENV.setdefault("PRAXIS_SERVED_COMMIT", git_commit(REPO))

image = modal.Image.from_dockerfile(REPO / "docker" / "Dockerfile", context_dir=REPO)
volume = modal.Volume.from_name(
    os.environ.get("PRAXIS_MODAL_VOLUME", "praxis-weights")
).read_only()
app = modal.App(os.environ.get("PRAXIS_MODAL_APP_NAME", "praxis-serve"), image=image)


@app.cls(
    gpu=os.environ.get("PRAXIS_MODAL_GPU", "H100"),
    # One WebSocket connection is one input, so a session can last 1 hour.
    timeout=3600,
    # The boot (weights from the volume, warmup) has a separate limit.
    startup_timeout=7200,
    scaledown_window=int(os.environ.get("PRAXIS_MODAL_SCALEDOWN_S", "1800")),
    # A client tries to connect again during a cold start. Without this
    # limit, Modal starts a new GPU container for each try.
    max_containers=1,
    volumes={str(WEIGHTS): volume},
    env=CONTAINER_ENV,
)
class PolicyServer:
    @modal.enter()
    def boot(self) -> None:
        """Load the model once and run the warmup."""
        from praxis.serve.server import build_server, load_config

        logging.basicConfig(level=logging.INFO, force=True)
        config = os.environ.get("PRAXIS_CONFIG", "configs/serve_droid.yaml")
        cfg = load_config(IMAGE_REPO / config)
        model = cfg["model"]
        model["checkpoint"] = os.environ.get("PRAXIS_CHECKPOINT") or model["checkpoint"]
        for key in ("checkpoint", "vae_weights", "text_encoder"):
            # A path in the volume: a leading "/" is the root of the volume.
            model[key] = str(WEIGHTS / str(model[key]).lstrip("/"))
        self.server = build_server(cfg)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self) -> FastAPI:
        return self.server.app
