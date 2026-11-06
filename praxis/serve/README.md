# praxis policy server

`praxis/serve/` is a policy server for Positronic robot clients. It serves
the single-arm checkpoint `praxis-1-droid` (`configs/serve_droid.yaml`) and
the bimanual checkpoint `praxis-1-yam` (`configs/serve_yam.yaml`). It speaks the
Positronic offboard protocol over a WebSocket. Its paths, session
parameters, handshake and message formats follow that protocol. A
Positronic client changes only its policy address.

## Serving

The server is a separate module. Only its own files and its tests in
`tests/serve/` import it. The core package needs none of its packages. The
`[serve]` extra installs them:

```bash
pip install -e ".[serve,kernels]"
```

Install PyTorch and torchvision for your CUDA version first, as the install
section of the top README tells. Then serve the checkpoint on one GPU:

- Local machine:
  `python -m praxis.serve --config configs/serve_droid.yaml --host 0.0.0.0 --port 8000`.
  The client URI is `ws://<host>:8000/api/v1/session`.
- Docker: build `docker/Dockerfile`, then mount the weights at `/weights`
  and publish port 8000. The client URI is the same as on a local machine.
- Modal, with your own Modal credentials: put the weights into a Modal
  volume, then `modal deploy praxis/serve/modal_app.py`. The client URI is
  `wss://<workspace>--praxis-serve-policyserver-web.modal.run/api/v1/session`.
  The client also sends a proxy auth token.

The sections below give the steps for each host and the client settings.

## Run the server on a local machine

Set the weight paths in `configs/serve_droid.yaml`. Then start the server:

```bash
python -m praxis.serve --config configs/serve_droid.yaml --host 0.0.0.0 --port 8000
```

The server loads the model and runs a warmup over the context depths 0 to
`max_context_chunks` of the config (0 to 1). Then it accepts connections.
Set `PRAXIS_SKIP_WARMUP=1` to skip the warmup.

The checkpoint uses the `flash` attention backend. It needs the
`flash-attn-4` package. The section "Get the weights" of the top README
tells how to use the `triton` backend instead.

`model.embodiment` of the serve config (`single_arm` or `bimanual`) names
the rig of the checkpoint. A checkpoint of another rig fails at the boot.

The serve config holds the current recommended sampling settings. These
settings can change when we have more robot results:

| Setting | `serve_droid.yaml` |
|---|---|
| `num_steps` | 2 |
| `cfg_scale` | 1.0 |
| `cfg_video_scale` | 4.0 |
| `t_shift` | 12.5 |
| `layout` | `window_joint` |
| `context_mode` | `observed` |
| `max_context_chunks` | 1 |
| `carry_mode` | `clean` |
| `exec_horizon` | 24 |
| `control_mode` | `droid_impedance` |

A client can change each value for one session in the query of the session
URL, for example `?max_context_chunks=3`. `layout` is the only exception: it
is a boot setting of the server.

## Run the server in Docker

`docker/Dockerfile` is an example image. It holds CUDA 13.0, PyTorch 2.13,
praxis with the `[serve]` and `[kernels]` extras, and `flash-attn-4` for the
`flash` attention backend of the single-arm checkpoint. The image
holds no weights. Build the image from the repository root:

```bash
docker build -f docker/Dockerfile -t praxis-serve .
```

The build compiles `causal-conv1d` from source. This step takes about
9 minutes on 8 CPU cores and needs about 8 GB of memory.

Put the checkpoint directory, `Wan2.2_VAE.pth` and `Qwen3.5-2B/` into one
directory. Mount that directory at `/weights`:

```bash
docker run --gpus all -p 8000:8000 -v /path/to/weights:/weights praxis-serve
```

The default command serves the checkpoint
`/weights/praxis-1-droid` with `configs/serve_droid.yaml`.

## Run the server on Modal

`praxis/serve/modal_app.py` serves the same app on one Modal GPU (default
H100). Modal builds the image from `docker/Dockerfile`. Use `modal>=1.2`
and your own Modal account. Do these steps from the repository root:

1. Set your Modal credentials:

   ```bash
   modal token set --token-id <id> --token-secret <secret>
   ```

2. Put the weights into a Modal volume. The volume holds `checkpoints/` and
   `weights/` as in the repository layout:

   ```bash
   modal volume create praxis-weights
   modal volume put praxis-weights checkpoints/praxis-1-droid /checkpoints/praxis-1-droid
   modal volume put praxis-weights weights /weights
   ```

3. Deploy the app:

   ```bash
   modal deploy praxis/serve/modal_app.py
   ```

4. Make a proxy auth token in the Modal dashboard (Settings, Proxy Auth
   Tokens). The Modal edge refuses each connection without this token.

The output of `modal deploy` gives the URL of the app, for example
`https://<workspace>--praxis-serve-policyserver-web.modal.run`. Each
container holds one session at a time. The app stops the container after
30 minutes without a session. The first connection after that waits for a
new container. The boot then loads the weights and runs the warmup.

These environment variables of `modal deploy` change the deployment:

| Variable | Default | Effect |
|---|---|---|
| `PRAXIS_CONFIG` | `configs/serve_droid.yaml` | Serve config. |
| `PRAXIS_CHECKPOINT` | `model.checkpoint` of the config | Checkpoint directory. |
| `PRAXIS_MODAL_APP_NAME` | `praxis-serve` | App name. Deploy each checkpoint as a separate app. |
| `PRAXIS_MODAL_GPU` | `H100` | GPU type. |
| `PRAXIS_MODAL_VOLUME` | `praxis-weights` | Volume with the weights. |
| `PRAXIS_MODAL_SCALEDOWN_S` | `1800` | Seconds without a session before the container stops. |

The model paths of the config and `PRAXIS_CHECKPOINT` are paths in the
volume, relative to the root of the volume. `/checkpoints/x` and
`checkpoints/x` are the same path.

The app copies all `PRAXIS_*` variables into the container, for example
`PRAXIS_SKIP_WARMUP` of the server.

A fine-tuned checkpoint as a second app (upload its directory first):

```bash
PRAXIS_CHECKPOINT=checkpoints/my-finetune PRAXIS_MODAL_APP_NAME=praxis-serve-finetune \
    modal deploy praxis/serve/modal_app.py
```

### Request size on Modal

The Modal edge accepts WebSocket messages of up to 2 MiB. We measured the
requests of the Positronic client on 32 recorded episodes. Each episode
has 4 calls, and the client used its default JPEG quality 90. The largest
request was 1.74 MiB. No request of the 128 was above 2 MiB.

### Measure the latency

`praxis.serve.latency` runs sessions against a running server, like a robot
client on the chunk grid, and times each call. It works against a local
server and against a Modal app:

```bash
python -m praxis.serve.latency --url ws://<host>:8000/api/v1/session
python -m praxis.serve.latency \
    --url wss://<workspace>--praxis-serve-policyserver-web.modal.run/api/v1/session \
    --headers-file ~/.config/praxis/modal_headers.json --out outputs/latency.jsonl
```

The headers file holds the proxy auth token:
`{"Modal-Key": "<id>", "Modal-Secret": "<secret>"}`. Keep it outside the
checkout and readable only by you (`chmod 600`); `.gitignore` also skips
`modal_headers.json` in case a copy lands in the repository. The client sends JPEG
frames at the image size of the handshake. It reports p50, p90 and max of
the client encode, the round trip, the request decode and the policy call of
the server (`server_timing`), and the rest (network and answer). The first
session of the run is not counted: it warms each context depth of the
server, also when the server skipped its startup warm-up. To time the model alone on a local GPU, use
`scripts/benchmark_inference.py` (top README, "Profile inference").

## Connect a Positronic client

The server declares the rig-side pipeline in its handshake. The client then
builds the frame stack, resizes the camera frames and sends them as JPEG.
Change only the policy address of the client:

- Local server: `ws://<host>:8000/api/v1/session`. Use the `websocket` wire
  with `--policy.address.host=<host> --policy.address.port=8000`.
- Modal: `wss://<workspace>--praxis-serve-policyserver-web.modal.run/api/v1/session`.
  Use the `websocket_tls` wire with port 443, and send the proxy auth token
  in the headers `Modal-Key` and `Modal-Secret`. With `.file_authed_remote`,
  `--policy.headers.path` names a JSON file with these two headers.

The query of the URL holds the session parameters, for example
`?sample_seed=0&rollout_id=run1`. Without `sample_seed`, each session draws a
random seed.

## Action clamp

Before the server sends the commands, it clips the predicted actions to the
training range. First it clips the normalized xyz and grip values to
[-1, 1]. Then it clips the grippers to [0, 1]. With `smoothing=savgol`, the
grip clip comes after the filter, and the xyz gets no second clip. For
normalized values in [-1, 1] and no smoothing, the commands of the
checkpoint do not change. With `smoothing=savgol`, the grip clip can change
the grips of these values too, because the filter can go past 0 and 1. The
section "Action clamp" of the top README gives the steps.

## Determinism

With the same requests and the same seed, the server gives the same answer
bits. Three conditions apply:

- The fused VAE norm kernel uses the block size of `model.vae_norm_block_c`.
- The text encoder kernels of `flash-linear-attention` use fixed kernel
  settings (`FLA_KERNEL_CONFIGS` in `praxis/text_encoder.py`). Without fixed
  settings, each new process picks them by timing, and some picks change the
  output bits.
- The packages of the parity path have the versions of `docker/Dockerfile`.
  Example: scipy 1.18 gives other bits than scipy 1.15 for the smoothed
  actions of `smoothing=savgol`.

## Delete the server

A release without the server deletes these files and blocks:

- `praxis/serve/`, `docker/`, `.dockerignore`, `configs/serve_*.yaml` and
  `tests/serve/`.
- The `[serve]` extra in `pyproject.toml`.

`tests/serve/test_serve_isolation.py` makes sure that the rest of the code
does not import the server and does not need its packages.
