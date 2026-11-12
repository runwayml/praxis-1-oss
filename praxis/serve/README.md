# praxis policy server

`praxis/serve/` is a policy server for Positronic robot clients. It serves
the single-arm checkpoint `praxis-1-droid` (`configs/serve_droid.yaml`), the
bimanual checkpoint `praxis-1-yam` (`configs/serve_yam.yaml`) and an SO-100 /
SO-101 checkpoint (`configs/serve_so101.yaml`, see "Serve an SO-100 / SO-101
checkpoint"). It speaks the Positronic offboard protocol over a WebSocket.
Its paths, session parameters, handshake and message formats follow that
protocol. A Positronic client changes only its policy address.

## Serving

The server is a separate module. Only its own files and its tests in
`tests/serve/` import it. The core package needs none of its packages. The
`[serve]` extra installs them:

```bash
uv pip install -e ".[serve,kernels]"
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

The handshake includes `action_fps`, the checkpoint's action grid rate in Hz.
Each image marker holds 1–25 frames; a 3D JPEG marker holds exactly one.
Duplicate map keys and conflicting markers are rejected. Decoded array data
across all cameras is limited to 256 MiB per request.

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

`model.embodiment` of the serve config (`single_arm`, `bimanual` or `so101`)
names the rig of the checkpoint. A checkpoint of another rig fails at the boot.

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

The server returns commands only, never the predicted video, so each call
skips the video decoder of the last Euler step (`skip_video_decoder`,
default `auto` = on). The action expert reads only the encoder of the trunk,
so the commands keep their bits; the call is shorter by one decoder pass.
Both serve configs set `skip_video_decoder: on`. To run the full last step,
a deploy sets `skip_video_decoder: off` in the `session` block of its config,
and a client sends `?skip_video_decoder=off` for one session. A pipeline
session in your own code keeps the full step by default
(`PraxisPipeline.session(skip_last_video=False)`), so that
`Session.decode_video` can decode the predicted video.

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

The client times the chunked schedule (its default `--query` holds
`schedule=chunked`): a real-time session is periodic samples on protocol v2,
which this client does not speak. The positronic client times a real-time
session end to end.

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

The task string goes to the text encoder verbatim. The bimanual checkpoint
`praxis-1-yam` was trained on window captions such as "Pick up the pen from
the center of the table, then place the pen in the bag", with no closing
period, so phrase its tasks in that form (top README, "Bimanual rig").

The query of the URL holds the session parameters, for example
`?sample_seed=0&rollout_id=run1`. Without `sample_seed`, each session draws a
random seed.

The default session is the chunked schedule. Real-time chunking is opt-in
with `schedule=rtc` in the query (see "Real-time chunking"); it needs
Positronic 0.4.4 or later.

## Serve an SO-100 / SO-101 checkpoint

An so101 checkpoint (`"embodiment": "so101"`, `praxis/so101.py`; top README,
"SO-101 rig") has a 6-D joint head: the 5 arm joints and the gripper of an
SO-100 / SO-101 follower in LeRobot's `use_degrees=True` units. Serve it with
`configs/serve_so101.yaml`. The checkpoint was trained with NVFP4 QAT of the
routed experts, and the config serves them as NVFP4 (`routed_precision:
nvfp4`, an SM120 or SM100 GPU with the `[nvfp4]` extra); on another GPU set
the bf16 masters:

```bash
python -m praxis.serve --config configs/serve_so101.yaml \
    --set model.checkpoint=checkpoints/praxis-1-so101-nvfp4 \
    --set model.nvfp4_cache=outputs/nvfp4_cache      # keeps the packed banks between boots
python -m praxis.serve --config configs/serve_so101.yaml --set model.routed_precision=bf16
# the bf16 checkpoint praxis-1-so101
python -m praxis.serve --config configs/serve_so101.yaml \
    --set model.checkpoint=checkpoints/praxis-1-so101 --set model.routed_precision=bf16
```

The client is `demo/so101/so101_client.py` (LeRobot; no praxis import). A
Positronic client with a LeRobot driver works too: the session is the
observed-context session of the other rigs, with these rig settings:

- The frame stack keys are `image.exterior`, `image.wrist` (with `views=2`,
  the default) and `state`, the `(6,)` LeRobot reading `shoulder_pan.pos ..
  wrist_roll.pos` in degrees and `gripper.pos` in 0..100. A chunked
  observation carries the executed chunk and the current entry
  (`history_frames` 24; the meta's `stack_entries` is 25).
- The session parameters of the rig: `views` (1 or 2; a 1-view session reads
  `image.exterior` only), `grip_closed` and `grip_open` (the follower's
  `gripper.pos` at closure 1 and 0; measure them with
  `so101_client.py --measure-grip`). The other rigs refuse them.
- The commands are `{"joints": (6,), "timestamp": s}`: absolute targets in
  the units of `state` (the `so101_joints` wire). The answer also holds
  `actions` `(R, 6)` float32, `timestamps` and `joint_names` next to
  `result`.
- The canvas is the fit rule of training on the session's views: the client
  bounds each view to a 416x416 box (a 640x480 view gives 416x312), the
  server makes a 288x416 tile of it. The warmup runs the 2-view and the
  1-view canvas.
- `schedule=rtc` serves real-time chunking as on the other rigs (see
  "Real-time chunking"), with the junction rule on the joint angles.

## Real-time chunking

With the chunked schedule, the robot plays each chunk to its end, then asks
for the next one and waits for the answer. With real-time chunking the robot
asks while it still runs the current chunk, so it does not stop between
chunks. It is opt-in: the serve configs default to the chunked schedule
(`schedule: chunked` in `configs/serve_droid.yaml`, `configs/serve_yam.yaml`
and `configs/serve_so101.yaml`). A client asks for real-time chunking with
`/api/v1/session?schedule=rtc`, and a deploy turns it on for every session
with `schedule: rtc` in the `session` block of its config.

**When to use it.** Real-time chunking removes the stop between chunks and
shortens the episodes, but it needs a short round trip. Measure the round
trip and convert it to rows at 15 Hz. If it is longer than `rtc_stride`, use
`schedule=chunked` or raise `rtc_stride`. Also keep `rtc_stride` plus the
round trip within the chunk that the robot runs. That chunk has 21 rows
while its window holds no context chunk yet: the first one or two answers of
a session, and of the context after a `reset_context`. Every later chunk has
24 rows. At the default stride of 12 the round trip must stay under
(21 - 12) / 15 = 0.6 s during a 21-row chunk, and under (24 - 12) / 15 =
0.8 s during a 24-row chunk. Past that the plan runs out before the answer
arrives and that junction loses its alignment. So a round trip from 0.6 s
to 0.8 s misaligns the first junctions of a session and after each
`reset_context`, and a round trip under 0.6 s keeps every junction aligned.
When no stride satisfies both limits (for example a 1.0 s round trip), use
`schedule=chunked`. In our robot
runs it also replanned during marginal grasps, where the chunked schedule
commits to a grasp and lifts: on a five-object task it was faster (9.1 s
against 15.2 s median time to success) with a lower, not significant, success
rate (24/30 against 28/30). Measure task success on your task too before you
turn it on.

A real-time session uses the periodic flow of Positronic
([Positronic-Robotics/positronic#1139](https://github.com/Positronic-Robotics/positronic/pull/1139)).
**The client needs Positronic 0.4.4 or later**, the first release with the
`periodic` and `inference_schedule` components. An older client refuses the
handshake of a real-time session ("This client cannot build the client
stack that the server declares", an unknown local-stack entry); it keeps the
default chunked schedule.

- The server declares the recipe
  `stop_on_fault | trajectory_player | periodic(15 Hz) | inference_schedule |
  restrict_image_size`. The client samples every 1/15 s and sends each
  sample as a call with the current frames and state only. It has no frame
  stack.
- The server keeps the samples. It answers None on most samples. Every
  `rtc_stride` rows (12 by default, 0.8 s) it rebuilds the frame stack of the
  chunked recipe from the samples and predicts. The window of each
  prediction holds the frames that the robot ran since the context started,
  one every 1/15 s, as in training.
- Calls overlap. The ready message declares `max_in_flight` (16 by default).
  A real-time session speaks protocol v2: the ready message holds a
  `session_id`, and each request carries it.
- The client sends the plan it runs with each call (the `action_prefix`
  plan mapping of
  [Positronic-Robotics/positronic#1036](https://github.com/Positronic-Robotics/positronic/pull/1036)).
  The robot runs the first rows of that plan before the answer arrives: the
  rows of the delay estimate, the longest of the last 5 prediction durations.
  The answer keeps these rows. After them the model's motion continues from
  the end of these rows. The offset between the two goes back to zero over
  `rtc_rebase_decay` rows, and the velocity fades over `rtc_vel_fade` rows.
  The grippers fade from the plan to the model over `rtc_grip_rows` rows.
- On the bimanual rig, targets past the reach of a YAM arm move back onto a
  reach radius (`rtc_reach_max_m`). The reach model puts the arm bases where
  Positronic's `yam_bimanual` embodiment puts them. Set
  `rtc_reach_max_m=off` on a rig with other arm bases. The single-arm and
  the so101 rig have no reach clamp and refuse `rtc_reach_max_m`.
- On the single-arm rig, the rows stay in the frame of the client (the frame
  the robot runs), every row keeps the control law of the session
  (`droid_impedance`), and the server checks each prediction's current pose
  against the forward kinematics of `robot_state.q`, as on a chunked call.
- On the so101 rig, the rows are joint targets (degrees x 5, `gripper.pos`).
  The junction is the same rule with each joint angle as a linear channel
  (`praxis.serve.rtc.rebase_joints`): the prefix rows of the plan, then the
  model's joints plus the offset at the prefix end, handed back over
  `rtc_rebase_decay` rows with the velocity fade, and the gripper fade of
  `rtc_grip_rows`. `server_timing` reports the offset as `rtc_junction_deg`.
  The reach clamp is a pose rule and does not apply: the joint targets are
  bounded by the follower's calibration and by LeRobot's
  `max_relative_target` on the client.

The session parameters of real-time chunking:

| Parameter | Default | Effect |
|---|---|---|
| `schedule` | `chunked` | `rtc` turns real-time chunking on (Positronic 0.4.4 or later). |
| `rtc_stride` | 12 | Rows between two predictions (1 to 20). |
| `rtc_rebase_decay` | 24 | Rows over which the junction offset goes back to zero (0 keeps it). |
| `rtc_vel_fade` | 6 | Rows over which the velocity fades from the plan to the model (0 turns it off). |
| `rtc_grip` | `cont` | `cont` fades the grippers from the plan, `model` takes the model's grips. |
| `rtc_grip_rows` | 16 | Rows of the gripper fade. |
| `rtc_reach_max_m` | `auto` | Bimanual rig: reach radius in m, `auto` (0.595 on both arms), `off`, one radius, or `left,right`. The other rigs refuse it. |
| `sample_fps` | 15 | Samples per second, a whole multiple of the 15 Hz action rate. Above 15 Hz, a prediction off the row grid of the previous one (the client skipped its sample, or a replan) restarts the context, which refills from the frame stack (one context chunk). |
| `max_in_flight` | 16 | Calls that the client may have in flight (1 to 16). |
| `precompute` | `on` | Runs the window encode and the trunk passes of each prediction on the samples before it (see below). `off` turns it off. Real-time sessions only. |

A real-time session needs `context_mode=observed`, `max_context_chunks`
of 1 or more, and the full chunk (`exec_horizon` 24 or null). Under the
`rtc` schedule set in a deploy's config, a session that asks for a
stateless context or a shorter `exec_horizon` gets the chunked schedule; with
an explicit `schedule=rtc` in its query it is refused. These defaults are the junction rule of our robot runs with
real-time chunking. They can change when we have more robot results.

### Precompute of the next prediction

A real-time prediction encodes its window (the frames that the robot ran
since the context started), runs the trunk of both Euler steps and runs the
action expert. The window rows are samples that arrive before the
prediction, and no trunk pass reads the actions or the current pose. So the
server does this work on the samples before the prediction
(`praxis/serve/precompute.py`). It is on by default (`precompute: on` in
the serve configs), and it applies to real-time sessions only
(`schedule=rtc`). The default session is chunked: it does not use the
precompute, and it is not faster with it. A client turns it off with
`?precompute=off`, and a deploy with `precompute: off` in the `session`
block of its config.

- Each window row goes into a streamed encode as soon as its sample
  arrives (`praxis.vae.StreamEncode`: the chunks of `Autoencoder.encode` in
  the same order on the same buffers).
- When the last window row is in, one row before the prediction, the server
  runs the trunk passes (`PraxisPolicy.plan_window_joint`). The prediction
  then runs only the expert passes (`finish_window_joint`).
- The prediction checks each value before it uses it: a latent only for an
  equal clip, the trunk passes only for an equal anchor, cleans, text, seed
  and settings (`torch.equal`). On any difference it computes everything
  as without the precompute.

The answers keep their bits: the same model calls run on the same inputs.
Each prediction reports `rtc_plan` in `server_timing`: `hit` (the trunk
passes ran ahead), `late` (the sample of the last row was skipped, so the
prediction finished the window), `miss` (an input differed), `stale` (a
replan or a skipped prediction sample) or `none` (a cold start, or a session
that does not plan).

The default applies only where it can: on a chunked session it does
nothing, with no error. An explicit `precompute=on` on a chunked session is
refused, as the other real-time parameters are. A session with
`cfg_batched=on` makes no plan yet: the plan runs the guided and the
unguided passes one after the other, so those predictions report `none` and
run the trunk passes in the call (the window encode still runs ahead).
`cfg_batched` has no effect on the `window_joint` sampler today. When
batched guidance lands, the plan must run the same batched passes before
this refusal can go.

The trunk passes need the last window row, so they start at most one row
(67 ms at 15 Hz) before the prediction, and the prediction waits for the
rest of them on the inference thread. The gain is the encode plus that head
start, so it depends on how early the client sends the last row. Most of
the plan's GPU time stays on the critical path. Measured on one H100, with
the median over the warm predictions of each session. The columns are both
off, the precompute alone, and both on (the default of a real-time session):

| Rig | `precompute=off&skip_video_decoder=off` | `precompute=on&skip_video_decoder=off` | both on (default) |
|---|---|---|---|
| `praxis-1-yam`, round trip (client) | 638 ms | 523 ms | 397 ms |
| `praxis-1-yam`, server time with the plan, no head start | 641 ms | 603 ms | 475 ms |
| `praxis-1-droid`, round trip (client) | 694 ms | 601 ms | 379 ms |
| `praxis-1-droid`, server time with the plan, no head start | 690 ms | 688 ms | 469 ms |

- The round trip: a recorded episode per rig, replayed on the wall clock by
  Positronic's client over a loopback WebSocket, 36 warm predictions per
  session. The client sends each sample on its tick, so the plan gets the
  one-row head start.
- The server time: the samples go in back to back with no wall-clock gap,
  timed from the start of the sample that runs the plan to the end of the
  answer. With no head start, the precompute alone gains little (YAM
  -38 ms, DROID -2 ms: the DROID plan, with video guidance 4.0, takes
  about 580 ms of GPU time).
- Most of the combined gain comes from `skip_video_decoder`. The
  precompute alone gains the encode plus up to one row.

The `served_ms` of the answers (YAM: 634 ms to 213 ms) leaves out the wait:
the server reads the prediction request only after the sample before it,
which runs the trunk passes. Read the round trip instead. The precompute
holds the taps of one plan per session (about 0.6 GB on the YAM canvas at
depth 1), and drops them before a fallback runs.

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
