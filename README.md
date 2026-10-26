praxis-1

Praxis is a video-action policy for a robot arm. The checkpoint is
`praxis-1-droid`, the single-arm checkpoint. It is for a
single-arm robot with an exterior camera and a wrist camera (the DROID rig).
For each policy call, the model predicts the next video chunk and the next
action chunk. The robot then executes the action chunk.

This repository holds the inference code and the instructions to run it. The
code needs PyTorch and a few small packages. It does not need any other code
base. `praxis/train/README.md` tells how to fine-tune the checkpoint on the
DROID dataset. `praxis/sim/EVAL.md` tells how to measure a checkpoint on the
simulated DROID tasks.

## What the model does

One policy call takes these inputs:

- A task string, for example `Put the red block in the bowl.`.
- The current frame of two cameras: the exterior camera and the wrist
  camera.
- The frames of up to three executed chunks (the context).
- The current pose of the arm: position, orientation and gripper.

One policy call gives these outputs:

- The robot commands at 15 Hz. The first call of a context window gives
  21 rows (1.4 s). Each later call gives 24 rows (1.6 s). Each row holds a
  target pose (`[x, y, z, R_flat9]`, metres, row-major rotation) and a
  target gripper value.
- The predicted video of the same time span (6 latents, 21 or 24 frames).

## Model parts

| Part | File | Size | Function |
|---|---|---|---|
| Video backbone | `praxis/backbone.py`, `praxis/layers.py` | 14 + 6 transformer blocks | Denoises the video latents. Gives hidden states ("taps") to the action expert. |
| Action expert | `praxis/action_expert.py` | 16 blocks, dim 1024 | Denoises the actions. Reads the taps through cross-attention. |
| Video autoencoder | `praxis/vae.py` | Wan2.2 VAE | Maps frames to latents and back. Frozen. |
| Text encoder | `praxis/text_encoder.py` | Qwen3.5-2B | Maps the task string to tokens. Frozen. |
| Sampler | `praxis/sampler.py` | - | Flow-matching Euler steps from noise to the sample. |
| Robot input and output | `praxis/single_arm.py`, `praxis/robot_io.py` | - | Camera canvas, pose conversions, normalization, context window, command format, control modes. |
| Pipeline | `praxis/model.py`, `praxis/pipeline.py` | - | Connects all parts. |

The backbone and the action expert share one checkpoint (about 54 GB in
bfloat16). The autoencoder and the text encoder use their public weights.

## Custom kernels

The folder `praxis/kernels/` holds four custom Triton kernels. They are not
PyTorch built-ins. They set the rounding and the summation order of the
checkpoint, so that praxis gives the same output bits as the original
implementation. No kernel is necessary for speed.

| Kernel | File | Used by | Why |
|---|---|---|---|
| `swiglu_packed` | `layers_triton.py` | Routed experts of the MoE layers | Its sigmoid rounds differently from `F.silu`. |
| `modulate_fused` | `layers_triton.py` | Decoder modulation | The compiler fuses the multiply and the add. |
| `rmsnorm_silu` | `vae_encoder_triton.py` | Fused VAE encoder | The channel block size sets the summation order. |
| `avg_down` | `vae_encoder_triton.py` | Fused VAE encoder | Channels-last 2x2 average pool. |

The attention uses `torch.compile(flex_attention)` and the experts use
`torch._grouped_mm`. Both are PyTorch built-ins.

The two kernels of `layers_triton.py` also have a backward pass for
training. The forward kernels are the same for inference and training.

`Autoencoder.encode_eager` is a plain PyTorch fallback for GPUs without
Triton. Its output is close to the fused path, but it is not bitwise equal.
The original implementation uses another kernel of the VAE encoder (see
"Validation").

## Install

Use Python 3.12 and a CUDA GPU with at least 80 GB of memory (one H100 is
sufficient). Run all the steps in the root directory of the repository. The
example is for CUDA 13.0:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install packaging ninja wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 \
  pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.1
pip install -e ".[kernels,sim]"
pip install --no-deps nvidia-cudnn-cu13==9.24.0.43
python -c "import causal_conv1d_cuda; print('causal-conv1d OK')"
```

- `transformers` needs torchvision to load the processor of Qwen3.5.
- There is no `causal-conv1d` wheel for PyTorch 2.13. Thus pip compiles it
  from source with `nvcc`. `MAX_JOBS` sets the number of parallel compile
  jobs (about 1 GB of memory each).
- We made the validation results with cuDNN 9.24. PyTorch 2.13 asks for
  cuDNN 9.20, so the cuDNN pin must come after each install that resolves
  torch. The convolutions of the video autoencoder run in cuDNN.

The extras of `pyproject.toml`:

| Extra | Packages | Use |
|---|---|---|
| (none) | torch, torchvision, triton, transformers, safetensors, numpy, pillow, pyyaml | the model |
| `kernels` | flash-linear-attention, causal-conv1d | the text encoder kernels of the validation results (see below) |
| `sim` | mujoco, imageio, imageio-ffmpeg | the MuJoCo simulation |
| `video` | imageio, imageio-ffmpeg | mp4 export of `scripts/run_inference.py` |
| `serve` | the web server, the image decoders and the cloud host client | the policy server (`praxis/serve/README.md`) |
| `train` | huggingface_hub, av, pyarrow | the fine-tune (`praxis/train/README.md`) |
| `dev` | pytest, ruff | the tests |

The text encoder runs Qwen3.5 through `transformers`. If you install
`flash-linear-attention` and `causal-conv1d`, `transformers` uses their
kernels. We made the validation results with these kernels. Without them the
last bits of the text tokens change. Bitwise
parity needs `transformers` 5.2.0, `flash-linear-attention` 0.5.0 and
`causal-conv1d` 1.6.1.

## Get the weights

You need three sets of weights:

1. The praxis checkpoint `praxis-1-droid`: a directory with
   `config.json`, `model.safetensors.index.json`, 17 safetensors shards and
   `SHA256SUMS`.
2. The Wan2.2 VAE: `Wan2.2_VAE.pth` from the Hugging Face model
   `Wan-AI/Wan2.2-TI2V-5B`.
3. The text encoder: the Hugging Face model `Qwen/Qwen3.5-2B`.

The configs use these paths. The `hf` command comes with `transformers`:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir weights
hf download Qwen/Qwen3.5-2B --local-dir weights/Qwen3.5-2B
C=checkpoints/praxis-1-droid
mkdir -p $C   # copy the checkpoint files into this directory
(cd $C && sha256sum -c --quiet SHA256SUMS && echo "checkpoint OK")
```

`SHA256SUMS` lists the shards, `config.json` and the index.

The checkpoint asks for the `flash` attention backend of the trunk
(`backbone.attention.backend` in `config.json`). We made the validation
results with this backend. It needs the `flash-attn-4` package:

```bash
pip install "flash-attn-4[cu13]==4.0.0b36"
```

Without this package, set the backend to `triton`. Then the trunk uses the
Triton kernel of flex attention. The model is the same, but the last bits of
the outputs differ. This command sets the backend when the package is
missing:

```bash
python -c "import flash_attn.cute" 2>/dev/null || python - "$C/config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
c["backbone"]["attention"]["backend"] = "triton"
json.dump(c, open(p, "w"), indent=1)
print("attention backend: triton")
PYEOF
```

## Run

The fastest check is one closed-loop episode in the simulation (see
"Simulation"). It needs one GPU and the `sim` extra:

```bash
TASK=block_in_bowl   # lift_cube | cube_on_plate | block_in_bowl | block_out_of_bowl | move_block | stack_block | press_button
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
  python scripts/run_sim.py --config configs/sim_droid.yaml \
    --set task.name=$TASK --set task.prompt=null --set viewer.host=0.0.0.0
```

`task.prompt=null` selects the prompt of the task. `viewer.host=0.0.0.0`
shows the live view to all machines that can reach the port. See the
warning in `praxis/sim/README.md` ("Live view"). The earlier config name
`configs/sim_droid_k.yaml` links to the same file.

To run the policy on recorded episodes:

1. Put your episodes into a directory, one sub-directory per episode:

   ```
   episodes/<name>/meta.json                 {"prompt": "...", "fps": 15, "num_frames": 93, "views": ["exterior", "wrist"]}
   episodes/<name>/frames/exterior/000.png   one PNG per frame and camera
   episodes/<name>/frames/wrist/000.png
   episodes/<name>/observed_pose8.npy        (num_frames, 8) float32 arm poses
   ```

   A pose row is `[x, y, z, qw, qx, qy, qz, grip]`.

2. Set the paths in `configs/inference_droid.yaml`.

3. Run the script:

   ```bash
   python scripts/run_inference.py --config configs/inference_droid.yaml
   ```

   To change one value without an edit of the file, use `--set`:

   ```bash
   python scripts/run_inference.py --config configs/inference_droid.yaml --set sampler.num_steps=4
   ```

The script writes one file per policy call to `output.dir`. Each file
holds the robot commands, the actions, the predicted video latent, the
model inputs and the decoded frames:

- `pred_frames`: one frame per command row. The script decodes the chunk
  after its context latents, so frame `i` shows the time of command row `i`.
- `pred_frames_standalone`: the chunk decoded alone (21 frames). Its first
  frame is a key frame, so it does not line up with the command rows.

With `output.save_video: true` the script also writes the predicted video as
`.npy` and `.mp4`. The mp4 needs the `video` extra.

## Use the pipeline in code

```python
from praxis.pipeline import PraxisPipeline

pipe = PraxisPipeline(
    checkpoint="checkpoints/praxis-1-droid",
    vae_weights="weights/Wan2.2_VAE.pth",
    text_encoder="weights/Qwen3.5-2B",
)
session = pipe.session(prompt="Put the red block in the bowl.", seed=0)
result = session.step(observation)   # one call per executed chunk
result.commands                      # list of 21 or 24 command dicts
```

The code defaults of `SamplerSettings` and `PraxisPipeline.session` are the
recommended settings (see "Recommended settings"). To change a sampler
value, give the settings: `sampler=SamplerSettings(num_steps=4)` (from
`praxis.model`).

An observation is a dict with these keys. Each value can have a leading time
axis (a frame stack). The last row is the current time.

| Key | Value |
|---|---|
| `image.exterior` | RGB uint8, `(T, H, W, 3)` |
| `image.wrist` | RGB uint8, `(T, H, W, 3)` |
| `robot_state.ee_pose` | `(T, 7)`: `x, y, z, qw, qx, qy, qz` |
| `grip` | `(T, 1)` or `(T,)` gripper value, 0 = open |
| `task` | task string (optional, it replaces the session prompt) |

From the second call on, the observation must hold the frames and poses of
the executed chunk as a stack. The stack has 24 rows plus the current row. The session
encodes them as context. If the stack is too short, the session starts a new
context window.

A command is one dict per row:
`{"robot_command": {"type": "cartesian_pos", "pose": [xyz3, R_flat9], "mode": ...}, "target_grip": g, "timestamp": t}`.
The rotation is row-major.

### Action clamp

Training clips the q99-normalized action targets to [-1, 1]. The sampler
can give values outside this range. Before the commands, the decode clips
the actions to the training range:

1. It clips the normalized xyz and grip values to [-1, 1]. Thus the xyz of a
   command stays in the `[q01, q99]` range of the checkpoint stats, relative
   to the current pose.
2. It applies the inverse q99 map.
3. It clips the gripper to [0, 1]. With `smoothing=savgol` on the server,
   this clip comes after the filter, because the filter can go past the end
   stops of the gripper. The decode does not clip the xyz after the filter,
   so a smoothed xyz can go a small distance outside `[q01, q99]`.

`result.action_rows_norm` holds the normalized rows with no clip.
`result.action_rel` holds the relative actions of the commands, with the
clips. The checkpoint has the grip stats q01 = 0 and q99 = 1.
With these stats and no smoothing, the clips change no bit of the commands
for normalized values in [-1, 1]. With `smoothing=savgol`, the filter can go
past 0 and 1 near a fast grip change. Then the grip clip also changes the
grips of normalized values in [-1, 1].

## Settings

The defaults in this table are the values of `configs/inference_droid.yaml`.

| Setting | Default | Effect |
|---|---|---|
| `sampler.num_steps` | 2 | Euler steps per chunk. |
| `sampler.t_shift` | 1.0 | Shift of the timestep grid. 1.0 is a uniform grid. |
| `sampler.solver` | `euler` | The `window_joint` layout needs `euler`. |
| `sampler.action_steps` | null | Extra expert-only steps. The `window_joint` layout needs null. |
| `sampler.layout` | `window_joint` | One Euler update moves the video and the actions of the whole context window (see "Model variant of the checkpoint"). |
| `sampler.cfg_scale` | 3.0 | Text guidance scale of the actions. 1.0 turns it off. |
| `sampler.cfg_video_scale` | 1.0 | Text guidance scale of the video. 1.0 turns it off. |
| `sampler.cfg_batched` | false | No effect: the guided and the unguided pass run one after the other. |
| `session.max_context_chunks` | 1 | Executed chunks kept as context. |
| `session.exec_horizon` | 24 | Command rows per call. null gives the full chunk. A value below 24 moves the next observation off the chunk grid, so each call becomes a new context window. |
| `session.control_mode` | `droid_impedance` | `droid_impedance` adds the impedance gains of the DROID demonstrations to each command (`robot_command["mode"]`). `none` adds no mode. The arm then keeps its own control law. |
| `session.seed` | 0 | Chunk `k` samples with seed `seed + k`. |
| `model.vae_norm_block_c` | null | Channel block size of the fused VAE norm kernel per channel count. null means `{160: 64, 320: 64, 640: 256}`. |

### Recommended settings

These are the current recommended sampling settings of the checkpoint.
`configs/inference_droid.yaml`,
`configs/serve_droid.yaml` and `configs/sim_droid.yaml` hold them. The
settings can change when we have more robot results.

| Setting | Value |
|---|---|
| `sampler.num_steps` | 2 |
| `sampler.cfg_scale` | 3.0 |
| `sampler.cfg_video_scale` | 1.0 |
| `sampler.t_shift` | 1.0 |
| `sampler.layout` | `window_joint` |
| `session.max_context_chunks` | 1 |
| `session.exec_horizon` | 24 |
| `session.control_mode` | `droid_impedance` |

The configs also give an alternative recipe: flow shift 8.0, no text
guidance and 3 context chunks.

The earlier names `configs/*_droid_k.yaml` link to the same files, so
commands with those names still work.

## Single-arm rig

The checkpoint config selects the rig. praxis supports one rig:
`"embodiment": "single_arm"` in `config.json`. The loader refuses a config
without this key.

The data of the rig:

- Canvas: the pipeline crops each view to the tile aspect (centre crop) and
  resizes it to a 256x416 tile. The exterior tile and the wrist tile make the 256x832
  canvas.
- Actions: 10 values per row, `[xyz3, rot6d6, grip1]`. The position and the
  rotation are relative to the arm pose at the start of the chunk. The
  gripper is absolute.
- Pose frame: the model uses the Franka flange pose with no turn. Each client
  reports its pose in its own frame, named by the `descriptor` of the
  observation (`RIG_FRAMES` in `praxis/single_arm.py`). The FR3 rig (no
  descriptor) and RoboLab report the Franka hand frame: 103.4 mm along the
  flange z, turned -45 deg about it. The session converts the observed poses
  to the flange and the commands back to the frame of the client. With
  `robot_state.q`, it refuses a pose more than 5 mm from the flange forward
  kinematics. On recorded FR3 requests, the converted pose is 0.18 mm
  (mean) from the flange forward kinematics.
- Prompt: the session gives the task string to the text encoder without a
  rewrite.

### Model variant of the checkpoint

The `config.json` of the checkpoint (`format_version` 2) selects this
variant of the model:

- Trunk (`backbone.layout: chunk_causal`).
  - The encoder sequence is registers, anchor, text, the noised window and
    the clean copies of the earlier chunks.
  - A chunk-causal attention mask lets each noised chunk see the clean
    copies of the chunks before it. The clean copies get the timestep 0.
  - The position ids put the anchor before the text.
- Action expert.
  - Rotary positions come from the position ids of the trunk (dim 64, base
    2.6e7), applied before the query and key norms.
  - The global tokens come in the order registers, anchor, text.
  - The window has d + 1 chunks.
- Sampler (`sampler.layout: window_joint`).
  - One Euler update moves the video and the actions of the whole window
    together.
  - The time grid is in float64.
  - The video noise and the action noise come from separate seeds.
  - Text guidance swaps the instruction for the blank caption `" "`.
- Data.
  - The pipeline resizes each view to a 256 x 416 tile. The two tiles make
    a 256 x 832 canvas.
  - The pose is the flange pose with no turn.
  - The normalization statistics come from the checkpoint.

The trunk uses the `flash` attention backend (`backbone.attention.backend`),
which needs the `flash-attn-4` package (`pip install "flash-attn-4[cu13]==4.0.0b36"`;
`docker/Dockerfile` installs it). Set the backend to `triton` to use the
Triton kernel of flex attention instead (see "Get the weights"). The two
backends differ in the last bits.

## Simulation

`praxis/sim` runs the checkpoint in closed loop on simple MuJoCo tabletop
tasks with open-source robot models. A web page shows the run live and plays
the MP4 after the run:

```bash
MUJOCO_GL=egl python scripts/run_sim.py --config configs/sim_droid.yaml \
    --set task.name=block_in_bowl --set task.prompt=null
```

`scripts/run_sim_eval.py` runs the fixed evaluation suites of the checkpoint,
and `scripts/compare_sim_eval.py` compares two runs (see
[praxis/sim/EVAL.md](praxis/sim/EVAL.md)).

See [praxis/sim/README.md](praxis/sim/README.md) for the install, the assets
and their licenses, the live view on a remote machine, the settings, the
tasks and the results.

## Determinism

With the same inputs, the same seed and the same packages, praxis gives the
same output bits on every run. The fused VAE norm kernel has a fixed block
size. It does not autotune, because the block size changes the summation
order. To reproduce a run that used another block size, set
`model.vae_norm_block_c`.

The text encoder kernels of `flash-linear-attention` select their kernel
settings by timing in each new process. Some settings change the output
bits. Thus the text encoder sets one kernel setting for each of these
kernels: the setting of the validation runs (`FLA_KERNEL_CONFIGS` in
`praxis/text_encoder.py`).

## Validation

We compared praxis with the original implementation of the checkpoint on
one H100 in bfloat16, with PyTorch 2.13 and the `flash` attention backend.
We compared raw bytes, so `-0.0` and `+0.0` count as different.

| Part | Result |
|---|---|
| Action expert | 259 of 259 values equal per case, 6 cases (1 to 3 chunks, two timesteps). The values cover the inputs, the rotary tables, the masks, every block output and the velocity. |
| Trunk | 64 of 64 checks equal per case, 4 cases (1 to 3 chunks, and the first sampler step of 1 chunk). The checks cover the rotary positions, the masks, every block output of both towers, the taps and the velocity. |
| Full sampler | 8 fixed cases equal in fresh processes: context depth 0 to 2 with two seeds, and two cases with text guidance 3.0 and flow shift 1.0. The checks cover the noise, the time grid, the velocity inputs and outputs, the guidance combine and the outputs (19 values per case, 43 with guidance). |
| Robot input and output | The tile canvas, the action and proprio codec on the checkpoint stats and the pose frame conversion equal the values of the original implementation (`tests/test_data_path.py`). |
| Video autoencoder | Not bitwise equal. The original implementation uses another kernel of the encoder. The latents differ by 1 or 2 bfloat16 units in the last place in a part of the values. The weights are equal. |
| Closed loop | 3 tasks x 4 seeds in the simulation (`block_in_bowl`, `stack_block`, `press_button`). With the encoder of the original implementation, all 12 episodes are equal on every tick. With the encoder of praxis, 7 of 12 episodes succeed, the same count as the original implementation. |

The text encoder (Qwen3.5-2B, frozen) does not depend on the checkpoint.
Its check against the original implementation gave 1079 of 1079 values equal (token ids,
every hidden layer, output).

## Tests

```bash
pytest tests/
```

The unit tests run on the CPU and need no weights. The tests of the training
loop start 1 or 2 processes with the gloo backend. The tests of the DROID
data module need the `train` extra. Without it, pytest skips them.

## License

See the license file of each set of weights. Wan2.2 and Qwen3.5 have their
own licenses.
