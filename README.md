# praxis-1

[![Tech report](https://img.shields.io/badge/Tech%20Report-Runway-black)](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/collections/runwayml/praxis-1)

Praxis is a video-action policy for robot arms. There are two checkpoints of
one model variant: `praxis-1-droid`, the single-arm checkpoint, for a
single-arm robot with an exterior camera and a wrist camera (the DROID rig),
and `praxis-1-yam`, the bimanual checkpoint, for two YAM arms with a top
camera and two wrist cameras (see [Bimanual rig](#bimanual-rig)).
For each policy call, the model predicts the next video chunk and the next
action chunk. The robot then executes the action chunk.

This repository holds the inference code and the instructions to run it. The
code needs PyTorch and a few small packages. It does not need any other code
base. `praxis/train/README.md` tells how to fine-tune `praxis-1-base` for
your rig, with DROID as the worked example, and how to adapt `praxis-1-droid`
to a task with LoRA on one 80 GB GPU. `praxis/sim/EVAL.md` tells how to measure a checkpoint on the
simulated DROID tasks.

## Quick start

Clone the repository, install it, download the weights from Hugging Face
and run the single-arm checkpoint in the simulation. It needs Python 3.12,
one CUDA GPU with at least 80 GB of memory and the CUDA 13.0 driver. Log in
to Hugging Face once first (`hf auth login`); the `hf` command comes with
`transformers`, so log in after the install if you don't have it yet. The
install uses [uv](https://docs.astral.sh/uv/): `pip install uv`, or
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
git clone https://github.com/runwayml/praxis-1-oss.git praxis-1 && cd praxis-1
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install packaging ninja wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 \
  uv pip install --no-build-isolation --no-cache causal-conv1d==1.6.1
uv pip install -e packages/praxis-sim -e ".[kernels]" "imageio>=2.30" "imageio-ffmpeg>=0.4"   # praxis and praxis-sim from this checkout
uv pip install "flash-attn-4[cu13]==4.0.0b36"
uv pip install --no-deps nvidia-cudnn-cu13==9.24.0.43   # after every install that resolves torch

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir weights
hf download Qwen/Qwen3.5-2B --local-dir weights/Qwen3.5-2B
hf download runwayml/praxis-1-droid --local-dir checkpoints/praxis-1-droid
(cd checkpoints/praxis-1-droid && sha256sum -c --quiet SHA256SUMS && echo "checkpoint OK")

MUJOCO_GL=egl python scripts/run_sim.py --config configs/sim_droid.yaml \
  --set task.name=block_in_bowl --set task.prompt=null --set viewer.host=0.0.0.0
```

The last command runs three episodes of `block_in_bowl`. A web page on port
8000 shows each episode live, and the videos go to `outputs/sim_droid`.
"Install", "Get the weights" and "Simulation" below give the details.

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
| Robot input and output | `praxis/single_arm.py`, `praxis/bimanual.py`, `praxis/robot_io.py` | - | Camera canvas, pose conversions, normalization, context window, command format, control modes. |
| Pipeline | `praxis/model.py`, `praxis/pipeline.py` | - | Connects all parts. |

The backbone and the action expert share one checkpoint (about 54 GB in
bfloat16). The autoencoder and the text encoder use their public weights.

## Optional routed NVFP4 inference (experimental)

The `nvfp4` extra adds an inference-only path for SM120 GPUs (RTX 5090,
RTX PRO 6000 Blackwell). It packs only the routed MoE experts as NVFP4 and runs
them with FlashInfer grouped GEMMs. Dense layers, shared experts, routers,
modulation, the action expert, the VAE and the text encoder keep their
checkpoint dtypes. Router scores and the routed output accumulation retain the
source FP32 operations. Other GPUs (H100, B200 and older) are refused when the
model loads; the default bf16 path is unchanged on them.

Enable it with `model.routed_precision: nvfp4` in the config of the policy
server or of `scripts/run_inference.py` (`--set model.routed_precision=nvfp4`),
or pass `routed_precision="nvfp4"` to `PraxisPipeline`. The default remains bf16.

**Accuracy.** This option changes the actions, and has not been run in closed
loop on an SM120 GPU yet. The released checkpoints are bf16 checkpoints, so
for them NVFP4 is post-training quantization. Earlier checks simulated the
same 4-bit rounding on H100s with two NVFP4 fine-tunes (quantization-aware training), and the results differ:
- an NVFP4 fine-tune of `praxis-1-droid` on RoboLab (action CFG 1.0, video CFG 4.0,
  `t_shift` 12.5): 279/600 with NVFP4, against 287/600 for the same weights in
  bf16, no clear difference;
- an NVFP4 fine-tune of `praxis-1-droid-k` on the `hard` simulation suite (action CFG 1.0,
  video CFG 1.0, `t_shift` 8.0, 2 steps, `max_context_chunks` 3): 138/352 with
  NVFP4, against 188/352 for `praxis-1-droid-k` in bf16, clearly lower. This compares with
  the bf16 starting checkpoint, so the extra training is part of the gap. The
  `praxis-1-droid-k` run here is a later run than the 175/352 of `praxis/sim/EVAL.md`, with
  the same settings.

Measure the checkpoint you use before you deploy it, for example on the `hard`
simulation suite:

```bash
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
  --checkpoint CKPT --out outputs/eval/bf16
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
  --checkpoint CKPT --set model.routed_precision=nvfp4 --out outputs/eval/nvfp4
python scripts/compare_sim_eval.py outputs/eval/bf16 outputs/eval/nvfp4
```

**Rounding.** Weights and activations are rounded with the recipe of an NVFP4
quantization-aware training forward: FP4 E2M1 values in blocks of 16, E4M3
block scales, and one global scale per expert matrix (weights) and per
expert's rows in each dispatch chunk (activations), both `amax / 2688`. A
checkpoint trained with that recipe sees the same rounding at inference.

**Memory.** The routed experts shrink from 51.3 GB to 14.4 GB, and the policy
from 54.4 GB to 17.5 GB. With the text encoder (4.6 GB) and the VAE (1.4 GB),
the weights take about 23.5 GB before activations, so a 32 GB RTX 5090 should
fit (the peak is not measured yet) and a 24 GB GPU does not. The pipeline puts
the text encoder on the policy's GPU and there is no option yet to keep it
elsewhere; since each encoded prompt is cached, such an option would cost
little after the first call of each prompt.

**Speed.** Not yet measured. Measure it with
`python scripts/benchmark_inference.py --config configs/serve_droid.yaml --set model.routed_precision=nvfp4`;
the summary records the routed precision and the attention backend in effect.

**Install and first run.** Run `uv pip install -e '.[nvfp4]'` in the CUDA 13
environment of "Install". It resolves torch again, so repeat the cuDNN pin
afterwards (`uv pip install --no-deps nvidia-cudnn-cu13==9.24.0.43`). The first
call JIT-compiles FlashInfer's SM120 grouped GEMM with `nvcc` (CUDA 12.8 or
newer, a few minutes; the server warm-up absorbs it). The first load packs the
4,352 expert matrices on the CPU, which takes minutes (about 18 with 22
threads). Set `model.nvfp4_cache` to keep the packed banks (14.4 GB on disk).
The cache is keyed on the checkpoint, the packing recipe and the TorchAO
version; a cached load still reads and hashes the bf16 banks.

**On SM120, also with bf16.** FlashAttention-4 block sparsity cannot run on
SM120, so attention uses the Triton backend of `praxis/layers.py` on that
device. The text encoder pins the two-stage FLA recurrent kernel on GPUs
whose supported settings omit three stages. Both change the bits from the
H100 validation (see "Determinism").

A non-finite value in a routed row makes the global scale of its expert
non-finite for that dispatch chunk, so every row of that expert in the chunk
is affected, not only the token's own row.

## Custom kernels

The native path uses four custom Triton kernels in `praxis/kernels/`. They are not
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
sufficient). This is for the bf16 default; for SM120 GPUs with less memory, see
"Optional routed NVFP4 inference". Run all the steps in the root directory of the repository. The
example is for CUDA 13.0 and uses [uv](https://docs.astral.sh/uv/) (`pip install uv`):

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install packaging ninja wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 \
  uv pip install --no-build-isolation --no-cache causal-conv1d==1.6.1
uv pip install -e ".[kernels,sim]"   # the sim extra pulls in the praxis-sim package, see "Simulation"
uv pip install --no-deps nvidia-cudnn-cu13==9.24.0.43
python -c "import torch, causal_conv1d_cuda; print('causal-conv1d OK')"
```

- `transformers` needs torchvision to load the processor of Qwen3.5.
- There is no `causal-conv1d` wheel for PyTorch 2.13. Thus uv builds it
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
| `sim` | praxis-sim (mujoco, gymnasium; `packages/praxis-sim` of this repository, from PyPI), imageio, imageio-ffmpeg, scipy | the MuJoCo simulation |
| `video` | imageio, imageio-ffmpeg | mp4 export of `scripts/run_inference.py` |
| `serve` | the web server, the image decoders and the cloud host client | the policy server (`praxis/serve/README.md`) |
| `train` | huggingface_hub, av, pyarrow | the fine-tune (`praxis/train/README.md`) |
| `eval` | huggingface_hub, av, openai | the autoregressive eval (`praxis/eval/autoregressive/README.md`) |
| `nvfp4` | FlashInfer, TorchAO, ninja | optional SM120 routed-expert inference |
| `dev` | pytest, ruff | the tests |

The policy server serves the chunked schedule by default (`schedule: chunked`
in `configs/serve_*.yaml`). Real-time chunking is opt-in: a client asks for it
with `?schedule=rtc` in the session URL (see "Real-time chunking" in
`praxis/serve/README.md`). It needs Positronic 0.4.4 or later. A real-time
session precomputes the window encode and the trunk passes of each
prediction on the samples before it, with the same answer bits
(`precompute: on`; `?precompute=off` turns it off). The precompute applies
to real-time sessions only. A default session is chunked: it does not use
the precompute and is not faster with it.

The text encoder runs Qwen3.5 through `transformers`. If you install
`flash-linear-attention` and `causal-conv1d`, `transformers` uses their
kernels. We made the validation results with these kernels. Without them the
last bits of the text tokens change. Bitwise
parity needs `transformers` 5.2.0, `flash-linear-attention` 0.5.0 and
`causal-conv1d` 1.6.1.

## Get the weights

The praxis checkpoints are on Hugging Face, in the `runwayml` organization:

| Checkpoint | Hugging Face repository | Local directory | What it is |
|---|---|---|---|
| Single-arm (DROID) | [`runwayml/praxis-1-droid`](https://huggingface.co/runwayml/praxis-1-droid) | `checkpoints/praxis-1-droid` | `praxis-1-droid`, the default of the configs |
| Bimanual (YAM) | [`runwayml/praxis-1-yam`](https://huggingface.co/runwayml/praxis-1-yam) | `checkpoints/praxis-1-yam` | the bimanual checkpoint (see [Bimanual rig](#bimanual-rig)) |
| Base | [`runwayml/praxis-1-base`](https://huggingface.co/runwayml/praxis-1-base) | `checkpoints/praxis-1-base` | the pretrained model without action heads: the start of a fine-tune for a new rig |
| Single-arm (DROID), NVFP4 QAT | [`runwayml/praxis-1-droid-nvfp4`](https://huggingface.co/runwayml/praxis-1-droid-nvfp4) | `checkpoints/praxis-1-droid-nvfp4` | `praxis-1-droid` with quantization-aware training of the routed experts for NVFP4 |
| Bimanual (YAM), NVFP4 QAT | [`runwayml/praxis-1-yam-nvfp4`](https://huggingface.co/runwayml/praxis-1-yam-nvfp4) | `checkpoints/praxis-1-yam-nvfp4` | `praxis-1-yam` with quantization-aware training of the routed experts for NVFP4 |

Each checkpoint is a directory with `config.json`,
`model.safetensors.index.json`, 17 safetensors shards and `SHA256SUMS`
(54.4 GB). `SHA256SUMS` lists the shards, `config.json` and the index. The
configs read each checkpoint from its local directory.

The NVFP4 repositories hold bf16 weights in the same format, so they load
like the other checkpoints. On SM120 GPUs, `--set model.routed_precision=nvfp4`
packs their routed experts to NVFP4 at load time (see "Optional routed NVFP4
inference").

You also need two public models:

- The Wan2.2 VAE: `Wan2.2_VAE.pth` from the Hugging Face model
  `Wan-AI/Wan2.2-TI2V-5B`.
- The text encoder: the Hugging Face model `Qwen/Qwen3.5-2B`.

The `hf` command comes with `transformers`. Log in once with an account
that can read the `runwayml` repositories (`hf auth login`), then download
the weights into the paths of the configs:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir weights
hf download Qwen/Qwen3.5-2B --local-dir weights/Qwen3.5-2B
for C in praxis-1-droid praxis-1-yam praxis-1-base; do   # take only the ones you need
  hf download runwayml/$C --local-dir checkpoints/$C
  (cd checkpoints/$C && sha256sum -c --quiet SHA256SUMS && echo "$C OK")
done
```

The NVFP4 checkpoints download the same way
(`for C in praxis-1-droid-nvfp4 praxis-1-yam-nvfp4; do ...`).

The tests also use the config of the earlier single-arm checkpoint `praxis-1-droid-k`
(`tests/data/praxis-1-droid-k.json`, SHA-256
`f09961f066aa9fde871cd5032daedfd986dfd1f4b1901ada15f97a858b4ba6b9`): the same
architecture with 256x416 tiles and a resize without antialias. The
recommended sampler settings were measured on `praxis-1-droid`.

`praxis-1-base` is the pretrained model without the action heads of a rig,
so it does not run on a robot by itself. `praxis/train/README.md`,
"Fine-tune the base checkpoint", tells how to make a policy for your own
single-arm rig from it.

The checkpoint asks for the `flash` attention backend of the trunk
(`backbone.attention.backend` in `config.json`). We made the validation
results with this backend. It needs the `flash-attn-4` package:

```bash
uv pip install "flash-attn-4[cu13]==4.0.0b36"
uv pip install --no-deps nvidia-cudnn-cu13==9.24.0.43   # the install above resolves torch again
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

## Download, then run

All the commands below, from a fresh clone after "Install" and `hf auth
login`. They read the weights from the local directories of the configs.

```bash
# Weights (see "Get the weights").
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir weights
hf download Qwen/Qwen3.5-2B --local-dir weights/Qwen3.5-2B
for C in praxis-1-droid praxis-1-yam praxis-1-base; do
  hf download runwayml/$C --local-dir checkpoints/$C
  (cd checkpoints/$C && sha256sum -c --quiet SHA256SUMS && echo "$C OK")
done

# Inference: one closed-loop episode per checkpoint in the simulation (uv pip install -e ".[sim]").
MUJOCO_GL=egl python scripts/run_sim.py --config configs/sim_droid.yaml \
    --set task.name=block_in_bowl --set task.prompt=null --set viewer.enabled=false
MUJOCO_GL=egl python scripts/run_sim_yam.py --config configs/sim_yam.yaml \
    --checkpoint checkpoints/praxis-1-yam --tasks block_tower --seeds 1 --out outputs/sim_yam

# Evaluation: the fixed suites of the single-arm checkpoint (praxis/sim/EVAL.md).
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite droid_guard \
    --checkpoint checkpoints/praxis-1-droid --out outputs/eval/droid_guard
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
    --checkpoint checkpoints/praxis-1-droid --out outputs/eval/hard

# Training (uv pip install -e ".[train]", praxis/train/README.md).
# A new rig: the full fine-tune of the base on 8 GPUs. The statistics and the heads come first.
python scripts/compute_norm_stats.py --config configs/finetune_base.yaml
python scripts/init_heads.py --config configs/finetune_base.yaml
torchrun --nproc_per_node 8 scripts/finetune.py --config configs/finetune_base.yaml
# The DROID rig, a new task: LoRA on praxis-1-droid on one GPU.
torchrun --nproc_per_node 1 scripts/finetune.py --config configs/finetune_droid_lora.yaml
```

Both training configs read DROID from the Hugging Face hub
(`data.source: cadene/droid_1.0.1`). For your own data, or for a long run
(the hub has a rate limit), set `--set data.source=<a local LeRobot tree>`.

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
warning in `praxis/sim/README.md` ("Live view").

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

## Profile inference

Quick start, after "Install" and "Get the weights":

```bash
python scripts/benchmark_inference.py --config configs/serve_droid.yaml --out outputs/bench
```

The script prints a table of p50, p90 and max per stage and writes
`outputs/bench/summary.json`. The details:

`scripts/benchmark_inference.py` times the policy calls on a local GPU, per
stage. It needs no server. It loads the pipeline with the settings of a
config (default `configs/serve_droid.yaml`, the recommended settings) and
runs sessions on synthetic observations at the shapes of a robot client:

```bash
# Video guidance off (one pass per sampler step instead of two):
python scripts/benchmark_inference.py --set sampler.cfg_video_scale=1.0 --out outputs/bench_nocfg
# With a torch.profiler trace (trace.json) and a kernel table (kernels.txt):
python scripts/benchmark_inference.py --trace --out outputs/bench_trace
# With the video decoder of the last Euler step:
python scripts/benchmark_inference.py --set session.skip_video_decoder=off --out outputs/bench_decode
```

Each session sends a first call with one frame (context depth 0, 21 command
rows) and then calls with a stack of 25 frames (24 rows). As on the policy
server, each call skips the video decoder of the last Euler step
(`session.skip_video_decoder`, default on): the commands keep their bits, and
with action guidance off (`cfg_scale` 1.0) the last step runs no unguided
pass either, since that pass only guides the video. The first session
is a warm-up: it compiles the attention kernels of each context depth, so
its calls are not counted. The script writes the time of each call and of
its stages to `calls.jsonl`, and `summary.json` with the median (p50), the
p90 and the max, the GPU, the dtype, the sampler settings and the package
versions. The stages (`praxis/profiling.py`) are:

| Stage | What it holds |
|---|---|
| `text_encode` | The text encoder. The pipeline caches each prompt, so this is 0 after the first call. |
| `vae_encode` | The video encodes of the canvas: the current frame, and with context the executed frames and the window anchor. |
| `backbone` | The video backbone calls: one per sampler pass, two passes per step with video guidance. With `skip_video_decoder` on and action guidance off, the last step runs the guided pass only, without its video decoder. |
| `expert` | The action expert calls on the taps of the backbone. |
| `total_staged` | The whole policy call of the same sessions, with the canvas, the pose conversions and the command decode. |
| `total` | The whole policy call in one more session without the stage timers: one synchronize before and one after the call. This is the call time a server sees. |

Each stage time includes a device synchronize before and after it. The sync
also stops the CPU from queuing the next stage while the GPU runs this one,
so `total_staged` is higher than `total`. The sampler loop has no host
synchronize between its backbone and expert calls, so without the stage
timers the CPU queues the next kernels while the GPU runs; the stage syncs
take that overlap away. Compare `total`, not the sum of the stages.
The trace marks each stage as a named range. A trace of 10 calls is about
0.8 GB. The stages are NVTX ranges too, so under Nsight Systems
(`nsys profile -t cuda,nvtx`) the timeline shows them.

`python -m praxis.serve.latency --url <session URL>` measures a running
policy server end to end: the request encode, the round trip and the
server's own times (see `praxis/serve/README.md`, "Measure the latency").

### Measured times

We ran the script on one NVIDIA H100 80GB HBM3 (Modal), with the image of
`docker/Dockerfile`: PyTorch 2.13.0, CUDA 13.0, cuDNN 9.24, `flash-attn-4`
4.0.0b36, bfloat16 weights, the `flash` attention backend and no
`torch.compile` beyond the flex attention kernels. The checkpoint was
`praxis-1-droid` with the settings of `configs/serve_droid.yaml`
(2 steps, flow shift 12.5, 1 context chunk, the video decoder of the last
step skipped) and synthetic 416x234 views.
Each row had a warm-up session of 10 calls, then staged sessions of 10
calls (3 with video guidance 4.0, 2 with 1.0) and one plain session of 10
calls. `Total` comes from the plain session, the other columns from the
staged sessions. Each step runs one expert call, so a call has two. With
video guidance the last step runs its guided pass only, so a call has three
backbone passes. Times in ms as p50 / p90 / max:

| Video guidance | Calls | Total | Total (staged) | Backbone | Expert | VAE encode | Text encode |
|---|---|---|---|---|---|---|---|
| `cfg_video_scale` 4.0 (default): 2 passes per step, 1 on the last | first of a session (n=1; staged n=3) | 266 / 266 / 266 | 289 / 310 / 316 | 204 / 213 / 216 | 72 / 80 / 82 | 8 / 8 / 9 | 0 |
| | later (n=9; staged n=27) | 501 / 513 / 530 | 544 / 557 / 612 | 372 / 378 / 381 | 78 / 81 / 83 | 66 / 67 / 69 | 0 |
| `cfg_video_scale` 1.0: 1 pass per step | first of a session (n=1; staged n=2) | 192 / 192 / 192 | 210 / 212 / 213 | 125 / 126 / 126 | 73 / 74 / 74 | 7 / 7 / 7 | 0 |
| | later (n=9; staged n=18) | 344 / 345 / 346 | 387 / 390 / 392 | 223 / 223 / 223 | 75 / 78 / 78 | 65 / 65 / 66 | 0 |

In the same container, with the video decoder of the last step
(`--set session.skip_video_decoder=off`), the later calls took 714 ms
(`cfg_video_scale` 4.0) and 422 ms (1.0) at the p50 of `total`. The commands
were bit-identical with and without the decoder on all 90 calls.

These times include the video decode of the last Euler step. The policy
server skips that decode by default (`skip_video_decoder`, see
`praxis/serve/README.md`), so a served call is shorter by one decoder pass.

A run from a fresh clone of this repository, with the steps of "Install"
and "Get the weights" on another H100, took 17 minutes from the clone to
the numbers: 13 to install (most of it the `causal-conv1d` compile), 3 to
download the weights, 1.5 to load and run. The `flash-attn-4` install
resolves torch again, so the step above re-pins cuDNN 9.24 after it.

- The first call of a session has no context (one chunk in the window).
  The later calls hold one context chunk, so the backbone runs on twice the
  tokens and the VAE encodes the 24 executed frames.
- The warm-up calls compile the attention kernels: the first call of the
  process took 35.0 s (32.5 s with guidance off), the first call with
  context 1.1 s (0.5 s).
- A later call gives 24 command rows, 1.6 s of motion at 15 Hz. With the
  default settings a call takes about a third of that time.
- The inputs are synthetic. The routed experts of the backbone depend on
  the tokens, so real camera frames can give other times.
- A robot client adds the image encode, the network and the server around
  the call. In RoboLab runs against a hosted server of this checkpoint
  with these sampler settings (another server implementation, not `praxis/serve`, on
  one H100 or with the two guidance passes on two H100s), the client saw a
  median of 1.1 to 1.4 s per call. Measure your own path
  with `praxis.serve.latency`.

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
| `sampler.t_shift` | 12.5 | Shift of the timestep grid. 1.0 is a uniform grid. |
| `sampler.solver` | `euler` | The `window_joint` layout needs `euler`. |
| `sampler.action_steps` | null | Extra expert-only steps. The `window_joint` layout needs null. |
| `sampler.layout` | `window_joint` | One Euler update moves the video and the actions of the whole context window (see "Model variant of the checkpoint"). |
| `sampler.cfg_scale` | 1.0 | Text guidance scale of the actions. 1.0 turns it off. |
| `sampler.cfg_video_scale` | 4.0 | Text guidance scale of the video. 1.0 turns it off. |
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
| `sampler.cfg_scale` | 1.0 |
| `sampler.cfg_video_scale` | 4.0 |
| `sampler.t_shift` | 12.5 |
| `sampler.layout` | `window_joint` |
| `session.max_context_chunks` | 1 |
| `session.exec_horizon` | 24 |
| `session.control_mode` | `droid_impedance` |

The configs also give an alternative recipe: flow shift 8.0, no text
guidance (`cfg_scale` and `cfg_video_scale` 1.0) and 3 context chunks.

## Single-arm rig

The checkpoint config selects the rig: `"embodiment": "single_arm"` or
`"embodiment": "bimanual"` in `config.json`. The loader refuses a config
without one of them.

The data of the rig:

- Canvas: the pipeline crops each view to the tile aspect (centre crop) and
  resizes it to the tile of the checkpoint config: 224x416 for `praxis-1-droid`, with
  antialias (`video.antialias: true`), 256x416 for `praxis-1-droid-k`. The exterior tile
  and the wrist tile make the canvas (224x832 for `praxis-1-droid`).
- Actions: 10 values per row, `[xyz3, rot6d6, grip1]`. The position and the
  rotation are relative to the arm pose at the start of the chunk. The
  gripper is absolute.
- Pose frame: the model uses the Franka flange pose with no turn. Each client
  reports its pose in its own frame, named by the `descriptor` of the
  observation (`RIG_FRAMES` in `praxis/single_arm.py`). The FR3 rig (no
  descriptor) and RoboLab report the Franka hand frame: 103.4 mm along the
  flange z, turned -45 deg about it. A rig fine-tuned from the base sends
  `"flange"`: the model frame itself. The session converts the observed poses
  to the flange and the commands back to the frame of the client. With
  `robot_state.q`, it refuses a pose more than 5 mm from the flange forward
  kinematics. On recorded FR3 requests, the converted pose is 0.18 mm
  (mean) from the flange forward kinematics.
- Prompt: the session gives the task string to the text encoder without a
  rewrite.

## Bimanual rig

The bimanual checkpoint `praxis-1-yam` (`"embodiment": "bimanual"`) is the
model variant of the single-arm checkpoint with 20-D actions over two arms
and a canvas of three views. The weights are the Hugging Face repository
[`runwayml/praxis-1-yam`](https://huggingface.co/runwayml/praxis-1-yam) (54.4
GB, the same layout as the single-arm checkpoint; see "Get the weights").
`tests/data/praxis-1-yam.json` is a copy of its `config.json`. Download
it into `checkpoints/praxis-1-yam` and serve it with
`configs/serve_yam.yaml` (the
Modal app: `PRAXIS_CONFIG=configs/serve_yam.yaml`). The rig code is
`praxis/bimanual.py`.

The data of the rig:

- Observations: `robot_state.{left,right}.ee_pose` (`[xyz3, quat_wxyz4]`, in
  the world frame of the rig), `grip.{left,right}`, and the views
  `image.exterior` (the top camera), `image.wrist_left`, `image.wrist_right`.
- Canvas: three 256x416 tiles side by side (256x1248), in the order top,
  left wrist, right wrist. Each view goes through the fit rule of training:
  an antialiased resize so that it fits 416 px with its aspect kept, then a
  centre trim to the 32-px grid (1200x1920 or 600x960 -> 260x416 ->
  256x416). The client bounds each view to a 416x416 box. praxis serves
  16:10 views only; it refuses views at another aspect.
- Actions: 20 values per row, `[L: xyz3 rot6d6 grip1 | R: same]`. Each arm is
  relative to its own pose at the start of the chunk. The grippers are
  absolute. The proprio is the absolute 20-D pose, mean/std-normalized, with
  the gripper value of the rig (no clip).
- Pose frame: `"pose_frame": "rig"`. The poses stay in the world frame of
  the rig, as recorded in training. There is no frame conversion.
- Commands: `robot_command.left`, `robot_command.right` (`cartesian_pos`,
  `[xyz3, R_flat9]`), `target_grip.left`, `target_grip.right` and
  `timestamp`; the handshake names the `yam_ee2` wire. The rig has one
  control mode, `none`.
- Prompt: the task string goes to the text encoder verbatim. The checkpoint
  was trained on window captions such as "Pick up the pen from the center of
  the table, then place the pen in the bag" (no closing period), so send
  prompts in that form.

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
  - The pipeline resizes each view to the tile of the checkpoint config:
    224 x 416 for `praxis-1-droid` (a 224 x 832 canvas), 256 x 416 for `praxis-1-droid-k`. The
    resize uses antialias when the config has `video.antialias: true`
    (`praxis-1-droid`).
  - The pose is the flange pose with no turn.
  - The normalization statistics come from the checkpoint.

The trunk uses the `flash` attention backend (`backbone.attention.backend`),
which needs the `flash-attn-4` package (`uv pip install "flash-attn-4[cu13]==4.0.0b36"`;
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

### Install a simulator

Each simulator is its own package under `packages/`, with only its own
dependencies and its own Python range, so it installs on its own (in a LeRobot
environment, or next to a sim stack on another Python). The `sim` extra of
`praxis` pulls in the one that shares praxis's Python.

| Sim | Package | Python | Install | Reaches the model |
| --- | --- | --- | --- | --- |
| MuJoCo rig (this repository) | `praxis-sim` | >= 3.10 | `uv pip install -e ".[sim]"`, or on its own `uv pip install praxis-sim` | in process (`praxis.sim.loop`), or as Gymnasium environments (`praxis_sim.gym_env`) |

A simulator that cannot share praxis's Python or numpy reaches the model through
the policy server of the `serve` extra (two processes: the server in the praxis
environment, the sim client in its own), as the RoboLab client of `eval/robolab`
does. Such a simulator gets its own `packages/praxis-<sim>` and a row here when
it lands.

## Offline action error

`scripts/run_action_error_eval.py` measures the action error of the
checkpoint on fixed windows of 19 held-out (val) DROID episodes, with no
robot and no simulation. Each window gives two errors per chunk: the oracle error (the
actions read the recorded future video) and the deployment error (the video
and the actions sample jointly from noise). The difference is the cost of the
video prediction. The simulation suites of `praxis/sim/EVAL.md` measure the
closed loop; this eval measures the open-loop action readout on real data.
See the docstring of `praxis/eval/offline_action_error.py` for the windows,
the sampler and the metrics.

```bash
python scripts/run_action_error_eval.py --checkpoint checkpoints/praxis-1-droid \
    --vae weights/Wan2.2_VAE.pth --text-encoder weights/Qwen3.5-2B \
    --data droid_hires --out outputs/action_error/start
```

## Autoregressive prompt adherence

`praxis/eval/autoregressive/` scores whether a model's own video follows the
prompt. The model rolls out its video from one real first frame and a
prompt for 12.6 s. A VLM judge (GPT-6 Astra, any OpenAI-compatible
endpoint) checks the target object, the destination and the action. The
sets are in the Hugging Face dataset `runwayml/praxis-1-eval`. The judge and
the scorer take the mp4s of any model. See
`praxis/eval/autoregressive/README.md`.

```bash
uv pip install -e ".[eval]"
python scripts/score_autoregressive_eval.py judge --set droid --run outputs/ar/mine
python scripts/score_autoregressive_eval.py score --set droid outputs/ar/praxis-1-droid outputs/ar/mine
```

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

On SM120 GPUs, attention runs on the Triton backend instead of Flash, and on
GPUs with less shared memory one text-encoder kernel is pinned to two stages.
The outputs there are deterministic but differ in bits from the H100 results.

## Validation

We compared praxis with the original implementation of the checkpoint on
one H100 in bfloat16, with PyTorch 2.13 and the `flash` attention backend.
We compared raw bytes, so `-0.0` and `+0.0` count as different. These runs
used `praxis-1-droid-k`. `praxis-1-droid` has the same architecture, normalization statistics and
action codec, other weights, and a 224x416 canvas with an antialiased
resize. The `praxis-1-droid-k` comparisons do not show `praxis-1-droid` parity. The last column
gives the `praxis-1-droid` status: the CPU data path checks below ran on `praxis-1-droid`, the
GPU checks are pending.

| Part | Result (`praxis-1-droid-k`) | `praxis-1-droid` |
|---|---|---|
| Action expert | 259 of 259 values equal per case, 6 cases (1 to 3 chunks, two timesteps). The values cover the inputs, the rotary tables, the masks, every block output and the velocity. | Pending (GPU). |
| Trunk | 64 of 64 checks equal per case, 4 cases (1 to 3 chunks, and the first sampler step of 1 chunk). The checks cover the rotary positions, the masks, every block output of both towers, the taps and the velocity. | Pending (GPU). |
| Full sampler | 8 fixed cases equal in fresh processes: context depth 0 to 2 with two seeds, and two cases with text guidance 3.0 and flow shift 1.0. The checks cover the noise, the time grid, the velocity inputs and outputs, the guidance combine and the outputs (19 values per case, 43 with guidance). | Pending (GPU). |
| Robot input and output | The tile canvas, the action and proprio codec on the checkpoint stats and the pose frame conversion equal the values of the original implementation (`tests/test_data_path.py`). | Codec and pose frame: the pinned codec values hold on the `praxis-1-droid` statistics (`test_pinned_stats_are_the_checkpoint_stats`: they equal its `normalization` block). Canvas: equal to the training data path for 416x234 frames, the RoboLab input (`test_antialias_clip_equals_the_training_canvas`). Other frame sizes are not bitwise equal: training resizes to 416 wide and then trims to the 32-px grid, praxis crops to the tile aspect and then resizes. In an offline check against the training code, the antialiased canvases of 1280x720 and 512x288 frames differed by about 3 levels (mean absolute, 0 to 255) on random noise and by less than 0.4 on smooth gradients. |
| Video autoencoder | Not bitwise equal. The original implementation uses another kernel of the encoder. The latents differ by 1 or 2 bfloat16 units in the last place in a part of the values. The weights are equal. | Pending (GPU). The autoencoder and its weights are shared; its output on `praxis-1-droid` inputs has not been compared. |
| Closed loop | 3 tasks x 4 seeds in the simulation (`block_in_bowl`, `stack_block`, `press_button`). With the encoder of the original implementation, all 12 episodes are equal on every tick. With the encoder of praxis, 7 of 12 episodes succeed, the same count as the original implementation. | Pending (GPU). |

The text encoder (Qwen3.5-2B, frozen) does not depend on the checkpoint.
Its check against the original implementation gave 1079 of 1079 values equal (token ids,
every hidden layer, output).

## Tests

```bash
pytest tests/
```

The unit tests run on the CPU and need no weights. The tests of the training
loop start 1 or 2 processes with the gloo backend. `pytest tests/` runs with
any set of extras. pytest skips a test whose extra is not installed: the DROID
data module needs `train`, the policy server `serve`, the simulator `sim` (or
`uv pip install -e packages/praxis-sim`).

## License

See the license file of each set of weights. Wan2.2 and Qwen3.5 have their
own licenses.
