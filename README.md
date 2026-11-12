# Praxis-1

[![Tech report](https://img.shields.io/badge/Tech%20Report-Runway-black)](https://runway.com/research/praxis-1-video-pretraining-to-robot-control)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/collections/runwayml/praxis-1)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue.svg)](pyproject.toml)

Praxis is a video-action policy for robot arms. There are two checkpoints of
one model variant: `praxis-1-droid`, the single-arm checkpoint, for a
single-arm robot with an exterior camera and a wrist camera (the DROID rig),
and `praxis-1-yam`, the bimanual checkpoint, for two YAM arms with a top
camera and two wrist cameras (see [Bimanual rig](#bimanual-rig)).
For each policy call, the model predicts the next video chunk and the next
action chunk. The robot then executes the action chunk.

Both checkpoints also have NVFP4 versions, fine-tuned with quantization-aware
training of the routed experts:
[`runwayml/praxis-1-droid-nvfp4`](https://huggingface.co/runwayml/praxis-1-droid-nvfp4)
and [`runwayml/praxis-1-yam-nvfp4`](https://huggingface.co/runwayml/praxis-1-yam-nvfp4)
(see [Get the weights](#get-the-weights)). For the SO-100 / SO-101 arm there
are [`runwayml/praxis-1-so101`](https://huggingface.co/runwayml/praxis-1-so101)
and its NVFP4 version
[`runwayml/praxis-1-so101-nvfp4`](https://huggingface.co/runwayml/praxis-1-so101-nvfp4)
(see [SO-101 rig](#so-101-rig)).

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
| Video autoencoder | `praxis/vae.py`, `praxis/vae_decoder.py` | Wan2.2 VAE | Maps frames to latents and back. Frozen. |
| Text encoder | `praxis/text_encoder.py` | Qwen3.5-2B | Maps the task string to tokens. Frozen. |
| Sampler | `praxis/sampler.py` | - | Flow-matching Euler steps from noise to the sample. |
| Robot input and output | `praxis/single_arm.py`, `praxis/bimanual.py`, `praxis/robot_io.py` | - | Camera canvas, pose conversions, normalization, context window, command format, control modes. |
| Pipeline | `praxis/model.py`, `praxis/pipeline.py` | - | Connects all parts. |

The backbone and the action expert share one checkpoint (about 54 GB in
bfloat16). The autoencoder and the text encoder use their public weights.

## Execution

### Inference optimizations

The baseline operations are packed SwiGLU, fused decoder modulation, fused VAE
encoder RMSNorm/SiLU and average pooling, and prompt embedding caching. The
sampler also skips unused unconditional action/video branches. The
optimizations below keep the checkpoint, sampler
recipe and shared server/simulation pipeline. Defaults refer to
`PraxisPipeline` sessions; lower-level APIs can have different defaults.

The tables separate [default transformer operations](#default-transformer-and-sampler-optimizations),
[camera, VAE and text operations](#default-camera-vae-and-text-optimizations),
and [optional paths](#optional-and-hardware-specific-paths).
[Thor and architecture selection](#thor-and-architecture-selection) summarizes
the hardware guards. An enabled flag still requires compatible inputs.

### Default transformer and sampler optimizations

| Optimization | Baseline behavior | Current default and implementation |
|---|---|---|
| Rotary embedding | Separate eager rotation operations | `FUSED_ROPE=True`: one Triton pass with eager FP32 rounding; `layers.py`, `kernels/rope_triton.py`. |
| Device tables and indices | Repeated table transfers and CPU-built expert indices | `CACHE_DEVICE_TABLES=True`: reuse rotary frequencies and position IDs on the device. Build action-expert tap indices on the device. |
| Encoder context reuse | Recompute the full window at every Euler step | `session.ctx_cache=True`: cache context K/V and taps within each call and guidance branch; `model.py`, `backbone.py`. Disabled for tracing and NVFP4. |
| Cached projection shapes | Full-window projections without query slicing | `CTX_FULL_ROWS=True`: preserve full attention-gate and FP32 router GEMM shapes when processing only current queries. `KV_INPLACE=True` updates current K/V rows without concatenation. |
| Rotary Q/K normalization | RoPE followed by separate RMSNorm calls | `FUSED_ROPE_QK_NORM=None`: fuse supported BF16/FP16 CUDA rotation and normalization, retaining intermediate rounding. |
| Transformer normalization | Separate residual addition, RMSNorm and layer scaling | `FUSED_NORM=None`: fuse supported BF16 CUDA calls. Reuse versioned affine weights and defer eligible residual additions into the next normalization. |
| Decoder modulation and normalization | Fused modulation followed by separate normalization | `FUSED_MOD_NORM=True`: combine deferred residual addition, modulation, RMSNorm and layer scaling when supported. |
| Attention output gate | Separate head merge and gate product | `FUSED_GATE=True`: merge heads and apply the gate in one kernel. |
| Router input conversion | Widen each FP32 router input chunk | `ROUTER_F32=True`: reuse FP32 rows written by fused normalization. Router GEMM shapes and chunk boundaries stay fixed. |
| Guided encoder execution | Run conditional and unconditional passes separately | `session.pair_cfg=True`: run encoder blocks in lockstep and share grouped expert-weight reads. `PAIR_BUFFER=True` makes merged feed-forward rows a buffer view. `PAIR_GEMM_CONFIG` fixes grouped-GEMM tiles. |
| Shared-expert activation | Separate SiLU and product | `SHARED_SILU_MUL=True`: fuse the product after the ordinary shared-expert GEMM. |
| MoE dispatch and output reduction | PyTorch gather, weighting, inverse permutation, reduction and scaling around grouped GEMMs | Automatic stable Triton dispatch and top-8 combine paths, including chunked FP32 accumulation and shared-output addition; `kernels/layers_triton.py`. Unsupported calls retain their ordinary path. |
| Action-expert normalization | Separate RMSNorm/AdaLN tail operations | `FUSED_AE_NORM=True`: fuse supported normalization tails while retaining the eager square/mean calculation. |
| Action-expert launch overhead | Eager launches | `action_cuda_graphs=True`: replay validated action-expert graphs on CUDA `window_joint` inference. DiT and VAE remain outside capture. |
| Expert/decoder scheduling | Sequential execution | `session.overlap_expert=True`: guided inference runs the action expert on a side stream beside the video decoder. Tracing disables overlap. |
| Teacher-forcing predicate | Read five per-token metadata tables | `TF_PACKED_MASK=True`: use one integer role/chunk code per token. Retain partial/full sparse tile lists and traversal. Interval and metadata predicates remain available. |
| Attention compatibility and plan reuse | Checkpoint-selected attention with the original dispatch | Flash adapters cache converted masks and use compatible sparse block sizes. Unsupported SM110/SM120 Flash layouts fall back to Triton. Per-tower overrides and cuDNN plans are available. |
| Flash compilation cache | Caller/PyTorch limits | `FLASH_RECOMPILE_LIMIT=64`: raise the process-wide minimum to 64 compiled variants and 1024 accumulated variants. `None` preserves caller limits. |

### Default camera, VAE and text optimizations

| Optimization | Baseline behavior | Current default and implementation |
|---|---|---|
| Causal VAE encode deduplication | Encode the current frame, context and anchor separately | `session.state.dedup_encodes=True`: reuse the causal context's first latent as the anchor at depth one and above; `single_arm.py`. |
| Current-frame preparation | Build a current-frame clip on every call | `LAZY_CURRENT_CLIP=True`: build it only when an encode needs it. Resets, short stacks and stateless calls still build it. |
| Camera crop/copy | CPU preparation before transfer | `TILE_ON_DEVICE=True`: prepare on CUDA only when neither view needs resizing. Other sizes keep CPU resizing to preserve input bytes. |
| VAE encoder cache storage | Clone causal cache slices | `ENCODER_CACHE_VIEWS=True`: retain views of buffers that will not be overwritten. |
| VAE encoder casts | Repeat FP16 autocast conversions | `VAE_HOIST_CASTS=True`: cache FP16 convolution weights/biases and construct FP16 input chunk buffers. |
| VAE encoder residual normalization | Separate residual addition and the next normalization | `VAE_FUSED_ADD_NORM=True`: fuse the residual add with the next fixed-setting RMSNorm/SiLU kernel. |
| VAE RGB decoding | Reference PyTorch decoder | `vae_decoder_impl="fused"`: fused RMSNorm/SiLU, FP16 convolutions, implicit spatial padding and streamed RGB assembly; `vae_decoder.py`. RGB pixels can differ from reference decoding. |
| VAE decoder cache assembly | Separate history/input assembly | `fuse_cache_input=True`: one Triton pass assembles causal convolution inputs and history; `kernels/vae_decoder_cache_triton.py`. |
| Text-kernel compatibility | Pin preferred FLA settings when available | Keep those fixed settings; use the supported two-stage recurrent-state setting when the preferred three-stage setting is unavailable. H100 retains its existing setting; `text_encoder.py`. |

Automatic selection checks supported device, dtype, shape, layout, gradients,
autocast and module hooks where applicable. `None` means automatic selection
for `FUSED_NORM`, `FUSED_ROPE_QK_NORM` and `FUSED_MOE_SOLO`. Explicit booleans
override selection. Set module switches before warmup and CUDA
Graph capture. The DiT towers are not compiled as whole modules. Fixed kernel
settings retain their specified accumulation and rounding order.

The fused VAE decoder defaults are `fuse_norm_silu=True`,
`stream_output=True`, `fuse_cache_input=True`, `compile_decoder=False` and
`norm_output_dtype="float32"`. They control RGB decoding independently of
the transformer and action-expert optimizations.

### Optional and hardware-specific paths

These paths are separate from the default portable operations above.

| Path or control | Default | Scope |
|---|---|---|
| `FUSED_MOE` | `False` | Opt-in paired gather + gate/up + SwiGLU and combine fusion. Paired encoder execution still works while this flag is off. |
| `FUSED_MOE_DEC` | `False` | Opt-in fused expert GEMMs outside the pair driver. H100 and B200 policy measurements found regressions. |
| `FUSED_MOE_SOLO` | `None` | Unchunked combine/scale/shared-add fusion on supported CUDA calls except SM90. Inactive for the DROID checkpoint's chunked MoE. |
| `PAIR_DEC` | `False` | Opt-in: with `session.pair_cfg` and both decoder guidance passes, run the decoder passes block by block in lockstep so each MoE layer reads its routed weights once (`forward_pair`). Bitwise equal to separate passes on Thor (BF16 with `FUSED_MOE`/`CUTE_EXPERTS`, NVFP4 with `PAIR_NVFP4`). Not measured on other GPUs. |
| `MOE_TAIL_NORM` | `False` | Memory-traffic fusion: a chunked MoE layer driven by the block generator returns its routed tail unevaluated. The next block's fused residual-add/norm, or the residual add of a block that does not defer, then does the token combine and shared-expert add in the same kernel (`kernels/moe_tail_triton.py`; the ordinary roundings). The BF16 tail (with `FUSED_MOE`/`FUSED_MOE_DEC`) is pure Triton and selectable on every CUDA GPU. It is unmeasured off Thor and showed no gain with Thor BF16. The NVFP4 tails (one-chunk and paired combines) are limited to the SM110 NVFP4 backend and are faster with Thor NVFP4. Other NVFP4 backends keep their own combine. |
| `ROUTER_FUSED` | `True` on SM100/SM103/SM110 | Requires CuTe DSL. H100, SM120 and other unsupported architectures keep ordinary routing. On supported GPUs, after the ordinary FP32 router GEMM, sigmoid, bias, top-8, gates and the expert-dispatch metadata run as one barrier-free CuTe DSL kernel pair (`kernels/router_fused.py`) instead of about ten PyTorch ops per MoE layer. Bitwise with the pinned torch: it reproduces `torch.topk` tie order and the gate-sum order, so a torch upgrade needs the `ROUTER_FUSED_AUDIT` replay audit rerun. |
| `MOD_COMPACT` | `True` | Portable (torch + Triton): the decoder keeps its per-token conditioning as distinct rows (registers plus one row per conditioning token instead of each repeated per patch), runs each block's modulator GEMM on those rows (4x fewer rows for DROID), and the fused modulation-norm kernels read the shift/scale row of each token by index. Bitwise by construction (each GEMM row depends only on its own input); other consumers get the expanded tensors. Tested bytewise on H100 and Thor. |
| `action_expert.FUSED_KV_SOURCE` | `True` for supported CUDA inputs | Portable Triton: the action expert's cross-attention KV source `(rms_norm(tap) * tap_norm_scale).to(dtype)` runs its widen, rstd product, cast, weight product and cast as one kernel after the eager mean of squares (`norm_triton.rms_scale`). Tested bytewise on H100 and Thor. `FUSED_AE_NORM` (default on) covers the other action-expert norms. |
| `ENC_GATE_UP_AUTO`, `DEC_GATE_UP_CONFIG`, `DEC_DOWN_CONFIG` | Fixed tuning enabled/configured | Row-count-based encoder tiles and measured decoder tiles. These apply when the corresponding fused expert path is selected. They do not independently enable it. |
| `routed_precision="nvfp4"` | Off; BF16 remains default | Native routed-expert NVFP4 on SM100/SM103, SM110 and SM120. Dense layers and the action expert retain checkpoint dtypes. H100 rejects this mode. |
| `nvfp4_cache` | No disk cache | Reuse validated CPU-packed expert weights instead of repacking at startup. GPU execution retains no BF16 routed-weight masters. |
| `praxis.kernels.nvfp4.QUANT_WIDE` | `True` within NVFP4 | Fixed 64-row packing tiles and eight warps for the native packing kernel. This is separate from the SM110 experiment with the same short name. |
| `praxis.quantization.nvfp4.FUSED_NVFP4` | `True` within compatible NVFP4 backends | Fuse rescaling, SwiGLU, amax and output preparation where supported. SM100/SM103 retain their native fusion paths; SM110/SM120 use compatible elementwise stages. |
| `praxis.quantization.nvfp4.PAIR_NVFP4` | `True` where compatible | Paired CFG dispatch requires the SM110 scale layout. Other backends retain ordinary dispatch. NVFP4 context caching stays disabled. |
| `CUTE_DENSE_FFN` | `False` | Memory-traffic fusion for tcgen05 GPUs (SM100/SM103/SM110; needs Quack): the dense SwiGLU gate/up GEMM applies SiLU and the product in its epilogue, so the gate/up outputs never reach memory; down stays cuBLAS. Rows dispatch through a per-architecture table in `kernels/cute_dense.py`. Only SM110 has a measured table (bitwise; faster at 256 to 17,480 rows); SM100/SM103 run the ordinary ops until `scripts/tune_cute_dense.py` builds theirs. SM90 and SM120 are not selected. |
| `nvfp4_sm110.NATIVE_E2M1` | `False` | Thor-specific (SM110 NVFP4 backend): the quantize kernels pack FP4 codes with the hardware E2M1 conversion, as the native SM100/SM120 packing kernel does, instead of seven threshold compares. Same codes for every float32 input. |
| `CUTE_EXPERTS`, `SHARED_QUACK`, `ROUTED_QUACK_DOWN` | `False` | Thor-specific CuTe expert GEMMs, guarded for SM110 and loaded with optional dependencies. |
| `single_arm.TILE_U8_PINNED` | `True` | Portable: when tiles are resized on the CPU for a CUDA device, write the uint8 tiles into a persistent pinned canvas, copy it non-blocking and convert to FP32 on the GPU (exact; same values, dtype and strides). Faster on Thor, partly through less shared-DRAM contention; not measured on discrete GPUs. |
| `THOR_TILE_RESIZE`, `TILE_ON_DEVICE_ANTIALIAS` | `False` | Explicit Thor GPU-resize experiments. The default crop/copy path avoids GPU resizing. |
| `nvfp4_sm110.EPI_SWIGLU`, `ZERO_PAD_ONLY`, `QUANT_WIDE` | `False` | Thor-specific epilogue, padding-only initialization and quantization-tile experiments. |
| `praxis.attention.SM110_FLASH_MASK_256`, `praxis.backbone.SM110_INTERVAL_MASK` | `True` / `False` | Thor only (`is_thor`). The first selects 256-row Flash masks on SM110 (`False`: 128-row masks and the Triton fallback). The second selects the interval predicate for Thor Flash masks. FA4 on Thor changes action and latent bits relative to Triton attention. |
| Per-tower attention overrides | The checkpoint's backend (Flash in the released checkpoints) | Select Triton or opt-in cuDNN independently for encoder and decoder. Switching attention backend can change action and latent bits. |
| `skip_last_video` (`session.skip_video_decoder` in configs) | Off in `PraxisPipeline.session`; on in the serve configs and the benchmark | Skips the video decoder of the last Euler step, and its unguided pass when action guidance is 1.0. Actions and commands keep their bits, but no predicted video latent is returned. Keep it off when latents are required. |
| Decoder-step compilation / BF16 norm history | Off / FP32 history | `compile_decoder=True` and `norm_output_dtype="bfloat16"` are explicit decoder experiments. Validate RGB behavior separately. |
| Historical bimanual CUDA Graph helper | Off | `pipe.enable_cuda_graphs()` captures the unguided bimanual `chunk` path. It is separate from the default action-expert graphs. |

Tuning or validating an opt-in fusion on a new GPU: first compare action, command
and predicted-latent bytes of the full policy with the flag off and on, using
fixed inputs and the same GPU. Then compare policy latency. For
`CUTE_DENSE_FFN`, run `python scripts/tune_cute_dense.py OUT [--ckpt DIR]` on that
GPU. It checks each candidate configuration bitwise against the eager module, times
it, and prints the `GATE_UP_TABLES` entry to add to `kernels/cute_dense.py`.

### Thor and architecture selection

Several portable optimizations were adapted from the Thor work: paired CFG
execution, fused normalization and rotary normalization, attention gating,
KV updates, FP32 router inputs, shared-expert activation, VAE cache/cast
reuse, lazy clips and action-expert normalization. They now use the shared
input-compatibility guards. Their origin does not restrict them to SM110.
Portable source comments describe the operation; Thor labels identify the
retained device-specific implementations.

The following table describes the current dispatch rules. It is not a
performance or bitwise-parity result for every listed architecture.

| Path | SM90 (H100) | SM100/SM103 | SM110 (Thor) | SM120 |
|---|---|---|---|---|
| Portable BF16 normalization, rotary and elementwise fusions | Automatic when supported | Automatic when supported | Automatic when supported | Automatic when supported |
| `FUSED_MOE`, `FUSED_MOE_DEC` | Off | Off | Off | Off |
| `FUSED_MOE_SOLO=None` | Off | Eligible unchunked calls | Eligible unchunked calls | Eligible unchunked calls |
| Native NVFP4, when explicitly requested | Unsupported | FlashInfer SM100 backend | Quack SM110 backend | FlashInfer SM120 backend |
| Paired NVFP4 dispatch | Unavailable | Ordinary per-pass dispatch | Paired when compatible | Ordinary per-pass dispatch |
| Native packing `nvfp4.QUANT_WIDE=True` | Not used | Wider tiles | Separate SM110 packing path | Wider tiles |
| Retained CuTe BF16 expert variants | Not selected | Not selected | Explicit opt-in; off by default | Not selected |
| GPU resize experiments | Not selected | Not selected | Explicit opt-in; off by default | Not selected |
| `CUTE_DENSE_FFN` (opt-in) | Not selected | Selectable; no table, ordinary ops | Measured table | Not selected |
| `MOE_TAIL_NORM` (opt-in) | BF16 tail | BF16 tail | BF16 and NVFP4 tails | BF16 tail |
| Requested Flash attention | Flash | Flash with compatible 256-row tiles | Flash with a compatible mask block size divisible by 256; otherwise Triton | Triton fallback |

`praxis.hardware.is_thor` matches capability `(11, 0)` exactly.
The SM110 NVFP4 backend retains its own scale layout and grouped GEMMs.
`PAIR_NVFP4=True` therefore does not enable paired quantized dispatch on
SM100/SM103 or SM120. All NVFP4 backends disable encoder context caching.

The two packing switches have different scopes:

- `praxis.kernels.nvfp4.QUANT_WIDE=True` selects the default native packing
  tiles on the SM100/SM103 and SM120 paths.
- `praxis.kernels.nvfp4_sm110.QUANT_WIDE=False` leaves the Thor tile experiment
  off. `ZERO_PAD_ONLY` and `EPI_SWIGLU` in that module also remain off.

For native NVFP4 on Thor, install both optional extras:

```bash
uv pip install -e '.[nvfp4,thor]'
```

The `thor` extra supplies Quack for the SM110 backend and optional CuTe BF16
experiments. Ordinary BF16 inference on H100 and B200 does not load those
optional kernels. No separate optimization profile is needed.
`SHARED_QUACK_MIN_ROWS=2048` limits the optional shared-expert CuTe path to
larger inputs. Smaller inputs retain the ordinary shared-expert path.

New measurement tools include `scripts/benchmark_inference.py` and
`python -m praxis.serve.latency`.
`PRAXIS_NVTX=1` enables detailed ranges from `praxis/nvtx.py`; they are off by
default. Stage timing synchronizes the GPU and can reduce overlap, so stage
gains must not be added to predict ordinary policy latency.

### Numerical agreement

Bitwise checks compare action tensors, predicted video latents and decoded
commands. They are separate from decoded RGB comparisons. Native NVFP4,
attention-backend changes and optional experiments do not imply bitwise
agreement with the BF16 reference. Use matching weights, inputs, seeds, packages,
settings and GPU, with the precision mode held fixed.

A flag being enabled does not establish a speedup on every architecture.

### VAE decoder behavior

The VAE uses a fused encoder and decoder by default. The decoder combines
RMSNorm and SiLU, uses FP16 convolutions with implicit spatial padding, and
writes decoded chunks directly into the RGB output buffer. Checkpoint weights
and the network architecture are unchanged.

The FP16 decoder also assembles causal convolution inputs and cache history
in one Triton pass by default. Disable this with
`pipe.vae.enable_fused_decoder(fuse_cache_input=False)`.
This can change decoded RGB pixels. A comparison with fusion disabled used
the same single-arm inputs and seeds at context depths zero through three.
Actions, video latents and commands stayed byte-identical. RGB mean absolute
error was 0.131 on the 0–255 scale, with a maximum difference of 4.

Select the decoder when constructing the pipeline:

```python
from praxis.pipeline import PraxisPipeline

pipe = PraxisPipeline(
    checkpoint="checkpoints/praxis-1-droid",
    vae_weights="weights/Wan2.2_VAE.pth",
    text_encoder="weights/Qwen3.5-2B",
    vae_decoder_impl="fused",  # default; alternative: "reference"
)
```

`PRAXIS_VAE_DECODER_IMPL=fused|reference` selects the mode when no explicit
argument is supplied. Standalone `Autoencoder` construction accepts the same
modes through `decoder_impl`. Fused decoding changes numerical results;
validate it on representative inputs when reference parity matters.

The fused decoder uses FP16 convolutions and requires no native extension.
Enable decoder-step compilation with
`pipe.vae.enable_fused_decoder(compile_decoder=True)`.

### CUDA Graph configuration

Action-expert CUDA Graphs are enabled by default on the current CUDA
`window_joint` pipeline. CPU execution and historical sampler layouts retain
their existing paths. Disable action graphs with
`PraxisPipeline(..., action_cuda_graphs=False)` or set
`model.action_cuda_graphs: false` in an inference, server or sim config.
For example, benchmark the default graph path with:

```bash
python scripts/benchmark_inference.py --config configs/serve_droid.yaml
```

This option captures `ActionExpert.forward_trunk_rope`. Tensor inputs are copied
into stable buffers on every replay. The sampler still prepares its inputs and
runs the DiT, text encoder and VAE directly. Guided sampling retains its expert
side stream. Stream events protect graph buffers between policy calls. Each
graph owns its memory pool, and replay returns independent output tensors.

Capture is lazy for each context depth, prompt length, dtype and execution mode.
The default cache holds at most eight shapes; further shapes run eagerly, with
one warning and an `eager` count in the pool's `stats()`. Capture warms the
expert three times and checks output bytes against eager execution. It raises
on changed bits or an unsupported shape after freezing. To set the cache limit
or freeze it after application warmup, construct the pipeline with
`action_cuda_graphs=False`, then use
`pipe.enable_action_cuda_graphs(max_graphs=16)` and call the returned pool's
`freeze()`.
Keep weights and model settings fixed after capture. Concurrent host calls are
rejected. Graph capture adds startup time and GPU memory use.
The first Flash attention call raises Dynamo's process-wide `recompile_limit`
to at least 64 and `accumulated_recompile_limit` to at least 1024.
This retains compiled variants for the policy's context depths and guidance modes.
Set `praxis.attention.FLASH_RECOMPILE_LIMIT` before the first Flash call to
change this minimum. Set it to `None` to retain the caller's limits.
Higher existing limits are preserved. Other attention backends do not set them.

The retained `pipe.enable_cuda_graphs()` helper supports only the historical
bimanual chunk path. It rejects the current single-arm pipeline. It captures text and action-expert stages. DiT and VAE encoding
and decoding execute directly, outside CUDA Graphs. Warm all intended shapes under
`torch.inference_mode()`, then call the returned pool's `freeze()` to reject
new shapes. Recreate graphs after changing captured model settings or weights.
The pool validates replay against the same execution path and returns
independent output tensors.

## Optional routed NVFP4 inference

The same codebase supports native BF16 on H100, B200 and B300. The `nvfp4` extra
adds opt-in NVFP4 for SM100/SM103 GPUs, including B200 and B300, and SM120 GPUs such
as RTX 5090. H100 uses BF16. An explicit NVFP4 request on H100 raises before
packing or transferring weights.

The SM110 NVFP4 backend is selected only on Thor. Its backend and the
optional CuTe expert paths retain their hardware guards.

Native BF16 on B200 uses the standard `configs/serve_droid.yaml` defaults.
No separate optimization profile is required.

NVFP4 packs only the routed MoE experts.
Dense layers, shared experts, routers, modulation, the action
expert, the VAE and the text encoder keep their checkpoint dtypes. Router
scores and the routed output accumulation retain the source FP32 operations.

Install `uv pip install -e '.[nvfp4]'` in the CUDA 13 environment described under Install.
Set `model.routed_precision: nvfp4` in the config of `scripts/run_inference.py`
or of the policy server (`--set model.routed_precision=nvfp4`), or pass
`routed_precision="nvfp4"` to `PraxisPipeline`. The default remains BF16.
Loading packs one expert bank at a time on the CPU and retains no BF16 expert
masters on the GPU. `model.nvfp4_cache` (or `nvfp4_cache`) optionally stores
checked packed weights.

For example, download the NVFP4-QAT checkpoint and run the server with a
reusable packed-weight cache:

```bash
hf download runwayml/praxis-1-droid-nvfp4 --local-dir checkpoints/praxis-1-droid-nvfp4
python -m praxis.serve --config configs/serve_droid.yaml \
    --set model.checkpoint=checkpoints/praxis-1-droid-nvfp4 \
    --set model.routed_precision=nvfp4 \
    --set model.nvfp4_cache=outputs/nvfp4-cache
```

SM100/SM103 uses FlashInfer's contiguous grouped GEMM with deterministic
identity finalization. Praxis retains its own BF16 rescaling, SwiGLU and FP32
routed reduction. SM120 uses FlashInfer's SM120 grouped GEMM backend.
BF16 works without installing the `nvfp4` extra.

Weights and activations are rounded with the recipe of the NVFP4 QAT training
forward: one global scale per expert matrix (weights) and per expert's rows
in each dispatch chunk (activations), both `amax / 2688`. The packed cache
checks the source tensor bytes, packed shapes and packed checksums before reuse.

Flash attention uses 256-row sparse mask tiles on SM100/SM103 and retains
128-row tiles on H100. This selection is shared by serving, simulation, and
recorded inference.
FlashAttention-4 block sparsity cannot run on SM120, so attention uses the
Triton backend of `praxis/layers.py` on that device. The text encoder pins the
supported two-stage FLA recurrent kernel when the three-stage setting is
unavailable.

This option changes the action outputs. The bitwise agreement of the native
path does not apply to quantized inference.

**Accuracy.** This option changes the actions, and has not been run in closed
loop on an SM120 GPU yet. The released single-arm and bimanual checkpoints are
bf16 checkpoints, so for them NVFP4 is post-training quantization. The SO-101
checkpoint `praxis-1-so101-nvfp4` is the exception: it was trained with NVFP4 QAT of the routed
experts (17 MoE layers, 34 matrices), so `routed_precision: nvfp4` serves it
in its training precision (see [SO-101 rig](#so-101-rig)). Earlier checks
simulated the same 4-bit rounding on H100s with two NVFP4 fine-tunes (quantization-aware training), and the results differ:
- an NVFP4 fine-tune of `praxis-1-droid` on RoboLab (action CFG 1.0, video CFG 4.0,
  `t_shift` 12.5): 279/600 with NVFP4, against 287/600 for the same weights in
  bf16, no clear difference;
- an NVFP4 fine-tune of an earlier single-arm checkpoint on the `hard` simulation suite (action CFG 1.0,
  video CFG 1.0, `t_shift` 8.0, 2 steps, `max_context_chunks` 3): 138/352 with
  NVFP4, against 188/352 for that checkpoint in bf16, clearly lower. This compares with
  the bf16 starting checkpoint, so the extra training is part of the gap.

Measure the checkpoint you use before you deploy it, for example on the `hard`
simulation suite:

```bash
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
  --checkpoint CKPT --out outputs/eval/bf16
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
  --checkpoint CKPT --set model.routed_precision=nvfp4 --out outputs/eval/nvfp4
python scripts/compare_sim_eval.py outputs/eval/bf16 outputs/eval/nvfp4
```

**Memory.** The routed experts shrink from 51.3 GB to 14.4 GB, and the policy
from 54.4 GB to 17.5 GB. With the text encoder (4.6 GB) and the VAE (1.4 GB),
the weights take about 23.5 GB before activations, so a 32 GB RTX 5090 should
fit (the peak is not measured yet) and a 24 GB GPU does not. The pipeline puts
the text encoder on the policy's GPU and there is no option yet to keep it
elsewhere; since each encoded prompt is cached, such an option would cost
little after the first call of each prompt.

**Install and first run.** `uv pip install -e '.[nvfp4]'` resolves torch again,
so repeat the cuDNN pin afterwards
(`uv pip install --no-deps nvidia-cudnn-cu13==9.24.0.43`). The first call
JIT-compiles FlashInfer's grouped GEMM with `nvcc` (CUDA 12.8 or newer, a few
minutes; the server warm-up absorbs it). The first load packs the 4,352
expert matrices on the CPU, which takes minutes (about 18 with 22 threads).
Set `model.nvfp4_cache` to keep the packed banks (14.4 GB on disk). The cache
is keyed on the checkpoint, the packing recipe and the TorchAO version; a
cached load still reads and hashes the bf16 banks.

A non-finite value in a routed row makes the global scale of its expert
non-finite for that dispatch chunk, so every row of that expert in the chunk
is affected, not only the token's own row.

## Custom kernels

The pipeline uses the attention backend of the checkpoint's `config.json`;
the released checkpoints select Flash in both the DiT encoder and decoder.
Both towers also support `triton` and `cudnn`. To select cuDNN for the decoder
while keeping Flash in the encoder:

```python
pipe = PraxisPipeline(
    checkpoint="checkpoints/praxis-1-droid",
    vae_weights="weights/Wan2.2_VAE.pth",
    text_encoder="weights/Qwen3.5-2B",
    encoder_attention_backend="flash",
    decoder_attention_backend="cudnn",
)
```

Install the optional backend with `uv pip install '.[cudnn]'` or
`uv pip install '.[flash]'`. The cuDNN extra installs the v1.31.0 tag
of [NVIDIA/cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend),
including its CuTeDSL dependencies; building from source requires Git, a C++
compiler and a CUDA-enabled environment. CUTLASS tracks public
upstream source. FlashAttention and the text-encoder kernels use the versions
specified in `pyproject.toml`; transitive dependencies follow each upstream
package's own constraints.

The server accepts `--encoder-attention-backend` and
`--decoder-attention-backend`; inference and server YAML accept the same
names with underscores under `model`. Environment defaults are
`PRAXIS_ENCODER_ATTENTION_BACKEND` and `PRAXIS_DECODER_ATTENTION_BACKEND`.
Explicit arguments take precedence over environment defaults. Without either,
each tower keeps the backend in the checkpoint's `config.json`.

For a server, add `--decoder-attention-backend cudnn`. For inference or a
benchmark, override only the decoder in the config:

```bash
python scripts/benchmark_inference.py --set model.decoder_attention_backend=cudnn --timing total --out outputs/bench_cudnn_decoder
```

This selects the DiT decoder attention backend. The VAE decoder mode is a
separate option.

cuDNN supports batch-one inference with shared-head interval masks, including
the disjoint visible regions of observed-context windows. The adapter preserves
every visible key when converting a mask to interval endpoints. Other batch
sizes or masks that vary by head raise an error. Plans are cached; warm each
shape before CUDA Graph capture.

Triton and cuDNN can change actions and video latents relative to Flash.
Bitwise agreement between attention backends is not guaranteed. Flash remains
the default for both towers.

The DiT encoder and decoder execute without whole-module `torch.compile` to
preserve their numerical behavior. FlexAttention is compiled separately;
The historical CUDA Graph path captures the text encoder and action expert.

Triton kernels live in `praxis/kernels/`:

| File | Operations |
|---|---|
| `layers_triton.py` | Packed SwiGLU, stable MoE dispatch, top-8 weighted reduction, rounded scaling/shared-expert addition, and fused modulation. |
| `norm_triton.py` | Fixed-reduction RMSNorm, residual and decoder-modulation fusion, attention gate merge and action-expert normalization tails. |
| `rope_triton.py` | Rotary embedding and fused Q/K RMSNorm with eager rounding. |
| `moe_gemm_triton.py` | Fixed-tile grouped expert GEMMs, optional gather/SwiGLU fusion and shared-expert activation. |
| `nvfp4.py`, `nvfp4_moe.py`, `nvfp4_fused.py` | Native NVFP4 packing, scales, rescaling and routed output reduction. |
| `nvfp4_sm110.py`, `nvfp4_pair.py`, `nvfp4_epi*.py` | Guarded Thor NVFP4 backend, paired dispatch and optional epilogues. |
| `cute_experts.py`, `cute_act.py`, `shared_quack.py`, `routed_quack.py` | Optional Thor CuTe expert GEMMs and activations. |
| `vae_encoder_triton.py` | Encoder RMSNorm/SiLU and channels-last average pooling. |
| `vae_decoder_norm_triton.py` | Decoder RMSNorm/SiLU with FP32 arithmetic. |
| `vae_decoder_triton.py` | Patch expansion and streamed RGB conversion. |
| `vae_decoder_cache_triton.py` | Causal convolution input and history assembly. |

Attention uses compiled FlexAttention or the selected cuDNN adapter.
Ordinary expert projections use `torch._grouped_mm`; paired execution and
explicit fused variants also use `moe_gemm_triton.py`.

Top-8 non-chunked CUDA MoE inference fuses gate weighting, inverse permutation and expert
summation while preserving BF16/FP16 product rounding and the eager FP32
accumulation order. Routed scaling and the shared-expert addition keep their
original rounding boundaries. Other layouts, autocast and
gradient-tracked calls retain their existing operation sequences.

Top-8 chunked CUDA MoE inference also fuses dispatch and combination by default.
It keeps `dispatch_chunk_size` from the checkpoint. Gates are scaled in FP32.
Expert outputs use four interleaved FP32 accumulators. The routed sum rounds
to BF16 or FP16 before the shared-expert addition. The combiner writes each
chunk directly to the final output. Training, autocast and unsupported
layouts retain the existing path. The kernels use fixed 256-column combine
tiles and disable multiply-add fusion.
CPU tests are in `tests/test_moe_chunked.py`.

Non-chunked CUDA MoE dispatch gathers expert inputs directly from the stable
expert sort and prepares gate weights, inverse indices and expert offsets in
two Triton launches. Routed scaling and the shared-expert addition use a fused
epilogue that rounds the scaled value before adding the shared output. Router
scores, top-k selection and the stable expert sort retain their PyTorch paths.

The two kernels of `layers_triton.py` also have a backward pass for
training. The forward kernels are the same for inference and training.

`Autoencoder.encode_eager` is a plain PyTorch fallback for GPUs without
Triton. Its output is close to the fused path, but it is not bitwise equal.
The original implementation uses another kernel of the VAE encoder.

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
- praxis is checked with cuDNN 9.24. PyTorch 2.13 asks for
  cuDNN 9.20, so the cuDNN pin must come after each install that resolves
  torch. The convolutions of the video autoencoder run in cuDNN.

The extras of `pyproject.toml`:

| Extra | Packages | Use |
|---|---|---|
| (none) | torch, torchvision, triton, transformers, safetensors, numpy, pillow, pyyaml | the model |
| `kernels` | flash-linear-attention, causal-conv1d | the text encoder kernels that praxis is checked with (see below) |
| `sim` | praxis-sim (mujoco, gymnasium; `packages/praxis-sim` of this repository, from PyPI), imageio, imageio-ffmpeg, scipy | the MuJoCo simulation |
| `video` | imageio, imageio-ffmpeg | mp4 export of `scripts/run_inference.py` |
| `serve` | the web server, the image decoders and the cloud host client | the policy server (`praxis/serve/README.md`) |
| `train` | huggingface_hub, av, pyarrow | the fine-tune (`praxis/train/README.md`) |
| `eval` | huggingface_hub, av, openai | the autoregressive eval (`praxis/eval/autoregressive/README.md`) |
| `nvfp4` | FlashInfer, TorchAO, ninja | optional SM100/SM103/SM120 routed-expert inference |
| `thor` | Quack 0.6.5 with CUDA 13 support | SM110 NVFP4 backend and optional CuTe BF16 expert kernels |
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
kernels. praxis is checked with these kernels. Without them the
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
| SO-100 / SO-101 | [`runwayml/praxis-1-so101`](https://huggingface.co/runwayml/praxis-1-so101) | `checkpoints/praxis-1-so101` | the SO-100 / SO-101 joint-space checkpoint (see [SO-101 rig](#so-101-rig)) |
| SO-100 / SO-101, NVFP4 QAT | [`runwayml/praxis-1-so101-nvfp4`](https://huggingface.co/runwayml/praxis-1-so101-nvfp4) | `checkpoints/praxis-1-so101-nvfp4` | `praxis-1-so101` with quantization-aware training of the routed experts for NVFP4, the default of `configs/serve_so101.yaml` |

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

The NVFP4 and SO-101 checkpoints download the same way
(`for C in praxis-1-droid-nvfp4 praxis-1-yam-nvfp4 praxis-1-so101 praxis-1-so101-nvfp4; do ...`).

The recommended sampler settings were measured on `praxis-1-droid`.

`praxis-1-base` is the pretrained model without the action heads of a rig,
so it does not run on a robot by itself. `praxis/train/README.md`,
"Fine-tune the base checkpoint", tells how to make a policy for your own
single-arm rig from it.

The checkpoint asks for the `flash` attention backend of the trunk
(`backbone.attention.backend` in `config.json`). praxis is checked
with this backend. It needs the `flash-attn-4` package:

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
# Measure total latency with the default optimizations:
python scripts/benchmark_inference.py --timing total --out outputs/bench_optimized
# Disable the default optimizations, the CUDA graphs included:
python scripts/benchmark_inference.py --no-optimizations --timing total --out outputs/bench_unoptimized
```

`--no-optimizations` disables only fused RoPE, device-table caching, encoder
context caching, action-expert overlap and VAE encode deduplication. It leaves
other fusions and action-expert CUDA Graphs as configured. It does not restore
the unoptimized code. A before/after comparison needs separate source checkouts
with the same weights, settings, inputs, seeds and packages.

The five optimizations described in "Execution" are enabled by default.
`--no-optimizations` disables them for every benchmark session.
The flag applies only to the benchmark process. Library defaults and saved
inference configurations stay enabled.
`summary.json` records the five option values. The profiler
times direct encoder and decoder calls when expert overlap is enabled.
Stage synchronization can reduce overlap. These timings include profiling overhead.
Use `--timing total` to measure latency with synchronization only at call
boundaries. This mode preserves expert overlap and omits per-stage timings.
`--save-outputs` writes actions, video latents and commands for every call to
`outputs/` inside the run directory. Use these files for before/after bitwise
comparisons. Copies and disk writes are outside the call timer, but can
affect pacing.

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
| `vae_encode` | The current-frame canvas on cold calls, or the executed stream with its anchor on context calls. |
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

The checkpoint config selects the rig: `"embodiment": "single_arm"`,
`"embodiment": "bimanual"` or `"embodiment": "so101"` in `config.json`. The
loader refuses a config without one of them.

The single-arm inference and server configs use
`checkpoints/praxis-1-droid`.

The data of the rig:

- Canvas: the pipeline crops each view to the tile aspect (centre crop) and
  resizes it to the tile of the checkpoint config: 224x416 for `praxis-1-droid`, with
  antialias (`video.antialias: true`). The exterior tile
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

The local server config uses `checkpoints/praxis-1-yam`.
Pass this path as `checkpoint` when constructing `PraxisPipeline` for YAM.

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
    224 x 416 for `praxis-1-droid` (a 224 x 832 canvas). The
    resize uses antialias when the config has `video.antialias: true`
    (`praxis-1-droid`).
  - The pose is the flange pose with no turn.
  - The normalization statistics come from the checkpoint.

The trunk uses the `flash` attention backend (`backbone.attention.backend`),
which needs the `flash-attn-4` package (`uv pip install "flash-attn-4[cu13]==4.0.0b36"`;
`docker/Dockerfile` installs it). Set the backend to `triton` to use the
Triton kernel of flex attention instead (see "Get the weights"). The two
backends differ in the last bits.

## SO-101 rig

The SO-100 / SO-101 checkpoint (`"embodiment": "so101"`) is the model variant
of the single-arm checkpoint with a 6-D joint-space head: the 5 arm joints and
the gripper of an SO-100 / SO-101 follower arm (LeRobot), reading one or two
cameras. The rig code is `praxis/so101.py`; the server serves it with
`configs/serve_so101.yaml` (`praxis/serve/README.md`, "Serve an SO-100 /
SO-101 checkpoint"); the client on the robot is `demo/so101/so101_client.py`.

Weights: [`runwayml/praxis-1-so101`](https://huggingface.co/runwayml/praxis-1-so101)
and [`runwayml/praxis-1-so101-nvfp4`](https://huggingface.co/runwayml/praxis-1-so101-nvfp4)
(see [Get the weights](#get-the-weights)). `praxis-1-so101-nvfp4` is
`praxis-1-so101` fine-tuned with NVFP4 quantization-aware training of the
routed experts (`ffn.routed`: 17 MoE layers, 34 matrices) with bf16 masters,
so `routed_precision: nvfp4` is its training precision, not post-training
quantization (see "Optional routed NVFP4 inference"). The serve config reads
it from `checkpoints/praxis-1-so101-nvfp4`. The two checkpoints have the same
`config.json`; `tests/data/praxis-1-so101.json` is a copy of it.

The data of the rig, in LeRobot terms (`SOFollower` with `use_degrees=True`,
the LeRobot 0.7 default):

- Observations: `state` `(T, 6)` = `shoulder_pan.pos`, `shoulder_lift.pos`,
  `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos` in degrees (0 = the
  calibrated mid-range) and `gripper.pos` in the follower's 0..100 scale;
  the views `image.exterior` and, with two cameras, `image.wrist`. Each value
  carries a leading time axis: one entry per executed 15 Hz row, the current
  entry last. A joint reading past 200 deg is refused as a raw tick (not
  `use_degrees`).
- Canvas: the fit rule of training on the session's views, side by side,
  exterior then wrist: each view scaled down (antialiased, never up) to fit
  416x416 with its aspect kept, the views to one common height, each trimmed
  to the 32-px grid. A 640x480 camera gives a 288x416 tile, so two views
  make a 288x832 canvas and one view 288x416; other camera sizes give other
  canvases, as in training. The checkpoint config's 288x416 tile is the tile
  of the native 480x640 view.
- Actions: 6 values per row, `[q0 .. q4, g]`: the joints in radians,
  relative to the joints at the start of the chunk, and the absolute gripper
  closure `g` in 0..1. All six are q99-normalized; the proprio is the
  absolute row at each chunk start, mean/std-normalized. The decode adds the
  anchor joints and converts back to degrees and `gripper.pos`. Chunk 0 of a
  context window gives 21 rows, every later chunk 24, at 15 Hz; the context
  is one chunk (22 entries behind the current one) by default.
- Gripper calibration: the closure is `1 - (gripper.pos - closed) / (open -
  closed)`, with `closed` / `open` the follower's `gripper.pos` when fully
  closed / open (the session parameters `grip_closed`, `grip_open`). Training
  mapped the p1 / p99 reading of each table to closure 1 / 0, so "low =
  closed" is an assumption about the training data, not a rule of LeRobot;
  a follower whose reading grows as it closes gives `closed > open`, which the
  rig accepts. The defaults 1.4 / 42.1 are the training medians; measure your
  own follower with `so101_client.py --measure-grip` and pass the readings.
- Commands: `joints` (`(6,)` absolute targets in the units of `state`, the
  values of LeRobot's `send_action`) and `timestamp`; the handshake names the
  `so101_joints` wire. The rig has one control mode, `none`: the follower
  keeps its own position loop. The client sends the arm the 15 Hz rows
  interpolated at 30 Hz. It moves the arm to the folded rest pose of the
  training episodes (`shoulder_lift` about -100 deg, `elbow_flex` about
  95 deg) before the first chunk and back to it when the run ends; it stops
  when the arm does not reach that pose.
- Prompt: the task string goes to the text encoder verbatim. Send a short
  imperative task sentence, as in the serve preset: "Pick up the red cube and
  put it in the box."
- Schedules: the chunked schedule by default; `schedule=rtc` serves
  real-time chunking with the junction rule on the joint angles
  (`praxis/serve/README.md`, "Real-time chunking").

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
kernels: the setting that praxis is checked with (`FLA_KERNEL_CONFIGS` in
`praxis/text_encoder.py`).

On SM120 GPUs, attention runs on the Triton backend instead of Flash, and on
GPUs with less shared memory one text-encoder kernel is pinned to two stages.
The outputs there are deterministic but differ in bits from the H100 results.

## Tests

```bash
pytest tests/
```

The unit tests run on the CPU and need no weights. The tests of the training
loop start 1 or 2 processes with the gloo backend. `pytest tests/` runs with
any set of extras. pytest skips a test whose extra is not installed: the DROID
data module needs `train`, the policy server `serve`, the simulator `sim` (or
`uv pip install -e packages/praxis-sim`).

CUDA kernel tests run when a GPU is available.

## Citation

```bibtex
@techreport{runway2026praxis1,
  title       = {Praxis-1: From Video Pretraining to Robot Control},
  author      = {{Runway}},
  institution = {Runway},
  year        = {2026},
  month       = nov,
  url         = {https://runway.com/research/praxis-1-video-pretraining-to-robot-control}
}
```

## License

The code in this repository is licensed under the Apache License 2.0 (see
`LICENSE`). See the license file of each set of weights for the weights.
Wan2.2 and Qwen3.5 have their own licenses.
