# Notes for coding agents

praxis-1 is a standalone implementation of a robot policy: a video and action
model that predicts the next action chunk from camera frames, the robot
state and a text prompt. It runs two checkpoints of one model variant: the
single-arm DROID checkpoint `praxis-1-droid` and the bimanual YAM checkpoint
`praxis-1-yam`. This file
points to the parts of the code base and gives the rules that keep the
results reproducible. `README.md` is the user guide.

## Map of the code

| Path | What it holds |
|---|---|
| `praxis/model.py` | `PraxisPolicy`: the backbone and the action expert, `from_pretrained`, `from_config`, `SamplerSettings` |
| `praxis/backbone.py`, `praxis/layers.py` | the video and text backbone (MoE transformer), its `chunk_causal` layout and the attention |
| `praxis/action_expert.py` | the action expert that reads the backbone taps |
| `praxis/sampler.py` | the flow sampler: timestep grids, the seeded noise, guidance |
| `praxis/pipeline.py` | `PraxisPipeline` and its sessions: one call turns observations into robot commands |
| `praxis/single_arm.py`, `praxis/bimanual.py`, `praxis/robot_io.py` | the robot input and output of each rig: canvases, pose frames, normalization, command decoding |
| `praxis/checkpoint.py` | the checkpoint format: `config.json` validation, strict load, `save_checkpoint`, `SHA256SUMS` |
| `praxis/vae.py`, `praxis/text_encoder.py` | the video autoencoder and the text encoder |
| `praxis/kernels/` | Triton kernels (forward and backward) that keep the bits of the checkpoint |
| `praxis/quantization/`, `praxis/kernels/nvfp4.py` | the optional routed NVFP4 inference on SM120 GPUs (`model.routed_precision: nvfp4`): packing, the cached load and the activation kernels |
| `praxis/serve/` | the policy server (WebSocket, Positronic protocol) and the Modal app. See `praxis/serve/README.md` |
| `praxis/sim/` | the closed loop of the MuJoCo simulation (the model loop, the video writer, the live viewer). See `praxis/sim/README.md` |
| `packages/praxis-sim/` | the simulator as its own package, `praxis_sim` (no torch): the single-arm DROID rig, tasks, scripted oracles, the Gymnasium wrapper. `praxis.sim.<module>` names still resolve to it |
| `packages/praxis-sim/src/praxis_sim/eval_suites.py`, `praxis/sim/EVAL.md` | the fixed evaluation scenarios of the single-arm checkpoint |
| `praxis/eval/` | the offline action error on fixed DROID windows (oracle and deployment legs) |
| `praxis/eval/autoregressive/` | the video-only autoregressive prompt-adherence eval: sets, VLM judge, scorer. See its `README.md` |
| `praxis/train/` | the fine-tune: data windows (`DatasetSpec` for other single-arm rigs), objective, MoE balance, FSDP trainer, normalization statistics. See `praxis/train/README.md` |
| `praxis/rig.py` | the base checkpoint `praxis-1-base` to a single-arm checkpoint: rig config fields and new head tensors (`scripts/init_heads.py`) |
| `configs/` | the YAML settings of inference, serving, simulation and fine-tuning |
| `scripts/` | the command-line entry points |
| `tests/` | the CPU unit tests; `tests/tiny_model.py` builds a small model without weights |

## Entry points

| Task | Command |
|---|---|
| Inference on recorded episodes | `python scripts/run_inference.py --config configs/inference_droid.yaml` |
| One closed-loop sim episode | `MUJOCO_GL=egl python scripts/run_sim.py --config configs/sim_droid.yaml --set task.name=block_in_bowl` |
| The weights of a checkpoint (`hf auth login` first) | `hf download runwayml/praxis-1-droid --local-dir checkpoints/praxis-1-droid` (also `praxis-1-yam`, `praxis-1-base`) |
| An evaluation suite | `MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard --checkpoint CKPT --out OUT` |
| Compare two suite runs | `python scripts/compare_sim_eval.py OUT_A OUT_B` |
| The policy server | `python -m praxis.serve --config configs/serve_droid.yaml` |
| The offline action error | `python scripts/run_action_error_eval.py --checkpoint CKPT --vae VAE --text-encoder TEXT --data DROID --out OUT` |
| Judge and score autoregressive rollouts | `python scripts/score_autoregressive_eval.py judge --set droid --run RUN`, then `... score --set droid RUN_A RUN_B` |
| Inference speed, per stage, on a local GPU | `python scripts/benchmark_inference.py --config configs/serve_droid.yaml` |
| Latency of a running server | `python -m praxis.serve.latency --url ws://<host>:8000/api/v1/session` |
| A fine-tune of the base on your rig (full, 8 GPUs) | `scripts/compute_norm_stats.py`, `scripts/init_heads.py`, then `scripts/finetune.py`, all with `--config configs/finetune_base.yaml` |
| A LoRA fine-tune of `praxis-1-droid` on the DROID rig (one 80 GB GPU) | `torchrun --nproc_per_node 1 scripts/finetune.py --config configs/finetune_droid_lora.yaml` |

The extras of `pyproject.toml`: `kernels` (the text encoder kernels of the
validation results), `video`, `serve`, `sim`, `train`, `eval`, `nvfp4` (the
optional routed NVFP4 inference on SM120 GPUs), `dev`.

## Checks before a commit

```bash
ruff check --fix praxis scripts tests
ruff format praxis scripts tests
pytest tests/
```

- The unit tests run on the CPU and need no weights. Keep new tests that way:
  small tensors, `tests/tiny_model.py`, no `.cuda()`.
- The sim tests need the `sim` extra. With a CPU only, set
  `MUJOCO_GL=egl` and an EGL vendor library that renders on the CPU (Mesa).
- `tests/test_sim_isolation.py` checks the import rules of the sim package:
  only `praxis/sim`, the `tests/test_sim_*.py` files and the sim scripts
  import `praxis.sim`, and the core modules do not import the sim packages. A
  new sim script must be added to the list in that test.

## Rules that keep the results

- Output bits are part of the contract. praxis gives the same bits as the
  original implementation for the same inputs and seeds (see "Validation"
  in `README.md`). A change in a kernel, an operation order, a dtype, an
  autotune setting or a compile option can change the bits. If a change can
  change the bits, say so and give a check that compares the outputs before
  and after.
- The fused VAE norm kernel and the text encoder kernels use fixed settings,
  not autotuning (see "Determinism" in `README.md`). Keep them fixed.
- The settings of `configs/*_droid.yaml` are the recommended settings.
  Change them only with a measurement.
- The server and the sim use the same pipeline. Do not add a second code path
  for the same model.
- Do not commit weights, checkpoints, datasets or outputs. `outputs/` is
  ignored.
- The judge prompts and schemas of `praxis/eval/autoregressive/` are part of
  the published results. `tests/test_eval_autoregressive.py` pins the judge
  requests by hash. Change them only with a new judge validation.

## Measure a change in behavior

For a change that can change the actions (a checkpoint, a sampler setting, a
fine-tune), run the evaluation suites of `praxis/sim/EVAL.md` and compare
with the same episodes of the start:

```bash
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard --checkpoint NEW --out outputs/eval/new
python scripts/compare_sim_eval.py outputs/eval/start outputs/eval/new
```

Give the paired result (the difference with its 95 % interval, the episodes
won and lost), not only two success rates. With the same checkpoint, config
and packages, a suite run gives the same episodes on every run, so a rerun of
the start is a check of the setup.

## Fine-tuning

`praxis/train/README.md` describes the data, the objective, the recipe, the
memory, the overfit check and the step-0 checks of the training
objective. A change to the objective or the data must keep those
checks: run the overfit check (`praxis/train/README.md`, "Overfit check") and the
tests of `tests/test_train_*.py`.

## Writing style

Docs and comments use short sentences and one idea per sentence. Name the
file, the function and the value. Give the measured number and how it was
measured.
