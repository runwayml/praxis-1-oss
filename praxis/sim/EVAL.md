# Evaluation of the single-arm checkpoint in simulation

This page describes the fixed evaluation scenarios of the single-arm DROID
checkpoint, how to run them and how to compare two checkpoints. The
scenarios are in `packages/praxis-sim/src/praxis_sim/eval_suites.py`. `scripts/run_sim_eval.py` runs
a suite and `scripts/compare_sim_eval.py` compares two runs.

## The scenarios

An episode is one `(task, scene seed, sampler seed)` unit:

- The task sets the objects, the prompt, the success check, the dense
  progress value and the time limit. Each suite uses the default prompt and
  the default time limit of each task.
- The scene seed places the objects. In the color tasks it also selects the
  target color, and in `move_block` the direction. The scene seeds are
  0 to N - 1.
- The sampler seed is `session.seed`. Chunk `k` of the episode with scene
  seed `s` samples with the seed `session.seed + 1000 * s + k`.

The scene is the DROID rig of `configs/sim_droid.yaml`: a Franka arm with a
Robotiq 2F-85 gripper, the exterior camera and the wrist camera at 512x288,
15 Hz control. An episode stops 1 s after the first success, or at the time
limit. Success is the success check of the task. Progress is a value from 0
to 1 that is 1 at success. A scripted oracle solves each task in 18 or more
of 20 scenes (`tests/test_sim_suite.py`).

| Suite | Tasks | Scene seeds | Sampler seeds | Episodes | Use |
|---|---|---|---|---|---|
| `droid` | the 7 DROID tasks | 16 | 0, 1 | 224 | regression check, full |
| `droid_guard` | the 7 DROID tasks | 8 | 0 | 56 | regression check during a run |
| `hard` | the 11 harder tasks | 16 | 0, 1 | 352 | the main measure |

The checkpoint solves most episodes of the 7 DROID tasks. Thus a
fine-tune can show only a small gain there. The harder tasks were made so
that the checkpoint solves about 20 to 60 % of the episodes of each
task. Thus they can show a gain or a loss.

### The 7 DROID tasks

| Task | Prompt | Success | Time limit | Checkpoint |
|---|---|---|---|---|
| `lift_cube` | Pick up the red cube. | the cube is 5 cm or more above its rest height for 0.25 s | 11.0 s | 30/32 |
| `cube_on_plate` | Put the red cube on the blue plate. | the cube is on the plate, released | 16.0 s | 31/32 |
| `block_in_bowl` | Put the red block in the bowl. | the block is in the bowl, released | 20.6 s | 32/32 |
| `block_out_of_bowl` | Remove the red block from the bowl. | the block is on the table, 13 cm or more from the bowl center | 20.6 s | 28/32 |
| `move_block` | Move the red block to the left. (or right) | the block moved 12 cm or more in that direction | 16.0 s | 30/32 |
| `stack_block` | Put the red block on top of the green block. | the red block rests on top of the green block | 20.6 s | 27/32 |
| `press_button` | Press the red button. | the button cap is 6 mm or more down | 11.0 s | 1/32 |

### The 11 harder tasks

| Task | Prompt | What makes it hard | Time limit | Checkpoint |
|---|---|---|---|---|
| `cube_in_tray` | Put the red cube in the tray. | an 8 cm open tray, the cube far from it | 14.2 s | 18/32 |
| `cube_on_color` | Put the red cube on the blue square. | three 6 cm color squares, the seed names the target | 14.2 s | 19/32 |
| `pick_color_block` | Pick up the red block. | the named block among distractor blocks | 7.8 s | 16/32 |
| `color_block_in_bowl` | Put the red block in the bowl. | the named block among distractors, a small bowl | 11.0 s | 16/32 |
| `stack_color_blocks` | Put the red block on the green block. | two named blocks among five colors | 17.4 s | 18/32 |
| `peg_in_hole_6mm` | Put the blue peg in the square hole. | a 3 cm square peg, 6 mm clearance per side | 25.4 s | 8/32 |
| `block_in_small_bowl` | Put the red block in the bowl. | a small bowl, wider start positions | 14.2 s | 20/32 |
| `cube_on_small_plate` | Put the red cube on the blue plate. | a small plate, wider start positions | 14.2 s | 20/32 |
| `stack_small_block` | Put the red block on top of the green block. | a small base block, the top block near its center | 20.6 s | 17/32 |
| `two_blocks_in_bowl` | Put the red and the green block in the bowl. | two pick-and-place steps | 27.0 s | 14/32 |
| `stack_three_blocks` | Stack the blocks. | three blocks in one tower, any order | 31.8 s | 9/32 |

The color tasks name the target in the prompt, so the prompt of the table is
one example. The checkpoint column gives the successes of
`praxis-1-droid` with `configs/sim_droid.yaml` (16 scenes x 2
sampler seeds). `peg_in_hole_4mm` and `peg_in_hole_2mm` are registered but
not in a suite: the checkpoint solves 2 of 32 episodes at 4 mm.

## Run a suite

The suite runner needs the `sim` extra and one GPU per process. A process
loads the model once and runs its episodes in turn. The harder suite takes
about 45 minutes on two H100 GPUs (one sampler seed on each).

```bash
pip install -e ".[sim]"   # pulls in the praxis-sim package (see praxis/sim/README.md)

# One GPU, both sampler seeds.
MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
    --checkpoint checkpoints/praxis-1-droid --out outputs/eval/start

# Two GPUs: one sampler seed on each, into the same output directory.
for SS in 0 1; do
  CUDA_VISIBLE_DEVICES=$SS MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard \
      --checkpoint checkpoints/praxis-1-droid --out outputs/eval/start \
      --sampler-seeds $SS &
done
wait
```

- Each episode writes `<out>/suite/<task>/episode_NNN.json` and `.mp4` for
  sampler seed 0, and `<out>/suite_s1/...` for sampler seed 1. The JSON has
  the success, the progress, the ticks and the policy calls.
- A rerun skips the episodes that have a JSON file. Thus a stopped run
  continues where it stopped.
- `--shard I/N` runs every N-th episode from episode I, for more processes
  per sampler seed.
- `--tasks a,b` runs a part of the suite. `--no-video` skips the MP4 files.
- `--policy scripted` runs the scripted oracle (no GPU, no weights).
- `--set key=value` changes the config, for example
  `--set sampler.num_steps=4`. A changed config gives other episodes, so
  compare only runs with the same config.

With the same checkpoint, config and packages, a run gives the same episodes
on every run.

## Compare two checkpoints

```bash
python scripts/compare_sim_eval.py outputs/eval/start outputs/eval/finetuned
```

The script pairs the episodes of the two runs by task, scene seed and sampler
seed. For each task it gives the successes of A and B, the episodes that B
won (A fails, B succeeds) and lost, the exact McNemar p and the mean
progress. For the total it also gives the mean difference with a 95 %
bootstrap interval (10,000 resamples). The result is "clearly above" or
"clearly below" only when the interval does not include 0.

Read the result with these points:

- Compare the same episodes. A paired comparison is more sensitive than a
  comparison of two success rates.
- One save of a fine-tune can move by about 5 points on the harder suite by
  chance. Read the trend over several saves, not one save.
- If you pick the best of several saves, its result is optimistic. Fix the
  rule that picks the best save before the first evaluation, and give the
  mean of the later saves too.

## Use the suites during a fine-tune

- Run `droid_guard` and `hard` on each save. On 2 H100 GPUs the two suites
  take about 50 minutes.
- Pick the best save by the success on `hard`. A save must not be clearly
  below the start on `droid_guard`.
- Stop the run after two saves in a row that are clearly below the best save.

## Use the tasks for training

The progress value is a dense reward from 0 to 1 and the success is a
terminal signal. If you train on these tasks, train on other scene seeds, for
example 1000 and higher. Keep scene seeds 0 to 15 for the evaluation, and
say so with the results.

## Result: the DROID fine-tune of the checkpoint

We fine-tuned the checkpoint on DROID with
`configs/finetune_droid.yaml` (see `praxis/train/README.md`) and ran the two
suites on each save. The best save by the rule above is step 600:

| Suite | Start | Step 600 | B - A, 95 % interval | Won / lost | McNemar p |
|---|---|---|---|---|---|
| `hard` | 175/352 (49.7 %) | 213/352 (60.5 %) | +10.8 points [+4.5, +17.1] | 88 / 50 | 0.0015 |
| `droid_guard` | 46/56 | 46/56 | level | | |

The mean of saves 200 to 800 on `hard` is 55.5 %. Per task, step 600 gains on
`stack_small_block` (+9), `cube_on_color` (+8), `cube_on_small_plate` (+6),
`pick_color_block` (+6), `two_blocks_in_bowl` (+5) and
`color_block_in_bowl` (+4). It does not gain on `stack_three_blocks` (+0),
`peg_in_hole_6mm` (-1) and `block_in_small_bowl` (-2).
