# Fine-tuning

`scripts/finetune.py` fine-tunes a praxis checkpoint with PyTorch FSDP on one
node (`torchrun`). The objective needs the chunk-causal layout of the
checkpoint.

## Which recipe?

There are two recipes. Pick by your rig first, then by your GPUs:

| Your case | Recipe | Start | Trains | Config | GPUs |
|---|---|---|---|---|---|
| **A new rig** (another arm, other cameras, other poses or gripper) | full fine-tune | the base `praxis-1-base` + new action heads | all weights | `configs/finetune_base.yaml` | one node of 8 80 GB GPUs, about 15 hours measured |
| **The DROID rig, a new task** | LoRA | the single-arm `praxis-1-droid` | LoRA adapters of the action expert | `configs/finetune_droid_lora.yaml` | one 80 GB GPU (65 GiB measured), about 5 hours; more GPUs optional |

- A new rig needs the full fine-tune. Its action heads and its normalization
  are new, and the trunk must learn its cameras and its motion. LoRA keeps
  the trunk frozen, so it is not a recipe for a new rig.
- LoRA adapts a checkpoint that already acts on its rig. Use it only when
  your data comes from the DROID rig: the Franka with an exterior and a wrist
  camera, the DROID poses and gripper. It keeps the heads and the
  normalization of `praxis-1-droid`.
- The DROID rig with enough GPUs: either recipe fits. The full fine-tune
  from the base is the worked example of this file.

Both configs read the public Hugging Face copy of DROID (`cadene/droid_1.0.1`)
by default, so each step runs as written; point `data.source` at your data.
The sections "Fine-tune the base checkpoint" and "LoRA on praxis-1-droid"
give the steps of each recipe. The sections after them are common to both.

## Install

Install the `train` extra (`huggingface_hub`, `av` and `pyarrow`) and get the
weights ("Get the weights" in the main README):

```bash
pip install -e ".[train]"
```

To change one value without an edit of the file, use `--set`, for example
`--set train.steps=100`. To continue a run, give a save directory with
`--resume`.

## Fine-tune the base checkpoint

`praxis-1-base` is the pretrained model before any rig: the video backbone
and the action expert, trained on the videos and the actions of many
robots. It has no action heads and no normalization, so it does not run on
a robot. A fine-tune makes it a policy for one rig.

What the base has, and what it does not have:

| Part | In the base | Comes from |
|---|---|---|
| Backbone (encoder, decoder) | yes | the pretraining |
| Action expert blocks, taps, timestep and text layers | yes | the pretraining |
| Action input layer `action_expert.action_encoder.w1` | no | `scripts/init_heads.py` |
| Proprio input layer `action_expert.embed_proprio` | no | `scripts/init_heads.py` |
| Action output layer `action_expert.head` | no | `scripts/init_heads.py` |
| `normalization` of the config | no | `scripts/compute_norm_stats.py` |
| Camera canvas (tile size, antialias) | no | the `rig` section of the run config |

`load_config` refuses the base (`"embodiment": "base"`) with a message that
points here. `praxis.checkpoint.load_base_config` reads it.

### Your rig

This version supports the rig of the single-arm checkpoint:

- One arm. The action is the pose of the flange (or of a fixed point of
  the hand) and one gripper value. The model sees 10 values per row:
  `xyz`, the rotation as `rot6d` and the gripper.
- Two cameras: an exterior camera (the left tile) and a wrist camera (the
  right tile).
- 15 Hz control.

A rig with two arms, joint actions or other cameras needs new code in
`praxis/single_arm.py` and in the data module.

### Your data

The data module reads a local LeRobot v2.x tree (`data.source`), the same
layout as DROID. `data.spec` names its columns
(`praxis.train.data.DatasetSpec`; the defaults are the DROID columns):

| Key | Content |
|---|---|
| `views` | the video keys of the exterior and the wrist camera |
| `observed_pose`, `observed_gripper` | the measured pose and gripper of each row |
| `commanded_pose`, `commanded_gripper` | the commanded pose and gripper of each row (the action) |
| `rotation` | the rotation format of the pose columns: `euler_xyz` (extrinsic x-y-z, radians), `quat_xyzw` or `quat_wxyz` |
| `gripper_range` | the raw gripper values `[open, closed]`; the data module maps them to 0 and 1 |
| `success` | a boolean column; episodes with a false row are dropped. `null` keeps all episodes |
| `val_every` | episode `i` is a validation episode when `i % val_every == 0` |

A column can be a slice of a wider column: `commanded_pose: "action[0:7]"`
and `commanded_gripper: "action[7]"`.

The data must follow these rules. The data module checks the first two;
the other rules are your part:

- 15 fps: row `i` of an episode has the timestamp `i / 15` s, and frame `i`
  of each video belongs to row `i`. Resample other rates before.
- Each episode has a task in `meta/episodes.jsonl` and at least 102 rows
  (6.7 s).
- Poses are absolute, in metres, in one fixed base frame: the same frame
  for every episode and for the robot at serve time (the proprio is the
  absolute pose, and the relative actions are in the axes of this frame).
  The commanded pose is the target pose of the row (absolute, not a delta).
- The gripper is open at `gripper_range[0]` and closed at
  `gripper_range[1]`.
- The camera frames are RGB. With `antialias: true` (the template) the
  canvas code crops each frame to the DROID view aspect 16:9, resizes it to
  the fit size (234x416 for 416-wide tiles) and trims it to the tile; a 4:3
  camera loses about a quarter of its height. With `antialias: false` it
  crops to the tile aspect and resizes. Tile sides are multiples of 32.

### Steps

All steps read one run config. `configs/finetune_base.yaml` is the
template: set the paths, the `rig` section and `data.spec`.

1. Compute the normalization statistics. The script draws
   `rig.stats_windows` training windows (poses only, no video) and writes
   the quantiles and the moments to `rig.stats`
   (`praxis/train/norm_stats.py`):

   ```bash
   python scripts/compute_norm_stats.py --config configs/finetune_base.yaml
   ```

   Look at the printed vectors. The `xyz` quantiles of the actions are the
   motion within one chunk (1.6 s): about 0.15 to 0.2 m on DROID. The gripper
   always gets q01 = 0 and q99 = 1 (`gripper_range` maps it to [0, 1]).

2. Make the checkpoint of your rig. The script adds the six head tensors
   and the rig fields of the config, checks the result with a strict load,
   and writes `model.checkpoint` (about 55 GB of CPU memory):

   ```bash
   python scripts/init_heads.py --config configs/finetune_base.yaml
   ```

   The new heads are random: the init of a new `ActionExpert`. The output
   head starts small (std 1e-3), so the first predicted action velocities
   are near zero. The sampled actions of an untrained head are then near
   the start noise, not near a zero motion: do not serve the checkpoint
   before the fine-tune.
   `rig.heads` can name a safetensors file with the six tensors instead
   (the names of `BASE_HEAD_TENSORS`), for example the heads of an earlier
   fine-tune of the same rig saved with `safetensors.torch.save_file`.

3. Run the overfit check on your data (see "Overfit check"):

   ```bash
   torchrun --nproc_per_node 4 scripts/finetune.py --config configs/finetune_base.yaml \
     --set data.overfit_windows=10 --set objective.fixed_draws=true \
     --set objective.text_dropout=0.0 --set objective.proprio_dropout=0.0 \
     --set train.steps=600 --set train.grad_accum=1 --set train.lr=5e-5 \
     --set train.warmup_steps=10 --set train.lr_schedule=flat \
     --set train.weight_decay=0.0 --set train.save=false
   ```

   The pass rule is that of "Overfit check": the mean loss of the last 10
   steps is below 5 % of step 1. A fail with new heads points to the data
   first: the columns, the rotation format, the gripper range or the frame
   rate.

4. Run the fine-tune:

   ```bash
   torchrun --nproc_per_node 8 scripts/finetune.py --config configs/finetune_base.yaml
   ```

The saves under `output.dir` are full single-arm checkpoints. The policy
server, the inference script and the simulation load them like
`praxis-1-droid`: set their `checkpoint` path.

### The recipe of a fine-tune from the base

The heads are new and the rest of the model is pretrained. Two settings of
the `train` section give the heads time first (`TrainSettings`):

- `head_only_steps`: in the first steps only the heads train. The trainer
  drops the gradients of the other weights, so the gradient clip and the
  logged `grad_norm` use the norm of the heads, and the other weights, their
  AdamW moments and the MoE routing biases stay as they are. Their warmup
  and their cosine decay start after these steps.
- `head_lr_scale`: the heads train at this multiple of the learning rate,
  during the whole run.

A resume (`--resume`) must keep both values: the trainer refuses a save
made with other values.

`configs/finetune_base.yaml` leaves both off and trains all weights at one
learning rate: AdamW at a flat 2.5e-5 after 200 warmup steps, 32 windows per
step (8 GPUs with `grad_accum` 4), 5,000 steps. On 8 H100 it took 10.6 s per
step and 53.3 GiB per GPU, 15.7 hours in all. With the two settings on,
the log shows `lr` (the trunk) and `head_lr`. The trainer refuses them when
no head tensor trains.

### What to expect

We ran `configs/finetune_base.yaml` on 8 H100 (with `train.save_every=500` and
`data.source` a local copy of the hub dataset), on all of the hub
copy of DROID (`cadene/droid_1.0.1`, 320x180 videos, 69,932 train episodes
before the success filter), and ran the simulation suites of
`praxis/sim/EVAL.md` on its saves:

| | `hard` (352 episodes) | `droid_guard` (56 episodes) |
|---|---|---|
| step 2,000 | 54/352 (15.3 %) | 24/56 (42.9 %) |
| step 4,000 | 81/352 (23.0 %) | 35/56 (62.5 %) |
| step 5,000 (the last save) | 67/352 (19.0 %) | 35/56 (62.5 %) |
| `praxis-1-droid` | 199/352 (56.5 %) | 51/56 (91.1 %) |

- The recipe turns the base into a working policy in one node-day: the new
  heads start from nothing (action loss 1.31) and the policy picks, places
  and stacks in the simulation.
- It does not reach `praxis-1-droid`, and it is not meant to. That
  checkpoint was fine-tuned on DROID with far more compute, on the raw
  1280x720 videos. Most of the gap is on stacking and `peg_in_hole_6mm`.
- The training loss is flat from about step 2,000 (action loss 0.035 to
  0.040), while the simulation scores still move by several points from save
  to save. Judge a fine-tune by closed-loop runs, not by the loss.

### Serve a fine-tune

The fine-tuned checkpoint is a single-arm checkpoint: serve it with
`configs/serve_droid.yaml` and its path in `model.checkpoint`. The
`data.spec` transforms (`gripper_range`, `rotation`, the column slices) apply
only in training and are not in the checkpoint, so your robot client must
send what the model trained on:

- `descriptor: "flange"`. The server converts each pose from the frame of
  the client's `descriptor` (`RIG_FRAMES` in `praxis/single_arm.py`) to the
  model frame, and each command back. `"flange"` is the model frame itself,
  the frame your training data used. With no descriptor the server takes the
  FR3 hand frame (103.4 mm past the flange, turned -45 deg): your poses would
  then reach the model out of distribution, and your commands would come
  back moved;
- `robot_state.ee_pose`: `[x, y, z, qw, qx, qy, qz]` in metres, in the same
  base frame and for the same tool point as your data;
- `grip`: the gripper in [0, 1], 0 = open, after your `gripper_range` map;
- `image.exterior` and `image.wrist`: RGB frames of the two cameras.

Two more settings of the serve path are for the Franka of DROID:

- Do not send `robot_state.q` unless the arm is a Franka and your poses are
  its flange. With it, the server checks the pose against the Franka flange
  forward kinematics (5 mm) and refuses any other tool point.
- Set `session.control_mode: none` unless your arm takes the DROID
  impedance gains (`configs/serve_droid.yaml` sets `droid_impedance`).

With an antialiased canvas the server asks the client to bound each view
to the fit size (416x234) with its aspect kept, and then crops to 16:9.
Training crops the full frame to 16:9 first. For 16:9 cameras the two
orders give the same tile; for other aspects they do not (a 640x480
frame: about 4 levels of 255, mean absolute). With a non-16:9 camera,
crop it to 16:9 on the robot before you send it.

The commands come back in the same spaces: map the gripper back with your
`gripper_range` before you send it to the hand.

### Measure a fine-tune

- The action loss of the log. There is no script for a validation loss
  yet: `scripts/finetune.py` with `data.split=val` would train on it.
- Closed-loop runs on your robot: the only measure of a policy. For a
  DROID-like setup, the simulation suites of `praxis/sim/EVAL.md` also
  work.

## LoRA on praxis-1-droid (one 80 GB GPU)

`configs/finetune_droid_lora.yaml` adapts the single-arm checkpoint
`praxis-1-droid` to your task on the DROID rig: the Franka with an exterior
camera and a wrist camera, the frames and the poses of DROID. The weights of
the checkpoint stay frozen (`praxis/train/lora.py`). Each target `nn.Linear`
layer gets a trainable low-rank update `(alpha / rank) * B @ A`. `B` starts
at zero, so at step 0 the model is the checkpoint. The heads and the
normalization of the checkpoint stay, so your data must use the DROID
columns, frames and gripper. For another rig, use the full fine-tune of the
base ("Fine-tune the base checkpoint").

This section is a worked example: it teaches `praxis-1-droid` one
simulation task on one 80 GB GPU, end to end. The steps are the same for a
task on a real DROID rig; there you bring your own demonstrations instead of
step 1 and judge the result on the robot instead of in the simulation.

### Your data

The data is a LeRobot v2.x tree with the DROID columns (the defaults of
`data.spec`), in the flange frame of the checkpoint, at 15 fps. The rules
are those of "Your data" above, with two more:

- The cameras, the poses and the gripper must be those of DROID: LoRA keeps
  the heads and the normalization of `praxis-1-droid`.
- Keep the scenes you measure on out of the training data. The worked
  example records on scene seeds 1000 and up; the evaluation uses seeds 0
  to 15.

For the worked example, `scripts/record_sim_demos.py` records the data: a
scripted oracle solves the task in the simulation and the recorder writes
each successful episode as a LeRobot v2.1 tree with the DROID columns, so
the config reads it with only `data.source` changed. `--vary` draws the
oracle speed, approach height and timing per seed.

### Steps

The commands use `block_in_small_bowl`, one of the `hard` tasks of
`praxis/sim/EVAL.md`; any task of the simulation works the same way.

1. Record the demonstrations: 3,000 episodes, 1 to 2 hours on one GPU (the
   GPU renders the views). The recorder keeps the lowest successful seeds
   and skips the seeds the oracle fails.

   ```bash
   MUJOCO_GL=egl python scripts/record_sim_demos.py --task block_in_small_bowl \
       --episodes 3000 --first-seed 1000 --vary --workers 64 \
       --out data/sim_block_in_small_bowl
   ```

2. Measure the start: `praxis-1-droid` on the task, and on `droid_guard`,
   the seven easy DROID tasks that show forgetting.

   ```bash
   MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard --tasks block_in_small_bowl \
       --checkpoint checkpoints/praxis-1-droid --out outputs/eval/start --no-video
   MUJOCO_GL=egl python scripts/run_sim_eval.py --suite droid_guard \
       --checkpoint checkpoints/praxis-1-droid --out outputs/eval/start_guard --no-video
   ```

   Expect about **14 of 32** on `block_in_small_bowl` (most failures get
   close: mean progress 0.92) and **51 of 56** on `droid_guard`. Another
   machine can move these by a few episodes (the simulation is not bit-equal
   across machines), so compare only runs made on the same setup.

3. Train the adapters: 1,200 steps of 16 windows, about 5 hours on one
   H100. A save is 54 GB, so save every 200 steps (or 400) rather than the
   config's 100.

   ```bash
   torchrun --nproc_per_node 1 scripts/finetune.py --config configs/finetune_droid_lora.yaml \
       --set data.source=data/sim_block_in_small_bowl --set train.save_every=200 \
       --set output.dir=outputs/lora_block_in_small_bowl
   ```

4. Measure several saves (400, 800 and 1,200), each on the task and on
   `droid_guard`, and pair each with the start:

   ```bash
   MUJOCO_GL=egl python scripts/run_sim_eval.py --suite hard --tasks block_in_small_bowl \
       --checkpoint outputs/lora_block_in_small_bowl/step-001200/model \
       --out outputs/eval/step1200 --no-video
   MUJOCO_GL=egl python scripts/run_sim_eval.py --suite droid_guard \
       --checkpoint outputs/lora_block_in_small_bowl/step-001200/model \
       --out outputs/eval/step1200_guard --no-video
   python scripts/compare_sim_eval.py outputs/eval/start outputs/eval/step1200
   python scripts/compare_sim_eval.py outputs/eval/start_guard outputs/eval/step1200_guard
   ```

   Expect about **19, 16 and 20 of 32** for saves 400, 800 and 1,200 (14 at
   the start), and `droid_guard` at **54, 51 and 50 of 56**: the task goes
   up and the easy tasks hold. These are our runs on sampler seeds 0 and 1;
   on seeds 0 to 3 (64 episodes) the same saves gave 34, 35 and 38 of 64
   against 27 ("What to expect" below).

   Read the mean of the saves, not the best one: a single save on 32 or 64
   episodes moves by 10 to 25 points by chance. The stock `hard` suite runs
   sampler seeds 0 and 1 (32 episodes per task); our runs added seeds 2 and
   3 for 64 (`run_sim_eval.py` runs only the sampler seeds of the suite, so
   these runs used a copy of the `hard` suite with `sampler_seeds` 0 to 3).

The saves are full checkpoints (see "The `lora` section" below): serve one
with `configs/serve_droid.yaml` like `praxis-1-droid`.

### The `lora` section

| Key | Default | Meaning |
|---|---|---|
| `rank` | 0 | The rank of the update. 0 turns LoRA off (a full fine-tune). |
| `alpha` | 16.0 | The scale of the update is `alpha / rank`. |
| `dropout` | 0.0 | Dropout on the input of the update. |
| `targets` | `[expert]` | Group names of `LORA_TARGETS` or `fnmatch` patterns of `nn.Linear` module names. `expert` is the attention and MLP projections of the action expert. `backbone_attention` is the attention output projections of both towers. |
| `train_also` | `[]` | `fnmatch` patterns of parameter names that train in full next to the adapters, for example `action_expert.head.*`. |

- With `rank: 16` on the `expert` targets, 80 layers get adapters with
  5.5 M weights in total.
- **GPUs:** on one H100 80 GB (`--nproc_per_node 1`, 16 windows per step
  through `train.grad_accum: 16`) the peak is **65 GiB** (one GPU holds the
  whole frozen 27B checkpoint, so it needs an 80 GB card) and a step takes
  **14 to 15 s**, of which about 5.5 s encode the windows: about 5 hours
  for 1,200 steps. This is the default and the recipe of the results below.
  The adapters have few weights, so gradient accumulation adds little
  memory. More GPUs are an option (N GPUs x `grad_accum` windows per step):
  on 6 GPUs (96 windows per step) it needs about 27 GB per GPU and 21 s per
  step; on 4 GPUs with one window per GPU per step, 30 GiB per GPU and
  1.15 s per step. The larger batches were not measured in the simulation.
- The routers of the MoE layers are frozen, so the config sets
  `objective.moe_balance_speed: 0`.
- The adapters start at zero and need larger steps than full weights. The
  config uses a peak learning rate of 1e-4 and no weight decay.
- A save holds the merged weights in `model/`: a full praxis checkpoint with
  the names of the start checkpoint. `PraxisPolicy.from_pretrained`, the
  pipeline, the server and the simulation load it as they load any other
  checkpoint. Only the target layers differ from the start checkpoint.
- `trainable.safetensors` holds the adapters (and the `train_also`
  weights). A LoRA run resumes from the start checkpoint, these tensors and
  `optimizer/`: give the save with `--resume`, as for a full fine-tune.

### What to expect

We ran the steps above on four `hard` tasks, one adapter per task, each on
one H100, with 64 episodes per task (sampler seeds 0 to 3). The number is
the mean of saves 400, 800 and 1,200 minus `praxis-1-droid`, paired per
episode, with a 95 % bootstrap interval.

| Task | `praxis-1-droid` | step 400 | step 800 | step 1,200 | mean of the saves − start | `droid_guard`, start → mean of the saves |
|---|---|---|---|---|---|---|
| `block_in_small_bowl` | 27/64 | 34/64 | 35/64 | 38/64 | +13.5 points [−1.0, +28.1] | 51/56 → 51.7/56 |
| `two_blocks_in_bowl` | 33/64 | 40/64 | 40/64 | 41/64 | +11.5 points [−3.1, +26.0] | 51/56 → 49.7/56 |
| `peg_in_hole_6mm` | 30/64 | 25/64 | 42/64 | 37/64 | +7.3 points [−7.8, +21.9] | 51/56 → 50.3/56 |
| `stack_three_blocks` | 12/64 | 13/64 | 11/64 | 11/64 | −0.5 points [−10.4, +8.9] | 51/56 → 53.0/56 |
| **all four tasks** (256 episodes) | | | | | **+7.9 points [+1.3, +14.8]** | |
| **the first three** (192 episodes) | | | | | **+10.8 points [+2.4, +19.3]** | |

- **The adapters help on three of the four tasks: about +10 points**, for
  example `block_in_small_bowl` from 42 % to 53 to 59 %, and 8 of the 9
  saves of those tasks are above the start. Pooled over the tasks the
  interval excludes 0; the pooling was chosen after the evaluation (the
  rule fixed before it was per task), so read it as a post-hoc check. One
  task alone on 64 episodes is too small to show a gain of this size, so
  each per-task interval still includes 0.
- `stack_three_blocks`, the hardest of the four (19 % at the start), does
  not move. A task that far from the checkpoint needs more than adapters on
  3,000 demonstrations.
- **The easy tasks hold:** `droid_guard` stays at 48 to 54 of 56 for every
  save (51 at the start).
- **It does not overfit the demonstrations.** On 50 held-out demonstrations
  (other scene seeds) the action loss falls through step 1,200 and stays
  within 6 % of the loss on the training demonstrations, for every task.
  The success does not follow the loss: the loss falls at every save while
  `peg_in_hole_6mm` goes 25, 42, 37 of 64. As for the full fine-tune, judge
  the adapters by closed-loop runs, not by the loss.

## Data

The data module reads DROID in the LeRobot v2 layout (`data.source`): a
Hugging Face dataset id or a local directory. The configs use the public
Hugging Face copy `cadene/droid_1.0.1`, read from the hub, so anyone can
reproduce a run. Its videos have 320x180 frames; the checkpoints were
trained on the raw DROID videos (MP4 files with 1280x720 frames), so with
the hub copy the tiles are upscales with less detail.

With many loader processes (8 GPUs x 12), reads from the hub hit its rate
limit for anonymous users. For a long run, download the dataset first
(`huggingface-cli download cadene/droid_1.0.1 --repo-type dataset
--local-dir <dir>`, about 290,000 files) and set `data.source` to the
directory.

For the full resolution, make a local tree with the raw videos as follows:

1. Download the metadata and the parquet files of the Hugging Face copy.
   Do not download its videos:

   ```bash
   hf download cadene/droid_1.0.1 --repo-type dataset --local-dir droid_hires \
       --include "meta/info.json" --include "meta/episodes.jsonl" --include "data/*"
   ```

2. Download the raw release of DROID 1.0.1 (see the DROID project page).
   For each successful episode, you need these files:
   - `metadata_<id>.json`;
   - `trajectory.h5` (only for the check in step 3);
   - the MP4 file of the exterior camera 1, `recordings/MP4/<serial>.mp4`
     with the serial `ext1_cam_serial` of the metadata;
   - the MP4 file of the wrist camera, with the serial `wrist_cam_serial`.

   The MP4 files of 51,235 episodes take about 0.7 TB.

3. Find the raw episode of each episode of the Hugging Face copy. The
   Hugging Face copy does not name the raw episode. These values identify
   it:
   - The first row of `camera_extrinsics.exterior_1_left` and of
     `camera_extrinsics.exterior_2_left` in the parquet file is equal (in
     float32) to `ext1_cam_extrinsics` and `ext2_cam_extrinsics` of the
     metadata.
   - The parquet file has one row less than `trajectory_length` of the
     metadata.
   - When more than one raw episode matches, the language annotations of
     the raw release select one. Leave out the episodes that you cannot
     match to exactly one raw episode.

   Check each match: the poses of `trajectory.h5` (without its last row)
   agree with the poses of the parquet file to float32 precision.

4. Put a link to each of the two MP4 files of a matched episode at the
   video path of the episode. `video_path` of `meta/info.json` gives the
   path:

   ```
   droid_hires/videos/chunk-NNN/observation.images.exterior_1_left/episode_EEEEEE.mp4 -> <raw episode>/recordings/MP4/<ext1 serial>.mp4
   droid_hires/videos/chunk-NNN/observation.images.wrist_left/episode_EEEEEE.mp4      -> <raw episode>/recordings/MP4/<wrist serial>.mp4
   ```

   `EEEEEE` is the episode index and `NNN` is the episode index divided by
   1000.

The dataset uses only the episodes that have the parquet file and the two
videos. Thus a part of the tree is sufficient. Our tree held 51,235
successful episodes: 49,636 in the pool of the `train` split and 993 in
the pool of the `val` split. It did not hold the 1,153 matched episodes
with a wrist video of 672x376 frames.

The raw MP4 files have time stamps at 50 to 60 fps and one frame for each
15 Hz step. Thus the data module reads frame i as the i-th decoded frame
(decode order). It decodes each file from the start, because many raw
files have only one key frame. This decode needs CPU time:
`data.num_workers` sets the number of loader processes for each GPU.

The data recipe is the DROID recipe of the checkpoint. The canvas follows
the checkpoint config (see "Canvas"). For `praxis-1-droid` the canvas of a 1280x720
frame is close to its training canvas but not bitwise equal (see
"Validation" in the top README):

- Episodes: the successful episodes with a language instruction and at
  least 102 frames (the window and 0.5 s more).
- Split: the episodes with `episode_index % 50 == 0` are the `val` split
  (`data.split`). All other episodes are the `train` split.
- Window: 93 frames at 15 Hz in 4 chunks of 21, 24, 24 and 24 frames.
- Start: the start time is uniform in `[0, D - 6.2 s]`, with `D` the
  duration of the episode video. Then it snaps down to the frame grid.
- Canvas: each camera gets its own tile (the tile size of the checkpoint
  config) with the tile code of the serve path. The code crops the frame
  to the tile aspect at the centre and resizes it (bilinear). The resize
  uses antialias when the checkpoint config has `video.antialias: true`,
  as for `praxis-1-droid`. Without the key, or with `false`, it has no antialias. The
  same switch sets the resize of inference. The exterior camera 1 goes left
  and the wrist camera right (two tiles wide, 256x832 or 224x832).
- Poses: the flange pose of DROID, with no turn. The proprio is the
  observed pose at the start of each chunk.
- Actions: the commanded pose of each frame relative to the observed pose
  at the start of its chunk. The gripper is absolute.
- Normalization: the statistics of the checkpoint config.
- Prompt: the language instruction of the episode, without changes.

Each GPU draws its windows from its own part of the episodes. The docstring
of `praxis/train/data.py` gives the full data recipe.

We compared the windows with the data code of the original training. The
frames of 25 windows were bitwise equal. The actions and the proprio of 888
windows were bitwise equal. The prompts of all 51,235 episodes were equal.

## The loss of a window

`praxis/train/objective.py` computes the loss of one training window. A
window holds 4 chunks of 6 latents. The trunk reads the registers, the
anchor frame, the text, the 4 noised chunks and the clean copies of chunks
0 to 2. A chunk-causal mask lets each noised chunk see the clean copies of
the chunks before it. Thus one window trains every context depth from 0 to
3. The clean copies get the timestep 0 and no loss.

- Video noise has two stages. First the window gets one timestep and one
  noise. Then each chunk gets its own timestep and the chunk is noised
  again. The video timesteps use the shift 5.0, so most of them are near
  full noise. The target is `noise - clean` of the first stage.
- Each chunk gets an action timestep. The action timesteps are uniform and
  independent of the video timesteps. The target is `noise - clean`.
- With the probability 0.1 the text of the window becomes the encoded
  blank caption `" "`. This trains the unconditional branch of the text
  guidance. With the probability 0.1 the proprio of all chunks becomes
  zero.
- The rotary positions of the anchor, the text and the video of the trunk,
  and of the video of the decoder, move by random offsets. The inference
  uses no offset.
- The loss is `1.0 * video loss + 1.0 * action loss + 1e-4 * MoE balance
  loss`. The MoE balance loss is the sequence balance loss of the router.
- The action loss is the mean of the squared velocity error over the
  channels and then over the tokens, in float32. When the action loss of a
  window is not finite or is above 5.0, the window gets the action loss 0.
- After each optimizer step, the MoE routing biases move by 1e-3 toward an
  equal load of the experts.
- The MoE layers dispatch the tokens to the experts in chunks of 65,536
  tokens.

The `objective` section of the configs holds these values (the LoRA config
sets `moe_balance_speed: 0`, see "LoRA on praxis-1-droid"). The docstring of the module gives each term in detail.

## The recipe

The training loop:

- Each GPU holds a shard of the policy. The autoencoder and the text
  encoder are frozen. Each GPU holds a full copy of them.
- The optimizer is AdamW with betas (0.9, 0.95), eps 1e-8 and weight decay
  1e-4 on all trained parameters (the LoRA config: 0). With the learning
  rate of `configs/finetune_base.yaml` (2.5e-5), `lr * weight_decay` is less
  than 3e-8. Then the float32 decay factor is
  1.0, and the decay has no effect.
- The weights and the AdamW moments stay in bfloat16. The update runs in
  float32. Then a stochastic rounding writes it back to bfloat16. A plain
  rounding loses most updates of this size.
- The gradient clip is 1.0 on the norm of all gradients.
- Activation checkpointing is on (`train.activation_checkpointing`).

The values of each recipe are in its section: "The recipe of a fine-tune
from the base" and "LoRA on praxis-1-droid". `train.lr_schedule: flat` keeps
the learning rate constant after the warmup; `cosine` decays it to
`train.min_lr` at the last step.

## GPUs and memory

The weights, the gradients and the AdamW moments take 8 bytes per
parameter: 203 GiB for the 27.2B parameters. FSDP divides them by the number
of GPUs. Each GPU also holds the activations, a copy of the autoencoder and a
copy of the text encoder. We measured these values on H100 GPUs with 80 GB,
with `praxis-1-droid-k` (`praxis-1-droid` has the same model and a smaller canvas):

| GPUs | Gradient accumulation | Windows per step | Peak memory per GPU | Time per step |
|---|---|---|---|---|
| 4 | 1 | 4 | 73.0 GiB | 3.7 s (the overfit check: no data loading) |
| 6 | 16 | 96 | 64.1 GiB | 43 s (about 11 s of it is the wait for data) |
| 8 | 4 | 32 | 53.3 GiB | 10.6 s (`configs/finetune_base.yaml` from `praxis-1-base`, the 320x180 hub videos; 2.4 s of it is data) |

The 6-GPU row used 12 loader processes per GPU and the raw 1280x720 videos.

The script turns on the expandable segments of the CUDA memory allocator.
Without them, the GPU memory can fragment, and a run can stop when the
memory is full. To use other allocator settings, set
`PYTORCH_CUDA_ALLOC_CONF`.

## Gradient accumulation

One micro-step reads one window on each GPU. `train.grad_accum` sets the
number of micro-steps of one optimizer step. Thus one step uses
`N * train.grad_accum` windows on N GPUs. `configs/finetune_base.yaml` uses
32 windows per step on 8 GPUs (`grad_accum` 4); on 16 GPUs, set
`train.grad_accum=2`.

After each micro-step, FSDP reduce-scatters the gradients and adds them to
the sharded gradients. The sharded gradients stay in bfloat16 between the
micro-steps. From the second micro-step on, the gradients of the earlier
micro-steps stay in memory during the whole backward pass. Thus gradient
accumulation needs more memory than one micro-step. On 4 GPUs one
micro-step already takes 73.0 GiB per GPU. Thus use 6 or more GPUs for
gradient accumulation.

With `train.no_sync: true`, each GPU keeps the full float32 gradients until
the last micro-step. For this model they take 109 GB per GPU. Thus use
`no_sync` only for small models.

## Outputs

The script adds the metrics of each step to `output.dir/metrics.jsonl`. It
does not delete the lines of an earlier run in the same directory. Thus give
each new run its own `output.dir`. Every `train.save_every`
steps and after the last step, it writes a save directory `step-NNNNNN`:

- `model/` is a praxis checkpoint (54 GB). `PraxisPolicy.from_pretrained`
  loads it. To run the inference with it, set `model.checkpoint` in
  `configs/inference_droid.yaml`.
- `optimizer/` (109 GB) and `trainer_state.json` hold the state to continue
  the run. To continue, give the save directory with `--resume`.

One save takes 163 GB. The script does not delete old saves.
`configs/finetune_base.yaml` (5,000 steps, a save every 1,000 steps) writes 5
saves, about 815 GB.
Make sure that the disk has space for all saves of the run, or set a larger
`train.save_every`. Only `--resume` reads `optimizer/`. Thus you can delete
`optimizer/` of the old saves and keep `model/`. `train.save: false` turns
off all saves, also the save after the last step.

## Overfit check

Before a long run, do the overfit check. It trains on the same 10 windows
again and again, so the model learns them by heart:

```bash
torchrun --nproc_per_node 4 scripts/finetune.py --config configs/finetune_base.yaml \
  --set data.overfit_windows=10 --set objective.fixed_draws=true \
  --set objective.text_dropout=0.0 --set objective.proprio_dropout=0.0 \
  --set train.steps=600 --set train.grad_accum=1 --set train.lr=5e-5 \
  --set train.warmup_steps=10 --set train.lr_schedule=flat \
  --set train.weight_decay=0.0 --set train.save=false
```

The overrides change these values of `configs/finetune_base.yaml`:

- `data.overfit_windows: 10`: rank 0 draws the first 10 windows of the
  `train` split and encodes them once. All GPUs then use only these windows.
  On N GPUs, rank r at micro-step k of step s uses the window
  `((s * grad_accum + k) * N + r) % 10`. The run reads no more files.
- `objective.fixed_draws: true`: the random values of a window come from
  the seed and the window index, not from the step. These values are the
  timesteps, the noise, the dropout values and the rotary offsets. Thus each
  use of a window gets the same values.
  With new values in each step, the flow-matching loss cannot go to 0.
- No text dropout, no proprio dropout and no weight decay.
- 600 steps with one window per GPU in each step, a flat learning rate of
  5e-5 and 10 warmup steps.
- `train.save: false`: the check writes no checkpoint.

The check finds errors in the data, the objective, the trainer and the
hardware. Pass rule:

- The loss goes to about 0. The mean loss of the last 10 steps is below 5 %
  of the loss of step 1.
- The loss and the gradient norm are finite in each step.
- All GPUs complete all steps.

If the loss still falls at the last step, run more steps
(`--set train.steps=1200`).

We ran the check on 4 H100 GPUs from `praxis-1-droid-k`, a single-arm checkpoint with
trained heads. From the base, the new heads start with a larger action loss,
so the loss of step 1 is larger; the pass rule is the same.

| | 4 H100 GPUs |
|---|---|
| Windows per step | 4 |
| Loss at step 1 | 0.1196 |
| Mean loss of the last 10 steps | 0.0043 (3.6 % of step 1) |
| Video loss, mean of the first 5 and the last 20 steps | 0.141 to 0.0039 |
| Action loss, mean of the first 5 and the last 20 steps | 0.038 to 0.0005 |
| Time per step (median) | 3.7 s (39 minutes in total) |
| Peak memory per GPU | 73.0 GiB |

The mean loss of each block of 50 steps was lower than the mean loss of the
block before it. All losses and gradient norms were finite, and all GPUs
completed all steps. After 300 steps the mean loss of the last 10 steps was
still 10 % of step 1. Thus the check uses 600 steps. The loss of step 1 is
the mean over the windows of step 1 only. It changes with the number of
GPUs.

## Validation of the training step

We compared the training step of praxis with the original training code
on one H100, with `praxis-1-droid-k`. Both sides used the same windows and
the same random values (timesteps, noise, dropout and rotary offsets). Both
sides used the Triton kernel of flex attention in the trunk, activation
checkpointing and deterministic algorithms. The test computed the gradients
of 277 parameter tensors: the registers, the timestep projections, the input
and output projections of both towers, encoder blocks 0, 2 and 13, decoder
block 1 with its modulation, and the full action expert.

There were 15 cases:

- 3 synthetic windows: a normal window, a window with the blank caption and
  a window with the zero proprio.
- 12 DROID windows: 10 windows of the `train` split, and one of them also
  with the blank caption and with the zero proprio.

| Value | Result |
|---|---|
| Loss, video loss, action loss and MoE balance loss | bitwise equal in 15 of 15 cases |
| Rotary positions of the trunk and of the decoder | equal in 15 of 15 cases |
| Update of the routing biases | bitwise equal in 17 of 17 MoE layers, all cases |
| Gradients | 276 or 277 of 277 tensors bitwise equal in each case |

The one other tensor is the weight of the text input projection. Only the
sign of some zeros is different (`-0.0` and `+0.0`). This difference has no
effect on the AdamW update.

The trunk uses the `flash` attention backend by default, also in training.
The backward pass of this kernel is not deterministic on the H100. Thus a
run with it does not give the same gradient bits two times, also for the
original training code. With the `flash` backend on both sides, the losses
were bitwise equal and the 200 gradients of the action expert were bitwise
equal. The differences of the trunk gradients were in the range of the
differences between two runs of the original training code.

## Validation of the training loop

- The CPU tests compare the trainer with plain training of a small random
  policy with the chunk-causal layout. The flex attention has no backward
  on the CPU. Thus these tests use a test loss over the parts of the policy
  that run on the CPU. They run FSDP on 2 ranks, gradient accumulation of 2
  micro-steps, and both. In float32 the losses and the gradient norms agree
  within a relative 1e-4. The difference of the weights is below 1e-4 of
  the change of the weights. A save loads strictly, and a resumed run
  continues bit for bit.
- On 6 H100 GPUs with 96 windows per step, two runs with the same seed gave
  the same bits of the loss and of the gradient norm in step 1. A run of 20
  steps had finite losses and gradient norms in all steps.

## Known differences

These are the differences from the original training code:

- Optimizer: this script uses AdamW on all weights (see "The recipe"). It
  does not give the same bits as the original optimizer.
- Gradient reduce: this script reduces the gradients in float32 by default
  (`train.reduce_dtype`).
- Video encoder: the original implementation uses another kernel of the
  VAE encoder. The latents differ by 1 or 2 bfloat16 units in the last
  place in a part of the values. The pipeline and the simulation use the
  same encoder as the fine-tune.
- Poses: the parquet files keep the poses in float32. The original data
  code read float64 poses. The normalized actions and proprio of a window
  differ by less than 2e-6.
- Split: the checkpoint was trained on most of the `val` episodes of this
  split. Thus a loss of the checkpoint on the `val` split is not a held-out
  loss. Use it to compare saves of one run.
- `praxis/train/data.py` gives the other differences of the data.
