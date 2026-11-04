# Closed-loop simulation of the single-arm rig

This package runs the single-arm checkpoint `praxis-1-droid` in
closed loop on simple MuJoCo tasks. Use it to see what the model does when
it controls a robot arm. You can watch the run live in a web browser, also
on a cloud machine without a screen. After the run, the same page plays the
MP4.

The loop of one episode:

1. Reset the task. The seed sets the cube position and the sampler seeds.
2. Render the exterior camera and the wrist camera. Read the arm pose and
   the gripper value.
3. Call the policy. It returns one chunk of command rows: 21 rows for the
   first call, 24 rows for each call after it.
4. Execute each row for one tick of 1/15 s. Render the views after each
   tick.
5. Go to step 3. Stop at success or at the time limit.

The loop executes all the rows of a chunk before the next call. The model
then gets the 24 executed frames plus the current frame. It keeps up to
`session.max_context_chunks` executed chunks as context (3 in
`configs/sim_droid.yaml`).

The package uses only the core praxis API (`praxis.pipeline`,
`praxis.single_arm`). The core package does not use this package.

## Install

```bash
pip install -e ".[sim]"      # mujoco 3.16, imageio, imageio-ffmpeg
```

Use the `sim` extra in addition to the install steps of the top README. The
scripted policy needs no GPU and no model weights. The model policy needs the
weights of the top README and one GPU with 80 GB of memory.

## Assets and licenses

The robot models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at
commit `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`:

| Model | Menagerie directory | License |
|---|---|---|
| Franka Emika Panda arm, no hand (`panda_nohand.xml`) | `franka_emika_panda` | Apache-2.0 |
| Robotiq 2F-85 gripper (`2f85.xml`) | `robotiq_2f85` | BSD-2-Clause, Copyright (c) 2013, ROS-Industrial |

This repository does not contain the model files. At the first run, the code
downloads the 71 files (36 MB) of the two models from the pinned commit. It
checks the SHA-256 of each file (`praxis/sim/assets.py`). The download
includes the LICENSE file of each model. The cache directory is
`$PRAXIS_SIM_CACHE`, or `$XDG_CACHE_HOME/praxis/sim`, or
`~/.cache/praxis/sim`.

On a machine without a network, clone Menagerie at the pinned commit on a
different machine and copy it. Then set `scene.asset_source` to its
directory. The code copies the files and checks the same SHA-256 values.

The scene code makes the table, the cube, the lights and the cameras. It
attaches the gripper to the arm flange with `mujoco.MjSpec`.

## Headless rendering

MuJoCo renders offscreen with OpenGL. On a Linux machine without a screen,
the package selects EGL (`MUJOCO_GL=egl`) when `MUJOCO_GL` has no value.

- **NVIDIA GPU.** Some containers have the NVIDIA EGL library but not its
  glvnd vendor file. Then EGL uses the software renderer (llvmpipe) and
  gives no warning. The package writes a vendor file for the NVIDIA
  library into the cache directory and selects it. The run log shows the
  renderer: `MuJoCo renders with NVIDIA ...`. With llvmpipe, the log shows a
  warning, and the loop is approximately 20 times slower.
- **GPU selection.** `render.egl_device` is the physical GPU of the
  renderer. The EGL device index counts all GPUs and ignores
  `CUDA_VISIBLE_DEVICES`. With `null`, the run uses the first entry of
  `CUDA_VISIBLE_DEVICES`.
- **Black frames.** On some GPUs, an EGL renderer that shares the GPU
  with CUDA work returns black or corrupt frames and gives no error. The
  loop checks each model view. At the first black or flat view it logs a
  warning. After 15 bad ticks in a row it stops the run. Then set
  `render.egl_device` to a GPU that does not run the model. The
  `bad_view_ticks` field of each episode JSON counts the bad ticks.
- **No GPU.** Set `MUJOCO_GL=egl` with the Mesa EGL library (llvmpipe), or
  set `MUJOCO_GL=osmesa` with the OSMesa library installed. Both are slow,
  but the scripted policy and the tests work.

Set `MUJOCO_GL` before Python imports `mujoco`. The package reads it when
you import `praxis.sim`.

## Run

```bash
# Scripted pick (no weights): checks the scene, the renderer, the video and the viewer.
python scripts/run_sim.py --config configs/sim_scripted.yaml

# The DROID checkpoint in closed loop on the lift_cube task.
python scripts/run_sim.py --config configs/sim_droid.yaml

# The same checkpoint on the block_in_bowl task, with the prompt of the task.
python scripts/run_sim.py --config configs/sim_droid.yaml \
    --set task.name=block_in_bowl --set task.prompt=null

# Change single values on the command line.
python scripts/run_sim.py --config configs/sim_droid.yaml \
    --set task.prompt="Lift the red block." --set "episodes.seeds=[0,1,2,3]"
```

The script prints the URL of the live view and the output paths:

```
live view: http://127.0.0.1:8000/
episode seed 0: success ..., ticks ..., calls ..., mean call ... s, ...
outputs: /path/to/outputs/sim_droid
  video seed 0: /path/to/outputs/sim_droid/episode_000.mp4
  summary: /path/to/outputs/sim_droid/summary.json
```

Outputs in `output.dir` (a relative path starts at the current directory):

| File | Content |
|---|---|
| `config.yaml` | the settings of the run, after the overrides |
| `episode_<seed>.mp4` | the human view of the episode (H.264, 15 fps) |
| `episode_<seed>.json` | the result: success, ticks, calls, time per call, progress metrics, tracking errors |
| `episode_<seed>.npz` | per tick: observed pose and grip, commanded pose and grip, cube position, call index |
| `summary.json` | the result of all the episodes |

The human view has the overview camera, the exterior camera and the wrist
camera at the top. The exterior and wrist views are the model input. With the
model policy, the bottom row shows the video that the model predicted for the
same ticks. A text band shows the time, the call index, the prompt and the
success flag.

## Live view

The run starts a small HTTP server (standard library, MJPEG, Pillow). Open
the printed URL in a browser. The page shows the live stream and a status
line. After an episode, the page plays its MP4. These are the routes:

| Route | Content |
|---|---|
| `/` | the page |
| `/stream.mjpg` | the live MJPEG stream |
| `/frame.jpg` | the latest frame |
| `/status` | the status as JSON (episode, tick, calls, success, metrics) |
| `/video` | the MP4 of the last finished episode (with HTTP Range support) |

The default address is `127.0.0.1`. On a cloud VM, forward the port with SSH
from your local machine:

```bash
ssh -L 8000:127.0.0.1:8000 user@vm
```

Then open http://127.0.0.1:8000 on your local machine.

**WARNING:** `viewer.host: 0.0.0.0` makes the stream available to all
machines that can reach the port. The stream has no password and no
encryption. Use it only on a private network, and only when SSH port
forwarding is not possible.

After the run, the viewer stays open for `viewer.keep_open_s` seconds
(`null`: until Ctrl-C). Ctrl-C stops the viewer and the run at all times.

With `praxis-1-droid-k` and the settings of that run (`cfg_scale` 1.0,
`cfg_video_scale` 1.0, `t_shift` 8.0, `max_context_chunks` 3),
one call took 0.7 to 1.3 s
for 1.6 s of robot time. We measured this time with
`scripts/run_inference.py` on one H100, without the video decode. The first
call of a process took about 22 s, because it compiles the model. The loop
stops the physics during a call, so the live view stops during each call.
The current settings (`cfg_video_scale` 4.0) also run the decoder on the
unguided pass. We have not measured the call time of `praxis-1-droid` yet. Set `policy.decode_video: false` to save the video decode time. Then the
human view has no predicted video. The decode time increases with the
context depth.

## Config reference

`configs/sim_droid.yaml` holds all the settings of a model run.
`configs/sim_scripted.yaml` holds a scripted run. A key that is not in the
file keeps its default. An unknown key stops the run with an error.

| Block | Keys | Effect |
|---|---|---|
| `model` | `checkpoint`, `vae_weights`, `text_encoder`, `device`, `vae_norm_block_c` | the weights, as in `configs/inference_droid.yaml` |
| `sampler` | `num_steps`, `cfg_scale`, `cfg_video_scale`, `cfg_batched`, `layout`, `t_shift`, `solver`, `action_steps` | the sampler of each call. The defaults are the recommended settings of the single-arm checkpoint: 2 steps, `cfg_scale` 1.0, `cfg_video_scale` 4.0, `t_shift` 12.5, `window_joint` layout. |
| `session` | `exec_horizon`, `control_mode`, `max_context_chunks`, `seed`, `seed_stride` | chunk `k` of the episode with seed `s` samples with `seed + seed_stride * s + k`. The defaults: 24 rows per call, `droid_impedance`, 1 context chunk. |
| `policy` | `type`, `decode_video`, `scripted` | `model`, `scripted` or `hold`. `hold` keeps the start pose. |
| `task` | `name`, `prompt`, `time_limit_s`, `cube_size`, `cube_mass`, `cube_x`, `cube_y`, `cube_yaw_deg`, `lift_height`, `hold_s` | the task, the prompt (`null`: the default of the task), the time limit and the cube placement |
| `episodes` | `seeds`, `stop_on_success`, `after_success_s` | one episode per seed |
| `scene` | `assets_dir`, `asset_source`, `download`, `substeps`, `image_width`, `image_height`, `overview_width`, `overview_height`, `shadow_size`, `msaa_samples`, `exterior`, `wrist`, `table_*` | the images, the cameras and the physics |
| `robot` | `frame`, `ik`, `home`, `home_noise`, `settle_s`, `feedforward` | the pose convention of the policy, the inverse kinematics and the reset pose |
| `render` | `egl_device` | the physical GPU of the renderer |
| `output` | `dir`, `video`, `video_crf`, `save_arrays` | the outputs |
| `viewer` | `enabled`, `host`, `port`, `max_fps`, `jpeg_quality`, `keep_open_s`, `pace_realtime` | the live view. `port: 0` selects a free port. |

YAML reads `1e-3` (no dot) as a string. The loader converts it to a number
for a number key.

Tasks:

- `lift_cube`, prompt "Pick up the red cube.". Success: the cube centre
  stays 5 cm or more above its rest height for 0.25 s or more. A cube that
  falls from 5 cm touches the table after 0.10 s. Thus a cube that the arm
  hits cannot pass. Default time limit: 11 s (165 ticks, one chunk of 21
  rows and six chunks of 24 rows). `configs/sim_droid.yaml` sets 20.6 s
  (309 ticks, 13 calls).
- `cube_on_plate`, prompt "Put the red cube on the blue plate.". Success:
  the cube is on the plate for 0.25 s, and the gripper does not touch it.
  Time limit: 16 s.
- The DROID task families of `praxis/sim/suite.py`: `block_in_bowl`,
  `block_out_of_bowl`, `move_block`, `stack_block` and `press_button`. Each
  task has a prompt in the style of the DROID instructions, a success check
  and a dense progress value (`metrics.progress`).
- The harder tasks. The DROID checkpoint solves most episodes of the tasks
  above, so these tasks make it fail in part of the episodes. Each has a
  seeded start, a prompt in the DROID style, a success check, a dense
  progress value and a scripted oracle (`Task.oracle_steps`).
  - `praxis/sim/suite_place.py`: `cube_in_tray` (an 8 cm open tray) and
    `cube_on_color` (three colour squares, the seed names the target colour).
  - `praxis/sim/suite_insert.py`: `peg_in_hole_6mm`, `peg_in_hole_4mm` and
    `peg_in_hole_2mm` (a 3 cm square peg in a square hole with this
    clearance per side, one prompt for all three).
  - `praxis/sim/suite_hard.py`: `pick_color_block`, `color_block_in_bowl`
    and `stack_color_blocks` (five blocks of different colours, the seed
    names the target block).
  - `praxis/sim/suite_multi.py`: `block_in_small_bowl`,
    `cube_on_small_plate`, `stack_small_block` (smaller targets, wider
    starts), `two_blocks_in_bowl` and `stack_three_blocks` (two and three
    pick-and-place steps).

## The simulated rig

- Robot: the Panda arm with the Robotiq 2F-85 gripper. The robot base is at
  the origin of the world frame. The table top is at the height of the robot
  base.
- Pose convention: `[x, y, z, qw, qx, qy, qz]` in the robot base frame. The
  point is the Franka flange. The rotation is the flange frame turned
  -45 deg about the tool axis (the DROID convention). At the DROID reset
  joints, the pose is (0.360, 0.000, 0.490) m with the tool down and a yaw
  of +45 deg. The gripper opens and closes along the flange y axis. The
  pipeline converts the pose to the flange frame of the checkpoint.
- Gripper: the command 0 opens the gripper and 1 closes it. The observed
  value is `1 - width / 0.085 m`.
- Control: one command row per tick. The tick solves damped least-squares
  inverse kinematics for the row. Then it moves the joint targets of the
  position servos along a ramp, with a velocity feed-forward. A row that
  the solver cannot reach moves the arm to the nearest solution and counts
  as an IK miss. A row that is not finite keeps the previous target.
- Cameras: the exterior camera stays at one position on the left of the
  robot. The wrist
  camera is on the gripper, with the fingers at the bottom of the image.
  Both render 512x288. The pipeline makes one tile of each view (224x416
  for `praxis-1-droid`, a 224x832 model canvas; 256x416 for `praxis-1-droid-k`).
- Physics: 34 substeps per tick (1/510 s), implicit-fast integrator,
  elliptic friction cones, gravity compensation on the arm.

## Results

Measured on one H100 80GB.

The scripted rows used a second GPU for the renderer:

| Check | Result |
|---|---|
| Scripted pick, `lift_cube`, seeds 0 to 9 | 10/10 successes at tick 66 to 74. Tracking error (commanded vs executed pose, after each tick): p95 0.38 to 0.41 mm, max 0.48 mm, rotation max 0.09 deg. 0 IK misses. |
| Speed of the scripted loop | 50 ticks/s (3.3 times real time) with 3 views, the MP4, the arrays and the viewer |

The model rows used `praxis-1-droid-k` with the settings of that run (`cfg_scale` 1.0,
`cfg_video_scale` 1.0, `t_shift` 8.0, `max_context_chunks` 3), the default
time limit and prompt of each task, and `policy.decode_video: false`. The
renderer used the GPU of the model:

| Check | Result |
|---|---|
| `block_in_bowl`, seeds 0 to 3 | 4/4 successes. Episode length 120 to 182 ticks, with the 1 s after success. |
| `stack_block`, seeds 0 to 3 | 3/4 successes |
| `press_button`, seeds 0 to 3 | 0/4 successes. Progress 0.25 to 0.58. |
| Model loop | All 12 episodes: context depth 0, 1, 2, then 3. 0 context restarts. 0 bad view ticks. |
| Time per call | Mean per episode 0.72 to 0.84 s. Two episodes include the compile. Their means are 3.9 s and 4.1 s. |

## Known gaps

- **Sim-to-real gap.** The model runs zero-shot in a rendered scene. The
  table is plain, the room has no clutter, the light is simple, and the
  cube is a synthetic object. The DROID data has none of these properties.
  Thus we expect a low success rate. The sim shows whether the loop, the
  conventions and the model behaviour are plausible. It does not measure
  the success rate of a real robot.
- **Camera poses.** The camera positions and fields of view come from a
  comparison by eye with DROID frames. The DROID exterior cameras have many
  different positions.
- **Controller.** The sim arm reaches each target in less than one tick,
  so the commands jump at the chunk boundaries. The real DROID arm follows its commands approximately 4 ticks
  later. The policy learned the real lag. The `droid_impedance` gains in
  the commands are not used by the sim controller.
- **Pose point.** The sim reports the flange as the end-effector point, as
  the public DROID code does. No sim-to-real trajectory comparison checks
  it. `robot.frame` can change the point and the yaw.
- **Speed.** The loop stops the physics during each model call and each
  video decode, so the live view stops during each call.
- **Memory.** In the model runs of "Results", the model and the renderer
  used the same 80 GB GPU. If the renderer gives black frames, use a
  second GPU for the renderer (`render.egl_device`).
- **Live view.** We checked the page with an HTTP client (status, MJPEG
  frames, MP4 with Range requests), not in a real browser.
