# praxis-sim

The MuJoCo simulation of the Praxis-1 single-arm rig (a Franka arm with a Robotiq 2F-85
gripper, an exterior and a wrist camera, 15 Hz control) and its fixed evaluation suites
(`droid`, `droid_guard`, `hard`), packaged on their own: MuJoCo, Gymnasium and numpy, no
torch. The model loop (`praxis.sim.loop`), the video writer and the live viewer stay in the
`praxis` package, which runs the simulation through this one.

```bash
pip install "praxis-sim @ git+https://github.com/runwayml/praxis-1-oss.git#subdirectory=packages/praxis-sim"
# or, from a checkout of this repository:
pip install -e packages/praxis-sim
```

The robot models (MuJoCo Menagerie, pinned commit) download on first use into a cache
directory and are checked by SHA-256 (`praxis_sim.assets`).

## Gymnasium environments

`praxis_sim.gym_env.PraxisSimGymEnv` is one task as a Gymnasium environment: the reset seed
is the scene seed, the observation holds the two camera views (`pixels/exterior`,
`pixels/wrist`, 512 x 288 uint8), the pose of the Franka flange as
`[x, y, z, roll, pitch, yaw, gripper]` (`agent_pos`, the DROID convention) and the scene
seed; the action is the same seven values as an absolute target.
`create_praxis_sim_envs(task, n_envs, ...)` builds the `{suite: {task_index: VectorEnv}}`
layout of a suite, the same episodes as `scripts/run_sim_eval.py` of the praxis repository.
LeRobot's `--env.type=praxis_sim` is a thin config over it.

## Comparing runs

`python -m praxis_sim.compare <lerobot eval_info.json> <praxis runner output directory>` pairs
the episodes of a LeRobot evaluation with those of the praxis runner by task and scene seed
and reports wins, losses, the exact McNemar p and a bootstrap interval of the mean difference.

## Layout

| Module | Contents |
| --- | --- |
| `gl`, `assets`, `scene`, `robot`, `tasks`, `env` | The simulator: OpenGL backend, robot models, MuJoCo scene, control tick, tasks, environment |
| `suite*`, `eval_suites` | The task families and the fixed evaluation suites |
| `scripted`, `config` | The scripted pick policy and the run settings (YAML loader) |
| `pose` | Pose math and command helpers, copies of `praxis.robot_io` / `praxis.single_arm` (checked bit for bit by the praxis tests) |
| `gym_env`, `compare` | The Gymnasium wrapper and the LeRobot-vs-runner comparison |
