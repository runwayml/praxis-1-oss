# RoboLab client

A client that runs a praxis policy server on [RoboLab](https://github.com/NVLabs/RoboLab)
tasks with RoboLab's own evaluation runner.

The client follows RoboLab's client/server setup. `PraxisRemoteClient`
subclasses `robolab.eval.base_client.InferenceClient` (see RoboLab's
`policies/<name>/client.py`). RoboLab's episode loop calls `infer` once per
15 Hz control step with the sim paused, and the client returns RoboLab's
8-dim absolute end-effector action (`DroidIKActionCfg`:
`[x, y, z, qw, qx, qy, qz, grip]`). The policy runs on a remote server that
speaks the Positronic WebSocket protocol.

## Versions

| Component | Version |
|---|---|
| RoboLab | NVLabs/RoboLab @ `7d45d74904eade3b578a8eb1f2f9f89bc3d40326` |
| Isaac | Isaac Lab 2.2, Isaac Sim 5.0 (installed by RoboLab's `uv sync`) |
| Positronic | [Positronic-Robotics/positronic](https://github.com/Positronic-Robotics/positronic) @ `a25e9e731f8c3d1203eb741ff4e6281c74c837d3` (version 0.4.3) |

The client differs from a stock Positronic setup in a few places:

- Positronic's convention reports the end-effector pose at the gripper site, and a
  policy states its own frame (Positronic's `ChangeEEFrame`). This client keeps the
  pose in RoboLab's `eef_frame` and applies a measured constant frame change in front
  of `RemotePolicy` (`positronic_cfg/ee_frame_policy_cfg.py`), so the server gets the
  pose in the frame it expects without declaring a frame change of its own.
- The client drops `call_id` and `call_timings`, which the Positronic schedule adds
  to each request, and leaves out `robot_state.status`. The server reads none of them.
- The client records no `ee_frames` entry in the episode metadata (nothing in the
  RoboLab runner reads it).
- At server action rates other than 15 Hz, an action starts on the first control
  tick at or after its due time, so some actions can start up to one tick later
  than their timestamps. Keep this in mind before reusing the client with a server
  or rig that runs at another rate.

## Files

| File | Runs in | Role |
|---|---|---|
| `robolab_stock/run_praxis.py` | RoboLab's Isaac interpreter | Entry point for RoboLab's runner. Modelled on RoboLab's `policies/cosmos3/run.py`: registers the tasks with RoboLab's AbsIK registration and the `WRIST_LEFT` cameras, then calls `run_evaluation`. |
| `robolab_stock/praxis_client.py` | RoboLab's Isaac interpreter | The `InferenceClient`. Imports only numpy, RoboLab and `praxis_stock_wire`. Starts the sidecar and talks to it over pipes. |
| `robolab_stock/praxis_stock_wire.py` | RoboLab's Isaac interpreter | Pure numpy: wire pose to the 8-dim `DroidIKActionCfg` action. |
| `robolab_stock/praxis_sidecar.py` | Positronic's venv | The Positronic client half: the RoboLab adapter, the observation build and `RemotePolicy` (image resize, temporal stack, chunked schedule and JPEG encoding, as the server declares), on Positronic's `Executor` with a virtual clock. |
| `positronic_cfg/ee_frame_policy_cfg.py` | Positronic's venv | Puts a constant end-effector frame change (RoboLab's `eef_frame` to the frame the server expects, and back for commands) in front of `RemotePolicy`. Reads the auth headers from `PRAXIS_POLICY_HEADERS`. |
| `positronic_cfg/ee_frame_wire.py` | Positronic's venv | Pure numpy frame math for the wrapper above. |
| `robolab_stock/example_client.py` | Positronic's venv + RoboLab on `PYTHONPATH` | The fake-data example: drives `PraxisRemoteClient` the way RoboLab's episode loop does, on synthetic observations. No Isaac needed. |

RoboLab's Isaac interpreter and Positronic need different Python
environments, so the client runs the Positronic half in a separate process
(the sidecar) and exchanges length-prefixed msgpack frames with it
(Positronic's `simulator/env_server/protocol.py` codec).

## Setup

You need a running praxis policy server (see `praxis/serve/README.md` for local,
Docker and Modal serving), `git` with [Git LFS](https://git-lfs.com/),
[`uv`](https://docs.astral.sh/uv/) (it downloads Python if it is missing), a C/C++
compiler (`build-essential` on Debian/Ubuntu: one Positronic dependency builds from
source) and `ffmpeg` (RoboLab records videos). Set these absolute paths once; every later command uses them, so
they work from any directory:

```bash
export PRAXIS_DIR=/abs/path/praxis-1             # this repository
export ROBOLAB_DIR=/abs/path/RoboLab             # the RoboLab checkout
export POSITRONIC_DIR=/abs/path/positronic       # the Positronic checkout
export POSITRONIC_VENV=$POSITRONIC_DIR/.venv
```

1. Install RoboLab at the commit above (its scene files are in Git LFS):

   ```bash
   git lfs install
   git clone https://github.com/NVLabs/RoboLab.git $ROBOLAB_DIR
   git -C $ROBOLAB_DIR checkout 7d45d74904eade3b578a8eb1f2f9f89bc3d40326
   (cd $ROBOLAB_DIR && git lfs pull && uv venv --python 3.11 && uv sync)
   export OMNI_KIT_ACCEPT_EULA=Y   # Isaac Sim asks to accept its EULA on first start
   ```

2. Check out Positronic at the commit above and make its venv from its lock file:

   ```bash
   git clone https://github.com/Positronic-Robotics/positronic.git $POSITRONIC_DIR
   git -C $POSITRONIC_DIR checkout a25e9e731f8c3d1203eb741ff4e6281c74c837d3
   (cd $POSITRONIC_DIR && uv sync --frozen --no-dev --python 3.12)
   ```

   With an empty uv cache this takes several minutes and makes a venv of about
   1.2 GB.

3. Optional: the sidecar inherits the client's environment (proxy and TLS
   settings included) minus `PYTHONPATH` and `VIRTUAL_ENV`, and the client puts
   `positronic_cfg/` on its `PYTHONPATH`. To override entries, write them to a
   JSON file and point `PRAXIS_SIDECAR_ENV_FILE` at it. A file that sets
   `PYTHONPATH` must include `positronic_cfg/`.

   ```bash
   cat > $PRAXIS_DIR/sidecar_env.json <<EOF
   {"HTTPS_PROXY": "http://proxy.example:3128"}
   EOF
   export PRAXIS_SIDECAR_ENV_FILE=$PRAXIS_DIR/sidecar_env.json
   ```

## Environment variables

| Variable | Meaning |
|---|---|
| `PRAXIS_POLICY_HEADERS` | JSON object of HTTP headers for the endpoint's proxy auth, e.g. `{"Modal-Key": "<token-id>", "Modal-Secret": "<token-secret>"}`. Passed to the sidecar through the environment only, never on the command line. No credentials are stored in this code. Leave it unset for a server without auth. |
| `PRAXIS_SIDECAR_PYTHON` | The Python of the Positronic venv. |
| `PRAXIS_POSITRONIC_DIR` | The Positronic checkout (the client loads its wire codec by path). |
| `PRAXIS_SIDECAR_ENV_FILE` | Optional JSON file of sidecar environment overrides (step 3). |
| `PRAXIS_SIDECAR_LOG` | Sidecar stderr log (default `/tmp/praxis_sidecar.<pid>.log`). |
| `PRAXIS_CALLS_LOG` | One JSON line per policy call (default `/tmp/praxis_client_calls.<pid>.jsonl`). The server's session metadata goes to `<calls log name>.handshake.json` next to it. |

## Fake-data example

`example_client.py` builds RoboLab-shaped observations: RoboLab's DROID reset
joints and end-effector pose, plus two 1280x720 uniform-noise frames
(exterior and wrist). It runs `--steps` control steps and prints the actions.
The endpoint is a `wss://` or `https://` URL (`ws://`, `http://` or `host:port`
for a server without TLS); the client appends `/api/v1/session?<query>`. This
example imports only `robolab.eval.base_client` from RoboLab, so a clone of RoboLab
at the pinned commit on `PYTHONPATH` is enough; it needs no Isaac install.

```bash
export PRAXIS_POLICY_HEADERS='{"Modal-Key": "<token-id>", "Modal-Secret": "<token-secret>"}'
export PRAXIS_SIDECAR_PYTHON=$POSITRONIC_VENV/bin/python
export PRAXIS_POSITRONIC_DIR=$POSITRONIC_DIR
PYTHONPATH=$PRAXIS_DIR/eval/robolab/robolab_stock:<robolab checkout> \
  $POSITRONIC_VENV/bin/python $PRAXIS_DIR/eval/robolab/robolab_stock/example_client.py \
    --policy-url wss://<endpoint> --steps 48
```

The default `--query` holds the recommended server settings:
`declare_stop_on_fault=false&record_denoised=false&max_context_chunks=1&t_shift=12.5`.
A server that accepts no session settings needs `--query ""`.

The expected output, for a server that returns 24-action chunks at 15 Hz:
policy calls at ticks 2, 23 and 47 (logged to `--calls-log`), and actions of
shape `(48, 8)`.

`example_client.py` takes the server's base URL and adds the session path and
query itself; `run_praxis.py` below takes the full session URL.

## Full evaluation

Run from the RoboLab checkout, in RoboLab's venv, with the same environment
variables. For all 120 tasks at 10 episodes each, leave out `--task`:

```bash
cd $ROBOLAB_DIR
PYTHONPATH=$PRAXIS_DIR/eval/robolab/robolab_stock:$PYTHONPATH \
  uv run python $PRAXIS_DIR/eval/robolab/robolab_stock/run_praxis.py --headless \
    --num-envs 1 --num-runs 10 \
    --policy-url 'wss://<endpoint>/api/v1/session?declare_stop_on_fault=false&record_denoised=false&max_context_chunks=1&t_shift=12.5' \
    --output-folder-name <name>
```

Add `--task <Task> [<Task> ...]` to run some tasks only. The tasks use their
default instruction (`--instruction-type default`).

Run one env per process (`--num-envs 1`; the client refuses other values);
`--num-runs` sets the episodes per task. Without a `sample_seed` session setting
the server draws a new sampling seed for every session; add `&sample_seed=<int>` to
the URL for runs you want to repeat.

To resume a run that stopped, run the same command with the same
`--output-folder-name`: RoboLab skips the tasks and episodes it has already
finished. The client does not reconnect a dropped connection; a dropped
connection stops the run, and resuming continues it.

## Results

RoboLab writes the episodes to `output/<name>/episode_results.jsonl` in the RoboLab
checkout and prints a summary table at the end of the run. The client writes one
JSON line per policy call to `PRAXIS_CALLS_LOG` (a `fault` line marks a refused
episode) and the server's session metadata to `<calls log name>.handshake.json` next
to it. Once per
session the sidecar log notes that the server declares no action rate and that the
client assumes 15 Hz; that is expected.

## Behavior on a server refusal

If the server refuses a request because the end-effector pose no longer matches
the joint angles, or because the pose or gripper value is not finite (both happen
after a physics blow-up throws the arm), the client logs the refusal, closes that
env's session and holds the arm at its measured pose (its last finite pose if the
measurement is not finite) until RoboLab resets the env. The episode then ends
through RoboLab's own success check or timeout and is scored like any other. The
same refusal on an episode's first call, at the reset pose, points to a
misconfiguration and stops the run. Any other error (another server error, the
sidecar, the transport) stops the run.

If the server answers with an empty chunk, the client holds the arm at its measured
pose and asks the server again on the next control step.

Waits are bounded by Positronic's defaults: 180 s for one policy call, and up to
900 s to open a session while the server starts.
