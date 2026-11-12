# SO-100 / SO-101 client

`so101_client.py` runs a praxis policy server on an so101 checkpoint
(`"embodiment": "so101"`, top README "SO-101 rig") on an SO-100 / SO-101
follower arm through [LeRobot](https://github.com/huggingface/lerobot). The
server runs elsewhere: `python -m praxis.serve --config configs/serve_so101.yaml`
on a local GPU (a 5090 serves the routed experts as NVFP4), or the Modal app
(`praxis/serve/README.md`). The client is one file without a praxis import.

## Install, next to the robot

```bash
pip install "lerobot[feetech]" numpy msgpack websockets pillow
```

The follower is calibrated with LeRobot (`lerobot-calibrate`) and connects
with `use_degrees=True`, the LeRobot 0.7 default: joints in degrees, 0 at
the calibrated mid-range, `gripper.pos` in 0..100.

## Run

```bash
# the follower's gripper.pos fully closed and fully open (torque off, by hand)
python eval/so101/so101_client.py --url ws://<host>:8000/api/v1/session \
    --robot-port /dev/ttyACM0 --robot-id my_so101 --measure-grip

# a 2-view session; --dry-run prints the targets and never moves the arm
python eval/so101/so101_client.py --url ws://<host>:8000/api/v1/session \
    --robot-port /dev/ttyACM0 --robot-id my_so101 \
    --exterior-cam 0 --wrist-cam 2 --grip-closed 1.4 --grip-open 42.1 \
    --prompt "Pick up the red cube and put it in the box." --dry-run
```

- `--exterior-cam` / `--wrist-cam`: an OpenCV index, a device path, or the
  name of a camera in `--cameras` (LeRobot's `--robot.cameras` dict as JSON).
  Omit `--wrist-cam` for a 1-view session.
- `--grip-closed` / `--grip-open`: the readings of `--measure-grip`. The
  defaults 1.4 / 42.1 are the training medians, to be replaced.
- `--control-hz` (30): the arm gets the 15 Hz rows interpolated at this rate.
  A deadline missed during a stall (a camera read, a GC pause) slips: the
  client slows down, it never sends the overdue rows back to back.
- Safety: LeRobot's `--max-relative-target` caps each command on the
  follower; `--max-step-deg` / `--max-grip-step` add a per-row clamp of the
  client (off by default). The client stops when the arm does not start at the
  folded rest pose of training (`shoulder_lift` about -100 deg, `elbow_flex`
  about 95 deg; `--no-rest-check` warns instead). Ctrl-C holds, then
  disconnects; LeRobot releases the torque on disconnect unless
  `--hold-torque-on-exit`.
- `--session '{"num_steps": 4, "sample_seed": 0}'` sets other session
  parameters of the server under their own names; `--max-context-chunks` and
  `--exec-horizon` are flags.
- A Modal endpoint takes its proxy auth headers from `--headers-file` (a JSON
  file with `Modal-Key` and `Modal-Secret`).
- `--fake-robot` runs the loop without hardware against an endpoint;
  `--log-jsonl FILE` appends a record per chunk.

`tests/serve/test_so101_client.py` and `tests/serve/test_serve_so101.py` run the
client's helpers and a fake-robot session against the test server.
