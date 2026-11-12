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
python demo/so101/so101_client.py --url ws://<host>:8000/api/v1/session \
    --robot-port /dev/ttyACM0 --robot-id my_so101 --measure-grip

# a 2-view session; --dry-run prints the targets and never moves the arm
python demo/so101/so101_client.py --url ws://<host>:8000/api/v1/session \
    --robot-port /dev/ttyACM0 --robot-id my_so101 \
    --exterior-cam 0 --wrist-cam 2 --grip-closed 1.4 --grip-open 42.1 \
    --prompt "Pick up the red cube and put it in the box." --dry-run

# a live run with the page (views, prompt, buttons) at http://127.0.0.1:8780/
python demo/so101/so101_client.py --url ws://<host>:8000/api/v1/session \
    --robot-port /dev/ttyACM0 --robot-id my_so101 \
    --exterior-cam 0 --wrist-cam 2 --grip-closed 1.4 --grip-open 42.1 \
    --prompt "Pick up the red cube and put it in the box." \
    --max-relative-target 10 --ui-port 8780
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
  client (off by default).
- Rest pose: the client moves the arm to the folded rest pose of training
  before the first chunk, and back to it when the client ends (Ctrl-C,
  SIGTERM, SIGHUP; without the page also `--max-chunks` and a server error).
  The pose is `0,-100,95,70,0` deg for `shoulder_pan`, `shoulder_lift`,
  `elbow_flex`, `wrist_flex`, `wrist_roll` (`--rest-pose` gives another) and
  the gripper at `--grip-closed`. The move has three phases: shoulder lift and
  elbow, then the pan, then the wrist and the gripper. A wrist that is flexed
  past its rest value unflexes before these: the gripper would block the elbow
  (measured: the elbow stopped at 69.5 deg). The phases run a second time when
  the fold ends more than 10 deg from the rest value and the first pass moved
  the pan, the wrist or the gripper. The farthest joint of a phase goes at 40
  deg/s at most. After a fold that stopped short the pan phase is skipped: an
  arm that is not folded does not turn. No command leads the reading by more
  than 8 deg, so a blocked joint gets a bounded push; a joint that makes no
  progress for 1.5 s ends its phase. After the start move the client stops
  when `shoulder_lift` or `elbow_flex` is not at its rest value (-100 and 95
  deg, or those of `--rest-pose`) (25 deg of tolerance; `--no-rest-check`
  warns instead): another calibration, or an obstacle. After the end move
  LeRobot releases the torque on disconnect unless `--hold-torque-on-exit`.
  Ctrl-C during a rest move stops it: the arm is released where it is.
  `--no-rest-move` turns both moves off: the client then checks the start pose
  only and releases the arm where the run ends. `--dry-run` never moves the
  arm, so it has no rest move.
- Page: `--ui-port 8780` serves a page at `http://127.0.0.1:8780/`. It shows
  the live views as the model gets them, the state of the client, the prompt
  in use with a field to change it, the chunk number, the round trip, the
  model time, the joint readings and three buttons. With the page the client
  waits for `Start` after the start move (`--autostart` starts at once).
  `Start` runs a session, on a new connection each time. `Stop` ends the
  session or the rest move that runs; the arm holds its position with the
  torque on. `Return to rest` moves the arm to the rest pose. No button ends
  the client: Ctrl-C does, and the arm then returns to the rest pose. With the
  page, `--max-chunks` or a problem of the server ends the session, not the
  client; the page shows the reason. A new prompt goes out as the `task` of
  the next chunk request; the page shows the prompt that the server answers
  with. The browser keeps the history of the prompts; a click on one puts it
  in the field, and `Set prompt` applies it. The page has no password: keep
  the default `--ui-host`. It answers only requests addressed to a loopback
  name or to the address of `--ui-host` (a bind on `0.0.0.0` answers the
  loopback names only), and takes a `POST` only as JSON from its own origin.
  The page is one HTML document inside the client, with no request to another
  host.
- While the client waits for a cold server it prints the elapsed time every
  30 s.
- `--session '{"num_steps": 4, "sample_seed": 0}'` sets other session
  parameters of the server under their own names; `--max-context-chunks` and
  `--exec-horizon` are flags.
- A Modal endpoint takes its proxy auth headers from `--headers-file` (a JSON
  file with `Modal-Key` and `Modal-Secret`).
- `--fake-robot` runs the loop without hardware against an endpoint;
  `--log-jsonl FILE` appends a record per chunk.

`tests/serve/test_so101_client.py` and `tests/serve/test_serve_so101.py` run the
client's helpers and a fake-robot session against the test server.
