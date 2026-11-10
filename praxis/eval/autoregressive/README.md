# Autoregressive prompt-adherence eval (video only)

A model gets one real first frame and a task prompt. It rolls its video out
on its own predictions for 12.6 s (189 frames at 15 fps). No ground-truth
frame or action enters after frame 0. A vision-language judge then reads the
video and answers: did the robot act on the named object, bring it to the
named destination, and perform the named action?

The eval scores videos, so it compares any model that predicts video:
Praxis-1 and other world action models. The judge and the scorer do not
depend on praxis.

| File | What it holds |
|---|---|
| `sets.py` | load a set from the pinned Hugging Face revision or a local copy; verify every file |
| `judge.py` | the judge request (frames, question, system prompt) and the call |
| `score.py` | full adherence, the bootstrap intervals, the paired differences, the time cut-off |
| `judge_system_droid.txt` | the system prompt of the judge (2 views) |
| `judge_schema_droid.json` | the answer schema |
| `judge_tts_system.txt`, `judge_tts_schema.json` | the time-to-success judge |
| `scripts/score_autoregressive_eval.py` | the command line: `judge` and `score` |

## Inputs: the sets

The sets are in the Hugging Face dataset
[`runwayml/praxis-1-eval`](https://huggingface.co/datasets/runwayml/praxis-1-eval),
folder `autoregressive/`, commit `2e95455bc8497d6fe2c00da759e863e186a7d9f2`
(`sets.REVISION`). `sets.load_set` checks the SHA-256 of `rollouts.jsonl`,
`scenes.jsonl` and every frame, and refuses a mismatch. The `droid` scenes
also hold a second exterior view (`frames_ext2/<scene>.png`, one 416x224
tile, `EvalSet.second_exterior`): the episode's real second exterior frame
with the same objects inserted. Praxis-1 does not read it; the baselines of
the reference results, which take a second exterior camera, were given it.

| Set | Rig | Canvas | Scenes | Prompts | Rollouts |
|---|---|---|---|---|---|
| `droid` | single-arm Franka (DROID) | 832x224: exterior, wrist | 8 | 35 | 104 |

The dataset also holds two bimanual YAM sets.
This release evaluates `droid` only; the YAM sets come with a later release.

A rollout row names the prompt, the seed, the target object, the
destination, the action, the look-alike distractors and, for the spatial
and count features, the relation and the count. The dataset card describes
the scenes.

## The video of a model

A model writes one `<id>.mp4` per rollout of the set: 189 frames at 15 fps,
starting from the first frame of the scene. The judged canvas is 832x224.
The judge splits it into 416-px tiles, one per camera
(`SetSpec.judge_views`). The judge refuses a video of another size, frame count or
frame rate (`judge.judged_frames`). It is not validated on any other canvas.

Frame 0 of a Praxis-1 video is the model's own decode of the first frame,
not the frame itself. The first chunk call decodes 21 frames, and they
replace the real frame (`VideoRollout.step`). The reference rollouts behind
the published results did the same. On 4 published DROID clips, frame 0 is
34.5 dB PSNR from the scene frame, and the same as the decode of this
repository (34.0 to 34.8 dB). A CRF-18 encode of the real frame alone gives
37.5 dB.

The Praxis-1 rollout of this repository is `rollout.py`, run by
`scripts/run_autoregressive_eval.py` with `configs/autoregressive_droid.yaml`.

The rollout needs the install of the main `README.md`, with the
`flash-attn-4` step of "Get the weights" (or the `triton` backend of the
same section). `pip install -e ".[eval]"` alone covers the judge and the
scorer only.

```bash
python scripts/run_autoregressive_eval.py --config configs/autoregressive_droid.yaml \
    --out outputs/ar/praxis-1-droid [--shard 0/4]
```

Its recipe: 8 chunks (21 + 7 x 24 = 189 frames), a window of at most 4
chunks, the history re-encoded from the model's own decoded frames on every
call, 10 Euler steps, flow shift 8, text guidance 3.0 on the video with the
negative prompt "The robot does not move. Static scene, blurry, flickering,
objects morphing, duplicated objects." in the unguided pass. The noise of
chunk `c` is `seeded_normal(shape, 1000 * seed + c, VIDEO_SALT)`. The
sampler is `PraxisPolicy.sample_window_joint`; the video does not read the
proprio or the actions, so the rollout passes zero proprio and drops the
actions. One rollout takes about 51 s on one H100 (104 DROID rollouts: about 1.5
GPU-hours).

## The judge

The judge is GPT-6 Astra (`gpt-6-astra`) through any OpenAI-compatible
endpoint: `ASTRA_BASE_URL`, `ASTRA_API_KEY`, `ASTRA_MODEL`. One request per
rollout holds:

- the system prompt of the set;
- the question: the scene, the cameras, the task, the adherence targets of
  the rollout row (the destination distractor is listed with the
  distractors);
- 12 evenly spaced frames, each a grid of labelled camera tiles at 1.5x
  (JPEG quality 85), then the last frame at 2.25x.

The answer is strict JSON (at most 8000 tokens). The request sets no
temperature, so the endpoint's default applies. Every published verdict
used the default: the gateway of the published runs ignored the
`temperature: 0` that their requests carried. The call
retries 6 times with backoff. It falls back to `json_object` if the
endpoint refuses a JSON schema. A bad key (HTTP 401), no access (403), an
unknown model (404) or another 400 stops the call at once, without retries.
The `judge` command refuses an answer without the three booleans of full
adherence, or a time-to-success answer without a valid `success_time_s`.
It does not write such an answer, so a rerun asks again. The `judge`
command refuses a run folder with missing videos: every rollout of the set
must have its mp4, and the command gives the number of missing videos. On the DROID set
one rollout costs about $0.02.

The time-to-success judge reads 24 evenly spaced frames of a fully adherent
clip and gives `success_time_s`: the first frame time at which the target
rests at the destination.

The tile labels use DejaVu Sans Bold (`fonts-dejavu-core` 2.37). Another font
gives other image bytes, so the request parity test then skips.

## Full adherence and the statistics

A rollout is **fully adherent** when `target_object_acted_on`,
`target_destination_used` and `action_correct` are all true.

- The level of a run is the mean over prompts of the mean over the seeds of
  each prompt, in percent.
- The 95 % interval is the 2.5 and 97.5 percentiles of 10,000 bootstrap
  means over prompts.
- Two models compare paired on the same (prompt, seed): the per-prompt mean
  of the differences, its bootstrap interval, and a sign-flip permutation
  p value over prompts (10,000 draws). When none of the 10,000 flips reaches
  the observed difference, the scorer prints `p<=0.0001 (0 of 10000 sign
  flips)`: the resolution of 10,000 draws.
- The scorer refuses a verdict whose three fields are not booleans.

**Within 11.9 s** counts a rollout only if it is fully adherent and its
`success_time_s` is at most 11.894 s. `score --cutoff` refuses a run in
which a fully adherent rollout has no time-to-success answer. 11.894 s is
the smallest 95th percentile of the success times of the DROID baselines (DreamZero 11.89 s,
FLUX 3 Action 12.00 s, Cosmos3-Nano 12.53 s): the time by which the fastest
of them completes 95 % of its successes. The full clip favours models whose
video stays stable; the cut-off reduces that advantage. It does not remove
it: a point within the cut-off still needs the full-clip adherence verdict.

## Judge validation

The judge was checked against human labels on a bimanual YAM set: 80 rollouts
labelled by hand (one labeller). On full adherence the judge agrees with the
human on 94 % of the rollouts (Cohen's kappa 0.87). Two judge runs on the same videos agree on
97 %. Physical plausibility is not a reported metric: its human labels were
not reliable.

The scope of this check:

- The 80 labels are YAM rollouts only, from one labeller. The YAM sets are
  not in this release.
- The DROID set has no human labels. The DROID judge was checked against
  the earlier judge only, a model-vs-model check: the three fields agree on
  102 of 104 Praxis-1 rollouts.
- Full adherence scores the generated video only. It is not validated as a
  predictor of real-robot task success.
- It is not validated for ranking checkpoints of one model.

## Usage

`pip install -e ".[eval]"` is enough for `judge` and `score`. The rollout
needs the full install (see "The video of a model").

```bash
pip install -e ".[eval]"
# the videos of a model: outputs/ar/<model>/<id>.mp4
ASTRA_BASE_URL=https://api.openai.com/v1 ASTRA_API_KEY=... \
  python scripts/score_autoregressive_eval.py judge --set droid --run outputs/ar/mine --tts
python scripts/score_autoregressive_eval.py score --set droid \
  outputs/ar/praxis-1-droid outputs/ar/mine            # full clip
python scripts/score_autoregressive_eval.py score --set droid --cutoff 11.894 \
  outputs/ar/praxis-1-droid outputs/ar/mine            # within 11.9 s
```

The tests run on the CPU and need no weights (`pip install -e ".[eval,dev]"`):

```bash
pytest tests/test_eval_autoregressive.py
```

The request parity tests need DejaVu Sans Bold. Without the font they skip.

## Reference results

Full adherence, %, 95 % interval over prompts. The baselines ran in their
joint video and action mode, with their released default sampler and the
real frame-0 robot state.

DROID set (104 rollouts), paired against Praxis-1. The baselines were given
the second exterior view of the set; OpenWAM-alpha has no second exterior
input:

| Model | Full clip (12.6 s) | Paired | Within 11.9 s | Paired |
|---|---|---|---|---|
| Praxis-1 (`praxis-1-droid`) | 52.9 [38.6, 67.1] | | 46.4 [31.4, 61.4] | |
| FLUX 3 Action DROID | 48.6 [35.0, 62.9] | -4.3, p = 0.61 | 42.1 [27.9, 57.1] | -4.3, p = 0.64 |
| DreamZero-DROID | 42.1 [30.0, 55.0] | -10.7, p = 0.22 | 37.9 [25.7, 51.4] | -8.6, p = 0.37 |
| Cosmos3-Nano-Policy-DROID | 22.9 [11.4, 35.7] | -30.0, p = 0.0002 | 17.9 [7.1, 30.7] | -28.6, p = 0.002 |
| OpenWAM-alpha (foundation checkpoint) | 1.4 [0.0, 4.3] | -51.4, 0 of 10,000 sign flips | 0.0 | -46.4, 0 of 10,000 sign flips |

With a black second exterior view instead, the baselines score lower: FLUX 3
Action 38.6 [24.3, 53.6], DreamZero-DROID 26.4 [12.9, 40.7],
Cosmos3-Nano-Policy-DROID 23.6 [11.4, 37.1].

`tests/test_eval_autoregressive.py` checks that the scorer gives the
Praxis-1 and DROID baseline numbers above from the stored
verdicts (`tests/data/autoregressive_verdicts.json`).

## Parity with the published results

The bar was written before any number of this implementation:

1. **Scores, exact (CPU).** On the stored verdicts of the published runs,
   `score.py` gives the numbers of the tables above, the DROID paired
   differences and the within-11.9-s numbers, to the printed digit.
2. **Judge request, exact (CPU).** For the same video, `judge.py` builds the
   same request as the published judge, byte for byte: system prompt,
   question, image bytes, schema and call settings. The published requests
   also carried `temperature: 0`, which their gateway ignored; this judge
   leaves it out.
3. **Rollout (GPU).** On the DROID set with `praxis-1-droid` (the
   checkpoint of the published DROID results) and the same seeds: the
   rollout is not expected to be bitwise equal. Known sources of
   difference: the attention kernel (the published rollout used a KV cache
   over the context, this repository runs the full window, the
   bitwise-checked serve path) and the video encoder kernel (see
   "Validation" in the main `README.md`). The noise follows the published
   scheme (`1000 * seed + chunk`, salt 0). The bar: per-frame PSNR,
   reported, and full adherence inside the published 95 % interval, with a
   paired difference that is not significant.

Status:

| Check | Result |
|---|---|
| 1. Scores | Pass: every published number above, to the printed digit (`tests/test_eval_autoregressive.py`). |
| 2. Judge request | Pass: every request equal on all 84 DROID clips of the 7 scenes before the count-hard scene and 12 time-to-success clips of the published runs; 10 hashed fixtures in the tests. |
| 3. Rollout (GPU) | Pass, as a check of the 84 rollouts of the 7 DROID scenes before the count-hard scene was added (same metric and judge). 84 DROID rollouts of `praxis-1-droid` (H100, PyTorch 2.13, cuDNN 9.24, `flash` backend). Not bitwise equal: 0 of 84 first chunks equal. Mean PSNR against the published rollouts, chunks 0 to 7: 36.7, 25.6, 22.5, 21.1, 19.5, 18.0, 17.1, 16.1 dB. Three-part full adherence 49.2 [31.7, 65.8] vs the published 52.5 [35.8, 68.3]; paired -3.3 [-10.0, +2.5], p = 0.44 (4 gained, 5 lost, 75 the same). Within 9.8 s (the cut-off of those 84 rollouts and the black-ext2 baselines): 41.7 vs 42.5, paired -0.8, p = 1. |
| End-to-end validation | Pass. Fresh clone, fresh venv. All 104 DROID rollouts of `praxis-1-droid` with `scripts/run_autoregressive_eval.py` (GPU), judged and scored with the commands of "Usage". Full clip 50.7 [35.7, 65.7] vs the published 52.9; paired -2.1, p = 0.65. Within 11.9 s 45.0 [30.0, 60.0] vs the published 46.4; paired -1.4, p = 0.84. Mean PSNR against the published rollouts, chunks 0 to 7: 35.1, 23.8, 21.4, 20.2, 18.9, 17.4, 16.7, 15.8 dB. The judge re-run on the published clips gives the same verdict on 103 of 104 rollouts. |
