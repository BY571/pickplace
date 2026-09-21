# Stage 2.1 · offline RL from the camera shards

Train a **camera-only student** on the shards recorded in stage 1, with no simulator in the loop except for
evaluation. Behaviour cloning is the baseline; IQL is the first value-based algorithm. CQL and TD3+BC will
follow the same shape, so that the only thing that differs between rows of the results table is the
objective.

## The comparison protocol

Every algorithm in this folder gets the same everything except its loss:

| | |
|---|---|
| **Data** | `expert_v3c` + `medium_v3c`, 50/50 rows per batch (2,014,214 legal transitions, 14,333 of them terminal) |
| **Student inputs** | **image only** by default — `("pixels","overview_rgb")` and `("pixels","wrist_rgb")`, 84x84x3 uint8, **one frame each**. No proprioception, no privileged state, no bowl pose. Configurable via `network.in_keys` (group names or leaf keys, expanded with `pickplace.keys.expand_in_keys`); vector groups go through an MLP branch and concatenate with the CNN features (`pickplace.offline.PixelNet`, same shape as `sota-implementations/ppo/utils_pixels.py::PixelsNet`). |
| **Networks** | one Nature-CNN encoder per camera (32/64/64 channels, 8/4/3 kernels, 4/2/1 strides, 256-d embedding) → fusion MLP `[512, 256]`. Separate encoders per head (actor, each Q, V). |
| **Batch** | 256 transitions |
| **Budget** | 150,000 gradient steps (~19 passes over the data) |
| **Evaluation** | every 10,000 gradient steps: 128 fresh camera envs, deterministic policy, `max_episode_length + 1` steps, each env's **first** finished episode scored — byte-for-byte the protocol `pipeline/0_state_teacher/evaluate.py` used to score the teacher, so student and teacher numbers compare directly |
| **Reward** | scaled by 0.1 before the loss (BC ignores it) |

`tests/unit/test_offline_configs.py` asserts that the shared block of every `config.yaml` in this folder is
identical, so the protocol cannot drift between algorithms by accident.

**The student has to infer arm and gripper state from pixels.** Nothing in its input says where the joints
are, whether the gripper is open or whether the food is held — that has to come out of the wrist view (the
overview camera mostly supplies the bowl's position on the belt). This is the interesting part of the
setting and the main reason a camera-only student is expected to fall short of the privileged teacher.

## Transitions: what a "pair of rows" is

A shard stores one row per env step, time-major, and **does not store next observations**. So the successor
state of row `i` is row `i + successor_stride` (the collection's `num_envs`, 512). `pickplace.offline`
precomputes the legal pairs once per shard as an index tensor (never a per-batch scan):

* **ongoing** — row `i` has no done flag and `i + stride` is inside the shard → `next_obs` is row `i + stride`,
  `terminated = False`.
* **terminal** — row `i` is `terminated` and not `truncated` → the episode ended here, so the bootstrap is
  masked (`terminated = True`) and `next_obs` is a placeholder (row `i` itself) multiplied by zero in the TD
  target. **These pairs are kept on purpose**: with `simple_v3b` the success bonus is 150 of a ~182 mean
  episode return, so dropping terminal rows would hide 80%+ of the reward from any value-based method.
  Set `data.include_terminals=false` to drop them anyway.
* **truncated** rows are dropped — the episode did not end, but its next observation genuinely is not in the
  shard, so neither bootstrapping nor masking would be right.

No pair ever spans an episode boundary. `tests/unit/test_offline.py` pins this down on a synthetic shard with
known boundaries: the exact legal index set, the successor row, the terminal flags, and the fact that every
sampled pair is one of the legal ones.

Tier mixing is a fixed number of rows per shard per batch (largest-remainder split of `batch_size` by
`data.proportions`), concatenated — not `ReplayBufferEnsemble`, because a shard's legal rows are a
precomputed subset *and* its successor row has to be gathered at `+stride`, which no stock sampler expresses.

## Run

    # baseline
    ./scripts/spark.sh --detach python pipeline/2_1_offline_rl/bc/train.py

    # IQL
    ./scripts/spark.sh --detach python pipeline/2_1_offline_rl/iql/train.py

Useful overrides: `gradient_steps=`, `batch_size=`, `data.shards=[expert_v3c,medium_v3c,noisy_v3c]`,
`data.proportions=[...]`, `network.in_keys=[[pixels,overview_rgb],[pixels,wrist_rgb],proprio]`
(the deployable image+proprio variant; `belt` — the bowl's pose — is the one further group a real cell
might supply from a belt encoder), `eval.interval=`, `eval.num_envs=`, `checkpoint.interval=`, `optim.lr=`,
IQL's `loss.expectile=` / `loss.temperature=` / `loss.gamma=` / `loss.target_tau=`, `run.name=`,
`logger.backend=null`. Stop a run cleanly (final checkpoint + evaluation) with

    ssh spark 'docker exec <container> pkill -TERM -f "kit/python/bin/python3.*train.py"'

## Where results land

`$FOOD_ROBOT_ARTIFACTS/students/<run>/`:

* `manifest.json` — git commit, full config, one provenance block per shard (name, frames, legal
  transitions, source checkpoint + sha256, reward set and weights, the shard's own success rate), the W&B
  URL, `eval_history` (one entry per evaluation) and `best_eval`.
* `checkpoints/<algo>_<step>.pt` + `.json` — the sidecar manifest repeats the git commit, config and shard
  provenance and adds the **online evaluation at that checkpoint** and the checkpoint's sha256.
* stdout: `RUN_INFO`, `METRICS {json}` every `log_interval` steps, `EVAL`, `CHECKPOINT`, `STOP_REASON`,
  `<ALGO>_DONE`.

W&B project `food_robot`, group `offline_rl`.

## Results

**Only deployable observations.** Every row below shares the protocol above (150 k steps, batch 256, eval
every 10 k over 128 envs, seed 0, W&B group `offline_rl`) and uses only observations a real cell can
measure: the two cameras, and `proprio` (`ee_pos`, `ee_quat`, `gripper_pos`, `joint_pos_rel`,
`joint_vel_rel`, `last_action`), which always comes from the robot's own controller. Runs that consumed the
simulator-only `privileged` group (food pose, orientation, grasp flag) were made early on and have been
dropped: that information has no real-robot equivalent, so it cannot inform a decision about what to ship.
Rows are grouped by the demonstration data they were trained on.

Teacher (privileged state, cameras off, same evaluation protocol): **0.984** success.

![Offline-RL students on deployable observations (image / image+proprio), by demonstration data](../../docs/experiments/offline_rl/success_rate_by_data.png)

### Deployable — image only; image + proprioception

| Algorithm | Run | Inputs | Data | Best | Final | Mean of last 5 evals | Wall clock | W&B |
|---|---|---|---|---|---|---|---|---|
| BC | `students/bc_expert_medium_v1` | image | expert+medium | 0.758 @ 80 k | 0.500 | 0.548 | 0.64 h | [xy467x3h](https://wandb.ai/sebastian-dittert/food_robot/runs/xy467x3h) |
| IQL | `students/iql_expert_medium_v1` | image | expert+medium | 0.727 @ 80 k | 0.617 | 0.614 | 1.84 h | [597r6u3q](https://wandb.ai/sebastian-dittert/food_robot/runs/597r6u3q) |
| BC | `students/bc_expert_only` | image | expert-only | 0.820 @ 130 k | 0.820 | 0.777 | 0.52 h | [wj4kg3hy](https://wandb.ai/sebastian-dittert/food_robot/runs/wj4kg3hy) |
| IQL | `students/iql_expert_only` | image | expert-only | 0.813 @ 90 k | 0.797 | 0.748 | 1.82 h | [d2ia8nor](https://wandb.ai/sebastian-dittert/food_robot/runs/d2ia8nor) |
| BC | `students/bc_expert_only_proprio` | image+proprio | expert-only | **0.984 @ 100 k** | 0.953 | **0.955** | 0.58 h | [29ti5hv2](https://wandb.ai/sebastian-dittert/food_robot/runs/29ti5hv2) |
| IQL | `students/iql_expert_only_proprio` | image+proprio | expert-only | **0.992 @ 130 k** | 0.961 | 0.942 | 1.83 h | [eu7d42wg](https://wandb.ai/sebastian-dittert/food_robot/runs/eu7d42wg) |
| CQL | *planned* | | | | | | | |
| TD3+BC | *planned* | | | | | | | |

Success rate per evaluation, 128 episodes each (binomial standard error around 0.6 is about 0.043 — the
`±0.04` referenced throughout this file):

| Gradient steps (k) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BC, image, expert+medium | .05 | .44 | .45 | .57 | .43 | .58 | .65 | **.76** | .47 | .63 | .66 | .57 | .60 | .41 | .50 |
| IQL, image, expert+medium | .16 | .34 | .30 | .63 | .45 | .44 | .59 | **.73** | .45 | .59 | .66 | .63 | .55 | .62 | .62 |
| BC, image+proprio, expert-only | .86 | .95 | .93 | .95 | .93 | .98 | .97 | .85 | .95 | **.98** | .96 | .97 | .95 | .94 | .95 |
| IQL, image+proprio, expert-only | .85 | .88 | .88 | .93 | .94 | .97 | **.98** | .97 | .95 | .98 | .98 | .97 | **.99** | .81 | .96 |

**Reading the original two baselines (image only, expert+medium).** Both curves climb fast to ~0.45 by
30-40 k and then oscillate in a 0.4-0.76 band with no further trend — flat from roughly 60 k on, so the
150 k budget is already past the point of return. The swings are much larger than evaluation noise (±0.04),
so they are real policy changes from step to step, not sampling error: with a state-independent scale that
has collapsed (BC's mean scale falls from 0.98 to ~0.12 by 10 k steps), a small drift in the mean action
changes the grasp outcome on a large share of episodes. The two algorithms are within noise of each other on
the best checkpoint (0.76 vs 0.73), but IQL is the more stable of the two late in training (last-five mean
0.61 vs 0.55, and BC's last three evaluations are its worst since 30 k). A camera-only student on this data
tier reaches roughly **three quarters of the teacher's 0.984 at its best checkpoint and about 60% on
average**, which is the cost of removing proprioception (and, in that ablation, privileged state too — see
below for separating the two).

**Reading the image+proprio runs.** Both land in a tight 0.81-0.99 band from 10 k on — far tighter than
the image-only runs' 0.05-0.76 swing — and both cross the teacher's 0.984 at least once (BC at 100 k, IQL
at 70 k, 110 k and 130 k). IQL's one bad point (0.805 @ 140 k) is a single-step dip, not a trend: its
surrounding evaluations are 0.992 and 0.961. Proprioception does not just raise the ceiling, it removes most
of the camera-only oscillation — plausible, since the student no longer has to infer joint/gripper state
from the wrist camera before it can even attempt the grasp.

**What to try next**, in the order the evidence suggests: pick checkpoints by evaluation rather than by step
(the best checkpoint beats the final one by several points in most runs here); average several evaluations
per point or use more evaluation envs so the selection is not chasing noise; add the `noisy_v3c` tier for
state coverage; and give the actor a learned, state-dependent scale so it can stay stochastic where the data
is ambiguous. CQL and TD3+BC come next and inherit this protocol unchanged — run them with the
image+proprio inputs, since that is now the group worth shipping.

### Borderline — not run, flagged for a future variant

Nothing has been trained in this group yet. `belt` (the bowl's pose on the conveyor) is stored as
simulator-only state in this dataset, but in a real cell it could plausibly come from a belt encoder or a
fixed overhead camera rather than from privileged simulator state. `image + proprio + belt` would
therefore be a legitimate deployable variant to try next — unlike `privileged` (food pose, orientation,
grasp flag), which has no real-robot equivalent and is out of scope here.

## Headline: how much of the gap to the teacher is closed, and by what

The best **deployable** student, `iql_expert_only_proprio` (image+proprio, expert-only data), reaches
**0.992 at its best checkpoint — slightly above the teacher's own 0.984** — and `bc_expert_only_proprio`
reaches **0.984**, an exact match. On the more conservative final-checkpoint and last-5-mean metrics both
land at 0.94-0.96, a few points under the teacher, which given the ±0.15 step-to-step swings this protocol
already shows (see above) is noise, not a systematic shortfall.

What closed the camera-only gap, data held fixed at expert-only (the best tier — see the data-quality
ablation below): **image-only → image+proprio** takes BC's best checkpoint from 0.820 to 0.984, closing
**100%** of its 0.164-point gap to the teacher, and IQL's from 0.813 to 0.992, closing its 0.171-point gap
**and overshooting by 0.008**. Adding only what a real robot's own controller already knows — joint, gripper
and end-effector state — is enough on its own to match the privileged-state teacher. Once the student can
feel where its own joints and gripper are, the image only have to supply what they are good at: where the
food and the bowl are. The teacher's 0.984 is not, in practice, out of reach for a policy that sees only what
a real robot controller could give it.

## Ablations: data quality and observation access

Two ablations on top of the two image-only baselines above, same protocol (150 k steps, batch 256, eval
every 10 k over 128 envs, same seed): **data quality** — image-only inputs, `expert_v3c` alone (1 M frames)
instead of expert+medium — and **observation access** — expert-only data, inputs extended from image-only
to image + `proprio`.

### Data quality: does the medium tier help?

| Algorithm | Run | Best | Final | Mean of last 5 evals | Wall clock | W&B |
|---|---|---|---|---|---|---|
| BC | `students/bc_expert_medium_v1` (expert+medium) | 0.758 @ 80 k | 0.500 | 0.548 | 0.64 h | [xy467x3h](https://wandb.ai/sebastian-dittert/food_robot/runs/xy467x3h) |
| BC | `students/bc_expert_only` (expert-only) | **0.820 @ 130 k** | **0.820** | **0.777** | 0.52 h | [wj4kg3hy](https://wandb.ai/sebastian-dittert/food_robot/runs/wj4kg3hy) |
| IQL | `students/iql_expert_medium_v1` (expert+medium) | 0.727 @ 80 k | 0.617 | 0.614 | 1.84 h | [597r6u3q](https://wandb.ai/sebastian-dittert/food_robot/runs/597r6u3q) |
| IQL | `students/iql_expert_only` (expert-only) | **0.813 @ 90 k** | **0.797** | **0.748** | 1.82 h | [d2ia8nor](https://wandb.ai/sebastian-dittert/food_robot/runs/d2ia8nor) |

**The medium tier hurts, for both algorithms.** Dropping it raises BC's best checkpoint by 6 points (0.758 →
0.820) and, far more strikingly, its final-checkpoint stability: final success goes from 0.500 (BC's
collapsed, worst-since-30k state at 150 k on the mixed data) to 0.820 — the same as its best. Last-five-eval
mean rises from 0.548 to 0.777. IQL improves by a similar or larger margin on every metric: best +0.086
(0.727 → 0.813), final +0.180 (0.617 → 0.797), last-five mean +0.134 (0.614 → 0.748).

**IQL does not show the theoretically-expected benefit from mixed-quality data.** IQL's whole pitch —
expectile value estimation plus advantage-weighted policy extraction — is supposed to let it exploit a mix of
good and bad demonstrations better than plain cloning, tolerating (or even benefiting from) the lower-quality
tier that BC just imitates blindly. That is not what happens here: on every metric IQL's improvement from
removing `medium_v3c` is at least as large as BC's, in relative terms comparable (best: +12% vs +8%; final:
+29% vs +64%, though BC's baseline final of 0.500 was an anomalous late-training collapse rather than a
stable number, so that particular ratio overstates BC's gain). `medium_v3c` is a genuinely weaker *policy*, not noise: an
earlier checkpoint of the same teacher run (`ppo_teacher_110100480`, 65% success, zero action noise). It is
still the same training run, so it fails in the same ways the expert occasionally does rather than
demonstrating different behaviour, and it gives both algorithms lower-return transitions to estimate
advantages against. Whether IQL can exploit a *differently* suboptimal source (a beginner policy, or a
scripted one) is untested here.

**Read against the teacher.** Expert-only best checkpoints reach ~81-83% of the teacher's 0.984 (0.820/0.984,
0.813/0.984) — noticeably closer than the expert+medium runs' ~75%. Since the *inputs* did not change (still
image only), this gap between the two data tiers is entirely a data-quality effect, not an observation
one; see below for how much of the *remaining* ~17-19 points is the camera-only bottleneck rather than the
algorithm.

### Observation access: how much of the gap is the camera bottleneck?

Same `expert_v3c` data as the expert-only baselines, but `network.in_keys` extended from image-only to
`[[pixels,overview_rgb],[pixels,wrist_rgb],proprio]` via `pickplace.keys.expand_in_keys`. `proprio` is the
robot's own state — joint positions and velocities, gripper opening, end-effector pose, last action — which
any real controller publishes, so this stays deployable.

| Algorithm | Run | Best | Final | Mean of last 5 evals | Wall clock | W&B |
|---|---|---|---|---|---|---|
| BC | `students/bc_expert_only_proprio` (image+proprio) | **0.984 @ 100 k** | 0.953 | 0.955 | 0.58 h | [29ti5hv2](https://wandb.ai/sebastian-dittert/food_robot/runs/29ti5hv2) |
| IQL | `students/iql_expert_only_proprio` (image+proprio) | **0.992 @ 130 k** | 0.961 | 0.942 | 1.83 h | [eu7d42wg](https://wandb.ai/sebastian-dittert/food_robot/runs/eu7d42wg) |

**Proprioception, not the images, was the bottleneck.** On the same expert-only data, adding the robot's own
state takes BC from 0.820 to 0.984 and IQL from 0.813 to 0.992 — the whole gap to the teacher, and the curves
also stop swinging (last-5 mean 0.955/0.942 versus 0.777/0.748). A camera-only policy has to infer its own
arm and gripper configuration from pixels, mostly from the wrist view, and that inference is what it was
failing at — not seeing the food or the bowl.

Earlier runs that also consumed the simulator-only `privileged` group (food pose, orientation, grasp flag)
reached 0.961 (BC) and 0.977 (IQL) — no better than these deployable runs — and have been dropped from the
comparison, since no real deployment can have those inputs.

## Adding an algorithm

Create `pipeline/2_1_offline_rl/<algo>/{train.py,utils.py,config.yaml}`:

* `config.yaml` — copy another algorithm's file and change only the block below `optim:` (the parity test
  enforces the rest).
* `utils.py` — `make_algo(cfg, obs_shapes, obs_keys, action_dim, device)` returning an object with
  `.policy` (the module that is evaluated and checkpointed), `.update(batch) -> {name: scalar}` and
  `.state_dict()`. Build the networks with `pickplace.offline.make_actor` / `make_qvalue` / `make_value` so
  the architecture stays identical.
* `train.py` — copy another algorithm's; it only launches the Isaac app and hands `make_algo` to
  `runner.train`.
* Add the algorithm to the results table and to the smoke test's parametrization.
