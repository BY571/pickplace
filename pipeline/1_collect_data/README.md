# Stage 1 · collect data

One run of `collect.py` = one **shard**: a teacher checkpoint driven over a camera-enabled env, with every step
recorded to a memory-mapped TensorDict. Shards are the input of stage 2 (offline RL / distillation); they carry
the camera views a student will get plus everything needed to relabel the reward or imitate the teacher.

## Collect

    ./scripts/spark.sh --detach python pipeline/1_collect_data/collect.py \
        checkpoint=/workspace/artifacts/teachers/<run>/checkpoints/ppo_teacher_final.pt \
        frames=1000000 num_envs=512 image_size=84 seed=0 out=/workspace/artifacts/shards/expert_v3c

Options: `checkpoint=` (required), `frames=1000000`, `noise_sigma=0.0`, `seed=0`, `num_envs=512`,
`image_size=84`, `rollout_steps=16`, `out=` (default `$FOOD_ROBOT_ARTIFACTS/shards/<name>`), `name=`.
`frames` is rounded up to a whole number of collector batches (`num_envs x rollout_steps`).

The env config is the **checkpoint's own** (task variant, reward set, randomization), with cameras on and
`image_size` / `num_envs` / `seed` overridden — so a shard always matches the teacher that produced it.
The teacher acts deterministically (its mean action). `noise_sigma > 0` adds Gaussian noise to the executed
action, clipped to the action bounds (explicit, not `AdditiveGaussianModule`, which only perturbs under
`ExplorationType.RANDOM` and would make the teacher sample instead of taking its mean action).

Prints `COLLECT_INFO`, one `PROGRESS` line per batch, `COLLECT` (the manifest) and `COLLECT_DONE`.
Cost: 2 cameras x 84 x 84 x 3 uint8 = 42 KB/frame, i.e. ~42 GB per million frames.

## What a shard contains

`<out>/storage/` — a memmapped TensorDict, one row per env step, and `<out>/manifest.json`.

| Key | Dtype / shape (per row) | |
|---|---|---|
| `("pixels", "overview_rgb")`, `("pixels", "wrist_rgb")` | uint8 `(H, W, 3)` | newest frame of each camera |
| `proprio/*` | float32 | joint pos/vel, gripper, TCP pose, last action |
| `belt/*`, `privileged/*` | float32 | bowl pose; food pose + grasp flag (teacher-only) |
| `action`, `loc`, `scale` | float32 `(7,)` | executed action and the teacher's distribution parameters |
| `step_count` | int64 `(1,)` | step index inside the episode |
| `("collector", "traj_ids")` | int64 | trajectory id (unique per episode across the shard) |
| `("next", "reward")`, `("next", "reward_terms")` | float32 `(1,)`, `(13,)` | scalar reward and the **unweighted** term vector (`pickplace.rewards.REWARD_TERMS`) |
| `("next", "terminated")`, `("next", "truncated")`, `("next", "done")` | bool `(1,)` | |
| `("next", "outcome", <term>)` | bool `(1,)` | how the episode ended (`pickplace.metrics.OUTCOME_TERMS`) |

Rows are **time-major**: row `i` and row `i + manifest["successor_stride"]` (= `num_envs`) are consecutive steps
of the same sub-env. So the successor observation of row `i` is row `i + stride`, valid when
`("next", "done")[i]` is False and `i + stride < frames`. Next observations are not stored separately (that
would double the 42 GB); the final observation of a truncated episode is therefore not in the shard.

`manifest.json`: frames, episodes, noise sigma, seed, image size, num_envs, successor stride, the source
checkpoint with its sha256 and recorded eval metrics, the resolved env config, reward set and weights, git
commit, wall-clock and frames/h, size on disk, and `stats` — success rate, mean episode return and length,
outcome rates and mean per-term episodic sums over every episode that finished during the collection.

## Load

```python
from pickplace.datasets import load_shard, shard_manifest

buffer = load_shard("/workspace/artifacts/shards/expert_v3c", batch_size=256)  # TensorDictReplayBuffer
batch = buffer.sample()                     # uint8 images, nothing copied into RAM until sampled
manifest = shard_manifest("/workspace/artifacts/shards/expert_v3c")
```

Mixing tiers and relabelling rewards belongs to stage 2, not here.

## Tiers (collected 2026-09-20/23 from `teachers/teacher_v3c_20260920T120408Z`)

All four at 84 px, `num_envs=512`, `rollout_steps=16`, 1 M frames (rounded up to 1,007,616 = 123 batches),
in `$FOOD_ROBOT_ARTIFACTS/shards/`. Success rate is the shard's own (every episode that finished during the
collection), not the checkpoint's evaluation.

| Tier | Shard | Checkpoint (eval success) | σ | Seed | Success | Size | Speed |
|---|---|---|---|---|---|---|---|
| expert | `expert_v3c` | `ppo_teacher_final.pt` (0.984) | 0 | 0 | **0.983** | 40.1 GB | 11.4 M frames/h (317 s) |
| medium | `medium_v3c` | `ppo_teacher_110100480.pt` (0.652) | 0 | 1 | **0.674** | 40.1 GB | 11.6 M frames/h (314 s) |
| noisy | `noisy_v3c` | `ppo_teacher_final.pt` (0.984) | 0.2 | 2 | **0.841** | 40.1 GB | 11.2 M frames/h (323 s) |
| beginner | `beginner_v3c` | `ppo_teacher_100139008.pt` (0.219) | 0 | 3 | **0.255** | 40.1 GB | 3.5 M frames/h (1039 s) |

The tiers fail in different ways, which is the point of mixing them: the medium teacher mostly misses the bowl
(31% `bowl_exited_zone`, mean episode length 164 steps vs the expert's 114), the noisy expert mostly drops
food (9.5% `food_off_table`, 5% missed), and the beginner teacher — the earliest checkpoint whose eval success
is above 0% (the run jumps from 0% at 90 M frames to 21.9% at 100 M) — mostly reaches the bowl zone but leaves
it before releasing (73% `bowl_exited_zone`, mean episode length 231 steps, longest of any tier). σ = 0.2 came
from 80 k-frame probes of the expert checkpoint: σ 0.10 → 0.956, σ 0.15 → 0.916, σ 0.20 → 0.835 success. Peak
host memory was 31 GB (512 envs, 84 px); `beginner_v3c` collected at 3.5 M frames/h instead of the usual
~11.4 M frames/h because it ran concurrently with unrelated GPU load on the Spark (portfolio-management
training jobs), not because of anything in `collect.py`.

## Verification

`scripts/verify_shard.py` checks structure (keys/dtypes/shapes), value sanity (NaN/inf, action bounds,
image content), episode structure (trajectory-id runs, done/outcome exclusivity, the successor-stride
invariant), recomputed statistics vs the manifest, checkpoint provenance, and cross-tier plausibility:

    ./scripts/spark.sh python scripts/verify_shard.py \
        /workspace/artifacts/shards/expert_v3c /workspace/artifacts/shards/medium_v3c \
        /workspace/artifacts/shards/noisy_v3c /workspace/artifacts/shards/beginner_v3c

| Tier | Episodes | Success | Return | Length | Zero frames (sample) | Checkpoint sha256 | Result |
|---|---|---|---|---|---|---|---|
| expert_v3c | 8515 | 0.9834 | 182.46 | 114.2 | 0/200 | matches, file exists | 3/8515 done rows with 2 outcome flags set |
| medium_v3c | 5818 | 0.6741 | 163.52 | 164.3 | 0/200 | matches, file exists | clean |
| noisy_v3c | 6052 | 0.8409 | 177.56 | 158.3 | 0/200 | matches, file exists | 6/6052 done rows with 2 outcome flags set |
| beginner_v3c | 4086 | 0.2553 | 134.63 | 230.6 | 0/200 | matches, file exists | 1/4086 done rows with 2 outcome flags set |

All structure/sanity/episode/statistics/provenance checks passed on all four shards, and the tiers behave
as their names claim (expert > noisy > medium > beginner success; medium mostly misses the bowl, noisy
mostly drops food, beginner mostly reaches the bowl zone but exits it before releasing). The only defect:
`expert_v3c`, `noisy_v3c` and `beginner_v3c` each have a handful (<0.1% of episodes) of done rows where two
outcome flags are true at once (mostly `success` + `bowl_exited_zone`, all with `terminated=True`) instead
of exactly one — an upstream env outcome-classification edge case, not a `collect.py`/`load_shard` bug.
`success` is still correct on those rows; stage 2 should treat `success` as authoritative rather than
assuming outcome-flag exclusivity. Full report: `.superpowers/sdd/pipeline-stage0/verify-shards-report.md`.
