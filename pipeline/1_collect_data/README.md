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
| `("next", "reward")`, `("next", "reward_terms")` | float32 `(1,)`, `(13,)` | scalar reward and the **unweighted** term vector (`food_robot.rewards.REWARD_TERMS`) |
| `("next", "terminated")`, `("next", "truncated")`, `("next", "done")` | bool `(1,)` | |
| `("next", "outcome", <term>)` | bool `(1,)` | how the episode ended (`food_robot.metrics.OUTCOME_TERMS`) |

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
from food_robot.datasets import load_shard, shard_manifest

buffer = load_shard("/workspace/artifacts/shards/expert_v3c", batch_size=256)  # TensorDictReplayBuffer
batch = buffer.sample()                     # uint8 images, nothing copied into RAM until sampled
manifest = shard_manifest("/workspace/artifacts/shards/expert_v3c")
```

Mixing tiers and relabelling rewards belongs to stage 2, not here.

## Tiers (v3c teacher, `teacher_v3c_20260920T120408Z`)

| Tier | Checkpoint | Noise σ | Frames | Success rate | Size | Speed |
|---|---|---|---|---|---|---|
| expert | `ppo_teacher_final.pt` (eval 0.984) | 0 | | | | |
| medium | `ppo_teacher_110100480.pt` (eval 0.652) | 0 | | | | |
| noisy | `ppo_teacher_final.pt` | | | | | |
