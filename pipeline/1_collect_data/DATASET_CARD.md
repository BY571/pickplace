---
task_categories:
- robotics
- reinforcement-learning
tags:
- offline-rl
- robotics
- isaac-lab
- torchrl
- manipulation
license: cc-by-4.0
---

# Pick-and-place on a moving conveyor — offline RL dataset (images + state)

![One episode of the recorded task](assets/episode.gif)

*One episode as recorded: the arm picks the ball from the supply tray, places it into a bowl riding the
belt and returns home, which ends the episode. Top: a scene camera for illustration only, never part of
the data. Bottom: the two camera views that are stored (shown here at 128 px; the dataset stores 84 px).*

Offline-RL dataset of a simulated Franka arm picking food from a supply tray and placing it into bowls
riding a moving conveyor belt, recorded in Isaac Lab 3 by rolling out a privileged-state teacher policy with
cameras enabled. Four quality tiers (`expert`, `medium`, `beginner`, `noisy`) of the same task, generated
from different teacher checkpoints (and, for `noisy`, injected action noise), so the mix has both clean
demonstrations and characteristic failure modes to learn from. `expert` is the run's final checkpoint;
`medium` and `beginner` are two earlier, progressively weaker checkpoints of that *same* training run,
rolled out with zero action noise; `noisy` is the final checkpoint again, this time with Gaussian noise
injected into the executed action — it is the only tier with action diversity around a fixed policy.

**Success condition, which is also when an episode ends:** the food item has settled in the bowl **and**
the hand is back within 5 cm of its start pose. Episodes also end on a failure outcome (bowl leaves its
tracked zone, bowl falls off the belt or tips, food is dropped off the table) or a time-out.

**Code:** the environment, the teacher training and the collection script live at
[github.com/BY571/pickplace](https://github.com/BY571/pickplace) (`pipeline/0_state_teacher` and
`pipeline/1_collect_data`). The repository is private at the time of writing and will be opened later, so
this card is written to stand on its own.

## How the data was generated

1. A privileged-state teacher (PPO, [TorchRL](https://github.com/pytorch/rl)) was trained with full access
   to ground-truth state (food pose, grasp flag, bowl pose) — no cameras, no pixels — in
   `pipeline/0_state_teacher`. Training used a reward vector of 10 dense shaping terms (reach, grasp, lift,
   transport, disturbance, action/velocity penalties, food-in-bowl, return-home) plus 3 one-shot event terms
   (success, bowl failure, food dropped).
2. Three checkpoints from that run were then rolled out with cameras turned on, action clipped to the
   teacher's `[-1, 1]` `TanhNormal` support, and every step recorded to a memory-mapped `TensorDict`
   (`pipeline/1_collect_data/collect.py`). The teacher acts **deterministically** (its distribution mean),
   except for the `noisy` tier where Gaussian noise is added to the executed action before clipping — the
   only tier with any action diversity; `expert`, `medium` and `beginner` are all deterministic rollouts of
   different checkpoints.

| Tier | Source checkpoint | Checkpoint eval success | Action noise σ | Shard success rate | Dominant failure mode |
|---|---|---|---|---|---|
| `expert` | `ppo_teacher_final.pt` (final checkpoint) | 98.4% | 0 | **98.3%** | rare (`food_off_table` 1.2%) |
| `medium` | `ppo_teacher_110100480.pt` (earlier checkpoint, same run) | 65.2% | 0 | **67.4%** | mostly misses the bowl (`bowl_exited_zone` ~31.5%), episodes run longer (mean length 164 vs expert's 114) |
| `beginner` | `ppo_teacher_100139008.pt` (earlier still, same run) | 21.9% | 0 | **25.5%** | usually reaches the bowl zone but exits it before releasing (`bowl_exited_zone` ~73.4%), longest episodes of any tier (mean length 231) |
| `noisy` | `ppo_teacher_final.pt` (final checkpoint + action noise) | 98.4% | 0.2 | **84.1%** | mostly drops food (`food_off_table` ~9.5%) |

The tiers fail in different, characteristic ways by construction — that's the point of mixing them for
offline RL / distillation: `medium` is a weaker policy that struggles to place the food in the bowl at all,
`beginner` is weaker still and rarely gets past reaching the bowl's zone without leaving it again, and
`noisy` is the strong policy destabilized by execution noise, which more often drops what it's holding.
σ = 0.2 was chosen from short probes of the expert checkpoint (σ 0.10 → 95.6%, σ 0.15 → 91.6%, σ 0.20 →
83.5% success). The `beginner` checkpoint was chosen from the run's per-checkpoint evaluations as the
closest available to a 10-20% success band: the run jumps from 0% eval success at 90 M training frames to
21.9% at 100 M frames and 65.2% at 110 M frames, so `ppo_teacher_100139008.pt` (100.1 M frames, 21.9% eval
success) is the closest checkpoint below 30% and the earliest one that isn't simply a policy that never
grasps.

Each tier is 1,007,616 frames (1 M frames rounded up to a whole number of collector batches, 512 parallel
envs x 16 rollout steps x 123 batches), 84x84 px cameras, ~40.06 GB.

| Tier | Frames | Episodes | Size |
|---|---|---|---|
| `expert` | 1,007,616 | 8,515 | 40.06 GB |
| `medium` | 1,007,616 | 5,818 | 40.06 GB |
| `beginner` | 1,007,616 | 4,086 | 40.06 GB |
| `noisy` | 1,007,616 | 6,052 | 40.06 GB |

Total: ~4.03 M frames, ~160 GB.

### Provenance (internal identifiers, kept for reproducibility only)

This dataset was collected by rolling out checkpoints from one internal teacher training run:
- Run: `teachers/teacher_v3c_20260920T120408Z` (Hydra config `pipeline/0_state_teacher/config_v3c.yaml`,
  reward set `simple_v3b` — see [Reward function](#reward-function-simple_v3b) below), PPO via
  [TorchRL](https://github.com/pytorch/rl), Isaac Lab 3.
- Checkpoints: `ppo_teacher_final.pt` (used for the `expert` and `noisy` tiers), `ppo_teacher_110100480.pt`
  (used for `medium`) and `ppo_teacher_100139008.pt` (used for `beginner`) — two earlier and progressively
  weaker checkpoints from the same run; their standalone evaluation success rates are in the tier table
  above.
- Full reproducibility details (exact git commit, checkpoint sha256, resolved environment config, reward
  weights, per-tier collection stats) are recorded in each tier's own `manifest.json`.

## Reward function (`simple_v3b`)

The teacher that generated this data was trained against the `simple_v3b` reward set. It isn't a fixed
"task reward" a reader could otherwise infer — the terms and weights below fully define it.

| Term | Type | Weight | What it measures |
|---|---|---|---|
| `reach_food` | dense | 1.0 | shaping reward for closing the end-effector's distance to the food item |
| `grasp` | dense | 2.0 | reward while the gripper is closed around the food (grasp achieved) |
| `grasp_lift` | dense | 5.0 | reward while the grasped food is lifted more than 10 cm above the table |
| `transport` | dense | 10.0 | shaping reward for carrying the held (grasped *and* lifted) food toward a point above the moving bowl |
| `food_in_bowl` | dense | 20.0 | reward while the food is inside the bowl **and released** (the gripper must let go) |
| `return_home` | dense | 10.0 | shaping reward for bringing the hand back to its home position; paid only while the food is already released in the bowl |
| `success` | event | 150.0 | one-shot bonus on the step the episode ends in success |
| `action_rate` | dense | -0.01 | small penalty on the size of action changes (smoothness) |
| `joint_vel` | dense | -0.001 | small penalty on joint velocities (smoothness) |
| `bowl_disturbance` | dense | -10.0 | penalty proportional to how far the bowl has been pushed from where its pallet carries it |

(Three more terms exist in the reward vocabulary — `transport_fine`, `bowl_failure`, `food_dropped` — but
carry **zero weight** in `simple_v3b`, so they don't contribute to `("next", "reward")` in this dataset;
their raw values are still present in `reward_terms`, see below.)

**Dense vs. event terms:** dense terms are *reward per second the term holds* — the per-step value stored is
`term_value x dt` (the physics timestep), so the weight above is what you earn for holding that condition
for a full second, independent of control frequency. Event terms (`success`, and the zero-weighted
`bowl_failure` / `food_dropped`) are one-shot 0/1 bonuses applied only on the step the episode ends that way.

**Success condition:** an episode scores `success` and ends when the food item has settled in the bowl
**and** the end-effector is back within 5 cm of its *home* position (`success_requires_home: true` in the
generating config). "Home" is one fixed point in the cell frame — the TCP position at the arm's default joint
pose, measured once (`ArmCfg.home_tcp_pos`) — and is the same target in every episode, independent of that
episode's randomized start pose. It is defined in end-effector position space rather than joint space,
because the policy commands end-effector poses and the Franka's redundant elbow is free to drift.

**Start-pose randomization:** each episode starts from the robot's default pose perturbed by uniform noise:
joint positions ±0.25 rad, joint velocities ±0.1 rad/s.

**Relabelling the reward:** `("next", "reward_terms")` (see [Schema](#schema)) is the **unweighted** vector
in `pickplace.rewards.REWARD_TERMS` order — the same 13 terms as the table above plus the two zero-weighted
event terms, in the order `[reach_food, grasp, grasp_lift, transport, transport_fine, bowl_disturbance,
action_rate, joint_vel, food_in_bowl, return_home, success, bowl_failure, food_dropped]`. `("next",
"reward")` is just `simple_v3b`'s weighted sum of that vector, so you can substitute your own weights
without re-simulating anything:

```python
order = ["reach_food", "grasp", "grasp_lift", "transport", "transport_fine", "bowl_disturbance",
         "action_rate", "joint_vel", "food_in_bowl", "return_home", "success", "bowl_failure", "food_dropped"]
my_weights = torch.tensor([1.0, 2.0, 5.0, 10.0, 0.0, -10.0, -0.01, -0.001, 20.0, 10.0, 300.0, -50.0, -50.0])
custom_reward = (batch["next", "reward_terms"] * my_weights).sum(-1)  # drop-in replacement for batch["next", "reward"]
```

## Layout

Each tier is a folder (`expert/`, `medium/`, `beginner/`, `noisy/`) containing:
- `storage/` — a memory-mapped `TensorDict`, one row per environment step (this is what you load).
- `manifest.json` — frames, episodes, tier config (noise σ, seed, image size, num_envs), the source
  checkpoint (path, sha256, its own evaluation metrics), the resolved environment config, reward set and
  weights, git commit, wall-clock/throughput, size on disk, and recomputed dataset statistics (success
  rate, mean return/length, outcome rates, mean per-term episodic sums).

**Rows are time-major, not episode-major.** With `num_envs = 512` parallel sub-environments collected in
lockstep, row `i` and row `i + 512` are consecutive steps of the *same* sub-environment. So the successor
state of row `i` is row `i + 512` — valid only when `("next", "done")[i]` is `False` and `i + 512 <
total_frames`.

**Next observations are not stored separately** (that would double the ~40 GB/tier to ~80 GB): you get
`observation`, `action`, and `("next", reward/terminated/truncated/done/outcome/reward_terms)` for each
row, and you reconstruct `next_observation` as row `i + 512`'s `observation` when valid per the rule above.
A consequence: **the final observation of a truncated episode is not in the shard** (there is no row `i +
512` for it within the file, or that row belongs to the next episode) — mask these out rather than
treating them as a valid transition.

## Known data quirk: double outcome flags

A small number of `done` rows (<0.1% of episodes: 3/8,515 in `expert`, 6/6,052 in `noisy`, 1/4,086 in
`beginner`, none in `medium`) have **two** outcome flags set simultaneously instead of exactly one — almost
always `success` together with `bowl_exited_zone` (a couple are `bowl_exited_zone` + `food_off_table`). All
of these rows have `terminated=True`. This is an upstream environment outcome-classification edge case (the
bowl leaving its tracked zone can coincide with a success or a food-drop event), not a recording bug —
episode return, length and the `success` flag itself are all still correct on these rows. **Treat `success`
as the authoritative outcome flag**; do not assume the outcome flags are mutually exclusive.

## Schema

![Frames as stored in the dataset](assets/dataset_frames.png)

*Six frames sampled across the `expert` tier, both cameras, exactly as stored (84x84x3 uint8, upscaled
here with nearest-neighbour). The schema below is identical across all four tiers.*

Every row of `storage/` has the following keys.

| Key | Dtype | Shape | |
|---|---|---|---|
| `("pixels", "overview_rgb")` | uint8 | `(84, 84, 3)` | fixed overview camera, newest frame |
| `("pixels", "wrist_rgb")` | uint8 | `(84, 84, 3)` | wrist-mounted camera, newest frame |
| `("proprio", "ee_pos")` | float32 | `(3,)` | end-effector position |
| `("proprio", "ee_quat")` | float32 | `(4,)` | end-effector orientation (xyzw) |
| `("proprio", "gripper_pos")` | float32 | `(2,)` | gripper finger positions |
| `("proprio", "joint_pos_rel")` | float32 | `(9,)` | joint positions relative to default pose |
| `("proprio", "joint_vel_rel")` | float32 | `(9,)` | joint velocities relative to default |
| `("proprio", "last_action")` | float32 | `(7,)` | previous step's executed action |
| `("belt", "bowl_pos")` | float32 | `(3,)` | target bowl position on the conveyor |
| `("privileged", "food_pos")` | float32 | `(3,)` | food item position (teacher-only signal) |
| `("privileged", "food_quat")` | float32 | `(4,)` | food item orientation (xyzw) |
| `("privileged", "is_grasped")` | float32 | `(1,)` | 1.0 if the food is currently grasped |
| `action` | float32 | `(7,)` | executed action (post-noise for `noisy`) |
| `loc` | float32 | `(7,)` | teacher policy's `TanhNormal` mean, pre-noise |
| `scale` | float32 | `(7,)` | teacher policy's `TanhNormal` scale |
| `step_count` | int64 | `(1,)` | step index within the episode |
| `("collector", "traj_ids")` | int64 | `()` | trajectory id, unique per episode across the whole shard |
| `("next", "reward")` | float32 | `(1,)` | scalar reward (reward-set weights applied) |
| `("next", "reward_terms")` | float32 | `(13,)` | **unweighted** per-term reward vector, order below |
| `("next", "terminated")` | bool | `(1,)` | episode ended by a terminal condition |
| `("next", "truncated")` | bool | `(1,)` | episode ended by a time/step limit |
| `("next", "done")` | bool | `(1,)` | `terminated or truncated` |
| `("next", "outcome", "success")` | bool | `(1,)` | food settled in bowl, hand returned home |
| `("next", "outcome", "bowl_exited_zone")` | bool | `(1,)` | bowl left its tracked zone |
| `("next", "outcome", "bowl_off_belt")` | bool | `(1,)` | bowl fell off the conveyor |
| `("next", "outcome", "bowl_tipped")` | bool | `(1,)` | bowl tipped over |
| `("next", "outcome", "food_off_table")` | bool | `(1,)` | food fell off the table |
| `("next", "outcome", "time_out")` | bool | `(1,)` | episode hit the time limit |

`("next", "reward_terms")` is ordered `[reach_food, grasp, grasp_lift, transport, transport_fine,
bowl_disturbance, action_rate, joint_vel, food_in_bowl, return_home, success, bowl_failure, food_dropped]`
(`pickplace.rewards.REWARD_TERMS`: 10 dense shaping terms followed by 3 one-shot event terms). Dense terms
are `value x dt`; event terms are 0/1 on the step the episode ends that way. The scalar `("next", "reward")`
is the weighted sum of this vector under the collecting run's reward set (`simple_v3b`, recorded per-tier
in `manifest.json["reward_weights"]`); the unweighted vector lets you relabel the reward with different
weights without re-simulating. See [Reward function](#reward-function-simple_v3b) above for what each term
means and a relabelling example.

## How to load

With the (private) source repo, `pickplace.datasets.load_shard` opens a shard as a TorchRL replay buffer,
reading nothing into RAM until you sample:

```python
from pickplace.datasets import load_shard, shard_manifest

buffer = load_shard("expert", batch_size=256)  # TensorDictReplayBuffer over a memmap
batch = buffer.sample()                        # uint8 images, float32 everything else
manifest = shard_manifest("expert")
```

The plain TorchRL equivalent, for anyone without this repo (`pip install tensordict torchrl`):

```python
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, TensorStorage

storage = TensorStorage(TensorDict.load_memmap("expert/storage"))
buffer = TensorDictReplayBuffer(storage=storage, batch_size=256)
batch = buffer.sample()
```

Mixing tiers and relabelling rewards is left to the consumer — this dataset is deliberately just the raw,
per-tier shards.

## License

Released under CC BY 4.0. If you'd prefer a different license (e.g. MIT, Apache-2.0, or CC0), it's easy to
change — just ask.
