# PPO

Proximal Policy Optimization on `FoodRobot-Cell-v0`, following TorchRL's `sota-implementations` layout.

## Run

Nothing runs on the laptop; every command below runs on the DGX Spark through `./scripts/spark.sh`
(inside the container by default). A small run (e.g. for a quick sanity check):

    ./scripts/spark.sh python sota-implementations/ppo/ppo.py env.num_envs=16 collector.rollout_steps=8 max_iterations=3

Full-scale training:

    ./scripts/spark.sh python sota-implementations/ppo/ppo.py env.num_envs=4096 logger.backend=wandb

Both of the above run attached to your terminal (the ssh session must stay open for the duration).
For a real training run, start it detached instead so it survives a disconnect:

    ./scripts/spark.sh --detach python sota-implementations/ppo/ppo.py env.num_envs=4096 logger.backend=wandb

This starts the command in a named, detached container (`food-robot-<timestamp>`) and returns
immediately, printing the container name plus the commands to follow its logs
(`ssh spark docker logs -f <name>`) and stop it (`ssh spark docker stop <name>`).

Play a checkpoint (opens the viewer):

    python play.py play.checkpoint=outputs/<date>/<time>/checkpoints/ppo_final.pt

Isaac Lab enables/disables cameras at app launch, before the checkpoint's saved config can be read, so
`play.py` launches with the CLI's `env.cameras` (default `false`) and then checks it against the value the
checkpoint was trained with — if they don't match it raises before building the env. A checkpoint trained
with `env.cameras=true` (the asymmetric or deployment-like setups below) must be played with
`env.cameras=true` on the command line too:

    python play.py play.checkpoint=outputs/<date>/<time>/checkpoints/ppo_final.pt env.cameras=true

`env.device` must also match the device the checkpoint was trained on (both are read from the CLI/default
config before the checkpoint can be loaded); pass `env.device=<value>` if it differs from the default.

## Observation routing

`network.actor_in_keys` / `network.critic_in_keys` choose which observations each network sees.
Group names expand to all their terms; nested keys select one term.

| Setup | actor_in_keys | critic_in_keys | env flags | play needs |
|---|---|---|---|---|
| State teacher (default) | `[proprio, belt, privileged]` | `[proprio, belt, privileged]` | `cameras=false privileged_information=true` | (default) |
| Asymmetric actor-critic | `[proprio, belt, [pixels, wrist_rgb]]` | `[proprio, belt, privileged]` | `cameras=true privileged_information=true` | `env.cameras=true` |
| Deployment-like | `[proprio, belt, pixels]` | `[proprio, belt, pixels]` | `cameras=true privileged_information=false` | `env.cameras=true` |

Example: `python ppo.py env.cameras=true env.num_envs=256 'network.actor_in_keys=[proprio,belt,[pixels,wrist_rgb]]'`

## Pixels (`ppo_pixels.py`)

PPO from the two cameras (3 stacked frames each) plus robot state only — no privileged food state and no belt
bowl position — with geometric domain randomization of the supply bowl, bowl arrival time and lateral offset
(see `docs/media/ppo_pixels_run1_dr_ranges.png`). Actor and critic are separate networks with the same
structure: one CNN per camera (32/64/64 channels, kernels 8/4/3, strides 4/2/1 → 256), proprio → Linear 128 +
LayerNorm + ELU, fused by an MLP 512-256. Camera frames are stored as uint8 in the replay buffer.

    ./scripts/spark.sh --detach python sota-implementations/ppo/ppo_pixels.py

Config: `config_pixels.yaml` (scale settings from the benchmark in `docs/experiments/ppo_pixels_run1`).
`reward_scale` scales the rewards used for GAE and the loss; logged returns are unscaled.

Render a checkpoint (scene camera + the policy's camera inputs):

    python scripts/render_episode.py policy=outputs/<date>/<time>/checkpoints/ppo_pixels_final.pt

The run ends at `collector.total_frames` (1 billion env frames), or earlier once the training success rate reaches
`early_stop.success_rate` (85%) in `early_stop.consecutive_iterations` (10) iterations in a row, after `max_hours`, or on
SIGTERM (`ssh spark 'docker exec <container> pkill -TERM -f "kit/python/bin/python3.* ppo_pixels.py"'`; `docker stop`
does not reach Python through Isaac Sim's `python.sh` wrapper). In every case the policy is saved as
`checkpoints/ppo_pixels_final.pt` first, and a `STOP_REASON {json}` line says why it ended.

Logged metrics (W&B, and one `METRICS {json}` stdout line per iteration):

- `train/success_rate` and `train/<outcome>_rate`: fraction of the episodes that finished in the batch that ended
  by each termination (`success`, `bowl_exited_zone`, `bowl_off_belt`, `bowl_tipped`, `food_off_table`,
  `time_out`), from the env's `("next", "outcome", <term>)` entries; `train/episode_return`, `train/episode_length`
- `episode_reward/<term>`: per-term episodic reward (Isaac Lab reward manager)
- `eval/*`: every `eval.interval_iterations`, all envs are reset and run deterministically until each finished
  one episode; `eval/policy_view` is a video of env 0's camera inputs
- `perf/*`: env steps/s, collect/update seconds, frames/hour, memory

## Logged metrics

- `train/episode_return`, `train/episode_length`
- `episode_termination/<term>`: fraction of envs whose last episode ended by `success`, `bowl_exited_zone`,
  `bowl_off_belt`, `bowl_tipped`, `food_off_table` or `time_out` — `episode_termination/success` is the success rate.
- `train/loss_objective`, `train/loss_critic`, `train/loss_entropy`

## Measured on DGX Spark (2026-09-15)

`python ppo.py env.num_envs=4096 max_iterations=5 logger.backend=csv` (state-based teacher config:
`cameras=false privileged_information=true`), run three times via `./scripts/spark.sh --detach`:

- **FPS:** collector+update throughput was consistently ≈ 41,000 env-steps/s once the scene was built
  (`frames_per_batch = num_envs × rollout_steps = 4096 × 24 = 98,304`; all 5 iterations —
  491,520 frames total — completed in ≈ 12 s per tqdm). Isaac Sim startup and scene creation for 4096
  envs (not counted in the above) took ≈ 80-90 s.
- **Memory:** unified host memory (`free -g`, 121 GB total on this Spark) peaked at ≈ 14 GB used during
  the run, leaving large headroom; `nvidia-smi` reports `memory.used`/`memory.total` as `N/A` on this
  unified-memory board, but `utilization.gpu` samples showed brief spikes to 59-67% during the physics
  step. This state-based (no-camera) config is far from any memory ceiling at 4096 envs — see
  `docs/environment.md` for camera-training memory guidance instead.
- **PhysX GPU buffer overflow:** the *first* run at `num_envs=4096` hit 118
  `PxgAABBManager.cpp` "PhysX error: ... increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity"
  errors (requesting up to ~16,883 against the then-default capacity of 16,384 = `16*1024`). Fixed by
  raising `PhysxCfg.gpu_total_aggregate_pairs_capacity` to `32*1024` in
  `food_robot/envs/cell_env_cfg.py`; two subsequent runs at the same `num_envs=4096` produced zero PhysX
  errors. A later, longer teacher run at 4096 envs (2026-09-18) requested ~33.2k and logged ~25k errors,
  so the capacity is now `64*1024` and scales linearly above 4096 envs (`scale_physx_buffers`).
- **Episodes finishing:** with only 5×24 = 120 steps/env (≈ 2.4 s of the ≈ 9 s max episode length), most
  of the 4096 envs' episodes are still in progress when the run ends; whether *any* env's episode
  finishes early (e.g. `bowl_tipped`/`food_off_table` under the initial random policy) varies run to run.
  One of the three runs did finish episodes for a handful of envs, logging
  `train/episode_return ≈ -150.0` (matching the `bowl_failure_penalty`/`food_drop_penalty` one-shot
  penalty exactly) at `train/episode_length` 24 and 4 steps — confirming both that early-failure
  penalties land correctly and that the episode-return/length reconstruction in `ppo.py` (see below) is
  correct. The other two runs logged only `episode_termination/*` (all 0, as expected when no episode has
  finished yet) and no `train/episode_return`/`train/episode_length`, which is expected given the short
  window rather than a bug.

While investigating this run, `tests/isaac/test_torchrl_episodes.py` (added for finding 1) found that
`IsaacLabWrapper(native_autoreset=True)` zeroes/NaNs `("next", "step_count")` and
`("next", "episode_reward")` (RewardSum's key) on the exact row where `done=True`, since the auto-reset
happens inside that same `env.step()` call. Reading `data["next", "episode_reward"][done]` /
`data["next", "step_count"][done]` directly (the original code) therefore always produced `NaN` /
`0` for `train/episode_return` / `train/episode_length`. Fixed in `ppo.py` and `play.py` to reconstruct
the completed episode's return/length from the pre-step root tensordict instead:
`data["episode_reward"] + data["next", "reward"]` and `data["step_count"] + 1`.
