# PPO

Proximal Policy Optimization on `FoodRobot-Cell-v0`, following TorchRL's `sota-implementations` layout.

## Run

Laptop smoke run (small GPU):

    python ppo.py env.num_envs=16 collector.rollout_steps=8 max_iterations=3

DGX Spark:

    python ppo.py env.num_envs=4096 logger.backend=wandb

Play a checkpoint (opens the viewer):

    python play.py play.checkpoint=outputs/<date>/<time>/checkpoints/ppo_final.pt

## Observation routing

`network.actor_in_keys` / `network.critic_in_keys` choose which observations each network sees.
Group names expand to all their terms; nested keys select one term.

| Setup | actor_in_keys | critic_in_keys | env flags |
|---|---|---|---|
| State teacher (default) | `[proprio, belt, privileged]` | `[proprio, belt, privileged]` | `cameras=false privileged_information=true` |
| Asymmetric actor-critic | `[proprio, belt, [pixels, wrist_rgb]]` | `[proprio, belt, privileged]` | `cameras=true privileged_information=true` |
| Deployment-like | `[proprio, belt, pixels]` | `[proprio, belt, pixels]` | `cameras=true privileged_information=false` |

Example: `python ppo.py env.cameras=true env.num_envs=256 'network.actor_in_keys=[proprio,belt,[pixels,wrist_rgb]]'`

## Logged metrics

- `train/episode_return`, `train/episode_length`
- `episode_termination/<term>`: fraction of envs whose last episode ended by `success`, `bowl_exited_zone`,
  `bowl_off_belt`, `bowl_tipped`, `food_off_table` or `time_out` — `episode_termination/success` is the success rate.
- `train/loss_objective`, `train/loss_critic`, `train/loss_entropy`
