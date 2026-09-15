"""Roll out a PPO checkpoint. Usage: python play.py play.checkpoint=/path/ppo_final.pt [play.num_envs=16]"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if not cfg.play.checkpoint:
        raise ValueError("Set play.checkpoint=/path/to/checkpoint.pt")
    from food_robot.app import launch_app

    # cameras must match the checkpoint's env; cfg.env mirrors it unless overridden on the CLI
    launch_app(headless=cfg.play.headless, enable_cameras=cfg.env.cameras, device=cfg.env.device)

    import torch
    from torchrl.envs import ExplorationType, set_exploration_type

    from food_robot.torchrl_env import make_env, termination_stats
    from utils import make_ppo_models

    device = torch.device(cfg.env.device)
    state = torch.load(cfg.play.checkpoint, map_location=device, weights_only=False)
    saved = OmegaConf.create(state["config"])
    env = make_env({**OmegaConf.to_container(saved.env, resolve=True), "num_envs": cfg.play.num_envs})
    actor, _ = make_ppo_models(env, saved.network, device)
    actor.load_state_dict(state["actor"])
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.rollout(cfg.play.steps, actor, break_when_any_done=False)
    done = td["next", "done"]
    if done.any():
        print("episode return:", td["next", "episode_reward"][done].mean().item())
    print("termination stats:", termination_stats(env))
    print("PLAY_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
