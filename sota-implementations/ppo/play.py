"""Roll out a PPO checkpoint. Usage: python play.py play.checkpoint=/path/ppo_final.pt [play.num_envs=16]"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf


def _load_local_utils():
    """Load this folder's ``utils.py`` by file path, not by bare ``import utils``.

    Isaac Sim's camera/replicator extensions (loaded when ``enable_cameras=True``) put a bundled
    OpenCV ``cv2/utils`` package where it can shadow a plain ``import utils``/``from utils import
    ...`` regardless of ``sys.path`` order (observed: a top-level ``utils`` resolves to
    ``.../cv2/utils/__init__.py``). Loading by explicit file path and registering the result in
    ``sys.modules`` under the name ``utils`` sidesteps that shadowing.
    """
    import importlib.util
    import sys

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")
    spec = importlib.util.spec_from_file_location("utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if not cfg.play.checkpoint:
        raise ValueError("Set play.checkpoint=/path/to/checkpoint.pt")
    from food_robot.app import launch_app

    # cameras must be set at app launch, before the checkpoint (and its saved env.cameras) can be read;
    # pass env.cameras=<value> matching the checkpoint's training config, or the check below raises.
    launch_app(headless=cfg.play.headless, enable_cameras=cfg.env.cameras, device=cfg.env.device)

    import torch
    from torchrl.envs import ExplorationType, set_exploration_type

    from food_robot.torchrl_env import make_env, termination_stats

    make_ppo_models = _load_local_utils().make_ppo_models

    device = torch.device(cfg.env.device)
    state = torch.load(cfg.play.checkpoint, map_location=device, weights_only=False)
    saved = OmegaConf.create(state["config"])
    if bool(saved.env.cameras) != bool(cfg.env.cameras):
        raise ValueError(
            f"Checkpoint was trained with env.cameras={bool(saved.env.cameras)}, but this run launched "
            f"the app with env.cameras={bool(cfg.env.cameras)}. Isaac Lab must enable/disable cameras at "
            f"app launch time, before the checkpoint's config can be read. Rerun with "
            f"env.cameras={bool(saved.env.cameras)}."
        )
    device = torch.device(saved.env.device)  # build consistently with the checkpoint's saved env config
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
