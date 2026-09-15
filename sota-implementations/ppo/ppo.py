"""PPO on the food cell env. Usage: python ppo.py env.num_envs=4096 [key=value ...]"""

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
    from food_robot.app import launch_app

    launch_app(headless=cfg.app.headless, enable_cameras=cfg.env.cameras, device=cfg.env.device)

    import torch
    import tqdm
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger

    from food_robot.torchrl_env import make_env, termination_stats

    _utils = _load_local_utils()
    make_ppo_models, save_checkpoint = _utils.make_ppo_models, _utils.save_checkpoint

    torch.manual_seed(cfg.env.seed)
    device = torch.device(cfg.env.device)
    env = make_env(OmegaConf.to_container(cfg.env, resolve=True))
    num_envs = env.batch_size[0]
    frames_per_batch = num_envs * cfg.collector.rollout_steps
    if cfg.max_iterations:
        total_iterations = int(cfg.max_iterations)
    else:
        if cfg.collector.total_frames < frames_per_batch:
            raise ValueError(
                f"collector.total_frames ({cfg.collector.total_frames}) is smaller than frames_per_batch "
                f"({frames_per_batch} = env.num_envs [{num_envs}] * collector.rollout_steps "
                f"[{cfg.collector.rollout_steps}]), so no iteration would run. Set max_iterations "
                "explicitly or increase collector.total_frames."
            )
        total_iterations = cfg.collector.total_frames // frames_per_batch
    if total_iterations < 1:
        raise ValueError(f"total_iterations must be >= 1 (computed {total_iterations}); check max_iterations/total_frames.")
    mini_batch_size = frames_per_batch // cfg.loss.num_minibatches

    actor, critic = make_ppo_models(env, cfg.network, device)
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_iterations * frames_per_batch,
        device=device,
        no_cuda_sync=True,
        trust_policy=True,
    )
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=mini_batch_size,
    )
    adv_module = GAE(
        gamma=cfg.loss.gamma, lmbda=cfg.loss.gae_lambda, value_network=critic, average_gae=False, device=device
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr, eps=1e-5)

    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            cfg.logger.backend,
            logger_name="logs",
            experiment_name=generate_exp_name("PPO", cfg.logger.exp_name),
            wandb_kwargs={"config": OmegaConf.to_container(cfg, resolve=True), "project": cfg.logger.project_name},
        )

    frames = 0
    pbar = tqdm.tqdm(total=total_iterations * frames_per_batch)
    for iteration, data in enumerate(collector):
        frames += data.numel()
        pbar.update(data.numel())
        metrics = {}
        done = data["next", "done"]
        if done.any():
            # Under IsaacLabWrapper's native_autoreset=True, ("next", "episode_reward") and
            # ("next", "step_count") are already reset (NaN / 0, respectively) on the very row where
            # done=True -- the auto-reset happens inside that same env.step() call, so TorchRL treats
            # the returned "next" as the start of the following episode (see docs/environment.md and
            # tests/isaac/test_torchrl_episodes.py). The completed episode's return and length are
            # reconstructed from the pre-step root tensordict (still valid) plus this step's reward.
            completed_return = data["episode_reward"] + data["next", "reward"]
            completed_length = data["step_count"] + 1
            metrics["train/episode_return"] = completed_return[done].mean().item()
            metrics["train/episode_length"] = completed_length[done].float().mean().item()
        metrics.update({f"episode_termination/{k}": v for k, v in termination_stats(env).items()})

        loss_sums: dict[str, float] = {}
        num_updates = 0
        for _ in range(cfg.loss.ppo_epochs):
            with torch.no_grad():
                data = adv_module(data)
            # ClipPPOLoss only reads root observations, action, log-prob, advantage and value_target;
            # drop the "next" sub-tensordict (~4x larger for float32 camera frames, see
            # food_robot/envs/mdp/observations.py::image_float and docs/environment.md) before it goes
            # into the buffer.
            buffer.extend(data.exclude("next").reshape(-1))
            for batch in buffer:
                loss = loss_module(batch)
                total = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                optim.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.optim.max_grad_norm)
                optim.step()
                for key, value in loss.items():
                    if key.startswith("loss_"):
                        loss_sums[key] = loss_sums.get(key, 0.0) + value.detach().item()
                num_updates += 1
        metrics.update({f"train/{k}": v / num_updates for k, v in loss_sums.items()})

        if cfg.optim.anneal_lr:
            for group in optim.param_groups:
                group["lr"] = cfg.optim.lr * (1.0 - (iteration + 1) / total_iterations)
        collector.update_policy_weights_()

        if logger is not None:
            for key, value in metrics.items():
                logger.log_scalar(key, value, step=frames)
        if cfg.checkpoint.interval_iterations and (iteration + 1) % cfg.checkpoint.interval_iterations == 0:
            save_checkpoint(f"checkpoints/ppo_{frames}.pt", actor, critic, optim, cfg, frames)

    save_checkpoint("checkpoints/ppo_final.pt", actor, critic, optim, cfg, frames)
    collector.shutdown()
    print("PPO_DONE", flush=True)
    os._exit(0)  # Isaac Sim shutdown can hang


if __name__ == "__main__":
    main()
