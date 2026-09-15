"""TorchRL entry point: the official IsaacLabWrapper plus generic bookkeeping transforms."""

from __future__ import annotations

from collections.abc import Mapping


def make_env(env_cfg: Mapping):
    """Create the food-cell env as a TorchRL ``TransformedEnv``. Launch the Isaac app before calling.

    Observations stay raw (uint8-valued float images, unnormalized states); preprocessing belongs to
    each algorithm. No running-statistics transforms here: terminal next-observations are NaN
    under native auto-reset.
    """
    import gymnasium as gym
    import torch
    from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    import food_robot.envs  # noqa: F401  (gym registration)
    from food_robot.config import DEFAULT_ENV, build_cell_env_cfg

    cfg = build_cell_env_cfg(env_cfg)
    task = env_cfg.get("task", DEFAULT_ENV["task"])
    device = torch.device(env_cfg.get("device", DEFAULT_ENV["device"]))
    base = IsaacLabWrapper(gym.make(task, cfg=cfg), native_autoreset=True, device=device)
    return TransformedEnv(base, Compose(RewardSum(), StepCounter()))


def termination_stats(env) -> dict[str, float]:
    """Fraction of sub-envs whose most recent finished episode ended by each termination term."""
    base = env.base_env if hasattr(env, "base_env") else env
    log = base._env.unwrapped.extras.get("log", {})
    prefix = "Episode_Termination/"
    return {k[len(prefix):]: float(v) for k, v in log.items() if k.startswith(prefix)}
