"""Print the TorchRL spec tree, run check_env_specs and a short random rollout.

Usage: python scripts/check_env.py env.num_envs=4 env.cameras=false env.privileged_information=true
"""

import os
import sys

from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
env_cfg = OmegaConf.to_container(cli.env, resolve=True) if "env" in cli else {}

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=bool(env_cfg.get("cameras", True)))

from torchrl.envs.utils import check_env_specs  # noqa: E402

from food_robot.torchrl_env import make_env, termination_stats  # noqa: E402

env = make_env(env_cfg)
print("batch_size:", env.batch_size)
print("observation_spec:\n", env.observation_spec)
print("action_spec:\n", env.action_spec)
check_env_specs(env, break_when_any_done="both")
td = env.rollout(50, break_when_any_done=False)
print("rollout:", td)
print("termination stats:", termination_stats(env))
print("CHECK_ENV_OK", flush=True)
os._exit(0)
