"""Gym registration. Importing this module is cheap; the env config imports Isaac Lab lazily via gym.make."""

import gymnasium as gym

gym.register(
    id="FoodRobot-Cell-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.cell_env_cfg:FoodCellEnvCfg"},
    disable_env_checker=True,
)
