# food_robot/envs/mdp/observations.py
"""Observation terms. Positions are expressed in the cell frame (env origin); quaternions are xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def image_float(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Raw (unnormalized) camera image cast to float32; values stay in [0, 255] for rgb.

    Isaac Lab's ``ManagerBasedRLEnv`` builds every observation term's gym space as
    ``gym.spaces.Box(..., dtype=np.float32)`` regardless of the term's actual output dtype
    (``manager_based_rl_env.py``). The stock ``isaaclab.envs.mdp.image(normalize=False)`` returns
    the camera sensor's native dtype (``uint8`` for rgb), which makes TorchRL's ``check_env_specs``
    reject the mismatch between the declared float32 spec and the real uint8 tensor (verified on the
    Spark: ``AssertionError: ... Got fake=torch.float32 and real=torch.uint8``). Cast here instead of
    relaxing the spec, per deviation #2 in the plan overview.

    Memory note: a float32 RGB buffer is 4x the size of the native uint8 RGB buffer the camera
    sensor produces (4 bytes/channel vs. 1), and TorchRL stores one copy per transition at both the
    root and "next" keys. See ``docs/environment.md`` for the per-frame memory formula and
    recommended ``num_envs`` ranges when training with cameras on the Spark.
    """
    from isaaclab.envs import mdp as base_mdp

    return base_mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False).float()


def asset_pos_cell(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins


def asset_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_quat_w.torch.clone()


def ee_pos_cell(env: ManagerBasedEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :] - env.scene.env_origins


def ee_quat_w(env: ManagerBasedEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_quat_w.torch[:, 0, :].clone()


def gripper_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].clone()


def grasped_mask(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distance_threshold: float = 0.04,
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    margin: float = 0.004,
) -> torch.Tensor:
    """Food near the TCP while fingers stopped between open and closed (i.e. blocked by the food)."""
    food_pos = env.scene[food_cfg.name].data.root_pos_w.torch
    tcp_pos = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
    near = torch.linalg.vector_norm(food_pos - tcp_pos, dim=-1) < distance_threshold
    fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
    blocked = ((fingers < open_pos - margin) & (fingers > closed_pos + margin)).all(dim=-1)
    return near & blocked


def is_grasped(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distance_threshold: float = 0.04,
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    margin: float = 0.004,
) -> torch.Tensor:
    mask = grasped_mask(env, robot_cfg, food_cfg, ee_frame_cfg, distance_threshold, open_pos, closed_pos, margin)
    return mask.float().unsqueeze(-1)
