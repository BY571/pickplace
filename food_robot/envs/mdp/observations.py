# food_robot/envs/mdp/observations.py
"""Observation terms. Positions are expressed in the cell frame (env origin); quaternions are xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
