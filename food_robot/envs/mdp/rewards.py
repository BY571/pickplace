"""Reward terms (Isaac Lab scales each term by step_dt)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from food_robot.envs.mdp.observations import grasped_mask
from food_robot.envs.mdp.terminations import arm_home_distance, released_in_bowl_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tcp_w(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]


def reach_food(
    env: ManagerBasedRLEnv,
    std: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(env.scene[food_cfg.name].data.root_pos_w.torch - _tcp_w(env, ee_frame_cfg), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Fingers closed on the food near the TCP, no lift required: the stepping stone between reach_food and grasp_lift."""
    return grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos).float()


def grasp_lift(
    env: ManagerBasedRLEnv,
    lift_height: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    grasped = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return (grasped & (food_z > lift_height)).float()


def transport_to_bowl(
    env: ManagerBasedRLEnv,
    std: float,
    hover_height: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    lift_height: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """While the food is actually held (grasped *and* lifted above ``lift_height``), approach a point above the
    bowl's *current* position (a moving target).

    Requiring the lift (not just ``grasped_mask``) closes the exploit where a policy shepherds the food between
    its fingers on the table -- never lifting it -- purely to farm this term (see task-13-grasp-fix.md): that
    non-grasp used to satisfy ``grasped_mask`` alone and paid the full weight-10 reward with no hold required.
    """
    target = env.scene[bowl_cfg.name].data.root_pos_w.torch.clone()
    target[:, 2] += hover_height
    distance = torch.linalg.vector_norm(env.scene[food_cfg.name].data.root_pos_w.torch - target, dim=-1)
    food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    grasped = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    held = grasped & (food_z > lift_height)
    return held.float() * (1.0 - torch.tanh(distance / std))


def released_in_bowl(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """1.0 while the food is inside the bowl and released (same test as the success termination, minus its
    speed and settle requirements). Requiring the release stops a policy from lowering the held food into the
    bowl and hovering there."""
    return released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos, food_cfg, bowl_cfg
    ).float()


def return_home(
    env: ManagerBasedRLEnv,
    std: float,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    arm_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """While the food is released inside the bowl: ``1 - tanh(d / std)``, ``d`` = the arm joints' L2 distance
    [rad] to the default pose (``arm_home_distance``, the same distance the ``success_requires_home`` check uses);
    0 otherwise. Teaches finishing the job: after placing, go back home."""
    placed = released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos, food_cfg, bowl_cfg
    )
    return placed.float() * (1.0 - torch.tanh(arm_home_distance(env, arm_cfg) / std))


def termination_indicator(env: ManagerBasedRLEnv, term_names: list[str]) -> torch.Tensor:
    """1.0 where any of the named termination terms fired this step."""
    fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_names:
        fired |= env.termination_manager.get_term(name)
    return fired.float()


def bowl_disturbance(
    env: ManagerBasedRLEnv,
    belt_event_name: str = "reset_belt",
    pallet_cfg: SceneEntityCfg = SceneEntityCfg("pallet"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Distance [m] the bowl has been pushed away from where the pallet carries it."""
    belt = env.event_manager.get_term_cfg(belt_event_name).func
    pallet = env.scene[pallet_cfg.name].data
    plate_x = pallet.root_pos_w.torch[:, 0] + pallet.joint_pos.torch[:, belt.joint_ids[0]]
    expected = torch.stack([plate_x + belt.offset[:, 0], pallet.root_pos_w.torch[:, 1] + belt.offset[:, 1]], dim=-1)
    bowl_xy = env.scene[bowl_cfg.name].data.root_pos_w.torch[:, :2]
    return torch.linalg.vector_norm(bowl_xy - expected, dim=-1)
