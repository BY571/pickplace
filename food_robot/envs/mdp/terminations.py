"""Termination terms. Positions in the cell frame; quaternions xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from food_robot.envs.mdp.observations import grasped_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pos_cell(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[cfg.name].data.root_pos_w.torch - env.scene.env_origins


def bowl_exited_zone(env: ManagerBasedRLEnv, zone_end_x: float, bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl")):
    """The bowl passed the end of the reach zone: the placement deadline is over."""
    return _pos_cell(env, bowl_cfg)[:, 0] > zone_end_x


def bowl_off_belt(
    env: ManagerBasedRLEnv,
    belt_y: float,
    belt_half_width: float,
    surface_z: float,
    z_margin: float = 0.02,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
):
    """The bowl dropped below the belt surface or left the belt laterally."""
    pos = _pos_cell(env, bowl_cfg)
    return (pos[:, 2] < surface_z - z_margin) | ((pos[:, 1] - belt_y).abs() > belt_half_width)


def bowl_tipped(env: ManagerBasedRLEnv, max_tilt_rad: float, bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl")):
    """The bowl's up-axis tilted more than ``max_tilt_rad`` from vertical."""
    q = env.scene[bowl_cfg.name].data.root_quat_w.torch  # (x, y, z, w)
    up_z = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    return up_z < torch.cos(torch.tensor(max_tilt_rad, device=q.device))


def food_off_table(env: ManagerBasedRLEnv, minimum_height: float = -0.05, food_cfg: SceneEntityCfg = SceneEntityCfg("food")):
    """The food fell below the table surface."""
    return _pos_cell(env, food_cfg)[:, 2] < minimum_height


def released_in_bowl_mask(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The food is inside the bowl (radially within the inner wall, between base and rim) and not grasped.

    The single definition of "in the bowl" shared by the ``food_in_bowl`` success termination and the
    ``released_in_bowl`` reward, so the two can never disagree.
    """
    food_pos, bowl_pos = env.scene[food_cfg.name].data.root_pos_w.torch, env.scene[bowl_cfg.name].data.root_pos_w.torch
    rel = food_pos - bowl_pos  # bowl origin = bottom center
    radial_ok = torch.linalg.vector_norm(rel[:, :2], dim=-1) < inner_radius - 0.5 * item_radius
    height_ok = (rel[:, 2] > base_thickness) & (rel[:, 2] < base_thickness + rim_height)
    released = ~grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return radial_ok & height_ok & released


REL_SPEED_THRESHOLD = 0.05
"""Food-to-bowl speed [m/s] below which the food counts as at rest in the bowl (``settled_in_bowl_mask``)."""


def settled_in_bowl_mask(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
    rel_speed_threshold: float = REL_SPEED_THRESHOLD,
) -> torch.Tensor:
    """``released_in_bowl_mask`` and moving slower than ``rel_speed_threshold`` relative to the bowl.

    One step of the ``food_in_bowl`` success condition (which additionally requires it for ``settle_steps``
    consecutive steps); the continuous demo counts placements with the same function.
    """
    food, bowl = env.scene[food_cfg.name].data, env.scene[bowl_cfg.name].data
    in_bowl = released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos,
        food_cfg, bowl_cfg,
    )
    rel_speed = torch.linalg.vector_norm(food.root_lin_vel_w.torch - bowl.root_lin_vel_w.torch, dim=-1)
    return in_bowl & (rel_speed < rel_speed_threshold)


def never(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A termination that never fires (the continuous demo disables every episode end with it)."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class food_in_bowl(ManagerTermBase):
    """Success: food inside the target bowl, released, and at rest relative to the bowl for ``settle_steps``."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.counter.zero_()
        else:
            self.counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        inner_radius: float,
        base_thickness: float,
        rim_height: float,
        item_radius: float,
        settle_steps: int,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        rel_speed_threshold: float = REL_SPEED_THRESHOLD,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    ) -> torch.Tensor:
        inside = settled_in_bowl_mask(
            env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos,
            food_cfg, bowl_cfg, rel_speed_threshold,
        )
        self.counter = torch.where(inside, self.counter + 1, torch.zeros_like(self.counter))
        return self.counter >= settle_steps
