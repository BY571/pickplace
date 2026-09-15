"""Reset events for the conveyor belt and the ingredient (supply) bowl."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_belt(ManagerTermBase):
    """Put the pallet at a random start near the belt entry, sample the belt speed and place the bowl on it.

    Stores per-env ``speed`` (N,), pallet ``start`` joint position (N,) and bowl ``offset`` (N, 2) for other
    terms (e.g. the disturbance reward).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.pallet = env.scene[cfg.params["pallet_cfg"].name]
        self.bowl = env.scene[cfg.params["bowl_cfg"].name]
        self.joint_ids, _ = self.pallet.find_joints("slider")
        self.speed = torch.zeros(env.num_envs, device=env.device)
        self.start = torch.zeros(env.num_envs, device=env.device)
        self.offset = torch.zeros(env.num_envs, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        speed: float,
        speed_noise: float,
        pallet_start_range: tuple[float, float],
        bowl_offset_x: tuple[float, float],
        bowl_offset_y: tuple[float, float],
        entry_x: float,
        belt_y: float,
        plate_top_z: float,
        pallet_cfg: SceneEntityCfg = SceneEntityCfg("pallet"),
        bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    ):
        n, device = len(env_ids), env.device
        v = speed * (1.0 + math_utils.sample_uniform(-speed_noise, speed_noise, (n,), device))
        start = math_utils.sample_uniform(*pallet_start_range, (n,), device)
        ox = math_utils.sample_uniform(*bowl_offset_x, (n,), device)
        oy = math_utils.sample_uniform(*bowl_offset_y, (n,), device)
        self.speed[env_ids] = v
        self.start[env_ids] = start
        self.offset[env_ids, 0] = ox
        self.offset[env_ids, 1] = oy

        joint_v = v.unsqueeze(-1)
        self.pallet.write_joint_position_to_sim_index(
            position=start.unsqueeze(-1), joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.pallet.write_joint_velocity_to_sim_index(velocity=joint_v, joint_ids=self.joint_ids, env_ids=env_ids)
        self.pallet.set_joint_velocity_target_index(target=joint_v, joint_ids=self.joint_ids, env_ids=env_ids)

        origins = env.scene.env_origins[env_ids]
        pose = torch.zeros(n, 7, device=device)
        pose[:, 0] = origins[:, 0] + entry_x + start + ox
        pose[:, 1] = origins[:, 1] + belt_y + oy
        pose[:, 2] = origins[:, 2] + plate_top_z + 0.002
        pose[:, 6] = 1.0  # identity quaternion (x, y, z, w)
        self.bowl.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        velocity = torch.zeros(n, 6, device=device)
        velocity[:, 0] = v
        self.bowl.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


class reset_ingredient_bowl(ManagerTermBase):
    """Place the kinematic ingredient (supply) bowl at a random position on the table.

    Stores the sampled world-frame root positions in ``positions`` (N, 3) so the food reset can spawn the food
    inside the bowl's new position in the same reset (asset data buffers are not refreshed until the next
    simulation step).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.bowl = env.scene[cfg.params["bowl_cfg"].name]
        self.positions = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z: float,
        bowl_cfg: SceneEntityCfg = SceneEntityCfg("ingredient_bowl"),
    ):
        n, device = len(env_ids), env.device
        origins = env.scene.env_origins[env_ids]
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = origins[:, 0] + math_utils.sample_uniform(*x_range, (n,), device)
        pos[:, 1] = origins[:, 1] + math_utils.sample_uniform(*y_range, (n,), device)
        pos[:, 2] = origins[:, 2] + z
        self.positions[env_ids] = pos
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = pos
        pose[:, 6] = 1.0
        self.bowl.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)


def reset_food_in_bowl(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    spawn_range: float,
    height_above_bowl: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_event_name: str = "reset_ingredient_bowl",
):
    """Spawn the food at rest inside the ingredient bowl's position from this reset, +-spawn_range in x and y."""
    n, device = len(env_ids), env.device
    bowl_pos = env.event_manager.get_term_cfg(bowl_event_name).func.positions[env_ids]
    pose = torch.zeros(n, 7, device=device)
    pose[:, 0] = bowl_pos[:, 0] + math_utils.sample_uniform(-spawn_range, spawn_range, (n,), device)
    pose[:, 1] = bowl_pos[:, 1] + math_utils.sample_uniform(-spawn_range, spawn_range, (n,), device)
    pose[:, 2] = bowl_pos[:, 2] + height_above_bowl
    pose[:, 6] = 1.0
    food = env.scene[food_cfg.name]
    food.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    food.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids)
