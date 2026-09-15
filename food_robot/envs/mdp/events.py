"""Reset events for the conveyor belt."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_belt(ManagerTermBase):
    """Put the pallet at the belt entry, sample the belt speed and place the bowl with a random offset.

    Stores per-env ``speed`` (N,) and bowl ``offset`` (N, 2) for other terms (e.g. the disturbance reward).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.pallet = env.scene[cfg.params["pallet_cfg"].name]
        self.bowl = env.scene[cfg.params["bowl_cfg"].name]
        self.joint_ids, _ = self.pallet.find_joints("slider")
        self.speed = torch.zeros(env.num_envs, device=env.device)
        self.offset = torch.zeros(env.num_envs, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        speed: float,
        speed_noise: float,
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
        ox = math_utils.sample_uniform(*bowl_offset_x, (n,), device)
        oy = math_utils.sample_uniform(*bowl_offset_y, (n,), device)
        self.speed[env_ids] = v
        self.offset[env_ids, 0] = ox
        self.offset[env_ids, 1] = oy

        joint_v = v.unsqueeze(-1)
        self.pallet.write_joint_position_to_sim_index(
            position=torch.zeros_like(joint_v), joint_ids=self.joint_ids, env_ids=env_ids
        )
        self.pallet.write_joint_velocity_to_sim_index(velocity=joint_v, joint_ids=self.joint_ids, env_ids=env_ids)
        self.pallet.set_joint_velocity_target_index(target=joint_v, joint_ids=self.joint_ids, env_ids=env_ids)

        origins = env.scene.env_origins[env_ids]
        pose = torch.zeros(n, 7, device=device)
        pose[:, 0] = origins[:, 0] + entry_x + ox
        pose[:, 1] = origins[:, 1] + belt_y + oy
        pose[:, 2] = origins[:, 2] + plate_top_z + 0.002
        pose[:, 6] = 1.0  # identity quaternion (x, y, z, w)
        self.bowl.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        velocity = torch.zeros(n, 6, device=device)
        velocity[:, 0] = v
        self.bowl.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
