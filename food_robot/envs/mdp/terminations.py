"""Termination terms. Positions in the cell frame; quaternions xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

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
