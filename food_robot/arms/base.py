# food_robot/arms/base.py
"""Arm plug-in configuration: everything the env needs to know about a manipulator."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class ArmCfg:
    robot: ArticulationCfg = MISSING
    """Articulation used with joint-space actions (prim_path is overwritten by the env)."""
    ik_robot: ArticulationCfg = MISSING
    """Articulation with stiffer gains used with differential-IK actions."""
    arm_joint_names: list[str] = MISSING
    gripper_joint_names: list[str] = MISSING
    base_link_name: str = MISSING
    ee_body_name: str = MISSING
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Tool center point offset from the end-effector body frame."""
    gripper_open: float = MISSING
    gripper_closed: float = MISSING
    wrist_cam_offset: CameraCfg.OffsetCfg = MISSING
    """Wrist camera pose relative to the end-effector body."""
    reach_radius: float = MISSING
    """Conservative horizontal reach from the base, used to validate the belt zone."""
    joint_action_scale: float = 0.5
    ik_action_scale: float = 0.5
