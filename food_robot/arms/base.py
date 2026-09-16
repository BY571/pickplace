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
    gripper_body_names: list[str] = MISSING
    """Rigid body names the fingers spawn as, used to apply ``finger_friction`` (e.g. via a startup
    ``randomize_rigid_body_material`` event); distinct from ``gripper_joint_names``, which drives the gripper
    action and observations."""
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
    finger_friction: tuple[float, float] = (1.2, 1.0)
    """(static, dynamic) friction applied to ``gripper_body_names`` by the env's ``gripper_material`` startup
    event. Without this, the fingers spawn with no material of their own and inherit PhysX's 0.5/0.5 default,
    which (combine mode unset, i.e. averaged against the food's material) is too slippery to lift a smooth
    sphere gripped by flat pads -- see task-13-grasp-fix.md."""
