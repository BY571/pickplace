"""Franka Emika Panda with parallel gripper."""

from isaaclab.sensors import CameraCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

from food_robot.arms.base import ArmCfg


def _with_legacy_usd(cfg):
    """Work around Nucleus content drift: the live Isaac 6.0 asset pack moved this file under
    ``Legacy/``, but ``isaaclab_assets`` (pinned at v3.0.0-beta2.patch1) still points at the old,
    now-404 path. See task-4-report.md for evidence.
    """
    cfg = cfg.copy()
    cfg.spawn.usd_path = cfg.spawn.usd_path.replace(
        "FrankaEmika/panda_instanceable.usd", "FrankaEmika/Legacy/panda_instanceable.usd"
    )
    return cfg


FRANKA_CFG = ArmCfg(
    robot=_with_legacy_usd(FRANKA_PANDA_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot"),
    ik_robot=_with_legacy_usd(FRANKA_PANDA_HIGH_PD_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot"),
    arm_joint_names=["panda_joint.*"],
    gripper_joint_names=["panda_finger_joint.*"],
    base_link_name="panda_link0",
    ee_body_name="panda_hand",
    tcp_offset=(0.0, 0.0, 0.1034),
    gripper_open=0.04,
    gripper_closed=0.0,
    # same mount as Isaac Lab's Franka visuomotor stack task (xyzw, ROS convention)
    wrist_cam_offset=CameraCfg.OffsetCfg(
        pos=(0.13, 0.0, -0.15), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
    ),
    reach_radius=0.80,
)
