"""Franka Emika Panda with parallel gripper."""

from isaaclab.sensors import CameraCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

from food_robot.arms.base import ArmCfg


_OLD_USD_SUFFIX = "FrankaEmika/panda_instanceable.usd"
_NEW_USD_SUFFIX = "FrankaEmika/Legacy/panda_instanceable.usd"


def _with_legacy_usd(cfg):
    """Work around Nucleus content drift: the live Isaac 6.0 asset pack moved this file under
    ``Legacy/``, but ``isaaclab_assets`` (pinned at v3.0.0-beta2.patch1) still points at the old,
    now-404 path. See task-4-report.md for evidence.

    ``cfg.copy()`` is safe to mutate here: every ``@configclass`` (``ArticulationCfg``,
    ``UsdFileCfg``, ...) gets a ``__post_init__`` injected by
    ``isaaclab.utils.configclass._custom_post_init``, which "deepcopy[ies] all elements to avoid
    shared memory issues for mutable objects" (configclass.py:479-486). ``configclass.copy()``
    (``_copy_class``, configclass.py:183-185) is ``dataclasses.replace(obj)``, which re-runs
    ``__init__``/``__post_init__``, so the copy's nested ``spawn`` is an independent object from the
    ``isaaclab_assets`` singleton's. Verified on the Spark: mutating ``cfg.copy().spawn.usd_path``
    left ``FRANKA_PANDA_CFG.spawn.usd_path`` (and ``FRANKA_PANDA_CFG.spawn.rigid_props`` identity)
    untouched.
    """
    cfg = cfg.copy()
    path = cfg.spawn.usd_path
    if _NEW_USD_SUFFIX in path:
        return cfg  # already patched upstream: accept as-is (idempotent)
    if _OLD_USD_SUFFIX not in path:
        raise RuntimeError(
            f"Expected the Franka USD path to contain {_OLD_USD_SUFFIX!r} (or already-patched "
            f"{_NEW_USD_SUFFIX!r}) so food_robot/arms/franka.py can work around Nucleus content "
            f"drift, but got: {path!r}. isaaclab_assets or the Nucleus layout changed again; update "
            f"_OLD_USD_SUFFIX/_NEW_USD_SUFFIX in this file."
        )
    cfg.spawn.usd_path = path.replace(_OLD_USD_SUFFIX, _NEW_USD_SUFFIX)
    return cfg


FRANKA_CFG = ArmCfg(
    robot=_with_legacy_usd(FRANKA_PANDA_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot"),
    ik_robot=_with_legacy_usd(FRANKA_PANDA_HIGH_PD_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot"),
    arm_joint_names=["panda_joint.*"],
    gripper_joint_names=["panda_finger_joint.*"],
    gripper_body_names=["panda_leftfinger", "panda_rightfinger"],
    base_link_name="panda_link0",
    ee_body_name="panda_hand",
    tcp_offset=(0.0, 0.0, 0.1034),
    # TCP at the default joint pose, env-local [m]: measured 2026-09-19 on the Spark with scripts/measure_home_tcp.py
    # (ik_robot, default joints held by joint-position targets; joint error 0).
    home_tcp_pos=(0.4633, 0.0, 0.3855),
    gripper_open=0.04,
    gripper_closed=0.0,
    # Mounted on the back of the hand, off the finger axis (fingers slide along the hand's y axis), and tilted
    # about the hand's y axis so the optical axis (+z in ROS convention) looks past the fingertips at a point
    # 0.28 m ahead of the hand: from (0.10, 0, -0.03) towards (0, 0, 0.25) is a -19.7 deg rotation about y
    # (xyzw). Mounted closer, the fingers and hand body fill the image.
    wrist_cam_offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, -0.03), rot=(0.0, -0.1711, 0.0, 0.9853), convention="ros"),
    reach_radius=0.80,
)
