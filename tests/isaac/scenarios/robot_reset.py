"""robot_reset: DEFAULT_ENV equals today's reset_joints_by_offset params; an env.robot_reset override reaches
the built cfg's reset_robot_joints event params, restricted to the arm joints (not the gripper fingers) in both
cases -- construction only, no gym.make/stepping needed."""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True)

from pickplace.config import build_cell_env_cfg  # noqa: E402

BASE_ENV = {"cameras": False, "privileged_information": True}
OVERRIDE_ENV = {
    **BASE_ENV,
    "robot_reset": {"position_range": [-0.25, 0.25], "velocity_range": [-0.1, 0.1]},
}


def params(env):
    cfg = build_cell_env_cfg(env)
    p = cfg.events.reset_robot_joints.params
    return {
        "position_range": list(p["position_range"]),
        "velocity_range": list(p["velocity_range"]),
        "joint_names": list(p["asset_cfg"].joint_names),
        "arm_joint_names": list(cfg.arm.arm_joint_names),
        "gripper_joint_names": list(cfg.arm.gripper_joint_names),
    }


def main():
    finish(True, default=params(BASE_ENV), override=params(OVERRIDE_ENV))


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
