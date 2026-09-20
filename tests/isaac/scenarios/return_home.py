"""v3 semantics (success_requires_home): success = food settled in the bowl AND the TCP back home.

Home = the TCP within home_tolerance [m] of its home position (``mdp.tcp_home_distance``, ``ArmCfg.home_tcp_pos``).
The first steps drive both arms to their default joint pose; every env's TCP must then count as home.
Env 0: joint 1 is teleported 0.5 rad away from its default (the TCP swings ~0.2 m sideways) and the food is
teleported to rest in the bowl; the success termination must not fire although the food settles (its counter passes
settle_steps + 5), while ``return_home`` and ``food_in_bowl`` pay. Then the arm is written back to the default
joints and success must fire within a few steps. Env 1 is an untouched control.

Runs in ``joint_pos`` action mode so the arm's joint targets are exactly what the scenario writes (the arm action
``a`` targets ``default + scale * a``); with the IK action the controller's target would depend on when the
end-effector pose is refreshed after a teleport. The home check itself does not depend on the action mode.
The robot is the training articulation (``ArmCfg.ik_robot``: gravity-free, stiff gains); the low-gain
``ArmCfg.robot`` that ``joint_pos`` mode would use sags ~0.2 m below its default pose under gravity.
"""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pickplace.envs  # noqa: E402,F401
from pickplace.config import build_cell_env_cfg  # noqa: E402
from pickplace.envs import mdp  # noqa: E402

N = 2
FAR = 0.5  # [rad] offset of joint 1 in phase (a): moves the TCP well outside home_tolerance
ENV = {
    "num_envs": N,
    "cameras": False,
    "privileged_information": True,
    "action_mode": "joint_pos",
    "reward_set": "simple_v3",
    "success_requires_home": True,
    "belt": {"speed_noise": 0.0, "bowl_offset_x": [0.0, 0.0], "bowl_offset_y": [0.0, 0.0]},
}


def main():
    cfg = build_cell_env_cfg(ENV)
    cfg.scene.robot = cfg.arm.ik_robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    env.reset()
    u = env.unwrapped
    device = u.device
    robot, food, bowl = u.scene["robot"], u.scene["food"], u.scene["bowl"]
    tol, settle_steps = cfg.home_tolerance, cfg.success_settle_steps
    home = cfg.arm.home_tcp_pos
    joint_ids, _ = robot.find_joints(cfg.arm.arm_joint_names)  # the arm action term's joint order
    default = robot.data.default_joint_pos.torch[:, joint_ids].clone()
    success_term = u.termination_manager.get_term_cfg("success").func  # holds the settle counter
    rm = u.reward_manager
    names = list(rm.active_terms)

    def component(term):  # unweighted per-second value of a dense term (Isaac Lab stores value x weight)
        return rm._step_reward[:, names.index(term)] / rm.get_term_cfg(term).weight

    scale = cfg.arm.joint_action_scale
    action = torch.zeros(N, u.action_manager.total_action_dim, device=device)
    action[:, -1] = 1.0  # binary gripper: positive = open
    env0 = torch.tensor([0], device=device)

    def write_arm(q):
        robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids, env_ids=env0)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=joint_ids, env_ids=env0)
        robot.set_joint_position_target_index(target=q, joint_ids=joint_ids, env_ids=env0)

    # (c) start pose: joint_pos action 0 drives both arms from default + reset offset to the default joints
    first_step_distance = None
    for _ in range(10):
        env.step(action)
        if first_step_distance is None:  # reported only: the arms still carry their reset offsets here
            first_step_distance = mdp.tcp_home_distance(u, home).tolist()
    reset_distance = mdp.tcp_home_distance(u, home).tolist()

    # (a) arm far from home, food at rest in the bowl
    far = default[:1].clone()
    far[:, 0] -= FAR
    write_arm(far)
    action[0, : len(joint_ids)] = (far[0] - default[0]) / scale  # hold it there
    pos, quat = food.data.root_pos_w.torch[:1].clone(), food.data.root_quat_w.torch[:1].clone()
    pos[0] = bowl.data.root_pos_w.torch[0] + torch.tensor(
        [0.0, 0.0, cfg.belt.bowl.base_thickness + cfg.food.item_radius + 0.01], device=device
    )
    food.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env0)
    vel = torch.zeros(1, 6, device=device)
    vel[0, :3] = bowl.data.root_lin_vel_w.torch[0]  # the bowl rides the belt; match it so the food settles
    food.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env0)

    fired_away = other_done = False
    home_max, in_bowl_max = torch.zeros(N), torch.zeros(N)
    away_distance_min = float("inf")
    for _ in range(60):
        _, _, terminated, truncated, _ = env.step(action)
        fired_away |= bool(u.termination_manager.get_term("success")[0])
        other_done |= bool(terminated[0] | truncated[0])
        home_max = torch.maximum(home_max, component("return_home").cpu())
        in_bowl_max = torch.maximum(in_bowl_max, component("food_in_bowl").cpu())
        away_distance_min = min(away_distance_min, float(mdp.tcp_home_distance(u, home)[0]))
    counter_away = int(success_term.counter[0])

    # (b) arm back at the default pose: success must fire within a few steps
    write_arm(default[:1].clone())
    action[0, : len(joint_ids)] = 0.0  # joint_pos action 0 = the default pose
    fired_home_after = None
    for k in range(1, 11):
        env.step(action)
        if bool(u.termination_manager.get_term("success")[0]):
            fired_home_after = k
            break

    finish(
        True,
        home_tolerance=tol,
        settle_steps=settle_steps,
        reset_distance=reset_distance,
        first_step_distance=first_step_distance,
        fired_away=fired_away,
        env0_done_while_away=other_done,
        counter_away=counter_away,
        away_distance_min=away_distance_min,
        return_home_max=home_max.tolist(),
        food_in_bowl_max=in_bowl_max.tolist(),
        fired_home_after=fired_home_after,
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
