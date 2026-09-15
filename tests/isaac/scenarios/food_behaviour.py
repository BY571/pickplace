"""Food scenarios: placing food into the bowl -> success; dropping it -> food_off_table; spawn randomization."""

from _common import finish

from food_robot.app import launch_app

app = launch_app(headless=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.belt import BeltCfg  # noqa: E402
from food_robot.envs.cell_env_cfg import FoodCellEnvCfg  # noqa: E402

N = 3


def main():
    belt = BeltCfg(speed_noise=0.0, bowl_offset_x=(0.0, 0.0), bowl_offset_y=(0.0, 0.0))
    cfg = FoodCellEnvCfg(cameras=False, privileged_information=True, belt=belt)
    cfg.scene.num_envs = N
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    device, dt = u.device, u.step_dt
    food, bowl = u.scene["food"], u.scene["bowl"]

    supply_xy = u.scene["ingredient_bowl"].data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]
    spawn_offsets = (obs["privileged"]["food_pos"][:, :2] - supply_xy).tolist()

    action = torch.zeros(N, u.action_manager.total_action_dim, device=device)
    action[:, -1] = 1.0  # binary gripper: positive = open
    names = ["success", "food_off_table", "bowl_off_belt", "bowl_tipped", "bowl_exited_zone", "time_out"]
    first, reward_at, finite = {}, {}, True
    for step in range(int(2.0 / dt)):
        if step == 10:
            ids = torch.tensor([0, 1], device=device)
            pos = food.data.root_pos_w.torch[:2].clone()
            quat = food.data.root_quat_w.torch[:2].clone()
            bowl_pos = bowl.data.root_pos_w.torch[0]
            pos[0] = bowl_pos + torch.tensor([0.0, 0.0, cfg.belt.bowl.base_thickness + cfg.food.item_radius + 0.01], device=device)
            pos[1, 2] = u.scene.env_origins[1, 2] - 0.3
            food.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
            vel = torch.zeros(2, 6, device=device)
            vel[0, :3] = bowl.data.root_lin_vel_w.torch[0]
            food.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        _, reward, *_ = env.step(action)
        finite &= bool(torch.isfinite(reward).all())
        for name in names:
            for i in u.termination_manager.get_term(name).nonzero().flatten().tolist():
                if str(i) not in first:
                    first[str(i)] = name
                    reward_at[str(i)] = float(reward[i])
        if "0" in first and "1" in first and step > 60:
            break
    finish(
        True,
        first_termination=first,
        reward_at_termination=reward_at,
        rewards_finite=finite,
        success_bonus=cfg.success_bonus,
        food_drop_penalty=cfg.food_drop_penalty,
        food_spawn_offsets=spawn_offsets,
        food_spawn_range=cfg.food.spawn_range,
        reward_terms=sorted(u.reward_manager.active_terms),
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
