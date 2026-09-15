"""Belt scenarios. Mode 'events': timing, push-off and tip-over. Mode 'randomization': speed noise and offsets."""

import math
import sys

from _common import finish

MODE = sys.argv[1]

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.belt import BeltCfg  # noqa: E402
from food_robot.envs.cell_env_cfg import FoodCellEnvCfg  # noqa: E402

TERMS = ["bowl_off_belt", "bowl_tipped", "bowl_exited_zone", "time_out"]


def make(belt: BeltCfg, num_envs: int):
    cfg = FoodCellEnvCfg(cameras=False, privileged_information=True, belt=belt)
    cfg.scene.num_envs = num_envs
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    env.reset()
    return env, cfg


def events():
    belt = BeltCfg(speed_noise=0.0, bowl_offset_x=(0.0, 0.0), bowl_offset_y=(0.0, 0.0))
    env, cfg = make(belt, 4)
    u = env.unwrapped
    bowl, dt, device = u.scene["bowl"], u.step_dt, u.device
    action = torch.zeros(4, u.action_manager.total_action_dim, device=device)
    disturb_step = 25
    first: dict[str, list] = {}
    for step in range(int(7.5 / dt)):
        if step == disturb_step:
            pos = bowl.data.root_pos_w.torch.clone()
            quat = bowl.data.root_quat_w.torch.clone()
            pos[0, 1] += 0.3  # env 0: push sideways off the belt
            pos[1, 2] += 0.08  # env 1: lift and tilt 60 deg about x
            half = math.radians(60.0) / 2
            quat[1] = torch.tensor([math.sin(half), 0.0, 0.0, math.cos(half)], device=device)
            pose = torch.cat([pos, quat], dim=-1)[:2]
            bowl.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0, 1], device=device))
        env.step(action)
        for name in TERMS:
            fired = u.termination_manager.get_term(name)
            for i in fired.nonzero().flatten().tolist():
                first.setdefault(str(i), [name, (step + 1) * dt])
        if len(first) == 4:
            break
    zone = cfg.belt.zone()
    finish(
        len(first) == 4,
        first_termination=first,
        disturb_time=(disturb_step + 1) * dt,
        entry_margin=cfg.belt.entry_margin,
        zone_length=zone.length,
        speed=cfg.belt.speed,
    )


def randomization():
    belt = BeltCfg(speed_noise=0.1, bowl_offset_x=(-0.03, 0.03), bowl_offset_y=(-0.02, 0.02))
    env, cfg = make(belt, 16)
    u = env.unwrapped
    action = torch.zeros(16, u.action_manager.total_action_dim, device=u.device)
    term = u.event_manager.get_term_cfg("reset_belt").func
    for _ in range(int(1.0 / u.step_dt)):
        env.step(action)
    pallet = u.scene["pallet"]
    speeds = pallet.data.joint_vel.torch[:, term.joint_ids[0]].tolist()
    finish(
        True,
        pallet_speeds=speeds,
        speed=cfg.belt.speed,
        speed_noise=cfg.belt.speed_noise,
        bowl_offset_x=term.offset[:, 0].tolist(),
    )


try:
    {"events": events, "randomization": randomization}[MODE]()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
