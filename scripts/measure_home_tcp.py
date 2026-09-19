"""Print the arm's TCP position (env-local, [m]) at its default joint pose -- the value of ``ArmCfg.home_tcp_pos``.

One env with the training articulation (``ArmCfg.ik_robot``: gravity-free, stiff gains) driven by joint-position
actions of 0 (= hold the default joints; the low-gain ``ArmCfg.robot`` sags ~0.2 m under gravity): write the default
joints, step so the FrameTransformer refreshes, print the ``ee_frame`` target minus the env origin.

Usage: python scripts/measure_home_tcp.py [arm=franka]
"""

import os
import sys

from food_robot.app import launch_app

app = launch_app(headless=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.config import build_cell_env_cfg  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:])
cfg = build_cell_env_cfg(
    {"arm": args.get("arm", "franka"), "num_envs": 1, "cameras": False, "privileged_information": True,
     "action_mode": "joint_pos"}
)
cfg.scene.robot = cfg.arm.ik_robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
q = robot.data.default_joint_pos.torch.clone()
robot.write_joint_position_to_sim_index(position=q)
robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
action = torch.zeros(1, u.action_manager.total_action_dim, device=u.device)
action[:, -1] = 1.0  # gripper open
for _ in range(20):
    env.step(action)
joint_err = float((robot.data.joint_pos.torch - q).abs().max())
tcp = (u.scene["ee_frame"].data.target_pos_w.torch[0, 0] - u.scene.env_origins[0]).tolist()
print(f"HOME_TCP {tuple(round(x, 4) for x in tcp)} max_joint_error_rad={joint_err:.5f}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
