"""Render the food cell to an MP4 (headless, e.g. on the DGX Spark).

The video shows the whole cell from a wide third-person camera (top) and the policy cameras below it
(overview left, wrist right). A simple scripted motion, not a policy, drives the arm: hold for 1 s, hover
above the food in the ingredient bowl, then follow the bowl riding the belt. The gripper stays open;
nothing is picked. It is meant for looking at the scene and the cameras, not at behaviour.

Usage:
    python scripts/render_episode.py [out=outputs/render/episode.mp4] [seconds=8]
                                     [env.num_envs=1] [env.belt.speed=0.08 ...]
"""

import os
import sys

from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
out_path = str(cli.get("out", "outputs/render/episode.mp4"))
seconds = float(cli.get("seconds", 8.0))
env_overrides = OmegaConf.to_container(cli.env, resolve=True) if "env" in cli else {}

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.config import build_cell_env_cfg  # noqa: E402

SCENE_HW = (720, 1280)
POLICY_HW = (360, 640)  # two policy views side by side = scene width
HOVER_HEIGHT = 0.18  # TCP height above the target [m]
GAIN = 4.0  # position error [m] -> action; actions are clamped to [-1, 1] and scaled by the arm's IK scale

cfg = build_cell_env_cfg(
    {
        "num_envs": 1,
        **env_overrides,
        "cameras": True,
        "image_size": list(POLICY_HW),
        "privileged_information": True,
        "render_camera": True,
        "render_image_size": list(SCENE_HW),
    }
)
if cfg.action_mode != "ee_delta_pose":
    raise ValueError("The scripted motion needs env.action_mode=ee_delta_pose.")
env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
unwrapped = env.unwrapped
obs, _ = env.reset()

dt = unwrapped.step_dt
steps = int(seconds / dt)
action_dim = unwrapped.action_manager.total_action_dim
fps = round(1.0 / dt)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return image[..., :3].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def frame(o) -> np.ndarray:
    scene = to_uint8(unwrapped.scene["render_cam"].data.output["rgb"][0])
    overview = to_uint8(o["pixels"]["overview_rgb"][0])
    wrist = to_uint8(o["pixels"]["wrist_rgb"][0])
    return np.concatenate([scene, np.concatenate([overview, wrist], axis=1)], axis=0)


def scripted_action(o, t: float) -> torch.Tensor:
    action = torch.zeros(cfg.scene.num_envs, action_dim, device=unwrapped.device)
    action[:, -1] = 1.0  # gripper open
    if t < 1.0:
        return action
    ee = o["proprio"]["ee_pos"]
    target = (o["privileged"]["food_pos"] if t < 3.5 else o["belt"]["bowl_pos"]).clone()
    target[:, 2] += HOVER_HEIGHT
    action[:, :3] = (GAIN * (target - ee)).clamp(-1.0, 1.0)
    return action


os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
still_path = os.path.splitext(out_path)[0] + "_frame.png"

with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
    for step in range(steps):
        obs, *_ = env.step(scripted_action(obs, step * dt))
        image = frame(obs)
        writer.append_data(image)
        if step == int(3.0 / dt):  # hand above the food: a still that shows every camera doing its job
            imageio.imwrite(still_path, image)

print(f"RENDER_DONE video={out_path} still={still_path} frames={steps} fps={fps}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
