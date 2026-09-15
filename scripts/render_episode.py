"""Render the food cell to an MP4 (headless, e.g. on the DGX Spark).

Top: a wide third-person camera showing the whole cell (for humans only, not an observation).
Bottom: exactly what the policy sees, i.e. the ``pixels`` observations (``overview_rgb`` and ``wrist_rgb``)
at the env's ``image_size`` (default 128x128), upscaled with nearest-neighbour so the pixels stay visible.
The policy additionally receives the non-image groups (``proprio``, ``belt``, optionally ``privileged``).

A simple scripted motion, not a policy, drives the arm: hold for 1 s, hover above the food in the
ingredient bowl, then follow the bowl riding the belt. The gripper stays open; nothing is picked.

Usage:
    python scripts/render_episode.py [out=outputs/render/episode.mp4] [seconds=8]
                                     [env.image_size=[128,128]] [env.belt.speed=0.08 ...]
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

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.config import build_cell_env_cfg  # noqa: E402

SCENE_HW = (720, 1280)
PANEL = 640  # each policy view is shown as a PANEL x PANEL square; two side by side = scene width
HOVER_HEIGHT = 0.18  # TCP height above the target [m]
GAIN = 4.0  # position error [m] -> action; actions are clamped to [-1, 1] and scaled by the arm's IK scale

cfg = build_cell_env_cfg(
    {
        "num_envs": 1,
        **env_overrides,
        "cameras": True,
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
policy_h, policy_w = cfg.image_size


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return image[..., :3].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = np.ascontiguousarray(image)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def policy_panel(image: torch.Tensor, name: str) -> np.ndarray:
    small = to_uint8(image)
    big = cv2.resize(small, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    return label(big, f"policy input: {name} ({policy_h}x{policy_w})")


def frame(o) -> np.ndarray:
    scene = label(to_uint8(unwrapped.scene["render_cam"].data.output["rgb"][0]), "scene camera (not an observation)")
    overview = policy_panel(o["pixels"]["overview_rgb"][0], "overview_rgb")
    wrist = policy_panel(o["pixels"]["wrist_rgb"][0], "wrist_rgb")
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
