"""Render a documentation figure of the randomized robot start poses: one env per tile, reset once, settle a
few steps with zero-motion actions, then capture each env's scene camera and tile them into one PNG.

Usage:
    python scripts/render_start_poses.py [out=outputs/render/start_poses.png] [tiles=12] [cols=4]
                                         [env.robot_reset.position_range=[-0.25,0.25]]
                                         [env.robot_reset.velocity_range=[-0.1,0.1]]

Defaults to the v2r fine-tune's wide randomization (+-0.25 rad / +-0.1 rad/s); pass the training default
(+-0.02 rad / 0.0 rad/s) for the comparison figure, e.g.:
    python scripts/render_start_poses.py out=outputs/render/start_poses_default.png \\
        env.robot_reset.position_range=[-0.02,0.02] env.robot_reset.velocity_range=[0.0,0.0]
"""

import math
import os
import sys

from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
out_path = str(cli.get("out", "outputs/render/start_poses.png"))
tiles = int(cli.get("tiles", 12))
cols = int(cli.get("cols", 4))
settle_steps = int(cli.get("settle_steps", 5))

env_overrides = OmegaConf.to_container(cli.env, resolve=True) if "env" in cli else {}
robot_reset = {"position_range": [-0.25, 0.25], "velocity_range": [-0.1, 0.1]}
robot_reset.update(env_overrides.pop("robot_reset", {}))

TILE_HW = (270, 480)
env_cfg = {
    "num_envs": tiles,
    "cameras": True,
    "render_camera": True,
    "render_image_size": list(TILE_HW),
    "robot_reset": robot_reset,
    **env_overrides,
}

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs.utils import step_mdp  # noqa: E402

from food_robot.torchrl_env import make_env  # noqa: E402


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return img[..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


env = make_env(env_cfg)
td = env.reset()
for _ in range(settle_steps):
    td.set("action", torch.zeros(env.action_spec.shape, device=env.device))
    td = step_mdp(env.step(td))

u = env.base_env._env.unwrapped
frames = u.scene["render_cam"].data.output["rgb"]  # (tiles, H, W, C)

rows = math.ceil(tiles / cols)
pad = np.zeros((TILE_HW[0], TILE_HW[1], 3), dtype=np.uint8)
grid_rows = []
for r in range(rows):
    row_imgs = []
    for c in range(cols):
        i = r * cols + c
        img = label(to_uint8(frames[i]), f"env {i}") if i < tiles else pad
        row_imgs.append(img)
    grid_rows.append(np.concatenate(row_imgs, axis=1))
grid = np.concatenate(grid_rows, axis=0)

caption = f"robot_reset position_range={robot_reset['position_range']} velocity_range={robot_reset['velocity_range']} tiles={tiles}"
strip = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
cv2.putText(strip, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
grid = np.concatenate([strip, grid], axis=0)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
imageio.imwrite(out_path, grid)
print(f"WROTE {out_path} shape={grid.shape}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
