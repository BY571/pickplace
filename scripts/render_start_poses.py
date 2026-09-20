"""Render a documentation figure of the randomized robot start poses.

Two modes:
  mode=grid (default): one env per tile, reset once, settle a few steps with zero-motion actions, then
                        capture each env's scene camera and tile them into one PNG.
  mode=overlay: a single env, reset+settle N times (default 8), capturing the same scene camera each time;
                the frames are combined into one "ghosting" image (pixel-wise median background, each pose
                alpha-blended on top) so several arm poses are visible at once in one table scene.

Usage:
    python scripts/render_start_poses.py [out=outputs/render/start_poses.png] [tiles=12] [cols=4]
                                         [env.robot_reset.position_range=[-0.25,0.25]]
                                         [env.robot_reset.velocity_range=[-0.1,0.1]]
    python scripts/render_start_poses.py mode=overlay [out=outputs/render/start_poses_overlay.png] [poses=8] \\
                                         [env.robot_reset.position_range=[-0.25,0.25]] \\
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
mode = str(cli.get("mode", "grid"))
out_path = str(cli.get("out", "outputs/render/start_poses.png"))
tiles = int(cli.get("tiles", 12))
cols = int(cli.get("cols", 4))
settle_steps = int(cli.get("settle_steps", 5))
poses = int(cli.get("poses", 8))
diff_threshold = int(cli.get("diff_threshold", 12))
alpha = float(cli.get("alpha", 0.55))

env_overrides = OmegaConf.to_container(cli.env, resolve=True) if "env" in cli else {}
robot_reset = {"position_range": [-0.25, 0.25], "velocity_range": [-0.1, 0.1]}
robot_reset.update(env_overrides.pop("robot_reset", {}))

TILE_HW = (270, 480)
OVERLAY_HW = (480, 854)
RENDER_HW = OVERLAY_HW if mode == "overlay" else TILE_HW
num_envs = 1 if mode == "overlay" else tiles
env_cfg = {
    "num_envs": num_envs,
    "cameras": True,
    "render_camera": True,
    "render_image_size": list(RENDER_HW),
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

if mode == "overlay":
    frames = []
    for _ in range(poses):
        td = env.reset()
        for _ in range(settle_steps):
            td.set("action", torch.zeros(env.action_spec.shape, device=env.device))
            td = step_mdp(env.step(td))
        u = env.base_env._env.unwrapped
        frames.append(to_uint8(u.scene["render_cam"].data.output["rgb"][0]))

    stack = np.stack(frames).astype(np.float32)  # (poses, H, W, 3)
    median = np.median(stack, axis=0)  # (H, W, 3) static background (table/belt/bowls)

    canvas = median.copy()
    kernel = np.ones((5, 5), np.uint8)
    for i, frame in enumerate(stack):
        diff = np.abs(frame - median).max(axis=-1)  # (H, W)
        mask = (diff > diff_threshold).astype(np.uint8) * 255
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        a = 1.0 if i == len(stack) - 1 else alpha  # draw the last pose crisp
        canvas = canvas * (1 - mask_f * a) + frame * (mask_f * a)

    grid = canvas.clip(0, 255).astype(np.uint8)
    caption = (
        f"robot_reset position_range={robot_reset['position_range']} "
        f"velocity_range={robot_reset['velocity_range']} poses={poses}"
    )
else:
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

font_scale, thickness = 0.8, 2
(text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
while text_w > grid.shape[1] - 20 and font_scale > 0.3:
    font_scale -= 0.05
    (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
strip = np.zeros((text_h + 24, grid.shape[1], 3), dtype=np.uint8)
cv2.putText(strip, caption, (10, text_h + 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
grid = np.concatenate([strip, grid], axis=0)

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
imageio.imwrite(out_path, grid)
print(f"WROTE {out_path} shape={grid.shape}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
