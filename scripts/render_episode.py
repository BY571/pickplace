"""Render the food cell to an MP4 (headless, e.g. on the DGX Spark).

Top: a wide third-person camera showing the whole cell (for humans only, not an observation).
Bottom: exactly what the policy sees, i.e. the ``pixels`` observations (``overview_rgb`` and ``wrist_rgb``,
newest frame of each stack) at the env's ``image_size``, upscaled with nearest-neighbour so the pixels stay
visible. The policy additionally receives the non-image groups (``proprio``, optionally ``belt``/``privileged``).

Without ``policy=``, a scripted motion (not a policy) drives the arm: hold for 1 s, hover above the food in
the ingredient bowl, then follow the bowl riding the belt. With ``policy=<checkpoint>`` (from
sota-implementations/ppo/ppo_pixels.py), the trained actor drives it deterministically, with the env built
from the checkpoint's own config.

Usage:
    python scripts/render_episode.py [out=outputs/render/episode.mp4] [seconds=8]
                                     [env.image_size=[128,128]] [env.belt.speed=0.08 ...]
    python scripts/render_episode.py policy=outputs/<date>/<time>/checkpoints/ppo_pixels_final.pt
                                     [out=outputs/render/policy.mp4] [seconds=12] [until_done=true]
"""

import importlib.util
import os
import sys

from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
policy_path = cli.get("policy", None)
out_path = str(cli.get("out", "outputs/render/policy.mp4" if policy_path else "outputs/render/episode.mp4"))
seconds = float(cli.get("seconds", 12.0 if policy_path else 8.0))
until_done = bool(cli.get("until_done", True))
env_overrides = OmegaConf.to_container(cli.env, resolve=True) if "env" in cli else {}

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCENE_HW = (720, 1280)
PANEL = 640  # each policy view is shown as a PANEL x PANEL square; two side by side = scene width
HOVER_HEIGHT = 0.18  # TCP height above the target [m]
GAIN = 4.0  # position error [m] -> action; actions are clamped to [-1, 1] and scaled by the arm's IK scale
RENDER_KEYS = {"num_envs": 1, "cameras": True, "render_camera": True, "render_image_size": list(SCENE_HW)}


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return image[..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()  # newest frame of the stack


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = np.ascontiguousarray(image)
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def compose(scene: torch.Tensor, overview: torch.Tensor, wrist: torch.Tensor, image_size) -> np.ndarray:
    h, w = image_size

    def panel(image, name):
        big = cv2.resize(to_uint8(image), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        return label(big, f"policy input: {name} ({h}x{w})")

    top = label(to_uint8(scene), "scene camera (not an observation)")
    return np.concatenate([top, np.concatenate([panel(overview, "overview_rgb"), panel(wrist, "wrist_rgb")], axis=1)])


def write_video(frames_iter, fps: int, still_step: int) -> tuple[str, int]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    still_path = os.path.splitext(out_path)[0] + "_frame.png"
    n = 0
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for image in frames_iter:
            writer.append_data(image)
            if n == still_step:
                imageio.imwrite(still_path, image)
            n += 1
    return still_path, n


def run_scripted():
    import gymnasium as gym

    import food_robot.envs  # noqa: F401
    from food_robot.config import build_cell_env_cfg

    cfg = build_cell_env_cfg({**env_overrides, **RENDER_KEYS, "privileged_information": True})
    if cfg.action_mode != "ee_delta_pose":
        raise ValueError("The scripted motion needs env.action_mode=ee_delta_pose.")
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    u = env.unwrapped
    dt, action_dim = u.step_dt, u.action_manager.total_action_dim

    def scripted_action(o, t):
        action = torch.zeros(1, action_dim, device=u.device)
        action[:, -1] = 1.0  # gripper open
        if t < 1.0:
            return action
        target = (o["privileged"]["food_pos"] if t < 3.5 else o["belt"]["bowl_pos"]).clone()
        target[:, 2] += HOVER_HEIGHT
        action[:, :3] = (GAIN * (target - o["proprio"]["ee_pos"])).clamp(-1.0, 1.0)
        return action

    def frames():
        obs, _ = env.reset()
        for step in range(int(seconds / dt)):
            obs, *_ = env.step(scripted_action(obs, step * dt))
            scene = u.scene["render_cam"].data.output["rgb"][0]
            yield compose(scene, obs["pixels"]["overview_rgb"][0], obs["pixels"]["wrist_rgb"][0], cfg.image_size)

    still, n = write_video(frames(), round(1.0 / dt), still_step=int(3.0 / dt))
    return still, n, round(1.0 / dt), "none"


def run_policy():
    from torchrl.envs import ExplorationType, set_exploration_type

    from food_robot.metrics import OUTCOME_TERMS
    from food_robot.torchrl_env import make_env

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sota-implementations", "ppo", "utils_pixels.py")
    spec = importlib.util.spec_from_file_location("utils_pixels", path)
    utils_pixels = importlib.util.module_from_spec(spec)
    sys.modules["utils_pixels"] = utils_pixels
    spec.loader.exec_module(utils_pixels)

    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    env_cfg = {**checkpoint["config"]["env"], **env_overrides, **RENDER_KEYS}
    env = make_env(env_cfg)
    u = env.base_env._env.unwrapped
    actor = utils_pixels.load_actor(policy_path, env, env.device)
    dt = u.step_dt
    outcome = {"term": "none"}

    def frames():
        td = env.reset()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for _ in range(int(seconds / dt)):
                td = actor(td)
                stepped, td = env.step_and_maybe_reset(td)
                scene = u.scene["render_cam"].data.output["rgb"][0]
                yield compose(scene, td["pixels", "overview_rgb"][0], td["pixels", "wrist_rgb"][0], env_cfg["image_size"])
                if bool(stepped["next", "done"].any()):
                    fired = [t for t in OUTCOME_TERMS if bool(stepped["next", "outcome", t].any())]
                    outcome["term"] = ",".join(fired) or "unknown"
                    if until_done:
                        return

    still, n = write_video(frames(), round(1.0 / dt), still_step=int(3.0 / dt))
    return still, n, round(1.0 / dt), outcome["term"]


still_path, n_frames, fps, outcome_term = run_policy() if policy_path else run_scripted()
print(
    f"RENDER_DONE video={out_path} still={still_path} frames={n_frames} fps={fps} outcome={outcome_term}", flush=True
)
os._exit(0)  # Isaac Sim shutdown can hang
