"""Render clean, high-resolution stills of the cell from the wide scene camera — no overlays, no panels.

For figures: ``scripts/render_episode.py`` and the pipeline's render scripts compose the scene view with the
policy's camera panels and a caption, which is right for a debugging video and wrong for a document.

    python scripts/render_scene_stills.py checkpoint=<teacher.pt> out=<dir>
        [seconds=8] [width=1920] [height=1080] [every=0.4]
        [eye=[1.65,-1.35,1.25]] [target=[0.35,0.05,0.25]]

``eye``/``target`` move the camera (defaults frame the whole cell); rendering continues across episode
boundaries, so one call gives a sequence to pick a frame from.
"""
import os, sys
from pathlib import Path
from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
CKPT = Path(cli.checkpoint)
OUT = Path(cli.get("out", "/workspace/artifacts/stills")); OUT.mkdir(parents=True, exist_ok=True)
SECONDS = float(cli.get("seconds", 6.0))
W, H = int(cli.get("width", 1920)), int(cli.get("height", 1080))
EVERY = float(cli.get("every", 0.5))

from pickplace.app import launch_app
app = launch_app(headless=True, enable_cameras=True)

import imageio.v2 as imageio
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from pickplace.torchrl_env import make_env

import importlib.util
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "0_state_teacher")
spec = importlib.util.spec_from_file_location("teacher_utils", os.path.join(HERE, "utils.py"))
tu = importlib.util.module_from_spec(spec); sys.modules["teacher_utils"] = tu; spec.loader.exec_module(tu)

env_cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["config"]["env"]
eye = [float(x) for x in cli.get("eye", [1.65, -1.35, 1.25])]
target = [float(x) for x in cli.get("target", [0.35, 0.05, 0.25])]
STUDENT = bool(cli.get("student", False))   # also dump the policy's own 84 px camera views
IMAGE = int(cli.get("image", 84))
env_cfg = {**env_cfg, "num_envs": 1, "cameras": STUDENT, "image_size": [IMAGE, IMAGE], "render_camera": True,
           "render_image_size": [H, W], "render_cam_eye": eye, "render_cam_target": target}
env = make_env(env_cfg)
u = env.base_env._env.unwrapped
dt = u.step_dt
actor = tu.load_teacher_actor(CKPT, env, env.device)

td = env.reset()
saved = []
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for i in range(int(SECONDS / dt)):
        td = actor(td)
        stepped, td = env.step_and_maybe_reset(td)
        if i % max(1, int(EVERY / dt)) == 0:
            frame = u.scene["render_cam"].data.output["rgb"][0][..., :3].float().clamp(0, 255).to(torch.uint8).cpu().numpy()
            p = OUT / f"scene_{i * dt:05.2f}s.png"
            imageio.imwrite(p, frame)
            saved.append(str(p))
            print("STILL", p, flush=True)
            if STUDENT:
                for name in ("overview_rgb", "wrist_rgb"):
                    view = td["pixels", name][0][..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    imageio.imwrite(OUT / f"{name}_{i * dt:05.2f}s.png", view)
        if bool(stepped["next", "done"].any()):
            print("EPISODE_END at", round(i * dt, 2), "s", flush=True)
print("STILLS_DONE", len(saved), flush=True)
