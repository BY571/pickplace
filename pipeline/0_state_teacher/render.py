"""Render state-teacher checkpoints to MP4 (1 env, cameras on, deterministic policy).

Top: a wide scene camera (not an observation). Bottom: the two camera views a camera-based student will get
(``overview_rgb``, ``wrist_rgb`` at ``image`` px, upscaled). The teacher itself acts from state only.

Usage:
    python pipeline/0_state_teacher/render.py checkpoint=<run_dir>/checkpoints/ppo_teacher_final.pt
    python pipeline/0_state_teacher/render.py all=<run_dir>          # every checkpoint without a video
    options: [seconds=12] [until_done=true] [image=128] [force=false]
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
cli = OmegaConf.from_dotlist(sys.argv[1:])
if bool(cli.get("all")) == bool(cli.get("checkpoint")):
    raise SystemExit("Pass exactly one of checkpoint=<path.pt> or all=<run_dir>.")
seconds = float(cli.get("seconds", 12.0))
until_done = bool(cli.get("until_done", True))
image = int(cli.get("image", 128))
force = bool(cli.get("force", False))

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.artifacts import read_json, update_json  # noqa: E402
from pickplace.metrics import OUTCOME_TERMS  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

# Loaded by file path: with cameras enabled, Isaac Sim's bundled cv2/utils shadows `import utils`.
_spec = importlib.util.spec_from_file_location("teacher_utils", os.path.join(HERE, "utils.py"))
tu = importlib.util.module_from_spec(_spec)
sys.modules["teacher_utils"] = tu
_spec.loader.exec_module(tu)

SCENE_HW = (720, 1280)
PANEL = 640


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return img[..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def compose(scene, overview, wrist, caption: str) -> np.ndarray:
    def panel(img, name):
        big = cv2.resize(to_uint8(img), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        return label(big, f"student camera: {name} ({image}x{image})")

    top = label(to_uint8(scene), f"scene camera — {caption}")
    return np.concatenate([top, np.concatenate([panel(overview, "overview_rgb"), panel(wrist, "wrist_rgb")], axis=1)])


if cli.get("checkpoint"):
    todo = [Path(cli.checkpoint)]
else:
    todo = sorted(Path(cli.all, "checkpoints").glob("ppo_teacher_*.pt"), key=tu.frames_of)
todo = [p for p in todo if force or read_json(p.with_suffix(".json")).get("video") is None]

env = None
if todo:
    env_cfg = torch.load(todo[0], map_location="cpu", weights_only=False)["config"]["env"]
    env_cfg = {**env_cfg, "num_envs": 1, "cameras": True, "render_camera": True,
               "render_image_size": list(SCENE_HW), "image_size": [image, image]}
    env = make_env(env_cfg)
    u = env.base_env._env.unwrapped
    dt = u.step_dt

for path in todo:
    actor = tu.load_teacher_actor(path, env, env.device)
    meta = read_json(path.with_suffix(".json"))
    caption = f"{path.stem} · {meta['frames'] / 1e6:.1f} M frames"
    out = path.with_suffix(".mp4")
    still = path.with_name(f"{path.stem}_frame.png")
    outcome, n = "none", 0
    td = env.reset()
    with imageio.get_writer(out, fps=round(1.0 / dt), codec="libx264", quality=8, macro_block_size=1) as writer, \
            torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(int(seconds / dt)):
            td = actor(td)
            stepped, td = env.step_and_maybe_reset(td)
            frame = compose(u.scene["render_cam"].data.output["rgb"][0],
                            td["pixels", "overview_rgb"][0], td["pixels", "wrist_rgb"][0], caption)
            writer.append_data(frame)
            if n == int(3.0 / dt):
                imageio.imwrite(still, frame)
            n += 1
            if bool(stepped["next", "done"].any()):
                outcome = ",".join(t for t in OUTCOME_TERMS if bool(stepped["next", "outcome", t].any())) or "unknown"
                if until_done:
                    break
    update_json(path.with_suffix(".json"), video=out.name, video_outcome=outcome)
    print("RENDER " + json.dumps({"checkpoint": path.name, "video": str(out), "frames": n, "outcome": outcome}), flush=True)

print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2), "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print(f"RENDER_DONE rendered={len(todo)}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
