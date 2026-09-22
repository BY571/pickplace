"""Render an offline-RL student checkpoint (BC / IQL / TD3+BC) to MP4 (1 env, cameras on, deterministic policy).

Same layout as ``pipeline/0_state_teacher/render.py``: a wide scene camera on top, the two camera views the
student actually consumes below (``overview_rgb``, ``wrist_rgb``, upscaled for display). Unlike the teacher,
the student's own policy reads image + proprio, so the network (``pickplace.offline.make_actor`` /
``make_deterministic_actor``), the observation keys and the image shapes all come from the checkpoint itself
-- exactly what ``pipeline/2_1_offline_rl/runner.py`` built and trained -- not from a shared teacher module.

The camera env is built from the checkpoint's own training shard's manifest (``config.data.shards[0]``'s
``"env"`` block) -- the same source ``pickplace.offline.make_student_env`` uses for online evaluation -- so
action mode, reward set, belt geometry etc. all match training. The env's *actual* camera resolution is
pinned to the checkpoint's own trained image shape (typically 84x84): the student's CNN was built for that
exact input, so unlike the teacher's ``image=`` (which truly sets capture resolution, because the teacher
never consumes the pixels obs), here ``image=`` only sets the display panel's upscale target and label --
the network always gets its native training resolution.

Usage:
    python pipeline/2_1_offline_rl/render.py checkpoint=<run_dir>/checkpoints/<algo>_final.pt
    options: [seconds=12] [until_done=true] [image=128] [out=<path.mp4>] [seed=0]
             [outcome=any|success|failure] [attempts=1]

``outcome=failure attempts=10`` resets the same env up to 10 times looking for a non-success ending episode
(cheap -- a reset re-randomises the scene without relaunching Isaac Sim) and renders whichever episode
matches, or the last attempt with ``matched=false`` if none did.
"""

import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

cli = OmegaConf.from_dotlist(sys.argv[1:])
if not cli.get("checkpoint"):
    raise SystemExit("Pass checkpoint=<path.pt>.")
CHECKPOINT = Path(cli.checkpoint)
seconds = float(cli.get("seconds", 12.0))
until_done = bool(cli.get("until_done", True))
image = int(cli.get("image", 128))
seed = int(cli.get("seed", 0))
outcome_filter = str(cli.get("outcome", "any"))
attempts = int(cli.get("attempts", 1))
if outcome_filter not in ("any", "success", "failure"):
    raise SystemExit("outcome must be one of: any, success, failure")
OUT = Path(cli.get("out") or CHECKPOINT.with_name(f"{CHECKPOINT.stem}_render.mp4"))

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.artifacts import artifacts_root, read_json  # noqa: E402
from pickplace.datasets import shard_manifest  # noqa: E402
from pickplace.metrics import OUTCOME_TERMS  # noqa: E402
from pickplace.offline import make_actor, make_deterministic_actor  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

SCENE_HW = (720, 1280)
PANEL = 640


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return img[..., -3:].float().clamp(0, 255).to(torch.uint8).cpu().numpy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def compose(scene, overview, wrist, caption: str, native_hw: tuple[int, int]) -> np.ndarray:
    def panel(img, name):
        big = cv2.resize(to_uint8(img), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        return label(big, f"student camera: {name} ({native_hw[0]}x{native_hw[1]} -> {image}x{image} shown)")

    top = label(to_uint8(scene), f"scene camera — {caption}")
    return np.concatenate([top, np.concatenate([panel(overview, "overview_rgb"), panel(wrist, "wrist_rgb")], axis=1)])


def resolve_shard_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else artifacts_root() / "shards" / name


# --- load the checkpoint and its sidecar manifest (algorithm, gradient step, evaluated success rate) ------
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
manifest = read_json(CHECKPOINT.with_suffix(".json"))
algorithm = manifest["algorithm"]
gradient_step = manifest["gradient_steps"]
success_rate = (manifest.get("eval") or {}).get("success_rate")

obs_keys = [tuple(k) for k in checkpoint["obs_keys"]]
obs_shapes = [tuple(s) for s in checkpoint["image_shapes"]]
action_dim = int(checkpoint["action_dim"])
network_cfg = OmegaConf.create(checkpoint["config"]["network"])
native_hw = next(s[:2] for s in obs_shapes if len(s) == 3)

shard_name = checkpoint["config"]["data"]["shards"][0]
env_cfg = {
    **shard_manifest(resolve_shard_path(shard_name))["env"],
    "num_envs": 1,
    "cameras": True,
    "render_camera": True,
    "render_image_size": list(SCENE_HW),
    "image_size": list(native_hw),
    "frame_stack": 1,
    "seed": seed,
}
env = make_env(env_cfg)
u = env.base_env._env.unwrapped
dt = u.step_dt

make = make_deterministic_actor if algorithm == "td3_bc" else make_actor
actor = make(obs_shapes, obs_keys, action_dim, network_cfg, env.device)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

eval_str = f"{success_rate:.3f}" if success_rate is not None else "n/a"
caption = f"{algorithm} · step {gradient_step:,} · eval {eval_str}"
max_steps = int(seconds / dt)
still = OUT.with_name(f"{OUT.stem}_frame.png")

matched, frames, outcome, n = False, [], "none", 0
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for attempt in range(1, attempts + 1):
        td = env.reset()
        frames, outcome, n = [], "none", 0
        still_frame = None
        for _ in range(max_steps):
            td = actor(td)
            stepped, td = env.step_and_maybe_reset(td)
            frame = compose(u.scene["render_cam"].data.output["rgb"][0],
                            td["pixels", "overview_rgb"][0], td["pixels", "wrist_rgb"][0], caption, native_hw)
            frames.append(frame)
            if n == int(3.0 / dt):
                still_frame = frame
            n += 1
            if bool(stepped["next", "done"].any()):
                outcome = ",".join(t for t in OUTCOME_TERMS if bool(stepped["next", "outcome", t].any())) or "unknown"
                if until_done:
                    break
        is_success = outcome == "success"
        if (outcome_filter == "any"
                or (outcome_filter == "success" and is_success)
                or (outcome_filter == "failure" and outcome not in ("none", "unknown") and not is_success)):
            matched = True
            break
        print(f"RENDER_ATTEMPT " + json.dumps({"checkpoint": CHECKPOINT.name, "attempt": attempt,
                                                "outcome": outcome, "wanted": outcome_filter}), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
with imageio.get_writer(OUT, fps=round(1.0 / dt), codec="libx264", quality=8, macro_block_size=1) as writer:
    for frame in frames:
        writer.append_data(frame)
imageio.imwrite(still, still_frame if still_frame is not None else frames[-1])

print("RENDER " + json.dumps({
    "checkpoint": CHECKPOINT.name, "algorithm": algorithm, "gradient_step": gradient_step,
    "success_rate": success_rate, "video": str(OUT), "still": str(still), "frames": n,
    "outcome": outcome, "wanted": outcome_filter, "attempts_used": attempt, "matched": matched,
}), flush=True)
print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2), "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print("RENDER_DONE", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
