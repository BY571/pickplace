"""Evaluate one offline-RL student checkpoint online (deterministic policy, fresh camera envs).

Same protocol ``pipeline/0_state_teacher/evaluate.py`` uses for the teacher: ``max_episode_length + 1``
steps, each env's first finished episode scored. The checkpoint's own network, observation keys and image
shapes are used to rebuild the actor (as ``pipeline/2_1_offline_rl/render.py`` does), and its training
shard's own env config builds the camera env (as ``pickplace.offline.make_student_env`` does for the
training-time online evaluation), so results are directly comparable to the numbers already in the
checkpoint manifest.

This repo's convention (see ``benchmark.py``) is one Isaac process per measurement rather than several
``make_env`` calls in one process, so a sample bigger than one run's ``num_envs`` is built by running this
several times with different seeds and pooling the exact counts this script prints (not just the rate, so
the pooled total is exact) -- e.g. 5 runs of ``num_envs=200`` -> 1000 pooled episodes.

Usage:
    python pipeline/2_1_offline_rl/evaluate.py checkpoint=<path.pt> [num_envs=200] [seed=0]
Prints ``EVAL {json}`` with ``episodes`` (finished) and ``successes`` (exact counts).
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
num_envs = int(cli.get("num_envs", 200))
seed = int(cli.get("seed", 0))

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import torch  # noqa: E402

from pickplace.artifacts import artifacts_root, read_json  # noqa: E402
from pickplace.datasets import shard_manifest  # noqa: E402
from pickplace.offline import evaluate_student, make_actor, make_deterministic_actor, make_student_env  # noqa: E402

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
manifest = read_json(CHECKPOINT.with_suffix(".json"))
algorithm = manifest["algorithm"]
obs_keys = [tuple(k) for k in checkpoint["obs_keys"]]
obs_shapes = [tuple(s) for s in checkpoint["image_shapes"]]
action_dim = int(checkpoint["action_dim"])
network_cfg = OmegaConf.create(checkpoint["config"]["network"])
native_hw = next(s[:2] for s in obs_shapes if len(s) == 3)  # the checkpoint's own trained image resolution

shard_name = checkpoint["config"]["data"]["shards"][0]
shard_path = Path(shard_name) if Path(shard_name).is_absolute() else artifacts_root() / "shards" / shard_name
shard = shard_manifest(shard_path)

env = make_student_env(shard, num_envs, native_hw[0], seed)
make = make_deterministic_actor if algorithm == "td3_bc" else make_actor
actor = make(obs_shapes, obs_keys, action_dim, network_cfg, env.device)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

result = evaluate_student(actor, env)
episodes = int(round(result["episodes"]))
successes = int(round(result["success_rate"] * episodes))
out = {
    "checkpoint": str(CHECKPOINT), "algorithm": algorithm, "gradient_step": manifest["gradient_steps"],
    "num_envs": num_envs, "seed": seed, "episodes": episodes, "successes": successes,
    "success_rate": result["success_rate"], "finished_fraction": result.get("finished_fraction"),
}
print("EVAL " + json.dumps(out), flush=True)
print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2), "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print("EVALUATE_DONE", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
