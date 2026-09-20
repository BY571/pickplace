"""Deterministic evaluation of state-teacher checkpoints (cameras off); writes into each checkpoint manifest.

Every env is reset, the deterministic policy runs max_episode_length + 1 steps, and each env's first finished
episode is scored (success rate, return, length, outcome mix, per-term episodic sums).

Usage:
    python pipeline/0_state_teacher/evaluate.py run=<run_dir> [num_envs=256] [force=false] [seed=0]
    python pipeline/0_state_teacher/evaluate.py checkpoint=<run_dir>/checkpoints/ppo_teacher_final.pt
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
cli = OmegaConf.from_dotlist(sys.argv[1:])
if bool(cli.get("run")) == bool(cli.get("checkpoint")):
    raise SystemExit("Pass exactly one of run=<run_dir> or checkpoint=<path.pt>.")
num_envs = int(cli.get("num_envs", 256))
force = bool(cli.get("force", False))
seed = int(cli.get("seed", 0))

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.artifacts import read_json, update_json  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402
from pickplace.training import first_episode_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location("teacher_utils", os.path.join(HERE, "utils.py"))
tu = importlib.util.module_from_spec(_spec)
sys.modules["teacher_utils"] = tu
_spec.loader.exec_module(tu)

KEYS = ["episode_reward", "episode_reward_terms", "step_count",
        ("next", "done"), ("next", "reward"), ("next", "reward_terms"), ("next", "outcome")]


def pending(paths):
    return [p for p in paths if force or read_json(p.with_suffix(".json")).get("eval") is None]


if cli.get("checkpoint"):
    todo = [Path(cli.checkpoint)]
else:
    todo = sorted(Path(cli.run, "checkpoints").glob("ppo_teacher_*.pt"), key=tu.frames_of)
todo = pending(todo)

env, empty = None, []
if todo:
    env_cfg = torch.load(todo[0], map_location="cpu", weights_only=False)["config"]["env"]
    env_cfg = {**env_cfg, "num_envs": num_envs, "cameras": False, "seed": seed}
    torch.manual_seed(seed)
    env = make_env(env_cfg)
    steps = int(env.base_env._env.unwrapped.max_episode_length) + 1

for path in todo:
    t0 = time.monotonic()
    actor = tu.load_teacher_actor(path, env, env.device)
    td = env.reset()
    records = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(steps):
            td = actor(td)
            stepped, td = env.step_and_maybe_reset(td)
            records.append(stepped.select(*KEYS, strict=False).clone())
    metrics = first_episode_metrics(torch.stack(records, dim=1), "eval")
    result = {k[len("eval/"):]: v for k, v in metrics.items()}
    seconds = time.monotonic() - t0
    if not result:
        # No env finished within max_episode_length + 1 steps -- a timeout bug, not a score. Leave `eval`
        # null so the next pass retries this checkpoint; writing {} would read as "evaluated" forever.
        empty.append(path.name)
        print(
            "EVAL_EMPTY " + json.dumps({"checkpoint": path.name, "steps": steps, "num_envs": num_envs,
                                        "seconds": round(seconds, 1), "reason": "no episode finished"}),
            flush=True,
        )
        continue
    update_json(path.with_suffix(".json"), eval=result, eval_num_envs=num_envs, eval_seconds=round(seconds, 1))
    print("EVAL " + json.dumps({"checkpoint": path.name, "seconds": round(seconds, 1), **result}), flush=True)

print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2), "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print(f"EVALUATE_DONE evaluated={len(todo) - len(empty)} empty={len(empty)}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
