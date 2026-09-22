"""Sweep one student checkpoint over degraded observations (no retraining, one Isaac process per algorithm).

The env is built **once** and every condition is evaluated in it, because Isaac Sim startup costs far more
than an evaluation does. A condition is an :class:`pickplace.offline.ObservationPerturbation`: it rewrites
the observation between the env and the policy (see ``evaluate_student``'s ``perturb=``), so the simulation
is untouched and only the policy's input is degraded.

Conditions are **paired across the sweep**: the global RNG is reseeded to the same value before every
condition, so every condition's first (scored) episode starts from the same set of initial states, and the
perturbation draws its own noise from a private generator that never touches the global RNG.

Usage:
    python pipeline/2_1_offline_rl/robustness.py checkpoint=<path.pt> [num_envs=256] [seed=0]
                                                 [conditions=name1,name2] [out=<path.json>]
Prints ``ROBUSTNESS {json}`` per condition (exact ``episodes``/``successes``, so results pool exactly) and
writes them all to ``out`` as they arrive.
"""

import json
import os
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

# --------------------------------------------------------------------------------------------------
# The conditions. Severity 0 (``clean``) is the control of every family; image severities are in [0, 255]
# units, proprio severities in physical units (rad, rad/s, m).
#
# * ``image_noise``  -- additive Gaussian sensor read noise, sigma 2/5/10/20 of 255 (0.8%-7.8% of range).
# * ``brightness``   -- an additive offset of +/-10 and +/-25 of 255, i.e. an exposure or ambient-light
#                       shift of +/-4% and +/-10% of full range. Only +/-25 is run (the decisive end).
# * ``contrast``     -- a multiplicative gain of 0.9 and 0.8: a dirty lens or a stopped-down aperture
#                       compressing the range towards black.
# * ``blur``         -- separable Gaussian defocus, sigma 0.8 px (3x3 kernel) and 1.5 px (5x5) on an 84 px
#                       image, i.e. roughly a 1-px and a 2-px circle of confusion.
# * ``occlusion``    -- one black square per camera covering ~5% and ~15% of the frame, fixed for the whole
#                       evaluation (dirt on the lens / a fixture in the way), on **both** cameras.
# * ``proprio``      -- Gaussian noise on the robot's own sensors, scaled per entry type: joint positions
#                       (and the gripper opening) at 0.005/0.01/0.02 rad, joint velocities at 10x that in
#                       rad/s -- a finite difference of the noisy position over a 100 ms filter window (5
#                       control steps at the 50 Hz control rate) -- and the end-effector position at
#                       2/5/10 mm, the position error those joint-angle errors produce over a ~0.5 m arm.
# * ``combined``     -- the "realistic sensor": mid image noise (sigma 5) + the mid proprio level.
# --------------------------------------------------------------------------------------------------

CONDITIONS = [
    {"name": "clean", "family": "clean", "severity": 0.0, "label": "clean"},

    {"name": "noise_2", "family": "image_noise", "severity": 2.0, "label": "sigma 2", "image_noise": 2.0},
    {"name": "noise_5", "family": "image_noise", "severity": 5.0, "label": "sigma 5", "image_noise": 5.0},
    {"name": "noise_10", "family": "image_noise", "severity": 10.0, "label": "sigma 10", "image_noise": 10.0},
    {"name": "noise_20", "family": "image_noise", "severity": 20.0, "label": "sigma 20", "image_noise": 20.0},

    {"name": "bright_m25", "family": "brightness", "severity": -25.0, "label": "-25", "image_offset": -25.0},
    {"name": "bright_p25", "family": "brightness", "severity": 25.0, "label": "+25", "image_offset": 25.0},

    {"name": "gain_0.9", "family": "contrast", "severity": 0.1, "label": "gain 0.9", "image_gain": 0.9},
    {"name": "gain_0.8", "family": "contrast", "severity": 0.2, "label": "gain 0.8", "image_gain": 0.8},

    {"name": "blur_0.8", "family": "blur", "severity": 0.8, "label": "sigma 0.8 px (3x3)",
     "blur_sigma": 0.8, "blur_kernel": 3},
    {"name": "blur_1.5", "family": "blur", "severity": 1.5, "label": "sigma 1.5 px (5x5)",
     "blur_sigma": 1.5, "blur_kernel": 5},

    {"name": "occl_5", "family": "occlusion", "severity": 0.05, "label": "5% both cameras", "occlusion": 0.05},
    {"name": "occl_15", "family": "occlusion", "severity": 0.15, "label": "15% both cameras", "occlusion": 0.15},

    {"name": "proprio_low", "family": "proprio", "severity": 0.005, "label": "0.005 rad / 0.05 rad/s / 2 mm",
     "joint_pos_sigma": 0.005, "joint_vel_sigma": 0.05, "ee_pos_sigma": 0.002},
    {"name": "proprio_mid", "family": "proprio", "severity": 0.01, "label": "0.01 rad / 0.1 rad/s / 5 mm",
     "joint_pos_sigma": 0.01, "joint_vel_sigma": 0.1, "ee_pos_sigma": 0.005},
    {"name": "proprio_high", "family": "proprio", "severity": 0.02, "label": "0.02 rad / 0.2 rad/s / 10 mm",
     "joint_pos_sigma": 0.02, "joint_vel_sigma": 0.2, "ee_pos_sigma": 0.010},

    {"name": "combined", "family": "combined", "severity": 1.0, "label": "sigma 5 + proprio mid",
     "image_noise": 5.0, "joint_pos_sigma": 0.01, "joint_vel_sigma": 0.1, "ee_pos_sigma": 0.005},
]

META = ("name", "family", "severity", "label")

cli = OmegaConf.from_dotlist(sys.argv[1:])
if not cli.get("checkpoint"):
    raise SystemExit("Pass checkpoint=<path.pt>.")
CHECKPOINT = Path(cli.checkpoint)
num_envs = int(cli.get("num_envs", 256))
seed = int(cli.get("seed", 0))
wanted = [n for n in str(cli.get("conditions", "")).split(",") if n]
conditions = [c for c in CONDITIONS if not wanted or c["name"] in wanted]
if wanted and len(conditions) != len(wanted):
    raise SystemExit(f"Unknown condition(s): {sorted(set(wanted) - {c['name'] for c in conditions})}")

from pickplace.system import memory_used_gb  # noqa: E402

baseline_gb = memory_used_gb()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=True)

import torch  # noqa: E402

from pickplace.artifacts import artifacts_root, read_json, write_json  # noqa: E402
from pickplace.datasets import shard_manifest  # noqa: E402
from pickplace.offline import (  # noqa: E402
    evaluate_student,
    make_actor,
    make_deterministic_actor,
    make_perturbation,
    make_student_env,
)

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
manifest = read_json(CHECKPOINT.with_suffix(".json"))
algorithm = manifest["algorithm"]
obs_keys = [tuple(k) for k in checkpoint["obs_keys"]]
obs_shapes = [tuple(s) for s in checkpoint["image_shapes"]]
action_dim = int(checkpoint["action_dim"])
network_cfg = OmegaConf.create(checkpoint["config"]["network"])
native_hw = next(s[:2] for s in obs_shapes if len(s) == 3)

OUT = Path(cli.get("out") or artifacts_root() / "students" / "_robustness" / f"{algorithm}.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

shard_name = checkpoint["config"]["data"]["shards"][0]
shard_path = Path(shard_name) if Path(shard_name).is_absolute() else artifacts_root() / "shards" / shard_name
shard = shard_manifest(shard_path)

env = make_student_env(shard, num_envs, native_hw[0], seed)
make = make_deterministic_actor if algorithm == "td3_bc" else make_actor
actor = make(obs_shapes, obs_keys, action_dim, network_cfg, env.device)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

results = []
for condition in conditions:
    cfg = {k: v for k, v in condition.items() if k not in META}
    perturb = make_perturbation({"name": condition["name"], **cfg}, obs_keys, obs_shapes, seed=seed)
    torch.manual_seed(seed)  # paired: every condition scores the same set of initial states
    started = time.time()
    result = evaluate_student(actor, env, perturb=None if perturb.is_noop else perturb)
    episodes = int(round(result["episodes"]))
    successes = int(round(result["success_rate"] * episodes))
    row = {
        "checkpoint": str(CHECKPOINT), "algorithm": algorithm,
        "condition": condition["name"], "family": condition["family"],
        "severity": condition["severity"], "label": condition["label"],
        "perturbation": perturb.summary(), "num_envs": num_envs, "seed": seed,
        "episodes": episodes, "successes": successes, "success_rate": result["success_rate"],
        "finished_fraction": result.get("finished_fraction"), "seconds": round(time.time() - started, 1),
    }
    results.append(row)
    print("ROBUSTNESS " + json.dumps(row), flush=True)
    write_json(OUT, {"checkpoint": str(CHECKPOINT), "algorithm": algorithm, "num_envs": num_envs,
                     "seed": seed, "results": results})

print("MEMORY " + json.dumps({"used_gb": round(memory_used_gb(), 2),
                              "delta_gb": round(memory_used_gb() - baseline_gb, 2)}), flush=True)
print(f"ROBUSTNESS_DONE {OUT}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang
