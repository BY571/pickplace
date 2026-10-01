import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pickplace.datasets import load_shard, shard_manifest
from pickplace.rewards import REWARD_TERMS
from test_teacher_smoke import _train  # a tiny teacher checkpoint, with its short-episode env overrides

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
COLLECT = REPO / "pipeline" / "1_collect_data" / "collect.py"
NUM_ENVS, ROLLOUT, IMAGE = 16, 8, 64


@pytest.fixture(scope="module")
def teacher_checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("teacher")
    proc, tail = _train(tmp, "max_iterations=2", "checkpoint.interval_frames=0", "worker.enabled=false")
    assert "PPO_DONE" in proc.stdout, tail
    return tmp / "run" / "checkpoints" / "ppo_teacher_final.pt"


def _collect(checkpoint, out, frames, **options):
    args = [
        str(COLLECT), f"checkpoint={checkpoint}", f"out={out}", f"frames={frames}",
        f"num_envs={NUM_ENVS}", f"rollout_steps={ROLLOUT}", f"image_size={IMAGE}",
        *[f"{k}={v}" for k, v in options.items()],
    ]
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=2400, env=env, cwd=REPO)
    return proc, proc.stdout[-4000:] + proc.stderr[-4000:]


def test_collect_writes_a_loadable_shard_with_a_manifest(tmp_path, teacher_checkpoint):
    out = tmp_path / "shard"
    proc, tail = _collect(teacher_checkpoint, out, 2048)
    assert "COLLECT_DONE" in proc.stdout, tail

    manifest = shard_manifest(out)
    assert manifest["frames"] == 2048 and manifest["num_envs"] == NUM_ENVS
    assert manifest["successor_stride"] == NUM_ENVS and manifest["image_size"] == IMAGE
    assert manifest["noise_sigma"] == 0.0 and manifest["seed"] == 0
    assert manifest["checkpoint"] == str(teacher_checkpoint) and len(manifest["checkpoint_sha256"]) == 64
    assert manifest["checkpoint_eval"] is None  # the smoke teacher runs with worker.enabled=false
    assert manifest["env"]["cameras"] is True and manifest["env"]["image_size"] == [IMAGE, IMAGE]
    assert manifest["env"]["reward_set"] == "simple_v3b" and manifest["reward_weights"]["success"] > 0
    assert manifest["git_commit"] and manifest["frames_per_hour"] > 0 and manifest["size_bytes"] > 0
    stats = manifest["stats"]
    assert stats["episodes"] >= 1 and 0.0 <= stats["success_rate"] <= 1.0
    assert stats["episode_length"] > 0 and f"terms/{REWARD_TERMS[0]}" in stats
    outcomes = ("success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out")
    assert sum(stats[f"{term}_rate"] for term in outcomes) >= 1.0 - 1e-6

    buffer = load_shard(out, batch_size=32)
    assert len(buffer) == 2048
    rows = buffer[:]
    for camera in ("overview_rgb", "wrist_rgb"):
        images = rows["pixels", camera]
        assert images.dtype == torch.uint8 and images.shape == (2048, IMAGE, IMAGE, 3)
        assert int(images.max()) > 0, f"{camera} is all zeros"
    assert rows["action"].shape[0] == 2048 and rows["loc"].shape == rows["action"].shape
    assert rows["scale"].shape == rows["action"].shape
    assert rows["next", "reward_terms"].shape == (2048, len(REWARD_TERMS))
    assert rows["next", "done"].dtype == torch.bool and rows["next", "terminated"].dtype == torch.bool
    assert rows["next", "truncated"].dtype == torch.bool and rows["next", "outcome", "success"].dtype == torch.bool
    assert rows["step_count"].max() > 0 and rows["collector", "traj_ids"].shape[0] == 2048
    for key in (("proprio", "ee_pos"), ("belt", "bowl_pos"), ("privileged", "food_pos")):
        assert torch.isfinite(rows[key]).all(), key
    assert torch.allclose(rows["action"], torch.tanh(rows["loc"]), atol=1e-4)  # deterministic, no noise


def test_collect_with_noise_perturbs_the_deterministic_action(tmp_path, teacher_checkpoint):
    out = tmp_path / "noisy"
    proc, tail = _collect(teacher_checkpoint, out, 512, noise_sigma=0.2)
    assert "COLLECT_DONE" in proc.stdout, tail
    assert shard_manifest(out)["noise_sigma"] == 0.2

    rows = load_shard(out)[:]
    deviation = (rows["action"] - torch.tanh(rows["loc"])).abs()
    assert deviation.mean() > 0.01, deviation.mean()
    assert rows["action"].abs().max() <= 1.0  # clipped to the action bounds
