import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
PPO_DIR = REPO / "sota-implementations" / "ppo"
OUTCOMES = ("success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out")
ITERATIONS = 12


def test_ppo_pixels_trains_logs_outcomes_evaluates_and_checkpoints(tmp_path):
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(tmp_path),
        "WANDB_SILENT": "true",
    }
    args = [
        "ppo_pixels.py",
        "env.num_envs=8",
        "env.image_size=[64,64]",
        "env.belt.speed=0.3",  # short belt cycle: episodes end within ~1.2 s, so rates appear in 12 iterations
        "env.belt.place_window=0.6",
        "collector.rollout_steps=16",
        f"max_iterations={ITERATIONS}",
        "loss.ppo_epochs=1",
        "loss.mini_batch_size=64",
        f"eval.interval_iterations={ITERATIONS}",
        "eval.video_interval_evals=1",
        "checkpoint.interval_iterations=6",
        "logger.backend=wandb",
        f"hydra.run.dir={tmp_path}",
    ]
    proc = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, timeout=2400, env=env, cwd=PPO_DIR
    )
    tail = proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "PPO_DONE" in proc.stdout, tail

    metrics = [json.loads(line[len("METRICS "):]) for line in proc.stdout.splitlines() if line.startswith("METRICS ")]
    assert len(metrics) == ITERATIONS, tail
    assert all(math.isfinite(v) for m in metrics for v in m.values()), "NaN/inf in logged metrics"
    finished = [m for m in metrics if "train/success_rate" in m]
    assert finished, "no episode finished during training"
    for m in finished:
        assert sum(m[f"train/{t}_rate"] for t in OUTCOMES) >= 1.0 - 1e-6
    assert "episode_reward/reach_food" in metrics[-1]
    assert metrics[-1]["eval/finished_fraction"] == 1.0
    assert "eval/success_rate" in metrics[-1]
    assert metrics[-1]["perf/env_steps_per_s"] > 0

    assert (tmp_path / "checkpoints" / "ppo_pixels_final.pt").exists()
    assert len(list((tmp_path / "checkpoints").glob("ppo_pixels_*.pt"))) >= 3  # iterations 6 and 12, plus final
    assert list(tmp_path.rglob("offline-run-*")), "no offline W&B run written"
    assert list(tmp_path.rglob("*.mp4")), "no eval/policy_view video written"
    stop = [json.loads(line[len("STOP_REASON "):]) for line in proc.stdout.splitlines() if line.startswith("STOP_REASON ")]
    assert stop and stop[-1]["reason"] == "total_frames"


def test_ppo_pixels_early_stop_saves_policy_as_final_checkpoint(tmp_path):
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    args = [
        "ppo_pixels.py",
        "env.num_envs=8",
        "env.image_size=[64,64]",
        "env.belt.speed=0.3",
        "env.belt.place_window=0.6",
        "collector.rollout_steps=16",
        "max_iterations=40",
        "loss.ppo_epochs=1",
        "loss.mini_batch_size=64",
        "eval.interval_iterations=0",
        "checkpoint.interval_iterations=0",
        "logger.backend=null",
        "early_stop.success_rate=0.0",  # every iteration with a finished episode extends the streak
        "early_stop.consecutive_iterations=2",
        f"hydra.run.dir={tmp_path}",
    ]
    proc = subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, timeout=2400, env=env, cwd=PPO_DIR
    )
    assert "PPO_DONE" in proc.stdout, proc.stdout[-4000:] + proc.stderr[-4000:]
    stop = [json.loads(line[len("STOP_REASON "):]) for line in proc.stdout.splitlines() if line.startswith("STOP_REASON ")]
    assert stop[-1]["reason"] == "early_stop"
    assert sum(line.startswith("METRICS ") for line in proc.stdout.splitlines()) < 40
    assert (tmp_path / "checkpoints" / "ppo_pixels_final.pt").exists()
