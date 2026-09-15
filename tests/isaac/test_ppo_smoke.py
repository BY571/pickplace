import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
PPO_DIR = REPO / "sota-implementations" / "ppo"


def _run(args, tmp_path, timeout=1800):
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    return subprocess.run(
        [sys.executable, *args, f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=PPO_DIR,
    )


def test_ppo_runs_logs_and_checkpoints(tmp_path):
    proc = _run(
        ["ppo.py", "env.num_envs=16", "collector.rollout_steps=8", "max_iterations=3",
         "loss.ppo_epochs=2", "checkpoint.interval_iterations=1", "logger.backend=csv"],
        tmp_path,
    )
    assert "PPO_DONE" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert (tmp_path / "checkpoints" / "ppo_final.pt").exists()
    assert list((tmp_path / "checkpoints").glob("ppo_*.pt"))

    play = _run(["play.py", f"play.checkpoint={tmp_path / 'checkpoints' / 'ppo_final.pt'}", "play.steps=20",
                 "play.num_envs=4", "play.headless=true"], tmp_path / "play")
    assert "PLAY_DONE" in play.stdout, play.stdout[-3000:] + play.stderr[-3000:]
