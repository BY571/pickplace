"""Continuous demo (pipeline/0_state_teacher/demo.py) on a tiny freshly trained teacher."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "pipeline" / "0_state_teacher"


def _run(args, tmp_path, timeout):
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "FOOD_ROBOT_ARTIFACTS": str(tmp_path / "artifacts"),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(tmp_path),
        "WANDB_SILENT": "true",
    }
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=timeout,
                          env=env, cwd=REPO)
    return proc, proc.stdout[-4000:] + proc.stderr[-4000:]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("teacher")
    proc, tail = _run(
        [
            TEACHER / "train.py",
            "env.num_envs=32",
            "env.belt.speed=0.3",  # short belt: the demo carousel recycles pallets within seconds
            "env.belt.place_window=0.6",
            "collector.rollout_steps=16",
            "loss.ppo_epochs=1",
            "loss.mini_batch_size=256",
            "logger.backend=null",
            "max_iterations=2",
            "checkpoint.interval_frames=0",
            "worker.enabled=false",
            f"run.dir={tmp_path / 'run'}",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        tmp_path,
        timeout=2400,
    )
    assert "PPO_DONE" in proc.stdout, tail
    path = tmp_path / "run" / "checkpoints" / "ppo_teacher_final.pt"
    assert path.exists(), tail
    return path


def _demo(checkpoint, tmp_path, *options):
    out = tmp_path / "demo.mp4"
    proc, tail = _run([TEACHER / "demo.py", f"checkpoint={checkpoint}", f"out={out}", *options], tmp_path, 1800)
    assert "DEMO_DONE" in proc.stdout, tail
    lines = [line for line in proc.stdout.splitlines() if "DEMO {" in line]
    assert lines, tail
    summary = json.loads(lines[-1][lines[-1].index("DEMO {") + len("DEMO "):])
    assert out.stat().st_size > 0 and out.with_name("demo_frame.png").stat().st_size > 0
    assert json.loads(out.with_suffix(".json").read_text()) == summary
    # the robot is never reset: no episode ever ended and the episode counter ran through every step
    assert summary["robot_resets"] == 0 and summary["episode_ends"] == 0, summary
    assert summary["episode_length_steps"] == summary["steps"], summary
    counters = ("placed", "missed", "dropped", "misplaced", "pending")
    assert all(isinstance(summary[k], int) and summary[k] >= 0 for k in counters), summary
    assert summary["placed"] + summary["missed"] + summary["pending"] == summary["bowls_seen"], summary
    return summary


def test_demo_runs_continuously_and_recycles_bowls(checkpoint, tmp_path):
    s = _demo(checkpoint, tmp_path, "seconds=4", "bowls=2")
    assert s["steps"] == round(4 / s["settings"]["step_dt"]) and s["seconds"] == pytest.approx(4.0)
    # 2 pallets 0.29 m apart at 0.3 m/s: one bowl starts inside the belt, the carousel then sends one per ~1 s
    assert s["bowls_seen"] >= 3, s
    assert s["missed"] >= 1, s
    assert s["settings"]["bowls"] == 2 and s["settings"]["home_between"] is False


def test_demo_with_a_finite_bowl_count_and_homing_stops_early(checkpoint, tmp_path):
    s = _demo(checkpoint, tmp_path, "seconds=20", "bowls=2", "total_bowls=2", "home_between=true")
    assert s["bowls_seen"] == 2 and s["pending"] == 0, s
    assert s["seconds"] < 20.0, s  # stopped once both bowls were resolved
    assert s["settings"]["home_between"] is True
