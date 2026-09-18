import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "pipeline" / "0_state_teacher" / "train.py"
OUTCOMES = ("success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out")
ITERATIONS = 12
N, ROLLOUT = 32, 16  # 512 frames per iteration


def _lines(stdout, prefix):
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _train(tmp_path, *overrides, timeout=2400):
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "FOOD_ROBOT_ARTIFACTS": str(tmp_path / "artifacts"),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(tmp_path),
        "WANDB_SILENT": "true",
    }
    args = [
        str(TRAIN),
        f"env.num_envs={N}",
        "env.belt.speed=0.3",  # short belt cycle so episodes end within the smoke run
        "env.belt.place_window=0.6",
        f"collector.rollout_steps={ROLLOUT}",
        "loss.ppo_epochs=1",
        "loss.mini_batch_size=256",
        "logger.backend=null",
        f"run.dir={tmp_path / 'run'}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
        *overrides,
    ]
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    return proc, proc.stdout[-4000:] + proc.stderr[-4000:]


def test_teacher_trains_logs_terms_and_writes_checkpoints_with_manifests(tmp_path):
    proc, tail = _train(
        tmp_path, f"max_iterations={ITERATIONS}", "checkpoint.interval_frames=3000", "worker.enabled=false"
    )
    assert "PPO_DONE" in proc.stdout, tail

    metrics = _lines(proc.stdout, "METRICS ")
    assert len(metrics) == ITERATIONS, tail
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))), "NaN/inf"
    finished = [m for m in metrics if "train/success_rate" in m]
    assert finished, "no episode finished during training"
    for m in finished:
        assert sum(m[f"train/{t}_rate"] for t in OUTCOMES) >= 1.0 - 1e-6
        assert "train/terms/reach_food" in m and "train/terms/food_dropped" in m
    for key in ("train/kl_approx", "train/clip_fraction", "train/entropy", "perf/frames_per_hour"):
        assert key in metrics[-1], key

    run = tmp_path / "run"
    ckpts = sorted((run / "checkpoints").glob("ppo_teacher_*.pt"))
    names = {p.stem for p in ckpts}
    assert "ppo_teacher_final" in names
    assert len(names) >= 3, names  # 3000-frame interval over 6144 frames: two periodic + final
    for p in ckpts:
        m = json.loads(p.with_suffix(".json").read_text())
        assert m["frames"] > 0 and len(m["sha256"]) == 64 and m["config"]["env"]["reward_set"] == "simple_v1"
        assert m["eval"] is None and m["video"] is None
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["kind"] == "state_teacher" and manifest["stop_reason"] == "total_frames"
    assert set(manifest["checkpoints"]) == names
    assert manifest["config"]["env"]["reward_weights"] == {}


def test_teacher_early_stop_saves_final_checkpoint(tmp_path):
    proc, tail = _train(
        tmp_path, "max_iterations=40", "checkpoint.interval_frames=0", "worker.enabled=false",
        "early_stop.success_rate=0.0", "early_stop.consecutive_iterations=2",
    )
    assert "PPO_DONE" in proc.stdout, tail
    assert _lines(proc.stdout, "STOP_REASON ")[-1]["reason"] == "early_stop"
    assert len(_lines(proc.stdout, "METRICS ")) < 40
    assert (tmp_path / "run" / "checkpoints" / "ppo_teacher_final.pt").exists()


def test_worker_evaluates_and_renders_every_checkpoint(tmp_path):
    proc, tail = _train(
        tmp_path, "max_iterations=6", "checkpoint.interval_frames=1500",
        "worker.enabled=true", "worker.eval_num_envs=16", "worker.render_seconds=3",
        "logger.backend=wandb", timeout=3600,
    )
    assert "PPO_DONE" in proc.stdout, tail
    ckpts = sorted((tmp_path / "run" / "checkpoints").glob("ppo_teacher_*.pt"))
    assert {p.stem for p in ckpts} >= {"ppo_teacher_1536", "ppo_teacher_3072", "ppo_teacher_final"}
    for p in ckpts:
        m = json.loads(p.with_suffix(".json").read_text())
        assert m["eval"] is not None, (p.name, tail)
        assert 0.0 <= m["eval"]["success_rate"] <= 1.0 and m["eval"]["finished_fraction"] > 0
        assert "terms/reach_food" in m["eval"] and m["eval_num_envs"] == 16
        assert m["video"] == f"{p.stem}.mp4" and (p.parent / m["video"]).stat().st_size > 0
    worker_log = (tmp_path / "run" / "worker.log").read_text()
    assert "EVALUATE_DONE" in worker_log and "RENDER_DONE" in worker_log
    metrics = _lines(proc.stdout, "EVAL_LOGGED ")
    assert {m["checkpoint_frames"] for m in metrics} >= {1536, 3072}
