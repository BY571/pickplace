"""Every offline trainer really trains, evaluates online and writes checkpoints with eval-bearing manifests.

The shards are tiny synthetic ones (random images): this asserts the pipeline, not what is learned. Their
manifest carries a fast-belt env config, so the online evaluation's ``max_episode_length`` is short.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
OFFLINE = REPO / "pipeline" / "2_1_offline_rl"

STRIDE, STEPS = 8, 32  # 256 rows per shard, time-major
STEPS_TOTAL, EVAL_INTERVAL, CHECKPOINT_INTERVAL, LOG_INTERVAL = 200, 100, 200, 50

sys.path.insert(0, str(REPO))
from pickplace.artifacts import write_json  # noqa: E402
from pickplace.datasets import STORAGE_DIR  # noqa: E402


def _fake_shard(path: Path, seed: int):
    """A shard in collect.py's layout: 84 px cameras, a few episode boundaries, a fast-belt env config."""
    path.mkdir(parents=True, exist_ok=True)
    frames = STRIDE * STEPS
    generator = torch.Generator().manual_seed(seed)
    terminated = torch.zeros(frames, 1, dtype=torch.bool)
    truncated = torch.zeros(frames, 1, dtype=torch.bool)
    terminated[[STRIDE * 10 + 1, STRIDE * 20 + 2]] = True  # two episodes end mid-shard
    truncated[[STRIDE * 15 + 3]] = True
    td = TensorDict(
        {
            "pixels": TensorDict(
                {name: torch.randint(0, 256, (frames, 84, 84, 3), dtype=torch.uint8, generator=generator)
                 for name in ("overview_rgb", "wrist_rgb")},
                batch_size=[frames],
            ),
            "action": torch.rand(frames, 7, generator=generator) * 1.8 - 0.9,
            "next": TensorDict(
                {"reward": torch.randn(frames, 1, generator=generator),
                 "terminated": terminated, "truncated": truncated, "done": terminated | truncated},
                batch_size=[frames],
            ),
        },
        batch_size=[frames],
    )
    td.memmap_(str(path / STORAGE_DIR))
    write_json(path / "manifest.json", {
        "name": path.name, "frames": frames, "successor_stride": STRIDE, "seed": seed, "noise_sigma": 0.0,
        "stats": {"success_rate": 0.5},
        "env": {"task": "FoodRobot-Cell-v0", "num_envs": STRIDE, "cameras": True, "image_size": [84, 84],
                "privileged_information": True, "belt": {"speed": 0.3, "place_window": 0.6},
                "reward_set": "simple_v3b", "seed": 0, "device": "cuda:0"},
    })
    return path


def _lines(stdout, prefix):
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _train(algorithm: str, tmp_path: Path, *overrides, timeout=2400):
    shards = [_fake_shard(tmp_path / "shards" / name, seed) for seed, name in enumerate(("a", "b"))]
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "FOOD_ROBOT_ARTIFACTS": str(tmp_path / "artifacts")}
    args = [
        str(OFFLINE / algorithm / "train.py"),
        f"data.shards=[{shards[0]},{shards[1]}]",
        "data.proportions=[0.5,0.5]",
        "batch_size=32",
        f"gradient_steps={STEPS_TOTAL}",
        f"log_interval={LOG_INTERVAL}",
        f"eval.interval={EVAL_INTERVAL}",
        f"checkpoint.interval={CHECKPOINT_INTERVAL}",
        "eval.num_envs=4",
        "prefetch_depth=2",
        "prefetch_workers=1",
        "logger.backend=null",
        f"run.dir={tmp_path / 'run'}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
        *overrides,
    ]
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    return proc, proc.stdout[-4000:] + proc.stderr[-4000:]


@pytest.mark.parametrize(("algorithm", "losses"), [
    ("bc", ["train/loss_bc"]),
    ("iql", ["train/loss_actor", "train/loss_qvalue", "train/loss_value"]),
    ("cql", ["train/loss_actor", "train/loss_qvalue", "train/loss_cql", "train/loss_alpha"]),
    ("td3_bc", ["train/loss_actor", "train/loss_qvalue", "train/bc_loss"]),
])
def test_offline_trainer_trains_evaluates_and_checkpoints(algorithm, losses, tmp_path):
    proc, tail = _train(algorithm, tmp_path)
    assert f"{algorithm.upper()}_DONE" in proc.stdout, tail

    info = _lines(proc.stdout, "RUN_INFO ")[-1]
    assert info["obs_keys"] == [["pixels", "overview_rgb"], ["pixels", "wrist_rgb"]]
    assert info["image_shapes"] == [[84, 84, 3], [84, 84, 3]] and info["action_dim"] == 7

    metrics = _lines(proc.stdout, "METRICS ")
    assert [m["step"] for m in metrics] == list(range(LOG_INTERVAL, STEPS_TOTAL + 1, LOG_INTERVAL)), tail
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))), "NaN/inf"
    for key in [*losses, "perf/steps_per_s", "perf/sample_s_per_step", "perf/eval_fraction"]:
        assert key in metrics[-1], (key, tail)

    evals = _lines(proc.stdout, "EVAL ")
    assert [e["step"] for e in evals] == [EVAL_INTERVAL, STEPS_TOTAL], tail
    assert all(0.0 <= e["success_rate"] <= 1.0 and e["finished_fraction"] > 0 for e in evals), evals

    run = tmp_path / "run"
    names = {p.stem for p in (run / "checkpoints").glob(f"{algorithm}_*.pt")}
    assert names == {f"{algorithm}_{STEPS_TOTAL}", f"{algorithm}_final"}, names
    for path in (run / "checkpoints").glob(f"{algorithm}_*.json"):
        manifest = json.loads(path.read_text())
        assert manifest["algorithm"] == algorithm and len(manifest["sha256"]) == 64
        assert manifest["eval"] is not None and manifest["eval_step"] == STEPS_TOTAL
        assert 0.0 <= manifest["eval"]["success_rate"] <= 1.0 and manifest["eval_num_envs"] == 4
        assert len(manifest["shards"]) == 2 and manifest["shards"][0]["proportion"] == 0.5
        assert manifest["shards"][0]["successor_stride"] == STRIDE and manifest["git_commit"]
        assert manifest["config"]["network"]["in_keys"] == [["pixels", "overview_rgb"], ["pixels", "wrist_rgb"]]
        assert torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)["actor"]

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["algorithm"] == algorithm and manifest["stop_reason"] == "gradient_steps"
    assert manifest["gradient_steps"] == STEPS_TOTAL and manifest["transitions"] > 0
    assert [e["step"] for e in manifest["eval_history"]] == [EVAL_INTERVAL, STEPS_TOTAL]
    assert manifest["best_eval"]["step"] in (EVAL_INTERVAL, STEPS_TOTAL)
