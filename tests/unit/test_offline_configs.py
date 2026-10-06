"""A fair comparison needs the same data, inputs, evaluation and budget for every algorithm."""

from pathlib import Path

import pytest
import yaml

OFFLINE_DIR = Path(__file__).resolve().parents[2] / "pipeline" / "2_1_offline_rl"
ALGORITHMS = sorted(p.name for p in OFFLINE_DIR.iterdir() if (p / "config.yaml").exists())
# Everything except the per-algorithm objective, its optimizer and the bookkeeping that names the run.
SHARED_KEYS = ("device", "seed", "matmul_precision", "data", "batch_size", "gradient_steps", "max_hours",
               "reward_scale", "log_interval", "prefetch_depth", "prefetch_workers", "network", "eval",
               "checkpoint", "app")


def _config(name: str) -> dict:
    return yaml.safe_load((OFFLINE_DIR / name / "config.yaml").read_text())


def test_all_three_algorithms_exist():
    assert ALGORITHMS == ["bc", "iql", "td3_bc"]


@pytest.mark.parametrize("key", SHARED_KEYS)
def test_the_comparison_protocol_is_identical_across_algorithms(key):
    values = {name: _config(name)[key] for name in ALGORITHMS}
    assert len(set(map(repr, values.values()))) == 1, f"{key} differs across algorithms: {values}"


@pytest.mark.parametrize("name", ALGORITHMS)
def test_students_see_only_the_two_cameras(name):
    assert _config(name)["network"]["in_keys"] == [["pixels", "overview_rgb"], ["pixels", "wrist_rgb"]]


@pytest.mark.parametrize("name", ALGORITHMS)
def test_checkpoints_land_on_an_evaluation_step(name):
    cfg = _config(name)
    assert cfg["checkpoint"]["interval"] % cfg["eval"]["interval"] == 0
    assert cfg["logger"]["group"] == "offline_rl"
