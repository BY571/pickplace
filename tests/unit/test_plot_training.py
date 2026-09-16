import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plot_training", REPO / "scripts" / "plot_training.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

RATES = {f"train/{t}_rate": 0.0 for t in pt.OUTCOMES}


def row(i, ret, reach, success=0.1, nan=False):
    r = {"frames": 1000 * (i + 1), "iteration": i, "train/episode_return": ret, "episode_reward/reach_food": reach}
    r.update(RATES)
    r["train/success_rate"] = success
    r["train/bowl_exited_zone_rate"] = 1.0 - success
    if nan:
        r["train/loss_critic"] = float("nan")
    return r


def healthy():
    rows = [row(i, -100 + 5 * i, 0.1 * i) for i in range(10)]
    rows[-1]["eval/success_rate"] = 0.2
    return rows


def test_load_rows_accepts_raw_log_lines_and_bare_json(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text("noise\n2026 INFO METRICS " + json.dumps(row(0, 1.0, 0.0)) + "\n" + json.dumps(row(1, 2.0, 0.1)) + "\n")
    assert [r["iteration"] for r in pt.load_rows(path)] == [0, 1]


def test_healthy_run_passes_all_checks():
    assert all(pt.sanity_checks(healthy()).values())


def test_flat_reward_fails_trend_checks():
    checks = pt.sanity_checks([row(i, -100.0, 0.5) | ({"eval/success_rate": 0.0} if i == 9 else {}) for i in range(10)])
    assert not checks["return_trends_up"] and not checks["reach_food_trends_up"]


def test_nan_and_bad_outcome_sums_fail():
    rows = healthy()
    rows[3] = row(3, -85, 0.3, nan=True)
    rows[4]["train/bowl_exited_zone_rate"] = 0.2
    checks = pt.sanity_checks(rows)
    assert not checks["no_nan"]
    assert not checks["outcome_rates_sum_to_one"]


def test_missing_eval_fails():
    rows = [row(i, -100 + 5 * i, 0.1 * i) for i in range(10)]
    assert not pt.sanity_checks(rows)["eval_logged"]
