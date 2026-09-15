import pytest

pytestmark = pytest.mark.isaac

TERMS = {"time_out", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "success", "food_off_table"}


@pytest.mark.parametrize("cameras,privileged", [(0, 1), (1, 1)])
def test_specs_rollout_and_stats(run_scenario, cameras, privileged):
    r = run_scenario("torchrl_env", "specs", cameras, privileged)
    assert r["check_env_specs"] == "ok"
    assert r["batch_size"] == [4]
    assert r["rollout_ee_pos_shape"] == [4, 20, 3]
    assert r["action_shape"] == [4, 7]
    assert ["proprio", "ee_pos"] in r["actor_keys"] and ["belt", "bowl_pos"] in r["actor_keys"]
    assert (["pixels", "wrist_rgb"] in r["leaf_keys"]) == bool(cameras)
    assert (["privileged", "food_pos"] in r["leaf_keys"]) == bool(privileged)
    assert set(r["termination_stats"]) == TERMS
    assert r["config_errors"] == {"unknown_key": "ValueError", "unknown_arm": "KeyError"}


def test_partial_reset_only_resets_masked_env(run_scenario):
    r = run_scenario("torchrl_env", "partial_reset", 0, 1)
    assert r["reset_env_x"] == pytest.approx(r["entry_x"], abs=0.02)
    assert r["other_env_x_after"] == pytest.approx(r["other_env_x_before"], abs=1e-3)
    assert r["other_env_x_before"] > r["entry_x"] + 0.02  # it had moved and was left alone
