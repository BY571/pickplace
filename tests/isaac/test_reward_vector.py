import pytest

pytestmark = pytest.mark.isaac


def test_reward_vector_reproduces_isaac_lab_reward_and_specs(run_scenario):
    r = run_scenario("reward_vector")
    assert r["check_env_specs"] == "ok"
    assert r["reward_terms_shape"] == [16, 13]
    assert r["done_rows"] > 0, "rollout must include episode ends"
    # the TorchRL linearised reward equals Isaac Lab's own scalar, step by step
    assert r["max_abs_reward_diff"] < 1e-4
    # the rollout must actually produce event rows, or the event checks below are vacuous
    assert r["event_rows"] > 0
    # event components are 0/1 and only non-zero on done rows
    assert r["events_binary"] and r["events_only_on_done"]
    # episode_reward_terms sums the vector (checked on non-done rows, where the running sum is valid)
    assert r["max_abs_term_sum_diff"] < 1e-3
    # food_in_bowl pays only where the food rests released in its bowl: env 3 (forced there), no other env
    in_bowl = r["food_in_bowl_max"]
    assert in_bowl[3] > 0
    assert all(v == 0.0 for i, v in enumerate(in_bowl) if i != 3)
    # return_home is gated by the same "released in the bowl" condition (it pays wherever the arm is)
    home = r["return_home_max"]
    assert home[3] > 0
    assert all(v == 0.0 for i, v in enumerate(home) if i != 3)
    # success_requires_home false (the default): the success termination is exactly the pre-v3 one
    assert "home_tolerance" not in r["success_params"] and "arm_cfg" not in r["success_params"]


def test_reward_set_switch_changes_the_scalar_only(run_scenario):
    r = run_scenario("reward_vector", "--sparse")
    assert r["check_env_specs"] == "ok"
    # success-only weights: the scalar reward is exactly success_weight * success component
    assert r["max_abs_sparse_diff"] < 1e-6
