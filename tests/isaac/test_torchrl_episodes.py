import pytest

pytestmark = pytest.mark.isaac


def test_episode_boundaries_at_torchrl_level(run_scenario):
    r = run_scenario("torchrl_episodes", timeout=1200)

    assert r["n_done"] > 0, "no episode finished within the rollout window"
    assert min(r["per_env_done"]) >= 2, "every env must see at least 2 episodes"

    assert r["advantage_finite"]
    assert r["value_target_finite"]

    # StepCounter: native autoreset zeroes ("next", "step_count") on the done row itself, and the
    # following row (first step of the new episode) starts back at 1.
    assert r["done_step_counts"] == [0]
    assert r["after_done_step_counts"] == [1]

    # RewardSum's episode_reward is invalidated (NaN) on done rows by the same native-autoreset
    # mechanism as raw observations, and the row after a done shows a freshly-accumulated value
    # (matching that row's own reward, not a continuation of the finished episode's total).
    assert r["episode_reward_nan_only_on_done"]
    assert r["after_done_reward_matches"]

    # NaNs in ("next", <obs>) appear only on done rows, for every observation group checked.
    for key, ok in r["nan_only_on_done"].items():
        assert ok, f"NaN pattern for {key} does not line up exactly with done rows"

    # Episode outcome transform
    assert r["outcome_only_on_done"]
    assert r["every_done_has_outcome"]
    for term, rate in r["last_episode_rates"].items():
        assert rate == pytest.approx(r["termination_stats"][term], abs=1e-6), term
    assert r["outcome_counts"]["bowl_exited_zone"] > 0  # zero actions: the bowl rides out of the zone
    assert r["outcome_counts"]["success"] == 0
    assert {"reach_food", "grasp_lift", "transport"} <= set(r["reward_terms"])
