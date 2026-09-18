import pytest

from food_robot import rewards as R

STAGED = {
    "reach_food": 1.0, "grasp": 2.0, "grasp_lift": 5.0, "transport": 10.0, "transport_fine": 5.0,
    "bowl_disturbance": -1.0, "action_rate": -1.0e-4, "joint_vel": -1.0e-4,
    "success": 150.0, "bowl_failure": -150.0, "food_dropped": -150.0,
}


def test_vocabulary_is_dense_then_events_without_duplicates():
    assert R.REWARD_TERMS == R.DENSE_TERMS + R.EVENT_TERMS
    assert len(set(R.REWARD_TERMS)) == len(R.REWARD_TERMS)
    assert set(R.EVENT_SOURCES) == set(R.EVENT_TERMS)
    assert R.EVENT_SOURCES["bowl_failure"] == ("bowl_off_belt", "bowl_tipped")


def test_staged_v1_matches_todays_reward():
    assert "staged_v1" in R.available_reward_sets()
    assert R.load_reward_set("staged_v1") == STAGED


def test_simple_v1_is_the_five_positive_task_terms():
    simple = {"reach_food": 1.0, "grasp": 2.0, "grasp_lift": 5.0, "transport": 10.0, "success": 100.0}
    weights = R.load_reward_set("simple_v1")
    assert set(weights) == set(R.REWARD_TERMS)
    assert {t: w for t, w in weights.items() if w != 0.0} == simple
    assert all(weights[t] == 0.0 for t in R.REWARD_TERMS if t not in simple)


def test_unlisted_terms_default_to_zero(tmp_path):
    p = tmp_path / "sparse.yaml"
    p.write_text("success: 1.0\n")
    weights = R.load_reward_set(str(p))
    assert weights["success"] == 1.0
    assert all(weights[t] == 0.0 for t in R.REWARD_TERMS if t != "success")


def test_unknown_term_in_a_set_is_an_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("reach_fod: 1.0\n")
    with pytest.raises(KeyError, match="reach_fod"):
        R.load_reward_set(str(p))


def test_unknown_set_name_lists_available():
    with pytest.raises(FileNotFoundError, match="staged_v1"):
        R.load_reward_set("does_not_exist")


def test_resolve_defaults_to_staged_v1():
    assert R.resolve_reward_weights({}) == STAGED


def test_resolve_precedence_set_then_legacy_then_reward_weights():
    cfg = {
        "reward_set": "staged_v1",
        "rewards": {"reach_food": 2.0},
        "success_bonus": 100.0,
        "bowl_failure_penalty": None,
        "food_drop_penalty": 10.0,
        "reward_weights": {"reach_food": 3.0, "grasp": 0.0},
    }
    w = R.resolve_reward_weights(cfg)
    assert w["reach_food"] == 3.0          # reward_weights beats legacy `rewards`
    assert w["grasp"] == 0.0
    assert w["success"] == 100.0           # legacy one-shot field, positive bonus
    assert w["food_dropped"] == -10.0      # legacy penalty is stored as a negative weight
    assert w["bowl_failure"] == -150.0     # None leaves the set's value


@pytest.mark.parametrize("bad", [{"rewards": {"place_success": 5.0}}, {"rewards": {"bogus": 1.0}}])
def test_legacy_rewards_accepts_dense_terms_only(bad):
    with pytest.raises(KeyError):
        R.resolve_reward_weights(bad)


def test_reward_weights_rejects_unknown_names():
    with pytest.raises(KeyError, match="nope"):
        R.resolve_reward_weights({"reward_weights": {"nope": 1.0}})


def test_weight_vector_order_and_isaac_weight():
    w = R.load_reward_set("staged_v1")
    assert R.weight_vector(w) == [STAGED[t] for t in R.REWARD_TERMS]
    assert R.isaac_weight(0.0) == 1.0 and R.isaac_weight(-1e-4) == -1e-4
