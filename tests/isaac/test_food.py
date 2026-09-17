import pytest

pytestmark = pytest.mark.isaac


def test_success_drop_and_food_reset(run_scenario):
    r = run_scenario("food_behaviour")
    # env 0: food placed into the bowl and released -> success with the bonus in the reward
    assert r["first_termination"]["0"] == "success"
    assert r["reward_at_termination"]["0"] > 0.8 * r["success_bonus"]
    # env 1: food teleported below the table -> food_off_table with the penalty
    assert r["first_termination"]["1"] == "food_off_table"
    assert r["reward_at_termination"]["1"] < -0.8 * r["food_drop_penalty"]
    # env 2: untouched -> no success/food termination within the observed window, rewards finite
    assert r["first_termination"].get("2") not in ("success", "food_off_table")
    assert r["rewards_finite"]
    # food spawns randomized inside the ingredient bowl
    spread = r["food_spawn_offsets"]
    assert all(abs(dx) <= r["food_spawn_range"] + 5e-3 and abs(dy) <= r["food_spawn_range"] + 5e-3 for dx, dy in spread)
    assert r["reward_terms"] == sorted(
        ["reach_food", "grasp", "grasp_lift", "transport", "transport_fine", "place_success", "bowl_failure",
         "food_dropped", "bowl_disturbance", "action_rate", "joint_vel"]
    )
