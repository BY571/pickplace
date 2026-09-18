import pytest

pytestmark = pytest.mark.isaac


def test_success_requires_the_arm_back_home(run_scenario):
    r = run_scenario("return_home")
    tol = r["home_tolerance"]
    # (c) at reset the arm counts as home in every env (reset offsets stay inside home_tolerance)
    assert all(d <= tol for d in r["reset_distance"]), r["reset_distance"]
    # (a) food settled in the bowl (counter past settle_steps + 5) while the arm is away: no success, no episode end
    assert r["away_distance_min"] > tol
    assert r["counter_away"] >= r["settle_steps"] + 5, r
    assert not r["fired_away"] and not r["env0_done_while_away"]
    assert r["return_home_max"][0] > 0 and r["food_in_bowl_max"][0] > 0
    assert r["return_home_max"][1] == 0.0  # control env: food not in its bowl
    # (b) arm written back to the default pose: success within a few steps
    assert r["fired_home_after"] is not None and r["fired_home_after"] <= 5, r
