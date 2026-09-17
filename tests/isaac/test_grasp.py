import pytest

pytestmark = pytest.mark.isaac


def test_a_scripted_grasp_lifts_the_food(run_scenario):
    r = run_scenario("grasp_probe")
    assert r["lifted_fraction"] >= 0.9, r  # a closed, raised gripper holds the food in almost every env
    assert r["max_food_height"] > 0.10  # above grasp_lift's threshold
    assert r["slip_distance"] < 0.03  # the food stays with the TCP rather than squirting out


def test_transport_reward_requires_a_lift(run_scenario):
    r = run_scenario("grasp_probe", "--transport-gate")
    assert r["transport_while_resting"] == 0.0  # fingers around the food on the table pays nothing
    assert r["transport_while_lifted"] > 0.0
    assert r["grasp_while_resting"] > 0.0  # but the stepping-stone `grasp` term must pay before any lift
    assert r["grasp_while_lifted"] > 0.0
