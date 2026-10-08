import pytest

pytestmark = pytest.mark.isaac


def test_joint_adapters_drive_the_robot_in_their_own_units(run_scenario):
    r = run_scenario("action_adapter")

    # an adapter needs an action mode it can convert into, and says so rather than silently misbehaving
    assert r["rejected"]["raised"]
    assert "joint_pos" in r["rejected"]["message"]

    v = r["velocity"]
    assert v["action_dim"] == 8, "7 arm joints + gripper"

    # zero velocity holds station. The arm is position-controlled with the plain Franka gains, so it sags a
    # little against gravity; what must not happen is the target chasing that sag, which shows up as drift
    # that keeps growing with the step count.
    assert v["held_max_rad"] < 0.05, v
    assert v["held_50_rad"] < 2.0 * v["held_max_rad"] + 0.01, f"drift accumulates: {v}" 

    # a commanded velocity moves the commanded joints the commanded way, and leaves the others alone
    moved, expected = v["moved"], v["expected"]
    for i, (got, want) in enumerate(zip(moved, expected)):
        if want == 0.0:
            assert abs(got) < 0.05, f"joint {i} moved {got} on a zero command: {v}"
        else:
            assert got * want > 0, f"joint {i} moved {got} against a command of {want}: {v}"
            assert 0.4 * abs(want) < abs(got) < 1.6 * abs(want), f"joint {i}: {got} vs {want}: {v}"

    # an absolute target is reached and held, which is the property velocities do not have
    p = r["position"]
    assert p["moved_rad"] > 0.05, p
    assert p["error_rad"] < 0.05, p
