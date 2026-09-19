import pytest

pytestmark = pytest.mark.isaac


def test_robot_reset_default_and_override_reach_the_event_params(run_scenario):
    r = run_scenario("robot_reset")
    d, o = r["default"], r["override"]
    assert d["position_range"] == [-0.02, 0.02]  # today's reset_joints_by_offset params, unchanged
    assert d["velocity_range"] == [0.0, 0.0]
    assert o["position_range"] == [-0.25, 0.25]  # env.robot_reset override reaches the built event params
    assert o["velocity_range"] == [-0.1, 0.1]
    # restricted to the arm joints only, in both cases: the gripper fingers are never randomized by this event
    # (their soft limits are ~[0, 0.04] m, so a range this wide would otherwise clamp them mostly shut)
    assert d["joint_names"] == d["arm_joint_names"] == o["joint_names"] == o["arm_joint_names"]
    assert not set(d["joint_names"]) & set(d["gripper_joint_names"])
