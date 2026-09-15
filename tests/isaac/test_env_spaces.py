import pytest

pytestmark = pytest.mark.isaac

PROPRIO = {"joint_pos_rel": [9], "joint_vel_rel": [9], "gripper_pos": [2], "ee_pos": [3], "ee_quat": [4]}
BELT = {"bowl_pos": [3]}
PIXELS = {"wrist_rgb": [128, 128, 3], "overview_rgb": [128, 128, 3]}
PRIVILEGED = {"food_pos": [3], "food_quat": [4], "is_grasped": [1]}


@pytest.mark.parametrize(
    "cameras,privileged,action_mode,action_dim",
    [
        (0, 1, "ee_delta_pose", 7),
        (1, 0, "ee_delta_pose", 7),
        (1, 1, "ee_delta_pose", 7),
        (0, 1, "joint_pos", 8),
    ],
)
def test_groups_and_shapes_follow_flags_and_arm(run_scenario, cameras, privileged, action_mode, action_dim):
    r = run_scenario("env_spaces", cameras, privileged, action_mode)
    expected = {"proprio": {**PROPRIO, "last_action": [action_dim]}, "belt": BELT}
    if cameras:
        expected["pixels"] = PIXELS
    if privileged:
        expected["privileged"] = PRIVILEGED
    assert r["shapes"] == expected
    assert r["action_dim"] == action_dim
    assert r["all_finite"]


def test_both_food_information_sources_disabled_is_rejected(run_scenario):
    r = run_scenario("env_spaces", 0, 0, "ee_delta_pose", "--expect-error")
    assert "privileged_information" in r["error"]
