import pytest
import torch

from pickplace import action_adapters as A

SCALE = 0.5  # ArmCfg.joint_action_scale for every arm shipped here
DEFAULT = torch.tensor([[0.0, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74]])


def _env_applies(action: torch.Tensor, scale: float = SCALE) -> torch.Tensor:
    """What Isaac Lab's JointPositionActionCfg(use_default_offset=True) does with an action."""
    return DEFAULT + scale * action


def test_absolute_targets_invert_the_envs_own_mapping_exactly():
    target = DEFAULT + torch.tensor([[0.1, -0.2, 0.3, 0.05, -0.4, 0.15, 0.25]])
    native = A.joint_target_to_native(target, DEFAULT, SCALE)
    assert torch.allclose(_env_applies(native), target, atol=1e-6)


def test_a_target_equal_to_the_default_pose_is_the_zero_action():
    assert torch.allclose(A.joint_target_to_native(DEFAULT, DEFAULT, SCALE), torch.zeros_like(DEFAULT))


def test_targets_are_clamped_into_the_joint_range_before_inverting():
    low, high = DEFAULT - 0.1, DEFAULT + 0.1
    native = A.joint_target_to_native(DEFAULT + 1.0, DEFAULT, SCALE, (low, high))
    assert torch.allclose(_env_applies(native), high, atol=1e-6)
    native = A.joint_target_to_native(DEFAULT - 1.0, DEFAULT, SCALE, (low, high))
    assert torch.allclose(_env_applies(native), low, atol=1e-6)


def test_velocity_integrates_exactly_one_control_period():
    q = DEFAULT + 0.3
    q_dot = torch.tensor([[0.5, -0.25, 0.0, 1.0, -1.0, 0.1, 0.2]])
    dt = 0.02
    native = A.joint_velocity_to_native(q_dot, q, DEFAULT, SCALE, dt)
    assert torch.allclose(_env_applies(native), q + q_dot * dt, atol=1e-6)


def test_zero_velocity_holds_the_current_pose_wherever_it_is():
    # the property that makes velocity a safe default: the no-op does not depend on the default pose
    for offset in (-0.4, 0.0, 0.9):
        q = DEFAULT + offset
        native = A.joint_velocity_to_native(torch.zeros_like(q), q, DEFAULT, SCALE, 0.02)
        assert torch.allclose(_env_applies(native), q, atol=1e-6)


@pytest.mark.parametrize(("fraction", "closed_is_high", "expected_sign"), [
    (0.0, True, 1.0),    # DROID convention: 0 is open, the env opens on a positive action
    (1.0, True, -1.0),
    (0.49, True, 1.0),
    (0.51, True, -1.0),
    (0.0, False, -1.0),  # the opposite convention, in case a dataset uses it
    (1.0, False, 1.0),
])
def test_gripper_fraction_maps_to_the_envs_sign_convention(fraction, closed_is_high, expected_sign):
    out = A.gripper_fraction_to_binary(torch.tensor([[fraction]]), closed_is_high=closed_is_high)
    assert out.item() == expected_sign


def test_gripper_threshold_is_configurable():
    f = torch.tensor([[0.3]])
    assert A.gripper_fraction_to_binary(f, closed_at=0.5).item() == 1.0
    assert A.gripper_fraction_to_binary(f, closed_at=0.2).item() == -1.0


def test_velocity_and_position_adapters_agree_on_the_same_target():
    q, dt = DEFAULT + 0.2, 0.02
    q_dot = torch.tensor([[0.3, 0.0, -0.3, 0.1, 0.0, 0.2, -0.1]])
    from_velocity = A.joint_velocity_to_native(q_dot, q, DEFAULT, SCALE, dt)
    from_position = A.joint_target_to_native(q + q_dot * dt, DEFAULT, SCALE)
    assert torch.allclose(from_velocity, from_position, atol=1e-7)


def test_unknown_adapter_is_rejected_by_name():
    with pytest.raises(ValueError, match="joint_velocity"):
        A.make_joint_action_adapter("joint_velocities")  # plural: a plausible typo


def test_every_adapter_is_described_and_named():
    assert A.ADAPTERS[0] == "native"
    for mode in A.ADAPTERS:
        assert A.describe(mode)
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    assert A.adapted_action_names(joints, "native") == []
    for mode, unit in (("joint_velocity", "qd"), ("joint_position", "q")):
        names = A.adapted_action_names(joints, mode)
        assert len(names) == 8 and names[0] == f"{unit}[panda_joint1]" and names[-1] == "gripper"


def test_the_default_env_keeps_todays_action_space():
    from pickplace.config import DEFAULT_ENV

    assert DEFAULT_ENV["action_adapter"] == "native"
    assert DEFAULT_ENV["action_mode"] == "ee_delta_pose"


def test_a_full_arm_sweep_stays_inside_the_limits_it_was_given():
    low, high = DEFAULT - 0.5, DEFAULT + 0.5
    q = DEFAULT.clone()
    q_dot = torch.full_like(q, 10.0)  # far faster than any joint can move
    for _ in range(50):
        native = A.joint_velocity_to_native(q_dot, q, DEFAULT, SCALE, 0.02, (low, high))
        q = _env_applies(native)
        assert torch.all(q <= high + 1e-6) and torch.all(q >= low - 1e-6)
    assert torch.allclose(q, high, atol=1e-6)
