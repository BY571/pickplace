import pytest
import torch

from pickplace import action_adapters as A

SCALE = 0.5  # ArmCfg.joint_action_scale for every arm shipped here
DEFAULT = torch.tensor([[0.0, -0.57, 0.0, -2.81, 0.0, 3.04, 0.74]])


def _env_applies(action: torch.Tensor, scale: float = SCALE) -> torch.Tensor:
    """What Isaac Lab's JointPositionActionCfg(use_default_offset=True) does with an action."""
    return DEFAULT + scale * action


def _velocity_to_native(q_dot, q, q_default=DEFAULT, scale=SCALE, dt=0.02, limits=None):
    """One step of the transform's own path, seeded from the measurement (the just-reset case)."""
    target = A.integrate_velocity(q_dot, None, q, dt, limits)
    return A.joint_target_to_native(target, q_default, scale, limits)


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
    native = _velocity_to_native(q_dot, q, dt=dt)
    assert torch.allclose(_env_applies(native), q + q_dot * dt, atol=1e-6)


def test_zero_velocity_holds_the_current_pose_wherever_it_is():
    # the property that makes velocity a safe default: the no-op does not depend on the default pose
    for offset in (-0.4, 0.0, 0.9):
        q = DEFAULT + offset
        native = _velocity_to_native(torch.zeros_like(q), q)
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


def test_the_integrator_seeded_from_the_measurement_matches_the_one_shot_algebra():
    q, dt = DEFAULT + 0.2, 0.02
    q_dot = torch.tensor([[0.3, 0.0, -0.3, 0.1, 0.0, 0.2, -0.1]])
    from_velocity = _velocity_to_native(q_dot, q, dt=dt)
    from_position = A.joint_target_to_native(q + q_dot * dt, DEFAULT, SCALE)
    assert torch.allclose(from_velocity, from_position, atol=1e-7)


def test_a_held_target_ignores_a_sagging_measurement():
    """The regression test for the 0.26 rad drift: a zero command must not follow the arm down.

    Anchoring the integral to the measurement made the target chase a sagging arm. Here the measurement
    droops 0.01 rad every step while the command stays zero; the carried target must not move at all.
    """
    target = DEFAULT.clone()
    measured = DEFAULT.clone()
    for _ in range(20):
        measured = measured - 0.01
        target = A.integrate_velocity(torch.zeros_like(target), target, measured, dt=0.02)
    assert torch.allclose(target, DEFAULT, atol=1e-7), "the commanded target followed the sag"


def test_a_stale_row_is_reseeded_from_the_measurement():
    """What the transform does for a row whose episode ended: pass q_target=None for that row."""
    carried = DEFAULT + 0.5
    measured = DEFAULT - 0.2
    assert torch.allclose(A.integrate_velocity(torch.zeros_like(carried), None, measured, 0.02), measured)
    assert torch.allclose(A.integrate_velocity(torch.zeros_like(carried), carried, measured, 0.02), carried)


def test_unknown_adapter_is_rejected_by_name():
    with pytest.raises(ValueError, match="joint_velocity"):
        A.make_joint_action_adapter("joint_velocities")  # plural: a plausible typo


def test_native_is_not_a_transform_to_build():
    # "native" means no adapter at all; building one silently produced a *position* adapter before
    with pytest.raises(ValueError, match="native"):
        A.make_joint_action_adapter("native")


@pytest.mark.parametrize("mode", A.JOINT_ADAPTERS)
def test_each_adapter_builds_and_knows_its_mode(mode):
    assert A.make_joint_action_adapter(mode).mode == mode


@pytest.mark.parametrize("mode", A.JOINT_ADAPTERS)
def test_every_adapted_space_advertises_a_normalized_command(mode):
    """All adapters take [-1, 1] per arm dimension and [0, 1] for the gripper; `scale` carries the units.

    That is what lets one policy architecture drive every space, and what makes the scale an experimental
    variable rather than a constant buried in the arm config.
    """
    from torchrl.data import Composite, Unbounded

    adapter = A.make_joint_action_adapter(mode)
    spec = Composite(
        full_action_spec=Composite(action=Unbounded(shape=(2, 8)), shape=(2,)),
        other_key=Unbounded(shape=(2, 3)),  # a sibling the adapter must not drop
        shape=(2,),
    )
    out = adapter.transform_input_spec(spec)
    action = out["full_action_spec", "action"]
    assert "other_key" in out.keys(), "rebuilding the composite would drop siblings"
    assert action.space.low[0, 0] == -1.0 and action.space.high[0, 0] == 1.0   # arm
    assert action.space.low[0, -1] == 0.0 and action.space.high[0, -1] == 1.0  # gripper fraction


def test_each_space_scales_a_unit_action_into_its_own_units():
    """The defaults are anchored to the Franka: 2.17 rad/s joint limit, 0.02 s control period."""
    assert A.DEFAULT_SCALES["joint_velocity"] == 2.0                      # rad/s, near the 2.17 limit
    assert A.DEFAULT_SCALES["joint_delta"] * 50 == A.DEFAULT_SCALES["joint_velocity"]  # same speed at 50 Hz
    assert A.make_joint_action_adapter("joint_delta", scale=0.01).scale == 0.01
    assert A.make_joint_action_adapter("joint_delta").scale == A.DEFAULT_SCALES["joint_delta"]


def test_the_default_env_keeps_todays_action_space():
    from pickplace.config import DEFAULT_ENV

    assert DEFAULT_ENV["action_adapter"] == "native"
    assert DEFAULT_ENV["action_mode"] == "ee_delta_pose"


def test_a_full_arm_sweep_stays_inside_the_limits_it_was_given():
    low, high = DEFAULT - 0.5, DEFAULT + 0.5
    q = DEFAULT.clone()
    q_dot = torch.full_like(q, 10.0)  # far faster than any joint can move
    for _ in range(50):
        native = _velocity_to_native(q_dot, q, limits=(low, high))
        q = _env_applies(native)
        assert torch.all(q <= high + 1e-6) and torch.all(q >= low - 1e-6)
    assert torch.allclose(q, high, atol=1e-6)
