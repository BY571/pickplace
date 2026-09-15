import math

import pytest

from food_robot.timing import (
    BeltZone,
    belt_zone,
    look_at_quat_xyzw,
    quat_rotate_xyzw,
    validate_observation_flags,
    validate_zone_reachable,
)


def test_belt_zone_length_is_speed_times_window():
    zone = belt_zone(speed=0.08, place_window=5.0, zone_center_x=0.45)
    assert zone.length == pytest.approx(0.40)
    assert zone.start_x == pytest.approx(0.25)
    assert zone.end_x == pytest.approx(0.65)


@pytest.mark.parametrize("speed,window", [(0.0, 5.0), (-0.1, 5.0), (0.08, 0.0)])
def test_belt_zone_rejects_non_positive_inputs(speed, window):
    with pytest.raises(ValueError):
        belt_zone(speed=speed, place_window=window, zone_center_x=0.45)


def test_reachable_zone_passes():
    validate_zone_reachable(BeltZone(0.25, 0.65), belt_y=0.30, base_xy=(0.0, 0.0), reach_radius=0.80)


def test_unreachable_zone_raises_with_explanation():
    with pytest.raises(ValueError, match="reach_radius"):
        validate_zone_reachable(BeltZone(0.0, 1.2), belt_y=0.30, base_xy=(0.0, 0.0), reach_radius=0.80)


def test_observation_flags_require_some_food_information():
    validate_observation_flags(cameras=True, privileged_information=False)
    validate_observation_flags(cameras=False, privileged_information=True)
    with pytest.raises(ValueError, match="privileged_information"):
        validate_observation_flags(cameras=False, privileged_information=False)


@pytest.mark.parametrize(
    "eye,target",
    [((0, 0, 1), (1, 0, 0)), ((1.4, 0.0, 0.9), (0.45, 0.1, 0.0)), ((-1, 2, 0.5), (0, 0, 0)), ((0, 0, 0), (0, 3, 0))],
)
def test_look_at_points_forward_axis_at_target_without_roll(eye, target):
    q = look_at_quat_xyzw(eye, target)
    assert sum(c * c for c in q) == pytest.approx(1.0)
    d = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in d))
    forward = quat_rotate_xyzw(q, (1.0, 0.0, 0.0))
    assert forward == pytest.approx(tuple(c / n for c in d), abs=1e-9)
    right = quat_rotate_xyzw(q, (0.0, 1.0, 0.0))
    assert right[2] == pytest.approx(0.0, abs=1e-9)  # horizon stays level


def test_quat_rotate_identity():
    assert quat_rotate_xyzw((0.0, 0.0, 0.0, 1.0), (0.1, -0.2, 0.3)) == pytest.approx((0.1, -0.2, 0.3))
