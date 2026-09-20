import pytest

from pickplace.timing import (
    validate_bowl_on_pallet,
    validate_pallet_start,
    validate_supply_bowl_range,
    zone_exit_times,
)

BOWL_R = 0.076


def test_default_supply_range_is_valid():
    validate_supply_bowl_range((0.35, 0.55), (-0.20, 0.00), BOWL_R, reach_radius=0.80, belt_y=0.30, belt_half_width=0.15)


def test_supply_range_beyond_reach_raises():
    with pytest.raises(ValueError, match="reach"):
        validate_supply_bowl_range((0.35, 0.80), (-0.20, 0.00), BOWL_R, reach_radius=0.80, belt_y=0.30, belt_half_width=0.15)


def test_supply_range_touching_belt_raises():
    with pytest.raises(ValueError, match="belt"):
        validate_supply_bowl_range((0.35, 0.55), (-0.20, 0.10), BOWL_R, reach_radius=0.80, belt_y=0.30, belt_half_width=0.15)


def test_inverted_supply_range_raises():
    with pytest.raises(ValueError, match="min"):
        validate_supply_bowl_range((0.55, 0.35), (-0.20, 0.00), BOWL_R, reach_radius=0.80, belt_y=0.30, belt_half_width=0.15)


def test_default_bowl_fits_on_widened_pallet():
    validate_bowl_on_pallet((0.22, 0.26), BOWL_R, (-0.02, 0.02), (-0.04, 0.04), belt_half_width=0.15)


@pytest.mark.parametrize(
    "pallet,offset_x,offset_y,match",
    [
        ((0.22, 0.22), (-0.02, 0.02), (-0.04, 0.04), "across"),
        ((0.22, 0.26), (-0.05, 0.05), (-0.04, 0.04), "along"),
        ((0.22, 0.34), (-0.02, 0.02), (-0.04, 0.04), "wider than the belt"),
    ],
)
def test_bowl_or_pallet_not_fitting_raises(pallet, offset_x, offset_y, match):
    with pytest.raises(ValueError, match=match):
        validate_bowl_on_pallet(pallet, BOWL_R, offset_x, offset_y, belt_half_width=0.15)


def test_default_pallet_start_is_valid():
    validate_pallet_start((-0.08, 0.08), travel_lower=-0.12, travel_upper=1.0, zone_length=0.40, entry_margin=0.05)


@pytest.mark.parametrize(
    "start,lower,upper,match",
    [
        ((-0.08, 0.08), -0.05, 1.0, "travel_lower"),
        ((0.08, -0.08), -0.12, 1.0, "min"),
        ((-0.08, 0.08), -0.12, 0.5, "travel_upper"),
    ],
)
def test_invalid_pallet_start_raises(start, lower, upper, match):
    with pytest.raises(ValueError, match=match):
        validate_pallet_start(start, travel_lower=lower, travel_upper=upper, zone_length=0.40, entry_margin=0.05)


def test_zone_exit_times_match_the_design_figure():
    earliest, nominal, latest = zone_exit_times(
        entry_x=0.20, zone_end_x=0.65, speed=0.08, start_range=(-0.08, 0.08), offset_x=(-0.02, 0.02)
    )
    assert earliest == pytest.approx(6.875)
    assert nominal == pytest.approx(5.625)
    assert latest == pytest.approx(4.375)
