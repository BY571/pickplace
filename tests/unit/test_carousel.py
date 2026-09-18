import pytest

from food_robot.carousel import (
    DemoTally,
    carousel_layout,
    park_position,
    pick_target,
)

# the training belt: entry 0.20, zone 0.25..0.65, belt end 1.00, 0.22 m pallets, travel limit 1.0
BELT = dict(entry_x=0.20, belt_end_x=1.00, zone_end_x=0.65, pallet_length=0.22, bowl_outer_radius=0.076,
            travel_upper=1.0)


def test_default_spacing_fills_the_belt_evenly():
    layout = carousel_layout(bowls=3, **BELT)
    assert layout.spacing == pytest.approx(0.8 / 3)
    assert layout.recycle_q == pytest.approx(0.8)
    assert layout.start_q == pytest.approx((0.0, 0.8 / 3, 1.6 / 3))


def test_explicit_spacing_sets_the_recycle_point():
    layout = carousel_layout(bowls=2, spacing=0.3, **BELT)
    assert layout.recycle_q == pytest.approx(0.6)
    assert layout.start_q == pytest.approx((0.0, 0.3))


def test_single_bowl_uses_the_whole_belt():
    layout = carousel_layout(bowls=1, **BELT)
    assert layout.recycle_q == pytest.approx(0.8) and layout.start_q == (0.0,)


@pytest.mark.parametrize(
    "bowls,spacing,match",
    [
        (4, None, "pallet"),        # 0.2 m < 0.22 m pallet + gap
        (3, 0.23, "pallet"),        # bowls would touch
        (3, 0.35, "belt"),          # recycle point 1.05 past the belt end / joint limit
        (2, 0.24, "zone"),          # recycle at 0.48 -> x 0.68: the bowl has not left the zone
        (0, None, "bowls"),
    ],
)
def test_layout_rejects_impossible_carousels(bowls, spacing, match):
    with pytest.raises(ValueError, match=match):
        carousel_layout(bowls=bowls, spacing=spacing, **BELT)


def test_pick_target_is_the_most_downstream_open_bowl_still_in_reach():
    xs = [0.70, 0.40, 0.15]
    assert pick_target(xs, [True, True, True], zone_end_x=0.65) == 1  # 0.70 already left the zone
    assert pick_target(xs, [False, True, True], zone_end_x=0.65) == 1
    assert pick_target(xs, [True, False, True], zone_end_x=0.65) == 2  # upstream bowl approaching counts
    assert pick_target(xs, [True, False, False], zone_end_x=0.65) is None


def test_park_positions_are_distinct_and_below_the_table():
    spots = [park_position(j, ground_z=-1.05, item_radius=0.02) for j in range(20)]
    assert len(set(spots)) == 20
    for x, y, z in spots:
        assert -0.30 < x < 1.20 and -0.55 < y < 0.55  # under the table footprint
        assert -1.05 < z < -0.80


def test_tally_summary_counts_and_rate():
    tally = DemoTally()
    for _ in range(3):
        tally.bowl_entered()
    tally.placed += 1
    tally.missed += 1
    tally.dropped += 2
    summary = tally.summary(seconds=30.0)
    assert summary == {
        "placed": 1, "missed": 1, "dropped": 2, "misplaced": 0, "bowls_seen": 3, "pending": 1,
        "seconds": 30.0, "placements_per_min": 2.0,
    }


def test_tally_rate_is_zero_before_any_time_passed():
    assert DemoTally().summary(seconds=0.0)["placements_per_min"] == 0.0
