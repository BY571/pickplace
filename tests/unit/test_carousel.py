import pytest

from food_robot.carousel import (
    DemoTally,
    carousel_layout,
    park_position,
    pick_target,
)

# the training belt: entry 0.20, zone 0.25..0.65 (0.40 m), belt end 1.00, earliest pallet start -0.08, 0.22 m pallets
BELT = dict(entry_x=0.20, belt_end_x=1.00, zone_length=0.40, zone_end_x=0.65, earliest_start_q=-0.08,
            pallet_length=0.22, bowl_outer_radius=0.076)


def test_default_spacing_is_one_place_window_per_bowl_extending_the_belt_upstream():
    layout = carousel_layout(bowls=3, **BELT)
    assert layout.spacing == pytest.approx(0.40)
    assert layout.recycle_q == pytest.approx(0.80)  # the belt end
    assert layout.entry_q == pytest.approx(-0.40)  # 0.4 m upstream of the training entry
    assert layout.start_q == pytest.approx((-0.40, 0.0, 0.40))


def test_single_bowl_enters_where_a_training_episode_can_start_it():
    layout = carousel_layout(bowls=1, **BELT)
    assert layout.spacing == pytest.approx(0.88)
    assert layout.entry_q == pytest.approx(-0.08) and layout.start_q == pytest.approx((-0.08,))


def test_two_bowls_need_the_whole_training_path():
    layout = carousel_layout(bowls=2, **BELT)
    assert layout.spacing == pytest.approx(0.44) and layout.entry_q == pytest.approx(-0.08)


def test_explicit_spacing():
    layout = carousel_layout(bowls=3, spacing=0.5, **BELT)
    assert layout.entry_q == pytest.approx(-0.70)
    assert layout.start_q == pytest.approx((-0.70, -0.20, 0.30))


@pytest.mark.parametrize(
    "bowls,spacing,match",
    [
        (5, 0.23, "pallet"),        # pallets would touch
        (3, 0.25, "upstream"),      # 0.75 m loop: bowls would enter after where training starts them
        (0, None, "bowls"),
    ],
)
def test_layout_rejects_impossible_carousels(bowls, spacing, match):
    with pytest.raises(ValueError, match=match):
        carousel_layout(bowls=bowls, spacing=spacing, **BELT)


def test_layout_rejects_a_belt_ending_inside_the_zone():
    with pytest.raises(ValueError, match="zone"):
        carousel_layout(bowls=3, **{**BELT, "belt_end_x": 0.70})


def test_pick_target_is_the_most_downstream_open_bowl_between_the_training_start_and_the_zone_end():
    xs = [0.70, 0.40, 0.15, -0.20]
    everything = [True] * 4
    assert pick_target(xs, everything, min_x=0.10, zone_end_x=0.65) == 1  # 0.70 already left the zone
    assert pick_target(xs, [False, False, True, True], min_x=0.10, zone_end_x=0.65) == 2  # approaching counts
    assert pick_target(xs, [False, False, False, True], min_x=0.10, zone_end_x=0.65) is None  # still queued
    assert pick_target(xs, [False] * 4, min_x=0.10, zone_end_x=0.65) is None


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
