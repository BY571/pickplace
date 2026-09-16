import pytest

pytestmark = pytest.mark.isaac


def test_supply_bowl_is_fixed_and_food_and_pallet_start_are_randomized(run_scenario):
    r = run_scenario("dr_behaviour")
    xs = [p[0] for p in r["supply_bowl_xy"]]
    ys = [p[1] for p in r["supply_bowl_xy"]]
    # task 14: the container is fixed (ingredient_bowl_x_range/y_range are degenerate by default); the food
    # is randomized inside it instead, over a much wider range than the old bowl-position DR covered.
    assert r["x_range"][0] == r["x_range"][1] and r["y_range"][0] == r["y_range"][1]
    assert all(x == pytest.approx(r["x_range"][0], abs=1e-4) for x in xs)
    assert all(y == pytest.approx(r["y_range"][0], abs=1e-4) for y in ys)
    # food spawns inside the (fixed) bowl, within spawn_range, but its spread is a real randomization --
    # exceeding half the sampling range, the same bar the old bowl-position DR test held itself to
    dxs = [dx for dx, _ in r["food_offset_xy"]]
    dys = [dy for _, dy in r["food_offset_xy"]]
    assert all(abs(dx) <= r["spawn_range"] + 5e-3 and abs(dy) <= r["spawn_range"] + 5e-3 for dx, dy in r["food_offset_xy"])
    assert max(dxs) - min(dxs) > r["spawn_range"]
    assert max(dys) - min(dys) > r["spawn_range"]
    starts = r["pallet_start"]
    assert all(r["start_range"][0] - 1e-4 <= s <= r["start_range"][1] + 1e-4 for s in starts)
    assert max(starts) - min(starts) > 0.5 * (r["start_range"][1] - r["start_range"][0])
    # the bowl sits on the pallet where the start and offset say
    assert max(r["bowl_x_minus_expected"]) < 5e-3
    assert all(s == pytest.approx(r["speed"], abs=1e-3) for s in r["pallet_speeds"])  # speed_noise = 0


def test_invalid_dr_configs_are_rejected(run_scenario):
    r = run_scenario("dr_behaviour", "--errors")
    assert r["errors"] == {
        "supply_out_of_reach": "ValueError",
        "bowl_overhangs_pallet": "ValueError",
        "start_below_travel": "ValueError",
    }
