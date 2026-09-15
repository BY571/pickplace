import pytest

pytestmark = pytest.mark.isaac


def test_supply_bowl_food_and_pallet_start_are_randomized_within_range(run_scenario):
    r = run_scenario("dr_behaviour")
    xs = [p[0] for p in r["supply_bowl_xy"]]
    ys = [p[1] for p in r["supply_bowl_xy"]]
    assert all(r["x_range"][0] - 1e-3 <= x <= r["x_range"][1] + 1e-3 for x in xs)
    assert all(r["y_range"][0] - 1e-3 <= y <= r["y_range"][1] + 1e-3 for y in ys)
    assert max(xs) - min(xs) > 0.5 * (r["x_range"][1] - r["x_range"][0])  # actually randomized
    assert max(ys) - min(ys) > 0.5 * (r["y_range"][1] - r["y_range"][0])
    # food spawns inside the bowl it was reset into, not around the old fixed position
    assert all(abs(dx) <= r["spawn_range"] + 5e-3 and abs(dy) <= r["spawn_range"] + 5e-3 for dx, dy in r["food_offset_xy"])
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
