import pytest

pytestmark = pytest.mark.isaac


def test_bowl_timing_push_and_tip(run_scenario):
    r = run_scenario("belt_behaviour", "events")
    first = r["first_termination"]  # env -> [term_name, time_s]
    # env 0 was pushed off the belt, env 1 was tipped over; the termination must
    # land inside [disturb_time, disturb_time + 0.1], not merely before it
    assert first["0"][0] == "bowl_off_belt"
    assert r["disturb_time"] <= first["0"][1] < r["disturb_time"] + 0.1
    assert first["1"][0] == "bowl_tipped"
    assert r["disturb_time"] <= first["1"][1] < r["disturb_time"] + 0.1
    # undisturbed bowls leave the zone after entry_margin/speed + place_window
    expected_exit = (r["entry_margin"] + r["zone_length"]) / r["speed"]
    for env in ("2", "3"):
        assert first[env][0] == "bowl_exited_zone"
        assert first[env][1] == pytest.approx(expected_exit, abs=0.25)


def test_speed_noise_and_bowl_offset_randomization(run_scenario):
    r = run_scenario("belt_behaviour", "randomization")
    speeds, nominal, noise = r["pallet_speeds"], r["speed"], r["speed_noise"]
    assert all(nominal * (1 - noise) - 1e-3 <= s <= nominal * (1 + noise) + 1e-3 for s in speeds)
    assert max(speeds) - min(speeds) > 0.2 * nominal * noise  # actually randomized
    offsets = r["bowl_offset_x"]
    assert all(-0.03 - 5e-3 <= o <= 0.03 + 5e-3 for o in offsets)
    assert max(offsets) - min(offsets) > 0.02
