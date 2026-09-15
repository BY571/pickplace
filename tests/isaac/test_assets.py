import pytest

pytestmark = pytest.mark.isaac


def test_pallet_carries_free_bowl_at_commanded_speed(run_scenario):
    r = run_scenario("pallet_bowl_spike")
    for i, speed in enumerate(r["speeds"]):
        expected = speed * r["duration"]
        assert r["plate_dx"][i] == pytest.approx(expected, rel=0.05)
        assert r["bowl_dx"][i] == pytest.approx(expected, rel=0.10)
        assert r["bowl_z_drop"][i] < 0.01  # bowl still resting on the plate
        assert abs(r["base_dz"][i]) < 1e-3  # fixed base did not move
