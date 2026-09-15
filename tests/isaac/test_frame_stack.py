import pytest

pytestmark = pytest.mark.isaac


def test_stacked_pixels_shape_history_and_reset(run_scenario):
    r = run_scenario("frame_stack")
    assert r["wrist_shape"] == [64, 64, 9]
    assert r["overview_shape"] == [64, 64, 9]
    assert r["dtype"] == "torch.float32"
    assert 0.0 <= r["min_value"] and r["max_value"] <= 255.0
    # the bowl moves on the belt, so after several steps the oldest and newest overview frames differ
    assert r["oldest_vs_newest_diff_after_steps"] > 0.5
    # right after a reset every history slot holds the current frame
    assert r["oldest_vs_newest_diff_after_reset"] == pytest.approx(0.0, abs=1e-6)
    # frame_stack=1 keeps the old single-frame term (and therefore the old H x W x 3 spec)
    assert r["single_frame_term"] == "image_float"
