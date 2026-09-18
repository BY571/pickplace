import pytest

pytestmark = pytest.mark.isaac


def test_physx_buffers_scale_with_num_envs(run_scenario):
    r = run_scenario("physx_scaling")
    small, large = r["n16"], r["n16384"]
    # never below the values validated at 4096 envs
    assert small["gpu_total_aggregate_pairs_capacity"] == 64 * 1024
    assert small["gpu_found_lost_aggregate_pairs_capacity"] == 1024 * 1024 * 4
    # linear in num_envs above 4096 (x4 at 16384), for every scaled buffer
    for name in r["scaled_fields"]:
        assert large[name] >= 4 * small[name], name
