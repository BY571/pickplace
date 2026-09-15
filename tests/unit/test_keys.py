import pytest
from torchrl.data import Composite, Unbounded

from food_robot.keys import expand_in_keys


@pytest.fixture
def spec():
    return Composite(
        proprio=Composite(joint_pos_rel=Unbounded(shape=(9,)), ee_pos=Unbounded(shape=(3,))),
        belt=Composite(bowl_pos=Unbounded(shape=(3,))),
        pixels=Composite(wrist_rgb=Unbounded(shape=(64, 64, 3))),
        step_count=Unbounded(shape=(1,)),
    )


def test_group_name_expands_to_sorted_leaves(spec):
    assert expand_in_keys(spec, ["proprio"]) == [("proprio", "ee_pos"), ("proprio", "joint_pos_rel")]


def test_nested_key_and_top_level_leaf_pass_through(spec):
    assert expand_in_keys(spec, [["pixels", "wrist_rgb"], "step_count"]) == [("pixels", "wrist_rgb"), ("step_count",)]


def test_mixed_and_deduplicated_in_order(spec):
    keys = expand_in_keys(spec, ["belt", ["proprio", "ee_pos"], "proprio"])
    assert keys == [("belt", "bowl_pos"), ("proprio", "ee_pos"), ("proprio", "joint_pos_rel")]


def test_unknown_key_lists_available(spec):
    with pytest.raises(KeyError, match="privileged") as err:
        expand_in_keys(spec, ["privileged"])
    assert "proprio" in str(err.value)
