import torch

from food_robot.metrics import OUTCOME_TERMS, outcome_rates


def test_outcome_terms_cover_all_terminations():
    assert set(OUTCOME_TERMS) == {"success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out"}


def test_rates_are_fractions_of_finished_episodes():
    done = torch.tensor([[True], [False], [True], [True], [False]])
    outcomes = {
        "success": torch.tensor([[True], [False], [False], [True], [False]]),
        "bowl_exited_zone": torch.tensor([[False], [False], [True], [False], [False]]),
    }
    rates = outcome_rates(done, outcomes)
    assert rates["success"] == 2 / 3
    assert rates["bowl_exited_zone"] == 1 / 3


def test_flags_on_rows_that_are_not_done_are_ignored():
    done = torch.tensor([[True], [False]])
    rates = outcome_rates(done, {"success": torch.tensor([[False], [True]])})
    assert rates["success"] == 0.0


def test_no_finished_episode_returns_empty_dict():
    done = torch.zeros(4, 3, 1, dtype=torch.bool)
    assert outcome_rates(done, {"success": torch.ones(4, 3, 1, dtype=torch.bool)}) == {}


def test_batched_time_dimension_is_flattened():
    done = torch.tensor([[[True], [False]], [[True], [True]]])  # (envs=2, time=2, 1)
    success = torch.tensor([[[True], [False]], [[False], [True]]])
    assert outcome_rates(done, {"success": success})["success"] == 2 / 3
