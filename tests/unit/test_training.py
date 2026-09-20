import pytest
import torch
from tensordict import TensorDict

from pickplace.metrics import OUTCOME_TERMS
from pickplace.rewards import REWARD_TERMS
from pickplace.training import SuccessStreak, episode_metrics, first_episode_metrics

K = len(REWARD_TERMS)


def rollout(done, success, reward, episode_reward, step_count, terms, episode_terms):
    t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt).unsqueeze(-1)  # noqa: E731
    outcome = {o: torch.zeros_like(t(done, torch.bool)) for o in OUTCOME_TERMS}
    outcome["success"] = t(success, torch.bool)
    outcome["bowl_exited_zone"] = t(done, torch.bool) & ~t(success, torch.bool)
    return TensorDict(
        {
            "episode_reward": t(episode_reward),
            "step_count": t(step_count, torch.int64),
            "episode_reward_terms": torch.tensor(episode_terms, dtype=torch.float32),
            "next": {
                "done": t(done, torch.bool),
                "reward": t(reward),
                "reward_terms": torch.tensor(terms, dtype=torch.float32),
                "outcome": outcome,
            },
        },
        batch_size=[2, 2],
    )


def vec(x):
    v = [0.0] * K
    v[0] = x  # reach_food component
    return v


def test_episode_metrics_include_completed_return_length_rates_and_terms():
    data = rollout(
        done=[[False, True], [True, False]],
        success=[[False, True], [False, False]],
        reward=[[1.0, 150.0], [-150.0, 0.0]],
        episode_reward=[[5.0, 6.0], [1.0, 0.0]],
        step_count=[[3, 4], [9, 0]],
        terms=[[vec(0.1), vec(0.2)], [vec(0.3), vec(0.0)]],
        episode_terms=[[vec(1.0), vec(1.1)], [vec(2.0), vec(0.0)]],
    )
    m = episode_metrics(data, "train")
    assert m["train/episodes"] == 2
    assert m["train/episode_return"] == pytest.approx(((6.0 + 150.0) + (1.0 - 150.0)) / 2)
    assert m["train/episode_length"] == pytest.approx((5 + 10) / 2)
    assert m["train/success_rate"] == 0.5
    assert m["train/terms/reach_food"] == pytest.approx(((1.1 + 0.2) + (2.0 + 0.3)) / 2)
    assert m["train/terms/success"] == 0.0


def test_episode_metrics_empty_without_finished_episodes():
    data = rollout([[False] * 2] * 2, [[False] * 2] * 2, [[0.0] * 2] * 2, [[0.0] * 2] * 2, [[0] * 2] * 2,
                   [[vec(0.0)] * 2] * 2, [[vec(0.0)] * 2] * 2)
    assert episode_metrics(data, "train") == {}


def test_first_episode_metrics_count_each_env_once():
    # env 0 finishes at t=0 and t=1; env 1 never finishes
    data = rollout(
        done=[[True, True], [False, False]],
        success=[[True, False], [False, False]],
        reward=[[150.0, -150.0], [0.0, 0.0]],
        episode_reward=[[1.0, 0.0], [0.0, 0.0]],
        step_count=[[4, 0], [0, 1]],
        terms=[[vec(0.5), vec(0.0)], [vec(0.0), vec(0.0)]],
        episode_terms=[[vec(1.0), vec(0.0)], [vec(0.0), vec(0.0)]],
    )
    m = first_episode_metrics(data, "eval")
    assert m["eval/episodes"] == 1
    assert m["eval/success_rate"] == 1.0
    assert m["eval/episode_return"] == pytest.approx(151.0)
    assert m["eval/finished_fraction"] == 0.5
    assert m["eval/terms/reach_food"] == pytest.approx(1.5)


def test_success_streak():
    s = SuccessStreak(0.85, 2)
    assert not s.update(0.9)
    assert not s.update(None) and s.count == 1
    assert not s.update(0.5) and s.count == 0
    assert not s.update(0.86)
    assert s.update(0.99) and s.count == 2
