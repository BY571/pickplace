"""Training helpers shared by pipeline stages. Imports torch/tensordict only (no simulator)."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase

from pickplace.metrics import OUTCOME_TERMS, outcome_rates
from pickplace.rewards import REWARD_TERMS


def _metrics(data: TensorDictBase, done: torch.Tensor, prefix: str) -> dict[str, float]:
    if not bool(done.any()):
        return {}
    completed_return = data["episode_reward"] + data["next", "reward"]
    completed_length = data["step_count"] + 1
    metrics = {
        f"{prefix}/episode_return": completed_return[done].mean().item(),
        f"{prefix}/episode_length": completed_length[done].float().mean().item(),
        f"{prefix}/episodes": int(done.sum()),
    }
    outcomes = {term: data["next", "outcome", term] for term in OUTCOME_TERMS}
    metrics.update({f"{prefix}/{term}_rate": rate for term, rate in outcome_rates(done, outcomes).items()})
    if "episode_reward_terms" in data.keys() and ("next", "reward_terms") in data.keys(True):
        terms = (data["episode_reward_terms"] + data["next", "reward_terms"])[done.squeeze(-1)]
        means = terms.mean(dim=0)
        metrics.update({f"{prefix}/terms/{name}": means[k].item() for k, name in enumerate(REWARD_TERMS)})
    return metrics


def episode_metrics(data: TensorDictBase, prefix: str) -> dict[str, float]:
    """Return, length, outcome rates and per-term sums over every episode that finished in ``data`` (N x T).

    Under native auto-reset the running sums are reset on the done row itself, so a finished episode's totals
    are the pre-step running sum plus this step's value (``episode_reward + reward``).
    """
    return _metrics(data, data["next", "done"], prefix)


def first_episode_metrics(data: TensorDictBase, prefix: str) -> dict[str, float]:
    """As ``episode_metrics`` but only each env's first finished episode (evaluation after a reset)."""
    done = data["next", "done"]
    first = done & (done.long().cumsum(dim=1) == 1)
    metrics = _metrics(data, first, prefix)
    if metrics:
        metrics[f"{prefix}/finished_fraction"] = first.any(dim=1).float().mean().item()
    return metrics


class SuccessStreak:
    """Consecutive updates whose success rate reached ``threshold`` (early stopping)."""

    def __init__(self, threshold: float, required: int):
        self.threshold, self.required, self.count = threshold, int(required), 0

    def update(self, success_rate: float | None) -> bool:
        if success_rate is not None:
            self.count = self.count + 1 if success_rate >= self.threshold else 0
        return self.count >= self.required
