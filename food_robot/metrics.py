"""Simulator-free training metrics. Imports only torch."""

from __future__ import annotations

from collections.abc import Mapping

import torch

OUTCOME_TERMS: tuple[str, ...] = (
    "success",
    "bowl_exited_zone",
    "bowl_off_belt",
    "bowl_tipped",
    "food_off_table",
    "time_out",
)


def outcome_rates(done: torch.Tensor, outcomes: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Fraction of finished episodes that ended with each outcome.

    Args:
        done: boolean tensor marking the last step of each finished episode, any shape.
        outcomes: outcome name -> boolean tensor of the same shape as ``done``.

    Returns:
        name -> (outcome & done).sum() / done.sum(); an empty dict if no episode finished.
    """
    finished = done.reshape(-1).bool()
    n = int(finished.sum())
    if n == 0:
        return {}
    return {name: float((flags.reshape(-1).bool() & finished).sum()) / n for name, flags in outcomes.items()}
