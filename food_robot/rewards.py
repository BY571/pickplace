"""Reward-term vocabulary and named reward sets. Simulator-free.

The environment exposes every term below, unweighted, as the vector ``("next", "reward_terms")`` in
``REWARD_TERMS`` order; a reward set is a weight per term, applied with TorchRL's ``LineariseRewards``.
Dense components are ``term value x dt`` (weights mean reward per second held); event components are 0/1
on the step an episode ends that way (weights are the one-shot bonus/penalty).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

DENSE_TERMS: tuple[str, ...] = (
    "reach_food",
    "grasp",
    "grasp_lift",
    "transport",
    "transport_fine",
    "bowl_disturbance",
    "action_rate",
    "joint_vel",
    "food_in_bowl",
)
EVENT_TERMS: tuple[str, ...] = ("success", "bowl_failure", "food_dropped")
REWARD_TERMS: tuple[str, ...] = DENSE_TERMS + EVENT_TERMS

EVENT_SOURCES: dict[str, tuple[str, ...]] = {
    "success": ("success",),
    "bowl_failure": ("bowl_off_belt", "bowl_tipped"),
    "food_dropped": ("food_off_table",),
}

REWARD_SETS_DIR = Path(__file__).parent / "reward_sets"
DEFAULT_REWARD_SET = "staged_v1"

# Pre-reward-set env keys, kept so existing configs (runs 1-3) resolve to the same reward.
_LEGACY_EVENT_FIELDS = {
    "success_bonus": ("success", 1.0),
    "bowl_failure_penalty": ("bowl_failure", -1.0),
    "food_drop_penalty": ("food_dropped", -1.0),
}


def available_reward_sets() -> list[str]:
    return sorted(p.stem for p in REWARD_SETS_DIR.glob("*.yaml"))


def _complete(weights: Mapping, source: str) -> dict[str, float]:
    unknown = sorted(set(weights) - set(REWARD_TERMS))
    if unknown:
        raise KeyError(f"Unknown reward terms {unknown} in {source}. Valid terms: {list(REWARD_TERMS)}")
    return {term: float(weights.get(term, 0.0)) for term in REWARD_TERMS}


def load_reward_set(name_or_path: str) -> dict[str, float]:
    """Load a reward set by name (``food_robot/reward_sets/<name>.yaml``) or by file path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = REWARD_SETS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Reward set {name_or_path!r} not found. Available: {available_reward_sets()}")
    return _complete(yaml.safe_load(path.read_text()) or {}, source=str(path))


def resolve_reward_weights(env_cfg: Mapping) -> dict[str, float]:
    """Final per-term weights for an env config.

    Precedence (later wins): the reward set ``reward_set`` (default ``staged_v1``); legacy ``rewards``
    (dense terms only); legacy ``success_bonus`` / ``bowl_failure_penalty`` / ``food_drop_penalty`` when not
    None (penalties become negative weights); ``reward_weights``.
    """
    weights = load_reward_set(env_cfg.get("reward_set") or DEFAULT_REWARD_SET)
    for name, weight in (env_cfg.get("rewards") or {}).items():
        if name not in DENSE_TERMS:
            raise KeyError(
                f"`rewards` only overrides dense terms {list(DENSE_TERMS)}; got {name!r}. Set one-shot terms "
                "with `reward_weights` (e.g. reward_weights: {success: 150.0})."
            )
        weights[name] = float(weight)
    for field, (term, sign) in _LEGACY_EVENT_FIELDS.items():
        if env_cfg.get(field) is not None:
            weights[term] = sign * float(env_cfg[field])
    overrides = env_cfg.get("reward_weights") or {}
    _complete(overrides, source="reward_weights")  # validates names
    weights.update({name: float(w) for name, w in overrides.items()})
    return weights


def weight_vector(weights: Mapping[str, float]) -> list[float]:
    return [float(weights[term]) for term in REWARD_TERMS]


def isaac_weight(weight: float) -> float:
    """Isaac Lab skips zero-weight terms entirely, which would erase that component from the vector."""
    return weight if weight != 0.0 else 1.0
