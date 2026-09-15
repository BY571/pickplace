"""Build a FoodCellEnvCfg from a plain mapping (e.g. the ``env:`` section of a Hydra config)."""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping

ARMS = {"franka": "food_robot.arms.franka:FRANKA_CFG"}
FOODS = {"rigid": "food_robot.food.rigid:RigidFoodCfg"}

# One-shot reward terms already encode their bonus/penalty in `weight` (scaled to cancel Isaac Lab's
# step_dt multiplication, see FoodCellEnvCfg._build_food_terms) and are configured through the
# dedicated success_bonus / bowl_failure_penalty / food_drop_penalty fields instead.
ONE_SHOT_REWARD_TERMS = {"place_success", "bowl_failure", "food_dropped"}

DEFAULT_ENV: dict = {
    "task": "FoodRobot-Cell-v0",
    "num_envs": 16,
    "arm": "franka",
    "food": "rigid",
    "action_mode": "ee_delta_pose",
    "cameras": True,
    "image_size": [128, 128],
    "privileged_information": False,
    "belt": {},
    "seed": 0,
    "device": "cuda:0",
    "success_bonus": 150.0,
    "bowl_failure_penalty": 150.0,
    "food_drop_penalty": 150.0,
    "success_settle_steps": 5,
    "ingredient_bowl_pos": [0.45, -0.10, 0.0],
    "ingredient_bowl_x_range": [0.35, 0.55],
    "ingredient_bowl_y_range": [-0.20, 0.00],
    "overview_cam_eye": [1.5, 0.1, 1.0],
    "overview_cam_target": [0.3, 0.1, 0.3],
    "render_camera": False,
    "render_cam_eye": [2.3, -1.9, 1.9],
    "render_cam_target": [0.3, 0.0, 0.35],
    "render_image_size": [720, 1280],
    "rewards": {},
    "food_params": {},
}


def _resolve(registry: dict[str, str], name: str, kind: str):
    if name not in registry:
        raise KeyError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}")
    module, attr = registry[name].split(":")
    return getattr(importlib.import_module(module), attr)


def _tuples(d: Mapping) -> dict:
    return {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in d.items()}


def _geometry_kwargs(belt_kwargs: dict) -> dict:
    """Turn nested ``belt.bowl`` / ``belt.pallet`` dicts into geometry cfg instances."""
    from food_robot.assets.usd_builders import BowlGeometry, PalletGeometry

    out = dict(belt_kwargs)
    if "bowl" in out and isinstance(out["bowl"], Mapping):
        out["bowl"] = BowlGeometry(**_tuples(out["bowl"]))
    if "pallet" in out and isinstance(out["pallet"], Mapping):
        out["pallet"] = PalletGeometry(**_tuples(out["pallet"]))
    return out


def _apply_rewards(cfg, rewards: Mapping) -> None:
    if not rewards:
        return
    valid = {f.name for f in dataclasses.fields(cfg.rewards)}
    for name, weight in rewards.items():
        if name in ONE_SHOT_REWARD_TERMS:
            raise KeyError(
                f"Reward term {name!r} is a one-shot bonus/penalty term; its weight already encodes "
                "the bonus/penalty via the success_bonus/bowl_failure_penalty/food_drop_penalty env "
                "config fields, not via `rewards`."
            )
        if name not in valid:
            raise KeyError(f"Unknown reward term {name!r}. Valid terms: {sorted(valid)}")
        getattr(cfg.rewards, name).weight = float(weight)


def build_cell_env_cfg(env_cfg: Mapping):
    """Merge ``env_cfg`` over ``DEFAULT_ENV`` and construct the Isaac Lab env config."""
    from food_robot.belt import BeltCfg
    from food_robot.envs.cell_env_cfg import FoodCellEnvCfg

    unknown = set(env_cfg) - set(DEFAULT_ENV)
    if unknown:
        raise ValueError(f"Unknown env config keys {sorted(unknown)}. Allowed: {sorted(DEFAULT_ENV)}")
    c = {**DEFAULT_ENV, **env_cfg}
    arm = _resolve(ARMS, c["arm"], "arm")
    food_cls = _resolve(FOODS, c["food"], "food")
    food = food_cls(**_tuples(c["food_params"]))

    cfg = FoodCellEnvCfg(
        arm=arm,
        food=food,
        belt=BeltCfg(**_geometry_kwargs(_tuples(c["belt"]))),
        action_mode=c["action_mode"],
        cameras=bool(c["cameras"]),
        image_size=tuple(c["image_size"]),
        privileged_information=bool(c["privileged_information"]),
        success_bonus=float(c["success_bonus"]),
        bowl_failure_penalty=float(c["bowl_failure_penalty"]),
        food_drop_penalty=float(c["food_drop_penalty"]),
        success_settle_steps=int(c["success_settle_steps"]),
        ingredient_bowl_pos=tuple(c["ingredient_bowl_pos"]),
        ingredient_bowl_x_range=tuple(c["ingredient_bowl_x_range"]),
        ingredient_bowl_y_range=tuple(c["ingredient_bowl_y_range"]),
        overview_cam_eye=tuple(c["overview_cam_eye"]),
        overview_cam_target=tuple(c["overview_cam_target"]),
        render_camera=bool(c["render_camera"]),
        render_cam_eye=tuple(c["render_cam_eye"]),
        render_cam_target=tuple(c["render_cam_target"]),
        render_image_size=tuple(c["render_image_size"]),
    )
    cfg.scene.num_envs = int(c["num_envs"])
    cfg.seed = c["seed"]
    cfg.sim.device = c["device"]
    _apply_rewards(cfg, c["rewards"])
    return cfg
