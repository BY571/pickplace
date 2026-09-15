"""Build a FoodCellEnvCfg from a plain mapping (e.g. the ``env:`` section of a Hydra config)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

ARMS = {"franka": "food_robot.arms.franka:FRANKA_CFG"}
FOODS = {"rigid": "food_robot.food.rigid:RigidFoodCfg"}

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
}


def _resolve(registry: dict[str, str], name: str, kind: str):
    if name not in registry:
        raise KeyError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}")
    module, attr = registry[name].split(":")
    return getattr(importlib.import_module(module), attr)


def _tuples(d: Mapping) -> dict:
    return {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in d.items()}


def build_cell_env_cfg(env_cfg: Mapping):
    """Merge ``env_cfg`` over ``DEFAULT_ENV`` and construct the Isaac Lab env config."""
    from food_robot.belt import BeltCfg
    from food_robot.envs.cell_env_cfg import FoodCellEnvCfg

    unknown = set(env_cfg) - set(DEFAULT_ENV)
    if unknown:
        raise ValueError(f"Unknown env config keys {sorted(unknown)}. Allowed: {sorted(DEFAULT_ENV)}")
    c = {**DEFAULT_ENV, **env_cfg}
    arm = _resolve(ARMS, c["arm"], "arm")
    food = _resolve(FOODS, c["food"], "food")()
    cfg = FoodCellEnvCfg(
        arm=arm,
        food=food,
        belt=BeltCfg(**_tuples(c["belt"])),
        action_mode=c["action_mode"],
        cameras=bool(c["cameras"]),
        image_size=tuple(c["image_size"]),
        privileged_information=bool(c["privileged_information"]),
    )
    cfg.scene.num_envs = int(c["num_envs"])
    cfg.seed = c["seed"]
    cfg.sim.device = c["device"]
    return cfg
