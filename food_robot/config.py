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
    "frame_stack": 1,
    "privileged_information": False,
    "belt": {},
    # Reset randomization added to the arm joints' default pose (reset_joints_by_offset; see
    # FoodCellEnvCfg.robot_reset). Defaults are exactly today's values, so every existing config is unaffected.
    "robot_reset": {"position_range": [-0.02, 0.02], "velocity_range": [0.0, 0.0]},
    "seed": 0,
    "device": "cuda:0",
    # Legacy one-shot keys: None defers to the reward set (see food_robot.rewards.resolve_reward_weights).
    "success_bonus": None,
    "bowl_failure_penalty": None,
    "food_drop_penalty": None,
    "success_settle_steps": 5,
    # Success also requires the TCP back within home_tolerance [m] of its home position (FoodCellEnvCfg).
    "success_requires_home": False,
    "home_tolerance": 0.05,
    "ingredient_bowl_pos": [0.45, -0.10, 0.0],
    # Fixed as of task 14 (was [0.35, 0.55] / [-0.20, 0.00]): the food is randomized inside the wider supply
    # tray instead (RigidFoodCfg.spawn_range), so the container itself no longer needs its own position DR.
    # Kept as a real range (not folded into a single value) so it can be widened again later.
    "ingredient_bowl_x_range": [0.45, 0.45],
    "ingredient_bowl_y_range": [-0.10, -0.10],
    "overview_cam_eye": [1.5, 0.1, 1.0],
    "overview_cam_target": [0.3, 0.1, 0.3],
    "render_camera": False,
    "render_cam_eye": [2.3, -1.9, 1.9],
    "render_cam_target": [0.3, 0.0, 0.35],
    "render_image_size": [720, 1280],
    "rewards": {},
    "reward_set": "staged_v1",
    "reward_weights": {},
    "food_params": {},
    # Continuous demo (pipeline/0_state_teacher/demo.py only): None = the training scene; a mapping with any of
    # {bowls, food_pool, spacing} builds the demo scene (see FoodCellEnvCfg.demo).
    "demo": None,
}

DEMO_KEYS = {"bowls": "demo_bowls", "food_pool": "demo_food_pool", "spacing": "demo_spacing"}


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


def build_cell_env_cfg(env_cfg: Mapping):
    """Merge ``env_cfg`` over ``DEFAULT_ENV`` and construct the Isaac Lab env config."""
    # validated before the simulator imports below, so a typo fails fast (and is unit-testable without Isaac Sim)
    unknown = set(env_cfg) - set(DEFAULT_ENV)
    if unknown:
        raise ValueError(f"Unknown env config keys {sorted(unknown)}. Allowed: {sorted(DEFAULT_ENV)}")

    from food_robot.belt import BeltCfg
    from food_robot.envs.cell_env_cfg import FoodCellEnvCfg, scale_physx_buffers
    from food_robot.rewards import DENSE_TERMS, isaac_weight, resolve_reward_weights

    c = {**DEFAULT_ENV, **env_cfg}
    weights = resolve_reward_weights(c)
    arm = _resolve(ARMS, c["arm"], "arm")
    food_cls = _resolve(FOODS, c["food"], "food")
    food = food_cls(**_tuples(c["food_params"]))

    demo = {}
    if c["demo"] is not None:
        unknown_demo = set(c["demo"]) - set(DEMO_KEYS)
        if unknown_demo:
            raise ValueError(f"Unknown demo keys {sorted(unknown_demo)}. Allowed: {sorted(DEMO_KEYS)}")
        demo = {"demo": True, **{DEMO_KEYS[k]: v for k, v in c["demo"].items() if v is not None}}

    cfg = FoodCellEnvCfg(
        arm=arm,
        food=food,
        belt=BeltCfg(**_geometry_kwargs(_tuples(c["belt"]))),
        action_mode=c["action_mode"],
        cameras=bool(c["cameras"]),
        image_size=tuple(c["image_size"]),
        frame_stack=int(c["frame_stack"]),
        privileged_information=bool(c["privileged_information"]),
        success_bonus=weights["success"],
        bowl_failure_penalty=-weights["bowl_failure"],
        food_drop_penalty=-weights["food_dropped"],
        success_settle_steps=int(c["success_settle_steps"]),
        success_requires_home=bool(c["success_requires_home"]),
        home_tolerance=float(c["home_tolerance"]),
        robot_reset=_tuples({**DEFAULT_ENV["robot_reset"], **c["robot_reset"]}),
        ingredient_bowl_pos=tuple(c["ingredient_bowl_pos"]),
        ingredient_bowl_x_range=tuple(c["ingredient_bowl_x_range"]),
        ingredient_bowl_y_range=tuple(c["ingredient_bowl_y_range"]),
        overview_cam_eye=tuple(c["overview_cam_eye"]),
        overview_cam_target=tuple(c["overview_cam_target"]),
        render_camera=bool(c["render_camera"]),
        render_cam_eye=tuple(c["render_cam_eye"]),
        render_cam_target=tuple(c["render_cam_target"]),
        render_image_size=tuple(c["render_image_size"]),
        **demo,
    )
    cfg.scene.num_envs = int(c["num_envs"])
    scale_physx_buffers(cfg.sim.physics, cfg.scene.num_envs)
    cfg.seed = c["seed"]
    cfg.sim.device = c["device"]
    for term in DENSE_TERMS:
        # Isaac Lab keeps computing every dense term (non-zero weight) so the reward-term vector can recover
        # the unweighted value; the training reward itself comes from LineariseRewards in make_env.
        getattr(cfg.rewards, term).weight = isaac_weight(weights[term])
    return cfg
