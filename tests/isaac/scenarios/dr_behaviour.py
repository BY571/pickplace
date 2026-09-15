"""Geometric DR: supply bowl position, food spawn inside it, pallet start shift, bowl placement on the pallet."""

import sys

from _common import finish

ERRORS = "--errors" in sys.argv

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.belt import BeltCfg  # noqa: E402
from food_robot.envs.cell_env_cfg import FoodCellEnvCfg  # noqa: E402

N = 32
RESETS = 3


def config_errors() -> dict:
    cases = {
        "supply_out_of_reach": dict(ingredient_bowl_x_range=(0.35, 0.80)),
        "bowl_overhangs_pallet": dict(belt=BeltCfg(bowl_offset_y=(-0.08, 0.08))),
        "start_below_travel": dict(belt=BeltCfg(pallet_start_range=(-0.20, 0.08))),
    }
    out = {}
    for name, kwargs in cases.items():
        try:
            FoodCellEnvCfg(cameras=False, privileged_information=True, **kwargs)
            out[name] = "no error"
        except Exception as exc:  # noqa: BLE001
            out[name] = type(exc).__name__
    return out


def main():
    if ERRORS:
        return finish(True, errors=config_errors())
    cfg = FoodCellEnvCfg(cameras=False, privileged_information=True)
    cfg.scene.num_envs = N
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    u = env.unwrapped
    origins = u.scene.env_origins
    belt_term = u.event_manager.get_term_cfg("reset_belt").func
    supply, food_offset, starts, bowl_err = [], [], [], []
    for _ in range(RESETS):
        obs, _ = env.reset()
        supply_xy = u.scene["ingredient_bowl"].data.root_pos_w.torch[:, :2] - origins[:, :2]
        supply += supply_xy.tolist()
        food_offset += (obs["privileged"]["food_pos"][:, :2] - supply_xy).tolist()
        starts += belt_term.start.tolist()
        expected_x = cfg.belt.entry_x() + belt_term.start + belt_term.offset[:, 0]
        bowl_err += (obs["belt"]["bowl_pos"][:, 0] - expected_x).abs().tolist()
    action = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
    for _ in range(10):
        env.step(action)
    speeds = u.scene["pallet"].data.joint_vel.torch[:, belt_term.joint_ids[0]].tolist()
    finish(
        True,
        supply_bowl_xy=supply,
        food_offset_xy=food_offset,
        pallet_start=starts,
        bowl_x_minus_expected=bowl_err,
        pallet_speeds=speeds,
        x_range=list(cfg.ingredient_bowl_x_range),
        y_range=list(cfg.ingredient_bowl_y_range),
        spawn_range=cfg.food.spawn_range,
        start_range=list(cfg.belt.pallet_start_range),
        speed=cfg.belt.speed,
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
