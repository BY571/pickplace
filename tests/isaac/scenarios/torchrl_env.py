"""TorchRL-level checks: specs, rollout shapes, key routing, termination stats, partial reset."""

import sys

from _common import finish

MODE, CAMERAS, PRIVILEGED = sys.argv[1], bool(int(sys.argv[2])), bool(int(sys.argv[3]))

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=CAMERAS)

import torch  # noqa: E402
from torchrl.envs.utils import check_env_specs, step_mdp  # noqa: E402

from food_robot.config import build_cell_env_cfg  # noqa: E402
from food_robot.keys import expand_in_keys  # noqa: E402
from food_robot.torchrl_env import make_env, termination_stats  # noqa: E402

ENV = {"num_envs": 4, "cameras": CAMERAS, "privileged_information": PRIVILEGED}


def config_errors():
    out = {}
    cases = {
        "unknown_key": {"bogus": 1},
        "unknown_arm": {"arm": "nope"},
        "unknown_reward": {"rewards": {"bogus_term": 1.0}},
        "one_shot_reward": {"rewards": {"place_success": 5.0}},
    }
    for name, bad in cases.items():
        try:
            build_cell_env_cfg({**ENV, **bad})
            out[name] = "no error"
        except Exception as exc:  # noqa: BLE001
            out[name] = type(exc).__name__
    return out


def belt_pallet_override() -> float:
    """A nested belt.pallet override must actually reach the PalletGeometry the scene builds with."""
    cfg = build_cell_env_cfg({**ENV, "belt": {"pallet": {"travel_upper": 2.0}}})
    return cfg.belt.pallet.travel_upper


def specs():
    errors = config_errors()
    env = make_env(ENV)
    check_env_specs(env, break_when_any_done="both")
    td = env.rollout(20, break_when_any_done=False)
    leaf_keys = [list(k) for k in env.observation_spec.keys(True, True) if isinstance(k, tuple)]
    actor_keys = [list(k) for k in expand_in_keys(env.observation_spec, ["proprio", "belt"])]
    finish(
        True,
        check_env_specs="ok",
        batch_size=list(env.batch_size),
        rollout_ee_pos_shape=list(td["next", "proprio", "ee_pos"].shape),
        action_shape=list(env.action_spec.shape),
        leaf_keys=leaf_keys,
        actor_keys=actor_keys,
        termination_stats=termination_stats(env),
        config_errors=errors,
        belt_pallet_travel_upper=belt_pallet_override(),
    )


def partial_reset():
    belt = {"speed_noise": 0.0, "bowl_offset_x": [0.0, 0.0], "bowl_offset_y": [0.0, 0.0]}
    env = make_env({**ENV, "belt": belt})
    cfg = build_cell_env_cfg({**ENV, "belt": belt})
    td = env.reset()
    for _ in range(25):
        td.set("action", torch.zeros(env.action_spec.shape, device=env.device))
        td = step_mdp(env.step(td))
    before = td["belt", "bowl_pos"][:, 0].clone()
    mask = torch.zeros(4, 1, dtype=torch.bool, device=env.device)
    mask[0] = True
    td.set("_reset", mask)
    td = env.reset(td)
    after = td["belt", "bowl_pos"][:, 0]
    finish(
        True,
        entry_x=cfg.belt.entry_x(),
        reset_env_x=float(after[0]),
        other_env_x_before=float(before[1]),
        other_env_x_after=float(after[1]),
    )


try:
    {"specs": specs, "partial_reset": partial_reset}[MODE]()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
