"""Build the env for given flags, report observation group/term shapes and action dim."""

import sys

from _common import finish

cameras, privileged, action_mode = bool(int(sys.argv[1])), bool(int(sys.argv[2])), sys.argv[3]
expect_error = "--expect-error" in sys.argv

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=cameras)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.envs.cell_env_cfg import FoodCellEnvCfg  # noqa: E402


def main():
    try:
        cfg = FoodCellEnvCfg(cameras=cameras, privileged_information=privileged, action_mode=action_mode)
    except ValueError as exc:
        finish(expect_error, error=str(exc))
    if expect_error:
        finish(False, error="expected ValueError was not raised")
    cfg.scene.num_envs = 2
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    obs, _ = env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    finite = True
    for _ in range(10):
        obs, *_ = env.step(torch.zeros(2, action_dim, device=env.unwrapped.device))
    shapes = {}
    for group, terms in obs.items():
        shapes[group] = {name: list(t.shape[1:]) for name, t in terms.items()}
        finite &= all(bool(torch.isfinite(t.float()).all()) for t in terms.values())
    finish(True, shapes=shapes, action_dim=action_dim, all_finite=finite)


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
