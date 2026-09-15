"""Frame-stacked pixel observations: shape, dtype, history across steps, history reset."""

from _common import finish

from food_robot.app import launch_app

app = launch_app(headless=True, enable_cameras=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import food_robot.envs  # noqa: E402,F401
from food_robot.envs.cell_env_cfg import FoodCellEnvCfg  # noqa: E402


def single_frame_term() -> str:
    cfg = FoodCellEnvCfg(cameras=True, privileged_information=False, image_size=(64, 64), frame_stack=1)
    return cfg.observations.pixels.wrist_rgb.func.__name__


def main():
    cfg = FoodCellEnvCfg(cameras=True, privileged_information=False, image_size=(64, 64), frame_stack=3)
    cfg.scene.num_envs = 2
    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    action = torch.zeros(2, u.action_manager.total_action_dim, device=u.device)
    for _ in range(40):  # 0.8 s: the bowl moves ~6 cm on the belt
        obs, *_ = env.step(action)
    overview = obs["pixels"]["overview_rgb"]
    diff_after_steps = float((overview[..., 0:3] - overview[..., 6:9]).abs().mean())
    obs, _ = env.reset()
    overview = obs["pixels"]["overview_rgb"]
    diff_after_reset = float((overview[..., 0:3] - overview[..., 6:9]).abs().mean())
    finish(
        True,
        wrist_shape=list(obs["pixels"]["wrist_rgb"].shape[1:]),
        overview_shape=list(overview.shape[1:]),
        dtype=str(overview.dtype),
        min_value=float(overview.min()),
        max_value=float(overview.max()),
        oldest_vs_newest_diff_after_steps=diff_after_steps,
        oldest_vs_newest_diff_after_reset=diff_after_reset,
        single_frame_term=single_frame_term(),
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
