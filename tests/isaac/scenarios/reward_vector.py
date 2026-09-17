"""Reward-term vector: specs, equivalence with Isaac Lab's reward, events on done rows, term sums, set switch."""

import sys

from _common import finish

SPARSE = "--sparse" in sys.argv

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True)

import torch  # noqa: E402
from torchrl.envs.utils import check_env_specs  # noqa: E402

from food_robot.rewards import EVENT_TERMS, REWARD_TERMS  # noqa: E402
from food_robot.torchrl_env import make_env  # noqa: E402

N, STEPS = 16, 600
# Short belt cycle so episodes end often within the rollout (as in the torchrl_episodes scenario).
ENV = {
    "num_envs": N,
    "cameras": False,
    "privileged_information": True,
    "belt": {"speed": 0.3, "place_window": 0.6, "pallet_start_range": [0.0, 0.0]},
}
if SPARSE:
    ENV["reward_weights"] = {t: 0.0 for t in REWARD_TERMS if t != "success"} | {"success": 7.0}


def main():
    env = make_env(ENV)
    check_env_specs(env, break_when_any_done="both")
    u = env.base_env._env.unwrapped
    rm = u.reward_manager
    ev_idx = [REWARD_TERMS.index(t) for t in EVENT_TERMS]

    td = env.reset()
    max_diff = max_sparse = max_sum_diff = 0.0
    done_rows = 0
    events_binary = events_only_on_done = True
    shape = None
    prev_sum = prev_done = None
    for _ in range(STEPS):
        td.set("action", torch.rand(env.action_spec.shape, device=env.device) * 2 - 1)
        stepped, td_next_root = env.step_and_maybe_reset(td)
        terms = stepped["next", "reward_terms"]
        shape = list(terms.shape)
        reward = stepped["next", "reward"].squeeze(-1)
        done = stepped["next", "done"].squeeze(-1)
        done_rows += int(done.sum())
        if SPARSE:
            expected = 7.0 * terms[:, REWARD_TERMS.index("success")]
            max_sparse = max(max_sparse, float((reward - expected).abs().max()))
        else:
            max_diff = max(max_diff, float((reward - rm._reward_buf).abs().max()))
        events = terms[:, ev_idx]
        events_binary &= bool(((events == 0) | (events == 1)).all())
        events_only_on_done &= bool((events[~done] == 0).all())
        # The root running sum carried into this step must equal last step's sum plus last step's terms, for every
        # env whose previous step did not end an episode (those were reset and start again from zero).
        pre = stepped["episode_reward_terms"]
        if prev_sum is not None:
            keep = ~prev_done
            if keep.any():
                max_sum_diff = max(max_sum_diff, float((pre[keep] - prev_sum[keep]).abs().max()))
        prev_sum, prev_done = pre + terms, done
        td = td_next_root
    finish(
        True,
        check_env_specs="ok",
        reward_terms_shape=shape,
        done_rows=done_rows,
        max_abs_reward_diff=max_diff,
        max_abs_sparse_diff=max_sparse,
        events_binary=events_binary,
        events_only_on_done=events_only_on_done,
        max_abs_term_sum_diff=max_sum_diff,
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
