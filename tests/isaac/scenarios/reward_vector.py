"""Reward-term vector: specs, equivalence with Isaac Lab's reward, events on done rows, term sums, set switch."""

import math
import sys

from _common import finish

SPARSE = "--sparse" in sys.argv

from food_robot.app import launch_app  # noqa: E402

app = launch_app(headless=True)

import torch  # noqa: E402
from torchrl.envs.utils import check_env_specs  # noqa: E402

from food_robot.config import build_cell_env_cfg  # noqa: E402
from food_robot.rewards import DENSE_TERMS, EVENT_TERMS, REWARD_TERMS  # noqa: E402
from food_robot.torchrl_env import make_env, reward_weights  # noqa: E402

N, STEPS = 16, 600
# Short belt cycle so episodes end often within the rollout (as in the torchrl_episodes scenario).
ENV = {
    "num_envs": N,
    "cameras": False,
    "privileged_information": True,
    "belt": {"speed": 0.3, "place_window": 0.6, "pallet_start_range": [0.0, 0.0]},
    "success_requires_home": False,  # the default, stated: the forced success below must behave as before v3
}
if SPARSE:
    ENV["reward_weights"] = {t: 0.0 for t in REWARD_TERMS if t != "success"} | {"success": 7.0}


FORCE_EVERY, FORCE_AT = 200, 50  # force events at steps 50, 250, 450


def force_events(u, cfg):
    """Make the event half of the vector fire: random actions only ever end episodes by time_out /
    bowl_exited_zone, whose event components are all zero, which would leave those columns untested.

    Teleports (as `food_behaviour` / `belt_behaviour` do): env 0's food below the table (`food_off_table`),
    env 1's bowl sideways off the belt (`bowl_off_belt`), env 2's bowl tilted 60 deg (`bowl_tipped`), and
    env 3's food at rest in its bowl (`success` after the settle steps -- best effort, nothing asserts it; the
    `food_in_bowl` and `return_home` dense components must be > 0 on env 3 and stay 0 on every other env).
    """
    device = u.device
    food, bowl = u.scene["food"], u.scene["bowl"]
    fpos, fquat = food.data.root_pos_w.torch.clone(), food.data.root_quat_w.torch.clone()
    bpos, bquat = bowl.data.root_pos_w.torch.clone(), bowl.data.root_quat_w.torch.clone()

    fpos[0, 2] = u.scene.env_origins[0, 2] - 0.3
    fpos[3] = bpos[3] + torch.tensor(
        [0.0, 0.0, cfg.belt.bowl.base_thickness + cfg.food.item_radius + 0.01], device=device
    )
    food_ids = torch.tensor([0, 3], device=device)
    food.write_root_pose_to_sim_index(root_pose=torch.cat([fpos, fquat], dim=-1)[food_ids], env_ids=food_ids)
    fvel = torch.zeros(2, 6, device=device)
    fvel[1, :3] = bowl.data.root_lin_vel_w.torch[3]  # the bowl rides the belt; match it so the food settles
    food.write_root_velocity_to_sim_index(root_velocity=fvel, env_ids=food_ids)

    bpos[1, 1] += 0.3
    half = math.radians(60.0) / 2
    bquat[2] = torch.tensor([math.sin(half), 0.0, 0.0, math.cos(half)], device=device)
    bowl_ids = torch.tensor([1, 2], device=device)
    bowl.write_root_pose_to_sim_index(root_pose=torch.cat([bpos, bquat], dim=-1)[bowl_ids], env_ids=bowl_ids)


def main():
    cfg = build_cell_env_cfg(ENV)
    env = make_env(ENV)
    check_env_specs(env, break_when_any_done="both")
    u = env.base_env._env.unwrapped
    rm = u.reward_manager
    ev_idx = [REWARD_TERMS.index(t) for t in EVENT_TERMS]
    # Isaac Lab computes zero-weight dense terms at weight 1.0 (so the vector can recover them), and those
    # land in its own `_reward_buf`; the reward set weights them 0, so drop their contribution before comparing.
    weights = reward_weights(ENV)
    names = list(rm.active_terms)
    zero_idx = [names.index(t) for t in DENSE_TERMS if weights[t] == 0.0]
    in_bowl_idx = REWARD_TERMS.index("food_in_bowl")
    in_bowl_max = torch.zeros(N)
    home_idx = REWARD_TERMS.index("return_home")
    home_max = torch.zeros(N)
    success_params = sorted(u.termination_manager.get_term_cfg("success").params)

    td = env.reset()
    max_diff = max_sparse = max_sum_diff = 0.0
    done_rows = event_rows = 0
    events_binary = events_only_on_done = True
    shape = None
    prev_sum = prev_done = None
    for step in range(STEPS):
        if step % FORCE_EVERY == FORCE_AT:
            force_events(u, cfg)
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
            unweighted = rm._step_reward[:, zero_idx].sum(-1) * u.step_dt
            max_diff = max(max_diff, float((reward - (rm._reward_buf - unweighted)).abs().max()))
        in_bowl_max = torch.maximum(in_bowl_max, terms[:, in_bowl_idx].float().cpu())
        home_max = torch.maximum(home_max, terms[:, home_idx].float().cpu())
        events = terms[:, ev_idx]
        event_rows += int((events.sum(-1) > 0).sum())
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
        event_rows=event_rows,
        max_abs_reward_diff=max_diff,
        max_abs_sparse_diff=max_sparse,
        events_binary=events_binary,
        events_only_on_done=events_only_on_done,
        max_abs_term_sum_diff=max_sum_diff,
        food_in_bowl_max=in_bowl_max.tolist(),
        return_home_max=home_max.tolist(),
        success_params=success_params,
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
