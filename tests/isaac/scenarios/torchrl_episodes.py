"""Episode-boundary checks at the TorchRL/PPO level: short episodes, GAE, StepCounter/RewardSum reset.

Uses a short belt cycle (speed=0.3, place_window=0.6) so a long-enough rollout crosses several
episode boundaries per sub-env without needing a policy that succeeds at the task; the zero/no-op
action just lets the bowl ride through the reach zone and exit (``bowl_exited_zone``), ending the
episode quickly and repeatedly.
"""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True)

import torch  # noqa: E402
from torchrl.modules import MLP, ValueOperator  # noqa: E402
from torchrl.objectives.value.advantages import GAE  # noqa: E402

from pickplace.metrics import OUTCOME_TERMS  # noqa: E402
from pickplace.torchrl_env import make_env, reward_term_stats, termination_stats  # noqa: E402

NUM_ENVS = 4
NUM_STEPS = 600
MIN_DONE_PER_ENV = 2

# Representative observation leaf keys (deliberately excluding "step_count"/"episode_reward", which
# are bookkeeping keys added by the transform stack and are checked separately below).
OBS_KEYS = [("proprio", "ee_pos"), ("belt", "bowl_pos"), ("privileged", "food_pos")]


def _build_critic(env, device):
    in_features = sum(int(env.observation_spec[k].shape[-1]) for k in OBS_KEYS)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP(in_features=in_features, out_features=1, num_cells=[64, 64], activation_class=torch.nn.ELU)

        def forward(self, *xs: torch.Tensor) -> torch.Tensor:
            return self.mlp(torch.cat(xs, dim=-1))

    return ValueOperator(Critic(), in_keys=list(OBS_KEYS)).to(device)


def main():
    env_cfg = {
        "num_envs": NUM_ENVS,
        "cameras": False,
        "privileged_information": True,
        "belt": {"speed": 0.3, "place_window": 0.6, "pallet_start_range": [0.0, 0.0]},
    }
    env = make_env(env_cfg)
    device = env.device
    critic = _build_critic(env, device)

    with torch.no_grad():
        td = env.rollout(NUM_STEPS, break_when_any_done=False)

    done = td["next", "done"].squeeze(-1)  # (num_envs, num_steps)
    per_env_done = done.sum(dim=1)
    n_done = int(done.sum())
    if n_done == 0:
        return finish(False, error="no episode finished within the rollout window")
    if int(per_env_done.min()) < MIN_DONE_PER_ENV:
        return finish(
            False,
            error=f"some env saw fewer than {MIN_DONE_PER_ENV} episodes",
            per_env_done=per_env_done.tolist(),
        )

    with torch.no_grad():
        td = GAE(gamma=0.99, lmbda=0.95, value_network=critic, average_gae=False, device=device)(td)
    advantage_finite = bool(torch.isfinite(td["advantage"]).all())
    value_target_finite = bool(torch.isfinite(td["value_target"]).all())

    # StepCounter resets: IsaacLabWrapper's native_autoreset=True means the auto-reset happens inside
    # the same env.step() call that produced done=True, so TorchRL's StepCounter._reset_on_native_
    # autoreset zeroes ("next", "step_count") on the done row itself; the following row is the first
    # step of the new episode, so its ("next", "step_count") is 1. Verified empirically on the Spark.
    step_count_next = td["next", "step_count"].squeeze(-1)
    done_step_counts = sorted(set(step_count_next[done].tolist()))
    after_done_mask = torch.zeros_like(done)
    after_done_mask[:, 1:] = done[:, :-1]
    after_done_step_counts = sorted(set(step_count_next[after_done_mask].tolist())) if after_done_mask.any() else []

    # episode_reward resets after done: RewardSum's episode_reward is also invalidated (NaN, being a
    # float observation) on the done row by the same native-autoreset mechanism that NaNs raw
    # observations (see below); the row after must show a small, freshly-accumulated value rather
    # than a continuation of the completed episode's total (which is gone).
    episode_reward_next = td["next", "episode_reward"].squeeze(-1)
    episode_reward_nan_only_on_done = bool((torch.isnan(episode_reward_next) == done).all())
    reward_next = td["next", "reward"].squeeze(-1)
    after_done_reward_matches = True
    if after_done_mask.any():
        after_done_matches = torch.allclose(
            episode_reward_next[after_done_mask], reward_next[after_done_mask], atol=1e-5
        )
        after_done_reward_matches = bool(after_done_matches)

    # NaNs in ("next", <obs>) appear only on done rows, for every observation leaf checked.
    nan_only_on_done = {}
    for key in OBS_KEYS:
        obs_next = td.get(("next",) + key)
        nan_rows = torch.isnan(obs_next).any(dim=-1)
        nan_only_on_done[".".join(key)] = bool((nan_rows == done).all())

    # Episode outcomes: flags only on done rows, at least one per done row, and the last episode of every env
    # agrees with Isaac Lab's own last-episode termination stats.
    outcomes = {t: td["next", "outcome", t].squeeze(-1) for t in OUTCOME_TERMS}
    outcome_only_on_done = all(bool((flags & ~done).sum() == 0) for flags in outcomes.values())
    any_outcome = torch.stack(list(outcomes.values())).any(dim=0)
    every_done_has_outcome = bool((any_outcome[done]).all())
    last_done_idx = (done.float() * torch.arange(done.shape[1], device=done.device)).argmax(dim=1)
    rows = torch.arange(done.shape[0], device=done.device)
    last_episode_rates = {t: float(flags[rows, last_done_idx].float().mean()) for t, flags in outcomes.items()}
    stats = termination_stats(env)
    outcome_counts = {t: int(flags.sum()) for t, flags in outcomes.items()}

    finish(
        True,
        n_done=n_done,
        per_env_done=per_env_done.tolist(),
        advantage_finite=advantage_finite,
        value_target_finite=value_target_finite,
        done_step_counts=done_step_counts,
        after_done_step_counts=after_done_step_counts,
        episode_reward_nan_only_on_done=episode_reward_nan_only_on_done,
        after_done_reward_matches=after_done_reward_matches,
        nan_only_on_done=nan_only_on_done,
        outcome_only_on_done=outcome_only_on_done,
        every_done_has_outcome=every_done_has_outcome,
        last_episode_rates=last_episode_rates,
        termination_stats=stats,
        outcome_counts=outcome_counts,
        reward_terms=sorted(reward_term_stats(env)),
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
