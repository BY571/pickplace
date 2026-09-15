"""TorchRL entry point: the official IsaacLabWrapper plus generic bookkeeping transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from food_robot.metrics import OUTCOME_TERMS


def _unwrapped(env):
    base = env.base_env if hasattr(env, "base_env") else env
    return base._env.unwrapped


def _make_episode_outcome(terms: Sequence[str]):
    import torch
    from torchrl.data import Binary, Composite
    from torchrl.envs import Transform

    class EpisodeOutcome(Transform):
        """Writes ``("next", "outcome", <term>)`` bools: True on a done row whose episode ended by <term>.

        Reads Isaac Lab's termination manager right after the step. Under native auto-reset the manager's
        per-term flags still describe the step that just ended (the reset does not clear them), whereas the
        next observations are already NaN / reset. Reset tensordicts carry all-False flags.
        """

        def __init__(self):
            super().__init__(in_keys=[], out_keys=[("outcome", t) for t in terms])
            self.terms = tuple(terms)

        def _step(self, tensordict, next_tensordict):
            manager = _unwrapped(self.parent).termination_manager
            done = next_tensordict.get("done")
            for term in self.terms:
                flag = manager.get_term(term).reshape(done.shape).to(done.device)
                next_tensordict.set(("outcome", term), flag & done)
            return next_tensordict

        def _reset(self, tensordict, tensordict_reset):
            shape = (*tensordict_reset.batch_size, 1)
            for term in self.terms:
                tensordict_reset.set(
                    ("outcome", term), torch.zeros(shape, dtype=torch.bool, device=tensordict_reset.device)
                )
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            # Keep the flags of the episode that just ended on the done row (the default; stated explicitly
            # because RewardSum/StepCounter reset here instead).
            return tensordict_reset

        def transform_observation_spec(self, observation_spec):
            batch = observation_spec.shape
            # step_mdp_static=True: the generic native-autoreset "invalidate next observation on done
            # rows" pass (torchrl.envs.common.EnvBase._native_autoreset_set_invalid_next_observation)
            # sweeps every plain-Tensor leaf under observation_spec and zeroes/NaNs it on done rows,
            # which would erase the exact flags this transform means to keep. Marking the whole
            # "outcome" sub-composite step_mdp_static makes step_mdp copy it atomically instead of
            # descending into its leaves, so that pass skips it (it only touches `torch.Tensor` leaves).
            observation_spec["outcome"] = Composite(
                {
                    term: Binary(n=1, shape=(*batch, 1), dtype=torch.bool, device=observation_spec.device)
                    for term in self.terms
                },
                shape=batch,
                device=observation_spec.device,
                step_mdp_static=True,
            )
            return observation_spec

    return EpisodeOutcome()


def make_env(env_cfg: Mapping):
    """Create the food-cell env as a TorchRL ``TransformedEnv``. Launch the Isaac app before calling.

    Observations stay raw (uint8-valued float images, unnormalized states); preprocessing belongs to
    each algorithm. No running-statistics transforms here: terminal next-observations are NaN
    under native auto-reset. ``("next", "outcome", <term>)`` marks how each finished episode ended.
    """
    import gymnasium as gym
    import torch
    from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    import food_robot.envs  # noqa: F401  (gym registration)
    from food_robot.config import DEFAULT_ENV, build_cell_env_cfg

    cfg = build_cell_env_cfg(env_cfg)
    task = env_cfg.get("task", DEFAULT_ENV["task"])
    device = torch.device(env_cfg.get("device", DEFAULT_ENV["device"]))
    base = IsaacLabWrapper(gym.make(task, cfg=cfg), native_autoreset=True, device=device)
    return TransformedEnv(base, Compose(RewardSum(), StepCounter(), _make_episode_outcome(OUTCOME_TERMS)))


def _log_stats(env, prefix: str) -> dict[str, float]:
    log = _unwrapped(env).extras.get("log", {})
    return {k[len(prefix):]: float(v) for k, v in log.items() if k.startswith(prefix)}


def termination_stats(env) -> dict[str, float]:
    """Fraction of sub-envs whose most recent finished episode ended by each termination term."""
    return _log_stats(env, "Episode_Termination/")


def reward_term_stats(env) -> dict[str, float]:
    """Per reward term: mean episodic sum per second over the envs reset in the latest step (Isaac Lab log)."""
    return _log_stats(env, "Episode_Reward/")
