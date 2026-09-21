"""Behaviour cloning: maximise the log-probability of the dataset action. Loaded by file path.

**Why log-prob and not MSE to the teacher's ``loc``.** The dataset's ``action`` is what was actually
executed, which is what every other algorithm here regresses toward or re-weights; ``loc`` only exists
because the behaviour policy happened to be a Gaussian teacher and it ignores the exploration noise that
produced the medium tier's states. Log-prob also makes BC literally IQL's actor loss with the
advantage weight fixed at 1, so the baseline differs from IQL in exactly one term.
"""

from __future__ import annotations

import torch

from pickplace.offline import make_actor

# TanhNormal's log_prob goes through atanh, which is infinite at the bounds.
_EPS = 1e-6


class BC:
    def __init__(self, actor, lr: float, max_grad_norm: float):
        self.policy = actor
        self.optim = torch.optim.Adam(actor.parameters(), lr=lr)
        self.max_grad_norm = float(max_grad_norm)

    def update(self, batch) -> dict[str, float]:
        action = batch.get("action").clamp(-1.0 + _EPS, 1.0 - _EPS)
        dist = self.policy.get_dist(batch)
        log_prob = dist.log_prob(action)
        loss = -log_prob.mean()
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optim.step()
        with torch.no_grad():
            # The action the deterministic evaluation would take, vs the dataset's.
            mse = (dist.deterministic_sample - action).pow(2).mean()
        return {"loss_bc": loss.detach(), "log_prob": log_prob.mean().detach(), "action_mse": mse,
                "scale": dist.scale.mean().detach(), "grad_norm": grad_norm}

    def state_dict(self) -> dict:
        return {"actor": self.policy.state_dict(), "optim": self.optim.state_dict()}


def make_algo(cfg, obs_shapes, obs_keys, action_dim, device):
    actor = make_actor(obs_shapes, obs_keys, action_dim, cfg.network, device)
    return BC(actor, cfg.optim.lr, cfg.optim.max_grad_norm)
