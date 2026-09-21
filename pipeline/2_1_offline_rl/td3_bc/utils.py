"""TD3+BC: TD3 with the actor's Q term rescaled against a behaviour-cloning term. Loaded by file path.

Wiring follows TorchRL's ``sota-implementations/td3_bc/td3_bc.py``: ``qvalue_loss`` every step,
``actor_loss`` (and the target update) only every ``policy_update_delay`` steps. The actor is
deterministic — ``TD3BCLoss`` reads ``action`` straight out of it and adds the target-policy smoothing
noise itself — so this is the one algorithm here that uses ``make_deterministic_actor`` rather than the
stochastic ``make_actor``; the body is the same ``PixelNet``.
"""

from __future__ import annotations

import torch
from torchrl.objectives import SoftUpdate, TD3BCLoss

from pickplace.offline import make_deterministic_actor, make_qvalue


class TD3BC:
    def __init__(self, actor, loss_module, target_updater, lr: float, max_grad_norm: float,
                 policy_update_delay: int):
        self.policy = actor
        self.loss_module = loss_module
        self.target_updater = target_updater
        # One Adam over the whole loss module: the actor loss detaches the Q params and the Q loss uses the
        # *target* actor, so each backward only ever puts gradients on its own half.
        self.optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
        self.max_grad_norm = float(max_grad_norm)
        self.policy_update_delay = int(policy_update_delay)
        self.step = 0
        # The actor updates every policy_update_delay steps; its terms are re-reported on the steps in
        # between so the runner's log_interval average is the mean of the actor loss, not of half of it.
        self.actor_log: dict[str, torch.Tensor] = {}

    def _backward(self, loss) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
        self.optim.step()
        return grad_norm

    def update(self, batch) -> dict[str, float]:
        self.step += 1
        q_loss, q_meta = self.loss_module.qvalue_loss(batch)
        grad_norm = self._backward(q_loss)
        if self.step % self.policy_update_delay == 0:
            actor_loss, actor_meta = self.loss_module.actor_loss(batch)
            actor_grad_norm = self._backward(actor_loss)
            self.target_updater.step()
            self.actor_log = {
                "loss_actor": actor_loss.detach(),
                "bc_loss": actor_meta["bc_loss"],
                "lmbd": actor_meta["lmbd"].detach(),
                "q_actor": actor_meta["state_action_value_actor"].mean(),
                "actor_grad_norm": actor_grad_norm,
            }
        return {"loss_qvalue": q_loss.detach(), "grad_norm": grad_norm,
                "pred_value": q_meta["pred_value"].mean(), "target_value": q_meta["target_value"].mean(),
                **self.actor_log}

    def state_dict(self) -> dict:
        return {"actor": self.policy.state_dict(), "loss_module": self.loss_module.state_dict(),
                "optim": self.optim.state_dict()}


def make_algo(cfg, obs_shapes, obs_keys, action_dim, device):
    actor = make_deterministic_actor(obs_shapes, obs_keys, action_dim, cfg.network, device)
    qvalue = make_qvalue(obs_shapes, obs_keys, action_dim, cfg.network, device)
    loss_module = TD3BCLoss(
        actor,
        qvalue,
        bounds=(-1.0, 1.0),          # the teacher's action support; collect.py clips to it
        num_qvalue_nets=int(cfg.loss.num_qvalue_nets),
        loss_function=cfg.loss.loss_function,
        policy_noise=float(cfg.loss.policy_noise),
        noise_clip=float(cfg.loss.noise_clip),
        alpha=float(cfg.loss.alpha),
        # The twin Q nets are convolutional; looping over them is clearer (and no slower here) than vmapping
        # a functional call over conv weights.
        deactivate_vmap=True,
    )
    loss_module.make_value_estimator(gamma=float(cfg.loss.gamma), device=device)
    target_updater = SoftUpdate(loss_module, tau=float(cfg.loss.target_tau))
    return TD3BC(actor, loss_module.to(device), target_updater, cfg.optim.lr, cfg.optim.max_grad_norm,
                 cfg.loss.policy_update_delay)
