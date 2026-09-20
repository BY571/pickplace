"""IQL: expectile value loss + advantage-weighted policy, with TorchRL's ``IQLLoss``. Loaded by file path.

Wiring follows TorchRL's ``sota-implementations/iql/iql_offline.py`` (one optimizer group over the three
losses, ``SoftUpdate`` on the Q targets); only the networks differ — a Nature CNN per camera instead of an
MLP over a state vector, and separate encoders for the actor, the twin Q nets and the value net.
"""

from __future__ import annotations

import torch
from torchrl.objectives import IQLLoss, SoftUpdate

from pickplace.offline import make_actor, make_qvalue, make_value

_LOG_KEYS = ("loss_actor", "loss_qvalue", "loss_value", "entropy")


class IQL:
    def __init__(self, actor, loss_module, target_updater, lr: float, max_grad_norm: float):
        self.policy = actor
        self.loss_module = loss_module
        self.target_updater = target_updater
        self.optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
        self.max_grad_norm = float(max_grad_norm)

    def update(self, batch) -> dict[str, float]:
        info = self.loss_module(batch)
        loss = info["loss_actor"] + info["loss_qvalue"] + info["loss_value"]
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
        self.optim.step()
        self.target_updater.step()
        out = {k: info[k].detach() for k in _LOG_KEYS if k in info.keys()}
        return {**out, "loss_total": loss.detach(), "grad_norm": grad_norm}

    def state_dict(self) -> dict:
        return {"actor": self.policy.state_dict(), "loss_module": self.loss_module.state_dict(),
                "optim": self.optim.state_dict()}


def make_algo(cfg, image_shapes, obs_keys, action_dim, device):
    actor = make_actor(image_shapes, obs_keys, action_dim, cfg.network, device)
    qvalue = make_qvalue(image_shapes, obs_keys, action_dim, cfg.network, device)
    value = make_value(image_shapes, obs_keys, cfg.network, device)
    loss_module = IQLLoss(
        actor,
        qvalue,
        value_network=value,
        num_qvalue_nets=int(cfg.loss.num_qvalue_nets),
        loss_function=cfg.loss.loss_function,
        temperature=float(cfg.loss.temperature),
        expectile=float(cfg.loss.expectile),
        # The twin Q nets are convolutional; looping over them is clearer (and no slower here) than vmapping
        # a functional call over conv weights.
        deactivate_vmap=True,
    )
    loss_module.make_value_estimator(gamma=float(cfg.loss.gamma), device=device)
    target_updater = SoftUpdate(loss_module, tau=float(cfg.loss.target_tau))
    return IQL(actor, loss_module.to(device), target_updater, cfg.optim.lr, cfg.optim.max_grad_norm)
