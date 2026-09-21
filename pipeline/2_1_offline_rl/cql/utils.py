"""CQL: a SAC actor plus a conservative Q penalty, with TorchRL's ``CQLLoss``. Loaded by file path.

Wiring follows TorchRL's ``sota-implementations/cql/cql_offline.py``: the CQL penalty is added to the
Q loss, the entropy (and, with the Lagrange variant, the penalty-weight) duals are learned alongside, and
the actor is warm-started on the behaviour-cloning term for the first ``policy_eval_start`` steps. Only the
networks differ — a Nature CNN per camera instead of an MLP over a state vector, and separate encoders for
the actor and the twin Q nets. CQL needs no value net.
"""

from __future__ import annotations

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Bounded
from torchrl.objectives import CQLLoss, SoftUpdate

from pickplace.offline import as_key, make_actor, make_qvalue

_LOG_KEYS = ("loss_actor", "loss_actor_bc", "loss_qvalue", "loss_cql", "loss_alpha", "loss_alpha_prime",
             "alpha", "entropy")


def _flat_aliases(obs_keys) -> TensorDictModule:
    """Alias the shard's nested observation keys to flat ones, because ``CQLLoss`` needs flat keys.

    ``CQLLoss.cql_loss`` and ``_get_policy_actions`` repeat the observation once per random/policy action
    with ``tensordict.named_apply(f)``, which hands ``f`` a *leaf* name. A nested key such as
    ``("pixels", "wrist_rgb")`` therefore never matches ``actor_network.in_keys`` and is silently dropped
    from the repeated tensordict, so the Q nets would be called without their images (it raises). Giving
    CQL's networks flat key names and pointing them at the same tensors — a view, not a copy — is the
    smallest fix and leaves the other algorithms and the shard layout untouched.
    """
    keys = [as_key(k) for k in obs_keys]
    return TensorDictModule(lambda *xs: xs, in_keys=keys, out_keys=["_".join(k) for k in keys])


class CQL:
    def __init__(self, actor, aliases, loss_module, target_updater, lr: float, max_grad_norm: float,
                 policy_eval_start: int):
        # What is evaluated and checkpointed: the env's nested observations, aliased, then the actor.
        self.policy = TensorDictSequential(aliases, actor)
        self.aliases = aliases
        self.loss_module = loss_module
        self.target_updater = target_updater
        # One Adam over every parameter of the loss module, as IQL does here: that includes the entropy
        # dual (and the Lagrange dual), which is what the sota's grouped optimizers amount to at one lr.
        self.optim = torch.optim.Adam(loss_module.parameters(), lr=lr)
        self.max_grad_norm = float(max_grad_norm)
        self.policy_eval_start = int(policy_eval_start)
        self.step = 0

    def update(self, batch) -> dict[str, float]:
        self.step += 1
        self.aliases(batch)                 # s
        self.aliases(batch.get("next"))     # s' (a view into batch, so this writes through)
        info = self.loss_module(batch)
        # Warm-start: until policy_eval_start the actor clones the data instead of chasing a Q function
        # that has not seen enough of the batch distribution yet (sota-implementations/cql/cql_offline.py).
        actor_loss = info["loss_actor"] if self.step >= self.policy_eval_start else info["loss_actor_bc"]
        loss = actor_loss + info["loss_qvalue"] + info["loss_cql"] + info["loss_alpha"]
        if "loss_alpha_prime" in info.keys():
            loss = loss + info["loss_alpha_prime"]
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


def make_algo(cfg, obs_shapes, obs_keys, action_dim, device):
    aliases = _flat_aliases(obs_keys)
    flat_keys = aliases.out_keys
    actor = make_actor(obs_shapes, flat_keys, action_dim, cfg.network, device,
                       scale_lb=float(cfg.loss.actor_scale_lb))
    qvalue = make_qvalue(obs_shapes, flat_keys, action_dim, cfg.network, device)
    loss_module = CQLLoss(
        actor,
        qvalue,
        loss_function=cfg.loss.loss_function,
        temperature=float(cfg.loss.temperature),
        min_q_weight=float(cfg.loss.min_q_weight),
        max_q_backup=bool(cfg.loss.max_q_backup),
        deterministic_backup=bool(cfg.loss.deterministic_backup),
        num_random=int(cfg.loss.num_random),
        with_lagrange=bool(cfg.loss.with_lagrange),
        lagrange_thresh=float(cfg.loss.lagrange_thresh),
        # make_actor's ProbabilisticActor carries no spec, and target_entropy="auto" needs one.
        action_spec=Bounded(-1.0, 1.0, (action_dim,), device=device),
        # The twin Q nets are convolutional; looping over them is clearer (and no slower here) than vmapping
        # a functional call over conv weights.
        deactivate_vmap=True,
    )
    loss_module.make_value_estimator(gamma=float(cfg.loss.gamma), device=device)
    target_updater = SoftUpdate(loss_module, tau=float(cfg.loss.target_tau))
    return CQL(actor, aliases, loss_module.to(device), target_updater, cfg.optim.lr, cfg.optim.max_grad_norm,
               cfg.loss.policy_eval_start)
