"""State-teacher models and checkpoints. Loaded by file path from this folder's scripts (see train.py)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from food_robot.artifacts import git_commit, sha256_file, write_json
from food_robot.keys import expand_in_keys


class StateNet(nn.Module):
    """Flattens and concatenates vector observations -> MLP (ELU, orthogonal init)."""

    def __init__(self, shapes, hidden, out_dim: int, out_gain: float):
        super().__init__()
        self.shapes = shapes
        self.mlp = MLP(
            in_features=sum(math.prod(s) for s in shapes),
            out_features=out_dim,
            num_cells=list(hidden),
            activation_class=nn.ELU,
        )
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        flat = [x.float().reshape(*x.shape[: x.dim() - len(s)], -1) for x, s in zip(xs, self.shapes)]
        return self.mlp(torch.cat(flat, dim=-1))


class ActorNet(nn.Module):
    def __init__(self, body: StateNet, action_dim: int):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(action_dim, scale_lb=1e-4)

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def make_teacher_models(env, network_cfg, device: torch.device):
    """Separate MLP actor (TanhNormal on [-1, 1], state-independent scale) and MLP critic."""
    spec = env.observation_spec
    ndim = len(env.batch_size)
    action_dim = env.action_spec.shape[-1]

    def shapes(keys):
        return [tuple(spec[k].shape[ndim:]) for k in keys]

    actor_keys = expand_in_keys(spec, list(network_cfg.actor_in_keys))
    critic_keys = expand_in_keys(spec, list(network_cfg.critic_in_keys))
    for key in actor_keys + critic_keys:
        if len(spec[key].shape[ndim:]) > 1:
            raise ValueError(f"The state teacher takes vector observations only; {key} has shape {spec[key].shape}.")
    actor = ProbabilisticActor(
        TensorDictModule(
            ActorNet(StateNet(shapes(actor_keys), network_cfg.hidden, action_dim, out_gain=0.01), action_dim),
            in_keys=actor_keys,
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic = ValueOperator(StateNet(shapes(critic_keys), network_cfg.hidden, 1, out_gain=1.0), in_keys=critic_keys)
    return actor.to(device), critic.to(device)


def load_teacher_actor(checkpoint_path, env, device: torch.device):
    """Rebuild the actor from a teacher checkpoint (its own network config and weights)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor, _ = make_teacher_models(env, OmegaConf.create(checkpoint["config"]["network"]), device)
    actor.load_state_dict(checkpoint["actor"])
    return actor


def save_teacher_checkpoint(run_dir, name: str, actor, critic, optim, cfg, frames: int, iteration: int) -> Path:
    """``<run_dir>/checkpoints/<name>.pt`` plus the sidecar manifest ``<name>.json``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"
    config = OmegaConf.to_container(cfg, resolve=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optim": optim.state_dict(),
            "frames": frames,
            "iteration": iteration,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)
    write_json(
        path.with_suffix(".json"),
        {
            "checkpoint": path.name,
            "frames": frames,
            "iteration": iteration,
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": sha256_file(path),
            "config": config,
            "eval": None,
            "video": None,
        },
    )
    return path
