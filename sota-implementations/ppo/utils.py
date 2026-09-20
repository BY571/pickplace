"""PPO models for the food cell env: per-key encoders (CNN for images, flatten for vectors) + MLP heads."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from pickplace.keys import expand_in_keys


def _is_image(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3 and shape[-1] in (1, 3, 4)


class ImageEncoder(nn.Module):
    """HWC float images in [0, 255] -> embedding."""

    def __init__(self, shape: tuple[int, int, int], channels: list[int], embed_dim: int):
        super().__init__()
        h, w, c = shape
        k, s = [8, 4, 3], [4, 2, 1]
        layers, in_c = [], c
        for out_c, kk, ss in zip(channels, k, s):
            layers += [nn.Conv2d(in_c, out_c, kk, ss), nn.ELU()]
            in_c = out_c
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[-1]
        self.proj = nn.Sequential(nn.Linear(n_flat, embed_dim), nn.ELU())
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[: -3]
        x = x.reshape(-1, *self.shape).permute(0, 3, 1, 2) / 255.0
        return self.proj(self.conv(x)).reshape(*batch, -1)


class MultiKeyEncoder(nn.Module):
    """Encodes each input key and concatenates the features."""

    def __init__(self, shapes: list[tuple[int, ...]], channels: list[int], embed_dim: int):
        super().__init__()
        self.shapes = shapes
        self.encoders = nn.ModuleList(
            ImageEncoder(s, channels, embed_dim) if _is_image(s) else nn.Identity() for s in shapes
        )
        self.out_dim = sum(embed_dim if _is_image(s) else math.prod(s) for s in shapes)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        parts = []
        for x, shape, enc in zip(xs, self.shapes, self.encoders):
            if _is_image(shape):
                parts.append(enc(x))
            else:
                parts.append(x.reshape(*x.shape[: x.dim() - len(shape)], -1))
        return torch.cat(parts, dim=-1)


class Head(nn.Module):
    def __init__(self, encoder: MultiKeyEncoder, mlp: nn.Module, post: nn.Module | None = None):
        super().__init__()
        self.encoder, self.mlp, self.post = encoder, mlp, post

    def forward(self, *xs: torch.Tensor):
        out = self.mlp(self.encoder(*xs))
        return self.post(out) if self.post is not None else out


def _mlp(in_features: int, out_features: int, hidden: list[int], out_gain: float) -> MLP:
    mlp = MLP(in_features=in_features, out_features=out_features, num_cells=list(hidden), activation_class=nn.ELU)
    linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
    for layer in linears:
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linears[-1].weight, out_gain)
    return mlp


def make_ppo_models(env, network_cfg, device: torch.device):
    spec = env.observation_spec
    batch_ndim = len(env.batch_size)
    action_dim = env.action_spec.shape[-1]

    def encoder(keys):
        shapes = [tuple(spec[k].shape[batch_ndim:]) for k in keys]
        return MultiKeyEncoder(shapes, list(network_cfg.cnn_channels), network_cfg.embed_dim)

    actor_keys = expand_in_keys(spec, list(network_cfg.actor_in_keys))
    critic_keys = expand_in_keys(spec, list(network_cfg.critic_in_keys))

    actor_enc = encoder(actor_keys)
    actor_net = Head(
        actor_enc,
        _mlp(actor_enc.out_dim, action_dim, network_cfg.hidden, out_gain=0.01),
        AddStateIndependentNormalScale(action_dim, scale_lb=1e-4),
    )
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=actor_keys, out_keys=["loc", "scale"]),
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
    critic_enc = encoder(critic_keys)
    critic = ValueOperator(Head(critic_enc, _mlp(critic_enc.out_dim, 1, network_cfg.hidden, out_gain=1.0)), in_keys=critic_keys)
    return actor.to(device), critic.to(device)


def save_checkpoint(path: str | Path, actor, critic, optim, cfg, frames: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optim": optim.state_dict(),
            "frames": frames,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )
