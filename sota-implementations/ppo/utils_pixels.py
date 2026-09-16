"""Pixel PPO utilities: separate actor/critic networks with per-camera CNNs, storage, metrics, eval video.

Loaded by file path from ppo_pixels.py (see _load_local_module there for why).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from food_robot.keys import expand_in_keys
from food_robot.metrics import OUTCOME_TERMS, outcome_rates


def _is_image(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3


def image_keys(observation_spec, batch_size) -> list[tuple]:
    """Observation leaf keys whose per-env shape is H x W x C (camera observations, any frame stack)."""
    ndim = len(batch_size)
    return [
        key
        for key in observation_spec.keys(True, True)
        if isinstance(key, tuple) and _is_image(tuple(observation_spec[key].shape[ndim:]))
    ]


class ConvEncoder(nn.Module):
    """H x W x C images (uint8 or float in [0, 255]) -> ELU(Linear(embed)) over a Nature-CNN-style trunk."""

    def __init__(self, shape, channels, kernels, strides, embed_dim):
        super().__init__()
        h, w, c = shape
        layers, in_c = [], c
        for out_c, k, s in zip(channels, kernels, strides, strict=True):
            layers += [nn.Conv2d(in_c, out_c, k, s), nn.ELU()]
            in_c = out_c
        self.conv = nn.Sequential(*layers, nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[-1]
        self.proj = nn.Sequential(nn.Linear(n_flat, embed_dim), nn.ELU())
        self.shape = tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[:-3]
        x = x.reshape(-1, *self.shape).permute(0, 3, 1, 2).float() / 255.0
        return self.proj(self.conv(x)).reshape(*batch, -1)


class PixelsNet(nn.Module):
    """One CNN per camera + Linear/LayerNorm/ELU over all vector inputs -> fusion MLP -> output head."""

    def __init__(self, shapes, network_cfg, out_dim: int, out_gain: float):
        super().__init__()
        self.is_image = [_is_image(s) for s in shapes]
        self.shapes = shapes
        self.cnns = nn.ModuleList(
            ConvEncoder(
                s,
                list(network_cfg.cnn_channels),
                list(network_cfg.cnn_kernels),
                list(network_cfg.cnn_strides),
                network_cfg.image_embed,
            )
            for s, img in zip(shapes, self.is_image)
            if img
        )
        vec_dim = sum(math.prod(s) for s, img in zip(shapes, self.is_image) if not img)
        self.vec = (
            nn.Sequential(
                nn.Linear(vec_dim, network_cfg.proprio_embed), nn.LayerNorm(network_cfg.proprio_embed), nn.ELU()
            )
            if vec_dim
            else None
        )
        fused = len(self.cnns) * network_cfg.image_embed + (network_cfg.proprio_embed if vec_dim else 0)
        self.mlp = MLP(
            in_features=fused, out_features=out_dim, num_cells=list(network_cfg.fusion), activation_class=nn.ELU
        )
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        cnn_iter = iter(self.cnns)
        images, vectors = [], []
        for x, shape, img in zip(xs, self.shapes, self.is_image):
            if img:
                images.append(next(cnn_iter)(x))
            else:
                vectors.append(x.float().reshape(*x.shape[: x.dim() - len(shape)], -1))
        parts = images + ([self.vec(torch.cat(vectors, dim=-1))] if self.vec is not None else [])
        return self.mlp(torch.cat(parts, dim=-1))


class ActorNet(nn.Module):
    """PixelsNet -> (loc, state-independent scale). Takes several positional inputs, unlike nn.Sequential."""

    def __init__(self, body: PixelsNet, action_dim: int):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(action_dim, scale_lb=1e-4)

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def make_ppo_models(env, network_cfg, device: torch.device):
    """Separate actor (TanhNormal on [-1, 1], state-independent scale) and critic networks."""
    spec = env.observation_spec
    batch_ndim = len(env.batch_size)
    action_dim = env.action_spec.shape[-1]

    def shapes(keys):
        return [tuple(spec[k].shape[batch_ndim:]) for k in keys]

    actor_keys = expand_in_keys(spec, list(network_cfg.actor_in_keys))
    critic_keys = expand_in_keys(spec, list(network_cfg.critic_in_keys))

    actor = ProbabilisticActor(
        TensorDictModule(
            ActorNet(PixelsNet(shapes(actor_keys), network_cfg, action_dim, out_gain=0.01), action_dim),
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
    critic = ValueOperator(PixelsNet(shapes(critic_keys), network_cfg, 1, out_gain=1.0), in_keys=critic_keys)
    return actor.to(device), critic.to(device)


def load_actor(checkpoint_path, env, device: torch.device):
    """Rebuild the actor from a ppo_pixels checkpoint (network config and weights)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor, _ = make_ppo_models(env, OmegaConf.create(checkpoint["config"]["network"]), device)
    actor.load_state_dict(checkpoint["actor"])
    return actor


def compress_pixels(td: TensorDictBase, keys) -> TensorDictBase:
    """Store camera observations as uint8 (4x smaller than float32); the encoders cast back to float."""
    for key in keys:
        if key in td.keys(True, True):
            td.set(key, td.get(key).clamp(0, 255).to(torch.uint8))
    return td


def compute_advantage(adv_module, data: TensorDictBase, env_chunk: int) -> TensorDictBase:
    """GAE over chunks of envs (dim 0) so the critic never sees all envs x steps x images at once.

    The estimator's outputs are written back into ``data`` key by key instead of concatenating the chunk
    tensordicts. TorchRL replaces NaN next-observations on a shallow *copy* of the tensordict it was handed
    (``ValueEstimatorBase._sanitize_next_obs_nan``) and then writes ``state_value`` into that copy, so a
    chunk holding a NaN comes back without ``state_value`` while its siblings keep it, and ``torch.cat``
    rejects the mismatched key sets. Only keys every chunk produced are written, which is exactly what a
    single unchunked pass would leave behind.
    """
    if env_chunk <= 0 or data.shape[0] <= env_chunk:
        return adv_module(data)
    value_key = adv_module.tensor_keys.value
    keys = (adv_module.tensor_keys.advantage, adv_module.tensor_keys.value_target, value_key, ("next", value_key))
    num_chunks, outputs = 0, {}
    for chunk in data.split(env_chunk, dim=0):
        out = adv_module(chunk)
        num_chunks += 1
        for key in keys:
            tensor = out.get(key, default=None)
            if tensor is not None:
                outputs.setdefault(key, []).append(tensor)
    for key, tensors in outputs.items():
        if len(tensors) == num_chunks:
            data.set(key, torch.cat(tensors, dim=0))
    return data


def _metrics(data: TensorDictBase, done: torch.Tensor, prefix: str) -> dict[str, float]:
    if not bool(done.any()):
        return {}
    completed_return = data["episode_reward"] + data["next", "reward"]
    completed_length = data["step_count"] + 1
    outcomes = {term: data["next", "outcome", term] for term in OUTCOME_TERMS}
    metrics = {
        f"{prefix}/episode_return": completed_return[done].mean().item(),
        f"{prefix}/episode_length": completed_length[done].float().mean().item(),
        f"{prefix}/episodes": int(done.sum()),
    }
    metrics.update({f"{prefix}/{term}_rate": rate for term, rate in outcome_rates(done, outcomes).items()})
    return metrics


def episode_metrics(data: TensorDictBase, prefix: str) -> dict[str, float]:
    """Return, length and outcome rates over every episode that finished in ``data`` (shape N x T)."""
    return _metrics(data, data["next", "done"], prefix)


def first_episode_metrics(data: TensorDictBase, prefix: str) -> dict[str, float]:
    """As ``episode_metrics`` but only each env's first finished episode (for evaluation after a reset)."""
    done = data["next", "done"]
    first = done & (done.long().cumsum(dim=1) == 1)
    metrics = _metrics(data, first, prefix)
    if metrics:
        metrics[f"{prefix}/finished_fraction"] = first.any(dim=1).float().mean().item()
    return metrics


class SuccessStreak:
    """Consecutive iterations whose training success rate reached ``threshold``, for early stopping."""

    def __init__(self, threshold: float, required: int):
        self.threshold, self.required, self.count = threshold, int(required), 0

    def update(self, success_rate: float | None) -> bool:
        """Add one iteration's success rate (None: no episode finished, streak unchanged); True once reached."""
        if success_rate is not None:
            self.count = self.count + 1 if success_rate >= self.threshold else 0
        return self.count >= self.required


EVAL_KEYS = ["episode_reward", "step_count", ("next", "done"), ("next", "reward"), ("next", "outcome")]


def slim_eval_batch(batch: TensorDictBase, pixel_keys, video: bool):
    """Keep only what evaluation metrics need; optionally env 0's newest camera frames (CPU uint8)."""
    slim = batch.select(*EVAL_KEYS, strict=False).clone()
    frames = None
    if video:
        frames = {key: batch.get(key)[0, ..., -3:].clamp(0, 255).to(torch.uint8).cpu() for key in pixel_keys}
    return slim, frames


def policy_view_video(frames: list[dict], pixel_keys, upscale: int) -> np.ndarray:
    """Concatenate per-batch frames in time, cameras side by side: (T, 3, H*up, n_cams*W*up) uint8."""
    per_cam = [torch.cat([f[key] for f in frames], dim=0) for key in pixel_keys]  # (T, H, W, 3)
    video = torch.cat(per_cam, dim=2)
    video = video.repeat_interleave(upscale, dim=1).repeat_interleave(upscale, dim=2)
    return video.permute(0, 3, 1, 2).contiguous().numpy()
