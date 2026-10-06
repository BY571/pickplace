"""Offline RL from recorded camera shards: transition sampling, pixel networks, student evaluation.

Everything here is simulator-free except ``make_student_env`` / ``evaluate_student``, which import
``pickplace.torchrl_env`` lazily (the Isaac app must already be launched when they are called).

**Transitions.** A shard is time-major: row ``i`` and row ``i + stride`` (``stride`` = the collection's
``num_envs``) are consecutive steps of the same sub-env, and next observations are not stored. So a
transition is a *pair of rows*, and the set of legal pairs is precomputed once per shard as an index
tensor (``transition_index``) rather than re-scanned per batch:

* **ongoing**: row ``i`` has no done flag and ``i + stride`` exists -> ``next_obs`` = row ``i + stride``,
  ``terminated = False``.
* **terminal**: row ``i`` is ``terminated`` and not ``truncated`` -> the episode ended here, so there is no
  successor state and none is needed: the bootstrap is masked by ``terminated = True``. ``next_obs`` is row
  ``i`` itself, a placeholder multiplied by zero in the TD target. These pairs are kept (config
  ``include_terminals``) because the success bonus lives entirely on them: with ``simple_v3b`` the terminal
  reward is 150 of a ~182 mean episode return, so dropping terminal rows would hide 80%+ of the reward
  signal from every value-based algorithm.
* **truncated** rows are dropped: the episode did not end, but its next observation is genuinely not in the
  shard, so neither bootstrapping nor masking would be correct.

No pair ever spans an episode boundary: an ongoing pair's two rows are in the same episode, and a terminal
pair's "successor" is the row itself.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torch import nn
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from pickplace.datasets import STORAGE_DIR, shard_manifest
from pickplace.keys import expand_in_keys

#: Student inputs for the first runs: both 84 px camera views, nothing else.
CAMERA_KEYS: tuple[tuple[str, str], ...] = (("pixels", "overview_rgb"), ("pixels", "wrist_rgb"))


def as_key(key) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


def resolve_obs_keys(shard_path, in_keys: Sequence) -> list[tuple[str, ...]]:
    """Expand ``network.in_keys`` group names (e.g. ``"proprio"``) into the leaves a shard actually stores.

    Builds a spec from one memmapped row with ``make_composite_from_td`` -- no simulator needed -- and
    resolves it with ``pickplace.keys.expand_in_keys``, exactly as the pixel-PPO pipeline resolves its own
    ``network.actor_in_keys`` / ``critic_in_keys`` against a live env's observation spec. Explicit leaf keys
    (e.g. ``["pixels", "wrist_rgb"]``) pass through unchanged, so the cameras-only default is a no-op here.
    """
    from torchrl.envs.utils import make_composite_from_td

    row = TensorDict.load_memmap(str(Path(shard_path) / STORAGE_DIR))[0]
    spec = make_composite_from_td(row)
    return expand_in_keys(spec, in_keys)


# --------------------------------------------------------------------------------------------------
# Transition indexing
# --------------------------------------------------------------------------------------------------


def transition_index(data: TensorDictBase, stride: int, include_terminals: bool = True):
    """``(rows, terminal)``: the row index of every legal transition and whether it is a terminal one.

    ``rows`` is sorted and unique; ``terminal[k]`` says whether ``rows[k]``'s successor row is itself
    (terminal, bootstrap masked) rather than ``rows[k] + stride``. See the module docstring.
    """
    n = int(data.batch_size[0])
    done = data.get(("next", "done")).reshape(n).bool()
    terminated = data.get(("next", "terminated")).reshape(n).bool()
    truncated = data.get(("next", "truncated")).reshape(n).bool()
    ended = done | terminated | truncated

    has_successor = torch.zeros(n, dtype=torch.bool)
    if stride < n:
        has_successor[: n - stride] = True

    ongoing = (~ended) & has_successor
    terminal = (terminated & ~truncated) if include_terminals else torch.zeros(n, dtype=torch.bool)
    rows = (ongoing | terminal).nonzero(as_tuple=True)[0]
    return rows, terminal[rows]


class ShardTransitions:
    """Legal transitions of one shard, sampled straight out of its memory-mapped storage.

    Only the keys a batch needs are selected from the memmap, so a sample reads ~2 images per row instead
    of the whole 42 KB row. Images stay ``uint8`` here; the networks cast and scale them on the GPU.
    """

    def __init__(self, path, obs_keys: Sequence = CAMERA_KEYS, include_terminals: bool = True):
        self.path = Path(path)
        self.manifest = shard_manifest(self.path)
        self.stride = int(self.manifest["successor_stride"])
        self.obs_keys = [as_key(k) for k in obs_keys]
        data = TensorDict.load_memmap(self.path / STORAGE_DIR)
        self._cur = data.select(*self.obs_keys, "action", ("next", "reward"))
        self._nxt = data.select(*self.obs_keys)
        self.rows, self.terminal = transition_index(data, self.stride, include_terminals)
        self.frames = int(data.batch_size[0])

    def __len__(self) -> int:
        return int(self.rows.numel())

    def gather(self, pick: torch.Tensor) -> TensorDictBase:
        """Build the ``(obs, action, reward, next_obs, done)`` batch for positions ``pick`` into ``rows``."""
        rows = self.rows[pick]
        terminal = self.terminal[pick]
        cur = self._cur[rows]
        nxt = self._nxt[torch.where(terminal, rows, rows + self.stride)]
        flag = terminal.unsqueeze(-1)
        out = TensorDict({}, batch_size=[rows.numel()])
        nxt_td = TensorDict({}, batch_size=[rows.numel()])
        for key in self.obs_keys:
            out.set(key, cur.get(key))
            nxt_td.set(key, nxt.get(key))
        out.set("action", cur.get("action"))
        nxt_td.set("reward", cur.get(("next", "reward")))
        nxt_td.set("terminated", flag)
        nxt_td.set("done", flag.clone())
        out.set("next", nxt_td)
        return out

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> TensorDictBase:
        pick = torch.randint(len(self), (batch_size,), generator=generator)
        return self.gather(pick)

    def provenance(self) -> dict:
        """What this shard is, for a run/checkpoint manifest (no env config: it is long and identical)."""
        m = self.manifest
        return {
            "path": str(self.path),
            "name": m.get("name"),
            "frames": m.get("frames"),
            "transitions": len(self),
            "terminal_transitions": int(self.terminal.sum()),
            "successor_stride": self.stride,
            "noise_sigma": m.get("noise_sigma"),
            "seed": m.get("seed"),
            "checkpoint": m.get("checkpoint"),
            "checkpoint_sha256": m.get("checkpoint_sha256"),
            "reward_set": m.get("reward_set"),
            "reward_weights": m.get("reward_weights"),
            "git_commit": m.get("git_commit"),
            "success_rate": (m.get("stats") or {}).get("success_rate"),
        }


def split_batch(batch_size: int, proportions: Sequence[float]) -> list[int]:
    """Split ``batch_size`` over tiers by ``proportions`` (largest remainder, so the sum is exact)."""
    total = float(sum(proportions))
    if total <= 0:
        raise ValueError(f"Shard proportions must sum to something positive, got {list(proportions)}.")
    exact = [batch_size * p / total for p in proportions]
    counts = [int(x) for x in exact]
    for i in sorted(range(len(counts)), key=lambda i: exact[i] - counts[i], reverse=True)[: batch_size - sum(counts)]:
        counts[i] += 1
    return counts


class TransitionSampler:
    """Mixes several shards at fixed per-shard proportions into one batch.

    Tier mixing is done by sampling a fixed number of rows per shard and concatenating, not with
    ``ReplayBufferEnsemble``: a shard's legal rows are a precomputed subset and its successor row has to be
    gathered at ``+stride``, which no stock sampler/storage expresses, so the ensemble would need a custom
    sampler per shard anyway — strictly more machinery than ``split_batch`` + ``torch.cat``.
    """

    def __init__(
        self,
        shards: Sequence[str | Path],
        proportions: Sequence[float] | None = None,
        batch_size: int = 256,
        obs_keys: Sequence = CAMERA_KEYS,
        include_terminals: bool = True,
        pin_memory: bool = False,
        seed: int = 0,
    ):
        if not shards:
            raise ValueError("Pass at least one shard.")
        self.shards = [ShardTransitions(p, obs_keys, include_terminals) for p in shards]
        self.proportions = list(proportions) if proportions is not None else [1.0] * len(self.shards)
        if len(self.proportions) != len(self.shards):
            raise ValueError(f"Got {len(self.shards)} shards but {len(self.proportions)} proportions.")
        self.batch_size = int(batch_size)
        self.counts = split_batch(self.batch_size, self.proportions)
        self.obs_keys = [as_key(k) for k in obs_keys]
        self.pin_memory = pin_memory
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    def sample(self) -> TensorDictBase:
        """One CPU batch (images still ``uint8``); the caller moves it to the GPU."""
        parts = [s.sample(c, self.generator) for s, c in zip(self.shards, self.counts) if c]
        batch = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return batch.pin_memory() if self.pin_memory else batch

    def provenance(self) -> list[dict]:
        return [{**s.provenance(), "proportion": p, "rows_per_batch": c}
                for s, p, c in zip(self.shards, self.proportions, self.counts)]


def prefetch(sample_fn, depth: int = 4, workers: int = 2) -> Iterator:
    """Endlessly yield ``sample_fn()`` results produced by background threads.

    The memmap reads (~20 MB/batch of random pages) release the GIL, so they overlap the GPU update
    instead of serialising with it.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = deque(pool.submit(sample_fn) for _ in range(depth))
    try:
        while True:
            future = pending.popleft()
            pending.append(pool.submit(sample_fn))
            yield future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------------------------------
# Pixel encoder and network factories
# --------------------------------------------------------------------------------------------------


class ConvEncoder(nn.Module):
    """H x W x C images (uint8 or float in [0, 255]) -> ELU(Linear(embed)) over a Nature-CNN-style trunk.

    Same shape as ``sota-implementations/ppo/utils_pixels.py``'s encoder (which the pipeline may not
    import from). The cast to float and the /255 scaling happen here, i.e. on whatever device the batch
    is already on — the GPU.
    """

    def __init__(self, shape, channels, kernels, strides, embed_dim: int):
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


class PixelNet(nn.Module):
    """One CNN per image key + one shared MLP branch over vector keys, fused into an output head.

    ``shapes`` lists each key's trailing per-env shape, in the same order as the module's ``in_keys``: a
    3-D shape (H, W, C) gets its own ``ConvEncoder``; anything else is flattened and concatenated into one
    vector branch (``Linear`` -> ``LayerNorm`` -> ``ELU``) before the fusion MLP -- same shape as
    ``sota-implementations/ppo/utils_pixels.py``'s ``PixelsNet`` (which the pipeline may not import from).
    ``extra_dim`` appends one more *raw*, unembedded vector after the fused features -- used for the action
    in a Q-network, which should not be normalised like an observation.

    Each network (actor, every critic, the value net) gets its own encoders: the actor's objective and the
    critics' TD targets pull an encoder in different directions, and a shared trunk would need loss-specific
    gradient stopping to stay stable. A handful of 84 px Nature CNNs are cheap enough that sharing buys little.
    """

    def __init__(self, shapes: Sequence[tuple[int, ...]], network_cfg, out_dim: int, out_gain: float,
                 extra_dim: int = 0):
        super().__init__()
        self.shapes = [tuple(s) for s in shapes]
        self.is_image = [len(s) == 3 for s in self.shapes]
        self.cnns = nn.ModuleList(
            ConvEncoder(s, list(network_cfg.cnn_channels), list(network_cfg.cnn_kernels),
                        list(network_cfg.cnn_strides), int(network_cfg.image_embed))
            for s, image in zip(self.shapes, self.is_image) if image
        )
        vec_dim = sum(math.prod(s) for s, image in zip(self.shapes, self.is_image) if not image)
        self.vec = (
            nn.Sequential(nn.Linear(vec_dim, int(network_cfg.proprio_embed)),
                          nn.LayerNorm(int(network_cfg.proprio_embed)), nn.ELU())
            if vec_dim else None
        )
        self.extra_dim = int(extra_dim)
        fused = (len(self.cnns) * int(network_cfg.image_embed)
                 + (int(network_cfg.proprio_embed) if vec_dim else 0) + self.extra_dim)
        self.mlp = MLP(in_features=fused, out_features=out_dim, num_cells=list(network_cfg.fusion),
                       activation_class=nn.ELU)
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        n = len(self.shapes)
        cnn_iter = iter(self.cnns)
        images, vectors = [], []
        for x, shape, image in zip(xs[:n], self.shapes, self.is_image):
            if image:
                images.append(next(cnn_iter)(x))
            else:
                vectors.append(x.float().reshape(*x.shape[: x.dim() - len(shape)], -1))
        parts = images + ([self.vec(torch.cat(vectors, dim=-1))] if self.vec is not None else [])
        parts += [x.float().reshape(*x.shape[:-1], -1) for x in xs[n:]]
        return self.mlp(torch.cat(parts, dim=-1))


class _ActorNet(nn.Module):
    """PixelNet -> (loc, state-independent scale); several positional inputs, unlike nn.Sequential."""

    def __init__(self, body: PixelNet, action_dim: int, scale_lb: float = 1e-4):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(action_dim, scale_lb=scale_lb)

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def make_actor(shapes, obs_keys, action_dim: int, network_cfg, device,
               scale_lb: float = 1e-4) -> ProbabilisticActor:
    """TanhNormal actor on [-1, 1] with a state-independent scale (as in the teacher and pixel PPO).

    ``scale_lb`` floors that scale. The default lets it collapse, which is what cloning a near-deterministic
    teacher wants; an algorithm that learns a SAC entropy temperature needs a floor instead (0.1 is the
    usual value), or the policy's entropy falls below ``target_entropy`` and the temperature diverges. The
    floor never changes the *evaluated* action, which is the distribution's mode.
    """
    body = PixelNet(shapes, network_cfg, action_dim, out_gain=0.01)
    module = TensorDictModule(_ActorNet(body, action_dim, scale_lb), in_keys=[as_key(k) for k in obs_keys],
                              out_keys=["loc", "scale"])
    from torchrl.envs import ExplorationType

    actor = ProbabilisticActor(
        module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=False,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return actor.to(device)


def make_deterministic_actor(shapes, obs_keys, action_dim: int, network_cfg, device):
    """The same body as ``make_actor`` with a tanh head instead of a distribution: s -> a in [-1, 1].

    TD3+BC is a deterministic-policy algorithm: ``TD3BCLoss`` reads ``action`` straight out of the actor
    (and adds the target-policy smoothing noise itself), so it cannot take the stochastic ``make_actor``.
    Construction follows TorchRL's ``sota-implementations/td3_bc/utils.py`` (module -> ``TanhModule``);
    only the body differs, and it is the same ``PixelNet`` every other algorithm here uses.
    """
    from tensordict.nn import TensorDictSequential
    from torchrl.modules import TanhModule

    body = PixelNet(shapes, network_cfg, action_dim, out_gain=0.01)
    module = TensorDictModule(body, in_keys=[as_key(k) for k in obs_keys], out_keys=["param"])
    tanh = TanhModule(in_keys=["param"], out_keys=["action"], low=-1.0, high=1.0)
    return TensorDictSequential(module, tanh).to(device)


def make_qvalue(shapes, obs_keys, action_dim: int, network_cfg, device) -> TensorDictModule:
    """Q(s, a): the observation encoders plus the raw action, concatenated into the fusion MLP."""
    body = PixelNet(shapes, network_cfg, 1, out_gain=1.0, extra_dim=action_dim)
    keys = [as_key(k) for k in obs_keys] + ["action"]
    return TensorDictModule(body, in_keys=keys, out_keys=["state_action_value"]).to(device)


def make_value(shapes, obs_keys, network_cfg, device) -> ValueOperator:
    """V(s) over the observation encoders."""
    body = PixelNet(shapes, network_cfg, 1, out_gain=1.0)
    return ValueOperator(body, in_keys=[as_key(k) for k in obs_keys], out_keys=["state_value"]).to(device)


# --------------------------------------------------------------------------------------------------
# Evaluation-time observation perturbations (simulator-free)
# --------------------------------------------------------------------------------------------------

#: Images (camera views and shards alike) carry values in [0, 255], whatever their dtype.
IMAGE_MAX = 255.0

#: Which physical quantity each ``proprio`` leaf carries, i.e. which noise scale applies to it.
#: ``ee_quat`` (a unit quaternion — perturbing it needs a rotation, not an additive sigma) and
#: ``last_action`` (the policy's own previous output, not a sensor reading) are deliberately absent.
PROPRIO_NOISE_FIELDS: dict[str, str] = {
    "joint_pos_rel": "joint_pos",
    "gripper_pos": "joint_pos",
    "joint_vel_rel": "joint_vel",
    "ee_pos": "ee_pos",
}


class ObservationPerturbation:
    """Degrade an observation *after* the env produced it and *before* the policy sees it.

    The simulation is untouched: this only rewrites the tensordict handed to the policy (see
    ``evaluate_student``'s ``perturb=`` argument), so it measures how robust a *trained* policy is to a
    sensor that is noisier than the one it was trained on. Nothing here needs a simulator.

    Every knob defaults to its identity value, and a perturbation whose knobs are all at their defaults
    (``is_noop``) returns the input tensordict unchanged — severity 0 is an exact no-op, not "noise with
    sigma 0".

    Image knobs act on the image observations (any 3-D key, in [0, 255] whatever the dtype) and are applied
    in this order, with the result clamped back into [0, 255] and cast back to the input dtype:

    * ``image_noise`` — additive Gaussian, sigma in [0, 255] units (sensor read noise).
    * ``image_gain`` / ``image_offset`` — ``gain * x + offset``, i.e. contrast and brightness.
    * ``blur_sigma`` — separable Gaussian blur with reflect padding, kernel ``blur_kernel`` (odd; defaults
      to ``2 * ceil(2 * sigma) + 1``) — defocus.
    * ``occlusion`` — a black square patch covering that fraction of the image area, on the cameras named
      by ``occlusion_keys`` (leaf names, default: every image key). The patch position is drawn **once per
      sub-env** on first use and then held fixed for the whole evaluation: a static occluder (dirt on the
      lens, a fixture in the way), not a patch that flickers to a new place every control step.

    Proprio knobs add Gaussian noise to the ``proprio`` leaves listed in ``PROPRIO_NOISE_FIELDS``, in
    physical units: ``joint_pos_sigma`` rad on joint positions and the gripper opening, ``joint_vel_sigma``
    rad/s on joint velocities, ``ee_pos_sigma`` m on the end-effector position.

    Randomness comes from a dedicated per-device ``torch.Generator``, so perturbing does not advance the
    global RNG and the sequence of episodes an evaluation sees is identical across conditions.
    """

    def __init__(
        self,
        image_keys: Sequence = (),
        proprio_keys: Sequence = (),
        *,
        name: str = "clean",
        image_noise: float = 0.0,
        image_gain: float = 1.0,
        image_offset: float = 0.0,
        blur_sigma: float = 0.0,
        blur_kernel: int = 0,
        occlusion: float = 0.0,
        occlusion_keys: Sequence | None = None,
        joint_pos_sigma: float = 0.0,
        joint_vel_sigma: float = 0.0,
        ee_pos_sigma: float = 0.0,
        seed: int = 0,
    ):
        self.name = str(name)
        self.image_keys = [as_key(k) for k in image_keys]
        self.proprio_keys = [as_key(k) for k in proprio_keys]
        self.image_noise = float(image_noise)
        self.image_gain = float(image_gain)
        self.image_offset = float(image_offset)
        self.blur_sigma = float(blur_sigma)
        self.blur_kernel = int(blur_kernel) or (2 * math.ceil(2 * self.blur_sigma) + 1 if self.blur_sigma else 0)
        if self.blur_kernel and self.blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be odd, got {self.blur_kernel}.")
        self.occlusion = float(occlusion)
        self.occlusion_keys = (
            self.image_keys if occlusion_keys is None
            else [k for k in self.image_keys if k[-1] in {as_key(o)[-1] for o in occlusion_keys}]
        )
        self.sigmas = {"joint_pos": float(joint_pos_sigma), "joint_vel": float(joint_vel_sigma),
                       "ee_pos": float(ee_pos_sigma)}
        self.noisy_proprio_keys = [k for k in self.proprio_keys
                                   if self.sigmas.get(PROPRIO_NOISE_FIELDS.get(k[-1], ""), 0.0) > 0.0]
        self.seed = int(seed)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._patches: dict[tuple, torch.Tensor] = {}

    # -- introspection -----------------------------------------------------------------------------

    @property
    def perturbs_images(self) -> bool:
        return bool(self.image_keys) and bool(
            self.image_noise > 0.0 or self.image_gain != 1.0 or self.image_offset != 0.0
            or (self.blur_sigma > 0.0 and self.blur_kernel > 1) or (self.occlusion > 0.0 and self.occlusion_keys)
        )

    @property
    def is_noop(self) -> bool:
        return not self.perturbs_images and not self.noisy_proprio_keys

    def summary(self) -> dict:
        """The knobs that are actually doing something, for a result record."""
        knobs = {"image_noise": self.image_noise, "image_gain": self.image_gain,
                 "image_offset": self.image_offset, "blur_sigma": self.blur_sigma,
                 "blur_kernel": self.blur_kernel, "occlusion": self.occlusion,
                 **{f"{q}_sigma": s for q, s in self.sigmas.items()}}
        defaults = {"image_gain": 1.0}
        active = {k: v for k, v in knobs.items() if v != defaults.get(k, 0.0)}
        if self.occlusion > 0.0:
            active["occlusion_keys"] = [k[-1] for k in self.occlusion_keys]
        return {"name": self.name, **active}

    # -- internals ---------------------------------------------------------------------------------

    def _generator(self, device: torch.device) -> torch.Generator:
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device).manual_seed(self.seed)
            self._generators[device] = gen
        return gen

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape, generator=self._generator(x.device), device=x.device, dtype=torch.float32)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur of an ``(N, H, W, C)`` float image, reflect-padded."""
        k, radius = self.blur_kernel, self.blur_kernel // 2
        grid = torch.arange(k, device=x.device, dtype=torch.float32) - radius
        weight = torch.exp(-0.5 * (grid / self.blur_sigma) ** 2)
        weight = weight / weight.sum()
        n, h, w, c = x.shape
        img = x.permute(0, 3, 1, 2)  # NCHW
        for shape in ((1, 1, 1, k), (1, 1, k, 1)):
            pad = (radius, radius, 0, 0) if shape[-1] == k else (0, 0, radius, radius)
            img = nn.functional.conv2d(nn.functional.pad(img, pad, mode="reflect"),
                                       weight.view(*shape).expand(c, 1, *shape[2:]), groups=c)
        return img.permute(0, 2, 3, 1)

    def _patch(self, key: tuple, x: torch.Tensor) -> torch.Tensor:
        """A fixed-per-sub-env boolean mask of the occluded pixels, ``(N, H, W, 1)``."""
        n, h, w = x.shape[0], x.shape[-3], x.shape[-2]
        cached = self._patches.get((key, n, h, w))
        if cached is not None:
            return cached
        side_h = max(1, int(round(math.sqrt(self.occlusion) * h)))
        side_w = max(1, int(round(math.sqrt(self.occlusion) * w)))
        gen = self._generator(x.device)
        top = torch.randint(0, h - side_h + 1, (n, 1, 1), generator=gen, device=x.device)
        left = torch.randint(0, w - side_w + 1, (n, 1, 1), generator=gen, device=x.device)
        rows = torch.arange(h, device=x.device).view(1, h, 1)
        cols = torch.arange(w, device=x.device).view(1, 1, w)
        mask = ((rows >= top) & (rows < top + side_h) & (cols >= left) & (cols < left + side_w)).unsqueeze(-1)
        self._patches[(key, n, h, w)] = mask
        return mask

    def _degrade_image(self, key: tuple, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        if self.image_noise > 0.0:
            out = out + self.image_noise * self._randn_like(out)
        if self.image_gain != 1.0 or self.image_offset != 0.0:
            out = self.image_gain * out + self.image_offset
        if self.blur_sigma > 0.0 and self.blur_kernel > 1:
            out = self._blur(out)
        if self.occlusion > 0.0 and key in self.occlusion_keys:
            out = out.masked_fill(self._patch(key, out), 0.0)
        out = out.clamp(0.0, IMAGE_MAX)
        return out.round().to(x.dtype) if not x.dtype.is_floating_point else out.to(x.dtype)

    # -- application -------------------------------------------------------------------------------

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Return ``td`` with the perturbed observations (a shallow copy; ``td`` itself is untouched)."""
        if self.is_noop:
            return td
        out = td.copy()
        if self.perturbs_images:
            for key in self.image_keys:
                out.set(key, self._degrade_image(key, td.get(key)))
        for key in self.noisy_proprio_keys:
            x = td.get(key)
            sigma = self.sigmas[PROPRIO_NOISE_FIELDS[key[-1]]]
            out.set(key, (x.float() + sigma * self._randn_like(x)).to(x.dtype))
        return out


def make_perturbation(cfg: Mapping, obs_keys: Sequence, obs_shapes: Sequence, seed: int = 0
                      ) -> ObservationPerturbation:
    """Build an :class:`ObservationPerturbation` from a plain config mapping and the policy's own keys.

    ``cfg`` holds the constructor's knobs plus ``name``; keys with a 3-D ``obs_shapes`` entry are the
    images and the rest are the vector (proprio) inputs, exactly as ``PixelNet`` splits them.
    """
    keys = [as_key(k) for k in obs_keys]
    shapes = [tuple(s) for s in obs_shapes]
    images = [k for k, s in zip(keys, shapes, strict=True) if len(s) == 3]
    vectors = [k for k, s in zip(keys, shapes, strict=True) if len(s) != 3]
    return ObservationPerturbation(images, vectors, seed=seed, **{k: v for k, v in dict(cfg).items()})


# --------------------------------------------------------------------------------------------------
# Online evaluation of a student (needs a launched Isaac app)
# --------------------------------------------------------------------------------------------------


def student_env_cfg(manifest: Mapping, num_envs: int, image_size: int, seed: int) -> dict:
    """The shard's own env config with cameras on, one frame per camera and the eval scale."""
    return {
        **manifest["env"],
        "num_envs": int(num_envs),
        "cameras": True,
        "image_size": [int(image_size), int(image_size)],
        "frame_stack": 1,
        "seed": int(seed),
    }


def make_student_env(manifest: Mapping, num_envs: int, image_size: int, seed: int = 0):
    """Build the camera env a student is evaluated in, from the shard manifest's env config."""
    from pickplace.torchrl_env import make_env

    return make_env(student_env_cfg(manifest, num_envs, image_size, seed))


_EVAL_KEYS = ["episode_reward", "episode_reward_terms", "step_count",
              ("next", "done"), ("next", "reward"), ("next", "reward_terms"), ("next", "outcome")]


def evaluate_student(policy, env, prefix: str = "eval", perturb=None) -> dict[str, float]:
    """Run ``policy`` deterministically for ``max_episode_length + 1`` steps; score each env's first episode.

    Same protocol as ``pipeline/0_state_teacher/evaluate.py`` (which scores the teacher), so student and
    teacher numbers are directly comparable. Returns ``pickplace.training.first_episode_metrics``.

    ``perturb`` (an :class:`ObservationPerturbation`, or any ``td -> td``) degrades the observation between
    the env and the policy: the policy acts on ``perturb(td)`` but the env keeps stepping the clean ``td``,
    so only the policy's *input* is degraded and the simulation is bit-for-bit the same run.
    """
    from torchrl.envs import ExplorationType, set_exploration_type

    from pickplace.training import first_episode_metrics

    steps = int(env.base_env._env.unwrapped.max_episode_length) + 1
    td = env.reset()
    records = []
    was_training = policy.training
    policy.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(steps):
            if perturb is None:
                td = policy(td)
            else:
                td.set("action", policy(perturb(td)).get("action"))
            stepped, td = env.step_and_maybe_reset(td)
            records.append(stepped.select(*_EVAL_KEYS, strict=False).clone())
    if was_training:
        policy.train()
    metrics = first_episode_metrics(torch.stack(records, dim=1), prefix)
    return {k[len(prefix) + 1:]: v for k, v in metrics.items()}
