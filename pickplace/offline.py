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

#: Student inputs for the first runs: both 84 px camera views, nothing else.
CAMERA_KEYS: tuple[tuple[str, str], ...] = (("pixels", "overview_rgb"), ("pixels", "wrist_rgb"))


def as_key(key) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


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
    """One CNN per camera, concatenated with any vector inputs (e.g. the action) -> fusion MLP -> head.

    Each network (actor, every critic, the value net) gets its own encoders: the actor's objective and the
    critics' TD targets pull an encoder in different directions, and a shared trunk would need loss-specific
    gradient stopping to stay stable. Four 84 px Nature CNNs are cheap enough that sharing buys little.
    """

    def __init__(self, image_shapes: Sequence[tuple[int, int, int]], network_cfg, out_dim: int,
                 out_gain: float, vector_dim: int = 0):
        super().__init__()
        self.image_shapes = [tuple(s) for s in image_shapes]
        self.cnns = nn.ModuleList(
            ConvEncoder(s, list(network_cfg.cnn_channels), list(network_cfg.cnn_kernels),
                        list(network_cfg.cnn_strides), int(network_cfg.image_embed))
            for s in self.image_shapes
        )
        self.vector_dim = int(vector_dim)
        fused = len(self.cnns) * int(network_cfg.image_embed) + self.vector_dim
        self.mlp = MLP(in_features=fused, out_features=out_dim, num_cells=list(network_cfg.fusion),
                       activation_class=nn.ELU)
        linears = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        for layer in linears:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linears[-1].weight, out_gain)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        n = len(self.cnns)
        parts = [cnn(x) for cnn, x in zip(self.cnns, xs[:n])]
        parts += [x.float().reshape(*x.shape[:-1], -1) for x in xs[n:]]
        return self.mlp(torch.cat(parts, dim=-1))


class _ActorNet(nn.Module):
    """PixelNet -> (loc, state-independent scale); several positional inputs, unlike nn.Sequential."""

    def __init__(self, body: PixelNet, action_dim: int):
        super().__init__()
        self.body = body
        self.scale = AddStateIndependentNormalScale(action_dim, scale_lb=1e-4)

    def forward(self, *xs: torch.Tensor):
        return self.scale(self.body(*xs))


def make_actor(image_shapes, obs_keys, action_dim: int, network_cfg, device) -> ProbabilisticActor:
    """TanhNormal actor on [-1, 1] with a state-independent scale (as in the teacher and pixel PPO)."""
    body = PixelNet(image_shapes, network_cfg, action_dim, out_gain=0.01)
    module = TensorDictModule(_ActorNet(body, action_dim), in_keys=[as_key(k) for k in obs_keys],
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


def make_qvalue(image_shapes, obs_keys, action_dim: int, network_cfg, device) -> TensorDictModule:
    """Q(s, a): the camera encoders plus the action, concatenated into the fusion MLP."""
    body = PixelNet(image_shapes, network_cfg, 1, out_gain=1.0, vector_dim=action_dim)
    keys = [as_key(k) for k in obs_keys] + ["action"]
    return TensorDictModule(body, in_keys=keys, out_keys=["state_action_value"]).to(device)


def make_value(image_shapes, obs_keys, network_cfg, device) -> ValueOperator:
    """V(s) over the camera encoders."""
    body = PixelNet(image_shapes, network_cfg, 1, out_gain=1.0)
    return ValueOperator(body, in_keys=[as_key(k) for k in obs_keys], out_keys=["state_value"]).to(device)


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


def evaluate_student(policy, env, prefix: str = "eval") -> dict[str, float]:
    """Run ``policy`` deterministically for ``max_episode_length + 1`` steps; score each env's first episode.

    Same protocol as ``pipeline/0_state_teacher/evaluate.py`` (which scores the teacher), so student and
    teacher numbers are directly comparable. Returns ``pickplace.training.first_episode_metrics``.
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
            td = policy(td)
            stepped, td = env.step_and_maybe_reset(td)
            records.append(stepped.select(*_EVAL_KEYS, strict=False).clone())
    if was_training:
        policy.train()
    metrics = first_episode_metrics(torch.stack(records, dim=1), prefix)
    return {k[len(prefix) + 1:]: v for k, v in metrics.items()}
