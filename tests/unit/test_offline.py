"""The transition index is the piece most likely to be silently wrong, so it is tested exhaustively."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from pickplace.artifacts import write_json
from pickplace.datasets import STORAGE_DIR
from pickplace.offline import (
    CAMERA_KEYS,
    PixelNet,
    ShardTransitions,
    TransitionSampler,
    resolve_obs_keys,
    split_batch,
    student_env_cfg,
    transition_index,
)

STRIDE, STEPS = 3, 5
FRAMES = STRIDE * STEPS  # time-major: row i is sub-env i % STRIDE at step i // STRIDE

# sub-env 0 terminates at step 1 (row 3); sub-env 1 is truncated at step 2 (row 7); sub-env 2 never ends.
TERMINATED_ROWS, TRUNCATED_ROWS = [3], [7]


def _fake_shard(path, terminated=TERMINATED_ROWS, truncated=TRUNCATED_ROWS, frames=FRAMES, stride=STRIDE):
    """A shard directory in collect.py's layout, with known episode boundaries."""
    flags = {name: torch.zeros(frames, 1, dtype=torch.bool) for name in ("terminated", "truncated")}
    flags["terminated"][terminated] = True
    flags["truncated"][truncated] = True
    done = flags["terminated"] | flags["truncated"]
    td = TensorDict(
        {
            # Row r's images are filled with the value r, so a gathered pair identifies its two rows.
            "pixels": TensorDict(
                {name: (torch.arange(frames, dtype=torch.uint8).reshape(frames, 1, 1, 1) + offset).expand(frames, 2, 2, 3).clone()
                 for offset, name in enumerate(("overview_rgb", "wrist_rgb"))},
                batch_size=[frames],
            ),
            "proprio": TensorDict({"ee_pos": torch.randn(frames, 3)}, batch_size=[frames]),
            "action": torch.arange(frames, dtype=torch.float32).reshape(frames, 1).expand(frames, 7).clone(),
            "next": TensorDict(
                {"reward": torch.arange(frames, dtype=torch.float32).reshape(frames, 1),
                 "reward_terms": torch.randn(frames, 13), "done": done, **flags},
                batch_size=[frames],
            ),
        },
        batch_size=[frames],
    )
    td.memmap_(str(path / STORAGE_DIR))
    write_json(path / "manifest.json", {"name": path.name, "frames": frames, "successor_stride": stride,
                                        "env": {"num_envs": stride, "cameras": False}, "stats": {"success_rate": 0.5}})
    return td


def _index(path, include_terminals=True):
    td = TensorDict.load_memmap(str(path / STORAGE_DIR))
    rows, terminal = transition_index(td, STRIDE, include_terminals)
    return rows.tolist(), terminal.tolist()


def test_transition_index_is_exactly_the_legal_pairs(tmp_path):
    _fake_shard(tmp_path)
    rows, terminal = _index(tmp_path)

    # Ongoing: no done flag and a successor row inside the shard -> rows 0..11 minus the two ended rows.
    # Terminal: row 3 (terminated, not truncated). Row 7 is truncated, so its next observation is missing.
    assert rows == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    assert terminal == [r == 3 for r in rows]


def test_transition_index_can_drop_terminals(tmp_path):
    _fake_shard(tmp_path)
    rows, terminal = _index(tmp_path, include_terminals=False)
    assert rows == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    assert not any(terminal)


def test_transition_index_excludes_truncations_and_the_rows_without_a_successor(tmp_path):
    _fake_shard(tmp_path, terminated=[0, 4], truncated=[5, 11])
    rows, terminal = _index(tmp_path)
    # Ongoing 1,2,3,6,7,8,9,10 + terminal 0,4. Truncated 5 and 11 are dropped (no next observation stored),
    # and rows 12..14 have no successor row in the shard.
    assert rows == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert [r for r, t in zip(rows, terminal) if t] == [0, 4]


def test_transition_index_drops_a_row_that_is_both_terminated_and_truncated(tmp_path):
    _fake_shard(tmp_path, terminated=[6], truncated=[6])
    rows, _ = _index(tmp_path)
    assert 6 not in rows


def test_transition_index_handles_a_shard_shorter_than_one_stride(tmp_path):
    _fake_shard(tmp_path, terminated=[], truncated=[], frames=2, stride=STRIDE)
    td = TensorDict.load_memmap(str(tmp_path / STORAGE_DIR))
    rows, _ = transition_index(td, STRIDE)
    assert rows.numel() == 0


def test_gather_returns_the_successor_row_and_never_crosses_a_boundary(tmp_path):
    _fake_shard(tmp_path)
    shard = ShardTransitions(tmp_path)
    batch = shard.gather(torch.arange(len(shard)))  # every legal pair, in index order

    rows = shard.rows
    terminal = shard.terminal
    overview = batch["pixels", "overview_rgb"][:, 0, 0, 0].long()
    next_overview = batch["next", "pixels", "overview_rgb"][:, 0, 0, 0].long()
    wrist = batch["next", "pixels", "wrist_rgb"][:, 0, 0, 0].long()

    assert torch.equal(overview, rows)
    # The successor of an ongoing pair is row + stride; a terminal pair's placeholder successor is itself.
    expected_next = torch.where(terminal, rows, rows + STRIDE)
    assert torch.equal(next_overview, expected_next)
    assert torch.equal(wrist, expected_next + 1)  # per-camera offset: really the successor's own frame
    # The two rows of an ongoing pair are consecutive steps of the same sub-env, i.e. in the same episode.
    assert torch.equal((expected_next - rows) % STRIDE, torch.zeros_like(rows))

    assert torch.equal(batch["action"][:, 0].long(), rows)
    assert torch.equal(batch["next", "reward"].squeeze(-1).long(), rows)  # the reward of the *current* row
    assert torch.equal(batch["next", "terminated"].squeeze(-1), terminal)
    assert torch.equal(batch["next", "done"].squeeze(-1), terminal)
    assert batch["pixels", "overview_rgb"].dtype == torch.uint8  # scaled to [0, 1] on the GPU, not here


def test_sampled_pairs_are_always_legal(tmp_path):
    _fake_shard(tmp_path)
    shard = ShardTransitions(tmp_path)
    legal = {(int(r), bool(t)) for r, t in zip(shard.rows, shard.terminal)}

    batch = shard.sample(512, torch.Generator().manual_seed(0))
    rows = batch["pixels", "overview_rgb"][:, 0, 0, 0].long()
    next_rows = batch["next", "pixels", "overview_rgb"][:, 0, 0, 0].long()
    terminal = batch["next", "terminated"].squeeze(-1)

    assert batch.shape == (512,)
    assert {(int(r), bool(t)) for r, t in zip(rows, terminal)} <= legal
    assert torch.equal(next_rows, torch.where(terminal, rows, rows + STRIDE))
    assert set(rows.tolist()) == {r for r, _ in legal}  # every legal pair is reachable


def test_shard_length_and_provenance(tmp_path):
    _fake_shard(tmp_path)
    shard = ShardTransitions(tmp_path)
    assert len(shard) == 11
    provenance = shard.provenance()
    assert provenance["transitions"] == 11 and provenance["terminal_transitions"] == 1
    assert provenance["successor_stride"] == STRIDE and provenance["success_rate"] == 0.5


def test_custom_obs_keys(tmp_path):
    _fake_shard(tmp_path)
    shard = ShardTransitions(tmp_path, obs_keys=[("pixels", "wrist_rgb"), ("proprio", "ee_pos")])
    batch = shard.sample(4, torch.Generator().manual_seed(0))
    assert set(batch["next"].keys()) == {"pixels", "proprio", "reward", "terminated", "done"}
    assert ("pixels", "overview_rgb") not in batch.keys(True, True)


@pytest.mark.parametrize(
    ("size", "proportions", "expected"),
    [(256, [1.0], [256]), (256, [0.5, 0.5], [128, 128]), (10, [2.0, 1.0], [7, 3]), (4, [1, 1, 1], [2, 1, 1])],
)
def test_split_batch_is_exact(size, proportions, expected):
    assert split_batch(size, proportions) == expected
    assert sum(split_batch(size, proportions)) == size


def test_sampler_mixes_shards_at_the_given_proportions(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        _fake_shard(tmp_path / name)
    sampler = TransitionSampler([tmp_path / "a", tmp_path / "b"], proportions=[0.75, 0.25], batch_size=8, seed=0)

    assert sampler.counts == [6, 2]
    assert len(sampler) == 22
    batch = sampler.sample()
    assert batch.shape == (8,) and set(batch.keys()) == {"pixels", "action", "next"}
    assert [p["proportion"] for p in sampler.provenance()] == [0.75, 0.25]


def test_sampler_rejects_a_proportion_mismatch(tmp_path):
    _fake_shard(tmp_path)
    with pytest.raises(ValueError, match="proportions"):
        TransitionSampler([tmp_path], proportions=[1.0, 1.0])


def test_student_env_cfg_forces_cameras_on_and_a_single_frame():
    cfg = student_env_cfg({"env": {"num_envs": 512, "cameras": False, "task": "T", "frame_stack": 3}}, 64, 84, seed=7)
    assert cfg == {"task": "T", "num_envs": 64, "cameras": True, "image_size": [84, 84], "frame_stack": 1, "seed": 7}


def test_camera_keys_are_the_two_views():
    assert CAMERA_KEYS == (("pixels", "overview_rgb"), ("pixels", "wrist_rgb"))


# --------------------------------------------------------------------------------------------------
# network.in_keys expansion
# --------------------------------------------------------------------------------------------------


def test_resolve_obs_keys_expands_a_group_name(tmp_path):
    _fake_shard(tmp_path)
    assert resolve_obs_keys(tmp_path, ["proprio"]) == [("proprio", "ee_pos")]


def test_resolve_obs_keys_passes_explicit_leaves_through_unchanged(tmp_path):
    _fake_shard(tmp_path)
    keys = [["pixels", "overview_rgb"], ["pixels", "wrist_rgb"]]
    assert resolve_obs_keys(tmp_path, keys) == [("pixels", "overview_rgb"), ("pixels", "wrist_rgb")]


def test_resolve_obs_keys_mixes_leaves_and_groups_in_first_seen_order(tmp_path):
    _fake_shard(tmp_path)
    keys = [["pixels", "wrist_rgb"], "proprio", ["pixels", "overview_rgb"]]
    assert resolve_obs_keys(tmp_path, keys) == [
        ("pixels", "wrist_rgb"), ("proprio", "ee_pos"), ("pixels", "overview_rgb"),
    ]


# --------------------------------------------------------------------------------------------------
# PixelNet: image-only, vector-only and mixed forward passes
# --------------------------------------------------------------------------------------------------


def _network_cfg(**overrides):
    base = dict(cnn_channels=[4], cnn_kernels=[3], cnn_strides=[1], image_embed=8, proprio_embed=6, fusion=[16])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_pixelnet_forward_image_only():
    net = PixelNet([(8, 8, 3), (8, 8, 3)], _network_cfg(), out_dim=4, out_gain=1.0)
    out = net(torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8),
              torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8))
    assert out.shape == (2, 4)
    assert net.vec is None


def test_pixelnet_forward_vector_only():
    net = PixelNet([(5,), (3,)], _network_cfg(), out_dim=3, out_gain=1.0)
    out = net(torch.randn(2, 5), torch.randn(2, 3))
    assert out.shape == (2, 3)
    assert len(net.cnns) == 0
    assert net.vec is not None


def test_pixelnet_forward_mixed_image_and_vector():
    net = PixelNet([(8, 8, 3), (4,)], _network_cfg(), out_dim=2, out_gain=1.0)
    out = net(torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8), torch.randn(2, 4))
    assert out.shape == (2, 2)
    assert len(net.cnns) == 1 and net.vec is not None


def test_pixelnet_forward_with_extra_raw_vector_for_a_qvalue_head():
    # As make_qvalue does: obs keys (mixed) followed by the action as an unembedded extra input.
    net = PixelNet([(8, 8, 3), (4,)], _network_cfg(), out_dim=1, out_gain=1.0, extra_dim=7)
    out = net(torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8), torch.randn(2, 4), torch.randn(2, 7))
    assert out.shape == (2, 1)


# --------------------------------------------------------------------------------------------------
# Every algorithm's make_algo builds from its real config.yaml and takes gradient steps
# --------------------------------------------------------------------------------------------------

OFFLINE_DIR = Path(__file__).resolve().parents[2] / "pipeline" / "2_1_offline_rl"
ALGORITHMS = ("bc", "cql", "iql", "td3_bc")
# One camera and one vector group, both tiny: this asserts the loss wiring, not what is learned.
ALGO_SHAPES, ACTION_DIM = ((8, 8, 3), (4,)), 3
ALGO_KEYS = (("pixels", "wrist_rgb"), ("proprio",))
EXPECTED_LOSSES = {
    "bc": {"loss_bc"},
    "cql": {"loss_actor", "loss_actor_bc", "loss_qvalue", "loss_cql", "loss_alpha", "loss_alpha_prime"},
    "iql": {"loss_actor", "loss_qvalue", "loss_value"},
    "td3_bc": {"loss_qvalue", "loss_actor", "bc_loss", "lmbd"},
}


def _load_algo_module(name):
    """Load ``<algo>/utils.py`` the way train.py does: by file path, under its own module name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{name}_utils_test", OFFLINE_DIR / name / "utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _algo_cfg(name, **loss_overrides):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(OFFLINE_DIR / name / "config.yaml")
    cfg.network = OmegaConf.create(vars(_network_cfg()))
    for key, value in loss_overrides.items():
        cfg.loss[key] = value
    return cfg


def _algo_batch(batch_size=4):
    obs = {key: (torch.randint(0, 256, (batch_size, *shape), dtype=torch.uint8) if len(shape) == 3
                 else torch.randn(batch_size, *shape))
           for key, shape in zip(ALGO_KEYS, ALGO_SHAPES)}
    flag = torch.zeros(batch_size, 1, dtype=torch.bool)
    flag[0] = True  # one terminal pair, so the masked bootstrap is exercised too
    return TensorDict(
        {**obs, "action": torch.rand(batch_size, ACTION_DIM) * 1.8 - 0.9,
         "next": TensorDict({**obs, "reward": torch.randn(batch_size, 1),
                             "terminated": flag, "done": flag.clone()}, batch_size=[batch_size])},
        batch_size=[batch_size],
    )


@pytest.mark.parametrize("name", ALGORITHMS)
def test_make_algo_takes_gradient_steps_and_reports_its_losses(name):
    # CQL's actor warm-up ends after 2 steps, TD3+BC's actor updates on step 2: 3 steps cover both branches.
    overrides = {"policy_eval_start": 2} if name == "cql" else {}
    cfg = _algo_cfg(name, **overrides)
    algo = _load_algo_module(name).make_algo(cfg, ALGO_SHAPES, ALGO_KEYS, ACTION_DIM, torch.device("cpu"))
    before = [p.detach().clone() for p in algo.policy.parameters()]

    for _ in range(3):
        losses = algo.update(_algo_batch())

    assert EXPECTED_LOSSES[name] <= set(losses), (name, sorted(losses))
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values()), losses
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, algo.policy.parameters())), name
    assert "actor" in algo.state_dict()


def test_the_deterministic_actor_maps_observations_into_the_action_bounds():
    from pickplace.offline import make_deterministic_actor

    actor = make_deterministic_actor(ALGO_SHAPES, ALGO_KEYS, ACTION_DIM, _network_cfg(), torch.device("cpu"))
    out = actor(_algo_batch(batch_size=5))
    action = out.get("action")
    assert action.shape == (5, ACTION_DIM)
    assert action.abs().max() <= 1.0


def test_only_td3_bc_gets_a_deterministic_actor():
    """``TD3BCLoss`` reads ``action`` straight out of the actor and adds the exploration noise itself.

    Handing it the shared ``TanhNormal`` actor would make its policy extraction meaningless while still
    running, training and checkpointing without a single error — so the actor type is pinned here.
    """
    algos = {name: _load_algo_module(name).make_algo(_algo_cfg(name), ALGO_SHAPES, ALGO_KEYS, ACTION_DIM,
                                                     torch.device("cpu")) for name in ALGORITHMS}

    td3_actor = algos["td3_bc"].loss_module.actor_network
    assert "action" in td3_actor.out_keys
    assert not hasattr(td3_actor, "get_dist"), "TD3+BC's actor must not be a distribution"
    batch = _algo_batch(batch_size=4).select(*ALGO_KEYS)
    assert torch.equal(td3_actor(batch.clone()).get("action"), td3_actor(batch.clone()).get("action"))

    for name, actor in (("bc", algos["bc"].policy),
                        ("iql", algos["iql"].loss_module.actor_network),
                        ("cql", algos["cql"].loss_module.actor_network)):
        assert hasattr(actor, "get_dist"), f"{name} extracts its policy from a distribution"
