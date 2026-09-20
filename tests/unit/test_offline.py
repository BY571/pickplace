"""The transition index is the piece most likely to be silently wrong, so it is tested exhaustively."""

import pytest
import torch
from tensordict import TensorDict

from pickplace.artifacts import write_json
from pickplace.datasets import STORAGE_DIR
from pickplace.offline import (
    CAMERA_KEYS,
    ShardTransitions,
    TransitionSampler,
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
