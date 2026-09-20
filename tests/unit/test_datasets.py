import torch
from tensordict import TensorDict

from food_robot.artifacts import write_json
from food_robot.datasets import STORAGE_DIR, load_shard, shard_manifest

FRAMES, STRIDE = 8, 2


def _fake_shard(path):
    """A shard directory with the layout collect.py writes: a memmapped TensorDict plus a manifest."""
    td = TensorDict(
        {
            "pixels": TensorDict(
                {name: torch.arange(FRAMES * 12, dtype=torch.uint8).reshape(FRAMES, 2, 2, 3) for name in ("overview_rgb", "wrist_rgb")},
                batch_size=[FRAMES],
            ),
            "proprio": TensorDict({"ee_pos": torch.randn(FRAMES, 3)}, batch_size=[FRAMES]),
            "action": torch.randn(FRAMES, 7),
            "loc": torch.randn(FRAMES, 7),
            "next": TensorDict(
                {
                    "reward": torch.randn(FRAMES, 1),
                    "reward_terms": torch.randn(FRAMES, 13),
                    "done": torch.zeros(FRAMES, 1, dtype=torch.bool),
                },
                batch_size=[FRAMES],
            ),
        },
        batch_size=[FRAMES],
    )
    td.memmap_(str(path / STORAGE_DIR))
    write_json(path / "manifest.json", {"frames": FRAMES, "successor_stride": STRIDE, "stats": {"success_rate": 0.5}})
    return td


def test_load_shard_samples_the_stored_keys_and_dtypes(tmp_path):
    _fake_shard(tmp_path)
    buffer = load_shard(tmp_path, batch_size=4)

    assert len(buffer) == FRAMES
    batch = buffer.sample()
    assert batch.shape == (4,)
    assert batch["pixels", "overview_rgb"].dtype == torch.uint8
    assert batch["pixels", "wrist_rgb"].shape == (4, 2, 2, 3)
    assert batch["action"].shape == (4, 7) and batch["action"].dtype == torch.float32
    assert batch["next", "reward_terms"].shape == (4, 13)
    assert batch["next", "done"].dtype == torch.bool


def test_load_shard_returns_the_stored_rows_in_order(tmp_path):
    td = _fake_shard(tmp_path)
    stored = load_shard(tmp_path)[:]
    assert torch.equal(stored["action"], td["action"])
    assert torch.equal(stored["pixels", "wrist_rgb"], td["pixels", "wrist_rgb"])


def test_shard_manifest(tmp_path):
    _fake_shard(tmp_path)
    manifest = shard_manifest(tmp_path)
    assert manifest["frames"] == FRAMES and manifest["successor_stride"] == STRIDE
    assert manifest["stats"]["success_rate"] == 0.5
