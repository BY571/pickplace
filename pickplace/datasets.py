"""Read offline-RL dataset shards written by ``pipeline/1_collect_data/collect.py``. Simulator-free.

A shard directory holds ``storage/`` (a memory-mapped TensorDict, one row per env step) and ``manifest.json``.
Rows are time-major: row ``i`` and row ``i + manifest["successor_stride"]`` (the collection's ``num_envs``) are
consecutive steps of the same sub-env, so the successor state of row ``i`` is row ``i + stride`` — valid when
``("next", "done")[i]`` is False and ``i + stride < frames``.
"""

from __future__ import annotations

from pathlib import Path

from pickplace.artifacts import read_json

STORAGE_DIR = "storage"
MANIFEST_NAME = "manifest.json"


def shard_manifest(path: str | Path) -> dict:
    """The shard's ``manifest.json`` (frames, source checkpoint, resolved env config, data statistics)."""
    return read_json(Path(path) / MANIFEST_NAME)


def load_shard(path: str | Path, batch_size: int = 256, **kwargs):
    """Open a shard as a ``TensorDictReplayBuffer`` over its memmap; nothing is copied into RAM."""
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer, TensorStorage

    storage = TensorStorage(TensorDict.load_memmap(Path(path) / STORAGE_DIR))
    return TensorDictReplayBuffer(storage=storage, batch_size=batch_size, **kwargs)
