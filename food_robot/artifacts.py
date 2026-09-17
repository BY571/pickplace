"""Artifact locations and JSON manifests for pipeline runs. Simulator-free."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ARTIFACTS_ENV = "FOOD_ROBOT_ARTIFACTS"
GIT_COMMIT_ENV = "FOOD_ROBOT_GIT_COMMIT"


def artifacts_root() -> Path:
    """Where pipeline artifacts live — outside the repo, because scripts/spark.sh rsyncs it with --delete."""
    return Path(os.environ.get(ARTIFACTS_ENV, "~/food-robot-artifacts")).expanduser()


def git_commit() -> str:
    """The code version: $FOOD_ROBOT_GIT_COMMIT (set by scripts/spark.sh; the container has no .git), else git."""
    if os.environ.get(GIT_COMMIT_ENV):
        return os.environ[GIT_COMMIT_ENV]
    try:
        repo = Path(__file__).resolve().parents[1]
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, data: dict) -> None:
    """Atomic write (tmp file + rename), so readers never see a half-written manifest."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def update_json(path: str | Path, **fields) -> dict:
    data = read_json(path) if Path(path).exists() else {}
    data.update(fields)
    write_json(path, data)
    return data


def new_run_dir(kind: str, name: str | None = None) -> Path:
    """``<root>/<kind>/<name>`` (default name ``<kind>_<UTC timestamp>``), created."""
    name = name or f"{kind}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    run_dir = artifacts_root() / kind / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
