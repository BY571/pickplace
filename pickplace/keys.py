"""Resolve model ``in_keys`` against an environment observation spec."""

from __future__ import annotations

from collections.abc import Sequence

from torchrl.data import Composite


def expand_in_keys(spec: Composite, keys: Sequence[str | Sequence[str]]) -> list[tuple[str, ...]]:
    """Expand group names to all their leaf keys; pass nested/leaf keys through.

    Args:
        spec: observation spec (e.g. ``env.observation_spec``).
        keys: entries like ``"proprio"`` (a group -> all its leaves, sorted),
            ``["pixels", "wrist_rgb"]`` (a nested leaf) or ``"step_count"`` (a top-level leaf).

    Returns:
        De-duplicated list of tuple keys, in first-seen order.

    Raises:
        KeyError: if a key is not in the spec; the message lists available leaf keys.
    """
    out: list[tuple[str, ...]] = []
    for key in keys:
        path = (key,) if isinstance(key, str) else tuple(key)
        try:
            sub = spec[path]
        except KeyError:
            available = sorted(str(k) for k in spec.keys(True, True))
            raise KeyError(f"Unknown observation key {path!r}. Available leaf keys: {available}") from None
        if isinstance(sub, Composite):
            leaves = sorted(path + ((leaf,) if isinstance(leaf, str) else tuple(leaf)) for leaf in sub.keys(True, True))
        else:
            leaves = [path]
        for leaf in leaves:
            if leaf not in out:
                out.append(leaf)
    return out
