"""Simulator-free geometry and bookkeeping for the continuous (carousel) demo.

The demo scene has K pallets on one belt. Each pallet's ``slider`` joint position ``q`` places it at
``x = entry_x + q`` (``entry_x`` of the training belt); a pallet that reaches the belt end ``recycle_q`` is written
back to ``entry_q = recycle_q - K * spacing`` (upstream, on an extended belt when needed), so consecutive bowls stay
exactly ``spacing`` apart. Must not import Isaac Sim / Isaac Lab.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PALLET_GAP = 0.02
"""Minimum free space between two consecutive pallets [m]."""


@dataclass(frozen=True)
class CarouselLayout:
    spacing: float
    """Distance between consecutive pallets along the belt [m]."""
    recycle_q: float
    """Pallet joint position of the belt end, where a pallet is written back to ``entry_q``."""
    entry_q: float
    """Pallet joint position where recycled pallets re-enter (``recycle_q - bowls * spacing``); negative values lie
    upstream of the training entry, on the demo's extended belt."""
    start_q: tuple[float, ...]
    """Initial joint position of each pallet (``entry_q + i * spacing``)."""


def carousel_layout(
    bowls: int,
    entry_x: float,
    belt_end_x: float,
    zone_length: float,
    zone_end_x: float,
    earliest_start_q: float,
    pallet_length: float,
    bowl_outer_radius: float,
    spacing: float | None = None,
) -> CarouselLayout:
    """Evenly spaced pallets circulating between ``entry_q`` and the belt end.

    The loop (``bowls * spacing``) must reach back at least to ``earliest_start_q``, the earliest pallet start of a
    training episode, so every bowl approaches the zone from where training starts it; with more bowls the belt is
    extended upstream. Default spacing: one reach-zone length (one place window of belt travel) per bowl, or more
    when needed for that.

    Raises:
        ValueError: if the pallets would touch, bowls would enter downstream of the training start, or the belt
            ends before a bowl has left the reach zone.
    """
    if bowls < 1:
        raise ValueError(f"demo bowls must be >= 1, got {bowls}.")
    recycle_q = belt_end_x - entry_x
    if belt_end_x < zone_end_x + bowl_outer_radius:
        raise ValueError(
            f"The belt ends at x={belt_end_x:.3f} m, before the end of the reach zone plus a bowl radius "
            f"({zone_end_x + bowl_outer_radius:.3f} m): bowls would be recycled inside the zone."
        )
    pitch = pallet_length + PALLET_GAP
    path = recycle_q - earliest_start_q
    if spacing is None:
        spacing = max(zone_length, pitch, path / bowls)
    spacing = float(spacing)
    if bowls > 1 and spacing < pitch - 1e-9:
        raise ValueError(
            f"{bowls} bowls {spacing:.3f} m apart: pallets ({pallet_length} m long) need at least {pitch:.3f} m "
            "spacing."
        )
    if bowls * spacing < path - 1e-9:
        raise ValueError(
            f"{bowls} bowls x {spacing:.3f} m = {bowls * spacing:.3f} m is shorter than the {path:.3f} m from the "
            "earliest training start to the belt end: bowls must enter at or upstream of where a training episode "
            "starts them. Use more bowls or a larger spacing."
        )
    entry_q = recycle_q - bowls * spacing
    return CarouselLayout(spacing, recycle_q, entry_q, tuple(entry_q + i * spacing for i in range(bowls)))


def pick_target(xs: Sequence[float], open_: Sequence[bool], min_x: float, zone_end_x: float) -> int | None:
    """Index of the most downstream open bowl between ``min_x`` (where a training episode can start a bowl) and
    the end of the reach zone; None while every open bowl is still queued upstream."""
    candidates = [i for i, (x, ok) in enumerate(zip(xs, open_)) if ok and min_x <= x <= zone_end_x]
    return max(candidates, key=lambda i: xs[i]) if candidates else None


def park_position(j: int, ground_z: float, item_radius: float) -> tuple[float, float, float]:
    """Resting spot of spare food item ``j`` on the floor under the table (out of view, out of reach)."""
    row, col = divmod(j, 16)
    return (-0.15 + 0.08 * col, -0.40 + 0.08 * row, ground_z + item_radius + 0.005)


@dataclass
class DemoTally:
    """Counters of a continuous demo run."""

    placed: int = 0
    """Food settled in an open bowl (same condition as the ``success`` termination)."""
    missed: int = 0
    """Bowls that left the reach zone without food."""
    dropped: int = 0
    """Active food items that fell off the table (or rode off the belt end outside a bowl)."""
    misplaced: int = 0
    """Food settled in a bowl that was already filled or already missed."""
    bowls_seen: int = 0

    def bowl_entered(self) -> None:
        self.bowls_seen += 1

    def summary(self, seconds: float) -> dict:
        rate = self.placed / (seconds / 60.0) if seconds > 0 else 0.0
        return {
            "placed": self.placed,
            "missed": self.missed,
            "dropped": self.dropped,
            "misplaced": self.misplaced,
            "bowls_seen": self.bowls_seen,
            "pending": self.bowls_seen - self.placed - self.missed,
            "seconds": round(seconds, 3),
            "placements_per_min": round(rate, 3),
        }
