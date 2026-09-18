"""Simulator-free geometry and bookkeeping for the continuous (carousel) demo.

The demo scene has K pallets on one belt. Each pallet's ``slider`` joint position ``q`` places it at
``x = entry_x + q``; a pallet that reaches ``recycle_q = K * spacing`` is written back to ``q = 0`` (the entry),
so consecutive bowls stay exactly ``spacing`` apart. Must not import Isaac Sim / Isaac Lab.
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
    """Pallet joint position at which a pallet is written back to the entry (``bowls * spacing``)."""
    start_q: tuple[float, ...]
    """Initial joint position of each pallet."""


def carousel_layout(
    bowls: int,
    entry_x: float,
    belt_end_x: float,
    zone_end_x: float,
    pallet_length: float,
    bowl_outer_radius: float,
    travel_upper: float,
    spacing: float | None = None,
) -> CarouselLayout:
    """Evenly spaced pallets; by default they fill the belt from the entry to its visible end.

    Raises:
        ValueError: if the pallets would touch, the recycle point lies beyond the belt end or the pallet's
            joint limit, or a bowl would be recycled before it has left the reach zone.
    """
    if bowls < 1:
        raise ValueError(f"demo bowls must be >= 1, got {bowls}.")
    loop = min(belt_end_x - entry_x, travel_upper)
    if spacing is None:
        spacing = loop / bowls
    spacing = float(spacing)
    recycle_q = bowls * spacing
    if bowls > 1 and spacing < pallet_length + PALLET_GAP - 1e-9:
        raise ValueError(
            f"{bowls} bowls {spacing:.3f} m apart: pallets ({pallet_length} m long) need at least "
            f"{pallet_length + PALLET_GAP:.3f} m spacing. Use fewer bowls or a larger spacing."
        )
    if recycle_q > loop + 1e-9:
        raise ValueError(
            f"{bowls} bowls x {spacing:.3f} m = {recycle_q:.3f} m runs past the belt end ({loop:.3f} m from the "
            f"entry). Use fewer bowls or a smaller spacing."
        )
    if entry_x + recycle_q < zone_end_x + bowl_outer_radius:
        raise ValueError(
            f"The recycle point x={entry_x + recycle_q:.3f} m lies before the end of the reach zone plus a bowl "
            f"radius ({zone_end_x + bowl_outer_radius:.3f} m): bowls would be recycled inside the zone. "
            "Use a larger spacing."
        )
    return CarouselLayout(spacing, recycle_q, tuple(i * spacing for i in range(bowls)))


def pick_target(xs: Sequence[float], open_: Sequence[bool], zone_end_x: float) -> int | None:
    """Index of the most downstream open bowl that has not left the reach zone (upstream bowls count too)."""
    candidates = [i for i, (x, ok) in enumerate(zip(xs, open_)) if ok and x <= zone_end_x]
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
