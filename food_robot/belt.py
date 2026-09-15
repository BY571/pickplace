"""Conveyor belt plug-in: fixed line speed with small noise and an explicit placement window."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from food_robot.assets.usd_builders import BowlGeometry, PalletGeometry
from food_robot.timing import BeltZone, belt_zone


@configclass
class BeltCfg:
    speed: float = 0.08
    """Nominal line speed [m/s]."""
    speed_noise: float = 0.02
    """Per-episode relative speed noise: speed * (1 + U(-noise, noise))."""
    place_window: float = 5.0
    """Seconds the bowl spends inside the reach zone at nominal speed. Zone length = speed * place_window."""
    zone_center_x: float = 0.45
    belt_y: float = 0.30
    belt_half_width: float = 0.15
    entry_margin: float = 0.05
    """The bowl spawns this far before the zone entry."""
    bowl_offset_x: tuple[float, float] = (-0.03, 0.03)
    """Reset randomization of the bowl position on the pallet along the belt (shifts the timing)."""
    bowl_offset_y: tuple[float, float] = (-0.02, 0.02)
    """Reset randomization across the belt."""
    plate_top_z: float = 0.03
    """Height of the pallet top surface above the table."""
    pallet: PalletGeometry = PalletGeometry()
    bowl: BowlGeometry = BowlGeometry()
    pallet_damping: float = 1e4

    def zone(self) -> BeltZone:
        return belt_zone(self.speed, self.place_window, self.zone_center_x)

    def entry_x(self) -> float:
        return self.zone().start_x - self.entry_margin
