"""Simulator-free belt timing, reachability and camera geometry helpers.

Must not import Isaac Sim / Isaac Lab so it can be unit-tested without a simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BeltZone:
    """Belt-axis interval (cell frame) in which the arm must place the food."""

    start_x: float
    end_x: float

    @property
    def length(self) -> float:
        return self.end_x - self.start_x


def belt_zone(speed: float, place_window: float, zone_center_x: float) -> BeltZone:
    """Zone that a bowl moving at ``speed`` traverses in ``place_window`` seconds."""
    if speed <= 0.0:
        raise ValueError(f"Belt speed must be > 0 m/s, got {speed}.")
    if place_window <= 0.0:
        raise ValueError(f"Place window must be > 0 s, got {place_window}.")
    half = 0.5 * speed * place_window
    return BeltZone(zone_center_x - half, zone_center_x + half)


def validate_zone_reachable(
    zone: BeltZone, belt_y: float, base_xy: tuple[float, float], reach_radius: float
) -> None:
    """Raise if either zone endpoint on the belt centerline lies beyond the arm's reach."""
    for x in (zone.start_x, zone.end_x):
        distance = math.hypot(x - base_xy[0], belt_y - base_xy[1])
        if distance > reach_radius:
            raise ValueError(
                f"Belt zone endpoint (x={x:.3f}, y={belt_y:.3f}) is {distance:.3f} m from the arm base, "
                f"beyond reach_radius={reach_radius:.3f} m. Reduce speed * place_window or move zone_center_x."
            )


def validate_observation_flags(cameras: bool, privileged_information: bool) -> None:
    """Raise if no observation group would carry information about the food."""
    if not cameras and not privileged_information:
        raise ValueError(
            "cameras=False and privileged_information=False leaves no observation that carries food "
            "information; enable at least one of them."
        )


def look_at_quat_xyzw(eye: Sequence[float], target: Sequence[float]) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) whose +X axis points from ``eye`` to ``target`` with no roll (+Z up)."""
    d = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(c * c for c in d))
    if n == 0.0:
        raise ValueError("eye and target must differ.")
    fx, fy, fz = (c / n for c in d)
    yaw = math.atan2(fy, fx)
    pitch = -math.asin(max(-1.0, min(1.0, fz)))
    sz, cz = math.sin(yaw / 2), math.cos(yaw / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    # q = q_z(yaw) * q_y(pitch)
    return (-sz * sp, cz * sp, sz * cp, cz * cp)


def quat_rotate_xyzw(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate vector ``v`` by unit quaternion ``q`` given as (x, y, z, w)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _check_ordered(name: str, value_range: tuple[float, float]) -> None:
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} must be (min, max), got {value_range}.")


def validate_supply_bowl_range(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bowl_outer_radius: float,
    reach_radius: float,
    belt_y: float,
    belt_half_width: float,
    base_xy: tuple[float, float] = (0.0, 0.0),
    belt_clearance: float = 0.02,
) -> None:
    """Raise if any supply-bowl placement in the range is out of reach or too close to the belt strip."""
    _check_ordered("ingredient_bowl_x_range", x_range)
    _check_ordered("ingredient_bowl_y_range", y_range)
    for x in x_range:
        for y in y_range:
            far_edge = math.hypot(x - base_xy[0], y - base_xy[1]) + bowl_outer_radius
            if far_edge > reach_radius:
                raise ValueError(
                    f"Supply bowl at ({x:.3f}, {y:.3f}) reaches {far_edge:.3f} m from the arm base, beyond "
                    f"reach_radius={reach_radius:.3f} m. Shrink ingredient_bowl_x_range/y_range."
                )
    belt_edge = belt_y - belt_half_width - belt_clearance
    if y_range[1] + bowl_outer_radius > belt_edge:
        raise ValueError(
            f"Supply bowl rim at y={y_range[1] + bowl_outer_radius:.3f} m comes within {belt_clearance} m of the "
            f"belt strip (edge at y={belt_y - belt_half_width:.3f} m). Lower ingredient_bowl_y_range max."
        )


def validate_bowl_on_pallet(
    pallet_size_xy: tuple[float, float],
    bowl_outer_radius: float,
    offset_x: tuple[float, float],
    offset_y: tuple[float, float],
    belt_half_width: float,
) -> None:
    """Raise if the receiving bowl can hang over the pallet edge, or the pallet is wider than the belt."""
    _check_ordered("bowl_offset_x", offset_x)
    _check_ordered("bowl_offset_y", offset_y)
    if bowl_outer_radius + max(abs(v) for v in offset_x) > pallet_size_xy[0] / 2:
        raise ValueError(
            f"Bowl (outer radius {bowl_outer_radius} m) with offset {offset_x} along the belt overhangs the pallet "
            f"(length {pallet_size_xy[0]} m). Reduce bowl_offset_x or lengthen the pallet."
        )
    if bowl_outer_radius + max(abs(v) for v in offset_y) > pallet_size_xy[1] / 2:
        raise ValueError(
            f"Bowl (outer radius {bowl_outer_radius} m) with offset {offset_y} across the belt overhangs the pallet "
            f"(width {pallet_size_xy[1]} m). Reduce bowl_offset_y or widen the pallet."
        )
    if pallet_size_xy[1] / 2 > belt_half_width:
        raise ValueError(
            f"Pallet width {pallet_size_xy[1]} m is wider than the belt ({2 * belt_half_width} m)."
        )


def validate_pallet_start(
    start_range: tuple[float, float],
    travel_lower: float,
    travel_upper: float,
    zone_length: float,
    entry_margin: float,
) -> None:
    """Raise if the pallet's reset position range does not fit its joint travel limits."""
    _check_ordered("pallet_start_range", start_range)
    if start_range[0] < travel_lower:
        raise ValueError(
            f"pallet_start_range min {start_range[0]} is below the pallet joint's travel_lower {travel_lower}."
        )
    needed = start_range[1] + entry_margin + zone_length + 0.1
    if travel_upper < needed:
        raise ValueError(
            f"Pallet travel_upper {travel_upper} m is too short: the latest start plus the zone needs {needed:.3f} m."
        )


def zone_exit_times(
    entry_x: float,
    zone_end_x: float,
    speed: float,
    start_range: tuple[float, float],
    offset_x: tuple[float, float],
) -> tuple[float, float, float]:
    """Seconds until the bowl center passes the zone end: (earliest start, nominal start, latest start)."""
    earliest = (zone_end_x - (entry_x + start_range[0] + offset_x[0])) / speed
    nominal = (zone_end_x - entry_x) / speed
    latest = (zone_end_x - (entry_x + start_range[1] + offset_x[1])) / speed
    return earliest, nominal, latest
