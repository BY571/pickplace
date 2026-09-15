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
