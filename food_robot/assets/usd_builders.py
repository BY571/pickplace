"""Procedurally generated USD assets: a concave bowl and a prismatic belt pallet.

Isaac Lab has no primitive for a concave bowl or a jointed conveyor segment, so both are
authored with pxr and cached on disk (keyed by geometry). Requires a launched Isaac Sim app.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.utils.configclass import configclass

CACHE_DIR = Path(os.environ.get("FOOD_ROBOT_CACHE", Path.home() / ".cache" / "food_robot")) / "usd"


@configclass
class BowlGeometry:
    """Bowl made of a square base plate and a ring of box walls (all convex colliders)."""

    inner_radius: float = 0.07
    wall_height: float = 0.05
    wall_thickness: float = 0.006
    base_thickness: float = 0.006
    num_segments: int = 16
    mass: float = 0.2
    color: tuple[float, float, float] = (0.92, 0.92, 0.88)


@configclass
class PalletGeometry:
    """Fixed base + plate on a prismatic joint ``slider`` along +x."""

    size: tuple[float, float, float] = (0.22, 0.22, 0.01)
    mass: float = 2.0
    travel_lower: float = -0.05
    travel_upper: float = 1.0
    color: tuple[float, float, float] = (0.25, 0.25, 0.25)


def _cache_path(prefix: str, geom) -> Path:
    digest = hashlib.sha1(json.dumps(geom.to_dict(), sort_keys=True).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{prefix}_{digest}.usda"


def _new_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _add_box(stage, path, size, translate, rotate_z_deg=0.0, color=None, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z_deg:
        cube.AddRotateZOp().Set(float(rotate_z_deg))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _build_atomically(path: Path, author) -> str:
    """Author into a temp file and rename, so concurrent processes never read a half-written asset."""
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.usda")
    stage = _new_stage(tmp)
    author(stage)
    stage.GetRootLayer().Save()
    os.replace(tmp, path)
    return str(path)


def bowl_usd_path(geom: BowlGeometry) -> str:
    def author(stage):
        root = UsdGeom.Xform.Define(stage, "/Bowl")
        stage.SetDefaultPrim(root.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(geom.mass)
        r, t, h, bt = geom.inner_radius, geom.wall_thickness, geom.wall_height, geom.base_thickness
        _add_box(stage, "/Bowl/base", (2 * (r + t), 2 * (r + t), bt), (0.0, 0.0, bt / 2), color=geom.color)
        ring = r + t / 2
        seg_len = 2 * ring * math.tan(math.pi / geom.num_segments) + t
        for i in range(geom.num_segments):
            a = 2 * math.pi * i / geom.num_segments
            _add_box(
                stage,
                f"/Bowl/wall_{i:02d}",
                (t, seg_len, h),
                (ring * math.cos(a), ring * math.sin(a), bt + h / 2),
                rotate_z_deg=math.degrees(a),
                color=geom.color,
            )

    return _build_atomically(_cache_path("bowl", geom), author)


def pallet_usd_path(geom: PalletGeometry) -> str:
    def author(stage):
        root = UsdGeom.Xform.Define(stage, "/Pallet")
        stage.SetDefaultPrim(root.GetPrim())
        # root link: carries the articulation root; Isaac Lab's fix_root_link welds it to the world
        base = UsdGeom.Xform.Define(stage, "/Pallet/base").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        base_mass = UsdPhysics.MassAPI.Apply(base)
        base_mass.CreateMassAttr(1.0)
        base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-3, 1e-3, 1e-3))
        # moving plate
        plate = UsdGeom.Xform.Define(stage, "/Pallet/plate").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(plate)
        UsdPhysics.MassAPI.Apply(plate).CreateMassAttr(geom.mass)
        _add_box(stage, "/Pallet/plate/collision", geom.size, (0.0, 0.0, geom.size[2] / 2), color=geom.color)
        # prismatic joint with a linear drive (gains come from the Isaac Lab actuator cfg)
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/Pallet/slider")
        joint.CreateBody0Rel().SetTargets(["/Pallet/base"])
        joint.CreateBody1Rel().SetTargets(["/Pallet/plate"])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(geom.travel_lower)
        joint.CreateUpperLimitAttr(geom.travel_upper)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(1e4)

    return _build_atomically(_cache_path("pallet", geom), author)
