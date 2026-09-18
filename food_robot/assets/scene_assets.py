"""Isaac Lab asset configs for the generated pallet and bowls."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg

from food_robot.assets.usd_builders import BowlGeometry, PalletGeometry, bowl_usd_path, pallet_usd_path
from food_robot.belt import BELT_COLOR


def make_pallet_cfg(
    geom: PalletGeometry, prim_path: str, pos: tuple[float, float, float], damping: float
) -> ArticulationCfg:
    """Fixed-base pallet; joint ``slider`` tracks a velocity target through a pure damping drive."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=pallet_usd_path(geom),
            # the plate reads as part of the conveyor: the belt's own material (visual only; geometry unchanged)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BELT_COLOR),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos, joint_pos={"slider": 0.0}, joint_vel={"slider": 0.0}),
        actuators={
            "belt": ImplicitActuatorCfg(
                joint_names_expr=["slider"],
                stiffness=0.0,
                damping=damping,
                effort_limit_sim=1e4,
                velocity_limit_sim=1.0,
            )
        },
    )


def make_bowl_cfg(
    geom: BowlGeometry, prim_path: str, pos: tuple[float, float, float], kinematic: bool
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=bowl_usd_path(geom),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
