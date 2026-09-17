# food_robot/food/rigid.py
"""Rigid spherical food item (e.g. a meatball) with friction and mass randomization."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg

from food_robot.envs.mdp.observations import asset_quat_w
from food_robot.envs.mdp.events import reset_food_in_bowl
from food_robot.food.base import FoodSourceCfg


@configclass
class RigidFoodCfg(FoodSourceCfg):
    name: str = "rigid_sphere"
    item_radius: float = 0.02
    mass: float = 0.03
    static_friction_range: tuple[float, float] = (0.6, 1.2)
    dynamic_friction_range: tuple[float, float] = (0.5, 1.0)
    """Raised from (0.3, 1.0) / (0.2, 0.8) in task 13: against the fingers' own ``finger_friction`` (default
    1.2/1.0 static/dynamic, PhysX's averaging combine mode), the old range's lower end left the effective
    dynamic coefficient as low as ~0.35 -- too slippery to lift a smooth 4 cm sphere gripped by flat pads.
    See scripts/probe_grasp.py's friction sweep (task-13-report.md) for the measured slip-vs-friction curve
    that justifies this range."""
    restitution_range: tuple[float, float] = (0.0, 0.1)
    mass_scale_range: tuple[float, float] = (0.7, 1.3)
    spawn_range: float = 0.06
    """Food spawn xy randomization (+- m) around the ingredient bowl's position in this reset. Raised from
    0.02 in task 14 alongside the wider supply tray (``FoodCellEnvCfg.supply_bowl``, inner_radius 0.11 m):
    ``reset_food_in_bowl`` samples a square of +-spawn_range, so the worst-case corner is
    0.06 * sqrt(2) = 0.085 m from the tray centre, plus the 0.02 m food radius = 0.105 m, inside the 0.11 m
    inner wall. The bowl's own position is now fixed (see ``ingredient_bowl_x_range``/``_y_range``), so this
    is where the policy's xy localization challenge now lives."""

    def __post_init__(self):
        self.asset = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Food",
            spawn=sim_utils.SphereCfg(
                radius=self.item_radius,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.mass),
                collision_props=CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7),
                # Green (task 14): the scene is a white robot, off-white bowls and a grey table, so green
                # separates the food from everything else far better than the old brown, especially at
                # 84x84-128x128 px.
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.28, 0.05)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        # PhysX material/mass randomization uses CPU tensors: apply once at startup (Isaac Lab guidance)
        self.events = {
            "food_material": EventTerm(
                func=base_mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("food"),
                    "static_friction_range": self.static_friction_range,
                    "dynamic_friction_range": self.dynamic_friction_range,
                    "restitution_range": self.restitution_range,
                    "num_buckets": 64,
                    "make_consistent": True,
                },
            ),
            "food_mass": EventTerm(
                func=base_mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("food"),
                    "mass_distribution_params": self.mass_scale_range,
                    "operation": "scale",
                },
            ),
            # reset-mode: merged into the env's EventCfg after `reset_ingredient_bowl` (setattr on a configclass
            # appends new fields after existing ones). height_above_bowl is filled in by the env, which knows
            # the bowl geometry.
            "reset_food": EventTerm(
                func=reset_food_in_bowl,
                mode="reset",
                params={
                    "spawn_range": self.spawn_range,
                    "height_above_bowl": 0.0,
                    "food_cfg": SceneEntityCfg("food"),
                },
            ),
        }
        self.privileged_obs = {
            "food_quat": ObsTerm(func=asset_quat_w, params={"asset_cfg": SceneEntityCfg("food")}),
        }
