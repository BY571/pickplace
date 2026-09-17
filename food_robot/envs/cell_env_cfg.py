"""FoodCellEnvCfg: one arm, one conveyor pallet with a free bowl, one ingredient bowl, one food source."""

from __future__ import annotations

import copy
import math
from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import CollisionPropertiesCfg

from food_robot.arms import FRANKA_CFG, ArmCfg
from food_robot.assets.scene_assets import make_bowl_cfg, make_pallet_cfg
from food_robot.assets.usd_builders import BowlGeometry
from food_robot.belt import BeltCfg
from food_robot.envs import mdp
from food_robot.food import FoodSourceCfg, RigidFoodCfg
from food_robot.timing import (
    look_at_quat_xyzw,
    validate_bowl_on_pallet,
    validate_observation_flags,
    validate_pallet_start,
    validate_supply_bowl_range,
    validate_zone_reachable,
)

ActionMode = Literal["ee_delta_pose", "joint_pos"]


@configclass
class FoodCellSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    pallet: ArticulationCfg = MISSING
    bowl: RigidObjectCfg = MISSING
    ingredient_bowl: RigidObjectCfg = MISSING
    food: RigidObjectCfg | DeformableObjectCfg = MISSING
    belt_visual: AssetBaseCfg | None = None
    wrist_cam: CameraCfg | None = None
    overview_cam: CameraCfg | None = None
    render_cam: CameraCfg | None = None
    """Wide third-person camera for videos; never part of the observations (see ``render_camera``)."""

    # Plain box table, top surface at z = 0 in the cell frame. It spans x in [-0.30, 1.20] and y in [-0.55, 0.55],
    # covering the robot base at the origin, the ingredient bowl and the whole belt strip (y = 0.30 +- 0.15).
    # Replaces Isaac Lab's SeattleLabTable asset, which ships with a mounting rail that cluttered the scene.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.40)),
        spawn=sim_utils.CuboidCfg(
            size=(1.50, 1.10, 0.80),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.22, 0.24), roughness=0.8),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: object = MISSING
    gripper_action: base_mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=base_mdp.joint_vel_rel)
        gripper_pos = ObsTerm(func=mdp.gripper_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pos = ObsTerm(func=mdp.ee_pos_cell)
        ee_quat = ObsTerm(func=mdp.ee_quat_w)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class BeltObsCfg(ObsGroup):
        bowl_pos = ObsTerm(func=mdp.asset_pos_cell, params={"asset_cfg": SceneEntityCfg("bowl")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PixelsCfg(ObsGroup):
        wrist_rgb = ObsTerm(
            func=mdp.image_float,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb"},
        )
        overview_rgb = ObsTerm(
            func=mdp.image_float,
            params={"sensor_cfg": SceneEntityCfg("overview_cam"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        food_pos = ObsTerm(func=mdp.asset_pos_cell, params={"asset_cfg": SceneEntityCfg("food")})
        # food_quat is contributed by the food plug-in (see FoodSourceCfg.privileged_obs / RigidFoodCfg)
        is_grasped = ObsTerm(func=mdp.is_grasped, params={"robot_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    proprio: ProprioCfg = ProprioCfg()
    belt: BeltObsCfg = BeltObsCfg()
    pixels: PixelsCfg | None = PixelsCfg()
    privileged: PrivilegedCfg | None = PrivilegedCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    reset_belt = EventTerm(func=mdp.reset_belt, mode="reset", params={})  # params set by _build_belt_terms
    reset_ingredient_bowl = EventTerm(func=mdp.reset_ingredient_bowl, mode="reset", params={})  # params set by _build_food_terms
    gripper_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup", params={}
    )  # params (asset_cfg, friction ranges) set by _build_events from arm.gripper_body_names/finger_friction
    # reset_food is contributed by the food plug-in (see FoodSourceCfg.events / RigidFoodCfg) and merged
    # in via _build_events(); it is appended after the fields above, so it always runs after reset_all and
    # reset_ingredient_bowl.


@configclass
class RewardsCfg:
    reach_food = RewTerm(func=mdp.reach_food, weight=1.0, params={"std": 0.3})
    grasp = RewTerm(func=mdp.grasped, weight=2.0, params={})
    grasp_lift = RewTerm(func=mdp.grasp_lift, weight=5.0, params={"lift_height": 0.10})
    transport = RewTerm(func=mdp.transport_to_bowl, weight=10.0, params={"std": 0.3})
    transport_fine = RewTerm(func=mdp.transport_to_bowl, weight=5.0, params={"std": 0.05})
    place_success = RewTerm(func=mdp.termination_indicator, weight=1.0, params={"term_names": ["success"]})
    bowl_failure = RewTerm(
        func=mdp.termination_indicator, weight=-1.0, params={"term_names": ["bowl_off_belt", "bowl_tipped"]}
    )
    food_dropped = RewTerm(func=mdp.termination_indicator, weight=-1.0, params={"term_names": ["food_off_table"]})
    bowl_disturbance = RewTerm(func=mdp.bowl_disturbance, weight=-1.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bowl_exited_zone = DoneTerm(func=mdp.bowl_exited_zone, params={})
    bowl_off_belt = DoneTerm(func=mdp.bowl_off_belt, params={})
    bowl_tipped = DoneTerm(func=mdp.bowl_tipped, params={"max_tilt_rad": math.radians(45.0)})
    success = DoneTerm(func=mdp.food_in_bowl, params={})
    food_off_table = DoneTerm(func=mdp.food_off_table, params={"minimum_height": -0.05})


@configclass
class FoodCellEnvCfg(ManagerBasedRLEnvCfg):
    # --- plug-ins and flags (set through constructor kwargs) ---
    arm: ArmCfg = FRANKA_CFG
    food: FoodSourceCfg = RigidFoodCfg()
    belt: BeltCfg = BeltCfg()
    action_mode: ActionMode = "ee_delta_pose"
    cameras: bool = True
    image_size: tuple[int, int] = (128, 128)
    frame_stack: int = 1
    """Number of most recent camera frames stacked along the channel axis of each pixel observation.
    1 = single frame (H x W x 3); 3 = H x W x 9, oldest frame first. History resets per env."""
    privileged_information: bool = False
    ingredient_bowl_pos: tuple[float, float, float] = (0.45, -0.10, 0.0)
    ingredient_bowl_x_range: tuple[float, float] = (0.45, 0.45)
    """Reset randomization of the ingredient bowl's x position (cell frame). Degenerate (fixed) by default as
    of task 14: the food is randomized inside the tray instead (see ``RigidFoodCfg.spawn_range``), which
    covers a comparable xy spread while the container itself stays put. The field is kept, range and all, so
    bowl position DR can be switched back on."""
    ingredient_bowl_y_range: tuple[float, float] = (-0.10, -0.10)
    """Reset randomization of the ingredient bowl's y position (cell frame). See ``ingredient_bowl_x_range``."""
    supply_bowl: BowlGeometry = BowlGeometry(inner_radius=0.11, wall_height=0.015)
    """Ingredient container: wide and shallow (task 14) so a top-down gripper can straddle the food without
    bottoming its hand on the rim -- see ``task-13-debug-report.md``'s H2 (a 5 cm rim sat 1 cm above the top
    of the food; the hand fouled the rim, not the fingers fouling the wall). The destination bowl on the belt
    keeps its own deeper geometry (``belt.bowl``) -- the two are deliberately separate configs; widening this
    one does not change the pallet/success-termination geometry at all."""
    overview_cam_eye: tuple[float, float, float] = (1.5, 0.1, 1.0)
    overview_cam_target: tuple[float, float, float] = (0.3, 0.1, 0.3)
    render_camera: bool = False
    """Spawn ``scene.render_cam``, a wide view of the whole cell for videos. It is not an observation, so it
    does not change the TorchRL specs; the app must be launched with cameras enabled."""
    render_cam_eye: tuple[float, float, float] = (2.3, -1.9, 1.9)
    render_cam_target: tuple[float, float, float] = (0.3, 0.0, 0.35)
    render_image_size: tuple[int, int] = (720, 1280)
    # Anti-exploit constraint (spec §5.5): each one-shot term below must exceed whatever dense shaping
    # a policy could still earn by deliberately ending the episode early instead of trying, so failing
    # early is never more profitable than a real attempt. Dense shaping (reach_food, grasp, grasp_lift,
    # transport, transport_fine) is non-negative by construction, so simply ending the episode sooner
    # never *gains* reward on its own -- the risk is a policy that pays the small per-step
    # regularization cost (action_rate, joint_vel, bowl_disturbance, ~1e-4-1/s) in exchange for a
    # one-shot bonus/penalty. Over a normal zone traversal -- (entry_margin 0.05 + earliest pallet start
    # 0.08 + max bowl offset 0.02 + zone_length 0.40) / belt speed 0.08 ~= 6.9 s -- the maximum dense
    # shaping obtainable is the sum of the non-one-shot weights, now ~23/s with `grasp`'s weight 2 added
    # (was ~21/s) * 6.9 s ~= 159 -- slightly *above* the 150 bonus/penalty, so this bound is no longer
    # comfortable on its own. The invariant it guards still holds, though: dense shaping is non-negative
    # while the one-shot penalty is strictly -150, so any real attempt (0 <= reward <= ~159) always beats
    # deliberately failing (-150) regardless of how tight the margin is. An arm that instead holds the
    # bowl in place (never triggering bowl_exited_zone) until time_out can extend that window to
    # episode_length_s ~= 10.1 s, collecting up to ~23/s * 10.1 s ~= 232 -- more than a single one-shot
    # term, but that policy still forgoes the 150 success_bonus it could have earned by actually placing
    # the food, so it is not the optimum. `grasp`'s own weight (2) is deliberately kept below `grasp_lift`
    # (5) and `transport` (10) so holding without lifting is always worth less than lifting; taken alone
    # (ignoring the other terms it can coincide with), over the longest episode (~10 s) it can add at
    # most ~20, far below the 150 success bonus.
    # Re-check this arithmetic whenever belt.speed, belt.speed_noise, belt.place_window or the dense
    # reward weights change.
    success_bonus: float = 150.0
    """Return added once when the food settles in the bowl."""
    bowl_failure_penalty: float = 150.0
    """Return subtracted once when the bowl falls off the belt or tips over."""
    food_drop_penalty: float = 150.0
    """Return subtracted once when the food falls off the table."""
    success_settle_steps: int = 5

    # --- managers ---
    scene: FoodCellSceneCfg = FoodCellSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            # 16*1024 overflowed at num_envs=4096 (measured on the Spark 2026-09-15: PhysX asked for up
            # to ~16,883 vs. a 16,384 capacity, ~118 "PxgAABBManager.cpp" errors over a 5-iteration PPO
            # run). Raised to 32*1024 for headroom; see sota-implementations/ppo/README.md.
            gpu_total_aggregate_pairs_capacity=32 * 1024,
            friction_correlation_distance=0.00625,
        )
        validate_observation_flags(self.cameras, self.privileged_information)
        if not isinstance(self.frame_stack, int) or self.frame_stack < 1:
            raise ValueError(f"frame_stack must be an int >= 1, got {self.frame_stack!r}.")
        if self.action_mode not in ("ee_delta_pose", "joint_pos"):
            raise ValueError(f"Unknown action_mode {self.action_mode!r}; use 'ee_delta_pose' or 'joint_pos'.")
        if self.food.num_items != 1:
            raise NotImplementedError("Only num_items=1 is implemented in sub-project 1.")
        zone = self.belt.zone()
        validate_zone_reachable(zone, self.belt.belt_y, (0.0, 0.0), self.arm.reach_radius)
        belt, bowl_outer_radius = self.belt, self.belt.bowl.inner_radius + self.belt.bowl.wall_thickness
        validate_pallet_start(
            belt.pallet_start_range, belt.pallet.travel_lower, belt.pallet.travel_upper, zone.length, belt.entry_margin
        )
        validate_bowl_on_pallet(
            belt.pallet.size[:2], bowl_outer_radius, belt.bowl_offset_x, belt.bowl_offset_y, belt.belt_half_width
        )
        supply_bowl_outer_radius = self.supply_bowl.inner_radius + self.supply_bowl.wall_thickness
        validate_supply_bowl_range(
            self.ingredient_bowl_x_range,
            self.ingredient_bowl_y_range,
            supply_bowl_outer_radius,
            self.arm.reach_radius,
            belt.belt_y,
            belt.belt_half_width,
        )
        slowest = belt.speed * (1.0 - belt.speed_noise)
        # safety cap only: from the earliest possible start the bowl leaves the zone (a terminated failure) before this
        longest_path = belt.entry_margin - belt.pallet_start_range[0] - belt.bowl_offset_x[0] + zone.length + 0.1
        self.episode_length_s = longest_path / slowest + 2.0

        self._build_scene(zone)
        self._build_actions()
        self._build_observations()
        self._build_events()
        self._build_belt_terms(zone)
        self._build_food_terms()

    # ------------------------------------------------------------------
    def _build_scene(self, zone) -> None:
        arm, belt, s = self.arm, self.belt, self.scene
        robot = arm.ik_robot if self.action_mode == "ee_delta_pose" else arm.robot
        s.robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        s.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{arm.base_link_name}",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{arm.ee_body_name}",
                    name="end_effector",
                    offset=OffsetCfg(pos=arm.tcp_offset),
                )
            ],
        )
        base_z = belt.plate_top_z - belt.pallet.size[2]
        s.pallet = make_pallet_cfg(belt.pallet, "{ENV_REGEX_NS}/Pallet", (belt.entry_x(), belt.belt_y, base_z), belt.pallet_damping)
        s.bowl = make_bowl_cfg(belt.bowl, "{ENV_REGEX_NS}/Bowl", (belt.entry_x(), belt.belt_y, belt.plate_top_z + 0.002), kinematic=False)
        s.ingredient_bowl = make_bowl_cfg(self.supply_bowl, "{ENV_REGEX_NS}/IngredientBowl", self.ingredient_bowl_pos, kinematic=True)
        food_z = self.ingredient_bowl_pos[2] + self.supply_bowl.base_thickness + self.food.item_radius + 0.005
        s.food = self.food.asset.replace(
            prim_path="{ENV_REGEX_NS}/Food",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.ingredient_bowl_pos[0], self.ingredient_bowl_pos[1], food_z)),
        )
        s.belt_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/BeltVisual",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(belt.zone_center_x, belt.belt_y, base_z - 0.004)),
            spawn=sim_utils.CuboidCfg(
                size=(zone.length + 2 * belt.entry_margin + 0.6, 2 * belt.belt_half_width, 0.004),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
            ),
        )
        if self.cameras:
            h, w = self.image_size
            pinhole = sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 3.0)
            )
            s.wrist_cam = CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{arm.ee_body_name}/wrist_cam",
                update_period=0.0, height=h, width=w, data_types=["rgb"], spawn=pinhole, offset=arm.wrist_cam_offset,
            )
            s.overview_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/overview_cam",
                update_period=0.0, height=h, width=w, data_types=["rgb"], spawn=pinhole,
                offset=CameraCfg.OffsetCfg(
                    pos=self.overview_cam_eye,
                    rot=look_at_quat_xyzw(self.overview_cam_eye, self.overview_cam_target),
                    convention="world",
                ),
            )
        else:
            s.wrist_cam = None
            s.overview_cam = None
        if self.render_camera:
            rh, rw = self.render_image_size
            s.render_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/render_cam",
                update_period=0.0, height=rh, width=rw, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=self.render_cam_eye,
                    rot=look_at_quat_xyzw(self.render_cam_eye, self.render_cam_target),
                    convention="world",
                ),
            )
        else:
            s.render_cam = None

    def _build_actions(self) -> None:
        arm = self.arm
        if self.action_mode == "ee_delta_pose":
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=arm.arm_joint_names,
                body_name=arm.ee_body_name,
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
                scale=arm.ik_action_scale,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=arm.tcp_offset),
            )
        else:
            self.actions.arm_action = base_mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=arm.arm_joint_names, scale=arm.joint_action_scale, use_default_offset=True
            )
        self.actions.gripper_action = base_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=arm.gripper_joint_names,
            open_command_expr={name: arm.gripper_open for name in arm.gripper_joint_names},
            close_command_expr={name: arm.gripper_closed for name in arm.gripper_joint_names},
        )

    def _build_observations(self) -> None:
        arm, obs = self.arm, self.observations
        # NOTE: each SceneEntityCfg must be a distinct instance. SceneEntityCfg.resolve() mutates
        # joint_ids in place; sharing one instance across two term params makes the second
        # resolution compare the regex in joint_names against the now-concrete joint_ids and raise
        # "Both 'joint_names' and 'joint_ids' are specified, and are not consistent." (verified at
        # runtime on the Spark; see task-4-report.md).
        obs.proprio.gripper_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=arm.gripper_joint_names)
        if not self.cameras:
            obs.pixels = None
        elif self.frame_stack > 1:
            for name, sensor in (("wrist_rgb", "wrist_cam"), ("overview_rgb", "overview_cam")):
                setattr(
                    obs.pixels,
                    name,
                    ObsTerm(
                        func=mdp.stacked_image_float,
                        params={"sensor_cfg": SceneEntityCfg(sensor), "data_type": "rgb", "frame_stack": self.frame_stack},
                    ),
                )
        if self.privileged_information:
            obs.privileged.is_grasped.params.update(
                robot_cfg=SceneEntityCfg("robot", joint_names=arm.gripper_joint_names),
                open_pos=arm.gripper_open,
                closed_pos=arm.gripper_closed,
            )
            # deepcopy: a single food cfg instance (and its term objects) may build multiple envs.
            for name, term in self.food.privileged_obs.items():
                setattr(obs.privileged, name, copy.deepcopy(term))
        else:
            obs.privileged = None

    def _build_events(self) -> None:
        arm = self.arm
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=arm.arm_joint_names)
        static_friction, dynamic_friction = arm.finger_friction
        self.events.gripper_material.params = {
            # fresh SceneEntityCfg (never shared, see the note in _build_food_terms): a body-level resolution
            # must not collide with the joint-level SceneEntityCfg instances built for the gripper action/obs.
            "asset_cfg": SceneEntityCfg("robot", body_names=arm.gripper_body_names),
            "static_friction_range": (static_friction, static_friction),
            "dynamic_friction_range": (dynamic_friction, dynamic_friction),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        }
        # deepcopy: a single food cfg instance (and its term objects) may build multiple envs.
        for name, term in self.food.events.items():
            setattr(self.events, name, copy.deepcopy(term))

    def _build_belt_terms(self, zone) -> None:
        belt = self.belt
        self.events.reset_belt.params = {
            "speed": belt.speed,
            "speed_noise": belt.speed_noise,
            "pallet_start_range": belt.pallet_start_range,
            "bowl_offset_x": belt.bowl_offset_x,
            "bowl_offset_y": belt.bowl_offset_y,
            "entry_x": belt.entry_x(),
            "belt_y": belt.belt_y,
            "plate_top_z": belt.plate_top_z,
            "pallet_cfg": SceneEntityCfg("pallet"),
            "bowl_cfg": SceneEntityCfg("bowl"),
        }
        self.terminations.bowl_exited_zone.params = {"zone_end_x": zone.end_x}
        self.terminations.bowl_off_belt.params = {
            "belt_y": belt.belt_y,
            "belt_half_width": belt.belt_half_width,
            "surface_z": belt.plate_top_z,
        }

    def _build_food_terms(self) -> None:
        arm, bowl = self.arm, self.belt.bowl
        step_dt = self.sim.dt * self.decimation
        self.events.reset_ingredient_bowl.params = {
            "x_range": self.ingredient_bowl_x_range,
            "y_range": self.ingredient_bowl_y_range,
            "z": self.ingredient_bowl_pos[2],
            "bowl_cfg": SceneEntityCfg("ingredient_bowl"),
        }
        if hasattr(self.events, "reset_food"):
            # the food rests in the *supply* bowl, not the destination bowl `bowl` aliases below (belt.bowl,
            # used only for the success termination's geometry) -- task 14 split the two containers.
            self.events.reset_food.params["height_above_bowl"] = (
                self.supply_bowl.base_thickness + self.food.item_radius + 0.005
            )

        # NOTE: a fresh SceneEntityCfg is built for every term's params (never shared) because
        # Isaac Lab resolves joint_ids in place; sharing one instance across manager term params
        # causes "Both 'joint_names' and 'joint_ids' are specified, and are not consistent." (see
        # _build_observations above and task-4-report.md).
        def gripper_cfg() -> SceneEntityCfg:
            return SceneEntityCfg("robot", joint_names=arm.gripper_joint_names)

        def grip_params() -> dict:
            return {"robot_cfg": gripper_cfg(), "open_pos": arm.gripper_open, "closed_pos": arm.gripper_closed}

        self.terminations.success.params = {
            "inner_radius": bowl.inner_radius,
            "base_thickness": bowl.base_thickness,
            "rim_height": bowl.wall_height,
            "item_radius": self.food.item_radius,
            "settle_steps": self.success_settle_steps,
            **grip_params(),
        }
        hover = bowl.base_thickness + bowl.wall_height + self.food.item_radius + 0.03
        self.rewards.grasp.params.update(grip_params())
        self.rewards.grasp_lift.params.update(grip_params())
        # transport/transport_fine share grasp_lift's lift_height so the three staged terms agree on what
        # "holding it" means (see rewards.transport_to_bowl's docstring for why this gate exists).
        lift_height = self.rewards.grasp_lift.params["lift_height"]
        self.rewards.transport.params.update(hover_height=hover, lift_height=lift_height, **grip_params())
        self.rewards.transport_fine.params.update(hover_height=hover, lift_height=lift_height, **grip_params())
        # one-shot terms: undo Isaac Lab's step_dt scaling so the return changes by exactly the bonus/penalty
        self.rewards.place_success.weight = self.success_bonus / step_dt
        self.rewards.bowl_failure.weight = -self.bowl_failure_penalty / step_dt
        self.rewards.food_dropped.weight = -self.food_drop_penalty / step_dt
