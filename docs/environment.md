# Food cell environment

A robot arm stands next to a conveyor belt. Each episode one empty bowl rides the belt through the arm's
reach zone. The arm must pick a food item from the ingredient bowl and place it into the moving bowl before
the bowl leaves the zone — without knocking the bowl over or off the belt.

- Simulator: Isaac Lab 3 (PhysX), manager-based env, gym id `FoodRobot-Cell-v0`
- TorchRL: `food_robot.torchrl_env.make_env(cfg)` → `TransformedEnv(IsaacLabWrapper(...))`
- Control rate: 50 Hz (`sim.dt = 0.01`, `decimation = 2`)

## Layout (cell frame = env origin, meters)

```
 y
 ▲   belt (y = 0.30) ══[ entry ][======= reach zone =======][ exit ]══▶ x
 │                       bowl on a moving pallet
 │
 ●───────▶ x   Franka base at (0, 0, 0), table top at z = 0
 │
 │   ingredient bowl at (0.45, -0.30) with the food item
```

The bowl rests freely on a pallet driven by a prismatic joint whose velocity equals the belt speed, so
friction carries the bowl (and food inside it) exactly like a real conveyor; the bowl can still be pushed,
tipped or knocked off.

## Episode

| Phase | What happens |
|---|---|
| Reset | robot joints ± 0.02 rad; food spawns in the ingredient bowl (± `food.spawn_range`, via the food plug-in's `reset_food` event); pallet at the belt entry; belt speed = `speed · (1 + U(±speed_noise))`; bowl offset on the pallet sampled from `bowl_offset_x/y` |
| Running | bowl moves through the zone; the arm has ≈ `place_window` seconds |
| End | first termination below |

| Termination | Kind | Condition |
|---|---|---|
| `success` | terminated | food inside the bowl, released, at rest relative to the bowl for `success_settle_steps` steps |
| `bowl_exited_zone` | terminated | bowl passed the end of the reach zone (deadline missed) |
| `bowl_off_belt` | terminated | bowl below the pallet surface − 2 cm, or beyond the belt half-width |
| `bowl_tipped` | terminated | bowl tilted more than 45° |
| `food_off_table` | terminated | food below z = −5 cm |
| `time_out` | truncated | safety cap: `(entry_margin + zone_length + 0.1) / (speed · (1 − speed_noise)) + 2 s` |

The zone exit is *terminated*, not truncated: the deadline is part of the task and the bowl position is
observed, so value targets must not bootstrap past it. `time_out` is a generous upper bound (the slowest
possible belt speed traversing the whole zone plus margin, plus 2 s of slack); in practice almost every
episode ends via one of the `terminated` conditions above well before `time_out` fires.

### Termination-fraction metrics

`episode_termination/<term>` (from `torchrl_env.termination_stats`, logged by `sota-implementations/ppo`)
reports, per termination term, the fraction of sub-envs whose *most recently finished* episode ended by
that term. Two things to keep in mind when reading it:

- A sub-env that has not yet finished any episode counts as 0 for every term (not excluded from the
  denominator), so early in training these fractions can look artificially low.
- Because it is a per-term fraction over the same set of sub-envs rather than a single categorical
  distribution, the fractions can sum to more than 1 if multiple terms have fired (for different
  sub-envs' most recent episodes) since the stat was last read.

## Observations

All groups are nested (`concatenate_terms=False`), so TorchRL keys look like `("proprio", "ee_pos")`.

| Group | Present when | Term | Shape (Franka) | Meaning |
|---|---|---|---|---|
| `proprio` | always | `joint_pos_rel` | 9 | joint positions minus defaults |
| | | `joint_vel_rel` | 9 | joint velocities minus defaults |
| | | `gripper_pos` | 2 | finger joint positions |
| | | `ee_pos` | 3 | tool center point position, cell frame |
| | | `ee_quat` | 4 | tool orientation (x, y, z, w) |
| | | `last_action` | 7 / 8 | previous action |
| `belt` | always | `bowl_pos` | 3 | bowl position, cell frame |
| `pixels` | `cameras=True` | `wrist_rgb`, `overview_rgb` | H×W×3 | camera images from the `image_float` term: float32, values in [0, 255] (not normalized) |
| `privileged` | `privileged_information=True` | `food_pos` | 3 | food position, cell frame (simulation only) |
| | | `food_quat` | 4 | food orientation (simulation only) |
| | | `is_grasped` | 1 | 1.0 if the food is held (simulation only) |

`cameras=False` together with `privileged_information=False` is rejected: no observation would tell the
policy where the food is. Privileged terms are never available on a real robot — use them for critics,
teachers or debugging, not for deployable actors.

Belt speed is intentionally not observed: the line runs at a fixed speed.

#### Camera memory budget

`image_float` returns float32 (4 bytes/channel), 4x the native uint8 camera buffer (1 byte/channel),
because Isaac Lab's `ManagerBasedRLEnv` always declares observation spaces as float32 (see the term's
docstring in `food_robot/envs/mdp/observations.py`). TorchRL additionally stores one copy of every
observation at the root of a transition and one at `"next"`, so each camera term costs roughly:

```
bytes_per_frame ≈ n_cams × H × W × 3 × 4     (root)
bytes_per_frame ≈ n_cams × H × W × 3 × 4     (next)
total ≈ num_envs × 2 × n_cams × H × W × 3 × 4 bytes, per buffered transition
```

With the default two cameras (`wrist_rgb`, `overview_rgb`) at `image_size=(128, 128)`: `2 × 128 × 128 ×
3 × 4 = 393,216` bytes/env for one copy, so `≈ 786 KB/env` for root + next per transition held in the
collector/replay buffer (before accounting for `frames_per_batch` rollout steps buffered at once).
Multiply by `frames_per_batch = num_envs × collector.rollout_steps` to get the buffer's peak size, e.g.
at `rollout_steps=24` that is `≈ 18.9 MB/env` per PPO iteration's buffer.

On the 128 GB unified-memory Spark, leaving headroom for PhysX/rendering buffers and the network, a
reasonable starting range with both default cameras at 128×128 is `num_envs` in the low hundreds to
~1000 for camera-observing policies (`cameras=True`); the privileged/state-only teacher setup
(`cameras=False`) has no such constraint and is what the `sota-implementations/ppo/README.md` scale run
uses at `num_envs=4096`. `sota-implementations/ppo/ppo.py` also drops the `"next"` sub-tensordict before
it reaches the replay buffer (`ClipPPOLoss` only needs root observations, action, log-prob, advantage
and value_target), which roughly halves the buffered camera memory versus the naive formula above.

## Actions

| `action_mode` | Arm part | Gripper | Franka dim |
|---|---|---|---|
| `ee_delta_pose` (default) | differential IK, relative position (3) + axis-angle (3), scaled by `ik_action_scale` | binary: > 0 open, < 0 close | 7 |
| `joint_pos` | joint position offsets from defaults, scaled by `joint_action_scale` | binary | 8 |

`ee_delta_pose` has the same shape for every arm, so it is the portable choice.

The TorchRL action spec is unbounded (Isaac Lab declares no action limits), but actions are meant to lie in
[-1, 1]; the action terms multiply them by the arm's action scale. Squash policy outputs accordingly (the PPO
example uses a `TanhNormal` on [-1, 1]).

## Rewards

Isaac Lab multiplies each term by the step duration; one-shot terms are compensated so the episode return
changes by exactly the configured bonus/penalty.

| Term | Weight | Signal |
|---|---|---|
| `reach_food` | 1 | `1 − tanh(‖tcp − food‖ / 0.1)` |
| `grasp_lift` | 5 | food grasped and above z = 0.10 |
| `transport` | 10 | while grasped: `1 − tanh(‖food − above(bowl)‖ / 0.3)` (moving target) |
| `transport_fine` | 5 | same with σ = 0.05 |
| `place_success` | +`success_bonus` (150) once | success termination |
| `bowl_failure` | −`bowl_failure_penalty` (150) once | bowl off belt or tipped |
| `food_dropped` | −`food_drop_penalty` (150) once | food off table |
| `bowl_disturbance` | −1 | distance the bowl was pushed from where the pallet carries it |
| `action_rate`, `joint_vel` | −1e-4 | regularization |

Each one-shot penalty/bonus is deliberately large enough that ending an episode early is never more
profitable than a real attempt: dense shaping is non-negative, so early-ending never *gains* reward by
itself, and the maximum dense shaping obtainable over a realistic zone traversal (≈ 129) or even by
holding the bowl until `time_out` (≈ 189) does not exceed the 150 bonus/penalty by enough to make failing
worthwhile once the one-shot term's sign is accounted for. `food_robot/envs/cell_env_cfg.py` documents the
full anti-exploit argument next to the three constants, and asks you to re-check it whenever `belt.speed`,
`belt.speed_noise`, `belt.place_window` or the dense reward weights change.

Reward weights can be overridden per-term through the `rewards: {term_name: weight}` env config (applied
after construction; see below) — except the one-shot terms `place_success`, `bowl_failure` and
`food_dropped`, which are configured through `success_bonus` / `bowl_failure_penalty` / `food_drop_penalty`
instead and are rejected (`KeyError`) if passed in `rewards`.

## Parameters

Configure through constructor kwargs, e.g.
`FoodCellEnvCfg(cameras=False, privileged_information=True, belt=BeltCfg(speed=0.1))`, or through the
`env:` section of an algorithm's Hydra config.

### `FoodCellEnvCfg`

| Parameter | Default | Description |
|---|---|---|
| `arm` | `FRANKA_CFG` | arm plug-in (`ArmCfg`) |
| `food` | `RigidFoodCfg()` | food plug-in (`FoodSourceCfg`) |
| `belt` | `BeltCfg()` | belt plug-in |
| `action_mode` | `"ee_delta_pose"` | `"ee_delta_pose"` or `"joint_pos"` |
| `cameras` | `True` | spawn wrist + overview cameras and add the `pixels` group |
| `image_size` | `(128, 128)` | camera height, width |
| `privileged_information` | `False` | add the simulation-only `privileged` group |
| `ingredient_bowl_pos` | `(0.45, -0.30, 0.0)` | ingredient bowl position, cell frame |
| `overview_cam_eye` / `overview_cam_target` | `(1.6, -0.4, 1.4)` / `(0.35, 0.0, 0.25)` | overview camera placement |
| `render_camera` | `False` | spawn `scene.render_cam`, a wide third-person camera for videos; not an observation, so the TorchRL specs don't change (the app must be launched with cameras enabled) |
| `render_cam_eye` / `render_cam_target` | `(2.3, -1.9, 1.9)` / `(0.3, 0.0, 0.35)` | render camera placement (whole cell in view) |
| `render_image_size` | `(720, 1280)` | render camera height, width |
| `success_bonus` | `150.0` | return added on success |
| `bowl_failure_penalty` | `150.0` | return subtracted when the bowl falls off or tips |
| `food_drop_penalty` | `150.0` | return subtracted when the food falls off the table |
| `success_settle_steps` | `5` | consecutive at-rest steps required for success |
| `scene.num_envs` | `64` | parallel environments |

### `BeltCfg`

| Parameter | Default | Description |
|---|---|---|
| `speed` | `0.08` m/s | nominal line speed |
| `speed_noise` | `0.02` | per-episode relative noise: `speed · (1 + U(±noise))` |
| `place_window` | `5.0` s | time the bowl spends in the reach zone at nominal speed |
| `zone_center_x` | `0.45` m | zone center along the belt; zone length = `speed · place_window` |
| `belt_y` | `0.30` m | belt centerline |
| `belt_half_width` | `0.15` m | lateral limit for `bowl_off_belt` |
| `entry_margin` | `0.05` m | bowl spawns this far before the zone |
| `bowl_offset_x` | `(-0.03, 0.03)` m | reset randomization along the belt (shifts the timing) |
| `bowl_offset_y` | `(-0.02, 0.02)` m | reset randomization across the belt |
| `plate_top_z` | `0.03` m | pallet surface height |
| `pallet_damping` | `1e4` | velocity-drive gain of the pallet joint |
| `bowl` | `BowlGeometry()` | inner radius 7 cm, wall 5 cm, mass 0.2 kg |
| `pallet` | `PalletGeometry()` | size `(0.22, 0.22, 0.01)` m, mass 2.0 kg, `travel_lower=-0.05`, `travel_upper=1.0` m — the prismatic joint's travel range along the belt. `travel_upper` must be ≥ `zone.length + entry_margin + 0.1` (the env raises `ValueError` otherwise), so the pallet can never run out of travel before the safety-cap `time_out`. |

The env raises `ValueError` if the zone endpoints are outside `arm.reach_radius`.

### `RigidFoodCfg`

The rigid-food plug-in owns everything food-specific: the asset, its startup material/mass
randomization, the per-reset `reset_food` event (pose randomization around the ingredient bowl), and the
`food_quat` privileged observation term — merged into the env's `EventCfg` / privileged observation group
by `FoodCellEnvCfg._build_events` / `_build_observations` (each term is `copy.deepcopy`'d, so one
`RigidFoodCfg` instance can be reused to build multiple envs).

| Parameter | Default | Description |
|---|---|---|
| `item_radius` | `0.02` m | sphere radius |
| `mass` | `0.03` kg | nominal mass |
| `static_friction_range` | `(0.3, 1.0)` | randomized once at startup per env |
| `dynamic_friction_range` | `(0.2, 0.8)` | randomized once at startup per env |
| `restitution_range` | `(0.0, 0.1)` | randomized once at startup per env |
| `mass_scale_range` | `(0.7, 1.3)` | mass scale, randomized once at startup per env |
| `spawn_range` | `0.02` m | ± xy food spawn randomization around the ingredient bowl center, applied by the `reset_food` event |

### `ArmCfg` (Franka defaults)

| Parameter | Default | Description |
|---|---|---|
| `arm_joint_names` / `gripper_joint_names` | `panda_joint.*` / `panda_finger_joint.*` | joint regexes |
| `ee_body_name` | `panda_hand` | end-effector body |
| `tcp_offset` | `(0, 0, 0.1034)` | tool center point offset |
| `gripper_open` / `gripper_closed` | `0.04` / `0.0` | finger joint targets |
| `reach_radius` | `0.80` m | used for belt-zone validation |
| `ik_action_scale` / `joint_action_scale` | `0.5` / `0.5` | action scaling |

Adding an arm: create an `ArmCfg` with the arm's articulation configs, joint/body names, TCP offset and
wrist camera mount, and register it in `food_robot/config.py`. Observation and action specs follow
automatically.

The Franka's own asset path needs one workaround: `isaaclab_assets` (pinned at v3.0.0-beta2.patch1) still
points at `.../FrankaEmika/panda_instanceable.usd`, but the live Nucleus asset pack moved that file under a
`Legacy/` subpath. `food_robot/arms/franka.py` rewrites the path to
`.../FrankaEmika/Legacy/panda_instanceable.usd` (and raises loudly if neither path shape matches, so a
future Nucleus or `isaaclab_assets` change is caught instead of silently ignored).

### Hydra / plain-mapping config (`food_robot.config.build_cell_env_cfg`)

`build_cell_env_cfg` (used by `make_env` and every `sota-implementations/*` Hydra config's `env:` section)
merges a plain mapping over `DEFAULT_ENV` and builds the `FoodCellEnvCfg` above. In addition to the
constructor kwargs it forwards directly (`num_envs`, `arm`, `food`, `action_mode`, `cameras`, `image_size`,
`privileged_information`, `success_bonus`, `bowl_failure_penalty`, `food_drop_penalty`,
`success_settle_steps`, `ingredient_bowl_pos`, `overview_cam_eye`, `overview_cam_target`, `belt`, `seed`,
`device`), it accepts:

| Key | Meaning |
|---|---|
| `rewards` | `{term_name: weight}`, applied to `cfg.rewards.<term_name>.weight` after construction. Unknown term names raise `KeyError` listing the valid terms; the one-shot terms `place_success`, `bowl_failure`, `food_dropped` raise `KeyError` if passed here (use `success_bonus`/`bowl_failure_penalty`/`food_drop_penalty` instead). |
| `food_params` | Forwarded as constructor kwargs to the food plug-in class (e.g. `{"spawn_range": 0.03, "mass": 0.04}` for `RigidFoodCfg`). |
| `belt.bowl` / `belt.pallet` | Nested dicts, turned into `BowlGeometry(**belt.bowl)` / `PalletGeometry(**belt.pallet)` before constructing `BeltCfg`, e.g. `{"belt": {"pallet": {"travel_upper": 2.0}}}`. |

Unknown top-level keys raise `ValueError` listing the allowed set (`DEFAULT_ENV`'s keys).

## TorchRL usage

```python
from food_robot.app import launch_app
app = launch_app(headless=True, enable_cameras=False)   # before importing torch

from food_robot.torchrl_env import make_env
env = make_env({"task": "FoodRobot-Cell-v0", "num_envs": 16, "cameras": False, "privileged_information": True})
td = env.rollout(10)
print(td["next", "proprio", "ee_pos"].shape)   # (16, 10, 3)
```

Route observations to model parts with `food_robot.keys.expand_in_keys(env.observation_spec, ["proprio", "belt"])`.
Inspect the spec tree: `python scripts/check_env.py env.cameras=false env.privileged_information=true`.

## Training

See `sota-implementations/ppo/README.md`. Camera-trained checkpoints (`env.cameras=true`) must also be
played with `env.cameras=true`: `play.py` reads `env.cameras` from the CLI to launch the app before the
checkpoint's saved config is available, then raises a `ValueError` if the two disagree.
