"""Scripted, policy-free probe that the grasp is physically possible (task 13).

Drives the arm to the food, closes the gripper, lifts 0.25 m, and reports whether the food came along, at
the env's default food friction range. This is what justifies the ``finger_friction`` /
``RigidFoodCfg`` friction choices (``food_robot/arms/base.py``, ``food_robot/food/rigid.py``) instead of
guesswork: with ``friction_sweep``, the whole probe reruns once per food friction value, each in its own
subprocess (one Isaac Sim per process, as ``scripts/benchmark_pixels.py`` does), via a degenerate
``static_friction_range == dynamic_friction_range == (f, f)``, producing a measured slip-vs-friction curve.

Usage:
    python scripts/probe_grasp.py [num_envs=64]
    python scripts/probe_grasp.py friction_sweep=0.3,0.6,1.0,1.4 [num_envs=64]
    python scripts/probe_grasp.py worker [num_envs=64] [food_friction=0.3]    (internal, one configuration)

Prints one ``PROBE {json}`` line per configuration, then ``PROBE_DONE``.

Orientation (task 14): the probe commands and holds a top-down end-effector orientation through align,
descend, close, lift and hold, instead of leaving the rotation part of the ``ee_delta_pose`` action at zero.
task-13-debug-report.md measured why that mattered: in relative-mode IK a zero rotation command means "do
not correct", so with the Franka's reset pose ~44 deg off top-down the whole descent happened diagonally,
which drove ``panda_joint6`` into its 3.7525 rad hardware limit. Commanding the correction removes that
saturation entirely (q6 peaks at ~2.85 rad in the debug report's measurements) -- see ``docs/environment.md``
for the current numbers on this probe. The other half of task 13's finding -- the old ingredient bowl's 5 cm
rim sat 1 cm above the top of the food, so even a perfectly vertical gripper bottomed its hand on the rim and
squeezed the food out -- is fixed separately, by widening and lowering the supply tray
(``FoodCellEnvCfg.supply_bowl``); this file does not need to know about that geometry.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Phase timings for the scripted motion (settle -> align -> descend -> close -> lift -> hold), shared with
# tests/isaac/scenarios/grasp_probe.py, which imports run_grasp_probe from this file. `align` hovers above
# the food before `descend` comes straight down onto it (approaching directly, a single 3D target from
# wherever the arm settles, made the TCP graze the ingredient bowl's rim and stall).
#
# ALIGN_HEIGHT (6 cm) predates the orientation fix below and is kept at the same value: task-13-debug-report.md
# measured that the joint-6 saturation this comment used to attribute to hover height was actually caused by
# never commanding orientation (see ROT_GAIN below) -- with orientation held, the `topdown`/`orienthold`
# variants reached the food with q6 <= 2.85 rad regardless of hover height. There is no evidence a taller
# hover is needed or harmful; 6 cm is simply unchanged from task 13.
SETTLE_S = 0.5
ALIGN_HEIGHT = 0.06  # TCP height above the food while aligning horizontally
ALIGN_TIMEOUT_S = 2.0  # safety cap; the real transition is TCP-food(hover point) distance < ALIGN_THRESHOLD
ALIGN_THRESHOLD = 0.01
DESCEND_TIMEOUT_S = 1.0  # safety cap; the closest approach is reached quickly, then the food starts to roll
DESCEND_THRESHOLD = 0.010
# DESCEND_THRESHOLD tightened from 0.025 (task 14): with orientation held and the old 5 cm rim, 0.025 m was
# roughly the closest the TCP could ever get (task-13-debug-report.md's closest-approach column). With the
# rim gone, the TCP can and must get much closer -- the debug report's `orienthold` variants measured a
# closest approach of 0.0065-0.0163 m -- and the old, looser threshold was ending `descend` before the
# fingers reached the sphere's equator, closing on its narrow upper cap instead and squeezing it out.
CLOSE_S = 0.5
LIFT_S = 1.5
HOLD_S = 0.5
LIFT_HEIGHT = 0.25
GAIN = 4.0  # position error [m] -> action; clamped to [-1, 1] and scaled by the arm's IK action scale
LIFT_GAIN = 2.0  # gentler than GAIN for the `lift` phase alone: a handful of envs (spawns near the tray's
# edge) lost an otherwise solid grip (orientation error ~0 rad through `close`) in the first ~0.1 s of
# `lift`, where full GAIN commands the fastest allowed ascent and the resulting jerk showed up as a transient
# orientation error (measured 0.03-0.16 rad axis-angle over the following ~0.2 s) that a marginal grip did
# not survive. LIFT_S (1.5 s) has ample margin for a slower ascent.
ROT_GAIN = 4.0  # orientation error [rad, axis-angle] -> action; same clamp/scale treatment as GAIN
LIFTED_HEIGHT = 0.10  # matches grasp_lift's lift_height


def _rotate_local_z_xyzw(quat_xyzw) -> "torch.Tensor":
    """World direction of the hand's local +z axis for each env, given its orientation (x, y, z, w).

    The end-effector's TCP offset (``ArmCfg.tcp_offset``) is a pure +z translation in the hand's local frame,
    so this is exactly the "approach axis" direction task-13-debug-report.md measured convention-free.
    """
    import torch

    qxyz = quat_xyzw[..., :3]
    qw = quat_xyzw[..., 3:4]
    v = torch.zeros_like(qxyz)
    v[..., 2] = 1.0
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


def _topdown_axis_angle_error(quat_xyzw) -> "torch.Tensor":
    """Axis-angle rotation (world frame) that would rotate the hand's approach axis onto straight down.

    This is a *correction*, not a fixed target quaternion: it depends only on the current orientation, is
    zero once the gripper is exactly top-down, and says nothing about roll around the approach axis (free,
    since the food is a sphere and the fingers close symmetrically over it). Feeding it back every step as
    the ``ee_delta_pose`` rotation command is what "holds" a top-down orientation under relative-mode IK,
    where a zero command instead means "do not correct" (task-13-debug-report.md, H1).
    """
    import torch

    approach = _rotate_local_z_xyzw(quat_xyzw)
    down = torch.zeros_like(approach)
    down[..., 2] = -1.0
    axis = torch.cross(approach, down, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angle = torch.atan2(axis_norm.squeeze(-1), (approach * down).sum(-1))
    unit_axis = axis / axis_norm.clamp(min=1e-6)
    return unit_axis * angle.unsqueeze(-1)


def run_grasp_probe(cfg, capture_transport: bool = False) -> dict:
    """Build the env from ``cfg``, drive the scripted grasp motion, and report the outcome.

    Returns ``lifted_fraction`` (fraction of envs with final food height > ``LIFTED_HEIGHT``),
    ``max_food_height`` and ``slip_distance`` (mean final ``|food - tcp|``, all across envs at the end of
    the ``hold`` phase). If ``capture_transport``, also returns ``transport_while_resting`` /
    ``transport_while_lifted``: the ``transport`` reward term's value for each env, captured the instant its
    own gripper finishes closing (food still resting between the fingers on the table, not yet lifted) and
    the instant its own ``hold`` phase ends (food lifted), averaged across envs (envs that never reach that
    instant count as 0, same as the reward they'd actually receive by staying in an earlier phase).
    """
    import gymnasium as gym
    import torch

    import food_robot.envs  # noqa: F401
    from food_robot.envs.mdp.rewards import transport_to_bowl

    if cfg.action_mode != "ee_delta_pose":
        raise ValueError("run_grasp_probe needs env.action_mode='ee_delta_pose'.")

    env = gym.make("FoodRobot-Cell-v0", cfg=cfg)
    u = env.unwrapped
    device, dt, n = u.device, u.step_dt, cfg.scene.num_envs
    action_dim = u.action_manager.total_action_dim
    transport_params = dict(u.reward_manager.get_term_cfg("transport").params) if capture_transport else None

    obs, _ = env.reset()
    # phase: 0 settle, 1 align (hover above), 2 descend (straight down), 3 close, 4 lift, 5 hold, 6 done
    phase = torch.zeros(n, dtype=torch.long, device=device)
    phase_t = torch.zeros(n, device=device)
    descend_target = torch.zeros(n, 3, device=device)
    lift_target = torch.zeros(n, 3, device=device)
    transport_resting = torch.zeros(n, device=device)
    transport_lifted = torch.zeros(n, device=device)

    debug = bool(os.environ.get("PROBE_DEBUG"))
    debug_idx = int(os.environ.get("PROBE_DEBUG_IDX", "0"))
    max_steps = int((SETTLE_S + ALIGN_TIMEOUT_S + DESCEND_TIMEOUT_S + CLOSE_S + LIFT_S + HOLD_S) / dt) + 10
    for step in range(max_steps):
        settle, align, descend, close, lift, hold, done = (phase == i for i in range(7))

        action = torch.zeros(n, action_dim, device=device)
        action[:, -1] = 1.0  # binary gripper: positive = open
        # task 14: keep the gripper closed through `done` too, not just close|lift|hold. Envs that finish
        # their own hold phase before the slowest env in the batch used to fall through to the "default
        # open" branch above and drop the food while waiting -- invisible before task 14 because no env
        # ever got this far. `done` never issues a translation/orientation command either (see `move`/
        # `orient` below), so a finished env just sits still with the food in hand until the batch ends.
        action[close | lift | hold | done, -1] = -1.0

        food_pos_prev = obs["privileged"]["food_pos"]
        target = torch.zeros(n, 3, device=device)
        target[align] = food_pos_prev[align]
        target[align, 2] += ALIGN_HEIGHT
        target[descend] = descend_target[descend]
        target[lift | hold] = lift_target[lift | hold]
        move = align | descend | hold
        err = target[move] - obs["proprio"]["ee_pos"][move]
        action[move, :3] = (GAIN * err).clamp(-1.0, 1.0)
        # `lift` gets its own, gentler gain -- see LIFT_GAIN's comment above.
        lift_err = target[lift] - obs["proprio"]["ee_pos"][lift]
        action[lift, :3] = (LIFT_GAIN * lift_err).clamp(-1.0, 1.0)

        # Command and hold a top-down orientation through every active phase (not just `move`: `close`
        # holds position but must keep correcting orientation too, or the wrist drifts while the fingers
        # shut). See task-13-debug-report.md, H1: a zero rotation command in relative-mode IK means "do not
        # correct", which is what let the reset's ~44 deg tilt persist through descend and cause the
        # joint-6 saturation this fixes.
        orient = align | descend | close | lift | hold
        rot_err = _topdown_axis_angle_error(obs["proprio"]["ee_quat"])
        action[orient, 3:6] = (ROT_GAIN * rot_err[orient]).clamp(-1.0, 1.0)

        obs, *_ = env.step(action)
        phase_t += dt

        food_pos, tcp_pos = obs["privileged"]["food_pos"], obs["proprio"]["ee_pos"]
        dist = torch.linalg.vector_norm(food_pos - tcp_pos, dim=-1)
        hover_dist = torch.linalg.vector_norm(target - tcp_pos, dim=-1)  # only meaningful where align/descend
        to_align = settle & (phase_t >= SETTLE_S)
        to_descend = align & ((hover_dist < ALIGN_THRESHOLD) | (phase_t >= ALIGN_TIMEOUT_S))
        to_close = descend & ((dist < DESCEND_THRESHOLD) | (phase_t >= DESCEND_TIMEOUT_S))
        to_lift = close & (phase_t >= CLOSE_S)
        to_hold = lift & (phase_t >= LIFT_S)
        to_done = hold & (phase_t >= HOLD_S)

        if bool(to_descend.any()):
            # Freeze the descend target at the moment alignment finishes (rather than continuously re-tracking
            # the live food position): re-targeting a food item that is itself gently rolling from the
            # gripper's approach turns into a chase that never converges (see task-13-report.md).
            descend_target[to_descend] = food_pos[to_descend]

        if bool(to_close.any()):
            lift_target[to_close] = tcp_pos[to_close]
            lift_target[to_close, 2] += LIFT_HEIGHT

        if capture_transport and (bool(to_lift.any()) or bool(to_done.any())):
            value = transport_to_bowl(u, **transport_params)
            transport_resting[to_lift] = value[to_lift]
            transport_lifted[to_done] = value[to_done]

        for trans, new_phase in (
            (to_align, 1), (to_descend, 2), (to_close, 3), (to_lift, 4), (to_hold, 5), (to_done, 6)
        ):
            phase[trans] = new_phase
            phase_t[trans] = 0.0

        if debug and step % 5 == 0:
            i = debug_idx
            tilt = float(torch.linalg.vector_norm(rot_err[i]))
            print(
                f"DEBUG step={step} phase{i}={int(phase[i])} phase_t{i}={float(phase_t[i]):.3f} "
                f"food{i}={food_pos[i].tolist()} tcp{i}={tcp_pos[i].tolist()} dist{i}={float(dist[i]):.4f} "
                f"hover_dist{i}={float(hover_dist[i]):.4f} tilt{i}={tilt:.4f}",
                flush=True,
            )

        if bool((phase == 6).all()):
            break

    food_z = obs["privileged"]["food_pos"][:, 2]
    slip = torch.linalg.vector_norm(obs["privileged"]["food_pos"] - obs["proprio"]["ee_pos"], dim=-1)
    if debug:
        print(f"DEBUG final food_z={food_z.tolist()}", flush=True)
        print(f"DEBUG final dist={slip.tolist()}", flush=True)
    result = {
        "lifted_fraction": float((food_z > LIFTED_HEIGHT).float().mean()),
        "max_food_height": float(food_z.max()),
        "slip_distance": float(slip.mean()),
    }
    if capture_transport:
        result["transport_while_resting"] = float(transport_resting.mean())
        result["transport_while_lifted"] = float(transport_lifted.mean())
    env.close()
    return result


def _run_worker(num_envs: int, food_friction: float | None) -> None:
    from food_robot.app import launch_app

    launch_app(headless=True)

    from food_robot.config import build_cell_env_cfg

    food_params = {}
    if food_friction is not None:
        food_params = {
            "static_friction_range": [food_friction, food_friction],
            "dynamic_friction_range": [food_friction, food_friction],
        }
    cfg = build_cell_env_cfg(
        {"num_envs": num_envs, "cameras": False, "privileged_information": True, "food_params": food_params}
    )
    result = run_grasp_probe(cfg)
    if food_friction is not None:
        result = {"food_friction": food_friction, **result}
    print("PROBE " + json.dumps(result), flush=True)
    os._exit(0)  # Isaac Sim shutdown can hang


def main() -> None:
    argv = sys.argv[1:]
    positional = [a for a in argv if "=" not in a]
    kv = dict(a.split("=", 1) for a in argv if "=" in a)
    num_envs = int(kv.get("num_envs", 64))

    if positional and positional[0] == "worker":
        food_friction = kv.get("food_friction")
        _run_worker(num_envs, float(food_friction) if food_friction is not None else None)
        return

    sweep = kv.get("friction_sweep")
    values = [float(x) for x in sweep.split(",")] if sweep else [None]
    for value in values:
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", f"num_envs={num_envs}"]
        if value is not None:
            cmd.append(f"food_friction={value}")
        subprocess.run(cmd, cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, check=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
