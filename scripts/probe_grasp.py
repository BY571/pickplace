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

Caveat (task 13): as currently tuned, the scripted approach cannot reliably bring the TCP into contact
with the food at all -- ``panda_joint6`` pins at its hardware limit a few cm short, independent of food
friction (0% lift at every swept value, 0.3-1.4). See ``docs/environment.md``'s "Known limitation" note
under ``ArmCfg`` and ``task-13-report.md`` for the full diagnosis; this is a reach/kinematics issue, not
something the friction values in ``RigidFoodCfg``/``ArmCfg.finger_friction`` can fix.
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
# ALIGN_HEIGHT is deliberately low (6 cm, not e.g. 15-20 cm): with a fixed end-effector orientation (no
# rotation command -- see below), reaching down to the food from a *higher* hover point drives
# panda_joint6 into its hardware limit (measured on the Spark: it pins at 3.7525 rad, the real Panda's
# documented joint-6 maximum) before the TCP gets within a couple of cm of the food; a lower hover leaves
# less vertical travel for `descend` and keeps joint6 further from that limit. Adding a corrective
# rotation command during `align` (tried explicitly: axis-angle indices 3 and 4, both signs) did not avoid
# the limit -- panda_joint6 pinned at the same 3.7525 rad regardless -- so this is a real kinematic
# constraint of the arm for this reach, not a controller bug, and DESCEND_THRESHOLD is set to what the
# probe can actually reach (~2.2-2.4 cm) rather than an arbitrary tighter value. See task-13-report.md.
SETTLE_S = 0.5
ALIGN_HEIGHT = 0.06  # TCP height above the food while aligning horizontally
ALIGN_TIMEOUT_S = 2.0  # safety cap; the real transition is TCP-food(hover point) distance < ALIGN_THRESHOLD
ALIGN_THRESHOLD = 0.01
DESCEND_TIMEOUT_S = 0.6  # safety cap; the closest approach is reached quickly, then the food starts to roll
DESCEND_THRESHOLD = 0.025
CLOSE_S = 0.5
LIFT_S = 1.5
HOLD_S = 0.5
LIFT_HEIGHT = 0.25
GAIN = 4.0  # position error [m] -> action; clamped to [-1, 1] and scaled by the arm's IK action scale
LIFTED_HEIGHT = 0.10  # matches grasp_lift's lift_height


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
    max_steps = int((SETTLE_S + ALIGN_TIMEOUT_S + DESCEND_TIMEOUT_S + CLOSE_S + LIFT_S + HOLD_S) / dt) + 10
    for step in range(max_steps):
        settle, align, descend, close, lift, hold = (phase == i for i in range(6))

        action = torch.zeros(n, action_dim, device=device)
        action[:, -1] = 1.0  # binary gripper: positive = open
        action[close | lift | hold, -1] = -1.0

        food_pos_prev = obs["privileged"]["food_pos"]
        target = torch.zeros(n, 3, device=device)
        target[align] = food_pos_prev[align]
        target[align, 2] += ALIGN_HEIGHT
        target[descend] = descend_target[descend]
        target[lift | hold] = lift_target[lift | hold]
        move = align | descend | lift | hold
        err = target[move] - obs["proprio"]["ee_pos"][move]
        action[move, :3] = (GAIN * err).clamp(-1.0, 1.0)

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
            print(
                f"DEBUG step={step} phase0={int(phase[0])} phase_t0={float(phase_t[0]):.3f} "
                f"food0={food_pos[0].tolist()} tcp0={tcp_pos[0].tolist()} dist0={float(dist[0]):.4f} "
                f"hover_dist0={float(hover_dist[0]):.4f}",
                flush=True,
            )

        if bool((phase == 6).all()):
            break

    food_z = obs["privileged"]["food_pos"][:, 2]
    slip = torch.linalg.vector_norm(obs["privileged"]["food_pos"] - obs["proprio"]["ee_pos"], dim=-1)
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
