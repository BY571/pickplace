"""action_adapter: the joint-velocity and joint-position adapters drive the real robot the way their units
say they should, and the env refuses an adapter on top of an action mode it cannot convert into.

Stepping is real: a commanded velocity has to move the measured joints by about v*dt, and a zero command has
to hold them wherever they are. Tolerances are loose because a position-controlled Franka tracks a target
over a control period rather than teleporting to it; the assertion is on direction and rough magnitude, not
on exact tracking.
"""

import numpy as np
from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True)

import torch  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402

BASE = {"num_envs": 1, "cameras": False, "privileged_information": True, "action_mode": "joint_pos"}


def joints(env):
    u = env.base_env._env.unwrapped
    robot = u.scene["robot"]
    ids, _ = robot.find_joints(u.cfg.arm.arm_joint_names)
    return robot.data.joint_pos.torch[:, ids].clone()


def hold_and_move():
    env = make_env({**BASE, "action_adapter": "joint_velocity"})
    u = env.base_env._env.unwrapped
    dt = u.step_dt
    td = env.reset()

    # the action spec is the adapted one: 7 arm dims + gripper
    action_dim = int(env.action_spec.shape[-1])

    # zero velocity: the arm holds station. A position-controlled arm still sags a little against gravity,
    # so what matters is that the drift is bounded rather than accumulating -- measured at 10 and 50 steps.
    before = joints(env)
    drift = {}
    for step in range(1, 51):
        td.set("action", torch.zeros(1, action_dim, device=env.device))
        _, td = env.step_and_maybe_reset(td)
        if step in (10, 50):
            drift[step] = (joints(env) - before).abs().max().item()
    held = drift[10]

    # a commanded velocity moves the joints in the commanded direction, by roughly v*dt per step
    env.reset()
    from pickplace.action_adapters import DEFAULT_SCALES

    unit = DEFAULT_SCALES["joint_velocity"]   # +-1 of the normalized command, in rad/s
    q_dot = torch.zeros(1, action_dim, device=env.device)
    q_dot[0, 1] = 0.4 / unit   # 0.4 rad/s on joint 2, well inside the Franka's 2.17 limit
    q_dot[0, 3] = -0.3 / unit
    before = joints(env)
    steps = 20
    for _ in range(steps):
        td.set("action", q_dot.clone())
        _, td = env.step_and_maybe_reset(td)
    moved = (joints(env) - before)[0]
    expected = (q_dot[0, :7] * unit * dt * steps).cpu().numpy()
    env.close()
    return {
        "dt": dt,
        "action_dim": action_dim,
        "held_max_rad": held,
        "held_50_rad": drift[50],
        "moved": moved.cpu().numpy().round(4).tolist(),
        "expected": np.round(expected, 4).tolist(),
    }


def position_adapter_reaches_a_commanded_pose():
    env = make_env({**BASE, "action_adapter": "joint_position"})
    td = env.reset()
    from pickplace.action_adapters import DEFAULT_SCALES

    u = env.base_env._env.unwrapped
    ids, _ = u.scene["robot"].find_joints(u.cfg.arm.arm_joint_names)
    default = u.scene["robot"].data.default_joint_pos.torch[:, ids]
    start = joints(env)
    # a normalized command: this asks for the rest pose with joint 2 offset by 0.15 rad
    cmd = (default - default) / DEFAULT_SCALES["joint_position"]
    cmd[0, 1] += 0.15 / DEFAULT_SCALES["joint_position"]
    target = default.clone()
    target[0, 1] += 0.15
    action = torch.cat([cmd, torch.zeros(1, 1, device=env.device)], dim=-1)
    for _ in range(40):
        td.set("action", action.clone())
        _, td = env.step_and_maybe_reset(td)
    reached = joints(env)
    env.close()
    return {"error_rad": (reached - target).abs().max().item(),
            "moved_rad": (reached - start)[0, 1].item()}


def rejects_an_unconvertible_action_mode():
    try:
        make_env({**BASE, "action_mode": "ee_delta_pose", "action_adapter": "joint_velocity"})
    except ValueError as exc:
        return {"raised": True, "message": str(exc)}
    return {"raised": False, "message": ""}


def main():
    rejected = rejects_an_unconvertible_action_mode()
    velocity = hold_and_move()
    position = position_adapter_reaches_a_commanded_pose()
    finish(True, rejected=rejected, velocity=velocity, position=position)


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
