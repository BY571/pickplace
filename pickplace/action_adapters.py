"""Alternative action parameterizations on top of the cell env's native action modes.

The environment speaks two action languages natively (``pickplace.envs.cell_env_cfg.ActionMode``):
``ee_delta_pose``, a relative end-effector command fed to a differential IK controller, and ``joint_pos``,
*scaled offsets from the robot's default joint pose*. Policies trained elsewhere often speak a third:
**joint velocities** (what the DROID teleoperation stack and the VLAs fine-tuned on it emit), or **absolute
joint positions** (what many motion-planning stacks emit).

Rather than add action modes to the environment — which would change the thing every existing result was
measured on — this module converts at the boundary. ``JointActionAdapter`` is a TorchRL transform: it
rewrites the action spec, so an algorithm samples in the new space, and converts each action into the
env's native one on its way through. The physics, the reward and the observations are untouched, so a run
in a converted action space is comparable to every other run in this repository.

Which space a policy speaks is a real design choice with consequences, see ``docs/environment.md``:
velocities are state-relative and need no IK, but integrate drift and are timing-sensitive; absolute joint
targets are drift-free but encode a pose rather than a motion, so the same action means different things
from different starts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

ADAPTERS = ("native", "joint_velocity", "joint_position")
"""``native`` leaves the env's own action space alone; the others are defined in this module."""


def joint_target_to_native(
    q_target: torch.Tensor, q_default: torch.Tensor, scale: float,
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Absolute joint targets [rad] -> the env's ``joint_pos`` action.

    That action mode applies ``target = default + scale * action`` (``use_default_offset=True``), so the
    inverse is exact. With ``limits`` the target is clamped into the joint range *before* inverting, which
    keeps a policy from commanding through a limit and silently relying on the solver to absorb it.
    """
    if limits is not None:
        q_target = torch.clamp(q_target, limits[0], limits[1])
    return (q_target - q_default) / scale


def integrate_velocity(
    q_dot: torch.Tensor, q_target: torch.Tensor | None, q_measured: torch.Tensor, dt: float,
    band: float | None = None, limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Advance the commanded joint target by one control period of ``q_dot`` [rad/s].

    The integral runs on the *commanded target*, not on the measured position, and this is the whole
    subtlety. A real velocity-controlled arm holds station on a zero command because its own controller
    fights gravity. Ours is position-controlled with the plain (not high-PD) Franka gains, so re-anchoring
    the target to the measured joints every step would let a sagging arm drag its own target down: measured
    drift of 0.26 rad in 10 steps on a zero command, before this was fixed.

    ``band`` is an optional anti-windup for a blocked arm: the target is pulled back to within ``band``
    radians of the measurement before integrating. It defaults to off, because against a *continuously*
    sagging arm a band re-creates the very chasing it was meant to prevent -- the target simply follows the
    sag one band behind. Resets are handled by the caller re-seeding instead (``q_target=None``).
    """
    if q_target is None:
        q_target = q_measured
    elif band is not None:
        q_target = torch.clamp(q_target, q_measured - band, q_measured + band)
    q_target = q_target + q_dot * dt
    if limits is not None:
        q_target = torch.clamp(q_target, limits[0], limits[1])
    return q_target


def joint_velocity_to_native(
    q_dot: torch.Tensor, q_current: torch.Tensor, q_default: torch.Tensor, scale: float, dt: float,
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One-shot velocity conversion, anchored to the measurement: ``q_target = q_current + q_dot * dt``.

    Kept for the stateless case (and for the unit tests that pin the algebra). The transform uses
    ``integrate_velocity`` instead, which holds its own target across steps; see that docstring for why.
    """
    return joint_target_to_native(q_current + q_dot * dt, q_default, scale, limits)


def gripper_fraction_to_binary(fraction: torch.Tensor, closed_at: float = 0.5, closed_is_high: bool = True
                               ) -> torch.Tensor:
    """A continuous gripper command in [0, 1] -> the env's binary gripper action.

    ``BinaryJointPositionActionCfg`` reads only the sign: positive opens the fingers, negative closes them.
    ``closed_is_high`` says which end of the incoming range means closed (DROID: 1.0 is closed), and
    ``closed_at`` is the threshold. The magnitude of the returned action is irrelevant to the env; ±1 is
    used so logs are readable.
    """
    closed = fraction >= closed_at if closed_is_high else fraction <= closed_at
    return torch.where(closed, -torch.ones_like(fraction), torch.ones_like(fraction))


def make_joint_action_adapter(mode: str, *, gripper_closed_at: float = 0.5, gripper_closed_is_high: bool = True,
                              clamp_to_limits: bool = True, velocity_band: float | None = None):
    """Build the TorchRL transform for ``mode`` (imports TorchRL lazily, as the rest of this package does)."""
    from torchrl.data import Bounded, Composite, Unbounded
    from torchrl.envs import Transform

    if mode not in ADAPTERS:
        raise ValueError(f"Unknown action adapter {mode!r}. Available: {list(ADAPTERS)}")

    class JointActionAdapter(Transform):
        """Converts a joint-space action into the env's ``joint_pos`` action on its way to the simulator."""

        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.mode = mode
            self._target = None    # the commanded joint target, integrated across steps (velocity mode)
            self._reseed = None    # rows whose episode ended last step: their target is stale

        # -- the robot's own numbers, read from the simulator rather than configured twice ---------------
        def _robot(self):
            from pickplace.torchrl_env import _unwrapped

            env = _unwrapped(self.parent)
            robot = env.scene["robot"]
            arm = env.cfg.arm
            ids, _ = robot.find_joints(arm.arm_joint_names)
            return env, robot, ids, float(arm.joint_action_scale)

        def _limits(self, robot, ids):
            if not clamp_to_limits:
                return None
            lim = robot.data.soft_joint_pos_limits.torch[:, ids, :]
            return lim[..., 0], lim[..., 1]

        # -- actions travel "inverse" through transforms, from the policy towards the env ---------------
        def _inv_call(self, tensordict):
            action = tensordict.get("action")
            env, robot, ids, scale = self._robot()
            arm_cmd, gripper = action[..., :-1], action[..., -1:]
            q_default = robot.data.default_joint_pos.torch[:, ids]
            limits = self._limits(robot, ids)

            if self.mode == "joint_velocity":
                q_measured = robot.data.joint_pos.torch[:, ids]
                if self._target is not None and self._reseed is not None and self._reseed.any():
                    # an episode ended and the robot was teleported: those rows' targets describe the old
                    # episode, so start them again from where the arm actually is
                    self._target = torch.where(self._reseed.unsqueeze(-1), q_measured, self._target)
                    self._reseed = None
                self._target = integrate_velocity(arm_cmd, self._target, q_measured, env.step_dt,
                                                  band=velocity_band, limits=limits)
                native_arm = joint_target_to_native(self._target, q_default, scale, limits)
            else:  # joint_position
                native_arm = joint_target_to_native(arm_cmd, q_default, scale, limits)

            native_gripper = gripper_fraction_to_binary(gripper, gripper_closed_at, gripper_closed_is_high)
            tensordict.set("action", torch.cat([native_arm, native_gripper], dim=-1))
            return tensordict

        def _step(self, tensordict, next_tensordict):
            done = next_tensordict.get("done")
            self._reseed = done.reshape(done.shape[0], -1).any(-1)
            return next_tensordict

        def _reset(self, tensordict, tensordict_reset):
            self._target = None  # re-seed from the measurement after the robot is teleported
            return tensordict_reset

        def transform_input_spec(self, input_spec):
            """Advertise the adapted space: velocities in rad/s, or joint angles in rad, plus [0, 1] gripper."""
            spec = input_spec["full_action_spec", "action"]
            n = spec.shape[-1]
            if self.mode == "joint_velocity":
                adapted = Unbounded(shape=spec.shape, dtype=spec.dtype, device=spec.device)
            else:
                low = torch.full(spec.shape, -torch.pi, device=spec.device)
                high = torch.full(spec.shape, torch.pi, device=spec.device)
                low[..., -1], high[..., -1] = 0.0, 1.0
                adapted = Bounded(low=low, high=high, shape=spec.shape, dtype=spec.dtype, device=spec.device)
            input_spec["full_action_spec"] = Composite(
                action=adapted, shape=input_spec["full_action_spec"].shape, device=spec.device
            )
            self._action_dim = n
            return input_spec

    return JointActionAdapter()


def describe(mode: str) -> str:
    """One line per adapter, for logs and manifests."""
    return {
        "native": "the env's own action mode (ee_delta_pose or joint_pos), unchanged",
        "joint_velocity": "7 joint velocities [rad/s] + gripper fraction [0, 1], integrated over one step",
        "joint_position": "7 absolute joint angles [rad] + gripper fraction [0, 1]",
    }[mode]


def adapted_action_names(arm_joint_names: Sequence[str], mode: str) -> list[str]:
    """Human-readable per-dimension names, for figures and debugging."""
    if mode == "native":
        return []
    unit = "qd" if mode == "joint_velocity" else "q"
    return [f"{unit}[{name}]" for name in arm_joint_names] + ["gripper"]
