"""Alternative action parameterizations on top of the cell env's native action modes.

The environment speaks two action languages natively (``pickplace.envs.cell_env_cfg.ActionMode``):
``ee_delta_pose``, a relative end-effector command fed to a differential IK controller, and ``joint_pos``,
*scaled offsets from the robot's default joint pose*. Policies trained elsewhere often speak a third:
**joint velocities** (what the DROID teleoperation stack and the VLAs fine-tuned on it emit), or **absolute
joint positions** (what many motion-planning stacks emit).

Rather than add action modes to the environment — which would change the thing every existing result was
measured on — this module converts at the boundary. ``make_joint_action_adapter`` builds a TorchRL
transform: it rewrites the action spec, so an algorithm samples in the new space, and converts each action
into the env's native one on its way through. The physics, the reward and the observations are untouched,
so a run in a converted action space is comparable to every other run in this repository.

Which space a policy speaks is a real design choice with consequences, see ``docs/environment.md``:
velocities are state-relative and need no IK, but integrate drift and are timing-sensitive; absolute joint
targets are drift-free but encode a pose rather than a motion, so the same action means different things
from different starts.
"""

from __future__ import annotations

import torch

ADAPTERS = ("native", "joint_velocity", "joint_position")
"""``native`` leaves the env's own action space alone; the others are defined in this module."""

JOINT_ADAPTERS = ADAPTERS[1:]
"""The adapters this module implements. ``native`` is the absence of one, not a mode to build."""


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
    limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Advance the commanded joint target by one control period of ``q_dot`` [rad/s].

    The integral runs on the *commanded target*, not on the measured position, and this is the whole
    subtlety. A real velocity-controlled arm holds station on a zero command because its own controller
    fights gravity. Ours is position-controlled, so re-anchoring the target to the measured joints every
    step would let a sagging arm drag its own target down: measured drift of 0.26 rad in 0.2 s on a zero
    command, before this was fixed.

    A band-limited anti-windup for a blocked arm (pull the target back to within N radians of the
    measurement before integrating) was tried and rejected: against a *continuously* sagging arm it
    re-creates the very chasing it was meant to prevent, the target simply following the sag one band
    behind. Staleness is handled by re-seeding instead: ``q_target=None`` starts from the measurement.
    """
    if q_target is None:
        q_target = q_measured
    q_target = q_target + q_dot * dt
    if limits is not None:
        q_target = torch.clamp(q_target, limits[0], limits[1])
    return q_target


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


def make_joint_action_adapter(mode: str):
    """Build the transform for ``mode`` (TorchRL is imported lazily, as elsewhere in this package)."""
    if mode not in JOINT_ADAPTERS:
        raise ValueError(f"Unknown action adapter {mode!r}. Available: {list(JOINT_ADAPTERS)}")

    from torchrl.data import Bounded, Unbounded
    from torchrl.envs import Transform

    class JointActionAdapter(Transform):
        """Converts a joint-space action into the env's ``joint_pos`` action on its way to the simulator.

        It is placed last in ``make_env``'s ``Compose``: ``Compose._inv_call`` runs its transforms in
        reverse, so last in the list is first to see an action. ``Transform.inv`` hands the env a shallow
        copy, so what a collector stores stays the action the policy actually sampled.
        """

        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.mode = mode
            self._target = None    # the commanded joint target, integrated across steps (velocity mode)
            self._reseed = None    # rows whose target is stale: their episode ended, or they were reset
            self._ids = None       # arm joint indices: find_joints resolves names by regex, so cache them
            self._scale = None

        def _robot(self):
            from pickplace.torchrl_env import _unwrapped

            env = _unwrapped(self.parent)
            robot = env.scene["robot"]
            if self._ids is None:
                self._ids, _ = robot.find_joints(env.cfg.arm.arm_joint_names)
                self._scale = float(env.cfg.arm.joint_action_scale)
            return env, robot

        def _joint_limits(self):
            """The arm's soft joint range, or None before the transform is attached to an env."""
            try:
                _, robot = self._robot()
            except (AttributeError, KeyError, TypeError):
                return None
            lim = robot.data.soft_joint_pos_limits.torch[0, self._ids, :]
            return lim[..., 0], lim[..., 1]

        def _mark_stale(self, rows: torch.Tensor) -> None:
            self._reseed = rows if self._reseed is None else (self._reseed | rows)

        # -- actions travel "inverse" through transforms, from the policy towards the env ---------------
        def _inv_call(self, tensordict):
            action = tensordict.get("action")
            env, robot = self._robot()
            arm_cmd, gripper = action[..., :-1], action[..., -1:]
            q_default = robot.data.default_joint_pos.torch[:, self._ids]
            lim = robot.data.soft_joint_pos_limits.torch[:, self._ids, :]
            limits = (lim[..., 0], lim[..., 1])

            if self.mode == "joint_velocity":
                q_measured = robot.data.joint_pos.torch[:, self._ids]
                if self._target is not None and self._reseed is not None and self._reseed.any():
                    # these rows' targets describe an episode that has ended or a row that was reset:
                    # start them again from where the arm actually is now
                    self._target = torch.where(self._reseed.unsqueeze(-1), q_measured, self._target)
                self._reseed = None
                self._target = integrate_velocity(arm_cmd, self._target, q_measured, env.step_dt, limits)
                native_arm = joint_target_to_native(self._target, q_default, self._scale, limits)
            else:  # joint_position
                native_arm = joint_target_to_native(arm_cmd, q_default, self._scale, limits)

            tensordict.set("action", torch.cat([native_arm, gripper_fraction_to_binary(gripper)], dim=-1))
            return tensordict

        def _step(self, tensordict, next_tensordict):
            # Under native auto-reset a row whose episode ended is teleported inside step(), so its
            # integrated target is stale. This asks the termination manager rather than reading `done` off
            # the tensordict: transforms above this one in the Compose can *add* a done (StepCounter with
            # max_steps set), and such a row is not physically reset, so re-seeding it would be wrong.
            from pickplace.torchrl_env import _unwrapped

            dones = _unwrapped(self.parent).termination_manager.dones
            self._mark_stale(dones.reshape(dones.shape[0], -1).any(-1))
            return next_tensordict

        def _reset(self, tensordict, tensordict_reset):
            # A reset may cover a subset of the sub-envs (the standard "_reset" mask, which IsaacLabWrapper
            # supports), so only the masked rows are stale. Nulling every row here would throw away the
            # integrated target of rows that were not reset at all.
            mask = None if tensordict is None else tensordict.get("_reset", None)
            if mask is None:
                self._target, self._reseed = None, None
            else:
                self._mark_stale(mask.reshape(mask.shape[0], -1).any(-1))
            return tensordict_reset

        def transform_input_spec(self, input_spec):
            """Advertise the adapted space.

            ``joint_velocity`` is unbounded throughout: the env declares no joint-velocity limits, and a
            spec cannot bound the gripper dimension alone. ``joint_position`` advertises the robot's own
            soft joint limits, so a policy does not spend distribution density on angles ``_inv_call``
            would only clamp away; ±π stands in when the robot cannot be read (no parent yet). The gripper
            is [0, 1] and thresholded either way, so an out-of-range value cannot misbehave.
            """
            spec = input_spec["full_action_spec", "action"]
            if self.mode == "joint_velocity":
                adapted = Unbounded(shape=spec.shape, dtype=spec.dtype, device=spec.device)
            else:
                low = torch.full(spec.shape, -torch.pi, device=spec.device)
                high = torch.full(spec.shape, torch.pi, device=spec.device)
                joint_limits = self._joint_limits()
                if joint_limits is not None:
                    low[..., :-1], high[..., :-1] = joint_limits
                low[..., -1], high[..., -1] = 0.0, 1.0
                adapted = Bounded(low=low, high=high, shape=spec.shape, dtype=spec.dtype, device=spec.device)
            # assign the leaf rather than rebuilding the composite, which would drop any sibling key
            input_spec["full_action_spec", "action"] = adapted
            return input_spec

    return JointActionAdapter()
