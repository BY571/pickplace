"""Isaac Sim application launcher. Call before importing torch in every simulator process."""

from __future__ import annotations

import sys


def launch_app(headless: bool = True, enable_cameras: bool = False, device: str | None = None):
    """Start Isaac Sim through Isaac Lab's ``AppLauncher`` and return the ``SimulationApp``.

    Raises:
        RuntimeError: if ``torch`` was imported before the app was launched (Isaac Sim requirement).
    """
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before launching Isaac Sim. Call food_robot.app.launch_app() at the very top "
            "of the entry point, before importing torch, torchrl, tensordict or food_robot.envs."
        )
    from isaaclab.app import AppLauncher

    kwargs: dict = {"headless": headless, "enable_cameras": enable_cameras}
    if device is not None:
        kwargs["device"] = device
    return AppLauncher(**kwargs).app
