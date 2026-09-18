"""scale_physx_buffers: capacities at 16 and 16384 envs (config only; no simulation is created)."""

from _common import finish

from food_robot.app import launch_app

app = launch_app(headless=True)

from food_robot.config import build_cell_env_cfg  # noqa: E402
from food_robot.envs.cell_env_cfg import SCALED_PHYSX_FIELDS  # noqa: E402


def capacities(n):
    physics = build_cell_env_cfg({"num_envs": n, "cameras": False, "privileged_information": True}).sim.physics
    return {name: int(getattr(physics, name)) for name in SCALED_PHYSX_FIELDS}


try:
    finish(True, n16=capacities(16), n16384=capacities(16384), scaled_fields=list(SCALED_PHYSX_FIELDS))
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
