# pickplace/food/base.py
"""Food-source plug-in: the asset plus the manager terms it contributes."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.configclass import configclass


@configclass
class FoodSourceCfg:
    name: str = MISSING
    asset: RigidObjectCfg = MISSING
    """Food asset (prim_path and init_state are overwritten by the env)."""
    item_radius: float = MISSING
    """Characteristic half-size [m]; used for spawn height and success geometry."""
    num_items: int = 1
    success_items: int = 1
    events: dict = {}
    """name -> EventTermCfg merged into the env's EventCfg (use SceneEntityCfg("food")).

    Merged as ``copy.deepcopy(term)`` so a single food cfg instance can build multiple envs
    without the envs sharing (and mutating) the same term object.
    """
    privileged_obs: dict = {}
    """name -> ObservationTermCfg merged into the privileged group when enabled.

    Merged as ``copy.deepcopy(term)``, for the same reason as ``events``.
    """
