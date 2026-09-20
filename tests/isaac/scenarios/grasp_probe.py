"""Grasp scenario: reuses scripts/probe_grasp.py's scripted grasp motion (settle -> descend -> close -> lift
-> hold) at the env's default food/finger friction and reports whether the food came along.

With ``--transport-gate`` it additionally captures the `transport` and `grasp` reward terms' values while the
food merely rests between the fingers on the table (right as the gripper finishes closing) versus while it
is lifted (right as the hold phase ends) -- `transport` must pay nothing for the former (the case task 13
closes off), while `grasp` -- the stepping-stone term added for run 3 -- must pay for it.
"""

import importlib.util
import sys
from pathlib import Path

from _common import finish

TRANSPORT_GATE = "--transport-gate" in sys.argv

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True)

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("probe_grasp", REPO / "scripts" / "probe_grasp.py")
probe_grasp = importlib.util.module_from_spec(_spec)
sys.modules["probe_grasp"] = probe_grasp
_spec.loader.exec_module(probe_grasp)

import pickplace.envs  # noqa: E402,F401
from pickplace.config import build_cell_env_cfg  # noqa: E402

N = 16


def main():
    cfg = build_cell_env_cfg({"num_envs": N, "cameras": False, "privileged_information": True})
    result = probe_grasp.run_grasp_probe(cfg, capture_transport=TRANSPORT_GATE)
    finish(True, **result)


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
