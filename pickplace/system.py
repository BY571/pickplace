"""System-resource helpers. Stdlib-only (no torch): callable before ``launch_app()``.

Every simulator process must call ``pickplace.app.launch_app(...)`` before importing torch, so
this module must never import torch or tensordict (unlike ``pickplace.training``), letting
pipeline scripts read the memory baseline before launching the app.
"""

from __future__ import annotations


def memory_used_gb() -> float:
    """System-wide used memory [GB]; on the DGX Spark CPU and GPU share this unified memory."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / 1024**2
