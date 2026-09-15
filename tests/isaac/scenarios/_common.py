"""Shared helpers for simulator scenario scripts (each runs in its own process)."""

import json
import os
import sys


def finish(ok: bool, **payload) -> None:
    """Print a machine-readable result line and hard-exit (Isaac Sim shutdown can hang)."""
    print("RESULT " + json.dumps({"ok": bool(ok), **payload}, default=float), flush=True)
    sys.stderr.flush()
    os._exit(0 if ok else 1)
