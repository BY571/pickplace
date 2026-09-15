import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCENARIOS = Path(__file__).parent / "scenarios"
REPO = Path(__file__).resolve().parents[2]


def _tail(text: str, n: int = 60) -> str:
    return "\n".join(text.splitlines()[-n:])


@pytest.fixture
def run_scenario():
    def _run(name: str, *args, timeout: int = 1200) -> dict:
        cmd = [sys.executable, str(SCENARIOS / f"{name}.py"), *map(str, args)]
        env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
        lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            raise AssertionError(
                f"scenario {name} {args} produced no RESULT (exit {proc.returncode})\n"
                f"--- stdout ---\n{_tail(proc.stdout)}\n--- stderr ---\n{_tail(proc.stderr)}"
            )
        result = json.loads(lines[-1][len("RESULT "):])
        assert result["ok"], f"scenario {name} {args} failed: {result}\n--- stderr ---\n{_tail(proc.stderr)}"
        return result

    return _run
