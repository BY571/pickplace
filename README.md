# food-robot

Isaac Lab environment of a robot arm placing food into bowls moving on a conveyor belt, runnable with TorchRL.

## Install

Two install paths: Docker (for servers, e.g. the DGX Spark) or a bare `uv`
virtual environment (for a local workstation).

### Server / DGX Spark (Docker)

Requires Docker with the NVIDIA Container Toolkit (`--gpus all`) and an
NVIDIA GPU with >= 16 GB VRAM recommended.

    ./docker/build.sh
    ./docker/run.sh python scripts/verify_install.py

From a laptop checkout, `scripts/spark.sh` rsyncs the repo to the server and
runs the command inside the container by default; a leading `--host` runs it
directly on the server instead (e.g. to build the image):

    ./scripts/spark.sh python -m pytest tests/unit -q
    ./scripts/spark.sh --host ./docker/build.sh

### Local (bare uv)

Requires an NVIDIA GPU with >= 16 GB VRAM recommended. On aarch64 this also
needs the apt packages Isaac Lab's arm64 build depends on:

    sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev cmake build-essential

Then:

    OMNI_KIT_ACCEPT_EULA=YES ./scripts/setup_env.sh
    source .venv/bin/activate

See `docs/environment.md` for the environment and its parameters.
