# food-robot

Isaac Lab environment of a robot arm placing food into bowls moving on a conveyor belt, runnable with TorchRL.

- Environment and parameters: [`docs/environment.md`](docs/environment.md)
- Algorithms: [`sota-implementations/`](sota-implementations/) (PPO first)

## Install

Two install paths: Docker (for servers, e.g. the DGX Spark) or a bare `uv`
virtual environment (for a local workstation).

### Server / DGX Spark (Docker)

Requires Docker with the NVIDIA Container Toolkit (`--gpus all`) and an
NVIDIA GPU with >= 16 GB VRAM recommended.

    ./docker/build.sh
    ./docker/run.sh python scripts/verify_install.py

The container runs as the non-root `isaaclab` user and keeps Isaac Sim / Isaac Lab caches and the
generated USD assets in named Docker volumes (listed in `docker/volumes.sh`). `docker/build.sh`
makes every volume writable for that user; if a volume ever becomes unwritable (e.g. created by an
older image or by root), repair it with:

    ./docker/fix-volume-permissions.sh

From a laptop checkout, `scripts/spark.sh` rsyncs the repo to the server and
runs the command inside the container by default; a leading `--host` runs it
directly on the server instead (e.g. to build the image):

    ./scripts/spark.sh python -m pytest tests/unit -q
    ./scripts/spark.sh --host ./docker/build.sh

### Local (bare uv)

**Untested in this repo.** Development and all verification (unit tests, simulator tests, the PPO
scale run) happened exclusively on the DGX Spark through Docker; `scripts/setup_env.sh` and the
steps below have never been exercised end-to-end here. Treat this path as a starting point, not a
verified install.

Requires an NVIDIA GPU with >= 16 GB VRAM recommended. On aarch64 this also
needs the apt packages Isaac Lab's arm64 build depends on:

    sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev cmake build-essential

Then:

    OMNI_KIT_ACCEPT_EULA=YES ./scripts/setup_env.sh
    source .venv/bin/activate

**aarch64 note:** Isaac Sim's bundled OpenMP/carb libraries must be preloaded before Python starts on
aarch64, per the Isaac Lab docs, or extension loading fails with unresolved-symbol errors:

    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:<venv site-packages>/omni/client/libcarb.so

(`<venv site-packages>` is typically `.venv/lib/python3.12/site-packages`.)

See `docs/environment.md` for the environment and its parameters.

## Quick check

Prints the TorchRL spec tree, runs `check_env_specs` and a short random rollout:

    python scripts/check_env.py env.num_envs=4 env.cameras=false env.privileged_information=true          # inside the container / venv
    ./scripts/spark.sh python scripts/check_env.py env.num_envs=4 env.cameras=false env.privileged_information=true   # from a laptop, runs on the Spark

## Train

    cd sota-implementations/ppo && python ppo.py env.num_envs=4096

From a laptop: `./scripts/spark.sh python sota-implementations/ppo/ppo.py env.num_envs=4096`. This runs
attached to your terminal (the ssh session must stay open). For a long run, start it detached instead:

    ./scripts/spark.sh --detach python sota-implementations/ppo/ppo.py env.num_envs=4096 logger.backend=wandb

`--detach` rsyncs the repo, then starts the command in a named, detached container
(`food-robot-<timestamp>`) on the Spark and returns immediately, printing the container name plus the
commands to follow its logs (`ssh spark docker logs -f <name>`) and stop it (`ssh spark docker stop
<name>`). See [`sota-implementations/ppo/README.md`](sota-implementations/ppo/README.md) for observation
routing and how to play a trained checkpoint.

## Tests

    python -m pytest tests/unit -q     # no simulator
    python -m pytest tests/isaac -q    # simulator scenarios (slow)

From a laptop, both run on the Spark via `./scripts/spark.sh python -m pytest tests/unit -q` (and likewise
for `tests/isaac`).
