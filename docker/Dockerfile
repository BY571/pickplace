# food-robot on top of Isaac Lab's official base image (built by
# docker/build.sh from third_party/IsaacLab/docker/Dockerfile.base).
ARG BASE_IMAGE=food-robot-isaaclab-base:v3.0.0-beta2.patch1
FROM ${BASE_IMAGE}

# ISAACLAB_PATH and DOCKER_USER_HOME are inherited as ENV from the base image
# (/workspace/isaaclab and /root respectively). Isaac Lab's own install
# commands (and, further down, pip installs into the same Python) need to run
# as root, matching Dockerfile.base's own convention -- it only switches to
# the non-root `isaaclab` user at the very end, for runtime.
USER root

# Make `python`/`python3` on PATH resolve to the Isaac Sim/Isaac Lab Python
# (the base image only wires this up as a bashrc alias, which does not apply
# to non-interactive `docker run image python ...` invocations). A plain
# symlink breaks python.sh: it locates its own directory via
# `dirname "${BASH_SOURCE[0]}"` to find sibling setup_python_env.sh, and a
# symlink at /usr/local/bin makes that resolve to /usr/local/bin instead of
# the real _isaac_sim dir. Use a tiny wrapper that execs it by its real path.
RUN { \
      echo '#!/bin/bash'; \
      echo 'exec "${ISAACLAB_PATH}/_isaac_sim/python.sh" "$@"'; \
    } > /usr/local/bin/python && \
    chmod +x /usr/local/bin/python && \
    cp /usr/local/bin/python /usr/local/bin/python3

# The base NVIDIA isaac-sim image bakes in ENTRYPOINT ["/isaac-sim/runheadless.sh"],
# which starts a persistent headless streaming server and ignores any command
# passed to `docker run` (args become arguments to runheadless.sh, not a
# replacement command). Isaac Lab's own docker-compose.yaml overrides this
# with `entrypoint: bash` for the same reason; we clear it entirely so
# `docker run food-robot:latest <cmd> [args...]` runs <cmd> directly.
ENTRYPOINT []

# Isaac Lab's base image already installs isaaclab_assets and isaaclab_physx:
# `isaaclab.sh --install` (no args) defaults to install_type="all", and both
# are part of CORE_ISAACLAB_SUBMODULES, which is always installed regardless
# of install_type. Nothing to duplicate here -- just add TorchRL on top.
#
# torchrl/tensordict are installed with --no-deps so they cannot touch the
# torch build Isaac Lab already installed; their few non-torch runtime deps
# (numpy, cloudpickle, packaging, orjson) are installed separately, alongside
# hydra-core/omegaconf/tqdm/pytest with their own deps resolved normally.
RUN "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-deps \
      "torchrl>=0.14.0,<0.15" "tensordict>=0.14.2,<0.15" && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install \
      "hydra-core>=1.3" "omegaconf>=2.3" tqdm "pytest>=8" \
      numpy cloudpickle packaging orjson && \
    "${ISAACLAB_PATH}/isaaclab.sh" -p -c "\
import torch, torchrl, tensordict, isaaclab_assets, isaaclab_physx; \
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} torchrl={torchrl.__version__} tensordict={tensordict.__version__}')"

# Experiment logging and plots: W&B (video logging needs moviepy) and seaborn/matplotlib for result figures.
# A separate layer so the Isaac Lab / TorchRL layers above stay cached.
RUN "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install \
      "wandb>=0.19" "moviepy>=2" "seaborn>=0.13" "matplotlib>=3.8"

# Copy just pyproject.toml + food_robot/ (not the whole repo) so a runtime
# bind-mount of the repo root at /workspace/food-robot takes effect without
# invalidating this layer on every code edit.
COPY pyproject.toml /workspace/food-robot/pyproject.toml
COPY food_robot/ /workspace/food-robot/food_robot/
WORKDIR /workspace/food-robot
RUN "${ISAACLAB_PATH}/isaaclab.sh" -p -m pip install --no-deps -e . && \
    chown -R isaaclab:isaaclab /workspace/food-robot

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_DRIVER_CAPABILITIES=all

# Pre-create every path docker/run.sh mounts a named volume on (listed in
# docker/volumes.sh) owned by the runtime user. The container runs as `isaaclab`
# but DOCKER_USER_HOME (inherited from the base image) is /root, so all caches
# live under /root and /isaac-sim. Docker seeds a freshly-created named volume
# from whatever exists at that path in the image (ownership included), so this
# keeps first-use writes from hitting a root-owned, isaaclab-unwritable
# directory. Volumes that already exist are not re-seeded by Docker; repair
# those with docker/fix-volume-permissions.sh (build.sh runs it).
COPY docker/volumes.sh /tmp/food-robot-volumes.sh
RUN . /tmp/food-robot-volumes.sh && \
    for spec in "${FOOD_ROBOT_VOLUMES[@]}"; do \
      path="${spec#*:}"; \
      mkdir -p "$path" && chown -R isaaclab:isaaclab "$path"; \
    done && \
    rm /tmp/food-robot-volumes.sh

USER isaaclab
WORKDIR /workspace/food-robot
