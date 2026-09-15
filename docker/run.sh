#!/usr/bin/env bash
# Runs a command inside the food-robot image, bind-mounting the repo so code
# edits on the host take effect without rebuilding, and using named volumes
# for the Isaac Sim / Isaac Lab caches (paths copied from Isaac Lab's own
# third_party/IsaacLab/docker/docker-compose.yaml) plus a food_robot cache.
#
# Usage: ./docker/run.sh <command> [args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FOOD_ROBOT_IMAGE:-food-robot:latest}"

TTY_FLAGS=()
if [[ -t 0 ]]; then
  TTY_FLAGS=(-it)
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
  --gpus all --network host --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$ROOT":/workspace/food-robot \
  -v food-robot-cache-kit:/isaac-sim/kit/cache \
  -v food-robot-cache-ov:/root/.cache/ov \
  -v food-robot-cache-pip:/root/.cache/pip \
  -v food-robot-cache-gl:/root/.cache/nvidia/GLCache \
  -v food-robot-cache-compute:/root/.nv/ComputeCache \
  -v food-robot-logs:/root/.nvidia-omniverse/logs \
  -v food-robot-data:/root/.local/share/ov/data \
  -v food-robot-docs:/root/Documents \
  -v food-robot-cache:/root/.cache/food_robot \
  -w /workspace/food-robot \
  "$IMAGE" "$@"
