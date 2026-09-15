#!/usr/bin/env bash
# Runs a command inside the food-robot image, bind-mounting the repo so code
# edits on the host take effect without rebuilding, and using the named volumes
# listed in docker/volumes.sh for the Isaac Sim / Isaac Lab caches plus a
# food_robot cache.
#
# Usage: ./docker/run.sh <command> [args...]
#
# Env overrides:
#   DOCKER_DETACH=1   start the container detached (`docker run -d`) instead of
#                      attached; the command keeps running after this script
#                      returns. Combine with DOCKER_NAME to name it (see
#                      scripts/spark.sh --detach, which sets both).
#   DOCKER_NAME=<name> container name; only meaningful with DOCKER_DETACH=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FOOD_ROBOT_IMAGE:-food-robot:latest}"

RUN_FLAGS=()
if [[ "${DOCKER_DETACH:-0}" == "1" ]]; then
  RUN_FLAGS+=(-d)
  if [[ -n "${DOCKER_NAME:-}" ]]; then
    RUN_FLAGS+=(--name "$DOCKER_NAME")
  fi
  # No --rm here: a detached run's whole point is to be inspected (`docker logs`)
  # and cleaned up (`docker stop`/`docker rm`) after this script has returned.
else
  RUN_FLAGS+=(--rm)
  if [[ -t 0 ]]; then
    RUN_FLAGS+=(-it)
  fi
fi

# shellcheck source=docker/volumes.sh
source "$ROOT/docker/volumes.sh"
VOLUME_FLAGS=()
for spec in "${FOOD_ROBOT_VOLUMES[@]}"; do
  VOLUME_FLAGS+=(-v "$spec")
done

exec docker run "${RUN_FLAGS[@]}" \
  --gpus all --network host --ipc=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$ROOT":/workspace/food-robot \
  "${VOLUME_FLAGS[@]}" \
  -w /workspace/food-robot \
  "$IMAGE" "$@"
