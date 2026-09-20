#!/usr/bin/env bash
# Builds the two food-robot Docker images used on servers (e.g. the DGX Spark):
#   1. food-robot-isaaclab-base:<tag> — Isaac Lab's own official base image,
#      built unmodified from its Dockerfile.base (reuses Isaac Lab's arm64
#      handling: GL/X11 dev headers, nlopt source build, etc. — do not
#      re-implement any of that here).
#   2. food-robot:latest — our image on top, adding TorchRL + pickplace.
set -euo pipefail

ISAACLAB_TAG="v3.0.0-beta2.patch1"
ISAACSIM_VERSION="6.0.1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d third_party/IsaacLab ]]; then
  git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi

BASE_TAG="food-robot-isaaclab-base:${ISAACLAB_TAG}"

# Build args match third_party/IsaacLab/docker/.env.base (the defaults Isaac
# Lab's own docker-compose.yaml passes to Dockerfile.base).
docker build \
  -f third_party/IsaacLab/docker/Dockerfile.base \
  --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim \
  --build-arg ISAACSIM_VERSION_ARG="$ISAACSIM_VERSION" \
  --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
  --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
  --build-arg DOCKER_USER_HOME_ARG=/root \
  -t "$BASE_TAG" \
  third_party/IsaacLab

docker build \
  -f docker/Dockerfile \
  --build-arg BASE_IMAGE="$BASE_TAG" \
  -t food-robot:latest \
  .

echo "Built ${BASE_TAG} and food-robot:latest"

# Existing named volumes are not re-seeded by a new image; make sure all of
# them are writable by the container's runtime user.
"$ROOT/docker/fix-volume-permissions.sh"
