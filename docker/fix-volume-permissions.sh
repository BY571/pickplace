#!/usr/bin/env bash
# Makes every named volume from docker/volumes.sh owned by the image's runtime
# user. Docker only seeds ownership from the image when a volume is first
# created, so volumes created earlier (or by another image) can stay
# root-owned and unwritable for the non-root container user. Idempotent;
# docker/build.sh runs it after building. Creates volumes that do not exist yet.
#
# Usage: ./docker/fix-volume-permissions.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FOOD_ROBOT_IMAGE:-food-robot:latest}"
# shellcheck source=docker/volumes.sh
source "$ROOT/docker/volumes.sh"

RUNTIME_USER="$(docker image inspect "$IMAGE" --format '{{.Config.User}}')"
RUNTIME_USER="${RUNTIME_USER:-root}"
OWNER="$(docker run --rm --entrypoint id "$IMAGE" -u "$RUNTIME_USER"):$(docker run --rm --entrypoint id "$IMAGE" -g "$RUNTIME_USER")"

MOUNT_FLAGS=()
TARGETS=()
for spec in "${FOOD_ROBOT_VOLUMES[@]}"; do
  name="${spec%%:*}"
  target="/volumes/${name}"
  MOUNT_FLAGS+=(-v "${name}:${target}")
  TARGETS+=("$target")
done

docker run --rm --user root --entrypoint chown "${MOUNT_FLAGS[@]}" "$IMAGE" -R "$OWNER" "${TARGETS[@]}"
echo "Volumes owned by ${RUNTIME_USER} (${OWNER}): ${FOOD_ROBOT_VOLUMES[*]%%:*}"
