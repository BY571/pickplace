# Named Docker volumes used by docker/run.sh, as "volume-name:container-path".
# Single source of truth for run.sh (mounts), docker/Dockerfile (pre-creates the
# paths owned by the runtime user) and docker/fix-volume-permissions.sh (repairs
# ownership of volumes that already exist). Paths follow Isaac Lab's own
# third_party/IsaacLab/docker/docker-compose.yaml, plus the food_robot USD cache.
# shellcheck disable=SC2034
FOOD_ROBOT_VOLUMES=(
  "food-robot-cache-kit:/isaac-sim/kit/cache"
  "food-robot-cache-ov:/root/.cache/ov"
  "food-robot-cache-pip:/root/.cache/pip"
  "food-robot-cache-gl:/root/.cache/nvidia/GLCache"
  "food-robot-cache-compute:/root/.nv/ComputeCache"
  "food-robot-logs:/root/.nvidia-omniverse/logs"
  "food-robot-data:/root/.local/share/ov/data"
  "food-robot-docs:/root/Documents"
  "food-robot-cache:/root/.cache/food_robot"
)
