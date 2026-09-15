#!/usr/bin/env bash
# Creates .venv with Isaac Sim 6.0.1, Isaac Lab v3.0.0-beta2.patch1 (PhysX), torch 2.10 and food_robot.
# Works on x86_64 (laptop, cu128) and aarch64 (DGX Spark, cu130).
set -euo pipefail

ISAACLAB_TAG="v3.0.0-beta2.patch1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES

ARCH="$(uname -m)"
if [[ "$ARCH" == "x86_64" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [[ "$ARCH" == "aarch64" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu130"
  echo "aarch64: requires 'sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev cmake build-essential'"
else
  echo "Unsupported architecture: $ARCH" >&2; exit 1
fi

[[ -d .venv ]] || uv venv --python 3.12 --seed .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --upgrade pip

uv pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX"

if [[ ! -d third_party/IsaacLab ]]; then
  git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
(cd third_party/IsaacLab && ./isaaclab.sh --install assets,physx)

uv pip install -e ".[dev]"
python scripts/verify_install.py
