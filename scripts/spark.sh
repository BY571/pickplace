#!/usr/bin/env bash
# Syncs the laptop working tree to the DGX Spark and runs a command there.
# By default the command runs inside the food-robot Docker image
# (docker/run.sh), which handles all aarch64 / Isaac Sim quirks. Pass a
# leading --host to run the command directly on the Spark host instead
# (e.g. to build the image itself).
#
# Usage: ./scripts/spark.sh [--host] <command> [args...]
# Env overrides: SPARK_HOST (default: spark), SPARK_DIR (default: ~/food-robot)
#
# Note: args are embedded (shell-quoted via printf %q) into the heredoc sent
# over ssh's stdin rather than passed as ssh command-line arguments, because
# ssh concatenates its remote-command argv with plain spaces and re-parses it
# on the remote side, which silently splits any argument containing spaces.
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"
SPARK_DIR="${SPARK_DIR:-~/food-robot}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST_MODE=0
if [[ "${1:-}" == "--host" ]]; then
  HOST_MODE=1
  shift
fi

rsync -az --delete \
  --exclude .git --exclude .venv --exclude third_party --exclude .superpowers \
  --exclude outputs --exclude multirun --exclude checkpoints \
  --exclude __pycache__ --exclude .pytest_cache \
  "$ROOT"/ "${SPARK_HOST}:${SPARK_DIR}/"

printf -v CMD '%q ' "$@"

RUN_PREFIX=""
[[ "$HOST_MODE" -eq 1 ]] || RUN_PREFIX="./docker/run.sh "

ssh "$SPARK_HOST" bash -s <<EOF
set -euo pipefail
cd $SPARK_DIR
exec ${RUN_PREFIX}$CMD
EOF
