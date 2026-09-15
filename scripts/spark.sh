#!/usr/bin/env bash
# Syncs the laptop working tree to the DGX Spark and runs a command there
# inside the project venv, with the aarch64 Isaac Lab preloads set up.
#
# Usage: ./scripts/spark.sh <command> [args...]
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

rsync -az --delete \
  --exclude .git --exclude .venv --exclude third_party --exclude .superpowers \
  --exclude outputs --exclude multirun --exclude checkpoints \
  --exclude __pycache__ --exclude .pytest_cache \
  "$ROOT"/ "${SPARK_HOST}:${SPARK_DIR}/"

printf -v CMD '%q ' "$@"

ssh "$SPARK_HOST" bash -s <<EOF
set -euo pipefail
cd $SPARK_DIR
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PATH="\$HOME/.local/bin:\$PATH"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
  CARB_SO="\$(python3 -c "import sys, glob
for p in sys.path:
    hits = glob.glob(p + '/omni/client/libcarb.so')
    if hits:
        print(hits[0]); break")"
  if [[ -n "\$CARB_SO" ]]; then
    export LD_PRELOAD="\$LD_PRELOAD:\$CARB_SO"
  fi
fi
exec $CMD
EOF
