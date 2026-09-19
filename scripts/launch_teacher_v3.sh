#!/usr/bin/env bash
# Overnight launch of a state teacher config (pipeline/0_state_teacher/config_v3.yaml by default) on the DGX Spark.
#
# One detached container: first the tests that cover v3 (unit, reward vector, return-home semantics, teacher smoke
# incl. config_v3); only if all pass, the training run. The log shows TESTS_PASSED before training starts, or
# TESTS_FAILED (and the container exits) otherwise.
#
# Usage: ./scripts/launch_teacher_v3.sh [hydra overrides for train.py, e.g. env.num_envs=16384]
#        (SPARK_HOST overrides the ssh host, as in spark.sh; CONFIG picks the Hydra config name, e.g.
#        CONFIG=config_v3b ./scripts/launch_teacher_v3.sh)
set -euo pipefail

SPARK_HOST="${SPARK_HOST:-spark}"
CONFIG="${CONFIG:-config_v3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="teacher_${CONFIG#config_}_$(date -u +%Y%m%dT%H%M%SZ)"
EXTRA=""; (( $# )) && EXTRA="$(printf ' %q' "$@")"  # extra Hydra overrides, passed on to train.py
TESTS="tests/unit tests/isaac/test_reward_vector.py tests/isaac/test_return_home.py tests/isaac/test_teacher_smoke.py"

# Stale carb semaphores from killed Isaac Sim processes block the next start; clear them only when no Isaac process
# is running (never under a live run).
ssh "$SPARK_HOST" 'pgrep -f "[k]it/python" >/dev/null || rm -f /dev/shm/carb-* /dev/shm/sem.carb*'

OUT="$("$ROOT/scripts/spark.sh" --detach bash -c "
  { python -m pytest -q $TESTS || { echo TESTS_FAILED; exit 1; }; } \
  && echo TESTS_PASSED \
  && python pipeline/0_state_teacher/train.py --config-name $CONFIG run.name=$RUN_NAME$EXTRA
")"
echo "$OUT"
NAME="$(sed -n 's/^Started detached container: //p' <<<"$OUT")"

echo
echo "container: $NAME   run: $RUN_NAME"
echo "tests:     ssh $SPARK_HOST 'docker logs $NAME 2>&1 | grep -E \"passed|failed|error|TESTS_(PASSED|FAILED)\" | tail -5'"
echo "metrics:   ssh $SPARK_HOST 'docker logs $NAME 2>&1 | grep -o \"METRICS .*\" | tail -1 | cut -c1-600'"
