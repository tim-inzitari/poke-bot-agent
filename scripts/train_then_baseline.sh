#!/usr/bin/env bash
#
# Bootstrap train, then baseline-only self-play (official Kaggle heuristic agents).
#
# Full chain (fresh train):
#   bash scripts/train_then_baseline.sh --no-resume
#
# Train already running — wait for it, then baseline:
#   WAIT_FOR_TRAIN=1 bash scripts/train_then_baseline.sh
#
# Env:
#   CHECKPOINT   — checkpoint for baseline phase (default: temporal_current.pt)
#   PYTHON       — python executable (default: python)
#   BASELINE_ARGS — extra args for run_self_play.py (e.g. --workers 8)
#   TRAIN_TENSOR_CACHE — auto (default): skip JSONL build when cache exists

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
export TRAIN_TENSOR_CACHE="${TRAIN_TENSOR_CACHE:-auto}"
mkdir -p "$LOG_DIR"

_train_running() {
  pgrep -f "[p]ython.*scripts/train_agent\\.py" >/dev/null 2>&1
}

if [[ "${WAIT_FOR_TRAIN:-0}" != "0" ]]; then
  echo "==> Waiting for running train_agent.py to finish..."
  while _train_running; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') train still running..."
    sleep 120
  done
  echo "==> Train process ended."
  if [[ ! -f "$CHECKPOINT" && -f "${CHECKPOINT%.pt}.latest.pt" ]]; then
  CHECKPOINT="${CHECKPOINT%.pt}.latest.pt"
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT" >&2
    exit 1
  fi
else
  echo "==> [1/2] Bootstrap train"
  if ! $PYTHON scripts/train_agent.py "$@"; then
    echo "ERROR: bootstrap training failed." >&2
    exit 1
  fi
fi

echo "==> [2/2] Baseline-only phase (vs official heuristic agents)"
# shellcheck disable=SC2086
if ! $PYTHON scripts/run_self_play.py --baseline-only --checkpoint "$CHECKPOINT" ${BASELINE_ARGS:-}; then
  echo "ERROR: baseline-only phase failed." >&2
  exit 1
fi

echo "==> train + baseline-only complete"
