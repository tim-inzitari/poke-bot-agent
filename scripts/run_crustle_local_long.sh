#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/crustle_local_long_${RUN_ID}.log}"
LATEST_LOG="${LATEST_LOG:-${LOG_DIR}/latest_crustle_local_long.log}"

mkdir -p \
  "$LOG_DIR" \
  outputs/checkpoints/crustle_local_long \
  outputs/rollouts/crustle_local_long \
  outputs/reports/crustle_local_long

touch "$LOG_PATH"
abs_log="$(cd "$(dirname "$LOG_PATH")" && pwd)/$(basename "$LOG_PATH")"
ln -sf "$abs_log" "$LATEST_LOG"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "Crustle local long run"
echo "started: $(date)"
echo "log:     $abs_log"
echo

CAFFEINATE_PID=""
if [[ "${USE_CAFFEINATE:-1}" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w "$$" &
  CAFFEINATE_PID="$!"
  echo "caffeinate enabled: pid=$CAFFEINATE_PID"
  echo
fi

cleanup() {
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" >/dev/null 2>&1 || true
    wait "$CAFFEINATE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export PYTHONPATH="${PYTHONPATH:-.}"

# Keep this intentionally small for local Mac training. The checkpoint shape must
# match these values on every retrain or PyTorch will reject the reload.
export MODEL_D_MODEL="${MODEL_D_MODEL:-32}"
export MODEL_HEADS="${MODEL_HEADS:-4}"
export MODEL_LAYERS="${MODEL_LAYERS:-2}"
export MODEL_USE_KAN="${MODEL_USE_KAN:-0}"
export WINDOW_SIZE="${WINDOW_SIZE:-128}"
export TENSOR_BUILD_WORKERS="${TENSOR_BUILD_WORKERS:-8}"

export ITERATIONS="${ITERATIONS:-100000}"
export START_ITERATION="${START_ITERATION:-1}"
export GAMES="${GAMES:-80}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export BATCH_GAMES="${BATCH_GAMES:-16}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
export TARGET_WIN_RATE="${TARGET_WIN_RATE:-0.99}"
export BASELINES="${BASELINES:-public}"
export POLICY_PORT="${POLICY_PORT:-19143}"
export POLICY_TIMEOUT="${POLICY_TIMEOUT:-180}"

export DECK="${DECK:-baselines/kaggle_public/dashimaki_day1_crustle/deck.csv}"
export CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/crustle_local_long/seed.pt}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/crustle_local_long}"
export ROLLOUTS="${ROLLOUTS:-outputs/rollouts/crustle_local_long/rollouts.jsonl}"
export REPORT_DIR="${REPORT_DIR:-outputs/reports/crustle_local_long}"

./scripts/run_remote_public_baseline_rl.sh
