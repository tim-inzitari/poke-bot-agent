#!/usr/bin/env bash
#
# Phase 1 end-to-end pipeline (terminal, minimal footprint).
#
#   [1/4] Bootstrap train  -> outputs/checkpoints/temporal_current.pt
#         (periodic .latest.pt / .best.pt every TRAIN_CHECKPOINT_EVERY epochs;
#          resumable after an OOM/crash. Expect many hours.)
#   [2/4] Submit the bootstrapped model to Kaggle (when training early-stops/finishes)
#   [3/4] Baseline curriculum vs official heuristic agents
#         -> submit champion to Kaggle when the >=60% gate is beaten
#   [4/4] Transformer self-play -> submit champion when it stops
#
# Usage:
#   conda activate poke-bot-agent
#   bash scripts/run_full_pipeline.sh
#
# Resume behaviour (bootstrap): defaults to TRAIN_RESUME=auto, so re-running this
# script after a crash continues from temporal_current.latest.pt. Force a clean
# restart with:  FRESH=1 bash scripts/run_full_pipeline.sh
#
# Run unattended in the background and keep logs:
#   nohup bash scripts/run_full_pipeline.sh > outputs/logs/pipeline.out 2>&1 &
#
# Env knobs:
#   PYTHON       python interpreter (default: python)
#   CHECKPOINT   bootstrap output checkpoint (default: outputs/checkpoints/temporal_current.pt)
#   FRESH=1      ignore any .latest.pt and bootstrap from scratch
#   SKIP_BOOTSTRAP=1   skip step 1 (re-use existing checkpoint), go straight to submit+curriculum
#   MESSAGE      base Kaggle submission message

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
MESSAGE="${MESSAGE:-Phase1 pipeline}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

RESUME_FLAG=""
if [[ "${FRESH:-0}" != "0" ]]; then
  RESUME_FLAG="--no-resume"
fi

if [[ "${SKIP_BOOTSTRAP:-0}" == "0" ]]; then
  echo "==> [1/4] Bootstrap train (periodic checkpoints on; this can take many hours)"
  if ! $PYTHON scripts/train_agent.py $RESUME_FLAG 2>&1 | tee "$LOG_DIR/bootstrap_$STAMP.log"; then
    echo "ERROR: bootstrap training failed; aborting pipeline." >&2
    exit 1
  fi
else
  echo "==> [1/4] Skipping bootstrap (SKIP_BOOTSTRAP=1); using $CHECKPOINT"
fi

echo "==> [2/4] Submit bootstrapped model to Kaggle"
if ! $PYTHON scripts/submit_kaggle.py --checkpoint "$CHECKPOINT" \
      --message "$MESSAGE bootstrap $STAMP" 2>&1 | tee "$LOG_DIR/submit_bootstrap_$STAMP.log"; then
  echo "WARN: bootstrap Kaggle submission failed; continuing to curriculum." >&2
fi

echo "==> [3/4] Baseline curriculum + [4/4] transformer self-play"
echo "         (submits champion at the baseline gate and again when self-play stops)"
if ! $PYTHON scripts/run_self_play.py --curriculum \
      --checkpoint "$CHECKPOINT" \
      --submit-after-baseline --submit-on-stop \
      2>&1 | tee "$LOG_DIR/curriculum_$STAMP.log"; then
  echo "ERROR: curriculum/self-play stage failed." >&2
  exit 1
fi

echo "==> pipeline complete (logs in $LOG_DIR, *_$STAMP.log)"
