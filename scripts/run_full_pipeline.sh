#!/usr/bin/env bash
#
# Phase 1 end-to-end pipeline (terminal, minimal footprint).
#
#   [0/5] Prepare data — scrape top-of-ladder replays, convert, generate multideck,
#         merge -> data/training_rollouts_merged.jsonl, validate
#   [1/5] Bootstrap train  -> outputs/checkpoints/temporal_current.pt
#         (periodic .latest.pt / .best.pt every TRAIN_CHECKPOINT_EVERY epochs;
#          resumable after an OOM/crash. Expect many hours.)
#   [2/5] Submit the bootstrapped model to Kaggle (when training early-stops/finishes)
#   [3/5] Baseline curriculum vs official heuristic agents
#         -> submit champion to Kaggle when the >=60% gate is beaten
#   [4/5] Transformer self-play -> submit champion when it stops
#
# Output goes straight to the terminal so tqdm renders as a single, live,
# self-updating progress bar (no log files, no screen bloat). Run directly:
#
#   conda activate poke-bot-agent
#   bash scripts/run_full_pipeline.sh
#
# Resume after bootstrap OOM (skip data prep + reuse .latest.pt):
#   SKIP_DATA=1 bash scripts/run_full_pipeline.sh
#
# Fresh bootstrap (ignore .latest.pt):
#   FRESH=1 bash scripts/run_full_pipeline.sh
#
# Env knobs:
#   PYTHON             python interpreter (default: python)
#   CHECKPOINT         bootstrap output checkpoint (default: outputs/checkpoints/temporal_current.pt)
#   SKIP_DATA=1        skip step 0 (data already prepared)
#   SKIP_BOOTSTRAP=1   skip step 1 (re-use existing checkpoint)
#   NO_SCRAPE=1        step 0: merge local replays only, no Kaggle API scrape
#   NO_GENERATE=1      step 0: skip multideck CABT generation
#   DATASET_GAMES      multideck games to generate (default from config.py)
#   MESSAGE            base Kaggle submission message

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
MESSAGE="${MESSAGE:-Phase1 pipeline}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATA_PREP_ARGS=()
if [[ "${NO_SCRAPE:-0}" != "0" ]]; then
  DATA_PREP_ARGS+=(--no-scrape)
fi
if [[ "${NO_GENERATE:-0}" != "0" ]]; then
  DATA_PREP_ARGS+=(--no-generate)
fi

if [[ "${SKIP_DATA:-0}" == "0" ]]; then
  echo "==> [0/5] Prepare training data (ladder + multideck -> merged JSONL)"
  if ! $PYTHON scripts/prepare_training_data.py "${DATA_PREP_ARGS[@]}"; then
    echo "ERROR: data preparation failed; fix scrape/Kaggle API or local replays." >&2
    exit 1
  fi
else
  echo "==> [0/5] Skipping data prep (SKIP_DATA=1)"
fi

RESUME_FLAG=""
if [[ "${FRESH:-0}" != "0" ]]; then
  RESUME_FLAG="--no-resume"
fi

if [[ "${SKIP_BOOTSTRAP:-0}" == "0" ]]; then
  echo "==> [1/5] Bootstrap train (periodic checkpoints on; this can take many hours)"
  if ! $PYTHON scripts/train_agent.py $RESUME_FLAG; then
    echo "ERROR: bootstrap training failed; aborting pipeline." >&2
    exit 1
  fi
else
  echo "==> [1/5] Skipping bootstrap (SKIP_BOOTSTRAP=1); using $CHECKPOINT"
fi

echo "==> [2/5] Submit bootstrapped model to Kaggle"
if ! $PYTHON scripts/submit_kaggle.py --checkpoint "$CHECKPOINT" \
      --message "$MESSAGE bootstrap $STAMP"; then
  echo "WARN: bootstrap Kaggle submission failed; continuing to curriculum." >&2
fi

echo "==> [3/5] Baseline curriculum + [4/5] transformer self-play"
echo "         (submits champion at the baseline gate and again when self-play stops)"
if ! $PYTHON scripts/run_self_play.py --curriculum \
      --checkpoint "$CHECKPOINT" \
      --submit-after-baseline --submit-on-stop; then
  echo "ERROR: curriculum/self-play stage failed." >&2
  exit 1
fi

echo "==> pipeline complete"
