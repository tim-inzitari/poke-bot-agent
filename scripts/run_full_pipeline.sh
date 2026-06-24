#!/usr/bin/env bash
#
# Phase 1 end-to-end pipeline (terminal, minimal footprint).
#
#   [0/5] Prepare data — episodes-index top games + multideck CABT -> merged JSONL
#   [1/5] Bootstrap train  -> outputs/checkpoints/temporal_current.pt
#   [2/5] Submit bootstrapped model to Kaggle
#   [3/5] Baseline curriculum (+ submit at gate)
#   [4/5] Transformer self-play (+ submit on stop)
#
#   conda activate poke-bot-agent
#   bash scripts/run_full_pipeline.sh
#
# Resume bootstrap after OOM:  SKIP_DATA=1 bash scripts/run_full_pipeline.sh
# Fresh bootstrap:             FRESH=1 bash scripts/run_full_pipeline.sh
# Rebuild tensor cache:        REBUILD_TENSORS=1 bash scripts/run_full_pipeline.sh

set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
MESSAGE="${MESSAGE:-Phase1 pipeline}"
STAMP="$(date +%Y%m%d_%H%M%S)"
export TRAIN_TENSOR_CACHE="${TRAIN_TENSOR_CACHE:-auto}"

DATA_PREP_ARGS=()
if [[ "${NO_DOWNLOAD:-0}" != "0" ]]; then
  DATA_PREP_ARGS+=(--no-download)
fi
if [[ "${NO_GENERATE:-0}" != "0" ]]; then
  DATA_PREP_ARGS+=(--no-generate)
fi

if [[ "${SKIP_DATA:-0}" == "0" ]]; then
  echo "==> [0/5] Prepare training data (episodes-index top games + multideck -> merged JSONL)"
  if ! $PYTHON scripts/prepare_training_data.py "${DATA_PREP_ARGS[@]}"; then
    echo "ERROR: data preparation failed." >&2
    echo "  First-time setup: bash scripts/download-episodes-index.sh" >&2
    echo "  Needs Kaggle API token at ~/.kaggle/kaggle.json" >&2
    exit 1
  fi
else
  echo "==> [0/5] Skipping data prep (SKIP_DATA=1)"
fi

TENSOR_CACHE_ARGS=()
if [[ "${REBUILD_TENSORS:-0}" != "0" ]]; then
  TENSOR_CACHE_ARGS+=(--rebuild)
elif [[ "${SKIP_TENSOR_CACHE:-0}" == "0" ]]; then
  if ! $PYTHON scripts/build_tensor_cache.py "${TENSOR_CACHE_ARGS[@]}"; then
    echo "WARN: tensor cache prep failed; train_agent will build from JSONL if needed." >&2
  fi
else
  echo "==> Skipping tensor cache prep (SKIP_TENSOR_CACHE=1)"
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
if ! $PYTHON scripts/run_self_play.py --curriculum \
      --checkpoint "$CHECKPOINT" \
      --submit-after-baseline --submit-on-stop; then
  echo "ERROR: curriculum/self-play stage failed." >&2
  exit 1
fi

echo "==> pipeline complete"
