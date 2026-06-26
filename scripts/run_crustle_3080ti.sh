#!/usr/bin/env bash
# Fresh Crustle training on 3080 Ti. Run in its own terminal.
set -euo pipefail
cd "$(dirname "$0")/.."

unset TORCH_DEVICE CUDA_DEVICE COLLECTION_INFERENCE_DEVICE TRAIN_DATA_DEVICE CUDA_VISIBLE_DEVICES
# shellcheck source=gpu_pin.sh
source scripts/gpu_pin.sh
pin_training_gpu "3080"

export COLLECTION_INFERENCE_DEVICE=cuda
export TRAIN_DATA_DEVICE=auto
export BATCH_GAMES=12

export AGENT_DECK_PATH="decks/competitive/high_performing/2026-04_regional-prague-2026_7th_crustle.csv"
export SELF_PLAY_OUR_ARCHETYPE=crustle
export SELF_PLAY_BASELINE_ARCHETYPE_DECKS_ONLY=0
export SELF_PLAY_AGENT_DECK_DIR=decks/crustle-only
export SELF_PLAY_FIELD_DECK_DIR=decks/crustle-only

export PRIMARY_ROLLOUT_DATA=data/crustle_bootstrap.jsonl
export MODEL_OUTPUT_PATH=outputs/checkpoints/crustle_fresh.pt
export TRAIN_TENSOR_CACHE_DIR=outputs/cache/training_tensors/crustle_fresh
export SELF_PLAY_OUTPUT_PATH=outputs/rollouts/crustle_self_play.jsonl
export SELF_PLAY_CHECKPOINT_DIR=outputs/checkpoints/self_play/crustle
export SELF_PLAY_WORKERS=6
export SELF_PLAY_BASELINE_GAMES=1000
export SELF_PLAY_BASELINE_EVAL_GAMES=200
export SELF_PLAY_TRAIN_WINDOW_GAMES=1000

echo "==> Crustle / 3080 Ti — fresh start"
rm -rf "${TRAIN_TENSOR_CACHE_DIR}" "${SELF_PLAY_CHECKPOINT_DIR}"
rm -f "${SELF_PLAY_OUTPUT_PATH}" outputs/checkpoints/crustle_fresh.{pt,best.pt,latest.pt}

echo "==> Bootstrap train (rebuild tensors, no resume)"
python3 scripts/train_agent.py --no-resume --rebuild-tensors

echo "==> Curriculum self-play"
python3 scripts/run_self_play.py --curriculum \
  --checkpoint "${MODEL_OUTPUT_PATH}" \
  --workers "${SELF_PLAY_WORKERS}"
