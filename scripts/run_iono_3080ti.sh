#!/usr/bin/env bash
# Fresh Iono (Bellibolt) training on 3080 Ti. Run in its own terminal.
set -euo pipefail
cd "$(dirname "$0")/.."

unset TORCH_DEVICE CUDA_DEVICE COLLECTION_INFERENCE_DEVICE TRAIN_DATA_DEVICE CUDA_VISIBLE_DEVICES
# shellcheck source=gpu_pin.sh
source scripts/gpu_pin.sh
pin_training_gpu "3080"

export COLLECTION_INFERENCE_DEVICE=cuda
export TRAIN_DATA_DEVICE=auto
export BATCH_GAMES=12

export AGENT_DECK_PATH="baselines/official/iono/deck.csv"
export SELF_PLAY_OUR_ARCHETYPE=iono
export SELF_PLAY_FIELD_DECK_DIR=decks/iono-only

export PRIMARY_ROLLOUT_DATA=data/iono_bootstrap.jsonl
export MODEL_OUTPUT_PATH=outputs/checkpoints/iono_fresh.pt
export TRAIN_TENSOR_CACHE_DIR=outputs/cache/training_tensors/iono_fresh
export SELF_PLAY_OUTPUT_PATH=outputs/rollouts/iono_self_play.jsonl
export SELF_PLAY_CHECKPOINT_DIR=outputs/checkpoints/self_play/iono
export SELF_PLAY_WORKERS=6
export SELF_PLAY_BASELINE_GAMES=1000
export SELF_PLAY_BASELINE_EVAL_GAMES=200
export SELF_PLAY_TRAIN_WINDOW_GAMES=1000

echo "==> Iono / 3080 Ti — fresh start"
rm -rf "${TRAIN_TENSOR_CACHE_DIR}" "${SELF_PLAY_CHECKPOINT_DIR}"
rm -f "${SELF_PLAY_OUTPUT_PATH}" outputs/checkpoints/iono_fresh.{pt,best.pt,latest.pt}

echo "==> Bootstrap train (rebuild tensors, no resume)"
python3 scripts/train_agent.py --no-resume --rebuild-tensors

echo "==> Curriculum self-play"
python3 scripts/run_self_play.py --curriculum \
  --checkpoint "${MODEL_OUTPUT_PATH}" \
  --workers "${SELF_PLAY_WORKERS}"
