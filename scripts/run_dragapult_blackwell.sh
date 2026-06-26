#!/usr/bin/env bash
# Fresh Dragapult training on Blackwell. Run in its own terminal.
set -euo pipefail
cd "$(dirname "$0")/.."

unset TORCH_DEVICE CUDA_DEVICE COLLECTION_INFERENCE_DEVICE TRAIN_DATA_DEVICE CUDA_VISIBLE_DEVICES
# shellcheck source=gpu_pin.sh
source scripts/gpu_pin.sh
pin_training_gpu "blackwell"

export COLLECTION_INFERENCE_DEVICE=cuda
export TRAIN_DATA_DEVICE=cuda
export MODEL_D_MODEL=512 MODEL_HEADS=8 MODEL_LAYERS=8 MODEL_FF=2048
export BATCH_GAMES=32

export AGENT_DECK_PATH="decks/competitive/high_performing/2026-05_regional-indianapolis-2026_3rd_dragapult-dusknoir.csv"
export SELF_PLAY_OUR_ARCHETYPE=dragapult-ex
export SELF_PLAY_FIELD_DECK_DIR=decks/dragapult-only

export PRIMARY_ROLLOUT_DATA=data/dragapult_bootstrap.jsonl
export MODEL_OUTPUT_PATH=outputs/checkpoints/dragapult_blackwell.pt
export TRAIN_TENSOR_CACHE_DIR=outputs/cache/training_tensors/dragapult_blackwell
export SELF_PLAY_OUTPUT_PATH=outputs/rollouts/dragapult_self_play.jsonl
export SELF_PLAY_CHECKPOINT_DIR=outputs/checkpoints/self_play/dragapult
export SELF_PLAY_WORKERS=12
export SELF_PLAY_BASELINE_GAMES=1000
export SELF_PLAY_BASELINE_EVAL_GAMES=200
export SELF_PLAY_TRAIN_WINDOW_GAMES=1000

echo "==> Dragapult / Blackwell — fresh start (features changed)"
rm -rf "${TRAIN_TENSOR_CACHE_DIR}" "${SELF_PLAY_CHECKPOINT_DIR}"
rm -f "${SELF_PLAY_OUTPUT_PATH}" outputs/checkpoints/dragapult_blackwell.{pt,best.pt,latest.pt}

echo "==> Bootstrap train (rebuild tensors, no resume)"
python3 scripts/train_agent.py --no-resume --rebuild-tensors

echo "==> Curriculum self-play"
python3 scripts/run_self_play.py --curriculum \
  --checkpoint "${MODEL_OUTPUT_PATH}" \
  --workers "${SELF_PLAY_WORKERS}"
