#!/usr/bin/env bash
# Unified archetype training launcher. Replaces the six run_<archetype>_*.sh scripts.
#
#   bash scripts/run_archetype.sh lucario              # bootstrap + curriculum self-play
#   bash scripts/run_archetype.sh dragapult            # uses Blackwell + 512d profile
#   bash scripts/run_archetype.sh starmie --self-play-only   # skip bootstrap, resume self-play
#   bash scripts/run_archetype.sh crustle --bootstrap-only   # bootstrap, then stop
#
# Profiles (deck, GPU, model size, deck pools) live in scripts/archetype_profiles.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:-}"
MODE="full"
for arg in "${@:2}"; do
  case "$arg" in
    --self-play-only) MODE="self-play-only" ;;
    --bootstrap-only) MODE="bootstrap-only" ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  echo "usage: $0 <lucario|dragapult|abomasnow|iono|starmie|crustle> [--self-play-only|--bootstrap-only]" >&2
  exit 1
fi

# shellcheck source=archetype_profiles.sh
source scripts/archetype_profiles.sh
# shellcheck source=ensure_bootstrap_data.sh
source scripts/ensure_bootstrap_data.sh
# shellcheck source=gpu_pin.sh
source scripts/gpu_pin.sh

load_archetype_profile "$ARCH"

unset TORCH_DEVICE CUDA_DEVICE COLLECTION_INFERENCE_DEVICE TRAIN_DATA_DEVICE CUDA_VISIBLE_DEVICES
pin_training_gpu "$ARCH_GPU"

export COLLECTION_INFERENCE_DEVICE=cuda
export TRAIN_DATA_DEVICE="$ARCH_TRAIN_DATA_DEVICE"
export BATCH_GAMES="$ARCH_BATCH_GAMES"
if [[ -n "$ARCH_MODEL_DIMS" ]]; then
  read -r d heads layers ff <<<"$ARCH_MODEL_DIMS"
  export MODEL_D_MODEL="$d" MODEL_HEADS="$heads" MODEL_LAYERS="$layers" MODEL_FF="$ff"
fi

export AGENT_DECK_PATH="$ARCH_DECK"
export SELF_PLAY_OUR_ARCHETYPE="$ARCH_SLUG"
export SELF_PLAY_FIELD_DECK_DIR="$ARCH_FIELD_DIR"
if [[ "$ARCH_BASELINE_DECKS_ONLY" == "0" ]]; then
  export SELF_PLAY_BASELINE_ARCHETYPE_DECKS_ONLY=0
  export SELF_PLAY_AGENT_DECK_DIR="$ARCH_AGENT_DECK_DIR"
fi

# Single explicit training corpus — no PRIMARY/MERGED precedence juggling. The ladder
# gate auto-bypasses for CABT self-play corpora (poke_agent/dataset.py).
export TRAINING_DATA_PATH="$ARCH_BOOTSTRAP_DATA"
[[ "$ARCH_MATCHUP_DIVERSITY" == "0" ]] && export REQUIRE_TRAINING_MATCHUP_DIVERSITY=0

export MODEL_OUTPUT_PATH="$ARCH_MODEL_OUTPUT"
export TRAIN_TENSOR_CACHE_DIR="$ARCH_TENSOR_CACHE_DIR"
export SELF_PLAY_OUTPUT_PATH="$ARCH_SELF_PLAY_OUTPUT"
export SELF_PLAY_CHECKPOINT_DIR="$ARCH_SELF_PLAY_CKPT_DIR"
export SELF_PLAY_WORKERS="$ARCH_WORKERS"
export SELF_PLAY_BASELINE_GAMES=1000
export SELF_PLAY_BASELINE_EVAL_GAMES=200
export SELF_PLAY_TRAIN_WINDOW_GAMES=1000
export SELF_PLAY_ITERATIONS="${SELF_PLAY_ITERATIONS:-1000}"

echo "==> ${ARCH_SLUG} / ${ARCH_GPU} (mode=${MODE})"

if [[ "$MODE" != "self-play-only" ]]; then
  ensure_bootstrap_data "$TRAINING_DATA_PATH" "$ARCH_FIELD_DIR" "$ARCH_BOOTSTRAP_EPISODES"

  echo "==> fresh start: clearing tensor cache + self-play checkpoints"
  rm -rf "${TRAIN_TENSOR_CACHE_DIR}" "${SELF_PLAY_CHECKPOINT_DIR}"
  rm -f "${SELF_PLAY_OUTPUT_PATH}" "${ARCH_MODEL_OUTPUT%.pt}".{pt,best.pt,latest.pt}

  echo "==> bootstrap train (rebuild tensors, no resume)"
  python3 scripts/train_agent.py --no-resume --rebuild-tensors
fi

if [[ "$MODE" == "bootstrap-only" ]]; then
  echo "==> bootstrap complete: ${MODEL_OUTPUT_PATH}"
  exit 0
fi

echo "==> curriculum self-play"
python3 scripts/run_self_play.py --curriculum \
  --checkpoint "${MODEL_OUTPUT_PATH}" \
  --workers "${SELF_PLAY_WORKERS}"
