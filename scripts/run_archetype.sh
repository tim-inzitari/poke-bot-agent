#!/usr/bin/env bash
# Unified archetype training launcher. Replaces the six run_<archetype>_*.sh scripts.
#
#   bash scripts/run_archetype.sh lucario              # bootstrap + curriculum self-play
#   bash scripts/run_archetype.sh dragapult            # uses Blackwell + 512d profile
#   bash scripts/run_archetype.sh starmie --self-play-only   # skip bootstrap, resume self-play
#   bash scripts/run_archetype.sh dragapult --skip-merge   # train on existing dragapult_training.jsonl
#   bash scripts/run_archetype.sh dragapult --ladder-only  # top ladder + 1000 CABT, dragapult games only
#
# Profiles (deck, GPU, model size, deck pools) live in scripts/archetype_profiles.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:-}"
MODE="full"
SKIP_MERGE=0
LADDER_ONLY=0
for arg in "${@:2}"; do
  case "$arg" in
    --self-play-only) MODE="self-play-only" ;;
    --bootstrap-only) MODE="bootstrap-only" ;;
    --skip-merge) SKIP_MERGE=1 ;;
    --ladder-only) LADDER_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  echo "usage: $0 <lucario|dragapult|...> [--self-play-only|--bootstrap-only|--skip-merge|--ladder-only]" >&2
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
if [[ "$LADDER_ONLY" == "1" ]]; then
  export TRAINING_DATA_PATH="$ARCH_BOOTSTRAP_DATA"
  export REQUIRE_TOP_OF_LADDER_DATA=1
else
  export TRAINING_DATA_PATH="$ARCH_BOOTSTRAP_DATA"
fi
[[ "$ARCH_MATCHUP_DIVERSITY" == "0" ]] && export REQUIRE_TRAINING_MATCHUP_DIVERSITY=0

export MODEL_OUTPUT_PATH="$ARCH_MODEL_OUTPUT"
export TRAIN_TENSOR_CACHE_DIR="$ARCH_TENSOR_CACHE_DIR"
export SELF_PLAY_OUTPUT_PATH="$ARCH_SELF_PLAY_OUTPUT"
export SELF_PLAY_CHECKPOINT_DIR="$ARCH_SELF_PLAY_CKPT_DIR"
export SELF_PLAY_WORKERS="$ARCH_WORKERS"
export TENSOR_BUILD_WORKERS="${TENSOR_BUILD_WORKERS:-$ARCH_WORKERS}"
export SELF_PLAY_BASELINE_GAMES=1000
export SELF_PLAY_BASELINE_EVAL_GAMES=200
export SELF_PLAY_TRAIN_WINDOW_GAMES=1000
export SELF_PLAY_ITERATIONS="${SELF_PLAY_ITERATIONS:-1000}"

echo "==> ${ARCH_SLUG} / ${ARCH_GPU} (mode=${MODE})"

if [[ "$MODE" != "self-play-only" ]]; then
  if [[ "$LADDER_ONLY" == "1" ]]; then
    cabt_episodes="${ARCH_CABT_EPISODES:-1000}"
    echo "==> ladder-only: building ${ARCH_LADDER_EPISODES:-5000} top ladder episodes -> ${ARCH_LADDER_DATA}"
    python3 -c "
import os
from pathlib import Path
from poke_agent.config import build_config
from poke_agent.data_pipeline import convert_episodes_index_to_rollouts, ensure_episodes_index_data
from poke_agent.paths import resolve_root
root = resolve_root()
cfg = build_config(root)
idx = Path(cfg['episodes_index_path'])
workers = int(os.environ.get('TENSOR_BUILD_WORKERS', cfg.get('tensor_build_workers', 2)))
slugs = ensure_episodes_index_data(root, index_path=idx)
convert_episodes_index_to_rollouts(
    root, out_path=Path('${ARCH_LADDER_DATA}'), episodes_index=idx,
    top_percent=100.0, max_episodes=int('${ARCH_LADDER_EPISODES:-5000}'),
    daily_slugs=slugs, workers=workers,
)
"
    echo "==> ladder-only: generating ${cabt_episodes} CABT games -> ${ARCH_CABT_DATA}"
    rm -f "${ARCH_CABT_DATA}"
    ensure_bootstrap_data "${ARCH_CABT_DATA}" "${ARCH_FIELD_DIR}" "${cabt_episodes}"
    echo "==> ladder-only: merge ladder + CABT, keep ${ARCH_SLUG} games only -> ${ARCH_BOOTSTRAP_DATA}"
    python3 scripts/merge_rollouts.py \
      "${ARCH_LADDER_DATA}" "${ARCH_CABT_DATA}" \
      --out "${ARCH_BOOTSTRAP_DATA}" \
      --workers "${SELF_PLAY_WORKERS}"
  elif [[ "$SKIP_MERGE" == "1" ]]; then
    if [[ ! -s "$TRAINING_DATA_PATH" ]]; then
      echo "ERROR: --skip-merge but training corpus missing: ${TRAINING_DATA_PATH}" >&2
      exit 1
    fi
    echo "==> skip merge: using existing ${TRAINING_DATA_PATH}"
  elif [[ "${ARCH_MERGE_LADDER:-0}" == "1" ]]; then
    ensure_bootstrap_data "$ARCH_CABT_DATA" "$ARCH_FIELD_DIR" "$ARCH_BOOTSTRAP_EPISODES"
    if [[ ! -s "${ARCH_LADDER_DATA}" ]]; then
      echo "==> ladder rollouts missing; building ${ARCH_LADDER_EPISODES:-5000} episodes -> ${ARCH_LADDER_DATA}"
      python3 -c "
import os
from pathlib import Path
from poke_agent.config import build_config
from poke_agent.data_pipeline import convert_episodes_index_to_rollouts, ensure_episodes_index_data
from poke_agent.paths import resolve_root
root = resolve_root()
cfg = build_config(root)
idx = Path(cfg['episodes_index_path'])
workers = int(os.environ.get('TENSOR_BUILD_WORKERS', cfg.get('tensor_build_workers', 2)))
slugs = ensure_episodes_index_data(root, index_path=idx)
convert_episodes_index_to_rollouts(
    root, out_path=Path('${ARCH_LADDER_DATA}'), episodes_index=idx,
    top_percent=100.0, max_episodes=int('${ARCH_LADDER_EPISODES:-5000}'),
    daily_slugs=slugs, workers=workers,
)
"
    fi
    echo "==> merge CABT + ladder -> ${ARCH_BOOTSTRAP_DATA} (${SELF_PLAY_WORKERS} workers)"
    python3 scripts/merge_rollouts.py \
      "$ARCH_CABT_DATA" "$ARCH_LADDER_DATA" \
      --out "$ARCH_BOOTSTRAP_DATA" \
      --workers "${SELF_PLAY_WORKERS}"
  else
    ensure_bootstrap_data "$TRAINING_DATA_PATH" "$ARCH_FIELD_DIR" "$ARCH_BOOTSTRAP_EPISODES"
  fi

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

SELF_PLAY_CHECKPOINT="${MODEL_OUTPUT_PATH}"
if [[ "$MODE" == "self-play-only" ]] && [[ -f "${SELF_PLAY_CHECKPOINT_DIR}/manifest.json" ]]; then
  SELF_PLAY_CHECKPOINT="$(python3 -c "
from pathlib import Path
from poke_agent.self_play.rollout_io import load_manifest
from poke_agent.kaggle_submit import champion_checkpoint_from_manifest
manifest = load_manifest(Path('${SELF_PLAY_CHECKPOINT_DIR}/manifest.json'))
print(champion_checkpoint_from_manifest(manifest, Path('${MODEL_OUTPUT_PATH}')))
")"
  echo "==> self-play resume checkpoint: ${SELF_PLAY_CHECKPOINT}"
fi

echo "==> curriculum self-play"
python3 scripts/run_self_play.py --curriculum \
  --checkpoint "${SELF_PLAY_CHECKPOINT}" \
  --workers "${SELF_PLAY_WORKERS}"
