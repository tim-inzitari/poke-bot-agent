#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Train one neural model per major public-baseline deck family.
# CABT simulations run in Docker/Linux; neural inference/training stays on Mac.

DECKS="${DECKS:-dragapult,lucario,crustle,alakazam,stonjourner}"
ITERATIONS="${ITERATIONS:-10}"
GAMES="${GAMES:-40}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_GAMES="${BATCH_GAMES:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
TARGET_WIN_RATE="${TARGET_WIN_RATE:-0.60}"
BASELINES="${BASELINES:-public}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-240}"
BASE_POLICY_PORT="${BASE_POLICY_PORT:-19000}"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "missing BASE_CHECKPOINT=$BASE_CHECKPOINT" >&2
  echo "Create one with scripts/train_agent.py or set BASE_CHECKPOINT=/path/to/current-format.pt" >&2
  exit 1
fi

deck_path_for() {
  case "$1" in
    dragapult) echo "decks/submission.csv" ;;
    lucario) echo "baselines/kaggle_public/kojimar_simple_lucario/deck.csv" ;;
    crustle) echo "baselines/kaggle_public/dashimaki_day1_crustle/deck.csv" ;;
    alakazam) echo "baselines/kaggle_public/ryota_alakazam_best5/deck.csv" ;;
    stonjourner) echo "baselines/kaggle_public/alyce_lucario_v2_bot/deck.csv" ;;
    *)
      echo "unknown deck key: $1" >&2
      echo "known: dragapult,lucario,crustle,alakazam,stonjourner" >&2
      exit 2
      ;;
  esac
}

mkdir -p outputs/checkpoints/deck_models outputs/rollouts/deck_models outputs/reports/deck_models

echo "Deck model farm"
echo "  decks:           $DECKS"
echo "  iterations/deck: $ITERATIONS"
echo "  games/iteration: $GAMES"
echo "  train epochs:    $TRAIN_EPOCHS"
echo "  batch games:     $BATCH_GAMES"
echo "  early stop:      patience=$EARLY_STOP_PATIENCE min_delta=$EARLY_STOP_MIN_DELTA"
echo "  target winrate:  $TARGET_WIN_RATE"
echo "  baselines:       $BASELINES"
echo "  base checkpoint: $BASE_CHECKPOINT"
echo

IFS=',' read -r -a deck_keys <<< "$DECKS"
deck_index=0
for raw_key in "${deck_keys[@]}"; do
  deck_key="$(echo "$raw_key" | xargs)"
  [[ -z "$deck_key" ]] && continue
  deck_path="$(deck_path_for "$deck_key")"
  if [[ ! -f "$deck_path" ]]; then
    echo "missing deck for $deck_key: $deck_path" >&2
    exit 1
  fi

  checkpoint_dir="outputs/checkpoints/deck_models/${deck_key}"
  rollouts="outputs/rollouts/deck_models/${deck_key}.jsonl"
  report_dir="outputs/reports/deck_models/${deck_key}"
  mkdir -p "$checkpoint_dir" "$report_dir"

  latest_checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -name 'iter_*.pt' -type f 2>/dev/null | sort | tail -1 || true)"
  if [[ -z "$latest_checkpoint" ]]; then
    latest_checkpoint="$BASE_CHECKPOINT"
  fi

  port="$((BASE_POLICY_PORT + deck_index))"
  deck_index="$((deck_index + 1))"

  echo
  echo "################################################################################"
  echo "# ${deck_key}: neural model from ${deck_path}"
  echo "# checkpoint in: ${latest_checkpoint}"
  echo "# checkpoint dir: ${checkpoint_dir}"
  echo "################################################################################"

  POLICY_PORT="$port" \
  POLICY_TIMEOUT="$POLICY_TIMEOUT" \
  ITERATIONS="$ITERATIONS" \
  GAMES="$GAMES" \
  TRAIN_EPOCHS="$TRAIN_EPOCHS" \
  BATCH_GAMES="$BATCH_GAMES" \
  EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
  EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
  TARGET_WIN_RATE="$TARGET_WIN_RATE" \
  BASELINES="$BASELINES" \
  CHECKPOINT="$latest_checkpoint" \
  CHECKPOINT_DIR="$checkpoint_dir" \
  ROLLOUTS="$rollouts" \
  DECK="$deck_path" \
    ./scripts/run_remote_public_baseline_rl.sh

  final_checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -name 'iter_*.pt' -type f | sort | tail -1)"
  cp "$final_checkpoint" "$checkpoint_dir/${deck_key}_latest.pt"
  echo "$final_checkpoint" > "$checkpoint_dir/latest.txt"
  echo "saved ${deck_key} latest model: $checkpoint_dir/${deck_key}_latest.pt"
done

echo
echo "Deck model farm complete."
for raw_key in "${deck_keys[@]}"; do
  deck_key="$(echo "$raw_key" | xargs)"
  [[ -z "$deck_key" ]] && continue
  echo "  ${deck_key}: outputs/checkpoints/deck_models/${deck_key}/${deck_key}_latest.pt"
done
