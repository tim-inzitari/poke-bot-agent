#!/usr/bin/env bash
# Install official Kaggle rule-based baseline agents under baselines/official/.
# Sources: Kiyota sample notebooks + deck datasets from discussion #708584.
set -euo pipefail

cd "$(dirname "$0")/.."
KAGGLE="${KAGGLE_BIN:-$(command -v kaggle || true)}"
if [[ -z "$KAGGLE" && -x "$HOME/miniconda3/envs/poke-bot-agent/bin/kaggle" ]]; then
  KAGGLE="$HOME/miniconda3/envs/poke-bot-agent/bin/kaggle"
fi
if [[ -z "$KAGGLE" ]]; then
  echo "kaggle CLI not found; install kaggle and configure ~/.kaggle/kaggle.json" >&2
  exit 1
fi

BASE="baselines"
mkdir -p "$BASE/kernels" "$BASE/decks" "$BASE/official"

declare -A KERNELS=(
  [iono]="kiyotah/a-sample-rule-based-agent-iono-s-deck"
  [dragapult-ex]="kiyotah/a-sample-rule-based-agent-dragapult-ex-deck"
  [mega-abomasnow-ex]="kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck"
  [mega-lucario-ex]="kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck"
)
declare -A DECKS=(
  [iono]="kiyotah/iono-deck"
  [dragapult-ex]="kiyotah/dragapult-ex-deck"
  [mega-abomasnow-ex]="kiyotah/mega-abomasnow-ex-deck"
  [mega-lucario-ex]="kiyotah/mega-lucario-ex-deck"
)

for dir in "${!KERNELS[@]}"; do
  echo "pull kernel ${KERNELS[$dir]}"
  mkdir -p "$BASE/kernels/$dir"
  "$KAGGLE" kernels pull "${KERNELS[$dir]}" -p "$BASE/kernels/$dir" -m

  echo "pull deck ${DECKS[$dir]}"
  mkdir -p "$BASE/decks/$dir"
  "$KAGGLE" datasets download -d "${DECKS[$dir]}" -p "$BASE/decks/$dir" --unzip
done

python3 scripts/extract_baseline_agents.py
echo "installed baseline agents under $BASE/official/"
