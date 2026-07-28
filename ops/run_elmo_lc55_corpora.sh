#!/usr/bin/env bash
# Build one authoritative latest-10 temporal corpus on Elmo, then derive both
# the balanced deck-agnostic core and exact acting-seat Starmie corpus.
set -euo pipefail

IMAGE="${POKEBOT_EXPERT_IMAGE:-poke-bot-truenas-worker:matchup-router-v22-clean}"
NAME="pokebot-lc55-corpora"
ROOT="/mnt/Main/main/poke-bot-agent/expert-builder"
SOURCE="$ROOT/src"
RAW="/mnt/Main/main/poke-feature-refresh-20260721/data/episodes/raw"
OUTPUT="$ROOT/lc55-corpora"

for required in \
  "$SOURCE/scripts/materialize_authoritative_archetype_window.py" \
  "$SOURCE/scripts/assemble_feature_manifest.py" \
  "$SOURCE/scripts/build_balanced_core_manifest.py" \
  "$SOURCE/scripts/filter_feature_manifest.py" \
  "$SOURCE/cards/EN_Card_Data.csv"; do
  test -s "$required"
done
for day in {11..20}; do
  test -s "$RAW/pokemon-tcg-ai-battle-episodes-2026-07-$day.zip"
done

mkdir -p "$OUTPUT"
if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(sudo -n docker inspect -f '{{.State.Status}}' "$NAME")"
  if [[ "$state" == "running" ]]; then
    echo "$NAME already running"
    exit 0
  fi
  sudo -n docker rm "$NAME" >/dev/null
fi

sudo -n docker run -d \
  --name "$NAME" \
  --restart on-failure:5 \
  --cpus 4 \
  --memory 20g \
  --memory-swap 20g \
  --pids-limit 1024 \
  -e PYTHONUNBUFFERED=1 \
  -v "$SOURCE:/workspace:ro" \
  -v "$RAW:/input:ro" \
  -v "$OUTPUT:/output" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
dates=()
for day in {11..20}; do dates+=(--expected-date "2026-07-$day"); done
targets=(
  temporal_action_rows
  opponent_hand_rows
  opponent_remainder_rows
  opponent_private_prize_rows
  lethal_threat_rows
  prize_race_rows
)
target_args=()
for target in "${targets[@]}"; do target_args+=(--require-target-coverage "$target"); done

python -u scripts/materialize_authoritative_archetype_window.py \
  --start 2026-07-11 --end 2026-07-20 \
  --archetype "*" \
  --archive-dir /input \
  --out-dir /output/all-recognized \
  --status /output/all-recognized-status.json \
  --mix /workspace/data/training_mixes/top_ladder.v1.json \
  --representatives /workspace/data/training_mixes/top_ladder_representatives.v1.json \
  --card-csv /workspace/cards/EN_Card_Data.csv \
  --workers 2 --max-in-flight 4 --max-context 320 \
  --memory-floor-gib 8 --min-records 1

python -u scripts/assemble_feature_manifest.py \
  --staging-dir /output/all-recognized \
  --out /output/all-recognized/manifest.json \
  "${dates[@]}" \
  --compact-mode temporal-expert-v1 \
  --required-archetype "*" \
  --expected-max-context 320 \
  "${target_args[@]}" \
  --min-free-gib 100

python -u scripts/build_balanced_core_manifest.py \
  --source-manifest /output/all-recognized/manifest.json \
  --output-dir /output/core-balanced-v1 \
  --max-records-per-archetype 2500 \
  --max-decisions-per-archetype 220000

python -u scripts/filter_feature_manifest.py \
  --source-manifest /output/all-recognized/manifest.json \
  --output-dir /output/starmie \
  --archetype starmie \
  --workers 2

python - <<"PY"
import json
from pathlib import Path
core = json.loads(Path("/output/core-balanced-v1/PROTECTED_CORE_CORPUS.json").read_text())
starmie = json.loads(Path("/output/starmie/PROTECTED_EXPERT_CORPUS.json").read_text())
Path("/output/READY.json").write_text(json.dumps({
    "schema": "poke_bot.lc55_handoff_corpora_ready/v1",
    "core": core,
    "starmie": starmie,
}, indent=2, sort_keys=True) + "\n")
PY
'

echo "started $NAME; logs: sudo docker logs -f $NAME"
