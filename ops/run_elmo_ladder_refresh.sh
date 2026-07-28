#!/usr/bin/env bash
# Refresh the newest complete ten-day Kaggle ladder window on Elmo.
# Existing validated archives are hard-linked into the versioned staging root;
# only missing days cross the WAN.  This does not touch the live expert corpus.
set -euo pipefail

ROOT="${POKEBOT_REFRESH_ROOT:-/mnt/Main/main/poke-feature-refresh-20260721}"
REUSE_ROOT="${POKEBOT_REUSE_ROOT:-/mnt/Main/main/poke-feature-latest10}"
PYDEPS="$ROOT/.pydeps"
INDEX="$ROOT/kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"
RAW="$ROOT/data/episodes/raw"
STATE="$ROOT/refresh-status.tsv"

export HOME="${POKEBOT_KAGGLE_HOME:-/home/admin}"
export PYTHONPATH="$PYDEPS"

mkdir -p "$RAW"
if [[ ! -s "$INDEX" ]]; then
  echo "missing refreshed episodes index: $INDEX" >&2
  exit 1
fi
if [[ ! -d "$PYDEPS/kaggle" ]]; then
  echo "missing isolated Kaggle client: $PYDEPS/kaggle" >&2
  exit 1
fi

mapfile -t DAYS < <(tail -n 10 "$INDEX" | cut -d, -f1)
if [[ "${#DAYS[@]}" -ne 10 ]]; then
  echo "episodes index did not yield ten complete days" >&2
  exit 1
fi

write_state() {
  local stage="$1" day="${2:-}" completed="${3:-0}" detail="${4:-}"
  local temporary="${STATE}.tmp"
  printf 'stage\t%s\nday\t%s\ncompleted\t%s\ntotal\t10\nwindow_start\t%s\nwindow_end\t%s\ndetail\t%s\nupdated_epoch\t%s\n' \
    "$stage" "$day" "$completed" "${DAYS[0]}" "${DAYS[9]}" "$detail" "$(date +%s)" >"$temporary"
  mv "$temporary" "$STATE"
}

completed=0
write_state preparing "" "$completed" "selecting latest complete ten-day window"
for day in "${DAYS[@]}"; do
  archive="pokemon-tcg-ai-battle-episodes-${day}.zip"
  destination="$RAW/$archive"
  reusable="$REUSE_ROOT/data/episodes/raw/$archive"

  if [[ -s "$destination" ]]; then
    write_state validating "$day" "$completed" "validating existing staged archive"
  elif [[ -s "$reusable" ]]; then
    write_state reusing "$day" "$completed" "hard-linking previously validated archive"
    ln "$reusable" "$destination"
  else
    write_state downloading "$day" "$completed" "downloading missing Kaggle day on Elmo"
    python3 -m kaggle.cli datasets download \
      "kaggle/pokemon-tcg-ai-battle-episodes-${day}" \
      -p "$RAW" -o
  fi

  if [[ ! -s "$destination" ]]; then
    write_state failed "$day" "$completed" "archive missing after download"
    echo "archive missing after download: $destination" >&2
    exit 1
  fi
  write_state validating "$day" "$completed" "testing archive integrity"
  unzip -tq "$destination" >/dev/null
  completed=$((completed + 1))
  write_state ready_day "$day" "$completed" "$archive validated"
  echo "[$completed/10] ready $day ($(stat -c %s "$destination") bytes)"
done

write_state ready "${DAYS[9]}" "$completed" "latest ten-day archive window validated on Elmo"
echo "latest-ten refresh ready: ${DAYS[0]} through ${DAYS[9]}"
