#!/usr/bin/env bash
# Convert and featurize exactly one official ladder day with bounded memory.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 YYYY-MM-DD COLLECT_WORKERS FEATURE_WORKERS OUTPUT_DIR" >&2
  exit 2
fi

DAY="$1"
COLLECT_WORKERS="$2"
FEATURE_WORKERS="$3"
OUTPUT_DIR="$4"
PYTHON_BIN="${POKEBOT_PYTHON:-python3}"

if [[ ! "$DAY" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "invalid date: $DAY" >&2
  exit 2
fi
if (( COLLECT_WORKERS < 1 || FEATURE_WORKERS < 1 )); then
  echo "worker counts must be positive" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
JSONL="$OUTPUT_DIR/top_ladder_all_${DAY}.jsonl"
FEATURES="$OUTPUT_DIR/top_ladder_all_${DAY}.features"
ARCHIVE="data/episodes/raw/pokemon-tcg-ai-battle-episodes-${DAY}.zip"

if [[ ! -s "$ARCHIVE" ]]; then
  echo "missing daily archive: $ARCHIVE" >&2
  exit 1
fi
if [[ -e "$FEATURES" ]]; then
  echo "refusing to replace completed feature shard: $FEATURES" >&2
  exit 1
fi

AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( AVAILABLE_KIB < 8 * 1024 * 1024 )); then
  echo "free-memory guard failed: MemAvailable=${AVAILABLE_KIB} KiB" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ "${POKEBOT_REUSE_JSONL:-0}" == "1" ]]; then
  if [[ ! -s "$JSONL" ]]; then
    echo "requested JSONL reuse but file is missing: $JSONL" >&2
    exit 1
  fi
  echo "[stage collect] day=$DAY reuse=$JSONL"
else
  echo "[stage collect] day=$DAY workers=$COLLECT_WORKERS archive=$ARCHIVE"
  "$PYTHON_BIN" -u scripts/collect_top_ladder_replays.py \
    --start-date "$DAY" \
    --end-date "$DAY" \
    --workers "$COLLECT_WORKERS" \
    --min-sequences 8000 \
    --min-recognized-seat-frac 0.90 \
    --out "$JSONL" \
    --skip-download \
    --replace
fi

echo "[stage feature] day=$DAY workers=$FEATURE_WORKERS jsonl=$JSONL"
"$PYTHON_BIN" -u scripts/featurize_bootstrap_shard.py \
  --jsonl "$JSONL" \
  --out "$FEATURES" \
  --source-date "$DAY" \
  --workers "$FEATURE_WORKERS" \
  --max-in-flight "$(( FEATURE_WORKERS * 2 ))"

echo "[complete] day=$DAY features=$FEATURES metadata=${FEATURES}.json"
