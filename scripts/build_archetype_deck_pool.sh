#!/usr/bin/env bash
# Build focused deck directories for single-archetype training runs.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 <abomasnow|iono|starmie|crustle>" >&2
  exit 1
}

ARCH="${1:-}"
[[ -n "$ARCH" ]] || usage

case "$ARCH" in
  abomasnow)
    mkdir -p decks/abomasnow-only
    cp baselines/official/mega-abomasnow-ex/deck.csv \
      decks/abomasnow-only/official-mega-abomasnow-ex.csv
    ;;
  iono)
    mkdir -p decks/iono-only
    cp baselines/official/iono/deck.csv decks/iono-only/official-iono-bellibolt.csv
    ;;
  starmie)
    for src in decks/competitive/high_performing decks/competitive/the_rest; do
      python3 scripts/filter_deck_dir.py --pattern starmie --source "$src" --dest decks/starmie-only
    done
    ;;
  crustle)
    for src in decks/competitive/high_performing decks/competitive/the_rest; do
      python3 scripts/filter_deck_dir.py --pattern crustle --source "$src" --dest decks/crustle-only
    done
    ;;
  *)
    usage
    ;;
esac

echo "built decks/${ARCH}-only"
