#!/usr/bin/env bash
# Download immutable daily Kaggle trace archives for temporal expert materialization.
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 ARCHIVE_DIR YYYY-MM-DD [YYYY-MM-DD ...]" >&2
  exit 2
fi

archive_dir="$1"
shift
python_bin="${POKEBOT_PYTHON:-/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python}"
minimum_free_gib="${POKEBOT_EXPERT_DOWNLOAD_MIN_FREE_GIB:-200}"

mkdir -p "$archive_dir/.download"
for day in "$@"; do
  if [[ ! "$day" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "invalid date: $day" >&2
    exit 2
  fi
  free_kib="$(df -Pk "$archive_dir" | awk 'NR == 2 {print $4}')"
  if (( free_kib < minimum_free_gib * 1024 * 1024 )); then
    echo "free-space guard failed before $day: ${free_kib} KiB available" >&2
    exit 1
  fi

  slug="pokemon-tcg-ai-battle-episodes-${day}"
  destination="$archive_dir/${slug}.zip"
  if [[ -s "$destination" ]]; then
    unzip -tq "$destination" >/dev/null
    echo "[download] reuse validated day=$day path=$destination"
    continue
  fi

  staging="$archive_dir/.download/$day"
  rm -rf "$staging"
  mkdir -p "$staging"
  echo "[download] begin day=$day dataset=kaggle/$slug"
  "$python_bin" -m kaggle datasets download "kaggle/$slug" -p "$staging"
  candidate="$staging/${slug}.zip"
  if [[ ! -s "$candidate" ]]; then
    echo "Kaggle download did not create the expected archive: $candidate" >&2
    exit 1
  fi
  unzip -tq "$candidate" >/dev/null
  mv "$candidate" "$destination"
  chmod 0444 "$destination"
  rmdir "$staging"
  echo "[download] complete day=$day bytes=$(stat -c %s "$destination") path=$destination"
done
