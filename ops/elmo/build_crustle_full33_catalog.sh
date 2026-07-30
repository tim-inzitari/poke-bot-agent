#!/usr/bin/env bash
# Scan the full current public archive for the collision-safe Crustle family.
set -euo pipefail

source_root="${POKEBOT_CRUSTLE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output="${POKEBOT_CRUSTLE_CATALOG_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/crustle-public-family-full33-v1.json}"
scanner="$source_root/scripts/build_public_plain_dragapult_catalog.py"
teacher="$source_root/poke_bot/crustle_heuristics.py"
scanner_sha256="${POKEBOT_CRUSTLE_SCANNER_SHA256:?scanner checksum is required}"
teacher_sha256="${POKEBOT_CRUSTLE_TEACHER_SHA256:?teacher checksum is required}"

test -s "$scanner"
test -s "$teacher"
test "$(sha256sum "$scanner" | awk '{print $1}')" = "$scanner_sha256"
test "$(sha256sum "$teacher" | awk '{print $1}')" = "$teacher_sha256"

cd "$source_root"
python3 "$scanner" \
  --archive-dir "$archive_root" \
  --start 2026-06-26 \
  --end 2026-07-28 \
  --specialist-id crustle \
  --minimum-records 1 \
  --workers "${POKEBOT_CRUSTLE_CATALOG_WORKERS:-4}" \
  --out "$output"
