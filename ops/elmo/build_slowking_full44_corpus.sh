#!/usr/bin/env bash
# Materialize and seal the exact pictured Slowking combo/toolbox corpus.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-slowking-r75-src-v2}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_root="${POKEBOT_SLOWKING_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/slowking-guide-corpus-full44-v3}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
catalog_source="${POKEBOT_SLOWKING_CATALOG_SOURCE:-$source_root/data/training_mixes/slowking-public-full44.v1.json}"
teacher_module="$source_root/poke_bot/slowking_heuristics.py"
catalog="$output_root/PUBLIC_DECK_ARCHETYPE_CATALOG.json"
catalog_sha256="${POKEBOT_SLOWKING_CATALOG_SHA256:?catalog checksum is required}"
teacher_sha256="${POKEBOT_SLOWKING_TEACHER_SHA256:?teacher checksum is required}"
status="$output_root/status/window.json"
start_date="${POKEBOT_SLOWKING_START_DATE:-2026-06-16}"
end_date="${POKEBOT_SLOWKING_END_DATE:-2026-07-29}"
day_parallelism="${POKEBOT_SLOWKING_DAY_PARALLELISM:-3}"
workers_per_day="${POKEBOT_SLOWKING_WORKERS_PER_DAY:-2}"
max_in_flight_per_day="${POKEBOT_SLOWKING_MAX_IN_FLIGHT_PER_DAY:-4}"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$catalog_source" | awk '{print $1}')" = "$catalog_sha256"
test "$(sha256sum "$teacher_module" | awk '{print $1}')" = "$teacher_sha256"

minimum_records="$(
  python3 - "$catalog_source" "$start_date" "$end_date" <<'PY'
import json
from pathlib import Path
import sys

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
window = dict(catalog.get("source_window") or {})
identity = dict(catalog.get("identity_contract") or {})
if (
    catalog.get("schema") != "poke_bot.public_deck_archetype_catalog/v1"
    or catalog.get("specialist_id") != "slowking"
    or window.get("start") != sys.argv[2]
    or window.get("end") != sys.argv[3]
    or int(window.get("days") or 0) != 44
    or int(catalog.get("observed_acting_seat_games") or 0) != 311
    or int(catalog.get("source_match_rows") or 0) != 311
    or len(catalog.get("source_match_facts") or []) != 311
    or catalog.get("deck_fingerprints") != [
        "sha256:56ac56d0d2bf1ae0c2d18562422492cffbfafc04b7fe8292cf355809a84c1cf7"
    ]
    or identity.get("mode")
    != "owner_exact_60_card_slowking_public_replay_identity"
    or identity.get("broad_archetype_name_filter_sufficient") is not False
):
    raise SystemExit("Slowking public catalog identity is invalid")
print(311)
PY
)"

mkdir -p "$output_root/status"
if [[ -e "$catalog" ]]; then
  test "$(sha256sum "$catalog" | awk '{print $1}')" = "$catalog_sha256"
else
  install -m 0444 "$catalog_source" "$catalog"
fi

export CG_LIB_PATH="$cg_runtime"
cd "$source_root"
empty_template="$output_root/slowking-2026-06-16.features"
if [[ ! -s "$empty_template" ]]; then
  python3 scripts/materialize_authoritative_alakazam_day.py \
    --date 2026-06-16 \
    --archive "$archive_root/pokemon-tcg-ai-battle-episodes-2026-06-16.zip" \
    --out "$empty_template" \
    --workers "$workers_per_day" \
    --max-in-flight "$max_in_flight_per_day" \
    --max-context 320 \
    --memory-floor-gib 20 \
    --min-records 0 \
    --required-archetype slowking \
    --current-deck-guide slowking \
    --authoritative-deck-catalog "$catalog" \
    --authoritative-only-archetype slowking
fi
python3 scripts/materialize_catalog_index_empty_days.py \
  --catalog "$catalog" \
  --output-root "$output_root" \
  --template-day 2026-06-16

python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype slowking \
  --current-deck-guide slowking \
  --authoritative-deck-catalog "$catalog" \
  --authoritative-only-archetype slowking \
  --day-parallelism "$day_parallelism" \
  --workers-per-day "$workers_per_day" \
  --max-in-flight-per-day "$max_in_flight_per_day" \
  --max-context 320 \
  --memory-floor-gib 20 \
  --min-records 0

python3 scripts/finalize_current_deck_guide_window.py \
  --status "$status" \
  --out-dir "$output_root" \
  --start "$start_date" \
  --end "$end_date" \
  --specialist-id slowking \
  --guide-version slowking-north-star-v1 \
  --minimum-records "$minimum_records" \
  --public-deck-catalog "$catalog"

python3 - "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" "$catalog" <<'PY'
import json
from pathlib import Path
import sys

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
catalog = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {
    str(day): int(records)
    for day, records in dict(catalog.get("observed_by_day") or {}).items()
}
actual = {
    str(row.get("date")): int(row.get("records") or 0)
    for row in ready.get("daily_shards") or []
}
guide = {
    str(row.get("date")): int(row.get("guide_rows") or 0)
    for row in ready.get("daily_shards") or []
}
if (
    ready.get("status") != "ready"
    or ready.get("specialist_id") != "slowking"
    or int(ready.get("records") or 0) != 311
    or actual != expected
    or int(ready.get("guide_rows") or 0) <= 0
    or any(day not in guide or guide[day] < 0 for day in expected)
    or (ready.get("source_policy") or {}).get("mode")
    != "public_full_history_exact_deck_identity"
):
    raise SystemExit("sealed Slowking corpus failed its final contract")
print(json.dumps(ready, sort_keys=True))
PY
