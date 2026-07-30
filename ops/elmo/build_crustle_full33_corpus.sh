#!/usr/bin/env bash
# Materialize and seal the full available family-identity Crustle guide corpus.
set -euo pipefail

source_root="${POKEBOT_CRUSTLE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
catalog_source="${POKEBOT_CRUSTLE_CATALOG_SOURCE:-/mnt/Main/main/poke-bot-agent/archive/crustle-public-family-full33-v1.json}"
output_root="${POKEBOT_CRUSTLE_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/crustle-guide-corpus-family-full33-v1}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
teacher_module="$source_root/poke_bot/crustle_heuristics.py"
catalog="$output_root/PUBLIC_DECK_ARCHETYPE_CATALOG.json"
teacher_sha256="${POKEBOT_CRUSTLE_TEACHER_SHA256:?teacher checksum is required}"
status="$output_root/status/window.json"
start_date="${POKEBOT_CRUSTLE_START_DATE:-2026-06-26}"
end_date="${POKEBOT_CRUSTLE_END_DATE:-2026-07-28}"
day_parallelism="${POKEBOT_CRUSTLE_DAY_PARALLELISM:-2}"
workers_per_day="${POKEBOT_CRUSTLE_WORKERS_PER_DAY:-2}"
max_in_flight_per_day="${POKEBOT_CRUSTLE_MAX_IN_FLIGHT_PER_DAY:-4}"
minimum_owner_records="${POKEBOT_CRUSTLE_MINIMUM_RECORDS:-16639}"

test -f "$cg_runtime/cg/__init__.py"
test -s "$catalog_source"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$teacher_module" | awk '{print $1}')" = "$teacher_sha256"

catalog_sha256="$(sha256sum "$catalog_source" | awk '{print $1}')"
minimum_records="$(
  python3 - "$catalog_source" "$archive_root" "$start_date" "$end_date" \
    "$minimum_owner_records" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
archive_root = Path(sys.argv[2])
window = dict(catalog.get("source_window") or {})
rows = list(catalog.get("source_deck_rows") or [])
fingerprints = list(catalog.get("deck_fingerprints") or [])
identity = dict(catalog.get("identity_contract") or {})
observed = int(catalog.get("observed_acting_seat_games") or 0)
minimum_owner_records = int(sys.argv[5])
if (
    catalog.get("schema") != "poke_bot.public_deck_archetype_catalog/v1"
    or catalog.get("specialist_id") != "crustle"
    or window.get("start") != sys.argv[3]
    or window.get("end") != sys.argv[4]
    or int(window.get("days") or 0) != 33
    or observed < minimum_owner_records
    or int(catalog.get("source_match_rows") or 0) != observed
    or not rows
    or not fingerprints
    or identity.get("mode")
    != "crustle_card_signature_public_replay_identity"
    or identity.get("broad_archetype_name_filter_sufficient") is not False
):
    raise SystemExit("Crustle public catalog identity is invalid")
for row in catalog.get("source_archives") or []:
    archive = archive_root / str(row.get("archive") or "")
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise SystemExit(f"missing public replay archive: {archive}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if "sha256:" + digest.hexdigest() != row.get("archive_sha256"):
        raise SystemExit(f"public replay archive checksum changed: {archive}")
print(minimum_owner_records)
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
python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype crustle \
  --current-deck-guide crustle \
  --authoritative-deck-catalog "$catalog" \
  --authoritative-only-archetype crustle \
  --day-parallelism "$day_parallelism" \
  --workers-per-day "$workers_per_day" \
  --max-in-flight-per-day "$max_in_flight_per_day" \
  --max-context 320 \
  --memory-floor-gib 20 \
  --min-records 0

test "$(sha256sum "$catalog_source" | awk '{print $1}')" = "$catalog_sha256"
python3 scripts/finalize_current_deck_guide_window.py \
  --status "$status" \
  --out-dir "$output_root" \
  --start "$start_date" \
  --end "$end_date" \
  --specialist-id crustle \
  --guide-version crustle-north-star-v1 \
  --minimum-records "$minimum_records" \
  --public-deck-catalog "$catalog"

python3 - \
  "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" \
  "$catalog" "$catalog_sha256" <<'PY'
import json
from pathlib import Path
import sys

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
catalog = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_by_day = {
    str(day): int(records)
    for day, records in dict(catalog.get("observed_by_day") or {}).items()
}
actual_by_day = {
    str(row.get("date")): int(row.get("records") or 0)
    for row in ready.get("daily_shards") or []
}
expected_records = int(catalog.get("observed_acting_seat_games") or 0)
actual_records = int(ready.get("records") or 0)
source_policy = dict(ready.get("source_policy") or {})
minimum_records = int(source_policy.get("minimum_records") or 0)
excluded_records = expected_records - actual_records
if (
    ready.get("status") != "ready"
    or ready.get("specialist_id") != "crustle"
    or ready.get("guide_version") != "crustle-north-star-v1"
    or int(ready.get("days") or 0) != 33
    or actual_records < minimum_records
    or actual_records > expected_records
    or set(actual_by_day) != set(expected_by_day)
    or any(
        actual_by_day[day] > expected_by_day[day]
        for day in expected_by_day
    )
    or sum(actual_by_day.values()) != actual_records
    or excluded_records != sum(
        expected_by_day[day] - actual_by_day[day]
        for day in expected_by_day
    )
    or int(ready.get("guide_rows") or 0) <= 0
    or source_policy.get("mode")
    != "public_full_history_card_signature_identity"
    or str(
        source_policy.get("public_deck_catalog_sha256") or ""
    ).removeprefix("sha256:")
    != sys.argv[3]
):
    raise SystemExit(
        "sealed Crustle full-history corpus failed its final contract"
    )
print(json.dumps(ready, sort_keys=True))
PY
