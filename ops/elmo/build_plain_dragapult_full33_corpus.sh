#!/usr/bin/env bash
# Materialize and seal the full available exact-identity plain Dragapult corpus.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_root="${POKEBOT_DRAGAPULT_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/dragapult-guide-corpus-plain-full33-v1}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
catalog_source="${POKEBOT_DRAGAPULT_CATALOG_SOURCE:-$source_root/data/training_mixes/dragapult-public-plain-full33.v1.json}"
teacher_module="$source_root/poke_bot/dragapult_heuristics.py"
catalog="$output_root/PUBLIC_DECK_ARCHETYPE_CATALOG.json"
catalog_sha256="${POKEBOT_DRAGAPULT_CATALOG_SHA256:?catalog checksum is required}"
teacher_sha256="${POKEBOT_DRAGAPULT_TEACHER_SHA256:?teacher checksum is required}"
status="$output_root/status/window.json"
start_date="${POKEBOT_DRAGAPULT_START_DATE:-2026-06-26}"
end_date="${POKEBOT_DRAGAPULT_END_DATE:-2026-07-28}"
day_parallelism="${POKEBOT_DRAGAPULT_DAY_PARALLELISM:-3}"
workers_per_day="${POKEBOT_DRAGAPULT_WORKERS_PER_DAY:-2}"
max_in_flight_per_day="${POKEBOT_DRAGAPULT_MAX_IN_FLIGHT_PER_DAY:-4}"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$catalog_source" | awk '{print $1}')" = "$catalog_sha256"
test "$(sha256sum "$teacher_module" | awk '{print $1}')" = "$teacher_sha256"

minimum_records="$(
  python3 - "$catalog_source" "$start_date" "$end_date" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
window = dict(catalog.get("source_window") or {})
rows = list(catalog.get("source_deck_rows") or [])
fingerprints = list(catalog.get("deck_fingerprints") or [])
identity = dict(catalog.get("identity_contract") or {})
if (
    catalog.get("schema") != "poke_bot.public_deck_archetype_catalog/v1"
    or catalog.get("specialist_id") != "dragapult"
    or window.get("start") != sys.argv[2]
    or window.get("end") != sys.argv[3]
    or int(window.get("days") or 0) != 33
    or int(catalog.get("observed_acting_seat_games") or 0) <= 0
    or int(catalog.get("source_match_rows") or 0)
    != int(catalog.get("observed_acting_seat_games") or 0)
    or not rows
    or not fingerprints
    or identity.get("mode") != "exact_60_card_public_replay_identity"
    or identity.get("broad_archetype_name_filter_sufficient") is not False
):
    raise SystemExit("plain Dragapult public catalog identity is invalid")
print(int(catalog["observed_acting_seat_games"]))
PY
)"

python3 - "$catalog_source" "$archive_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
for row in catalog.get("source_archives") or []:
    archive = root / str(row.get("archive") or "")
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise SystemExit(f"missing public replay archive: {archive}")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if "sha256:" + digest.hexdigest() != row.get("archive_sha256"):
        raise SystemExit(f"public replay archive checksum changed: {archive}")
print(
    json.dumps(
        {
            "status": "all_public_archives_checksum_verified",
            "days": len(catalog.get("source_archives") or []),
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

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
  --required-archetype dragapult \
  --current-deck-guide dragapult \
  --authoritative-deck-catalog "$catalog" \
  --authoritative-only-archetype dragapult \
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
  --specialist-id dragapult \
  --guide-version dragapult-north-star-v1 \
  --minimum-records "$minimum_records" \
  --public-deck-catalog "$catalog"

python3 - \
  "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" \
  "$catalog" <<'PY'
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
guide_by_day = {
    str(row.get("date")): int(row.get("guide_rows") or 0)
    for row in ready.get("daily_shards") or []
}
expected_records = int(catalog.get("observed_acting_seat_games") or 0)
invalid_guide_days = {
    day: guide_by_day.get(day)
    for day in expected_by_day
    if day not in guide_by_day or guide_by_day[day] < 0
}
if (
    ready.get("status") != "ready"
    or ready.get("specialist_id") != "dragapult"
    or int(ready.get("records") or 0) != expected_records
    or actual_by_day != expected_by_day
    or int(ready.get("guide_rows") or 0) <= 0
    # A checksum-valid game day may legitimately have no supported exact
    # guide labels when every candidate is masked. Require complete,
    # nonnegative daily accounting and positive corpus-wide coverage instead.
    or invalid_guide_days
    or (ready.get("source_policy") or {}).get("mode")
    != "public_full_history_exact_deck_identity"
):
    raise SystemExit(
        "sealed plain Dragapult full-history corpus failed its final contract"
    )
print(json.dumps(ready, sort_keys=True))
PY
