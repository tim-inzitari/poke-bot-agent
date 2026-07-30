#!/usr/bin/env bash
# Materialize and seal the full available public Teal Mask Ogerpon ex corpus.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_root="${POKEBOT_TEAL_MASK_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/teal-mask-ogerpon-ex-guide-corpus-full-v2}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
catalog_source="${POKEBOT_TEAL_MASK_CATALOG_SOURCE:-$source_root/data/training_mixes/teal-mask-ogerpon-ex-public-full32.v1.json}"
teacher_module="$source_root/poke_bot/teal_mask_ogerpon_heuristics.py"
catalog="$output_root/PUBLIC_DECK_ARCHETYPE_CATALOG.json"
catalog_sha256="${POKEBOT_TEAL_MASK_CATALOG_SHA256:?catalog checksum is required}"
teacher_sha256="${POKEBOT_TEAL_MASK_TEACHER_SHA256:?teacher checksum is required}"
status="$output_root/status/window.json"
start_date="${POKEBOT_TEAL_MASK_START_DATE:-2026-06-26}"
end_date="${POKEBOT_TEAL_MASK_END_DATE:-2026-07-27}"
minimum_records="${POKEBOT_TEAL_MASK_MINIMUM_RECORDS:-1135}"
day_parallelism="${POKEBOT_TEAL_MASK_DAY_PARALLELISM:-4}"
workers_per_day="${POKEBOT_TEAL_MASK_WORKERS_PER_DAY:-3}"
max_in_flight_per_day="${POKEBOT_TEAL_MASK_MAX_IN_FLIGHT_PER_DAY:-6}"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$catalog_source" | awk '{print $1}')" = "$catalog_sha256"
test "$(sha256sum "$teacher_module" | awk '{print $1}')" = "$teacher_sha256"

python3 - "$archive_root" "$start_date" "$end_date" <<'PY'
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
start = date.fromisoformat(sys.argv[2])
end = date.fromisoformat(sys.argv[3])
expected_manifest_only_ids = {
    "2026-07-24": {"87841523"},
}
expected_archive_sha256 = {
    "2026-07-24": "68a5c1be539bef579f03b5de29b901a1fab1dc4904af78824fbf7666d73bc8ab",
    "2026-07-28": "067b71f93fb5ebc35b727117b5f61c30fa9881f4f7d90ce3de5c27be973573cd",
}
validated = []
for offset in range((end - start).days + 1):
    day = (start + timedelta(days=offset)).isoformat()
    archive = root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise SystemExit(f"missing public replay archive: {archive}")
    expected_checksum = expected_archive_sha256.get(day)
    if expected_checksum is not None:
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_checksum:
            raise SystemExit(f"audited public replay archive changed: {archive}")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if "manifest.csv" not in names:
            raise SystemExit(f"public replay archive has no manifest: {archive}")
        json_ids = [
            Path(name).stem
            for name in names
            if name.endswith(".json") and not name.endswith("/")
        ]
        manifest_rows = list(
            csv.DictReader(
                io.StringIO(source.read("manifest.csv").decode("utf-8-sig"))
            )
        )
    manifest_ids = [str(row.get("episode_id") or "") for row in manifest_rows]
    if (
        not json_ids
        or len(json_ids) != len(set(json_ids))
        or not manifest_ids
        or "" in manifest_ids
        or len(manifest_ids) != len(set(manifest_ids))
    ):
        raise SystemExit(f"public replay archive has duplicate/invalid IDs: {archive}")
    missing = set(manifest_ids) - set(json_ids)
    orphaned = set(json_ids) - set(manifest_ids)
    if missing != expected_manifest_only_ids.get(day, set()) or orphaned:
        raise SystemExit(
            "public replay archive differs from its manifest: "
            f"day={day} missing={sorted(missing)} orphaned={sorted(orphaned)}"
        )
    validated.append(
        {
            "date": day,
            "json_replays": len(json_ids),
            "manifest_rows": len(manifest_ids),
            "manifest_only_episode_ids": sorted(missing),
        }
    )
print(
    json.dumps(
        {
            "status": "all_public_archives_match_manifests",
            "days": len(validated),
            "json_replays": sum(row["json_replays"] for row in validated),
            "manifest_rows": sum(row["manifest_rows"] for row in validated),
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
  --required-archetype teal-mask-ogerpon-ex \
  --current-deck-guide teal-mask-ogerpon-ex \
  --authoritative-deck-catalog "$catalog" \
  --authoritative-only-archetype teal-mask-ogerpon-ex \
  --day-parallelism "$day_parallelism" \
  --workers-per-day "$workers_per_day" \
  --max-in-flight-per-day "$max_in_flight_per_day" \
  --max-context 320 \
  --memory-floor-gib 16 \
  --min-records 0

python3 scripts/finalize_current_deck_guide_window.py \
  --status "$status" \
  --out-dir "$output_root" \
  --start "$start_date" \
  --end "$end_date" \
  --specialist-id teal-mask-ogerpon-ex \
  --guide-version teal-mask-ogerpon-ex-slop-box-north-star-v3 \
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
expected_records = int(catalog.get("observed_acting_seat_games") or 0)
recent = {
    str(row.get("date")): int(row.get("guide_rows") or 0)
    for row in ready.get("daily_shards") or []
    if expected_by_day.get(str(row.get("date")), 0) > 0
}
expected_nonzero_days = {
    day for day, records in expected_by_day.items() if records > 0
}
if (
    ready.get("status") != "ready"
    or ready.get("specialist_id") != "teal-mask-ogerpon-ex"
    or int(ready.get("records") or 0) != expected_records
    or actual_by_day != expected_by_day
    or int(ready.get("guide_rows") or 0) <= 0
    or set(recent) != expected_nonzero_days
    or any(value <= 0 for value in recent.values())
    or (ready.get("source_policy") or {}).get("mode")
    != "public_full_history_exact_deck_identity"
):
    raise SystemExit("sealed Teal Mask full-history corpus failed its final contract")
print(json.dumps(ready, sort_keys=True))
PY
