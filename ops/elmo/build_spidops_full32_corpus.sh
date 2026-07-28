#!/usr/bin/env bash
# Materialize and seal the owner-required full public Spidops guide corpus.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_root="${POKEBOT_SPIDOPS_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/spidops-guide-corpus-full-v2}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
catalog_source="$source_root/data/training_mixes/team-rockets-spidops-public-full32.v1.json"
catalog="$output_root/PUBLIC_DECK_ARCHETYPE_CATALOG.json"
catalog_sha256="5ae34b29539394fbe0624700787713a144792443de67ea592e0b8780143bd4ff"
status="$output_root/status/window.json"
start_date="2026-06-26"
end_date="2026-07-27"
minimum_records="16639"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$catalog_source" | awk '{print $1}')" = "$catalog_sha256"

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
        observed_checksum = digest.hexdigest()
        if observed_checksum != expected_checksum:
            raise SystemExit(
                f"audited public replay archive changed: {archive}"
            )
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
                io.StringIO(
                    source.read("manifest.csv").decode("utf-8-sig")
                )
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
    json_id_set = set(json_ids)
    manifest_id_set = set(manifest_ids)
    missing = manifest_id_set - json_id_set
    orphaned = json_id_set - manifest_id_set
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
            "manifest_only_episode_ids": [
                episode_id
                for row in validated
                for episode_id in row["manifest_only_episode_ids"]
            ],
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
python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype team-rockets-spidops \
  --current-deck-guide team-rockets-spidops \
  --authoritative-deck-catalog "$catalog" \
  --day-parallelism 4 \
  --workers-per-day 3 \
  --max-in-flight-per-day 6 \
  --max-context 320 \
  --memory-floor-gib 16 \
  --min-records 0

python3 scripts/finalize_current_deck_guide_window.py \
  --status "$status" \
  --out-dir "$output_root" \
  --start "$start_date" \
  --end "$end_date" \
  --specialist-id team-rockets-spidops \
  --guide-version team-rockets-spidops-north-star-v1 \
  --minimum-records "$minimum_records" \
  --public-deck-catalog "$catalog"

python3 - "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" "$minimum_records" <<'PY'
import json
from pathlib import Path
import sys

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
minimum = int(sys.argv[2])
if (
    ready.get("status") != "ready"
    or ready.get("specialist_id") != "team-rockets-spidops"
    or int(ready.get("records") or 0) < minimum
    or (ready.get("source_policy") or {}).get("mode")
    != "public_full_history_exact_deck_identity"
):
    raise SystemExit("sealed Spidops full-history corpus failed its final contract")
print(json.dumps(ready, sort_keys=True))
PY
