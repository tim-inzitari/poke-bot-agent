#!/usr/bin/env bash
# Materialize and seal Archaludon ex guide targets over its validated latest20
# identity corpus. This script prepares an inactive corpus only.
set -euo pipefail

mode="${1:-run}"
if [[ "$mode" != "run" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-teal-clean-src-v2}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
identity_root="${POKEBOT_ARCHALUDON_IDENTITY_ROOT:-/mnt/Main/main/poke-bot-agent/archive/archaludon-ex-latest20-2026-07-08_2026-07-27-v1}"
output_root="${POKEBOT_ARCHALUDON_GUIDE_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/archaludon-ex-guide-latest20-v2}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
teacher="$source_root/poke_bot/archaludon_ex_heuristics.py"
representatives="$source_root/data/training_mixes/specialist_representatives.v1.json"
identity_ready="$identity_root/ARCHALUDON_EX_LATEST20_CORPUS_READY.json"
identity_sources="$identity_root/SOURCE_ARCHIVES.json"
status="$output_root/status/window.json"
start_date="2026-07-08"
end_date="2026-07-27"
minimum_records="1458"

teacher_sha256="4ec28e4f26c281f7488b20c31bbe17a6e4320cb54cb7dc5e1e4722e839aceca3"
representatives_sha256="b6381debcd588e8ed4614d447d31fae2704a8bc7e7d1c0d9d33b9b773c493ed3"
identity_ready_sha256="53a9ce2d3bb9f8644128b692353ee1a6925217e0eef7bb2f11666af2154ecdb0"
identity_sources_sha256="1048f036c697f097b2d6b8824b7e5cb258f7884e73db7a211625987135ad26ce"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test "$(sha256sum "$teacher" | awk '{print $1}')" = "$teacher_sha256"
test "$(sha256sum "$representatives" | awk '{print $1}')" = "$representatives_sha256"
test "$(sha256sum "$identity_ready" | awk '{print $1}')" = "$identity_ready_sha256"
test "$(sha256sum "$identity_sources" | awk '{print $1}')" = "$identity_sources_sha256"

python3 - "$identity_ready" "$identity_sources" "$start_date" "$end_date" <<'PY'
import json
from pathlib import Path
import sys

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sources = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
start, end = sys.argv[3], sys.argv[4]
daily = list(ready.get("daily_receipts") or [])
source_rows = list(sources.get("archives") or [])
source_policy = dict(ready.get("source_policy") or {})
if (
    ready.get("schema") != "poke_bot.archaludon_ex_latest20_corpus/v1"
    or ready.get("specialist_id") != "archaludon-ex"
    or ready.get("status") != "ready_checksum_validated"
    or source_policy.get("date_start") != start
    or source_policy.get("date_end") != end
    or int(source_policy.get("days") or 0) != 20
    or int(ready.get("records") or 0) != 1458
    or int(ready.get("decisions") or 0) != 83980
    or int(ready.get("guide_rows") or 0) != 0
    or ready.get("non_guide_corpus") is not True
    or ready.get("immutable_staging_artifact") is not True
    or len(daily) != 20
    or sources.get("all_archive_checksums_validated") is not True
    or sources.get("all_archive_manifest_memberships_validated") is not True
    or len(source_rows) != 20
):
    raise SystemExit("validated Archaludon latest20 identity contract changed")
PY

if [[ "$mode" == "--check" ]]; then
  echo "archaludon-ex latest20 guide build preflight passed"
  exit 0
fi

mkdir -p "$output_root/status"
export CG_LIB_PATH="$cg_runtime"
cd "$source_root"
python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype archaludon-ex \
  --current-deck-guide archaludon-ex \
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
  --specialist-id archaludon-ex \
  --guide-version archaludon-ex-north-star-v1 \
  --minimum-records "$minimum_records"

python3 - \
  "$identity_ready" \
  "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" <<'PY'
import json
from pathlib import Path
import sys

identity = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
guide = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {
    str(row["date"]): {
        "records": int(row["records"]),
        "decisions": int(row["decisions"]),
    }
    for row in identity.get("daily_receipts") or []
}
actual = {
    str(row["date"]): {
        "records": int(row["records"]),
        "decisions": int(row["decisions"]),
        "guide_rows": int(row["guide_rows"]),
    }
    for row in guide.get("daily_shards") or []
}
actual_identity = {
    day: {"records": row["records"], "decisions": row["decisions"]}
    for day, row in actual.items()
}
nonzero_source_days = {
    day for day, row in expected.items() if row["records"] > 0
}
positive_guide_days = {
    day for day, row in actual.items() if row["guide_rows"] > 0
}
if (
    guide.get("status") != "ready"
    or guide.get("specialist_id") != "archaludon-ex"
    or guide.get("guide_version") != "archaludon-ex-north-star-v1"
    or int(guide.get("records") or 0) != 1458
    or int(guide.get("decisions") or 0) != 83980
    or int(guide.get("guide_rows") or 0) <= 0
    or actual_identity != expected
    or positive_guide_days != nonzero_source_days
    or guide.get("source_policy") is not None
    or guide.get("active_training_modified") is not False
):
    raise SystemExit("sealed Archaludon guide corpus failed its final contract")
print(json.dumps(guide, sort_keys=True))
PY
