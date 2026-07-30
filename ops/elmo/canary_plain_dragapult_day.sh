#!/usr/bin/env bash
# Prove one nonempty exact-identity plain-Dragapult day before the full build.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
catalog="${POKEBOT_DRAGAPULT_CATALOG_SOURCE:-$source_root/data/training_mixes/dragapult-public-plain-full33.v1.json}"
catalog_sha256="${POKEBOT_DRAGAPULT_CATALOG_SHA256:?catalog checksum is required}"
source_date="${POKEBOT_DRAGAPULT_CANARY_DATE:?nonempty canary date is required}"
output_root="${POKEBOT_DRAGAPULT_CANARY_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/dragapult-guide-corpus-plain-canary-v1-$source_date}"
status="$output_root/status/window.json"

test -f "$cg_runtime/cg/__init__.py"
test -s "$catalog"
test "$(sha256sum "$catalog" | awk '{print $1}')" = "$catalog_sha256"

expected_records="$(
  python3 - "$catalog" "$source_date" <<'PY'
import json
from pathlib import Path
import sys

catalog = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
day = sys.argv[2]
window = dict(catalog.get("source_window") or {})
by_day = dict(catalog.get("observed_by_day") or {})
expected = int(by_day.get(day) or 0)
if (
    catalog.get("schema") != "poke_bot.public_deck_archetype_catalog/v1"
    or catalog.get("specialist_id") != "dragapult"
    or int(window.get("days") or 0) != 33
    or day not in by_day
    or expected <= 0
):
    raise SystemExit("plain Dragapult canary date is not a nonempty catalog day")
print(expected)
PY
)"

mkdir -p "$output_root/status"
export CG_LIB_PATH="$cg_runtime"
cd "$source_root"
python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$source_date" \
  --end "$source_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype dragapult \
  --current-deck-guide dragapult \
  --authoritative-deck-catalog "$catalog" \
  --authoritative-only-archetype dragapult \
  --day-parallelism 1 \
  --workers-per-day 2 \
  --max-in-flight-per-day 4 \
  --max-context 320 \
  --memory-floor-gib 20 \
  --min-records "$expected_records"

python3 - "$status" "$output_root" "$source_date" "$expected_records" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

status_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
day = sys.argv[3]
expected = int(sys.argv[4])
status = json.loads(status_path.read_text(encoding="utf-8"))
receipt_path = output_root / f"dragapult-{day}.features.receipt.json"
feature_path = output_root / f"dragapult-{day}.features"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
selection = dict(receipt.get("selection") or {})
stats = dict(receipt.get("stats") or {})
coverage = dict(stats.get("target_coverage") or {})
output = dict(receipt.get("output") or {})
digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
if (
    status.get("state") != "complete"
    or [row.get("date") for row in status.get("completed") or []] != [day]
    or receipt.get("source_date") != day
    or selection.get("acting_seat_archetype") != "dragapult"
    or selection.get("current_deck_guide") != "dragapult"
    or int(stats.get("records_kept") or 0) != expected
    or int(coverage.get("guide_rows") or 0) <= 0
    or output.get("sha256") != "sha256:" + digest
):
    raise SystemExit("plain Dragapult exact-identity one-day canary failed")
print(
    json.dumps(
        {
            "status": "passed",
            "date": day,
            "records": expected,
            "guide_rows": int(coverage["guide_rows"]),
            "feature_sha256": output["sha256"],
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
