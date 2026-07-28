#!/usr/bin/env bash
# Materialize and seal the owner-required full public Spidops guide corpus.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_root="${POKEBOT_SPIDOPS_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/spidops-guide-corpus-full-v1}"
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
from datetime import date, timedelta
from pathlib import Path
import sys

root = Path(sys.argv[1])
start = date.fromisoformat(sys.argv[2])
end = date.fromisoformat(sys.argv[3])
for offset in range((end - start).days + 1):
    day = (start + timedelta(days=offset)).isoformat()
    archive = root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise SystemExit(f"missing public replay archive: {archive}")
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
