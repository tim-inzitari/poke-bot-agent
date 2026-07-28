#!/usr/bin/env bash
# Launch one isolated, checksum-producing current-deck-guide window on Elmo.
set -euo pipefail

if (( $# < 3 || $# > 3 )); then
  echo "usage: $0 SPECIALIST_ID START_DATE END_DATE" >&2
  exit 2
fi

specialist_id="$1"
start_date="$2"
end_date="$3"

[[ "$specialist_id" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "invalid specialist ID: $specialist_id" >&2
  exit 2
}
python3 - "$start_date" "$end_date" <<'PY'
from datetime import date
import sys
start = date.fromisoformat(sys.argv[1])
end = date.fromisoformat(sys.argv[2])
if end < start or (end - start).days != 19:
    raise SystemExit("guide window must be exactly 20 consecutive calendar days")
PY

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
roster="${POKEBOT_MATCHUP_ROSTER:-$source_root/state/matchup_adapter_roster.json}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
output_parent="${POKEBOT_GUIDE_OUTPUT_ROOT:-/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/daily/current-deck-guides-v1}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
day_parallelism="${POKEBOT_GUIDE_DAY_PARALLELISM:-4}"
workers_per_day="${POKEBOT_GUIDE_WORKERS_PER_DAY:-3}"
memory_max="${POKEBOT_GUIDE_MEMORY_MAX:-25769803776}"
output_root="$output_parent/$specialist_id"
status="$output_root/status/window.json"
unit="pokebot-${specialist_id}-guide-window-v1.service"

test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$roster"
test -f "$cg_runtime/cg/__init__.py"
[[ "$day_parallelism" =~ ^[1-9][0-9]*$ ]]
[[ "$workers_per_day" =~ ^[1-9][0-9]*$ ]]
[[ "$memory_max" =~ ^[1-9][0-9]*$ ]]

if systemctl is-active --quiet "$unit"; then
  echo "$unit is already active"
  exit 0
fi

mkdir -p "$output_root/status"
chmod 2770 "$output_root" "$output_root/status"
sudo -n systemctl reset-failed "$unit" 2>/dev/null || true

logical_alias_args=()
while IFS= read -r alias; do
  [[ -n "$alias" ]] || continue
  logical_alias_args+=(--logical-alias "$alias")
done < <(
  python3 - "$roster" "$specialist_id" <<'PY'
import json
import sys

roster = json.load(open(sys.argv[1], encoding="utf-8"))
specialist_id = sys.argv[2].strip().casefold()
for source, target in sorted((roster.get("logical_aliases") or {}).items()):
    if str(target).strip().casefold() == specialist_id:
        print(f"{str(source).strip().casefold()}={specialist_id}")
PY
)

sudo -n systemd-run \
  --unit="$unit" \
  --description="Materialize $specialist_id 20-day current deck guide window" \
  --setenv="CG_LIB_PATH=$cg_runtime" \
  --property=User=admin \
  --property="WorkingDirectory=$source_root" \
  --property=Nice=10 \
  --property=CPUWeight=10 \
  --property=IOWeight=10 \
  --property="MemoryMax=$memory_max" \
  /usr/bin/python3 \
  scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype "$specialist_id" \
  --current-deck-guide "$specialist_id" \
  "${logical_alias_args[@]}" \
  --day-parallelism "$day_parallelism" \
  --workers-per-day "$workers_per_day" \
  --max-in-flight-per-day 6 \
  --max-context 320 \
  --memory-floor-gib 16 \
  --min-records 0

systemctl is-active "$unit"
systemctl show "$unit" -p MainPID -p NRestarts -p Environment -p MemoryMax
