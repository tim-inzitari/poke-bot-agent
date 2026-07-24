#!/usr/bin/env bash
# Materialize only missing canonical latest-20 daily feature shards on Elmo.
set -euo pipefail

SOURCE="${POKEBOT_SOURCE:-/home/admin/pokebot-expert-src-v41}"
RAW="${POKEBOT_RAW:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
ARCHIVE_RECEIPT="${POKEBOT_ARCHIVE_RECEIPT:-/mnt/Main/main/poke-bot-agent/archive/expert-latest20/current.json}"
OUTPUT="${POKEBOT_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/daily/roster18-v5}"
IMAGE="${POKEBOT_IMAGE:-poke-bot-truenas-worker:matchup-v33-runtime}"
NATIVE_IMAGE="${POKEBOT_NATIVE_IMAGE:-pokebot-native-expert-featurizer:x86_64}"
NAME="${POKEBOT_CONTAINER_NAME:-pokebot-expert-latest20-missing-days}"
STATUS_ROOT="$OUTPUT/status"
LOCK="${POKEBOT_LOCK:-/tmp/pokebot-expert-latest20-missing-days.lock}"

# A second launcher must reuse the first launch, never race its per-day
# temporary receipts or create a duplicate feature container.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "latest-20 missing-day feature launch already in progress"
  exit 0
fi

days=(
  2026-07-04 2026-07-05 2026-07-06 2026-07-07
  2026-07-08 2026-07-09 2026-07-10 2026-07-21
)

test -s "$ARCHIVE_RECEIPT"
test -s "$SOURCE/scripts/materialize_authoritative_archetype_window.py"
test -s "$SOURCE/state/matchup_adapter_roster.json"
sudo -n mkdir -p "$STATUS_ROOT/native"
sudo -n chown -R "$(id -u):$(id -g)" "$OUTPUT"

mapfile -t archetypes < <(
  python3 - "$SOURCE/state/matchup_adapter_roster.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema"] == "poke_bot.matchup_adapter_roster/v1"
assert value["required_specialist_count"] == 18
ids = value["expert_ids"]
assert len(ids) == len(set(ids)) == 18
print(*ids, sep="\n")
PY
)
[[ "${#archetypes[@]}" -eq 18 ]]

expected_for_day() {
  sudo -n jq -er --arg day "$1" \
    '.archives[] | select(.date == $day and .validated == true) | .episode_count' \
    "$ARCHIVE_RECEIPT"
}

native_validate_day() {
  local day="$1"
  local archive="$RAW/pokemon-tcg-ai-battle-episodes-$day.zip"
  local output="$STATUS_ROOT/native/$day.json"
  local temporary="$output.tmp"
  local expected
  expected="$(expected_for_day "$day")"
  test -s "$archive"
  sudo -n docker run --rm --cpus 3 --memory 4g \
    -v "$RAW:/data:ro" \
    "$NATIVE_IMAGE" \
    /usr/local/bin/replay_ingest_probe \
    "/data/${archive##*/}" 3 >"$temporary"
  python3 - "$temporary" "$expected" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
value = json.load(open(path, encoding="utf-8"))
assert value["schema"] == "pokebot-native-replay-ingest/v1"
assert value["episodes"] == expected
assert value["rejected"] == 0
PY
  mv "$temporary" "$output"
}

# The native validator is the first stage for every missing day. Run all eight
# day-level validations concurrently; each receipt is independently reusable.
native_pids=()
for day in "${days[@]}"; do
  if [[ -s "$STATUS_ROOT/native/$day.json" ]] \
      && python3 - "$STATUS_ROOT/native/$day.json" "$(expected_for_day "$day")" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(
    0 if value.get("episodes") == int(sys.argv[2])
    and value.get("rejected") == 0 else 1
)
PY
  then
    continue
  fi
  native_validate_day "$day" &
  native_pids+=("$!")
done
for pid in "${native_pids[@]}"; do
  wait "$pid"
done

if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(sudo -n docker inspect -f '{{.State.Status}}' "$NAME")"
  if [[ "$state" == "running" ]]; then
    echo "$NAME already running"
    exit 0
  fi
  sudo -n docker rm "$NAME" >/dev/null
fi

day_words="${days[*]}"
sudo -n docker run -d \
  --name "$NAME" \
  --restart on-failure:5 \
  --cpus 24 \
  --memory 56g \
  --memory-swap 56g \
  --pids-limit 4096 \
  -e PYTHONUNBUFFERED=1 \
  -e POKEBOT_DAYS="$day_words" \
  -v "$SOURCE:/workspace:ro" \
  -v "$RAW:/input:ro" \
  -v "$OUTPUT:/output" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -euo pipefail
cd /workspace
mapfile -t archetypes < <(
  python - <<"PY"
import json
value=json.load(open("state/matchup_adapter_roster.json"))
assert value["required_specialist_count"] == 18
print(*value["expert_ids"], sep="\n")
PY
)
additive=()
for value in "${archetypes[@]}"; do
  additive+=(--additive-archetype "$value")
done

run_day() {
  local day="$1"
  python -u scripts/materialize_authoritative_archetype_window.py \
    --start "$day" --end "$day" \
    --archetype "*" \
    --archive-dir /input \
    --out-dir /output \
    --status "/output/status/$day.json" \
    --mix /workspace/data/training_mixes/top_ladder.v1.json \
    --representatives /workspace/data/training_mixes/top_ladder_representatives.v1.json \
    --card-csv /workspace/cards/EN_Card_Data.csv \
    "${additive[@]}" \
    --workers 3 --max-in-flight 6 --max-context 320 \
    --memory-floor-gib 2 --min-records 1 \
    >"/output/status/$day.log" 2>&1
}

pids=()
for day in $POKEBOT_DAYS; do
  run_day "$day" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "one or more daily feature jobs failed" >&2
  exit 1
fi

python - <<PY
import json
import os
from datetime import datetime, timezone
from pathlib import Path
days=os.environ["POKEBOT_DAYS"].split()
root=Path("/output")
rows=[]
for day in days:
    status=json.load(open(root/"status"/f"{day}.json"))
    assert status["state"] == "complete"
    assert status["date_window"] == {"start": day, "end": day, "days": 1}
    completed=status["completed"]
    assert len(completed) == 1 and completed[0]["date"] == day
    rows.append(completed[0])
payload={
    "schema":"poke_bot.expert_missing_daily_features/v1",
    "status":"ready",
    "days":days,
    "completed":rows,
    "completed_at":datetime.now(timezone.utc).isoformat(),
    "corpus_location":"elmo_only",
}
tmp=root/".MISSING_DAYS_READY.json.tmp"
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
tmp.replace(root/"MISSING_DAYS_READY.json")
PY
'

echo "started $NAME"
echo "logs: sudo docker logs -f $NAME"
