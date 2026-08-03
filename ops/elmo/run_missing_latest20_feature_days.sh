#!/usr/bin/env bash
# Materialize only missing canonical latest-20 daily feature shards on Elmo.
set -euo pipefail

SOURCE="${POKEBOT_SOURCE:-/home/admin/pokebot-expert-src-v6-strategic}"
RAW="${POKEBOT_RAW:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
ARCHIVE_RECEIPT="${POKEBOT_ARCHIVE_RECEIPT:-/mnt/Main/main/poke-bot-agent/archive/expert-latest20/current.json}"
OUTPUT="${POKEBOT_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/daily/roster18-v5}"
CG_RUNTIME_SOURCE="${POKEBOT_CG_RUNTIME_SOURCE:-/mnt/Main/main/poke-bot-agent/deployments/persistent-workers-20260720-v1/kaggle/input/cg-lib}"
IMAGE="${POKEBOT_IMAGE:-poke-bot-truenas-worker:matchup-v33-runtime}"
NATIVE_IMAGE="${POKEBOT_NATIVE_IMAGE:-pokebot-native-expert-featurizer:x86_64}"
NAME="${POKEBOT_CONTAINER_NAME:-pokebot-expert-latest20-missing-days}"
STATUS_ROOT="$OUTPUT/status"
LOCK="${POKEBOT_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/${NAME}.lock}"
CONTAINER_CPUS="${POKEBOT_CPUS:-24}"
CONTAINER_MEMORY="${POKEBOT_MEMORY:-56g}"
CPU_SHARES="${POKEBOT_CPU_SHARES:-256}"
DAY_PARALLELISM="${POKEBOT_DAY_PARALLELISM:-8}"
NATIVE_DAY_PARALLELISM="${POKEBOT_NATIVE_DAY_PARALLELISM:-$DAY_PARALLELISM}"
WORKERS_PER_DAY="${POKEBOT_WORKERS_PER_DAY:-3}"
MAX_IN_FLIGHT_PER_DAY="${POKEBOT_MAX_IN_FLIGHT_PER_DAY:-6}"
NATIVE_CPUS_PER_DAY="${POKEBOT_NATIVE_CPUS_PER_DAY:-2}"
READY_RECEIPT_NAME="${POKEBOT_READY_RECEIPT_NAME:-MISSING_DAYS_READY.json}"
REQUIRED_DATASET_SCHEMA="${POKEBOT_REQUIRED_DATASET_SCHEMA:-6}"
REQUIRED_EXPANDED_TARGET_SCHEMA="${POKEBOT_REQUIRED_EXPANDED_TARGET_SCHEMA:-poke_bot.expanded_strategic_targets/v2}"
REQUIRED_EXPANDED_TARGET_DIGEST="${POKEBOT_REQUIRED_EXPANDED_TARGET_DIGEST:-sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f}"

[[ "$READY_RECEIPT_NAME" != */* && "$READY_RECEIPT_NAME" == *.json ]] || {
  echo "feature ready receipt must be a JSON basename" >&2
  exit 2
}

for value in \
  "$DAY_PARALLELISM" \
  "$NATIVE_DAY_PARALLELISM" \
  "$WORKERS_PER_DAY" \
  "$MAX_IN_FLIGHT_PER_DAY" \
  "$NATIVE_CPUS_PER_DAY" \
  "$CPU_SHARES" \
  "$REQUIRED_DATASET_SCHEMA"
do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "feature-pipeline concurrency values must be positive integers" >&2
    exit 2
  }
done

# A second launcher must reuse the first launch, never race its per-day
# temporary receipts or create a duplicate feature container.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "latest-20 missing-day feature launch already in progress"
  exit 0
fi

if [[ -n "${POKEBOT_DAYS_OVERRIDE:-}" ]]; then
  read -r -a days <<<"$POKEBOT_DAYS_OVERRIDE"
else
  days=(
    2026-07-04 2026-07-05 2026-07-06 2026-07-07
    2026-07-08 2026-07-09 2026-07-10 2026-07-21
    2026-07-22 2026-07-23
  )
fi

test -s "$ARCHIVE_RECEIPT"
test -s "$SOURCE/scripts/materialize_authoritative_archetype_window.py"
test -s "$SOURCE/state/matchup_adapter_roster.json"
test -f "$CG_RUNTIME_SOURCE/cg/__init__.py"
test -s "$CG_RUNTIME_SOURCE/cg/libcg.so"

# Fail before expensive native validation or featurization when the mounted
# source snapshot cannot produce the exact strategic shard contract expected
# by the finalizer and trainer.
sudo -n docker run --rm \
  -v "$SOURCE:/workspace:ro" \
  -e REQUIRED_DATASET_SCHEMA="$REQUIRED_DATASET_SCHEMA" \
  -e REQUIRED_EXPANDED_TARGET_SCHEMA="$REQUIRED_EXPANDED_TARGET_SCHEMA" \
  -e REQUIRED_EXPANDED_TARGET_DIGEST="$REQUIRED_EXPANDED_TARGET_DIGEST" \
  --entrypoint python \
  "$IMAGE" -c '
import os
import sys
sys.path.insert(0, "/workspace")
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
)
assert DATASET_CACHE_SCHEMA_VERSION == int(os.environ["REQUIRED_DATASET_SCHEMA"])
assert EXPANDED_STRATEGIC_SCHEMA == os.environ["REQUIRED_EXPANDED_TARGET_SCHEMA"]
assert EXPANDED_STRATEGIC_SCHEMA_DIGEST == os.environ["REQUIRED_EXPANDED_TARGET_DIGEST"]
'
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
    '.archives[]
      | select(.date == $day and .validated == true)
      | (.validated_episode_count // .episode_count)' \
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
  sudo -n docker run --rm --cpus "$NATIVE_CPUS_PER_DAY" --memory 4g \
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

# The native validator is the first stage for every missing day. Run the
# selected day-level validations concurrently; each receipt is independently
# reusable.
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
  if (( ${#native_pids[@]} >= NATIVE_DAY_PARALLELISM )); then
    for pid in "${native_pids[@]}"; do
      wait "$pid"
    done
    native_pids=()
  fi
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
  --cpus "$CONTAINER_CPUS" \
  --cpu-shares "$CPU_SHARES" \
  --memory "$CONTAINER_MEMORY" \
  --memory-swap "$CONTAINER_MEMORY" \
  --pids-limit 4096 \
  -e PYTHONUNBUFFERED=1 \
  -e POKEBOT_DAYS="$day_words" \
  -e POKEBOT_DAY_PARALLELISM="$DAY_PARALLELISM" \
  -e POKEBOT_WORKERS_PER_DAY="$WORKERS_PER_DAY" \
  -e POKEBOT_MAX_IN_FLIGHT_PER_DAY="$MAX_IN_FLIGHT_PER_DAY" \
  -e POKEBOT_READY_RECEIPT_NAME="$READY_RECEIPT_NAME" \
  -e CG_LIB_PATH=/workspace/kaggle/input/cg-lib \
  -v "$SOURCE:/workspace:ro" \
  -v "$CG_RUNTIME_SOURCE:/workspace/kaggle/input/cg-lib:ro" \
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
  # Container PIDs are reused after a managed restart. Any partial bearing a
  # PID from the previous container is non-resumable by design and can make a
  # fresh process falsely look like its own stale writer. No day worker has
  # started yet at this point, so these are exclusively abandoned temporaries.
  find /output -maxdepth 1 -type f \
    -name ".*all-recognized-$day*.partial.*" -delete
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
    --workers "$POKEBOT_WORKERS_PER_DAY" \
    --max-in-flight "$POKEBOT_MAX_IN_FLIGHT_PER_DAY" \
    --max-context 320 \
    --memory-floor-gib 2 --min-records 1 \
    >"/output/status/$day.log" 2>&1
}

pids=()
failed=0
for day in $POKEBOT_DAYS; do
  run_day "$day" &
  pids+=("$!")
  if (( ${#pids[@]} >= POKEBOT_DAY_PARALLELISM )); then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    pids=()
  fi
done
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
name=os.environ["POKEBOT_READY_RECEIPT_NAME"]
assert "/" not in name and name.endswith(".json")
tmp=root/f".{name}.tmp"
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
tmp.replace(root/name)
PY
'

echo "started $NAME"
echo "logs: sudo docker logs -f $NAME"
