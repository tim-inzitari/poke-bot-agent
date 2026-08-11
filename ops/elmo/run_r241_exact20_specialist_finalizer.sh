#!/usr/bin/env bash
# Build only the checksum-bound r241 Jul-22--Aug-10 specialist expert window.
#
# This launcher deliberately does not reuse the generic latest-20 finalizer:
# r241 has three immutable source ranges, an r236-owned edge range, and a
# different completion marker for the Aug-03--Aug-10 materialization.  It is
# safe to start before the last range lands because the isolated container
# waits without opening any feature shard; malformed or drifting receipts fail
# closed before the finalizer is invoked.
set -euo pipefail

readonly R241_CANDIDATE_ID="alakazam-new-list-direct-policy-r241"
readonly R241_ARCHIVE_RECEIPT_SHA256="sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
readonly R241_EXPANDED_TARGET_SCHEMA="poke_bot.expanded_strategic_targets/v2"
readonly R241_EXPANDED_TARGET_DIGEST="sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"
readonly R241_DATASET_SCHEMA="6"
readonly R241_FEATURE_SCHEMA="5"
readonly R241_WINDOW_START="2026-07-22"
readonly R241_WINDOW_END="2026-08-10"
readonly R241_WINDOW_DAYS="20"

# These paths are deliberately fixed.  In particular, the two edge days have
# their own r241/r236-owned feature root and may not fall back to older v5 or
# generic v6 materializations.
readonly SOURCE="/home/admin/pokebot-expert-src-v6-strategic"
readonly ARCHIVE_RECEIPT="/mnt/Main/main/poke-bot-agent/archive/expert-r241-20260722-20260810/current.json"
readonly EARLY_FEATURES="/mnt/Main/main/poke-bot-agent/archive/expert-r241-derived/daily/roster18-v6-strategic-20260722-23"
readonly MID_FEATURES="/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/daily/roster18-v6-strategic-2026-07-14_2026-08-02"
readonly NEW_FEATURES="/mnt/Main/main/poke-bot-agent/archive/expert-r241-derived/daily/roster18-v6-strategic"
readonly OUTPUT="/mnt/Main/main/poke-bot-agent/archive/expert-r241-derived/windows/2026-07-22_2026-08-10/roster18-v6-strategic"
readonly IMAGE="poke-bot-truenas-worker:matchup-v33-runtime"
readonly NAME="pokebot-r241-exact20-specialist-finalizer-a2"
readonly SHARED_GID="950"
readonly READY_WAIT_SECONDS="${POKEBOT_R241_FINALIZER_READY_WAIT_SECONDS:-86400}"
readonly READY_POLL_SECONDS="30"
readonly LOCK="${POKEBOT_R241_FINALIZER_LOCK:-/tmp/${NAME}.lock}"

# A caller must not turn this r241-specific launcher back into a generic
# latest-20 tool by redirecting its sealed identity inputs or output root.
for override in \
  POKEBOT_SOURCE \
  POKEBOT_MAIN \
  POKEBOT_ARCHIVE_RECEIPT \
  POKEBOT_CURRENT_FEATURES \
  POKEBOT_EXISTING_FEATURES \
  POKEBOT_OUTPUT \
  POKEBOT_IMAGE \
  POKEBOT_CONTAINER_NAME
do
  if [[ -n "${!override:-}" ]]; then
    echo "r241 exact20 finalizer forbids $override overrides" >&2
    exit 2
  fi
done

[[ "$READY_WAIT_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "POKEBOT_R241_FINALIZER_READY_WAIT_SECONDS must be a nonnegative integer" >&2
  exit 2
}
[[ "$READY_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "r241 exact20 finalizer poll interval is invalid" >&2
  exit 2
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "r241 exact20 specialist finalizer launch already in progress"
  exit 0
fi

test -s "$SOURCE/scripts/finalize_latest20_specialist_corpora.py"
test -s "$SOURCE/state/matchup_adapter_roster.json"
test -s "$ARCHIVE_RECEIPT"
[[ ! -L "$SOURCE" && ! -L "$ARCHIVE_RECEIPT" ]] || {
  echo "r241 exact20 source or archive receipt may not be a symlink" >&2
  exit 1
}
for root in "$EARLY_FEATURES" "$MID_FEATURES" "$NEW_FEATURES"; do
  [[ -d "$root" && ! -L "$root" ]] || {
    echo "r241 exact20 feature root is absent or unsafe: $root" >&2
    exit 1
  }
done

if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(sudo -n docker inspect -f '{{.State.Status}}' "$NAME")"
  echo "r241 exact20 finalizer container already exists ($state): $NAME" >&2
  echo "preserving its logs and receipt; inspect it before any explicit rerun" >&2
  exit 1
fi

if [[ -e "$OUTPUT/LATEST20_SPECIALIST_CORPORA_READY.json" ]]; then
  echo "r241 exact20 specialist corpora already has a final receipt: $OUTPUT" >&2
  echo "refusing to overwrite or relaunch without an explicit receipt audit" >&2
  exit 1
fi
[[ ! -L "$OUTPUT" ]] || {
  echo "r241 exact20 output root may not be a symlink: $OUTPUT" >&2
  exit 1
}

# The only host path the container may mutate is this new r241 output root.
sudo -n mkdir -p "$OUTPUT"

read -r -d '' CONTAINER_COMMAND <<'CONTAINER_SCRIPT' || true
set -euo pipefail
cd /workspace

wait_for_marker() {
  local marker="$1"
  local label="$2"
  local waited=0
  until [[ -s "$marker" ]]; do
    if (( waited >= R241_READY_WAIT_SECONDS )); then
      echo "timed out waiting for $label marker: $marker" >&2
      exit 75
    fi
    echo "waiting for $label marker: $marker" >&2
    sleep "$R241_READY_POLL_SECONDS"
    waited=$((waited + R241_READY_POLL_SECONDS))
  done
}

write_launch_receipt() {
  python3 - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

output = Path("/output")
target = output / "R241_EXACT20_SPECIALIST_FINALIZER_LAUNCH.json"
payload = {
    "schema": "poke_bot.alakazam_new_list_direct_r241_exact20_specialist_finalizer_launch/v1",
    "status": "launched_waiting_for_exact_markers",
    "candidate_id": os.environ["R241_CANDIDATE_ID"],
    "container_name": os.environ["R241_CONTAINER_NAME"],
    "archive_receipt_host_path": os.environ["R241_HOST_ARCHIVE_RECEIPT"],
    "archive_receipt_sha256": os.environ["R241_ARCHIVE_RECEIPT_SHA256"],
    "window": {
        "start": os.environ["R241_WINDOW_START"],
        "end": os.environ["R241_WINDOW_END"],
        "days": int(os.environ["R241_WINDOW_DAYS"]),
    },
    "source": os.environ["R241_HOST_SOURCE"],
    "candidate_roots": {
        "official_r236_edge_days": os.environ["R241_HOST_OFFICIAL_R236_EDGE_DAYS"],
        "schema6_mid_days": os.environ["R241_HOST_SCHEMA6_MID_DAYS"],
        "r241_tail_days": os.environ["R241_HOST_R241_TAIL_DAYS"],
    },
    "markers": {
        "official_r236_edge_days": "MISSING_DAYS_READY.json",
        "schema6_mid_days": "MISSING_DAYS_READY.json",
        "r241_tail_days": "R241_MISSING_DAYS_READY.json",
    },
    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
}

if target.exists() or target.is_symlink():
    if target.is_symlink():
        raise RuntimeError("r241 launch receipt path is a symlink")
    existing = json.loads(target.read_text(encoding="utf-8"))
    for key in (
        "schema", "candidate_id", "container_name", "archive_receipt_host_path", "archive_receipt_sha256",
        "window", "source", "candidate_roots", "markers",
    ):
        if existing.get(key) != payload.get(key):
            raise RuntimeError("existing r241 launch receipt identity changed")
else:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=output
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
PY
}

write_launch_receipt
wait_for_marker /input/early/MISSING_DAYS_READY.json "official-r236 Jul22-Jul23"
wait_for_marker /input/mid/MISSING_DAYS_READY.json "schema6 Jul24-Aug02"
wait_for_marker /input/new/R241_MISSING_DAYS_READY.json "r241 Aug03-Aug10"

python3 - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION
from poke_bot.features import FEATURE_SCHEMA_VERSION
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    EXPANDED_HEAD_IDS,
    merge_expanded_strategic_coverages,
)
from scripts.finalize_latest20_specialist_corpora import (
    feature_identity,
    read_json,
    select_sources,
)

EXPECTED_DATES = [
    "2026-07-22", "2026-07-23",
    "2026-07-24", "2026-07-25", "2026-07-26", "2026-07-27",
    "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31",
    "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04",
    "2026-08-05", "2026-08-06", "2026-08-07", "2026-08-08",
    "2026-08-09", "2026-08-10",
]
EARLY_DATES = EXPECTED_DATES[:2]
MID_DATES = EXPECTED_DATES[2:12]
TAIL_DATES = EXPECTED_DATES[12:]
ROOTS = {
    "official_r236_edge_days": Path("/input/early").resolve(),
    "schema6_mid_days": Path("/input/mid").resolve(),
    "r241_tail_days": Path("/input/new").resolve(),
}
MARKERS = {
    "official_r236_edge_days": (ROOTS["official_r236_edge_days"] / "MISSING_DAYS_READY.json", EARLY_DATES),
    "schema6_mid_days": (ROOTS["schema6_mid_days"] / "MISSING_DAYS_READY.json", MID_DATES),
    "r241_tail_days": (ROOTS["r241_tail_days"] / "R241_MISSING_DAYS_READY.json", TAIL_DATES),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ready_marker(path: Path, dates: list[str], *, label: str) -> dict:
    require(path.is_file() and not path.is_symlink(), f"{label} ready marker is absent or unsafe")
    value = read_json(path)
    require(value.get("schema") == "poke_bot.expert_missing_daily_features/v1", f"{label} marker schema changed")
    require(value.get("status") == "ready", f"{label} marker is not ready")
    require(list(value.get("days") or ()) == dates, f"{label} marker dates changed")
    completed = list(value.get("completed") or ())
    require([str(row.get("date") or "") for row in completed] == dates, f"{label} completed dates changed")
    return value


require(DATASET_CACHE_SCHEMA_VERSION == int(os.environ["R241_DATASET_SCHEMA"]), "source dataset schema is not r241 schema6")
require(FEATURE_SCHEMA_VERSION == int(os.environ["R241_FEATURE_SCHEMA"]), "source feature schema is not r241 schema5")
require(EXPANDED_STRATEGIC_SCHEMA == os.environ["R241_EXPANDED_TARGET_SCHEMA"], "expanded target schema changed")
require(EXPANDED_STRATEGIC_SCHEMA_DIGEST == os.environ["R241_EXPANDED_TARGET_DIGEST"], "expanded target digest changed")

marker_rows = {
    label: ready_marker(path, dates, label=label)
    for label, (path, dates) in MARKERS.items()
}
receipt_path = Path("/input/archive/current.json")
require(receipt_path.is_file() and not receipt_path.is_symlink(), "r241 archive receipt is absent or unsafe")
require(sha256(receipt_path) == os.environ["R241_ARCHIVE_RECEIPT_SHA256"], "r241 archive receipt digest changed")
receipt = read_json(receipt_path)
require(receipt.get("schema") == "poke_bot.expert_latest20_receipt/v1", "r241 archive receipt schema changed")
require(receipt.get("status") == "ready", "r241 archive receipt is not ready")
require(receipt.get("window_policy") == "exact_20_consecutive_calendar_days", "r241 archive receipt window policy changed")
require(receipt.get("window_start") == os.environ["R241_WINDOW_START"], "r241 archive receipt start changed")
require(receipt.get("window_end") == os.environ["R241_WINDOW_END"], "r241 archive receipt end changed")
require(int(receipt.get("days") or 0) == int(os.environ["R241_WINDOW_DAYS"]), "r241 archive receipt day count changed")
archives = list(receipt.get("archives") or ())
require([str(row.get("date") or "") for row in archives] == EXPECTED_DATES, "r241 archive receipt dates changed")
require(all(row.get("validated") is True and str(row.get("sha256") or "").startswith("sha256:") for row in archives), "r241 archive receipt validation changed")

selected = select_sources(receipt, [ROOTS["official_r236_edge_days"], ROOTS["schema6_mid_days"], ROOTS["r241_tail_days"]])
require([row["date"] for row in selected] == EXPECTED_DATES, "r241 selected feature dates changed")
expected_root_by_date = {
    **{day: ROOTS["official_r236_edge_days"] for day in EARLY_DATES},
    **{day: ROOTS["schema6_mid_days"] for day in MID_DATES},
    **{day: ROOTS["r241_tail_days"] for day in TAIL_DATES},
}
for row in selected:
    source = Path(str(row["source"])).resolve()
    require(source.parent == expected_root_by_date[row["date"]], f"{row['date']} selected a noncanonical root")
    identity = feature_identity(source)
    header = identity["header"]
    metadata = identity["metadata"]
    require(int(header.get("dataset_schema") or -1) == DATASET_CACHE_SCHEMA_VERSION, f"{row['date']} dataset schema drifted")
    require(int(header.get("feature_schema") or -1) == FEATURE_SCHEMA_VERSION, f"{row['date']} feature schema drifted")
    require(int(metadata.get("dataset_schema") or -1) == DATASET_CACHE_SCHEMA_VERSION, f"{row['date']} sidecar dataset schema drifted")
    require(int(metadata.get("feature_schema") or -1) == FEATURE_SCHEMA_VERSION, f"{row['date']} sidecar feature schema drifted")
    stats = dict(metadata.get("stats") or {})
    expanded = stats.get("expanded_strategic_targets")
    coverage = merge_expanded_strategic_coverages((expanded,))
    require(coverage.get("schema") == EXPANDED_STRATEGIC_SCHEMA, f"{row['date']} expanded schema drifted")
    require(coverage.get("digest") == EXPANDED_STRATEGIC_SCHEMA_DIGEST, f"{row['date']} expanded digest drifted")
    require(set(coverage.get("head_coverage") or ()) == set(EXPANDED_HEAD_IDS), f"{row['date']} expanded head inventory drifted")
    require(int(coverage.get("decisions") or -1) == int(stats.get("decisions_kept") or -2), f"{row['date']} expanded decisions drifted")

roster_path = Path("/workspace/state/matchup_adapter_roster.json")
roster = read_json(roster_path)
expert_ids = list(roster.get("expert_ids") or ())
require(roster.get("schema") == "poke_bot.matchup_adapter_roster/v1", "r241 roster schema changed")
require(int(roster.get("required_specialist_count") or 0) == 18, "r241 roster cardinality changed")
require(len(expert_ids) == len(set(expert_ids)) == 18, "r241 roster identities changed")

output = Path("/output")
target = output / "R241_EXACT20_SPECIALIST_FINALIZER_PREFLIGHT.json"
payload = {
    "schema": "poke_bot.alakazam_new_list_direct_r241_exact20_specialist_finalizer_preflight/v1",
    "status": "ready",
    "candidate_id": os.environ["R241_CANDIDATE_ID"],
    "archive_receipt": {
        "host_path": os.environ["R241_HOST_ARCHIVE_RECEIPT"],
        "container_path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "window_start": receipt["window_start"],
        "window_end": receipt["window_end"],
        "days": int(receipt["days"]),
    },
    "source": {
        "host_path": os.environ["R241_HOST_SOURCE"],
        "finalizer_sha256": sha256(Path("/workspace/scripts/finalize_latest20_specialist_corpora.py")),
        "roster_sha256": sha256(roster_path),
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": FEATURE_SCHEMA_VERSION,
    },
    "expanded_strategic_targets": {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "head_ids": list(EXPANDED_HEAD_IDS),
    },
    "ready_markers": {
        label: {
            "host_root": os.environ[f"R241_HOST_{label.upper()}"],
            "container_path": str(MARKERS[label][0]),
            "sha256": sha256(MARKERS[label][0]),
            "days": list(marker_rows[label]["days"]),
        }
        for label in MARKERS
    },
    "selected_sources": [
        {
            **row,
            "canonical_host_root": os.environ[
                "R241_HOST_OFFICIAL_R236_EDGE_DAYS"
                if row["date"] in EARLY_DATES
                else "R241_HOST_SCHEMA6_MID_DAYS"
                if row["date"] in MID_DATES
                else "R241_HOST_R241_TAIL_DAYS"
            ],
        }
        for row in selected
    ],
    "roster": {
        "expert_ids": expert_ids,
        "required_specialist_count": 18,
    },
    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
}

if target.exists() or target.is_symlink():
    if target.is_symlink():
        raise RuntimeError("r241 preflight receipt path is a symlink")
    existing = read_json(target)
    for key in (
        "schema", "status", "candidate_id", "archive_receipt", "source",
        "expanded_strategic_targets", "ready_markers", "selected_sources", "roster",
    ):
        if existing.get(key) != payload.get(key):
            raise RuntimeError("existing r241 preflight receipt identity changed")
else:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=output
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
PY

python -u scripts/finalize_latest20_specialist_corpora.py \
  --archive-receipt /input/archive/current.json \
  --candidate-root /input/early \
  --candidate-root /input/mid \
  --candidate-root /input/new \
  --output-root /output \
  --roster /workspace/state/matchup_adapter_roster.json \
  --source-repo /workspace \
  --minimum-decisions 1 \
  --publish-reader admin

chgrp -R "$R241_SHARED_GID" /output
chmod -R g+rX /output
CONTAINER_SCRIPT

archive_parent="$(dirname "$ARCHIVE_RECEIPT")"
sudo -n docker run -d \
  --name "$NAME" \
  --restart on-failure:5 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
  --cpus 12 \
  --memory 32g \
  --memory-swap 32g \
  --pids-limit 2048 \
  --label "pokebot.candidate=$R241_CANDIDATE_ID" \
  --label "pokebot.role=exact20-specialist-finalizer" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -e R241_CANDIDATE_ID="$R241_CANDIDATE_ID" \
  -e R241_CONTAINER_NAME="$NAME" \
  -e R241_ARCHIVE_RECEIPT_SHA256="$R241_ARCHIVE_RECEIPT_SHA256" \
  -e R241_EXPANDED_TARGET_SCHEMA="$R241_EXPANDED_TARGET_SCHEMA" \
  -e R241_EXPANDED_TARGET_DIGEST="$R241_EXPANDED_TARGET_DIGEST" \
  -e R241_DATASET_SCHEMA="$R241_DATASET_SCHEMA" \
  -e R241_FEATURE_SCHEMA="$R241_FEATURE_SCHEMA" \
  -e R241_WINDOW_START="$R241_WINDOW_START" \
  -e R241_WINDOW_END="$R241_WINDOW_END" \
  -e R241_WINDOW_DAYS="$R241_WINDOW_DAYS" \
  -e R241_READY_WAIT_SECONDS="$READY_WAIT_SECONDS" \
  -e R241_READY_POLL_SECONDS="$READY_POLL_SECONDS" \
  -e R241_SHARED_GID="$SHARED_GID" \
  -e R241_HOST_SOURCE="$SOURCE" \
  -e R241_HOST_ARCHIVE_RECEIPT="$ARCHIVE_RECEIPT" \
  -e R241_HOST_OFFICIAL_R236_EDGE_DAYS="$EARLY_FEATURES" \
  -e R241_HOST_SCHEMA6_MID_DAYS="$MID_FEATURES" \
  -e R241_HOST_R241_TAIL_DAYS="$NEW_FEATURES" \
  -v "$SOURCE:/workspace:ro" \
  -v "$archive_parent:/input/archive:ro" \
  -v "$EARLY_FEATURES:/input/early:ro" \
  -v "$MID_FEATURES:/input/mid:ro" \
  -v "$NEW_FEATURES:/input/new:ro" \
  -v "$OUTPUT:/output" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc "$CONTAINER_COMMAND"

echo "started $NAME"
echo "launch receipt: $OUTPUT/R241_EXACT20_SPECIALIST_FINALIZER_LAUNCH.json"
echo "preflight receipt: $OUTPUT/R241_EXACT20_SPECIALIST_FINALIZER_PREFLIGHT.json"
echo "status: sudo docker logs -f $NAME"
