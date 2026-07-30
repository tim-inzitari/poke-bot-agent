#!/usr/bin/env bash
# Audit every sealed exact-identity plain Dragapult guide shard.
set -euo pipefail

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
corpus_root="${POKEBOT_DRAGAPULT_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/dragapult-guide-corpus-plain-full33-v1}"
ready="$corpus_root/CURRENT_DECK_GUIDE_CORPUS_READY.json"
output="$corpus_root/DRAGAPULT_GUIDE_LABEL_AUDIT_FULL33.json"

test -s "$ready"
test -s "$source_root/scripts/audit_current_deck_guide_labels.py"

if [[ -e "$output" ]]; then
  python3 - "$output" "$ready" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

audit_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2]).resolve()
audit = json.loads(audit_path.read_text(encoding="utf-8"))
ready_digest = "sha256:" + hashlib.sha256(ready_path.read_bytes()).hexdigest()
if (
    audit.get("schema") != "poke_bot.current_deck_guide_label_audit/v1"
    or audit.get("status")
    != "passed_structural_and_observational_validation"
    or audit.get("specialist_id") != "dragapult"
    or audit.get("guide_version") != "dragapult-north-star-v1"
    or int(audit.get("shard_count") or 0) != 33
    or (audit.get("corpus_ready_receipt") or {}).get("path")
    != str(ready_path)
    or (audit.get("corpus_ready_receipt") or {}).get("sha256")
    != ready_digest
):
    raise SystemExit("existing full33 guide-label audit is not reusable")
print(json.dumps(audit, sort_keys=True))
PY
  exit 0
fi

mapfile -t source_dates < <(
  python3 - "$ready" "$corpus_root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

ready_path = Path(sys.argv[1])
root = Path(sys.argv[2])
ready = json.loads(ready_path.read_text(encoding="utf-8"))
rows = list(ready.get("daily_shards") or [])
dates = [str(row.get("date") or "") for row in rows]
if (
    ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
    or ready.get("status") != "ready"
    or ready.get("specialist_id") != "dragapult"
    or ready.get("guide_version") != "dragapult-north-star-v1"
    or int(ready.get("days") or 0) != 33
    or int(ready.get("records") or 0) != 1412
    or len(rows) != 33
    or len(set(dates)) != 33
    or any(not date for date in dates)
):
    raise SystemExit("plain Dragapult full33 ready receipt is invalid")
for row, date in sorted(zip(rows, dates), key=lambda value: value[1]):
    shard = root / f"dragapult-{date}.features"
    metadata = shard.with_suffix(shard.suffix + ".json")
    if not shard.is_file() or not metadata.is_file():
        raise SystemExit(f"missing sealed guide shard: {shard}")
    digest = "sha256:" + hashlib.sha256(shard.read_bytes()).hexdigest()
    if digest != row.get("sha256"):
        raise SystemExit(f"sealed guide shard checksum changed: {shard}")
    print(date)
PY
)

if [[ "${#source_dates[@]}" -ne 33 ]]; then
  echo "expected exactly 33 sealed source dates" >&2
  exit 1
fi

audit_args=()
for source_date in "${source_dates[@]}"; do
  audit_args+=(
    --shard "$corpus_root/dragapult-$source_date.features"
  )
done

cd "$source_root"
python3 scripts/audit_current_deck_guide_labels.py \
  "${audit_args[@]}" \
  --specialist-id dragapult \
  --guide-version dragapult-north-star-v1 \
  --corpus-ready-receipt "$ready" \
  --out "$output"

python3 - "$output" <<'PY'
import json
from pathlib import Path
import sys

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metrics = dict(audit.get("metrics") or {})
if (
    audit.get("status")
    != "passed_structural_and_observational_validation"
    or int(audit.get("shard_count") or 0) != 33
    or int(metrics.get("records") or 0) != 1412
    or int(metrics.get("decisions") or 0) <= 0
    or int(metrics.get("policy_stages") or 0) <= 0
    or int(metrics.get("labeled_stages") or 0) <= 0
    or int(metrics.get("abstained_stages") or 0) <= 0
    or int(metrics.get("invalid_target_indices") or 0) != 0
    or int(metrics.get("invalid_confidences") or 0) != 0
):
    raise SystemExit("plain Dragapult full33 guide-label audit failed")
print(json.dumps(audit, sort_keys=True))
PY
