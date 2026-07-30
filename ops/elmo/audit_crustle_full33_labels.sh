#!/usr/bin/env bash
# Audit every sealed family-identity Crustle guide shard.
set -euo pipefail

source_root="${POKEBOT_CRUSTLE_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
corpus_root="${POKEBOT_CRUSTLE_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/crustle-guide-corpus-family-full33-v1}"
ready="$corpus_root/CURRENT_DECK_GUIDE_CORPUS_READY.json"
output="$corpus_root/CRUSTLE_GUIDE_LABEL_AUDIT_FULL33.json"
validated="$corpus_root/CRUSTLE_GUIDE_CORPUS_VALIDATED.json"

test -s "$ready"
test -s "$source_root/scripts/audit_current_deck_guide_labels.py"

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
    or ready.get("specialist_id") != "crustle"
    or ready.get("guide_version") != "crustle-north-star-v1"
    or int(ready.get("days") or 0) != 33
    or int(ready.get("records") or 0) <= 0
    or len(rows) != 33
    or len(set(dates)) != 33
    or any(not date for date in dates)
):
    raise SystemExit("Crustle full33 ready receipt is invalid")
for row, date in sorted(zip(rows, dates), key=lambda value: value[1]):
    shard = root / f"crustle-{date}.features"
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
  audit_args+=(--shard "$corpus_root/crustle-$source_date.features")
done

cd "$source_root"
python3 scripts/audit_current_deck_guide_labels.py \
  "${audit_args[@]}" \
  --specialist-id crustle \
  --guide-version crustle-north-star-v1 \
  --corpus-ready-receipt "$ready" \
  --out "$output"

python3 - "$output" "$ready" "$validated" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

audit_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
validated_path = Path(sys.argv[3])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
ready = json.loads(ready_path.read_text(encoding="utf-8"))
metrics = dict(audit.get("metrics") or {})
if (
    audit.get("status")
    != "passed_structural_and_observational_validation"
    or audit.get("specialist_id") != "crustle"
    or audit.get("guide_version") != "crustle-north-star-v1"
    or int(audit.get("shard_count") or 0) != 33
    or int(metrics.get("records") or 0)
    != int(ready.get("records") or 0)
    or int(metrics.get("decisions") or 0) <= 0
    or int(metrics.get("policy_stages") or 0) <= 0
    or int(metrics.get("labeled_stages") or 0) <= 0
    or int(metrics.get("abstained_stages") or 0) <= 0
    or int(metrics.get("invalid_target_indices") or 0) != 0
    or int(metrics.get("invalid_confidences") or 0) != 0
):
    raise SystemExit("Crustle full33 guide-label audit failed")
def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

identity = {
    "schema": "poke_bot.crustle_guide_corpus_validation/v1",
    "status": "ready_checksum_validated",
    "specialist_id": "crustle",
    "guide_version": "crustle-north-star-v1",
    "guide_ready_receipt_sha256": sha256(ready_path),
    "label_audit_sha256": sha256(audit_path),
    "records": int(ready["records"]),
    "decisions": int(ready["decisions"]),
    "guide_rows": int(ready["guide_rows"]),
    "active_training_modified": False,
}
if validated_path.exists():
    existing = json.loads(validated_path.read_text(encoding="utf-8"))
    if any(existing.get(key) != value for key, value in identity.items()):
        raise SystemExit("immutable Crustle corpus validation receipt differs")
else:
    receipt = {
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = validated_path.with_name(
        f".{validated_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, validated_path)
print(json.dumps(audit, sort_keys=True))
PY
