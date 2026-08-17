#!/usr/bin/env bash
set -euo pipefail

source_root="/dev/shm/alakazam-refeature-2332383-1"
expected_source="/dev/shm/alakazam-refeature-2332383-1"
archive_root="/mnt/Main/main/poke-bot-agent/archive/alakazam-rule-derivative"
batch_id="r11-elmo-tmpfs-recovery-2332383-20260813T142000Z-d74152bc"
final="${archive_root}/batches/${batch_id}"
archive_receipt="${archive_root}/receipts/${batch_id}.json"
reclaim_receipt="${archive_root}/receipts/${batch_id}-RAM-RECLAIMED.json"
contract_sha="d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"

# The copy-only archival service publishes this receipt only after exact
# source/destination SHA-256 and size parity and durable ZFS publication.
for _ in $(seq 1 720); do
  if [ -f "${archive_receipt}" ]; then
    break
  fi
  sleep 30
done
test -f "${archive_receipt}"
test -d "${final}"
test ! -L "${final}"
test "$(jq -r .goal_contract_sha256 "${archive_receipt}")" = "sha256:${contract_sha}"
test "$(jq -r .goal_revision "${archive_receipt}")" = "11"
test "$(jq -r .source_destination_parity_passed "${archive_receipt}")" = "true"
test "$(jq -r .source_bytes_removed "${archive_receipt}")" = "false"

resolved_source="$(realpath -e -- "${source_root}")"
test "${resolved_source}" = "${expected_source}"
test -d "${resolved_source}"
test ! -L "${resolved_source}"
test "$(findmnt -T "${resolved_source}" -n -o FSTYPE)" = "tmpfs"
if find -P "${resolved_source}" -type l -print -quit | grep -q .; then
  echo "refusing tmpfs tree containing symbolic links" >&2
  exit 1
fi
if lsof -nP +D "${resolved_source}" 2>/dev/null | tail -n +2 | grep -q .; then
  echo "refusing tmpfs tree with open handles" >&2
  exit 1
fi

current_manifest="${final}/SOURCE-PRE-RECLAIM-MANIFEST.tsv"
test ! -e "${current_manifest}"
find -P "${resolved_source}" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
  rel="${f#/}"
  size="$(stat -c %s -- "${f}")"
  digest="$(sha256sum -- "${f}" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "${rel}" "${size}" "${digest}"
done > "${current_manifest}.partial"
mv "${current_manifest}.partial" "${current_manifest}"
cmp "${final}/SOURCE-MANIFEST.tsv" "${current_manifest}"

source_file_count="$(wc -l < "${current_manifest}" | tr -d ' ')"
source_total_bytes="$(awk -F '\t' '{s += $2} END {printf "%.0f", s}' "${current_manifest}")"
expected_file_count="$(jq -r .source_file_count "${archive_receipt}")"
expected_total_bytes="$(jq -r .source_total_bytes "${archive_receipt}")"
test "${source_file_count}" = "${expected_file_count}"
test "${source_total_bytes}" = "${expected_total_bytes}"

# This exact, fully resolved tmpfs root is the sole destructive target.
find -P "${resolved_source}" -depth -delete
test ! -e "${expected_source}"

available_kib_after="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
sealed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
test ! -e "${reclaim_receipt}"
python3 - "${reclaim_receipt}.partial" <<PY
import json, sys
payload = {
  "schema": "poke_bot.alakazam_rule_derivative_elmo_tmpfs_reclamation_receipt/v1",
  "goal_contract_sha256": "sha256:${contract_sha}",
  "goal_revision": 11,
  "archive_batch_id": "${batch_id}",
  "archive_receipt_path": "${archive_receipt}",
  "archive_receipt_sha256": "sha256:$(sha256sum "${archive_receipt}" | awk '{print $1}')",
  "source_root": "${expected_source}",
  "source_file_count": int("${source_file_count}"),
  "source_total_bytes": int("${source_total_bytes}"),
  "source_manifest_sha256": "sha256:$(sha256sum "${current_manifest}" | awk '{print $1}')",
  "source_destination_parity_passed": True,
  "source_root_removed": True,
  "inzi_source_bytes_removed": False,
  "mem_available_kib_after": int("${available_kib_after}"),
  "service_or_interactive_session_changed": False,
  "sealed_at_utc": "${sealed_at}"
}
with open(sys.argv[1], "x", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True, separators=(",", ":"))
    f.write("\n")
PY
mv "${reclaim_receipt}.partial" "${reclaim_receipt}"
sync -f "${archive_root}/receipts"
echo "reclaimed ${source_total_bytes} tmpfs bytes from ${expected_source}"
