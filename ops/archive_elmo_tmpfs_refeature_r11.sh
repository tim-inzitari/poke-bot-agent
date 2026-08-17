#!/usr/bin/env bash
set -euo pipefail

source_root="/dev/shm/alakazam-refeature-2332383-1"
expected_source="/dev/shm/alakazam-refeature-2332383-1"
archive_root="/mnt/Main/main/poke-bot-agent/archive/alakazam-rule-derivative"
batch_id="r11-elmo-tmpfs-recovery-2332383-20260813T142000Z-d74152bc"
incoming="${archive_root}/.incoming/${batch_id}"
final="${archive_root}/batches/${batch_id}"
state_root="${archive_root}/receipts/${batch_id}"
contract_sha="d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"

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

test ! -e "${final}"
mkdir -p "${archive_root}/.incoming" "${archive_root}/batches" "${archive_root}/receipts"
test ! -L "${archive_root}"
test ! -L "${archive_root}/.incoming"
test ! -L "${archive_root}/batches"
test ! -L "${archive_root}/receipts"
if [ -e "${incoming}" ]; then
  test -d "${incoming}"
  test ! -L "${incoming}"
else
  mkdir "${incoming}"
fi

rsync -rltpR --safe-links --partial --append-verify --human-readable --info=stats2,progress2 \
  "${resolved_source}" "${incoming}/"

source_manifest="${incoming}/SOURCE-MANIFEST.tsv"
destination_manifest="${incoming}/DESTINATION-MANIFEST.tsv"
find -P "${resolved_source}" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
  rel="${f#/}"
  size="$(stat -c %s -- "${f}")"
  digest="$(sha256sum -- "${f}" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "${rel}" "${size}" "${digest}"
done > "${source_manifest}.partial"
mv "${source_manifest}.partial" "${source_manifest}"

cd "${incoming}"
find -P dev -type f -print0 | sort -z | while IFS= read -r -d '' f; do
  size="$(stat -c %s -- "${f}")"
  digest="$(sha256sum -- "${f}" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "${f}" "${size}" "${digest}"
done > "${destination_manifest}.partial"
mv "${destination_manifest}.partial" "${destination_manifest}"
cmp "${source_manifest}" "${destination_manifest}"

file_count="$(wc -l < "${source_manifest}" | tr -d ' ')"
total_bytes="$(awk -F '\t' '{s += $2} END {printf "%.0f", s}' "${source_manifest}")"
inventory_sha="$(sha256sum "${source_manifest}" | awk '{print $1}')"
sync -f "${incoming}"
test ! -e "${final}"
mv "${incoming}" "${final}"
sync -f "${archive_root}/batches"

# Revalidate the exact source after archival. The owner directed the ZFS copy
# to continue despite the disk warning, but source removal remains disabled.
test "$(realpath -e -- "${source_root}")" = "${expected_source}"
test "$(findmnt -T "${source_root}" -n -o FSTYPE)" = "tmpfs"
if lsof -nP +D "${source_root}" 2>/dev/null | tail -n +2 | grep -q .; then
  echo "source acquired an open handle after archival; preserving it" >&2
  exit 1
fi
recheck_manifest="${final}/SOURCE-RECHECK-MANIFEST.tsv"
find -P "${source_root}" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
  rel="${f#/}"
  size="$(stat -c %s -- "${f}")"
  digest="$(sha256sum -- "${f}" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "${rel}" "${size}" "${digest}"
done > "${recheck_manifest}.partial"
mv "${recheck_manifest}.partial" "${recheck_manifest}"
cmp "${final}/SOURCE-MANIFEST.tsv" "${recheck_manifest}"
test -d "${source_root}"

sealed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
receipt_tmp="${archive_root}/receipts/.${batch_id}.partial"
receipt="${archive_root}/receipts/${batch_id}.json"
test ! -e "${receipt_tmp}"
test ! -e "${receipt}"
python3 - "${receipt_tmp}" <<PY
import json, sys
payload = {
  "schema": "poke_bot.alakazam_rule_derivative_elmo_zfs_archive_receipt/v1",
  "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
  "goal_contract_sha256": "sha256:${contract_sha}",
  "goal_revision": 11,
  "archive_host": "elmo",
  "archive_root": "${archive_root}",
  "batch_id": "${batch_id}",
  "source_host": "elmo",
  "source_roots": ["${expected_source}"],
  "source_boundary_receipt_sha256s": [],
  "ordered_relative_path_type_size_sha256_inventory_sha256": "sha256:${inventory_sha}",
  "source_file_count": int("${file_count}"),
  "source_total_bytes": int("${total_bytes}"),
  "destination_object_paths": ["${final}"],
  "destination_file_count": int("${file_count}"),
  "destination_total_bytes": int("${total_bytes}"),
  "source_destination_parity_passed": True,
  "create_only_copy_skip_and_conflict_counts": {"copied_batch": 1, "skipped_exact_batch": 0, "conflict_count": 0},
  "source_bytes_removed": False,
  "source_removal_authorized_only_for_exact_verified_elmo_tmpfs_root": False,
  "active_or_unsealed_pending_inventory": [],
  "durable_sync_or_equivalent_passed": True,
  "service_or_runtime_change_performed": False,
  "sealed_at_utc": "${sealed_at}"
}
with open(sys.argv[1], "x", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True, separators=(",", ":"))
    f.write("\n")
PY
mv "${receipt_tmp}" "${receipt}"
sync -f "${archive_root}/receipts"
echo "archived exact tmpfs tree and preserved source: ${expected_source}"
