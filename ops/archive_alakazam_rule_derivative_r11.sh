#!/usr/bin/env bash
set -euo pipefail

batch_id="r11-inzi-sealed-20260813T141600Z-d74152bc"
contract_sha="d74152bca415c80e4983172b5fdcd8c03313e0c8a0a18e24e0c68a7bcfe84245"
archive_host="elmo"
archive_root="/mnt/Main/main/poke-bot-agent/archive/alakazam-rule-derivative"
incoming="${archive_root}/.incoming/${batch_id}"
final="${archive_root}/batches/${batch_id}"
state_root="/home/inzi/poke-bot-agent/outputs/archive-transfer/alakazam-rule-derivative/${batch_id}"

sources=(
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-07-23
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-07-25
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-07-27
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-07-29
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-07-31
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-08-02
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-08-04
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-08-06
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-08-08
  /home/inzi/alakazam-r311-work/recent20-intraday15gb-inzi-r311/day-2026-08-10
  /home/inzi/poke-bot-agent/outputs/bootstrap/alakazam-rule-derivative-r10-semantic-pack-all20-v3
  /home/inzi/poke-bot-agent/outputs/pure_rl/alakazam_rule_derivative_g5/bootstrap-rev9-1d9026d8d461
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-blackwell-preflight-1b9effd2f5671ca415309267920f08138c46b33f4ca125cda7b0286839eeba54
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-candidate-validation-dee52d86966b
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-fleet-preflight2-bfc5736373bf
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-frozen-tensor-e0f6067f17c4a90b6c20555e0a93ec6f4f71b90a65552ec4cb62fdbf7fd26292
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-kaggle-package-bfc5736373bf
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-kaggle-queue-bfc5736373bf
  /home/inzi/poke-bot-agent/outputs/receipts/alakazam-rule-derivative-g5/revision9-kaggle-upload-bfc5736373bf
  /home/inzi/poke-bot-agent/outputs/submissions/alakazam-rule-derivative-g5/package-rev9-bfc5736373bf5d2b63d30218851f5907148a3ff18e237236d971c8af2eabc7e5
)

umask 077
mkdir -p "${state_root}"
test ! -L "${state_root}"

for source_path in "${sources[@]}"; do
  test -d "${source_path}"
  test ! -L "${source_path}"
  if find -P "${source_path}" -type l -print -quit | grep -q .; then
    echo "refusing source tree containing a symbolic link: ${source_path}" >&2
    exit 1
  fi
done

ssh -o BatchMode=yes -o ConnectTimeout=8 "${archive_host}" \
  "set -eu; test ! -e '${final}'; mkdir -p '${archive_root}/.incoming' '${archive_root}/batches'; if [ -e '${incoming}' ]; then test -d '${incoming}' && test ! -L '${incoming}'; else mkdir '${incoming}'; fi"

# Copy starts before the expensive full-tree digest pass. No source bytes are
# removed, and no destination file is published outside the private incoming
# batch until exact parity succeeds.
rsync -rltpR --safe-links --partial --append-verify --human-readable --info=stats2,progress2 \
  "${sources[@]}" "${archive_host}:${incoming}/"

source_manifest="${state_root}/SOURCE-MANIFEST.tsv"
source_manifest_tmp="${source_manifest}.partial"
: > "${source_manifest_tmp}"
for source_path in "${sources[@]}"; do
  while IFS= read -r -d '' file_path; do
    rel_path="${file_path#/}"
    size="$(stat -c %s -- "${file_path}")"
    digest="$(sha256sum -- "${file_path}" | awk '{print $1}')"
    printf '%s\t%s\t%s\n' "${rel_path}" "${size}" "${digest}" >> "${source_manifest_tmp}"
  done < <(find -P "${source_path}" -type f -print0 | sort -z)
done
mv "${source_manifest_tmp}" "${source_manifest}"

destination_manifest="${state_root}/DESTINATION-MANIFEST.tsv"
ssh -o BatchMode=yes "${archive_host}" \
  "set -eu; cd '${incoming}'; find -P home -type f -print0 | sort -z | while IFS= read -r -d '' f; do size=\$(stat -c %s -- \"\$f\"); digest=\$(sha256sum -- \"\$f\" | awk '{print \$1}'); printf '%s\\t%s\\t%s\\n' \"\$f\" \"\$size\" \"\$digest\"; done" \
  > "${destination_manifest}.partial"
mv "${destination_manifest}.partial" "${destination_manifest}"
cmp "${source_manifest}" "${destination_manifest}"

source_file_count="$(wc -l < "${source_manifest}" | tr -d ' ')"
source_total_bytes="$(awk -F '\t' '{s += $2} END {printf "%.0f", s}' "${source_manifest}")"
inventory_sha="$(sha256sum "${source_manifest}" | awk '{print $1}')"
sealed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
receipt="${state_root}/ARCHIVE-RECEIPT.json"

python3 - "${receipt}.partial" <<PY
import json
import sys

path = sys.argv[1]
payload = {
    "schema": "poke_bot.alakazam_rule_derivative_elmo_zfs_archive_receipt/v1",
    "goal_contract_path": "goals/alakazam-elmo-rule-derivative/contract.json",
    "goal_contract_sha256": "sha256:${contract_sha}",
    "goal_revision": 11,
    "archive_host": "elmo",
    "archive_root": "${archive_root}",
    "batch_id": "${batch_id}",
    "source_host": "inzi",
    "source_roots": ${sources[@]+$(printf '%s\n' "${sources[@]}" | python3 -c 'import json,sys; print(json.dumps([x.rstrip("\n") for x in sys.stdin]))')},
    "source_boundary_receipt_sha256s": [],
    "ordered_relative_path_type_size_sha256_inventory_sha256": "sha256:${inventory_sha}",
    "source_file_count": int("${source_file_count}"),
    "source_total_bytes": int("${source_total_bytes}"),
    "destination_object_paths": ["${final}"],
    "destination_file_count": int("${source_file_count}"),
    "destination_total_bytes": int("${source_total_bytes}"),
    "source_destination_parity_passed": True,
    "create_only_copy_skip_and_conflict_counts": {"copied_batch": 1, "skipped_exact_batch": 0, "conflict_count": 0},
    "source_bytes_removed": False,
    "source_removal_authorized_only_for_exact_verified_elmo_tmpfs_root": False,
    "active_or_unsealed_pending_inventory": [
        "/home/inzi/poke-bot-agent/outputs/bootstrap/alakazam-rule-derivative-r10-full-25epochs-c",
        "all_future_derivative_self_play_and_training_outputs not yet sealed at this batch boundary"
    ],
    "durable_sync_or_equivalent_passed": True,
    "service_or_runtime_change_performed": False,
    "sealed_at_utc": "${sealed_at}"
}
with open(path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
mv "${receipt}.partial" "${receipt}"

rsync -ltp --safe-links "${source_manifest}" "${destination_manifest}" "${receipt}" \
  "${archive_host}:${incoming}/"
ssh -o BatchMode=yes "${archive_host}" \
  "set -eu; sync -f '${incoming}'; test ! -e '${final}'; mv '${incoming}' '${final}'; sync -f '${archive_root}/batches'; test -f '${final}/ARCHIVE-RECEIPT.json'"

echo "archive batch published: ${final}"
