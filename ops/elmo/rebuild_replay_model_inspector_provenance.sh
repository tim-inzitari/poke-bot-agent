#!/usr/bin/env bash
# Rebuild exact replay-byte bindings from frozen evidence, then promote atomically.
set -euo pipefail

root="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}"
inspector_source="$root/replay-model-inspector-r176"
archive="$root/archive/replay-model-inspector"
replays="$root/archive/submission-replays"
output="$root/archive/replay-model-inspector-output-r194"
checkpoints="/mnt/Main/main/poke-bot-agent/containers/truenas-worker/checkpoint"
legacy_runtime="$archive/artifacts/runtimes/sha256-335d91c0f3f239d885c153a154531c0a43f33a9199e63a285be15deed55c0c5b/package"
image="poke-bot-truenas-worker:r125-checkpoint-digest-verify-v2"
candidate="$output/submissions-v1.hourly-candidate.json"
live="$archive/provenance/submissions-v1.json"

shopt -s nullglob
submission_evidence=("$archive"/provenance/submission-evidence/*.json)
runtime_parity_evidence=("$archive"/provenance/runtime-parity-accepted/*.json)

for required in \
    "$inspector_source/scripts/build_replay_inspector_provenance.py" \
    "$inspector_source/state/replay-model-inspector-submission-55217604-special-case-r184.json" \
    "$inspector_source/ops/elmo/replay-model-inspector-submission-55362452-r194.json" \
    "$archive/provenance/submissions-v1.pre-r176-fidelity-gates.json" \
    "$archive/provenance/runtime-parity-55315274-r176.json" \
    "$archive/provenance/runtime-parity-55324802-r176.json" \
    "$archive/provenance/runtime-parity-55362452-r194.json"; do
  [[ -r "$required" ]] || { echo "missing provenance input: $required" >&2; exit 1; }
done
for required in "${submission_evidence[@]}" "${runtime_parity_evidence[@]}"; do
  [[ -r "$required" ]] || { echo "missing generated provenance input: $required" >&2; exit 1; }
done
mkdir -p "$output"

generated_evidence_args=()
generated_parity_args=()

refresh_generated_args() {
  submission_evidence=("$archive"/provenance/submission-evidence/*.json)
  runtime_parity_evidence=("$archive"/provenance/runtime-parity-accepted/*.json)
  generated_evidence_args=()
  for evidence in "${submission_evidence[@]}"; do
    generated_evidence_args+=(
      --evidence "/data/inspector/provenance/submission-evidence/$(basename "$evidence")"
    )
  done
  generated_parity_args=()
  for receipt in "${runtime_parity_evidence[@]}"; do
    receipt_name="$(basename "$receipt")"
    submission_id="${receipt_name%.json}"
    [[ "$submission_id" =~ ^[0-9]+$ ]] || {
      echo "invalid generated runtime-parity receipt name: $receipt_name" >&2
      exit 1
    }
    case "$submission_id" in
      55315274|55324802|55362452)
        # These three historical receipts are supplied by their immutable,
        # explicitly named legacy paths below.  A generated compatibility
        # copy must not create a duplicate builder argument.
        continue
        ;;
    esac
    generated_parity_args+=(
      --runtime-parity-receipt
      "$submission_id=/data/inspector/provenance/runtime-parity-accepted/$receipt_name"
    )
  done
}

build_candidate() {
  /usr/bin/docker run --rm --network none --read-only \
  --user 950:950 --group-add 3000 \
  --memory 2g --memory-swap 2g --cpus 2 --pids-limit 64 \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m \
  -v "$inspector_source:/inspector:ro" \
  -v "$replays:/data/replays:ro" \
  -v "$archive:/data/inspector:ro" \
  -v "$output:/data/output:rw" \
  -v "$checkpoints:/data/checkpoints:ro" \
  -v "$legacy_runtime:/data/submitted-runtime:ro" \
  --entrypoint /usr/local/bin/python "$image" \
  /inspector/scripts/build_replay_inspector_provenance.py \
  --replay-root /data/replays \
  --evidence /data/inspector/provenance/submissions-v1.pre-r176-fidelity-gates.json \
  --evidence /inspector/state/replay-model-inspector-submission-55217604-special-case-r184.json \
  --evidence /inspector/ops/elmo/replay-model-inspector-submission-55362452-r194.json \
  "${generated_evidence_args[@]}" \
  --artifact-root /data/checkpoints \
  --artifact-root /data/inspector/provenance \
  --artifact-root /data/inspector/artifacts \
  --artifact-root /data/submitted-runtime \
  --runtime-parity-receipt 55315274=/data/inspector/provenance/runtime-parity-55315274-r176.json \
  --runtime-parity-receipt 55324802=/data/inspector/provenance/runtime-parity-55324802-r176.json \
  --runtime-parity-receipt 55362452=/data/inspector/provenance/runtime-parity-55362452-r194.json \
  "${generated_parity_args[@]}" \
  --output /data/output/"$(basename "$candidate")"
}

# First discover every newly archived exact package.  Then attest the exact
# extracted bytes and rebuild once with those submission-specific receipts so
# a newly accepted model becomes trace-ready in the same managed refresh.
refresh_generated_args
build_candidate
/usr/bin/python3 "$inspector_source/scripts/attest_archived_replay_inspector_runtimes.py" \
  --archive-root "$archive" \
  --manifest "$candidate" \
  --receipt-root "$archive/provenance/runtime-parity-accepted" \
  --verified-by "managed-r211-exact-bundle-extraction-attestation"
refresh_generated_args
build_candidate

# Parse and verify all exact artifact references before replacing the live snapshot.
/usr/bin/docker run --rm --network none --read-only \
  --user 950:950 --group-add 3000 \
  --memory 2g --memory-swap 2g --cpus 2 --pids-limit 64 \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m \
  -v "$inspector_source:/inspector:ro" \
  -v "$replays:/data/replays:ro" \
  -v "$archive:/data/inspector:ro" \
  -v "$output:/data/output:ro" \
  -v "$checkpoints:/data/checkpoints:ro" \
  -v "$legacy_runtime:/data/submitted-runtime:ro" \
  -e PYTHONPATH=/inspector \
  --entrypoint /usr/local/bin/python "$image" -c \
  'from pathlib import Path; from replay_inspector.provenance import load_provenance_manifest; m=load_provenance_manifest(Path("/data/output/submissions-v1.hourly-candidate.json"), source_roots=(Path("/data/replays"),Path("/data/checkpoints"),Path("/data/inspector/artifacts"),Path("/data/inspector/provenance"),Path("/data/submitted-runtime"))); assert len(m.entries) >= 1 and not m.issues; print(f"validated_submissions={len(m.entries)}")'

backup="$archive/provenance/submissions-v1.pre-hourly-refresh-r194.json"
if [[ ! -e "$backup" ]]; then
  install -m 0444 "$live" "$backup"
fi
install -m 0444 "$candidate" "$live.next"
mv -f "$live.next" "$live"
echo "[inspector-refresh] promoted provenance sha256=$(sha256sum "$live" | awk '{print $1}')"
