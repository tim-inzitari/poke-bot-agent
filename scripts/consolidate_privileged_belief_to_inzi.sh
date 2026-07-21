#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "usage: $0 CANONICAL_DIR INZI_DIR ELMO_HOST ELMO_DIR BERT_HOST BERT_DIR REQUIRE_GAMES POLL_SECONDS [EXTRA_INZI_DIR]" >&2
  exit 2
fi

canonical=$1
inzi_dir=$2
elmo_host=$3
elmo_dir=$4
bert_host=$5
bert_dir=$6
require_games=$7
poll_seconds=$8
extra_inzi_dir=${9:-}
root=$(cd "$(dirname "$0")/.." && pwd)
status="$canonical/consolidation.status.json"

mkdir -p "$canonical/inzi" "$canonical/elmo" "$canonical/bert"
if [[ -n "$extra_inzi_dir" ]]; then
  mkdir -p "$canonical/inzi-extra"
  if [[ "$(realpath "$extra_inzi_dir")" != "$(realpath "$canonical/inzi-extra")" ]]; then
    echo "EXTRA_INZI_DIR must be CANONICAL_DIR/inzi-extra" >&2
    exit 2
  fi
fi
if [[ "$(realpath "$inzi_dir")" != "$(realpath "$canonical/inzi")" ]]; then
  echo "INZI_DIR must be CANONICAL_DIR/inzi so local shards remain canonical" >&2
  exit 2
fi

write_status() {
  local phase=$1
  local message=$2
  local tmp="$status.tmp.$$"
  python3 - "$tmp" "$phase" "$message" <<'PY'
import json, os, sys, time
path, phase, message = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({
        "schema": "poke_bot.privileged_belief_consolidation_status/v1",
        "phase": phase,
        "message": message,
        "updated_unix": int(time.time()),
    }, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  mv "$tmp" "$status"
}

sync_host() {
  local host=$1
  local source=$2
  local dest=$3
  rsync -rt \
    --timeout=30 \
    --include='shard_*.jsonl' \
    --include='shard_*.meta.json' \
    --include='manifest.json' \
    --exclude='*' \
    "$host:$source/" "$dest/"
}

write_status waiting "waiting for distributed collectors"
while true; do
  sync_host "$elmo_host" "$elmo_dir" "$canonical/elmo" || true
  sync_host "$bert_host" "$bert_dir" "$canonical/bert" || true

  inzi_shards=$(find "$canonical/inzi" -maxdepth 1 -name 'shard_*.meta.json' | wc -l)
  elmo_shards=$(find "$canonical/elmo" -maxdepth 1 -name 'shard_*.meta.json' | wc -l)
  bert_shards=$(find "$canonical/bert" -maxdepth 1 -name 'shard_*.meta.json' | wc -l)
  extra_shards=0
  if [[ -n "$extra_inzi_dir" ]]; then
    extra_shards=$(find "$canonical/inzi-extra" -maxdepth 1 -name 'shard_*.meta.json' | wc -l)
  fi
  write_status collecting "verified-source sidecars present: inzi=$inzi_shards inzi-extra=$extra_shards elmo=$elmo_shards bert=$bert_shards"

  if [[ -f "$canonical/inzi/manifest.json" \
        && -f "$canonical/elmo/manifest.json" \
        && -f "$canonical/bert/manifest.json" \
        && ( -z "$extra_inzi_dir" || -f "$canonical/inzi-extra/manifest.json" ) ]]; then
    write_status verifying "all host manifests present; hashing canonical shards"
    "$root/.venv/bin/python" --version >/dev/null 2>&1 && py="$root/.venv/bin/python" || py=python3
    manifest_args=(
      --input-manifest "$canonical/inzi/manifest.json"
      --input-manifest "$canonical/elmo/manifest.json"
      --input-manifest "$canonical/bert/manifest.json"
    )
    if [[ -n "$extra_inzi_dir" ]]; then
      manifest_args+=(--input-manifest "$canonical/inzi-extra/manifest.json")
    fi
    if "$py" "$root/scripts/assemble_privileged_belief_manifest.py" \
      "${manifest_args[@]}" \
      --output-manifest "$canonical/manifest.json" \
      --require-games "$require_games"; then
      write_status complete "canonical hash-verified corpus saved on Inzi"
      exit 0
    fi
    write_status error "canonical validation failed; see service log"
  fi
  sleep "$poll_seconds"
done
