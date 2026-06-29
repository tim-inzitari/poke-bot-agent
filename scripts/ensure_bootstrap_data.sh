#!/usr/bin/env bash
# Ensure an archetype bootstrap JSONL exists, generating it via CABT self-play if not.
#
# Source this and call: ensure_bootstrap_data <out_jsonl> <deck_dir> [episodes]
#
# Archetype runs train on a single-archetype bootstrap corpus (e.g.
# data/lucario_bootstrap.jsonl). If the file is missing, a fresh run would crash
# with FileNotFoundError; this generates it from the archetype deck pool instead.
ensure_bootstrap_data() {
  local out="$1"
  local deck_dir="$2"
  local episodes="${3:-${BOOTSTRAP_EPISODES:-5000}}"

  if [[ -s "$out" ]]; then
    echo "==> bootstrap data present: ${out}"
    return 0
  fi

  if [[ ! -d "$deck_dir" ]]; then
    echo "ERROR: deck pool ${deck_dir} not found; run scripts/build_archetype_deck_pool.sh first" >&2
    return 1
  fi

  echo "==> bootstrap data missing; generating ${episodes} CABT games -> ${out}"
  mkdir -p "$(dirname "$out")"
  python3 scripts/generate_cabt_data.py \
    --episodes "${episodes}" \
    --deck-dir "${deck_dir}" \
    --matchups sample \
    --out "${out}"
}
