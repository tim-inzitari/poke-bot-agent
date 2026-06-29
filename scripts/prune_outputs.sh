#!/usr/bin/env bash
# Retention/cleanup for outputs/ disk bloat (tensor caches + self-play checkpoints).
#
# The tensor cache (outputs/cache/training_tensors/) accumulates one content-hash
# directory per data fingerprint and never self-cleans; self-play writes a
# {pt,best.pt,latest.pt} triple per baseline iteration. Both dominate disk use.
#
# Default is a DRY RUN — it prints what would be deleted. Pass --apply to delete.
#
# Usage:
#   bash scripts/prune_outputs.sh                 # dry run, default retention
#   bash scripts/prune_outputs.sh --apply         # actually delete
#   KEEP_CACHE=3 KEEP_CKPT=10 bash scripts/prune_outputs.sh --apply
#
# Env:
#   KEEP_CACHE  hash-dir tensor caches to keep (most recent by mtime)   [default 2]
#   KEEP_CKPT   baseline_NNN checkpoint iterations to keep per archetype [default 10]
set -euo pipefail
cd "$(dirname "$0")/.."

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

KEEP_CACHE="${KEEP_CACHE:-2}"
KEEP_CKPT="${KEEP_CKPT:-10}"
CACHE_DIR="outputs/cache/training_tensors"
SELF_PLAY_DIR="outputs/checkpoints/self_play"

run() {
  if [[ "$APPLY" == "1" ]]; then
    rm -rf "$@"
  else
    echo "  [dry-run] would remove: $*"
  fi
}

human() { du -sh "$1" 2>/dev/null | cut -f1; }

echo "==> outputs/ retention (apply=${APPLY}, KEEP_CACHE=${KEEP_CACHE}, KEEP_CKPT=${KEEP_CKPT})"

# 1. Tensor caches: keep named dirs (e.g. lucario_fresh) + KEEP_CACHE newest hash dirs.
if [[ -d "$CACHE_DIR" ]]; then
  echo "-- tensor caches in ${CACHE_DIR} ($(human "$CACHE_DIR") total)"
  # Hash dirs are 64-char hex; named dirs (lucario_fresh, dragapult_blackwell) are kept.
  mapfile -t hash_dirs < <(
    find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9a-f]{64}$' -printf '%T@ %p\n' \
      | sort -rn | awk '{print $2}'
  )
  if (( ${#hash_dirs[@]} > KEEP_CACHE )); then
    for dir in "${hash_dirs[@]:KEEP_CACHE}"; do
      run "$dir"
    done
  else
    echo "  nothing to prune (${#hash_dirs[@]} hash dirs <= KEEP_CACHE)"
  fi
fi

# 2. Self-play baseline checkpoints: keep last KEEP_CKPT iterations per archetype dir.
if [[ -d "$SELF_PLAY_DIR" ]]; then
  echo "-- self-play checkpoints in ${SELF_PLAY_DIR} ($(human "$SELF_PLAY_DIR") total)"
  while IFS= read -r dir; do
    mapfile -t iters < <(
      find "$dir" -maxdepth 1 -name 'baseline_*.pt' ! -name '*.best.pt' ! -name '*.latest.pt' \
        -printf '%f\n' | sed -E 's/baseline_0*([0-9]+)\.pt/\1/' | sort -rn
    )
    (( ${#iters[@]} > KEEP_CKPT )) || continue
    for n in "${iters[@]:KEEP_CKPT}"; do
      pad=$(printf '%03d' "$n")
      run "${dir}/baseline_${pad}.pt" "${dir}/baseline_${pad}.best.pt" "${dir}/baseline_${pad}.latest.pt"
    done
  done < <(find "$SELF_PLAY_DIR" -mindepth 1 -maxdepth 1 -type d)
fi

if [[ "$APPLY" != "1" ]]; then
  echo "==> dry run only. Re-run with --apply to delete."
fi
