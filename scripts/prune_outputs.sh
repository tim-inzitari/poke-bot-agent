#!/usr/bin/env bash
# Retention/cleanup for outputs/ disk bloat (tensor caches, checkpoints, rollouts, logs).
#
# Default is DRY RUN. Pass --apply to delete.
#
#   bash scripts/prune_outputs.sh
#   bash scripts/prune_outputs.sh --apply
#   KEEP_CACHE=2 KEEP_NAMED_CACHE=1 bash scripts/prune_outputs.sh --apply
set -euo pipefail
cd "$(dirname "$0")/.."

APPLY=0
STRIP_ROLLOUTS=1
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --no-strip-rollouts) STRIP_ROLLOUTS=0 ;;
  esac
done

KEEP_CACHE="${KEEP_CACHE:-2}"              # orphan top-level hash caches
KEEP_NAMED_CACHE="${KEEP_NAMED_CACHE:-1}"  # newest hash under lucario_fresh/, etc.
KEEP_CKPT="${KEEP_CKPT:-8}"                # baseline_/iter_ iterations per archetype
KEEP_LOGS="${KEEP_LOGS:-30}"               # newest log files per archetype glob
PRUNE_MIN_AGE_SEC="${PRUNE_MIN_AGE_SEC:-7200}"  # skip dirs/files touched in last 2h
DISK_USE_TRIGGER="${DISK_USE_TRIGGER:-80}"   # only used by prune_disk_unattended.sh

CACHE_DIR="outputs/cache/training_tensors"
SELF_PLAY_DIR="outputs/checkpoints/self_play"
ROLLOUT_DIR="outputs/rollouts"
LOG_DIR="outputs/logs"

run_rm() {
  if [[ "$APPLY" == "1" ]]; then
    rm -rf "$@"
  else
    echo "  [dry-run] would remove: $*"
  fi
}

human() { du -sh "$1" 2>/dev/null | cut -f1; }

too_new() {
  local path="$1"
  local now age
  now=$(date +%s)
  age=$((now - $(stat -c %Y "$path")))
  [[ "$age" -lt "$PRUNE_MIN_AGE_SEC" ]]
}

echo "==> prune outputs (apply=${APPLY}, KEEP_CACHE=${KEEP_CACHE}, KEEP_NAMED_CACHE=${KEEP_NAMED_CACHE}, KEEP_CKPT=${KEEP_CKPT})"

# Collect manifest-protected checkpoint paths.
declare -A PROTECT_CKPT=()
if [[ -d "$SELF_PLAY_DIR" ]]; then
  while IFS= read -r manifest; do
    while IFS= read -r ckpt; do
      [[ -n "$ckpt" ]] && PROTECT_CKPT["$ckpt"]=1
    done < <(
      python3 - "$manifest" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
paths = set()
for key in ("baseline_iterations", "iterations"):
    for entry in manifest.get(key) or []:
        saved = entry.get("saved_checkpoint")
        if saved:
            paths.add(str(Path(saved).resolve()))
            p = Path(saved)
            for suffix in (".best.pt", ".latest.pt"):
                alt = p.with_name(p.stem + suffix)
                if alt.exists():
                    paths.add(str(alt.resolve()))
champ = (manifest.get("champion") or {}).get("saved_checkpoint")
if champ:
    p = Path(champ)
    paths.add(str(p.resolve()))
    for suffix in (".best.pt", ".latest.pt"):
        alt = p.with_name(p.stem + suffix)
        if alt.exists():
            paths.add(str(alt.resolve()))
for line in sorted(paths):
    print(line)
PY
    )
  done < <(find "$SELF_PLAY_DIR" -name manifest.json)
fi

is_protected_ckpt() {
  local f="$1"
  local abs
  abs=$(readlink -f "$f" 2>/dev/null || realpath "$f")
  [[ -n "${PROTECT_CKPT[$abs]:-}" ]]
}

prune_hash_dirs() {
  local parent="$1"
  local keep="$2"
  local label="$3"
  mapfile -t hash_dirs < <(
    find "$parent" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/[0-9a-f]{64}$' -printf '%T@ %p\n' \
      | sort -rn | awk '{print $2}'
  )
  if (( ${#hash_dirs[@]} <= keep )); then
    echo "  ${label}: ${#hash_dirs[@]} hash dirs (<= keep ${keep})"
    return
  fi
  local removed=0
  for dir in "${hash_dirs[@]:keep}"; do
    if too_new "$dir"; then
      echo "  skip recent ${dir##*/}"
      continue
    fi
    echo "  ${label}: $(human "$dir") ${dir##*/}"
    run_rm "$dir"
    removed=$((removed + 1))
  done
  echo "  ${label}: pruned ${removed} hash dir(s), kept ${keep}"
}

# 1. Orphan top-level tensor hash caches.
if [[ -d "$CACHE_DIR" ]]; then
  echo "-- tensor caches (${CACHE_DIR}, $(human "$CACHE_DIR") total)"
  prune_hash_dirs "$CACHE_DIR" "$KEEP_CACHE" "orphan cache"

  # 2. Nested hashes under named archetype dirs (lucario_fresh, dragapult_fresh, ...).
  while IFS= read -r named; do
    [[ "$named" == "$CACHE_DIR" ]] && continue
    prune_hash_dirs "$named" "$KEEP_NAMED_CACHE" "${named##*/}"
  done < <(find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -type d ! -regextype posix-extended -regex '.*/[0-9a-f]{64}$')
fi

# 3. Self-play iteration checkpoints (baseline_* and iter_*).
if [[ -d "$SELF_PLAY_DIR" ]]; then
  echo "-- self-play checkpoints (${SELF_PLAY_DIR}, $(human "$SELF_PLAY_DIR") total)"
  while IFS= read -r dir; do
    mapfile -t nums < <(
      find "$dir" -maxdepth 1 \( -name 'baseline_*.pt' -o -name 'iter_*.pt' \) \
        ! -name '*.best.pt' ! -name '*.latest.pt' -printf '%f\n' \
        | sed -E 's/(baseline|iter)_0*([0-9]+)\.pt/\2/' | sort -rn | uniq
    )
    (( ${#nums[@]} > KEEP_CKPT )) || continue
    for n in "${nums[@]:KEEP_CKPT}"; do
      for prefix in baseline iter; do
        pad=$(printf '%03d' "$n")
        for f in "${dir}/${prefix}_${pad}.pt" "${dir}/${prefix}_${pad}.best.pt" "${dir}/${prefix}_${pad}.latest.pt"; do
          [[ -f "$f" ]] || continue
          if is_protected_ckpt "$f"; then
            echo "  keep manifest ckpt ${f##*/}"
            continue
          fi
          if too_new "$f"; then
            echo "  skip recent ${f##*/}"
            continue
          fi
          run_rm "$f"
        done
      done
    done
  done < <(find "$SELF_PLAY_DIR" -mindepth 1 -maxdepth 1 -type d)
fi

# 4. Strip observations from active self-play rollout buffers (features remain).
if [[ "$STRIP_ROLLOUTS" == "1" ]] && [[ -d "$ROLLOUT_DIR" ]]; then
  echo "-- strip rollout observations (${ROLLOUT_DIR})"
  mapfile -t rollout_files < <(find "$ROLLOUT_DIR" -maxdepth 1 -name '*_self_play.jsonl' -type f)
  if (( ${#rollout_files[@]} )); then
    if [[ "$APPLY" == "1" ]]; then
      python3 scripts/strip_rollout_observations.py "${rollout_files[@]}"
    else
      python3 scripts/strip_rollout_observations.py --dry-run "${rollout_files[@]}"
    fi
  fi
fi

# 5. Old unattended logs (keep newest KEEP_LOGS per archetype prefix).
if [[ -d "$LOG_DIR" ]]; then
  echo "-- logs (${LOG_DIR}, $(human "$LOG_DIR") total)"
  for prefix in lucario dragapult watchdog; do
    mapfile -t old_logs < <(ls -t "$LOG_DIR"/${prefix}_*.log 2>/dev/null | tail -n +$((KEEP_LOGS + 1)) || true)
    for f in "${old_logs[@]:-}"; do
      [[ -f "$f" ]] || continue
      run_rm "$f"
    done
  done
fi

if [[ "$APPLY" != "1" ]]; then
  echo "==> dry run only. Re-run with --apply to delete."
fi

df -h . | tail -1
