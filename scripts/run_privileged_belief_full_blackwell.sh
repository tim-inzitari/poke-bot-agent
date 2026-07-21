#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/inzi/poke-bot-agent
PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python
CORPUS="$ROOT/outputs/privileged_belief/exact_core_20k_v1"
MANIFEST="$CORPUS/manifest.json"
CACHE="$CORPUS/.feature-cache"
SOURCE="$ROOT/outputs/checkpoints/canary/state_core_latest10_epoch23_20260719T1500.bootstrap-source.pt"
RUN=state_core_privileged_belief_20k_resident_v1
LATEST="$ROOT/outputs/checkpoints/${RUN}.latest.pt"
BEST="$ROOT/outputs/checkpoints/${RUN}.best.pt"
STATUS="$CORPUS/resident_train.status.json"
CACHE_STATUS="$CORPUS/feature_cache.status.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 24 * 1024 * 1024 )); then
  echo "[belief-full] host memory guard failed: MemAvailable=${available_kib} KiB" >&2
  exit 1
fi
test -s "$MANIFEST"
test -s "$SOURCE"
mkdir -p "$CACHE" "$ROOT/outputs/logs" "$ROOT/outputs/checkpoints"

cd "$ROOT"
"$PYTHON" - "$SOURCE" <<'PY'
import sys
import torch
from poke_bot import checkpoint

source = sys.argv[1]
name = torch.cuda.get_device_name(0)
if "Blackwell" not in name:
    raise SystemExit(f"full replay training must use Blackwell, got {name!r}")
saved = checkpoint.load_checkpoint(source, map_location="cpu")
if int(saved.get("epoch", -1)) != 23:
    raise SystemExit(f"expected immutable epoch-23 source, got {saved.get('epoch')}")
print(f"[belief-full] device={name} source_epoch=23", flush=True)
PY

test -s "$CACHE_STATUS"

exec "$PYTHON" -u scripts/train_privileged_belief_resident.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE" \
  --init-checkpoint "$SOURCE" \
  --latest-checkpoint "$LATEST" \
  --best-checkpoint "$BEST" \
  --status-json "$STATUS" \
  --device cuda:0 \
  --epochs 26 \
  --batch-size 32768 \
  --min-free-gib 12 \
  --min-step-headroom-gib 3 \
  --checkpoint-every 50 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --val-modulus 10 \
  --patience 6 \
  --min-delta 1e-4 \
  --aux-weight 0.10 \
  --hand-weight 0.40 \
  --remainder-weight 0.30 \
  --lethal-weight 0.10 \
  --prize-race-weight 0.10 \
  --value-weight 1.0 \
  --awr-beta 0.5 \
  --awr-weight-max 20.0 \
  --entropy-bonus 0.01 \
  --seed 20260719
