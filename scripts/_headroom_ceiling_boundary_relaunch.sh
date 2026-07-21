#!/usr/bin/env bash
# Wait for collect to finish (promotion / iter boundary), then relaunch with
# max>>default headroom ceilings while keeping remote endpoints untouched.
# Does NOT kill mid-collection.
set -euo pipefail
cd /home/inzi/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T175340Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT="${RUN_DIR}/checkpoints/seed.pt"
LOG=outputs/logs/pure_rl_core.log
STATUS=outputs/logs/pure_rl_core.progress.status
WAKE_LOG=outputs/logs/pure_rl_headroom_ceiling_boundary.log
PIDFILE=outputs/state/HEADROOM_CEILING_BOUNDARY.pid

exec >>"$WAKE_LOG" 2>&1
echo "===== HEADROOM BOUNDARY WAKE $(date -Is) pid=$$ ====="
echo $$ >"$PIDFILE"

# Poll until collect is no longer the active stage (promotion / train / next iter).
while true; do
  st="$(cat "$STATUS" 2>/dev/null || true)"
  echo "$(date -Is) status=${st:0:160}"
  if [[ "$st" != *collect* ]]; then
    echo "collect cleared; proceeding with headroom relaunch"
    break
  fi
  # Also proceed if trainer died.
  if ! pgrep -f "scripts/train_pure_rl.py.*${RUN_NAME}" >/dev/null 2>&1; then
    echo "trainer gone; proceeding"
    break
  fi
  sleep 20
done

# Brief settle so promotion-boundary consumers can finish cleanly if mid-handoff.
sleep 5

"$PY" - <<'PY'
import json, time
from pathlib import Path
from poke_bot.live_pool import write_live_pool_plan, _MAX_WORKERS, _MAX_LEAF_SERVERS

run = Path("outputs/pure_rl/pure_rl_core_overnight_20260716T175340Z")
assert _MAX_WORKERS >= 160 and _MAX_LEAF_SERVERS >= 80, (_MAX_WORKERS, _MAX_LEAF_SERVERS)
(run / "rebalance_state.json").write_text(json.dumps({
    "seq": 0,
    "sim_workers": 96,
    "local_share": 0.55,
    "min_workers": 64,
    "max_workers": 160,
    "ema_alpha": 0.35,
    "metrics": "wave_wall_clock_gps",
    "remote_batch": "chunked_rtt_amortize",
    "note": "default 96 << max 160",
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}, indent=2) + "\n")
(run / "wave_boundary_knobs.json").write_text(json.dumps({
    "min_local_frac": 0.40,
    "max_remote_frac": 0.60,
    "prefer_local_frac": 0.55,
    "mid_iter_scheduler": True,
    "remote_dispatch_chunk": 128,
    "tqdm_inplace": True,
    "target_workers": 96,
    "max_workers": 160,
    "leaf_default": [18, 24],
    "leaf_max_total": 80,
    "note": "max>>default headroom relaunch",
}, indent=2) + "\n")
# Steady-ish start (one step above 18/24); watcher still has room to 80.
plan = write_live_pool_plan(
    seq=int(time.time()) % 1_000_000,
    workers=96,
    leaf_servers=56,
    leaf_gpu0=24,
    leaf_gpu1=32,
    promotion_workers=8,
    reason="headroom ceilings: start 96/24/32; max 160w/80leaves",
    apply="next_iter",
)
proof = {
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "defaults": {"workers": 96, "leaf_gpu0": 18, "leaf_gpu1": 24},
    "plan_start": {
        "workers": plan.workers,
        "leaf_gpu0": plan.leaf_gpu0,
        "leaf_gpu1": plan.leaf_gpu1,
        "leaf_servers": plan.leaf_servers,
    },
    "max_ceilings": {
        "workers": _MAX_WORKERS,
        "leaf_servers": _MAX_LEAF_SERVERS,
    },
    "max_gt_default": {
        "workers": _MAX_WORKERS > 96,
        "leaves": _MAX_LEAF_SERVERS > 42,
    },
    "remote_endpoints_untouched": "192.168.1.143:8765,bert.local:8766",
}
Path("outputs/state/HEADROOM_CEILING_PROOF.json").write_text(
    json.dumps(proof, indent=2) + "\n", encoding="utf-8"
)
print("proof", proof)
print("live_pool", plan.workers, plan.leaf_gpu0, plan.leaf_gpu1)
PY

# Stop local trainer/watcher only — do not touch Elmo/bert remotes.
pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/resource_watcher.py" 2>/dev/null || true
pkill -f "scripts/pure_rl_auto_progress.py.*${RUN_NAME}" 2>/dev/null || true
sleep 3

export POKEBOT_PYTHON="$PY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_LIVE_POOL=1
export POKEBOT_MULTI_ENV=1
export POKEBOT_MULTI_ENV_PER_WORKER=4
export POKEBOT_WORKER_CPU_ONLY=1
export POKEBOT_REMOTE_ENDPOINT_WEIGHTS="192.168.1.143:8765=1.0,bert.local:8766=0.40"
export PURE_RL_SIM_WORKERS=96
export PURE_RL_GAMES_IN_FLIGHT=96
export PURE_RL_LEAF_GPU0_REPLICAS=18
export PURE_RL_LEAF_GPU1_REPLICAS=24
export PURE_RL_LEAF_COALESCE_MS=0
export PURE_RL_MID_ITER_SCHEDULER=1
export PURE_RL_REMOTE_DISPATCH_CHUNK=128
export PURE_RL_REBALANCE_PREFER_LOCAL_FRAC=0.55
export PURE_RL_REBALANCE_MIN_LOCAL_FRAC=0.40
export PURE_RL_REBALANCE_MAX_REMOTE_FRAC=0.60
export PURE_RL_REBALANCE_MAX_WORKERS=160
export PURE_RL_REBALANCE_MIN_WORKERS=64
export POKEBOT_LIVE_POOL_MAX_WORKERS=160
export POKEBOT_LIVE_POOL_MAX_LEAF_SERVERS=80
export PURE_RL_TQDM_INPLACE=1
export PURE_RL_WAVE_GPS_MIN_WINDOW_S=20
export POKEBOT_LEAF_SERVER_COALESCE_MS=0
export POKEBOT_SKIP_GAME_ACCURACY=1

printf '\n===== HEADROOM CEILING RELAUNCH %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

nohup "$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile none \
  --log "$LOG" \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  --auto-progress \
  -- \
  --base-checkpoint "$BASE_CKPT" \
  --iterations 1000 \
  --games-per-iter 4096 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  >>"$WAKE_LOG" 2>&1 &

echo "relaunched launch_pure_rl pid=$!"
sleep 8
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader || true
cat outputs/state/live_pool_plan.json || true
rg -n "knob headroom|HEADROOM|ceiling" outputs/logs/resource_watcher.log | tail -20 || true
rm -f "$PIDFILE"
echo "done $(date -Is)"
