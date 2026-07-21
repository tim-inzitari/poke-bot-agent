#!/usr/bin/env bash
# Contest urgency: kill mismatched collect and relaunch matched full-HW.
# Thruput fix: multi_env=1 (classic 96 OS workers). multi_env=4 was ~2x slower.
set -euo pipefail
cd /home/inzi/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T175340Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT="${RUN_DIR}/checkpoints/seed.pt"
LOG=outputs/logs/pure_rl_core.log
RESTART_LOG=outputs/logs/pure_rl_capacity_scheduler_restart.log

[[ -f "$BASE_CKPT" ]] || { echo "missing seed: $BASE_CKPT" >&2; exit 1; }

pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/resource_watcher.py" 2>/dev/null || true
pkill -f "scripts/pure_rl_auto_progress.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/_mirror_shard_progress.py" 2>/dev/null || true
sleep 3

"$PY" - <<'PY'
import json, time
from pathlib import Path
from poke_bot.train import load_model_from_checkpoint
from poke_bot.pure_rl.model_profile import count_params
from poke_bot.live_pool import write_live_pool_plan

run = Path("outputs/pure_rl/pure_rl_core_overnight_20260716T175340Z")
ckpt = run / "checkpoints" / "seed.pt"
m = load_model_from_checkpoint(str(ckpt))
n = count_params(m)
print("seed_ok", n, "dense", getattr(m, "dense_card2vec", None))
assert 1_600_000 <= n <= 2_000_000, n

(run / "rebalance_state.json").write_text(json.dumps({
    "seq": 0,
    "sim_workers": 96,
    "local_share": 0.55,
    "min_workers": 64,
    "max_workers": 160,
    "max_total_workers": 10000,
    "min_remote_frac": 0.25,
    "ema_alpha": 0.35,
    "metrics": "wave_wall_clock_gps",
    "remote_batch": "chunked_rtt_amortize",
    "remote_demand": {
        "elmo_default": 20, "elmo_max": 40,
        "bert_default": 10, "bert_max": 20,
    },
    "note": "defaults << max: local 96<<160; peak 160+40+20=220 << max_total 10000; elmo 20<<40; bert 10<<20; remotes additive; multi_env=1",
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
    "max_total_workers": 10000,
    "min_remote_frac": 0.25,
    "leaf_default": [10, 40],
    "leaf_max_gpu0": 12,
    "leaf_max_gpu1": 48,
    "leaf_max_total": 60,
    "elmo_default_workers": 20,
    "elmo_max_workers": 40,
    "bert_default_workers": 10,
    "bert_max_workers": 20,
    "multi_env_per_worker": 1,
    "games_per_iter": 2048,
    "note": "3080-safe: GPU0=10 (hard max12); BW=40 (max48); multi_env=1; games=2048",
}, indent=2) + "\n")
plan = write_live_pool_plan(
    seq=4,
    workers=96,
    leaf_servers=50,
    leaf_gpu0=10,
    leaf_gpu1=40,
    promotion_workers=8,
    reason="3080-safe thruput: multi_env=1 96w; GPU0=10 (≤12 hard) GPU1=40; games=2048",
    apply="next_iter",
)
print("live_pool", plan.workers, plan.leaf_gpu0, plan.leaf_gpu1, plan.leaf_servers)
shard = run / "shards" / "iter_00000.jsonl"
if shard.is_file():
    bak = shard.with_suffix(".jsonl.prev_incomplete")
    if bak.exists():
        bak.unlink()
    shard.rename(bak)
    print("moved_shard", bak)
print("prepared")
PY

printf '\n===== THRUPUT RECOVERY RELAUNCH %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

export POKEBOT_PYTHON="$PY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_LIVE_POOL=1
# Classic path: 96 OS workers (NOT multi_env=4 which starved leaves).
export POKEBOT_MULTI_ENV=0
export POKEBOT_MULTI_ENV_PER_WORKER=1
export PURE_RL_MULTI_ENV=0
export PURE_RL_MULTI_ENV_PER_WORKER=1
export POKEBOT_WORKER_CPU_ONLY=1
export POKEBOT_REMOTE_ENDPOINT_WEIGHTS="192.168.1.143:8765=1.0,bert.local:8766=1.0"
export POKEBOT_REMOTE_DEFAULT_WORKERS="192.168.1.143:8765=20,bert.local:8766=10"
export POKEBOT_REMOTE_MAX_WORKERS="192.168.1.143:8765=40,bert.local:8766=20"
export PURE_RL_SIM_WORKERS=96
export PURE_RL_GAMES_IN_FLIGHT=96
export PURE_RL_GAMES_PER_ITER=2048
export PURE_RL_LEAF_GPU0_REPLICAS=10
export PURE_RL_LEAF_GPU1_REPLICAS=40
export PURE_RL_LEAF_COALESCE_MS=0
export PURE_RL_MID_ITER_SCHEDULER=1
export PURE_RL_REMOTE_DISPATCH_CHUNK=128
export PURE_RL_REBALANCE_PREFER_LOCAL_FRAC=0.55
export PURE_RL_REBALANCE_MIN_LOCAL_FRAC=0.40
export PURE_RL_REBALANCE_MAX_REMOTE_FRAC=0.60
export PURE_RL_REBALANCE_MIN_REMOTE_FRAC=0.25
export PURE_RL_REBALANCE_MAX_WORKERS=160
export PURE_RL_REBALANCE_MAX_TOTAL_WORKERS=10000
export PURE_RL_REBALANCE_MIN_WORKERS=64
export POKEBOT_LIVE_POOL_MAX_WORKERS=160
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU0=12
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU1=48
export POKEBOT_LIVE_POOL_MAX_LEAF_SERVERS=60
export PURE_RL_LEAF_GPU0_FRAC=0.20
# Soft-connect: one remote blip must not zero the whole farm.
export POKEBOT_REMOTE_REQUIRE_ALL=0
# Belt: train uses max_batch=None (per-device); keep env low if anything reads it.
export POKEBOT_LEAF_SERVER_MAX_BATCH=256
export PURE_RL_TQDM_INPLACE=1
export PURE_RL_WAVE_GPS_MIN_WINDOW_S=20
export POKEBOT_LEAF_SERVER_COALESCE_MS=0
export POKEBOT_SKIP_GAME_ACCURACY=1

nohup "$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile none \
  --log "$LOG" \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  --multi-env-per-worker 1 \
  --leaf-coalesce-ms 0.0 \
  --auto-progress \
  -- \
  --base-checkpoint "$BASE_CKPT" \
  --iterations 1000 \
  --games-per-iter 2048 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  >"$RESTART_LOG" 2>&1 &
echo "launcher_bg=$!"
sleep 30
echo "=== restart log ==="
cat "$RESTART_LOG"
echo "=== procs ==="
ps -eo pid,etime,cmd | awk '/python.*scripts\/(train|launch)_pure_rl|resource_watcher|pure_rl_auto_progress/ && !/awk/ {print}'
echo "=== prove ==="
rg -n 'THRUPUT RECOVERY|PURE_RL_PROGRESS_SPLIT|PURE_RL_WATCHER|PURE_RL_AUTO_PROGRESS|multi_env=|full hardware|leaves_gpu0|games-per-iter|Error|refuse' "$LOG" "$RESTART_LOG" 2>/dev/null | tail -40
