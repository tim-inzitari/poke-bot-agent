#!/usr/bin/env bash
# Boundary relaunch: even-spread leaf map + least-queue GPU0 feed + 96 workers.
# Do NOT raise GPU0 leaf count (hard max 12). Keep Elmo+bert chunk=128.
set -euo pipefail
cd /home/pokebot/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T175340Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT="${RUN_DIR}/checkpoints/seed.pt"
# Prefer latest trained ckpt if present.
if [[ -f "${RUN_DIR}/checkpoints/iter_00000.pt" ]]; then
  BASE_CKPT="${RUN_DIR}/checkpoints/iter_00000.pt"
fi
LOG=outputs/logs/pure_rl_core.log
RESTART_LOG=outputs/logs/pure_rl_gpu0_least_queue_restart.log

[[ -f "$BASE_CKPT" ]] || { echo "missing ckpt: $BASE_CKPT" >&2; exit 1; }

# Env must be set before profile asserts (and before kill so prepare can run first).
export POKEBOT_PYTHON="$PY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_LIVE_POOL=1
export POKEBOT_WORKER_CPU_ONLY=1
export POKEBOT_REMOTE_ENDPOINT_WEIGHTS="elmo:8765=1.0,bert.local:8766=0.40"
export PURE_RL_SIM_WORKERS=96
export PURE_RL_GAMES_IN_FLIGHT=96
export PURE_RL_LEAF_GPU0_REPLICAS=10
export PURE_RL_LEAF_GPU1_REPLICAS=40
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU0=12
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU1=48
export PURE_RL_LEAF_GPU0_FRAC=0.20
export PURE_RL_GPU0_CLIENT_FRAC=0.38
export PURE_RL_LEAF_COALESCE_MS=0
export PURE_RL_MID_ITER_SCHEDULER=1
export PURE_RL_REMOTE_DISPATCH_CHUNK=128
export PURE_RL_REBALANCE_PREFER_LOCAL_FRAC=0.60
export PURE_RL_REBALANCE_MIN_LOCAL_FRAC=0.40
export PURE_RL_REBALANCE_MAX_REMOTE_FRAC=0.55
export PURE_RL_REBALANCE_MAX_WORKERS=160
export PURE_RL_REBALANCE_MAX_TOTAL_WORKERS=10000
export PURE_RL_REBALANCE_MIN_WORKERS=64
export POKEBOT_LIVE_POOL_MAX_WORKERS=160
export POKEBOT_LIVE_POOL_MAX_LEAF_SERVERS=60
export POKEBOT_LEAF_SERVER_MAX_BATCH=256
export PURE_RL_TQDM_INPLACE=1
export PURE_RL_WAVE_GPS_MIN_WINDOW_S=20
export POKEBOT_LEAF_SERVER_COALESCE_MS=0
export POKEBOT_SKIP_GAME_ACCURACY=1
export POKEBOT_MULTI_ENV=0
export POKEBOT_MULTI_ENV_PER_WORKER=1
export PURE_RL_MULTI_ENV=0
export PURE_RL_MULTI_ENV_PER_WORKER=1

"$PY" - <<PY
import json, time
from pathlib import Path
from poke_bot.train import load_model_from_checkpoint
from poke_bot.pure_rl.model_profile import count_params
from poke_bot.pure_rl.hardware import full_hardware_profile, sticky_leaf_server_index
from poke_bot.live_pool import write_live_pool_plan

run = Path("${RUN_DIR}")
m = load_model_from_checkpoint("${BASE_CKPT}")
n = count_params(m)
print("ckpt_ok", "${BASE_CKPT}", n)
hw = full_hardware_profile()
devs = hw.leaf_cuda_devices()
assert set(devs) == {0, 1}, devs
assert devs.count(0) == 10 and devs.count(1) == 40, (devs.count(0), devs.count(1))
g0 = [i for i, d in enumerate(devs) if d == 0]
assert g0[-1] >= 40, g0  # even-spread, not bunched in first 20
binds = [sticky_leaf_server_index(s, devs, gpu0_client_frac=0.38) for s in range(96)]
g0c = sum(1 for b in binds if devs[b] == 0)
assert 30 <= g0c <= 45, g0c
print("even_spread_ok", "g0_idx", g0[:4], "...", g0[-1], "sticky_g0_clients", g0c)

(run / "rebalance_state.json").write_text(json.dumps({
    "seq": 0, "sim_workers": 96, "local_share": 0.60,
    "min_workers": 64, "max_workers": 160, "max_total_workers": 10000,
    "ema_alpha": 0.35,
    "metrics": "wave_wall_clock_gps", "remote_batch": "chunked_rtt_amortize",
    "gpu0_feed": "least_queue+sticky_bias0.38+even_spread",
    "remote_demand": {
        "elmo_default": 20, "elmo_max": 40,
        "bert_default": 10, "bert_max": 20,
    },
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}, indent=2) + "\n")
(run / "wave_boundary_knobs.json").write_text(json.dumps({
    "min_local_frac": 0.40, "max_remote_frac": 0.55, "prefer_local_frac": 0.60,
    "mid_iter_scheduler": True, "remote_dispatch_chunk": 128, "tqdm_inplace": True,
    "target_workers": 96, "max_workers": 160, "max_total_workers": 10000,
    "leaf_default": [10, 40], "leaf_max_gpu0": 12, "leaf_max_gpu1": 48,
    "gpu0_client_frac": 0.38,
    "elmo_default_workers": 20, "elmo_max_workers": 40,
    "bert_default_workers": 10, "bert_max_workers": 20,
    "note": "least-queue GPU0 feed; workers floor 96; GPU0 leaves=10",
}, indent=2) + "\n")
plan = write_live_pool_plan(
    seq=20, workers=96, leaf_servers=50, leaf_gpu0=10, leaf_gpu1=40,
    promotion_workers=8,
    reason="gpu0 least-queue feed: workers=96 leaves 10/40 even-spread",
    apply="next_iter",
)
print("live_pool", plan.workers, plan.leaf_gpu0, plan.leaf_gpu1)
# Incomplete overlapping shard — move aside so relaunch starts clean wave.
for name in ("iter_00001.jsonl",):
    shard = run / "shards" / name
    if shard.is_file() and shard.stat().st_size > 0:
        bak = shard.with_suffix(".jsonl.prev_least_queue")
        if bak.exists():
            bak.unlink()
        shard.rename(bak)
        print("moved_shard", bak)
print("prepared")
PY

pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/resource_watcher.py" 2>/dev/null || true
pkill -f "scripts/pure_rl_auto_progress.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/unattended_monitor.py" 2>/dev/null || true
pkill -f "scripts/_mirror_shard_progress.py" 2>/dev/null || true
sleep 3

printf '\n===== GPU0-LEAST-QUEUE RELAUNCH %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

for i in 1 2 3 4 5 6 7 8 9 10; do
  if "$PY" - <<'PY'
from poke_bot.remote_jobs import RemoteWorkerFarm
f=RemoteWorkerFarm(['elmo:8765','bert.local:8766'], timeout_s=8)
ok=False
try:
    for info in f.connect():
        kinds=set(info.job_kinds or ())
        print(info.endpoint, sorted(kinds), info.workers)
        if 'self_play' in kinds:
            ok=True
    f.close()
except Exception as e:
    print('wait_remotes', type(e).__name__, e)
    raise SystemExit(1)
raise SystemExit(0 if ok else 2)
PY
  then echo "remotes_ready"; break; fi
  echo "remotes_wait_$i"; sleep 6
done

nohup "$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile none \
  --log "$LOG" \
  --remote-worker-endpoints elmo:8765,bert.local:8766 \
  --auto-progress \
  -- \
  --base-checkpoint "$BASE_CKPT" \
  --iterations 1000 \
  --games-per-iter 2048 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  >"$RESTART_LOG" 2>&1 &
echo "launcher_bg=$!"
sleep 50
echo "=== restart log ==="
cat "$RESTART_LOG"
echo "=== procs ==="
ps -eo pid,etime,cmd | awk '/python.*scripts\/(train|launch)_pure_rl|resource_watcher|pure_rl_auto_progress/ && !/awk/ {print}'
echo "=== prove ==="
rg -n 'LEAST-QUEUE|full hardware|leaf-eval|gpu0_client_frac|self_play_pool|leaves_gpu0|Error|refuse' "$LOG" "$RESTART_LOG" 2>/dev/null | tail -40
