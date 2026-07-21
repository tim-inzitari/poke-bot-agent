#!/usr/bin/env bash
# Fix GPU0 starvation: stripe leaves + full sim_workers fan-out with remotes.
# GPU0 leaves were up but received no self_play traffic (24 clients → first 24
# contiguous GPU1 servers). Kill justified: "receiving traffic" check failed.
set -euo pipefail
cd /home/inzi/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T175340Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT="${RUN_DIR}/checkpoints/seed.pt"
LOG=outputs/logs/pure_rl_core.log
RESTART_LOG=outputs/logs/pure_rl_gpu_feed_restart.log
BEFORE_JSON=outputs/state/gpu_util_before_feed_fix.json

[[ -f "$BASE_CKPT" ]] || { echo "missing seed: $BASE_CKPT" >&2; exit 1; }

# Capture before numbers (best-effort).
"$PY" - <<'PY' || true
import json, subprocess, time
from pathlib import Path
out = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
    "--format=csv,noheader,nounits",
], text=True)
samples = []
for _ in range(4):
    row = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    samples.append(row)
    time.sleep(1.0)
Path("outputs/state").mkdir(parents=True, exist_ok=True)
Path("outputs/state/gpu_util_before_feed_fix.json").write_text(json.dumps({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "static": out.strip().splitlines(),
    "samples": samples,
    "root_cause": "self_play_pool=24 pinned to contiguous GPU1 leaf servers 0-23; GPU0 leaves idle",
}, indent=2) + "\n")
print("wrote before json")
PY

pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/resource_watcher.py" 2>/dev/null || true
pkill -f "scripts/pure_rl_auto_progress.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/_mirror_shard_progress.py" 2>/dev/null || true
sleep 3

"$PY" - <<PY
import json, time
from pathlib import Path
from poke_bot.train import load_model_from_checkpoint
from poke_bot.pure_rl.model_profile import count_params
from poke_bot.pure_rl.hardware import full_hardware_profile
from poke_bot.live_pool import write_live_pool_plan

run = Path("${RUN_DIR}")
m = load_model_from_checkpoint("${BASE_CKPT}")
n = count_params(m)
print("seed_ok", n)
assert 1_600_000 <= n <= 2_500_000, n
hw = full_hardware_profile()
devs = hw.leaf_cuda_devices()
assert set(devs[:24]) == {0, 1}, devs[:24]
print("stripe_ok", devs[:8], "n0", devs.count(0), "n1", devs.count(1))

(run / "rebalance_state.json").write_text(json.dumps({
    "seq": 0, "sim_workers": 96, "local_share": 0.60,
    "min_workers": 64, "max_workers": 160, "max_total_workers": 10000,
    "ema_alpha": 0.35,
    "metrics": "wave_wall_clock_gps", "remote_batch": "chunked_rtt_amortize",
    "remote_demand": {
        "elmo_default": 20, "elmo_max": 40,
        "bert_default": 10, "bert_max": 20,
    },
    "note": "gpu-feed fix: stripe leaves + full 96 local fanout; max_total=10000 never binds",
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}, indent=2) + "\n")
(run / "wave_boundary_knobs.json").write_text(json.dumps({
    "min_local_frac": 0.40, "max_remote_frac": 0.55, "prefer_local_frac": 0.60,
    "mid_iter_scheduler": True, "remote_dispatch_chunk": 128, "tqdm_inplace": True,
    "target_workers": 96, "max_workers": 160, "max_total_workers": 10000,
    "leaf_default": [10, 40], "leaf_max_gpu0": 12, "leaf_max_gpu1": 48,
    "leaf_max_total": 60,
    "elmo_default_workers": 20, "elmo_max_workers": 40,
    "bert_default_workers": 10, "bert_max_workers": 20,
    "note": "3080-safe stripe: GPU0=10 (max12); BW=40 (max48); prefer_local 0.60",
}, indent=2) + "\n")
plan = write_live_pool_plan(
    seq=4, workers=96, leaf_servers=50, leaf_gpu0=10, leaf_gpu1=40,
    promotion_workers=8,
    reason="3080-safe gpu-feed: stripe+full fanout; GPU0=10 (≤12) GPU1=40",
    apply="next_iter",
)
print("live_pool", plan.workers, plan.leaf_gpu0, plan.leaf_gpu1, plan.leaf_servers)
shard = run / "shards" / "iter_00000.jsonl"
if shard.is_file():
    bak = shard.with_suffix(".jsonl.prev_gpu_feed")
    if bak.exists():
        bak.unlink()
    shard.rename(bak)
    print("moved_shard", bak)
print("prepared")
PY

printf '\n===== GPU-FEED RELAUNCH %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

export POKEBOT_PYTHON="$PY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_LIVE_POOL=1
export POKEBOT_WORKER_CPU_ONLY=1
export POKEBOT_REMOTE_ENDPOINT_WEIGHTS="192.168.1.143:8765=1.0,bert.local:8766=0.40"
export PURE_RL_SIM_WORKERS=96
export PURE_RL_GAMES_IN_FLIGHT=96
# 3080 Ti 12GB: CUBLAS OOM at 18–20 leaves — steady 10, hard max 12.
export PURE_RL_LEAF_GPU0_REPLICAS=10
export PURE_RL_LEAF_GPU1_REPLICAS=40
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU0=12
export POKEBOT_LIVE_POOL_MAX_LEAF_GPU1=48
export PURE_RL_LEAF_GPU0_FRAC=0.20
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
# N=1: full OS fan-out onto striped leaves (N=4 starved GPU0 via sticky binds).
export POKEBOT_MULTI_ENV=0
export POKEBOT_MULTI_ENV_PER_WORKER=1
export PURE_RL_MULTI_ENV=0
export PURE_RL_MULTI_ENV_PER_WORKER=1

# Wait briefly for remotes (sibling may be redeploying).
for i in 1 2 3 4 5 6 7 8 9 10; do
  if "$PY" - <<'PY'
from poke_bot.remote_jobs import RemoteWorkerFarm
f=RemoteWorkerFarm(['192.168.1.143:8765','bert.local:8766'], timeout_s=8)
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
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  --auto-progress \
  -- \
  --base-checkpoint "$BASE_CKPT" \
  --iterations 1000 \
  --games-per-iter 4096 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  >"$RESTART_LOG" 2>&1 &
echo "launcher_bg=$!"
sleep 45
echo "=== restart log ==="
cat "$RESTART_LOG"
echo "=== procs ==="
ps -eo pid,etime,cmd | awk '/python.*scripts\/(train|launch)_pure_rl|resource_watcher|pure_rl_auto_progress/ && !/awk/ {print}'
echo "=== prove ==="
rg -n 'GPU-FEED|full hardware|leaf-eval|self_play_pool|leaves_gpu0|mid_iter_rebalance|scheduled_dispatch|Error|refuse|remote=' "$LOG" "$RESTART_LOG" 2>/dev/null | tail -80
