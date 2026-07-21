#!/usr/bin/env bash
# Fold thruput recovery (2048 / multi_env=1) with GPU0 OOM shrink + Blackwell feed.
# Keep 3080 Ti leaf footprint small; put more leaves/work on GPU1 only.
set -euo pipefail
cd /home/inzi/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T175340Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT="${RUN_DIR}/checkpoints/seed.pt"
LOG=outputs/logs/pure_rl_core.log
RESTART_LOG=outputs/logs/pure_rl_blackwell_feed_restart.log
BEFORE_JSON=outputs/state/gpu_util_before_blackwell_feed.json

[[ -f "$BASE_CKPT" ]] || { echo "missing seed: $BASE_CKPT" >&2; exit 1; }

"$PY" - <<'PY' || true
import json, subprocess, time
from pathlib import Path
Path("outputs/state").mkdir(parents=True, exist_ok=True)
samples = []
for _ in range(3):
    samples.append(subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True).strip())
    time.sleep(0.5)
Path("outputs/state/gpu_util_before_blackwell_feed.json").write_text(json.dumps({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "samples": samples,
    "root_cause": "GPU0 OOM/orphan leaves; Blackwell idle VRAM; prefer GPU1 leaves only",
    "target": {"leaf_gpu0": 10, "leaf_gpu1": 40, "workers": 96, "games": 2048},
}, indent=2) + "\n")
print("wrote before json")
PY

pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/resource_watcher.py" 2>/dev/null || true
pkill -f "scripts/pure_rl_auto_progress.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/_mirror_shard_progress.py" 2>/dev/null || true
# Sweep any stray leaf servers not owned by a live train.
sleep 2
"$PY" - <<'PY'
import os, signal, subprocess
from collections import defaultdict
children = defaultdict(list)
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        with open(f"/proc/{name}/stat") as f:
            ppid = int(f.read().split()[3])
        children[ppid].append(int(name))
    except Exception:
        pass
train = []
for line in subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True).splitlines():
    if "train_pure_rl.py" in line and "grep" not in line:
        train.append(int(line.split()[0]))
desc = set()
stack = list(train)
while stack:
    p = stack.pop()
    if p in desc:
        continue
    desc.add(p)
    stack.extend(children.get(p, []))
out = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
)
killed = 0
for line in out.strip().splitlines():
    if not line.strip():
        continue
    pid = int(line.strip())
    if pid in desc:
        continue
    try:
        os.kill(pid, signal.SIGKILL)
        killed += 1
    except ProcessLookupError:
        pass
print(f"swept_orphan_gpu_procs={killed} train_alive={train}")
PY
sleep 1

"$PY" - <<'PY'
import json, time
from pathlib import Path
from poke_bot.train import load_model_from_checkpoint
from poke_bot.pure_rl.model_profile import count_params
from poke_bot.pure_rl.hardware import full_hardware_profile
from poke_bot.live_pool import write_live_pool_plan

run = Path("outputs/pure_rl/pure_rl_core_overnight_20260716T175340Z")
ckpt = run / "checkpoints" / "seed.pt"
m = load_model_from_checkpoint(str(ckpt))
n = count_params(m)
print("seed_ok", n, "dense", getattr(m, "dense_card2vec", None))
assert 1_600_000 <= n <= 2_000_000, n

# Preview stripe: more GPU1 slots after interleaved prefix.
import os
os.environ["PURE_RL_LEAF_GPU0_REPLICAS"] = "10"
os.environ["PURE_RL_LEAF_GPU1_REPLICAS"] = "40"
hw = full_hardware_profile()
devs = hw.leaf_cuda_devices()
assert devs.count(0) == 10 and devs.count(1) == 40, (devs.count(0), devs.count(1))
print("stripe_ok", devs[:12], "...", "n0", devs.count(0), "n1", devs.count(1))

(run / "rebalance_state.json").write_text(json.dumps({
    "seq": 0,
    "sim_workers": 96,
    "local_share": 0.60,
    "min_workers": 64,
    "max_workers": 160,
    "ema_alpha": 0.35,
    "metrics": "wave_wall_clock_gps",
    "remote_batch": "chunked_rtt_amortize",
    "remote_demand": {
        "elmo_default": 20, "elmo_max": 40,
        "bert_default": 10, "bert_max": 20,
    },
    "note": "blackwell feed: GPU0=10 (OOM shrink) GPU1=40; multi_env=1 games=2048",
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}, indent=2) + "\n")
(run / "wave_boundary_knobs.json").write_text(json.dumps({
    "min_local_frac": 0.40,
    "max_remote_frac": 0.55,
    "prefer_local_frac": 0.60,
    "mid_iter_scheduler": True,
    "remote_dispatch_chunk": 128,
    "tqdm_inplace": True,
    "target_workers": 96,
    "max_workers": 160,
    "leaf_default": [10, 40],
    "leaf_max_total": 80,
    "elmo_default_workers": 20,
    "elmo_max_workers": 40,
    "bert_default_workers": 10,
    "bert_max_workers": 20,
    "multi_env_per_worker": 1,
    "games_per_iter": 2048,
    "note": "fold thruput+blackwell: shrink GPU0 leaves, feed GPU1; games=2048",
}, indent=2) + "\n")
plan = write_live_pool_plan(
    seq=8,
    workers=96,
    leaf_servers=50,
    leaf_gpu0=10,
    leaf_gpu1=40,
    promotion_workers=8,
    reason="blackwell-feed: GPU0=10 OOM-safe; GPU1=40 compute; 96w; games=2048",
    apply="next_iter",
)
print("live_pool", plan.workers, plan.leaf_gpu0, plan.leaf_gpu1, plan.leaf_servers)
Path("outputs/state/PURE_RL_BOUNDARY_RESTART_OWNER.md").write_text(
    "# owner blackwell-feed fold\n"
    "# keeps a07a45d4 thruput: multi_env=1 games=2048\n"
    "# overrides leaf split for 60e7ee1f GPU0 shrink: 10/40 (not 18/24)\n"
    "# do not bump GPU0 leaves without VRAM proof\n"
)
shard = run / "shards" / "iter_00000.jsonl"
if shard.is_file():
    bak = shard.with_suffix(".jsonl.prev_blackwell_feed")
    if bak.exists():
        bak.unlink()
    shard.rename(bak)
    print("moved_shard", bak)
print("prepared")
PY

printf '\n===== BLACKWELL-FEED RELAUNCH %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

export POKEBOT_PYTHON="$PY"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_LIVE_POOL=1
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
# GPU0 shrink + GPU1 feed (do not raise GPU0)
export PURE_RL_LEAF_GPU0_REPLICAS=10
export PURE_RL_LEAF_GPU1_REPLICAS=40
export PURE_RL_LEAF_COALESCE_MS=0
export PURE_RL_MID_ITER_SCHEDULER=1
export PURE_RL_REMOTE_DISPATCH_CHUNK=128
export PURE_RL_REBALANCE_PREFER_LOCAL_FRAC=0.60
export PURE_RL_REBALANCE_MIN_LOCAL_FRAC=0.40
export PURE_RL_REBALANCE_MAX_REMOTE_FRAC=0.55
export PURE_RL_REBALANCE_MAX_WORKERS=160
export PURE_RL_REBALANCE_MIN_WORKERS=64
export POKEBOT_LIVE_POOL_MAX_WORKERS=160
export POKEBOT_LIVE_POOL_MAX_LEAF_SERVERS=80
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
sleep 45
echo "=== restart log ==="
cat "$RESTART_LOG"
echo "=== procs ==="
ps -eo pid,etime,cmd | awk '/python.*scripts\/(train|launch)_pure_rl|resource_watcher|pure_rl_auto_progress/ && !/awk/ {print}'
echo "=== prove log ==="
rg -n 'BLACKWELL-FEED|full hardware|leaf-eval|leaves_gpu0|self_play_pool|games-per-iter|Error|refuse' "$LOG" "$RESTART_LOG" 2>/dev/null | tail -40
echo "=== leaf counts / util samples ==="
"$PY" - <<'PY'
import subprocess, time
def counts():
    out = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader",
    ], text=True)
    g0, g1 = "72bf89ff", "79cf504f"
    c0 = c1 = m0 = m1 = 0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        u, _p, m = [x.strip() for x in line.split(",")]
        mb = int(m.split()[0]) if m.split() else 0
        if g0 in u:
            c0 += 1; m0 += mb
        elif g1 in u:
            c1 += 1; m1 += mb
    return c0, m0, c1, m1
print("leaves", counts())
for i in range(8):
    row = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader",
    ], text=True).strip()
    print(f"sample{i}", row.replace("\n", " | "))
    time.sleep(1.5)
print("progress:", open("outputs/logs/pure_rl_core.progress.status").read().strip()[:240])
PY
