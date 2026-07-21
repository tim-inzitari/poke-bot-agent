#!/usr/bin/env bash
# One-shot boundary relaunch with bottleneck-aware rebalance (prefer=local).
set -euo pipefail
cd /home/inzi/poke-bot-agent
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"
RUN_NAME=pure_rl_core_overnight_20260716T172245Z
RUN_DIR=outputs/pure_rl/${RUN_NAME}
BASE_CKPT=outputs/pure_rl/pure_rl_core_overnight_20260716T170537Z/checkpoints/seed.pt
LOG=outputs/logs/pure_rl_core.log
RESTART_LOG=outputs/logs/pure_rl_boundary_rebalance_restart.log

# Stop prior trainers for this run only.
pkill -f "scripts/train_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
pkill -f "scripts/launch_pure_rl.py.*${RUN_NAME}" 2>/dev/null || true
sleep 2

"$PY" - <<PY
import json, time
from pathlib import Path
from poke_bot.train import load_model_from_checkpoint
from poke_bot.pure_rl.model_profile import count_params

run = Path("${RUN_DIR}")
m = load_model_from_checkpoint("${BASE_CKPT}")
print("seed_ok", count_params(m), "dense", getattr(m, "dense_card2vec", None))

state = {
    "seq": 0,
    "sim_workers": 72,
    "local_gps": 12.6,
    "remote_gps": 18.4,
    "local_share": 0.55,
    "min_workers": 32,
    "max_workers": 96,
    "ema_alpha": 0.35,
    "history": [],
    "last_decision": {},
    "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
(run / "rebalance_state.json").write_text(json.dumps(state, indent=2) + "\n")
knobs = {
    "min_local_frac": 0.40,
    "max_remote_frac": 0.60,
    "prefer_local_frac": 0.55,
    "max_gps_ratio": 3.0,
    "early_elapsed_floor": 0.35,
    "tqdm_inplace": True,
    "tqdm_mininterval": 1.5,
    "start_wave": "",
    "note": "Bottleneck controller live; prefer=local; corrected wave GPS; soft floors rails.",
}
(run / "wave_boundary_knobs.json").write_text(json.dumps(knobs, indent=2) + "\n")
shard = run / "shards" / "iter_00000.jsonl"
if shard.is_file():
    bak = shard.with_suffix(".jsonl.prev_incomplete")
    shard.rename(bak)
    print("moved_shard", bak)
print("prepared")
PY

printf '\n===== BOUNDARY RESTART-2 bottleneck-rebalance %s =====\n' "$(date -Is)" >>"$LOG"
: > outputs/logs/pure_rl_core.progress.log
: > outputs/logs/pure_rl_core.progress.status

export POKEBOT_PYTHON="$PY"
export PURE_RL_ADAPTIVE_REBALANCE=1
export PURE_RL_SIM_WORKERS=72
export PURE_RL_GAMES_IN_FLIGHT=96
export PURE_RL_LEAF_GPU0_REPLICAS=18
export PURE_RL_LEAF_GPU1_REPLICAS=24
export PURE_RL_REBALANCE_PREFER_LOCAL_FRAC=0.55
export PURE_RL_REBALANCE_MIN_LOCAL_FRAC=0.40
export PURE_RL_REBALANCE_MAX_REMOTE_FRAC=0.60
export PURE_RL_REBALANCE_MAX_GPS_RATIO=3.0
export PURE_RL_REBALANCE_EARLY_ELAPSED_FLOOR=0.35
export PURE_RL_TQDM_INPLACE=1
export POKEBOT_LEAF_SERVER_COALESCE_MS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0

nohup "$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile none \
  --log "$LOG" \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  -- \
  --base-checkpoint "$BASE_CKPT" \
  --iterations 1000 \
  --games-per-iter 2048 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  >"$RESTART_LOG" 2>&1 &
echo "launcher_bg=$!"
sleep 18
echo "=== procs ==="
ps -eo pid,etime,cmd | awk '/python.*scripts\/(train|launch)_pure_rl/ && !/awk/ {print}'
echo "=== restart log ==="
cat "$RESTART_LOG"
echo "=== prove ==="
rg -n 'BOUNDARY RESTART-2|prefer=local|soft_floor|adaptive_rebalance|split self_play|wave_boundary|full hardware|loaded checkpoint|Error|refuse|workers=' "$LOG" | tail -40
