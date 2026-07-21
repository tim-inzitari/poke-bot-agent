#!/usr/bin/env bash
# One-shot: restore overnight pure-RL core (schema v5 / ladder mix).
# Do NOT use pkill -f with patterns that match this script's argv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${POKEBOT_PYTHON:-/home/inzi/miniconda3/envs/poke-bot-agent/bin/python}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_WORKER_CPU_ONLY=1
export POKEBOT_SKIP_GAME_ACCURACY=1
export PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1
export PURE_RL_SIM_WORKERS=48
export PURE_RL_PER_WORKER_RSS_GB=1.3

"$PY" - <<'PY'
import os, signal, time, subprocess
from pathlib import Path

def procs():
    out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        yield pid, args

my = os.getpid()
needles = (
    "scripts/launch_pure_rl.py",
    "scripts/train_pure_rl.py",
    "scripts/resource_watcher.py",
    "unattended_monitor.py",
)
targets = []
for pid, args in procs():
    if pid == my:
        continue
    # Only real python children — never bash wrappers that mention the scripts.
    if "/python" not in args and "python3" not in args.split()[0:1]:
        # allow env python path forms
        if "bin/python" not in args:
            continue
    if any(n in args for n in needles):
        targets.append(pid)
        print(f"TERM {pid} {args[:160]}")
for pid in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in targets:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
print(f"cleared={len(targets)}")

root = Path("/home/inzi/poke-bot-agent")
# Remove empty failed relaunch dir (no loop_state / bad base)
bad = root / "outputs/pure_rl/pure_rl_core_overnight_20260717T123500Z"
if bad.is_dir() and not (bad / "loop_state.json").is_file():
    import shutil
    shutil.rmtree(bad)
    print("removed empty failed run", bad)

ladder = root / "outputs/pure_rl/pure_rl_core_ladder_v5_overnight_20260716T225411Z"
for name in ("MONITOR_STOP_REQUESTED.json",):
    p = ladder / name
    if p.is_file():
        p.unlink()
        print("removed", p)
PY

CKPT="$ROOT/outputs/pure_rl/pure_rl_core_v5_20260716T213552Z/checkpoints/seed.pt"
test -f "$CKPT"

# Prefer remote-canary weights if present (schema-5 + pure_rl flag + trained)
CANARY="$ROOT/outputs/pure_rl/pure_rl_core_remote_canary_20260716T2247Z/checkpoints/iter_00001.pt"
if [[ -f "$CANARY" ]]; then
  CKPT="$CANARY"
fi

UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="pure_rl_core_overnight_${UTC}"
LOG="$ROOT/outputs/logs/pure_rl_core.log"
PROG="$ROOT/outputs/logs/pure_rl_core.progress.log"
LAUNCH_LOG="$ROOT/outputs/logs/pure_rl_core_launcher.stdout.log"

# Rotate watch paths
for f in "$LOG" "$PROG" "$LAUNCH_LOG"; do
  if [[ -f "$f" ]]; then
    mv "$f" "${f}.pre_relaunch_${UTC}"
  fi
done
: > "$ROOT/outputs/logs/pure_rl_core.progress.status"

cat > "$ROOT/outputs/logs/SINGLE_RELAUNCH_OWNER.md" <<EOF
# SINGLE RELAUNCH OWNER — ACTIVE

**Status**: ACTIVE $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Owner**: scripts/_relaunch_overnight_now.sh

## Why it was down
- Original \`pure_rl_core_overnight_20260716T175340Z\` died ~Jul16 16:47 local; run dir
  was quarantined and lacked \`loop_state.json\`. Checkpoints are feature_schema=4
  and missing top-level \`extra.pure_rl\` — **cannot** resume on current runtime
  (FEATURE_SCHEMA=5).
- Ladder_v5 overnight (\`…225411Z\`) was the schema-5 successor; monitor stopped it
  for \`no_progress_20.1m\` near end of iter0 collect (~19:56 local). No iter ckpt
  saved. Crash monitor last \`ALERT train DEAD\` 2026-07-16T23:57Z and never
  relaunched.

## Relaunch
- New run: \`$RUN_NAME\`
- Base: \`$CKPT\`
- Mode core, games 2048, heldout 200, gate 0.70, multi-env 1
- Remotes: 192.168.1.143:8765,bert.local:8766
- stall-minutes=40 (avoid false stop on remote straggle)
- PUBLIC_MIX_LOCAL_ONLY=1, workers=48, PER_WORKER_RSS=1.3
EOF

nohup "$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --run-name "$RUN_NAME" \
  --preflight-profile none \
  --stall-minutes 40 \
  --log outputs/logs/pure_rl_core.log \
  --remote-worker-endpoints 192.168.1.143:8765,bert.local:8766 \
  --multi-env-per-worker 1 \
  --leaf-coalesce-ms 0.0 \
  -- \
  --base-checkpoint "$CKPT" \
  --iterations 1000 \
  --games-per-iter 2048 \
  --heldout-games 200 \
  --gate-wr 0.70 \
  > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$ROOT/outputs/state/PURE_RL_CORE_LAUNCHER.pid"
echo "LAUNCHER_PID=$LAUNCH_PID RUN=$RUN_NAME CKPT=$CKPT"

# Arm crash monitor (alert-only; does not relaunch)
nohup bash "$ROOT/outputs/state/TRAIN_CRASH_MONITOR.sh" \
  >> "$ROOT/outputs/logs/train_crash_monitor.log" 2>&1 &
echo "CRASH_MONITOR_PID=$!"
echo $! > "$ROOT/outputs/state/TRAIN_CRASH_MONITOR.pid"

# Brief smoke: wait for PURE_RL_PIDS + collecting
for i in $(seq 1 60); do
  if grep -q 'PURE_RL_PIDS' "$LAUNCH_LOG" 2>/dev/null; then
    break
  fi
  sleep 1
done
grep 'PURE_RL_PIDS' "$LAUNCH_LOG" || true
sleep 25
if ! "$PY" - <<'PY'
import subprocess, sys
out = subprocess.check_output(["ps", "-eo", "args="], text=True)
ok = any("scripts/train_pure_rl.py" in line and "bin/python" in line for line in out.splitlines())
sys.exit(0 if ok else 1)
PY
then
  echo "FAIL: train_pure_rl not alive after launch" >&2
  tail -80 "$LAUNCH_LOG" >&2 || true
  tail -80 "$PROG" >&2 || true
  exit 3
fi

echo "=== status ==="
cat "$ROOT/outputs/logs/pure_rl_core.progress.status" || true
echo "=== core log tail ==="
tail -40 "$LOG" || true
echo "=== progress tail ==="
tail -c 2500 "$PROG" | tr '\r' '\n' | tail -25 || true
echo RELAUNCH_OK
