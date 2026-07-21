#!/usr/bin/env bash
set -u

root="/home/inzi/poke-bot-agent-deployments/state-core-v1"
python="/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
log_file="/home/inzi/poke-bot-agent/outputs/logs/bootstrap.log"
supervisor_log="/home/inzi/poke-bot-agent/outputs/logs/bootstrap.supervisor.log"
result_file="/home/inzi/poke-bot-agent/outputs/train/state_core_top_ladder_5day_full_20260719_result.json"
state_pattern='train_bootstrap.py.*--run-name state_core_top_ladder_5day_full_20260719'
child_pid=""

log() {
  printf '%s [state-supervisor] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$supervisor_log"
}

stop_temporal_conflicts() {
  mapfile -t pids < <(
    pgrep -f 'run_top_ladder_hotstart.py|activate_top_ladder_hotstart.py|train_bootstrap.py.*top_ladder_core_hotstart_10day_20260719' || true
  )
  for pid in "${pids[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$command" == *state_core_top_ladder_5day_full_20260719* ]] && continue
    log "terminating obsolete temporal pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  done
}

shutdown() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap shutdown TERM INT HUP

mkdir -p "$(dirname "$log_file")"
while [[ ! -f "$result_file" ]]; do
  stop_temporal_conflicts
  existing="$(pgrep -fo "$state_pattern" || true)"
  if [[ -n "$existing" ]]; then
    child_pid="$existing"
    log "adopting existing state bootstrap pid=$child_pid"
  else
    log "launching five-day state bootstrap (30 max epochs, patience 4)"
    (
      cd "$root" || exit 1
      export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
      exec "$python" -u scripts/train_bootstrap.py \
        --jsonl /home/inzi/poke-bot-agent/data/bootstrap/top_ladder_all_2026-07-03_to_2026-07-07.jsonl \
        --archetype top-ladder-core \
        --run-name state_core_top_ladder_5day_full_20260719 \
        --model-profile pure-rl \
        --epochs 30 \
        --lr 5e-5 \
        --games-per-batch 16 \
        --max-decisions-per-batch 2048 \
        --val-frac 0.10 \
        --split-by-episode \
        --patience 4 \
        --aux-loss-weight 0 \
        --opp-hand-loss-weight 0 \
        --opp-remainder-loss-weight 0 \
        --min-usable-record-frac 0.98 \
        --min-decisions 2400000 \
        --seed 20260725 \
        --resume auto \
        --init-checkpoint /home/inzi/poke-bot-agent/outputs/checkpoints/state_core_top_ladder_5day_20260719.latest.pt \
        >> "$log_file" 2>&1
    ) &
    child_pid=$!
    log "state bootstrap pid=$child_pid"
  fi

  while kill -0 "$child_pid" 2>/dev/null; do
    stop_temporal_conflicts
    rss_kb="$(awk '/VmRSS:/{print $2}' "/proc/$child_pid/status" 2>/dev/null || echo 0)"
    avail_kb="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
    if ((rss_kb > 83886080 || avail_kb < 20971520)); then
      log "memory guard stopping pid=$child_pid rss_gib=$((rss_kb/1048576)) available_gib=$((avail_kb/1048576))"
      kill -TERM "$child_pid" 2>/dev/null || true
      break
    fi
    sleep 3
  done
  wait "$child_pid" 2>/dev/null
  code=$?
  child_pid=""
  [[ -f "$result_file" ]] && break
  log "bootstrap exited code=$code without result; restarting in 5s"
  sleep 5
done
log "bootstrap result present; supervisor complete"
