#!/usr/bin/env bash
set -uo pipefail

readonly repo="/home/inzi/poke-bot-agent-deployments/state-core-v1"
readonly python="/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
readonly log_file="/home/inzi/poke-bot-agent/outputs/logs/bootstrap.log"
readonly supervisor_log="/home/inzi/poke-bot-agent/outputs/logs/bootstrap.supervisor.log"
readonly result_file="/home/inzi/poke-bot-agent/outputs/train/state_core_top_ladder_5day_full_20260719_result.json"
child_pid=""

log() {
  printf '%s [root-state-owner] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$supervisor_log"
}

stop_obsolete_temporal_jobs() {
  local pid command
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$command" == *state_core_top_ladder_5day_full_20260719* ]] && continue
    log "terminating obsolete temporal process pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
  done < <(
    pgrep -f \
      '[r]un_top_ladder_hotstart.py|[a]ctivate_top_ladder_hotstart.py|[t]rain_bootstrap.py.*top_ladder_core_hotstart_10day_20260719' \
      || true
  )
}

shutdown() {
  if [[ -n "$child_pid" ]]; then
    log "shutdown requested; forwarding TERM to trainer pid=$child_pid"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 0
}
trap shutdown TERM INT HUP

install -d -m 0775 -o inzi -g inzi "$(dirname "$log_file")" "$(dirname "$result_file")"
touch "$log_file" "$supervisor_log"
chgrp inzi "$log_file" "$supervisor_log"
chmod 0664 "$log_file" "$supervisor_log"

while [[ ! -f "$result_file" ]]; do
  stop_obsolete_temporal_jobs
  log "launching five-day state bootstrap (30 max epochs, patience 4)"
  (
    cd "$repo" || exit 1
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=1
    export HOME=/home/inzi
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
  log "trainer pid=$child_pid"

  while kill -0 "$child_pid" 2>/dev/null; do
    stop_obsolete_temporal_jobs
    available_kb="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
    if ((available_kb < 20971520)); then
      log "host-memory guard stopping trainer; available_gib=$((available_kb / 1048576))"
      kill -TERM "$child_pid" 2>/dev/null || true
      break
    fi
    sleep 3
  done

  wait "$child_pid" 2>/dev/null
  exit_code=$?
  child_pid=""
  [[ -f "$result_file" ]] && break
  log "trainer exited code=$exit_code without result; restarting in 5s"
  sleep 5
done

log "bootstrap result present; supervisor complete"
