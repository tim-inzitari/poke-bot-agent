#!/usr/bin/env bash
set -u

state_unit="pokebot-state-core-bootstrap-v30.service"
result_file="/home/inzi/poke-bot-agent/outputs/train/state_core_top_ladder_5day_full_20260719_result.json"
patterns='ptcg-replay-|replay-growth|pokebot-top-ladder-hotstart|pokebot-activate-top-ladder'

log() {
  printf '%s [gpu-owner] %s\n' "$(date --iso-8601=seconds)" "$*"
}

while true; do
  mapfile -t conflicts < <(
    systemctl --user --no-legend --plain list-units --type=service --state=active \
      | awk '{print $1}' \
      | grep -E "$patterns" || true
  )
  if ((${#conflicts[@]})); then
    log "terminating obsolete temporal units: ${conflicts[*]}"
    for unit in "${conflicts[@]}"; do
      cgroup="$(systemctl --user show "$unit" -p ControlGroup --value 2>/dev/null || true)"
      procs="/sys/fs/cgroup${cgroup}/cgroup.procs"
      if [[ -n "$cgroup" && -r "$procs" ]]; then
        while read -r pid; do
          [[ "$pid" =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null || true
        done < "$procs"
      fi
    done
  fi

  if [[ ! -f "$result_file" ]] && ! systemctl --user is-active --quiet "$state_unit"; then
    log "restoring $state_unit"
    systemctl --user unmask --runtime "$state_unit" >/dev/null 2>&1 || true
    systemctl --user unmask "$state_unit" >/dev/null 2>&1 || true
    systemctl --user enable "$state_unit" >/dev/null 2>&1 || true
    systemctl --user start "$state_unit" || true
  fi
  sleep 3
done
