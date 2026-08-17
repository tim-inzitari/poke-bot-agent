#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---check}"
PYTHON="/Users/example/workspace/poke-bot-agent/.venv/bin/python"
LIVE_ROOT="/Users/example/pokebot-dashboard/v1"
INZI="trainer@inzi"
ELMO="admin@elmo"

if [[ "${MODE}" != "--check" && "${MODE}" != "--apply" ]]; then
  echo "usage: $0 [--check|--apply]" >&2
  exit 2
fi

cd "${ROOT}"
"${PYTHON}" -m py_compile scripts/dashboard_snapshot.py dashboard/lan/server.py
node -e '
const fs = require("fs");
const source = fs.readFileSync("dashboard/lan/index.html", "utf8");
const match = source.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error("dashboard script block missing");
new Function(match[1]);
'
PYTHONPATH=. "${PYTHON}" -m pytest -q tests/test_dashboard_regressions.py

if [[ "${MODE}" == "--check" ]]; then
  echo "DASHBOARD_SYNC_CHECK_OK"
  exit 0
fi

rsync -a dashboard/lan/index.html "${LIVE_ROOT}/index.html"
rsync -a dashboard/lan/server.py "${LIVE_ROOT}/server.py"
rsync -a scripts/fleet_host_snapshot.py "${LIVE_ROOT}/fleet_host_snapshot.py"
rsync -a ops/current_goal_requirements.json \
  "${LIVE_ROOT}/current_goal_requirements.json"
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/pokebot/poke-bot-agent/scripts/dashboard_snapshot.py"
rsync -a scripts/fleet_host_snapshot.py \
  "${INZI}:/home/pokebot/poke-bot-agent/scripts/fleet_host_snapshot.py"
rsync -a scripts/fleet_host_snapshot.py \
  "${ELMO}:/mnt/Main/Elmo/poke-bot-agent/dashboard/fleet_host_snapshot.py"
rsync -a ops/current_goal_requirements.json \
  "${INZI}:/home/pokebot/poke-bot-agent/ops/current_goal_requirements.json"
rsync -a state/alakazam-new-list-direct-policy-r241.json \
  "${INZI}:/home/pokebot/poke-bot-agent/state/alakazam-new-list-direct-policy-r241.json"
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/pokebot/poke-bot-agent-deployments/state-core-v1/scripts/dashboard_snapshot.py"
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/pokebot/poke-bot-agent-deployments/pure-rl-resident-v41-specialist-matchup-runtime/scripts/dashboard_snapshot.py"
SELECTOR_RUNTIME_ROOT="$(
  ssh -o BatchMode=yes -o ConnectTimeout=25 \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "${INZI}" \
    "sed -n 's/^POKEBOT_SPECIALIST_RUNTIME_ROOT=//p' /home/pokebot/.config/pokebot/specialist_runtime.env"
)"
case "${SELECTOR_RUNTIME_ROOT}" in
  /home/pokebot/poke-bot-agent|/home/pokebot/poke-bot-agent-deployments/*)
    ;;
  *)
    echo "selector-owned specialist runtime root is invalid" >&2
    exit 1
    ;;
esac
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:${SELECTOR_RUNTIME_ROOT}/scripts/dashboard_snapshot.py"
rsync -a scripts/fleet_host_snapshot.py \
  "${INZI}:${SELECTOR_RUNTIME_ROOT}/scripts/fleet_host_snapshot.py"
rsync -a ops/current_goal_requirements.json \
  "${INZI}:${SELECTOR_RUNTIME_ROOT}/ops/current_goal_requirements.json"

launchctl kickstart -k "gui/$(id -u)/com.pokebot.training-dashboard"
echo "DASHBOARD_SYNC_APPLIED"
