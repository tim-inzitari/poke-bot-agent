#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---check}"
PYTHON="/Users/tsinzitari/workspace/poke-bot-agent/.venv/bin/python"
LIVE_ROOT="/Users/tsinzitari/pokebot-dashboard/v1"
INZI="inzi@192.168.1.151"

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
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/inzi/poke-bot-agent/scripts/dashboard_snapshot.py"
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/inzi/poke-bot-agent-deployments/state-core-v1/scripts/dashboard_snapshot.py"
rsync -a scripts/dashboard_snapshot.py \
  "${INZI}:/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v41-specialist-matchup-runtime/scripts/dashboard_snapshot.py"

launchctl kickstart -k "gui/$(id -u)/com.pokebot.training-dashboard"
echo "DASHBOARD_SYNC_APPLIED"
