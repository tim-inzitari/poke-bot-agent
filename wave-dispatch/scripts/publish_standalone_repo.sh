#!/usr/bin/env bash
# Publish wave_dispatch as its own GitHub repository.
#
# This cloud agent cannot create repos under tim-inzitari (token 403).
# Run locally with your GitHub credentials:
#
#   bash wave-dispatch/scripts/publish_standalone_repo.sh
#   # or:
#   bash wave-dispatch/scripts/publish_standalone_repo.sh tim-inzitari/wave-dispatch --public
#
set -euo pipefail

TARGET="${1:-tim-inzitari/wave-dispatch}"
VISIBILITY="${2:---private}"  # --private or --public
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/wave-dispatch-publish.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "Cloning lib/wave-dispatch history from poke-bot-agent..."
git clone --branch lib/wave-dispatch --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git "$WORK/repo"
cd "$WORK/repo"
git checkout -B main

echo "Creating GitHub repo ${TARGET} (${VISIBILITY})..."
gh repo create "$TARGET" "$VISIBILITY" --description \
  "Standalone C++ LAN job-dispatch library for multi-machine RL collect (Python bindings)" \
  --source=. --remote=origin --push

git push origin v0.3.1 2>/dev/null || git push origin --tags

echo
echo "Done: https://github.com/${TARGET}"
echo "Clone: git clone https://github.com/${TARGET}.git"
echo "Pip:   pip install git+https://github.com/${TARGET}.git"
