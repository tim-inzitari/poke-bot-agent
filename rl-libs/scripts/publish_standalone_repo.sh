#!/usr/bin/env bash
# Publish rl-libs as its own GitHub repository.
#
# This cloud agent cannot create repos under tim-inzitari (token 403).
# Run locally with your GitHub credentials:
#
#   bash rl-libs/scripts/publish_standalone_repo.sh
#   # or:
#   bash rl-libs/scripts/publish_standalone_repo.sh tim-inzitari/rl-libs --public
#
set -euo pipefail

TARGET="${1:-tim-inzitari/rl-libs}"
VISIBILITY="${2:---private}"  # --private or --public
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rl-libs-publish.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "Cloning lib/rl-libs history from poke-bot-agent..."
git clone --branch lib/rl-libs --single-branch \
  https://github.com/tim-inzitari/poke-bot-agent.git "$WORK/repo"
cd "$WORK/repo"
git checkout -B main

echo "Creating GitHub repo ${TARGET} (${VISIBILITY})..."
gh repo create "$TARGET" "$VISIBILITY" --description \
  "Generalized C++/Python RL infrastructure: IO, leaf IPC, process pools, eval, checkpoints" \
  --source=. --remote=origin --push

git push origin v0.1.1 2>/dev/null || git push origin --tags

echo
echo "Done: https://github.com/${TARGET}"
echo "Clone: git clone https://github.com/${TARGET}.git"
echo "Pip:   pip install git+https://github.com/${TARGET}.git"
