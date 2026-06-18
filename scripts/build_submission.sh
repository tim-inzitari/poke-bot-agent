#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f submission/main.py
test -f submission/deck.csv
test -d submission/cg

mkdir -p dist
tar -czf dist/submission.tar.gz -C submission main.py deck.csv cg
tar -tzf dist/submission.tar.gz
echo "built dist/submission.tar.gz"
