#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f submission/main.py
test -f submission/deck.csv
test -d submission/cg

CG_LIB="kaggle/input/cg-lib/cg/libcg.so"
if [ ! -f "$CG_LIB" ]; then
  echo "missing $CG_LIB; run scripts/download-kaggle-inputs.sh first" >&2
  exit 1
fi

mkdir -p dist
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp submission/main.py submission/deck.csv "$STAGING/"
cp -r submission/cg "$STAGING/"
cp "$CG_LIB" "$STAGING/cg/libcg.so"

tar -czf dist/submission.tar.gz -C "$STAGING" main.py deck.csv cg
tar -tzf dist/submission.tar.gz
echo "built dist/submission.tar.gz"
