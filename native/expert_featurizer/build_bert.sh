#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/native/expert-featurizer"
mkdir -p "$OUT"

SIMDJSON="$(brew --prefix simdjson)"
LIBARCHIVE="$(brew --prefix libarchive)"

clang++ -std=c++20 -O3 -DNDEBUG -pthread \
  -I"$SIMDJSON/include" -I"$LIBARCHIVE/include" \
  "$ROOT/native/expert_featurizer/replay_ingest_probe.cpp" \
  -L"$SIMDJSON/lib" -L"$LIBARCHIVE/lib" \
  -Wl,-rpath,"$SIMDJSON/lib" -Wl,-rpath,"$LIBARCHIVE/lib" \
  -lsimdjson -larchive \
  -o "$OUT/replay_ingest_probe"

echo "$OUT/replay_ingest_probe"
