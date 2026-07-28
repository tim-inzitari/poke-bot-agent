#!/usr/bin/env bash
set -euo pipefail

STATUS="/mnt/Main/main/poke-feature-refresh-20260721/data/bootstrap/expert-latest20-additive/elmo.status.json"
SOURCE="/mnt/Main/main/poke-native-expert-featurizer"
OUT="/mnt/Main/main/poke-native-expert-featurizer/output"
ARCHIVE="/mnt/Main/main/poke-feature-refresh-20260721/data/episodes/raw/pokemon-tcg-ai-battle-episodes-2026-07-15.zip"

while ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("state") == "complete" else 1)' "$STATUS"; do
  sleep 20
done

mkdir -p "$OUT"
sudo -n docker build -f "$SOURCE/Dockerfile.x86_64" \
  -t pokebot-native-expert-featurizer:x86_64 "$SOURCE"
container="$(sudo -n docker create pokebot-native-expert-featurizer:x86_64 true)"
trap 'sudo -n docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
sudo -n docker cp "$container:/usr/local/bin/replay_ingest_probe" \
  "$OUT/replay_ingest_probe.x86_64"
sudo -n docker run --rm \
  -v "$(dirname "$ARCHIVE"):/data:ro" \
  pokebot-native-expert-featurizer:x86_64 \
  /usr/local/bin/replay_ingest_probe "/data/$(basename "$ARCHIVE")" 18
