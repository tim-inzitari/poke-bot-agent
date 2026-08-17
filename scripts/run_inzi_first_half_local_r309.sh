#!/bin/sh
set -eu

manifest=/home/pokebot/alakazam-r309-work/raw_expert_corpus_manifest.json
work=/home/pokebot/alakazam-r309-work
raw=$work/raw-first-half
mkdir -p "$raw"

python3 - "$manifest" <<'PY' | xargs -d '\n' -P8 -I{} cp -p "{}" /home/pokebot/alakazam-r309-work/raw-first-half/
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
for index, archive in enumerate(manifest["archives"]):
    if index % 2 == 0:
        print("/mnt/alakazam-episode-days/" + archive["path"].rsplit("/", 1)[-1])
PY

cp "$manifest" "$work/raw_expert_corpus_manifest.json"
cd /home/pokebot/alakazam-fast-split-r309
exec /usr/bin/python3 scripts/run_alakazam_fast_distributed_refeature.py \
  --manifest "$work/raw_expert_corpus_manifest.json" \
  --archive-root "$raw" \
  --cg-runtime-root /tmp/truenas_main/poke-bot-agent/outputs/experiments/alakazam-elmo-rule-derivative-g1/source-canonical-libcg-r236-d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7 \
  --output-root "$work/first-half-r309" \
  --host-index 0 \
  --workers 32
