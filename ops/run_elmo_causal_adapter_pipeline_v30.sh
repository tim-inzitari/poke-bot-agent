#!/usr/bin/env bash
set -euo pipefail

root=/mnt/Main/main/poke-adapter-oracle-v29
image=poke-bot-truenas-worker:adapter-v29-go-first
tree=$root/output/public-matchup-tree-latest20/public-matchup-tree.json
ready=$root/output/public-matchup-tree-latest20/PUBLIC_MATCHUP_TREE_READY.json

until [[ -s "$tree" && -s "$ready" ]]; do
  status=$(sudo docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' \
    pokebot-public-tree-v29 2>/dev/null || true)
  if [[ $status == exited\ * && $status != "exited 0" ]]; then
    printf 'public tree failed before causal oracle: %s\n' "$status" >&2
    exit 1
  fi
  sleep 10
done

router_digest=$(python3 - "$tree" <<'PY'
import hashlib
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
print("sha256:" + hashlib.sha256(p.read_bytes()).hexdigest())
PY
)

sudo docker rm -f pokebot-adapter-oracle-causal-v30 \
  pokebot-adapter-stage-causal-v30 >/dev/null 2>&1 || true
sudo docker run --name pokebot-adapter-oracle-causal-v30 \
  --entrypoint /bin/bash \
  -v "$root:/work" -w /work/src -e PYTHONPATH=/work/src \
  "$image" -lc '
    exec python -u scripts/build_matchup_adapter_oracle.py \
      --feature-manifest /work/data/alakazam-adapter-routes/manifest.json \
      --archive-dir /work/data/raw \
      --output-dir /work/output/alakazam-adapter-oracle-causal \
      --mix /work/config/top_ladder.v1.json \
      --representatives /work/config/top_ladder_representatives.v1.json \
      --card-csv /work/config/EN_Card_Data.csv \
      --additive-archetype dudunsparce \
      --additive-archetype hops-trevenant \
      --additive-archetype walrein \
      --jobs 6 \
      --public-tree /work/output/public-matchup-tree-latest20/public-matchup-tree.json
  '

sudo docker run --name pokebot-adapter-stage-causal-v30 \
  --entrypoint /bin/bash \
  -v "$root:/work" -w /work/src -e PYTHONPATH=/work/src \
  "$image" -lc "
    exec python -u scripts/stage_matchup_adapter_corpus.py \\
      --feature-manifest /work/data/alakazam-adapter-routes/manifest.json \\
      --oracle-manifest /work/output/alakazam-adapter-oracle-causal/oracle-manifest.json \\
      --package-registry /work/output/alakazam-adapter-oracle-causal/opponent-registry.json \\
      --active-gate-contract /work/config/alakazam_gate_program_v1.json \\
      --output-dir /work/output/alakazam-adapter-staged-causal \\
      --val-frac 0.10 --seed 42 --disk-floor-gib 100 \\
      --require-public-router-digest '$router_digest'
  "

printf 'CAUSAL_ADAPTER_CORPUS_READY router=%s manifest=%s\n' \
  "$router_digest" \
  "$root/output/alakazam-adapter-staged-causal/manifest.json"
