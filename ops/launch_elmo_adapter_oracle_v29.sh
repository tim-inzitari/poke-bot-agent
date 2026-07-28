#!/usr/bin/env bash
set -euo pipefail

root=/mnt/Main/main/poke-adapter-oracle-v29
image=poke-bot-truenas-worker:adapter-v29-go-first
targets=(
  crustle
  marnie-s-grimmsnarl-ex
  garchomp
  cornerstone-ogerpon
  rockets-mewtwo
  starmie
  hammer-pult
  alakazam
  lucario
  archaludon-ex
  dragapult-dudunsparce
  dragapult
  dudunsparce
  hops-trevenant
  walrein
  dragapult-dusknoir
  dragapult-blaziken
  lopunny
  gardevoir
  ns-zoroark
  raging-bolt
  festival-lead
)

tree_args=()
for archive in "$root"/data/raw/pokemon-tcg-ai-battle-episodes-*.zip; do
  tree_args+=(--archive "/work/data/raw/${archive##*/}")
done
for target in "${targets[@]}"; do
  tree_args+=(--target-archetype "$target" --additive-archetype "$target")
done

sudo docker rm -f pokebot-adapter-oracle-v29 pokebot-public-tree-v29 \
  >/dev/null 2>&1 || true

sudo docker run -d --name pokebot-adapter-oracle-v29 \
  --entrypoint /bin/bash \
  -v "$root:/work" -w /work/src \
  -e PYTHONPATH=/work/src \
  "$image" -lc '
    exec python -u scripts/build_matchup_adapter_oracle.py \
      --feature-manifest /work/data/alakazam-adapter-routes/manifest.json \
      --archive-dir /work/data/raw \
      --output-dir /work/output/alakazam-adapter-oracle \
      --mix /work/config/top_ladder.v1.json \
      --representatives /work/config/top_ladder_representatives.v1.json \
      --card-csv /work/config/EN_Card_Data.csv \
      --additive-archetype dudunsparce \
      --additive-archetype hops-trevenant \
      --additive-archetype walrein
  ' >/dev/null

sudo docker run -d --name pokebot-public-tree-v29 \
  --entrypoint /bin/bash \
  -v "$root:/work" -w /work/src \
  -e PYTHONPATH=/work/python-packages:/work/src \
  "$image" -lc 'exec "$@"' bash \
  python -u scripts/train_public_matchup_tree.py \
    "${tree_args[@]}" \
    --mix /work/config/top_ladder.v1.json \
    --representatives /work/config/top_ladder_representatives.v1.json \
    --card-csv /work/config/EN_Card_Data.csv \
    --output-dir /work/output/public-matchup-tree-latest20 \
    --jobs 8 --max-depth 18 --min-samples-leaf 50 >/dev/null

printf 'oracle=%s tree=%s routes=%s\n' \
  "$(sudo docker inspect -f '{{.State.Status}}' pokebot-adapter-oracle-v29)" \
  "$(sudo docker inspect -f '{{.State.Status}}' pokebot-public-tree-v29)" \
  "${#targets[@]}"
