#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p kaggle/input/pokemon-tcg-ai-battle-episodes-index
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-index \
  -p kaggle/input/pokemon-tcg-ai-battle-episodes-index --unzip
echo "downloaded kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"
