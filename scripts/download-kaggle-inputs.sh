#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p kaggle/input/cg-lib kaggle/input/pokemon-tcg-ai-battle-episodes-index
kaggle datasets download kiyotah/cg-lib -p kaggle/input/cg-lib --unzip
kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-index \
  -p kaggle/input/pokemon-tcg-ai-battle-episodes-index --unzip
