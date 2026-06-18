#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p kaggle/input/cg-lib
kaggle datasets download kiyotah/cg-lib -p kaggle/input/cg-lib --unzip
