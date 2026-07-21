#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SOURCE_DIR OUTPUT_LIBRARY [native|znver3|portable]" >&2
  exit 2
fi

source_dir=$1
output=$2
profile=${3:-native}
cxx=${CXX:-c++}
root=$(cd "$(dirname "$0")/.." && pwd)

case "$profile" in
  native)
    arch_flags=(-march=native)
    ;;
  znver3)
    arch_flags=(-march=znver3 -mtune=znver3)
    ;;
  portable)
    arch_flags=()
    ;;
  *)
    echo "unsupported CPU profile: $profile" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
link_flags=(-shared -Wl,-z,relro,-z,now)
if [[ $(uname -s) == Darwin ]]; then
  link_flags=(-dynamiclib)
fi

"$cxx" \
  -std=c++20 \
  -O3 \
  -DNDEBUG \
  -fPIC \
  -pthread \
  -fvisibility=hidden \
  -fno-semantic-interposition \
  "${arch_flags[@]}" \
  "${link_flags[@]}" \
  -I"$source_dir" \
  "$source_dir/Export.cpp" \
  "$root/engine_patches/HiddenExport.cpp" \
  -o "$output"

echo "$output"
