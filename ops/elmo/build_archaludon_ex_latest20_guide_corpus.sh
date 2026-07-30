#!/usr/bin/env bash
# Compatibility entry point for the full-public schema-7 guide builder.
set -euo pipefail
exec "$(dirname "$0")/build_archaludon_ex_full_public_schema7_guide_corpus.sh" "$@"
