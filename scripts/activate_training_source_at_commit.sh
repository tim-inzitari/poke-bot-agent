#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 COMMIT_FILE STAGED_SOURCE TARGET_SOURCE SYSTEMD_SERVICE" >&2
  exit 64
fi

commit_file=$1
staged_source=$2
target_source=$3
service=$4

while [[ ! -s "$commit_file" ]]; do
  sleep 5
done

python_bin=/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python
"$python_bin" -m py_compile "$staged_source"
install -m 0644 "$staged_source" "${target_source}.next"
mv -f "${target_source}.next" "$target_source"
systemctl --user restart "$service"
