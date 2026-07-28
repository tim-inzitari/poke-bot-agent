#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-/tmp/pokebot-goal-audit-v18}"
remote_host="${2:-elmo}"
remote_root="${3:-/mnt/Main/main/poke-bot-agent/archive}"
archive_name="${4:-inzi-goal-audit-v18-20260726.tar.zst}"
archive_path="${remote_root%/}/${archive_name}"
partial_path="${archive_path}.partial"

if [[ "$(realpath "$source_root")" != "/tmp/pokebot-goal-audit-v18" ]]; then
  echo "refusing unexpected source: $source_root" >&2
  exit 91
fi
if [[ ! -d "$source_root" || -L "$source_root" ]]; then
  echo "source is absent or unsafe: $source_root" >&2
  exit 92
fi

source_bytes="$(du -s -B1 "$source_root" | awk '{print $1}')"
echo "OFFLOAD_BEGIN source=$source_root bytes=$source_bytes remote=$remote_host:$archive_path"

ssh -o BatchMode=yes "$remote_host" \
  "sudo -n install -d -m 0755 '$remote_root' &&
   sudo -n rm -f '$partial_path'"

# The archive is deliberately CPU- and I/O-idle so the live trainer remains
# authoritative for resource use. pipefail ensures a truncated SSH transfer
# can never be promoted as a completed archive.
ionice -c3 nice -n 19 tar -C "$(dirname "$source_root")" \
  --one-file-system --sparse -cf - "$(basename "$source_root")" |
  ionice -c3 nice -n 19 zstd -1 -q |
  ssh -o BatchMode=yes "$remote_host" \
    "sudo -n dd of='$partial_path' bs=8M status=progress conv=fsync"

ssh -o BatchMode=yes "$remote_host" \
  "sudo -n zstd -q -t '$partial_path' &&
   sudo -n tar -I zstd -tf '$partial_path' >/dev/null &&
   sudo -n mv '$partial_path' '$archive_path' &&
   sudo -n sh -c \"sha256sum '$archive_path' > '$archive_path.sha256'\" &&
   sudo -n stat -c 'OFFLOAD_VERIFIED archive=%n bytes=%s' '$archive_path' &&
   sudo -n cat '$archive_path.sha256'"

# The source is removed only after the remote compressed stream and tar index
# both validate and the checksum sidecar is durably written.
find "$source_root" -depth -mindepth 1 -delete
rmdir "$source_root"
sync
echo "OFFLOAD_LOCAL_RELEASED bytes=$source_bytes source=$source_root"
df -h "$(dirname "$source_root")"
