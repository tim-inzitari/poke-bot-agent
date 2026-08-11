#!/usr/bin/env bash
# One-time root staging boundary for the r259 side-store worker source.
#
# This script does not start, stop, reload, or enable any service.  It copies
# the audited admin-origin tree as data into a root-only /var/lib snapshot,
# but accepts it only if a controller-supplied, root-owned expected inventory
# and tree digest match before the atomic publish.
#
# Trust bootstrap (run the verified, root-owned copy — never sudo-execute this
# file from /home/admin):
#   install -o root -g root -m 0444 CONTROLLER_APPROVED_FILE_INVENTORY \
#     /etc/pokebot-own-deck-r259/de844af19ca6/expected-file-inventory.json
#   install -o root -g root -m 0555 STAGED_SCRIPT \
#     /etc/pokebot-own-deck-r259/de844af19ca6/stage-source
#   printf '%s  %s\n' CONTROLLER_APPROVED_SCRIPT_SHA256_HEX \
#     /etc/pokebot-own-deck-r259/de844af19ca6/stage-source | sha256sum -c -
#   /etc/pokebot-own-deck-r259/de844af19ca6/stage-source \
#     --expected-source-tree-sha256 SHA256 \
#     --expected-file-inventory \
#       /etc/pokebot-own-deck-r259/de844af19ca6/expected-file-inventory.json
# The controller inventory is an ordered JSON list of exact
# {"path": relative_path, "sha256": "sha256:<hex>"} rows.  Its paired tree
# digest is sha256 of canonical JSON for
# {"schema":"poke_bot.own_deck_rollout_file_inventory/v1","files":rows}
# (sorted object keys, compact separators, UTF-8).  This script deliberately
# does not derive either approval input from the mutable origin.
set -euo pipefail

readonly NAME="pokebot-own-deck-rollout-store-r259"
readonly ORIGIN="/home/admin/pokebot-own-deck-r259-src-de844af19ca6"
readonly SEALED="/var/lib/pokebot-own-deck-r259-src-de844af19ca6"
readonly LOCK_DIR="/etc/pokebot-own-deck-r259/de844af19ca6"
readonly LOCK="${LOCK_DIR}/source-tree.lock.json"
readonly IMAGE="poke-bot-truenas-worker:matchup-v33-runtime"
readonly IMAGE_ID="sha256:74d66c41fda841e96ee89e88fab1fa800b82ab8c6a06cabdff146803a1b05a0f"

if [[ $# -ne 4 || "$1" != "--expected-source-tree-sha256" || "$3" != "--expected-file-inventory" ]]; then
  echo "usage: ${0##*/} --expected-source-tree-sha256 sha256:... --expected-file-inventory root-owned-inventory.json" >&2
  exit 2
fi
readonly EXPECTED_SOURCE_TREE_SHA256="$2"
readonly EXPECTED_FILE_INVENTORY="$4"
[[ "${EXPECTED_SOURCE_TREE_SHA256}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "${NAME}: expected source-tree checksum is invalid" >&2
  exit 2
}

if [[ "$(id -u)" != "0" ]]; then
  echo "${NAME}: source staging must run as root" >&2
  exit 2
fi
[[ -f "${EXPECTED_FILE_INVENTORY}" && ! -L "${EXPECTED_FILE_INVENTORY}" ]] || {
  echo "${NAME}: expected controller inventory is absent or unsafe" >&2
  exit 1
}
[[ "$(stat -c '%u:%g:%a' "${EXPECTED_FILE_INVENTORY}")" == "0:0:444" ]] || {
  echo "${NAME}: expected controller inventory must be root:root 0444" >&2
  exit 1
}
[[ -d "${ORIGIN}" && ! -L "${ORIGIN}" ]] || {
  echo "${NAME}: origin is absent or unsafe: ${ORIGIN}" >&2
  exit 1
}
if find "${ORIGIN}" -xdev -type l -print -quit | grep -q .; then
  echo "${NAME}: origin contains a symlink; refusing to stage it" >&2
  exit 1
fi
if find "${ORIGIN}" -xdev \( ! -type d -a ! -type f \) -print -quit | grep -q .; then
  echo "${NAME}: origin contains a special file; refusing to stage it" >&2
  exit 1
fi

[[ -d /var/lib && ! -L /var/lib && "$(stat -c '%u:%g:%a' /var/lib)" == "0:0:755" ]] || {
  echo "${NAME}: /var/lib must already be root:root 0755" >&2
  exit 1
}
install -d -o root -g root -m 0755 "${LOCK_DIR}"
[[ "$(stat -c '%u:%g:%a' "${LOCK_DIR}")" == "0:0:755" ]] || {
  echo "${NAME}: runtime source-lock directory is unsafe" >&2
  exit 1
}

candidate="${SEALED}"
new_candidate=0
if [[ -e "${SEALED}" || -L "${SEALED}" ]]; then
  [[ -d "${SEALED}" && ! -L "${SEALED}" ]] || {
    echo "${NAME}: existing sealed snapshot is unsafe" >&2
    exit 1
  }
else
  stage="$(mktemp -d /var/lib/.pokebot-own-deck-r259-src-de844af19ca6.XXXXXX)"
  cp -a "${ORIGIN}"/. "${stage}"/
  if find "${stage}" -xdev \( -type l -o \( ! -type d -a ! -type f \) \) -print -quit | grep -q .; then
    echo "${NAME}: copied snapshot is unsafe; preserving staging directory for audit: ${stage}" >&2
    exit 1
  fi
  [[ -f "${stage}/ops/elmo/run_own_deck_rollout_store_r259.sh" ]] || {
    echo "${NAME}: staged snapshot lacks r259 launcher" >&2
    exit 1
  }
  chown -R root:root "${stage}"
  find "${stage}" -xdev -type d -exec chmod 0555 {} +
  find "${stage}" -xdev -type f -exec chmod 0444 {} +
  chmod 0555 "${stage}/ops/elmo/run_own_deck_rollout_store_r259.sh"
  candidate="${stage}"
  new_candidate=1
fi

lock_temporary="$(mktemp "${LOCK_DIR}/.source-tree.lock.XXXXXX")"
python3 - "${candidate}" "${EXPECTED_SOURCE_TREE_SHA256}" "${EXPECTED_FILE_INVENTORY}" "${lock_temporary}" <<PY
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

root = Path(sys.argv[1])
expected_tree_sha = sys.argv[2]
expected_inventory_path = Path(sys.argv[3])
temporary_lock = Path(sys.argv[4])
def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
inventory = []
for directory, names, files in os.walk(root, followlinks=False):
    current = Path(directory)
    info = current.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
    or stat.S_IMODE(info.st_mode) != 0o555
    ):
        raise SystemExit(f"unsafe sealed source directory: {current}")
    names[:] = sorted(names)
    for name in names:
        child = current / name
        if child.is_symlink():
            raise SystemExit(f"sealed source contains symlink: {child}")
    for name in sorted(files):
        child = current / name
        info = child.lstat()
        relative = str(child.relative_to(root))
        expected_mode = (
            0o555
            if relative == "ops/elmo/run_own_deck_rollout_store_r259.sh"
            else 0o444
        )
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != expected_mode
        ):
            raise SystemExit(f"unsafe sealed source file: {child}")
        file_hash = hashlib.sha256()
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                file_hash.update(block)
        inventory.append({"path": relative, "sha256": "sha256:" + file_hash.hexdigest()})
payload = {
    "schema": "poke_bot.own_deck_rollout_runtime_source_lock/v1",
    "version": 1,
    "owner_decision_revision": 259,
    "status": "sealed",
    "snapshot_origin": "${ORIGIN}",
    "sealed_snapshot": "${SEALED}",
    "source_tree_sha256": digest({"schema": "poke_bot.own_deck_rollout_file_inventory/v1", "files": inventory}),
    "file_inventory_sha256": digest(inventory),
    "file_count": len(inventory),
    "file_inventory": inventory,
    "image": {"tag": "${IMAGE}", "id": "${IMAGE_ID}"},
}
payload["lock_sha256"] = digest(payload)
expected_inventory = json.loads(expected_inventory_path.read_text(encoding="utf-8"))
if expected_inventory != inventory:
    raise SystemExit("sealed source inventory differs from controller-supplied inventory")
if payload["source_tree_sha256"] != expected_tree_sha:
    raise SystemExit("sealed source tree digest differs from controller-supplied digest")
if payload["file_inventory_sha256"] != digest(expected_inventory):
    raise SystemExit("controller inventory checksum is malformed")
with temporary_lock.open("wb") as stream:
    stream.write(canonical(payload) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chown(temporary_lock, 0, 0)
os.chmod(temporary_lock, 0o444)
PY

if [[ -e "${LOCK}" || -L "${LOCK}" ]]; then
  [[ -f "${LOCK}" && ! -L "${LOCK}" && "$(stat -c '%u:%g:%a' "${LOCK}")" == "0:0:444" ]] || {
    echo "${NAME}: existing runtime source lock is unsafe" >&2
    exit 1
  }
  cmp -s "${lock_temporary}" "${LOCK}" || {
    echo "${NAME}: existing runtime source lock differs; refusing replacement" >&2
    exit 1
  }
  rm -f -- "${lock_temporary}"
else
  if (( new_candidate )); then
    mv -T "${candidate}" "${SEALED}"
  fi
  mv -T "${lock_temporary}" "${LOCK}"
fi
if (( new_candidate )) && [[ -e "${LOCK}" && ! -e "${SEALED}" ]]; then
  # An existing identical lock cannot authorize a missing/new source tree.
  echo "${NAME}: runtime source lock existed without the sealed snapshot" >&2
  exit 1
fi

[[ "$(stat -c '%u:%g:%a' "${LOCK}")" == "0:0:444" ]] || {
  echo "${NAME}: runtime source lock did not seal root:root 0444" >&2
  exit 1
}
echo "${NAME}: sealed source staged at ${SEALED}; lock=${LOCK}"
