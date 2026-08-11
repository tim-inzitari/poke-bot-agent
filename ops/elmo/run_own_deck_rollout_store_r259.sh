#!/usr/bin/env bash
# Build only the dormant, immutable r259 OwnDeckLedger replay side store.
#
# This launcher is intentionally archive-native.  It never calls the generic
# replay collector and therefore never creates or touches its mutable
# index/raw-cache directories.  Root validates the protected 0600 receipts,
# projects read-only copies for the nonroot container, and mounts only the
# exact r241 archive tree plus a new sidecar output root.
set -euo pipefail

readonly NAME="pokebot-own-deck-rollout-store-r259"
readonly NORMAL_SMOKE_CONTAINER_NAME="${NAME}-normal-preflight"
readonly UNKNOWN_OUTCOME_SMOKE_CONTAINER_NAME="${NAME}-87394115-preflight"
readonly IMAGE="poke-bot-truenas-worker:matchup-v33-runtime"
readonly IMAGE_ID="sha256:74d66c41fda841e96ee89e88fab1fa800b82ab8c6a06cabdff146803a1b05a0f"
readonly SOURCE_MANIFEST="/mnt/Main/main/poke-bot-agent/archive/expert-r241-20260722-20260810/current.json"
readonly SOURCE_MANIFEST_SHA256="sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
readonly VERSIONED_RECEIPT_SHA256="sha256:d377cd5b4558150588d1461539d50bcfb2ca46898120b4e3ad97e9d95e479551"
readonly ARCHIVE_EPISODE_DAYS="/mnt/Main/main/poke-bot-agent/archive/episode-days"
readonly SOURCE_ORIGIN="/home/admin/pokebot-own-deck-r259-src-de844af19ca6"
readonly SOURCE_SNAPSHOT="/var/lib/pokebot-own-deck-r259-src-de844af19ca6"
readonly SOURCE_RUNTIME_LOCK="/etc/pokebot-own-deck-r259/de844af19ca6/source-tree.lock.json"
readonly CONTAINER_SOURCE="/r259-source"
readonly CLASSIFIER_ROOT="/home/admin/pokebot-expert-src-v6-strategic/data/training_mixes"
readonly OUTPUT="/mnt/Main/main/poke-bot-agent/archive/expert-r258-own-deck-ledger-sidecar/2026-07-22_2026-08-10"
readonly MIX_NAME="top_ladder.v1.json"
readonly REPRESENTATIVES_NAME="top_ladder_representatives.v1.json"
readonly SHARED_UID="1000"
readonly SHARED_GID="950"
readonly RUN_LOCK="/run/lock/${NAME}.lock"
readonly LOCK_ROOT="/run/pokebot-own-deck-r259-de844af19ca6"
readonly LOCK_CURRENT="${LOCK_ROOT}/current.r241.lock.json"
readonly LOCK_VERSIONED="${LOCK_ROOT}/window.r241.lock.json"
readonly LOCK_MIX="${LOCK_ROOT}/top_ladder.mix.lock.json"
readonly LOCK_REPRESENTATIVES="${LOCK_ROOT}/top_ladder.representatives.lock.json"
readonly SOURCE_LOCK_RECEIPT="${LOCK_ROOT}/SOURCE_LOCK.json"

if [[ "$(id -u)" != "0" ]]; then
  echo "${NAME}: must run as root so protected 0600 receipts can be verified" >&2
  exit 2
fi

for override in \
  POKEBOT_SOURCE \
  POKEBOT_MAIN \
  POKEBOT_ARCHIVE_RECEIPT \
  POKEBOT_OUTPUT \
  POKEBOT_IMAGE \
  POKEBOT_CONTAINER_NAME \
  POKEBOT_R259_SOURCE \
  POKEBOT_R259_OUTPUT
do
  if [[ -n "${!override:-}" ]]; then
    echo "${NAME}: fixed r259 identities forbid ${override} overrides" >&2
    exit 2
  fi
done

require_regular_root_0600() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "${NAME}: protected receipt is absent or unsafe: ${path}" >&2
    exit 1
  }
  [[ "$(stat -c '%u:%g:%a' "$path")" == "0:0:600" ]] || {
    echo "${NAME}: protected receipt must remain root:root 0600: ${path}" >&2
    exit 1
  }
}

require_regular() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "${NAME}: required regular source is absent or unsafe: ${path}" >&2
    exit 1
  }
}

require_regular_root_0444() {
  local path="$1"
  require_regular "$path"
  [[ "$(stat -c '%u:%g:%a' "$path")" == "0:0:444" ]] || {
    echo "${NAME}: root receipt projection must remain root:root 0444: ${path}" >&2
    exit 1
  }
}

sha256() {
  printf 'sha256:%s' "$(sha256sum "$1" | awk '{print $1}')"
}

exec 9>"${RUN_LOCK}"
if ! flock -n 9; then
  echo "${NAME}: an invocation is already validating or materializing; preserving its immutable output"
  exit 0
fi

require_regular_root_0600 "${SOURCE_MANIFEST}"
[[ "$(sha256 "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] || {
  echo "${NAME}: protected current receipt checksum drifted" >&2
  exit 1
}

VERSIONED_ORIGINAL="$(python3 - "${SOURCE_MANIFEST}" <<'PY'
from __future__ import annotations
import json
import os
import stat
import sys

manifest = os.path.realpath(sys.argv[1])
def reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
with open(manifest, encoding="utf-8") as stream:
    value = json.load(stream, object_pairs_hook=reject_duplicate_json_keys)
path = value.get("versioned_receipt")
if not isinstance(path, str) or not path.startswith("/"):
    raise SystemExit("current receipt has no absolute versioned receipt path")
entry = os.lstat(path)
if not stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
    raise SystemExit("versioned receipt is not a regular non-symlink file")
print(path)
PY
)"
require_regular_root_0600 "${VERSIONED_ORIGINAL}"
[[ "$(sha256 "${VERSIONED_ORIGINAL}")" == "${VERSIONED_RECEIPT_SHA256}" ]] || {
  echo "${NAME}: protected versioned receipt checksum drifted" >&2
  exit 1
}

[[ -d "${SOURCE_SNAPSHOT}" && ! -L "${SOURCE_SNAPSHOT}" ]] || {
  echo "${NAME}: immutable source snapshot is absent or unsafe" >&2
  exit 1
}
if find "${SOURCE_SNAPSHOT}" -xdev -type l -print -quit | grep -q .; then
  echo "${NAME}: immutable source snapshot must not contain symlinks" >&2
  exit 1
fi
[[ -d "${CLASSIFIER_ROOT}" && ! -L "${CLASSIFIER_ROOT}" ]] || {
  echo "${NAME}: pinned classifier root is absent or unsafe" >&2
  exit 1
}
readonly MIX="${CLASSIFIER_ROOT}/${MIX_NAME}"
readonly REPRESENTATIVES="${CLASSIFIER_ROOT}/${REPRESENTATIVES_NAME}"
readonly CARD_CSV="${SOURCE_SNAPSHOT}/cards/EN_Card_Data.csv"
require_regular "${MIX}"
require_regular "${REPRESENTATIVES}"
require_regular "${CARD_CSV}"
[[ "$(stat -c '%u:%g:%a' "$(dirname "${SOURCE_RUNTIME_LOCK}")")" == "0:0:755" ]] || {
  echo "${NAME}: runtime source-lock directory must be root:root 0755" >&2
  exit 1
}
require_regular_root_0444 "${SOURCE_RUNTIME_LOCK}"
[[ -d "${ARCHIVE_EPISODE_DAYS}" && ! -L "${ARCHIVE_EPISODE_DAYS}" ]] || {
  echo "${NAME}: exact episode-day archive root is absent or unsafe" >&2
  exit 1
}

# Re-hash the entire root-owned snapshot and compare the exact ordered file
# inventory with the pre-execution root lock.  The lock was created at the
# explicit staging boundary; this service never trusts or copies the mutable
# /home/admin origin at start time.
SOURCE_TREE_SHA256="$(python3 - "${SOURCE_SNAPSHOT}" "${SOURCE_RUNTIME_LOCK}" <<PY
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
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
        or info.st_mode & 0o022
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
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or info.st_mode & 0o022
        ):
            raise SystemExit(f"unsafe sealed source file: {child}")
        file_hash = hashlib.sha256()
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                file_hash.update(block)
        inventory.append({"path": str(child.relative_to(root)), "sha256": "sha256:" + file_hash.hexdigest()})
inventory_sha = digest(inventory)
tree_sha = digest({"schema": "poke_bot.own_deck_rollout_file_inventory/v1", "files": inventory})
value = json.loads(lock_path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit("runtime source lock must be an object")
detached = dict(value)
declared = detached.pop("lock_sha256", None)
required = {
    "schema": "poke_bot.own_deck_rollout_runtime_source_lock/v1",
    "version": 1,
    "owner_decision_revision": 259,
    "status": "sealed",
    "snapshot_origin": "${SOURCE_ORIGIN}",
    "sealed_snapshot": "${SOURCE_SNAPSHOT}",
    "source_tree_sha256": tree_sha,
    "file_inventory_sha256": inventory_sha,
    "file_count": len(inventory),
    "file_inventory": inventory,
    "image": {"tag": "${IMAGE}", "id": "${IMAGE_ID}"},
}
if set(value) != set(required) | {"lock_sha256"}:
    raise SystemExit("runtime source lock field set drifted")
if declared != digest(detached) or any(value.get(key) != item for key, item in required.items()):
    raise SystemExit("runtime source lock does not bind the sealed source inventory")
print(tree_sha)
PY
)"
RUNTIME_SOURCE_LOCK_SHA256="$(sha256 "${SOURCE_RUNTIME_LOCK}")"

if [[ -e "${OUTPUT}" || -L "${OUTPUT}" ]]; then
  [[ -d "${OUTPUT}" && ! -L "${OUTPUT}" ]] || {
    echo "${NAME}: output root is unsafe" >&2
    exit 1
  }
  [[ "$(stat -c '%u:%g:%a' "${OUTPUT}")" == "${SHARED_UID}:${SHARED_GID}:2770" ]] || {
    echo "${NAME}: existing output root ownership/mode drifted" >&2
    exit 1
  }
else
  install -d -o "${SHARED_UID}" -g "${SHARED_GID}" -m 2770 "${OUTPUT}"
fi
install -d -o root -g root -m 0755 "${LOCK_ROOT}"
[[ "$(stat -c '%u:%g:%a' "${LOCK_ROOT}")" == "0:0:755" ]] || {
  echo "${NAME}: root input lock directory ownership/mode drifted" >&2
  exit 1
}

# The copies preserve current.json's original bytes and the versioned receipt's
# original bytes.  They are root-produced, immutable-readable projections;
# the worker receives their paths only as read-only bind mounts.
for pair in \
  "${SOURCE_MANIFEST}:${LOCK_CURRENT}" \
  "${VERSIONED_ORIGINAL}:${LOCK_VERSIONED}" \
  "${MIX}:${LOCK_MIX}" \
  "${REPRESENTATIVES}:${LOCK_REPRESENTATIVES}"
do
  source_path="${pair%%:*}"
  lock_path="${pair#*:}"
  if [[ -e "${lock_path}" || -L "${lock_path}" ]]; then
    require_regular_root_0444 "${lock_path}"
    [[ "$(sha256 "${source_path}")" == "$(sha256 "${lock_path}")" ]] || {
      echo "${NAME}: root receipt projection identity drifted: ${lock_path}" >&2
      exit 1
    }
  else
    install -o root -g root -m 0444 "${source_path}" "${lock_path}"
  fi
done

python3 - "${SOURCE_LOCK_RECEIPT}" "${SOURCE_TREE_SHA256}" "$(sha256 "${LOCK_MIX}")" "$(sha256 "${LOCK_REPRESENTATIVES}")" "$(sha256 "${CARD_CSV}")" <<PY
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

target = Path(__import__("sys").argv[1])
tree, mix, representatives, card_csv = __import__("sys").argv[2:]
payload = {
    "schema": "poke_bot.own_deck_rollout_source_lock/v1",
    "version": 1,
    "owner_decision_revision": 259,
    "status": "sealed_read_only_input_lock",
    "source": {
        "snapshot_root": "${SOURCE_SNAPSHOT}",
        "snapshot_origin": "${SOURCE_ORIGIN}",
        "snapshot_tree_sha256": tree,
        "runtime_source_lock": {"path": "${SOURCE_RUNTIME_LOCK}", "sha256": "${RUNTIME_SOURCE_LOCK_SHA256}"},
        "manifest": {"original_path": "${SOURCE_MANIFEST}", "locked_path": "${LOCK_CURRENT}", "sha256": "${SOURCE_MANIFEST_SHA256}"},
        "versioned_receipt": {"original_path": "${VERSIONED_ORIGINAL}", "locked_path": "${LOCK_VERSIONED}", "sha256": "${VERSIONED_RECEIPT_SHA256}"},
        "classifier": {
            "root": "${CLASSIFIER_ROOT}",
            "mix": {"original_path": "${MIX}", "locked_path": "${LOCK_MIX}", "sha256": mix},
            "representatives": {"original_path": "${REPRESENTATIVES}", "locked_path": "${LOCK_REPRESENTATIVES}", "sha256": representatives},
            "card_csv": {"path": "${CARD_CSV}", "sha256": card_csv},
        },
    },
    "image": {"tag": "${IMAGE}", "id": "${IMAGE_ID}"},
    "output_root": "${OUTPUT}",
    "training_eligibility": {"active_r241": False, "sidecar_only": True},
}
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
def digest(value):
    copy = dict(value)
    copy.pop("source_lock_sha256", None)
    return "sha256:" + hashlib.sha256(canonical(copy)).hexdigest()
payload["source_lock_sha256"] = digest(payload)
if target.exists() or target.is_symlink():
    if target.is_symlink() or not target.is_file():
        raise SystemExit("r259 source lock target is unsafe")
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing.get("source_lock_sha256") != digest(existing) or existing != payload:
        raise SystemExit("existing r259 source lock differs; refusing replacement")
else:
    fd, temporary_name = tempfile.mkstemp(prefix=".SOURCE_LOCK.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
PY

[[ "$(docker image inspect -f '{{.Id}}' "${IMAGE}")" == "${IMAGE_ID}" ]] || {
  echo "${NAME}: local image ID does not match the pinned r259 image" >&2
  exit 1
}
[[ "$(docker image inspect -f '{{json .Config.Entrypoint}}' "${IMAGE}")" == '["/entrypoint.sh"]' ]] || {
  echo "${NAME}: image entrypoint contract changed" >&2
  exit 1
}
if docker inspect "${NAME}" >/dev/null 2>&1; then
  echo "${NAME}: a container with the managed name already exists; preserve it for audit" >&2
  exit 1
fi
if docker inspect "${NORMAL_SMOKE_CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "${NAME}: a prior normal preflight container exists; preserve it for audit" >&2
  exit 1
fi
if docker inspect "${UNKNOWN_OUTCOME_SMOKE_CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "${NAME}: a prior malformed-reward preflight container exists; preserve it for audit" >&2
  exit 1
fi

# Do not mask the image's /workspace/kaggle/input/cg-lib: features.cg_env
# resolves that baked-in runtime path.  The sealed worker source is isolated
# at /r259-source instead.
#
# Before sidecar publication, run two fully isolated in-memory preflights.
# The normal 1 GiB record smoke deliberately excludes 87394115.json.  The
# malformed-reward regression receives its own production-sized 2 GiB
# container, so it cannot retain memory in the normal process.  Neither has
# an output bind mount or any corpus-write capability.
docker run --rm --init \
  --name "${NORMAL_SMOKE_CONTAINER_NAME}" \
  --workdir "${CONTAINER_SOURCE}" \
  --network none \
  --runtime runc \
  --read-only \
  --user "${SHARED_UID}:${SHARED_GID}" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 128 \
  --cpus 0.5 \
  --memory 1g \
  --memory-swap 1g \
  --blkio-weight 100 \
  --ulimit nofile=4096:4096 \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --tmpfs /run:rw,noexec,nosuid,size=16m \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=none \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env CUDA_VISIBLE_DEVICES= \
  --env LEAF_GPU=cpu \
  --mount "type=bind,src=${SOURCE_SNAPSHOT},dst=${CONTAINER_SOURCE},readonly" \
  --mount "type=bind,src=${ARCHIVE_EPISODE_DAYS},dst=${ARCHIVE_EPISODE_DAYS},readonly" \
  --mount "type=bind,src=${LOCK_CURRENT},dst=/input/current.json,readonly" \
  --mount "type=bind,src=${LOCK_VERSIONED},dst=/input/window.json,readonly" \
  --mount "type=bind,src=${LOCK_MIX},dst=/input/top_ladder.mix.json,readonly" \
  --mount "type=bind,src=${LOCK_REPRESENTATIVES},dst=/input/top_ladder.representatives.json,readonly" \
  --entrypoint /usr/local/bin/python \
  "${IMAGE}" \
  "${CONTAINER_SOURCE}/scripts/update_own_deck_rollout_store.py" \
  --archive-native-smoke \
  --source-manifest /input/current.json \
  --original-manifest-path "${SOURCE_MANIFEST}" \
  --versioned-receipt-lock /input/window.json \
  --expected-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --expected-versioned-receipt-sha256 "${VERSIONED_RECEIPT_SHA256}" \
  --classifier-mix /input/top_ladder.mix.json \
  --classifier-representatives /input/top_ladder.representatives.json \
  --card-csv "${CONTAINER_SOURCE}/cards/EN_Card_Data.csv"

docker run --rm --init \
  --name "${UNKNOWN_OUTCOME_SMOKE_CONTAINER_NAME}" \
  --workdir "${CONTAINER_SOURCE}" \
  --network none \
  --runtime runc \
  --read-only \
  --user "${SHARED_UID}:${SHARED_GID}" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 256 \
  --cpus 1.0 \
  --memory 2g \
  --memory-swap 2g \
  --blkio-weight 100 \
  --ulimit nofile=4096:4096 \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --tmpfs /run:rw,noexec,nosuid,size=16m \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=none \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env CUDA_VISIBLE_DEVICES= \
  --env LEAF_GPU=cpu \
  --mount "type=bind,src=${SOURCE_SNAPSHOT},dst=${CONTAINER_SOURCE},readonly" \
  --mount "type=bind,src=${ARCHIVE_EPISODE_DAYS},dst=${ARCHIVE_EPISODE_DAYS},readonly" \
  --mount "type=bind,src=${LOCK_CURRENT},dst=/input/current.json,readonly" \
  --mount "type=bind,src=${LOCK_VERSIONED},dst=/input/window.json,readonly" \
  --mount "type=bind,src=${LOCK_MIX},dst=/input/top_ladder.mix.json,readonly" \
  --mount "type=bind,src=${LOCK_REPRESENTATIVES},dst=/input/top_ladder.representatives.json,readonly" \
  --entrypoint /usr/local/bin/python \
  "${IMAGE}" \
  "${CONTAINER_SOURCE}/scripts/update_own_deck_rollout_store.py" \
  --archive-native-unknown-outcome-smoke \
  --source-manifest /input/current.json \
  --original-manifest-path "${SOURCE_MANIFEST}" \
  --versioned-receipt-lock /input/window.json \
  --expected-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --expected-versioned-receipt-sha256 "${VERSIONED_RECEIPT_SHA256}" \
  --classifier-mix /input/top_ladder.mix.json \
  --classifier-representatives /input/top_ladder.representatives.json \
  --card-csv "${CONTAINER_SOURCE}/cards/EN_Card_Data.csv"

exec docker run --rm --init \
  --name "${NAME}" \
  --workdir "${CONTAINER_SOURCE}" \
  --network none \
  --runtime runc \
  --read-only \
  --user "${SHARED_UID}:${SHARED_GID}" \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 256 \
  --cpus 1.0 \
  --memory 2g \
  --memory-swap 2g \
  --blkio-weight 100 \
  --ulimit nofile=4096:4096 \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --tmpfs /run:rw,noexec,nosuid,size=16m \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=none \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env CUDA_VISIBLE_DEVICES= \
  --env LEAF_GPU=cpu \
  --mount "type=bind,src=${SOURCE_SNAPSHOT},dst=${CONTAINER_SOURCE},readonly" \
  --mount "type=bind,src=${ARCHIVE_EPISODE_DAYS},dst=${ARCHIVE_EPISODE_DAYS},readonly" \
  --mount "type=bind,src=${LOCK_CURRENT},dst=/input/current.json,readonly" \
  --mount "type=bind,src=${LOCK_VERSIONED},dst=/input/window.json,readonly" \
  --mount "type=bind,src=${LOCK_MIX},dst=/input/top_ladder.mix.json,readonly" \
  --mount "type=bind,src=${LOCK_REPRESENTATIVES},dst=/input/top_ladder.representatives.json,readonly" \
  --mount "type=bind,src=${OUTPUT},dst=/output" \
  --entrypoint /usr/local/bin/python \
  "${IMAGE}" \
  "${CONTAINER_SOURCE}/scripts/update_own_deck_rollout_store.py" \
  --archive-native \
  --source-manifest /input/current.json \
  --original-manifest-path "${SOURCE_MANIFEST}" \
  --versioned-receipt-lock /input/window.json \
  --output-root /output \
  --source-snapshot "${SOURCE_ORIGIN}" \
  --source-snapshot-tree-sha256 "${SOURCE_TREE_SHA256}" \
  --expected-manifest-sha256 "${SOURCE_MANIFEST_SHA256}" \
  --expected-versioned-receipt-sha256 "${VERSIONED_RECEIPT_SHA256}" \
  --image-tag "${IMAGE}" \
  --image-id "${IMAGE_ID}" \
  --classifier-mix /input/top_ladder.mix.json \
  --classifier-representatives /input/top_ladder.representatives.json \
  --card-csv "${CONTAINER_SOURCE}/cards/EN_Card_Data.csv"
