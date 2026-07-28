#!/usr/bin/env bash
# Download immutable Kaggle daily episode archives atomically on Elmo.
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 ARCHIVE_DIR YYYY-MM-DD:EXPECTED_EPISODES [...]" >&2
  exit 2
fi

archive_dir="$1"
shift
minimum_free_gib="${POKEBOT_EXPERT_DOWNLOAD_MIN_FREE_GIB:-200}"
maximum_mbit="${POKEBOT_DOWNLOAD_MAX_MBIT:-12}"

mkdir -p "$archive_dir/.download"

# Kaggle archives share Elmo's constrained Ethernet path with production RPC.
# Apply an ingress policer inside the download container when it has NET_ADMIN.
# Production worker traffic is in another container/network namespace and is
# therefore never throttled. Fail closed when a positive limit was requested
# but the launcher did not provide the required capability/tooling.
if [[ "$maximum_mbit" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$maximum_mbit" 'BEGIN { exit !(value > 0) }'; then
  if ! command -v tc >/dev/null 2>&1; then
    echo "tc is required for POKEBOT_DOWNLOAD_MAX_MBIT=$maximum_mbit" >&2
    exit 1
  fi
  tc qdisc replace dev eth0 handle ffff: ingress
  tc filter replace dev eth0 parent ffff: protocol ip priority 10 u32 \
    match u32 0 0 \
    police rate "${maximum_mbit}mbit" burst 256k drop flowid :1
  echo "[network] Kaggle ingress capped at ${maximum_mbit} Mbit/s"
fi

for item in "$@"; do
  day="${item%%:*}"
  expected="${item##*:}"
  if [[ ! "$day" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
      || [[ ! "$expected" =~ ^[0-9]+$ ]]; then
    echo "invalid day/count pair: $item" >&2
    exit 2
  fi

  free_kib="$(df -Pk "$archive_dir" | awk 'NR == 2 {print $4}')"
  if (( free_kib < minimum_free_gib * 1024 * 1024 )); then
    echo "free-space guard failed before $day: ${free_kib} KiB available" >&2
    exit 1
  fi

  slug="pokemon-tcg-ai-battle-episodes-${day}"
  destination="$archive_dir/${slug}.zip"
  staging="$archive_dir/.download/$day"
  candidate="$staging/${slug}.zip"

  validate_archive() {
    python - "$1" "$expected" <<'PY'
import sys
import zipfile

path, expected_text = sys.argv[1:]
expected = int(expected_text)
with zipfile.ZipFile(path) as archive:
    archive.testzip()
    actual = sum(
        name.endswith(".json") and not name.endswith("/")
        for name in archive.namelist()
    )
if actual != expected:
    raise SystemExit(
        f"archive episode-count mismatch: actual={actual} expected={expected}"
    )
print(f"[validate] path={path} episodes={actual}", flush=True)
PY
  }

  if [[ -s "$destination" ]]; then
    validate_archive "$destination"
    echo "[download] reuse validated day=$day path=$destination"
    continue
  fi

  rm -rf "$staging"
  mkdir -p "$staging"
  echo "[download] begin day=$day dataset=kaggle/$slug"
  python -m kaggle datasets download "kaggle/$slug" -p "$staging"
  if [[ ! -s "$candidate" ]]; then
    echo "Kaggle download did not create expected archive: $candidate" >&2
    exit 1
  fi
  validate_archive "$candidate"
  mv "$candidate" "$destination"
  chmod 0444 "$destination"
  rmdir "$staging"
  echo "[download] complete day=$day bytes=$(stat -c %s "$destination") path=$destination"
done

echo "[complete] all requested daily archives are validated"
