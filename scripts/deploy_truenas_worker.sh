#!/usr/bin/env bash
# Stage / build / export the TrueNAS poke-bot remote worker image.
# Does not require SSH to TrueNAS for --stage (uses SMB share main).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMB_BASE="${TRUENAS_SMB:-/run/user/$(id -u)/gvfs/smb-share:server=truenas.local,share=main/poke-bot-agent}"
# Keep stage/build trees separate so concurrent --stage and --build cannot race.
STAGING_BUILD="${REPO_ROOT}/.truenas_worker_build"
STAGING_SMB="${REPO_ROOT}/.truenas_worker_stage"
IMAGE_NAME="${IMAGE_NAME:-poke-bot-truenas-worker:latest}"
TAR_NAME="poke-bot-truenas-worker.tar"

usage() {
  cat <<'EOF'
Usage: scripts/deploy_truenas_worker.sh [--stage|--build|--export-image|--all|--status]

  --stage         Copy compose + build context to SMB //truenas.local/main/poke-bot-agent
  --build         Build Docker image on this host (needs local Docker)
  --export-image  docker save → SMB (for `docker load` on TrueNAS)
  --all           stage + build + export-image
  --status        Print auth/connectivity snapshot
EOF
}

status() {
  local ssh_user="${TRUENAS_SSH_USER:-}"
  echo "=== TrueNAS connectivity ==="
  ping -c 1 -W 1 truenas.local >/dev/null 2>&1 && echo "ping: ok" || echo "ping: FAIL"
  if [[ -n "$ssh_user" ]]; then
    echo -n "ssh (BatchMode) ${ssh_user}@192.168.1.143: "
    ssh -o BatchMode=yes -o ConnectTimeout=4 "${ssh_user}@192.168.1.143" 'echo ok' 2>&1 | tail -1 || true
  else
    echo "ssh probe skipped (set TRUENAS_SSH_USER to test BatchMode key auth; never put passwords in this script)"
  fi
  echo -n "API /system/info: "
  curl -sk -o /dev/null -w "%{http_code}\n" https://192.168.1.143/api/v2.0/system/info || true
  echo -n "SMB staging writable: "
  if [[ -d "$(dirname "$SMB_BASE")" ]] || [[ -d "$SMB_BASE" ]]; then
    mkdir -p "$SMB_BASE" 2>/dev/null && touch "$SMB_BASE/.write-test" 2>/dev/null \
      && echo "yes ($SMB_BASE)" || echo "NO"
  else
    echo "share not mounted (open smb://truenas.local/main in Files)"
  fi
  echo "pubkey to install on TrueNAS (preferred over passwords):"
  cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "(missing ~/.ssh/id_ed25519.pub)"
}

prepare_context() {
  local STAGING="${1:-$STAGING_BUILD}"
  echo "[deploy] preparing build context at $STAGING"
  rm -rf "$STAGING"
  mkdir -p "$STAGING/kaggle/input" "$STAGING/containers/truenas-worker"
  rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
    "$REPO_ROOT/poke_bot/" "$STAGING/poke_bot/"
  rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/scripts/" "$STAGING/scripts/"
  rsync -a --delete "$REPO_ROOT/cards/" "$STAGING/cards/"
  rsync -a --delete "$REPO_ROOT/decks/" "$STAGING/decks/"
  rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/baselines/" "$STAGING/baselines/"
  # Prefer compact cg-lib mirror; also ship competition sample_submission parent.
  rsync -a --delete "$REPO_ROOT/kaggle/input/cg-lib/" "$STAGING/kaggle/input/cg-lib/"
  if [[ -d "$REPO_ROOT/kaggle/input/pokemon-tcg-ai-battle/sample_submission" ]]; then
    mkdir -p "$STAGING/kaggle/input/pokemon-tcg-ai-battle"
    rsync -a --delete \
      "$REPO_ROOT/kaggle/input/pokemon-tcg-ai-battle/sample_submission/" \
      "$STAGING/kaggle/input/pokemon-tcg-ai-battle/sample_submission/"
  fi
  if [[ -f "$REPO_ROOT/cards/EN_Card_Data.csv" ]]; then
    mkdir -p "$STAGING/kaggle/input/pokemon-tcg-ai-battle"
    cp -f "$REPO_ROOT/cards/EN_Card_Data.csv" \
      "$STAGING/kaggle/input/pokemon-tcg-ai-battle/EN_Card_Data.csv"
  fi
  rsync -a \
    "$REPO_ROOT/containers/truenas-worker/" \
    "$STAGING/containers/truenas-worker/"
  # Flatten Dockerfile to context root for simple `docker build -f`.
  cp -f "$REPO_ROOT/containers/truenas-worker/Dockerfile" "$STAGING/Dockerfile"
  cp -f "$REPO_ROOT/containers/truenas-worker/entrypoint.sh" "$STAGING/entrypoint.sh"
  # Adjust Dockerfile COPY paths when building from staging root:
  cat >"$STAGING/Dockerfile" <<'DOF'
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    POKEBOT_WORKER_CPU_ONLY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libstdc++6 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
    && python -m pip install numpy tqdm psutil kaggle-environments

COPY poke_bot/ /workspace/poke_bot/
COPY scripts/ /workspace/scripts/
COPY cards/ /workspace/cards/
COPY decks/ /workspace/decks/
COPY baselines/ /workspace/baselines/
COPY kaggle/ /workspace/kaggle/
COPY entrypoint.sh /entrypoint.sh

ENV CG_LIB_PATH=/workspace/kaggle/input/cg-lib \
    PYTHONPATH=/workspace \
    SIM_WORKERS=20 \
    LEAF_SERVERS=2 \
    LEAF_GPU=cuda:0 \
    LEAF_MAX_BATCH=192 \
    LEAF_QUEUE_DEPTH=96 \
    LEAF_COALESCE_MS=2 \
    POKEBOT_PRIMARY_ARCHETYPE=hammer-pult

RUN chmod +x /entrypoint.sh /workspace/scripts/run_remote_worker.py \
        /workspace/scripts/canary_remote_worker.py

EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
DOF
  chmod +x "$STAGING/entrypoint.sh"
  du -sh "$STAGING"
}

_smb_copy_tree() {
  # gvfs SMB rejects chmod on directory `.` after extract; treat that as OK
  # if content landed (tar still exits non-zero).
  local src="$1" dest="$2"
  mkdir -p "$dest"
  find "$dest" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
  set +e
  (cd "$src" && tar cf - .) | (cd "$dest" && tar xf - --no-same-owner --no-same-permissions)
  local rc=${PIPESTATUS[1]:-0}
  set -e
  # Non-zero solely from chmod on SMB is common; fail only if tree is empty.
  local count
  count=$(find "$dest" -type f 2>/dev/null | wc -l)
  if [[ "$count" -lt 1 ]]; then
    echo "[deploy] ERROR: SMB copy produced no files ($src -> $dest, tar_rc=$rc)" >&2
    return 1
  fi
  echo "[deploy] copied $count files -> $dest"
}

_smb_copy_files() {
  local dest="$1"
  shift
  mkdir -p "$dest"
  for f in "$@"; do
    cp -f "$f" "$dest/"
  done
}

stage() {
  prepare_context "$STAGING_SMB"
  if [[ ! -d "$(dirname "$SMB_BASE")" ]] && [[ ! -e "$SMB_BASE" ]]; then
    echo "[deploy] ERROR: SMB path missing: $SMB_BASE" >&2
    echo "Mount smb://truenas.local/main then retry." >&2
    exit 1
  fi
  mkdir -p "$SMB_BASE/containers/truenas-worker" "$SMB_BASE/checkpoint" "$SMB_BASE/runtime-logs" "$SMB_BASE/scripts"
  echo "[deploy] syncing to $SMB_BASE (tar over SMB; gvfs-safe)"
  _smb_copy_tree "$STAGING_SMB" "$SMB_BASE/build-context"
  _smb_copy_files "$SMB_BASE/containers/truenas-worker" \
    "$REPO_ROOT/containers/truenas-worker/docker-compose.yml" \
    "$REPO_ROOT/containers/truenas-worker/OPS.md" \
    "$REPO_ROOT/containers/truenas-worker/Dockerfile" \
    "$REPO_ROOT/containers/truenas-worker/entrypoint.sh"
  _smb_copy_tree "$REPO_ROOT/poke_bot" "$SMB_BASE/poke_bot"
  _smb_copy_files "$SMB_BASE/scripts" \
    "$REPO_ROOT/scripts/run_remote_worker.py" \
    "$REPO_ROOT/scripts/canary_remote_worker.py" \
    "$REPO_ROOT/scripts/deploy_truenas_worker.sh"
  # Install instructions + pubkey (never write passwords into this file)
  cat >"$SMB_BASE/INSTALL_AUTH.txt" <<EOF
TrueNAS deploy is staged here. Prefer SSH public-key auth (never commit passwords).

Add this pubkey to a docker-capable TrueNAS / elmo user in the UI:
  $(cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo 'MISSING_PUBKEY')

Then from the training box (set TRUENAS_SSH_USER yourself; no passwords in git):
  ssh "\${TRUENAS_SSH_USER}@truenas.local"
  # or export TRUENAS_API_KEY in your shell env (never commit it)

Sized for elmo (Ryzen 9 5900X + RTX 3060 LHR 12 GB):
  SIM_WORKERS=20  LEAF_SERVERS=2  LEAF_MAX_BATCH=192  cpus=20  mem=56g

Deploy target: TrueNAS Incus instance elmo (not host Apps).
See containers/truenas-worker/OPS.md
EOF
  echo "[deploy] staged. On elmo load image + compose from:"
  echo "  $SMB_BASE/containers/truenas-worker/"
  echo "  $SMB_BASE/build-context/"
  ls -la "$SMB_BASE/containers/truenas-worker/" | head
  du -sh "$SMB_BASE/build-context" 2>/dev/null || true
}

build() {
  prepare_context "$STAGING_BUILD"
  echo "[deploy] docker build $IMAGE_NAME (this downloads CUDA+torch; long)"
  docker build -t "$IMAGE_NAME" "$STAGING_BUILD"
  docker image ls "$IMAGE_NAME"
}

export_image() {
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[deploy] image missing; building first"
    build
  fi
  mkdir -p "$SMB_BASE/containers/truenas-worker" "$REPO_ROOT/.truenas_worker_build"
  LOCAL_GZ="$REPO_ROOT/.truenas_worker_build/${TAR_NAME}.gz"
  echo "[deploy] docker save → $LOCAL_GZ"
  docker save "$IMAGE_NAME" | gzip >"$LOCAL_GZ"
  ls -lh "$LOCAL_GZ"
  echo "[deploy] copying gzip to SMB (may take a while)..."
  cp -f "$LOCAL_GZ" "$SMB_BASE/containers/truenas-worker/${TAR_NAME}.gz"
  ls -lh "$SMB_BASE/containers/truenas-worker/${TAR_NAME}.gz"
  echo "[deploy] On TrueNAS: gunzip -c ${TAR_NAME}.gz | docker load"
}

cmd="${1:-}"
case "$cmd" in
  --status|status) status ;;
  --stage|stage) stage ;;
  --build|build) build ;;
  --export-image|export-image) export_image ;;
  --all|all) stage; build; export_image ;;
  -h|--help|"") usage; exit 0 ;;
  *) echo "unknown: $cmd" >&2; usage; exit 2 ;;
esac
