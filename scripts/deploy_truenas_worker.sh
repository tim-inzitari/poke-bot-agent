#!/usr/bin/env bash
# Stage / build / export the TrueNAS poke-bot remote worker image.
# Does not require SSH to TrueNAS for --stage (uses SMB share main).
#
# Safe to run while overnight training is live: --stage/--build/--export-image
# never restart workers. Cutover restarts belong in
# scripts/redeploy_remote_boundary.sh (promotion/iter boundary only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMB_BASE="${TRUENAS_SMB:-/run/user/$(id -u)/gvfs/smb-share:server=truenas.local,share=main/poke-bot-agent}"
# Keep stage/build trees separate so concurrent --stage and --build cannot race.
STAGING_BUILD="${REPO_ROOT}/.truenas_worker_build"
STAGING_SMB="${REPO_ROOT}/.truenas_worker_stage"
IMAGE_NAME="${IMAGE_NAME:-poke-bot-truenas-worker:latest}"
TAR_NAME="poke-bot-truenas-worker.tar"
LIBCG_FORK="${LIBCG_FORK:-}"

usage() {
  cat <<'EOF'
Usage: scripts/deploy_truenas_worker.sh [options] <command>

Commands:
  --stage         Copy compose + build context to SMB //truenas.local/main/poke-bot-agent
  --build         Build Docker image on this host (needs local Docker)
  --export-image  docker save → SMB (for `docker load` on TrueNAS)
  --all           stage + build + export-image
  --status        Print auth/connectivity snapshot
  --print-load    Print exact TrueNAS host docker load + compose commands

Options:
  --libcg-fork PATH   Bake/stage a fork cg tree (dir with libcg.so or cg/libcg.so)
  --image NAME        Image tag (default: poke-bot-truenas-worker:latest)

Env:
  TRUENAS_SMB, TRUENAS_SSH_USER, IMAGE_NAME, LIBCG_FORK
EOF
}

status() {
  local ssh_user="${TRUENAS_SSH_USER:-}"
  echo "=== TrueNAS connectivity ==="
  ping -c 1 -W 1 truenas.local >/dev/null 2>&1 && echo "ping: ok" || echo "ping: FAIL"
  ping -c 1 -W 1 192.168.1.143 >/dev/null 2>&1 && echo "ping 192.168.1.143: ok" || echo "ping 192.168.1.143: FAIL"
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
  if command -v docker >/dev/null 2>&1; then
    echo -n "local docker image: "
    docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 && echo "present ($IMAGE_NAME)" || echo "missing ($IMAGE_NAME)"
  else
    echo "local docker: not installed"
  fi
  echo "pubkey to install on TrueNAS (preferred over passwords):"
  cat ~/.ssh/id_ed25519.pub 2>/dev/null || echo "(missing ~/.ssh/id_ed25519.pub)"
}

_resolve_fork_src() {
  local src="${1:-}"
  if [[ -z "$src" ]]; then
    return 1
  fi
  if [[ -f "$src/libcg.so" ]] || [[ -f "$src/libcg.dylib" ]]; then
    echo "$src"
    return 0
  fi
  if [[ -f "$src/cg/libcg.so" ]] || [[ -f "$src/cg/libcg.dylib" ]]; then
    echo "$src"
    return 0
  fi
  if [[ -f "$src" && "$(basename "$src")" == libcg.so ]]; then
    echo "$(cd "$(dirname "$src")" && pwd)"
    return 0
  fi
  echo "[deploy] ERROR: --libcg-fork must be a cg dir or parent containing cg/libcg.so (got $src)" >&2
  return 2
}

_stage_fork_into() {
  local STAGING="$1"
  if [[ -z "$LIBCG_FORK" ]]; then
    return 0
  fi
  local fork_src
  fork_src="$(_resolve_fork_src "$LIBCG_FORK")"
  mkdir -p "$STAGING/kaggle/input/libcg_fork" "$STAGING/containers/truenas-worker/libcg_fork"
  echo "[deploy] staging libcg fork from $fork_src"
  if [[ -f "$fork_src/libcg.so" ]] || [[ -f "$fork_src/libcg.dylib" ]]; then
    rsync -a --delete "$fork_src/" "$STAGING/kaggle/input/libcg_fork/"
    rsync -a --delete "$fork_src/" "$STAGING/containers/truenas-worker/libcg_fork/"
  else
    rsync -a --delete "$fork_src/cg/" "$STAGING/kaggle/input/libcg_fork/"
    rsync -a --delete "$fork_src/cg/" "$STAGING/containers/truenas-worker/libcg_fork/"
  fi
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
  if [[ -d "$REPO_ROOT/kaggle/input/cg-lib" ]]; then
    rsync -a --delete "$REPO_ROOT/kaggle/input/cg-lib/" "$STAGING/kaggle/input/cg-lib/"
  else
    mkdir -p "$STAGING/kaggle/input/cg-lib"
    echo "[deploy] WARN: kaggle/input/cg-lib missing on this host" >&2
  fi
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
  _stage_fork_into "$STAGING"
  # Flatten Dockerfile to context root for simple `docker build`.
  # Prefer competition sample cg when present; fork wins via entrypoint resolve.
  local default_cg=/workspace/kaggle/input/cg-lib
  if [[ -f "$STAGING/kaggle/input/libcg_fork/libcg.so" ]]; then
    default_cg=/workspace/kaggle/input/libcg_fork
  elif [[ -f "$STAGING/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so" ]]; then
    default_cg=/workspace/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg
  fi
  cp -f "$REPO_ROOT/containers/truenas-worker/entrypoint.sh" "$STAGING/entrypoint.sh"
  cat >"$STAGING/Dockerfile" <<DOF
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    CUDA_DEVICE_ORDER=PCI_BUS_ID \\
    POKEBOT_WORKER_CPU_ONLY=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
        python3 python3-pip python3-venv libstdc++6 libgomp1 ca-certificates curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel \\
    && python -m pip install \\
        --index-url https://download.pytorch.org/whl/cu124 \\
        torch==2.6.0 \\
    && python -m pip install numpy tqdm psutil kaggle-environments

COPY poke_bot/ /workspace/poke_bot/
COPY scripts/ /workspace/scripts/
COPY cards/ /workspace/cards/
COPY decks/ /workspace/decks/
COPY baselines/ /workspace/baselines/
COPY kaggle/ /workspace/kaggle/
COPY entrypoint.sh /entrypoint.sh

ENV CG_LIB_PATH=${default_cg} \\
    PYTHONPATH=/workspace \\
    SIM_WORKERS=20 \\
    LEAF_SERVERS=2 \\
    LEAF_GPU=cuda:0 \\
    LEAF_MAX_BATCH=192 \\
    LEAF_QUEUE_DEPTH=96 \\
    LEAF_COALESCE_MS=2 \\
    POKEBOT_MULTI_ENV=1 \\
    POKEBOT_MULTI_ENV_PER_WORKER=4 \\
    POKEBOT_PRIMARY_ARCHETYPE=hammer-pult

RUN chmod +x /entrypoint.sh /workspace/scripts/run_remote_worker.py \\
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

print_load() {
  cat <<EOF
# Exact load + restart commands on TrueNAS host Docker (192.168.1.143:8765)
# Run ONLY at a promotion/iter boundary (see docs/REMOTE_WORKER_CUTOVER.md).

cd /mnt/Main/main/poke-bot-agent/containers/truenas-worker \\
  || cd /mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker

# 1) Load image from staged tar (from deploy --export-image)
gunzip -fk poke-bot-truenas-worker.tar.gz || true
docker load -i poke-bot-truenas-worker.tar

# 2) Optional fork bind-mount (if not baked into image)
# mkdir -p libcg_fork && rsync -a /path/to/cg/ libcg_fork/
# export LIBCG_FORK_HOST=./libcg_fork CG_LIB_PATH=/workspace/libcg_fork

# 3) Ensure champion checkpoint exists
mkdir -p checkpoint runtime-logs
# cp /path/to/champion.pt checkpoint/model.pt

# 4) Recreate worker (brief remote outage — host trainer must be paused or
#    at iter boundary so in-flight collect is not mid-wave)
docker compose down
docker compose up -d
docker compose ps
docker logs --tail 80 poke-bot-truenas-worker

# 5) Canary from training box (fail-closed on simulator_version mismatch)
# python scripts/canary_remote_worker.py 192.168.1.143:8765 --require-match-local
EOF
}

stage() {
  prepare_context "$STAGING_SMB"
  if [[ ! -d "$(dirname "$SMB_BASE")" ]] && [[ ! -e "$SMB_BASE" ]]; then
    echo "[deploy] ERROR: SMB path missing: $SMB_BASE" >&2
    echo "Mount smb://truenas.local/main then retry." >&2
    exit 1
  fi
  mkdir -p "$SMB_BASE/containers/truenas-worker" "$SMB_BASE/checkpoint" \
    "$SMB_BASE/runtime-logs" "$SMB_BASE/scripts" "$SMB_BASE/containers/truenas-worker/libcg_fork"
  echo "[deploy] syncing to $SMB_BASE (tar over SMB; gvfs-safe)"
  _smb_copy_tree "$STAGING_SMB" "$SMB_BASE/build-context"
  _smb_copy_files "$SMB_BASE/containers/truenas-worker" \
    "$REPO_ROOT/containers/truenas-worker/docker-compose.yml" \
    "$REPO_ROOT/containers/truenas-worker/OPS.md" \
    "$REPO_ROOT/containers/truenas-worker/Dockerfile" \
    "$REPO_ROOT/containers/truenas-worker/entrypoint.sh"
  if [[ -d "$STAGING_SMB/containers/truenas-worker/libcg_fork" ]]; then
    _smb_copy_tree "$STAGING_SMB/containers/truenas-worker/libcg_fork" \
      "$SMB_BASE/containers/truenas-worker/libcg_fork"
  fi
  _smb_copy_tree "$REPO_ROOT/poke_bot" "$SMB_BASE/poke_bot"
  _smb_copy_files "$SMB_BASE/scripts" \
    "$REPO_ROOT/scripts/run_remote_worker.py" \
    "$REPO_ROOT/scripts/canary_remote_worker.py" \
    "$REPO_ROOT/scripts/deploy_truenas_worker.sh" \
    "$REPO_ROOT/scripts/redeploy_remote_boundary.sh"
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
  POKEBOT_MULTI_ENV=1 POKEBOT_MULTI_ENV_PER_WORKER=4

Deploy path: TrueNAS host Docker on 192.168.1.143:8765 (nested elmo Docker blocked).
Boundary cutover: scripts/redeploy_remote_boundary.sh
See containers/truenas-worker/OPS.md and docs/REMOTE_WORKER_CUTOVER.md
EOF
  print_load >"$SMB_BASE/containers/truenas-worker/LOAD_ON_ELMO.sh.txt"
  echo "[deploy] staged. On TrueNAS host load image + compose from:"
  echo "  $SMB_BASE/containers/truenas-worker/"
  echo "  $SMB_BASE/build-context/"
  ls -la "$SMB_BASE/containers/truenas-worker/" | head
  du -sh "$SMB_BASE/build-context" 2>/dev/null || true
}

build() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[deploy] ERROR: docker not available on this host" >&2
    echo "Build on the training box, or stage context and build on TrueNAS:" >&2
    echo "  cd …/build-context && docker build -t $IMAGE_NAME ." >&2
    exit 1
  fi
  prepare_context "$STAGING_BUILD"
  echo "[deploy] docker build $IMAGE_NAME (this downloads CUDA+torch; long)"
  docker build -t "$IMAGE_NAME" "$STAGING_BUILD"
  docker image ls "$IMAGE_NAME"
}

export_image() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[deploy] ERROR: docker not available" >&2
    exit 1
  fi
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
  print_load
}

CMD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --libcg-fork)
      LIBCG_FORK="$2"
      shift 2
      ;;
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --status|status|--stage|stage|--build|build|--export-image|export-image|--all|all|--print-load|print-load)
      CMD="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "${CMD:-}" in
  --status|status) status ;;
  --stage|stage) stage ;;
  --build|build) build ;;
  --export-image|export-image) export_image ;;
  --all|all) stage; build; export_image ;;
  --print-load|print-load) print_load ;;
  "") usage; exit 0 ;;
  *) echo "unknown: $CMD" >&2; usage; exit 2 ;;
esac
