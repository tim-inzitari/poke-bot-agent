# Elmo remote worker image. Build from the staging context produced by the
# TrueNAS deployment script (the context includes the vendored cg runtime).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="poke-bot Elmo remote worker" \
      org.opencontainers.image.version="safety-20260719.7-portable-baselines" \
      org.opencontainers.image.description="Bounded, recycling Pokemon RL remote worker"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    POKEBOT_WORKER_CPU_ONLY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libstdc++6 \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
    && python -m pip install numpy tqdm psutil kaggle-environments orjson==3.11.9

COPY poke_bot/ /workspace/poke_bot/
COPY scripts/ /workspace/scripts/
COPY cards/ /workspace/cards/
COPY decks/ /workspace/decks/
COPY baselines/ /workspace/baselines/
COPY kaggle/ /workspace/kaggle/
COPY containers/truenas-worker/entrypoint.sh /entrypoint.sh
COPY containers/truenas-worker/supervise-production.sh /supervise-production.sh

# Four is the initial canary, not an advertised-only value hiding a larger
# process pool. Production values are supplied by the guarded override.
ENV CG_LIB_PATH=/workspace/kaggle/input/cg-lib \
    PYTHONPATH=/workspace \
    POKEBOT_REMOTE_WORKER_SAFETY_VERSION=20260717 \
    POKEBOT_REMOTE_WORKER_ARM_FILE=/workspace/runtime-logs/REMOTE_WORKER_ARMED \
    POKEBOT_WORKER_RECYCLE_GAMES=16 \
    WORKER_RECYCLE_GAMES=16 \
    POKEBOT_REMOTE_MAX_SERVICE_JOBS=100 \
    POKEBOT_REMOTE_TREE_RSS_LIMIT_GB=30 \
    POKEBOT_REMOTE_MIN_FREE_RAM_GB=24 \
    POKEBOT_REMOTE_MAX_CONNECTIONS=16 \
    POKEBOT_REMOTE_WATCHDOG_INTERVAL_S=5 \
    POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S=60 \
    SIM_WORKERS=4 \
    SIM_DEFAULT_WORKERS=4 \
    ELMO_SIM_WORKER_CEILING=20 \
    LEAF_SERVERS=2 \
    LEAF_GPU=cuda:0 \
    LEAF_MAX_BATCH=192 \
    LEAF_QUEUE_DEPTH=96 \
    LEAF_COALESCE_MS=2 \
    POKEBOT_PRIMARY_ARCHETYPE=hammer-pult

RUN chmod +x \
    /entrypoint.sh \
    /supervise-production.sh \
    /workspace/scripts/run_remote_worker.py

EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
