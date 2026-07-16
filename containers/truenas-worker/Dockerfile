# poke-bot remote worker for TrueNAS SCALE (AMD Ryzen 9 5900X + RTX 3060 LHR 12 GB).
# Build from the staged context produced by scripts/deploy_truenas_worker.sh
# (includes vendored cg runtime — repo .dockerignore excludes kaggle/input).
#
# Optional faster libcg fork: copy into build context as kaggle/input/libcg_fork/
# (deploy_truenas_worker.sh --libcg-fork) or bind-mount at runtime via compose.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# Torch cu124 wheels run on 3060 LHR (sm_86). Keep CPU sims CUDA-masked via env.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
    && python -m pip install \
        numpy \
        tqdm \
        psutil \
        kaggle-environments

# Staged context layout (see deploy_truenas_worker.sh):
#   poke_bot/  scripts/  cards/  decks/  baselines/  kaggle/input/...
COPY poke_bot/ /workspace/poke_bot/
COPY scripts/ /workspace/scripts/
COPY cards/ /workspace/cards/
COPY decks/ /workspace/decks/
COPY baselines/ /workspace/baselines/
COPY kaggle/ /workspace/kaggle/
COPY containers/truenas-worker/entrypoint.sh /entrypoint.sh

# Default CG path: competition sample_submission when present in context,
# else vendored cg-lib. Fork overrides via CG_LIB_PATH at runtime.
ENV CG_LIB_PATH=/workspace/kaggle/input/cg-lib \
    PYTHONPATH=/workspace \
    SIM_WORKERS=20 \
    LEAF_SERVERS=2 \
    LEAF_GPU=cuda:0 \
    LEAF_MAX_BATCH=192 \
    LEAF_QUEUE_DEPTH=96 \
    LEAF_COALESCE_MS=2 \
    POKEBOT_MULTI_ENV=1 \
    POKEBOT_MULTI_ENV_PER_WORKER=4 \
    POKEBOT_PRIMARY_ARCHETYPE=hammer-pult

RUN chmod +x /entrypoint.sh /workspace/scripts/run_remote_worker.py \
        /workspace/scripts/canary_remote_worker.py

EXPOSE 8765
ENTRYPOINT ["/entrypoint.sh"]
