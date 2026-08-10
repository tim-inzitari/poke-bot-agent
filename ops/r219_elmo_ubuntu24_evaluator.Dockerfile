# Deliberately inert, local-only evaluator image for the r219 preflight.
# The caller must pass an amd64 Ubuntu 24.04 digest as BASE_IMAGE.  The image
# contains immutable copies of the frozen r195 package and the exact b77
# seeded evaluator overlay; its default command is only the smoke receipt.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="pokebot r219 Elmo Ubuntu 24 evaluator" \
      org.opencontainers.image.description="sealed local-only r195+b77 compatibility evaluator; default command is smoke only" \
      org.opencontainers.image.version="r219-r195-261d367e-b77afbd3" \
      org.opencontainers.image.r195.bundle.sha256="dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145" \
      org.opencontainers.image.r195.checkpoint.sha256="261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a" \
      org.opencontainers.image.r195.matchup-tree.sha256="e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049" \
      org.opencontainers.image.seeded-engine.sha256="b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6"

ARG R195_MODEL_SHA256=261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a
ARG R195_TREE_SHA256=e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049
ARG B77_SHA256=b77afbd363fe80de968c7cf20a0bbf5eb616fefcacbeab7eeeda94213fad9ea6

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/pokebot/r195-package

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
    && /opt/venv/bin/python -m pip install \
        numpy==1.26.4 \
        tqdm==4.67.1

COPY r195-package/ /opt/pokebot/r195-package/
COPY libcg_hidden_pristine_batch_b77afbd3.so /opt/pokebot/libcg_hidden_pristine_batch_b77afbd3.so
COPY smoke.py /opt/pokebot/smoke.py

RUN test "$(sha256sum /opt/pokebot/r195-package/model.pt | awk '{print $1}')" = "$R195_MODEL_SHA256" \
    && test "$(sha256sum /opt/pokebot/r195-package/matchup_tree.json | awk '{print $1}')" = "$R195_TREE_SHA256" \
    && test "$(sha256sum /opt/pokebot/libcg_hidden_pristine_batch_b77afbd3.so | awk '{print $1}')" = "$B77_SHA256" \
    && chmod 0555 /opt/pokebot/smoke.py

WORKDIR /opt/pokebot/r195-package
CMD ["/opt/venv/bin/python", "/opt/pokebot/smoke.py"]
