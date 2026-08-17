#!/usr/bin/env bash
# Poll one checksum-bound successor corpus and trigger pre-stage only after a
# successful import. A missing remote ready receipt is an ordinary timer poll,
# not an import success.
set -euo pipefail

: "${PRESTAGE_REMOTE_HOST:?missing PRESTAGE_REMOTE_HOST}"
: "${PRESTAGE_REMOTE_ROOT:?missing PRESTAGE_REMOTE_ROOT}"
: "${PRESTAGE_DESTINATION:?missing PRESTAGE_DESTINATION}"
: "${PRESTAGE_SPECIALIST_ID:?missing PRESTAGE_SPECIALIST_ID}"
: "${PRESTAGE_GUIDE_VERSION:?missing PRESTAGE_GUIDE_VERSION}"
: "${PRESTAGE_IMPORT_RECEIPT:?missing PRESTAGE_IMPORT_RECEIPT}"
: "${PRESTAGE_BWLIMIT_KIB:?missing PRESTAGE_BWLIMIT_KIB}"
: "${PRESTAGE_MINIMUM_RECORDS:?missing PRESTAGE_MINIMUM_RECORDS}"
: "${PRESTAGE_SUPERSEDED_READY_SHA256:?missing PRESTAGE_SUPERSEDED_READY_SHA256}"

ready_receipt="${PRESTAGE_REMOTE_ROOT%/}/CURRENT_DECK_GUIDE_CORPUS_READY.json"
if ! /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=5 \
  "$PRESTAGE_REMOTE_HOST" /usr/bin/test -s "$ready_receipt"; then
  exit 0
fi

/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python \
  /home/pokebot/poke-bot-agent/scripts/import_prestaged_specialist_corpus.py \
  --host "$PRESTAGE_REMOTE_HOST" \
  --remote-root "$PRESTAGE_REMOTE_ROOT" \
  --destination "$PRESTAGE_DESTINATION" \
  --specialist-id "$PRESTAGE_SPECIALIST_ID" \
  --guide-version "$PRESTAGE_GUIDE_VERSION" \
  --receipt "$PRESTAGE_IMPORT_RECEIPT" \
  --bwlimit-kib "$PRESTAGE_BWLIMIT_KIB" \
  --minimum-records "$PRESTAGE_MINIMUM_RECORDS" \
  --superseded-ready-sha256="$PRESTAGE_SUPERSEDED_READY_SHA256"

/usr/bin/systemctl --user start pokebot-next-specialist-prestage.service
