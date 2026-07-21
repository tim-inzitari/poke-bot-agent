#!/usr/bin/env bash
#
# Download the pokemon-tcg-ai-battle competition data bundle (the hard authority
# for this project) plus the episodes index used for ladder bootstrap.
#
# The two "Card_ID List_*.pdf" files (~130MB + ~180MB) are human-readable card
# references only; the machine-usable data lives in EN_Card_Data.csv, so we skip
# the PDFs to save disk/time. Everything else (sample_submission/cg runtime,
# ptcg_engine C++ source headers, card CSVs) is downloaded individually with
# directory structure preserved.
#
# Uses the project venv kaggle CLI. Kaggle creds are expected at ~/.kaggle/kaggle.json.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV_KAGGLE="${ROOT}/.venv/bin/kaggle"
COMP="pokemon-tcg-ai-battle"
DEST="kaggle/input/pokemon-tcg-ai-battle"

if [[ ! -x "${VENV_KAGGLE}" ]]; then
  echo "ERROR: kaggle CLI not found at ${VENV_KAGGLE}" >&2
  exit 1
fi

mkdir -p "${DEST}"

# Files we deliberately skip (giant human-reference PDFs).
SKIP=(
  "Card_ID List_EN.pdf"
  "Card_ID List_JP.pdf"
)

is_skipped() {
  local name="$1"
  for s in "${SKIP[@]}"; do
    [[ "${name}" == "${s}" ]] && return 0
  done
  return 1
}

echo ">> Enumerating competition files ..."
mapfile -t FILES < <("${ROOT}/.venv/bin/python" - "${COMP}" <<'PY'
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
comp = sys.argv[1]
token = None
names = []
while True:
    res = api.competition_list_files(comp, page_token=token, page_size=200)
    files = res.files if hasattr(res, "files") else res
    for f in files:
        names.append(str(f.name if hasattr(f, "name") else f))
    token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
    if not token:
        break
print("\n".join(names))
PY
)

echo ">> ${#FILES[@]} files listed; downloading (skipping ${#SKIP[@]} PDFs) ..."
for name in "${FILES[@]}"; do
  if is_skipped "${name}"; then
    echo "   skip: ${name}"
    continue
  fi
  subdir="$(dirname "${name}")"
  outdir="${DEST}/${subdir}"
  mkdir -p "${outdir}"
  # -o overwrites; kaggle downloads a single file (may be zipped for some files).
  "${VENV_KAGGLE}" competitions download -c "${COMP}" -f "${name}" -p "${outdir}" -o >/dev/null 2>&1 || {
    echo "   WARN: failed to download ${name}" >&2
    continue
  }
  base="$(basename "${name}")"
  # If Kaggle returned a zip wrapper, unzip and remove it.
  if [[ -f "${outdir}/${base}.zip" ]]; then
    (cd "${outdir}" && unzip -o -q "${base}.zip" && rm -f "${base}.zip")
  fi
done

echo ">> Competition bundle present under ${DEST}"

# --- Episodes index (for per-archetype ladder bootstrap; later phase) ---
EPISODES_DEST="kaggle/input/pokemon-tcg-ai-battle-episodes-index"
if [[ "${SKIP_EPISODES:-0}" != "1" ]]; then
  echo ">> Downloading episodes index ..."
  mkdir -p "${EPISODES_DEST}"
  "${VENV_KAGGLE}" datasets download kaggle/pokemon-tcg-ai-battle-episodes-index \
    -p "${EPISODES_DEST}" --unzip || echo "   WARN: episodes index download failed (non-fatal)"
else
  echo ">> SKIP_EPISODES=1 set; skipping episodes index."
fi

echo ">> Done."
