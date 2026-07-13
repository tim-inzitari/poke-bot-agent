"""Canonical filesystem paths for the poke_bot project.

Everything is anchored to the repository root (the parent of the ``poke_bot``
package) so imports work regardless of the current working directory. Later
phases (dataset, training, submission) import from here rather than hard-coding
relative paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

#: ``poke_bot/`` package directory.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

#: Repository root (parent of ``poke_bot/``).
REPO_ROOT: Path = PACKAGE_DIR.parent

# ---------------------------------------------------------------------------
# Kaggle inputs (competition data is the hard authority)
# ---------------------------------------------------------------------------

KAGGLE_INPUT: Path = REPO_ROOT / "kaggle" / "input"

#: Full competition bundle (downloaded by scripts/setup_competition_data.sh).
COMPETITION_DIR: Path = KAGGLE_INPUT / "pokemon-tcg-ai-battle"

#: Competition-shipped ``cg`` runtime (preferred for Kaggle parity).
COMPETITION_CG_PARENT: Path = COMPETITION_DIR / "sample_submission" / "sample_submission"
COMPETITION_CG_DIR: Path = COMPETITION_CG_PARENT / "cg"

#: Convenience mirror of the cg runtime (already vendored in the repo checkout).
CG_LIB_PARENT: Path = KAGGLE_INPUT / "cg-lib"
CG_LIB_DIR: Path = CG_LIB_PARENT / "cg"

#: C++ simulator source (headers) — reference for mechanics / search semantics.
PTCG_ENGINE_DIR: Path = COMPETITION_DIR / "ptcg_engine" / "ptcgProgram 22"

#: Episodes index dataset used for ladder bootstrap (later phase).
EPISODES_INDEX_DIR: Path = KAGGLE_INPUT / "pokemon-tcg-ai-battle-episodes-index"

# ---------------------------------------------------------------------------
# Card data
# ---------------------------------------------------------------------------

#: Committed copy of the English card table.
EN_CARD_DATA_REPO: Path = REPO_ROOT / "cards" / "EN_Card_Data.csv"
#: Competition copy of the English card table (identical schema).
EN_CARD_DATA_COMPETITION: Path = COMPETITION_DIR / "EN_Card_Data.csv"

# ---------------------------------------------------------------------------
# Decks / submission
# ---------------------------------------------------------------------------

DECKS_DIR: Path = REPO_ROOT / "decks"
SUBMISSION_DIR: Path = REPO_ROOT / "submission"
SUBMISSION_DECK: Path = SUBMISSION_DIR / "deck.csv"

#: v1 pure-Dragapult submission list (Campinas 2026 2nd — not hammer-signature).
DRAGAPULT_DECK: Path = (
    DECKS_DIR
    / "dragapult-only"
    / "2026-05_regional-campinas-2026_2nd_dragapult.csv"
)

#: Hammer-Pult list kept for Phase 6 / later specialist (Campinas 2026 4th).
HAMMER_PULT_DECK: Path = (
    DECKS_DIR
    / "dragapult-only"
    / "2026-05_regional-campinas-2026_4th_dragapult-dudunsparce.csv"
)

BASELINES_DIR: Path = REPO_ROOT / "baselines"
BASELINES_MANIFEST: Path = BASELINES_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Generated artifacts (RAM-resident cache preferred; see config.py)
# ---------------------------------------------------------------------------

DATA_DIR: Path = REPO_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"


def cg_runtime_dir() -> Path:
    """Return the directory containing the ``cg`` package to import.

    Priority:
      1. ``$CG_LIB_PATH`` if set and it contains a ``cg`` package.
      2. The competition-shipped runtime (Kaggle parity).
      3. The vendored ``cg-lib`` mirror.

    Returns the *parent* directory that should be placed on ``sys.path`` so that
    ``import cg`` resolves. Raises ``FileNotFoundError`` if none is present.
    """
    env = os.environ.get("CG_LIB_PATH")
    candidates: list[Path] = []
    if env:
        p = Path(env).expanduser().resolve()
        # Accept either the parent of cg/ or the cg/ dir itself.
        candidates.append(p if (p / "cg").is_dir() else p.parent)
    candidates.append(COMPETITION_CG_PARENT)
    candidates.append(CG_LIB_PARENT)

    for parent in candidates:
        if (parent / "cg" / "api.py").is_file():
            return parent
    raise FileNotFoundError(
        "Could not locate a 'cg' runtime. Run scripts/setup_competition_data.sh "
        "or set CG_LIB_PATH to the directory containing the cg package."
    )


def en_card_data_path() -> Path:
    """Return the first available English card-data CSV (repo copy preferred)."""
    for p in (EN_CARD_DATA_REPO, EN_CARD_DATA_COMPETITION):
        if p.is_file():
            return p
    raise FileNotFoundError("EN_Card_Data.csv not found in repo or competition bundle.")


def ensure_runtime_dirs() -> None:
    """Create the generated-artifact directories if they do not exist."""
    for d in (DATA_DIR, CACHE_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
