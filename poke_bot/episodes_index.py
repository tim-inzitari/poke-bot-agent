"""Episodes-index manifest helpers for ladder bootstrap.

The Kaggle dataset ``kaggle/pokemon-tcg-ai-battle-episodes-index`` ships a
``manifest.csv`` listing one row per calendar day of ladder episodes. Each day
has a companion dataset ``kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD``
(~0.7 GB zip of per-episode JSON files).

This module:
  - loads / refreshes the index;
  - selects the latest N calendar days and expands prior days when filters
    undershoot ``min_games`` / opponent diversity;
  - downloads daily bundles under ``data/episodes/`` (gitignored).
"""

from __future__ import annotations

import csv
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from . import paths

EPISODES_INDEX_SLUG = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DEFAULT_KAGGLE_BIN = paths.REPO_ROOT / ".venv" / "bin" / "kaggle"

#: Where daily episode zips / extracted JSON land (gitignored ``data/``).
EPISODES_DATA_DIR: Path = paths.DATA_DIR / "episodes"
EPISODES_RAW_DIR: Path = EPISODES_DATA_DIR / "raw"


@dataclass(frozen=True)
class DailyDatasetEntry:
    date: str
    slug: str
    url: str
    episode_count: int
    total_bytes: int
    top_avg_score: float
    median_avg_score: float

    @property
    def kaggle_ref(self) -> str:
        return f"kaggle/{self.slug}" if not self.slug.startswith("kaggle/") else self.slug


def default_index_path() -> Path:
    return paths.EPISODES_INDEX_DIR / "manifest.csv"


def ensure_episodes_index(
    *,
    index_dir: Optional[Path] = None,
    kaggle_bin: Optional[Path] = None,
) -> Path:
    """Download the episodes index if missing; return path to ``manifest.csv``."""
    index_dir = index_dir or paths.EPISODES_INDEX_DIR
    manifest = index_dir / "manifest.csv"
    if manifest.is_file():
        return manifest

    index_dir.mkdir(parents=True, exist_ok=True)
    kaggle = Path(kaggle_bin) if kaggle_bin else DEFAULT_KAGGLE_BIN
    if not kaggle.is_file():
        raise FileNotFoundError(
            f"Episodes index missing at {manifest} and kaggle CLI not found at {kaggle}"
        )
    cmd = [
        str(kaggle),
        "datasets",
        "download",
        EPISODES_INDEX_SLUG,
        "-p",
        str(index_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)
    if not manifest.is_file():
        raise FileNotFoundError(f"Downloaded episodes index but {manifest} is still missing")
    return manifest


def load_daily_manifest(path: Optional[Path] = None) -> list[DailyDatasetEntry]:
    """Parse ``manifest.csv`` into chronological :class:`DailyDatasetEntry` rows."""
    path = path or ensure_episodes_index()
    if not path.is_file():
        raise FileNotFoundError(f"episodes index manifest not found: {path}")

    entries: list[DailyDatasetEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slug = str(row.get("daily_dataset_slug", "") or "").strip()
            if not slug:
                continue
            entries.append(
                DailyDatasetEntry(
                    date=str(row.get("date", "")).strip(),
                    slug=slug,
                    url=str(row.get("daily_dataset_url", "") or ""),
                    episode_count=int(float(row.get("episode_count", 0) or 0)),
                    total_bytes=int(float(row.get("total_bytes", 0) or 0)),
                    top_avg_score=float(row.get("top_avg_score", 0) or 0),
                    median_avg_score=float(row.get("median_avg_score", 0) or 0),
                )
            )
    return sorted(entries, key=lambda e: e.date)


def latest_n_days(manifest: Iterable[DailyDatasetEntry], n: int = 3) -> list[DailyDatasetEntry]:
    """Return the latest ``n`` calendar days (oldest→newest within the window)."""
    rows = sorted(manifest, key=lambda e: e.date)
    if n <= 0:
        return []
    return rows[-n:]


def expand_prior_days(
    manifest: list[DailyDatasetEntry],
    already: list[DailyDatasetEntry],
    *,
    extra: int = 1,
) -> list[DailyDatasetEntry]:
    """Append up to ``extra`` prior days not already in ``already`` (newest-first prior)."""
    have = {e.date for e in already}
    prior = [e for e in sorted(manifest, key=lambda e: e.date, reverse=True) if e.date not in have]
    added = prior[: max(0, extra)]
    # Keep chronological order for downstream logging.
    return sorted(list(already) + added, key=lambda e: e.date)


def local_daily_dir(entry: DailyDatasetEntry, root: Optional[Path] = None) -> Path:
    """Directory for extracted daily JSON (``data/episodes/raw/<slug>/``)."""
    root = root or EPISODES_RAW_DIR
    return root / entry.slug


def local_daily_zip(entry: DailyDatasetEntry, root: Optional[Path] = None) -> Path:
    root = root or EPISODES_RAW_DIR
    return root / f"{entry.slug}.zip"


def download_daily_dataset(
    entry: DailyDatasetEntry,
    *,
    root: Optional[Path] = None,
    kaggle_bin: Optional[Path] = None,
    unzip: bool = False,
) -> Path:
    """Download a daily episodes zip into ``data/episodes/raw/``.

    Returns the zip path. Full unzip of a day is ~20 GB — prefer
    :func:`iter_episode_payloads` which streams JSON from the zip.
    """
    root = root or EPISODES_RAW_DIR
    root.mkdir(parents=True, exist_ok=True)
    zip_path = local_daily_zip(entry, root)
    day_dir = local_daily_dir(entry, root)

    if day_dir.is_dir() and any(day_dir.glob("*.json")):
        return day_dir
    if zip_path.is_file() and zip_path.stat().st_size > 0:
        if unzip:
            _unzip_daily(zip_path, day_dir)
            return day_dir
        return zip_path

    kaggle = Path(kaggle_bin) if kaggle_bin else DEFAULT_KAGGLE_BIN
    if not kaggle.is_file():
        raise FileNotFoundError(f"kaggle CLI not found at {kaggle}")

    ref = entry.slug if "/" in entry.slug else f"kaggle/{entry.slug}"
    cmd = [
        str(kaggle),
        "datasets",
        "download",
        ref,
        "-p",
        str(root),
        "-o",
    ]
    subprocess.run(cmd, check=True)

    # Kaggle names the zip after the dataset slug.
    candidates = [
        root / f"{entry.slug}.zip",
        root / f"{ref.split('/')[-1]}.zip",
    ]
    for cand in candidates:
        if cand.is_file():
            if cand != zip_path:
                cand.rename(zip_path)
            break
    if not zip_path.is_file():
        # Sometimes download lands as already-unzipped files.
        if day_dir.is_dir() and any(day_dir.glob("*.json")):
            return day_dir
        raise FileNotFoundError(f"Download finished but zip not found for {entry.slug}")

    if unzip:
        _unzip_daily(zip_path, day_dir)
        return day_dir
    return zip_path


def _unzip_daily(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def list_local_episode_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def iter_episode_paths(
    entry: DailyDatasetEntry,
    *,
    root: Optional[Path] = None,
    max_files: int = 0,
) -> Iterator[tuple[str, Path | str]]:
    """Yield ``(episode_id, path_or_zip_member)`` for a daily bundle.

    Prefers an extracted directory; otherwise yields members from the zip as
    ``("zip:" + zip_path, member_name)`` markers consumed by
    :func:`poke_bot.replay_import.load_episode_payload`.
    """
    root = root or EPISODES_RAW_DIR
    day_dir = local_daily_dir(entry, root)
    files = list_local_episode_files(day_dir)
    if files:
        n = 0
        for path in files:
            yield path.stem, path
            n += 1
            if max_files > 0 and n >= max_files:
                return
        return

    zip_path = local_daily_zip(entry, root)
    if not zip_path.is_file():
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/"))
        n = 0
        for name in names:
            stem = Path(name).stem
            yield stem, f"zip:{zip_path}::{name}"
            n += 1
            if max_files > 0 and n >= max_files:
                return
