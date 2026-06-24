from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _default_score_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2 if cpu_count > 4 else cpu_count)


def _score_replay_path(path_str: str) -> tuple[str, str, float] | None:
    """Worker: score one replay. Returns (path, episode_id, score) or None on error."""
    path = Path(path_str)
    try:
        episode_id, score = load_replay_score(path)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return (path_str, episode_id, score)


@dataclass(frozen=True)
class DailyDatasetEntry:
    date: str
    slug: str
    url: str
    episode_count: int
    total_bytes: int
    top_avg_score: float
    median_avg_score: float


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    replay_path: Path | None = None
    score: float = 0.0
    submission_id: str = ""
    team_id: str = ""
    team_name: str = ""
    source: str = ""


def default_index_path(root: Path) -> Path:
    return root / "kaggle/input/pokemon-tcg-ai-battle-episodes-index/manifest.csv"


# Substrings that mark a rollout row's ``source`` as real competition replay data
# (episodes-index daily bundles), as opposed to synthetic CABT self-play.
TOP_OF_LADDER_SOURCE_MARKERS: tuple[str, ...] = (
    "pokemon-tcg-ai-battle",  # daily-dataset slugs from episodes-index manifest
    "episode",                # episode-ids-file pools
)


def is_top_of_ladder_source(source: str | None, markers: tuple[str, ...] | None = None) -> bool:
    """True if a rollout ``source`` denotes a top-of-ladder / replay-derived game."""
    if not source:
        return False
    lowered = str(source).strip().lower()
    if not lowered:
        return False
    for marker in (markers or TOP_OF_LADDER_SOURCE_MARKERS):
        if marker in lowered:
            return True
    return False


def load_daily_manifest(path: Path) -> list[DailyDatasetEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"episodes index manifest not found: {path}")

    entries: list[DailyDatasetEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                DailyDatasetEntry(
                    date=str(row.get("date", "")),
                    slug=str(row.get("daily_dataset_slug", "")),
                    url=str(row.get("daily_dataset_url", "")),
                    episode_count=int(float(row.get("episode_count", 0) or 0)),
                    total_bytes=int(float(row.get("total_bytes", 0) or 0)),
                    top_avg_score=float(row.get("top_avg_score", 0) or 0),
                    median_avg_score=float(row.get("median_avg_score", 0) or 0),
                )
            )
    return entries


def latest_daily_entry(manifest: Iterable[DailyDatasetEntry]) -> DailyDatasetEntry | None:
    rows = sorted(manifest, key=lambda entry: entry.date)
    return rows[-1] if rows else None


def daily_slugs_for_top_games(
    manifest: list[DailyDatasetEntry],
    *,
    top_percent: float = 100.0,
    max_slugs: int = 0,
) -> list[str]:
    """Pick daily episode-bundle slugs from the episodes-index manifest.

    Days are ranked by ``top_avg_score`` (competition metadata). ``top_percent``
    keeps the best N%% of days; at 100%% only the latest day is used so we do
    not download the entire history every run.
    """
    if not manifest:
        return []
    ranked = sorted(manifest, key=lambda entry: entry.top_avg_score, reverse=True)
    percent = max(0.0, min(float(top_percent), 100.0))
    if percent >= 100.0:
        latest = latest_daily_entry(manifest)
        slugs = [latest.slug] if latest and latest.slug else []
    else:
        keep = max(1, int(round(len(ranked) * percent / 100.0)))
        slugs = [entry.slug for entry in ranked[:keep] if entry.slug]
    if max_slugs > 0:
        slugs = slugs[:max_slugs]
    return slugs


def local_daily_episodes_dir(root: Path, slug: str) -> Path:
    return root / "kaggle/input" / slug


def list_local_episode_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def replay_episode_id(replay_path: Path, payload: dict[str, Any] | None = None) -> str:
    if payload is not None:
        info = payload.get("info") or {}
        episode_id = info.get("EpisodeId")
        if episode_id is not None:
            return str(episode_id)
    return replay_path.stem


def score_replay(payload: dict[str, Any]) -> float:
    """Rank episodes for top-percent selection.

    Prefer longer games with decisive outcomes (more training transitions).
    """
    steps = payload.get("steps") or []
    rewards = payload.get("rewards") or []
    reward_span = 0.0
    if rewards:
        reward_span = max(float(value) for value in rewards) - min(float(value) for value in rewards)
    return float(len(steps)) + reward_span * 100.0


def load_replay_score(replay_path: Path) -> tuple[str, float]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    return replay_episode_id(replay_path, payload), score_replay(payload)


def episode_records_from_directory(
    directory: Path,
    *,
    source: str,
    scored: bool = True,
    workers: int | None = None,
) -> list[EpisodeRecord]:
    """Build episode records for a daily replay directory.

    ``scored=False`` skips opening any files (episode id taken from the filename
    stem, score 0) — use it when top-percent ranking is not needed, which avoids
    a full single-threaded parse pass over the whole bundle. When scoring is
    needed, replays are parsed in parallel across ``workers`` processes.
    """
    files = list_local_episode_files(directory)
    if not files:
        return []

    if not scored:
        return [
            EpisodeRecord(episode_id=path.stem, replay_path=path, score=0.0, source=source)
            for path in files
        ]

    worker_count = _default_score_workers() if workers is None else max(1, int(workers))
    if worker_count <= 1 or len(files) < worker_count * 2:
        records: list[EpisodeRecord] = []
        for replay_path in files:
            try:
                episode_id, score = load_replay_score(replay_path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    replay_path=replay_path,
                    score=score,
                    source=source,
                )
            )
        return records

    records = []
    chunksize = max(1, len(files) // (worker_count * 4))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for result in executor.map(
            _score_replay_path,
            [str(path) for path in files],
            chunksize=chunksize,
        ):
            if result is None:
                continue
            path_str, episode_id, score = result
            records.append(
                EpisodeRecord(
                    episode_id=episode_id,
                    replay_path=Path(path_str),
                    score=score,
                    source=source,
                )
            )
    print(f"scored {len(records):,} replays in {directory.name} using {worker_count} workers")
    return records


def episode_records_from_scrape_index(index_csv: Path) -> list[EpisodeRecord]:
    if not index_csv.is_file():
        return []

    records: list[EpisodeRecord] = []
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            replay_name = row.get("replay_file") or ""
            replay_path = index_csv.parent / "replays" / replay_name if replay_name else None
            score = float(row.get("submission_score") or 0)
            if replay_path is not None and replay_path.is_file():
                try:
                    _, replay_score = load_replay_score(replay_path)
                    score = max(score, replay_score)
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
            records.append(
                EpisodeRecord(
                    episode_id=str(row.get("episode_id") or ""),
                    replay_path=replay_path if replay_path and replay_path.is_file() else None,
                    score=score,
                    submission_id=str(row.get("submission_id") or ""),
                    team_id=str(row.get("team_id") or ""),
                    team_name=str(row.get("team_name") or ""),
                    source="ladder-scrape",
                )
            )
    return records


def filter_top_percent(records: list[EpisodeRecord], percent: float) -> list[EpisodeRecord]:
    if not records:
        return []
    percent = max(0.0, min(float(percent), 100.0))
    if percent <= 0:
        return []
    if percent >= 100:
        return list(records)

    ranked = sorted(records, key=lambda record: record.score, reverse=True)
    keep = max(1, int(round(len(ranked) * percent / 100.0)))
    return ranked[:keep]


def load_episode_pool(
    root: Path,
    *,
    index_path: Path | None = None,
    scrape_index_csv: Path | None = None,
    daily_slugs: list[str] | None = None,
    include_scrape: bool = False,
    top_percent_days: float = 100.0,
    needs_scoring: bool = True,
    workers: int | None = None,
) -> list[EpisodeRecord]:
    """Load replay JSON paths from the episodes-index dataset (and optionally ladder scrape).

    Bootstrap training uses ``include_scrape=False`` — top games come from the
    official ``pokemon-tcg-ai-battle-episodes-index`` dataset only. ``needs_scoring``
    can be set False to skip the (otherwise single-threaded) score pass when no
    top-percent ranking is required; scoring otherwise runs in parallel.
    """
    pool: list[EpisodeRecord] = []
    manifest_path = index_path or default_index_path(root)
    if manifest_path.is_file():
        manifest = load_daily_manifest(manifest_path)
        slugs = list(daily_slugs or [])
        if not slugs:
            slugs = daily_slugs_for_top_games(manifest, top_percent=top_percent_days)
        for slug in slugs:
            directory = local_daily_episodes_dir(root, slug)
            pool.extend(
                episode_records_from_directory(
                    directory,
                    source=slug,
                    scored=needs_scoring,
                    workers=workers,
                )
            )

    if include_scrape:
        scrape_path = scrape_index_csv or (root / "data/ladder-replays/index.csv")
        pool.extend(episode_records_from_scrape_index(scrape_path))

    deduped: dict[str, EpisodeRecord] = {}
    for record in pool:
        if not record.episode_id:
            continue
        existing = deduped.get(record.episode_id)
        if existing is None or record.score > existing.score:
            deduped[record.episode_id] = record
    return list(deduped.values())


def export_episode_ids(records: Iterable[EpisodeRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [record.episode_id for record in records if record.episode_id]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
