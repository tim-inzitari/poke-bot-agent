from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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


def episode_records_from_directory(directory: Path, *, source: str) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for replay_path in list_local_episode_files(directory):
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
) -> list[EpisodeRecord]:
    pool: list[EpisodeRecord] = []
    manifest_path = index_path or default_index_path(root)
    if manifest_path.is_file():
        manifest = load_daily_manifest(manifest_path)
        slugs = daily_slugs or []
        if not slugs:
            latest = latest_daily_entry(manifest)
            if latest is not None:
                slugs = [latest.slug]
        for slug in slugs:
            directory = local_daily_episodes_dir(root, slug)
            pool.extend(episode_records_from_directory(directory, source=slug))

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
