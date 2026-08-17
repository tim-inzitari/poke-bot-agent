"""Calendar-day train/validation splits for Slowking replay seats."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .authority import RESEARCH_ONLY, SPLIT_SCHEMA


@dataclass(frozen=True)
class DaySplit:
    """Immutable day-based split; never frame-level."""

    train_dates: tuple[str, ...]
    val_dates: tuple[str, ...]
    train_game_ids: tuple[str, ...]
    val_game_ids: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SPLIT_SCHEMA,
            "research_only": RESEARCH_ONLY,
            "split_unit": "calendar_day",
            "train_dates": list(self.train_dates),
            "val_dates": list(self.val_dates),
            "train_games": len(self.train_game_ids),
            "val_games": len(self.val_game_ids),
            "train_game_ids": list(self.train_game_ids),
            "val_game_ids": list(self.val_game_ids),
        }


def game_id(game: dict[str, Any], *, source_date: str = "") -> str:
    existing = game.get("game_id")
    if existing:
        return str(existing)
    episode = str(game.get("episode_id") or "")
    seat = int(game.get("seat", -1))
    date = str(game.get("source_date") or source_date or "")
    return f"{date}:{episode}:{seat}"


def assign_dates_from_aggregate(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach source_date to each game using daily receipt metadata when present."""
    date_by_path = {
        str(row.get("path")): str(row.get("date") or "")
        for row in (aggregate.get("daily_receipts") or [])
    }
    # If games already carry source_date, keep it.
    out: list[dict[str, Any]] = []
    for game in aggregate.get("games") or []:
        row = dict(game)
        if not row.get("source_date"):
            # Fallback: unknown date bucket keyed empty; callers should stamp dates
            # when extracting from archives.
            row["source_date"] = row.get("source_date") or ""
        out.append(row)
    del date_by_path
    return out


def build_day_split(
    games: Iterable[dict[str, Any]],
    *,
    val_dates: Iterable[str],
    require_dates: bool = True,
) -> DaySplit:
    """Split by calendar day. Validation days are held out entirely."""
    val = {str(d) for d in val_dates}
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        date = str(game.get("source_date") or "")
        if require_dates and not date:
            raise ValueError("game missing source_date; cannot day-split")
        by_date[date].append(game)
    train_dates = tuple(sorted(d for d in by_date if d not in val))
    val_dates_t = tuple(sorted(d for d in by_date if d in val))
    train_ids = tuple(
        game_id(g) for d in train_dates for g in by_date[d]
    )
    val_ids = tuple(game_id(g) for d in val_dates_t for g in by_date[d])
    return DaySplit(
        train_dates=train_dates,
        val_dates=val_dates_t,
        train_game_ids=train_ids,
        val_game_ids=val_ids,
    )


def default_val_dates_for_window(dates: Iterable[str]) -> list[str]:
    """Hold out the last distinct calendar day when >= 2 days exist."""
    ordered = sorted({str(d) for d in dates if d})
    if not ordered:
        return []
    if len(ordered) == 1:
        return []
    return [ordered[-1]]


def write_split(split: DaySplit, out: Path) -> str:
    payload = split.to_json()
    text = json.dumps(payload, indent=2) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
