"""Aggregate daily Slowking distill receipts into a multi-day boundary."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .authority import RESEARCH_ONLY, RUNTIME_AUTHORITY, TRAINING_AUTHORITY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_daily_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "poke_bot.slowking_top_replay_distillation/v1":
        raise ValueError(f"unexpected daily schema in {path}")
    return payload


def aggregate_daily_receipts(
    daily_paths: list[Path],
    *,
    window_start: str,
    window_end: str,
) -> dict[str, Any]:
    """Fold checksum-bound daily receipts into one research boundary artifact."""
    dailies: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    list_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    manifest_sum = 0
    members_sum = 0
    scanned_sum = 0

    for path in daily_paths:
        receipt = load_daily_receipt(path)
        source = receipt.get("source") or {}
        identity = receipt.get("identity") or {}
        outcomes = receipt.get("outcomes") or {}
        daily_games = list(receipt.get("games") or [])
        fingerprint_games = {
            row["fingerprint"]: int(row["games"])
            for row in (identity.get("decks") or [])
            if isinstance(row, dict) and "fingerprint" in row
        }
        for fp, n in fingerprint_games.items():
            list_counts[fp] += n
        for game in daily_games:
            games.append(dict(game))
            team_counts[str(game.get("team_name") or "")] += 1
        manifest_sum += int(source.get("manifest_episode_count") or 0)
        members_sum += int(source.get("archive_json_members") or 0)
        scanned_sum += int(source.get("episodes_scanned") or len(daily_games))
        dailies.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "date": source.get("date"),
                "games": int(outcomes.get("games") or len(daily_games)),
                "archive_sha256": source.get("archive_sha256"),
            }
        )

    wins = sum(1 for g in games if g.get("result") == "win")
    losses = sum(1 for g in games if g.get("result") == "loss")
    draws = len(games) - wins - losses
    strata = [
        {
            "fingerprint": fp,
            "games": count,
            "team_names": sorted(
                {
                    str(g.get("team_name") or "")
                    for g in games
                    if g.get("deck_fingerprint") == fp
                }
            ),
        }
        for fp, count in list_counts.most_common()
    ]
    return {
        "schema": "poke_bot.slowking_multi_day_replay_distillation/v1",
        "status": "research_only_aggregated_from_daily_receipts",
        "authority": RUNTIME_AUTHORITY,
        "training_authority": TRAINING_AUTHORITY,
        "research_only": RESEARCH_ONLY,
        "window": {"start": window_start, "end": window_end},
        "archive_scan": {
            "daily_archives": len(dailies),
            "manifest_episode_count_sum": manifest_sum,
            "archive_json_members_sum": members_sum,
            "episodes_scanned": scanned_sum,
        },
        "identity": {
            "slowking_seats": len(games),
            "unique_team_names": sorted(k for k in team_counts if k),
            "unique_deck_lists": len(list_counts),
            "strata": strata,
            "team_counts": dict(team_counts),
        },
        "outcomes": {
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(games) if games else None,
        },
        "daily_receipts": dailies,
        "games": games,
    }


def write_aggregate(payload: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
