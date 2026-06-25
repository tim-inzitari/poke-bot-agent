"""Standalone prize helpers for the Kaggle submission bundle (no poke_agent package)."""

from __future__ import annotations

from typing import Any

DEFAULT_PRIZE_COUNT = 6


def prize_count(obs: dict[str, Any] | None, player_index: int) -> int:
    """Remaining prize cards for a fixed seat (0 or 1)."""
    if obs is None:
        return DEFAULT_PRIZE_COUNT
    players = (obs.get("current") or {}).get("players") or [{}, {}]
    if player_index >= len(players):
        return DEFAULT_PRIZE_COUNT
    player = players[player_index] or {}
    prize = player.get("prize")
    if isinstance(prize, list):
        return len(prize)
    raw = player.get("prizeCount")
    if raw is not None:
        return int(raw)
    return DEFAULT_PRIZE_COUNT


def winner_from_prizes(terminal_obs: dict[str, Any] | None) -> int | None:
    """Winner is the player whose own prize count reached 0."""
    if terminal_obs is None:
        return None
    p0_own = prize_count(terminal_obs, 0)
    p1_own = prize_count(terminal_obs, 1)
    if p0_own == 0 and p1_own > 0:
        return 0
    if p1_own == 0 and p0_own > 0:
        return 1
    return None


def winner_from_prizes(terminal_obs: dict[str, Any] | None) -> int | None:
    """Winner is the player whose own prize count reached 0."""
    if terminal_obs is None:
        return None
    p0_own = prize_count(terminal_obs, 0)
    p1_own = prize_count(terminal_obs, 1)
    if p0_own == 0 and p1_own > 0:
        return 0
    if p1_own == 0 and p0_own > 0:
        return 1
    return None
