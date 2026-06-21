from __future__ import annotations

from typing import Any

# Defaults; overridden via poke_agent.config when labeling rollouts.
VALUE_WIN = 1.0
VALUE_NOT_WIN = -1.0
VALUE_TIMEOUT = -2.0  # harsh penalty for stalling to the time limit


def is_timeout_observation(obs: dict[str, Any] | None) -> bool:
    if obs is None:
        return False
    remaining = obs.get("remainingOverageTime")
    if remaining is not None and float(remaining) <= 0:
        return True
    current = obs.get("current") or {}
    return int(current.get("result", -1)) == 2


def is_decisive_result(result: int) -> bool:
    """True when the game ended with a winner (not draw/timeout/incomplete)."""
    return result in (0, 1)


def is_complete_episode(
    result: int,
    *,
    terminal_obs: dict[str, Any] | None = None,
    truncated: bool = False,
) -> bool:
    """Training uses complete decisive games only."""
    if truncated:
        return False
    if not is_decisive_result(result):
        return False
    if is_timeout_observation(terminal_obs):
        return False
    return True


def player_outcome_value(
    result: int,
    player: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
    value_timeout: float = VALUE_TIMEOUT,
    terminal_obs: dict[str, Any] | None = None,
) -> float:
    """Return training value: win +value_win, timeout value_timeout, else value_not_win."""
    if is_timeout_observation(terminal_obs):
        return float(value_timeout)
    if result >= 0 and result != 2 and player == result:
        return float(value_win)
    if result < 0 or result == 2:
        return float(value_not_win)
    return float(value_not_win)


def assign_episode_values(
    rows: list[dict[str, Any]],
    result: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
    value_timeout: float = VALUE_TIMEOUT,
    terminal_obs: dict[str, Any] | None = None,
) -> None:
    terminal_observation = terminal_obs
    if terminal_observation is None:
        for row in reversed(rows):
            if row.get("terminal"):
                terminal_observation = row.get("next_observation") or row.get("observation")
                break
    for row in rows:
        row["result"] = result
        row["value"] = player_outcome_value(
            result,
            int(row["player"]),
            value_win=value_win,
            value_not_win=value_not_win,
            value_timeout=value_timeout,
            terminal_obs=terminal_observation,
        )
        if row.get("terminal"):
            row["reward"] = row["value"]


def filter_complete_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop incomplete, truncated, draw, or timeout episodes from a JSONL row list."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)

    kept: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode):
        episode_rows = sorted(by_episode[episode_id], key=lambda row: int(row["step"]))
        if not episode_rows:
            continue
        result = int(episode_rows[-1].get("result", -1))
        terminal_obs = episode_rows[-1].get("next_observation") or episode_rows[-1].get("observation")
        truncated = bool(episode_rows[-1].get("truncated"))
        if not is_complete_episode(result, terminal_obs=terminal_obs, truncated=truncated):
            continue
        kept.extend(episode_rows)
    return kept
