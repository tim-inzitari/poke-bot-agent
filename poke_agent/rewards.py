from __future__ import annotations

from typing import Any

# Defaults; overridden via poke_agent.config when labeling rollouts.
VALUE_WIN = 1.0
VALUE_NOT_WIN = -1.0


def is_timeout_observation(obs: dict[str, Any] | None) -> bool:
    if obs is None:
        return False
    remaining = obs.get("remainingOverageTime")
    if remaining is not None and float(remaining) <= 0:
        return True
    current = obs.get("current") or {}
    return int(current.get("result", -1)) == 2


def player_outcome_value(
    result: int,
    player: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
    terminal_obs: dict[str, Any] | None = None,
) -> float:
    """Return training value: win +value_win, else value_not_win (loss/draw/timeout)."""
    if result >= 0 and result != 2 and player == result:
        return float(value_win)
    if is_timeout_observation(terminal_obs):
        return float(value_not_win)
    if result < 0 or result == 2:
        return float(value_not_win)
    return float(value_not_win)


def assign_episode_values(
    rows: list[dict[str, Any]],
    result: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
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
            terminal_obs=terminal_observation,
        )
        if row.get("terminal"):
            row["reward"] = row["value"]
