from __future__ import annotations

from typing import Any

# Defaults; overridden via poke_agent.config when labeling rollouts.
VALUE_WIN = 1.0
VALUE_NOT_WIN = -1.0
VALUE_TIMEOUT = -2.0  # harsh penalty for stalling to the time limit

# Standard prize count at game start. Win = YOUR remaining prizes hit 0 (by taking prizes on enemy KOs).
DEFAULT_PRIZE_COUNT = 6
VALUE_PER_OWN_PRIZE_TAKEN = VALUE_WIN / DEFAULT_PRIZE_COUNT
VALUE_PER_OPP_PRIZE_TAKEN = VALUE_NOT_WIN / DEFAULT_PRIZE_COUNT


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


def seat_prize_counts(obs: dict[str, Any] | None, *, seat: int) -> tuple[int, int]:
    """Return (your_remaining_prizes, opponent_remaining_prizes) for the acting seat."""
    opponent = 1 - int(seat)
    return prize_count(obs, seat), prize_count(obs, opponent)


def winner_from_prizes(terminal_obs: dict[str, Any] | None) -> int | None:
    """Winner is the player whose **own** prize count reached 0.

  PTCG rule: you take prize cards when you KO opponent Pokémon; you win when
  **your** prize pile is empty — not when the opponent's prizes hit 0.
    """
    if terminal_obs is None:
        return None
    p0_own = prize_count(terminal_obs, 0)
    p1_own = prize_count(terminal_obs, 1)
    if p0_own == 0 and p1_own > 0:
        return 0
    if p1_own == 0 and p0_own > 0:
        return 1
    return None


def inverted_opponent_prize_wincon(terminal_obs: dict[str, Any] | None) -> int | None:
    """WRONG wincon (opponent prizes → 0). Used only in tests to guard against regressions."""
    if terminal_obs is None:
        return None
    p0_own = prize_count(terminal_obs, 0)
    p1_own = prize_count(terminal_obs, 1)
    if p1_own == 0 and p0_own > 0:
        return 0
    if p0_own == 0 and p1_own > 0:
        return 1
    return None


def resolve_game_result(result: int, terminal_obs: dict[str, Any] | None) -> int:
    """Prefer prize-based winner; fall back to CABT result index."""
    prize_winner = winner_from_prizes(terminal_obs)
    if prize_winner is not None:
        return prize_winner
    return result


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
    resolved = resolve_game_result(result, terminal_obs)
    if not is_decisive_result(resolved):
        return False
    if is_timeout_observation(terminal_obs):
        return False
    return True


def step_prize_reward(
    obs_before: dict[str, Any] | None,
    obs_after: dict[str, Any] | None,
    player: int,
    *,
    value_per_own_prize_taken: float = VALUE_PER_OWN_PRIZE_TAKEN,
    value_per_opp_prize_taken: float = VALUE_PER_OPP_PRIZE_TAKEN,
) -> float:
    """Reward taking YOUR prizes (enemy KO); penalize when opponent takes theirs."""
    self_before, opp_before = seat_prize_counts(obs_before, seat=player)
    self_after, opp_after = seat_prize_counts(obs_after, seat=player)
    own_taken = max(0, self_before - self_after)
    opp_taken = max(0, opp_before - opp_after)
    return (
        own_taken * float(value_per_own_prize_taken)
        + opp_taken * float(value_per_opp_prize_taken)
    )


def prize_progress_value(
    obs: dict[str, Any] | None,
    player: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
    default_prizes: int = DEFAULT_PRIZE_COUNT,
) -> float:
    """Value estimate from prize progress: good when YOUR prizes approach 0, bad when theirs do."""
    self_prizes, opp_prizes = seat_prize_counts(obs, seat=player)
    own_progress = (default_prizes - self_prizes) / float(default_prizes)
    opp_progress = (default_prizes - opp_prizes) / float(default_prizes)
    return float(own_progress * value_win + opp_progress * value_not_win)


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
    resolved = resolve_game_result(result, terminal_obs)
    if resolved >= 0 and resolved != 2 and player == resolved:
        return float(value_win)
    if resolved < 0 or resolved == 2:
        return float(value_not_win)
    return float(value_not_win)


def assign_episode_values(
    rows: list[dict[str, Any]],
    result: int,
    *,
    value_win: float = VALUE_WIN,
    value_not_win: float = VALUE_NOT_WIN,
    value_timeout: float = VALUE_TIMEOUT,
    value_per_own_prize_taken: float = VALUE_PER_OWN_PRIZE_TAKEN,
    value_per_opp_prize_taken: float = VALUE_PER_OPP_PRIZE_TAKEN,
    terminal_obs: dict[str, Any] | None = None,
) -> None:
    terminal_observation = terminal_obs
    if terminal_observation is None:
        for row in reversed(rows):
            if row.get("terminal"):
                terminal_observation = row.get("next_observation") or row.get("observation")
                break

    resolved_result = resolve_game_result(result, terminal_observation)

    for row in rows:
        player = int(row["player"])
        obs_before = row.get("observation")
        obs_after = row.get("next_observation") or obs_before
        row["result"] = resolved_result
        row["self_prize_remaining"] = prize_count(obs_before, player)
        row["opp_prize_remaining"] = prize_count(obs_before, 1 - player)

        step_reward = step_prize_reward(
            obs_before,
            obs_after,
            player,
            value_per_own_prize_taken=value_per_own_prize_taken,
            value_per_opp_prize_taken=value_per_opp_prize_taken,
        )
        row["reward"] = step_reward

        if row.get("terminal"):
            row["value"] = player_outcome_value(
                resolved_result,
                player,
                value_win=value_win,
                value_not_win=value_not_win,
                value_timeout=value_timeout,
                terminal_obs=terminal_observation,
            )
            row["reward"] = step_reward + row["value"]
        else:
            row["value"] = prize_progress_value(
                obs_after,
                player,
                value_win=value_win,
                value_not_win=value_not_win,
            )


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
        raw_result = int(episode_rows[-1].get("result", -1))
        terminal_obs = episode_rows[-1].get("next_observation") or episode_rows[-1].get("observation")
        truncated = bool(episode_rows[-1].get("truncated"))
        if not is_complete_episode(raw_result, terminal_obs=terminal_obs, truncated=truncated):
            continue
        kept.extend(episode_rows)
    return kept
