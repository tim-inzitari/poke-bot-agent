"""Fail-closed aggregation for the r229 fleet MCTS mirror receipts."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping

SCHEMA = "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r229_summary/v1"
EXPECTED_GAMES = 1000
EXPECTED_PAIRS = 500


class R229MetricsError(ValueError):
    pass


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    ordered = sorted(values)
    at = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": at(0.10),
        "p90": at(0.90),
    }


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float | None]:
    if trials <= 0:
        return [None, None]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_games(rows: Iterable[Mapping[str, Any]], *, require_complete: bool = True) -> dict[str, Any]:
    games = [dict(row) for row in rows]
    expected = EXPECTED_GAMES if require_complete else len(games)
    if len(games) != expected:
        raise R229MetricsError(f"expected {expected} games, got {len(games)}")
    ids = [str(row.get("game_id", "")) for row in games]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise R229MetricsError("game identities are missing or duplicated")

    pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    wins = Counter()
    host_games = Counter()
    host_seconds: dict[str, float] = defaultdict(float)
    host_started: dict[str, list[datetime]] = defaultdict(list)
    host_completed: dict[str, list[datetime]] = defaultdict(list)
    decisions_per_game: list[float] = []
    eligible_per_game: list[float] = []
    searched_per_game: list[float] = []
    changed_per_game: list[float] = []
    meaningful_per_game: list[float] = []
    forced_per_game: list[float] = []
    fallback_per_game: list[float] = []
    search_latencies: list[float] = []
    backups: list[float] = []
    microbatches: list[float] = []
    influence_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_seat: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_host: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)

    for row in games:
        pair = row.get("pair_index")
        seat = row.get("mcts_seat")
        winner = row.get("winner_seat")
        if not isinstance(pair, int) or pair < 0 or pair >= EXPECTED_PAIRS or seat not in (0, 1):
            raise R229MetricsError("invalid pair or MCTS seat")
        if winner not in (0, 1, 2):
            raise R229MetricsError("winner_seat must be 0, 1, or 2 for draw")
        pairs[pair].append(row)
        outcome = "draw" if winner == 2 else ("mcts_win" if winner == seat else "direct_win")
        wins[outcome] += 1
        wins[f"{outcome}_mcts_seat_{seat}"] += 1
        host = str(row.get("host", ""))
        elapsed = float(row.get("elapsed_seconds", 0.0))
        if not host or not math.isfinite(elapsed) or elapsed <= 0.0:
            raise R229MetricsError("host and positive finite elapsed_seconds are required")
        host_games[host] += 1
        host_seconds[host] += elapsed
        try:
            host_started[host].append(
                datetime.fromisoformat(str(row["started_at_utc"]).replace("Z", "+00:00"))
            )
            host_completed[host].append(
                datetime.fromisoformat(str(row["completed_at_utc"]).replace("Z", "+00:00"))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise R229MetricsError("game start/completion UTC timestamps are required") from exc
        metrics = row.get("decision_metrics")
        if not isinstance(metrics, Mapping):
            raise R229MetricsError("decision_metrics are required")
        fields = {
            "decisions_seen": decisions_per_game,
            "mcts_eligible": eligible_per_game,
            "searched": searched_per_game,
            "action_changed": changed_per_game,
            "meaningful_choice_change": meaningful_per_game,
            "forced": forced_per_game,
            "fallback": fallback_per_game,
        }
        for name, target in fields.items():
            value = metrics.get(name)
            if not isinstance(value, int) or value < 0:
                raise R229MetricsError(f"invalid decision metric: {name}")
            target.append(float(value))
        if metrics["meaningful_choice_change"] > metrics["action_changed"] or metrics["action_changed"] > metrics["searched"]:
            raise R229MetricsError("decision influence counts are inconsistent")
        decision_rows = row.get("mcts_decisions", [])
        if not isinstance(decision_rows, list):
            raise R229MetricsError("mcts_decisions must be a list")
        telemetry_searched = telemetry_changed = telemetry_meaningful = 0
        for decision in decision_rows:
            if not isinstance(decision, Mapping):
                raise R229MetricsError("malformed MCTS decision telemetry")
            mode = str(decision.get("mode", ""))
            changed_flag = bool(decision.get("action_changed"))
            meaningful_flag = bool(decision.get("meaningful_choice_change"))
            if meaningful_flag and not changed_flag:
                raise R229MetricsError("meaningful decision change lacks raw action change")
            stage = str(decision.get("selection_context", "unavailable"))
            actor = str(decision.get("actor_seat", seat))
            for bucket in (
                influence_by_stage[stage], influence_by_seat[actor],
                influence_by_host[host], influence_by_outcome[outcome],
            ):
                bucket["decisions"] += 1
                bucket["action_changed"] += int(changed_flag)
                bucket["meaningful_choice_change"] += int(meaningful_flag)
            if mode == "shared_tree_mcts":
                telemetry_searched += 1
                latency = decision.get("search_elapsed_seconds")
                completed = decision.get("completed_backups")
                if latency is None or completed is None:
                    raise R229MetricsError("searched decision lacks latency or backup telemetry")
                search_latencies.append(float(latency))
                backups.append(float(completed))
                microbatches.extend(
                    float(value) for value in decision.get("microbatch_sizes", [])
                )
            telemetry_changed += int(changed_flag)
            telemetry_meaningful += int(meaningful_flag)
        if (
            len(decision_rows) != metrics["mcts_eligible"]
            or telemetry_searched != metrics["searched"]
            or telemetry_changed != metrics["action_changed"]
            or telemetry_meaningful != metrics["meaningful_choice_change"]
        ):
            raise R229MetricsError("per-decision telemetry does not match game counters")

    if require_complete:
        if set(pairs) != set(range(EXPECTED_PAIRS)) or any(len(rows) != 2 for rows in pairs.values()):
            raise R229MetricsError("pair coverage is incomplete")
        for pair_rows in pairs.values():
            if {row["mcts_seat"] for row in pair_rows} != {0, 1}:
                raise R229MetricsError("each pair must seat-swap MCTS exactly once")

    changed = int(sum(changed_per_game))
    meaningful = int(sum(meaningful_per_game))
    searched = int(sum(searched_per_game))
    pair_outcomes = Counter()
    for pair_rows in pairs.values():
        pair_mcts = sum(row["winner_seat"] == row["mcts_seat"] for row in pair_rows)
        pair_direct = sum(row["winner_seat"] in (0, 1) and row["winner_seat"] != row["mcts_seat"] for row in pair_rows)
        pair_outcomes[f"mcts_{pair_mcts}_direct_{pair_direct}"] += 1
    decisive = wins["mcts_win"] + wins["direct_win"]
    win_rate = wins["mcts_win"] / max(1, decisive)
    return {
        "schema": SCHEMA,
        "games": len(games),
        "pairs": len(pairs),
        "outcomes": dict(wins),
        "mcts_win_rate_excluding_draws": win_rate,
        "mcts_win_rate_wilson_95": _wilson(wins["mcts_win"], decisive),
        "mcts_win_rate_effect_vs_even": win_rate - 0.5,
        "pair_outcomes": dict(pair_outcomes),
        "decisions": {
            "seen_total": int(sum(decisions_per_game)),
            "eligible_total": int(sum(eligible_per_game)),
            "seen_per_game": _quantiles(decisions_per_game),
            "eligible_per_game": _quantiles(eligible_per_game),
            "searched_per_game": _quantiles(searched_per_game),
            "changed_per_game": _quantiles(changed_per_game),
            "meaningful_changed_per_game": _quantiles(meaningful_per_game),
            "forced_per_game": _quantiles(forced_per_game),
            "fallback_per_game": _quantiles(fallback_per_game),
            "searched_total": searched,
            "action_changed_total": changed,
            "meaningful_choice_change_total": meaningful,
            "action_change_rate_per_searched": changed / max(1, searched),
            "meaningful_change_rate_per_searched": meaningful / max(1, searched),
            "influence_by_stage": {
                key: dict(value) for key, value in sorted(influence_by_stage.items())
            },
            "influence_by_seat": {
                key: dict(value) for key, value in sorted(influence_by_seat.items())
            },
            "influence_by_host": {
                key: dict(value) for key, value in sorted(influence_by_host.items())
            },
            "influence_by_game_outcome": {
                key: dict(value) for key, value in sorted(influence_by_outcome.items())
            },
        },
        "search": {
            "latency_seconds": _quantiles(search_latencies),
            "completed_backups": _quantiles(backups),
            "microbatch_size": _quantiles(microbatches),
        },
        "throughput": {
            "by_host": {
                host: {
                    "games": count,
                    "summed_game_seconds": host_seconds[host],
                    "worker_time_games_per_second": count / host_seconds[host],
                    "wall_span_seconds": max(
                        1e-9,
                        (max(host_completed[host]) - min(host_started[host])).total_seconds(),
                    ),
                    "wall_span_games_per_hour": count * 3600.0 / max(
                        1e-9,
                        (max(host_completed[host]) - min(host_started[host])).total_seconds(),
                    ),
                }
                for host, count in sorted(host_games.items())
            },
            "aggregate_worker_games_per_second": len(games) / sum(host_seconds.values()),
        },
    }


__all__ = ["R229MetricsError", "summarize_games"]
