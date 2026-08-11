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
    recovered_per_game: list[float] = []
    exhausted_recovery_per_game: list[float] = []
    contained_lane_faults_per_game: list[float] = []
    internal_value_boundaries_per_game: list[float] = []
    decisions_with_internal_boundary_per_game: list[float] = []
    max_internal_ordered_actions_per_game: list[float] = []
    search_latencies: list[float] = []
    backups: list[float] = []
    microbatches: list[float] = []
    mcts_wall_latencies: list[float] = []
    direct_wall_latencies: list[float] = []
    setup_wall_latencies: list[float] = []
    influence_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_seat: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_host: dict[str, Counter[str]] = defaultdict(Counter)
    influence_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    recovery_fault_codes: Counter[str] = Counter()
    recovery_fault_operations: Counter[str] = Counter()
    internal_boundary_reasons: Counter[str] = Counter()
    recovery_attempts = recovered_searches = exhausted_recovery_fallbacks = 0

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
            "recovered_searches": recovered_per_game,
            "exhausted_recovery_direct_fallbacks": exhausted_recovery_per_game,
            "contained_native_lane_faults": contained_lane_faults_per_game,
            "internal_value_boundaries": internal_value_boundaries_per_game,
            "decisions_with_internal_value_boundary": (
                decisions_with_internal_boundary_per_game
            ),
            "max_internal_ordered_action_count": (
                max_internal_ordered_actions_per_game
            ),
        }
        for name, target in fields.items():
            value = metrics.get(name)
            if not isinstance(value, int) or value < 0:
                raise R229MetricsError(f"invalid decision metric: {name}")
            target.append(float(value))
        for name in (
            "mcts_seat_decisions_seen", "direct_seat_decisions_seen",
            "setup_decisions",
        ):
            if not isinstance(metrics.get(name), int) or metrics[name] < 0:
                raise R229MetricsError(f"invalid decision metric: {name}")
        if metrics["decisions_seen"] != (
            metrics["mcts_seat_decisions_seen"]
            + metrics["direct_seat_decisions_seen"]
        ):
            raise R229MetricsError("per-arm decisions do not match decisions_seen")
        latency_rows = row.get("decision_latency_seconds")
        if not isinstance(latency_rows, Mapping):
            raise R229MetricsError("per-arm decision latency telemetry is required")
        for name, expected_count, target in (
            ("mcts_seat_all", metrics["mcts_seat_decisions_seen"], mcts_wall_latencies),
            ("direct_r195_seat_all", metrics["direct_seat_decisions_seen"], direct_wall_latencies),
            ("deterministic_setup", metrics["setup_decisions"], setup_wall_latencies),
        ):
            values = latency_rows.get(name)
            if not isinstance(values, list) or len(values) != expected_count:
                raise R229MetricsError(f"decision latency count mismatch: {name}")
            for value in values:
                clean = float(value)
                if not math.isfinite(clean) or clean < 0.0:
                    raise R229MetricsError(f"invalid decision latency: {name}")
                target.append(clean)
        if metrics["meaningful_choice_change"] > metrics["action_changed"] or metrics["action_changed"] > metrics["searched"]:
            raise R229MetricsError("decision influence counts are inconsistent")
        for name in (
            "internal_explicit_chance_boundaries",
            "internal_deterministic_fanout_boundaries",
        ):
            if not isinstance(metrics.get(name), int) or metrics[name] < 0:
                raise R229MetricsError(f"invalid decision metric: {name}")
        if (
            metrics["decisions_with_internal_value_boundary"]
            > metrics["mcts_eligible"]
            or metrics["internal_explicit_chance_boundaries"]
            + metrics["internal_deterministic_fanout_boundaries"]
            != metrics["internal_value_boundaries"]
        ):
            raise R229MetricsError("internal leaf-boundary counts are inconsistent")
        decision_rows = row.get("mcts_decisions", [])
        if not isinstance(decision_rows, list):
            raise R229MetricsError("mcts_decisions must be a list")
        telemetry_searched = telemetry_changed = telemetry_meaningful = 0
        telemetry_recovered = telemetry_exhausted = telemetry_lane_faults = 0
        telemetry_boundaries = telemetry_boundary_decisions = 0
        telemetry_max_internal = 0
        telemetry_boundary_reasons: Counter[str] = Counter()
        for decision in decision_rows:
            if not isinstance(decision, Mapping):
                raise R229MetricsError("malformed MCTS decision telemetry")
            mode = str(decision.get("mode", ""))
            recovery = decision.get("lane_process_recovery")
            if not isinstance(recovery, Mapping):
                raise R229MetricsError("decision lacks process-lane recovery telemetry")
            if recovery.get("serial_lane_count") != 1:
                raise R229MetricsError("decision recovery telemetry is not serial")
            boundary_count = decision.get("internal_value_boundary_count")
            reasons = decision.get("internal_value_boundary_reasons")
            max_internal = decision.get("max_internal_ordered_action_count")
            if (
                isinstance(boundary_count, bool)
                or not isinstance(boundary_count, int)
                or boundary_count < 0
                or not isinstance(reasons, Mapping)
                or any(
                    reason
                    not in {
                        "explicit_chance_pre_random",
                        "deterministic_internal_fanout_over_64",
                    }
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    for reason, count in reasons.items()
                )
                or sum(reasons.values()) != boundary_count
                or isinstance(max_internal, bool)
                or not isinstance(max_internal, int)
                or max_internal < 0
                or decision.get("internal_ordered_action_expansion_ceiling") != 64
                or decision.get("explicit_chance_probability_distribution_assumed")
                is not False
                or decision.get(
                    "explicit_chance_always_stops_before_random_resolution"
                )
                is not True
                or decision.get("internal_boundary_has_action_or_child_authority")
                is not False
            ):
                raise R229MetricsError(
                    "decision lacks exact r252 internal-boundary telemetry"
                )
            if (
                reasons.get("deterministic_internal_fanout_over_64", 0) > 0
                and max_internal <= 64
            ):
                raise R229MetricsError(
                    "oversized internal boundary did not exceed 64"
                )
            telemetry_boundaries += boundary_count
            telemetry_boundary_decisions += int(boundary_count > 0)
            telemetry_max_internal = max(telemetry_max_internal, max_internal)
            telemetry_boundary_reasons.update(reasons)
            internal_boundary_reasons.update(reasons)
            attempt_count = recovery.get("attempt_count")
            attempts = recovery.get("attempts")
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count not in (1, 2)
                or not isinstance(attempts, list)
                or len(attempts) != attempt_count
            ):
                raise R229MetricsError("decision has malformed recovery attempts")
            recovered = recovery.get("recovered_search") is True
            exhausted = recovery.get("exhausted_direct_fallback") is True
            if recovered != (mode == "shared_tree_mcts" and attempt_count == 2):
                raise R229MetricsError("recovered-search telemetry is inconsistent")
            if exhausted != (
                mode == "bounded_lane_recovery_exhausted_direct_fallback"
            ):
                raise R229MetricsError("exhausted-recovery telemetry is inconsistent")
            changed_flag = bool(decision.get("action_changed"))
            if exhausted and changed_flag:
                raise R229MetricsError("degraded direct fallback gained change credit")
            decision_faults = 0
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    raise R229MetricsError("malformed recovery attempt row")
                faults = attempt.get("new_lane_faults", [])
                if not isinstance(faults, list):
                    raise R229MetricsError("malformed lane-fault telemetry")
                for fault in faults:
                    if not isinstance(fault, Mapping):
                        raise R229MetricsError("malformed lane-fault row")
                    recovery_fault_codes[str(fault.get("code", "unavailable"))] += 1
                    recovery_fault_operations[
                        str(fault.get("operation", "unavailable"))
                    ] += 1
                    reap = fault.get("reap")
                    if isinstance(reap, Mapping) and reap.get("reaped") is not True:
                        raise R229MetricsError("lane fault lacks a successful bounded reap")
                    decision_faults += 1
            recovery_attempts += attempt_count
            recovered_searches += int(recovered)
            exhausted_recovery_fallbacks += int(exhausted)
            telemetry_recovered += int(recovered)
            telemetry_exhausted += int(exhausted)
            telemetry_lane_faults += decision_faults
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
            or telemetry_recovered != metrics["recovered_searches"]
            or telemetry_exhausted
            != metrics["exhausted_recovery_direct_fallbacks"]
            or telemetry_lane_faults != metrics["contained_native_lane_faults"]
            or telemetry_boundaries != metrics["internal_value_boundaries"]
            or telemetry_boundary_decisions
            != metrics["decisions_with_internal_value_boundary"]
            or telemetry_boundary_reasons.get("explicit_chance_pre_random", 0)
            != metrics["internal_explicit_chance_boundaries"]
            or telemetry_boundary_reasons.get(
                "deterministic_internal_fanout_over_64", 0
            )
            != metrics["internal_deterministic_fanout_boundaries"]
            or telemetry_max_internal != metrics["max_internal_ordered_action_count"]
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
            "recovered_searches_per_game": _quantiles(recovered_per_game),
            "exhausted_recovery_fallbacks_per_game": _quantiles(
                exhausted_recovery_per_game
            ),
            "contained_native_lane_faults_per_game": _quantiles(
                contained_lane_faults_per_game
            ),
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
            "mcts_seat_wall_latency_seconds": _quantiles(mcts_wall_latencies),
            "direct_r195_wall_latency_seconds": _quantiles(direct_wall_latencies),
            "deterministic_setup_wall_latency_seconds": _quantiles(setup_wall_latencies),
            "mcts_seat_decisions_per_second": (
                len(mcts_wall_latencies) / max(1e-9, sum(mcts_wall_latencies))
            ),
            "direct_r195_decisions_per_second": (
                len(direct_wall_latencies) / max(1e-9, sum(direct_wall_latencies))
            ),
            "mean_mcts_to_direct_decision_latency_ratio": (
                statistics.fmean(mcts_wall_latencies)
                / max(1e-9, statistics.fmean(direct_wall_latencies))
                if mcts_wall_latencies and direct_wall_latencies
                else None
            ),
        },
        "process_lane_recovery": {
            "decision_attempts_total": recovery_attempts,
            "extra_attempts_total": recovery_attempts - int(sum(eligible_per_game)),
            "recovered_searches_total": recovered_searches,
            "exhausted_direct_fallbacks_total": exhausted_recovery_fallbacks,
            "recovered_search_rate_per_eligible": recovered_searches
            / max(1, int(sum(eligible_per_game))),
            "exhausted_fallback_rate_per_eligible": exhausted_recovery_fallbacks
            / max(1, int(sum(eligible_per_game))),
            "contained_lane_faults_total": int(sum(contained_lane_faults_per_game)),
            "fault_codes": dict(recovery_fault_codes),
            "fault_operations": dict(recovery_fault_operations),
        },
        "internal_leaf_boundaries": {
            "root_complete_ordered_action_ceiling": 65536,
            "deterministic_internal_expansion_ceiling": 64,
            "all_explicit_chance_contexts_stop_pre_random": True,
            "probability_distribution_assumed": False,
            "total": int(sum(internal_value_boundaries_per_game)),
            "decisions_with_boundary_total": int(
                sum(decisions_with_internal_boundary_per_game)
            ),
            "total_per_game": _quantiles(internal_value_boundaries_per_game),
            "decisions_with_boundary_per_game": _quantiles(
                decisions_with_internal_boundary_per_game
            ),
            "max_internal_ordered_action_count_per_game": _quantiles(
                max_internal_ordered_actions_per_game
            ),
            "reasons": dict(internal_boundary_reasons),
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
