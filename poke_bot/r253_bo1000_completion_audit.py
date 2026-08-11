"""Immutable post-run audit for the revision-253 fleet BO1000.

The live dispatcher intentionally carries only the code that is needed to run
games safely.  This module is a separate, read-only finalizer: it revalidates
all 1,000 game receipts, binds their bytes, and aggregates the counterfactual
decision-quality fields needed for the owner review.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .r229_fleet_mirror_metrics import R229MetricsError, summarize_games


SCHEMA = "poke_bot.r253_bo1000_completion_audit/v1"
EXPECTED_GAMES = 1000


class R253CompletionAuditError(ValueError):
    """The completed fleet evidence is missing, inconsistent, or mutable."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise R253CompletionAuditError(f"unreadable JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise R253CompletionAuditError(f"JSON receipt is not an object: {path}")
    return payload


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in clean):
        raise R253CompletionAuditError("decision-quality metric is non-finite")
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    def at(quantile: float) -> float:
        position = (len(clean) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(clean) - 1)
        return clean[lower] * (upper - position) + clean[upper] * (position - lower)

    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p10": at(0.10),
        "p90": at(0.90),
        "p95": at(0.95),
        "min": clean[0],
        "max": clean[-1],
    }


def summarize_decision_quality(
    games: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate how far MCTS moved from its same-state direct control."""

    decisions: list[dict[str, Any]] = []
    for game in games:
        rows = game.get("mcts_decisions")
        if not isinstance(rows, list):
            raise R253CompletionAuditError("game lacks MCTS decision rows")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise R253CompletionAuditError("MCTS decision row is malformed")
            row = dict(raw)
            if row.get("mode") == "shared_tree_mcts":
                decisions.append(row)

    required = (
        "direct_action_probability",
        "mcts_action_direct_probability",
        "direct_probability_gap",
        "mcts_action_direct_rank",
        "selected_action_value",
        "selected_action_visits",
        "root_visits",
        "legal_action_count",
        "distinct_root_actions_visited",
        "rollout_count",
    )
    for row in decisions:
        if any(name not in row for name in required):
            raise R253CompletionAuditError(
                "searched decision lacks counterfactual quality telemetry"
            )
        if any(
            isinstance(row[name], bool)
            or not isinstance(row[name], (int, float))
            or not math.isfinite(float(row[name]))
            for name in required
        ):
            raise R253CompletionAuditError(
                "searched decision has non-numeric quality telemetry"
            )
        direct = float(row["direct_action_probability"])
        selected = float(row["mcts_action_direct_probability"])
        gap = float(row["direct_probability_gap"])
        value = float(row["selected_action_value"])
        rank = row["mcts_action_direct_rank"]
        selected_visits = row["selected_action_visits"]
        root_visits = row["root_visits"]
        legal_count = row["legal_action_count"]
        distinct = row["distinct_root_actions_visited"]
        rollouts = row["rollout_count"]
        if (
            not 0.0 <= direct <= 1.0
            or not 0.0 <= selected <= 1.0
            or abs((direct - selected) - gap) > 1e-6
            or not -1.0 <= value <= 1.0
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in (
                    rank,
                    selected_visits,
                    root_visits,
                    legal_count,
                    distinct,
                    rollouts,
                )
            )
            or not 1 <= rank <= legal_count
            or not 1 <= selected_visits <= root_visits
            or not 1 <= distinct <= min(legal_count, root_visits)
            or rollouts != root_visits
        ):
            raise R253CompletionAuditError(
                "searched decision has inconsistent quality telemetry"
            )

    def group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            name: _distribution(float(row[name]) for row in rows)
            for name in required
        }
        result["selected_visit_fraction"] = _distribution(
            float(row["selected_action_visits"]) / float(row["root_visits"])
            for row in rows
        )
        result["direct_confidence_threshold_count"] = {
            "gte_0_50": sum(float(row["direct_action_probability"]) >= 0.50 for row in rows),
            "gte_0_80": sum(float(row["direct_action_probability"]) >= 0.80 for row in rows),
            "gte_0_90": sum(float(row["direct_action_probability"]) >= 0.90 for row in rows),
            "gte_0_95": sum(float(row["direct_action_probability"]) >= 0.95 for row in rows),
            "gte_0_99": sum(float(row["direct_action_probability"]) >= 0.99 for row in rows),
        }
        result["selected_negative_value_count"] = sum(
            float(row["selected_action_value"]) < 0.0 for row in rows
        )
        return result

    changed = [row for row in decisions if row.get("action_changed") is True]
    meaningful = [
        row for row in decisions if row.get("meaningful_choice_change") is True
    ]
    if any(row.get("action_changed") is not True for row in meaningful):
        raise R253CompletionAuditError("meaningful change lacks a raw action change")
    return {
        "searched_decisions": len(decisions),
        "changed_decisions": len(changed),
        "meaningful_changed_decisions": len(meaningful),
        "changed_rate": len(changed) / max(1, len(decisions)),
        "meaningful_changed_rate": len(meaningful) / max(1, len(decisions)),
        "changed": group(changed),
        "unchanged": group(
            [row for row in decisions if row.get("action_changed") is not True]
        ),
    }


def build_completion_audit(run_root: Path) -> dict[str, Any]:
    """Re-open and bind a dispatcher-complete r253 result directory."""

    root = Path(run_root).resolve()
    identity_path = root / "run-identity.json"
    final_path = root / "final-review.json"
    identity = _read_object(identity_path)
    final = _read_object(final_path)
    if final.get("status") != "complete":
        raise R253CompletionAuditError("dispatcher final review is not complete")
    if identity.get("serial_rollout_revision") != 253:
        raise R253CompletionAuditError("run identity is not revision 253")

    paths = sorted((root / "games").glob("*.json"))
    if len(paths) != EXPECTED_GAMES:
        raise R253CompletionAuditError(
            f"expected {EXPECTED_GAMES} game receipts, got {len(paths)}"
        )
    games = [_read_object(path) for path in paths]
    try:
        regenerated = summarize_games(games, require_complete=True)
    except R229MetricsError as exc:
        raise R253CompletionAuditError(str(exc)) from exc

    recorded = final.get("summary")
    if not isinstance(recorded, Mapping):
        raise R253CompletionAuditError("dispatcher final review lacks a summary")
    recorded_base = json.loads(json.dumps(recorded))
    throughput = recorded_base.get("throughput")
    if isinstance(throughput, dict):
        for name in (
            "fleet_wall_seconds_this_invocation",
            "fleet_games_per_second_this_invocation",
            "fleet_games_per_hour_this_invocation",
        ):
            throughput.pop(name, None)
    if recorded_base != regenerated:
        raise R253CompletionAuditError(
            "dispatcher summary does not reproduce from the game receipts"
        )

    game_receipts = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in paths
    ]
    events_path = root / "events.jsonl"
    events: list[dict[str, Any]] = []
    if events_path.is_file():
        for number, line in enumerate(events_path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise R253CompletionAuditError(
                    f"malformed event JSON on line {number}"
                ) from exc
            if not isinstance(event, dict):
                raise R253CompletionAuditError(
                    f"event line {number} is not an object"
                )
            events.append(event)
    event_counts = Counter(str(event.get("disposition", "missing")) for event in events)
    event_host_counts = Counter(
        f"{event.get('host', 'missing')}:{event.get('disposition', 'missing')}"
        for event in events
    )
    quality = summarize_decision_quality(games)
    if (
        quality["searched_decisions"] != regenerated["decisions"]["searched_total"]
        or quality["changed_decisions"]
        != regenerated["decisions"]["action_changed_total"]
        or quality["meaningful_changed_decisions"]
        != regenerated["decisions"]["meaningful_choice_change_total"]
    ):
        raise R253CompletionAuditError(
            "decision-quality aggregation disagrees with validated game counters"
        )

    return {
        "schema": SCHEMA,
        "status": "complete_validated",
        "training_eligible": False,
        "run_root": str(root),
        "run_identity_sha256": _sha(identity_path),
        "dispatcher_final_review_sha256": _sha(final_path),
        "game_receipt_count": len(game_receipts),
        "game_receipts_rollup_sha256": _canonical_sha(game_receipts),
        "game_receipts": game_receipts,
        "events_sha256": _sha(events_path) if events_path.is_file() else None,
        "event_counts": dict(sorted(event_counts.items())),
        "event_counts_by_host": dict(sorted(event_host_counts.items())),
        "validated_summary": regenerated,
        "decision_quality": quality,
    }


def render_markdown(audit: Mapping[str, Any], *, audit_sha256: str) -> str:
    summary = audit["validated_summary"]
    decisions = summary["decisions"]
    search = summary["search"]
    quality = audit["decision_quality"]
    outcomes = summary["outcomes"]
    throughput = summary["throughput"]
    lines = [
        "# r253 restarting-serial MCTS vs r195 direct-policy BO1000",
        "",
        f"Completion audit: `{audit_sha256}`",
        f"Game receipt rollup: `{audit['game_receipts_rollup_sha256']}`",
        "",
        "## Outcome",
        "",
        f"- Games: {summary['games']} in {summary['pairs']} mirrored pairs",
        f"- MCTS wins: {outcomes.get('mcts_win', 0)}",
        f"- Direct-policy wins: {outcomes.get('direct_win', 0)}",
        f"- Draws: {outcomes.get('draw', 0)}",
        f"- MCTS decisive-game win rate: {summary['mcts_win_rate_excluding_draws']:.6f}",
        f"- Wilson 95% interval: {summary['mcts_win_rate_wilson_95']}",
        "",
        "## Decision influence",
        "",
        f"- Decisions seen: {decisions['seen_total']}",
        f"- Searched: {decisions['searched_total']}",
        f"- Meaningful changes: {decisions['meaningful_choice_change_total']} "
        f"({decisions['meaningful_change_rate_per_searched']:.6f})",
        f"- Changed-action direct confidence >=0.80: "
        f"{quality['changed']['direct_confidence_threshold_count']['gte_0_80']}",
        f"- Changed-action direct confidence >=0.90: "
        f"{quality['changed']['direct_confidence_threshold_count']['gte_0_90']}",
        "",
        "## Throughput",
        "",
        f"- Aggregate worker games/second: "
        f"{throughput['aggregate_worker_games_per_second']:.9f}",
        f"- Mean MCTS/direct decision latency ratio: "
        f"{search['mean_mcts_to_direct_decision_latency_ratio']:.3f}",
        f"- Search latency distribution: `{search['latency_seconds']}`",
        "",
        "This report is evaluation-only and training-ineligible.",
        "",
    ]
    return "\n".join(lines)


def create_once(path: Path, data: bytes) -> None:
    """Create and fsync an immutable artifact without replacing an old one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "R253CompletionAuditError",
    "build_completion_audit",
    "create_once",
    "render_markdown",
    "summarize_decision_quality",
]
