#!/usr/bin/env python3
"""Select the next executable specialist from canonical protocol state.

Selection preserves the configured priority order, but a target is executable
only when it is unfinished and its protected expert corpus satisfies the exact
minimum-decision contract. Insufficient corpora remain required targets and
are reported as deferred; they are never silently removed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


RESULT_SCHEMA = "poke_bot.next_specialist_selection/v1"
UNFINISHED = {
    "unstarted",
    "restart_required",
    "bootstrap_partial",
    "bootstrap_complete",
    "blocked",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root is not an object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def select(
    *,
    state_path: Path,
    corpus_root: Path,
    minimum_decisions: int,
    minimum_decisions_by_specialist: dict[str, int] | None = None,
    strict_priority_prefix: list[str] | None = None,
    completed_ids: set[str] | None = None,
    active_id: str | None = None,
    routable_ids: set[str] | None = None,
) -> dict[str, Any]:
    state_path = state_path.expanduser().resolve()
    corpus_root = corpus_root.expanduser().resolve()
    state = _load_yaml(state_path)
    rows = {
        str(row.get("id") or ""): dict(row)
        for row in (state.get("specialists") or [])
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    expected_total = int(
        ((state.get("target_registry") or {}).get("required_target_count") or 0)
    )
    order = list(
        ((state.get("training_priority") or {}).get(
            "ordered_unfinished_ids_after_active"
        ) or [])
    )
    progress = dict((state.get("current") or {}).get("program_progress") or {})
    unfinished_ids = {
        specialist_id
        for specialist_id, row in rows.items()
        if str(row.get("status") or "") in UNFINISHED
    }
    if (
        expected_total != 22
        or len(rows) != expected_total
        or int(progress.get("remaining_after_active", -1)) != len(order)
        or len(set(order)) != len(order)
        or not set(order).issubset(rows)
        or not unfinished_ids.issubset(set(order))
    ):
        raise RuntimeError(
            "canonical 22-specialist unfinished priority roster drifted"
        )
    minimum_overrides = {
        str(key): int(value)
        for key, value in (minimum_decisions_by_specialist or {}).items()
    }
    strict_prefix = [str(value) for value in (strict_priority_prefix or [])]
    if (
        any(value <= 0 for value in minimum_overrides.values())
        or not set(minimum_overrides).issubset(rows)
        or len(set(strict_prefix)) != len(strict_prefix)
        or not set(strict_prefix).issubset(rows)
    ):
        raise RuntimeError("specialist selection override contract changed")

    deferred: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    completed = {str(value) for value in (completed_ids or set())}
    active = str(active_id or "")
    routable = (
        {str(value) for value in routable_ids}
        if routable_ids is not None
        else None
    )
    for rank, specialist_id in enumerate(order):
        specialist_id = str(specialist_id)
        row = rows.get(specialist_id)
        if row is None:
            raise RuntimeError(f"priority target missing from state: {specialist_id}")
        if specialist_id in completed or specialist_id == active:
            continue
        if routable is not None and specialist_id not in routable:
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "validated_causal_runtime_route_missing",
                }
            )
            continue
        if str(row.get("status") or "") not in UNFINISHED:
            continue
        pointer = corpus_root / specialist_id / "PROTECTED_EXPERT_CORPUS.json"
        if not pointer.is_file():
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "protected_expert_corpus_missing",
                    "pointer": str(pointer),
                }
            )
            continue
        payload = _load_json(pointer)
        decisions = int((payload.get("totals") or {}).get("decisions_kept") or 0)
        required_decisions = minimum_overrides.get(
            specialist_id, int(minimum_decisions)
        )
        if (
            payload.get("schema") != "poke_bot.pinned_expert_corpus/v1"
            or payload.get("protected") is not True
            or decisions < required_decisions
        ):
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "protected_expert_corpus_below_contract",
                    "decisions": decisions,
                    "minimum_decisions": required_decisions,
                    "pointer": str(pointer),
                }
            )
            continue
        eligible.append(
            {
                "specialist_id": specialist_id,
                "priority_rank": rank,
                "decisions": decisions,
                "minimum_decisions": required_decisions,
                "pointer": str(pointer),
            }
        )

    selected = eligible[0] if eligible else None
    if selected is None:
        raise RuntimeError(
            "no unfinished specialist currently has a protocol-valid corpus"
        )
    strict_expected = next(
        (
            specialist_id
            for specialist_id in strict_prefix
            if specialist_id in order
            and specialist_id not in completed
            and specialist_id != active
            and str(rows[specialist_id].get("status") or "") in UNFINISHED
        ),
        None,
    )
    if (
        strict_expected is not None
        and str(selected["specialist_id"]) != strict_expected
    ):
        reason = next(
            (
                str(row["reason"])
                for row in deferred
                if row["specialist_id"] == strict_expected
            ),
            "not_eligible",
        )
        raise RuntimeError(
            f"strict priority specialist {strict_expected} is not executable: "
            f"{reason}"
        )
    return {
        "schema": RESULT_SCHEMA,
        "required_specialists_total": expected_total,
        "remaining_unfinished": len(order),
        # Retained for v1 receipt compatibility. The value is now the live
        # unfinished count rather than a permanent post-Starmie constant.
        "remaining_after_starmie": len(order),
        "minimum_decisions": int(minimum_decisions),
        "minimum_decisions_by_specialist": minimum_overrides,
        "strict_priority_prefix": strict_prefix,
        "completed_specialist_ids": sorted(completed),
        "active_specialist_id": active or None,
        "routable_specialist_ids": (
            sorted(routable) if routable is not None else None
        ),
        "selected": selected,
        "deferred_higher_priority": [
            row
            for row in deferred
            if int(row["priority_rank"]) < int(selected["priority_rank"])
        ],
        "eligible_in_priority_order": eligible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--minimum-decisions", type=int, default=20_000)
    parser.add_argument(
        "--minimum-decisions-override",
        action="append",
        default=[],
        metavar="SPECIALIST_ID=COUNT",
    )
    parser.add_argument(
        "--strict-priority-specialist",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--completed-specialist",
        action="append",
        default=[],
        help="Checksum-verified frozen specialist ID to exclude from selection.",
    )
    parser.add_argument("--active-specialist")
    parser.add_argument(
        "--runtime-tree",
        type=Path,
        help="Require selection to have a validated accepted causal route.",
    )
    args = parser.parse_args()
    if int(args.minimum_decisions) <= 0:
        raise ValueError("minimum decisions must be positive")
    minimum_overrides: dict[str, int] = {}
    for raw in args.minimum_decisions_override:
        specialist_id, separator, count = str(raw).partition("=")
        if not separator or not specialist_id or int(count) <= 0:
            raise ValueError(
                "minimum decision override must be SPECIALIST_ID=positive-count"
            )
        minimum_overrides[specialist_id] = int(count)
    routable_ids = None
    if args.runtime_tree is not None:
        tree = _load_json(args.runtime_tree.expanduser().resolve())
        routable_ids = {
            str(value)
            for value in (tree.get("runtime_contract") or {}).get(
                "accepted_archetype_ids", ()
            )
        }
    print(
        json.dumps(
            select(
                state_path=args.state,
                corpus_root=args.corpus_root,
                minimum_decisions=int(args.minimum_decisions),
                minimum_decisions_by_specialist=minimum_overrides,
                strict_priority_prefix=list(args.strict_priority_specialist),
                completed_ids=set(args.completed_specialist),
                active_id=args.active_specialist,
                routable_ids=routable_ids,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
