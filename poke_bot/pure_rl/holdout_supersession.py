"""Frozen-specialist rules that supersede redundant external holdout agents."""

from __future__ import annotations

import copy
from typing import Any, Mapping


POLICY_KEY = "external_premium_archetype_supersession"


def superseded_external_archetypes(
    registry: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return external archetypes retired by an already-frozen specialist."""

    policy = dict(registry.get("policy") or {})
    contract = dict(policy.get(POLICY_KEY) or {})
    if not contract or contract.get("enabled") is not True:
        return ()
    if (
        contract.get("scope") != "premium_holdout_external_opponents"
        or contract.get("preserve_historical_results") is not True
        or contract.get("keep_triggering_frozen_specialist") is not True
    ):
        raise ValueError("external premium holdout supersession policy is unsafe")
    rules = contract.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("external premium holdout supersession rules are missing")
    frozen_ids = {
        str(row.get("specialist_id") or "")
        for row in (registry.get("specialists") or [])
        if isinstance(row, dict) and row.get("frozen") is True
    }
    retired: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("external holdout supersession rule must be an object")
        trigger = str(rule.get("trigger_specialist_id") or "")
        archetype = str(rule.get("external_archetype_id") or "")
        if (
            not trigger
            or not archetype
            or rule.get("remove_external_opponents") is not True
        ):
            raise ValueError("external holdout supersession rule is incomplete")
        if trigger in frozen_ids:
            retired.add(archetype)
    return tuple(sorted(retired))


def apply_external_holdout_supersession(
    roster: list[dict[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove superseded external rows while retaining every frozen row."""

    archetypes = set(superseded_external_archetypes(registry))
    if not archetypes:
        return [copy.deepcopy(row) for row in roster], []
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row in roster:
        target = removed if (
            row.get("frozen_specialist") is not True
            and str(row.get("archetype_id") or "") in archetypes
        ) else kept
        target.append(copy.deepcopy(row))
    return kept, removed
