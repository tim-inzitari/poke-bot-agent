"""Fresh r221 wrapper for pre-random-boundary local BeliefMCTS evaluation.

R221 preserves r219's shared 45-second actual-turn pool and 15-second
meaningful-segment cap, but makes the chance fallback explicit: a native
BeliefMCTS result must attest that unresolved random events stopped at the
pre-random frozen-model leaf boundary.  It never grants action authority from
simulation count alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from .r215_full_turn_belief_mcts import R215ActionDecision, R215PlanResult, canonical_sha256
from .r219_multi_search_turn_belief_mcts import (
    R219FiniteChanceReceipt,
    R219MultiSearchTurnBeliefMCTS,
    R219PolicyTurnBridge,
    R219TimingConfig,
    commit_verified_cached_belief_action,
    r219_observation_from_raw,
    r219_plan_result_from_mcts_result,
)


R221_SCHEMA = "poke_bot.alakazam_local_multi_search_turn_belief_mcts_bo1000_r221/v1"
R221_DEFAULT_TURN_POOL_SECONDS = 45.0
R221_DEFAULT_SEARCH_SEGMENT_SECONDS = 15.0


class R221TimingConfig(R219TimingConfig):
    """Typed r221 identity; numerical timing is deliberately unchanged."""

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256({"schema": R221_SCHEMA, "timing": asdict(self)})


def _integer(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _r221_action_authority_receipt(
    diagnostics: Mapping[str, Any], selected_action: Sequence[int]
) -> tuple[bool, str]:
    """Require an explicit completed root-edge backup for every selected action."""

    receipt = diagnostics.get("selected_root_action_backup_receipt")
    fields: Mapping[str, Any]
    if isinstance(receipt, Mapping):
        fields = receipt
    else:
        fields = diagnostics
    selected = tuple(int(item) for item in selected_action)
    receipt_action = fields.get("selected_action")
    if receipt_action is not None:
        try:
            if tuple(int(item) for item in receipt_action) != selected:
                return False, "selected_root_action_receipt_action_mismatch"
        except (TypeError, ValueError):
            return False, "selected_root_action_receipt_action_malformed"
    if fields.get("selected_action_legal") is not True:
        return False, "selected_root_action_legal_receipt_missing"
    if fields.get("selected_action_fully_backed_up") is not True:
        return False, "selected_root_action_backup_receipt_missing"
    visits = _integer(fields.get("selected_action_visit_count"))
    completed = _integer(
        fields.get(
            "selected_action_completed_backups", fields.get("completed_backups")
        )
    )
    if visits < 1 or completed < 1:
        return False, "selected_root_action_has_no_completed_backup"
    for forbidden in (
        "private_random_outcome_samples",
        "guessed_random_rules_or_successors",
        "unobserved_random_outcome_advances",
    ):
        # A missing counter is not a zero counter.  The r221 wrapper must not
        # turn an old/native diagnostic shape into a private-random safety
        # claim by implication.
        if forbidden not in diagnostics:
            return False, f"r221_random_safety_telemetry_missing:{forbidden}"
        if _integer(diagnostics.get(forbidden)) != 0:
            return False, f"r221_forbidden_random_telemetry:{forbidden}"
    return True, "selected_root_action_explicitly_backed_up"


class R221MultiSearchTurnBeliefMCTS(R219MultiSearchTurnBeliefMCTS):
    """R219 timing/cache controller with r221 receipt identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("timing", R221TimingConfig())
        super().__init__(*args, **kwargs)

    def finish_actual_turn(self) -> Mapping[str, Any] | None:
        base = super().finish_actual_turn()
        if base is None:
            return None
        receipt = dict(base)
        receipt.update(
            {
                "schema": R221_SCHEMA,
                "base_controller_schema": base.get("schema"),
                "unforceable_randomness_mode": "pre_random_leaf_no_private_sampling",
            }
        )
        return receipt

    def act(self, *args: Any, **kwargs: Any) -> R215ActionDecision:
        base = super().act(*args, **kwargs)
        receipt = dict(base.receipt)
        receipt.update(
            {
                "schema": R221_SCHEMA,
                "base_controller_schema": base.receipt.get("schema"),
                "unforceable_randomness_mode": "pre_random_leaf_no_private_sampling",
            }
        )
        return R215ActionDecision(
            selected_action=base.selected_action, source=base.source, receipt=receipt
        )


class R221PolicyTurnBridge(R219PolicyTurnBridge):
    """R221 name for the same charged raw-observation/cache-commit bridge."""


def r221_observation_from_raw(
    raw_observation: Mapping[str, Any], policy: Any
) -> Any:
    """Build an r221 cache key while retaining the sealed r219 key format."""

    return r219_observation_from_raw(raw_observation, policy)


def r221_plan_result_from_mcts_result(
    result: Any,
    *,
    selected_action: Sequence[int] | None = None,
    extra_diagnostics: Mapping[str, Any] | None = None,
) -> R215PlanResult:
    """Convert a native result only when its selected root action is backed up."""

    plan = r219_plan_result_from_mcts_result(
        result, selected_action=selected_action, extra_diagnostics=extra_diagnostics
    )
    diagnostics = dict(plan.diagnostics)
    accepted, reason = _r221_action_authority_receipt(
        diagnostics, plan.selected_action
    )
    diagnostics["r221_selected_root_action_authority"] = accepted
    diagnostics["r221_selected_root_action_authority_reason"] = reason
    if not accepted:
        return R215PlanResult(
            selected_action=plan.selected_action,
            sims_run=0,
            continuation=(),
            diagnostics=diagnostics,
            root_action_stable=False,
            root_stability_receipt=None,
        )
    return R215PlanResult(
        selected_action=plan.selected_action,
        sims_run=plan.sims_run,
        continuation=plan.continuation,
        diagnostics=diagnostics,
        root_action_stable=plan.root_action_stable,
        root_stability_receipt=plan.root_stability_receipt,
    )


__all__ = [
    "R221_DEFAULT_SEARCH_SEGMENT_SECONDS",
    "R221_DEFAULT_TURN_POOL_SECONDS",
    "R221_SCHEMA",
    "R221MultiSearchTurnBeliefMCTS",
    "R221PolicyTurnBridge",
    "R221TimingConfig",
    "R219FiniteChanceReceipt",
    "commit_verified_cached_belief_action",
    "r219_observation_from_raw",
    "r221_observation_from_raw",
    "r221_plan_result_from_mcts_result",
]
