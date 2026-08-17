"""Typed, capability-masked archetype loss aggregation.

This module supplies no action route or policy head.  It only combines losses
that existing heads already produce, with explicit guide authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch


SCHEMA = "poke_bot.archetype_loss_contract/v1"
GROUPS = (
    "action_semantics",
    "short_horizon_tactics_opponent_response",
    "state_setup_resource_forecasting",
    "long_horizon_outcome_remaining_turns",
    "marnie_residual_objectives",
)
RESIDUALS = (
    "core_setup_continuity",
    "resource_attack_readiness",
    "long_horizon_prize_pressure",
)


class ArchetypeLossContractError(ValueError):
    pass


def validate_loss_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if data.get("schema") != SCHEMA:
        raise ArchetypeLossContractError("wrong loss contract schema")
    if tuple(data.get("optimizer_groups") or ()) != GROUPS:
        raise ArchetypeLossContractError("optimizer groups changed")
    residuals = data.get("residual_objectives") or {}
    if set(residuals) != set(RESIDUALS):
        raise ArchetypeLossContractError("residual objective set changed")
    for name in RESIDUALS:
        row = residuals[name]
        weight = float(row.get("weight", math.nan))
        if not math.isfinite(weight) or not 0.0 <= weight <= 0.05:
            raise ArchetypeLossContractError(f"invalid residual weight: {name}")
        if float(row.get("initial_weight", math.nan)) != 0.0125:
            raise ArchetypeLossContractError(f"invalid residual initial weight: {name}")
        if row.get("existing_head_targets") in (None, [], {}):
            raise ArchetypeLossContractError(f"residual lacks existing targets: {name}")
    if data.get("new_policy_heads") not in (False, 0) or data.get("new_routes") not in (False, 0):
        raise ArchetypeLossContractError("v1 may not add heads or routes")
    guides = data.get("guide_gradient_authorization")
    if not isinstance(guides, list):
        raise ArchetypeLossContractError("guide gradient authorization must be explicit")
    return data


def guide_gradient_allowed(contract: Mapping[str, Any], head_name: str) -> bool:
    """Availability is insufficient; the exact head must be allowlisted."""
    return str(head_name) in set(contract.get("guide_gradient_authorization") or ())


def canonical_residual_weights(
    values: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Return the complete bounded residual vector used by the learner."""
    raw = dict(values or {})
    unknown = set(raw) - set(RESIDUALS)
    if unknown:
        raise ArchetypeLossContractError(
            f"unknown archetype residual objectives: {sorted(unknown)}"
        )
    result = {name: 0.0 for name in RESIDUALS}
    for name, value in raw.items():
        weight = float(value)
        if not math.isfinite(weight) or not 0.0 <= weight <= 0.05:
            raise ArchetypeLossContractError(
                f"invalid residual objective weight: {name}"
            )
        result[name] = weight
    return result


@dataclass(frozen=True)
class MaskedObjective:
    name: str
    values: torch.Tensor
    row_mask: torch.Tensor
    weight: float
    capability: str
    row_applicable: torch.Tensor | None = None
    target_observable: torch.Tensor | None = None
    label_valid: torch.Tensor | None = None


def macro_list_loss(
    objectives: Sequence[MaskedObjective],
    *,
    variant_ids: Sequence[str],
    family_applicable: torch.Tensor,
    capabilities: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize active objectives/rows and then macro-average exact lists.

    A fully masked objective or list contributes exactly zero and creates no
    gradient. Factorized rows cannot dominate because each objective first
    reduces to one per-list scalar before objectives and lists are averaged.
    """
    if not objectives:
        raise ArchetypeLossContractError("no objectives supplied")
    n = len(variant_ids)
    if family_applicable.ndim != 1 or int(family_applicable.shape[0]) != n:
        raise ArchetypeLossContractError("family mask/variant length mismatch")
    device = objectives[0].values.device
    dtype = objectives[0].values.dtype
    per_list: list[torch.Tensor] = []
    for variant in sorted(set(str(value) for value in variant_ids)):
        list_mask = torch.tensor(
            [str(value) == variant for value in variant_ids],
            device=device,
            dtype=torch.bool,
        )
        active: list[torch.Tensor] = []
        for objective in objectives:
            values = objective.values.reshape(n, -1)
            row_valid = objective.row_mask.reshape(n, -1).to(torch.bool)
            for explicit in (
                objective.row_applicable,
                objective.target_observable,
                objective.label_valid,
            ):
                if explicit is not None:
                    row_valid = row_valid & explicit.reshape(n, -1).to(torch.bool)
            capability = capabilities.get(objective.capability)
            if capability is None:
                continue
            cap = capability.reshape(n, -1).to(torch.bool)
            applicable = family_applicable.reshape(n, 1).to(torch.bool)
            mask = row_valid & cap & applicable & list_mask.reshape(n, 1)
            count = mask.sum()
            if int(count.detach().cpu()) == 0:
                continue
            weight = float(objective.weight)
            if not math.isfinite(weight) or weight < 0:
                raise ArchetypeLossContractError("objective weight is invalid")
            active.append(values.masked_select(mask).mean() * weight)
        if active:
            per_list.append(torch.stack(active).mean())
    if not per_list:
        # Preserve a graph-connected exact zero for fully masked batches.
        return sum((obj.values.sum() * 0.0 for obj in objectives), torch.zeros((), device=device, dtype=dtype))
    return torch.stack(per_list).mean()


__all__ = [
    "ArchetypeLossContractError",
    "canonical_residual_weights",
    "GROUPS",
    "MaskedObjective",
    "RESIDUALS",
    "SCHEMA",
    "guide_gradient_allowed",
    "macro_list_loss",
    "validate_loss_contract",
]
