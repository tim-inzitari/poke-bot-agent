"""Validated curriculum schedule for the additive V6 strategic heads.

The numerical source of truth remains
``config/rl_protocol.yaml#/specialist_training/expanded_strategic_heads``.
This module deliberately accepts that mapping as input instead of embedding a
second copy of the weights.  Callers can therefore pin the exact canonical
mapping and its digest into every bootstrap checkpoint and receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


EXPANDED_SCHEDULE_SCHEMA = "poke_bot.expanded_head_schedule/v1"
EXPANDED_HEAD_IDS = (
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcome",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
)
_HEAD_ALIASES = {
    "tactical_outcomes": "tactical_outcome",
}


def canonical_head_id(value: object) -> str:
    """Return one canonical head id, accepting documented display aliases."""

    raw = str(value or "").strip()
    result = _HEAD_ALIASES.get(raw, raw)
    if result not in EXPANDED_HEAD_IDS:
        raise ValueError(f"unknown expanded strategic head: {raw!r}")
    return result


@dataclass(frozen=True)
class ExpandedHeadEpochPlan:
    """Effective loss contract for one exact supervised bootstrap epoch."""

    epoch: int
    stage_index: int
    enabled_heads: tuple[str, ...]
    loss_weights: dict[str, float]
    schedule_digest: str
    target_schema: str
    target_schema_digest: str
    runtime_enabled_heads: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPANDED_SCHEDULE_SCHEMA,
            "epoch": int(self.epoch),
            "stage_index": int(self.stage_index),
            "enabled_heads": list(self.enabled_heads),
            "loss_weights": dict(self.loss_weights),
            "schedule_digest": self.schedule_digest,
            "target_schema": self.target_schema,
            "target_schema_digest": self.target_schema_digest,
            # The first V6 rollout is training-only. Runtime activation requires
            # its own later validation receipt and is never implied by epochs.
            "runtime_enabled_heads": list(self.runtime_enabled_heads),
        }


def _canonical_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    if str(raw.get("schema") or "") != "poke_bot.expanded_strategic_heads/v1":
        raise ValueError("expanded strategic-head schema mismatch")
    head_rows = raw.get("heads")
    if not isinstance(head_rows, Mapping):
        raise ValueError("expanded strategic-head contract has no heads mapping")

    weights: dict[str, float] = {}
    for raw_name, raw_row in head_rows.items():
        name = canonical_head_id(raw_name)
        if name in weights:
            raise ValueError(f"duplicate expanded head after normalization: {name}")
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"expanded head {name} must be a mapping")
        weight = float(raw_row.get("weight", -1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"expanded head {name} has invalid weight")
        weights[name] = weight
    if set(weights) != set(EXPANDED_HEAD_IDS):
        missing = sorted(set(EXPANDED_HEAD_IDS) - set(weights))
        extra = sorted(set(weights) - set(EXPANDED_HEAD_IDS))
        raise ValueError(
            f"expanded head inventory mismatch: missing={missing} extra={extra}"
        )

    schedule = raw.get("bootstrap_stage_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("expanded head contract has no bootstrap schedule")
    total_epochs = int(schedule.get("total_epochs", -1))
    if total_epochs != 25:
        raise ValueError("expanded strategic-head bootstrap must be exactly 25 epochs")
    raw_stages = schedule.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("expanded strategic-head bootstrap stages are missing")

    stages: list[dict[str, Any]] = []
    covered: list[int] = []
    cumulative: list[str] = []
    for index, row in enumerate(raw_stages, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("expanded strategic-head stage must be a mapping")
        bounds = row.get("epochs")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or int(bounds[0]) > int(bounds[1])
        ):
            raise ValueError("expanded strategic-head stage has invalid epoch bounds")
        first, last = int(bounds[0]), int(bounds[1])
        covered.extend(range(first, last + 1))

        if row.get("enable_all") is True:
            cumulative = list(EXPANDED_HEAD_IDS)
        else:
            source = row.get("enable") if index == 1 else row.get("add")
            if not isinstance(source, list) or not source:
                raise ValueError(
                    "expanded strategic-head stage must enable or add named heads"
                )
            for raw_name in source:
                name = canonical_head_id(raw_name)
                if name not in cumulative:
                    cumulative.append(name)
        stages.append(
            {
                "index": index,
                "epochs": [first, last],
                "enabled_heads": [
                    name for name in EXPANDED_HEAD_IDS if name in cumulative
                ],
            }
        )

    if covered != list(range(1, total_epochs + 1)):
        raise ValueError(
            "expanded strategic-head stages must cover epochs 1..25 exactly once"
        )
    if set(stages[-1]["enabled_heads"]) != set(EXPANDED_HEAD_IDS):
        raise ValueError("final expanded strategic-head stage must enable all heads")

    return {
        "schema": EXPANDED_SCHEDULE_SCHEMA,
        "target_schema": str(raw.get("target_schema") or ""),
        "checkpoint_contract_schema": str(
            raw.get("checkpoint_contract_schema") or ""
        ),
        "total_epochs": total_epochs,
        "weights": {name: weights[name] for name in EXPANDED_HEAD_IDS},
        "stages": stages,
        "existing_enabled_heads_train_in_every_epoch": bool(
            schedule.get("existing_enabled_heads_train_in_every_epoch")
        ),
        "exact_epoch_count_may_not_be_shortened": bool(
            schedule.get("exact_epoch_count_may_not_be_shortened")
        ),
    }


def validated_expanded_head_schedule(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the deterministic machine/checkpoint projection."""

    contract = _canonical_contract(raw)
    if contract["target_schema"] != "poke_bot.expanded_strategic_targets/v2":
        raise ValueError("expanded strategic target schema mismatch")
    if contract["checkpoint_contract_schema"] != "poke_bot.expanded_head_training/v1":
        raise ValueError("expanded strategic checkpoint schema mismatch")
    if contract["existing_enabled_heads_train_in_every_epoch"] is not True:
        raise ValueError("expanded strategic schedule must be cumulative")
    if contract["exact_epoch_count_may_not_be_shortened"] is not True:
        raise ValueError("expanded strategic schedule must prohibit shortened bootstrap")
    return contract


def expanded_schedule_digest(raw: Mapping[str, Any]) -> str:
    """Digest the validated canonical projection, independent of YAML syntax."""

    contract = validated_expanded_head_schedule(raw)
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def expanded_head_epoch_plan(
    raw: Mapping[str, Any], epoch: int
) -> ExpandedHeadEpochPlan:
    """Resolve one epoch's cumulative nonzero loss weights."""

    contract = validated_expanded_head_schedule(raw)
    epoch = int(epoch)
    if not 1 <= epoch <= int(contract["total_epochs"]):
        raise ValueError(f"expanded strategic epoch outside 1..25: {epoch}")
    stage = next(
        row
        for row in contract["stages"]
        if int(row["epochs"][0]) <= epoch <= int(row["epochs"][1])
    )
    enabled = tuple(str(name) for name in stage["enabled_heads"])
    weights = {
        name: (
            float(contract["weights"][name])
            if name in enabled
            else 0.0
        )
        for name in EXPANDED_HEAD_IDS
    }
    # Imported lazily because strategic_heads owns the target schema digest and
    # imports this module only for the canonical ordered head ids.
    from .strategic_heads import TARGET_SCHEMA_DIGEST

    return ExpandedHeadEpochPlan(
        epoch=epoch,
        stage_index=int(stage["index"]),
        enabled_heads=enabled,
        loss_weights=weights,
        schedule_digest=expanded_schedule_digest(raw),
        target_schema=str(contract["target_schema"]),
        target_schema_digest=TARGET_SCHEMA_DIGEST,
    )


def all_expanded_heads_live_epoch_plan(
    raw: Mapping[str, Any], epoch: int
) -> ExpandedHeadEpochPlan:
    """Force every expanded head into the live nonzero loss schedule.

    Owner r175 Alakazam CE/rebootstrap must not stage later heads at weight 0.
    Preserves the protocol schedule digest; only the per-epoch enable mask and
    loss weights change so every architecture-present head backprops.
    """

    contract = validated_expanded_head_schedule(raw)
    epoch = int(epoch)
    if not 1 <= epoch <= int(contract["total_epochs"]):
        raise ValueError(f"expanded strategic epoch outside 1..25: {epoch}")
    weights = {
        name: float(contract["weights"][name]) for name in EXPANDED_HEAD_IDS
    }
    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError(
            "all-heads-live requires nonzero protocol expanded-head weights"
        )
    from .strategic_heads import TARGET_SCHEMA_DIGEST

    return ExpandedHeadEpochPlan(
        epoch=epoch,
        stage_index=int(contract["stages"][-1]["index"]),
        enabled_heads=tuple(EXPANDED_HEAD_IDS),
        loss_weights=weights,
        schedule_digest=expanded_schedule_digest(raw),
        target_schema=str(contract["target_schema"]),
        target_schema_digest=TARGET_SCHEMA_DIGEST,
    )
