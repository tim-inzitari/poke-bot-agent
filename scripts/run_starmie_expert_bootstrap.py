#!/usr/bin/env python3
"""Build and protect one temporal1 specialist from the distilled core.

The historical command-line defaults remain Starmie-compatible.  New
specialists pass ``--archetype`` and ``--expert-corpus``.  The protocol's 25
supervised epochs are exact: validation still selects the frozen checkpoint,
but diagnostic patience never shortens the bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import archetypes, checkpoint, device as device_mod
from poke_bot.archetype_loss_contract import canonical_residual_weights
from poke_bot.guide_evidence_rebind import (
    validation_or_evidence_only_rebind,
)
from poke_bot.expert_pilot_importance import load_training_weights_for_corpus
from poke_bot.model import (
    COMBO_STATE_HEAD_NAME,
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
    DECISION_FUSION_V2_SCHEMA,
    SETUP_BOARD_OUTCOME_HEAD_NAME,
)
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
)
from poke_bot.strategic_schedule import (
    EXPANDED_HEAD_IDS,
    EXPANDED_SCHEDULE_SCHEMA,
    expanded_head_epoch_plan,
    expanded_schedule_digest,
    validated_expanded_head_schedule,
)
from poke_bot.pure_rl.expert_rehearsal import (
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import freeze_model, verify_frozen_model
from poke_bot.slowking_candidate_validation import (
    validate_candidate as validate_slowking_candidate,
    validate_final_receipt as validate_slowking_final_receipt,
    validate_pretraining_authorization as validate_slowking_pretraining,
)
from poke_bot.train import (
    GUIDE_TRAINING_MODE_LEGACY,
    GUIDE_TRAINING_MODE_STRATEGIC,
    GUIDE_TRAINING_MODE_DIRECTIONAL,
    belief_card_vocab_from_state,
    supervised_rehearsal_step,
)


FAMILY = "starmie_expert_bootstrap_from_distilled_core_v1"
TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)
SPECIALIST_AUX_EXPANSION_SCHEMA = (
    "poke_bot.specialist_aux_archetype_head_expansion/v1"
)
EXPANDED_HEAD_MIGRATION_SCHEMA = "poke_bot.expanded_head_migration/v1"
EXPANDED_HEAD_TRAINING_SCHEMA = "poke_bot.expanded_head_training/v1"
DEFAULT_RL_PROTOCOL = ROOT / "config/rl_protocol.yaml"


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _tensor_bytes_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Compare serialized tensor payload bytes, including identical NaN bits."""

    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    left = first.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right = second.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left, right)


def _named_tensor_digest(
    state: dict[str, Any],
    names: list[str],
) -> str:
    """Digest exact named tensor identities and bytes in stable order."""

    digest = hashlib.sha256()
    for name in names:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"missing tensor required for digest: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def load_expanded_head_contract(
    protocol_path: Path = DEFAULT_RL_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the one canonical V6 head/schedule contract."""

    protocol_path = Path(protocol_path).expanduser().resolve()
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("RL protocol must be a YAML object")
    raw = (
        (payload.get("specialist_training") or {}).get(
            "expanded_strategic_heads"
        )
    )
    if not isinstance(raw, dict):
        raise RuntimeError("canonical expanded strategic-head contract is absent")
    schedule = validated_expanded_head_schedule(raw)
    return dict(raw), {
        "canonical_config": str(protocol_path),
        "canonical_config_sha256": checkpoint.checkpoint_digest(protocol_path),
        "source_digest": canonical_json_digest(raw),
        "architecture_schema": str(raw["schema"]),
        "target_schema": str(raw["target_schema"]),
        "target_schema_digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "checkpoint_contract_schema": str(raw["checkpoint_contract_schema"]),
        "schedule_schema": EXPANDED_SCHEDULE_SCHEMA,
        "schedule_digest": expanded_schedule_digest(raw),
        "schedule": schedule,
    }


def expanded_handoff_training_contract(
    protocol_path: Path = DEFAULT_RL_PROTOCOL,
) -> dict[str, Any]:
    """Return the immutable projection carried by generated handoffs."""

    _raw, identity = load_expanded_head_contract(protocol_path)
    schedule = dict(identity["schedule"])
    return {
        "schema": EXPANDED_HEAD_TRAINING_SCHEMA,
        "architecture_schema": identity["architecture_schema"],
        "target_schema": identity["target_schema"],
        "target_schema_digest": identity["target_schema_digest"],
        "canonical_config": (
            "config/rl_protocol.yaml#/"
            "specialist_training/expanded_strategic_heads"
        ),
        "canonical_config_sha256": identity["canonical_config_sha256"],
        "source_digest": identity["source_digest"],
        "schedule_schema": identity["schedule_schema"],
        "schedule_digest": identity["schedule_digest"],
        "schedule": schedule,
        "activation_boundary": (
            "next_safe_v6_cumulative_core_and_specialist_handoff"
        ),
        "current_live_v5_must_remain_unchanged": True,
        "initial_runtime_mode": "shadow_only",
        "runtime_enabled_heads": [],
        "flat_policy_remains_authoritative": True,
        "exact_bootstrap_epochs": int(schedule["total_epochs"]),
        "schedule_is_cumulative": True,
        "checkpoint_and_receipt_metadata_required": True,
    }


def decision_fusion_handoff_contract(
    protocol_path: Path = DEFAULT_RL_PROTOCOL,
    *,
    strategic_curriculum: bool = False,
    combo_state_head: bool = False,
) -> dict[str, Any]:
    """Project the canonical all-head action contract into each handoff."""

    protocol_path = Path(protocol_path).expanduser().resolve()
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    raw = (
        (payload.get("specialist_training") or {}).get("decision_fusion")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(raw, dict):
        raise RuntimeError("canonical causal decision-fusion contract is absent")
    required = tuple(str(value) for value in raw.get("required_causal_head_inputs") or ())
    if (
        raw.get("schema") != DECISION_FUSION_SCHEMA
        or raw.get("required") is not True
        or required != DECISION_FUSION_REQUIRED_HEADS
        or raw.get("applies_to_every_successor_specialist") is not True
    ):
        raise RuntimeError("canonical causal decision-fusion contract changed")
    if combo_state_head and not strategic_curriculum:
        raise ValueError("combo-state head requires strategic curriculum")
    if strategic_curriculum:
        guide = (
            (payload.get("specialist_training") or {}).get(
                "current_deck_guide"
            )
            or {}
        )
        branch = (
            ((guide.get("training_target_modes") or {}).get(
                "future_specialists"
            )
            or {}).get("strategic_branch_scope")
            or {}
        )
        required = (*required, "setup_board_outcome")
        if combo_state_head:
            required = (*required, "combo_state")
        if (
            branch.get("owner_decision_revision") != 56
            or branch.get("decision_fusion_schema")
            != "option_conditioned_per_head/v2"
            or branch.get("action_route_granularity")
            != "one_distinct_route_per_learned_decision_head"
            or branch.get("route_aggregation") != "fixed_mean"
            or float(branch.get("aggregate_route_delta_logit_cap") or -1.0)
            != 1.0
            or branch.get("route_final_projection_initialization")
            != "exact_zero"
            or branch.get("guide_is_only_action_route_exception") is not True
        ):
            raise RuntimeError(
                "canonical strategic decision-fusion contract changed"
            )
    return {
        "schema": (
            DECISION_FUSION_V2_SCHEMA
            if strategic_curriculum
            else DECISION_FUSION_SCHEMA
        ),
        "canonical_config": (
            "config/rl_protocol.yaml#/specialist_training/decision_fusion"
        ),
        "canonical_config_sha256": checkpoint.checkpoint_digest(protocol_path),
        "source_digest": canonical_json_digest(raw),
        "required_heads": list(required),
        "head_count": len(required),
        "training_mode": "joint_full_model",
        "runtime_required_after_bootstrap_validation": True,
        "matchup_adapter_behavior": "causal_route_gated",
        "absent_deck_guide_behavior": "exact_bypass",
        "future_specialists_required": True,
        **(
            {
                "owner_decision_revision": 56,
                "route_schema": "option_conditioned_per_head/v2",
                "one_route_per_learned_head": True,
                "route_input": "option_hidden_plus_typed_output",
                "route_aggregation": "fixed_mean",
                "aggregate_absolute_logit_cap": 1.0,
                "zero_safe_final_projection": True,
                "guide_is_only_action_route_exception": True,
            }
            if strategic_curriculum
            else {}
        ),
    }


def current_deck_guide_handoff_contract(
    *,
    specialist_id: str,
    contract_path: Path,
    expected_contract_sha256: str,
    guide_version: str,
    corpus_ready_receipt: Path,
    expected_corpus_ready_sha256: str,
    strategic_curriculum_spec: Path | None = None,
    expected_strategic_curriculum_spec_sha256: str = "",
    strategic_head_role_map: Path | None = None,
    expected_strategic_head_role_map_sha256: str = "",
    strategic_validation_receipt: Path | None = None,
    expected_strategic_validation_receipt_sha256: str = "",
    protocol_path: Path = DEFAULT_RL_PROTOCOL,
) -> dict[str, Any]:
    """Validate and project one successor's checksum-bound guide schedule."""

    specialist_id = str(specialist_id).strip().casefold()
    contract_path = Path(contract_path).expanduser().resolve()
    corpus_ready_receipt = Path(corpus_ready_receipt).expanduser().resolve()
    protocol_path = Path(protocol_path).expanduser().resolve()
    if not contract_path.is_file() or not corpus_ready_receipt.is_file():
        raise RuntimeError("current-deck guide artifacts are absent")
    contract_digest = checkpoint.checkpoint_digest(contract_path)
    ready_digest = checkpoint.checkpoint_digest(corpus_ready_receipt)
    if (
        contract_digest != str(expected_contract_sha256)
        or ready_digest != str(expected_corpus_ready_sha256)
    ):
        raise RuntimeError("current-deck guide checksum changed")
    guide = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    ready = json.loads(corpus_ready_receipt.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    canonical = (
        (protocol.get("specialist_training") or {}).get("current_deck_guide")
        if isinstance(protocol, dict)
        else None
    )
    bootstrap = dict((canonical or {}).get("bootstrap") or {})
    if (
        not isinstance(guide, dict)
        or not isinstance(canonical, dict)
        or guide.get("schema_version") != "poke_bot.current_deck_guide/v1"
        or guide.get("specialist_id") != specialist_id
        or guide.get("guide_version") != str(guide_version)
        or canonical.get("required_for_every_new_specialist") is not True
        or canonical.get("separate_parameterized_action_head") is not False
        or canonical.get("runtime_action_override_allowed") is not False
        or bootstrap.get("ramp_epochs") != [1, 5]
        or float(bootstrap.get("ramp_start_weight", -1.0)) != 0.01
        or float(bootstrap.get("ramp_end_weight", -1.0)) != 0.05
        or int(bootstrap.get("hold_through_epoch", -1)) != 25
        or float(canonical.get("maximum_loss_weight", -1.0)) != 0.05
        or ready.get("schema")
        != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != specialist_id
        or ready.get("guide_version") != str(guide_version)
        or int(ready.get("guide_rows") or 0) <= 0
    ):
        raise RuntimeError("current-deck guide handoff contract changed")
    policy_target = dict((guide or {}).get("policy_target") or {})
    if not policy_target:
        policy_target = dict(
            ((guide or {}).get("heuristic_research") or {}).get(
                "policy_target"
            )
            or {}
        )
    guide_mode = str(
        policy_target.get("training_mode") or GUIDE_TRAINING_MODE_LEGACY
    )
    if guide_mode not in {
        GUIDE_TRAINING_MODE_LEGACY,
        GUIDE_TRAINING_MODE_STRATEGIC,
        GUIDE_TRAINING_MODE_DIRECTIONAL,
    }:
        raise RuntimeError("current-deck guide training mode is invalid")
    supplied_strategic = any(
        (
            strategic_curriculum_spec is not None,
            bool(expected_strategic_curriculum_spec_sha256),
            strategic_head_role_map is not None,
            bool(expected_strategic_head_role_map_sha256),
            strategic_validation_receipt is not None,
            bool(expected_strategic_validation_receipt_sha256),
        )
    )
    strategic_bundle = None
    if guide_mode in {
        GUIDE_TRAINING_MODE_STRATEGIC,
        GUIDE_TRAINING_MODE_DIRECTIONAL,
    }:
        from scripts.register_next_specialist_runtime import (
            _validate_strategic_curriculum_bundle,
        )

        strategic_bundle = _validate_strategic_curriculum_bundle(
            specialist_id=specialist_id,
            guide_contract_sha256=contract_digest.removeprefix("sha256:"),
            training_mode=guide_mode,
            curriculum_spec=strategic_curriculum_spec,
            curriculum_spec_sha256=(
                expected_strategic_curriculum_spec_sha256
            ),
            head_role_map=strategic_head_role_map,
            head_role_map_sha256=expected_strategic_head_role_map_sha256,
            validation_receipt=strategic_validation_receipt,
            validation_receipt_sha256=(
                expected_strategic_validation_receipt_sha256
            ),
        )
    elif supplied_strategic:
        raise RuntimeError(
            "legacy current-deck guide cannot bind strategic artifacts"
        )
    return {
        "schema": "poke_bot.current_deck_guide_handoff/v1",
        "specialist_id": specialist_id,
        "guide_version": str(guide_version),
        "contract": str(contract_path),
        "contract_sha256": contract_digest,
        "corpus_ready_receipt": str(corpus_ready_receipt),
        "corpus_ready_receipt_sha256": ready_digest,
        "policy_target": (
            "observed_causal_strategic_heads_only"
            if guide_mode in {
                GUIDE_TRAINING_MODE_STRATEGIC,
                GUIDE_TRAINING_MODE_DIRECTIONAL,
            }
            else "shared_flat_policy"
        ),
        "training_mode": guide_mode,
        "runtime_action_override": False,
        "bootstrap_schedule": {
            "ramp_epochs": [1, 5],
            "ramp_start_weight": 0.01,
            "ramp_end_weight": 0.05,
            "hold_through_epoch": 25,
            "maximum_loss_weight": 0.05,
        },
        "runtime_initial_weight": float(canonical["initial_loss_weight"]),
        "adaptive_annealing_after_bootstrap": bool(
            (canonical.get("adaptive_annealing") or {}).get(
                "enabled_after_bootstrap"
            )
        ),
        **(
            {"strategic_curriculum": strategic_bundle}
            if strategic_bundle is not None
            else {}
        ),
    }


def current_deck_guide_epoch_weight(
    contract: dict[str, Any],
    epoch: int,
) -> float:
    """Return the exact canonical bootstrap weight for one epoch."""

    schedule = dict(contract.get("bootstrap_schedule") or {})
    start_epoch, end_epoch = [
        int(value) for value in schedule.get("ramp_epochs") or ()
    ]
    start_weight = float(schedule["ramp_start_weight"])
    end_weight = float(schedule["ramp_end_weight"])
    hold_through = int(schedule["hold_through_epoch"])
    epoch = int(epoch)
    if not 1 <= epoch <= hold_through:
        raise ValueError("current-deck guide epoch is outside bootstrap")
    if epoch >= end_epoch:
        return end_weight
    fraction = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return start_weight + fraction * (end_weight - start_weight)


def specialist_bootstrap_guide_weight(
    contract: dict[str, Any],
    epoch: int,
    *,
    active_through_epoch: int = 0,
) -> float:
    """Resolve guide pressure for an owner-scoped extended bootstrap.

    The ordinary specialist contract remains the exact 25-epoch guide
    schedule.  Crustle's post-Marnie refresh keeps guide pressure active for
    the entire 35-epoch bootstrap so the expert games stay grounded in its
    north-star strategy.
    """

    active_through_epoch = int(active_through_epoch)
    epoch = int(epoch)
    if active_through_epoch > 0 and epoch > active_through_epoch:
        return 0.0
    # The canonical guide schedule reaches its held weight by epoch 5 and is
    # defined through epoch 25.  Extended Crustle epochs remain at that same
    # held weight rather than being silently converted to guide-free training.
    return current_deck_guide_epoch_weight(
        contract, min(epoch, int((contract.get("bootstrap_schedule") or {}).get("hold_through_epoch", 25)))
    )


def validate_expanded_handoff_training_contract(
    value: Any,
    *,
    protocol_path: Path = DEFAULT_RL_PROTOCOL,
) -> dict[str, Any]:
    """Fail closed when a generated handoff drifts from canonical V6."""

    if not isinstance(value, dict):
        raise RuntimeError("expanded-head handoff contract is absent")
    expected = expanded_handoff_training_contract(protocol_path)
    for key in (
        "schema",
        "architecture_schema",
        "target_schema",
        "target_schema_digest",
        "source_digest",
        "schedule_schema",
        "schedule_digest",
        "schedule",
        "activation_boundary",
        "current_live_v5_must_remain_unchanged",
        "initial_runtime_mode",
        "runtime_enabled_heads",
        "flat_policy_remains_authoritative",
        "exact_bootstrap_epochs",
        "schedule_is_cumulative",
        "checkpoint_and_receipt_metadata_required",
    ):
        if value.get(key) != expected[key]:
            raise RuntimeError(
                f"expanded-head handoff contract changed at {key}"
            )
    return expected


def _manifest_expanded_targets(
    manifest_path: Path,
    *,
    decisions: int,
) -> dict[str, Any]:
    """Validate manifest-level target identity and per-head mask coverage."""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw = payload.get("expanded_strategic_targets")
    if not isinstance(raw, dict):
        raise RuntimeError(
            "expert corpus lacks expanded_strategic_targets metadata"
        )
    coverage = raw.get("head_coverage")
    if (
        raw.get("schema") != EXPANDED_STRATEGIC_SCHEMA
        or raw.get("digest") != EXPANDED_STRATEGIC_SCHEMA_DIGEST
        or int(raw.get("decisions", -1)) != int(decisions)
        or not isinstance(coverage, dict)
        or set(coverage) != set(EXPANDED_HEAD_IDS)
    ):
        raise RuntimeError("expert corpus expanded target identity changed")
    normalized: dict[str, dict[str, int]] = {}
    zero_labeled: list[str] = []
    for head_id in EXPANDED_HEAD_IDS:
        row = coverage.get(head_id)
        if not isinstance(row, dict):
            raise RuntimeError(
                f"expert corpus expanded target coverage is invalid: {head_id}"
            )
        try:
            labeled = int(row["labeled_rows"])
            masked = int(row["masked_rows"])
            total = int(row["total_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"expert corpus expanded target coverage is invalid: {head_id}"
            ) from exc
        if (
            labeled < 0
            or masked < 0
            or total != int(decisions)
            or labeled + masked != total
        ):
            raise RuntimeError(
                f"expert corpus expanded target coverage is invalid: {head_id}"
            )
        if labeled == 0:
            zero_labeled.append(head_id)
        normalized[head_id] = {
            "labeled_rows": labeled,
            "masked_rows": masked,
            "total_rows": total,
        }
    if zero_labeled:
        raise RuntimeError(
            "expert corpus has zero labeled rows for scheduled expanded heads: "
            + ", ".join(zero_labeled)
        )
    return {
        "schema": EXPANDED_STRATEGIC_SCHEMA,
        "digest": EXPANDED_STRATEGIC_SCHEMA_DIGEST,
        "decisions": int(decisions),
        "head_coverage": normalized,
    }


def _validate_metric_split(
    value: Any,
    *,
    enabled_heads: tuple[str, ...],
    split: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"expanded {split} metrics are absent")
    metrics = value.get("expanded_head_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"expanded {split} metrics are absent")
    labeled = metrics.get("labeled")
    total = metrics.get("total")
    losses = metrics.get("losses")
    if not all(isinstance(row, dict) for row in (labeled, total, losses)):
        raise RuntimeError(f"expanded {split} metrics are malformed")
    for head_id in enabled_heads:
        try:
            labeled_rows = int(labeled[head_id])
            total_rows = int(total[head_id])
            loss = float(losses[head_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"expanded {split} metrics are missing {head_id}"
            ) from exc
        if (
            labeled_rows <= 0
            or total_rows < labeled_rows
            or not math.isfinite(loss)
        ):
            raise RuntimeError(
                f"expanded {split} has no valid labeled rows for {head_id}"
            )
    return dict(metrics)


def validate_expanded_epoch_checkpoint(
    checkpoint_path: Path,
    *,
    plan: Any,
    identity: dict[str, Any],
    train_metrics: Any,
    validation_metrics: Any,
    epochs_total: int = 25,
    archetype_residual_loss_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Verify one epoch trained exactly the cumulative scheduled head set."""

    _validate_metric_split(
        train_metrics,
        enabled_heads=plan.enabled_heads,
        split="train",
    )
    _validate_metric_split(
        validation_metrics,
        enabled_heads=plan.enabled_heads,
        split="validation",
    )
    payload = checkpoint.load_checkpoint(checkpoint_path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    contract = extra.get("expanded_head_training")
    if not isinstance(contract, dict):
        raise RuntimeError(
            "expanded-head checkpoint lacks extra.expanded_head_training"
        )
    declared = {
        str(value)
        for value in contract.get("architecture_present_heads") or ()
    }
    trained = {
        str(value)
        for value in (
            contract.get("trained_this_epoch")
            or contract.get("trained_heads")
            or ()
        )
    }
    gradient = {
        str(value)
        for value in contract.get("gradient_enabled_heads") or ()
    }
    weights = {
        str(name): float(weight)
        for name, weight in dict(contract.get("loss_weights") or {}).items()
    }
    residuals = canonical_residual_weights(archetype_residual_loss_weights)
    recorded_residuals = canonical_residual_weights(
        contract.get("archetype_residual_loss_weights")
    )
    effective_weights = dict(plan.loss_weights)
    effective_weights["action_resource"] += residuals[
        "resource_attack_readiness"
    ]
    effective_weights["resource_forecast"] += residuals[
        "resource_attack_readiness"
    ]
    effective_weights["outcome_distribution"] += residuals[
        "long_horizon_prize_pressure"
    ]
    effective_weights["remaining_turns"] += residuals[
        "long_horizon_prize_pressure"
    ]
    recorded_effective_weights = {
        str(name): float(weight)
        for name, weight in dict(
            contract.get("effective_loss_weights") or weights
        ).items()
    }
    expected_gradient_heads = {
        name for name, weight in effective_weights.items() if float(weight) > 0.0
    }
    target_schema = contract.get(
        "target_schema", contract.get("target_schema_version")
    )
    schedule_schema = contract.get(
        "schedule_schema", contract.get("schedule_version")
    )
    if (
        contract.get("schema") != EXPANDED_HEAD_TRAINING_SCHEMA
        or target_schema != identity["target_schema"]
        or contract.get("target_schema_digest")
        != identity["target_schema_digest"]
        or schedule_schema != identity["schedule_schema"]
        or contract.get("schedule_digest") != identity["schedule_digest"]
        or int(contract.get("epoch", -1)) != int(plan.epoch)
        or int(contract.get("epochs_total", -1)) != int(epochs_total)
        or declared != set(EXPANDED_HEAD_IDS)
        or trained != expected_gradient_heads
        or gradient != expected_gradient_heads
        or contract.get("runtime_enabled_heads") != []
        or weights != dict(plan.loss_weights)
        or recorded_residuals != residuals
        or recorded_effective_weights != effective_weights
    ):
        raise RuntimeError(
            f"expanded-head epoch {plan.epoch} checkpoint contract changed"
        )
    return dict(contract)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _specialist_hot_start_from_core(
    core_path: Path,
    *,
    run_dir: Path,
    archetype: str,
    enable_expanded_heads: bool = False,
    expanded_identity: dict[str, Any] | None = None,
    enable_decision_fusion: bool = False,
    enable_strategic_curriculum: bool = False,
    enable_combo_state_head: bool = False,
    allow_h10_specialist_parent: bool = False,
) -> tuple[Path, str, dict[str, Any]]:
    """Build an append-only specialist hot start without rewriting the core.

    V6 is explicit. Its checkpoint advertises the new architecture while
    intentionally omitting new-head tensors; the audited loader then creates
    only those missing tensors under its fixed isolated RNG seed. Every source
    tensor remains byte-identical (or, for an older archetype classifier, its
    named rows are copied byte-identically into the append-expanded tensor).
    """

    if enable_decision_fusion and not enable_expanded_heads:
        raise ValueError("decision fusion requires expanded strategic heads")
    if enable_strategic_curriculum and not enable_decision_fusion:
        raise ValueError(
            "strategic curriculum requires causal decision fusion"
        )
    if enable_combo_state_head and not enable_strategic_curriculum:
        raise ValueError(
            "combo-state head requires strategic curriculum and decision fusion"
        )
    payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    source_state = dict(payload.get("model_state_dict") or {})
    state = dict(source_state)
    parent_digest = checkpoint.checkpoint_digest(core_path)
    model_config = dict(payload.get("model_config") or {})
    h10_specialist_parent = bool(
        allow_h10_specialist_parent
        and int(model_config.get("spatial_layers", -1)) == 7
        and int(model_config.get("temporal_layers", -1)) == 3
        and int(model_config.get("option_decoder_layers", -1)) == 7
        and int(model_config.get("ff_dim", -1)) == 2496
        and int(model_config.get("h10_head_residual_width", -1)) == 512
        and model_config.get("h10_capacity_enabled") is True
        and model_config.get(
            "decision_fusion_typed_output_centered_routes_enabled"
        )
        is True
        and str(payload.get("archetype_id") or "") not in {"", "unknown"}
    )
    if allow_h10_specialist_parent and not h10_specialist_parent:
        raise RuntimeError(
            "requested H10 specialist parent is not the exact H10/Fusion-v3 shape"
        )
    if h10_specialist_parent:
        route_keys = sorted(
            name
            for name in source_state
            if name.startswith("decision_fusion.dedicated_routes.")
        )
        route_names = sorted(
            {
                name.split(".")[2]
                for name in route_keys
                if len(name.split(".")) > 3
            }
        )
        if (
            not enable_expanded_heads
            or not enable_decision_fusion
            or not enable_strategic_curriculum
            or model_config.get("expanded_heads_enabled") is not True
            or model_config.get("decision_fusion_enabled") is not True
            or model_config.get("decision_fusion_runtime_enabled") is not True
            or model_config.get("decision_fusion_dedicated_routes_enabled")
            is not True
            or len(route_names) != 19
        ):
            raise RuntimeError(
                "H10 specialist parent lacks the complete 19-route strategic runtime"
            )
        expansion = {
            "schema": SPECIALIST_AUX_EXPANSION_SCHEMA,
            "status": "h10_parent_reused_zero_safe",
            "parent_checkpoint": str(core_path),
            "parent_checkpoint_digest": parent_digest,
            "source_archetype_ids": list(archetypes.archetype_ids()),
            "target_archetype_ids": list(archetypes.archetype_ids()),
            "source_classes": len(archetypes.archetype_ids()) + 1,
            "target_classes": len(archetypes.archetype_ids()) + 1,
            "copied_named_rows": list(archetypes.archetype_ids()),
            "newly_initialized_rows": [],
            "unknown_row_moved_to_final": False,
            "optimizer_state_imported": False,
            "all_parent_tensors_reused_byte_exact": True,
            "dedicated_fusion_route_names": route_names,
        }
        return core_path, parent_digest, {
            **expansion,
            "checkpoint_digest": parent_digest,
            "expanded_head_migration": {
                "status": "already_h10",
                "zero_safe_parent_reuse": True,
            },
            "decision_fusion_migration": {
                "status": "already_fusion_v3",
                "zero_safe_parent_reuse": True,
                "one_option_conditioned_route_per_learned_head": True,
                "route_names": route_names,
            },
        }
    weight_key = "aux_head.3.weight"
    bias_key = "aux_head.3.bias"
    old_weight = state.get(weight_key)
    old_bias = state.get(bias_key)
    fusion_archetype_key = (
        "decision_fusion.state_projections.archetype.weight"
    )
    fusion_archetype_expanded = False
    if not isinstance(old_weight, torch.Tensor) or not isinstance(
        old_bias, torch.Tensor
    ):
        raise RuntimeError("shared core lacks the canonical archetype head")

    target_ids = list(archetypes.archetype_ids())
    target_classes = len(target_ids) + 1
    old_classes = int(old_weight.shape[0])
    extra = dict(payload.get("extra") or {})
    recorded_expansion = dict(
        extra.get("specialist_aux_archetype_head_expansion") or {}
    )
    recorded_ids = recorded_expansion.get("target_archetype_ids")
    old_ids: list[str] | None = None
    if (
        isinstance(recorded_ids, list)
        and len(recorded_ids) + 1 == old_classes
    ):
        old_ids = [str(value) for value in recorded_ids]
        if len(old_ids) != len(set(old_ids)):
            raise RuntimeError(
                "shared core archetype row identity contains duplicates"
            )
    elif old_classes == target_classes:
        # Backward compatibility for a current-width checkpoint created before
        # explicit row identities were embedded. The stable registry contract
        # makes current width unambiguous only while the full order matches.
        old_ids = list(target_ids)
    else:
        compatible_orders = (
            archetypes.CUMULATIVE_V4_AUX_ARCHETYPE_IDS,
            archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS,
            archetypes.LEGACY_AUX_ARCHETYPE_IDS,
        )
        matches = [
            list(order)
            for order in compatible_orders
            if old_classes == len(order) + 1
        ]
        if len(matches) == 1:
            old_ids = matches[0]
    if old_ids is None:
        raise RuntimeError(
            "shared core archetype head cannot be append-expanded safely: "
            f"classes={old_classes}"
        )
    if any(name not in target_ids for name in old_ids):
        raise RuntimeError(
            "shared core archetype order is not append-compatible"
        )

    if old_classes == target_classes and old_ids == target_ids:
        expansion = {
            "schema": SPECIALIST_AUX_EXPANSION_SCHEMA,
            "status": "already_current",
            "parent_checkpoint": str(core_path),
            "parent_checkpoint_digest": parent_digest,
            "source_archetype_ids": old_ids,
            "target_archetype_ids": target_ids,
            "source_classes": old_classes,
            "target_classes": target_classes,
            "copied_named_rows": old_ids,
            "newly_initialized_rows": [],
            "unknown_row_moved_to_final": False,
            "optimizer_state_imported": False,
        }
        if not enable_expanded_heads:
            return (
                core_path,
                parent_digest,
                {
                    **expansion,
                    "checkpoint_digest": parent_digest,
                    "expanded_head_migration": None,
                },
            )
    else:
        seed_material = (
            f"{parent_digest}|{archetype}|{SPECIALIST_AUX_EXPANSION_SCHEMA}"
        ).encode("utf-8")
        expansion_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big"
        ) % (2**31)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(expansion_seed)
            expanded = nn.Linear(
                int(old_weight.shape[1]),
                target_classes,
                bias=True,
                device="cpu",
                dtype=old_weight.dtype,
            )
        with torch.no_grad():
            for old_index, name in enumerate(old_ids):
                new_index = target_ids.index(name)
                expanded.weight[new_index].copy_(old_weight[old_index])
                expanded.bias[new_index].copy_(old_bias[old_index])
            expanded.weight[-1].copy_(old_weight[-1])
            expanded.bias[-1].copy_(old_bias[-1])

        expansion = {
            "schema": SPECIALIST_AUX_EXPANSION_SCHEMA,
            "status": "expanded_append_only",
            "parent_checkpoint": str(core_path),
            "parent_checkpoint_digest": parent_digest,
            "source_archetype_ids": old_ids,
            "target_archetype_ids": target_ids,
            "source_classes": old_classes,
            "target_classes": target_classes,
            "copied_named_rows": old_ids,
            "newly_initialized_rows": [
                name for name in target_ids if name not in old_ids
            ],
            "unknown_row_moved_to_final": old_classes != target_classes,
            "expansion_seed": expansion_seed,
            "optimizer_state_imported": False,
        }
        state[weight_key] = expanded.weight.detach().clone()
        state[bias_key] = expanded.bias.detach().clone()
        fusion_archetype = state.get(fusion_archetype_key)
        if fusion_archetype is not None:
            if (
                not isinstance(fusion_archetype, torch.Tensor)
                or fusion_archetype.ndim != 2
                or int(fusion_archetype.shape[1]) != old_classes
            ):
                raise RuntimeError(
                    "shared core decision-fusion archetype projection "
                    "cannot be append-expanded safely"
                )
            expanded_fusion = fusion_archetype.new_zeros(
                (int(fusion_archetype.shape[0]), target_classes)
            )
            with torch.no_grad():
                for old_index, name in enumerate(old_ids):
                    new_index = target_ids.index(name)
                    expanded_fusion[:, new_index].copy_(
                        fusion_archetype[:, old_index]
                    )
                expanded_fusion[:, -1].copy_(fusion_archetype[:, -1])
            state[fusion_archetype_key] = expanded_fusion
            fusion_archetype_expanded = True
            expansion["decision_fusion_archetype_projection"] = {
                "tensor": fusion_archetype_key,
                "source_columns": old_classes,
                "target_columns": target_classes,
                "copied_named_columns": old_ids,
                "new_columns_zero_initialized": [
                    name for name in target_ids if name not in old_ids
                ],
                "unknown_column_moved_to_final": (
                    old_classes != target_classes
                ),
                "all_inherited_columns_byte_identical": True,
            }

    unchanged = []
    changed = []
    permitted_changed = {weight_key, bias_key}
    if fusion_archetype_expanded:
        permitted_changed.add(fusion_archetype_key)
    for name, source_tensor in source_state.items():
        target_tensor = state.get(name)
        if not isinstance(source_tensor, torch.Tensor) or not isinstance(
            target_tensor, torch.Tensor
        ):
            if source_tensor != target_tensor:
                raise RuntimeError(f"hot-start non-tensor state changed: {name}")
            unchanged.append(name)
            continue
        if (
            _tensor_bytes_equal(source_tensor, target_tensor)
        ):
            unchanged.append(name)
            continue
        if name not in permitted_changed or old_classes == target_classes:
            raise RuntimeError(f"hot-start inherited V5 tensor changed: {name}")
        changed.append(name)
    if set(changed) not in (set(), permitted_changed):
        raise RuntimeError("archetype append migration changed an incomplete pair")

    migration: dict[str, Any] | None = None
    fusion_migration: dict[str, Any] | None = None
    inherited_extra = dict(payload.get("extra") or {})
    if enable_expanded_heads:
        if expanded_identity is None:
            _raw, expanded_identity = load_expanded_head_contract()
        if (
            expanded_identity.get("architecture_schema")
            != "poke_bot.expanded_strategic_heads/v1"
            or expanded_identity.get("target_schema")
            != EXPANDED_STRATEGIC_SCHEMA
            or expanded_identity.get("target_schema_digest")
            != EXPANDED_STRATEGIC_SCHEMA_DIGEST
            or expanded_identity.get("schedule_schema")
            != EXPANDED_SCHEDULE_SCHEMA
            or not str(expanded_identity.get("schedule_digest") or "").startswith(
                "sha256:"
            )
        ):
            raise RuntimeError("expanded-head migration identity is invalid")
        model_config = dict(payload.get("model_config") or {})
        source_expanded_enabled = bool(
            model_config.get("expanded_heads_enabled", False)
        )
        source_expanded_tensor_keys = sorted(
            name
            for name in source_state
            if any(
                name.startswith(f"{head_id}_head.")
                for head_id in EXPANDED_HEAD_IDS
            )
        )
        expected_expanded_tensor_keys = {
            f"{head_id}_head.{suffix}"
            for head_id in EXPANDED_HEAD_IDS
            for suffix in ("weight", "bias")
        }
        if (
            bool(source_expanded_tensor_keys) != source_expanded_enabled
            or (
                source_expanded_tensor_keys
                and set(source_expanded_tensor_keys)
                != expected_expanded_tensor_keys
            )
        ):
            raise RuntimeError(
                "shared core has a partial/inconsistent expanded-head architecture"
            )
        inherited_training = inherited_extra.pop(
            "expanded_head_training", None
        )
        model_config["expanded_heads_enabled"] = True
        payload["model_config"] = model_config
        migration = {
            "schema": EXPANDED_HEAD_MIGRATION_SCHEMA,
            "status": "staged_shadow_only",
            "target_architecture_schema": expanded_identity[
                "architecture_schema"
            ],
            "target_schema": expanded_identity["target_schema"],
            "target_schema_digest": expanded_identity[
                "target_schema_digest"
            ],
            "runtime_enabled_heads": [],
            "source_checkpoint": str(core_path),
            "source_checkpoint_digest": parent_digest,
            "source_digest": expanded_identity["source_digest"],
            "schedule_schema": expanded_identity["schedule_schema"],
            "schedule_digest": expanded_identity["schedule_digest"],
            "inherited_v5_tensor_count": len(source_state),
            "whole_tensor_byte_identical_count": len(unchanged),
            "append_expanded_tensor_keys": sorted(changed),
            "all_inherited_v5_values_preserved": True,
            "source_expanded_heads_enabled": source_expanded_enabled,
            "source_expanded_tensor_count": len(
                source_expanded_tensor_keys
            ),
            "source_expanded_head_training_digest": (
                canonical_json_digest(inherited_training)
                if isinstance(inherited_training, dict)
                else None
            ),
            "specialist_training_metadata_reset": True,
            "new_head_tensors_materialized_by_audited_loader": (
                not source_expanded_tensor_keys
            ),
        }
    if enable_decision_fusion:
        model_config = dict(payload.get("model_config") or {})
        source_fusion_enabled = bool(
            model_config.get("decision_fusion_enabled", False)
        )
        source_fusion_tensor_keys = sorted(
            name
            for name in source_state
            if name.startswith("decision_fusion.")
        )
        if bool(source_fusion_tensor_keys) != source_fusion_enabled:
            raise RuntimeError(
                "shared core has a partial/inconsistent decision-fusion architecture"
            )
        source_fusion_base_keys = sorted(
            name
            for name in source_fusion_tensor_keys
            if not name.startswith("decision_fusion.dedicated_routes.")
        )
        source_fusion_route_keys = sorted(
            name
            for name in source_fusion_tensor_keys
            if name.startswith("decision_fusion.dedicated_routes.")
        )
        if source_fusion_tensor_keys and (
            source_fusion_route_keys
            or len(source_fusion_base_keys) != 30
        ):
            raise RuntimeError(
                "shared core is not a complete causal decision-fusion-v1 "
                "parent for additive fusion-v2 migration"
            )
        model_config["decision_fusion_enabled"] = True
        model_config["decision_fusion_runtime_enabled"] = True
        model_config.setdefault("decision_fusion_width", 16)
        if enable_strategic_curriculum:
            model_config["setup_board_outcome_head_enabled"] = True
            model_config["combo_state_head_enabled"] = bool(
                enable_combo_state_head
            )
            model_config[
                "decision_fusion_dedicated_routes_enabled"
            ] = True
            model_config[
                "decision_fusion_dedicated_routes_runtime_enabled"
            ] = True
        payload["model_config"] = model_config
        if not source_fusion_tensor_keys:
            fusion_migration = {
                "schema": "poke_bot.causal_decision_fusion_migration/v1",
                "target_schema": (
                    DECISION_FUSION_V2_SCHEMA
                    if enable_strategic_curriculum
                    else DECISION_FUSION_SCHEMA
                ),
                "source_checkpoint": str(core_path),
                "source_checkpoint_digest": parent_digest,
                "zero_safe_initialization": True,
                "runtime_enabled": True,
                "activation_scope": "isolated_specialist_bootstrap",
                "serving_eligible": False,
                "all_inherited_tensors_preserved": True,
                "one_option_conditioned_route_per_learned_head": (
                    enable_strategic_curriculum
                ),
                "new_auxiliary_head_names": (
                    [
                        SETUP_BOARD_OUTCOME_HEAD_NAME,
                        *(
                            [COMBO_STATE_HEAD_NAME]
                            if enable_combo_state_head
                            else []
                        ),
                    ]
                    if enable_strategic_curriculum
                    else []
                ),
            }
        elif enable_strategic_curriculum:
            fusion_migration = {
                "schema": "poke_bot.causal_decision_fusion_v2_migration/v1",
                "source_schema": DECISION_FUSION_SCHEMA,
                "target_schema": DECISION_FUSION_V2_SCHEMA,
                "source_checkpoint": str(core_path),
                "source_checkpoint_digest": parent_digest,
                "inherited_fusion_tensor_keys": source_fusion_base_keys,
                "inherited_fusion_tensor_count": len(
                    source_fusion_base_keys
                ),
                "inherited_fusion_tensor_digest": _named_tensor_digest(
                    source_state,
                    source_fusion_base_keys,
                ),
                "new_dedicated_route_names": sorted(
                    [
                        *DECISION_FUSION_REQUIRED_HEADS,
                        "setup_board_outcome",
                        *(
                            ["combo_state"]
                            if enable_combo_state_head
                            else []
                        ),
                    ]
                ),
                "new_auxiliary_head_names": [
                    SETUP_BOARD_OUTCOME_HEAD_NAME,
                    *(
                        [COMBO_STATE_HEAD_NAME]
                        if enable_combo_state_head
                        else []
                    ),
                ],
                "zero_safe_initialization": True,
                "runtime_enabled": True,
                "activation_scope": "isolated_specialist_bootstrap",
                "serving_eligible": False,
                "all_inherited_tensors_preserved": True,
                "one_option_conditioned_route_per_learned_head": True,
            }

    payload["model_state_dict"] = state
    payload["step"] = 0
    payload["epoch"] = 0
    payload["rl_iteration"] = 0
    payload["archetype_id"] = "unknown"
    payload["model_id"] = f"{archetype}.shared_core_hot_start"
    for key in (
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "early_stop_state",
    ):
        payload.pop(key, None)
    payload["extra"] = {
        **inherited_extra,
        "specialist_aux_archetype_head_expansion": expansion,
        **(
            {"expanded_head_migration": migration}
            if migration is not None
            else {}
        ),
        **(
            {"decision_fusion_migration": fusion_migration}
            if fusion_migration is not None
            else {}
        ),
    }
    effective_fusion_migration = payload["extra"].get(
        "decision_fusion_migration"
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    hot_start = run_dir / (
        (
            (
                "shared_core_hot_start.expanded-v6-fusion-v2-combo-v1.pt"
                if enable_combo_state_head
                else "shared_core_hot_start.expanded-v6-fusion-v2-v1.pt"
            )
            if enable_strategic_curriculum and source_fusion_tensor_keys
            else (
                "shared_core_hot_start.expanded-v6-fused-archetype-v2.pt"
                if fusion_archetype_expanded
                else "shared_core_hot_start.expanded-v6-fused-v1.pt"
            )
        )
        if enable_decision_fusion
        else (
            "shared_core_hot_start.expanded-v6.pt"
            if enable_expanded_heads
            else "shared_core_hot_start.current_archetypes.pt"
        )
    )
    if hot_start.is_file():
        existing = checkpoint.load_checkpoint(hot_start, map_location="cpu")
        existing_expansion = dict(
            (existing.get("extra") or {}).get(
                "specialist_aux_archetype_head_expansion"
            )
            or {}
        )
        existing_migration = (existing.get("extra") or {}).get(
            "expanded_head_migration"
        )
        existing_fusion_migration = (existing.get("extra") or {}).get(
            "decision_fusion_migration"
        )
        identity_differences = {
            "archetype_expansion": existing_expansion != expansion,
            "expanded_head_migration": existing_migration != migration,
            "decision_fusion_migration": (
                existing_fusion_migration != effective_fusion_migration
            ),
            "expanded_heads_enabled": dict(existing.get("model_config") or {}).get(
                "expanded_heads_enabled", False
            )
            is not bool(enable_expanded_heads),
            "decision_fusion_enabled": dict(existing.get("model_config") or {}).get(
                "decision_fusion_enabled", False
            )
            is not bool(enable_decision_fusion),
            "setup_board_outcome_head_enabled": dict(existing.get("model_config") or {}).get(
                "setup_board_outcome_head_enabled", False
            )
            is not bool(enable_strategic_curriculum),
            "combo_state_head_enabled": dict(existing.get("model_config") or {}).get(
                "combo_state_head_enabled", False
            )
            is not bool(enable_combo_state_head),
            "decision_fusion_dedicated_routes_enabled": dict(existing.get("model_config") or {}).get(
                "decision_fusion_dedicated_routes_enabled", False
            )
            is not bool(enable_strategic_curriculum),
        }
        changed = sorted(
            name for name, differs in identity_differences.items() if differs
        )
        if changed:
            detail = ""
            if "decision_fusion_migration" in changed:
                detail = (
                    f" existing_fusion={existing_fusion_migration!r}"
                    f" requested_fusion={effective_fusion_migration!r}"
                )
            raise RuntimeError(
                "existing specialist hot-start identity changed: "
                + ", ".join(changed)
                + detail
            )
    else:
        checkpoint.atomic_torch_save(payload, hot_start)
    hot_digest = checkpoint.checkpoint_digest(hot_start)
    return hot_start, hot_digest, {
        **expansion,
        "checkpoint_digest": hot_digest,
        "expanded_head_migration": migration,
        "decision_fusion_migration": effective_fusion_migration,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    corpus_group = parser.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--expert-corpus", type=Path)
    corpus_group.add_argument("--starmie-corpus", type=Path)
    parser.add_argument("--archetype", default="starmie")
    parser.add_argument("--family", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--core-family", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--owner-crustle-guide-active-epochs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--owner-crustle-guide-free-refresh-epochs",
        type=int,
        default=0,
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-decisions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--split-seed", type=int, default=20260722)
    parser.add_argument("--pilot-importance-index", type=Path)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--required-target", action="append", default=[])
    parser.add_argument("--expanded-heads", action="store_true")
    parser.add_argument("--decision-fusion", action="store_true")
    parser.add_argument(
        "--allow-h10-specialist-parent",
        action="store_true",
        help=(
            "Allow an immutable H10/Fusion-v3 specialist checkpoint as the "
            "zero-safe parent of a separately versioned successor."
        ),
    )
    parser.add_argument("--rl-protocol", type=Path, default=DEFAULT_RL_PROTOCOL)
    parser.add_argument("--expected-expanded-schedule-digest", default="")
    parser.add_argument("--expected-expanded-target-digest", default="")
    parser.add_argument("--current-deck-guide-contract", type=Path)
    parser.add_argument("--expected-current-deck-guide-sha256", default="")
    parser.add_argument("--current-deck-guide-version", default="")
    parser.add_argument("--current-deck-guide-corpus-ready", type=Path)
    parser.add_argument(
        "--expected-current-deck-guide-corpus-ready-sha256",
        default="",
    )
    parser.add_argument("--strategic-curriculum-spec", type=Path)
    parser.add_argument("--strategic-curriculum-spec-sha256", default="")
    parser.add_argument("--strategic-head-role-map", type=Path)
    parser.add_argument("--strategic-head-role-map-sha256", default="")
    parser.add_argument("--strategic-validation-receipt", type=Path)
    parser.add_argument("--strategic-validation-receipt-sha256", default="")
    parser.add_argument("--combo-state-head", action="store_true")
    parser.add_argument(
        "--retain-inherited-h10-combo-state-head",
        action="store_true",
        help=(
            "Retain an already-trained H10 combo-state head and its bounded "
            "fusion route without asserting Slowking-specific pretraining."
        ),
    )
    parser.add_argument("--combo-state-implementation-receipt", type=Path)
    parser.add_argument("--combo-state-implementation-receipt-sha256", default="")
    parser.add_argument("--combo-state-corpus-validation-receipt", type=Path)
    parser.add_argument(
        "--combo-state-corpus-validation-receipt-sha256", default=""
    )
    parser.add_argument("--combo-state-cpu-pack-validation-receipt", type=Path)
    parser.add_argument(
        "--combo-state-cpu-pack-validation-receipt-sha256", default=""
    )
    parser.add_argument("--combo-state-parameter-inventory-receipt", type=Path)
    parser.add_argument(
        "--combo-state-parameter-inventory-receipt-sha256", default=""
    )
    parser.add_argument("--combo-state-validation-output", type=Path)
    # Retained only to reject stale callers that try to authorize training
    # with a pre-existing post-training receipt.
    parser.add_argument("--combo-state-validation-receipt", type=Path)
    parser.add_argument("--combo-state-validation-receipt-sha256", default="")
    args = parser.parse_args(argv)
    required_targets = tuple(args.required_target or TARGETS)
    if (
        "temporal_action_rows" not in required_targets
        or len(set(required_targets)) != len(required_targets)
        or not set(required_targets).issubset(TARGETS)
    ):
        raise ValueError("invalid specialist target-coverage contract")
    all_auxiliary_heads_trained = set(required_targets) == set(TARGETS)
    archetype = str(args.archetype).strip().casefold()
    if not archetype or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in archetype):
        raise ValueError("specialist archetype must be a non-empty lowercase slug")
    crustle_extended_refresh = (
        archetype == "crustle"
        and int(args.owner_crustle_guide_active_epochs) == 35
        and int(args.owner_crustle_guide_free_refresh_epochs) == 0
        and int(args.epochs) == 35
    )
    if not crustle_extended_refresh and (
        int(args.epochs) != 25
        or int(args.owner_crustle_guide_active_epochs) != 0
        or int(args.owner_crustle_guide_free_refresh_epochs) != 0
    ):
        raise ValueError(
            "specialist bootstrap is locked to 25 epochs except for the "
            "owner-scoped Crustle all-guide 35-epoch bootstrap"
        )
    retained_h10_combo = bool(args.retain_inherited_h10_combo_state_head)
    combo_state_architecture = bool(args.combo_state_head or retained_h10_combo)
    if retained_h10_combo and (
        not args.allow_h10_specialist_parent
        or args.combo_state_head
        or not args.decision_fusion
    ):
        raise ValueError(
            "inherited H10 combo-state retention requires an H10 parent and "
            "decision fusion, and is distinct from Slowking pretraining"
        )
    expanded_raw: dict[str, Any] | None = None
    expanded_identity: dict[str, Any] | None = None
    if args.expanded_heads:
        expanded_raw, expanded_identity = load_expanded_head_contract(
            args.rl_protocol
        )
        if (
            str(args.expected_expanded_schedule_digest or "")
            != expanded_identity["schedule_digest"]
            or str(args.expected_expanded_target_digest or "")
            != expanded_identity["target_schema_digest"]
        ):
            raise ValueError(
                "generated handoff expanded-head digests do not match canon"
            )
    elif (
        args.expected_expanded_schedule_digest
        or args.expected_expanded_target_digest
    ):
        raise ValueError(
            "expanded-head digests require --expanded-heads"
        )
    if args.decision_fusion and not args.expanded_heads:
        raise ValueError("--decision-fusion requires --expanded-heads")
    guide_arguments = (
        args.current_deck_guide_contract,
        str(args.expected_current_deck_guide_sha256 or ""),
        str(args.current_deck_guide_version or ""),
        args.current_deck_guide_corpus_ready,
        str(args.expected_current_deck_guide_corpus_ready_sha256 or ""),
    )
    if any(value not in {None, ""} for value in guide_arguments) and not all(
        value not in {None, ""} for value in guide_arguments
    ):
        raise ValueError("current-deck guide arguments must be complete")
    strategic_arguments = (
        args.strategic_curriculum_spec,
        str(args.strategic_curriculum_spec_sha256 or ""),
        args.strategic_head_role_map,
        str(args.strategic_head_role_map_sha256 or ""),
        args.strategic_validation_receipt,
        str(args.strategic_validation_receipt_sha256 or ""),
    )
    if any(value not in {None, ""} for value in strategic_arguments) and not all(
        value not in {None, ""} for value in strategic_arguments
    ):
        raise ValueError("strategic curriculum arguments must be complete")
    guide_identity = (
        current_deck_guide_handoff_contract(
            specialist_id=archetype,
            contract_path=args.current_deck_guide_contract,
            expected_contract_sha256=args.expected_current_deck_guide_sha256,
            guide_version=args.current_deck_guide_version,
            corpus_ready_receipt=args.current_deck_guide_corpus_ready,
            expected_corpus_ready_sha256=(
                args.expected_current_deck_guide_corpus_ready_sha256
            ),
            strategic_curriculum_spec=args.strategic_curriculum_spec,
            expected_strategic_curriculum_spec_sha256=(
                args.strategic_curriculum_spec_sha256
            ),
            strategic_head_role_map=args.strategic_head_role_map,
            expected_strategic_head_role_map_sha256=(
                args.strategic_head_role_map_sha256
            ),
            strategic_validation_receipt=args.strategic_validation_receipt,
            expected_strategic_validation_receipt_sha256=(
                args.strategic_validation_receipt_sha256
            ),
            protocol_path=args.rl_protocol,
        )
        if all(value not in {None, ""} for value in guide_arguments)
        else None
    )
    if crustle_extended_refresh:
        if args.pilot_importance_index is None:
            raise ValueError(
                "Crustle extended refresh requires checksum-bound expert-pilot "
                "importance for all 35 epochs"
            )
        if guide_identity is None:
            raise ValueError("Crustle extended refresh requires its guide")
        guide_identity = {
            **guide_identity,
            "owner_epoch_schedule": {
                "schema": "poke_bot.crustle_guide_all_epochs/v1",
                "owner_decision_revision": 163,
                "guide_active_epochs": [1, 35],
                "guide_free_expert_refresh_epochs": [],
                "guide_free_expert_refresh_epoch_count": 0,
                "final_selection_eligible_epochs": [1, 35],
                "expert_pilot_importance_epochs": [1, 35],
            },
        }
    elif args.pilot_importance_index is not None:
        raise ValueError(
            "generic specialist bootstrap pilot weighting is currently scoped "
            "to the owner-authorized Crustle 35-epoch run"
        )
    strategic_curriculum = (
        guide_identity is not None
        and guide_identity.get("training_mode")
        in {GUIDE_TRAINING_MODE_STRATEGIC, GUIDE_TRAINING_MODE_DIRECTIONAL}
    )
    if strategic_curriculum and not args.decision_fusion:
        raise ValueError(
            "strategic current-deck guide requires --decision-fusion"
        )
    if bool(strategic_arguments[0]) != bool(strategic_curriculum):
        raise ValueError(
            "strategic artifacts must match the guide training mode"
        )
    stale_combo_receipt_arguments = (
        args.combo_state_validation_receipt,
        str(args.combo_state_validation_receipt_sha256 or ""),
    )
    if any(
        value not in {None, ""} for value in stale_combo_receipt_arguments
    ):
        raise ValueError(
            "a pre-existing final combo receipt cannot authorize candidate training"
        )
    combo_pretraining_arguments = {
        "implementation": (
            args.combo_state_implementation_receipt,
            str(args.combo_state_implementation_receipt_sha256 or ""),
        ),
        "corpus": (
            args.combo_state_corpus_validation_receipt,
            str(args.combo_state_corpus_validation_receipt_sha256 or ""),
        ),
        "cpu_pack": (
            args.combo_state_cpu_pack_validation_receipt,
            str(args.combo_state_cpu_pack_validation_receipt_sha256 or ""),
        ),
        "parameter_inventory": (
            args.combo_state_parameter_inventory_receipt,
            str(args.combo_state_parameter_inventory_receipt_sha256 or ""),
        ),
    }
    flattened_combo_pretraining = [
        value
        for pair in combo_pretraining_arguments.values()
        for value in pair
    ]
    if any(
        value not in {None, ""} for value in flattened_combo_pretraining
    ) and not all(
        value not in {None, ""} for value in flattened_combo_pretraining
    ):
        raise ValueError("combo-state pretraining receipt arguments must be complete")
    combo_receipt: dict[str, Any] | None = None
    combo_receipt_path: Path | None = None
    combo_receipt_digest: str | None = None
    combo_pretraining: dict[str, Any] | None = None
    if args.combo_state_head:
        if archetype != "slowking":
            raise ValueError("combo-state head is scoped only to slowking")
        if not strategic_curriculum:
            raise ValueError(
                "slowking combo-state head requires strategic curriculum"
            )
        if not all(
            value not in {None, ""} for value in flattened_combo_pretraining
        ):
            raise ValueError(
                "slowking combo-state head requires pretraining authorization"
            )
        combo_pretraining = validate_slowking_pretraining(
            {
                name: (path, digest)
                for name, (path, digest) in combo_pretraining_arguments.items()
                if path is not None
            }
        )
        combo_receipt_path = (
            args.combo_state_validation_output.expanduser().resolve()
            if args.combo_state_validation_output is not None
            else args.run_dir.expanduser().resolve()
            / "slowking_combo_head_validation.json"
        )
    elif archetype == "slowking":
        raise ValueError("slowking bootstrap requires --combo-state-head")
    elif any(
        value not in {None, ""} for value in flattened_combo_pretraining
    ) or args.combo_state_validation_output is not None:
        raise ValueError("combo-state artifacts are valid only for slowking")
    fusion_identity = (
        decision_fusion_handoff_contract(
            args.rl_protocol,
            strategic_curriculum=bool(strategic_curriculum),
            combo_state_head=combo_state_architecture,
        )
        if args.decision_fusion
        else None
    )
    corpus_argument = args.expert_corpus or args.starmie_corpus
    assert corpus_argument is not None
    family_name = str(args.family).strip() or (
        f"{archetype}_expert_bootstrap_from_distilled_core_v1"
    )
    display_name = str(args.display_name).strip() or (
        f"{archetype.replace('-', ' ').title()} Expert Bootstrap from Distilled Core"
    )
    state_schema = (
        "poke_bot.starmie_expert_bootstrap_state/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_expert_bootstrap_state/v1"
    )
    epoch_schema = (
        "poke_bot.starmie_bootstrap_epoch/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_bootstrap_epoch/v1"
    )
    ready_schema = (
        "poke_bot.starmie_expert_bootstrap_ready/v1"
        if archetype == "starmie"
        else "poke_bot.specialist_expert_bootstrap_ready/v1"
    )

    core = verify_frozen_model(args.core_family.expanduser().resolve())
    core_path = Path(str(core["model_path"])).resolve()
    core_payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    core_cfg = dict(core_payload.get("model_config") or {})
    ordinary_core = bool(
        int(core_cfg.get("temporal_layers", -1)) == 1
        and str(core_cfg.get("decision_context") or "") == "history"
        and int(core_cfg.get("max_context", -1)) == 320
        and str(core_payload.get("archetype_id") or "") == "unknown"
    )
    h10_specialist_parent = bool(
        args.allow_h10_specialist_parent
        and int(core_cfg.get("spatial_layers", -1)) == 7
        and int(core_cfg.get("temporal_layers", -1)) == 3
        and int(core_cfg.get("option_decoder_layers", -1)) == 7
        and int(core_cfg.get("ff_dim", -1)) == 2496
        and int(core_cfg.get("h10_head_residual_width", -1)) == 512
        and core_cfg.get("h10_capacity_enabled") is True
        and core_cfg.get("decision_fusion_typed_output_centered_routes_enabled")
        is True
        and str(core_payload.get("archetype_id") or "") not in {"", "unknown"}
    )
    if not (ordinary_core or h10_specialist_parent):
        raise ValueError(
            "bootstrap parent is not an accepted temporal core or exact H10 specialist"
        )
    corpus_pointer = corpus_argument.expanduser().resolve()
    manifest_requirements = {
        "min_decisions": int(args.min_decisions),
        "require_protected": True,
        "required_archetype": archetype,
        "required_compact_mode": "temporal-expert-v1",
        "required_max_context": 320,
        "required_target_coverage": required_targets,
        **(
            {
                "required_expanded_target_schema": expanded_identity[
                    "target_schema"
                ],
                "required_expanded_target_digest": expanded_identity[
                    "target_schema_digest"
                ],
                "required_expanded_heads": tuple(EXPANDED_HEAD_IDS),
            }
            if expanded_identity is not None
            else {}
        ),
    }
    identity = resolve_expert_manifest(
        corpus_pointer,
        **manifest_requirements,
    )
    if guide_identity is not None:
        guide_ready = json.loads(
            Path(guide_identity["corpus_ready_receipt"]).read_text(
                encoding="utf-8"
            )
        )
        manifest_payload = json.loads(
            Path(identity.path).read_text(encoding="utf-8")
        )
        guide_rows = int(
            ((manifest_payload.get("totals") or {}).get("target_coverage") or {}).get(
                "guide_rows"
            )
            or 0
        )
        if (
            guide_ready.get("manifest_sha256") != identity.digest
            or guide_ready.get("protected_pointer_sha256")
            != checkpoint.checkpoint_digest(corpus_pointer)
            or int(guide_ready.get("decisions") or 0) != int(identity.decisions)
            or int(guide_ready.get("guide_rows") or 0) != guide_rows
            or guide_rows <= 0
        ):
            raise RuntimeError(
                "current-deck guide corpus binding changed"
            )
        os.environ["POKEBOT_CURRENT_DECK_GUIDE"] = archetype
        os.environ["POKEBOT_CURRENT_DECK_GUIDE_TARGETS"] = "1"
    expanded_targets = (
        _manifest_expanded_targets(
            Path(identity.path),
            decisions=identity.decisions,
        )
        if expanded_identity is not None
        else None
    )
    pilot_importance_digest = (
        checkpoint.checkpoint_digest(
            args.pilot_importance_index.expanduser().resolve()
        )
        if args.pilot_importance_index is not None
        else None
    )
    family_dir = args.registry_root.expanduser().resolve() / family_name
    if args.ready.is_file() and family_dir.is_dir():
        ready = json.loads(args.ready.read_text(encoding="utf-8"))
        frozen = verify_frozen_model(family_dir)
        existing_combo_valid = not bool(args.combo_state_head)
        if args.combo_state_head:
            existing_combo = dict(
                ready.get("slowking_combo_state_validation") or {}
            )
            existing_combo_path = Path(
                str(existing_combo.get("receipt") or "")
            ).expanduser().resolve()
            if existing_combo_path.is_file():
                existing_combo_payload = json.loads(
                    existing_combo_path.read_text(encoding="utf-8")
                )
                validate_slowking_final_receipt(
                    existing_combo_payload,
                    checkpoint_path=Path(str(frozen["model_path"])).resolve(),
                )
                existing_combo_valid = (
                    checkpoint.checkpoint_digest(existing_combo_path)
                    == existing_combo.get("receipt_sha256")
                    and existing_combo.get("payload")
                    == existing_combo_payload
                )
        if (
            ready.get("status") == "ready"
            and ready.get("checkpoint_digest") == frozen.get("checkpoint_digest")
            and ready.get("core_checkpoint_digest") == core.get("checkpoint_digest")
            and ready.get("expert_manifest_sha256") == identity.digest
            and ready.get("acting_seat_archetype") == archetype
            and ready.get("current_deck_guide") == guide_identity
            and ready.get("expert_pilot_importance_index_sha256")
            == pilot_importance_digest
            and existing_combo_valid
            and (
                expanded_identity is None
                or (
                    ready.get("expanded_target_schema_digest")
                    == expanded_identity["target_schema_digest"]
                    and ready.get("expanded_schedule_digest")
                    == expanded_identity["schedule_digest"]
                )
            )
        ):
            print(json.dumps(ready, indent=2), flush=True)
            return 0
        raise RuntimeError("existing specialist readiness identity changed")

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hot_start, hot_start_digest, hot_start_expansion = (
        _specialist_hot_start_from_core(
            core_path,
            run_dir=run_dir,
            archetype=archetype,
            enable_expanded_heads=expanded_identity is not None,
            expanded_identity=expanded_identity,
            enable_decision_fusion=bool(args.decision_fusion),
            enable_strategic_curriculum=bool(strategic_curriculum),
            enable_combo_state_head=combo_state_architecture,
            allow_h10_specialist_parent=bool(
                args.allow_h10_specialist_parent
            ),
        )
    )
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if state:
        fixed_identity_changed = bool(
            state.get("core_checkpoint_digest")
            != core.get("checkpoint_digest")
            or state.get("manifest_digest") != identity.digest
            or state.get("hot_start_checkpoint_digest") != hot_start_digest
            or (
                args.pilot_importance_index is not None
                and state.get("expert_pilot_importance_index_sha256")
                != pilot_importance_digest
            )
            or (
                expanded_identity is not None
                and (
                    state.get("expanded_target_schema_digest")
                    != expanded_identity["target_schema_digest"]
                    or state.get("expanded_schedule_digest")
                    != expanded_identity["schedule_digest"]
                )
            )
        )
        saved_guide = dict(state.get("current_deck_guide") or {})
        guide_changed = saved_guide != guide_identity
        receipt_only_rebind = bool(
            guide_changed
            and validation_or_evidence_only_rebind(
                saved_guide,
                guide_identity,
                old_contract_snapshot=(
                    run_dir
                    / "guide-contract-before-evidence-rebind.yaml"
                ),
            )
        )
        if fixed_identity_changed or (
            guide_changed and not receipt_only_rebind
        ):
            raise RuntimeError("specialist bootstrap state identity changed")
        if receipt_only_rebind:
            state = {
                **state,
                "current_deck_guide": guide_identity,
                "validation_receipt_rebind": {
                    "schema":
                        "poke_bot.bootstrap_validation_receipt_rebind/v1",
                    "old_receipt_sha256": (
                        saved_guide.get("strategic_curriculum") or {}
                    ).get("validation_receipt_sha256"),
                    "new_receipt_sha256": (
                        guide_identity.get("strategic_curriculum") or {}
                    ).get("validation_receipt_sha256"),
                    "training_identity_unchanged": True,
                },
            }
            atomic_json(state_path, state)

    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    core_payload = checkpoint.load_checkpoint(core_path, map_location="cpu")
    belief_card_vocab = belief_card_vocab_from_state(
        dict(core_payload.get("model_state_dict") or {})
    )
    cache = ResidentExpertCorpusCache(cpu_pack_root=args.cpu_pack_root)
    print(
        f"[specialist-bootstrap:{archetype}] loading protected acting-seat corpus "
        f"records={identity.records} decisions={identity.decisions} device={device}",
        flush=True,
    )
    corpus = cache.prepare(
        identity,
        device=device,
        seed=int(args.split_seed),
        max_context=320,
        belief_card_vocab=belief_card_vocab,
    )
    if not corpus.has_exact_targets:
        raise RuntimeError(
            "specialist bootstrap corpus lost required all-head targets"
        )
    pilot_weights: torch.Tensor | None = None
    pilot_contract: dict[str, Any] | None = None
    if args.pilot_importance_index is not None:
        loaded_weights, pilot_contract = load_training_weights_for_corpus(
            args.pilot_importance_index.expanduser().resolve(),
            expected_manifest_digest=identity.digest,
            split_seed=int(args.split_seed),
            validation_fraction=0.10,
            max_context=320,
            train_games=corpus.train_games,
            validation_games=corpus.val_games,
        )
        pilot_weights = torch.tensor(
            loaded_weights, device=corpus.device, dtype=torch.float32
        )
        if (
            pilot_contract.get("actions_and_labels_unchanged") is not True
            or pilot_contract.get("validation_unweighted") is not True
            or int(pilot_contract.get("matched_top_100_train_games", 0)) <= 0
        ):
            raise RuntimeError("Crustle weighted expert contract is not usable")
    history = list(state.get("history") or [])
    best_metric = float(state.get("best_metric", math.inf))
    best_path = str(state.get("best_path") or "")
    best_digest = str(state.get("best_digest") or "")
    bad_epochs = int(state.get("bad_epochs", 0))
    parent = hot_start
    parent_digest = hot_start_digest
    start_epoch = 1
    if history:
        last = history[-1]
        parent = Path(str(last["checkpoint"])).resolve()
        parent_digest = str(last["checkpoint_digest"])
        if checkpoint.checkpoint_digest(parent) != parent_digest:
            raise RuntimeError("specialist resume parent digest mismatch")
        start_epoch = int(last["epoch"]) + 1
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            guide_weight = (
                specialist_bootstrap_guide_weight(
                    guide_identity,
                    epoch,
                    active_through_epoch=(
                        int(args.owner_crustle_guide_active_epochs)
                        if crustle_extended_refresh
                        else 0
                    ),
                )
                if guide_identity is not None
                else 0.0
            )
            plan = (
                expanded_head_epoch_plan(
                    expanded_raw,
                    min(epoch, 25) if crustle_extended_refresh else epoch,
                )
                if expanded_raw is not None
                else None
            )
            output = checkpoint_dir / f"epoch_{epoch:02d}.pt"
            if output.is_file():
                saved = checkpoint.load_checkpoint(output, map_location="cpu")
                extra = dict((saved.get("extra") or {}).get("specialist_bootstrap") or {})
                rehearsal = dict((saved.get("extra") or {}).get("expert_rehearsal") or {})
                if (
                    extra.get("schema") != epoch_schema
                    or int(extra.get("epoch", -1)) != epoch
                    or extra.get("core_checkpoint_digest") != core.get("checkpoint_digest")
                    or extra.get("parent_digest") != parent_digest
                    or extra.get("manifest_digest") != identity.digest
                    or extra.get("acting_seat_archetype") != archetype
                    or extra.get("current_deck_guide") != (
                        {
                            **guide_identity,
                            "epoch": epoch,
                            "loss_weight": guide_weight,
                        }
                        if guide_identity is not None
                        else None
                    )
                    or extra.get("expert_pilot_importance")
                    != rehearsal.get("expert_pilot_importance")
                    or (
                        pilot_importance_digest is not None
                        and dict(
                            rehearsal.get("expert_pilot_importance") or {}
                        ).get("importance_index_sha256")
                        != pilot_importance_digest
                    )
                    or (
                        plan is not None
                        and (
                            extra.get("expanded_target_schema_digest")
                            != expanded_identity["target_schema_digest"]
                            or extra.get("expanded_schedule_digest")
                            != expanded_identity["schedule_digest"]
                            or extra.get("expanded_epoch_plan")
                            != plan.as_dict()
                        )
                    )
                ):
                    raise RuntimeError(f"existing specialist epoch identity drift: {output}")
                result = {
                    "candidate_digest": checkpoint.checkpoint_digest(output),
                    "train_metrics": rehearsal.get("train_metrics"),
                    "validation_metrics": rehearsal.get("validation_metrics"),
                    "reused": True,
                }
            else:
                guide_training_active = bool(
                    guide_identity is not None and guide_weight > 0.0
                )
                specialist_bootstrap_extra = {
                    "schema": epoch_schema,
                    "epoch": epoch,
                    "acting_seat_archetype": archetype,
                    "core_checkpoint_digest": core["checkpoint_digest"],
                    "hot_start_checkpoint_digest": hot_start_digest,
                    "hot_start_expansion": hot_start_expansion,
                    "parent_digest": parent_digest,
                    "manifest_digest": identity.digest,
                    "all_auxiliary_heads_trained": (
                        all_auxiliary_heads_trained
                    ),
                    "current_deck_guide": (
                        {
                            **guide_identity,
                            "epoch": epoch,
                            "loss_weight": guide_weight,
                        }
                        if guide_identity is not None
                        else None
                    ),
                    "expert_pilot_importance": pilot_contract,
                    **(
                        {"decision_fusion": fusion_identity}
                        if fusion_identity is not None
                        else {}
                    ),
                    "trained_target_coverage": list(required_targets),
                    "inherited_target_coverage": [
                        target
                        for target in TARGETS
                        if target not in required_targets
                    ],
                    **(
                        {
                            "expanded_target_schema": expanded_identity[
                                "target_schema"
                            ],
                            "expanded_target_schema_digest": (
                                expanded_identity["target_schema_digest"]
                            ),
                            "expanded_schedule_schema": expanded_identity[
                                "schedule_schema"
                            ],
                            "expanded_schedule_digest": expanded_identity[
                                "schedule_digest"
                            ],
                            "expanded_epoch_plan": plan.as_dict(),
                            "expanded_manifest_targets": expanded_targets,
                            "runtime_enabled_heads": [],
                        }
                        if plan is not None
                        else {}
                    ),
                }
                rehearsal_kwargs = {
                    "base_ckpt": parent,
                    "output_path": output,
                    "parent_digest": parent_digest,
                    "rehearsal_iteration": epoch,
                    "manifest_identity": {
                        **identity.as_dict(),
                        **(
                            {
                                "expanded_strategic_targets": (
                                    expanded_targets
                                )
                            }
                            if expanded_targets is not None
                            else {}
                        ),
                    },
                    "epochs": 1,
                    "lr": 5e-5,
                    "requested_batch_size": int(args.batch_size),
                    "seed": 20260722 + epoch,
                    "corpus_split_seed": int(args.split_seed),
                    "device": device,
                    "aux_loss_weight": 0.05,
                    "opp_hand_loss_weight": 0.05,
                    "opp_remainder_loss_weight": 0.05,
                    "lethal_threat_loss_weight": 0.025,
                    "prize_race_loss_weight": 0.025,
                    "alakazam_guide_loss_weight": guide_weight,
                    "combo_state_loss_weight": (
                        0.025 if bool(args.combo_state_head) else 0.0
                    ),
                    "current_deck_guide_training_mode": (
                        str(guide_identity["training_mode"])
                        if guide_training_active
                        else GUIDE_TRAINING_MODE_LEGACY
                    ),
                    "current_deck_guide_curriculum_spec": (
                        str(
                            guide_identity["strategic_curriculum"][
                                "curriculum_spec"
                            ]
                        )
                        if strategic_curriculum and guide_training_active
                        else ""
                    ),
                    "current_deck_guide_head_role_map": (
                        str(
                            guide_identity["strategic_curriculum"][
                                "head_role_map"
                            ]
                        )
                        if strategic_curriculum and guide_training_active
                        else ""
                    ),
                    "current_deck_guide_curriculum_validation_receipt": (
                        str(
                            guide_identity["strategic_curriculum"][
                                "validation_receipt"
                            ]
                        )
                        if strategic_curriculum and guide_training_active
                        else ""
                    ),
                    "output_archetype_id": archetype,
                    "output_model_id": f"{args.run_name}.epoch{epoch:02d}",
                    "extra_updates": {
                        "specialist_bootstrap": specialist_bootstrap_extra,
                    },
                    "training_game_sampling_weights": pilot_weights,
                    "training_game_importance_contract": pilot_contract,
                }
                if plan is not None:
                    rehearsal_kwargs.update(
                        {
                            "expanded_head_loss_weights": (
                                dict(plan.loss_weights)
                            ),
                            "expanded_head_schedule": plan.as_dict(),
                        }
                    )
                result = supervised_rehearsal_step(
                    corpus,
                    **rehearsal_kwargs,
                )
            expanded_checkpoint_contract = (
                validate_expanded_epoch_checkpoint(
                    output,
                    plan=plan,
                    identity=expanded_identity,
                    train_metrics=result.get("train_metrics"),
                    validation_metrics=result.get("validation_metrics"),
                )
                if plan is not None
                else None
            )
            metric = float((result.get("validation_metrics") or {}).get("total_loss", math.inf))
            if not math.isfinite(metric):
                raise RuntimeError("specialist validation metric is not finite")
            if guide_identity is not None and guide_weight > 0.0 and (
                int(
                    (result.get("train_metrics") or {}).get(
                        "n_alakazam_guide_rows", 0
                    )
                )
                <= 0
                or int(
                    (result.get("validation_metrics") or {}).get(
                        "n_alakazam_guide_rows", 0
                    )
                )
                <= 0
            ):
                raise RuntimeError(
                    "current-deck guide received no labeled bootstrap rows"
                )
            if bool(args.combo_state_head):
                train_combo = dict(
                    (result.get("train_metrics") or {}).get(
                        "combo_state_metrics"
                    )
                    or {}
                )
                validation_combo = dict(
                    (result.get("validation_metrics") or {}).get(
                        "combo_state_metrics"
                    )
                    or {}
                )
                if (
                    int(train_combo.get("eligible_rows") or 0) <= 0
                    or int(validation_combo.get("eligible_rows") or 0) <= 0
                ):
                    raise RuntimeError(
                        "Slowking combo-state head received no causal labeled rows"
                    )
            row = {
                "epoch": epoch,
                "checkpoint": str(output),
                "checkpoint_digest": str(result["candidate_digest"]),
                "parent_digest": parent_digest,
                "validation_loss": metric,
                "validation_accuracy": float(
                    (result.get("validation_metrics") or {}).get("policy_acc", 0.0)
                ),
                "train_metrics": result.get("train_metrics"),
                "validation_metrics": result.get("validation_metrics"),
                "reused": bool(result.get("reused")),
                **(
                    {
                        "expanded_epoch_plan": plan.as_dict(),
                        "expanded_head_training": (
                            expanded_checkpoint_contract
                        ),
                    }
                    if plan is not None
                    else {}
                ),
            }
            history.append(row)
            selection_eligible = (
                (
                not crustle_extended_refresh
                or int(args.owner_crustle_guide_free_refresh_epochs) == 0
                or epoch > int(args.owner_crustle_guide_active_epochs)
                )
                and (
                    plan is None
                    or set(plan.enabled_heads) == set(EXPANDED_HEAD_IDS)
                )
            )
            if selection_eligible:
                if metric < best_metric - float(args.min_delta):
                    best_metric = metric
                    best_path = str(output)
                    best_digest = str(result["candidate_digest"])
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            parent = output
            parent_digest = str(result["candidate_digest"])
            state = {
                "schema": state_schema,
                "status": "training",
                "acting_seat_archetype": archetype,
                "core_checkpoint": str(core_path),
                "core_checkpoint_digest": core["checkpoint_digest"],
                "hot_start_checkpoint": str(hot_start),
                "hot_start_checkpoint_digest": hot_start_digest,
                "hot_start_expansion": hot_start_expansion,
                "manifest": identity.as_dict(),
                "manifest_digest": identity.digest,
                "current_deck_guide": guide_identity,
                "expert_pilot_importance": pilot_contract,
                "expert_pilot_importance_index_sha256": (
                    pilot_contract.get("importance_index_sha256")
                    if pilot_contract is not None
                    else None
                ),
                **(
                    {
                        "expanded_target_schema": expanded_identity[
                            "target_schema"
                        ],
                        "expanded_target_schema_digest": expanded_identity[
                            "target_schema_digest"
                        ],
                        "expanded_schedule_schema": expanded_identity[
                            "schedule_schema"
                        ],
                        "expanded_schedule_digest": expanded_identity[
                            "schedule_digest"
                        ],
                        "expanded_schedule": expanded_identity["schedule"],
                        "expanded_manifest_targets": expanded_targets,
                        "expanded_head_migration": (
                            hot_start_expansion.get(
                                "expanded_head_migration"
                            )
                        ),
                        "runtime_enabled_heads": [],
                    }
                    if expanded_identity is not None
                    else {}
                ),
                "history": history,
                "best_path": best_path,
                "best_digest": best_digest,
                "best_metric": best_metric,
                "bad_epochs": bad_epochs,
                "patience": int(args.patience),
                "epochs_max": int(args.epochs),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(state_path, state)
            print(
                f"[specialist-bootstrap:{archetype}] "
                f"epoch={epoch}/{int(args.epochs)} "
                f"val_loss={metric:.6f} "
                f"best={best_metric:.6f} patience={bad_epochs}/{int(args.patience)}",
                flush=True,
            )
    finally:
        cache.release()

    if [int(row.get("epoch", -1)) for row in history] != list(
        range(1, int(args.epochs) + 1)
    ):
        raise RuntimeError(
            "specialist bootstrap did not complete its exact epoch contract"
        )
    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("selected specialist bootstrap identity is invalid")
    best_payload = checkpoint.load_checkpoint(best, map_location="cpu")
    if str(best_payload.get("archetype_id") or "") != archetype:
        raise RuntimeError("selected specialist checkpoint lost archetype metadata")
    best_expanded_training = (
        dict((best_payload.get("extra") or {}).get("expanded_head_training") or {})
        if expanded_identity is not None
        else None
    )
    if expanded_identity is not None and (
        best_expanded_training.get("schema")
        != EXPANDED_HEAD_TRAINING_SCHEMA
        or set(best_expanded_training.get("trained_heads") or ())
        != set(EXPANDED_HEAD_IDS)
        or best_expanded_training.get("runtime_enabled_heads") != []
        or best_expanded_training.get("target_schema_digest")
        != expanded_identity["target_schema_digest"]
        or best_expanded_training.get("schedule_digest")
        != expanded_identity["schedule_digest"]
    ):
        raise RuntimeError(
            "selected specialist checkpoint has incomplete expanded-head training"
        )
    if fusion_identity is not None:
        best_config = dict(best_payload.get("model_config") or {})
        best_provenance = dict(best_payload.get("provenance") or {})
        fusion_inventory = dict(best_provenance.get("decision_fusion") or {})
        fusion_state = dict(best_payload.get("model_state_dict") or {})
        output_weight = fusion_state.get("decision_fusion.residual.2.weight")
        route_weights = {
            name: tensor
            for name, tensor in fusion_state.items()
            if name.startswith("decision_fusion.dedicated_routes.")
            and name.endswith(".network.2.weight")
        }
        expected_fusion_schema = (
            DECISION_FUSION_V2_SCHEMA
            if strategic_curriculum
            else DECISION_FUSION_SCHEMA
        )
        if (
            best_config.get("decision_fusion_enabled") is not True
            or best_config.get("decision_fusion_runtime_enabled") is not True
            or fusion_inventory.get("schema") != expected_fusion_schema
            or fusion_inventory.get("runtime_enabled") is not True
            or fusion_inventory.get("required_heads")
            != list(fusion_identity["required_heads"])
            or not isinstance(output_weight, torch.Tensor)
            or not bool(torch.count_nonzero(output_weight).item())
            or (
                strategic_curriculum
                and (
                    best_config.get(
                        "setup_board_outcome_head_enabled"
                    )
                    is not True
                    or best_config.get("combo_state_head_enabled", False)
                    is not combo_state_architecture
                    or best_config.get(
                        "decision_fusion_dedicated_routes_enabled"
                    )
                    is not True
                    or best_config.get(
                        "decision_fusion_dedicated_routes_runtime_enabled"
                    )
                    is not True
                    or set(route_weights)
                    != {
                        "decision_fusion.dedicated_routes."
                        f"{name}.network.2.weight"
                        for name in fusion_identity["required_heads"]
                    }
                    or any(
                        not isinstance(tensor, torch.Tensor)
                        or not bool(torch.count_nonzero(tensor).item())
                        for tensor in route_weights.values()
                    )
                )
            )
        ):
            raise RuntimeError(
                "selected specialist checkpoint lacks trained causal decision fusion"
            )
    if args.combo_state_head:
        if (
            combo_pretraining is None
            or combo_receipt_path is None
            or expanded_raw is None
            or guide_identity is None
        ):
            raise RuntimeError(
                "Slowking candidate validation inputs were not retained"
            )
        final_plan = expanded_head_epoch_plan(expanded_raw, int(args.epochs))
        combo_receipt = validate_slowking_candidate(
            checkpoint_path=best,
            parent_checkpoint=hot_start,
            corpus=corpus,
            device=device,
            pretraining=combo_pretraining,
            output=combo_receipt_path,
            expanded_head_weights=dict(final_plan.loss_weights),
            guide_loss_weight=specialist_bootstrap_guide_weight(
                guide_identity,
                int(args.epochs),
                active_through_epoch=(
                    int(args.owner_crustle_guide_active_epochs)
                    if crustle_extended_refresh
                    else 0
                ),
            ),
        )
        combo_receipt_digest = checkpoint.checkpoint_digest(
            combo_receipt_path
        )
    frozen = freeze_model(
        registry_root=args.registry_root,
        family=family_name,
        display_name=display_name,
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "initialized_from_family": core.get("family"),
            "initialized_from_digest": core["checkpoint_digest"],
            "specialist_hot_start": {
                "checkpoint": str(hot_start),
                "checkpoint_digest": hot_start_digest,
                "aux_archetype_head_expansion": hot_start_expansion,
            },
            "expert_manifest": identity.as_dict(),
            **(
                {"starmie_manifest": identity.as_dict()}
                if archetype == "starmie"
                else {}
            ),
            "acting_seat_archetype": archetype,
            "all_auxiliary_heads_trained": all_auxiliary_heads_trained,
            "trained_target_coverage": list(required_targets),
            "inherited_target_coverage": [
                target
                for target in TARGETS
                if target not in required_targets
            ],
            "epochs_max": int(args.epochs),
            "early_stop_patience": int(args.patience),
            "history": history,
            "current_deck_guide": guide_identity,
            "expert_pilot_importance": pilot_contract,
            "slowking_combo_state_validation": (
                {
                    "receipt": str(combo_receipt_path),
                    "receipt_sha256": combo_receipt_digest,
                    "payload": combo_receipt,
                }
                if combo_receipt is not None
                else None
            ),
            **(
                {
                    "decision_fusion": {
                        **fusion_identity,
                        "runtime_enabled": True,
                    }
                }
                if fusion_identity is not None
                else {}
            ),
            **(
                {
                    "expanded_head_migration": hot_start_expansion.get(
                        "expanded_head_migration"
                    ),
                    "expanded_target_contract": expanded_targets,
                    "expanded_target_schema_digest": expanded_identity[
                        "target_schema_digest"
                    ],
                    "expanded_schedule_schema": expanded_identity[
                        "schedule_schema"
                    ],
                    "expanded_schedule_digest": expanded_identity[
                        "schedule_digest"
                    ],
                    "expanded_schedule": expanded_identity["schedule"],
                    "expanded_head_training": best_expanded_training,
                    "runtime_enabled_heads": [],
                }
                if expanded_identity is not None
                else {}
            ),
        },
        evidence={
            "kind": "specialist_episode_disjoint_expert_validation",
            "best_metric": best_metric,
            "epochs_completed": len(history),
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(frozen["model_path"]).parent)
    ready = {
        "schema": ready_schema,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_digest": frozen["checkpoint_digest"],
        "core_checkpoint_digest": core["checkpoint_digest"],
        "hot_start_checkpoint": str(hot_start),
        "hot_start_checkpoint_digest": hot_start_digest,
        "acting_seat_archetype": archetype,
        "family": family_name,
        "expert_manifest": identity.path,
        "expert_manifest_sha256": identity.digest,
        **(
            {
                "starmie_manifest": identity.path,
                "starmie_manifest_sha256": identity.digest,
            }
            if archetype == "starmie"
            else {}
        ),
        "records": identity.records,
        "decisions": identity.decisions,
        "epochs_completed": len(history),
        "epochs_max": int(args.epochs),
        "early_stop_patience": int(args.patience),
        "best_metric": best_metric,
        "trained_target_coverage": list(required_targets),
        "current_deck_guide": guide_identity,
        "expert_pilot_importance": pilot_contract,
        "expert_pilot_importance_index_sha256": pilot_importance_digest,
        "slowking_combo_state_validation": (
            {
                "receipt": str(combo_receipt_path),
                "receipt_sha256": combo_receipt_digest,
                "payload": combo_receipt,
            }
            if combo_receipt is not None
            else None
        ),
        "inherited_target_coverage": [
            target for target in TARGETS if target not in required_targets
        ],
        **(
            {
                "expanded_head_migration": hot_start_expansion.get(
                    "expanded_head_migration"
                ),
                "expanded_target_contract": expanded_targets,
                "expanded_target_schema": expanded_identity["target_schema"],
                "expanded_target_schema_digest": expanded_identity[
                    "target_schema_digest"
                ],
                "expanded_schedule_schema": expanded_identity[
                    "schedule_schema"
                ],
                "expanded_schedule_digest": expanded_identity[
                    "schedule_digest"
                ],
                "expanded_schedule": expanded_identity["schedule"],
                "expanded_head_training": best_expanded_training,
                "expanded_heads_trained": list(EXPANDED_HEAD_IDS),
                "runtime_enabled_heads": [],
                "flat_policy_remains_authoritative": fusion_identity is None,
            }
            if expanded_identity is not None
            else {}
        ),
        **(
            {
                "decision_fusion": {
                    **fusion_identity,
                    "runtime_enabled": True,
                }
            }
            if fusion_identity is not None
            else {}
        ),
    }
    atomic_json(args.ready, ready)
    atomic_json(state_path, {**state, "status": "complete", "ready": ready})
    print(json.dumps(ready, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
