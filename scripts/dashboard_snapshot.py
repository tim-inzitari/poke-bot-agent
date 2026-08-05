#!/usr/bin/env python3
"""Emit one lightweight JSON snapshot for the LAN training dashboard."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any


SERVICE = "pokemon-state-bootstrap.service"
EXACT_SERVICE = "pokemon-privileged-belief-full-blackwell-v1.service"
LATEST10_BOOTSTRAP_SERVICE = "pokemon-latest10-bootstrap.service"
LATEST10_FINALIZER_SERVICE = "pokemon-latest10-finalize.service"
CORE_RL_SERVICE = "pokebot-pure-rl-continuous-rehearsal.service"
ALAKAZAM_BOOTSTRAP_SERVICE = "pokebot-pure-rl-alakazam-bootstrap.service"
ALAKAZAM_SPECIALIST_SERVICE = "pokebot-pure-rl-alakazam.service"
FINAL_FORMAT_ALAKAZAM_SERVICE = (
    "pokebot-final-format-alakazam-r79-ordinary-bootstrap.service"
)
FINAL_FORMAT_ALAKAZAM_H10_SERVICE = (
    "pokebot-final-format-alakazam-r79-h10.service"
)
FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_SERVICE = (
    "pokebot-final-format-marnie-r104-h10-bootstrap.service"
)
FINAL_FORMAT_MARNIE_H10_RL_SERVICE = (
    "pokebot-final-format-marnie-r104-h10-rl.service"
)
FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE = (
    "pokebot-final-format-crustle-r113-h10-bootstrap.service"
)
MARNIE_POSTUPLOAD_FAMILY_STUDY_SERVICE = (
    "pokebot-marnie-postupload-family-study-r136.service"
)
MARNIE_POSTUPLOAD_BOOTSTRAP_SERVICE = (
    "pokebot-marnie-postupload-weighted-bootstrap-r135.service"
)
STRONG_PUBLIC_GATE_SERVICE = "pokebot-alakazam-strong-public-gate.service"
ROOT = Path("/home/inzi/poke-bot-agent")


def _selected_specialist_runtime_root(
    selector: Path = Path("/home/inzi/.config/pokebot/specialist_runtime.env"),
    fallback: Path = Path(
        "/home/inzi/poke-bot-agent-deployments/specialist-handoff-current"
    ),
) -> Path:
    """Resolve the same runtime root as the managed trainer selector."""

    try:
        for raw_line in selector.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw_line.partition("=")
            if (
                separator
                and key.strip() == "POKEBOT_SPECIALIST_RUNTIME_ROOT"
                and value.strip()
            ):
                return Path(value.strip()).expanduser()
    except OSError:
        pass
    return fallback


SPECIALIST_RUNTIME_ROOT = Path(
    os.environ.get(
        "POKEBOT_SPECIALIST_RUNTIME_ROOT",
        str(_selected_specialist_runtime_root()),
    )
)
BOOTSTRAP_LOG = ROOT / "outputs/logs/bootstrap.log"
ALAKAZAM_BOOTSTRAP_LOG = ROOT / "outputs/logs/alakazam_expert_bootstrap.log"
ALAKAZAM_TRANSITION_LOG = ROOT / "outputs/logs/deck_agnostic_core_transition.log"
ALAKAZAM_TRANSITION_STATE = (
    ROOT / "outputs/state/deck-agnostic-core-transition.json"
)
ALAKAZAM_BUILD_READY = ROOT / "outputs/state/alakazam-specialist-build-ready.json"
ALAKAZAM_BOOTSTRAP_READY = (
    ROOT / "outputs/state/alakazam-expert-bootstrap-ready.json"
)
FINAL_FORMAT_ALAKAZAM_ROOT = ROOT / "outputs/final_format_alakazam_r79"
FINAL_FORMAT_ALAKAZAM_LOG = (
    FINAL_FORMAT_ALAKAZAM_ROOT / "logs/ordinary_bootstrap.log"
)
FINAL_FORMAT_ALAKAZAM_H10_LOG = FINAL_FORMAT_ALAKAZAM_ROOT / "logs/h10_rl.log"
FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_LOG = (
    FINAL_FORMAT_ALAKAZAM_ROOT / "logs/h10_rl.progress.log"
)
FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_STATUS = (
    FINAL_FORMAT_ALAKAZAM_ROOT / "logs/h10_rl.progress.status"
)
FINAL_FORMAT_ALAKAZAM_H10_RUN_DIR = (
    ROOT
    / "outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k"
)
FINAL_FORMAT_ALAKAZAM_H10_REGISTRY = (
    FINAL_FORMAT_ALAKAZAM_ROOT
    / "runtime/specialist_runtime_registry_h10_r100_rating1150_minimum_iter11_all_remotes.json"
)
FINAL_FORMAT_ALAKAZAM_H10_CAPACITY_RECEIPT = (
    ROOT / "state/final_format_alakazam_h10_mix_r82.json"
)
FINAL_FORMAT_ALAKAZAM_READY = (
    FINAL_FORMAT_ALAKAZAM_ROOT
    / "receipts/ordinary_alakazam_refresh_bootstrap_ready.json"
)
FINAL_FORMAT_ALAKAZAM_STATE = (
    FINAL_FORMAT_ALAKAZAM_ROOT
    / "output/bootstrap/ordinary_fallback_core9/state.json"
)
FINAL_FORMAT_ALAKAZAM_MODEL_INVENTORY = (
    FINAL_FORMAT_ALAKAZAM_ROOT
    / "receipts/final_format_alakazam_model_inventory_r79.json"
)
FINAL_FORMAT_MARNIE_ROOT = ROOT / "outputs/final_format_marnie_r104"
FINAL_FORMAT_CRUSTLE_ROOT = ROOT / "outputs/final_format_crustle_r113"
FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG = (
    FINAL_FORMAT_CRUSTLE_ROOT / "logs/bootstrap.log"
)
FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE = (
    ROOT / "outputs/state/marnie-canonical-training-freeze-r163.json"
)
FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_LOG = (
    FINAL_FORMAT_MARNIE_ROOT / "logs/h10_bootstrap.log"
)
FINAL_FORMAT_MARNIE_H10_LOG = FINAL_FORMAT_MARNIE_ROOT / "logs/h10_rl.log"
FINAL_FORMAT_MARNIE_H10_PROGRESS_LOG = (
    FINAL_FORMAT_MARNIE_ROOT / "logs/h10_rl.progress.log"
)
FINAL_FORMAT_MARNIE_H10_PROGRESS_STATUS = (
    FINAL_FORMAT_MARNIE_ROOT / "logs/h10_rl.progress.status"
)
MARNIE_POSTUPLOAD_FAMILY_STUDY_LOG = (
    FINAL_FORMAT_MARNIE_ROOT / "logs/postupload_family_study_r136.log"
)
MARNIE_POSTUPLOAD_BOOTSTRAP_LOG = (
    FINAL_FORMAT_MARNIE_ROOT / "logs/postupload_bootstrap_r138.log"
)
MARNIE_POSTUPLOAD_FAMILY_STUDY_ROOT = (
    ROOT / "outputs/studies/marnie-archetype-family-r136"
)
MARNIE_POSTUPLOAD_FAMILY_ACTIVATION_REQUEST = (
    ROOT
    / "outputs/state/marnie-archetype-family-r130"
    / "activation-request.json"
)
MARNIE_POSTUPLOAD_FAMILY_MIGRATION = (
    ROOT
    / "outputs/state/marnie-archetype-family-r130"
    / "migration-receipt.json"
)
MARNIE_GUIDE_SHADOW_NONAUTHORITY = (
    ROOT / "state/marnie_guide_shadow_non_authority_r141.json"
)
MARNIE_FAMILY_GUIDE_SHADOW_RUNTIME = (
    ROOT / "state/marnie_family_guide_shadow_runtime_r142.json"
)
MARNIE_EPOCH_RECOVERY = (
    ROOT / "state/marnie_postupload_epoch1_recovery_r141.json"
)
MARNIE_POSTUPLOAD_PAUSE = (
    ROOT
    / "outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k"
    / "family_activation/await_upload_after_iter_00009.json"
)
MARNIE_ITERATION9_UPLOAD_TRIGGER = (
    ROOT
    / "outputs/state/marnie-archetype-family-r130"
    / "iteration9-upload-trigger.json"
)
FINAL_FORMAT_MARNIE_H10_RUN_DIR = (
    ROOT / "outputs/pure_rl/final_format_marnie_r104_h10_i_v6_8k"
)
FINAL_FORMAT_MARNIE_H10_REGISTRY = (
    FINAL_FORMAT_MARNIE_ROOT
    / "runtime/specialist_runtime_registry_h10_r104_fusion_v3.json"
)
FINAL_FORMAT_MARNIE_H10_READY = (
    ROOT / "outputs/state/final-format-marnie-r104-h10-bootstrap-ready.json"
)
FINAL_FORMAT_MARNIE_H10_VALIDATION = (
    ROOT / "outputs/state/final-format-marnie-r104-h10-validation.json"
)
FINAL_FORMAT_MARNIE_ROUTER_V6_FIX = (
    ROOT
    / "outputs/state/final-format-marnie-h10-router-v6-registration-fix-r107.json"
)
FINAL_FORMAT_MARNIE_EXPERT = (
    ROOT
    / "data/bootstrap/expert-latest20-2026-07-04-2026-07-23-roster18-v6-strategic"
    / "marnie-s-grimmsnarl-ex/PROTECTED_EXPERT_CORPUS.json"
)
EXACT_LOG = ROOT / "outputs/logs/privileged-belief-full-blackwell.log"
EXACT_ROOT = ROOT / "outputs/privileged_belief/exact_core_20k_v1"
EXACT_RESIDENT_STATUS = EXACT_ROOT / "resident_train.status.json"
EXACT_STREAM_STATUS = EXACT_ROOT / "full_train.status.json"
LATEST10_FINALIZER_LOG = ROOT / "outputs/logs/latest10-finalize.log"
LATEST10_READY = (
    ROOT / "data/bootstrap/latest10-20260709-20260718/READY.json"
)
LATEST10_BERT_STATUS = (
    ROOT / "data/bootstrap/latest10-20260709-20260718/bert-staging-status.json"
)
TRAINING_STATUS = ROOT / "outputs/logs/training.progress.status"
TRAINING_LOG = ROOT / "outputs/logs/training.log"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LATEST10_STATUS = ROOT / "scripts/latest10_status.py"
DASHBOARD_ITERATION_TIMER = ROOT / "outputs/state/dashboard_iteration_timer.json"
MODEL_PROFILE_REGISTRY = ROOT / "outputs/state/pure_rl_model_profiles.json"
DORMANT_MODEL_MODULES = ROOT / "outputs/state/alakazam_dormant_model_modules_v1.json"
STAGED_MATCHUP_ADAPTER_ROSTER = ROOT / "state/matchup_adapter_roster_v4.json"
CANONICAL_MATCHUP_ADAPTER_ROSTER = ROOT / "state/matchup_adapter_roster.json"
MATCHUP_RUNTIME_PRODUCTION_READY = (
    ROOT / "outputs/state/matchup-runtime-v31-production-ready.json"
)
MATCHUP_RUNTIME_BOUNDARY = (
    ROOT / "outputs/state/alakazam-matchup-runtime-iter26-v31.json"
)
PUBLIC_MIX_LIVE_WR = ROOT / "outputs/state/public_mix_live_wr.json"
COMPETITION_GATE_PROGRAM = ROOT / "ops/alakazam_gate_program_v1.json"
RESEARCH_CONTROL_REGISTRY = ROOT / "ops/research_control_registry_v1.json"
RESEARCH_CONTROL_REGISTRY_LATEST = (
    ROOT / "outputs/state/research_control_registry_latest.json"
)
SPECIALIST_PROTOCOL_STATE = Path(
    os.environ.get(
        "POKEBOT_SPECIALIST_PROTOCOL_STATE",
        # GOAL.md owns this as mutable canonical program state. Runtime
        # deployment copies are immutable launch inputs and can legitimately
        # lag a separately versioned post-fleet refresh.
        str(ROOT / "state/specialists.yaml"),
    )
)
NEXT_SPECIALIST_PRESTAGE_STATE = (
    ROOT / "outputs/state/next-specialist-prestage-v1.json"
)
POPULATION_ROUND_ROBIN_STATE = (
    ROOT / "outputs/state/population-round-robin-state-v1.json"
)
POPULATION_ROUND_ROBIN_SERVICE = "pokebot-population-round-robin.service"
LATEST20_SPECIALIST_SYNC_SERVICE = (
    "pokebot-expert-latest20-specialist-sync.service"
)
LATEST20_SPECIALIST_CURRENT = (
    ROOT / "data/bootstrap/current-specialist-latest20"
)
LATEST20_SPECIALIST_SYNC_STATE = (
    ROOT / "outputs/state/expert-latest20-specialist-sync.json"
)
V6_STRATEGIC_SPECIALIST_SYNC_SERVICE = (
    "pokebot-v6-strategic-corpus-sync.service"
)
V6_STRATEGIC_SPECIALIST_CURRENT = (
    ROOT / "data/bootstrap/current-specialist-latest20-v6-strategic"
)
V6_STRATEGIC_SPECIALIST_SYNC_STATE = (
    ROOT / "outputs/state/expert-latest20-v6-strategic-sync.json"
)
V6_STRATEGIC_STAGED_SYNC_SERVICE = (
    "pokebot-expert-latest20-v6-strategic-r109-stage-sync-11m.service"
)
V6_STRATEGIC_STAGED_CURRENT = (
    ROOT / "data/bootstrap/staged-specialist-latest20-v6-strategic-r109"
)
V6_STRATEGIC_STAGED_SYNC_STATE = (
    ROOT
    / "outputs/state/expert-latest20-v6-strategic-r109-staged-sync.json"
)
V6_STRATEGIC_STAGED_DATES = [
    (date(2026, 7, 14) + timedelta(days=offset)).isoformat()
    for offset in range(20)
]
MARNIE_LATEST20_RUNTIME_ACTIVATION_STATE = (
    ROOT
    / "outputs/state/final-format-marnie-r104-latest20-runtime-activation-r109.json"
)
V6_STRATEGIC_TARGET_SCHEMA = "poke_bot.expanded_strategic_targets/v2"
V6_STRATEGIC_TARGET_DIGEST = (
    "sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"
)
SPECIALIST_RUNTIME_REGISTRY = (
    SPECIALIST_RUNTIME_ROOT / "ops/specialist_runtime_registry_v1.json"
)
FROZEN_SPECIALIST_REGISTRY = (
    SPECIALIST_RUNTIME_ROOT / "ops/frozen_specialist_registry_v1.json"
)
OWNER_SPECIALIST_HANDOFF_SERVICE = "pokebot-owner-alakazam-handoff.service"
OWNER_SPECIALIST_HANDOFF_STATE = (
    ROOT / "outputs/state/post-alakazam-specialist-handoff-v2.json"
)
OWNER_SPECIALIST_HANDOFF_LOG = (
    ROOT / "outputs/logs/owner-accepted-alakazam-handoff.log"
)
OWNER_CORE_DISTILL_STATE = (
    ROOT / "outputs/bootstrap/owner-accepted-alakazam-balanced-core-v1/state.json"
)
OWNER_TREVENANT_BOOTSTRAP_STATE = (
    ROOT / "outputs/bootstrap/hops-trevenant-expert-bootstrap-from-core-v1/state.json"
)
POST_STARMIE_HANDOFF_SERVICE = (
    "pokebot-post-starmie-next-specialist-handoff.service"
)
POST_STARMIE_HANDOFF_STATE = (
    ROOT / "outputs/state/post-starmie-core-v2-handoff-v1.json"
)
POST_STARMIE_HANDOFF_LOG = (
    ROOT / "outputs/logs/post-starmie-core-v2-handoff.log"
)
SPECIALIST_CYCLE_HANDOFF_SERVICE = "pokebot-specialist-cycle-handoff.service"
SPECIALIST_CYCLE_HANDOFF_LOG = (
    ROOT / "outputs/logs/specialist-transition-graph.log"
)
SPECIALIST_TRANSITION_GRAPH_STATE = (
    ROOT / "outputs/state/specialist-transition-graph.json"
)
STARMIE_PASSED_GATE_HANDLER_STATE = (
    ROOT / "outputs/state/starmie-passed-gate-handler-v1.json"
)
EXPERT20_ROOT = ROOT / "data/bootstrap/expert-latest20-20260702-20260721"
EXPERT20_CURRENT_RECEIPT = (
    ROOT / "outputs/state/expert-latest20-current.json"
)
EXPERT20_REFRESH_STATUS = EXPERT20_ROOT / "refresh.status.json"
EXPERT20_INZI_STATUS = EXPERT20_ROOT / "inzi.features.status.json"
EXPERT20_INZI_FINAL_STATUS = EXPERT20_ROOT / "inzi-final.features.status.json"
EXPERT20_INZI_TAIL_STATUS = EXPERT20_ROOT / "inzi-tail.status.json"
EXPERT20_FEATURE_DIR = EXPERT20_ROOT / "features-inzi"
EXPERT20_ASSEMBLED_MANIFEST = (
    EXPERT20_FEATURE_DIR / "all-recognized-latest20.manifest.json"
)
EXPERT20_ALAKAZAM_CORPUS = EXPERT20_ROOT / "alakazam/PROTECTED_EXPERT_CORPUS.json"
EXPERT20_ELMO_DAILY_STATUS_GLOB = (
    "/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/"
    "daily/roster18-v5/status/*.json"
)
EXPERT20_V6_STRATEGIC_ELMO_DAILY_STATUS_GLOB = (
    "/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/"
    "daily/roster18-v6-strategic/status/*.json"
)
STRONG_PUBLIC_GATE_PROGRESS = (
    ROOT / "outputs/logs/alakazam_strong_public_gate.progress.status"
)
STRONG_PUBLIC_GATE_LOG = ROOT / "outputs/logs/alakazam_strong_public_gate.progress.log"
PROTECTED_BASELINE_GATE = Path(
    "/home/inzi/poke-bot-model-registry/alakazam_baseline_gate/manifest.json"
)
LEGACY_RESEARCH_CONTROL_DIGESTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
OFFICIAL_BASELINE_IDS = tuple(LEGACY_RESEARCH_CONTROL_DIGESTS)
FROZEN_SPECIALIST_DISPLAY_NAMES = {
    "alakazam": "Alakazam",
    "hops-trevenant": "Hop's Trevenant",
    "starmie": "Mega Starmie ex",
    "lucario": "Mega Lucario ex",
    "dragapult-dusknoir": "Dragapult Dusknoir",
}
EXPANDED_HEAD_CONTRACT_SCHEMA = "poke_bot.expanded_head_training/v1"
EXPANDED_HEAD_MODULES = {
    "action_q": "action_q_head",
    "action_type": "action_type_head",
    "action_target": "action_target_head",
    "action_resource": "action_resource_head",
    "action_utility": "action_utility_head",
    "tactical_outcome": "tactical_outcome_head",
    "opponent_response": "opponent_response_head",
    "resource_forecast": "resource_forecast_head",
    "game_phase": "game_phase_head",
    "outcome_distribution": "outcome_distribution_head",
    "remaining_turns": "remaining_turns_head",
}
EXPANDED_HEAD_IDS_BY_MODULE = {
    module: head_id for head_id, module in EXPANDED_HEAD_MODULES.items()
}
DECISION_FUSION_SCHEMA = "poke_bot.causal_decision_fusion/v1"
DECISION_FUSION_V2_SCHEMA = "poke_bot.causal_decision_fusion/v2"
DECISION_FUSION_V2_ROUTE_SCHEMA = "option_conditioned_per_head/v2"
DECISION_FUSION_V3_SCHEMA = "poke_bot.causal_decision_fusion/v3"
DECISION_FUSION_V3_ROUTE_SCHEMA = "typed_output_centered_per_head/v3"
DECISION_FUSION_V3_MIN_RELIABILITY = 0.25
DECISION_FUSION_V3_MAX_RELIABILITY = 4.0
DECISION_FUSION_REQUIRED_HEADS = (
    "value",
    "archetype",
    "opponent_hand",
    "opponent_remainder",
    "lethal_threat",
    "prize_race",
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcomes",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
)
DECISION_FUSION_OPTIONAL_HEAD_FLAGS = (
    ("setup_board_outcome_head_enabled", "setup_board_outcome"),
    ("combo_state_head_enabled", "combo_state"),
)


def _valid_activation_fusion_inventory(
    schema: object,
    required_heads: object,
) -> bool:
    """Validate the exact learned-head inventory named by an activation receipt.

    V1 has the original 17 fused heads.  V2 adds independently routed heads
    one at a time (setup, combo, or both) according to the specialist's
    materialized architecture.  An activation receipt has no model-config
    flags to derive that choice from, so accept only those four exact V2
    inventories rather than treating every non-guide list as equivalent.
    """

    if not isinstance(required_heads, list):
        return False
    required = tuple(str(head) for head in required_heads)
    base = tuple(DECISION_FUSION_REQUIRED_HEADS)
    if schema == DECISION_FUSION_SCHEMA:
        return required == base
    if schema not in {DECISION_FUSION_V2_SCHEMA, DECISION_FUSION_V3_SCHEMA}:
        return False
    return required in {
        base,
        (*base, "setup_board_outcome"),
        (*base, "combo_state"),
        (*base, "setup_board_outcome", "combo_state"),
    }


def reconcile_frozen_specialist_label(row: dict[str, Any]) -> dict[str, Any]:
    """Prevent a stale handoff label from cross-labeling a frozen checkpoint."""

    result = dict(row)
    if result.get("frozen_specialist") is not True:
        return result
    archetype_id = str(result.get("archetype_id") or "")
    canonical = FROZEN_SPECIALIST_DISPLAY_NAMES.get(archetype_id)
    if not canonical:
        return result
    expected = f"Frozen {canonical} specialist"
    observed = str(result.get("archetype_label") or "")
    if observed != expected:
        result["source_archetype_label"] = observed
        result["archetype_label"] = expected
        result["archetype_label_reconciled"] = True
    return result


def run(argv: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def as_number(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_sha256_digest(value: object) -> bool:
    return re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")) is not None


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _file_sha256_matches(path: Path, expected: object) -> bool:
    actual = _file_sha256(path)
    if actual is None:
        return False
    return actual.removeprefix("sha256:") == str(expected or "").removeprefix(
        "sha256:"
    )


def _expanded_head_id(value: object) -> str | None:
    """Normalize checkpoint-contract names without guessing unknown modules."""

    raw = str(value or "").strip()
    if raw in EXPANDED_HEAD_MODULES:
        return raw
    return EXPANDED_HEAD_IDS_BY_MODULE.get(raw)


def _expanded_head_set(value: object) -> tuple[set[str], set[str]]:
    """Return recognized and unknown head identifiers from a contract field."""

    if isinstance(value, dict):
        raw_values = [
            key
            for key, enabled in value.items()
            if enabled is not False and enabled is not None
        ]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif value in (None, ""):
        raw_values = []
    else:
        raw_values = [value]
    recognized: set[str] = set()
    unknown: set[str] = set()
    for raw in raw_values:
        head_id = _expanded_head_id(raw)
        if head_id is None:
            unknown.add(str(raw))
        else:
            recognized.add(head_id)
    return recognized, unknown


def _head_number(
    sources: list[dict[str, Any]],
    head_id: str,
    *fields: str,
) -> float | None:
    """Read one per-head metric from nested or flat receipt mappings."""

    module = EXPANDED_HEAD_MODULES[head_id]
    aliases = (head_id, module)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for field in fields:
            number = as_float(source.get(field))
            if number is not None:
                return number
        for alias in aliases:
            nested = source.get(alias)
            if isinstance(nested, dict):
                for field in fields:
                    number = as_float(nested.get(field))
                    if number is not None:
                        return number
            elif fields:
                number = as_float(nested)
                if number is not None:
                    return number
        for alias in aliases:
            for field in fields:
                for key in (f"{alias}_{field}", f"{field}_{alias}"):
                    number = as_float(source.get(key))
                    if number is not None:
                        return number
    return None


def _expanded_head_checkpoint_contract(
    state_dict: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Audit expanded-head metadata against the exact checkpoint tensors.

    Absence is a valid legacy-V5 state. Once any expanded-head tensor or
    metadata exists, the contract becomes mandatory and every declaration is
    checked fail-closed.
    """

    tensor_inventory: dict[str, dict[str, Any]] = {}
    for head_id, module in EXPANDED_HEAD_MODULES.items():
        tensors = {
            str(key): value
            for key, value in state_dict.items()
            if str(key).startswith(module + ".") and hasattr(value, "numel")
        }
        if not tensors:
            continue
        tensor_inventory[head_id] = {
            "module": module,
            "tensor_count": len(tensors),
            "parameter_count": sum(int(value.numel()) for value in tensors.values()),
            "tensor_shapes": {
                key: list(getattr(value, "shape", ())) for key, value in tensors.items()
            },
        }
    actual = set(tensor_inventory)
    raw = extra.get("expanded_head_training")
    contract = dict(raw) if isinstance(raw, dict) else {}
    if not contract:
        return {
            "schema": EXPANDED_HEAD_CONTRACT_SCHEMA,
            "available": False,
            "verified": not actual,
            "legacy_v5": not actual,
            "reason": (
                "legacy checkpoint has no expanded strategic heads"
                if not actual
                else "expanded-head tensors exist without required metadata"
            ),
            "actual_tensor_heads": sorted(actual),
            "heads": [
                {
                    "id": head_id,
                    **tensor_inventory[head_id],
                    "present": True,
                    "trained": False,
                    "gradient_enabled": False,
                    "runtime_enabled": False,
                    "contract_valid": False,
                }
                for head_id in sorted(actual)
            ],
        }

    declared_value = contract.get(
        "architecture_present_heads",
        contract.get("present_heads"),
    )
    if declared_value is None and isinstance(contract.get("heads"), dict):
        declared_value = {
            name: row.get("present", True) if isinstance(row, dict) else True
            for name, row in dict(contract["heads"]).items()
        }
    declared, unknown_declared = _expanded_head_set(declared_value)
    trained, unknown_trained = _expanded_head_set(contract.get("trained_heads"))
    gradient, unknown_gradient = _expanded_head_set(
        contract.get("gradient_enabled_heads")
    )
    runtime, unknown_runtime = _expanded_head_set(
        contract.get("runtime_enabled_heads")
    )
    head_rows = dict(contract.get("heads") or {})
    unknown_head_rows: set[str] = set()
    for raw_name, row in head_rows.items():
        if not isinstance(row, dict):
            continue
        head_id = _expanded_head_id(raw_name)
        if head_id is None:
            unknown_head_rows.add(str(raw_name))
            continue
        if row.get("trained") is True:
            trained.add(head_id)
        if row.get("gradient_enabled") is True:
            gradient.add(head_id)
        if row.get("runtime_enabled") is True:
            runtime.add(head_id)

    missing_tensors = declared - actual
    undeclared_tensors = actual - declared
    invalid_subsets = {
        "trained_without_tensor": sorted(trained - actual),
        "gradient_without_tensor": sorted(gradient - actual),
        "runtime_without_tensor": sorted(runtime - actual),
    }
    unknown = sorted(
        unknown_declared
        | unknown_trained
        | unknown_gradient
        | unknown_runtime
        | unknown_head_rows
    )
    schema_valid = contract.get("schema") == EXPANDED_HEAD_CONTRACT_SCHEMA
    declaration_present = declared_value is not None
    verified = bool(
        schema_valid
        and declaration_present
        and not missing_tensors
        and not undeclared_tensors
        and not unknown
        and not any(invalid_subsets.values())
    )
    failures: list[str] = []
    if not schema_valid:
        failures.append("unsupported expanded-head contract schema")
    if not declaration_present:
        failures.append("architecture-present head declaration is absent")
    if missing_tensors:
        failures.append("declared heads lack checkpoint tensors")
    if undeclared_tensors:
        failures.append("checkpoint tensors are absent from head declaration")
    if unknown:
        failures.append("contract names unknown expanded heads")
    if any(invalid_subsets.values()):
        failures.append("trained/gradient/runtime sets are not tensor-backed")

    loss_weights = dict(contract.get("loss_weights") or {})
    train_metrics = dict(contract.get("train_metrics") or {})
    validation_metrics = dict(contract.get("validation_metrics") or {})
    coverage_metrics = dict(contract.get("coverage") or {})
    result_heads: list[dict[str, Any]] = []
    for head_id in sorted(actual | declared):
        module = EXPANDED_HEAD_MODULES[head_id]
        raw_row = head_rows.get(head_id, head_rows.get(module, {}))
        row = dict(raw_row) if isinstance(raw_row, dict) else {}
        labeled_rows = _head_number(
            [row, coverage_metrics],
            head_id,
            "labeled_rows",
            "label_rows",
            "rows",
        )
        masked_rows = _head_number(
            [row, coverage_metrics],
            head_id,
            "masked_rows",
            "mask_rows",
        )
        total_rows = _head_number(
            [row, coverage_metrics],
            head_id,
            "total_rows",
        )
        if total_rows is None and labeled_rows is not None and masked_rows is not None:
            total_rows = labeled_rows + masked_rows
        coverage = _head_number(
            [row, coverage_metrics],
            head_id,
            "coverage",
            "coverage_fraction",
        )
        if coverage is None and labeled_rows is not None and total_rows:
            coverage = labeled_rows / total_rows
        inventory = tensor_inventory.get(
            head_id,
            {
                "module": module,
                "tensor_count": 0,
                "parameter_count": 0,
                "tensor_shapes": {},
            },
        )
        result_heads.append(
            {
                "id": head_id,
                **inventory,
                "present": head_id in actual,
                "declared_present": head_id in declared,
                "trained": head_id in trained,
                "gradient_enabled": head_id in gradient,
                "runtime_enabled": head_id in runtime,
                "loss_weight": _head_number(
                    [row, loss_weights], head_id, "weight", "loss_weight"
                )
                or 0.0,
                "train_loss": _head_number(
                    [row, train_metrics], head_id, "loss", "train_loss"
                ),
                "validation_loss": _head_number(
                    [row, validation_metrics], head_id, "loss", "validation_loss"
                ),
                "labeled_rows": (
                    int(labeled_rows) if labeled_rows is not None else None
                ),
                "masked_rows": (
                    int(masked_rows) if masked_rows is not None else None
                ),
                "total_rows": int(total_rows) if total_rows is not None else None,
                "coverage": coverage,
                "contract_valid": verified,
            }
        )
    return {
        "schema": EXPANDED_HEAD_CONTRACT_SCHEMA,
        "available": True,
        "verified": verified,
        "legacy_v5": False,
        "reason": None if verified else "; ".join(failures),
        "contract_schema": contract.get("schema"),
        "contract_digest": _canonical_json_digest(contract),
        "target_schema_version": contract.get("target_schema_version"),
        "target_schema_digest": contract.get("target_schema_digest"),
        "schedule_version": contract.get("schedule_version"),
        "schedule_digest": contract.get("schedule_digest"),
        "stage": contract.get("stage"),
        "epoch": contract.get("epoch"),
        "epochs_total": contract.get("epochs_total"),
        "actual_tensor_heads": sorted(actual),
        "declared_present_heads": sorted(declared),
        "trained_heads": sorted(trained),
        "gradient_enabled_heads": sorted(gradient),
        "runtime_enabled_heads": sorted(runtime),
        "missing_tensor_heads": sorted(missing_tensors),
        "undeclared_tensor_heads": sorted(undeclared_tensors),
        "unknown_declared_heads": unknown,
        **invalid_subsets,
        "heads": result_heads,
    }


def _decision_fusion_checkpoint_contract(
    state_dict: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Prove all-head decision use from the exact committed checkpoint.

    Architecture presence, training-warmup, and runtime activation are distinct
    states.  The dashboard must never translate a staged or training-only
    fusion into a serving claim.
    """

    model_config = dict(payload.get("model_config") or {})
    provenance = dict(payload.get("provenance") or {})
    declared = dict(provenance.get("decision_fusion") or {})
    tensors = {
        str(key): value
        for key, value in state_dict.items()
        if str(key).startswith("decision_fusion.")
        and hasattr(value, "numel")
    }
    architecture_enabled = model_config.get("decision_fusion_enabled") is True
    runtime_enabled = (
        model_config.get("decision_fusion_runtime_enabled") is True
    )
    expected_required = [
        *DECISION_FUSION_REQUIRED_HEADS,
        *(
            head
            for flag, head in DECISION_FUSION_OPTIONAL_HEAD_FLAGS
            if model_config.get(flag) is True
        ),
    ]
    required = list(declared.get("required_heads") or [])
    declared_schema = declared.get("schema")
    dedicated = dict(declared.get("dedicated_routes") or {})
    routed_fusion_verified = bool(
        declared_schema in {
            DECISION_FUSION_V2_SCHEMA,
            DECISION_FUSION_V3_SCHEMA,
        }
        and model_config.get("decision_fusion_dedicated_routes_enabled") is True
        and bool(dedicated.get("runtime_enabled"))
        == bool(
            model_config.get(
                "decision_fusion_dedicated_routes_runtime_enabled"
            )
        )
        and dedicated.get("schema")
        == (
            DECISION_FUSION_V3_ROUTE_SCHEMA
            if declared_schema == DECISION_FUSION_V3_SCHEMA
            else DECISION_FUSION_V2_ROUTE_SCHEMA
        )
        and list(dedicated.get("route_names") or []) == expected_required
        and int(dedicated.get("route_count") or -1) == len(expected_required)
        and dedicated.get("aggregation") == "fixed_mean"
        and float(dedicated.get("total_delta_cap") or -1.0) == 1.0
        and dedicated.get("zero_safe_final_projection") is True
        and declared.get("guide_excluded") is True
    )
    if declared_schema == DECISION_FUSION_V3_SCHEMA:
        routed_fusion_verified = bool(
            routed_fusion_verified
            and model_config.get(
                "decision_fusion_typed_output_centered_routes_enabled"
            )
            is True
            and float(
                model_config.get(
                    "decision_fusion_action_type_reliability_cap", -1.0
                )
            )
            == 0.25
            and (
                declared.get("typed_output_centered_routes") is True
                or dedicated.get("typed_output_centered") is True
            )
            and dedicated.get("positive_bounded_reliability") is True
            and list(dedicated.get("reliability_bounds") or [])
            == [
                DECISION_FUSION_V3_MIN_RELIABILITY,
                DECISION_FUSION_V3_MAX_RELIABILITY,
            ]
            and float(dedicated.get("action_type_reliability_cap") or -1.0)
            == 0.25
        )
    tensor_parameters = sum(int(value.numel()) for value in tensors.values())
    final_weight = tensors.get("decision_fusion.residual.2.weight")
    final_nonzero = bool(
        final_weight is not None
        and hasattr(final_weight, "count_nonzero")
        and int(final_weight.count_nonzero().item()) > 0
    )
    available = bool(architecture_enabled or tensors or declared)
    if not available:
        return {
            "schema": DECISION_FUSION_SCHEMA,
            "available": False,
            "verified": True,
            "phase": "not_materialized",
            "runtime_enabled": False,
            "training_enabled": False,
            "required_heads": expected_required,
            "required_head_count": len(expected_required),
            "reason": "decision fusion is not present in this checkpoint",
        }
    verified = bool(
        architecture_enabled
        and tensors
        and (
            declared_schema == DECISION_FUSION_SCHEMA
            or routed_fusion_verified
        )
        and required == expected_required
        and bool(declared.get("runtime_enabled")) == runtime_enabled
        and tensor_parameters > 0
    )
    phase = (
        "runtime_active"
        if verified and runtime_enabled
        else "training_warmup"
        if verified
        else "contract_mismatch"
    )
    return {
        "schema": declared_schema or DECISION_FUSION_SCHEMA,
        "available": True,
        "verified": verified,
        "phase": phase,
        "runtime_enabled": runtime_enabled if verified else False,
        "training_enabled": bool(verified),
        "serving_eligible": bool(verified and runtime_enabled),
        "required_heads": required,
        "required_head_count": len(required),
        "expected_required_head_count": len(expected_required),
        "all_required_heads_declared": required == expected_required,
        "tensor_count": len(tensors),
        "parameter_count": tensor_parameters,
        "trained_nonzero": final_nonzero,
        "matchup_adapter_behavior": "causal_route_gated",
        "absent_deck_guide_behavior": "exact_bypass",
        "reason": (
            None
            if verified
            else "checkpoint fusion tensors, model config, and provenance disagree"
        ),
    }


def _successor_decision_fusion_activation(
    *,
    state_root: Path,
    specialist_id: str,
    checkpoint_digest: str,
    run_dir: Path | None = None,
    design_fingerprint: str = "",
    initial_checkpoint_digest: str = "",
) -> dict[str, Any]:
    """Find a checksum-bound successor handoff that authorizes fused actions.

    The original Dudunsparce migration writes its activation into the run's
    loop state.  A generated successor is different: it starts from an
    immutable, already-fused bootstrap and its authorization is recorded by
    the sequential handoff receipt.  Treating only the former shape as valid
    made the dashboard report a false fail-closed regression even while the
    selector and checkpoint were correctly executing the fused policy.
    """

    specialist_id = str(specialist_id or "").strip().casefold()
    checkpoint_digest = str(checkpoint_digest or "").strip()
    if (
        not specialist_id
        or not checkpoint_digest.startswith("sha256:")
        or not state_root.is_dir()
    ):
        return {}
    candidates = sorted(
        state_root.glob(
            f"{specialist_id}-specialist-rl-activation-v*.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        receipt = read_json(path)
        identity = dict(receipt.get("identity") or {})
        bootstrap = dict(identity.get("next_specialist_bootstrap") or {})
        fusion = dict(bootstrap.get("decision_fusion") or {})
        registration = dict(identity.get("runtime_registration") or {})
        runtime_row = dict(registration.get("runtime_row") or {})
        runtime_fusion = dict(runtime_row.get("decision_fusion") or {})
        row_digest = "sha256:" + str(
            runtime_row.get("initial_checkpoint_sha256") or ""
        ).removeprefix("sha256:")
        receipt_valid = bool(
            receipt.get("schema") == "poke_bot.specialist_rl_activation/v2"
            and receipt.get("status") == "ready"
            and bootstrap.get("specialist_id") == specialist_id
            and registration.get("specialist_id") == specialist_id
            and row_digest == bootstrap.get("checkpoint_digest")
            and _valid_activation_fusion_inventory(
                fusion.get("schema"),
                fusion.get("required_heads"),
            )
            and fusion.get("runtime_enabled") is True
            and runtime_fusion.get("schema") == fusion.get("schema")
            and runtime_fusion.get("required") is True
            and runtime_fusion.get("runtime_enabled") is True
            and runtime_fusion.get("required_heads")
            == fusion.get("required_heads")
        )
        if not receipt_valid:
            continue
        bootstrap_digest = str(bootstrap.get("checkpoint_digest") or "")
        activation_scope = "successor_bootstrap"
        lineage_commit: Path | None = None
        if bootstrap_digest != checkpoint_digest:
            # A specialist's first immutable activation receipt binds its
            # bootstrap.  Every subsequently committed learner is a new
            # checksum, so exact-bootstrap-only matching falsely reports a
            # regression after the first successful iteration.  Accept a
            # descendant only when the run manifest binds the same bootstrap
            # and design fingerprint and an immutable completed commit binds
            # that fingerprint to the exact active learner checksum.
            if not (
                run_dir is not None
                and run_dir.is_dir()
                and design_fingerprint.startswith("sha256:")
                and initial_checkpoint_digest == bootstrap_digest
            ):
                continue
            lineage_fingerprints = _source_only_design_lineage_fingerprints(
                run_dir,
                design_fingerprint,
            )
            for commit_path in sorted(
                (run_dir / "commits").glob("iter_*.json"),
                key=lambda candidate: candidate.stat().st_mtime_ns,
                reverse=True,
            ):
                commit = read_json(commit_path)
                commit_fingerprint = str(
                    commit.get("design_fingerprint") or ""
                )
                if commit_fingerprint not in lineage_fingerprints:
                    continue
                # Current commit schema keeps the complete boundary record in
                # history[0], while the compact fixture and older schemas put
                # it at the top level.  Both are immutable receipt shapes.
                boundary_rows = [commit]
                boundary_rows.extend(
                    row
                    for row in (commit.get("history") or [])
                    if isinstance(row, dict)
                )
                for boundary in boundary_rows:
                    learner_after = dict(
                        boundary.get("learner_after") or {}
                    )
                    candidate = dict(boundary.get("candidate") or {})
                    publish = dict(
                        boundary.get("next_collection_publish") or {}
                    )
                    if not (
                        boundary.get("completed") is True
                        and learner_after.get("digest") == checkpoint_digest
                        and candidate.get("digest") == checkpoint_digest
                        and publish.get("digest") == checkpoint_digest
                        and publish.get("local_ok") is True
                        and str(
                            (
                                boundary.get("learner_before") or {}
                            ).get("digest")
                            or ""
                        ).startswith("sha256:")
                    ):
                        continue
                    lineage_commit = commit_path
                    activation_scope = (
                        "successor_committed_descendant"
                        if commit_fingerprint == design_fingerprint
                        else (
                            "successor_committed_descendant_"
                            "source_only_migration"
                        )
                    )
                    break
                if lineage_commit is not None:
                    break
            if lineage_commit is None:
                continue
        return {
            "schema": "poke_bot.successor_decision_fusion_activation/v1",
            "specialist_id": specialist_id,
            "checkpoint_digest": checkpoint_digest,
            "bootstrap_checkpoint_digest": bootstrap_digest,
            "runtime_enabled": True,
            "training_action_eligible": True,
            # Terminal freeze/serving still requires this child's own exact
            # premium and official gate receipt.
            "terminal_serving_eligible": False,
            "receipt": str(path),
            "receipt_digest": _file_sha256(path),
            "activation_scope": activation_scope,
            "lineage_commit": (
                str(lineage_commit) if lineage_commit is not None else None
            ),
            "lineage_commit_digest": (
                _file_sha256(lineage_commit)
                if lineage_commit is not None
                else None
            ),
        }
    return {}


def _final_refresh_decision_fusion_continuity(
    *,
    state_root: Path,
    specialist_id: str,
    checkpoint_digest: str,
) -> dict[str, Any]:
    """Read the explicit H10 committed-descendant fusion authorization.

    The final-format Alakazam refresh is not a normal roster successor and
    intentionally has no selector authority.  Its first RL learner is still a
    descendant of a separately activated H10 bootstrap, so validate its
    dedicated immutable continuity receipt rather than incorrectly demanding a
    V1 loop activation row.  This authorizes fused-policy *training* only;
    terminal serving remains false until the child's exact gates pass.
    """

    if (
        specialist_id.strip().casefold() != "alakazam"
        or not checkpoint_digest.startswith("sha256:")
    ):
        return {}
    path = state_root / "final_format_alakazam_fusion_continuity_r98.json"
    receipt = read_json(path)
    bootstrap = dict(receipt.get("bootstrap_activation") or {})
    bootstrap_fusion = dict(bootstrap.get("decision_fusion") or {})
    descendant = dict(receipt.get("committed_descendant") or {})
    registration = dict(receipt.get("runtime_registration") or {})
    runtime_fusion = dict(registration.get("decision_fusion") or {})
    required = bootstrap_fusion.get("required_heads")
    valid = bool(
        receipt.get("schema")
        == "poke_bot.final_format_alakazam_fusion_continuity/v1"
        and receipt.get("status") == "ready"
        and receipt.get("specialist_id") == "alakazam"
        and bootstrap.get("bootstrap_checkpoint_sha256")
        == descendant.get("learner_before_sha256")
        and descendant.get("checkpoint_sha256") == checkpoint_digest
        and descendant.get("publication_local_ok") is True
        and descendant.get("remote_checkpoint_identity_verified") is True
        and _valid_activation_fusion_inventory(
            bootstrap_fusion.get("schema"), required
        )
        and bootstrap_fusion.get("runtime_enabled") is True
        and runtime_fusion.get("required") is True
        and runtime_fusion.get("schema") == bootstrap_fusion.get("schema")
        and runtime_fusion.get("runtime_enabled") is True
        and runtime_fusion.get("required_heads") == required
        and str(bootstrap.get("receipt_sha256") or "").startswith("sha256:")
        and str(descendant.get("commit_sha256") or "").startswith("sha256:")
        and str(registration.get("registry_sha256") or "").startswith("sha256:")
    )
    if not valid:
        return {}
    return {
        "schema": "poke_bot.final_refresh_decision_fusion_activation/v1",
        "specialist_id": "alakazam",
        "checkpoint_digest": checkpoint_digest,
        "bootstrap_checkpoint_digest": bootstrap.get(
            "bootstrap_checkpoint_sha256"
        ),
        "runtime_enabled": True,
        "training_action_eligible": True,
        "terminal_serving_eligible": False,
        "receipt": str(path),
        "receipt_digest": _file_sha256(path),
        "activation_scope": "final_refresh_committed_descendant",
        "lineage_commit": descendant.get("commit"),
        "lineage_commit_digest": descendant.get("commit_sha256"),
    }


def _source_only_design_lineage_fingerprints(
    run_dir: Path,
    current_fingerprint: str,
) -> set[str]:
    """Return fingerprints equivalent for checkpoint/fusion lineage.

    A receipt-backed recovery may update only the source-tree identity after a
    learner commit.  That changes the run design fingerprint without changing
    the checkpoint, fusion tensors, or their activation.  Accept only the
    contiguous, checksum-verified suffix of migrations whose sole changed path
    is ``source.source_tree_sha256``; any malformed or behavioral migration
    stops traversal and therefore fails closed.
    """

    allowed = {current_fingerprint}
    manifest = read_json(run_dir / "manifest.json")
    contract = manifest.get("design_contract")
    expected = str(manifest.get("design_fingerprint") or "")
    if (
        not isinstance(contract, dict)
        or not _is_sha256_digest(expected)
        or _canonical_design_digest(contract) != expected
    ):
        return allowed

    verified: list[tuple[str, str, tuple[str, ...]]] = []
    effective = contract
    for receipt_path in sorted(
        (run_dir / "design_migrations").glob("migration_*.json")
    ):
        receipt = read_json(receipt_path)
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        previous_fingerprint = str(
            receipt.get("previous_fingerprint") or ""
        )
        current_migration_fingerprint = str(
            receipt.get("current_fingerprint") or ""
        )
        changed_paths = receipt.get("changed_paths")
        if (
            int(receipt.get("schema", -1)) != 1
            or not isinstance(previous, dict)
            or not isinstance(current, dict)
            or not isinstance(changed_paths, list)
            or not all(isinstance(path, str) for path in changed_paths)
            or previous != effective
            or previous_fingerprint != expected
            or _canonical_design_digest(previous) != previous_fingerprint
            or _canonical_design_digest(current)
            != current_migration_fingerprint
        ):
            break
        verified.append(
            (
                previous_fingerprint,
                current_migration_fingerprint,
                tuple(changed_paths),
            )
        )
        effective = current
        expected = current_migration_fingerprint

    if expected != current_fingerprint:
        return allowed
    cursor = current_fingerprint
    for previous, current, changed_paths in reversed(verified):
        if current != cursor:
            break
        if changed_paths != ("source.source_tree_sha256",):
            break
        allowed.add(previous)
        cursor = previous
    return allowed


def _active_run_design_fingerprint(
    loop: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Resolve the design identity at the current committed loop boundary."""

    return str(
        loop.get("design_fingerprint")
        or manifest.get("design_fingerprint")
        or ""
    )


def _initial_learner_checkpoint_digest(
    learner: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Resolve the handoff bootstrap identity before the run-local seed copy."""

    for candidate in (
        learner.get("initial_checkpoint"),
        manifest.get("initial_learner_checkpoint"),
    ):
        if isinstance(candidate, dict):
            digest = str(candidate.get("digest") or "")
            if digest.startswith("sha256:"):
                return digest
    return str(manifest.get("checkpoint_digest") or "")


def process_rows() -> dict[int, tuple[int, float, int, str]]:
    """Return pid -> (ppid, cpu%, rss KiB, command) for one cheap snapshot."""
    raw = run(["ps", "-eo", "pid=,ppid=,pcpu=,rss=,args="], timeout=5)
    rows: dict[int, tuple[int, float, int, str]] = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            rows[int(parts[0])] = (
                int(parts[1]),
                float(parts[2]),
                int(parts[3]),
                parts[4],
            )
        except ValueError:
            continue
    return rows


def _unit_values(name: str, *, user: bool = False) -> dict[str, str]:
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(
        [
            "show",
            name,
            "--property=MainPID,ControlGroup,MemoryCurrent,TasksCurrent,Environment,EnvironmentFiles",
        ]
    )
    values: dict[str, str] = {}
    for line in run(argv, timeout=4).splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def _environment_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep:
            values[key] = value
    return values


def _environment_file_values(raw: str) -> dict[str, str]:
    """Read the literal EnvironmentFile paths reported by systemd.

    ``systemctl show`` appends metadata such as ``(ignore_errors=no)`` after
    every path.  The files are managed unit inputs, not guessed dashboard
    configuration, and provide the effective topology when ``Environment=`` is
    empty.  The live process environment is applied after this fallback.
    """

    values: dict[str, str] = {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    for token in tokens:
        if token.startswith("(") or not token.startswith("/"):
            continue
        path = Path(token)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].lstrip()
            key, sep, value = stripped.partition("=")
            if not sep or not key.strip():
                continue
            raw_value = value.strip()
            try:
                decoded = shlex.split(raw_value, comments=True)
            except ValueError:
                decoded = []
            values[key.strip()] = (
                decoded[0] if len(decoded) == 1 else raw_value.strip("'\"")
            )
    return values


def _process_environment(pid: int) -> dict[str, str]:
    """Return the running managed controller's effective environment."""

    if pid <= 0:
        return {}
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for entry in entries:
        key, sep, value = entry.partition(b"=")
        if not sep:
            continue
        try:
            values[key.decode("utf-8")] = value.decode("utf-8")
        except UnicodeError:
            continue
    return values


def _cgroup_pids(control_group: str) -> set[int]:
    if not control_group.startswith("/") or ".." in control_group.split("/"):
        return set()
    path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.procs"
    try:
        return {int(line) for line in path.read_text().splitlines() if line.strip()}
    except (OSError, ValueError):
        return set()


def curriculum_worker_state(
    active_units: list[str], active_pids: list[int]
) -> dict[str, Any]:
    """Aggregate the live user-service cgroup rather than guessing names.

    Pure-RL workers are multiprocessing children whose command line is only
    ``spawn_main``.  A process-name filter therefore misses almost the whole
    tree; systemd's control group is the authoritative membership boundary.
    """
    rows = process_rows()
    selected: set[int] = set()
    memory_current = 0
    tasks_current = 0
    environment: dict[str, str] = {}
    for unit in active_units:
        values = _unit_values(unit, user=True)
        selected.update(_cgroup_pids(values.get("ControlGroup", "")))
        memory_current += as_number(values.get("MemoryCurrent", "")) or 0
        tasks_current += as_number(values.get("TasksCurrent", "")) or 0
        environment.update(
            _environment_file_values(values.get("EnvironmentFiles", ""))
        )
        environment.update(_environment_values(values.get("Environment", "")))
        environment.update(
            _process_environment(
                as_number(values.get("MainPID", "")) or 0
            )
        )

    # Older/non-systemd test environments may not expose cgroup.procs. Fall
    # back to a complete descendant closure from the unit MainPID(s).
    if not selected:
        selected.update(pid for pid in active_pids if pid > 0)
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _cpu, _rss, _command) in rows.items():
                if ppid in selected and pid not in selected:
                    selected.add(pid)
                    changed = True

    # ``launch_pure_rl.py`` materializes the validated hardware profile into
    # the environment passed to ``train_pure_rl.py``.  Those effective
    # PURE_RL_LEAF_GPU* values therefore live on the trainer child, not
    # necessarily on systemd's launcher MainPID.  Read that child from the
    # same managed cgroup before projecting the live topology.  Without this,
    # a healthy 4/12 leaf farm is mislabeled as zero leaves / out of fleet.
    trainer_pids = sorted(
        pid
        for pid in selected
        if pid in rows and "train_pure_rl.py" in rows[pid][3]
    )
    if trainer_pids:
        environment.update(_process_environment(trainer_pids[0]))

    cpu_percent = sum(rows[pid][1] for pid in selected if pid in rows)
    rss_bytes = sum(rows[pid][2] for pid in selected if pid in rows) * 1024
    root_pid = next((pid for pid in active_pids if pid in rows), 0)
    command = rows[root_pid][3] if root_pid else ""
    workers = as_number(environment.get("PURE_RL_SIM_WORKERS", ""))
    leaves0 = as_number(environment.get("PURE_RL_LEAF_GPU0_REPLICAS", "")) or 0
    leaves1 = as_number(environment.get("PURE_RL_LEAF_GPU1_REPLICAS", "")) or 0
    multi_env = as_number(environment.get("POKEBOT_MULTI_ENV_PER_WORKER", ""))
    optimizer_runtime = {
        "awr_beta": as_float(environment.get("PURE_RL_AWR_BETA")),
        "awr_weight_max": as_float(
            environment.get("PURE_RL_AWR_WEIGHT_MAX")
        ),
    }
    try:
        command_tokens = shlex.split(command)
    except ValueError:
        command_tokens = command.split()
    for flag, key, converter in (
        (
            "--dormant-matchup-adapter-epochs",
            "dormant_matchup_adapter_epochs",
            as_number,
        ),
        (
            "--dormant-matchup-adapter-lr",
            "dormant_matchup_adapter_lr",
            as_float,
        ),
    ):
        try:
            raw_value = command_tokens[command_tokens.index(flag) + 1]
        except (ValueError, IndexError):
            continue
        value = converter(raw_value)
        if value is not None:
            optimizer_runtime[key] = value
    if "POKEBOT_MATCHUP_ADAPTER_RUNTIME" in environment:
        optimizer_runtime["matchup_adapter_runtime_required"] = (
            str(environment["POKEBOT_MATCHUP_ADAPTER_RUNTIME"])
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
    optimizer_runtime = {
        key: value for key, value in optimizer_runtime.items() if value is not None
    }
    return {
        "active": bool(active_units and selected),
        "listening": None,
        "controller_pids": list(active_pids),
        "processes": len(selected),
        "tasks": tasks_current or None,
        "workers": workers,
        "multi_env_per_worker": multi_env,
        "leaf_servers": leaves0 + leaves1,
        "leaf_gpu0_replicas": leaves0,
        "leaf_gpu1_replicas": leaves1,
        "cpu_percent": cpu_percent,
        "rss_bytes": memory_current or rss_bytes,
        "command": command or ", ".join(active_units),
        "optimizer_runtime": optimizer_runtime,
        "source": "systemd-user-cgroup",
        "topology_source": (
            "active managed trainer effective environment"
            if environment
            else "managed service process topology"
        ),
    }


def process_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def unit_state(name: str, *, user: bool = False) -> dict[str, Any]:
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(
        [
            "show",
            name,
            "--property=ActiveState,SubState,MainPID,MemoryCurrent,MemoryPeak,CPUUsageNSec,ExecMainStartTimestamp,ExecMainStartTimestampMonotonic,Result,NRestarts,ExecMainStatus,ExecStart",
        ]
    )
    raw = run(argv)
    values: dict[str, str] = {}
    for line in raw.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return {
        "name": name,
        "active": values.get("ActiveState") in {"active", "activating"},
        "active_state": values.get("ActiveState", "not-found"),
        "sub_state": values.get("SubState", "dead"),
        "pid": as_number(values.get("MainPID", "0")) or 0,
        "memory_bytes": as_number(values.get("MemoryCurrent", "")),
        "memory_peak_bytes": as_number(values.get("MemoryPeak", "")),
        "cpu_ns": as_number(values.get("CPUUsageNSec", "")),
        "started": values.get("ExecMainStartTimestamp", ""),
        "started_monotonic_us": as_number(
            values.get("ExecMainStartTimestampMonotonic", "")
        ),
        "result": values.get("Result", ""),
        "restart_count": as_number(values.get("NRestarts", "")) or 0,
        "exit_status": as_number(values.get("ExecMainStatus", "")),
        "exec_start": values.get("ExecStart", ""),
    }


def service_state() -> dict[str, Any]:
    # Specialist unit names are lineage-specific.  Discover the live
    # curriculum unit first so a current Trevenant (or later specialist)
    # service cannot be shadowed by the retired Alakazam service below.
    active_curriculum_units, _, _ = _active_curriculum_services()
    loaded_units = run(
        [
            "systemctl",
            "--user",
            "--no-legend",
            "--plain",
            "list-units",
            "--all",
            "--type=service",
        ]
    )
    dynamic_names = list(active_curriculum_units)
    for line in loaded_units.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0].lstrip("●")
        lowered = name.lower()
        if (
            "pure-rl" in lowered
            and ("staged" in lowered or "specialist" in lowered)
            and name not in dynamic_names
        ):
            dynamic_names.append(name)
    dynamic_candidates = [unit_state(name, user=True) for name in dynamic_names]
    historical_candidates = [
        unit_state(ALAKAZAM_SPECIALIST_SERVICE, user=True),
        unit_state(FINAL_FORMAT_ALAKAZAM_H10_SERVICE, user=True),
        unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True),
        unit_state(CORE_RL_SERVICE, user=True),
        unit_state(EXACT_SERVICE),
        unit_state(LATEST10_BOOTSTRAP_SERVICE),
        unit_state(SERVICE),
        unit_state(FINAL_FORMAT_ALAKAZAM_SERVICE, user=True),
    ]
    candidates = dynamic_candidates + [
        row
        for row in historical_candidates
        if row["name"] not in {candidate["name"] for candidate in dynamic_candidates}
    ]
    # RemainAfterExit bootstrap units legitimately report ActiveState=active
    # with SubState=exited and MainPID=0.  They are historical receipts, not a
    # live trainer.  Prefer an operational process and otherwise report the
    # specialist service (the current production identity) as stopped/failed.
    operational = next(
        (
            row
            for row in candidates
            if row["active"]
            and (
                int(row.get("pid") or 0) > 0
                or str(row.get("sub_state") or "") == "running"
            )
        ),
        None,
    )
    if operational is not None:
        values = operational
    else:
        recovering = [
            row
            for row in dynamic_candidates
            if str(row.get("active_state") or "") == "activating"
            or str(row.get("sub_state") or "") == "auto-restart"
        ]
        values = max(
            recovering or dynamic_candidates or candidates,
            key=lambda row: int(row.get("started_monotonic_us") or 0),
        )
    service_pid = int(values["pid"] or 0)
    raw_pid = run(
        [
            "pgrep",
            "-fo",
            "train_privileged_belief_resident.py|train_privileged_belief_shards.py|train_bootstrap.py",
        ]
    )
    trainer_pid = as_number(raw_pid) or 0
    pid = trainer_pid or service_pid
    fallback = not service_pid and bool(trainer_pid)
    command = ""
    if pid:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            ).strip()
        except OSError:
            pass
    if not command:
        command = str(values.get("exec_start") or "")
    return {
        "name": values["name"],
        "active": bool(
            fallback
            or (
                values["active"]
                and (
                    int(values.get("pid") or 0) > 0
                    or str(values.get("sub_state") or "") == "running"
                )
            )
        ),
        "active_state": "process-fallback" if fallback else values["active_state"],
        "sub_state": "running" if fallback else values["sub_state"],
        "pid": pid,
        "supervisor_pid": service_pid,
        "memory_bytes": process_rss_bytes(pid) if fallback else values["memory_bytes"],
        "memory_peak_bytes": values["memory_peak_bytes"],
        "cpu_ns": values["cpu_ns"],
        "started": values["started"],
        "command": command,
        "result": values.get("result"),
        "restart_count": values.get("restart_count"),
        "exit_status": values.get("exit_status"),
    }


def transition_state() -> dict[str, Any]:
    """Return a compact, current view of the core-to-Alakazam handoff."""
    raw = read_json(ALAKAZAM_TRANSITION_STATE)
    status = str(raw.get("status") or "waiting")
    bootstrap = unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True)
    specialist = unit_state(ALAKAZAM_SPECIALIST_SERVICE, user=True)
    core = unit_state(CORE_RL_SERVICE, user=True)
    labels = {
        "training_alakazam_expert_bootstrap_blackwell_device_resident": (
            "Deck Agnostic Core → Alakazam · expert bootstrap on Blackwell"
        ),
        "alakazam_specialist_bootstrap_ready_launching": (
            "Alakazam bootstrap ready · launching specialist RL"
        ),
        "launching_alakazam_specialist": "Launching Alakazam specialist RL fleet",
        "complete": "Alakazam specialist RL · transition complete",
    }
    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    best = decision.get("best") if isinstance(decision.get("best"), dict) else {}
    triggered = raw.get("triggered") is True or decision.get("triggered") is True
    updated = None
    try:
        updated = ALAKAZAM_TRANSITION_STATE.stat().st_mtime
    except OSError:
        pass
    bootstrap_running = bool(
        bootstrap.get("active")
        and (
            int(bootstrap.get("pid") or 0) > 0
            or bootstrap.get("sub_state") == "running"
        )
    )
    active = bool(
        bootstrap_running
        or (
            triggered
            and status not in {"complete", "specialist_preparation_failed_core_continues"}
        )
    )
    return {
        "available": bool(raw),
        "active": active,
        "triggered": triggered,
        "status": status,
        "label": labels.get(status, status.replace("_", " ").strip().title()),
        "reason": decision.get("reason") or raw.get("handoff_wait_reason"),
        "source": str(ALAKAZAM_TRANSITION_STATE),
        "updated_at": updated,
        "core_iteration": best.get("iteration"),
        "core_win_rate": best.get("win_rate"),
        "bootstrap": bootstrap,
        "specialist": specialist,
        "core": core,
    }


def read_tail(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def owner_specialist_handoff_state() -> dict[str, Any]:
    """Expose the owner-accepted Alakazam → core → Trevenant handoff."""

    service = unit_state(OWNER_SPECIALIST_HANDOFF_SERVICE, user=True)
    handoff = read_json(OWNER_SPECIALIST_HANDOFF_STATE)
    core = read_json(OWNER_CORE_DISTILL_STATE)
    trevenant = read_json(OWNER_TREVENANT_BOOTSTRAP_STATE)
    raw_tail = ANSI_RE.sub("", read_tail(OWNER_SPECIALIST_HANDOFF_LOG, 256_000))
    lines = [
        line.strip()
        for line in raw_tail.replace("\r", "\n").splitlines()
        if line.strip()
    ]
    latest_line = lines[-1] if lines else None
    active = bool(service.get("active") and int(service.get("pid") or 0) > 0)
    phase = str(handoff.get("phase") or "waiting")
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    epoch: int | None = None
    epochs_target: int | None = None
    rate: float | None = None
    rate_unit: str | None = None

    pack_line = next(
        (line for line in reversed(lines) if "pack Blackwell corpus:" in line),
        None,
    )
    pack_match = (
        re.search(
            r"pack Blackwell corpus:\s*(\d+)%.*?\|\s*(\d+)/(\d+)"
            r".*?([\d.]+)game/s",
            pack_line,
        )
        if pack_line
        else None
    )
    if phase in {"owner_acceptance_verified", "protected_corpora_verified"}:
        stage = "core_corpus_pack"
        if pack_match:
            percent = float(pack_match.group(1))
            current = int(pack_match.group(2))
            total = int(pack_match.group(3))
            rate = float(pack_match.group(4))
            rate_unit = "game/s"
            latest_line = pack_line
    elif phase == "distilled_core_frozen":
        stage = "trevenant_expert_bootstrap"
    elif phase in {
        "next_specialist_bootstrap_frozen",
        "next_specialist_rl_armed",
        "next_specialist_rl_started",
    }:
        stage = phase
        if phase == "next_specialist_rl_started":
            latest_line = "Hop's Trevenant specialist RL is active."
    else:
        stage = phase

    core_history = core.get("history") if isinstance(core.get("history"), list) else []
    trevenant_history = (
        trevenant.get("history")
        if isinstance(trevenant.get("history"), list)
        else []
    )
    if core_history and phase in {
        "owner_acceptance_verified",
        "protected_corpora_verified",
    }:
        stage = "deck_agnostic_core_distillation"
        epoch = int((core_history[-1] or {}).get("epoch") or len(core_history))
        epochs_target = int(core.get("epochs_max") or 25)
        current = epoch
        total = epochs_target
        percent = 100.0 * epoch / max(1, epochs_target)
        latest_line = next(
            (
                line
                for line in reversed(lines)
                if line.startswith("[core-distill] epoch=")
            ),
            latest_line,
        )
        batch_line = next(
            (
                line
                for line in reversed(lines)
                if "expert rehearsal before iter" in line
                and "batch/s" in line
            ),
            None,
        )
        batch_match = (
            re.search(
                r"expert rehearsal before iter(\d+)\s+ep1/1:\s*(\d+)%"
                r".*?\|\s*(\d+)/(\d+).*?([\d.]+)batch/s",
                batch_line,
            )
            if batch_line
            else None
        )
        if batch_match:
            epoch = int(batch_match.group(1))
            batch_percent = float(batch_match.group(2))
            current = int(batch_match.group(3))
            total = int(batch_match.group(4))
            percent = 100.0 * (
                (epoch - 1) + batch_percent / 100.0
            ) / max(1, epochs_target)
            rate = float(batch_match.group(5))
            rate_unit = "batch/s"
            latest_line = batch_line
    elif trevenant_history and phase != "next_specialist_rl_started":
        stage = "trevenant_expert_bootstrap"
        epoch = int(
            (trevenant_history[-1] or {}).get("epoch") or len(trevenant_history)
        )
        epochs_target = int(trevenant.get("epochs_max") or 25)
        current = epoch
        total = epochs_target
        percent = 100.0 * epoch / max(1, epochs_target)

    return {
        "available": bool(handoff or service.get("load_state") == "loaded"),
        "active": active,
        "phase": phase,
        "stage": stage,
        "label": "Alakazam frozen → deck-agnostic core → Hop's Trevenant",
        "pid": service.get("pid"),
        "memory_bytes": service.get("memory_bytes"),
        "latest_line": latest_line,
        "current": current,
        "total": total,
        "percent": percent,
        "epoch": epoch,
        "epochs_target": epochs_target,
        "rate": rate,
        "rate_unit": rate_unit,
        "source": str(OWNER_SPECIALIST_HANDOFF_STATE),
        "log": str(OWNER_SPECIALIST_HANDOFF_LOG),
        "updated_at": max(
            (
                path.stat().st_mtime
                for path in (
                    OWNER_SPECIALIST_HANDOFF_STATE,
                    OWNER_CORE_DISTILL_STATE,
                    OWNER_TREVENANT_BOOTSTRAP_STATE,
                    OWNER_SPECIALIST_HANDOFF_LOG,
                )
                if path.is_file()
            ),
            default=None,
        ),
        "service": service,
    }


def post_starmie_specialist_handoff_state() -> dict[str, Any]:
    """Expose the current reusable specialist-cycle handoff.

    The historical post-Starmie unit remains a fallback for old snapshots.
    Once the reusable cycle service is active, its newest source-bound state
    and log are authoritative.
    """

    cycle_service = unit_state(SPECIALIST_CYCLE_HANDOFF_SERVICE, user=True)
    cycle_active = bool(
        cycle_service.get("name") == SPECIALIST_CYCLE_HANDOFF_SERVICE
        and
        (
            cycle_service.get("active")
            or str(cycle_service.get("active_state") or "") == "activating"
            or str(cycle_service.get("sub_state") or "") == "auto-restart"
        )
    )
    transition_graph_path = SPECIALIST_TRANSITION_GRAPH_STATE
    try:
        transition_graph_path.relative_to(ROOT)
    except ValueError:
        # Tests and alternate runtime roots may override ROOT without
        # rebuilding every derived constant. Do not leak a real host's
        # transition receipt into that isolated projection.
        transition_graph_path = (
            ROOT / "outputs/state/specialist-transition-graph.json"
        )
    transition_graph = read_json(transition_graph_path)
    transition_rows = [
        dict(row)
        for row in (transition_graph.get("transitions") or {}).values()
        if isinstance(row, dict)
    ]

    def transition_activity(row: dict[str, Any]) -> str:
        timestamps: list[str] = []
        for receipt in (row.get("receipts") or {}).values():
            if not isinstance(receipt, dict):
                continue
            for key in ("completed_at", "failed_at", "started_at", "updated_at"):
                value = str(receipt.get(key) or "").strip()
                if value:
                    timestamps.append(value)
        return max(timestamps, default="")

    # Atomic JSON is key-sorted, so digest-map insertion order is not
    # chronological. Bind live telemetry to the newest receipt instead.
    current_transition = (
        max(transition_rows, key=transition_activity)
        if transition_rows
        else {}
    )
    transition_source_id = str(
        current_transition.get("active_specialist") or ""
    ).strip()
    cycle_states = sorted(
        (
            path
            for path in (ROOT / "outputs/state").glob(
                "post-*-core-v*-handoff.json"
            )
            if "-cumulative-core-" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    latest_any_cycle_state_path = cycle_states[0] if cycle_states else None
    latest_any_cycle_state = (
        read_json(latest_any_cycle_state_path)
        if latest_any_cycle_state_path is not None
        else {}
    )
    latest_any_cycle_settled = bool(
        latest_any_cycle_state.get("phase")
        in {
            "next_specialist_selected",
            "next_specialist_bootstrap_frozen",
            "next_specialist_rl_armed",
            "next_specialist_rl_started",
        }
        and str(
            (latest_any_cycle_state.get("source") or {}).get(
                "specialist_id"
            )
            or ""
        ).strip()
        and (
            (
                latest_any_cycle_state.get("selection") or {}
            ).get("selected")
            or latest_any_cycle_state.get("next_specialist")
        )
        and (
            not POST_STARMIE_HANDOFF_STATE.is_file()
            or (
                latest_any_cycle_state_path is not None
                and latest_any_cycle_state_path.stat().st_mtime
                > POST_STARMIE_HANDOFF_STATE.stat().st_mtime
            )
        )
    )
    matching_cycle_states = (
        [
            path
            for path in cycle_states
            if (
                str(
                    (read_json(path).get("source") or {}).get("specialist_id")
                    or ""
                ).strip()
                == transition_source_id
            )
        ]
        if transition_source_id
        else cycle_states
    )
    if latest_any_cycle_settled and latest_any_cycle_state_path is not None:
        # A completed immutable cycle receipt is newer execution truth than a
        # stale transition-graph pointer. Keep the graph as recovery metadata,
        # but do not let it force telemetry back to an older specialist.
        matching_cycle_states = [latest_any_cycle_state_path]
    cycle_current = bool(current_transition and matching_cycle_states)
    latest_cycle_state = (
        read_json(matching_cycle_states[0])
        if matching_cycle_states
        else {}
    )
    settled_cycle_current = bool(
        latest_cycle_state.get("phase")
        in {
            "next_specialist_selected",
            "next_specialist_bootstrap_frozen",
            "next_specialist_rl_armed",
            "next_specialist_rl_started",
        }
        and str(
            (latest_cycle_state.get("source") or {}).get("specialist_id")
            or ""
        ).strip()
        and (
            (
                latest_cycle_state.get("selection") or {}
            ).get("selected")
            or latest_cycle_state.get("next_specialist")
        )
        and (
            not POST_STARMIE_HANDOFF_STATE.is_file()
            or matching_cycle_states[0].stat().st_mtime
            > POST_STARMIE_HANDOFF_STATE.stat().st_mtime
        )
    )
    if (
        cycle_active
        or cycle_current
        or settled_cycle_current
    ) and matching_cycle_states:
        service = cycle_service
        state_path = matching_cycle_states[0]
        log_path = SPECIALIST_CYCLE_HANDOFF_LOG
    elif cycle_active and current_transition:
        service = cycle_service
        state_path = transition_graph_path
        log_path = SPECIALIST_CYCLE_HANDOFF_LOG
    else:
        service = unit_state(POST_STARMIE_HANDOFF_SERVICE, user=True)
        state_path = POST_STARMIE_HANDOFF_STATE
        log_path = POST_STARMIE_HANDOFF_LOG
    state = read_json(state_path)
    raw_tail = ANSI_RE.sub(
        "", read_tail(log_path, 512_000)
    ).replace("\r", "\n")
    lines = [line.strip() for line in raw_tail.splitlines() if line.strip()]
    latest_line = lines[-1] if lines else None
    active = bool(
        service.get("active")
        or str(service.get("active_state") or "") == "activating"
        or str(service.get("sub_state") or "") == "auto-restart"
    )
    phase = str(state.get("phase") or "waiting_for_starmie_gate")
    source = dict(state.get("source") or {})
    source_id = str(
        source.get("specialist_id")
        or current_transition.get("active_specialist")
        or ""
    ).strip()
    if not source_id:
        matched = re.match(r"post-(.+)-core-v\d+-handoff\.json", state_path.name)
        source_id = matched.group(1) if matched else "starmie"
    source_label = source_id.replace("-", " ").title()
    if cycle_active and phase == "starmie_pass_verified":
        phase = "source_specialist_verified"
    version_match = re.match(
        rf"post-{re.escape(source_id)}-core-v(\d+)-handoff\.json",
        state_path.name,
    )
    versioned_contract_path = (
        ROOT
        / "outputs/state"
        / f"post-{source_id}-cumulative-core-v{version_match.group(1)}-handoff.json"
        if version_match
        else None
    )
    legacy_contract_path = (
        ROOT / "outputs/state" / f"post-{source_id}-cumulative-core-handoff.json"
    )
    cumulative_contract_path = (
        versioned_contract_path
        if versioned_contract_path is not None
        and versioned_contract_path.is_file()
        else legacy_contract_path
    )
    cumulative_contract = read_json(cumulative_contract_path)
    core_refresh = dict(cumulative_contract.get("core_refresh") or {})
    configured_epochs = int(core_refresh.get("max_epochs") or 25)
    # The cycle log is append-only across core versions. Limit live progress
    # parsing to the newest core-refresh load marker so a newly started v4
    # cannot inherit the completed v3 epoch-25 tqdm frame.
    progress_lines = lines
    if cycle_active and phase == "source_specialist_verified":
        load_indexes = [
            index
            for index, line in enumerate(lines)
            if line.startswith(
                "[core-refresh] loading protected balanced corpus"
            )
        ]
        if load_indexes:
            progress_lines = lines[load_indexes[-1] :]
            latest_line = progress_lines[-1]
    frozen_registry = read_json(FROZEN_SPECIALIST_REGISTRY)
    frozen_rows = [
        dict(row)
        for row in (frozen_registry.get("specialists") or [])
        if isinstance(row, dict)
    ]
    frozen_ids = {str(row.get("specialist_id") or "") for row in frozen_rows}
    frozen_count = len(frozen_rows)
    completed_count = frozen_count + int(
        cycle_active and source_id not in frozen_ids
    )
    try:
        import yaml  # type: ignore[import-not-found]

        program_state = yaml.safe_load(
            (ROOT / "state/specialists.yaml").read_text(encoding="utf-8")
        )
    except (ImportError, OSError, UnicodeError, ValueError, TypeError):
        program_state = {}
    required_specialist_count = int(
        (
            ((program_state or {}).get("target_registry") or {}).get(
                "required_target_count"
            )
            or 0
        )
    )
    if required_specialist_count <= 0:
        required_specialist_count = len(
            (program_state or {}).get("specialists") or []
        )
    if required_specialist_count <= 0:
        # Isolated dashboard fixtures and older deployments may not carry the
        # program-state YAML. The route roster is a compatibility fallback
        # only; the canonical training plan still wins whenever it is present.
        canonical_roster = read_json(CANONICAL_MATCHUP_ADAPTER_ROSTER)
        required_specialist_count = int(
            canonical_roster.get("required_specialist_count") or 0
        )
        if required_specialist_count <= 0:
            required_specialist_count = len(
                canonical_roster.get("expert_ids") or []
            )
    remaining_count = max(0, required_specialist_count - completed_count)
    selected_value = (
        state.get("selected_specialist")
        or (state.get("selection") or {}).get("selected")
    )
    selected = (
        dict(selected_value)
        if isinstance(selected_value, dict)
        else {"id": str(selected_value or "")}
    )
    next_specialist_id = str(
        selected.get("specialist_id") or selected.get("id") or ""
    ).strip()
    prestage = read_json(NEXT_SPECIALIST_PRESTAGE_STATE)
    prestage_selected_value = prestage.get("selected_specialist")
    prestage_selected = (
        dict(prestage_selected_value)
        if isinstance(prestage_selected_value, dict)
        else {"id": str(prestage_selected_value or "")}
    )
    next_specialist_id = (
        next_specialist_id
        or str(
            prestage_selected.get("specialist_id")
            or prestage_selected.get("id")
            or ""
        ).strip()
    )
    stage = phase
    current: int | None = None
    total: int | None = None
    percent: float | None = None
    epoch: int | None = None
    epochs_target: int | None = None
    rate: float | None = None
    rate_unit: str | None = None
    v6_sync = read_json(V6_STRATEGIC_SPECIALIST_SYNC_STATE)
    v6_sync_status = str(v6_sync.get("status") or "")
    v6_syncing = bool(
        cycle_active
        and not V6_STRATEGIC_SPECIALIST_CURRENT.exists()
        and v6_sync_status.startswith("syncing")
    )
    core_regression = dict(state.get("core_gameplay_regression") or {})
    core_regression_results = [
        dict(row)
        for row in (core_regression.get("results") or [])
        if isinstance(row, dict)
    ]
    core_regression_criteria = dict(core_regression.get("criteria") or {})
    core_regression_threshold = float(
        core_regression_criteria.get("per_teacher_raw_win_rate_minimum")
        or 0.0
    )
    failed_core_teachers = [
        {
            "specialist_id": str(row.get("specialist_id") or ""),
            "games": int((row.get("report") or {}).get("games") or 0),
            "win_rate": float((row.get("report") or {}).get("wr") or 0.0),
        }
        for row in core_regression_results
        if float((row.get("report") or {}).get("wr") or 0.0)
        < core_regression_threshold
    ]
    core_regression_failed = bool(
        phase == "core_gameplay_regression_complete"
        and core_regression.get("schema")
        == "poke_bot.multi_teacher_core_gameplay_regression/v1"
        and core_regression.get("passed") is False
        and core_regression_criteria.get("all_reports_valid") is True
    )
    if v6_syncing:
        phase = "waiting_for_v6_corpus_sync"
        stage = "atomic_checksum_sync_to_inzi"
        current = int(v6_sync.get("copied_bytes") or 0)
        total = int(v6_sync.get("source_bytes") or 0)
        percent = float(v6_sync.get("percent") or 0.0)
        rate = float(v6_sync.get("bandwidth_limit_kib_per_second") or 0.0)
        rate_unit = "KiB/s"
        latest_line = (
            "Frozen source checkpoint is safe; checksum-bound Matchup Router Format 6 corpus sync "
            f"is {percent:.1f}% complete before the next specialist handoff."
        )
    elif core_regression_failed:
        phase = "core_gameplay_regression_failed"
        stage = "deck_agnostic_cumulative_core_gate_failed"
        current = sum(
            int((row.get("report") or {}).get("games") or 0)
            for row in core_regression_results
        )
        total = current
        percent = 100.0
        aggregate = float(
            core_regression_criteria.get("aggregate_raw_win_rate") or 0.0
        )
        failed_summary = ", ".join(
            f"{row['specialist_id']} {100.0 * row['win_rate']:.2f}%"
            for row in failed_core_teachers
        ) or "unknown matchup"
        latest_line = (
            "Cumulative core gameplay gate failed closed: "
            f"aggregate {100.0 * aggregate:.2f}%; below "
            f"{100.0 * core_regression_threshold:.2f}% raw floor: "
            f"{failed_summary}. No successor bootstrap was started."
        )

    pack_line = next(
        (
            line
            for line in reversed(progress_lines)
            if "pack Blackwell corpus:" in line
        ),
        None,
    )
    pack_match = (
        re.search(
            r"pack Blackwell corpus:\s*(\d+)%.*?\|\s*(\d+)/(\d+)"
            r".*?([\d.]+)game/s",
            pack_line,
        )
        if pack_line
        else None
    )
    epoch_line = next(
        (
            line
            for line in reversed(progress_lines)
            if line.startswith("[core-refresh] epoch=")
        ),
        None,
    )
    epoch_match = (
        re.search(r"\[core-refresh\] epoch=(\d+)/(\d+)", epoch_line)
        if epoch_line
        else None
    )
    train_line = next(
        (
            line
            for line in reversed(progress_lines)
            if "expert rehearsal before iter" in line and "batch/s" in line
        ),
        None,
    )
    train_match = (
        re.search(
            r"expert rehearsal before iter(\d+)\s+ep1/1:\s*(\d+)%"
            r".*?\|\s*(\d+)/(\d+).*?([\d.]+)batch/s",
            train_line,
        )
        if train_line
        else None
    )
    if v6_syncing or core_regression_failed:
        pass
    elif train_match:
        if phase == "next_specialist_selected":
            stage = (
                "next_specialist_expert_bootstrap_training"
                if cycle_active
                else "lucario_expert_bootstrap_training"
            )
        else:
            stage = (
                "deck_agnostic_cumulative_core_training"
                if cycle_active
                else "deck_agnostic_core_v2_training"
            )
        epoch = int(train_match.group(1))
        epochs_target = configured_epochs
        batch_percent = float(train_match.group(2))
        current = int(train_match.group(3))
        total = int(train_match.group(4))
        percent = 100.0 * (
            (epoch - 1) + batch_percent / 100.0
        ) / epochs_target
        rate = float(train_match.group(5))
        rate_unit = "batch/s"
        latest_line = train_line
    elif epoch_match:
        stage = (
            "deck_agnostic_cumulative_core_training"
            if cycle_active
            else "deck_agnostic_core_v2_training"
        )
        epoch = int(epoch_match.group(1))
        epochs_target = int(epoch_match.group(2))
        current = epoch
        total = epochs_target
        percent = 100.0 * epoch / max(1, epochs_target)
        latest_line = epoch_line
    elif pack_match:
        stage = (
            "deck_agnostic_cumulative_core_corpus_pack"
            if cycle_active
            else "deck_agnostic_core_v2_corpus_pack"
        )
        percent = float(pack_match.group(1))
        current = int(pack_match.group(2))
        total = int(pack_match.group(3))
        rate = float(pack_match.group(4))
        rate_unit = "game/s"
        latest_line = pack_line
    elif active and phase in {
        "starmie_pass_verified",
        "source_specialist_verified",
    }:
        packing_cpu_corpus = any(
            line.startswith("[expert-cpu-pack]")
            for line in progress_lines
        )
        stage = (
            (
                "deck_agnostic_cumulative_core_corpus_pack"
                if packing_cpu_corpus
                else "deck_agnostic_cumulative_core_refresh"
            )
            if cycle_active
            else "deck_agnostic_core_v2_refresh"
        )
        latest_line = latest_line or (
            f"{source_label} is frozen; refreshing the shared core before the "
            f"next specialist. {remaining_count} specialists remain."
        )

    generated_handoff_value = str(
        (cumulative_contract.get("next_specialist") or {}).get(
            "generated_handoff_contract"
        )
        or ""
    ).strip()
    generated_handoff = (
        read_json(Path(generated_handoff_value).expanduser().resolve())
        if generated_handoff_value
        else {}
    )
    staged_expanded_heads = staged_expanded_head_training_state(
        state,
        generated_handoff,
        cumulative_core_contract=cumulative_contract,
    )

    return {
        "available": bool(
            state or service.get("load_state") == "loaded"
        ),
        "active": active,
        "phase": phase,
        "stage": stage,
        "label": (
            (
                f"{source_label} frozen → cumulative shared core → "
                "next unfinished specialist"
            )
            if cycle_active or settled_cycle_current
            else (
                "Starmie frozen → shared core v2 → specialist "
                f"{completed_count + 1} of {required_specialist_count}"
            )
        ),
        "pid": service.get("pid"),
        "memory_bytes": service.get("memory_bytes"),
        "latest_line": latest_line,
        "current": current,
        "total": total,
        "percent": percent,
        "epoch": epoch,
        "epochs_target": epochs_target,
        "rate": rate,
        "rate_unit": rate_unit,
        "source": str(state_path),
        "log": str(log_path),
        "updated_at": max(
            (
                path.stat().st_mtime
                for path in (
                    state_path,
                    log_path,
                    cumulative_contract_path,
                )
                if path.is_file()
            ),
            default=None,
        ),
        "service": service,
        "completed_specialists_after_starmie": completed_count,
        "remaining_specialists_after_starmie": remaining_count,
        "program_complete": False,
        "population_transition_ready": False,
        "source_specialist_id": source_id,
        "next_specialist_id": next_specialist_id or None,
        "staged_expanded_head_training": staged_expanded_heads,
        "terminal_failure": core_regression_failed,
        "core_gameplay_regression": (
            {
                "passed": False,
                "aggregate_raw_win_rate": float(
                    core_regression_criteria.get("aggregate_raw_win_rate")
                    or 0.0
                ),
                "aggregate_raw_win_rate_minimum": float(
                    core_regression_criteria.get(
                        "aggregate_raw_win_rate_minimum"
                    )
                    or 0.0
                ),
                "per_teacher_raw_win_rate_minimum": (
                    core_regression_threshold
                ),
                "games": current,
                "failed_teachers": failed_core_teachers,
            }
            if core_regression_failed
            else None
        ),
    }


def reconcile_current_specialist_handoff(
    handoff: dict[str, Any],
    *,
    active_specialist: str,
    program_progress: dict[str, Any] | None = None,
    next_specialist: str | None = None,
    active_runtime_refresh: bool = False,
) -> dict[str, Any]:
    """Do not present a successful historical handoff as the current one."""

    active_specialist = str(active_specialist or "").strip().lower()
    phase = str(handoff.get("phase") or "").strip()
    settled_successor_phases = {
        "next_specialist_selected",
        "next_specialist_bootstrap_frozen",
        "next_specialist_rl_armed",
        "next_specialist_rl_started",
    }
    settled_current_transition = bool(
        not active_specialist
        and phase in settled_successor_phases
        and str(
            handoff.get("source_specialist_id")
            or (handoff.get("source_specialist") or {}).get("specialist_id")
            or ""
        ).strip()
        and (
            str(handoff.get("next_specialist_id") or "").strip()
            or str(next_specialist or "").strip()
        )
    )
    if (
        handoff.get("active") is True
        or handoff.get("terminal_failure") is True
        or active_specialist == "hops-trevenant"
        or settled_current_transition
    ):
        result = dict(handoff)
        if not result.get("next_specialist_id"):
            result["next_specialist_id"] = (
                str(next_specialist or "").strip() or None
            )
        if "completed_specialists" not in result:
            result["completed_specialists"] = int(
                result.get("completed_specialists_after_starmie") or 0
            )
        if "remaining_specialists_after_active" not in result:
            result["remaining_specialists_after_active"] = int(
                result.get("remaining_specialists_after_starmie") or 0
            )
        if settled_current_transition:
            result["transition_current"] = True
            result["source_specialist_id"] = str(
                result.get("source_specialist_id")
                or (result.get("source_specialist") or {}).get("specialist_id")
                or ""
            ).strip()
            result["latest_line"] = (
                f"{result['source_specialist_id'].replace('-', ' ').title()} "
                f"is frozen; {str(result['next_specialist_id']).replace('-', ' ').title()} "
                "is selected and remains fail-closed until its readiness "
                "receipts pass."
            )
        return result
    progress = dict(program_progress or {})
    remaining = int(progress.get("remaining_after_active") or 0)
    label_name = active_specialist.replace("-", " ").title() or "Active specialist"
    historical_source = handoff.get("source_specialist_id")
    historical_next = handoff.get("next_specialist_id")
    if active_runtime_refresh and active_specialist == "marnie-s-grimmsnarl-ex":
        return {
            **handoff,
            "available": True,
            "active": False,
            "phase": "waiting_for_active_specialist_gate",
            "stage": "waiting_for_active_specialist_gate",
            "label": "Marnie's Grimmsnarl ex refresh → H10 Crustle",
            "latest_line": (
                "Marnie's Grimmsnarl ex refresh is training through exact "
                "iteration 20. Freeze and register iter_00020 without "
                "collecting iter_00021; then begin the staged new H10 "
                "Crustle specialist."
            ),
            "current": None,
            "total": None,
            "percent": None,
            "epoch": None,
            "epochs_target": None,
            "rate": None,
            "rate_unit": None,
            "completed_specialists": int(progress.get("completed_frozen") or 0),
            "remaining_specialists_after_active": 1,
            "source_specialist_id": active_specialist,
            "next_specialist_id": "crustle",
            "refresh_terminal_iteration": 20,
            "next_collection_forbidden": 21,
            "historical_source_specialist_id": historical_source,
            "historical_next_specialist_id": historical_next,
            "historical_source_suppressed": True,
        }
    return {
        **handoff,
        "available": True,
        "active": False,
        "phase": "waiting_for_active_specialist_gate",
        "stage": "waiting_for_active_specialist_gate",
        "label": f"{label_name} → next unfinished specialist",
        "latest_line": (
            f"{label_name} is training; {remaining} specialists remain after it. "
            "The next handoff begins only after its exact gate pass."
        ),
        "current": None,
        "total": None,
        "percent": None,
        "epoch": None,
        "epochs_target": None,
        "rate": None,
        "rate_unit": None,
        "completed_specialists": int(progress.get("completed_frozen") or 0),
        "remaining_specialists_after_active": remaining,
        "source_specialist_id": active_specialist or None,
        "next_specialist_id": str(next_specialist or "").strip() or None,
        "historical_source_specialist_id": historical_source,
        "historical_next_specialist_id": historical_next,
        "historical_source_suppressed": True,
    }


def reconcile_protocol_with_active_handoff(
    protocol: dict[str, Any],
    handoff: dict[str, Any],
) -> dict[str, Any]:
    """Let a live boundary handoff supersede stale mutable protocol wording."""

    if protocol.get("available") is not True or handoff.get("active") is not True:
        return protocol
    label = str(handoff.get("label") or "specialist boundary handoff")
    stage = str(handoff.get("stage") or handoff.get("phase") or "running")
    epoch = handoff.get("epoch")
    target = handoff.get("epochs_target")
    progress = (
        f" Epoch {int(epoch)}/{int(target)}."
        if isinstance(epoch, int)
        and isinstance(target, int)
        and epoch > 0
        and target > 0
        else ""
    )
    specialist_bootstrap = stage in {
        "lucario_expert_bootstrap_training",
        "next_specialist_expert_bootstrap_training",
    }
    next_specialist_id = str(handoff.get("next_specialist_id") or "")
    bootstrap_specialist_id = (
        next_specialist_id
        or ("lucario" if stage == "lucario_expert_bootstrap_training" else "")
    )
    bootstrap_specialist_name = (
        bootstrap_specialist_id.replace("-", " ").title()
        or "selected specialist"
    )
    reconciled_progress = dict(protocol.get("program_progress") or {})
    completed = int(
        handoff.get("completed_specialists_after_starmie")
        or reconciled_progress.get("completed_frozen")
        or 0
    )
    required_total = int(
        reconciled_progress.get("required_specialists_total") or 0
    )
    if completed:
        completed_ids = [
            str(value)
            for value in reconciled_progress.get("completed_specialist_ids") or []
            if str(value)
        ]
        source_id = str(handoff.get("source_specialist_id") or "")
        if source_id and source_id not in completed_ids:
            completed_ids.append(source_id)
        reconciled_progress.update(
            {
                "completed_frozen": completed,
                "completed_specialist_ids": completed_ids,
                "remaining_after_active": max(
                    0,
                    required_total
                    - completed
                    - int(specialist_bootstrap),
                ),
                "active_specialists": int(specialist_bootstrap),
                "active_specialist_ids": (
                    [next_specialist_id]
                    if specialist_bootstrap and next_specialist_id
                    else []
                ),
            }
        )
    return {
        **protocol,
        "phase": (
            "specialist_bootstrap"
            if specialist_bootstrap
            else "shared_core_derivation"
        ),
        "active_specialist": (
            next_specialist_id or "lucario"
            if specialist_bootstrap
            else ""
        ),
        "program_progress": reconciled_progress,
        "shared_core_status": (
            "validated" if specialist_bootstrap else "refreshing"
        ),
        "handoff_reconciled": True,
        "canonical_pointer_stale": bool(
            protocol.get("canonical_active_specialist")
        ),
        "accuracy_warning": (
            "The prior specialist is frozen. The live managed handoff "
            "supersedes the mutable active-specialist pointer."
        ),
        "next_action": (
            f"Continue the active {label} at stage "
            f"{stage.replace('_', ' ')}.{progress} "
            + (
                f"When it completes, freeze the exact 25-epoch "
                f"{bootstrap_specialist_name} bootstrap and start "
                f"{bootstrap_specialist_name} curriculum RL."
                if specialist_bootstrap
                else (
                    "When it completes, validate the refreshed shared core, "
                    "materialize every frozen predecessor into the next S+ "
                    "gate, and start the selected unfinished specialist."
                )
            )
        ),
    }


def checkpoint_parameter_telemetry(log_path: Path) -> dict[str, Any]:
    """Return the latest parameter count produced by an actual checkpoint load."""
    raw = ANSI_RE.sub("", read_tail(log_path, 512_000)).replace("\r", "\n")
    matches = list(
        re.finditer(
            r"\[pure_rl\] loaded checkpoint params=(\d+) path=(\S+)",
            raw,
        )
    )
    if not matches:
        return {}
    latest = matches[-1]
    count = int(latest.group(1))
    if count <= 0:
        return {}
    return {
        "trainable_parameters": count,
        "checkpoint": latest.group(2),
        "source": str(log_path),
    }


def _checkpoint_adapter_registry_contract(
    adapter_config: dict[str, Any],
    expert_indexes: set[int],
) -> dict[str, Any]:
    """Validate legacy dense rosters and Router Format 6 slot registries."""

    adapter_format = str(adapter_config.get("format") or "")
    expert_count = max(expert_indexes) + 1 if expert_indexes else 0
    tensor_slots_contiguous = expert_indexes == set(range(expert_count))
    if adapter_format != "poke-bot-matchup-adapter-bank-v6":
        expert_ids = [
            str(value)
            for value in (adapter_config.get("expert_ids") or [])
            if str(value)
        ]
        verified = bool(
            expert_count > 0
            and tensor_slots_contiguous
            and len(expert_ids) == expert_count
            and len(set(expert_ids)) == expert_count
        )
        return {
            "verified": verified,
            "format": adapter_format or None,
            "physical_slot_capacity": expert_count,
            "routable_expert_ids": expert_ids,
            "slot_registry_digest": None,
            "reason": (
                None
                if verified
                else "checkpoint adapter tensors and embedded route registry disagree"
            ),
        }

    registry = adapter_config.get("slot_registry")
    registry = dict(registry) if isinstance(registry, dict) else {}
    slots = list(registry.get("slots") or [])
    allowed_statuses = {"active", "dormant", "retired", "unused"}
    routable_statuses = {"active", "dormant"}
    allocated_ids: list[str] = []
    routable_ids: list[str] = []
    slot_rows_valid = len(slots) == expert_count
    for expected_slot, row in enumerate(slots):
        if not isinstance(row, dict):
            slot_rows_valid = False
            continue
        status = str(row.get("status") or "")
        identity = row.get("archetype_id")
        if row.get("slot") != expected_slot or status not in allowed_statuses:
            slot_rows_valid = False
        if status == "unused":
            if identity is not None:
                slot_rows_valid = False
            continue
        if not isinstance(identity, str) or not identity:
            slot_rows_valid = False
            continue
        allocated_ids.append(identity)
        if status in routable_statuses:
            routable_ids.append(identity)
    configured_routable_ids = [
        str(value)
        for value in (registry.get("active_expert_ids") or [])
        if str(value)
    ]
    compatibility_ids = [
        str(value)
        for value in (registry.get("expert_ids") or [])
        if str(value)
    ]
    embedded_digest = str(adapter_config.get("slot_registry_digest") or "")
    registry_digest = _canonical_json_digest(registry) if registry else ""
    verified = bool(
        expert_count == 64
        and tensor_slots_contiguous
        and int(adapter_config.get("slot_capacity") or 0) == expert_count
        and registry.get("schema") == "poke_bot.matchup_adapter_roster/v1"
        and registry.get("slot_schema")
        == "poke_bot.matchup_adapter_slot_registry/v1"
        and registry.get("checkpoint_format") == adapter_format
        and int(registry.get("slot_capacity") or 0) == expert_count
        and slot_rows_valid
        and len(allocated_ids) == len(set(allocated_ids))
        and routable_ids
        and routable_ids == configured_routable_ids
        and routable_ids == compatibility_ids
        and _is_sha256_digest(embedded_digest)
        and embedded_digest == registry_digest
    )
    return {
        "verified": verified,
        "format": adapter_format,
        "physical_slot_capacity": expert_count,
        "routable_expert_ids": routable_ids,
        "slot_registry_digest": embedded_digest or None,
        "reason": (
            None
            if verified
            else "Matchup Router Format 6 tensors and embedded slot registry disagree"
        ),
    }


def checkpoint_structure_telemetry(
    checkpoint_path: Path | str,
    checkpoint_digest: str,
    *,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect the committed checkpoint instead of inferring its shape from logs.

    A startup log describes the seed as it existed at load time.  Specialist
    handoffs can subsequently materialize a wider adapter bank, so that log is
    not authoritative for the active committed learner.  Cache the small
    structural summary by immutable digest and file identity; the checkpoint is
    loaded only once per newly committed learner.
    """

    path = Path(checkpoint_path) if str(checkpoint_path or "") else Path()
    expected_digest = str(checkpoint_digest or "")
    if (
        not path.is_file()
        or not _is_sha256_digest(expected_digest)
    ):
        return {
            "available": False,
            "verified": False,
            "reason": "committed checkpoint path or digest unavailable",
            "checkpoint": str(path) if str(checkpoint_path or "") else None,
            "checkpoint_digest": expected_digest or None,
        }
    cache_path = cache_path or (
        ROOT / "outputs/state/dashboard-checkpoint-structure-cache.json"
    )
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "available": False,
            "verified": False,
            "reason": f"checkpoint stat failed: {exc}",
            "checkpoint": str(path),
            "checkpoint_digest": expected_digest,
        }
    cache = read_json(cache_path)
    if (
        cache.get("schema") == "poke_bot.dashboard_checkpoint_structure/v5"
        and cache.get("checkpoint") == str(path)
        and cache.get("checkpoint_digest") == expected_digest
        and int(cache.get("size_bytes") or -1) == int(stat.st_size)
        and int(cache.get("mtime_ns") or -1) == int(stat.st_mtime_ns)
        and cache.get("verified") is True
    ):
        return cache

    actual_digest = _file_sha256(path)
    if actual_digest != expected_digest:
        return {
            "schema": "poke_bot.dashboard_checkpoint_structure/v5",
            "available": True,
            "verified": False,
            "reason": "checkpoint digest does not match immutable commit",
            "checkpoint": str(path),
            "checkpoint_digest": expected_digest,
            "actual_digest": actual_digest,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    try:
        import torch

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(payload, dict):
            raise TypeError("checkpoint payload is not a mapping")
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint model_state_dict is unavailable")
        extra = payload.get("extra")
        extra = dict(extra) if isinstance(extra, dict) else {}
        adapter_config = extra.get("matchup_adapter_config")
        if not isinstance(adapter_config, dict):
            dormant_bank = extra.get("dormant_matchup_adapter_bank")
            dormant_bank = (
                dormant_bank if isinstance(dormant_bank, dict) else {}
            )
            adapter_config = dormant_bank.get("adapter_config")
        adapter_config = (
            dict(adapter_config) if isinstance(adapter_config, dict) else {}
        )
        adapter_keys = {
            str(key): value
            for key, value in state_dict.items()
            if str(key).startswith("matchup_adapter_bank.experts.")
            and hasattr(value, "numel")
        }
        expert_indexes = {
            int(parts[2])
            for key in adapter_keys
            for parts in [key.split(".")]
            if len(parts) > 3 and parts[2].isdigit()
        }
        adapter_parameters = sum(
            int(value.numel()) for value in adapter_keys.values()
        )
        expert_count = (
            max(expert_indexes) + 1 if expert_indexes else 0
        )
        adapter_registry = _checkpoint_adapter_registry_contract(
            adapter_config,
            expert_indexes,
        )
        expert_ids = list(adapter_registry["routable_expert_ids"])
        cached_parameter_count = int(extra.get("param_count") or 0)
        state_tensor_elements = sum(
            int(value.numel())
            for value in state_dict.values()
            if hasattr(value, "numel")
        )
        # The state dictionary is the executable model.  ``extra.param_count``
        # is only a historical display cache and can describe the pre-adapter
        # seed after a boundary materializes additional heads or adapter rows.
        # Reporting that cache made the dashboard regress to an older, smaller
        # parameter count even though the committed checkpoint was wider.
        model_parameters = state_tensor_elements
        expanded_head_training = _expanded_head_checkpoint_contract(
            state_dict,
            extra,
        )
        decision_fusion = _decision_fusion_checkpoint_contract(
            state_dict,
            payload,
        )
        model_config = dict(payload.get("model_config") or {})
        latent_keys = {
            str(key): value
            for key, value in state_dict.items()
            if str(key).startswith("latent_lookahead.")
            and hasattr(value, "numel")
        }
        latent_parameters = sum(
            int(value.numel()) for value in latent_keys.values()
        )
        latent_enabled = bool(
            model_config.get("latent_lookahead_enabled", False)
        )
        latent_authority_enabled = bool(
            model_config.get(
                "latent_lookahead_action_authority_enabled", False
            )
        )
        latent_lookahead = {
            "schema": "poke_bot.action_conditioned_latent_lookahead/v1",
            "enabled": latent_enabled,
            "action_authority_enabled": latent_authority_enabled,
            "parameters": latent_parameters,
            "width": int(model_config.get("latent_lookahead_width") or 0),
            "policy_aid_cap": float(
                model_config.get("latent_lookahead_policy_aid_cap") or 0.0
            ),
            "verified": bool(
                latent_enabled
                and latent_authority_enabled
                and latent_parameters > 0
                and int(model_config.get("latent_lookahead_width") or 0)
                == 512
                and float(
                    model_config.get("latent_lookahead_policy_aid_cap") or 0.0
                )
                == 0.25
            ),
        }
        # ``extra.param_count`` is a display cache, not checkpoint identity.
        # Immutable bootstrap-family freezing may omit it while preserving the
        # complete model state, adapter registry, and all executable tensors.
        # Do not turn that optional cache miss into a structural regression.
        structure_valid = bool(
            adapter_registry.get("verified") is True
            and adapter_parameters > 0
            and expanded_head_training.get("verified") is True
            and decision_fusion.get("verified") is True
        )
        result = {
            "schema": "poke_bot.dashboard_checkpoint_structure/v5",
            "available": True,
            "verified": structure_valid,
            "reason": (
                None
                if structure_valid
                else (
                    expanded_head_training.get("reason")
                    if expanded_head_training.get("verified") is not True
                    else decision_fusion.get("reason")
                    if decision_fusion.get("verified") is not True
                    else adapter_registry.get("reason")
                )
            ),
            "checkpoint": str(path),
            "checkpoint_digest": expected_digest,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "model_parameters": model_parameters or None,
            "cached_parameter_count": cached_parameter_count or None,
            "cached_parameter_count_matches_state": bool(
                cached_parameter_count == state_tensor_elements
            ),
            "state_tensor_elements": state_tensor_elements,
            "adapter_parameters": adapter_parameters,
            "adapter_expert_count": expert_count,
            "adapter_expert_ids": expert_ids,
            "adapter_format": adapter_registry.get("format"),
            "adapter_registry_verified": adapter_registry.get("verified"),
            "adapter_slot_capacity": adapter_registry.get(
                "physical_slot_capacity"
            ),
            "adapter_slot_registry_digest": adapter_registry.get(
                "slot_registry_digest"
            ),
            "model_config": model_config,
            "runtime_enabled_at_save": bool(
                extra.get("matchup_adapters_runtime_enabled")
            ),
            "ordinary_optimizer_included_at_save": bool(
                extra.get("matchup_adapter_optimizer_included")
            ),
            "expanded_head_training": expanded_head_training,
            "decision_fusion": decision_fusion,
            "latent_lookahead": latent_lookahead,
            "rl_iteration": payload.get("rl_iteration"),
            "saved_at": payload.get("saved_at"),
            "source": "active committed checkpoint payload + tensor structure",
        }
    except Exception as exc:
        return {
            "schema": "poke_bot.dashboard_checkpoint_structure/v5",
            "available": True,
            "verified": False,
            "reason": f"checkpoint inspection failed: {type(exc).__name__}: {exc}",
            "checkpoint": str(path),
            "checkpoint_digest": expected_digest,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    if result.get("verified") is True:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(
                cache_path.suffix + f".{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(result, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            os.replace(temporary, cache_path)
        except OSError:
            pass
    return result


def staged_expanded_head_training_state(
    handoff_state: dict[str, Any],
    handoff_contract: dict[str, Any],
    *,
    cumulative_core_contract: dict[str, Any] | None = None,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve transition heads without confusing them with live runtime.

    Prefer the actively checkpointed cumulative-core refresh when present.
    Otherwise, a generated handoff contract may describe a future schedule,
    but only a checksum-matched immutable bootstrap checkpoint can prove
    tensors, losses, or training coverage.
    """

    core_contract = (
        dict(cumulative_core_contract)
        if isinstance(cumulative_core_contract, dict)
        else {}
    )
    core_refresh = dict(core_contract.get("core_refresh") or {})
    core_run_dir_value = str(core_refresh.get("run_dir") or "").strip()
    core_run_state_path = (
        Path(core_run_dir_value).expanduser().resolve() / "state.json"
        if core_run_dir_value
        else None
    )
    core_run_state = (
        read_json(core_run_state_path)
        if core_run_state_path is not None
        else {}
    )
    core_history = [
        dict(row)
        for row in (core_run_state.get("history") or [])
        if isinstance(row, dict)
    ]
    bootstrap = dict(handoff_state.get("next_specialist_bootstrap") or {})
    target = dict(handoff_contract.get("next_specialist") or {})
    specialist_id = str(
        bootstrap.get("specialist_id") or target.get("id") or ""
    ).strip()
    run_state: dict[str, Any] = {}
    run_dir_value = str(target.get("run_dir") or "").strip()
    if run_dir_value:
        run_state = read_json(
            Path(run_dir_value).expanduser().resolve() / "state.json"
        )
    history = [
        dict(row)
        for row in (run_state.get("history") or [])
        if isinstance(row, dict)
    ]
    target_selected = bool(
        specialist_id
        and str(handoff_state.get("phase") or "")
        in {
            "next_specialist_selected",
            "next_specialist_started",
        }
    )
    if core_history and not target_selected:
        latest_core = core_history[-1]
        core_checkpoint = str(latest_core.get("checkpoint") or "").strip()
        core_digest = str(
            latest_core.get("checkpoint_digest") or ""
        ).strip()
        if core_checkpoint and _is_sha256_digest(core_digest):
            expanded = dict(
                latest_core.get("expanded_head_training") or {}
            )
            checkpoint_verified = _file_sha256_matches(
                Path(core_checkpoint).expanduser().resolve(),
                core_digest,
            )
            heads = dict(expanded.get("heads") or {})
            declared_heads = [
                str(value)
                for value in (
                    expanded.get("architecture_present_heads") or []
                )
                if str(value)
            ]
            metadata_verified = bool(
                expanded.get("schema") == EXPANDED_HEAD_CONTRACT_SCHEMA
                and declared_heads
                and set(declared_heads) == set(heads)
                and all(
                    isinstance(heads.get(head_id), dict)
                    and heads[head_id].get("present") is True
                    for head_id in declared_heads
                )
            )
            verified = bool(checkpoint_verified and metadata_verified)
            return {
                **expanded,
                "available": True,
                "verified": verified,
                "reason": (
                    None
                    if verified
                    else (
                        "active core checkpoint checksum mismatch"
                        if not checkpoint_verified
                        else "active core expanded-head metadata is incomplete"
                    )
                ),
                "scope": "active_cumulative_core_refresh",
                "specialist_id": "deck-agnostic-core",
                "checkpoint": core_checkpoint,
                "checkpoint_digest": core_digest,
                "checkpoint_pending": False,
                "epoch": latest_core.get("epoch"),
                "epochs_target": core_refresh.get("max_epochs"),
                "source": str(core_run_state_path),
            }

    checkpoint_path = ""
    checkpoint_digest = str(bootstrap.get("checkpoint_digest") or "")
    ready_path: Path | None = None
    ready_value = str(bootstrap.get("ready") or target.get("ready") or "").strip()
    if ready_value:
        ready_path = Path(ready_value).expanduser().resolve()
        expected_ready_digest = str(bootstrap.get("ready_sha256") or "")
        ready = (
            read_json(ready_path)
            if (
                not expected_ready_digest
                or _file_sha256_matches(ready_path, expected_ready_digest)
            )
            else {}
        )
        checkpoint_path = str(ready.get("checkpoint") or "")
        checkpoint_digest = str(
            ready.get("checkpoint_digest") or checkpoint_digest
        )

    if history:
        latest = history[-1]
        checkpoint_path = str(latest.get("checkpoint") or checkpoint_path)
        checkpoint_digest = str(
            latest.get("checkpoint_digest") or checkpoint_digest
        )

    training_contract = (
        dict(handoff_contract.get("training") or {})
        if isinstance(handoff_contract.get("training"), dict)
        else {}
    )
    schedule = dict(
        training_contract.get("expanded_head_training")
        or training_contract.get("expanded_heads")
        or {}
    )
    if not checkpoint_path or not _is_sha256_digest(checkpoint_digest):
        return {
            "schema": EXPANDED_HEAD_CONTRACT_SCHEMA,
            "available": bool(schedule),
            "verified": False,
            "checkpoint_pending": True,
            "scope": "staged_next_specialist",
            "specialist_id": specialist_id or None,
            "reason": (
                "staged head schedule exists; checksum-bound bootstrap checkpoint pending"
                if schedule
                else "no staged expanded-head checkpoint"
            ),
            "schedule": schedule,
            "heads": [],
        }
    structure = checkpoint_structure_telemetry(
        checkpoint_path,
        checkpoint_digest,
        cache_path=cache_path
        or (ROOT / "outputs/state/dashboard-staged-head-structure-cache.json"),
    )
    expanded = dict(structure.get("expanded_head_training") or {})
    return {
        **expanded,
        "scope": "staged_next_specialist",
        "specialist_id": specialist_id or None,
        "checkpoint": checkpoint_path,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_pending": False,
        "source": (
            str(ready_path)
            if ready_path is not None
            else str(Path(run_dir_value) / "state.json")
        ),
    }


def _exact_calendar_dates(
    values: list[Any] | tuple[Any, ...],
    *,
    count: int = 20,
) -> list[str]:
    """Return an exact contiguous calendar window or an empty list."""

    raw = [str(value) for value in values if str(value)]
    if len(raw) != count or len(set(raw)) != count:
        return []
    try:
        parsed = [date.fromisoformat(value) for value in raw]
    except ValueError:
        return []
    ordered = sorted(parsed)
    if any(
        right - left != timedelta(days=1)
        for left, right in zip(ordered, ordered[1:])
    ):
        return []
    return [value.isoformat() for value in ordered]


def _latest20_window_dates(
    source_window: dict[str, Any],
    manifest_dates: list[str],
    archive_refresh: dict[str, Any],
) -> list[str]:
    """Resolve exactly 20 primary calendar slots without using fallback data."""

    source_dates = _exact_calendar_dates(
        list(source_window.get("dates") or [])
    )
    if source_dates:
        return source_dates
    manifest_window = _exact_calendar_dates(manifest_dates)
    if manifest_window:
        return manifest_window
    refresh_days = [
        str(row.get("day") or row.get("date") or "")
        for row in (archive_refresh.get("days") or [])
        if isinstance(row, dict)
    ]
    refresh_window = _exact_calendar_dates(refresh_days)
    if refresh_window:
        return refresh_window
    try:
        start = date.fromisoformat(str(archive_refresh.get("window_start") or ""))
        end = date.fromisoformat(str(archive_refresh.get("window_end") or ""))
    except ValueError:
        return []
    if end - start != timedelta(days=19):
        return []
    return [(start + timedelta(days=offset)).isoformat() for offset in range(20)]


def _checksum_receipted_historical_fallback(
    protected_path: Path,
    protected: dict[str, Any],
    manifest: dict[str, Any],
    *,
    latest20_all_zero: bool,
) -> dict[str, Any]:
    """Validate a separately receipted historical corpus fallback."""

    raw = (
        manifest.get("historical_fallback")
        if isinstance(manifest.get("historical_fallback"), dict)
        else protected.get("historical_fallback")
        if isinstance(protected.get("historical_fallback"), dict)
        else {}
    )
    if not raw:
        return {
            "available": False,
            "used": False,
            "label": "Historical fallback · none",
            "not_latest20": True,
        }
    receipt_name = str(raw.get("receipt") or raw.get("path") or "").strip()
    receipt_path = (
        protected_path.parent / receipt_name if receipt_name else Path()
    )
    expected_digest = str(
        raw.get("receipt_sha256") or raw.get("sha256") or ""
    )
    receipt_valid = bool(
        receipt_name
        and receipt_path.is_file()
        and _is_sha256_digest(expected_digest)
        and _file_sha256(receipt_path) == expected_digest
    )
    reason = str(raw.get("reason") or "")
    requested = raw.get("used") is True
    allowed = bool(
        requested
        and latest20_all_zero
        and reason
        in {
            "latest20_all_zero_matches",
            "all_latest20_days_zero_matches",
        }
        and receipt_valid
    )
    receipt = read_json(receipt_path) if receipt_valid else {}
    return {
        "available": receipt_valid,
        "used": allowed,
        "requested": requested,
        "label": (
            "Historical checksum-receipted fallback · not latest20"
            if receipt_valid
            else "Historical fallback · receipt invalid"
        ),
        "not_latest20": True,
        "reason": reason or None,
        "receipt": str(receipt_path) if receipt_name else None,
        "receipt_checksum": expected_digest or None,
        "records_kept": int(
            receipt.get("records_kept")
            or (receipt.get("totals") or {}).get("records_kept")
            or 0
        ),
        "decisions_kept": int(
            receipt.get("decisions_kept")
            or (receipt.get("totals") or {}).get("decisions_kept")
            or 0
        ),
        "rejection_reason": (
            None
            if allowed or not requested
            else "historical fallback requires 20 validated zero-match primary days"
            if not latest20_all_zero
            else "historical fallback checksum receipt is invalid"
            if not receipt_valid
            else "historical fallback reason is not allowed"
        ),
    }


def active_expert_corpus_state(
    curriculum: dict[str, Any],
    archive_refresh: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the expert card from the protected corpus pinned by this run."""

    rehearsal = curriculum.get("expert_rehearsal") or {}
    protected_text = str(rehearsal.get("manifest") or "").strip()
    protected_path = Path(protected_text) if protected_text else Path()
    protected = read_json(protected_path)
    manifest_name = str(protected.get("manifest") or "").strip()
    manifest_path = (
        protected_path.parent / manifest_name
        if protected_text and manifest_name
        else Path()
    )
    manifest = read_json(manifest_path)
    expected_manifest_digest = str(protected.get("manifest_sha256") or "")
    manifest_digest_valid = bool(
        manifest_path.is_file()
        and _is_sha256_digest(expected_manifest_digest)
        and _file_sha256(manifest_path) == expected_manifest_digest
    )
    if not protected_text or not manifest:
        return {
            **archive_refresh,
            "authoritative_for_active_run": False,
            "reason": "active run has no readable pinned expert corpus",
        }
    shards = (
        manifest.get("shards")
        if isinstance(manifest.get("shards"), list)
        else []
    )
    source_window = (
        manifest.get("source_window")
        if isinstance(manifest.get("source_window"), dict)
        else {}
    )
    source_days = (
        manifest.get("source_days")
        if isinstance(manifest.get("source_days"), list)
        else []
    )
    dates = [str(value) for value in manifest.get("dates") or [] if str(value)]
    day_rows: list[dict[str, Any]] = []
    ready_shards = 0
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        shard_path = protected_path.parent / str(shard.get("path") or "")
        expected_bytes = int(shard.get("bytes") or 0)
        shard_ready = bool(
            shard_path.is_file()
            and expected_bytes > 0
            and shard_path.stat().st_size == expected_bytes
            and _is_sha256_digest(str(shard.get("sha256") or ""))
        )
        ready_shards += int(shard_ready)
        stats = shard.get("stats") if isinstance(shard.get("stats"), dict) else {}
        shard_dates = [
            str(value) for value in shard.get("source_dates") or [] if str(value)
        ]
        day_rows.append(
            {
                "day": shard_dates[0] if shard_dates else "unknown",
                "host": "Inzi",
                "stage": "feature_ready" if shard_ready else "missing",
                "percent": 100.0 if shard_ready else 0.0,
                "archive_bytes": expected_bytes or None,
                "feature_records": int(stats.get("records_kept") or 0),
                "feature_decisions": int(stats.get("decisions_kept") or 0),
                "partial_bytes": None,
                "progress_estimated": False,
                "service": {"active": False},
            }
        )
    ready_filtered_shards = ready_shards
    latest20_dates = _latest20_window_dates(
        source_window,
        dates,
        archive_refresh,
    )
    source_day_by_date = {
        str(row.get("date") or ""): row
        for row in source_days
        if isinstance(row, dict) and str(row.get("date") or "")
    }
    archive_day_by_date = {
        str(row.get("day") or row.get("date") or ""): row
        for row in (archive_refresh.get("days") or [])
        if isinstance(row, dict)
        and str(row.get("day") or row.get("date") or "")
    }
    active_filter_archetype = str(
        source_window.get("filter_archetype")
        or (
            (manifest.get("selection") or {}).get("value")
            if isinstance(manifest.get("selection"), dict)
            else ""
        )
        or "active specialist"
    )
    source_day_contract = bool(
        latest20_dates
        and int(source_window.get("days") or 0) == 20
        and str(source_window.get("unit") or "") == "calendar_day"
        and source_window.get("filter_applied_after_window_selection") is True
        and set(source_day_by_date) == set(latest20_dates)
    )
    if latest20_dates:
        day_rows = []
        ready_shards = 0
        for day_value in latest20_dates:
            row = source_day_by_date.get(day_value)
            archive_row = archive_day_by_date.get(day_value) or {}
            row = row if isinstance(row, dict) else {}
            present = bool(
                row.get("source_feature_validated") is True
                and row.get("source_archive_validated") is True
                and _is_sha256_digest(
                    str(row.get("source_feature_sha256") or "")
                )
                and _is_sha256_digest(
                    str(row.get("source_archive_sha256") or "")
                )
            )
            ready_shards += int(present)
            matching_games = (
                int(row.get("matching_games") or 0)
                if row and row.get("matching_games") is not None
                else None
            )
            matching_decisions = (
                int(row.get("matching_decisions") or 0)
                if row and row.get("matching_decisions") is not None
                else None
            )
            archive_present = bool(
                archive_row.get("stage")
                in {"archive_ready", "feature_ready", "ready", "complete"}
                or archive_row.get("percent") == 100.0
            )
            archive_featurizing = bool(
                archive_row.get("stage") == "featurizing"
                and (archive_row.get("service") or {}).get("active") is True
            )
            day_rows.append(
                {
                    "day": day_value,
                    "host": (
                        "source receipt"
                        if row
                        else str(archive_row.get("host") or "")
                        if archive_featurizing
                        else "latest20 source window"
                    ),
                    "stage": (
                        "feature_ready"
                        if present
                        else "featurizing"
                        if archive_featurizing
                        else "source_ready_unfiltered"
                        if archive_present
                        else "missing"
                    ),
                    "percent": (
                        100.0
                        if present
                        else float(archive_row.get("percent") or 0.0)
                        if archive_featurizing
                        else 0.0
                    ),
                    "archive_bytes": None,
                    "feature_records": matching_games,
                    "feature_decisions": matching_decisions,
                    "matching_games": matching_games,
                    "matching_decisions": matching_decisions,
                    "zero_match_present": bool(
                        present and matching_games == 0
                    ),
                    "specialist_id": active_filter_archetype,
                    "matching_status": (
                        "zero_matches"
                        if present and matching_games == 0
                        else "matches_present"
                        if present and isinstance(matching_games, int)
                        else "filter_receipt_missing"
                    ),
                    "active_specialist_filter_receipt": bool(row),
                    "present": present,
                    "partial_bytes": None,
                    "progress_estimated": archive_featurizing,
                    "service": {"active": archive_featurizing},
                }
            )
    quality = (
        manifest.get("quality_gates")
        if isinstance(manifest.get("quality_gates"), dict)
        else {}
    )
    totals = (
        manifest.get("totals")
        if isinstance(manifest.get("totals"), dict)
        else {}
    )
    selection = (
        manifest.get("selection")
        if isinstance(manifest.get("selection"), dict)
        else {}
    )
    filtered_shards_complete = bool(
        protected.get("schema") == "poke_bot.pinned_expert_corpus/v1"
        and protected.get("protected") is True
        and manifest_digest_valid
        and quality.get("passed") is True
        and shards
    )
    complete = bool(
        filtered_shards_complete
        and ready_filtered_shards == len(shards)
    )
    if source_day_contract:
        required_source_days = int(source_window.get("days") or 0)
        complete = bool(
            filtered_shards_complete
            and required_source_days == 20
            and len(day_rows) == required_source_days
            and ready_shards == required_source_days
        )
    else:
        complete = False
    evidence_start = None
    evidence_end = None
    evidence_match = re.search(
        r"expert-evidence\d+-(\d{8})-(\d{8})",
        protected_text,
    )
    if evidence_match:
        evidence_start = (
            f"{evidence_match.group(1)[:4]}-"
            f"{evidence_match.group(1)[4:6]}-"
            f"{evidence_match.group(1)[6:]}"
        )
        evidence_end = (
            f"{evidence_match.group(2)[:4]}-"
            f"{evidence_match.group(2)[4:6]}-"
            f"{evidence_match.group(2)[6:]}"
        )
    records = int(totals.get("records_kept") or 0)
    decisions = int(totals.get("decisions_kept") or 0)
    specialist = str(selection.get("value") or "active specialist")
    latest20_all_zero = bool(
        source_day_contract
        and ready_shards == 20
        and all(row.get("matching_games") == 0 for row in day_rows)
    )
    historical_fallback = _checksum_receipted_historical_fallback(
        protected_path,
        protected,
        manifest,
        latest20_all_zero=latest20_all_zero,
    )
    archive_window_ready = bool(
        archive_refresh.get("available") is True
        and archive_refresh.get("archive_window_ready") is True
        and int(archive_refresh.get("total_days") or 0) == 20
        and len(archive_refresh.get("days") or ()) == 20
    )
    archive_ready_days = (
        20
        if archive_window_ready
        else int(archive_refresh.get("archive_ready_days") or 0)
    )
    refresh_day_rows = [
        {
            **dict(row),
            "binding_status": "staged_for_next_safe_boundary",
        }
        for row in (archive_refresh.get("days") or [])
        if isinstance(row, dict)
    ]
    display_refresh = bool(
        not source_day_contract
        and archive_window_ready
        and len(refresh_day_rows) == 20
        and int(archive_refresh.get("feature_ready_days") or 0) == 20
    )
    display_rows = refresh_day_rows if display_refresh else day_rows
    display_ready_days = (
        int(archive_refresh.get("feature_ready_days") or 0)
        if display_refresh
        else ready_shards
    )
    display_complete = (
        bool(archive_refresh.get("complete"))
        if display_refresh
        else complete
    )
    expanded_v6 = (
        dict(archive_refresh.get("expanded_v6") or {})
        if isinstance(archive_refresh.get("expanded_v6"), dict)
        else {}
    )
    return {
        "available": True,
        "active": bool(
            rehearsal.get("active") or archive_refresh.get("active")
        ),
        "complete": display_complete,
        "archive_window_ready": archive_window_ready,
        "host": (
            "Inzi"
            if rehearsal.get("active")
            else str(archive_refresh.get("host") or "Inzi")
        ),
        "stage": (
            "rehearsal_active"
            if rehearsal.get("active")
            else str(archive_refresh.get("stage") or "preparing_filtered_corpus")
            if display_refresh
            else "preparing_filtered_corpus"
            if archive_refresh.get("active")
            else "ready"
        ),
        "phase": (
            str(rehearsal.get("state") or "scheduled")
            if rehearsal.get("active")
            else str(archive_refresh.get("phase") or "latest20_refresh")
            if display_refresh or archive_refresh.get("active")
            else "scheduled"
        ),
        "window_start": latest20_dates[0] if latest20_dates else None,
        "window_end": latest20_dates[-1] if latest20_dates else None,
        "evidence_window_start": evidence_start,
        "evidence_window_end": evidence_end,
        "current_day": None,
        "current": display_ready_days,
        "total": len(display_rows),
        "progress_estimated": False,
        "completed_days": display_ready_days,
        "archive_ready_days": archive_ready_days,
        "feature_ready_days": display_ready_days,
        "local_feature_ready_days": (
            int(archive_refresh.get("local_feature_ready_days") or 0)
            if display_refresh
            else ready_shards
        ),
        "total_days": len(display_rows),
        "percent": (
            float(archive_refresh.get("percent") or 0.0)
            if display_refresh
            else 100.0 * ready_shards / len(day_rows)
            if day_rows
            else 0.0
        ),
        "day_percent": None,
        "days": display_rows,
        "latest_line": (
            f"ACTIVE RUN · {specialist} protected expert corpus · "
            f"{records:,} selected games · {decisions:,} decisions"
            + (
                " · immutable run binding retained until the next safe "
                "rehearsal/bootstrap boundary"
                if display_refresh
                else f" · {ready_shards}/{len(day_rows)} latest20 "
                "checksum-pinned source days ready"
            )
            + (
                " · "
                + str(
                    archive_refresh.get("latest_line")
                    or "Elmo daily feature materialization active"
                )
                if archive_refresh.get("active")
                else ""
            )
            + (
                " · " + str(expanded_v6.get("latest_line"))
                if expanded_v6.get("available") is True
                and expanded_v6.get("latest_line")
                else ""
            )
            + (
                " · HISTORICAL FALLBACK USED (not latest20)"
                if historical_fallback.get("used") is True
                else ""
            )
        ),
        "reason": (
            None
            if display_complete
            else (
                "latest20 specialist corpus checksum sync active; the current "
                "run remains pinned to its immutable expert corpus"
            )
            if display_refresh
            else "active-specialist filtered corpus preparation pending"
            if archive_window_ready
            else "active protected corpus validation failed"
        ),
        "assembled_manifest_ready": manifest_digest_valid,
        "filtered_corpus_ready": display_complete,
        "source": str(protected_path),
        "manifest_source": str(manifest_path),
        "manifest_digest": expected_manifest_digest or None,
        "specialist_id": specialist,
        "records_kept": records,
        "decisions_kept": decisions,
        "quality_gates": quality,
        "source_window": source_window,
        "source_day_contract": source_day_contract,
        "source_day_contract_satisfied": bool(
            source_day_contract
            and int(source_window.get("days") or 0) == 20
            and len(day_rows) == 20
            and ready_shards == 20
        ),
        "latest20": {
            "label": "Latest 20 calendar days",
            "not_fallback": True,
            "dates": latest20_dates,
            "days": day_rows,
            "matching_games": sum(
                int(row.get("matching_games") or 0) for row in day_rows
            ),
            "matching_decisions": sum(
                int(row.get("matching_decisions") or 0) for row in day_rows
            ),
            "all_zero_matches": latest20_all_zero,
            "complete": bool(
                source_day_contract and ready_shards == 20
            ),
        },
        "historical_fallback": historical_fallback,
        "rehearsal": rehearsal,
        "authoritative_for_active_run": True,
        "active_bound_corpus": {
            "source": str(protected_path),
            "manifest_source": str(manifest_path),
            "records_kept": records,
            "decisions_kept": decisions,
            "source_day_contract": source_day_contract,
            "latest20_ready_days": ready_shards,
            "immutable_until_safe_boundary": True,
        },
        "archive_refresh_history": {
            "source": archive_refresh.get("source"),
            "window_start": archive_refresh.get("window_start"),
            "window_end": archive_refresh.get("window_end"),
            "complete": archive_refresh.get("complete"),
            "superseded_by": "active_run_pinned_expert_corpus",
        },
        "next_boundary_expanded_corpus": expanded_v6,
        "updated_at": (
            manifest_path.stat().st_mtime if manifest_path.is_file() else None
        ),
        "metric_definition": (
            "Exact filtered replay records and decisions from the protected "
            "expert corpus pinned by the active run."
        ),
    }


def parse_metric(line: str, name: str) -> float | None:
    match = re.search(rf"(?:^|[ ,]){re.escape(name)}=(-?[0-9.]+)%?", line)
    return float(match.group(1)) if match else None


def bootstrap_progress() -> dict[str, Any]:
    raw = read_tail(BOOTSTRAP_LOG)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    marker = clean.rfind("== train_bootstrap")
    if marker >= 0:
        clean = clean[marker:]
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if "train ep" in line and re.search(r"\d+/\d+", line):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "featurize " in line and "seq" in line:
                latest = line
                break
    if not latest and lines:
        latest = lines[-1]

    result: dict[str, Any] = {
        "log": str(BOOTSTRAP_LOG),
        "latest_line": latest,
        "log_exists": BOOTSTRAP_LOG.exists(),
        "log_bytes": BOOTSTRAP_LOG.stat().st_size if BOOTSTRAP_LOG.exists() else 0,
        "updated_at": BOOTSTRAP_LOG.stat().st_mtime if BOOTSTRAP_LOG.exists() else None,
        "epoch": None,
        "current": None,
        "total": None,
        "percent": None,
        "batch_per_second": None,
        "eta": None,
        "metrics": {},
        "phase": "loading",
        "sequences": None,
        "sequences_per_second": None,
    }
    match = re.search(r"train ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]", latest)
    if match:
        epoch, percent, current, total, timing = match.groups()
        result.update(
            epoch=int(epoch) + 1,
            current=int(current),
            total=int(total),
            percent=float(percent),
        )
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            result["batch_per_second"] = float(rate.group(1))
        eta = re.search(r"<([^,]+),", timing)
        if eta:
            result["eta"] = eta.group(1)
        result["phase"] = "training"
    featurize = re.search(r"featurize\s+.*?:\s*(\d+)seq\s+\[[^,]+,\s*([0-9.]+)seq/s\]", latest)
    if featurize:
        result["phase"] = "featurizing"
        result["sequences"] = int(featurize.group(1))
        result["sequences_per_second"] = float(featurize.group(2))
    result["metrics"] = {
        name: parse_metric(latest, name)
        for name in (
            "acc", "loss", "p", "policy", "v", "value", "aux", "hand",
            "rem", "lethal", "prize", "guide", "step",
        )
    }
    return result


def alakazam_bootstrap_progress() -> dict[str, Any]:
    """Parse the live, device-resident Alakazam expert bootstrap."""
    service = unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True)
    raw = read_tail(ALAKAZAM_BOOTSTRAP_LOG, 2_000_000)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if re.search(
            r"(?:train|val) ep\d+:.*?\d+/\d+"
            r"|expert rehearsal before iter\d+ ep\d+/\d+:.*?\d+/\d+",
            line,
        ):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "pack Blackwell corpus" in line or line.startswith("[train] device="):
                latest = line
                break
    if not latest and lines:
        latest = lines[-1]

    build = read_json(ALAKAZAM_BUILD_READY)
    corpus = build.get("expert_corpus") if isinstance(build.get("expert_corpus"), dict) else {}
    corpus_games = as_number(str(corpus.get("records") or "")) or 39_467
    corpus_decisions = as_number(str(corpus.get("decisions") or "")) or 2_579_178
    train_games = int(round(corpus_games * 0.90))
    train_decisions = int(round(corpus_decisions * 0.90))
    phase = "loading"
    epoch = None
    current = None
    total = None
    percent = None
    batch_rate = None
    eta = None

    match = re.search(
        r"(train|val) ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
        latest,
    )
    split = "train"
    if match:
        split, epoch_raw, percent_raw, current_raw, total_raw, timing = match.groups()
        phase = "training" if split == "train" else "validation"
        epoch = int(epoch_raw) + 1
        percent = float(percent_raw)
        current = int(current_raw)
        total = int(total_raw)
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            batch_rate = float(rate.group(1))
        eta_match = re.search(r"<([^,]+),", timing)
        if eta_match:
            eta = eta_match.group(1)
    else:
        rehearsal = re.search(
            r"expert rehearsal before iter(\d+) ep\d+/\d+:\s*"
            r"(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
            latest,
        )
        if rehearsal:
            epoch_raw, percent_raw, current_raw, total_raw, timing = (
                rehearsal.groups()
            )
            phase = "training"
            epoch = int(epoch_raw)
            percent = float(percent_raw)
            current = int(current_raw)
            total = int(total_raw)
            rate = re.search(r"([0-9.]+)batch/s", timing)
            if rate:
                batch_rate = float(rate.group(1))
            eta_match = re.search(r"<([^,]+),", timing)
            if eta_match:
                eta = eta_match.group(1)
    if phase == "loading" and "pack Blackwell corpus" in latest:
        phase = "packing"
        match = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", latest)
        if match:
            percent, current, total = (
                float(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

    # Derive rates only from the measured batch rate and the exact corpus
    # partition. This is an epoch-average conversion, not a fabricated game
    # simulator rate.
    samples_per_second = None
    game_equivalents_per_second = None
    if batch_rate is not None and total:
        denominator = int(total)
        split_games = train_games if split == "train" else corpus_games - train_games
        split_decisions = (
            train_decisions if split == "train" else corpus_decisions - train_decisions
        )
        samples_per_second = batch_rate * split_decisions / denominator
        game_equivalents_per_second = batch_rate * split_games / denominator

    updated = None
    try:
        updated = ALAKAZAM_BOOTSTRAP_LOG.stat().st_mtime
    except OSError:
        pass
    ready = ALAKAZAM_BOOTSTRAP_READY.is_file()
    status = "complete" if ready else "running" if service.get("active") else "waiting"
    metrics = {
        name: parse_metric(latest, name)
        for name in (
            "acc", "loss", "p", "policy", "v", "value", "aux", "hand",
            "rem", "lethal", "prize", "guide", "step",
        )
    }
    return {
        "authoritative": True,
        "source": str(ALAKAZAM_BOOTSTRAP_LOG),
        "log": str(ALAKAZAM_BOOTSTRAP_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(updated and time.time() - updated < 30),
        "status": status,
        "mode": "alakazam_expert_bootstrap_device_resident",
        "phase": "complete" if ready else phase,
        "epoch": epoch,
        "epochs_target": 25,
        "current": current,
        "total": total,
        "percent": 100.0 if ready else percent,
        "batch_per_second": batch_rate,
        "samples_per_second": samples_per_second,
        "game_equivalents_per_second": game_equivalents_per_second,
        "acting_sequences_per_second": game_equivalents_per_second,
        "corpus_games": corpus_games,
        "corpus_records": corpus_games,
        "corpus_decisions": corpus_decisions,
        "eta": eta,
        "metrics": metrics,
        "all_training_tensors_device_resident": phase in {"training", "validation"},
        "gpu_name": "NVIDIA RTX PRO 5000 Blackwell",
        "service": service,
    }


def final_format_crustle_progress() -> dict[str, Any]:
    """Project the managed all-guide Crustle H10 bootstrap."""

    service = unit_state(FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_SERVICE, user=True)
    active = bool(
        (
            service.get("active")
            or service.get("active_state") == "activating"
        )
        and (
            int(service.get("pid") or 0) > 0
            or service.get("sub_state") in {"running", "start"}
        )
    )
    if not active:
        return {"status": "waiting", "available": False}
    raw = read_tail(FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG, 2_000_000)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if re.search(
            r"(?:train|val) ep\d+:.*?\d+/\d+"
            r"|expert rehearsal before iter\d+ ep\d+/\d+:.*?\d+/\d+",
            line,
        ):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "pack Blackwell corpus" in line or "loading protected" in line:
                latest = line
                break
    phase = "loading"
    epoch = current = total = None
    percent = rate = None
    eta = None
    match = re.search(
        r"(?:train|val) ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
        latest,
    )
    if not match:
        match = re.search(
            r"expert rehearsal before iter(\d+) ep\d+/\d+:\s*"
            r"(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
            latest,
        )
    if match:
        epoch_raw, percent_raw, current_raw, total_raw, timing = match.groups()
        phase = "training"
        epoch = int(epoch_raw)
        percent = float(percent_raw)
        current = int(current_raw)
        total = int(total_raw)
        rate_match = re.search(r"([0-9.]+)batch/s", timing)
        rate = float(rate_match.group(1)) if rate_match else None
        eta_match = re.search(r"<([^,]+),", timing)
        eta = eta_match.group(1) if eta_match else None
    elif "pack Blackwell corpus" in latest:
        phase = "packing"
        pack = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", latest)
        if pack:
            percent, current, total = (
                float(pack.group(1)), int(pack.group(2)), int(pack.group(3))
            )
    freeze = read_json(FINAL_FORMAT_CRUSTLE_TRAINING_FREEZE)
    checkpoint = freeze.get("checkpoint")
    digest = freeze.get("checkpoint_sha256")
    structure = checkpoint_structure_telemetry(
        checkpoint,
        digest,
        cache_path=ROOT / "outputs/state/dashboard-crustle-h10-parent-structure-cache.json",
    )
    updated = (
        FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG.stat().st_mtime
        if FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG.is_file()
        else None
    )
    return {
        "available": True,
        "authoritative": True,
        "source": str(FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG),
        "log": str(FINAL_FORMAT_CRUSTLE_H10_BOOTSTRAP_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(updated and time.time() - updated < 30),
        "status": "running",
        "mode": "final_format_crustle_h10_bootstrap",
        "phase": phase,
        "run": "final_format_crustle_r113_h10_bootstrap",
        "specialist_id": "crustle",
        "epoch": epoch,
        "epochs_target": 35,
        "current": current,
        "total": total,
        "percent": percent,
        "rate": rate,
        "rate_unit": "batch/s" if rate is not None else None,
        "eta": eta,
        "metrics": {
            name: parse_metric(latest, name)
            for name in (
                "acc", "loss", "p", "policy", "v", "value", "aux",
                "hand", "rem", "lethal", "prize", "guide", "step",
            )
        },
        "corpus_games": 26932,
        "corpus_decisions": 1428142,
        "checkpoint": checkpoint,
        "checkpoint_digest": digest,
        "checkpoint_structure": structure,
        "model_parameters": int(structure.get("model_parameters") or 0),
        "capacity_profile": "H10-I/v1",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "guide_state": "active_all_35_epochs",
        "pilot_weighting": "active_all_35_epochs",
        "service": service,
    }


def final_format_marnie_progress() -> dict[str, Any]:
    """Return the current receipt-bound Marnie H10 bootstrap state."""

    bootstrap_service = unit_state(
        FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_SERVICE,
        user=True,
    )
    rl_service = unit_state(FINAL_FORMAT_MARNIE_H10_RL_SERVICE, user=True)
    bootstrap_active = bool(
        bootstrap_service.get("active")
        and (
            int(bootstrap_service.get("pid") or 0) > 0
            or bootstrap_service.get("sub_state") in {"running", "start"}
        )
    )
    rl_active = bool(
        rl_service.get("active")
        and (
            int(rl_service.get("pid") or 0) > 0
            or rl_service.get("sub_state") in {"running", "start"}
        )
    )
    ready = read_json(FINAL_FORMAT_MARNIE_H10_READY)
    validation = read_json(FINAL_FORMAT_MARNIE_H10_VALIDATION)
    status_text = read_tail(FINAL_FORMAT_MARNIE_H10_PROGRESS_STATUS, 20_000)
    progress_log = read_tail(FINAL_FORMAT_MARNIE_H10_PROGRESS_LOG, 2_000_000)
    loop_state = read_json(FINAL_FORMAT_MARNIE_H10_RUN_DIR / "loop_state.json")
    registry = read_json(FINAL_FORMAT_MARNIE_H10_REGISTRY)
    rl_history = bool(
        loop_state
        and registry
        and (
            status_text.strip()
            or FINAL_FORMAT_MARNIE_H10_PROGRESS_LOG.is_file()
            or FINAL_FORMAT_MARNIE_H10_LOG.is_file()
        )
    )
    if not (bootstrap_active or rl_active or rl_history or ready or validation):
        return {"status": "waiting", "available": False}

    # A live bootstrap owns the display until its own managed boundary
    # completes. Historical RL files from a later/other fixture must never
    # preempt an actually active bootstrap.
    if rl_active or (rl_history and not bootstrap_active):
        progress = parse_curriculum_progress(
            status_text,
            progress_log,
            iteration_hint=int(loop_state.get("next_iteration") or 0),
        )
        progress = infer_post_train_gate_progress(
            progress,
            read_tail(FINAL_FORMAT_MARNIE_H10_LOG, 500_000),
            iteration_hint=int(loop_state.get("next_iteration") or 0),
        )
        updated = max(
            (
                path.stat().st_mtime
                for path in (
                    FINAL_FORMAT_MARNIE_H10_LOG,
                    FINAL_FORMAT_MARNIE_H10_PROGRESS_LOG,
                    FINAL_FORMAT_MARNIE_H10_PROGRESS_STATUS,
                )
                if path.is_file()
            ),
            default=None,
        )
        specialist = dict(
            (registry.get("specialists") or {}).get(
                "marnie-s-grimmsnarl-ex"
            )
            or {}
        )
        learner = dict(loop_state.get("learner") or {})
        checkpoint = learner.get("path") or specialist.get("initial_checkpoint")
        checkpoint_digest = (
            learner.get("digest") or specialist.get("initial_checkpoint_sha256")
        )
        if checkpoint_digest and not str(checkpoint_digest).startswith("sha256:"):
            checkpoint_digest = f"sha256:{checkpoint_digest}"
        structure = checkpoint_structure_telemetry(
            checkpoint,
            checkpoint_digest,
            cache_path=(
                ROOT
                / "outputs/state/dashboard-final-marnie-h10-live-structure-cache.json"
            ),
        )
        model_parameters = int(structure.get("model_parameters") or 0)
        if model_parameters <= 0:
            matches = re.findall(
                r"(?:model_params=|loaded checkpoint params=)(\d+)",
                read_tail(FINAL_FORMAT_MARNIE_H10_LOG, 200_000),
            )
            model_parameters = int(matches[-1]) if matches else 0
        trainer_args = [str(item) for item in registry.get("common_trainer_args") or []]
        remote_endpoints: list[str] = []
        try:
            endpoint_arg = trainer_args.index("--remote-worker-endpoints")
            remote_endpoints = [
                endpoint.strip()
                for endpoint in trainer_args[endpoint_arg + 1].split(",")
                if endpoint.strip()
            ]
        except (ValueError, IndexError):
            pass
        if rl_active:
            scheduler_queues = scheduler_queue_state(
                specialist.get("run_name")
                or "final_format_marnie_r104_h10_i_v6_8k",
                log_path=FINAL_FORMAT_MARNIE_H10_LOG,
            )
            scheduler_queues = scope_scheduler_queues_to_progress(
                progress,
                scheduler_queues,
            )
            drain_projection = result_drain_projection(progress, scheduler_queues)
        else:
            scheduler_queues = {
                "available": False,
                "mode": "stopped",
                "local": {"active_or_claimed": 0},
                "endpoints": {},
                "unassigned": 0,
                "results": {"waiting_ingest": 0},
            }
            drain_projection = {}
        progress_metrics = dict(progress.get("metrics") or {})
        progress_metrics.update(drain_projection.get("metrics") or {})
        last_phase = drain_projection.get(
            "phase", progress.get("stage") or "collect"
        )
        latest_line = drain_projection.get(
            "latest_line", progress.get("line")
        )
        if not rl_active:
            last_phase = f"stopped:{last_phase}"
            latest_line = f"STOPPED · last progress: {latest_line or '—'}"
        phase_fresh_window_s = (
            20 * 60
            if progress.get("stage") == "heldout:checkpoint_staging"
            else 30
        )
        return {
            "available": True,
            "authoritative": True,
            "source": str(FINAL_FORMAT_MARNIE_H10_PROGRESS_STATUS),
            "log": str(FINAL_FORMAT_MARNIE_H10_LOG),
            "latest_line": latest_line,
            "raw_latest_line": progress.get("line"),
            "updated_at": updated,
            "fresh": bool(
                rl_active
                and updated
                and time.time() - updated < phase_fresh_window_s
            ),
            "status": "running" if rl_active else "stopped",
            "mode": "final_format_marnie_h10_rl",
            "phase": last_phase,
            "run": specialist.get("run_name") or "final_format_marnie_r104_h10_i_v6_8k",
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "iteration": progress.get("iteration", loop_state.get("next_iteration")),
            "iterations_target": 21,
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "rate": progress.get("rate"),
            "rate_unit": progress.get("rate_unit"),
            "games_per_second": (
                drain_projection.get("games_per_second", progress.get("gps"))
                if rl_active
                else 0.0
            ),
            "samples_per_second": progress.get("sps") if rl_active else 0.0,
            "eta": progress.get("eta") if rl_active else None,
            "metrics": progress_metrics,
            "remote_workers": (
                drain_projection.get("remote_workers", progress.get("remotes"))
                if rl_active
                else 0
            ),
            "remote_endpoints": remote_endpoints,
            "scheduler_queues": scheduler_queues,
            "checkpoint": checkpoint,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_structure": structure,
            "model_parameters": model_parameters,
            "capacity_profile": validation.get("capacity_profile") or "H10-I/v1",
            "architecture": dict(validation.get("architecture") or {}),
            "learned_head_count": len(
                ((specialist.get("decision_fusion") or {}).get("required_heads") or [])
            ) or 19,
            "learned_route_count": 19,
            "decision_fusion_schema": (
                (specialist.get("decision_fusion") or {}).get("schema")
                or validation.get("decision_fusion_schema")
            ),
            "service": rl_service,
        }

    raw = read_tail(FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_LOG, 2_000_000)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if re.search(
            r"(?:train|val) ep\d+:.*?\d+/\d+"
            r"|expert rehearsal before iter\d+ ep\d+/\d+:.*?\d+/\d+",
            line,
        ):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "pack Blackwell corpus" in line or "expert-cpu-pack" in line:
                latest = line
                break
    if not latest and lines:
        latest = lines[-1]

    phase = "registered_rl" if rl_active else "loading"
    epoch = None
    current = None
    total = None
    percent = None
    batch_rate = None
    eta = None
    match = re.search(
        r"(train|val) ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
        latest,
    )
    if match:
        split, epoch_raw, percent_raw, current_raw, total_raw, timing = (
            match.groups()
        )
        phase = "training" if split == "train" else "validation"
        epoch = int(epoch_raw) + 1
        percent = float(percent_raw)
        current = int(current_raw)
        total = int(total_raw)
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            batch_rate = float(rate.group(1))
        eta_match = re.search(r"<([^,]+),", timing)
        if eta_match:
            eta = eta_match.group(1)
    else:
        rehearsal = re.search(
            r"expert rehearsal before iter(\d+) ep\d+/\d+:\s*"
            r"(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
            latest,
        )
        if rehearsal:
            epoch_raw, percent_raw, current_raw, total_raw, timing = (
                rehearsal.groups()
            )
            phase = "training"
            epoch = int(epoch_raw)
            percent = float(percent_raw)
            current = int(current_raw)
            total = int(total_raw)
            rate = re.search(r"([0-9.]+)batch/s", timing)
            if rate:
                batch_rate = float(rate.group(1))
            eta_match = re.search(r"<([^,]+),", timing)
            if eta_match:
                eta = eta_match.group(1)
    if phase == "loading" and "pack Blackwell corpus" in latest:
        phase = "packing"
        pack = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", latest)
        if pack:
            percent, current, total = (
                float(pack.group(1)),
                int(pack.group(2)),
                int(pack.group(3)),
            )
    elif phase == "loading" and ready:
        phase = "complete"
        percent = 100.0

    expert = read_json(FINAL_FORMAT_MARNIE_EXPERT)
    totals = dict(expert.get("totals") or {})
    corpus_games = int(totals.get("records_kept") or 0)
    corpus_decisions = int(totals.get("decisions_kept") or 0)
    checkpoint = validation.get("checkpoint")
    checkpoint_digest = validation.get("checkpoint_sha256")
    structure = checkpoint_structure_telemetry(
        checkpoint,
        checkpoint_digest,
        cache_path=(
            ROOT
            / "outputs/state/dashboard-final-marnie-h10-live-structure-cache.json"
        ),
    )
    updated = max(
        (
            path.stat().st_mtime
            for path in (
                FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_LOG,
                FINAL_FORMAT_MARNIE_H10_READY,
                FINAL_FORMAT_MARNIE_H10_VALIDATION,
            )
            if path.is_file()
        ),
        default=None,
    )
    service = rl_service if rl_active else bootstrap_service
    status = "running" if (bootstrap_active or rl_active) else "complete"
    return {
        "available": True,
        "authoritative": True,
        "source": str(
            FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_LOG
            if bootstrap_active
            else FINAL_FORMAT_MARNIE_H10_READY
        ),
        "log": str(FINAL_FORMAT_MARNIE_H10_BOOTSTRAP_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(status == "running" and updated and time.time() - updated < 30),
        "status": status,
        "mode": (
            "final_format_marnie_h10_rl"
            if rl_active
            else "final_format_marnie_h10_bootstrap"
        ),
        "phase": phase,
        "run": (
            "final_format_marnie_r104_h10_i_v6_8k"
            if rl_active
            else "final_format_marnie_r104_h10_bootstrap"
        ),
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "epoch": epoch,
        "epochs_target": 25,
        "current": current,
        "total": total,
        "percent": percent,
        "rate": batch_rate,
        "rate_unit": "batch/s" if batch_rate is not None else None,
        "batch_per_second": batch_rate,
        "eta": eta,
        "metrics": {
            name: parse_metric(latest, name)
            for name in (
                "acc", "loss", "p", "policy", "v", "value", "aux",
                "hand", "rem", "lethal", "prize", "guide", "step",
            )
        },
        "corpus_games": corpus_games,
        "corpus_decisions": corpus_decisions,
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_structure": structure,
        "model_parameters": int(structure.get("model_parameters") or 0),
        "capacity_profile": validation.get("capacity_profile") or "H10-I/v1",
        "architecture": dict(validation.get("architecture") or {}),
        "learned_head_count": int(validation.get("learned_head_count") or 19),
        "learned_route_count": int(validation.get("learned_route_count") or 19),
        "decision_fusion_schema": validation.get("decision_fusion_schema"),
        "service": service,
    }


def marnie_postupload_family_study_state() -> dict[str, Any]:
    """Project the managed status-75 boundary as the active pipeline phase.

    Iteration 9 intentionally stops the ordinary trainer before iteration-10
    collection.  The checksum-bound family study then owns the managed work
    while producing training-ineligible shadow evidence.  Treating the stopped
    trainer as the only possible live service makes this safe boundary look
    degraded even while its successor unit is healthy and making progress.
    """

    service = unit_state(MARNIE_POSTUPLOAD_FAMILY_STUDY_SERVICE, user=True)
    pause = read_json(MARNIE_POSTUPLOAD_PAUSE)
    trigger = read_json(MARNIE_ITERATION9_UPLOAD_TRIGGER)
    study = read_json(MARNIE_POSTUPLOAD_FAMILY_STUDY_ROOT / "study.json")
    active = bool(
        service.get("active")
        and int(service.get("pid") or 0) > 0
        and str(service.get("active_state") or "") in {"active", "activating"}
    )
    pause_digest = str(pause.get("learner_sha256") or "")
    trigger_digest = str(
        ((trigger.get("bindings") or {}).get("checkpoint") or {}).get("sha256")
        or ""
    )
    boundary_valid = bool(
        pause.get("schema") == "poke_bot.marnie_family_boundary_pause/v1"
        and int(pause.get("committed_iteration") or -1) == 9
        and int(pause.get("target_iteration") or -1) == 10
        and int(pause.get("restart_prevent_status") or -1) == 75
        and pause.get("next_collection_started") is False
        and trigger.get("schema")
        == "poke_bot.marnie_family_iteration9_upload_trigger/v1"
        and int(trigger.get("iteration") or -1) == 9
        and pause_digest
        and pause_digest == trigger_digest
    )
    paused_inconclusive = bool(
        boundary_valid
        and study.get("schema")
        == "poke_bot.marnie_archetype_family_shadow_study/v1"
        and study.get("status")
        == "failed_closed_inconclusive_after_two_rounds"
        and study.get("passed") is False
        and study.get("training_eligible") is False
        and study.get("replay_eligible") is False
        and len(study.get("rounds") or []) == 2
        and int(service.get("exit_status") or -1) == 76
        and not MARNIE_POSTUPLOAD_FAMILY_ACTIVATION_REQUEST.exists()
    )
    raw_log = read_tail(MARNIE_POSTUPLOAD_FAMILY_STUDY_LOG, 2_000_000)
    progress = parse_curriculum_progress("", raw_log, iteration_hint=9)
    study_round = progress.get("iteration")
    if str(progress.get("stage") or "").startswith("family-shadow:"):
        progress = {
            **progress,
            "iteration": 9,
            "metrics": {
                **(progress.get("metrics") or {}),
                "study_round": study_round,
                "training_eligible": False,
            },
        }
    updated = max(
        (
            path.stat().st_mtime
            for path in (
                MARNIE_POSTUPLOAD_FAMILY_STUDY_LOG,
                MARNIE_POSTUPLOAD_FAMILY_STUDY_ROOT
                / "sealed_training_rows.jsonl",
                MARNIE_POSTUPLOAD_FAMILY_STUDY_ROOT / "study.json",
            )
            if path.is_file()
        ),
        default=None,
    )
    progress_stage = str(progress.get("stage") or "")
    fresh = bool(
        active
        and boundary_valid
        and updated is not None
        and time.time() - updated <= 35.0
        # This isolated service owns a sequence of run-bound phases: sealed
        # shadow collection, policy training/validation, and locked/package
        # confirmation.  The ordinary progress parser intentionally names
        # those phases differently, so require a fresh recognized frame rather
        # than hard-coding only the initial ``family-shadow:*`` prefix.
        and bool(progress_stage)
    )
    current = bool(fresh or paused_inconclusive)
    if paused_inconclusive:
        progress_stage = "family-shadow:failed-closed-inconclusive"
        progress = {
            "line": "Two valid antithetic rounds were inconclusive; paused before iteration 10.",
            "stage": progress_stage,
            "iteration": 9,
            "current": 2,
            "total": 2,
            "percent": 100.0,
            "unit": "study rounds",
            "rate": None,
            "rate_unit": None,
            "eta": "owner-authorized passing activation boundary required",
            "metrics": {
                "study_status": study.get("status"),
                "training_eligible": False,
                "replay_eligible": False,
                "activation_request_created": False,
            },
        }
    return {
        "available": bool(active or pause or trigger),
        "active": active,
        "current": current,
        "paused": paused_inconclusive,
        "authoritative": boundary_valid,
        "status": (
            "running"
            if active
            else "paused_inconclusive"
            if paused_inconclusive
            else "waiting"
        ),
        "mode": "marnie_postupload_family_shadow_study",
        "phase": progress_stage or "family-shadow:starting",
        "run": "final_format_marnie_r104_h10_i_v6_8k",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "iteration": 9,
        "target_iteration": 10,
        "latest_line": progress.get("line"),
        "updated_at": updated,
        "source": str(MARNIE_POSTUPLOAD_FAMILY_STUDY_LOG),
        "outcome_source": str(
            MARNIE_POSTUPLOAD_FAMILY_STUDY_ROOT / "study.json"
        ),
        "progress": progress,
        "service": service,
        "boundary": {
            "valid": boundary_valid,
            "pause": str(MARNIE_POSTUPLOAD_PAUSE),
            "upload_trigger": str(MARNIE_ITERATION9_UPLOAD_TRIGGER),
            "checkpoint_digest": pause_digest or None,
            "trainer_exit_status": 75,
            "next_collection_started": pause.get("next_collection_started"),
        },
    }


def marnie_postupload_bootstrap_state() -> dict[str, Any]:
    """Project the activated family system's exact 25-epoch bootstrap."""

    service = unit_state(MARNIE_POSTUPLOAD_BOOTSTRAP_SERVICE, user=True)
    request = read_json(MARNIE_POSTUPLOAD_FAMILY_ACTIVATION_REQUEST)
    migration = read_json(MARNIE_POSTUPLOAD_FAMILY_MIGRATION)
    guide_shadow = read_json(MARNIE_GUIDE_SHADOW_NONAUTHORITY)
    guide_shadow_runtime = read_json(MARNIE_FAMILY_GUIDE_SHADOW_RUNTIME)
    epoch_recovery = read_json(MARNIE_EPOCH_RECOVERY)
    active = bool(
        str(service.get("active_state") or "") in {"active", "activating"}
        and int(service.get("pid") or 0) > 0
    )
    authoritative = bool(
        migration.get("schema") == "poke_bot.marnie_family_design_migration/v1"
        and migration.get("status") == "activated_atomically"
        and _file_sha256_matches(
            MARNIE_POSTUPLOAD_FAMILY_ACTIVATION_REQUEST,
            migration.get("request_sha256"),
        )
        and request.get("schema") == "poke_bot.marnie_family_activation_request/v1"
    )
    raw_log = read_tail(MARNIE_POSTUPLOAD_BOOTSTRAP_LOG, 2_000_000)
    progress = parse_curriculum_progress("", raw_log, iteration_hint=9)
    stage = str(progress.get("stage") or "")
    if not stage or stage.startswith("stopped:"):
        stage = (
            "bootstrap:expert-cpu-pack"
            if "[expert-cpu-pack]" in raw_log
            else "bootstrap:family-weighted-25-epoch"
        )
        pack_matches = re.findall(
            r"pack Blackwell corpus:\s*(\d+)%[^\r\n]*?(\d+)/(\d+)",
            raw_log,
        )
        pack_percent, pack_current, pack_total = (
            tuple(int(value) for value in pack_matches[-1])
            if pack_matches
            else (0, 0, 25)
        )
        progress = {
            "line": (
                raw_log.rstrip().splitlines()[-1]
                if raw_log.rstrip().splitlines()
                else "Starting exact family-weighted 25-epoch bootstrap."
            ),
            "stage": stage,
            "iteration": 9,
            "epoch": 0,
            "current": pack_current,
            "total": pack_total,
            "percent": float(pack_percent),
            "unit": "games" if pack_matches else "epochs",
            "rate": None,
            "rate_unit": None,
            "eta": "building exact expert pack" if "cpu-pack" in stage else None,
            "metrics": {
                "family_sampler_active": authoritative,
                "typed_family_loss_active": authoritative,
                "owner_ceiling_authority": True,
            },
        }
    progress = {
        **progress,
        "metrics": {
            **dict(progress.get("metrics") or {}),
            "guide": 0.0,
            "guide_status": "shadow_only_non_authoritative",
        },
    }
    # ``supervised_rehearsal_step`` uses its historical
    # ``before iterN`` label for the rehearsal counter.  In this managed
    # phase that counter is the 1..25 bootstrap epoch, not the pure-RL
    # iteration.  Keep the immutable raw line, but expose the two clocks
    # separately so the dashboard cannot imply that iterations 10+ have
    # already collected or committed.
    bootstrap_epoch = int(progress.get("iteration") or 0)
    if bootstrap_epoch > 0 and str(progress.get("stage") or "").startswith(
        "train:"
    ):
        epoch_percent = float(progress.get("percent") or 0.0)
        overall_percent = 100.0 * (
            (bootstrap_epoch - 1) + epoch_percent / 100.0
        ) / 25.0
        progress = {
            **progress,
            "iteration": 9,
            "epoch": bootstrap_epoch,
            "epochs": 25,
            "bootstrap_epoch": bootstrap_epoch,
            "bootstrap_epochs_target": 25,
            "bootstrap_epochs_completed": bootstrap_epoch - 1,
            "epoch_percent": epoch_percent,
            "percent": overall_percent,
            "rl_iteration": 9,
            "target_rl_iteration": 10,
        }
    guide_shadow_valid = bool(
        guide_shadow.get("schema")
        == "poke_bot.marnie_guide_shadow_non_authority/v1"
        and guide_shadow.get("status") == "active_nonblocking_shadow_only"
        and float(guide_shadow.get("guide_loss_weight", -1.0)) == 0.0
        and (guide_shadow.get("authority") or {}).get("blocking") is False
    )
    guide_shadow_runtime_valid = bool(
        guide_shadow_runtime.get("schema")
        == "poke_bot.marnie_family_guide_shadow_runtime/v1"
        and guide_shadow_runtime.get("status") == "active_next_start_overlay"
        and int(guide_shadow_runtime.get("owner_revision", -1)) == 142
        and (guide_shadow_runtime.get("proof") or {}).get("guide_weight") == 0.0
        and (guide_shadow_runtime.get("proof") or {}).get(
            "guide_runtime_authority"
        )
        is False
        and (guide_shadow_runtime.get("proof") or {}).get(
            "guide_blocking_authority"
        )
        is False
        and (guide_shadow_runtime.get("proof") or {}).get(
            "family_and_typed_loss_system_preserved"
        )
        is True
    )
    recovery_valid = bool(
        epoch_recovery.get("schema")
        == "poke_bot.marnie_postupload_epoch_recovery/v1"
        and epoch_recovery.get("status") == "validated_resume_without_retraining"
        and float(epoch_recovery.get("guide_weight", -1.0)) == 0.0
        and epoch_recovery.get("guide_enabled") is False
    )
    updated = max(
        (
            path.stat().st_mtime
            for path in (
                MARNIE_POSTUPLOAD_BOOTSTRAP_LOG,
                MARNIE_POSTUPLOAD_FAMILY_MIGRATION,
            )
            if path.is_file()
        ),
        default=None,
    )
    return {
        "available": bool(active or migration),
        "active": active,
        "current": bool(active and authoritative),
        "paused": False,
        "authoritative": authoritative,
        "status": "running" if active else "waiting",
        "mode": "marnie_postupload_family_weighted_bootstrap",
        "phase": str(progress.get("stage") or stage),
        "run": "marnie_r138_postupload_weighted_bootstrap",
        "specialist_id": "marnie-s-grimmsnarl-ex",
        "iteration": 9,
        "target_iteration": 10,
        "bootstrap_epoch": bootstrap_epoch,
        "bootstrap_epochs_target": 25,
        "bootstrap_epochs_completed": max(bootstrap_epoch - 1, 0),
        "latest_line": progress.get("line"),
        "updated_at": updated,
        "source": str(MARNIE_POSTUPLOAD_BOOTSTRAP_LOG),
        "outcome_source": str(MARNIE_POSTUPLOAD_FAMILY_MIGRATION),
        "progress": progress,
        "service": service,
        "guide": {
            "status": "shadow_only_non_authoritative",
            "enabled": False,
            "shadow_available": guide_shadow_valid,
            "runtime_merge_valid": guide_shadow_runtime_valid,
            "owner_revision": 142,
            "shadow_optional": True,
            "live_target_generation_enabled": False,
            "loss_weight": 0.0,
            "gradient_authority": False,
            "fusion_authority": False,
            "action_authority": False,
            "serving_authority": False,
            "gate_authority": False,
            "blocking_authority": False,
            "missing_shadow_behavior": "mark_unavailable_and_continue",
            "receipt": str(MARNIE_GUIDE_SHADOW_NONAUTHORITY),
            "runtime_receipt": str(MARNIE_FAMILY_GUIDE_SHADOW_RUNTIME),
            "runtime_registry": str(
                (guide_shadow_runtime.get("merged_registry") or {}).get("path")
                or ""
            ),
        },
        "epoch_recovery": {
            "valid": recovery_valid,
            "epoch": 1,
            "retrained": False,
            "receipt": str(MARNIE_EPOCH_RECOVERY),
        },
        "boundary": {
            "valid": authoritative,
            "migration": str(MARNIE_POSTUPLOAD_FAMILY_MIGRATION),
            "checkpoint_digest": str(
                (request.get("bindings") or {}).get("learner_sha256") or ""
            )
            or None,
            "trainer_exit_status": 75,
            "next_collection_started": False,
        },
    }


def marnie_shadow_guide_projection(
    postupload_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Return Marnie's durable guide-shadow state for every live RL phase.

    The post-upload bootstrap owns the checksum-backed guide authority facts,
    but those facts do not expire when the bootstrap service stops.  Marnie's
    later RL phases must continue to render the retired guide as shadow-only,
    weight zero, and nonblocking rather than collapsing that third state into
    the generic ``absent`` fallback.
    """

    guide = dict(postupload_boundary.get("guide") or {})
    if (
        guide.get("status") != "shadow_only_non_authoritative"
        or guide.get("enabled") is not False
        or float(guide.get("loss_weight", -1.0)) != 0.0
        or guide.get("blocking_authority") is not False
        or guide.get("action_authority") is not False
        or guide.get("gradient_authority") is not False
    ):
        return {}
    return {
        **guide,
        "active_specialist": "marnie-s-grimmsnarl-ex",
        "guide_archetype": "marnie-s-grimmsnarl-ex",
        "parameterized_head": False,
    }


def final_format_alakazam_progress() -> dict[str, Any]:
    """Return the receipt-bound live state of the revision-79 refresh."""

    h10_service = unit_state(FINAL_FORMAT_ALAKAZAM_H10_SERVICE, user=True)
    h10_active = bool(
        h10_service.get("active")
        and (
            int(h10_service.get("pid") or 0) > 0
            or h10_service.get("sub_state") == "running"
        )
    )
    if h10_active:
        status = read_tail(FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_STATUS, 20_000)
        progress_log = read_tail(FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_LOG, 2_000_000)
        loop_state = read_json(FINAL_FORMAT_ALAKAZAM_H10_RUN_DIR / "loop_state.json")
        # Resolve the registry selected by the live managed unit. The final
        # refresh can temporarily use a receipt-bound fleet registry (for
        # example, Elmo-only while Bert runs an isolated Apple benchmark), so
        # the static preparation default is not always runtime truth.
        registry_path = FINAL_FORMAT_ALAKAZAM_H10_REGISTRY
        service_command = " ".join(
            str(h10_service.get(key) or "")
            for key in ("command", "exec_start")
        )
        registry_match = re.search(r'--registry\s+([^\s;}]+)', service_command)
        if registry_match:
            candidate = Path(registry_match.group(1))
            if candidate.is_file():
                registry_path = candidate
        registry = read_json(registry_path)
        progress = parse_curriculum_progress(
            status,
            progress_log,
            iteration_hint=int(loop_state.get("next_iteration") or 0),
        )
        updated = max(
            (
                path.stat().st_mtime
                for path in (
                    FINAL_FORMAT_ALAKAZAM_H10_LOG,
                    FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_LOG,
                    FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_STATUS,
                )
                if path.is_file()
            ),
            default=None,
        )
        isolated = dict(registry.get("isolated_refresh_contract") or {})
        specialist = dict((registry.get("specialists") or {}).get("alakazam") or {})
        live_learner = dict(loop_state.get("learner") or {})
        checkpoint = (
            live_learner.get("path") or specialist.get("initial_checkpoint")
        )
        checkpoint_digest = (
            live_learner.get("digest")
            or specialist.get("initial_checkpoint_sha256")
        )
        checkpoint_digest = (
            f"sha256:{checkpoint_digest}"
            if checkpoint_digest
            and not str(checkpoint_digest).startswith("sha256:")
            else checkpoint_digest
        )
        checkpoint_structure = checkpoint_structure_telemetry(
            checkpoint,
            checkpoint_digest,
            cache_path=(
                ROOT
                / "outputs/state/dashboard-final-alakazam-h10-live-structure-cache.json"
            ),
        )
        model_params = None
        log_tail = read_tail(FINAL_FORMAT_ALAKAZAM_H10_LOG, 200_000)
        matches = re.findall(
            r"(?:model_params=|loaded checkpoint params=)(\d+)",
            log_tail,
        )
        if matches:
            model_params = int(matches[-1])
        # Startup parameter lines can age out of the bounded live log after a
        # long iteration.  The immutable mix receipt independently binds the
        # selected H10 checkpoint to its benchmarked parameter count.
        if model_params is None:
            capacity = read_json(FINAL_FORMAT_ALAKAZAM_H10_CAPACITY_RECEIPT)
            benchmark = dict(capacity.get("bert_apple_device_check") or {})
            selected_digest = str(specialist.get("initial_checkpoint_sha256") or "")
            selected_digest = (
                selected_digest
                if selected_digest.startswith("sha256:")
                else f"sha256:{selected_digest}"
            )
            receipt_params = int(benchmark.get("model_parameters") or 0)
            if (
                capacity.get("schema")
                == "poke_bot.final_format_alakazam_h10_mix_activation/v1"
                and capacity.get("status") == "activated"
                and capacity.get("run_name")
                == "final_format_alakazam_r79_h10_i_v6_8k"
                and capacity.get("checkpoint_sha256") == selected_digest
                and receipt_params > 0
            ):
                model_params = receipt_params
        if checkpoint_structure.get("verified") is True:
            model_params = int(checkpoint_structure.get("model_parameters") or 0)
        trainer_args = [
            str(item) for item in registry.get("common_trainer_args") or []
        ]
        # The managed command is runtime truth for a receipt-backed terminal
        # ceiling.  Preparation registries can retain an older broad safety
        # horizon after a safe boundary migration (for example 189 while the
        # live process is hard-stopped after zero-indexed iteration 20).
        # Never let that stale preparation value reset the dashboard counter.
        live_iterations_target: int | None = None
        iterations_match = re.search(r"--iterations\s+(\d+)", service_command)
        if iterations_match:
            live_iterations_target = int(iterations_match.group(1))
        if live_iterations_target is None:
            try:
                iterations_arg = trainer_args.index("--iterations")
                live_iterations_target = int(trainer_args[iterations_arg + 1])
            except (ValueError, IndexError, TypeError):
                pass
        if live_iterations_target is None:
            exact_ceiling = registry.get("iteration_ceiling")
            if exact_ceiling is None:
                exact_ceiling = specialist.get("iteration_ceiling")
            if exact_ceiling is not None:
                # The registry ceiling is zero-indexed; the UI schedule is a
                # count, so terminal iter_00020 means 21 total iterations.
                live_iterations_target = int(exact_ceiling) + 1
        remote_endpoints: list[str] = []
        try:
            endpoint_arg = trainer_args.index("--remote-worker-endpoints")
            remote_endpoints = [
                endpoint.strip()
                for endpoint in trainer_args[endpoint_arg + 1].split(",")
                if endpoint.strip()
            ]
        except (ValueError, IndexError):
            pass
        scheduler_queues = scheduler_queue_state(
            specialist.get("run_name")
            or "final_format_alakazam_r79_h10_i_v6_8k",
            log_path=FINAL_FORMAT_ALAKAZAM_H10_LOG,
        )
        scheduler_queues = scope_scheduler_queues_to_progress(
            progress,
            scheduler_queues,
        )
        return {
            "authoritative": True,
            "source": str(FINAL_FORMAT_ALAKAZAM_H10_PROGRESS_STATUS),
            "log": str(FINAL_FORMAT_ALAKAZAM_H10_LOG),
            "latest_line": progress.get("line"),
            "updated_at": updated,
            "fresh": bool(updated and time.time() - updated < 30),
            "status": "running",
            "mode": "final_format_alakazam_h10_rl",
            "phase": progress.get("stage") or "collect",
            "run": specialist.get("run_name") or "final_format_alakazam_r79_h10_i_v6_8k",
            "specialist_id": "alakazam",
            "iteration": progress.get("iteration", loop_state.get("next_iteration")),
            "iterations_target": int(
                live_iterations_target
                or isolated.get("maximum_iterations")
                or 21
            ),
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "rate": progress.get("rate"),
            "rate_unit": progress.get("rate_unit"),
            "games_per_second": progress.get("gps"),
            "samples_per_second": progress.get("sps"),
            "eta": progress.get("eta"),
            "metrics": progress.get("metrics") or {},
            "games_per_iteration": int(isolated.get("games_per_iteration") or 16384),
            "remote_workers": progress.get("remotes"),
            "remote_endpoints": remote_endpoints,
            "scheduler_queues": scheduler_queues,
            "runtime_registry": str(registry_path),
            "training_seat_split": isolated.get("training_seat_split"),
            "blackwell_workers": int(isolated.get("blackwell_workers") or 96),
            "elmo_workers": int(isolated.get("elmo_workers") or 36),
            "bert_workers": int(isolated.get("bert_workers") or 16),
            "model_parameters": model_params,
            "model_profile_id": "H10-I/v1",
            "checkpoint": checkpoint,
            "checkpoint_digest": checkpoint_digest,
            "checkpoint_structure": checkpoint_structure,
            "decision_fusion": dict(
                checkpoint_structure.get("decision_fusion") or {}
            ),
            "expanded_head_training": dict(
                checkpoint_structure.get("expanded_head_training") or {}
            ),
            "learned_head_count": len(
                ((specialist.get("decision_fusion") or {}).get("required_heads") or [])
            ),
            "matchup_router_format": 6,
            "premium_strength_gate": float(
                isolated.get("premium_skill_weighted_win_rate") or 0.65
            ),
            "kaggle_rating_lower_bound": float(
                isolated.get("kaggle_rating_simulation_projected_lower_bound") or 1150
            ),
            "rating_gate_separate": bool(
                isolated.get("strength_gate_and_rating_simulation_are_independent")
            ),
            "architecture_stage": "h10_router6_high_volume_rl",
            "guide_runtime_route_count": 0,
            "guide_imitation_weight": 0.0,
            "service": h10_service,
        }

    service = unit_state(FINAL_FORMAT_ALAKAZAM_SERVICE, user=True)
    raw = read_tail(FINAL_FORMAT_ALAKAZAM_LOG, 2_000_000)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if re.search(r"expert rehearsal before iter\d+ ep\d+/\d+:.*?\d+/\d+", line):
            latest = line
            break
    if not latest:
        latest = lines[-1] if lines else ""

    epoch = current = total = None
    epoch_percent = overall_percent = batch_rate = None
    eta = None
    phase = "loading"
    match = re.search(
        r"expert rehearsal before iter(\d+) ep\d+/\d+:\s*"
        r"(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
        latest,
    )
    if match:
        epoch_raw, percent_raw, current_raw, total_raw, timing = match.groups()
        epoch = int(epoch_raw)
        epoch_percent = float(percent_raw)
        current = int(current_raw)
        total = int(total_raw)
        overall_percent = 100.0 * (
            (epoch - 1) + (current / total if total else 0.0)
        ) / 25.0
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            batch_rate = float(rate.group(1))
        eta_match = re.search(r"<([^,]+),", timing)
        if eta_match:
            eta = eta_match.group(1)
        phase = "training"
    elif "pack Blackwell corpus" in latest or "cpu-pack" in latest.lower():
        phase = "packing"

    updated = None
    try:
        updated = FINAL_FORMAT_ALAKAZAM_LOG.stat().st_mtime
    except OSError:
        pass
    ready = FINAL_FORMAT_ALAKAZAM_READY.is_file()
    active = bool(
        service.get("active")
        and (
            int(service.get("pid") or 0) > 0
            or service.get("sub_state") == "running"
        )
    )
    metrics = {
        name: parse_metric(latest, name)
        for name in (
            "acc", "loss", "policy", "value", "aux", "hand", "rem",
            "lethal", "prize", "guide", "step",
        )
    }
    return {
        "authoritative": True,
        "source": str(FINAL_FORMAT_ALAKAZAM_LOG),
        "log": str(FINAL_FORMAT_ALAKAZAM_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(active and updated and time.time() - updated < 30),
        "status": "complete" if ready else "running" if active else "waiting",
        "mode": "final_format_alakazam_ordinary_refresh",
        "phase": "complete" if ready else phase,
        "run": "final_format_alakazam_r79",
        "specialist_id": "alakazam",
        "epoch": epoch,
        "epochs_target": 25,
        "current": current,
        "total": total,
        "percent": 100.0 if ready else overall_percent,
        "epoch_percent": epoch_percent,
        "batch_per_second": batch_rate,
        "rate": batch_rate,
        "rate_unit": "batch/s" if batch_rate is not None else None,
        "eta": eta,
        "metrics": metrics,
        "corpus_games": 64_411,
        "corpus_decisions": 5_152_754,
        "corpus_manifest_digest": (
            "sha256:3836852129511fdffd2767f6701dcc562d1723bb0c345c3cf5068ad9774b9acb"
        ),
        "guide_runtime_route_count": 0,
        "guide_imitation_weight": 0.0,
        "architecture_stage": "ordinary_core9_refresh_before_h10_migration",
        "h10_step_zero_canary_passed": True,
        "service": service,
    }


def final_format_alakazam_model_inventory() -> dict[str, Any]:
    """Project the live ordinary checkpoint and validated H10 target."""

    inventory = read_json(FINAL_FORMAT_ALAKAZAM_MODEL_INVENTORY)
    state = read_json(FINAL_FORMAT_ALAKAZAM_STATE)
    history = state.get("history")
    latest = (
        history[-1]
        if isinstance(history, list) and history and isinstance(history[-1], dict)
        else {}
    )
    if (
        inventory.get("schema")
        != "poke_bot.final_format_alakazam_model_inventory/v1"
        or inventory.get("specialist_id") != "alakazam"
    ):
        return {
            "available": False,
            "reason": "final-format Alakazam model inventory receipt is absent",
            "source": str(FINAL_FORMAT_ALAKAZAM_MODEL_INVENTORY),
        }
    ordinary = dict(inventory.get("ordinary_refresh") or {})
    ordinary.update(
        {
            "latest_checkpoint": latest.get("checkpoint"),
            "latest_checkpoint_sha256": latest.get("checkpoint_digest"),
            "latest_completed_epoch": latest.get("epoch"),
            "epochs_target": int(state.get("epochs_max") or 25),
            "seed_checkpoint": state.get("hot_start_checkpoint"),
            "seed_checkpoint_sha256": state.get(
                "hot_start_checkpoint_digest"
            ),
            "validation_loss": latest.get("validation_loss"),
            "validation_accuracy": latest.get("validation_accuracy"),
            "expanded_head_training": latest.get(
                "expanded_head_training"
            ),
            "training_active": unit_state(
                FINAL_FORMAT_ALAKAZAM_SERVICE, user=True
            ).get("active")
            is True,
        }
    )
    return {
        **inventory,
        "available": True,
        "ordinary_refresh": ordinary,
        "source": str(FINAL_FORMAT_ALAKAZAM_MODEL_INVENTORY),
        "live_state_source": str(FINAL_FORMAT_ALAKAZAM_STATE),
    }


def exact_training_state() -> dict[str, Any]:
    """Return the authoritative active exact-replay training state."""
    candidates = [
        (EXACT_RESIDENT_STATUS, read_json(EXACT_RESIDENT_STATUS)),
        (EXACT_STREAM_STATUS, read_json(EXACT_STREAM_STATUS)),
    ]
    candidates = [row for row in candidates if row[1]]
    if not candidates:
        legacy = bootstrap_progress()
        legacy.update(authoritative=False, source=str(BOOTSTRAP_LOG))
        return legacy
    status_path, status = max(
        candidates,
        key=lambda row: float(row[1].get("updated_unix") or 0),
    )
    raw = ANSI_RE.sub("", read_tail(EXACT_LOG)).replace("\r", "\n")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    latest = lines[-1] if lines else ""
    resident = "resident" in str(status.get("mode") or "")
    phase = str(status.get("phase") or "training")
    current = status.get("current")
    total = status.get("total")
    percent = status.get("percent")
    if resident and phase == "packing":
        packing = next(
            (line for line in reversed(lines) if "pack exact Blackwell corpus" in line),
            "",
        )
        match = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", packing)
        if match:
            percent, current, total = (
                float(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            latest = packing
    if not resident:
        current = status.get("shard_cursor")
        total = status.get("train_shards")
        percent = (
            100.0 * float(current) / float(total)
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total
            else None
        )
    raw_metrics = status.get("metrics") or status.get("last_train") or {}
    corpus_manifest = read_json(EXACT_ROOT / "manifest.json")
    corpus_totals = corpus_manifest.get("totals") or {}
    corpus_games = corpus_totals.get("games")
    corpus_records = corpus_totals.get("records")
    batch_rate = status.get("batch_per_second")
    batch_size = status.get("batch_size")
    samples_per_second = (
        float(batch_rate) * float(batch_size)
        if isinstance(batch_rate, (int, float))
        and isinstance(batch_size, (int, float))
        else None
    )
    total_samples = (
        int(status.get("train_samples") or 0)
        + int(status.get("val_samples") or 0)
    )
    game_equivalents_per_second = (
        samples_per_second * float(corpus_games) / total_samples
        if samples_per_second is not None
        and isinstance(corpus_games, (int, float))
        and total_samples > 0
        else None
    )
    acting_sequences_per_second = (
        samples_per_second * float(corpus_records) / total_samples
        if samples_per_second is not None
        and isinstance(corpus_records, (int, float))
        and total_samples > 0
        else None
    )
    metrics = {
        "acc": (
            100.0 * float(raw_metrics["policy_accuracy"])
            if isinstance(raw_metrics.get("policy_accuracy"), (int, float))
            else None
        ),
        "loss": raw_metrics.get("total") or raw_metrics.get("objective"),
        "p": raw_metrics.get("policy"),
        "v": raw_metrics.get("value"),
        "aux": raw_metrics.get("aux"),
        "hand": raw_metrics.get("hand"),
        "remainder": raw_metrics.get("remainder"),
        "lethal": raw_metrics.get("lethal"),
        "prize_race": raw_metrics.get("prize_race"),
        "step": status.get("global_step"),
    }

    status_updated = float(status.get("updated_unix") or 0)
    try:
        log_updated = EXACT_LOG.stat().st_mtime
    except OSError:
        log_updated = 0.0
    updated = max(status_updated, log_updated) or None
    eta_seconds = status.get("eta_seconds")
    return {
        "authoritative": True,
        "source": str(status_path),
        "log": str(EXACT_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(updated and time.time() - updated < 30),
        "status": status.get("status"),
        "mode": status.get("mode"),
        "phase": phase,
        "epoch": status.get("total_epoch_index"),
        "epochs_target": status.get("epochs_target", 26),
        "current": current,
        "total": total,
        "percent": percent,
        "batch_per_second": batch_rate,
        "samples_per_second": samples_per_second,
        "game_equivalents_per_second": game_equivalents_per_second,
        "acting_sequences_per_second": acting_sequences_per_second,
        "corpus_games": corpus_games,
        "corpus_records": corpus_records,
        "eta_seconds": eta_seconds,
        "eta": (
            time.strftime("%H:%M:%S", time.gmtime(float(eta_seconds)))
            if isinstance(eta_seconds, (int, float))
            else None
        ),
        "metrics": metrics,
        "all_training_tensors_device_resident": status.get(
            "all_training_tensors_device_resident", False
        ),
        "resident_bytes": status.get("resident_bytes"),
        "train_samples": status.get("train_samples"),
        "val_samples": status.get("val_samples"),
        "batch_size": batch_size,
        "gpu_name": status.get("gpu_name"),
    }


def baseline_eval_state() -> dict[str, Any]:
    candidates = sorted(
        (ROOT / "outputs/eval").glob(
            "state_core_resident_epoch*_official_core17.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return {"available": False}
    path = candidates[-1]
    report = read_json(path)
    pooled = report.get("pooled_formal") or {}
    checkpoint_info = report.get("checkpoint") or {}
    matchups = [
        {
            "opponent_id": row.get("opponent_id"),
            "games": row.get("games"),
            "wr": row.get("wr"),
            "lower": (row.get("draw_aware_score_interval") or {}).get("lower"),
        }
        for row in report.get("matchups") or []
    ]
    return {
        "available": True,
        "source": str(path),
        "updated_at": path.stat().st_mtime,
        "valid": report.get("valid"),
        "passed": report.get("all_pass"),
        "promotion_eligible": report.get("promotion_eligible"),
        "games": pooled.get("games"),
        "wr": pooled.get("wr"),
        "lower": pooled.get("interval_lower"),
        "upper": pooled.get("interval_upper"),
        "checkpoint": checkpoint_info.get("path"),
        "checkpoint_digest": checkpoint_info.get("digest"),
        "matchups": matchups,
        "deck_count": (report.get("deck_agnostic_gate") or {}).get("deck_count"),
    }


def committed_official_heldout_state(
    loop: dict[str, Any],
    run_dir: Path | None,
    *,
    global_iteration_offset: int = 0,
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the best append-only, exact official heldout result.

    The curriculum's ``metrics/latest.json`` describes the latest candidate,
    which may be worse than the protected heldout champion.  The Official
    Gateline card instead follows ``heldout_champion_evidence`` and reconciles
    it to the matching committed history row before exposing matchup results.
    This keeps a live/partial heldout wave from replacing audited evidence.
    """

    evidence = loop.get("heldout_champion_evidence")
    identity = loop.get("heldout_champion")
    if not isinstance(evidence, dict) or not isinstance(identity, dict):
        inherited = (
            handoff.get("inherited_official_heldout")
            if isinstance(handoff, dict)
            else None
        )
        inherited_identity = (
            loop.get("heldout_champion") or loop.get("champion")
            if isinstance(loop, dict)
            else None
        )
        if not isinstance(inherited, dict) or not isinstance(inherited_identity, dict):
            return {"available": False}
        digest = str(inherited.get("checkpoint_digest") or "")
        if not digest or digest != str(inherited_identity.get("digest") or ""):
            return {"available": False, "reason": "inherited heldout identity mismatch"}
        audit = inherited.get("audit")
        gate_matchups = inherited.get("per_opponent")
        audit_matchups = audit.get("per_opponent") if isinstance(audit, dict) else None
        games = int(inherited.get("games") or 0)
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("exact_distribution") is not True
            or audit.get("exact_weights") is not True
            or audit.get("greedy_required") is not True
            or int(audit.get("valid_games") or 0) != games
            or not isinstance(gate_matchups, dict)
            or not isinstance(audit_matchups, dict)
        ):
            return {"available": False, "reason": "inherited heldout audit mismatch"}
        matchups: list[dict[str, Any]] = []
        for opponent_id, row in gate_matchups.items():
            if not isinstance(row, dict):
                return {"available": False, "reason": "inherited heldout matchup mismatch"}
            seats = audit_matchups.get(opponent_id)
            if not isinstance(seats, dict):
                return {"available": False, "reason": "inherited heldout matchup mismatch"}
            matchup_games = int(row.get("games") or 0)
            seat0 = int(seats.get("seat0") or row.get("seat0_games") or 0)
            seat1 = int(seats.get("seat1") or row.get("seat1_games") or 0)
            if matchup_games <= 0 or seat0 + seat1 != matchup_games:
                return {"available": False, "reason": "inherited heldout seat mismatch"}
            matchups.append(
                {
                    "opponent_id": str(opponent_id),
                    "games": matchup_games,
                    "wr": as_float(row.get("win_rate")),
                    "wins": as_float(row.get("wins")),
                    "draws": as_float(row.get("draws")),
                    "losses": as_float(row.get("losses")),
                    "seat0": seat0,
                    "seat1": seat1,
                }
            )
        if sum(int(row["games"]) for row in matchups) != games:
            return {"available": False, "reason": "inherited heldout game mismatch"}
        matchups.sort(key=lambda row: str(row["opponent_id"]))
        lineage_iteration = int(inherited.get("lineage_iteration") or 0)
        display_iteration = inherited.get("iteration")
        if not isinstance(display_iteration, int):
            source_offset = int((handoff or {}).get("source_global_iteration_offset") or 0)
            display_iteration = source_offset + lineage_iteration
        return {
            "available": True,
            "kind": "inherited_official_heldout_champion",
            "valid": True,
            "passed": inherited.get("passed") is True,
            "reason": inherited.get("reason"),
            "games": games,
            "wr": as_float(inherited.get("wr")),
            "lower": as_float(inherited.get("lower")),
            "upper": as_float(inherited.get("upper")),
            "iteration": int(display_iteration),
            "lineage_iteration": lineage_iteration,
            "checkpoint": inherited_identity.get("path"),
            "checkpoint_digest": digest,
            "matchups": matchups,
            "opponent_count": len(matchups),
            "audit_passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "source": f"lineage handoff from {(handoff or {}).get('source_run') or 'prior run'}",
            "updated_at": None,
        }

    iteration = evidence.get("iteration")
    digest = str(evidence.get("checkpoint_digest") or "")
    identity_digest = str(identity.get("digest") or "")
    if not isinstance(iteration, int) or not digest or digest != identity_digest:
        return {"available": False, "reason": "heldout champion identity mismatch"}

    matching: dict[str, Any] | None = None
    for row in reversed(loop.get("history") or []):
        if not isinstance(row, dict) or row.get("iteration") != iteration:
            continue
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        if str(candidate.get("digest") or "") == digest:
            matching = row
            break
    if matching is None:
        audit = evidence.get("audit")
        report_identity = audit.get("report") if isinstance(audit, dict) else None
        if (
            iteration == -1
            and isinstance(audit, dict)
            and audit.get("passed") is True
            and audit.get("source") == "trusted_external_new_lineage_anchor"
            and audit.get("terminal_gate_eligible") is False
            and isinstance(report_identity, dict)
        ):
            report_path = Path(str(report_identity.get("path") or ""))
            try:
                payload = report_path.read_bytes()
                report_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                report = json.loads(payload)
            except (OSError, ValueError, TypeError):
                return {"available": False, "reason": "seed audit report is unreadable"}
            expected_digest = str(report_identity.get("digest") or "")
            checkpoint = report.get("checkpoint")
            pooled = report.get("pooled_formal")
            deck_gate = report.get("deck_agnostic_gate")
            report_matchups = report.get("matchups")
            evidence_matchups = evidence.get("per_opponent")
            if (
                report_digest != expected_digest
                or not isinstance(checkpoint, dict)
                or str(checkpoint.get("digest") or "") != digest
                or report.get("valid") is not True
                or report.get("trusted_formal") is not True
                or report.get("formal_mode") != "policy"
                or list(report.get("failures") or [])
                or not isinstance(pooled, dict)
                or not isinstance(deck_gate, dict)
                or deck_gate.get("exact_deck_seat_balance") is not True
                or not isinstance(report_matchups, list)
                or not isinstance(evidence_matchups, dict)
            ):
                return {"available": False, "reason": "seed audit report failed reconciliation"}
            games = int(evidence.get("games") or 0)
            pooled_wr = as_float(pooled.get("wr"))
            evidence_wr = as_float(evidence.get("win_rate"))
            if (
                games <= 0
                or int(report.get("scheduled_jobs") or 0) != games
                or int(report.get("completed_jobs") or 0) != games
                or int(pooled.get("games") or 0) != games
                or pooled_wr is None
                or evidence_wr is None
                or abs(pooled_wr - evidence_wr) > 1e-12
            ):
                return {"available": False, "reason": "seed audit totals mismatch"}
            by_id = {
                str(row.get("opponent_id") or ""): row
                for row in report_matchups
                if isinstance(row, dict)
            }
            if set(by_id) != set(OFFICIAL_BASELINE_IDS):
                return {"available": False, "reason": "seed audit opponent mismatch"}
            matchups: list[dict[str, Any]] = []
            for opponent_id in OFFICIAL_BASELINE_IDS:
                row = by_id[opponent_id]
                anchored = evidence_matchups.get(opponent_id)
                if not isinstance(anchored, dict):
                    return {"available": False, "reason": "seed audit matchup missing"}
                matchup_games = int(row.get("games") or 0)
                wins = as_float(row.get("wins"))
                draws = as_float(row.get("draws"))
                anchored_wins = as_float(anchored.get("wins"))
                if wins is None or draws is None or anchored_wins is None:
                    return {"available": False, "reason": "seed audit matchup mismatch"}
                score = wins + 0.5 * draws
                if (
                    matchup_games <= 0
                    or matchup_games % 2
                    or int(anchored.get("games") or 0) != matchup_games
                    or abs(anchored_wins - score) > 1e-12
                ):
                    return {"available": False, "reason": "seed audit matchup mismatch"}
                matchups.append(
                    {
                        "opponent_id": opponent_id,
                        "games": matchup_games,
                        "wr": score / matchup_games,
                        "wins": score,
                        "draws": draws,
                        "losses": float(row.get("losses") or 0.0),
                        "seat0": matchup_games // 2,
                        "seat1": matchup_games // 2,
                    }
                )
            return {
                "available": True,
                "kind": "external_seed_official_heldout_anchor",
                "valid": True,
                "passed": False,
                "reason": "nonterminal_seed_audit",
                "games": games,
                "wr": as_float(evidence.get("win_rate")),
                "lower": as_float(evidence.get("confidence_lower")),
                "upper": as_float(evidence.get("confidence_upper")),
                "iteration": -1,
                "lineage_iteration": -1,
                "checkpoint": identity.get("path"),
                "checkpoint_digest": digest,
                "matchups": matchups,
                "opponent_count": len(matchups),
                "audit_passed": True,
                "exact_distribution": True,
                "exact_weights": True,
                "greedy_required": True,
                "terminal_gate_eligible": False,
                "source": str(report_path),
                "updated_at": report_path.stat().st_mtime,
            }
        return {"available": False, "reason": "heldout champion commit is missing"}

    gate = matching.get("raw_heldout_gate")
    audit = matching.get("heldout_audit")
    if not isinstance(gate, dict) or not isinstance(audit, dict):
        return {"available": False, "reason": "heldout gate or audit is missing"}

    games = int(evidence.get("games") or 0)
    audit_games = int(audit.get("valid_games") or 0)
    gate_games = int(gate.get("games") or 0)
    evidence_wr = as_float(evidence.get("win_rate"))
    # ``poke_bot.public_agent_gate_result/v1`` stores the aggregate as
    # ``skill_weighted_wr``.  Earlier loop-state rows used ``win_rate``.
    gate_wr = as_float(
        gate.get("win_rate")
        if gate.get("win_rate") is not None
        else gate.get("skill_weighted_wr")
    )
    audit_passed = audit.get("passed") is True
    reconciled = bool(
        audit_passed
        and games > 0
        and games == audit_games == gate_games
        and evidence_wr is not None
        and gate_wr is not None
        and abs(evidence_wr - gate_wr) < 1e-12
        and str(audit.get("checkpoint_digest") or "") == digest
    )
    if not reconciled:
        return {"available": False, "reason": "heldout evidence failed reconciliation"}

    audit_matchups = audit.get("per_opponent")
    gate_matchups = gate.get("per_opponent")
    # Gate-result schema v1 stores matchup evidence as a list whose rate field
    # is ``wr``.  Normalize that immutable schema to the legacy mapping shape
    # before the existing audit reconciliation and dashboard projection.
    if not isinstance(gate_matchups, dict):
        matchup_rows = gate.get("matchups")
        if isinstance(matchup_rows, list):
            gate_matchups = {
                str(row.get("opponent_id") or ""): {
                    **row,
                    "win_rate": (
                        row.get("win_rate")
                        if row.get("win_rate") is not None
                        else row.get("wr")
                    ),
                }
                for row in matchup_rows
                if isinstance(row, dict) and str(row.get("opponent_id") or "")
            }
    if not isinstance(audit_matchups, dict) or not isinstance(gate_matchups, dict):
        return {"available": False, "reason": "heldout matchup evidence is missing"}
    matchups: list[dict[str, Any]] = []
    for opponent_id, row in gate_matchups.items():
        if not isinstance(row, dict):
            continue
        seats = audit_matchups.get(opponent_id)
        seats = seats if isinstance(seats, dict) else {}
        matchups.append(
            {
                "opponent_id": str(opponent_id),
                "games": int(row.get("games") or 0),
                "wr": as_float(row.get("win_rate")),
                "wins": as_float(row.get("wins")),
                "draws": as_float(row.get("draws")),
                "losses": as_float(row.get("losses")),
                "seat0": int(seats.get("seat0") or row.get("seat0_games") or 0),
                "seat1": int(seats.get("seat1") or row.get("seat1_games") or 0),
            }
        )
    matchups.sort(key=lambda row: str(row["opponent_id"]))

    source = (
        run_dir / "commits" / f"iter_{iteration:05d}.json"
        if run_dir is not None
        else None
    )
    if source is None or not source.is_file():
        source = run_dir / "loop_state.json" if run_dir is not None else None
    return {
        "available": True,
        "kind": "official_heldout_champion",
        "valid": True,
        "passed": gate.get("passed") is True,
        "reason": gate.get("reason"),
        "games": games,
        "wr": evidence_wr,
        "lower": as_float(evidence.get("confidence_lower")),
        "upper": as_float(evidence.get("confidence_upper")),
        "iteration": int(iteration) + int(global_iteration_offset),
        "lineage_iteration": int(iteration),
        "checkpoint": identity.get("path"),
        "checkpoint_digest": digest,
        "matchups": matchups,
        "opponent_count": len(matchups),
        "audit_passed": True,
        "exact_distribution": audit.get("exact_distribution") is True,
        "exact_weights": audit.get("exact_weights") is True,
        "greedy_required": audit.get("greedy_required") is True,
        "matchup_runtime": dict(audit.get("matchup_runtime") or {}),
        "source": str(source) if source is not None else None,
        "updated_at": source.stat().st_mtime if source is not None and source.is_file() else None,
    }


def latest_committed_official_heldout_state(
    loop: dict[str, Any],
    run_dir: Path | None,
    *,
    global_iteration_offset: int = 0,
) -> dict[str, Any]:
    """Return the newest fully audited official-baseline holdout attempt.

    This is deliberately separate from :func:`committed_official_heldout_state`.
    The latter tracks the protected best checkpoint; this view answers whether
    the most recent candidate actually ran its exact holdout, even when that
    candidate was rejected and the protected checkpoint therefore stayed put.
    """

    if run_dir is None:
        return {"available": False, "reason": "run directory is unavailable"}
    heldout_identity = (
        loop.get("heldout_champion")
        if isinstance(loop.get("heldout_champion"), dict)
        else {}
    )
    heldout_digest = str(heldout_identity.get("digest") or "")

    # Research controls became a dedicated, non-training measurement phase.
    # Prefer its newest immutable iteration-bound artifact over the retired
    # raw_heldout_gate shape, otherwise this panel can remain stuck on the last
    # pre-separation iteration even while newer exact controls are committed.
    research_result, research_source = latest_committed_research_control_result(
        run_dir
    )
    if research_result and research_source is not None:
        iteration = research_result.get("iteration")
        digest = str(research_result.get("checkpoint_digest") or "")
        rows = research_result.get("matchups")
        audit = research_result.get("audit")
        expected_ids = set(OFFICIAL_BASELINE_IDS)
        if (
            isinstance(iteration, int)
            and _is_sha256_digest(digest)
            and isinstance(rows, list)
            and {str(row.get("opponent_id") or "") for row in rows} == expected_ids
            and isinstance(audit, dict)
            and audit.get("passed") is True
        ):
            matchups = [
                {
                    "opponent_id": str(row.get("opponent_id") or ""),
                    "games": int(row.get("games") or 0),
                    "wr": as_float(row.get("win_rate")),
                    "wins": as_float(row.get("wins")),
                    "draws": as_float(row.get("draws")),
                    "losses": as_float(row.get("losses")),
                    "seat0": int(row.get("seat0") or 0),
                    "seat1": int(row.get("seat1") or 0),
                }
                for row in rows
            ]
            history_row = next(
                (
                    row
                    for row in reversed(loop.get("history") or [])
                    if isinstance(row, dict) and row.get("iteration") == iteration
                ),
                {},
            )
            learner_after = (
                history_row.get("learner_after")
                if isinstance(history_row.get("learner_after"), dict)
                else {}
            )
            return {
                "available": True,
                "kind": "latest_committed_research_control_result",
                "valid": True,
                "passed": None,
                "reason": "research_only_non_gate",
                "games": int(research_result.get("games") or 0),
                "wr": as_float(research_result.get("win_rate")),
                "lower": None,
                "upper": None,
                "iteration": iteration + int(global_iteration_offset),
                "lineage_iteration": iteration,
                "checkpoint": research_result.get("checkpoint"),
                "checkpoint_digest": digest,
                "matchups": matchups,
                "opponent_count": len(matchups),
                "audit_passed": True,
                "exact_distribution": audit.get("exact_distribution") is True,
                "exact_weights": audit.get("exact_weights") is True,
                "greedy_required": research_result.get("action_selection")
                == "greedy",
                "matchup_runtime": dict(audit.get("matchup_runtime") or {}),
                "protected_champion": digest == heldout_digest,
                "heldout_champion_updated": False,
                "learner_retained": str(learner_after.get("digest") or "")
                == digest,
                "training_eligible": False,
                "replay_eligible": False,
                "source": str(research_source),
                "updated_at": research_source.stat().st_mtime,
            }
    for history_row in reversed(loop.get("history") or []):
        if not isinstance(history_row, dict) or history_row.get("completed") is not True:
            continue
        iteration = history_row.get("iteration")
        candidate = history_row.get("candidate")
        audit = history_row.get("heldout_audit")
        gate = history_row.get("raw_heldout_gate")
        if (
            not isinstance(iteration, int)
            or not isinstance(candidate, dict)
            or not isinstance(audit, dict)
            or not isinstance(gate, dict)
        ):
            continue
        source = run_dir / "commits" / f"iter_{iteration:05d}.json"
        if not source.is_file():
            continue
        digest = str(candidate.get("digest") or "")
        audit_games = int(audit.get("valid_games") or 0)
        gate_games = int(gate.get("games") or 0)
        gate_wr = as_float(gate.get("win_rate"))
        audit_matchups = audit.get("per_opponent")
        gate_matchups = gate.get("per_opponent")
        if (
            not digest
            or audit.get("passed") is not True
            or audit.get("exact_distribution") is not True
            or audit.get("exact_weights") is not True
            or audit.get("greedy_required") is not True
            or str(audit.get("checkpoint_digest") or "") != digest
            or audit_games <= 0
            or audit_games != gate_games
            or gate_wr is None
            or not isinstance(audit_matchups, dict)
            or not isinstance(gate_matchups, dict)
            or set(audit_matchups) != set(OFFICIAL_BASELINE_IDS)
            or set(gate_matchups) != set(OFFICIAL_BASELINE_IDS)
        ):
            continue
        matchups: list[dict[str, Any]] = []
        valid = True
        for opponent_id in OFFICIAL_BASELINE_IDS:
            audit_row = audit_matchups.get(opponent_id)
            gate_row = gate_matchups.get(opponent_id)
            if not isinstance(audit_row, dict) or not isinstance(gate_row, dict):
                valid = False
                break
            games = int(gate_row.get("games") or 0)
            audit_row_games = int(audit_row.get("games") or 0)
            seat0 = int(audit_row.get("seat0") or gate_row.get("seat0_games") or 0)
            seat1 = int(audit_row.get("seat1") or gate_row.get("seat1_games") or 0)
            wr = as_float(gate_row.get("win_rate"))
            if (
                games <= 0
                or games != audit_row_games
                or seat0 + seat1 != games
                or wr is None
            ):
                valid = False
                break
            matchups.append(
                {
                    "opponent_id": opponent_id,
                    "games": games,
                    "wr": wr,
                    "wins": as_float(gate_row.get("wins")),
                    "draws": as_float(gate_row.get("draws")),
                    "losses": as_float(gate_row.get("losses")),
                    "seat0": seat0,
                    "seat1": seat1,
                }
            )
        if not valid or sum(int(row["games"]) for row in matchups) != gate_games:
            continue
        learner_after = (
            history_row.get("learner_after")
            if isinstance(history_row.get("learner_after"), dict)
            else {}
        )
        return {
            "available": True,
            "kind": "latest_committed_official_heldout_attempt",
            "valid": True,
            "passed": gate.get("passed") is True,
            "reason": gate.get("reason"),
            "games": gate_games,
            "wr": gate_wr,
            "lower": as_float(gate.get("confidence_lower")),
            "upper": as_float(gate.get("confidence_upper")),
            "iteration": iteration + int(global_iteration_offset),
            "lineage_iteration": iteration,
            "checkpoint": candidate.get("path"),
            "checkpoint_digest": digest,
            "matchups": matchups,
            "opponent_count": len(matchups),
            "audit_passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "matchup_runtime": dict(audit.get("matchup_runtime") or {}),
            "protected_champion": digest == heldout_digest,
            "heldout_champion_updated": history_row.get("heldout_champion_updated")
            is True,
            "learner_retained": str(learner_after.get("digest") or "") == digest,
            "source": str(source),
            "updated_at": source.stat().st_mtime,
        }
    return {"available": False, "reason": "no committed exact holdout attempt"}


def latest_committed_formal_holdout_state(
    loop: dict[str, Any],
    run_dir: Path | None,
    *,
    global_iteration_offset: int = 0,
) -> dict[str, Any]:
    """Return the newest immutable, fully audited active-gate holdout.

    Formal premium holdouts and the four-agent research controls are separate
    measurement programs.  A formal result remains displayable when it fails
    performance thresholds; only its distribution/audit must pass.
    """

    if run_dir is None:
        return {"available": False, "reason": "run directory is unavailable"}
    heldout_identity = (
        loop.get("heldout_champion")
        if isinstance(loop.get("heldout_champion"), dict)
        else {}
    )
    heldout_digest = str(heldout_identity.get("digest") or "")
    for history_row in reversed(loop.get("history") or []):
        if not isinstance(history_row, dict) or history_row.get("completed") is not True:
            continue
        iteration = history_row.get("iteration")
        candidate = history_row.get("candidate")
        gate = history_row.get("active_gate_result")
        if not isinstance(gate, dict):
            gate = history_row.get("raw_heldout_gate")
        audit = history_row.get("heldout_audit")
        if not isinstance(audit, dict) and isinstance(gate, dict):
            audit = gate.get("audit")
        if (
            not isinstance(iteration, int)
            or not isinstance(candidate, dict)
            or not isinstance(gate, dict)
            or not isinstance(audit, dict)
        ):
            continue
        source = run_dir / "commits" / f"iter_{iteration:05d}.json"
        if not source.is_file():
            continue
        digest = str(candidate.get("digest") or "")
        rows = gate.get("matchups")
        audit_rows = audit.get("per_opponent")
        gate_games = int(gate.get("games") or 0)
        if (
            not _is_sha256_digest(digest)
            or audit.get("passed") is not True
            or audit.get("exact_distribution") is not True
            or audit.get("exact_weights") is not True
            or audit.get("greedy_required") is not True
            or str(audit.get("checkpoint_digest") or "") != digest
            or int(audit.get("valid_games") or 0) != gate_games
            or gate_games <= 0
            or not isinstance(rows, list)
            or not rows
            or not isinstance(audit_rows, dict)
        ):
            continue
        row_ids = [str(row.get("opponent_id") or "") for row in rows]
        if not all(row_ids) or len(row_ids) != len(set(row_ids)):
            continue
        if set(row_ids) != set(audit_rows):
            continue
        matchups: list[dict[str, Any]] = []
        valid = True
        for row in rows:
            opponent_id = str(row.get("opponent_id") or "")
            audit_row = audit_rows.get(opponent_id)
            games = int(row.get("games") or 0)
            seat0 = int(row.get("seat0") or 0)
            seat1 = int(row.get("seat1") or 0)
            wr = as_float(row.get("wr"))
            if wr is None:
                wr = as_float(row.get("win_rate"))
            if (
                not isinstance(audit_row, dict)
                or games <= 0
                or int(audit_row.get("games") or 0) != games
                or int(audit_row.get("seat0") or 0) != seat0
                or int(audit_row.get("seat1") or 0) != seat1
                or seat0 + seat1 != games
                or wr is None
            ):
                valid = False
                break
            matchups.append(
                {
                    "opponent_id": opponent_id,
                    "games": games,
                    "wr": wr,
                    "wins": as_float(row.get("wins")),
                    "draws": as_float(row.get("draws")),
                    "losses": as_float(row.get("losses")),
                    "seat0": seat0,
                    "seat1": seat1,
                }
            )
        if not valid or sum(int(row["games"]) for row in matchups) != gate_games:
            continue
        learner_after = (
            history_row.get("learner_after")
            if isinstance(history_row.get("learner_after"), dict)
            else {}
        )
        return {
            "available": True,
            "kind": "latest_committed_formal_holdout",
            "valid": True,
            "passed": gate.get("passed") is True,
            "reason": gate.get("pipeline_gate_reason") or gate.get("reason"),
            "games": gate_games,
            "wr": as_float(gate.get("skill_weighted_wr"))
            if as_float(gate.get("skill_weighted_wr")) is not None
            else as_float(gate.get("win_rate")),
            "lower": as_float(gate.get("confidence_lower")),
            "upper": as_float(gate.get("confidence_upper")),
            "iteration": iteration + int(global_iteration_offset),
            "lineage_iteration": iteration,
            "checkpoint": candidate.get("path"),
            "checkpoint_digest": digest,
            "matchups": matchups,
            "opponent_count": len(matchups),
            "audit_passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "matchup_runtime": dict(audit.get("matchup_runtime") or {}),
            "protected_champion": digest == heldout_digest,
            "heldout_champion_updated": history_row.get("heldout_champion_updated")
            is True,
            "learner_retained": str(learner_after.get("digest") or "") == digest,
            "source": str(source),
            "updated_at": source.stat().st_mtime,
        }
    return {"available": False, "reason": "no committed exact formal holdout attempt"}


def matchup_runtime_collection_state(
    run_dir: Path | None,
) -> dict[str, Any]:
    """Expose the newest immutable collection's causal-router receipt.

    Mutable logs are intentionally ignored.  The dashboard may call routing
    active only after the collection receipt has committed aggregate game
    audits written by the simulator result path.
    """
    if run_dir is None:
        return {"available": False, "reason": "run directory unavailable"}
    candidates = sorted(
        (run_dir / "collection_receipts").glob("iter_*.json"),
        reverse=True,
    )
    for source in candidates:
        payload = read_json(source)
        stats = payload.get("stats")
        if (
            payload.get("schema") != "poke_bot.completed_collection/v1"
            or not isinstance(payload.get("iteration"), int)
            or not isinstance(stats, dict)
        ):
            continue
        combined = stats.get("matchup_runtime")
        self_play = stats.get("matchup_runtime_self_play")
        public_mix = stats.get("matchup_runtime_public_mix")
        enforcement = stats.get("matchup_runtime_enforcement")
        if not all(
            isinstance(row, dict)
            and row.get("schema")
            == "poke_bot.matchup_runtime_collection_audit/v1"
            for row in (combined, self_play, public_mix)
        ) or not (
            isinstance(enforcement, dict)
            and enforcement.get("schema")
            == "poke_bot.matchup_runtime_collection_enforcement/v1"
            and enforcement.get("required") is True
            and enforcement.get("passed") is True
        ):
            continue
        return {
            "available": True,
            "iteration": int(payload["iteration"]),
            "checkpoint_digest": payload.get("checkpoint_digest"),
            "combined": combined,
            "self_play": self_play,
            "public_mix": public_mix,
            "enforcement": enforcement,
            "source": str(source),
            "updated_at": source.stat().st_mtime,
        }
    return {
        "available": False,
        "reason": "no committed causal-router collection receipt",
    }


def gpu_state() -> list[dict[str, Any]]:
    raw = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        gpus.append(
            {
                "index": as_number(parts[0]),
                "name": parts[1],
                "utilization": number(parts[2]),
                "memory_used_mib": number(parts[3]),
                "memory_total_mib": number(parts[4]),
                "power_w": number(parts[5]),
                "power_limit_w": number(parts[6]),
                "temperature_c": number(parts[7]),
            }
        )
    return gpus


def system_state() -> dict[str, Any]:
    mem: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if value:
                mem[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    loads = os.getloadavg()
    cpu_percent = None
    try:
        def cpu_ticks() -> tuple[int, int]:
            fields = [
                int(value)
                for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            ]
            return sum(fields), fields[3] + fields[4]

        before = cpu_ticks()
        time.sleep(0.15)
        after = cpu_ticks()
        total_delta = after[0] - before[0]
        if total_delta > 0:
            cpu_percent = 100.0 * (
                total_delta - (after[1] - before[1])
            ) / total_delta
    except (OSError, ValueError, IndexError):
        pass
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "cpu_utilization_percent": cpu_percent,
        "load_1m": loads[0],
        "load_5m": loads[1],
        "load_15m": loads[2],
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
    }


def recent_events(run_name: str | None = None) -> list[str]:
    # Keep the active run last: callers retain the final lines, so placing a
    # completed bootstrap after it silently hid live scheduler GPS/SPS and
    # result-buffer telemetry.
    active_log = (
        ROOT / "outputs/logs" / f"{run_name}.log" if run_name else None
    )
    raw = (
        read_tail(EXACT_LOG, 40_000)
        + "\n"
        + read_tail(TRAINING_LOG, 40_000)
        + "\n"
        + read_tail(ALAKAZAM_TRANSITION_LOG, 40_000)
        + "\n"
        + read_tail(ALAKAZAM_BOOTSTRAP_LOG, 80_000)
        + "\n"
        + (read_tail(active_log, 120_000) if active_log is not None else "")
    )
    lines = [ANSI_RE.sub("", line).strip() for line in raw.replace("\r", "\n").splitlines()]
    return [line for line in lines if line][-12:]


def scheduler_queue_state(
    run_name: str | None,
    *,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Read the active dispatch's endpoint-owned queue contract.

    The trainer emits the protected controller depths before request threads
    start.  Remote worker snapshots provide the changing server-side queue;
    keeping these two grains separate prevents a full protected reserve from
    being mislabeled as an idle worker.
    """
    if not run_name:
        return {"available": False, "mode": "waiting"}
    log_path = log_path or (ROOT / "outputs/logs" / f"{run_name}.log")
    # Result-drain telemetry can continue for hours after the phase's queue
    # contract line. Keep enough of the active log to retain that identity;
    # otherwise the dashboard falls back to a fake live-generation ETA while
    # only completed results are being compacted.
    raw = ANSI_RE.sub("", read_tail(log_path, 2_000_000)).replace("\r", "\n")
    matches = list(
        re.finditer(
            r"\[remote\] endpoint_owned_queues depths=(\{[^\n]+?\}) "
            r"(?:caps|high_water)=(\{[^\n]+?\})(?: safety_ceiling=\{[^\n]+?\})? "
            r"shared_endpoint_race=disabled",
            raw,
        )
    )
    if not matches:
        return {"available": False, "mode": "legacy_or_starting"}
    latest = matches[-1]
    try:
        depths_raw = ast.literal_eval(latest.group(1))
        caps_raw = ast.literal_eval(latest.group(2))
    except (SyntaxError, ValueError):
        return {"available": False, "mode": "invalid_telemetry"}
    if not isinstance(depths_raw, dict) or not isinstance(caps_raw, dict):
        return {"available": False, "mode": "invalid_telemetry"}

    # Scope socket telemetry to this dispatch generation. A prefetch factor of
    # one intentionally emits no ``socket_prefetch=`` line, so searching an
    # undelimited prefix would reuse the prior generation's larger factor.
    phase_start = max(
        raw.rfind("[pure_rl] mid_iter_rebalance=start", 0, latest.start()),
        raw.rfind("[pure_rl] self_play remote_weights", 0, latest.start()),
        raw.rfind("[pure_rl] play remote_weights", 0, latest.start()),
        0,
    )
    dispatch_prefix = raw[phase_start : latest.start()]
    dispatch_tail = raw[phase_start:]
    # A bounded prefix is useful for the socket-prefetch lines emitted just
    # before this queue generation. Remaining-job counters are phase-local,
    # however: carrying the preceding self-play ``remaining=0`` into a newly
    # started public-mix wave falsely reports that all public work is assigned.
    phase_tail = raw[latest.end() :]
    socket_prefetch: dict[str, int] = {}
    for endpoint, count in re.findall(
        r"\[remote\]\s+(\S+)\s+socket_prefetch=(\d+)", dispatch_prefix
    ):
        socket_prefetch[str(endpoint)] = int(count)
    # Factor-one dispatch logs the execution-sized slot count instead of a
    # redundant prefetch row. Use that current-generation count only when no
    # explicit prefetch expansion was emitted.
    for endpoint, slots in re.findall(
        r"\[remote\]\s+(\S+)\s+demand=\d+\s+slots=(\d+)/\d+",
        dispatch_prefix,
    ):
        socket_prefetch.setdefault(str(endpoint), int(slots))
    rebalance = list(re.finditer(r"\bremaining=(\d+)\b", phase_tail))
    unassigned = int(rebalance[-1].group(1)) if rebalance else None
    drain_events = list(
        re.finditer(
            r"remote_slots_live=(\d+)\s+remaining=(\d+)\s+"
            r"result_buffer=(\{[^\n]+\})",
            dispatch_tail,
        )
    )
    result_drain: dict[str, Any] = {"active": False}
    if drain_events:
        event = drain_events[-1]
        try:
            buffer = ast.literal_eval(event.group(3))
        except (SyntaxError, ValueError):
            buffer = {}
        if not isinstance(buffer, dict):
            buffer = {}
        remote_slots_live = int(event.group(1))
        remaining_jobs = int(event.group(2))
        spool_files = max(0, int(buffer.get("spool_files") or 0))
        memory_items = max(0, int(buffer.get("memory_items") or 0))
        # The last non-empty buffer sample remains in the bounded log after
        # collection has sealed. Do not keep presenting that historical tail
        # once the trainer has explicitly crossed into collection completion
        # or learner work.
        drain_phase_closed = bool(
            re.search(
                r"\[pure_rl\]\s+(?:collect done|completed collection committed|"
                r"train begin)\b",
                dispatch_tail[event.end() :],
            )
        )
        if (
            remote_slots_live == 0
            and remaining_jobs == 0
            and memory_items + spool_files > 0
            and not drain_phase_closed
        ):
            result_drain = {
                "active": True,
                "all_jobs_claimed": True,
                "producers_complete": True,
                "remote_slots_live": 0,
                "remaining_unassigned": 0,
                "memory_items": memory_items,
                "spool_files": spool_files,
                "spool_bytes": max(0, int(buffer.get("spool_bytes") or 0)),
                "buffered_results": memory_items + spool_files,
            }
    controller_contract = list(
        re.finditer(
            r"\[remote\] queue_refill_controller interval=([\d.]+)s "
            r"low_water=(\d+)% action=(\S+) endpoints=(\S+) "
            r"ingest_coupled=(\S+)",
            phase_tail,
        )
    )
    refill_events = list(
        re.finditer(
            r"\[remote\]\s+(\S+)\s+LOW_WATER_REFILL "
            r"active=(\d+) queued=(\d+)<(\d+) added=(\d+) "
            r"fill=high_water target_active=(\d+) high_water=(\d+)",
            phase_tail,
        )
    )

    def host_key(endpoint: str) -> str:
        lowered = endpoint.lower()
        if "bert" in lowered or "192.168.1.158" in lowered:
            return "bert"
        if "elmo" in lowered or "192.168.1.143" in lowered:
            return "elmo"
        return endpoint

    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint, cap_value in caps_raw.items():
        endpoint = str(endpoint)
        cap = max(0, int(cap_value))
        sockets = max(0, int(socket_prefetch.get(endpoint, 0)))
        endpoints[host_key(endpoint)] = {
            "endpoint": endpoint,
            "dispatch_reserved": max(0, int(depths_raw.get(endpoint, 0))),
            "protected_high_water": cap,
            "socket_capacity": sockets or None,
            "controller_reserve_target": max(0, cap - sockets) if sockets else None,
        }
    for event in refill_events:
        row = endpoints.get(host_key(event.group(1)))
        if row is None:
            continue
        row["last_refill"] = {
            "sampled_active": int(event.group(2)),
            "sampled_queued": int(event.group(3)),
            "low_water": int(event.group(4)),
            "added": int(event.group(5)),
            "target_active": int(event.group(6)),
            "high_water": int(event.group(7)),
            "action": "fill_to_high_water",
        }
    contract: dict[str, Any] = {
        "probe_interval_s": 0.2,
        "low_water_fraction": 0.5,
        "action": "fill_to_high_water",
        "endpoints_parallel": True,
        "ingest_coupled": False,
    }
    if controller_contract:
        event = controller_contract[-1]
        contract.update(
            probe_interval_s=float(event.group(1)),
            low_water_fraction=float(event.group(2)) / 100.0,
            action=str(event.group(3)),
            endpoints_parallel=str(event.group(4)).lower() == "parallel",
            ingest_coupled=str(event.group(5)).lower() == "true",
        )
    return {
        "available": True,
        "mode": "endpoint_owned",
        "shared_endpoint_race_disabled": True,
        "unassigned": unassigned,
        "result_drain": result_drain,
        "refill_contract": contract,
        "endpoints": endpoints,
        "source": str(log_path),
    }


def scope_scheduler_queues_to_progress(
    progress: dict[str, Any],
    scheduler_queues: dict[str, Any],
) -> dict[str, Any]:
    """Remove phase-local queue state after game generation has ended.

    Queue contracts are intentionally retained in long log tails for drain
    observability. Optimizer and replay-prep progress is newer runtime truth,
    however, so a preceding collection's non-empty buffer cannot remain an
    active scheduler condition after the progress stream crosses that phase.
    """

    stage = str(progress.get("stage") or "")
    generation_active = bool(
        stage.startswith("collect:")
        or stage.startswith("heldout")
        or stage.startswith("promotion")
        or stage.startswith("drain:")
    )
    if generation_active or not stage:
        return scheduler_queues
    scoped = dict(scheduler_queues)
    scoped["result_drain"] = {"active": False}
    scoped["unassigned"] = 0
    scoped["phase_active"] = False
    scoped["phase"] = stage
    return scoped


def result_drain_projection(
    progress: dict[str, Any],
    scheduler_queues: dict[str, Any],
) -> dict[str, Any]:
    """Project producer-complete spool compaction without fake live remotes."""

    drain = dict(scheduler_queues.get("result_drain") or {})
    if drain.get("active") is not True:
        return {}
    stage = str(progress.get("stage") or "collect")
    if not stage.startswith("collect:"):
        return {}
    phase_name = stage.split(":", 1)[1]
    current = progress.get("current")
    total = progress.get("total")
    buffered = int(drain.get("buffered_results") or 0)
    if isinstance(current, (int, float)) and isinstance(total, (int, float)):
        buffered = min(buffered, max(0, int(total) - int(current)))
    spool_files = int(drain.get("spool_files") or 0)
    memory_items = int(drain.get("memory_items") or 0)
    iteration = progress.get("iteration")
    progress_metrics = dict(progress.get("metrics") or {})
    configured_remote_demand = int(
        progress_metrics.get("remote_request_sockets")
        or progress.get("remotes")
        or 0
    )
    return {
        "phase": f"drain:{phase_name}_results",
        "remote_workers": 0,
        "games_per_second": None,
        "latest_line": (
            f"pure_rl drain:{phase_name}_results iter={iteration}: "
            f"{current}/{total} · {buffered} buffered results "
            f"({spool_files} spool / {memory_items} memory) · "
            "all simulations claimed; producers complete"
        ),
        "metrics": {
            "result_spool_drain": True,
            "all_jobs_claimed": True,
            "remote_slots_live": 0,
            "remote_request_sockets": 0,
            "remote_queue_capacity": 0,
            "configured_remote_demand": configured_remote_demand,
            "buffered_results": buffered,
            "result_spool_files": spool_files,
            "result_spool_bytes": int(drain.get("spool_bytes") or 0),
        },
    }


def learner_model_state(
    manifest: dict[str, Any],
    loop: dict[str, Any] | None = None,
    *,
    iteration: int | None = None,
    runtime_optimizer: dict[str, Any] | None = None,
    runtime_parameter_contract: dict[str, Any] | None = None,
    runtime_collection: dict[str, Any] | None = None,
    checkpoint_structure: dict[str, Any] | None = None,
    dormant_modules_path: Path = DORMANT_MODEL_MODULES,
    staged_adapter_roster_path: Path = STAGED_MATCHUP_ADAPTER_ROSTER,
    matchup_runtime_ready_path: Path = MATCHUP_RUNTIME_PRODUCTION_READY,
    matchup_runtime_boundary_path: Path = MATCHUP_RUNTIME_BOUNDARY,
    specialist_runtime_registry_path: Path = SPECIALIST_RUNTIME_REGISTRY,
    specialist_activation_state_root: Path = ROOT / "outputs/state",
) -> dict[str, Any]:
    """Describe the exact live model plus explicitly non-live staged profiles.

    The old dashboard hard-coded one parameter count. That looked current even
    after a model-profile change. Prefer immutable manifest metadata, then an
    independently deployed profile registry whose full config must match. If
    neither source matches, report an unknown count instead of a plausible lie.
    """
    design_contract = manifest.get("design_contract") or {}
    learner = design_contract.get("learner") or {}
    expert = design_contract.get("expert_rehearsal") or {}
    selected_environment = (
        manifest.get("selected_environment")
        if isinstance(manifest.get("selected_environment"), dict)
        else {}
    )
    active_specialist = str(
        manifest.get("specialist_archetype")
        or selected_environment.get("POKEBOT_ACTIVE_SPECIALIST")
        or selected_environment.get("POKEBOT_PRIMARY_ARCHETYPE")
        or ""
    ).strip().casefold()
    generic_guide_enabled = bool(
        learner.get("current_deck_guide_targets_enabled")
    )
    legacy_alakazam_guide_enabled = bool(
        learner.get("alakazam_guide_targets_enabled")
    )
    guide_archetype = str(
        learner.get("current_deck_guide_archetype")
        or ("alakazam" if legacy_alakazam_guide_enabled else "")
    ).strip().casefold()
    guide_loss_weight = (
        as_float(learner.get("current_deck_guide_loss_weight"))
        if generic_guide_enabled
        else as_float(learner.get("alakazam_guide_loss_weight"))
    ) or 0.0
    current_deck_guide_enabled = bool(
        (generic_guide_enabled or legacy_alakazam_guide_enabled)
        and guide_loss_weight > 0.0
        and guide_archetype
        and guide_archetype == active_specialist
    )
    profile = learner.get("profile") if isinstance(learner.get("profile"), dict) else {}
    loop = loop if isinstance(loop, dict) else {}
    runtime_optimizer = (
        runtime_optimizer if isinstance(runtime_optimizer, dict) else {}
    )
    runtime_parameter_contract = (
        runtime_parameter_contract
        if isinstance(runtime_parameter_contract, dict)
        else {}
    )
    runtime_collection = (
        runtime_collection if isinstance(runtime_collection, dict) else {}
    )

    registry = read_json(MODEL_PROFILE_REGISTRY)
    registry_profiles = registry.get("profiles") or []
    matched_profile: dict[str, Any] = {}
    planned_profile: dict[str, Any] = {}
    for candidate in registry_profiles:
        if not isinstance(candidate, dict):
            continue
        candidate_profile = candidate.get("profile")
        if isinstance(candidate_profile, dict) and candidate_profile == profile:
            matched_profile = candidate
        status = str(candidate.get("status") or "")
        # A registry entry can retain its historical ``staged_*`` label after
        # that exact profile has become the immutable live manifest profile.
        # Never surface the active profile as a future plan: the manifest is
        # authoritative for what the trainer is actually running.
        if (
            not planned_profile
            and status.startswith("staged")
            and candidate_profile != profile
        ):
            planned_profile = candidate

    runtime_parameter_count = as_number(
        str(runtime_parameter_contract.get("trainable_parameters") or "")
    )
    parameter_count = (
        runtime_parameter_count
        if runtime_parameter_count is not None and runtime_parameter_count > 0
        else as_number(str(learner.get("trainable_parameters") or ""))
    )
    parameter_source = None
    if runtime_parameter_count is not None and runtime_parameter_count > 0:
        parameter_source = "runtime checkpoint load"
    elif parameter_count is not None:
        parameter_source = "manifest.design_contract.learner"
    if parameter_count is None:
        base_contract = manifest.get("base_checkpoint_contract") or {}
        parameter_count = as_number(
            str(base_contract.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = "manifest.base_checkpoint_contract"
    if parameter_count is None:
        parameter_count = as_number(
            str(manifest.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = "manifest"
    if parameter_count is None and matched_profile:
        parameter_count = as_number(
            str(matched_profile.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = f"profile_registry:{matched_profile.get('id') or 'matched'}"

    steady_cap = as_number(str(learner.get("max_decisions_per_batch") or ""))
    warmup_cap = as_number(
        str(learner.get("warmup_max_decisions_per_batch") or "")
    )
    warmup_iterations = as_number(str(learner.get("warmup_iterations") or "")) or 0
    active_cap = steady_cap
    schedule_phase = "steady"
    if (
        warmup_cap is not None
        and iteration is not None
        and int(iteration) < int(warmup_iterations)
    ):
        active_cap = warmup_cap
        schedule_phase = "head_focus"

    active_checkpoint = loop.get("learner")
    if not isinstance(active_checkpoint, dict):
        active_checkpoint = {}
    checkpoint_structure = (
        checkpoint_structure
        if isinstance(checkpoint_structure, dict)
        else checkpoint_structure_telemetry(
            str(active_checkpoint.get("path") or ""),
            str(active_checkpoint.get("digest") or ""),
        )
    )
    checkpoint_structure_identity_current = bool(
        checkpoint_structure.get("checkpoint")
        == str(active_checkpoint.get("path") or "")
        and checkpoint_structure.get("checkpoint_digest")
        == str(active_checkpoint.get("digest") or "")
    )
    checkpoint_structure_valid = bool(
        checkpoint_structure.get("verified") is True
        and checkpoint_structure_identity_current
    )
    if checkpoint_structure_valid:
        checkpoint_parameter_count = int(
            checkpoint_structure.get("model_parameters") or 0
        )
        if checkpoint_parameter_count > 0:
            parameter_count = checkpoint_parameter_count
            parameter_source = "active committed checkpoint structure"
            runtime_parameter_contract = {
                **runtime_parameter_contract,
                "trainable_parameters": checkpoint_parameter_count,
                "checkpoint": str(active_checkpoint.get("path") or ""),
                "source": str(checkpoint_structure.get("source") or ""),
            }

    def weighted_head(weight_key: str, *, outputs: int | None = None) -> dict[str, Any]:
        weight = as_float(learner.get(weight_key)) or 0.0
        row: dict[str, Any] = {"enabled": weight > 0.0, "loss_weight": weight}
        if outputs is not None:
            row["outputs"] = outputs
        return row

    initial = learner.get("initial_checkpoint")
    if not isinstance(initial, dict):
        initial = {}
    temporal_layers = as_number(str(profile.get("temporal_layers") or 0)) or 0
    architecture = (
        "full-game temporal state evaluator"
        if temporal_layers > 0 or profile.get("decision_context") == "history"
        else "stateless state evaluator"
    )
    dormant_contract = read_json(dormant_modules_path)
    staged_adapter_roster = read_json(staged_adapter_roster_path)
    staged_adapter_valid = bool(
        staged_adapter_roster.get("schema")
        == "poke_bot.matchup_adapter_roster_stage/v1"
        and staged_adapter_roster.get("status") == "tested_staged_not_active"
        and staged_adapter_roster.get("runtime_enabled") is False
        and staged_adapter_roster.get("mutually_exclusive_route_per_decision") is True
        and staged_adapter_roster.get("unknown_route_exact_bypass") is True
        and int(staged_adapter_roster.get("expert_count") or 0)
        == len(staged_adapter_roster.get("expert_ids") or [])
        and int(staged_adapter_roster.get("parameter_count") or 0) > 0
        and int((staged_adapter_roster.get("validation") or {}).get("tests_passed") or 0) > 0
    )
    dormant_modules: list[dict[str, Any]] = []
    if dormant_contract.get("schema") == "poke_bot.dormant_model_modules/v1":
        for candidate in dormant_contract.get("modules") or []:
            if not isinstance(candidate, dict):
                continue
            expert_count = int(candidate.get("expert_count") or 0)
            hidden_dim = int(candidate.get("hidden_dim") or 0)
            bottleneck_dim = int(candidate.get("bottleneck_dim") or 0)
            expected_parameters = expert_count * (
                hidden_dim * bottleneck_dim
                + bottleneck_dim
                + bottleneck_dim * hidden_dim
                + hidden_dim
            )
            candidate_parameters = int(candidate.get("parameter_count") or 0)
            status = str(candidate.get("status") or "")
            staged = (
                status == "staged_non_active"
                and candidate.get("present_in_active_checkpoint") is False
            )
            deployed = (
                status == "deployed_dormant"
                and candidate.get("present_in_active_checkpoint") is True
                and (
                    candidate.get("zero_output") is True
                    or (
                        candidate.get("trained_shadow") is True
                        and candidate.get("fit_receipt_valid") is True
                    )
                )
            )
            if (
                not (staged or deployed)
                or candidate.get("runtime_enabled") is not False
                or candidate.get("optimizer_active") is not False
                or candidate_parameters <= 0
                or candidate_parameters != expected_parameters
            ):
                continue
            row = dict(candidate)
            fit = loop.get("dormant_matchup_adapter_fit")
            if not isinstance(fit, dict):
                fit = {}
            fit_matches_learner = bool(
                fit.get("schema") == "poke_bot.dormant_matchup_adapter_fit/v1"
                and fit.get("runtime_enabled") is False
                and fit.get("base_frozen") is True
                and fit.get("optimizer_scope") == "matchup_adapter_bank_only"
                and str(fit.get("checkpoint_digest") or "")
                == str(active_checkpoint.get("digest") or "")
                and int(fit.get("steps") or 0) > 0
                and int(fit.get("rows") or 0) > 0
            )
            if deployed and fit_matches_learner:
                row.update(
                    {
                        "zero_output": False,
                        "trained_shadow": True,
                        "fit_receipt_valid": True,
                        "fit_epochs": int(fit.get("epochs") or 0),
                        "fit_steps": int(fit.get("steps") or 0),
                        "fit_rows": int(fit.get("rows") or 0),
                        "route_sequences": dict(
                            fit.get("route_sequences") or {}
                        ),
                        "route_decisions": dict(
                            fit.get("route_decisions") or {}
                        ),
                    }
                )
            dormant_modules.append(row)

    # The historical dormant-module file deliberately remains immutable, so
    # it still describes the ten-route shadow checkpoint.  Once the clean
    # iteration boundary has activated v31, that file is no longer evidence
    # for the live model.  Promote only a fully cross-linked production-ready
    # receipt, boundary receipt, active learner identity, and validated
    # 22-position roster.  Any mismatch leaves the older/staged view intact.
    runtime_ready = read_json(matchup_runtime_ready_path)
    runtime_boundary = read_json(matchup_runtime_boundary_path)
    runtime_artifacts = runtime_ready.get("artifacts") or {}
    runtime_checkpoint = runtime_artifacts.get("merged_checkpoint") or {}
    activated_learner = runtime_boundary.get("activated_learner") or {}
    parent_learner = runtime_boundary.get("parent_learner") or {}
    adapter_fit = runtime_boundary.get("adapter_fit") or {}
    runtime_tree = runtime_boundary.get("runtime_tree") or {}
    active_path = str(active_checkpoint.get("path") or "")
    active_digest = str(active_checkpoint.get("digest") or "")
    specialist_registry = read_json(specialist_runtime_registry_path)
    specialist_rows = (
        specialist_registry.get("specialists")
        if isinstance(specialist_registry.get("specialists"), dict)
        else {}
    )
    run_name = str(manifest.get("run_name") or "")
    specialist_id = ""
    specialist_row: dict[str, Any] = {}
    for candidate_id, candidate in specialist_rows.items():
        if (
            isinstance(candidate, dict)
            and str(candidate.get("run_name") or "") == run_name
        ):
            specialist_id = str(candidate_id)
            specialist_row = dict(candidate)
            break
    specialist_tree_path = Path(
        str(specialist_row.get("matchup_runtime_tree") or "/nonexistent")
    )
    specialist_authorization_path = Path(
        str(
            specialist_row.get("matchup_adapter_authorization")
            or "/nonexistent"
        )
    )
    specialist_tree = read_json(specialist_tree_path)
    specialist_authorization = read_json(specialist_authorization_path)
    specialist_runtime_contract = dict(
        specialist_tree.get("runtime_contract") or {}
    )
    specialist_targets = [
        str(value) for value in (specialist_tree.get("targets") or [])
    ]
    specialist_accepted = [
        str(value)
        for value in (
            specialist_runtime_contract.get("accepted_archetype_ids") or []
        )
    ]
    specialist_origin_valid = bool(
        specialist_registry.get("schema")
        == "poke_bot.specialist_runtime_registry/v1"
        and specialist_row.get("status") == "ready"
        and len(specialist_targets) == 22
        and len(set(specialist_targets)) == 22
        and specialist_id in specialist_targets
        and specialist_id in specialist_accepted
        and specialist_tree.get("runtime_enabled") is True
        and specialist_runtime_contract.get("one_route_per_decision") is True
        and specialist_runtime_contract.get("unknown_route_exact_bypass") is True
        and set(specialist_accepted).issubset(set(specialist_targets))
        and specialist_runtime_contract.get("checkpoint") == active_path
        and specialist_runtime_contract.get("checkpoint_digest")
        == active_digest
        and _file_sha256_matches(
            specialist_tree_path,
            specialist_row.get("matchup_runtime_tree_sha256"),
        )
        and specialist_authorization.get("schema")
        in {
            "poke_bot.matchup_adapter_rehearsal_authorization/v1",
            "poke_bot.matchup_adapter_specialist_bootstrap_authorization/v1",
        }
        and specialist_authorization.get("optimizer_scope")
        == "matchup_adapter_bank_only"
        and specialist_authorization.get("runtime_enabled") is False
        and specialist_authorization.get("parent_checkpoint") == active_path
        and specialist_authorization.get("parent_checkpoint_digest")
        == active_digest
        and _file_sha256_matches(
            specialist_authorization_path,
            specialist_row.get("matchup_adapter_authorization_sha256"),
        )
        and int(
            specialist_row.get("matchup_adapter_epochs_per_rl_iteration") or 0
        )
        >= 1
    )
    if specialist_origin_valid:
        runtime_ready = {
            "schema": "poke_bot.matchup_runtime_production_ready/v1",
            "runtime_enabled": True,
            "iteration": specialist_authorization.get(
                "first_eligible_iteration"
            ),
            "artifacts": {
                "merged_checkpoint": {
                    "path": active_path,
                    "digest": active_digest,
                }
            },
        }
        runtime_boundary = {
            "schema": "poke_bot.matchup_runtime_boundary_activation/v1",
            "activated_learner": {
                "path": active_path,
                "digest": active_digest,
            },
            "parent_learner": {"path": active_path},
            "boundary": {
                "next_iteration": specialist_authorization.get(
                    "first_eligible_iteration"
                )
            },
            "adapter_fit": {
                "trained_archetype_ids": specialist_accepted,
                "route_decisions": {
                    route: 0 for route in specialist_targets
                },
            },
            "runtime_tree": {
                "accepted_archetype_ids": specialist_accepted,
                "continuous_re_evaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "digest": specialist_row.get("matchup_runtime_tree_sha256"),
            },
        }
        runtime_artifacts = runtime_ready["artifacts"]
        runtime_checkpoint = runtime_artifacts["merged_checkpoint"]
        activated_learner = runtime_boundary["activated_learner"]
        parent_learner = runtime_boundary["parent_learner"]
        adapter_fit = runtime_boundary["adapter_fit"]
        runtime_tree = runtime_boundary["runtime_tree"]
    trained_ids = [
        str(value)
        for value in adapter_fit.get("trained_archetype_ids") or []
    ]
    accepted_ids = [
        str(value)
        for value in runtime_tree.get("accepted_archetype_ids") or []
    ]
    staged_ids = [
        str(value) for value in staged_adapter_roster.get("expert_ids") or []
    ]
    runtime_origin_valid = bool(
        (staged_adapter_valid or specialist_origin_valid)
        and runtime_ready.get("schema")
        == "poke_bot.matchup_runtime_production_ready/v1"
        and runtime_ready.get("runtime_enabled") is True
        and runtime_boundary.get("schema")
        == "poke_bot.matchup_runtime_boundary_activation/v1"
        and str(runtime_checkpoint.get("path") or "")
        == str(activated_learner.get("path") or "")
        and str(runtime_checkpoint.get("digest") or "")
        == str(activated_learner.get("digest") or "")
        and _is_sha256_digest(str(runtime_checkpoint.get("digest") or ""))
        and int(
            runtime_ready["iteration"]
            if runtime_ready.get("iteration") is not None
            else -1
        )
        == int(
            (runtime_boundary.get("boundary") or {})["next_iteration"]
            if (runtime_boundary.get("boundary") or {}).get("next_iteration")
            is not None
            else -2
        )
        and runtime_tree.get("continuous_re_evaluation") is True
        and runtime_tree.get("one_route_per_decision") is True
        and runtime_tree.get("unknown_route_exact_bypass") is True
        and set(trained_ids).issubset(staged_ids)
        and set(accepted_ids).issubset(trained_ids)
        and bool(accepted_ids)
    )
    active_fit = loop.get("dormant_matchup_adapter_fit")
    if not isinstance(active_fit, dict):
        active_fit = {}
    active_fit_trained_ids = [
        str(value)
        for value in active_fit.get("trained_archetype_ids") or []
    ]
    active_is_origin = bool(
        active_path
        and _is_sha256_digest(active_digest)
        and str(runtime_checkpoint.get("path") or "") == active_path
        and str(runtime_checkpoint.get("digest") or "") == active_digest
    )
    # The immutable activation receipt names the first v31 checkpoint. Every
    # later pure-RL candidate is a descendant with a new digest, so requiring
    # permanent equality to the activation checkpoint made the dashboard fall
    # back to the stale ten-route marker after the very next iteration. The
    # loop's adapter-only fit receipt is the per-checkpoint continuation proof:
    # it is checksum-pinned to the active learner and records the isolated
    # optimizer scope plus full trained-route coverage.
    active_is_receipted_descendant = bool(
        active_path
        and _is_sha256_digest(active_digest)
        and active_fit.get("schema")
        == "poke_bot.dormant_matchup_adapter_fit/v1"
        and active_fit.get("base_frozen") is True
        and active_fit.get("optimizer_scope") == "matchup_adapter_bank_only"
        and str(active_fit.get("checkpoint_path") or "") == active_path
        and str(active_fit.get("checkpoint_digest") or "") == active_digest
        and int(active_fit.get("steps") or 0) > 0
        and int(active_fit.get("rows") or 0) > 0
        and set(active_fit_trained_ids).issubset(staged_ids)
        and set(accepted_ids).issubset(active_fit_trained_ids)
    )
    # A safety rollback can select an older post-activation learner while
    # deliberately clearing the adapter-only fit receipt that belonged to the
    # rejected candidate.  That does not remove the adapter bank from the
    # selected checkpoint.  A committed collection receipt is stronger direct
    # evidence: it checksum-pins the exact selected checkpoint, audits every
    # game, and records the one-route runtime roster.  Accept that evidence
    # only when it reconciles exactly with the immutable activation receipts.
    collection_combined = runtime_collection.get("combined")
    if not isinstance(collection_combined, dict):
        collection_combined = {}
    collection_enforcement = runtime_collection.get("enforcement")
    if not isinstance(collection_enforcement, dict):
        collection_enforcement = {}
    collection_rosters = collection_combined.get("accepted_roster_counts")
    if not isinstance(collection_rosters, dict):
        collection_rosters = {}
    collection_roster_ids: set[str] = set()
    if len(collection_rosters) == 1:
        collection_roster_ids = {
            value
            for value in str(next(iter(collection_rosters))).split("|")
            if value
        }
    collection_proves_runtime = bool(
        runtime_collection.get("available") is True
        and str(runtime_collection.get("checkpoint_digest") or "") == active_digest
        and collection_combined.get("all_games_audited") is True
        and collection_combined.get("all_runtime_enabled") is True
        and collection_combined.get("contract_clean") is True
        and int(collection_combined.get("audited_games") or 0)
        == int(collection_combined.get("games") or -1)
        and int(collection_combined.get("games") or 0) > 0
        and collection_enforcement.get("required") is True
        and collection_enforcement.get("passed") is True
        and collection_roster_ids == set(accepted_ids)
    )
    checkpoint_expert_ids = [
        str(value)
        for value in (
            checkpoint_structure.get("adapter_expert_ids") or []
        )
    ]
    checkpoint_proves_runtime = bool(
        checkpoint_structure_valid
        and checkpoint_expert_ids
        and len(checkpoint_expert_ids)
        == int(checkpoint_structure.get("adapter_expert_count") or 0)
        and collection_roster_ids == set(checkpoint_expert_ids)
        and runtime_collection.get("available") is True
        and str(runtime_collection.get("checkpoint_digest") or "")
        == active_digest
        and collection_combined.get("all_games_audited") is True
        and collection_combined.get("all_runtime_enabled") is True
        and collection_combined.get("contract_clean") is True
        and int(collection_combined.get("audited_games") or 0)
        == int(collection_combined.get("games") or -1)
        and int(collection_combined.get("games") or 0) > 0
        and collection_enforcement.get("required") is True
        and collection_enforcement.get("passed") is True
    )
    checkpoint_descendant_proves_runtime = bool(
        checkpoint_structure_valid
        and checkpoint_expert_ids
        and len(checkpoint_expert_ids)
        == int(checkpoint_structure.get("adapter_expert_count") or 0)
        and collection_roster_ids == set(checkpoint_expert_ids)
        and runtime_collection.get("available") is True
        and collection_combined.get("all_games_audited") is True
        and collection_combined.get("all_runtime_enabled") is True
        and collection_combined.get("contract_clean") is True
        and int(collection_combined.get("audited_games") or 0)
        == int(collection_combined.get("games") or -1)
        and int(collection_combined.get("games") or 0) > 0
        and collection_enforcement.get("required") is True
        and collection_enforcement.get("passed") is True
        and active_fit.get("schema")
        == "poke_bot.dormant_matchup_adapter_fit/v1"
        and active_fit.get("base_frozen") is True
        and active_fit.get("optimizer_scope") == "matchup_adapter_bank_only"
        and str(active_fit.get("checkpoint_path") or "") == active_path
        and str(active_fit.get("checkpoint_digest") or "") == active_digest
        and int(active_fit.get("steps") or 0) > 0
        and int(active_fit.get("rows") or 0) > 0
        and set(active_fit_trained_ids).issubset(set(checkpoint_expert_ids))
    )
    if checkpoint_proves_runtime or checkpoint_descendant_proves_runtime:
        # The current immutable checkpoint plus its committed collection audit
        # (or its checksum-pinned descendant fit plus the immediately prior
        # clean collection audit) is the live source of truth. Historical
        # activation receipts remain useful lineage evidence, but they must
        # not make a later specialist fall back to a stale ten-route marker.
        staged_ids = checkpoint_expert_ids
        accepted_ids = checkpoint_expert_ids
    runtime_adapter_valid = bool(
        checkpoint_proves_runtime
        or checkpoint_descendant_proves_runtime
        or (
            runtime_origin_valid
            and (
                active_is_origin
                or active_is_receipted_descendant
                or collection_proves_runtime
            )
        )
    )
    matchup_adapter_runtime: dict[str, Any] = {}
    if runtime_adapter_valid:
        runtime_adapter_parameters = int(
            (
                checkpoint_structure.get("adapter_parameters")
                if checkpoint_structure_valid
                else staged_adapter_roster.get("parameter_count")
            )
            or 0
        )
        evidence_checkpoint = str(
            runtime_parameter_contract.get("checkpoint") or ""
        )
        parent_path = str(parent_learner.get("path") or "")
        if parameter_count is not None and evidence_checkpoint == parent_path:
            # Runtime checkpoint loading materializes the canonical current
            # adapter architecture even for the boundary parent. Therefore
            # its parameter count already includes all 22 slots; replacing the
            # stale marker's ten-slot count here would double-count 12 routes.
            parameter_source = "runtime checkpoint load + activation receipt"
            runtime_parameter_contract = {
                **runtime_parameter_contract,
                "checkpoint": active_path,
                "source": str(matchup_runtime_ready_path),
            }
        effective_fit = (
            active_fit
            if (
                active_is_receipted_descendant
                or checkpoint_proves_runtime
                or checkpoint_descendant_proves_runtime
            )
            else adapter_fit
        )
        trained_ids = [
            str(value)
            for value in effective_fit.get("trained_archetype_ids") or []
        ]
        route_decisions = {
            str(key): int(value)
            for key, value in dict(
                effective_fit.get("route_decisions") or {}
            ).items()
        }
        zero_example_ids = [
            value for value in staged_ids if int(route_decisions.get(value, 0)) <= 0
        ]
        adapter_epochs = int(
            runtime_optimizer.get("dormant_matchup_adapter_epochs")
            or ((learner.get("dormant_matchup_adapter") or {}).get("epochs") or 0)
            or specialist_row.get("matchup_adapter_epochs_per_rl_iteration")
            or 0
        )
        dormant_modules = [
            {
                "id": "matchup_adapter_bank_v4",
                "label": "22-position causal matchup residual adapter bank",
                "status": "deployed_runtime",
                "present_in_active_checkpoint": True,
                "runtime_enabled": True,
                "optimizer_active": False,
                "ordinary_optimizer_included": False,
                "isolated_adapter_updates_enabled": adapter_epochs > 0,
                "isolated_adapter_epochs_per_rl_iteration": adapter_epochs,
                "isolated_adapter_learning_rate": as_float(
                    runtime_optimizer.get("dormant_matchup_adapter_lr")
                ),
                "zero_output": False,
                "partially_trained": bool(trained_ids),
                "parameter_count": runtime_adapter_parameters,
                "expert_count": len(staged_ids),
                "hidden_dim": 96,
                "bottleneck_dim": 8,
                "residual_scale": 0.25,
                "architecture": f"{len(staged_ids)} × 96→8→96 residual MLP",
                "expert_ids": staged_ids,
                "trained_archetype_ids": trained_ids,
                "accepted_runtime_archetype_ids": accepted_ids,
                "zero_example_archetype_ids": zero_example_ids,
                "route_decisions": route_decisions,
                "router_shadow_active": False,
                "router_model_application_enabled": True,
                "router_mode": "causal_public_prefix_runtime",
                "continuous_reevaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "checkpoint_digest": active_digest,
                "runtime_tree_digest": runtime_tree.get("digest"),
            }
        ]
        matchup_adapter_runtime = {
            "enabled": True,
            "iteration": int(
                runtime_collection.get("iteration")
                if (
                    collection_proves_runtime
                    or checkpoint_proves_runtime
                    or checkpoint_descendant_proves_runtime
                )
                else runtime_ready.get("iteration")
                or 0
            ),
            "checkpoint": active_path,
            "checkpoint_digest": active_digest,
            "expert_count": len(staged_ids),
            "expert_ids": staged_ids,
            "trained_count": len(trained_ids),
            "accepted_runtime_count": len(accepted_ids),
            "accepted_archetype_ids": accepted_ids,
            "continuous_reevaluation": True,
            "one_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "live_collection_verified": collection_proves_runtime,
            "checkpoint_descendant_chain_verified": (
                checkpoint_descendant_proves_runtime
            ),
            "production_ready_receipt": str(matchup_runtime_ready_path),
            "boundary_receipt": str(matchup_runtime_boundary_path),
            "checkpoint_structure_verified": checkpoint_structure_valid,
            "checkpoint_structure_source": checkpoint_structure.get("source"),
        }
    # The immutable checkpoint structure and stable roster supersede historical
    # ten-head/v4 dashboard marker files.  Derive the displayed bank from those
    # two sources so roster changes never require another hard-coded dashboard
    # parameter total or version-specific status record.
    canonical_adapter_roster = read_json(CANONICAL_MATCHUP_ADAPTER_ROSTER)
    canonical_adapter_ids = [
        str(value)
        for value in (canonical_adapter_roster.get("expert_ids") or [])
        if str(value)
    ]
    checkpoint_is_canonical_roster = bool(
        checkpoint_structure_valid
        and canonical_adapter_roster.get("schema")
        == "poke_bot.matchup_adapter_roster/v1"
        and canonical_adapter_ids
        and canonical_adapter_ids == checkpoint_expert_ids
        and len(canonical_adapter_ids)
        == int(checkpoint_structure.get("adapter_expert_count") or 0)
        and int(checkpoint_structure.get("adapter_parameters") or 0) > 0
    )
    if checkpoint_is_canonical_roster:
        effective_fit = (
            active_fit
            if isinstance(active_fit, dict)
            and str(active_fit.get("checkpoint_digest") or "") == active_digest
            else {}
        )
        route_sequences = {
            specialist_id: int(
                dict(effective_fit.get("route_sequences") or {}).get(
                    specialist_id, 0
                )
            )
            for specialist_id in canonical_adapter_ids
        }
        route_decisions = {
            specialist_id: int(
                dict(effective_fit.get("route_decisions") or {}).get(
                    specialist_id, 0
                )
            )
            for specialist_id in canonical_adapter_ids
        }
        trained_ids = [
            specialist_id
            for specialist_id in canonical_adapter_ids
            if route_sequences[specialist_id] > 0
            or route_decisions[specialist_id] > 0
        ]
        adapter_parameters = int(
            checkpoint_structure.get("adapter_parameters") or 0
        )
        adapter_runtime_enabled = matchup_adapter_runtime.get("enabled") is True
        adapter_epochs = int(
            runtime_optimizer.get("dormant_matchup_adapter_epochs")
            or ((learner.get("dormant_matchup_adapter") or {}).get("epochs") or 0)
            or specialist_row.get("matchup_adapter_epochs_per_rl_iteration")
            or 0
        )
        dormant_modules = [
            {
                "id": "matchup_adapter_bank_v5_roster18",
                "format": "poke-bot-matchup-adapter-bank-v5-roster18",
                "label": (
                    f"{len(canonical_adapter_ids)}-position causal matchup "
                    "residual adapter bank"
                ),
                "status": (
                    "deployed_runtime"
                    if adapter_runtime_enabled
                    else "deployed_dormant"
                ),
                "present_in_active_checkpoint": True,
                "runtime_enabled": adapter_runtime_enabled,
                "optimizer_active": False,
                "ordinary_optimizer_included": False,
                "isolated_adapter_updates_enabled": adapter_epochs > 0,
                "isolated_adapter_epochs_per_rl_iteration": adapter_epochs,
                "isolated_adapter_learning_rate": as_float(
                    runtime_optimizer.get("dormant_matchup_adapter_lr")
                ),
                "zero_output": not bool(trained_ids),
                "partially_trained": bool(trained_ids),
                "trained_shadow": bool(trained_ids),
                "fit_receipt_valid": bool(effective_fit),
                "parameter_count": adapter_parameters,
                "expert_count": len(canonical_adapter_ids),
                "hidden_dim": 96,
                "bottleneck_dim": 8,
                "residual_scale": 0.25,
                "architecture": (
                    f"{len(canonical_adapter_ids)} × 96→8→96 residual MLP"
                ),
                "expert_ids": canonical_adapter_ids,
                "trained_archetype_ids": trained_ids,
                "accepted_runtime_archetype_ids": list(
                    matchup_adapter_runtime.get("accepted_archetype_ids") or []
                ),
                "zero_example_archetype_ids": [
                    specialist_id
                    for specialist_id in canonical_adapter_ids
                    if specialist_id not in trained_ids
                ],
                "route_sequences": route_sequences,
                "route_decisions": route_decisions,
                "router_shadow_active": not adapter_runtime_enabled,
                "router_model_application_enabled": adapter_runtime_enabled,
                "router_mode": (
                    "causal_public_prefix_runtime"
                    if adapter_runtime_enabled
                    else "causal_public_prefix_shadow"
                ),
                "continuous_reevaluation": True,
                "one_route_per_decision": True,
                "unknown_route_exact_bypass": True,
                "checkpoint_digest": active_digest,
                "roster_source": str(CANONICAL_MATCHUP_ADAPTER_ROSTER),
            }
        ]
        staged_adapter_roster = {
            **canonical_adapter_roster,
            "status": "deployed_current",
            "checkpoint_format": "poke-bot-matchup-adapter-bank-v5-roster18",
            "runtime_enabled": adapter_runtime_enabled,
            "expert_count": len(canonical_adapter_ids),
            "parameter_count": adapter_parameters,
            "mutually_exclusive_route_per_decision": True,
            "unknown_route_exact_bypass": True,
        }
    staged_non_active_parameters = sum(
        int(row.get("parameter_count") or 0) for row in dormant_modules
        if row.get("present_in_active_checkpoint") is False
    )
    deployed_non_active_parameters = sum(
        int(row.get("parameter_count") or 0) for row in dormant_modules
        if row.get("present_in_active_checkpoint") is True
    )
    # Runtime checkpoint telemetry counts every tensor in the model, including
    # a frozen adapter bank.  Subtract dormant parameters for the optimizer
    # figure; adding them here used to double-count the bank in the total.
    current_checkpoint_parameters = (
        int(parameter_count) if parameter_count is not None else None
    )
    optimizer_active_parameters = (
        max(0, current_checkpoint_parameters - deployed_non_active_parameters)
        if current_checkpoint_parameters is not None
        else None
    )
    parameter_breakdown = {
        "current_checkpoint_total": current_checkpoint_parameters,
        "optimizer_active_current": optimizer_active_parameters,
        "current_non_active": (
            deployed_non_active_parameters
            if optimizer_active_parameters is not None
            else None
        ),
        "staged_non_active": staged_non_active_parameters,
        "staged_architecture_total": (
            current_checkpoint_parameters + staged_non_active_parameters
            if current_checkpoint_parameters is not None
            else None
        ),
        "staged_modules": sum(
            row.get("present_in_active_checkpoint") is False
            for row in dormant_modules
        ),
        "deployed_dormant_modules": sum(
            row.get("present_in_active_checkpoint") is True
            for row in dormant_modules
        ),
        "source": str(dormant_modules_path),
    }
    # Keep the deployed V5 checkpoint contract and the staged V6 registry
    # separate.  CANONICAL_MATCHUP_ADAPTER_ROSTER belongs to the immutable
    # runtime deployment; the state-core copy beside specialists.yaml is the
    # authoritative planning registry for a pending safe-boundary migration.
    # Reading V6 from the runtime file made the staged panel disappear whenever
    # the healthy trainer correctly remained on V5.
    v6_roster_path = SPECIALIST_PROTOCOL_STATE.parent / (
        "matchup_adapter_roster.json"
    )
    v6_roster = read_json(v6_roster_path)
    v6_slots = list(v6_roster.get("slots") or [])
    v6_stage_valid = bool(
        v6_roster.get("schema")
        == "poke_bot.matchup_adapter_roster/v1"
        and v6_roster.get("slot_schema")
        == "poke_bot.matchup_adapter_slot_registry/v1"
        and v6_roster.get("checkpoint_format")
        == "poke-bot-matchup-adapter-bank-v6"
        and int(v6_roster.get("slot_capacity") or 0) == 64
        and len(v6_slots) == 64
        and [int(row.get("slot", -1)) for row in v6_slots] == list(range(64))
    )
    matchup_adapter_v6 = (
        {
            "status": v6_roster.get("status"),
            "checkpoint_format": v6_roster.get(
                "checkpoint_format"
            ),
            "registry_revision": v6_roster.get("revision"),
            "registry_digest": _canonical_json_digest(v6_roster),
            "physical_slot_capacity": 64,
            "logical_active_count": sum(
                row.get("status") in {"active", "dormant"}
                for row in v6_slots
            ),
            "unused_count": sum(
                row.get("status") == "unused" for row in v6_slots
            ),
            "retired_count": sum(
                row.get("status") == "retired" for row in v6_slots
            ),
            "live_selector_changed": False,
            "activation_boundary": v6_roster.get(
                "activation_policy"
            ),
            "source": str(v6_roster_path),
        }
        if v6_stage_valid
        else {}
    )
    expanded_head_training = (
        dict(checkpoint_structure.get("expanded_head_training") or {})
        if checkpoint_structure_identity_current
        else {
            "schema": EXPANDED_HEAD_CONTRACT_SCHEMA,
            "available": False,
            "verified": False,
            "legacy_v5": False,
            "reason": "expanded-head telemetry is not bound to the active checkpoint",
            "heads": [],
        }
    )
    checkpoint_fusion = (
        dict(checkpoint_structure.get("decision_fusion") or {})
        if checkpoint_structure_identity_current
        else {
            "schema": DECISION_FUSION_SCHEMA,
            "available": False,
            "verified": False,
            "phase": "telemetry_unbound",
            "runtime_enabled": False,
            "training_enabled": False,
            "reason": "decision-fusion telemetry is not bound to the active checkpoint",
        }
    )
    loop_fusion = dict(loop.get("decision_fusion_activation") or {})
    active_digest = str(active_checkpoint.get("digest") or "")
    loop_fusion_bound = bool(
        loop_fusion.get("learner_digest") == active_digest
        and str(loop_fusion.get("schema") or "").startswith(
            "poke_bot.causal_decision_fusion_"
        )
    )
    successor_fusion = _successor_decision_fusion_activation(
        state_root=specialist_activation_state_root,
        specialist_id=specialist_id,
        checkpoint_digest=active_digest,
        run_dir=(
            ROOT / "outputs/pure_rl" / run_name
            if run_name
            else None
        ),
        design_fingerprint=_active_run_design_fingerprint(loop, manifest),
        initial_checkpoint_digest=_initial_learner_checkpoint_digest(
            learner,
            manifest,
        ),
    )
    successor_fusion_bound = bool(
        successor_fusion.get("runtime_enabled") is True
        and successor_fusion.get("training_action_eligible") is True
    )
    final_refresh_fusion = _final_refresh_decision_fusion_continuity(
        state_root=specialist_activation_state_root,
        specialist_id=specialist_id,
        checkpoint_digest=active_digest,
    )
    final_refresh_fusion_bound = bool(
        final_refresh_fusion.get("runtime_enabled") is True
        and final_refresh_fusion.get("training_action_eligible") is True
    )
    activation_bound = bool(
        loop_fusion_bound
        or successor_fusion_bound
        or final_refresh_fusion_bound
    )
    receipt_fusion = (
        successor_fusion
        if successor_fusion_bound
        else final_refresh_fusion
    )
    receipt_fusion_bound = bool(
        successor_fusion_bound or final_refresh_fusion_bound
    )
    decision_fusion = {
        **checkpoint_fusion,
        "checkpoint_digest": active_digest or None,
        "loop_activation_bound": loop_fusion_bound,
        "successor_activation_bound": successor_fusion_bound,
        "final_refresh_activation_bound": final_refresh_fusion_bound,
        "activation_bound": activation_bound,
        "activation_scope": (
            "loop_boundary"
            if loop_fusion_bound
            else receipt_fusion.get("activation_scope")
            if receipt_fusion_bound
            else None
        ),
        "activation_receipt": (
            loop_fusion.get("receipt")
            if loop_fusion_bound
            else receipt_fusion.get("receipt")
            if receipt_fusion_bound
            else None
        ),
        "boundary_next_iteration": (
            loop_fusion.get("boundary_next_iteration")
            if loop_fusion_bound
            else None
        ),
        "terminal_serving_eligible": (
            loop_fusion.get("serving_eligible")
            if loop_fusion_bound
            else successor_fusion.get("terminal_serving_eligible")
            if successor_fusion_bound
            else False
        ),
    }
    if checkpoint_fusion.get("runtime_enabled") is True and not (
        (
            loop_fusion_bound
            and loop_fusion.get("runtime_enabled") is True
        )
        or receipt_fusion_bound
    ):
        decision_fusion.update(
            verified=False,
            serving_eligible=False,
            phase="activation_receipt_mismatch",
            reason=(
                "runtime-enabled checkpoint lacks its checksum-bound loop "
                "activation receipt"
            ),
        )
    elif (
        checkpoint_fusion.get("runtime_enabled") is True
        and receipt_fusion_bound
    ):
        decision_fusion.update(
            verified=True,
            serving_eligible=bool(
                receipt_fusion.get("terminal_serving_eligible")
            ),
            phase=(
                "runtime_active_successor"
                if successor_fusion_bound
                else "runtime_active_final_refresh_training"
            ),
            reason=(
                "checksum-bound fused-policy activation verified; terminal "
                "freeze and serving still require this child's exact gates"
            ),
        )
    base_heads: dict[str, dict[str, Any]] = {
        "policy": {"enabled": True},
        "value": {"enabled": True},
        "archetype": weighted_head("archetype_aux_loss_weight", outputs=20),
        "opponent_hand": weighted_head("opp_hand_loss_weight", outputs=1268),
        "opponent_remainder": weighted_head(
            "opp_remainder_loss_weight", outputs=1268
        ),
        "lethal_threat": weighted_head("lethal_threat_loss_weight", outputs=1),
        "prize_race": weighted_head("prize_race_loss_weight", outputs=2),
    }
    latent_lookahead = dict(
        checkpoint_structure.get("latent_lookahead") or {}
    )
    if latent_lookahead.get("enabled") is True:
        base_heads["latent_policy_aid"] = {
            "enabled": latent_lookahead.get("verified") is True,
            "used_in_decisions": bool(
                latent_lookahead.get("action_authority_enabled") is True
            ),
            "outputs": 4,
            "parameters": int(latent_lookahead.get("parameters") or 0),
            "policy_aid_cap": float(
                latent_lookahead.get("policy_aid_cap") or 0.0
            ),
            "scope": "active_committed_checkpoint",
        }
    expanded_heads: dict[str, dict[str, Any]] = {}
    fusion_required_heads = {
        str(value)
        for value in (decision_fusion.get("required_heads") or ())
        if str(value)
    }
    fusion_decision_path_active = bool(
        decision_fusion.get("verified") is True
        and decision_fusion.get("runtime_enabled") is True
        and decision_fusion.get("serving_eligible") is True
        and decision_fusion.get("activation_bound") is True
    )
    for raw_row in expanded_head_training.get("heads") or []:
        if not isinstance(raw_row, dict):
            continue
        head_id = _expanded_head_id(raw_row.get("id"))
        if head_id is None:
            continue
        expanded_heads[head_id] = {
            **dict(raw_row),
            "expanded": True,
            "enabled": bool(
                raw_row.get("present") is True
                and raw_row.get("contract_valid") is True
            ),
            "used_in_decisions": bool(
                fusion_decision_path_active
                and (
                    head_id in fusion_required_heads
                    or (
                        head_id == "tactical_outcome"
                        and "tactical_outcomes" in fusion_required_heads
                    )
                )
            ),
            "scope": "active_committed_checkpoint",
        }
    for head_id, row in base_heads.items():
        row["used_in_decisions"] = bool(
            head_id == "policy"
            or (
                fusion_decision_path_active
                and head_id in fusion_required_heads
            )
        )
    return {
        "implementation": "TemporalCabtTransformer",
        "architecture": architecture,
        "run": manifest.get("run_name"),
        "profile": profile,
        "profile_id": matched_profile.get("id"),
        "trainable_parameters": parameter_count,
        "parameter_source": parameter_source,
        "parameter_evidence_checkpoint": runtime_parameter_contract.get("checkpoint"),
        "parameter_evidence_source": runtime_parameter_contract.get("source"),
        "checkpoint_structure": checkpoint_structure,
        "latent_lookahead": latent_lookahead,
        "parameter_breakdown": parameter_breakdown,
        "dormant_modules": dormant_modules,
        "matchup_adapter_roster_stage": (
            {}
            if runtime_adapter_valid
            else staged_adapter_roster if staged_adapter_valid else {}
        ),
        "matchup_adapter_runtime": matchup_adapter_runtime,
        "matchup_adapter_v6": matchup_adapter_v6,
        "active_checkpoint": active_checkpoint.get("path"),
        "active_checkpoint_digest": active_checkpoint.get("digest"),
        "training_schedule": {
            "iteration": iteration,
            "phase": schedule_phase,
            "active_max_decisions_per_batch": active_cap,
            "warmup_max_decisions_per_batch": warmup_cap,
            "warmup_iterations": warmup_iterations,
            "steady_max_decisions_per_batch": steady_cap,
        },
        "optimizer": {
            "curriculum": {
                "name": "AdamW",
                "learning_rate": as_float(learner.get("learning_rate")) or 3e-4,
                "weight_decay": as_float(learner.get("weight_decay")) or 1e-4,
                "gradient_clip_norm": as_float(learner.get("gradient_clip_norm"))
                or 1.0,
                "scheduler": "constant_per_iteration",
                "precision": "bf16_autocast",
                "optimizer_state_restored": True,
                "epochs": as_number(str(learner.get("epochs") or "")),
                "games_per_batch": as_number(
                    str(learner.get("games_per_batch") or "")
                ),
                "max_decisions_per_batch": active_cap,
                "awr_frozen_baseline": True,
                "awr_beta": as_float(runtime_optimizer.get("awr_beta"))
                or as_float(learner.get("awr_beta"))
                or 0.5,
                "awr_weight_max": as_float(
                    runtime_optimizer.get("awr_weight_max")
                )
                or as_float(learner.get("awr_weight_max"))
                or 20.0,
                "entropy_bonus": as_float(learner.get("entropy_bonus")) or 0.01,
            },
            "expert_rehearsal": {
                "name": "AdamW",
                "learning_rate": as_float(expert.get("learning_rate")),
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "scheduler": "constant",
                "precision": "bf16_autocast",
                "epochs": as_number(str(expert.get("epochs") or "")),
                "requested_batch_size": as_number(
                    str(expert.get("requested_batch_size") or "")
                ),
            },
            "source": (
                "live systemd environment + immutable manifest contract"
                if runtime_optimizer
                else "live trainer implementation + immutable manifest contract"
            ),
        },
        "planned_profile": planned_profile,
        "heads": {**base_heads, **expanded_heads},
        "expanded_head_training": expanded_head_training,
        "decision_fusion": decision_fusion,
        "training_targets": {
            "current_deck_guide": {
                "enabled": current_deck_guide_enabled,
                "status": "active" if current_deck_guide_enabled else "absent",
                "active_specialist": active_specialist or None,
                "guide_archetype": guide_archetype or None,
                "loss_weight": guide_loss_weight if current_deck_guide_enabled else 0.0,
                "shared_head": "policy",
                "parameterized_head": False,
            },
            "alakazam_guide": {
                "enabled": current_deck_guide_enabled
                and guide_archetype == "alakazam",
                "loss_weight": as_float(learner.get("alakazam_guide_loss_weight")) or 0.0,
                "shared_head": "policy",
                "parameterized_head": False,
            }
        },
        "seed_checkpoint": initial.get("path"),
        "seed_checkpoint_digest": initial.get("digest"),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _canonical_design_digest(contract: dict[str, Any]) -> str:
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def effective_design_contract_for_run(
    run_dir: Path | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the last checksum-verified append-only design contract.

    The immutable manifest is the root of the lineage, not necessarily its
    current operational contract. Dashboard panels must follow the same
    migration chain as the trainer or newly enabled phases can be mislabeled
    as disabled. A malformed or broken receipt fails closed at the last
    verified contract.
    """
    contract = manifest.get("design_contract")
    if not isinstance(contract, dict):
        return {}
    effective = contract
    expected_digest = str(manifest.get("design_fingerprint") or "")
    if (
        not expected_digest
        or _canonical_design_digest(effective) != expected_digest
    ):
        return effective
    if run_dir is None:
        return effective
    for receipt_path in sorted(
        (run_dir / "design_migrations").glob("migration_*.json")
    ):
        receipt = read_json(receipt_path)
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        previous_digest = str(receipt.get("previous_fingerprint") or "")
        current_digest = str(receipt.get("current_fingerprint") or "")
        if (
            int(receipt.get("schema", -1)) != 1
            or not isinstance(previous, dict)
            or not isinstance(current, dict)
            or previous != effective
            or previous_digest != expected_digest
            or _canonical_design_digest(previous) != previous_digest
            or _canonical_design_digest(current) != current_digest
        ):
            break
        effective = current
        expected_digest = current_digest
    return effective


def active_gate_contract_for_run(run_dir: Path | None) -> Path:
    """Resolve the active gate from the run's append-only design chain.

    The dashboard deployment has a historical fallback contract in ``ops/``.
    Continuous training may migrate to a newer immutable contract, so using
    that fallback would render a fully committed newer gate as ``NOT RUN``.
    Only accept a migrated path when its recorded size and SHA-256 still match.
    """
    if run_dir is None:
        return COMPETITION_GATE_PROGRAM

    # The H10 refresh is governed by the live selector's registry, not by the
    # immutable iteration manifest that originally launched the run.  The
    # manifest intentionally retains r94 so its collection history remains
    # reproducible; r100 is the receipt-backed terminal contract now enforced
    # by the managed gate handler.  Resolve the registry's relative path under
    # its declared runtime root and require the expected final-format identity
    # before letting it supersede the historical manifest projection.
    try:
        is_h10_run = run_dir.resolve() == FINAL_FORMAT_ALAKAZAM_H10_RUN_DIR.resolve()
    except OSError:
        is_h10_run = False
    if is_h10_run:
        registry = read_json(FINAL_FORMAT_ALAKAZAM_H10_REGISTRY)
        raw_contract = str(registry.get("active_gate_contract") or "").strip()
        runtime_root = str(registry.get("runtime_root") or "").strip()
        if raw_contract and runtime_root:
            contract_path = Path(raw_contract).expanduser()
            if not contract_path.is_absolute():
                contract_path = Path(runtime_root).expanduser() / contract_path
            contract = read_json(contract_path)
            if (
                contract.get("active_gate_id")
                == registry.get("terminal_active_gate_id")
                or str(contract.get("active_gate_id") or "").startswith(
                    "final-format-alakazam-r100-strength75-rating1150-"
                )
            ):
                return contract_path.resolve()
    identities: list[dict[str, Any]] = []
    candidates = sorted((run_dir / "design_migrations").glob("migration_*.json"))
    for receipt_path in reversed(candidates):
        receipt = read_json(receipt_path)
        current = receipt.get("current_contract")
        gates = current.get("gates") if isinstance(current, dict) else None
        identity = gates.get("active_contract") if isinstance(gates, dict) else None
        if isinstance(identity, dict):
            identities.append(identity)
    # A new specialist lineage starts with its gate already embedded in the
    # immutable manifest and may have no migration receipt yet.  Falling
    # directly back to the dashboard deployment's historical contract makes
    # a valid 2,000-game commit appear as NOT RUN until the first migration.
    manifest = read_json(run_dir / "manifest.json")
    initial = manifest.get("design_contract")
    initial_gates = initial.get("gates") if isinstance(initial, dict) else None
    initial_identity = (
        initial_gates.get("active_contract")
        if isinstance(initial_gates, dict)
        else None
    )
    if isinstance(initial_identity, dict):
        identities.append(initial_identity)
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        raw_path = str(identity.get("path") or "").strip()
        expected_digest = str(identity.get("digest") or "")
        expected_size = int(identity.get("size") or -1)
        if not raw_path or not _is_sha256_digest(expected_digest):
            continue
        path = Path(raw_path).expanduser().resolve()
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if len(payload) == expected_size and actual_digest == expected_digest:
            return path
    return COMPETITION_GATE_PROGRAM


def latest_committed_active_gate_result(
    run_dir: Path | None,
    *,
    mutable_result_pointer: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Return the newest active-gate result bound to immutable commit history.

    Eval files are written before the immutable iteration commit and therefore
    are never evidence by themselves.  The history row inside the commit is
    authoritative.  A mutable compact pointer may supply the returned payload
    only when its core, commit path, and canonical commit digest exactly match
    that immutable row; a stale or conflicting pointer is ignored.
    """

    if run_dir is None:
        return {}, None
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "commits").glob("iter_*.json"):
        match = re.fullmatch(r"iter_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for iteration, commit_path in sorted(candidates, reverse=True):
        commit = read_json(commit_path)
        if (
            commit.get("last_completed_iteration") != iteration
            or commit.get("next_iteration") != iteration + 1
        ):
            continue
        history = commit.get("history")
        matching_rows = (
            [
                row
                for row in history
                if isinstance(row, dict) and row.get("iteration") == iteration
            ]
            if isinstance(history, list)
            else []
        )
        if len(matching_rows) != 1 or matching_rows[0].get("completed") is not True:
            continue
        result = matching_rows[0].get("active_gate_result")
        if not isinstance(result, dict) or result.get("iteration") != iteration:
            continue

        if mutable_result_pointer is not None:
            pointer_path = Path(mutable_result_pointer).expanduser().resolve()
            pointer = read_json(pointer_path)
            pointer_core = {
                key: value
                for key, value in pointer.items()
                if key
                not in {"committed", "commit", "commit_digest", "created_at_utc"}
            }
            raw_commit_path = str(pointer.get("commit") or "").strip()
            pointer_commit_path = (
                Path(raw_commit_path).expanduser().resolve()
                if raw_commit_path
                else None
            )
            if (
                pointer.get("committed") is True
                and pointer_core == result
                and pointer_commit_path == commit_path.resolve()
                and str(pointer.get("commit_digest") or "")
                == _canonical_json_digest(commit)
            ):
                return dict(pointer), pointer_path
        return dict(result), commit_path
    return {}, None


def latest_committed_research_control_result(
    run_dir: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    """Return only a dedicated control artifact bound to iteration commit history."""
    if run_dir is None:
        return {}, None
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "commits").glob("iter_*.json"):
        match = re.fullmatch(r"iter_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for iteration, commit_path in sorted(candidates, reverse=True):
        commit = read_json(commit_path)
        if (
            commit.get("last_completed_iteration") != iteration
            or commit.get("next_iteration") != iteration + 1
        ):
            continue
        history = commit.get("history")
        matches = (
            [
                row
                for row in history
                if isinstance(row, dict)
                and row.get("iteration") == iteration
                and row.get("completed") is True
            ]
            if isinstance(history, list)
            else []
        )
        if len(matches) != 1:
            continue
        result = matches[0].get("research_control_result")
        result_rows = (
            result.get("matchups") if isinstance(result, dict) else None
        )
        result_audit = (
            result.get("audit") if isinstance(result, dict) else None
        )
        if (
            not isinstance(result, dict)
            or result.get("schema")
            != "poke_bot.research_control_measurement_result/v1"
            or result.get("iteration") != iteration
            or result.get("training_eligible") is not False
            or result.get("replay_eligible") is not False
            or result.get("diagnostic_only") is not True
            or result.get("formal_eval") is not False
            or result.get("included_in_gate_pass") is not False
            or result.get("gate_weight") != 0.0
            or result.get("action_selection") != "greedy"
            or result.get("seed_namespace")
            != "eval/research-controls-fixed-manifest-v1"
            or not _is_sha256_digest(result.get("checkpoint_digest"))
            or not _is_sha256_digest(result.get("schedule_digest"))
            or not isinstance(result_rows, list)
            or not result_rows
            or result.get("games") != 250 * len(result_rows)
            or any(
                not isinstance(row, dict)
                or row.get("games") != 250
                or row.get("seat0") != 125
                or row.get("seat1") != 125
                or not _is_sha256_digest(row.get("content_digest"))
                for row in result_rows
            )
            or not isinstance(result_audit, dict)
            or result_audit.get("passed") is not True
            or result_audit.get("exact_distribution") is not True
            or result_audit.get("exact_weights") is not True
            or result_audit.get("seed_disjoint") is not True
            or result_audit.get("package_disjoint_from_active_gate") is not True
            or result_audit.get("replay_records_written") != 0
        ):
            continue
        expected_path = (
            run_dir / "research_controls" / f"iter_{iteration:05d}.json"
        ).resolve()
        raw_path = str(result.get("result_path") or "").strip()
        if not raw_path or Path(raw_path).expanduser().resolve() != expected_path:
            continue
        artifact = read_json(expected_path)
        if artifact != result:
            continue
        return dict(result), expected_path
    return {}, None


def research_control_registry_state(
    public_mix_live: dict[str, Any],
    *,
    registry_path: Path | None = None,
    measurement_result: dict[str, Any] | None = None,
    measurement_source: Path | None = None,
) -> dict[str, Any]:
    """Expose committed additive controls independently of training and gate state."""
    path = Path(
        registry_path
        or (
            RESEARCH_CONTROL_REGISTRY_LATEST
            if RESEARCH_CONTROL_REGISTRY_LATEST.is_file()
            else RESEARCH_CONTROL_REGISTRY
        )
    )
    registry = read_json(path)
    raw_controls = registry.get("controls")
    raw_retirements = registry.get("retirements")
    controls = (
        [dict(row) for row in raw_controls if isinstance(row, dict)]
        if isinstance(raw_controls, list)
        else []
    )
    ids = [str(row.get("opponent_id") or "") for row in controls]
    digests = [str(row.get("content_digest") or "") for row in controls]
    version = registry.get("version")

    def zero_gate_weight(row: dict[str, Any]) -> bool:
        value = row.get("gate_weight")
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) == 0.0
        )

    def nonnegative_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def finite_number(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        parsed = as_float(value)
        return parsed if parsed is not None and math.isfinite(parsed) else None

    valid = bool(
        registry.get("schema") == "poke_bot.research_control_registry/v1"
        and str(registry.get("registry_id") or "")
        and isinstance(version, int)
        and not isinstance(version, bool)
        and version >= 1
        and isinstance(raw_controls, list)
        and len(controls) == len(raw_controls)
        and controls
        and all(ids)
        and len(ids) == len(set(ids))
        and len(digests) == len(set(digests))
        and all(_is_sha256_digest(digest) for digest in digests)
        and all(
            bool(str(row.get("source_gate_id") or "").strip())
            and zero_gate_weight(row)
            and row.get("included_in_gate_pass") is False
            and row.get("formal_eval") is False
            and row.get("training_eligible") is False
            for row in controls
        )
        and isinstance(raw_retirements, list)
    )
    if not valid:
        return {
            "available": False,
            "reason": "research-control registry failed validation",
            "source": str(path),
        }

    controls_by_id = {
        str(row["opponent_id"]): row for row in controls
    }
    retirement_gate_ids: set[str] = set()
    retired_by_opponent: dict[str, dict[str, Any]] = {}
    retirement_valid = True
    for raw in raw_retirements:
        if not isinstance(raw, dict):
            retirement_valid = False
            break
        gate_id = str(raw.get("gate_id") or "")
        opponent_ids = raw.get("opponent_ids")
        if (
            not gate_id
            or gate_id in retirement_gate_ids
            or not isinstance(opponent_ids, list)
            or not opponent_ids
            or any(not isinstance(value, str) or not value for value in opponent_ids)
            or len(opponent_ids) != len(set(opponent_ids))
            or not set(opponent_ids).issubset(controls_by_id)
            or not _is_sha256_digest(raw.get("exact_result_digest"))
            or not _is_sha256_digest(raw.get("checkpoint_digest"))
            or not isinstance(raw.get("iteration"), int)
            or isinstance(raw.get("iteration"), bool)
            or int(raw["iteration"]) < 0
        ):
            retirement_valid = False
            break
        retirement_gate_ids.add(gate_id)
        for opponent_id in opponent_ids:
            if opponent_id in retired_by_opponent:
                retirement_valid = False
                break
            control = controls_by_id[opponent_id]
            if (
                str(control.get("source_gate_id") or "") != gate_id
                or str(control.get("retired_exact_result_digest") or "")
                != str(raw["exact_result_digest"])
                or str(control.get("retired_checkpoint_digest") or "")
                != str(raw["checkpoint_digest"])
                or not str(control.get("retired_at_utc") or "").strip()
            ):
                retirement_valid = False
                break
            retired_by_opponent[opponent_id] = raw
        if not retirement_valid:
            break
    if retirement_valid:
        legacy_controls = {
            str(control["opponent_id"]): str(control.get("content_digest") or "")
            for control in controls
            if str(control.get("source_gate_id") or "")
            == "legacy-original-four"
        }
        if legacy_controls != LEGACY_RESEARCH_CONTROL_DIGESTS:
            retirement_valid = False
    if retirement_valid:
        for control in controls:
            opponent_id = str(control["opponent_id"])
            source_gate_id = str(control.get("source_gate_id") or "")
            if source_gate_id == "legacy-original-four":
                continue
            elif opponent_id not in retired_by_opponent:
                retirement_valid = False
                break
    if not retirement_valid:
        return {
            "available": False,
            "reason": "research-control retirement history failed validation",
            "source": str(path),
        }

    exact_measurement = (
        dict(measurement_result)
        if isinstance(measurement_result, dict) and measurement_result
        else None
    )
    if exact_measurement is not None:
        native = {
            **exact_measurement,
            "available": int(exact_measurement.get("games") or 0) > 0,
            "active": False,
            "stage": "measure:research_controls:complete",
        }
        dedicated_telemetry = True
    else:
        native = public_mix_live.get("research_controls")
        dedicated_telemetry = isinstance(native, dict)
    if not dedicated_telemetry:
        # Migration compatibility for a sidecar written before schema v3.
        legacy_rows = [
            dict(row)
            for row in (public_mix_live.get("matchups") or [])
            if isinstance(row, dict)
            and str(row.get("opponent_id") or "") in set(ids)
        ]
        legacy_game_counts = [nonnegative_int(row.get("games")) for row in legacy_rows]
        if any(value is None for value in legacy_game_counts):
            return {
                "available": False,
                "reason": "research telemetry game totals are malformed",
                "source": str(path),
            }
        legacy_games = sum(value or 0 for value in legacy_game_counts)
        legacy_wins = sum(finite_number(row.get("wins")) or 0.0 for row in legacy_rows)
        legacy_draws = sum(nonnegative_int(row.get("draws")) or 0 for row in legacy_rows)
        legacy_losses = sum(
            nonnegative_int(row.get("losses")) or 0 for row in legacy_rows
        )
        native = {
            **public_mix_live,
            "available": legacy_games > 0,
            "games": legacy_games,
            "wins": legacy_wins,
            "draws": legacy_draws,
            "losses": legacy_losses,
            "win_rate": legacy_wins / legacy_games if legacy_games else None,
            "matchups": legacy_rows,
        }
    raw_measured = native.get("matchups")
    if raw_measured is None:
        raw_measured = []
    if not isinstance(raw_measured, list) or any(
        not isinstance(row, dict) for row in raw_measured
    ):
        return {
            "available": False,
            "reason": "research telemetry matchup rows are malformed",
            "source": str(path),
        }
    measured = [dict(row) for row in raw_measured]
    measured_ids = [str(row.get("opponent_id") or "") for row in measured]
    unexpected = sorted(
        {
            opponent_id
            for opponent_id in measured_ids
            if opponent_id not in set(ids)
        }
    )
    if dedicated_telemetry and unexpected:
        return {
            "available": False,
            "reason": "research telemetry contains an unregistered opponent",
            "unexpected_opponents": unexpected,
            "source": str(path),
        }
    if any(not opponent_id for opponent_id in measured_ids) or len(measured_ids) != len(
        set(measured_ids)
    ):
        return {
            "available": False,
            "reason": "research telemetry contains a missing or duplicate opponent",
            "source": str(path),
        }

    for row in measured:
        control = controls_by_id[str(row["opponent_id"])]
        reported_digests = {
            str(value)
            for value in (
                row.get("content_digest"),
                row.get("opponent_content_digest"),
            )
            if value is not None and str(value)
        }
        if reported_digests and reported_digests != {
            str(control["content_digest"])
        }:
            return {
                "available": False,
                "reason": "research telemetry package digest does not match registry",
                "opponent_id": str(row["opponent_id"]),
                "source": str(path),
            }

    measured_game_counts = [nonnegative_int(row.get("games")) for row in measured]
    if any(value is None for value in measured_game_counts):
        return {
            "available": False,
            "reason": "research telemetry game totals are malformed",
            "source": str(path),
        }
    measured_games = sum(value or 0 for value in measured_game_counts)
    measured_wins: list[float] = []
    measured_draws: list[int] = []
    measured_losses: list[int] = []
    measured_seat0: list[int] = []
    measured_seat1: list[int] = []
    for row, games in zip(measured, measured_game_counts):
        wins = finite_number(row.get("wins"))
        draws = nonnegative_int(row.get("draws"))
        losses = nonnegative_int(row.get("losses"))
        seat0 = nonnegative_int(row.get("seat0"))
        seat1 = nonnegative_int(row.get("seat1"))
        win_rate = finite_number(row.get("win_rate"))
        true_wins = None if wins is None or draws is None else wins - 0.5 * draws
        if (
            games is None
            or wins is None
            or draws is None
            or losses is None
            or seat0 is None
            or seat1 is None
            or win_rate is None
            or not 0.0 <= win_rate <= 1.0
            or true_wins is None
            or true_wins < 0.0
            or abs(true_wins - round(true_wins)) > 1e-9
            or abs((true_wins + draws + losses) - games) > 1e-9
            or seat0 + seat1 != games
            or abs(win_rate - (wins / games if games else 0.0)) > 1e-9
        ):
            return {
                "available": False,
                "reason": "research telemetry matchup aggregates do not reconcile",
                "source": str(path),
            }
        measured_wins.append(wins)
        measured_draws.append(draws)
        measured_losses.append(losses)
        measured_seat0.append(seat0)
        measured_seat1.append(seat1)
    native_games = native.get("games")
    native_wins = finite_number(native.get("wins"))
    native_draws = nonnegative_int(native.get("draws"))
    native_losses = nonnegative_int(native.get("losses"))
    native_win_rate = finite_number(native.get("win_rate"))
    if (
        nonnegative_int(native_games) is None
        or native_games != measured_games
        or native_wins is None
        or abs(native_wins - sum(measured_wins)) > 1e-9
        or native_draws is None
        or native_draws != sum(measured_draws)
        or native_losses is None
        or native_losses != sum(measured_losses)
        or (native.get("available") is True) != (measured_games > 0)
        or (
            measured_games > 0
            and (
                native_win_rate is None
                or abs(native_win_rate - native_wins / measured_games) > 1e-9
            )
        )
    ):
        return {
            "available": False,
            "reason": "research telemetry game totals do not reconcile",
            "source": str(path),
        }
    checkpoint_digest = str(native.get("checkpoint_digest") or "")
    raw_checkpoint_digests = native.get("checkpoint_digests")
    known_checkpoint_digests: set[str] = set()
    checkpoint_counts_valid = raw_checkpoint_digests is None or isinstance(
        raw_checkpoint_digests, dict
    )
    unknown_checkpoint_games = 0
    checkpoint_games = 0
    if isinstance(raw_checkpoint_digests, dict):
        for digest, count in raw_checkpoint_digests.items():
            parsed_count = nonnegative_int(count)
            if parsed_count is None:
                checkpoint_counts_valid = False
                break
            if parsed_count <= 0:
                continue
            checkpoint_games += parsed_count
            if str(digest) == "unknown":
                unknown_checkpoint_games += parsed_count
            else:
                known_checkpoint_digests.add(str(digest))
    digest_invalid = bool(
        measured_games > 0
        and (
            not checkpoint_counts_valid
            or (
                isinstance(raw_checkpoint_digests, dict)
                and checkpoint_games != measured_games
            )
            or unknown_checkpoint_games > 0
            or native.get("checkpoint_mixed") is True
            or len(known_checkpoint_digests) > 1
            or not _is_sha256_digest(checkpoint_digest)
            or (
                known_checkpoint_digests
                and known_checkpoint_digests != {checkpoint_digest}
            )
        )
    )
    if digest_invalid:
        return {
            "available": False,
            "reason": "research telemetry checkpoint digest is missing or mixed",
            "source": str(path),
        }

    measured_by_id = {
        str(row.get("opponent_id") or ""): row for row in measured
    }
    rows: list[dict[str, Any]] = []
    total_games = 0
    weighted_wins = 0.0
    for control in controls:
        opponent_id = str(control["opponent_id"])
        measurement = measured_by_id.get(opponent_id, {})
        games = nonnegative_int(measurement.get("games")) or 0
        win_rate = as_float(measurement.get("win_rate"))
        total_games += games
        if win_rate is not None:
            weighted_wins += win_rate * games
        rows.append(
            {
                **control,
                "games": games,
                "win_rate": win_rate,
                "wins": as_float(measurement.get("wins")),
                "draws": nonnegative_int(measurement.get("draws")) or 0,
                "losses": nonnegative_int(measurement.get("losses")) or 0,
                "seat0": nonnegative_int(measurement.get("seat0")) or 0,
                "seat1": nonnegative_int(measurement.get("seat1")) or 0,
            }
        )
    return {
        "available": True,
        "schema": "poke_bot.dashboard_research_controls/v1",
        "registry_id": registry.get("registry_id"),
        "registry_version": int(registry["version"]),
        "source": str(path),
        "result_source": (
            str(Path(measurement_source).resolve())
            if measurement_source is not None
            else None
        ),
        "controls": rows,
        "control_count": len(rows),
        "games": total_games,
        "win_rate": weighted_wins / total_games if total_games else None,
        "checkpoint_digest": checkpoint_digest or None,
        "checkpoint_mixed": False,
        "iteration": native.get("iteration"),
        "active": native.get("active") is True,
        "stage": native.get("stage") or "waiting",
        "definition": (
            "per-iteration additive greedy diagnostic controls; excluded from "
            "training/replay and active-gate pass/fail"
        ),
    }


def _offset_public_mix_iterations(
    public_mix_live: dict[str, Any],
    global_iteration_offset: int,
) -> dict[str, Any]:
    """Apply a lineage handoff offset to public and nested research telemetry.

    ``lineage_iteration`` makes this idempotent so a dashboard retry cannot
    accidentally add the handoff offset twice.
    """

    shifted = dict(public_mix_live)

    def shift(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        lineage_iteration = result.get("lineage_iteration")
        if not isinstance(lineage_iteration, int) or isinstance(
            lineage_iteration, bool
        ):
            lineage_iteration = result.get("iteration")
        if isinstance(lineage_iteration, int) and not isinstance(
            lineage_iteration, bool
        ):
            result["lineage_iteration"] = lineage_iteration
            result["iteration"] = lineage_iteration + int(global_iteration_offset)
        return result

    shifted = shift(shifted)
    for nested_name in ("research_controls", "strong_public_practice"):
        nested = shifted.get(nested_name)
        if isinstance(nested, dict):
            shifted[nested_name] = shift(nested)
    return shifted


def competition_gate_program_state(
    official_heldout: dict[str, Any],
    public_mix_live: dict[str, Any],
    *,
    contract_path: Path = COMPETITION_GATE_PROGRAM,
    registry_path: Path = PROTECTED_BASELINE_GATE,
    exact_result_override: dict[str, Any] | None = None,
    exact_result_source: Path | None = None,
    completed_iteration: int | None = None,
) -> dict[str, Any]:
    """Reconcile the accepted gate and the next public-agent gate fail-closed.

    The exact heldout result, protected model registry, and owner decision are
    distinct facts.  The dashboard may call the prior milestone ``accepted``
    only when all three point to the same checkpoint and exact game totals.
    Sampled public-mix trajectories are kept as a separately labeled progress
    diagnostic; they can never populate the exact gate result.
    """

    contract = read_json(contract_path)
    registry = read_json(registry_path)
    if contract.get("schema") != "poke_bot.competition_gate_program/v1":
        return {
            "available": False,
            "reason": "competition gate program is missing or has the wrong schema",
            "source": str(contract_path),
        }

    active_gate_id = str(contract.get("active_gate_id") or "")
    active_semantics = (
        contract.get("active_gate_semantics")
        if isinstance(contract.get("active_gate_semantics"), dict)
        else {}
    )
    accepted_contract = contract.get("accepted_gate")
    next_contract = contract.get("next_gate")
    if not isinstance(accepted_contract, dict) or not isinstance(next_contract, dict):
        return {
            "available": False,
            "reason": "competition gate program is incomplete",
            "source": str(contract_path),
        }

    accepted_digest = str(accepted_contract.get("checkpoint_digest") or "")
    exact_expected = accepted_contract.get("exact_holdout")
    if not isinstance(exact_expected, dict):
        exact_expected = {}
    registry_digest = str(registry.get("checkpoint_digest") or "")
    registry_evidence = (
        registry.get("evidence")
        if isinstance(registry.get("evidence"), dict)
        else {}
    )
    registry_audit = (
        registry_evidence.get("audit")
        if isinstance(registry_evidence.get("audit"), dict)
        else {}
    )
    expected_games = int(exact_expected.get("games") or 0)
    registry_games = int(registry_evidence.get("games") or 0)
    expected_wr = as_float(exact_expected.get("win_rate"))
    registry_wr = as_float(registry_evidence.get("win_rate"))
    expected_lower = as_float(exact_expected.get("confidence_lower"))
    registry_lower = as_float(registry_evidence.get("confidence_lower"))
    exact_values_match = bool(
        expected_wr is not None
        and registry_wr is not None
        and abs(expected_wr - registry_wr) <= 1e-12
        and expected_lower is not None
        and registry_lower is not None
        and abs(expected_lower - registry_lower) <= 1e-12
    )
    accepted_reconciled = bool(
        accepted_contract.get("status") == "accepted"
        and accepted_digest
        and accepted_digest == registry_digest
        and str(registry_evidence.get("checkpoint_digest") or "")
        == accepted_digest
        and str(registry_audit.get("checkpoint_digest") or "")
        == accepted_digest
        and registry_audit.get("passed") is True
        and registry_audit.get("exact_distribution") is True
        and registry_audit.get("exact_weights") is True
        and registry_audit.get("greedy_required") is True
        and expected_games > 0
        and registry_games == expected_games
        and int(registry_audit.get("valid_games") or 0) == expected_games
        and exact_values_match
        and registry.get("immutable") is True
        and registry.get("automatic_pruning_allowed") is False
    )
    submissions = [
        row
        for row in (accepted_contract.get("submissions") or [])
        if isinstance(row, dict)
    ]
    accepted = {
        "available": True,
        "accepted": accepted_reconciled,
        "status": "accepted" if accepted_reconciled else "identity mismatch",
        "id": accepted_contract.get("id"),
        "label": accepted_contract.get("label"),
        "checkpoint_digest": accepted_digest,
        "exact_holdout": dict(exact_expected) if accepted_reconciled else {},
        "decision_basis": accepted_contract.get("decision_basis"),
        "raw_legacy_gate": accepted_contract.get("raw_legacy_gate") or {},
        "submissions": submissions,
        "submission_bundle_sha256": accepted_contract.get(
            "submission_bundle_sha256"
        ),
        "identity_reconciled": accepted_reconciled,
        "registry_protected": bool(
            registry.get("immutable") is True
            and registry.get("automatic_pruning_allowed") is False
        ),
    }

    roster = [
        reconcile_frozen_specialist_label(dict(row))
        for row in (next_contract.get("roster") or [])
        if isinstance(row, dict)
    ]
    evaluation = (
        next_contract.get("evaluation")
        if isinstance(next_contract.get("evaluation"), dict)
        else {}
    )
    pass_criteria = (
        next_contract.get("pass_criteria")
        if isinstance(next_contract.get("pass_criteria"), dict)
        else {}
    )
    fallback = (
        contract.get("fallback_transition")
        if isinstance(contract.get("fallback_transition"), dict)
        else {}
    )
    fallback_activate_after = int(
        fallback.get("activate_after_completed_iteration", -1)
    )
    observed_result = (
        exact_result_override
        if isinstance(exact_result_override, dict)
        else {}
    )
    prior_gate_passed = bool(
        observed_result.get("gate_id") == next_contract.get("id")
        and observed_result.get("passed") is True
    )
    fallback_active = bool(
        isinstance(completed_iteration, int)
        and completed_iteration >= fallback_activate_after >= 0
        and fallback.get("only_if_prior_gate_unpassed") is True
        and str(fallback.get("prior_gate_id") or "")
        == str(next_contract.get("id") or "")
        and str(fallback.get("id") or "")
        and not prior_gate_passed
    )
    effective_pass_criteria = dict(pass_criteria)
    effective_gate_id = str(next_contract.get("id") or "")
    if fallback_active:
        effective_gate_id = str(fallback["id"])
        effective_pass_criteria["skill_weighted_confidence_lower"] = float(
            fallback["skill_weighted_confidence_lower"]
        )
    roster_ids = [str(row.get("opponent_id") or "") for row in roster]
    content_digests = [str(row.get("content_digest") or "") for row in roster]
    research_measurements = [
        dict(row)
        for row in (next_contract.get("research_measurements") or [])
        if isinstance(row, dict)
    ]
    research_ids = [
        str(row.get("opponent_id") or "") for row in research_measurements
    ]
    research_valid = bool(
        len(research_measurements) == 4
        and set(research_ids) == set(OFFICIAL_BASELINE_IDS)
        and len(research_ids) == len(set(research_ids))
        and sum(int(row.get("games") or 0) for row in research_measurements) == 1000
        and all(
            int(row.get("games") or 0) == 250
            and int(row.get("seat0_games") or 0) == 125
            and int(row.get("seat1_games") or 0) == 125
            and bool(str(row.get("archetype_id") or "").strip())
            and bool(str(row.get("archetype_label") or "").strip())
            and (as_float(row.get("gate_weight")) or 0.0) == 0.0
            and row.get("diagnostic_only") is True
            and row.get("included_in_gate_pass") is False
            for row in research_measurements
        )
    )
    per_opponent_games = int(evaluation.get("games_per_opponent") or 0)
    seat0 = int(evaluation.get("seat0_games_per_opponent") or 0)
    seat1 = int(evaluation.get("seat1_games_per_opponent") or 0)
    total_games = int(evaluation.get("games_total") or 0)
    original_four_gate_weight = as_float(
        active_semantics.get("original_four_gate_weight")
    )
    semantics_valid = bool(
        active_gate_id
        and active_gate_id == str(next_contract.get("id") or "")
        and int(active_semantics.get("gate_roster_size") or 0) == len(roster)
        and int(active_semantics.get("games_per_opponent") or 0)
        == per_opponent_games
        and int(active_semantics.get("gate_games_total") or 0) == total_games
        and active_semantics.get("original_four_role") == "research_control_only"
        and original_four_gate_weight is not None
        and original_four_gate_weight == 0.0
    )
    roster_valid = bool(
        roster
        and semantics_valid
        and all(roster_ids)
        and all(str(row.get("archetype_id") or "").strip() for row in roster)
        and all(str(row.get("archetype_label") or "").strip() for row in roster)
        and len(roster_ids) == len(set(roster_ids))
        and set(roster_ids).isdisjoint(OFFICIAL_BASELINE_IDS)
        and set(roster_ids).isdisjoint(research_ids)
        and all(content_digests)
        and len(content_digests) == len(set(content_digests))
        and all((as_float(row.get("weight")) or 0.0) > 0.0 for row in roster)
        and per_opponent_games > 0
        and seat0 + seat1 == per_opponent_games
        and total_games == len(roster) * per_opponent_games
        and int(evaluation.get("minimum_games_per_opponent") or 0)
        == per_opponent_games
        and evaluation.get("all_matchups_must_complete") is True
        and evaluation.get("partial_results_gate_eligible") is False
        and evaluation.get("sequential_early_stop") is False
        and evaluation.get("mode") == "greedy"
        and evaluation.get("fixed_seed_manifest_required") is True
        and evaluation.get("formal_eval_disjoint_from_training") is True
        and evaluation.get("checkpoint_digest_required") is True
        and evaluation.get("package_digest_deduplicated") is True
        and research_valid
    )

    sampled_rows = {
        str(row.get("opponent_id") or ""): row
        for row in (public_mix_live.get("matchups") or [])
        if isinstance(row, dict)
    }
    diagnostic_rows: list[dict[str, Any]] = []
    weighted_score = 0.0
    covered_weight = 0.0
    diagnostic_games = 0
    for member in roster:
        opponent_id = str(member.get("opponent_id") or "")
        sampled = sampled_rows.get(opponent_id) or {}
        games = int(sampled.get("games") or 0)
        wr = as_float(sampled.get("win_rate"))
        weight = as_float(member.get("weight")) or 0.0
        if games > 0 and wr is not None and weight > 0:
            weighted_score += weight * wr
            covered_weight += weight
            diagnostic_games += games
        diagnostic_rows.append(
            {
                "opponent_id": opponent_id,
                "tier": member.get("tier"),
                "weight": weight,
                "games": games,
                "wr": wr,
                "seat0": int(sampled.get("seat0") or 0),
                "seat1": int(sampled.get("seat1") or 0),
                "content_digest": member.get("content_digest"),
            }
        )
    matchup_games = sum(
        int(row.get("games") or 0)
        for row in (public_mix_live.get("matchups") or [])
        if isinstance(row, dict)
    )
    public_games = int(public_mix_live.get("games") or 0)
    diagnostic_valid = bool(
        public_mix_live.get("available") is True
        and public_mix_live.get("checkpoint_mixed") is not True
        and public_mix_live.get("checkpoint_digest")
        and matchup_games == public_games
        and covered_weight > 0
    )
    diagnostic = {
        "available": diagnostic_valid,
        "definition": next_contract.get("diagnostic_definition"),
        "iteration": public_mix_live.get("iteration"),
        "checkpoint_digest": public_mix_live.get("checkpoint_digest"),
        "games": diagnostic_games,
        "roster_coverage": (
            sum(1 for row in diagnostic_rows if int(row["games"]) > 0)
            / len(diagnostic_rows)
            if diagnostic_rows
            else 0.0
        ),
        "skill_weighted_wr": (
            weighted_score / covered_weight if diagnostic_valid else None
        ),
        "rows": diagnostic_rows,
        "source": str(next_contract.get("diagnostic_pointer") or ""),
    }

    if exact_result_override is not None:
        exact_result = dict(exact_result_override)
        result_path = exact_result_source
    else:
        configured_result_path = str(next_contract.get("exact_result_pointer") or "")
        result_path = Path(configured_result_path) if configured_result_path else None
        exact_result = read_json(result_path) if result_path is not None else {}
    result_matchups = [
        row
        for row in (exact_result.get("matchups") or [])
        if isinstance(row, dict)
    ]
    result_ids = [str(row.get("opponent_id") or "") for row in result_matchups]
    result_distribution_valid = bool(
        len(result_matchups) == len(roster)
        and len(result_ids) == len(set(result_ids))
        and set(result_ids) == set(roster_ids)
        and all(
            int(row.get("games") or 0) == per_opponent_games
            and int(row.get("seat0") or 0) == seat0
            and int(row.get("seat1") or 0) == seat1
            for row in result_matchups
        )
    )
    result_audit = (
        exact_result.get("audit")
        if isinstance(exact_result.get("audit"), dict)
        else {}
    )
    fixed_seed_manifest = (
        result_audit.get("fixed_seed_manifest")
        if isinstance(result_audit.get("fixed_seed_manifest"), dict)
        else {}
    )
    fixed_seed_evidence = bool(
        result_audit.get("fixed_seeds") is True
        or (
            int(fixed_seed_manifest.get("gate_games") or 0) == total_games
            and bool(str(fixed_seed_manifest.get("mapping") or "").strip())
            and bool(str(result_audit.get("fixed_seed_manifest_digest") or "").strip())
        )
    )
    result_checkpoint_digest = str(exact_result.get("checkpoint_digest") or "")
    exact_attempt_valid = bool(
        exact_result.get("schema") == "poke_bot.public_agent_gate_result/v1"
        and exact_result.get("gate_id") == next_contract.get("id")
        and result_checkpoint_digest
        and int(exact_result.get("games") or 0) == total_games
        and result_distribution_valid
        and result_audit.get("passed") is True
        and str(result_audit.get("checkpoint_digest") or "")
        == result_checkpoint_digest
        and result_audit.get("exact_distribution") is True
        and result_audit.get("both_seats") is True
        and result_audit.get("greedy") is True
        and fixed_seed_evidence
    )
    exact_result_valid = exact_attempt_valid
    result_checks = (
        exact_result.get("checks")
        if isinstance(exact_result.get("checks"), dict)
        else {}
    )
    required_result_checks = (
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
        "s_plus_matchup_floor_allowance",
    )
    exact_passed = bool(
        exact_result_valid
        and exact_result.get("passed") is True
        and all(result_checks.get(name) is True for name in required_result_checks)
    )
    next_gate = {
        "available": roster_valid,
        "status": (
            "passed"
            if exact_passed
            else "failed"
            if exact_result_valid
            else str(next_contract.get("status") or "queued")
        ),
        "id": next_contract.get("id"),
        "label": next_contract.get("label"),
        "purpose": next_contract.get("purpose"),
        "candidate_source": next_contract.get("candidate_source"),
        "evaluation": evaluation,
        "pass_criteria": pass_criteria,
        "effective_gate_id": effective_gate_id,
        "effective_pass_criteria": effective_pass_criteria,
        "fallback_active": fallback_active,
        "threshold_transition": (
            {
                "status": "active",
                "prior_gate_id": next_contract.get("id"),
                "effective_gate_id": effective_gate_id,
                "activate_after_completed_iteration": fallback_activate_after,
                "observed_completed_iteration": completed_iteration,
                "only_changed_criterion": (
                    "skill_weighted_confidence_lower"
                ),
                "prior_confidence_lower": pass_criteria.get(
                    "skill_weighted_confidence_lower"
                ),
                "effective_confidence_lower": effective_pass_criteria.get(
                    "skill_weighted_confidence_lower"
                ),
            }
            if fallback_active
            else {
                "status": "staged" if fallback else "absent",
                "activate_after_completed_iteration": (
                    fallback_activate_after if fallback else None
                ),
            }
        ),
        "milestones": next_contract.get("milestones") or [],
        "roster": roster,
        "excluded_aliases": next_contract.get("excluded_aliases") or [],
        "research_measurements": research_measurements,
        "research_measurements_valid": research_valid,
        "diagnostic": diagnostic,
        "exact_result_available": exact_result_valid,
        "exact_result": exact_result if exact_result_valid else {},
        "latest_exact_attempt_available": exact_attempt_valid,
        "latest_exact_attempt_current": exact_result_valid,
        "latest_exact_attempt": exact_result if exact_attempt_valid else {},
        "exact_result_source": str(result_path) if result_path is not None else None,
        "contract_valid": roster_valid,
        "contract_reason": (
            None
            if roster_valid
            else "active gate identity, semantics, roster, or exact allocation is invalid"
        ),
    }
    return {
        "available": True,
        "active_gate_id": active_gate_id,
        "active_gate_semantics": active_semantics,
        "accepted_gate": accepted,
        "next_gate": next_gate,
        "source": str(contract_path),
        "updated_at_utc": contract.get("updated_at_utc"),
    }


def strong_public_practice_plan_state(
    run_dir: Path | None,
    iteration: int | None,
    active_gate: dict[str, Any] | None,
    *,
    global_iteration_offset: int = 0,
) -> dict[str, Any]:
    """Read one immutable training-only plan and reconcile it to the active gate.

    This deliberately does not scan backward.  A prior iteration's plan is
    useful history, but showing it as the current allocation would make the
    dashboard lie during startup or after a launch regression.
    """

    if run_dir is None or not isinstance(iteration, int) or iteration < 0:
        return {
            "available": False,
            "reason": "current run or iteration is unavailable",
        }
    plan_path = run_dir / "collection_plans" / f"iter_{iteration:05d}.json"
    if not plan_path.is_file():
        return {
            "available": False,
            "iteration": iteration + global_iteration_offset,
            "reason": "current iteration practice plan is not written yet",
            "source": str(plan_path),
        }

    plan = read_json(plan_path)
    gate = active_gate if isinstance(active_gate, dict) else {}
    roster = gate.get("roster") if isinstance(gate.get("roster"), list) else []
    roster_rows = [dict(row) for row in roster if isinstance(row, dict)]
    roster_ids = [str(row.get("opponent_id") or "") for row in roster_rows]
    roster_by_id = {str(row.get("opponent_id") or ""): row for row in roster_rows}
    raw_per_opponent = plan.get("per_opponent")
    per_opponent = raw_per_opponent if isinstance(raw_per_opponent, dict) else {}
    plan_ids = [str(value) for value in per_opponent]
    raw_weights = plan.get("adaptive_weights")
    adaptive_weights = raw_weights if isinstance(raw_weights, dict) else {}
    weight_ids = [str(value) for value in adaptive_weights]

    def finite_positive(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    rows: list[dict[str, Any]] = []
    row_validation_ok = True
    for opponent_id in roster_ids:
        raw_row = per_opponent.get(opponent_id)
        row = raw_row if isinstance(raw_row, dict) else {}
        roster_row = roster_by_id.get(opponent_id) or {}
        games = row.get("games")
        seat0 = row.get("seat0")
        seat1 = row.get("seat1")
        expected_archetype = str(roster_row.get("archetype_id") or "")
        actual_archetype = str(row.get("archetype_id") or "")
        integers_valid = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (games, seat0, seat1)
        )
        row_valid = bool(
            integers_valid
            and games > 0
            and seat0 + seat1 == games
            and abs(seat0 - seat1) <= 1
            and expected_archetype
            and actual_archetype == expected_archetype
            and finite_positive(adaptive_weights.get(opponent_id))
        )
        row_validation_ok = row_validation_ok and row_valid
        rows.append(
            {
                "opponent_id": opponent_id,
                "tier": roster_row.get("tier"),
                "archetype_id": actual_archetype,
                "archetype_label": roster_row.get("archetype_label"),
                "games": games if integers_valid else 0,
                "seat0": seat0 if integers_valid else 0,
                "seat1": seat1 if integers_valid else 0,
                "adaptive_weight": (
                    float(adaptive_weights[opponent_id])
                    if finite_positive(adaptive_weights.get(opponent_id))
                    else None
                ),
            }
        )

    games = plan.get("games")
    temperature = plan.get("temperature")
    seed_namespace = str(plan.get("seed_namespace") or "")
    formal_seed_namespace = str(plan.get("formal_seed_namespace") or "")
    totals_reconcile = bool(
        isinstance(games, int)
        and not isinstance(games, bool)
        and games > 0
        and sum(int(row.get("games") or 0) for row in rows) == games
    )
    contract_aligned = bool(
        gate.get("available") is True
        and gate.get("contract_valid") is True
        and roster_ids
        and all(roster_ids)
        and len(roster_ids) == len(set(roster_ids))
        and set(plan_ids) == set(roster_ids)
        and len(plan_ids) == len(set(plan_ids))
        and set(weight_ids) == set(roster_ids)
        and len(weight_ids) == len(set(weight_ids))
        and str(plan.get("active_gate_id") or "") == str(gate.get("id") or "")
    )
    semantics_valid = bool(
        plan.get("schema") == "poke_bot.strong_public_practice_plan/v1"
        and plan.get("iteration") == iteration
        and plan.get("training_eligible") is True
        and plan.get("formal_eval") is False
        and plan.get("sampled_policy") is True
        and plan.get("seed_disjoint") is True
        and finite_positive(temperature)
        and seed_namespace.startswith("train/")
        and formal_seed_namespace.startswith("eval/")
        and seed_namespace != formal_seed_namespace
    )
    valid = bool(
        contract_aligned
        and semantics_valid
        and row_validation_ok
        and totals_reconcile
    )
    if not valid:
        failed_checks = [
            name
            for name, passed in (
                ("active gate identity/roster", contract_aligned),
                ("training-only sampled semantics", semantics_valid),
                ("per-opponent archetype/seat/weight", row_validation_ok),
                ("game totals", totals_reconcile),
            )
            if not passed
        ]
        return {
            "available": False,
            "iteration": iteration + global_iteration_offset,
            "reason": "practice plan failed: " + ", ".join(failed_checks),
            "source": str(plan_path),
        }

    return {
        "available": True,
        "iteration": iteration + global_iteration_offset,
        "lineage_iteration": iteration,
        "active_gate_id": plan.get("active_gate_id"),
        "games": games,
        "roster_size": len(rows),
        "temperature": float(temperature),
        "sampled_policy": True,
        "training_eligible": True,
        "formal_eval": False,
        "seed_disjoint": True,
        "seed_namespace": seed_namespace,
        "formal_seed_namespace": formal_seed_namespace,
        "per_opponent": rows,
        "source": str(plan_path),
    }


def replay_window_state(
    run_dir: Path | None,
    loop: dict[str, Any],
    manifest: dict[str, Any],
    progress: dict[str, Any],
    raw_training_log: str,
) -> dict[str, Any]:
    """Describe the live rolling replay window without touching trainer state.

    During JSONL ingestion the trainer has the current shard open. Linux
    ``fdinfo`` exposes its byte position, giving the dashboard a real loading
    percentage without adding logging or allocations to the training process.
    """
    if run_dir is None:
        return {"available": False, "stage": "waiting", "percent": None}
    design = manifest.get("design_contract") or {}
    collection = design.get("collection") or {}
    training_design = manifest.get("training_design") or {}
    window = as_number(str(collection.get("replay_window_shards", "")))
    if window is None:
        window = as_number(str(training_design.get("replay_window_shards", "")))
    window = max(1, int(window or 2))
    iteration = progress.get("iteration")
    if not isinstance(iteration, int):
        iteration = as_number(str(loop.get("next_iteration", "")))
    if iteration is None:
        return {
            "available": False,
            "stage": "waiting",
            "percent": None,
            "window_shards": window,
        }
    first = max(0, int(iteration) - window + 1)
    indices = list(range(first, int(iteration) + 1))
    shard_rows: list[dict[str, Any]] = []
    learner = design.get("learner") or {}
    game_contract = design.get("games") or {}
    per_iteration = int(as_number(str(game_contract.get("per_iteration", ""))) or 0)
    if int(iteration) == 0 and window > 1:
        inherited = list(learner.get("initial_replay_shards") or [])[-(window - 1) :]
        for offset, identity in enumerate(inherited, start=1):
            if not isinstance(identity, dict) or not identity.get("path"):
                continue
            path = Path(str(identity["path"])).expanduser()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            shard_rows.append(
                {
                    "iteration": -offset,
                    "path": str(path),
                    "name": path.name,
                    "bytes": size,
                    "inherited": True,
                }
            )
    for index in indices:
        path = run_dir / "shards" / f"iter_{index:05d}.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        # A recovery/gate-only lineage may intentionally collect zero fresh
        # games and train exclusively from its immutable handoff shard.
        if per_iteration == 0 and size == 0:
            continue
        shard_rows.append(
            {"iteration": index, "path": str(path), "name": path.name, "bytes": size}
        )
    total_bytes = sum(int(row["bytes"]) for row in shard_rows)
    stage = str(progress.get("stage") or "")
    collecting = stage.startswith("collect:")
    ready_shards = sum(
        1
        for row in shard_rows
        if int(row["bytes"]) > 0
        and (int(row["iteration"]) < int(iteration) or not collecting)
    )
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    unit = "shards"
    state = "READY"
    detail = f"{ready_shards}/{len(shard_rows)} shards ready"

    clean_log = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    train_begin = re.findall(
        rf"\[pure_rl\] train begin iter={int(iteration)} seqs=(\d+)", clean_log
    )
    sequences = int(train_begin[-1]) if train_begin else None
    cache: dict[str, Any] = {}
    status_candidates = [run_dir / "replay_window.cache.status.json"]
    status_candidates.extend(
        Path(str(row["path"])).parent.parent / "replay_window.cache.status.json"
        for row in shard_rows
    )
    row_sources = {
        str(Path(str(row["path"])).resolve()) for row in shard_rows
    }
    for status_path in dict.fromkeys(status_candidates):
        cache_raw = read_json(status_path)
        try:
            cache_source = Path(str(cache_raw.get("source_shard") or "")).resolve()
            cache_age = max(
                0.0, time.time() - float(cache_raw.get("updated_at") or 0.0)
            )
            if str(cache_source) not in row_sources or cache_age > 300.0:
                continue
            candidate = {
                "stage": cache_raw.get("stage"),
                "source_shard": str(cache_source),
                "workers": cache_raw.get("workers"),
                "parts_complete": cache_raw.get("parts_complete"),
                "parts_total": cache_raw.get("parts_total"),
                "bytes_complete": cache_raw.get("bytes_complete"),
                "bytes_total": cache_raw.get("bytes_total"),
                "percent": cache_raw.get("percent"),
                "sequences": cache_raw.get("sequences")
                or cache_raw.get("sequences_loaded"),
                "age_s": cache_age,
            }
            if not cache or cache_age < float(cache.get("age_s") or float("inf")):
                cache = candidate
        except (OSError, TypeError, ValueError):
            continue

    if collecting:
        cache_stage = str(cache.get("stage") or "")
        state = (
            "BUILDING + FEATURIZING WINDOW"
            if cache_stage == "streaming_featurize"
            else "BUILDING WINDOW"
        )
        if isinstance(progress.get("percent"), (int, float)):
            percent = float(progress["percent"])
        current = progress.get("current") if isinstance(progress.get("current"), int) else None
        total = progress.get("total") if isinstance(progress.get("total"), int) else None
        unit = str(progress.get("unit") or "games")
        detail = (
            f"writing iter_{int(iteration):05d}.jsonl"
            + (f" · {current}/{total} {unit}" if current is not None and total else "")
        )
        if cache_stage == "streaming_featurize":
            done = int(cache.get("parts_complete") or 0)
            submitted = int(cache.get("parts_total") or 0)
            workers = int(cache.get("workers") or 0)
            detail += (
                f" · stream cache {done}/{submitted} chunks complete"
                f" on {workers} CPU workers"
            )
    elif stage == "train:preparing":
        state = "LOADING WINDOW"
        unit = "bytes"
        open_row: dict[str, Any] | None = None
        loaded_bytes = 0
        pid_text = run(["pgrep", "-f", "scripts/train_pure_rl.py"], timeout=2)
        by_path = {
            str(Path(str(row["path"])).resolve()): (offset, row)
            for offset, row in enumerate(shard_rows)
        }
        for pid_raw in pid_text.splitlines():
            pid = as_number(pid_raw.strip())
            if not pid:
                continue
            fd_root = Path(f"/proc/{pid}/fd")
            try:
                fds = list(fd_root.iterdir())
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                try:
                    resolved_target = str(Path(target).resolve())
                except OSError:
                    resolved_target = target
                matched = by_path.get(resolved_target)
                if matched is None:
                    continue
                offset, row = matched
                pos = 0
                try:
                    for line in Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text().splitlines():
                        if line.startswith("pos:"):
                            pos = int(line.split()[1])
                            break
                except (OSError, ValueError, IndexError):
                    pass
                loaded_bytes = sum(int(item["bytes"]) for item in shard_rows[:offset])
                loaded_bytes += min(max(0, pos), int(row["bytes"]))
                open_row = row
                break
            if open_row is not None:
                break
        if open_row is not None and total_bytes > 0:
            current = loaded_bytes
            total = total_bytes
            percent = min(100.0, 100.0 * loaded_bytes / total_bytes)
            detail = f"reading {open_row['name']} · {percent:.1f}% of window bytes"
        elif sequences is not None:
            percent = 100.0
            current = total_bytes
            total = total_bytes
            state = "WINDOW READY"
            detail = f"assembled {sequences:,} train sequences · preparing AWR"
        else:
            detail = f"opening {len(shard_rows)}-shard window · measuring byte position"
        cache_stage = str(cache.get("stage") or "")
        if cache_stage in {"parallel_featurize", "cache_load", "stream_cache_ready"}:
            state = {
                "parallel_featurize": "PARALLEL FEATURIZING",
                "cache_load": "LOADING FEATURE CACHE",
                "stream_cache_ready": "STREAM CACHE READY",
            }[cache_stage]
            if isinstance(cache.get("percent"), (int, float)):
                percent = float(cache["percent"])
            done = int(cache.get("parts_complete") or 0)
            count = int(cache.get("parts_total") or 0)
            detail = f"{state.lower()} · {done}/{count} chunks"
    elif stage.startswith("train:") or stage in {"heldout", "promotion"}:
        percent = 100.0
        current = total_bytes
        total = total_bytes
        unit = "bytes"
        state = "WINDOW READY"
        detail = (
            f"{sequences:,} train sequences · {len(shard_rows)} shard window"
            if sequences is not None
            else f"{len(shard_rows)} shard window retained on disk"
        )

    return {
        "available": True,
        "iteration": int(iteration),
        "stage": state,
        "detail": detail,
        "percent": percent,
        "current": current,
        "total": total,
        "unit": unit,
        "window_shards": window,
        "target_shards": len(shard_rows),
        "ready_shards": ready_shards,
        "bytes_total": total_bytes,
        "sequences": sequences,
        "shards": shard_rows,
        "cache": cache,
    }


def _tqdm_rate(timing: str, units: tuple[str, ...]) -> tuple[float | None, str | None]:
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    match = re.search(rf"([0-9.]+)({unit_pattern})", timing)
    if not match:
        return None, None
    return float(match.group(1)), match.group(2)


def _tqdm_eta(timing: str) -> str | None:
    match = re.search(r"<([^,\]]+)", timing)
    return match.group(1).strip() if match else None


def annotate_expert_optimizer_sps(
    progress: dict[str, Any],
    raw_training_log: str,
) -> dict[str, Any]:
    """Convert the live expert batch rate to exact optimizer sample SPS.

    The expert tqdm reports batches/second, while the device-corpus pack line
    records the exact train and validation sample counts.  Combining those
    two run-bound values avoids both a blank SPS card and reuse of rollout SPS
    from the preceding collection phase.
    """
    stage = str(progress.get("stage") or "")
    if stage not in {"train:expert", "train:expert:validation"}:
        return progress
    if isinstance(progress.get("sps"), (int, float)):
        return progress
    rate = as_float(progress.get("rate"))
    total_batches = as_number(str(progress.get("total") or ""))
    rate_unit = str(progress.get("rate_unit") or "")
    if rate is None or rate <= 0.0 or not total_batches or total_batches <= 0:
        return progress
    if rate_unit == "batch/s":
        batches_per_second = rate
    elif rate_unit == "s/batch":
        batches_per_second = 1.0 / rate
    else:
        return progress

    split_rows = list(
        re.finditer(
            r"\[device-corpus\]\s+CPU pack=.*?\bsamples=(\d+)\s+"
            r"train=(\d+)\s+val=(\d+)",
            ANSI_RE.sub("", raw_training_log).replace("\r", "\n"),
        )
    )
    if not split_rows:
        return progress
    _all_samples, train_samples, val_samples = (
        int(value) for value in split_rows[-1].groups()
    )
    split_samples = (
        val_samples if stage == "train:expert:validation" else train_samples
    )
    if split_samples <= 0:
        return progress

    enriched = dict(progress)
    enriched["sps"] = batches_per_second * split_samples / total_batches
    enriched["sps_source"] = "exact device-corpus split × live tqdm batch rate"
    enriched["optimizer_samples"] = split_samples
    return enriched


def parse_curriculum_progress(
    raw_status: str,
    raw_progress_log: str,
    *,
    iteration_hint: int | None = None,
) -> dict[str, Any]:
    """Parse the newest collect, policy-epoch, or validation tqdm frame.

    ``*.progress.status`` is intentionally written by the game collector only.
    Mid-iteration training uses tqdm directly, so its outer epoch and nested
    validation bars live in the run-specific ``*.progress.log`` instead.  We
    preserve stream order and take the newest recognized frame, never a bar
    from a different run or an older global alias.
    """
    progress: dict[str, Any] = {
        "line": raw_status.strip(),
        "stage": None,
        "iteration": iteration_hint,
        "epoch": None,
        "percent": None,
        "current": None,
        "total": None,
        "unit": None,
        "rate": None,
        "rate_unit": None,
        "eta": None,
        "gps": None,
        "sps": None,
        "remotes": None,
        "metrics": {},
    }
    clean_log = ANSI_RE.sub("", raw_progress_log).replace("\r", "\n")
    lines = []
    for raw_line in clean_log.splitlines():
        # A forced service stop can append Python's resource-tracker warning
        # directly after a truncated tqdm frame. Preserve the valid progress
        # prefix and discard the unrelated shutdown warning.
        line = re.split(
            r"(?=/[^\s]*multiprocessing/resource_tracker\.py:|UserWarning:\s*resource_tracker:)",
            raw_line,
            maxsplit=1,
        )[0].strip()
        if line and not line.startswith("warnings.warn("):
            lines.append(line)
    # A just-created progress log can be empty for its first instant. Only in
    # that case use the already run-bound single-line status mirror.
    if not lines and raw_status.strip():
        lines = [ANSI_RE.sub("", raw_status).strip()]

    last_train_metrics: dict[str, float | None] = {}
    for line in lines:
        resident_pack = re.search(
            r"pack Blackwell corpus:\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if resident_pack:
            percent, current, total, timing = resident_pack.groups()
            rate, rate_unit = _tqdm_rate(timing, ("game/s", "s/game"))
            progress.update(
                line=line,
                stage="train:packing",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="games",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        replay_cache = re.search(
            r"replay-cache load\s+(\S+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if replay_cache:
            shard_name, percent, current, total, timing = replay_cache.groups()
            rate, rate_unit = _tqdm_rate(timing, ("part/s", "s/part"))
            progress.update(
                line=line,
                stage="train:preparing",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="parts",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={"replay_shard": shard_name},
            )
            continue

        collect = re.search(
            r"pure_rl\s+(\S+)\s+iter=(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if collect:
            stage, iteration, percent, current, total, timing = collect.groups()
            if stage == "train:expert":
                progress.update(
                    line=line,
                    stage="train:expert:loading",
                    iteration=int(iteration),
                    epoch=0,
                    epochs=None,
                    percent=float(percent),
                    current=int(current),
                    total=int(total),
                    unit="expert pass",
                    rate=None,
                    rate_unit=None,
                    eta="loading corpus",
                    gps=None,
                    sps=None,
                    remotes=0,
                    metrics={},
                )
                last_train_metrics = {}
                continue
            rate, rate_unit = _tqdm_rate(timing, ("game/s", "s/game"))
            gps = None
            if rate is not None:
                gps = rate if rate_unit == "game/s" else 1.0 / max(rate, 1e-9)
            request_sockets = parse_metric(timing, "rsock")
            if request_sockets is None:
                # Backward-compatible read for immutable pre-revision-124
                # progress logs, where ``remotes`` meant request sockets.
                request_sockets = parse_metric(timing, "remotes")
            remote_demand = parse_metric(timing, "rdmd")
            remote_outstanding = parse_metric(timing, "rout")
            remote_outstanding_elmo = parse_metric(timing, "eout")
            remote_outstanding_bert = parse_metric(timing, "bout")
            # With socket prefetch, ``remotes`` is the number of admitted TCP
            # requests, while ``rdmd`` remains execution-worker demand.  Keep
            # the public ``remotes`` metric at worker grain so the dashboard
            # never labels queued requests as extra simulator workers.
            remote_workers = (
                int(remote_demand)
                if remote_demand is not None
                else int(request_sockets)
                if request_sockets is not None
                else None
            )
            remote_queue_capacity = (
                max(0, int(request_sockets) - int(remote_workers))
                if request_sockets is not None and remote_workers is not None
                else None
            )
            sps = parse_metric(timing, "sps")
            progress.update(
                line=line,
                stage=stage,
                iteration=int(iteration),
                epoch=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="games",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=gps,
                sps=sps,
                remotes=remote_workers,
                metrics={
                    key: value
                    for key, value in {
                        "remote_request_sockets": (
                            int(request_sockets)
                            if request_sockets is not None
                            else None
                        ),
                        "remote_queue_capacity": remote_queue_capacity,
                        "remote_outstanding": (
                            int(remote_outstanding)
                            if remote_outstanding is not None
                            else None
                        ),
                        "remote_outstanding_elmo": (
                            int(remote_outstanding_elmo)
                            if remote_outstanding_elmo is not None
                            else None
                        ),
                        "remote_outstanding_bert": (
                            int(remote_outstanding_bert)
                            if remote_outstanding_bert is not None
                            else None
                        ),
                    }.items()
                    if value is not None
                },
            )
            last_train_metrics = {}
            continue

        expert_loading = re.search(
            r"pure_rl train:expert iter=(\d+):\s*(\d+)%.*?"
            r"(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_loading:
            iteration, percent, current, total, timing = expert_loading.groups()
            progress.update(
                line=line,
                stage="train:expert:loading",
                iteration=int(iteration),
                epoch=0,
                epochs=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="expert pass",
                rate=None,
                rate_unit=None,
                eta="loading corpus",
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        expert_batch = re.search(
            r"expert rehearsal before iter(\d+) ep(\d+)/(\d+):\s*"
            r"(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_batch:
            (
                iteration,
                epoch,
                epochs,
                percent,
                current,
                total,
                timing,
            ) = expert_batch.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            metrics = {
                name: parse_metric(timing, name)
                for name in (
                    "acc",
                    "loss",
                    "policy",
                    "value",
                    "aux",
                    "hand",
                    "rem",
                    "lethal",
                    "prize",
                    "guide",
                    "step",
                )
            }
            metrics = {
                key: value for key, value in metrics.items() if value is not None
            }
            progress.update(
                line=line,
                stage="train:expert",
                iteration=int(iteration),
                epoch=int(epoch),
                epochs=int(epochs),
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=metrics,
            )
            continue

        expert_validation = re.search(
            r"expert validation before iter(\d+):\s*(\d+)%.*?"
            r"(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_validation:
            iteration, percent, current, total, timing = expert_validation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage="train:expert:validation",
                iteration=int(iteration),
                epoch=None,
                epochs=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        training_batch = re.search(
            r"rl-train\s+ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if training_batch:
            epoch, percent, current, total, timing = training_batch.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            optimizer_sps = parse_metric(timing, "sps")
            metrics = {
                name: parse_metric(timing, name)
                for name in (
                    "acc",
                    "loss",
                    "p",
                    "v",
                    "hand",
                    "rem",
                    "aux",
                    "lethal",
                    "prize",
                    "guide",
                )
            }
            metrics = {key: value for key, value in metrics.items() if value is not None}
            if metrics:
                last_train_metrics = metrics
            progress.update(
                line=line,
                stage="train:policy",
                iteration=iteration_hint,
                epoch=int(epoch) + 1,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=optimizer_sps,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
            continue

        adapter_batch = re.search(
            r"rl-adapters\s+ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if adapter_batch:
            epoch, percent, current, total, timing = adapter_batch.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage="train:matchup-adapters:shadow",
                iteration=iteration_hint,
                epoch=int(epoch) + 1,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={
                    key: value
                    for key, value in {
                        "loss": parse_metric(timing, "loss"),
                        "rows": parse_metric(timing, "rows"),
                    }.items()
                    if value is not None
                },
            )
            continue

        preparation = re.search(
            r"rl-(prep|agreement)\s+(baseline|parent|candidate):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if preparation:
            family, phase, percent, current, total, timing = preparation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage=f"train:{family}:{phase}",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        train = re.search(
            r"rl-train:\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if train:
            percent, current, total, timing = train.groups()
            rate, rate_unit = _tqdm_rate(timing, ("ep/s", "s/ep"))
            metrics = {
                name: parse_metric(timing, name)
                for name in (
                    "acc",
                    "loss",
                    "p",
                    "v",
                    "hand",
                    "rem",
                    "aux",
                    "lethal",
                    "prize",
                    "guide",
                    "best",
                    "pat",
                )
            }
            metrics = {key: value for key, value in metrics.items() if value is not None}
            if metrics:
                last_train_metrics = metrics
            progress.update(
                line=line,
                stage="train:policy",
                iteration=iteration_hint,
                epoch=int(current),
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="epochs",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
            continue

        validation = re.search(
            r"rl-val\s+ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if validation:
            epoch, percent, current, total, timing = validation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage="train:validation",
                iteration=iteration_hint,
                epoch=int(epoch) + 1,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
    return progress


def annotate_collection_budget(
    progress: dict[str, Any], raw_training_log: str
) -> dict[str, Any]:
    """Label simulation attempts separately from exact retained game quotas."""

    stage = str(progress.get("stage") or "")
    iteration = progress.get("iteration")
    if not stage.startswith("collect:self_play") or not isinstance(iteration, int):
        return progress
    matches = re.findall(
        r"collect iter=(\d+) bounded self-play refill capacity=(\d+) "
        r"primary_self_play=(\d+) target_games=(\d+)",
        ANSI_RE.sub("", raw_training_log),
    )
    match = next(
        (row for row in reversed(matches) if int(row[0]) == int(iteration)),
        None,
    )
    if match is None:
        return progress
    _it, reserve, primary, iteration_target = match
    out = dict(progress)
    metrics = dict(out.get("metrics") or {})
    metrics.update(
        {
            "simulation_attempts": True,
            "primary_retained_target": int(primary),
            "reserve_attempt_capacity": int(reserve),
            "iteration_retained_target": int(iteration_target),
            "unused_reserve_training_eligible": False,
        }
    )
    out["metrics"] = metrics
    return out


def infer_between_bar_progress(
    progress: dict[str, Any],
    raw_training_log: str,
    *,
    iteration_hint: int | None,
    train_epochs: int = 2,
) -> dict[str, Any]:
    """Expose CPU replay/AWR preparation between collect and tqdm epochs."""
    if iteration_hint is None or str(progress.get("stage") or "").startswith("train"):
        return progress
    clean = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    marker = ""
    marker_kind = ""
    for line in lines:
        patterns = (
            ("collect_done", rf"\[pure_rl\] collect done iter={iteration_hint}(?:\s|$)"),
            ("train_begin", rf"\[pure_rl\] train begin iter={iteration_hint}(?:\s|$)"),
            ("train_released", rf"\[pure_rl\] replay memory released iter={iteration_hint}(?:\s|$)"),
            ("promotion", rf"\[pure_rl\] promotion begin iter={iteration_hint}(?:\s|$)"),
        )
        for kind, pattern in patterns:
            if re.search(pattern, line):
                marker, marker_kind = line, kind
    if marker_kind not in {"collect_done", "train_begin"}:
        return progress
    # Only replace the just-completed collection bar for this same iteration.
    if progress.get("iteration") != iteration_hint or float(progress.get("percent") or 0) < 100:
        return progress
    updated = dict(progress)
    updated.update(
        stage="train:preparing",
        epoch=0,
        percent=None,
        current=0,
        total=max(1, int(train_epochs)),
        unit="epochs",
        rate=None,
        rate_unit=None,
        eta="preparing",
        gps=None,
        sps=None,
        remotes=0,
        metrics={},
        line=(
            marker
            if marker_kind == "train_begin"
            else f"[pure_rl] train preparing iter={iteration_hint}: "
            "assembling rolling replay window + AWR baselines"
        ),
    )
    return updated


def reconcile_completed_train_epoch(
    progress: dict[str, Any],
    raw_training_log: str,
    *,
    iteration_hint: int | None,
    train_epochs: int = 2,
) -> dict[str, Any]:
    """Advance a stale tqdm mirror from authoritative completed-epoch lines."""

    if iteration_hint is None:
        return progress
    clean = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    marker = f"[pure_rl] train begin iter={int(iteration_hint)}"
    offset = clean.rfind(marker)
    if offset < 0:
        return progress
    segment = clean[offset:]
    matches = list(
        re.finditer(
            r"\[rl-train\]\s+(?:NEW BEST\s+)?epoch=(\d+)\b[^\n]*",
            segment,
        )
    )
    if not matches:
        return progress
    latest = matches[-1]
    completed_epoch = int(latest.group(1)) + 1
    current_epoch = int(progress.get("epoch") or 0)
    stage = str(progress.get("stage") or "")
    # Never replace a genuinely later phase such as adapter fitting, heldout,
    # promotion, or research controls with an earlier epoch summary.
    if stage not in {
        "",
        "train:preparing",
        "train:policy",
        "train:validation",
    } or completed_epoch < current_epoch:
        return progress
    line = latest.group(0).strip()
    metrics = dict(progress.get("metrics") or {})
    loss = parse_metric(line, "val_loss")
    accuracy = parse_metric(line, "acc")
    if loss is not None:
        metrics["loss"] = loss
    if accuracy is not None:
        metrics["acc"] = accuracy
    updated = dict(progress)
    updated.update(
        line=line,
        stage="train:policy",
        iteration=int(iteration_hint),
        epoch=completed_epoch,
        percent=min(100.0, 100.0 * completed_epoch / max(1, int(train_epochs))),
        current=completed_epoch,
        total=max(1, int(train_epochs)),
        unit="epochs",
        rate=None,
        rate_unit=None,
        eta=("next phase" if completed_epoch >= int(train_epochs) else None),
        gps=None,
        sps=None,
        remotes=0,
        metrics=metrics,
    )
    return updated


def infer_post_train_gate_progress(
    progress: dict[str, Any],
    raw_training_log: str,
    *,
    iteration_hint: int | None,
) -> dict[str, Any]:
    """Expose checkpoint publication between promotion and formal holdout.

    A newly trained candidate has a unique digest even when promotion rejects
    it, because the formal gate must evaluate that exact candidate rather than
    substitute the incumbent. Remote digest publication can take minutes and
    previously left the dashboard showing a completed training bar as if the
    service were degraded.
    """
    if iteration_hint is None:
        return progress
    stage = str(progress.get("stage") or "")
    if (
        stage.startswith("heldout")
        and stage not in {
            "heldout:checkpoint_staging",
            "heldout:starting",
        }
    ) or stage.startswith("measure:") or stage in {
        "promotion",
        "research_controls",
    }:
        return progress
    clean = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    train_marker = f"[pure_rl] train begin iter={int(iteration_hint)}"
    offset = clean.rfind(train_marker)
    if offset < 0:
        return progress
    marker_kind = ""
    marker_line = ""
    for line in (value.strip() for value in clean[offset:].splitlines()):
        if re.search(
            rf"\[pure_rl\] BETWEEN_ITER_HARD_GATE begin iter={int(iteration_hint)}(?:\s|$)",
            line,
        ):
            # A promoted candidate is published before the trainer can emit
            # its PROMOTED line. Reloading a genuinely new digest across the
            # local leaves and remotes can take several minutes, so this begin
            # marker is itself the authoritative active phase.
            marker_kind, marker_line = "checkpoint_staging", line
        elif re.search(
            rf"\[pure_rl\] (?:REJECTED|PROMOTED) iter={int(iteration_hint)}(?:\s|$)",
            line,
        ):
            marker_kind, marker_line = "checkpoint_staging", line
        elif re.search(
            r"\[pure_rl\] BETWEEN_ITER_HARD_GATE ok\b", line
        ):
            marker_kind, marker_line = "heldout_starting", line
        elif re.search(r"\[pure_rl\] heldout local worker cap=", line):
            marker_kind, marker_line = "heldout_starting", line
    if not marker_kind:
        return progress
    updated = dict(progress)
    updated.update(
        stage=(
            "heldout:checkpoint_staging"
            if marker_kind == "checkpoint_staging"
            else "heldout:starting"
        ),
        iteration=int(iteration_hint),
        epoch=None,
        percent=None,
        current=0,
        total=None,
        unit="games",
        rate=None,
        rate_unit=None,
        eta=(
            "publishing candidate weights"
            if marker_kind == "checkpoint_staging"
            else "starting formal games"
        ),
        gps=None,
        sps=None,
        remotes=0,
        metrics={
            "candidate_checkpoint_publication": bool(
                marker_kind == "checkpoint_staging"
            ),
            "formal_holdout": True,
        },
        line=(
            f"[pure_rl] heldout checkpoint staging iter={int(iteration_hint)}: "
            "publishing the exact candidate digest to evaluation endpoints"
            if marker_kind == "checkpoint_staging"
            else marker_line
        ),
    )
    return updated


def _expert_rehearsal_exclusion_seconds(
    run_dir: Path | None,
    iteration: int,
    extra: dict[str, Any] | None = None,
) -> float | None:
    """Return expert-only wall time, or ``None`` when it cannot be proven.

    Rehearsal is an out-of-band correction pass rather than part of curriculum
    iteration throughput.  New receipts can carry an exact duration.  Older
    runs are reconstructed conservatively from the immutable completed-
    collection timestamp through the immutable rehearsal-receipt timestamp.
    A missing rehearsal returns ``0``; a known rehearsal without trustworthy
    timing returns ``None`` so it cannot contaminate iteration averages.
    """
    extra = extra if isinstance(extra, dict) else {}
    record = (
        extra.get("expert_rehearsal")
        if isinstance(extra.get("expert_rehearsal"), dict)
        else {}
    )
    receipt_path = (
        run_dir / "rehearsals" / f"before_iter_{int(iteration):05d}.json"
        if run_dir is not None
        else None
    )
    receipt_exists = bool(receipt_path is not None and receipt_path.is_file())
    if not record and not receipt_exists:
        return 0.0

    for candidate in (
        record.get("wall_elapsed_sec"),
        record.get("elapsed_sec"),
        (record.get("rehearsal") or {}).get("elapsed_sec")
        if isinstance(record.get("rehearsal"), dict)
        else None,
    ):
        seconds = as_float(candidate)
        if seconds is not None and seconds >= 0:
            return seconds

    if run_dir is None or not receipt_exists:
        return None
    collection = read_json(
        run_dir / "collection_receipts" / f"iter_{int(iteration):05d}.json"
    )
    rehearsal = read_json(receipt_path)
    started_at = as_float(collection.get("completed_at"))
    completed_at = as_float(rehearsal.get("completed_at"))
    if completed_at is None:
        try:
            completed_at = receipt_path.stat().st_mtime
        except OSError:
            completed_at = None
    if (
        started_at is None
        or completed_at is None
        or completed_at < started_at
    ):
        return None
    return completed_at - started_at


def _metric_iteration_wall_seconds(
    payload: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> float | None:
    """Return curriculum work time with expert rehearsal removed."""
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    explicit = as_float(extra.get("iteration_wall_sec"))
    if explicit is not None and explicit >= 0:
        wall_seconds = explicit
    else:
        post_collect = as_float(
            extra.get("post_collect_elapsed_sec", extra.get("elapsed_sec"))
        )
        collect_stats = (
            extra.get("collect_stats")
            if isinstance(extra.get("collect_stats"), dict)
            else {}
        )
        collect = as_float(collect_stats.get("collect_elapsed_sec"))
        if post_collect is not None and collect is not None:
            wall_seconds = max(0.0, post_collect) + max(0.0, collect)
        elif post_collect is not None:
            wall_seconds = max(0.0, post_collect)
        else:
            return None
    iteration = payload.get("iteration")
    if not isinstance(iteration, int):
        return wall_seconds
    excluded = _expert_rehearsal_exclusion_seconds(run_dir, iteration, extra)
    if excluded is None:
        return None
    return max(0.0, wall_seconds - excluded)


def iteration_timing_state(
    run_dir: Path | None,
    *,
    active: bool,
    global_iteration_offset: int,
    next_iteration: int | None = None,
    progress_iteration: int | None = None,
    progress_stage: str | None = None,
) -> dict[str, Any]:
    """Source-backed current/latest/rolling iteration timing telemetry."""
    if run_dir is None:
        return {
            "available": False,
            "current_seconds": None,
            "latest_seconds": None,
            "rolling5_seconds": None,
            "history": [],
        }
    history: list[dict[str, Any]] = []
    throughput_history: list[dict[str, Any]] = []
    metrics_dir = run_dir / "metrics"
    for path in sorted(metrics_dir.glob("iter_*.json")):
        payload = read_json(path)
        iteration = payload.get("iteration")
        if not isinstance(iteration, int):
            continue
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        collect_stats = (
            extra.get("collect_stats")
            if isinstance(extra.get("collect_stats"), dict)
            else {}
        )
        games = as_float(payload.get("games"))
        decisions = as_float(payload.get("decisions"))
        collect_seconds = as_float(collect_stats.get("collect_elapsed_sec"))
        gps = as_float(payload.get("games_per_sec"))
        sps = as_float(payload.get("decisions_per_sec"))
        if collect_seconds is not None and collect_seconds > 0:
            if games is not None and games >= 0:
                gps = games / collect_seconds
            if decisions is not None and decisions >= 0:
                sps = decisions / collect_seconds
        if gps is not None or sps is not None:
            throughput_history.append(
                {
                    "iteration": iteration + int(global_iteration_offset),
                    "lineage_iteration": iteration,
                    "gps": gps,
                    "sps": sps,
                    "games": games,
                    "decisions": decisions,
                    "collect_seconds": collect_seconds,
                }
            )
        seconds = _metric_iteration_wall_seconds(payload, run_dir=run_dir)
        if seconds is None:
            continue
        excluded = _expert_rehearsal_exclusion_seconds(
            run_dir,
            iteration,
            payload.get("extra") if isinstance(payload.get("extra"), dict) else {},
        )
        history.append(
            {
                "iteration": iteration + int(global_iteration_offset),
                "lineage_iteration": iteration,
                "seconds": seconds,
                "expert_rehearsal_excluded_seconds": excluded or 0.0,
            }
        )
    history.sort(key=lambda row: int(row["lineage_iteration"]))
    history = history[-20:]
    throughput_history.sort(key=lambda row: int(row["lineage_iteration"]))
    throughput_history = throughput_history[-20:]
    latest = history[-1] if history else None
    rolling = history[-5:]
    latest_throughput = throughput_history[-1] if throughput_history else None
    rolling_throughput = throughput_history[-5:]

    def weighted_rate(rows: list[dict[str, Any]], numerator: str) -> float | None:
        exact = [
            row
            for row in rows
            if as_float(row.get(numerator)) is not None
            and as_float(row.get("collect_seconds")) is not None
            and float(row["collect_seconds"]) > 0
        ]
        if exact:
            return sum(float(row[numerator]) for row in exact) / sum(
                float(row["collect_seconds"]) for row in exact
            )
        rate_name = "gps" if numerator == "games" else "sps"
        rates = [
            float(row[rate_name])
            for row in rows
            if as_float(row.get(rate_name)) is not None
        ]
        return sum(rates) / len(rates) if rates else None

    runtime = read_json(run_dir / "iteration_runtime.json")
    current_iteration = runtime.get("iteration")
    started_at = as_float(runtime.get("started_at"))
    phase = str(runtime.get("phase") or "")
    current_seconds: float | None = None
    display_current_iteration: int | None = None
    current_source: str | None = None
    current_paused_for_expert_rehearsal = False
    if (
        active
        and isinstance(current_iteration, int)
        and (next_iteration is None or current_iteration == next_iteration)
        and started_at is not None
        and phase != "completed"
        and 0 <= started_at <= time.time() + 5
    ):
        raw_current_seconds = max(0.0, time.time() - started_at)
        if str(progress_stage or "").startswith("train:expert"):
            collection = read_json(
                run_dir
                / "collection_receipts"
                / f"iter_{int(current_iteration):05d}.json"
            )
            stats = (
                collection.get("stats")
                if isinstance(collection.get("stats"), dict)
                else {}
            )
            collected_seconds = as_float(stats.get("collect_elapsed_sec"))
            current_seconds = (
                max(0.0, collected_seconds)
                if collected_seconds is not None
                else None
            )
            current_paused_for_expert_rehearsal = True
            current_source = "collection receipt; expert rehearsal excluded"
        else:
            excluded = _expert_rehearsal_exclusion_seconds(
                run_dir, current_iteration
            )
            current_seconds = (
                max(0.0, raw_current_seconds - excluded)
                if excluded is not None
                else None
            )
            current_source = "trainer runtime; expert rehearsal excluded"
        display_current_iteration = current_iteration + int(global_iteration_offset)
    elif (
        active
        and isinstance(next_iteration, int)
        and progress_iteration == next_iteration
        and bool(progress_stage)
    ):
        # A trainer already running when this telemetry feature is deployed
        # cannot import the new runtime writer until its next natural restart.
        # Persist the first dashboard observation of each new progress-bound
        # iteration so browser refreshes and snapshot subprocesses do not
        # reset the live timer. Committed metrics remain the exact authority.
        observed = read_json(DASHBOARD_ITERATION_TIMER)
        observed_started = as_float(observed.get("started_at"))
        if (
            observed.get("run") != run_dir.name
            or observed.get("iteration") != next_iteration
            or observed_started is None
            or observed_started > time.time() + 5
        ):
            observed_started = time.time()
            payload = {
                "schema": "poke_bot.dashboard_iteration_timer/v1",
                "run": run_dir.name,
                "iteration": next_iteration,
                "started_at": observed_started,
                "source": "first run-bound progress observation",
            }
            DASHBOARD_ITERATION_TIMER.parent.mkdir(parents=True, exist_ok=True)
            tmp = DASHBOARD_ITERATION_TIMER.with_name(
                f".{DASHBOARD_ITERATION_TIMER.name}.{os.getpid()}.tmp"
            )
            tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, DASHBOARD_ITERATION_TIMER)
        current_seconds = max(0.0, time.time() - float(observed_started))
        display_current_iteration = next_iteration + int(global_iteration_offset)
        current_source = "dashboard progress observation"
    return {
        "available": bool(history or current_seconds is not None),
        "current_iteration": display_current_iteration,
        "current_seconds": current_seconds,
        "current_source": current_source,
        "current_paused_for_expert_rehearsal": current_paused_for_expert_rehearsal,
        "latest_iteration": latest.get("iteration") if latest else None,
        "latest_seconds": latest.get("seconds") if latest else None,
        "rolling5_seconds": (
            sum(float(row["seconds"]) for row in rolling) / len(rolling)
            if rolling
            else None
        ),
        "rolling5_samples": len(rolling),
        "history": history,
        "latest_throughput_iteration": (
            latest_throughput.get("iteration") if latest_throughput else None
        ),
        "latest_gps": latest_throughput.get("gps") if latest_throughput else None,
        "latest_sps": latest_throughput.get("sps") if latest_throughput else None,
        "rolling5_gps": weighted_rate(rolling_throughput, "games"),
        "rolling5_sps": weighted_rate(rolling_throughput, "decisions"),
        "rolling5_throughput_samples": len(rolling_throughput),
        "throughput_history": throughput_history,
        "source": (
            "committed metrics + persisted live iteration timer; "
            "expert rehearsal excluded"
        ),
    }


def _run_name_from_command(command: str) -> str | None:
    """Extract the launcher's explicit run identity from argv/systemd text."""
    match = re.search(r"(?:^|\s)--run-name(?:=|\s+)([^\s;\]}]+)", str(command))
    return match.group(1).strip("'\"") if match else None


def _specialist_id_from_runtime(command: str, run_name: str | None) -> str | None:
    """Resolve the live specialist without trusting a stale mutable tracker."""
    match = re.search(
        r"(?:^|\s)--specialist-archetype(?:=|\s+)([^\s;\]}]+)",
        str(command),
    )
    if match:
        return match.group(1).strip("'\"").replace("_", "-").lower()
    normalized = str(run_name or "").lower().replace("_", "-")
    for specialist_id in (
        "hops-trevenant",
        "starmie",
        "alakazam",
        "mega-lucario-ex",
        "dragapult-ex",
        "walrein",
        "dudunsparce",
    ):
        if specialist_id in normalized:
            return specialist_id
    return None


def _is_curriculum_service_unit(unit: str) -> bool:
    """Return whether an active managed unit owns live curriculum work."""

    lowered = str(unit).lower()
    if "pure-rl" in lowered or "curriculum" in lowered:
        return True
    if (
        "final-format-alakazam" not in lowered
        and "final-format-marnie" not in lowered
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "-h10.service",
            "-h10-rl.service",
            "-h10-bootstrap.service",
        )
    )


def _active_curriculum_services() -> tuple[list[str], list[int], str | None]:
    """Return active units/PIDs and their authoritative ``--run-name``."""
    units = run(
        [
            "systemctl",
            "--user",
            "--no-legend",
            "--plain",
            "list-units",
            "--type=service",
            "--state=active",
        ]
    )
    active_units: list[str] = []
    active_pids: list[int] = []
    # RemainAfterExit bootstrap units intentionally stay ``active (exited)``.
    # They are useful history, but they are not live trainers.  A no-PID unit
    # must never make the dashboard active or select its historical run.  A
    # newly starting simple service will publish MainPID by the next dashboard
    # sample; showing one brief inactive sample is safer than lying about the
    # lineage/model contract.
    live_run_name: str | None = None
    for line in units.splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not _is_curriculum_service_unit(unit):
            continue
        pid = as_number(
            run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=MainPID",
                    "--value",
                ]
            )
        )
        if not pid:
            continue
        active_units.append(unit)
        command = ""
        active_pids.append(pid)
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace")
        except OSError:
            command = ""
        if not command:
            command = run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=ExecStart",
                    "--value",
                ]
            )
        candidate_run_name = _run_name_from_command(command)
        if candidate_run_name:
            live_run_name = live_run_name or candidate_run_name
    return active_units, active_pids, live_run_name


def _select_curriculum_run_dir(
    root: Path,
    candidates: set[Path],
    active_run_name: str | None,
) -> Path | None:
    """Select the active service's run; mtime is fallback-only history."""
    if active_run_name:
        return root / active_run_name
    return (
        max(
            candidates,
            key=lambda path: max(
                (child.stat().st_mtime for child in path.glob("*.json")),
                default=path.stat().st_mtime,
            ),
        )
        if candidates
        else None
    )


def active_expert_pack_state() -> dict[str, Any]:
    """Measure the live expert CPU-pack reader before tqdm epochs begin.

    Packing is intentionally outside the GPU epoch loop and previously looked
    idle on the dashboard.  The builder holds ``.build.lock`` and reads one
    manifest shard at a time; Linux exposes its exact byte position in fdinfo.
    This is read-only telemetry and survives launcher/dashboard restarts.
    """
    candidates: list[dict[str, Any]] = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        fd_dir = process_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except OSError:
            continue
        targets: dict[Path, str] = {}
        for descriptor in descriptors:
            try:
                targets[descriptor] = os.readlink(descriptor)
            except OSError:
                continue
        if not any(target.endswith("/expert_cpu_pack/.build.lock") for target in targets.values()):
            continue
        feature_fd = next(
            (
                descriptor
                for descriptor, target in targets.items()
                if target.endswith(".features")
            ),
            None,
        )
        if feature_fd is None:
            candidates.append(
                {"active": True, "pid": int(process_dir.name), "phase": "finalizing"}
            )
            continue
        feature_path = Path(targets[feature_fd])
        manifest_path = feature_path.parent / "manifest.json"
        manifest = read_json(manifest_path)
        shards = [row for row in manifest.get("shards") or [] if isinstance(row, dict)]
        names = [Path(str(row.get("path") or "")).name for row in shards]
        try:
            index = names.index(feature_path.name)
        except ValueError:
            index = 0
        sizes = [max(0, int(row.get("bytes") or 0)) for row in shards]
        position = 0
        try:
            match = re.search(r"^pos:\s*(\d+)$", (process_dir / "fdinfo" / feature_fd.name).read_text(), re.M)
            if match:
                position = int(match.group(1))
        except OSError:
            pass
        total_bytes = sum(sizes)
        completed_bytes = sum(sizes[:index]) + min(position, sizes[index] if index < len(sizes) else position)
        read_chars = 0
        try:
            io_match = re.search(
                r"^rchar:\s*(\d+)$", (process_dir / "io").read_text(), re.M
            )
            if io_match:
                read_chars = int(io_match.group(1))
        except OSError:
            pass
        # Full-file digest verification, episode grouping, train materialize,
        # and validation materialize each stream the immutable shards once.
        expected_passes = 4
        candidates.append({
            "active": True,
            "pid": int(process_dir.name),
            "phase": "packing_corpus",
            "current_shard": feature_path.name,
            "current_shard_number": index + 1,
            "total_shards": len(shards),
            "completed_bytes": completed_bytes,
            "total_bytes": total_bytes,
            "percent": (100.0 * completed_bytes / total_bytes) if total_bytes > 0 else None,
            "expected_passes": expected_passes,
            "read_chars": read_chars,
            "overall_percent_estimate": (
                min(99.9, 100.0 * read_chars / (expected_passes * total_bytes))
                if total_bytes > 0 and read_chars > 0
                else None
            ),
            "source": str(manifest_path),
        })
    if not candidates:
        return {"active": False}
    readers = [row for row in candidates if row.get("phase") == "packing_corpus"]
    if readers:
        # DataLoader workers inherit the pack lock and may read different
        # shards concurrently.  Show the furthest proven file position rather
        # than whichever /proc PID happens to sort first.
        furthest = max(
            readers,
            key=lambda row: (
                int(row.get("current_shard_number") or 0),
                int(row.get("completed_bytes") or 0),
            ),
        )
        furthest["active_readers"] = len(readers)
        furthest["active_shards"] = sorted(
            {str(row.get("current_shard")) for row in readers if row.get("current_shard")}
        )
        return furthest
    result = candidates[0]
    result["active_readers"] = len(candidates)
    return result


def expert_rehearsal_state(
    run_dir: Path | None,
    contract: dict[str, Any],
    loop: dict[str, Any],
    progress: dict[str, Any],
    *,
    global_iteration_offset: int,
    trainer_active: bool,
) -> dict[str, Any]:
    """Describe the recurring expert tune-up separately from bootstrap."""
    every = int(contract.get("every_iterations") or 0)
    epochs = int(contract.get("epochs") or 0)
    lineage_iteration = loop.get("next_iteration")
    due = bool(
        every > 0
        and isinstance(lineage_iteration, int)
        and lineage_iteration > 0
        and lineage_iteration % every == 0
    )
    receipts: list[Path] = []
    if run_dir is not None:
        receipts = sorted(
            (run_dir / "rehearsals").glob("before_iter_*.json"),
            key=lambda path: path.name,
        )
    latest_receipt = read_json(receipts[-1]) if receipts else {}
    latest_before = latest_receipt.get("before_iteration")
    latest_metric_sets = latest_receipt.get("metrics")
    latest_metric_sets = (
        latest_metric_sets if isinstance(latest_metric_sets, dict) else {}
    )
    latest_train_metrics = latest_metric_sets.get("train")
    latest_train_metrics = (
        latest_train_metrics if isinstance(latest_train_metrics, dict) else {}
    )
    latest_validation_metrics = latest_metric_sets.get("validation")
    latest_validation_metrics = (
        latest_validation_metrics
        if isinstance(latest_validation_metrics, dict)
        else {}
    )
    loss_weights = latest_receipt.get("loss_weights")
    loss_weights = loss_weights if isinstance(loss_weights, dict) else {}
    adapter_contract = contract.get("matchup_adapters")
    adapter_contract = (
        adapter_contract if isinstance(adapter_contract, dict) else {}
    )
    adapter_enabled = bool(adapter_contract.get("enabled"))
    adapter_receipts: list[Path] = []
    adapter_authorization: Path | None = None
    adapter_progress: dict[str, Any] = {}
    if run_dir is not None:
        adapter_root = run_dir / "rehearsals" / "matchup_adapters"
        adapter_receipts = sorted(
            (
                path
                for path in adapter_root.glob("before_iter_*.json")
                if not path.name.endswith(".authorization.json")
            ),
            key=lambda path: path.name,
        )
        if isinstance(lineage_iteration, int):
            adapter_stem = f"before_iter_{lineage_iteration:05d}"
            adapter_authorization = (
                adapter_root / f"{adapter_stem}.authorization.json"
            )
            adapter_progress = read_json(
                adapter_root / f"{adapter_stem}.fit" / "progress.json"
            )
    latest_adapter_receipt = (
        read_json(adapter_receipts[-1]) if adapter_receipts else {}
    )
    latest_adapter_fit = latest_adapter_receipt.get("fit")
    latest_adapter_fit = (
        latest_adapter_fit if isinstance(latest_adapter_fit, dict) else {}
    )
    latest_adapter_before = latest_adapter_receipt.get("before_iteration")
    adapter_complete_current = bool(
        due
        and isinstance(lineage_iteration, int)
        and latest_adapter_before == lineage_iteration
    )
    adapter_progress_complete = adapter_progress.get("complete") is True
    adapter_running = bool(
        trainer_active
        and due
        and adapter_enabled
        and not adapter_complete_current
        and bool(adapter_progress)
        and not adapter_progress_complete
    )
    adapter_epoch = (
        as_number(str(adapter_progress.get("epoch")))
        if "epoch" in adapter_progress
        else None
    )
    adapter_epochs = as_number(
        str(
            adapter_progress.get("epochs")
            or adapter_contract.get("epochs")
            or ""
        )
    )
    adapter_cursor = (
        as_number(str(adapter_progress.get("train_sequences_consumed")))
        if "train_sequences_consumed" in adapter_progress
        else None
    )
    adapter_train_sequences = as_number(
        str(adapter_progress.get("train_sequences") or "")
    )
    adapter_percent: float | None = None
    if (
        isinstance(adapter_epoch, (int, float))
        and isinstance(adapter_epochs, (int, float))
        and isinstance(adapter_cursor, (int, float))
        and isinstance(adapter_train_sequences, (int, float))
        and adapter_epochs > 0
        and adapter_train_sequences > 0
    ):
        adapter_percent = 100.0 * (
            adapter_epoch * adapter_train_sequences + adapter_cursor
        ) / (adapter_epochs * adapter_train_sequences)

    def receipt_count(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    latest_head_receipt = {
        "policy": {
            "rows": receipt_count(latest_train_metrics.get("n_decisions")),
            "loss": as_float(latest_validation_metrics.get("policy_loss")),
            "weight": 1.0,
        },
        "value": {
            "rows": receipt_count(latest_train_metrics.get("n_decisions")),
            "loss": as_float(latest_validation_metrics.get("value_loss")),
            "weight": as_float(loss_weights.get("value")),
        },
        "archetype": {
            "rows": receipt_count(latest_train_metrics.get("n_archetype_rows")),
            "loss": as_float(latest_validation_metrics.get("aux_loss")),
            "weight": as_float(loss_weights.get("archetype")),
        },
        "opponent_hand": {
            "rows": receipt_count(latest_train_metrics.get("n_opp_hand_rows")),
            "loss": as_float(latest_validation_metrics.get("opp_hand_loss")),
            "weight": as_float(loss_weights.get("opponent_hand")),
        },
        "opponent_remainder": {
            "rows": receipt_count(latest_train_metrics.get("n_opp_remainder_rows")),
            "loss": as_float(latest_validation_metrics.get("opp_remainder_loss")),
            "weight": as_float(loss_weights.get("opponent_hidden_remainder")),
        },
        "lethal_threat": {
            "rows": receipt_count(latest_train_metrics.get("n_lethal_threat_rows")),
            "loss": as_float(latest_validation_metrics.get("lethal_threat_loss")),
            "weight": as_float(loss_weights.get("lethal_threat")),
        },
        "prize_race": {
            "rows": receipt_count(latest_train_metrics.get("n_prize_race_rows")),
            "loss": as_float(latest_validation_metrics.get("prize_race_loss")),
            "weight": as_float(loss_weights.get("prize_race")),
        },
        "alakazam_guide": {
            "rows": receipt_count(latest_train_metrics.get("n_alakazam_guide_rows")),
            "loss": as_float(latest_validation_metrics.get("alakazam_guide_loss")),
            "weight": as_float(loss_weights.get("alakazam_guide")),
        },
        "matchup_adapters": {
            "rows": receipt_count(
                latest_adapter_fit.get("phase_rows")
                if latest_adapter_fit
                else latest_train_metrics.get("n_matchup_adapter_rows")
            ),
            "loss": None,
            "weight": 1.0 if latest_adapter_fit else 0.0,
        },
    }
    completed_current = bool(
        due
        and isinstance(lineage_iteration, int)
        and latest_before == lineage_iteration
    )
    progress_stage = str(progress.get("stage") or "")
    pack = active_expert_pack_state() if trainer_active and due and not completed_current else {"active": False}
    running = bool(
        trainer_active
        and (
            progress_stage.startswith("train:expert")
            or pack.get("active")
            or adapter_running
        )
    )
    if every <= 0:
        state = "disabled"
    elif adapter_running:
        state = "running · adapter-only phase"
    elif running:
        state = "running"
    elif due and not completed_current:
        state = "due · waiting/retry"
    elif completed_current:
        state = "complete for this boundary"
    else:
        state = "scheduled"
    next_lineage: int | None = None
    if every > 0 and isinstance(lineage_iteration, int):
        if due and not completed_current:
            next_lineage = lineage_iteration
        else:
            next_lineage = ((lineage_iteration // every) + 1) * every
    current = dict(progress) if progress_stage.startswith("train:expert") else {}
    if adapter_running:
        current = {
            "stage": "train:expert:matchup-adapters",
            "percent": adapter_percent,
            "current": adapter_cursor,
            "total": adapter_train_sequences,
            "unit": "routed sequences",
            "epoch": (
                int(adapter_epoch) + 1
                if isinstance(adapter_epoch, (int, float))
                else None
            ),
            "epochs": (
                int(adapter_epochs)
                if isinstance(adapter_epochs, (int, float))
                else None
            ),
            "eta": "measuring",
            "line": (
                "pure_rl train:expert:matchup-adapters "
                "base frozen · one causal route per sequence"
            ),
        }
    if pack.get("active") and not current:
        current = {
            "stage": "train:expert:packing",
            "percent": pack.get("percent"),
            "current": pack.get("current_shard_number", 0),
            "total": pack.get("total_shards", 0),
            "unit": "corpus shards",
            "eta": "measuring",
            "line": (
                "pure_rl train:expert:packing "
                f"{pack.get('current_shard') or 'finalizing CPU pack'}"
            ),
        }
    if pack.get("active") and pack.get("overall_percent_estimate") is not None:
        current["percent"] = pack.get("overall_percent_estimate")
    return {
        "available": bool(contract),
        "kind": "periodic_expert_tune_up",
        "is_bootstrap": False,
        "state": state,
        "active": running,
        "due": due and not completed_current,
        "every_iterations": every,
        "epochs": epochs,
        "learning_rate": as_float(contract.get("learning_rate")),
        "requested_batch_size": as_number(
            str(contract.get("requested_batch_size") or "")
        ),
        "minimum_decisions": as_number(
            str(contract.get("minimum_decisions") or "")
        ),
        "manifest": contract.get("rolling_manifest_pointer"),
        "current": current,
        "cpu_pack": pack,
        "latest_completed_iteration": (
            int(latest_before) + global_iteration_offset
            if isinstance(latest_before, int)
            else None
        ),
        "latest_receipt": str(receipts[-1]) if receipts else None,
        "latest_checkpoint": latest_receipt.get("checkpoint"),
        "latest_checkpoint_digest": latest_receipt.get("checkpoint_digest"),
        "latest_head_receipt": latest_head_receipt,
        "latest_validation_policy_accuracy": as_float(
            latest_validation_metrics.get("policy_acc")
        ),
        "matchup_adapter_rehearsal": {
            "enabled": adapter_enabled,
            "state": (
                "running"
                if adapter_running
                else "complete for this boundary"
                if adapter_complete_current
                else "authorization staged"
                if (
                    due
                    and adapter_authorization is not None
                    and adapter_authorization.is_file()
                )
                else "due · authorization missing"
                if due and adapter_enabled
                else "scheduled"
                if adapter_enabled
                else "disabled"
            ),
            "active": adapter_running,
            "epochs": as_number(str(adapter_contract.get("epochs") or "")),
            "learning_rate": as_float(
                adapter_contract.get("learning_rate")
            ),
            "games_per_batch": as_number(
                str(adapter_contract.get("games_per_batch") or "")
            ),
            "max_decisions_per_batch": as_number(
                str(
                    adapter_contract.get("max_decisions_per_batch")
                    or ""
                )
            ),
            "optimizer_scope": adapter_contract.get("optimizer_scope"),
            "base_frozen": adapter_contract.get("enabled") is True,
            "runtime_enabled_during_fit": False,
            "manifest": adapter_contract.get("staged_manifest"),
            "authorization": (
                str(adapter_authorization)
                if adapter_authorization is not None
                and adapter_authorization.is_file()
                else None
            ),
            "current": (
                current
                if adapter_running
                and current.get("stage")
                == "train:expert:matchup-adapters"
                else {}
            ),
            "latest_completed_iteration": (
                int(latest_adapter_before) + global_iteration_offset
                if isinstance(latest_adapter_before, int)
                else None
            ),
            "latest_receipt": (
                str(adapter_receipts[-1]) if adapter_receipts else None
            ),
            "latest_checkpoint": latest_adapter_receipt.get("checkpoint"),
            "latest_checkpoint_digest": latest_adapter_receipt.get(
                "checkpoint_digest"
            ),
            "latest_fit": latest_adapter_fit,
        },
        "next_iteration": (
            next_lineage + global_iteration_offset
            if isinstance(next_lineage, int)
            else None
        ),
    }


def strong_public_gate_runtime_state(
    active_gate: dict[str, Any] | None = None,
    *,
    curriculum_progress: dict[str, Any] | None = None,
    curriculum_active: bool = False,
) -> dict[str, Any]:
    """Return live progress normalized to the pinned active-gate contract.

    The service runs the active strong-public gate first and may then run the
    original-four research controls.  Once that second phase starts, its
    1,000-game counter must not replace the completed 2,000-game active gate
    in dashboard telemetry.
    """

    recognized_stages = {
        "heldout:strong_public_gate",
        "measure:research_controls",
    }
    main_progress = (
        curriculum_progress if isinstance(curriculum_progress, dict) else {}
    )
    main_stage = str(main_progress.get("stage") or "")
    if curriculum_active:
        progress = dict(main_progress)
        stage = main_stage
        recognized = stage in recognized_stages
        active = recognized
        updated_at = None
        source = "main curriculum run-bound progress"
    else:
        standalone_active = run(
            ["systemctl", "--user", "is-active", STRONG_PUBLIC_GATE_SERVICE],
            timeout=2,
        ) == "active"
        if standalone_active:
            status = read_tail(STRONG_PUBLIC_GATE_PROGRESS, 20_000).strip()
            log = read_tail(STRONG_PUBLIC_GATE_LOG, 300_000)
            progress = parse_curriculum_progress(status, log)
            stage = str(progress.get("stage") or "")
            recognized = stage in recognized_stages
        else:
            progress = {}
            stage = ""
            recognized = False
        active = bool(standalone_active and recognized)
        updated_at = max(
            (
                path.stat().st_mtime
                for path in (STRONG_PUBLIC_GATE_PROGRESS, STRONG_PUBLIC_GATE_LOG)
                if path.is_file()
            ),
            default=None,
        )
        source = str(STRONG_PUBLIC_GATE_PROGRESS)
    age_s = max(0.0, time.time() - updated_at) if updated_at else None
    gate = active_gate if isinstance(active_gate, dict) else {}
    evaluation = (
        gate.get("evaluation") if isinstance(gate.get("evaluation"), dict) else {}
    )
    roster = gate.get("roster") if isinstance(gate.get("roster"), list) else []
    gate_total = int(evaluation.get("games_total") or 0)
    games_per_opponent = int(evaluation.get("games_per_opponent") or 0)
    roster_size = len(roster)
    contract_aligned = bool(
        gate.get("available") is True
        and gate.get("contract_valid") is True
        and roster_size > 0
        and games_per_opponent > 0
        and gate_total == roster_size * games_per_opponent
    )
    raw_current = int(progress.get("current") or 0) if recognized else 0
    raw_total = int(progress.get("total") or 0) if recognized else 0
    active_phase_aligned = bool(
        stage != "heldout:strong_public_gate"
        or (
            contract_aligned
            and raw_total == gate_total
            and 0 <= raw_current <= gate_total
        )
    )
    if contract_aligned and stage == "heldout:strong_public_gate" and active_phase_aligned:
        gate_current = raw_current
    elif contract_aligned and stage == "measure:research_controls":
        gate_current = gate_total
    else:
        gate_current = 0
    gate_percent = (
        100.0 * gate_current / gate_total
        if contract_aligned and gate_total > 0
        else None
    )
    return {
        "available": contract_aligned,
        "telemetry_available": bool(active or recognized),
        "active": active,
        "current": gate_current,
        "total": gate_total if contract_aligned else 0,
        "percent": gate_percent,
        "roster_size": roster_size if contract_aligned else 0,
        "games_per_opponent": games_per_opponent if contract_aligned else 0,
        "allocation_label": (
            f"{roster_size} x {games_per_opponent}"
            if contract_aligned
            else None
        ),
        "contract_aligned": contract_aligned,
        "progress_aligned": active_phase_aligned,
        "active_gate_complete": bool(contract_aligned and gate_current == gate_total),
        "phase_current": raw_current,
        "phase_total": raw_total,
        "iteration": progress.get("iteration") if recognized else None,
        "stage": stage if recognized else ("starting" if active else "idle"),
        "phase": (
            "active_gate"
            if stage == "heldout:strong_public_gate"
            else "research_controls"
            if stage == "measure:research_controls"
            else "starting"
            if active
            else "idle"
        ),
        "gps": progress.get("gps") if recognized else None,
        "sps": progress.get("sps") if recognized else None,
        "remotes": progress.get("remotes") if recognized else None,
        "line": progress.get("line") if recognized else None,
        "updated_at": updated_at,
        "age_s": age_s,
        "source": source,
    }


def curriculum_state() -> dict[str, Any]:
    root = ROOT / "outputs/pure_rl"
    active_units, active_pids, active_run_name = _active_curriculum_services()
    candidates = {
        p.parent
        for pattern in ("*/manifest.json", "*/loop_state.json")
        for p in root.glob(pattern)
        if p.is_file()
    }
    run_dir = _select_curriculum_run_dir(
        root,
        candidates,
        active_run_name,
    )
    metrics = read_json(run_dir / "metrics/latest.json") if run_dir else {}
    loop = read_json(run_dir / "loop_state.json") if run_dir else {}
    manifest = read_json(run_dir / "manifest.json") if run_dir else {}
    handoff = read_json(run_dir / "lineage_handoff.json") if run_dir else {}
    # A fail-closed heldout repair can finish after the trainer intentionally
    # stops but before the normal append-only iteration commit.  That exact
    # audit is newer and more authoritative than inherited historical WR, so
    # surface it explicitly instead of leaving the dashboard on a stale tqdm.
    recovery_path: Path | None = None
    recovery: dict[str, Any] = {}
    if run_dir is not None:
        recovery_candidates = sorted(
            (run_dir / "eval").glob("iter_*.heldout_recovery.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if recovery_candidates:
            recovery_path = recovery_candidates[-1]
            candidate = read_json(recovery_path)
            audit = candidate.get("audit")
            gate = candidate.get("gate")
            if (
                isinstance(audit, dict)
                and audit.get("passed") is True
                and isinstance(gate, dict)
                and int(gate.get("games") or 0) == int(audit.get("valid_games") or -1)
            ):
                recovery = candidate
    global_iteration_offset = int(handoff.get("global_iteration_offset") or 0)
    official_heldout = committed_official_heldout_state(
        loop,
        run_dir,
        global_iteration_offset=global_iteration_offset,
        handoff=handoff,
    )
    latest_official_heldout = latest_committed_official_heldout_state(
        loop,
        run_dir,
        global_iteration_offset=global_iteration_offset,
    )
    latest_formal_holdout = latest_committed_formal_holdout_state(
        loop,
        run_dir,
        global_iteration_offset=global_iteration_offset,
    )
    matchup_runtime = matchup_runtime_collection_state(run_dir)
    run_name = (
        active_run_name
        or loop.get("run_name")
        or (run_dir.name if run_dir else None)
    )
    run_status = (
        ROOT / "outputs/logs" / f"{run_name}.progress.status"
        if run_name
        else None
    )
    run_progress_log = (
        ROOT / "outputs/logs" / f"{run_name}.progress.log"
        if run_name
        else None
    )
    # Once a run identity exists, never fall back to the global alias: it may
    # still point at a previous lineage during the first seconds of launch.
    status_path = run_status if run_status is not None else TRAINING_STATUS
    raw_status = read_tail(status_path, 20_000).strip()
    raw_progress_log = (
        read_tail(run_progress_log, 500_000)
        if run_progress_log is not None
        else ""
    )
    iteration_hint = as_number(str(loop.get("next_iteration", "")))
    progress = parse_curriculum_progress(
        raw_status,
        raw_progress_log,
        iteration_hint=iteration_hint,
    )
    run_training_log = (
        ROOT / "outputs/logs" / f"{run_name}.log" if run_name else TRAINING_LOG
    )
    raw_training_log = read_tail(
        run_training_log if run_training_log.is_file() else TRAINING_LOG,
        250_000,
    )
    progress = reconcile_completed_train_epoch(
        progress,
        raw_training_log,
        iteration_hint=iteration_hint,
        train_epochs=2,
    )
    progress = annotate_collection_budget(progress, raw_training_log)
    progress = infer_between_bar_progress(
        progress,
        raw_training_log,
        iteration_hint=iteration_hint,
        train_epochs=2,
    )
    progress = infer_post_train_gate_progress(
        progress,
        raw_training_log,
        iteration_hint=iteration_hint,
    )
    progress = annotate_expert_optimizer_sps(progress, raw_training_log)
    replay_window = replay_window_state(
        run_dir,
        loop,
        manifest,
        progress,
        raw_training_log,
    )
    lineage_iteration = progress.get("iteration")
    display_progress = dict(progress)
    if global_iteration_offset and isinstance(lineage_iteration, int):
        display_iteration = lineage_iteration + global_iteration_offset
        display_progress["lineage_iteration"] = lineage_iteration
        display_progress["iteration"] = display_iteration
        display_progress["line"] = re.sub(
            rf"\biter={lineage_iteration}\b",
            f"iter={display_iteration}",
            str(display_progress.get("line") or ""),
        )
    if global_iteration_offset and isinstance(replay_window.get("iteration"), int):
        replay_window["lineage_iteration"] = replay_window["iteration"]
        replay_window["iteration"] = (
            int(replay_window["iteration"]) + global_iteration_offset
        )
    effective_design = effective_design_contract_for_run(run_dir, manifest)
    # The manifest is an immutable launch root.  Its learner batch caps can be
    # superseded at a receipt-backed boundary, so the model panel must consume
    # the verified effective contract just as the trainer does rather than
    # displaying the stale launch-time cap.
    model_manifest = dict(manifest)
    model_manifest["design_contract"] = effective_design
    expert_contract = effective_design.get("expert_rehearsal") or {}
    expert_receipt = (
        run_dir / "rehearsals" / "before_iter_00000.json"
        if run_dir is not None
        else None
    )
    expert_startup_pending = bool(
        active_units
        and int(loop.get("next_iteration") or 0) == 0
        and expert_contract.get("before_first_iteration") is True
        and expert_receipt is not None
        and not expert_receipt.is_file()
        and not raw_status
        and progress.get("stage") is None
    )
    if expert_startup_pending:
        lineage_it = int(loop.get("next_iteration") or 0)
        display_it = lineage_it + global_iteration_offset
        progress.update(
            {
                "line": (
                    f"pure_rl train:expert iter={display_it}: loading exact "
                    "top-ladder corpus onto Blackwell"
                ),
                "stage": "train:expert",
                "iteration": lineage_it,
                "percent": None,
                "current": 0,
                "total": 1,
                "unit": "expert pass",
                "eta": "loading corpus",
            }
        )
        display_progress = dict(progress)
        display_progress["lineage_iteration"] = lineage_it
        display_progress["iteration"] = display_it
    expert_rehearsal = expert_rehearsal_state(
        run_dir,
        expert_contract,
        loop,
        display_progress,
        global_iteration_offset=global_iteration_offset,
        trainer_active=bool(active_units),
    )
    worker = curriculum_worker_state(active_units, active_pids)
    iteration_timing = iteration_timing_state(
        run_dir,
        active=bool(active_units),
        global_iteration_offset=global_iteration_offset,
        next_iteration=(
            int(loop.get("next_iteration"))
            if isinstance(loop.get("next_iteration"), int)
            else None
        ),
        progress_iteration=(
            int(progress.get("iteration"))
            if isinstance(progress.get("iteration"), int)
            else None
        ),
        progress_stage=str(progress.get("stage") or "") or None,
    )
    public_mix_live = read_json(PUBLIC_MIX_LIVE_WR)
    public_mix_age = (
        max(0.0, time.time() - float(public_mix_live.get("updated_at") or 0.0))
        if public_mix_live
        else None
    )
    public_mix_iteration = public_mix_live.get("iteration")
    if (
        not public_mix_live
        or public_mix_live.get("run") != run_name
        or not isinstance(public_mix_iteration, int)
        or public_mix_age is None
        or public_mix_age > 15.0
    ):
        public_mix_live = {
            "available": False,
            "active": False,
            "reason": "live public-mix outcome sidecar is unavailable or stale",
        }
    else:
        public_mix_live = _offset_public_mix_iterations(
            public_mix_live,
            global_iteration_offset,
        )
        public_mix_live["age_s"] = public_mix_age
    committed_research_result, committed_research_source = (
        latest_committed_research_control_result(run_dir)
    )
    research_controls = research_control_registry_state(
        public_mix_live,
        measurement_result=committed_research_result,
        measurement_source=committed_research_source,
    )
    active_gate_contract_path = active_gate_contract_for_run(run_dir)
    gate_contract = read_json(active_gate_contract_path)
    configured_next_gate = gate_contract.get("next_gate")
    configured_result_pointer: Path | None = None
    if isinstance(configured_next_gate, dict):
        raw_result_pointer = str(
            configured_next_gate.get("exact_result_pointer") or ""
        ).strip()
        if raw_result_pointer:
            configured_result_pointer = Path(raw_result_pointer)
    committed_gate_result, committed_gate_source = (
        latest_committed_active_gate_result(
            run_dir,
            mutable_result_pointer=configured_result_pointer,
        )
    )
    gate_program = competition_gate_program_state(
        official_heldout,
        public_mix_live,
        contract_path=active_gate_contract_path,
        # Never let a mutable pointer bypass immutable curriculum history.  The
        # helper above returns that pointer only after an exact commit match.
        exact_result_override=committed_gate_result,
        exact_result_source=committed_gate_source,
        completed_iteration=(
            int(loop["last_completed_iteration"])
            if isinstance(loop.get("last_completed_iteration"), int)
            else None
        ),
    )
    if isinstance(gate_program.get("next_gate"), dict):
        active_gate = gate_program["next_gate"]
        active_gate["runtime"] = strong_public_gate_runtime_state(
            active_gate,
            curriculum_progress=display_progress,
            curriculum_active=bool(active_units),
        )
    else:
        active_gate = {}
    practice_iteration = (
        int(progress["iteration"])
        if isinstance(progress.get("iteration"), int)
        else (
            int(loop["next_iteration"])
            if isinstance(loop.get("next_iteration"), int)
            else None
        )
    )
    strong_public_practice = strong_public_practice_plan_state(
        run_dir,
        practice_iteration,
        active_gate,
        global_iteration_offset=global_iteration_offset,
    )
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    promotion = extra.get("promotion") if isinstance(extra.get("promotion"), dict) else {}
    champion = loop.get("champion") if isinstance(loop.get("champion"), dict) else {}
    inherited_heldout = (
        handoff.get("inherited_heldout")
        if isinstance(handoff.get("inherited_heldout"), dict)
        else {}
    )
    metrics_iteration = metrics.get("iteration")
    recovery_iteration = recovery.get("iteration")
    recovery_is_latest = bool(recovery) and (
        not isinstance(metrics_iteration, int)
        or not isinstance(recovery_iteration, int)
        or recovery_iteration >= metrics_iteration
    )
    heldout_inherited = (
        metrics.get("heldout_wr") is None
        and not recovery_is_latest
        and bool(inherited_heldout)
    )
    recovery_gate = recovery.get("gate") if recovery_is_latest else {}
    heldout_wr = (
        recovery_gate.get("win_rate")
        if recovery_is_latest
        else (
            inherited_heldout.get("win_rate")
            if heldout_inherited
            else metrics.get("heldout_wr")
        )
    )
    heldout_games = (
        recovery_gate.get("games")
        if recovery_is_latest
        else (
            inherited_heldout.get("games")
            if heldout_inherited
            else metrics.get("heldout_games")
        )
    )
    gate_passed = (
        recovery_gate.get("passed")
        if recovery_is_latest
        else (
            inherited_heldout.get("passed")
            if heldout_inherited
            else metrics.get("gate_passed")
        )
    )
    if recovery_is_latest and not active_units:
        display_iteration = (
            int(recovery_iteration) + global_iteration_offset
            if isinstance(recovery_iteration, int)
            else display_progress.get("iteration")
        )
        display_progress.update(
            {
                "line": (
                    f"pure_rl heldout COMPLETE iter={display_iteration}: "
                    f"{int(heldout_games or 0)}/{int(heldout_games or 0)} "
                    f"[WR={float(heldout_wr or 0.0) * 100:.1f}% · exact audit PASS]"
                ),
                "stage": "heldout:complete",
                "iteration": display_iteration,
                "lineage_iteration": recovery_iteration,
                "percent": 100.0,
                "current": int(heldout_games or 0),
                "total": int(heldout_games or 0),
                "unit": "games",
                "eta": "done",
            }
        )
    status_updated_at = status_path.stat().st_mtime if status_path.is_file() else None
    log_updated_at = (
        run_progress_log.stat().st_mtime
        if run_progress_log is not None and run_progress_log.is_file()
        else None
    )
    progress_updated_at = max(
        (value for value in (status_updated_at, log_updated_at) if value is not None),
        default=None,
    )
    status_age_s = time.time() - progress_updated_at if progress_updated_at else None
    progress_current = bool(
        run_name
        and run_status is not None
        and status_path == run_status
        and status_path.is_file()
    ) or bool(
        run_progress_log is not None
        and run_progress_log.is_file()
        and progress.get("stage") is not None
    ) or expert_startup_pending
    assertions = {
        "active_service_has_pid": not active_units or bool(active_pids),
        "run_identity_present": not active_units or bool(run_name),
        "active_run_is_authoritative": (
            not active_run_name
            or bool(run_dir is not None and run_dir.name == active_run_name)
        ),
        "progress_bound_to_run": not active_units or progress_current,
        "progress_not_cross_run": run_status is None or status_path == run_status,
        "progress_log_bound_to_run": (
            not active_units
            or bool(run_progress_log is not None and run_progress_log.is_file())
        ),
    }
    progress_source = (
        run_dir / "manifest.json"
        if expert_startup_pending and run_dir is not None
        else run_progress_log
        if str(progress.get("stage") or "").startswith("train")
        and run_progress_log is not None
        else status_path
    )
    return {
        "active": bool(active_units),
        "active_units": active_units,
        "active_pids": active_pids,
        "run": run_name,
        "last_completed_iteration": (
            int(loop.get("last_completed_iteration")) + global_iteration_offset
            if isinstance(loop.get("last_completed_iteration"), int)
            else loop.get("last_completed_iteration")
        ),
        "next_iteration": (
            int(loop.get("next_iteration")) + global_iteration_offset
            if isinstance(loop.get("next_iteration"), int)
            else loop.get("next_iteration")
        ),
        "global_iteration_offset": global_iteration_offset,
        "lineage_iteration": lineage_iteration,
        "stage": (
            "heldout:complete"
            if recovery_is_latest and not active_units
            else progress["stage"] or metrics.get("stage") or ("starting" if active_units else None)
        ),
        "iteration": (
            display_progress["iteration"]
            if display_progress["iteration"] is not None
            else (
                int(metrics.get("iteration", loop.get("next_iteration")))
                + global_iteration_offset
                if isinstance(metrics.get("iteration", loop.get("next_iteration")), int)
                else metrics.get("iteration", loop.get("next_iteration"))
            )
        ),
        "progress": display_progress,
        "replay_window": replay_window,
        "iteration_timing": iteration_timing,
        "expert_rehearsal": expert_rehearsal,
        "public_mix_live": public_mix_live,
        "research_controls": research_controls,
        "strong_public_practice": strong_public_practice,
        "gate_program": gate_program,
        "progress_source": str(progress_source),
        "progress_status_source": str(status_path),
        "progress_log_source": str(run_progress_log) if run_progress_log else None,
        "progress_updated_at": progress_updated_at,
        "progress_age_s": status_age_s,
        "source_assertions": assertions,
        "source_current": all(assertions.values()),
        "worker": worker,
        "games": metrics.get("games"),
        "gps": progress["gps"] if progress["gps"] is not None else metrics.get("games_per_sec"),
        "sps": progress["sps"] if progress["sps"] is not None else metrics.get("decisions_per_sec"),
        "heldout_wr": heldout_wr,
        "heldout_games": heldout_games,
        "heldout_inherited": heldout_inherited,
        "heldout_recovery": recovery_is_latest,
        "heldout_audit_passed": bool(
            recovery_is_latest and (recovery.get("audit") or {}).get("passed")
        ),
        "heldout_matchups": (
            recovery_gate.get("per_opponent")
            if recovery_is_latest
            else committed_gate_result.get("matchups")
            if committed_gate_result
            else None
        ),
        "heldout_source": (
            str(recovery_path)
            if recovery_is_latest
            else str(committed_gate_source)
            if committed_gate_source is not None
            else None
        ),
        "heldout_source_run": handoff.get("source_run") if heldout_inherited else run_name,
        "heldout_source_iteration": (
            recovery_iteration
            if recovery_is_latest
            else (
                handoff.get("source_iteration")
                if heldout_inherited
                else metrics.get("iteration")
            )
        ),
        "official_heldout": official_heldout,
        "latest_official_heldout": latest_official_heldout,
        "latest_formal_holdout": latest_formal_holdout,
        "matchup_runtime": matchup_runtime,
        "gate_passed": gate_passed,
        "promotion_wr": promotion.get("wr"),
        "promotion_passed": promotion.get("passed"),
        "remote_workers": extra.get("remote_workers", progress.get("remotes")),
        "remote_request_sockets": (
            (progress.get("metrics") or {}).get("remote_request_sockets")
        ),
        "remote_queue_capacity": (
            (progress.get("metrics") or {}).get("remote_queue_capacity")
        ),
        "remote_outstanding": (
            (progress.get("metrics") or {}).get("remote_outstanding")
        ),
        "remote_outstanding_elmo": (
            (progress.get("metrics") or {}).get("remote_outstanding_elmo")
        ),
        "remote_outstanding_bert": (
            (progress.get("metrics") or {}).get("remote_outstanding_bert")
        ),
        "remote_dispatch": (
            handoff.get("remote_dispatch")
            if isinstance(handoff.get("remote_dispatch"), dict)
            else {}
        ),
        "scheduler_queues": scheduler_queue_state(run_name),
        "model_contract": learner_model_state(
            model_manifest,
            loop,
            iteration=(
                int(lineage_iteration)
                if isinstance(lineage_iteration, int)
                else None
            ),
            runtime_optimizer=worker.get("optimizer_runtime"),
            runtime_parameter_contract=checkpoint_parameter_telemetry(
                ROOT / "outputs/logs" / f"{run_name}.log"
            ) if run_name else {},
            runtime_collection=matchup_runtime,
        ),
        "champion": champion.get("path"),
        "updated_at": (
            recovery_path.stat().st_mtime
            if recovery_is_latest and recovery_path is not None
            else (
                (run_dir / "metrics/latest.json").stat().st_mtime
                if run_dir and (run_dir / "metrics/latest.json").is_file()
                else status_updated_at
            )
        ),
        "remote_endpoints": (
            (((manifest.get("design_contract") or {}).get("remotes") or {}).get("endpoints"))
            or (((manifest.get("contract") or {}).get("remotes") or {}).get("endpoints"))
            or []
        ),
    }


def elmo_state() -> dict[str, Any]:
    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-elmo-ssh",
            "elmo",
            "/mnt/Main/Elmo/poke-bot-agent/dashboard/fleet_host_snapshot.py",
            "--role",
            "simulator",
            "--name",
            "Elmo",
        ],
        timeout=6,
    )
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {
            "reachable": False,
            "name": "Elmo",
            "role": "simulator",
            "error": "telemetry unavailable",
        }


def current_deck_guide_prestage_state() -> dict[str, Any]:
    """Read checksum-backed guide-preparation progress from Elmo."""

    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-guide-prestage-ssh",
            "elmo",
            "python3",
            "/home/admin/pokebot-expert-guide-src-v1/scripts/"
            "current_deck_guide_prestage_snapshot.py",
        ],
        timeout=6,
    )
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        payload = {}
    if (
        isinstance(payload, dict)
        and payload.get("schema")
        == "poke_bot.current_deck_guide_prestage_snapshot/v1"
    ):
        return payload
    return {
        "schema": "poke_bot.current_deck_guide_prestage_snapshot/v1",
        "available": False,
        "observed_at": time.time(),
        "active": None,
        "windows": [],
    }


def matchup_pipeline_state() -> dict[str, Any]:
    """Report Elmo staging plus the authoritative Blackwell adapter fit."""

    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-matchup-ssh",
            "elmo",
            "sudo",
            "-n",
            "python3",
            "/mnt/Main/main/poke-adapter-oracle-v29/src/scripts/"
            "matchup_pipeline_snapshot.py",
        ],
        timeout=6,
    )
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            rare_raw = run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=3",
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    "ControlPersist=60",
                    "-o",
                    "ControlPath=/tmp/pokebot-dashboard-matchup-ssh",
                    "elmo",
                    "sudo",
                    "-n",
                    "python3",
                    "/mnt/Main/main/poke-adapter-oracle-v29/src/scripts/"
                    "rare_route_pipeline_snapshot.py",
                ],
                timeout=6,
            )
            try:
                rare = json.loads(rare_raw)
                if not isinstance(rare, dict):
                    rare = {}
            except (TypeError, json.JSONDecodeError):
                rare = {}
            importer = unit_state(
                "pokebot-rare-route-assets-v37-import.service", user=True
            )
            receipt_path = Path(
                "/home/inzi/poke-bot-agent/outputs/state/"
                "rare-route-assets-v37-ready.json"
            )
            receipt = read_json(receipt_path)
            promotion_ready = (
                receipt.get("schema")
                == "poke_bot.rare_route_asset_promotion/v1"
                and receipt.get("status") == "ready"
            )
            if promotion_ready:
                importer = {
                    **importer,
                    "name": "pokebot-rare-route-router-v37-import.service",
                    "active": False,
                    "active_state": "inactive",
                    "sub_state": "dead",
                    "result": "success",
                    "status": "complete",
                }
            payload["rare_route_preparation"] = {
                **rare,
                "importer": importer,
                "promotion_ready": promotion_ready,
                "promotion_receipt": str(receipt_path),
                "ready_rare_archetype_ids": list(
                    receipt.get("ready_rare_archetype_ids") or ()
                ),
            }
            fit_root = (
                Path("/home/inzi/poke-bot-agent/outputs/matchup_adapters")
                / "alakazam-iter26-all22-v31"
            )
            progress = read_json(fit_root / "progress.json")
            fit_services = [
                unit_state("pokebot-matchup-adapter-v31b.service", user=True),
                unit_state(
                    "pokebot-matchup-adapter-v31-recovery.service", user=True
                ),
                unit_state("pokebot-matchup-adapter-v31.service", user=True),
            ]
            fit_service = next(
                (row for row in fit_services if row.get("active")),
                fit_services[0],
            )
            final_path = fit_root / "final.pt"
            epoch = int(progress.get("epoch") or 0)
            epochs = int(progress.get("epochs") or 25)
            consumed = int(progress.get("train_sequences_consumed") or 0)
            sequences = int(progress.get("train_sequences") or 0)
            percent = 100.0 if final_path.is_file() else (
                100.0 * (epoch + consumed / max(1, sequences)) / max(1, epochs)
            )
            log_path = Path(
                "/home/inzi/poke-bot-agent/outputs/logs/matchup-adapter-v31.log"
            )
            latest_line = ""
            latest_line_age_seconds: float | None = None
            try:
                latest_line = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-1]
                latest_line_age_seconds = max(
                    0.0, time.time() - log_path.stat().st_mtime
                )
            except (OSError, IndexError):
                pass
            live_match = re.search(
                r"epoch=(\d+)/(\d+)\s+step=(\d+).*?sequences=(\d+)/(\d+)",
                latest_line,
            )
            if live_match:
                live_epoch, live_epochs, live_step, live_consumed, live_sequences = (
                    int(value) for value in live_match.groups()
                )
                # In per-batch lines the displayed epoch is one-based while
                # progress.json stores completed epochs.
                epoch = max(epoch, live_epoch - 1)
                epochs = live_epochs
                consumed = live_consumed
                sequences = live_sequences
                percent = 100.0 * (
                    epoch + consumed / max(1, sequences)
                ) / max(1, epochs)
            else:
                live_step = int(progress.get("step") or 0)
            fit_phase = (
                "validation"
                if bool(fit_service.get("active"))
                and sequences > 0
                and consumed / sequences >= 0.995
                and latest_line_age_seconds is not None
                and latest_line_age_seconds >= 20.0
                else "training"
            )
            fit_started_us = as_number(
                fit_service.get("started_monotonic_us")
            )
            fit_elapsed_s = (
                max(0.0, time.monotonic() - fit_started_us / 1_000_000.0)
                if fit_started_us and fit_service.get("active")
                else None
            )
            fit_eta_s = (
                fit_elapsed_s * (100.0 - percent) / percent
                if fit_elapsed_s is not None
                and 0.0 < percent < 100.0
                else None
            )
            fit_total_sequences_consumed = epoch * sequences + consumed
            fit_sequence_rate = (
                fit_total_sequences_consumed / fit_elapsed_s
                if fit_elapsed_s is not None and fit_elapsed_s > 0.0
                else None
            )
            payload["adapter_fit"] = {
                **dict(payload.get("adapter_fit") or {}),
                **fit_service,
                # A remotely queried user unit can lack the local LoadState
                # marker even though systemd returned a live PID/state.  Do
                # not render that successfully observed process as missing.
                "exists": bool(
                    fit_service.get("exists") or fit_service.get("active")
                ),
                "host": "Inzi / Blackwell",
                "running": bool(fit_service.get("active")),
                "complete": final_path.is_file(),
                "epochs_target": epochs,
                "epoch": epoch,
                "step": max(int(progress.get("step") or 0), live_step),
                "train_sequences_consumed": consumed,
                "train_sequences": sequences,
                "total_sequences_consumed": fit_total_sequences_consumed,
                "sequence_rate": fit_sequence_rate,
                "percent": max(0.0, min(100.0, percent)),
                "elapsed_seconds": fit_elapsed_s,
                "eta_seconds": fit_eta_s,
                "phase": fit_phase,
                "latest_line": latest_line,
                "latest_line_age_seconds": latest_line_age_seconds,
                "result": str(final_path),
                "runtime_enabled": False,
                "base_checkpoint_frozen": True,
            }
            fleet_progress_path = (
                Path("/home/inzi/poke-bot-agent/outputs/matchup_adapters")
                / "alakazam-iter26-fleet-v31"
                / "fleet-progress.json"
            )
            fleet_progress = read_json(fleet_progress_path)
            fleet_observed_at = as_number(fleet_progress.get("observed_at"))
            fleet_fresh = bool(
                fleet_progress.get("schema")
                == "poke_bot.matchup_adapter_fleet_progress/v1"
                and fleet_observed_at is not None
                and time.time() - fleet_observed_at < 75.0
            )
            if fleet_fresh:
                source_epoch = int(fleet_progress.get("source_epoch") or 2)
                target_epochs = int(fleet_progress.get("target_epochs") or 25)
                workers: list[dict[str, Any]] = []
                worker_epochs: list[int] = []
                worker_etas: list[float] = []
                worker_updates: list[float] = []
                all_complete = True
                for raw_worker in fleet_progress.get("workers") or []:
                    if not isinstance(raw_worker, dict):
                        continue
                    worker = dict(raw_worker)
                    worker_progress = worker.get("progress")
                    progress_row = (
                        dict(worker_progress)
                        if isinstance(worker_progress, dict)
                        else {}
                    )
                    worker_epoch = int(progress_row.get("epoch") or source_epoch)
                    worker_target = int(
                        progress_row.get("target_epochs") or target_epochs
                    )
                    worker_complete = bool(progress_row.get("complete"))
                    elapsed = as_number(progress_row.get("elapsed_seconds"))
                    worker_updated_at = as_number(progress_row.get("updated_at"))
                    if worker_updated_at is not None:
                        worker_updates.append(worker_updated_at)
                    completed = max(0, worker_epoch - source_epoch)
                    remaining = max(0, worker_target - worker_epoch)
                    worker_eta = (
                        elapsed * remaining / completed
                        if elapsed is not None and elapsed > 0 and completed > 0
                        else None
                    )
                    if worker_eta is not None:
                        worker_etas.append(worker_eta)
                    worker_epochs.append(worker_epoch)
                    all_complete = all_complete and worker_complete
                    workers.append(
                        {
                            **worker,
                            "epoch": worker_epoch,
                            "target_epochs": worker_target,
                            "complete": worker_complete,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": worker_eta,
                            "steps": int(progress_row.get("steps") or 0),
                            "active_epoch": int(
                                progress_row.get("active_epoch") or worker_epoch
                            ),
                            "active_route": str(
                                progress_row.get("active_route") or ""
                            ),
                            "active_route_games": int(
                                progress_row.get("active_route_games") or 0
                            ),
                            "active_route_decisions": int(
                                progress_row.get("active_route_decisions") or 0
                            ),
                            "active_route_batches": int(
                                progress_row.get("active_route_batches") or 0
                            ),
                            "execution_backend": str(
                                progress_row.get("execution_backend") or ""
                            ),
                            "routes_label": ", ".join(
                                str(value)
                                for value in (
                                    progress_row.get("routes")
                                    or worker.get("routes")
                                    or []
                                )
                            ),
                        }
                    )
                if workers:
                    floor_epoch = min(worker_epochs)
                    fleet_percent = 100.0 * (
                        floor_epoch - source_epoch
                    ) / max(1, target_epochs - source_epoch)
                    fleet_line = " · ".join(
                        f"{row.get('host')}/{row.get('device')}: "
                        f"epoch {row['epoch']}/{row['target_epochs']}"
                        for row in workers
                    )
                    merged_final = (
                        Path(
                            "/home/inzi/poke-bot-agent/outputs/"
                            "matchup_adapters/alakazam-iter26-fleet-v31/final.pt"
                        ).is_file()
                    )
                    fleet_eta_seconds = (
                        max(worker_etas)
                        if len(worker_etas) == len(workers)
                        else None
                    )
                    fleet_receipt_age_seconds = (
                        max(0.0, time.time() - max(worker_updates))
                        if worker_updates
                        else None
                    )
                    fleet_fields = {
                        "auxiliary_fleet": bool(fit_service.get("active")),
                        "fleet_workers": workers,
                        "fleet_observed_at": fleet_observed_at,
                        "fleet_percent": max(0.0, min(100.0, fleet_percent)),
                        "fleet_eta_seconds": fleet_eta_seconds,
                        "fleet_complete": merged_final,
                        "fleet_all_workers_complete": all_complete,
                        "fleet_latest_line": fleet_line,
                        "fleet_receipt_age_seconds": fleet_receipt_age_seconds,
                    }
                    if fit_service.get("active"):
                        # The persistent single-fit recovery service owns the
                        # canonical output directory. Auxiliary disjoint-route
                        # workers may be useful, but must never replace its
                        # progress or imply that their merger is authoritative.
                        payload["adapter_fit"] = {
                            **payload["adapter_fit"],
                            **fleet_fields,
                            "fleet": False,
                        }
                    else:
                        payload["adapter_fit"] = {
                            **payload["adapter_fit"],
                            **fleet_fields,
                            "name": "pokebot-adapter-fleet",
                            "pid": 0,
                            "active": not merged_final,
                            "running": not merged_final,
                            "complete": merged_final,
                            "host": "Inzi + Elmo + Bert",
                            "phase": (
                                "merge_pending"
                                if all_complete
                                else "fleet_training"
                            ),
                            "epoch": floor_epoch,
                            "epochs_target": target_epochs,
                            "percent": max(
                                0.0, min(100.0, fleet_percent)
                            ),
                            "eta_seconds": fleet_eta_seconds,
                            "latest_line": fleet_line,
                            # The fleet line is built from worker receipts, not
                            # the retired single-fitter log.  Carrying the old
                            # log mtime made a healthy fleet look stale.
                            "latest_line_age_seconds": fleet_receipt_age_seconds,
                            "fleet": True,
                            "base_checkpoint_frozen": True,
                            "runtime_enabled": False,
                            "result": str(
                                Path(
                                    "/home/inzi/poke-bot-agent/outputs/"
                                    "matchup_adapters/"
                                    "alakazam-iter26-fleet-v31/final.pt"
                                )
                            ),
                        }
            finalizer_services = [
                unit_state(
                    "pokebot-matchup-runtime-v31-finalizer4.service", user=True
                ),
                unit_state(
                    "pokebot-matchup-runtime-v31-finalizer3.service", user=True
                ),
            ]
            finalizer_service = next(
                (row for row in finalizer_services if row.get("active")),
                finalizer_services[0],
            )
            finalizer_status = read_json(
                Path(
                    "/home/inzi/poke-bot-agent/outputs/state/"
                    "matchup-runtime-v31-finalizer.json"
                )
            )
            payload["finalizer"] = {
                **finalizer_service,
                **finalizer_status,
                "exists": bool(
                    finalizer_service.get("exists")
                    or finalizer_service.get("active")
                ),
                "running": bool(finalizer_service.get("active")),
                "complete": finalizer_status.get("phase") == "complete",
            }
            payload["production_blocking"] = (
                finalizer_status.get("phase") != "complete"
            )
            return payload
    except (TypeError, json.JSONDecodeError):
        pass
    return {
        "schema": "poke_bot.matchup_pipeline_dashboard/v1",
        "host": "Elmo",
        "available": False,
        "error": "matchup pipeline telemetry unavailable",
    }


def reconcile_canonical_router_candidate(
    matchup_pipeline: dict[str, Any],
    specialist_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Make the audited canonical candidate control dashboard router status.

    Elmo retains historical build telemetry for operational diagnosis.  That
    telemetry must not outrank the canonical, checksum-pinned candidate in the
    specialist tracker—especially when an older build used a superseded
    calibration contract.
    """

    pipeline = dict(matchup_pipeline)
    head_requirements = specialist_protocol.get("head_requirements")
    head_requirements = (
        head_requirements if isinstance(head_requirements, dict) else {}
    )
    candidate = head_requirements.get("staged_router_candidate")
    if not isinstance(candidate, dict) or not candidate:
        return pipeline

    rare = pipeline.get("rare_route_preparation")
    rare = rare if isinstance(rare, dict) else {}
    receipt_path = Path(str(rare.get("promotion_receipt") or "/nonexistent"))
    receipt = read_json(receipt_path)
    promoted_audit_path = Path(
        str(receipt.get("candidate_audit") or "/nonexistent")
    )
    promoted_tree_path = Path(
        str(receipt.get("candidate_tree") or "/nonexistent")
    )
    promoted_audit = read_json(promoted_audit_path)
    promoted_ids = [
        str(value)
        for value in receipt.get("accepted_specialist_ids") or ()
    ]
    if (
        rare.get("promotion_ready") is True
        and receipt.get("schema")
        == "poke_bot.rare_route_asset_promotion/v1"
        and receipt.get("status") == "ready"
        and int(receipt.get("accepted_count") or 0) == 22
        and len(promoted_ids) == 22
        and len(set(promoted_ids)) == 22
        and _file_sha256_matches(
            promoted_tree_path, receipt.get("candidate_tree_sha256")
        )
        and _file_sha256_matches(
            promoted_audit_path, receipt.get("candidate_audit_sha256")
        )
        and promoted_audit.get("schema")
        == "poke_bot.public_matchup_tree_candidate_audit/v1"
        and promoted_audit.get("runtime_enabled") is False
        and int(promoted_audit.get("accepted_count") or 0) == 22
    ):
        candidate = {
            "version": 35,
            "status": "validated_inactive",
            "runtime_enabled": False,
            "accepted_route_count": 22,
            "accepted_routes": promoted_ids,
            "evidence_blocked_routes": [],
            "source_days": len(
                (
                    read_json(promoted_tree_path).get("sources")
                    if promoted_tree_path.is_file()
                    else []
                )
                or []
            ),
            "artifact": str(promoted_tree_path),
            "artifact_checksum": receipt.get("candidate_tree_sha256"),
            "audit": str(promoted_audit_path),
            "calibration_contract": (
                "expanded_to_canonical_class_indexes_all22"
            ),
            "minimum_validation_precision": promoted_audit.get(
                "minimum_precision"
            ),
            "minimum_validation_weighted_support": promoted_audit.get(
                "minimum_weighted_support"
            ),
            "activation_policy": (
                "Bind at the next safe specialist boundary; never mutate the "
                "current active specialist."
            ),
        }

    prior = pipeline.get("router_refresh")
    prior = dict(prior) if isinstance(prior, dict) else {}
    accepted_routes = [
        str(value) for value in candidate.get("accepted_routes") or []
    ]
    blocked_routes = [
        str(value) for value in candidate.get("evidence_blocked_routes") or []
    ]
    observed_calibrated_routes = [
        str(value) for value in prior.get("calibrated_route_ids") or []
    ]
    observed_calibrated_count = int(
        prior.get("calibrated_route_count") or len(observed_calibrated_routes)
    )
    target_routes = len(accepted_routes) + len(blocked_routes)
    if target_routes <= 0:
        target_routes = int(candidate.get("target_routes") or 22)
    activation_policy = str(candidate.get("activation_policy") or "")

    pipeline["router_refresh"] = {
        "name": f"canonical-public-matchup-tree-v{candidate.get('version')}",
        "phase": str(candidate.get("status") or "unknown"),
        "status": str(candidate.get("status") or "unknown"),
        "candidate_ready": candidate.get("status") == "validated_inactive",
        "candidate_runtime_enabled": candidate.get("runtime_enabled") is True,
        "production_active": candidate.get("runtime_enabled") is True,
        "target_routes": target_routes,
        # Elmo's completed build reports routes with a calibrated threshold.
        # The canonical audit applies the stricter precision and weighted-
        # support contract.  These are intentionally separate dashboard
        # metrics: calibration alone does not make a route runtime-ready.
        "calibrated_route_count": observed_calibrated_count,
        "calibrated_route_ids": observed_calibrated_routes,
        "protocol_ready_route_count": int(
            candidate.get("accepted_route_count") or len(accepted_routes)
        ),
        "protocol_ready_route_ids": accepted_routes,
        "protocol_blocked_route_count": len(blocked_routes),
        "evidence_blocked_routes": blocked_routes,
        "source_days": candidate.get("source_days"),
        "source_window_start": candidate.get("source_window_start"),
        "source_window_end": candidate.get("source_window_end"),
        "artifact": candidate.get("artifact"),
        "artifact_checksum": candidate.get("artifact_checksum"),
        "audit": candidate.get("audit"),
        "calibration_contract": candidate.get("calibration_contract"),
        "minimum_validation_precision": candidate.get(
            "minimum_validation_precision"
        ),
        "minimum_validation_weighted_support": candidate.get(
            "minimum_validation_weighted_support"
        ),
        "activation_policy": activation_policy or None,
        "activation_requires_safe_boundary": (
            "safe" in activation_policy.lower()
            and "boundary" in activation_policy.lower()
        ),
        "canonical": True,
        "superseded_observation": prior or None,
        "superseded_reason": (
            "Elmo v32 used compressed predict_proba class indexes; canonical "
            "v33 expands probabilities to canonical class indexes and passed "
            "the checksum-pinned audit."
        ),
    }
    return pipeline


def _elmo_latest20_daily_materialization(
    status_glob: str = EXPERT20_ELMO_DAILY_STATUS_GLOB,
) -> dict[str, dict[str, Any]]:
    """Read the current per-day Elmo build statuses in one bounded SSH call."""

    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-expert-daily-ssh",
            "elmo",
            "cat",
            status_glob,
        ],
        timeout=6,
    )
    decoder = json.JSONDecoder()
    offset = 0
    statuses: dict[str, dict[str, Any]] = {}
    while offset < len(raw):
        while offset < len(raw) and raw[offset].isspace():
            offset += 1
        if offset >= len(raw):
            break
        try:
            payload, offset = decoder.raw_decode(raw, offset)
        except json.JSONDecodeError:
            break
        if not isinstance(payload, dict):
            continue
        window = (
            payload.get("date_window")
            if isinstance(payload.get("date_window"), dict)
            else {}
        )
        day_value = str(
            payload.get("current_date")
            or window.get("start")
            or (
                (payload.get("completed") or [{}])[0].get("date")
                if isinstance(payload.get("completed"), list)
                and payload.get("completed")
                and isinstance(payload["completed"][0], dict)
                else ""
            )
        )
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day_value):
            statuses[day_value] = payload
    return statuses


def v6_strategic_corpus_state(
    dates: list[str],
) -> dict[str, Any]:
    """Report the separate V6 strategic-label corpus without masking V5 truth."""

    if len(dates) != 20:
        return {
            "available": False,
            "active": False,
            "complete": False,
            "reason": "exact latest-20 archive window is unavailable",
        }
    staged_sync = read_json(V6_STRATEGIC_STAGED_SYNC_STATE)
    staged_dates = _exact_calendar_dates(
        [str(value) for value in staged_sync.get("dates") or ()]
    )
    staged_current = bool(
        staged_sync.get("schema") == "poke_bot.latest20_specialist_sync/v1"
        and staged_sync.get("status") in {"syncing", "incomplete", "ready"}
        and staged_dates == dates
        and int(staged_sync.get("specialist_count") or 0) == 18
        and staged_sync.get("expanded_target_schema")
        == V6_STRATEGIC_TARGET_SCHEMA
        and staged_sync.get("expanded_target_digest")
        == V6_STRATEGIC_TARGET_DIGEST
    )
    staged_service_active = (
        run(
            [
                "systemctl",
                "--user",
                "is-active",
                V6_STRATEGIC_STAGED_SYNC_SERVICE,
            ],
            timeout=4,
        ).strip()
        in {"active", "activating"}
    )
    runtime_activation = read_json(MARNIE_LATEST20_RUNTIME_ACTIVATION_STATE)
    activated_for_marnie = bool(
        runtime_activation.get("schema")
        == "poke_bot.marnie_latest20_runtime_activation/v1"
        and runtime_activation.get("status") == "activated"
        and runtime_activation.get("window_start") == dates[0]
        and runtime_activation.get("window_end") == dates[-1]
        and runtime_activation.get("active_training_corpus") is True
    )
    if staged_current or (
        staged_service_active and dates == V6_STRATEGIC_STAGED_DATES
    ):
        using_staged_pipeline = True
        sync = staged_sync
        sync_state_path = V6_STRATEGIC_STAGED_SYNC_STATE
        current_pointer = V6_STRATEGIC_STAGED_CURRENT
        sync_service = V6_STRATEGIC_STAGED_SYNC_SERVICE
        activation_state = (
            "active_marnie_runtime"
            if activated_for_marnie
            else "staged_not_active"
        )
    else:
        using_staged_pipeline = False
        sync = read_json(V6_STRATEGIC_SPECIALIST_SYNC_STATE)
        sync_state_path = V6_STRATEGIC_SPECIALIST_SYNC_STATE
        current_pointer = V6_STRATEGIC_SPECIALIST_CURRENT
        sync_service = V6_STRATEGIC_SPECIALIST_SYNC_SERVICE
        activation_state = "historical_or_active_pointer"
    sync_dates = _exact_calendar_dates(
        [str(value) for value in sync.get("dates") or ()]
    )
    sync_current = bool(
        sync.get("schema") == "poke_bot.latest20_specialist_sync/v1"
        and sync.get("status") in {"syncing", "incomplete", "ready"}
        and sync_dates == dates
        and int(sync.get("specialist_count") or 0) == 18
        and sync.get("expanded_target_schema")
        == V6_STRATEGIC_TARGET_SCHEMA
        and sync.get("expanded_target_digest")
        == V6_STRATEGIC_TARGET_DIGEST
    )
    service_active = (
        staged_service_active
        if sync_service == V6_STRATEGIC_STAGED_SYNC_SERVICE
        else (
            run(
                [
                    "systemctl",
                    "--user",
                    "is-active",
                    sync_service,
                ],
                timeout=4,
            ).strip()
            in {"active", "activating"}
        )
    )
    pointer_ready = bool(
        sync_current
        and sync.get("status") == "ready"
        and current_pointer.is_symlink()
        and current_pointer.resolve().is_dir()
    )
    daily_status = (
        {}
        if pointer_ready
        else _elmo_latest20_daily_materialization(
            (
                EXPERT20_V6_STRATEGIC_ELMO_DAILY_STATUS_GLOB
                if dates == [f"2026-07-{day:02d}" for day in range(4, 24)]
                else (
                    "/mnt/Main/main/poke-bot-agent/archive/"
                    "expert-latest20-derived/daily/"
                    f"roster18-v6-strategic-{dates[0]}_{dates[-1]}/"
                    "status/*.json"
                )
            )
        )
    )
    running_days = {
        day
        for day, status in daily_status.items()
        if day in dates and status.get("state") == "running"
    }
    ready_days = {
        day
        for day, status in daily_status.items()
        if day in dates
        and status.get("state") == "complete"
        and len(status.get("completed") or ()) == 1
    }
    failed_days = {
        day
        for day, status in daily_status.items()
        if day in dates and status.get("state") == "failed"
    }
    # A current sync receipt can only be written after the finalizer's exact
    # 20-date aggregate receipt has passed validation.  From that point the
    # aggregate receipt is the coverage authority; per-day worker receipts are
    # merely lower-grain build history and may exist only for the rebuilt half
    # of a window that reused ten already-validated shards.
    if sync_current:
        ready_days = set(dates)
        running_days = set()
        failed_days = set()
    source_bytes = int(sync.get("source_bytes") or 0)
    copied_bytes = min(source_bytes, int(sync.get("copied_bytes") or 0))
    sync_percent = (
        100.0 * copied_bytes / source_bytes
        if sync_current and source_bytes > 0
        else None
    )
    daily_complete = len(ready_days) == 20 and not failed_days
    phase = (
        "ready"
        if pointer_ready
        else "atomic_checksum_sync_to_inzi"
        if sync_current
        else "specialist_corpus_finalization"
        if daily_complete
        else "parallel_daily_materialization"
        if running_days
        else "daily_feature_failure"
        if failed_days
        else "waiting_for_daily_materialization"
    )
    percent = (
        100.0
        if pointer_ready
        else 75.0 + 0.25 * float(sync_percent or 0.0)
        if sync_current
        else 50.0 + 25.0 * len(ready_days) / 20.0
    )
    return {
        "available": bool(
            service_active or sync_current or daily_status or pointer_ready
        ),
        "active": bool(service_active and not pointer_ready),
        "complete": pointer_ready,
        "phase": phase,
        "stage": phase,
        "target_schema": V6_STRATEGIC_TARGET_SCHEMA,
        "target_digest": V6_STRATEGIC_TARGET_DIGEST,
        "window_start": dates[0],
        "window_end": dates[-1],
        "completed_days": len(ready_days),
        "running_days": len(running_days),
        "failed_days": len(failed_days),
        "total_days": 20,
        "percent": percent,
        "sync_status": sync.get("status") if sync_current else None,
        "sync_percent": sync_percent,
        "atomic_pointer_ready": pointer_ready,
        "current_pointer": str(current_pointer),
        "sync_receipt": str(sync_state_path),
        "activation_state": activation_state,
        "active_training_corpus": (
            activated_for_marnie if using_staged_pipeline else None
        ),
        "latest_line": (
            "NEXT BOUNDARY EXPANDED STRATEGIC CORPUS · "
            "Accepted Policy Generation 15 · "
            f"{len(ready_days)}/20 daily feature shards ready · "
            f"{len(running_days)} running · {phase.replace('_', ' ')}"
            + (
                f" · sync {sync_percent:.1f}%"
                if sync_percent is not None
                else ""
            )
        ),
    }


def expert_refresh_state() -> dict[str, Any]:
    """Report the authoritative split-host 20-day expert refresh."""
    current = read_json(EXPERT20_CURRENT_RECEIPT)
    if (
        current.get("schema") == "poke_bot.expert_latest20_receipt/v1"
        and current.get("status") == "ready"
        and int(current.get("days") or 0) == 20
        and len(current.get("archives") or ()) == 20
    ):
        rows = [
            {
                "day": str(row.get("date") or ""),
                "date": str(row.get("date") or ""),
                "host": "Elmo",
                "stage": "ready",
                "percent": 100.0,
                "archive_bytes": int(row.get("bytes") or 0),
                "episodes": int(row.get("episode_count") or 0),
                "sha256": row.get("sha256"),
                "service": {"active": False},
            }
            for row in current.get("archives") or ()
            if row.get("validated") is True
        ]
        dates = _exact_calendar_dates(
            [str(row.get("day") or "") for row in rows]
        )
        if len(rows) == 20 and len(dates) == 20:
            specialist_sync = read_json(LATEST20_SPECIALIST_SYNC_STATE)
            specialist_sync_dates = _exact_calendar_dates(
                [
                    str(value)
                    for value in specialist_sync.get("dates") or ()
                ]
            )
            specialist_sync_current = bool(
                specialist_sync.get("schema")
                == "poke_bot.latest20_specialist_sync/v1"
                and specialist_sync.get("status")
                in {"syncing", "incomplete", "ready"}
                and specialist_sync_dates == dates
                and int(specialist_sync.get("specialist_count") or 0) == 18
                and int(specialist_sync.get("source_bytes") or 0) > 0
            )
            daily_status = (
                {}
                if specialist_sync_current
                and specialist_sync.get("status") == "ready"
                else _elmo_latest20_daily_materialization()
            )
            running_days = {
                day_value
                for day_value, status in daily_status.items()
                if status.get("state") == "running"
            }
            ready_days = {
                day_value
                for day_value, status in daily_status.items()
                if status.get("state") == "complete"
                and len(status.get("completed") or ()) == 1
            }
            failed_days = {
                day_value
                for day_value, status in daily_status.items()
                if status.get("state") == "failed"
            }
            # The sync state is written only after the finalizer's canonical
            # 20-date/18-specialist receipt has validated. It supersedes
            # disposable per-day worker statuses once finalization completes.
            if specialist_sync_current:
                ready_days = set(dates)
                running_days = set()
                failed_days = set()
            strategic_v6 = v6_strategic_corpus_state(dates)
            if daily_status or specialist_sync_current:
                rows_by_day = {
                    str(row.get("day") or ""): dict(row) for row in rows
                }
                rows = []
                for day_value in dates:
                    row = rows_by_day[day_value]
                    if day_value in ready_days:
                        row.update(
                            {
                                "host": "Elmo",
                                "stage": "feature_ready",
                                "percent": 100.0,
                            }
                        )
                    elif day_value in running_days:
                        row.update(
                            {
                                "host": "Elmo",
                                "stage": "featurizing",
                                "percent": 0.0,
                                "service": {"active": True},
                            }
                        )
                    else:
                        row.update(
                            {
                                "stage": "archive_ready",
                                "percent": 50.0,
                            }
                        )
                    rows.append(row)
            daily_active = bool(running_days)
            selected_daily_days = (
                20 if specialist_sync_current else len(daily_status)
            )
            daily_materialization_ready = bool(
                selected_daily_days
                and len(ready_days) == selected_daily_days
                and not running_days
                and not failed_days
            )
            specialist_sync_active = (
                run(
                    [
                        "systemctl",
                        "--user",
                        "is-active",
                        LATEST20_SPECIALIST_SYNC_SERVICE,
                    ],
                    timeout=4,
                ).strip()
                in {"active", "activating"}
            )
            specialist_pointer_ready = (
                LATEST20_SPECIALIST_CURRENT.is_symlink()
                and LATEST20_SPECIALIST_CURRENT.resolve().is_dir()
            )
            source_bytes = int(specialist_sync.get("source_bytes") or 0)
            copied_bytes = min(
                source_bytes,
                int(specialist_sync.get("copied_bytes") or 0),
            )
            sync_percent = (
                100.0 * copied_bytes / source_bytes
                if specialist_sync_current and source_bytes > 0
                else None
            )
            finalization_label = (
                "latest20 specialist corpus READY on Inzi"
                if specialist_pointer_ready
                else (
                    "Inzi checksum sync active · atomic pointer withheld until "
                    "validation"
                    if specialist_sync_active
                    else "managed corpus finalizer and Inzi sync pending"
                )
            )
            return {
                "available": True,
                # The source pipeline remains active after its selected missing
                # days finish because the managed corpus finalizer and atomic
                # Inzi promotion still have to publish their receipts.
                "active": bool(daily_active or specialist_sync_active),
                "complete": specialist_pointer_ready,
                "archive_window_ready": True,
                "host": "Bert Wi-Fi → Elmo Ethernet",
                "stage": (
                    "ready"
                    if specialist_pointer_ready
                    else "syncing_specialist_corpora"
                    if specialist_sync_active and specialist_sync_current
                    else "specialist_corpus_sync_pending"
                    if specialist_sync_current
                    else "featurizing"
                    if daily_active
                    else "daily_features_ready"
                    if daily_materialization_ready
                    else "daily_feature_retry_pending"
                    if failed_days
                    else "archive_ready"
                ),
                "phase": (
                    "corpus_ready"
                    if specialist_pointer_ready
                    else "atomic_checksum_sync_to_inzi"
                    if specialist_sync_active and specialist_sync_current
                    else "atomic_checksum_sync_pending"
                    if specialist_sync_current
                    else "parallel_daily_materialization"
                    if daily_active
                    else "managed_corpus_finalization_pending"
                    if daily_materialization_ready
                    else "managed_daily_retry_pending"
                    if failed_days
                    else "archive_window_ready"
                ),
                "window_start": dates[0],
                "window_end": dates[-1],
                "completed_days": len(ready_days),
                "archive_ready_days": 20,
                "feature_ready_days": len(ready_days),
                "total_days": 20,
                "percent": (
                    100.0
                    if specialist_pointer_ready
                    else 75.0 + 0.25 * float(sync_percent or 0.0)
                    if specialist_sync_current
                    else 50.0
                    + (
                        25.0 * len(ready_days) / selected_daily_days
                        if selected_daily_days
                        else 0.0
                    )
                ),
                "daily_materialization": {
                    "selected_days": selected_daily_days,
                    "completed_days": len(ready_days),
                    "running_days": len(running_days),
                    "failed_days": len(failed_days),
                    "ready": daily_materialization_ready,
                    "finalization_pending": bool(
                        daily_materialization_ready
                        and not specialist_sync_current
                    ),
                    "finalization_ready": specialist_sync_current,
                },
                "specialist_sync": {
                    "status": specialist_sync.get("status"),
                    "current_bytes": copied_bytes,
                    "total_bytes": source_bytes,
                    "percent": sync_percent,
                    "bandwidth_limit_kib_per_second": specialist_sync.get(
                        "bandwidth_limit_kib_per_second"
                    ),
                    "atomic_pointer_ready": specialist_pointer_ready,
                    "source": str(LATEST20_SPECIALIST_SYNC_STATE),
                },
                "expanded_v6": strategic_v6,
                "days": rows,
                "latest_line": (
                    f"Latest 20 daily archives ready · {dates[0]} through "
                    f"{dates[-1]} · {len(ready_days)}/"
                    f"{selected_daily_days or 20} selected missing daily "
                    f"features complete · {len(running_days)} active"
                    + (
                        f" · {len(failed_days)} awaiting managed retry"
                        if failed_days
                        else ""
                    )
                    + (
                        f" · {finalization_label}"
                        if daily_materialization_ready
                        else ""
                    )
                    + (
                        f" · {copied_bytes:,}/{source_bytes:,} bytes "
                        f"({float(sync_percent or 0.0):.1f}%)"
                        if specialist_sync_current and source_bytes > 0
                        else ""
                    )
                ),
                "source": str(EXPERT20_CURRENT_RECEIPT),
                "updated_at": current.get("committed_at"),
                "ingress": current.get("ingress"),
                "metric_definition": (
                    "archive readiness from the atomic latest-20 receipt; "
                    "specialist filtering is reported separately"
                ),
            }
    refresh = read_json(EXPERT20_REFRESH_STATUS)
    inzi = read_json(EXPERT20_INZI_STATUS)
    inzi_final = read_json(EXPERT20_INZI_FINAL_STATUS)
    inzi_tail = (
        read_json(EXPERT20_INZI_TAIL_STATUS)
        if EXPERT20_INZI_TAIL_STATUS.parent == EXPERT20_ROOT
        else {}
    )
    remote_raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-expert-refresh-ssh",
            "elmo",
            "cat",
            "/mnt/Main/main/poke-feature-refresh-20260721/data/bootstrap/"
            "expert-latest20-additive/elmo.status.json",
        ],
        timeout=6,
    )
    try:
        remote = json.loads(remote_raw)
        if not isinstance(remote, dict):
            remote = {}
    except (TypeError, json.JSONDecodeError):
        remote = {}
    remote_current_for_partial = str(remote.get("current_date") or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", remote_current_for_partial):
        partial_raw = run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=3",
                "elmo",
                "find",
                "/mnt/Main/main/poke-feature-refresh-20260721/data/bootstrap/"
                "expert-latest20-additive/features-elmo",
                "-maxdepth",
                "1",
                "-name",
                f".all-recognized-{remote_current_for_partial}.features.partial.*",
                "-printf",
                "%s\n",
            ],
            timeout=5,
        )
        remote["_partial_bytes"] = max(
            (int(line) for line in partial_raw.splitlines() if line.isdigit()),
            default=0,
        )

    window = refresh.get("window") if isinstance(refresh.get("window"), dict) else {}
    start = str(window.get("start") or "2026-07-02")
    end = str(window.get("end") or "2026-07-21")
    try:
        first = date.fromisoformat(start)
        last = date.fromisoformat(end)
        day_names = [
            (first + timedelta(days=offset)).isoformat()
            for offset in range((last - first).days + 1)
        ]
    except ValueError:
        day_names = [f"2026-07-{day:02d}" for day in range(2, 22)]
        start, end = day_names[0], day_names[-1]

    archive_rows = {
        str(row.get("date")): row
        for row in refresh.get("ready") or []
        if isinstance(row, dict) and row.get("date")
    }
    completed_by_day: dict[str, tuple[str, dict[str, Any]]] = {}
    local_feature_days: set[str] = set()
    for host, status in (
        ("Inzi", inzi),
        ("Inzi", inzi_final),
        ("Inzi", inzi_tail),
        ("Elmo", remote),
    ):
        for row in status.get("completed") or []:
            if isinstance(row, dict) and row.get("date"):
                completed_by_day[str(row["date"])] = (host, row)
    for path in EXPERT20_FEATURE_DIR.glob("all-recognized-*.features"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})\.features$", path.name)
        if match:
            local_feature_days.add(match.group(1))
            if match.group(1) not in completed_by_day:
                completed_by_day[match.group(1)] = (
                    "Inzi",
                    {"date": match.group(1), "output": str(path)},
                )

    current_by_day: dict[str, str] = {}
    for host, status in (("Inzi", inzi), ("Inzi", inzi_tail), ("Elmo", remote)):
        current = status.get("current_date")
        if status.get("state") == "running" and current:
            current_by_day[str(current)] = host

    completed_sizes = [
        path.stat().st_size
        for path in EXPERT20_FEATURE_DIR.glob("all-recognized-*.features")
        if path.is_file()
    ]
    typical_feature_bytes = (
        sorted(completed_sizes)[len(completed_sizes) // 2]
        if completed_sizes
        else None
    )
    local_partial_bytes: dict[str, int] = {}
    for status in (inzi, inzi_tail):
        current = str(status.get("current_date") or "")
        if not current:
            continue
        matches = list(
            EXPERT20_FEATURE_DIR.glob(
                f".all-recognized-{current}.features.partial.*"
            )
        )
        local_partial_bytes[current] = max(
            (path.stat().st_size for path in matches if path.is_file()), default=0
        )
    remote_current = str(remote.get("current_date") or "")
    remote_partial_bytes = int(remote.get("_partial_bytes") or 0)

    days: list[dict[str, Any]] = []
    for day in day_names:
        archive = archive_rows.get(day) or {}
        completed = completed_by_day.get(day)
        current_host = current_by_day.get(day)
        if completed is not None:
            host, feature = completed
            stage, percent = "feature_ready", 100.0
        elif current_host:
            host, feature = current_host, {}
            partial_bytes = (
                remote_partial_bytes
                if current_host == "Elmo" and day == remote_current
                else local_partial_bytes.get(day, 0)
            )
            percent = (
                min(99.0, 100.0 * partial_bytes / typical_feature_bytes)
                if typical_feature_bytes and partial_bytes
                else 0.0
            )
            feature = {"partial_bytes": partial_bytes, "progress_estimated": True}
            stage = "featurizing"
        elif archive:
            host, feature = "Inzi", {}
            stage, percent = "archive_ready", 50.0
        else:
            host, feature = "—", {}
            stage, percent = "waiting", 0.0
        days.append(
            {
                "day": day,
                "host": host,
                "stage": stage,
                "percent": percent,
                "archive_bytes": archive.get("bytes"),
                "feature_records": feature.get("records"),
                "feature_decisions": feature.get("decisions"),
                "partial_bytes": feature.get("partial_bytes"),
                "progress_estimated": feature.get("progress_estimated", False),
                "service": {"active": bool(current_host)},
            }
        )

    archive_ready = sum(day in archive_rows for day in day_names)
    feature_ready = sum(day in completed_by_day for day in day_names)
    local_feature_ready = sum(day in local_feature_days for day in day_names)
    total_days = len(day_names)
    inzi_units = (
        unit_state("pokebot-expert-v29-inzi.service", user=True),
        unit_state("pokebot-expert-v29-inzi-tail.service", user=True),
    )
    active = bool(current_by_day) or any(bool(unit.get("active")) for unit in inzi_units)
    assembled = EXPERT20_ASSEMBLED_MANIFEST.is_file()
    corpus_ready = EXPERT20_ALAKAZAM_CORPUS.is_file()
    sync_unit = unit_state("pokebot-expert-v29-elmo-sync.service", user=True)
    complete = bool(
        local_feature_ready == total_days and assembled and corpus_ready
    )
    if complete:
        phase, stage, host = "corpus_ready", "ready", "Inzi"
    elif feature_ready == total_days and local_feature_ready < total_days:
        phase, stage, host = (
            "returning_feature_shards",
            "returning completed shards",
            "Elmo → Inzi",
        )
        active = active or bool(sync_unit.get("active"))
    elif local_feature_ready == total_days:
        phase, stage, host = "assemble_filter", "assembling manifest", "Inzi"
    elif active:
        phase, stage = "parallel_featurization", "featurizing"
        host = (
            "Inzi + Elmo"
            if any(value == "Elmo" for value in current_by_day.values())
            else "Inzi"
        )
    else:
        phase, stage, host = (
            "featurization_recovery",
            "waiting / recovery",
            "Inzi + Elmo",
        )
    current_day = next(iter(current_by_day), None)
    active_partial_bytes = sum(
        int(row.get("partial_bytes") or 0)
        for row in days
        if row.get("stage") == "featurizing"
    )
    active_total_bytes = (
        typical_feature_bytes
        * sum(row.get("stage") == "featurizing" for row in days)
        if typical_feature_bytes
        else None
    )
    percent = 100.0 * (
        0.25 * archive_ready
        + 0.50 * feature_ready
        + 0.25 * local_feature_ready
    ) / max(1, total_days)
    reason = None
    if remote.get("state") == "failed":
        reason = f"Elmo feature worker failed: {remote.get('error') or 'unknown error'}"
    latest_line = (
        f"20-day expert window: {archive_ready}/{total_days} archives verified; "
        f"{feature_ready}/{total_days} feature shards ready; "
        f"{local_feature_ready}/{total_days} landed on Inzi; "
        f"manifest={'ready' if assembled else 'pending'}; "
        f"Alakazam corpus={'ready' if corpus_ready else 'pending'}."
    )
    source_paths = (
        EXPERT20_REFRESH_STATUS,
        EXPERT20_INZI_STATUS,
        EXPERT20_INZI_FINAL_STATUS,
        EXPERT20_INZI_TAIL_STATUS,
        EXPERT20_ASSEMBLED_MANIFEST,
        EXPERT20_ALAKAZAM_CORPUS,
    )
    updated = [path.stat().st_mtime for path in source_paths if path.exists()]
    return {
        "available": bool(refresh or archive_rows),
        "active": active,
        "complete": complete,
        "archive_window_ready": archive_ready == total_days,
        "host": host,
        "stage": stage,
        "phase": phase,
        "window_start": start,
        "window_end": end,
        "current_day": current_day,
        "current": active_partial_bytes or None,
        "total": active_total_bytes,
        "progress_estimated": True,
        "completed_days": feature_ready,
        "archive_ready_days": archive_ready,
        "feature_ready_days": feature_ready,
        "local_feature_ready_days": local_feature_ready,
        "total_days": total_days,
        "percent": percent,
        "day_percent": next(
            (float(row["percent"]) for row in days if row["day"] == current_day),
            None,
        ),
        "days": days,
        "latest_line": latest_line,
        "reason": reason,
        "assembled_manifest_ready": assembled,
        "filtered_corpus_ready": corpus_ready,
        "source": str(EXPERT20_REFRESH_STATUS),
        "updated_at": max(updated) if updated else None,
        "metric_definition": (
            "overall percent allocates 25% to archive validation, 50% to "
            "authoritative feature completion, and 25% to landing shards on Inzi"
        ),
    }


def latest10_state() -> dict[str, Any]:
    local_raw = run(
        [
            str(LATEST10_STATUS),
            "--root",
            str(ROOT),
            "--host",
            "Inzi",
        ],
        timeout=5,
    )
    remote_raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "elmo",
            "/mnt/Main/main/poke-feature-latest10/scripts/latest10_status.py",
            "--root",
            "/mnt/Main/main/poke-feature-latest10",
            "--host",
            "Elmo",
        ],
        timeout=6,
    )

    def decoded(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    candidates: dict[str, list[dict[str, Any]]] = {}
    for payload in (decoded(local_raw), decoded(remote_raw)):
        for row in payload.get("days") or []:
            if isinstance(row, dict) and row.get("day"):
                candidates.setdefault(str(row["day"]), []).append(row)
    days: list[dict[str, Any]] = []
    for day in [f"2026-07-{value:02d}" for value in range(9, 19)]:
        options = candidates.get(day) or [
            {"day": day, "host": "—", "stage": "waiting", "percent": 0.0}
        ]
        days.append(
            max(
                options,
                key=lambda row: (
                    float(row.get("percent") or 0.0),
                    bool((row.get("service") or {}).get("active")),
                ),
            )
        )
    bert_status = read_json(LATEST10_BERT_STATUS)
    bert_days = dict(bert_status.get("days") or {})
    for row in days:
        staged = bert_days.get(str(row.get("day")))
        if isinstance(staged, dict):
            row["bert"] = staged
    active_days = [
        row for row in days if bool((row.get("service") or {}).get("active"))
    ]
    ready_days = [row for row in days if row.get("stage") == "ready"]
    ready_marker = LATEST10_READY.is_file()
    if ready_marker:
        for row in days:
            row.setdefault(
                "bert",
                {
                    "day": row.get("day"),
                    "stage": "ready",
                    "host": "Bert",
                    "message": "Shard is included in the Bert-verified final manifest.",
                },
            )
    bert_ready_days = [
        row for row in days if (row.get("bert") or {}).get("stage") == "ready"
    ]
    shards_ready = len(ready_days) == 10
    finalizer = unit_state(LATEST10_FINALIZER_SERVICE)
    bootstrap = unit_state(LATEST10_BOOTSTRAP_SERVICE)
    finalizer_log = ANSI_RE.sub("", read_tail(LATEST10_FINALIZER_LOG)).replace(
        "\r", "\n"
    )
    finalizer_lines = [line.strip() for line in finalizer_log.splitlines() if line.strip()]

    current = max(
        active_days
        or [row for row in days if float(row.get("percent") or 0) < 100]
        or days,
        key=lambda row: float(row.get("percent") or 0.0),
    )
    stage = current.get("stage")
    host = current.get("host")
    latest_line = current.get("latest_line")
    current_value = current.get("current")
    total_value = current.get("total")
    unit = current.get("unit")
    rate = current.get("rate")
    current_service = current.get("service")

    if bootstrap["active"]:
        stage = "training on Blackwell"
        host = "Inzi"
        latest_line = "Latest-ten manifest validated; Blackwell bootstrap is active."
        current_value = len(ready_days)
        total_value = 10
        unit = "validated shards"
        rate = None
        current_service = bootstrap
    elif finalizer["active"]:
        last_run = next(
            (line for line in reversed(finalizer_lines) if line.startswith("[run]")),
            "",
        )
        if "assemble_feature_manifest.py" in last_run:
            stage = "assembling manifest"
            host = "Bert"
            latest_line = "Bert is hashing ten compact feature shards and assembling the manifest."
        elif "rsync" in last_run and re.search(r"bert:\S+/\s+/home/inzi/", last_run):
            stage = "returning manifest"
            host = "Bert → Inzi"
            latest_line = "Bert assembly is returning to Inzi for final digest verification."
        elif "rsync" in last_run and re.search(r"\sbert:\S+/?$", last_run):
            stage = "staging on Bert"
            host = "Inzi → Bert"
            latest_line = "Ten validated compact feature shards are staging on Bert."
        else:
            stage = "finalizing"
            host = "Inzi"
            latest_line = "All day shards are ready; the finalizer is validating the bundle."
        current_value = len(ready_days)
        total_value = 10
        unit = "validated shards"
        rate = None
        current_service = finalizer
    elif ready_marker:
        stage = "ready"
        host = "Inzi"
        latest_line = "Latest-ten manifest and post-transfer digests are validated."

    percent = sum(float(row.get("percent") or 0.0) for row in days) / 10.0
    if shards_ready and not ready_marker:
        percent = (
            min(99.0, 90.0 + len(bert_ready_days))
            if bert_days
            else 99.0
        )
    return {
        "active": bool(active_days) or finalizer["active"] or bootstrap["active"],
        "complete": ready_marker,
        "shards_ready": shards_ready,
        "started": any(float(row.get("percent") or 0.0) > 0 for row in days),
        "completed_days": len(ready_days),
        "bert_ready_days": len(bert_ready_days),
        "total_days": 10,
        "percent": percent,
        "stage": stage,
        "host": host,
        "current_day": current.get("day"),
        "current": current_value,
        "total": total_value,
        "unit": unit,
        "rate": rate,
        "latest_line": latest_line,
        "current_service": current_service,
        "finalizer_service": finalizer,
        "bootstrap_service": bootstrap,
        "days": days,
    }


def reconcile_protocol_with_live_curriculum(
    protocol: dict[str, Any],
    *,
    service: dict[str, Any],
    curriculum: dict[str, Any],
) -> dict[str, Any]:
    """Project the selected live gate instead of a prior specialist boundary."""

    if (
        protocol.get("available") is not True
        or not curriculum.get("run")
        or service.get("active") is not True
    ):
        return protocol
    gate_program = dict(curriculum.get("gate_program") or {})
    next_gate = dict(gate_program.get("next_gate") or {})
    evaluation = dict(next_gate.get("evaluation") or {})
    research_measurements = [
        row
        for row in (next_gate.get("research_measurements") or [])
        if isinstance(row, dict)
    ]
    premium_games = int(evaluation.get("games_total") or 0)
    official_games = sum(
        int(row.get("games") or 0) for row in research_measurements
    )
    if official_games <= 0:
        official_games = int(
            (curriculum.get("latest_official_heldout") or {}).get("games")
            or 0
        )
    command = str(service.get("command") or "")
    floor_match = re.search(
        r"(?:^|\s)--minimum-terminal-iteration\s+(\d+)(?:\s|$)",
        command,
    )
    iterations_match = re.search(
        r"(?:^|\s)--iterations\s+(\d+)(?:\s|$)",
        command,
    )
    preparation = dict(protocol.get("preparation") or {})
    preparation.update(
        {
            "terminal_protocol_active": True,
            "terminal_active_gate_id": (
                gate_program.get("active_gate_id")
                or next_gate.get("effective_gate_id")
                or next_gate.get("id")
            ),
            "active_specialist_service": service.get("name"),
            "current_premium_gate_games": premium_games,
            "current_official_research_games": official_games,
            "current_total_evaluation_games": (
                premium_games + official_games
            ),
            "gate_handler_minimum_completed_iteration": (
                int(floor_match.group(1))
                if floor_match
                else preparation.get(
                    "gate_handler_minimum_completed_iteration"
                )
            ),
            "terminal_iteration_ceiling": (
                max(0, int(iterations_match.group(1)) - 1)
                if iterations_match
                else preparation.get("terminal_iteration_ceiling")
            ),
            "gate_handler_source": "live_service_and_gate_program",
        }
    )
    return {**protocol, "preparation": preparation}


def authoritative_training_state(
    curriculum: dict[str, Any],
    transition: dict[str, Any],
    owner_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the live trainer before completed bootstrap history.

    ``ALAKAZAM_BOOTSTRAP_READY`` is intentionally durable, so its existence is
    not evidence that bootstrap is still the active workload. Once a
    curriculum run is known, mirror that run into the legacy ``training``
    payload instead of allowing the completed marker to shadow production.
    """
    owner_handoff = owner_handoff or {}
    if owner_handoff.get("active") or owner_handoff.get("terminal_failure"):
        handoff_active = bool(owner_handoff.get("active"))
        terminal_failure = bool(owner_handoff.get("terminal_failure"))
        return {
            "authoritative": True,
            "source": owner_handoff.get("source"),
            "log": owner_handoff.get("log"),
            "latest_line": owner_handoff.get("latest_line"),
            "updated_at": owner_handoff.get("updated_at"),
            "fresh": True,
            "status": "running" if handoff_active else "failed",
            "mode": "specialist_handoff",
            "phase": owner_handoff.get("stage"),
            "epoch": owner_handoff.get("epoch"),
            "epochs_target": owner_handoff.get("epochs_target"),
            "current": owner_handoff.get("current"),
            "total": owner_handoff.get("total"),
            "percent": owner_handoff.get("percent"),
            "rate": owner_handoff.get("rate"),
            "rate_unit": owner_handoff.get("rate_unit"),
            "metrics": {},
            "run": (
                owner_handoff.get("source_specialist_id")
                or owner_handoff.get("next_specialist_id")
                or "specialist-cycle-handoff"
            ),
            "service": {
                "active": handoff_active,
                "pid": owner_handoff.get("pid"),
                "memory_bytes": owner_handoff.get("memory_bytes"),
                "source": owner_handoff.get("source"),
            },
            "terminal_failure": terminal_failure,
            "core_gameplay_regression": owner_handoff.get(
                "core_gameplay_regression"
            ),
        }
    if curriculum.get("run"):
        progress = curriculum.get("progress") or {}
        worker = curriculum.get("worker") or {}
        active_pids = curriculum.get("active_pids") or []
        curriculum_active = bool(curriculum.get("active"))
        latest_line = progress.get("line")
        if not curriculum_active and latest_line:
            latest_line = f"Last stopped-run progress (historical): {latest_line}"
        return {
            "authoritative": True,
            "source": curriculum.get("progress_source"),
            "log": curriculum.get("progress_log_source"),
            "latest_line": latest_line,
            "updated_at": curriculum.get("progress_updated_at"),
            # A recently-written progress file is not live evidence after the
            # owning curriculum service has stopped. Keep the last frame for
            # diagnosis, but never label it current/running in the legacy API.
            "fresh": bool(curriculum_active and curriculum.get("source_current")),
            "status": "running" if curriculum_active else "stopped",
            "mode": "curriculum_rl",
            "phase": progress.get("stage") or curriculum.get("stage"),
            "epoch": progress.get("epoch"),
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "rate": progress.get("rate"),
            "rate_unit": progress.get("rate_unit"),
            "samples_per_second": progress.get("sps"),
            "game_equivalents_per_second": progress.get("gps"),
            "eta": progress.get("eta"),
            "metrics": progress.get("metrics") or {},
            "run": curriculum.get("run"),
            "iteration": progress.get("iteration", curriculum.get("iteration")),
            "service": {
                "active": bool(curriculum.get("active")),
                "pid": active_pids[0] if active_pids else None,
                "memory_bytes": worker.get("rss_bytes"),
                "source": worker.get("source"),
            },
        }

    bootstrap = transition.get("bootstrap") or {}
    bootstrap_live = bool(
        bootstrap.get("active")
        and (
            int(bootstrap.get("pid") or 0) > 0
            or bootstrap.get("sub_state") == "running"
        )
    )
    if (
        ALAKAZAM_BOOTSTRAP_READY.is_file()
        or bootstrap_live
        or transition.get("status")
        == "training_alakazam_expert_bootstrap_blackwell_device_resident"
    ):
        return alakazam_bootstrap_progress()
    return exact_training_state()


def active_specialist_commit_overlay(
    active_run: dict[str, Any],
) -> dict[str, Any]:
    """Resolve live counters/results from the append-only active-run ledger."""
    run_value = str(active_run.get("path") or "").strip()
    if not run_value:
        return {"available": False, "reason": "active run path is absent"}
    run_dir = Path(run_value).expanduser().resolve()
    loop_path = run_dir / "loop_state.json"
    loop = read_json(loop_path)
    try:
        last_completed = int(loop["last_completed_iteration"])
        next_iteration = int(loop["next_iteration"])
    except (KeyError, TypeError, ValueError):
        return {
            "available": False,
            "reason": "active run loop state is unavailable or incomplete",
            "run_path": str(run_dir),
        }
    if next_iteration != last_completed + 1:
        return {
            "available": False,
            "reason": "active run loop iteration boundary is inconsistent",
            "run_path": str(run_dir),
        }
    commit_path = run_dir / "commits" / f"iter_{last_completed:05d}.json"
    commit = read_json(commit_path)
    history = commit.get("history")
    if not isinstance(history, list):
        return {
            "available": False,
            "reason": "latest immutable commit history is unavailable",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    completed_rows: list[dict[str, Any]] = []
    for raw in history:
        if not isinstance(raw, dict) or raw.get("completed") is not True:
            continue
        try:
            iteration = int(raw.get("iteration", -1))
        except (TypeError, ValueError):
            continue
        if iteration >= 0:
            completed_rows.append(dict(raw))
    latest = next(
        (
            row
            for row in reversed(completed_rows)
            if int(row.get("iteration", -1)) == last_completed
        ),
        None,
    )
    if latest is None:
        return {
            "available": False,
            "reason": "latest immutable commit lacks its completed history row",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    candidate = latest.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    checkpoint = str(candidate.get("path") or "")
    checkpoint_digest = str(candidate.get("digest") or "")
    if not checkpoint or not checkpoint_digest.startswith("sha256:"):
        return {
            "available": False,
            "reason": "latest immutable commit lacks candidate identity",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    learner = commit.get("learner")
    learner = learner if isinstance(learner, dict) else {}
    latest_learner = latest.get("learner_after")
    latest_learner = (
        latest_learner if isinstance(latest_learner, dict) else {}
    )
    learner_checkpoint = str(
        learner.get("path") or latest_learner.get("path") or checkpoint
    )
    learner_digest = str(
        learner.get("digest") or latest_learner.get("digest") or checkpoint_digest
    )
    if (
        not learner_checkpoint
        or not learner_digest.startswith("sha256:")
        or (
            learner
            and latest_learner
            and (
                learner_checkpoint != str(latest_learner.get("path") or "")
                or learner_digest != str(latest_learner.get("digest") or "")
            )
        )
    ):
        return {
            "available": False,
            "reason": "latest immutable commit learner identity is inconsistent",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    premium = latest.get("active_gate_result")
    premium = dict(premium) if isinstance(premium, dict) else {}
    if premium and (
        int(premium.get("iteration", -1)) != last_completed
        or str(premium.get("checkpoint_digest") or "") != checkpoint_digest
    ):
        return {
            "available": False,
            "reason": "latest active-gate result identity is inconsistent",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    research_path = (
        run_dir / "research_controls" / f"iter_{last_completed:05d}.json"
    )
    research = read_json(research_path)
    if research and (
        int(research.get("iteration", -1)) != last_completed
        or str(research.get("checkpoint_digest") or "") != checkpoint_digest
    ):
        return {
            "available": False,
            "reason": "latest research-control result identity is inconsistent",
            "run_path": str(run_dir),
            "commit_path": str(commit_path),
        }
    incumbent = latest.get("incumbent_after")
    incumbent = incumbent if isinstance(incumbent, dict) else {}
    protected_champion = commit.get("champion")
    protected_champion = (
        protected_champion
        if isinstance(protected_champion, dict)
        else incumbent
    )
    promotion = latest.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}
    continuous_learner = promotion.get("continuous_learner")
    continuous_learner = (
        continuous_learner
        if isinstance(continuous_learner, dict)
        else {}
    )
    exact_gate_regression = continuous_learner.get(
        "exact_gate_regression"
    )
    exact_gate_regression = (
        exact_gate_regression
        if isinstance(exact_gate_regression, dict)
        else {}
    )
    return {
        "available": True,
        "source": "latest_immutable_active_run_commit",
        "run_path": str(run_dir),
        "loop_state_path": str(loop_path),
        "commit_path": str(commit_path),
        "last_completed_iteration": last_completed,
        "next_iteration": next_iteration,
        "rl_iterations_completed": len(completed_rows),
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "learner_checkpoint": learner_checkpoint,
        "learner_digest": learner_digest,
        "protected_champion": protected_champion or None,
        "candidate_promoted": latest.get("promoted") is True,
        "heldout_champion_updated": (
            latest.get("heldout_champion_updated") is True
        ),
        "exact_gate_regression": exact_gate_regression or None,
        "premium_holdout": premium or None,
        "official_research": research or None,
    }


def reconcile_frozen_specialist_rows(
    rows: list[dict[str, Any]],
    frozen_specialist_ids: set[str],
    active_specialist_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """Let the immutable frozen registry supersede stale mutable row fields."""

    reconciled: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if str(row.get("id") or "") in frozen_specialist_ids:
            row.update(
                {
                    "status": "passed_frozen",
                    "active": False,
                    "frozen": True,
                    "public_mix_eligible": True,
                    "immutable_registry_overlay": True,
                }
            )
        else:
            row["immutable_registry_overlay"] = False
        reconciled.append(row)
    counts: dict[str, int] = {}
    for row in reconciled:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    active = str(active_specialist_id or "")
    if active in frozen_specialist_ids:
        active = ""
    return reconciled, counts, active


def prestage_receipt_is_current(
    receipt: dict[str, Any],
    active_specialist_id: str,
) -> bool:
    """Accept an explicit blocked receipt without inventing a next target."""
    status = str(receipt.get("status") or "")
    return bool(
        receipt.get("schema") == "poke_bot.next_specialist_prestage/v1"
        and receipt.get("active_specialist") == active_specialist_id
        and status in {"ready", "blocked"}
        and (
            status == "blocked"
            or bool(str(receipt.get("selected_specialist") or ""))
        )
    )


def canonical_next_prestage_overlay(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the staged next-specialist requirement owned by canonical state.

    A checksum-ready historical V5 pre-stage remains useful evidence, but it
    cannot be rendered as executable when the canonical V6 handoff requires
    the separate expanded-target corpus.  Keeping this projection pure makes
    the dashboard behavior regression-testable without live services.
    """

    current = payload.get("current")
    current_row = (
        current.get("next_successor_prestage")
        if isinstance(current, dict)
        else None
    )
    if isinstance(current_row, dict):
        corpus = dict(current_row.get("corpus") or {})
        runtime_route = dict(current_row.get("runtime_route") or {})
        blocker = str(
            runtime_route.get("blocker")
            or (current_row.get("blockers") or [None])[0]
            or ""
        )
        corpus_ready = str(corpus.get("status") or "").startswith("ready")
        return {
            "status": str(current_row.get("status") or "") or None,
            "blocker": blocker or None,
            "intended_specialist": (
                str(current_row.get("specialist_id") or "") or None
            ),
            "blocks_v6_handoff": current_row.get("pre_stage_ready") is not True,
            "receipt": current_row.get("pre_stage_receipt"),
            "cpu_pack_status": (
                "not_built"
                if current_row.get("pre_stage_ready") is not True
                else "ready"
            ),
            "representative_ready": (
                current_row.get("representative_ready") is True
            ),
            "expert_corpus_ready": corpus_ready,
            "expert_corpus_pointer": corpus.get("protected_pointer"),
            "expert_records": int(corpus.get("records") or 0),
            "expert_decisions": int(corpus.get("decisions") or 0),
            "guide_rows": int(corpus.get("guide_rows") or 0),
        }
    archive = payload.get("expert_corpus_archive")
    policy = (
        archive.get("canonical_policy")
        if isinstance(archive, dict)
        else None
    )
    row = (
        policy.get("next_specialist_prestage")
        if isinstance(policy, dict)
        else None
    )
    if not isinstance(row, dict):
        return {}
    status = str(row.get("status") or "")
    blocker = str(row.get("blocker") or "")
    intended = str(
        row.get("intended_next_specialist_after_corpus_validation") or ""
    )
    blocks_v6 = bool(
        status == "blocked_waiting_for_expanded_v6_corpus"
        or blocker == "protocol_valid_expert_corpus_not_ready"
    )
    return {
        "status": status or None,
        "blocker": blocker or None,
        "intended_specialist": intended or None,
        "blocks_v6_handoff": blocks_v6,
        "receipt": row.get("receipt"),
        "cpu_pack_status": row.get("cpu_pack_status"),
    }


def specialist_protocol_state(
    path: Path | None = None,
    *,
    runtime_specialist_id: str | None = None,
    runtime_run_name: str | None = None,
    runtime_service_state: str | None = None,
) -> dict[str, Any]:
    """Return a compact, validated view of the canonical specialist tracker."""
    source = Path(path or SPECIALIST_PROTOCOL_STATE)
    if not source.is_file():
        return {
            "available": False,
            "reason": "canonical specialist state is unavailable",
            "source": str(source),
        }
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError, TypeError) as exc:
        return {
            "available": False,
            "reason": f"canonical specialist state failed to parse: {exc}",
            "source": str(source),
        }
    if not isinstance(payload, dict):
        return {
            "available": False,
            "reason": "canonical specialist state is not a mapping",
            "source": str(source),
        }
    current = payload.get("current")
    program_progress = (
        dict(current.get("program_progress") or {})
        if isinstance(current, dict)
        else {}
    )
    registry = payload.get("target_registry")
    shared_core = payload.get("shared_core")
    heads_and_datasets = payload.get("heads_and_datasets")
    specialists = payload.get("specialists")
    population = payload.get("population_training")
    post_fleet_refresh = payload.get("post_fleet_refresh")
    allowed = payload.get("allowed_status_values")
    allowed_statuses = (
        set(allowed.get("specialist") or []) if isinstance(allowed, dict) else set()
    )
    if (
        payload.get("schema_version")
        not in {
            "poke_bot.specialist_state/v1",
            "poke_bot.specialist_state/v2",
        }
        or not isinstance(current, dict)
        or not isinstance(registry, dict)
        or not isinstance(shared_core, dict)
        or not isinstance(heads_and_datasets, dict)
        or not isinstance(specialists, list)
        or not specialists
        or not isinstance(population, dict)
        or not allowed_statuses
    ):
        return {
            "available": False,
            "reason": "canonical specialist state schema mismatch",
            "source": str(source),
        }
    canonical_roster_path = source.parent / "matchup_adapter_roster.json"
    canonical_roster = read_json(canonical_roster_path)
    canonical_roster_present = canonical_roster_path.is_file()
    canonical_ids: list[str] = []
    if canonical_roster_present:
        canonical_ids = [
            str(value)
            for value in (canonical_roster.get("active_expert_ids") or [])
            if str(value)
        ]
        if (
            canonical_roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
            or len(canonical_ids) != len(set(canonical_ids))
            or len(canonical_ids)
            != int(canonical_roster.get("required_specialist_count") or 0)
        ):
            return {
                "available": False,
                "reason": "stable canonical matchup roster is invalid",
                "source": str(canonical_roster_path),
            }
        # The stable matchup roster owns causal adapter routes, not the
        # required training/completion plan. Preserve and validate it
        # independently, but never use it to reintroduce owner-removed targets
        # or drop a newly staged specialist that has not received a route yet.
    active_run = current.get("active_run")
    active_run = active_run if isinstance(active_run, dict) else {}
    canonical_live_execution = active_specialist_commit_overlay(active_run)
    canonical_active_id = str(current.get("active_specialist") or "")
    canonical_refresh_id = ""
    if isinstance(post_fleet_refresh, dict):
        canonical_refresh_id = str(
            post_fleet_refresh.get("active_refresh_specialist_id") or ""
        ).strip().lower()
    canonical_execution_id = canonical_active_id or canonical_refresh_id
    runtime_specialist_id = str(runtime_specialist_id or "").strip().lower()
    runtime_run_name = str(runtime_run_name or "").strip()
    runtime_service_state = str(runtime_service_state or "").strip()
    canonical_active_run_id = str(
        active_run.get("active_specialist") or ""
    ).strip().lower()
    canonical_active_run_name = str(
        active_run.get("run_name") or ""
    ).strip()
    runtime_identity_reconciled = bool(
        runtime_specialist_id
        and (
            runtime_specialist_id != canonical_execution_id
            or (
                canonical_active_run_id
                and canonical_active_run_id != runtime_specialist_id
            )
            or (
                runtime_run_name
                and canonical_active_run_name
                and canonical_active_run_name != runtime_run_name
            )
        )
    )
    canonical_pointer_stale = bool(
        runtime_specialist_id
        and runtime_specialist_id != canonical_execution_id
    )
    active_id = runtime_specialist_id or canonical_active_id
    live_execution = canonical_live_execution
    if runtime_identity_reconciled and runtime_run_name:
        runtime_path = (
            Path("/home/inzi/poke-bot-agent/outputs/pure_rl")
            / Path(runtime_run_name).name
        )
        live_execution = active_specialist_commit_overlay(
            {"path": str(runtime_path)}
        )
    runtime_execution_reconciled = bool(
        live_execution.get("available") is True
        and (
            int(
                active_run.get("last_completed_iteration")
                if active_run.get("last_completed_iteration") is not None
                else -1
            )
            != int(
                live_execution.get("last_completed_iteration")
                if live_execution.get("last_completed_iteration") is not None
                else -1
            )
            or int(
                active_run.get("next_iteration")
                if active_run.get("next_iteration") is not None
                else -1
            )
            != int(
                live_execution.get("next_iteration")
                if live_execution.get("next_iteration") is not None
                else -1
            )
            or str(active_run.get("learner_checksum") or "")
            != str(
                live_execution.get("learner_digest")
                or live_execution.get("checkpoint_digest")
                or ""
            )
        )
    )
    runtime_reconciled = (
        runtime_identity_reconciled or runtime_execution_reconciled
    )
    rows: list[dict[str, Any]] = []
    retained_non_specialist_opponents: list[dict[str, Any]] = []
    ids: list[str] = []
    for raw in specialists:
        if not isinstance(raw, dict):
            return {
                "available": False,
                "reason": "canonical specialist record is invalid",
                "source": str(source),
            }
        specialist_id = str(raw.get("id") or "")
        status = str(raw.get("status") or "")
        if not specialist_id or status not in allowed_statuses:
            return {
                "available": False,
                "reason": "canonical specialist identity/status is invalid",
                "source": str(source),
            }
        ids.append(specialist_id)
        required_specialist = raw.get("required_specialist") is not False
        compact_row = {
                "id": specialist_id,
                "name": str(raw.get("name") or specialist_id),
                "deck_family_name": raw.get("deck_family_name"),
                "secondary_search_alias": raw.get("secondary_search_alias"),
                "status": status,
                "active": specialist_id == active_id,
                "frozen": raw.get("frozen") is True,
                "public_mix_eligible": raw.get("public_mix_eligible") is True,
                "public_mix_opponent_ids": [
                    str(value)
                    for value in (raw.get("public_mix_opponent_ids") or [])
                    if str(value)
                ],
                "premium_holdout_tier": raw.get("premium_holdout_tier"),
                "required_specialist": required_specialist,
                "bootstrap_epochs_completed": int(
                    ((raw.get("counters") or {}).get("bootstrap_epochs_completed"))
                    or 0
                ),
                "rl_iterations_completed": int(
                    ((raw.get("counters") or {}).get("rl_iterations_completed"))
                    or 0
                ),
                "last_completed_iteration": (
                    (raw.get("counters") or {}).get("last_completed_iteration")
                ),
                "next_iteration": (
                    (raw.get("counters") or {}).get("next_iteration")
                ),
            }
        if not required_specialist:
            public_opponent = dict(
                raw.get("public_practice_gate_opponent") or {}
            )
            matchup_router = dict(raw.get("matchup_router") or {})
            post_fleet_required = (
                raw.get("post_fleet_specialist_required") is True
            )
            retained_non_specialist_opponents.append(
                {
                    "id": specialist_id,
                    "name": str(raw.get("name") or specialist_id),
                    "display_status": str(
                        raw.get("display_status")
                        or (
                            "historical_artifacts_preserved_"
                            "inference_only_not_planned_for_training"
                        )
                    ),
                    "role_label": (
                        "FROZEN S-TIER H10 TRAINING MODEL"
                        if raw.get("training_use_only") is True
                        else (
                        "NEXT H10 SPECIALIST AFTER MARNIE"
                        if post_fleet_required
                        else (
                            "PUBLIC OPPONENT + ACTIVE ROUTE, "
                            "NO SPECIALIST TRAIN"
                        ))
                    ),
                    "premium_holdout_tier": raw.get("premium_holdout_tier"),
                    "frozen": raw.get("frozen") is True,
                    "public_mix_eligible": raw.get("public_mix_eligible") is True,
                    "required_specialist": False,
                    "selection_eligible": (
                        raw.get("selector_eligible") is True
                    ),
                    "completion_eligible": (
                        raw.get("completion_eligible") is True
                    ),
                    "training_authorized": False,
                    "submission_authorized": (
                        raw.get("submission_authorized") is True
                    ),
                    "matchup_route_preserved": (
                        raw.get(
                            "public_opponent_and_matchup_router_preserved"
                        )
                        is True
                    ),
                    "stable_matchup_slot": matchup_router.get(
                        "stable_matchup_slot"
                    ),
                    "stable_matchup_slot_status": matchup_router.get(
                        "status"
                    ),
                    "public_practice_gate_opponent": public_opponent,
                    "inference_only": (
                        public_opponent.get("inference_only") is True
                    ),
                    "historical_artifacts_preserved": True,
                    "future_specialist_training_planned": post_fleet_required,
                    "post_fleet_specialist_required": post_fleet_required,
                    "source": str(source),
                }
            )
            continue
        if specialist_id == active_id and live_execution.get("available") is True:
            compact_row.update(
                {
                    "rl_iterations_completed": int(
                        live_execution["rl_iterations_completed"]
                    ),
                    "last_completed_iteration": int(
                        live_execution["last_completed_iteration"]
                    ),
                    "next_iteration": int(live_execution["next_iteration"]),
                    "live_commit_overlay": True,
                }
            )
        else:
            compact_row["live_commit_overlay"] = False
        rows.append(compact_row)
    expected_count = int(registry.get("required_target_count") or 0)
    canonical_active_rows = [
        str(raw.get("id") or "")
        for raw in specialists
        if (
            isinstance(raw, dict)
            and raw.get("required_specialist") is not False
            and raw.get("active") is True
        )
    ]
    phase = str(current.get("phase") or "")
    allows_no_active = phase in {
        "shared_core_derivation",
        "specialist_core_refresh_handoff",
        "specialist_handoff_waiting_for_teal_full33_corpus",
        "next_specialist_selected_readiness_blocked",
        "post_fleet_specialist_refresh",
        "post_fleet_specialist_refresh_rl",
    }
    post_fleet_required_count = sum(
        1
        for raw in specialists
        if (
            isinstance(raw, dict)
            and raw.get("required_specialist") is False
            and raw.get("post_fleet_specialist_required") is True
        )
    )
    effective_program_target_count = len(rows) + post_fleet_required_count
    if (
        len(ids) != len(set(ids))
        or effective_program_target_count != expected_count
        or (
            allows_no_active
            and (canonical_active_rows or canonical_active_id)
        )
        or (
            not allows_no_active
            and (
                len(canonical_active_rows) != 1
                or canonical_active_rows[0] != canonical_active_id
            )
        )
    ):
        return {
            "available": False,
            "reason": "canonical specialist roster invariants failed",
            "source": str(source),
        }
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    priority = payload.get("training_priority")
    priority = priority if isinstance(priority, dict) else {}
    strict_prefix_contract = dict(
        priority.get("strict_post_spidops_prefix") or {}
    )
    owner_removal_contract = dict(priority.get("owner_removal") or {})
    goal_projection = read_json(ROOT / "ops/current_goal_requirements.json")
    projected_overrides = goal_projection.get("current_owner_overrides") or {}
    projected_guide_snapshot = (
        (goal_projection.get("verified_snapshot") or {}).get(
            "current_deck_guides"
        )
        or {}
    )
    projected_guide_policy = (
        projected_guide_snapshot.get("goal_path_guidance") or {}
    )
    projected_future_guide_scope = (
        projected_overrides.get("future_guide_strategic_branch_scope") or {}
    )
    projected_future_head_action = projected_future_guide_scope
    projected_future_setup_head = (
        projected_future_guide_scope.get("setup_board_outcome_head") or {}
    )
    projected_teal_legacy_guide = (
        projected_overrides.get("teal_guide_weight_nonwinning_reduction")
        or {}
    )
    projected_marnie_milestones = dict(
        projected_overrides.get("final_format_marnie_milestone_submissions")
        or {}
    )
    projected_marnie_policy = dict(
        projected_overrides.get("marnie_neural_policy_challenger") or {}
    )
    current_deck_guide_weight_policy: dict[str, Any] = {}
    if projected_guide_policy:
        current_deck_guide_weight_policy = {
            **projected_guide_policy,
            "guide_curriculum_revision": projected_future_guide_scope.get(
                "guide_curriculum_revision"
            ),
            "strategic_branch_scope_revision": (
                projected_future_guide_scope.get(
                    "strategic_branch_scope_revision"
                )
            ),
            "head_action_scope_revision": (
                projected_future_guide_scope.get(
                    "head_action_scope_revision"
                )
            ),
            "learning_effect": (
                "literal_multiplier_on_bounded_guide_conditioned_"
                "strategic_head_curriculum"
            ),
            "gradient_effect": (
                "scales_guide_conditioned_strategic_head_gradient_"
                "contribution"
            ),
            "direct_policy_cross_entropy_allowed": bool(
                projected_future_guide_scope.get(
                    "direct_policy_cross_entropy_allowed"
                )
            ),
            "bootstrap_weight_ramp": projected_guide_snapshot.get(
                "bootstrap_weight_ramp"
            ),
            "bootstrap_maximum_weight": projected_guide_snapshot.get(
                "maximum_weight"
            ),
            "bootstrap_maximum_weight_scope": projected_guide_snapshot.get(
                "maximum_weight_scope"
            ),
            "maximum_post_bootstrap_auxiliary_weight": (
                projected_guide_snapshot.get(
                    "maximum_post_bootstrap_auxiliary_weight"
                )
            ),
            "post_bootstrap_behavior": projected_guide_snapshot.get(
                "post_bootstrap_behavior"
            ),
            "source": "current_goal_requirements_owner_projection",
        }
    current_deck_guide_training_modes: dict[str, Any] = {}
    if projected_future_guide_scope:
        legacy_weight = projected_teal_legacy_guide.get(
            "active_iteration_13_weight"
        )
        if legacy_weight is None:
            legacy_weight = projected_teal_legacy_guide.get("target_weight")
        current_deck_guide_training_modes = {
            "active_started_lineage": {
                "specialist_id": "teal-mask-ogerpon-ex",
                "display_name": "Slop Box (Teal Mask Ogerpon ex)",
                "is_active": active_id == "teal-mask-ogerpon-ex",
                "scope": "already_started_legacy_run",
                "mode": "confidence_weighted_policy_cross_entropy",
                "guide_weight": legacy_weight,
                "revision_51_retrofit_allowed": False,
                "runtime_input_authority": False,
                "action_selection_authority": False,
                "serving_authority": False,
            },
            "future_lineage": {
                "scope": projected_future_guide_scope.get("scope"),
                "effective_from_specialist": (
                    projected_future_guide_scope.get(
                        "prospective_effective_specialist"
                    )
                ),
                "guide_curriculum_revision": (
                    projected_future_guide_scope.get(
                        "guide_curriculum_revision"
                    )
                ),
                "mode": projected_future_guide_scope.get(
                    "training_target_mode"
                ),
                "direct_policy_cross_entropy_allowed": bool(
                    projected_future_guide_scope.get(
                        "direct_policy_cross_entropy_allowed"
                    )
                ),
                "guide_runtime_input_allowed": bool(
                    projected_future_guide_scope.get(
                        "guide_runtime_input_allowed"
                    )
                ),
                "guide_action_selection_allowed": bool(
                    projected_future_guide_scope.get(
                        "guide_action_selection_allowed"
                    )
                ),
                "replace_observed_outcome_targets_allowed": bool(
                    projected_future_guide_scope.get(
                        "replace_observed_outcome_targets_allowed"
                    )
                ),
                "curriculum_focus": projected_future_guide_scope.get(
                    "curriculum_focus"
                ),
                "fused_policy_learning_authority": (
                    projected_future_guide_scope.get(
                        "fused_policy_learning_authority"
                    )
                ),
                "activation_requires_prestage_validation_receipt": (
                    projected_future_guide_scope.get(
                        "activation_requires_prestage_validation_receipt"
                    )
                    is True
                ),
            },
            "future_head_action_contract": {
                "head_action_scope_revision": (
                    projected_future_guide_scope.get(
                        "head_action_scope_revision"
                    )
                ),
                "all_future_heads_must_influence_actions": (
                    projected_future_guide_scope.get(
                        "all_future_heads_must_influence_actions"
                    )
                    is True
                ),
                "owner_decision_revision": (
                    projected_future_head_action.get(
                        "owner_decision_revision"
                    )
                ),
                "schema": projected_future_head_action.get(
                    "decision_fusion_schema"
                ),
                "preserve_v1_additive_residual": (
                    projected_future_head_action.get(
                        "parent_v1_fusion_residual_preserved"
                    )
                    is True
                ),
                "computation_role": projected_future_head_action.get(
                    "required_computation_role"
                ),
                "fusion_role": (
                    (
                        projected_future_head_action.get(
                            "allowed_fusion_roles"
                        )
                        or [None]
                    )[0]
                ),
                "action_influence": projected_future_head_action.get(
                    "required_action_influence"
                ),
                "state_head_action_conditioning": (
                    projected_future_head_action.get(
                        "state_head_action_conditioning"
                    )
                ),
                "option_head_action_conditioning": (
                    projected_future_head_action.get(
                        "option_head_action_conditioning"
                    )
                ),
                "route_architecture": projected_future_head_action.get(
                    "action_route_granularity"
                ),
                "existing_learned_decision_source_count": (
                    projected_future_head_action.get(
                        "existing_learned_decision_source_count"
                    )
                ),
                "canonical_learned_decision_source_count_with_setup": (
                    projected_future_head_action.get(
                        "canonical_learned_decision_source_count_with_setup"
                    )
                ),
                "setup_source_included_when_present": (
                    projected_future_head_action.get(
                        "setup_source_included_when_present"
                    )
                    is True
                ),
                "guide_is_sole_no_route_exception": (
                    projected_future_head_action.get(
                        "guide_is_only_action_route_exception"
                    )
                    is True
                ),
                "route_reduction": projected_future_head_action.get(
                    "route_aggregation"
                ),
                "aggregate_absolute_cap": projected_future_head_action.get(
                    "aggregate_route_delta_logit_cap"
                ),
                "zero_safe_final_projections": (
                    projected_future_head_action.get(
                        "route_final_projection_initialization"
                    )
                    == "exact_zero"
                ),
                "independent_means_pre_fusion_computation_not_action_isolation": (
                    projected_future_head_action.get(
                        "independent_means_pre_fusion_computation_"
                        "not_action_isolation"
                    )
                    is True
                ),
                "direct_action_selection_authority": (
                    projected_future_head_action.get(
                        "direct_action_selection_authority"
                    )
                    is True
                ),
                "fusion_selects_action": (
                    projected_future_head_action.get("fusion_selects_action")
                    is True
                ),
                "materially_influences_fused_logits": (
                    projected_future_head_action.get(
                        "materially_influences_fused_logits"
                    )
                    is True
                ),
                "runtime_enabled": (
                    projected_future_head_action.get("runtime_enabled") is True
                ),
                "runtime_activation_requirement": (
                    projected_future_head_action.get(
                        "runtime_activation_requirement"
                    )
                ),
                "setup_board_outcome_head": {
                    "id": projected_future_setup_head.get("id"),
                    "owner_decision_revision": (
                        projected_future_setup_head.get(
                            "owner_decision_revision"
                        )
                    ),
                    "computation_role": projected_future_setup_head.get(
                        "computation_role"
                    ),
                    "fusion_role": projected_future_setup_head.get(
                        "fusion_role"
                    ),
                    "action_influence": projected_future_setup_head.get(
                        "action_influence"
                    ),
                    "causal_input": projected_future_setup_head.get(
                        "causal_input"
                    ),
                    "fusion_route_initialization": (
                        projected_future_setup_head.get(
                            "fusion_route_initialization"
                        )
                    ),
                },
            },
            "source": "current_goal_requirements_owner_projection",
        }
    projected_plan = (
        projected_overrides.get("required_specialist_plan")
        or {}
    )
    projected_revision = int(projected_plan.get("goal_revision") or 0)
    state_revision = max(
        int(strict_prefix_contract.get("decision_revision") or 0),
        int(owner_removal_contract.get("decision_revision") or 0),
    )
    if projected_revision > state_revision:
        projected_prefix = [
            str(value)
            for value in (
                projected_plan.get("strict_post_spidops_prefix") or []
            )
            if str(value)
        ]
        projected_removed = [
            str(value)
            for value in (
                projected_plan.get("removed_specialist_ids") or []
            )
            if str(value)
        ]
        if projected_prefix:
            strict_prefix_contract = {
                "decision_revision": projected_revision,
                "ids": projected_prefix,
                "missing_input_behavior": projected_plan.get(
                    "missing_strict_prefix_input_behavior"
                ),
                "activation": projected_plan.get("activation_boundary"),
                "source": "current_goal_requirements_owner_projection",
            }
        if projected_removed:
            owner_removal_contract = {
                "decision_revision": projected_revision,
                "specialist_ids": projected_removed,
                "selection_eligible": bool(
                    projected_plan.get("removed_ids_selection_eligible")
                ),
                "counts_toward_completion": bool(
                    projected_plan.get("removed_ids_count_toward_completion")
                ),
                "preserve_historical_corpus_router_and_audit_artifacts": bool(
                    projected_plan.get(
                        "removed_ids_historical_artifacts_preserved"
                    )
                ),
                "source": "current_goal_requirements_owner_projection",
            }
    strict_prefix_ids = [
        str(value)
        for value in (strict_prefix_contract.get("ids") or [])
        if str(value)
    ]
    owner_removed_ids = {
        str(value)
        for value in (owner_removal_contract.get("specialist_ids") or [])
        if str(value)
    }
    planning_required_count = int(
        projected_plan.get("required_specialists_total") or expected_count
    )
    projected_version_namespaces = dict(
        projected_overrides.get("version_namespaces") or {}
    )
    projected_core_system = dict(
        projected_version_namespaces.get("core_system") or {}
    )
    projected_core = dict(
        projected_version_namespaces.get("cumulative_core")
        or projected_overrides.get("cumulative_core")
        or {}
    )
    projected_matchup_adapter = dict(
        projected_version_namespaces.get("matchup_adapter") or {}
    )
    current_core_refresh = dict(payload.get("current_cumulative_core_refresh") or {})
    accepted_core = dict(current_core_refresh.get("latest_accepted_core") or {})
    latest_accepted_core_version = int(
        projected_core.get("latest_accepted_version")
        or accepted_core.get("version")
        or 0
    )
    active_policy_generation = int(
        projected_marnie_policy.get("accepted_policy_generation") or 0
    )
    active_policy_receipt_bound = bool(
        str(projected_marnie_policy.get("status") or "").startswith(
            "active_generation_"
        )
        and str(
            projected_marnie_policy.get("activation_receipt_sha256") or ""
        ).startswith("sha256:")
        and str(
            projected_marnie_policy.get("activation_checkpoint_sha256") or ""
        ).startswith("sha256:")
    )
    if active_policy_receipt_bound and active_policy_generation > 0:
        latest_accepted_core_version = active_policy_generation
    attempted_core_versions: list[tuple[int, str]] = []
    if current_core_refresh.get("output_core_version"):
        attempted_core_versions.append(
            (
                int(current_core_refresh["output_core_version"]),
                str(current_core_refresh.get("status") or ""),
            )
        )
    immutable_core_receipts_found = False
    for ready_receipt in (
        ROOT / "outputs/state"
    ).glob("deck-agnostic-core-cumulative-v*-fused-v1-ready.json"):
        match = re.search(
            r"cumulative-v(\d+)-fused-v1-ready\.json$",
            ready_receipt.name,
        )
        if not match:
            continue
        immutable_core_receipts_found = True
        version = int(match.group(1))
        regression = read_json(
            ready_receipt.with_name(
                ready_receipt.name.replace(
                    "-ready.json",
                    "-gameplay-regression.json",
                )
            )
        )
        if regression.get("passed") is False:
            receipt_status = "rejected_gameplay_regression"
        elif regression.get("passed") is True:
            receipt_status = "gameplay_regression_passed"
        else:
            receipt_status = str(
                read_json(ready_receipt).get("status")
                or "candidate_receipt_present"
            )
        attempted_core_versions.append((version, receipt_status))
    # The explicit owner projection is the display authority. Add it last so
    # older mutable-state labels (for example, a fallback activation suffix)
    # cannot overwrite the canonical rejection stage for the same generation.
    for key, value in projected_core.items():
        match = re.search(r"(?:^|_)v(\d+)_status$", str(key))
        if match:
            attempted_core_versions.append((int(match.group(1)), str(value)))
    projected_latest_attempted_version = int(
        projected_core.get("latest_attempted_version") or 0
    )
    if projected_latest_attempted_version:
        attempted_core_versions.append(
            (
                projected_latest_attempted_version,
                str(
                    projected_core.get(
                        f"v{projected_latest_attempted_version}_status"
                    )
                    or ""
                ),
            )
        )
    latest_attempted_core = (
        max(attempted_core_versions, key=lambda row: row[0])
        if attempted_core_versions
        else (0, "")
    )
    core_generation = {
        "core_system_revision": (
            int(projected_core_system.get("current_revision") or 0) or None
        ),
        "core_system_display_name": (
            (
                f"{projected_core_system.get('display_namespace')} "
                f"{int(projected_core_system.get('current_revision') or 0)}"
            )
            if projected_core_system.get("display_namespace")
            and projected_core_system.get("current_revision")
            else None
        ),
        "checkpoint_display_namespace": (
            projected_core.get("display_namespace")
            or "Accepted Policy Generation"
        ),
        "latest_accepted_version": latest_accepted_core_version or None,
        "latest_accepted_checkpoint": (
            projected_marnie_policy.get("activation_checkpoint_sha256")
            if active_policy_receipt_bound
            else projected_core.get("latest_accepted_checkpoint")
            or accepted_core.get("checkpoint")
            or shared_core.get("checkpoint")
        ),
        "active_policy_status": (
            projected_marnie_policy.get("status")
            if active_policy_receipt_bound
            else None
        ),
        "active_policy_activation_receipt": (
            projected_marnie_policy.get("activation_receipt")
            if active_policy_receipt_bound
            else None
        ),
        "latest_attempted_version": latest_attempted_core[0] or None,
        "latest_attempted_status": latest_attempted_core[1] or None,
        "attempted_statuses": {
            str(version): status
            for version, status in sorted(
                {
                    version: status
                    for version, status in attempted_core_versions
                    if version and status
                }.items()
            )
        },
        "matchup_adapter_format_version": (
            int(projected_matchup_adapter.get("checkpoint_format_version") or 0)
            or None
        ),
        "matchup_adapter_display_name": (
            projected_matchup_adapter.get("display_name")
            or "Matchup Router Format 6"
        ),
        "source": (
            "current_goal_requirements_owner_projection"
            if projected_core
            else "immutable_core_receipts"
            if immutable_core_receipts_found
            else "specialist_state"
        ),
    }
    raw_canonical_priority = [
        canonical_active_id,
        *strict_prefix_ids,
        *[
            str(value)
            for value in (
                (priority.get("handoff_override") or {}).get(
                    "priority_prefix"
                )
                or []
            )
            if str(value)
        ],
        *[
            str(value)
            for value in (
                priority.get("ordered_unfinished_ids_after_active") or []
            )
            if str(value)
        ],
    ]
    canonical_priority: list[str] = []
    seen_priority_ids: set[str] = set()
    for specialist_id in raw_canonical_priority:
        if specialist_id and specialist_id not in seen_priority_ids:
            canonical_priority.append(specialist_id)
            seen_priority_ids.add(specialist_id)
    ordered_priority = [
        specialist_id
        for specialist_id in canonical_priority
        if specialist_id != canonical_active_id
        and specialist_id not in owner_removed_ids
        and specialist_id in ids
        and str(
            next(
                row["status"]
                for row in rows
                if str(row["id"]) == specialist_id
            )
        )
        not in {"passed_frozen", "population_training", "failed_experiment"}
    ]
    raw_priority_rows = priority.get("rows") or []
    if not isinstance(raw_priority_rows, list):
        raw_priority_rows = []
    raw_priority_by_id = {
        str(row.get("id") or ""): dict(row)
        for row in raw_priority_rows
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    if "thwackey" not in raw_priority_by_id and "festival-lead" in raw_priority_by_id:
        raw_priority_by_id["thwackey"] = {
            **raw_priority_by_id["festival-lead"],
            "id": "thwackey",
        }
    raw_priority_rows = [
        raw_priority_by_id.get(
            specialist_id,
            {
                "id": specialist_id,
                "share": None,
                "source_archetype": None,
                "mapping": "canonical_roster",
                "existing_model_artifact": False,
                "availability_group": "missing_model",
            },
        )
        for specialist_id in ordered_priority
    ]
    priority_by_id: dict[str, dict[str, Any]] = {}
    for rank, raw in enumerate(raw_priority_rows, start=1):
        if not isinstance(raw, dict):
            continue
        specialist_id = str(raw.get("id") or "")
        share = raw.get("share")
        if specialist_id:
            priority_by_id[specialist_id] = {
                "rank_after_active": rank,
                "meta_share": float(share) if share is not None else None,
                "source_archetype": raw.get("source_archetype"),
                "mapping": raw.get("mapping"),
                "existing_model_artifact": raw.get("existing_model_artifact") is True,
                "availability_group": raw.get("availability_group"),
                "priority_override": raw.get("priority_override"),
            }
    expected_priority_ids = {
        str(row["id"])
        for row in rows
        if str(row["id"]) != canonical_active_id
        and str(row["id"]) not in owner_removed_ids
        and row["status"]
        not in {"passed_frozen", "population_training", "failed_experiment"}
    }
    if (
        len(ordered_priority) != len(set(ordered_priority))
        or set(ordered_priority) != expected_priority_ids
        or set(priority_by_id) != expected_priority_ids
        or ordered_priority != [str(row.get("id") or "") for row in raw_priority_rows]
    ):
        return {
            "available": False,
            "reason": "canonical specialist priority invariants failed",
            "source": str(source),
        }
    for row in rows:
        row.update(priority_by_id.get(str(row["id"]), {}))
    verified = payload.get("last_verified_at_utc")
    if hasattr(verified, "isoformat"):
        verified = verified.isoformat()
    priority_source = dict(priority.get("source") or {})
    for key, value in list(priority_source.items()):
        # PyYAML intentionally decodes unquoted ISO-8601 scalars as datetime
        # objects.  The dashboard API is JSON, so canonical tracker timestamps
        # must be normalized before they reach the top-level encoder.
        if hasattr(value, "isoformat"):
            priority_source[key] = value.isoformat()
    head_template = heads_and_datasets.get("specialist_head_template")
    head_template = head_template if isinstance(head_template, dict) else {}
    matchup_routing = head_template.get("matchup_routing")
    matchup_routing = matchup_routing if isinstance(matchup_routing, dict) else {}
    trevenant_pointer = (
        EXPERT20_ROOT / "hops-trevenant-v2/PROTECTED_EXPERT_CORPUS.json"
    )
    trevenant = read_json(trevenant_pointer)
    core_pointer = EXPERT20_ROOT / "core-balanced-v2/PROTECTED_CORE_CORPUS.json"
    core_preparation = read_json(core_pointer)
    core_service = unit_state(
        "pokebot-balanced-core-corpus-v2.service", user=True
    )
    handler_state_path = Path(
        str(
            active_run.get("gate_handler_state")
            or (
                "/home/inzi/poke-bot-agent/outputs/state/"
                "alakazam-protocol-passed-gate-handler-v31.json"
            )
        )
    )
    handler_service_name = str(
        active_run.get("gate_handler_service")
        or "pokebot-passed-gate-handler.service"
    )
    handler_state = read_json(handler_state_path)
    handler_service = unit_state(handler_service_name, user=True)
    active_specialist_service_name = str(active_run.get("service") or "")
    active_specialist_service = (
        unit_state(active_specialist_service_name, user=True)
        if active_specialist_service_name
        else {}
    )
    submission_queue = read_json(
        Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "kaggle-submission-queue.json"
        )
    )
    pending_copies = sum(
        1
        for row in (submission_queue.get("queue") or [])
        if isinstance(row, dict) and row.get("queue_status") == "pending"
    )
    owner_handoff = owner_specialist_handoff_state()
    owner_handoff_relevant = bool(
        owner_handoff.get("active") or active_id == "hops-trevenant"
    )
    selector_env_path = Path("/home/inzi/.config/pokebot/specialist_runtime.env")
    selector_values: dict[str, str] = {}
    try:
        for line in selector_env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            selector_values[key.strip()] = value.strip().strip("'\"")
    except OSError:
        selector_values = {}
    selected_runtime_root = Path(
        selector_values.get("POKEBOT_SPECIALIST_RUNTIME_ROOT") or ""
    )
    selected_runtime_registry = (
        selected_runtime_root / "ops/specialist_runtime_registry_v1.json"
    )
    switch_registry_source = (
        selected_runtime_registry
        if selected_runtime_registry.is_file()
        else SPECIALIST_RUNTIME_REGISTRY
    )
    switch_registry = read_json(switch_registry_source)
    switch_rows = (
        switch_registry.get("specialists")
        if isinstance(switch_registry.get("specialists"), dict)
        else {}
    )
    ready_switch_ids = sorted(
        str(specialist_id)
        for specialist_id, row in switch_rows.items()
        if isinstance(row, dict) and row.get("status") == "ready"
    )
    blocked_switches = {
        str(specialist_id): str(row.get("reason") or row.get("status") or "")
        for specialist_id, row in switch_rows.items()
        if isinstance(row, dict) and row.get("status") != "ready"
    }
    runtime_frozen_registry = (
        Path(str(switch_registry.get("runtime_root") or ""))
        / str(
            switch_registry.get("frozen_specialist_registry")
            or "ops/frozen_specialist_registry_v1.json"
        )
    )
    frozen_registry_source = (
        runtime_frozen_registry
        if runtime_frozen_registry.is_file()
        else FROZEN_SPECIALIST_REGISTRY
    )
    frozen_registry = read_json(frozen_registry_source)
    frozen_runtime_rows: list[dict[str, Any]] = []
    for raw in frozen_registry.get("specialists") or []:
        if not isinstance(raw, dict):
            continue
        group = str(raw.get("baseline_group") or "specialists")
        baseline_dir = str(raw.get("baseline_dir") or "")
        package = ROOT / "baselines" / group / baseline_dir
        tree = read_json(package / "matchup_tree.json")
        runtime = dict(tree.get("runtime_contract") or {})
        frozen_runtime_rows.append(
            {
                "specialist_id": str(raw.get("specialist_id") or ""),
                "opponent_id": str(raw.get("opponent_id") or ""),
                "archetype_id": str(raw.get("archetype_id") or ""),
                "checkpoint_checksum": str(
                    raw.get("checkpoint_digest") or ""
                ),
                "deck_present": (package / "deck.csv").is_file(),
                "model_present": (package / "model.pt").is_file(),
                "matchup_tree_present": (package / "matchup_tree.json").is_file(),
                "matchup_runtime_enabled": (
                    tree.get("runtime_enabled") is True
                    and runtime.get("one_route_per_decision") is True
                    and runtime.get("unknown_route_exact_bypass") is True
                ),
                "inference_only": raw.get("frozen") is True,
            }
        )
    next_selection: dict[str, Any] = {}
    try:
        completed_ids = {
            str(row.get("specialist_id") or "")
            for row in (frozen_registry.get("specialists") or [])
            if isinstance(row, dict) and row.get("frozen") is True
        }
        active_tree_path = Path(
            str(
                (
                    switch_rows.get(active_id)
                    if isinstance(switch_rows.get(active_id), dict)
                    else {}
                ).get("matchup_runtime_tree")
                or "/nonexistent"
            )
        )
        active_tree = read_json(active_tree_path)
        routable_ids = {
            str(value)
            for value in (active_tree.get("runtime_contract") or {}).get(
                "accepted_archetype_ids", ()
            )
        }
        eligible: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        corpus_root = EXPERT20_ROOT / "specialist-corpora-v1"
        for rank, specialist_id in enumerate(ordered_priority):
            if specialist_id in completed_ids or specialist_id == active_id:
                continue
            if specialist_id not in routable_ids:
                deferred.append(
                    {
                        "specialist_id": specialist_id,
                        "priority_rank": rank,
                        "reason": "validated_causal_runtime_route_missing",
                    }
                )
                continue
            pointer = (
                corpus_root
                / specialist_id
                / "PROTECTED_EXPERT_CORPUS.json"
            )
            corpus = read_json(pointer)
            decisions = int(
                ((corpus.get("totals") or {}).get("decisions_kept")) or 0
            )
            if (
                corpus.get("schema") != "poke_bot.pinned_expert_corpus/v1"
                or corpus.get("protected") is not True
                or decisions < 20_000
            ):
                deferred.append(
                    {
                        "specialist_id": specialist_id,
                        "priority_rank": rank,
                        "reason": "protected_expert_corpus_below_contract",
                        "decisions": decisions,
                        "minimum_decisions": 20_000,
                    }
                )
                continue
            eligible.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "decisions": decisions,
                    "pointer": str(pointer),
                }
            )
        if eligible:
            next_selection = {
                "selected": eligible[0],
                "eligible_in_priority_order": eligible,
                "deferred_higher_priority": [
                    row
                    for row in deferred
                    if int(row["priority_rank"])
                    < int(eligible[0]["priority_rank"])
                ],
            }
    except (OSError, RuntimeError, TypeError, ValueError):
        next_selection = {}
    selected_next = dict(next_selection.get("selected") or {})
    prestage_state = read_json(NEXT_SPECIALIST_PRESTAGE_STATE)
    prestage_available = prestage_receipt_is_current(
        prestage_state,
        active_id,
    )
    if prestage_available:
        staged_selection = dict(prestage_state.get("selection") or {})
        selected_next = dict(staged_selection.get("selected") or {})
        next_selection = staged_selection
    canonical_prestage = canonical_next_prestage_overlay(payload)
    v6_handoff_blocked = (
        canonical_prestage.get("blocks_v6_handoff") is True
    )
    legacy_v5_selected = dict(selected_next)
    if v6_handoff_blocked:
        selected_next = {
            "specialist_id": canonical_prestage.get(
                "intended_specialist"
            )
        }
    selected_pointer = Path(
        str(legacy_v5_selected.get("pointer") or "/nonexistent")
    )
    selected_corpus = read_json(selected_pointer)
    frozen_program_ids = sorted(
        {
            str(row.get("specialist_id") or "")
            for row in (frozen_registry.get("specialists") or [])
            if isinstance(row, dict)
            and row.get("frozen") is True
            and str(row.get("specialist_id") or "")
        }
    )
    # The immutable registry may overlay only specialists present in the
    # currently loaded stable roster.  The roster cardinality is intentionally
    # dynamic; tying this evidence to an old literal count made every completed
    # specialist disappear when the canonical roster changed.
    if not set(frozen_program_ids).issubset(set(ids)):
        frozen_program_ids = []
    starmie_handler = read_json(STARMIE_PASSED_GATE_HANDLER_STATE)
    starmie_frozen = dict(starmie_handler.get("frozen_model") or {})
    starmie_model_path = Path(
        str(starmie_frozen.get("model_path") or "/nonexistent")
    )
    if (
        "starmie" in ids
        and str(starmie_handler.get("phase") or "")
        in {"model_frozen", "submission_queued", "complete_handoff_started"}
        and _is_sha256_digest(starmie_frozen.get("checkpoint_digest"))
        and starmie_model_path.is_file()
    ):
        frozen_program_ids = sorted({*frozen_program_ids, "starmie"})
    rows, status_counts, program_active_id = reconcile_frozen_specialist_rows(
        rows,
        set(frozen_program_ids),
        active_id,
    )
    runtime_service_active = runtime_service_state.startswith(
        ("active", "activating")
    )
    selected_runtime_refresh = bool(
        runtime_specialist_id
        and runtime_run_name.startswith("final_format_")
    )
    active_runtime_refresh = bool(
        runtime_service_active
        and selected_runtime_refresh
    )
    # A post-fleet refresh reuses an archetype whose historical specialist row
    # is already frozen.  Keep that immutable row frozen, but do not erase the
    # separately versioned live refresh from the operator-facing active card.
    display_active_id = (
        program_active_id
        or (runtime_specialist_id if selected_runtime_refresh else "")
    )
    planning_frozen_ids = [
        specialist_id
        for specialist_id in frozen_program_ids
        if specialist_id not in owner_removed_ids
    ]
    terminal_exception_ids = {
        str(value)
        for value in program_progress.get(
            "terminal_failed_experiment_specialist_ids", ()
        )
        if str(value)
    }
    terminal_exception_ids &= {
        str(row["id"])
        for row in rows
        if str(row["status"]) == "failed_experiment"
    }
    terminal_disposition_count = len(terminal_exception_ids)
    live_program_progress = {
        **program_progress,
        "required_specialists_total": planning_required_count,
        "completed_frozen": len(planning_frozen_ids),
        "completed_specialist_ids": planning_frozen_ids,
        "active_specialists": int(bool(program_active_id)),
        "active_specialist_ids": (
            [program_active_id] if program_active_id else []
        ),
        "remaining_unfinished": max(
            0,
            planning_required_count
            - len(planning_frozen_ids)
            - terminal_disposition_count,
        ),
        "remaining_after_active": max(
            0,
            planning_required_count
            - len(planning_frozen_ids)
            - terminal_disposition_count
            - int(bool(program_active_id)),
        ),
        "population_transition_ready": (
            len(planning_frozen_ids) == planning_required_count
        ),
    }
    unfinished_including_active = max(
        0,
        planning_required_count
        - len(planning_frozen_ids)
        - terminal_disposition_count,
    )
    status_by_id = {
        str(row["id"]): str(row["status"])
        for row in rows
    }
    live_priority_source = [*canonical_priority, *ids]
    live_ordered_priority: list[str] = []
    for specialist_id in live_priority_source:
        if (
            specialist_id
            and specialist_id != program_active_id
            and specialist_id not in owner_removed_ids
            and status_by_id.get(specialist_id)
            not in {"passed_frozen", "population_training", "failed_experiment"}
            and specialist_id not in live_ordered_priority
        ):
            live_ordered_priority.append(specialist_id)
    live_rank_by_id = {
        specialist_id: rank
        for rank, specialist_id in enumerate(
            live_ordered_priority, start=1
        )
    }
    for row in rows:
        row["rank_after_active"] = live_rank_by_id.get(str(row["id"]))
    effective_next_action = current.get("next_action")
    if live_execution.get("available") is True and display_active_id:
        active_label = display_active_id.replace("-", " ").title()
        if active_runtime_refresh and display_active_id == "alakazam":
            terminal_iteration = int(
                (post_fleet_refresh or {}).get(
                    "terminal_ceiling_completed_iteration", 20
                )
            )
            effective_next_action = (
                f"Continue live {active_label} iteration "
                f"{int(live_execution['next_iteration'])} through the exact "
                f"iteration-{terminal_iteration} refresh boundary. Freeze and "
                f"register exact iteration {terminal_iteration}: retain a "
                "measured pass when every gate passes, otherwise record owner "
                "ceiling acceptance while preserving the failed gate evidence. "
                f"Do not collect iteration {terminal_iteration + 1}. Then run "
                "the post-Alakazam core refresh attempt and launch the staged "
                "H10-I Fusion-v3 Marnie's Grimmsnarl refresh. Population "
                "training remains blocked until both refreshes are truthfully "
                "complete, frozen, and registered."
            )
        elif active_runtime_refresh and display_active_id == "marnie-s-grimmsnarl-ex":
            terminal_iteration = int(
                (post_fleet_refresh or {}).get(
                    "terminal_ceiling_completed_iteration", 20
                )
            )
            effective_next_action = (
                f"Continue live {active_label} iteration "
                f"{int(live_execution['next_iteration'])} through the exact "
                f"iteration-{terminal_iteration} refresh boundary. Freeze and "
                f"register exact iteration {terminal_iteration}: retain a "
                "measured pass when every gate passes, otherwise record owner "
                "ceiling acceptance while preserving the failed gate evidence. "
                f"Do not collect iteration {terminal_iteration + 1}. Then launch "
                "the staged new H10 Crustle specialist; its public package remains "
                "an inference-only baseline. Population training remains blocked "
                "until the Marnie refresh and new Crustle specialist are both "
                "truthfully complete, frozen, and registered."
            )
        else:
            effective_next_action = (
                f"Continue live {active_label} iteration "
                f"{int(live_execution['next_iteration'])} through its exact "
                f"training and gate contract. {unfinished_including_active} "
                "specialists remain unfinished including the active specialist; "
                f"{live_program_progress['remaining_after_active']} remain after it. "
                "Freeze and register it only after both gates pass, then begin the "
                "next unfinished specialist. Population training remains blocked "
                f"until all {planning_required_count} specialists are frozen."
            )
    elif runtime_identity_reconciled and display_active_id:
        # A selector may commit the next specialist before that run has
        # published its first immutable iteration receipt. The selector is
        # still authoritative for identity, so never leave the previous
        # specialist's planning action on screen during this brief boundary.
        active_label = display_active_id.replace("-", " ").title()
        effective_next_action = (
            f"Continue live {active_label} specialist under the selected "
            "runtime contract while its first immutable execution receipt is "
            f"pending. {unfinished_including_active} specialists remain "
            "unfinished including the active specialist; preserve all frozen "
            "predecessors and begin normal receipt-backed iteration tracking "
            "as soon as it appears. Population training remains blocked until "
            f"all {planning_required_count} specialists are frozen."
        )
    population_runtime: dict[str, Any] = {}
    if POPULATION_ROUND_ROBIN_STATE.is_file():
        try:
            candidate = json.loads(
                POPULATION_ROUND_ROBIN_STATE.read_text(encoding="utf-8")
            )
            members = candidate.get("members") if isinstance(candidate, dict) else None
            if (
                isinstance(candidate, dict)
                and candidate.get("schema")
                == "poke_bot.population_round_robin_state/v1"
                and int(candidate.get("member_count") or 0) == expected_count
                and isinstance(members, list)
                and len(members) == expected_count
            ):
                unit = unit_state(POPULATION_ROUND_ROBIN_SERVICE, user=True)
                population_runtime = {
                    "available": True,
                    "service": unit,
                    "status": candidate.get("status"),
                    "population_cycle": candidate.get("population_cycle"),
                    "active_member_index": candidate.get(
                        "active_member_index"
                    ),
                    "active_specialist_id": candidate.get(
                        "active_specialist_id"
                    ),
                    "completed_member_cycles": sum(
                        int(row.get("cycles_completed") or 0)
                        for row in members
                        if isinstance(row, dict)
                    ),
                    "rl_epochs_completed": sum(
                        int(row.get("rl_epochs_completed") or 0)
                        for row in members
                        if isinstance(row, dict)
                    ),
                    "rehearsal_epochs_completed": sum(
                        int(row.get("rehearsal_epochs_completed") or 0)
                        for row in members
                        if isinstance(row, dict)
                    ),
                    "source": str(POPULATION_ROUND_ROBIN_STATE),
                    "updated_at": POPULATION_ROUND_ROBIN_STATE.stat().st_mtime,
                }
        except (OSError, UnicodeError, ValueError, TypeError):
            population_runtime = {}
    dashboard_rows = [
        row for row in rows if str(row["id"]) not in owner_removed_ids
    ]
    dashboard_status_counts: dict[str, int] = {}
    for row in dashboard_rows:
        status = str(row["status"])
        dashboard_status_counts[status] = (
            dashboard_status_counts.get(status, 0) + 1
        )
    return {
        "available": True,
        "schema_version": payload.get("schema_version"),
        "protocol_schema_version": payload.get("protocol_schema_version"),
        "last_verified_at_utc": str(verified or "") or None,
        "phase": current.get("phase"),
        "active_specialist": display_active_id,
        "active_runtime_refresh": {
            "active": active_runtime_refresh,
            "specialist_id": (
                runtime_specialist_id if selected_runtime_refresh else None
            ),
            "run_name": runtime_run_name if selected_runtime_refresh else None,
            "service_state": (
                runtime_service_state if selected_runtime_refresh else None
            ),
            "historical_specialist_row_remains_frozen": True,
            "policy_scope": "refresh_lineage_not_cumulative_core_generation",
        },
        "canonical_active_specialist": canonical_active_id,
        "canonical_active_refresh_specialist": canonical_refresh_id or None,
        "runtime_active_specialist": runtime_specialist_id or None,
        "runtime_run_name": runtime_run_name or None,
        "runtime_service_state": runtime_service_state or None,
        "runtime_reconciled": runtime_reconciled,
        "runtime_identity_reconciled": runtime_identity_reconciled,
        "runtime_execution_reconciled": runtime_execution_reconciled,
        # A newer immutable iteration commit makes planning counters stale,
        # not the canonical specialist pointer.  Only an actual selected
        # specialist identity mismatch invalidates that pointer.
        "canonical_pointer_stale": canonical_pointer_stale,
        "accuracy_warning": (
            (
                "Selected production service supersedes the stale canonical "
                "active-specialist identity; immutable completed-specialist "
                "history is preserved."
                if canonical_pointer_stale
                else (
                    "Selected production runtime supersedes stale active-run "
                    "metadata in the canonical planning snapshot."
                    if runtime_identity_reconciled
                    else "Live immutable commits supersede stale execution "
                    "counters in the canonical planning snapshot."
                )
            )
            if runtime_reconciled
            else None
        ),
        "next_action": effective_next_action,
        "program_progress": live_program_progress,
        "live_execution": live_execution,
        "shared_core_status": shared_core.get("status"),
        "shared_core_checkpoint": shared_core.get("checkpoint"),
        "core_generation": core_generation,
        "current_deck_guide_weight_policy": (
            current_deck_guide_weight_policy
        ),
        "current_deck_guide_training_modes": (
            current_deck_guide_training_modes
        ),
        "final_format_milestone_submissions": (
            projected_marnie_milestones
        ),
        "head_requirements": {
            "archetype_policy": head_template.get(
                "archetype_policy_head_required"
            )
            is True,
            "game_plan_policies": head_template.get(
                "game_plan_policy_heads_required"
            )
            is True,
            "matchup_policies": head_template.get(
                "matchup_policy_heads_required"
            )
            is True,
            "causal_observable_state_only": matchup_routing.get(
                "causal_observable_state_only"
            )
            is True,
            "opponent_package_identity_allowed": matchup_routing.get(
                "opponent_package_identity_allowed"
            )
            is True,
            "relevant_matchup_sequences_only": matchup_routing.get(
                "relevant_matchup_sequences_only"
            )
            is True,
            "staged_router_candidate": dict(
                matchup_routing.get("staged_router_candidate") or {}
            ),
        },
        "required_target_count": planning_required_count,
        "status_counts": dashboard_status_counts,
        "specialists": dashboard_rows,
        "retained_non_specialist_opponents": (
            retained_non_specialist_opponents
        ),
        "training_priority": {
            "policy": priority.get("policy"),
            "ordered_unfinished_ids_after_active": live_ordered_priority,
            "next_specialist": (
                live_ordered_priority[0]
                if live_ordered_priority
                else None
            ),
            "next_executable_specialist": (
                None
                if v6_handoff_blocked
                else selected_next.get("specialist_id")
            ),
            "next_executable_decisions": (
                None if v6_handoff_blocked else selected_next.get("decisions")
            ),
            "deferred_higher_priority": next_selection.get(
                "deferred_higher_priority", []
            ),
            "primary_sort": "missing specialist model first",
            "handoff_override": dict(priority.get("handoff_override") or {}),
            "strict_post_spidops_prefix": strict_prefix_contract,
            "owner_removal": owner_removal_contract,
            "staged_v5_transition": dict(
                priority.get("staged_v5_transition") or {}
            ),
            "source": priority_source,
        },
        "switching": {
            "available": (
                switch_registry.get("schema")
                == "poke_bot.specialist_runtime_registry/v1"
            ),
            "selector_environment_variable": switch_registry.get(
                "selector_environment_variable"
            ),
            "configured_active_specialist": active_id,
            "registry_version": switch_registry.get("version"),
            "runtime_root": switch_registry.get("runtime_root"),
            "runtime_build": Path(
                str(switch_registry.get("runtime_root") or "")
            ).name,
            "ready_specialist_ids": ready_switch_ids,
            "blocked_specialists": blocked_switches,
            "one_variable_dispatch_ready": active_id in ready_switch_ids,
            "source": str(switch_registry_source),
        },
        "frozen_inference_opponents": frozen_runtime_rows,
        "preparation": {
            "next_specialist": selected_next.get("specialist_id"),
            "prestage_available": bool(
                prestage_available or canonical_prestage
            ),
            "prestage_status": (
                canonical_prestage.get("status")
                if v6_handoff_blocked
                else prestage_state.get("status")
                if prestage_available
                else None
            ),
            "prestage_blockers": (
                [canonical_prestage.get("blocker")]
                if v6_handoff_blocked
                and canonical_prestage.get("blocker")
                else list(prestage_state.get("blockers") or ())
                if prestage_available
                else []
            ),
            "prestage_representative_ready": (
                canonical_prestage.get("representative_ready") is True
                if v6_handoff_blocked
                else (prestage_state.get("representative") or {}).get("ready")
                is True
                if prestage_available
                else False
            ),
            "prestage_cpu_pack_status": (
                canonical_prestage.get("cpu_pack_status")
                if v6_handoff_blocked
                else (prestage_state.get("cpu_pack") or {}).get("status")
                if prestage_available
                else None
            ),
            "prestage_cpu_pack_bytes": (
                int((prestage_state.get("cpu_pack") or {}).get("bytes") or 0)
                if prestage_available
                else 0
            ),
            "prestage_receipt": (
                str(NEXT_SPECIALIST_PRESTAGE_STATE)
                if prestage_available
                else None
            ),
            "expert_corpus_ready": bool(
                canonical_prestage.get("expert_corpus_ready") is True
                if v6_handoff_blocked
                else selected_pointer.is_file()
            ),
            "expert_corpus_pointer": (
                canonical_prestage.get("expert_corpus_pointer")
                if v6_handoff_blocked
                else str(selected_pointer)
                if selected_pointer.is_file()
                else None
            ),
            "expert_records": int(
                canonical_prestage.get("expert_records")
                if v6_handoff_blocked
                else ((selected_corpus.get("totals") or {}).get("records_kept"))
                or 0
            ),
            "expert_decisions": int(
                canonical_prestage.get("expert_decisions")
                if v6_handoff_blocked
                else ((selected_corpus.get("totals") or {}).get("decisions_kept"))
                or 0
            ),
            "required_expert_corpus": (
                "selected_checksum_pinned_corpus"
                if v6_handoff_blocked
                else "selected_checksum_pinned_corpus"
            ),
            "legacy_v5_prestage": {
                "available": prestage_available,
                "status": (
                    prestage_state.get("status")
                    if prestage_available
                    else None
                ),
                "specialist_id": legacy_v5_selected.get("specialist_id"),
                "pointer": (
                    str(selected_pointer)
                    if selected_pointer.is_file()
                    else None
                ),
                "decisions": int(
                    (
                        (selected_corpus.get("totals") or {}).get(
                            "decisions_kept"
                        )
                    )
                    or 0
                ),
                "v6_bootstrap_eligible": not v6_handoff_blocked,
            },
            "balanced_core_corpus_ready": core_pointer.is_file(),
            "balanced_core_builder_active": bool(core_service.get("active")),
            "balanced_core_builder_pid": core_service.get("pid"),
            "gate_handler_active": bool(handler_service.get("active")),
            "gate_handler_phase": (
                handler_state.get("phase")
                if handler_service.get("active")
                else "retired"
            ),
            "gate_handler_service": handler_service_name,
            "gate_handler_state": str(handler_state_path),
            "gate_handler_marker": active_run.get("terminal_gate_marker"),
            "gate_handler_minimum_completed_iteration": active_run.get(
                "minimum_terminal_iteration"
            ),
            "gate_handler_source": "current.active_run",
            "terminal_protocol_active": bool(
                active_specialist_service.get("active")
            ),
            "terminal_active_gate_id": active_run.get(
                "terminal_active_gate_id"
            ),
            "terminal_iteration_ceiling": active_run.get("iteration_ceiling"),
            "active_specialist_service": active_specialist_service_name or None,
            "current_premium_gate_games": active_run.get("premium_gate_games"),
            "current_official_research_games": active_run.get(
                "official_research_games"
            ),
            "current_total_evaluation_games": active_run.get(
                "total_evaluation_games"
            ),
            "pending_kaggle_copies": pending_copies,
            "submission_queue_source": (
                "/home/inzi/poke-bot-agent/outputs/state/"
                "kaggle-submission-queue.json"
            ),
            "core_pointer_schema": core_preparation.get("schema"),
            "owner_handoff_active": (
                owner_handoff_relevant
                and owner_handoff.get("active") is True
            ),
            "owner_handoff_phase": (
                owner_handoff.get("phase") if owner_handoff_relevant else None
            ),
            "owner_handoff_stage": (
                owner_handoff.get("stage") if owner_handoff_relevant else None
            ),
            "owner_handoff_percent": (
                owner_handoff.get("percent") if owner_handoff_relevant else None
            ),
            "owner_handoff_current": (
                owner_handoff.get("current") if owner_handoff_relevant else None
            ),
            "owner_handoff_total": (
                owner_handoff.get("total") if owner_handoff_relevant else None
            ),
            "owner_handoff_latest_line": (
                owner_handoff.get("latest_line")
                if owner_handoff_relevant
                else None
            ),
        },
        "population_training": {
            "status": (
                population_runtime.get("status")
                if population_runtime.get("available")
                else population.get("status")
            ),
            "enabled": (
                bool(
                    (
                        population_runtime.get("service") or {}
                    ).get("active")
                )
                if population_runtime.get("available")
                else population.get("enabled") is True
            ),
            "all_required_specialists_passed": (
                True
                if population_runtime.get("available")
                else population.get("all_required_specialists_passed")
                is True
            ),
            "runtime": population_runtime,
        },
        "post_fleet_refresh": (
            dict(post_fleet_refresh)
            if isinstance(post_fleet_refresh, dict)
            else {}
        ),
        "unresolved_count": len(payload.get("unresolved_facts") or []),
        "source": str(source),
        "updated_at": source.stat().st_mtime,
    }


def annotate_gpu_production_assignments(
    gpus: list[dict[str, Any]],
    curriculum: dict[str, Any],
    specialist_handoff: dict[str, Any],
) -> None:
    """Bind GPU labels to the active managed trainer's effective topology."""

    worker = (
        curriculum.get("worker")
        if isinstance(curriculum.get("worker"), dict)
        else {}
    )
    curriculum_active = curriculum.get("active") is True
    topology_source = str(
        worker.get("topology_source")
        or "active managed trainer effective environment"
    )
    gpu0_replicas = int(worker.get("leaf_gpu0_replicas") or 0)
    gpu1_replicas = int(worker.get("leaf_gpu1_replicas") or 0)
    for gpu in gpus:
        index = int(gpu.get("index") or 0)
        if index == 0 and curriculum_active:
            gpu["production_active"] = gpu0_replicas > 0
            gpu["assignment"] = (
                f"PRODUCTION · {gpu0_replicas} policy leaf replicas"
                if gpu0_replicas > 0
                else "OUT OF FLEET · no active trainer leaf replicas"
            )
            gpu["assignment_source"] = topology_source
            gpu["production_leaf_replicas"] = gpu0_replicas
        elif index == 1 and (
            curriculum_active or specialist_handoff.get("active")
        ):
            gpu["production_active"] = True
            gpu["assignment"] = (
                "PRODUCTION · core/specialist trainer"
                if specialist_handoff.get("active")
                else "PRODUCTION · policy leaves + trainer"
            )
            gpu["assignment_source"] = topology_source
            gpu["production_leaf_replicas"] = gpu1_replicas


def main() -> None:
    # Elmo is an independent host. Fetch its three views concurrently so a
    # slow SSH handshake cannot serialize into the outer Bert→Inzi timeout.
    with ThreadPoolExecutor(max_workers=5) as remote_pool:
        elmo_future = remote_pool.submit(elmo_state)
        latest10_future = remote_pool.submit(latest10_state)
        expert_refresh_future = remote_pool.submit(expert_refresh_state)
        matchup_pipeline_future = remote_pool.submit(matchup_pipeline_state)
        guide_prestage_future = remote_pool.submit(
            current_deck_guide_prestage_state
        )
        system = system_state()
        service = service_state()
        transition = transition_state()
        curriculum = curriculum_state()
        final_alakazam = final_format_alakazam_progress()
        final_marnie = final_format_marnie_progress()
        final_crustle = final_format_crustle_progress()
        postupload_bootstrap = marnie_postupload_bootstrap_state()
        postupload_family = marnie_postupload_family_study_state()
        postupload_boundary = (
            postupload_bootstrap
            if postupload_bootstrap.get("current") is True
            else postupload_family
        )
        active_final_refresh = (
            final_crustle
            if final_crustle.get("status") == "running"
            else (
                final_marnie
                if final_marnie.get("status")
                in {"running", "complete", "stopped"}
                else final_alakazam
            )
        )
        final_alakazam_models = final_format_alakazam_model_inventory()
        if active_final_refresh.get("status") in {
            "running",
            "complete",
            "stopped",
        }:
            final_service = active_final_refresh.get("service") or {}
            final_stage = (
                active_final_refresh.get("phase")
                or "train:ordinary_alakazam_refresh"
            )
            final_run = (
                active_final_refresh.get("run")
                or "final_format_alakazam_r79"
            )
            final_service_name = (
                final_service.get("name")
                or FINAL_FORMAT_ALAKAZAM_SERVICE
            )
            if active_final_refresh.get("status") in {"running", "stopped"}:
                service = final_service
            curriculum = {
                **curriculum,
                "active": active_final_refresh.get("status") == "running",
                "active_units": (
                    [final_service_name]
                    if active_final_refresh.get("status") == "running"
                    else []
                ),
                "active_pids": (
                    [int(final_service.get("pid") or 0)]
                    if int(final_service.get("pid") or 0) > 0
                    else []
                ),
                "run": final_run,
                "iteration": active_final_refresh.get(
                    "iteration", active_final_refresh.get("epoch")
                ),
                "stage": final_stage,
                "progress": {
                    "line": active_final_refresh.get("latest_line"),
                    "stage": final_stage,
                    "iteration": active_final_refresh.get("iteration"),
                    "epoch": active_final_refresh.get("epoch"),
                    "current": active_final_refresh.get("current"),
                    "total": active_final_refresh.get("total"),
                    "percent": active_final_refresh.get("percent"),
                    "rate": active_final_refresh.get("rate"),
                    "rate_unit": active_final_refresh.get("rate_unit"),
                    "gps": active_final_refresh.get("games_per_second"),
                    "sps": active_final_refresh.get("samples_per_second"),
                    "remotes": active_final_refresh.get("remote_workers"),
                    "metrics": active_final_refresh.get("metrics") or {},
                },
                "progress_source": active_final_refresh.get("source"),
                "progress_status_source": active_final_refresh.get("source"),
                "progress_log_source": active_final_refresh.get("log"),
                "progress_updated_at": active_final_refresh.get("updated_at"),
                "source_current": bool(active_final_refresh.get("fresh")),
                "remote_workers": active_final_refresh.get("remote_workers"),
                "remote_endpoints": active_final_refresh.get("remote_endpoints") or [],
                "scheduler_queues": (
                    active_final_refresh.get("scheduler_queues")
                    or {"available": False, "mode": "waiting"}
                ),
                "worker": {
                    **(curriculum.get("worker") or {}),
                    "active": active_final_refresh.get("status") == "running",
                    "rss_bytes": final_service.get("memory_bytes"),
                    "source": "systemd-user-cgroup",
                    "command": final_service.get("command"),
                },
                "final_format_refresh": active_final_refresh,
            }
        if postupload_boundary.get("current") is True:
            boundary_service = dict(postupload_boundary.get("service") or {})
            boundary_progress = dict(postupload_boundary.get("progress") or {})
            boundary_active = postupload_boundary.get("active") is True
            boundary_stage = str(
                postupload_boundary.get("phase") or "family-shadow:starting"
            )
            curriculum = {
                **curriculum,
                "active": boundary_active,
                "active_units": (
                    [
                        str(
                            boundary_service.get("name")
                            or MARNIE_POSTUPLOAD_FAMILY_STUDY_SERVICE
                        )
                    ]
                    if boundary_active
                    else []
                ),
                "active_pids": (
                    [int(boundary_service.get("pid") or 0)]
                    if boundary_active
                    else []
                ),
                "run": postupload_boundary.get("run"),
                "iteration": 9,
                "stage": boundary_stage,
                "progress": boundary_progress,
                "progress_source": postupload_boundary.get("source"),
                "progress_status_source": postupload_boundary.get("source"),
                "progress_log_source": postupload_boundary.get("source"),
                "progress_updated_at": postupload_boundary.get("updated_at"),
                "source_current": True,
                "remote_workers": 0,
                "remote_endpoints": [],
                "scheduler_queues": {
                    "available": False,
                    "mode": (
                        "isolated_shadow_study"
                        if boundary_active
                        else "protocol_pause"
                    ),
                },
                "worker": {
                    **(curriculum.get("worker") or {}),
                    "active": boundary_active,
                    "rss_bytes": boundary_service.get("memory_bytes"),
                    "source": "systemd-user-cgroup",
                    "command": boundary_service.get("command"),
                },
                "managed_boundary": postupload_boundary,
            }
        gpus = gpu_state()
        elmo = elmo_future.result()
        latest10 = latest10_future.result()
        expert_refresh = expert_refresh_future.result()
        matchup_pipeline = matchup_pipeline_future.result()
        guide_prestage = guide_prestage_future.result()
    runtime_service_selected = bool(
        service.get("active")
        or str(service.get("active_state") or "") == "activating"
        or str(service.get("sub_state") or "") == "auto-restart"
    )
    runtime_run_name = (
        (
            _run_name_from_command(str(service.get("command") or ""))
            or str(curriculum.get("run") or "")
            or None
        )
        if runtime_service_selected
        else None
    )
    runtime_specialist_id = _specialist_id_from_runtime(
        str(service.get("command") or ""),
        runtime_run_name,
    )
    # Bootstrap/materialization services do not necessarily carry the normal
    # ``--archetype`` argument.  Their receipt-backed final-refresh projection
    # is therefore the authoritative runtime identity while that managed
    # service is selected; falling back to the old production command makes a
    # live Marnie refresh look like the historical Slowking/Alakazam runtime.
    if (
        active_final_refresh.get("status") in {"running", "stopped"}
        and str(active_final_refresh.get("specialist_id") or "").strip()
    ):
        runtime_specialist_id = str(
            active_final_refresh["specialist_id"]
        ).strip().lower()
        runtime_run_name = str(
            active_final_refresh.get("run") or runtime_run_name or ""
        ).strip() or None
    protocol_service = (
        active_final_refresh.get("service") or service
        if active_final_refresh.get("status") in {"running", "stopped"}
        else service
    )
    specialist_protocol = specialist_protocol_state(
        runtime_specialist_id=runtime_specialist_id,
        runtime_run_name=runtime_run_name,
        runtime_service_state=(
            f"{protocol_service.get('active_state')}/"
            f"{protocol_service.get('sub_state')}"
        ),
    )
    post_starmie_handoff = post_starmie_specialist_handoff_state()
    handoff_source = (
        post_starmie_handoff
        if post_starmie_handoff.get("active")
        or str(post_starmie_handoff.get("phase") or "")
        not in {"", "waiting", "waiting_for_starmie_gate"}
        else owner_specialist_handoff_state()
    )
    specialist_handoff = reconcile_current_specialist_handoff(
        handoff_source,
        active_specialist=str(
            specialist_protocol.get("active_specialist") or ""
        ),
        program_progress=(
            specialist_protocol.get("program_progress")
            if isinstance(
                specialist_protocol.get("program_progress"), dict
            )
            else {}
        ),
        next_specialist=(
            (
                specialist_protocol.get("training_priority") or {}
            ).get("next_executable_specialist")
            or (
                (
                    specialist_protocol.get("training_priority") or {}
                ).get("ordered_unfinished_ids_after_active")
                or [None]
            )[0]
        ),
        active_runtime_refresh=bool(
            (specialist_protocol.get("active_runtime_refresh") or {}).get(
                "active"
            )
        ),
    )
    specialist_protocol = reconcile_protocol_with_active_handoff(
        specialist_protocol,
        specialist_handoff,
    )
    specialist_protocol = reconcile_protocol_with_live_curriculum(
        specialist_protocol,
        service=service,
        curriculum=curriculum,
    )
    if isinstance(specialist_protocol.get("preparation"), dict):
        specialist_protocol["preparation"][
            "current_deck_guide_prestage"
        ] = guide_prestage
    matchup_pipeline = reconcile_canonical_router_candidate(
        matchup_pipeline,
        specialist_protocol,
    )
    annotate_gpu_production_assignments(gpus, curriculum, specialist_handoff)
    training = authoritative_training_state(
        curriculum, transition, specialist_handoff
    )
    if active_final_refresh.get("status") in {
        "running",
        "complete",
        "stopped",
    }:
        training = active_final_refresh
    if postupload_boundary.get("current") is True:
        boundary_progress = dict(postupload_boundary.get("progress") or {})
        training = {
            **postupload_boundary,
            "current": boundary_progress.get("current"),
            "total": boundary_progress.get("total"),
            "percent": boundary_progress.get("percent"),
            "rate": boundary_progress.get("rate"),
            "rate_unit": boundary_progress.get("rate_unit"),
            "games_per_second": boundary_progress.get("gps"),
            "samples_per_second": boundary_progress.get("sps"),
            "metrics": boundary_progress.get("metrics") or {},
        }
        guide_modes = specialist_protocol.setdefault(
            "current_deck_guide_training_modes", {}
        )
        guide_modes["active_started_lineage"] = {
            "is_active": True,
            "specialist_id": "marnie-s-grimmsnarl-ex",
            "display_name": "Marnie's Grimmsnarl ex",
            "owner_revision": 141,
            "mode": "optional_offline_shadow_non_authoritative",
            "guide_weight": 0.0,
            "shadow_available": bool(
                (postupload_boundary.get("guide") or {}).get("shadow_available")
            ),
            "live_target_generation_enabled": False,
            "gradient_authority": False,
            "fusion_authority": False,
            "action_authority": False,
            "serving_authority": False,
            "gate_authority": False,
            "blocking_authority": False,
        }
    expert_refresh = active_expert_corpus_state(
        curriculum,
        expert_refresh,
    )
    switching = specialist_protocol.get("switching") or {}
    frozen_runtime_rows = (
        specialist_protocol.get("frozen_inference_opponents") or []
    )
    model_runtime_identity = {
        "active_learner": specialist_protocol.get("active_specialist"),
        "runtime_build": switching.get("runtime_build"),
        "runtime_root": switching.get("runtime_root"),
        "service_active": service.get("active") is True,
        "service_state": (
            f"{service.get('active_state')}/{service.get('sub_state')}"
        ),
        "frozen_inference_opponents": frozen_runtime_rows,
    }
    final_model_override: dict[str, Any] = {}
    final_matchup_transition: dict[str, Any] = {}
    if final_alakazam_models.get("available") is True:
        ordinary_model = dict(
            final_alakazam_models.get("ordinary_refresh") or {}
        )
        ordinary_arch = dict(ordinary_model.get("architecture") or {})
        ordinary_parameters = int(
            ordinary_model.get("learned_parameters") or 0
        )
        ordinary_tensor_elements = int(
            ordinary_model.get("serialized_tensor_elements") or 0
        )
        ordinary_checkpoint = ordinary_model.get("latest_checkpoint")
        ordinary_digest = ordinary_model.get("latest_checkpoint_sha256")
        ordinary_expanded = dict(
            ordinary_model.get("expanded_head_training") or {}
        )
        final_model_override = {
            "implementation": "TemporalCabtTransformer",
            "architecture": "Alakazam ordinary refresh",
            "run": "final_format_alakazam_r79",
            "profile_id": "alakazam-ordinary-refresh-r79",
            "profile": {
                "d_model": ordinary_arch.get("d_model"),
                "n_heads": ordinary_arch.get("attention_heads"),
                "spatial_layers": ordinary_arch.get("spatial_layers"),
                "temporal_layers": ordinary_arch.get("temporal_layers"),
                "option_decoder_layers": ordinary_arch.get("option_layers"),
                "ff_dim": ordinary_arch.get("feed_forward_width"),
                "max_context": ordinary_arch.get("history_context"),
                "decision_context": "history",
                "temporal_pos": "rope",
                "kv_cache": True,
            },
            "heads": {
                name: {"enabled": True}
                for name in DECISION_FUSION_REQUIRED_HEADS
            },
            "trainable_parameters": ordinary_parameters,
            "active_checkpoint": ordinary_checkpoint,
            "active_checkpoint_digest": ordinary_digest,
            "parameter_source": final_alakazam_models.get("source"),
            "parameter_breakdown": {
                "optimizer_active_current": ordinary_parameters,
                "current_non_active": 0,
                "staged_non_active": 0,
                "current_checkpoint_total": ordinary_parameters,
                "staged_architecture_total": ordinary_parameters,
            },
            "checkpoint_structure": {
                "verified": bool(ordinary_checkpoint and ordinary_digest),
                "checkpoint": ordinary_checkpoint,
                "checkpoint_digest": ordinary_digest,
                "model_parameters": ordinary_parameters,
                "state_tensor_elements": ordinary_tensor_elements,
                "adapter_parameters": 0,
                "adapter_expert_count": 0,
                "adapter_expert_ids": [],
                "expanded_head_training": ordinary_expanded,
                "source": final_alakazam_models.get("live_state_source"),
            },
            "dormant_modules": [],
            "matchup_adapter_roster_stage": {},
            "matchup_adapter_v6": {},
            "expanded_head_training": ordinary_expanded,
            "training_schedule": {
                "phase": "exact_25_epoch_specialist_bootstrap",
                "active_max_decisions_per_batch": 12_288,
                "epochs_completed": ordinary_model.get(
                    "latest_completed_epoch"
                ),
                "epochs_target": ordinary_model.get("epochs_target"),
            },
            "decision_fusion": {
                "available": True,
                "verified": True,
                "schema": DECISION_FUSION_SCHEMA,
                "training_enabled": True,
                "runtime_enabled": True,
                "serving_eligible": False,
                "phase": "ordinary_refresh_training",
                "required_heads": list(DECISION_FUSION_REQUIRED_HEADS),
                "required_head_count": len(DECISION_FUSION_REQUIRED_HEADS),
                "expected_required_head_count": len(
                    DECISION_FUSION_REQUIRED_HEADS
                ),
                "authoritative_action_path": "fused_policy",
                "matchup_adapter_behavior": "exact_bypass_not_materialized",
                "absent_deck_guide_behavior": "exact_bypass",
            },
            "runtime_identity": {
                "active_learner": "alakazam-refresh-r79",
                "runtime_build": "final-format-alakazam-r79",
                "runtime_root": str(
                    Path(__file__).resolve().parents[1]
                ),
                "service_active": service.get("active") is True,
                "service_state": (
                    f"{service.get('active_state')}/{service.get('sub_state')}"
                ),
                "frozen_inference_opponents": [],
            },
            "seed_checkpoint": ordinary_model.get("seed_checkpoint"),
            "seed_checkpoint_digest": ordinary_model.get(
                "seed_checkpoint_sha256"
            ),
        }
    if final_alakazam.get("mode") == "final_format_alakazam_h10_rl":
        h10_heads = (
            *DECISION_FUSION_REQUIRED_HEADS,
            "setup_board_outcome",
            "combo_state",
        )
        h10_parameters = int(final_alakazam.get("model_parameters") or 0)
        h10_checkpoint = final_alakazam.get("checkpoint")
        h10_digest = final_alakazam.get("checkpoint_digest")
        h10_structure = dict(final_alakazam.get("checkpoint_structure") or {})
        h10_fusion = dict(final_alakazam.get("decision_fusion") or {})
        h10_expanded = dict(final_alakazam.get("expanded_head_training") or {})
        h10_registry = str(
            final_alakazam.get("runtime_registry")
            or FINAL_FORMAT_ALAKAZAM_H10_REGISTRY
        )
        final_model_override = {
            "implementation": "TemporalCabtTransformer",
            "architecture": "Alakazam H10-I high-volume RL",
            "run": final_alakazam.get("run"),
            "profile_id": "H10-I/v1",
            "profile": {
                "d_model": 96,
                "n_heads": 8,
                "spatial_layers": 7,
                "temporal_layers": 3,
                "option_decoder_layers": 7,
                "ff_dim": 2496,
                "max_context": 320,
                "decision_context": "history",
                "temporal_pos": "rope",
                "kv_cache": True,
            },
            "heads": {name: {"enabled": True} for name in h10_heads},
            "trainable_parameters": h10_parameters,
            "active_checkpoint": h10_checkpoint,
            "active_checkpoint_digest": h10_digest,
            "parameter_source": h10_registry,
            "parameter_breakdown": {
                "optimizer_active_current": h10_parameters,
                "current_non_active": 0,
                "staged_non_active": 0,
                "current_checkpoint_total": h10_parameters,
                "staged_architecture_total": h10_parameters,
            },
            "checkpoint_structure": h10_structure,
            "dormant_modules": [],
            "matchup_adapter_roster_stage": {},
            "matchup_adapter_v6": {
                "active": True,
                "format": 6,
                "physical_slot_capacity": 64,
                "materialized_routes": 20,
            },
            "expanded_head_training": h10_expanded,
            "training_schedule": {
                "phase": "high_volume_final_submit_rl",
                "games_per_iteration": int(
                    final_alakazam.get("games_per_iteration") or 16384
                ),
                "maximum_iterations": int(
                    final_alakazam.get("iterations_target") or 21
                ),
                "training_seat_split": final_alakazam.get("training_seat_split"),
            },
            "decision_fusion": h10_fusion,
            "runtime_identity": {
                "active_learner": "alakazam-refresh-r79-h10",
                "runtime_build": "final-format-alakazam-r79-h10",
                "runtime_root": str(SPECIALIST_RUNTIME_ROOT),
                "service_active": True,
                "service_state": "active/running",
                "frozen_inference_opponents": [],
            },
            "release_gate": {
                "premium_skill_weighted_win_rate": final_alakazam.get(
                    "premium_strength_gate"
                ),
                "kaggle_rating_lower_bound": final_alakazam.get(
                    "kaggle_rating_lower_bound"
                ),
                "independent_checks": final_alakazam.get("rating_gate_separate"),
            },
        }
    if active_final_refresh.get("mode") == "final_format_crustle_h10_bootstrap":
        crustle_heads = (
            *DECISION_FUSION_REQUIRED_HEADS,
            "setup_board_outcome",
            "combo_state",
        )
        crustle_parameters = int(active_final_refresh.get("model_parameters") or 0)
        crustle_structure = dict(
            active_final_refresh.get("checkpoint_structure") or {}
        )
        final_model_override = {
            "implementation": "TemporalCabtTransformer",
            "architecture": "Crustle H10-I Fusion v3 bootstrap",
            "run": active_final_refresh.get("run"),
            "profile_id": "H10-I/v1",
            "heads": {name: {"enabled": True} for name in crustle_heads},
            "trainable_parameters": crustle_parameters,
            "active_checkpoint": active_final_refresh.get("checkpoint"),
            "active_checkpoint_digest": active_final_refresh.get(
                "checkpoint_digest"
            ),
            "parameter_source": active_final_refresh.get("source"),
            "parameter_breakdown": {
                "optimizer_active_current": crustle_parameters,
                "current_non_active": 0,
                "staged_non_active": 0,
                "current_checkpoint_total": crustle_parameters,
                "staged_architecture_total": crustle_parameters,
            },
            "checkpoint_structure": crustle_structure,
            "dormant_modules": [],
            "matchup_adapter_v6": {
                "active": True,
                "format": 6,
                "physical_slot_capacity": 64,
            },
            "training_schedule": {
                "phase": "weighted_all-guide_expert_bootstrap",
                "epoch": active_final_refresh.get("epoch"),
                "epochs_target": 35,
                "guide_active_epochs": [1, 35],
                "pilot_weighting_epochs": [1, 35],
            },
            "decision_fusion": {
                "schema": "poke_bot.causal_decision_fusion/v3",
                "available": True,
                "verified": True,
                "phase": "bootstrap_training",
                "runtime_enabled": True,
                "training_enabled": True,
                "required_heads": list(crustle_heads),
                "required_head_count": len(crustle_heads),
            },
            "runtime_identity": {
                "active_learner": "crustle",
                "runtime_build": "final-format-crustle-r113-h10",
                "runtime_root": str(SPECIALIST_RUNTIME_ROOT),
                "service_active": True,
                "service_state": "active/start",
                "frozen_inference_opponents": frozen_runtime_rows,
            },
        }
    if active_final_refresh.get("mode") in {
        "final_format_marnie_h10_bootstrap",
        "final_format_marnie_h10_rl",
    }:
        marnie_heads = (
            *DECISION_FUSION_REQUIRED_HEADS,
            "setup_board_outcome",
            "combo_state",
        )
        marnie_parameters = int(
            active_final_refresh.get("model_parameters") or 0
        )
        marnie_checkpoint = active_final_refresh.get("checkpoint")
        marnie_digest = active_final_refresh.get("checkpoint_digest")
        marnie_structure = dict(
            active_final_refresh.get("checkpoint_structure") or {}
        )
        marnie_adapter_format = str(
            marnie_structure.get("adapter_format") or ""
        )
        marnie_router_v6_active = (
            marnie_adapter_format == "poke-bot-matchup-adapter-bank-v6"
            and int(marnie_structure.get("adapter_slot_capacity") or 0) == 64
        )
        marnie_router_fix = read_json(FINAL_FORMAT_MARNIE_ROUTER_V6_FIX)
        if (
            marnie_router_fix.get("schema")
            == "poke_bot.final_format_marnie_h10_router_v6_registration_fix/v1"
        ):
            correction = dict(marnie_router_fix.get("correction") or {})
            final_matchup_transition = {
                "status": (
                    "activated"
                    if marnie_router_v6_active
                    else "staged_for_post_bootstrap_registration_boundary"
                ),
                "source_format": marnie_adapter_format or None,
                "target_format": correction.get("target_adapter_format"),
                "target_physical_slot_capacity": correction.get(
                    "physical_slot_capacity"
                ),
                "target_logical_active_route_count": correction.get(
                    "logical_active_route_count"
                ),
                "boundary": correction.get("boundary"),
                "source_checkpoint_immutable": correction.get(
                    "source_family_remains_immutable"
                ),
                "receipt": str(FINAL_FORMAT_MARNIE_ROUTER_V6_FIX),
                "training_interrupted": False,
            }
        marnie_service = dict(active_final_refresh.get("service") or {})
        final_model_override = {
            "implementation": "TemporalCabtTransformer",
            "architecture": "Marnie's Grimmsnarl ex H10-I Fusion v3",
            "run": active_final_refresh.get("run"),
            "profile_id": "H10-I/v1",
            "profile": {
                "d_model": 96,
                "n_heads": 8,
                "spatial_layers": 7,
                "temporal_layers": 3,
                "option_decoder_layers": 7,
                "ff_dim": 2496,
                "max_context": 320,
                "decision_context": "history",
                "temporal_pos": "rope",
                "kv_cache": True,
                "strategic_head_residual_width": 512,
            },
            "heads": {name: {"enabled": True} for name in marnie_heads},
            "trainable_parameters": marnie_parameters,
            "active_checkpoint": marnie_checkpoint,
            "active_checkpoint_digest": marnie_digest,
            "parameter_source": str(FINAL_FORMAT_MARNIE_H10_VALIDATION),
            "parameter_breakdown": {
                "optimizer_active_current": marnie_parameters,
                "current_non_active": 0,
                "staged_non_active": 0,
                "current_checkpoint_total": marnie_parameters,
                "staged_architecture_total": marnie_parameters,
            },
            "checkpoint_structure": marnie_structure,
            "dormant_modules": [],
            "matchup_adapter_roster_stage": {},
            "matchup_adapter_v6": {
                "active": marnie_router_v6_active,
                "format": 6,
                "physical_slot_capacity": 64,
                "materialized_routes": (
                    20
                    if marnie_router_v6_active
                    else int(marnie_structure.get("adapter_expert_count") or 0)
                ),
                "activation_boundary": (
                    None
                    if marnie_router_v6_active
                    else "post_bootstrap_registration"
                ),
            },
            "expanded_head_training": {
                "enabled": True,
                "learned_head_count": int(
                    active_final_refresh.get("learned_head_count") or 19
                ),
                "learned_route_count": int(
                    active_final_refresh.get("learned_route_count") or 19
                ),
            },
            "training_schedule": {
                "phase": (
                    "exact_25_epoch_h10_specialist_bootstrap"
                    if active_final_refresh.get("mode")
                    == "final_format_marnie_h10_bootstrap"
                    else "specialist_rl_5_plus_5"
                ),
                "epochs_completed": active_final_refresh.get("epoch"),
                "epochs_target": active_final_refresh.get("epochs_target"),
                "games_per_iteration": 8192,
            },
            "decision_fusion": {
                "available": True,
                "verified": True,
                "schema": active_final_refresh.get("decision_fusion_schema"),
                "training_enabled": True,
                "runtime_enabled": True,
                "required_heads": list(marnie_heads),
                "required_head_count": len(marnie_heads),
                "authoritative_action_path": "typed_output_centered_fused_policy",
                "route_schema": "typed_output_centered_per_head/v3",
                "reliability_bounds": [0.25, 4.0],
                "action_type_reliability_cap": 0.25,
            },
            "runtime_identity": {
                "active_learner": "marnie-s-grimmsnarl-ex",
                "runtime_build": "final-format-marnie-r104-h10",
                "runtime_root": str(SPECIALIST_RUNTIME_ROOT),
                "service_active": active_final_refresh.get("status") == "running",
                "service_state": (
                    f"{marnie_service.get('active_state')}/"
                    f"{marnie_service.get('sub_state')}"
                ),
                "frozen_inference_opponents": [],
            },
        }
    # ``postupload_boundary`` intentionally selects only the currently active
    # managed phase.  The durable guide-shadow authority is owned by the
    # bootstrap projection even after that phase completes, so source it from
    # ``postupload_bootstrap`` rather than from the inactive phase selector.
    marnie_guide_state = marnie_shadow_guide_projection(postupload_bootstrap)
    if (
        active_final_refresh.get("mode")
        in {"final_format_marnie_h10_bootstrap", "final_format_marnie_h10_rl"}
        and marnie_guide_state
    ):
        final_model_override = {
            **final_model_override,
            "training_targets": {
                **dict(final_model_override.get("training_targets") or {}),
                "current_deck_guide": marnie_guide_state,
            },
        }
    baseline_eval = baseline_eval_state()
    # Retain compatibility payloads for old dashboard clients, but label every
    # superseded or aliased view so it cannot masquerade as current evidence.
    training_alias = {
        **training,
        "compatibility_alias": True,
        "alias_of": "training",
    }
    if expert_refresh.get("available") is True:
        latest10 = {
            **latest10,
            "historical": True,
            "active": False,
            "superseded_by": "expert_refresh",
        }
    if baseline_eval.get("available") is True:
        baseline_eval = {
            **baseline_eval,
            "historical": True,
            "active": False,
            "superseded_by": "curriculum.latest_committed_research_controls",
        }
    if transition.get("active") is not True:
        transition = {
            **transition,
            "historical": True,
            "superseded_by": "specialist_protocol",
        }
    live_adapter_runtime = (
        (curriculum.get("model_contract") or {}).get("matchup_adapter_runtime")
        or {}
    )
    if live_adapter_runtime.get("enabled") is True:
        historical_fit = matchup_pipeline.get("adapter_fit") or {}
        if historical_fit:
            matchup_pipeline = {
                **matchup_pipeline,
                "adapter_fit": {
                    **historical_fit,
                    "historical": True,
                    "active": False,
                    "superseded_by": (
                        "curriculum.model_contract.matchup_adapter_runtime"
                    ),
                },
            }
    print(
        json.dumps(
            {
                "ok": True,
                "observed_at": time.time(),
                "system": system,
                "service": service,
                "transition": transition,
                "training": training,
                "bootstrap": training_alias,
                "baseline_eval": baseline_eval,
                "latest10": latest10,
                "expert_refresh": expert_refresh,
                "matchup_pipeline": matchup_pipeline,
                "specialist_protocol": specialist_protocol,
                "specialist_handoff": specialist_handoff,
                "managed_boundary": postupload_boundary,
                "curriculum": curriculum,
                "gpus": gpus,
                "fleet": {
                    "inzi": {
                        "reachable": True,
                        "observed_at": time.time(),
                        "name": "Inzi",
                        "role": "trainer + simulator",
                        "platform": "linux",
                        "system": system,
                        "gpus": gpus,
                        "worker": {
                            **(curriculum.get("worker") or {}),
                            "active": (
                                service["active"]
                                or curriculum["active"]
                                or specialist_handoff["active"]
                            ),
                            "command": (
                                (curriculum.get("worker") or {}).get("command")
                                or service.get("command")
                                or (
                                    specialist_handoff.get("service") or {}
                                ).get("command")
                                or ", ".join(curriculum["active_units"])
                            ),
                        },
                    },
                    "elmo": elmo,
                },
                "model": {
                    **(curriculum.get("model_contract") or {}),
                    **final_model_override,
                    "final_format_alakazam": final_alakazam_models,
                    "final_format_marnie": final_marnie,
                    "run": (
                        final_model_override.get("run")
                        or curriculum.get("run")
                    ),
                    "staged_expanded_head_training": (
                        specialist_handoff.get(
                            "staged_expanded_head_training"
                        )
                        or {}
                    ),
                    "matchup_pipeline": matchup_pipeline,
                    "runtime_identity": (
                        final_model_override.get("runtime_identity")
                        or model_runtime_identity
                    ),
                    "canonical_matchup_transition": (
                        final_matchup_transition
                        or
                        (
                            specialist_protocol.get("training_priority") or {}
                        ).get("staged_v5_transition")
                        or {}
                    ),
                },
                "pure_rl_status": (
                    curriculum.get("progress", {}).get("line", "")
                    if curriculum.get("active")
                    else (
                        "STOPPED AT COMMITTED ITERATION "
                        f"{curriculum.get('last_completed_iteration', '—')} · "
                        f"{str(switching.get('runtime_build') or 'specialist runtime')} "
                        "RELAUNCH PENDING\nlast progress: "
                        + str(curriculum.get("progress", {}).get("line", ""))
                    )
                ),
                "recent_events": recent_events(curriculum.get("run")),
            },
            separators=(",", ":"),
            # Keep telemetry available if a future human-edited YAML field
            # contains another standards-compliant timestamp scalar.
            default=str,
        )
    )


if __name__ == "__main__":
    main()
