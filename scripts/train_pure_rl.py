#!/usr/bin/env python
"""Single-lineage pure-RL loop: full-hardware collect → AWR train → held-out gate.

Modes:
  --mode core         deck-agnostic Stage A (default)
  --mode specialist   one explicitly named pinned archetype after warm-start
  --smoke             synthetic games (no CABT) for CI / canary wiring

Production collect saturates local CPU + dual-GPU leaves and optionally
additive whole-game farms (Elmo/bert) into the same shard stream. One AWR
trainee on the host — remotes are collect capacity, not a second trainer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("POKEBOT_BLACKWELL_STRATEGY_HEADS", "0")

from poke_bot import alakazam_heuristics, config, deck_guides, paths  # noqa: E402
from poke_bot.dataset import BootstrapDataset  # noqa: E402
from poke_bot.matchup_adapters import (  # noqa: E402
    EXPERT_IDS,
    UNKNOWN_ROUTE,
    route_for_archetype,
)
from poke_bot.matchup_adapter_activation import (  # noqa: E402
    TRAINING_TICKET_SCHEMA,
    training_route_for_archetype,
)
from poke_bot.matchup_adapter_routes import (  # noqa: E402
    resolve_matchup_adapter_route_contract,
)
from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT  # noqa: E402
from poke_bot.pure_rl.aborts import evaluate_aborts  # noqa: E402
from poke_bot.pure_rl.curriculum import (  # noqa: E402
    stage_for_iteration,
    stage_to_dict,
)
from poke_bot.pure_rl.dataset_bridge import (  # noqa: E402
    StreamingReplayCache,
    dataset_from_shard,
    ensure_replay_cache_manifest,
    validated_replay_cache_manifest,
)
from poke_bot.pure_rl.eval_public import (  # noqa: E402
    HeldOutGateResult,
    OFFICIAL_BASELINE_IDS,
    active_gate_goal_rank,
    aggregate_heldout_wr,
    heldout_exploration_decision,
    heldout_goal_rank,
)
from poke_bot.pure_rl.holdout_supersession import (  # noqa: E402
    apply_external_holdout_supersession,
    superseded_external_archetypes,
)
from poke_bot.pure_rl.hardware import full_hardware_profile  # noqa: E402
from poke_bot.pure_rl.metrics import IterationMetrics, metrics_to_dict  # noqa: E402
from poke_bot.pure_rl.artifact_retention import apply_artifact_retention  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    carry_learner_candidate,
    commit_rehearsal_receipt,
    recover_rehearsal,
    rehearsal_due,
    rehearsal_epochs_for_iteration,
    rehearsal_paths,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.expert_adapter_rehearsal import (  # noqa: E402
    rehearsal_paths as expert_adapter_rehearsal_paths,
    run_or_recover_expert_adapter_rehearsal,
)
from poke_bot.pure_rl.model_profile import (  # noqa: E402
    build_pure_rl_model,
    count_params,
    model_config_dict,
    pure_rl_model_config,
    validate_param_budget,
)
from poke_bot.pure_rl.guide_weight_review import (  # noqa: E402
    emit_review_request,
)
from poke_bot.promotion import CheckpointIdentity  # noqa: E402
from poke_bot.r241_checkpoint_receipts import (  # noqa: E402
    PEAK_R195_PRESERVATION_SCHEMA as R241_PEAK_R195_PRESERVATION_RECEIPT_SCHEMA,
)
from poke_bot.process_memory import close_mp_queue, release_process_heap  # noqa: E402
from poke_bot.slop_box_combo_targets import (  # noqa: E402
    attach_slop_box_combo_state_labels,
    is_slop_box_combo_deck,
)
from poke_bot.slowking_combo_targets import (  # noqa: E402
    attach_slowking_combo_state_labels,
    is_exact_slowking_deck,
)
from poke_bot.pure_rl.shards import (  # noqa: E402
    CompactDecision,
    CompactGame,
    CompactShardWriter,
)
from poke_bot.train import (  # noqa: E402
    COMBO_STATE_BASE_LOSS_WEIGHT,
    GUIDE_TRAINING_MODE_LEGACY,
    GUIDE_TRAINING_MODE_STRATEGIC,
    GUIDE_STRATEGIC_TRAINING_MODES,
    GUIDE_TRAINING_MODES,
    R241_PEAK_R195_ACTIVE_SOURCES,
    R241_PEAK_R195_CLARIFICATION_REVISION,
    R241_PEAK_R195_PHYSICAL_SOURCES,
    R241_PEAK_R195_TRAINING_CONTRACT_SCHEMA,
    R241_R195_HISTORICAL_OWNER_CONTRACT_SHA256,
    SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT,
    TrainConfig,
    assert_strategic_curriculum_receipt_contract,
    belief_card_vocab_from_state,
    canonical_r241_peak_r195_training_contract,
    rl_train_step,
    streaming_r260_host_rehearsal_step,
    supervised_rehearsal_step,
    validate_r241_isolated_adapter_continuation,
    validate_r241_peak_r195_live_fusion_record,
)

DEFAULT_REMOTE_ENDPOINTS = "192.168.1.143:8765,bert.local:8766"
ITERATION_SEED_STRIDE = 100_000
FORMAL_GATE_SEED_OFFSET = 19_000_000
RESEARCH_CONTROL_SEED_OFFSET = 39_000_000
STRONG_PUBLIC_PRACTICE_GROUP = "strong_public_practice"
RESEARCH_CONTROL_GROUP = "research_controls"
EXPERT_REHEARSAL_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)
EXPERT_REHEARSAL_TARGET_CHOICES = (
    *EXPERT_REHEARSAL_TARGETS,
    "combo_state_rows",
)
LADDER_DECK_MIX_PATH = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
LADDER_DECK_REPRESENTATIVES_PATH = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)
SPECIALIST_DECK_REPRESENTATIVES_PATH = (
    ROOT / "data" / "training_mixes" / "specialist_representatives.v1.json"
)


def _effective_boundary_design_migration_reason(
    args: argparse.Namespace,
) -> str:
    """Resolve the same clean-boundary authorization at every use site."""

    return str(
        os.environ.get("PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE")
        or args.boundary_design_migration_reason
        or ""
    ).strip()


def _registered_matchup_target_ids(
    initial_checkpoint: Path | None,
) -> tuple[str, ...]:
    """Resolve the acting-specialist roster from the exact warm start."""

    if initial_checkpoint is None:
        if (
            os.environ.get("POKEBOT_MATCHUP_ADAPTER_FORMAT", "").strip()
            == "poke-bot-matchup-adapter-bank-v6"
        ):
            from poke_bot.matchup_adapters_v6 import load_slot_registry

            registry_raw = os.environ.get(
                "POKEBOT_MATCHUP_ADAPTER_REGISTRY_PATH", ""
            ).strip()
            if not registry_raw:
                raise RuntimeError(
                    "resumed Router Format 6 run lacks its registry path"
                )
            registry = load_slot_registry(
                Path(registry_raw).expanduser().resolve()
            )
            return tuple(str(value) for value in registry["active_expert_ids"])
        return tuple(EXPERT_IDS)
    import torch

    saved = torch.load(
        initial_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    adapter_config = dict(
        (saved.get("extra") or {}).get("matchup_adapter_config") or {}
    )
    return tuple(
        resolve_matchup_adapter_route_contract(adapter_config).target_ids
    )


def _default_research_control_registry() -> Path:
    durable = paths.OUTPUTS_DIR / "state" / "research_control_registry_latest.json"
    return durable if durable.is_file() else ROOT / "ops" / "research_control_registry_v1.json"


def _default_frozen_specialist_registry() -> Path:
    return ROOT / "ops" / "frozen_specialist_registry_v1.json"


def _load_frozen_specialist_registry(path: Path) -> dict[str, Any]:
    """Load the mutable roster of immutable, completed specialist packages."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = list(payload.get("specialists") or [])
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("frozen specialist registry rows must be objects")
        opponent_id = str(row.get("opponent_id") or "")
        archetype_id = str(row.get("archetype_id") or "")
        checkpoint_digest = str(row.get("checkpoint_digest") or "")
        content_digest = str(row.get("content_digest") or "")
        if (
            not opponent_id
            or opponent_id in ids
            or not archetype_id
            or not checkpoint_digest.startswith("sha256:")
            or not content_digest.startswith("sha256:")
            or row.get("frozen") is not True
            or row.get("public_mix_eligible") is not True
            # Frozen predecessors belong to the additive premium S+ gate and
            # inference-only public mix.  They must never be admitted to the
            # separate official research-control registry.
            or row.get("research_eligible") is not False
        ):
            raise ValueError(
                f"invalid frozen specialist registry row: {opponent_id!r}"
            )
        ids.add(opponent_id)
    if (
        payload.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or int(payload.get("version") or 0) < 1
    ):
        raise ValueError("invalid frozen specialist registry schema/version")
    return {
        **payload,
        "path": str(source),
        "specialists": rows,
    }


def _augment_gate_with_frozen_specialists(
    contract: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Add every completed specialist to the premium gate as an S-tier agent."""
    result = json.loads(json.dumps(contract))
    gate = dict(result.get("next_gate") or {})
    roster, newly_retired = apply_external_holdout_supersession(
        list(gate.get("roster") or []),
        registry,
    )
    existing_ids = {str(row.get("opponent_id") or "") for row in roster}
    additions = [
        dict(row)
        for row in (registry.get("specialists") or [])
        if str(row.get("opponent_id") or "") not in existing_ids
    ]
    if not additions and not newly_retired:
        return result
    for row in additions:
        roster.append(
            {
                "opponent_id": str(row["opponent_id"]),
                "archetype_id": str(row["archetype_id"]),
                "archetype_label": str(
                    row.get("archetype_label") or row["archetype_id"]
                ),
                "tier": "S+",
                "weight": 2.0,
                "content_digest": str(row["content_digest"]),
                "source": str(
                    row.get("source")
                    or f"frozen specialist {row['checkpoint_digest']}"
                ),
                "frozen_specialist": True,
                "frozen_checkpoint_digest": str(row["checkpoint_digest"]),
            }
        )
    version = int(registry["version"])
    frozen_count = sum(
        1 for row in roster if row.get("frozen_specialist") is True
    )
    base_count = len(roster) - frozen_count
    old_gate_id = str(gate.get("id") or "")
    gate_id = f"{old_gate_id}+frozen-specialists-r{version}"
    evaluation = dict(gate.get("evaluation") or {})
    per_opponent = int(evaluation.get("games_per_opponent") or 250)
    if (
        per_opponent != 250
        or int(evaluation.get("seat0_games_per_opponent") or 125) != 125
        or int(evaluation.get("seat1_games_per_opponent") or 125) != 125
    ):
        raise ValueError("frozen specialists require the exact 250/125/125 gate")
    evaluation["games_total"] = 250 * len(roster)
    gate.update(
        {
            "id": gate_id,
            "label": (
                str(gate.get("label") or "Strong public-agent gate")
                + f" + {len(additions)} frozen specialist"
                + ("s" if len(additions) != 1 else "")
            ),
            "evaluation": evaluation,
            "roster": roster,
            "frozen_specialist_registry": {
                "version": version,
                "path": str(registry["path"]),
                "opponent_ids": [
                    str(row["opponent_id"])
                    for row in registry.get("specialists") or []
                ],
            },
        }
    )
    result["active_gate_id"] = gate_id
    result["next_gate"] = gate
    semantics = dict(result.get("active_gate_semantics") or {})
    superseded_archetypes = superseded_external_archetypes(registry)
    prior_retired_ids = {
        str(value)
        for value in semantics.get("superseded_external_premium_opponent_ids")
        or []
    }
    retired_ids = sorted(
        prior_retired_ids
        | {str(row.get("opponent_id") or "") for row in newly_retired}
    )
    semantics.update(
        {
            "gate_roster_size": len(roster),
            "gate_games_total": 250 * len(roster),
            "base_premium_agents": base_count,
            "original_base_premium_agents": 8,
            "frozen_specialist_agents": frozen_count,
            "frozen_specialist_tier": "S+",
            "superseded_external_premium_archetypes": list(
                superseded_archetypes
            ),
            "superseded_external_premium_opponent_ids": retired_ids,
            "historical_superseded_results_preserved": True,
            "invariant": (
                "The active premium gate contains every non-superseded external "
                "agent plus every frozen completed specialist. Each receives "
                "exactly 250 greedy games split 125/125 by seat."
            ),
        }
    )
    result["active_gate_semantics"] = semantics
    fallback = dict(result.get("fallback_transition") or {})
    if fallback:
        fallback["prior_gate_id"] = gate_id
        fallback["id"] = (
            str(fallback.get("id") or "specialist-fallback")
            + f"+frozen-specialists-r{version}"
        )
        result["fallback_transition"] = fallback
    return result


def _research_control_registry_for_lineage(
    requested_path: Path,
    *,
    snapshot_dir: Path,
    immutable_manifest: Optional[dict[str, Any]],
) -> Path:
    """Pin once for a new lineage and reuse that exact snapshot on resume."""
    from poke_bot.pure_rl.research_controls import (
        pin_research_control_registry_file,
    )

    if immutable_manifest is None:
        return pin_research_control_registry_file(
            requested_path,
            snapshot_dir=snapshot_dir,
        )
    stored = dict(immutable_manifest.get("research_control_registry") or {})
    stored_path = str(stored.get("path") or "").strip()
    if not stored_path:
        raise RuntimeError(
            "resumed lineage manifest lacks its pinned research-control registry"
        )
    pinned = Path(stored_path).expanduser().resolve()
    actual = _path_content_identity(pinned)
    if actual != stored:
        raise RuntimeError(
            "resumed lineage research-control registry changed: "
            f"stored={stored!r} actual={actual!r}"
        )
    return pinned


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument(
        "--population-own-models-only",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_POPULATION_OWN_MODELS_ONLY", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Population-phase specialist cycle: public collection may use only "
            "the checksum-verified frozen-specialist registry. External agents "
            "remain evaluation/research-only and never enter replay."
        ),
    )
    p.add_argument(
        "--population-opponent-registry",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_POPULATION_OPPONENT_REGISTRY"])
            if os.environ.get("PURE_RL_POPULATION_OPPONENT_REGISTRY")
            else None
        ),
        help=(
            "Checksum-bound current + selected-history registry for the "
            "14-member own-model population. Required in population mode."
        ),
    )
    p.add_argument(
        "--specialist-archetype",
        default=None,
        help=(
            "Exact pinned deck ID for specialist mode. Required for "
            "--mode specialist; the trainer never falls back to Hammer-Pult."
        ),
    )
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument(
        "--fixed-cycle-updates",
        type=int,
        default=int(os.environ.get("PURE_RL_FIXED_CYCLE_UPDATES", "0")),
        help=(
            "Enable the receipt-backed r241 fixed cycle. A nonzero value "
            "keeps running through every configured update even if a gate "
            "passes (0=ordinary gate-directed behavior)."
        ),
    )
    p.add_argument(
        "--r241-peak-r195-preservation-receipt",
        type=Path,
        default=None,
        help=(
            "Checkpoint-derived v2 host receipt authorizing the exact peak-r195 "
            "profile for the r241 ten-update cycle. Forbidden for ordinary runs."
        ),
    )
    p.add_argument(
        "--r241-peak-r195-preservation-receipt-sha256",
        default="",
        help=(
            "Canonical sha256: digest of --r241-peak-r195-preservation-receipt."
        ),
    )
    p.add_argument("--r241-own-deck-migration-receipt", type=Path, default=None)
    p.add_argument("--r241-own-deck-migration-receipt-sha256", default="")
    p.add_argument("--r241-own-deck-canary-activation-receipt", type=Path, default=None)
    p.add_argument("--r241-own-deck-canary-activation-receipt-sha256", default="")
    p.add_argument("--r241-own-deck-sidecar-binding", type=Path, default=None)
    p.add_argument("--r241-own-deck-sidecar-binding-sha256", default="")
    p.add_argument("--r241-own-deck-inzi-dataset-binding", type=Path, default=None)
    p.add_argument("--r241-own-deck-inzi-dataset-binding-sha256", default="")
    p.add_argument("--r274-expert-tactical-overlay", type=Path, default=None)
    p.add_argument("--r274-expert-tactical-overlay-sha256", default="")
    p.add_argument(
        "--r274-bootstrap-handoff-receipt",
        type=Path,
        default=None,
        help=(
            "Immutable post-submission tactical-route activation receipt. "
            "When supplied, the exact external 25-epoch bootstrap is already "
            "complete and must not be repeated before update zero."
        ),
    )
    p.add_argument("--r274-bootstrap-handoff-receipt-sha256", default="")
    p.add_argument(
        "--r274-r195-research-baseline-receipt",
        type=Path,
        default=None,
        help=(
            "Immutable local-install receipt for the exact NO-RTP Kaggle "
            "submission 55378392 research opponent."
        ),
    )
    p.add_argument("--r274-r195-research-baseline-receipt-sha256", default="")
    p.add_argument("--r280-contiguous-expert-pack", type=Path, default=None)
    p.add_argument("--r280-contiguous-expert-pack-receipt", type=Path, default=None)
    p.add_argument("--r280-refresh-device", default="cuda:1")
    p.add_argument(
        "--r274-submission-boundary-dir",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_R274_SUBMISSION_BOUNDARY_DIR"])
            if os.environ.get("PURE_RL_R274_SUBMISSION_BOUNDARY_DIR")
            else None
        ),
        help=(
            "Receipt exchange directory for the r274 five-update submission "
            "fence. Required for the exact 25-update cycle."
        ),
    )
    p.add_argument(
        "--r274-submission-boundary-poll-seconds",
        type=float,
        default=float(
            os.environ.get("PURE_RL_R274_SUBMISSION_BOUNDARY_POLL_SECONDS", "15")
        ),
        help="Polling interval while a required r274 upload receipt is pending.",
    )
    p.add_argument("--games-per-iter", type=int, default=256)
    p.add_argument(
        "--require-exact-training-seat-split",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_REQUIRE_EXACT_TRAINING_SEAT_SPLIT", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Fail closed unless assigned source games, retained source games, "
            "and replay sequences consumed by training are each exactly 50/50 "
            "between seats 0 and 1. This is the final-format Alakazam contract; "
            "it does not create a second-seat-priority curriculum."
        ),
    )
    p.add_argument(
        "--train-epochs",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_EPOCHS", "2")),
        help=(
            "Fresh-data AWR passes per iteration. Two is the production default; "
            "treat this as a measured hyperparameter, not a fixed design rule."
        ),
    )
    p.add_argument(
        "--train-games-per-batch",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_GAMES_PER_BATCH", "48")),
        help="Whole-game accumulation cap for each RL optimizer micro-batch.",
    )
    p.add_argument(
        "--train-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get("PURE_RL_TRAIN_MAX_DECISIONS_PER_BATCH", "4096")
        ),
        help="Decision-timestep cap per RL micro-batch; CUDA OOMs split safely.",
    )
    p.add_argument(
        "--train-warmup-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_TRAIN_WARMUP_MAX_DECISIONS_PER_BATCH", "0"
            )
        ),
        help=(
            "Optional decision cap used for the first N RL iterations. Set "
            "this together with --train-warmup-iterations; 0 disables the "
            "schedule. This is useful for denser early updates to auxiliary "
            "heads before returning to the steady-state throughput cap."
        ),
    )
    p.add_argument(
        "--train-warmup-iterations",
        type=int,
        default=int(os.environ.get("PURE_RL_TRAIN_WARMUP_ITERATIONS", "0")),
        help=(
            "Number of zero-based iterations that use the warmup decision "
            "cap; iteration N and later use --train-max-decisions-per-batch."
        ),
    )
    p.add_argument(
        "--train-device-resident",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_TRAIN_DEVICE_RESIDENT", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Pack the exact stateless replay window once into CUDA memory and "
            "train from GPU-resident indices; capacity failure falls back to "
            "the bounded host path."
        ),
    )
    p.add_argument(
        "--train-device-resident-min-free-gib",
        type=float,
        default=float(
            os.environ.get("PURE_RL_TRAIN_DEVICE_RESIDENT_MIN_FREE_GIB", "8")
        ),
        help="VRAM headroom retained while packing the resident replay corpus.",
    )
    p.add_argument(
        "--train-optimizer-sparse-prefetch",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_TRAIN_OPTIMIZER_SPARSE_PREFETCH", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Pack and pin exactly one next host batch's sparse board/action/"
            "option tensors and transfer it on a dedicated CUDA stream. "
            "Default-off; targets, losses, ordering, and optimizer semantics "
            "remain on the existing history-policy path."
        ),
    )
    p.add_argument(
        "--prize-plan-h3-cache-receipt",
        type=Path,
        default=None,
        help=(
            "Inactive-by-default trainer-only H3 additive cache for the exact "
            "current replay window. Requires the paired clean-boundary activation "
            "receipt and is rejected on any identity or coverage mismatch."
        ),
    )
    p.add_argument("--prize-plan-h3-cache-receipt-sha256", default="")
    p.add_argument(
        "--prize-plan-h3-activation-receipt",
        type=Path,
        default=None,
        help=(
            "Later owner-authorized H3 activation receipt. Merely supplying a "
            "cache never enables the provider."
        ),
    )
    p.add_argument("--prize-plan-h3-activation-receipt-sha256", default="")
    p.add_argument(
        "--prize-plan-h3-live-sidecar",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_PRIZE_PLAN_H3_LIVE_SIDECAR"])
            if os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_SIDECAR")
            else None
        ),
        help="Frozen trainer-only Prize-plan-v2 checkpoint used to materialize each sealed replay window before optimization.",
    )
    p.add_argument(
        "--prize-plan-h3-live-sidecar-sha256",
        default=os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_SIDECAR_SHA256", ""),
    )
    p.add_argument(
        "--prize-plan-h3-live-scale-support",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_PRIZE_PLAN_H3_LIVE_SCALE_SUPPORT"])
            if os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_SCALE_SUPPORT")
            else None
        ),
    )
    p.add_argument(
        "--prize-plan-h3-live-scale-support-sha256",
        default=os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_SCALE_SUPPORT_SHA256", ""),
    )
    p.add_argument(
        "--prize-plan-h3-live-c3-support",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_PRIZE_PLAN_H3_LIVE_C3_SUPPORT"])
            if os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_C3_SUPPORT")
            else None
        ),
    )
    p.add_argument(
        "--prize-plan-h3-live-c3-support-sha256",
        default=os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_C3_SUPPORT_SHA256", ""),
    )
    p.add_argument(
        "--prize-plan-h3-live-contract",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_PRIZE_PLAN_H3_LIVE_CONTRACT"])
            if os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_CONTRACT")
            else None
        ),
    )
    p.add_argument(
        "--prize-plan-h3-live-contract-sha256",
        default=os.environ.get("PURE_RL_PRIZE_PLAN_H3_LIVE_CONTRACT_SHA256", ""),
    )
    p.add_argument(
        "--archetype-aux-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_ARCHETYPE_AUX_LOSS_WEIGHT", "0.05")),
        help="Opponent-archetype classification loss used by curriculum RL.",
    )
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_OPP_HAND_LOSS_WEIGHT", "0.05")),
        help="Masked opponent-hand belief loss; exact simulator labels only.",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_OPP_REMAINDER_LOSS_WEIGHT", "0.05")),
        help="Masked hidden-remainder belief loss; exact simulator labels only.",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_LETHAL_THREAT_LOSS_WEIGHT", "0.025")),
        help="Masked near-term prize-take/lethal trajectory loss.",
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=float(os.environ.get("PURE_RL_PRIZE_RACE_LOSS_WEIGHT", "0.025")),
        help="Masked public prize-race state regression loss.",
    )
    p.add_argument(
        "--current-deck-guide-loss-weight",
        "--alakazam-guide-loss-weight",
        dest="alakazam_guide_loss_weight",
        type=float,
        default=float(
            os.environ.get(
                "PURE_RL_CURRENT_DECK_GUIDE_LOSS_WEIGHT",
                os.environ.get("PURE_RL_ALAKAZAM_GUIDE_LOSS_WEIGHT", "0"),
            )
        ),
        help=(
            "Receipt-backed training-only guide multiplier. Legacy started "
            "runs apply confidence-weighted policy CE; future strategic mode "
            "applies it only to observed-target head losses. Core stays "
            "exactly 0. The Alakazam option name is a compatibility alias."
        ),
    )
    p.add_argument(
        "--current-deck-guide-training-mode",
        choices=tuple(sorted(GUIDE_TRAINING_MODES)),
        default=os.environ.get(
            "PURE_RL_CURRENT_DECK_GUIDE_TRAINING_MODE",
            GUIDE_TRAINING_MODE_LEGACY,
        ),
        help=(
            "Explicit guide-loss semantics. Already-started runs retain "
            "legacy policy CE; future specialists use observed-target "
            "strategic curriculum only."
        ),
    )
    p.add_argument(
        "--setup-board-outcome-loss-weight",
        type=float,
        default=float(
            os.environ.get(
                "PURE_RL_SETUP_BOARD_OUTCOME_LOSS_WEIGHT",
                str(SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT),
            )
        ),
        help=(
            "Ordinary observed-target weight for the future option-conditioned "
            "setup/bench outcome head."
        ),
    )
    p.add_argument(
        "--combo-state-loss-weight",
        type=float,
        default=float(
            os.environ.get("PURE_RL_COMBO_STATE_LOSS_WEIGHT", "0")
        ),
        help=(
            "Ordinary observed-target weight for the causal combo-state head. "
            "Required at 0.025 for Slowking and H10 specialists that retain "
            "the head (Alakazam/Marnie/Crustle/Slop Box); zero elsewhere. "
            "Missing labels remain masked."
        ),
    )
    p.add_argument(
        "--current-deck-guide-curriculum-spec",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_CURRENT_DECK_GUIDE_CURRICULUM_SPEC"])
            if os.environ.get(
                "PURE_RL_CURRENT_DECK_GUIDE_CURRICULUM_SPEC"
            )
            else None
        ),
        help="Checksum-bound future strategic curriculum specification.",
    )
    p.add_argument(
        "--current-deck-guide-head-role-map",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_CURRENT_DECK_GUIDE_HEAD_ROLE_MAP"])
            if os.environ.get("PURE_RL_CURRENT_DECK_GUIDE_HEAD_ROLE_MAP")
            else None
        ),
        help="Checksum-bound per-head action-route inventory.",
    )
    p.add_argument(
        "--current-deck-guide-curriculum-validation-receipt",
        type=Path,
        default=(
            Path(
                os.environ[
                    "PURE_RL_CURRENT_DECK_GUIDE_CURRICULUM_VALIDATION_RECEIPT"
                ]
            )
            if os.environ.get(
                "PURE_RL_CURRENT_DECK_GUIDE_CURRICULUM_VALIDATION_RECEIPT"
            )
            else None
        ),
        help="Immutable validation receipt for future guide curriculum wiring.",
    )
    p.add_argument(
        "--tactical-outcome-loss-weight-override",
        type=float,
        default=None,
        help=(
            "Receipt-backed clean-boundary override for the inherited expanded "
            "tactical-outcome head. Ordinary runs must leave this unset."
        ),
    )
    p.add_argument(
        "--dormant-matchup-adapter-epochs",
        type=int,
        default=int(
            os.environ.get("PURE_RL_DORMANT_MATCHUP_ADAPTER_EPOCHS", "0")
        ),
        help=(
            "Behavior-inert adapter-only passes over the active specialist's "
            "exact mirror and active-gate strong-public rows after ordinary "
            "RL (0 disables)."
        ),
    )
    p.add_argument(
        "--dormant-matchup-adapter-lr",
        type=float,
        default=float(
            os.environ.get("PURE_RL_DORMANT_MATCHUP_ADAPTER_LR", "1e-4")
        ),
    )
    p.add_argument(
        "--dormant-matchup-adapter-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_DORMANT_MATCHUP_ADAPTER_MAX_DECISIONS_PER_BATCH",
                "2048",
            )
        ),
        help=(
            "Independent decision cap for adapter-only training. Adapter batches "
            "also split recursively on CUDA OOM without changing the corpus."
        ),
    )
    p.add_argument(
        "--dormant-matchup-adapter-activation-receipt",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_DORMANT_MATCHUP_ADAPTER_ACTIVATION_RECEIPT"])
            if os.environ.get(
                "PURE_RL_DORMANT_MATCHUP_ADAPTER_ACTIVATION_RECEIPT"
            )
            else None
        ),
        help="Immutable boundary authorization for adapter-only training.",
    )
    p.add_argument("--collect-temperature", type=float, default=1.0)
    p.add_argument(
        "--official-collect-frac",
        type=float,
        default=float(os.environ.get("PURE_RL_OFFICIAL_COLLECT_FRAC", "0")),
        help=(
            "Fraction of the public-training wave assigned to sampled practice "
            "against the active strong-public gate roster (specialist mode only). "
            "Formal evaluation uses a separate greedy seed schedule."
        ),
    )
    p.add_argument(
        "--research-control-games-per-iter",
        type=int,
        default=int(
            os.environ.get("PURE_RL_RESEARCH_CONTROL_GAMES_PER_ITER", "1000")
        ),
        help=(
            "Exact additive greedy measurement games assigned to the pinned "
            "research-control registry. They are training_eligible=false, use a "
            "seed namespace disjoint from collection and the formal gate, and "
            "never enter replay/AWR or gate pass/fail. The same number of former "
            "fixed-budget control slots is reclaimed for active-gate practice."
        ),
    )
    p.add_argument(
        "--disable-research-control-measurement",
        action="store_true",
        default=str(
            os.environ.get("PURE_RL_DISABLE_RESEARCH_CONTROL_MEASUREMENT", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Suppress the separate research-control evaluation wave while "
            "retaining its historical reclaimed training slots. This is valid "
            "only with the receipt-backed derivative revision-25 migration."
        ),
    )
    p.add_argument(
        "--frozen-specialist-registry",
        type=Path,
        default=Path(
            os.environ.get(
                "PURE_RL_FROZEN_SPECIALIST_REGISTRY",
                str(_default_frozen_specialist_registry()),
            )
        ),
        help=(
            "Mutable registry of immutable completed-specialist packages. Every "
            "entry is included in public-mix training and appended to the "
            "premium holdout as an S-tier 250-game opponent."
        ),
    )
    p.add_argument(
        "--official-adaptive-targeting",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_TARGETING", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Allocate the official-training quota from the latest exact "
            "heldout gaps, concentrating games on weaker opponents while "
            "retaining a minimum share for every official baseline."
        ),
    )
    p.add_argument(
        "--official-adaptive-min-share",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_MIN_SHARE", "0.05")
        ),
        help="Minimum fraction of the official quota retained per opponent.",
    )
    p.add_argument(
        "--official-adaptive-gap-power",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_ADAPTIVE_GAP_POWER", "2.0")
        ),
        help="Exponent applied to each exact heldout gap before normalization.",
    )
    p.add_argument(
        "--strong-public-practice-target-wr",
        type=float,
        default=float(
            os.environ.get("PURE_RL_STRONG_PUBLIC_PRACTICE_TARGET_WR", "0.55")
        ),
        help=(
            "Training target used to allocate the active strong-public practice "
            "quota. This is intentionally separate from the gate's individual "
            "safety floor."
        ),
    )
    p.add_argument(
        "--strong-public-practice-temperature",
        type=float,
        default=float(
            os.environ.get("PURE_RL_STRONG_PUBLIC_PRACTICE_TEMPERATURE", "0.35")
        ),
        help=(
            "Sampled-policy temperature for training-only games against the "
            "active gate roster; formal evaluation remains greedy."
        ),
    )
    p.add_argument(
        "--official-exploit-opponents",
        default=os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_OPPONENTS", ""),
        help=(
            "Comma-separated official baseline IDs whose training rows use a "
            "seat-paired lower-temperature mixture. Empty disables it."
        ),
    )
    p.add_argument(
        "--official-exploit-frac",
        type=float,
        default=float(os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_FRAC", "0")),
        help=(
            "Fraction of each selected official opponent's training seat-pairs "
            "that use --official-exploit-temperature."
        ),
    )
    p.add_argument(
        "--official-exploit-temperature",
        type=float,
        default=float(
            os.environ.get("PURE_RL_OFFICIAL_EXPLOIT_TEMPERATURE", "0.35")
        ),
        help="Sharpened sampling temperature for the selected official rows.",
    )
    p.add_argument("--base-checkpoint", type=Path, default=None)
    p.add_argument(
        "--initial-learner-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional new-lineage learner warm start. The protected rollout "
            "champion remains --base-checkpoint until normal promotion passes."
        ),
    )
    p.add_argument(
        "--initial-heldout-evidence",
        type=Path,
        default=None,
        help=(
            "Optional exact, trusted eval_vs_baselines report for the initial "
            "learner. It anchors learner retention without counting as a "
            "terminal gate or a completed RL iteration."
        ),
    )
    p.add_argument(
        "--initial-replay-shard",
        type=Path,
        action="append",
        default=[],
        help="Prior immutable replay shard(s) used only in the first two-shard window.",
    )
    p.add_argument("--smoke", action="store_true", help="Synthetic loop, no CABT")
    p.add_argument("--smoke-games", type=int, default=8)
    p.add_argument(
        "--heldout-games",
        type=int,
        default=None,
        help=(
            "Compatibility assertion only. With --active-gate-contract the "
            "trainer derives this count from the contract and rejects a mismatch."
        ),
    )
    p.add_argument(
        "--active-gate-contract",
        type=Path,
        default=None,
        help=(
            "Authoritative active-gate program. Required for production "
            "specialist runs; roster, allocation, seats, and package digests "
            "are derived from it."
        ),
    )
    p.add_argument(
        "--formal-holdout-contract",
        type=Path,
        default=None,
        help=(
            "Optional formal-only gate program. Training practice continues "
            "to use --active-gate-contract; only formal heldout evaluation "
            "uses this separately checksum-bound roster."
        ),
    )
    p.add_argument(
        "--r284-iteration-boundary-receipt",
        type=Path,
        default=None,
        help=(
            "Immutable owner receipt authorizing update-0 holdout deferral "
            "and the combo-off iteration-1 continuation."
        ),
    )
    p.add_argument(
        "--r327-evaluation-boundary-receipt",
        type=Path,
        default=None,
        help=(
            "Immutable derivative revision-25 receipt authorizing a 50-game-"
            "per-opponent formal contract and zero future research-control wave."
        ),
    )
    p.add_argument(
        "--research-control-registry",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_RESEARCH_CONTROL_REGISTRY"])
            if os.environ.get("PURE_RL_RESEARCH_CONTROL_REGISTRY")
            else None
        ),
        help=(
            "Versioned zero-gate-weight diagnostic-control roster. A fully "
            "committed passed active gate is appended only after the iteration "
            "commit, for use by a later lineage."
        ),
    )
    p.add_argument(
        "--measurement-decks",
        default=os.environ.get("PURE_RL_MEASUREMENT_DECKS", ""),
        help=(
            "Comma-separated candidate-side deck IDs used by promotion and "
            "formal heldout games. Empty keeps the full training deck pool."
        ),
    )
    p.add_argument(
        "--heldout-remotes",
        action="store_true",
        default=str(os.environ.get("PURE_RL_HELDOUT_REMOTES", "0")).lower()
        in ("1", "true", "yes", "on"),
        help="Use capability-verified additive remotes for the formal held-out gate.",
    )
    p.add_argument("--gate-wr", type=float, default=0.70)
    p.add_argument(
        "--heldout-per-opponent-floor",
        type=float,
        default=0.50,
        help="Minimum point win rate against every held-out opponent.",
    )
    p.add_argument("--promotion-games", type=int, default=80)
    p.add_argument("--promotion-workers", type=int, default=8)
    p.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.45,
        help="Candidate-vs-incumbent non-regression floor for the confidence bound.",
    )
    p.add_argument("--promotion-confidence", type=float, default=0.90)
    p.add_argument("--promotion-bootstrap-resamples", type=int, default=2000)
    p.add_argument(
        "--min-usable-game-frac",
        type=float,
        default=0.98,
        help="Fail the collection wave below this retained-trajectory fraction.",
    )
    p.add_argument(
        "--resume",
        choices=("auto", "never"),
        default="auto",
        help="Resume only from the append-only loop_state.json ledger.",
    )
    p.add_argument(
        "--start-iteration",
        type=int,
        default=None,
        help="Optional assertion for the ledger's next iteration; never rewinds it.",
    )
    p.add_argument(
        "--allow-clean-boundary-design-migration",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get(
                "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION", "0"
            )
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Permit one audited, append-only migration of operational batch/source "
            "fields, but only at a clean committed iteration boundary."
        ),
    )
    p.add_argument(
        "--boundary-design-migration-reason",
        default=None,
        help="Operator reason stored in the append-only design migration receipt.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game-timeout-s", type=int, default=600)
    p.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Skip dual-GPU leaf requirement (CI / laptop)",
    )
    p.add_argument(
        "--remote-worker-endpoints",
        default=None,
        help=(
            "Comma-separated whole-game farms (host:port). "
            f"Default production: {DEFAULT_REMOTE_ENDPOINTS}. "
            "Pass empty string to disable."
        ),
    )
    p.add_argument(
        "--no-remote-workers",
        action="store_true",
        help="Disable remote whole-game farms even if endpoints are up",
    )
    p.add_argument(
        "--leaf-eval",
        choices=("gpu-server", "cpu"),
        default="gpu-server",
        help="Local leaf inference mode for host workers",
    )
    p.add_argument(
        "--multi-env-per-worker",
        type=int,
        default=None,
        help=(
            "In-process LibcgMultiEnv battles per OS worker (default: off / 1). "
            "Also: POKEBOT_MULTI_ENV=1 (→4) or POKEBOT_MULTI_ENV_PER_WORKER=N. "
            "Keeps WorkerPool as default until set."
        ),
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=None,
        help=(
            "Pure-RL leaf coalesce window (ms). Default 0 for ~1.6M policy "
            "(does not change Hope-large RR config default of 4). "
            "Env: PURE_RL_LEAF_COALESCE_MS (scoped; ignores RR LEAF_SERVER_COALESCE_MS)."
        ),
    )
    p.add_argument(
        "--expert-rehearsal-every",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_EVERY", "0")),
        help="Run one saved ladder-policy rehearsal before every Nth RL iteration (0=off).",
    )
    p.add_argument(
        "--expert-rehearsal-before-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one expert complement pass before local iteration 0, then use normal cadence.",
    )
    p.add_argument(
        "--expert-rehearsal-force-before",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_FORCE_BEFORE", "-1")),
        help=(
            "Run one durable expert rehearsal before this exact iteration in "
            "addition to the normal cadence (-1=disabled)."
        ),
    )
    p.add_argument(
        "--expert-manifest",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_EXPERT_MANIFEST"])
            if os.environ.get("PURE_RL_EXPERT_MANIFEST")
            else None
        ),
        help="Feature manifest or atomic rolling-window pointer used for rehearsal.",
    )
    p.add_argument(
        "--expert-rehearsal-epochs",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_EPOCHS", "1")),
    )
    p.add_argument(
        "--expert-rehearsal-guide-loss-weight",
        type=float,
        default=(
            float(os.environ["PURE_RL_EXPERT_REHEARSAL_GUIDE_LOSS_WEIGHT"])
            if os.environ.get("PURE_RL_EXPERT_REHEARSAL_GUIDE_LOSS_WEIGHT")
            else None
        ),
        help=(
            "Optional guide weight used only by expert soft refreshes. "
            "Ordinary RL keeps --current-deck-guide-loss-weight."
        ),
    )
    p.add_argument(
        "--terminal-expert-rehearsal",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get(
            "PURE_RL_TERMINAL_EXPERT_REHEARSAL", "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "After the final durable RL commit, run the due expert soft "
            "refresh without collecting another iteration."
        ),
    )
    p.add_argument(
        "--expert-rehearsal-one-time-before",
        type=int,
        default=int(
            os.environ.get("PURE_RL_EXPERT_REHEARSAL_ONE_TIME_BEFORE", "-1")
        ),
        help=(
            "Use the one-time expert epoch override before this exact "
            "iteration; the ordinary cadence remains unchanged (-1=off)."
        ),
    )
    p.add_argument(
        "--expert-rehearsal-one-time-epochs",
        type=int,
        default=int(
            os.environ.get("PURE_RL_EXPERT_REHEARSAL_ONE_TIME_EPOCHS", "0")
        ),
        help=(
            "Expert epochs for --expert-rehearsal-one-time-before (0=off)."
        ),
    )
    p.add_argument(
        "--expert-rehearsal-lr",
        type=float,
        default=float(os.environ.get("PURE_RL_EXPERT_REHEARSAL_LR", "2e-5")),
    )
    p.add_argument(
        "--expert-rehearsal-batch-size",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_REHEARSAL_BATCH_SIZE", "8192")),
    )
    p.add_argument(
        "--rule-derivative-semantic-pack-completion",
        type=Path,
        default=(
            Path(os.environ["POKEBOT_RULE_DERIVATIVE_SEMANTIC_PACK_COMPLETION"])
            if os.environ.get("POKEBOT_RULE_DERIVATIVE_SEMANTIC_PACK_COMPLETION")
            else None
        ),
        help=(
            "Exact every-occurrence semantic-pack completion receipt. When set "
            "for a derivative parent, each scheduled expert refresh trains the "
            "semantic projection for the same number of epochs as the base pass."
        ),
    )
    p.add_argument(
        "--rule-derivative-semantic-refresh-lr",
        type=float,
        default=float(
            os.environ.get("POKEBOT_RULE_DERIVATIVE_SEMANTIC_REFRESH_LR", "3e-4")
        ),
    )
    p.add_argument(
        "--rule-derivative-semantic-block-decisions",
        type=int,
        default=int(
            os.environ.get(
                "POKEBOT_RULE_DERIVATIVE_SEMANTIC_BLOCK_DECISIONS", "4096"
            )
        ),
    )
    p.add_argument(
        "--expert-manifest-workers",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_MANIFEST_WORKERS", "32")),
        help=(
            "CPU workers for the one-time immutable expert-manifest scan; "
            "the verified split is reused across refresh epochs."
        ),
    )
    p.add_argument(
        "--expert-matchup-adapter-manifest",
        type=Path,
        default=(
            Path(os.environ["PURE_RL_EXPERT_MATCHUP_ADAPTER_MANIFEST"])
            if os.environ.get("PURE_RL_EXPERT_MATCHUP_ADAPTER_MANIFEST")
            else None
        ),
        help=(
            "Validated causal per-route corpus for an isolated adapter-only "
            "pass on each scheduled expert rehearsal boundary (off if unset)."
        ),
    )
    p.add_argument(
        "--expert-matchup-adapter-epochs",
        type=int,
        default=int(
            os.environ.get("PURE_RL_EXPERT_MATCHUP_ADAPTER_EPOCHS", "5")
        ),
    )
    p.add_argument(
        "--expert-matchup-adapter-lr",
        type=float,
        default=float(
            os.environ.get("PURE_RL_EXPERT_MATCHUP_ADAPTER_LR", "1e-4")
        ),
    )
    p.add_argument(
        "--expert-matchup-adapter-games-per-batch",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_EXPERT_MATCHUP_ADAPTER_GAMES_PER_BATCH", "4"
            )
        ),
    )
    p.add_argument(
        "--expert-matchup-adapter-max-decisions-per-batch",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_EXPERT_MATCHUP_ADAPTER_MAX_DECISIONS_PER_BATCH",
                "512",
            )
        ),
    )
    p.add_argument(
        "--expert-min-decisions",
        type=int,
        default=int(os.environ.get("PURE_RL_EXPERT_MIN_DECISIONS", "5000000")),
    )
    p.add_argument(
        "--expert-required-target",
        action="append",
        choices=EXPERT_REHEARSAL_TARGET_CHOICES,
        default=None,
        help=(
            "Require this exact expert-corpus target during rehearsal. "
            "Repeat for sparse, verified specialist corpora; omitted requires "
            "the complete canonical target set."
        ),
    )
    p.add_argument(
        "--continuous-learner-min-wr",
        type=float,
        default=float(os.environ.get("PURE_RL_CONTINUOUS_LEARNER_MIN_WR", "0.35")),
        help="Carry rejected learner candidates unless head-to-head WR falls below this floor.",
    )
    p.add_argument(
        "--continuous-learner-exact-regression-margin",
        type=float,
        default=float(
            os.environ.get(
                "PURE_RL_CONTINUOUS_LEARNER_EXACT_REGRESSION_MARGIN", "0.01"
            )
        ),
        help=(
            "Point-WR tolerance below the protected exact-gate best before a "
            "candidate counts as a material gate regression."
        ),
    )
    p.add_argument(
        "--continuous-learner-exact-regression-patience",
        type=int,
        default=int(
            os.environ.get(
                "PURE_RL_CONTINUOUS_LEARNER_EXACT_REGRESSION_PATIENCE", "2"
            )
        ),
        help=(
            "Consecutive material exact-gate regressions allowed before the "
            "rollout learner returns to the protected gate-best checkpoint."
        ),
    )
    p.add_argument(
        "--artifact-history-iterations",
        type=int,
        default=int(os.environ.get("PURE_RL_ARTIFACT_HISTORY_ITERATIONS", "5")),
        help="Retain this many completed iteration shards/checkpoints, plus protected identities.",
    )
    p.add_argument(
        "--continue-after-gate",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("PURE_RL_CONTINUE_AFTER_GATE", "0"))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Keep the first committed terminal-gate marker immutable, but continue "
            "training after an external handler has archived that exact checkpoint. "
            "The default remains terminal-on-pass."
        ),
    )
    p.add_argument(
        "--terminal-active-gate-id",
        default=str(os.environ.get("PURE_RL_TERMINAL_ACTIVE_GATE_ID", "")).strip(),
        help=(
            "When continuing past an archived earlier gate in the same lineage, "
            "stop cleanly once this active gate ID passes."
        ),
    )
    p.add_argument(
        "--terminal-gate-marker-name",
        default=str(
            os.environ.get("PURE_RL_TERMINAL_GATE_MARKER_NAME", "")
        ).strip(),
        help=(
            "Optional run-directory marker filename for this gate protocol. "
            "Use a new name when an older immutable gate marker must remain "
            "preserved for audit."
        ),
    )
    p.add_argument(
        "--minimum-terminal-iteration",
        type=int,
        default=int(
            os.environ.get("PURE_RL_MINIMUM_TERMINAL_ITERATION", "-1")
        ),
        help=(
            "Do not publish a terminal marker or stop on a passing gate before "
            "this completed iteration. Earlier passes remain committed research "
            "evidence and training continues."
        ),
    )
    p.add_argument(
        "--gate-boundary-pause-seconds",
        type=float,
        default=float(
            os.environ.get("PURE_RL_GATE_BOUNDARY_PAUSE_SECONDS", "30")
        ),
        help=(
            "After every completed iteration at or beyond the terminal floor, "
            "pause this many seconds before starting another collection. A "
            "terminal pass still stops immediately after its immutable marker "
            "is committed."
        ),
    )
    p.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=float(os.environ.get("PURE_RL_MIN_FREE_DISK_GB", "100")),
        help="Refuse to open a new large replay shard below this free-space floor.",
    )
    args = p.parse_args(argv)
    specialist = str(args.specialist_archetype or "").strip().lower()
    if args.mode == "specialist" and not specialist:
        p.error("--mode specialist requires --specialist-archetype")
    if args.mode == "core" and specialist:
        p.error("--specialist-archetype is valid only with --mode specialist")
    if bool(args.require_exact_training_seat_split) and args.mode != "specialist":
        p.error("--require-exact-training-seat-split requires --mode specialist")
    if bool(args.require_exact_training_seat_split) and int(args.games_per_iter) % 2:
        p.error("--require-exact-training-seat-split requires an even game count")
    if bool(args.population_own_models_only) and args.mode != "specialist":
        p.error("--population-own-models-only requires --mode specialist")
    if bool(args.population_own_models_only) and float(
        args.official_collect_frac
    ) != 0.0:
        p.error(
            "--population-own-models-only requires --official-collect-frac 0"
        )
    if not 0.0 <= float(args.official_collect_frac) <= 1.0:
        p.error("--official-collect-frac must be in [0, 1]")
    if int(args.research_control_games_per_iter) < 0:
        p.error("--research-control-games-per-iter cannot be negative")
    if bool(args.disable_research_control_measurement) and (
        args.r327_evaluation_boundary_receipt is None
    ):
        p.error(
            "--disable-research-control-measurement requires "
            "--r327-evaluation-boundary-receipt"
        )
    if int(args.minimum_terminal_iteration) >= 0 and not bool(
        args.continue_after_gate
    ):
        p.error(
            "--minimum-terminal-iteration requires --continue-after-gate"
        )
    if float(args.gate_boundary_pause_seconds) < 0:
        p.error("--gate-boundary-pause-seconds cannot be negative")
    if args.terminal_gate_marker_name and not re.fullmatch(
        r"[A-Za-z0-9_.-]+", str(args.terminal_gate_marker_name)
    ):
        p.error("--terminal-gate-marker-name must be a safe filename")
    if args.mode != "specialist" and float(args.official_collect_frac) > 0.0:
        p.error("--official-collect-frac is valid only with --mode specialist")
    if bool(args.official_adaptive_targeting) and (
        args.mode != "specialist" or float(args.official_collect_frac) <= 0.0
    ):
        p.error(
            "--official-adaptive-targeting requires specialist mode and a "
            "positive --official-collect-frac"
        )
    adaptive_min_share = float(args.official_adaptive_min_share)
    if not 0.0 <= adaptive_min_share <= 1.0 / len(OFFICIAL_BASELINE_IDS):
        p.error(
            "--official-adaptive-min-share must be in [0, "
            f"{1.0 / len(OFFICIAL_BASELINE_IDS):.2f}]"
        )
    if not 0.0 < float(args.official_adaptive_gap_power) <= 4.0:
        p.error("--official-adaptive-gap-power must be in (0, 4]")
    if not 0.0 < float(args.strong_public_practice_target_wr) <= 1.0:
        p.error("--strong-public-practice-target-wr must be in (0, 1]")
    if not 0.0 < float(args.strong_public_practice_temperature) <= 1.0:
        p.error("--strong-public-practice-temperature must be in (0, 1]")
    exploit_frac = float(args.official_exploit_frac)
    if not 0.0 <= exploit_frac <= 1.0:
        p.error("--official-exploit-frac must be in [0, 1]")
    exploit_temperature = float(args.official_exploit_temperature)
    if not 0.0 < exploit_temperature <= 1.0:
        p.error("--official-exploit-temperature must be in (0, 1]")
    exploit_raw = [
        value.strip().lower()
        for value in str(args.official_exploit_opponents or "").split(",")
        if value.strip()
    ]
    if len(exploit_raw) != len(set(exploit_raw)):
        p.error("--official-exploit-opponents contains duplicates")
    unknown_exploit = sorted(set(exploit_raw) - set(OFFICIAL_BASELINE_IDS))
    if unknown_exploit:
        p.error(
            "--official-exploit-opponents contains unknown official IDs: "
            + ",".join(unknown_exploit)
        )
    exploit_opponents = tuple(
        opponent_id
        for opponent_id in OFFICIAL_BASELINE_IDS
        if opponent_id in set(exploit_raw)
    )
    if bool(exploit_opponents) != bool(exploit_frac > 0.0):
        p.error(
            "--official-exploit-opponents and a positive "
            "--official-exploit-frac must be enabled together"
        )
    if exploit_opponents and (
        args.mode != "specialist" or float(args.official_collect_frac) <= 0.0
    ):
        p.error(
            "official exploit collection requires specialist mode and a positive "
            "--official-collect-frac"
        )
    for option, value in (
        ("--archetype-aux-loss-weight", args.archetype_aux_loss_weight),
        ("--opp-hand-loss-weight", args.opp_hand_loss_weight),
        ("--opp-remainder-loss-weight", args.opp_remainder_loss_weight),
        ("--lethal-threat-loss-weight", args.lethal_threat_loss_weight),
        ("--prize-race-loss-weight", args.prize_race_loss_weight),
    ):
        if float(value) < 0.0:
            p.error(f"{option} cannot be negative")
    guide_weight = float(args.alakazam_guide_loss_weight)
    if not math.isfinite(guide_weight) or guide_weight < 0.0:
        p.error(
            "--current-deck-guide-loss-weight must be finite and nonnegative"
        )
    if guide_weight > 0.0 and args.mode != "specialist":
        p.error(
            "--current-deck-guide-loss-weight is valid only for "
            "--mode specialist"
        )
    if guide_weight > 0.0 and not deck_guides.enabled():
        p.error(
            "nonzero --current-deck-guide-loss-weight requires an authorized "
            "POKEBOT_CURRENT_DECK_GUIDE and "
            "POKEBOT_CURRENT_DECK_GUIDE_TARGETS=1"
        )
    if args.tactical_outcome_loss_weight_override is not None:
        if float(args.tactical_outcome_loss_weight_override) < 0.0:
            p.error("--tactical-outcome-loss-weight-override cannot be negative")
        if (
            _effective_boundary_design_migration_reason(args)
            != "receipt_backed_teal_auxiliary_head_rebalance_v1"
        ):
            p.error(
                "--tactical-outcome-loss-weight-override requires the exact "
                "receipt-backed Teal auxiliary-head migration reason"
            )
    if guide_weight > 0.0 and deck_guides.selected_id() != specialist:
        p.error(
            "current-deck guide selector must equal --specialist-archetype"
        )
    setup_weight = float(args.setup_board_outcome_loss_weight)
    if not math.isfinite(setup_weight) or setup_weight < 0.0:
        p.error("--setup-board-outcome-loss-weight must be finite and nonnegative")
    combo_weight = float(args.combo_state_loss_weight)
    if not math.isfinite(combo_weight) or combo_weight < 0.0:
        p.error("--combo-state-loss-weight must be finite and nonnegative")
    # Combo-state CE weight 0.025 is required for Slowking, Marnie, and Slop Box
    # H10. Alakazam/Crustle H10 may still carry legacy zero from older
    # registries; authorize 0.025 there without forcing a healthy-run restart.
    combo_required_specialists = {
        "slowking",
        "marnie-s-grimmsnarl-ex",
        "teal-mask-ogerpon-ex",
    }
    combo_optional_h10_specialists = {"alakazam", "crustle"}
    if specialist in combo_required_specialists:
        if not math.isclose(
            combo_weight,
            COMBO_STATE_BASE_LOSS_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            p.error(
                f"{specialist} requires --combo-state-loss-weight 0.025 during "
                "ordinary RL and scheduled expert rehearsal"
            )
    elif specialist in combo_optional_h10_specialists:
        if combo_weight != 0.0 and not math.isclose(
            combo_weight,
            COMBO_STATE_BASE_LOSS_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            p.error(
                f"{specialist} combo-state loss weight must be 0.0 (legacy) or "
                "exactly 0.025"
            )
    elif combo_weight != 0.0:
        p.error(
            "--combo-state-loss-weight is authorized only for Slowking and "
            "H10 specialists that retain the combo-state head "
            "(Alakazam, Marnie, Crustle, Slop Box / teal-mask-ogerpon-ex)"
        )
    guide_training_mode = str(args.current_deck_guide_training_mode)
    if guide_training_mode not in GUIDE_TRAINING_MODES:
        p.error(
            "--current-deck-guide-training-mode has an unsupported value"
        )
    strategic_inputs = (
        args.current_deck_guide_curriculum_spec,
        args.current_deck_guide_head_role_map,
        args.current_deck_guide_curriculum_validation_receipt,
    )
    if guide_training_mode in GUIDE_STRATEGIC_TRAINING_MODES:
        if args.mode != "specialist":
            p.error(
                "a strategic --current-deck-guide-training-mode "
                "requires --mode specialist"
            )
        if not math.isclose(
            setup_weight,
            SETUP_BOARD_OUTCOME_BASE_LOSS_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            p.error(
                "strategic guide training requires "
                "--setup-board-outcome-loss-weight 0.025"
            )
        if any(value is None for value in strategic_inputs):
            p.error(
                "strategic guide training requires its curriculum spec, "
                "head-role map, and validation receipt"
            )
        try:
            assert_strategic_curriculum_receipt_contract(
                specialist_id=specialist,
                curriculum_spec=str(
                    args.current_deck_guide_curriculum_spec
                ),
                head_role_map=str(args.current_deck_guide_head_role_map),
                validation_receipt=str(
                    args.current_deck_guide_curriculum_validation_receipt
                ),
                expected_training_mode=guide_training_mode,
            )
        except (OSError, TypeError, ValueError) as exc:
            p.error(f"strategic curriculum receipt gate failed: {exc}")
    elif any(value is not None for value in strategic_inputs):
        p.error(
            "strategic curriculum artifacts require "
            "a strategic --current-deck-guide-training-mode"
        )
    adapter_epochs = int(args.dormant_matchup_adapter_epochs)
    if adapter_epochs < 0:
        p.error("--dormant-matchup-adapter-epochs cannot be negative")
    if float(args.dormant_matchup_adapter_lr) <= 0.0:
        p.error("--dormant-matchup-adapter-lr must be positive")
    if int(args.dormant_matchup_adapter_max_decisions_per_batch) <= 0:
        p.error(
            "--dormant-matchup-adapter-max-decisions-per-batch must be positive"
        )
    registered_matchup_ids = tuple(EXPERT_IDS)
    if adapter_epochs > 0:
        try:
            registered_matchup_ids = _registered_matchup_target_ids(
                args.initial_learner_checkpoint
            )
        except Exception as exc:
            p.error(f"cannot validate checkpoint matchup roster: {exc}")
    if adapter_epochs > 0 and not (
        args.mode == "specialist"
        and specialist in registered_matchup_ids
        and args.dormant_matchup_adapter_activation_receipt is not None
        and float(args.official_collect_frac) > 0.0
    ):
        p.error(
            "dormant matchup adapter training requires a registered specialist, "
            "active-gate practice, and a boundary authorization "
            f"(mode={args.mode!r}, specialist={specialist!r}, "
            f"registered={specialist in registered_matchup_ids}, "
            f"registered_count={len(registered_matchup_ids)}, "
            f"initial_checkpoint={args.initial_learner_checkpoint!s}, "
            "authorization="
            f"{args.dormant_matchup_adapter_activation_receipt is not None}, "
            f"official_collect_frac={float(args.official_collect_frac)})"
        )
    if adapter_epochs > 0 and bool(args.train_device_resident):
        p.error(
            "dormant matchup adapter training currently requires retained "
            "temporal GameSequence rows; disable --train-device-resident"
        )
    if float(args.train_device_resident_min_free_gib) < 2.0:
        p.error("--train-device-resident-min-free-gib must be at least 2 GiB")
    warmup_cap = int(args.train_warmup_max_decisions_per_batch)
    warmup_iterations = int(args.train_warmup_iterations)
    if warmup_cap < 0 or warmup_iterations < 0:
        p.error("training warmup cap and iteration count cannot be negative")
    if (warmup_cap == 0) != (warmup_iterations == 0):
        p.error(
            "--train-warmup-max-decisions-per-batch and "
            "--train-warmup-iterations must be enabled together"
        )
    try:
        _validate_fixed_cycle_configuration(args)
    except ValueError as exc:
        p.error(str(exc))
    args.specialist_archetype = specialist or None
    args.official_exploit_opponents = exploit_opponents
    return args


R241_FIXED_CYCLE_UPDATES = 10
R274_FIXED_CYCLE_UPDATES = 20
R241_FIXED_CYCLE_REHEARSAL_EVERY = 5
R241_FIXED_CYCLE_REHEARSAL_EPOCHS = 5
R274_BOOTSTRAP_EPOCHS = 25
R274_CANDIDATE_ID = "alakazam-new-list-direct-policy-r274"
R274_SUBMISSION_REQUEST_SCHEMA = "poke_bot.r274_rl_submission_request/v1"
R274_SUBMISSION_UPLOAD_SCHEMA = "poke_bot.r274_rl_submission_upload/v1"
R274_R195_RESEARCH_BASELINE_SCHEMA = (
    "poke_bot.r274_r195_submission_55378392_research_baseline/v1"
)
R274_R195_RESEARCH_OPPONENT_ID = "alakazam-r195-no-rtp-submission-55378392"
R274_R195_RESEARCH_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R274_R195_RESEARCH_MODEL_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R274_R195_RESEARCH_GAMES_MINIMUM = 128
R274_R195_RESEARCH_EACH_SEAT_MINIMUM = 64


def _fixed_cycle_updates(args: argparse.Namespace) -> int:
    """Return the optional fixed-cycle horizon without changing legacy runs."""

    try:
        updates = int(getattr(args, "fixed_cycle_updates", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("--fixed-cycle-updates must be an integer") from exc
    if updates < 0:
        raise ValueError("--fixed-cycle-updates cannot be negative")
    return updates


def _validate_fixed_cycle_configuration(args: argparse.Namespace) -> int:
    """Fail closed around r241's one exact direct-policy cycle.

    The terminal refresh is intentionally a boundary operation rather than an
    eleventh RL update.  Keeping this contract in the trainer protects it from
    a launcher accidentally combining the r241 selector with a generic
    cadence, force-refresh, or population-cycle option.
    """

    updates = _fixed_cycle_updates(args)
    receipt = getattr(args, "r241_peak_r195_preservation_receipt", None)
    receipt_sha256 = str(
        getattr(args, "r241_peak_r195_preservation_receipt_sha256", "") or ""
    ).strip()
    if (receipt is None) != (not receipt_sha256):
        raise ValueError(
            "--r241-peak-r195-preservation-receipt and its sha256 must be "
            "provided together"
        )
    handoff = getattr(args, "r274_bootstrap_handoff_receipt", None)
    handoff_sha256 = str(
        getattr(args, "r274_bootstrap_handoff_receipt_sha256", "") or ""
    ).strip()
    submission_boundary_dir = getattr(args, "r274_submission_boundary_dir", None)
    submission_poll_seconds = float(
        getattr(args, "r274_submission_boundary_poll_seconds", 15.0)
    )
    if not math.isfinite(submission_poll_seconds) or submission_poll_seconds <= 0.0:
        raise ValueError("--r274-submission-boundary-poll-seconds must be positive")
    research_receipt = getattr(args, "r274_r195_research_baseline_receipt", None)
    research_receipt_sha256 = str(
        getattr(args, "r274_r195_research_baseline_receipt_sha256", "") or ""
    ).strip()
    r280_pack = getattr(args, "r280_contiguous_expert_pack", None)
    r280_pack_receipt = getattr(
        args, "r280_contiguous_expert_pack_receipt", None
    )
    if (r280_pack is None) != (r280_pack_receipt is None):
        raise ValueError(
            "--r280-contiguous-expert-pack and its receipt must be supplied together"
        )
    if (research_receipt is None) != (not research_receipt_sha256):
        raise ValueError(
            "--r274-r195-research-baseline-receipt and its sha256 must be supplied together"
        )
    if (handoff is None) != (not handoff_sha256):
        raise ValueError(
            "--r274-bootstrap-handoff-receipt and its sha256 must be supplied together"
        )
    if handoff_sha256 and not re.fullmatch(r"sha256:[0-9a-f]{64}", handoff_sha256):
        raise ValueError("--r274-bootstrap-handoff-receipt-sha256 must be canonical")
    if updates == 0:
        if receipt is not None:
            raise ValueError(
                "the r241 peak-r195 preservation receipt is allowed only with "
                "--fixed-cycle-updates 10"
            )
        if handoff is not None:
            raise ValueError("the r274 bootstrap handoff requires --fixed-cycle-updates 25")
        if submission_boundary_dir is not None:
            raise ValueError(
                "the r274 submission boundary directory requires --fixed-cycle-updates 25"
            )
        if research_receipt is not None:
            raise ValueError(
                "the r274 r195 research baseline requires --fixed-cycle-updates 25"
            )
        if r280_pack is not None:
            raise ValueError(
                "the r280 contiguous refresh pack requires --fixed-cycle-updates 25"
            )
        return 0
    if updates not in {R241_FIXED_CYCLE_UPDATES, R274_FIXED_CYCLE_UPDATES}:
        raise ValueError(
            "--fixed-cycle-updates supports only the exact r241 10-update "
            "cycle or r274 25-update cycle"
        )
    if receipt is None:
        raise ValueError(
            "the r241 fixed cycle requires its checksum-pinned peak-r195 "
            "preservation receipt"
        )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt_sha256):
        raise ValueError(
            "--r241-peak-r195-preservation-receipt-sha256 must be canonical"
        )
    if int(args.iterations) != updates:
        raise ValueError(
            "--fixed-cycle-updates must exactly equal --iterations "
            f"({updates})"
        )
    if not bool(getattr(args, "terminal_expert_rehearsal", False)):
        raise ValueError(
            "the r241 fixed cycle requires --terminal-expert-rehearsal"
        )
    if int(args.expert_rehearsal_every) != R241_FIXED_CYCLE_REHEARSAL_EVERY:
        raise ValueError(
            "the r241 fixed cycle requires --expert-rehearsal-every "
            f"{R241_FIXED_CYCLE_REHEARSAL_EVERY}"
        )
    if int(args.expert_rehearsal_epochs) != R241_FIXED_CYCLE_REHEARSAL_EPOCHS:
        raise ValueError(
            "the r241 fixed cycle requires --expert-rehearsal-epochs "
            f"{R241_FIXED_CYCLE_REHEARSAL_EPOCHS}"
        )
    rehearsal_before_first = bool(
        getattr(args, "expert_rehearsal_before_first", False)
    )
    external_bootstrap = handoff is not None
    if external_bootstrap and updates != R274_FIXED_CYCLE_UPDATES:
        raise ValueError("the r274 bootstrap handoff is allowed only for the 25-update cycle")
    if updates == R241_FIXED_CYCLE_UPDATES and rehearsal_before_first:
        raise ValueError(
            "the r241 fixed cycle permits no expert rehearsal before iter 0"
        )
    if (
        updates == R274_FIXED_CYCLE_UPDATES
        and not external_bootstrap
        and not rehearsal_before_first
    ):
        raise ValueError(
            "the r274 fixed cycle requires its bootstrap before iteration 0"
        )
    if external_bootstrap and rehearsal_before_first:
        raise ValueError("the externally completed r274 bootstrap must not be repeated")
    if updates == R274_FIXED_CYCLE_UPDATES and submission_boundary_dir is None:
        raise ValueError(
            "the r274 fixed cycle requires --r274-submission-boundary-dir"
        )
    if updates == R274_FIXED_CYCLE_UPDATES and research_receipt is None:
        raise ValueError(
            "the r274 fixed cycle requires --r274-r195-research-baseline-receipt"
        )
    if updates == R274_FIXED_CYCLE_UPDATES and r280_pack is None:
        raise ValueError(
            "the r274 fixed cycle requires the reusable r280 contiguous expert pack"
        )
    if research_receipt_sha256 and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", research_receipt_sha256
    ):
        raise ValueError(
            "--r274-r195-research-baseline-receipt-sha256 must be canonical"
        )
    if updates != R274_FIXED_CYCLE_UPDATES and submission_boundary_dir is not None:
        raise ValueError(
            "the r274 submission boundary directory is valid only for the 25-update cycle"
        )
    if updates != R274_FIXED_CYCLE_UPDATES and research_receipt is not None:
        raise ValueError(
            "the r274 r195 research baseline is valid only for the 25-update cycle"
        )
    if updates != R274_FIXED_CYCLE_UPDATES and r280_pack is not None:
        raise ValueError(
            "the r280 contiguous refresh pack is valid only for the 25-update cycle"
        )
    if int(getattr(args, "expert_rehearsal_force_before", -1)) != -1:
        raise ValueError(
            "the r241 fixed cycle permits no forced expert rehearsal"
        )
    one_time_before = int(
        getattr(args, "expert_rehearsal_one_time_before", -1)
    )
    one_time_epochs = int(getattr(args, "expert_rehearsal_one_time_epochs", 0))
    if updates == R241_FIXED_CYCLE_UPDATES:
        if one_time_before != -1 or one_time_epochs != 0:
            raise ValueError(
                "the r241 fixed cycle permits no one-time expert rehearsal override"
            )
    elif external_bootstrap:
        if one_time_before != -1 or one_time_epochs != 0:
            raise ValueError(
                "the r274 bootstrap handoff forbids a second one-time bootstrap"
            )
    elif one_time_before != 0 or one_time_epochs != R274_BOOTSTRAP_EPOCHS:
        raise ValueError(
            "the r274 fixed cycle requires an exact 25-epoch one-time expert "
            "bootstrap before iteration 0"
        )
    if bool(getattr(args, "population_own_models_only", False)):
        raise ValueError(
            "the r241 fixed cycle cannot run a population closing rehearsal"
        )
    return updates


def _r274_submission_boundary_paths(
    root: Path, boundary_update: int
) -> tuple[Path, Path]:
    boundary_update = int(boundary_update)
    if boundary_update not in {1, 5, 10, 15, 20, 25}:
        raise RuntimeError(f"invalid r274 submission boundary: {boundary_update}")
    boundary_dir = Path(root).expanduser().resolve() / f"boundary-{boundary_update:05d}"
    return boundary_dir / "request.json", boundary_dir / "upload.json"


def _materialize_r274_submission_request(
    *,
    root: Path,
    boundary_update: int,
    checkpoint_identity: Mapping[str, Any],
    rehearsal_receipt_identity: Mapping[str, Any],
    owner_contract_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Create or revalidate the immutable trainer-to-uploader request."""

    request_path, _ = _r274_submission_boundary_paths(root, boundary_update)
    request = {
        "schema": R274_SUBMISSION_REQUEST_SCHEMA,
        "candidate_id": R274_CANDIDATE_ID,
        "boundary_update": int(boundary_update),
        "rl_updates_completed": int(boundary_update),
        "checkpoint": dict(checkpoint_identity),
        "expert_rehearsal_receipt": dict(rehearsal_receipt_identity),
        "owner_contract_sha256": str(owner_contract_sha256),
        "direct_policy": True,
        "rtp_enabled": False,
        "next_collection_started": False,
    }
    request["request_sha256"] = _canonical_digest(request)
    if request_path.is_file():
        try:
            existing = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"r274 submission request is unreadable: {request_path}"
            ) from exc
        if existing != request:
            raise RuntimeError(
                f"r274 submission request drift at boundary {boundary_update}"
            )
    else:
        _write_json_exclusive(request_path, request)
    return request_path, request


def _validate_r274_submission_upload(
    *,
    upload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable upload receipt that unlocks the next block."""

    receipt = dict(upload)
    expected_request_sha256 = str(request.get("request_sha256") or "")
    if (
        receipt.get("schema") != R274_SUBMISSION_UPLOAD_SCHEMA
        or receipt.get("status") not in {"submitted", "reconciled"}
        or receipt.get("candidate_id") != R274_CANDIDATE_ID
        or int(receipt.get("boundary_update", -1))
        != int(request.get("boundary_update", -2))
        or str(receipt.get("request_sha256") or "") != expected_request_sha256
        or dict(receipt.get("checkpoint") or {})
        != dict(request.get("checkpoint") or {})
        or receipt.get("direct_policy") is not True
        or receipt.get("rtp_enabled") is not False
    ):
        raise RuntimeError("r274 submission upload receipt does not match its request")
    try:
        remote_submission_id = int(receipt.get("remote_submission_id"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("r274 upload receipt lacks a remote submission id") from exc
    if remote_submission_id <= 0:
        raise RuntimeError("r274 upload receipt remote submission id must be positive")
    declared = str(receipt.get("receipt_sha256") or "")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if declared != _canonical_digest(unsigned):
        raise RuntimeError("r274 upload receipt digest mismatch")
    return receipt


def _load_r274_r195_research_baseline_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate the exact locally installed submission-55378392 opponent."""

    if _fixed_cycle_updates(args) != R274_FIXED_CYCLE_UPDATES:
        return {}
    path = Path(args.r274_r195_research_baseline_receipt).expanduser().resolve()
    identity = _r241_file_identity(path)
    if identity["digest"] != str(
        args.r274_r195_research_baseline_receipt_sha256
    ):
        raise RuntimeError("r274 r195 research baseline receipt digest mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("r274 r195 research baseline receipt is invalid") from exc
    if (
        receipt.get("schema") != R274_R195_RESEARCH_BASELINE_SCHEMA
        or receipt.get("status") != "installed_exact_local_research_baseline"
        or receipt.get("baseline_id") != R274_R195_RESEARCH_OPPONENT_ID
        or int(receipt.get("submission_id", -1)) != 55378392
        or receipt.get("source_bundle_sha256")
        != R274_R195_RESEARCH_BUNDLE_SHA256
        or receipt.get("model_sha256") != R274_R195_RESEARCH_MODEL_SHA256
        or receipt.get("direct_policy_only") is not True
        or receipt.get("rtp_enabled") is not False
        or int(receipt.get("training_games_minimum_each_update", -1))
        != R274_R195_RESEARCH_GAMES_MINIMUM
        or int(receipt.get("learner_first_minimum", -1))
        != R274_R195_RESEARCH_EACH_SEAT_MINIMUM
        or int(receipt.get("learner_second_minimum", -1))
        != R274_R195_RESEARCH_EACH_SEAT_MINIMUM
        or receipt.get("transport") != "singleton_play"
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("installed_content_sha256") or ""),
        )
    ):
        raise RuntimeError("r274 r195 research baseline contract changed")
    return {**receipt, "receipt": identity}


def _gate_pass_should_stop(
    *,
    fixed_cycle_updates: int,
    continue_after_gate: bool,
    terminal_target_reached: bool,
) -> bool:
    """Keep an r241 gate receipt auditable without turning it into a stop."""

    return int(fixed_cycle_updates) == 0 and (
        not bool(continue_after_gate) or bool(terminal_target_reached)
    )


def _assert_r241_precollection_refresh_binding(
    *,
    iteration: int,
    collection_behavior_digest: str,
    rehearsal_record: Optional[Mapping[str, Any]],
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]],
) -> None:
    """Bind each five-update collection boundary to its soft refresh."""

    if not r241_peak_r195_training_contract:
        return
    iteration = int(iteration)
    if iteration <= 0 or iteration % R241_FIXED_CYCLE_REHEARSAL_EVERY != 0:
        return
    record = dict(rehearsal_record or {})
    refreshed_digest = str(record.get("checkpoint_digest") or "")
    checkpoint_identity = dict(record.get("checkpoint_identity") or {})
    if (
        int(record.get("before_iteration", -1))
        != iteration
        or int(record.get("epochs", -1))
        != R241_FIXED_CYCLE_REHEARSAL_EPOCHS
        or not refreshed_digest.startswith("sha256:")
        or str(checkpoint_identity.get("digest") or refreshed_digest)
        != refreshed_digest
        or str(collection_behavior_digest) != refreshed_digest
    ):
        raise RuntimeError(
            f"fixed-cycle iter-{iteration} collection must use the exact checkpoint produced by "
            "the completed 5-epoch precollection refresh"
        )
    validate_r241_peak_r195_live_fusion_record(
        dict(record.get("peak_r195_live_fusion") or {}),
        contract=r241_peak_r195_training_contract,
        phase="expert_refresh",
        boundary_iteration=iteration,
    )


def _r241_precollection_refresh_due(
    iteration: int,
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]],
    fixed_cycle_updates: int = R241_FIXED_CYCLE_UPDATES,
) -> bool:
    """Refresh at each nonterminal five-update fixed-cycle boundary."""

    iteration = int(iteration)
    return bool(
        int(fixed_cycle_updates) > 0
        and iteration > 0
        and iteration < int(fixed_cycle_updates)
        and iteration % R241_FIXED_CYCLE_REHEARSAL_EVERY == 0
    )


def _assert_fixed_cycle_completion(
    run_dir: Path,
    state: Mapping[str, Any],
    *,
    updates: int,
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Prove that the fixed cycle ended at its exact non-collection boundary."""

    updates = int(updates)
    if updates <= 0:
        raise ValueError("fixed-cycle completion requires a positive update count")
    expected_last = updates - 1
    if (
        int(state.get("last_completed_iteration", -1)) != expected_last
        or int(state.get("next_iteration", -1)) != updates
    ):
        raise RuntimeError(
            "fixed cycle did not reach its exact final commit: "
            f"last={state.get('last_completed_iteration')!r} "
            f"next={state.get('next_iteration')!r} expected_last={expected_last}"
        )

    expected_names = [f"iter_{iteration:05d}.json" for iteration in range(updates)]
    commits_dir = Path(run_dir) / "commits"
    observed_names = sorted(path.name for path in commits_dir.glob("iter_*.json"))
    if observed_names != expected_names:
        raise RuntimeError(
            "fixed cycle requires exactly immutable commits "
            f"{expected_names[0]}..{expected_names[-1]}; observed={observed_names}"
        )
    for iteration, name in enumerate(expected_names):
        commit_path = commits_dir / name
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if (
            int(commit.get("last_completed_iteration", -1)) != iteration
            or int(commit.get("next_iteration", -1)) != iteration + 1
        ):
            raise RuntimeError(
                f"fixed cycle commit is inconsistent: {commit_path}"
            )

    history = list(state.get("history") or ())
    completed_history = [
        int(row.get("iteration", -1))
        for row in history
        if isinstance(row, Mapping) and row.get("completed") is True
    ]
    if len(history) != updates or completed_history != list(range(updates)):
        raise RuntimeError(
            "fixed cycle history is not exactly the completed update range: "
            f"history={completed_history} expected={list(range(updates))}"
        )

    for boundary_iteration in range(
        R241_FIXED_CYCLE_REHEARSAL_EVERY,
        updates,
        R241_FIXED_CYCLE_REHEARSAL_EVERY,
    ):
        scheduled_refresh = (
            Path(run_dir)
            / "rehearsals"
            / f"before_iter_{boundary_iteration:05d}.json"
        )
        if not scheduled_refresh.is_file():
            raise RuntimeError(
                "fixed cycle is missing its required 5-epoch refresh before "
                f"iter {boundary_iteration}"
            )
        try:
            refresh = json.loads(scheduled_refresh.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "fixed cycle cannot read its precollection refresh before "
                f"iter {boundary_iteration}"
            ) from exc
        if (
            int(refresh.get("before_iteration", -1)) != boundary_iteration
            or int(refresh.get("epochs", -1))
            != R241_FIXED_CYCLE_REHEARSAL_EPOCHS
        ):
            raise RuntimeError(
                f"fixed cycle before-iter-{boundary_iteration} refresh does "
                "not have the exact 5-epoch receipt"
            )
        if r241_peak_r195_training_contract:
            validate_r241_peak_r195_live_fusion_record(
                dict(refresh.get("peak_r195_live_fusion") or {}),
                contract=r241_peak_r195_training_contract,
                phase="expert_refresh",
                boundary_iteration=boundary_iteration,
            )
            collection_path = _collection_receipt_path(
                Path(run_dir), boundary_iteration
            )
            try:
                boundary_collection = json.loads(
                    collection_path.read_text(encoding="utf-8")
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "fixed cycle cannot verify its collection receipt for "
                    f"iter {boundary_iteration}"
                ) from exc
            _assert_r241_precollection_refresh_binding(
                iteration=boundary_iteration,
                collection_behavior_digest=str(
                    boundary_collection.get("checkpoint_digest") or ""
                ),
                rehearsal_record=refresh,
                r241_peak_r195_training_contract=(
                    r241_peak_r195_training_contract
                ),
            )

    next_iteration_artifacts = [
        _collection_receipt_path(Path(run_dir), updates),
        Path(run_dir) / "collection_plans" / f"iter_{updates:05d}.json",
        *_iteration_artifact_paths(Path(run_dir), updates),
    ]
    unexpected = [str(path) for path in next_iteration_artifacts if path.exists()]
    if unexpected:
        raise RuntimeError(
            "fixed cycle must not start iter_"
            f"{updates:05d} collection/artifacts: {unexpected}"
        )
    return {
        "updates_completed": updates,
        "last_completed_iteration": expected_last,
        "next_collection_started": False,
    }


def _scheduled_train_decision_cap(
    iteration: int,
    *,
    steady_cap: int,
    warmup_cap: int = 0,
    warmup_iterations: int = 0,
) -> tuple[int, str]:
    """Resolve the deterministic per-iteration optimizer decision cap."""
    iteration = int(iteration)
    steady_cap = int(steady_cap)
    warmup_cap = int(warmup_cap)
    warmup_iterations = int(warmup_iterations)
    if iteration < 0:
        raise ValueError("training iteration cannot be negative")
    if steady_cap <= 0:
        raise ValueError("steady training decision cap must be positive")
    if warmup_cap < 0 or warmup_iterations < 0:
        raise ValueError("training warmup values cannot be negative")
    if (warmup_cap == 0) != (warmup_iterations == 0):
        raise ValueError("training warmup cap and iteration count must be paired")
    if warmup_iterations and iteration < warmup_iterations:
        return warmup_cap, "head_focus_warmup"
    return steady_cap, "steady_state"


def _continuous_learner_carry_decision(
    *,
    candidate_safety_ok: bool,
    candidate_safety_reason: str,
    heldout_audit_ok: bool,
    promoted: bool,
) -> tuple[bool, str]:
    """Separate safe learner progress from protected-record selection."""
    if not bool(candidate_safety_ok):
        return False, str(candidate_safety_reason or "candidate_safety_failed")
    if not bool(heldout_audit_ok):
        return False, "heldout_contract_audit_failed"
    return (
        True,
        "promoted_safety_carry"
        if bool(promoted)
        else "continuous_learner_safety_carry",
    )


def _exact_gate_regression_streak(
    *,
    history: list[dict[str, Any]],
    current_gate_result: dict[str, Any],
    anchor_evidence: dict[str, Any],
    regression_margin: float,
) -> dict[str, Any]:
    """Count consecutive exact-gate regressions against one protected anchor.

    A different gate contract, malformed audit, score inside the tolerance, or
    a prior patience rollback ends the streak.  This makes rollback branch
    local: after returning to the anchor, a fresh branch receives its own
    exploration allowance instead of inheriting an old branch's failures.
    """
    margin = max(0.0, float(regression_margin))
    gate_id = str(current_gate_result.get("gate_id") or "").strip()
    anchor_gate_id = str(anchor_evidence.get("gate_id") or "").strip()
    try:
        anchor_wr = float(anchor_evidence.get("win_rate"))
    except (TypeError, ValueError):
        anchor_wr = -1.0
    if not gate_id or gate_id != anchor_gate_id or not 0.0 <= anchor_wr <= 1.0:
        return {
            "enabled": False,
            "reason": "missing_or_mismatched_exact_gate_anchor",
            "streak": 0,
            "regression_margin": margin,
        }

    streak = 0
    regressed_iterations: list[int] = []

    def _material_regression(result: dict[str, Any]) -> bool | None:
        if str(result.get("gate_id") or "").strip() != gate_id:
            return None
        audit = result.get("audit")
        if not isinstance(audit, dict) or not bool(audit.get("passed")):
            return None
        try:
            score = float(result.get("skill_weighted_wr"))
        except (TypeError, ValueError):
            return None
        return score < anchor_wr - margin - 1e-12

    current_regressed = _material_regression(current_gate_result)
    if current_regressed is not True:
        return {
            "enabled": True,
            "reason": (
                "within_anchor_margin"
                if current_regressed is False
                else "invalid_current_exact_gate_evidence"
            ),
            "streak": 0,
            "anchor_win_rate": anchor_wr,
            "candidate_win_rate": current_gate_result.get("skill_weighted_wr"),
            "regression_margin": margin,
            "regressed_iterations": [],
        }
    streak = 1
    regressed_iterations.append(int(current_gate_result.get("iteration", -1)))

    for row in reversed(history):
        if not isinstance(row, dict):
            break
        continuous = (row.get("promotion") or {}).get("continuous_learner")
        if isinstance(continuous, dict) and str(continuous.get("reason") or "") == (
            "exact_gate_regression_patience_exhausted"
        ):
            break
        prior = row.get("raw_heldout_gate")
        if not isinstance(prior, dict) or _material_regression(prior) is not True:
            break
        streak += 1
        regressed_iterations.append(int(row.get("iteration", -1)))

    return {
        "enabled": True,
        "reason": "material_exact_gate_regression",
        "streak": streak,
        "anchor_win_rate": anchor_wr,
        "candidate_win_rate": float(current_gate_result["skill_weighted_wr"]),
        "regression_margin": margin,
        "regressed_iterations": regressed_iterations,
    }


def _run_dir(run_name: str, *, smoke: bool = False) -> Path:
    # Smoke artifacts are intentionally outside the production lineage
    # namespace.  A canary may use the same human-readable run name, but it
    # must never occupy or replace production shards/checkpoints/markers.
    namespace = "pure_rl_smoke" if smoke else "pure_rl"
    d = paths.OUTPUTS_DIR / namespace / run_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "shards").mkdir(exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "metrics").mkdir(exist_ok=True)
    (d / "eval").mkdir(exist_ok=True)
    (d / "commits").mkdir(exist_ok=True)
    (d / "rehearsals").mkdir(exist_ok=True)
    (d / "artifact_receipts").mkdir(exist_ok=True)
    return d


LOOP_STATE_VERSION = 2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a mutable pointer/ledger atomically without a torn JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable JSON artifact and fail rather than overwrite it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(data)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}") from exc


def _write_json_exclusive_or_verify(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable JSON artifact, or verify an identical prior write.

    This is for restartable stages whose durable output may have been committed
    immediately before a later step failed.  It never replaces existing bytes.
    """
    path = Path(path)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"immutable artifact differs on resume: {path}")
        return
    _write_json_exclusive(path, payload)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _source_snapshot(root: Optional[Path] = None) -> dict[str, Any]:
    """Capture enough source identity to reproduce a dirty-tree experiment."""
    source_root = Path(root or ROOT).resolve()

    def _git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=source_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def _git_bytes(*args: str) -> bytes:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=source_root,
                check=False,
                capture_output=True,
                timeout=30,
            )
        except Exception:
            return b""
        return proc.stdout if proc.returncode == 0 else b""

    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    diff_bytes = _git_bytes("diff", "--binary", "HEAD")
    untracked_raw = _git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    untracked: list[dict[str, Any]] = []
    for raw_name in sorted(x for x in untracked_raw.split(b"\0") if x):
        name = os.fsdecode(raw_name)
        first_part = Path(name).parts[0] if Path(name).parts else ""
        if first_part in {"outputs", "data", ".pytest_cache"}:
            continue
        path = source_root / name
        if not path.is_file():
            continue
        untracked.append(
            {
                "path": name,
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )

    git_head = _git("rev-parse", "HEAD") or None
    tree_hash = hashlib.sha256()
    tree_hash.update((git_head or "no-git-head").encode("utf-8"))
    tree_hash.update(b"\0tracked-diff\0")
    tree_hash.update(diff_bytes)
    for row in untracked:
        tree_hash.update(b"\0untracked\0")
        tree_hash.update(str(row["path"]).encode("utf-8", errors="surrogateescape"))
        tree_hash.update(b"\0")
        tree_hash.update(str(row["sha256"]).encode("ascii"))

    # Some deployed copies are intentionally not Git worktrees.  In that
    # case, hash the executable Python/config source rather than silently
    # reporting the same empty identity for every build.
    fallback_files: list[dict[str, Any]] = []
    if git_head is None:
        candidates: list[Path] = []
        for dirname in ("poke_bot", "scripts"):
            base = source_root / dirname
            if base.is_dir():
                candidates.extend(x for x in base.rglob("*") if x.is_file())
        for name in ("requirements.txt", "pytest.ini"):
            path = source_root / name
            if path.is_file():
                candidates.append(path)
        for path in sorted(set(candidates)):
            if (
                "__pycache__" in path.parts
                or path.suffix in (".pyc", ".log")
                or path.name.startswith(".")
            ):
                continue
            rel = path.relative_to(source_root).as_posix()
            digest = _sha256_file(path)
            fallback_files.append(
                {"path": rel, "size": int(path.stat().st_size), "sha256": digest}
            )
            tree_hash.update(b"\0source-file\0")
            tree_hash.update(rel.encode("utf-8"))
            tree_hash.update(b"\0")
            tree_hash.update(digest.encode("ascii"))

    return {
        "git_head": git_head,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": f"sha256:{hashlib.sha256(diff_bytes).hexdigest()}",
        "untracked_files": untracked,
        "fallback_source_files": fallback_files,
        "source_tree_sha256": f"sha256:{tree_hash.hexdigest()}",
        "python": sys.version,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _canonical_digest(payload: Any) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _ticket_dormant_matchup_adapter_sequences(
    dataset: BootstrapDataset,
    *,
    active_gate: dict[str, Any],
    specialist_archetype: str,
    registered_specialist_ids: Sequence[str] = EXPERT_IDS,
) -> dict[str, Any]:
    """Authorize exact scheduler-labeled mirror/strong-public adapter rows.

    Diverse public games remain ordinary RL because their coarse metadata is
    not a sufficient package identity. Formal heldout and research controls
    are rejected explicitly even if a malformed caller places them in the
    replay window. Runtime/public-prefix routing is never enabled here.
    """

    specialist = str(specialist_archetype or "").strip().casefold()
    registered = {
        str(value).strip().casefold()
        for value in registered_specialist_ids
        if str(value).strip()
    }
    if specialist not in registered:
        raise ValueError(
            "dormant matchup adapter tickets require a canonical specialist"
        )
    gate_id = str(active_gate.get("id") or "").strip()
    roster = list(active_gate.get("roster") or [])
    if not gate_id or not roster:
        raise ValueError("adapter ticketing requires a non-empty active gate")
    roster_by_id = {
        str(row.get("opponent_id") or ""): {
            "archetype_id": str(row.get("archetype_id") or "").strip().casefold(),
            "content_digest": str(row.get("content_digest") or "").strip().lower(),
        }
        for row in roster
    }
    if "" in roster_by_id:
        raise ValueError("active gate roster contains an empty opponent identity")
    gate_digest = _canonical_digest(active_gate)

    eligible: list[tuple[Any, str, str, str]] = []
    excluded = {
        "wrong_acting_archetype": 0,
        "formal_or_research": 0,
        "unsupported_or_unproven": 0,
        "wrong_gate": 0,
    }
    for sequence in dataset.sequences:
        if str(sequence.archetype or "").strip().casefold() != specialist:
            excluded["wrong_acting_archetype"] += 1
            continue
        provenance = dict(sequence.target_provenance or {})
        group = str(provenance.get("opponent_training_group") or "")
        if (
            bool(provenance.get("formal_eval"))
            or group in {"formal_eval", RESEARCH_CONTROL_GROUP}
            or str(provenance.get("collect") or "") == "research_controls"
        ):
            excluded["formal_or_research"] += 1
            continue

        opponent_id = str(provenance.get("opponent_id") or "").strip()
        opponent_archetype = str(
            provenance.get("opponent_archetype_id")
            or sequence.opp_archetype
            or ""
        ).strip().casefold()
        package_digest = ""
        if bool(provenance.get("self_play")):
            # A same-specialist rollout is the exact mirror route. The opponent
            # checkpoint digest is its immutable package identity.
            if (
                str(provenance.get("collect") or "") == "self_play"
                and opponent_archetype == specialist
                and str(sequence.opp_archetype or "").strip().casefold()
                == specialist
            ):
                package_digest = str(
                    provenance.get("opponent_checkpoint_digest") or ""
                ).strip().lower()
                opponent_id = opponent_id or f"self:{specialist}"
            else:
                excluded["unsupported_or_unproven"] += 1
                continue
        elif group == STRONG_PUBLIC_PRACTICE_GROUP:
            if str(provenance.get("active_gate_id") or "") != gate_id:
                excluded["wrong_gate"] += 1
                continue
            expected = roster_by_id.get(opponent_id)
            package_digest = str(
                provenance.get("opponent_content_digest") or ""
            ).strip().lower()
            if not expected or (
                opponent_archetype != expected["archetype_id"]
                or str(sequence.opp_archetype or "").strip().casefold()
                != expected["archetype_id"]
                or package_digest != expected["content_digest"]
            ):
                excluded["unsupported_or_unproven"] += 1
                continue
        else:
            excluded["unsupported_or_unproven"] += 1
            continue

        route = training_route_for_archetype(opponent_archetype)
        if route == UNKNOWN_ROUTE or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", package_digest
        ):
            excluded["unsupported_or_unproven"] += 1
            continue
        if not str(sequence.episode_id or "") or int(sequence.seat) not in (0, 1):
            raise RuntimeError("eligible adapter sequence lacks exact episode/seat identity")
        eligible.append(
            (sequence, opponent_id, opponent_archetype, package_digest)
        )

    if not eligible:
        raise RuntimeError("no exact mirror/strong-public adapter rows were ticketed")
    corpus_digest = _canonical_digest(
        {
            "schema": "poke_bot.live_matchup_adapter_corpus/v1",
            "gate_digest": gate_digest,
            "episodes": sorted(
                (
                    str(sequence.episode_id),
                    int(sequence.seat),
                    opponent_id,
                    opponent_archetype,
                    package_digest,
                )
                for sequence, opponent_id, opponent_archetype, package_digest in eligible
            ),
        }
    )
    route_sequences: dict[str, int] = {}
    route_decisions: dict[str, int] = {}
    for sequence, opponent_id, opponent_archetype, package_digest in eligible:
        route = training_route_for_archetype(opponent_archetype)
        sequence.matchup_adapter_training_ticket = {
            "schema": TRAINING_TICKET_SCHEMA,
            "opponent_id": opponent_id,
            "package_digest": package_digest,
            "archetype_id": opponent_archetype,
            "route": route,
            "corpus_manifest_digest": corpus_digest,
            "gate_contract_digest": gate_digest,
            "episode_id": str(sequence.episode_id),
            "seat": int(sequence.seat),
            "acting_archetype_id": specialist,
        }
        for decision in sequence.decisions:
            decision.matchup_adapter_oracle_route = route
            # This phase is offline oracle supervision only. Never smuggle the
            # scheduler label into the runtime/public-prefix route.
            decision.matchup_adapter_public_route = UNKNOWN_ROUTE
        route_sequences[opponent_archetype] = (
            route_sequences.get(opponent_archetype, 0) + 1
        )
        route_decisions[opponent_archetype] = (
            route_decisions.get(opponent_archetype, 0) + len(sequence.decisions)
        )
    return {
        "schema": "poke_bot.live_matchup_adapter_ticketing/v1",
        "gate_id": gate_id,
        "gate_digest": gate_digest,
        "corpus_digest": corpus_digest,
        "ticketed_sequences": len(eligible),
        "route_sequences": route_sequences,
        "route_decisions": route_decisions,
        "excluded": excluded,
        "runtime_enabled": False,
    }


def _path_content_identity(path: Path) -> dict[str, Any]:
    """Content identity for a file or an installed baseline directory."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "size": int(path.stat().st_size),
            "digest": _sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError(f"identity path does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for child in sorted(x for x in path.rglob("*") if x.is_file()):
        if "__pycache__" in child.parts or child.suffix in (".pyc", ".log"):
            continue
        rows.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size": int(child.stat().st_size),
                "digest": _sha256_file(child),
            }
        )
    return {
        "kind": "directory",
        "path": str(path),
        "files": rows,
        "digest": _canonical_digest(rows),
    }


def _configured_prize_plan_h3_actor_provider(
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    """Return the immutable H3 boundary identity, or exact legacy ``None``.

    The cache is an optimizer-unit input rather than policy state.  Recording
    both receipt files in the design contract prevents a crash-recovered H3
    candidate from being accepted by a legacy unit (or vice versa).  The
    section is deliberately absent in legacy mode so merely deploying the
    staged source does not fork an already-running legacy lineage.
    """

    raw = (
        getattr(args, "prize_plan_h3_cache_receipt", None),
        str(getattr(args, "prize_plan_h3_cache_receipt_sha256", "") or ""),
        getattr(args, "prize_plan_h3_activation_receipt", None),
        str(getattr(args, "prize_plan_h3_activation_receipt_sha256", "") or ""),
    )
    live_raw = (
        getattr(args, "prize_plan_h3_live_sidecar", None),
        str(getattr(args, "prize_plan_h3_live_sidecar_sha256", "") or ""),
        getattr(args, "prize_plan_h3_live_scale_support", None),
        str(getattr(args, "prize_plan_h3_live_scale_support_sha256", "") or ""),
        getattr(args, "prize_plan_h3_live_c3_support", None),
        str(getattr(args, "prize_plan_h3_live_c3_support_sha256", "") or ""),
        getattr(args, "prize_plan_h3_live_contract", None),
        str(getattr(args, "prize_plan_h3_live_contract_sha256", "") or ""),
    )
    if any(raw) and any(live_raw):
        raise RuntimeError("static and live Prize-plan H3 providers are mutually exclusive")
    if any(live_raw):
        if not all(live_raw):
            raise RuntimeError(
                "Prize-plan H3 live sidecar/scale/c3/contract paths and SHA-256 "
                "values must be supplied as one complete boundary binding"
            )
        identities = [
            _path_content_identity(Path(live_raw[index]).expanduser().resolve())
            for index in (0, 2, 4, 6)
        ]
        expected = [str(live_raw[index]) for index in (1, 3, 5, 7)]
        if [item.get("digest") for item in identities] != expected:
            raise RuntimeError("Prize-plan H3 live input digest mismatch")
        return {
            "schema": "poke_bot.alakazam_prize_plan_v2_h3_actor_design/v1",
            "mode": "receipt_bound_live_h3_additive_at_clean_optimizer_boundary",
            "sidecar_checkpoint": identities[0],
            "h3_scale_support": identities[1],
            "c3_support": identities[2],
            "owner_contract": identities[3],
            "semantic_owner_goal_revision": 23,
            "activation_owner_goal_revision": 26,
            "coefficient": 0.025,
            "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
            "exact_legacy_baseline_computed_in_batch": True,
            "runtime_critic_calls": False,
        }
    if not any(raw):
        return None
    if not all(raw):
        raise RuntimeError(
            "Prize-plan H3 cache/activation paths and SHA-256 values must be "
            "supplied as one complete boundary binding"
        )
    cache = _path_content_identity(Path(raw[0]).expanduser().resolve())
    activation = _path_content_identity(Path(raw[2]).expanduser().resolve())
    if cache.get("digest") != raw[1] or activation.get("digest") != raw[3]:
        raise RuntimeError("Prize-plan H3 configured receipt digest mismatch")
    return {
        "schema": "poke_bot.alakazam_prize_plan_v2_h3_actor_design/v1",
        "mode": "receipt_bound_h3_additive_at_clean_optimizer_boundary",
        "cache_receipt": cache,
        "activation_receipt": activation,
        "semantic_owner_goal_revision": 23,
        "coefficient": 0.025,
        "h1_h6_h12_actor_coefficients": [0.0, 0.0, 0.0],
        "exact_legacy_baseline_computed_in_batch": True,
        "runtime_critic_calls": False,
    }


def _r241_file_identity(path: Path) -> dict[str, Any]:
    identity = _path_content_identity(path)
    if identity.get("kind") != "file":
        raise RuntimeError(f"r241 identity is not a file: {path}")
    return {
        "path": str(identity["path"]),
        "digest": str(identity["digest"]),
        "size_bytes": int(identity["size"]),
    }


def _r260_file_identity(path: Path | str) -> dict[str, Any]:
    """Return a canonical, rehashable FileIdentity for successor evidence."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"r260 identity must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _rehashed_r260_file_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Validate an exact FileIdentity against the sealed local bytes."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"r260 {label} identity must be an object")
    row = dict(value)
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise RuntimeError(f"r260 {label} identity key shape changed")
    path = str(row.get("path") or "")
    raw_digest = str(row.get("sha256") or "")
    digest = raw_digest.lower()
    size = row.get("size_bytes")
    if (
        not path
        or raw_digest != digest
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in digest[7:])
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise RuntimeError(f"r260 {label} identity is invalid")
    observed = _r260_file_identity(Path(path))
    if observed != {"path": path, "sha256": digest, "size_bytes": size}:
        raise RuntimeError(f"r260 {label} FileIdentity drifted")
    return observed


def _rehashed_r260_tactical_overlay_identity(
    value: Any, *, label: str
) -> dict[str, Any]:
    """Validate either historical or coverage-bearing tactical evidence.

    The tactical overlay validator returns the rehashable file identity plus
    its exact root/label/status coverage.  Scheduled rehearsal receipts retain
    that complete result, while older receipts carried only FileIdentity.
    """

    if not isinstance(value, Mapping):
        raise RuntimeError(f"r260 {label} identity must be an object")
    row = dict(value)
    file_keys = {"path", "sha256", "size_bytes"}
    if set(row) == file_keys:
        return _rehashed_r260_file_identity(row, label=label)
    if set(row) != file_keys | {"roots", "labels", "status_counts"}:
        raise RuntimeError(f"r260 {label} identity key shape changed")
    file_identity = _rehashed_r260_file_identity(
        {name: row[name] for name in file_keys}, label=label
    )
    from poke_bot.tactical_sequence_materialization import (
        validate_tactical_target_overlay,
    )

    try:
        observed = validate_tactical_target_overlay(
            Path(file_identity["path"]), minimum_roots=1024
        )
    except Exception as exc:
        raise RuntimeError(f"r260 {label} validation failed") from exc
    if observed != row:
        raise RuntimeError(f"r260 {label} coverage identity drifted")
    return observed


def _r260_tactical_cotrain_disabled(
    r260_own_deck_training_inputs: Mapping[str, Any],
) -> bool:
    """Resolve cotrain authority from both the override and checkpoint ABI."""

    explicit_disable = os.environ.get(
        "POKEBOT_R274_DISABLE_RL_TACTICAL_COTRAIN", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if explicit_disable and not r260_own_deck_training_inputs:
        raise RuntimeError(
            "RL tactical cotrain override is valid only for the r274 own-deck successor"
        )
    checkpoint_disables = bool(r260_own_deck_training_inputs) and not bool(
        r260_own_deck_training_inputs.get("tactical_sequence_enabled", False)
    )
    return bool(explicit_disable or checkpoint_disables)


def _r241_checkpoint_state_schema_digest(payload: Mapping[str, Any]) -> str:
    state = dict(
        payload.get("model_state_dict")
        or payload.get("base_model_state_dict")
        or {}
    )
    rows = [
        {
            "name": str(name),
            "shape": [int(value) for value in getattr(tensor, "shape", ())],
            "dtype": str(getattr(tensor, "dtype", "")),
        }
        for name, tensor in sorted(state.items())
    ]
    if not rows:
        raise RuntimeError("r241 peak-r195 checkpoint has no tensor schema")
    return _canonical_digest(rows)


def _assert_r241_initial_learner_anchor(
    requested_checkpoint: Optional[Path],
    *,
    sealed_parent: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Allow no initial override, or bytes identical to the sealed r195 anchor."""

    if requested_checkpoint is None:
        return None
    requested = _r241_file_identity(
        Path(requested_checkpoint).expanduser().resolve()
    )
    if (
        requested["digest"] != str(sealed_parent.get("digest") or "")
        or int(requested["size_bytes"])
        != int(sealed_parent.get("size_bytes", -1))
    ):
        raise RuntimeError(
            "r241 fixed cycle initial learner is not the sealed r195 anchor"
        )
    return requested


def _load_r260_own_deck_training_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the additive r260 evidence without changing the r195 auditor.

    The launcher supplies these only for the new successor.  Legacy r241
    fixtures remain valid with all six arguments absent; a partial set is a
    fail-closed configuration error.
    """

    def _path_arg_or_env(name: str, env_name: str) -> Path | None:
        value = getattr(args, name)
        if value is not None:
            return Path(value)
        text = os.environ.get(env_name, "").strip()
        return Path(text) if text else None

    def _digest_arg_or_env(name: str, env_name: str) -> str:
        value = str(getattr(args, name) or "").strip()
        return value or os.environ.get(env_name, "").strip()

    rows = (
        (
            "migration",
            _path_arg_or_env(
                "r241_own_deck_migration_receipt",
                "PURE_RL_R260_MIGRATION_RECEIPT",
            ),
            _digest_arg_or_env(
                "r241_own_deck_migration_receipt_sha256",
                "PURE_RL_R260_MIGRATION_RECEIPT_SHA256",
            ),
        ),
        (
            "canary_activation",
            _path_arg_or_env(
                "r241_own_deck_canary_activation_receipt",
                "PURE_RL_R260_CANARY_ACTIVATION_RECEIPT",
            ),
            _digest_arg_or_env(
                "r241_own_deck_canary_activation_receipt_sha256",
                "PURE_RL_R260_CANARY_ACTIVATION_RECEIPT_SHA256",
            ),
        ),
        (
            "sidecar",
            _path_arg_or_env(
                "r241_own_deck_sidecar_binding",
                "PURE_RL_R260_SIDECAR_BINDING",
            ),
            _digest_arg_or_env(
                "r241_own_deck_sidecar_binding_sha256",
                "PURE_RL_R260_SIDECAR_BINDING_SHA256",
            ),
        ),
        (
            "inzi_dataset",
            _path_arg_or_env(
                "r241_own_deck_inzi_dataset_binding",
                "PURE_RL_R260_INZI_DATASET_BINDING",
            ),
            _digest_arg_or_env(
                "r241_own_deck_inzi_dataset_binding_sha256",
                "PURE_RL_R260_INZI_DATASET_BINDING_SHA256",
            ),
        ),
    )
    supplied = [(path is not None, bool(str(digest or "").strip())) for _, path, digest in rows]
    if not any(any(pair) for pair in supplied):
        return {}
    if not all(path_present and digest_present for path_present, digest_present in supplied):
        raise RuntimeError("r260 own-deck receipt paths and sha256 values must be supplied together")
    tactical_path = _path_arg_or_env(
        "r274_expert_tactical_overlay", "PURE_RL_R260_TACTICAL_OVERLAY"
    )
    tactical_digest = _digest_arg_or_env(
        "r274_expert_tactical_overlay_sha256",
        "PURE_RL_R260_TACTICAL_OVERLAY_SHA256",
    )
    if (tactical_path is None) != (not tactical_digest):
        raise RuntimeError(
            "r274 expert tactical overlay path and sha256 must be supplied together"
        )
    if tactical_path is None:
        raise RuntimeError("r274 own-deck successor requires an expert tactical overlay")
    from poke_bot.tactical_sequence_materialization import (
        validate_tactical_target_overlay,
    )
    tactical_identity = validate_tactical_target_overlay(
        tactical_path, minimum_roots=1024
    )
    if tactical_identity["sha256"] != tactical_digest:
        raise RuntimeError("r274 expert tactical overlay digest mismatch")
    derivative_full_model = os.environ.get(
        "POKEBOT_DERIVATIVE_FULL_MODEL_BOOTSTRAP", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if _fixed_cycle_updates(args) == 0 and not derivative_full_model:
        raise RuntimeError("r260 own-deck inputs are allowed only for the r241 fixed cycle")
    from poke_bot import checkpoint as checkpoint_mod
    from poke_bot.r241_own_deck_successor import (
        R241OwnDeckSuccessorError,
        load_r260_owner_contract,
        validate_r260_canary_activation,
        validate_r260_inzi_dataset_binding,
        validate_r260_migration_receipt,
        validate_r260_sidecar_binding,
    )

    raw: dict[str, dict[str, Any]] = {}
    for name, path, expected_sha in rows:
        candidate = Path(path).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"r260 {name} receipt must be a regular non-symlink file")
        identity = _r241_file_identity(candidate.resolve())
        if identity["digest"] != str(expected_sha):
            raise RuntimeError(f"r260 {name} receipt digest mismatch")
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"r260 {name} receipt is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"r260 {name} receipt must be an object")
        raw[name] = payload
    try:
        historical_owner = dict(raw["migration"].get("owner_contract") or {})
        owner = load_r260_owner_contract(
            Path(str(historical_owner.get("path") or "")),
            expected_sha256=str(historical_owner.get("sha256") or ""),
            expected_clarification_revision=None,
        )
        migration_receipt = validate_r260_migration_receipt(raw["migration"], owner_contract=owner, require_local_child=True)
        canary = validate_r260_canary_activation(raw["canary_activation"], migration_receipt=migration_receipt, owner_contract=owner, require_local_checkpoint=True)
        sidecar = validate_r260_sidecar_binding(raw["sidecar"], owner_contract=owner)
        inzi = validate_r260_inzi_dataset_binding(raw["inzi_dataset"], sidecar_binding=sidecar, owner_contract=owner, require_local_dataset=True)
    except R241OwnDeckSuccessorError as exc:
        raise RuntimeError(f"r260 own-deck evidence validation failed: {exc}") from exc
    canary_checkpoint = dict(canary["canary_checkpoint"])
    if args.initial_learner_checkpoint is None:
        raise RuntimeError("r260 own-deck successor requires --initial-learner-checkpoint")
    initial_path = Path(args.initial_learner_checkpoint).expanduser().resolve()
    initial = checkpoint_mod.checkpoint_digest(initial_path)
    initial_payload = checkpoint_mod.load_checkpoint(initial_path, map_location="cpu")
    initial_model_config = dict(initial_payload.get("model_config") or {})
    tactical_free_model_config = dict(initial_model_config)
    for field in (
        "tactical_sequence_outcome_head_enabled",
        "tactical_sequence_outcome_route_enabled",
        "tactical_sequence_outcome_route_present",
        "tactical_sequence_outcome_route_runtime_enabled",
    ):
        tactical_free_model_config[field] = False
    tactical_free_payload = dict(initial_payload)
    tactical_free_payload["model_config"] = tactical_free_model_config
    tactical_free_payload["model_state_dict"] = {
        name: value
        for name, value in dict(initial_payload.get("model_state_dict") or {}).items()
        if not name.startswith("tactical_sequence_outcome_head.")
        and not name.startswith("tactical_sequence_outcome_route.")
    }
    handoff_path = getattr(args, "r274_bootstrap_handoff_receipt", None)
    handoff_sha256 = str(
        getattr(args, "r274_bootstrap_handoff_receipt_sha256", "") or ""
    ).strip()
    handoff: dict[str, Any] | None = None
    if handoff_path is not None:
        from poke_bot.r274_bootstrap_handoff import validate_handoff_receipt

        handoff_identity = _r241_file_identity(
            Path(handoff_path).expanduser().resolve()
        )
        if handoff_identity["digest"] != handoff_sha256:
            raise RuntimeError("r274 bootstrap handoff receipt digest mismatch")
        try:
            handoff = validate_handoff_receipt(
                handoff_path, expected_initial_checkpoint=initial_path
            )
        except Exception as exc:
            raise RuntimeError(f"r274 bootstrap handoff validation failed: {exc}") from exc
    elif initial != canary_checkpoint["sha256"] and not derivative_full_model:
        raise RuntimeError(
            "r260 initial learner does not match the runtime-enabled canary checkpoint"
        )
    if derivative_full_model and (
        initial_model_config.get("own_deck_ledger_enabled") is not True
        or initial_model_config.get("decision_fusion_enabled") is not True
    ):
        raise RuntimeError(
            "derivative streamed rehearsal requires OwnDeck and decision Fusion"
        )
    tactical_sequence_enabled = bool(
        initial_model_config.get("tactical_sequence_outcome_head_enabled", False)
    )
    daily_meta_sha256s: dict[str, str] = {}
    for day, receipt in sorted(
        dict(sidecar["daily_sidecar_meta_receipts"]).items()
    ):
        meta_path = Path(str(receipt.get("path") or "")).expanduser().resolve()
        meta_identity = _r241_file_identity(meta_path)
        if (
            meta_identity["digest"] != str(receipt.get("sha256") or "")
            or int(meta_identity["size_bytes"]) != int(receipt.get("size_bytes", -1))
        ):
            raise RuntimeError(f"r260 daily meta receipt mismatch: {day}")
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"r260 daily meta is invalid JSON: {day}") from exc
        embedded_digest = str(meta_payload.get("meta_sha256") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", embedded_digest):
            raise RuntimeError(f"r260 daily meta has invalid payload identity: {day}")
        daily_meta_sha256s[str(day)] = embedded_digest
    return {
        "owner_contract_sha256": owner.sha256,
        "migration_receipt_sha256": migration_receipt["receipt_sha256"],
        "canary_activation_receipt_sha256": canary["receipt_sha256"],
        "canary_checkpoint_sha256": canary_checkpoint["sha256"],
        "sidecar_binding_sha256": sidecar["binding_sha256"],
        "inzi_dataset_sha256": inzi["inzi_joined_dataset_sha256"],
        "inzi_dataset_identity": dict(inzi["inzi_joined_dataset"]),
        "inzi_sidecar_root": str(inzi["inzi_sidecar_root"]),
        "source_manifest_sha256": owner.source_manifest_sha256,
        "daily_meta_sha256s": daily_meta_sha256s,
        "runtime_gates": dict(canary["runtime_gates"]),
        "successor_model_config_sha256": _canonical_digest(
            dict(initial_payload.get("model_config") or {})
        ),
        "successor_state_schema_sha256": _r241_checkpoint_state_schema_digest(
            initial_payload
        ),
        "tactical_free_model_config_sha256": _canonical_digest(
            tactical_free_model_config
        ),
        "tactical_free_state_schema_sha256": _r241_checkpoint_state_schema_digest(
            tactical_free_payload
        ),
        "expert_tactical_overlay_identity": tactical_identity,
        "tactical_sequence_enabled": tactical_sequence_enabled,
        "derivative_streaming_only": derivative_full_model,
        "r274_bootstrap_handoff": (
            {
                "receipt_sha256": handoff["receipt_sha256"],
                "file_identity": dict(handoff["file_identity"]),
                "bootstrap_submission_checkpoint": dict(
                    handoff["bootstrap_submission_checkpoint"]
                ),
                "bootstrap_upload_receipt": dict(
                    handoff["bootstrap_upload_receipt"]
                ),
                "bootstrap_remote_submission_id": int(
                    handoff["bootstrap_remote_submission_id"]
                ),
                "activated_checkpoint": dict(handoff["activated_checkpoint"]),
                "impact": dict(handoff["impact"]),
            }
            if handoff is not None
            else None
        ),
    }


def _validate_r260_scheduled_rehearsal_receipt(
    value: Mapping[str, Any],
    *,
    expected_r260_inputs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fail closed on a resumed bounded host-refresh receipt."""
    receipt = dict(value)
    declared = receipt.pop("receipt_sha256", "")
    required = {
        "schema", "status", "owner_contract_sha256", "migration_receipt_sha256",
        "canary_activation_receipt_sha256", "sidecar_binding_sha256",
        "pre_checkpoint", "post_checkpoint", "index", "inzi_dataset",
        "index_provenance", "loss_weights", "rl_iteration_before_after",
        "bounded_batch_games", "full_window_device_resident", "sampled_keys",
        "gradient_reachability", "expert_tactical_overlay",
        "tactical_exact_root_count",
    }
    if (
        set(receipt) != required
        or receipt.get("schema") != "poke_bot.r260_scheduled_expert_rehearsal/v1"
        or receipt.get("status") != "passed"
        or declared != _canonical_digest(receipt)
    ):
        raise RuntimeError("r260 rehearsal receipt schema/digest drifted")
    for name in (
        "owner_contract_sha256",
        "migration_receipt_sha256",
        "canary_activation_receipt_sha256",
        "sidecar_binding_sha256",
    ):
        raw_digest = str(receipt.get(name) or "")
        digest = raw_digest.lower()
        if (
            raw_digest != digest
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in digest[7:])
        ):
            raise RuntimeError(f"r260 rehearsal receipt has invalid {name}")
        receipt[name] = digest
    pre = _rehashed_r260_file_identity(receipt["pre_checkpoint"], label="pre-checkpoint")
    post = _rehashed_r260_file_identity(receipt["post_checkpoint"], label="post-checkpoint")
    index = _rehashed_r260_file_identity(receipt["index"], label="streaming index")
    dataset = _rehashed_r260_file_identity(receipt["inzi_dataset"], label="Inzi dataset")
    tactical_overlay = _rehashed_r260_tactical_overlay_identity(
        receipt["expert_tactical_overlay"], label="expert tactical overlay"
    )
    if pre["sha256"] == post["sha256"]:
        raise RuntimeError("r260 rehearsal receipt did not create a child checkpoint")
    provenance = dict(receipt.get("index_provenance") or {})
    if set(provenance) != {
        "schema", "source_manifest_sha256", "daily_meta_sha256s"
    } or provenance.get("schema") != "poke_bot.r260_inzi_sidecar_index/v1":
        raise RuntimeError("r260 rehearsal receipt index provenance changed")
    raw_source_manifest = str(provenance.get("source_manifest_sha256") or "")
    source_manifest = raw_source_manifest.lower()
    daily_meta = provenance.get("daily_meta_sha256s")
    if (
        raw_source_manifest != source_manifest
        or len(source_manifest) != 71
        or not source_manifest.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in source_manifest[7:])
        or not isinstance(daily_meta, Mapping)
        or len(daily_meta) != 20
    ):
        raise RuntimeError("r260 rehearsal receipt index provenance is incomplete")
    normalized_daily: dict[str, str] = {}
    for day, digest in sorted(daily_meta.items()):
        day_text = str(day)
        raw_digest_text = str(digest or "")
        digest_text = raw_digest_text.lower()
        if (
            len(day_text) != 10
            or raw_digest_text != digest_text
            or len(digest_text) != 71
            or not digest_text.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in digest_text[7:])
        ):
            raise RuntimeError("r260 rehearsal receipt daily provenance is invalid")
        normalized_daily[day_text] = digest_text
    if len(normalized_daily) != 20:
        raise RuntimeError("r260 rehearsal receipt daily provenance is not unique")
    try:
        from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex

        R260InziSidecarIndex(
            Path(index["path"]),
            source_manifest_sha256=source_manifest,
            daily_meta_sha256s=normalized_daily,
        ).assert_verified(
            expected_source_manifest_sha256=source_manifest,
            daily_meta_sha256s=normalized_daily,
        )
    except Exception as exc:
        raise RuntimeError("r260 rehearsal receipt streaming index drifted") from exc
    tactical_expected = (
        "tactical_sequence_outcome" in dict(receipt.get("loss_weights") or {})
        if expected_r260_inputs is None
        else bool(expected_r260_inputs.get("tactical_sequence_enabled", True))
    )
    expected_loss_weights = {
        "visible_tutor_completion": 0.025,
        "terminal_conversion": 0.025,
    }
    if tactical_expected:
        expected_loss_weights["tactical_sequence_outcome"] = 0.025
    if receipt["loss_weights"] != expected_loss_weights:
        raise RuntimeError("r260 rehearsal receipt loss profile drifted")
    bounds = receipt["rl_iteration_before_after"]
    if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] != bounds[1] or int(receipt["bounded_batch_games"]) <= 0 or receipt["full_window_device_resident"] is not False:
        raise RuntimeError("r260 rehearsal receipt boundedness/iteration drifted")
    required_prefixes = {"own_deck_ledger_adapter.", "visible_tutor_completion_head.", "terminal_conversion_head.", "visible_tutor_completion_route.", "terminal_conversion_route.", "decision_fusion."}
    if tactical_expected:
        required_prefixes.add("tactical_sequence_outcome_head.")
    if tactical_expected and expected_r260_inputs is not None and expected_r260_inputs.get(
        "r274_bootstrap_handoff"
    ):
        required_prefixes.add("tactical_sequence_outcome_route.")
    if (
        set(receipt["gradient_reachability"] or {}) != required_prefixes
        or not all(receipt["gradient_reachability"].values())
        or not receipt["sampled_keys"]
    ):
        raise RuntimeError("r260 rehearsal receipt lacks finite gradient/key evidence")
    if tactical_expected and int(receipt["tactical_exact_root_count"]) < 1024:
        raise RuntimeError("r260 rehearsal receipt lacks tactical root coverage")
    if (
        not tactical_expected
        and int(receipt["tactical_exact_root_count"]) != 0
        and int(receipt["tactical_exact_root_count"]) < 1024
    ):
        raise RuntimeError(
            "tactical-free rehearsal receipt has incomplete shadow target coverage"
        )
    for key in receipt["sampled_keys"]:
        if (
            not isinstance(key, list)
            or len(key) != 4
            or not isinstance(key[0], str)
            or key[1] not in (0, 1)
            or isinstance(key[2], bool)
            or not isinstance(key[2], int)
            or not isinstance(key[3], str)
        ):
            raise RuntimeError("r260 rehearsal receipt sampled key shape changed")
    if expected_r260_inputs is not None:
        expected = dict(expected_r260_inputs)
        for name in (
            "owner_contract_sha256",
            "migration_receipt_sha256",
            "canary_activation_receipt_sha256",
            "sidecar_binding_sha256",
        ):
            if receipt[name] != str(expected.get(name) or "").lower():
                raise RuntimeError(f"r260 rehearsal receipt changed {name}")
        if dataset != dict(expected.get("inzi_dataset_identity") or {}):
            raise RuntimeError("r260 rehearsal receipt Inzi dataset identity drifted")
        if tactical_overlay != dict(
            expected.get("expert_tactical_overlay_identity") or {}
        ):
            raise RuntimeError("r260 rehearsal receipt tactical overlay drifted")
        expected_daily = {
            str(day): str(digest).lower()
            for day, digest in sorted(
                dict(expected.get("daily_meta_sha256s") or {}).items()
            )
        }
        if (
            source_manifest != str(expected.get("source_manifest_sha256") or "").lower()
            or normalized_daily != expected_daily
        ):
            raise RuntimeError("r260 rehearsal receipt sidecar provenance drifted")
    receipt["pre_checkpoint"] = pre
    receipt["post_checkpoint"] = post
    receipt["index"] = index
    receipt["inzi_dataset"] = dataset
    receipt["expert_tactical_overlay"] = tactical_overlay
    receipt["index_provenance"] = {
        "schema": "poke_bot.r260_inzi_sidecar_index/v1",
        "source_manifest_sha256": source_manifest,
        "daily_meta_sha256s": normalized_daily,
    }
    receipt["receipt_sha256"] = declared
    return receipt


def _load_r241_peak_r195_training_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve checkpoint-derived v2 evidence into a training-only sidecar."""

    if _fixed_cycle_updates(args) == 0:
        return {}
    from poke_bot import checkpoint as checkpoint_mod
    from poke_bot import r241_checkpoint_receipts
    from poke_bot.strategic_losses import canonical_expanded_loss_weights

    raw_receipt_path = Path(
        args.r241_peak_r195_preservation_receipt
    ).expanduser()
    if raw_receipt_path.is_symlink() or not raw_receipt_path.is_file():
        raise RuntimeError(
            "r241 peak-r195 preservation receipt must be a regular non-symlink file"
        )
    receipt_path = raw_receipt_path.resolve()
    receipt_identity = _r241_file_identity(receipt_path)
    if receipt_identity["digest"] != str(
        args.r241_peak_r195_preservation_receipt_sha256
    ):
        raise RuntimeError("r241 peak-r195 preservation receipt digest mismatch")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("r241 peak-r195 preservation receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("r241 peak-r195 preservation receipt must be an object")
    if (
        receipt.get("schema") != R241_PEAK_R195_PRESERVATION_RECEIPT_SCHEMA
        or int(receipt.get("revision", -1)) != 241
        or int(receipt.get("owner_clarification_revision", -1))
        != R241_PEAK_R195_CLARIFICATION_REVISION
        or receipt.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("derived_not_self_asserted") is not True
    ):
        raise RuntimeError(
            "r241 peak-r195 preservation receipt is not derived v2/r251 evidence"
        )

    parent = dict(receipt.get("parent") or {})
    parent_path = Path(
        str(parent.get("checkpoint") or parent.get("path") or "")
    ).expanduser().resolve()
    parent_identity = _r241_file_identity(parent_path)
    if (
        parent_identity["digest"] != str(parent.get("sha256") or "")
        or int(parent_identity["size_bytes"])
        != int(parent.get("size_bytes", -1))
    ):
        raise RuntimeError("r241 preservation receipt parent identity changed")
    requested_parent = (
        Path(args.base_checkpoint).expanduser().resolve()
        if args.base_checkpoint is not None
        else None
    )
    if requested_parent is not None and requested_parent != parent_path:
        raise RuntimeError("r241 fixed cycle base checkpoint is not the sealed parent")
    # The r260 successor is validated separately below against its own
    # zero-safe migration receipt.  The legacy preservation auditor continues
    # to audit this r195 parent; it must not pretend the additive child is an
    # r195 byte-identical initial learner.
    if getattr(args, "r241_own_deck_migration_receipt", None) is None:
        _assert_r241_initial_learner_anchor(
            (
                Path(getattr(args, "initial_learner_checkpoint"))
                if getattr(args, "initial_learner_checkpoint", None) is not None
                else None
            ),
            sealed_parent=parent_identity,
        )

    adapter = dict(receipt.get("matchup_adapter") or {})
    learner_tree_binding = dict(adapter.get("learner_matchup_tree") or {})
    h10_tree_binding = dict(receipt.get("h10_marnie_matchup_tree") or {})
    learner_tree_path = Path(
        str(learner_tree_binding.get("path") or "")
    ).expanduser().resolve()
    h10_tree_path = Path(
        str(h10_tree_binding.get("path") or "")
    ).expanduser().resolve()
    cg_root_raw = os.environ.get("CG_LIB_PATH", "").strip()
    if not cg_root_raw:
        raise RuntimeError("r241 checkpoint-derived validation requires CG_LIB_PATH")
    cg_root = Path(cg_root_raw).expanduser().resolve()
    historical_direct = dict(receipt.get("direct_environment") or {})
    historical_cg_root = Path(
        str(historical_direct.get("cg_lib_path") or "")
    ).expanduser().resolve()
    historical_environment = dict(os.environ)
    historical_environment["CG_LIB_PATH"] = str(historical_cg_root)
    try:
        current_library = _r241_file_identity(cg_root / "cg" / "libcg.so")
        for forbidden_name in ("POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"):
            if str(os.environ.get(forbidden_name) or "").strip():
                raise RuntimeError(
                    f"r274 live simulator uses forbidden {forbidden_name}"
                )
        if (
            str(os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER") or "0")
            not in {"0", "false", "False"}
            or str(os.environ.get("POKEBOT_SEARCH_MODE") or "policy") != "policy"
            or str(os.environ.get("POKEBOT_SUBMISSION_SEARCH_DISABLE") or "1")
            not in {"1", "true", "True"}
        ):
            raise RuntimeError("r274 live learner is not direct-policy/RTP-off")
        validated_receipt = (
            r241_checkpoint_receipts.validate_peak_r195_preservation_receipt(
                receipt_path=receipt_path,
                parent_checkpoint=parent_path,
                learner_matchup_tree=learner_tree_path,
                h10_matchup_tree=h10_tree_path,
                official_cg_root=historical_cg_root,
                environment=historical_environment,
            )
        )
    except r241_checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise RuntimeError(
            f"r241 checkpoint-derived preservation validation failed: {exc}"
        ) from exc
    if validated_receipt != receipt:
        raise RuntimeError("r241 validated preservation receipt changed in memory")
    historical_library = dict(
        historical_direct.get("linux_x86_64_library") or {}
    )
    if (
        current_library["digest"] != historical_library.get("sha256")
        or current_library["size_bytes"]
        != int(historical_library.get("size_bytes", -1))
    ):
        raise RuntimeError("r274 live libcg bytes differ from the sealed r195 parent")

    if (
        str(args.mode) != "specialist"
        or str(args.specialist_archetype or "").casefold() != "alakazam"
        or int(args.games_per_iter) != 8_196
        or bool(args.require_exact_training_seat_split) is not True
        or not math.isclose(
            float(config.PURE_RL.self_play_frac),
            1_024 / 8_196,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or float(args.official_collect_frac) != 0.50
        or int(args.research_control_games_per_iter) != 1_000
    ):
        raise RuntimeError(
            "r241 requires Alakazam with exactly 8196 games, 1024 self-play, "
            "7172 established-public, and exact training-seat parity"
        )
    r241_collection_plan = _planned_collection_group_counts(
        games_per_iteration=int(args.games_per_iter),
        self_play_fraction=float(config.PURE_RL.self_play_frac),
        strong_public_fraction_of_public=float(args.official_collect_frac),
        research_control_games=int(args.research_control_games_per_iter),
    )
    if r241_collection_plan != {
        "self_play": 1_024,
        STRONG_PUBLIC_PRACTICE_GROUP: 4_586,
        "diverse_public": 2_586,
    }:
        raise RuntimeError(f"r241 collection split changed: {r241_collection_plan}")

    public_mix = dict(receipt.get("public_mix") or {})
    public_bindings = (
        ("active_gate_contract", args.active_gate_contract),
        ("frozen_specialist_registry", args.frozen_specialist_registry),
        ("research_control_registry", args.research_control_registry),
    )
    for name, requested_path in public_bindings:
        binding = dict(public_mix.get(name) or {})
        bound_path = Path(str(binding.get("path") or "")).expanduser().resolve()
        if requested_path is None:
            raise RuntimeError(f"r241 preservation receipt lacks requested {name}")
        identity = _r241_file_identity(bound_path)
        requested_matches_historical = (
            bound_path == Path(requested_path).expanduser().resolve()
        )
        successor_public_overlay = (
            args.r274_bootstrap_handoff_receipt is not None
            and name in {"active_gate_contract", "frozen_specialist_registry"}
        )
        if (
            (not requested_matches_historical and not successor_public_overlay)
            or identity["digest"] != str(binding.get("sha256") or "")
            or identity["size_bytes"] != int(binding.get("size_bytes", -1))
        ):
            raise RuntimeError(f"r241 preservation receipt changed {name}")
    if (
        public_mix.get("established_diverse_public_mix_preserved") is not True
        or public_mix.get("research_control_phase_preserved") is not True
        or float(public_mix.get("official_collect_fraction", -1.0)) != 0.50
        or int(public_mix.get("research_control_games_per_iter", -1)) != 1_000
    ):
        raise RuntimeError("r241 preservation receipt changed the public mix")

    owner_binding = dict(receipt.get("contract") or {})
    owner_path = Path(str(owner_binding.get("path") or ""))
    owner_identity = _r241_file_identity(owner_path.expanduser().resolve())
    if (
        owner_identity["digest"]
        != R241_R195_HISTORICAL_OWNER_CONTRACT_SHA256
        or owner_identity["digest"] != str(owner_binding.get("sha256") or "")
        or owner_identity["size_bytes"] != int(owner_binding.get("size_bytes", -1))
    ):
        raise RuntimeError("r241 typed owner-source identity changed")

    migration = dict(receipt.get("adapter_slot_migration") or {})
    if (
        migration.get("status") != "no_slot_change"
        or migration.get("existing_slots_byte_immutable") is not True
        or int(migration.get("retained_slot_count", -1)) != 20
        or list(migration.get("new_slots") or [])
        or list(migration.get("new_slot_proofs") or [])
        or migration.get("candidate_slot_registry_sha256")
        != migration.get("parent_slot_registry_sha256")
    ):
        raise RuntimeError(
            "r241 fixed cycle requires the unchanged 20-slot E60 adapter roster"
        )

    parent_payload = checkpoint_mod.load_checkpoint(parent_path, map_location="cpu")
    model_config = dict(parent_payload.get("model_config") or {})
    state_dict = dict(parent_payload.get("model_state_dict") or {})
    model_config_digest = _canonical_digest(model_config)
    state_schema_digest = _r241_checkpoint_state_schema_digest(parent_payload)
    trainable_parameter_count = sum(
        int(tensor.numel())
        for _name, tensor in state_dict.items()
        if hasattr(tensor, "numel")
    )
    if trainable_parameter_count != 10_750_146:
        raise RuntimeError("r241 peak-r195 serialized model profile changed")

    heads = dict(receipt.get("heads") or {})
    checkpoint_audit = dict(receipt.get("checkpoint_audit") or {})
    role_binding = dict(checkpoint_audit.get("head_role_map") or {})
    role_path = Path(str(role_binding.get("path") or "")).expanduser().resolve()
    requested_role_path = Path(
        str(args.current_deck_guide_head_role_map or "")
    ).expanduser().resolve()
    role_identity = _r241_file_identity(role_path)
    requested_role_identity = _r241_file_identity(requested_role_path)
    if (
        role_identity["digest"] != str(role_binding.get("sha256") or "")
        or int(role_identity["size_bytes"])
        != int(role_binding.get("size_bytes", -1))
        or requested_role_identity["digest"] != role_identity["digest"]
        or requested_role_identity["size_bytes"] != role_identity["size_bytes"]
    ):
        raise RuntimeError("r241 peak-r195 head-role map identity changed")
    try:
        role_map = json.loads(role_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("r241 head-role map is invalid JSON") from exc
    role_heads = dict(role_map.get("heads") or {})
    if (
        tuple(role_map.get("canonical_learned_decision_sources") or ())
        != R241_PEAK_R195_ACTIVE_SOURCES
        or set(role_heads) != set(R241_PEAK_R195_ACTIVE_SOURCES)
        or any(
            dict(role_heads[name]).get("trainable") is not True
            or dict(role_heads[name]).get("enters_decision_fusion") is not True
            or dict(role_heads[name]).get("route_id") != f"{name}-route"
            for name in R241_PEAK_R195_ACTIVE_SOURCES
        )
    ):
        raise RuntimeError("r241 head-role map no longer names the exact 18 routes")
    if (
        int(heads.get("architecture_present_head_count", -1)) != 19
        or int(heads.get("non_combo_head_count", -1)) != 18
        or int(heads.get("non_combo_route_count", -1)) != 18
        or set(heads.get("physical_head_names") or ())
        != set(R241_PEAK_R195_PHYSICAL_SOURCES)
        or set(heads.get("active_non_combo_head_names") or ())
        != set(R241_PEAK_R195_ACTIVE_SOURCES)
        or set(heads.get("active_non_combo_route_names") or ())
        != set(R241_PEAK_R195_ACTIVE_SOURCES)
        or heads.get("every_non_combo_head_trainable") is not True
        or heads.get("every_non_combo_fusion_route_enabled") is not True
        or dict(heads.get("combo_state") or {}).get("head_present") is not True
        or dict(heads.get("combo_state") or {}).get("physical_route_present")
        is not True
        or float(
            dict(heads.get("combo_state") or {}).get("loss_weight", -1.0)
        )
        != 0.0
        or dict(heads.get("combo_state") or {}).get("route_enabled") is not False
    ):
        raise RuntimeError("r241 peak-r195 head/route preservation proof changed")

    trainer = dict(receipt.get("trainer") or {})
    expected_ordinary = {
        "value": 1.0,
        "archetype": 0.05,
        "opponent_hand": 0.05,
        "opponent_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "setup_board_outcome": 0.025,
        "combo_state": 0.0,
        "guide": 0.05,
    }
    expected_refresh = {**expected_ordinary, "guide": 0.0}
    expected_preserved_args = (
        "--archetype-aux-loss-weight",
        "0.05",
        "--opp-hand-loss-weight",
        "0.05",
        "--opp-remainder-loss-weight",
        "0.05",
        "--lethal-threat-loss-weight",
        "0.025",
        "--prize-race-loss-weight",
        "0.025",
        "--setup-board-outcome-loss-weight",
        "0.025",
    )
    if tuple(trainer.get("r195_non_combo_arguments") or ()) != expected_preserved_args:
        raise RuntimeError("r241 preservation receipt changed the r195 loss arguments")
    actual_arg_loss = {
        "value": 1.0,
        "archetype": float(args.archetype_aux_loss_weight),
        "opponent_hand": float(args.opp_hand_loss_weight),
        "opponent_remainder": float(args.opp_remainder_loss_weight),
        "lethal_threat": float(args.lethal_threat_loss_weight),
        "prize_race": float(args.prize_race_loss_weight),
        "setup_board_outcome": float(args.setup_board_outcome_loss_weight),
        "combo_state": float(args.combo_state_loss_weight),
        "guide": float(args.alakazam_guide_loss_weight),
    }
    expected_command_loss = dict(expected_ordinary)
    if (
        args.r274_bootstrap_handoff_receipt is not None
        and args.r284_iteration_boundary_receipt is None
    ):
        # The immutable r195 receipt proves ancestry.  Its combo route was
        # deliberately dormant; the independently receipt-backed r274 handoff
        # activates that resident head for successor learning.
        expected_command_loss["combo_state"] = 0.025
    if (
        actual_arg_loss != expected_command_loss
        or args.expert_rehearsal_guide_loss_weight is None
        or float(args.expert_rehearsal_guide_loss_weight) != 0.0
    ):
        raise RuntimeError(
            "r241 command changed the sealed training loss profile: "
            f"actual={actual_arg_loss} expected={expected_command_loss} "
            f"refresh_guide={args.expert_rehearsal_guide_loss_weight}"
        )
    inherited_expanded = canonical_expanded_loss_weights(
        dict(
            dict(
                (parent_payload.get("extra") or {}).get(
                    "expanded_head_training"
                )
                or {}
            ).get("loss_weights")
            or {}
        )
    )
    if any(float(value) <= 0.0 for value in inherited_expanded.values()):
        raise RuntimeError("r241 expanded-head loss profile changed")

    audit_adapter = dict(checkpoint_audit.get("matchup_adapter") or {})
    checkpoint_dormant = dict(
        audit_adapter.get("checkpoint_dormant_state") or {}
    )
    isolated_optimizer = dict(audit_adapter.get("isolated_optimizer") or {})
    fit = dict(audit_adapter.get("fit") or {})
    expected_isolated_fit = {
        "serialized_runtime_enabled": False,
        "serialized_training_enabled": False,
        "main_optimizer_included": False,
        "base_frozen_during_fit": True,
        "optimizer_scope": "matchup_adapter_bank_only",
        "continuation_state_required": True,
    }
    runtime_receipt_binding = dict(adapter.get("training_activation") or {})
    runtime_receipt_path = Path(str(runtime_receipt_binding.get("path") or ""))
    runtime_receipt_path = runtime_receipt_path.expanduser().resolve()
    public_tree_identity = _r241_file_identity(learner_tree_path)
    runtime_receipt_identity = _r241_file_identity(runtime_receipt_path)
    configured_tree = os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
    requested_activation = Path(
        str(args.dormant_matchup_adapter_activation_receipt or "")
    ).expanduser().resolve()
    if (
        adapter.get("bank_preserved") is not True
        or adapter.get("checkpoint_runtime_enabled") is not False
        or adapter.get("checkpoint_training_enabled") is not False
        or adapter.get("runtime_package_activation_required") is not True
        or adapter.get("training_enabled") is not True
        or adapter.get("runtime_enabled") is not True
        or checkpoint_dormant
        != {
            "runtime_enabled": False,
            "training_enabled": False,
            "ordinary_optimizer_included": False,
        }
        or fit.get("optimizer_scope") != "matchup_adapter_bank_only"
        or int(fit.get("epochs", 0)) <= 0
        or int(fit.get("steps", 0)) <= 0
        or int(fit.get("rows", 0)) <= 0
        or int(isolated_optimizer.get("parameter_count", -1)) != 256
        or int(isolated_optimizer.get("state_count", 0)) <= 0
        or int(adapter.get("epochs_per_rl_update", -1))
        != int(args.dormant_matchup_adapter_epochs)
        or public_tree_identity["digest"]
        != str(learner_tree_binding.get("sha256") or "")
        or int(public_tree_identity["size_bytes"])
        != int(learner_tree_binding.get("size_bytes", -1))
        or runtime_receipt_identity["digest"]
        != str(runtime_receipt_binding.get("sha256") or "")
        or int(runtime_receipt_identity["size_bytes"])
        != int(runtime_receipt_binding.get("size_bytes", -1))
        or not configured_tree
        or (
            args.r274_bootstrap_handoff_receipt is None
            and Path(configured_tree).expanduser().resolve() != learner_tree_path
        )
        or requested_activation != runtime_receipt_path
        or os.environ.get("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "").strip().lower()
        not in {"1", "true", "yes", "on"}
    ):
        raise RuntimeError("r241 Matchup Adapter preservation/runtime proof changed")

    source_snapshot = dict(receipt.get("source_snapshot") or {})
    if source_snapshot.get("owner_contract_sha256") != owner_identity["digest"]:
        raise RuntimeError("r241 source snapshot binds another owner contract")
    baseline_binding = dict(receipt.get("baseline_adapter_roster") or {})
    baseline_path = Path(str(baseline_binding.get("path") or ""))
    baseline_identity = _r241_file_identity(baseline_path.expanduser().resolve())
    if (
        baseline_identity["digest"] != str(baseline_binding.get("sha256") or "")
        or baseline_identity["size_bytes"]
        != int(baseline_binding.get("size_bytes", -1))
    ):
        raise RuntimeError("r241 baseline adapter roster identity changed")

    normalized = {
        "schema": R241_PEAK_R195_TRAINING_CONTRACT_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "owner_decision_revision": 241,
        "owner_clarification_revision": R241_PEAK_R195_CLARIFICATION_REVISION,
        "fixed_cycle_updates": 10,
        "anchor_parent_sha256": parent_identity["digest"],
        "serialized_model_config_sha256": model_config_digest,
        "serialized_state_schema_sha256": state_schema_digest,
        "trainable_parameter_count": trainable_parameter_count,
        "preservation_receipt": receipt_identity,
        "owner_contract": owner_identity,
        "head_role_map": role_identity,
        "source_snapshot": source_snapshot,
        "baseline_adapter_roster": baseline_identity,
        "adapter_slot_migration": migration,
        "checkpoint_audit_fingerprint_sha256": str(
            checkpoint_audit.get("audit_fingerprint_sha256") or ""
        ),
        "physical_source_names": list(R241_PEAK_R195_PHYSICAL_SOURCES),
        "active_non_combo_source_names": list(R241_PEAK_R195_ACTIVE_SOURCES),
        "disabled_source_names": ["combo_state"],
        "combo_state_loss_weight": 0.0,
        "combo_state_fusion_route_enabled": False,
        "ordinary_rl_loss_profile": expected_ordinary,
        "expert_refresh_loss_profile": expected_refresh,
        "expanded_head_loss_weights": inherited_expanded,
        "matchup_adapter": {
            "isolated_fit": expected_isolated_fit,
            "external_runtime_activation": {
                "collection_enabled": True,
                "terminal_package_enabled": True,
                "public_tree_sha256": public_tree_identity["digest"],
                "activation_receipt_sha256": runtime_receipt_identity["digest"],
            },
        },
    }
    return canonical_r241_peak_r195_training_contract(normalized)


def _load_initial_heldout_evidence(
    path: Path,
    *,
    checkpoint: CheckpointIdentity,
    heldout_games: int,
) -> dict[str, Any]:
    """Validate an exact external audit used only as the new-lineage anchor."""
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_ids = set(OFFICIAL_BASELINE_IDS)
    expected_per_opponent = int(heldout_games) // len(expected_ids)
    checkpoint_report = dict(report.get("checkpoint") or {})
    pooled = dict(report.get("pooled_formal") or {})
    deck_gate = dict(report.get("deck_agnostic_gate") or {})
    matchups = list(report.get("matchups") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for raw in matchups:
        row = dict(raw or {})
        opponent_id = str(row.get("opponent_id") or "")
        if opponent_id in by_id:
            raise RuntimeError(
                f"initial heldout evidence repeats opponent {opponent_id!r}"
            )
        by_id[opponent_id] = row

    failures: list[str] = []
    if not bool(report.get("valid")):
        failures.append("report_not_valid")
    if not bool(report.get("trusted_formal")):
        failures.append("report_not_trusted_formal")
    if str(report.get("formal_mode") or "") != "policy":
        failures.append("formal_mode_not_policy")
    if list(report.get("failures") or []):
        failures.append("game_failures_present")
    if str(checkpoint_report.get("digest") or "") != checkpoint.digest:
        failures.append("checkpoint_digest_mismatch")
    if set(report.get("expected_opponents") or []) != expected_ids:
        failures.append("official_opponent_set_mismatch")
    if set(by_id) != expected_ids:
        failures.append("matchup_set_mismatch")
    if int(report.get("scheduled_jobs", -1)) != int(heldout_games):
        failures.append("scheduled_game_count_mismatch")
    if int(report.get("completed_jobs", -1)) != int(heldout_games):
        failures.append("completed_game_count_mismatch")
    if int(pooled.get("games", -1)) != int(heldout_games):
        failures.append("pooled_game_count_mismatch")
    if int(report.get("min_games_per_opponent", -1)) < expected_per_opponent:
        failures.append("minimum_per_opponent_too_small")
    if not bool(deck_gate.get("exact_deck_seat_balance")):
        failures.append("seat_balance_not_exact")

    per_opponent: dict[str, dict[str, float]] = {}
    for opponent_id in OFFICIAL_BASELINE_IDS:
        row = by_id.get(opponent_id, {})
        try:
            games = int(row.get("games", -1))
            wins = float(row.get("wins", -1.0))
            losses = float(row.get("losses", -1.0))
            draws = float(row.get("draws", -1.0))
        except (TypeError, ValueError):
            failures.append(f"invalid_matchup_numbers:{opponent_id}")
            continue
        if games != expected_per_opponent:
            failures.append(f"per_opponent_game_count_mismatch:{opponent_id}")
        if min(wins, losses, draws) < 0.0 or not math.isclose(
            wins + losses + draws, float(games), abs_tol=1e-6
        ):
            failures.append(f"invalid_matchup_outcomes:{opponent_id}")
        per_opponent[opponent_id] = {
            "games": games,
            "wins": wins + 0.5 * draws,
            "losses": losses,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / max(games, 1),
        }

    if failures:
        raise RuntimeError(
            "initial heldout evidence failed closed: " + ",".join(failures)
        )
    return {
        "evidence_schema": 2,
        "iteration": -1,
        "checkpoint_digest": checkpoint.digest,
        "games": int(heldout_games),
        "win_rate": float(pooled["wr"]),
        "confidence_lower": float(pooled["interval_lower"]),
        "confidence_upper": float(pooled["interval_upper"]),
        "per_opponent": per_opponent,
        "audit": {
            "passed": True,
            "source": "trusted_external_new_lineage_anchor",
            "report": _path_content_identity(path),
            "terminal_gate_eligible": False,
        },
    }


def _deck_distribution_contract(
    decks: list[tuple[str, list[int]]],
    *,
    ladder_contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Bind collection to the exact resolved pool and current sampler."""
    n = len(decks)
    ladder_weights = dict((ladder_contract or {}).get("weights") or {})
    entries = []
    for name, cards in decks:
        normalized = [int(card) for card in cards]
        entries.append(
            {
                "name": str(name),
                "cards_sha256": _canonical_digest(normalized),
                "card_count": len(normalized),
                "sampling_weight": (
                    float(ladder_weights[str(name)])
                    if str(name) in ladder_weights
                    else (1.0 / n if n else 0.0)
                ),
            }
        )
    contract = {
        "schema": 1,
        "sampler": (
            "official_ladder_hamilton_sha256_schedule_v1"
            if ladder_contract is not None
            else "deterministic_uniform_round_robin_v1"
        ),
        "entries": entries,
        "digest": _canonical_digest(entries),
    }
    if ladder_contract is not None:
        contract["ladder"] = dict(ladder_contract)
    return contract


def _checkpoint_contract(
    path: Path,
    *,
    smoke: bool,
    allow_legacy_inference_profile: bool = False,
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]] = None,
    r274_successor_training_contract: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fail closed unless a checkpoint is the exact intended trusted profile."""
    from poke_bot import checkpoint, features

    path = Path(path).expanduser().resolve()
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    extra = dict(payload.get("extra") or {})
    if extra.get("pure_rl") is not True:
        raise RuntimeError(f"checkpoint is not explicitly pure_rl: {path}")
    if bool(extra.get("smoke", False)) != bool(smoke):
        raise RuntimeError(
            f"checkpoint smoke/production profile mismatch at {path}: "
            f"checkpoint_smoke={bool(extra.get('smoke', False))} requested={smoke}"
        )
    trusted = checkpoint.assert_trusted_policy_checkpoint(path)
    actual = dict(payload.get("model_config") or {})
    r241_contract = canonical_r241_peak_r195_training_contract(
        r241_peak_r195_training_contract
    )
    if r241_contract:
        expected_true = (
            "expanded_heads_enabled",
            "setup_board_outcome_head_enabled",
            "combo_state_head_enabled",
            "h10_capacity_enabled",
            "decision_fusion_enabled",
            "decision_fusion_runtime_enabled",
            "decision_fusion_dedicated_routes_enabled",
            "decision_fusion_dedicated_routes_runtime_enabled",
            "decision_fusion_typed_output_centered_routes_enabled",
        )
        if (
            _canonical_digest(actual)
            != r241_contract["serialized_model_config_sha256"]
            or _r241_checkpoint_state_schema_digest(payload)
            != r241_contract["serialized_state_schema_sha256"]
            or str(actual.get("decision_context") or "") != "history"
            or any(actual.get(name) is not True for name in expected_true)
            or actual.get("combo_state_route_enabled") is not False
            or actual.get("matchup_adapters_enabled") is not False
            or not math.isclose(
                float(
                    actual.get(
                        "decision_fusion_action_type_reliability_cap", -1.0
                    )
                ),
                0.25,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"r241 checkpoint changed the sealed peak-r195 model profile: {path}"
            )
        state_dict = dict(payload.get("model_state_dict") or {})
        parameter_count = sum(
            int(value.numel())
            for key, value in state_dict.items()
            if hasattr(value, "numel")
            and not str(key).endswith("_feature_schema_version")
            and not str(key).startswith("matchup_adapter_bank.")
        )
        if parameter_count != int(r241_contract["trainable_parameter_count"]):
            raise RuntimeError("r241 checkpoint trainable parameter count changed")
        provenance = dict(payload.get("provenance") or {})
        expanded = dict(provenance.get("expanded_heads") or {})
        fusion = dict(provenance.get("decision_fusion") or {})
        routes = dict(fusion.get("dedicated_routes") or {})
        if (
            tuple(fusion.get("required_heads") or ())
            != R241_PEAK_R195_PHYSICAL_SOURCES
            or tuple(fusion.get("active_required_heads") or ())
            != R241_PEAK_R195_ACTIVE_SOURCES
            or fusion.get("runtime_enabled") is not True
            or tuple(routes.get("route_names") or ())
            != R241_PEAK_R195_PHYSICAL_SOURCES
            or tuple(routes.get("active_route_names") or ())
            != R241_PEAK_R195_ACTIVE_SOURCES
            or tuple(routes.get("disabled_route_names") or ())
            != ("combo_state",)
            or routes.get("runtime_enabled") is not True
            or tuple(expanded.get("runtime_disabled_heads") or ())
            != ("combo_state_head",)
        ):
            raise RuntimeError("r241 checkpoint head/fusion provenance changed")
        validate_r241_isolated_adapter_continuation(
            dict(extra.get("dormant_matchup_adapter_fit") or {}),
            dict(extra.get("dormant_matchup_adapter_optimizer_state") or {}),
        )
        if (
            extra.get("matchup_adapters_runtime_enabled") is not False
            or extra.get("matchup_adapter_training_enabled") is not False
            or extra.get("matchup_adapter_optimizer_included") is not False
        ):
            raise RuntimeError(
                "r241 checkpoint changed serialized dormant-adapter semantics"
            )
        digest = checkpoint.checkpoint_digest(path)
        sidecar = dict(extra.get("peak_r195_live_fusion") or {})
        if digest != r241_contract["anchor_parent_sha256"]:
            validated_sidecar = validate_r241_peak_r195_live_fusion_record(
                sidecar,
                contract=r241_contract,
            )
            sidecar_phase = str(validated_sidecar.get("phase") or "")
            sidecar_boundary = int(
                validated_sidecar.get("boundary_iteration", -1)
            )
            if sidecar_phase == "rl":
                training_provenance = dict(
                    extra.get("training_provenance") or {}
                )
                if sidecar_boundary != int(
                    training_provenance.get("iteration", -1)
                ):
                    raise RuntimeError(
                        "r241 RL sidecar does not match checkpoint iteration"
                    )
            else:
                rehearsal = dict(extra.get("expert_rehearsal") or {})
                if (
                    sidecar_boundary
                    != int(rehearsal.get("before_iteration", -1))
                    or int(rehearsal.get("epochs", -1))
                    != R241_FIXED_CYCLE_REHEARSAL_EPOCHS
                    or dict(rehearsal.get("peak_r195_live_fusion") or {})
                    != validated_sidecar
                ):
                    raise RuntimeError(
                        "r241 expert-refresh sidecar does not match its checkpoint"
                    )
        elif sidecar:
            raise RuntimeError(
                "sealed r195 anchor unexpectedly carries an r241 successor sidecar"
            )
        from poke_bot import features
        feature_schema = provenance.get("feature_schema")
        if feature_schema != features.FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint feature schema mismatch at {path}: "
                f"checkpoint={feature_schema!r} "
                f"runtime={features.FEATURE_SCHEMA_VERSION!r}"
            )
        from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint

        validate_zero_dormant_checkpoint(path, allow_trained=True)
        return {
            "path": str(path),
            "digest": digest,
            "pure_rl": True,
            "smoke": bool(smoke),
            "trusted_policy": trusted,
            "model_profile": actual,
            "decision_context": "history",
            "max_context": int(actual.get("max_context", 0)),
            "feature_schema": feature_schema,
            "trainable_parameters": parameter_count,
            "rl_iteration": int(payload.get("rl_iteration", 0)),
            "legacy_inference_profile": False,
            "r241_peak_r195_preserved": True,
        }
    r274_contract = dict(r274_successor_training_contract or {})
    # The derivative borrows only the established aligned host stream. Its
    # checkpoint remains governed by the derivative envelope rather than the
    # historical r274 route-gate profile.
    if r274_contract.get("derivative_streaming_only") is True:
        r274_contract = {}
    if r274_contract:
        tactical_free = os.environ.get(
            "POKEBOT_R274_DISABLE_RL_TACTICAL_COTRAIN", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        expected_true = (
            "expanded_heads_enabled",
            "setup_board_outcome_head_enabled",
            "combo_state_head_enabled",
            "combo_state_route_enabled",
            "decision_fusion_enabled",
            "decision_fusion_runtime_enabled",
            "decision_fusion_dedicated_routes_enabled",
            "decision_fusion_dedicated_routes_runtime_enabled",
            "decision_fusion_typed_output_centered_routes_enabled",
            "h10_capacity_enabled",
            "own_deck_ledger_enabled",
            "own_deck_ledger_runtime_enabled",
            "visible_tutor_completion_head_enabled",
            "visible_tutor_completion_route_enabled",
            "visible_tutor_completion_route_runtime_enabled",
            "terminal_conversion_head_enabled",
            "terminal_conversion_route_enabled",
            "terminal_conversion_route_runtime_enabled",
        )
        tactical_fields = (
            "tactical_sequence_outcome_head_enabled",
            "tactical_sequence_outcome_route_enabled",
            "tactical_sequence_outcome_route_present",
            "tactical_sequence_outcome_route_runtime_enabled",
        )
        actual_is_tactical_free = all(
            actual.get(name) is False for name in tactical_fields
        )
        expected_config_digests = {
            str(r274_contract.get("successor_model_config_sha256") or "")
        }
        expected_schema_digests = {
            str(r274_contract.get("successor_state_schema_sha256") or "")
        }
        if tactical_free:
            expected_config_digests.add(
                str(r274_contract.get("tactical_free_model_config_sha256") or "")
            )
            expected_schema_digests.add(
                str(r274_contract.get("tactical_free_state_schema_sha256") or "")
            )
        if (
            _canonical_digest(actual) not in expected_config_digests
            or _r241_checkpoint_state_schema_digest(payload)
            not in expected_schema_digests
            or str(actual.get("decision_context") or "") != "history"
            or any(actual.get(name) is not True for name in expected_true)
            or (
                not actual_is_tactical_free
                and any(actual.get(name) is not True for name in tactical_fields)
            )
            or (actual_is_tactical_free and not tactical_free)
            or str(actual.get("matchup_adapter_format") or "")
            != "poke-bot-matchup-adapter-bank-v6"
        ):
            raise RuntimeError(
                f"r274 successor checkpoint architecture/schema changed: {path}"
            )
        provenance = dict(payload.get("provenance") or {})
        feature_schema = provenance.get("feature_schema")
        if feature_schema != features.FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"r274 successor feature schema mismatch at {path}: "
                f"checkpoint={feature_schema!r} runtime={features.FEATURE_SCHEMA_VERSION!r}"
            )
        state_dict = dict(payload.get("model_state_dict") or {})
        parameter_count = sum(
            int(value.numel())
            for key, value in state_dict.items()
            if hasattr(value, "numel")
            and not str(key).endswith("_feature_schema_version")
        )
        return {
            "path": str(path),
            "digest": checkpoint.checkpoint_digest(path),
            "pure_rl": True,
            "smoke": bool(smoke),
            "trusted_policy": trusted,
            "model_profile": actual,
            "decision_context": "history",
            "max_context": int(actual.get("max_context", 0)),
            "feature_schema": feature_schema,
            "trainable_parameters": parameter_count,
            "rl_iteration": int(payload.get("rl_iteration", 0)),
            "legacy_inference_profile": False,
            "r274_successor_receipt_bound": True,
        }
    # Legacy checkpoints predate the dormant-bank flag. Missing and explicit
    # false are the same inactive contract; true remains a real profile change.
    actual.setdefault("matchup_adapters_enabled", False)
    # Decision fusion is an additive, zero-safe architecture introduced after
    # this lineage began.  Legacy inference-role checkpoints omit these fields;
    # omission is exactly the disabled/default contract, just as it is for the
    # dormant matchup bank above.  A true value remains a real profile change.
    actual.setdefault("decision_fusion_enabled", False)
    actual.setdefault("decision_fusion_runtime_enabled", False)
    actual.setdefault("decision_fusion_width", 16)
    # This is a non-architectural, process-resolved route gate.  A pinned H10
    # checkpoint can therefore run with combo's tensors resident but its route
    # disabled; do not reject that explicit operator control as a profile
    # migration.  In the absence of an override, missing historical config
    # preserves the route's original enabled behavior.
    actual.setdefault("combo_state_route_enabled", True)
    expected_cfg = pure_rl_model_config(**({"dropout": 0.0} if smoke else {}))
    expected = model_config_dict(expected_cfg)
    expected.setdefault("matchup_adapters_enabled", False)
    combo_route_scope = os.environ.get(
        "POKEBOT_COMBO_STATE_ROUTE_SPECIALIST", ""
    ).strip().casefold()
    checkpoint_specialist = str(
        payload.get("archetype_id") or ""
    ).strip().casefold()
    route_checkpoint_digest = os.environ.get(
        "POKEBOT_COMBO_STATE_ROUTE_CHECKPOINT_DIGEST", ""
    ).strip().casefold()
    route_digest_matches = bool(
        route_checkpoint_digest
        and checkpoint.checkpoint_digest(path).casefold()
        == route_checkpoint_digest
    )
    combo_route_override_applies = (
        "POKEBOT_COMBO_STATE_ROUTE_ENABLED" in os.environ
        and (
            not combo_route_scope
            or checkpoint_specialist == combo_route_scope
            or (not checkpoint_specialist and route_digest_matches)
        )
    )
    if combo_route_override_applies:
        actual["combo_state_route_enabled"] = expected[
            "combo_state_route_enabled"
        ]
    changed = sorted(
        key
        for key in set(actual) | set(expected)
        if actual.get(key) != expected.get(key)
    )
    legacy_fusion_differences = {
        "decision_fusion_enabled",
        *(
            {"decision_fusion_runtime_enabled"}
            if expected.get("decision_fusion_runtime_enabled") is True
            else set()
        ),
    }
    state_dict = payload.get("model_state_dict") or {}
    legacy_inference_profile = bool(
        allow_legacy_inference_profile
        and expected.get("decision_fusion_enabled") is True
        and actual.get("decision_fusion_enabled") is False
        and actual.get("decision_fusion_runtime_enabled") is False
        and changed == sorted(legacy_fusion_differences)
        and not any(
            str(key).startswith("decision_fusion.") for key in state_dict
        )
    )
    v2_to_v3_inference_profile = bool(
        allow_legacy_inference_profile
        and actual.get("decision_fusion_enabled") is True
        and actual.get("decision_fusion_runtime_enabled") is True
        and expected.get("decision_fusion_typed_output_centered_routes_enabled")
        is True
        and expected.get("decision_fusion_action_type_reliability_cap") == 0.25
        and actual.get("decision_fusion_typed_output_centered_routes_enabled")
        in {None, False}
        and actual.get("decision_fusion_action_type_reliability_cap") in {None, 1.0}
        and changed
        == [
            "decision_fusion_action_type_reliability_cap",
            "decision_fusion_typed_output_centered_routes_enabled",
        ]
    )
    if actual != expected:
        if not (legacy_inference_profile or v2_to_v3_inference_profile):
            raise RuntimeError(
                f"checkpoint does not match exact intended pure-RL model profile at "
                f"{path}; changed_fields={changed}"
            )
    provenance = dict(payload.get("provenance") or {})
    feature_schema = provenance.get("feature_schema")
    if feature_schema != features.FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint feature schema mismatch at {path}: "
            f"checkpoint={feature_schema!r} runtime={features.FEATURE_SCHEMA_VERSION!r}"
        )
    if any(str(key).startswith("matchup_adapter_bank.") for key in state_dict):
        from poke_bot.dormant_adapter_compat import (
            validate_zero_dormant_checkpoint,
        )

        validate_zero_dormant_checkpoint(path, allow_trained=True)
    # The state dict is the architecture authority. ``extra.param_count`` is
    # training-time telemetry and can be smaller when heads were temporarily
    # frozen while saving a checkpoint. Trusting it makes a valid checkpoint's
    # immutable model contract change across a restart.
    #
    # State dicts include the feature-schema scalar buffer; it is not a
    # trainable parameter. The production profile has no other persistent
    # non-parameter tensors, and every remaining value is shape-countable.
    parameter_count = sum(
        int(value.numel())
        for key, value in state_dict.items()
        if hasattr(value, "numel")
        and not str(key).endswith("_feature_schema_version")
        # Frozen dormant modules are not part of the ordinary learner
        # optimizer/design size. Their separate pinned config is validated by
        # checkpoint loading and the boundary migration receipt.
        and not str(key).startswith("matchup_adapter_bank.")
    )
    return {
        "path": str(path),
        "digest": checkpoint.checkpoint_digest(path),
        "pure_rl": True,
        "smoke": bool(smoke),
        "trusted_policy": trusted,
        # A pre-fusion champion may remain the immutable promotion incumbent
        # while the active learner is migrated.  In that one explicitly
        # declared inference role, report the process's intended learner
        # profile to the design contract while preserving the old checkpoint
        # bytes and loading it with its own serialized flat-policy config.
        "model_profile": expected if legacy_inference_profile else actual,
        "decision_context": str(
            (expected if legacy_inference_profile else actual).get(
                "decision_context"
            )
        ),
        "max_context": int(
            (expected if legacy_inference_profile else actual).get(
                "max_context", 0
            )
        ),
        "feature_schema": feature_schema,
        "trainable_parameters": int(parameter_count),
        "rl_iteration": int(payload.get("rl_iteration", 0)),
        "legacy_inference_profile": legacy_inference_profile,
    }


def _opponent_specs_contract(specs: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "id": str(spec.id),
                "source": str(spec.source),
                "content": _path_content_identity(Path(spec.path)),
            }
        )
    return rows


def _population_collect_specs(
    *,
    enabled: bool,
    frozen_specialist_ids: tuple[str, ...],
    by_id: dict[str, Any],
    opponent_registry: dict[str, Any] | None = None,
) -> list[Any] | None:
    """Resolve the exact own-model-only population training field."""

    if not enabled:
        return None
    from poke_bot.baselines_runtime import baseline_content_digest

    if opponent_registry is not None:
        rows = [
            dict(row)
            for row in (opponent_registry.get("opponents") or [])
            if isinstance(row, dict)
        ]
        specialist_ids = {
            str(value)
            for value in (opponent_registry.get("specialist_ids") or ())
        }
        opponent_ids = [
            str(row.get("opponent_id") or "") for row in rows
        ]
        covered = {str(row.get("specialist_id") or "") for row in rows}
        if (
            opponent_registry.get("schema")
            != "poke_bot.population_opponent_registry/v1"
            or opponent_registry.get("external_agents_training_eligible")
            is not False
            or int(opponent_registry.get("member_count") or 0) < 1
            or len(specialist_ids)
            != int(opponent_registry.get("member_count") or 0)
            or covered != specialist_ids
            or not rows
            or "" in opponent_ids
            or len(opponent_ids) != len(set(opponent_ids))
            or any(row.get("external_agent") is not False for row in rows)
        ):
            raise RuntimeError(
                "population opponent registry is not exact own-model history"
            )
        missing = [
            opponent_id
            for opponent_id in opponent_ids
            if opponent_id not in by_id
        ]
        if missing:
            raise RuntimeError(
                f"population own-model packages are unavailable: {missing}"
            )
        for row in rows:
            spec = by_id[str(row["opponent_id"])]
            if (
                baseline_content_digest(spec.path)
                != str(row.get("content_digest") or "")
            ):
                raise RuntimeError(
                    "population opponent content changed: "
                    f"{row['opponent_id']}"
                )
        return [by_id[opponent_id] for opponent_id in opponent_ids]
    if (
        not frozen_specialist_ids
        or len(set(frozen_specialist_ids)) != len(frozen_specialist_ids)
    ):
        raise RuntimeError(
            "population collection requires a nonempty unique frozen roster"
        )
    missing = [
        specialist_id
        for specialist_id in frozen_specialist_ids
        if specialist_id not in by_id
    ]
    if missing:
        raise RuntimeError(
            f"population own-model packages are unavailable: {missing}"
        )
    return [by_id[specialist_id] for specialist_id in frozen_specialist_ids]


POPULATION_RL_EPOCHS_PER_CYCLE = 5
POPULATION_REHEARSAL_EPOCHS_PER_CYCLE = 5


def population_cycle_rehearsal_due(
    *,
    population_enabled: bool,
    next_iteration: int,
    configured_rehearsal_every: int,
    configured_rehearsal_epochs: int,
) -> bool:
    """Return whether an exact population RL block needs its closing rehearsal.

    Population members run in distinct five-RL-iteration lineages.  The normal
    trainer cadence rehearses *before* iteration five, which would require a
    sixth RL iteration merely to trigger the rehearsal.  The population
    controller instead closes each five-iteration lineage with this explicit
    boundary and starts the member's next lineage from its rehearsed checkpoint.
    """

    if not population_enabled:
        return False
    if (
        int(configured_rehearsal_every) != POPULATION_RL_EPOCHS_PER_CYCLE
        or int(configured_rehearsal_epochs)
        != POPULATION_REHEARSAL_EPOCHS_PER_CYCLE
    ):
        raise RuntimeError(
            "population phase requires the exact 5-RL/5-rehearsal schedule"
        )
    return (
        int(next_iteration) > 0
        and int(next_iteration) % POPULATION_RL_EPOCHS_PER_CYCLE == 0
    )


def _design_contract(
    *,
    args: argparse.Namespace,
    checkpoint_profile: dict[str, Any],
    lineage_base: dict[str, str],
    initial_learner: dict[str, str],
    initial_replay_shards: list[dict[str, Any]],
    source: dict[str, Any],
    decks: list[tuple[str, list[int]]],
    measurement_decks: list[tuple[str, list[int]]],
    ladder_contract: Optional[dict[str, Any]],
    family_training_contract: Optional[dict[str, Any]],
    endpoints: list[str],
    collect_specs: list[Any],
    research_control_specs: list[Any],
    practice_specs: list[Any],
    practice_minimum_games: Optional[dict[str, int]],
    heldout_specs: list[Any],
    research_control_registry: dict[str, Any],
    frozen_specialist_registry: dict[str, Any],
    active_gate: Optional[dict[str, Any]],
    seed_namespace_contract: dict[str, Any],
    multi_env_per_worker: int,
    leaf_coalesce_ms: float,
    initial_heldout_evidence: Optional[dict[str, Any]] = None,
    r241_peak_r195_training_contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Immutable inputs whose drift would fork the experiment lineage."""
    prize_plan_h3_actor_provider = _configured_prize_plan_h3_actor_provider(args)
    hidden_engine_raw = os.environ.get("POKEBOT_LIBCG_PATH", "").strip()
    hidden_engine = (
        _path_content_identity(Path(hidden_engine_raw).expanduser().resolve())
        if hidden_engine_raw
        else None
    )
    cg_runtime_raw = os.environ.get("CG_LIB_PATH", "").strip()
    official_engine = None
    if cg_runtime_raw:
        runtime = Path(cg_runtime_raw).expanduser().resolve()
        if not (runtime / "cg").is_dir():
            raise RuntimeError(
                "CG_LIB_PATH must name the runtime parent containing cg/"
            )
        member = runtime / "cg" / (
            "libcg.dylib" if sys.platform == "darwin" else "libcg.so"
        )
        official_engine = {
            "runtime_root": str(runtime),
            "member": _path_content_identity(member),
        }
        expected = os.environ.get(
            "POKEBOT_EXPECTED_LIBCG_SHA256", ""
        ).strip()
        if expected and official_engine["member"]["digest"] != (
            "sha256:" + expected.removeprefix("sha256:")
        ):
            raise RuntimeError("CG_LIB_PATH simulator digest mismatch")
    specialist_deck_raw = os.environ.get(
        "POKEBOT_SPECIALIST_DECK_PATH", ""
    ).strip()
    specialist_deck_override = (
        _path_content_identity(Path(specialist_deck_raw).expanduser().resolve())
        if specialist_deck_raw
        else None
    )
    return {
        "schema": 1,
        "r241_peak_r195_training_contract": dict(
            r241_peak_r195_training_contract or {}
        ),
        "run": {
            "mode": str(args.mode),
            "iterations": int(args.iterations),
            "fixed_cycle_updates": _fixed_cycle_updates(args),
        },
        "games": {
            "per_iteration": int(args.games_per_iter),
            "heldout": int(args.heldout_games),
            "heldout_additive_remotes": bool(args.heldout_remotes),
            "promotion": int(args.promotion_games),
            "minimum_usable_fraction": float(args.min_usable_game_frac),
        },
        "learner": {
            "epochs": int(args.train_epochs),
            "games_per_batch": int(args.train_games_per_batch),
            "max_decisions_per_batch": int(
                args.train_max_decisions_per_batch
            ),
            "warmup_max_decisions_per_batch": int(
                args.train_warmup_max_decisions_per_batch
            ),
            "warmup_iterations": int(args.train_warmup_iterations),
            "device_resident": bool(args.train_device_resident),
            "device_resident_min_free_gib": float(
                args.train_device_resident_min_free_gib
            ),
            "seed": int(args.seed),
            "continuous": True,
            **(
                {"prize_plan_h3_actor_provider": prize_plan_h3_actor_provider}
                if prize_plan_h3_actor_provider is not None
                else {}
            ),
            "carry_min_head_to_head_wr": float(args.continuous_learner_min_wr),
            "exact_gate_regression_margin": float(
                args.continuous_learner_exact_regression_margin
            ),
            "exact_gate_regression_patience": int(
                args.continuous_learner_exact_regression_patience
            ),
            "archetype_aux_loss_weight": float(
                args.archetype_aux_loss_weight
            ),
            "opp_hand_loss_weight": float(args.opp_hand_loss_weight),
            "opp_remainder_loss_weight": float(
                args.opp_remainder_loss_weight
            ),
            "lethal_threat_loss_weight": float(
                args.lethal_threat_loss_weight
            ),
            "prize_race_loss_weight": float(args.prize_race_loss_weight),
            "alakazam_guide_loss_weight": float(
                args.alakazam_guide_loss_weight
            ),
            "current_deck_guide_loss_weight": float(
                args.alakazam_guide_loss_weight
            ),
            **(
                {
                    "current_deck_guide_training_mode": str(
                        args.current_deck_guide_training_mode
                    ),
                    "setup_board_outcome_loss_weight": float(
                        args.setup_board_outcome_loss_weight
                    ),
                    "current_deck_guide_strategic_curriculum": {
                        "curriculum_spec": _path_content_identity(
                            Path(
                                args.current_deck_guide_curriculum_spec
                            ).expanduser().resolve()
                        ),
                        "head_role_map": _path_content_identity(
                            Path(
                                args.current_deck_guide_head_role_map
                            ).expanduser().resolve()
                        ),
                        "validation_receipt": _path_content_identity(
                            Path(
                                args.current_deck_guide_curriculum_validation_receipt
                            ).expanduser().resolve()
                        ),
                    },
                }
                if args.current_deck_guide_training_mode
                in GUIDE_STRATEGIC_TRAINING_MODES
                else {}
            ),
            "expanded_head_loss_weight_overrides": (
                {
                    "tactical_outcome": float(
                        args.tactical_outcome_loss_weight_override
                    )
                }
                if args.tactical_outcome_loss_weight_override is not None
                else {}
            ),
            "current_deck_guide_archetype": deck_guides.selected_id(),
            "dormant_matchup_adapter": {
                "epochs": int(args.dormant_matchup_adapter_epochs),
                "learning_rate": float(args.dormant_matchup_adapter_lr),
                "max_decisions_per_batch": int(
                    args.dormant_matchup_adapter_max_decisions_per_batch
                ),
                "activation_receipt": (
                    _path_content_identity(
                        Path(
                            args.dormant_matchup_adapter_activation_receipt
                        ).expanduser().resolve()
                    )
                    if args.dormant_matchup_adapter_activation_receipt
                    is not None
                    else None
                ),
                "runtime_enabled": False,
                "ticket_sources": [
                    "exact_alakazam_mirror",
                    "active_gate_strong_public_practice",
                ],
            },
            "alakazam_guide_targets_enabled": bool(alakazam_heuristics.enabled()),
            "current_deck_guide_targets_enabled": bool(deck_guides.enabled()),
            "current_deck_guide_version": deck_guides.guide_version(),
            "profile": dict(checkpoint_profile["model_profile"]),
            "trainable_parameters": int(
                checkpoint_profile["trainable_parameters"]
            ),
            "decision_context": str(checkpoint_profile["decision_context"]),
            "max_context": int(checkpoint_profile["max_context"]),
            "feature_schema": checkpoint_profile["feature_schema"],
            "initial_checkpoint": dict(initial_learner),
            "initial_heldout_evidence": (
                dict(initial_heldout_evidence)
                if initial_heldout_evidence is not None
                else None
            ),
            "initial_replay_shards": list(initial_replay_shards),
        },
        "expert_rehearsal": {
            "every_iterations": int(args.expert_rehearsal_every),
            "before_first_iteration": bool(args.expert_rehearsal_before_first),
            "epochs": int(args.expert_rehearsal_epochs),
            "terminal_after_final_iteration": bool(
                args.terminal_expert_rehearsal
            ),
            "one_time_override": {
                "before_iteration": int(args.expert_rehearsal_one_time_before),
                "epochs": int(args.expert_rehearsal_one_time_epochs),
            },
            "learning_rate": float(args.expert_rehearsal_lr),
            "requested_batch_size": int(args.expert_rehearsal_batch_size),
            "minimum_decisions": int(args.expert_min_decisions),
            "required_target_coverage": list(
                args.expert_required_target or EXPERT_REHEARSAL_TARGETS
            ),
            "loss_weights": {
                "archetype": float(args.archetype_aux_loss_weight),
                "opponent_hand": float(args.opp_hand_loss_weight),
                "opponent_hidden_remainder": float(
                    args.opp_remainder_loss_weight
                ),
                "lethal_threat": float(args.lethal_threat_loss_weight),
                "prize_race": float(args.prize_race_loss_weight),
                "alakazam_guide": float(
                    args.alakazam_guide_loss_weight
                    if args.expert_rehearsal_guide_loss_weight is None
                    else args.expert_rehearsal_guide_loss_weight
                ),
            },
            "training_seat_split": {
                "required": bool(args.require_exact_training_seat_split),
                "policy": (
                    "exact_50_50_per_train_and_validation_partition"
                    if bool(args.require_exact_training_seat_split)
                    else "source_manifest_distribution"
                ),
                "stages": ["assigned", "actual", "consumed"],
                "second_seat_priority": False,
            },
            # The pointer is mutable by design; every actual manifest digest is
            # frozen in a per-rehearsal receipt instead of this lineage contract.
            "rolling_manifest_pointer": (
                str(Path(args.expert_manifest).expanduser().resolve())
                if args.expert_manifest is not None
                else None
            ),
            "matchup_adapters": {
                "enabled": args.expert_matchup_adapter_manifest is not None,
                "staged_manifest": (
                    str(
                        Path(args.expert_matchup_adapter_manifest)
                        .expanduser()
                        .resolve()
                    )
                    if args.expert_matchup_adapter_manifest is not None
                    else None
                ),
                "epochs": int(args.expert_matchup_adapter_epochs),
                "learning_rate": float(args.expert_matchup_adapter_lr),
                "games_per_batch": int(
                    args.expert_matchup_adapter_games_per_batch
                ),
                "max_decisions_per_batch": int(
                    args.expert_matchup_adapter_max_decisions_per_batch
                ),
                "optimizer_scope": "matchup_adapter_bank_only",
                "restore_parent_optimizer_state": True,
                "runtime_enabled_during_fit": False,
            },
        },
        "artifact_retention": {
            "history_iterations": int(args.artifact_history_iterations),
            "minimum_free_disk_gb": float(args.min_free_disk_gb),
            "protected_identities": (
                "champion,heldout_champion,learner,lineage_base,opponent_pool"
            ),
        },
        "gates": {
            "heldout_wr": float(args.gate_wr),
            "heldout_per_opponent_floor": float(args.heldout_per_opponent_floor),
            "active_contract": (
                _path_content_identity(
                    Path(
                        args.formal_holdout_contract
                        or args.active_gate_contract
                    ).expanduser().resolve()
                )
                if (
                    args.formal_holdout_contract is not None
                    or args.active_gate_contract is not None
                )
                else None
            ),
            "frozen_specialist_registry": dict(frozen_specialist_registry),
            "promotion_threshold": float(args.promotion_threshold),
            "promotion_confidence": float(args.promotion_confidence),
            "promotion_bootstrap_resamples": int(args.promotion_bootstrap_resamples),
        },
        "collection": {
            "training_seat_split": {
                "required": bool(args.require_exact_training_seat_split),
                "policy": (
                    "exact_50_50_first_second_training"
                    if bool(args.require_exact_training_seat_split)
                    else "ordinary_balanced_scheduler"
                ),
                "second_seat_priority": False,
            },
            "training_opponent_scope": (
                "own_frozen_population_only"
                if bool(args.population_own_models_only)
                else "baseline_public_mix_plus_active_gate_practice"
            ),
            "external_agents_training_eligible": not bool(
                args.population_own_models_only
            ),
            "behavior_policy": (
                "gate_aligned_continuous_learner_with_exact_regression_rollback_v3"
            ),
            "temperature": float(args.collect_temperature),
            "game_timeout_s": int(args.game_timeout_s),
            "self_play_fraction": float(config.PURE_RL.self_play_frac),
            "group_games_per_iteration": _planned_collection_group_counts(
                games_per_iteration=int(args.games_per_iter),
                self_play_fraction=float(config.PURE_RL.self_play_frac),
                strong_public_fraction_of_public=float(
                    args.official_collect_frac
                ),
                research_control_games=int(
                    args.research_control_games_per_iter
                ),
            ),
            "official_target_fraction_of_public": float(
                args.official_collect_frac
            ),
            "official_targeting": {
                "strategy": (
                    "latest_exact_active_gate_gap_tier_weighted_v1"
                    if bool(args.official_adaptive_targeting)
                    else "uniform_v1"
                ),
                "minimum_share_per_opponent": float(
                    args.official_adaptive_min_share
                ),
                "gap_power": float(args.official_adaptive_gap_power),
                "target_win_rate": float(
                    args.strong_public_practice_target_wr
                    if active_gate is not None
                    else args.heldout_per_opponent_floor
                ),
                "tier_weights": (
                    {
                        str(row["opponent_id"]): float(row["weight"])
                        for row in active_gate["roster"]
                    }
                    if active_gate is not None
                    else None
                ),
                "formal_eval_disjoint": (
                    "same_policies_disjoint_seeds_jobs_and_replay_v1"
                ),
            },
            "strong_public_practice": {
                "enabled": bool(
                    active_gate is not None
                    and float(args.official_collect_frac) > 0.0
                ),
                "active_gate_id": (
                    str(active_gate["id"]) if active_gate is not None else None
                ),
                "configured_fraction_of_public": float(args.official_collect_frac),
                "effective_fraction_of_public": (
                    _planned_collection_group_counts(
                        games_per_iteration=int(args.games_per_iter),
                        self_play_fraction=float(config.PURE_RL.self_play_frac),
                        strong_public_fraction_of_public=float(
                            args.official_collect_frac
                        ),
                        research_control_games=int(
                            args.research_control_games_per_iter
                        ),
                    )[STRONG_PUBLIC_PRACTICE_GROUP]
                    / max(
                        1,
                        int(args.games_per_iter)
                        - _planned_collection_group_counts(
                            games_per_iteration=int(args.games_per_iter),
                            self_play_fraction=float(config.PURE_RL.self_play_frac),
                            strong_public_fraction_of_public=float(
                                args.official_collect_frac
                            ),
                            research_control_games=int(
                                args.research_control_games_per_iter
                            ),
                        )["self_play"],
                    )
                ),
                "reclaimed_control_slots": int(
                    args.research_control_games_per_iter
                ),
                "temperature": float(args.strong_public_practice_temperature),
                "sample_actions": True,
                "training_eligible": True,
                "formal_eval": False,
                "roster": _opponent_specs_contract(practice_specs),
                "minimum_games_by_opponent": (
                    dict(practice_minimum_games)
                    if practice_minimum_games is not None
                    else None
                ),
                "seed_contract": dict(seed_namespace_contract),
            },
            "research_control_phase": {
                "enabled": bool(
                    int(args.research_control_games_per_iter) > 0
                    and not bool(args.disable_research_control_measurement)
                ),
                "stage": "measure:research_controls",
                "source": "research_control_registry",
                "games_per_iteration": (
                    0
                    if bool(args.disable_research_control_measurement)
                    else int(args.research_control_games_per_iter)
                ),
                "legacy_reclaimed_training_slots": int(
                    args.research_control_games_per_iter
                ),
                "disabled_by_revision_25": bool(
                    args.disable_research_control_measurement
                ),
                "revision_25_activation_receipt": (
                    _path_content_identity(
                        Path(args.r327_evaluation_boundary_receipt)
                        .expanduser()
                        .resolve()
                    )
                    if args.r327_evaluation_boundary_receipt is not None
                    else None
                ),
                "games_per_control": 250,
                "seat0_games_per_control": 125,
                "seat1_games_per_control": 125,
                "action_selection": "greedy",
                "sampled_behavior_policy": False,
                "training_eligible": False,
                "replay_eligible": False,
                "diagnostic_only": True,
                "additive_to_training_budget": True,
                "formal_eval": False,
                "included_in_gate_pass": False,
                "gate_weight": 0.0,
                "seed_namespace": "eval/research-controls-fixed-manifest-v1",
                "separate_result_artifact": True,
                "roster": _opponent_specs_contract(research_control_specs),
                "registry": dict(research_control_registry),
            },
            "r274_submission_55378392_training_cell": (
                {
                    "enabled": True,
                    "submission_id": 55378392,
                    "opponent_id": R274_R195_RESEARCH_OPPONENT_ID,
                    "bundle_sha256": R274_R195_RESEARCH_BUNDLE_SHA256,
                    "model_sha256": R274_R195_RESEARCH_MODEL_SHA256,
                    "minimum_games_each_update": R274_R195_RESEARCH_GAMES_MINIMUM,
                    "minimum_games_each_seat": R274_R195_RESEARCH_EACH_SEAT_MINIMUM,
                    "inside_public_mix": True,
                    "transport": "singleton_play",
                    "receipt": _path_content_identity(
                        Path(args.r274_r195_research_baseline_receipt)
                        .expanduser()
                        .resolve()
                    ),
                }
                if _fixed_cycle_updates(args) == R274_FIXED_CYCLE_UPDATES
                else None
            ),
            "r280_gpu_resident_expert_refresh": (
                {
                    "enabled": True,
                    "owner_revision": 280,
                    "pack": _path_content_identity(
                        Path(args.r280_contiguous_expert_pack)
                        .expanduser()
                        .resolve()
                    ),
                    "pack_receipt": _path_content_identity(
                        Path(args.r280_contiguous_expert_pack_receipt)
                        .expanduser()
                        .resolve()
                    ),
                    "device": str(args.r280_refresh_device),
                    "epochs_each_boundary": 5,
                    "boundaries": [5, 10, 15, 20],
                    "full_numeric_pack_resident": True,
                    "device_side_batch_gather": True,
                    "host_batch_streaming_used": False,
                    "resident_python_objects_used": False,
                }
                if _fixed_cycle_updates(args) == R274_FIXED_CYCLE_UPDATES
                else None
            ),
            "official_exploit": {
                "opponents": list(args.official_exploit_opponents),
                "fraction_per_selected_opponent": float(
                    args.official_exploit_frac
                ),
                "temperature": float(args.official_exploit_temperature),
                "pairing": "consecutive_occurrence_seat_pairs_v1",
            },
            "replay_window_shards": int(config.PURE_RL.replay_window_shards),
            "leaf_eval": str(args.leaf_eval),
            "multi_env_per_worker": int(multi_env_per_worker),
            "leaf_coalesce_ms": float(leaf_coalesce_ms),
            "auxiliary_targets": {
                "archetype": "sequence_opponent_archetype_when_registered",
                "belief": "private_training_engine_exact_same_state_only",
                "lethal": "posthoc_played_line_prize_take_horizon",
                "prize_race": "public_observation_prize_counts",
                "missing_targets_masked": True,
                "hidden_engine": hidden_engine,
            },
            "simulator_engine": official_engine,
        },
        "remotes": {
            "disabled": bool(args.no_remote_workers),
            "endpoints": list(endpoints),
        },
        "deck_distribution": _deck_distribution_contract(
            decks, ladder_contract=ladder_contract
        ),
        "specialist_deck_override": specialist_deck_override,
        "measurement_deck_distribution": _deck_distribution_contract(
            measurement_decks, ladder_contract=None
        ),
        "archetype_family": (
            {
                "schema": "poke_bot.activated_archetype_family/v1",
                "family_id": str(
                    family_training_contract["manifest"]["family_id"]
                ),
                "manifest": {
                    "path": str(family_training_contract["manifest_path"]),
                    "sha256": str(
                        family_training_contract["manifest_sha256"]
                    ),
                },
                "loss_contract": {
                    "path": str(family_training_contract["loss_contract_path"]),
                    "sha256": str(
                        family_training_contract["loss_contract_sha256"]
                    ),
                },
                "loss_vector": {
                    "path": str(family_training_contract["loss_vector_path"]),
                    "sha256": str(
                        family_training_contract["loss_vector_sha256"]
                    ),
                },
                "residual_objectives": dict(
                    family_training_contract["residual_objectives"]
                ),
                "sampler": "deterministic_hamilton_family_macro_v1",
                "package_probability": 0.20,
                "measurement_deck_remains_singular": True,
            }
            if family_training_contract is not None
            else None
        ),
        "opponents": {
            "collect": _opponent_specs_contract(collect_specs),
            "research_controls": _opponent_specs_contract(
                research_control_specs
            ),
            "heldout": _opponent_specs_contract(heldout_specs),
            "official_target_training": (
                _opponent_specs_contract(practice_specs)
                if float(args.official_collect_frac) > 0.0
                else []
            ),
        },
        "lineage_base": dict(lineage_base),
        "source": {
            "git_head": source.get("git_head"),
            "source_tree_sha256": source.get("source_tree_sha256"),
        },
    }


def _design_fingerprint(contract: dict[str, Any]) -> str:
    return _canonical_digest(contract)


def _validate_design_fingerprint(
    *, state: dict[str, Any], manifest: dict[str, Any], current: dict[str, Any]
) -> str:
    current_digest = _design_fingerprint(current)
    state_digest = str(state.get("design_fingerprint") or "")
    manifest_digest = str(manifest.get("design_fingerprint") or "")
    stored_contract = manifest.get("design_contract")
    if not state_digest or not manifest_digest or not isinstance(stored_contract, dict):
        raise RuntimeError("resume lineage is missing its immutable design fingerprint")
    if state_digest != manifest_digest:
        raise RuntimeError(
            "resume design fingerprint disagrees between ledger and manifest: "
            f"ledger={state_digest} manifest={manifest_digest}"
        )
    if _design_fingerprint(stored_contract) != manifest_digest:
        raise RuntimeError("immutable manifest design contract digest is corrupt")
    if current_digest != manifest_digest:
        changed = sorted(
            key
            for key in set(current) | set(stored_contract)
            if current.get(key) != stored_contract.get(key)
        )
        raise RuntimeError(
            "resume design drift detected; use a new run name. "
            f"changed_sections={changed} stored={manifest_digest} current={current_digest}"
        )
    return current_digest


_BOUNDARY_MIGRATABLE_DESIGN_PATHS = frozenset(
    {
        # Revision 23 may add one receipt-bound trainer-only H3 provider at a
        # clean optimizer boundary.  Its complete identity remains in the
        # fingerprint; later cache/activation drift requires another explicit
        # boundary receipt and can never be recovered as legacy training.
        "learner.prize_plan_h3_actor_provider",
        "learner.games_per_batch",
        "learner.max_decisions_per_batch",
        "learner.warmup_max_decisions_per_batch",
        "learner.warmup_iterations",
        # One clean boundary may add the serialized, frozen zero-output bank.
        # Runtime activation and adapter fitting remain independently false.
        "learner.profile.matchup_adapters_enabled",
        # Receipt-backed fitting is an append-only learner capability. It may
        # enter only through the explicit completed-iteration migration path.
        "learner.dormant_matchup_adapter",
        # A source repair changes the checksum-bound training implementation.
        # The replacement curriculum receipt is fully revalidated by launch
        # preflight and may replace only its identity tuple at a clean boundary.
        "learner.current_deck_guide_training_mode",
        "learner.current_deck_guide_strategic_curriculum.curriculum_spec",
        "learner.current_deck_guide_strategic_curriculum.head_role_map",
        "learner.current_deck_guide_strategic_curriculum.validation_receipt",
        "learner.exact_gate_regression_margin",
        "learner.exact_gate_regression_patience",
        "games.per_iteration",
        "games.heldout",
        "gates.heldout_wr",
        "gates.heldout_per_opponent_floor",
        "gates.active_contract",
        "gates.frozen_specialist_registry",
        "opponents.collect",
        "opponents.heldout",
        "expert_rehearsal.rolling_manifest_pointer",
        "expert_rehearsal.loss_weights",
        "expert_rehearsal.minimum_decisions",
        "expert_rehearsal.required_target_coverage",
        # Receipt-gated, adapter-only expert rehearsal is additive. It may be
        # enabled only through an explicit completed-boundary migration.
        "expert_rehearsal.matchup_adapters",
        "collection.behavior_policy",
        # Revision 283 removes only the failed new tactical-sequence
        # head/route at the already sealed update-0 pre-gradient boundary.
        "r260_own_deck_successor.tactical_free_model_config_sha256",
        "r260_own_deck_successor.tactical_free_state_schema_sha256",
        # Explicitly recording the already-effective specialist/population
        # opponent eligibility is a schema-only boundary change.  The values
        # remain fingerprinted after migration, so a later scope change still
        # fails closed.
        "collection.training_opponent_scope",
        "collection.external_agents_training_eligible",
        "collection.auxiliary_targets.hidden_engine.digest",
        "collection.auxiliary_targets.hidden_engine.size",
        "collection.official_exploit",
        "collection.official_targeting",
        "collection.group_games_per_iteration",
        "collection.self_play_fraction",
        "collection.research_control_phase",
        "collection.strong_public_practice",
        # Owner revision 83 permits a checksum-identical remote endpoint to be
        # removed or restored at an uncommitted clean boundary while that host
        # runs an isolated device benchmark. The exact endpoint set remains
        # fingerprinted and every connected endpoint still passes the hard
        # checkpoint/runtime gate before collection.
        "remotes.endpoints",
        "remotes.disabled",
        "opponents.research_controls",
        "opponents.official_target_training",
        "measurement_deck_distribution",
        "source.source_tree_sha256",
    }
)


_DECISION_FUSION_WARMUP_MIGRATION_PATHS = frozenset(
    {
        "learner.current_deck_guide_archetype",
        "learner.current_deck_guide_loss_weight",
        "learner.current_deck_guide_targets_enabled",
        "learner.current_deck_guide_version",
        "learner.profile.decision_fusion_enabled",
        "learner.profile.decision_fusion_runtime_enabled",
        "learner.profile.decision_fusion_width",
    }
)

_DECISION_FUSION_RUNTIME_MIGRATION_PATHS = frozenset(
    {"learner.profile.decision_fusion_runtime_enabled"}
)


_ALAKAZAM_COMBO_ROUTE_DISABLE_MIGRATION_PATHS = frozenset(
    {"learner.profile.combo_state_route_enabled"}
)


_R284_ITERATION1_BOUNDARY_MIGRATION_PATHS = frozenset(
    {
        "collection.r280_gpu_resident_expert_refresh.boundaries",
        "games.heldout_additive_remotes",
        "remotes.disabled",
    }
)


_ALAKAZAM_R193_LARGE_REFRESH_MIGRATION_PATHS = frozenset(
    {"expert_rehearsal.one_time_override"}
)


_ALAKAZAM_R327_EVALUATION_MIGRATION_PATHS = frozenset(
    {
        "games.heldout",
        "gates.active_contract",
        "collection.research_control_phase",
        "collection.strong_public_practice.seed_contract.formal_games",
        "collection.strong_public_practice.seed_contract.research_control_games",
    }
)


_ALAKAZAM_R340_EXPERT_REHEARSAL_REPAIR_PATHS = frozenset(
    {"expert_rehearsal.minimum_decisions"}
)


_CURRENT_DECK_GUIDE_WEIGHT_MIGRATION_PATHS = frozenset(
    {
        "learner.alakazam_guide_loss_weight",
        "learner.current_deck_guide_loss_weight",
        "expert_rehearsal.loss_weights.alakazam_guide",
    }
)

_TEAL_AUXILIARY_HEAD_REBALANCE_PATHS = frozenset(
    {"learner.expanded_head_loss_weight_overrides"}
)

_ITERATION15_OPTIMIZER_CAP_RECOVERY_PATHS = frozenset(
    {
        "learner.max_decisions_per_batch",
        "learner.warmup_max_decisions_per_batch",
    }
)

_ITERATION15_R105_RELOCATION_PATHS = frozenset(
    {
        "gates.active_contract.path",
        "gates.frozen_specialist_registry.path",
        "learner.current_deck_guide_strategic_curriculum.curriculum_spec.path",
        "learner.current_deck_guide_strategic_curriculum.head_role_map.path",
        "learner.current_deck_guide_strategic_curriculum.validation_receipt.path",
    }
)

_FUSION_V3_CONTRACT_REPAIR_PATHS = frozenset(
    {
        "learner.profile.decision_fusion_action_type_reliability_cap",
        "learner.profile.decision_fusion_typed_output_centered_routes_enabled",
        "learner.trainable_parameters",
    }
)


_LATENT_LOOKAHEAD_R118_ACTIVATION_PATHS = frozenset(
    {
        "learner.profile.latent_lookahead_action_authority_enabled",
        "learner.profile.latent_lookahead_enabled",
        "learner.profile.latent_lookahead_policy_aid_cap",
        "learner.profile.latent_lookahead_width",
        "learner.trainable_parameters",
    }
)


_MARNIE_GUIDE_SHADOW_RETIREMENT_MIGRATION_PATHS = frozenset(
    {
        "learner.alakazam_guide_loss_weight",
        "learner.current_deck_guide_archetype",
        "learner.current_deck_guide_loss_weight",
        "learner.current_deck_guide_strategic_curriculum",
        "learner.current_deck_guide_targets_enabled",
        "learner.current_deck_guide_version",
        "learner.setup_board_outcome_loss_weight",
    }
)


def _safe_marnie_family_design_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only atomic family-sampler/loss activation or its rollback."""
    reason = str(reason or "").strip()
    if reason not in {
        "receipt_backed_marnie_family_activation_r133",
        "owner_ceiling_marnie_family_activation_r139",
        "receipt_backed_marnie_family_rollback_r133",
    }:
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    owner_ceiling_activation = (
        reason == "owner_ceiling_marnie_family_activation_r139"
    )
    independently_migratable = {
        path
        for path in non_source
        if any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _BOUNDARY_MIGRATABLE_DESIGN_PATHS
        )
    }
    family_scoped_paths = non_source - independently_migratable
    family_paths = {
        path
        for path in family_scoped_paths
        if path == "archetype_family"
        or path.startswith("archetype_family.")
        or path == "deck_distribution"
        or path.startswith("deck_distribution.")
    }
    guide_retirement_paths = family_scoped_paths - family_paths
    if not family_scoped_paths or not all(
        path == "archetype_family"
        or path.startswith("archetype_family.")
        or path == "deck_distribution"
        or path.startswith("deck_distribution.")
        or (
            owner_ceiling_activation
            and path in _MARNIE_GUIDE_SHADOW_RETIREMENT_MIGRATION_PATHS
        )
        for path in family_scoped_paths
    ):
        return False
    if not any(path.startswith("archetype_family") for path in non_source):
        return False
    if not any(path.startswith("deck_distribution") for path in non_source):
        return False
    if stored.get("measurement_deck_distribution") != current.get(
        "measurement_deck_distribution"
    ):
        return False
    before = stored.get("archetype_family")
    after = current.get("archetype_family")
    if reason in {
        "receipt_backed_marnie_family_activation_r133",
        "owner_ceiling_marnie_family_activation_r139",
    }:
        if before is not None or not isinstance(after, dict):
            return False
        if (
            after.get("schema") != "poke_bot.activated_archetype_family/v1"
            or after.get("family_id") != "marnie-s-grimmsnarl-ex"
            or after.get("sampler")
            != "deterministic_hamilton_family_macro_v1"
            or float(after.get("package_probability", -1.0)) != 0.20
            or after.get("measurement_deck_remains_singular") is not True
        ):
            return False
        for key in ("manifest", "loss_contract", "loss_vector"):
            row = after.get(key) or {}
            path = Path(str(row.get("path", ""))).expanduser().resolve()
            if (
                not path.is_file()
                or _sha256_file(path) != str(row.get("sha256", ""))
            ):
                return False
        if owner_ceiling_activation:
            learner = dict(current.get("learner") or {})
            rehearsal_weights = dict(
                ((current.get("expert_rehearsal") or {}).get("loss_weights"))
                or {}
            )
            if (
                guide_retirement_paths
                != _MARNIE_GUIDE_SHADOW_RETIREMENT_MIGRATION_PATHS
                or float(learner.get("alakazam_guide_loss_weight", -1.0))
                != 0.0
                or float(
                    learner.get("current_deck_guide_loss_weight", -1.0)
                )
                != 0.0
                or learner.get("current_deck_guide_targets_enabled") is not False
                or float(rehearsal_weights.get("alakazam_guide", -1.0))
                != 0.0
            ):
                return False
        return True
    return bool(
        isinstance(before, dict)
        and after is None
        and before.get("family_id") == "marnie-s-grimmsnarl-ex"
    )


def _safe_latent_lookahead_r118_activation(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only the owner-requested Generation-15 latent-head shape.

    This is intentionally narrower than the ordinary boundary allowlist: the
    protected Marnie parent must gain exactly the four latent-profile fields
    and the corresponding 412,130 trainable parameters.  No other model,
    collection, gate, optimizer, or opponent field may ride this migration.
    """
    if str(reason or "").strip() != "receipt_backed_latent_policy_activation_r118":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _LATENT_LOOKAHEAD_R118_ACTIVATION_PATHS:
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})
    before_profile = dict(before.get("profile") or {})
    after_profile = dict(after.get("profile") or {})
    before_params = int(before.get("trainable_parameters", -1))
    after_params = int(after.get("trainable_parameters", -1))
    return bool(
        before_profile.get("latent_lookahead_enabled") in (None, False)
        and before_profile.get("latent_lookahead_action_authority_enabled")
        in (None, False)
        and after_profile.get("latent_lookahead_enabled") is True
        and after_profile.get("latent_lookahead_action_authority_enabled") is True
        and int(after_profile.get("latent_lookahead_width", -1)) == 512
        and float(after_profile.get("latent_lookahead_policy_aid_cap", -1.0))
        == 0.25
        and before_params == 10_645_185
        and after_params == 11_057_315
        and after_params - before_params == 412_130
    )


def _safe_fusion_v3_contract_repair(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Repair only the iter-17 Fusion-v3 schema/19-route omission.

    The learner checkpoint is the architecture authority.  The prior design
    receipt omitted the two typed-routing fields and was written before its 19
    learned route-reliability scalars were represented in the parameter count.
    This is deliberately exact so it cannot authorize a wider model change.
    """
    if str(reason or "").strip() != "receipt_backed_fusion_v3_contract_repair_v1":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _FUSION_V3_CONTRACT_REPAIR_PATHS:
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})
    before_profile = dict(before.get("profile") or {})
    after_profile = dict(after.get("profile") or {})
    return bool(
        before_profile.get(
            "decision_fusion_typed_output_centered_routes_enabled"
        )
        is None
        and before_profile.get("decision_fusion_action_type_reliability_cap")
        is None
        and after_profile.get(
            "decision_fusion_typed_output_centered_routes_enabled"
        )
        is True
        and after_profile.get("decision_fusion_action_type_reliability_cap")
        == 0.25
        and int(before.get("trainable_parameters", -1)) == 10_645_166
        and int(after.get("trainable_parameters", -1)) == 10_645_185
        and after_profile.get("decision_fusion_enabled") is True
        and after_profile.get("decision_fusion_runtime_enabled") is True
    )


def _safe_iteration15_optimizer_cap_recovery(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Allow only the revision-105 2,048 -> 1,536 optimizer recovery."""
    if (
        str(reason or "").strip()
        != "receipt_backed_iteration15_optimizer_cap_recovery_v1"
    ):
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    relocation_paths = non_source - _ITERATION15_OPTIMIZER_CAP_RECOVERY_PATHS
    if (
        not _ITERATION15_OPTIMIZER_CAP_RECOVERY_PATHS.issubset(non_source)
        or not relocation_paths.issubset(_ITERATION15_R105_RELOCATION_PATHS)
    ):
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})

    def path_value(root: dict[str, Any], path: str) -> Any:
        value: Any = root
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    for path in relocation_paths:
        before_path = str(path_value(stored, path) or "")
        after_path = str(path_value(current, path) or "")
        if (
            "/final-format-alakazam-fusion-v3-r104/" not in before_path
            or after_path
            != before_path.replace(
                "/final-format-alakazam-fusion-v3-r104/",
                "/final-format-alakazam-fusion-v3-r105/",
            )
        ):
            return False
    return bool(
        int(before.get("games_per_batch", -1)) == 240
        and int(after.get("games_per_batch", -1)) == 240
        and int(before.get("warmup_iterations", -1)) == 30
        and int(after.get("warmup_iterations", -1)) == 30
        and int(before.get("max_decisions_per_batch", -1)) == 2048
        and int(before.get("warmup_max_decisions_per_batch", -1)) == 2048
        and int(after.get("max_decisions_per_batch", -1)) == 1536
        and int(after.get("warmup_max_decisions_per_batch", -1)) == 1536
    )


def _safe_teal_auxiliary_head_rebalance(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only Teal's measured tactical-outcome 0.05 -> 0.01 step."""
    if (
        str(reason or "").strip()
        != "receipt_backed_teal_auxiliary_head_rebalance_v1"
    ):
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _TEAL_AUXILIARY_HEAD_REBALANCE_PATHS:
        return False
    before = dict(
        (stored.get("learner") or {}).get(
            "expanded_head_loss_weight_overrides"
        )
        or {}
    )
    after = dict(
        (current.get("learner") or {}).get(
            "expanded_head_loss_weight_overrides"
        )
        or {}
    )
    return bool(
        before == {}
        and after == {"tactical_outcome": 0.01}
        and (stored.get("learner") or {}).get(
            "current_deck_guide_archetype"
        )
        == "teal-mask-ogerpon-ex"
        and (current.get("learner") or {}).get(
            "current_deck_guide_loss_weight"
        )
        == 0.05
    )


def _safe_current_deck_guide_weight_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Allow only an auxiliary guide-weight change at a clean boundary."""
    if (
        str(reason or "").strip()
        != "receipt_backed_current_deck_guide_weight_curve_v1"
    ):
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _CURRENT_DECK_GUIDE_WEIGHT_MIGRATION_PATHS:
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})
    before_rehearsal = dict(
        (stored.get("expert_rehearsal") or {}).get("loss_weights") or {}
    )
    after_rehearsal = dict(
        (current.get("expert_rehearsal") or {}).get("loss_weights") or {}
    )
    before_weight = float(before.get("current_deck_guide_loss_weight", -1.0))
    after_weight = float(after.get("current_deck_guide_loss_weight", -1.0))
    return bool(
        math.isfinite(before_weight)
        and math.isfinite(after_weight)
        and 0.0 <= before_weight <= 0.50
        and 0.0 <= after_weight <= 0.50
        and after_weight != before_weight
        and float(before.get("alakazam_guide_loss_weight", -1.0))
        == before_weight
        and float(after.get("alakazam_guide_loss_weight", -1.0))
        == after_weight
        and float(before_rehearsal.get("alakazam_guide", -1.0))
        == before_weight
        and float(after_rehearsal.get("alakazam_guide", -1.0))
        == after_weight
        and before.get("current_deck_guide_archetype")
        == after.get("current_deck_guide_archetype")
        and before.get("current_deck_guide_version")
        == after.get("current_deck_guide_version")
        and before.get("current_deck_guide_targets_enabled") is True
        and after.get("current_deck_guide_targets_enabled") is True
    )


def _safe_decision_fusion_warmup_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only the exact zero-safe, runtime-disabled fusion boundary."""
    if str(reason or "").strip() != "receipt_backed_decision_fusion_warmup_v1":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if not _DECISION_FUSION_WARMUP_MIGRATION_PATHS.issubset(non_source):
        return False
    # A receipt-backed fusion boundary may coincide with an independently safe
    # operational migration (for example a batch-size adjustment). Requiring
    # the fusion paths to equal the entire change set made those two strict
    # validators conflict. Permit only paths already accepted by the general
    # boundary allowlist; this does not broaden either contract.
    extra_paths = non_source - _DECISION_FUSION_WARMUP_MIGRATION_PATHS
    if any(
        not any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _BOUNDARY_MIGRATABLE_DESIGN_PATHS
        )
        for path in extra_paths
    ):
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})
    before_profile = dict(before.get("profile") or {})
    after_profile = dict(after.get("profile") or {})
    if (
        bool(before_profile.get("decision_fusion_enabled", False))
        or bool(before_profile.get("decision_fusion_runtime_enabled", False))
        or after_profile.get("decision_fusion_enabled") is not True
        or after_profile.get("decision_fusion_runtime_enabled") is not False
        or int(after_profile.get("decision_fusion_width", -1)) != 16
        or after_profile.get("expanded_heads_enabled") is not True
    ):
        return False
    # The guide fields are compatibility aliases only at this migration. They
    # must exactly mirror the already-authoritative legacy guide contract.
    if (
        after.get("current_deck_guide_loss_weight")
        != after.get("alakazam_guide_loss_weight")
        or after.get("current_deck_guide_targets_enabled")
        != after.get("alakazam_guide_targets_enabled")
        or after.get("current_deck_guide_archetype") is not None
        or after.get("current_deck_guide_version") is not None
    ):
        return False
    return True


def _safe_decision_fusion_runtime_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only audited serving activation of an already-trained fusion."""
    if str(reason or "").strip() != "receipt_backed_decision_fusion_runtime_v1":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if not _DECISION_FUSION_RUNTIME_MIGRATION_PATHS.issubset(non_source):
        return False
    extra_paths = non_source - _DECISION_FUSION_RUNTIME_MIGRATION_PATHS
    if any(
        not any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _BOUNDARY_MIGRATABLE_DESIGN_PATHS
        )
        for path in extra_paths
    ):
        return False
    before_profile = dict((stored.get("learner") or {}).get("profile") or {})
    after_profile = dict((current.get("learner") or {}).get("profile") or {})
    return bool(
        before_profile.get("expanded_heads_enabled") is True
        and before_profile.get("decision_fusion_enabled") is True
        and before_profile.get("decision_fusion_runtime_enabled") is False
        and after_profile.get("expanded_heads_enabled") is True
        and after_profile.get("decision_fusion_enabled") is True
        and after_profile.get("decision_fusion_runtime_enabled") is True
        and int(before_profile.get("decision_fusion_width", -1)) == 16
        and int(after_profile.get("decision_fusion_width", -1)) == 16
    )


def _safe_alakazam_combo_route_disable_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only Alakazam's owner-required combo-route shutdown.

    The legacy r175 design receipts predate the separately serialized runtime
    route gate, so a missing value means the resident combo route was active.
    Keep the H10 architecture/head present while allowing exactly that one
    route to become inactive at the empty iter-0 recovery boundary.
    """
    if str(reason or "").strip() != "receipt_backed_completed_collection_resume_v1":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _ALAKAZAM_COMBO_ROUTE_DISABLE_MIGRATION_PATHS:
        return False
    before = dict(stored.get("learner") or {})
    after = dict(current.get("learner") or {})
    before_profile = dict(before.get("profile") or {})
    after_profile = dict(after.get("profile") or {})
    return bool(
        before.get("current_deck_guide_archetype") == "alakazam"
        and after.get("current_deck_guide_archetype") == "alakazam"
        and before_profile.get("combo_state_head_enabled") is True
        and after_profile.get("combo_state_head_enabled") is True
        and before_profile.get("combo_state_route_enabled") in (None, True)
        and after_profile.get("combo_state_route_enabled") is False
    )


def _safe_alakazam_r193_large_refresh_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only the owner-r193 one-time 25-epoch Alakazam refresh."""
    if str(reason or "").strip() != (
        "owner_r193_large_expert_refresh_resubmit_post_iteration14"
    ):
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _ALAKAZAM_R193_LARGE_REFRESH_MIGRATION_PATHS:
        return False
    before = dict(stored.get("expert_rehearsal") or {})
    after = dict(current.get("expert_rehearsal") or {})
    learner = dict(current.get("learner") or {})
    override = dict(after.get("one_time_override") or {})
    return bool(
        learner.get("current_deck_guide_archetype") == "alakazam"
        and int(before.get("every_iterations") or -1) == 5
        and int(after.get("every_iterations") or -1) == 5
        and int(before.get("epochs") or -1) == 5
        and int(after.get("epochs") or -1) == 5
        and "one_time_override" not in before
        and int(override.get("before_iteration") or -1) == 15
        and int(override.get("epochs") or -1) == 25
    )


def _safe_alakazam_r327_evaluation_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Authorize only the 50/deck, no-research future evaluation migration."""

    if str(reason or "").strip() != "owner_r327_future_holdout_50_no_research":
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if not non_source or any(
        not any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _ALAKAZAM_R327_EVALUATION_MIGRATION_PATHS
        )
        for path in non_source
    ):
        return False
    before_games = dict(stored.get("games") or {})
    after_games = dict(current.get("games") or {})
    before_collection = dict(stored.get("collection") or {})
    after_collection = dict(current.get("collection") or {})
    before_research = dict(before_collection.get("research_control_phase") or {})
    after_research = dict(after_collection.get("research_control_phase") or {})
    before_practice = dict(before_collection.get("strong_public_practice") or {})
    after_practice = dict(after_collection.get("strong_public_practice") or {})
    before_seed = dict(before_practice.get("seed_contract") or {})
    after_seed = dict(after_practice.get("seed_contract") or {})
    return bool(
        int(before_games.get("heldout", -1)) == 4_500
        and int(after_games.get("heldout", -1)) == 900
        and before_collection.get("group_games_per_iteration")
        == after_collection.get("group_games_per_iteration")
        and before_research.get("enabled") is True
        and int(before_research.get("games_per_iteration", -1)) == 1_000
        and after_research.get("enabled") is False
        and int(after_research.get("games_per_iteration", -1)) == 0
        and int(after_research.get("legacy_reclaimed_training_slots", -1))
        == 1_000
        and after_research.get("disabled_by_revision_25") is True
        and before_research.get("registry") == after_research.get("registry")
        and int(before_seed.get("formal_games", -1)) == 4_500
        and int(after_seed.get("formal_games", -1)) == 900
        and int(before_seed.get("research_control_games", -1)) == 1_000
        and int(after_seed.get("research_control_games", -1)) == 0
        and before_practice.get("roster") == after_practice.get("roster")
    )


def _safe_alakazam_r340_expert_rehearsal_migration(
    *,
    stored: dict[str, Any],
    current: dict[str, Any],
    changed: Sequence[str],
    reason: Optional[str],
) -> bool:
    """Admit the exact protected 20-day Alakazam corpus at iteration 5."""

    if str(reason or "").strip() != (
        "owner_r340_protected_expert_rehearsal_threshold"
    ):
        return False
    non_source = {path for path in changed if not path.startswith("source.")}
    if non_source != _ALAKAZAM_R340_EXPERT_REHEARSAL_REPAIR_PATHS:
        return False
    before = dict(stored.get("expert_rehearsal") or {})
    after = dict(current.get("expert_rehearsal") or {})
    manifest_path = Path(str(after.get("rolling_manifest_pointer") or ""))
    pointer_path = manifest_path.with_name("PROTECTED_EXPERT_CORPUS.json")
    if (
        int(before.get("minimum_decisions") or -1) != 5_000_000
        or int(after.get("minimum_decisions") or -1) != 2_000_000
        or before.get("rolling_manifest_pointer")
        != after.get("rolling_manifest_pointer")
        or not manifest_path.is_file()
        or not pointer_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        raw_manifest = str(pointer.get("manifest") or "")
        candidate = Path(raw_manifest).expanduser()
        protected_manifest = (
            candidate.resolve()
            if candidate.is_absolute()
            else (pointer_path.parent / candidate).resolve()
        )
        decisions = int((manifest.get("totals") or {}).get("decisions_kept") or 0)
        return bool(
            manifest.get("format") == "pokebot-bootstrap-feature-manifest"
            and pointer.get("schema") == "poke_bot.pinned_expert_corpus/v1"
            and pointer.get("protected") is True
            and protected_manifest == manifest_path.resolve()
            and pointer.get("manifest_sha256") == _sha256_file(manifest_path)
            and decisions >= 2_000_000
            and decisions < 5_000_000
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _changed_design_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[str] = set()
        for key in set(before) | set(after):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(child)
            else:
                changed.update(_changed_design_paths(before[key], after[key], child))
        return changed
    return set() if before == after else {prefix or "<root>"}


def _load_design_migration_chain(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Verify the immutable manifest plus every append-only boundary migration."""
    contract = manifest.get("design_contract")
    digest = str(manifest.get("design_fingerprint") or "")
    if not digest or not isinstance(contract, dict):
        raise RuntimeError("resume lineage is missing its immutable design fingerprint")
    if _design_fingerprint(contract) != digest:
        raise RuntimeError("immutable manifest design contract digest is corrupt")

    receipts: list[dict[str, Any]] = []
    migration_dir = Path(run_dir) / "design_migrations"
    for path in sorted(migration_dir.glob("migration_*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if int(receipt.get("schema", -1)) != 1:
            raise RuntimeError(f"unsupported design migration receipt: {path}")
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        if not isinstance(previous, dict) or not isinstance(current, dict):
            raise RuntimeError(f"malformed design migration receipt: {path}")
        previous_digest = str(receipt.get("previous_fingerprint") or "")
        current_digest = str(receipt.get("current_fingerprint") or "")
        if (
            previous_digest != digest
            or _design_fingerprint(previous) != previous_digest
            or previous != contract
            or _design_fingerprint(current) != current_digest
        ):
            raise RuntimeError(f"broken design migration chain at {path}")
        contract = current
        digest = current_digest
        receipts.append(receipt)
    return contract, digest, receipts


def _validate_or_migrate_design_fingerprint(
    *,
    run_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    current: dict[str, Any],
    allow_clean_boundary_migration: bool,
    migration_reason: Optional[str],
    owner_r284_iteration1_boundary: bool,
) -> str:
    """Validate lineage, or append one tightly-scoped clean-boundary migration."""
    stored, stored_digest, receipts = _load_design_migration_chain(run_dir, manifest)
    state_digest = str(state.get("design_fingerprint") or "")

    # A receipt is written before the mutable ledger. Recover that one safe crash
    # window without mutating the original manifest or losing the committed state.
    if state_digest != stored_digest:
        latest = receipts[-1] if receipts else None
        if (
            latest is not None
            and state_digest == str(latest.get("previous_fingerprint") or "")
            and int(state.get("next_iteration", -1))
            == int(latest.get("boundary_next_iteration", -2))
        ):
            state["design_fingerprint"] = stored_digest
            state.setdefault("design_migration_history", []).append(
                {
                    "receipt": str(latest.get("receipt") or ""),
                    "fingerprint": stored_digest,
                    "recovered": True,
                }
            )
            _atomic_json(Path(run_dir) / "loop_state.json", state)
            state_digest = stored_digest
        else:
            raise RuntimeError(
                "resume design fingerprint disagrees with effective migration chain: "
                f"ledger={state_digest} effective={stored_digest}"
            )

    current_digest = _design_fingerprint(current)
    if current_digest == stored_digest:
        return current_digest
    changed = sorted(_changed_design_paths(stored, current))
    if not allow_clean_boundary_migration:
        raise RuntimeError(
            "resume design drift detected; use a new run name or an explicit clean-"
            "boundary migration. "
            f"changed_paths={changed} stored={stored_digest} current={current_digest}"
        )
    if not migration_reason or not str(migration_reason).strip():
        raise RuntimeError("clean-boundary design migration requires an operator reason")
    next_iteration = int(state.get("next_iteration", -1))
    last_completed = int(state.get("last_completed_iteration", -2))

    # A specialist ceiling reduction is safe at a committed boundary when the
    # new ceiling remains strictly beyond the next iteration.  Revision 113's
    # one exact Marnie extension (16 -> 21 launched iterations at the committed
    # 5 -> 6 boundary) is also safe and owner-authorized: it changes only the
    # future stopping boundary and cannot reinterpret an existing collection.
    # Every other increase, lowering past current progress, or run-identity
    # change remains forbidden. The paired strong-public seed-contract field
    # is already covered by the collection migration allowlist.
    stored_iterations = int((stored.get("run") or {}).get("iterations", -1))
    current_iterations = int((current.get("run") or {}).get("iterations", -1))
    safe_ceiling_reduction = bool(
        "run.iterations" in changed
        and stored_iterations > 0
        and next_iteration >= 0
        and next_iteration < current_iterations < stored_iterations
    )
    safe_marnie_revision113_ceiling_extension = bool(
        "run.iterations" in changed
        and stored_iterations == 16
        and current_iterations == 21
        and last_completed == 5
        and next_iteration == 6
        and str(migration_reason).strip() == "receipt_backed_opponent_tiers_r111"
    )
    safe_decision_fusion_warmup = _safe_decision_fusion_warmup_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason=migration_reason,
    )
    safe_decision_fusion_runtime = _safe_decision_fusion_runtime_migration(
        stored=stored,
        current=current,
        changed=changed,
        reason=migration_reason,
    )
    safe_alakazam_combo_route_disable = (
        _safe_alakazam_combo_route_disable_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_alakazam_r193_large_refresh = (
        _safe_alakazam_r193_large_refresh_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_alakazam_r327_evaluation = (
        _safe_alakazam_r327_evaluation_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_alakazam_r340_expert_rehearsal = (
        _safe_alakazam_r340_expert_rehearsal_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_alakazam_r334_terminal_refresh = bool(
        str(migration_reason or "").strip()
        == "owner_r334_terminal_one_epoch_expert_refresh"
        and next_iteration == 14
        and last_completed == 13
        and {path for path in changed if not path.startswith("source.")}
        == {
            "run.iterations",
            "collection.strong_public_practice.seed_contract.iterations",
            "expert_rehearsal.every_iterations",
            "expert_rehearsal.epochs",
        }
    )
    safe_current_deck_guide_weight = (
        _safe_current_deck_guide_weight_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_teal_auxiliary_head_rebalance = (
        _safe_teal_auxiliary_head_rebalance(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_iteration15_optimizer_cap_recovery = (
        _safe_iteration15_optimizer_cap_recovery(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_fusion_v3_contract_repair = _safe_fusion_v3_contract_repair(
        stored=stored,
        current=current,
        changed=changed,
        reason=migration_reason,
    )
    safe_latent_lookahead_r118_activation = (
        _safe_latent_lookahead_r118_activation(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    safe_marnie_family_design_migration = (
        _safe_marnie_family_design_migration(
            stored=stored,
            current=current,
            changed=changed,
            reason=migration_reason,
        )
    )
    disallowed = [
        path
        for path in changed
        if not any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in _BOUNDARY_MIGRATABLE_DESIGN_PATHS
        )
        and not path.startswith("source.")
        and not (
            path in {"run.iterations", "run.fixed_cycle_updates"}
            and safe_ceiling_reduction
        )
        and not (
            path == "run.iterations"
            and safe_marnie_revision113_ceiling_extension
        )
        and not (
            safe_alakazam_r334_terminal_refresh
            and path
            in {
                "run.iterations",
                "collection.strong_public_practice.seed_contract.iterations",
                "expert_rehearsal.every_iterations",
                "expert_rehearsal.epochs",
            }
        )
        and not (
            safe_decision_fusion_warmup
            and path in _DECISION_FUSION_WARMUP_MIGRATION_PATHS
        )
        and not (
            safe_decision_fusion_runtime
            and path in _DECISION_FUSION_RUNTIME_MIGRATION_PATHS
        )
        and not (
            safe_alakazam_combo_route_disable
            and path in _ALAKAZAM_COMBO_ROUTE_DISABLE_MIGRATION_PATHS
        )
        and not (
            owner_r284_iteration1_boundary
            and path in _R284_ITERATION1_BOUNDARY_MIGRATION_PATHS
        )
        and not (
            safe_alakazam_r193_large_refresh
            and path in _ALAKAZAM_R193_LARGE_REFRESH_MIGRATION_PATHS
        )
        and not (
            safe_current_deck_guide_weight
            and path in _CURRENT_DECK_GUIDE_WEIGHT_MIGRATION_PATHS
        )
        and not (
            safe_teal_auxiliary_head_rebalance
            and path in _TEAL_AUXILIARY_HEAD_REBALANCE_PATHS
        )
        and not (
            safe_fusion_v3_contract_repair
            and path in _FUSION_V3_CONTRACT_REPAIR_PATHS
        )
        and not (
            safe_latent_lookahead_r118_activation
            and path in _LATENT_LOOKAHEAD_R118_ACTIVATION_PATHS
        )
        and not (
            safe_marnie_family_design_migration
            and (
                path == "archetype_family"
                or path.startswith("archetype_family.")
                or path == "deck_distribution"
                or path.startswith("deck_distribution.")
                or (
                    str(migration_reason).strip()
                    == "owner_ceiling_marnie_family_activation_r139"
                    and path
                    in _MARNIE_GUIDE_SHADOW_RETIREMENT_MIGRATION_PATHS
                )
            )
        )
    ]
    if disallowed:
        raise RuntimeError(
            "clean-boundary design migration changes non-operational fields: "
            f"{disallowed}"
        )
    initial_collection_resume = bool(
        next_iteration == 0
        and last_completed == -1
        and str(migration_reason).strip()
        == "receipt_backed_completed_collection_resume_v1"
    )
    if (
        not initial_collection_resume
        and (next_iteration <= 0 or last_completed != next_iteration - 1)
    ):
        raise RuntimeError(
            "design migration requires a completed N+1 boundary: "
            f"last={last_completed} next={next_iteration}"
        )
    commit_path = Path(run_dir) / "commits" / f"iter_{last_completed:05d}.json"
    next_artifacts = _iteration_artifact_paths(run_dir, next_iteration)
    preserved_collection, _preserved_collection_contract = (
        _verified_completed_collection_across_design_chain(
            run_dir, state, manifest
        )
    )
    # No committed iteration history does not make the run empty after an
    # immutable iteration-0 collection receipt and shard exist.  That is the
    # exact initial-collection recovery transaction authorized above.
    initial_empty_run = bool(
        initial_collection_resume
        and not list(state.get("history") or ())
        and not next_artifacts
        and preserved_collection is None
    )
    expected_shard = (
        Path(run_dir) / "shards" / f"iter_{next_iteration:05d}.jsonl"
    ).resolve()
    expected_candidate = (
        Path(run_dir) / "checkpoints" / f"iter_{next_iteration:05d}.pt"
    ).resolve()
    expected_research = _research_control_result_path(
        run_dir, next_iteration
    ).resolve()
    next_artifact_set = {path.resolve() for path in next_artifacts}
    preserved_partial_collection = (
        _verified_partial_collection_resume(
            expected_shard, iteration=next_iteration
        )
        if expected_shard.is_file()
        else None
    )
    source_only_change = bool(changed) and all(
        path.startswith("source.") for path in changed
    )
    preserved_collection_fingerprint = str(
        (preserved_collection or {}).get("design_fingerprint_at_collection") or ""
    )
    latest_preserved_collection = dict(
        (receipts[-1].get("preserved_completed_collection") or {})
        if receipts
        else {}
    )
    collection_was_carried_to_stored_design = bool(
        preserved_collection is not None
        and latest_preserved_collection
        and int(latest_preserved_collection.get("iteration", -1))
        == int(preserved_collection.get("iteration", -2))
        and str(latest_preserved_collection.get("checkpoint_digest") or "")
        == str(preserved_collection.get("checkpoint_digest") or "")
    )
    # A zero-safe decision-fusion warmup may be committed while the completed
    # N+1 transaction is temporarily quarantined.  In that case migration N
    # cannot record ``preserved_completed_collection`` even though the
    # checksum-bound warmup receipt proves that the new learner is an exact
    # legacy-policy child of the behavior checkpoint that generated the shard.
    # A later source-only fix must not strand or recollect that transaction.
    collection_is_verified_zero_safe_parent = bool(
        preserved_collection is not None
        and _completed_collection_checkpoint_matches_state(
            run_dir=Path(run_dir),
            state=state,
            collection_digest=str(
                preserved_collection.get("checkpoint_digest") or ""
            ),
        )
    )
    # The immutable N+1 collection is governed by the design fingerprint bound
    # into its own receipt.  At the receipt-backed train boundary, a later
    # *allowlisted* operational or gate upgrade governs only the continuation.
    # Do not force recollection merely because its stricter terminal gate or a
    # recovery source fix now has a different current fingerprint.
    receipt_backed_allowed_migration = bool(
        preserved_collection is not None
        and (
            preserved_collection_fingerprint == stored_digest
            or (
                safe_alakazam_r340_expert_rehearsal
                and collection_was_carried_to_stored_design
            )
        )
        and all(
            path.startswith("source.")
            # A collection is bound to its original operational contract.
            # Only its future formal gate identity may be upgraded at this
            # recovery boundary; learner, scheduling, and collection-shape
            # changes must start a genuinely clean next transaction.
            or path == "collection.strong_public_practice.active_gate_id"
            # A receipt-backed terminal-ceiling reduction does not alter the
            # already sealed N+1 games.  The seed contract records the same
            # horizon and must migrate atomically with run.iterations.
            or (
                safe_ceiling_reduction
                and path
                in {
                    "run.iterations",
                    "run.fixed_cycle_updates",
                    "collection.strong_public_practice.seed_contract.iterations",
                }
            )
            or path == "gates.active_contract"
            or path.startswith("gates.active_contract.")
            or path
            in {
                "r260_own_deck_successor.tactical_free_model_config_sha256",
                "r260_own_deck_successor.tactical_free_state_schema_sha256",
            }
            or (
                safe_iteration15_optimizer_cap_recovery
                and path
                in (
                    _ITERATION15_OPTIMIZER_CAP_RECOVERY_PATHS
                    | _ITERATION15_R105_RELOCATION_PATHS
                )
            )
            or (
                safe_fusion_v3_contract_repair
                and path in _FUSION_V3_CONTRACT_REPAIR_PATHS
            )
            or (
                safe_alakazam_combo_route_disable
                and path in _ALAKAZAM_COMBO_ROUTE_DISABLE_MIGRATION_PATHS
            )
            or (
                owner_r284_iteration1_boundary
                and path in _R284_ITERATION1_BOUNDARY_MIGRATION_PATHS
            )
            or (
                safe_alakazam_r327_evaluation
                and any(
                    path == allowed or path.startswith(allowed + ".")
                    for allowed in _ALAKAZAM_R327_EVALUATION_MIGRATION_PATHS
                )
            )
            or (
                safe_alakazam_r340_expert_rehearsal
                and path in _ALAKAZAM_R340_EXPERT_REHEARSAL_REPAIR_PATHS
            )
            for path in changed
        )
    )
    artifacts_are_preserved_collection = bool(
        preserved_collection is not None
        and (
            initial_collection_resume
            or preserved_collection_fingerprint == current_digest
            or receipt_backed_allowed_migration
            or (
                source_only_change
                and (
                    preserved_collection_fingerprint == stored_digest
                    or collection_was_carried_to_stored_design
                    or collection_is_verified_zero_safe_parent
                )
            )
        )
        and next_artifact_set == {expected_shard}
    )
    artifacts_are_preserved_trained_candidate = False
    if (
        preserved_collection is not None
        and next_artifact_set
        in (
            {expected_shard, expected_candidate},
            {expected_shard, expected_candidate, expected_research},
        )
    ):
        candidate_result = _verified_orphan_candidate_result(
            expected_candidate,
            iteration=next_iteration,
            parent_digest=_orphan_recovery_parent_digest(
                Path(run_dir), state, next_iteration
            ),
            behavior_digest=str(
                preserved_collection.get("checkpoint_digest") or ""
            ),
            design_fingerprint=_orphan_recovery_design_fingerprints(
                state,
                preserved_collection.get("design_fingerprint_at_collection"),
                stored_digest,
                current_digest,
            ),
            shard_path=expected_shard,
            r241_peak_r195_training_contract=dict(
                current.get("r241_peak_r195_training_contract") or {}
            ),
            expected_awr_provider=(
                (current.get("learner") or {}).get(
                    "prize_plan_h3_actor_provider"
                )
            ),
        )
        if expected_research in next_artifact_set and not (
            _safe_research_control_recovery_artifact(
                expected_research,
                iteration=next_iteration,
                candidate_digest=str(candidate_result["candidate_digest"]),
            )
        ):
            raise RuntimeError(
                "preserved research-control result is not a safe nontraining "
                "artifact for the trained candidate"
            )
        artifacts_are_preserved_trained_candidate = True
    partial_source_migration_receipt_ok = False
    partial_source_receipt_path = os.environ.get(
        "POKEBOT_PARTIAL_SOURCE_MIGRATION_RECEIPT", ""
    ).strip()
    if (
        source_only_change
        and preserved_partial_collection is not None
        and partial_source_receipt_path
        and next_artifact_set == {expected_shard}
    ):
        try:
            source_receipt = json.loads(
                Path(partial_source_receipt_path).read_text(encoding="utf-8")
            )
            source_root = Path(__file__).resolve().parent.parent
            repair = dict(source_receipt.get("repair") or {})
            partial = dict(source_receipt.get("partial_resume") or {})
            partial_source_migration_receipt_ok = bool(
                source_receipt.get("schema")
                == "poke_bot.alakazam_r274_dual_gpu_router_fix/v1"
                and source_receipt.get("status")
                == "managed_restart_authorized_and_partial_collection_sealed"
                and source_receipt.get("previous_design_fingerprint")
                == stored_digest
                and source_receipt.get("current_design_fingerprint")
                == current_digest
                and partial.get("shard_sha256")
                == preserved_partial_collection.get("shard_sha256")
                and partial.get("sidecar_sha256")
                == _sha256_file(
                    Path(preserved_partial_collection["sidecar_path"])
                )
                and repair.get("hardware_source_sha256")
                == _sha256_file(source_root / "poke_bot/pure_rl/hardware.py")
                and repair.get("multi_env_source_sha256")
                == _sha256_file(
                    source_root / "poke_bot/pure_rl/multi_env_self_play.py"
                )
                and repair.get("trainer_source_sha256")
                == _sha256_file(Path(__file__).resolve())
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            partial_source_migration_receipt_ok = False

    artifacts_are_preserved_transaction = bool(
        artifacts_are_preserved_collection
        or artifacts_are_preserved_trained_candidate
        or (
            preserved_partial_collection is not None
            and next_artifact_set == {expected_shard}
            and (
                source_only_change
                or {
                    path for path in changed if not path.startswith("source.")
                }
                == {
                    "games.heldout_additive_remotes",
                    "remotes.disabled",
                    "remotes.endpoints",
                }
            )
            and str(migration_reason).strip()
            == "receipt_backed_completed_collection_resume_v1"
        )
        or (
            initial_collection_resume
            and preserved_partial_collection is not None
            and next_artifact_set == {expected_shard}
        )
        or partial_source_migration_receipt_ok
    )
    if (
        (
            not initial_collection_resume
            and not commit_path.is_file()
            and not artifacts_are_preserved_trained_candidate
        )
        or (next_artifacts and not artifacts_are_preserved_transaction)
        or (
            initial_empty_run
            and (
                next_artifacts
                or preserved_collection is not None
            )
        )
        or (
            initial_collection_resume
            and not initial_empty_run
            and (
                (
                    preserved_collection is None
                    or int(preserved_collection.get("iteration", -1)) != 0
                    or not artifacts_are_preserved_transaction
                )
                and preserved_partial_collection is None
            )
        )
    ):
        raise RuntimeError(
            "design migration requires a clean boundary or one verified completed "
            "collection shard with no train/eval artifacts; "
            f"next={next_iteration} last={last_completed} changed={changed} "
            f"source_only={source_only_change} "
            f"preserved_collection={preserved_collection is not None} "
            f"preserved_fingerprint={preserved_collection_fingerprint} "
            f"stored={stored_digest} current={current_digest} "
            f"next_artifacts={sorted(str(path) for path in next_artifact_set)} "
            f"preserved_collection_ok={artifacts_are_preserved_collection} "
            f"preserved_partial_ok={preserved_partial_collection is not None} "
            f"preserved_transaction_ok={artifacts_are_preserved_transaction}"
        )
    committed = (
        json.loads(commit_path.read_text(encoding="utf-8"))
        if commit_path.is_file()
        else {}
    )
    commit_digest = str(committed.get("design_fingerprint") or "")
    same_boundary_digests = {stored_digest}
    for prior_receipt in receipts:
        if int(prior_receipt.get("boundary_next_iteration", -1)) == next_iteration:
            same_boundary_digests.add(
                str(prior_receipt.get("previous_fingerprint") or "")
            )
            same_boundary_digests.add(
                str(prior_receipt.get("current_fingerprint") or "")
            )
    if not initial_collection_resume and (
        int(committed.get("next_iteration", -1)) != next_iteration
        or commit_digest not in same_boundary_digests
    ):
        raise RuntimeError("boundary commit does not match the effective design lineage")
    for field in ("games_per_batch", "max_decisions_per_batch"):
        if int(current["learner"][field]) <= 0:
            raise RuntimeError(f"migrated learner.{field} must remain positive")
    if (
        isinstance(current.get("games"), dict)
        and "per_iteration" in current["games"]
        and int(current["games"]["per_iteration"]) <= 0
    ):
        raise RuntimeError("migrated games.per_iteration must remain positive")
    try:
        _scheduled_train_decision_cap(
            next_iteration,
            steady_cap=int(current["learner"]["max_decisions_per_batch"]),
            warmup_cap=int(
                current["learner"].get("warmup_max_decisions_per_batch", 0)
            ),
            warmup_iterations=int(
                current["learner"].get("warmup_iterations", 0)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid migrated learner batch schedule: {exc}") from exc

    migration_dir = Path(run_dir) / "design_migrations"
    migration_dir.mkdir(parents=True, exist_ok=True)
    index = len(receipts) + 1
    receipt_path = migration_dir / f"migration_{index:04d}.json"
    receipt = {
        "schema": 1,
        "receipt": str(receipt_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": str(migration_reason).strip(),
        "boundary_next_iteration": next_iteration,
        "last_completed_iteration": last_completed,
        "changed_paths": changed,
        "previous_fingerprint": stored_digest,
        "current_fingerprint": current_digest,
        "previous_contract": stored,
        "current_contract": current,
        "preserved_completed_collection": (
            {
                "receipt": str(preserved_collection.get("receipt_path") or ""),
                "iteration": int(preserved_collection["iteration"]),
                "checkpoint_digest": str(
                    preserved_collection["checkpoint_digest"]
                ),
            }
            if artifacts_are_preserved_transaction
            and preserved_collection is not None
            else None
        ),
        "preserved_partial_collection": (
            {
                "sidecar": str(preserved_partial_collection["sidecar_path"]),
                "sidecar_sha256": _sha256_file(
                    Path(preserved_partial_collection["sidecar_path"])
                ),
                "iteration": int(preserved_partial_collection["iteration"]),
                "retained_source_games": int(
                    preserved_partial_collection["retained_source_games"]
                ),
                "missing_source_games": len(
                    preserved_partial_collection["missing_job_indices"]
                ),
                "shard_sha256": str(
                    preserved_partial_collection["shard_sha256"]
                ),
            }
            if preserved_partial_collection is not None
            else None
        ),
        "preserved_trained_candidate": (
            str(expected_candidate)
            if artifacts_are_preserved_trained_candidate
            else None
        ),
    }
    _write_json_exclusive(receipt_path, receipt)
    state["design_fingerprint"] = current_digest
    state.setdefault("design_migration_history", []).append(
        {
            "receipt": str(receipt_path),
            "fingerprint": current_digest,
            "boundary_next_iteration": next_iteration,
            "reason": str(migration_reason).strip(),
        }
    )
    _atomic_json(Path(run_dir) / "loop_state.json", state)
    print(
        f"[pure_rl] DESIGN_MIGRATION boundary_next={next_iteration} "
        f"changed={changed} receipt={receipt_path}",
        flush=True,
    )
    return current_digest


def _iteration_artifact_paths(run_dir: Path, iteration: int) -> list[Path]:
    stem = f"iter_{int(iteration):05d}"
    paths_out = [
        run_dir / "shards" / f"{stem}.jsonl",
        run_dir / "checkpoints" / f"{stem}.pt",
        run_dir / "eval" / f"{stem}.json",
        run_dir / "metrics" / f"{stem}.json",
        run_dir / "research_controls" / f"{stem}.json",
    ]
    for parent, pattern in (
        (run_dir / "checkpoints", f"{stem}.pt.tmp.*"),
        (run_dir / "shards", f".{stem}*.tmp"),
    ):
        paths_out.extend(sorted(parent.glob(pattern)))
    latest = run_dir / "metrics" / "latest.json"
    if latest.is_file():
        try:
            if int(json.loads(latest.read_text(encoding="utf-8")).get("iteration")) == int(
                iteration
            ):
                paths_out.append(latest)
        except Exception:
            # A torn/malformed mutable pointer is also recovery debris.
            paths_out.append(latest)
    return [path for path in paths_out if path.exists()]


_COMPLETED_COLLECTION_SCHEMA = "poke_bot.completed_collection/v1"
_PARTIAL_COLLECTION_RESUME_SCHEMA = (
    "poke_bot.partial_collection_resume_authorization/v1"
)
_TRAINING_SEAT_SPLIT_SCHEMA = "poke_bot.alakazam_refresh_seat_split/v1"
_TRAINING_SEAT_SPLIT_SUMMARY_SCHEMA = (
    "poke_bot.alakazam_refresh_seat_split_summary/v1"
)


def _training_seat_stage(
    stage: str,
    seats: Sequence[Any],
    *,
    expected_total: Optional[int] = None,
) -> dict[str, Any]:
    """Return a fail-closed two-seat accounting stage.

    Seats are counted at the policy/source-game boundary supplied by the
    caller. Values other than the two engine seats are evidence failures, not
    values that can be rounded away to preserve a nominal 50/50 claim.
    """

    counts = {"seat0": 0, "seat1": 0}
    invalid: list[str] = []
    for raw in seats:
        try:
            seat = int(raw)
        except (TypeError, ValueError):
            invalid.append(repr(raw))
            continue
        if seat == 0:
            counts["seat0"] += 1
        elif seat == 1:
            counts["seat1"] += 1
        else:
            invalid.append(str(seat))
    total = int(counts["seat0"] + counts["seat1"])
    expected_matches = expected_total is None or total == int(expected_total)
    passed = bool(
        not invalid
        and total > 0
        and total % 2 == 0
        and counts["seat0"] == counts["seat1"]
        and expected_matches
    )
    return {
        "stage": str(stage),
        "seat0": int(counts["seat0"]),
        "seat1": int(counts["seat1"]),
        "total": total,
        "expected_total": (
            int(expected_total) if expected_total is not None else None
        ),
        "invalid_seats": invalid,
        "exact_50_50": passed,
    }


def _assert_exact_training_seat_stage(stage: dict[str, Any]) -> None:
    if stage.get("exact_50_50") is not True:
        raise RuntimeError(
            "exact 50/50 training-seat contract failed: "
            f"stage={stage.get('stage')} seat0={stage.get('seat0')} "
            f"seat1={stage.get('seat1')} total={stage.get('total')} "
            f"expected={stage.get('expected_total')} "
            f"invalid={stage.get('invalid_seats')}"
        )


def _assert_valid_replay_seat_projection(stage: dict[str, Any]) -> None:
    """Validate replay provenance without imposing source-game scheduling on it.

    A source game has exactly one *assigned* learner seat, which is the
    population governed by the final-format 50/50 contract.  A self-play game
    can, however, emit records from both player perspectives.  The replay
    window additionally spans immutable adjacent shards.  Its raw sequence
    count is therefore a projection of the scheduled games, not a second game
    scheduler and is not required to be even.
    """

    if (
        stage.get("invalid_seats")
        or int(stage.get("total", 0)) <= 0
        or int(stage.get("total", -1)) != int(stage.get("expected_total", -2))
    ):
        raise RuntimeError(
            "invalid replay-seat projection: "
            f"stage={stage.get('stage')} seat0={stage.get('seat0')} "
            f"seat1={stage.get('seat1')} total={stage.get('total')} "
            f"expected={stage.get('expected_total')} "
            f"invalid={stage.get('invalid_seats')}"
        )


def _replay_sampling_design_fingerprint(
    *, current_design_fingerprint: str, collection_receipt: dict[str, Any]
) -> str:
    """Keep replay resampling stable across recovery-only source migrations."""

    fingerprint = str(
        collection_receipt.get("design_fingerprint_at_collection")
        or current_design_fingerprint
    )
    if not fingerprint.startswith("sha256:"):
        raise RuntimeError("replay sampling lacks a valid design fingerprint")
    return fingerprint


def _commit_training_seat_split_receipt(
    *,
    run_dir: Path,
    iteration: int,
    design_fingerprint: str,
    collection_receipt: dict[str, Any],
    sequences: Sequence[Any],
) -> dict[str, Any]:
    """Bind source-game seat populations and the derived replay projection."""

    collection_receipt_path = Path(
        collection_receipt.get("receipt_path")
        or _collection_receipt_path(run_dir, iteration)
    )
    if not collection_receipt_path.is_file():
        raise RuntimeError(
            "training-seat receipt cannot bind missing collection receipt: "
            f"{collection_receipt_path}"
        )
    collection_evidence = collection_receipt
    if (
        collection_receipt.get("shard") is not None
        and not isinstance(collection_receipt.get("shard"), dict)
    ):
        try:
            sealed_collection = json.loads(
                collection_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "training-seat recovery cannot read its sealed collection receipt"
            ) from exc
        if (
            not isinstance(sealed_collection, dict)
            or sealed_collection.get("schema") != _COMPLETED_COLLECTION_SCHEMA
            or int(sealed_collection.get("iteration", -1)) != int(iteration)
        ):
            raise RuntimeError(
                "training-seat recovery found an invalid sealed collection receipt"
            )
        collection_evidence = {
            **sealed_collection,
            "receipt_path": str(collection_receipt_path),
        }
    # Seat receipts describe the already-collected population, so bind them to
    # the design that produced that collection.  A recovery-only source patch
    # may legitimately change the current design fingerprint without changing
    # the immutable collection or its seat assignment.
    receipt_design_fingerprint = str(
        collection_evidence.get("design_fingerprint_at_collection")
        or design_fingerprint
    )
    if not receipt_design_fingerprint.startswith("sha256:"):
        raise RuntimeError(
            "training-seat receipt lacks a valid collection design fingerprint"
        )
    split = dict(
        (collection_evidence.get("stats") or {}).get("training_seat_split")
        or {}
    )
    assigned = dict(split.get("assigned_source_games") or {})
    actual = dict(split.get("retained_source_games") or {})
    _assert_exact_training_seat_stage(assigned)
    _assert_exact_training_seat_stage(actual)
    consumed = _training_seat_stage(
        "replay_sequences_consumed",
        [getattr(sequence, "seat", None) for sequence in sequences],
        expected_total=len(sequences),
    )
    _assert_valid_replay_seat_projection(consumed)
    sequence_identity_digest = _canonical_digest(
        [
            {
                "episode_id": str(getattr(sequence, "episode_id", "")),
                "seat": int(getattr(sequence, "seat", -1)),
                "archetype": str(getattr(sequence, "archetype", "")),
                "source": str(getattr(sequence, "source", "")),
            }
            for sequence in sequences
        ]
    )
    receipt_root = Path(run_dir) / "seat_split_receipts"
    receipt_path = receipt_root / f"iter_{int(iteration):05d}.index.json"

    # A deployment-root migration can relocate the same hard-linked run tree
    # after collection and seat receipts have already been sealed.  Their
    # embedded absolute paths are immutable audit evidence, not mutable runtime
    # pointers.  Validate the existing index semantically and reuse it byte for
    # byte instead of rebuilding it with the new root spelling.
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        sequence_digest_matches = bool(
            existing.get("sequence_identity_digest") == sequence_identity_digest
        )
        migrated_recovery = bool(design_fingerprint != receipt_design_fingerprint)
        sealed_shard = dict(collection_evidence.get("shard") or {})
        sealed_replay = dict(collection_evidence.get("replay_cache") or {})
        migrated_replay_evidence_valid = bool(
            migrated_recovery
            and str(sealed_shard.get("sha256") or "").startswith("sha256:")
            # Family sampling can emit more replay sequences than the exact
            # scheduled source-game population (for example, both self-play
            # perspectives).  The cache must match the sealed shard, not the
            # independently audited assigned/retained source-game count.
            and int(sealed_shard.get("games", 0)) > 0
            and int(sealed_replay.get("sequences", -1))
            == int(sealed_shard.get("games", -2))
            and int(consumed.get("total", -1))
            >= int(sealed_replay.get("sequences", 0))
        )
        expected_populations = {
            "assigned_source_games": assigned,
            "retained_source_games": actual,
            "replay_sequences_consumed": consumed,
        }
        recovery_mismatches = {
            key: {"sealed": existing.get(key), "recovered": value}
            for key, value in expected_populations.items()
            if existing.get(key) != value
        }
        invariant_mismatches = {
            "schema": existing.get("schema")
            != "poke_bot.alakazam_refresh_seat_split_index/v1",
            "iteration": int(existing.get("iteration", -1)) != int(iteration),
            "design_fingerprint": existing.get("design_fingerprint")
            != receipt_design_fingerprint,
            "policy": existing.get("policy")
            != "exact_50_50_first_second_training",
            "second_seat_priority": existing.get("second_seat_priority") is not False,
            "passed": existing.get("passed") is not True,
            "sequence_identity": not sequence_digest_matches
            and not migrated_replay_evidence_valid,
        }
        invariant_mismatches = {
            key: value for key, value in invariant_mismatches.items() if value
        }
        if invariant_mismatches or recovery_mismatches:
            raise RuntimeError(
                "immutable training-seat receipt changed on recovery: "
                + json.dumps(
                    {
                        "invariants": invariant_mismatches,
                        "populations": recovery_mismatches,
                        "sequence_digest_matches": sequence_digest_matches,
                        "migrated_replay_evidence_valid": (
                            migrated_replay_evidence_valid
                        ),
                    },
                    sort_keys=True,
                )
            )
        bound_collection = dict(existing.get("collection_receipt") or {})
        if bound_collection.get("sha256") != _sha256_file(collection_receipt_path):
            raise RuntimeError(
                "immutable training-seat receipt collection digest changed on recovery"
            )
        existing_stage_receipts = dict(existing.get("stage_receipts") or {})
        for stage_name in ("assigned", "actual", "consumed"):
            binding = dict(existing_stage_receipts.get(stage_name) or {})
            bound_path = Path(str(binding.get("path") or ""))
            if (
                not bound_path.is_file()
                or binding.get("sha256") != _sha256_file(bound_path)
            ):
                raise RuntimeError(
                    f"immutable {stage_name} training-seat receipt binding changed"
                )
        print(
            "[pure_rl] exact training-seat split reused "
            f"iter={iteration} receipt={receipt_path} "
            f"sequence_order_digest_match={int(sequence_digest_matches)} "
            f"migrated_shard_proof={int(migrated_replay_evidence_valid)}",
            flush=True,
        )
        return {**existing, "receipt_path": str(receipt_path)}

    stage_manifest_digests = dict(split.get("stage_manifest_sha256") or {})
    stage_manifest_digests["consumed"] = sequence_identity_digest
    stages = {
        "assigned": assigned,
        "actual": actual,
        "consumed": consumed,
    }
    stage_receipts: dict[str, dict[str, Any]] = {}
    for stage_name, stage_row in stages.items():
        stage_path = (
            receipt_root
            / f"iter_{int(iteration):05d}.{stage_name}.json"
        )
        is_source_game_stage = stage_name in {"assigned", "actual"}
        immutable_stage = {
            "schema": _TRAINING_SEAT_SPLIT_SCHEMA,
            "template_only": False,
            "status": "issued_passed",
            "runtime_authority": "none",
            "specialist_id": "alakazam",
            "research_derivative": "final_format_alakazam_h10_i",
            "iteration": int(iteration),
            "design_fingerprint": receipt_design_fingerprint,
            "stage": stage_name,
            "allowed_stages": ["assigned", "actual", "consumed"],
            "first_games": int(stage_row["seat0"]),
            "second_games": int(stage_row["seat1"]),
            "total_games": int(stage_row["total"]),
            # The first two rows are the exact scheduled source-game
            # population.  ``consumed`` is an integrity-audited replay
            # projection and intentionally may contain both perspectives of a
            # self-play source game.
            "exact_even_split": bool(is_source_game_stage),
            "seat_balance_applicable": bool(is_source_game_stage),
            "deterministic_assignment_manifest_sha256": str(
                stage_manifest_digests.get(stage_name) or ""
            ),
            "second_focus_1_to_7_used": False,
            "package_preference": "first_if_allowed",
        }
        if not immutable_stage["deterministic_assignment_manifest_sha256"].startswith(
            "sha256:"
        ):
            raise RuntimeError(
                f"training-seat stage {stage_name} lacks its manifest digest"
            )
        if stage_path.is_file():
            existing_stage = json.loads(stage_path.read_text(encoding="utf-8"))
            comparable_stage = {
                key: value
                for key, value in existing_stage.items()
                if key != "issued_at_utc"
            }
            if comparable_stage != immutable_stage:
                raise RuntimeError(
                    f"immutable {stage_name} training-seat receipt changed"
                )
        else:
            existing_stage = {
                **immutable_stage,
                "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_exclusive(stage_path, existing_stage)
        stage_receipts[stage_name] = {
            "path": str(stage_path),
            "sha256": _sha256_file(stage_path),
        }

    immutable = {
        "schema": "poke_bot.alakazam_refresh_seat_split_index/v1",
        "iteration": int(iteration),
        "design_fingerprint": receipt_design_fingerprint,
        "policy": "exact_50_50_first_second_training",
        "second_seat_priority": False,
        "assigned_source_games": assigned,
        "retained_source_games": actual,
        "replay_sequences_consumed": consumed,
        "sequence_identity_digest": sequence_identity_digest,
        "stage_receipts": stage_receipts,
        "collection_receipt": {
            "path": str(collection_receipt_path),
            "sha256": _sha256_file(collection_receipt_path),
        },
        "passed": True,
    }
    payload = {
        **immutable,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_exclusive(receipt_path, payload)
    print(
        "[pure_rl] exact training-seat split committed "
        f"iter={iteration} assigned={assigned['seat0']}/{assigned['seat1']} "
        f"retained={actual['seat0']}/{actual['seat1']} "
        f"replay_projection={consumed['seat0']}/{consumed['seat1']} "
        f"receipt={receipt_path}",
        flush=True,
    )
    return {**payload, "receipt_path": str(receipt_path)}


def _commit_expert_rehearsal_seat_split_receipts(
    *,
    run_dir: Path,
    before_iteration: int,
    design_fingerprint: str,
    evidence: dict[str, Any],
    pack_info: dict[str, Any],
) -> dict[str, Any]:
    """Bind the exact even expert view actually packed for rehearsal.

    The immutable public manifest remains untouched.  This receipt identifies
    the deterministic subset assigned from it, the checksummed CPU pack that
    materialized that subset, and the same pack consumed by the rehearsal
    optimizer.  Both train and validation partitions must independently be
    even so no gradient-bearing partition can hide a seat skew.
    """

    if not (
        evidence.get("schema")
        == "poke_bot.expert_rehearsal_seat_selection/v1"
        and evidence.get("passed") is True
        and evidence.get("second_focus_1_to_7_used") is False
        and evidence.get("package_preference") == "first_if_allowed"
    ):
        raise RuntimeError("expert rehearsal lacks exact-seat selection evidence")
    partitions = dict(evidence.get("partitions") or {})
    first_games = 0
    second_games = 0
    for name in ("train", "validation"):
        row = dict(partitions.get(name) or {})
        first = int(row.get("first_games", -1))
        second = int(row.get("second_games", -1))
        total = int(row.get("total_games", -1))
        if not (
            row.get("exact_even_split") is True
            and first > 0
            and first == second
            and total == first + second
        ):
            raise RuntimeError(
                f"expert rehearsal {name} partition is not exact 50/50"
            )
        first_games += first
        second_games += second
    if first_games != second_games:
        raise RuntimeError("expert rehearsal aggregate seat split is not exact")

    pack_manifest = Path(str(pack_info.get("manifest") or "")).resolve()
    if not pack_manifest.is_file():
        raise RuntimeError("expert rehearsal CPU-pack manifest is missing")
    pack_manifest_sha256 = _sha256_file(pack_manifest)
    assignment_sha256 = str(
        evidence.get("deterministic_assignment_manifest_sha256") or ""
    )
    if not assignment_sha256.startswith("sha256:"):
        raise RuntimeError("expert rehearsal assignment digest is invalid")

    receipt_root = Path(run_dir) / "seat_split_receipts"
    stage_digests = {
        "assigned": assignment_sha256,
        "actual": _canonical_digest(
            {
                "assignment": assignment_sha256,
                "cpu_pack_key": str(pack_info.get("key") or ""),
                "cpu_pack_manifest_sha256": pack_manifest_sha256,
                "partitions": partitions,
            }
        ),
        "consumed": _canonical_digest(
            {
                "cpu_pack_manifest_sha256": pack_manifest_sha256,
                "selected_games": int(evidence.get("selected_games", -1)),
                "selected_packed_decisions": int(
                    evidence.get("selected_packed_decisions", -1)
                ),
                "partitions": partitions,
            }
        ),
    }
    stage_receipts: dict[str, dict[str, str]] = {}
    for stage in ("assigned", "actual", "consumed"):
        stage_path = receipt_root / (
            f"rehearsal_before_iter_{int(before_iteration):05d}.{stage}.json"
        )
        immutable_stage = {
            "schema": _TRAINING_SEAT_SPLIT_SCHEMA,
            "template_only": False,
            "status": "issued_passed",
            "runtime_authority": "none",
            "specialist_id": "alakazam",
            "research_derivative": "final_format_alakazam_h10_i",
            "iteration": int(before_iteration),
            "training_phase": "expert_rehearsal",
            "stage": stage,
            "allowed_stages": ["assigned", "actual", "consumed"],
            "first_games": int(first_games),
            "second_games": int(second_games),
            "total_games": int(first_games + second_games),
            "exact_even_split": True,
            "deterministic_assignment_manifest_sha256": stage_digests[stage],
            "second_focus_1_to_7_used": False,
            "package_preference": "first_if_allowed",
        }
        if stage_path.is_file():
            existing = json.loads(stage_path.read_text(encoding="utf-8"))
            if {
                key: value
                for key, value in existing.items()
                if key != "issued_at_utc"
            } != immutable_stage:
                raise RuntimeError(
                    f"immutable rehearsal {stage} seat receipt changed"
                )
        else:
            _write_json_exclusive(
                stage_path,
                {
                    **immutable_stage,
                    "issued_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        stage_receipts[stage] = {
            "path": str(stage_path),
            "sha256": _sha256_file(stage_path),
        }

    index_path = receipt_root / (
        f"rehearsal_before_iter_{int(before_iteration):05d}.index.json"
    )
    immutable_index = {
        "schema": (
            "poke_bot.alakazam_refresh_rehearsal_seat_split_index/v1"
        ),
        "before_iteration": int(before_iteration),
        "design_fingerprint": str(design_fingerprint),
        "policy": "exact_50_50_first_second_training",
        "second_seat_priority": False,
        "source_manifest_seats": dict(evidence.get("source") or {}),
        "partitions": partitions,
        "selected_games": int(evidence.get("selected_games", -1)),
        "selected_raw_decisions": int(
            evidence.get("selected_raw_decisions", -1)
        ),
        "selected_packed_decisions": int(
            evidence.get("selected_packed_decisions", -1)
        ),
        "cpu_pack": {
            "key": str(pack_info.get("key") or ""),
            "manifest": str(pack_manifest),
            "manifest_sha256": pack_manifest_sha256,
        },
        "stage_receipts": stage_receipts,
        "package_preference": "first_if_allowed",
        "second_focus_1_to_7_used": False,
        "passed": True,
    }
    if index_path.is_file():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
        existing_immutable = {
            key: value
            for key, value in existing_index.items()
            if key != "issued_at_utc"
        }
        if existing_immutable != immutable_index:
            # A checksum-gated source repair can legitimately advance the
            # loop design fingerprint after this exact CPU pack and its three
            # immutable seat stages were committed.  Preserve the original
            # receipt in that recovery case; every gradient-bearing identity
            # still has to match byte-for-byte.
            existing_without_design = dict(existing_immutable)
            current_without_design = dict(immutable_index)
            existing_design = str(
                existing_without_design.pop("design_fingerprint", "")
            )
            current_without_design.pop("design_fingerprint", None)
            if not (
                existing_design.startswith("sha256:")
                and existing_without_design == current_without_design
            ):
                raise RuntimeError("immutable rehearsal seat index changed")
    else:
        _write_json_exclusive(
            index_path,
            {
                **immutable_index,
                "issued_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    print(
        "[pure_rl] exact rehearsal-seat split committed "
        f"before_iter={before_iteration} "
        f"first={first_games} second={second_games} receipt={index_path}",
        flush=True,
    )
    return {
        "schema": immutable_index["schema"],
        "path": str(index_path),
        "sha256": _sha256_file(index_path),
    }


def _collection_receipt_path(run_dir: Path, iteration: int) -> Path:
    return (
        Path(run_dir)
        / "collection_receipts"
        / f"iter_{int(iteration):05d}.json"
    )


def _verified_partial_collection_resume(
    shard: Path,
    *,
    iteration: int,
) -> Optional[dict[str, Any]]:
    """Verify the explicit append-only authorization for one partial shard.

    Partial collection is never inferred from a parseable file.  The operator
    must seal the exact bytes, source-game identities, record/decision counts,
    behavior checkpoint, and missing identities in an immutable sidecar, then
    opt in with ``POKEBOT_PARTIAL_COLLECTION_RESUME_SIDECAR``.
    """
    raw_path = os.environ.get("POKEBOT_PARTIAL_COLLECTION_RESUME_SIDECAR", "").strip()
    if not raw_path:
        return None
    sidecar_path = Path(raw_path).expanduser().resolve()
    shard = Path(shard).resolve()
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        stat = shard.stat()
        retained = [int(value) for value in payload.get("retained_job_indices") or ()]
        missing = [int(value) for value in payload.get("missing_job_indices") or ()]
        if (
            payload.get("schema") != _PARTIAL_COLLECTION_RESUME_SCHEMA
            or payload.get("status") != "authorized_append_only_resume"
            or int(payload.get("iteration", -1)) != int(iteration)
            or Path(str(payload.get("shard_path") or "")).resolve() != shard
            or int(payload.get("shard_size", -1)) != int(stat.st_size)
            or int(payload.get("shard_mtime_ns", -1)) != int(stat.st_mtime_ns)
            or str(payload.get("shard_sha256") or "") != _sha256_file(shard)
            or int(payload.get("trajectory_records", -1)) <= 0
            or int(payload.get("decisions", -1)) <= 0
            or len(retained) != int(payload.get("retained_source_games", -1))
            or len(set(retained)) != len(retained)
            or len(set(missing)) != len(missing)
            or set(retained) & set(missing)
            or sorted(retained + missing) != list(range(int(payload.get("requested_games", -1))))
            or not str(payload.get("behavior_checkpoint_digest") or "").startswith("sha256:")
        ):
            return None
        return {**payload, "sidecar_path": str(sidecar_path)}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_collection_digest(path: Path) -> str:
    """Hash the shard at every receipt trust boundary."""
    return _sha256_file(Path(path).resolve())


def _scan_completed_compact_shard(
    path: Path,
    *,
    expected_checkpoint_digest: Optional[str] = None,
) -> dict[str, Any]:
    """Validate/count one shard with bounded memory while hashing it.

    Legacy recovery has lost the in-memory dispatch audit.  Reconstruct its
    strongest invariants directly from every durable row: episode identities
    are unique and every trajectory was generated by the exact learner at the
    ledger boundary.  A merely parseable mixture of stale weights is not a
    recoverable completed collection.
    """
    path = Path(path)
    digest = hashlib.sha256()
    games = 0
    decisions = 0
    episode_ids: set[str] = set()
    observed_checkpoint_digests: set[str] = set()
    matchup_runtime_rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise RuntimeError(
                    f"completed collection has a partial final row: {path}"
                )
            digest.update(raw)
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise RuntimeError(f"compact shard row is not an object: {path}")
            episode_id = str(obj.get("episode_id") or "")
            if not episode_id:
                raise RuntimeError(
                    f"completed collection row has no episode_id: {path}"
                )
            if episode_id in episode_ids:
                raise RuntimeError(
                    "completed collection has a duplicate episode_id: "
                    f"{episode_id!r}"
                )
            episode_ids.add(episode_id)
            provenance = obj.get("target_provenance")
            if not isinstance(provenance, dict):
                raise RuntimeError(
                    f"completed collection row has no target provenance: {path}"
                )
            behavior_digest = str(
                provenance.get("behavior_checkpoint_digest") or ""
            )
            if not behavior_digest:
                raise RuntimeError(
                    "completed collection row has no behavior checkpoint digest: "
                    f"{episode_id!r}"
                )
            observed_checkpoint_digests.add(behavior_digest)
            if (
                expected_checkpoint_digest is not None
                and behavior_digest != str(expected_checkpoint_digest)
            ):
                raise RuntimeError(
                    "completed collection contains a stale/mixed behavior digest: "
                    f"episode={episode_id!r} got={behavior_digest!r} "
                    f"expected={expected_checkpoint_digest!r}"
                )
            matchup_runtime_rows.append(
                {
                    "self_play": bool(provenance.get("self_play")),
                    "matchup_runtime_audit": provenance.get(
                        "matchup_runtime_audit"
                    ),
                }
            )
            rows = obj.get("decisions")
            if not isinstance(rows, list) or not rows:
                raise RuntimeError(
                    f"completed collection row has no decisions: {path}"
                )
            games += 1
            decisions += len(rows)
    if games <= 0 or decisions <= 0:
        raise RuntimeError(f"completed collection is empty: {path}")
    stat = path.stat()
    digest_value = f"sha256:{digest.hexdigest()}"
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest_value,
        "games": games,
        "decisions": decisions,
        "unique_episode_ids": len(episode_ids),
        "behavior_checkpoint_digests": sorted(observed_checkpoint_digests),
        "recovered_matchup_runtime": _summarize_matchup_runtime_rows(
            matchup_runtime_rows
        ),
        "recovered_matchup_runtime_self_play": (
            _summarize_matchup_runtime_rows(
                [row for row in matchup_runtime_rows if row["self_play"]]
            )
        ),
        "recovered_matchup_runtime_public_mix": (
            _summarize_matchup_runtime_rows(
                [row for row in matchup_runtime_rows if not row["self_play"]]
            )
        ),
    }


def _completed_collection_contract(
    contract: dict[str, Any],
) -> tuple[int, float, int]:
    games = dict(contract.get("games") or {})
    learner = dict(contract.get("learner") or {})
    expected = int(games.get("per_iteration") or 0)
    minimum_fraction = float(games.get("minimum_usable_fraction") or 0.0)
    max_context = int(learner.get("max_context") or 0)
    if expected <= 0 or not 0.0 < minimum_fraction <= 1.0 or max_context <= 0:
        raise RuntimeError("invalid completed-collection design contract")
    return expected, minimum_fraction, max_context


def _verified_completed_collection_receipt(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Verify a receipt against the immutable shard, cache, and ledger."""
    iteration = int(state.get("next_iteration", -1))
    receipt_path = _collection_receipt_path(run_dir, iteration)
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected, minimum_fraction, max_context = _completed_collection_contract(
            contract
        )
        expected_design_fingerprint = _design_fingerprint(contract)
        shard_row = dict(receipt.get("shard") or {})
        shard = (
            Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
        ).resolve()
        stat = shard.stat()
        learner = dict(state.get("learner") or state.get("champion") or {})
        stats = dict(receipt.get("stats") or {})
        runtime_required = os.environ.get(
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        runtime_enforcement = dict(
            stats.get("matchup_runtime_enforcement") or {}
        )
        seat_split_required = bool(
            ((contract.get("collection") or {}).get("training_seat_split") or {}).get(
                "required"
            )
        )
        seat_split = dict(stats.get("training_seat_split") or {})
        assigned_seats = dict(seat_split.get("assigned_source_games") or {})
        retained_seats = dict(seat_split.get("retained_source_games") or {})
        exact_seat_split_valid = bool(
            seat_split.get("schema") == _TRAINING_SEAT_SPLIT_SUMMARY_SCHEMA
            and seat_split.get("required") is True
            and seat_split.get("second_seat_priority") is False
            and seat_split.get("passed") is True
            and assigned_seats.get("exact_50_50") is True
            and retained_seats.get("exact_50_50") is True
            and int(assigned_seats.get("total", -1)) == expected
            and int(retained_seats.get("total", -1)) == expected
        )
        manifest = validated_replay_cache_manifest(
            shard,
            verify_info_set=False,
            max_context=max_context,
        )
        retained = int(stats.get("retained_source_games") or 0)
        if (
            receipt.get("schema") != _COMPLETED_COLLECTION_SCHEMA
            or str(receipt.get("design_fingerprint_at_collection") or "")
            != expected_design_fingerprint
            or int(receipt.get("iteration", -1)) != iteration
            or Path(str(shard_row.get("path") or "")).resolve() != shard
            or int(shard_row.get("size", -1)) != int(stat.st_size)
            or int(shard_row.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
            or not str(shard_row.get("sha256") or "").startswith("sha256:")
            or _completed_collection_digest(shard)
            != str(shard_row.get("sha256") or "")
            or int(receipt.get("requested_games", -1)) != expected
            or not _completed_collection_checkpoint_matches_state(
                run_dir=run_dir,
                state=state,
                collection_digest=str(receipt.get("checkpoint_digest") or ""),
            )
            or retained != expected
            or int(receipt.get("source_games", retained)) != retained
            or int(shard_row.get("games", -1)) <= 0
            or int(
                receipt.get(
                    "trajectory_records",
                    shard_row.get("games", -1),
                )
            )
            != int(shard_row.get("games", -2))
            or int(shard_row.get("decisions", -1)) <= 0
            or manifest is None
            or int(manifest.get("records", -1))
            != int(shard_row.get("games", -2))
            or int(manifest.get("sequences", -1))
            != int(shard_row.get("games", -2))
            or int(manifest.get("dropped", -1)) != 0
            or int(manifest.get("covered_bytes", -1)) != int(stat.st_size)
            or (
                runtime_required
                and not (
                    runtime_enforcement.get("schema")
                    == "poke_bot.matchup_runtime_collection_enforcement/v1"
                    and runtime_enforcement.get("required") is True
                    and runtime_enforcement.get("passed") is True
                )
            )
            or (seat_split_required and not exact_seat_split_valid)
        ):
            return None
        return {**receipt, "receipt_path": str(receipt_path)}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_collection_checkpoint_matches_state(
    *,
    run_dir: Path,
    state: dict[str, Any],
    collection_digest: str,
) -> bool:
    """Allow a zero-safe fusion child to train on its parent's finished shard."""
    learner = dict(state.get("learner") or state.get("champion") or {})
    learner_digest = str(learner.get("digest") or "")
    if collection_digest == learner_digest:
        return True
    activation = dict(state.get("decision_fusion_activation") or {})
    receipt_path = Path(str(activation.get("receipt") or "")).expanduser()
    if not (
        activation.get("schema")
        == "poke_bot.causal_decision_fusion_boundary_warmup/v1"
        and activation.get("phase") == "training_warmup"
        and activation.get("runtime_enabled") is False
        and activation.get("serving_eligible") is False
        and int(activation.get("boundary_next_iteration", -1))
        == int(state.get("next_iteration", -2))
        and str(activation.get("learner_digest") or "") == learner_digest
        and receipt_path.is_file()
        and _sha256_file(receipt_path)
        == str(activation.get("receipt_digest") or "")
    ):
        return False
    boundary = json.loads(receipt_path.read_text(encoding="utf-8"))
    material_row = dict(boundary.get("materialization_receipt") or {})
    material_path = Path(str(material_row.get("path") or "")).expanduser()
    if not (
        boundary.get("schema")
        == "poke_bot.causal_decision_fusion_boundary_warmup/v1"
        and Path(str(boundary.get("run_dir") or "")).resolve()
        == Path(run_dir).resolve()
        and int((boundary.get("boundary") or {}).get("next_iteration", -1))
        == int(state.get("next_iteration", -2))
        and str((boundary.get("parent_learner") or {}).get("digest") or "")
        == collection_digest
        and str((boundary.get("warmup_learner") or {}).get("digest") or "")
        == learner_digest
        and material_path.is_file()
        and _sha256_file(material_path) == str(material_row.get("digest") or "")
    ):
        return False
    material = json.loads(material_path.read_text(encoding="utf-8"))
    proof = dict(material.get("proof") or {})
    return bool(
        material.get("schema")
        == "poke_bot.causal_decision_fusion_checkpoint_migration/v1"
        and str(material.get("parent_checkpoint_digest") or "")
        == collection_digest
        and str(material.get("migrated_checkpoint_digest") or "")
        == learner_digest
        and proof.get("legacy_tensors_bit_identical") is True
        and proof.get("optimizer_existing_state_preserved") is True
        and proof.get("zero_safe_initialization") is True
        and (material.get("decision_fusion") or {}).get("runtime_enabled") is False
    )


def _commit_completed_collection_receipt(
    *,
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
    iteration: int,
    shard: Path,
    checkpoint: Path,
    checkpoint_digest: str,
    stats: dict[str, Any],
    started_at: float,
    writer: Optional[CompactShardWriter] = None,
    recovery_derived: bool = False,
) -> dict[str, Any]:
    """Commit the collection/rehearsal transaction boundary exactly once."""
    expected, minimum_fraction, max_context = _completed_collection_contract(
        contract
    )
    if int(iteration) != int(state.get("next_iteration", -1)):
        raise RuntimeError("completed collection does not match ledger boundary")
    learner = dict(state.get("learner") or state.get("champion") or {})
    if str(checkpoint_digest) != str(learner.get("digest") or ""):
        raise RuntimeError("completed collection checkpoint disagrees with ledger")
    manifest = validated_replay_cache_manifest(
        shard,
        verify_info_set=False,
        max_context=max_context,
    )
    if manifest is None:
        try:
            print(
                "[pure_rl] completed collection recovery building lossless "
                f"replay cache iter={iteration}",
                flush=True,
            )
            manifest = ensure_replay_cache_manifest(
                shard,
                verify_info_set=False,
                max_context=max_context,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "completed collection recovery cache build failed: "
                f"{type(exc).__name__}: {exc}"
            )
    if manifest is None or int(manifest.get("dropped", -1)) != 0:
        raise RuntimeError("completed collection lacks a lossless replay cache")
    if recovery_derived:
        shard_row = _scan_completed_compact_shard(
            shard,
            expected_checkpoint_digest=str(checkpoint_digest),
        )
        recovered_runtime = {
            "matchup_runtime": shard_row.pop("recovered_matchup_runtime"),
            "matchup_runtime_self_play": shard_row.pop(
                "recovered_matchup_runtime_self_play"
            ),
            "matchup_runtime_public_mix": shard_row.pop(
                "recovered_matchup_runtime_public_mix"
            ),
        }
    else:
        recovered_runtime = {}
        stat = Path(shard).stat()
        if writer is None:
            raise RuntimeError("live collection receipt requires writer counters")
        shard_row = {
            "path": str(Path(shard).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _sha256_file(shard),
            "games": int(writer.n_games),
            "decisions": int(writer.n_decisions),
        }
    if (
        int(manifest.get("records", -1)) != int(shard_row["games"])
        or int(manifest.get("sequences", -1)) != int(shard_row["games"])
        or int(manifest.get("covered_bytes", -1)) != int(shard_row["size"])
    ):
        raise RuntimeError("completed collection counters disagree with replay cache")
    normalized_stats = dict(stats)
    for key, value in recovered_runtime.items():
        normalized_stats.setdefault(key, value)
    if recovery_derived:
        runtime_required = os.environ.get(
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        recovered_audit = dict(normalized_stats.get("matchup_runtime") or {})
        enforcement = _matchup_runtime_collection_enforcement(
            recovered_audit,
            valid_games=int(recovered_audit.get("games") or 0),
            required=runtime_required,
            self_play_audit=dict(
                normalized_stats.get("matchup_runtime_self_play") or {}
            ),
            required_mirror_archetype=(
                os.environ.get("POKEBOT_ACTIVE_SPECIALIST", "")
                if runtime_required
                else None
            ),
        )
        normalized_stats["matchup_runtime_enforcement"] = enforcement
        if enforcement["passed"] is not True:
            raise RuntimeError(
                "recovered collection lacks the required activated matchup "
                f"runtime proof: {enforcement['assertions']}"
            )
    retained = int(normalized_stats.get("retained_source_games") or 0)
    if retained != expected:
        raise RuntimeError(
            "completed collection does not contain the exact configured "
            f"source-game count: retained={retained} expected={expected}"
        )
    completed_at = float(
        normalized_stats.get("collect_completed_at")
        or (float(shard_row["mtime_ns"]) / 1_000_000_000.0)
    )
    elapsed = max(
        float(normalized_stats.get("collect_elapsed_sec") or 0.0),
        completed_at - float(started_at),
        1e-6,
    )
    normalized_stats["collect_elapsed_sec"] = elapsed
    payload = {
        "schema": _COMPLETED_COLLECTION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(iteration),
        "requested_games": expected,
        # A source game may intentionally emit two independently causal
        # acting-seat trajectories during same-checkpoint self-play. Keep
        # protocol game counts separate from replay-record counts everywhere.
        "source_games": retained,
        "trajectory_records": int(shard_row["games"]),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_digest": str(checkpoint_digest),
        "design_fingerprint_at_collection": str(
            state.get("design_fingerprint") or ""
        ),
        "started_at": float(started_at),
        "completed_at": completed_at,
        "recovery_derived": bool(recovery_derived),
        "shard": shard_row,
        "replay_cache": {
            "manifest_path": manifest.get("manifest_path"),
            "records": int(manifest["records"]),
            "sequences": int(manifest["sequences"]),
            "dropped": int(manifest["dropped"]),
            "covered_bytes": int(manifest["covered_bytes"]),
            "signature": manifest.get("signature"),
        },
        "stats": normalized_stats,
    }
    receipt_path = _collection_receipt_path(run_dir, iteration)
    _write_json_exclusive(receipt_path, payload)
    print(
        f"[pure_rl] completed collection committed iter={iteration} "
        f"source_games={retained} "
        f"trajectories={shard_row['games']} "
        f"decisions={shard_row['decisions']} "
        f"receipt={receipt_path}",
        flush=True,
    )
    return {**payload, "receipt_path": str(receipt_path)}


def _ensure_recoverable_completed_collection(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Recover the narrow crash window after collect and before train commit."""
    existing = _verified_completed_collection_receipt(run_dir, state, contract)
    if existing is not None:
        return existing
    iteration = int(state.get("next_iteration", -1))
    # A present-but-invalid immutable receipt must be quarantined together with
    # its shard.  Never overwrite it or infer that it described this retry.
    if _collection_receipt_path(run_dir, iteration).exists():
        return None
    shard = Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
    artifacts = _iteration_artifact_paths(run_dir, iteration)
    if not shard.is_file() or {path.resolve() for path in artifacts} != {
        shard.resolve()
    }:
        return None
    if _verified_partial_collection_resume(shard, iteration=iteration) is not None:
        print(
            "[pure_rl] partial collection is explicitly sealed; defer completed-"
            "collection recovery until append-only refill finishes",
            flush=True,
        )
        return None
    expected, minimum_fraction, max_context = _completed_collection_contract(
        contract
    )
    runtime_path = Path(run_dir) / "iteration_runtime.json"
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    learner = dict(state.get("learner") or state.get("champion") or {})
    manifest = validated_replay_cache_manifest(
        shard,
        verify_info_set=False,
        max_context=max_context,
    )
    if manifest is None:
        try:
            print(
                "[pure_rl] completed collection recovery building lossless "
                f"replay cache iter={iteration}",
                flush=True,
            )
            manifest = ensure_replay_cache_manifest(
                shard,
                verify_info_set=False,
                max_context=max_context,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                "[pure_rl] completed collection recovery cache build failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
    # With the original in-memory stats gone, require the strongest exact
    # case. A merely 98%-usable partial wave is retried instead of guessed.
    if (
        str(runtime.get("phase") or "") != "collect"
        or int(runtime.get("iteration", -1)) != iteration
        or str(runtime.get("checkpoint_digest") or "")
        != str(learner.get("digest") or "")
        or manifest is None
        or int(manifest.get("records", -1)) != expected
        or int(manifest.get("sequences", -1)) != expected
        or int(manifest.get("dropped", -1)) != 0
    ):
        return None
    started_at = float(runtime.get("started_at") or shard.stat().st_mtime)
    elapsed = max(shard.stat().st_mtime - started_at, 1e-6)
    stats = {
        "ok": expected,
        "with_record": expected,
        "requested_games": expected,
        "retained_source_games": expected,
        "retained_trajectories": expected,
        "usable_game_fraction": 1.0,
        "collect_elapsed_sec": elapsed,
        "claimed_games_per_sec": expected / elapsed,
        "valid_source_games_per_sec": expected / elapsed,
        "trajectory_games_per_sec": expected / elapsed,
        "recovered_completed_collection": True,
        "minimum_usable_fraction": minimum_fraction,
    }
    try:
        return _commit_completed_collection_receipt(
            run_dir=run_dir,
            state=state,
            contract=contract,
            iteration=iteration,
            shard=shard,
            checkpoint=Path(str(learner.get("path") or "")),
            checkpoint_digest=str(learner.get("digest") or ""),
            stats=stats,
            started_at=started_at,
            recovery_derived=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            "[pure_rl] completed collection recovery rejected; "
            f"will quarantine/recollect iter={iteration}: {exc}",
            flush=True,
        )
        return None


def _completed_collection_bundle(
    run_dir: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
) -> Optional[dict[str, Any]]:
    receipt = _verified_completed_collection_receipt(run_dir, state, contract)
    if receipt is None:
        return None
    shard_row = dict(receipt["shard"])
    elapsed = float((receipt.get("stats") or {}).get("collect_elapsed_sec") or 0.0)
    writer = CompactShardWriter.from_completed_shard(
        Path(str(shard_row["path"])),
        n_games=int(shard_row["games"]),
        n_decisions=int(shard_row["decisions"]),
        elapsed_sec=elapsed,
    )
    print(
        f"[pure_rl] resume completed collection iter={receipt['iteration']} "
        f"games={writer.n_games} decisions={writer.n_decisions} "
        "(skip recollection)",
        flush=True,
    )
    return {
        "iteration": int(receipt["iteration"]),
        "shard": Path(str(shard_row["path"])),
        "writer": writer,
        "stats": dict(receipt["stats"]),
        "checkpoint": Path(str(receipt["checkpoint"])),
        "digest": str(receipt["checkpoint_digest"]),
        "started_at": float(receipt["started_at"]),
        # A receipt-backed candidate was trained against the immutable design
        # that governed collection.  A source-only boundary migration may be
        # committed while recovering that transaction, so retain the receipt
        # fingerprint instead of accidentally validating against the newly
        # migrated runtime fingerprint later in the loop.
        "design_fingerprint_at_collection": str(
            receipt.get("design_fingerprint_at_collection") or ""
        ),
        "recovered": True,
        "receipt": str(receipt["receipt_path"]),
    }


def _verified_completed_collection_across_design_chain(
    run_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Find a receipt under the verified design that governed its collection.

    Boundary migrations govern future work.  They must not invalidate an
    append-only N+1 receipt (or its trained candidate) created under an earlier
    verified link in the same migration chain.
    """
    current, _digest, receipts = _load_design_migration_chain(run_dir, manifest)
    candidates = [current]
    candidates.extend(
        dict(receipt["previous_contract"]) for receipt in reversed(receipts)
    )
    seen: set[str] = set()
    for contract in candidates:
        fingerprint = _design_fingerprint(contract)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        completed = _verified_completed_collection_receipt(run_dir, state, contract)
        if completed is not None:
            return completed, contract
    return None, None


def _collection_behavior_identity(bundle: dict[str, Any]):
    """Return the digest-verified policy that produced this collection.

    The rollout champion and the continuous learner can intentionally differ.
    Training provenance must therefore come from the immutable collection
    receipt/bundle, never from the H2H incumbent selected later in the loop.
    """
    return _verified_checkpoint_identity(
        {
            "path": str(bundle.get("checkpoint") or ""),
            "digest": str(bundle.get("digest") or ""),
        }
    )


def _safe_research_control_recovery_artifact(
    path: Path,
    *,
    iteration: int,
    candidate_digest: str,
) -> bool:
    """Recognize a nontraining research result that is safe to preserve.

    This is deliberately only a crash-recovery/artifact-retention check. The
    exact regenerated measurement plan, registry, opponent package digests,
    seats, seeds, and weights are validated by
    ``_research_control_measurement`` before the result can be reused.
    """
    try:
        resolved = Path(path).resolve()
        result = json.loads(resolved.read_text(encoding="utf-8"))
        audit = dict(result.get("audit") or {})
        measurement_plan = dict(audit.get("measurement_plan") or {})
        return bool(
            result.get("schema") == RESEARCH_CONTROL_RESULT_SCHEMA
            and int(result.get("iteration", -1)) == int(iteration)
            and int(measurement_plan.get("iteration", -1)) == int(iteration)
            and str(result.get("checkpoint_digest") or "")
            == str(candidate_digest)
            and result.get("training_eligible") is False
            and result.get("replay_eligible") is False
            and result.get("diagnostic_only") is True
            and result.get("included_in_gate_pass") is False
            and float(result.get("gate_weight", -1.0)) == 0.0
            and result.get("formal_eval") is False
            and str(result.get("action_selection") or "") == "greedy"
            and int(result.get("games", 0)) > 0
            and audit.get("passed") is True
            and audit.get("exact_distribution") is True
            and audit.get("exact_weights") is True
            and audit.get("seed_disjoint") is True
            and audit.get("package_disjoint_from_active_gate") is True
            and int(audit.get("replay_records_written", -1)) == 0
            and Path(str(result.get("result_path") or "")).resolve()
            == resolved
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _recover_interrupted_iteration(
    run_dir: Path,
    state: dict[str, Any],
    *,
    preserve_completed_collection: bool = True,
    research_control_registry: Optional[dict[str, Any]] = None,
    expected_awr_provider: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    """Transactionally quarantine an uncommitted iteration so it can retry."""
    iteration = int(state.get("next_iteration", 0))
    commit = run_dir / "commits" / f"iter_{iteration:05d}.json"
    if commit.exists():
        raise RuntimeError(
            f"recovery called before committed iteration {iteration} was replayed"
        )
    artifacts = _iteration_artifact_paths(run_dir, iteration)
    expected_partial_shard = (
        Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
    ).resolve()
    partial_resume = (
        _verified_partial_collection_resume(
            expected_partial_shard, iteration=iteration
        )
        if expected_partial_shard.is_file()
        else None
    )
    if (
        partial_resume is not None
        and {path.resolve() for path in artifacts} == {expected_partial_shard}
    ):
        print(
            "[pure_rl] preserve explicitly sealed partial collection for "
            f"append-only resume iter={iteration} "
            f"retained={partial_resume['retained_source_games']} "
            f"missing={len(partial_resume['missing_job_indices'])}",
            flush=True,
        )
        return None
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        immutable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed, _receipt_contract = (
            _verified_completed_collection_across_design_chain(
                run_dir, state, immutable_manifest
            )
        )
        expected_shard = (
            Path(run_dir) / "shards" / f"iter_{iteration:05d}.jsonl"
        ).resolve()
        if preserve_completed_collection and completed is not None:
            artifact_set = {path.resolve() for path in artifacts}
            expected_candidate = (
                Path(run_dir) / "checkpoints" / f"iter_{iteration:05d}.pt"
            ).resolve()
            expected_research = _research_control_result_path(
                run_dir, iteration
            ).resolve()
            if artifact_set == {expected_shard}:
                print(
                    f"[pure_rl] preserve receipt-verified completed collection "
                    f"iter={iteration}; resume at rehearsal/train",
                    flush=True,
                )
                return None
            candidate_transaction = {
                expected_shard,
                expected_candidate,
            }
            if artifact_set in (
                candidate_transaction,
                candidate_transaction | {expected_research},
            ):
                candidate_result = _verified_orphan_candidate_result(
                    expected_candidate,
                    iteration=iteration,
                    parent_digest=_orphan_recovery_parent_digest(
                        Path(run_dir), state, iteration
                    ),
                    behavior_digest=str(completed.get("checkpoint_digest") or ""),
                    design_fingerprint=_orphan_recovery_design_fingerprints(
                        state,
                        completed.get("design_fingerprint_at_collection"),
                    ),
                    shard_path=expected_shard,
                    r241_peak_r195_training_contract=dict(
                        (immutable_manifest.get("design_contract") or {}).get(
                            "r241_peak_r195_training_contract"
                        )
                        or {}
                    ),
                    expected_awr_provider=(
                        expected_awr_provider
                        if expected_awr_provider is not None
                        else (
                            (
                                (
                                    immutable_manifest.get("design_contract")
                                    or {}
                                ).get("learner")
                                or {}
                            ).get("prize_plan_h3_actor_provider")
                        )
                    ),
                )
                research_is_safe = bool(
                    expected_research not in artifact_set
                    or _safe_research_control_recovery_artifact(
                        expected_research,
                        iteration=iteration,
                        candidate_digest=str(
                            candidate_result["candidate_digest"]
                        ),
                    )
                )
                if research_is_safe:
                    print(
                        f"[pure_rl] preserve receipt-verified trained candidate "
                        f"iter={iteration}; resume at promotion/heldout"
                        + (
                            " with completed research controls"
                            if expected_research in artifact_set
                            else ""
                        ),
                        flush=True,
                    )
                    return None
                print(
                    "[pure_rl] research-control recovery artifact is not a "
                    "safe nontraining result; quarantine the interrupted "
                    f"transaction iter={iteration}",
                    flush=True,
                )
    receipt_path = _collection_receipt_path(run_dir, iteration)
    if receipt_path.exists() and receipt_path not in artifacts:
        artifacts.append(receipt_path)
    qroot = run_dir / "quarantine" / f"iter_{iteration:05d}"

    # Resume a quarantine transaction if the process died while moving files.
    in_progress: Optional[Path] = None
    if qroot.is_dir():
        for attempt in sorted(qroot.glob("attempt_*")):
            if (attempt / "plan.json").is_file() and not (
                attempt / "failure.json"
            ).is_file():
                in_progress = attempt
                break
    if in_progress is not None:
        plan = json.loads((in_progress / "plan.json").read_text(encoding="utf-8"))
    elif not artifacts:
        return None
    else:
        qroot.mkdir(parents=True, exist_ok=True)
        attempt_i = 1
        while (qroot / f"attempt_{attempt_i:04d}").exists():
            attempt_i += 1
        in_progress = qroot / f"attempt_{attempt_i:04d}"
        in_progress.mkdir()
        plan_rows = []
        for source in artifacts:
            rel = source.relative_to(run_dir)
            plan_rows.append(
                {
                    "source": str(source),
                    "relative_path": rel.as_posix(),
                    "destination": str(in_progress / rel),
                    "size": int(source.stat().st_size),
                    "digest": _sha256_file(source),
                }
            )
        plan = {
            "schema": 1,
            "iteration": iteration,
            "reason": "interrupted_before_append_only_commit",
            "ledger_next_iteration": iteration,
            "artifacts": plan_rows,
        }
        _write_json_exclusive(in_progress / "plan.json", plan)

    for row in plan.get("artifacts") or []:
        source = Path(row["source"])
        destination = Path(row["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _sha256_file(destination) != str(row["digest"]):
                raise RuntimeError(
                    f"quarantine destination digest mismatch: {destination}"
                )
            if source.exists():
                raise RuntimeError(
                    f"quarantine has both source and destination: {source}"
                )
            continue
        if not source.exists():
            raise RuntimeError(
                f"quarantine lost both source and destination for {row['source']}"
            )
        if _sha256_file(source) != str(row["digest"]):
            raise RuntimeError(f"quarantine source changed during recovery: {source}")
        source.replace(destination)

    failure_path = in_progress / "failure.json"
    if not failure_path.exists():
        _write_json_exclusive(
            failure_path,
            {
                **plan,
                "quarantine_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    # Restore latest.json to the last committed metric when the interrupted
    # pointer was quarantined.  The pointer is mutable; iteration metrics are not.
    latest = run_dir / "metrics" / "latest.json"
    prior = run_dir / "metrics" / f"iter_{iteration - 1:05d}.json"
    if not latest.exists() and prior.is_file():
        _atomic_json(latest, json.loads(prior.read_text(encoding="utf-8")))
    return failure_path


def _load_loop_state(run_dir: Path) -> Optional[dict[str, Any]]:
    path = Path(run_dir) / "loop_state.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state.get("version", -1)) != LOOP_STATE_VERSION:
        raise RuntimeError(
            f"unsupported pure-RL loop state version at {path}: "
            f"{state.get('version')!r}"
        )
    # A commit record is the transaction boundary. If the process died after
    # writing it but before advancing the mutable pointer, replay it exactly.
    while True:
        next_it = int(state.get("next_iteration", 0))
        commit_path = Path(run_dir) / "commits" / f"iter_{next_it:05d}.json"
        if not commit_path.is_file():
            break
        committed = json.loads(commit_path.read_text(encoding="utf-8"))
        if (
            int(committed.get("version", -1)) != LOOP_STATE_VERSION
            or int(committed.get("last_completed_iteration", -1)) != next_it
            or int(committed.get("next_iteration", -1)) != next_it + 1
            or committed.get("run_name") != state.get("run_name")
            or committed.get("mode") != state.get("mode")
        ):
            raise RuntimeError(f"invalid append-only commit record: {commit_path}")
        state = committed
        _atomic_json(Path(run_dir) / "loop_state.json", state)
    return state


def _validate_resume_state(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    state: dict[str, Any],
) -> tuple[Path, int]:
    """Resolve the immutable incumbent and monotonic next iteration."""
    from poke_bot.promotion import CheckpointIdentity

    if args.resume == "never":
        raise RuntimeError(
            f"run {args.run_name!r} already has loop_state.json; "
            "use a new run name or --resume auto"
        )
    if str(state.get("run_name")) != str(args.run_name):
        raise RuntimeError("loop state run_name does not match requested run")
    if str(state.get("mode")) != str(args.mode):
        raise RuntimeError("loop state mode does not match requested mode")
    champion = dict(state.get("champion") or {})
    ckpt = Path(str(champion.get("path") or "")).expanduser().resolve()
    identity = CheckpointIdentity.from_path(ckpt)
    expected = str(champion.get("digest") or "")
    if not expected or identity.digest != expected:
        raise RuntimeError(
            "resume champion digest mismatch: "
            f"ledger={expected!r} disk={identity.digest!r} path={ckpt}"
        )
    next_it = int(state.get("next_iteration", 0))
    if next_it < 0:
        raise RuntimeError(f"invalid next_iteration={next_it} in loop ledger")
    if args.start_iteration is not None and int(args.start_iteration) != next_it:
        raise RuntimeError(
            f"--start-iteration={args.start_iteration} would rewind/skip ledger "
            f"next_iteration={next_it}"
        )
    if args.base_checkpoint is not None:
        requested = Path(args.base_checkpoint).expanduser().resolve()
        successor_resume = bool(
            getattr(args, "r241_own_deck_migration_receipt", None) is not None
            and getattr(args, "initial_learner_checkpoint", None) is not None
            and Path(args.initial_learner_checkpoint).expanduser().resolve()
            == ckpt
        )
        if requested != ckpt and not successor_resume:
            raise RuntimeError(
                "--base-checkpoint conflicts with resumed champion: "
                f"{requested} != {ckpt}"
            )
    return ckpt, next_it


def _verified_checkpoint_identity(entry: Any):
    """Return a path+digest identity and verify both against immutable bytes."""
    from poke_bot.promotion import CheckpointIdentity

    if isinstance(entry, CheckpointIdentity):
        expected_path, expected_digest = entry.path, entry.digest
    elif isinstance(entry, dict):
        expected_path = str(entry.get("path") or "")
        expected_digest = str(entry.get("digest") or "")
    else:
        raise RuntimeError(
            "opponent pool entries must be path+digest identities, got "
            f"{type(entry).__name__}"
        )
    if not expected_path or not expected_digest:
        raise RuntimeError(f"incomplete checkpoint identity: {entry!r}")
    actual = CheckpointIdentity.from_path(expected_path)
    if actual.digest != expected_digest:
        raise RuntimeError(
            "checkpoint identity digest mismatch: "
            f"path={actual.path} ledger={expected_digest} disk={actual.digest}"
        )
    return actual


def _exact_regression_rollback_identity(
    state: dict[str, Any],
    *,
    exact_anchor: Any,
    behavior_before: Any,
) -> tuple[Any, str]:
    """Return a rollback checkpoint that preserves an active runtime contract.

    The protected heldout checkpoint can predate an additive matchup-runtime
    migration. In that case, publishing it directly would fail every resident
    worker's runtime reload preflight. The boundary receipt records the exact
    behavior-equivalent child that added the required adapters, so use that
    immutable child when the heldout anchor is its parent. If the receipt is
    unavailable or malformed, preserve the already-published behavior identity
    instead of entering a deterministic crash/restart loop.
    """

    activation = dict(state.get("matchup_runtime_activation") or {})
    if not activation:
        return exact_anchor, "heldout_champion"

    receipt_path = Path(str(activation.get("receipt") or "")).expanduser()
    expected_activated_digest = str(activation.get("learner_digest") or "")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema")
            != "poke_bot.matchup_runtime_boundary_activation/v1"
        ):
            raise ValueError("runtime boundary receipt schema mismatch")
        parent = dict(receipt.get("parent_learner") or {})
        activated = dict(receipt.get("activated_learner") or {})
        parent_digest = str(parent.get("digest") or "")
        # A newer heldout champion was produced after runtime activation and
        # therefore already contains the runtime-compatible adapter state.
        if parent_digest and parent_digest != str(exact_anchor.digest):
            return exact_anchor, "post_activation_heldout_champion"
        identity = _verified_checkpoint_identity(activated)
        if (
            not expected_activated_digest
            or identity.digest != expected_activated_digest
        ):
            raise ValueError("runtime activated learner digest mismatch")
        return identity, "runtime_activated_exact_anchor"
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            "[pure_rl] EXACT_GATE_ROLLBACK_RUNTIME_FALLBACK "
            f"reason={type(exc).__name__}:{exc}; preserve_behavior="
            f"{behavior_before.digest[:19]}…",
            flush=True,
        )
        return behavior_before, "runtime_receipt_invalid_preserve_behavior"


def _verify_learner_lineage(
    result: dict[str, Any], *, candidate: Any, parent: Any
) -> None:
    reported_candidate_digest = str(result.get("candidate_digest") or "")
    if reported_candidate_digest != str(candidate.digest):
        raise RuntimeError(
            "learner returned a candidate digest that does not match disk: "
            f"reported={reported_candidate_digest!r} "
            f"disk={candidate.digest!r} path={candidate.path}"
        )
    reported_parent_digest = str(result.get("parent_digest") or "")
    if reported_parent_digest != str(parent.digest):
        raise RuntimeError(
            "learner returned the wrong parent lineage: "
            f"reported={reported_parent_digest!r} expected={parent.digest!r}"
        )


def _validate_orphan_awr_provider(
    *,
    extra: Mapping[str, Any],
    provenance: Mapping[str, Any],
    expected: Optional[Mapping[str, Any]],
) -> None:
    """Reject recovery across the legacy/H3 optimizer-unit boundary."""

    actual = extra.get("awr_advantage_provider")
    recorded = provenance.get("prize_plan_h3_actor_provider")
    if expected is None:
        if actual not in (None, {}) or recorded not in (None, {}):
            raise RuntimeError(
                "orphan candidate has an unrequested Prize-plan H3 provider"
            )
        return
    if not isinstance(expected, Mapping):
        raise RuntimeError("expected Prize-plan H3 provider design is malformed")
    if expected.get("mode") == "receipt_bound_live_h3_additive_at_clean_optimizer_boundary":
        if (
            expected.get("schema")
            != "poke_bot.alakazam_prize_plan_v2_h3_actor_design/v1"
            or not isinstance(actual, Mapping)
            or not isinstance(recorded, Mapping)
            or actual.get("actor_activation") is not True
            or actual.get("exact_legacy_baseline_computed_in_batch") is not True
            or actual.get("runtime_critic_calls") is not False
            or dict(recorded) != dict(actual.get("provider_binding") or {})
            or dict(recorded.get("live_source_design") or {}) != dict(expected)
        ):
            raise RuntimeError(
                "orphan Prize-plan H3 live provider disagrees with the "
                "receipt-bound optimizer unit"
            )
        return
    cache = expected.get("cache_receipt")
    activation = expected.get("activation_receipt")
    if (
        expected.get("schema")
        != "poke_bot.alakazam_prize_plan_v2_h3_actor_design/v1"
        or expected.get("mode")
        != "receipt_bound_h3_additive_at_clean_optimizer_boundary"
        or not isinstance(cache, Mapping)
        or not isinstance(activation, Mapping)
        or not isinstance(actual, Mapping)
        or not isinstance(recorded, Mapping)
    ):
        raise RuntimeError("orphan Prize-plan H3 provider identity is incomplete")
    binding = actual.get("provider_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema")
        != "poke_bot.alakazam_prize_plan_v2_h3_actor_provider_binding/v1"
        or binding.get("cache_receipt_sha256") != cache.get("digest")
        or binding.get("activation_receipt_sha256") != activation.get("digest")
        or dict(recorded) != dict(binding)
        or actual.get("actor_activation") is not True
        or actual.get("exact_legacy_baseline_computed_in_batch") is not True
        or actual.get("runtime_critic_calls") is not False
    ):
        raise RuntimeError(
            "orphan candidate Prize-plan H3 provider disagrees with the "
            "receipt-bound optimizer unit"
        )


def _verified_orphan_candidate_result(
    path: Path,
    *,
    iteration: int,
    parent_digest: str,
    behavior_digest: str,
    design_fingerprint: str | Sequence[str],
    shard_path: Path,
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]] = None,
    expected_awr_provider: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Reconstruct ``rl_train_step`` output for one exact crash window."""
    from poke_bot import checkpoint as checkpoint_mod

    candidate_path = Path(path).resolve()
    payload = checkpoint_mod.load_checkpoint(candidate_path, map_location="cpu")
    checkpoint_mod.assert_trusted_policy_checkpoint(candidate_path)
    digest = checkpoint_mod.checkpoint_digest(candidate_path)
    extra = dict(payload.get("extra") or {})
    provenance = dict(extra.get("training_provenance") or {})
    behavior = dict(provenance.get("behavior_checkpoint") or {})
    learner_parent = dict(provenance.get("learner_parent") or {})
    provenance_shard = Path(str(provenance.get("shard") or "")).resolve()
    metrics = dict(extra.get("rl_metrics") or {})
    validation_metrics = dict(extra.get("validation_metrics") or {})
    fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    route_rows = dict(fit.get("route_decisions") or {})
    train_games = int(metrics.get("n_games") or 0)
    validation_games = int(validation_metrics.get("n_games") or 0)
    allowed_design_fingerprints = (
        {str(value) for value in design_fingerprint}
        if not isinstance(design_fingerprint, str)
        else {design_fingerprint}
    )
    if not allowed_design_fingerprints or any(
        not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        for value in allowed_design_fingerprints
    ):
        raise RuntimeError("orphan recovery received an invalid design fingerprint set")
    _validate_orphan_awr_provider(
        extra=extra,
        provenance=provenance,
        expected=expected_awr_provider,
    )
    if not (
        extra.get("pure_rl") is True
        and str(extra.get("parent_digest") or "") == str(parent_digest)
        and provenance.get("pure_rl") is True
        and int(provenance.get("iteration", -1)) == int(iteration)
        and provenance.get("append_only") is True
        and str(provenance.get("design_fingerprint") or "")
        in allowed_design_fingerprints
        and str(behavior.get("digest") or "") == str(behavior_digest)
        and str(learner_parent.get("digest") or "") == str(parent_digest)
        and provenance_shard == Path(shard_path).resolve()
        and extra.get("optimizer_state_restored") is True
        and isinstance(payload.get("optimizer_state_dict"), dict)
        and bool(payload.get("optimizer_state_dict"))
        and train_games > 0
        and validation_games > 0
        and extra.get("matchup_adapters_runtime_enabled") is False
        and (
            (
                extra.get("matchup_adapter_training_enabled") is False
                and extra.get("matchup_adapter_optimizer_included") is False
            )
            or (
                extra.get("matchup_adapter_training_enabled") is True
                and extra.get("matchup_adapter_optimizer_included") is True
            )
        )
    ):
        raise RuntimeError("orphan candidate does not match the interrupted RL transaction")
    if fit and not (
        fit.get("schema") == "poke_bot.dormant_matchup_adapter_fit/v1"
        and fit.get("runtime_enabled") is False
        and fit.get("base_frozen") is True
        and fit.get("optimizer_scope") == "matchup_adapter_bank_only"
        and int(fit.get("steps") or 0) > 0
        and int(fit.get("rows") or 0) > 0
        and sum(int(value) for value in route_rows.values()) > 0
        and bool(extra.get("dormant_matchup_adapter_optimizer_state"))
    ):
        raise RuntimeError("orphan candidate has an incomplete dormant-adapter fit")
    if r241_peak_r195_training_contract:
        validate_r241_isolated_adapter_continuation(
            fit,
            dict(extra.get("dormant_matchup_adapter_optimizer_state") or {}),
        )
        validate_r241_peak_r195_live_fusion_record(
            dict(extra.get("peak_r195_live_fusion") or {}),
            contract=r241_peak_r195_training_contract,
            phase="rl",
            boundary_iteration=int(iteration),
        )
    elif extra.get("peak_r195_live_fusion"):
        raise RuntimeError("orphan candidate has an unrequested r241 sidecar")
    return {
        "latest_path": str(candidate_path),
        "candidate_path": str(candidate_path),
        "candidate_digest": digest,
        "parent_digest": str(parent_digest),
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "validation_source": extra.get("validation_source"),
        "step": int(payload.get("step") or extra.get("global_step") or 0),
        "parent_step": int(extra.get("optimizer_parent_step") or 0),
        "optimizer_state_restored": True,
        "awr_baseline_mode": str(extra.get("awr_baseline_mode") or "unknown"),
        "epochs_ran": int(extra.get("rl_epochs_ran") or 0),
        "policy_prev_agreement": float(extra.get("policy_prev_agreement") or 0.0),
        "policy_prev_agreement_rows": int(
            extra.get("policy_prev_agreement_rows") or 0
        ),
        "dormant_matchup_adapter_fit": fit,
        "n_train_sequences": train_games + validation_games,
        "recovered_immutable_candidate": True,
    }


def _orphan_recovery_design_fingerprints(
    state: Mapping[str, Any], *extra: Any
) -> tuple[str, ...]:
    """Return only ledger-recorded source designs eligible for recovery."""

    values = [*extra, state.get("design_fingerprint")]
    values.extend(
        row.get("fingerprint")
        for row in state.get("design_migration_history") or ()
        if isinstance(row, Mapping)
    )
    return tuple(
        dict.fromkeys(
            str(value)
            for value in values
            if re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or ""))
        )
    )


def _orphan_recovery_parent_digest(
    run_dir: Path,
    state: Mapping[str, Any],
    iteration: int,
) -> str:
    """Resolve the exact post-rehearsal parent for an interrupted candidate."""

    learner = dict(state.get("learner") or state.get("champion") or {})
    parent_digest = str(learner.get("digest") or "")
    receipt_path = Path(run_dir) / "rehearsals" / f"before_iter_{iteration:05d}.json"
    if not receipt_path.is_file():
        return parent_digest
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") == "poke_bot.r260_scheduled_expert_rehearsal/v1":
        validated = _validate_r260_scheduled_rehearsal_receipt(receipt)
        expected_bounds = [int(iteration) - 1, int(iteration) - 1]
        if (
            validated.get("rl_iteration_before_after") != expected_bounds
            or str(validated["pre_checkpoint"]["sha256"]) != parent_digest
        ):
            raise RuntimeError(
                "orphan recovery r260 rehearsal receipt has invalid lineage"
            )
        return str(validated["post_checkpoint"]["sha256"])
    if (
        int(receipt.get("before_iteration", -1)) != int(iteration)
        or str(receipt.get("parent_digest") or "") != parent_digest
    ):
        raise RuntimeError("orphan recovery rehearsal receipt has invalid lineage")
    identity = _verified_checkpoint_identity(
        {
            "path": str(receipt.get("checkpoint") or ""),
            "digest": str(receipt.get("checkpoint_digest") or ""),
        }
    )
    return identity.digest


_REQUIRED_ACTIVE_GATE_CHECKS = frozenset(
    {
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
    }
)
_REQUIRED_ACTIVE_GATE_CHECKS_WITH_S_PLUS = (
    _REQUIRED_ACTIVE_GATE_CHECKS | {"s_plus_matchup_floor_allowance"}
)


def _formal_active_gate_pass(
    row: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Return the exact formal specialist pass committed in ``row``.

    The short incumbent head-to-head promotion is a learner-selection safety
    diagnostic.  It is not one of the canonical official/premium specialist
    gates and therefore cannot veto a fully audited formal gate pass.
    """

    if row.get("completed") is not True:
        return None
    result = dict(row.get("active_gate_result") or {})
    checks = dict(result.get("checks") or {})
    audit = dict(result.get("audit") or {})
    candidate = dict(row.get("candidate") or {})
    if (
        result.get("passed") is not True
        or frozenset(checks)
        not in {
            _REQUIRED_ACTIVE_GATE_CHECKS,
            _REQUIRED_ACTIVE_GATE_CHECKS_WITH_S_PLUS,
        }
        or not all(checks.get(name) is True for name in checks)
        or audit.get("passed") is not True
        or audit.get("exact_distribution") is not True
        or audit.get("exact_weights") is not True
        or audit.get("both_seats") is not True
        or str(candidate.get("path") or "") != str(result.get("checkpoint") or "")
        or str(candidate.get("digest") or "")
        != str(result.get("checkpoint_digest") or "")
    ):
        return None
    return result


def _terminal_gate_payload(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    history = list(state.get("history") or [])
    if not history:
        return None
    row = dict(history[-1])
    result = _formal_active_gate_pass(row)
    if result is None:
        return None
    candidate = _verified_checkpoint_identity(row.get("candidate") or {})
    return {
        "iteration": int(row["iteration"]),
        "wr": float(result["skill_weighted_wr"]),
        "confidence_lower": float(result["confidence_lower"]),
        "games": int(result["games"]),
        "checkpoint": candidate.path,
        "checkpoint_digest": candidate.digest,
    }


def _reconciled_heldout_champion_evidence(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade legacy pooled-only evidence from its committed exact gate.

    v6 committed the full per-opponent gate in history but omitted it from the
    heldout-champion summary.  Reconstruct only when iteration, digest, audit,
    game count, and pooled WR all agree; otherwise leave the evidence untouched.
    """
    evidence = dict(state.get("heldout_champion_evidence") or {})
    if not evidence or isinstance(evidence.get("per_opponent"), dict):
        return evidence
    digest = str(evidence.get("checkpoint_digest") or "")
    try:
        evidence_iteration = int(evidence.get("iteration"))
        evidence_games = int(evidence.get("games"))
        evidence_wr = float(evidence.get("win_rate"))
    except (TypeError, ValueError):
        return evidence
    for raw_row in reversed(list(state.get("history") or [])):
        row = dict(raw_row or {})
        try:
            row_iteration = int(row.get("iteration", -1))
        except (TypeError, ValueError):
            continue
        if row_iteration != evidence_iteration:
            continue
        champion = dict(row.get("heldout_champion") or {})
        audit = dict(row.get("heldout_audit") or {})
        gate = dict(row.get("stage_gate") or {})
        per_opponent = gate.get("per_opponent")
        try:
            gate_games = int(gate.get("games", -1))
            gate_wr = float(gate.get("win_rate", -1.0))
        except (TypeError, ValueError):
            return evidence
        if (
            str(champion.get("digest") or "") != digest
            or not bool(audit.get("passed"))
            or not isinstance(per_opponent, dict)
            or any(oid not in per_opponent for oid in OFFICIAL_BASELINE_IDS)
            or gate_games != evidence_games
            or not math.isclose(
                gate_wr,
                evidence_wr,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return evidence
        upgraded = dict(evidence)
        upgraded["evidence_schema"] = 2
        upgraded["per_opponent"] = json.loads(json.dumps(per_opponent))
        upgraded["reconciled_from"] = "committed_history.stage_gate"
        return upgraded
    return evidence


def _terminal_marker_matches_committed_history(
    marker_payload: dict[str, Any], state: dict[str, Any]
) -> bool:
    """Return whether an immutable first-pass marker is in committed history.

    Continued training may later pass or fail another gate.  The first pass is
    still the archival boundary, so validation must use the append-only row
    that created the marker rather than whichever checkpoint is current now.
    """

    try:
        marker_iteration = int(marker_payload["iteration"])
    except (KeyError, TypeError, ValueError):
        return False
    for raw_row in list(state.get("history") or []):
        row = dict(raw_row or {})
        if int(row.get("iteration", -1)) != marker_iteration:
            continue
        candidate = dict(row.get("candidate") or {})
        result = _formal_active_gate_pass(row)
        if result is None:
            return False
        expected = {
            "iteration": marker_iteration,
            "wr": float(result["skill_weighted_wr"]),
            "confidence_lower": float(result["confidence_lower"]),
            "games": int(result["games"]),
            "checkpoint": str(candidate["path"]),
            "checkpoint_digest": str(candidate["digest"]),
        }
        return marker_payload == expected
    return False


def _ensure_terminal_gate_marker(
    run_dir: Path,
    state: dict[str, Any],
    *,
    preserve_first: bool = False,
    marker_name: str = "",
) -> Optional[Path]:
    """Recreate or validate the derived terminal marker idempotently."""
    selected_name = str(marker_name or "").strip() or (
        "CORE_GATE_PASSED"
        if str(state.get("mode")) == "core"
        else "SPECIALIST_GATE_PASSED"
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", selected_name):
        raise ValueError("terminal gate marker name must be a safe filename")
    marker = run_dir / selected_name
    payload = _terminal_gate_payload(state)
    if payload is None:
        if preserve_first and marker.exists():
            try:
                existing = json.loads(marker.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"invalid terminal gate marker: {marker}") from exc
            if not _terminal_marker_matches_committed_history(existing, state):
                raise RuntimeError(
                    f"terminal gate marker is absent from committed history: {marker}"
                )
            return marker
        return None
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid terminal gate marker: {marker}") from exc
        if existing != payload:
            if preserve_first and _terminal_marker_matches_committed_history(
                existing, state
            ):
                return marker
            raise RuntimeError(
                f"terminal gate marker disagrees with committed ledger: {marker}"
            )
    else:
        _write_json_exclusive(marker, payload)
    return marker


def _terminal_marker_gate_id(marker: Path, state: dict[str, Any]) -> str:
    """Resolve the exact active-gate ID that created a legacy marker."""
    try:
        marker_iteration = int(
            json.loads(Path(marker).read_text(encoding="utf-8"))["iteration"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ""
    for raw_row in list(state.get("history") or []):
        row = dict(raw_row or {})
        if int(row.get("iteration", -1)) != marker_iteration:
            continue
        return str((row.get("active_gate_result") or {}).get("gate_id") or "")
    return ""


def _gate_boundary_pause_seconds(
    args: argparse.Namespace,
    *,
    completed_iteration: int,
) -> float:
    """Return the mandatory post-gate pause for this committed boundary."""

    minimum = int(args.minimum_terminal_iteration)
    configured = float(args.gate_boundary_pause_seconds)
    if minimum < 0 or int(completed_iteration) < minimum or configured <= 0:
        return 0.0
    return configured


def _smoke_games(n: int, *, seed: int, archetype: str) -> list[CompactGame]:
    games: list[CompactGame] = []
    for i in range(n):
        games.append(
            CompactGame(
                episode_id=f"smoke-{seed}-{i}",
                seat=i % 2,
                archetype=archetype,
                opp_archetype="iono",
                deck=[1 + (i % 5)] * 60,
                value=1.0 if i % 3 else -1.0,
                decisions=[
                    CompactDecision(
                        env_step=0,
                        selected_index=i % 2,
                        n_options=2,
                        action=[i % 2],
                        observation={},
                    )
                ],
                target_provenance={"smoke": True, "pure_rl": True},
            )
        )
    return games


def _smoke_dataset(n: int, seed: int):
    from poke_bot import features
    from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, PolicyStage

    def sparse(words: int, offset: int = 0):
        sv = features.SparseVector()
        for i in range(words):
            sv.word_start()
            sv.add((offset + i) % 32, 1.0)
        return sv

    seqs = []
    for i in range(n):
        combos = [[0], [1]]
        dec = DecisionSample(
            board=sparse(features.NUM_BOARD_TOKENS, i),
            options=sparse(2, i + 3),
            action=[i % 2],
            action_combo_index=i % 2,
            action_combos=combos,
            env_step=0,
            action_token=sparse(1, i + 7),
            policy_stages=[
                PolicyStage(
                    options=sparse(2, i + 3),
                    action_combos=combos,
                    target_index=i % 2,
                )
            ],
        )
        seqs.append(
            GameSequence(
                episode_id=f"smoke-seq-{i}",
                seat=0,
                archetype="core",
                opp_archetype="iono",
                deck=[1] * 60,
                value=1.0 if i % 2 == 0 else -1.0,
                decisions=[dec],
                policy_targets=None,
                factorized_policy_targets=None,
                target_provenance={"pure_rl": True, "soft_policy_targets": False},
            )
        )
    return BootstrapDataset(sequences=seqs)


def _ensure_pure_rl_checkpoint(
    path: Path,
    seed: int,
    *,
    smoke: bool = False,
    allow_legacy_inference_profile: bool = False,
    r241_peak_r195_training_contract: Optional[Mapping[str, Any]] = None,
    r274_successor_training_contract: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Build or validate a fresh small Pure-RL seed (not AZ / not starter prior)."""
    import torch
    from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
    from poke_bot.train import load_model_from_checkpoint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Abhyuday: "The starter is terrible." — never CE-clone competition starter.
    banned = ("starter", "sample_submission", "rl_starter", "kaggle_starter")
    name_l = path.name.lower()
    path_l = str(path).lower()
    if any(tok in name_l or tok in path_l for tok in banned):
        raise SystemExit(
            f"PURE_RL refuse competition starter prior at {path}; "
            "use a fresh small pure_rl seed (Abhyuday: starter is terrible)"
        )
    if path.is_file():
        _checkpoint_contract(
            path,
            smoke=smoke,
            allow_legacy_inference_profile=allow_legacy_inference_profile,
            r241_peak_r195_training_contract=r241_peak_r195_training_contract,
            r274_successor_training_contract=r274_successor_training_contract,
        )
        model = load_model_from_checkpoint(path, device=torch.device("cpu"))
        n = count_params(model)
        print(f"[pure_rl] loaded checkpoint params={n} path={path}", flush=True)
        if r241_peak_r195_training_contract:
            expected = canonical_r241_peak_r195_training_contract(
                r241_peak_r195_training_contract
            )
            if n != int(expected["trainable_parameter_count"]):
                raise RuntimeError(
                    "r241 loaded model parameter count differs from its sealed receipt"
                )
        elif not r274_successor_training_contract:
            validate_param_budget(n, fail_max=int(config.PURE_RL.param_fail_max))
        cfg = getattr(model, "cfg", None)
        # Pure-RL overnight is d_model=96 L4/4/4 (~1.77M). Hope/legacy giants
        # are d_model>=192 — refuse those, not the intentional dense seed.
        if (
            not r241_peak_r195_training_contract
            and not r274_successor_training_contract
            and cfg is not None
            and int(getattr(cfg, "d_model", 0)) >= 192
        ):
            raise SystemExit(
                f"PURE_RL refuse Hope-sized checkpoint d_model={cfg.d_model} "
                f"at {path}; pass a small pure_rl seed or omit --base-checkpoint"
            )
        return path

    torch.manual_seed(seed)
    if r241_peak_r195_training_contract:
        raise RuntimeError(
            "r241 peak-r195 mode requires its immutable sealed parent checkpoint"
        )
    cfg = pure_rl_model_config(**({"dropout": 0.0} if smoke else {}))
    if smoke:
        # Tiny vocabs for CPU canary speed; still ≤3.5M with real vocab too.
        model = build_pure_rl_model(
            device=torch.device("cpu"),
            cfg=cfg,
            validate=True,
            aux_archetype_classes=8,
            encoder_vocab=128,
            decoder_vocab=128,
            belief_card_vocab=128,
        )
    else:
        model = build_pure_rl_model(device=torch.device("cpu"), cfg=cfg, validate=True)
    n = count_params(model)
    atomic_torch_save(
        build_checkpoint(
            model=model,
            step=0,
            epoch=0,
            model_config=cfg,
            extra={
                "pure_rl": True,
                "smoke": smoke,
                "param_count": n,
                "model_profile": model_config_dict(cfg),
            },
        ),
        path,
    )
    _checkpoint_contract(path, smoke=smoke)
    print(f"[pure_rl] wrote small seed params={n} path={path}", flush=True)
    return path


def _collect_temperature(args: argparse.Namespace, iteration: int) -> float:
    t0 = float(args.collect_temperature)
    t1 = float(getattr(config.PURE_RL, "collect_temperature_final", 0.7))
    anneal = max(1, int(getattr(config.PURE_RL, "temperature_anneal_iters", 50)))
    if iteration >= anneal:
        return t1
    frac = float(iteration) / float(anneal)
    return t0 + (t1 - t0) * frac


def _write_metrics(run_dir: Path, it: int, metrics: IterationMetrics) -> None:
    out = run_dir / "metrics" / f"iter_{it:05d}.json"
    _write_json_exclusive(out, metrics_to_dict(metrics))
    latest = run_dir / "metrics" / "latest.json"
    _atomic_json(latest, metrics_to_dict(metrics))


def _tqdm_mininterval() -> float:
    """Throttle bar redraws; floor 1s so file viewers stay calm."""
    raw = os.environ.get("PURE_RL_TQDM_MININTERVAL", "1.5")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1.5


def _heartbeat_sec() -> float:
    """Sparse stdout heartbeat so the monitor log does not look stalled."""
    raw = os.environ.get("PURE_RL_HEARTBEAT_SEC", "300")
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 300.0


def _progress_log_path() -> Path:
    return Path(
        os.environ.get(
            "PURE_RL_PROGRESS_LOG",
            str((ROOT / "outputs/logs/pure_rl_core.progress.log").resolve()),
        )
    )


def _progress_status_path(progress_log: Optional[Path] = None) -> Path:
    """Sibling single-line status file rewritten each bar tick."""
    p = progress_log or _progress_log_path()
    name = p.name
    if name.endswith(".progress.log"):
        return p.with_name(name[: -len(".progress.log")] + ".progress.status")
    return p.with_name(p.stem + ".progress.status")


def _progress_snapshot_sec() -> float:
    """How often to emit an archival newline into progress.log (0=off)."""
    raw = os.environ.get("PURE_RL_PROGRESS_SNAPSHOT_SEC", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def _tqdm_inplace_enabled() -> bool:
    raw = os.environ.get("PURE_RL_TQDM_INPLACE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _public_mix_local_only() -> bool:
    """Public mix (baseline vs public/roster opponents) finishes fast locally.

    Default ON: keep the light public-mix slice entirely on local workers so
    additive remotes (Elmo/bert) stay free for the (much larger) self-play
    wave and any other/slower work. Set 0/false to fall back to the shared
    heavy-local frac split (``PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC``) instead of
    excluding remotes outright. Self-play routing is untouched either way.
    """
    raw = os.environ.get("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _public_mix_min_local_frac() -> float:
    """Heavy-local fallback frac when ``PURE_RL_PUBLIC_MIX_LOCAL_ONLY=0``."""
    raw = os.environ.get("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "0.95")
    try:
        frac = float(raw)
    except ValueError:
        frac = 0.95
    return min(0.95, max(0.05, frac))


class _InPlaceProgressStream:
    """TTY-like stderr so tqdm keeps \\r in-place (no \\r→\\n spam).

    ``tail -F`` on an append-only file cannot show a true single line; use
    ``scripts/watch_pure_rl_progress.sh``, ``watch -n1 cat *.progress.status``,
    or ``less -r +F *.progress.log``. Sparse newlines are emitted on bar close
    (tqdm) and every ``PURE_RL_PROGRESS_SNAPSHOT_SEC`` for archival crumbs.
    """

    __slots__ = (
        "_stream",
        "_status_path",
        "_last_visible",
        "_last_snapshot",
        "_snapshot_sec",
    )

    def __init__(self, stream: Any, *, status_path: Path) -> None:
        self._stream = stream
        self._status_path = status_path
        self._last_visible = ""
        self._last_snapshot = time.time()
        self._snapshot_sec = _progress_snapshot_sec()

    def write(self, s: str) -> int:
        if not s:
            return 0
        written = self._stream.write(s)
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()
        chunk = s.replace("\n", "\r")
        if "\r" in chunk:
            part = chunk.rsplit("\r", 1)[-1]
        else:
            part = chunk
        visible = part.strip()
        if visible:
            self._last_visible = visible
            try:
                self._status_path.parent.mkdir(parents=True, exist_ok=True)
                self._status_path.write_text(visible + "\n", encoding="utf-8")
            except OSError:
                pass
        if (
            self._snapshot_sec > 0
            and "\n" not in s
            and self._last_visible
            and (time.time() - self._last_snapshot) >= self._snapshot_sec
        ):
            self._last_snapshot = time.time()
            self._stream.write("\n")
            if flush is not None:
                flush()
        return written

    def flush(self) -> None:
        flush = getattr(self._stream, "flush", None)
        if flush is not None:
            flush()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return int(self._stream.fileno())


class _TqdmProgress:
    """Collect/heldout bar on stderr; rare stdout heartbeat for monitors."""

    def __init__(
        self,
        *,
        stage: str,
        iteration: int,
        total: int,
        remotes: int = 0,
        inplace: Optional[bool] = None,
        mininterval: Optional[float] = None,
    ) -> None:
        from tqdm.auto import tqdm

        self.stage = stage
        self.iteration = iteration
        self.total = max(0, int(total))
        self.remotes = int(remotes)  # live open remote sockets (dispatch)
        self.remote_demand: Optional[int] = None  # scheduler demand (may lag sockets)
        self.remote_outstanding: Optional[int] = None
        self.remote_outstanding_elmo: Optional[int] = None
        self.remote_outstanding_bert: Optional[int] = None
        self.wr: Optional[str] = None  # live win-rate readout (heldout gate)
        self._t0 = time.time()
        self._last_heartbeat = self._t0
        self._done = 0
        self._decisions = 0
        err = sys.stderr
        to_file = not getattr(err, "isatty", lambda: False)()
        progress_hint = _progress_log_path()
        status_hint = _progress_status_path(progress_hint)
        use_inplace = _tqdm_inplace_enabled() if inplace is None else bool(inplace)
        bar_interval = (
            float(mininterval) if mininterval is not None else _tqdm_mininterval()
        )
        bar_file: Any = err
        if to_file and use_inplace:
            bar_file = _InPlaceProgressStream(err, status_path=status_hint)
        # Force a visible fill bar ({bar:N}). Default ascii+long postfix crushed
        # the meter to "|2|" in *.progress.status / watch.
        bar_format = (
            "{desc}: {percentage:3.0f}%|{bar:36}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        self._bar = tqdm(
            total=self.total,
            desc=f"pure_rl {stage} iter={iteration}",
            unit="game",
            file=bar_file,
            mininterval=max(1.0, bar_interval),
            miniters=1,
            ascii=False,  # unicode █/░ — readable in iTerm + status mirror
            bar_format=bar_format,
            dynamic_ncols=False,
            # The formal heldout postfix includes a two-decimal WR, game count,
            # and gate target.  140 columns clipped the second decimal in the
            # canonical status/tail file, so reserve enough fixed width for the
            # complete readout while retaining the 36-cell progress bar.
            ncols=168 if to_file else 140,
            leave=True,
            smoothing=0.1,
        )
        self._bar.set_postfix(**self._postfix(sps="0"))
        if to_file:
            mode = "in-place" if use_inplace else "line-spam"
            print(
                f"[pure_rl] tqdm games bar {mode} on stderr "
                f"(stage={stage} iter={iteration} total={self.total}); "
                f"rsock=active_sockets, rout=outstanding_remote_games "
                f"(rdmd=demand when desynced); "
                f"watch: bash scripts/watch_pure_rl_progress.sh "
                f"(or: watch -n1 cat {status_hint}; less -r +F {progress_hint})",
                flush=True,
            )

    def _postfix(self, *, sps: str) -> dict[str, Any]:
        """Report remote socket capacity separately from owned game results."""
        out: dict[str, Any] = {"rsock": int(self.remotes), "sps": sps}
        if self.remote_outstanding is not None:
            out["rout"] = int(self.remote_outstanding)
        if self.remote_outstanding_elmo is not None:
            out["eout"] = int(self.remote_outstanding_elmo)
        if self.remote_outstanding_bert is not None:
            out["bout"] = int(self.remote_outstanding_bert)
        if self.wr is not None:
            out["wr"] = self.wr
        dem = self.remote_demand
        if dem is not None and int(dem) != int(self.remotes):
            out["rdmd"] = int(dem)
        return out

    def set_remotes(
        self,
        active: int,
        *,
        demand: Optional[int] = None,
        outstanding: Optional[int] = None,
        outstanding_elmo: Optional[int] = None,
        outstanding_bert: Optional[int] = None,
    ) -> None:
        """Hot-update bar when mid-iter demand grows/shrinks dispatch sockets."""
        self.remotes = max(0, int(active))
        if demand is not None:
            self.remote_demand = max(0, int(demand))
        if outstanding is not None:
            self.remote_outstanding = max(0, int(outstanding))
        if outstanding_elmo is not None:
            self.remote_outstanding_elmo = max(0, int(outstanding_elmo))
        if outstanding_bert is not None:
            self.remote_outstanding_bert = max(0, int(outstanding_bert))
        try:
            # Keep last sps token if present; tick() refreshes on next game.
            prev = getattr(self._bar, "postfix", None) or {}
            sps = str(prev.get("sps", "0")) if isinstance(prev, dict) else "0"
            self._bar.set_postfix(**self._postfix(sps=sps), refresh=True)
        except Exception:
            pass

    def set_wr(
        self, win_rate: float, games: int, *, target: Optional[float] = None
    ) -> None:
        """Live win-rate readout — heldout gate games report outcomes eagerly.

        Called as each heldout game result lands, so the bar shows WR
        climbing/settling in real time instead of only at iter-end.
        """
        if games <= 0:
            return
        label = f"{win_rate:.2%}/{int(games)}g"
        if target is not None:
            label += f" (gate {target:.0%})"
        self.wr = label
        try:
            prev = getattr(self._bar, "postfix", None) or {}
            sps = str(prev.get("sps", "0")) if isinstance(prev, dict) else "0"
            self._bar.set_postfix(**self._postfix(sps=sps), refresh=True)
        except Exception:
            pass

    def tick(self, *, decisions: int = 0) -> None:
        self._done += 1
        self._decisions = int(decisions)
        elapsed = max(time.time() - self._t0, 1e-6)
        sps = self._decisions / elapsed
        # Postfix before update so the written status line includes sps.
        self._bar.set_postfix(
            **self._postfix(sps=f"{sps:.1f}"), refresh=False
        )
        self._bar.update(1)
        now = time.time()
        if (now - self._last_heartbeat) >= _heartbeat_sec():
            self._last_heartbeat = now
            gps = self._done / elapsed
            dem = (
                f" rdmd={self.remote_demand}"
                if self.remote_demand is not None
                and int(self.remote_demand) != int(self.remotes)
                else ""
            )
            print(
                f"[pure_rl] heartbeat stage={self.stage} iter={self.iteration} "
                f"games={self._done}/{self.total} "
                f"gps={gps:.2f} sps={sps:.1f} rsock={self.remotes} "
                f"rout={self.remote_outstanding if self.remote_outstanding is not None else '-'} "
                f"eout={self.remote_outstanding_elmo if self.remote_outstanding_elmo is not None else '-'} "
                f"bout={self.remote_outstanding_bert if self.remote_outstanding_bert is not None else '-'}{dem}",
                flush=True,
            )

    def close(self) -> None:
        try:
            self._bar.close()
        except Exception:
            pass


def _record_to_compact_game(record: dict[str, Any]) -> Optional[CompactGame]:
    """Strip soft behavior π; keep selected_index + observation for AWR."""
    steps = list(record.get("steps") or [])
    # Trajectory shards may use `decisions` instead of `steps`.
    acting_rows = steps if steps else list(record.get("decisions") or [])
    if not acting_rows:
        return None
    deck = [int(x) for x in (record.get("deck") or [])]
    combo_coverage: Optional[dict[str, int]] = None
    combo_provenance_key: Optional[str] = None
    if is_exact_slowking_deck(deck):
        combo_coverage = attach_slowking_combo_state_labels(
            acting_rows,
            deck=deck,
        )
        combo_provenance_key = "slowking_combo_state_targets"
    elif is_slop_box_combo_deck(deck):
        combo_coverage = attach_slop_box_combo_state_labels(
            acting_rows,
            deck=deck,
        )
        combo_provenance_key = "slop_box_combo_state_targets"
    soft = list(record.get("factorized_policy_targets") or [])
    decisions: list[CompactDecision] = []
    for i, step in enumerate(acting_rows):
        sel = 0
        n_opt = 1
        if i < len(soft) and soft[i]:
            row0 = soft[i][0] if isinstance(soft[i], list) else soft[i]
            if isinstance(row0, dict):
                sel = int(row0.get("selected_index", 0))
                combos = row0.get("action_combos") or []
                n_opt = max(len(combos), sel + 1, 1)
        elif step.get("selected_index") is not None:
            sel = int(step.get("selected_index") or 0)
            n_opt = max(int(step.get("n_options") or 0), sel + 1, 1)
        decisions.append(
            CompactDecision(
                env_step=int(step.get("env_step", i)),
                selected_index=sel,
                n_options=n_opt,
                action=[int(x) for x in (step.get("action") or [])],
                observation=dict(step.get("observation") or {}),
                aux_labels=dict(step.get("aux_labels") or {}),
                tactical_sequence_supervision=(
                    dict(step["tactical_sequence_supervision"])
                    if isinstance(
                        step.get("tactical_sequence_supervision"), dict
                    )
                    else None
                ),
            )
        )
    if not decisions:
        return None
    provenance = {
        **dict(record.get("target_provenance") or {}),
        "pure_rl": True,
        "soft_policy_targets": False,
    }
    if combo_coverage is not None and combo_provenance_key is not None:
        provenance[combo_provenance_key] = combo_coverage
    return CompactGame(
        episode_id=str(record.get("episode_id") or f"pure-rl-{time.time_ns()}"),
        seat=int(record.get("seat") or 0),
        archetype=str(record.get("archetype") or "core"),
        opp_archetype=str(record.get("opp_archetype") or ""),
        deck=deck,
        value=float(record.get("value") or 0.0),
        decisions=decisions,
        source="pure_rl",
        target_provenance=provenance,
    )


def _resolve_remote_endpoints(args: argparse.Namespace) -> list[str]:
    from poke_bot.remote_jobs import expand_endpoint_specs

    if args.smoke or args.no_remote_workers:
        return []
    raw = args.remote_worker_endpoints
    if raw is None:
        raw = os.environ.get("PURE_RL_REMOTE_WORKER_ENDPOINTS")
        if raw is None:
            raw = os.environ.get(
                "POKEBOT_REMOTE_WORKER_ENDPOINTS", DEFAULT_REMOTE_ENDPOINTS
            )
    if not str(raw).strip():
        return []
    return expand_endpoint_specs([str(raw)])


def _our_decks(
    mode: str, specialist_archetype: Optional[str] = None
) -> list[tuple[str, list[int]]]:
    """Stage A core: wide multi-archetype pool (baselines + archetype-samples).

    A specialist must name one exact deck from the immutable ladder
    representative catalog. Core must not collapse to ``default_pool()``
    (dragapult + hammer-pult only).
    """
    from poke_bot.deck_pool import default_pool, primary_deck, primary_archetype, read_deck
    from poke_bot import paths as _paths

    if mode == "specialist":
        requested = str(specialist_archetype or "").strip().lower()
        if not requested:
            raise ValueError("specialist mode requires an explicit archetype")
        override_raw = os.environ.get("POKEBOT_SPECIALIST_DECK_PATH", "").strip()
        if override_raw:
            from poke_bot.archetypes import classify_deck

            override = Path(override_raw).expanduser().resolve()
            expected_file = os.environ.get(
                "POKEBOT_SPECIALIST_DECK_SHA256", ""
            ).removeprefix("sha256:")
            expected_multiset = os.environ.get(
                "POKEBOT_SPECIALIST_DECK_MULTISET_SHA256", ""
            ).removeprefix("sha256:")
            if (
                not override.is_file()
                or len(expected_file) != 64
                or _sha256_file(override).removeprefix("sha256:")
                != expected_file
            ):
                raise ValueError("specialist deck override file identity mismatch")
            cards = read_deck(override)
            canonical = hashlib.sha256(
                json.dumps(sorted(cards), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                len(cards) != 60
                or canonical != expected_multiset
                or classify_deck(cards) != requested
            ):
                raise ValueError("specialist deck override multiset mismatch")
            return [(requested, list(cards))]
        pinned, _mix, _representatives, _contract = _core_ladder_decks()
        matches = [
            (name, list(cards))
            for name, cards in pinned
            if str(name).strip().lower() == requested
        ]
        if not matches:
            from poke_bot.archetypes import classify_deck
            from poke_bot.ladder_deck_mix import canonical_payload_digest

            try:
                payload = json.loads(
                    SPECIALIST_DECK_REPRESENTATIVES_PATH.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "cannot load the pinned specialist representative catalog "
                    f"{SPECIALIST_DECK_REPRESENTATIVES_PATH}: {exc}"
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("schema")
                != "poke_bot.specialist_deck_representatives/v1"
            ):
                raise ValueError("invalid pinned specialist representative schema")
            declared_digest = str(payload.get("artifact_sha256") or "")
            actual_digest = canonical_payload_digest(payload)
            if declared_digest != actual_digest:
                raise ValueError(
                    "pinned specialist representative digest mismatch: "
                    f"declared={declared_digest!r} actual={actual_digest!r}"
                )
            row = dict((payload.get("decks") or {}).get(requested) or {})
            cards = row.get("card_ids")
            if (
                not isinstance(cards, list)
                or len(cards) != 60
                or any(isinstance(card, bool) or not isinstance(card, int) for card in cards)
            ):
                raise ValueError(
                    f"specialist representative {requested!r} is not exactly "
                    "one valid 60-card integer list"
                )
            sorted_cards = sorted(cards)
            legacy_canonical = ",".join(
                str(card_id) for card_id in sorted_cards
            ).encode("ascii")
            json_canonical = json.dumps(
                sorted_cards, separators=(",", ":")
            ).encode("utf-8")
            canonical_digests = {
                "sha256:" + hashlib.sha256(legacy_canonical).hexdigest(),
                "sha256:" + hashlib.sha256(json_canonical).hexdigest(),
            }
            if str(row.get("canonical_multiset_sha256") or "") not in canonical_digests:
                raise ValueError(
                    f"specialist representative {requested!r} multiset digest mismatch"
                )
            classified = classify_deck(cards)
            source_deck_id = str(row.get("source_deck_id") or "")
            from poke_bot.matchup_adapters import LOGICAL_EXPERT_ALIASES_V5

            aliased_identity = (
                LOGICAL_EXPERT_ALIASES_V5.get(source_deck_id) == requested
            )
            if classified != requested and not aliased_identity:
                raise ValueError(
                    f"specialist representative {requested!r} classifies as "
                    f"{classified!r}"
                )
            matches = [(requested, list(cards))]
        if len(matches) != 1:
            raise ValueError(
                f"specialist archetype {requested!r} is not one exact pinned "
                "representative in either immutable catalog; "
                f"ladder_available={sorted(name for name, _ in pinned)}"
            )
        return matches

    out: list[tuple[str, list[int]]] = []
    seen_names: set[str] = set()
    seen_lists: set[tuple[int, ...]] = set()

    def _add(name: str, deck: list[int]) -> None:
        n = str(name).strip()
        if not n or n in seen_names:
            return
        key = tuple(int(x) for x in deck)
        if len(key) != 60 or key in seen_lists:
            return
        seen_names.add(n)
        seen_lists.add(key)
        out.append((n, list(deck)))

    # 1) All installable baseline decks from baselines/manifest.json
    try:
        from poke_bot.baselines_runtime import ensure_baselines_installed, load_manifest

        for spec in ensure_baselines_installed(load_manifest()):
            try:
                if spec.deck_csv.is_file():
                    _add(spec.id, read_deck(spec.deck_csv))
            except Exception:
                continue
    except Exception:
        pass

    # 2) decks/archetype-samples/ (dozens of archetype variants)
    samples_dir = _paths.DECKS_DIR / "archetype-samples"
    if samples_dir.is_dir():
        for path in sorted(samples_dir.glob("*.csv")):
            try:
                _add(path.stem, read_deck(path))
            except Exception:
                continue

    # 3) Keep default_pool entries when they add a unique list/name
    try:
        pool = default_pool()
        for name in pool.names():
            try:
                _add(name, pool.get(name).load())
            except Exception:
                continue
    except Exception:
        pass

    if not out:
        out.append((primary_archetype(), primary_deck()))
    return out


def _select_measurement_decks(
    decks: list[tuple[str, list[int]]], requested: str | None
) -> list[tuple[str, list[int]]]:
    """Resolve a strict, ordered evaluation-only subset of the training pool."""
    tokens = [token.strip().lower() for token in str(requested or "").split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        return [(name, list(cards)) for name, cards in decks]
    if len(tokens) != len(set(tokens)):
        raise ValueError("--measurement-decks contains duplicate deck IDs")
    by_id = {str(name).strip().lower(): (name, cards) for name, cards in decks}
    missing = [token for token in tokens if token not in by_id]
    if missing:
        raise ValueError(
            "--measurement-decks contains IDs outside the active training pool: "
            f"{missing}; available={sorted(by_id)}"
        )
    return [(by_id[token][0], list(by_id[token][1])) for token in tokens]


def _core_ladder_decks() -> tuple[
    list[tuple[str, list[int]]], Any, Any, dict[str, Any]
]:
    """Load the pinned top-ladder family mix and exact modal representatives."""
    from poke_bot.ladder_deck_mix import (
        load_ladder_deck_mix,
        load_ladder_deck_representatives,
    )

    mix = load_ladder_deck_mix(LADDER_DECK_MIX_PATH)
    representatives = load_ladder_deck_representatives(
        LADDER_DECK_REPRESENTATIVES_PATH
    )
    bound = representatives.bind(mix)
    decks = [
        (entry.bucket.deck_id, list(entry.card_ids))
        for entry in bound
    ]
    contract = representatives.contract(mix)
    return decks, mix, representatives, contract


def _derange_deck_schedule(
    ours: tuple[str, ...], opponents: tuple[str, ...]
) -> tuple[str, ...]:
    """Preserve exact quotas and minimize same-family self-play deterministically.

    Both schedules are sampled from the same ladder quotas.  Grouping equal
    families and rotating the opponent tokens by the largest quota constructs
    a multiset derangement whenever one is possible (``2 * max_quota <= n``).
    When a tiny wave makes that impossible, the same construction leaves the
    theoretical minimum ``2 * max_quota - n`` same-family pairs instead of
    crashing the unattended trainer.
    """
    from collections import Counter

    if len(ours) != len(opponents):
        raise ValueError("self-play deck schedules have different lengths")
    n = len(ours)
    if n <= 0:
        return ()

    our_counts = Counter(ours)
    opponent_counts = Counter(opponents)
    if our_counts != opponent_counts:
        raise ValueError(
            "self-play deck schedules must have identical ladder quotas"
        )

    # Keeping the largest group first makes a shift by its size disjoint from
    # every group when a full derangement is feasible.  Label is the stable
    # tie-break, so the result is reproducible across Python processes.
    families = sorted(our_counts, key=lambda family: (-our_counts[family], family))
    grouped_positions = [
        index
        for family in families
        for index, scheduled_family in enumerate(ours)
        if scheduled_family == family
    ]
    grouped_opponents = [
        family for family in families for _ in range(opponent_counts[family])
    ]
    shift = max(our_counts.values())
    rotated = grouped_opponents[shift:] + grouped_opponents[:shift]
    candidate = [""] * n
    for position, opponent in zip(grouped_positions, rotated):
        candidate[position] = opponent
    return tuple(candidate)


def _spec_payload(spec) -> dict[str, Any]:
    from poke_bot.baselines_runtime import baseline_spec_payload

    return baseline_spec_payload(spec)


class _LeafFarm:
    """Local dual-GPU leaf servers for host CPU workers."""

    def __init__(self) -> None:
        self.procs: list = []
        self.req_qs: list = []
        self.ctrl_qs: list = []
        self.status_qs: list = []
        self.alive_evts: list = []
        self.resp_qs: list = []
        self.remote_channel = None
        self.version = 0
        self.digest: Optional[str] = None

    def start(
        self,
        *,
        ckpt: Path,
        digest: str,
        leaf_devices: list[int],
        n_workers: int,
        max_batch: Optional[int],
        coalesce_ms: float,
    ) -> None:
        import multiprocessing as mp
        from queue import Empty

        from poke_bot.batched_infer import run_leaf_server

        # Rebuild-safe: never append onto stale queues from a prior farm.
        if self.procs or self.status_qs:
            self.stop()

        self.digest = digest
        self.version = 0
        mpctx = mp.get_context("spawn")
        # Leaf replicas may exceed CPU worker slots (VRAM-packing for util).
        n_servers = max(1, len(leaf_devices))
        devices = leaf_devices[:n_servers]
        self.procs = []
        self.req_qs = []
        self.ctrl_qs = []
        self.status_qs = []
        self.alive_evts = []
        self.resp_qs = [mpctx.Queue(maxsize=2) for _ in range(n_workers)]
        slot_counter = mpctx.Value("i", 0)
        readies = []
        for j, dev in enumerate(devices):
            rq = mpctx.Queue(maxsize=64)
            cq = mpctx.Queue(maxsize=8)
            sq = mpctx.Queue(maxsize=16)
            ev = mpctx.Event()
            alive = mpctx.Event()
            proc = mpctx.Process(
                target=run_leaf_server,
                args=(str(ckpt), f"cuda:{dev}", rq, self.resp_qs),
                kwargs=dict(
                    ready_evt=ev,
                    alive_evt=alive,
                    ctrl_q=cq,
                    status_q=sq,
                    expected_digest=digest,
                    initial_version=self.version,
                    bf16=True,
                    max_batch=max_batch,
                    coalesce_ms=coalesce_ms,
                ),
                daemon=True,
            )
            proc.start()
            self.procs.append(proc)
            self.req_qs.append(rq)
            self.ctrl_qs.append(cq)
            self.status_qs.append(sq)
            self.alive_evts.append(alive)
            readies.append(ev)
        for j, ev in enumerate(readies):
            if not ev.wait(timeout=240):
                self.stop()
                raise RuntimeError(f"leaf server {j} not ready in 240s")
            try:
                status = self.status_qs[j].get(timeout=60)
            except Empty as exc:
                self.stop()
                raise RuntimeError(
                    f"leaf server {j} ready without status ack (device={devices[j]})"
                ) from exc
            if not status.get("ok") or not self.procs[j].is_alive():
                self.stop()
                raise RuntimeError(f"leaf server {j} bad ready ack: {status}")
        self.remote_channel = {
            "req_qs": self.req_qs,
            "resp_qs": self.resp_qs,
            "slot_counter": slot_counter,
            "ctrl_qs": self.ctrl_qs,
            "generation": 0,
            "alive_evts": self.alive_evts,
            "expected_digest": digest,
            "expected_version": self.version,
            "timeout_s": config.SEARCH.remote_request_timeout_s,
            # Even-spread device map + GPU0-biased sticky / least-queue feed.
            "leaf_devices": list(devices),
            "gpu0_client_frac": float(
                os.environ.get("PURE_RL_GPU0_CLIENT_FRAC", "0.38")
            ),
        }
        print(
            f"[pure_rl] leaf-eval=gpu-server x{len(self.procs)} devices={devices} "
            f"workers={n_workers} coalesce_ms={coalesce_ms} "
            f"gpu0_client_frac={self.remote_channel['gpu0_client_frac']:.2f}",
            flush=True,
        )

    def reload(self, ckpt: Path, digest: str) -> None:
        """Reload all leaf servers; hard-fail unless every ack matches ``digest``."""
        if not self.ctrl_qs:
            return
        requested = self.version + 1
        mismatches: list[str] = []
        # One fail-closed deadline covers the entire fan-out.  A separate
        # wait per queue can serialize missing acknowledgements into an
        # hour-long boundary stall.  The deadline must begin before control
        # publication as well: ``multiprocessing.Queue.put`` can block when a
        # stale/full leaf control queue never drains.
        reload_deadline = time.monotonic() + 240.0
        sent: list[bool] = []
        for i, cq in enumerate(self.ctrl_qs):
            remaining = reload_deadline - time.monotonic()
            if remaining <= 0.0:
                mismatches.append(
                    f"leaf[{i}] control publish timeout/error: "
                    "global reload deadline exceeded"
                )
                sent.append(False)
                continue
            try:
                cq.put(
                    {
                        "cmd": "reload",
                        "path": str(ckpt),
                        "digest": digest,
                        "version": requested,
                    },
                    timeout=min(5.0, remaining),
                )
                sent.append(True)
            except Exception as exc:
                mismatches.append(
                    f"leaf[{i}] control publish timeout/error: {exc}"
                )
                sent.append(False)
        for i, sq in enumerate(self.status_qs):
            if i >= len(sent) or not sent[i]:
                continue
            remaining = reload_deadline - time.monotonic()
            if remaining <= 0.0:
                mismatches.append(
                    f"leaf[{i}] status timeout/error: global reload deadline exceeded"
                )
                continue
            try:
                status = sq.get(timeout=remaining)
            except Exception as exc:
                mismatches.append(f"leaf[{i}] status timeout/error: {exc}")
                continue
            if status.get("type") != "reload" or not status.get("ok"):
                mismatches.append(f"leaf[{i}] bad ack: {status!r}")
                continue
            got = status.get("checkpoint_digest")
            if got != digest:
                mismatches.append(
                    f"leaf[{i}] digest mismatch: expected {digest}, got {got}"
                )
        if mismatches:
            raise RuntimeError(
                "leaf reload hard-gate failed: " + "; ".join(mismatches)
            )
        self.version = requested
        self.digest = digest
        if self.remote_channel is not None:
            self.remote_channel["expected_digest"] = digest
            self.remote_channel["expected_version"] = self.version

    def stop(self) -> None:
        procs = list(self.procs)
        queues = [
            *self.req_qs,
            *self.ctrl_qs,
            *self.status_qs,
            *self.resp_qs,
        ]
        for cq in self.ctrl_qs:
            try:
                cq.put({"cmd": "stop"})
            except Exception:
                pass
        for proc in procs:
            try:
                proc.join(timeout=5)
            except Exception:
                pass
            try:
                alive = bool(proc.is_alive())
            except Exception:
                alive = False
            if alive:
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Reap a force-terminated child.  Omitting this second join
                # leaked process handles and named multiprocessing semaphores
                # on every leaf-farm rebuild.
                try:
                    proc.join(timeout=5)
                except Exception:
                    pass
            try:
                alive = bool(proc.is_alive())
            except Exception:
                alive = False
            if alive and hasattr(proc, "kill"):
                try:
                    proc.kill()
                    proc.join(timeout=5)
                except Exception:
                    pass
            try:
                if not proc.is_alive() and hasattr(proc, "close"):
                    proc.close()
            except Exception:
                pass
        for q in queues:
            close_mp_queue(q)
        self.procs.clear()
        self.req_qs.clear()
        self.ctrl_qs.clear()
        self.status_qs.clear()
        self.alive_evts.clear()
        self.resp_qs.clear()
        self.remote_channel = None


def run_smoke_loop(args: argparse.Namespace) -> int:
    hw = full_hardware_profile()
    hw = replace(
        hw,
        allow_single_gpu=True,
        leaf_gpu0_replicas=max(1, hw.leaf_gpu0_replicas),
        leaf_gpu1_replicas=max(1, hw.leaf_gpu1_replicas),
    )
    hw.validate_or_raise(visible_gpu_count=1)

    run_dir = _run_dir(args.run_name, smoke=True)
    ckpt = args.base_checkpoint or (run_dir / "checkpoints" / "seed.pt")
    ckpt = _ensure_pure_rl_checkpoint(Path(ckpt), args.seed, smoke=True)

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    adv_hist: list[float] = []
    agr_hist: list[float] = []

    manifest = {
        "run_name": args.run_name,
        "mode": args.mode,
        "smoke": True,
        "hardware": hw.as_dict(),
        "stage": stage_to_dict(stage),
        "created": time.time(),
        "gate_wr": float(args.gate_wr),
        "heldout_games": int(args.heldout_games),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for it in range(args.iterations):
        t0 = time.time()
        next_shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
        writer = CompactShardWriter(next_shard)
        n_games = args.smoke_games
        collect_future_games = _smoke_games(
            n_games,
            seed=args.seed + it,
            archetype=(
                "core" if args.mode == "core" else str(args.specialist_archetype)
            ),
        )

        def _collect() -> None:
            writer.write_games(collect_future_games)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_collect)
            dataset = _smoke_dataset(n_games, args.seed + it)
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
                aux_loss_weight=float(args.archetype_aux_loss_weight),
                opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
                lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
                prize_race_loss_weight=float(args.prize_race_loss_weight),
                alakazam_guide_loss_weight=float(
                    args.alakazam_guide_loss_weight
                ),
                current_deck_guide_training_mode=str(
                    args.current_deck_guide_training_mode
                ),
                setup_board_outcome_loss_weight=float(
                    args.setup_board_outcome_loss_weight
                ),
                combo_state_loss_weight=float(args.combo_state_loss_weight),
                # Smoke never resolves the receipt-bound Inzi sidecar.  Keep
                # its synthetic legacy path incapable of minting r260 labels
                # or pretending to exercise the runtime-enabled successor.
                visible_tutor_completion_loss_weight=0.0,
                terminal_conversion_loss_weight=0.0,
                collect_own_deck_promotion_metrics=False,
                current_deck_guide_curriculum_spec=str(
                    args.current_deck_guide_curriculum_spec or ""
                ),
                current_deck_guide_head_role_map=str(
                    args.current_deck_guide_head_role_map or ""
                ),
                current_deck_guide_curriculum_validation_receipt=str(
                    args.current_deck_guide_curriculum_validation_receipt or ""
                ),
            )
            import torch
            from poke_bot.checkpoint import atomic_torch_save, build_checkpoint
            from poke_bot.train import load_model_from_checkpoint, batch_losses

            model = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
            model.train()
            opt = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=train_cfg.lr,
            )
            total, bm = batch_losses(
                model,
                list(dataset.sequences),
                value_weight=1.0,
                aux_weight=train_cfg.aux_loss_weight,
                opp_hand_weight=train_cfg.opp_hand_loss_weight,
                opp_remainder_weight=train_cfg.opp_remainder_loss_weight,
                lethal_threat_weight=train_cfg.lethal_threat_loss_weight,
                prize_race_weight=train_cfg.prize_race_loss_weight,
                alakazam_guide_weight=train_cfg.alakazam_guide_loss_weight,
                current_deck_guide_training_mode=(
                    train_cfg.current_deck_guide_training_mode
                ),
                setup_board_outcome_loss_weight=(
                    train_cfg.setup_board_outcome_loss_weight
                ),
                combo_state_loss_weight=train_cfg.combo_state_loss_weight,
                expanded_head_weights=train_cfg.expanded_head_loss_weights,
                pure_rl=True,
                awr_beta=train_cfg.awr_beta,
                awr_weight_max=train_cfg.awr_weight_max,
            )
            if bm.n_decisions > 0 and torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            atomic_torch_save(
                build_checkpoint(
                    model=model,
                    step=it + 1,
                    epoch=1,
                    model_config=getattr(model, "cfg", None),
                    extra={"pure_rl": True, "iteration": it, "mode": args.mode},
                ),
                out_ckpt,
            )
            ckpt = out_ckpt
            fut.result()

        thr = writer.throughput()
        smoke_heldout_games = int(args.heldout_games or 200)
        rows = [
            {
                "opponent_id": oid,
                "our_seat": 0,
                "winner": 0,
                "baseline_failed": False,
            }
            for oid in OFFICIAL_BASELINE_IDS
            for _ in range(
                max(1, smoke_heldout_games // len(OFFICIAL_BASELINE_IDS))
            )
        ]
        gate = aggregate_heldout_wr(
            rows, target_wr=args.gate_wr, min_games=smoke_heldout_games
        )
        adv_hist.append(float(bm.mean_advantage))
        agr_hist.append(0.5)
        abort = evaluate_aborts(
            mean_advantages=adv_hist, policy_prev_agreements=agr_hist, k=3
        )
        elapsed = max(time.time() - t0, 1e-6)
        metrics = IterationMetrics(
            iteration=it,
            stage=stage.stage.value,
            games=writer.n_games,
            decisions=writer.n_decisions,
            games_per_sec=thr["games_per_sec"],
            decisions_per_sec=thr["decisions_per_sec"],
            games_per_hour=thr["games_per_sec"] * 3600.0,
            mean_return=float(bm.target_value_mean),
            mean_advantage=float(bm.mean_advantage),
            awr_weight_mean=float(bm.awr_weight_mean),
            awr_weight_p50=float(bm.awr_weight_p50),
            awr_weight_p95=float(bm.awr_weight_p95),
            awr_weight_clip_frac=float(bm.awr_weight_clip_frac),
            policy_selected_nll=float(bm.policy_selected_nll),
            policy_prev_agreement=0.5,
            self_distill_flag=abort.self_distill_flag,
            heldout_wr=gate.win_rate,
            heldout_games=gate.games,
            gate_passed=gate.passed and not abort.abort,
            extra={
                "abort": asdict(abort),
                "elapsed_sec": elapsed,
                "hardware": hw.as_dict(),
                "checkpoint": str(ckpt),
            },
        )
        _write_metrics(run_dir, it, metrics)
        print(
            f"[pure_rl smoke] iter={it} games={metrics.games} "
            f"awr_w={metrics.awr_weight_mean:.3f} heldout_wr={gate.win_rate:.3f} "
            f"gate={gate.passed} abort={abort.abort}",
            flush=True,
        )
        if gate.passed and not abort.abort:
            if args.mode == "core":
                (run_dir / "CORE_GATE_PASSED").write_text(
                    json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
                )
                print("[pure_rl] CORE GATE PASSED", flush=True)
                break
            (run_dir / "SPECIALIST_GATE_PASSED").write_text(
                json.dumps({"iteration": it, "wr": gate.win_rate}), encoding="utf-8"
            )
            print("[pure_rl] SPECIALIST GATE PASSED", flush=True)
            break
        if abort.abort:
            print(f"[pure_rl] abort promote: {abort.reason}", flush=True)
            return 2
    return 0


def _latest_official_heldout_win_rates(
    state: dict[str, Any],
    opponent_ids: tuple[str, ...] = OFFICIAL_BASELINE_IDS,
) -> dict[str, float]:
    """Return the newest exact heldout rates that cover every official ID."""
    wanted = tuple(str(opponent_id) for opponent_id in opponent_ids)
    candidates: list[Any] = []
    history = state.get("history") if isinstance(state, dict) else None
    if isinstance(history, list):
        for entry in reversed(history):
            if not isinstance(entry, dict) or not bool(entry.get("completed")):
                continue
            candidates.extend(
                [entry.get("raw_heldout_gate"), entry.get("heldout_gate")]
            )
    candidates.append(
        state.get("heldout_champion_evidence") if isinstance(state, dict) else None
    )
    for candidate in candidates:
        per_opponent = None
        if isinstance(candidate, dict):
            raw_per_opponent = candidate.get("per_opponent")
            if isinstance(raw_per_opponent, dict):
                per_opponent = raw_per_opponent
            else:
                matchups = candidate.get("matchups")
                if isinstance(matchups, list):
                    indexed = {
                        str(row.get("opponent_id") or ""): row
                        for row in matchups
                        if isinstance(row, dict)
                        and str(row.get("opponent_id") or "")
                    }
                    if len(indexed) == len(matchups):
                        per_opponent = indexed
        if not isinstance(per_opponent, dict):
            continue
        rates: dict[str, float] = {}
        for opponent_id in wanted:
            row = per_opponent.get(opponent_id)
            if not isinstance(row, dict) or float(row.get("games") or 0) <= 0:
                break
            value = float(row.get("win_rate", row.get("wr", -1.0)))
            if not 0.0 <= value <= 1.0:
                break
            rates[opponent_id] = value
        if len(rates) == len(wanted):
            return rates
    return {}


def _adaptive_official_target_weights(
    opponent_ids: tuple[str, ...],
    win_rates: dict[str, float],
    *,
    target_win_rate: float,
    minimum_share: float,
    gap_power: float,
    skill_weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Blend a non-starvation floor with tier-weighted exact-gate deficits."""
    ids = tuple(str(opponent_id) for opponent_id in opponent_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("official target IDs must be non-empty and unique")
    floor = float(minimum_share)
    power = float(gap_power)
    target = float(target_win_rate)
    if not 0.0 <= floor <= 1.0 / len(ids):
        raise ValueError("minimum official target share is infeasible")
    if not 0.0 < power <= 4.0 or not 0.0 <= target <= 1.0:
        raise ValueError("invalid official target gap configuration")
    if set(win_rates) != set(ids) or any(
        not 0.0 <= float(win_rates[opponent_id]) <= 1.0
        for opponent_id in ids
    ):
        return {opponent_id: 1.0 / len(ids) for opponent_id in ids}
    if skill_weights is None:
        normalized_skill_weights = {opponent_id: 1.0 for opponent_id in ids}
    else:
        if set(skill_weights) != set(ids):
            raise ValueError("skill weights must cover the exact target roster")
        normalized_skill_weights = {
            opponent_id: float(skill_weights[opponent_id])
            for opponent_id in ids
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in normalized_skill_weights.values()
        ):
            raise ValueError("skill weights must be finite and positive")
    deficits = {
        opponent_id: (
            max(0.0, target - float(win_rates[opponent_id])) ** power
            * normalized_skill_weights[opponent_id]
        )
        for opponent_id in ids
    }
    deficit_total = sum(deficits.values())
    if deficit_total <= 1e-12:
        return {opponent_id: 1.0 / len(ids) for opponent_id in ids}
    adaptive_mass = 1.0 - floor * len(ids)
    weights = {
        opponent_id: floor
        + adaptive_mass * deficits[opponent_id] / deficit_total
        for opponent_id in ids
    }
    # Normalize once to remove floating-point accumulation without changing
    # the relative allocation used by the integer quota scheduler.
    total = sum(weights.values())
    return {opponent_id: weights[opponent_id] / total for opponent_id in ids}


def _planned_collection_group_counts(
    *,
    games_per_iteration: int,
    self_play_fraction: float,
    strong_public_fraction_of_public: float,
    research_control_games: int,
) -> dict[str, int]:
    """Resolve the exact *training* quotas before any work starts.

    Research controls used to consume ``research_control_games`` slots inside
    this budget.  They are now an additive diagnostic wave, so those former
    slots are reclaimed for the active-gate practice group.  The returned
    counts describe only replay/AWR-eligible games and must sum to ``total``.
    """
    total = int(games_per_iteration)
    if total <= 0:
        raise ValueError("games per iteration must be positive")
    self_frac = min(1.0, max(0.0, float(self_play_fraction)))
    practice_frac = min(
        1.0, max(0.0, float(strong_public_fraction_of_public))
    )
    n_self = int(round(total * self_frac))
    if self_frac > 0.0 and n_self == 0:
        n_self = 1
    if self_frac < 1.0 and n_self == total:
        n_self = max(0, total - 1)
    n_public = total - n_self
    base_practice = int(round(n_public * practice_frac))
    reclaimed = int(research_control_games)
    if reclaimed < 0 or reclaimed > n_public - base_practice:
        raise ValueError(
            "research-control reclaim does not fit the fixed training budget: "
            f"total={total} self_play={n_self} base_strong_public={base_practice} "
            f"reclaimed_for_strong_public={reclaimed} available_after_practice="
            f"{n_public - base_practice}"
        )
    n_practice = base_practice + reclaimed
    result = {
        "self_play": n_self,
        STRONG_PUBLIC_PRACTICE_GROUP: n_practice,
        "diverse_public": n_public - n_practice,
    }
    if sum(result.values()) != total:
        raise RuntimeError("collection group quotas do not conserve the game budget")
    return result


def _strong_public_practice_minimum_games(
    *,
    active_gate: dict[str, Any],
    expected_practice_games: int,
    minimum_share: float,
) -> dict[str, int]:
    """Reserve explicit gate-row floors without starving the other gate rows.

    Ordinary gates retain their established weighted scheduler. Once a gate
    declares an explicit practice floor, every row receives at least the normal
    share floor and the named row receives the larger explicit value. The
    remaining slots are the only ones available to adaptive Hamilton weights.
    """
    roster = list(active_gate.get("roster") or [])
    ids = tuple(str(row.get("opponent_id") or "") for row in roster)
    if not ids or "" in ids or len(set(ids)) != len(ids):
        raise RuntimeError("active practice roster IDs must be non-empty and unique")
    generic_floor = math.floor(
        int(expected_practice_games) * float(minimum_share)
    )
    if generic_floor < 0:
        raise RuntimeError("strong-public practice minimum share is invalid")
    explicit: dict[str, int] = {}
    for row, opponent_id in zip(roster, ids):
        raw = row.get("strong_public_practice_floor_games")
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RuntimeError(
                "strong-public practice floor must be a positive integer: "
                f"{opponent_id}={raw!r}"
            )
        explicit[opponent_id] = int(raw)
    if not explicit:
        return {}
    floors = {
        opponent_id: max(generic_floor, explicit.get(opponent_id, 0))
        for opponent_id in ids
    }
    reserved = sum(floors.values())
    if reserved > int(expected_practice_games):
        raise RuntimeError(
            "strong-public practice floors exceed the fixed quota: "
            f"reserved={reserved} quota={int(expected_practice_games)} "
            f"minimum_share={float(minimum_share)} floors={floors}"
        )
    return floors


def _active_archetype_family_training_contract(
    *, specialist_archetype: str,
) -> Optional[dict[str, Any]]:
    """Resolve the atomically activated family manifest and loss vector."""
    names = {
        "manifest": "POKEBOT_ARCHETYPE_FAMILY_MANIFEST",
        "loss_contract": "POKEBOT_ARCHETYPE_LOSS_CONTRACT",
        "loss_vector": "POKEBOT_ARCHETYPE_LOSS_VECTOR",
    }
    supplied = {key: str(os.environ.get(env, "")).strip() for key, env in names.items()}
    if not any(supplied.values()):
        return None
    if not all(supplied.values()):
        raise RuntimeError("archetype family activation is not atomic")
    from poke_bot.archetype_loss_contract import (
        canonical_residual_weights,
        validate_loss_contract,
    )
    from poke_bot.specialist_archetype_family import (
        FamilyDeckMix,
        SPECIALIST_ID,
        validate_manifest,
    )

    if str(specialist_archetype) != SPECIALIST_ID:
        raise RuntimeError("Marnie family contract selected for another specialist")
    paths = {key: Path(value).expanduser().resolve() for key, value in supplied.items()}
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    manifest = validate_manifest(payloads["manifest"], require_activation_ready=True)
    loss_contract = validate_loss_contract(payloads["loss_contract"])
    vector = payloads["loss_vector"]
    vector_status = str(vector.get("status") or "")
    owner_ceiling_vector = (
        vector_status
        == "selected_by_owner_ceiling_after_inconclusive_study"
    )
    vector_without_digest = dict(vector)
    declared_vector_digest = str(
        vector_without_digest.pop("artifact_sha256", "")
    )
    if (
        vector.get("schema") != "poke_bot.archetype_loss_vector/v1"
        or vector.get("specialist_id") != SPECIALIST_ID
        or vector_status
        not in {
            "selected_by_passing_shadow_study",
            "selected_by_owner_ceiling_after_inconclusive_study",
        }
        or vector.get("manifest_sha256") != _sha256_file(paths["manifest"])
        or vector.get("loss_contract_sha256") != _sha256_file(paths["loss_contract"])
        or vector.get("activates_only_with_family_sampler") is not True
        or vector.get("serving_authority") is not False
        or declared_vector_digest != _canonical_digest(vector_without_digest)
        or (
            owner_ceiling_vector
            and (
                vector.get("measured_study_passed") is not False
                or not str(
                    vector.get("activation_authority_sha256") or ""
                ).startswith("sha256:")
            )
        )
        or (
            not owner_ceiling_vector
            and "activation_authority_sha256" in vector
        )
    ):
        raise RuntimeError("selected archetype loss vector identity is invalid")
    residuals = canonical_residual_weights(vector.get("residual_objectives"))
    mix = FamilyDeckMix.from_manifest(manifest)
    by_id = {str(row["variant_id"]): row for row in manifest["variants"]}
    decks = [
        (entry.deck_id, list(by_id[entry.deck_id]["card_ids"]))
        for entry in mix.decks
    ]
    return {
        "manifest": manifest,
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "loss_contract": loss_contract,
        "loss_contract_path": str(paths["loss_contract"]),
        "loss_contract_sha256": _sha256_file(paths["loss_contract"]),
        "loss_vector": vector,
        "loss_vector_path": str(paths["loss_vector"]),
        "loss_vector_sha256": _sha256_file(paths["loss_vector"]),
        "residual_objectives": residuals,
        "mix": mix,
        "decks": decks,
    }


def _assert_seed_namespace_contract(
    *,
    root_seed: int,
    iterations: int,
    games_per_iteration: int,
    formal_games: int,
    research_control_games: int = 0,
) -> dict[str, Any]:
    """Fail closed if collection, gate, or control seed ranges can overlap."""
    n_iterations = int(iterations)
    collect_size = int(games_per_iteration)
    gate_size = int(formal_games)
    research_size = int(research_control_games)
    if n_iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < collect_size < ITERATION_SEED_STRIDE:
        raise ValueError("games per iteration must fit inside its seed stride")
    if not 0 < gate_size < ITERATION_SEED_STRIDE:
        raise ValueError("formal gate games must fit inside its seed stride")
    if not 0 <= research_size < ITERATION_SEED_STRIDE:
        raise ValueError("research-control games must fit inside its seed stride")

    collect_ranges = [
        (
            int(root_seed) + iteration * ITERATION_SEED_STRIDE,
            int(root_seed) + iteration * ITERATION_SEED_STRIDE + collect_size - 1,
        )
        for iteration in range(n_iterations)
    ]
    gate_ranges = [
        (
            int(root_seed)
            + FORMAL_GATE_SEED_OFFSET
            + iteration * ITERATION_SEED_STRIDE,
            int(root_seed)
            + FORMAL_GATE_SEED_OFFSET
            + iteration * ITERATION_SEED_STRIDE
            + gate_size
            - 1,
        )
        for iteration in range(n_iterations)
    ]
    research_ranges = [
        (
            int(root_seed)
            + RESEARCH_CONTROL_SEED_OFFSET
            + iteration * ITERATION_SEED_STRIDE,
            int(root_seed)
            + RESEARCH_CONTROL_SEED_OFFSET
            + iteration * ITERATION_SEED_STRIDE
            + research_size
            - 1,
        )
        for iteration in range(n_iterations)
        if research_size > 0
    ]
    namespaces = {
        "training": collect_ranges,
        "formal-gate": gate_ranges,
        "research-control": research_ranges,
    }
    names = tuple(namespaces)
    for left_i, left_name in enumerate(names):
        for right_name in names[left_i + 1 :]:
            for left_iteration, (left_start, left_end) in enumerate(
                namespaces[left_name]
            ):
                for right_iteration, (right_start, right_end) in enumerate(
                    namespaces[right_name]
                ):
                    if max(left_start, right_start) > min(left_end, right_end):
                        continue
                    raise RuntimeError(
                        "seed namespaces overlap: "
                        f"{left_name}_iter={left_iteration} "
                        f"[{left_start},{left_end}] "
                        f"{right_name}_iter={right_iteration} "
                        f"[{right_start},{right_end}]"
                    )
    return {
        "schema": "poke_bot.seed_namespace_contract/v1",
        "root_seed": int(root_seed),
        "iterations": n_iterations,
        "iteration_stride": ITERATION_SEED_STRIDE,
        "training_namespace": "train/global-collection-v1",
        "formal_gate_namespace": "eval/strong-public-fixed-manifest-v1",
        "formal_gate_offset": FORMAL_GATE_SEED_OFFSET,
        "research_control_namespace": "eval/research-controls-fixed-manifest-v1",
        "research_control_offset": RESEARCH_CONTROL_SEED_OFFSET,
        "games_per_iteration": collect_size,
        "formal_games": gate_size,
        "research_control_games": research_size,
        "disjoint": True,
    }


def _assert_strong_public_practice_jobs(
    *,
    all_jobs: list[dict[str, Any]],
    public_jobs: list[dict[str, Any]],
    active_gate: dict[str, Any],
    expected_practice_games: int,
    iteration: int,
    root_seed: int,
    formal_games: int,
    minimum_share: float,
    practice_temperature: float,
    minimum_games_by_opponent: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Audit the training-only active-roster slice before any game launches."""
    from collections import Counter

    roster = list(active_gate.get("roster") or [])
    roster_ids = tuple(str(row.get("opponent_id") or "") for row in roster)
    if not roster_ids or len(set(roster_ids)) != len(roster_ids) or "" in roster_ids:
        raise RuntimeError("active practice roster IDs must be non-empty and unique")
    archetypes = {
        str(row["opponent_id"]): str(row.get("archetype_id") or "")
        for row in roster
    }
    if any(not value for value in archetypes.values()):
        raise RuntimeError("active practice roster is missing an archetype ID")

    seeds = [int(job["seed"]) for job in all_jobs]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("global collection schedule contains duplicate seeds")
    practice_jobs = [
        job
        for job in public_jobs
        if str((job.get("target_provenance") or {}).get("opponent_training_group"))
        == STRONG_PUBLIC_PRACTICE_GROUP
    ]
    if len(practice_jobs) != int(expected_practice_games):
        raise RuntimeError(
            "strong-public practice quota mismatch: "
            f"actual={len(practice_jobs)} expected={int(expected_practice_games)}"
        )
    actual_ids = {str(job.get("opponent_id") or "") for job in practice_jobs}
    if actual_ids != set(roster_ids):
        raise RuntimeError(
            "strong-public practice roster mismatch: "
            f"actual={sorted(actual_ids)} expected={sorted(roster_ids)}"
        )
    leaked_gate_ids = sorted(
        str(job.get("opponent_id") or "")
        for job in public_jobs
        if str(job.get("opponent_id") or "") in set(roster_ids)
        and str((job.get("target_provenance") or {}).get("opponent_training_group"))
        != STRONG_PUBLIC_PRACTICE_GROUP
    )
    if leaked_gate_ids:
        raise RuntimeError(
            "active gate opponent leaked into a non-practice training group: "
            f"{sorted(set(leaked_gate_ids))}"
        )

    counts = Counter(str(job["opponent_id"]) for job in practice_jobs)
    seat_counts: dict[str, dict[str, int]] = {}
    if minimum_games_by_opponent is None:
        quota_floors = {
            opponent_id: math.floor(
                int(expected_practice_games) * float(minimum_share)
            )
            for opponent_id in roster_ids
        }
    else:
        if set(minimum_games_by_opponent) != set(roster_ids):
            raise RuntimeError(
                "strong-public practice minimums must cover the exact active roster"
            )
        quota_floors = {}
        for opponent_id in roster_ids:
            value = minimum_games_by_opponent[opponent_id]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    "strong-public practice minimum must be a nonnegative integer: "
                    f"{opponent_id}={value!r}"
                )
            quota_floors[opponent_id] = int(value)
        if sum(quota_floors.values()) > int(expected_practice_games):
            raise RuntimeError("strong-public practice minimums exceed the quota")
    gate_id = str(active_gate.get("id") or "")
    for opponent_id in roster_ids:
        rows = [job for job in practice_jobs if str(job["opponent_id"]) == opponent_id]
        quota_floor = quota_floors[opponent_id]
        if int(counts[opponent_id]) < quota_floor:
            raise RuntimeError(
                f"practice opponent {opponent_id} is below its minimum quota"
            )
        seat0 = sum(int(job.get("our_seat", -1)) == 0 for job in rows)
        seat1 = sum(int(job.get("our_seat", -1)) == 1 for job in rows)
        if abs(seat0 - seat1) > 1:
            raise RuntimeError(f"practice seats are imbalanced for {opponent_id}")
        seat_counts[opponent_id] = {"seat0": seat0, "seat1": seat1}
        for job in rows:
            provenance = dict(job.get("target_provenance") or {})
            if (
                job.get("training_eligible") is not True
                or job.get("sample_actions") is not True
                or bool(job.get("greedy", False))
                or str(provenance.get("collect")) != STRONG_PUBLIC_PRACTICE_GROUP
                or str(provenance.get("active_gate_id")) != gate_id
                or provenance.get("formal_eval") is not False
                or str(provenance.get("seed_namespace"))
                != "train/strong-public-practice-v1"
                or str(provenance.get("opponent_id") or "") != opponent_id
                or str(provenance.get("opponent_archetype_id") or "")
                != archetypes[opponent_id]
            ):
                raise RuntimeError(
                    f"practice job contract mismatch for {opponent_id}"
                )
            if str(job.get("opp_archetype") or "") != archetypes[opponent_id]:
                raise RuntimeError(
                    f"practice archetype mismatch for {opponent_id}: "
                    f"{job.get('opp_archetype')} != {archetypes[opponent_id]}"
                )
            if minimum_games_by_opponent is not None and int(
                provenance.get("strong_public_practice_floor_games", -1)
            ) != quota_floor:
                raise RuntimeError(
                    f"practice floor provenance mismatch for {opponent_id}"
                )
            if not math.isclose(
                float(job.get("action_temperature", -1.0)),
                float(practice_temperature),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"practice temperature mismatch for {opponent_id}"
                )

    formal_start = (
        int(root_seed)
        + FORMAL_GATE_SEED_OFFSET
        + int(iteration) * ITERATION_SEED_STRIDE
    )
    formal_seed_set = set(range(formal_start, formal_start + int(formal_games)))
    practice_seed_set = {int(job["seed"]) for job in practice_jobs}
    if practice_seed_set & formal_seed_set:
        raise RuntimeError("practice jobs overlap the current formal-gate seed set")
    return {
        "schema": "poke_bot.strong_public_practice_plan/v1",
        "active_gate_id": gate_id,
        "iteration": int(iteration),
        "training_eligible": True,
        "formal_eval": False,
        "sampled_policy": True,
        "temperature": float(practice_temperature),
        "seed_namespace": "train/strong-public-practice-v1",
        "formal_seed_namespace": "eval/strong-public-fixed-manifest-v1",
        "seed_disjoint": True,
        "games": len(practice_jobs),
        "minimum_games_by_opponent": dict(quota_floors),
        "per_opponent": {
            opponent_id: {
                "games": int(counts[opponent_id]),
                "minimum_games": int(quota_floors[opponent_id]),
                **seat_counts[opponent_id],
                "archetype_id": archetypes[opponent_id],
            }
            for opponent_id in roster_ids
        },
    }


def _interleaved_opponent_schedule(
    n_games: int,
    *,
    priority_specs: list[Any],
    diverse_specs: list[Any],
    priority_frac: float,
    seed: int,
    iteration: int,
    priority_weights: Optional[dict[str, float]] = None,
    priority_minimum_games: Optional[dict[str, int]] = None,
    diverse_minimum_games: Optional[dict[str, int]] = None,
    priority_group: str = "official_target",
) -> tuple[tuple[Any, str], ...]:
    """Build an exact, evenly interleaved practice/diverse training schedule.

    Training against a known public baseline is not a formal-evaluation row:
    the caller supplies a disjoint seed range for the later greedy gate.  This
    schedule merely controls which policy produces experience. Research
    controls are deliberately absent: they run later as an additive greedy
    measurement transaction and cannot enter this replay-eligible schedule.
    """
    total = max(0, int(n_games))
    if total <= 0:
        return ()
    priority = list(priority_specs)
    diverse = list(diverse_specs)
    if not priority and not diverse:
        return ()
    if not priority:
        n_priority = 0
    elif not diverse:
        n_priority = total
    else:
        n_priority = int(round(total * min(1.0, max(0.0, float(priority_frac)))))
    remainder = total - n_priority
    n_diverse = remainder
    if n_diverse > 0 and not diverse:
        raise ValueError(
            "public schedule has unallocated remainder but no diverse roster"
        )

    group_ids = {
        "priority": {str(spec.id) for spec in priority},
        "diverse_public": {str(spec.id) for spec in diverse},
    }
    if group_ids["priority"] & group_ids["diverse_public"]:
        raise ValueError("opponent IDs cannot appear in multiple training groups")

    def _rotation(rows: list[Any], label: str) -> int:
        if not rows:
            return 0
        token = f"{int(seed)}:{int(iteration)}:{label}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % len(rows)

    priority_label = str(priority_group).strip()
    if not priority_label:
        raise ValueError("priority opponent group must be non-empty")
    p_offset = _rotation(priority, priority_label)
    d_offset = _rotation(diverse, "diverse_public")
    priority_sequence: list[Any] = []
    if (
        (priority_weights is not None or priority_minimum_games is not None)
        and n_priority > 0
    ):
        ids = [str(spec.id) for spec in priority]
        if len(set(ids)) != len(ids):
            raise ValueError("weighted official target specs must be unique")
        if priority_weights is None:
            raw = [1.0] * len(ids)
        else:
            if set(priority_weights) != set(ids):
                raise ValueError("official target weights must cover the exact roster")
            raw = [float(priority_weights[opponent_id]) for opponent_id in ids]
        if any(not math.isfinite(value) or value < 0.0 for value in raw):
            raise ValueError("official target weights must be finite and nonnegative")
        raw_total = sum(raw)
        if raw_total <= 0.0:
            raise ValueError("official target weights must contain positive mass")
        if priority_minimum_games is None:
            floor_by_id = {opponent_id: 0 for opponent_id in ids}
        else:
            if set(priority_minimum_games) != set(ids):
                raise ValueError("priority minimums must cover the exact roster")
            floor_by_id = {}
            for opponent_id in ids:
                value = priority_minimum_games[opponent_id]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "priority minimum games must be nonnegative integers"
                    )
                floor_by_id[opponent_id] = int(value)
        quotas = [floor_by_id[opponent_id] for opponent_id in ids]
        reserved = sum(quotas)
        if reserved > n_priority:
            raise ValueError(
                "priority minimum games exceed the fixed priority quota: "
                f"reserved={reserved} quota={n_priority}"
            )
        remaining_slots = n_priority - reserved
        ideals = [remaining_slots * value / raw_total for value in raw]
        increments = [int(math.floor(value)) for value in ideals]
        quotas = [quota + increment for quota, increment in zip(quotas, increments)]
        remaining = remaining_slots - sum(increments)
        tie_rank = {
            (p_offset + index) % len(priority): index
            for index in range(len(priority))
        }
        remainder_order = sorted(
            range(len(priority)),
            key=lambda index: (
                -(ideals[index] - increments[index]),
                tie_rank[index],
            ),
        )
        for index in remainder_order[:remaining]:
            quotas[index] += 1
        # Smooth weighted round robin spreads the exact Hamilton quotas across
        # the wave instead of emitting a large weakest-opponent burst.
        current = [0] * len(priority)
        order = [
            (p_offset + index) % len(priority)
            for index in range(len(priority))
        ]
        order_rank = {index: rank for rank, index in enumerate(order)}
        for _ in range(n_priority):
            for index, quota in enumerate(quotas):
                current[index] += quota
            selected = max(
                range(len(priority)),
                key=lambda index: (current[index], -order_rank[index]),
            )
            current[selected] -= n_priority
            priority_sequence.append(priority[selected])
        assert {
            str(spec.id): sum(str(row.id) == str(spec.id) for row in priority_sequence)
            for spec in priority
        } == {str(spec.id): quotas[index] for index, spec in enumerate(priority)}
    diverse_sequence: list[Any] = []
    if diverse_minimum_games is not None and n_diverse > 0:
        ids = [str(spec.id) for spec in diverse]
        if set(diverse_minimum_games) != set(ids):
            raise ValueError("diverse minimums must cover the exact diverse roster")
        floors: list[int] = []
        for opponent_id in ids:
            value = diverse_minimum_games[opponent_id]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("diverse minimum games must be nonnegative integers")
            floors.append(int(value))
        if sum(floors) > n_diverse:
            raise ValueError("diverse minimum games exceed the fixed diverse quota")
        remaining = n_diverse - sum(floors)
        quotas = [
            floor + remaining // len(diverse)
            for floor in floors
        ]
        for offset in range(remaining % len(diverse)):
            quotas[(d_offset + offset) % len(diverse)] += 1
        current = [0] * len(diverse)
        order = [(d_offset + index) % len(diverse) for index in range(len(diverse))]
        rank = {index: position for position, index in enumerate(order)}
        for _ in range(n_diverse):
            for index, quota in enumerate(quotas):
                current[index] += quota
            selected = max(
                range(len(diverse)),
                key=lambda index: (current[index], -rank[index]),
            )
            current[selected] -= n_diverse
            diverse_sequence.append(diverse[selected])
        assert {
            str(spec.id): sum(str(row.id) == str(spec.id) for row in diverse_sequence)
            for spec in diverse
        } == {str(spec.id): quotas[index] for index, spec in enumerate(diverse)}
    p_index = d_index = 0
    schedule: list[tuple[Any, str]] = []
    for position in range(total):
        # Cumulative integer quotas spread target rows through the whole wave.
        take_priority = (
            ((position + 1) * n_priority) // total
            > (position * n_priority) // total
        )
        if take_priority:
            spec = (
                priority_sequence[p_index]
                if priority_sequence
                else priority[(p_offset + p_index) % len(priority)]
            )
            p_index += 1
            schedule.append((spec, priority_label))
        else:
            spec = (
                diverse_sequence[d_index]
                if diverse_sequence
                else diverse[(d_offset + d_index) % len(diverse)]
            )
            d_index += 1
            schedule.append((spec, "diverse_public"))
    assert p_index == n_priority and d_index == n_diverse
    return tuple(schedule)


def _public_training_seat(
    *,
    seed: int,
    iteration: int,
    archetype: str,
    opponent_id: str,
    occurrence: int,
) -> int:
    """Alternate seats independently for every deck/opponent cell.

    Public-group interleaving can have the same parity as the global game
    index (the 50/50 specialist wave alternates official/diverse exactly).
    Using ``game_i % 2`` therefore pins an entire group to one seat.  A stable
    per-cell offset plus occurrence parity gives exact balance for even cells
    and at most one-game skew for odd cells without correlating seat to group.
    """
    token = (
        f"{int(seed)}:{int(iteration)}:{str(archetype)}:"
        f"{str(opponent_id)}:public_training_seat_v1"
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1
    return int((offset + max(0, int(occurrence))) & 1)


def _assert_r274_r195_research_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    expected_content_digest: str,
    iteration: int,
) -> dict[str, Any]:
    """Prove the exact 55378392 cell before dispatching any update games."""

    selected = [
        dict(job)
        for job in jobs
        if str(job.get("opponent_id") or "") == R274_R195_RESEARCH_OPPONENT_ID
    ]
    seat_counts = {
        seat: sum(int(job.get("our_seat", -1)) == seat for job in selected)
        for seat in (0, 1)
    }
    if (
        len(selected) < R274_R195_RESEARCH_GAMES_MINIMUM
        or seat_counts[0] < R274_R195_RESEARCH_EACH_SEAT_MINIMUM
        or seat_counts[1] < R274_R195_RESEARCH_EACH_SEAT_MINIMUM
    ):
        raise RuntimeError(
            "r274 submission-55378392 research cell lacks its 128-game 64/64 floor"
        )
    for job in selected:
        provenance = dict(job.get("target_provenance") or {})
        spec = dict(job.get("spec") or {})
        # Pre-dispatch rows are complete simulator jobs.  Post-collection rows
        # are the deliberately smaller canonical runtime audit projection;
        # _consume_results binds the same immutable opponent fields onto that
        # projection before the row can be retained.
        # Dispatched job plans carry ``spec``.  Consumed result projections
        # deliberately omit it while retaining target provenance for the
        # immutable opponent/audit binding.  Provenance presence therefore
        # cannot distinguish the two representations.
        runtime_projection = not bool(spec)
        content_digest = str(
            provenance.get("opponent_content_digest")
            or job.get("opponent_content_digest")
            or ""
        )
        training_group = str(
            provenance.get("opponent_training_group")
            or job.get("opponent_training_group")
            or ""
        )
        formal_eval = (
            provenance.get("formal_eval")
            if provenance
            else job.get("formal_eval")
        )
        if (
            training_group != "diverse_public"
            or formal_eval is not False
            or content_digest != expected_content_digest
            or (not runtime_projection and spec.get("content_digest") != expected_content_digest)
            or job.get("training_eligible") is not True
        ):
            raise RuntimeError(
                "r274 submission-55378392 research job identity changed"
            )
    return {
        "schema": "poke_bot.r274_submission_55378392_research_plan/v1",
        "iteration": int(iteration),
        "submission_id": 55378392,
        "opponent_id": R274_R195_RESEARCH_OPPONENT_ID,
        "bundle_sha256": R274_R195_RESEARCH_BUNDLE_SHA256,
        "model_sha256": R274_R195_RESEARCH_MODEL_SHA256,
        "installed_content_sha256": expected_content_digest,
        "games": len(selected),
        "learner_first_games": seat_counts[0],
        "learner_second_games": seat_counts[1],
        "inside_public_mix": True,
        "training_eligible": True,
        "formal_eval": False,
        "rtp_enabled": False,
        "transport": "singleton_play",
        "passed": True,
    }


def _paired_official_exploit(
    *,
    seed: int,
    iteration: int,
    archetype: str,
    opponent_id: str,
    occurrence: int,
    fraction: float,
) -> bool:
    """Select a deterministic fraction of whole two-seat occurrence pairs.

    Consecutive occurrences for a public deck/opponent cell use opposite seats.
    Making the temperature decision on ``occurrence // 2`` prevents behavior
    temperature from becoming a seat label.  The hashed phase spreads selected
    pairs across the wave while preserving the requested fraction to within one
    pair for any prefix.
    """
    frac = min(1.0, max(0.0, float(fraction)))
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    pair_index = max(0, int(occurrence)) // 2
    token = (
        f"{int(seed)}:{int(iteration)}:{str(archetype)}:"
        f"{str(opponent_id)}:official_exploit_pair_v1"
    ).encode("utf-8")
    phase = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 1009
    before = math.floor((phase + pair_index) * frac)
    after = math.floor((phase + pair_index + 1) * frac)
    return bool(after > before)


def _build_collect_jobs(
    *,
    n_games: int,
    ckpt: Path,
    digest: str,
    model_generation: int,
    decks: list[tuple[str, list[int]]],
    specs: list,
    seed: int,
    game_timeout_s: int,
    mode: str,
    collect_temperature: float = 1.0,
    max_context: Optional[int] = None,
    opponent_pool: Optional[list[Any]] = None,
    self_play_frac: Optional[float] = None,
    balanced_eval: bool = False,
    ladder_mix: Any = None,
    family_manifest: Optional[Mapping[str, Any]] = None,
    iteration: int = 0,
    priority_specs: Optional[list[Any]] = None,
    priority_frac: float = 0.0,
    priority_weights: Optional[dict[str, float]] = None,
    priority_minimum_games: Optional[dict[str, int]] = None,
    diverse_minimum_games: Optional[dict[str, int]] = None,
    priority_group: str = "official_target",
    priority_temperature: Optional[float] = None,
    priority_archetypes: Optional[dict[str, str]] = None,
    priority_context: Optional[dict[str, Any]] = None,
    official_exploit_opponents: Optional[tuple[str, ...]] = None,
    official_exploit_frac: float = 0.0,
    official_exploit_temperature: float = 0.35,
    exact_training_seat_split: bool = False,
    own_deck_ledger_enabled: Optional[bool] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(self_play_jobs, baseline_jobs)`` — self-play is the primary signal."""
    self_jobs: list[dict[str, Any]] = []
    base_jobs: list[dict[str, Any]] = []
    if not decks:
        return self_jobs, base_jobs
    ctx = int(max_context if max_context is not None else pure_rl_model_config().max_context)
    if own_deck_ledger_enabled is None:
        own_deck_ledger_enabled = os.environ.get(
            "POKEBOT_OWN_DECK_LEDGER_RUNTIME", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    frac = float(
        self_play_frac
        if self_play_frac is not None
        else getattr(config.PURE_RL, "self_play_frac", 0.85)
    )
    frac = min(1.0, max(0.0, frac))
    n_self = int(round(n_games * frac))
    if n_games > 0 and frac > 0 and n_self == 0:
        n_self = 1
    has_public_specs = bool(specs or priority_specs)
    if n_games > 0 and frac < 1 and n_self == n_games and has_public_specs:
        n_self = max(0, n_games - 1)
    if not has_public_specs:
        n_self = n_games
    ladder_catalog: dict[str, tuple[str, list[int]]] = {}
    self_our_schedule: tuple[str, ...] = ()
    self_opp_schedule: tuple[str, ...] = ()
    public_our_schedule: tuple[str, ...] = ()
    public_spec_schedule: tuple[tuple[Any, str], ...] = ()
    public_seat_counts: dict[tuple[str, str], int] = {}
    exploit_opponents = frozenset(
        str(opponent_id) for opponent_id in (official_exploit_opponents or ())
    )
    ladder_wave_provenance: dict[str, Any] = {}
    family_variants: dict[str, dict[str, Any]] = {}
    family_id = ""
    family_manifest_digest = ""
    if family_manifest is not None:
        from poke_bot.specialist_archetype_family import validate_manifest

        validated_family = validate_manifest(
            family_manifest, require_activation_ready=True
        )
        if ladder_mix is None:
            raise RuntimeError("family collection requires its bound deck mix")
        family_id = str(validated_family["family_id"])
        family_manifest_digest = str(validated_family["artifact_sha256"])
        family_variants = {
            str(row["variant_id"]): dict(row)
            for row in validated_family["variants"]
            if str(row.get("split")) == "train"
        }

    def family_provenance(variant_id: str) -> dict[str, Any]:
        if not family_variants:
            return {}
        row = family_variants.get(str(variant_id))
        if row is None:
            raise RuntimeError(
                f"scheduled family variant is not train-admitted: {variant_id}"
            )
        return {
            "family_id": family_id,
            "variant_id": str(variant_id),
            "manifest_digest": family_manifest_digest,
            "ordered_digest": str(row["ordered_digest"]),
            "multiset_digest": str(row["multiset_digest"]),
        }
    if ladder_mix is not None:
        if balanced_eval:
            raise ValueError("ladder-weighted collection is not a formal eval sampler")
        ladder_catalog = {str(name): (str(name), list(deck)) for name, deck in decks}
        expected_ids = {entry.deck_id for entry in ladder_mix.decks}
        if set(ladder_catalog) != expected_ids:
            raise RuntimeError(
                "bound ladder catalog does not match the mix IDs: "
                f"missing={sorted(expected_ids - set(ladder_catalog))} "
                f"extra={sorted(set(ladder_catalog) - expected_ids)}"
            )
        self_our_schedule = ladder_mix.schedule_ids(
            n_self,
            seed=seed,
            iteration=iteration,
            stream="self_play_our",
        )
        raw_opp_schedule = ladder_mix.schedule_ids(
            n_self,
            seed=seed,
            iteration=iteration,
            stream="self_play_opp",
        )
        self_opp_schedule = _derange_deck_schedule(
            self_our_schedule, raw_opp_schedule
        )
        public_our_schedule = ladder_mix.schedule_ids(
            n_games - n_self,
            seed=seed,
            iteration=iteration,
            stream="public_our",
        )
        ladder_wave_provenance = {
            "mix_id": ladder_mix.mix_id,
            "artifact_sha256": ladder_mix.artifact_sha256,
            "basis": "train",
            "iteration": int(iteration),
            "self_play_our_quotas": ladder_mix.quotas(n_self),
            "self_play_opp_quotas": ladder_mix.quotas(n_self),
            "public_our_quotas": ladder_mix.quotas(n_games - n_self),
            **(
                {
                    "family_id": family_id,
                    "manifest_digest": family_manifest_digest,
                }
                if family_variants
                else {}
            ),
        }
    if n_games > n_self and float(priority_frac) > 0.0:
        public_spec_schedule = _interleaved_opponent_schedule(
            n_games - n_self,
            priority_specs=list(priority_specs or []),
            diverse_specs=list(specs),
            priority_frac=float(priority_frac),
            seed=int(seed),
            iteration=int(iteration),
            priority_weights=priority_weights,
            priority_minimum_games=priority_minimum_games,
            diverse_minimum_games=diverse_minimum_games,
            priority_group=str(priority_group),
        )
    # Portable baseline identity hashing walks each installed source tree.
    # A 64k public wave must do that once per opponent, not once per game.
    spec_payloads: dict[str, dict[str, Any]] = {}
    for spec in [
        *list(specs),
        *list(priority_specs or []),
    ]:
        spec_id = str(spec.id)
        payload = dict(_spec_payload(spec))
        content_digest = str(payload.get("content_digest") or "").lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", content_digest):
            if content_digest:
                raise RuntimeError(
                    f"opponent {spec_id!r} has a malformed content digest"
                )
            # Minimal test specs and legacy portable payloads may predate the
            # explicit field. Bind them to their exact canonical payload rather
            # than dropping provenance or weakening scheduler assertions.
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            payload["content_digest"] = (
                "sha256:" + hashlib.sha256(canonical).hexdigest()
            )
        previous = spec_payloads.setdefault(spec_id, payload)
        if previous != payload:
            raise RuntimeError(
                f"opponent ID {spec_id!r} resolves to conflicting payloads"
            )
    raw_pool = list(opponent_pool or [{"path": str(ckpt), "digest": str(digest)}])
    pool: list[dict[str, str]] = []
    for entry in raw_pool:
        if isinstance(entry, dict):
            opp_path = str(entry.get("path") or "")
            opp_digest = str(entry.get("digest") or "")
        elif hasattr(entry, "path") and hasattr(entry, "digest"):
            opp_path = str(entry.path)
            opp_digest = str(entry.digest)
        else:
            opp_path = str(entry)
            opp_digest = str(digest) if opp_path == str(ckpt) else ""
        if not opp_path or not opp_digest:
            raise RuntimeError(
                "self-play opponent pool requires path+digest identities; "
                f"invalid entry={entry!r}"
            )
        pool.append({"path": opp_path, "digest": opp_digest})
    if not pool:
        pool = [{"path": str(ckpt), "digest": str(digest)}]
    for game_i in range(n_games):
        our_variant_id = ""
        if balanced_eval and specs:
            # Adjacent games are the same deck/opponent with candidate seat
            # flipped. This removes the old opponent↔seat parity confound.
            pair_i = game_i // 2
            spec_i = pair_i % len(specs)
            deck_i = (pair_i // len(specs)) % len(decks)
            arch, deck = decks[deck_i]
            our_seat = game_i % 2
        elif ladder_mix is not None:
            if game_i < n_self:
                scheduled_id = self_our_schedule[game_i]
            else:
                scheduled_id = public_our_schedule[game_i - n_self]
            scheduled_arch, deck = ladder_catalog[scheduled_id]
            our_variant_id = str(scheduled_id) if family_variants else ""
            arch = family_id if family_variants else scheduled_arch
            # The family schedule is already SHA-256 permuted. Alternating the
            # seat keeps global balance without pinning a family to one seat.
            our_seat = (game_i + int(seed)) % 2
        else:
            arch, deck = decks[game_i % len(decks)]
            # Seat changes after one complete deck rotation, so each deck is
            # observed from both seats instead of being pinned to index parity.
            our_seat = (game_i // len(decks)) % 2
        common = {
            "job_index": game_i,
            "checkpoint": str(ckpt),
            "checkpoint_digest": digest,
            "model_generation": model_generation,
            "model_max_context": ctx,
            "our_deck": list(deck),
            "our_seat": our_seat,
            "mcts_sims": 0,
            "mcts_move_time": 0.0,
            "game_timeout_s": int(game_timeout_s),
            "agent_mode": "policy",
            "sample_actions": True,
            "action_temperature": float(collect_temperature),
            "seed": int(seed + game_i),
            "device": "cpu",
            "training_eligible": True,
            "archetype": arch,
            "collect_privileged_belief": True,
            # GPU-leaf workers intentionally have model=None and therefore
            # cannot infer successor capabilities from a local checkpoint.
            # Bind the receipt-selected ledger route into the immutable job.
            "own_deck_ledger_enabled": bool(own_deck_ledger_enabled),
            # The rule-derivative sidecar consumes local option-hidden states;
            # it cannot be reconstructed from the legacy remote-leaf logits
            # ABI.  Bind that requirement into every collection job so the
            # worker loads the exact checkpoint locally and fails closed if
            # the trained projection is absent.
            "public_rule_semantic_projection_required": os.environ.get(
                "POKEBOT_PUBLIC_RULE_SEMANTIC_PROJECTION_REQUIRED", ""
            ).strip().lower() in {"1", "true", "yes", "on"},
        }
        if ladder_mix is not None:
            common["deck_mix"] = {
                **ladder_wave_provenance,
                "scheduled_our_deck_id": our_variant_id or arch,
                "stream": (
                    "self_play_our" if game_i < n_self else "public_our"
                ),
            }
        if our_variant_id:
            common.update(family_provenance(our_variant_id))
        if game_i < n_self or not has_public_specs:
            opp_identity = pool[game_i % len(pool)]
            opp_ckpt = opp_identity["path"]
            opp_digest = opp_identity["digest"]
            if ladder_mix is not None:
                scheduled_opp_id = self_opp_schedule[game_i]
                _opp_name, opp_deck = ladder_catalog[scheduled_opp_id]
                opp_variant_id = (
                    str(scheduled_opp_id) if family_variants else ""
                )
                opp_arch = family_id if family_variants else scheduled_opp_id
            elif len(decks) > 1:
                opp_variant_id = ""
                our_i = game_i % len(decks)
                matchup_round = game_i // len(decks)
                opp_i = (
                    our_i + 1 + (matchup_round % (len(decks) - 1))
                ) % len(decks)
                opp_arch, opp_deck = decks[opp_i]
            else:
                opp_variant_id = ""
                opp_arch, opp_deck = arch, deck
            self_jobs.append(
                {
                    **common,
                    "opponent_checkpoint": opp_ckpt,
                    "opponent_checkpoint_digest": opp_digest,
                    "opponent_id": f"self:{Path(opp_ckpt).name}",
                    "opp_deck": list(opp_deck),
                    "opp_archetype": opp_arch,
                    "collect_both_seats": bool(str(opp_digest) == str(digest)),
                    "target_provenance": {
                        "pure_rl": True,
                        "soft_policy_targets": False,
                        "collect": "self_play",
                        "self_play": True,
                        "collection_job_index": int(game_i),
                        "opponent_id": f"self:{Path(opp_ckpt).name}",
                        "opponent_archetype_id": str(opp_arch),
                        "behavior_checkpoint": str(ckpt),
                        "behavior_checkpoint_digest": str(digest),
                        "opponent_checkpoint": str(opp_ckpt),
                        "opponent_checkpoint_digest": str(opp_digest),
                        "mcts_sims": 0,
                        "action_temperature": float(collect_temperature),
                        **family_provenance(our_variant_id),
                        **(
                            {
                                "opponent_variant_id": opp_variant_id,
                                "opponent_ordered_digest": str(
                                    family_variants[opp_variant_id][
                                        "ordered_digest"
                                    ]
                                ),
                                "opponent_multiset_digest": str(
                                    family_variants[opp_variant_id][
                                        "multiset_digest"
                                    ]
                                ),
                            }
                            if opp_variant_id
                            else {}
                        ),
                        **(
                            {
                                "deck_mix": {
                                    **ladder_wave_provenance,
                                    "scheduled_our_deck_id": our_variant_id or arch,
                                    "scheduled_opp_deck_id": opp_variant_id or opp_arch,
                                    "stream": "self_play",
                                }
                            }
                            if ladder_mix is not None
                            else {}
                        ),
                    },
                }
            )
        else:
            if balanced_eval:
                spec = specs[(game_i // 2) % len(specs)]
                opponent_training_group = "formal_eval"
            elif public_spec_schedule:
                spec, opponent_training_group = public_spec_schedule[
                    game_i - n_self
                ]
            else:
                spec = specs[(game_i - n_self) % len(specs)]
                opponent_training_group = "diverse_public"
            if not balanced_eval:
                seat_key = (str(arch), str(spec.id))
                occurrence = int(public_seat_counts.get(seat_key, 0))
                common["our_seat"] = _public_training_seat(
                    seed=int(seed),
                    iteration=int(iteration),
                    archetype=str(arch),
                    opponent_id=str(spec.id),
                    occurrence=occurrence,
                )
                public_seat_counts[seat_key] = occurrence + 1
                is_priority = bool(opponent_training_group == str(priority_group))
                sharpened = bool(
                    is_priority
                    and str(spec.id) in exploit_opponents
                    and _paired_official_exploit(
                        seed=int(seed),
                        iteration=int(iteration),
                        archetype=str(arch),
                        opponent_id=str(spec.id),
                        occurrence=occurrence,
                        fraction=float(official_exploit_frac),
                    )
                )
            else:
                is_priority = False
                sharpened = False
            behavior_temperature = (
                float(priority_temperature)
                if is_priority and priority_temperature is not None
                else (
                    float(official_exploit_temperature)
                    if sharpened
                    else float(collect_temperature)
                )
            )
            common["action_temperature"] = behavior_temperature
            collect_kind = (
                "formal_eval"
                if balanced_eval
                else str(priority_group)
                if is_priority
                else "public_mix"
            )
            opponent_archetype = str(
                (priority_archetypes or {}).get(str(spec.id))
                or ""
            )
            base_jobs.append(
                {
                    **common,
                    "spec": dict(spec_payloads[str(spec.id)]),
                    "require_portable_baseline_contract": True,
                    "opponent_id": spec.id,
                    **(
                        {"opp_archetype": opponent_archetype}
                        if opponent_archetype
                        else {}
                    ),
                    "target_provenance": {
                        "pure_rl": True,
                        "soft_policy_targets": False,
                        "collect": collect_kind,
                        "opponent_training_group": opponent_training_group,
                        "opponent_sampling_weight": (
                            float(priority_weights[str(spec.id)])
                            if is_priority
                            and priority_weights is not None
                            else None
                        ),
                        **(
                            {
                                "strong_public_practice_floor_games": int(
                                    priority_minimum_games[str(spec.id)]
                                )
                            }
                            if is_priority
                            and str(priority_group)
                            == STRONG_PUBLIC_PRACTICE_GROUP
                            and priority_minimum_games is not None
                            else {}
                        ),
                        "opponent_schedule": (
                            "adaptive_exact_gate_gap_tier_weighted_v1"
                            if is_priority
                            and str(priority_group)
                            == STRONG_PUBLIC_PRACTICE_GROUP
                            and priority_weights is not None
                            else "adaptive_exact_heldout_gap_v1"
                            if is_priority and priority_weights is not None
                            else "uniform_round_robin_v1"
                        ),
                        "self_play": False,
                        "collection_job_index": int(game_i),
                        "behavior_checkpoint": str(ckpt),
                        "behavior_checkpoint_digest": str(digest),
                        "mcts_sims": 0,
                        "action_temperature": behavior_temperature,
                        **family_provenance(our_variant_id),
                        "behavior_mode": (
                            "strong_public_sampled_practice_v1"
                            if is_priority and priority_temperature is not None
                            else (
                                "official_exploit_sharpened_v1"
                                if sharpened
                                else "base_sampling_temperature_v1"
                            )
                        ),
                        **(
                            dict(priority_context or {})
                            if is_priority
                            else {}
                        ),
                        # Identity and gate-safety fields are canonical and
                        # cannot be weakened by optional group context.
                        "opponent_id": str(spec.id),
                        "opponent_archetype_id": opponent_archetype,
                        "opponent_content_digest": str(
                            spec_payloads[str(spec.id)]["content_digest"]
                        ),
                        "formal_eval": bool(balanced_eval),
                        "seat_schedule": (
                            "formal_eval_paired_v1"
                            if balanced_eval
                            else "per_opponent_archetype_alternating_v1"
                        ),
                        **(
                            {
                                "deck_mix": {
                                    **ladder_wave_provenance,
                                    "scheduled_our_deck_id": our_variant_id or arch,
                                    "stream": "public_our",
                                }
                            }
                            if ladder_mix is not None
                            else {}
                        ),
                    },
                }
            )
    if exact_training_seat_split:
        rows = sorted(
            [*self_jobs, *base_jobs], key=lambda row: int(row["job_index"])
        )
        if len(rows) % 2:
            raise ValueError("exact training-seat scheduling requires an even game count")
        seat0 = sum(int(row.get("our_seat", -1)) == 0 for row in rows)
        seat1 = sum(int(row.get("our_seat", -1)) == 1 for row in rows)
        majority = 0 if seat0 > seat1 else 1 if seat1 > seat0 else None
        flips_needed = abs(seat0 - seat1) // 2
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            provenance = dict(row.get("target_provenance") or {})
            key = (
                str(provenance.get("opponent_training_group") or "self_play"),
                str(row.get("archetype") or ""),
                str(row.get("opponent_id") or ""),
            )
            grouped.setdefault(key, []).append(row)
        candidates: list[tuple[tuple[str, str, str], int, dict[str, Any]]] = []
        if majority is not None:
            for key, group_rows in grouped.items():
                group_majority = sum(
                    int(row.get("our_seat", -1)) == majority for row in group_rows
                )
                group_minority = len(group_rows) - group_majority
                if group_majority == group_minority + 1:
                    row = max(
                        (
                            row
                            for row in group_rows
                            if int(row.get("our_seat", -1)) == majority
                        ),
                        key=lambda item: int(item["job_index"]),
                    )
                    candidates.append((key, int(row["job_index"]), row))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < flips_needed:
            raise RuntimeError(
                "cannot make the training-seat schedule globally exact without "
                "breaking per-opponent balance"
            )
        for _key, _job_index, row in candidates[:flips_needed]:
            row["our_seat"] = 1 - int(row["our_seat"])
        if sum(int(row.get("our_seat", -1)) == 0 for row in rows) != len(rows) // 2:
            raise RuntimeError("exact training-seat scheduler failed to reach parity")
        for row in rows:
            provenance = row.get("target_provenance")
            if isinstance(provenance, dict):
                provenance["seat_schedule"] = (
                    "per_opponent_balanced_global_exact_v1"
                )
    return self_jobs, base_jobs


def _self_play_refill_capacity_jobs(
    jobs: list[dict[str, Any]],
    *,
    fraction: float,
    first_job_index: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Build bounded spare self-play jobs with disjoint seeds and identities.

    Self-play can legitimately terminate without a replay record (for example,
    when one policy cannot produce a legal action).  Those outcomes must not
    make the whole unattended iteration restart after its public games have
    already completed.  Spare jobs are scheduled in the same wave and only
    provide replacement capacity for missing records; the configured
    ``games_per_iter`` remains the retention target.
    """
    if not jobs:
        return []
    frac = max(0.0, min(float(fraction), 1.0))
    count = int(math.ceil(len(jobs) * frac))
    if count <= 0:
        return []
    next_index = (
        int(first_job_index)
        if first_job_index is not None
        else max(int(job.get("job_index", -1)) for job in jobs) + 1
    )
    out: list[dict[str, Any]] = []
    for offset in range(count):
        source = jobs[offset % len(jobs)]
        source_index = int(source.get("job_index", offset % len(jobs)))
        provenance = dict(source.get("target_provenance") or {})
        out.append(
            {
                **source,
                "job_index": next_index + offset,
                # Stay inside this iteration's 100k seed namespace while
                # avoiding every primary job seed.
                "seed": int(source.get("seed", 0)) + 50_000,
                "target_provenance": {
                    **provenance,
                    "replacement_capacity": True,
                    "replacement_for_job_index": source_index,
                },
            }
        )
    return out


def _append_replacement_spool(
    path: Path,
    *,
    replacement_for_job_index: int,
    records: list[dict[str, Any]],
    runtime_audit_row: dict[str, Any],
    schedule_contract: dict[str, Any],
) -> None:
    """Durably stage a successful spare result outside the training shard.

    Replacement-capacity games are simulation attempts, not additional
    training games.  They may enter the canonical shard only when the matching
    primary source game failed to produce a usable record.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "poke_bot.collection_replacement_spool/v2",
        "replacement_for_job_index": int(replacement_for_job_index),
        "records": records,
        # A successful spare is not part of the canonical audit population
        # unless it replaces a failed primary schedule cell.  Keep its audit
        # proof beside the staged record so promotion can add both atomically.
        "runtime_audit_row": dict(runtime_audit_row),
        "schedule_contract": dict(schedule_contract),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _promote_replacement_spool(
    path: Path,
    *,
    missing_job_indices: set[int],
    writer: CompactShardWriter,
    replay_cache: Optional[StreamingReplayCache] = None,
    primary_jobs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Promote one contract-equivalent successful spare per missing primary.

    A spare's originally paired source cell is preferred.  If that cell did
    not fail, the spare may fill another missing cell only when the complete
    seat/opponent/archetype/training-group contract is identical.  This keeps
    the exact schedule distribution while avoiding deterministic retry loops
    for game states that never yield a record.
    """

    path = Path(path)
    missing = {int(index) for index in missing_job_indices}
    promoted: set[int] = set()
    trajectories = 0
    decisions = 0
    runtime_audit_rows: list[dict[str, Any]] = []
    job_contracts = {
        int(job.get("job_index", -1)): _replacement_schedule_contract_from_job(job)
        for job in (primary_jobs or [])
    }
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("schema") != "poke_bot.collection_replacement_spool/v2":
                    raise RuntimeError("replacement spool schema mismatch")
                source_index = int(payload.get("replacement_for_job_index", -1))
                payload_contract = dict(payload.get("schedule_contract") or {})
                if not payload_contract:
                    raise RuntimeError("replacement spool lost schedule contract")
                raw_records = list(payload.get("records") or [])
                rescheduled_inapplicable = any(
                    bool(
                        (dict(record).get("target_provenance") or {}).get(
                            "replacement_rescheduled_inapplicable_opponent"
                        )
                    )
                    for record in raw_records
                )
                target_index: Optional[int] = None
                if source_index in missing and source_index not in promoted:
                    expected = job_contracts.get(source_index)
                    if (
                        expected is None
                        or expected == payload_contract
                        or rescheduled_inapplicable
                    ):
                        target_index = source_index
                if target_index is None and job_contracts:
                    target_index = next(
                        (
                            index
                            for index in sorted(missing - promoted)
                            if job_contracts.get(index) == payload_contract
                        ),
                        None,
                    )
                if target_index is None:
                    continue
                rewritten_records: list[dict[str, Any]] = []
                for raw_record in raw_records:
                    record = dict(raw_record)
                    provenance = dict(record.get("target_provenance") or {})
                    provenance["replacement_original_for_job_index"] = source_index
                    provenance["replacement_for_job_index"] = int(target_index)
                    # The promoted bytes now occupy the missing canonical
                    # schedule cell.  Persist that identity in the same field
                    # used by shard receipts/resume audits; retaining the
                    # spare attempt's job index here can make an in-memory
                    # exact-count pass seal a durably short shard.
                    provenance["collection_job_index"] = int(target_index)
                    record["target_provenance"] = provenance
                    rewritten_records.append(record)
                games = [
                    game
                    for game in (
                        _record_to_compact_game(dict(record))
                        for record in rewritten_records
                    )
                    if game is not None
                ]
                if not games:
                    continue
                for game in games:
                    flushed = writer.write_game(game)
                    if replay_cache is not None and flushed:
                        replay_cache.note_append()
                    trajectories += 1
                    decisions += len(game.decisions)
                audit_row = dict(payload.get("runtime_audit_row") or {})
                if not audit_row:
                    raise RuntimeError(
                        "replacement spool lost canonical runtime audit proof"
                    )
                audit_row["replacement_attempt_job_index"] = audit_row.get(
                    "job_index"
                )
                audit_row["replacement_original_for_job_index"] = source_index
                audit_row["job_index"] = int(target_index)
                audit_row["promoted_replacement"] = True
                runtime_audit_rows.append(audit_row)
                promoted.add(int(target_index))
    path.unlink(missing_ok=True)
    return {
        "promoted_job_indices": promoted,
        "promoted_source_games": len(promoted),
        "promoted_trajectories": trajectories,
        "promoted_decisions": decisions,
        "promoted_runtime_audit_rows": runtime_audit_rows,
    }


def _replacement_schedule_contract_from_job(job: dict[str, Any]) -> dict[str, Any]:
    """Canonical distribution identity for an exact training schedule cell."""

    provenance = dict(job.get("target_provenance") or {})
    return {
        "our_seat": int(job.get("our_seat", -1)),
        "opponent_id": str(job.get("opponent_id") or ""),
        "archetype": str(job.get("archetype") or ""),
        "opp_archetype": str(job.get("opp_archetype") or ""),
        "opponent_checkpoint_digest": str(
            job.get("opponent_checkpoint_digest")
            or provenance.get("opponent_checkpoint_digest")
            or ""
        ),
        "opponent_content_digest": str(
            provenance.get("opponent_content_digest") or ""
        ),
        "opponent_training_group": str(
            provenance.get("opponent_training_group") or ""
        ),
    }


def _replacement_schedule_contract_from_result(
    runtime_audit_row: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("replacement result lacks a schedule record")
    record = dict(records[0])
    provenance = dict(record.get("target_provenance") or {})
    return {
        "our_seat": int(
            runtime_audit_row.get("our_seat")
            if runtime_audit_row.get("our_seat") is not None
            else -1
        ),
        "opponent_id": str(runtime_audit_row.get("opponent_id") or ""),
        "archetype": str(record.get("archetype") or runtime_audit_row.get("archetype") or ""),
        "opp_archetype": str(record.get("opp_archetype") or ""),
        "opponent_checkpoint_digest": str(
            provenance.get("opponent_checkpoint_digest") or ""
        ),
        "opponent_content_digest": str(
            provenance.get("opponent_content_digest") or ""
        ),
        "opponent_training_group": str(
            provenance.get("opponent_training_group") or ""
        ),
    }


_PUBLIC_MIX_TARGETED_REPLACEMENT_ROUNDS = 32


def _targeted_replacement_seed_offset(retry_round: int) -> int:
    """Return a deterministic seed lane disjoint from ordinary collections.

    The first four lanes preserve the historical 60k/70k/80k/90k identity.
    Later public-mix recovery lanes live in a high, bounded namespace so a
    one-game compaction failure cannot force an otherwise exact 8,192-game
    iteration to be recollected.  The high namespace remains below signed
    32-bit seed limits and is never used by ordinary iteration seeds.
    """

    if retry_round < 0 or retry_round >= _PUBLIC_MIX_TARGETED_REPLACEMENT_ROUNDS:
        raise ValueError(
            "targeted replacement retry_round must be in "
            f"[0, {_PUBLIC_MIX_TARGETED_REPLACEMENT_ROUNDS - 1}]"
        )
    if retry_round < 4:
        return 60_000 + (10_000 * int(retry_round))
    return 1_000_000_000 + (10_000 * (int(retry_round) - 4))


def _targeted_replacement_jobs(
    primary_jobs: list[dict[str, Any]],
    *,
    missing_job_indices: set[int],
    retry_round: int,
    first_job_index: int,
) -> list[dict[str, Any]]:
    """Reissue missing cells from an equivalent, non-pathological source.

    Retrying the identical source job can deterministically reproduce the same
    no-record game forever.  A different primary job with the exact same
    distribution contract is therefore preferred; the replacement remains
    assigned to the original missing cell and uses a disjoint seed.
    """

    seed_offset = _targeted_replacement_seed_offset(retry_round)
    by_index = {int(job.get("job_index", -1)): job for job in primary_jobs}
    by_contract: dict[str, list[dict[str, Any]]] = {}
    for job in primary_jobs:
        key = _canonical_digest(_replacement_schedule_contract_from_job(job))
        by_contract.setdefault(key, []).append(job)
    for bucket in by_contract.values():
        bucket.sort(key=lambda job: int(job.get("job_index", -1)))
    missing = sorted(int(index) for index in missing_job_indices)
    absent = [index for index in missing if index not in by_index]
    if absent:
        raise RuntimeError(
            f"missing replacement source jobs: {absent[:16]}"
        )
    out: list[dict[str, Any]] = []
    for offset, source_index in enumerate(missing):
        target = by_index[source_index]
        contract_key = _canonical_digest(
            _replacement_schedule_contract_from_job(target)
        )
        candidates = by_contract.get(contract_key) or [target]
        source_pos = (source_index + retry_round + 1) % len(candidates)
        source = candidates[source_pos]
        if (
            len(candidates) > 1
            and int(source.get("job_index", -1)) == source_index
        ):
            source = candidates[(source_pos + 1) % len(candidates)]
        provenance = dict(source.get("target_provenance") or {})
        reschedule_to_current_mirror = bool(
            retry_round >= 1
            and provenance.get("self_play")
            and target.get("checkpoint")
            and target.get("checkpoint_digest")
        )
        original_opponent = {
            "opponent_id": str(target.get("opponent_id") or ""),
            "opponent_checkpoint": str(target.get("opponent_checkpoint") or ""),
            "opponent_checkpoint_digest": str(
                target.get("opponent_checkpoint_digest") or ""
            ),
        }
        effective = dict(source)
        if reschedule_to_current_mirror:
            current_checkpoint = str(target["checkpoint"])
            current_digest = str(target["checkpoint_digest"])
            effective.update(
                {
                    "checkpoint": current_checkpoint,
                    "checkpoint_digest": current_digest,
                    "opponent_checkpoint": current_checkpoint,
                    "opponent_checkpoint_digest": current_digest,
                    "opponent_id": f"self:{Path(current_checkpoint).name}",
                    "collect_both_seats": True,
                }
            )
        out.append(
            {
                **effective,
                "job_index": int(first_job_index) + offset,
                "seed": int(source.get("seed", 0)) + seed_offset,
                "target_provenance": {
                    **provenance,
                    "collection_job_index": source_index,
                    "replacement_capacity": True,
                    "replacement_for_job_index": source_index,
                    "replacement_retry_source_job_index": int(
                        source.get("job_index", -1)
                    ),
                    "replacement_round": int(retry_round) + 1,
                    "replacement_rescheduled_inapplicable_opponent": (
                        reschedule_to_current_mirror
                    ),
                    **(
                        {
                            "replacement_original_opponent": original_opponent,
                            "opponent_id": str(effective["opponent_id"]),
                            "opponent_checkpoint": str(
                                effective["opponent_checkpoint"]
                            ),
                            "opponent_checkpoint_digest": str(
                                effective["opponent_checkpoint_digest"]
                            ),
                        }
                        if reschedule_to_current_mirror
                        else {}
                    ),
                },
            }
        )
    return out


def _dataset_from_replay_window(
    run_dir: Path,
    it: int,
    *,
    initial_replay_shards: Optional[list[Path]] = None,
    tactical_overlay_dir: Optional[Path] = None,
) -> Any:
    """Fresh-data bias (Spinning Up on-policy theme): short window only.

    ``bootstrap_mix`` must stay 0 — refuse soft CE / starter replay contamination.
    """
    from poke_bot.dataset import BootstrapDataset

    mix = float(getattr(config.PURE_RL, "bootstrap_mix", 0.0))
    if mix > 0.0:
        raise SystemExit(
            f"PURE_RL fail-closed: bootstrap_mix={mix} > 0 "
            "(Spinning Up / Abhyuday: no starter CE clone; fresh AWR data only)"
        )
    window = max(1, int(getattr(config.PURE_RL, "replay_window_shards", 2)))
    seqs = []
    # A clean lineage handoff may carry the immediately preceding immutable
    # shard without copying 3+ GiB or pretending it belongs to local iter 0.
    # It participates only in the first update; iter 1 naturally has two local
    # shards and restores the ordinary rolling-window rule.
    if int(it) == 0 and window > 1:
        for shard in list(initial_replay_shards or [])[-(window - 1) :]:
            ds = dataset_from_shard(
                Path(shard),
                verify_info_set=False,
                max_context=pure_rl_model_config().max_context,
            )
            seqs.extend(list(ds.sequences))
    for j in range(max(0, it - window + 1), it + 1):
        shard = run_dir / "shards" / f"iter_{j:05d}.jsonl"
        if not shard.is_file():
            continue
        ds = dataset_from_shard(
            shard,
            verify_info_set=False,
            max_context=pure_rl_model_config().max_context,
        )
        if tactical_overlay_dir is not None:
            overlay = Path(tactical_overlay_dir) / f"iter_{j:05d}.json"
            if overlay.is_file():
                from poke_bot.tactical_sequence_materialization import (
                    attach_tactical_target_overlay,
                )

                attach_tactical_target_overlay(ds.sequences, overlay)
        seqs.extend(list(ds.sequences))
    return BootstrapDataset(sequences=seqs)


def _replay_window_paths(
    run_dir: Path,
    it: int,
    *,
    initial_replay_shards: Optional[list[Path]] = None,
) -> list[Path]:
    """Return the exact ordered raw shards consumed by the replay loader."""

    window = max(1, int(getattr(config.PURE_RL, "replay_window_shards", 2)))
    paths: list[Path] = []
    if int(it) == 0 and window > 1:
        paths.extend(
            Path(path).expanduser().resolve()
            for path in list(initial_replay_shards or [])[-(window - 1) :]
        )
    for index in range(max(0, it - window + 1), it + 1):
        path = (run_dir / "shards" / f"iter_{index:05d}.jsonl").resolve()
        if path.is_file():
            paths.append(path)
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("exact replay-window shard inventory is incomplete")
    return paths


def _macro_resample_family_sequences(
    dataset: Any,
    *,
    manifest: Mapping[str, Any],
    checksum_seed: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply deterministic equal-cluster family weights to the replay window.

    Rows produced before family activation are admitted only when their exact
    60-card multiset is the singular package variant.  Current family rows must
    carry the checksum-bound variant provenance emitted by collection.  Dev and
    locked variants fail closed through ``validate_training_row``.
    """
    from collections import Counter, defaultdict

    from poke_bot.specialist_archetype_family import (
        family_probabilities,
        hamilton_quotas,
        multiset_digest,
        validate_manifest,
        validate_training_row,
    )

    validated = validate_manifest(manifest, require_activation_ready=True)
    source = list(dataset.sequences)
    if not source:
        raise RuntimeError("family replay weighting received no sequences")
    buckets: dict[str, list[Any]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    legacy_package_rows = 0
    for sequence in source:
        raw = dict(getattr(sequence, "target_provenance", {}) or {})
        had_variant = bool(raw.get("variant_id"))
        row = {
            **raw,
            "multiset_digest": str(
                raw.get("multiset_digest")
                or multiset_digest(list(getattr(sequence, "deck", ()) or ()))
            ),
        }
        normalized = validate_training_row(validated, row)
        variant_id = str(normalized["variant_id"])
        sequence = replace(
            sequence,
            target_provenance={
                **raw,
                **normalized,
                "family_macro_replay": True,
            },
        )
        buckets[variant_id].append(sequence)
        source_counts[variant_id] += 1
        legacy_package_rows += int(not had_variant)

    probabilities = family_probabilities(validated)
    quotas = hamilton_quotas(len(source), probabilities)
    selected: list[Any] = []
    duplicate_draws = 0
    for variant_id, quota in sorted(quotas.items()):
        bucket = list(buckets.get(variant_id) or ())
        if quota and not bucket:
            raise RuntimeError(
                "family macro replay lacks an admitted train variant: "
                f"{variant_id}"
            )
        bucket.sort(
            key=lambda sequence: hashlib.sha256(
                (
                    f"{checksum_seed}|{variant_id}|"
                    f"{getattr(sequence, 'episode_id', '')}|"
                    f"{getattr(sequence, 'seat', -1)}|"
                    f"{getattr(sequence, 'source', '')}"
                ).encode("utf-8")
            ).digest()
        )
        duplicate_draws += max(0, int(quota) - len(bucket))
        for index in range(int(quota)):
            selected.append(bucket[index % len(bucket)])
    selected.sort(
        key=lambda sequence: hashlib.sha256(
            (
                f"{checksum_seed}|final|"
                f"{getattr(sequence, 'episode_id', '')}|"
                f"{getattr(sequence, 'seat', -1)}|"
                f"{(getattr(sequence, 'target_provenance', {}) or {}).get('variant_id', '')}"
            ).encode("utf-8")
        ).digest()
    )
    selected_counts = Counter(
        str(sequence.target_provenance["variant_id"])
        for sequence in selected
    )
    if dict(selected_counts) != {key: int(value) for key, value in quotas.items()}:
        raise RuntimeError("family macro replay quota realization changed")
    receipt = {
        "schema": "poke_bot.archetype_family_replay_weighting/v1",
        "family_id": str(validated["family_id"]),
        "manifest_digest": str(validated["artifact_sha256"]),
        "checksum_seed": str(checksum_seed),
        "source_sequences": len(source),
        "selected_sequences": len(selected),
        "legacy_package_sequences": int(legacy_package_rows),
        "source_variant_counts": dict(sorted(source_counts.items())),
        "selected_variant_counts": dict(sorted(selected_counts.items())),
        "target_probabilities": dict(sorted(probabilities.items())),
        "duplicate_draws": int(duplicate_draws),
        "development_or_locked_rows": 0,
        "passed": True,
    }
    return BootstrapDataset(sequences=selected), receipt


def _consume_results(
    results_iter,
    writer: CompactShardWriter,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    progress: Optional[_TqdmProgress] = None,
    live_wr_gate: Optional[tuple[float, int]] = None,
    replay_cache: Optional[StreamingReplayCache] = None,
    required_checkpoint_digest: Optional[str] = None,
    live_wr_opponent_ids: Optional[tuple[str, ...]] = None,
    practice_record_contracts: Optional[dict[int, dict[str, str]]] = None,
    practice_seen_indices: Optional[set[int]] = None,
    practice_successful_indices: Optional[set[int]] = None,
    practice_written_indices: Optional[set[int]] = None,
    replacement_spool: Optional[Path] = None,
    retained_job_indices: Optional[set[int]] = None,
    collection_job_contracts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> None:
    """``live_wr_gate=(target_wr, min_games)`` streams a running WR onto the
    bar as heldout games land (official baselines only) — passed only from
    ``_heldout_eval`` so regular collect waves don't mislabel a practice
    mix's partial WR as the gate signal.
    """
    wr_wins = 0.0
    wr_games = 0
    for res in results_iter:
        practice_contract: Optional[dict[str, str]] = None
        practice_job_index: Optional[int] = None
        if practice_record_contracts:
            try:
                practice_job_index = int(res.get("job_index"))
            except (TypeError, ValueError):
                practice_job_index = None
            if practice_job_index in practice_record_contracts:
                practice_contract = practice_record_contracts[practice_job_index]
                if practice_seen_indices is None:
                    raise RuntimeError("practice receipt tracker is missing")
                if practice_job_index in practice_seen_indices:
                    raise RuntimeError(
                        "duplicate strong-public practice result: "
                        f"job_index={practice_job_index}"
                    )
                practice_seen_indices.add(practice_job_index)
                expected_opponent = practice_contract["opponent_id"]
                if str(res.get("opponent_id") or "") != expected_opponent:
                    raise RuntimeError(
                        "strong-public practice result identity mismatch: "
                        f"job_index={practice_job_index} "
                        f"actual={res.get('opponent_id')} "
                        f"expected={expected_opponent}"
                    )
                if int(res.get("our_seat", -1)) != int(
                    practice_contract["our_seat"]
                ):
                    raise RuntimeError(
                        "strong-public practice result seat mismatch: "
                        f"job_index={practice_job_index}"
                    )
        heldout_contract_invalid = bool(
            required_checkpoint_digest is not None
            and (
                str(res.get("checkpoint_digest") or "")
                != str(required_checkpoint_digest)
                or str(res.get("action_selection") or "") != "greedy"
            )
        )
        runtime_audit_row = {
                "opponent_id": res.get("opponent_id"),
                "our_seat": res.get("our_seat"),
                "winner": res.get("winner"),
                "baseline_failed": bool(res.get("baseline_failed")),
                "our_failed": bool(res.get("our_failed")),
                "self_play": bool(res.get("self_play")),
                "leaf_self_play_mode": res.get("leaf_self_play_mode"),
                "leaf_remote": bool(res.get("leaf_remote")),
                "multi_env": bool(res.get("multi_env")),
                "job_index": res.get("job_index"),
                "archetype": res.get("archetype"),
                "checkpoint_digest": res.get("checkpoint_digest"),
                "action_selection": res.get("action_selection"),
                "matchup_runtime_audit": res.get("matchup_runtime_audit"),
                "opponent_matchup_runtime_audit": res.get(
                    "opponent_matchup_runtime_audit"
                ),
                "policy_terminal_failure": bool(
                    res.get("policy_terminal_failure")
                ),
                "stall_terminated": bool(res.get("stall_terminated")),
                "stall_turns": int(res.get("stall_turns") or 0),
                "training_return_override": res.get(
                    "training_return_override"
                ),
                "failed_seat": res.get("failed_seat"),
                "error": res.get("error"),
                "invalid": bool(
                    res.get("our_failed")
                    or res.get("resource_error")
                    or res.get("cancelled")
                    or res.get("trust_failure")
                    or res.get("game_timeout")
                    or heldout_contract_invalid
                ),
                "heldout_contract_invalid": heldout_contract_invalid,
            }
        if collection_job_contracts is not None:
            try:
                collection_job_index = int(res.get("job_index"))
            except (TypeError, ValueError):
                collection_job_index = -1
            collection_contract = dict(
                collection_job_contracts.get(collection_job_index) or {}
            )
            if not collection_contract:
                raise RuntimeError(
                    "canonical collection result lacks its immutable job contract: "
                    f"job_index={collection_job_index}"
                )
            runtime_audit_row.update(collection_contract)
        canonical_collection = retained_job_indices is not None
        # Evaluation rows describe every attempted evaluation game.  Training
        # rows instead describe only records retained in the canonical shard;
        # failed/malformed primaries and unused reserve capacity are excluded.
        if not canonical_collection:
            rows.append(runtime_audit_row)
        if live_wr_gate is not None and progress is not None:
            opp = str(res.get("opponent_id") or "")
            counted_ids = set(live_wr_opponent_ids or OFFICIAL_BASELINE_IDS)
            counted = bool(
                opp in counted_ids
                and not res.get("baseline_failed")
                and not res.get("our_failed")
                and not res.get("resource_error")
                and not res.get("cancelled")
                and not res.get("trust_failure")
                and not res.get("game_timeout")
                and not heldout_contract_invalid
                and res.get("winner") is not None
            )
            if counted:
                winner = int(res["winner"])
                our_seat = int(res.get("our_seat") or 0)
                wr_games += 1
                if winner == 2:
                    wr_wins += 0.5
                elif winner == our_seat:
                    wr_wins += 1.0
                target_wr, _min_games = live_wr_gate
                progress.set_wr(
                    (wr_wins / wr_games) if wr_games else 0.0,
                    wr_games,
                    target=target_wr,
                )
        mode = str(res.get("leaf_self_play_mode") or "")
        if mode:
            leaf_modes = stats.setdefault("leaf_modes", {})
            leaf_modes[mode] = int(leaf_modes.get(mode, 0)) + 1
        if res.get("leaf_remote"):
            stats["leaf_remote"] = int(stats.get("leaf_remote", 0)) + 1
        if res.get("multi_env"):
            stats["multi_env_games"] = int(stats.get("multi_env_games", 0)) + 1
        if res.get("baseline_failed"):
            stats["baseline_failed"] += 1
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        if res.get("our_failed") or res.get("resource_error") or res.get("cancelled"):
            if res.get("our_failed"):
                stats["our_failed"] += 1
            if res.get("resource_error"):
                stats["resource_error"] += 1
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        stats["ok"] += 1
        if practice_contract is not None:
            if practice_successful_indices is None:
                raise RuntimeError("practice success tracker is missing")
            practice_successful_indices.add(int(practice_job_index))
        if res.get("self_play"):
            stats["self_play"] = int(stats.get("self_play", 0)) + 1
        encoded_records = list(res.get("record_jsons") or [])
        if not encoded_records and res.get("record_json"):
            encoded_records = [res["record_json"]]
        records = []
        for encoded in encoded_records:
            try:
                parsed = json.loads(encoded) if isinstance(encoded, str) else encoded
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                records.append(parsed)
        if practice_contract is not None and not records:
            # A completed simulation without a usable trajectory is a missing
            # schedule cell, not an iteration-fatal schema event. Leave it
            # unretained so the exact per-cell replacement loop can repair it.
            if practice_successful_indices is not None:
                practice_successful_indices.discard(int(practice_job_index))
            stats["strong_public_practice_recordless_results"] = int(
                stats.get("strong_public_practice_recordless_results", 0)
            ) + 1
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        if practice_contract is not None and len(records) != 1:
            raise RuntimeError(
                "strong-public practice result must contain exactly one record: "
                f"job_index={practice_job_index} records={len(records)}"
            )
        if not records:
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        written = 0
        compact_records: list[tuple[dict[str, Any], CompactGame]] = []
        for record in records:
            if practice_contract is not None:
                expected_opponent = practice_contract["opponent_id"]
                expected_archetype = practice_contract["opponent_archetype_id"]
                prior_provenance = dict(record.get("target_provenance") or {})
                repaired = bool(
                    str(record.get("opp_archetype") or "") != expected_archetype
                    or str(prior_provenance.get("opponent_id") or "")
                    != expected_opponent
                    or str(
                        prior_provenance.get("opponent_archetype_id") or ""
                    )
                    != expected_archetype
                )
                record = {
                    **record,
                    "opp_archetype": expected_archetype,
                    "target_provenance": {
                        **prior_provenance,
                        "opponent_id": expected_opponent,
                        "opponent_archetype_id": expected_archetype,
                        "opponent_training_group": STRONG_PUBLIC_PRACTICE_GROUP,
                        "active_gate_id": practice_contract["active_gate_id"],
                    },
                }
                if repaired:
                    stats["strong_public_practice_records_repaired"] = int(
                        stats.get("strong_public_practice_records_repaired", 0)
                    ) + 1
            game = _record_to_compact_game(record)
            if game is None:
                if practice_contract is not None:
                    raise RuntimeError(
                        "strong-public practice record failed compaction: "
                        f"job_index={practice_job_index}"
                    )
                continue
            if practice_contract is not None:
                if (
                    game.opp_archetype
                    != practice_contract["opponent_archetype_id"]
                    or str(game.target_provenance.get("opponent_id") or "")
                    != practice_contract["opponent_id"]
                    or str(
                        game.target_provenance.get("opponent_archetype_id") or ""
                    )
                    != practice_contract["opponent_archetype_id"]
                ):
                    raise RuntimeError(
                        "strong-public practice compact record identity mismatch: "
                        f"job_index={practice_job_index}"
                    )
            compact_records.append((record, game))
        replacement_flags = {
            bool(game.target_provenance.get("replacement_capacity"))
            for _record, game in compact_records
        }
        if len(replacement_flags) > 1:
            raise RuntimeError("one source result mixed primary and spare records")
        is_replacement = replacement_flags == {True}
        if is_replacement:
            if replacement_spool is None:
                raise RuntimeError(
                    "replacement-capacity result reached a training writer "
                    "without an isolated replacement spool"
                )
            source_indices = {
                int(
                    game.target_provenance.get(
                        "replacement_for_job_index", -1
                    )
                )
                for _record, game in compact_records
            }
            if len(source_indices) != 1 or next(iter(source_indices), -1) < 0:
                raise RuntimeError("replacement result lost its primary job identity")
            _append_replacement_spool(
                replacement_spool,
                replacement_for_job_index=next(iter(source_indices)),
                records=[record for record, _game in compact_records],
                runtime_audit_row=runtime_audit_row,
                schedule_contract=_replacement_schedule_contract_from_result(
                    runtime_audit_row,
                    [record for record, _game in compact_records],
                ),
            )
            written = len(compact_records)
            stats["replacement_capacity_staged_source_games"] = int(
                stats.get("replacement_capacity_staged_source_games", 0)
            ) + 1
            stats["replacement_capacity_staged_trajectories"] = int(
                stats.get("replacement_capacity_staged_trajectories", 0)
            ) + written
        else:
            for _record, game in compact_records:
                flushed = writer.write_game(game)
                if replay_cache is not None and flushed:
                    # A buffered writer exposes only whole newline-terminated
                    # records.  Do not wake the prefix cache for bytes that
                    # are still deliberately private in its bounded RAM spool.
                    replay_cache.note_append()
                written += 1
        if practice_contract is not None:
            if written != 1 or practice_written_indices is None:
                raise RuntimeError(
                    "strong-public practice write receipt mismatch: "
                    f"job_index={practice_job_index} written={written}"
                )
            if int(practice_job_index) in practice_written_indices:
                raise RuntimeError(
                    "duplicate strong-public practice write receipt: "
                    f"job_index={practice_job_index}"
                )
            practice_written_indices.add(int(practice_job_index))
        if written <= 0:
            if progress is not None:
                progress.tick(decisions=writer.n_decisions)
            continue
        if not is_replacement:
            if canonical_collection:
                rows.append(runtime_audit_row)
            stats["with_record"] += 1
            stats["trajectories_written"] = int(
                stats.get("trajectories_written", 0)
            ) + written
            if retained_job_indices is not None:
                retained_job_indices.add(int(res.get("job_index", -1)))
        if progress is not None:
            progress.tick(decisions=writer.n_decisions)


def _flatten_batch_results(results_iter):
    """Expand multi-env worker outputs (list[dict]) into a flat result stream."""
    for item in results_iter:
        if isinstance(item, list):
            for res in item:
                yield res
        else:
            yield item


class BetweenIterSyncError(RuntimeError):
    """Between-iter weight sync hard-gate failed (digest mismatch / incomplete)."""


def _hard_gate_publish_weights(
    *,
    leaf: "_LeafFarm",
    remote_farm: Any,
    ckpt: Path,
    digest: str,
    version: int,
    required_endpoints: list[str],
    reload_local: bool = True,
) -> dict[str, Any]:
    """Hard-gate local+remote weight publish before next-iter work continues.

    Fail-closed: every local leaf ack and every required remote reload/pin must
    report ``checkpoint_digest == digest``. Soft WARN-and-continue is forbidden
    at this boundary — workers must not dispatch on mixed digests.
    """
    from poke_bot.remote_jobs import (
        RemoteJobsError,
        cache_exact_resident_remote_checkpoint,
        parse_endpoint,
    )

    proof: dict[str, Any] = {
        "digest": digest,
        "version": int(version),
        "checkpoint": str(ckpt),
        "local_ok": False,
        "remote_ok": False,
        "remote_endpoints": [],
    }
    dig = str(digest)
    if not dig.startswith("sha256:"):
        raise BetweenIterSyncError(f"invalid publish digest: {dig!r}")
    runtime_required = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    expected_runtime_tree_digest: Optional[str] = None
    expected_runtime_roster: tuple[str, ...] = ()
    if runtime_required:
        runtime_tree_path = Path(
            os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
        ).expanduser()
        if not runtime_tree_path.is_file():
            raise BetweenIterSyncError(
                "activated matchup runtime lacks a readable canonical tree"
            )
        runtime_tree_payload = json.loads(
            runtime_tree_path.read_text(encoding="utf-8")
        )
        expected_runtime_tree_digest = _sha256_file(runtime_tree_path)
        expected_runtime_roster = tuple(
            sorted(
                str(value)
                for value in dict(
                    runtime_tree_payload.get("runtime_contract") or {}
                ).get("accepted_archetype_ids", ())
            )
        )
        if not expected_runtime_roster:
            raise BetweenIterSyncError(
                "activated matchup runtime canonical tree has no accepted routes"
            )

    # 1) Local GPU leaves — reload already hard-fails on bad/mismatched acks.
    if leaf.remote_channel is not None:
        already = (
            not reload_local
            and leaf.digest == dig
            and leaf.remote_channel.get("expected_digest") == dig
        )
        if already:
            proof["local_reload_skipped"] = True
        else:
            leaf.reload(ckpt, dig)
        if leaf.digest != dig:
            raise BetweenIterSyncError(
                f"local leaf digest not published: have={leaf.digest} want={dig}"
            )
        ch = leaf.remote_channel
        if ch.get("expected_digest") != dig:
            raise BetweenIterSyncError(
                "local remote_channel expected_digest not updated after reload"
            )
        ch["generation"] = int(version)
        ch["expected_version"] = int(leaf.version)
        proof["local_ok"] = True
        proof["local_version"] = int(leaf.version)
    else:
        proof["local_ok"] = True  # cpu leaf-eval path
        proof["local_skipped"] = True

    # 2) Remotes — require every configured endpoint; no soft-drop at boundary.
    req = [e.strip() for e in required_endpoints if str(e).strip()]
    if not req:
        proof["remote_ok"] = True
        proof["remote_skipped"] = True
        print(
            f"[pure_rl] BETWEEN_ITER_HARD_GATE ok local digest={dig[:19]}… "
            f"version={version} remotes=none",
            flush=True,
        )
        return proof
    if remote_farm is None:
        raise BetweenIterSyncError(
            "remote farm missing at between-iter hard-gate "
            f"(required={req})"
        )

    # Re-attach any soft-dropped clients before the strict sync.
    try:
        remote_farm._reconnect_missing()
    except Exception as exc:  # noqa: BLE001
        raise BetweenIterSyncError(
            f"remote reconnect before hard-gate failed: {exc}"
        ) from exc

    alive = {(c.host, int(c.port)): c for c in remote_farm.clients}
    missing: list[str] = []
    clients_ordered = []
    for ep in req:
        host, port = parse_endpoint(ep)
        key = (host, int(port))
        client = alive.get(key)
        if client is None:
            missing.append(ep)
        else:
            clients_ordered.append((ep, client))
    if missing:
        raise BetweenIterSyncError(
            "between-iter hard-gate: required remotes not connected: "
            + ", ".join(missing)
        )

    remote_proof: list[dict[str, Any]] = []
    errors: list[str] = []
    for ep, client in clients_ordered:
        try:
            # A farm client may still be present in ``clients`` after its
            # worker was replaced or recycled.  ``_reconnect_missing`` only
            # recreates absent entries, so explicitly refresh every control
            # socket at this strict boundary before issuing reload/pin.  This
            # is intentionally fail-closed: if a required endpoint cannot
            # reconnect, no next-iteration work may start on mixed weights.
            initial_info = client.reconnect()
            # A managed remote may have completed the exact reload/pin before
            # its prior controller socket disappeared (for example, during a
            # receipt-backed service rotation).  Reissuing the same control
            # transaction can then wait on that dead socket even though every
            # live leaf already advertises the required immutable identity.
            # Accept the idempotent state only from a fresh, complete health
            # proof; anything missing or mismatched retains the normal reload.
            exact_health = None
            health_call = getattr(client, "health", None)
            if callable(health_call):
                try:
                    candidate_health = health_call()
                    leaves = list(candidate_health.get("leaves") or [])
                    pinned = {
                        str(value)
                        for value in candidate_health.get("pinned_digests") or ()
                    }
                    if (
                        candidate_health.get("ok") is True
                        and candidate_health.get("controller_healthy") is True
                        and candidate_health.get("leaf_alive") is True
                        and candidate_health.get("leaf_identity_ok") is True
                        and candidate_health.get("checkpoint_digest") == dig
                        and dig in pinned
                        and leaves
                        and all(
                            row.get("healthy") is True
                            and row.get("checkpoint_digest") == dig
                            for row in leaves
                        )
                    ):
                        exact_health = candidate_health
                except Exception:
                    exact_health = None

            if exact_health is not None:
                try:
                    cache_exact_resident_remote_checkpoint(
                        str(getattr(client, "host", "")),
                        str(ckpt),
                        digest=dig,
                    )
                except RemoteJobsError:
                    # A health-exact worker with no matching staged object is
                    # not enough to prepare future jobs. Fall back to the
                    # ordinary verified staging/reload transaction.
                    exact_health = None

            if exact_health is not None:
                reload_reply = {
                    "ok": True,
                    "checkpoint_digest": dig,
                    "version": exact_health.get("checkpoint_version"),
                }
                pin_reply = {"ok": True, "checkpoint_digest": dig}
                reload_skipped_exact_health = True
            else:
                reload_reply = client.reload_checkpoint(
                    str(ckpt), digest=dig, version=int(version)
                )
                pin_reply = client.pin_checkpoint(str(ckpt), digest=dig)
                reload_skipped_exact_health = False
            got = reload_reply.get("checkpoint_digest")
            if not reload_reply.get("ok", False) or got != dig:
                raise RemoteJobsError(
                    f"reload digest mismatch on {ep}: reply={reload_reply!r}"
                )
            pin_got = pin_reply.get("checkpoint_digest")
            if not pin_reply.get("ok", False) or pin_got != dig:
                raise RemoteJobsError(
                    f"pin digest mismatch on {ep}: reply={pin_reply!r}"
                )
            # Re-hello confirms the worker's published primary digest.
            info = client.reconnect()
            hello_dig = getattr(info, "checkpoint_digest", None)
            if hello_dig != dig:
                raise RemoteJobsError(
                    f"hello digest mismatch on {ep}: expected {dig}, got {hello_dig}"
                )
            runtime = getattr(info, "matchup_runtime", None)
            runtime_probe: Optional[dict[str, Any]] = None
            if runtime_required:
                if not isinstance(runtime, dict):
                    raise RemoteJobsError(
                        f"matchup runtime is not active on {ep}; worker hello "
                        "did not advertise a digest-verified activation contract"
                    )
                runtime_checkpoint = str(runtime.get("checkpoint_digest") or "")
                accepted = tuple(
                    sorted(
                        str(value)
                        for value in runtime.get("accepted_archetype_ids") or ()
                    )
                )
                if not (
                    runtime_checkpoint == dig
                    and str(runtime.get("tree_digest") or "")
                    == expected_runtime_tree_digest
                    and accepted == expected_runtime_roster
                    and runtime.get("continuous_reevaluation") is True
                    and runtime.get("one_route_per_decision") is True
                    and runtime.get("unknown_route_exact_bypass") is True
                ):
                    raise RemoteJobsError(
                        f"matchup runtime contract mismatch on {ep}: {runtime!r}"
                    )
                runtime_probe_result = client.submit_job(
                    {},
                    kind="runtime_probe",
                )
                runtime_probe = dict(
                    runtime_probe_result.get("runtime_probe") or {}
                )
                probe_roster = tuple(
                    sorted(
                        str(value)
                        for value in runtime_probe.get(
                            "accepted_archetype_ids"
                        )
                        or ()
                    )
                )
                if not (
                    runtime_probe.get("runtime_enabled") is True
                    and str(runtime_probe.get("tree_digest") or "")
                    == expected_runtime_tree_digest
                    and probe_roster == expected_runtime_roster
                ):
                    reason = (
                        "simulator-child matchup runtime differs from its "
                        f"controller on {ep}: expected_tree="
                        f"{expected_runtime_tree_digest} expected_routes="
                        f"{expected_runtime_roster!r} probe={runtime_probe!r}"
                    )
                    try:
                        client.request_rotation(reason)
                    except Exception as rotation_exc:  # noqa: BLE001
                        reason += (
                            "; controlled rotation request failed: "
                            f"{type(rotation_exc).__name__}: {rotation_exc}"
                        )
                    raise RemoteJobsError(reason)
            remote_proof.append(
                {
                    "endpoint": ep,
                    "reload_digest": got,
                    "pin_digest": pin_got,
                    "hello_digest": hello_dig,
                    "version": reload_reply.get("version"),
                    "reload_skipped_exact_health": reload_skipped_exact_health,
                    "matchup_runtime": dict(runtime) if runtime is not None else None,
                    "matchup_runtime_worker_probe": runtime_probe,
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ep}: {exc}")
    if errors:
        raise BetweenIterSyncError(
            "between-iter hard-gate remote sync failed: " + "; ".join(errors)
        )
    proof["remote_ok"] = True
    proof["remote_endpoints"] = remote_proof
    print(
        f"[pure_rl] BETWEEN_ITER_HARD_GATE ok digest={dig[:19]}… "
        f"version={version} local=1 remotes={len(remote_proof)} "
        f"endpoints={[r['endpoint'] for r in remote_proof]}",
        flush=True,
    )
    return proof


def _remote_dispatch_slots(
    *,
    remote_farm: Any,
    scheduler: Any,
    baseline_workers: int,
    kind: str = "self_play",
    allow_remote_play: bool = False,
) -> tuple[int, int, str]:
    """Return (local_slots, remote_cap, weight_bits) for additive collect.

    ``kind="play"`` is the public-mix wave (``baseline_jobs`` vs the public /
    recent-self roster). It finishes fast locally, so by default it is
    routed local-only (``PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1``), keeping remotes
    free for the self-play wave and other/slower work. Self-play
    (``kind="self_play"``) always keeps the original shared
    ``PURE_RL_REBALANCE_MIN_LOCAL_FRAC`` behavior, unaffected.
    """
    from poke_bot.remote_jobs import describe_endpoint_weights, weighted_remote_capacity

    assert remote_farm is not None
    is_public_mix = kind == "play"
    if is_public_mix and _public_mix_local_only() and not allow_remote_play:
        local_slots = max(1, int(baseline_workers))
        weight_bits = (
            "local_only (PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1; remotes free for "
            "self_play/other work)"
        )
        return local_slots, 0, weight_bits
    demand = None
    if scheduler is not None:
        try:
            scheduler.bind_remote_endpoints(remote_farm.clients)
            demand = scheduler.remote_demand()
        except Exception:
            demand = None
    weight_rows = describe_endpoint_weights(
        remote_farm.clients, demand_by_endpoint=demand
    )
    remote_cap = max(
        1,
        weighted_remote_capacity(remote_farm.clients, demand_by_endpoint=demand),
    )
    weight_bits = ", ".join(
        f"{r['endpoint']} demand={r.get('demand_workers', '?')}"
        f"→slots={r['dispatch_slots']}/{r.get('capacity_workers', r['advertised_workers'])}"
        f"(default={r.get('default_workers', '?')} max={r.get('max_workers', '?')})"
        for r in weight_rows
    )
    # Additive: keep full local baseline (≈96) and pile remotes on top.
    # Do NOT halve local_slots for RR — that stole from local until remotes
    # filled, capping the pie at ~baseline instead of local+remote.
    if is_public_mix and allow_remote_play:
        # Formal held-out is deliberately additive: keep the proven local pool
        # full and add compatible remote sockets.  The 95% public-mix bias would
        # otherwise manufacture hundreds of local slots and exceed leaf queues.
        local_slots = max(1, int(baseline_workers))
        weight_bits = "formal_heldout_additive; " + weight_bits
        return local_slots, remote_cap, weight_bits
    if is_public_mix:
        # Heavy-local fallback (LOCAL_ONLY explicitly disabled): bias much
        # harder toward local than the self-play share below.
        min_local_frac = _public_mix_min_local_frac()
    else:
        min_local_frac = float(
            os.environ.get("PURE_RL_REBALANCE_MIN_LOCAL_FRAC", "0.40")
        )
    min_local_frac = min(0.95, max(0.05, min_local_frac))
    min_local_slots = max(
        1,
        int(math.ceil(remote_cap * min_local_frac / (1.0 - min_local_frac))),
    )
    local_slots = max(min_local_slots, max(1, int(baseline_workers)))
    return local_slots, remote_cap, weight_bits


def _build_research_control_jobs(
    *,
    n_games: int,
    ckpt: Path,
    digest: str,
    model_generation: int,
    decks: list[tuple[str, list[int]]],
    specs: list[Any],
    seed: int,
    game_timeout_s: int,
    mode: str,
    registry: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    """Build the additive exact control wave outside all training schedules."""
    controls = list(registry.get("controls") or [])
    registry_ids = tuple(str(row.get("opponent_id") or "") for row in controls)
    specs_by_id = {str(spec.id): spec for spec in specs}
    if (
        not registry_ids
        or "" in registry_ids
        or len(set(registry_ids)) != len(registry_ids)
        or set(specs_by_id) != set(registry_ids)
    ):
        raise RuntimeError("research-control registry/spec roster is inconsistent")
    if int(n_games) != 250 * len(registry_ids):
        raise RuntimeError(
            "research controls require exactly 250 games per registered control"
        )
    ordered_specs = [specs_by_id[opponent_id] for opponent_id in registry_ids]
    _self_jobs, jobs = _build_collect_jobs(
        n_games=int(n_games),
        ckpt=Path(ckpt),
        digest=str(digest),
        model_generation=int(model_generation),
        decks=decks,
        specs=ordered_specs,
        seed=int(seed),
        game_timeout_s=int(game_timeout_s),
        mode=str(mode),
        self_play_frac=0.0,
        balanced_eval=True,
        iteration=int(iteration),
    )
    archetypes = {
        str(row["opponent_id"]): str(row.get("archetype_id") or "")
        for row in controls
    }
    registry_id = str(registry.get("registry_id") or "")
    registry_version = int(registry.get("version") or 0)
    for job in jobs:
        opponent_id = str(job.get("opponent_id") or "")
        provenance = dict(job.get("target_provenance") or {})
        job.update(
            {
                "training_eligible": False,
                "sample_actions": False,
                "greedy": True,
                "action_temperature": 1.0,
                "collect_privileged_belief": False,
                "opp_archetype": archetypes[opponent_id],
            }
        )
        job["target_provenance"] = {
            **provenance,
            "pure_rl": False,
            "soft_policy_targets": False,
            "collect": "research_controls_measurement",
            "opponent_training_group": RESEARCH_CONTROL_GROUP,
            "self_play": False,
            "behavior_checkpoint": str(ckpt),
            "behavior_checkpoint_digest": str(digest),
            "action_temperature": 1.0,
            "behavior_mode": "greedy_research_control_measurement_v1",
            "research_control_registry_id": registry_id,
            "research_control_registry_version": registry_version,
            "diagnostic_only": True,
            "included_in_gate_pass": False,
            "gate_weight": 0.0,
            "formal_eval": False,
            "training_eligible": False,
            "replay_eligible": False,
            "seed_namespace": "eval/research-controls-fixed-manifest-v1",
            "seat_schedule": "research_control_paired_v1",
        }
    return jobs


def _assert_training_jobs_exclude_research_controls(
    jobs: list[dict[str, Any]], registry: dict[str, Any]
) -> None:
    """Fail closed if a control ID or package alias enters replay collection."""
    controls = list(registry.get("controls") or [])
    control_ids = {str(row.get("opponent_id") or "") for row in controls}
    control_digests = {str(row.get("content_digest") or "") for row in controls}
    for job in jobs:
        if job.get("training_eligible") is not True:
            raise RuntimeError("training collection contains a non-training job")
        provenance = dict(job.get("target_provenance") or {})
        spec = dict(job.get("spec") or {})
        opponent_id = str(job.get("opponent_id") or "")
        digests = {
            str(value)
            for value in (
                spec.get("content_digest"),
                provenance.get("opponent_content_digest"),
            )
            if value
        }
        if (
            opponent_id in control_ids
            or bool(digests & control_digests)
            or str(provenance.get("opponent_training_group") or "")
            == RESEARCH_CONTROL_GROUP
        ):
            raise RuntimeError(
                f"research control leaked into replay-eligible collection: {opponent_id}"
            )


def _assert_research_control_jobs(
    jobs: list[dict[str, Any]],
    *,
    expected_games: int,
    registry: dict[str, Any],
    iteration: int,
    root_seed: int,
    training_jobs: list[dict[str, Any]],
    formal_games: int,
    checkpoint_digest: str,
    active_gate_digests: set[str],
) -> dict[str, Any]:
    """Audit the exact registry-backed, non-training measurement schedule."""
    from collections import Counter

    expected = int(expected_games)
    if len(jobs) != expected:
        raise RuntimeError(
            "research-control quota mismatch: "
            f"actual={len(jobs)} expected={expected}"
        )
    controls = list(registry.get("controls") or [])
    expected_ids = tuple(str(row.get("opponent_id") or "") for row in controls)
    if not expected_ids or "" in expected_ids or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise RuntimeError("research-control registry roster is invalid")
    expected_archetypes = {
        str(row["opponent_id"]): str(row.get("archetype_id") or "")
        for row in controls
    }
    registry_id = str(registry.get("registry_id") or "")
    registry_version = int(registry.get("version") or 0)
    registry_digests = {
        str(row["opponent_id"]): str(row.get("content_digest") or "")
        for row in controls
    }
    if expected != 250 * len(expected_ids):
        raise RuntimeError("research-control allocation must be exactly 250/control")
    if set(registry_digests.values()) & set(active_gate_digests):
        raise RuntimeError("research-control package aliases the active gate")
    counts = Counter(str(job.get("opponent_id") or "") for job in jobs)
    if set(counts) != set(expected_ids) or any(
        counts[opponent_id] != 250 for opponent_id in expected_ids
    ):
        raise RuntimeError(
            "research-control scheduled roster does not match the pinned registry"
        )
    job_indexes = [int(job.get("job_index", -1)) for job in jobs]
    if set(job_indexes) != set(range(expected)) or len(job_indexes) != len(
        set(job_indexes)
    ):
        raise RuntimeError("research-control job indexes are not exact and unique")
    expected_seed_start = (
        int(root_seed)
        + RESEARCH_CONTROL_SEED_OFFSET
        + int(iteration) * ITERATION_SEED_STRIDE
    )
    expected_seeds = set(range(expected_seed_start, expected_seed_start + expected))
    observed_seeds = {int(job.get("seed", -1)) for job in jobs}
    if observed_seeds != expected_seeds:
        raise RuntimeError("research-control seeds do not match their fixed namespace")
    training_seeds = {int(job.get("seed", -1)) for job in training_jobs}
    formal_seed_start = (
        int(root_seed)
        + FORMAL_GATE_SEED_OFFSET
        + int(iteration) * ITERATION_SEED_STRIDE
    )
    formal_seeds = set(range(formal_seed_start, formal_seed_start + int(formal_games)))
    if observed_seeds & training_seeds or observed_seeds & formal_seeds:
        raise RuntimeError("research-control seeds overlap training or formal gate")
    seat_counts: dict[str, list[int]] = {}
    for job in jobs:
        opponent_id = str(job.get("opponent_id") or "")
        provenance = dict(job.get("target_provenance") or {})
        spec = dict(job.get("spec") or {})
        content_digest = registry_digests.get(opponent_id, "")
        if (
            opponent_id not in expected_archetypes
            or str(job.get("opp_archetype") or "")
            != expected_archetypes[opponent_id]
            or str(job.get("checkpoint_digest") or "") != str(checkpoint_digest)
            or str(spec.get("content_digest") or "") != content_digest
            or str(provenance.get("opponent_content_digest") or "")
            != content_digest
            or str(provenance.get("research_control_registry_id") or "")
            != registry_id
            or int(provenance.get("research_control_registry_version") or 0)
            != registry_version
            or provenance.get("formal_eval") is not False
            or provenance.get("diagnostic_only") is not True
            or provenance.get("included_in_gate_pass") is not False
            or float(provenance.get("gate_weight", -1.0)) != 0.0
            or provenance.get("training_eligible") is not False
            or provenance.get("replay_eligible") is not False
            or str(provenance.get("collect") or "")
            != "research_controls_measurement"
            or str(provenance.get("seed_namespace") or "")
            != "eval/research-controls-fixed-manifest-v1"
            or job.get("training_eligible") is not False
            or job.get("sample_actions") is not False
            or job.get("greedy") is not True
            or float(job.get("action_temperature", -1.0)) != 1.0
        ):
            raise RuntimeError(
                f"research-control identity/context mismatch for {opponent_id}"
            )
        bucket = seat_counts.setdefault(opponent_id, [0, 0])
        seat = int(job.get("our_seat", -1))
        if seat not in (0, 1):
            raise RuntimeError("research-control job has an invalid candidate seat")
        bucket[seat] += 1
    if any(seats != [125, 125] for seats in seat_counts.values()):
        raise RuntimeError("research-control seats must be exactly 125/125")
    schedule_digest = _canonical_digest(
        [
            {
                "job_index": int(job["job_index"]),
                "seed": int(job["seed"]),
                "opponent_id": str(job["opponent_id"]),
                "opponent_content_digest": str(
                    (job.get("target_provenance") or {}).get(
                        "opponent_content_digest"
                    )
                    or ""
                ),
                "our_seat": int(job["our_seat"]),
                "checkpoint_digest": str(job["checkpoint_digest"]),
            }
            for job in jobs
        ]
    )
    return {
        "schema": "poke_bot.research_control_measurement_plan/v1",
        "registry_id": registry_id,
        "registry_version": registry_version,
        "iteration": int(iteration),
        "training_eligible": False,
        "replay_eligible": False,
        "formal_eval": False,
        "diagnostic_only": True,
        "greedy": True,
        "sample_actions": False,
        "checkpoint_digest": str(checkpoint_digest),
        "seed_namespace": "eval/research-controls-fixed-manifest-v1",
        "seed_start": expected_seed_start,
        "seed_disjoint": True,
        "package_disjoint_from_active_gate": True,
        "schedule_digest": schedule_digest,
        "games": len(jobs),
        "per_opponent": {
            opponent_id: {
                "games": int(counts.get(opponent_id, 0)),
                "seat0": int(seat_counts[opponent_id][0]),
                "seat1": int(seat_counts[opponent_id][1]),
                "content_digest": registry_digests[opponent_id],
            }
            for opponent_id in expected_ids
        },
    }


def _exclude_protected_baseline_aliases(
    *,
    specs: list[Any],
    excluded_ids: set[str],
    digest_by_id: dict[str, str],
    protected_digests: set[str],
) -> tuple[list[Any], list[str]]:
    """Exclude exact IDs and content aliases of gate/control packages."""
    candidates = [spec for spec in specs if str(spec.id) not in excluded_ids]
    aliases = sorted(
        str(spec.id)
        for spec in candidates
        if str(digest_by_id.get(str(spec.id)) or "") in protected_digests
    )
    alias_ids = set(aliases)
    return [spec for spec in candidates if str(spec.id) not in alias_ids], aliases


def _rehydrate_partial_collection_resume(
    *,
    shard_path: Path,
    sidecar: dict[str, Any],
    self_play_jobs: list[dict[str, Any]],
    baseline_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct source-game audit rows from every sealed trajectory row."""
    jobs = {
        int(job.get("job_index", -1)): job
        for job in [*self_play_jobs, *baseline_jobs]
    }
    if -1 in jobs or len(jobs) != len(self_play_jobs) + len(baseline_jobs):
        raise RuntimeError("partial collection resume job schedule is not unique")
    retained_expected = {int(value) for value in sidecar["retained_job_indices"]}
    self_play_job_indices = {
        int(job.get("job_index", -1)) for job in self_play_jobs
    }
    grouped: dict[int, list[dict[str, Any]]] = {}
    episode_ids: set[str] = set()
    records = 0
    decisions = 0
    with Path(shard_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            episode_id = str(obj.get("episode_id") or "")
            provenance = dict(obj.get("target_provenance") or {})
            job_index = int(provenance.get("collection_job_index", -1))
            decision_rows = obj.get("decisions")
            if (
                not episode_id
                or episode_id in episode_ids
                or job_index not in jobs
                or not isinstance(decision_rows, list)
                or not decision_rows
                or str(provenance.get("behavior_checkpoint_digest") or "")
                != str(sidecar["behavior_checkpoint_digest"])
            ):
                raise RuntimeError(
                    "partial collection resume shard provenance is invalid: "
                    f"episode={episode_id!r} job_index={job_index}"
                )
            episode_ids.add(episode_id)
            grouped.setdefault(job_index, []).append(obj)
            records += 1
            decisions += len(decision_rows)
    if (
        set(grouped) != retained_expected
        or records != int(sidecar["trajectory_records"])
        or decisions != int(sidecar["decisions"])
    ):
        raise RuntimeError("partial collection resume counters changed after sealing")

    rows: list[dict[str, Any]] = []
    for job_index in sorted(grouped):
        job = jobs[job_index]
        job_provenance = dict(job.get("target_provenance") or {})
        job_spec = dict(job.get("spec") or {})
        trajectory_rows = grouped[job_index]
        self_play = job_index in self_play_job_indices
        # Mirror jobs only collect both trajectories when both seats use the
        # exact same checkpoint.  Learner-vs-prior self-play intentionally
        # records the learner's assigned seat only; seat balance is provided
        # by the alternating job schedule.  Treating every self-play job as a
        # two-row source game made valid partial shards impossible to resume.
        collect_both = bool(self_play and job.get("collect_both_seats", False))
        expected_records = 2 if collect_both else 1
        if len(trajectory_rows) != expected_records:
            raise RuntimeError(
                "partial collection resume source-game trajectory cardinality changed: "
                f"job={job_index} got={len(trajectory_rows)} expected={expected_records}"
            )
        assigned_seat = int(job.get("our_seat", -1))
        by_seat = {int(row.get("seat", -1)): row for row in trajectory_rows}
        if assigned_seat not in by_seat or (
            collect_both and set(by_seat) != {0, 1}
        ):
            raise RuntimeError(
                f"partial collection resume seat provenance changed: job={job_index}"
            )
        acting = dict(by_seat[assigned_seat].get("target_provenance") or {})
        opponent = (
            dict(by_seat[1 - assigned_seat].get("target_provenance") or {})
            if collect_both
            else {}
        )
        rows.append(
            {
                "job_index": job_index,
                "our_seat": assigned_seat,
                "archetype": str(job.get("archetype") or ""),
                "opponent_id": str(job.get("opponent_id") or ""),
                "opponent_content_digest": str(
                    job_provenance.get("opponent_content_digest")
                    or job.get("opponent_content_digest")
                    or job_spec.get("content_digest")
                    or ""
                ),
                "opponent_training_group": str(
                    job_provenance.get("opponent_training_group")
                    or job.get("opponent_training_group")
                    or ""
                ),
                "formal_eval": (
                    job_provenance.get("formal_eval")
                    if "formal_eval" in job_provenance
                    else job.get("formal_eval")
                ),
                "training_eligible": bool(
                    job_provenance.get(
                        "training_eligible", job.get("training_eligible", False)
                    )
                ),
                "self_play": self_play,
                "invalid": False,
                "policy_terminal_failure": bool(acting.get("terminal_policy_failure")),
                "matchup_runtime_audit": acting.get("matchup_runtime_audit"),
                "opponent_matchup_runtime_audit": opponent.get(
                    "matchup_runtime_audit"
                ),
            }
        )
    return {
        "rows": rows,
        "trajectory_records": records,
        "decisions": decisions,
        "retained_job_indices": retained_expected,
    }


def _collect_wave(
    *,
    self_play_jobs: list[dict[str, Any]],
    baseline_jobs: list[dict[str, Any]],
    shard_path: Path,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
    worker_self_play,
    multi_env_per_worker: int = 1,
    worker_self_play_multi=None,
    iteration: int = 0,
    stage_label: Optional[str] = None,
    live_wr_gate: Optional[tuple[float, int]] = None,
    allow_remote_play: bool = False,
    required_checkpoint_digest: Optional[str] = None,
    live_wr_opponent_ids: Optional[tuple[str, ...]] = None,
    replay_eligible: bool = True,
    required_mirror_archetype: Optional[str] = None,
) -> tuple[CompactShardWriter, list[dict[str, Any]], dict[str, Any]]:
    """Self-play plus public training games on one replay/AWR shard.

    ``stage_label`` overrides the tqdm bar's default ``collect:public_mix``
    name (e.g. ``"heldout"`` so the heldout gate wave doesn't look like a
    regular training collect). ``live_wr_gate`` streams a running WR onto
    that bar for the baseline_jobs wave only — see ``_heldout_eval``.
    """
    from poke_bot.pure_rl.multi_env_self_play import (
        chunk_jobs,
        process_worker_count,
    )
    from poke_bot.pure_rl.mid_iter_scheduler import (
        MidIterScheduler,
        log_scheduler_banner,
        mid_iter_scheduler_enabled,
    )
    from poke_bot.remote_jobs import (
        iter_additive_results,
        iter_scheduled_additive_results,
        weighted_remote_capacity,
    )
    from poke_bot.worker_pool import WorkerPool

    # Bind the controller's routing identity into every job before it can be
    # claimed by either a local WorkerPool child or a remote endpoint. Remote
    # servers replace this reserved field with their equivalent host-local
    # tree path; local children use this value to reassert the canonical tree
    # at the exact job boundary. Without the local binding, a recycled process
    # can retain an older process-global router even though the controller and
    # remote hard-gates are current.
    runtime_required = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if runtime_required:
        runtime_tree = Path(
            os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
        ).expanduser().resolve()
        try:
            runtime_payload = json.loads(runtime_tree.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "cannot bind controller matchup runtime to collection jobs"
            ) from exc
        accepted_ids = sorted(
            str(value)
            for value in dict(
                runtime_payload.get("runtime_contract") or {}
            ).get("accepted_archetype_ids", ())
        )
        if not accepted_ids:
            raise RuntimeError(
                "cannot bind controller matchup runtime with an empty roster"
            )
        runtime_binding = {
            "tree": str(runtime_tree),
            "tree_digest": _sha256_file(runtime_tree),
            "accepted_archetype_ids": accepted_ids,
        }
        for job in (*self_play_jobs, *baseline_jobs):
            job["_controller_matchup_runtime"] = dict(runtime_binding)

    writer = CompactShardWriter(shard_path)
    replay_cache: Optional[StreamingReplayCache] = None
    rows: list[dict[str, Any]] = []
    multi_n = max(1, int(multi_env_per_worker))
    proc_workers = process_worker_count(n_workers, multi_n)
    stats = {
        "ok": 0,
        "baseline_failed": 0,
        "our_failed": 0,
        "resource_error": 0,
        "with_record": 0,
        "self_play": 0,
        "n_self_play_jobs": len(self_play_jobs),
        "n_baseline_jobs": len(baseline_jobs),
        "n_research_control_jobs": 0,
        "n_public_mix_jobs": len(baseline_jobs),
        "multi_env_per_worker": multi_n,
        "proc_workers": proc_workers,
        "leaf_remote": 0,
        "multi_env_games": 0,
        "leaf_modes": {},
        "execution_origin_counts": {},
        "remote_endpoint_counts": {},
        "remote_self_play_endpoint_counts": {},
        # Default 0 preserves one append/close per completed game.  A
        # receipt-gated throughput trial may raise this bounded value; the
        # realized flush count is recorded before the collection receipt is
        # committed so it can never be mistaken for a policy/data change.
        "compact_shard_write_buffer_bytes": int(writer.buffer_bytes),
        "compact_shard_write_flushes": 0,
    }
    practice_record_contracts: dict[int, dict[str, str]] = {}
    for job in baseline_jobs:
        provenance = dict(job.get("target_provenance") or {})
        if (
            str(provenance.get("opponent_training_group") or "")
            != STRONG_PUBLIC_PRACTICE_GROUP
        ):
            continue
        job_index = int(job.get("job_index", -1))
        opponent_id = str(job.get("opponent_id") or "")
        opponent_archetype_id = str(job.get("opp_archetype") or "")
        active_gate_id = str(provenance.get("active_gate_id") or "")
        if (
            job_index < 0
            or job_index in practice_record_contracts
            or not opponent_id
            or not opponent_archetype_id
            or not active_gate_id
            or str(provenance.get("opponent_id") or "") != opponent_id
            or str(provenance.get("opponent_archetype_id") or "")
            != opponent_archetype_id
        ):
            raise RuntimeError(
                "invalid strong-public practice record contract: "
                f"job_index={job_index} opponent={opponent_id!r} "
                f"archetype={opponent_archetype_id!r}"
            )
        practice_record_contracts[job_index] = {
            "opponent_id": opponent_id,
            "opponent_archetype_id": opponent_archetype_id,
            "active_gate_id": active_gate_id,
            "our_seat": str(int(job.get("our_seat", -1))),
        }
    practice_seen_indices: set[int] = set()
    practice_successful_indices: set[int] = set()
    practice_written_indices: set[int] = set()
    primary_self_play_jobs = [
        job
        for job in self_play_jobs
        if not bool(
            (job.get("target_provenance") or {}).get("replacement_capacity")
        )
    ]
    primary_self_play_indices = {
        int(job.get("job_index", -1)) for job in primary_self_play_jobs
    }
    if -1 in primary_self_play_indices:
        raise RuntimeError("self-play collection job lacks an exact job index")
    retained_self_play_indices: set[int] = set()
    retained_public_indices: set[int] = set()
    replacement_spool = shard_path.with_suffix(
        shard_path.suffix + ".replacement-capacity.jsonl"
    )
    replacement_spool.unlink(missing_ok=True)
    partial_resume = _verified_partial_collection_resume(
        shard_path, iteration=int(iteration)
    )
    dispatch_self_play_jobs = list(self_play_jobs)
    dispatch_baseline_jobs = list(baseline_jobs)
    if partial_resume is not None:
        restored = _rehydrate_partial_collection_resume(
            shard_path=shard_path,
            sidecar=partial_resume,
            self_play_jobs=self_play_jobs,
            baseline_jobs=baseline_jobs,
        )
        retained = set(restored["retained_job_indices"])
        retained_self_play_indices.update(primary_self_play_indices & retained)
        retained_public_indices.update(
            {int(job.get("job_index", -1)) for job in baseline_jobs} & retained
        )
        rows.extend(restored["rows"])
        writer._n_games = int(restored["trajectory_records"])
        writer._n_decisions = int(restored["decisions"])
        writer._t0 = time.time() - max(
            float(partial_resume.get("elapsed_sec") or 1.0), 1.0
        )
        stats["ok"] = len(retained)
        stats["with_record"] = len(retained)
        stats["self_play"] = len(retained_self_play_indices)
        stats["trajectories_written"] = int(restored["trajectory_records"])
        stats["partial_collection_resume"] = {
            "schema": _PARTIAL_COLLECTION_RESUME_SCHEMA,
            "sidecar": str(partial_resume["sidecar_path"]),
            "sidecar_sha256": _sha256_file(Path(partial_resume["sidecar_path"])),
            "retained_source_games": len(retained),
            "missing_source_games": len(partial_resume["missing_job_indices"]),
            "original_shard_sha256": str(partial_resume["shard_sha256"]),
        }
        dispatch_self_play_jobs = [
            job
            for job in self_play_jobs
            if int(job.get("job_index", -1)) not in retained
        ]
        dispatch_baseline_jobs = [
            job
            for job in baseline_jobs
            if int(job.get("job_index", -1)) not in retained
        ]
        for job_index in practice_record_contracts:
            if int(job_index) in retained_public_indices:
                practice_seen_indices.add(int(job_index))
                practice_successful_indices.add(int(job_index))
                practice_written_indices.add(int(job_index))
        print(
            "[pure_rl] partial collection resume verified "
            f"retained={len(retained)} "
            f"missing={len(partial_resume['missing_job_indices'])} "
            f"records={writer.n_games} decisions={writer.n_decisions}",
            flush=True,
        )
    if not self_play_jobs and not baseline_jobs:
        return writer, rows, stats
    if replay_eligible and partial_resume is None:
        replay_cache = StreamingReplayCache.maybe_start(
            shard_path,
            verify_info_set=False,
            max_context=pure_rl_model_config().max_context,
        )

    # Hello advertised defaults — bar uses demand-based dispatch slots after bind
    # (must track live open sockets, not frozen hello workers).
    remotes_hello = int(remote_farm.total_workers) if remote_farm is not None else 0
    use_remotes = bool(remote_farm is not None and remotes_hello > 0)
    # Multi-env batching only when remotes are down. With remotes up we dispatch
    # single-game jobs so the LAN farm can share the same list — use full
    # sim_workers fan-out so sticky leaf binds cover the striped GPU0/GPU1 map
    # (proc_workers=24 would pin every client onto the first 24 GPU1 servers).
    use_multi_env_batches = bool(
        (not use_remotes) and multi_n > 1 and worker_self_play_multi is not None
    )
    local_workers = (
        max(1, int(proc_workers))
        if use_multi_env_batches
        else max(1, int(n_workers))
    )
    # Fail-closed guard: WorkerPool requires one resp_q per worker. Never open
    # a pool larger than the leaf farm's queue list (live_pool can bump workers
    # while leaf rebuild is deferred).
    if isinstance(leaf_channel, dict):
        n_qs = len(leaf_channel.get("resp_qs") or [])
        if n_qs > 0 and local_workers > n_qs:
            print(
                f"[pure_rl] WARN clamp local_workers {local_workers}->{n_qs} "
                f"to match leaf resp_qs (avoid queues<workers crash)",
                flush=True,
            )
            local_workers = n_qs
    use_scheduler = bool(use_remotes and mid_iter_scheduler_enabled())
    scheduler = (
        MidIterScheduler.from_env(baseline_workers=max(1, int(n_workers)))
        if use_scheduler
        else None
    )
    if scheduler is not None and remote_farm is not None:
        try:
            scheduler.bind_remote_endpoints(remote_farm.clients)
        except Exception as exc:
            print(f"[pure_rl] bind_remote_endpoints failed: {exc!r}", flush=True)
    remotes = remotes_hello
    if use_remotes and remote_farm is not None:
        try:
            demand0 = scheduler.remote_demand() if scheduler is not None else None
            remotes = max(
                1,
                weighted_remote_capacity(
                    remote_farm.clients, demand_by_endpoint=demand0
                ),
            )
        except Exception:
            remotes = remotes_hello
    if scheduler is not None:
        log_scheduler_banner(scheduler)
    print(
        f"[pure_rl] self_play_pool workers={local_workers} "
        f"multi_env_batches={int(use_multi_env_batches)} "
        f"proc_cap={proc_workers} sim_workers={n_workers} remotes={remotes} "
        f"(hello_defaults={remotes_hello}; bar tracks live dispatch sockets)",
        flush=True,
    )

    execution_lock = threading.Lock()

    def _record_execution(info: dict[str, Any]) -> None:
        origin = str(info.get("origin") or "unknown")
        kind = str(info.get("kind") or "unknown")
        endpoint = str(info.get("endpoint") or "")
        with execution_lock:
            origins = stats["execution_origin_counts"]
            origins[origin] = int(origins.get(origin, 0)) + 1
            if origin == "remote" and endpoint:
                endpoints_seen = stats["remote_endpoint_counts"]
                endpoints_seen[endpoint] = int(endpoints_seen.get(endpoint, 0)) + 1
                if kind == "self_play":
                    self_play_seen = stats["remote_self_play_endpoint_counts"]
                    self_play_seen[endpoint] = int(self_play_seen.get(endpoint, 0)) + 1

    def _additive_iter(
        *,
        pool,
        local_fn,
        jobs,
        kind: str,
        baseline_workers: int,
        progress: Optional[_TqdmProgress] = None,
    ):
        local_slots, remote_cap, weight_bits = _remote_dispatch_slots(
            remote_farm=remote_farm,
            scheduler=scheduler,
            baseline_workers=baseline_workers,
            kind=kind,
            allow_remote_play=allow_remote_play,
        )
        public_mix_local_only = bool(kind == "play" and remote_cap == 0)
        total_slots = local_slots + remote_cap
        share = local_slots / float(total_slots)
        remote_max_total = 0
        if not public_mix_local_only:
            try:
                remote_max_total = int(
                    sum(
                        int(scheduler.remote_maxima.get(c.endpoint, 0))
                        for c in remote_farm.clients
                    )
                    if scheduler is not None
                    else 0
                )
            except Exception:
                remote_max_total = 0
        max_total = local_slots + max(remote_cap, remote_max_total)
        dem = remote_cap
        if scheduler is not None and not public_mix_local_only:
            try:
                dem = int(sum(scheduler.remote_demand().values()))
            except Exception:
                dem = remote_cap
        if progress is not None:
            progress.set_remotes(remote_cap, demand=dem)
        print(
            f"[pure_rl] {kind} remote_weights {weight_bits} "
            f"local_slots={local_slots} remote_slots={remote_cap} "
            f"total_slots={total_slots} max_total≈{max_total} "
            f"local_share={share:.0%} "
            f"(additive kind={kind}; local stays full, remotes on top; "
            f"bar remotes=live_sockets start={remote_cap}"
            + (
                "; mid_iter_scheduler=on, remote_chunked_rtt)"
                if scheduler is not None
                else ")"
            ),
            flush=True,
        )

        def _on_remote_slots(info: dict[str, int]) -> None:
            if progress is None:
                return
            try:
                progress.set_remotes(
                    int(info.get("active", 0)),
                    demand=int(info["demand"]) if "demand" in info else None,
                    outstanding=(
                        int(info["outstanding"])
                        if "outstanding" in info
                        else None
                    ),
                    outstanding_elmo=(
                        int(info["outstanding_elmo"])
                        if "outstanding_elmo" in info
                        else None
                    ),
                    outstanding_bert=(
                        int(info["outstanding_bert"])
                        if "outstanding_bert" in info
                        else None
                    ),
                )
            except Exception:
                pass

        release_pool_before_drain = str(
            os.environ.get(
                "POKEBOT_RELEASE_LOCAL_POOL_BEFORE_RESULT_DRAIN", "0"
            )
        ).strip().lower() in {"1", "true", "yes", "on"}

        def _release_exhausted_local_pool() -> None:
            before = len(getattr(pool, "live_worker_pids", ()))
            pool.release()
            stats["early_local_pool_releases"] = int(
                stats.get("early_local_pool_releases", 0)
            ) + 1
            print(
                "[pure_rl] result producers drained; released exhausted "
                f"local WorkerPool children={before} before compaction tail",
                flush=True,
            )

        if scheduler is not None:
            return iter_scheduled_additive_results(
                local_pool=pool,
                local_fn=local_fn,
                jobs=jobs,
                remote_clients=remote_farm.clients,
                kind=kind,
                scheduler=scheduler,
                local_workers=local_slots,
                remote_workers=remote_cap,
                on_remote_slots=_on_remote_slots,
                on_execution=_record_execution,
                on_producers_drained=(
                    _release_exhausted_local_pool
                    if release_pool_before_drain
                    else None
                ),
            )
        return iter_additive_results(
            local_pool=pool,
            local_fn=local_fn,
            jobs=jobs,
            remote_clients=remote_farm.clients,
            kind=kind,
            local_workers=local_slots,
            remote_workers=remote_cap,
            on_execution=_record_execution,
            on_producers_drained=(
                _release_exhausted_local_pool
                if release_pool_before_drain
                else None
            ),
        )

    # Primary: pure self-play — local MultiEnv + additive remote self_play sockets.
    if dispatch_self_play_jobs:
        progress = _TqdmProgress(
            stage="collect:self_play",
            iteration=int(iteration),
            total=len(dispatch_self_play_jobs),
            remotes=remotes,
        )
        try:
            with WorkerPool(
                num_workers=local_workers, remote_channel=leaf_channel
            ) as pool:
                if use_remotes:
                    # Single-game local fn so remotes can share the same job list.
                    # MultiEnv stays available when remotes are down.
                    results_iter = _additive_iter(
                        pool=pool,
                        local_fn=worker_self_play,
                        jobs=dispatch_self_play_jobs,
                        kind="self_play",
                        baseline_workers=local_workers,
                        progress=progress,
                    )
                elif use_multi_env_batches:
                    batches = [
                        {"jobs": chunk}
                        for chunk in chunk_jobs(dispatch_self_play_jobs, multi_n)
                    ]
                    results_iter = _flatten_batch_results(
                        pool.imap_unordered(worker_self_play_multi, batches)
                    )
                else:
                    results_iter = pool.imap_unordered(
                        worker_self_play, dispatch_self_play_jobs
                    )
                _consume_results(
                    results_iter,
                    writer,
                    rows,
                    stats,
                    progress=progress,
                    replay_cache=replay_cache,
                    replacement_spool=(replacement_spool if replay_eligible else None),
                    retained_job_indices=(
                        retained_self_play_indices if replay_eligible else None
                    ),
                )
        finally:
            progress.close()
        if replay_eligible:
            missing_self_play = (
                primary_self_play_indices - retained_self_play_indices
            )
            promoted = _promote_replacement_spool(
                replacement_spool,
                missing_job_indices=missing_self_play,
                writer=writer,
                replay_cache=replay_cache,
                primary_jobs=primary_self_play_jobs,
            )
            promoted_indices = set(promoted["promoted_job_indices"])
            retained_self_play_indices.update(promoted_indices)
            rows.extend(promoted["promoted_runtime_audit_rows"])
            stats["replacement_capacity_promoted_source_games"] = int(
                promoted["promoted_source_games"]
            )
            stats["replacement_capacity_promoted_trajectories"] = int(
                promoted["promoted_trajectories"]
            )
            stats["replacement_capacity_promoted_decisions"] = int(
                promoted["promoted_decisions"]
            )
            stats["with_record"] += int(promoted["promoted_source_games"])
            stats["trajectories_written"] = int(
                stats.get("trajectories_written", 0)
            ) + int(promoted["promoted_trajectories"])
            missing_self_play = (
                primary_self_play_indices - retained_self_play_indices
            )
            next_replacement_job_index = (
                max(
                    [
                        int(job.get("job_index", -1))
                        for job in [*self_play_jobs, *baseline_jobs]
                    ],
                    default=-1,
                )
                + 1
            )
            for retry_round in range(4):
                if not missing_self_play:
                    break
                retry_jobs = _targeted_replacement_jobs(
                    primary_self_play_jobs,
                    missing_job_indices=missing_self_play,
                    retry_round=retry_round,
                    first_job_index=next_replacement_job_index,
                )
                next_replacement_job_index += len(retry_jobs)
                stats["targeted_self_play_retry_attempts"] = int(
                    stats.get("targeted_self_play_retry_attempts", 0)
                ) + len(retry_jobs)
                retry_progress = _TqdmProgress(
                    stage="collect:self_play_refill",
                    iteration=int(iteration),
                    total=len(retry_jobs),
                    remotes=remotes,
                )
                try:
                    with WorkerPool(
                        num_workers=local_workers, remote_channel=leaf_channel
                    ) as retry_pool:
                        if use_remotes:
                            retry_results = _additive_iter(
                                pool=retry_pool,
                                local_fn=worker_self_play,
                                jobs=retry_jobs,
                                kind="self_play",
                                baseline_workers=local_workers,
                                progress=retry_progress,
                            )
                        elif use_multi_env_batches:
                            retry_batches = [
                                {"jobs": chunk}
                                for chunk in chunk_jobs(retry_jobs, multi_n)
                            ]
                            retry_results = _flatten_batch_results(
                                retry_pool.imap_unordered(
                                    worker_self_play_multi, retry_batches
                                )
                            )
                        else:
                            retry_results = retry_pool.imap_unordered(
                                worker_self_play, retry_jobs
                            )
                        _consume_results(
                            retry_results,
                            writer,
                            rows,
                            stats,
                            progress=retry_progress,
                            replay_cache=replay_cache,
                            replacement_spool=replacement_spool,
                            retained_job_indices=retained_self_play_indices,
                        )
                finally:
                    retry_progress.close()
                retry_promoted = _promote_replacement_spool(
                    replacement_spool,
                    missing_job_indices=missing_self_play,
                    writer=writer,
                    replay_cache=replay_cache,
                    primary_jobs=primary_self_play_jobs,
                )
                retry_promoted_indices = set(
                    retry_promoted["promoted_job_indices"]
                )
                retained_self_play_indices.update(retry_promoted_indices)
                rows.extend(retry_promoted["promoted_runtime_audit_rows"])
                stats["replacement_capacity_promoted_source_games"] += int(
                    retry_promoted["promoted_source_games"]
                )
                stats["replacement_capacity_promoted_trajectories"] += int(
                    retry_promoted["promoted_trajectories"]
                )
                stats["replacement_capacity_promoted_decisions"] += int(
                    retry_promoted["promoted_decisions"]
                )
                stats["with_record"] += int(
                    retry_promoted["promoted_source_games"]
                )
                stats["trajectories_written"] = int(
                    stats.get("trajectories_written", 0)
                ) + int(retry_promoted["promoted_trajectories"])
                missing_self_play = (
                    primary_self_play_indices - retained_self_play_indices
                )
            stats["retained_self_play_source_games"] = len(
                retained_self_play_indices
            )
            stats["self_play_attempts_with_record"] = int(
                stats.get("self_play", 0)
            )
            stats["self_play"] = len(retained_self_play_indices)
            if missing_self_play:
                if replay_cache is not None:
                    replay_cache.abort()
                raise RuntimeError(
                    "exact self-play retention failed after bounded replacements: "
                    f"retained={len(retained_self_play_indices)}/"
                    f"{len(primary_self_play_indices)} "
                    f"missing_job_indices={sorted(missing_self_play)[:16]}"
                )
    # Public agents use the cg.game singleton, so each phase stays one game per
    # process. Exact research controls are run separately and never reach this
    # replay writer.
    baseline_workers = max(1, int(n_workers))

    def _run_baseline_phase(
        jobs: list[dict[str, Any]],
        *,
        stage: str,
        phase_live_wr_gate: Optional[tuple[float, int]] = None,
    ) -> None:
        if not jobs:
            return
        collection_job_contracts = {}
        for job in jobs:
            provenance = dict(job.get("target_provenance") or {})
            collection_job_contracts[int(job.get("job_index", -1))] = {
                "opponent_content_digest": str(
                    provenance.get("opponent_content_digest") or ""
                ),
                "opponent_training_group": str(
                    provenance.get("opponent_training_group") or ""
                ),
                "formal_eval": provenance.get("formal_eval"),
                "training_eligible": job.get("training_eligible") is True,
            }
        progress = _TqdmProgress(
            stage=stage,
            iteration=int(iteration),
            total=len(jobs),
            remotes=remotes,
        )
        try:
            with WorkerPool(
                num_workers=baseline_workers, remote_channel=leaf_channel
            ) as pool:
                if use_remotes:
                    results_iter = _additive_iter(
                        pool=pool,
                        local_fn=worker_play,
                        jobs=jobs,
                        kind="play",
                        baseline_workers=baseline_workers,
                        progress=progress,
                    )
                else:
                    results_iter = pool.imap_unordered(worker_play, jobs)
                _consume_results(
                    results_iter,
                    writer,
                    rows,
                    stats,
                    progress=progress,
                    live_wr_gate=phase_live_wr_gate,
                    replay_cache=replay_cache,
                    required_checkpoint_digest=required_checkpoint_digest,
                    live_wr_opponent_ids=live_wr_opponent_ids,
                    practice_record_contracts=practice_record_contracts,
                    practice_seen_indices=practice_seen_indices,
                    practice_successful_indices=practice_successful_indices,
                    practice_written_indices=practice_written_indices,
                    replacement_spool=(replacement_spool if replay_eligible else None),
                    retained_job_indices=(
                        retained_public_indices if replay_eligible else None
                    ),
                    collection_job_contracts=collection_job_contracts,
                )
        finally:
            progress.close()

    _run_baseline_phase(
        dispatch_baseline_jobs,
        stage=stage_label or "collect:public_mix",
        phase_live_wr_gate=live_wr_gate,
    )
    if replay_eligible:
        missing_public = {
            int(job.get("job_index", -1)) for job in baseline_jobs
        } - retained_public_indices
        next_public_retry_index = (
            max(
                [
                    int(job.get("job_index", -1))
                    for job in [*self_play_jobs, *baseline_jobs]
                ],
                default=-1,
            )
            + 1
            + int(stats.get("targeted_self_play_retry_attempts", 0))
        )
        for retry_round in range(_PUBLIC_MIX_TARGETED_REPLACEMENT_ROUNDS):
            if not missing_public:
                break
            retry_jobs = _targeted_replacement_jobs(
                baseline_jobs,
                missing_job_indices=missing_public,
                retry_round=retry_round,
                first_job_index=next_public_retry_index,
            )
            next_public_retry_index += len(retry_jobs)
            for retry_job in retry_jobs:
                source_index = int(
                    (retry_job.get("target_provenance") or {}).get(
                        "replacement_for_job_index", -1
                    )
                )
                if source_index in practice_record_contracts:
                    practice_record_contracts[
                        int(retry_job["job_index"])
                    ] = dict(practice_record_contracts[source_index])
            stats["targeted_public_mix_retry_attempts"] = int(
                stats.get("targeted_public_mix_retry_attempts", 0)
            ) + len(retry_jobs)
            _run_baseline_phase(
                retry_jobs,
                stage="collect:public_mix_refill",
                phase_live_wr_gate=None,
            )
            promoted = _promote_replacement_spool(
                replacement_spool,
                missing_job_indices=missing_public,
                writer=writer,
                replay_cache=replay_cache,
                primary_jobs=baseline_jobs,
            )
            promoted_indices = set(promoted["promoted_job_indices"])
            retained_public_indices.update(promoted_indices)
            rows.extend(promoted["promoted_runtime_audit_rows"])
            stats["replacement_capacity_promoted_source_games"] = int(
                stats.get("replacement_capacity_promoted_source_games", 0)
            ) + int(promoted["promoted_source_games"])
            stats["replacement_capacity_promoted_trajectories"] = int(
                stats.get("replacement_capacity_promoted_trajectories", 0)
            ) + int(promoted["promoted_trajectories"])
            stats["replacement_capacity_promoted_decisions"] = int(
                stats.get("replacement_capacity_promoted_decisions", 0)
            ) + int(promoted["promoted_decisions"])
            stats["with_record"] += int(promoted["promoted_source_games"])
            stats["trajectories_written"] = int(
                stats.get("trajectories_written", 0)
            ) + int(promoted["promoted_trajectories"])
            missing_public = {
                int(job.get("job_index", -1)) for job in baseline_jobs
            } - retained_public_indices
        stats["retained_public_mix_source_games"] = len(retained_public_indices)
        expected_retained = len(primary_self_play_jobs) + len(baseline_jobs)
        if (
            len(retained_self_play_indices) != len(primary_self_play_jobs)
            or len(retained_public_indices) != len(baseline_jobs)
            or int(stats.get("with_record", 0)) != expected_retained
        ):
            if replay_cache is not None:
                replay_cache.abort()
            replacement_spool.unlink(missing_ok=True)
            raise RuntimeError(
                "exact collection contract failed: "
                f"self_play={len(retained_self_play_indices)}/"
                f"{len(primary_self_play_jobs)} "
                f"public_mix={len(retained_public_indices)}/"
                f"{len(baseline_jobs)} "
                f"retained={stats.get('with_record', 0)}/{expected_retained} "
                f"missing_public_job_indices={sorted(missing_public)[:16]} "
                f"targeted_public_mix_retry_attempts="
                f"{stats.get('targeted_public_mix_retry_attempts', 0)}"
            )
    if practice_record_contracts:
        expected_indices = set(practice_record_contracts)
        missing_results = sorted(expected_indices - practice_seen_indices)
        unexpected_results = sorted(practice_seen_indices - expected_indices)
        missing_records = sorted(
            practice_successful_indices - practice_written_indices
        )
        unexpected_records = sorted(
            practice_written_indices - practice_successful_indices
        )
        receipt_passed = not (
            missing_results
            or unexpected_results
            or missing_records
            or unexpected_records
        )
        stats["strong_public_practice_record_receipt"] = {
            "schema": "poke_bot.strong_public_practice_record_receipt/v1",
            "expected_results": len(expected_indices),
            "seen_results": len(practice_seen_indices),
            "successful_results": len(practice_successful_indices),
            "failed_results": len(expected_indices - practice_successful_indices),
            "canonical_records_written": len(practice_written_indices),
            "stale_records_repaired": int(
                stats.get("strong_public_practice_records_repaired", 0)
            ),
            "missing_result_job_indexes": missing_results,
            "unexpected_result_job_indexes": unexpected_results,
            "missing_record_job_indexes": missing_records,
            "unexpected_record_job_indexes": unexpected_records,
            "passed": receipt_passed,
        }
        if not receipt_passed:
            if replay_cache is not None:
                replay_cache.abort()
            raise RuntimeError(
                "strong-public practice record receipt failed: "
                f"missing_results={missing_results[:8]} "
                f"unexpected_results={unexpected_results[:8]} "
                f"missing_records={missing_records[:8]} "
                f"unexpected_records={unexpected_records[:8]}"
            )
    strict_remote_execution = str(
        os.environ.get("POKEBOT_REMOTE_REQUIRE_ALL", "0")
    ).strip().lower() in ("1", "true", "yes", "on") or str(
        os.environ.get("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "0")
    ).strip().lower() in ("1", "true", "yes", "on")
    if dispatch_self_play_jobs and use_remotes:
        expected_endpoints = sorted(
            {str(client.endpoint) for client in remote_farm.clients}
        )
        self_play_seen = dict(stats["remote_self_play_endpoint_counts"])
        missing_endpoints = [
            endpoint
            for endpoint in expected_endpoints
            if int(self_play_seen.get(endpoint, 0)) <= 0
        ]
        fallback_count = int(
            stats["execution_origin_counts"].get("local_fallback", 0)
        )
        stats["remote_execution_proof"] = {
            "strict": strict_remote_execution,
            "expected_endpoints": expected_endpoints,
            "self_play_completions_by_endpoint": self_play_seen,
            "missing_endpoints": missing_endpoints,
            "local_fallback_count": fallback_count,
        }
        if strict_remote_execution and (missing_endpoints or fallback_count):
            if replay_cache is not None:
                replay_cache.abort()
            raise RuntimeError(
                "strict remote execution proof failed: "
                f"missing_self_play_endpoints={missing_endpoints} "
                f"local_fallback_count={fallback_count}"
            )
    valid_runtime_rows = [row for row in rows if not bool(row.get("invalid"))]
    stall_rows = [row for row in valid_runtime_rows if row.get("stall_terminated")]
    stall_rows_by_seat = Counter(
        str(int(row.get("our_seat", -1)))
        for row in stall_rows
        if int(row.get("our_seat", -1)) in (0, 1)
    )
    # Same-checkpoint self-play retains both acting seats from one source
    # game.  The source row's assigned seat plus its opposite seat therefore
    # each receive the explicit -1 override; public-mix rows retain only the
    # scheduled acting seat.
    for row in stall_rows:
        if bool(row.get("self_play")):
            assigned = int(row.get("our_seat", -1))
            if assigned in (0, 1):
                stall_rows_by_seat[str(1 - assigned)] += 1
    ordinary_results = Counter(
        str(int(row["winner"]))
        for row in valid_runtime_rows
        if not row.get("stall_terminated") and row.get("winner") is not None
    )
    stagnant_turn_distribution = Counter(
        str(int(row.get("stall_turns") or 0)) for row in stall_rows
    )
    stats["no_progress_stall_receipt"] = {
        "maximum_complete_turns_without_win_progress": int(
            os.environ.get("PURE_RL_NO_PROGRESS_MAX_TURNS", "0") or 0
        ),
        "stall_games": len(stall_rows),
        "stall_rows_by_seat": dict(sorted(stall_rows_by_seat.items())),
        "stagnant_turn_distribution": dict(
            sorted(stagnant_turn_distribution.items())
        ),
        "training_return_override_counts": {
            "-1.0": sum(stall_rows_by_seat.values())
        },
        "ordinary_completed_result_counts": dict(sorted(ordinary_results.items())),
        "disabled_path_parity": not bool(
            int(os.environ.get("PURE_RL_NO_PROGRESS_MAX_TURNS", "0") or 0)
        ),
    }
    stats["terminal_policy_failure_retained_source_games"] = sum(
        1
        for row in valid_runtime_rows
        if bool(row.get("policy_terminal_failure"))
    )
    stats["matchup_runtime"] = _summarize_matchup_runtime_rows(
        valid_runtime_rows
    )
    self_play_runtime_rows = [
        row for row in valid_runtime_rows if bool(row.get("self_play"))
    ]
    stats["matchup_runtime_self_play"] = _summarize_matchup_runtime_rows(
        self_play_runtime_rows
    )
    stats["opponent_matchup_runtime_self_play"] = (
        _summarize_matchup_runtime_rows(
            self_play_runtime_rows,
            field="opponent_matchup_runtime_audit",
        )
    )
    stats["matchup_runtime_public_mix"] = _summarize_matchup_runtime_rows(
        [row for row in valid_runtime_rows if not bool(row.get("self_play"))]
    )
    stats["opponent_matchup_runtime"] = _summarize_matchup_runtime_rows(
        valid_runtime_rows, field="opponent_matchup_runtime_audit"
    )
    runtime_required = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    runtime_audit = dict(stats["matchup_runtime"])
    mirror_runtime_audit = _combine_self_play_matchup_runtime_audits(
        stats["matchup_runtime_self_play"],
        stats["opponent_matchup_runtime_self_play"],
    )
    enforcement = _matchup_runtime_collection_enforcement(
        runtime_audit,
        valid_games=len(valid_runtime_rows),
        required=runtime_required,
        self_play_audit=mirror_runtime_audit,
        required_mirror_archetype=required_mirror_archetype,
    )
    stats["matchup_runtime_enforcement"] = enforcement
    if enforcement["passed"] is not True:
        if replay_cache is not None:
            replay_cache.abort()
        raise RuntimeError(
            "activated matchup runtime collection audit failed before training: "
            f"{enforcement['assertions']}"
        )
    # The normal collection path may opt into bounded sequential shard writes.
    # Seal the last complete prefix before cache finalization or any caller
    # receives the writer for shard hashing/receipt construction.
    writer.flush()
    stats["compact_shard_write_flushes"] = int(writer.flush_count)
    stats["compact_shard_write_pending_bytes"] = int(writer.pending_bytes)
    if replay_cache is not None:
        # A cache failure is optimization-only: finish abandons its staging
        # directory and the normal post-collect builder retries fail-closed.
        replay_cache.note_append()
        replay_cache.finish()
    return writer, rows, stats


def _remote_heldout_capability_audit(
    remote_farm: Any,
    *,
    required_endpoints: Optional[list[str]] = None,
) -> dict[str, Any]:
    required = {
        "greedy_play_v1",
        "active_checkpoint_job_barrier_v1",
        "play_result_contract_v1",
        "portable_baseline_spec_v1",
    }
    endpoints: list[dict[str, Any]] = []
    for client in list(getattr(remote_farm, "clients", ()) or ()):
        info = getattr(client, "info", None)
        capabilities = set(getattr(info, "capabilities", ()) or ())
        kinds = set(getattr(info, "job_kinds", ()) or ())
        missing = sorted(required - capabilities)
        if "play" not in kinds:
            missing.append("job_kind:play")
        endpoints.append(
            {
                "endpoint": str(getattr(client, "endpoint", "unknown")),
                "capabilities": sorted(capabilities),
                "missing": missing,
                "ok": not missing,
            }
        )
    connected = {str(row["endpoint"]) for row in endpoints}
    required_endpoints = [
        str(value).strip() for value in (required_endpoints or ()) if str(value).strip()
    ]
    missing_endpoints = sorted(set(required_endpoints) - connected)
    return {
        "required": sorted(required),
        "endpoints": endpoints,
        "required_endpoints": required_endpoints,
        "missing_endpoints": missing_endpoints,
        "passed": (
            bool(endpoints)
            and not missing_endpoints
            and all(row["ok"] for row in endpoints)
        ),
    }


def _summarize_matchup_runtime_rows(
    rows: list[dict[str, Any]],
    *,
    field: str = "matchup_runtime_audit",
) -> dict[str, Any]:
    """Aggregate per-game causal-router receipts without affecting a gate.

    The router exposes one scalar ``model_route`` per decision, so a valid
    final snapshot can have at most one active adapter.  Runtime activation is
    audited separately from game validity: callers can inspect this receipt or
    enforce it at an explicit deployment boundary without silently changing an
    established win-rate gate.
    """
    from collections import Counter

    schema_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    final_route_counts: Counter[str] = Counter()
    active_archetype_counts: Counter[str] = Counter()
    per_route_observations: Counter[str] = Counter()
    per_archetype_observations: Counter[str] = Counter()
    tree_digests: Counter[str] = Counter()
    accepted_rosters: Counter[str] = Counter()
    audited = 0
    missing = 0
    malformed = 0
    runtime_enabled_games = 0
    runtime_disabled_games = 0
    observations = 0
    recognized_observations = 0
    zero_observation_games = 0
    route_transitions = 0
    initial_bypass_violations = 0
    transition_contract_violations = 0
    unexpected_active_routes: Counter[str] = Counter()

    for row in rows:
        audit = row.get(field)
        if not isinstance(audit, dict):
            missing += 1
            continue
        audited += 1
        schema_counts[str(audit.get("schema") or "missing")] += 1
        mode_counts[str(audit.get("mode") or "missing")] += 1
        enabled = audit.get("runtime_enabled")
        if enabled is True:
            runtime_enabled_games += 1
        elif enabled is False:
            runtime_disabled_games += 1
        else:
            malformed += 1
        try:
            route = int(audit.get("model_route"))
            obs = int(audit.get("observations"))
            recognized = int(audit.get("recognized_observations"))
        except (TypeError, ValueError):
            malformed += 1
            continue
        if obs < 0 or recognized < 0 or recognized > obs:
            malformed += 1
            continue
        observations += obs
        recognized_observations += recognized
        zero_observation_games += int(obs == 0)
        final_route_counts[str(route)] += 1
        active_archetype = audit.get("active_archetype_id")
        if active_archetype is not None:
            active_archetype_counts[str(active_archetype)] += 1
        digest = str(audit.get("tree_digest") or "")
        if digest:
            tree_digests[digest] += 1
        accepted = tuple(
            sorted(str(value) for value in audit.get("accepted_archetype_ids") or ())
        )
        if accepted:
            accepted_rosters["|".join(accepted)] += 1
        accepted_routes = {
            int(value)
            for value in dict(audit.get("accepted_routes") or {}).values()
            if isinstance(value, int)
        }
        if enabled is True and route >= 0 and route not in accepted_routes:
            unexpected_active_routes[str(route)] += 1
        if enabled is True:
            try:
                initial_route = int(audit.get("initial_model_route"))
            except (TypeError, ValueError):
                initial_route = 0
                initial_bypass_violations += 1
            else:
                initial_bypass_violations += int(initial_route != -1)
            transitions = audit.get("route_transitions")
            if not isinstance(transitions, list):
                malformed += 1
                transition_contract_violations += 1
            else:
                prior_observation = 0
                prior_route = initial_route
                for transition in transitions:
                    if not isinstance(transition, dict):
                        transition_contract_violations += 1
                        continue
                    try:
                        at = int(transition.get("observation"))
                        from_route = int(transition.get("from_route"))
                        to_route = int(transition.get("to_route"))
                    except (TypeError, ValueError):
                        transition_contract_violations += 1
                        continue
                    if (
                        at <= prior_observation
                        or at > obs
                        or from_route != prior_route
                        or from_route == to_route
                        or (to_route >= 0 and to_route not in accepted_routes)
                    ):
                        transition_contract_violations += 1
                    prior_observation = at
                    prior_route = to_route
                route_transitions += len(transitions)
                if (
                    audit.get("route_transitions_truncated") is not True
                    and prior_route != route
                ):
                    transition_contract_violations += 1
                try:
                    declared_count = int(audit.get("route_transition_count"))
                except (TypeError, ValueError):
                    transition_contract_violations += 1
                else:
                    transition_contract_violations += int(
                        declared_count != len(transitions)
                    )
        route_counts = audit.get("per_route") or {}
        if not isinstance(route_counts, dict):
            malformed += 1
            continue
        for route_id, count in route_counts.items():
            try:
                count_int = int(count)
            except (TypeError, ValueError):
                malformed += 1
                continue
            if count_int < 0:
                malformed += 1
                continue
            per_route_observations[str(route_id)] += count_int
            accepted_route_map = {
                str(archetype_id): int(accepted_route)
                for archetype_id, accepted_route in dict(
                    audit.get("accepted_routes") or {}
                ).items()
                if isinstance(accepted_route, int)
            }
            for archetype_id, accepted_route in accepted_route_map.items():
                if str(accepted_route) == str(route_id):
                    per_archetype_observations[archetype_id] += count_int

    return {
        "schema": "poke_bot.matchup_runtime_collection_audit/v1",
        "field": str(field),
        "games": len(rows),
        "audited_games": audited,
        "missing_games": missing,
        "malformed_games": malformed,
        "runtime_enabled_games": runtime_enabled_games,
        "runtime_disabled_games": runtime_disabled_games,
        "active_final_route_games": sum(
            count for route, count in final_route_counts.items() if int(route) >= 0
        ),
        "exact_bypass_final_games": int(final_route_counts.get("-1", 0)),
        "zero_observation_games": zero_observation_games,
        "route_transitions": route_transitions,
        "initial_bypass_violations": initial_bypass_violations,
        "transition_contract_violations": transition_contract_violations,
        "observations": observations,
        "recognized_observations": recognized_observations,
        "schema_counts": dict(sorted(schema_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "final_route_counts": dict(sorted(final_route_counts.items())),
        "active_archetype_counts": dict(sorted(active_archetype_counts.items())),
        "per_route_observations": dict(sorted(per_route_observations.items())),
        "per_archetype_observations": dict(
            sorted(per_archetype_observations.items())
        ),
        "tree_digest_counts": dict(sorted(tree_digests.items())),
        "accepted_roster_counts": dict(sorted(accepted_rosters.items())),
        "unexpected_active_routes": dict(sorted(unexpected_active_routes.items())),
        "all_games_audited": audited == len(rows),
        "all_runtime_enabled": bool(rows) and runtime_enabled_games == len(rows),
        "contract_clean": (
            malformed == 0
            and not unexpected_active_routes
            and initial_bypass_violations == 0
            and transition_contract_violations == 0
        ),
    }


def _matchup_runtime_collection_enforcement(
    audit: dict[str, Any],
    *,
    valid_games: int,
    required: bool,
    self_play_audit: Optional[dict[str, Any]] = None,
    required_mirror_archetype: Optional[str] = None,
) -> dict[str, Any]:
    """Build the explicit collect-before-train runtime-routing gate."""

    mirror = str(required_mirror_archetype or "").strip().casefold()
    expected_tree_digest: Optional[str] = None
    expected_roster_key: Optional[str] = None
    if required:
        runtime_tree_path = os.environ.get(
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH", ""
        ).strip()
        if not runtime_tree_path:
            raise RuntimeError(
                "activated matchup runtime lacks its configured tree path"
            )
        runtime_tree = Path(runtime_tree_path).expanduser().resolve()
        try:
            runtime_tree_payload = json.loads(
                runtime_tree.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "activated matchup runtime tree is unreadable"
            ) from exc
        expected_tree_digest = _sha256_file(runtime_tree)
        expected_roster = sorted(
            str(value)
            for value in dict(
                runtime_tree_payload.get("runtime_contract") or {}
            ).get("accepted_archetype_ids", ())
        )
        if not expected_roster:
            raise RuntimeError(
                "activated matchup runtime tree has no accepted routes"
            )
        expected_roster_key = "|".join(expected_roster)
    mirror_observations = int(
        dict((self_play_audit or {}).get("per_archetype_observations") or {}).get(
            mirror, 0
        )
        if mirror
        else 0
    )
    assertions = {
        "has_valid_games": int(valid_games) > 0,
        "all_valid_games_audited": audit.get("all_games_audited") is True,
        "all_valid_games_runtime_enabled": audit.get("all_runtime_enabled") is True,
        "contract_clean": audit.get("contract_clean") is True,
        "every_valid_game_observed": int(audit.get("zero_observation_games") or 0)
        == 0,
        "one_tree_identity": len(dict(audit.get("tree_digest_counts") or {}))
        == 1,
        "one_accepted_route_roster": len(
            dict(audit.get("accepted_roster_counts") or {})
        )
        == 1,
        "configured_tree_identity_only": (
            not required
            or set(dict(audit.get("tree_digest_counts") or {}))
            == {expected_tree_digest}
        ),
        "configured_accepted_route_roster_only": (
            not required
            or set(dict(audit.get("accepted_roster_counts") or {}))
            == {expected_roster_key}
        ),
        "active_specialist_mirror_route_observed": (
            not mirror or mirror_observations >= 1
        ),
    }
    return {
        "schema": "poke_bot.matchup_runtime_collection_enforcement/v1",
        "required": bool(required),
        "required_mirror_archetype": mirror or None,
        "expected_tree_digest": expected_tree_digest,
        "expected_accepted_route_roster": expected_roster_key,
        "mirror_route_observations": mirror_observations,
        "assertions": assertions,
        "passed": (not required) or all(assertions.values()),
    }


def _combine_self_play_matchup_runtime_audits(
    acting_seat_audit: dict[str, Any],
    opponent_seat_audit: dict[str, Any],
) -> dict[str, Any]:
    """Combine route observations from both independently audited mirror seats."""

    combined = dict(acting_seat_audit)
    observations = {
        str(archetype_id): int(count)
        for archetype_id, count in dict(
            acting_seat_audit.get("per_archetype_observations") or {}
        ).items()
    }
    for archetype_id, count in dict(
        opponent_seat_audit.get("per_archetype_observations") or {}
    ).items():
        key = str(archetype_id)
        observations[key] = int(observations.get(key, 0)) + int(count)
    combined["per_archetype_observations"] = observations
    return combined


def _audit_heldout_rows(
    rows: list[dict[str, Any]],
    *,
    n_games: int,
    checkpoint_digest: str,
    opponent_ids: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    from collections import Counter

    ordered_opponents = tuple(opponent_ids or OFFICIAL_BASELINE_IDS)
    if not ordered_opponents or len(set(ordered_opponents)) != len(ordered_opponents):
        raise ValueError("heldout opponent IDs must be unique and non-empty")
    if int(n_games) % (2 * len(ordered_opponents)):
        raise ValueError(
            "heldout games must divide evenly across opponents and both seats"
        )

    expected_ids = set(range(int(n_games)))
    counts = Counter(
        int(row["job_index"])
        for row in rows
        if isinstance(row.get("job_index"), int)
    )
    duplicates = sorted(job_id for job_id, count in counts.items() if count != 1)
    missing = sorted(expected_ids - set(counts))
    unexpected = sorted(set(counts) - expected_ids)
    for row in rows:
        job_id = row.get("job_index")
        if not isinstance(job_id, int) or counts.get(job_id, 0) != 1 or job_id not in expected_ids:
            row["invalid"] = True
            row["heldout_contract_invalid"] = True

    valid = [row for row in rows if not bool(row.get("invalid"))]
    per_opponent: dict[str, dict[str, int]] = {}
    for opponent in ordered_opponents:
        selected = [row for row in valid if row.get("opponent_id") == opponent]
        per_opponent[opponent] = {
            "games": len(selected),
            "seat0": sum(int(row.get("our_seat") == 0) for row in selected),
            "seat1": sum(int(row.get("our_seat") == 1) for row in selected),
        }
    expected_per_opponent = int(n_games) // len(ordered_opponents)
    exact_distribution = all(
        cell["games"] == expected_per_opponent
        and cell["seat0"] == expected_per_opponent // 2
        and cell["seat1"] == expected_per_opponent // 2
        for cell in per_opponent.values()
    )
    exact_weights = all(
        str(row.get("checkpoint_digest") or "") == str(checkpoint_digest)
        and str(row.get("action_selection") or "") == "greedy"
        for row in rows
    )
    passed = bool(
        len(rows) == int(n_games)
        and len(valid) == int(n_games)
        and not duplicates
        and not missing
        and not unexpected
        and exact_distribution
        and exact_weights
    )
    return {
        "passed": passed,
        "requested_games": int(n_games),
        "rows": len(rows),
        "valid_games": len(valid),
        "checkpoint_digest": str(checkpoint_digest),
        "greedy_required": True,
        "duplicate_job_ids": duplicates,
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "per_opponent": per_opponent,
        "expected_games_per_opponent": expected_per_opponent,
        "exact_distribution": exact_distribution,
        "exact_weights": exact_weights,
        "matchup_runtime": _summarize_matchup_runtime_rows(valid),
    }


def _heldout_eval(
    *,
    ckpt: Path,
    digest: str,
    n_games: int,
    decks: list[tuple[str, list[int]]],
    official_specs: list,
    seed: int,
    game_timeout_s: int,
    n_workers: int,
    leaf_channel,
    remote_farm,
    worker_play,
    worker_self_play,
    mode: str,
    allow_remote_play: bool = False,
    iteration: int = 0,
    gate_wr: float = 0.70,
    opponent_ids: Optional[tuple[str, ...]] = None,
    stage_label: str = "heldout",
    owner_deferred: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy gate against official bots on a disjoint, balanced seed schedule.

    Streams a live win-rate readout onto the tqdm bar as games land, and
    labels the bar ``heldout`` (not ``collect:public_mix``) so it reads
    clearly as the gate check rather than a regular training collect wave.
    Specialist target training may use the same public opponent policies, but
    never these greedy jobs or their seed range.
    """
    if owner_deferred:
        return [], {
            "passed": False,
            "owner_deferred": True,
            "reason": "owner_r284_moved_update_0_holdout_to_iteration_1",
            "requested_games": int(n_games),
            "rows": 0,
            "valid_games": 0,
            "checkpoint_digest": str(digest),
            "greedy_required": True,
            "per_opponent": {},
            "exact_distribution": False,
            "exact_weights": False,
            "partial_results_gate_eligible": False,
        }
    _self_jobs, base_jobs = _build_collect_jobs(
        n_games=n_games,
        ckpt=ckpt,
        digest=digest,
        model_generation=0,
        decks=decks,
        specs=official_specs,
        seed=seed,
        game_timeout_s=game_timeout_s,
        mode=mode,
        self_play_frac=0.0,
        balanced_eval=True,
    )
    for job in base_jobs:
        job["training_eligible"] = False
        job["agent_mode"] = "policy"
        job["sample_actions"] = False
        job["greedy"] = True
        job["action_temperature"] = 1.0
    shard = paths.OUTPUTS_DIR / "pure_rl" / "_heldout_tmp" / f"{os.getpid()}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    try:
        _writer, rows, _stats = _collect_wave(
            self_play_jobs=[],
            baseline_jobs=base_jobs,
            shard_path=shard,
            n_workers=n_workers,
            leaf_channel=leaf_channel,
            remote_farm=remote_farm,
            worker_play=worker_play,
            worker_self_play=worker_self_play,
            iteration=int(iteration),
            stage_label=str(stage_label),
            live_wr_gate=(float(gate_wr), int(n_games)),
            allow_remote_play=bool(allow_remote_play),
            required_checkpoint_digest=str(digest),
            live_wr_opponent_ids=opponent_ids,
            replay_eligible=False,
        )
    finally:
        try:
            shard.unlink(missing_ok=True)
        except Exception:
            pass
    audit = _audit_heldout_rows(
        rows,
        n_games=int(n_games),
        checkpoint_digest=str(digest),
        opponent_ids=opponent_ids,
    )
    return rows, audit


RESEARCH_CONTROL_RESULT_SCHEMA = "poke_bot.research_control_measurement_result/v1"


def _research_control_result_path(run_dir: Path, iteration: int) -> Path:
    return (
        Path(run_dir)
        / "research_controls"
        / f"iter_{int(iteration):05d}.json"
    )


def _validate_research_control_result(
    result: dict[str, Any],
    *,
    plan: dict[str, Any],
    registry: dict[str, Any],
    checkpoint_digest: str,
) -> dict[str, Any]:
    """Validate a complete immutable measurement result for safe reuse."""
    controls = list(registry.get("controls") or [])
    expected_ids = tuple(str(row.get("opponent_id") or "") for row in controls)
    rows = list(result.get("matchups") or [])
    by_id = {
        str(row.get("opponent_id") or ""): row
        for row in rows
        if isinstance(row, dict)
    }
    audit = dict(result.get("audit") or {})
    if (
        result.get("schema") != RESEARCH_CONTROL_RESULT_SCHEMA
        or int(result.get("iteration", -1)) != int(plan["iteration"])
        or str(result.get("registry_id") or "") != str(plan["registry_id"])
        or int(result.get("registry_version") or 0)
        != int(plan["registry_version"])
        or str(result.get("checkpoint_digest") or "")
        != str(checkpoint_digest)
        or str(result.get("schedule_digest") or "")
        != str(plan["schedule_digest"])
        or result.get("training_eligible") is not False
        or result.get("replay_eligible") is not False
        or result.get("diagnostic_only") is not True
        or result.get("included_in_gate_pass") is not False
        or float(result.get("gate_weight", -1.0)) != 0.0
        or result.get("formal_eval") is not False
        or str(result.get("action_selection") or "") != "greedy"
        or str(result.get("seed_namespace") or "")
        != "eval/research-controls-fixed-manifest-v1"
        or int(result.get("games", -1)) != int(plan["games"])
        or set(by_id) != set(expected_ids)
        or len(by_id) != len(rows)
        or audit.get("passed") is not True
        or audit.get("exact_distribution") is not True
        or audit.get("exact_weights") is not True
        or audit.get("seed_disjoint") is not True
        or audit.get("package_disjoint_from_active_gate") is not True
        or int(audit.get("replay_records_written", -1)) != 0
        or any(
            int(by_id[opponent_id].get("games", -1)) != 250
            or int(by_id[opponent_id].get("seat0", -1)) != 125
            or int(by_id[opponent_id].get("seat1", -1)) != 125
            or str(by_id[opponent_id].get("content_digest") or "")
            != str(plan["per_opponent"][opponent_id]["content_digest"])
            for opponent_id in expected_ids
        )
    ):
        raise RuntimeError("research-control result conflicts with its exact plan")
    return json.loads(json.dumps(result))


def _research_control_measurement(
    *,
    run_dir: Path,
    iteration: int,
    root_seed: int,
    n_games: int,
    training_games: int,
    formal_games: int,
    ckpt: Path,
    digest: str,
    decks: list[tuple[str, list[int]]],
    specs: list[Any],
    registry: dict[str, Any],
    active_gate_digests: set[str],
    game_timeout_s: int,
    n_workers: int,
    leaf_channel: Any,
    remote_farm: Any,
    worker_play: Any,
    worker_self_play: Any,
    mode: str,
    allow_remote_play: bool,
) -> dict[str, Any]:
    """Run or recover one exact additive research-control transaction."""
    seed = (
        int(root_seed)
        + RESEARCH_CONTROL_SEED_OFFSET
        + int(iteration) * ITERATION_SEED_STRIDE
    )
    jobs = _build_research_control_jobs(
        n_games=int(n_games),
        ckpt=Path(ckpt),
        digest=str(digest),
        model_generation=int(iteration) + 1,
        decks=decks,
        specs=specs,
        seed=seed,
        game_timeout_s=int(game_timeout_s),
        mode=str(mode),
        registry=registry,
        iteration=int(iteration),
    )
    training_seed_jobs = [
        {
            "seed": int(root_seed)
            + int(iteration) * ITERATION_SEED_STRIDE
            + index
        }
        for index in range(int(training_games))
    ]
    plan = _assert_research_control_jobs(
        jobs,
        expected_games=int(n_games),
        registry=registry,
        iteration=int(iteration),
        root_seed=int(root_seed),
        training_jobs=training_seed_jobs,
        formal_games=int(formal_games),
        checkpoint_digest=str(digest),
        active_gate_digests=set(active_gate_digests),
    )
    result_path = _research_control_result_path(run_dir, iteration)
    if result_path.is_file():
        recovered = _validate_research_control_result(
            json.loads(result_path.read_text(encoding="utf-8")),
            plan=plan,
            registry=registry,
            checkpoint_digest=str(digest),
        )
        print(
            "[pure_rl] research-control measurement recovered "
            f"iter={iteration} result={result_path}",
            flush=True,
        )
        return recovered

    temporary_shard = (
        paths.OUTPUTS_DIR
        / "pure_rl"
        / "_research_control_tmp"
        / f"{os.getpid()}-{int(iteration):05d}.jsonl"
    )
    temporary_shard.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer, rows, stats = _collect_wave(
            self_play_jobs=[],
            baseline_jobs=jobs,
            shard_path=temporary_shard,
            n_workers=int(n_workers),
            leaf_channel=leaf_channel,
            remote_farm=remote_farm,
            worker_play=worker_play,
            worker_self_play=worker_self_play,
            iteration=int(iteration),
            stage_label="measure:research_controls",
            live_wr_gate=(0.0, int(n_games)),
            allow_remote_play=bool(allow_remote_play),
            required_checkpoint_digest=str(digest),
            live_wr_opponent_ids=tuple(
                str(row["opponent_id"]) for row in registry["controls"]
            ),
            replay_eligible=False,
        )
        if writer.n_games != 0 or writer.n_decisions != 0 or int(
            stats.get("with_record", 0)
        ) != 0:
            raise RuntimeError(
                "research-control measurement produced replay/AWR records"
            )
    finally:
        temporary_shard.unlink(missing_ok=True)

    opponent_ids = tuple(str(row["opponent_id"]) for row in registry["controls"])
    row_audit = _audit_heldout_rows(
        rows,
        n_games=int(n_games),
        checkpoint_digest=str(digest),
        opponent_ids=opponent_ids,
    )
    if not bool(row_audit.get("passed")):
        raise RuntimeError("research-control measurement failed its exact row audit")
    valid = [row for row in rows if not bool(row.get("invalid"))]
    content_by_id = {
        str(row["opponent_id"]): str(row["content_digest"])
        for row in registry["controls"]
    }
    matchups: list[dict[str, Any]] = []
    total_wins = 0.0
    total_draws = 0
    total_losses = 0
    for opponent_id in opponent_ids:
        selected = [row for row in valid if row.get("opponent_id") == opponent_id]
        wins = sum(
            0.5
            if int(row.get("winner", 2)) == 2
            else 1.0
            if int(row.get("winner", 2)) == int(row.get("our_seat", 0))
            else 0.0
            for row in selected
        )
        draws = sum(int(row.get("winner", 2)) == 2 for row in selected)
        losses = len(selected) - int(round(wins - 0.5 * draws)) - draws
        total_wins += wins
        total_draws += draws
        total_losses += losses
        matchups.append(
            {
                "opponent_id": opponent_id,
                "content_digest": content_by_id[opponent_id],
                "games": len(selected),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "seat0": sum(int(row.get("our_seat", -1)) == 0 for row in selected),
                "seat1": sum(int(row.get("our_seat", -1)) == 1 for row in selected),
                "win_rate": wins / len(selected) if selected else None,
            }
        )
    result = {
        "schema": RESEARCH_CONTROL_RESULT_SCHEMA,
        "iteration": int(iteration),
        "registry_id": str(registry["registry_id"]),
        "registry_version": int(registry["version"]),
        "checkpoint": str(Path(ckpt).resolve()),
        "checkpoint_digest": str(digest),
        "schedule_digest": str(plan["schedule_digest"]),
        "seed_namespace": str(plan["seed_namespace"]),
        "seed_start": int(plan["seed_start"]),
        "training_eligible": False,
        "replay_eligible": False,
        "diagnostic_only": True,
        "included_in_gate_pass": False,
        "gate_weight": 0.0,
        "formal_eval": False,
        "action_selection": "greedy",
        "games": len(valid),
        "wins": total_wins,
        "draws": total_draws,
        "losses": total_losses,
        "win_rate": total_wins / len(valid) if valid else None,
        "matchups": matchups,
        "audit": {
            **row_audit,
            "seed_disjoint": True,
            "package_disjoint_from_active_gate": True,
            "replay_records_written": 0,
            "measurement_plan": plan,
        },
        "result_path": str(result_path.resolve()),
    }
    validated = _validate_research_control_result(
        result,
        plan=plan,
        registry=registry,
        checkpoint_digest=str(digest),
    )
    _write_json_exclusive(result_path, validated)
    return validated


def _promotion_eval(
    *,
    candidate,
    incumbent,
    decks: list[tuple[str, list[int]]],
    n_games: int,
    n_workers: int,
    threshold: float,
    confidence: float,
    bootstrap_resamples: int,
    seed: int,
    game_timeout_s: int,
    model_generation: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Seat/deck-balanced candidate-vs-incumbent non-regression gate."""
    from poke_bot.promotion import PromotionGateConfig, evaluate_candidate_gate
    from poke_bot.remote_sim_jobs import remote_promotion_job
    from poke_bot.worker_pool import WorkerPool

    n_games = int(n_games)
    if n_games < 4 or n_games % 2:
        raise ValueError("--promotion-games must be an even integer >= 4")
    if not decks:
        raise ValueError("promotion requires at least one deck")
    cfg = PromotionGateConfig(
        min_games=n_games,
        min_complete_pairs=n_games // 2,
        threshold=float(threshold),
        confidence=float(confidence),
        bootstrap_resamples=int(bootstrap_resamples),
    )
    cfg.validate()
    jobs: list[dict[str, Any]] = []
    for game_i in range(n_games):
        pair_i = game_i // 2
        _arch, deck = decks[pair_i % len(decks)]
        jobs.append(
            {
                "candidate_checkpoint": candidate.path,
                "parent_checkpoint": incumbent.path,
                "candidate_seat": game_i % 2,
                "cluster_id": pair_i,
                "seed": int(seed + game_i),
                "deck": list(deck),
                "device": "cpu",
                "agent_mode": "policy",
                "mcts_sims": 0,
                "mcts_move_time": 0.0,
                "candidate_digest": candidate.digest,
                "parent_digest": incumbent.digest,
                "model_generation": int(model_generation),
                "model_max_context": int(pure_rl_model_config().max_context),
                "timeout_s": int(game_timeout_s),
            }
        )
    def _job_identity(row: dict[str, Any]) -> tuple[int, int, int]:
        """Identity of one exact promotion game across transport retries."""

        return (
            int(row["seed"]),
            int(row["candidate_seat"]),
            int(row["cluster_id"]),
        )

    def _run_wave(
        wave: list[dict[str, Any]], *, worker_cap: int
    ) -> list[dict[str, Any]]:
        with WorkerPool(
            num_workers=max(1, min(int(worker_cap), len(wave)))
        ) as pool:
            return list(pool.imap_unordered(remote_promotion_job, wave))

    rows = _run_wave(jobs, worker_cap=int(n_workers))
    jobs_by_identity = {_job_identity(job): job for job in jobs}
    rows_by_identity = {_job_identity(row): row for row in rows}
    retry_audit: list[dict[str, Any]] = []

    # A promotion game is an evaluation identity, not a particular worker
    # process attempt.  Retry only failed exact identities, with the same seed,
    # deck, seats, checkpoints, and cluster.  The smaller recovery wave avoids
    # turning transient process/model-load pressure into a learner rollback;
    # valid results are never replayed or replaced.  Two bounded waves still
    # fail closed if the exact game cannot execute successfully.
    for retry_index in range(1, 3):
        invalid_identities = [
            identity
            for identity, row in rows_by_identity.items()
            if not row.get("valid", True)
        ]
        missing_identities = [
            identity
            for identity in jobs_by_identity
            if identity not in rows_by_identity
        ]
        retry_identities = sorted(
            set(invalid_identities) | set(missing_identities)
        )
        if not retry_identities:
            break
        retry_jobs = [jobs_by_identity[identity] for identity in retry_identities]
        retry_workers = max(1, min(8, int(n_workers), len(retry_jobs)))
        prior_errors = {
            str(identity): str(
                rows_by_identity.get(identity, {}).get("error")
                or "missing worker result"
            )
            for identity in retry_identities
        }
        retry_rows = _run_wave(retry_jobs, worker_cap=retry_workers)
        retry_rows_by_identity = {
            _job_identity(row): row for row in retry_rows
        }
        for identity in retry_identities:
            replacement = retry_rows_by_identity.get(identity)
            if replacement is None:
                original = jobs_by_identity[identity]
                replacement = {
                    "seed": int(original["seed"]),
                    "candidate_seat": int(original["candidate_seat"]),
                    "cluster_id": int(original["cluster_id"]),
                    "valid": False,
                    "winner": 2,
                    "error": "promotion recovery wave returned no result",
                }
            rows_by_identity[identity] = replacement
        remaining_invalid = sum(
            not rows_by_identity[identity].get("valid", True)
            for identity in retry_identities
        )
        retry_audit.append(
            {
                "retry_wave": retry_index,
                "exact_games_retried": len(retry_identities),
                "worker_cap": retry_workers,
                "prior_errors": prior_errors,
                "recovered_games": len(retry_identities) - remaining_invalid,
                "remaining_invalid_games": remaining_invalid,
            }
        )

    rows = [rows_by_identity[_job_identity(job)] for job in jobs]
    report = evaluate_candidate_gate(rows, cfg)
    report.update(
        {
            "candidate": candidate.as_dict(),
            "incumbent": incumbent.as_dict(),
            "deck_schedule": "paired_same_deck_both_candidate_seats",
            "search": False,
            "transport_recovery": {
                "schema": "poke_bot.promotion_exact_game_transport_recovery/v1",
                "same_seed_deck_seat_checkpoint_and_cluster_required": True,
                "valid_games_retried": 0,
                "maximum_retry_waves": 2,
                "waves": retry_audit,
            },
        }
    )
    return report, rows


def _gate_passed_in_history(
    state: dict[str, Any],
    *,
    gate_id: str,
    through_iteration: int,
) -> bool:
    """Return true only for an immutable recorded pass at/before a boundary."""

    for row in list(state.get("history") or []):
        if not isinstance(row, dict):
            continue
        if int(row.get("iteration", -1)) > int(through_iteration):
            continue
        result = row.get("active_gate_result")
        if (
            isinstance(result, dict)
            and str(result.get("gate_id") or "") == str(gate_id)
            and result.get("passed") is True
        ):
            return True
    return False


def _terminal_gate_target_matches(
    *,
    requested_gate_id: str,
    passed_gate_id: str,
    base_contract: dict[str, Any] | None,
) -> bool:
    if not requested_gate_id:
        return False
    if str(passed_gate_id) == str(requested_gate_id):
        return True
    fallback = (
        dict(base_contract.get("fallback_transition") or {})
        if isinstance(base_contract, dict)
        else {}
    )
    return bool(
        fallback
        and str(fallback.get("prior_gate_id") or "") == str(requested_gate_id)
        and str(fallback.get("id") or "") == str(passed_gate_id)
        and fallback.get("only_if_prior_gate_unpassed") is True
    )


def _design_contract_active_gate_id(contract: dict[str, Any]) -> str:
    """Resolve the gate identity pinned by one design-contract snapshot."""

    active = dict((contract.get("gates") or {}).get("active_contract") or {})
    raw_path = str(active.get("path") or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(
        (payload.get("next_gate") or {}).get("id")
        or payload.get("active_gate_id")
        or ""
    )


def _receipt_backed_gate_pointer_transition_authorized(
    *,
    run_dir: Path,
    existing_gate_id: str,
    active_gate_id: str,
    existing_iteration: int,
    iteration: int,
) -> bool:
    """Allow only an append-only, boundary-recorded gate-identity chain.

    A global result pointer is mutable presentation state.  Its historical
    result remains immutable in the old iteration commit, but a later owner
    gate upgrade must be able to publish the next committed result.  Accept
    neither an arbitrary pointer rewrite nor a merely similarly named gate:
    every identity hop must be present in the design-migration receipts between
    the two immutable commits.
    """

    if (
        not existing_gate_id
        or not active_gate_id
        or existing_gate_id == active_gate_id
        or existing_iteration >= iteration
    ):
        return False
    cursor = str(existing_gate_id)
    saw_transition = False
    migration_dir = Path(run_dir) / "design_migrations"
    for path in sorted(migration_dir.glob("migration_*.json")):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        boundary = int(receipt.get("boundary_next_iteration", -1))
        if not (existing_iteration < boundary <= iteration):
            continue
        changed = set(receipt.get("changed_paths") or ())
        if not any(
            item == "gates.active_contract.path"
            or item == "gates.active_contract.digest"
            for item in changed
        ):
            continue
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        if not isinstance(previous, dict) or not isinstance(current, dict):
            return False
        prior_id = _design_contract_active_gate_id(previous)
        next_id = _design_contract_active_gate_id(current)
        if not prior_id or not next_id:
            return False
        if prior_id == next_id:
            continue
        if cursor != prior_id:
            return False
        cursor = next_id
        saw_transition = True
    return saw_transition and cursor == str(active_gate_id)


def _publish_committed_active_gate_result(
    *,
    run_dir: Path,
    active_gate: dict[str, Any],
    result_pointer: Path,
) -> Optional[tuple[Path, Path]]:
    """Recover the mutable gate pointer from the latest immutable commit."""
    loop = _load_loop_state(run_dir)
    if not isinstance(loop, dict):
        return None
    iteration = int(loop.get("last_completed_iteration", -1))
    if iteration < 0:
        return None
    commit_path = (
        Path(run_dir) / "commits" / f"iter_{iteration:05d}.json"
    ).resolve()
    if not commit_path.is_file():
        raise RuntimeError("latest immutable iteration commit is missing")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    history = commit.get("history")
    row = (
        next(
            (
                value
                for value in reversed(history)
                if isinstance(value, dict)
                and int(value.get("iteration", -1)) == iteration
            ),
            None,
        )
        if isinstance(history, list)
        else None
    )
    result = (
        row.get("active_gate_result")
        if isinstance(row, dict)
        and isinstance(row.get("active_gate_result"), dict)
        else None
    )
    if result is None or str(result.get("gate_id") or "") != str(
        active_gate.get("id") or ""
    ):
        return None
    committed = {
        **dict(result),
        "committed": True,
        "commit": str(commit_path),
        "commit_digest": _canonical_digest(commit),
        "created_at_utc": str(
            commit.get("updated_at_utc")
            or datetime.fromtimestamp(
                commit_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        ),
    }
    result_pointer = Path(result_pointer).expanduser().resolve()
    existing = (
        json.loads(result_pointer.read_text(encoding="utf-8"))
        if result_pointer.is_file()
        else None
    )
    if isinstance(existing, dict):
        existing_iteration = int(existing.get("iteration", -1))
        raw_existing_commit = str(existing.get("commit") or "").strip()
        existing_commit = (
            Path(raw_existing_commit).expanduser().resolve()
            if raw_existing_commit
            else None
        )
        same_lineage = bool(
            existing_commit is not None
            and existing_commit.parent.name == "commits"
            and existing_commit.parent.parent == Path(run_dir).resolve()
        )
        same_gate = str(existing.get("gate_id") or "") == str(
            active_gate.get("id") or ""
        )
        roster_ids = {
            str(item.get("opponent_id") or "")
            for item in (active_gate.get("roster") or [])
        }
        existing_matchup_ids = {
            str(item.get("opponent_id") or "")
            for item in (existing.get("matchups") or [])
        }
        authorized_lc55_revision = bool(
            str(existing.get("gate_id") or "")
            == "alakazam-strong-public-roster-v1"
            and str(active_gate.get("id") or "")
            == "alakazam-strong-public-roster-lc55-v2"
            and len(roster_ids) == 8
            and existing_matchup_ids == roster_ids
            and int((active_gate.get("evaluation") or {}).get("games_total", 0))
            == 2000
            and int(
                (active_gate.get("evaluation") or {}).get(
                    "games_per_opponent", 0
                )
            )
            == 250
            and float(
                (active_gate.get("pass_criteria") or {}).get(
                    "skill_weighted_confidence_lower", 0.0
                )
            )
            == 0.55
            and existing_iteration < iteration
        )
        fallback_activation = dict(active_gate.get("activation") or {})
        fallback_prior_id = str(
            fallback_activation.get("prior_gate_id") or ""
        )
        fallback_active_id = str(active_gate.get("id") or "")
        fallback_games = int(
            (active_gate.get("evaluation") or {}).get("games_total", 0)
        )
        fallback_per_opponent = int(
            (active_gate.get("evaluation") or {}).get(
                "games_per_opponent", 0
            )
        )
        fallback_prior_lower = float(
            fallback_activation.get("prior_confidence_lower", -1.0)
        )
        fallback_active_lower = float(
            fallback_activation.get("active_confidence_lower", -1.0)
        )
        authorized_configured_lc50_fallback = bool(
            fallback_prior_id
            and fallback_active_id
            and fallback_prior_id != fallback_active_id
            and str(existing.get("gate_id") or "") == fallback_prior_id
            and fallback_activation.get("schema")
            == "poke_bot.iteration_gate_fallback_activation/v1"
            and int(
                fallback_activation.get(
                    "activate_after_completed_iteration", -1
                )
            )
            >= 0
            and fallback_activation.get("prior_gate_passed") is False
            and fallback_activation.get("only_changed_criterion")
            == "skill_weighted_confidence_lower"
            and 0.0 <= fallback_active_lower <= fallback_prior_lower <= 1.0
            and len(roster_ids) > 0
            and existing_matchup_ids == roster_ids
            and fallback_per_opponent == 250
            and fallback_games == len(roster_ids) * fallback_per_opponent
            and float(
                (active_gate.get("pass_criteria") or {}).get(
                    "skill_weighted_confidence_lower", -1.0
                )
            )
            == fallback_active_lower
            and existing_iteration >= 0
            and existing_iteration < iteration
        )
        authorized_receipt_backed_gate_transition = (
            _receipt_backed_gate_pointer_transition_authorized(
                run_dir=run_dir,
                existing_gate_id=str(existing.get("gate_id") or ""),
                active_gate_id=str(active_gate.get("id") or ""),
                existing_iteration=existing_iteration,
                iteration=iteration,
            )
        )
        if (
            same_lineage
            and not same_gate
            and not authorized_lc55_revision
            and not authorized_configured_lc50_fallback
            and not authorized_receipt_backed_gate_transition
        ):
            raise RuntimeError(
                "active-gate result pointer changes gate inside one lineage"
            )
        if same_lineage and existing_iteration > iteration:
            raise RuntimeError("active-gate result pointer is ahead of commit history")
        existing_core = {
            key: value
            for key, value in existing.items()
            if key
            not in {"committed", "commit", "commit_digest", "created_at_utc"}
        }
        if (
            existing_iteration == iteration
            and existing_core != result
            and (same_lineage or (not raw_existing_commit and same_gate))
        ):
            raise RuntimeError("active-gate result pointer conflicts with immutable commit")
        if existing == committed:
            return result_pointer, commit_path
    _atomic_json(result_pointer, committed)
    print(
        "[pure_rl] ACTIVE_GATE_RESULT_PUBLISHED "
        f"iter={iteration} gate={result.get('gate_id')} source={commit_path}",
        flush=True,
    )
    return result_pointer, commit_path


def _reconcile_passed_gate_research_controls(
    *,
    registry_path: Path,
    gate_contract: dict[str, Any],
    exact_result_path: Path,
    commit_path: Path,
    output_path: Path,
) -> Optional[dict[str, Any]]:
    """Durably retire a committed passed gate, and ignore incomplete attempts."""
    exact_result_path = Path(exact_result_path).expanduser().resolve()
    if not exact_result_path.is_file():
        return None
    result = json.loads(exact_result_path.read_text(encoding="utf-8"))
    if not (result.get("committed") is True and result.get("passed") is True):
        return None
    from poke_bot.pure_rl.research_controls import (
        load_research_control_registry,
        retire_passed_gate_file,
    )

    # The registry records a one-way agent-pool transition, not every
    # specialist that later passes the same roster.  A newer specialist may
    # therefore pass after the roster has already been retired by an earlier
    # lineage/gate revision.  Treat the fully applied, identity-exact state as
    # an idempotent success; partial or conflicting states still flow into the
    # strict retirement primitive and fail closed.
    destination = Path(output_path).expanduser().resolve()
    if destination.is_file():
        existing = load_research_control_registry(destination)
        active_gate = gate_contract.get("next_gate")
        roster = (
            active_gate.get("roster")
            if isinstance(active_gate, dict)
            else None
        )
        controls_by_id = {
            str(row.get("opponent_id") or ""): row
            for row in existing["controls"]
        }
        roster_already_retired = (
            isinstance(roster, list)
            and bool(roster)
            and all(isinstance(row, dict) for row in roster)
            and len(
                {
                    str(row.get("opponent_id") or "")
                    for row in roster
                }
            )
            == len(roster)
            and all(
                (
                    str(row.get("opponent_id") or "") in controls_by_id
                    and str(
                        controls_by_id[
                            str(row.get("opponent_id") or "")
                        ].get("content_digest")
                        or ""
                    )
                    == str(row.get("content_digest") or "")
                    and controls_by_id[
                        str(row.get("opponent_id") or "")
                    ].get("training_eligible")
                    is False
                    and controls_by_id[
                        str(row.get("opponent_id") or "")
                    ].get("included_in_gate_pass")
                    is False
                    and controls_by_id[
                        str(row.get("opponent_id") or "")
                    ].get("formal_eval")
                    is False
                    and float(
                        controls_by_id[
                            str(row.get("opponent_id") or "")
                        ].get("gate_weight", float("nan"))
                    )
                    == 0.0
                )
                for row in roster
            )
        )
        if roster_already_retired:
            print(
                "[pure_rl] RESEARCH_CONTROL_GATE_ALREADY_RETIRED "
                f"gate={result.get('gate_id')} "
                f"registry_version={existing['version']} "
                f"controls={len(existing['controls'])} path={destination}",
                flush=True,
            )
            return existing

    updated = retire_passed_gate_file(
        registry_path=registry_path,
        gate_contract=gate_contract,
        exact_result_path=exact_result_path,
        commit_path=commit_path,
        output_path=output_path,
    )
    print(
        "[pure_rl] RESEARCH_CONTROL_GATE_RETIRED "
        f"gate={result.get('gate_id')} registry_version={updated['version']} "
        f"controls={len(updated['controls'])} path={output_path}",
        flush=True,
    )
    return updated


def run_full_loop(args: argparse.Namespace) -> int:
    """Real CABT collect → AWR → held-out loop with optional remote farms."""
    import torch
    from poke_bot.baselines_runtime import (
        BaselineSpec,
        baseline_content_digest,
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from poke_bot.pure_rl.strong_public_gate import (
        build_active_gate_result,
        load_active_gate_contract,
        materialize_fallback_gate_contract,
        verify_roster_content,
    )
    from poke_bot.pure_rl.research_controls import (
        load_research_control_registry,
        pin_research_control_registry_file,
        research_control_ids,
        validate_research_control_registry,
    )
    from poke_bot.promotion import CheckpointIdentity
    from poke_bot.remote_jobs import RemoteWorkerFarm

    prize_plan_h3_actor_provider = _configured_prize_plan_h3_actor_provider(args)
    fixed_cycle_updates = _validate_fixed_cycle_configuration(args)
    r274_r195_research_contract = (
        _load_r274_r195_research_baseline_contract(args)
        if fixed_cycle_updates == R274_FIXED_CYCLE_UPDATES
        else {}
    )
    r241_peak_r195_training_contract = (
        _load_r241_peak_r195_training_contract(args)
        if fixed_cycle_updates
        else {}
    )
    derivative_full_model = os.environ.get(
        "POKEBOT_DERIVATIVE_FULL_MODEL_BOOTSTRAP", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    r260_own_deck_training_inputs = (
        _load_r260_own_deck_training_inputs(args)
        if fixed_cycle_updates or derivative_full_model
        else {}
    )
    if r260_own_deck_training_inputs.get("r274_bootstrap_handoff"):
        activated = dict(
            r260_own_deck_training_inputs["r274_bootstrap_handoff"].get(
                "activated_checkpoint"
            )
            or {}
        )
        os.environ["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] = (
            "0" if args.r284_iteration_boundary_receipt is not None else "1"
        )
        os.environ["POKEBOT_COMBO_STATE_ROUTE_SPECIALIST"] = "alakazam"
        os.environ["POKEBOT_COMBO_STATE_ROUTE_CHECKPOINT_DIGEST"] = str(
            activated.get("sha256") or activated.get("digest") or ""
        )
    disable_rl_tactical_cotrain = _r260_tactical_cotrain_disabled(
        r260_own_deck_training_inputs
    )
    # The peak-r195 receipt remains mandatory ancestry evidence for the fixed
    # cycle, but its serialized profile validator describes the pre-successor
    # architecture.  Once the independently validated r260/r274 handoff is
    # present, applying that legacy profile to descendant checkpoints would
    # incorrectly reject the OwnDeck/tactical architecture and its active
    # routes.  The successor evidence above owns runtime-profile validation.
    if r260_own_deck_training_inputs:
        r241_peak_r195_training_contract = {}
    if int(args.expert_rehearsal_every) < 0:
        raise ValueError("--expert-rehearsal-every cannot be negative")
    if int(args.expert_rehearsal_force_before) < -1:
        raise ValueError("--expert-rehearsal-force-before must be -1 or nonnegative")
    if (
        int(args.expert_rehearsal_every) > 0
        or int(args.expert_rehearsal_force_before) >= 0
    ) and args.expert_manifest is None:
        raise ValueError("--expert-manifest is required when expert rehearsal is enabled")
    if int(args.expert_rehearsal_epochs) <= 0:
        raise ValueError("--expert-rehearsal-epochs must be positive")
    if (
        args.expert_rehearsal_guide_loss_weight is not None
        and float(args.expert_rehearsal_guide_loss_weight) < 0.0
    ):
        raise ValueError(
            "--expert-rehearsal-guide-loss-weight cannot be negative"
        )
    if bool(args.terminal_expert_rehearsal) and (
        int(args.expert_rehearsal_every) <= 0
        or int(args.iterations) % int(args.expert_rehearsal_every) != 0
    ):
        raise ValueError(
            "terminal expert rehearsal requires a due positive cadence at "
            "the configured iteration count"
        )
    one_time_before = int(args.expert_rehearsal_one_time_before)
    one_time_epochs = int(args.expert_rehearsal_one_time_epochs)
    if one_time_before < -1:
        raise ValueError(
            "--expert-rehearsal-one-time-before must be -1 or nonnegative"
        )
    if (one_time_before == -1) != (one_time_epochs == 0):
        raise ValueError(
            "one-time expert rehearsal requires both an iteration and positive epochs"
        )
    if one_time_epochs < 0:
        raise ValueError("--expert-rehearsal-one-time-epochs cannot be negative")
    if float(args.expert_rehearsal_lr) <= 0.0:
        raise ValueError("--expert-rehearsal-lr must be positive")
    if int(args.expert_rehearsal_batch_size) <= 0:
        raise ValueError("--expert-rehearsal-batch-size must be positive")
    if args.rule_derivative_semantic_pack_completion is not None:
        if int(args.expert_rehearsal_every) <= 0:
            raise ValueError(
                "rule-derivative semantic refresh requires expert rehearsal"
            )
        if float(args.rule_derivative_semantic_refresh_lr) <= 0.0:
            raise ValueError(
                "--rule-derivative-semantic-refresh-lr must be positive"
            )
        if int(args.rule_derivative_semantic_block_decisions) <= 0:
            raise ValueError(
                "--rule-derivative-semantic-block-decisions must be positive"
            )
    if int(args.expert_manifest_workers) <= 0:
        raise ValueError("--expert-manifest-workers must be positive")
    if int(args.expert_min_decisions) <= 0:
        raise ValueError("--expert-min-decisions must be positive")
    expert_required_targets = tuple(
        args.expert_required_target or EXPERT_REHEARSAL_TARGETS
    )
    if (
        len(set(expert_required_targets)) != len(expert_required_targets)
        or "temporal_action_rows" not in expert_required_targets
    ):
        raise ValueError(
            "--expert-required-target values must be unique and include "
            "temporal_action_rows"
        )
    if bool(args.population_own_models_only):
        if args.population_opponent_registry is None:
            raise ValueError(
                "--population-opponent-registry is required for current + "
                "selected-history own-model round robin"
            )
        if int(args.iterations) != POPULATION_RL_EPOCHS_PER_CYCLE:
            raise ValueError(
                "each population member lineage must contain exactly 5 RL "
                "iterations; start a new lineage from the closing rehearsal "
                "checkpoint for the next population cycle"
            )
        population_cycle_rehearsal_due(
            population_enabled=True,
            next_iteration=int(args.iterations),
            configured_rehearsal_every=int(args.expert_rehearsal_every),
            configured_rehearsal_epochs=int(args.expert_rehearsal_epochs),
        )
    if args.expert_matchup_adapter_manifest is not None:
        if int(args.expert_rehearsal_every) <= 0:
            raise ValueError(
                "--expert-matchup-adapter-manifest requires a positive "
                "--expert-rehearsal-every cadence"
            )
        if int(args.expert_matchup_adapter_epochs) <= 0:
            raise ValueError("--expert-matchup-adapter-epochs must be positive")
        if float(args.expert_matchup_adapter_lr) <= 0.0:
            raise ValueError("--expert-matchup-adapter-lr must be positive")
        if int(args.expert_matchup_adapter_games_per_batch) <= 0:
            raise ValueError(
                "--expert-matchup-adapter-games-per-batch must be positive"
            )
        if int(args.expert_matchup_adapter_max_decisions_per_batch) <= 0:
            raise ValueError(
                "--expert-matchup-adapter-max-decisions-per-batch must be positive"
            )
    if int(args.train_games_per_batch) <= 0:
        raise ValueError("--train-games-per-batch must be positive")
    if int(args.train_max_decisions_per_batch) <= 0:
        raise ValueError("--train-max-decisions-per-batch must be positive")
    _scheduled_train_decision_cap(
        0,
        steady_cap=int(args.train_max_decisions_per_batch),
        warmup_cap=int(args.train_warmup_max_decisions_per_batch),
        warmup_iterations=int(args.train_warmup_iterations),
    )
    if not 0.0 <= float(args.continuous_learner_min_wr) <= 1.0:
        raise ValueError("--continuous-learner-min-wr must be in [0, 1]")
    if not 0.0 <= float(args.continuous_learner_exact_regression_margin) <= 1.0:
        raise ValueError(
            "--continuous-learner-exact-regression-margin must be in [0, 1]"
        )
    if int(args.continuous_learner_exact_regression_patience) <= 0:
        raise ValueError(
            "--continuous-learner-exact-regression-patience must be positive"
        )
    if int(args.artifact_history_iterations) <= 0:
        raise ValueError("--artifact-history-iterations must be positive")
    if float(args.min_free_disk_gb) < 0.0:
        raise ValueError("--min-free-disk-gb cannot be negative")
    active_gate_contract: Optional[dict[str, Any]] = None
    active_gate_contract_identity: Optional[dict[str, Any]] = None
    active_gate: Optional[dict[str, Any]] = None
    formal_gate_contract: Optional[dict[str, Any]] = None
    formal_gate_contract_identity: Optional[dict[str, Any]] = None
    formal_gate: Optional[dict[str, Any]] = None
    requested_heldout_games = args.heldout_games
    frozen_specialist_registry_path = (
        Path(args.frozen_specialist_registry).expanduser().resolve()
    )
    frozen_specialist_registry = _load_frozen_specialist_registry(
        frozen_specialist_registry_path
    )
    frozen_specialist_registry_identity = _path_content_identity(
        frozen_specialist_registry_path
    )
    if args.active_gate_contract is None:
        raise RuntimeError(
            "production full-loop launch requires --active-gate-contract; "
            "research controls are diagnostic-only and can never be the fallback "
            "formal holdout"
        )
    else:
        active_gate_path = Path(args.active_gate_contract).expanduser().resolve()
        base_active_gate_contract = load_active_gate_contract(active_gate_path)
        active_gate_contract_identity = _path_content_identity(active_gate_path)
        raw_gate = dict(base_active_gate_contract["next_gate"])
        raw_evaluation = dict(raw_gate["evaluation"])
        raw_roster = list(raw_gate.get("roster") or [])
        frozen_registry_ids = {
            str(row["opponent_id"])
            for row in frozen_specialist_registry.get("specialists") or []
        }
        established_roster_size = sum(
            1
            for row in raw_roster
            if str(row.get("opponent_id") or "") not in frozen_registry_ids
        )
        games_per_opponent = int(
            raw_evaluation.get("games_per_opponent") or 0
        )
        if established_roster_size <= 0 or games_per_opponent <= 0:
            raise RuntimeError(
                "active gate contract has no established non-specialist roster"
            )
        # The canonical gate may already contain the frozen-specialist
        # extension. Keep accepting the service unit's established-roster
        # count while always running the effective augmented count.
        base_contract_games = established_roster_size * games_per_opponent
        active_gate_contract = _augment_gate_with_frozen_specialists(
            base_active_gate_contract,
            frozen_specialist_registry,
        )
        active_gate = dict(active_gate_contract["next_gate"])
        contract_games = int(active_gate["evaluation"]["games_total"])
        r327_formal_override_requested = bool(
            args.formal_holdout_contract is not None
            and args.r327_evaluation_boundary_receipt is not None
        )
        if (
            args.heldout_games is not None
            and int(args.heldout_games)
            not in {base_contract_games, contract_games}
            and not r327_formal_override_requested
        ):
            raise RuntimeError(
                "--heldout-games disagrees with active gate contract: "
                f"cli={args.heldout_games} base={base_contract_games} "
                f"effective={contract_games}"
            )
        args.heldout_games = contract_games
        args.gate_wr = float(
            active_gate["pass_criteria"]["skill_weighted_win_rate"]
        )
        args.heldout_per_opponent_floor = float(
            active_gate["pass_criteria"].get("individual_opponent_floor", 0.0)
        )
        print(
            "[pure_rl] ACTIVE_GATE_ASSERT "
            f"id={active_gate['id']} opponents={len(active_gate['roster'])} "
            f"games={contract_games} per_opponent="
            f"{active_gate['evaluation']['games_per_opponent']} "
            "stage=heldout:strong_public_gate",
            flush=True,
        )
    r284_boundary_receipt: dict[str, Any] = {}
    r327_evaluation_boundary_receipt: dict[str, Any] = {}
    formal_gate_contract = active_gate_contract
    formal_gate_contract_identity = active_gate_contract_identity
    formal_gate = active_gate
    if args.formal_holdout_contract is not None:
        formal_gate_path = Path(args.formal_holdout_contract).expanduser().resolve()
        # Revision 25 preserves the sealed base contract's historical
        # ``active_gate_semantics`` block byte-for-byte while deriving a smaller
        # future formal gate.  The dedicated receipt validator below proves the
        # exact permitted delta; the legacy generic loader rejects that preserved
        # historical summary before it can reach the dedicated validator.
        if args.r327_evaluation_boundary_receipt is not None:
            formal_gate_contract = json.loads(
                formal_gate_path.read_text(encoding="utf-8")
            )
        else:
            formal_gate_contract = load_active_gate_contract(formal_gate_path)
        formal_gate_contract_identity = _path_content_identity(formal_gate_path)
        formal_gate = dict(formal_gate_contract["next_gate"])
        formal_roster = list(formal_gate.get("roster") or [])
        formal_ids = tuple(str(row.get("opponent_id") or "") for row in formal_roster)
        if args.r327_evaluation_boundary_receipt is not None:
            from poke_bot.alakazam_rule_derivative_evaluation_r327 import (
                validate_revision25_activation_receipt,
            )

            r327_receipt_path = (
                Path(args.r327_evaluation_boundary_receipt).expanduser().resolve()
            )
            r327_evaluation_boundary_receipt = json.loads(
                r327_receipt_path.read_text(encoding="utf-8")
            )
            boundary_iteration = int(
                r327_evaluation_boundary_receipt.get(
                    "boundary_commit_iteration", -1
                )
            )
            boundary_commit_path = (
                _run_dir(args.run_name)
                / "commits"
                / f"iter_{boundary_iteration:05d}.json"
            )
            r327_evaluation_boundary_receipt = (
                validate_revision25_activation_receipt(
                    r327_evaluation_boundary_receipt,
                    base_contract_path=active_gate_path,
                    formal_contract_path=formal_gate_path,
                    boundary_commit_path=boundary_commit_path,
                )
            )
            formal_games = int(formal_gate["evaluation"]["games_total"])
            if (
                requested_heldout_games is not None
                and int(requested_heldout_games) != formal_games
            ):
                raise RuntimeError(
                    "--heldout-games disagrees with r327 formal contract: "
                    f"cli={requested_heldout_games} formal={formal_games}"
                )
            if formal_ids != tuple(
                str(row.get("opponent_id") or "")
                for row in list((active_gate or {}).get("roster") or [])
            ):
                raise RuntimeError("r327 formal holdout roster identity changed")
            if not bool(args.disable_research_control_measurement):
                raise RuntimeError(
                    "r327 formal evaluation requires zero research-control games"
                )
        else:
            if (
                formal_gate_contract.get("owner_decision_revision") != 284
                or formal_gate_contract.get("owner_restricted_formal_only_roster")
                is not True
                or formal_gate_contract.get(
                    "frozen_specialist_registry_augmentation_allowed"
                )
                is not False
            ):
                raise RuntimeError(
                    "formal holdout override is not an authorized contract"
                )
            if formal_ids != (
                "alakazam-r195-no-rtp-submission-55378392",
                "specialist-marnie-final-format-h10-f20efb20f5c3",
            ):
                raise RuntimeError("r284 formal holdout roster identity changed")
        args.heldout_games = int(formal_gate["evaluation"]["games_total"])
        args.gate_wr = float(formal_gate["pass_criteria"]["skill_weighted_win_rate"])
        args.heldout_per_opponent_floor = float(
            formal_gate["pass_criteria"].get("individual_opponent_floor", 0.0)
        )
        print(
            "[pure_rl] FORMAL_HOLDOUT_OVERRIDE "
            f"id={formal_gate['id']} opponents={len(formal_roster)} "
            f"games={args.heldout_games} first_iteration="
            f"{r327_evaluation_boundary_receipt.get('first_formal_holdout_iteration', 1)}",
            flush=True,
        )
        if not r327_evaluation_boundary_receipt:
            if args.r284_iteration_boundary_receipt is None:
                raise RuntimeError("r284 formal holdout requires its boundary receipt")
            r284_receipt_path = (
                Path(args.r284_iteration_boundary_receipt).expanduser().resolve()
            )
            r284_boundary_receipt = json.loads(
                r284_receipt_path.read_text(encoding="utf-8")
            )
            if (
                r284_boundary_receipt.get("schema")
                != "poke_bot.alakazam_r274_iteration_1_boundary_r284/v1"
                or r284_boundary_receipt.get("owner_revision") != 284
                or r284_boundary_receipt.get("status") != "authorized"
                or r284_boundary_receipt.get("run_name")
                != "alakazam_new_list_direct_policy_r274"
                or r284_boundary_receipt.get(
                    "defer_formal_holdout_through_iteration"
                )
                != 0
                or r284_boundary_receipt.get("first_formal_holdout_iteration") != 1
                or float(
                    r284_boundary_receipt.get("combo_state_loss_weight", -1.0)
                )
                != 0.0
                or r284_boundary_receipt.get("combo_state_route_enabled") is not False
                or r284_boundary_receipt.get("combo_state_tensors_preserved") is not True
                or r284_boundary_receipt.get("formal_holdout_contract_sha256")
                != _sha256_file(formal_gate_path)
                or r284_boundary_receipt.get("candidate_checkpoint_sha256")
                != "sha256:645f8e6a0bc5e0cb98695e5a65151f38eef2e5011ca63f3ea4eded42cf4a11a2"
                or int(
                    r284_boundary_receipt.get("partial_broad_holdout_games", -1)
                )
                != 2142
                or r284_boundary_receipt.get(
                    "partial_broad_holdout_gate_eligible"
                )
                is not False
            ):
                raise RuntimeError("r284 iteration boundary receipt is invalid")
    if args.r327_evaluation_boundary_receipt is not None and not (
        args.formal_holdout_contract is not None
        and r327_evaluation_boundary_receipt
        and bool(args.disable_research_control_measurement)
    ):
        raise RuntimeError(
            "r327 evaluation migration requires its formal contract and zero "
            "research-control measurement"
        )
    if int(args.heldout_games) <= 0:
        raise ValueError("heldout game count must be positive")
    seed_namespace_contract = _assert_seed_namespace_contract(
        root_seed=int(args.seed),
        iterations=int(args.iterations),
        games_per_iteration=int(args.games_per_iter),
        formal_games=int(args.heldout_games),
        research_control_games=(
            0
            if bool(args.disable_research_control_measurement)
            else int(args.research_control_games_per_iter)
        ),
    )
    if active_gate is not None and float(args.official_adaptive_min_share) > (
        1.0 / len(active_gate["roster"])
    ):
        raise ValueError(
            "--official-adaptive-min-share is infeasible for the active gate roster"
        )
    collection_group_plan = _planned_collection_group_counts(
        games_per_iteration=int(args.games_per_iter),
        self_play_fraction=float(config.PURE_RL.self_play_frac),
        strong_public_fraction_of_public=float(args.official_collect_frac),
        research_control_games=int(args.research_control_games_per_iter),
    )
    practice_minimum_games = (
        _strong_public_practice_minimum_games(
            active_gate=active_gate,
            expected_practice_games=int(
                collection_group_plan[STRONG_PUBLIC_PRACTICE_GROUP]
            ),
            minimum_share=float(args.official_adaptive_min_share),
        )
        if active_gate is not None
        and float(args.official_collect_frac) > 0.0
        else {}
    )
    if fixed_cycle_updates:
        from poke_bot.r241_direct_policy_runtime import (
            R241_H10_CONTENT_SHA256,
            R241_H10_MODEL_SHA256,
            R241_H10_OPPONENT_ID,
        )

        h10_rows = [
            dict(row)
            for row in list((active_gate or {}).get("roster") or ())
            if str(dict(row).get("opponent_id") or "") == R241_H10_OPPONENT_ID
        ]
        h10 = h10_rows[0] if len(h10_rows) == 1 else {}
        if (
            collection_group_plan
            != {
                "self_play": 1_024,
                STRONG_PUBLIC_PRACTICE_GROUP: 4_586,
                "diverse_public": 2_586,
            }
            or len(h10_rows) != 1
            or h10.get("content_digest") != R241_H10_CONTENT_SHA256
            or h10.get("frozen_checkpoint_digest") != R241_H10_MODEL_SHA256
            or h10.get("frozen_specialist") is not True
            or h10.get("tier") != "S++"
            or float(h10.get("weight", 0.0)) != 4.0
            or int(h10.get("strong_public_practice_floor_games", -1)) != 1_024
            or int(practice_minimum_games.get(R241_H10_OPPONENT_ID, -1))
            < 1_024
        ):
            raise RuntimeError(
                "r241 active public mix must pin direct H10 Marnie for at "
                "least 1024 of the 7172 public games"
            )

    hw = full_hardware_profile()
    if args.allow_single_gpu:
        hw = replace(hw, allow_single_gpu=True)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    hw.validate_or_raise(visible_gpu_count=visible)
    run_dir = _run_dir(args.run_name)
    loop_state = _load_loop_state(run_dir)
    resumed = loop_state is not None
    immutable_manifest: Optional[dict[str, Any]] = None
    if loop_state is not None:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume ledger exists but immutable manifest is missing")
        immutable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ckpt, start_iteration = _validate_resume_state(
            args=args, run_dir=run_dir, state=loop_state
        )
        ckpt = _ensure_pure_rl_checkpoint(
            ckpt,
            args.seed,
            smoke=False,
            allow_legacy_inference_profile=True,
            r241_peak_r195_training_contract=(
                {} if r260_own_deck_training_inputs else r241_peak_r195_training_contract
            ),
            r274_successor_training_contract=r260_own_deck_training_inputs,
        )
        print(
            f"[pure_rl] RESUME next_iteration={start_iteration} "
            f"champion={ckpt}",
            flush=True,
        )
    else:
        if args.start_iteration not in (None, 0):
            raise RuntimeError(
                "a new lineage must start at iteration 0; use its loop ledger "
                "to resume a later iteration"
            )
        if (run_dir / "manifest.json").exists() or any(
            any((run_dir / part).glob("iter_*"))
            for part in ("shards", "checkpoints", "metrics", "eval")
        ):
            raise RuntimeError(
                f"run directory {run_dir} has artifacts but no loop_state.json; "
                "refusing destructive/inferred resume — use a new run name"
            )
        seed_path = (
            Path(args.initial_learner_checkpoint).expanduser().resolve()
            if r260_own_deck_training_inputs
            and args.initial_learner_checkpoint is not None
            else (
                Path(args.base_checkpoint).expanduser().resolve()
                if args.base_checkpoint is not None
                else (run_dir / "checkpoints" / "seed.pt")
            )
        )
        ckpt = _ensure_pure_rl_checkpoint(
            seed_path,
            args.seed,
            smoke=False,
            r241_peak_r195_training_contract=(
                {} if r260_own_deck_training_inputs else r241_peak_r195_training_contract
            ),
            r274_successor_training_contract=r260_own_deck_training_inputs,
        )
        start_iteration = 0

    if args.research_control_registry is None:
        stored_registry_path = ""
        if resumed and isinstance(immutable_manifest, dict):
            stored_registry = (
                (immutable_manifest.get("design_contract") or {})
                .get("collection", {})
                .get("research_control_phase", {})
                .get("registry", {})
            )
            if isinstance(stored_registry, dict):
                stored_registry_path = str(stored_registry.get("path") or "")
        # Existing lineages stay pinned to the registry they started with. An
        # older manifest predating this field uses the seed registry, never a
        # newly retired roster that still overlaps its terminal active gate.
        args.research_control_registry = Path(
            stored_registry_path
            or (
                str(ROOT / "ops" / "research_control_registry_v1.json")
                if resumed
                else str(_default_research_control_registry())
            )
        )

    os.environ["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    os.environ.setdefault(
        "POKEBOT_PRIMARY_ARCHETYPE",
        str(args.specialist_archetype or "core"),
    )
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"

    stage = stage_for_iteration(core_gate_passed=(args.mode == "specialist"))
    identity = CheckpointIdentity.from_path(ckpt)
    endpoints = _resolve_remote_endpoints(args)

    ensure_baselines_installed()
    manifest_baselines = load_manifest()
    # Validate the checksum-bound data-only H10 package before importing any
    # historical baseline entry point that may transiently touch CG_LIB_PATH.
    manifest_baselines.sort(
        key=lambda spec: 0 if spec.id == "specialist-marnie-final-format-h10-f20efb20f5c3" else 1
    )
    if r274_r195_research_contract:
        installed_path = Path(
            str(r274_r195_research_contract.get("installed_path") or "")
        ).expanduser().resolve()
        manifest_r195 = [
            spec
            for spec in manifest_baselines
            if spec.id == R274_R195_RESEARCH_OPPONENT_ID
        ]
        if len(manifest_r195) > 1:
            raise RuntimeError("submission-55378392 appears more than once in manifest")
        if manifest_r195:
            if manifest_r195[0].path.resolve() != installed_path:
                raise RuntimeError(
                    "submission-55378392 manifest path disagrees with install receipt"
                )
        else:
            manifest_baselines.append(
                BaselineSpec(
                    id=R274_R195_RESEARCH_OPPONENT_ID,
                    name="Alakazam r195 NO-RTP submission 55378392",
                    dir_name=installed_path.name,
                    group="specialists",
                    source="receipt-bound exact Kaggle submission 55378392",
                    path=installed_path,
                )
            )
    loadable, _failed_baselines = filter_loadable_baselines(manifest_baselines)
    by_id = {s.id: s for s in loadable}
    r274_r195_research_spec = None
    if r274_r195_research_contract:
        r274_r195_research_spec = by_id.get(R274_R195_RESEARCH_OPPONENT_ID)
        if r274_r195_research_spec is None:
            raise RuntimeError(
                "exact submission-55378392 research baseline is unavailable"
            )
        installed_path = Path(
            str(r274_r195_research_contract.get("installed_path") or "")
        ).expanduser().resolve()
        installed_digest = baseline_content_digest(r274_r195_research_spec.path)
        if (
            r274_r195_research_spec.path.resolve() != installed_path
            or installed_digest
            != str(r274_r195_research_contract["installed_content_sha256"])
        ):
            raise RuntimeError(
                "submission-55378392 research baseline bytes changed"
            )
    frozen_specialist_ids = tuple(
        str(row["opponent_id"])
        for row in frozen_specialist_registry.get("specialists") or []
    )
    missing_frozen_specialists = [
        opponent_id
        for opponent_id in frozen_specialist_ids
        if opponent_id not in by_id
    ]
    if missing_frozen_specialists:
        raise RuntimeError(
            "frozen specialist packages are unavailable: "
            f"{missing_frozen_specialists}"
        )
    for row in frozen_specialist_registry.get("specialists") or []:
        opponent_id = str(row["opponent_id"])
        actual_digest = baseline_content_digest(by_id[opponent_id].path)
        if actual_digest != str(row["content_digest"]):
            raise RuntimeError(
                "frozen specialist package digest mismatch: "
                f"{opponent_id} expected={row['content_digest']} "
                f"actual={actual_digest}"
            )
    requested_research_control_path = (
        Path(args.research_control_registry).expanduser().resolve()
    )
    research_control_path = _research_control_registry_for_lineage(
        requested_research_control_path,
        snapshot_dir=(
            paths.OUTPUTS_DIR
            / "state"
            / "research_control_registry_snapshots"
        ),
        immutable_manifest=immutable_manifest if resumed else None,
    )
    args.research_control_registry = research_control_path
    research_control_registry = load_research_control_registry(
        research_control_path
    )
    research_ids = research_control_ids(research_control_registry)
    research_control_specs = [by_id[i] for i in research_ids if i in by_id]
    if len(research_control_specs) < len(research_ids):
        missing = [i for i in research_ids if i not in by_id]
        raise RuntimeError(
            f"research controls are unavailable; missing baseline packages: {missing}"
        )
    if active_gate is None or active_gate_contract is None:
        raise RuntimeError("full-loop active gate contract was not initialized")
    practice_gate_ids = tuple(
        str(row["opponent_id"]) for row in active_gate["roster"]
    )
    if formal_gate is None or formal_gate_contract is None:
        raise RuntimeError("full-loop formal holdout contract was not initialized")
    active_gate_ids = tuple(
        str(row["opponent_id"]) for row in formal_gate["roster"]
    )
    missing = [
        opponent_id
        for opponent_id in set(practice_gate_ids) | set(active_gate_ids)
        if opponent_id not in by_id
    ]
    if missing:
        raise RuntimeError(f"active gate packages are unavailable: {missing}")
    heldout_specs = [by_id[opponent_id] for opponent_id in active_gate_ids]
    practice_specs = [by_id[opponent_id] for opponent_id in practice_gate_ids]
    installed_gate_digests = {
        spec.id: baseline_content_digest(spec.path) for spec in heldout_specs
    }
    verify_roster_content(
        formal_gate,
        installed_gate_digests,
    )
    active_gate_content_digests = set(installed_gate_digests.values())
    installed_research_digests = {
        spec.id: baseline_content_digest(spec.path)
        for spec in research_control_specs
    }
    research_control_registry = validate_research_control_registry(
        research_control_registry,
        installed_digests=installed_research_digests,
        active_gate_ids=active_gate_ids,
        active_gate_digests=tuple(sorted(active_gate_content_digests)),
    )
    research_control_registry_identity = _path_content_identity(
        research_control_path
    )
    research_control_registry_output = (
        paths.OUTPUTS_DIR / "state" / "research_control_registry_latest.json"
    )
    if formal_gate_contract is not None and formal_gate is not None:
        raw_prior_pointer = str(formal_gate.get("exact_result_pointer") or "").strip()
        if not raw_prior_pointer:
            raise RuntimeError("active gate has no exact_result_pointer")
        published = _publish_committed_active_gate_result(
            run_dir=run_dir,
            active_gate=formal_gate,
            result_pointer=Path(raw_prior_pointer),
        )
        if published is not None:
            published_result, published_commit = published
            _reconcile_passed_gate_research_controls(
                registry_path=research_control_path,
                gate_contract=formal_gate_contract,
                exact_result_path=published_result,
                commit_path=published_commit,
                output_path=research_control_registry_output,
            )
    if (
        args.mode == "specialist"
        and frozen_specialist_ids
        and float(args.official_collect_frac) <= 0.0
        and not bool(args.population_own_models_only)
    ):
        raise RuntimeError(
            "specialist training must keep frozen specialists in replay-eligible "
            "public practice; --official-collect-frac must be positive"
        )
    practice_archetypes = {
        str(row["opponent_id"]): str(row["archetype_id"])
        for row in active_gate["roster"]
    }
    practice_skill_weights = {
        str(row["opponent_id"]): float(row["weight"])
        for row in active_gate["roster"]
    }
    # The active strong-public roster is formal held-out evidence, not part of
    # the generic public mixture. Active-gate practice has its own replay-
    # eligible schedule; research controls have a separate non-training,
    # provenance-checked measurement transaction.
    population_opponent_registry = None
    if bool(args.population_own_models_only):
        population_registry_path = (
            Path(args.population_opponent_registry).expanduser().resolve()
        )
        try:
            population_opponent_registry = json.loads(
                population_registry_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "population opponent registry is missing or corrupt"
            ) from exc
        if not isinstance(population_opponent_registry, dict):
            raise RuntimeError("population opponent registry is not an object")
    population_specs = _population_collect_specs(
        enabled=bool(args.population_own_models_only),
        frozen_specialist_ids=frozen_specialist_ids,
        by_id=by_id,
        opponent_registry=population_opponent_registry,
    )
    if population_specs is not None:
        collect_specs = population_specs
        excluded_gate_digest_aliases: list[str] = []
        excluded_research_digest_aliases: list[str] = []
    else:
        # The exact frozen r195 submission deliberately has two roles in r274:
        # a seed-disjoint formal holdout opponent and a >=128-game training
        # research cell.  Keep every other protected opponent out of collection,
        # while allowing that one receipt-bound identity into the public mix.
        formal_training_exceptions = (
            {R274_R195_RESEARCH_OPPONENT_ID}
            if r274_r195_research_contract
            else set()
        )
        heldout_ids = (
            set(research_ids) | set(active_gate_ids) | set(practice_gate_ids)
        ) - formal_training_exceptions
        collect_candidates = [
            spec for spec in loadable if spec.id not in heldout_ids
        ]
        training_protected_gate_digests = {
            digest
            for opponent_id, digest in installed_gate_digests.items()
            if opponent_id not in formal_training_exceptions
        }
        protected_training_digests = (
            set(installed_research_digests.values())
            | training_protected_gate_digests
        )
        collect_content_digests = {
            spec.id: baseline_content_digest(spec.path)
            for spec in collect_candidates
        }
        excluded_gate_digest_aliases = sorted(
            spec.id
            for spec in collect_candidates
            if collect_content_digests.get(spec.id)
            in training_protected_gate_digests
        )
        excluded_research_digest_aliases = sorted(
            spec.id
            for spec in collect_candidates
            if collect_content_digests.get(spec.id)
            in set(installed_research_digests.values())
        )
        collect_specs, excluded_protected_digest_aliases = (
            _exclude_protected_baseline_aliases(
                specs=list(loadable),
                excluded_ids=heldout_ids,
                digest_by_id=collect_content_digests,
                protected_digests=protected_training_digests,
            )
        )
        if excluded_protected_digest_aliases != sorted(
            set(excluded_gate_digest_aliases)
            | set(excluded_research_digest_aliases)
        ):
            raise RuntimeError(
                "protected baseline alias accounting is inconsistent"
            )
    if active_gate is not None:
        print(
            "[pure_rl] ACTIVE_GATE_PRACTICE_SEED_DISJOINT "
            f"gate_ids={len(active_gate_ids)} "
            f"gate_digests={len(active_gate_content_digests)} "
            f"excluded_digest_aliases={excluded_gate_digest_aliases} "
            f"excluded_research_aliases={excluded_research_digest_aliases} "
            f"diverse_ids={len(collect_specs)} "
            f"practice_ids={len(practice_specs)} "
            f"research_control_ids={len(research_control_specs)}",
            flush=True,
        )
    if not collect_specs:
        raise RuntimeError(
            "no training opponents remain after excluding the formal held-out set"
        )
    r274_diverse_minimum_games: Optional[dict[str, int]] = None
    if r274_r195_research_contract:
        collect_ids = {str(spec.id) for spec in collect_specs}
        if R274_R195_RESEARCH_OPPONENT_ID not in collect_ids:
            raise RuntimeError(
                "submission-55378392 research opponent is absent from diverse public replay"
            )
        r274_diverse_minimum_games = {
            opponent_id: (
                R274_R195_RESEARCH_GAMES_MINIMUM
                if opponent_id == R274_R195_RESEARCH_OPPONENT_ID
                else 0
            )
            for opponent_id in collect_ids
        }
    ladder_mix = None
    ladder_representatives = None
    ladder_contract: Optional[dict[str, Any]] = None
    if args.mode == "core":
        (
            decks,
            ladder_mix,
            ladder_representatives,
            ladder_contract,
        ) = _core_ladder_decks()
        family_training_contract = None
    else:
        package_decks = _our_decks(args.mode, args.specialist_archetype)
        family_training_contract = _active_archetype_family_training_contract(
            specialist_archetype=str(args.specialist_archetype or ""),
        )
        if family_training_contract is None:
            decks = package_decks
        else:
            decks = list(family_training_contract["decks"])
            ladder_mix = family_training_contract["mix"]
            print(
                "[pure_rl] ARCHETYPE_FAMILY_ACTIVE "
                f"manifest={family_training_contract['manifest_sha256']} "
                f"loss_vector={family_training_contract['loss_vector_sha256']} "
                f"train_variants={len(decks)}",
                flush=True,
            )
    measurement_decks = _select_measurement_decks(
        package_decks if args.mode != "core" else decks,
        args.measurement_decks,
    )
    deck_names = [n for n, _ in decks]
    measurement_deck_names = [n for n, _ in measurement_decks]
    print(
        f"[pure_rl] our_decks n_decks={len(decks)} sample={deck_names[:16]}"
        + (
            f" ladder_mix={ladder_mix.artifact_sha256}"
            if ladder_mix is not None
            else ""
        ),
        flush=True,
    )
    print(
        "[pure_rl] measurement_decks "
        f"n_decks={len(measurement_decks)} ids={measurement_deck_names}",
        flush=True,
    )

    from poke_bot.remote_sim_jobs import (
        remote_play_job as worker_play,
        remote_self_play_job as worker_self_play,
        remote_self_play_multi_job as worker_self_play_multi,
    )
    from poke_bot.pure_rl.multi_env_self_play import (
        process_worker_count,
        pure_rl_leaf_coalesce_ms,
        resolve_multi_env_per_worker,
    )

    multi_env_n = resolve_multi_env_per_worker(args.multi_env_per_worker)
    proc_workers = process_worker_count(hw.sim_workers, multi_env_n)
    if args.leaf_coalesce_ms is not None:
        coalesce_ms = float(args.leaf_coalesce_ms)
    else:
        coalesce_ms = pure_rl_leaf_coalesce_ms(default=0.0)

    from poke_bot.live_pool import LIVE_POOL_PLAN_PATH, live_pool_enabled
    from poke_bot.pure_rl.live_pool_apply import apply_live_pool_plan

    live_pool_on = live_pool_enabled()

    checkpoint_profile = _checkpoint_contract(
        ckpt,
        smoke=False,
        allow_legacy_inference_profile=resumed,
        r241_peak_r195_training_contract=(
            {} if r260_own_deck_training_inputs else r241_peak_r195_training_contract
        ),
        r274_successor_training_contract=r260_own_deck_training_inputs,
    )
    initial_replay_shards = [
        Path(path).expanduser().resolve() for path in args.initial_replay_shard
    ]
    for shard in initial_replay_shards:
        if not shard.is_file():
            raise FileNotFoundError(f"initial replay shard is missing: {shard}")
    initial_replay_contract = [
        _path_content_identity(path) for path in initial_replay_shards
    ]
    if resumed:
        assert loop_state is not None
        initial_learner_identity = _verified_checkpoint_identity(
            loop_state.get("learner") or loop_state.get("champion") or {}
        )
        assert immutable_manifest is not None
        stored_initial_learner = (
            (immutable_manifest.get("design_contract") or {})
            .get("learner", {})
            .get("initial_checkpoint")
        )
        design_initial_learner_identity = _verified_checkpoint_identity(
            stored_initial_learner
        )
        if args.initial_learner_checkpoint is not None:
            requested_learner = CheckpointIdentity.from_path(
                Path(args.initial_learner_checkpoint).expanduser().resolve()
            )
            if requested_learner.digest != design_initial_learner_identity.digest:
                raise RuntimeError(
                    "--initial-learner-checkpoint conflicts with original lineage: "
                    f"requested={requested_learner.digest} "
                    f"original={design_initial_learner_identity.digest}"
                )
    else:
        initial_learner_identity = (
            CheckpointIdentity.from_path(
                Path(args.initial_learner_checkpoint).expanduser().resolve()
            )
            if args.initial_learner_checkpoint is not None
            else identity
        )
        design_initial_learner_identity = initial_learner_identity
        learner_profile = _checkpoint_contract(
            initial_learner_identity.path,
            smoke=False,
            r241_peak_r195_training_contract=(
                {} if r260_own_deck_training_inputs else r241_peak_r195_training_contract
            ),
            r274_successor_training_contract=r260_own_deck_training_inputs,
        )
        for field in (
            "model_profile",
            "decision_context",
            "max_context",
            "feature_schema",
        ):
            if learner_profile[field] != checkpoint_profile[field]:
                raise RuntimeError(
                    "initial learner is incompatible with rollout champion: "
                    f"field={field}"
                )
    initial_heldout_evidence_contract: Optional[dict[str, Any]] = None
    initial_heldout_evidence: Optional[dict[str, Any]] = None
    if args.initial_heldout_evidence is not None:
        initial_heldout_evidence_path = (
            Path(args.initial_heldout_evidence).expanduser().resolve()
        )
        initial_heldout_evidence_contract = _path_content_identity(
            initial_heldout_evidence_path
        )
        # The accepted original-four report remains immutable lineage history,
        # but it is not evidence for the active contract and must never seed
        # the active-gate champion selector.
        if active_gate is None:
            initial_heldout_evidence = _load_initial_heldout_evidence(
                initial_heldout_evidence_path,
                checkpoint=design_initial_learner_identity,
                heldout_games=int(args.heldout_games),
            )
    source_snapshot = _source_snapshot()
    if resumed:
        assert loop_state is not None and immutable_manifest is not None
        lineage_base = _verified_checkpoint_identity(
            loop_state.get("lineage_base") or {}
        ).as_dict()
    else:
        lineage_base = identity.as_dict()
    design_contract = _design_contract(
        args=args,
        checkpoint_profile=checkpoint_profile,
        lineage_base=lineage_base,
        initial_learner=design_initial_learner_identity.as_dict(),
        initial_heldout_evidence=initial_heldout_evidence_contract,
        initial_replay_shards=initial_replay_contract,
        source=source_snapshot,
        decks=decks,
        measurement_decks=measurement_decks,
        ladder_contract=ladder_contract,
        family_training_contract=family_training_contract,
        endpoints=endpoints,
        collect_specs=collect_specs,
        research_control_specs=research_control_specs,
        practice_specs=practice_specs,
        practice_minimum_games=(practice_minimum_games or None),
        heldout_specs=heldout_specs,
        research_control_registry=research_control_registry_identity,
        frozen_specialist_registry=frozen_specialist_registry_identity,
        active_gate=active_gate,
        seed_namespace_contract=seed_namespace_contract,
        multi_env_per_worker=multi_env_n,
        leaf_coalesce_ms=coalesce_ms,
        r241_peak_r195_training_contract=(
            r241_peak_r195_training_contract
        ),
    )
    if r260_own_deck_training_inputs and not r260_own_deck_training_inputs.get(
        "derivative_streaming_only"
    ):
        # The zero-safe migration remains historical provenance; update zero
        # uses only the separately verified runtime-enabled canary checkpoint.
        # Recording the complete Inzi transport chain cannot itself enable a
        # route or bypass the canary/evaluation gate.
        design_contract["r260_own_deck_successor"] = dict(
            r260_own_deck_training_inputs
        )
    design_fingerprint = _design_fingerprint(design_contract)
    pending_collection_contract = design_contract

    if not resumed:
        loop_state = {
            "version": LOOP_STATE_VERSION,
            "run_name": args.run_name,
            "mode": args.mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "next_iteration": 0,
            "last_completed_iteration": -1,
            "champion": identity.as_dict(),
            "heldout_champion": (
                initial_learner_identity.as_dict()
                if initial_heldout_evidence is not None
                else identity.as_dict()
            ),
            "heldout_champion_evidence": initial_heldout_evidence,
            "learner": initial_learner_identity.as_dict(),
            "lineage_base": lineage_base,
            "design_fingerprint": design_fingerprint,
            "opponent_pool": [identity.as_dict()],
            "raw_advantage_history": [],
            "policy_prev_agreement_history": [],
            "history": [],
            "live_pool_last_seq": 0,
        }
        _atomic_json(run_dir / "loop_state.json", loop_state)
        manifest = {
                "run_name": args.run_name,
                "mode": args.mode,
                "specialist_archetype": args.specialist_archetype,
                "smoke": False,
                "hardware": hw.as_dict(),
                "stage": stage_to_dict(stage),
                "base_checkpoint": str(ckpt),
                "checkpoint_digest": identity.digest,
                "base_checkpoint_contract": checkpoint_profile,
                "initial_learner_checkpoint": initial_learner_identity.as_dict(),
                "initial_heldout_evidence": initial_heldout_evidence_contract,
                "initial_replay_shards": initial_replay_contract,
                "model_profile": checkpoint_profile["model_profile"],
                "trainable_parameters": int(
                    checkpoint_profile["trainable_parameters"]
                ),
                "decision_context": checkpoint_profile["decision_context"],
                "max_context": checkpoint_profile["max_context"],
                "feature_schema": checkpoint_profile["feature_schema"],
                "design_fingerprint": design_fingerprint,
                "design_contract": design_contract,
                "param_fail_max": int(config.PURE_RL.param_fail_max),
                "gate_wr": float(args.gate_wr),
                "heldout_games": int(args.heldout_games),
                "heldout_per_opponent_floor": float(
                    args.heldout_per_opponent_floor
                ),
                "promotion_gate": {
                    "games": int(args.promotion_games),
                    "workers": int(args.promotion_workers),
                    "threshold": float(args.promotion_threshold),
                    "confidence": float(args.promotion_confidence),
                    "bootstrap_resamples": int(
                        args.promotion_bootstrap_resamples
                    ),
                },
                "training_design": {
                    "spinning_up": "https://spinningup.openai.com/en/latest/spinningup/spinningup.html",
                    "algorithm": "AWR_actor_critic",
                    "not_alphazero": True,
                    "gamma": 1.0,
                    "return_type": "terminal_monte_carlo",
                    "awr_baseline": "frozen_precomputed_per_iteration",
                    "normalize_advantages": bool(config.PURE_RL.normalize_advantages),
                    "entropy_bonus": float(config.PURE_RL.entropy_bonus),
                    "bootstrap_mix": float(config.PURE_RL.bootstrap_mix),
                    "replay_window_shards": int(config.PURE_RL.replay_window_shards),
                    "self_play_frac": float(config.PURE_RL.self_play_frac),
                    "collect_temperature": float(args.collect_temperature),
                    "collect_temperature_final": float(
                        config.PURE_RL.collect_temperature_final
                    ),
                    "train_epochs": int(args.train_epochs),
                },
                "leaf_devices": hw.leaf_cuda_devices(),
                "remote_worker_endpoints": endpoints,
                "collect_opponents": [s.id for s in collect_specs],
                "research_control_opponents": [
                    s.id for s in research_control_specs
                ],
                "strong_public_practice_opponents": (
                    [s.id for s in practice_specs]
                    if float(args.official_collect_frac) > 0.0
                    else []
                ),
                "research_control_registry": research_control_registry_identity,
                "active_gate_id": (
                    str(active_gate["id"]) if active_gate is not None else None
                ),
                "active_gate_contract": active_gate_contract_identity,
                "formal_holdout_contract": formal_gate_contract_identity,
                "heldout_opponents": list(active_gate_ids),
                "our_decks": deck_names,
                "n_decks": len(decks),
                "ladder_deck_mix": ladder_contract,
                "multi_env_per_worker": multi_env_n,
                "proc_workers_self_play": proc_workers,
                "leaf_coalesce_ms": coalesce_ms,
                "live_pool": live_pool_on,
                "live_pool_plan": str(LIVE_POOL_PLAN_PATH),
                "min_usable_game_frac": float(args.min_usable_game_frac),
                "source": source_snapshot,
                "selected_environment": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith(("PURE_RL_", "POKEBOT_"))
                    or key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
                },
                "note": (
                    "Single small AWR trainee (~1-3M); remotes are whole-game "
                    "collect farms merging into the same shards. Kaggle reports "
                    "motivate experiments but do not prescribe these hyperparameters."
                ),
        }
        _write_json_exclusive(run_dir / "manifest.json", manifest)
    else:
        assert loop_state is not None and immutable_manifest is not None
        effective_before_migration, _effective_digest, _migration_receipts = (
            _load_design_migration_chain(run_dir, immutable_manifest)
        )
        # Quarantine any uncommitted partial N+1 attempt before evaluating a
        # clean-boundary gate migration. Otherwise a stale partial shard can
        # block the very launch that is supposed to replace its bad contract.
        recovery = _recover_interrupted_iteration(
            run_dir,
            loop_state,
            research_control_registry=research_control_registry,
            expected_awr_provider=(
                (effective_before_migration.get("learner") or {}).get(
                    "prize_plan_h3_actor_provider"
                )
            ),
        )
        if recovery is not None:
            print(
                f"[pure_rl] recovered interrupted iteration into {recovery}",
                flush=True,
            )
        # An already-receipted N+1 transaction remains governed by the design
        # in force when its games were collected.  The migration below governs
        # newly collected work only; using it to reopen this receipt would
        # incorrectly discard a fully trained, provenance-verified candidate.
        recovered_collection, recovered_collection_contract = (
            _verified_completed_collection_across_design_chain(
                run_dir, loop_state, immutable_manifest
            )
        )
        pending_collection_contract = (
            recovered_collection_contract or effective_before_migration
        )
        if recovered_collection is None:
            recovered_collection = _ensure_recoverable_completed_collection(
                run_dir,
                loop_state,
                effective_before_migration,
            )
        if recovered_collection is not None:
            print(
                "[pure_rl] completed collection recovery proof ready "
                f"iter={recovered_collection['iteration']} "
                f"receipt={recovered_collection['receipt_path']}",
                flush=True,
            )
        design_fingerprint = _validate_or_migrate_design_fingerprint(
            run_dir=run_dir,
            state=loop_state,
            manifest=immutable_manifest,
            current=design_contract,
            allow_clean_boundary_migration=bool(
                args.allow_clean_boundary_design_migration
            ),
            migration_reason=_effective_boundary_design_migration_reason(args),
            owner_r284_iteration1_boundary=bool(r284_boundary_receipt),
        )

    configured_terminal_marker_name = str(
        args.terminal_gate_marker_name or ""
    ).strip() or (
        "CORE_GATE_PASSED"
        if str(loop_state.get("mode")) == "core"
        else "SPECIALIST_GATE_PASSED"
    )
    configured_terminal_marker = run_dir / configured_terminal_marker_name
    terminal_payload = _terminal_gate_payload(loop_state)
    terminal_payload_iteration = (
        int(terminal_payload["iteration"])
        if isinstance(terminal_payload, dict)
        and terminal_payload.get("iteration") is not None
        else -1
    )
    terminal_payload_is_eligible = bool(
        int(args.minimum_terminal_iteration) < 0
        or terminal_payload_iteration >= int(args.minimum_terminal_iteration)
    )
    terminal_marker = (
        _ensure_terminal_gate_marker(
            run_dir,
            loop_state,
            preserve_first=(
                bool(args.continue_after_gate) or bool(fixed_cycle_updates)
            ),
            marker_name=configured_terminal_marker_name,
        )
        if configured_terminal_marker.exists() or terminal_payload_is_eligible
        else None
    )
    if terminal_marker is not None:
        marker_gate_id = _terminal_marker_gate_id(terminal_marker, loop_state)
        terminal_target_reached = _terminal_gate_target_matches(
            requested_gate_id=str(args.terminal_active_gate_id or ""),
            passed_gate_id=marker_gate_id,
            base_contract=active_gate_contract,
        )
        try:
            marker_iteration = int(
                json.loads(terminal_marker.read_text(encoding="utf-8"))[
                    "iteration"
                ]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"terminal gate marker has no valid iteration: {terminal_marker}"
            ) from exc
        terminal_minimum_reached = (
            int(args.minimum_terminal_iteration) < 0
            or marker_iteration >= int(args.minimum_terminal_iteration)
        )
        print(
            f"[pure_rl] committed terminal gate already complete: {terminal_marker} "
            f"gate_id={marker_gate_id or 'unknown'}",
            flush=True,
        )
        if _gate_pass_should_stop(
            fixed_cycle_updates=fixed_cycle_updates,
            continue_after_gate=bool(args.continue_after_gate),
            terminal_target_reached=(
                terminal_target_reached and terminal_minimum_reached
            ),
        ):
            return 0
        if (
            fixed_cycle_updates == 0
            and terminal_target_reached
            and not terminal_minimum_reached
        ):
            raise RuntimeError(
                "terminal marker predates the configured minimum terminal "
                f"iteration: marker={marker_iteration} "
                f"minimum={int(args.minimum_terminal_iteration)}"
            )
        if fixed_cycle_updates:
            print(
                "[pure_rl] FIXED_CYCLE_GATE_MARKER_NONTERMINAL "
                f"updates={fixed_cycle_updates} marker_iteration={marker_iteration}",
                flush=True,
            )
        else:
            print(
                "[pure_rl] CONTINUE_AFTER_GATE armed; preserving first-pass marker "
                "and resuming the next committed iteration",
                flush=True,
            )

    assert loop_state is not None
    learner_identity = _verified_checkpoint_identity(
        loop_state.get("learner") or loop_state.get("champion") or {}
    )
    heldout_champion_identity = _verified_checkpoint_identity(
        loop_state.get("heldout_champion") or loop_state.get("champion") or {}
    )
    raw_heldout_champion_evidence = dict(
        loop_state.get("heldout_champion_evidence") or {}
    )
    heldout_champion_evidence = _reconciled_heldout_champion_evidence(
        loop_state
    )
    if heldout_champion_evidence != raw_heldout_champion_evidence:
        loop_state["heldout_champion_evidence"] = heldout_champion_evidence
        print(
            "[pure_rl] reconciled legacy heldout champion evidence from "
            "committed per-opponent gate",
            flush=True,
        )
    if heldout_champion_evidence and str(
        heldout_champion_evidence.get("checkpoint_digest") or ""
    ) != str(heldout_champion_identity.digest):
        raise RuntimeError(
            "heldout champion evidence digest does not match its checkpoint"
        )
    live_pool_last_seq = int(loop_state.get("live_pool_last_seq", 0))
    opponent_pool_entries = list(loop_state.get("opponent_pool") or [])
    opponent_pool = [
        _verified_checkpoint_identity(entry) for entry in opponent_pool_entries
    ]
    if not any(entry.digest == identity.digest for entry in opponent_pool):
        opponent_pool.append(identity)
    adv_hist: list[float] = [
        float(x) for x in (loop_state.get("raw_advantage_history") or [])
    ]
    agr_hist: list[float] = [
        float(x)
        for x in (loop_state.get("policy_prev_agreement_history") or [])
    ]

    print(
        f"[pure_rl] full hardware workers={hw.sim_workers} "
        f"self_play_procs={proc_workers} multi_env={multi_env_n} "
        f"leaf_coalesce_ms={coalesce_ms} "
        f"leaves_gpu0={hw.leaf_gpu0_replicas} leaves_gpu1={hw.leaf_gpu1_replicas} "
        f"train_cuda={hw.train_cuda_device} remotes={endpoints or 'none'} "
        f"live_pool={'on' if live_pool_on else 'off'} "
        f"champion={identity.digest[:19]}… "
        f"heldout_champion={heldout_champion_identity.digest[:19]}… "
        f"learner={learner_identity.digest[:19]}…",
        flush=True,
    )

    leaf = _LeafFarm()
    remote_farm: Optional[RemoteWorkerFarm] = None
    train_dev = torch.device(f"cuda:{hw.train_cuda_device}")
    expert_cache = ResidentExpertCorpusCache()

    try:
        if args.leaf_eval == "gpu-server" and visible >= 1:
            leaf_devices = hw.leaf_cuda_devices()
            if visible < 2:
                leaf_devices = [0] * max(1, hw.leaf_gpu0_replicas)
            # Resp queues cover max(process counts) used by self-play or baseline pools.
            leaf_slots = max(proc_workers, hw.sim_workers)
            # max_batch=None → each server uses leaf_batch_for_device (256 on
            # 3080 Ti / 512 on Blackwell). A shared 1024 blows 12GB GPU0.
            leaf.start(
                ckpt=learner_identity.path,
                digest=learner_identity.digest,
                leaf_devices=leaf_devices,
                n_workers=leaf_slots,
                max_batch=None,
                coalesce_ms=coalesce_ms,
            )
        else:
            print("[pure_rl] leaf-eval=cpu (local workers load model)", flush=True)

        if endpoints:
            remote_job_buffer_s = float(
                os.environ.get("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", "600") or "600"
            )
            remote_job_timeout = max(
                float(getattr(config.SEARCH, "remote_request_timeout_s", 120.0)),
                float(args.game_timeout_s) + remote_job_buffer_s,
            )
            remote_farm = RemoteWorkerFarm(endpoints, timeout_s=remote_job_timeout)
            require_all = str(
                os.environ.get("POKEBOT_REMOTE_REQUIRE_ALL", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
            no_local_fallback = str(
                os.environ.get("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "0")
            ).strip().lower() in ("1", "true", "yes", "on")
            try:
                infos = remote_farm.connect(require_all=require_all)
            except Exception as exc:
                print(
                    f"[pure_rl] ERROR: remote connect failed ({exc}); "
                    "fail-closed for production remotes requirement",
                    file=sys.stderr,
                    flush=True,
                )
                (run_dir / "REMOTE_CONNECT_FAILED").write_text(
                    json.dumps({"error": str(exc), "endpoints": endpoints}, indent=2),
                    encoding="utf-8",
                )
                try:
                    remote_farm.close()
                except Exception:
                    pass
                if require_all or no_local_fallback:
                    raise RuntimeError(
                        "required remote farm did not connect; refusing local-only "
                        "production collection"
                    ) from exc
                remote_farm = None
                infos = []
            for info in infos:
                print(
                    f"[pure_rl] remote={info.endpoint} host={info.hostname} "
                    f"gpu={info.gpu_name!r} workers={info.workers}",
                    flush=True,
                )
            if remote_farm is not None:
                print(
                    f"[pure_rl] remote additive capacity={remote_farm.total_workers}",
                    flush=True,
                )
                # Boot boundary: same hard gate as between-iter (no soft WARN).
                # Local leaves just started on this digest — skip redundant reload.
                _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=learner_identity.path,
                    digest=learner_identity.digest,
                    version=int(start_iteration) * 10,
                    required_endpoints=endpoints,
                    reload_local=False,
                )

        if args.heldout_remotes:
            if remote_farm is None:
                raise RuntimeError(
                    "--heldout-remotes requires a connected remote farm; "
                    "refusing to run a partially local formal gate"
                )
            heldout_remote_audit = _remote_heldout_capability_audit(
                remote_farm,
                required_endpoints=endpoints,
            )
            _atomic_json(
                run_dir / "heldout_remote_capability.json",
                heldout_remote_audit,
            )
            if not heldout_remote_audit["passed"]:
                raise RuntimeError(
                    "formal heldout remote capability audit failed: "
                    + json.dumps(heldout_remote_audit, sort_keys=True)
                )
            print(
                "[pure_rl] formal heldout remotes VERIFIED "
                f"endpoints={len(heldout_remote_audit['endpoints'])}",
                flush=True,
            )

        def _retain_completed_artifacts(
            state: dict[str, Any], completed_iteration: int
        ) -> dict[str, Any]:
            report = apply_artifact_retention(
                run_dir,
                state,
                completed_iteration=int(completed_iteration),
                replay_window_shards=int(config.PURE_RL.replay_window_shards),
                history_iterations=int(args.artifact_history_iterations),
            )
            _atomic_json(run_dir / "artifact_receipts" / "latest.json", report)
            if report.get("retired_shards") or report.get("retired_checkpoints"):
                print(
                    "[pure_rl] artifact retention "
                    f"shards={report.get('retired_shards')} "
                    f"checkpoints={report.get('retired_checkpoints')} "
                    f"reclaimed_gib={float(report.get('reclaimed_bytes', 0)) / (1024 ** 3):.2f}",
                    flush=True,
                )
            return report

        if int(loop_state.get("last_completed_iteration", -1)) >= 0:
            _retain_completed_artifacts(
                loop_state,
                int(loop_state["last_completed_iteration"]),
            )

        pending_collect: Optional[dict[str, Any]] = None

        def _rebuild_leaves_if_needed(champion: Path, dig: str) -> None:
            """Restart leaf farm after live-pool topology change (iter boundary)."""
            if args.leaf_eval != "gpu-server" or visible < 1:
                return
            leaf_devices = hw.leaf_cuda_devices()
            if visible < 2:
                leaf_devices = [0] * max(1, hw.leaf_gpu0_replicas or hw.leaf_replicas_total)
            leaf_slots = max(proc_workers, hw.sim_workers)
            print(
                f"[pure_rl] live_pool rebuild leaves "
                f"gpu0={hw.leaf_gpu0_replicas} gpu1={hw.leaf_gpu1_replicas} "
                f"slots={leaf_slots}",
                flush=True,
            )
            leaf.stop()
            leaf.start(
                ckpt=champion,
                digest=dig,
                leaf_devices=leaf_devices,
                n_workers=leaf_slots,
                max_batch=None,
                coalesce_ms=coalesce_ms,
            )

        def _maybe_apply_live_pool(
            champion: Path, dig: str, *, allow_leaf_rebuild: bool = True
        ) -> None:
            nonlocal hw, proc_workers, live_pool_last_seq
            new_hw, new_proc, new_seq, plan, leaf_changed = apply_live_pool_plan(
                hw=hw,
                last_seq=live_pool_last_seq,
                multi_env_per_worker=multi_env_n,
                visible_gpu_count=visible,
            )
            if plan is None:
                return
            # Response queues are 1:1 with WorkerPool slots. Growing workers
            # without growing resp_qs → ValueError queues=32 workers=96 and a
            # dead train (2026-07-16). leaf_changed only tracks GPU topology;
            # also detect slot/queue undersize.
            n_qs = len(leaf.resp_qs) if leaf.resp_qs else 0
            slots_need = max(int(new_hw.sim_workers), int(new_proc))
            slots_changed = bool(n_qs > 0 and slots_need > n_qs)
            need_rebuild = bool(leaf_changed or slots_changed)
            # Defer leaf *topology* rebuild at boot (CUDA teardown race), but
            # never adopt a worker count above current queue capacity.
            do_leaf = bool(
                need_rebuild and allow_leaf_rebuild and leaf.remote_channel
            )
            deferred = bool(need_rebuild and not do_leaf)
            print(
                f"[pure_rl] live_pool_plan seq={plan.seq} apply "
                f"workers={hw.sim_workers}->{new_hw.sim_workers} "
                f"leaves={hw.leaf_replicas_total}->{new_hw.leaf_replicas_total} "
                f"procs={proc_workers}->{new_proc}"
                + (f" reason={plan.reason}" if plan.reason else "")
                + (
                    f" (leaf rebuild deferred; clamp workers to queues={n_qs})"
                    if deferred and n_qs > 0
                    else (
                        " (leaf rebuild deferred)"
                        if deferred
                        else ""
                    )
                ),
                flush=True,
            )
            if deferred:
                # Keep current leaf topology AND clamp workers to queue slots.
                clamp_w = min(int(new_hw.sim_workers), n_qs) if n_qs > 0 else int(
                    new_hw.sim_workers
                )
                clamp_proc = min(int(new_proc), n_qs) if n_qs > 0 else int(new_proc)
                new_hw = replace(
                    new_hw,
                    leaf_gpu0_replicas=hw.leaf_gpu0_replicas,
                    leaf_gpu1_replicas=hw.leaf_gpu1_replicas,
                    sim_workers=max(1, clamp_w),
                    games_in_flight=max(1, clamp_w),
                )
                new_proc = max(1, clamp_proc)
            hw = new_hw
            proc_workers = new_proc
            # A deferred topology plan is not consumed. Re-read it at the next
            # safe iteration boundary, where the leaf farm can actually rebuild.
            if not deferred:
                live_pool_last_seq = new_seq
            if do_leaf:
                _rebuild_leaves_if_needed(champion, dig)

        def _kick_collect(it: int, champion: Path, dig: str) -> dict[str, Any]:
            import shutil

            if fixed_cycle_updates and int(it) >= fixed_cycle_updates:
                raise RuntimeError(
                    "fixed cycle forbids collection at or beyond iter_"
                    f"{fixed_cycle_updates:05d}"
                )
            free_bytes = int(shutil.disk_usage(run_dir).free)
            minimum_free = int(float(args.min_free_disk_gb) * (1024 ** 3))
            if free_bytes < minimum_free:
                raise RuntimeError(
                    "free-disk guard refused a new replay shard: "
                    f"free={free_bytes / (1024 ** 3):.1f}GiB "
                    f"minimum={minimum_free / (1024 ** 3):.1f}GiB"
                )
            temp = _collect_temperature(args, it)
            pool_n = max(1, int(getattr(config.PURE_RL, "opponent_pool_size", 4)))
            recent = [
                _verified_checkpoint_identity(entry)
                for entry in opponent_pool[-pool_n:]
            ]
            official_target_weights: Optional[dict[str, float]] = None
            if bool(args.official_adaptive_targeting):
                exact_rates = _latest_official_heldout_win_rates(
                    loop_state,
                    tuple(str(spec.id) for spec in practice_specs),
                )
                official_target_weights = _adaptive_official_target_weights(
                    tuple(str(spec.id) for spec in practice_specs),
                    exact_rates,
                    target_win_rate=float(
                        args.strong_public_practice_target_wr
                        if active_gate is not None
                        else args.heldout_per_opponent_floor
                    ),
                    minimum_share=float(args.official_adaptive_min_share),
                    gap_power=float(args.official_adaptive_gap_power),
                    skill_weights=(
                        practice_skill_weights
                        if active_gate is not None
                        else None
                    ),
                )
                print(
                    "[pure_rl] adaptive strong-public practice targeting "
                    f"source={'latest_exact_heldout' if exact_rates else 'uniform_fallback'} "
                    f"rates={json.dumps(exact_rates, sort_keys=True)} "
                    f"weights={json.dumps(official_target_weights, sort_keys=True)}",
                    flush=True,
                )
            public_training_games = int(args.games_per_iter) - int(
                collection_group_plan["self_play"]
            )
            effective_practice_fraction = (
                float(collection_group_plan[STRONG_PUBLIC_PRACTICE_GROUP])
                / max(1, public_training_games)
            )
            self_jobs, base_jobs = _build_collect_jobs(
                n_games=args.games_per_iter,
                ckpt=champion,
                digest=dig,
                model_generation=it + 1,
                decks=decks,
                specs=collect_specs,
                seed=args.seed + it * ITERATION_SEED_STRIDE,
                game_timeout_s=args.game_timeout_s,
                mode=args.mode,
                collect_temperature=temp,
                max_context=pure_rl_model_config().max_context,
                opponent_pool=recent,
                self_play_frac=float(getattr(config.PURE_RL, "self_play_frac", 0.85)),
                ladder_mix=ladder_mix,
                family_manifest=(
                    family_training_contract["manifest"]
                    if family_training_contract is not None
                    else None
                ),
                iteration=int(it),
                priority_specs=(
                    practice_specs
                    if float(args.official_collect_frac) > 0.0
                    else None
                ),
                priority_frac=effective_practice_fraction,
                priority_weights=official_target_weights,
                priority_minimum_games=(practice_minimum_games or None),
                diverse_minimum_games=r274_diverse_minimum_games,
                priority_group=(
                    STRONG_PUBLIC_PRACTICE_GROUP
                    if active_gate is not None
                    else "official_target"
                ),
                priority_temperature=(
                    float(args.strong_public_practice_temperature)
                    if active_gate is not None
                    else None
                ),
                priority_archetypes=(
                    practice_archetypes if active_gate is not None else None
                ),
                priority_context=(
                    {
                        "active_gate_id": str(active_gate["id"]),
                        "formal_eval": False,
                        "seed_namespace": "train/strong-public-practice-v1",
                        "formal_gate_seed_namespace": (
                            "eval/strong-public-fixed-manifest-v1"
                        ),
                    }
                    if active_gate is not None
                    else None
                ),
                official_exploit_opponents=tuple(
                    args.official_exploit_opponents
                ),
                official_exploit_frac=float(args.official_exploit_frac),
                official_exploit_temperature=float(
                    args.official_exploit_temperature
                ),
                exact_training_seat_split=bool(
                    args.require_exact_training_seat_split
                ),
            )
            _assert_training_jobs_exclude_research_controls(
                [*self_jobs, *base_jobs], research_control_registry
            )
            r274_r195_research_plan: Optional[dict[str, Any]] = None
            r274_r195_research_plan_path: Optional[Path] = None
            if r274_r195_research_contract:
                r274_r195_research_plan = _assert_r274_r195_research_jobs(
                    base_jobs,
                    expected_content_digest=str(
                        r274_r195_research_contract["installed_content_sha256"]
                    ),
                    iteration=int(it),
                )
                r274_r195_research_plan["install_receipt"] = dict(
                    r274_r195_research_contract["receipt"]
                )
                r274_r195_research_plan_path = (
                    run_dir
                    / "collection_plans"
                    / f"r274-r195-55378392-iter-{it:05d}.json"
                )
                if r274_r195_research_plan_path.is_file():
                    prior = json.loads(
                        r274_r195_research_plan_path.read_text(encoding="utf-8")
                    )
                    if prior != r274_r195_research_plan:
                        raise RuntimeError(
                            "existing submission-55378392 research plan changed"
                        )
                else:
                    _write_json_exclusive(
                        r274_r195_research_plan_path,
                        r274_r195_research_plan,
                    )
            from collections import Counter

            public_groups = Counter(
                str((job.get("target_provenance") or {}).get(
                    "opponent_training_group"
                ) or "")
                for job in base_jobs
            )
            observed_group_plan = {
                "self_play": len(self_jobs),
                STRONG_PUBLIC_PRACTICE_GROUP: int(
                    public_groups.get(STRONG_PUBLIC_PRACTICE_GROUP, 0)
                ),
                "diverse_public": int(public_groups.get("diverse_public", 0)),
            }
            if observed_group_plan != collection_group_plan:
                raise RuntimeError(
                    "collection schedule disagrees with its fixed group budget: "
                    f"observed={observed_group_plan} expected={collection_group_plan}"
                )
            assigned_seat_stage = _training_seat_stage(
                "assigned_source_games",
                [job.get("our_seat") for job in [*self_jobs, *base_jobs]],
                expected_total=int(args.games_per_iter),
            )
            if bool(args.require_exact_training_seat_split):
                _assert_exact_training_seat_stage(assigned_seat_stage)
            assigned_manifest_sha256 = _canonical_digest(
                [
                    {
                        "job_index": int(job.get("job_index", -1)),
                        "our_seat": int(job.get("our_seat", -1)),
                        "archetype": str(job.get("archetype") or ""),
                        "opponent_id": str(job.get("opponent_id") or ""),
                        "opponent_training_group": str(
                            (job.get("target_provenance") or {}).get(
                                "opponent_training_group"
                            )
                            or "self_play"
                        ),
                    }
                    for job in [*self_jobs, *base_jobs]
                ]
            )
            practice_plan: Optional[dict[str, Any]] = None
            practice_plan_path: Optional[Path] = None
            if active_gate is not None and float(args.official_collect_frac) > 0.0:
                practice_plan = _assert_strong_public_practice_jobs(
                    all_jobs=[*self_jobs, *base_jobs],
                    public_jobs=base_jobs,
                    active_gate=active_gate,
                    expected_practice_games=int(
                        collection_group_plan[STRONG_PUBLIC_PRACTICE_GROUP]
                    ),
                    iteration=int(it),
                    root_seed=int(args.seed),
                    formal_games=int(args.heldout_games),
                    minimum_share=float(args.official_adaptive_min_share),
                    practice_temperature=float(
                        args.strong_public_practice_temperature
                    ),
                    minimum_games_by_opponent=(practice_minimum_games or None),
                )
                practice_plan["adaptive_weights"] = dict(
                    official_target_weights or {}
                )
                practice_plan["research_controls"] = {
                    "training_eligible": False,
                    "replay_eligible": False,
                    "additive_measurement_games": (
                        0
                        if bool(args.disable_research_control_measurement)
                        else int(args.research_control_games_per_iter)
                    ),
                    "legacy_reclaimed_training_slots": int(
                        args.research_control_games_per_iter
                    ),
                    "disabled_by_revision_25": bool(
                        args.disable_research_control_measurement
                    ),
                    "stage": "measure:research_controls",
                }
                practice_plan["group_games_per_iteration"] = dict(
                    collection_group_plan
                )
                practice_plan_path = (
                    run_dir / "collection_plans" / f"iter_{it:05d}.json"
                )
                if practice_plan_path.is_file():
                    prior_plan = json.loads(
                        practice_plan_path.read_text(encoding="utf-8")
                    )
                    if prior_plan != practice_plan:
                        raise RuntimeError(
                            "existing strong-public practice plan changed on retry"
                        )
                else:
                    _write_json_exclusive(practice_plan_path, practice_plan)
            refill_fraction = float(
                os.environ.get("POKEBOT_SELF_PLAY_REFILL_CAPACITY_FRAC", "0")
            )
            refill_jobs = _self_play_refill_capacity_jobs(
                self_jobs,
                fraction=refill_fraction,
                first_job_index=(
                    max(
                        [
                            int(job.get("job_index", -1))
                            for job in [*self_jobs, *base_jobs]
                        ],
                        default=-1,
                    )
                    + 1
                ),
            )
            if refill_jobs:
                print(
                    f"[pure_rl] collect iter={it} bounded self-play refill "
                    f"capacity={len(refill_jobs)} primary_self_play={len(self_jobs)} "
                    f"target_games={int(args.games_per_iter)}",
                    flush=True,
                )
                self_jobs = [*self_jobs, *refill_jobs]
            shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
            if shard.is_file():
                partial_resume = _verified_partial_collection_resume(
                    shard, iteration=int(it)
                )
                if partial_resume is None or str(
                    partial_resume.get("behavior_checkpoint_digest") or ""
                ) != str(dig):
                    raise RuntimeError(
                        f"refusing to overwrite immutable collect shard: {shard}"
                    )
                print(
                    "[pure_rl] append-only partial collection kick authorized "
                    f"iter={it} retained={partial_resume['retained_source_games']} "
                    f"missing={len(partial_resume['missing_job_indices'])}",
                    flush=True,
                )
            collect_started = time.time()
            _atomic_json(
                run_dir / "iteration_runtime.json",
                {
                    "schema": "poke_bot.iteration_runtime/v1",
                    "iteration": int(it),
                    "phase": "collect",
                    "started_at": collect_started,
                    "checkpoint_digest": str(dig),
                    "updated_at": collect_started,
                },
            )
            print(
                f"[pure_rl] collect iter={it} self_play={len(self_jobs)} "
                "research_controls=0(replay) "
                f"public_mix={len(base_jobs)} "
                f"public_total={len(base_jobs)} "
                f"strong_public_practice="
                f"{public_groups.get(STRONG_PUBLIC_PRACTICE_GROUP, 0)} "
                f"diverse_public={public_groups.get('diverse_public', 0)} "
                f"n_decks={len(decks)} sample={deck_names[:12]} "
                f"local_workers={hw.sim_workers} "
                f"self_play_procs={proc_workers} multi_env={multi_env_n} "
                f"remote_workers="
                f"{remote_farm.total_workers if remote_farm else 0}",
                flush=True,
            )
            writer, rows, stats = _collect_wave(
                self_play_jobs=self_jobs,
                baseline_jobs=base_jobs,
                shard_path=shard,
                n_workers=hw.sim_workers,
                leaf_channel=leaf.remote_channel,
                remote_farm=remote_farm,
                worker_play=worker_play,
                worker_self_play=worker_self_play,
                multi_env_per_worker=multi_env_n,
                worker_self_play_multi=worker_self_play_multi,
                iteration=int(it),
                required_mirror_archetype=(
                    str(args.specialist_archetype)
                    if args.mode == "specialist"
                    else None
                ),
            )
            collect_elapsed = max(time.time() - collect_started, 1e-6)
            requested = int(args.games_per_iter)
            scheduled_attempts = len(self_jobs) + len(base_jobs)
            retained_jobs = int(stats.get("with_record", 0))
            r274_r195_research_retained: Optional[dict[str, Any]] = None
            if r274_r195_research_contract:
                r274_r195_research_retained = _assert_r274_r195_research_jobs(
                    rows,
                    expected_content_digest=str(
                        r274_r195_research_contract["installed_content_sha256"]
                    ),
                    iteration=int(it),
                )
                if (
                    r274_r195_research_plan is None
                    or int(r274_r195_research_retained["games"])
                    != int(r274_r195_research_plan["games"])
                    or int(r274_r195_research_retained["learner_first_games"])
                    != int(r274_r195_research_plan["learner_first_games"])
                    or int(r274_r195_research_retained["learner_second_games"])
                    != int(r274_r195_research_plan["learner_second_games"])
                ):
                    raise RuntimeError(
                        "retained submission-55378392 cell differs from its immutable schedule"
                    )
            stats.update(
                {
                    "requested_games": requested,
                    "scheduled_game_attempts": scheduled_attempts,
                    "self_play_refill_capacity_games": len(refill_jobs),
                    "retained_source_games": retained_jobs,
                    "retained_trajectories": writer.n_games,
                    "usable_game_fraction": (
                        min(1.0, retained_jobs / requested) if requested else 1.0
                    ),
                    "collect_elapsed_sec": collect_elapsed,
                    "claimed_games_per_sec": scheduled_attempts / collect_elapsed,
                    "valid_source_games_per_sec": retained_jobs / collect_elapsed,
                    "trajectory_games_per_sec": writer.n_games / collect_elapsed,
                    "strong_public_practice_plan": (
                        str(practice_plan_path)
                        if practice_plan_path is not None
                        else None
                    ),
                    "r274_submission_55378392_research_plan": (
                        {
                            **dict(r274_r195_research_plan or {}),
                            "retained": r274_r195_research_retained,
                            "path": str(r274_r195_research_plan_path),
                            "sha256": _sha256_file(r274_r195_research_plan_path),
                        }
                        if r274_r195_research_plan_path is not None
                        else None
                    ),
                }
            )
            retained_seat_stage = _training_seat_stage(
                "retained_source_games",
                [row.get("our_seat") for row in rows],
                expected_total=requested,
            )
            actual_manifest_sha256 = _canonical_digest(
                [
                    {
                        "job_index": int(row.get("job_index", -1)),
                        "our_seat": int(row.get("our_seat", -1)),
                        "archetype": str(row.get("archetype") or ""),
                        "opponent_id": str(row.get("opponent_id") or ""),
                    }
                    for row in sorted(
                        rows, key=lambda value: int(value.get("job_index", -1))
                    )
                ]
            )
            stats["training_seat_split"] = {
                "schema": _TRAINING_SEAT_SPLIT_SUMMARY_SCHEMA,
                "required": bool(args.require_exact_training_seat_split),
                "policy": (
                    "exact_50_50_first_second_training"
                    if bool(args.require_exact_training_seat_split)
                    else "ordinary_balanced_scheduler"
                ),
                "second_seat_priority": False,
                "assigned_source_games": assigned_seat_stage,
                "retained_source_games": retained_seat_stage,
                "stage_manifest_sha256": {
                    "assigned": assigned_manifest_sha256,
                    "actual": actual_manifest_sha256,
                },
                "passed": bool(
                    assigned_seat_stage["exact_50_50"]
                    and retained_seat_stage["exact_50_50"]
                ),
            }
            if bool(args.require_exact_training_seat_split):
                _assert_exact_training_seat_stage(retained_seat_stage)
            print(
                f"[pure_rl] collect done iter={it} ok={stats.get('ok')} "
                f"leaf_remote={stats.get('leaf_remote')} "
                f"leaf_modes={stats.get('leaf_modes')} "
                f"multi_env_games={stats.get('multi_env_games')}",
                flush=True,
            )
            if retained_jobs != requested:
                failed = shard.with_name(
                    f"{shard.stem}.failed."
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    f"{shard.suffix}"
                )
                if shard.exists():
                    shard.replace(failed)
                _write_json_exclusive(
                    failed.with_suffix(failed.suffix + ".failure.json"),
                    {
                        "iteration": it,
                        "reason": "exact_source_game_count_mismatch",
                        "expected_source_games": requested,
                        "stats": stats,
                        "quarantined_shard": str(failed),
                    },
                )
                raise RuntimeError(
                    f"collect iter={it} retained {retained_jobs}/{requested} "
                    "source games (exact count required); partial shard quarantined "
                    f"at {failed}"
                )
            stats["collect_completed_at"] = time.time()
            collection_receipt = _commit_completed_collection_receipt(
                run_dir=run_dir,
                state=loop_state,
                contract=design_contract,
                iteration=it,
                shard=shard,
                checkpoint=champion,
                checkpoint_digest=dig,
                stats=stats,
                started_at=collect_started,
                writer=writer,
            )
            return {
                "iteration": it,
                "shard": shard,
                "writer": writer,
                "stats": stats,
                "checkpoint": champion,
                "digest": dig,
                "started_at": collect_started,
                "receipt": collection_receipt["receipt_path"],
            }

        def _prepare_expert_rehearsal(
            it: int, parent: Any
        ) -> tuple[Any, dict[str, Any]]:
            """Recover or run the immutable ladder complement pass."""
            from poke_bot import checkpoint as checkpoint_mod

            rehearsal_epochs = rehearsal_epochs_for_iteration(
                it,
                ordinary_epochs=int(args.expert_rehearsal_epochs),
                one_time_before=int(args.expert_rehearsal_one_time_before),
                one_time_epochs=int(args.expert_rehearsal_one_time_epochs),
            )
            if rehearsal_epochs != int(args.expert_rehearsal_epochs):
                print(
                    "[pure_rl] OWNER_ONE_TIME_EXPERT_REFRESH "
                    f"before_iter={it} epochs={rehearsal_epochs} "
                    f"ordinary_epochs={int(args.expert_rehearsal_epochs)}",
                    flush=True,
                )

            adapter_rehearsal_record: Optional[dict[str, Any]] = None
            if args.expert_matchup_adapter_manifest is not None:
                print(
                    "[pure_rl] expert matchup-adapter rehearsal begin "
                    f"before_iter={it} epochs={int(args.expert_matchup_adapter_epochs)} "
                    "optimizer=adapter_bank_only runtime=off",
                    flush=True,
                )
                adapter_rehearsal_record = (
                    run_or_recover_expert_adapter_rehearsal(
                        run_dir=run_dir,
                        before_iteration=it,
                        parent_checkpoint=parent.path,
                        parent_digest=parent.digest,
                        staged_manifest=Path(
                            args.expert_matchup_adapter_manifest
                        ),
                        epochs=int(args.expert_matchup_adapter_epochs),
                        learning_rate=float(args.expert_matchup_adapter_lr),
                        games_per_batch=int(
                            args.expert_matchup_adapter_games_per_batch
                        ),
                        max_decisions_per_batch=int(
                            args.expert_matchup_adapter_max_decisions_per_batch
                        ),
                        seed=args.seed + 5_050_000 + it,
                        device=train_dev,
                        max_process_rss_gib=24.0,
                        min_available_ram_gib=12.0,
                    )
                )
                parent = _verified_checkpoint_identity(
                    adapter_rehearsal_record["checkpoint"]
                )
                print(
                    "[pure_rl] expert matchup-adapter rehearsal committed "
                    f"before_iter={it} checkpoint={parent.digest[:19]}… "
                    f"rows={int(dict(adapter_rehearsal_record.get('fit') or {}).get('phase_rows', 0))}",
                    flush=True,
                )
            loss_weights = {
                "value": 1.0,
                "archetype": float(args.archetype_aux_loss_weight),
                "opponent_hand": float(args.opp_hand_loss_weight),
                "opponent_hidden_remainder": float(
                    args.opp_remainder_loss_weight
                ),
                "lethal_threat": float(args.lethal_threat_loss_weight),
                "prize_race": float(args.prize_race_loss_weight),
                "alakazam_guide": float(
                    args.alakazam_guide_loss_weight
                    if args.expert_rehearsal_guide_loss_weight is None
                    else args.expert_rehearsal_guide_loss_weight
                ),
            }
            family_residual_loss_weights = (
                dict(family_training_contract["residual_objectives"])
                if family_training_contract is not None
                else {}
            )
            corpus_split_seed = int(args.seed) + 5_000_000

            trusted_parent = checkpoint_mod.assert_trusted_policy_checkpoint(
                parent.path
            )
            parent_profile = dict(trusted_parent.get("model_config") or {})
            parent_checkpoint = checkpoint_mod.load_checkpoint(
                parent.path, map_location="cpu"
            )
            parent_expanded_training = dict(
                (parent_checkpoint.get("extra") or {}).get(
                    "expanded_head_training"
                )
                or {}
            )
            expanded_head_contract: dict[str, Any] = {}
            if bool(parent_profile.get("expanded_heads_enabled", False)):
                if (
                    parent_expanded_training.get("schema")
                    != "poke_bot.expanded_head_training/v1"
                    or parent_expanded_training.get("runtime_enabled_heads")
                    != []
                ):
                    raise RuntimeError(
                        "expanded specialist parent lacks its shadow-only "
                        "training contract"
                    )
                expanded_head_contract = {
                    "schema": "poke_bot.expanded_head_schedule/v1",
                    "target_schema": str(
                        parent_expanded_training.get(
                            "target_schema_version"
                        )
                        or ""
                    ),
                    "target_schema_digest": str(
                        parent_expanded_training.get(
                            "target_schema_digest"
                        )
                        or ""
                    ),
                    "schedule_digest": str(
                        parent_expanded_training.get("schedule_digest") or ""
                    ),
                    "epoch": 25,
                    "stage_index": 5,
                    "loss_weights": dict(
                        parent_expanded_training.get("loss_weights") or {}
                    ),
                    "runtime_enabled_heads": [],
                    "rehearsal_iteration": int(it),
                }
            rehearsal_context = (
                int(parent_profile.get("max_context") or 0)
                if trusted_parent.get("decision_context") == "history"
                else None
            )
            if rehearsal_context is not None and rehearsal_context <= 0:
                raise RuntimeError(
                    "temporal expert rehearsal requires a positive max_context"
                )
            exact_rehearsal_enabled = any(
                float(weight) > 0.0
                for weight in (
                    args.archetype_aux_loss_weight,
                    args.opp_hand_loss_weight,
                    args.opp_remainder_loss_weight,
                    args.lethal_threat_loss_weight,
                    args.prize_race_loss_weight,
                )
            ) or bool(expanded_head_contract)
            rehearsal_belief_card_vocab: Optional[int] = None
            if exact_rehearsal_enabled:
                # ``assert_trusted_policy_checkpoint`` deliberately returns
                # only validated metadata.  Read tensors from the already
                # trusted checkpoint instead of assuming the metadata view
                # contains ``model_state_dict``.
                parent_state = dict(
                    parent_checkpoint.get("model_state_dict") or {}
                )
                rehearsal_belief_card_vocab = belief_card_vocab_from_state(
                    parent_state
                )
                missing_rehearsal_heads = sorted(
                    name
                    for name in (
                        "opp_hand_head",
                        "opp_remainder_head",
                        "lethal_threat_head",
                        "prize_race_head",
                    )
                    if f"{name}.weight" not in parent_state
                )
                if missing_rehearsal_heads:
                    print(
                        "[pure_rl] EXPERT_HEAD_WARM_START "
                        f"before_iter={it} missing={missing_rehearsal_heads} "
                        f"belief_card_vocab={rehearsal_belief_card_vocab} "
                        "optimizer=fresh",
                        flush=True,
                    )
            manifest_identity = resolve_expert_manifest(
                Path(args.expert_manifest),
                min_decisions=int(args.expert_min_decisions),
                require_protected=True,
                required_archetype=str(args.specialist_archetype or ""),
                required_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
                required_max_context=rehearsal_context,
                required_target_coverage=expert_required_targets,
                required_expanded_target_schema=str(
                    expanded_head_contract.get("target_schema") or ""
                ),
                required_expanded_target_digest=str(
                    expanded_head_contract.get("target_schema_digest") or ""
                ),
                required_expanded_heads=tuple(
                    str(name)
                    for name, weight in dict(
                        expanded_head_contract.get("loss_weights") or {}
                    ).items()
                    if float(weight) > 0.0
                ),
            )
            option_conditioned_loss_weights = {
                "combo_state": float(args.combo_state_loss_weight)
            }
            if option_conditioned_loss_weights["combo_state"] <= 0.0:
                option_conditioned_loss_weights = {}
            exact_rehearsal_seats = bool(
                args.require_exact_training_seat_split
            )
            training_seat_split_receipt: Optional[dict[str, Any]] = None
            if exact_rehearsal_seats:
                _checkpoint_path, existing_rehearsal_path = rehearsal_paths(
                    run_dir, it
                )
                if existing_rehearsal_path.is_file():
                    existing_rehearsal = json.loads(
                        existing_rehearsal_path.read_text(encoding="utf-8")
                    )
                    declared_receipt = dict(
                        existing_rehearsal.get("training_seat_split_receipt")
                        or {}
                    )
                    if declared_receipt:
                        training_seat_split_receipt = declared_receipt
                if training_seat_split_receipt is None:
                    existing_seat_index = (
                        run_dir
                        / "seat_split_receipts"
                        / f"rehearsal_before_iter_{int(it):05d}.index.json"
                    )
                    if existing_seat_index.is_file():
                        existing_index = json.loads(
                            existing_seat_index.read_text(encoding="utf-8")
                        )
                        if existing_index.get("schema") != (
                            "poke_bot.alakazam_refresh_rehearsal_seat_split_index/v1"
                        ):
                            raise RuntimeError(
                                "existing rehearsal seat index schema changed"
                            )
                        training_seat_split_receipt = {
                            "schema": str(existing_index["schema"]),
                            "path": str(existing_seat_index.resolve()),
                            "sha256": _sha256_file(existing_seat_index),
                        }
            if args.r280_contiguous_expert_pack is not None:
                from poke_bot.r280_gpu_resident_refresh import (
                    run_refresh as run_r280_gpu_resident_refresh,
                    validate_refresh_receipt as validate_r280_refresh_receipt,
                )

                rehearsal_checkpoint, receipt_path = rehearsal_paths(run_dir, it)
                pack_path = Path(args.r280_contiguous_expert_pack).expanduser().resolve()
                pack_receipt_path = Path(
                    args.r280_contiguous_expert_pack_receipt
                ).expanduser().resolve()
                if receipt_path.is_file():
                    record = validate_r280_refresh_receipt(
                        receipt_path,
                        expected_parent_sha256=parent.digest,
                        expected_pack_sha256=_sha256_file(pack_path),
                        expected_before_iteration=int(it),
                    )
                else:
                    record = run_r280_gpu_resident_refresh(
                        pack_path=pack_path,
                        pack_receipt_path=pack_receipt_path,
                        parent_path=Path(parent.path),
                        parent_digest=parent.digest,
                        output_path=rehearsal_checkpoint,
                        receipt_path=receipt_path,
                        before_iteration=int(it),
                        device_name=str(args.r280_refresh_device),
                        requested_batch_size=int(args.expert_rehearsal_batch_size),
                        learning_rate=float(args.expert_rehearsal_lr),
                        seed=int(args.seed) + 5_100_000 + int(it),
                        loss_weights={
                            "archetype": float(args.archetype_aux_loss_weight),
                            "opponent_hand": float(args.opp_hand_loss_weight),
                            "opponent_hidden_remainder": float(
                                args.opp_remainder_loss_weight
                            ),
                            "lethal_threat": float(
                                args.lethal_threat_loss_weight
                            ),
                            "prize_race": float(args.prize_race_loss_weight),
                            "setup_board_outcome": float(
                                args.setup_board_outcome_loss_weight
                            ),
                            "combo_state": float(args.combo_state_loss_weight),
                        },
                        expanded_head_schedule=expanded_head_contract,
                    )
                prepared = _verified_checkpoint_identity(
                    {
                        "path": record["checkpoint"]["path"],
                        "digest": record["checkpoint"]["sha256"],
                    }
                )
                record = {
                    **record,
                    "checkpoint_identity": prepared.as_dict(),
                    "reused": bool(receipt_path.is_file()),
                }
                print(
                    "[pure_rl] r280 GPU-resident expert refresh committed "
                    f"before_iter={it} checkpoint={prepared.digest[:19]}…",
                    flush=True,
                )
                return prepared, record
            if r260_own_deck_training_inputs:
                # The successor never enters the resident corpus path: its
                # Inzi-only index streams and exact-key attaches one bounded
                # host batch at a time.
                from poke_bot.r260_inzi_sidecar_index import R260InziSidecarIndex

                rehearsal_checkpoint, receipt_path = rehearsal_paths(run_dir, it)
                index_path = run_dir / "r260_sidecar_indexes" / (
                    f"expert-{r260_own_deck_training_inputs['sidecar_binding_sha256'][7:19]}.sqlite3"
                )
                if index_path.exists():
                    index = R260InziSidecarIndex(
                        index_path,
                        source_manifest_sha256=r260_own_deck_training_inputs["source_manifest_sha256"],
                        daily_meta_sha256s=r260_own_deck_training_inputs["daily_meta_sha256s"],
                    )
                    index.assert_verified(
                        expected_source_manifest_sha256=r260_own_deck_training_inputs["source_manifest_sha256"],
                        daily_meta_sha256s=r260_own_deck_training_inputs["daily_meta_sha256s"],
                    )
                else:
                    index = R260InziSidecarIndex.build(
                        sidecar_root=r260_own_deck_training_inputs["inzi_sidecar_root"],
                        output=index_path,
                        source_manifest_sha256=r260_own_deck_training_inputs["source_manifest_sha256"],
                        daily_meta_sha256s=r260_own_deck_training_inputs["daily_meta_sha256s"],
                    )
                if receipt_path.exists():
                    existing = _validate_r260_scheduled_rehearsal_receipt(
                        json.loads(receipt_path.read_text(encoding="utf-8")),
                        expected_r260_inputs=r260_own_deck_training_inputs,
                    )
                    prepared = _verified_checkpoint_identity(
                        {
                            "path": existing["post_checkpoint"]["path"],
                            "digest": existing["post_checkpoint"]["sha256"],
                        }
                    )
                    return prepared, existing
                stream_cfg = TrainConfig.pure_rl_defaults(
                    epochs=rehearsal_epochs, seed=args.seed + 5_100_000 + it,
                    amp=train_dev.type == "cuda", max_decisions_per_batch=int(args.expert_rehearsal_batch_size),
                    visible_tutor_completion_loss_weight=0.025,
                    terminal_conversion_loss_weight=0.025,
                    tactical_sequence_outcome_loss_weight=(
                        0.025
                        if r260_own_deck_training_inputs[
                            "tactical_sequence_enabled"
                        ]
                        else 0.0
                    ),
                    collect_own_deck_promotion_metrics=True,
                )
                result = streaming_r260_host_rehearsal_step(
                    manifest_path=manifest_identity.path, manifest_digest=manifest_identity.digest,
                    sidecar_index=index, base_ckpt=parent.path, output_path=rehearsal_checkpoint,
                    archetype_id=str(args.specialist_archetype), epochs=rehearsal_epochs,
                    cfg=stream_cfg, seed=corpus_split_seed, max_context=rehearsal_context,
                    batch_games=max(1, int(args.expert_rehearsal_batch_size) // 128),
                    manifest_workers=int(args.expert_manifest_workers),
                    tactical_overlay_path=(
                        None
                        if disable_rl_tactical_cotrain
                        else r260_own_deck_training_inputs[
                            "expert_tactical_overlay_identity"
                        ]["path"]
                    ),
                )
                rehearsal_loss_weights = {
                    "visible_tutor_completion": 0.025,
                    "terminal_conversion": 0.025,
                }
                if r260_own_deck_training_inputs["tactical_sequence_enabled"]:
                    rehearsal_loss_weights["tactical_sequence_outcome"] = 0.025
                record = {
                    "schema": "poke_bot.r260_scheduled_expert_rehearsal/v1",
                    "status": "passed",
                    "owner_contract_sha256": r260_own_deck_training_inputs[
                        "owner_contract_sha256"
                    ],
                    "migration_receipt_sha256": r260_own_deck_training_inputs[
                        "migration_receipt_sha256"
                    ],
                    "canary_activation_receipt_sha256": (
                        r260_own_deck_training_inputs[
                            "canary_activation_receipt_sha256"
                        ]
                    ),
                    "sidecar_binding_sha256": r260_own_deck_training_inputs[
                        "sidecar_binding_sha256"
                    ],
                    "pre_checkpoint": _r260_file_identity(parent.path),
                    "post_checkpoint": _r260_file_identity(
                        Path(result["candidate_path"])
                    ),
                    "index": _r260_file_identity(index.path),
                    "inzi_dataset": dict(
                        r260_own_deck_training_inputs["inzi_dataset_identity"]
                    ),
                    "expert_tactical_overlay": dict(
                        r260_own_deck_training_inputs[
                            "expert_tactical_overlay_identity"
                        ]
                    ),
                    "index_provenance": {
                        "schema": "poke_bot.r260_inzi_sidecar_index/v1",
                        "source_manifest_sha256": (
                            r260_own_deck_training_inputs[
                                "source_manifest_sha256"
                            ]
                        ),
                        "daily_meta_sha256s": dict(
                            r260_own_deck_training_inputs[
                                "daily_meta_sha256s"
                            ]
                        ),
                    },
                    "loss_weights": rehearsal_loss_weights,
                    "rl_iteration_before_after": [
                        int(result["rl_iteration"]),
                        int(result["rl_iteration"]),
                    ],
                    "bounded_batch_games": result["max_games_per_batch"],
                    "full_window_device_resident": False,
                    "sampled_keys": [list(key) for key in result["sampled_keys"]],
                    "gradient_reachability": result["gradient_reachability"],
                    "tactical_exact_root_count": result[
                        "tactical_exact_root_count"
                    ],
                }
                record["receipt_sha256"] = _canonical_digest(record)
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                with receipt_path.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                validated = _validate_r260_scheduled_rehearsal_receipt(
                    record,
                    expected_r260_inputs=r260_own_deck_training_inputs,
                )
                return _verified_checkpoint_identity(
                    {
                        "path": validated["post_checkpoint"]["path"],
                        "digest": validated["post_checkpoint"]["sha256"],
                    }
                ), validated
            # A completed rehearsal is immutable training evidence.  Reuse it
            # before rebuilding the CPU pack, including in exact-seat mode;
            # otherwise a benign later learner-cap migration would recompute a
            # receipt-bound index under a new design fingerprint and fail.
            record = recover_rehearsal(
                run_dir,
                before_iteration=it,
                parent_digest=parent.digest,
                epochs=rehearsal_epochs,
                learning_rate=float(args.expert_rehearsal_lr),
                manifest_identity=manifest_identity,
                loss_weights=loss_weights,
                corpus_split_seed=corpus_split_seed,
                expanded_head_contract=expanded_head_contract,
                option_conditioned_loss_weights=option_conditioned_loss_weights,
                archetype_residual_loss_weights=family_residual_loss_weights,
                training_seat_split_receipt=training_seat_split_receipt,
                r241_peak_r195_training_contract=(
                    {} if r260_own_deck_training_inputs else r241_peak_r195_training_contract
                ),
            )
            if record is None:
                print(
                    f"pure_rl train:expert iter={it}:   0%|"
                    "                                    | 0/1 "
                    "[00:00<?, ?expert pass/s, loading corpus]",
                    file=sys.stderr,
                    flush=True,
                )
                corpus = expert_cache.prepare(
                    manifest_identity,
                    device=train_dev,
                    seed=corpus_split_seed,
                    max_context=rehearsal_context,
                    belief_card_vocab=rehearsal_belief_card_vocab,
                    require_exact_seat_split=exact_rehearsal_seats,
                )
                if exact_rehearsal_seats:
                    training_seat_split_receipt = (
                        _commit_expert_rehearsal_seat_split_receipts(
                            run_dir=run_dir,
                            before_iteration=it,
                            design_fingerprint=design_fingerprint,
                            evidence=dict(expert_cache.seat_split_evidence),
                            pack_info=dict(expert_cache.pack_info or {}),
                        )
                    )
                    record = recover_rehearsal(
                        run_dir,
                        before_iteration=it,
                        parent_digest=parent.digest,
                        epochs=rehearsal_epochs,
                        learning_rate=float(args.expert_rehearsal_lr),
                        manifest_identity=manifest_identity,
                        loss_weights=loss_weights,
                        corpus_split_seed=corpus_split_seed,
                        expanded_head_contract=expanded_head_contract,
                        option_conditioned_loss_weights=(
                            option_conditioned_loss_weights
                        ),
                        archetype_residual_loss_weights=(
                            family_residual_loss_weights
                        ),
                        training_seat_split_receipt=(
                            training_seat_split_receipt
                        ),
                        r241_peak_r195_training_contract=(
                            r241_peak_r195_training_contract
                        ),
                    )
            if record is None:
                rehearsal_checkpoint, _receipt_path = rehearsal_paths(run_dir, it)
                derivative_semantic_refresh = (
                    args.rule_derivative_semantic_pack_completion is not None
                    and parent_checkpoint.get("schema")
                    == "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
                )
                base_rehearsal_checkpoint = rehearsal_checkpoint
                if derivative_semantic_refresh:
                    base_rehearsal_checkpoint = rehearsal_checkpoint.with_name(
                        rehearsal_checkpoint.stem
                        + f".base-only-{parent.digest[7:19]}"
                        + rehearsal_checkpoint.suffix
                    )
                print(
                    f"[pure_rl] expert rehearsal begin before_iter={it} "
                    f"decisions={manifest_identity.decisions} "
                    f"days={manifest_identity.dates[0]}..{manifest_identity.dates[-1]} "
                    f"parent={parent.digest[:19]}…",
                    flush=True,
                )
                def _run_supervised_rehearsal(
                    args: argparse.Namespace,
                ) -> dict[str, Any]:
                    return supervised_rehearsal_step(
                        corpus,
                        base_ckpt=parent.path,
                        output_path=base_rehearsal_checkpoint,
                        parent_digest=parent.digest,
                        rehearsal_iteration=it,
                        manifest_identity=manifest_identity.as_dict(),
                        epochs=rehearsal_epochs,
                        lr=float(args.expert_rehearsal_lr),
                        requested_batch_size=int(
                            args.expert_rehearsal_batch_size
                        ),
                        seed=args.seed + 5_100_000 + it,
                        corpus_split_seed=corpus_split_seed,
                        device=train_dev,
                        aux_loss_weight=float(args.archetype_aux_loss_weight),
                        opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                        opp_remainder_loss_weight=float(
                            args.opp_remainder_loss_weight
                        ),
                        lethal_threat_loss_weight=float(
                            args.lethal_threat_loss_weight
                        ),
                        prize_race_loss_weight=float(
                            args.prize_race_loss_weight
                        ),
                        alakazam_guide_loss_weight=float(
                            args.alakazam_guide_loss_weight
                        ),
                        current_deck_guide_training_mode=str(
                            args.current_deck_guide_training_mode
                        ),
                        setup_board_outcome_loss_weight=float(
                            args.setup_board_outcome_loss_weight
                        ),
                        combo_state_loss_weight=float(args.combo_state_loss_weight),
                        current_deck_guide_curriculum_spec=str(
                            args.current_deck_guide_curriculum_spec or ""
                        ),
                        current_deck_guide_head_role_map=str(
                            args.current_deck_guide_head_role_map or ""
                        ),
                        current_deck_guide_curriculum_validation_receipt=str(
                            args.current_deck_guide_curriculum_validation_receipt
                            or ""
                        ),
                        expanded_head_loss_weights=dict(
                            expanded_head_contract.get("loss_weights") or {}
                        ),
                        archetype_residual_loss_weights=(
                            family_residual_loss_weights
                        ),
                        expanded_head_schedule=expanded_head_contract,
                        output_archetype_id=(
                            "core"
                            if args.mode == "core"
                            else str(args.specialist_archetype)
                        ),
                        output_model_id=(
                            f"{args.run_name}.expert-before-iter{it:05d}"
                        ),
                        training_seat_split_receipt=(
                            training_seat_split_receipt
                        ),
                        r241_peak_r195_training_contract=(
                            r241_peak_r195_training_contract
                        ),
                    )

                rehearsal_args = argparse.Namespace(**vars(args))
                if args.expert_rehearsal_guide_loss_weight is not None:
                    rehearsal_args.alakazam_guide_loss_weight = float(
                        loss_weights["alakazam_guide"]
                    )
                rehearsal_result = _run_supervised_rehearsal(rehearsal_args)
                semantic_refresh_record: Optional[dict[str, Any]] = None
                if derivative_semantic_refresh:
                    from scripts.train_alakazam_rule_derivative_full_r10 import (
                        refresh_semantic_checkpoint,
                    )

                    semantic_refresh_record = refresh_semantic_checkpoint(
                        base_checkpoint=base_rehearsal_checkpoint,
                        output=rehearsal_checkpoint,
                        completion_path=Path(
                            args.rule_derivative_semantic_pack_completion
                        ),
                        device=train_dev,
                        epochs=rehearsal_epochs,
                        block_decisions=int(
                            args.rule_derivative_semantic_block_decisions
                        ),
                        learning_rate=float(
                            args.rule_derivative_semantic_refresh_lr
                        ),
                        seed=args.seed + 5_200_000 + it,
                        before_iteration=it,
                    )
                    rehearsal_result = {
                        **rehearsal_result,
                        "candidate_path": semantic_refresh_record[
                            "checkpoint_path"
                        ],
                        "candidate_digest": semantic_refresh_record[
                            "checkpoint_sha256"
                        ],
                        "semantic_refresh": semantic_refresh_record,
                    }
                record = commit_rehearsal_receipt(
                    run_dir,
                    before_iteration=it,
                    parent_digest=parent.digest,
                    manifest=manifest_identity,
                    epochs=rehearsal_epochs,
                    learning_rate=float(args.expert_rehearsal_lr),
                    loss_weights=loss_weights,
                    corpus_split_seed=corpus_split_seed,
                    result=rehearsal_result,
                    expanded_head_contract=expanded_head_contract,
                    option_conditioned_loss_weights=(
                        option_conditioned_loss_weights
                    ),
                    archetype_residual_loss_weights=(
                        family_residual_loss_weights
                    ),
                    training_seat_split_receipt=(
                        training_seat_split_receipt
                    ),
                    r241_peak_r195_training_contract=(
                        r241_peak_r195_training_contract
                    ),
                )
                if semantic_refresh_record is not None:
                    record = {
                        **record,
                        "semantic_refresh": semantic_refresh_record,
                    }
            prepared = _verified_checkpoint_identity(record["checkpoint_identity"])
            print(
                f"[pure_rl] expert rehearsal committed before_iter={it} "
                f"checkpoint={prepared.digest[:19]}… "
                f"reused={int(bool(record.get('reused')))}",
                flush=True,
            )
            if adapter_rehearsal_record is not None:
                record = {
                    **record,
                    "matchup_adapter_rehearsal": adapter_rehearsal_record,
                }
            return prepared, record

        def _wait_for_r274_submission(
            *,
            boundary_update: int,
            checkpoint_identity: Any,
            rehearsal_receipt_path: Path,
        ) -> Optional[dict[str, Any]]:
            """Fence every r274 five-update block on its Kaggle upload receipt."""

            if fixed_cycle_updates != R274_FIXED_CYCLE_UPDATES:
                return None
            boundary_root = Path(args.r274_submission_boundary_dir)
            owner_contract_sha256 = str(
                r260_own_deck_training_inputs.get("owner_contract_sha256") or ""
            )
            if int(boundary_update) == 1:
                r304_contract = os.environ.get("POKEBOT_R304_OWNER_CONTRACT", "").strip()
                if not r304_contract:
                    raise RuntimeError(
                        "revision-304 submission fence lacks its owner contract path"
                    )
                owner_contract_sha256 = _r241_file_identity(
                    Path(r304_contract)
                )["digest"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", owner_contract_sha256):
                raise RuntimeError(
                    "r274 submission fence lacks its canonical owner contract digest"
                )
            request_path, request = _materialize_r274_submission_request(
                root=boundary_root,
                boundary_update=boundary_update,
                checkpoint_identity=checkpoint_identity.as_dict(),
                rehearsal_receipt_identity=_r241_file_identity(
                    rehearsal_receipt_path
                ),
                owner_contract_sha256=owner_contract_sha256,
            )
            _, upload_path = _r274_submission_boundary_paths(
                boundary_root, boundary_update
            )
            print(
                "[pure_rl] R274_SUBMISSION_FENCE_WAIT "
                f"boundary={boundary_update} request={request_path} "
                f"upload={upload_path}",
                flush=True,
            )
            while not upload_path.is_file():
                time.sleep(float(args.r274_submission_boundary_poll_seconds))
            try:
                upload = json.loads(upload_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"r274 submission upload receipt is unreadable: {upload_path}"
                ) from exc
            validated = _validate_r274_submission_upload(
                upload=upload,
                request=request,
            )
            result = {
                "request": _r241_file_identity(request_path),
                "upload": _r241_file_identity(upload_path),
                "remote_submission_id": int(validated["remote_submission_id"]),
            }
            print(
                "[pure_rl] R274_SUBMISSION_FENCE_PASSED "
                f"boundary={boundary_update} "
                f"remote_submission_id={result['remote_submission_id']}",
                flush=True,
            )
            return result

        def _prepare_r241_precollection_refresh(
            it: int, parent: Any
        ) -> tuple[Any, dict[str, Any]]:
            """Commit/recover boundary 5 before any iter-5 game is dispatched."""

            if not _r241_precollection_refresh_due(
                it, r241_peak_r195_training_contract, fixed_cycle_updates
            ):
                raise RuntimeError(
                    "fixed-cycle precollection refresh requested outside a "
                    "nonterminal five-update boundary"
                )
            state_key = (
                "r241_precollection_expert_refresh"
                if int(it) == R241_FIXED_CYCLE_REHEARSAL_EVERY
                else f"r274_precollection_expert_refresh_{int(it):05d}"
            )
            boundary = dict(loop_state.get(state_key) or {})
            receipt_path = rehearsal_paths(run_dir, it)[1]
            leaves_stopped = False
            if boundary:
                receipt_identity = _r241_file_identity(receipt_path)
                stored_receipt = dict(boundary.get("receipt") or {})
                refreshed = _verified_checkpoint_identity(
                    boundary.get("refreshed") or {}
                )
                sealed_parent = _verified_checkpoint_identity(
                    boundary.get("parent") or {}
                )
                if (
                    boundary.get("schema")
                    != "poke_bot.r241_precollection_expert_refresh/v1"
                    or int(boundary.get("before_iteration", -1)) != int(it)
                    or int(boundary.get("epochs", -1))
                    != R241_FIXED_CYCLE_REHEARSAL_EPOCHS
                    or stored_receipt != receipt_identity
                    or str((loop_state.get("learner") or {}).get("digest") or "")
                    != refreshed.digest
                    or parent.digest != refreshed.digest
                ):
                    raise RuntimeError(
                        "r241 precollection expert-refresh ledger is inconsistent"
                    )
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "r241 precollection expert-refresh receipt is unreadable"
                    ) from exc
                if (
                    str(receipt.get("parent_digest") or "")
                    != sealed_parent.digest
                    or str(receipt.get("checkpoint_digest") or "")
                    != refreshed.digest
                ):
                    raise RuntimeError(
                        "r241 precollection expert-refresh receipt changed"
                    )
                record = {
                    **receipt,
                    "checkpoint_identity": refreshed.as_dict(),
                    "reused": True,
                }
            else:
                if leaf.remote_channel is not None:
                    print(
                        "[pure_rl] suspend leaf farm before r241 precollection "
                        f"refresh boundary={it} servers={len(leaf.procs)}",
                        flush=True,
                    )
                    leaf.stop()
                    leaves_stopped = True
                    release_process_heap()
                refreshed, record = _prepare_expert_rehearsal(it, parent)
                if str(record.get("parent_digest") or "") != parent.digest:
                    raise RuntimeError(
                        "r241 boundary-5 refresh is not a direct child of commit 4"
                    )
                receipt_identity = _r241_file_identity(receipt_path)
                from poke_bot import checkpoint as checkpoint_mod

                refreshed_payload = checkpoint_mod.load_checkpoint(
                    refreshed.path, map_location="cpu"
                )
                refreshed_fit = dict(
                    (refreshed_payload.get("extra") or {}).get(
                        "dormant_matchup_adapter_fit"
                    )
                    or {}
                )
                next_boundary = {
                    "schema": "poke_bot.r241_precollection_expert_refresh/v1",
                    "before_iteration": int(it),
                    "rl_updates_completed": int(it),
                    "epochs": R241_FIXED_CYCLE_REHEARSAL_EPOCHS,
                    "parent": parent.as_dict(),
                    "refreshed": refreshed.as_dict(),
                    "receipt": receipt_identity,
                    "next_collection_started": False,
                    "runtime_publish": None,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                refreshed_state = json.loads(json.dumps(loop_state))
                refreshed_state["learner"] = refreshed.as_dict()
                refreshed_state[state_key] = next_boundary
                if refreshed_fit:
                    refreshed_state["dormant_matchup_adapter_fit"] = {
                        **refreshed_fit,
                        "checkpoint_digest": refreshed.digest,
                        "checkpoint_path": str(refreshed.path),
                    }
                _atomic_json(run_dir / "loop_state.json", refreshed_state)
                loop_state.clear()
                loop_state.update(refreshed_state)
                boundary = next_boundary

            _assert_r241_precollection_refresh_binding(
                iteration=it,
                collection_behavior_digest=refreshed.digest,
                rehearsal_record=record,
                r241_peak_r195_training_contract=(
                    r241_peak_r195_training_contract
                ),
            )
            submission_upload = _wait_for_r274_submission(
                boundary_update=it,
                checkpoint_identity=refreshed,
                rehearsal_receipt_path=receipt_path,
            )
            if leaves_stopped:
                _rebuild_leaves_if_needed(refreshed.path, refreshed.digest)
            publish = _hard_gate_publish_weights(
                leaf=leaf,
                remote_farm=remote_farm,
                ckpt=refreshed.path,
                digest=refreshed.digest,
                version=int(it) * 10 + 5,
                required_endpoints=endpoints,
                reload_local=False,
            )
            published_state = json.loads(json.dumps(loop_state))
            published_boundary = dict(published_state.get(state_key) or boundary)
            published_boundary["runtime_publish"] = publish
            published_boundary["next_collection_started"] = False
            if submission_upload is not None:
                published_boundary["submission_upload"] = submission_upload
            published_state[state_key] = published_boundary
            _atomic_json(run_dir / "loop_state.json", published_state)
            loop_state.clear()
            loop_state.update(published_state)
            print(
                "[pure_rl] R241_PRECOLLECTION_EXPERT_SOFT_REFRESH "
                f"before_iter={it} epochs={R241_FIXED_CYCLE_REHEARSAL_EPOCHS} "
                f"checkpoint={refreshed.digest[:19]}… next_collection_started=false",
                flush=True,
            )
            return refreshed, record

        def _bind_r241_precollection_collection(
            it: int,
            collect_bundle: Mapping[str, Any],
            rehearsal_record: Mapping[str, Any],
        ) -> None:
            """Seal the completed iter-5 collection against its refreshed digest."""

            if not _r241_precollection_refresh_due(
                it, r241_peak_r195_training_contract, fixed_cycle_updates
            ):
                return
            behavior = _collection_behavior_identity(dict(collect_bundle))
            _assert_r241_precollection_refresh_binding(
                iteration=it,
                collection_behavior_digest=behavior.digest,
                rehearsal_record=rehearsal_record,
                r241_peak_r195_training_contract=(
                    r241_peak_r195_training_contract
                ),
            )
            receipt_path = Path(str(collect_bundle.get("receipt") or ""))
            receipt_identity = _r241_file_identity(receipt_path)
            state_key = (
                "r241_precollection_expert_refresh"
                if int(it) == R241_FIXED_CYCLE_REHEARSAL_EVERY
                else f"r274_precollection_expert_refresh_{int(it):05d}"
            )
            boundary_state = json.loads(json.dumps(loop_state))
            boundary = dict(boundary_state.get(state_key) or {})
            if (
                boundary.get("schema")
                != "poke_bot.r241_precollection_expert_refresh/v1"
                or int(boundary.get("before_iteration", -1)) != int(it)
                or str((boundary.get("refreshed") or {}).get("digest") or "")
                != behavior.digest
            ):
                raise RuntimeError(
                    "r241 iter-5 collection lost its precollection refresh ledger"
                )
            boundary.update(
                {
                    "next_collection_started": True,
                    "collection_completed": True,
                    "collection_checkpoint_digest": behavior.digest,
                    "collection_receipt": receipt_identity,
                }
            )
            boundary_state[state_key] = boundary
            _atomic_json(run_dir / "loop_state.json", boundary_state)
            loop_state.clear()
            loop_state.update(boundary_state)

        def _complete_terminal_expert_rehearsal() -> Optional[dict[str, Any]]:
            """Run/recover the due final soft refresh without another collect."""

            if not bool(args.terminal_expert_rehearsal):
                return None
            boundary_iteration = int(loop_state.get("next_iteration") or 0)
            if boundary_iteration != int(args.iterations):
                raise RuntimeError(
                    "terminal expert rehearsal requires the exact completed "
                    "iteration count"
                )
            if not rehearsal_due(
                boundary_iteration, int(args.expert_rehearsal_every)
            ):
                raise RuntimeError("terminal expert rehearsal is not due")
            parent = _verified_checkpoint_identity(loop_state["learner"])
            if (
                r241_peak_r195_training_contract
                and leaf.remote_channel is not None
            ):
                print(
                    "[pure_rl] suspend leaf farm before r241 terminal expert "
                    f"refresh servers={len(leaf.procs)}",
                    flush=True,
                )
                leaf.stop()
                release_process_heap()
            rehearsed, rehearsal_record = _prepare_expert_rehearsal(
                boundary_iteration, parent
            )
            if r241_peak_r195_training_contract:
                validate_r241_peak_r195_live_fusion_record(
                    dict(rehearsal_record.get("peak_r195_live_fusion") or {}),
                    contract=r241_peak_r195_training_contract,
                    phase="expert_refresh",
                    boundary_iteration=boundary_iteration,
                )
            boundary_path = run_dir / "terminal_expert_refresh.json"
            if boundary_path.is_file():
                boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
                if (
                    boundary.get("schema")
                    != "poke_bot.terminal_expert_soft_refresh/v1"
                    or int(boundary.get("before_iteration", -1))
                    != boundary_iteration
                    or int(boundary.get("epochs_completed", -1))
                    != int(args.expert_rehearsal_epochs)
                    or dict(boundary.get("parent") or {}).get("digest")
                    != parent.digest
                    or dict(boundary.get("refreshed") or {}).get("digest")
                    != rehearsed.digest
                    or (
                        bool(r241_peak_r195_training_contract)
                        and dict(boundary.get("expert_rehearsal") or {})
                        != dict(rehearsal_record)
                    )
                    or boundary.get("next_collection_started") is not False
                ):
                    raise RuntimeError(
                        "terminal expert refresh receipt is inconsistent"
                    )
            else:
                boundary = {
                    "schema": "poke_bot.terminal_expert_soft_refresh/v1",
                    "specialist_id": str(args.specialist_archetype or ""),
                    "before_iteration": boundary_iteration,
                    "rl_updates_completed": boundary_iteration,
                    "epochs_completed": int(args.expert_rehearsal_epochs),
                    "parent": parent.as_dict(),
                    "refreshed": rehearsed.as_dict(),
                    "expert_rehearsal": rehearsal_record,
                    "next_collection_started": False,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_json_exclusive(boundary_path, boundary)
            terminal_state = json.loads(json.dumps(loop_state))
            terminal_state["terminal_expert_refresh"] = {
                "path": str(boundary_path),
                "parent": parent.as_dict(),
                "refreshed": rehearsed.as_dict(),
                "epochs": int(args.expert_rehearsal_epochs),
            }
            submission_upload = _wait_for_r274_submission(
                boundary_update=boundary_iteration,
                checkpoint_identity=rehearsed,
                rehearsal_receipt_path=rehearsal_paths(
                    run_dir, boundary_iteration
                )[1],
            )
            if submission_upload is not None:
                terminal_state["terminal_submission_upload"] = submission_upload
            _atomic_json(run_dir / "loop_state.json", terminal_state)
            loop_state.clear()
            loop_state.update(terminal_state)
            print(
                "[pure_rl] TERMINAL_EXPERT_SOFT_REFRESH "
                f"before_iter={boundary_iteration} "
                f"epochs={int(args.expert_rehearsal_epochs)} "
                f"checkpoint={rehearsed.digest[:19]}…",
                flush=True,
            )
            return boundary

        def _complete_fixed_cycle_boundary() -> Optional[dict[str, Any]]:
            """Close r241 only after the exact tenth immutable RL commit."""

            if fixed_cycle_updates:
                _assert_fixed_cycle_completion(
                    run_dir,
                    loop_state,
                    updates=fixed_cycle_updates,
                    r241_peak_r195_training_contract=(
                        r241_peak_r195_training_contract
                    ),
                )
            return _complete_terminal_expert_rehearsal()

        # Start exactly at the ledger boundary. Collection is synchronous and
        # never overlaps a train/promotion/reload boundary.
        if start_iteration >= int(args.iterations):
            _complete_fixed_cycle_boundary()
            print(
                f"[pure_rl] ledger already complete next={start_iteration} "
                f"iterations={args.iterations}",
                flush=True,
            )
            return 0
        precollect_rehearsals: dict[int, dict[str, Any]] = {}
        if int(start_iteration) == 0 and bool(args.expert_rehearsal_before_first):
            learner_identity, first_record = _prepare_expert_rehearsal(
                0, learner_identity
            )
            precollect_rehearsals[0] = first_record
        if _r241_precollection_refresh_due(
            start_iteration,
            r241_peak_r195_training_contract,
            fixed_cycle_updates,
        ):
            learner_identity, boundary_record = (
                _prepare_r241_precollection_refresh(
                    start_iteration, learner_identity
                )
            )
            precollect_rehearsals[int(start_iteration)] = boundary_record
        _maybe_apply_live_pool(
            learner_identity.path,
            learner_identity.digest,
            allow_leaf_rebuild=False,
        )
        pending_collect = _completed_collection_bundle(
            run_dir,
            loop_state,
            pending_collection_contract,
        )
        if pending_collect is None:
            pending_collect = _kick_collect(
                start_iteration,
                learner_identity.path,
                learner_identity.digest,
            )
        if _r241_precollection_refresh_due(
            start_iteration,
            r241_peak_r195_training_contract,
            fixed_cycle_updates,
        ):
            _bind_r241_precollection_collection(
                start_iteration,
                pending_collect,
                precollect_rehearsals[start_iteration],
            )

        for it in range(start_iteration, int(args.iterations)):
            t0 = time.time()
            assert pending_collect is not None
            collect_bundle = pending_collect
            shard_path: Path = collect_bundle["shard"]
            writer: CompactShardWriter = collect_bundle["writer"]
            behavior_before = _collection_behavior_identity(collect_bundle)
            incumbent_before = identity
            incumbent_path = ckpt
            learner_before = learner_identity
            rehearsal_record: Optional[dict[str, Any]] = precollect_rehearsals.pop(
                int(it), None
            )
            _assert_r241_precollection_refresh_binding(
                iteration=it,
                collection_behavior_digest=behavior_before.digest,
                rehearsal_record=rehearsal_record,
                r241_peak_r195_training_contract=(
                    r241_peak_r195_training_contract
                ),
            )

            # SERIALIZE (no AZ-style overlap): collect → train → hard-gate →
            # then start next collect on the *new* digest. Prefetching collect
            # N+1 on old weights caused digest FAIL-CLOSED storms and wasted
            # farm time on a stale build.
            _maybe_apply_live_pool(learner_before.path, learner_before.digest)

            # Collection leaves are large, replicated model processes. No
            # inference work is required during either expert rehearsal or
            # replay-window training, so release the entire farm before *any*
            # corpus packing. The previous ordering stopped leaves only after
            # rehearsal and unnecessarily overlapped ~20 GiB of inference
            # replicas with the large expert CPU pack.
            leaves_suspended_for_train = bool(leaf.remote_channel is not None)
            if leaves_suspended_for_train:
                print(
                    f"[pure_rl] suspend leaf farm before rehearsal/replay "
                    f"iter={it} servers={len(leaf.procs)}",
                    flush=True,
                )
                leaf.stop()
                collected_objects, heap_trimmed = release_process_heap()
                print(
                    f"[pure_rl] leaf memory released before corpus packing "
                    f"iter={it} gc={collected_objects} "
                    f"malloc_trim={int(heap_trimmed)}",
                    flush=True,
                )

            tactical_overlay_record: Optional[dict[str, Any]] = None
            tactical_overlay_dir: Optional[Path] = None
            if r260_own_deck_training_inputs and not disable_rl_tactical_cotrain:
                from poke_bot.tactical_sequence_materialization import (
                    materialize_tactical_shard_overlay,
                    validate_tactical_target_overlay,
                )

                tactical_overlay_dir = run_dir / "tactical_sequence_overlays"
                tactical_overlay_path = (
                    tactical_overlay_dir / f"iter_{it:05d}.json"
                )
                if tactical_overlay_path.is_file():
                    tactical_overlay_record = validate_tactical_target_overlay(
                        tactical_overlay_path,
                        shard_path=shard_path,
                        checkpoint_path=behavior_before.path,
                        checkpoint_digest=behavior_before.digest,
                        minimum_roots=1024,
                    )
                else:
                    tactical_overlay_record = materialize_tactical_shard_overlay(
                        shard_path,
                        checkpoint_path=behavior_before.path,
                        checkpoint_digest=behavior_before.digest,
                        output_path=tactical_overlay_path,
                        minimum_roots=1024,
                        device=train_dev,
                    )
                    tactical_overlay_record = validate_tactical_target_overlay(
                        tactical_overlay_path,
                        shard_path=shard_path,
                        checkpoint_path=behavior_before.path,
                        checkpoint_digest=behavior_before.digest,
                        minimum_roots=1024,
                    )
                _write_json_exclusive_or_verify(
                    run_dir
                    / "tactical_sequence_overlay_receipts"
                    / f"iter_{it:05d}.json",
                    {
                        "schema": "poke_bot.r274_tactical_sequence_overlay_receipt/v1",
                        "iteration": int(it),
                        "mode": "shadow_only",
                        "planner_dispatch_authority": False,
                        "source": "ordinary_direct_policy_training_games_only",
                        "overlay": tactical_overlay_record,
                        "collection_receipt": str(collect_bundle["receipt"]),
                    },
                )
                print(
                    "[pure_rl] tactical shadow overlay committed "
                    f"iter={it} roots={tactical_overlay_record['roots']} "
                    f"sha256={tactical_overlay_record['sha256']}",
                    flush=True,
                )

            # The ladder pass is durable and happens before loading the large
            # replay window into host RAM.  Recovery checks the receipt/orphan
            # checkpoint first, so a reboot can never repeat a completed pass
            # or silently switch to a newly advanced rolling-data pointer.
            forced_rehearsal = int(args.expert_rehearsal_force_before) == int(it)
            if rehearsal_record is None and (
                forced_rehearsal
                or rehearsal_due(it, int(args.expert_rehearsal_every))
            ):
                if forced_rehearsal:
                    print(
                        f"[pure_rl] forced expert rehearsal boundary iter={it}",
                        flush=True,
                    )
                learner_before, rehearsal_record = _prepare_expert_rehearsal(
                    it, learner_before
                )
                learner_identity = learner_before

            out_ckpt = run_dir / "checkpoints" / f"iter_{it:05d}.pt"
            recovered_candidate_result: Optional[dict[str, Any]] = None
            if out_ckpt.exists():
                recovered_candidate_result = _verified_orphan_candidate_result(
                    out_ckpt,
                    iteration=it,
                    parent_digest=learner_before.digest,
                    behavior_digest=behavior_before.digest,
                    design_fingerprint=_orphan_recovery_design_fingerprints(
                        loop_state,
                        collect_bundle.get("design_fingerprint_at_collection"),
                        design_fingerprint,
                    ),
                    shard_path=shard_path,
                    r241_peak_r195_training_contract=(
                        r241_peak_r195_training_contract
                    ),
                    expected_awr_provider=(
                        (design_contract.get("learner") or {}).get(
                            "prize_plan_h3_actor_provider"
                        )
                    ),
                )
                print(
                    f"[pure_rl] RECOVER_TRAINED_CANDIDATE iter={it} "
                    f"digest={recovered_candidate_result['candidate_digest'][:19]}… "
                    "skip_retrain=1 resume=promotion/heldout",
                    flush=True,
                )

            dataset = _dataset_from_replay_window(
                run_dir,
                it,
                initial_replay_shards=initial_replay_shards,
                tactical_overlay_dir=tactical_overlay_dir,
            )
            family_replay_weighting: Optional[dict[str, Any]] = None
            if family_training_contract is not None:
                family_sampling_design_fingerprint = (
                    _replay_sampling_design_fingerprint(
                        current_design_fingerprint=design_fingerprint,
                        collection_receipt=collect_bundle,
                    )
                )
                dataset, family_replay_weighting = (
                    _macro_resample_family_sequences(
                        dataset,
                        manifest=family_training_contract["manifest"],
                        checksum_seed=(
                            f"{family_sampling_design_fingerprint}|{int(it)}|"
                            f"{family_training_contract['manifest_sha256']}|"
                            f"{family_training_contract['loss_vector_sha256']}"
                        ),
                    )
                )
                print(
                    "[pure_rl] ARCHETYPE_FAMILY_REPLAY_WEIGHTED "
                    f"iter={it} sequences={len(dataset.sequences)} "
                    f"variants={len(family_replay_weighting['selected_variant_counts'])} "
                    f"legacy_package={family_replay_weighting['legacy_package_sequences']} "
                    f"duplicate_draws={family_replay_weighting['duplicate_draws']}",
                    flush=True,
                )
            training_seat_split_receipt: Optional[dict[str, Any]] = None
            if bool(args.require_exact_training_seat_split):
                training_seat_split_receipt = _commit_training_seat_split_receipt(
                    run_dir=run_dir,
                    iteration=it,
                    design_fingerprint=design_fingerprint,
                    collection_receipt=collect_bundle,
                    sequences=dataset.sequences,
                )
            adapter_ticketing: dict[str, Any] = {}
            if int(args.dormant_matchup_adapter_epochs) > 0:
                if active_gate is None:
                    raise RuntimeError(
                        "dormant adapter training lost its active gate contract"
                    )
                adapter_ticketing = _ticket_dormant_matchup_adapter_sequences(
                    dataset,
                    active_gate=active_gate,
                    specialist_archetype=str(args.specialist_archetype),
                    registered_specialist_ids=_registered_matchup_target_ids(
                        args.initial_learner_checkpoint
                    ),
                )
                print(
                    "[pure_rl] dormant adapter tickets "
                    f"iter={it} sequences={adapter_ticketing['ticketed_sequences']} "
                    f"routes={json.dumps(adapter_ticketing['route_sequences'], sort_keys=True)} "
                    "runtime=off",
                    flush=True,
                )
            n_train_sequences = len(dataset.sequences)
            h3_advantage_cache = None
            h3_advantage_provider_receipt = None
            live_h3_mode = bool(
                prize_plan_h3_actor_provider is not None
                and prize_plan_h3_actor_provider.get("mode")
                == "receipt_bound_live_h3_additive_at_clean_optimizer_boundary"
            )
            h3_inputs = (
                args.prize_plan_h3_cache_receipt,
                str(args.prize_plan_h3_cache_receipt_sha256 or ""),
                args.prize_plan_h3_activation_receipt,
                str(args.prize_plan_h3_activation_receipt_sha256 or ""),
            )
            if any(h3_inputs) and not all(h3_inputs):
                raise RuntimeError(
                    "Prize-plan H3 cache/activation paths and SHA-256 values "
                    "must be supplied as one complete boundary binding"
                )
            if live_h3_mode and any(h3_inputs):
                raise RuntimeError("live and static Prize-plan H3 providers cannot mix")
            if live_h3_mode:
                from poke_bot.prize_plan_live_cache import materialize_live_h3_cache

                live_root = run_dir / "prize_plan_h3_live" / f"iter_{it:05d}"
                if live_root.exists():
                    cache_receipt_path = live_root / "cache-receipt.json"
                    activation_receipt_path = live_root / "activation-receipt.json"
                    cache_receipt_sha256 = _sha256_file(cache_receipt_path)
                    activation_receipt_sha256 = _sha256_file(
                        activation_receipt_path
                    )
                    live_summary = {
                        "recovered_create_only_cache": True,
                        "cache_receipt_sha256": cache_receipt_sha256,
                        "activation_receipt_sha256": activation_receipt_sha256,
                    }
                else:
                    (
                        cache_receipt_path,
                        cache_receipt_sha256,
                        activation_receipt_path,
                        activation_receipt_sha256,
                        live_summary,
                    ) = materialize_live_h3_cache(
                        sequences=dataset.sequences,
                        replay_shards=_replay_window_paths(
                            run_dir,
                            it,
                            initial_replay_shards=initial_replay_shards,
                        ),
                        learner_checkpoint_sha256=learner_before.digest,
                        sidecar_checkpoint=Path(args.prize_plan_h3_live_sidecar),
                        sidecar_checkpoint_sha256=str(
                            args.prize_plan_h3_live_sidecar_sha256
                        ),
                        scale_support=Path(
                            args.prize_plan_h3_live_scale_support
                        ),
                        scale_support_sha256=str(
                            args.prize_plan_h3_live_scale_support_sha256
                        ),
                        c3_support=Path(args.prize_plan_h3_live_c3_support),
                        c3_support_sha256=str(
                            args.prize_plan_h3_live_c3_support_sha256
                        ),
                        contract=Path(args.prize_plan_h3_live_contract),
                        contract_sha256=str(
                            args.prize_plan_h3_live_contract_sha256
                        ),
                        output_dir=live_root,
                        device=train_dev,
                    )
                from poke_bot.prize_plan_actor_boundary import load_h3_actor_provider

                h3_advantage_cache, h3_advantage_provider_receipt = (
                    load_h3_actor_provider(
                        sequences=dataset.sequences,
                        policy_checkpoint_sha256=learner_before.digest,
                        cache_receipt_path=cache_receipt_path,
                        cache_receipt_sha256=cache_receipt_sha256,
                        activation_receipt_path=activation_receipt_path,
                        activation_receipt_sha256=activation_receipt_sha256,
                    )
                )
                h3_advantage_provider_receipt["provider_binding"][
                    "live_source_design"
                ] = dict(prize_plan_h3_actor_provider)
                print(
                    "[pure_rl] PRIZE_PLAN_H3_LIVE_BOUNDARY_PROVIDER "
                    f"iter={it} rows={len(h3_advantage_cache)} "
                    f"cache={cache_receipt_sha256} "
                    f"summary={json.dumps(live_summary, sort_keys=True)}",
                    flush=True,
                )
            if all(h3_inputs):
                from poke_bot.prize_plan_actor_boundary import load_h3_actor_provider

                h3_advantage_cache, h3_advantage_provider_receipt = (
                    load_h3_actor_provider(
                        sequences=dataset.sequences,
                        policy_checkpoint_sha256=learner_before.digest,
                        cache_receipt_path=Path(args.prize_plan_h3_cache_receipt),
                        cache_receipt_sha256=str(
                            args.prize_plan_h3_cache_receipt_sha256
                        ),
                        activation_receipt_path=Path(
                            args.prize_plan_h3_activation_receipt
                        ),
                        activation_receipt_sha256=str(
                            args.prize_plan_h3_activation_receipt_sha256
                        ),
                    )
                )
                if bool(args.train_device_resident):
                    raise RuntimeError(
                        "Prize-plan H3 activation requires the receipt-auditable "
                        "host replay path; device-resident training is unsupported"
                    )
                print(
                    "[pure_rl] PRIZE_PLAN_H3_BOUNDARY_PROVIDER "
                    f"iter={it} rows={len(h3_advantage_cache)} "
                    f"cache={args.prize_plan_h3_cache_receipt_sha256}",
                    flush=True,
                )
            train_metrics = {
                "mean_advantage": 0.0,
                "raw_advantage_mean": 0.0,
                "raw_advantage_std": 0.0,
                "raw_advantage_mean_abs": 0.0,
                "normalized_advantage_mean": 0.0,
                "normalized_advantage_std": 0.0,
                "awr_weight_mean": 0.0,
                "awr_weight_p50": 0.0,
                "awr_weight_p95": 0.0,
                "awr_weight_clip_frac": 0.0,
                "awr_effective_sample_size": 0.0,
                "awr_effective_sample_fraction": 0.0,
                "policy_selected_nll": 0.0,
                "target_value_mean": 0.0,
                "policy_acc": 0.0,
                "policy_prev_agreement": 0.0,
                "n_sequences": n_train_sequences,
                "collect_temperature": _collect_temperature(args, it),
            }
            if not n_train_sequences:
                del dataset
                release_process_heap()
                raise RuntimeError(
                    f"collect iter={it} passed retention checks but produced no "
                    "trainable sequences"
                )
            if (
                recovered_candidate_result is not None
                and int(recovered_candidate_result["n_train_sequences"])
                != int(n_train_sequences)
            ):
                raise RuntimeError(
                    "orphan candidate train/validation sequence count disagrees "
                    "with the receipt-backed replay window"
                )
            print(
                f"[pure_rl] train begin iter={it} seqs={n_train_sequences} "
                f"behavior={behavior_before.digest[:19]}… "
                f"learner_parent={learner_before.digest[:19]}… "
                f"(promotion precedes any weight publish)",
                flush=True,
            )
            decision_cap, batch_schedule_phase = _scheduled_train_decision_cap(
                it,
                steady_cap=int(args.train_max_decisions_per_batch),
                warmup_cap=int(args.train_warmup_max_decisions_per_batch),
                warmup_iterations=int(args.train_warmup_iterations),
            )
            train_cfg = TrainConfig.pure_rl_defaults(
                epochs=max(1, args.train_epochs),
                seed=args.seed + it,
                amp=train_dev.type == "cuda",
                games_per_batch=int(args.train_games_per_batch),
                max_decisions_per_batch=decision_cap,
                aux_loss_weight=float(args.archetype_aux_loss_weight),
                opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
                lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
                prize_race_loss_weight=float(args.prize_race_loss_weight),
                alakazam_guide_loss_weight=float(
                    args.alakazam_guide_loss_weight
                ),
                current_deck_guide_training_mode=str(
                    args.current_deck_guide_training_mode
                ),
                setup_board_outcome_loss_weight=float(
                    args.setup_board_outcome_loss_weight
                ),
                combo_state_loss_weight=float(args.combo_state_loss_weight),
                current_deck_guide_curriculum_spec=str(
                    args.current_deck_guide_curriculum_spec or ""
                ),
                current_deck_guide_head_role_map=str(
                    args.current_deck_guide_head_role_map or ""
                ),
                current_deck_guide_curriculum_validation_receipt=str(
                    args.current_deck_guide_curriculum_validation_receipt or ""
                ),
                dormant_matchup_adapter_epochs=int(
                    args.dormant_matchup_adapter_epochs
                ),
                dormant_matchup_adapter_lr=float(
                    args.dormant_matchup_adapter_lr
                ),
                dormant_matchup_adapter_max_decisions_per_batch=int(
                    args.dormant_matchup_adapter_max_decisions_per_batch
                ),
                dormant_matchup_adapter_activation_receipt=str(
                    args.dormant_matchup_adapter_activation_receipt or ""
                ),
                r241_peak_r195_training_contract=(
                    r241_peak_r195_training_contract
                ),
                optimizer_sparse_prefetch_batches=(
                    1 if bool(args.train_optimizer_sparse_prefetch) else 0
                ),
            )
            if family_training_contract is not None:
                train_cfg.archetype_residual_loss_weights = dict(
                    family_training_contract["residual_objectives"]
                )
            if r260_own_deck_training_inputs:
                train_cfg.visible_tutor_completion_loss_weight = 0.025
                train_cfg.terminal_conversion_loss_weight = 0.025
                train_cfg.tactical_sequence_outcome_loss_weight = (
                    0.0 if disable_rl_tactical_cotrain else 0.025
                )
                train_cfg.collect_own_deck_promotion_metrics = True
            if args.tactical_outcome_loss_weight_override is not None:
                from poke_bot import checkpoint as checkpoint_mod

                parent_payload = checkpoint_mod.load_checkpoint(
                    learner_before.path, map_location="cpu"
                )
                parent_expanded = dict(
                    (parent_payload.get("extra") or {}).get(
                        "expanded_head_training"
                    )
                    or {}
                )
                inherited_weights = dict(
                    parent_expanded.get("loss_weights") or {}
                )
                if not inherited_weights:
                    raise RuntimeError(
                        "expanded-head weight override lacks inherited weights"
                    )
                inherited_weights["tactical_outcome"] = float(
                    args.tactical_outcome_loss_weight_override
                )
                train_cfg.expanded_head_loss_weights = inherited_weights
                train_cfg.expanded_head_weight_migration_reason = (
                    _effective_boundary_design_migration_reason(args)
                )
            train_metrics["games_per_batch"] = int(train_cfg.games_per_batch)
            train_metrics["max_decisions_per_batch"] = int(
                train_cfg.max_decisions_per_batch
            )
            train_metrics["batch_schedule_phase"] = batch_schedule_phase
            train_metrics["batch_schedule_warmup_iterations"] = int(
                args.train_warmup_iterations
            )
            print(
                f"[pure_rl] epoch batch contract iter={it} "
                f"games_cap={train_cfg.games_per_batch} "
                f"decisions_cap={train_cfg.max_decisions_per_batch} "
                f"schedule={batch_schedule_phase}",
                flush=True,
            )
            try:
                result = recovered_candidate_result or rl_train_step(
                    dataset,
                    base_ckpt=learner_before.path,
                    out_run_name=f"{args.run_name}.iter{it:05d}",
                    archetype_id=(
                        "core"
                        if args.mode == "core"
                        else str(args.specialist_archetype)
                    ),
                    epochs=max(1, args.train_epochs),
                    device=train_dev,
                    cfg=train_cfg,
                    seed=args.seed + it,
                    output_path=out_ckpt,
                    parent_digest=learner_before.digest,
                    training_provenance={
                        "pure_rl": True,
                        "iteration": it,
                        "mode": args.mode,
                        "shard": str(shard_path),
                        "behavior_checkpoint": behavior_before.as_dict(),
                        "learner_parent": learner_before.as_dict(),
                        "expert_rehearsal": rehearsal_record,
                        "tactical_sequence_overlay": tactical_overlay_record,
                        "opponent_pool": [
                            _verified_checkpoint_identity(entry).as_dict()
                            for entry in opponent_pool
                        ],
                        "design_fingerprint": design_fingerprint,
                        **(
                            {
                                "prize_plan_h3_actor_provider": dict(
                                    h3_advantage_provider_receipt[
                                        "provider_binding"
                                    ]
                                )
                            }
                            if h3_advantage_provider_receipt is not None
                            else {}
                        ),
                        "dormant_matchup_adapter_ticketing": adapter_ticketing,
                        "archetype_family": (
                            {
                                "manifest_path": family_training_contract[
                                    "manifest_path"
                                ],
                                "manifest_sha256": family_training_contract[
                                    "manifest_sha256"
                                ],
                                "loss_contract_path": family_training_contract[
                                    "loss_contract_path"
                                ],
                                "loss_contract_sha256": family_training_contract[
                                    "loss_contract_sha256"
                                ],
                                "loss_vector_path": family_training_contract[
                                    "loss_vector_path"
                                ],
                                "loss_vector_sha256": family_training_contract[
                                    "loss_vector_sha256"
                                ],
                                "residual_objectives": dict(
                                    family_training_contract[
                                        "residual_objectives"
                                    ]
                                ),
                                "replay_weighting": family_replay_weighting,
                            }
                            if family_training_contract is not None
                            else None
                        ),
                        "append_only": True,
                    },
                    replace_existing=False,
                    device_resident=bool(args.train_device_resident),
                    device_resident_min_free_gib=float(
                        args.train_device_resident_min_free_gib
                    ),
                    awr_advantage_cache=h3_advantage_cache,
                    awr_advantage_provider_receipt=(
                        h3_advantage_provider_receipt
                    ),
                )
            finally:
                # A replay window can retain several GiB of trajectories.  It
                # must not overlap promotion, held-out evaluation, or the next
                # worker pool merely because this trainer is long-lived.
                # Device-resident training can likewise leave tens of GiB in
                # PyTorch's CUDA caching allocator after rl_train_step returns.
                # Those blocks are no longer live training tensors, but they
                # still prevent the rebuilt inference leaves from allocating
                # their small per-process working sets unless the long-lived
                # trainer explicitly returns the cache to CUDA.
                del dataset
                collected_objects, heap_trimmed = release_process_heap()
                cuda_reserved_before = 0
                cuda_reserved_after = 0
                if train_dev.type == "cuda":
                    torch.cuda.synchronize(train_dev)
                    cuda_reserved_before = int(
                        torch.cuda.memory_reserved(train_dev)
                    )
                    torch.cuda.empty_cache()
                    cuda_reserved_after = int(
                        torch.cuda.memory_reserved(train_dev)
                    )
                print(
                    f"[pure_rl] replay memory released iter={it} "
                    f"seqs={n_train_sequences} gc={collected_objects} "
                    f"malloc_trim={int(heap_trimmed)} "
                    f"cuda_reserved_before={cuda_reserved_before} "
                    f"cuda_reserved_after={cuda_reserved_after}",
                    flush=True,
                )
            if leaves_suspended_for_train:
                _rebuild_leaves_if_needed(
                    learner_before.path,
                    learner_before.digest,
                )
                print(
                    f"[pure_rl] leaf farm restored after train iter={it} "
                    f"servers={len(leaf.procs)}",
                    flush=True,
                )
            candidate_path = Path(
                result.get("candidate_path")
                or result.get("latest_path")
                or out_ckpt
            )
            candidate = CheckpointIdentity.from_path(candidate_path)
            from poke_bot import checkpoint as checkpoint_mod

            # The trust assertion intentionally returns a minimal deployment
            # projection, not the complete checkpoint payload.  Validate
            # trust first, then read identity fields from the same immutable
            # bytes; otherwise every correctly tagged specialist candidate is
            # misreported as archetype_id=None.
            checkpoint_mod.assert_trusted_policy_checkpoint(candidate.path)
            saved_candidate = checkpoint_mod.load_checkpoint(
                candidate.path, map_location="cpu"
            )
            expected_archetype = (
                "core"
                if args.mode == "core"
                else str(args.specialist_archetype)
            )
            if (
                str(saved_candidate.get("archetype_id") or "")
                != expected_archetype
            ):
                raise RuntimeError(
                    "candidate checkpoint archetype identity mismatch: "
                    f"expected={expected_archetype!r} "
                    f"actual={saved_candidate.get('archetype_id')!r} "
                    f"path={candidate.path}"
                )
            model_id = str(saved_candidate.get("model_id") or "")
            if str(args.run_name) not in model_id:
                raise RuntimeError(
                    "candidate checkpoint model identity does not name the "
                    f"active run: run={args.run_name!r} model_id={model_id!r}"
                )
            _verify_learner_lineage(
                result, candidate=candidate, parent=learner_before
            )
            candidate_adapter_fit = dict(
                result.get("dormant_matchup_adapter_fit") or {}
            )
            if candidate_adapter_fit:
                candidate_adapter_fit["checkpoint_digest"] = candidate.digest
                candidate_adapter_fit["checkpoint_path"] = str(candidate.path)
            m = result.get("metrics") or {}
            if hasattr(m, "__dict__"):
                m = (
                    asdict(m)
                    if hasattr(m, "__dataclass_fields__")
                    else dict(m.__dict__)
                )
            for key in tuple(train_metrics):
                if key in m and key not in ("n_sequences", "collect_temperature"):
                    train_metrics[key] = float(m.get(key) or 0.0)
            train_metrics["optimizer_state_restored"] = bool(
                result.get("optimizer_state_restored", False)
            )
            train_metrics["global_step"] = int(result.get("step", 0))
            train_metrics["epochs_ran"] = int(result.get("epochs_ran", 0))
            train_metrics["awr_baseline_mode"] = str(
                result.get("awr_baseline_mode") or "unknown"
            )

            adv_hist.append(float(train_metrics["raw_advantage_mean_abs"]))
            agr_hist.append(
                float(train_metrics.get("policy_prev_agreement") or 0.0)
            )
            abort = evaluate_aborts(
                mean_advantages=adv_hist,
                advantage_mean_abs=adv_hist,
                policy_prev_agreements=agr_hist,
                k=3,
            )

            promotion_rows: list[dict[str, Any]] = []
            if abort.abort:
                promotion_report = {
                    "passed": False,
                    "valid": False,
                    "skipped": True,
                    "reason": f"abort:{abort.reason}",
                    "candidate": candidate.as_dict(),
                    "incumbent": incumbent_before.as_dict(),
                }
            else:
                print(
                    f"[pure_rl] promotion begin iter={it} "
                    f"candidate={candidate.digest[:19]}… "
                    f"incumbent={incumbent_before.digest[:19]}…",
                    flush=True,
                )
                promotion_report, promotion_rows = _promotion_eval(
                    candidate=candidate,
                    incumbent=incumbent_before,
                    decks=measurement_decks,
                    n_games=args.promotion_games,
                    n_workers=args.promotion_workers,
                    threshold=args.promotion_threshold,
                    confidence=args.promotion_confidence,
                    bootstrap_resamples=args.promotion_bootstrap_resamples,
                    seed=args.seed + 7_000_000 + it * 10_000,
                    game_timeout_s=args.game_timeout_s,
                    model_generation=it + 1,
                )
                promotion_report["skipped"] = False

            promoted = bool(promotion_report.get("passed")) and not abort.abort
            candidate_safety_ok, candidate_safety_reason = carry_learner_candidate(
                promotion_report,
                abort=bool(abort.abort),
                minimum_head_to_head_wr=float(args.continuous_learner_min_wr),
            )
            prior_heldout_champion_evidence = dict(heldout_champion_evidence)
            weight_gate_proof: Optional[dict[str, Any]] = None
            heldout_publish_proof: Optional[dict[str, Any]] = None
            heldout_rollback_proof: Optional[dict[str, Any]] = None
            publish_version_base = (it + 1) * 10
            if promoted:
                ckpt = candidate_path
                identity = candidate
                opponent_pool.append(candidate)
                pool_n = max(
                    1, int(getattr(config.PURE_RL, "opponent_pool_size", 6))
                )
                del opponent_pool[:-pool_n]
                print(
                    f"[pure_rl] BETWEEN_ITER_HARD_GATE begin iter={it} "
                    f"digest={identity.digest[:19]}… version={publish_version_base + 1} "
                    f"(before next collect)",
                    flush=True,
                )
                weight_gate_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=ckpt,
                    digest=identity.digest,
                    version=publish_version_base + 1,
                    required_endpoints=endpoints,
                )
                promotion_report["deployment"] = weight_gate_proof
                print(
                    f"[pure_rl] PROMOTED iter={it} "
                    f"wr={promotion_report.get('wr')} "
                    f"lower={promotion_report.get('interval_lower')}",
                    flush=True,
                )
            else:
                ckpt = incumbent_path
                identity = incumbent_before
                print(
                    f"[pure_rl] REJECTED iter={it} candidate retained={candidate_path} "
                    f"incumbent={identity.digest[:19]}… "
                    f"reason={promotion_report.get('reason') or promotion_report.get('failures')}",
                    flush=True,
                )

            # Always evaluate the candidate, never the incumbent substituted
            # after rejection.  A rejected candidate is published only for this
            # drained greedy wave and the protected champion is restored in a
            # finally block before any next-iteration collection can begin.
            if not promoted:
                heldout_publish_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=(remote_farm if args.heldout_remotes else None),
                    ckpt=candidate.path,
                    digest=candidate.digest,
                    version=publish_version_base + 2,
                    required_endpoints=(endpoints if args.heldout_remotes else []),
                )
            research_control_result: Optional[dict[str, Any]] = None
            heldout_local_workers = max(
                1,
                min(
                    int(hw.sim_workers),
                    int(
                        os.environ.get(
                            "PURE_RL_HELDOUT_LOCAL_WORKERS",
                            str(hw.sim_workers),
                        )
                    ),
                ),
            )
            print(
                "[pure_rl] heldout local worker cap="
                f"{heldout_local_workers} collection_workers={hw.sim_workers} "
                "(bounded formal-eval memory envelope)",
                flush=True,
            )
            formal_holdout_deferred = bool(
                int(it) == 0
                and r284_boundary_receipt
                and candidate.digest
                == str(r284_boundary_receipt.get("candidate_checkpoint_sha256") or "")
            )
            terminal_eval_waived = bool(
                int(it) == 1
                and str(args.run_name)
                == "alakazam_new_list_direct_policy_r274"
                and os.environ.get("POKEBOT_R307_SKIP_TERMINAL_EVAL", "0")
                == "1"
            )
            r329_holdout_waiver: dict[str, Any] = {}
            r329_holdout_waiver_path_raw = os.environ.get(
                "POKEBOT_R329_ITER3_HOLDOUT_WAIVER_RECEIPT", ""
            ).strip()
            # The r329 receipt waives only iteration 3.  Keep the immutable
            # receipt configured for resume provenance, but do not rebind it
            # to later candidates when their normal formal holdout begins.
            if r329_holdout_waiver_path_raw and int(it) == 3:
                r329_holdout_waiver_path = Path(
                    r329_holdout_waiver_path_raw
                ).expanduser().resolve()
                r329_holdout_waiver = json.loads(
                    r329_holdout_waiver_path.read_text(encoding="utf-8")
                )
                r329_contract_path = Path(
                    os.environ["POKEBOT_R329_GOAL_CONTRACT"]
                ).expanduser().resolve()
                r329_contract_sha256 = _sha256_file(r329_contract_path)
                expected_waiver = {
                    "schema": "poke_bot.alakazam_rule_derivative_iter3_holdout_waiver/v1",
                    "owner_goal_revision": 27,
                    "root_owner_revision": 329,
                    "run_name": str(args.run_name),
                    "iteration": 3,
                    "candidate_checkpoint_sha256": str(candidate.digest),
                    "formal_holdout_games_skipped": int(args.heldout_games),
                    "formal_holdout_only": True,
                    "promotion_comparison_preserved": True,
                    "measured_holdout_pass_claim_allowed": False,
                    "future_holdouts_unchanged": True,
                    "goal_contract_sha256": r329_contract_sha256,
                }
                for key, expected_value in expected_waiver.items():
                    if r329_holdout_waiver.get(key) != expected_value:
                        raise RuntimeError(
                            "invalid r329 iteration-3 holdout waiver field "
                            f"{key}: expected={expected_value!r} "
                            f"actual={r329_holdout_waiver.get(key)!r}"
                        )
            r329_holdout_waived = bool(
                int(it) == 3 and bool(r329_holdout_waiver)
            )
            formal_holdout_deferred = bool(
                formal_holdout_deferred
                or terminal_eval_waived
                or r329_holdout_waived
            )
            if terminal_eval_waived:
                print(
                    "[pure_rl] TERMINAL_EVALUATION_OWNER_WAIVED "
                    f"iter={it} candidate={candidate.digest[:19]}… "
                    "formal_holdout=skipped research_control=skipped "
                    "measured_pass_claim=0 owner_revision=307",
                    flush=True,
                )
            if r329_holdout_waived:
                print(
                    "[pure_rl] FORMAL_HOLDOUT_OWNER_WAIVED "
                    f"iter={it} candidate={candidate.digest[:19]}… "
                    "formal_holdout=skipped measured_pass_claim=0 "
                    "owner_revision=329 scope=iteration_3_only",
                    flush=True,
                )
            if (
                formal_holdout_deferred
                and not terminal_eval_waived
                and not r329_holdout_waived
            ):
                print(
                    "[pure_rl] FORMAL_HOLDOUT_OWNER_DEFERRED "
                    f"iter={it} candidate={candidate.digest[:19]}… "
                    "first_formal_iteration=1 partial_broad_prefix=2142/4500 "
                    "gate_eligible=0",
                    flush=True,
                )
            try:
                heldout_rows, heldout_audit = _heldout_eval(
                    ckpt=Path(candidate.path),
                    digest=candidate.digest,
                    n_games=args.heldout_games,
                    decks=measurement_decks,
                    official_specs=heldout_specs,
                    seed=(
                        args.seed
                        + FORMAL_GATE_SEED_OFFSET
                        + it * ITERATION_SEED_STRIDE
                    ),
                    game_timeout_s=args.game_timeout_s,
                    n_workers=heldout_local_workers,
                    leaf_channel=leaf.remote_channel,
                    remote_farm=(remote_farm if args.heldout_remotes else None),
                    worker_play=worker_play,
                    worker_self_play=worker_self_play,
                    mode=args.mode,
                    allow_remote_play=bool(args.heldout_remotes),
                    iteration=int(it),
                    gate_wr=args.gate_wr,
                    opponent_ids=active_gate_ids,
                    stage_label=(
                        "heldout:strong_public_gate"
                        if active_gate is not None
                        else "heldout"
                    ),
                    owner_deferred=formal_holdout_deferred,
                )
                if (
                    int(args.research_control_games_per_iter) > 0
                    and not bool(args.disable_research_control_measurement)
                    and not formal_holdout_deferred
                ):
                    research_control_result = _research_control_measurement(
                        run_dir=run_dir,
                        iteration=int(it),
                        root_seed=int(args.seed),
                        n_games=int(args.research_control_games_per_iter),
                        training_games=int(args.games_per_iter),
                        formal_games=int(args.heldout_games),
                        ckpt=Path(candidate.path),
                        digest=candidate.digest,
                        decks=measurement_decks,
                        specs=research_control_specs,
                        registry=research_control_registry,
                        active_gate_digests=set(active_gate_content_digests),
                        game_timeout_s=int(args.game_timeout_s),
                        n_workers=heldout_local_workers,
                        leaf_channel=leaf.remote_channel,
                        remote_farm=(
                            remote_farm if args.heldout_remotes else None
                        ),
                        worker_play=worker_play,
                        worker_self_play=worker_self_play,
                        mode=args.mode,
                        allow_remote_play=bool(args.heldout_remotes),
                    )
            finally:
                if not promoted:
                    # Restore the exact receipt-proven behavior identity that
                    # was active before this temporary candidate evaluation.
                    # The rollout champion can intentionally lag the
                    # cumulative learner and may predate an append-only runtime
                    # feature (for example matchup adapters), so restoring the
                    # champion here can be both semantically stale and
                    # impossible under the fail-closed runtime contract.
                    heldout_rollback_proof = _hard_gate_publish_weights(
                        leaf=leaf,
                        remote_farm=(remote_farm if args.heldout_remotes else None),
                        ckpt=behavior_before.path,
                        digest=behavior_before.digest,
                        version=publish_version_base + 3,
                        required_endpoints=(
                            endpoints if args.heldout_remotes else []
                        ),
                    )
            gate = aggregate_heldout_wr(
                heldout_rows,
                target_wr=args.gate_wr,
                min_games=args.heldout_games,
                official_ids=active_gate_ids,
                min_games_per_opponent=(
                    int(args.heldout_games) // len(active_gate_ids)
                ),
                per_opponent_floor=args.heldout_per_opponent_floor,
            )
            evaluation_gate_contract = formal_gate_contract
            evaluation_active_gate = formal_gate
            if formal_gate_contract is not None and formal_gate is not None:
                primary_gate_id = str(formal_gate["id"])
                prior_gate_passed = _gate_passed_in_history(
                    loop_state,
                    gate_id=primary_gate_id,
                    through_iteration=int(it) - 1,
                )
                fallback_contract = materialize_fallback_gate_contract(
                    formal_gate_contract,
                    completed_iteration=int(it) - 1,
                    prior_gate_passed=prior_gate_passed,
                )
                if fallback_contract is not None:
                    evaluation_gate_contract = fallback_contract
                    evaluation_active_gate = dict(fallback_contract["next_gate"])
                    print(
                        "[pure_rl] ACTIVE_GATE_FALLBACK "
                        f"iter={it} prior={primary_gate_id} "
                        f"active={evaluation_active_gate['id']} "
                        "confidence_lower=0.50 other_criteria=unchanged",
                        flush=True,
                    )
            active_gate_result: Optional[dict[str, Any]] = None
            if evaluation_gate_contract is not None:
                active_gate_result = build_active_gate_result(
                    contract=evaluation_gate_contract,
                    checkpoint=candidate.path,
                    checkpoint_digest=candidate.digest,
                    iteration=int(it),
                    gate_rows=heldout_rows,
                    gate_audit=heldout_audit,
                    gate_seed=(
                        args.seed
                        + FORMAL_GATE_SEED_OFFSET
                        + it * ITERATION_SEED_STRIDE
                    ),
                )
                official_floor = dict(evaluation_active_gate or {}).get(
                    "pass_criteria", {}
                ).get("accepted_official_holdout_non_regression")
                if official_floor is not None:
                    research = dict(research_control_result or {})
                    research_audit = dict(research.get("audit") or {})
                    same_checkpoint = (
                        str(research.get("checkpoint_digest") or "")
                        == str(candidate.digest)
                    )
                    official_ok = bool(
                        same_checkpoint
                        and research_audit.get("passed") is True
                        and int(research.get("games") or 0) == 1000
                        and float(research.get("win_rate") or 0.0)
                        >= float(official_floor)
                    )
                    active_gate_result["checks"][
                        "accepted_official_holdout_non_regression"
                    ] = official_ok
                    active_gate_result["official_control_gate"] = {
                        "passed": official_ok,
                        "training_eligible": False,
                        "replay_eligible": False,
                        "checkpoint_digest_matches": same_checkpoint,
                        "games": int(research.get("games") or 0),
                        "win_rate": float(research.get("win_rate") or 0.0),
                        "minimum_win_rate": float(official_floor),
                        "audit_passed": research_audit.get("passed") is True,
                    }
                    active_gate_result["passed"] = all(
                        bool(value)
                        for value in active_gate_result["checks"].values()
                    )
                gate.win_rate = float(active_gate_result["skill_weighted_wr"])
                gate.confidence_lower = float(
                    active_gate_result["confidence_lower"]
                )
                gate.confidence_upper = float(
                    active_gate_result["confidence_upper"]
                )
                gate.passed = bool(active_gate_result["passed"])
                gate.reason = "ok" if gate.passed else "active_gate_criteria_failed"
                raw_heldout_gate = dict(active_gate_result)
            else:
                raw_heldout_gate = asdict(gate)
            if formal_holdout_deferred:
                assert active_gate_result is not None
                deferred_record = {
                    "passed": False,
                    "eligible": False,
                    "owner_deferred": True,
                    "reason": "owner_r284_moved_update_0_holdout_to_iteration_1",
                    "partial_broad_holdout_games": 2142,
                    "partial_broad_holdout_planned_games": 4500,
                    "partial_broad_holdout_gate_eligible": False,
                    "boundary_receipt": (
                        str(
                            Path(args.r284_iteration_boundary_receipt)
                            .expanduser()
                            .resolve()
                        )
                        if args.r284_iteration_boundary_receipt is not None
                        else None
                    ),
                }
                if terminal_eval_waived:
                    deferred_record.update(
                        {
                            "owner_revision": 307,
                            "owner_deferred": False,
                            "owner_waived": True,
                            "reason": "owner_r307_terminal_evaluation_waiver",
                            "formal_holdout_skipped": True,
                            "research_control_measurement_skipped": True,
                            "measured_pass_claim_allowed": False,
                            "evaluation_waiver_scope": "r274_iteration_1_only",
                        }
                    )
                elif r329_holdout_waived:
                    deferred_record.update(
                        {
                            "owner_revision": 329,
                            "owner_deferred": False,
                            "owner_waived": True,
                            "reason": "owner_r329_iteration_3_formal_holdout_waiver",
                            "formal_holdout_skipped": True,
                            "research_control_measurement_skipped": True,
                            "measured_pass_claim_allowed": False,
                            "evaluation_waiver_scope": "derivative_iteration_3_only",
                            "waiver_receipt": str(r329_holdout_waiver_path),
                        }
                    )
                active_gate_result.update(deferred_record)
                gate.passed = False
                gate.reason = (
                    "owner_terminal_evaluation_waiver"
                    if terminal_eval_waived
                    else (
                        "owner_iteration_3_formal_holdout_waiver"
                        if r329_holdout_waived
                        else "owner_deferred_to_iteration_1"
                    )
                )
                raw_heldout_gate = dict(active_gate_result)
            heldout_champion_updated = False
            candidate_heldout_evidence: Optional[dict[str, Any]] = None
            prior_active_evidence: dict[str, Any] = {}
            learner_exploration = {
                "eligible": False,
                "reason": "heldout_contract_audit_failed",
            }
            if bool(heldout_audit.get("passed")):
                candidate_heldout_evidence = {
                    "evidence_schema": 2,
                    "gate_id": (
                        str(evaluation_active_gate["id"])
                        if evaluation_active_gate is not None
                        else None
                    ),
                    "iteration": int(it),
                    "checkpoint_digest": candidate.digest,
                    "games": int(gate.games),
                    "win_rate": float(gate.win_rate),
                    "confidence_lower": float(gate.confidence_lower),
                    "confidence_upper": float(gate.confidence_upper),
                    "per_opponent": json.loads(json.dumps(gate.per_opponent)),
                    "audit": dict(heldout_audit),
                }
                comparable_gate_ids = {
                    str(value)
                    for value in (
                        (formal_gate or {}).get("id"),
                        (evaluation_active_gate or {}).get("id"),
                    )
                    if value
                }
                prior_active_evidence = (
                    heldout_champion_evidence
                    if evaluation_active_gate is None
                    or str(heldout_champion_evidence.get("gate_id") or "")
                    in comparable_gate_ids
                    else {}
                )
                if evaluation_active_gate is not None:
                    candidate_rank = active_gate_goal_rank(
                        candidate_heldout_evidence,
                        active_gate=evaluation_active_gate,
                    )
                    prior_rank = active_gate_goal_rank(
                        prior_active_evidence,
                        active_gate=evaluation_active_gate,
                    )
                else:
                    candidate_rank = heldout_goal_rank(
                        candidate_heldout_evidence,
                        official_ids=active_gate_ids,
                        per_opponent_floor=float(args.heldout_per_opponent_floor),
                    )
                    prior_rank = heldout_goal_rank(
                        prior_active_evidence,
                        official_ids=active_gate_ids,
                        per_opponent_floor=float(args.heldout_per_opponent_floor),
                    )
                if candidate_safety_ok and (
                    not prior_active_evidence or candidate_rank > prior_rank
                ):
                    heldout_champion_identity = candidate
                    heldout_champion_evidence = candidate_heldout_evidence
                    heldout_champion_updated = True
                    print(
                        f"[pure_rl] HELDOUT_CHAMPION iter={it} "
                        f"digest={candidate.digest[:19]}… "
                        f"wr={gate.win_rate:.3f} lower={gate.confidence_lower:.3f}",
                        flush=True,
                    )
                elif candidate_safety_ok and prior_active_evidence:
                    learner_exploration = heldout_exploration_decision(
                        candidate_heldout_evidence,
                        prior_active_evidence,
                        official_ids=active_gate_ids,
                        per_opponent_floor=float(
                            args.heldout_per_opponent_floor
                        ),
                    )
                    if bool(learner_exploration.get("eligible")):
                        print(
                            f"[pure_rl] CONTINUOUS_LEARNER iter={it} "
                            f"digest={candidate.digest[:19]}… "
                            "heldout_anchor_unchanged=1 "
                            f"progress_gain={float(learner_exploration['clipped_progress_gain']):.4f}",
                            flush=True,
                        )
            # Keep the exact heldout record protected while allowing short
            # cumulative branches. H2H safety alone is insufficient: repeated
            # material regression on the exact objective must return the next
            # rollout policy to the protected gate-aligned best.
            carry_candidate, learner_carry_reason = (
                _continuous_learner_carry_decision(
                    candidate_safety_ok=bool(candidate_safety_ok),
                    candidate_safety_reason=str(candidate_safety_reason),
                    heldout_audit_ok=bool(heldout_audit.get("passed")),
                    promoted=bool(promoted),
                )
            )
            if formal_holdout_deferred and candidate_safety_ok:
                carry_candidate = True
                learner_carry_reason = (
                    "owner_r307_terminal_evaluation_waiver_carry"
                    if terminal_eval_waived
                    else (
                        "owner_r329_iteration_3_holdout_waiver_carry"
                        if r329_holdout_waived
                        else "owner_r284_deferred_holdout_promotion_carry"
                    )
                )
            exact_gate_regression = {
                "enabled": False,
                "reason": "active_gate_or_anchor_unavailable",
                "streak": 0,
                "regression_margin": float(
                    args.continuous_learner_exact_regression_margin
                ),
                "patience": int(
                    args.continuous_learner_exact_regression_patience
                ),
            }
            if (
                carry_candidate
                and active_gate_result is not None
                and prior_active_evidence
                and not heldout_champion_updated
            ):
                exact_gate_regression = _exact_gate_regression_streak(
                    history=list(loop_state.get("history") or []),
                    current_gate_result=active_gate_result,
                    anchor_evidence=prior_active_evidence,
                    regression_margin=float(
                        args.continuous_learner_exact_regression_margin
                    ),
                )
                exact_gate_regression["patience"] = int(
                    args.continuous_learner_exact_regression_patience
                )
                if int(exact_gate_regression.get("streak", 0)) >= int(
                    args.continuous_learner_exact_regression_patience
                ):
                    carry_candidate = False
                    learner_carry_reason = (
                        "exact_gate_regression_patience_exhausted"
                    )
            if carry_candidate:
                learner_after = candidate
            elif learner_carry_reason == "exact_gate_regression_patience_exhausted":
                learner_after, exact_rollback_source = (
                    _exact_regression_rollback_identity(
                        loop_state,
                        exact_anchor=heldout_champion_identity,
                        behavior_before=behavior_before,
                    )
                )
                print(
                    f"[pure_rl] EXACT_GATE_ROLLBACK iter={it} "
                    f"candidate={candidate.digest[:19]}… "
                    f"anchor={heldout_champion_identity.digest[:19]}… "
                    f"rollout={learner_after.digest[:19]}… "
                    f"source={exact_rollback_source} "
                    f"streak={int(exact_gate_regression['streak'])} "
                    f"margin={float(exact_gate_regression['regression_margin']):.3f}",
                    flush=True,
                )
            else:
                # Invalid evidence or a head-to-head collapse rejects only the
                # current step and preserves the last safety-approved learner.
                learner_after = learner_before
            learner_identity = learner_after
            promotion_report["continuous_learner"] = {
                "carried_candidate": bool(carry_candidate),
                "candidate_head_to_head_safe": bool(candidate_safety_ok),
                "reason": learner_carry_reason,
                "selection_objective": (
                    "gate_aligned_best_plus_bounded_continuous_learner_v4"
                ),
                "exploration": learner_exploration,
                "exact_gate_regression": exact_gate_regression,
                "minimum_head_to_head_wr": float(args.continuous_learner_min_wr),
                "learner_before": learner_before.as_dict(),
                "learner_after": learner_after.as_dict(),
            }
            if not bool(heldout_audit.get("passed")):
                gate.passed = False
                gate.reason = "heldout_contract_audit_failed"
            if active_gate_result is not None:
                # A formal specialist gate and the short incumbent H2H answer
                # different questions.  Preserve the H2H result for learner
                # selection and diagnostics, but never let it veto the exact
                # official/premium gate or specialist handoff.
                if (
                    bool(active_gate_result.get("passed"))
                    and bool(heldout_audit.get("passed"))
                ):
                    gate.passed = True
                    gate.reason = str(
                        active_gate_result.get("reason")
                        or "formal_active_gate_passed"
                    )
                active_gate_result["pipeline_gate_passed"] = bool(gate.passed)
                active_gate_result["pipeline_gate_reason"] = str(gate.reason)
                active_gate_result["promotion_passed"] = bool(promoted)
                active_gate_result["promotion_blocks_specialist_transition"] = False
                active_gate_result["candidate_safety_passed"] = bool(
                    candidate_safety_ok
                )
            # The protected champion remains the promotion/rollback identity,
            # while the safety-approved continuous learner is the next behavior
            # policy. Publish it only after promotion and heldout work is drained
            # so each immutable collection shard uses one exact digest.
            collection_publish_proof: Optional[dict[str, Any]] = None
            if (
                (
                    not gate.passed
                    or bool(args.continue_after_gate)
                    or bool(fixed_cycle_updates)
                )
                and it + 1 < int(args.iterations)
                and not _r241_precollection_refresh_due(
                    it + 1,
                    r241_peak_r195_training_contract,
                    fixed_cycle_updates,
                )
            ):
                collection_publish_proof = _hard_gate_publish_weights(
                    leaf=leaf,
                    remote_farm=remote_farm,
                    ckpt=learner_after.path,
                    digest=learner_after.digest,
                    version=publish_version_base + 4,
                    required_endpoints=endpoints,
                )
                collection_publish_proof["role"] = "continuous_learner"
            eval_payload = {
                "iteration": it,
                "candidate": candidate.as_dict(),
                "incumbent_before": incumbent_before.as_dict(),
                "incumbent_after": identity.as_dict(),
                "learner_before": learner_before.as_dict(),
                "learner_after": learner_after.as_dict(),
                "expert_rehearsal": rehearsal_record,
                "promotion": promotion_report,
                "promotion_rows": promotion_rows,
                "raw_heldout_gate": raw_heldout_gate,
                "active_gate_result": active_gate_result,
                "research_control_result": research_control_result,
                "heldout_gate": asdict(gate),
                "heldout_audit": heldout_audit,
                "heldout_candidate": candidate.as_dict(),
                "heldout_champion": heldout_champion_identity.as_dict(),
                "heldout_champion_evidence": heldout_champion_evidence,
                "heldout_champion_updated": heldout_champion_updated,
                "heldout_publish": heldout_publish_proof,
                "heldout_rollback": heldout_rollback_proof,
                "next_collection_publish": collection_publish_proof,
                "heldout_rows": heldout_rows,
            }
            _write_json_exclusive(
                run_dir / "eval" / f"iter_{it:05d}.json", eval_payload
            )
            collect_stats = dict(collect_bundle["stats"])
            collect_elapsed = max(
                float(collect_stats.get("collect_elapsed_sec") or 0.0), 1e-6
            )
            completed_at = time.time()
            elapsed = max(completed_at - t0, 1e-6)
            iteration_started_at = float(
                collect_bundle.get("started_at") or (completed_at - collect_elapsed - elapsed)
            )
            iteration_wall_sec = max(completed_at - iteration_started_at, 1e-6)
            metrics = IterationMetrics(
                iteration=it,
                stage=stage.stage.value,
                games=writer.n_games,
                decisions=writer.n_decisions,
                games_per_sec=writer.n_games / collect_elapsed,
                decisions_per_sec=writer.n_decisions / collect_elapsed,
                games_per_hour=(writer.n_games / collect_elapsed) * 3600.0,
                mean_return=float(train_metrics["target_value_mean"]),
                mean_advantage=float(train_metrics["mean_advantage"]),
                raw_advantage_mean=float(train_metrics["raw_advantage_mean"]),
                raw_advantage_std=float(train_metrics["raw_advantage_std"]),
                raw_advantage_mean_abs=float(
                    train_metrics["raw_advantage_mean_abs"]
                ),
                normalized_advantage_mean=float(
                    train_metrics["normalized_advantage_mean"]
                ),
                normalized_advantage_std=float(
                    train_metrics["normalized_advantage_std"]
                ),
                awr_weight_mean=float(train_metrics["awr_weight_mean"]),
                awr_weight_p50=float(train_metrics["awr_weight_p50"]),
                awr_weight_p95=float(train_metrics["awr_weight_p95"]),
                awr_weight_clip_frac=float(train_metrics["awr_weight_clip_frac"]),
                awr_effective_sample_size=float(
                    train_metrics["awr_effective_sample_size"]
                ),
                awr_effective_sample_fraction=float(
                    train_metrics["awr_effective_sample_fraction"]
                ),
                policy_selected_nll=float(train_metrics["policy_selected_nll"]),
                policy_accuracy=float(train_metrics["policy_acc"]),
                policy_prev_agreement=float(agr_hist[-1]),
                self_distill_flag=abort.self_distill_flag,
                heldout_wr=gate.win_rate,
                heldout_games=gate.games,
                gate_passed=gate.passed,
                extra={
                    "abort": asdict(abort),
                    # elapsed_sec is retained for compatibility with existing
                    # reports. It historically measured only post-collection
                    # work, so the explicit wall metric is authoritative.
                    "elapsed_sec": elapsed,
                    "post_collect_elapsed_sec": elapsed,
                    "iteration_started_at": iteration_started_at,
                    "iteration_completed_at": completed_at,
                    "iteration_wall_sec": iteration_wall_sec,
                    "collect_stats": collect_stats,
                    "checkpoint": str(ckpt),
                    "candidate_checkpoint": candidate.as_dict(),
                    "incumbent_before": incumbent_before.as_dict(),
                    "learner_before": learner_before.as_dict(),
                    "learner_after": learner_after.as_dict(),
                    "expert_rehearsal": rehearsal_record,
                    "promotion": promotion_report,
                    "raw_heldout_gate": raw_heldout_gate,
                    "active_gate_result": active_gate_result,
                    "research_control_result": research_control_result,
                    "heldout_gate": asdict(gate),
                    "heldout_audit": heldout_audit,
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_evidence": heldout_champion_evidence,
                    "heldout_champion_updated": heldout_champion_updated,
                    "heldout_publish": heldout_publish_proof,
                    "heldout_rollback": heldout_rollback_proof,
                    "next_collection_publish": collection_publish_proof,
                    "train_metrics": train_metrics,
                    "remote_workers": (
                        remote_farm.total_workers if remote_farm else 0
                    ),
                    "remote_endpoints": endpoints,
                    "n_train_sequences": train_metrics["n_sequences"],
                    "between_iter_hard_gate": weight_gate_proof
                    or {"skipped": not promoted},
                },
            )
            _write_metrics(run_dir, it, metrics)

            # Commit all immutable artifacts, then advance the mutable pointer.
            next_state = json.loads(json.dumps(loop_state))
            next_state.update(
                {
                    "next_iteration": it + 1,
                    "last_completed_iteration": it,
                    "champion": identity.as_dict(),
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_evidence": heldout_champion_evidence,
                    "learner": learner_after.as_dict(),
                    "dormant_matchup_adapter_fit": (
                        candidate_adapter_fit
                        if learner_after.digest == candidate.digest
                        else dict(
                            loop_state.get("dormant_matchup_adapter_fit") or {}
                        )
                        if (
                            learner_after.digest == learner_before.digest
                            and str(
                                (
                                    loop_state.get("dormant_matchup_adapter_fit")
                                    or {}
                                ).get("checkpoint_digest")
                                or ""
                            )
                            == learner_before.digest
                        )
                        else {}
                    ),
                    "opponent_pool": [
                        _verified_checkpoint_identity(entry).as_dict()
                        for entry in opponent_pool
                    ],
                    "raw_advantage_history": adv_hist,
                    "policy_prev_agreement_history": agr_hist,
                    "live_pool_last_seq": live_pool_last_seq,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            next_state.setdefault("history", []).append(
                {
                    "iteration": it,
                    "completed": True,
                    "candidate": candidate.as_dict(),
                    "promoted": promoted,
                    "learner_before": learner_before.as_dict(),
                    "learner_after": learner_after.as_dict(),
                    "expert_rehearsal": rehearsal_record,
                    "promotion": promotion_report,
                    "incumbent_after": identity.as_dict(),
                    "raw_heldout_gate": raw_heldout_gate,
                    "active_gate_result": active_gate_result,
                    "research_control_result": research_control_result,
                    "heldout_audit": heldout_audit,
                    "heldout_champion": heldout_champion_identity.as_dict(),
                    "heldout_champion_updated": heldout_champion_updated,
                    "next_collection_publish": collection_publish_proof,
                    "stage_gate": asdict(gate),
                    "metrics": str(
                        run_dir / "metrics" / f"iter_{it:05d}.json"
                    ),
                    "eval": str(run_dir / "eval" / f"iter_{it:05d}.json"),
                }
            )
            _write_json_exclusive(
                run_dir / "commits" / f"iter_{it:05d}.json", next_state
            )
            _atomic_json(run_dir / "loop_state.json", next_state)
            if (
                args.mode == "specialist"
                and float(args.alakazam_guide_loss_weight) > 0.0
                and deck_guides.enabled()
                and os.environ.get(
                    "POKEBOT_FUTURE_GUIDE_WEIGHT_POLICY_REVISION"
                )
                == "44"
                and os.environ.get(
                    "POKEBOT_GUIDE_LEARNING_SEMANTICS_REVISION"
                )
                == "46"
            ):
                review_request = emit_review_request(
                    run_dir=run_dir,
                    specialist_id=str(args.specialist_archetype),
                    completed_iteration=it,
                    current_weight=float(args.alakazam_guide_loss_weight),
                    iteration_commit=(
                        run_dir / "commits" / f"iter_{it:05d}.json"
                    ),
                    guide_contract=Path(
                        os.environ["POKEBOT_CURRENT_DECK_GUIDE_CONTRACT"]
                    ),
                    guide_version=deck_guides.guide_version(),
                    prospective_policy_revision=44,
                    learning_semantics_revision=46,
                    consecutive_nonpositive_evaluations=int(
                        os.environ.get(
                            "POKEBOT_GUIDE_CONSECUTIVE_NONPOSITIVE_EVALUATIONS",
                            "0",
                        )
                    ),
                )
                if review_request is not None:
                    print(
                        "[pure_rl] GUIDE_WEIGHT_REVIEW_REQUEST "
                        f"iter={it} request={review_request}",
                        flush=True,
                    )
            next_it = it + 1
            if (
                args.expert_matchup_adapter_manifest is not None
                and next_it < int(args.iterations)
                and rehearsal_due(
                    next_it, int(args.expert_rehearsal_every)
                )
            ):
                from poke_bot.matchup_adapter_activation import (
                    build_adapter_rehearsal_authorization,
                )

                adapter_paths = expert_adapter_rehearsal_paths(
                    run_dir, next_it
                )
                if not adapter_paths.authorization.is_file():
                    proof = build_adapter_rehearsal_authorization(
                        run_dir=run_dir,
                        completed_iteration=it,
                        output_path=adapter_paths.authorization,
                    )
                    print(
                        "[pure_rl] EXPERT_MATCHUP_ADAPTER_BOUNDARY_STAGED "
                        f"before_iter={next_it} "
                        f"parent={proof.parent_checkpoint_digest[:19]}… "
                        f"authorization={proof.path}",
                        flush=True,
                    )
            if active_gate_result is not None and evaluation_active_gate is not None:
                raw_result_pointer = str(
                    evaluation_active_gate.get("exact_result_pointer") or ""
                ).strip()
                if not raw_result_pointer:
                    raise RuntimeError("active gate has no exact_result_pointer")
                published = _publish_committed_active_gate_result(
                    run_dir=run_dir,
                    active_gate=evaluation_active_gate,
                    result_pointer=Path(raw_result_pointer),
                )
                if published is None:
                    raise RuntimeError(
                        "immutable commit lost its active-gate result"
                    )
                result_pointer, result_commit = published
                _reconcile_passed_gate_research_controls(
                    registry_path=research_control_path,
                    gate_contract=evaluation_gate_contract,
                    exact_result_path=result_pointer,
                    commit_path=result_commit,
                    output_path=research_control_registry_output,
                )
            if it == 15:
                adapter_receipt = str(
                    os.environ.get(
                        "POKEBOT_MATCHUP_ADAPTER_BOUNDARY_RECEIPT", ""
                    )
                ).strip()
                if adapter_receipt:
                    # This is the only race-free point: iter15 is immutable and
                    # iter16 collection has not started. The receipt remains
                    # staged; creating it does not activate or train adapters.
                    from poke_bot.matchup_adapter_activation import (
                        build_activation_receipt,
                    )

                    proof = build_activation_receipt(
                        run_dir=run_dir,
                        output_path=Path(adapter_receipt),
                    )
                    print(
                        "[pure_rl] MATCHUP_ADAPTER_BOUNDARY_STAGED "
                        f"parent={proof.parent_checkpoint_digest} "
                        f"receipt={proof.path}",
                        flush=True,
                    )
            _atomic_json(
                run_dir / "iteration_runtime.json",
                {
                    "schema": "poke_bot.iteration_runtime/v1",
                    "iteration": int(it),
                    "phase": "completed",
                    "started_at": iteration_started_at,
                    "completed_at": completed_at,
                    "elapsed_sec": iteration_wall_sec,
                    "updated_at": time.time(),
                },
            )
            loop_state = next_state
            _retain_completed_artifacts(loop_state, it)
            print(
                f"[pure_rl] iter={it} games={metrics.games} "
                f"seqs={train_metrics['n_sequences']} "
                f"awr_w={metrics.awr_weight_mean:.3f} "
                f"heldout_wr={gate.win_rate:.3f} ({gate.games}g) "
                f"promoted={promoted} learner_carry={carry_candidate} "
                f"stage_gate={gate.passed} audit={heldout_audit.get('passed')} "
                f"abort={abort.abort} "
                f"gps={metrics.games_per_sec:.2f}",
                flush=True,
            )
            try:
                from poke_bot.pure_rl.wr_trend import wr_trend_line

                print(
                    f"[pure_rl] {wr_trend_line(run_dir, gate_wr=args.gate_wr)}",
                    flush=True,
                )
            except Exception as exc:  # never let a display helper break training
                print(f"[pure_rl] wr_trend WARN: {exc!r}", flush=True)

            owner_terminal_after = int(
                os.environ.get("POKEBOT_R304_TERMINAL_AFTER_ITERATION", "-1")
            )
            if owner_terminal_after >= 0 and int(it) == owner_terminal_after:
                if (
                    str(args.run_name) != "alakazam_new_list_direct_policy_r274"
                    or fixed_cycle_updates != R274_FIXED_CYCLE_UPDATES
                    or owner_terminal_after != 1
                ):
                    raise RuntimeError(
                        "terminal submission fence is valid only for r274 iter 1"
                    )
                collection_receipt_path = _collection_receipt_path(run_dir, it)
                submission_upload = _wait_for_r274_submission(
                    boundary_update=1,
                    checkpoint_identity=learner_after,
                    rehearsal_receipt_path=collection_receipt_path,
                )
                terminal_state = json.loads(json.dumps(loop_state))
                terminal_state["r304_owner_terminal"] = {
                    "schema": "poke_bot.alakazam_r274_iter1_owner_terminal/v1",
                    "owner_revision": 307,
                    "formal_holdout_skipped": True,
                    "research_control_measurement_skipped": True,
                    "measured_evaluation_pass_claimed": False,
                    "completed_iteration": 1,
                    "checkpoint": learner_after.as_dict(),
                    "collection_receipt": _r241_file_identity(
                        collection_receipt_path
                    ),
                    "submission_upload": submission_upload,
                    "next_collection_started": False,
                    "iteration_2_collection_allowed": False,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json(run_dir / "loop_state.json", terminal_state)
                loop_state.clear()
                loop_state.update(terminal_state)
                print(
                    "[pure_rl] R304_OWNER_TERMINAL_COMPLETE "
                    "iteration=1 next_collection_started=false",
                    flush=True,
                )
                return 0
            if gate.passed:
                terminal_minimum_reached = (
                    int(args.minimum_terminal_iteration) < 0
                    or int(it) >= int(args.minimum_terminal_iteration)
                )
                marker = (
                    _ensure_terminal_gate_marker(
                        run_dir,
                        loop_state,
                        preserve_first=(
                            bool(args.continue_after_gate)
                            or bool(fixed_cycle_updates)
                        ),
                        marker_name=str(args.terminal_gate_marker_name or ""),
                    )
                    if terminal_minimum_reached
                    else None
                )
                if not terminal_minimum_reached:
                    print(
                        "[pure_rl] GATE_PASS_PRE_TERMINAL_BOUNDARY "
                        f"iteration={it} "
                        f"minimum={int(args.minimum_terminal_iteration)}; "
                        "pass retained in immutable history; no terminal marker",
                        flush=True,
                    )
                    terminal_target_reached = False
                else:
                    terminal_target_reached = None
                if terminal_minimum_reached and marker is None:
                    raise RuntimeError(
                        "committed gate passed but terminal marker payload was absent"
                    )
                if marker is not None:
                    print(f"[pure_rl] {marker.name}", flush=True)
                passed_gate_id = str(
                    (active_gate_result or {}).get("gate_id") or ""
                )
                if terminal_target_reached is None:
                    terminal_target_reached = _terminal_gate_target_matches(
                        requested_gate_id=str(args.terminal_active_gate_id or ""),
                        passed_gate_id=passed_gate_id,
                        base_contract=active_gate_contract,
                    )
                if _gate_pass_should_stop(
                    fixed_cycle_updates=fixed_cycle_updates,
                    continue_after_gate=bool(args.continue_after_gate),
                    terminal_target_reached=bool(terminal_target_reached),
                ):
                    if terminal_target_reached:
                        print(
                            "[pure_rl] TERMINAL_ACTIVE_GATE_REACHED "
                            f"gate_id={passed_gate_id}",
                            flush=True,
                        )
                    break
                if fixed_cycle_updates:
                    print(
                        "[pure_rl] FIXED_CYCLE_GATE_NONTERMINAL "
                        f"updates={fixed_cycle_updates} completed={it}; "
                        "continuing through the exact cycle",
                        flush=True,
                    )
                else:
                    print(
                        "[pure_rl] CONTINUE_AFTER_GATE first-pass archive boundary "
                        "preserved; continuing curriculum",
                        flush=True,
                    )
            pause_seconds = _gate_boundary_pause_seconds(
                args,
                completed_iteration=int(it),
            )
            if pause_seconds > 0:
                print(
                    "[pure_rl] GATE_BOUNDARY_HARD_PAUSE "
                    f"iteration={it} seconds={pause_seconds:.1f} "
                    f"stage_gate_passed={bool(gate.passed)} "
                    "next_collection_blocked=true",
                    flush=True,
                )
                time.sleep(pause_seconds)
                print(
                    "[pure_rl] GATE_BOUNDARY_HARD_PAUSE_COMPLETE "
                    f"iteration={it} next_collection_blocked=false",
                    flush=True,
                )
            family_request = str(
                os.environ.get("POKEBOT_MARNIE_FAMILY_ACTIVATION_REQUEST", "")
            ).strip()
            family_upload_trigger = str(
                os.environ.get("POKEBOT_MARNIE_ITERATION9_UPLOAD_TRIGGER", "")
            ).strip()
            if (
                (family_request or family_upload_trigger)
                and args.mode == "specialist"
                and str(args.specialist_archetype) == "marnie-s-grimmsnarl-ex"
                and next_it < int(args.iterations)
            ):
                from poke_bot.archetype_family_activation import boundary_pause_hook

                family_decision = boundary_pause_hook(
                    request_path=(
                        Path(family_request).expanduser().resolve()
                        if family_request
                        else None
                    ),
                    trigger_path=(
                        Path(family_upload_trigger).expanduser().resolve()
                        if family_upload_trigger
                        else None
                    ),
                    run_dir=run_dir,
                    committed_iteration=int(it),
                    learner_digest=learner_after.digest,
                    next_collection_started=False,
                )
                print(
                    "[pure_rl] MARNIE_FAMILY_BOUNDARY "
                    f"action={family_decision['action']} "
                    f"target={family_decision.get('target_iteration')}",
                    flush=True,
                )
                if family_decision["action"] in {
                    "pause_for_atomic_activation",
                    "pause_for_required_evidence",
                    "already_paused",
                }:
                    # launch_pure_rl treats 75 as a deliberate monitor-stop
                    # boundary and will not restart the old selector.
                    raise SystemExit(75)
                if family_decision["action"] in {
                    "pause_for_post_activation_monitor",
                    "pause_for_required_rollback",
                }:
                    # Status 78 is the separate post-activation monitor and
                    # rollback boundary. It cannot be mistaken for the
                    # pre-activation evidence pause at status 75.
                    raise SystemExit(78)
            if next_it < int(args.iterations):
                collection_learner = learner_after
                if _r241_precollection_refresh_due(
                    next_it,
                    r241_peak_r195_training_contract,
                    fixed_cycle_updates,
                ):
                    collection_learner, boundary_record = (
                        _prepare_r241_precollection_refresh(
                            next_it, learner_after
                        )
                    )
                    learner_identity = collection_learner
                    precollect_rehearsals[next_it] = boundary_record
                print(
                    f"[pure_rl] kick collect iter={next_it} on continuous learner "
                    f"digest={collection_learner.digest[:19]}…",
                    flush=True,
                )
                pending_collect = _kick_collect(
                    next_it,
                    collection_learner.path,
                    collection_learner.digest,
                )
                if _r241_precollection_refresh_due(
                    next_it,
                    r241_peak_r195_training_contract,
                    fixed_cycle_updates,
                ):
                    _bind_r241_precollection_collection(
                        next_it,
                        pending_collect,
                        precollect_rehearsals[next_it],
                    )
            else:
                pending_collect = None
        _complete_fixed_cycle_boundary()
        if population_cycle_rehearsal_due(
            population_enabled=bool(args.population_own_models_only),
            next_iteration=int(loop_state.get("next_iteration") or 0),
            configured_rehearsal_every=int(args.expert_rehearsal_every),
            configured_rehearsal_epochs=int(args.expert_rehearsal_epochs),
        ):
            boundary_iteration = int(loop_state["next_iteration"])
            if boundary_iteration != int(args.iterations):
                raise RuntimeError(
                    "population RL lineage ended before its exact five-iteration "
                    "boundary"
                )
            parent = _verified_checkpoint_identity(loop_state["learner"])
            rehearsed, rehearsal_record = _prepare_expert_rehearsal(
                boundary_iteration,
                parent,
            )
            boundary = {
                "schema": "poke_bot.population_member_cycle_boundary/v1",
                "specialist_id": str(args.specialist_archetype),
                "rl_iterations_completed": POPULATION_RL_EPOCHS_PER_CYCLE,
                "expert_rehearsal_epochs_completed": (
                    POPULATION_REHEARSAL_EPOCHS_PER_CYCLE
                ),
                "parent": parent.as_dict(),
                "rehearsed": rehearsed.as_dict(),
                "expert_rehearsal": rehearsal_record,
                "external_agents_training_eligible": False,
                "next_lineage_must_start_from": rehearsed.as_dict(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            boundary_path = (
                run_dir
                / "population"
                / f"cycle_after_iter_{boundary_iteration:05d}.json"
            )
            _write_json_exclusive(boundary_path, boundary)
            population_state = json.loads(json.dumps(loop_state))
            population_state["learner"] = rehearsed.as_dict()
            population_state["population_cycle_boundary"] = {
                "path": str(boundary_path),
                "specialist_id": str(args.specialist_archetype),
                "rl_iterations_completed": POPULATION_RL_EPOCHS_PER_CYCLE,
                "expert_rehearsal_epochs_completed": (
                    POPULATION_REHEARSAL_EPOCHS_PER_CYCLE
                ),
                "checkpoint": rehearsed.as_dict(),
            }
            population_state["updated_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _atomic_json(run_dir / "loop_state.json", population_state)
            loop_state = population_state
            print(
                "[pure_rl] POPULATION_MEMBER_CYCLE_COMPLETE "
                f"specialist={args.specialist_archetype} "
                f"rl_epochs={POPULATION_RL_EPOCHS_PER_CYCLE} "
                f"rehearsal_epochs={POPULATION_REHEARSAL_EPOCHS_PER_CYCLE} "
                f"checkpoint={rehearsed.digest[:19]}…",
                flush=True,
            )
        return 0
    finally:
        expert_cache.release()
        leaf.stop()
        if remote_farm is not None:
            try:
                remote_farm.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.allow_single_gpu:
        os.environ["PURE_RL_ALLOW_SINGLE_GPU"] = "1"
    if args.smoke:
        return run_smoke_loop(args)
    return run_full_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
