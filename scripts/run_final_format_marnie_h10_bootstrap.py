#!/usr/bin/env python3
"""Run Marnie's exact 25-epoch bootstrap on an already-H10 Fusion-v3 child."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint, device as device_mod  # noqa: E402
from poke_bot.pure_rl.expert_rehearsal import (  # noqa: E402
    ResidentExpertCorpusCache,
    resolve_expert_manifest,
)
from poke_bot.pure_rl.model_registry import freeze_model, verify_frozen_model  # noqa: E402
from poke_bot.strategic_schedule import (  # noqa: E402
    EXPANDED_HEAD_IDS,
    expanded_head_epoch_plan,
)
from poke_bot.train import (  # noqa: E402
    GUIDE_TRAINING_MODE_DIRECTIONAL,
    belief_card_vocab_from_state,
    supervised_rehearsal_step,
)
from scripts.run_starmie_expert_bootstrap import (  # noqa: E402
    TARGETS,
    _manifest_expanded_targets,
    atomic_json,
    load_expanded_head_contract,
    validate_expanded_epoch_checkpoint,
)
from scripts.validate_final_format_marnie_h10 import validate as validate_h10  # noqa: E402


SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
STATE_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_state/v1"
READY_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_ready/v1"
EPOCH_SCHEMA = "poke_bot.final_format_marnie_h10_bootstrap_epoch/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    body = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"Marnie bootstrap receipt changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink(missing_ok=True)


def _validate_boundary(
    *,
    prestage_path: Path,
    latest_core_path: Path,
    child_path: Path,
    migration_path: Path,
    roles_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prestage = _read(prestage_path)
    if (
        prestage.get("schema")
        != "poke_bot.final_format_marnie_refresh_prestage/v1"
        or prestage.get("status") != "authorized_preparation_started"
        or prestage.get("specialist_id") != SPECIALIST_ID
        or int(prestage.get("boundary_iteration") or -1) != 20
        or prestage.get("final_capacity_profile") != "H10-I/v1"
        or prestage.get("final_decision_fusion_schema")
        != "poke_bot.causal_decision_fusion/v3"
        or prestage.get("guide_training_mode") != GUIDE_TRAINING_MODE_DIRECTIONAL
        or prestage.get("training_authority") is not False
        or prestage.get("selector_authority") is not False
    ):
        raise RuntimeError("Marnie iteration-20 pre-stage is not authoritative")

    pointer = _read(latest_core_path)
    ready_path = Path(str(pointer.get("ready") or "")).expanduser().resolve()
    family_path = Path(str(pointer.get("family") or "")).expanduser().resolve()
    if (
        pointer.get("schema") != "poke_bot.latest_cumulative_core_pointer/v1"
        or not ready_path.is_file()
        or checkpoint.checkpoint_digest(ready_path) != pointer.get("ready_digest")
    ):
        raise RuntimeError("latest accepted cumulative-core pointer changed")
    ready = _read(ready_path)
    frozen = verify_frozen_model(family_path)
    if (
        ready.get("checkpoint_digest") != pointer.get("checkpoint_digest")
        or frozen.get("checkpoint_digest") != pointer.get("checkpoint_digest")
        or Path(str(frozen.get("model_path") or "")).resolve()
        != Path(str(ready.get("checkpoint") or "")).resolve()
    ):
        raise RuntimeError("latest accepted cumulative-core identity disagrees")

    migration = _read(migration_path)
    h10 = validate_h10(
        child_path=child_path,
        migration_path=migration_path,
        roles_path=roles_path,
    )
    if migration.get("parent_checkpoint_sha256") != pointer.get(
        "checkpoint_digest"
    ):
        raise RuntimeError("Marnie H10 child was not built from latest accepted core")
    return prestage, pointer, h10


def _directional_rows(metrics: dict[str, Any]) -> int:
    curriculum = dict(metrics.get("guide_curriculum_head_metrics") or {})
    directional = dict(curriculum.get("directional_route_ranking") or {})
    heads = set(directional.get("heads") or [])
    required = {
        "action_q",
        "action_resource",
        "action_utility",
        "setup_board_outcome",
        "combo_state",
    }
    if not required.issubset(heads):
        raise RuntimeError("directional guide omitted a required Marnie route")
    return int(directional.get("eligible_rows") or 0)


def _combo_rows(metrics: dict[str, Any]) -> int:
    """Return observed combo rows while preserving valid all-masked corpora."""

    combo = dict(metrics.get("combo_state_metrics") or {})
    total = int(combo.get("total_rows") or 0)
    eligible = int(combo.get("eligible_rows") or 0)
    if total <= 0 or eligible < 0 or eligible > total:
        raise RuntimeError("Marnie combo-head masking telemetry is invalid")
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prestage", type=Path, required=True)
    parser.add_argument("--latest-core", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--role-route-receipt", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--curriculum-spec", type=Path, required=True)
    parser.add_argument("--head-role-map", type=Path, required=True)
    parser.add_argument("--curriculum-validation", type=Path, required=True)
    parser.add_argument("--rl-protocol", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-pack-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--split-seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.epochs != 25:
        raise ValueError("Marnie H10 bootstrap is locked to exactly 25 epochs")
    if args.batch_size <= 0 or args.batch_size > 3072:
        raise ValueError("Marnie H10 bootstrap batch size must be in 1..3072")

    prestage, core_pointer, h10 = _validate_boundary(
        prestage_path=args.prestage.expanduser().resolve(),
        latest_core_path=args.latest_core.expanduser().resolve(),
        child_path=args.child.expanduser().resolve(),
        migration_path=args.migration_receipt.expanduser().resolve(),
        roles_path=args.role_route_receipt.expanduser().resolve(),
    )
    for key, path, expected in (
        ("curriculum_spec", args.curriculum_spec, prestage.get("curriculum_spec_sha256")),
        ("head_role_map", args.head_role_map, prestage.get("head_role_map_sha256")),
        (
            "curriculum_validation",
            args.curriculum_validation,
            prestage.get("curriculum_validation_sha256"),
        ),
        ("expert", args.expert_manifest, prestage.get("expert_manifest_sha256")),
    ):
        if checkpoint.checkpoint_digest(path.expanduser().resolve()) != expected:
            raise RuntimeError(f"Marnie pre-stage {key} checksum changed")

    expanded_raw, expanded_identity = load_expanded_head_contract(args.rl_protocol)
    identity = resolve_expert_manifest(
        args.expert_manifest.expanduser().resolve(),
        min_decisions=100_000,
        require_protected=True,
        required_archetype=SPECIALIST_ID,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=TARGETS,
        required_expanded_target_schema=expanded_identity["target_schema"],
        required_expanded_target_digest=expanded_identity["target_schema_digest"],
        required_expanded_heads=tuple(EXPANDED_HEAD_IDS),
    )
    expanded_targets = _manifest_expanded_targets(
        Path(identity.path), decisions=identity.decisions
    )
    child = args.child.expanduser().resolve()
    child_digest = checkpoint.checkpoint_digest(child)
    child_payload = checkpoint.load_checkpoint(child, map_location="cpu")
    belief_vocab = belief_card_vocab_from_state(
        dict(child_payload.get("model_state_dict") or {})
    )
    device = device_mod.training_device(prefer_name="RTX PRO 5000", allow_cpu=False)
    cache = ResidentExpertCorpusCache(
        cpu_pack_root=args.cpu_pack_root.expanduser().resolve()
    )
    corpus = cache.prepare(
        identity,
        device=device,
        seed=args.split_seed,
        max_context=320,
        belief_card_vocab=belief_vocab,
    )
    if not corpus.has_exact_targets:
        raise RuntimeError("Marnie H10 corpus lost required exact targets")

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state = _read(state_path) if state_path.is_file() else {}
    if state and (
        state.get("schema") != STATE_SCHEMA
        or state.get("initial_checkpoint_sha256") != child_digest
        or state.get("expert_manifest_sha256") != identity.digest
        or state.get("core_checkpoint_sha256")
        != core_pointer.get("checkpoint_digest")
    ):
        raise RuntimeError("Marnie H10 bootstrap resume identity changed")
    history = list(state.get("history") or [])
    parent = child
    parent_digest = child_digest
    if history:
        last = dict(history[-1])
        parent = Path(str(last.get("checkpoint") or "")).resolve()
        parent_digest = str(last.get("checkpoint_sha256") or "")
        if checkpoint.checkpoint_digest(parent) != parent_digest:
            raise RuntimeError("Marnie H10 resume parent changed")
    start_epoch = len(history) + 1
    best_loss = float(state.get("best_validation_loss") or math.inf)
    best_path = str(state.get("best_checkpoint") or "")
    best_digest = str(state.get("best_checkpoint_sha256") or "")

    try:
        for epoch in range(start_epoch, 26):
            plan = expanded_head_epoch_plan(expanded_raw, epoch)
            output = checkpoint_dir / f"epoch_{epoch:02d}.pt"
            if output.exists():
                raise RuntimeError(f"unregistered Marnie epoch already exists: {output}")
            result = supervised_rehearsal_step(
                corpus,
                base_ckpt=parent,
                output_path=output,
                parent_digest=parent_digest,
                rehearsal_iteration=epoch,
                manifest_identity={
                    **identity.as_dict(),
                    "expanded_strategic_targets": expanded_targets,
                },
                epochs=1,
                lr=5e-5,
                requested_batch_size=args.batch_size,
                seed=20260801 + epoch,
                corpus_split_seed=args.split_seed,
                device=device,
                aux_loss_weight=0.05,
                opp_hand_loss_weight=0.05,
                opp_remainder_loss_weight=0.05,
                lethal_threat_loss_weight=0.025,
                prize_race_loss_weight=0.025,
                alakazam_guide_loss_weight=0.05,
                current_deck_guide_training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
                setup_board_outcome_loss_weight=0.025,
                combo_state_loss_weight=0.025,
                current_deck_guide_curriculum_spec=str(args.curriculum_spec.resolve()),
                current_deck_guide_head_role_map=str(args.head_role_map.resolve()),
                current_deck_guide_curriculum_validation_receipt=str(
                    args.curriculum_validation.resolve()
                ),
                expanded_head_loss_weights=dict(plan.loss_weights),
                expanded_head_schedule=plan.as_dict(),
                # The H10 step-zero child retains accepted-core training
                # telemetry as immutable provenance.  Marnie's exact
                # specialist-local 25-epoch schedule begins here, so epoch 1
                # must not count the core's historical head updates as Marnie
                # bootstrap training.
                reset_expanded_training_history=(epoch == 1),
                output_archetype_id=SPECIALIST_ID,
                output_model_id=f"{args.run_name}.epoch{epoch:02d}",
                extra_updates={
                    "final_format_marnie_h10_bootstrap": {
                        "schema": EPOCH_SCHEMA,
                        "epoch": epoch,
                        "initial_checkpoint_sha256": child_digest,
                        "core_checkpoint_sha256": core_pointer.get(
                            "checkpoint_digest"
                        ),
                        "expert_manifest_sha256": identity.digest,
                        "guide_mode": GUIDE_TRAINING_MODE_DIRECTIONAL,
                        "guide_weight": 0.05,
                        "decision_fusion_schema": (
                            "poke_bot.causal_decision_fusion/v3"
                        ),
                    }
                },
            )
            train_metrics = dict(result.get("train_metrics") or {})
            validation_metrics = dict(result.get("validation_metrics") or {})
            if _directional_rows(train_metrics) <= 0 or _directional_rows(
                validation_metrics
            ) <= 0:
                raise RuntimeError("Marnie directional guide received no labeled rows")
            # Marnie's public feature corpus predates combo-state annotations.
            # Missing exact causal targets must remain masked rather than be
            # fabricated.  The directional check above still requires the
            # combo route (and every other guided route) to receive real guide
            # ranking updates on labeled legal-option rows.
            train_combo_rows = _combo_rows(train_metrics)
            validation_combo_rows = _combo_rows(validation_metrics)
            expanded_contract = validate_expanded_epoch_checkpoint(
                output,
                plan=plan,
                identity=expanded_identity,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            payload = checkpoint.load_checkpoint(output, map_location="cpu")
            config = dict(payload.get("model_config") or {})
            fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
            if (
                payload.get("archetype_id") != SPECIALIST_ID
                or config.get("h10_capacity_enabled") is not True
                or config.get("decision_fusion_typed_output_centered_routes_enabled")
                is not True
                or fusion.get("schema") != "poke_bot.causal_decision_fusion/v3"
                or len(fusion.get("required_heads") or []) != 19
            ):
                raise RuntimeError("Marnie epoch escaped H10/Fusion-v3")
            metric = float(validation_metrics.get("total_loss") or math.inf)
            if not math.isfinite(metric):
                raise RuntimeError("Marnie validation loss is not finite")
            row = {
                "epoch": epoch,
                "checkpoint": str(output),
                "checkpoint_sha256": str(result["candidate_digest"]),
                "parent_sha256": parent_digest,
                "validation_loss": metric,
                "expanded_head_training": expanded_contract,
                "train_directional_rows": _directional_rows(train_metrics),
                "validation_directional_rows": _directional_rows(validation_metrics),
                "train_combo_observed_rows": train_combo_rows,
                "validation_combo_observed_rows": validation_combo_rows,
            }
            history.append(row)
            if set(plan.enabled_heads) == set(EXPANDED_HEAD_IDS) and metric < best_loss:
                best_loss = metric
                best_path = str(output)
                best_digest = str(result["candidate_digest"])
            parent = output
            parent_digest = str(result["candidate_digest"])
            atomic_json(
                state_path,
                {
                    "schema": STATE_SCHEMA,
                    "status": "training",
                    "specialist_id": SPECIALIST_ID,
                    "initial_checkpoint": str(child),
                    "initial_checkpoint_sha256": child_digest,
                    "core_checkpoint_sha256": core_pointer.get("checkpoint_digest"),
                    "expert_manifest_sha256": identity.digest,
                    "history": history,
                    "best_validation_loss": best_loss,
                    "best_checkpoint": best_path,
                    "best_checkpoint_sha256": best_digest,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    finally:
        cache.release()

    if [int(row.get("epoch") or 0) for row in history] != list(range(1, 26)):
        raise RuntimeError("Marnie H10 bootstrap did not complete epochs 1..25")
    best = Path(best_path).resolve()
    if not best.is_file() or checkpoint.checkpoint_digest(best) != best_digest:
        raise RuntimeError("Marnie selected H10 checkpoint is invalid")
    frozen = freeze_model(
        registry_root=args.registry_root.expanduser().resolve(),
        family=args.family,
        display_name="Marnie's Grimmsnarl ex Final-Format H10 Refresh",
        checkpoint=best,
        expected_digest=best_digest,
        provenance={
            "specialist_id": SPECIALIST_ID,
            "initial_h10_validation": h10,
            "latest_accepted_core": core_pointer,
            "expert_manifest": identity.as_dict(),
            "epochs_completed": 25,
            "guide_mode": GUIDE_TRAINING_MODE_DIRECTIONAL,
            "guide_weight": 0.05,
            "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
            "learned_head_count": 19,
            "learned_route_count": 19,
            "runtime_authority": "none_until_managed_rl_registration",
        },
        evidence={
            "kind": "final_format_marnie_h10_expert_validation",
            "best_validation_loss": best_loss,
            "epochs_completed": 25,
        },
        harden_permissions=True,
    )
    frozen = verify_frozen_model(Path(str(frozen["model_path"])).parent)
    ready = {
        "schema": READY_SCHEMA,
        "status": "ready_for_managed_rl_registration",
        "specialist_id": SPECIALIST_ID,
        "run_name": args.run_name,
        "checkpoint": frozen["model_path"],
        "checkpoint_sha256": frozen["checkpoint_digest"],
        "initial_checkpoint_sha256": child_digest,
        "core_checkpoint_sha256": core_pointer.get("checkpoint_digest"),
        "expert_manifest_sha256": identity.digest,
        "epochs_completed": 25,
        "capacity_profile": "H10-I/v1",
        "decision_fusion_schema": "poke_bot.causal_decision_fusion/v3",
        "learned_head_count": 19,
        "learned_route_count": 19,
        "guide_mode": GUIDE_TRAINING_MODE_DIRECTIONAL,
        "guide_weight": 0.05,
        "training_authority": False,
        "selector_authority": False,
        "next_boundary": "router_format_6_migration_and_managed_rl_registration",
    }
    _write_once(args.ready.expanduser().resolve(), ready)
    atomic_json(state_path, {**_read(state_path), "status": "complete", "ready": ready})
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
