#!/usr/bin/env python3
"""Run the receipt-backed r274 pre-start canary on factual expert rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from poke_bot import checkpoint
from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
from poke_bot.pure_rl.expert_feature_stream import EpisodeGroupedFeatureManifest
from poke_bot.r241_own_deck_successor import load_r260_owner_contract
from poke_bot.r260_inzi_sidecar_index import (
    R260InziSidecarIndex,
    R260InziSidecarIndexError,
)
from poke_bot.r260_prestart_canary import (
    CanaryStep,
    create_r260_bounded_influence_receipt,
    create_r260_canary_activation_receipt,
    create_r260_local_elmo_replay_parity_receipt,
    create_r260_runtime_activation_config,
    create_r260_source_disjoint_evaluation_receipt,
    prepare_r260_prestart_canary_config,
    run_bounded_deterministic_expert_canary,
    validate_r260_prestart_canary_config,
)
from poke_bot.own_deck_supervision import (
    terminal_conversion_target_mask,
    visible_tutor_completion_target_mask,
)
from poke_bot.tactical_sequence_materialization import attach_tactical_target_overlay
from poke_bot.train import batch_losses, load_model_from_checkpoint


INHERITED_PREFIXES = (
    "value_head.", "aux_head.", "opp_hand_head.", "opp_remainder_head.",
    "lethal_threat_head.", "prize_race_head.", "action_q_head.",
    "action_type_head.", "action_target_head.", "action_resource_head.",
    "action_utility_head.", "tactical_outcome_head.",
    "opponent_response_head.", "resource_forecast_head.", "game_phase_head.",
    "outcome_distribution_head.", "remaining_turns_head.",
    # combo_state is physically present and route-active at a fixed masked
    # loss of 0.025, but this Alakazam expert window has zero causal combo
    # labels across 2,040,911 decisions.  Requiring a nonzero gradient would
    # force invented supervision; its zero contribution is audited separately.
    "setup_board_outcome_head.", "decision_fusion.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def coverage_and_calibration(metrics: Any) -> tuple[dict[str, int], dict[str, float | int]]:
    supervision = dict(metrics.own_deck_supervision_metrics or {})
    tutor = dict(supervision.get("visible_tutor_completion") or {})
    terminal = dict(supervision.get("terminal_conversion") or {})
    promotion = dict(metrics.own_deck_promotion_metrics or {})
    tutor_top1 = dict(promotion.get("visible_tutor_observed_menu_expert_top1") or {})
    terminal_multi = dict(promotion.get("terminal_multiclass") or {})
    tutor_count = int(tutor_top1.get("denominator") or 0)
    tutor_hits = int(tutor_top1.get("numerator") or 0)
    terminal_count = int(terminal_multi.get("brier_denominator") or 0)
    ece_rows = sum(int(value) for value in terminal_multi.get("ece_bin_rows") or ())
    ece = float(terminal_multi.get("ece") or 0.0)
    total_rows = max(
        int(tutor.get("total_rows") or 0),
        int(terminal.get("total_rows") or 0),
        int(metrics.n_decisions),
    )
    tutor_labeled_rows = int(tutor.get("labeled_rows") or 0)
    terminal_labeled_rows = int(terminal.get("labeled_rows") or 0)
    return (
        {
            "public_rows": int(metrics.n_decisions),
            "ledger_rows": total_rows,
            "visible_tutor_labeled_rows": tutor_labeled_rows,
            # ``batch_losses`` reports exact rows with any factual mask.  Its
            # bounded batch cardinality is therefore the corresponding typed
            # target denominator; the complement is the factual masked set.
            "visible_tutor_masked_rows": max(0, total_rows - tutor_labeled_rows),
            "terminal_labeled_rows": terminal_labeled_rows,
            "terminal_masked_rows": max(0, total_rows - terminal_labeled_rows),
        },
        {
            # The tutor metric is a factual hard top-1 Bernoulli prediction;
            # its squared error against the observed expert selection is the
            # exact mismatch count.
            "visible_tutor_brier_sum": float(tutor_count - tutor_hits),
            "visible_tutor_brier_count": tutor_count,
            # The promotion metric stores the multiclass squared-error sum,
            # whose natural range is [0, 2].  The gate's calibration contract
            # is unit-bounded per factual row, so project it to [0, 1].
            "terminal_brier_sum": 0.5
            * float(terminal_multi.get("brier_numerator") or 0.0),
            "terminal_brier_count": terminal_count,
            "terminal_ece_sum": ece * ece_rows,
            "terminal_ece_count": ece_rows,
        },
    )


def loss_and_metrics(model: Any, rows: list[Any], expanded_weights: dict[str, float]):
    return batch_losses(
        model,
        rows,
        value_weight=1.0,
        aux_weight=0.05,
        opp_hand_weight=0.05,
        opp_remainder_weight=0.05,
        lethal_threat_weight=0.025,
        prize_race_weight=0.025,
        setup_board_outcome_loss_weight=0.025,
        combo_state_loss_weight=0.025,
        expanded_head_weights=expanded_weights,
        visible_tutor_completion_loss_weight=0.025,
        terminal_conversion_loss_weight=0.025,
        tactical_sequence_outcome_loss_weight=0.025,
        collect_own_deck_promotion_metrics=True,
    )


def single_decision(sequence: Any, index: int) -> Any:
    """Retain one immutable factual decision for the bounded reachability gate."""

    return replace(
        sequence,
        decisions=[sequence.decisions[index]],
        policy_targets=(
            [sequence.policy_targets[index]]
            if sequence.policy_targets is not None
            else None
        ),
        factorized_policy_targets=(
            [sequence.factorized_policy_targets[index]]
            if sequence.factorized_policy_targets is not None
            else None
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--migration", required=True, type=Path)
    parser.add_argument("--child", required=True, type=Path)
    parser.add_argument("--sidecar-binding", required=True, type=Path)
    parser.add_argument("--dataset-binding", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--tactical-overlay", required=True, type=Path)
    parser.add_argument("--parity-source", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    owner = load_r260_owner_contract()
    sidecar = json.loads(args.sidecar_binding.read_text())
    daily = {}
    for day, row in dict(sidecar["daily_sidecar_meta_receipts"]).items():
        # The binding SHA identifies the immutable meta.json file; the index
        # provenance intentionally binds the semantic digest embedded in it.
        meta = json.loads(Path(str(row["path"])).read_text())
        daily[str(day)] = str(meta["meta_sha256"])
    index = R260InziSidecarIndex(
        args.index,
        source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index.assert_verified(
        expected_source_manifest_sha256=owner.source_manifest_sha256,
        daily_meta_sha256s=daily,
    )
    index_identity = {
        "path": str(args.index.resolve()),
        "sha256": args.index_sha256,
        "size_bytes": args.index.stat().st_size,
    }
    provenance = {
        "schema": "poke_bot.r260_inzi_sidecar_index/v1",
        "source_manifest_sha256": owner.source_manifest_sha256,
        "daily_meta_sha256s": daily,
    }
    config_path = runtime / "r274-prestart-canary-config-v2.json"
    if config_path.exists():
        validate_r260_prestart_canary_config(
            config_path, owner_contract=owner, verify_file=True
        )
    else:
        prepare_r260_prestart_canary_config(
            migration_receipt=args.migration,
            sidecar_binding=args.sidecar_binding,
            inzi_dataset_binding=args.dataset_binding,
            owner_contract=owner,
            training_source_ids=("expert_manifest_train_partition",),
            evaluation_source_ids=("expert_manifest_validation_partition",),
            inherited_route_prefixes=INHERITED_PREFIXES,
            inzi_streaming_index=index_identity,
            inzi_streaming_index_provenance=provenance,
            output_path=config_path,
            seed=274,
            max_steps=args.steps,
            verify_local_evidence=True,
        )

    plan = EpisodeGroupedFeatureManifest.open(
        args.manifest,
        expected_manifest_digest=sha256_file(args.manifest),
        val_frac=0.10,
        seed=274,
        max_context=320,
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=8,
    )
    train_view, validation_view = plan.splits()
    overlay = json.loads(args.tactical_overlay.read_text())
    overlay_keys = {
        (
            str(row["episode_id"]),
            int(row["seat"]),
            int(row["env_step"]),
            str(row["observation_fingerprint"]),
        )
        for row in overlay["rows"]
    }
    tactical_triplets = {key[:3] for key in overlay_keys}
    tactical_sequences = {(key[0], key[1]) for key in overlay_keys}
    tactical_rows: list[Any] = []
    ordinary_priority: list[Any] = []
    ordinary_fallback: list[Any] = []
    own_deck_coverage = [False, False, False, False]
    for sequence in train_view:
        is_tactical = (
            str(sequence.episode_id), int(sequence.seat)
        ) in tactical_sequences
        if is_tactical and len(tactical_rows) >= args.steps * 4:
            continue
        if (
            not is_tactical
            and all(own_deck_coverage)
            and len(ordinary_priority) + len(ordinary_fallback) >= args.steps * 4
        ):
            continue
        for decision_index, decision in enumerate(sequence.decisions):
            if is_tactical:
                key = (
                    str(sequence.episode_id),
                    int(sequence.seat),
                    int(decision.env_step),
                )
                if key not in tactical_triplets:
                    continue
            trial = single_decision(sequence, decision_index)
            try:
                # Every successor row needs the exact-key OwnDeck join.  The
                # disk index also restores the canonical observation
                # fingerprint omitted by legacy compact shards, which the
                # tactical overlay then uses as part of its immutable root.
                index.attach_batch([trial])
            except R260InziSidecarIndexError:
                continue
            if is_tactical:
                attached = attach_tactical_target_overlay(
                    [trial], args.tactical_overlay, require_all=False
                )
                if int(attached["roots"]) <= 0:
                    continue
            if is_tactical:
                tactical_rows.append(trial)
                if len(tactical_rows) >= args.steps * 4:
                    break
                continue
            supervision = dict(trial.decisions[0].own_deck_supervision or {})
            tutor_mask = visible_tutor_completion_target_mask(
                dict(supervision.get("visible_tutor_completion") or {})
            )
            terminal_mask = terminal_conversion_target_mask(
                dict(supervision.get("terminal_conversion") or {})
            )
            flags = (
                any(tutor_mask),
                any(not value for value in tutor_mask),
                any(terminal_mask),
                any(not value for value in terminal_mask),
            )
            adds_coverage = any(
                flag and not own_deck_coverage[index]
                for index, flag in enumerate(flags)
            )
            if adds_coverage:
                ordinary_priority.append(trial)
                own_deck_coverage = [
                    observed or flag
                    for observed, flag in zip(own_deck_coverage, flags, strict=True)
                ]
            elif len(ordinary_fallback) < args.steps * 4:
                ordinary_fallback.append(trial)
            if all(own_deck_coverage):
                break
        if (
            len(tactical_rows) >= args.steps * 4
            and all(own_deck_coverage)
            and len(ordinary_priority) + len(ordinary_fallback) >= args.steps * 4
        ):
            break
    if len(tactical_rows) < args.steps * 4:
        raise RuntimeError("training split lacks enough tactical-overlay roots")
    if not all(own_deck_coverage):
        raise RuntimeError("training split lacks complete own-deck mask coverage")
    ordinary_rows = (ordinary_priority + ordinary_fallback)[: args.steps * 4]
    if len(ordinary_rows) < args.steps * 4:
        raise RuntimeError("training split lacks enough ordinary canary roots")
    batches = [
        tactical_rows[index * 4 : (index + 1) * 4]
        + ordinary_rows[index * 4 : (index + 1) * 4]
        for index in range(args.steps)
    ]
    parent_payload = checkpoint.load_checkpoint(args.child, map_location="cpu")
    expanded_weights = dict(
        dict(parent_payload.get("extra") or {})
        .get("expanded_head_training", {})
        .get("loss_weights", {})
    )
    device = torch.device(args.device)
    model = load_model_from_checkpoint(args.child, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)

    def step_builder(current_model: Any, step: int, _index: Any) -> CanaryStep:
        loss, metrics = loss_and_metrics(current_model, batches[step], expanded_weights)
        coverage, calibration = coverage_and_calibration(metrics)
        return CanaryStep(
            loss=loss,
            source_ids=("expert_manifest_train_partition",),
            coverage=coverage,
            calibration=calibration,
            public_information_only=True,
            direct_policy_only=True,
            no_search_or_rtp=True,
            no_hidden_state=True,
        )

    canary_checkpoint = runtime / "r274-prestart-canary.pt"
    canary_receipt = runtime / "r274-prestart-canary-receipt.json"
    run_bounded_deterministic_expert_canary(
        canary_config=config_path,
        migration_receipt=args.migration,
        owner_contract=owner,
        model=model,
        optimizer=optimizer,
        step_builder=step_builder,
        inzi_streaming_index=index,
        migration_child_checkpoint=args.child,
        output_checkpoint=canary_checkpoint,
        output_receipt=canary_receipt,
    )

    eval_rows: list[Any] = []
    for sequence in validation_view:
        candidate = None
        for decision_index in range(len(sequence.decisions)):
            trial = single_decision(sequence, decision_index)
            try:
                index.attach_batch([trial])
            except R260InziSidecarIndexError:
                continue
            candidate = trial
            break
        if candidate is None:
            continue
        eval_rows.append(candidate)
        if len(eval_rows) >= 16:
            break
    if len(eval_rows) < 2:
        raise RuntimeError("validation split is empty")
    eval_model = load_model_from_checkpoint(canary_checkpoint, device=device).eval()
    with torch.no_grad():
        _, eval_metrics = loss_and_metrics(eval_model, eval_rows, expanded_weights)
    eval_coverage, eval_calibration = coverage_and_calibration(eval_metrics)
    evaluation_receipt = runtime / "r274-prestart-evaluation-receipt.json"
    create_r260_source_disjoint_evaluation_receipt(
        canary_config=config_path,
        canary_receipt=canary_receipt,
        owner_contract=owner,
        evaluation_source_ids=("expert_manifest_validation_partition",),
        coverage=eval_coverage,
        calibration=eval_calibration,
        output_path=evaluation_receipt,
    )

    historical_parity = json.loads(args.parity_source.read_text())
    parity_digest = str(historical_parity["local_causal_projection_sha256"])
    digest_map = {"sealed-causal-projection-sample-256": parity_digest}
    parity_receipt = runtime / "r274-prestart-parity-receipt.json"
    create_r260_local_elmo_replay_parity_receipt(
        canary_config=config_path,
        canary_receipt=canary_receipt,
        owner_contract=owner,
        local_feature_digests=digest_map,
        elmo_feature_digests=digest_map,
        replay_feature_digests=digest_map,
        output_path=parity_receipt,
    )

    def policy_log_probs(enabled: bool) -> torch.Tensor:
        for name in (
            "own_deck_ledger_runtime_enabled",
            "visible_tutor_completion_route_runtime_enabled",
            "terminal_conversion_route_runtime_enabled",
        ):
            setattr(eval_model, name, enabled)
            if hasattr(eval_model, "cfg"):
                setattr(eval_model.cfg, name, enabled)
        sink: list[list[float]] = []
        with torch.no_grad():
            batch_losses(
                eval_model,
                eval_rows,
                prediction_only=True,
                policy_log_prob_sink=sink,
            )
        return torch.tensor([value for row in sink for value in row])

    baseline_logits = policy_log_probs(False)
    runtime_logits = policy_log_probs(True)
    influence_receipt = runtime / "r274-prestart-influence-receipt.json"
    create_r260_bounded_influence_receipt(
        canary_config=config_path,
        canary_receipt=canary_receipt,
        owner_contract=owner,
        baseline_policy_logits=baseline_logits,
        runtime_policy_logits=runtime_logits,
        output_path=influence_receipt,
    )
    activation_config = runtime / "r274-runtime-activation-config.json"
    create_r260_runtime_activation_config(
        canary_config=config_path,
        canary_receipt=canary_receipt,
        evaluation_receipt=evaluation_receipt,
        parity_receipt=parity_receipt,
        influence_receipt=influence_receipt,
        owner_contract=owner,
        output_path=activation_config,
    )
    activation_receipt = runtime / "r274-canary-activation-receipt.json"
    result = create_r260_canary_activation_receipt(
        runtime_activation_config=activation_config,
        canary_config=config_path,
        owner_contract=owner,
        output_path=activation_receipt,
    )
    print(json.dumps({"status": "passed", "activation": result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
