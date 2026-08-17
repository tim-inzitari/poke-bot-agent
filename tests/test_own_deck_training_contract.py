"""Focused receipts for the dormant r258/r259 next-train plan."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from poke_bot import own_deck_successor as successor
from poke_bot import own_deck_training_contract as contract


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _manifest() -> successor.OwnDeckSuccessorManifest:
    return successor.load_canonical_manifest()


def _refresh(manifest: successor.OwnDeckSuccessorManifest) -> dict[str, object]:
    return successor.seal_receipt(
        {
            "schema": successor.REFRESH_COMPLETION_SCHEMA,
            "status": "completed",
            "specialist_id": "alakazam",
            "terminal_completion": True,
            "frozen": True,
            "registered": True,
            "candidate_id": successor.CANDIDATE_ID,
            "manifest_sha256": manifest.identity.sha256,
            "immutable_refresh_lineage": {"id": "r241", "sha256": _sha("1")},
            "completed_refresh_boundary": {"id": "iter_00009", "sha256": _sha("2")},
            "checkpoint": {"id": "terminal.pt", "sha256": _sha("3")},
            "source_receipt_chain": {"integrity_verified": True, "sha256": _sha("4")},
            "runtime_receipt_chain": {"integrity_verified": True, "sha256": _sha("5")},
        }
    )


def _stages(
    manifest: successor.OwnDeckSuccessorManifest,
) -> dict[successor.OwnDeckSuccessorStage, dict[str, object]]:
    rows: dict[successor.OwnDeckSuccessorStage, dict[str, object]] = {}
    prior: dict[str, str] = {}
    for index, stage in enumerate(manifest.pre_refresh_stages):
        receipt = successor.seal_receipt(
            {
                "schema": successor.STAGE_RECEIPT_SCHEMA,
                "candidate_id": successor.CANDIDATE_ID,
                "owner_decision_revision": successor.OWNER_DECISION_REVISION,
                "manifest_sha256": manifest.identity.sha256,
                "stage_id": stage.value,
                "status": "passed",
                "source_sha256s": {"implementation": _sha(f"{index + 1:x}")},
                "test_command_or_fixture_identity": f"pytest {stage.value}",
                "test_result": "passed",
                "public_information_audit": True,
                "direct_policy_audit": True,
                "r241_nonmutation_audit": True,
                "prior_stage_receipt_sha256s": dict(prior),
            }
        )
        rows[stage] = receipt
        prior[stage.value] = str(receipt["receipt_sha256"])
    return rows


def _migration(
    manifest: successor.OwnDeckSuccessorManifest,
    refresh: dict[str, object],
    stages: dict[successor.OwnDeckSuccessorStage, dict[str, object]],
) -> dict[str, object]:
    parsed_refresh = successor.validate_refresh_completion_receipt(
        refresh, manifest=manifest
    )
    parsed_stages = successor.validate_prior_stage_receipts(stages, manifest=manifest)
    return successor.seal_receipt(
        {
            "schema": successor.POST_REFRESH_RECEIPT_SCHEMA,
            "kind": successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION.value,
            "status": "passed",
            "candidate_id": successor.CANDIDATE_ID,
            "owner_decision_revision": successor.OWNER_DECISION_REVISION,
            "manifest_sha256": manifest.identity.sha256,
            "refresh_completion_receipt_sha256": parsed_refresh.sha256,
            "prior_stage_receipt_sha256s": {
                stage.value: receipt.sha256 for stage, receipt in parsed_stages.items()
            },
            "depends_on_receipt_sha256s": {},
            "migration_schema": contract.MIGRATION_RECEIPT_SCHEMA,
            "parent_checkpoint": {
                "path": "/immutable/terminal-refresh.pt",
                "sha256": parsed_refresh.checkpoint_sha256,
            },
            "child_checkpoint": {
                "path": "/immutable/own-deck-successor.pt",
                "sha256": _sha("b"),
            },
            "runtime_authority": {
                "own_deck_ledger_runtime_enabled": False,
                "visible_tutor_completion_route_runtime_enabled": False,
                "terminal_conversion_route_runtime_enabled": False,
                "selector_change_authorized": False,
                "package_or_submission_authorized": False,
                "serving_eligible": False,
            },
            "verification": {
                "parent_checkpoint_sha256": parsed_refresh.checkpoint_sha256,
                "child_checkpoint_sha256": _sha("b"),
                "inherited_tensor_count": 100,
                "added_tensor_keys": [
                    "own_deck_ledger_adapter.output.weight",
                    "visible_tutor_completion_head.weight",
                    "terminal_conversion_head.weight",
                ],
            },
        }
    )


def _label_counts(seed: int = 0) -> dict[str, object]:
    # Every value is an observed selected-action fact count.  The counts vary
    # slightly by day so aggregate/digest order mistakes are visible.
    return {
        "terminal_conversion": {
            "terminal_class": {
                "nonterminal": 20 + seed,
                "own_win": 2 + (seed % 2),
                "own_loss": 1,
                "draw": 1,
            },
            "terminal_class_labeled": 24 + seed + (seed % 2),
            "prize_closeout": {"labeled": 22 + (seed % 2), "positive": 2 + (seed % 2)},
            "opponent_knockout": {"labeled": 22 + seed, "positive": 4},
        },
        "visible_tutor_completion": {
            "visible_tutor_stages": 5 + seed,
            "selected_from_visible_deck": {"labeled": 5 + seed, "positive": 5 + seed},
            "selected_target_observed_after_action": {"labeled": 5 + seed, "positive": 4},
            "same_actor_followup": {"labeled": 5 + seed, "positive": 3},
            "same_actor_terminal_class": {
                "nonterminal": 5 + seed,
                "own_win": 1,
                "own_loss": 0,
                "draw": 0,
            },
            "same_actor_terminal_class_labeled": 6 + seed,
        },
        "ledger": {"integrity_ok": 100 + seed, "fail_closed": 0},
        "joinable_policy_stage_count": 100 + seed,
    }


def _daily(
    manifest: successor.OwnDeckSuccessorManifest,
    *,
    build_mode: str = "archive_native",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, source_day in enumerate(
        contract.expected_sidecar_days(manifest.elmo_side_store)
    ):
        shard_sha = _sha(f"{(index % 15) + 1:x}")
        payload: dict[str, object] = {
            "schema": contract.SIDE_STORE_DAILY_META_SCHEMA,
            "version": contract.SIDE_STORE_DAILY_META_VERSION,
            "owner_decision_revision": 259,
            "status": "complete_immutable_sidecar",
            "day": source_day,
            "shard": {
                "path": "own_deck_rollouts.jsonl.gz",
                "sha256": shard_sha,
                "bytes": 1_000 + index,
                "compression": "gzip",
                "format": "jsonl",
                "row_schema": "poke_bot.own_deck_rollout_sidecar/v1",
                "row_version": 1,
            },
            "shard_sha256": shard_sha,
            "rows_sha256": _sha(f"{(index % 14) + 1:x}"),
            "row_count": 100 + index,
            "source_record_count": 100 + index,
            "source_records_sha256": _sha(f"{(index % 13) + 1:x}"),
            "source": {
                "manifest": {
                    "original_path": manifest.elmo_side_store.source_manifest,
                    "locked_path": "/locked/current.json",
                    "sha256": manifest.elmo_side_store.source_manifest_sha256,
                    "schema": "poke_bot.expert_latest20_receipt/v1",
                    "window_start": manifest.elmo_side_store.source_window.start_date,
                    "window_end": manifest.elmo_side_store.source_window.end_date,
                    "days": manifest.elmo_side_store.source_window.day_count,
                    "total_episodes": manifest.elmo_side_store.source_window.validated_episode_count,
                },
                "versioned_receipt": {
                    "path": "/source/versioned.json",
                    "original_path": "/source/versioned.json",
                    "locked_path": "/locked/versioned.json",
                    "sha256": contract.EXPECTED_VERSIONED_RECEIPT_SHA256,
                },
                "archive": {
                    "date": source_day,
                    "path": f"/archive/{source_day}.zip",
                    "sha256": _sha(f"{(index % 12) + 1:x}"),
                    "bytes": 2_000 + index,
                    "validated_episode_count": 100 + index,
                    "source_slug": f"pokemon-tcg-ai-battle-episodes-{source_day}",
                },
            },
            "build": {
                "mode": build_mode,
                "source_snapshot": {
                    "path": manifest.elmo_side_store.source_snapshot_root,
                    "tree_sha256": _sha("9"),
                },
                "image": {
                    "tag": manifest.elmo_side_store.container_image,
                    "id": manifest.elmo_side_store.container_image_id,
                },
                "code": {
                    "own_deck_rollout_store.py": _sha("a"),
                    "own_deck_ledger.py": _sha("b"),
                    "own_deck_supervision.py": _sha("c"),
                },
                "classifier": {
                    "contract": {
                        "schema": "poke_bot.test_archive_native_classifier/v1",
                        "version": 1,
                    },
                    "mix": {"path": "/input/top_ladder.mix.json", "sha256": _sha("d")},
                    "representatives": {
                        "path": "/input/top_ladder.representatives.json",
                        "sha256": _sha("e"),
                    },
                    "card_csv": {
                        "path": "/workspace/cards/EN_Card_Data.csv",
                        "sha256": _sha("f"),
                    },
                },
                "protected_stream_sha256": None,
            },
            "label_counts": _label_counts(index),
            "training_eligibility": {
                "active_r241": False,
                "sidecar_only": True,
                "successor": "pending_refresh_join_parity_receipt",
            },
        }
        payload["meta_sha256"] = contract.daily_meta_digest(payload)
        rows.append(payload)
    return rows


def _receipts(
    manifest: successor.OwnDeckSuccessorManifest,
    daily_raw: list[dict[str, object]],
    *,
    model_sha: str,
    code_sha: str,
    migration_sha: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    daily = contract.validate_daily_sidecar_receipts(daily_raw, manifest=manifest)
    labels = contract.aggregate_label_counts(daily)
    dataset_sha = contract.sidecar_dataset_sha256(daily)
    daily_map = {item.source_day: item.meta_sha256 for item in daily}
    shard_map = {item.source_day: item.shard_sha256 for item in daily}
    records = sum(item.record_count for item in daily)
    provenance = {
        "schema": contract.SIDE_STORE_JOIN_PROVENANCE_SCHEMA,
        "source_manifest_sha256": manifest.elmo_side_store.source_manifest_sha256,
        "daily_meta_sha256s": daily_map,
        "sidecar_meta_identity": contract.sidecar_join_meta_identity(
            source_manifest_sha256=manifest.elmo_side_store.source_manifest_sha256,
            daily_meta_sha256s=daily_map,
        ),
        "record_key": list(manifest.elmo_side_store.record_key),
        "sidecar_record_count": records,
        "joined_decision_count": records,
        "unmatched_record_count": 0,
        "duplicate_key_count": 0,
        "raw_reconstruction_parity_count": records,
        "one_to_one_coverage": True,
        "canonical_record_key_coverage": True,
        "observation_fingerprint_parity_count": records,
        "active_r241_training_eligible": False,
    }
    join_provenance_sha = contract.sha256_bytes(
        contract.canonical_json_bytes(provenance)
    )
    join = contract.seal_receipt(
        {
            "schema": contract.SIDE_STORE_JOIN_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest.identity.sha256,
            "source_manifest_sha256": manifest.elmo_side_store.source_manifest_sha256,
            "code_sha256": code_sha,
            "training_code_sha256": code_sha,
            "sidecar_dataset_sha256": dataset_sha,
            "model_sha256": model_sha,
            "migration_receipt_sha256": migration_sha,
            "join_provenance_schema": contract.SIDE_STORE_JOIN_PROVENANCE_SCHEMA,
            "join_provenance_sha256": join_provenance_sha,
            **{
                key: value
                for key, value in provenance.items()
                if key != "schema"
            },
        }
    )
    common = {
        "manifest_sha256": manifest.identity.sha256,
        "source_manifest_sha256": manifest.elmo_side_store.source_manifest_sha256,
        "training_code_sha256": code_sha,
        "sidecar_build_code_sha256": daily[0].sidecar_build_code_sha256,
        "migration_receipt_sha256": migration_sha,
        "sidecar_dataset_sha256": dataset_sha,
        "daily_meta_sha256s": daily_map,
        "daily_shard_sha256s": shard_map,
    }
    parity = contract.seal_receipt(
        {
            "schema": contract.SIDE_STORE_PARITY_RECEIPT_SCHEMA,
            "status": "passed",
            **common,
            "model_sha256": model_sha,
            "join_receipt_sha256": join["receipt_sha256"],
            "join_provenance_sha256": join_provenance_sha,
            "exact_key_join_parity": True,
            "local_remote_parity": True,
            "ledger_parity": True,
            "supervision_parity": True,
            "public_information_only": True,
            "direct_policy_only": True,
        }
    )
    terminal_scalars = dict(labels.terminal_scalars)
    tutor_scalars = dict(labels.tutor_scalars)
    support = {
        "visible_tutor_observed_menu_expert_top1_denominator": tutor_scalars["selected_from_visible_deck"].positive,
        "terminal_multiclass_brier_ece_denominator": sum(labels.terminal_class_counts),
        "selected_option_factual_recall_own_win_denominator": labels.terminal_class_counts[1],
        "selected_option_factual_recall_prize_closeout_denominator": terminal_scalars["prize_closeout"].positive,
        "selected_option_factual_recall_opponent_knockout_denominator": terminal_scalars["opponent_knockout"].positive,
        "missed_observed_expert_closeout_denominator": labels.terminal_class_counts[1],
    }
    metric = contract.seal_receipt(
        {
            "schema": contract.METRIC_SUPPORT_RECEIPT_SCHEMA,
            "status": "passed",
            **common,
            "model_sha256": model_sha,
            "join_receipt_sha256": join["receipt_sha256"],
            "join_provenance_sha256": join_provenance_sha,
            "parity_receipt_sha256": parity["receipt_sha256"],
            "metric_schema": "poke_bot.own_deck_promotion_metrics/v1",
            "missed_expert_closeout_basis": (
                "policy_top1_vs_observed_expert_selected_option"
            ),
            "observed_selected_action_labels_only": True,
            "counterfactual_legal_action_labels_absent": True,
            "hidden_deck_or_prize_labels_absent": True,
            "metric_support": support,
        }
    )
    return join, parity, metric


def _prepared_plan() -> dict[str, object]:
    manifest = _manifest()
    refresh = _refresh(manifest)
    stages = _stages(manifest)
    migration = _migration(manifest, refresh, stages)
    code_sha = _sha("a")
    model_sha = _sha("b")
    daily = _daily(manifest)
    join, parity, metric = _receipts(
        manifest,
        daily,
        model_sha=model_sha,
        code_sha=code_sha,
        migration_sha=str(migration["receipt_sha256"]),
    )
    return contract.prepare_next_train_plan(
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        migration_receipt=migration,
        daily_meta_receipts=daily,
        join_receipt=join,
        parity_receipt=parity,
        metric_support_receipt=metric,
        source_manifest_identity={
            "id": "r241-exact20",
            "path": manifest.elmo_side_store.source_manifest,
            "sha256": manifest.elmo_side_store.source_manifest_sha256,
        },
        model_identity={"id": "migrated-successor", "sha256": model_sha},
        code_identity={"id": "r258-source", "sha256": code_sha},
    )


def test_weights_are_deterministic_bounded_and_neutral_for_zero_support() -> None:
    counts = contract.SupervisionLabelCounts(
        terminal_class_counts=(100, 10, 0, 1),
        terminal_scalars=(
            ("prize_closeout", contract.BinaryCounts(2, 200)),
            ("opponent_knockout", contract.BinaryCounts(0, 22)),
        ),
        tutor_terminal_class_counts=(20, 0, 0, 0),
        tutor_scalars=(
            ("selected_from_visible_deck", contract.BinaryCounts(10, 0)),
            ("selected_target_observed_after_action", contract.BinaryCounts(2, 20)),
            ("same_actor_followup", contract.BinaryCounts(3, 9)),
        ),
    )
    first = contract.derive_supervision_weights(counts)
    second = contract.derive_supervision_weights(counts)

    assert first == second
    assert first.terminal_conversion_class_weights[2] == 1.0
    assert first.terminal_conversion_positive_weight <= contract.TRAIN_MAX_REWEIGHT
    assert first.visible_tutor_completion_positive_weight > 0.0
    assert first.evidence["counterfactual_or_imputed_rows_used"] is False


def test_prepared_plan_binds_every_identity_and_keeps_runtime_inert() -> None:
    plan = _prepared_plan()
    validated = contract.validate_next_train_plan(plan)

    assert validated["gate"]["operation"] == "training_canary"
    assert validated["model_config"]["own_deck_ledger_enabled"] is True
    assert validated["model_config"]["visible_tutor_completion_head_enabled"] is True
    assert validated["model_config"]["terminal_conversion_route_enabled"] is True
    assert validated["runtime_gates"] == {
        "own_deck_ledger_runtime_enabled": False,
        "visible_tutor_completion_route_runtime_enabled": False,
        "terminal_conversion_route_runtime_enabled": False,
        "runtime_action_authority": False,
    }
    assert all(value is False for value in validated["authority"].values())
    assert validated["identities"]["sidecar_dataset"]["record_count"] > 0
    assert validated["identities"]["join_receipt_sha256"].startswith("sha256:")
    assert validated["identities"]["join_provenance_sha256"].startswith("sha256:")
    assert validated["receipt"]["training_execution_started"] is False
    assert validated["train_config"]["visible_tutor_completion_loss_weight"] > 0.0
    assert validated["train_config"]["terminal_conversion_loss_weight"] > 0.0


def test_plan_rejects_missing_day_or_bad_metric_support() -> None:
    manifest = _manifest()
    refresh = _refresh(manifest)
    stages = _stages(manifest)
    migration = _migration(manifest, refresh, stages)
    code_sha = _sha("a")
    model_sha = _sha("b")
    daily = _daily(manifest)
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="incomplete"):
        contract.validate_daily_sidecar_receipts(daily[:-1], manifest=manifest)

    join, parity, metric = _receipts(
        manifest,
        daily,
        model_sha=model_sha,
        code_sha=code_sha,
        migration_sha=str(migration["receipt_sha256"]),
    )
    bad_metric = copy.deepcopy(metric)
    bad_metric["metric_support"][
        "selected_option_factual_recall_own_win_denominator"
    ] += 1
    bad_metric = contract.seal_receipt(bad_metric)
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="metric support"):
        contract.prepare_next_train_plan(
            refresh_completion_receipt=refresh,
            stage_receipts=stages,
            migration_receipt=migration,
            daily_meta_receipts=daily,
            join_receipt=join,
            parity_receipt=parity,
            metric_support_receipt=bad_metric,
            source_manifest_identity={
                "id": "r241-exact20",
                "path": manifest.elmo_side_store.source_manifest,
                "sha256": manifest.elmo_side_store.source_manifest_sha256,
            },
            model_identity={"id": "migrated-successor", "sha256": model_sha},
            code_identity={"id": "r258-source", "sha256": code_sha},
        )


def test_daily_meta_rejects_protected_jsonl_code_drift_and_incomplete_join() -> None:
    manifest = _manifest()
    protected = _daily(manifest, build_mode="protected_jsonl")
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="archive-native"):
        contract.validate_daily_sidecar_receipts(protected, manifest=manifest)

    daily = _daily(manifest)
    parsed = contract.validate_daily_sidecar_receipts(daily, manifest=manifest)
    # This is the replacement for the old caller-provided
    # ``expected_code_sha256`` API: r259 build-code identity is derived from
    # each sealed daily meta, then all 20 exact identities must agree.
    code_drift = copy.deepcopy(daily)
    code_drift[1]["build"]["code"]["own_deck_ledger.py"] = _sha("d")
    code_drift[1]["meta_sha256"] = contract.daily_meta_digest(code_drift[1])
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="build-code"):
        contract.validate_daily_sidecar_receipts(code_drift, manifest=manifest)

    refresh = _refresh(manifest)
    stages = _stages(manifest)
    migration = _migration(manifest, refresh, stages)
    code_sha = _sha("a")
    model_sha = _sha("b")
    join, _, _ = _receipts(
        manifest,
        daily,
        model_sha=model_sha,
        code_sha=code_sha,
        migration_sha=str(migration["receipt_sha256"]),
    )
    incomplete_join = copy.deepcopy(join)
    incomplete_join.pop("record_key")
    incomplete_join = contract.seal_receipt(incomplete_join)
    join_validation_args = {
        "manifest": manifest,
        "source": contract.ContentIdentity(
            role="source",
            identity="r241-exact20",
            sha256=manifest.elmo_side_store.source_manifest_sha256,
            path=manifest.elmo_side_store.source_manifest,
        ),
        "model": contract.ContentIdentity(
            role="model", identity="migrated-successor", sha256=model_sha
        ),
        "code": contract.ContentIdentity(
            role="code", identity="r258-source", sha256=code_sha
        ),
        "daily": parsed,
        "dataset_sha256": contract.sidecar_dataset_sha256(parsed),
        "migration_receipt_sha256": str(migration["receipt_sha256"]),
    }
    unsealed_join = copy.deepcopy(join)
    unsealed_join.pop("receipt_sha256")
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="digest"):
        contract._validate_join_receipt(unsealed_join, **join_validation_args)
    with pytest.raises(contract.OwnDeckNextTrainContractError, match="canonical record key"):
        contract._validate_join_receipt(incomplete_join, **join_validation_args)


def test_atomic_plan_publish_never_replaces_existing_receipt(tmp_path: Path) -> None:
    plan = _prepared_plan()
    output = tmp_path / "next-train-plan.json"
    assert contract.write_next_train_plan(output, plan) == output
    assert contract.validate_next_train_plan(output)["plan_sha256"] == plan["plan_sha256"]
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        contract.write_next_train_plan(output, plan)
