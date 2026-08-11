"""Focused fail-closed tests for the dormant r258 OwnDeckLedger guard."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from poke_bot import own_deck_successor as successor

ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> successor.OwnDeckSuccessorManifest:
    return successor.load_canonical_manifest()


def _refresh_receipt(
    manifest: successor.OwnDeckSuccessorManifest,
) -> dict[str, object]:
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
            "immutable_refresh_lineage": {
                "id": "alakazam-r241-terminal-refresh",
                "sha256": "sha256:" + "1" * 64,
            },
            "completed_refresh_boundary": {
                "id": "iter_00009-terminal",
                "sha256": "sha256:" + "2" * 64,
            },
            "checkpoint": {"id": "expert_before_iter_00010.pt", "sha256": "sha256:" + "3" * 64},
            "source_receipt_chain": {
                "integrity_verified": True,
                "sha256": "sha256:" + "4" * 64,
            },
            "runtime_receipt_chain": {
                "integrity_verified": True,
                "sha256": "sha256:" + "5" * 64,
            },
        }
    )


def _stage_receipts(
    manifest: successor.OwnDeckSuccessorManifest,
) -> dict[successor.OwnDeckSuccessorStage, dict[str, object]]:
    receipts: dict[successor.OwnDeckSuccessorStage, dict[str, object]] = {}
    prior: dict[str, str] = {}
    for index, stage in enumerate(manifest.pre_refresh_stages):
        receipt = successor.seal_receipt(
            {
                "schema": successor.STAGE_RECEIPT_SCHEMA,
                "candidate_id": successor.CANDIDATE_ID,
                "owner_decision_revision": successor.OWNER_DECISION_REVISION,
                "manifest_sha256": manifest.identity.sha256,
                "stage_id": stage.value,
                "status": successor.OwnDeckSuccessorStageStatus.PASSED.value,
                "source_sha256s": {"implementation": "sha256:" + f"{index + 1:x}" * 64},
                "test_command_or_fixture_identity": f"pytest tests/test_{stage.value}.py",
                "test_result": "passed",
                "public_information_audit": True,
                "direct_policy_audit": True,
                "r241_nonmutation_audit": True,
                "prior_stage_receipt_sha256s": dict(prior),
            }
        )
        receipts[stage] = receipt
        prior[stage.value] = str(receipt["receipt_sha256"])
    return receipts


def _post_refresh_receipts(
    manifest: successor.OwnDeckSuccessorManifest,
    refresh: dict[str, object],
    stages: dict[successor.OwnDeckSuccessorStage, dict[str, object]],
) -> dict[successor.OwnDeckSuccessorPostRefreshReceiptKind, dict[str, object]]:
    all_stages = successor.validate_prior_stage_receipts(stages, manifest=manifest)
    stage_digests = {stage.value: parsed.sha256 for stage, parsed in all_stages.items()}
    refresh_digest = successor.validate_refresh_completion_receipt(
        refresh, manifest=manifest
    ).sha256
    receipts: dict[
        successor.OwnDeckSuccessorPostRefreshReceiptKind, dict[str, object]
    ] = {}
    prior: dict[str, str] = {}
    for kind in successor.OwnDeckSuccessorPostRefreshReceiptKind:
        receipt = successor.seal_receipt(
            {
                "schema": successor.POST_REFRESH_RECEIPT_SCHEMA,
                "kind": kind.value,
                "status": successor.OwnDeckSuccessorStageStatus.PASSED.value,
                "candidate_id": successor.CANDIDATE_ID,
                "owner_decision_revision": successor.OWNER_DECISION_REVISION,
                "manifest_sha256": manifest.identity.sha256,
                "refresh_completion_receipt_sha256": refresh_digest,
                "prior_stage_receipt_sha256s": stage_digests,
                "depends_on_receipt_sha256s": dict(prior),
            }
        )
        receipts[kind] = receipt
        prior[kind.value] = str(receipt["receipt_sha256"])
    return receipts


def test_canonical_manifest_is_typed_and_locked_to_r258() -> None:
    manifest = _manifest()

    assert manifest.identity.path == ROOT / "state/alakazam-own-deck-ledger-successor-r258.json"
    assert manifest.raw["schema"] == successor.MANIFEST_SCHEMA
    assert manifest.stages == tuple(successor.OwnDeckSuccessorStage)
    assert manifest.pre_refresh_stages == tuple(successor.OwnDeckSuccessorStage)[:-1]
    assert manifest.raw["latest_owner_clarification_revision"] == 259
    assert manifest.raw["authority"]["isolated_successor_build_and_offline_test_now"] is True
    assert (
        manifest.raw["authority"]["existing_managed_service_start_stop_restart_or_reconfigure"]
        is False
    )
    assert manifest.raw["authority"]["new_elmo_side_store_managed_service_start"] is True
    assert manifest.raw["authority"]["elmo_side_store_may_feed_active_r241"] is False
    assert (
        manifest.elmo_side_store.status
        is successor.ElmoOwnDeckSideStoreStatus.AUTHORIZED_FOR_BUILD_AND_MANAGED_START
    )
    assert manifest.elmo_side_store.managed_service == successor.ELMO_SIDE_STORE_MANAGED_SERVICE
    assert manifest.elmo_side_store.container_image_id == successor.ELMO_SIDE_STORE_CONTAINER_IMAGE_ID
    assert (
        manifest.elmo_side_store.sealed_runtime_snapshot_root
        == successor.ELMO_SIDE_STORE_SEALED_RUNTIME_SNAPSHOT_ROOT
    )
    assert (
        manifest.elmo_side_store.sealed_runtime_source_lock
        == successor.ELMO_SIDE_STORE_SEALED_RUNTIME_SOURCE_LOCK
    )
    assert (
        manifest.elmo_side_store.controller_expected_inventory
        == successor.ELMO_SIDE_STORE_CONTROLLER_EXPECTED_INVENTORY
    )
    assert manifest.elmo_side_store.source_window.validated_episode_count == 91_253
    assert manifest.elmo_side_store.record_key == successor.ELMO_SIDE_STORE_RECORD_KEY
    assert manifest.raw["authority"]["selector_change"] is False


def test_noncanonical_manifest_path_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "r258-copy.json"
    copied.write_bytes((ROOT / "state/alakazam-own-deck-ledger-successor-r258.json").read_bytes())

    with pytest.raises(successor.OwnDeckSuccessorError, match="canonical path"):
        successor.load_canonical_manifest(copied)


def test_successor_json_reader_rejects_same_valued_duplicate_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-r258.json"
    duplicate.write_text(
        '{"source_manifest_sha256":"same","source_manifest_sha256":"same"}',
        encoding="utf-8",
    )

    with pytest.raises(successor.OwnDeckSuccessorError, match="not readable JSON"):
        successor._read_json_object(duplicate, label="duplicate successor manifest")


def test_isolated_build_and_tests_are_allowed_without_refresh_receipt() -> None:
    build = successor.evaluate_successor_gate("build")
    offline_test = successor.evaluate_successor_gate("offline_test")

    assert build.allowed is True
    assert offline_test.allowed is True
    assert "isolated" in build.reason
    assert build.refresh_completion_sha256 is None


def test_r259_elmo_side_store_materialization_is_narrowly_allowed_now() -> None:
    decision = successor.evaluate_successor_gate("elmo_side_store_materialization")

    assert decision.allowed is True
    assert successor.ELMO_SIDE_STORE_MANAGED_SERVICE in decision.reason
    assert "active-r241" in decision.reason
    assert decision.refresh_completion_sha256 is None

    # The r259 exception does not relax any successor-training gate.
    training = successor.evaluate_successor_gate("training")
    assert training.allowed is False
    assert "terminal Alakazam-refresh completion receipt" in training.reason


def test_r259_elmo_side_store_contract_rejects_substitution() -> None:
    payload = copy.deepcopy(_manifest().raw)
    payload["elmo_expert_rollout_side_store"]["output_root"] = "/tmp/not-authorized"

    with pytest.raises(successor.OwnDeckSuccessorError, match="output root"):
        successor.validate_canonical_manifest(payload)

    payload = copy.deepcopy(_manifest().raw)
    payload["elmo_expert_rollout_side_store"]["container_image_id"] = "sha256:" + "0" * 64
    with pytest.raises(successor.OwnDeckSuccessorError, match="container image digest"):
        successor.validate_canonical_manifest(payload)

    payload = copy.deepcopy(_manifest().raw)
    payload["elmo_expert_rollout_side_store"]["container_image"] = (
        "poke-bot-truenas-worker:substituted"
    )
    with pytest.raises(successor.OwnDeckSuccessorError, match="container image"):
        successor.validate_canonical_manifest(payload)

    payload = copy.deepcopy(_manifest().raw)
    payload["authority"]["new_elmo_side_store_managed_service_start"] = False
    with pytest.raises(successor.OwnDeckSuccessorError, match="new_elmo_side_store"):
        successor.validate_canonical_manifest(payload)

    payload = copy.deepcopy(_manifest().raw)
    side_store_stage = next(
        stage
        for stage in payload["stages"]
        if stage["id"] == successor.OwnDeckSuccessorStage.ELMO_EXPERT_ROLLOUT_SIDE_STORE.value
    )
    side_store_stage["status"] = successor.OwnDeckSuccessorStageStatus.STAGED_DORMANT.value
    with pytest.raises(successor.OwnDeckSuccessorError, match="elmo_expert_rollout_side_store status"):
        successor.validate_canonical_manifest(payload)

    # A forged in-memory projection cannot replace the loaded canonical contract.
    manifest = _manifest()
    forged = replace(
        manifest,
        elmo_side_store=replace(manifest.elmo_side_store, managed_service="not-authorized"),
    )
    denied = successor.evaluate_successor_gate(
        "elmo_side_store_materialization", manifest=forged
    )
    assert denied.allowed is False
    assert "does not match" in denied.reason


@pytest.mark.parametrize("operation", ("migration", "training", "runtime", "promotion"))
def test_active_successor_operations_fail_closed_without_refresh_receipt(
    operation: str,
) -> None:
    decision = successor.evaluate_successor_gate(operation)

    assert decision.allowed is False
    assert "terminal Alakazam-refresh completion receipt" in decision.reason


def test_refresh_receipt_requires_terminal_lineage_boundary_checkpoint_and_chain() -> None:
    manifest = _manifest()
    receipt = _refresh_receipt(manifest)

    validated = successor.validate_refresh_completion_receipt(receipt, manifest=manifest)
    assert validated.checkpoint_sha256 == "sha256:" + "3" * 64

    missing_chain = copy.deepcopy(receipt)
    missing_chain.pop("runtime_receipt_chain")
    missing_chain = successor.seal_receipt(missing_chain)
    with pytest.raises(successor.OwnDeckSuccessorError, match="runtime receipt chain"):
        successor.validate_refresh_completion_receipt(missing_chain, manifest=manifest)

    bad_digest = copy.deepcopy(receipt)
    bad_digest["receipt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(successor.OwnDeckSuccessorError, match="fingerprint"):
        successor.validate_refresh_completion_receipt(bad_digest, manifest=manifest)


def test_migration_requires_every_prior_stage_and_a_digest_chain() -> None:
    manifest = _manifest()
    refresh = _refresh_receipt(manifest)
    stages = _stage_receipts(manifest)

    allowed = successor.evaluate_successor_gate(
        "migration",
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
    )
    assert allowed.allowed is True
    assert allowed.accepted_stage_receipts == manifest.pre_refresh_stages

    incomplete = dict(stages)
    incomplete.pop(successor.OwnDeckSuccessorStage.ELMO_EXPERT_ROLLOUT_SIDE_STORE)
    denied = successor.evaluate_successor_gate(
        "migration",
        refresh_completion_receipt=refresh,
        stage_receipts=incomplete,
    )
    assert denied.allowed is False
    assert "incomplete" in denied.reason

    broken = copy.deepcopy(stages)
    tutor = successor.OwnDeckSuccessorStage.VISIBLE_TUTOR_LEARNING
    broken[tutor]["prior_stage_receipt_sha256s"] = {}
    broken[tutor] = successor.seal_receipt(broken[tutor])
    denied = successor.evaluate_successor_gate(
        "migration",
        refresh_completion_receipt=refresh,
        stage_receipts=broken,
    )
    assert denied.allowed is False
    assert "prior-stage receipt chain" in denied.reason


def test_post_refresh_chain_is_strictly_ordered_and_activation_is_distinct() -> None:
    manifest = _manifest()
    refresh = _refresh_receipt(manifest)
    stages = _stage_receipts(manifest)
    post = _post_refresh_receipts(manifest, refresh, stages)

    training = successor.evaluate_successor_gate(
        "training",
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        post_refresh_receipts={
            successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION: post[
                successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION
            ]
        },
    )
    assert training.allowed is True

    runtime_without_activation = successor.evaluate_successor_gate(
        "runtime",
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        post_refresh_receipts={
            successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION: post[
                successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION
            ],
            successor.OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY: post[
                successor.OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY
            ],
            successor.OwnDeckSuccessorPostRefreshReceiptKind.SOURCE_DISJOINT_EVALUATION: post[
                successor.OwnDeckSuccessorPostRefreshReceiptKind.SOURCE_DISJOINT_EVALUATION
            ],
        },
    )
    assert runtime_without_activation.allowed is False
    assert "runtime_activation" in runtime_without_activation.reason

    runtime = successor.evaluate_successor_gate(
        "runtime",
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        post_refresh_receipts=post,
    )
    assert runtime.allowed is True

    promotion = successor.evaluate_successor_gate(
        "promotion",
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        post_refresh_receipts=post,
    )
    assert promotion.allowed is True


def test_selector_package_and_submission_remain_denied_even_with_all_evidence() -> None:
    manifest = _manifest()
    refresh = _refresh_receipt(manifest)
    stages = _stage_receipts(manifest)
    post = _post_refresh_receipts(manifest, refresh, stages)

    for operation in ("selector", "package", "submission"):
        decision = successor.evaluate_successor_gate(
            operation,
            refresh_completion_receipt=refresh,
            stage_receipts=stages,
            post_refresh_receipts=post,
        )
        assert decision.allowed is False
        assert "remain independently denied" in decision.reason


def test_require_gate_raises_instead_of_best_effort_activation() -> None:
    with pytest.raises(successor.OwnDeckSuccessorGateError, match="terminal Alakazam-refresh"):
        successor.require_successor_operation("runtime")
