"""Unit coverage for the revision-5 aggregate/migration fail-closed boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from poke_bot import alakazam_rule_derivative_handoff_r303 as handoff
from poke_bot import alakazam_rule_derivative_r303_aggregate_validation as aggregate
from poke_bot.alakazam_collision_census_r298 import revision_5_predecessor_classification


_DIGEST = "sha256:" + "a" * 64


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": aggregate.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _migration_payload(*, source_digest: str = _DIGEST) -> dict[str, Any]:
    return {
        "schema": aggregate.R303_CONSUMER_MIGRATION_SCHEMA,
        "status": "passed_revision_5_rebind_nonactivating",
        "goal_gateway_path": aggregate.R303_GOAL_PATH,
        "goal_gateway_sha256": aggregate.R303_GOAL_GATEWAY_SHA256,
        "goal_contract_path": aggregate.R303_CONTRACT_PATH,
        "goal_contract_sha256": aggregate.R303_CONTRACT_SHA256,
        "r241_typed_source_path": aggregate.R303_R241_TYPED_SOURCE_PATH,
        "r241_typed_source_sha256": aggregate.R303_R241_TYPED_SOURCE_SHA256,
        "goal_revision": aggregate.R303_GOAL_REVISION,
        "root_handoff_revision": aggregate.R303_ROOT_OWNER_REVISION,
        "predecessor_evidence": revision_5_predecessor_classification(),
        "predecessor_receipt_files": [],
        "consumers": [
            {
                "consumer_id": consumer_id,
                "source_path": source_path,
                "source_sha256": source_digest,
                "source_size_bytes": 1,
                "revision_5_authority_bound": True,
                "revision_4_receipt_is_historical_only": True,
                "zero_inert_and_layer_off_fail_closed": True,
                "runtime_wired": False,
                "test_evidence": {
                    "path": "tests.json",
                    "sha256": _DIGEST,
                    "size_bytes": 1,
                },
            }
            for consumer_id, source_path in aggregate.MIGRATION_CONSUMER_SOURCES.items()
        ],
        "test_assertions": {
            name: True for name in aggregate.MIGRATION_TEST_ASSERTIONS
        },
        "runtime_or_service_action_performed": False,
        "candidate_action_time_wrapper_sealed": False,
        "training_handoff_authorized": False,
        "kaggle_queue_or_fleet_authorized": False,
        "recorded_at_utc": "2026-08-12T20:00:00Z",
    }


def _candidate_receipt() -> dict[str, Any]:
    contract = handoff.load_r303_contract()
    baseline = contract["baselines_and_candidate"]
    fields = baseline["candidate_validation_receipt_required_fields"]
    row: dict[str, Any] = {field: "evidence" for field in fields}
    row.update(
        {
            "schema": baseline["candidate_validation_receipt_schema"],
            "goal_contract_path": aggregate.R303_CONTRACT_PATH,
            "goal_contract_sha256": aggregate.R303_CONTRACT_SHA256,
            "goal_revision": aggregate.R303_GOAL_REVISION,
            "root_owner_revision": aggregate.R303_ROOT_OWNER_REVISION,
            "candidate_id": baseline["derivative_candidate"]["candidate_id"],
            "baseline_r274_candidate_id": baseline["r241_r274_same_architecture_baseline"][
                "candidate_id"
            ],
            "exact_new_list_canonical_multiset_sha256": baseline[
                "r241_r274_same_architecture_baseline"
            ]["exact_new_list_canonical_multiset_sha256"],
            "candidate_checkpoint_size_bytes": 1,
            "baseline_r274_checkpoint_size_bytes": 1,
            "candidate_checkpoint_model_schema": "poke_bot.test_model/v1",
            "bootstrap_rows_steps_and_epochs": {"rows": 1, "steps": 1, "epochs": 1},
            "bootstrap_loss_gradient_and_resource_peaks": {"finite": True},
            "trainable_parameters_changed_and_finite": True,
            "frozen_backbone_true": True,
            "layer_off_bit_identical_baseline_logits": True,
            "layer_off_identical_legal_choice": True,
            "all_frozen_tensor_bit_identity": True,
            "public_information_metamorphic_tests_passed": True,
            "simulator_engine_tests_passed": True,
            "bootstrap_validation_passed": True,
            "production_serving_selector_authority": False,
        }
    )
    for field in fields:
        if field.endswith("_sha256") and field not in {
            "goal_contract_sha256",
            "exact_new_list_canonical_multiset_sha256",
        }:
            row[field] = _DIGEST
    return row


def _fleet_inventory() -> dict[str, Any]:
    contract = handoff.load_r303_contract()
    fleet = contract["full_available_fleet_self_play"]
    spec = fleet["receipt_contract"]["fleet_inventory"]
    hosts = fleet["known_candidate_hosts"]
    row: dict[str, Any] = {field: "evidence" for field in spec["required_fields"]}
    row.update(
        {
            "schema": spec["schema"],
            "goal_contract_path": aggregate.R303_CONTRACT_PATH,
            "goal_contract_sha256": aggregate.R303_CONTRACT_SHA256,
            "goal_revision": aggregate.R303_GOAL_REVISION,
            "inventory_at_utc": "2026-08-12T20:00:00Z",
            "known_candidate_hosts": hosts,
            "full_available_fleet_included": True,
        }
    )
    for field in spec["required_fields"]:
        if field.endswith("_sha256") and field != "goal_contract_sha256":
            row[field] = _DIGEST
    for field in (
        "per_host_availability_eligibility_and_exclusion_evidence",
        "per_host_os_arch_cpu_ram_and_network_identity",
        "per_host_gpu_device_models_uuids_memory_and_runtime_versions",
        "per_host_simulator_capacity_and_pack_sizes",
        "per_host_resource_caps",
        "per_host_managed_worker_service_names",
    ):
        row[field] = {host: {"captured": True} for host in hosts}
    return row


class Revision5AggregateValidationTests(unittest.TestCase):
    def test_portable_migration_requires_final_gateway_contract_and_r241(self) -> None:
        payload = _migration_payload()
        normalized = aggregate.validate_revision_5_consumer_migration_receipt(payload)

        self.assertEqual(normalized["goal_revision"], 5)
        self.assertFalse(normalized["training_handoff_authorized"])
        self.assertFalse(normalized["kaggle_queue_or_fleet_authorized"])

        payload["goal_contract_sha256"] = (
            "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
        )
        with self.assertRaisesRegex(
            aggregate.R303AggregateValidationError, "revision-5 evidence"
        ):
            aggregate.validate_revision_5_consumer_migration_receipt(payload)

    def test_trusted_migration_rehashes_source_and_refuses_claimed_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "tests.json"
            evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
            evidence.chmod(0o444)
            payload = _migration_payload()
            for consumer in payload["consumers"]:
                consumer["test_evidence"] = _identity(evidence)
                # Deliberately leave a valid-shaped but incorrect source
                # digest: the strict verifier must re-open the checkout.
                consumer["source_sha256"] = _DIGEST
                consumer["source_size_bytes"] = 1
            receipt = root / "migration.json"
            receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            receipt.chmod(0o444)

            with self.assertRaisesRegex(
                aggregate.R303AggregateValidationError, "source bytes"
            ):
                aggregate.inspect_revision_5_consumer_migration_receipt(
                    receipt, trusted_evidence_root=root
                )

    def test_preactivation_summary_cannot_be_relabelled_as_authority(self) -> None:
        identity = {
            "path": "/sealed/receipt.json",
            "sha256": _DIGEST,
            "size_bytes": 1,
        }
        summary = {
            "schema": aggregate.R303_PREACTIVATION_AGGREGATE_SCHEMA,
            "status": "validated_pre_activation_only_no_runtime_or_handoff_authority",
            "goal_gateway_path": aggregate.R303_GOAL_PATH,
            "goal_gateway_sha256": aggregate.R303_GOAL_GATEWAY_SHA256,
            "goal_contract_path": aggregate.R303_CONTRACT_PATH,
            "goal_contract_sha256": aggregate.R303_CONTRACT_SHA256,
            "r241_typed_source_path": aggregate.R303_R241_TYPED_SOURCE_PATH,
            "r241_typed_source_sha256": aggregate.R303_R241_TYPED_SOURCE_SHA256,
            "goal_revision": 5,
            "root_handoff_revision": 303,
            "migration_receipt": identity,
            "migration_predecessor_evidence_schema": (
                "poke_bot.alakazam_collision_census_r298_rev5_predecessor_classification/v1"
            ),
            "census_validation_receipt": identity,
            "census_validation_receipt_schema": (
                "poke_bot.alakazam_collision_census_r298_rev5_validation_receipt/v1"
            ),
            "handoff_receipts": {
                kind: identity
                for kind in (
                    "parent",
                    "corpus",
                    "schema_freeze",
                    "frozen_tensors",
                    "blackwell_preflight",
                    "rollback_plan",
                )
            },
            "immutable_parent_checkpoint_sha256": _DIGEST,
            "immutable_parent_optimizer_state_sha256": _DIGEST,
            "staged_corpus_parity_receipt_sha256": _DIGEST,
            "frozen_tensor_parent_checkpoint_sha256": _DIGEST,
            "blackwell_preflight_host": "inzi",
            "blackwell_preflight_device": "cuda:1",
            "blackwell_preflight_no_runtime_or_service_change": True,
            "staged_shards_training_eligible_now": False,
            "old_r274_services_paused": False,
            "new_derivative_service_started": False,
            "shared_kaggle_queue_service_touched": False,
            "production_serving_selector_authority": False,
            "candidate_action_time_wrapper_sealed": False,
            "training_handoff_authorized": False,
            "kaggle_queue_or_fleet_authorized": False,
        }
        self.assertEqual(aggregate.validate_pre_activation_aggregate(summary), summary)
        summary["training_handoff_authorized"] = True
        with self.assertRaisesRegex(aggregate.R303AggregateValidationError, "must be False"):
            aggregate.validate_pre_activation_aggregate(summary)

    def test_candidate_and_fleet_schemas_reject_authority_or_host_drop(self) -> None:
        candidate = _candidate_receipt()
        self.assertEqual(
            aggregate.validate_candidate_validation_receipt_r303(candidate)["candidate_id"],
            "alakazam-rule-derivative-g5",
        )
        candidate["production_serving_selector_authority"] = True
        with self.assertRaisesRegex(aggregate.R303AggregateValidationError, "mismatch"):
            aggregate.validate_candidate_validation_receipt_r303(candidate)

        inventory = _fleet_inventory()
        self.assertTrue(
            aggregate.validate_fleet_receipt_r303("fleet_inventory", inventory)[
                "full_available_fleet_included"
            ]
        )
        inventory["per_host_resource_caps"].pop("bert")
        with self.assertRaisesRegex(aggregate.R303AggregateValidationError, "every known host"):
            aggregate.validate_fleet_receipt_r303("fleet_inventory", inventory)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
