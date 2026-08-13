from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from durable_goals.errors import IntegrityError, ResolutionError, ValidationError
from durable_goals.io import sha256_file
from durable_goals.io import load_json
from durable_goals.pointers import apply_operations
from durable_goals.resolve import resolve_gateway
from durable_goals.validate import (
    validate_activations,
    validate_amendments,
    validate_contract,
)


EXAMPLE = Path(__file__).parents[1] / "examples" / "model-refresh"


class ResolverTests(unittest.TestCase):
    def test_example_separates_desired_intent_from_activated_reality(self) -> None:
        resolution = resolve_gateway(EXAMPLE / "gateway.json")

        self.assertEqual(resolution.current_revision, 2)
        self.assertEqual(resolution.active_revision, 1)
        self.assertEqual(resolution.active_contract["completion"]["all"][0]["gte"], 0.9)
        self.assertEqual(resolution.desired_contract["completion"]["all"][0]["gte"], 0.85)
        self.assertFalse(resolution.status["active_completion"]["satisfied"])
        self.assertTrue(resolution.status["desired_completion"]["satisfied"])
        self.assertFalse(resolution.status["authoritative"])

    def test_tampered_contract_fails_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "goal"
            shutil.copytree(EXAMPLE, package)
            contract = json.loads((package / "contract.json").read_text())
            contract["objective"] = "Tampered objective"
            (package / "contract.json").write_text(json.dumps(contract))

            with self.assertRaisesRegex(IntegrityError, "contract checksum mismatch"):
                resolve_gateway(package / "gateway.json")

    def test_evidence_payload_is_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "goal"
            shutil.copytree(EXAMPLE, package)
            release = json.loads((package / "receipts" / "release.json").read_text())
            release["activated"] = False
            (package / "receipts" / "release.json").write_text(json.dumps(release))

            with self.assertRaisesRegex(IntegrityError, "evidence release checksum mismatch"):
                resolve_gateway(package / "gateway.json")

    def test_amendment_precondition_detects_semantic_drift(self) -> None:
        contract = {"completion": {"threshold": 0.8}}
        operations = [
            {
                "op": "set",
                "path": "/completion/threshold",
                "expect": 0.9,
                "value": 0.85,
            }
        ]

        with self.assertRaisesRegex(ResolutionError, "precondition failed"):
            apply_operations(contract, operations)

    def test_activation_cannot_jump_over_pending_intent(self) -> None:
        common = {
            "schema": "durable-goals.amendment/v1",
            "goal_id": "goal",
            "recorded_at": "2026-08-12T18:00:00Z",
            "authority": "owner",
            "operations": [{"op": "set", "path": "/objective", "value": "new"}],
            "activation_mode": "manual",
        }
        amendments = [
            {
                **common,
                "revision": 2,
            },
            {
                **common,
                "revision": 3,
            },
        ]

        validated = validate_amendments(
            amendments,
            goal_id="goal",
            base_revision=1,
            current_revision=3,
        )
        activations = [
            {
                "schema": "durable-goals.activation/v1",
                "goal_id": "goal",
                "amendment_revision": 3,
                "activated_at": "2026-08-12T19:00:00Z",
            }
        ]

        with self.assertRaisesRegex(ValidationError, "ordered amendment prefix"):
            validate_activations(
                activations,
                goal_id="goal",
                amendment_revisions=[item["revision"] for item in validated],
            )

    def test_revision_gaps_fail_closed(self) -> None:
        amendments = [
            {
                "schema": "durable-goals.amendment/v1",
                "goal_id": "goal",
                "revision": 3,
                "recorded_at": "2026-08-12T18:00:00Z",
                "authority": "owner",
                "operations": [{"op": "set", "path": "/objective", "value": "new"}],
                "activation_mode": "boundary",
            }
        ]

        with self.assertRaisesRegex(ValidationError, "expected 2, got 3"):
            validate_amendments(
                amendments,
                goal_id="goal",
                base_revision=1,
                current_revision=3,
            )

    def test_gateway_reference_cannot_escape_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "goal"
            shutil.copytree(EXAMPLE, package)
            gateway_path = package / "gateway.json"
            gateway = json.loads(gateway_path.read_text())
            gateway["contract"] = {
                "path": "../outside.json",
                "sha256": "sha256:" + "0" * 64,
            }
            gateway_path.write_text(json.dumps(gateway))

            with self.assertRaisesRegex(ValidationError, "escapes the goal package"):
                resolve_gateway(gateway_path)

    def test_short_checksum_is_rejected_as_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "goal"
            shutil.copytree(EXAMPLE, package)
            gateway_path = package / "gateway.json"
            gateway = json.loads(gateway_path.read_text())
            gateway["contract"]["sha256"] = "sha256:abcd"
            gateway_path.write_text(json.dumps(gateway))

            with self.assertRaisesRegex(ValidationError, "must use sha256"):
                resolve_gateway(gateway_path)

    def test_example_gateway_checksums_are_current(self) -> None:
        gateway = json.loads((EXAMPLE / "gateway.json").read_text())
        for key in ("contract", "amendments", "activations", "evidence_index"):
            reference = gateway[key]
            self.assertEqual(
                sha256_file(EXAMPLE / reference["path"]), reference["sha256"]
            )

    def test_unknown_contract_fields_fail_closed(self) -> None:
        contract = json.loads((EXAMPLE / "contract.json").read_text())
        contract["unknown"] = True
        with self.assertRaisesRegex(ValidationError, "unknown fields"):
            validate_contract(contract, goal_id="example-model-refresh")

    def test_negative_and_leading_zero_array_indices_are_rejected(self) -> None:
        for pointer in ("/items/-1", "/items/01"):
            with self.subTest(pointer=pointer):
                with self.assertRaisesRegex(ResolutionError, "index"):
                    apply_operations(
                        {"items": [1, 2]},
                        [{"op": "remove", "path": pointer}],
                    )

    def test_invalid_pointer_escape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "RFC 6901 escape"):
            apply_operations(
                {"items": {}},
                [{"op": "set", "path": "/items/~2bad", "value": True}],
            )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"revision": 1, "revision": 2}\n')
            with self.assertRaisesRegex(ValidationError, "duplicate JSON object key"):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
