from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from durable_goals.errors import ResolutionError
from durable_goals.resolve import resolve_gateway
from durable_goals.writer import (
    _goal_write_lock,
    activate_amendment,
    chain_goal,
    initialize_goal_package,
    materialize_status,
    record_amendment,
    record_evidence,
)


EXAMPLE = Path(__file__).parents[1] / "examples" / "model-refresh"


class WriterTests(unittest.TestCase):
    def copy_example(self, directory: str) -> tuple[Path, Path]:
        package = Path(directory) / "goal"
        shutil.copytree(EXAMPLE, package)
        return package, package / "gateway.json"

    def test_amendment_uses_immutable_history_and_atomic_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, gateway_path = self.copy_example(directory)
            original_ledger = (package / "amendments.jsonl").read_bytes()

            resolution = record_amendment(
                gateway_path,
                operations=[
                    {
                        "op": "set",
                        "path": "/objective",
                        "expect": "Promote a validated model without losing provenance.",
                        "value": "Promote and monitor the validated model.",
                    }
                ],
                reason="Owner added post-promotion monitoring.",
                recorded_at="2026-08-12T20:00:00Z",
            )

            gateway = json.loads(gateway_path.read_text())
            self.assertEqual(gateway["current_revision"], 3)
            self.assertTrue(gateway["amendments"]["path"].startswith(".dgoal/history/"))
            self.assertEqual((package / "amendments.jsonl").read_bytes(), original_ledger)
            self.assertEqual(resolution.current_revision, 3)
            self.assertEqual(resolution.active_revision, 1)
            self.assertEqual(
                [item["revision"] for item in resolution.pending_activations], [2, 3]
            )

    def test_activation_and_status_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, gateway_path = self.copy_example(directory)

            resolution = activate_amendment(
                gateway_path,
                2,
                evidence_id="evaluation",
                activated_at="2026-08-12T20:05:00Z",
            )
            self.assertEqual(resolution.active_revision, 2)
            self.assertTrue(resolution.status["active_completion"]["satisfied"])
            gateway = json.loads(gateway_path.read_text())
            self.assertTrue(gateway["activations"]["path"].startswith(".dgoal/history/"))

            status_path = materialize_status(gateway_path)
            status = json.loads(status_path.read_text())
            self.assertFalse(status["authoritative"])
            self.assertEqual(status["active_revision"], 2)
            self.assertEqual(resolve_gateway(gateway_path).status, status)

    def test_initialize_and_chain_successor_goals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_gateway = initialize_goal_package(
                root / "source",
                goal_id="source-goal",
                objective="Complete the source work.",
            )
            successor_gateway = initialize_goal_package(
                root / "successor",
                goal_id="successor-goal",
                objective="Begin only after the source goal completes.",
            )
            source = resolve_gateway(source_gateway)
            self.assertFalse(source.status["active_completion"]["satisfied"])

            source = record_amendment(
                source_gateway,
                operations=[
                    {
                        "op": "set",
                        "path": "/completion",
                        "expect": {"literal": False},
                        "value": {"literal": True},
                    }
                ],
                reason="Test evidence marks the source complete.",
                recorded_at="2026-08-12T21:00:00Z",
            )
            self.assertEqual(source.current_revision, 2)
            source = activate_amendment(
                source_gateway, 2, activated_at="2026-08-12T21:01:00Z"
            )
            self.assertTrue(source.status["active_completion"]["satisfied"])

            source = chain_goal(
                source_gateway,
                successor_gateway,
                transition_id="then-successor",
                reason="Owner ordered the successor after source completion.",
            )
            self.assertEqual(source.current_revision, 3)
            self.assertEqual(source.status["transitions"], [])
            source = activate_amendment(
                source_gateway, 3, activated_at="2026-08-12T21:02:00Z"
            )
            self.assertEqual(source.status["ready_transitions"], ["then-successor"])
            transition = source.status["transitions"][0]
            self.assertEqual(transition["goal_id"], "successor-goal")
            self.assertTrue(transition["ready"])

    def test_safe_boundary_requires_and_binds_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = initialize_goal_package(
                root / "goal", goal_id="boundary", objective="Wait for a boundary."
            )
            amendment = record_amendment(
                gateway,
                operations=[
                    {
                        "op": "set",
                        "path": "/objective",
                        "expect": "Wait for a boundary.",
                        "value": "Continue after the boundary.",
                    }
                ],
                reason="Boundary-gated change.",
                activation_mode="next_safe_boundary",
                activation_condition={
                    "evidence": "boundary",
                    "field": "/safe",
                    "equals": True,
                },
            )
            with self.assertRaisesRegex(ResolutionError, "not satisfied"):
                activate_amendment(gateway, amendment.current_revision)

            receipt = root / "receipt.json"
            receipt.write_text('{"safe": true}\n')
            record_evidence(gateway, "boundary", receipt)
            activated = activate_amendment(gateway, amendment.current_revision)
            self.assertEqual(activated.active_revision, amendment.current_revision)
            gateway_value = json.loads(Path(gateway).read_text())
            activations = (
                Path(gateway).parent / gateway_value["activations"]["path"]
            ).read_text()
            self.assertIn('"id":"boundary"', activations)
            self.assertIn('"sha256":"sha256:', activations)

    def test_writer_rejects_symlinked_internal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, gateway = self.copy_example(directory)
            outside = root / "outside"
            outside.mkdir()
            (package / ".dgoal").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ResolutionError, "symbolic link"):
                record_amendment(
                    gateway,
                    operations=[
                        {
                            "op": "set",
                            "path": "/objective",
                            "value": "Must not escape.",
                        }
                    ],
                    reason="Containment test.",
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_competing_writer_fails_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, gateway = self.copy_example(directory)
            before = gateway.read_bytes()
            with _goal_write_lock(package):
                with self.assertRaisesRegex(ResolutionError, "another durable-goal writer"):
                    record_amendment(
                        gateway,
                        operations=[
                            {
                                "op": "set",
                                "path": "/objective",
                                "value": "Contended write.",
                            }
                        ],
                        reason="Contention test.",
                    )
            self.assertEqual(gateway.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
