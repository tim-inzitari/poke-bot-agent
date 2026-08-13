from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from durable_goals.cli import main


EXAMPLE_GATEWAY = (
    Path(__file__).parents[1] / "examples" / "model-refresh" / "gateway.json"
)


class CliTests(unittest.TestCase):
    def invoke(self, *args: str) -> tuple[int, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(list(args))
        return result, json.loads(output.getvalue())

    def test_validate_command(self) -> None:
        result, payload = self.invoke("validate", str(EXAMPLE_GATEWAY))
        self.assertEqual(result, 0)
        self.assertEqual(
            payload,
            {
                "active_revision": 1,
                "current_revision": 2,
                "goal_id": "example-model-refresh",
                "valid": True,
            },
        )

    def test_status_command_is_explicitly_non_authoritative(self) -> None:
        result, payload = self.invoke("status", str(EXAMPLE_GATEWAY))
        self.assertEqual(result, 0)
        self.assertFalse(payload["authoritative"])
        self.assertEqual(payload["pending_activation_revisions"], [2])

    def test_prompt_style_amend_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "goal"
            shutil.copytree(EXAMPLE_GATEWAY.parent, package)
            gateway = package / "gateway.json"
            result, payload = self.invoke(
                "amend",
                str(gateway),
                "--set",
                "/objective",
                '"Promote and monitor the validated model."',
                "--expect",
                "/objective",
                '"Promote a validated model without losing provenance."',
                "--reason",
                "Owner added monitoring.",
            )
            self.assertEqual(result, 0)
            self.assertEqual(payload["current_revision"], 3)
            self.assertEqual(payload["active_revision"], 1)
            self.assertEqual(payload["pending_activation_revisions"], [2, 3])

    def test_invalid_prompt_json_is_reported_as_a_goal_error(self) -> None:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = main(
                [
                    "amend",
                    str(EXAMPLE_GATEWAY),
                    "--set",
                    "/objective",
                    "not-json",
                    "--reason",
                    "Invalid input should fail before writing.",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("must be valid JSON", errors.getvalue())

    def test_workflow_cli_emits_next_prompt_without_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, first = self.invoke(
                "init",
                str(root / "goal"),
                "--goal-id",
                "goal",
                "--objective",
                "Do the goal.",
            )
            self.assertEqual(result, 0)
            workflow = root / "workflow.json"
            result, _ = self.invoke(
                "workflow",
                "init",
                str(workflow),
                "--workflow-id",
                "flow",
            )
            self.assertEqual(result, 0)
            result, _ = self.invoke(
                "workflow",
                "add-goal",
                str(workflow),
                first["gateway"],
                "--node-id",
                "goal",
            )
            self.assertEqual(result, 0)
            result, payload = self.invoke("workflow", "next", str(workflow))
            self.assertEqual(result, 0)
            self.assertEqual(payload["prompts"][0]["node_id"], "goal")
            self.assertNotIn("model", json.dumps(payload).lower())
            self.assertNotIn("assignee", json.dumps(payload).lower())

    def test_evidence_add_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, initialized = self.invoke(
                "init",
                str(root / "goal"),
                "--goal-id",
                "goal",
                "--objective",
                "Collect evidence.",
            )
            self.assertEqual(result, 0)
            receipt = root / "receipt.json"
            receipt.write_text('{"passed": true}\n')
            result, payload = self.invoke(
                "evidence",
                "add",
                initialized["gateway"],
                "check",
                str(receipt),
            )
            self.assertEqual(result, 0)
            self.assertEqual(payload["evidence_ids"], ["check"])

    def test_workflow_claim_cli_excludes_claimed_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, initialized = self.invoke(
                "init",
                str(root / "goal"),
                "--goal-id",
                "goal",
                "--objective",
                "Claim this goal.",
            )
            workflow = root / "workflow.json"
            self.invoke(
                "workflow",
                "init",
                str(workflow),
                "--workflow-id",
                "flow",
            )
            self.invoke(
                "workflow",
                "add-goal",
                str(workflow),
                initialized["gateway"],
                "--node-id",
                "goal",
            )
            result, payload = self.invoke(
                "workflow",
                "claim",
                str(workflow),
                "--claimant",
                "thread-a",
            )
            self.assertEqual(result, 0)
            self.assertEqual(payload["claim"]["node_id"], "goal")
            result, payload = self.invoke(
                "workflow",
                "claim",
                str(workflow),
                "--claimant",
                "thread-b",
            )
            self.assertEqual(result, 0)
            self.assertIsNone(payload["claim"])


if __name__ == "__main__":
    unittest.main()
