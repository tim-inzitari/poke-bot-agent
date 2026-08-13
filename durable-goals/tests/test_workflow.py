from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from durable_goals.errors import ResolutionError, ValidationError
from durable_goals.workflow import (
    add_dependency,
    add_goal_node,
    claim_next_prompt,
    initialize_workflow,
    next_prompts,
    remove_dependency,
    release_claim,
    resolve_workflow,
)
from durable_goals.writer import (
    activate_amendment,
    initialize_goal_package,
    record_amendment,
)


class WorkflowTests(unittest.TestCase):
    def complete(self, gateway: Path) -> None:
        resolution = record_amendment(
            gateway,
            operations=[
                {
                    "op": "set",
                    "path": "/completion",
                    "expect": {"literal": False},
                    "value": {"literal": True},
                }
            ],
            reason="Test completion.",
        )
        activate_amendment(gateway, resolution.current_revision)

    def test_dag_fan_in_readiness_and_prompt_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = initialize_goal_package(
                root / "goals" / "first", goal_id="first", objective="First."
            )
            second = initialize_goal_package(
                root / "goals" / "second", goal_id="second", objective="Second."
            )
            final = initialize_goal_package(
                root / "goals" / "final", goal_id="final", objective="Final."
            )
            self.complete(first)
            workflow = initialize_workflow(
                root / "workflow.json", workflow_id="release-flow"
            )
            add_goal_node(workflow, first, node_id="first")
            add_goal_node(workflow, second, node_id="second")
            add_goal_node(workflow, final, node_id="final")
            add_dependency(workflow, "first", "final", edge_id="first-final")
            add_dependency(workflow, "second", "final", edge_id="second-final")

            resolution = resolve_workflow(workflow)
            states = {item["id"]: item["state"] for item in resolution.nodes}
            self.assertEqual(
                states, {"first": "completed", "second": "ready", "final": "blocked"}
            )
            prompts = next_prompts(workflow)
            self.assertEqual([item["node_id"] for item in prompts], ["second"])
            self.assertIn("goals/second/GOAL.md", prompts[0]["prompt"])
            self.assertNotIn("model", json.dumps(resolution.to_dict()).lower())
            self.assertNotIn("assignee", json.dumps(resolution.to_dict()).lower())

    def test_cycle_is_rejected_without_advancing_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = initialize_goal_package(
                root / "goals" / "one", goal_id="one", objective="One."
            )
            two = initialize_goal_package(
                root / "goals" / "two", goal_id="two", objective="Two."
            )
            workflow = initialize_workflow(root / "workflow.json", workflow_id="flow")
            add_goal_node(workflow, one, node_id="one")
            add_goal_node(workflow, two, node_id="two")
            add_dependency(workflow, "one", "two", edge_id="one-two")
            before = Path(workflow).read_bytes()

            with self.assertRaisesRegex(ValidationError, "contains a cycle"):
                add_dependency(workflow, "two", "one", edge_id="two-one")
            self.assertEqual(Path(workflow).read_bytes(), before)

    def test_goal_outside_workflow_package_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = initialize_workflow(
                root / "inside" / "workflow.json", workflow_id="flow"
            )
            outside = initialize_goal_package(
                root / "outside", goal_id="outside", objective="Outside."
            )
            with self.assertRaisesRegex(ValidationError, "escapes"):
                add_goal_node(workflow, outside, node_id="outside")

    def test_remove_dependency_unblocks_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = initialize_goal_package(
                root / "one", goal_id="one", objective="One."
            )
            two = initialize_goal_package(
                root / "two", goal_id="two", objective="Two."
            )
            workflow = initialize_workflow(root / "workflow.json", workflow_id="flow")
            add_goal_node(workflow, one, node_id="one")
            add_goal_node(workflow, two, node_id="two")
            add_dependency(workflow, "one", "two", edge_id="one-two")
            remove_dependency(workflow, "one-two")
            states = {item["id"]: item["state"] for item in resolve_workflow(workflow).nodes}
            self.assertEqual(states, {"one": "ready", "two": "ready"})

    def test_parallel_threads_claim_different_independent_goals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = initialize_goal_package(
                root / "one", goal_id="one", objective="Independent one."
            )
            two = initialize_goal_package(
                root / "two", goal_id="two", objective="Independent two."
            )
            workflow = initialize_workflow(root / "workflow.json", workflow_id="flow")
            add_goal_node(workflow, one, node_id="one")
            add_goal_node(workflow, two, node_id="two")

            with ThreadPoolExecutor(max_workers=2) as pool:
                claims = list(
                    pool.map(
                        lambda claimant: claim_next_prompt(
                            workflow, claimant=claimant
                        ),
                        ("thread-a", "thread-b"),
                    )
                )
            self.assertEqual(
                {claim["node_id"] for claim in claims if claim is not None},
                {"one", "two"},
            )
            states = {item["id"]: item for item in resolve_workflow(workflow).nodes}
            self.assertEqual({item["state"] for item in states.values()}, {"claimed"})
            self.assertEqual(
                {item["claimed_by"] for item in states.values()},
                {"thread-a", "thread-b"},
            )
            self.assertEqual(next_prompts(workflow, all_ready=True), [])

    def test_claim_release_returns_goal_to_ready_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            goal = initialize_goal_package(
                root / "goal", goal_id="goal", objective="Claimable."
            )
            workflow = initialize_workflow(root / "workflow.json", workflow_id="flow")
            add_goal_node(workflow, goal, node_id="goal")
            claim = claim_next_prompt(workflow, claimant="thread-a")
            self.assertEqual(claim["node_id"], "goal")
            with self.assertRaisesRegex(ResolutionError, "claimed by thread-a"):
                release_claim(workflow, "goal", claimant="thread-b")
            release_claim(workflow, "goal", claimant="thread-a")
            self.assertEqual(next_prompts(workflow)[0]["node_id"], "goal")


if __name__ == "__main__":
    unittest.main()
