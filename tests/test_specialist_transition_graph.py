from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.run_specialist_transition_graph as graph_runner


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def graph() -> dict:
    return {
        "schema": graph_runner.GRAPH_SCHEMA,
        "graph_id": "canonical-specialist-transition",
        "description": "test",
        "policies": {
            "active_training_must_be_stopped": True,
            "fail_closed": True,
            "kaggle_submission_queue_blocks_transition": False,
            "shell_commands_in_graph_allowed": False,
        },
        "service_allowlist": {
            "source": "cycle_contract",
            "references": [
                "runtime.training_service",
                "runtime.handoff_service",
                "runtime.population_handoff_service",
                "runtime.population_training_service",
                "runtime.gate_handler_service",
            ],
        },
        "nodes": [
            {
                "id": "contract_preflight",
                "action": "validate_cycle_contract",
                "depends_on": [],
            },
            {
                "id": "training_boundary",
                "action": "assert_active_training_stopped",
                "depends_on": ["contract_preflight"],
            },
            {
                "id": "specialist_transition",
                "action": "execute_existing_idempotent_handoff",
                "depends_on": ["training_boundary"],
            },
        ],
    }


def cycle(selector: Path) -> dict:
    return {
        "schema": graph_runner.CYCLE_SCHEMA,
        "runtime": {
            "selector_env": str(selector),
            "training_service": "pokebot-specialist-training.service",
            "handoff_service": "pokebot-specialist-handoff.service",
            "population_handoff_service": "pokebot-population-handoff.service",
            "population_training_service": "pokebot-population.service",
            "gate_handler_service": "pokebot-specialist-gate-handler.service",
        },
    }


def context(tmp_path: Path):
    selector = tmp_path / "specialist.env"
    selector.write_text("POKEBOT_ACTIVE_SPECIALIST=active-specialist\n")
    graph_path = write(tmp_path / "graph.json", graph())
    cycle_path = write(tmp_path / "cycle.json", cycle(selector))
    return graph_runner.build_context(
        graph_path=graph_path,
        cycle_path=cycle_path,
        state_path=tmp_path / "state.json",
    )


def handlers(events: list[str], *, fail_once: str | None = None):
    failed: set[str] = set()

    def make(action: str):
        def invoke(_context):
            events.append(action)
            if action == fail_once and action not in failed:
                failed.add(action)
                raise RuntimeError("injected failure")
            return {"action": action}

        return invoke

    return {action: make(action) for action in graph_runner.ALLOWED_ACTIONS}


def test_dependency_order_and_idempotent_resume(tmp_path: Path) -> None:
    ctx, nodes = context(tmp_path)
    events: list[str] = []
    fake = handlers(events)
    result = graph_runner.run(ctx, nodes, handlers=fake)
    assert events == [node.action for node in nodes]
    assert all(row["status"] == "complete" for row in result["nodes"])
    graph_runner.run(ctx, nodes, handlers=fake)
    assert events == [node.action for node in nodes]


def test_resume_starts_at_failed_node(tmp_path: Path) -> None:
    ctx, nodes = context(tmp_path)
    events: list[str] = []
    fake = handlers(events, fail_once="assert_active_training_stopped")
    with pytest.raises(RuntimeError, match="injected failure"):
        graph_runner.run(ctx, nodes, handlers=fake)
    graph_runner.run(ctx, nodes, handlers=fake)
    assert events == [
        "validate_cycle_contract",
        "assert_active_training_stopped",
        "assert_active_training_stopped",
        "execute_existing_idempotent_handoff",
    ]


def test_contract_checksum_change_creates_new_transition(tmp_path: Path) -> None:
    ctx, nodes = context(tmp_path)
    events: list[str] = []
    graph_runner.run(ctx, nodes, handlers=handlers(events))
    changed = copy.deepcopy(ctx.cycle)
    changed["non_runtime_metadata"] = "changed"
    write(ctx.cycle_path, changed)
    updated, updated_nodes = graph_runner.build_context(
        graph_path=ctx.graph_path,
        cycle_path=ctx.cycle_path,
        state_path=ctx.state_path,
    )
    graph_runner.run(updated, updated_nodes, handlers=handlers(events))
    assert events == [node.action for node in nodes] * 2
    state = json.loads(ctx.state_path.read_text())
    assert len(state["transitions"]) == 2


def test_graph_checksum_change_fails_closed(tmp_path: Path) -> None:
    ctx, nodes = context(tmp_path)
    graph_runner.run(ctx, nodes, handlers=handlers([]))
    changed = copy.deepcopy(ctx.graph)
    changed["description"] = "reviewed wording"
    write(ctx.graph_path, changed)
    updated, updated_nodes = graph_runner.build_context(
        graph_path=ctx.graph_path,
        cycle_path=ctx.cycle_path,
        state_path=ctx.state_path,
    )
    with pytest.raises(RuntimeError, match="explicit migration required"):
        graph_runner.inspect(updated, updated_nodes)


def test_active_training_blocks_handoff(tmp_path: Path, monkeypatch) -> None:
    ctx, _nodes = context(tmp_path)
    monkeypatch.setattr(graph_runner, "service_active", lambda _name: True)
    with pytest.raises(RuntimeError, match="has not terminated"):
        graph_runner._assert_active_training_stopped(ctx)


def test_graph_is_specialist_agnostic_and_has_no_commands() -> None:
    path = Path("ops/specialist_transition_graph.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = json.dumps(value).lower()
    forbidden = (
        "alakazam",
        "starmie",
        "trevenant",
        "lucario",
        "dragapult",
        "2026",
        "_v2",
        "_v3",
        "_v4",
        "_v5",
        "systemctl",
        "subprocess",
        "argv",
    )
    assert not any(token in raw for token in forbidden)
    assert all(
        set(node) == {"id", "action", "depends_on"}
        for node in value["nodes"]
    )


def test_dry_run_never_invokes_actions(tmp_path: Path) -> None:
    ctx, nodes = context(tmp_path)
    result = graph_runner.run(
        ctx,
        nodes,
        dry_run=True,
        handlers={
            action: lambda _context: pytest.fail("action invoked")
            for action in graph_runner.ALLOWED_ACTIONS
        },
    )
    assert result["dry_run"] is True
    assert all(row["status"] == "pending" for row in result["nodes"])


def test_graph_rejects_arbitrary_command_or_service_reference(
    tmp_path: Path,
) -> None:
    value = graph()
    value["nodes"][0]["command"] = "systemctl stop anything"
    with pytest.raises(RuntimeError, match="only id/action/depends_on"):
        graph_runner.load_graph(write(tmp_path / "command.json", value))

    value = graph()
    value["service_allowlist"]["references"].append(
        "runtime.unreviewed_service"
    )
    with pytest.raises(RuntimeError, match="service allowlist changed"):
        graph_runner.load_graph(write(tmp_path / "service.json", value))
