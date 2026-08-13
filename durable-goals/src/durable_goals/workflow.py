"""Deterministic DAG resolution for a harness-owned prompt loop.

This module computes eligibility and emits prompts. It intentionally contains
no worker, model, process, or scheduler abstraction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ResolutionError, ValidationError
from .io import load_json, resolve_local_path
from .resolve import resolve_gateway
from .writer import _atomic_write, _goal_write_lock, _write_immutable


WORKFLOW_SCHEMA = "durable-goals.workflow/v1"
CLAIMS_SCHEMA = "durable-goals.workflow-claims/v1"
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


@dataclass(frozen=True)
class WorkflowResolution:
    workflow_id: str
    revision: int
    topological_order: list[str]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "durable-goals.workflow-status/v1",
            "workflow_id": self.workflow_id,
            "revision": self.revision,
            "topological_order": self.topological_order,
            "nodes": self.nodes,
            "edges": self.edges,
            "authoritative": False,
        }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValidationError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValidationError(
            f"{label} must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, dots, underscores, and hyphens"
        )
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def validate_workflow(value: Any) -> dict[str, Any]:
    workflow = _mapping(value, "workflow")
    _only_keys(
        workflow,
        {"schema", "workflow_id", "revision", "nodes", "edges"},
        "workflow",
    )
    if workflow.get("schema") != WORKFLOW_SCHEMA:
        raise ValidationError(f"workflow.schema must equal {WORKFLOW_SCHEMA}")
    _identifier(workflow.get("workflow_id"), "workflow.workflow_id")
    revision = workflow.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValidationError("workflow.revision must be a positive integer")
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")
    if not isinstance(nodes, list):
        raise ValidationError("workflow.nodes must be an array")
    if not isinstance(edges, list):
        raise ValidationError("workflow.edges must be an array")

    node_ids: set[str] = set()
    for index, raw_node in enumerate(nodes):
        label = f"workflow.nodes[{index}]"
        node = _mapping(raw_node, label)
        _only_keys(node, {"id", "goal_id", "gateway"}, label)
        node_id = _identifier(node.get("id"), f"{label}.id")
        if node_id in node_ids:
            raise ValidationError(f"duplicate workflow node id: {node_id}")
        node_ids.add(node_id)
        _nonempty(node.get("goal_id"), f"{label}.goal_id")
        _nonempty(node.get("gateway"), f"{label}.gateway")

    edge_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for index, raw_edge in enumerate(edges):
        label = f"workflow.edges[{index}]"
        edge = _mapping(raw_edge, label)
        _only_keys(edge, {"id", "from", "to"}, label)
        edge_id = _identifier(edge.get("id"), f"{label}.id")
        if edge_id in edge_ids:
            raise ValidationError(f"duplicate workflow edge id: {edge_id}")
        edge_ids.add(edge_id)
        source = _identifier(edge.get("from"), f"{label}.from")
        target = _identifier(edge.get("to"), f"{label}.to")
        if source not in node_ids or target not in node_ids:
            raise ValidationError(f"workflow edge {edge_id} references an unknown node")
        if source == target:
            raise ValidationError(f"workflow edge {edge_id} is a self-cycle")
        pair = (source, target)
        if pair in pairs:
            raise ValidationError(f"duplicate workflow dependency: {source} -> {target}")
        pairs.add(pair)

    _topological_order(node_ids, edges)
    return workflow


def _topological_order(node_ids: set[str], edges: list[dict[str, Any]]) -> list[str]:
    successors = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        successors[edge["from"]].append(edge["to"])
        indegree[edge["to"]] += 1
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered) != len(node_ids):
        cyclic = sorted(node_id for node_id, count in indegree.items() if count > 0)
        raise ValidationError("workflow contains a cycle involving: " + ", ".join(cyclic))
    return ordered


def resolve_workflow(workflow_path: str | Path) -> WorkflowResolution:
    path = Path(workflow_path).resolve()
    root = path.parent
    workflow = validate_workflow(load_json(path))
    claims = _load_claims(root, workflow["workflow_id"])
    node_by_id = {item["id"]: item for item in workflow["nodes"]}
    predecessors = {node_id: [] for node_id in node_by_id}
    for edge in workflow["edges"]:
        predecessors[edge["to"]].append(edge["from"])
    order = _topological_order(set(node_by_id), workflow["edges"])

    completed: dict[str, bool] = {}
    resolved_goals: dict[str, Any] = {}
    for node_id in order:
        node = node_by_id[node_id]
        gateway = resolve_local_path(root, node["gateway"])
        goal = resolve_gateway(gateway)
        if goal.goal_id != node["goal_id"]:
            raise ResolutionError(
                f"workflow node {node_id} expected goal_id {node['goal_id']}, "
                f"observed {goal.goal_id}"
            )
        resolved_goals[node_id] = goal
        completed[node_id] = bool(goal.status["active_completion"]["satisfied"])

    status_nodes: list[dict[str, Any]] = []
    for node_id in order:
        node = node_by_id[node_id]
        dependencies = sorted(predecessors[node_id])
        dependencies_complete = all(completed[item] for item in dependencies)
        claim = claims.get(node_id)
        if completed[node_id]:
            state = "completed"
        elif dependencies_complete and claim is not None:
            state = "claimed"
        elif dependencies_complete:
            state = "ready"
        else:
            state = "blocked"
        goal = resolved_goals[node_id]
        status_node = {
            "id": node_id,
            "goal_id": node["goal_id"],
            "gateway": node["gateway"],
            "state": state,
            "dependencies": dependencies,
            "active_revision": goal.active_revision,
            "current_revision": goal.current_revision,
        }
        if state == "claimed":
            status_node["claimed_by"] = claim["claimant"]
            status_node["claimed_at"] = claim["claimed_at"]
        status_nodes.append(status_node)

    return WorkflowResolution(
        workflow_id=workflow["workflow_id"],
        revision=workflow["revision"],
        topological_order=order,
        nodes=status_nodes,
        edges=deepcopy(workflow["edges"]),
    )


def next_prompts(
    workflow_path: str | Path, *, all_ready: bool = False
) -> list[dict[str, Any]]:
    path = Path(workflow_path).resolve()
    resolution = resolve_workflow(path)
    ready = [item for item in resolution.nodes if item["state"] == "ready"]
    if not all_ready:
        ready = ready[:1]
    return [_prompt_for_node(path, node) for node in ready]


def _claims_path(root: Path) -> Path:
    return root / ".dgoal" / "workflow-claims.json"


def _load_claims(root: Path, workflow_id: str) -> dict[str, dict[str, str]]:
    path = _claims_path(root)
    if not path.exists():
        return {}
    value = load_json(path)
    if not isinstance(value, dict) or value.get("schema") != CLAIMS_SCHEMA:
        raise ValidationError("workflow claims file has an invalid schema")
    if value.get("workflow_id") != workflow_id:
        raise ValidationError("workflow claims file belongs to another workflow")
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, dict):
        raise ValidationError("workflow claims must be an object")
    claims: dict[str, dict[str, str]] = {}
    for node_id, raw_claim in raw_claims.items():
        _identifier(node_id, "workflow claim node id")
        claim = _mapping(raw_claim, f"workflow claim {node_id}")
        _only_keys(claim, {"claimant", "claimed_at"}, f"workflow claim {node_id}")
        claimant = _nonempty(claim.get("claimant"), f"workflow claim {node_id}.claimant")
        claimed_at = _nonempty(
            claim.get("claimed_at"), f"workflow claim {node_id}.claimed_at"
        )
        claims[node_id] = {"claimant": claimant, "claimed_at": claimed_at}
    return claims


def _write_claims(
    root: Path, workflow_id: str, claims: dict[str, dict[str, str]]
) -> None:
    payload = (
        json.dumps(
            {
                "schema": CLAIMS_SCHEMA,
                "workflow_id": workflow_id,
                "claims": claims,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(root, _claims_path(root), payload)


def claim_next_prompt(
    workflow_path: str | Path, *, claimant: str
) -> dict[str, Any] | None:
    """Atomically claim the first ready goal for an otherwise unassigned thread."""

    _nonempty(claimant, "claimant")
    path = Path(workflow_path).resolve()
    root = path.parent
    with _goal_write_lock(root, wait=True):
        resolution = resolve_workflow(path)
        ready = [item for item in resolution.nodes if item["state"] == "ready"]
        claims = _load_claims(root, resolution.workflow_id)
        live_node_ids = {item["id"] for item in resolution.nodes}
        completed_node_ids = {
            item["id"] for item in resolution.nodes if item["state"] == "completed"
        }
        claims = {
            node_id: claim
            for node_id, claim in claims.items()
            if node_id in live_node_ids and node_id not in completed_node_ids
        }
        if not ready:
            _write_claims(root, resolution.workflow_id, claims)
            return None
        node = ready[0]
        claimed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        claims[node["id"]] = {"claimant": claimant, "claimed_at": claimed_at}
        _write_claims(root, resolution.workflow_id, claims)
        prompt = _prompt_for_node(path, node)
        prompt["claimant"] = claimant
        prompt["claimed_at"] = claimed_at
        return prompt


def release_claim(
    workflow_path: str | Path, node_id: str, *, claimant: str
) -> None:
    _nonempty(claimant, "claimant")
    path = Path(workflow_path).resolve()
    root = path.parent
    with _goal_write_lock(root, wait=True):
        workflow = validate_workflow(load_json(path))
        claims = _load_claims(root, workflow["workflow_id"])
        claim = claims.get(node_id)
        if claim is None:
            raise ResolutionError(f"workflow node is not claimed: {node_id}")
        if claim["claimant"] != claimant:
            raise ResolutionError(
                f"workflow node {node_id} is claimed by {claim['claimant']}, not {claimant}"
            )
        del claims[node_id]
        _write_claims(root, workflow["workflow_id"], claims)


def _prompt_for_node(path: Path, node: dict[str, Any]) -> dict[str, Any]:
    gateway = resolve_local_path(path.parent, node["gateway"])
    goal_entrypoint = gateway.parent / "GOAL.md"
    if not goal_entrypoint.is_file():
        raise ResolutionError(
            f"workflow node {node['id']} has no GOAL.md beside its gateway"
        )
    relative_goal = Path(os.path.relpath(goal_entrypoint, path.parent)).as_posix()
    return {
        "node_id": node["id"],
        "goal_id": node["goal_id"],
        "goal_entrypoint": relative_goal,
        "prompt": (
            f"Read {relative_goal} completely, then read every canonical source "
            "it requires for the current action. Continue the authoritative goal "
            "until its evidence-backed completion condition is satisfied."
        ),
    }


def _write_workflow(path: Path, workflow: dict[str, Any]) -> None:
    root = path.parent
    validate_workflow(workflow)
    payload = (json.dumps(workflow, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    history = (
        root
        / ".dgoal"
        / "workflow-history"
        / f"{workflow['workflow_id']}-r{workflow['revision']:06d}-{digest[:12]}.json"
    )
    _write_immutable(root, history, payload)
    _atomic_write(root, path, payload)


def initialize_workflow(path: str | Path, *, workflow_id: str) -> Path:
    target = Path(path).resolve()
    root = target.parent
    _identifier(workflow_id, "workflow_id")
    if target.exists():
        raise ResolutionError(f"workflow already exists: {target}")
    workflow = {
        "schema": WORKFLOW_SCHEMA,
        "workflow_id": workflow_id,
        "revision": 1,
        "nodes": [],
        "edges": [],
    }
    with _goal_write_lock(root):
        if target.exists():
            raise ResolutionError(f"workflow already exists: {target}")
        _write_workflow(target, workflow)
    return target


def _mutate_workflow(path: Path, mutation: Any) -> dict[str, Any]:
    root = path.parent
    with _goal_write_lock(root):
        workflow = validate_workflow(load_json(path))
        updated = deepcopy(workflow)
        mutation(updated)
        updated["revision"] = workflow["revision"] + 1
        validate_workflow(updated)
        _write_workflow(path, updated)
        return updated


def add_goal_node(
    workflow_path: str | Path, gateway_path: str | Path, *, node_id: str
) -> dict[str, Any]:
    path = Path(workflow_path).resolve()
    gateway = Path(gateway_path).resolve()
    goal = resolve_gateway(gateway)
    relative = os.path.relpath(gateway, path.parent)
    resolve_local_path(path.parent, relative)
    _identifier(node_id, "node_id")

    def mutate(workflow: dict[str, Any]) -> None:
        if any(item["id"] == node_id for item in workflow["nodes"]):
            raise ResolutionError(f"workflow node already exists: {node_id}")
        if any(item["goal_id"] == goal.goal_id for item in workflow["nodes"]):
            raise ResolutionError(f"workflow already contains goal_id: {goal.goal_id}")
        workflow["nodes"].append(
            {
                "id": node_id,
                "goal_id": goal.goal_id,
                "gateway": Path(relative).as_posix(),
            }
        )

    return _mutate_workflow(path, mutate)


def add_dependency(
    workflow_path: str | Path, source: str, target: str, *, edge_id: str
) -> dict[str, Any]:
    path = Path(workflow_path).resolve()
    _identifier(source, "source")
    _identifier(target, "target")
    _identifier(edge_id, "edge_id")

    def mutate(workflow: dict[str, Any]) -> None:
        workflow["edges"].append({"id": edge_id, "from": source, "to": target})

    return _mutate_workflow(path, mutate)


def remove_dependency(workflow_path: str | Path, edge_id: str) -> dict[str, Any]:
    path = Path(workflow_path).resolve()

    def mutate(workflow: dict[str, Any]) -> None:
        retained = [item for item in workflow["edges"] if item["id"] != edge_id]
        if len(retained) == len(workflow["edges"]):
            raise ResolutionError(f"unknown workflow edge: {edge_id}")
        workflow["edges"] = retained

    return _mutate_workflow(path, mutate)


def remove_goal_node(workflow_path: str | Path, node_id: str) -> dict[str, Any]:
    path = Path(workflow_path).resolve()

    def mutate(workflow: dict[str, Any]) -> None:
        retained = [item for item in workflow["nodes"] if item["id"] != node_id]
        if len(retained) == len(workflow["nodes"]):
            raise ResolutionError(f"unknown workflow node: {node_id}")
        workflow["nodes"] = retained
        workflow["edges"] = [
            item
            for item in workflow["edges"]
            if item["from"] != node_id and item["to"] != node_id
        ]

    return _mutate_workflow(path, mutate)
