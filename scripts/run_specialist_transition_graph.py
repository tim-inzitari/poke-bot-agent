#!/usr/bin/env python3
"""Run the canonical specialist transition as a small, durable task graph.

The graph is deliberately data-only: it may select only named Python actions,
never commands. Existing handoff modules continue to own model validation,
bootstrap receipts, selector mutation, service control, and the asynchronous
Kaggle submission queue. This wrapper adds dependency ordering, a persistent
per-specialist journal, checksum invalidation, dry-run, and status reporting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(ROOT))
except ValueError:
    pass
sys.path.insert(0, str(ROOT))

GRAPH_SCHEMA = "poke_bot.specialist_transition_graph/v1"
CYCLE_SCHEMA = "poke_bot.specialist_cycle_handoff_contract/v1"
STATE_SCHEMA = "poke_bot.specialist_transition_graph_state/v1"
RECEIPT_SCHEMA = "poke_bot.specialist_transition_node_receipt/v1"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
SAFE_SERVICE = re.compile(r"pokebot-[A-Za-z0-9_.@-]+\.service")
ALLOWED_ACTIONS = {
    "validate_cycle_contract",
    "assert_active_training_stopped",
    "execute_existing_idempotent_handoff",
}
ALLOWED_SERVICE_REFERENCES = {
    "runtime.training_service",
    "runtime.handoff_service",
    "runtime.population_handoff_service",
    "runtime.population_training_service",
    "runtime.gate_handler_service",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing/corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON object: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _active_specialist(runtime: dict[str, Any]) -> str:
    raw = str(runtime.get("selector_env") or "").strip()
    if not raw:
        raise RuntimeError("canonical selector path is missing")
    selector = Path(raw).expanduser().resolve()
    rows = [
        line.split("=", 1)[1].strip()
        for line in selector.read_text(encoding="utf-8").splitlines()
        if line.startswith("POKEBOT_ACTIVE_SPECIALIST=")
    ]
    if len(rows) != 1 or not rows[0]:
        raise RuntimeError("canonical specialist selector is absent or duplicated")
    return rows[0]


def service_active(name: str) -> bool:
    return (
        subprocess.run(
            ["/usr/bin/systemctl", "--user", "is-active", "--quiet", name],
            check=False,
        ).returncode
        == 0
    )


def _nested(value: dict[str, Any], reference: str) -> Any:
    current: Any = value
    for key in reference.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"service allowlist reference is missing: {reference}")
        current = current[key]
    return current


@dataclass(frozen=True)
class Node:
    identifier: str
    action: str
    dependencies: tuple[str, ...]


@dataclass
class Context:
    graph_path: Path
    graph: dict[str, Any]
    graph_digest: str
    cycle_path: Path
    cycle: dict[str, Any]
    cycle_digest: str
    state_path: Path
    transition_id: str
    active_specialist: str
    allowed_services: frozenset[str]


def load_graph(path: Path) -> tuple[dict[str, Any], list[Node], str]:
    path = path.expanduser().resolve()
    graph = _read(path)
    policies = dict(graph.get("policies") or {})
    allowlist = dict(graph.get("service_allowlist") or {})
    rows = graph.get("nodes")
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or not SAFE_ID.fullmatch(str(graph.get("graph_id") or ""))
        or policies.get("active_training_must_be_stopped") is not True
        or policies.get("fail_closed") is not True
        or policies.get("kaggle_submission_queue_blocks_transition") is not False
        or policies.get("shell_commands_in_graph_allowed") is not False
        or allowlist.get("source") != "cycle_contract"
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeError("specialist transition graph policy changed")
    references = tuple(str(value) for value in allowlist.get("references") or ())
    if (
        not references
        or len(references) != len(set(references))
        or not set(references).issubset(ALLOWED_SERVICE_REFERENCES)
    ):
        raise RuntimeError("specialist transition service allowlist changed")
    nodes: list[Node] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "action",
            "depends_on",
        }:
            raise RuntimeError("graph nodes may contain only id/action/depends_on")
        identifier = str(row["id"])
        action = str(row["action"])
        dependencies = tuple(str(value) for value in row["depends_on"])
        if (
            not SAFE_ID.fullmatch(identifier)
            or identifier in seen
            or action not in ALLOWED_ACTIONS
            or len(dependencies) != len(set(dependencies))
            or any(value not in seen for value in dependencies)
        ):
            raise RuntimeError("graph dependency order or action changed")
        seen.add(identifier)
        nodes.append(Node(identifier, action, dependencies))
    if {node.action for node in nodes} != ALLOWED_ACTIONS:
        raise RuntimeError("canonical transition actions are incomplete")
    return graph, nodes, _sha256(path)


def build_context(
    *,
    graph_path: Path,
    cycle_path: Path,
    state_path: Path,
) -> tuple[Context, list[Node]]:
    graph, nodes, graph_digest = load_graph(graph_path)
    cycle_path = cycle_path.expanduser().resolve()
    cycle = _read(cycle_path)
    if cycle.get("schema") != CYCLE_SCHEMA:
        raise RuntimeError("specialist cycle contract schema changed")
    runtime = dict(cycle.get("runtime") or {})
    active = _active_specialist(runtime)
    if not SAFE_ID.fullmatch(active):
        raise RuntimeError("active specialist selector is invalid")
    references = tuple(
        str(value)
        for value in dict(graph["service_allowlist"]).get("references") or ()
    )
    services = frozenset(str(_nested(cycle, value)) for value in references)
    if (
        len(services) != len(references)
        or any(not SAFE_SERVICE.fullmatch(value) for value in services)
    ):
        raise RuntimeError("resolved service allowlist is not exact and safe")
    cycle_digest = _sha256(cycle_path)
    transition_id = _digest(
        {
            "active_specialist": active,
            "cycle_contract_sha256": cycle_digest,
        }
    )
    return (
        Context(
            graph_path=graph_path.expanduser().resolve(),
            graph=graph,
            graph_digest=graph_digest,
            cycle_path=cycle_path,
            cycle=cycle,
            cycle_digest=cycle_digest,
            state_path=state_path.expanduser().resolve(),
            transition_id=transition_id,
            active_specialist=active,
            allowed_services=services,
        ),
        nodes,
    )


def _new_state(context: Context) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "graph_id": context.graph["graph_id"],
        "graph_sha256": context.graph_digest,
        "transitions": {},
    }


def _load_state(context: Context) -> dict[str, Any]:
    if not context.state_path.is_file():
        return _new_state(context)
    state = _read(context.state_path)
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("graph_id") != context.graph["graph_id"]
        or state.get("graph_sha256") != context.graph_digest
        or not isinstance(state.get("transitions"), dict)
    ):
        raise RuntimeError(
            "transition graph/state identity changed; explicit migration required"
        )
    return state


def _node_input_digest(
    context: Context,
    node: Node,
    receipts: dict[str, Any],
) -> str:
    dependencies = {
        dependency: (
            _digest(receipts[dependency])
            if dependency in receipts
            else None
        )
        for dependency in node.dependencies
    }
    return _digest(
        {
            "graph_sha256": context.graph_digest,
            "cycle_contract_sha256": context.cycle_digest,
            "transition_id": context.transition_id,
            "node": node.identifier,
            "action": node.action,
            "dependencies": dependencies,
        }
    )


def _validate_cycle_contract(context: Context) -> dict[str, Any]:
    runtime = dict(context.cycle["runtime"])
    if str(runtime["training_service"]) not in context.allowed_services:
        raise RuntimeError("training service is outside exact allowlist")
    return {
        "active_specialist": context.active_specialist,
        "cycle_contract": str(context.cycle_path),
        "cycle_contract_sha256": context.cycle_digest,
        "allowed_services": sorted(context.allowed_services),
        "kaggle_submission_queue_blocks_transition": False,
    }


def _assert_active_training_stopped(context: Context) -> dict[str, Any]:
    service = str(context.cycle["runtime"]["training_service"])
    if service not in context.allowed_services:
        raise RuntimeError("training service is outside exact allowlist")
    if service_active(service):
        raise RuntimeError("active specialist trainer has not terminated")
    return {"training_service": service, "active": False}


def _execute_existing_handoff(context: Context) -> dict[str, Any]:
    # The imported implementation contains its own filesystem lock, immutable
    # checkpoint checks, phase receipts, and exact systemd service validation.
    from scripts import run_specialist_cycle_handoff as handoff_module

    implementation = Path(str(handoff_module.__file__)).resolve()
    if not implementation.is_relative_to(ROOT):
        raise RuntimeError(
            "specialist handoff implementation escaped the stable runtime "
            f"pointer: {implementation}"
        )

    result = handoff_module.run(context.cycle_path)
    if result:
        raise RuntimeError(f"specialist cycle handoff failed: {result}")
    return {
        "cycle_contract": str(context.cycle_path),
        "cycle_contract_sha256": context.cycle_digest,
        "completed": True,
    }


DEFAULT_HANDLERS: dict[str, Callable[[Context], dict[str, Any]]] = {
    "validate_cycle_contract": _validate_cycle_contract,
    "assert_active_training_stopped": _assert_active_training_stopped,
    "execute_existing_idempotent_handoff": _execute_existing_handoff,
}


def inspect(
    context: Context,
    nodes: list[Node],
) -> dict[str, Any]:
    state = _load_state(context)
    transition = dict(
        (state.get("transitions") or {}).get(context.transition_id) or {}
    )
    receipts = dict(transition.get("receipts") or {})
    status_rows: list[dict[str, Any]] = []
    valid_receipts: dict[str, Any] = {}
    invalidated = False
    for node in nodes:
        expected = _node_input_digest(context, node, valid_receipts)
        receipt = dict(receipts.get(node.identifier) or {})
        valid = (
            not invalidated
            and receipt.get("schema") == RECEIPT_SCHEMA
            and receipt.get("node_id") == node.identifier
            and receipt.get("action") == node.action
            and receipt.get("status") == "complete"
            and receipt.get("input_sha256") == expected
        )
        if valid:
            valid_receipts[node.identifier] = receipt
        else:
            invalidated = True
        status_rows.append(
            {
                "id": node.identifier,
                "action": node.action,
                "status": "complete" if valid else "pending",
                "input_sha256": expected,
            }
        )
    return {
        "schema": "poke_bot.specialist_transition_graph_status/v1",
        "graph_id": context.graph["graph_id"],
        "graph_sha256": context.graph_digest,
        "active_specialist": context.active_specialist,
        "transition_id": context.transition_id,
        "nodes": status_rows,
    }


def run(
    context: Context,
    nodes: list[Node],
    *,
    dry_run: bool = False,
    handlers: dict[str, Callable[[Context], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if dry_run:
        result = inspect(context, nodes)
        result["dry_run"] = True
        return result
    handlers = handlers or DEFAULT_HANDLERS
    if set(handlers) != ALLOWED_ACTIONS:
        raise RuntimeError("action handler allowlist changed")
    lock_path = context.state_path.with_suffix(context.state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("specialist transition graph is already running") from exc
        state = _load_state(context)
        transitions = state["transitions"]
        transition = dict(transitions.get(context.transition_id) or {})
        if transition and (
            transition.get("active_specialist") != context.active_specialist
            or transition.get("cycle_contract_sha256") != context.cycle_digest
        ):
            raise RuntimeError("transition journal identity changed")
        transition.update(
            {
                "active_specialist": context.active_specialist,
                "cycle_contract": str(context.cycle_path),
                "cycle_contract_sha256": context.cycle_digest,
            }
        )
        receipts = dict(transition.get("receipts") or {})
        valid_receipts: dict[str, Any] = {}
        invalidate_rest = False
        for node in nodes:
            expected = _node_input_digest(context, node, valid_receipts)
            existing = dict(receipts.get(node.identifier) or {})
            reusable = (
                not invalidate_rest
                and existing.get("schema") == RECEIPT_SCHEMA
                and existing.get("node_id") == node.identifier
                and existing.get("action") == node.action
                and existing.get("status") == "complete"
                and existing.get("input_sha256") == expected
            )
            if reusable:
                valid_receipts[node.identifier] = existing
                continue
            invalidate_rest = True
            for dependency in node.dependencies:
                if dependency not in valid_receipts:
                    raise RuntimeError(
                        f"node dependency is incomplete: {node.identifier} <- {dependency}"
                    )
            try:
                output = handlers[node.action](context)
            except Exception as exc:
                receipts[node.identifier] = {
                    "schema": RECEIPT_SCHEMA,
                    "node_id": node.identifier,
                    "action": node.action,
                    "status": "failed",
                    "input_sha256": expected,
                    "failed_at": _now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                transition["receipts"] = receipts
                transition["status"] = "failed"
                transitions[context.transition_id] = transition
                _atomic(context.state_path, state)
                raise
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "node_id": node.identifier,
                "action": node.action,
                "status": "complete",
                "input_sha256": expected,
                "completed_at": _now(),
                "output": output,
                "output_sha256": _digest(output),
            }
            receipts[node.identifier] = receipt
            valid_receipts[node.identifier] = receipt
            transition["receipts"] = receipts
            transition["status"] = (
                "complete" if node is nodes[-1] else "in_progress"
            )
            transitions[context.transition_id] = transition
            _atomic(context.state_path, state)
        return inspect(context, nodes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--cycle-contract", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    context, nodes = build_context(
        graph_path=args.graph,
        cycle_path=args.cycle_contract,
        state_path=args.state,
    )
    result = inspect(context, nodes) if args.status else run(
        context, nodes, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
