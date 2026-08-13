from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .errors import GoalError, ValidationError
from .resolve import resolve_gateway
from .writer import (
    activate_amendment,
    chain_goal,
    initialize_goal_package,
    materialize_status,
    record_evidence,
    record_amendment,
)
from .workflow import (
    add_dependency,
    add_goal_node,
    claim_next_prompt,
    initialize_workflow,
    next_prompts,
    remove_dependency,
    remove_goal_node,
    release_claim,
    resolve_workflow,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dgoal", description="Resolve durable goal packages")
    parser.add_argument("--version", action="version", version="dgoal 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "validate all documents and checksums"),
        ("resolve", "print the normalized active and desired goal"),
        ("status", "print the derived non-authoritative status"),
        ("verify-evidence", "verify and list immutable evidence"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("gateway", type=Path)
    amend = subparsers.add_parser("amend", help="append an owner amendment")
    amend.add_argument("gateway", type=Path)
    amend.add_argument("--set", dest="sets", action="append", nargs=2, metavar=("POINTER", "JSON"))
    amend.add_argument("--remove", dest="removes", action="append", metavar="POINTER")
    amend.add_argument("--expect", dest="expects", action="append", nargs=2, metavar=("POINTER", "JSON"))
    amend.add_argument("--reason", required=True)
    amend.add_argument(
        "--activation-mode",
        choices=("manual", "immediate", "next_safe_boundary"),
        default="manual",
    )
    amend.add_argument(
        "--when",
        help="JSON activation predicate; required for next_safe_boundary",
    )
    amend.add_argument("--authority", default="owner")
    activate = subparsers.add_parser("activate", help="activate the next pending amendment")
    activate.add_argument("gateway", type=Path)
    activate.add_argument("revision", type=int)
    activate.add_argument("--evidence-id", action="append")
    materialize = subparsers.add_parser("materialize-status", help="regenerate STATUS.json")
    materialize.add_argument("gateway", type=Path)
    initialize = subparsers.add_parser("init", help="create a new draft goal package")
    initialize.add_argument("directory", type=Path)
    initialize.add_argument("--goal-id", required=True)
    initialize.add_argument("--objective", required=True)
    chain = subparsers.add_parser("chain", help="run a successor goal after completion")
    chain.add_argument("source_gateway", type=Path)
    chain.add_argument("successor_gateway", type=Path)
    chain.add_argument("--transition-id", required=True)
    chain.add_argument("--reason", required=True)

    evidence = subparsers.add_parser("evidence", help="manage immutable evidence")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_add = evidence_commands.add_parser("add", help="copy and index JSON evidence")
    evidence_add.add_argument("gateway", type=Path)
    evidence_add.add_argument("evidence_id")
    evidence_add.add_argument("source", type=Path)
    evidence_add.add_argument("--contract-revision", type=int)

    workflow = subparsers.add_parser("workflow", help="manage a goal DAG")
    workflow_commands = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_init = workflow_commands.add_parser("init", help="create an empty workflow")
    workflow_init.add_argument("workflow", type=Path)
    workflow_init.add_argument("--workflow-id", required=True)
    workflow_add = workflow_commands.add_parser("add-goal", help="add a goal node")
    workflow_add.add_argument("workflow", type=Path)
    workflow_add.add_argument("gateway", type=Path)
    workflow_add.add_argument("--node-id", required=True)
    workflow_depend = workflow_commands.add_parser("depend", help="add a dependency edge")
    workflow_depend.add_argument("workflow", type=Path)
    workflow_depend.add_argument("source")
    workflow_depend.add_argument("target")
    workflow_depend.add_argument("--edge-id", required=True)
    workflow_remove_goal = workflow_commands.add_parser(
        "remove-goal", help="remove a node and its edges"
    )
    workflow_remove_goal.add_argument("workflow", type=Path)
    workflow_remove_goal.add_argument("node_id")
    workflow_remove_edge = workflow_commands.add_parser(
        "remove-dependency", help="remove one dependency edge"
    )
    workflow_remove_edge.add_argument("workflow", type=Path)
    workflow_remove_edge.add_argument("edge_id")
    workflow_claim = workflow_commands.add_parser(
        "claim", help="atomically claim one ready goal prompt"
    )
    workflow_claim.add_argument("workflow", type=Path)
    workflow_claim.add_argument("--claimant", required=True)
    workflow_release = workflow_commands.add_parser(
        "release", help="release a goal claim"
    )
    workflow_release.add_argument("workflow", type=Path)
    workflow_release.add_argument("node_id")
    workflow_release.add_argument("--claimant", required=True)
    for name, help_text in (
        ("validate", "validate the DAG and every goal package"),
        ("status", "derive blocked, ready, and completed nodes"),
        ("next", "emit the next eligible goal prompt"),
    ):
        command = workflow_commands.add_parser(name, help=help_text)
        command.add_argument("workflow", type=Path)
        if name == "next":
            command.add_argument("--all", action="store_true", dest="all_ready")
    return parser


def _json_argument(raw: str, *, label: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} must be valid JSON: {exc.msg}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "workflow":
            if args.workflow_command == "init":
                workflow_path = initialize_workflow(
                    args.workflow, workflow_id=args.workflow_id
                )
                payload = {"workflow": str(workflow_path)}
            elif args.workflow_command == "add-goal":
                add_goal_node(args.workflow, args.gateway, node_id=args.node_id)
                payload = resolve_workflow(args.workflow).to_dict()
            elif args.workflow_command == "depend":
                add_dependency(
                    args.workflow,
                    args.source,
                    args.target,
                    edge_id=args.edge_id,
                )
                payload = resolve_workflow(args.workflow).to_dict()
            elif args.workflow_command == "remove-goal":
                remove_goal_node(args.workflow, args.node_id)
                payload = resolve_workflow(args.workflow).to_dict()
            elif args.workflow_command == "remove-dependency":
                remove_dependency(args.workflow, args.edge_id)
                payload = resolve_workflow(args.workflow).to_dict()
            elif args.workflow_command == "claim":
                payload = {
                    "workflow_id": resolve_workflow(args.workflow).workflow_id,
                    "claim": claim_next_prompt(
                        args.workflow, claimant=args.claimant
                    ),
                }
            elif args.workflow_command == "release":
                release_claim(
                    args.workflow, args.node_id, claimant=args.claimant
                )
                payload = resolve_workflow(args.workflow).to_dict()
            elif args.workflow_command == "next":
                payload = {
                    "workflow_id": resolve_workflow(args.workflow).workflow_id,
                    "prompts": next_prompts(
                        args.workflow, all_ready=args.all_ready
                    ),
                }
            else:
                workflow_resolution = resolve_workflow(args.workflow)
                payload = workflow_resolution.to_dict()
                if args.workflow_command == "validate":
                    payload = {
                        "valid": True,
                        "workflow_id": workflow_resolution.workflow_id,
                        "revision": workflow_resolution.revision,
                        "nodes": len(workflow_resolution.nodes),
                        "edges": len(workflow_resolution.edges),
                    }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "evidence":
            resolution = record_evidence(
                args.gateway,
                args.evidence_id,
                args.source,
                contract_revision=args.contract_revision,
            )
            print(json.dumps(resolution.status, indent=2, sort_keys=True))
            return 0
        if args.command == "init":
            gateway_path = initialize_goal_package(
                args.directory, goal_id=args.goal_id, objective=args.objective
            )
            print(json.dumps({"gateway": str(gateway_path)}, indent=2, sort_keys=True))
            return 0
        if args.command == "chain":
            resolution = chain_goal(
                args.source_gateway,
                args.successor_gateway,
                transition_id=args.transition_id,
                reason=args.reason,
            )
        elif args.command == "amend":
            expects: dict[str, object] = {}
            for pointer, raw in args.expects or []:
                expects[pointer] = _json_argument(raw, label=f"--expect {pointer}")
            operations: list[dict[str, object]] = []
            for pointer, raw in args.sets or []:
                operation: dict[str, object] = {
                    "op": "set",
                    "path": pointer,
                    "value": _json_argument(raw, label=f"--set {pointer}"),
                }
                if pointer in expects:
                    operation["expect"] = expects.pop(pointer)
                operations.append(operation)
            for pointer in args.removes or []:
                operation = {"op": "remove", "path": pointer}
                if pointer in expects:
                    operation["expect"] = expects.pop(pointer)
                operations.append(operation)
            if expects:
                raise ValidationError(
                    "--expect path has no matching --set/--remove: "
                    + ", ".join(sorted(expects))
                )
            resolution = record_amendment(
                args.gateway,
                operations=operations,
                reason=args.reason,
                activation_mode=args.activation_mode,
                activation_condition=(
                    _json_argument(args.when, label="--when")
                    if args.when is not None
                    else None
                ),
                authority=args.authority,
            )
        elif args.command == "activate":
            resolution = activate_amendment(
                args.gateway, args.revision, evidence_ids=args.evidence_id
            )
        elif args.command == "materialize-status":
            status_path = materialize_status(args.gateway)
            print(json.dumps({"written": str(status_path)}, indent=2, sort_keys=True))
            return 0
        else:
            resolution = resolve_gateway(args.gateway)
    except GoalError as exc:
        print(f"dgoal: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        payload = {
            "valid": True,
            "goal_id": resolution.goal_id,
            "current_revision": resolution.current_revision,
            "active_revision": resolution.active_revision,
        }
    elif args.command in {"status", "amend", "activate"}:
        payload = resolution.status
    elif args.command == "verify-evidence":
        payload = {
            "valid": True,
            "goal_id": resolution.goal_id,
            "evidence_ids": sorted(resolution.evidence),
        }
    else:
        payload = resolution.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
