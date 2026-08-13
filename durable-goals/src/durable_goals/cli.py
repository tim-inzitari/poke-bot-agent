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
    record_amendment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dgoal", description="Resolve durable goal packages")
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
    amend.add_argument("--activation-mode", default="next_safe_boundary")
    amend.add_argument("--authority", default="owner")
    activate = subparsers.add_parser("activate", help="activate the next pending amendment")
    activate.add_argument("gateway", type=Path)
    activate.add_argument("revision", type=int)
    activate.add_argument("--evidence-id")
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
    return parser


def _json_argument(raw: str, *, label: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} must be valid JSON: {exc.msg}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
                authority=args.authority,
            )
        elif args.command == "activate":
            resolution = activate_amendment(
                args.gateway, args.revision, evidence_id=args.evidence_id
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
