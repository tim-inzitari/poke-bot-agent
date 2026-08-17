#!/usr/bin/env python3
"""Add/list/remove elastic simulator remotes with atomic registry updates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.remote_fleet_gateway import FleetManifest, FleetManifestError
from poke_bot.remote_fleet_registry import (
    DEFAULT_MANIFEST,
    DEFAULT_TRAINER_ROOT,
    DEFAULT_WORKER_ROOT,
    add_backends,
    discover_backend,
    read_registry,
    remove_backends,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            os.environ.get("POKEBOT_REMOTE_FLEET_MANIFEST", str(DEFAULT_MANIFEST))
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="preflight and atomically add IP[:port] values")
    add.add_argument("endpoints", nargs="+")
    add.add_argument("--port", type=int, default=8765)
    add.add_argument("--checkpoint-path")
    add.add_argument("--capacity", type=int)
    add.add_argument("--id", help="explicit ID; valid only with one endpoint")
    add.add_argument("--timeout", type=float, default=10.0)
    add.add_argument("--replace", action="store_true")
    add.add_argument("--dry-run", action="store_true")
    add.add_argument("--trainer-root", default=DEFAULT_TRAINER_ROOT)
    add.add_argument("--worker-root", default=DEFAULT_WORKER_ROOT)
    add.add_argument(
        "--no-path-rewrite",
        action="store_true",
        help="do not add the standard Inzi-to-worker repository path rewrite",
    )

    listing = sub.add_parser("list", help="show the current registry")
    listing.add_argument("--json", action="store_true")

    remove = sub.add_parser("remove", help="remove backend IDs or endpoints")
    remove.add_argument("selectors", nargs="+")
    remove.add_argument("--dry-run", action="store_true")

    sub.add_parser("check", help="strictly validate the registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            manifest = FleetManifest.load(args.manifest)
            result = {
                "ok": True,
                "manifest": str(manifest.path),
                "manifest_sha256": manifest.identity,
                "activation_allowed": manifest.activation_allowed,
                "backend_ids": [backend.id for backend in manifest.backends],
            }
        elif args.command == "list":
            registry = read_registry(args.manifest)
            if args.json:
                print(json.dumps(registry, indent=2))
                return 0
            rows = list(registry.get("backends") or [])
            if not rows:
                print("No remotes registered.")
                return 0
            print("ID\tENDPOINT\tENABLED\tCAPACITY\tCHECKPOINT")
            for row in rows:
                print(
                    f"{row.get('id')}\t{row.get('endpoint')}\t"
                    f"{row.get('enabled')}\t{row.get('capacity')}\t"
                    f"{str(row.get('required_checkpoint_digest') or '')[:19]}…"
                )
            return 0
        elif args.command == "remove":
            result = remove_backends(
                args.manifest, args.selectors, dry_run=args.dry_run
            )
        else:
            if args.id and len(args.endpoints) != 1:
                raise FleetManifestError("--id is valid only with one endpoint")
            discovered = []
            for raw_endpoint in args.endpoints:
                endpoint = (
                    raw_endpoint
                    if ":" in raw_endpoint
                    else f"{raw_endpoint}:{args.port}"
                )
                discovered.append(
                    discover_backend(
                        endpoint,
                        checkpoint_path=args.checkpoint_path,
                        capacity=args.capacity,
                        backend_id=args.id,
                        timeout_s=args.timeout,
                        trainer_root=(None if args.no_path_rewrite else args.trainer_root),
                        worker_root=(None if args.no_path_rewrite else args.worker_root),
                    )
                )
            result = add_backends(
                args.manifest,
                discovered,
                replace=args.replace,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except FleetManifestError as exc:
        print(f"remote-fleet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
