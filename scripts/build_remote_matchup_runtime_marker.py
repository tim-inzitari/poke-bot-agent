#!/usr/bin/env python3
"""Build the strict marker consumed by LAN remote-worker startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from poke_bot.matchup_adapter_routes import (  # noqa: E402
    require_runtime_route_binding,
    resolve_matchup_adapter_route_contract,
)
from poke_bot.public_matchup_router import PublicMatchupDecisionTree  # noqa: E402


SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def build(checkpoint_path: Path, tree_path: Path, output: Path) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    tree_path = tree_path.expanduser().resolve()
    output = output.expanduser().resolve()
    tree = PublicMatchupDecisionTree.from_path(
        tree_path, require_runtime_enabled=True
    )
    saved = checkpoint.load_checkpoint(checkpoint_path, map_location="cpu")
    fit = dict((saved.get("extra") or {}).get("dormant_matchup_adapter_fit") or {})
    route_decisions = {
        str(key): int(value)
        for key, value in dict(fit.get("route_decisions") or {}).items()
    }
    accepted = sorted(tree.runtime_accepted_archetype_ids)
    runtime = json.loads(tree_path.read_text(encoding="utf-8")).get(
        "runtime_contract", {}
    )
    extra = dict(saved.get("extra") or {})
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    adapter_config = dict(extra.get("matchup_adapter_config") or {})
    route_contract = resolve_matchup_adapter_route_contract(adapter_config)
    if tuple(tree.targets) != route_contract.target_ids:
        raise RuntimeError("checkpoint and tree route rosters differ")
    try:
        require_runtime_route_binding(
            runtime,
            route_contract,
            allow_legacy_v5=True,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    dormant_config = dict(dormant.get("adapter_config") or {})
    zero_materialized = bool(
        runtime.get("zero_materialized_adapters_allowed") is True
        and dormant.get("schema") == "poke_bot.zero_dormant_matchup_adapter/v1"
        and dormant.get("zero_output") is True
        and dormant.get("runtime_enabled") is False
        and set(accepted).issubset(set(route_contract.target_ids))
        and (not dormant_config or dormant_config == adapter_config)
    )
    trained = bool(
        fit.get("schema") == "poke_bot.dormant_matchup_adapter_fit/v1"
        and set(accepted).issubset(set(route_contract.target_ids))
        and all(route_decisions.get(route, 0) > 0 for route in accepted)
    )
    if not (
        (trained or zero_materialized)
        and accepted
        and str(runtime.get("checkpoint_digest") or "") == _digest(checkpoint_path)
        and runtime.get("one_route_per_decision") is True
        and runtime.get("unknown_route_exact_bypass") is True
        and int(runtime.get("consecutive_required") or 0) >= 1
    ):
        raise RuntimeError("checkpoint/tree pair is not safe for remote activation")
    payload = {
        "schema": SCHEMA,
        "runtime_enabled": True,
        # The deployment copies the tree beside its active checkpoint.
        "tree_file": tree_path.name,
        "tree_digest": _digest(tree_path),
        "accepted_archetype_ids": accepted,
        "continuous_reevaluation": True,
        "one_route_per_decision": True,
        "zero_materialized_adapters_allowed": zero_materialized,
        **route_contract.runtime_binding(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.checkpoint, args.tree, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
