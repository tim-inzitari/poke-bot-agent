#!/usr/bin/env python3
"""Create the explicit marker consumed by LAN remote-worker deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.public_matchup_router import PublicMatchupDecisionTree


SCHEMA = "poke_bot.remote_matchup_runtime_activation/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = args.tree.expanduser().resolve()
    raw = tree.read_bytes()
    loaded = PublicMatchupDecisionTree.from_path(tree, require_runtime_enabled=True)
    tree_payload = json.loads(raw)
    runtime = dict(tree_payload.get("runtime_contract") or {})
    payload = {
        "schema": SCHEMA,
        "runtime_enabled": True,
        "tree_file": tree.name,
        "tree_digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "accepted_archetype_ids": sorted(loaded.runtime_accepted_archetype_ids),
        "continuous_reevaluation": True,
        "one_route_per_decision": True,
        "zero_materialized_adapters_allowed": bool(
            runtime.get("zero_materialized_adapters_allowed")
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior != payload:
            raise RuntimeError("existing remote runtime marker conflicts")
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
