#!/usr/bin/env python3
"""Validate a complete specialist-corpus split before runtime admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest
from scripts.filter_feature_manifest import sha256
from scripts.run_starmie_expert_bootstrap import TARGETS
from scripts.split_expert_manifest_by_archetype import (
    READY_SCHEMA,
    UNAVAILABLE_SCHEMA,
)


def validate_corpora(
    ready_path: Path,
    *,
    required_archetypes: tuple[str, ...] | list[str],
    required_compact_mode: str = "temporal-expert-v1",
    required_target_coverage: tuple[str, ...] = TARGETS,
) -> dict[str, Any]:
    ready_path = Path(ready_path).expanduser().resolve()
    payload = json.loads(ready_path.read_text(encoding="utf-8"))
    required = tuple(
        dict.fromkeys(
            str(value).strip().casefold() for value in required_archetypes
        )
    )
    identity = dict(payload.get("identity") or {})
    minimum = int(identity.get("minimum_decisions", 0))
    rows = [dict(row) for row in (payload.get("results") or [])]
    by_id = {str(row.get("archetype") or ""): row for row in rows}
    if (
        payload.get("schema") != READY_SCHEMA
        or not required
        or any(not value for value in required)
        or tuple(identity.get("archetypes") or ()) != required
        or minimum <= 0
        or len(rows) != len(by_id)
        or set(by_id) != set(required)
    ):
        raise RuntimeError("specialist corpus aggregate identity changed")
    counts = {"ready": 0, "insufficient_decisions": 0, "unavailable": 0}
    validated: list[dict[str, Any]] = []
    for archetype in required:
        row = by_id[archetype]
        status = str(row.get("status") or "")
        if status == "unavailable":
            unavailable = ready_path.parent / archetype / (
                "UNAVAILABLE_EXPERT_CORPUS.json"
            )
            receipt = json.loads(unavailable.read_text(encoding="utf-8"))
            if (
                receipt.get("schema") != UNAVAILABLE_SCHEMA
                or receipt.get("archetype") != archetype
                or int(row.get("records", -1)) != 0
                or int(row.get("decisions", -1)) != 0
            ):
                raise RuntimeError(
                    f"unavailable specialist receipt changed: {archetype}"
                )
            counts[status] += 1
            validated.append(
                {"archetype": archetype, "status": status, "decisions": 0}
            )
            continue
        if status not in {"ready", "insufficient_decisions"}:
            raise RuntimeError(f"unknown specialist corpus status: {archetype}")
        relative = Path(str(row.get("protected_corpus") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"specialist corpus pointer is not relocatable: {archetype}"
            )
        pointer = (ready_path.parent / relative).resolve()
        corpus = resolve_expert_manifest(
            pointer,
            min_decisions=1,
            require_protected=True,
            required_archetype=archetype,
            required_compact_mode=required_compact_mode,
            required_max_context=320,
            required_target_coverage=required_target_coverage,
        )
        expected_status = (
            "ready" if corpus.decisions >= minimum else "insufficient_decisions"
        )
        if (
            status != expected_status
            or corpus.records != int(row.get("records", -1))
            or corpus.decisions != int(row.get("decisions", -1))
            or sha256(Path(corpus.path))
            != str(row.get("manifest_sha256") or "")
        ):
            raise RuntimeError(
                f"specialist corpus evidence changed: {archetype}"
            )
        counts[status] += 1
        validated.append(
            {
                "archetype": archetype,
                "status": status,
                "records": corpus.records,
                "decisions": corpus.decisions,
                "manifest": corpus.path,
                "manifest_sha256": corpus.digest,
            }
        )
    return {
        "schema": "poke_bot.specialist_expert_corpora_validation/v1",
        "ready": str(ready_path),
        "ready_sha256": sha256(ready_path),
        "minimum_decisions": minimum,
        "counts": counts,
        "results": validated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--archetype", action="append", required=True)
    args = parser.parse_args()
    result = validate_corpora(
        args.ready,
        required_archetypes=args.archetype,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
