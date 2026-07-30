#!/usr/bin/env python3
"""Create the deterministic source lock required by the Archaludon schema-7 build.

This does not build, import, promote, or authorize a corpus. It is intended to
run only after the revision-56+ source tree is frozen on the preparation host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from materialize_archaludon_ex_full_public_schema7_corpus import (
    DATASET_SCHEMA,
    END,
    FEATURE_SCHEMA,
    MINIMUM_MATCHING_GAMES,
    REQUIRED_SOURCE_FILES,
    SOURCE_LOCK_SCHEMA,
    START,
    TARGET,
    _canonical_digest,
    _days,
    _read_json,
    _sha256,
)


def _goal_revision(path: Path) -> int:
    match = re.search(
        r"^Revision:\s*`(\d+)`\s*$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("GOAL.md revision is missing")
    return int(match.group(1))


def _classifier_digest(source_root: Path) -> str:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from poke_bot.ladder_replay import LadderReplayClassifier

    roster = _read_json(source_root / "state/matchup_adapter_roster.json")
    expert_ids = list(roster.get("expert_ids") or ())
    required = int(roster.get("required_specialist_count", -1))
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or required <= 0
        or len(expert_ids) != required
        or len(set(expert_ids)) != required
        or TARGET not in expert_ids
    ):
        raise RuntimeError("matchup classifier roster is invalid")
    classifier = LadderReplayClassifier.from_paths(
        source_root / "data/training_mixes/top_ladder.v1.json",
        source_root
        / "data/training_mixes/top_ladder_representatives.v1.json",
        card_csv=source_root / "cards/EN_Card_Data.csv",
        additive_registered_ids=expert_ids,
    )
    return _canonical_digest(classifier.contract)


def _validate_audit(
    path: Path,
    classifier_sha256: str,
    roster_sha256: str,
    required_specialist_count: int,
) -> None:
    audit = _read_json(path)
    rows = list(audit.get("daily_sources") or ())
    method = dict(audit.get("audit_method") or {})
    total = sum(int(row.get("matching_acting_seats", -1)) for row in rows)
    methods = sum(
        int(value) for value in dict(audit.get("label_methods") or {}).values()
    )
    if not (
        audit.get("schema")
        == "poke_bot.archaludon_ex_public_source_audit/v1"
        and audit.get("status")
        == "source_audit_complete_schema7_rematerialization_required"
        and audit.get("date_start") == START.isoformat()
        and audit.get("date_end") == END.isoformat()
        and int(audit.get("days", -1)) == len(_days())
        and [row.get("date") for row in rows] == _days()
        and total == int(audit.get("matching_acting_seats", -1))
        and methods == total
        and total >= MINIMUM_MATCHING_GAMES
        and method.get("classifier_contract_sha256")
        == classifier_sha256
        and method.get("classifier_roster_sha256") == roster_sha256
        and int(method.get("classifier_required_specialist_count", -1))
        == required_specialist_count
    ):
        raise RuntimeError("source audit is stale or invalid")


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable source lock differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "state/archaludon_ex_schema7_source_lock_v1.json",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=root / "state/archaludon_public_full44_source_audit_v1.json",
    )
    parser.add_argument(
        "--assembler",
        type=Path,
        default=root / "scripts/assemble_feature_manifest.py",
    )
    parser.add_argument(
        "--cg-runtime",
        type=Path,
        default=Path(
            "/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1"
        ),
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    files = {
        relative: _sha256(source_root / relative)
        for relative in REQUIRED_SOURCE_FILES
    }
    classifier_sha256 = _classifier_digest(source_root)
    roster_path = source_root / "state/matchup_adapter_roster.json"
    roster = _read_json(roster_path)
    roster_sha256 = _sha256(roster_path)
    required_specialist_count = int(
        roster.get("required_specialist_count", -1)
    )
    source_audit = args.source_audit.expanduser().resolve()
    _validate_audit(
        source_audit,
        classifier_sha256,
        roster_sha256,
        required_specialist_count,
    )
    if (
        _sha256(source_audit)
        != files["state/archaludon_public_full44_source_audit_v1.json"]
    ):
        raise RuntimeError("selected source audit is not the locked audit file")
    revision = _goal_revision(source_root / "GOAL.md")
    if revision < 56:
        raise RuntimeError("Archaludon source lock requires GOAL revision 56+")
    value = {
        "schema": SOURCE_LOCK_SCHEMA,
        "status": "locked_checksum_validated",
        "goal_revision": revision,
        "date_start": START.isoformat(),
        "date_end": END.isoformat(),
        "days": len(_days()),
        "minimum_matching_games": MINIMUM_MATCHING_GAMES,
        "dataset_schema": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "classifier_sha256": classifier_sha256,
        "source_audit_sha256": _sha256(source_audit),
        "files": files,
        "assembler_sha256": _sha256(args.assembler.expanduser().resolve()),
        "cg_library_sha256": _sha256(
            args.cg_runtime.expanduser().resolve() / "cg/libcg.so"
        ),
        "authorizes_materialization": True,
        "authorizes_import_or_training": False,
    }
    _immutable_json(args.output.expanduser().resolve(), value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
