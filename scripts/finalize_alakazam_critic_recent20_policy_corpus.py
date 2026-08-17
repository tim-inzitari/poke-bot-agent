#!/usr/bin/env python3
"""Create a receipt-bound July-23..August-11 policy corpus without reweighting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "poke_bot.alakazam_prize_plan_v2_recent20_policy_corpus_receipt/v1"
DATES = tuple(
    [f"2026-07-{day:02d}" for day in range(23, 32)]
    + [f"2026-08-{day:02d}" for day in range(1, 12)]
)


class Recent20PolicyCorpusError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise Recent20PolicyCorpusError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Recent20PolicyCorpusError(f"{label} must contain an object")
    return value


def _write(path: Path, value: Any) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, data); os.fsync(fd)
    finally:
        os.close(fd)
    return _sha(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_root).expanduser().resolve()
    source_receipt = _json(args.source_transfer_receipt, "source transfer receipt")
    if source_receipt.get("destination_root") != str(source):
        raise Recent20PolicyCorpusError("source receipt/root mismatch")
    old_manifest_path = source / "manifest.json"
    old_manifest = _json(old_manifest_path, "source manifest")
    if _sha(old_manifest_path) != source_receipt.get("manifest_sha256"):
        raise Recent20PolicyCorpusError("source manifest digest mismatch")
    old_by_day: dict[str, dict[str, Any]] = {}
    for row in old_manifest.get("shards") or ():
        if not isinstance(row, dict) or len(row.get("source_dates") or ()) != 1:
            raise Recent20PolicyCorpusError("source manifest row is malformed")
        old_by_day[str(row["source_dates"][0])] = dict(row)

    aug11 = Path(args.aug11_shard).expanduser().resolve()
    aug11_meta = _json(Path(str(aug11) + ".json"), "August-11 shard metadata")
    if (
        _sha(aug11) != aug11_meta.get("sha256")
        or aug11.stat().st_size != aug11_meta.get("bytes")
        or aug11_meta.get("compact_mode") != "temporal-expert-v1"
        or aug11_meta.get("required_archetype") != "alakazam"
        or aug11_meta.get("source_dates") != ["2026-08-11"]
    ):
        raise Recent20PolicyCorpusError("August-11 shard metadata/identity mismatch")
    aug11_row = dict(aug11_meta)
    aug11_row["path"] = aug11.name
    selected: list[dict[str, Any]] = []
    for day in DATES:
        row = aug11_row if day == "2026-08-11" else old_by_day.get(day)
        if row is None:
            raise Recent20PolicyCorpusError(f"source corpus lacks required day {day}")
        selected.append(dict(row))

    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise Recent20PolicyCorpusError("recent-20 output root is create-only")
    output.mkdir(parents=True)
    for row in selected:
        day = str(row["source_dates"][0])
        source_path = aug11 if day == "2026-08-11" else source / str(row["path"])
        if _sha(source_path) != row.get("sha256") or source_path.stat().st_size != row.get("bytes"):
            raise Recent20PolicyCorpusError(f"source shard identity mismatch for {day}")
        os.link(source_path, output / str(row["path"]))
    stats = [dict(row.get("stats") or {}) for row in selected]
    manifest = {
        "format": "pokebot-bootstrap-feature-manifest",
        "format_version": 1,
        "compact_mode": "temporal-expert-v1",
        "date_start": DATES[0],
        "date_end": DATES[-1],
        "dates": list(DATES),
        "max_context": 320,
        "selection": {
            "field": "GameSequence.archetype",
            "operator": "exact_casefold",
            "opponent_routes_only": False,
            "seat_semantics": "acting_seat_only",
            "value": "alakazam",
        },
        "shards": selected,
        "totals": {
            "bytes": sum(int(row["bytes"]) for row in selected),
            "records_kept": sum(int(item.get("records_kept", 0)) for item in stats),
            "decisions_kept": sum(int(item.get("decisions_kept", 0)) for item in stats),
        },
        "critic_lane_derived_recent20": True,
        "replay_membership_or_weights_changed_within_evaluation_arms": False,
    }
    manifest_sha = _write(output / "manifest.json", manifest)
    receipt = {
        "schema": SCHEMA,
        "destination_root": str(output),
        "source_mutated": False,
        "source_transfer_receipt_sha256": _sha(Path(args.source_transfer_receipt)),
        "aug11_source_jsonl_sha256": args.aug11_source_jsonl_sha256,
        "aug11_shard_sha256": aug11_meta["sha256"],
        "manifest_sha256": manifest_sha,
        "dates": list(DATES),
        "day_count": 20,
        "whole_day_membership": True,
        "hardlink_byte_identity_preserved": True,
        "replay_sampling_or_weights_changed": False,
        "actor_activation": False,
    }
    receipt_sha = _write(output / "receipt.json", receipt)
    return {"root": str(output), "manifest_sha256": manifest_sha, "receipt_sha256": receipt_sha}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-transfer-receipt", type=Path, required=True)
    p.add_argument("--aug11-shard", type=Path, required=True)
    p.add_argument("--aug11-source-jsonl-sha256", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), sort_keys=True))
    except (Recent20PolicyCorpusError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
