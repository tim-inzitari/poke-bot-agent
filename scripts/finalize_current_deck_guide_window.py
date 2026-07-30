#!/usr/bin/env python3
"""Seal a completed current-deck-guide feature window for specialist bootstrap.

The command is intentionally safe to run from a timer while materialization is
still active.  It exits successfully without publishing anything until the
window status says every expected day is complete.  Once ready, it validates
the specialist and guide identity of every receipt, invokes the established
feature-manifest assembler, and writes one immutable readiness receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
from typing import Any


READY_SCHEMA = "poke_bot.current_deck_guide_corpus_ready/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-version", required=True)
    parser.add_argument("--minimum-records", type=int, default=0)
    parser.add_argument(
        "--public-deck-catalog",
        type=Path,
        help=(
            "Catalog copied into the output directory for an owner-authorized "
            "public full-history exact-deck source contract."
        ),
    )
    parser.add_argument("--expected-max-context", type=int, default=320)
    parser.add_argument(
        "--assembler",
        type=Path,
        default=Path(__file__).with_name("assemble_feature_manifest.py"),
    )
    args = parser.parse_args()

    expected_dates = _dates(args.start, args.end)
    status = _load(args.status.resolve())
    if status.get("state") != "complete":
        print(
            json.dumps(
                {
                    "status": "not_ready",
                    "window_state": status.get("state"),
                    "completed_days": len(status.get("completed") or []),
                    "expected_days": len(expected_dates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    completed = list(status.get("completed") or [])
    completed_dates = [str(row.get("date")) for row in completed]
    if completed_dates != expected_dates:
        raise RuntimeError(
            f"completed date coverage changed: expected={expected_dates} "
            f"actual={completed_dates}"
        )

    out_dir = args.out_dir.resolve()
    rows: list[dict[str, Any]] = []
    compact_mode: str | None = None
    for source_date in expected_dates:
        stem = f"{args.specialist_id}-{source_date}.features"
        shard = out_dir / stem
        sidecar = out_dir / f"{stem}.json"
        receipt_path = out_dir / f"{stem}.receipt.json"
        if not shard.is_file() or not sidecar.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"incomplete day artifact set: {source_date}")
        receipt = _load(receipt_path)
        sidecar_payload = _load(sidecar)
        day_compact_mode = str(sidecar_payload.get("compact_mode") or "").strip()
        if not day_compact_mode:
            raise RuntimeError(f"guide shard compact mode missing: {sidecar}")
        if compact_mode is None:
            compact_mode = day_compact_mode
        elif day_compact_mode != compact_mode:
            raise RuntimeError(
                "guide shard compact modes differ: "
                f"expected={compact_mode!r} actual={day_compact_mode!r} "
                f"sidecar={sidecar}"
            )
        selection = dict(receipt.get("selection") or {})
        schemas = dict(receipt.get("schemas") or {})
        stats = dict(receipt.get("stats") or {})
        output = dict(receipt.get("output") or {})
        if (
            receipt.get("format") != "pokebot-authoritative-visual-day-receipt"
            or int(receipt.get("format_version", -1)) != 1
            or receipt.get("source_date") != source_date
            or selection.get("acting_seat_archetype") != args.specialist_id
            or selection.get("current_deck_guide") != args.specialist_id
            or schemas.get("guide") != args.guide_version
            or int(schemas.get("max_context", -1))
            != int(args.expected_max_context)
        ):
            raise RuntimeError(f"guide receipt identity mismatch: {receipt_path}")
        actual_digest = _sha256(shard)
        if output.get("sha256") != actual_digest:
            raise RuntimeError(f"guide shard checksum mismatch: {shard}")
        records = int(stats.get("records_kept", 0))
        decisions = int(stats.get("decisions_kept", 0))
        target_coverage = stats.get("target_coverage")
        guide_rows_present = (
            isinstance(target_coverage, dict)
            and "guide_rows" in target_coverage
        )
        guide_rows = (
            int(target_coverage["guide_rows"])
            if guide_rows_present
            else (0 if records == 0 and decisions == 0 else -1)
        )
        if guide_rows < 0:
            raise RuntimeError(
                f"nonempty guide shard is missing valid guide targets: {shard}"
            )
        rows.append(
            {
                "date": source_date,
                "records": records,
                "decisions": decisions,
                "guide_rows": guide_rows,
                "zero_guide_rows": guide_rows == 0,
                "sha256": actual_digest,
                "receipt_sha256": _sha256(receipt_path),
            }
        )

    manifest = out_dir / "manifest.json"
    pointer = out_dir / "PROTECTED_EXPERT_CORPUS.json"
    ready_path = out_dir / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    if not manifest.is_file() or not pointer.is_file():
        command = [
            sys.executable,
            str(args.assembler.resolve()),
            "--staging-dir",
            str(out_dir),
            "--out",
            str(manifest),
            "--verified-dir",
            str(out_dir / ".verified"),
            "--min-free-gib",
            "5",
            "--required-archetype",
            args.specialist_id,
            "--expected-max-context",
            str(args.expected_max_context),
            "--compact-mode",
            str(compact_mode),
            "--seal-protected",
            "--allow-empty-shards",
        ]
        for source_date in expected_dates:
            command.extend(["--expected-date", source_date])
        for target in (
            "temporal_action_rows",
            "opponent_hand_rows",
            "opponent_remainder_rows",
            "opponent_private_prize_rows",
            "lethal_threat_rows",
            "prize_race_rows",
        ):
            command.extend(["--require-target-coverage", target])
        subprocess.run(command, check=True)

    manifest_payload = _load(manifest)
    pointer_payload = _load(pointer)
    manifest_records = int(
        (manifest_payload.get("totals") or {}).get("records_kept") or 0
    )
    if args.minimum_records < 0 or manifest_records < int(args.minimum_records):
        raise RuntimeError(
            "sealed guide corpus is below the required record floor: "
            f"actual={manifest_records} required={int(args.minimum_records)}"
        )
    source_policy = None
    if args.public_deck_catalog is not None:
        catalog_path = args.public_deck_catalog.resolve()
        if catalog_path.parent != out_dir or not catalog_path.is_file():
            raise RuntimeError(
                "public deck catalog must be immutable inside the corpus root"
            )
        catalog = _load(catalog_path)
        source_window = dict(catalog.get("source_window") or {})
        catalog_identity_mode = str(
            (catalog.get("identity_contract") or {}).get("mode") or ""
        )
        if (
            catalog.get("schema")
            != "poke_bot.public_deck_archetype_catalog/v1"
            or catalog.get("specialist_id") != args.specialist_id
            or int(catalog.get("observed_acting_seat_games") or 0)
            < int(args.minimum_records)
            or source_window
            != {
                "start": expected_dates[0],
                "end": expected_dates[-1],
                "days": len(expected_dates),
            }
        ):
            raise RuntimeError("public deck catalog identity changed")
        source_mode = (
            "public_full_history_card_signature_identity"
            if catalog_identity_mode
            == "crustle_card_signature_public_replay_identity"
            else "public_full_history_exact_deck_identity"
        )
        source_policy = {
            "mode": source_mode,
            "public_deck_catalog": catalog_path.name,
            "public_deck_catalog_sha256": _sha256(catalog_path),
            "minimum_records": int(args.minimum_records),
            "source_window": source_window,
        }
        existing_policy = pointer_payload.get("source_policy")
        if existing_policy is not None and existing_policy != source_policy:
            raise RuntimeError("protected pointer source policy differs")
        if existing_policy is None:
            if ready_path.is_file():
                raise RuntimeError(
                    "ready corpus cannot gain a source policy after sealing"
                )
            pointer_payload["source_policy"] = source_policy
            _atomic_json(pointer, pointer_payload)
            pointer_payload = _load(pointer)
    total_guide_rows = int(
        ((manifest_payload.get("totals") or {}).get("target_coverage") or {}).get(
            "guide_rows", 0
        )
    )
    if total_guide_rows <= 0:
        raise RuntimeError("sealed guide corpus contains no guide targets")
    if total_guide_rows != sum(row["guide_rows"] for row in rows):
        raise RuntimeError("sealed manifest guide-row total changed")

    payload = {
        "schema": READY_SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specialist_id": args.specialist_id,
        "guide_version": args.guide_version,
        "date_start": expected_dates[0],
        "date_end": expected_dates[-1],
        "dates": expected_dates,
        "days": len(expected_dates),
        "records": sum(row["records"] for row in rows),
        "decisions": sum(row["decisions"] for row in rows),
        "guide_rows": total_guide_rows,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "protected_pointer": str(pointer),
        "protected_pointer_sha256": _sha256(pointer),
        "pointer_manifest_sha256": pointer_payload.get("manifest_sha256"),
        "source_policy": source_policy,
        "daily_shards": rows,
        "active_training_modified": False,
    }
    if ready_path.is_file():
        existing = _load(ready_path)
        comparable_existing = {k: v for k, v in existing.items() if k != "created_at_utc"}
        comparable_new = {k: v for k, v in payload.items() if k != "created_at_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"immutable ready receipt differs: {ready_path}")
    else:
        _atomic_json(ready_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
