#!/usr/bin/env python3
"""Atomically import one checksum-bound pre-staged specialist corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _validate(
    root: Path,
    *,
    specialist_id: str,
    guide_version: str,
    minimum_records: int,
    finalization_receipt_name: str = "",
    finalization_receipt_schema: str = "",
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    pointer_path = root / "PROTECTED_EXPERT_CORPUS.json"
    ready_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    manifest = _read(manifest_path)
    pointer = _read(pointer_path)
    ready = _read(ready_path)
    totals = dict(manifest.get("totals") or {})
    coverage = dict(totals.get("target_coverage") or {})
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or (pointer.get("selection") or {}).get("value") != specialist_id
        or pointer.get("manifest_sha256") != _sha256(manifest_path)
        or ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != specialist_id
        or ready.get("guide_version") != guide_version
        or ready.get("manifest_sha256") != _sha256(manifest_path)
        or ready.get("protected_pointer_sha256") != _sha256(pointer_path)
        or int(ready.get("decisions") or 0)
        != int(totals.get("decisions_kept") or 0)
        or int(ready.get("records") or 0)
        != int(totals.get("records_kept") or 0)
        or int(ready.get("records") or 0) < int(minimum_records)
        or int(ready.get("guide_rows") or 0)
        != int(coverage.get("guide_rows") or 0)
        or int(ready.get("decisions") or 0) <= 0
        or int(ready.get("guide_rows") or 0) <= 0
    ):
        raise RuntimeError("pre-staged specialist corpus identity failed")
    for row in manifest.get("shards") or ():
        path = root / str(row.get("path") or "")
        if not path.is_file() or row.get("sha256") != _sha256(path):
            raise RuntimeError(f"pre-staged shard checksum failed: {path}")
    identity = {
        "specialist_id": specialist_id,
        "guide_version": guide_version,
        "records": int(ready["records"]),
        "decisions": int(ready["decisions"]),
        "guide_rows": int(ready["guide_rows"]),
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer_sha256": _sha256(pointer_path),
        "ready_receipt_sha256": _sha256(ready_path),
    }
    if finalization_receipt_name:
        if (
            Path(finalization_receipt_name).name
            != finalization_receipt_name
            or not finalization_receipt_schema
        ):
            raise RuntimeError("finalization receipt contract is invalid")
        finalization_path = root / finalization_receipt_name
        finalization = _read(finalization_path)
        if (
            finalization.get("schema") != finalization_receipt_schema
            or finalization.get("status") != "ready_checksum_validated"
            or finalization.get("specialist_id") != specialist_id
            or finalization.get("guide_version") != guide_version
            or finalization.get("guide_ready_receipt_sha256")
            != identity["ready_receipt_sha256"]
            or int(finalization.get("records") or 0)
            != identity["records"]
            or int(finalization.get("decisions") or 0)
            != identity["decisions"]
            or int(finalization.get("guide_rows") or 0)
            != identity["guide_rows"]
            or finalization.get("active_training_modified") is not False
        ):
            raise RuntimeError(
                "pre-staged specialist finalization receipt failed"
            )
        identity.update(
            {
                "finalization_receipt_name": finalization_receipt_name,
                "finalization_receipt_sha256": _sha256(finalization_path),
            }
        )
    return identity


def _unavailable_placeholder(
    destination: Path, *, specialist_id: str
) -> dict[str, Any] | None:
    if not destination.is_dir():
        return None
    entries = list(destination.iterdir())
    if len(entries) != 1 or entries[0].name != "UNAVAILABLE_EXPERT_CORPUS.json":
        return None
    payload = _read(entries[0])
    if (
        payload.get("schema") != "poke_bot.specialist_expert_corpus_unavailable/v1"
        or payload.get("archetype") != specialist_id
        or payload.get("reason")
        != "no acting-seat records in the pinned source corpus"
    ):
        return None
    return {
        "path": str(entries[0]),
        "sha256": _sha256(entries[0]),
    }


def _superseded_existing_corpus(
    destination: Path,
    *,
    expected_ready_sha256: str = "",
    expected_pointer_sha256: str = "",
) -> dict[str, Any] | None:
    """Authorize replacement only for one explicitly checksummed old corpus."""

    expected = [
        ("CURRENT_DECK_GUIDE_CORPUS_READY.json", expected_ready_sha256),
        ("PROTECTED_EXPERT_CORPUS.json", expected_pointer_sha256),
    ]
    expected = [(name, digest) for name, digest in expected if digest]
    if not destination.is_dir() or not expected:
        return None
    if len(expected) != 1:
        raise RuntimeError(
            "exactly one superseded-corpus identity checksum is required"
        )
    identity_name, expected_digest = expected[0]
    identity_path = destination / identity_name
    if not identity_path.is_file():
        raise RuntimeError(
            f"authorized superseded-corpus identity is missing: {identity_path}"
        )
    observed = _sha256(identity_path)
    if observed != expected_digest:
        raise RuntimeError(
            "existing specialist corpus is not the authorized superseded "
            f"identity: {destination}"
        )
    return {
        "identity_name": identity_name,
        "identity_path": str(identity_path),
        "identity_sha256": observed,
        # Preserve the original receipt field for already-issued guide-corpus
        # supersessions while permitting an unguided corpus to be authorized
        # by its protected pointer.
        "ready_path": (
            str(identity_path)
            if identity_name == "CURRENT_DECK_GUIDE_CORPUS_READY.json"
            else None
        ),
        "ready_sha256": observed,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable import receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _validated_existing_receipt(
    path: Path,
    *,
    source_host: str,
    source_root: str,
    destination: Path,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Return an already-committed import receipt only when it is still exact."""

    if not path.is_file():
        return None
    receipt = _read(path)
    expected = {
        "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
        "status": "ready",
        "source_host": source_host,
        "source_root": source_root,
        "destination": str(destination),
        **identity,
        "active_training_modified": False,
    }
    if (
        not str(receipt.get("created_at_utc") or "")
        or any(receipt.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError(f"immutable import receipt identity differs: {path}")
    replaced = receipt.get("replaced_unavailable_placeholder")
    archive = receipt.get("unavailable_placeholder_archive")
    archive_digest = receipt.get("unavailable_placeholder_sha256")
    if replaced is True:
        archive_path = Path(str(archive or "")).expanduser()
        if (
            not archive_path.is_dir()
            or not str(archive_digest or "").startswith("sha256:")
        ):
            raise RuntimeError(f"immutable import receipt archive differs: {path}")
    elif not (
        replaced is False
        and archive is None
        and archive_digest is None
    ):
        raise RuntimeError(f"immutable import receipt replacement differs: {path}")
    superseded = receipt.get("superseded_existing_corpus")
    superseded_archive = receipt.get("superseded_existing_corpus_archive")
    superseded_digest = receipt.get(
        "superseded_existing_corpus_ready_sha256"
    )
    if superseded is True:
        archive_path = Path(str(superseded_archive or "")).expanduser()
        identity_name = str(
            receipt.get("superseded_existing_corpus_identity_name")
            or "CURRENT_DECK_GUIDE_CORPUS_READY.json"
        )
        identity_path = archive_path / identity_name
        if (
            not archive_path.is_dir()
            or identity_name
            not in {
                "CURRENT_DECK_GUIDE_CORPUS_READY.json",
                "PROTECTED_EXPERT_CORPUS.json",
            }
            or not identity_path.is_file()
            or _sha256(identity_path) != superseded_digest
        ):
            raise RuntimeError(
                f"immutable superseded-corpus archive differs: {path}"
            )
    elif not (
        superseded in (None, False)
        and superseded_archive is None
        and superseded_digest is None
    ):
        raise RuntimeError(
            f"immutable superseded-corpus replacement differs: {path}"
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--guide-version", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bwlimit-kib", type=int, default=8_000)
    parser.add_argument("--minimum-records", type=int, default=0)
    parser.add_argument("--finalization-receipt-name", default="")
    parser.add_argument("--finalization-receipt-schema", default="")
    parser.add_argument("--superseded-ready-sha256", default="")
    parser.add_argument("--superseded-pointer-sha256", default="")
    args = parser.parse_args()
    if args.bwlimit_kib <= 0 or args.minimum_records < 0:
        raise ValueError("bandwidth and minimum-record contracts are invalid")

    destination = args.destination.resolve()
    placeholder = _unavailable_placeholder(
        destination,
        specialist_id=args.specialist_id,
    )
    superseded = None
    existing_identity = None
    if destination.is_dir() and placeholder is None:
        try:
            existing_identity = _validate(
                destination,
                specialist_id=args.specialist_id,
                guide_version=args.guide_version,
                minimum_records=int(args.minimum_records),
                finalization_receipt_name=args.finalization_receipt_name,
                finalization_receipt_schema=args.finalization_receipt_schema,
            )
        # An explicitly authorized unguided predecessor has no guide-ready
        # receipt by design. Treat that exact missing file as validation
        # failure so its protected-pointer checksum can authorize replacement;
        # all other filesystem errors still fail closed.
        except (RuntimeError, FileNotFoundError):
            superseded = _superseded_existing_corpus(
                destination,
                expected_ready_sha256=str(
                    args.superseded_ready_sha256 or ""
                ),
                expected_pointer_sha256=str(
                    args.superseded_pointer_sha256 or ""
                ),
            )
            if superseded is None:
                raise
    if existing_identity is not None:
        identity = existing_identity
        replacement = {
            "replaced_unavailable_placeholder": False,
            "unavailable_placeholder_archive": None,
            "unavailable_placeholder_sha256": None,
            "superseded_existing_corpus": False,
            "superseded_existing_corpus_archive": None,
            "superseded_existing_corpus_ready_sha256": None,
        }
    else:
        staging = destination.with_name(
            f".{destination.name}.importing.{os.getpid()}"
        )
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    f"--bwlimit={int(args.bwlimit_kib)}",
                    f"{args.host}:{args.remote_root.rstrip('/')}/",
                    str(staging) + "/",
                ],
                check=True,
            )
            identity = _validate(
                staging,
                specialist_id=args.specialist_id,
                guide_version=args.guide_version,
                minimum_records=int(args.minimum_records),
                finalization_receipt_name=args.finalization_receipt_name,
                finalization_receipt_schema=args.finalization_receipt_schema,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            placeholder_archive = None
            superseded_archive = None
            if placeholder is not None:
                placeholder_archive = destination.with_name(
                    f".{destination.name}.unavailable-"
                    f"{str(placeholder['sha256']).removeprefix('sha256:')[:12]}"
                )
                if placeholder_archive.exists():
                    raise RuntimeError(
                        "unavailable placeholder archive already exists: "
                        f"{placeholder_archive}"
                    )
                os.replace(destination, placeholder_archive)
            elif superseded is not None:
                superseded_archive = destination.with_name(
                    f".{destination.name}.superseded-"
                    f"{str(superseded['identity_sha256']).removeprefix('sha256:')[:12]}"
                )
                if superseded_archive.exists():
                    raise RuntimeError(
                        "superseded corpus archive already exists: "
                        f"{superseded_archive}"
                    )
                os.replace(destination, superseded_archive)
            try:
                os.replace(staging, destination)
            except BaseException:
                if (
                    placeholder_archive is not None
                    and placeholder_archive.exists()
                    and not destination.exists()
                ):
                    os.replace(placeholder_archive, destination)
                elif (
                    superseded_archive is not None
                    and superseded_archive.exists()
                    and not destination.exists()
                ):
                    os.replace(superseded_archive, destination)
                raise
            replacement = {
                "replaced_unavailable_placeholder": placeholder is not None,
                "unavailable_placeholder_archive": (
                    str(placeholder_archive)
                    if placeholder_archive is not None
                    else None
                ),
                "unavailable_placeholder_sha256": (
                    placeholder["sha256"] if placeholder is not None else None
                ),
                "superseded_existing_corpus": superseded is not None,
                "superseded_existing_corpus_archive": (
                    str(superseded_archive)
                    if superseded_archive is not None
                    else None
                ),
                "superseded_existing_corpus_ready_sha256": (
                    superseded["identity_sha256"]
                    if superseded is not None
                    else None
                ),
                "superseded_existing_corpus_identity_name": (
                    superseded["identity_name"]
                    if superseded is not None
                    else None
                ),
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    receipt_path = args.receipt.resolve()
    existing_receipt = _validated_existing_receipt(
        receipt_path,
        source_host=args.host,
        source_root=args.remote_root,
        destination=destination,
        identity=identity,
    )
    if existing_receipt is not None:
        print(json.dumps(existing_receipt, indent=2, sort_keys=True))
        return 0
    receipt = {
        "schema": "poke_bot.prestaged_specialist_corpus_import/v1",
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_host": args.host,
        "source_root": args.remote_root,
        "destination": str(destination),
        **identity,
        **replacement,
        "active_training_modified": False,
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
