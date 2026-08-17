#!/usr/bin/env python3
"""Attest exact archived submission runtimes for replay reconstruction.

This verifier derives each runtime directory from that submission's bundle
digest, verifies the immutable bundle/checkpoint/tree bytes, and proves every
regular file extracted from the bundle still matches the tar member bytes.
Only then does it create a submission-specific runtime-parity receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "poke_bot.replay_model_inspector_runtime_parity_receipt/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"runtime source root is not a directory: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if "__pycache__" in relative.parts or candidate.suffix in {".pyc", ".pyo"}:
            continue
        if candidate.is_symlink() or (
            not candidate.is_dir() and not candidate.is_file()
        ):
            raise ValueError(f"unsafe runtime entry: {candidate}")
        if candidate.is_file():
            files.append(candidate)
    digest = hashlib.sha256()
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(b"file\0" + relative + b"\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _verify_extraction(bundle: Path, runtime: Path) -> None:
    runtime = runtime.resolve(strict=True)
    archive_files: dict[str, tuple[int, str]] = {}
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            raw_name = member.name.removeprefix("./")
            if raw_name in {"", "."} and member.isdir():
                continue
            relative = PurePosixPath(raw_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {member.name}")
            if "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}:
                continue
            extracted = (runtime / Path(*relative.parts)).resolve(strict=True)
            if runtime not in extracted.parents or not extracted.is_file():
                raise ValueError(f"archive extraction is missing: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda stream=source: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
            archive_files[relative.as_posix()] = (member.size, digest.hexdigest())
            if (
                extracted.stat().st_size != member.size
                or _sha256(extracted)[7:] != digest.hexdigest()
            ):
                raise ValueError(
                    f"extracted runtime differs from bundle: {member.name}"
                )
    runtime_files = {
        candidate.relative_to(runtime).as_posix()
        for candidate in runtime.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.relative_to(runtime).parts
        and candidate.suffix not in {".pyc", ".pyo"}
    }
    if runtime_files != set(archive_files):
        raise ValueError("extracted runtime file set differs from exact bundle")


def _artifact(record: dict[str, Any], name: str) -> tuple[str, str]:
    value = record.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"submission {record.get('submission_id')} lacks {name}")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise TypeError(f"submission {record.get('submission_id')} has invalid {name}")
    return path, digest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--skip-submission-id", type=int, action="append", default=[])
    parser.add_argument("--verified-by", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    archive_root = args.archive_root.resolve(strict=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt_root = args.receipt_root.resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    skipped = set(args.skip_submission_id)
    verified_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    results: list[dict[str, Any]] = []
    for record in manifest.get("records") or []:
        submission_id = record.get("submission_id")
        if isinstance(submission_id, bool) or not isinstance(submission_id, int):
            continue
        if submission_id in skipped:
            continue
        if record.get("status") != "verified":
            results.append(
                {
                    "submission_id": submission_id,
                    "status": "skipped_unverified_provenance",
                }
            )
            continue
        _, bundle_digest = _artifact(record, "bundle")
        _, checkpoint_digest = _artifact(record, "checkpoint")
        _, tree_digest = _artifact(record, "matchup_tree")
        output = receipt_root / f"{submission_id}.json"
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            expected_existing = {
                "submission_id": submission_id,
                "checkpoint_sha256": checkpoint_digest,
                "bundle_sha256": bundle_digest,
                "runtime_package_sha256": bundle_digest,
            }
            if (
                existing.get("schema") != SCHEMA
                or existing.get("status") != "verified"
                or any(
                    existing.get(key) != value
                    for key, value in expected_existing.items()
                )
            ):
                raise ValueError(f"submission {submission_id} receipt conflicts")
            results.append(
                {"submission_id": submission_id, "status": "already_attested"}
            )
            continue
        bundle_hex = bundle_digest.removeprefix("sha256:")
        runtime = (
            archive_root / "artifacts" / "runtimes" / f"sha256-{bundle_hex}" / "package"
        )
        bundle = (
            archive_root
            / "artifacts"
            / "bundles"
            / f"sha256-{bundle_hex}"
            / "submission.tar.gz"
        )
        checkpoint_hex = checkpoint_digest.removeprefix("sha256:")
        checkpoint = (
            archive_root
            / "artifacts"
            / "checkpoints"
            / f"sha256-{checkpoint_hex}"
            / "model.pt"
        )
        tree = runtime / "matchup_tree.json"
        if _sha256(bundle) != bundle_digest or _sha256(checkpoint) != checkpoint_digest:
            raise ValueError(f"submission {submission_id} artifact digest mismatch")
        if _sha256(tree) != tree_digest:
            raise ValueError(f"submission {submission_id} matchup-tree digest mismatch")
        _verify_extraction(bundle, runtime)
        payload = {
            "bundle_sha256": bundle_digest,
            "checkpoint_sha256": checkpoint_digest,
            "runtime_package_sha256": bundle_digest,
            "runtime_source_tree_sha256": _source_tree_sha256(runtime),
            "schema": SCHEMA,
            "status": "verified",
            "submission_id": submission_id,
            "verification": {
                "method": "independent_exact_runtime_parity",
                "verified_at_utc": verified_at,
                "verified_by": args.verified_by,
            },
            "version": 1,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(encoded)
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
        results.append({"submission_id": submission_id, "status": "attested"})
    print(json.dumps({"status": "complete", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
