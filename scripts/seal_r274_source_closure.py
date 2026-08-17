#!/usr/bin/env python3
"""Seal a create-only r274 source tree and its r260 migration closure receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


MANIFEST_SCHEMA = "poke_bot.alakazam_new_list_direct_r274_source_manifest/v1"
CLOSURE_SCHEMA = "poke_bot.r241_own_deck_successor_source_closure/v1"
INCLUDED_ROOTS = ("poke_bot", "scripts", "config", "state", "ops")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_create_only(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != body:
            raise RuntimeError(f"create-only artifact changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(source_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for top in INCLUDED_ROOTS:
        base = source_root / top
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(source_root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise RuntimeError(f"source closure contains a symlink: {relative}")
            if not path.is_file() or path.suffix in {".pyc", ".log"}:
                continue
            rows.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not rows:
        raise RuntimeError("source closure inventory is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    parser.add_argument("--owner-contract", required=True, type=Path)
    parser.add_argument("--closure-receipt", required=True, type=Path)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    owner = args.owner_contract.expanduser().resolve()
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError("source root must be a regular directory")
    if not output_base.is_dir() or output_base.is_symlink():
        raise RuntimeError("output base must already exist")
    if owner.is_symlink() or not owner.is_file():
        raise RuntimeError("owner contract must be a regular file")
    owner_sha = sha256_file(owner)
    rows = inventory(source)
    tree_sha = "sha256:" + hashlib.sha256(canonical_bytes(rows)).hexdigest()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r274",
        "owner_contract_sha256": owner_sha,
        "source_tree_sha256": tree_sha,
        "files": rows,
        "derived_from_r259_runtime_tree": False,
        "unlisted_pycache_present": False,
        "status": "sealed",
    }
    manifest_body = canonical_bytes(manifest)
    manifest_sha = "sha256:" + hashlib.sha256(manifest_body).hexdigest()
    destination = output_base / f"alakazam-r274-source-{manifest_sha[7:23]}"
    if destination.exists():
        raise RuntimeError(f"source snapshot destination already exists: {destination}")
    destination.mkdir(mode=0o700)
    try:
        for row in rows:
            relative = Path(str(row["path"]))
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target, follow_symlinks=False)
        (destination / "r274-source-manifest.json").write_bytes(manifest_body)
        for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
    except Exception:
        # Leave an incomplete exclusively-created directory for inspection;
        # never erase or overwrite a competing publication automatically.
        raise
    closure = {
        "schema": CLOSURE_SCHEMA,
        "status": "sealed",
        "owner_contract_sha256": owner_sha,
        "source_tree_sha256": tree_sha,
        "closure_receipt_sha256": manifest_sha,
        "derived_from_r259_runtime_tree": False,
        "unlisted_pycache_present": False,
        "source_root": str(destination),
        "manifest": str(destination / "r274-source-manifest.json"),
    }
    write_create_only(args.closure_receipt.expanduser().absolute(), canonical_bytes(closure))
    print(json.dumps({"source_root": str(destination), "manifest_sha256": manifest_sha, "source_tree_sha256": tree_sha, "closure_receipt": str(args.closure_receipt.expanduser().absolute())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
