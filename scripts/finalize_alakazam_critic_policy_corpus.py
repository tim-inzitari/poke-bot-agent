#!/usr/bin/env python3
"""Verify and create-only seal the temporary critic policy corpus copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "poke_bot.alakazam_prize_plan_v2_policy_corpus_transfer_receipt/v1"


class CorpusTransferError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CorpusTransferError(f"{label} must be a regular file")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    stage = root / "private-transfer"
    final = root / "sealed"
    receipt_path = root / "transfer-receipt.json"
    if root.is_symlink() or not root.is_dir() or stage.is_symlink() or not stage.is_dir():
        raise CorpusTransferError("transfer root/stage is unsafe")
    if final.exists() or receipt_path.exists():
        raise CorpusTransferError("final corpus and receipt are create-only")
    manifest_path = _regular(stage / "manifest.json", "manifest")
    manifest_sha = _sha(manifest_path)
    if manifest_sha != args.expected_manifest_sha256:
        raise CorpusTransferError("manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("shards")
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or manifest.get("compact_mode") != "temporal-expert-v1"
        or not isinstance(rows, list)
        or len(rows) != 20
    ):
        raise CorpusTransferError("manifest identity is not the protected 20-day temporal corpus")
    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise CorpusTransferError("manifest shard row is malformed")
        relative = str(row.get("path") or "")
        if not relative or relative in seen or Path(relative).name != relative:
            raise CorpusTransferError("manifest shard path is unsafe or duplicated")
        seen.add(relative)
        path = _regular(stage / relative, f"shard {relative}")
        observed_sha = _sha(path)
        expected_sha = str(row.get("sha256") or "")
        expected_size = int(row.get("bytes") or -1)
        if observed_sha != expected_sha or path.stat().st_size != expected_size:
            raise CorpusTransferError(f"shard identity mismatch: {relative}")
        identities.append(
            {"path": relative, "sha256": observed_sha, "size_bytes": expected_size}
        )
    pointer = _regular(stage / "PROTECTED_EXPERT_CORPUS.json", "protected pointer")
    pointer_sha = _sha(pointer)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    if (
        pointer_payload.get("protected") is not True
        or pointer_payload.get("manifest") != "manifest.json"
        or pointer_payload.get("manifest_sha256") != manifest_sha
    ):
        raise CorpusTransferError("protected pointer does not bind the copied manifest")
    os.rename(stage, final)
    dir_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    receipt = {
        "schema": SCHEMA,
        "source_host": "elmo",
        "source_read_only": True,
        "parallel_data_streams_exact": 4,
        "metadata_prefetch_stream_count": 1,
        "maximum_large_object_streams_concurrent": 4,
        "destination_root": str(final),
        "manifest_sha256": manifest_sha,
        "protected_pointer_sha256": pointer_sha,
        "shards": identities,
        "shard_count": len(identities),
        "total_shard_bytes": sum(row["size_bytes"] for row in identities),
        "source_mutated": False,
        "actor_activation": False,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(receipt_path, flags, 0o444)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt["receipt_sha256"] = _sha(receipt_path)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--expected-manifest-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), sort_keys=True))
    except (CorpusTransferError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
