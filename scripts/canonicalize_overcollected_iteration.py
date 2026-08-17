#!/usr/bin/env python3
"""Remove unused replacement-capacity attempts from an uncommitted RL shard.

This repair is deliberately narrow: it succeeds only when all configured
primary self-play and public-mix records are already present.  It never chooses
an arbitrary replacement for a missing primary game and never edits a committed
iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonicalize(
    shard: Path,
    *,
    self_play_games: int,
    public_mix_games: int,
    audit_path: Path,
) -> dict[str, Any]:
    shard = Path(shard).resolve()
    audit_path = Path(audit_path).resolve()
    if not shard.is_file():
        raise FileNotFoundError(shard)
    if self_play_games < 0 or public_mix_games < 0:
        raise ValueError("target game counts must be non-negative")

    original_stat = shard.stat()
    original_digest = _sha256(shard)
    counts = {
        "primary_self_play": 0,
        "replacement_capacity": 0,
        "public_mix": 0,
        "total": 0,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{shard.name}.canonical.", dir=str(shard.parent)
    )
    temporary = Path(temporary_name)
    try:
        kept_count = 0
        with shard.open("rb") as source, os.fdopen(fd, "wb") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid shard JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object shard row at line {line_number}")
                provenance = dict(row.get("target_provenance") or {})
                replacement = bool(provenance.get("replacement_capacity"))
                self_play = bool(provenance.get("self_play"))
                counts["total"] += 1
                if replacement:
                    if not self_play:
                        raise ValueError(
                            "replacement-capacity row is not marked as self-play: "
                            f"line {line_number}"
                        )
                    counts["replacement_capacity"] += 1
                    continue
                if self_play:
                    counts["primary_self_play"] += 1
                else:
                    counts["public_mix"] += 1
                target.write(line if line.endswith(b"\n") else line + b"\n")
                kept_count += 1
            target.flush()
            os.fsync(target.fileno())

        if counts["primary_self_play"] != int(self_play_games):
            raise RuntimeError(
                "refusing arbitrary replacement selection: primary self-play "
                f"records={counts['primary_self_play']} expected={self_play_games}"
            )
        if counts["public_mix"] != int(public_mix_games):
            raise RuntimeError(
                "public-mix records are incomplete: "
                f"records={counts['public_mix']} expected={public_mix_games}"
            )
        expected_total = int(self_play_games) + int(public_mix_games)
        if kept_count != expected_total:
            raise RuntimeError(
                f"canonical retained rows={kept_count} expected={expected_total}"
            )
        os.chmod(temporary, original_stat.st_mode & 0o777)
        os.replace(temporary, shard)
        directory_fd = os.open(shard.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    payload = {
        "schema": "poke_bot.exact_collection_canonicalization/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard": str(shard),
        "original": {
            "sha256": original_digest,
            "bytes": int(original_stat.st_size),
            "counts": counts,
        },
        "canonical": {
            "sha256": _sha256(shard),
            "bytes": int(shard.stat().st_size),
            "source_games": expected_total,
            "self_play_games": int(self_play_games),
            "public_mix_games": int(public_mix_games),
        },
        "discarded_replacement_capacity_games": int(
            counts["replacement_capacity"]
        ),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = audit_path.with_name(f".{audit_path.name}.tmp.{os.getpid()}")
    audit_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(audit_tmp, audit_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--self-play-games", type=int, required=True)
    parser.add_argument("--public-mix-games", type=int, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            canonicalize(
                args.shard,
                self_play_games=args.self_play_games,
                public_mix_games=args.public_mix_games,
                audit_path=args.audit,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
