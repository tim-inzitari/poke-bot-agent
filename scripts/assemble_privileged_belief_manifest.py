#!/usr/bin/env python
"""Verify distributed privileged-belief shards and publish one Inzi manifest.

The collector manifests may come from native x86-64 and arm64 builds.  Their
engine *binary* digests are expected to differ, while the engine source,
HiddenExport translation unit, checkpoint, ABI, deck contract, and sampling
temperature must match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--require-games", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _same(name: str, expected: Any, actual: Any) -> None:
    if _canonical(actual) != _canonical(expected):
        raise RuntimeError(f"distributed manifest mismatch for {name}")


def main() -> int:
    args = _parse_args()
    output = args.output_manifest.expanduser().resolve()
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for raw in args.input_manifest:
        path = raw.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "poke_bot.privileged_belief_lineage/v1":
            raise RuntimeError(f"unsupported collector manifest: {path}")
        manifests.append((path, payload))

    reference = manifests[0][1]
    common_keys = (
        "checkpoint_digest",
        "engine_source_digest",
        "hidden_export_digest",
        "hidden_snapshot_abi",
        "deck_contract",
        "temperature",
    )
    for key in common_keys:
        if reference.get(key) is None:
            raise RuntimeError(f"collector manifest lacks required lineage {key}")
    for path, payload in manifests[1:]:
        for key in common_keys:
            _same(key, reference.get(key), payload.get(key))

    seed_ranges: list[tuple[int, int, Path]] = []
    for path, payload in manifests:
        start = int(payload["seed"])
        stop = start + int(payload["requested_games"])
        for other_start, other_stop, other_path in seed_ranges:
            if start < other_stop and other_start < stop:
                raise RuntimeError(
                    f"sampling seed ranges overlap: {path} and {other_path}"
                )
        seed_ranges.append((start, stop, path))

    canonical_shards: list[dict[str, Any]] = []
    engine_builds: dict[str, dict[str, Any]] = {}
    total_games = total_records = total_decisions = total_bytes = 0
    seen_digests: set[str] = set()
    for manifest_path, payload in manifests:
        engine_digest = str(payload["engine_digest"])
        engine_builds[engine_digest] = {
            "engine_digest": engine_digest,
            "hosts": sorted(
                {
                    str(row.get("host") or "unknown")
                    for row in payload.get("shards") or []
                }
            ),
        }
        rows = list(payload.get("shards") or [])
        manifest_games = 0
        manifest_records = 0
        manifest_decisions = 0
        manifest_bytes = 0
        for row in rows:
            shard = manifest_path.parent / Path(str(row["path"])).name
            sidecar = shard.with_suffix(".meta.json")
            if not shard.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"missing shard or sidecar for {shard}")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            for key in (
                "sha256",
                "bytes",
                "games",
                "records",
                "decisions",
                "hand_labeled_decisions",
                "checkpoint_digest",
                "engine_digest",
                "engine_source_digest",
                "hidden_export_digest",
                "hidden_snapshot_abi",
            ):
                _same(f"{shard.name}:{key}", row.get(key), meta.get(key))
            digest = _sha256(shard)
            if digest != row.get("sha256"):
                raise RuntimeError(f"SHA-256 mismatch for {shard}")
            if digest in seen_digests:
                raise RuntimeError(f"duplicate shard payload detected: {shard}")
            seen_digests.add(digest)
            if int(row["decisions"]) != int(row["hand_labeled_decisions"]):
                raise RuntimeError(f"partially labeled shard rejected: {shard}")
            rel = os.path.relpath(shard, output.parent)
            canonical_shards.append(
                {
                    **row,
                    "path": rel,
                    "source_manifest": os.path.relpath(
                        manifest_path, output.parent
                    ),
                }
            )
            manifest_games += int(row["games"])
            manifest_records += int(row["records"])
            manifest_decisions += int(row["decisions"])
            manifest_bytes += int(row["bytes"])
        expected_totals = payload.get("totals") or {}
        for name, value in (
            ("games", manifest_games),
            ("records", manifest_records),
            ("decisions", manifest_decisions),
            ("hand_labeled_decisions", manifest_decisions),
            ("bytes", manifest_bytes),
        ):
            if int(expected_totals.get(name, -1)) != value:
                raise RuntimeError(
                    f"collector total mismatch for {manifest_path}: {name}"
                )
        total_games += manifest_games
        total_records += manifest_records
        total_decisions += manifest_decisions
        total_bytes += manifest_bytes

    if args.require_games > 0 and total_games != int(args.require_games):
        raise RuntimeError(
            f"canonical game count {total_games} != required {args.require_games}"
        )
    canonical_shards.sort(key=lambda row: (str(row["host"]), str(row["path"])))
    corpus_digest = "sha256:" + hashlib.sha256(
        _canonical(
            [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "games": row["games"],
                    "decisions": row["decisions"],
                }
                for row in canonical_shards
            ]
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "schema": "poke_bot.privileged_belief_corpus/v1",
        "storage_authority": "inzi",
        "created_unix": int(time.time()),
        "corpus_digest": corpus_digest,
        **{key: reference[key] for key in common_keys},
        "engine_builds": sorted(
            engine_builds.values(), key=lambda row: row["engine_digest"]
        ),
        "source_manifests": [
            os.path.relpath(path, output.parent) for path, _ in manifests
        ],
        "seed_ranges": [
            {"start": start, "stop_exclusive": stop, "source": str(path)}
            for start, stop, path in seed_ranges
        ],
        "totals": {
            "games": total_games,
            "records": total_records,
            "decisions": total_decisions,
            "hand_labeled_decisions": total_decisions,
            "bytes": total_bytes,
            "shards": len(canonical_shards),
        },
        "shards": canonical_shards,
    }
    _atomic_json(output, result)
    print(json.dumps({"manifest": str(output), **result["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
