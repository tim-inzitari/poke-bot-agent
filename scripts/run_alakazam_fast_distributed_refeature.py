#!/usr/bin/env python3
"""Fast, low-write card-743 re-featurization split across compute hosts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_alakazam_collision_census_r298 as census  # noqa: E402


RAM_CAP = 20 * 1024**3
BUCKETS = 64


def _relocate_closed_lane(
    result: dict[str, object],
    *,
    active_root: Path,
    completed_root: Path,
) -> dict[str, object]:
    """Move one fully closed lane from tmpfs to local durable scratch."""

    lane_index = int(result["lane_index"])
    relocated = dict(result)
    for stream_name, field in (
        ("collision-audit", "collision_audit_private_spool_shards"),
        ("materialized", "materialized_private_spool_shards"),
    ):
        source = active_root / stream_name / f"lane-{lane_index:02d}"
        destination = completed_root / stream_name / source.name
        if destination.exists():
            raise RuntimeError(f"completed lane destination already exists: {destination}")
        shutil.move(str(source), str(destination))
        rows = []
        for raw in result[field]:  # type: ignore[index]
            row = dict(raw)
            row["path"] = str(destination / Path(str(row["path"])).name)
            rows.append(row)
        relocated[field] = rows
    return relocated


def _import_precompleted_lane(
    lane_index: int,
    *,
    source_root: Path,
    completed_root: Path,
) -> dict[str, object]:
    """Hard-link one checksum-preserved complete lane into a fresh run."""

    source_lane = source_root / "materialized" / f"lane-{lane_index:02d}"
    if not source_lane.is_dir() or source_lane.is_symlink():
        raise RuntimeError(f"precompleted lane is missing or unsafe: {source_lane}")
    destination_lane = completed_root / "materialized" / source_lane.name
    destination_lane.mkdir(mode=0o700)
    metadata: list[dict[str, object]] = []
    for path in sorted(source_lane.glob("bucket-*.jsonl.partial")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"precompleted spool member is unsafe: {path}")
        try:
            bucket = int(path.name.removeprefix("bucket-").removesuffix(".jsonl.partial"))
        except ValueError as exc:
            raise RuntimeError(f"precompleted spool member has invalid name: {path}") from exc
        destination = destination_lane / path.name
        os.link(path, destination)
        digest = hashlib.sha256()
        record_count = 0
        with destination.open("rb") as stream:
            for raw_line in stream:
                digest.update(raw_line)
                record_count += 1
        if record_count < 1:
            raise RuntimeError(f"precompleted spool member is empty: {path}")
        metadata.append(
            {
                "lane_index": lane_index,
                "bucket": bucket,
                "path": str(destination),
                "sha256": f"sha256:{digest.hexdigest()}",
                "size_bytes": destination.stat().st_size,
                "record_count": record_count,
            }
        )
    if not metadata:
        raise RuntimeError(f"precompleted lane has no records: {source_lane}")
    return {
        "lane_index": lane_index,
        "collision_audit_private_spool_shards": [],
        "materialized_private_spool_shards": metadata,
    }


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--cg-runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--spool-root",
        type=Path,
        default=Path("/dev/shm"),
        help="parent directory for private lane spools (use local NVMe when tmpfs is too small)",
    )
    parser.add_argument("--host-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 33), required=True)
    parser.add_argument("--logical-lanes", type=int, choices=range(1, 257))
    parser.add_argument("--completed-spool-root", type=Path)
    parser.add_argument("--precompleted-spool-root", type=Path)
    parser.add_argument("--precompleted-lanes", default="")
    parser.add_argument("--active-spool-cap-bytes", type=int, default=RAM_CAP)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("create-only output root already exists")
    manifest = census._read_json(args.manifest)
    archives = census._manifest_archives(manifest, archive_root=args.archive_root)
    selected = [row for index, row in enumerate(archives) if index % 2 == args.host_index]
    if len(selected) != 15:
        raise SystemExit("expected exactly 15 UTC-day archives for this host")
    logical_lanes = args.logical_lanes or args.workers
    args.output_root.mkdir(parents=True, exist_ok=False)
    args.spool_root.mkdir(parents=True, exist_ok=True)
    ram_root = args.spool_root / f"alakazam-refeature-{os.getpid()}-{args.host_index}"
    ram_root.mkdir(mode=0o700)
    (ram_root / "collision-audit").mkdir()
    (ram_root / "materialized").mkdir()
    completed_root = None
    if args.completed_spool_root is not None:
        args.completed_spool_root.mkdir(parents=True, exist_ok=True)
        completed_root = args.completed_spool_root / ram_root.name
        completed_root.mkdir(mode=0o700)
        (completed_root / "collision-audit").mkdir()
        (completed_root / "materialized").mkdir()
    precompleted_lanes = {
        int(value)
        for value in args.precompleted_lanes.split(",")
        if value.strip()
    }
    if precompleted_lanes and (completed_root is None or args.precompleted_spool_root is None):
        raise SystemExit("precompleted lanes require both completed spool roots")
    if any(index < 0 or index >= logical_lanes for index in precompleted_lanes):
        raise SystemExit("precompleted lane index is outside the logical lane plan")
    worker_archives = tuple((dict(meta), str(path.resolve())) for meta, path in selected)
    active_spool_peak_bytes = 0
    succeeded = False
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    census._run_private_day_lane,
                    lane_index=index,
                    lane_count=logical_lanes,
                    archives=worker_archives,
                    spool_root=str(ram_root),
                    bucket_count=BUCKETS,
                    engine_transition_jsonl=None,
                    cg_runtime_root=str(args.cg_runtime_root.resolve()),
                    materialized_only=True,
                ): index
                for index in range(logical_lanes)
                if index not in precompleted_lanes
            }
            results = [
                _import_precompleted_lane(
                    index,
                    source_root=args.precompleted_spool_root.resolve(),
                    completed_root=completed_root,
                )
                for index in sorted(precompleted_lanes)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if completed_root is not None:
                    result = _relocate_closed_lane(
                        result,
                        active_root=ram_root,
                        completed_root=completed_root,
                    )
                active_bytes = _tree_bytes(ram_root)
                active_spool_peak_bytes = max(active_spool_peak_bytes, active_bytes)
                if active_bytes > args.active_spool_cap_bytes:
                    raise SystemExit(
                        f"active private spool exceeded cap: {active_bytes} > {args.active_spool_cap_bytes}"
                    )
                results.append(result)
        results.sort(key=lambda row: row["lane_index"])
        private_spool_bytes = sum(
            int(item["size_bytes"])
            for result in results
            for item in result["materialized_private_spool_shards"]
        )
        spool_is_tmpfs = args.spool_root.resolve() == Path("/dev/shm")
        merge_spool_root = completed_root or ram_root
        writer = census._ContentAddressedShardWriter(
            args.output_root / "refeatured-records",
            bucket_count=BUCKETS,
            raw_manifest_sha256=census.canonical_sha256(manifest),
            frozen_schema_manifest_sha256="sha256:41c9ae94f47c0983bccb9e13c680ad5cf5d93f547aa61aeb91fea20cd53f62af",
            zero_bypass_receipt_sha256="sha256:be3d8d0bba4be02d10358f6e0512b9d740ede5c3f8bad9eca8f848915fb60d23",
            record_scope=census.RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
        )
        records = census._merge_private_day_lane_spools(
            results,
            spool_root=merge_spool_root / "materialized",
            writer=writer,
            spool_field="materialized_private_spool_shards",
        )
        writer.close()
        shard_manifest = writer.manifest()
        receipt = {
            "schema": "poke_bot.alakazam_fast_distributed_refeature/v1",
            "status": "complete",
            "hostname": socket.gethostname(),
            "host_index": args.host_index,
            "worker_count": args.workers,
            "logical_lane_count": logical_lanes,
            "day_count": len(selected),
            "days": [str(meta["date"]) for meta, _ in selected],
            "card_id_filter": 743,
            "acting_seat_only": True,
            "record_count": records,
            "private_spool_bytes": private_spool_bytes,
            "private_spool_parent": str(args.spool_root.resolve()),
            "completed_spool_parent": str(args.completed_spool_root.resolve()) if args.completed_spool_root else None,
            "private_spool_storage": "bounded_tmpfs_then_local_filesystem" if completed_root else ("tmpfs" if spool_is_tmpfs else "local_filesystem"),
            "ram_spool_peak_bytes": active_spool_peak_bytes if spool_is_tmpfs else 0,
            "ram_spool_cap_bytes": args.active_spool_cap_bytes if spool_is_tmpfs else 0,
            "maximum_final_shard_bytes": 1024**3,
            "shard_manifest": shard_manifest,
        }
        census._write_create_only_json(args.output_root / "COMPLETE.json", receipt)
        succeeded = True
        print(json.dumps(receipt, sort_keys=True))
        return 0
    finally:
        if ram_root.exists():
            shutil.rmtree(ram_root)
        if succeeded and completed_root is not None and completed_root.exists():
            shutil.rmtree(completed_root)


if __name__ == "__main__":
    raise SystemExit(main())
