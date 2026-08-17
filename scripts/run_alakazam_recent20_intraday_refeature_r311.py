#!/usr/bin/env python3
"""One-day-at-a-time, low-write Alakazam re-featurization for revision 9.

Each host receives alternating dates from the fixed recent-20-day window.  A
host processes exactly one UTC day at a time while splitting that day's ZIP
members across multiple processes.  Workers write one sequential private file
per lane; the parent deterministically merges those files into content-
addressed JSONL shards capped at 15,000,000,000 bytes.
"""

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
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_alakazam_collision_census_r298 as census  # noqa: E402


WINDOW_START = "2026-07-23"
WINDOW_END = "2026-08-11"
WINDOW_DAYS = 20
FINAL_SHARD_LIMIT_BYTES = 15_000_000_000
PRIVATE_BUCKET_COUNT = 1
GOAL_GATEWAY_SHA256 = "sha256:8908c4e8bcf36a089ba7f230c137e259f024125807bdb04b03d77483f533c223"
GOAL_CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"


class FifteenGbDayShardWriter:
    """Sequential content-addressed writer compatible with the census merger."""

    bucket_count = PRIVATE_BUCKET_COUNT

    def __init__(
        self,
        root: Path,
        *,
        day: str,
        raw_manifest_sha256: str,
        frozen_schema_manifest_sha256: str,
        zero_bypass_receipt_sha256: str,
    ) -> None:
        self.root = root
        self.work_root = root / ".private-work"
        self.shards_root = root / "shards"
        self.day = day
        self.raw_manifest_sha256 = raw_manifest_sha256
        self.frozen_schema_manifest_sha256 = frozen_schema_manifest_sha256
        self.zero_bypass_receipt_sha256 = zero_bypass_receipt_sha256
        self.record_count = 0
        self._index = -1
        self._stream: Any | None = None
        self._path: Path | None = None
        self._digest: Any | None = None
        self._size = 0
        self._shard_records = 0
        self._metadata: list[dict[str, Any]] = []
        root.mkdir(parents=False, exist_ok=False)
        self.work_root.mkdir()
        self.shards_root.mkdir()

    @staticmethod
    def _bucket(record: Mapping[str, Any]) -> int:
        del record
        return 0

    def _header(self, index: int) -> bytes:
        return census.canonical_json_bytes(
            {
                "schema": "poke_bot.alakazam_recent20_intraday_refeature_shard/v1",
                "goal_revision": 9,
                "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
                "goal_contract_sha256": GOAL_CONTRACT_SHA256,
                "utc_day": self.day,
                "logical_shard_index": index,
                "record_scope": census.RECORD_SCOPE_MATERIALIZED_ACTING_SEAT_CARD_743,
                "eligibility_card_id": 743,
                "raw_expert_corpus_manifest_sha256": self.raw_manifest_sha256,
                "frozen_schema_manifest_sha256": self.frozen_schema_manifest_sha256,
                "zero_bypass_receipt_sha256": self.zero_bypass_receipt_sha256,
                "maximum_final_shard_bytes": FINAL_SHARD_LIMIT_BYTES,
                "create_only": True,
            }
        )

    def _open_next(self) -> None:
        self._index += 1
        self._path = self.work_root / f"day-{self.day}-part-{self._index:03d}.jsonl.partial"
        self._stream = self._path.open("xb")
        self._digest = hashlib.sha256()
        header = self._header(self._index)
        self._stream.write(header)
        self._digest.update(header)
        self._size = len(header)
        self._shard_records = 0

    def _finish_current(self) -> None:
        if self._stream is None or self._path is None or self._digest is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        digest = self._digest.hexdigest()
        final = self.shards_root / f"sha256-{digest}.jsonl"
        if final.exists():
            raise RuntimeError(f"create-only shard already exists: {final}")
        os.link(self._path, final)
        self._path.unlink()
        self._metadata.append(
            {
                "utc_day": self.day,
                "logical_shard_index": self._index,
                "filename": final.name,
                "sha256": f"sha256:{digest}",
                "size_bytes": self._size,
                "record_count": self._shard_records,
            }
        )
        self._stream = None
        self._path = None
        self._digest = None
        self._size = 0
        self._shard_records = 0

    def write(self, record: Mapping[str, Any]) -> None:
        encoded = census.canonical_json_bytes(record)
        if len(encoded) >= FINAL_SHARD_LIMIT_BYTES:
            raise RuntimeError("one record exceeds the 15 GB shard ceiling")
        if self._stream is None:
            self._open_next()
        if self._size + len(encoded) > FINAL_SHARD_LIMIT_BYTES:
            self._finish_current()
            self._open_next()
        assert self._stream is not None and self._digest is not None
        self._stream.write(encoded)
        self._digest.update(encoded)
        self._size += len(encoded)
        self._shard_records += 1
        self.record_count += 1

    def close(self) -> None:
        self._finish_current()
        self.work_root.rmdir()

    def manifest(self) -> dict[str, Any]:
        if self._stream is not None:
            raise RuntimeError("writer must be closed before reading its manifest")
        return {
            "schema": "poke_bot.alakazam_recent20_intraday_refeature_shard_manifest/v1",
            "utc_day": self.day,
            "maximum_final_shard_bytes": FINAL_SHARD_LIMIT_BYTES,
            "shard_count": len(self._metadata),
            "total_bytes": sum(int(row["size_bytes"]) for row in self._metadata),
            "record_count": sum(int(row["record_count"]) for row in self._metadata),
            "shards": self._metadata,
        }


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _existing_lane_result(*, lane_index: int, source_root: Path, combined_root: Path) -> dict[str, Any]:
    """Verify and hard-link one completed legacy lane without copying bytes."""

    source = source_root / "materialized" / f"lane-{lane_index:02d}" / "bucket-00000.jsonl.partial"
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"recovery lane is missing or unsafe: {source}")
    destination_dir = combined_root / "materialized" / f"lane-{lane_index:02d}"
    destination_dir.mkdir()
    destination = destination_dir / source.name
    os.link(source, destination)
    digest = hashlib.sha256()
    count = 0
    with destination.open("rb") as stream:
        for raw_line in stream:
            record = json.loads(raw_line)
            if not isinstance(record, Mapping) or census.canonical_json_bytes(record) != raw_line:
                raise RuntimeError(f"recovery lane contains a noncanonical record: {source}")
            digest.update(raw_line)
            count += 1
    if count < 1:
        raise RuntimeError(f"recovery lane is empty: {source}")
    return {
        "lane_index": lane_index,
        "materialized_private_spool_shards": [
            {
                "lane_index": lane_index,
                "bucket": 0,
                "path": str(destination),
                "sha256": f"sha256:{digest.hexdigest()}",
                "size_bytes": destination.stat().st_size,
                "record_count": count,
            }
        ],
    }


def _recover_day(
    *,
    day: str,
    archive: tuple[Mapping[str, Any], Path],
    output_root: Path,
    spool_root: Path,
    prior_private_root: Path,
    workers: int,
    rerun_lanes: set[int],
    cg_runtime_root: Path,
    raw_manifest_sha256: str,
) -> dict[str, Any]:
    """Reuse verified completed lanes and replay only explicitly failed lanes."""

    if not rerun_lanes or any(index < 0 or index >= workers for index in rerun_lanes):
        raise RuntimeError("recovery requires valid explicit failed lane indices")
    day_root = output_root / f"day-{day}"
    if not day_root.is_dir() or (day_root / "COMPLETE.json").exists():
        raise RuntimeError("recovery requires one existing incomplete day root")
    if (day_root / "refeatured-records").exists():
        raise RuntimeError("recovery final writer root already exists")
    combined_root = spool_root / f"recovery-{day}-{os.getpid()}"
    combined_root.mkdir(parents=True, exist_ok=False)
    (combined_root / "collision-audit").mkdir()
    (combined_root / "materialized").mkdir()
    succeeded = False
    try:
        results = [
            _existing_lane_result(
                lane_index=index,
                source_root=prior_private_root,
                combined_root=combined_root,
            )
            for index in range(workers)
            if index not in rerun_lanes
        ]
        worker_archive = ((dict(archive[0]), str(archive[1].resolve())),)
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(rerun_lanes)) as pool:
            futures = [
                pool.submit(
                    census._run_private_day_lane,
                    lane_index=index,
                    lane_count=workers,
                    archives=worker_archive,
                    spool_root=str(combined_root),
                    bucket_count=PRIVATE_BUCKET_COUNT,
                    engine_transition_jsonl=None,
                    cg_runtime_root=str(cg_runtime_root.resolve()),
                    materialized_only=True,
                )
                for index in sorted(rerun_lanes)
            ]
            results.extend(future.result() for future in futures)
        results.sort(key=lambda row: int(row["lane_index"]))
        writer = FifteenGbDayShardWriter(
            day_root / "refeatured-records",
            day=day,
            raw_manifest_sha256=raw_manifest_sha256,
            frozen_schema_manifest_sha256="sha256:41c9ae94f47c0983bccb9e13c680ad5cf5d93f547aa61aeb91fea20cd53f62af",
            zero_bypass_receipt_sha256="sha256:be3d8d0bba4be02d10358f6e0512b9d740ede5c3f8bad9eca8f848915fb60d23",
        )
        records = census._merge_private_day_lane_spools(
            results,
            spool_root=combined_root / "materialized",
            writer=writer,
            spool_field="materialized_private_spool_shards",
        )
        writer.close()
        manifest = writer.manifest()
        receipt = {
            "schema": "poke_bot.alakazam_recent20_intraday_refeature_day_receipt/v1",
            "status": "complete",
            "goal_revision": 9,
            "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
            "goal_contract_sha256": GOAL_CONTRACT_SHA256,
            "hostname": socket.gethostname(),
            "utc_day": day,
            "worker_count": workers,
            "recovery_source_root": str(prior_private_root),
            "verified_reused_lane_count": workers - len(rerun_lanes),
            "replayed_lane_indices": sorted(rerun_lanes),
            "private_spool_file_count": workers,
            "private_spool_bytes": _tree_bytes(combined_root / "materialized"),
            "record_count": records,
            "maximum_final_shard_bytes": FINAL_SHARD_LIMIT_BYTES,
            "shard_manifest": manifest,
        }
        census._write_create_only_json(day_root / "COMPLETE.json", receipt)
        succeeded = True
        return receipt
    finally:
        if succeeded:
            shutil.rmtree(combined_root)


def _day_dates(archives: list[tuple[Mapping[str, Any], Path]], host_index: int) -> list[str]:
    recent = [str(meta["date"]) for meta, _ in archives if WINDOW_START <= str(meta["date"]) <= WINDOW_END]
    if len(recent) != WINDOW_DAYS or recent[0] != WINDOW_START or recent[-1] != WINDOW_END:
        raise RuntimeError("manifest does not contain the exact recent 20-day window")
    return [day for index, day in enumerate(recent) if index % 2 == host_index]


def _run_day(
    *,
    day: str,
    archive: tuple[Mapping[str, Any], Path],
    output_root: Path,
    spool_root: Path,
    workers: int,
    cg_runtime_root: Path,
    raw_manifest_sha256: str,
) -> dict[str, Any]:
    day_root = output_root / f"day-{day}"
    if day_root.exists():
        complete = day_root / "COMPLETE.json"
        if not complete.is_file():
            raise RuntimeError(f"existing day output is incomplete: {day_root}")
        return census._read_json(complete)
    day_root.mkdir(parents=True, exist_ok=False)
    private_root = spool_root / f"alakazam-r311-{day}-{os.getpid()}"
    private_root.mkdir(parents=True, exist_ok=False)
    (private_root / "collision-audit").mkdir()
    (private_root / "materialized").mkdir()
    succeeded = False
    try:
        worker_archive = ((dict(archive[0]), str(archive[1].resolve())),)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    census._run_private_day_lane,
                    lane_index=index,
                    lane_count=workers,
                    archives=worker_archive,
                    spool_root=str(private_root),
                    bucket_count=PRIVATE_BUCKET_COUNT,
                    engine_transition_jsonl=None,
                    cg_runtime_root=str(cg_runtime_root.resolve()),
                    materialized_only=True,
                )
                for index in range(workers)
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda row: int(row["lane_index"]))
        writer = FifteenGbDayShardWriter(
            day_root / "refeatured-records",
            day=day,
            raw_manifest_sha256=raw_manifest_sha256,
            frozen_schema_manifest_sha256="sha256:41c9ae94f47c0983bccb9e13c680ad5cf5d93f547aa61aeb91fea20cd53f62af",
            zero_bypass_receipt_sha256="sha256:be3d8d0bba4be02d10358f6e0512b9d740ede5c3f8bad9eca8f848915fb60d23",
        )
        records = census._merge_private_day_lane_spools(
            results,
            spool_root=private_root / "materialized",
            writer=writer,
            spool_field="materialized_private_spool_shards",
        )
        writer.close()
        manifest = writer.manifest()
        receipt = {
            "schema": "poke_bot.alakazam_recent20_intraday_refeature_day_receipt/v1",
            "status": "complete",
            "goal_revision": 9,
            "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
            "goal_contract_sha256": GOAL_CONTRACT_SHA256,
            "hostname": socket.gethostname(),
            "utc_day": day,
            "worker_count": workers,
            "private_spool_file_count": sum(
                len(result["materialized_private_spool_shards"]) for result in results
            ),
            "private_spool_bytes": _tree_bytes(private_root / "materialized"),
            "record_count": records,
            "maximum_final_shard_bytes": FINAL_SHARD_LIMIT_BYTES,
            "shard_manifest": manifest,
        }
        census._write_create_only_json(day_root / "COMPLETE.json", receipt)
        succeeded = True
        return receipt
    finally:
        if succeeded:
            shutil.rmtree(private_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--cg-runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument("--host-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 65), required=True)
    parser.add_argument("--resume-existing-output-root", action="store_true")
    parser.add_argument("--recover-day")
    parser.add_argument("--recover-private-root", type=Path)
    parser.add_argument("--rerun-lanes", default="")
    args = parser.parse_args()
    manifest = census._read_json(args.manifest)
    archives = census._manifest_archives(manifest, archive_root=args.archive_root)
    selected_dates = _day_dates(archives, args.host_index)
    archive_by_day = {str(meta["date"]): (meta, path) for meta, path in archives}
    if args.output_root.exists():
        if not args.resume_existing_output_root or not args.output_root.is_dir() or args.output_root.is_symlink():
            raise SystemExit("existing output root requires --resume-existing-output-root")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
    args.spool_root.mkdir(parents=True, exist_ok=True)
    if args.recover_day is not None:
        if args.recover_day not in selected_dates or args.recover_private_root is None:
            raise SystemExit("recovery day/private root is invalid for this host")
        rerun_lanes = {int(value) for value in args.rerun_lanes.split(",") if value.strip()}
        receipt = _recover_day(
            day=args.recover_day,
            archive=archive_by_day[args.recover_day],
            output_root=args.output_root,
            spool_root=args.spool_root,
            prior_private_root=args.recover_private_root.resolve(),
            workers=args.workers,
            rerun_lanes=rerun_lanes,
            cg_runtime_root=args.cg_runtime_root,
            raw_manifest_sha256=census.canonical_sha256(manifest),
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    receipts = []
    for day in selected_dates:
        receipts.append(
            _run_day(
                day=day,
                archive=archive_by_day[day],
                output_root=args.output_root,
                spool_root=args.spool_root,
                workers=args.workers,
                cg_runtime_root=args.cg_runtime_root,
                raw_manifest_sha256=census.canonical_sha256(manifest),
            )
        )
    complete = {
        "schema": "poke_bot.alakazam_recent20_intraday_refeature_host_receipt/v1",
        "status": "complete",
        "goal_revision": 9,
        "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
        "goal_contract_sha256": GOAL_CONTRACT_SHA256,
        "hostname": socket.gethostname(),
        "host_index": args.host_index,
        "window_start_utc": WINDOW_START,
        "window_end_utc": WINDOW_END,
        "day_count": len(receipts),
        "days": selected_dates,
        "worker_count_per_day": args.workers,
        "maximum_final_shard_bytes": FINAL_SHARD_LIMIT_BYTES,
        "record_count": sum(int(row["record_count"]) for row in receipts),
        "finalized_shard_count": sum(int(row["shard_manifest"]["shard_count"]) for row in receipts),
        "finalized_shard_bytes": sum(int(row["shard_manifest"]["total_bytes"]) for row in receipts),
        "day_receipts": [f"day-{day}/COMPLETE.json" for day in selected_dates],
    }
    census._write_create_only_json(args.output_root / "COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
