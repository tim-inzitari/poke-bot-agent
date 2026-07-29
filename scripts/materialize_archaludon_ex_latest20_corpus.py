#!/usr/bin/env python3
"""Build an immutable non-guide Archaludon ex corpus for 2026-07-08..27.

The first sixteen daily shards are copied byte-for-byte from the validated
2026-07-04..23 strategic corpus.  Only the four newly available dates are
featurized.  Every archive, feature shard, sidecar, daily receipt, aggregate
manifest, and protected pointer is checksum-validated before the final ready
receipt is published.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile


TARGET = "archaludon-ex"
START = date(2026, 7, 8)
END = date(2026, 7, 27)
REUSED_END = date(2026, 7, 23)
EXPECTED_CLASSIFIER_SHA256 = (
    "sha256:04bd15eac4fe1ce2cd5010f198d89201884d21f2968f04cf8ee66e49773f8011"
)
EXPANDED_SCHEMA = "poke_bot.expanded_strategic_targets/v2"
EXPANDED_DIGEST = (
    "sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"
)
READY_SCHEMA = "poke_bot.archaludon_ex_latest20_corpus/v1"
SOURCE_ARCHIVE_SCHEMA = "poke_bot.archaludon_ex_latest20_sources/v1"
DAILY_RECEIPT_SCHEMA = "poke_bot.archaludon_ex_daily_corpus/v1"
STATUS_SCHEMA = "poke_bot.archaludon_ex_latest20_status/v1"
EXPECTED_MANIFEST_ONLY_IDS = {"2026-07-24": {"87841523"}}
PINNED_SOURCE_SHA256 = {
    "data/training_mixes/top_ladder.v1.json": (
        "de9f2f5f65794ed8f0ffd3fff41b04aafd68f7ffc9d562ab2cfc774ea5cac79d"
    ),
    "data/training_mixes/top_ladder_representatives.v1.json": (
        "ee42f146fd746ed3dd953515a974b22eeb264ea8eb9513c6e69a41a524454002"
    ),
    "cards/EN_Card_Data.csv": (
        "408bc978661c8b0628e5f17b27693dc8da9c732472168f5574999be4774031c1"
    ),
    "state/matchup_adapter_roster.json": (
        "3828adddb8ad35e8a7e59964e12c01779affc7f3be8915dcba99a443f62637e4"
    ),
    "poke_bot/authoritative_visual_trace.py": (
        "54ebda910715a919036351aea43e1742ee675a2cfa1403d319330f63c0421791"
    ),
    "poke_bot/ladder_replay.py": (
        "597282038cf3a2f852ff2b5dc3c45fcfe9ec475db152fc2d9b80ea784178ea04"
    ),
    "poke_bot/strategic_heads.py": (
        "05a319486818fdd9ae468f11f1bca9db484f5b233a923840c26a831636fc14ad"
    ),
    "poke_bot/feature_shards.py": (
        "8391b1a12b18a1ff8cd2b52cf09c4dc8e6bf40182123e9c8ab8f25afb9730d4d"
    ),
}


def _days(first: date = START, last: date = END) -> list[str]:
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != value:
            raise RuntimeError(f"immutable JSON identity changed: {path}")
        return
    _atomic_json(path, value)


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    if target.is_file():
        if _sha256(target) != digest:
            raise RuntimeError(f"existing copied shard changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.copy.{os.getpid()}")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if _sha256(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copy digest mismatch: {source}")
    os.replace(temporary, target)


def _available_memory_bytes() -> int:
    for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _validate_source_snapshot(
    source_root: Path, assembler: Path, cg_runtime: Path
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = source_root / relative
        digest = _sha256(path)
        if digest != f"sha256:{expected}":
            raise RuntimeError(
                f"pinned strategic source changed: {path} {digest}"
            )
        observed[str(path)] = digest
    assembler_digest = _sha256(assembler)
    if (
        assembler_digest
        != "sha256:9d46bc4960d2e0a9d091088050a7afd5955eef11b4e8052797da9882c637434a"
    ):
        raise RuntimeError("empty-day-aware manifest assembler changed")
    observed[str(assembler)] = assembler_digest
    library = cg_runtime / "cg/libcg.so"
    library_digest = _sha256(library)
    if (
        library_digest
        != "sha256:aba43d7d86fc714ff2efd414e254ecaba37df543c279f0b6ff1b58a96214868e"
    ):
        raise RuntimeError("native cg runtime changed")
    observed[str(library)] = library_digest
    return observed


def _validate_archive(path: Path, day: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing public replay archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "manifest.csv" not in names:
            raise RuntimeError(f"archive has no manifest.csv: {path}")
        json_ids = [
            Path(name).stem
            for name in names
            if name.endswith(".json") and not name.endswith("/")
        ]
        manifest_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("manifest.csv").decode("utf-8-sig")
                )
            )
        )
    manifest_ids = [str(row.get("episode_id") or "") for row in manifest_rows]
    if (
        not json_ids
        or len(json_ids) != len(set(json_ids))
        or not manifest_ids
        or "" in manifest_ids
        or len(manifest_ids) != len(set(manifest_ids))
    ):
        raise RuntimeError(f"archive has duplicate or invalid episode IDs: {path}")
    missing = set(manifest_ids) - set(json_ids)
    orphaned = set(json_ids) - set(manifest_ids)
    if missing != EXPECTED_MANIFEST_ONLY_IDS.get(day, set()) or orphaned:
        raise RuntimeError(
            "archive/manifest membership changed: "
            f"day={day} missing={sorted(missing)} orphaned={sorted(orphaned)}"
        )
    return {
        "date": day,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "json_replays": len(json_ids),
        "manifest_rows": len(manifest_ids),
        "manifest_only_episode_ids": sorted(missing),
        "validated": True,
    }


def _reused_paths(reuse_root: Path, day: str) -> tuple[Path, Path]:
    feature = (
        reuse_root / f"all-recognized-{day}.{TARGET}.features"
    )
    return feature, feature.with_suffix(feature.suffix + ".json")


def _new_paths(output_root: Path, day: str) -> tuple[Path, Path, Path]:
    feature = output_root / f"{TARGET}-{day}.features"
    return (
        feature,
        feature.with_suffix(feature.suffix + ".json"),
        feature.with_name(feature.name + ".receipt.json"),
    )


def _preflight_sources(
    archive_root: Path, reuse_root: Path, output_root: Path
) -> dict[str, Any]:
    archives = [
        _validate_archive(
            archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            day,
        )
        for day in _days()
    ]
    by_day = {row["date"]: row for row in archives}
    reused: list[dict[str, Any]] = []
    for day in _days(START, REUSED_END):
        feature, sidecar = _reused_paths(reuse_root, day)
        metadata = _read_json(sidecar)
        feature_digest = _sha256(feature)
        expected_archive = by_day[day]["sha256"]
        if not (
            metadata.get("format") == "pokebot-bootstrap-feature-shard"
            and int(metadata.get("format_version", -1)) == 1
            and int(metadata.get("dataset_schema", -1)) == 6
            and int(metadata.get("feature_schema", -1)) == 5
            and metadata.get("compact_mode") == "temporal-expert-v1"
            and metadata.get("required_archetype") == TARGET
            and metadata.get("selection_archetype") == TARGET
            and metadata.get("source_dates") == [day]
            and metadata.get("source_archive_sha256") == expected_archive
            and metadata.get("classifier_sha256")
            == EXPECTED_CLASSIFIER_SHA256
            and metadata.get("sha256") == feature_digest
        ):
            raise RuntimeError(f"reused Archaludon shard is invalid: {feature}")
        target_feature = output_root / feature.name
        target_sidecar = output_root / sidecar.name
        _copy_verified(feature, target_feature, feature_digest)
        sidecar_digest = _sha256(sidecar)
        _copy_verified(sidecar, target_sidecar, sidecar_digest)
        stats = dict(metadata.get("stats") or {})
        reused.append(
            {
                "date": day,
                "source_feature": str(feature),
                "feature": str(target_feature),
                "feature_sha256": feature_digest,
                "sidecar_sha256": sidecar_digest,
                "source_archive_sha256": expected_archive,
                "records": int(stats.get("records_kept", 0)),
                "decisions": int(stats.get("decisions_kept", 0)),
            }
        )
    payload = {
        "schema": SOURCE_ARCHIVE_SCHEMA,
        "status": "validated",
        "date_start": START.isoformat(),
        "date_end": END.isoformat(),
        "days": 20,
        "archives": archives,
        "reused_daily_features": reused,
        "new_feature_dates": _days(REUSED_END + timedelta(days=1), END),
        "all_archive_checksums_validated": True,
        "all_archive_manifest_memberships_validated": True,
    }
    _immutable_json(output_root / "SOURCE_ARCHIVES.json", payload)
    return payload


def _classifier(source_root: Path, roster_path: Path) -> Any:
    from poke_bot.ladder_replay import LadderReplayClassifier

    roster = _read_json(roster_path)
    expert_ids = list(roster.get("expert_ids") or ())
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or int(roster.get("required_specialist_count", -1)) != 18
        or len(expert_ids) != 18
        or len(set(expert_ids)) != 18
    ):
        raise RuntimeError("pinned roster-18 source is invalid")
    if TARGET not in expert_ids:
        raise RuntimeError("Archaludon is absent from the pinned classifier roster")
    classifier = LadderReplayClassifier.from_paths(
        source_root / "data/training_mixes/top_ladder.v1.json",
        source_root
        / "data/training_mixes/top_ladder_representatives.v1.json",
        card_csv=source_root / "cards/EN_Card_Data.csv",
        additive_registered_ids=expert_ids,
    )
    digest = _canonical_digest(classifier.contract)
    if digest != EXPECTED_CLASSIFIER_SHA256:
        raise RuntimeError(
            f"Archaludon classifier identity changed: {digest}"
        )
    return classifier


def _materialize_day(args: argparse.Namespace) -> int:
    os.environ["CG_LIB_PATH"] = str(args.cg_runtime)
    source_root = args.source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from poke_bot.authoritative_visual_trace import materialize_day

    classifier = _classifier(source_root, args.roster.resolve())
    archive = (
        args.archive_root.resolve()
        / f"pokemon-tcg-ai-battle-episodes-{args.day}.zip"
    )
    feature, _sidecar, _receipt = _new_paths(
        args.output_root.resolve(), args.day
    )
    result = materialize_day(
        archive,
        feature,
        classifier=classifier,
        source_date=args.day,
        workers=int(args.workers_per_day),
        max_in_flight=int(args.max_in_flight_per_day),
        max_context=320,
        resume=True,
        min_available_bytes=int(args.runtime_memory_floor_gib * 1024**3),
        min_records=0,
        required_archetype=TARGET,
    )
    print(
        json.dumps(
            {
                "date": args.day,
                "records": int((result.get("stats") or {}).get("records_kept", 0)),
                "decisions": int(
                    (result.get("stats") or {}).get("decisions_kept", 0)
                ),
                "sha256": (result.get("output") or {}).get("sha256"),
                "resumed": bool(result.get("resumed")),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _valid_new_day(output_root: Path, day: str) -> dict[str, Any] | None:
    feature, sidecar, receipt = _new_paths(output_root, day)
    if not (feature.is_file() and sidecar.is_file() and receipt.is_file()):
        return None
    value = _read_json(receipt)
    stats = dict(value.get("stats") or {})
    if not (
        value.get("format") == "pokebot-authoritative-visual-day-receipt"
        and int(value.get("format_version", -1)) == 1
        and value.get("source_date") == day
        and (value.get("selection") or {}).get("acting_seat_archetype")
        == TARGET
        and (value.get("classifier") or {}).get("sha256")
        == EXPECTED_CLASSIFIER_SHA256
        and (value.get("output") or {}).get("sha256") == _sha256(feature)
    ):
        return None
    return {
        "date": day,
        "records": int(stats.get("records_kept", 0)),
        "decisions": int(stats.get("decisions_kept", 0)),
        "sha256": (value.get("output") or {}).get("sha256"),
        "resumed": True,
    }


def _publish_status(
    path: Path,
    *,
    state: str,
    started_at: float,
    completed: dict[str, dict[str, Any]],
    running: dict[str, tuple[subprocess.Popen[bytes], Any]],
    args: argparse.Namespace,
    errors: list[str] | None = None,
) -> None:
    ordered = [completed[day] for day in sorted(completed)]
    value: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "state": state,
        "managed_service": args.managed_service,
        "started_at": started_at,
        "updated_at": time.time(),
        "date_window": {
            "start": START.isoformat(),
            "end": END.isoformat(),
            "days": 20,
        },
        "new_feature_window": {
            "start": (REUSED_END + timedelta(days=1)).isoformat(),
            "end": END.isoformat(),
            "days": 4,
        },
        "parallel_contract": {
            "day_parallelism": int(args.day_parallelism),
            "workers_per_day": int(args.workers_per_day),
            "maximum_worker_processes": (
                int(args.day_parallelism) * int(args.workers_per_day)
            ),
        },
        "current_dates": sorted(running),
        "current_pids": {
            day: process.pid for day, (process, _stream) in running.items()
        },
        "completed_new_days": ordered,
        "totals_new_days": {
            "days": len(ordered),
            "records": sum(int(row["records"]) for row in ordered),
            "decisions": sum(int(row["decisions"]) for row in ordered),
        },
        "current_promoted_corpus_modified": False,
    }
    if errors:
        value["errors"] = list(errors)
    if state == "complete":
        value["completed_at"] = time.time()
        value["elapsed_seconds"] = time.time() - started_at
    _atomic_json(path, value)


def _materialize_new_days(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    status_path = output_root / "status/window.json"
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    days = _days(REUSED_END + timedelta(days=1), END)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for day in days:
        row = _valid_new_day(output_root, day)
        if row is None:
            pending.append(day)
        else:
            completed[day] = row
    running: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
    errors: list[str] = []
    started_at = time.time()
    while pending or running:
        while pending and len(running) < int(args.day_parallelism):
            day = pending.pop(0)
            log_stream = (logs / f"{TARGET}-{day}.log").open("ab", buffering=0)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--day-mode",
                "--day",
                day,
                "--source-root",
                str(args.source_root),
                "--archive-root",
                str(args.archive_root),
                "--output-root",
                str(args.output_root),
                "--roster",
                str(args.roster),
                "--cg-runtime",
                str(args.cg_runtime),
                "--workers-per-day",
                str(args.workers_per_day),
                "--max-in-flight-per-day",
                str(args.max_in_flight_per_day),
                "--runtime-memory-floor-gib",
                str(args.runtime_memory_floor_gib),
            ]
            running[day] = (
                subprocess.Popen(
                    command,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                ),
                log_stream,
            )
        _publish_status(
            status_path,
            state="running",
            started_at=started_at,
            completed=completed,
            running=running,
            args=args,
            errors=errors,
        )
        if not running:
            continue
        time.sleep(1.0)
        for day, (process, stream) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[day]
            if return_code:
                errors.append(f"{day}: exit {return_code}")
                continue
            row = _valid_new_day(output_root, day)
            if row is None:
                errors.append(f"{day}: completed without a valid receipt")
            else:
                completed[day] = row
    if errors or sorted(completed) != days:
        _publish_status(
            status_path,
            state="failed",
            started_at=started_at,
            completed=completed,
            running=running,
            args=args,
            errors=errors or ["new daily feature set is incomplete"],
        )
        raise RuntimeError("one or more Archaludon daily jobs failed")
    _publish_status(
        status_path,
        state="complete",
        started_at=started_at,
        completed=completed,
        running=running,
        args=args,
    )


def _assemble(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(args.assembler.resolve()),
        "--staging-dir",
        str(args.output_root.resolve()),
        "--out",
        str(args.output_root.resolve() / "manifest.json"),
        "--min-free-gib",
        "5",
        "--compact-mode",
        "temporal-expert-v1",
        "--required-archetype",
        TARGET,
        "--expected-max-context",
        "320",
        "--seal-protected",
        "--allow-empty-shards",
    ]
    for day in _days():
        command.extend(["--expected-date", day])
    for target in (
        "temporal_action_rows",
        "opponent_hand_rows",
        "opponent_remainder_rows",
        "lethal_threat_rows",
        "prize_race_rows",
    ):
        command.extend(["--require-target-coverage", target])
    subprocess.run(command, check=True)


def _validate_and_seal(
    args: argparse.Namespace,
    source_receipt: dict[str, Any],
    source_checksums: dict[str, str],
) -> dict[str, Any]:
    from poke_bot.feature_shards import iter_feature_shard

    output_root = args.output_root.resolve()
    archives = {
        row["date"]: row for row in source_receipt.get("archives") or ()
    }
    sidecars: dict[str, Path] = {}
    for sidecar in output_root.glob("*.features.json"):
        metadata = _read_json(sidecar)
        dates = list(metadata.get("source_dates") or ())
        if len(dates) != 1:
            raise RuntimeError(f"daily sidecar has invalid dates: {sidecar}")
        day = str(dates[0])
        if day in sidecars:
            raise RuntimeError(f"duplicate feature sidecar for {day}")
        sidecars[day] = sidecar
    if sorted(sidecars) != _days():
        raise RuntimeError("Archaludon daily sidecar coverage is incomplete")

    duplicate_keys: set[tuple[str, int]] = set()
    seen_keys: set[tuple[str, int]] = set()
    daily_rows: list[dict[str, Any]] = []
    total_records = 0
    total_decisions = 0
    for day in _days():
        sidecar = sidecars[day]
        metadata = _read_json(sidecar)
        feature = output_root / str(metadata.get("path") or "")
        stats = dict(metadata.get("stats") or {})
        expanded = dict(stats.get("expanded_strategic_targets") or {})
        coverage = dict(stats.get("target_coverage") or {})
        archive = archives[day]
        feature_digest = _sha256(feature)
        with feature.open("rb") as stream:
            header = pickle.load(stream)
        if not (
            isinstance(header, dict)
            and header.get("format") == "pokebot-bootstrap-feature-shard"
            and int(header.get("format_version", -1)) == 1
            and int(header.get("dataset_schema", -1)) == 6
            and int(header.get("feature_schema", -1)) == 5
            and header.get("compact_mode") == "temporal-expert-v1"
            and header.get("required_archetype") == TARGET
            and header.get("source_dates") == [day]
            and header.get("source_archive_sha256") == archive["sha256"]
            and header.get("classifier_sha256")
            == EXPECTED_CLASSIFIER_SHA256
            and metadata.get("sha256") == feature_digest
            and expanded.get("schema") == EXPANDED_SCHEMA
            and expanded.get("digest") == EXPANDED_DIGEST
        ):
            raise RuntimeError(f"daily feature identity failed: {feature}")
        records = 0
        decisions = 0
        for sequence in iter_feature_shard(feature):
            if (
                str(sequence.archetype).strip().casefold() != TARGET
                or not bool(sequence.info_set_ok)
            ):
                raise RuntimeError(
                    f"noncausal or wrong-archetype sequence: {feature}"
                )
            key = (str(sequence.episode_id), int(sequence.seat))
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)
            records += 1
            decisions += len(sequence)
        if (
            records != int(stats.get("records_kept", -1))
            or decisions != int(stats.get("decisions_kept", -1))
            or int(expanded.get("decisions", -1)) != decisions
            or any(
                int(coverage.get(name, 0)) != decisions
                for name in (
                    "temporal_action_rows",
                    "opponent_hand_rows",
                    "opponent_remainder_rows",
                    "lethal_threat_rows",
                    "prize_race_rows",
                )
            )
            or int(coverage.get("guide_rows", 0)) != 0
        ):
            raise RuntimeError(f"daily count/coverage failed: {feature}")
        source_kind = (
            "reused_checksum_validated_strategic_shard"
            if day <= REUSED_END.isoformat()
            else "new_public_archive_featurization"
        )
        source_day_receipt: dict[str, Any] | None = None
        source_day_receipt_sha256: str | None = None
        if source_kind == "new_public_archive_featurization":
            receipt = feature.with_name(feature.name + ".receipt.json")
            value = _read_json(receipt)
            if not (
                value.get("source_date") == day
                and (value.get("source_archive") or {}).get("sha256")
                == archive["sha256"]
                and (value.get("output") or {}).get("sha256")
                == feature_digest
                and (value.get("selection") or {}).get(
                    "acting_seat_archetype"
                )
                == TARGET
            ):
                raise RuntimeError(f"new feature receipt failed: {receipt}")
            source_day_receipt = {
                "path": str(receipt.relative_to(output_root)),
                "sha256": _sha256(receipt),
            }
            source_day_receipt_sha256 = source_day_receipt["sha256"]
        daily_receipt = {
            "schema": DAILY_RECEIPT_SCHEMA,
            "status": "validated",
            "specialist_id": TARGET,
            "source_date": day,
            "source_kind": source_kind,
            "source_archive": archive,
            "feature": {
                "path": str(feature.relative_to(output_root)),
                "sha256": feature_digest,
                "bytes": feature.stat().st_size,
                "sidecar": str(sidecar.relative_to(output_root)),
                "sidecar_sha256": _sha256(sidecar),
                "classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
                "dataset_schema": 6,
                "feature_schema": 5,
                "expanded_strategic_schema": EXPANDED_SCHEMA,
                "expanded_strategic_digest": EXPANDED_DIGEST,
            },
            "source_day_materialization_receipt": source_day_receipt,
            "matching_games": records,
            "matching_decisions": decisions,
            "zero_match": records == 0,
            "guide_rows": 0,
            "causal_info_set_validated": True,
            "duplicate_episode_seat_keys": 0,
        }
        daily_path = output_root / f"daily-receipts/{day}.json"
        _immutable_json(daily_path, daily_receipt)
        daily_rows.append(
            {
                "date": day,
                "receipt": str(daily_path.relative_to(output_root)),
                "receipt_sha256": _sha256(daily_path),
                "source_archive_sha256": archive["sha256"],
                "feature_sha256": feature_digest,
                "materialization_receipt_sha256": source_day_receipt_sha256,
                "records": records,
                "decisions": decisions,
                "zero_match": records == 0,
                "source_kind": source_kind,
            }
        )
        total_records += records
        total_decisions += decisions
    if duplicate_keys:
        raise RuntimeError(
            f"duplicate episode/seat keys: {sorted(duplicate_keys)[:10]}"
        )

    manifest_path = output_root / "manifest.json"
    pointer_path = output_root / "PROTECTED_EXPERT_CORPUS.json"
    manifest = _read_json(manifest_path)
    pointer = _read_json(pointer_path)
    manifest_expanded = dict(manifest.get("expanded_strategic_targets") or {})
    manifest_totals = dict(manifest.get("totals") or {})
    if not (
        manifest.get("format") == "pokebot-bootstrap-feature-manifest"
        and manifest.get("dates") == _days()
        and (manifest.get("selection") or {}).get("value") == TARGET
        and int(manifest_totals.get("records_kept", -1)) == total_records
        and int(manifest_totals.get("decisions_kept", -1)) == total_decisions
        and manifest_expanded.get("schema") == EXPANDED_SCHEMA
        and manifest_expanded.get("digest") == EXPANDED_DIGEST
        and int(manifest_expanded.get("decisions", -1)) == total_decisions
        and pointer.get("schema") == "poke_bot.pinned_expert_corpus/v1"
        and pointer.get("protected") is True
        and pointer.get("manifest") == manifest_path.name
        and pointer.get("manifest_sha256") == _sha256(manifest_path)
    ):
        raise RuntimeError("aggregate manifest or protected pointer failed")

    ready = {
        "schema": READY_SCHEMA,
        "status": "ready_checksum_validated",
        "specialist_id": TARGET,
        "corpus_kind": "non_guide_latest20_primary",
        "source_policy": {
            "unit": "calendar_day",
            "window_selection": "latest_available_fully_validated_daily_sources",
            "filter_applied_after_window_selection": True,
            "filter_archetype": TARGET,
            "date_start": START.isoformat(),
            "date_end": END.isoformat(),
            "days": 20,
            "zero_match_days_retained": True,
        },
        "records": total_records,
        "decisions": total_decisions,
        "guide_rows": 0,
        "non_guide_corpus": True,
        "zero_match_dates": [
            row["date"] for row in daily_rows if row["zero_match"]
        ],
        "nonzero_dates": [
            row["date"] for row in daily_rows if not row["zero_match"]
        ],
        "daily_receipts": daily_rows,
        "source_archives": {
            "receipt": "SOURCE_ARCHIVES.json",
            "receipt_sha256": _sha256(output_root / "SOURCE_ARCHIVES.json"),
            "all_20_checksums_validated": True,
            "all_20_manifest_memberships_validated": True,
        },
        "manifest": "manifest.json",
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer": "PROTECTED_EXPERT_CORPUS.json",
        "protected_pointer_sha256": _sha256(pointer_path),
        "classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
        "dataset_schema": 6,
        "feature_schema": 5,
        "expanded_strategic_schema": EXPANDED_SCHEMA,
        "expanded_strategic_digest": EXPANDED_DIGEST,
        "all_daily_archive_and_feature_checksums_verified": True,
        "per_day_zero_nonzero_provenance_complete": True,
        "duplicate_episode_seat_keys": 0,
        "causal_info_set_validated": True,
        "immutable_staging_artifact": True,
        "promotion_status": "not_requested",
        "current_promoted_corpus_modified": False,
        "managed_service": args.managed_service,
        "build_provenance": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "pinned_source_files": source_checksums,
            "reused_days": 16,
            "newly_featurized_days": 4,
            "day_parallelism": int(args.day_parallelism),
            "workers_per_day": int(args.workers_per_day),
            "runtime_memory_floor_gib": float(
                args.runtime_memory_floor_gib
            ),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ready_path = output_root / "ARCHALUDON_EX_LATEST20_CORPUS_READY.json"
    _immutable_json(ready_path, ready)
    return ready


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in immutable corpus: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _validate_existing_ready(output_root: Path) -> dict[str, Any] | None:
    ready_path = output_root / "ARCHALUDON_EX_LATEST20_CORPUS_READY.json"
    if not ready_path.is_file():
        return None
    value = _read_json(ready_path)
    manifest = output_root / str(value.get("manifest") or "")
    pointer = output_root / str(value.get("protected_pointer") or "")
    daily = list(value.get("daily_receipts") or ())
    if not (
        value.get("schema") == READY_SCHEMA
        and value.get("status") == "ready_checksum_validated"
        and value.get("specialist_id") == TARGET
        and value.get("source_policy", {}).get("date_start")
        == START.isoformat()
        and value.get("source_policy", {}).get("date_end") == END.isoformat()
        and len(daily) == 20
        and [row.get("date") for row in daily] == _days()
        and _sha256(manifest) == value.get("manifest_sha256")
        and _sha256(pointer) == value.get("protected_pointer_sha256")
        and all(
            _sha256(output_root / str(row["receipt"]))
            == row.get("receipt_sha256")
            for row in daily
        )
    ):
        raise RuntimeError("existing Archaludon ready artifact is invalid")
    return value


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root,
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(
            "/mnt/Main/main/poke-bot-agent/archive/episode-days"
        ),
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        default=Path(
            "/mnt/Main/main/poke-bot-agent/archive/"
            "expert-latest20-derived/windows/2026-07-04_2026-07-23/"
            "roster18-v6-strategic-aliasfix-v2/specialist-corpora/"
            "archaludon-ex"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/Main/main/poke-bot-agent/archive/"
            "archaludon-ex-latest20-2026-07-08_2026-07-27-v1"
        ),
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=root / "state/matchup_adapter_roster.json",
    )
    parser.add_argument(
        "--assembler",
        type=Path,
        default=Path(
            "/home/admin/pokebot-expert-guide-src-v1/"
            "scripts/assemble_feature_manifest.py"
        ),
    )
    parser.add_argument(
        "--cg-runtime",
        type=Path,
        default=Path(
            "/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1"
        ),
    )
    parser.add_argument("--day-parallelism", type=int, default=2)
    parser.add_argument("--workers-per-day", type=int, default=3)
    parser.add_argument("--max-in-flight-per-day", type=int, default=6)
    parser.add_argument("--start-memory-floor-gib", type=float, default=30.0)
    parser.add_argument("--runtime-memory-floor-gib", type=float, default=20.0)
    parser.add_argument(
        "--managed-service",
        default="pokebot-archaludon-ex-latest20-corpus-v1.service",
    )
    parser.add_argument("--day-mode", action="store_true")
    parser.add_argument("--day")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.day_mode:
        if not args.day:
            parser.error("--day-mode requires --day")
        return _materialize_day(args)
    if (
        int(args.day_parallelism) < 1
        or int(args.workers_per_day) < 1
        or int(args.max_in_flight_per_day) < 1
        or float(args.start_memory_floor_gib) <= 0
        or float(args.runtime_memory_floor_gib) <= 0
    ):
        parser.error("parallelism and memory floors must be positive")
    output_root = args.output_root.resolve()
    existing = _validate_existing_ready(output_root)
    if existing is not None:
        print(json.dumps(existing, sort_keys=True), flush=True)
        return 0
    available = _available_memory_bytes()
    required = int(float(args.start_memory_floor_gib) * 1024**3)
    if available < required:
        raise RuntimeError(
            "start memory guard failed: "
            f"available={available} required={required}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    os.environ["CG_LIB_PATH"] = str(args.cg_runtime.resolve())
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    source_checksums = _validate_source_snapshot(
        source_root,
        args.assembler.resolve(),
        args.cg_runtime.resolve(),
    )
    source_receipt = _preflight_sources(
        args.archive_root.resolve(),
        args.reuse_root.resolve(),
        output_root,
    )
    _materialize_new_days(args)
    _assemble(args)
    ready = _validate_and_seal(args, source_receipt, source_checksums)
    _make_read_only(output_root)
    print(json.dumps(ready, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
