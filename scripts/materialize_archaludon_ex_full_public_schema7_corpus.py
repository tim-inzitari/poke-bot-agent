#!/usr/bin/env python3
"""Build an immutable full-public-history Archaludon ex schema-7 corpus.

Every selected daily shard is rematerialized from its original public replay
archive.  Schema-6 feature shards are intentionally ineligible for reuse
because they do not preserve ``PolicyStage.select_context`` or
``PolicyStage.selected_is_stop``.  Every archive, feature shard, sidecar,
daily receipt, aggregate manifest, protected pointer, and source-code lock is
checksum-validated before a final ready receipt can be published.
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
import subprocess
import sys
import time
from typing import Any
import zipfile


TARGET = "archaludon-ex"
START = date(2026, 6, 16)
END = date(2026, 7, 29)
MINIMUM_MATCHING_GAMES = 16_639
DATASET_SCHEMA = 7
FEATURE_SCHEMA = 5
EXPANDED_SCHEMA = "poke_bot.expanded_strategic_targets/v2"
EXPANDED_DIGEST = (
    "sha256:f086683173c94ff87360b4b692d2d5dcf81e122a2ce8271115d4ce9e2aba514f"
)
READY_SCHEMA = "poke_bot.archaludon_ex_full_public_corpus/v2"
SOURCE_ARCHIVE_SCHEMA = "poke_bot.archaludon_ex_full_public_sources/v2"
DAILY_RECEIPT_SCHEMA = "poke_bot.archaludon_ex_daily_corpus/v2"
STATUS_SCHEMA = "poke_bot.archaludon_ex_full_public_status/v2"
SOURCE_LOCK_SCHEMA = "poke_bot.archaludon_ex_schema7_source_lock/v1"
EXPECTED_MANIFEST_ONLY_IDS = {"2026-07-24": {"87841523"}}
REQUIRED_SOURCE_FILES = (
    "GOAL.md",
    "config/rl_protocol.yaml",
    "config/deck_guides/archaludon-ex.yaml",
    "docs/deck_guides/archaludon-ex-expert-brief.txt",
    "data/training_mixes/top_ladder.v1.json",
    "data/training_mixes/top_ladder_representatives.v1.json",
    "data/training_mixes/specialist_representatives.v1.json",
    "cards/EN_Card_Data.csv",
    "state/matchup_adapter_roster.json",
    "state/archaludon_public_full44_source_audit_v1.json",
    "poke_bot/authoritative_visual_trace.py",
    "poke_bot/archaludon_ex_heuristics.py",
    "poke_bot/ladder_replay.py",
    "poke_bot/deck_guides.py",
    "poke_bot/dataset.py",
    "poke_bot/feature_shards.py",
    "poke_bot/strategic_heads.py",
    "poke_bot/setup_board_outcome.py",
    "poke_bot/strategic_losses.py",
    "poke_bot/model.py",
    "poke_bot/train.py",
    "scripts/assemble_feature_manifest.py",
    "scripts/create_archaludon_ex_schema7_source_lock.py",
    "scripts/materialize_archaludon_ex_full_public_schema7_corpus.py",
    "scripts/materialize_archaludon_ex_latest20_corpus.py",
    "scripts/materialize_authoritative_guide_window_parallel.py",
    "scripts/materialize_authoritative_alakazam_day.py",
    "scripts/finalize_current_deck_guide_window.py",
    "ops/elmo/build_archaludon_ex_full_public_schema7_guide_corpus.sh",
    "ops/elmo/build_archaludon_ex_latest20_guide_corpus.sh",
    "ops/elmo/pokebot-archaludon-ex-full-public-schema7-r56-v1.service",
    "ops/elmo/pokebot-archaludon-ex-guide-full-public-schema7-r56-v1.service",
    "ops/systemd/pokebot-archaludon-ex-corpus-import.service",
)


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


def _available_memory_bytes() -> int:
    for row in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if row.startswith("MemAvailable:"):
            return int(row.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _validate_source_snapshot(
    source_root: Path,
    assembler: Path,
    cg_runtime: Path,
    source_lock_path: Path,
) -> dict[str, Any]:
    """Fail closed until revision-56+ source is frozen in a real lock."""

    lock = _read_json(source_lock_path)
    files = dict(lock.get("files") or {})
    if not (
        lock.get("schema") == SOURCE_LOCK_SCHEMA
        and lock.get("status") == "locked_checksum_validated"
        and int(lock.get("goal_revision", -1)) >= 56
        and int(lock.get("dataset_schema", -1)) == DATASET_SCHEMA
        and int(lock.get("feature_schema", -1)) == FEATURE_SCHEMA
        and lock.get("date_start") == START.isoformat()
        and lock.get("date_end") == END.isoformat()
        and int(lock.get("days", -1)) == len(_days())
        and int(lock.get("minimum_matching_games", -1))
        == MINIMUM_MATCHING_GAMES
        and set(REQUIRED_SOURCE_FILES).issubset(files)
        and str(lock.get("classifier_sha256") or "").startswith("sha256:")
        and lock.get("source_audit_sha256")
        == files.get("state/archaludon_public_full44_source_audit_v1.json")
        and str(lock.get("assembler_sha256") or "").startswith("sha256:")
        and str(lock.get("cg_library_sha256") or "").startswith("sha256:")
    ):
        raise RuntimeError(
            "Archaludon schema-7 source lock is absent, stale, or incomplete"
        )
    observed: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"locked strategic source is missing: {path}")
        digest = _sha256(path)
        if digest != str(files[relative]):
            raise RuntimeError(
                f"locked strategic source changed: {path} {digest}"
            )
        observed[relative] = digest
    assembler_digest = _sha256(assembler)
    if assembler_digest != lock["assembler_sha256"]:
        raise RuntimeError("locked manifest assembler changed")
    observed[str(assembler)] = assembler_digest
    library = cg_runtime / "cg/libcg.so"
    library_digest = _sha256(library)
    if library_digest != lock["cg_library_sha256"]:
        raise RuntimeError("locked native cg runtime changed")
    observed[str(library)] = library_digest
    return {
        "lock": lock,
        "lock_path": str(source_lock_path),
        "lock_sha256": _sha256(source_lock_path),
        "observed_files": observed,
        "classifier_sha256": str(lock["classifier_sha256"]),
    }


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


def _new_paths(output_root: Path, day: str) -> tuple[Path, Path, Path]:
    feature = output_root / f"{TARGET}-{day}.features"
    return (
        feature,
        feature.with_suffix(feature.suffix + ".json"),
        feature.with_name(feature.name + ".receipt.json"),
    )


def _preflight_sources(
    archive_root: Path,
    output_root: Path,
    source_audit_path: Path,
) -> dict[str, Any]:
    audit = _read_json(source_audit_path)
    audit_rows = {
        str(row["date"]): row for row in audit.get("daily_sources") or ()
    }
    audit_method = dict(audit.get("audit_method") or {})
    audited_matches = int(audit.get("matching_acting_seats", -1))
    daily_match_total = sum(
        int(row.get("matching_acting_seats", -1))
        for row in audit_rows.values()
    )
    label_method_total = sum(
        int(value) for value in dict(audit.get("label_methods") or {}).values()
    )
    if not (
        audit.get("schema")
        == "poke_bot.archaludon_ex_public_source_audit/v1"
        and audit.get("status")
        == "source_audit_complete_schema7_rematerialization_required"
        and audit.get("date_start") == START.isoformat()
        and audit.get("date_end") == END.isoformat()
        and int(audit.get("days", -1)) == len(_days())
        and audited_matches >= MINIMUM_MATCHING_GAMES
        and daily_match_total == audited_matches
        and label_method_total == audited_matches
        and int(audit.get("minimum_matching_games", -1))
        == MINIMUM_MATCHING_GAMES
        and audit.get("minimum_met_by_public_source_scan") is True
        and sorted(audit_rows) == _days()
        and str(
            audit_method.get("classifier_roster_sha256") or ""
        ).startswith("sha256:")
    ):
        raise RuntimeError("full-public Archaludon source audit is invalid")
    archives = [
        _validate_archive(
            archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            day,
        )
        for day in _days()
    ]
    for archive in archives:
        audited = audit_rows[archive["date"]]
        if (
            archive["sha256"] != audited.get("archive_sha256")
            or int(audited.get("matching_acting_seats", -1)) < 0
        ):
            raise RuntimeError(
                "public archive no longer matches its audited source: "
                f"{archive['date']}"
            )
    payload = {
        "schema": SOURCE_ARCHIVE_SCHEMA,
        "status": "validated",
        "date_start": START.isoformat(),
        "date_end": END.isoformat(),
        "days": len(_days()),
        "archives": archives,
        "source_audit": {
            "path": str(source_audit_path),
            "sha256": _sha256(source_audit_path),
            "matching_acting_seats": audited_matches,
            "classifier_roster_sha256": audit_method[
                "classifier_roster_sha256"
            ],
        },
        "materialization_dates": _days(),
        "schema6_feature_reuse_allowed": False,
        "reused_feature_dates": [],
        "all_archive_checksums_validated": True,
        "all_archive_manifest_memberships_validated": True,
    }
    _immutable_json(output_root / "SOURCE_ARCHIVES.json", payload)
    return payload


def _classifier(
    source_root: Path, roster_path: Path, expected_digest: str
) -> Any:
    from poke_bot.ladder_replay import LadderReplayClassifier

    roster = _read_json(roster_path)
    expert_ids = list(roster.get("expert_ids") or ())
    required_count = int(roster.get("required_specialist_count", -1))
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or required_count <= 0
        or len(expert_ids) != required_count
        or len(set(expert_ids)) != required_count
    ):
        raise RuntimeError("locked matchup-classifier roster is invalid")
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
    if digest != expected_digest:
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

    if _sha256(args.roster.resolve()) != str(args.roster_sha256):
        raise RuntimeError("selected classifier roster differs from source lock")
    classifier = _classifier(
        source_root,
        args.roster.resolve(),
        str(args.classifier_sha256),
    )
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


def _valid_day(
    output_root: Path,
    day: str,
    classifier_sha256: str,
    archive_sha256: str,
) -> dict[str, Any] | None:
    feature, sidecar, receipt = _new_paths(output_root, day)
    if not (feature.is_file() and sidecar.is_file() and receipt.is_file()):
        return None
    value = _read_json(receipt)
    metadata = _read_json(sidecar)
    stats = dict(value.get("stats") or {})
    if not (
        value.get("format") == "pokebot-authoritative-visual-day-receipt"
        and int(value.get("format_version", -1)) == 1
        and value.get("source_date") == day
        and (value.get("source_archive") or {}).get("sha256")
        == archive_sha256
        and (value.get("selection") or {}).get("acting_seat_archetype")
        == TARGET
        and (value.get("classifier") or {}).get("sha256")
        == classifier_sha256
        and int((value.get("schemas") or {}).get("dataset", -1))
        == DATASET_SCHEMA
        and int((value.get("schemas") or {}).get("feature", -1))
        == FEATURE_SCHEMA
        and (value.get("output") or {}).get("sha256") == _sha256(feature)
        and metadata.get("sha256") == _sha256(feature)
        and int(metadata.get("dataset_schema", -1)) == DATASET_SCHEMA
        and int(metadata.get("feature_schema", -1)) == FEATURE_SCHEMA
        and metadata.get("classifier_sha256") == classifier_sha256
        and metadata.get("source_archive_sha256") == archive_sha256
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
            "days": len(_days()),
        },
        "materialization_mode": "all_days_from_original_public_archives",
        "schema6_feature_reuse_allowed": False,
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
        "completed_days": ordered,
        "totals": {
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


def _materialize_days(
    args: argparse.Namespace,
    source_receipt: dict[str, Any],
    classifier_sha256: str,
) -> None:
    output_root = args.output_root.resolve()
    status_path = output_root / "status/window.json"
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    days = _days()
    archives = {
        str(row["date"]): str(row["sha256"])
        for row in source_receipt.get("archives") or ()
    }
    completed: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for day in days:
        row = _valid_day(
            output_root, day, classifier_sha256, archives[day]
        )
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
                "--classifier-sha256",
                classifier_sha256,
                "--roster-sha256",
                str(
                    source_receipt["source_audit"][
                        "classifier_roster_sha256"
                    ]
                ),
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
            row = _valid_day(
                output_root, day, classifier_sha256, archives[day]
            )
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
            errors=errors or ["daily feature set is incomplete"],
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
    source_snapshot: dict[str, Any],
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
    seen_episode_days: dict[str, str] = {}
    daily_rows: list[dict[str, Any]] = []
    total_records = 0
    total_decisions = 0
    total_policy_stages = 0
    total_setup_active_stages = 0
    total_setup_bench_stages = 0
    total_setup_stop_stages = 0
    classifier_sha256 = str(source_snapshot["classifier_sha256"])
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
            and int(header.get("dataset_schema", -1)) == DATASET_SCHEMA
            and int(header.get("feature_schema", -1)) == FEATURE_SCHEMA
            and header.get("compact_mode") == "temporal-expert-v1"
            and header.get("required_archetype") == TARGET
            and header.get("source_dates") == [day]
            and header.get("source_archive_sha256") == archive["sha256"]
            and header.get("classifier_sha256")
            == classifier_sha256
            and metadata.get("sha256") == feature_digest
            and int(metadata.get("dataset_schema", -1)) == DATASET_SCHEMA
            and int(metadata.get("feature_schema", -1)) == FEATURE_SCHEMA
            and metadata.get("classifier_sha256") == classifier_sha256
            and expanded.get("schema") == EXPANDED_SCHEMA
            and expanded.get("digest") == EXPANDED_DIGEST
        ):
            raise RuntimeError(f"daily feature identity failed: {feature}")
        records = 0
        decisions = 0
        day_episode_ids: set[str] = set()
        for sequence in iter_feature_shard(feature):
            if (
                str(sequence.archetype).strip().casefold() != TARGET
                or not bool(sequence.info_set_ok)
            ):
                raise RuntimeError(
                    f"noncausal or wrong-archetype sequence: {feature}"
                )
            episode_id = str(sequence.episode_id).strip()
            if not episode_id:
                raise RuntimeError(f"empty episode identity: {feature}")
            previous_day = seen_episode_days.get(episode_id)
            if previous_day is not None and previous_day != day:
                raise RuntimeError(
                    "episode identity appears in multiple source days: "
                    f"{episode_id} {previous_day} {day}"
                )
            seen_episode_days[episode_id] = day
            day_episode_ids.add(episode_id)
            seat = int(sequence.seat)
            if seat not in (0, 1):
                raise RuntimeError(f"invalid acting seat in {feature}: {seat}")
            key = (episode_id, seat)
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)
            records += 1
            decisions += len(sequence)
            for decision in sequence.decisions:
                for stage in decision.policy_stages:
                    if not (
                        hasattr(stage, "select_context")
                        and hasattr(stage, "selected_is_stop")
                    ):
                        raise RuntimeError(
                            "schema-7 setup metadata is missing from "
                            f"{feature}"
                        )
                    context = int(stage.select_context)
                    is_stop = bool(stage.selected_is_stop)
                    total_policy_stages += 1
                    if context == 1:
                        total_setup_active_stages += 1
                    elif context == 2:
                        total_setup_bench_stages += 1
                    if context in {1, 2} and is_stop:
                        total_setup_stop_stages += 1
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
        unique_episodes = len(day_episode_ids)
        mirror_episodes = records - unique_episodes
        if mirror_episodes < 0 or mirror_episodes > unique_episodes:
            raise RuntimeError(f"invalid daily mirror accounting: {feature}")
        source_kind = "original_public_archive_schema7_rematerialization"
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
            and (value.get("classifier") or {}).get("sha256")
            == classifier_sha256
            and int((value.get("schemas") or {}).get("dataset", -1))
            == DATASET_SCHEMA
            and int((value.get("schemas") or {}).get("feature", -1))
            == FEATURE_SCHEMA
        ):
            raise RuntimeError(f"schema-7 materialization receipt failed: {receipt}")
        source_day_receipt = {
            "path": str(receipt.relative_to(output_root)),
            "sha256": _sha256(receipt),
        }
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
                "classifier_sha256": classifier_sha256,
                "dataset_schema": DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "expanded_strategic_schema": EXPANDED_SCHEMA,
                "expanded_strategic_digest": EXPANDED_DIGEST,
            },
            "source_day_materialization_receipt": source_day_receipt,
            "matching_games": records,
            "matching_episodes": unique_episodes,
            "mirror_episodes": mirror_episodes,
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
                "materialization_receipt_sha256": source_day_receipt["sha256"],
                "records": records,
                "unique_episodes": unique_episodes,
                "mirror_episodes": mirror_episodes,
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
    if total_records < MINIMUM_MATCHING_GAMES:
        raise RuntimeError(
            "schema-7 Archaludon corpus is below the required public-game "
            f"floor: records={total_records} required={MINIMUM_MATCHING_GAMES}"
        )
    unique_episodes = len(seen_episode_days)
    mirror_episodes = total_records - unique_episodes
    single_seat_episodes = unique_episodes - mirror_episodes
    if (
        unique_episodes < MINIMUM_MATCHING_GAMES
        or mirror_episodes < 0
        or single_seat_episodes < 0
    ):
        raise RuntimeError(
            "schema-7 Archaludon corpus is below the literal unique-game "
            f"floor: unique_episodes={unique_episodes} "
            f"required={MINIMUM_MATCHING_GAMES}"
        )
    if (
        total_policy_stages <= 0
        or total_setup_active_stages <= 0
        or total_setup_bench_stages <= 0
    ):
        raise RuntimeError(
            "schema-7 setup-active/setup-bench metadata coverage is absent"
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
        "corpus_kind": "non_guide_full_public_history_schema7_primary",
        "source_policy": {
            "unit": "calendar_day",
            "window_selection": (
                "all_public_daily_sources_available_through_2026-07-29"
            ),
            "filter_applied_after_window_selection": True,
            "filter_archetype": TARGET,
            "date_start": START.isoformat(),
            "date_end": END.isoformat(),
            "days": len(_days()),
            "zero_match_days_retained": True,
            "schema6_feature_reuse_allowed": False,
        },
        "records": total_records,
        "unique_episodes": unique_episodes,
        "mirror_episodes": mirror_episodes,
        "single_seat_episodes": single_seat_episodes,
        "decisions": total_decisions,
        "minimum_matching_games": MINIMUM_MATCHING_GAMES,
        "minimum_matching_games_met": True,
        "minimum_unique_episode_games": MINIMUM_MATCHING_GAMES,
        "minimum_unique_episode_games_met": True,
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
            "all_checksums_validated": True,
            "all_manifest_memberships_validated": True,
            "days": len(_days()),
        },
        "manifest": "manifest.json",
        "manifest_sha256": _sha256(manifest_path),
        "protected_pointer": "PROTECTED_EXPERT_CORPUS.json",
        "protected_pointer_sha256": _sha256(pointer_path),
        "classifier_sha256": classifier_sha256,
        "dataset_schema": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "schema7_setup_metadata": {
            "policy_stages": total_policy_stages,
            "setup_active_stages": total_setup_active_stages,
            "setup_bench_stages": total_setup_bench_stages,
            "setup_stop_stages": total_setup_stop_stages,
            "select_context_preserved": True,
            "selected_is_stop_preserved": True,
        },
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
            "source_lock": {
                "path": source_snapshot["lock_path"],
                "sha256": source_snapshot["lock_sha256"],
            },
            "locked_source_files": source_snapshot["observed_files"],
            "reused_days": 0,
            "newly_featurized_days": len(_days()),
            "all_days_rematerialized_from_original_public_archives": True,
            "day_parallelism": int(args.day_parallelism),
            "workers_per_day": int(args.workers_per_day),
            "runtime_memory_floor_gib": float(
                args.runtime_memory_floor_gib
            ),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ready_path = output_root / "ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json"
    _immutable_json(ready_path, ready)
    return ready


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"unexpected symlink in immutable corpus: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _validate_existing_ready(
    output_root: Path,
    source_snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    ready_path = output_root / "ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json"
    if not ready_path.is_file():
        return None
    value = _read_json(ready_path)
    manifest = output_root / str(value.get("manifest") or "")
    pointer = output_root / str(value.get("protected_pointer") or "")
    sources = output_root / str(
        (value.get("source_archives") or {}).get("receipt") or ""
    )
    daily = list(value.get("daily_receipts") or ())
    provenance = dict(value.get("build_provenance") or {})
    locked_audit_digest = source_snapshot["lock"]["files"][
        "state/archaludon_public_full44_source_audit_v1.json"
    ]
    if not (
        value.get("schema") == READY_SCHEMA
        and value.get("status") == "ready_checksum_validated"
        and value.get("specialist_id") == TARGET
        and value.get("source_policy", {}).get("date_start")
        == START.isoformat()
        and value.get("source_policy", {}).get("date_end") == END.isoformat()
        and int(value.get("dataset_schema", -1)) == DATASET_SCHEMA
        and int(value.get("feature_schema", -1)) == FEATURE_SCHEMA
        and int(value.get("records", -1)) >= MINIMUM_MATCHING_GAMES
        and int(value.get("unique_episodes", -1)) >= MINIMUM_MATCHING_GAMES
        and value.get("minimum_matching_games_met") is True
        and value.get("minimum_unique_episode_games_met") is True
        and len(daily) == len(_days())
        and [row.get("date") for row in daily] == _days()
        and all(
            row.get("source_kind")
            == "original_public_archive_schema7_rematerialization"
            for row in daily
        )
        and value.get("classifier_sha256")
        == source_snapshot["classifier_sha256"]
        and (provenance.get("source_lock") or {}).get("sha256")
        == source_snapshot["lock_sha256"]
        and sources.is_file()
        and _sha256(sources)
        == (value.get("source_archives") or {}).get("receipt_sha256")
        and (_read_json(sources).get("source_audit") or {}).get("sha256")
        == locked_audit_digest
        and manifest.is_file()
        and _sha256(manifest) == value.get("manifest_sha256")
        and pointer.is_file()
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
            "/srv/poke-bot-agent/archive/episode-days"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/srv/poke-bot-agent/archive/"
            "archaludon-ex-full-public-2026-06-16_2026-07-29-"
            "schema7-r56-v1"
        ),
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=root / "state/archaludon_ex_schema7_source_lock_v1.json",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        default=root / "state/archaludon_public_full44_source_audit_v1.json",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=root / "state/matchup_adapter_roster.json",
    )
    parser.add_argument(
        "--assembler",
        type=Path,
        default=root / "scripts/assemble_feature_manifest.py",
    )
    parser.add_argument(
        "--cg-runtime",
        type=Path,
        default=Path(
            "/srv/poke-bot-agent/engine-runtimes/znver3-v1"
        ),
    )
    parser.add_argument("--day-parallelism", type=int, default=2)
    parser.add_argument("--workers-per-day", type=int, default=3)
    parser.add_argument("--max-in-flight-per-day", type=int, default=6)
    parser.add_argument("--start-memory-floor-gib", type=float, default=30.0)
    parser.add_argument("--runtime-memory-floor-gib", type=float, default=20.0)
    parser.add_argument(
        "--managed-service",
        default="pokebot-archaludon-ex-full-public-schema7-r56-v1.service",
    )
    parser.add_argument("--classifier-sha256", default="")
    parser.add_argument("--roster-sha256", default="")
    parser.add_argument("--day-mode", action="store_true")
    parser.add_argument("--day")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.day_mode:
        if (
            not args.day
            or not str(args.classifier_sha256).startswith("sha256:")
            or not str(args.roster_sha256).startswith("sha256:")
        ):
            parser.error(
                "--day-mode requires --day, --classifier-sha256, and "
                "--roster-sha256"
            )
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
    source_root = args.source_root.resolve()
    os.environ["CG_LIB_PATH"] = str(args.cg_runtime.resolve())
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    source_snapshot = _validate_source_snapshot(
        source_root,
        args.assembler.resolve(),
        args.cg_runtime.resolve(),
        args.source_lock.resolve(),
    )
    locked_audit_digest = source_snapshot["lock"]["files"][
        "state/archaludon_public_full44_source_audit_v1.json"
    ]
    locked_roster_digest = source_snapshot["lock"]["files"][
        "state/matchup_adapter_roster.json"
    ]
    if _sha256(args.source_audit.resolve()) != locked_audit_digest:
        raise RuntimeError("selected source audit differs from the source lock")
    if _sha256(args.roster.resolve()) != locked_roster_digest:
        raise RuntimeError("selected classifier roster differs from the source lock")
    source_audit = _read_json(args.source_audit.resolve())
    source_audit_method = dict(source_audit.get("audit_method") or {})
    if (
        source_audit_method.get("classifier_contract_sha256")
        != source_snapshot["classifier_sha256"]
        or source_audit_method.get("classifier_roster_sha256")
        != locked_roster_digest
    ):
        raise RuntimeError(
            "source audit classifier differs from the locked classifier"
        )
    existing = _validate_existing_ready(output_root, source_snapshot)
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
    source_receipt = _preflight_sources(
        args.archive_root.resolve(),
        output_root,
        args.source_audit.resolve(),
    )
    _materialize_days(
        args,
        source_receipt,
        str(source_snapshot["classifier_sha256"]),
    )
    _assemble(args)
    ready = _validate_and_seal(args, source_receipt, source_snapshot)
    _make_read_only(output_root)
    print(json.dumps(ready, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
