#!/usr/bin/env python3
"""Assemble and split an exact checksum-bound latest-20 expert window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT, SHARD_FORMAT
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    EXPANDED_STRATEGIC_SCHEMA_DIGEST,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from scripts.split_expert_manifest_by_archetype import split_manifest


RECEIPT_SCHEMA = "poke_bot.latest20_specialist_corpora/v1"
DAILY_SCHEMA = "poke_bot.latest20_daily_feature_selection/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def feature_identity(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"feature shard or sidecar is absent: {path}")
    sidecar = read_json(sidecar_path)
    with path.open("rb") as stream:
        header = pickle.load(stream)
    if not isinstance(header, dict) or header.get("format") != SHARD_FORMAT:
        raise RuntimeError(f"feature shard header is invalid: {path}")
    return {
        "path": path,
        "sidecar": sidecar_path,
        "header": header,
        "metadata": sidecar,
    }


def select_sources(
    receipt: dict[str, Any],
    candidate_roots: list[Path],
) -> list[dict[str, Any]]:
    if (
        receipt.get("schema") != "poke_bot.expert_latest20_receipt/v1"
        or receipt.get("status") != "ready"
        or int(receipt.get("days") or 0) != 20
        or len(receipt.get("archives") or ()) != 20
    ):
        raise RuntimeError("latest-20 archive receipt is not ready")
    selected: list[dict[str, Any]] = []
    for archive in receipt["archives"]:
        day = str(archive.get("date") or "")
        archive_digest = str(archive.get("sha256") or "")
        if (
            not day
            or not archive_digest.startswith("sha256:")
            or archive.get("validated") is not True
        ):
            raise RuntimeError("latest-20 archive row is invalid")
        valid: list[dict[str, Any]] = []
        for root in candidate_roots:
            path = root / f"all-recognized-{day}.features"
            if not path.is_file():
                continue
            identity = feature_identity(path)
            header = identity["header"]
            metadata = identity["metadata"]
            if (
                list(header.get("source_dates") or ()) == [day]
                and str(header.get("source_archive_sha256") or "")
                == archive_digest
                and str(header.get("required_archetype") or "") == "*"
                and str(header.get("compact_mode") or "")
                == COMPACT_MODE_TEMPORAL_EXPERT
                and int(header.get("max_context") or 0) == 320
                and list(metadata.get("source_dates") or ()) == [day]
                and str(metadata.get("source_archive_sha256") or "")
                == archive_digest
            ):
                valid.append(identity)
        if not valid:
            raise RuntimeError(
                f"no checksum-compatible all-recognized feature shard: {day}"
            )
        chosen = valid[0]
        selected.append(
            {
                "date": day,
                "archive_sha256": archive_digest,
                "source": str(chosen["path"]),
                "source_sha256": sha256(chosen["path"]),
                "sidecar": str(chosen["sidecar"]),
                "sidecar_sha256": sha256(chosen["sidecar"]),
            }
        )
    dates = [row["date"] for row in selected]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise RuntimeError("latest-20 selected dates are not unique and sorted")
    return selected


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def grant_publish_reader(root: Path, reader: str) -> None:
    """Grant the transfer identity read/traverse access without changing owners."""

    if not reader:
        return
    # Minimal worker images do not necessarily ship the ACL utilities. The
    # managed container wrapper applies the shared group and g+rX after this
    # finalizer exits successfully, so absence of setfacl must not invalidate
    # an otherwise complete checksum-bound corpus.
    setfacl = shutil.which("setfacl")
    if setfacl is None:
        return
    subprocess.run(
        [setfacl, "-R", "-m", f"u:{reader}:rX", str(root)],
        check=True,
    )
    subprocess.run(
        [setfacl, "-m", f"d:u:{reader}:rX", str(root)],
        check=True,
    )


def materialize_daily(
    output_root: Path,
    selected: list[dict[str, Any]],
    *,
    source_repo: Path,
) -> tuple[Path, Path]:
    daily = output_root / "daily"
    ready = daily / "DAILY_READY.json"
    identity = {
        "schema": DAILY_SCHEMA,
        "dates": [row["date"] for row in selected],
        "sources": selected,
    }
    if ready.is_file():
        existing = read_json(ready)
        if existing != identity:
            raise RuntimeError("existing latest-20 daily selection changed")
        return daily, daily / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".daily.", dir=str(output_root))
    )
    try:
        for row in selected:
            name = f"all-recognized-{row['date']}.features"
            link_or_copy(Path(row["source"]), temporary / name)
            link_or_copy(Path(row["sidecar"]), temporary / f"{name}.json")
        expected: list[str] = []
        for row in selected:
            expected.extend(["--expected-date", row["date"]])
        manifest = temporary / "manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(source_repo / "scripts/assemble_feature_manifest.py"),
                "--staging-dir",
                str(temporary),
                "--out",
                str(manifest),
                *expected,
                "--min-free-gib",
                "5",
                "--compact-mode",
                COMPACT_MODE_TEMPORAL_EXPERT,
                "--required-archetype",
                "*",
                "--expected-max-context",
                "320",
            ],
            check=True,
        )
        atomic_json(temporary / ready.name, identity)
        os.replace(temporary, daily)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return daily, daily / "manifest.json"


def materialize_balanced_core(
    manifest: Path,
    output_root: Path,
    *,
    archetypes: list[str],
    source_repo: Path,
) -> dict[str, Any] | None:
    """Build the expanded-head cumulative-core corpus when V2 targets exist."""

    source = read_json(manifest)
    expanded = source.get("expanded_strategic_targets")
    if expanded is None:
        return None
    if (
        not isinstance(expanded, dict)
        or expanded.get("schema") != EXPANDED_STRATEGIC_SCHEMA
        or expanded.get("digest") != EXPANDED_STRATEGIC_SCHEMA_DIGEST
        or set(expanded.get("head_coverage") or ())
        != set(EXPANDED_HEAD_IDS)
    ):
        raise RuntimeError("daily expanded strategic target identity changed")
    core_root = output_root / "core-balanced-v6"
    command = [
        sys.executable,
        str(source_repo / "scripts/build_balanced_core_manifest.py"),
        "--source-manifest",
        str(manifest),
        "--output-dir",
        str(core_root),
        "--max-records-per-archetype",
        "2500",
        "--max-decisions-per-archetype",
        "220000",
        "--required-expanded-target-schema",
        EXPANDED_STRATEGIC_SCHEMA,
        "--required-expanded-target-digest",
        EXPANDED_STRATEGIC_SCHEMA_DIGEST,
    ]
    for archetype in archetypes:
        command.extend(["--additive-archetype", archetype])
    subprocess.run(command, check=True)
    pointer = core_root / "PROTECTED_CORE_CORPUS.json"
    protected = read_json(pointer)
    core_manifest = core_root / str(protected.get("manifest") or "")
    manifest_payload = read_json(core_manifest)
    core_expanded = manifest_payload.get("expanded_strategic_targets")
    decisions = int(
        (manifest_payload.get("totals") or {}).get("decisions_kept") or 0
    )
    if (
        protected.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or protected.get("protected") is not True
        or not core_manifest.is_file()
        or sha256(core_manifest) != protected.get("manifest_sha256")
        or not isinstance(core_expanded, dict)
        or core_expanded.get("schema") != EXPANDED_STRATEGIC_SCHEMA
        or core_expanded.get("digest")
        != EXPANDED_STRATEGIC_SCHEMA_DIGEST
        or int(core_expanded.get("decisions") or -1) != decisions
        or set(core_expanded.get("head_coverage") or ())
        != set(EXPANDED_HEAD_IDS)
        or protected.get("expanded_strategic_targets") != core_expanded
    ):
        raise RuntimeError("balanced V6 core corpus failed validation")
    return {
        "root": str(core_root),
        "pointer": str(pointer),
        "pointer_sha256": sha256(pointer),
        "manifest": str(core_manifest),
        "manifest_sha256": sha256(core_manifest),
        "decisions": decisions,
        "expanded_strategic_targets": core_expanded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, default=ROOT)
    parser.add_argument("--minimum-decisions", type=int, default=1)
    parser.add_argument(
        "--publish-reader",
        default="admin",
        help="Elmo account granted recursive read/traverse access for promotion",
    )
    args = parser.parse_args()

    receipt = read_json(args.archive_receipt.resolve())
    roster = read_json(args.roster.resolve())
    archetypes = list(roster.get("expert_ids") or ())
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or int(roster.get("required_specialist_count") or 0) != 18
        or len(archetypes) != 18
        or len(set(archetypes)) != 18
    ):
        raise RuntimeError("canonical 18-specialist roster is invalid")
    selected = select_sources(
        receipt,
        [path.resolve() for path in args.candidate_root],
    )
    output_root = args.output_root.resolve()
    _daily, manifest = materialize_daily(
        output_root,
        selected,
        source_repo=args.source_repo.resolve(),
    )
    corpora = output_root / "specialist-corpora"
    ready = split_manifest(
        manifest,
        corpora,
        archetypes=archetypes,
        minimum_decisions=int(args.minimum_decisions),
        expected_source_days=20,
        logical_aliases=dict(roster.get("logical_aliases") or {}),
    )
    balanced_core = materialize_balanced_core(
        manifest,
        output_root,
        archetypes=archetypes,
        source_repo=args.source_repo.resolve(),
    )
    grant_publish_reader(corpora, str(args.publish_reader))
    if balanced_core is not None:
        grant_publish_reader(
            Path(str(balanced_core["root"])),
            str(args.publish_reader),
        )
    split_receipt = read_json(ready)
    final = {
        "schema": RECEIPT_SCHEMA,
        "status": "ready",
        "archive_receipt": str(args.archive_receipt.resolve()),
        "archive_receipt_sha256": sha256(args.archive_receipt.resolve()),
        "daily_selection": str(output_root / "daily/DAILY_READY.json"),
        "daily_manifest": str(manifest),
        "daily_manifest_sha256": sha256(manifest),
        "specialist_corpora": str(corpora),
        "specialist_corpora_receipt": str(ready),
        "specialist_corpora_receipt_sha256": sha256(ready),
        "dates": [row["date"] for row in selected],
        "archetypes": archetypes,
        "results": split_receipt.get("results") or [],
        "publish_reader": str(args.publish_reader),
        **(
            {"balanced_core": balanced_core}
            if balanced_core is not None
            else {}
        ),
    }
    final_path = output_root / "LATEST20_SPECIALIST_CORPORA_READY.json"
    if final_path.is_file() and read_json(final_path) != final:
        raise RuntimeError("existing latest-20 final receipt changed")
    atomic_json(final_path, final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
