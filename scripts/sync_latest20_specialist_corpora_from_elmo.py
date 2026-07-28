#!/usr/bin/env python3
"""Atomically promote Elmo's ready latest-20 specialist corpora to Inzi."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


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
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remote_ready(host: str, path: str) -> bool:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            f"test -s {path}",
        ],
        check=False,
    )
    return result.returncode == 0


def tree_bytes(root: Path) -> int:
    """Return bytes already landed without counting directory metadata."""

    if not root.is_dir():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    )


def remote_tree_bytes(host: str, root: str) -> int:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            f"du -sb {shlex.quote(root)}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.split()[0])


def validate_corpora(
    root: Path,
    roster_path: Path,
    *,
    required_expanded_target_schema: str = "",
    required_expanded_target_digest: str = "",
) -> dict[str, Any]:
    roster = read_json(roster_path)
    expected = list(roster.get("expert_ids") or ())
    if (
        roster.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or int(roster.get("required_specialist_count") or 0) != 18
        or len(expected) != 18
        or len(set(expected)) != 18
    ):
        raise RuntimeError("canonical specialist roster is invalid")
    receipt_path = root / "SPECIALIST_CORPORA_READY.json"
    receipt = read_json(receipt_path)
    if receipt.get("schema") != "poke_bot.specialist_expert_corpora_ready/v1":
        raise RuntimeError("specialist corpus receipt schema is invalid")
    rows = list(receipt.get("results") or ())
    ids = [str(row.get("archetype") or "") for row in rows]
    if len(rows) != 18 or set(ids) != set(expected) or len(ids) != len(set(ids)):
        raise RuntimeError("specialist corpus receipt does not match the roster")
    for row in rows:
        specialist_id = str(row["archetype"])
        status = str(row.get("status") or "")
        directory = root / specialist_id
        if status in {"ready", "insufficient_decisions"}:
            pointer = directory / "PROTECTED_EXPERT_CORPUS.json"
            protected = read_json(pointer)
            manifest = directory / str(protected.get("manifest") or "")
            expected_digest = str(protected.get("manifest_sha256") or "")
            manifest_payload = read_json(manifest) if manifest.is_file() else {}
            if (
                protected.get("schema") != "poke_bot.pinned_expert_corpus/v1"
                or protected.get("protected") is not True
                or not manifest.is_file()
                or sha256(manifest) != expected_digest
                or expected_digest != str(row.get("manifest_sha256") or "")
            ):
                raise RuntimeError(
                    f"protected corpus identity failed: {specialist_id}"
                )
            if required_expanded_target_schema:
                expanded = manifest_payload.get(
                    "expanded_strategic_targets"
                )
                pointer_expanded = protected.get(
                    "expanded_strategic_targets"
                )
                rows = (
                    expanded.get("head_coverage")
                    if isinstance(expanded, dict)
                    else None
                )
                decisions = int(
                    (manifest_payload.get("totals") or {}).get(
                        "decisions_kept"
                    )
                    or 0
                )
                if (
                    not isinstance(expanded, dict)
                    or expanded.get("schema")
                    != required_expanded_target_schema
                    or expanded.get("digest")
                    != required_expanded_target_digest
                    or int(expanded.get("decisions") or -1) != decisions
                    or not isinstance(rows, dict)
                    or set(rows) != set(EXPANDED_HEAD_IDS)
                    or any(
                        int(value.get("labeled_rows") or 0) <= 0
                        or int(value.get("labeled_rows") or 0)
                        + int(value.get("masked_rows") or 0)
                        != decisions
                        or int(value.get("total_rows") or -1)
                        != decisions
                        for value in rows.values()
                    )
                    or pointer_expanded != expanded
                ):
                    raise RuntimeError(
                        "expanded strategic corpus identity failed: "
                        f"{specialist_id}"
                    )
        elif status == "unavailable":
            unavailable = directory / "UNAVAILABLE_EXPERT_CORPUS.json"
            if not unavailable.is_file():
                raise RuntimeError(
                    f"unavailable receipt is absent: {specialist_id}"
                )
        else:
            raise RuntimeError(
                f"unknown specialist corpus status for {specialist_id}: {status}"
            )
    return receipt


def validate_balanced_core(
    root: Path,
    *,
    required_expanded_target_schema: str,
    required_expanded_target_digest: str,
) -> dict[str, Any]:
    pointer = root / "PROTECTED_CORE_CORPUS.json"
    protected = read_json(pointer)
    manifest = root / str(protected.get("manifest") or "")
    payload = read_json(manifest) if manifest.is_file() else {}
    expanded = payload.get("expanded_strategic_targets")
    pointer_expanded = protected.get("expanded_strategic_targets")
    decisions = int(
        (payload.get("totals") or {}).get("decisions_kept") or 0
    )
    coverage = (
        expanded.get("head_coverage")
        if isinstance(expanded, dict)
        else None
    )
    if (
        protected.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or protected.get("protected") is not True
        or not manifest.is_file()
        or sha256(manifest) != protected.get("manifest_sha256")
        or not isinstance(expanded, dict)
        or expanded.get("schema") != required_expanded_target_schema
        or expanded.get("digest") != required_expanded_target_digest
        or int(expanded.get("decisions") or -1) != decisions
        or not isinstance(coverage, dict)
        or set(coverage) != set(EXPANDED_HEAD_IDS)
        or any(
            int(row.get("labeled_rows") or 0) <= 0
            or int(row.get("labeled_rows") or 0)
            + int(row.get("masked_rows") or 0)
            != decisions
            or int(row.get("total_rows") or -1) != decisions
            for row in coverage.values()
        )
        or pointer_expanded != expanded
    ):
        raise RuntimeError("expanded balanced-core corpus identity failed")
    return {
        "pointer": str(pointer),
        "pointer_sha256": sha256(pointer),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "decisions": decisions,
        "expanded_strategic_targets": expanded,
    }


def atomic_symlink(target: Path, link: Path) -> None:
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="admin@192.168.1.143")
    parser.add_argument(
        "--remote-root",
        default=(
            "/mnt/Main/main/poke-bot-agent/archive/expert-latest20-derived/"
            "windows/2026-07-04_2026-07-23/roster18-v5"
        ),
    )
    parser.add_argument(
        "--local-parent",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/data/bootstrap"),
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/data/bootstrap/"
            "current-specialist-latest20"
        ),
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent-deployments/"
            "matchup-v6-safe-boundary-candidate/state/"
            "matchup_adapter_roster.json"
        ),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "expert-latest20-specialist-sync.json"
        ),
    )
    parser.add_argument("--bwlimit-kib", type=int, default=8_000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--version-suffix", default="roster18-v5")
    parser.add_argument("--required-expanded-target-schema", default="")
    parser.add_argument("--required-expanded-target-digest", default="")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(args.version_suffix)):
        raise ValueError("version suffix must be a safe path component")
    if bool(args.required_expanded_target_schema) != bool(
        args.required_expanded_target_digest
    ):
        raise ValueError(
            "expanded target schema and digest must be supplied together"
        )
    if (
        args.required_expanded_target_digest
        and not str(args.required_expanded_target_digest).startswith(
            "sha256:"
        )
    ):
        raise ValueError("expanded target digest must be sha256-prefixed")

    remote_final = f"{args.remote_root}/LATEST20_SPECIALIST_CORPORA_READY.json"
    while not remote_ready(args.host, remote_final):
        print("latest-20 specialist corpus receipt is not ready; waiting", flush=True)
        time.sleep(max(10, int(args.poll_seconds)))

    args.local_parent.mkdir(parents=True, exist_ok=True)
    final_receipt_raw = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host, f"cat {remote_final}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    final_receipt = json.loads(final_receipt_raw)
    dates = list(final_receipt.get("dates") or ())
    if (
        final_receipt.get("schema")
        != "poke_bot.latest20_specialist_corpora/v1"
        or final_receipt.get("status") != "ready"
        or len(dates) != 20
    ):
        raise RuntimeError("remote latest-20 specialist receipt is invalid")
    balanced_core = final_receipt.get("balanced_core")
    require_balanced_core = bool(args.required_expanded_target_schema)
    remote_core_root = f"{args.remote_root}/core-balanced-v6"
    if require_balanced_core and (
        not isinstance(balanced_core, dict)
        or str(balanced_core.get("root") or "") != remote_core_root
        or balanced_core.get("expanded_strategic_targets", {}).get("schema")
        != args.required_expanded_target_schema
        or balanced_core.get("expanded_strategic_targets", {}).get("digest")
        != args.required_expanded_target_digest
    ):
        raise RuntimeError("remote expanded balanced-core receipt is invalid")
    version = (
        f"expert-latest20-{dates[0]}-{dates[-1]}-"
        f"{args.version_suffix}"
    )
    destination = args.local_parent / version
    if destination.is_dir():
        split_receipt = validate_corpora(
            destination,
            args.roster,
            required_expanded_target_schema=(
                args.required_expanded_target_schema
            ),
            required_expanded_target_digest=(
                args.required_expanded_target_digest
            ),
        )
        core_receipt = (
            validate_balanced_core(
                destination / "core-balanced-v6",
                required_expanded_target_schema=(
                    args.required_expanded_target_schema
                ),
                required_expanded_target_digest=(
                    args.required_expanded_target_digest
                ),
            )
            if require_balanced_core
            else None
        )
        atomic_symlink(destination, args.current)
        landed_bytes = tree_bytes(destination)
        atomic_json(
            args.state,
            {
                "schema": "poke_bot.latest20_specialist_sync/v1",
                "status": "ready",
                "source_host": args.host,
                "source_root": args.remote_root,
                "destination": str(destination),
                "current_pointer": str(args.current),
                "dates": dates,
                "specialist_count": len(split_receipt["results"]),
                "source_bytes": landed_bytes,
                "copied_bytes": landed_bytes,
                "percent": 100.0,
                "bandwidth_limit_kib_per_second": int(args.bwlimit_kib),
                "version_suffix": str(args.version_suffix),
                "expanded_target_schema": (
                    args.required_expanded_target_schema or None
                ),
                "expanded_target_digest": (
                    args.required_expanded_target_digest or None
                ),
                "balanced_core": core_receipt,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0

    staging = args.local_parent / f".{version}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    source_bytes = remote_tree_bytes(
        args.host,
        f"{args.remote_root}/specialist-corpora",
    )
    if require_balanced_core:
        source_bytes += remote_tree_bytes(args.host, remote_core_root)
    started_at = datetime.now(timezone.utc).isoformat()
    state_base = {
        "schema": "poke_bot.latest20_specialist_sync/v1",
        "source_host": args.host,
        "source_root": args.remote_root,
        "staging": str(staging),
        "destination": str(destination),
        "current_pointer": str(args.current),
        "dates": dates,
        "specialist_count": len(final_receipt.get("archetypes") or ()),
        "source_bytes": source_bytes,
        "bandwidth_limit_kib_per_second": int(args.bwlimit_kib),
        "version_suffix": str(args.version_suffix),
        "expanded_target_schema": (
            args.required_expanded_target_schema or None
        ),
        "expanded_target_digest": (
            args.required_expanded_target_digest or None
        ),
        "started_at_utc": started_at,
    }
    try:
        process = subprocess.Popen(
            [
                "rsync",
                "-a",
                "--partial",
                "--append-verify",
                f"--bwlimit={int(args.bwlimit_kib)}",
                f"{args.host}:{args.remote_root}/specialist-corpora/",
                f"{staging}/",
            ]
        )
        while process.poll() is None:
            copied_bytes = tree_bytes(staging)
            atomic_json(
                args.state,
                {
                    **state_base,
                    "status": "syncing",
                    "copied_bytes": copied_bytes,
                    "percent": (
                        100.0 * copied_bytes / source_bytes
                        if source_bytes > 0
                        else 0.0
                    ),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            time.sleep(5.0)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
        if require_balanced_core:
            process = subprocess.Popen(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "--append-verify",
                    f"--bwlimit={int(args.bwlimit_kib)}",
                    f"{args.host}:{remote_core_root}/",
                    f"{staging}/core-balanced-v6/",
                ]
            )
            while process.poll() is None:
                copied_bytes = tree_bytes(staging)
                atomic_json(
                    args.state,
                    {
                        **state_base,
                        "status": "syncing_balanced_core",
                        "copied_bytes": copied_bytes,
                        "percent": (
                            100.0 * copied_bytes / source_bytes
                            if source_bytes > 0
                            else 0.0
                        ),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                time.sleep(5.0)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, process.args
                )
        (staging / "LATEST20_SPECIALIST_CORPORA_READY.json").write_text(
            json.dumps(final_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_receipt = validate_corpora(
            staging,
            args.roster,
            required_expanded_target_schema=(
                args.required_expanded_target_schema
            ),
            required_expanded_target_digest=(
                args.required_expanded_target_digest
            ),
        )
        core_receipt = (
            validate_balanced_core(
                staging / "core-balanced-v6",
                required_expanded_target_schema=(
                    args.required_expanded_target_schema
                ),
                required_expanded_target_digest=(
                    args.required_expanded_target_digest
                ),
            )
            if require_balanced_core
            else None
        )
        os.replace(staging, destination)
        atomic_symlink(destination, args.current)
        atomic_json(
            args.state,
            {
                "schema": "poke_bot.latest20_specialist_sync/v1",
                "status": "ready",
                "source_host": args.host,
                "source_root": args.remote_root,
                "destination": str(destination),
                "current_pointer": str(args.current),
                "dates": dates,
                "specialist_count": len(split_receipt["results"]),
                "source_bytes": source_bytes,
                "copied_bytes": source_bytes,
                "percent": 100.0,
                "bandwidth_limit_kib_per_second": int(args.bwlimit_kib),
                "version_suffix": str(args.version_suffix),
                "expanded_target_schema": (
                    args.required_expanded_target_schema or None
                ),
                "expanded_target_digest": (
                    args.required_expanded_target_digest or None
                ),
                "balanced_core": core_receipt,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except BaseException as exc:
        copied_bytes = tree_bytes(staging)
        atomic_json(
            args.state,
            {
                **state_base,
                "status": "incomplete",
                "copied_bytes": copied_bytes,
                "percent": (
                    100.0 * copied_bytes / source_bytes
                    if source_bytes > 0
                    else 0.0
                ),
                "failure_reason": repr(exc),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
