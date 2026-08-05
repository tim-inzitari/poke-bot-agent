#!/usr/bin/env python3
"""Materialize revision-138 top-100 expert importance with immutable joins."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any
from urllib import parse, request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.expert_pilot_importance import (  # noqa: E402
    PILOT_MAP_SCHEMA,
    SNAPSHOT_SCHEMA,
    TARGET_SCHEMA,
    canonical_digest,
    file_digest,
    materialize_importance_index,
)
def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: object) -> None:
    destination = Path(path).expanduser().resolve()
    body = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable output already differs: {destination}")
        return
    temporary = destination.with_name(
        f".{destination.name}.partial.{os.getpid()}"
    )
    temporary.write_text(body, encoding="utf-8")
    os.link(temporary, destination)
    temporary.unlink(missing_ok=True)


def _resolve_manifest(pointer_path: Path) -> tuple[Path, str, dict[str, Any]]:
    pointer = _read(pointer_path)
    if not isinstance(pointer, dict):
        raise ValueError("protected expert pointer must be an object")
    if pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1":
        raise ValueError("expert pointer is not protected/pinned")
    manifest = (
        pointer_path.expanduser().resolve().parent
        / str(pointer.get("manifest") or "")
    ).resolve()
    expected = str(pointer.get("manifest_sha256") or "")
    if not manifest.is_file() or file_digest(manifest) != expected:
        raise ValueError("protected expert manifest checksum changed")
    payload = _read(manifest)
    if not isinstance(payload, dict):
        raise ValueError("expert manifest must be an object")
    return manifest, expected, payload


def command_targets(args: argparse.Namespace) -> int:
    from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
    from poke_bot.pure_rl.expert_feature_stream import (
        EpisodeGroupedFeatureManifest,
    )

    pointer = args.expert_pointer.expanduser().resolve()
    manifest, digest, manifest_payload = _resolve_manifest(pointer)
    plan = EpisodeGroupedFeatureManifest.open(
        manifest,
        expected_manifest_digest=digest,
        val_frac=float(args.validation_fraction),
        seed=int(args.split_seed),
        max_context=int(args.max_context),
        expected_compact_mode=COMPACT_MODE_TEMPORAL_EXPERT,
        workers=max(1, int(args.workers)),
    )
    train_rows, validation_rows = plan.partition_identity_rows()
    archives: dict[str, str] = {}
    for shard in list(manifest_payload.get("shards") or ()):
        name = str(shard.get("source_archive") or "")
        digest_value = str(shard.get("source_archive_sha256") or "")
        if not name or not digest_value.startswith("sha256:"):
            # Older protected specialist manifests preserved the source archive
            # identity in the immutable feature-shard header rather than
            # duplicating it into the manifest row.  Reading that trusted,
            # checksum-verified local shard keeps the same provenance intact.
            shard_path = (
                manifest.parent / str(shard.get("path") or "")
            ).resolve()
            if not shard_path.is_file() or file_digest(shard_path) != str(
                shard.get("sha256") or ""
            ):
                raise ValueError("expert feature shard checksum changed")
            with shard_path.open("rb") as stream:
                header = pickle.load(stream)
            if not isinstance(header, dict):
                raise ValueError("expert feature shard header is invalid")
            name = Path(str(header.get("source_archive") or "")).name
            digest_value = str(header.get("source_archive_sha256") or "")
        if not name or not digest_value.startswith("sha256:"):
            raise ValueError("expert shard lacks source archive identity")
        if name in archives and archives[name] != digest_value:
            raise ValueError("source archive has conflicting checksums")
        archives[name] = digest_value
    output = {
        "schema": TARGET_SCHEMA,
        "owner_decision_revision": 138,
        "expert_pointer": str(pointer),
        "corpus_manifest": str(manifest),
        "corpus_manifest_sha256": digest,
        "split_seed": int(args.split_seed),
        "validation_fraction": float(args.validation_fraction),
        "max_context": int(args.max_context),
        "support_partition": "training_only",
        "train_rows": list(train_rows),
        "validation_rows": list(validation_rows),
        "source_archives": [
            {"name": name, "sha256": archives[name]}
            for name in sorted(archives)
        ],
        "train_identity_sha256": canonical_digest(list(train_rows)),
        "validation_identity_sha256": canonical_digest(list(validation_rows)),
    }
    _write_once(args.output, output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": file_digest(args.output.resolve()),
        "train_games": len(train_rows),
        "validation_games": len(validation_rows),
        "source_archives": len(archives),
    }, sort_keys=True))
    return 0


def command_snapshot(args: argparse.Namespace) -> int:
    raw = args.site_config.expanduser().resolve().read_text(encoding="utf-8")
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("site config does not contain a JSON object")
    config = json.loads(raw[start : end + 1])
    base = str(config.get("supabaseUrl") or "").rstrip("/")
    anon_key = str(config.get("anonKey") or "")
    email = str(config.get("teamEmail") or "")
    password = str(config.get("teamPassword") or "")
    if not all((base.startswith("https://"), anon_key, email, password)):
        raise ValueError("site config lacks authenticated Supabase inputs")

    auth_body = json.dumps({"email": email, "password": password}).encode()
    auth_request = request.Request(
        base + "/auth/v1/token?grant_type=password",
        data=auth_body,
        method="POST",
        headers={"apikey": anon_key, "Content-Type": "application/json"},
    )
    with request.urlopen(auth_request, timeout=30) as response:
        token = str(json.load(response).get("access_token") or "")
    if not token:
        raise ValueError("site authentication returned no access token")
    query = parse.urlencode({"select": "*", "order": "rank"})
    leaderboard_request = request.Request(
        base + "/rest/v1/leaderboard?" + query,
        headers={"apikey": anon_key, "Authorization": f"Bearer {token}"},
    )
    with request.urlopen(leaderboard_request, timeout=30) as response:
        source_rows = json.load(response)
    if not isinstance(source_rows, list):
        raise ValueError("leaderboard response is not a list")
    public_fields = (
        "rank",
        "team_name",
        "score",
        "submission_date",
        "archetype_id",
        "archetype_name",
        "fetched_at",
        "current_submission_id",
    )
    rows = [
        {field: row.get(field) for field in public_fields}
        for row in source_rows
        if isinstance(row, dict)
    ]
    if len(rows) != 100 or [int(row["rank"]) for row in rows] != list(
        range(1, 101)
    ):
        raise ValueError("live leaderboard is not an exact ordered top 100")
    output = {
        "schema": SNAPSHOT_SCHEMA,
        "owner_decision_revision": 138,
        "source": "https://ptcgreplay.netlify.app/",
        "source_table": "leaderboard",
        "row_count": len(rows),
        "rows": rows,
    }
    _write_once(args.output, output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": file_digest(args.output.resolve()),
        "rows": len(rows),
        "fetched_at": rows[0].get("fetched_at"),
    }, sort_keys=True))
    return 0


def command_extract(args: argparse.Namespace) -> int:
    targets_path = args.targets.expanduser().resolve()
    targets = _read(targets_path)
    if not isinstance(targets, dict) or targets.get("schema") != TARGET_SCHEMA:
        raise ValueError("invalid expert importance targets")
    all_rows = list(targets.get("train_rows") or ()) + list(
        targets.get("validation_rows") or ())
    requested: dict[str, set[int]] = {}
    for row in all_rows:
        episode_id = str(row.get("episode_id") or "")
        seat = int(row.get("seat", -1))
        if not episode_id or seat not in (0, 1):
            raise ValueError("target row has invalid episode/seat")
        requested.setdefault(episode_id, set()).add(seat)

    tasks: list[tuple[str, str, bool, dict[str, tuple[int, ...]]]] = []
    for archive_row in list(targets.get("source_archives") or ()):
        name = str(archive_row.get("name") or "")
        archive = (args.archive_dir.expanduser().resolve() / name).resolve()
        expected = str(archive_row.get("sha256") or "")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        tasks.append(
            (
                str(archive),
                expected,
                bool(args.verify_archive_digests),
                {key: tuple(sorted(value)) for key, value in requested.items()},
            )
        )
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            results = list(pool.map(_extract_archive, tasks))
    else:
        results = [_extract_archive(task) for task in tasks]
    found: dict[tuple[str, int], str] = {}
    archive_receipts: list[dict[str, Any]] = []
    for result_rows, receipt in results:
        archive_receipts.append(receipt)
        for episode_id, seat, team_name in result_rows:
            key = (episode_id, seat)
            if key in found and found[key] != team_name:
                raise ValueError(
                    "raw archives disagree on exact pilot identity: "
                    f"{episode_id} seat={seat}"
                )
            found[key] = team_name

    rows = [
        {"episode_id": episode_id, "seat": seat, "team_name": found[(episode_id, seat)]}
        for episode_id, seat in sorted(found)
    ]
    requested_keys = {
        (str(row["episode_id"]), int(row["seat"])) for row in all_rows
    }
    missing = [
        {"episode_id": episode_id, "seat": seat}
        for episode_id, seat in sorted(requested_keys - set(found))
    ]
    output = {
        "schema": PILOT_MAP_SCHEMA,
        "owner_decision_revision": 137,
        "targets_sha256": file_digest(targets_path),
        "rows": rows,
        "requested_rows": len(requested_keys),
        "resolved_rows": len(rows),
        "unverifiable_rows": len(missing),
        "unverifiable": missing,
        "source_archives": archive_receipts,
    }
    _write_once(args.output, output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": file_digest(args.output.resolve()),
        "resolved_rows": len(rows),
        "unverifiable_rows": len(missing),
    }, sort_keys=True))
    return 0


def _extract_archive(
    task: tuple[str, str, bool, dict[str, tuple[int, ...]]],
) -> tuple[list[tuple[str, int, str]], dict[str, Any]]:
    archive_raw, expected, verify_digest, requested = task
    archive = Path(archive_raw)
    actual = file_digest(archive) if verify_digest else None
    if actual is not None and actual != expected:
        raise ValueError(f"source archive checksum changed: {archive}")
    found: list[tuple[str, int, str]] = []
    opened = 0
    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, str] = {}
        for member in bundle.namelist():
            if member.endswith("/") or not member.casefold().endswith(".json"):
                continue
            episode_id = Path(member).stem
            if episode_id in requested:
                if episode_id in members:
                    raise ValueError(
                        f"archive has duplicate target episode: {episode_id}"
                    )
                members[episode_id] = member
        for episode_id, member in members.items():
            payload = json.loads(bundle.read(member))
            team_names = (payload.get("info") or {}).get("TeamNames")
            if not isinstance(team_names, list):
                continue
            for seat in requested[episode_id]:
                if seat >= len(team_names):
                    continue
                team_name = team_names[seat]
                if isinstance(team_name, str) and team_name:
                    found.append((episode_id, seat, team_name))
            opened += 1
    return found, {
        "path": str(archive),
        "expected_sha256": expected,
        "verified_sha256": actual,
        "target_episodes_opened": opened,
    }


def command_finalize(args: argparse.Namespace) -> int:
    targets_path = args.targets.expanduser().resolve()
    pilot_map_path = args.pilot_map.expanduser().resolve()
    snapshot_path = args.leaderboard_snapshot.expanduser().resolve()
    targets = _read(targets_path)
    pilot_map = _read(pilot_map_path)
    snapshot = _read(snapshot_path)
    if not all(isinstance(value, dict) for value in (targets, pilot_map, snapshot)):
        raise ValueError("importance inputs must be JSON objects")
    output = materialize_importance_index(
        targets=targets,
        pilot_map=pilot_map,
        leaderboard_snapshot=snapshot,
        targets_digest=file_digest(targets_path),
        pilot_map_digest=file_digest(pilot_map_path),
        leaderboard_snapshot_digest=file_digest(snapshot_path),
    )
    _write_once(args.output, output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": file_digest(args.output.resolve()),
        "train_games": output["train_games"],
        "matched_top_100_train_games": output["matched_top_100_train_games"],
        "effective_training_weight_mass": output["effective_training_weight_mass"],
        "tier_counts": output["tier_counts"],
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser("targets")
    targets.add_argument("--expert-pointer", type=Path, required=True)
    targets.add_argument("--output", type=Path, required=True)
    targets.add_argument("--split-seed", type=int, default=20260801)
    targets.add_argument("--validation-fraction", type=float, default=0.10)
    targets.add_argument("--max-context", type=int, default=320)
    targets.add_argument("--workers", type=int, default=4)
    targets.set_defaults(func=command_targets)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--site-config", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.set_defaults(func=command_snapshot)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--targets", type=Path, required=True)
    extract.add_argument("--archive-dir", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--workers", type=int, default=1)
    extract.add_argument(
        "--verify-archive-digests", action=argparse.BooleanOptionalAction,
        default=True,
    )
    extract.set_defaults(func=command_extract)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--targets", type=Path, required=True)
    finalize.add_argument("--pilot-map", type=Path, required=True)
    finalize.add_argument("--leaderboard-snapshot", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(func=command_finalize)

    args = parser.parse_args()
    if getattr(args, "workers", 1) <= 0:
        raise ValueError("workers must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
