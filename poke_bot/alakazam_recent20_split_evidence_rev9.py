"""Compact split evidence derived from sealed revision-9 rollout shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


GOAL_REVISION = 9
GOAL_GATEWAY_SHA256 = "sha256:8908c4e8bcf36a089ba7f230c137e259f024125807bdb04b03d77483f533c223"
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
CORPUS_MANIFEST_SHA256 = "sha256:9261bc6c52f55810db59c313631ec51966f71e49abcbdd43f6b3e1fd198965a1"
PARITY_RECEIPT_SHA256 = "sha256:92e5ad858598c670ca4ee7459f11009e61bd48d4e385ebed64c95bb6a7e69732"
CENSUS_RECEIPT_SHA256 = "sha256:34c51a59f4e843ff9d04ec07a807afe88ff4149210f6471833e422b41975fb9c"
BRANCH_RECEIPT_SHA256 = "sha256:084d068bebfa2a0da15209bda798842c38e59a52637bb49723fad063c487a52e"
SPLIT_SCHEMA = "poke_bot.alakazam_recent20_source_day_group_disjoint_split_manifest/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_recent20_split_evidence_receipt/v1"

TRAIN_DAYS = tuple(f"2026-07-{day:02d}" for day in range(23, 32)) + tuple(
    f"2026-08-{day:02d}" for day in range(1, 6)
)
VALIDATION_DAYS = ("2026-08-06", "2026-08-07", "2026-08-08")
EVALUATION_DAYS = ("2026-08-09", "2026-08-10", "2026-08-11")


class SplitEvidenceError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _text_field(fragment: bytes, marker: bytes) -> str:
    start = fragment.find(marker)
    if start < 0:
        raise SplitEvidenceError(f"row lacks {marker!r}")
    start += len(marker)
    end = fragment.find(b'"', start)
    if end < 0:
        raise SplitEvidenceError(f"row has unterminated {marker!r}")
    return fragment[start:end].decode("ascii")


def _int_field(fragment: bytes, marker: bytes) -> int:
    start = fragment.find(marker)
    if start < 0:
        raise SplitEvidenceError(f"row lacks {marker!r}")
    start += len(marker)
    end = start
    while end < len(fragment) and fragment[end : end + 1] in b"-0123456789":
        end += 1
    try:
        return int(fragment[start:end])
    except ValueError as exc:
        raise SplitEvidenceError(f"row has malformed {marker!r}") from exc


def _digest_parts(parts: Iterable[str]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(list(parts))).hexdigest()


def _scan_shard(task: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(task["path"]))
    if path.is_symlink() or not path.is_file():
        raise SplitEvidenceError(f"missing regular shard: {path}")
    expected_day = str(task["utc_day"])
    expected_sha = str(task["sha256"])
    expected_size = int(task["size_bytes"])
    digest = hashlib.sha256()
    rows = 0
    games: set[str] = set()
    seat_games: set[str] = set()
    decisions: set[str] = set()
    deck_variants: set[str] = set()
    source_archives: set[str] = set()
    with path.open("rb", buffering=16 * 1024 * 1024) as stream:
        header = stream.readline()
        digest.update(header)
        header_value = json.loads(header)
        if header_value.get("schema") != "poke_bot.alakazam_recent20_intraday_refeature_shard/v1":
            raise SplitEvidenceError(f"foreign shard header: {path}")
        if header_value.get("utc_day") != expected_day:
            raise SplitEvidenceError(f"shard day mismatch: {path}")
        for line in stream:
            digest.update(line)
            rows += 1
            source_start = line.rfind(b'"source":{')
            source_end = line.find(b',"transition":', source_start)
            if source_start < 0 or source_end < 0:
                raise SplitEvidenceError(f"row lacks source boundary: {path}:{rows + 1}")
            source = line[source_start:source_end]
            if (
                b'"acting_seat_setup_deck_contains_card_743":true' not in source
                or b'"row_materialization_eligible":true' not in source
            ):
                raise SplitEvidenceError(f"ineligible row in sealed shard: {path}:{rows + 1}")
            episode = _text_field(source, b'"episode_id":"')
            seat = _int_field(source, b'"acting_seat":')
            day = _text_field(source, b'"source_archive_date":"')
            archive_sha = _text_field(source, b'"source_archive_sha256":"')
            deck_sha = _text_field(source, b'"acting_deck_multiset_sha256":"')
            env_step = _int_field(source, b'"env_step":')
            factorized_stage = _int_field(source, b'"factorized_stage":')
            if day != expected_day or seat not in (0, 1):
                raise SplitEvidenceError(f"row source identity mismatch: {path}:{rows + 1}")
            observation = _text_field(line, b'"canonical_public_observation_hash":"')
            option_set = _text_field(line, b'"normalized_canonical_option_multiset_hash":"')
            games.add(episode)
            seat_games.add(_digest_parts((archive_sha, episode, str(seat))))
            decisions.add(
                _digest_parts(
                    (archive_sha, episode, str(seat), str(env_step), str(factorized_stage), observation, option_set)
                )
            )
            deck_variants.add(deck_sha)
            source_archives.add(archive_sha)
    observed_sha = "sha256:" + digest.hexdigest()
    if path.stat().st_size != expected_size or observed_sha != expected_sha:
        raise SplitEvidenceError(f"shard SHA/size mismatch: {path}")
    if rows != int(task["record_count"]):
        raise SplitEvidenceError(f"shard row count mismatch: {path}")
    return {
        "utc_day": expected_day,
        "path": str(path),
        "sha256": observed_sha,
        "size_bytes": expected_size,
        "row_count": rows,
        "game_ids": sorted(games),
        "seat_game_group_hashes": sorted(seat_games),
        "decision_count": len(decisions),
        "decision_inventory_sha256": _digest_parts(sorted(decisions)),
        "deck_variant_sha256s": sorted(deck_variants),
        "source_archive_sha256s": sorted(source_archives),
    }


def _load_exact(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise SplitEvidenceError(f"artifact identity mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SplitEvidenceError(f"artifact is not an object: {path}")
    return value


def _resolve_tasks(manifest: Mapping[str, Any], parity: Mapping[str, Any]) -> list[dict[str, Any]]:
    destination_paths = parity.get("destination_object_paths_by_sha256", {})
    tasks = []
    for row in manifest["shards"]:
        if row["source_host"] == "elmo":
            path = destination_paths[row["sha256"]]
        elif row["source_host"] == "inzi":
            path = str(
                Path(row["day_receipt_path"]).parent
                / "refeatured-records"
                / "shards"
                / row["filename"]
            )
        else:
            raise SplitEvidenceError("foreign source host in manifest")
        tasks.append({**row, "path": path})
    return sorted(tasks, key=lambda row: row["utc_day"])


def build_split_evidence(
    *,
    corpus_manifest_path: Path,
    parity_receipt_path: Path,
    census_receipt_path: Path,
    branch_receipt_path: Path,
    workers: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_exact(corpus_manifest_path, CORPUS_MANIFEST_SHA256)
    parity = _load_exact(parity_receipt_path, PARITY_RECEIPT_SHA256)
    census = _load_exact(census_receipt_path, CENSUS_RECEIPT_SHA256)
    branch = _load_exact(branch_receipt_path, BRANCH_RECEIPT_SHA256)
    if workers != 20 or manifest.get("utc_partition_count") != 20:
        raise SplitEvidenceError("revision-9 split scan requires exactly 20 workers/days")
    tasks = _resolve_tasks(manifest, parity)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        days = list(pool.map(_scan_shard, tasks))

    assignment = {
        "train": TRAIN_DAYS,
        "validation": VALIDATION_DAYS,
        "evaluation": EVALUATION_DAYS,
    }
    if set().union(*map(set, assignment.values())) != {row["utc_day"] for row in days}:
        raise SplitEvidenceError("split day inventory mismatch")
    if any(set(a) & set(b) for i, a in enumerate(assignment.values()) for b in list(assignment.values())[i + 1 :]):
        raise SplitEvidenceError("split day overlap")
    all_games: set[str] = set()
    all_groups: set[str] = set()
    all_archives: set[str] = set()
    splits: dict[str, Any] = {}
    for split, split_days in assignment.items():
        rows = [row for row in days if row["utc_day"] in split_days]
        games = {game for row in rows for game in row["game_ids"]}
        groups = {group for row in rows for group in row["seat_game_group_hashes"]}
        archives = {sha for row in rows for sha in row["source_archive_sha256s"]}
        if all_games & games or all_groups & groups or all_archives & archives:
            raise SplitEvidenceError("cross-split source/game/group overlap")
        all_games |= games
        all_groups |= groups
        all_archives |= archives
        splits[split] = {
            "utc_days": list(split_days),
            "utc_day_count": len(split_days),
            "source_archive_sha256s": sorted(archives),
            "source_archive_count": len(archives),
            "row_count": sum(row["row_count"] for row in rows),
            "decision_count": sum(row["decision_count"] for row in rows),
            "game_count": len(games),
            "qualifying_acting_seat_game_group_count": len(groups),
            "game_id_inventory_sha256": _digest_parts(sorted(games)),
            "group_inventory_sha256": _digest_parts(sorted(groups)),
            "shard_sha256s": [row["sha256"] for row in rows],
        }
    day_summaries = [
        {key: value for key, value in row.items() if key not in {"game_ids", "seat_game_group_hashes"}}
        for row in days
    ]
    split_manifest = {
        "schema": SPLIT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "window_start_utc": "2026-07-23",
        "window_end_utc": "2026-08-11",
        "utc_partition_count": 20,
        "split_rule": "whole_utc_day_and_source_archive_with_episode_acting_seat_group_fence",
        "splits": splits,
        "day_summaries": day_summaries,
        "cross_split_utc_day_overlap_count": 0,
        "cross_split_source_archive_overlap_count": 0,
        "cross_split_game_overlap_count": 0,
        "cross_split_group_overlap_count": 0,
        "total_row_count": sum(row["row_count"] for row in days),
        "total_decision_count": sum(row["decision_count"] for row in days),
        "total_game_count": len(all_games),
        "total_qualifying_acting_seat_game_group_count": len(all_groups),
        "all_20_shards_sha_size_and_row_count_revalidated": True,
    }
    if split_manifest["total_row_count"] != manifest["record_count"]:
        raise SplitEvidenceError("aggregate row count mismatch")
    if split_manifest["total_decision_count"] != census["decision_count"]:
        raise SplitEvidenceError("aggregate decision count mismatch")
    if branch.get("eligible_trainable_branches") != ["public_rule_semantic_projection"]:
        raise SplitEvidenceError("branch inventory drift")
    split_sha = "sha256:" + hashlib.sha256(canonical_bytes(split_manifest)).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_GATEWAY_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "source_remote_parity_receipt_sha256": PARITY_RECEIPT_SHA256,
        "collision_census_receipt_sha256": CENSUS_RECEIPT_SHA256,
        "branch_adjudication_receipt_sha256": BRANCH_RECEIPT_SHA256,
        "source_day_group_disjoint_split_manifest_sha256": split_sha,
        "finalized_shard_count": manifest["finalized_shard_count"],
        "finalized_shard_bytes": manifest["finalized_shard_bytes"],
        "row_count": split_manifest["total_row_count"],
        "decision_count": split_manifest["total_decision_count"],
        "game_count": split_manifest["total_game_count"],
        "qualifying_acting_seat_game_group_count": split_manifest[
            "total_qualifying_acting_seat_game_group_count"
        ],
        "all_shards_reopened_and_sha_size_row_count_revalidated": True,
        "source_day_group_disjointness_passed": True,
        "eligible_trainable_branches": ["public_rule_semantic_projection"],
        "unsupported_branches_remain_exact_zero_and_inert": True,
        "inzi_corpus_handoff_receipt_issued": False,
        "training_eligible_before_activation": False,
        "service_control_performed": False,
        "training_or_activation_performed": False,
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return split_manifest, receipt


def write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise SplitEvidenceError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        body = canonical_bytes(value)
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--census-receipt", type=Path, required=True)
    parser.add_argument("--branch-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SplitEvidenceError("output root exists")
    split, receipt = build_split_evidence(
        corpus_manifest_path=args.corpus_manifest,
        parity_receipt_path=args.parity_receipt,
        census_receipt_path=args.census_receipt,
        branch_receipt_path=args.branch_receipt,
        workers=args.workers,
    )
    split_sha = write_create_only(args.output_root / "source-day-group-split-manifest.json", split)
    if split_sha != receipt["source_day_group_disjoint_split_manifest_sha256"]:
        raise SplitEvidenceError("split digest drift")
    receipt_sha = write_create_only(args.output_root / "split-evidence-receipt.json", receipt)
    print(json.dumps({"split_manifest_sha256": split_sha, "receipt_sha256": receipt_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
