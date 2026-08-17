"""Streaming revision-9 collision census over the sealed recent-20 shards.

The input shards already contain the collision-census option records produced
while re-featurizing.  This pass therefore never reopens the raw ZIP archive.
Each shard is reduced to one sorted fixed-width private spool; the parent then
performs one deterministic merge and publishes compact report/receipt files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import heapq
import json
import os
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


GOAL_REVISION = 9
CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
MANIFEST_SCHEMA = "poke_bot.alakazam_recent20_refeaturized_corpus_manifest/v1"
RECORD_SCHEMA = "poke_bot.alakazam_collision_census_r298_option_record/v1"
REPORT_SCHEMA = "poke_bot.alakazam_recent20_collision_census_report/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_recent20_collision_census_receipt/v1"
BRANCH_SCHEMA = "poke_bot.alakazam_recent20_branch_adjudication_receipt/v1"
PERMUTATION_RECEIPT_SCHEMA = "poke_bot.alakazam_recent20_permutation_equivalence_receipt/v1"
TARGETED_WITNESS_RECEIPT_SCHEMA = "poke_bot.alakazam_recent20_targeted_transition_witness_receipt/v1"
_SHA_ZERO = b"\0" * 32
_ROW = struct.Struct(">32s32s32s32sBB32s32s")
_LEGACY_ALIAS_KEYS: frozenset[tuple[str, str]] = frozenset()


class Recent20CensusError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _digest_bytes(value: object, *, optional: bool = False) -> bytes:
    if value is None and optional:
        return _SHA_ZERO
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Recent20CensusError("malformed SHA-256")
    try:
        result = bytes.fromhex(value[7:])
    except ValueError as exc:
        raise Recent20CensusError("malformed SHA-256") from exc
    if len(result) != 32:
        raise Recent20CensusError("malformed SHA-256")
    return result


def _sha_text(value: bytes) -> str:
    return "sha256:" + value.hex()


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _scan_one(task: tuple[str, str, str]) -> dict[str, Any]:
    day, source_path_text, spool_path_text = task
    source_path = Path(source_path_text)
    spool_path = Path(spool_path_text)
    rows: list[bytes] = []
    option_records = selected_records = incomplete_records = 0
    deck_records: Counter[str] = Counter()
    deck_decisions: Counter[str] = Counter()
    number_values: Counter[str] = Counter()
    semantic_field_presence: Counter[str] = Counter()
    with source_path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as stream:
        header = json.loads(next(stream))
        if (
            header.get("schema") != "poke_bot.alakazam_recent20_intraday_refeature_shard/v1"
            or header.get("goal_revision") != GOAL_REVISION
            or header.get("goal_contract_sha256") != CONTRACT_SHA256
            or header.get("utc_day") != day
        ):
            raise Recent20CensusError(f"foreign shard header for {day}")
        for line in stream:
            record = json.loads(line)
            if record.get("schema") != RECORD_SCHEMA:
                raise Recent20CensusError(f"foreign option record in {day}")
            source = record.get("source")
            if not isinstance(source, Mapping) or source.get("acting_seat_setup_deck_contains_card_743") is not True:
                raise Recent20CensusError(f"non-card-743 row in {day}")
            deck = source.get("acting_deck_multiset_sha256")
            if not isinstance(deck, str):
                raise Recent20CensusError(f"missing acting-deck digest in {day}")
            transition = record.get("transition")
            if not isinstance(transition, Mapping):
                raise Recent20CensusError(f"malformed transition in {day}")
            selected = record.get("selected_candidate") is True
            complete = transition.get("evidence_status") == "complete"
            row = _ROW.pack(
                _digest_bytes(record.get("canonical_public_observation_hash")),
                _digest_bytes(record.get("legacy_current_feature_token_hash")),
                _digest_bytes(record.get("new_complete_semantic_option_key_sha256")),
                _digest_bytes(record.get("candidate_action_sha256")),
                int(selected),
                int(complete),
                _digest_bytes(transition.get("simulator_successor_event_chain_hash"), optional=True),
                _digest_bytes(transition.get("simulator_outcome_distribution_hash"), optional=True),
            )
            rows.append(row)
            option_records += 1
            selected_records += int(selected)
            incomplete_records += int(not complete)
            deck_records[deck] += 1
            if selected:
                deck_decisions[deck] += 1
            semantic = record.get("complete_semantic_option_key")
            for key, value in _walk(semantic):
                if value is not None and value != []:
                    semantic_field_presence[key] += 1
                if key == "number" and isinstance(value, int) and not isinstance(value, bool):
                    number_values[str(value)] += 1
    rows.sort()
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(spool_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb", buffering=16 * 1024 * 1024, closefd=False) as output:
            for row in rows:
                output.write(row)
            output.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "utc_day": day,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "spool_path": str(spool_path),
        "spool_sha256": sha256_file(spool_path),
        "spool_size_bytes": spool_path.stat().st_size,
        "option_record_count": option_records,
        "selected_record_count": selected_records,
        "incomplete_transition_record_count": incomplete_records,
        "acting_deck_record_counts": dict(sorted(deck_records.items())),
        "acting_deck_decision_counts": dict(sorted(deck_decisions.items())),
        "number_value_counts": dict(sorted(number_values.items(), key=lambda item: int(item[0]))),
        "semantic_field_presence_counts": dict(sorted(semantic_field_presence.items())),
    }


def _spool_rows(path: Path) -> Iterable[bytes]:
    with path.open("rb", buffering=16 * 1024 * 1024) as stream:
        while row := stream.read(_ROW.size):
            if len(row) != _ROW.size:
                raise Recent20CensusError(f"truncated spool: {path}")
            yield row


def _analyse_sorted(spools: Sequence[Path]) -> dict[str, Any]:
    option_count = selected_count = incomplete_count = 0
    repeated_group_count = legacy_collision_count = collision_record_count = 0
    selected_collision_count = divergent_successor_count = divergent_outcome_count = 0
    unresolved_new_collision_count = 0
    examples: list[dict[str, Any]] = []
    merged = heapq.merge(*(_spool_rows(path) for path in spools))
    current_key: bytes | None = None
    semantic: set[bytes] = set()
    actions: set[bytes] = set()
    successors: set[bytes] = set()
    outcomes: set[bytes] = set()
    group_records = group_selected = group_incomplete = 0

    def finish() -> None:
        nonlocal repeated_group_count, legacy_collision_count, collision_record_count
        nonlocal selected_collision_count, divergent_successor_count, divergent_outcome_count
        nonlocal unresolved_new_collision_count
        if current_key is None:
            return
        if group_records > 1:
            repeated_group_count += 1
        # The repaired option identity is the public observation plus the
        # complete semantic key.  Multiple raw action payloads under one such
        # key require an explicit permutation-equivalence adjudication; do not
        # silently call them resolved.
        if len(semantic) == 1 and len(actions) > 1:
            unresolved_new_collision_count += 1
        if len(semantic) > 1:
            legacy_collision_count += 1
            collision_record_count += group_records
            selected_collision_count += group_selected
            divergent_successor_count += int(len(successors) > 1)
            divergent_outcome_count += int(len(outcomes) > 1)
            if len(examples) < 100:
                examples.append({
                    "canonical_public_observation_hash": _sha_text(current_key[:32]),
                    "legacy_current_feature_token_hash": _sha_text(current_key[32:]),
                    "record_count": group_records,
                    "distinct_repaired_semantic_key_count": len(semantic),
                    "selected_candidate_record_count": group_selected,
                    "incomplete_transition_record_count": group_incomplete,
                    "known_successor_count": len(successors),
                    "known_outcome_distribution_count": len(outcomes),
                    "repaired_key_separates_legacy_alias": True,
                })

    for packed in merged:
        public, legacy, sem, action, selected, complete, successor, outcome = _ROW.unpack(packed)
        key = public + legacy
        if current_key is not None and key != current_key:
            finish()
            semantic.clear(); actions.clear(); successors.clear(); outcomes.clear()
            group_records = group_selected = group_incomplete = 0
        current_key = key
        semantic.add(sem); actions.add(action)
        if successor != _SHA_ZERO: successors.add(successor)
        if outcome != _SHA_ZERO: outcomes.add(outcome)
        group_records += 1; group_selected += selected; group_incomplete += int(not complete)
        option_count += 1; selected_count += selected; incomplete_count += int(not complete)
    finish()
    return {
        "all_option_record_count": option_count,
        "decision_count": selected_count,
        "repeated_legacy_key_group_count": repeated_group_count,
        "legacy_public_semantic_collision_group_count": legacy_collision_count,
        "legacy_collision_record_count": collision_record_count,
        "selected_action_in_legacy_collision_record_count": selected_collision_count,
        "known_successor_divergent_collision_group_count": divergent_successor_count,
        "known_outcome_divergent_collision_group_count": divergent_outcome_count,
        "incomplete_pinned_transition_evidence_record_count": incomplete_count,
        "unresolved_repaired_semantic_key_collision_group_count": unresolved_new_collision_count,
        "example_groups": examples,
    }


def extract_unresolved_repaired_key_inventory(spools: Sequence[Path]) -> list[dict[str, Any]]:
    """Return exact repaired-key groups that still bind multiple raw actions."""

    result: list[dict[str, Any]] = []
    current_key: bytes | None = None
    semantic: set[bytes] = set()
    actions: set[bytes] = set()

    def finish() -> None:
        if current_key is not None and len(semantic) == 1 and len(actions) > 1:
            result.append(
                {
                    "canonical_public_observation_hash": _sha_text(current_key[:32]),
                    "legacy_current_feature_token_hash": _sha_text(current_key[32:]),
                    "new_complete_semantic_option_key_sha256": _sha_text(next(iter(semantic))),
                    "candidate_action_sha256s": sorted(_sha_text(value) for value in actions),
                }
            )

    for packed in heapq.merge(*(_spool_rows(path) for path in spools)):
        public, legacy, sem, action, _selected, _complete, _successor, _outcome = _ROW.unpack(packed)
        key = public + legacy
        if current_key is not None and key != current_key:
            finish(); semantic.clear(); actions.clear()
        current_key = key
        semantic.add(sem); actions.add(action)
    finish()
    return result


def _extract_target_rows_one(task: tuple[str, str, frozenset[tuple[str, str]]]) -> list[dict[str, Any]]:
    day, path_text, keys = task
    found: list[dict[str, Any]] = []
    with Path(path_text).open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as stream:
        header = json.loads(next(stream))
        if header.get("utc_day") != day or header.get("goal_contract_sha256") != CONTRACT_SHA256:
            raise Recent20CensusError(f"foreign target-extraction shard {day}")
        for line in stream:
            record = json.loads(line)
            key = (
                str(record.get("canonical_public_observation_hash")),
                str(record.get("legacy_current_feature_token_hash")),
            )
            if key in keys:
                found.append(record)
    return found


def extract_targeted_equivalence_rows(
    *,
    manifest_path: Path,
    spool_root: Path,
    output_path: Path,
    workers: int = 20,
) -> dict[str, Any]:
    """Extract only unresolved repaired-key rows; never reopen raw ZIPs."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("finalized_shard_count") != 20:
        raise Recent20CensusError("foreign recent-20 manifest")
    spools = sorted(spool_root.glob("*.records.bin"))
    if len(spools) != 20:
        raise Recent20CensusError("completed private spool inventory is incomplete")
    inventory = extract_unresolved_repaired_key_inventory(spools)
    keys = frozenset(
        (row["canonical_public_observation_hash"], row["legacy_current_feature_token_hash"])
        for row in inventory
    )
    tasks = []
    for row in manifest["shards"]:
        if row["source_host"] == "elmo":
            source = Path("/home/inzi/poke-bot-agent/outputs/quarantine/alakazam-elmo-rule-derivative/g9-recent20-15gb-refeaturized-shards") / row["filename"]
        else:
            source = Path(row["day_receipt_path"]).parent / "refeatured-records" / "shards" / row["filename"]
        tasks.append((row["utc_day"], str(source), keys))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        per_day = list(executor.map(_extract_target_rows_one, tasks))
    records = [record for day in per_day for record in day]
    observed_keys = {
        (record["canonical_public_observation_hash"], record["legacy_current_feature_token_hash"])
        for record in records
    }
    if observed_keys != keys:
        raise Recent20CensusError("target extraction did not recover every unresolved key")
    payload = {
        "schema": "poke_bot.alakazam_recent20_targeted_permutation_equivalence_rows/v1",
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "unresolved_group_count": len(inventory),
        "target_record_count": len(records),
        "inventory": inventory,
        "records": records,
        "raw_zip_reopened": False,
    }
    digest = _write_create_only(output_path, payload)
    return {"path": str(output_path), "sha256": digest, "group_count": len(inventory), "record_count": len(records)}


def _actor_relative_raw_option_payload(record: Mapping[str, Any]) -> Any:
    """Normalize only a raw option's acting-seat player index.

    Candidate-list position is stored separately in ``candidate_action`` and
    is intentionally absent here.  No serial, source slot, target, card, or
    other simulator-bearing field is removed.
    """

    source = record.get("source")
    if not isinstance(source, Mapping) or source.get("acting_seat") not in (0, 1):
        raise Recent20CensusError("target row lacks a valid acting seat")
    acting_seat = source["acting_seat"]

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, child in value.items():
                if key == "playerIndex" and child == acting_seat:
                    result[str(key)] = "acting"
                else:
                    result[str(key)] = normalize(child)
            return result
        if isinstance(value, list):
            return [normalize(child) for child in value]
        return value

    return normalize(record.get("complete_raw_option_payload"))


def seal_permutation_equivalence_receipt(
    *,
    targeted_rows_path: Path,
    output_path: Path,
    collision_census_receipt_sha256: str,
) -> dict[str, Any]:
    """Adjudicate repaired-key duplicates without simulator guesswork.

    The receipt accepts only groups that are identical after the two
    explicitly nonsemantic presentation transforms required by the contract:
    candidate-list permutation and actor-relative player indexing.  Any other
    raw option difference fails closed.
    """

    targeted = json.loads(targeted_rows_path.read_text())
    if (
        targeted.get("schema") != "poke_bot.alakazam_recent20_targeted_permutation_equivalence_rows/v1"
        or targeted.get("goal_revision") != GOAL_REVISION
        or targeted.get("goal_contract_sha256") != CONTRACT_SHA256
        or targeted.get("raw_zip_reopened") is not False
    ):
        raise Recent20CensusError("foreign targeted equivalence rows")
    records = targeted.get("records")
    inventory = targeted.get("inventory")
    if not isinstance(records, list) or not isinstance(inventory, list):
        raise Recent20CensusError("malformed targeted equivalence inventory")

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise Recent20CensusError("malformed targeted equivalence row")
        key = (
            str(record.get("canonical_public_observation_hash")),
            str(record.get("legacy_current_feature_token_hash")),
        )
        groups.setdefault(key, []).append(record)
    expected_keys = {
        (str(row["canonical_public_observation_hash"]), str(row["legacy_current_feature_token_hash"]))
        for row in inventory
    }
    if set(groups) != expected_keys or len(groups) != targeted.get("unresolved_group_count"):
        raise Recent20CensusError("targeted equivalence group inventory mismatch")

    reason_counts: Counter[str] = Counter()
    witnesses: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) < 2:
            raise Recent20CensusError("equivalence group has fewer than two rows")
        semantic = {str(row.get("new_complete_semantic_option_key_sha256")) for row in rows}
        multisets = {str(row.get("normalized_canonical_option_multiset_hash")) for row in rows}
        public_semantics = {str(row.get("normalized_public_rule_semantic_token_hash")) for row in rows}
        actions = {str(row.get("candidate_action_sha256")) for row in rows}
        normalized_raw = {canonical_sha256(_actor_relative_raw_option_payload(row)) for row in rows}
        raw = {canonical_sha256(row.get("complete_raw_option_payload")) for row in rows}
        if (
            len(semantic) != 1
            or len(multisets) != 1
            or len(public_semantics) != 1
            or len(actions) < 2
            or len(normalized_raw) != 1
            or any(row.get("complete_raw_option_payload_audit_only") is not True for row in rows)
        ):
            raise Recent20CensusError("repaired-key group is not harmless permutation-equivalent")
        reason = (
            "candidate_order_only_identical_raw_payload"
            if len(raw) == 1
            else "candidate_order_and_actor_relative_player_index_only"
        )
        reason_counts[reason] += 1
        witnesses.append(
            {
                "canonical_public_observation_hash": key[0],
                "legacy_current_feature_token_hash": key[1],
                "repaired_semantic_option_key_sha256": next(iter(semantic)),
                "normalized_option_multiset_hash": next(iter(multisets)),
                "normalized_public_semantic_token_hash": next(iter(public_semantics)),
                "actor_relative_raw_option_payload_sha256": next(iter(normalized_raw)),
                "record_count": len(rows),
                "distinct_candidate_action_count": len(actions),
                "equivalence_reason": reason,
            }
        )
    if len(witnesses) != 99 or sum(row["record_count"] for row in witnesses) != 199:
        raise Recent20CensusError("unexpected repaired-key equivalence cardinality")

    receipt = {
        "schema": PERMUTATION_RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": targeted["recent20_manifest_sha256"],
        "collision_census_receipt_sha256": collision_census_receipt_sha256,
        "targeted_equivalence_rows_sha256": sha256_file(targeted_rows_path),
        "targeted_equivalence_group_count": len(witnesses),
        "targeted_equivalence_record_count": sum(row["record_count"] for row in witnesses),
        "equivalence_reason_counts": dict(sorted(reason_counts.items())),
        "all_groups_same_public_semantic_state": True,
        "all_groups_same_complete_semantic_option_key": True,
        "all_groups_same_normalized_option_multiset": True,
        "all_groups_same_actor_relative_raw_option_payload": True,
        "incidental_candidate_order_encoded": False,
        "blind_global_player_index_encoded": False,
        "pinned_simulator_execution_required_for_harmless_permutation_equivalence": False,
        "harmless_permutation_equivalence_passed": True,
        "witnesses": witnesses,
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    digest = _write_create_only(output_path, receipt)
    return {
        "path": str(output_path),
        "sha256": digest,
        "group_count": len(witnesses),
        "record_count": receipt["targeted_equivalence_record_count"],
        "equivalence_reason_counts": receipt["equivalence_reason_counts"],
    }


def extract_legacy_collision_key_inventory(spools: Sequence[Path]) -> list[dict[str, Any]]:
    """Return every legacy public/feature key split by repaired semantics."""

    result: list[dict[str, Any]] = []
    current_key: bytes | None = None
    semantics: set[bytes] = set()
    record_count = selected_count = 0

    def finish() -> None:
        if current_key is not None and len(semantics) > 1:
            result.append(
                {
                    "canonical_public_observation_hash": _sha_text(current_key[:32]),
                    "legacy_current_feature_token_hash": _sha_text(current_key[32:]),
                    "repaired_semantic_option_key_sha256s": sorted(_sha_text(value) for value in semantics),
                    "record_count": record_count,
                    "selected_record_count": selected_count,
                }
            )

    for packed in heapq.merge(*(_spool_rows(path) for path in spools)):
        public, legacy, semantic, _action, selected, _complete, _successor, _outcome = _ROW.unpack(packed)
        key = public + legacy
        if current_key is not None and key != current_key:
            finish(); semantics.clear(); record_count = selected_count = 0
        current_key = key
        semantics.add(semantic); record_count += 1; selected_count += selected
    finish()
    return result


def _init_legacy_alias_worker(keys: frozenset[tuple[str, str]]) -> None:
    global _LEGACY_ALIAS_KEYS
    _LEGACY_ALIAS_KEYS = keys


def _semantic_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    semantic = record.get("complete_semantic_option_key")
    if not isinstance(semantic, Mapping):
        raise Recent20CensusError("missing complete semantic option key")
    return {
        "factorized_stage": semantic.get("factorized_stage"),
        "normalized_selection": semantic.get("normalized_selection"),
        "normalized_canonical_option_multiset_hash": semantic.get("normalized_canonical_option_multiset_hash"),
    }


def _scan_legacy_alias_rows_one(task: tuple[str, str, str]) -> dict[str, Any]:
    day, source_path_text, output_path_text = task
    source_path = Path(source_path_text)
    output_path = Path(output_path_text)
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8", buffering=16 * 1024 * 1024) as stream:
        header = json.loads(next(stream))
        if header.get("utc_day") != day or header.get("goal_contract_sha256") != CONTRACT_SHA256:
            raise Recent20CensusError(f"foreign legacy-alias shard {day}")
        for line in stream:
            record = json.loads(line)
            key = (
                str(record.get("canonical_public_observation_hash")),
                str(record.get("legacy_current_feature_token_hash")),
            )
            if key not in _LEGACY_ALIAS_KEYS:
                continue
            source = record.get("source")
            if not isinstance(source, Mapping):
                raise Recent20CensusError("legacy alias row lacks source")
            successor = source.get("adjacent_public_successor")
            successor_hash = (
                successor.get("canonical_public_current_hash")
                if isinstance(successor, Mapping) and successor.get("availability", "").startswith("available")
                else None
            )
            rows.append(
                {
                    "public": key[0],
                    "legacy": key[1],
                    "semantic": str(record.get("new_complete_semantic_option_key_sha256")),
                    "projection": _semantic_projection(record),
                    "selected": record.get("selected_candidate") is True,
                    "adjacent_public_successor": successor_hash,
                    "source": {
                        "utc_day": source.get("source_archive_date"),
                        "episode_id": source.get("episode_id"),
                        "acting_seat": source.get("acting_seat"),
                        "env_step": source.get("env_step"),
                        "factorized_stage": source.get("factorized_stage"),
                    },
                }
            )
    rows.sort(key=lambda row: (row["public"], row["legacy"], row["semantic"], canonical_sha256(row["source"])))
    if output_path.exists() or output_path.is_symlink():
        raise Recent20CensusError("legacy alias worker output exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb", buffering=16 * 1024 * 1024, closefd=False) as raw:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0) as compressed:
                for row in rows:
                    compressed.write(canonical_bytes(row))
            raw.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "utc_day": day,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "record_count": len(rows),
    }


def _gzip_rows(path: Path) -> Iterable[tuple[tuple[str, str, str], dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            yield (row["public"], row["legacy"], row["semantic"]), row


def _semantic_diff_paths(left: Any, right: Any, path: str = "") -> set[str]:
    if type(left) is not type(right):
        return {path or "$"}
    if isinstance(left, Mapping):
        result: set[str] = set()
        for key in set(left) | set(right):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                result.add(child)
            else:
                result.update(_semantic_diff_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = set()
        if len(left) != len(right):
            result.add(f"{path}.length")
        for lchild, rchild in zip(left, right):
            result.update(_semantic_diff_paths(lchild, rchild, f"{path}[]"))
        return result
    return {path or "$"} if left != right else set()


def build_legacy_alias_class_report(
    *,
    manifest_path: Path,
    spool_root: Path,
    work_root: Path,
    output_path: Path,
    workers: int = 20,
) -> dict[str, Any]:
    """Classify all legacy aliases by exact repaired-semantic delta paths."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("finalized_shard_count") != 20:
        raise Recent20CensusError("foreign recent-20 manifest")
    spools = sorted(spool_root.glob("*.records.bin"))
    if len(spools) != 20:
        raise Recent20CensusError("completed private spool inventory is incomplete")
    inventory = extract_legacy_collision_key_inventory(spools)
    if len(inventory) != 127641 or sum(row["record_count"] for row in inventory) != 255398:
        raise Recent20CensusError("legacy collision inventory differs from sealed census")
    keys = frozenset(
        (row["canonical_public_observation_hash"], row["legacy_current_feature_token_hash"])
        for row in inventory
    )
    if work_root.exists() or work_root.is_symlink():
        raise Recent20CensusError("legacy alias work root must be absent/create-only")
    work_root.mkdir(parents=True)
    tasks = []
    for row in manifest["shards"]:
        if row["source_host"] == "elmo":
            source = Path("/home/inzi/poke-bot-agent/outputs/quarantine/alakazam-elmo-rule-derivative/g9-recent20-15gb-refeaturized-shards") / row["filename"]
        else:
            source = Path(row["day_receipt_path"]).parent / "refeatured-records" / "shards" / row["filename"]
        tasks.append((row["utc_day"], str(source), str(work_root / f"{row['utc_day']}.legacy-alias.jsonl.gz")))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_legacy_alias_worker,
        initargs=(keys,),
    ) as executor:
        day_results = list(executor.map(_scan_legacy_alias_rows_one, tasks))
    if sum(row["record_count"] for row in day_results) != 255398:
        raise Recent20CensusError("legacy alias extraction record count mismatch")

    merged = heapq.merge(*(_gzip_rows(Path(row["path"])) for row in day_results), key=lambda item: item[0])
    current_key: tuple[str, str] | None = None
    projections: dict[str, Any] = {}
    record_count = selected_count = 0
    total_selected_missing_successor = 0
    selected_semantics: set[str] = set()
    successors_by_semantic: dict[str, set[str]] = {}
    examples_by_semantic: dict[str, Mapping[str, Any]] = {}
    class_counts: Counter[tuple[str, ...]] = Counter()
    class_records: Counter[tuple[str, ...]] = Counter()
    class_selected: Counter[tuple[str, ...]] = Counter()
    class_examples: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    group_count = selected_multi_semantic_group_count = divergent_successor_group_count = 0

    def finish() -> None:
        nonlocal group_count, selected_multi_semantic_group_count, divergent_successor_group_count
        if current_key is None:
            return
        if len(projections) < 2:
            raise Recent20CensusError("legacy alias group lost a repaired semantic")
        ordered = sorted(projections)
        paths: set[str] = set()
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                paths.update(_semantic_diff_paths(projections[left], projections[right]))
        if not paths:
            raise Recent20CensusError("distinct repaired semantic hashes have no semantic delta")
        signature = tuple(sorted(paths))
        class_counts[signature] += 1
        class_records[signature] += record_count
        class_selected[signature] += selected_count
        group_count += 1
        selected_multi_semantic_group_count += int(len(selected_semantics) > 1)
        known_successors = set().union(*successors_by_semantic.values()) if successors_by_semantic else set()
        divergent_successor_group_count += int(len(known_successors) > 1 and len(successors_by_semantic) > 1)
        examples = class_examples.setdefault(signature, [])
        if len(examples) < 3:
            examples.append(
                {
                    "canonical_public_observation_hash": current_key[0],
                    "legacy_current_feature_token_hash": current_key[1],
                    "semantic_key_sha256s": ordered,
                    "record_count": record_count,
                    "selected_record_count": selected_count,
                    "selected_semantic_count": len(selected_semantics),
                    "known_selected_successor_count": len(known_successors),
                    "representative_sources": dict(sorted(examples_by_semantic.items())),
                }
            )

    for (_public, _legacy, _semantic), row in merged:
        key = (row["public"], row["legacy"])
        if current_key is not None and key != current_key:
            finish(); projections.clear(); selected_semantics.clear(); successors_by_semantic.clear(); examples_by_semantic.clear()
            record_count = selected_count = 0
        current_key = key
        semantic = row["semantic"]
        projection = row["projection"]
        previous = projections.setdefault(semantic, projection)
        if previous != projection:
            raise Recent20CensusError("one repaired semantic hash maps to multiple projections")
        record_count += 1
        if row["selected"]:
            selected_count += 1; selected_semantics.add(semantic)
            if row["adjacent_public_successor"] is None:
                total_selected_missing_successor += 1
            else:
                successors_by_semantic.setdefault(semantic, set()).add(row["adjacent_public_successor"])
        examples_by_semantic.setdefault(semantic, row["source"])
    finish()
    if group_count != 127641:
        raise Recent20CensusError("legacy alias class merge group count mismatch")

    classes = []
    for signature, count in class_counts.most_common():
        classes.append(
            {
                "semantic_delta_paths": list(signature),
                "group_count": count,
                "record_count": class_records[signature],
                "selected_record_count": class_selected[signature],
                "examples": class_examples[signature],
            }
        )
    report = {
        "schema": "poke_bot.alakazam_recent20_legacy_alias_semantic_delta_classes/v1",
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "legacy_alias_group_count": group_count,
        "legacy_alias_record_count": sum(row["record_count"] for row in day_results),
        "semantic_delta_class_count": len(classes),
        "groups_with_multiple_selected_repaired_semantics": selected_multi_semantic_group_count,
        "groups_with_divergent_recorded_adjacent_public_successors": divergent_successor_group_count,
        "selected_records_missing_adjacent_public_successor": total_selected_missing_successor,
        "classes": classes,
        "worker_outputs": sorted(day_results, key=lambda row: row["utc_day"]),
        "raw_zip_reopened": False,
        "pinned_fresh_simulator_execution_performed": False,
        "status": "complete_semantic_delta_class_inventory_pending_targeted_engine_witnesses",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    digest = _write_create_only(output_path, report)
    return {
        "path": str(output_path),
        "sha256": digest,
        "legacy_alias_group_count": group_count,
        "legacy_alias_record_count": report["legacy_alias_record_count"],
        "semantic_delta_class_count": len(classes),
        "groups_with_multiple_selected_repaired_semantics": selected_multi_semantic_group_count,
        "groups_with_divergent_recorded_adjacent_public_successors": divergent_successor_group_count,
    }


def seal_targeted_transition_witness_receipt(
    *,
    semantic_class_report_path: Path,
    permutation_receipt_path: Path,
    recent20_manifest_path: Path,
    frozen_schema_manifest_path: Path,
    canonical_libcg_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Seal corpus transition and legality witnesses for all delta classes.

    Selected actions are witnessed only by their recorded realized adjacent
    public successor.  Unselected legal options are never assigned invented
    counterfactual outcomes; their exact libcg-emitted menu distinctions are
    legality witnesses only, as required by the target contract.
    """

    classes = json.loads(semantic_class_report_path.read_text())
    permutation = json.loads(permutation_receipt_path.read_text())
    manifest = json.loads(recent20_manifest_path.read_text())
    freeze = json.loads(frozen_schema_manifest_path.read_text())
    if (
        classes.get("schema") != "poke_bot.alakazam_recent20_legacy_alias_semantic_delta_classes/v1"
        or classes.get("goal_revision") != GOAL_REVISION
        or classes.get("goal_contract_sha256") != CONTRACT_SHA256
        or classes.get("legacy_alias_group_count") != 127641
        or classes.get("legacy_alias_record_count") != 255398
        or classes.get("semantic_delta_class_count") != 6
        or classes.get("raw_zip_reopened") is not False
        or permutation.get("schema") != PERMUTATION_RECEIPT_SCHEMA
        or permutation.get("harmless_permutation_equivalence_passed") is not True
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("goal_revision") != GOAL_REVISION
        or manifest.get("goal_contract_sha256") != CONTRACT_SHA256
        or manifest.get("finalized_shard_count") != 20
        or manifest.get("record_count") != 18412973
        or freeze.get("schema") != "poke_bot.alakazam_collision_census_r298_frozen_schema_manifest/v1"
    ):
        raise Recent20CensusError("foreign targeted transition witness inputs")
    if classes.get("recent20_manifest_sha256") != sha256_file(recent20_manifest_path):
        raise Recent20CensusError("semantic class report does not bind the recent-20 manifest")
    canonical_simulator = freeze.get("canonical_simulator")
    if not isinstance(canonical_simulator, Mapping):
        raise Recent20CensusError("frozen schema lacks canonical simulator identity")
    libcg_sha = sha256_file(canonical_libcg_path)
    if (
        libcg_sha != "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
        or canonical_simulator.get("linux_x86_64_sha256") != libcg_sha
        or canonical_simulator.get("linux_x86_64_size_bytes") != canonical_libcg_path.stat().st_size
    ):
        raise Recent20CensusError("canonical libcg bytes differ from frozen schema")

    class_witnesses: list[dict[str, Any]] = []
    selected_transition_class_count = legality_only_class_count = 0
    covered_group_count = 0
    for index, row in enumerate(classes.get("classes", [])):
        if not isinstance(row, Mapping):
            raise Recent20CensusError("malformed semantic delta class")
        selected = int(row.get("selected_record_count", -1))
        count = int(row.get("group_count", -1))
        examples = row.get("examples")
        if count < 1 or not isinstance(examples, list) or not examples:
            raise Recent20CensusError("semantic delta class lacks witnesses")
        evidence_scope = (
            "recorded_selected_realized_adjacent_public_successor"
            if selected > 0
            else "pinned_libcg_emitted_legal_option_menu_only_no_counterfactual_outcome"
        )
        if selected > 0:
            selected_transition_class_count += 1
            if any(example.get("known_selected_successor_count", 0) < 1 for example in examples):
                raise Recent20CensusError("selected transition class example lacks adjacent successor")
        else:
            legality_only_class_count += 1
            if any(example.get("selected_record_count") != 0 for example in examples):
                raise Recent20CensusError("legality-only class unexpectedly has a selected action")
        covered_group_count += count
        class_witnesses.append(
            {
                "class_index": index,
                "semantic_delta_paths": row["semantic_delta_paths"],
                "group_count": count,
                "record_count": row["record_count"],
                "selected_record_count": selected,
                "evidence_scope": evidence_scope,
                "representative_witnesses": examples,
                "counterfactual_outcome_claimed": False,
            }
        )
    if (
        len(class_witnesses) != 6
        or covered_group_count != 127641
        or selected_transition_class_count != 5
        or legality_only_class_count != 1
        or classes.get("selected_records_missing_adjacent_public_successor") != 0
    ):
        raise Recent20CensusError("targeted witness class coverage is incomplete")

    receipt = {
        "schema": TARGETED_WITNESS_RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(recent20_manifest_path),
        "semantic_delta_class_report_sha256": sha256_file(semantic_class_report_path),
        "permutation_equivalence_receipt_sha256": sha256_file(permutation_receipt_path),
        "frozen_schema_manifest_sha256": sha256_file(frozen_schema_manifest_path),
        "canonical_libcg_linux_x86_64_sha256": libcg_sha,
        "canonical_libcg_linux_x86_64_size_bytes": canonical_libcg_path.stat().st_size,
        "legal_option_authority": "exact_pinned_official_libcg_binary_by_sha256",
        "semantic_delta_class_count": 6,
        "covered_legacy_alias_group_count": covered_group_count,
        "selected_transition_witness_class_count": selected_transition_class_count,
        "legality_only_witness_class_count": legality_only_class_count,
        "selected_records_missing_adjacent_public_successor": 0,
        "recorded_selected_transition_witnesses_complete_for_class_representatives": True,
        "counterfactual_labels_for_unchosen_actions": False,
        "fresh_simulator_counterfactual_execution_performed": False,
        "public_rule_semantic_projection_corpus_supported": True,
        "other_derivative_branches_training_supported_by_this_receipt": False,
        "class_witnesses": class_witnesses,
        "status": "passed_targeted_recorded_transition_and_pinned_legality_witnesses",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    digest = _write_create_only(output_path, receipt)
    return {"path": str(output_path), "sha256": digest, "status": receipt["status"]}


def reseal_with_targeted_witness_evidence(
    *,
    previous_report_path: Path,
    previous_census_receipt_path: Path,
    previous_branch_receipt_path: Path,
    targeted_witness_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Promote only corpus-supported branches; keep all others exact-zero."""

    report = json.loads(previous_report_path.read_text())
    previous_census = json.loads(previous_census_receipt_path.read_text())
    previous_branch = json.loads(previous_branch_receipt_path.read_text())
    witness = json.loads(targeted_witness_receipt_path.read_text())
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("goal_revision") != GOAL_REVISION
        or report.get("goal_contract_sha256") != CONTRACT_SHA256
        or previous_census.get("schema") != RECEIPT_SCHEMA
        or previous_census.get("collision_census_report_sha256") != sha256_file(previous_report_path)
        or previous_census.get("permutation_equivalence_evidence_complete") is not True
        or previous_census.get("actionable_unresolved_repaired_collision_group_count") != 0
        or previous_branch.get("schema") != BRANCH_SCHEMA
        or previous_branch.get("collision_census_receipt_sha256") != sha256_file(previous_census_receipt_path)
        or witness.get("schema") != TARGETED_WITNESS_RECEIPT_SCHEMA
        or witness.get("goal_revision") != GOAL_REVISION
        or witness.get("goal_contract_sha256") != CONTRACT_SHA256
        or witness.get("recent20_manifest_sha256") != report.get("recent20_manifest_sha256")
        or witness.get("public_rule_semantic_projection_corpus_supported") is not True
        or witness.get("other_derivative_branches_training_supported_by_this_receipt") is not False
    ):
        raise Recent20CensusError("foreign or inconsistent targeted witness bundle")
    if output_root.exists() or output_root.is_symlink():
        raise Recent20CensusError("targeted witness reseal root must be absent/create-only")
    output_root.mkdir(parents=True)

    updated_report = dict(report)
    updated_report["producer_source_sha256"] = sha256_file(Path(__file__))
    updated_report["supersedes_collision_census_report_sha256"] = sha256_file(previous_report_path)
    updated_report["targeted_transition_witness_evidence"] = {
        "receipt_sha256": sha256_file(targeted_witness_receipt_path),
        "semantic_delta_class_count": witness["semantic_delta_class_count"],
        "covered_legacy_alias_group_count": witness["covered_legacy_alias_group_count"],
        "selected_transition_witness_class_count": witness["selected_transition_witness_class_count"],
        "legality_only_witness_class_count": witness["legality_only_witness_class_count"],
        "counterfactual_labels_for_unchosen_actions": False,
    }
    updated_report["status"] = "passed_complete_collision_census_public_rule_semantic_projection_supported"
    report_path = output_root / "collision-census-report.json"
    report_sha = _write_create_only(report_path, updated_report)
    analysis = report["analysis"]
    census = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "superseded_collision_census_receipt_sha256": sha256_file(previous_census_receipt_path),
        "permutation_equivalence_receipt_sha256": previous_census["permutation_equivalence_receipt_sha256"],
        "targeted_transition_witness_receipt_sha256": sha256_file(targeted_witness_receipt_path),
        "utc_partition_count": 20,
        "all_option_record_count": analysis["all_option_record_count"],
        "decision_count": analysis["decision_count"],
        "legacy_collision_group_count": analysis["legacy_public_semantic_collision_group_count"],
        "actionable_unresolved_repaired_collision_group_count": 0,
        "permutation_equivalence_evidence_complete": True,
        "targeted_transition_witness_evidence_complete": True,
        "public_rule_semantic_projection_corpus_supported": True,
        "all_derivative_branches_transition_evidence_complete": False,
        "candidate_training_allowed": False,
        "status": updated_report["status"],
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    census_path = output_root / "collision-census-receipt.json"
    census_sha = _write_create_only(census_path, census)
    branch = {
        "schema": BRANCH_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_receipt_sha256": census_sha,
        "targeted_transition_witness_receipt_sha256": sha256_file(targeted_witness_receipt_path),
        "eligible_trainable_branches": ["public_rule_semantic_projection"],
        "representationally_supported_but_not_yet_trainable_branches": [],
        "unsupported_or_unproven_branches": [
            "public_rule_metadata_residual",
            "r298_repaired_auxiliary_heads",
            "eight_checklist_provenance_gates",
        ],
        "unsupported_branches_exact_zero_and_inert": True,
        "targeted_simulator_transition_evidence_required_for_public_rule_semantic_projection": False,
        "candidate_training_allowed": False,
        "candidate_training_blockers": [
            "parent_schema_corpus_frozen_tensor_blackwell_rollback_readiness_and_activation_receipts"
        ],
        "status": "passed_branch_adjudication_public_rule_semantic_projection_only",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    branch_path = output_root / "branch-adjudication-receipt.json"
    branch_sha = _write_create_only(branch_path, branch)
    completion = {
        "schema": "poke_bot.alakazam_recent20_collision_census_bundle/v1",
        "goal_revision": GOAL_REVISION,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "collision_census_receipt_sha256": census_sha,
        "targeted_transition_witness_receipt_sha256": sha256_file(targeted_witness_receipt_path),
        "branch_adjudication_receipt_sha256": branch_sha,
        "status": branch["status"],
    }
    completion_sha = _write_create_only(output_root / "COMPLETE.json", completion)
    return {**completion, "completion_receipt_sha256": completion_sha, "output_root": str(output_root)}


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise Recent20CensusError(f"output exists: {path}")
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


def reseal_completed_scan(
    *,
    previous_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Correct/reseal merge logic from verified completed private spools.

    This deliberately does not reopen any feature shard.  It validates every
    private spool identity recorded by the completed scan, reruns only the
    deterministic parent merge, and emits a new immutable bundle that names
    the superseded diagnostic report.
    """

    previous = json.loads(previous_report_path.read_text())
    if (
        previous.get("schema") != REPORT_SCHEMA
        or previous.get("goal_revision") != GOAL_REVISION
        or previous.get("goal_contract_sha256") != CONTRACT_SHA256
    ):
        raise Recent20CensusError("foreign previous census report")
    day_scans = previous.get("day_scans")
    if not isinstance(day_scans, list) or len(day_scans) != 20:
        raise Recent20CensusError("previous scan inventory is incomplete")
    spools: list[Path] = []
    for row in day_scans:
        if not isinstance(row, Mapping):
            raise Recent20CensusError("malformed previous day scan")
        path = Path(str(row.get("spool_path")))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("spool_size_bytes")
            or sha256_file(path) != row.get("spool_sha256")
        ):
            raise Recent20CensusError("private spool identity mismatch")
        spools.append(path)
    if output_root.exists() or output_root.is_symlink():
        raise Recent20CensusError("reseal output root must be absent/create-only")
    output_root.mkdir(parents=True)
    analysis = _analyse_sorted(spools)
    if analysis["all_option_record_count"] != sum(
        int(row["option_record_count"]) for row in day_scans
    ):
        raise Recent20CensusError("resealed merge record count changed")
    report = dict(previous)
    report["analysis"] = analysis
    report["producer_source_sha256"] = sha256_file(Path(__file__))
    report["supersedes_diagnostic_report_sha256"] = sha256_file(previous_report_path)
    report["status"] = (
        "complete_representational_census_pending_targeted_simulator_transition_and_permutation_evidence"
    )
    report_path = output_root / "collision-census-report.json"
    report_sha = _write_create_only(report_path, report)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "superseded_diagnostic_report_sha256": sha256_file(previous_report_path),
        "utc_partition_count": 20,
        "all_option_record_count": analysis["all_option_record_count"],
        "decision_count": analysis["decision_count"],
        "legacy_collision_group_count": analysis["legacy_public_semantic_collision_group_count"],
        "unresolved_repaired_collision_group_count": analysis["unresolved_repaired_semantic_key_collision_group_count"],
        "pinned_transition_evidence_complete": False,
        "permutation_equivalence_evidence_complete": analysis["unresolved_repaired_semantic_key_collision_group_count"] == 0,
        "candidate_training_allowed": False,
        "status": report["status"],
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    receipt_path = output_root / "collision-census-receipt.json"
    receipt_sha = _write_create_only(receipt_path, receipt)
    branch = {
        "schema": BRANCH_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_receipt_sha256": receipt_sha,
        "eligible_trainable_branches": [],
        "representationally_supported_but_not_yet_trainable_branches": [],
        "unsupported_or_unproven_branches": [
            "public_rule_semantic_projection",
            "public_rule_metadata_residual",
            "r298_repaired_auxiliary_heads",
            "eight_checklist_provenance_gates",
        ],
        "unsupported_branches_exact_zero_and_inert": True,
        "targeted_simulator_transition_evidence_required": True,
        "targeted_permutation_equivalence_evidence_required": True,
        "candidate_training_allowed": False,
        "status": "complete_fail_closed_branch_adjudication_pending_targeted_engine_and_permutation_evidence",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    branch_path = output_root / "branch-adjudication-receipt.json"
    branch_sha = _write_create_only(branch_path, branch)
    completion = {
        "schema": "poke_bot.alakazam_recent20_collision_census_bundle/v1",
        "goal_revision": GOAL_REVISION,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "collision_census_receipt_sha256": receipt_sha,
        "branch_adjudication_receipt_sha256": branch_sha,
        "status": branch["status"],
    }
    completion_sha = _write_create_only(output_root / "COMPLETE.json", completion)
    return {**completion, "completion_receipt_sha256": completion_sha, "output_root": str(output_root)}


def reseal_with_permutation_evidence(
    *,
    previous_report_path: Path,
    previous_census_receipt_path: Path,
    permutation_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Bind harmless-permutation evidence without rescanning feature shards."""

    report = json.loads(previous_report_path.read_text())
    previous_receipt = json.loads(previous_census_receipt_path.read_text())
    permutation = json.loads(permutation_receipt_path.read_text())
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("goal_revision") != GOAL_REVISION
        or report.get("goal_contract_sha256") != CONTRACT_SHA256
        or previous_receipt.get("schema") != RECEIPT_SCHEMA
        or previous_receipt.get("collision_census_report_sha256") != sha256_file(previous_report_path)
        or permutation.get("schema") != PERMUTATION_RECEIPT_SCHEMA
        or permutation.get("goal_revision") != GOAL_REVISION
        or permutation.get("goal_contract_sha256") != CONTRACT_SHA256
        or permutation.get("collision_census_receipt_sha256") != sha256_file(previous_census_receipt_path)
        or permutation.get("harmless_permutation_equivalence_passed") is not True
        or permutation.get("targeted_equivalence_group_count") != 99
    ):
        raise Recent20CensusError("foreign or inconsistent permutation evidence bundle")
    analysis = report.get("analysis")
    if (
        not isinstance(analysis, Mapping)
        or analysis.get("unresolved_repaired_semantic_key_collision_group_count") != 99
        or report.get("recent20_manifest_sha256") != permutation.get("recent20_manifest_sha256")
    ):
        raise Recent20CensusError("permutation evidence does not cover the census hold")
    if output_root.exists() or output_root.is_symlink():
        raise Recent20CensusError("permutation reseal output root must be absent/create-only")
    output_root.mkdir(parents=True)

    updated_report = dict(report)
    updated_report["producer_source_sha256"] = sha256_file(Path(__file__))
    updated_report["supersedes_collision_census_report_sha256"] = sha256_file(previous_report_path)
    updated_report["permutation_equivalence_evidence"] = {
        "receipt_sha256": sha256_file(permutation_receipt_path),
        "raw_repaired_key_duplicate_group_count": 99,
        "harmless_permutation_equivalent_group_count": 99,
        "actionable_unresolved_repaired_collision_group_count": 0,
        "reason_counts": permutation["equivalence_reason_counts"],
    }
    updated_report["status"] = "complete_representational_census_pending_targeted_simulator_transition_evidence"
    report_path = output_root / "collision-census-report.json"
    report_sha = _write_create_only(report_path, updated_report)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "superseded_collision_census_receipt_sha256": sha256_file(previous_census_receipt_path),
        "permutation_equivalence_receipt_sha256": sha256_file(permutation_receipt_path),
        "utc_partition_count": 20,
        "all_option_record_count": analysis["all_option_record_count"],
        "decision_count": analysis["decision_count"],
        "legacy_collision_group_count": analysis["legacy_public_semantic_collision_group_count"],
        "raw_repaired_key_duplicate_group_count": 99,
        "harmless_permutation_equivalent_group_count": 99,
        "actionable_unresolved_repaired_collision_group_count": 0,
        "pinned_transition_evidence_complete": False,
        "permutation_equivalence_evidence_complete": True,
        "candidate_training_allowed": False,
        "status": updated_report["status"],
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    receipt_path = output_root / "collision-census-receipt.json"
    receipt_sha = _write_create_only(receipt_path, receipt)
    branch = {
        "schema": BRANCH_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_receipt_sha256": receipt_sha,
        "permutation_equivalence_receipt_sha256": sha256_file(permutation_receipt_path),
        "eligible_trainable_branches": [],
        "representationally_supported_but_not_yet_trainable_branches": [
            "public_rule_semantic_projection"
        ],
        "unsupported_or_unproven_branches": [
            "public_rule_metadata_residual",
            "r298_repaired_auxiliary_heads",
            "eight_checklist_provenance_gates",
        ],
        "unsupported_branches_exact_zero_and_inert": True,
        "targeted_simulator_transition_evidence_required": True,
        "targeted_permutation_equivalence_evidence_required": False,
        "candidate_training_allowed": False,
        "status": "complete_fail_closed_branch_adjudication_pending_targeted_engine_evidence",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    branch_path = output_root / "branch-adjudication-receipt.json"
    branch_sha = _write_create_only(branch_path, branch)
    completion = {
        "schema": "poke_bot.alakazam_recent20_collision_census_bundle/v1",
        "goal_revision": GOAL_REVISION,
        "recent20_manifest_sha256": report["recent20_manifest_sha256"],
        "collision_census_report_sha256": report_sha,
        "collision_census_receipt_sha256": receipt_sha,
        "permutation_equivalence_receipt_sha256": sha256_file(permutation_receipt_path),
        "branch_adjudication_receipt_sha256": branch_sha,
        "status": branch["status"],
    }
    completion_sha = _write_create_only(output_root / "COMPLETE.json", completion)
    return {**completion, "completion_receipt_sha256": completion_sha, "output_root": str(output_root)}


def run(*, manifest_path: Path, output_root: Path, spool_root: Path, workers: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("goal_revision") != GOAL_REVISION
        or manifest.get("goal_contract_sha256") != CONTRACT_SHA256
        or manifest.get("finalized_shard_count") != 20
    ):
        raise Recent20CensusError("foreign or incomplete recent-20 manifest")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 20:
        raise Recent20CensusError("recent-20 shard inventory is incomplete")
    if output_root.exists() or output_root.is_symlink():
        raise Recent20CensusError("output root must be absent/create-only")
    output_root.mkdir(parents=True)
    spool_root.mkdir(parents=True, exist_ok=False)
    tasks = []
    for row in shards:
        digest = row["sha256"]
        # The parity receipt has already established the exact Inzi path.  Ten
        # Elmo objects live in quarantine; ten Inzi-local objects remain in
        # their day roots without a redundant copy.
        if row["source_host"] == "elmo":
            source = Path("/home/inzi/poke-bot-agent/outputs/quarantine/alakazam-elmo-rule-derivative/g9-recent20-15gb-refeaturized-shards") / row["filename"]
        else:
            source = Path(row["day_receipt_path"]).parent / "refeatured-records" / "shards" / row["filename"]
        tasks.append((row["utc_day"], str(source), str(spool_root / f"{row['utc_day']}.records.bin")))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        day_results = list(executor.map(_scan_one, tasks))
    analysis = _analyse_sorted([Path(row["spool_path"]) for row in day_results])
    deck_records: Counter[str] = Counter(); deck_decisions: Counter[str] = Counter()
    numbers: Counter[str] = Counter(); fields: Counter[str] = Counter()
    for row in day_results:
        deck_records.update(row["acting_deck_record_counts"])
        deck_decisions.update(row["acting_deck_decision_counts"])
        numbers.update(row["number_value_counts"])
        fields.update(row["semantic_field_presence_counts"])
    report = {
        "schema": REPORT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "window_start_utc": manifest["window_start_utc"],
        "window_end_utc": manifest["window_end_utc"],
        "utc_partition_count": 20,
        "row_scope": manifest["row_scope"],
        "analysis": analysis,
        "acting_deck_variant_count": len(deck_records),
        "acting_deck_variant_record_counts": dict(sorted(deck_records.items())),
        "acting_deck_variant_decision_counts": dict(sorted(deck_decisions.items())),
        "observed_number_value_counts": dict(sorted(numbers.items(), key=lambda item: int(item[0]))),
        "semantic_field_presence_counts": dict(sorted(fields.items())),
        "day_scans": sorted(day_results, key=lambda row: row["utc_day"]),
        "status": (
            "complete_representational_census_pending_targeted_simulator_transition_evidence"
            if analysis["incomplete_pinned_transition_evidence_record_count"]
            else "passed_complete_no_unresolved_repaired_semantic_collision"
        ),
    }
    report_path = output_root / "collision-census-report.json"
    report_sha = _write_create_only(report_path, report)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "collision_census_report_sha256": report_sha,
        "utc_partition_count": 20,
        "all_option_record_count": analysis["all_option_record_count"],
        "decision_count": analysis["decision_count"],
        "legacy_collision_group_count": analysis["legacy_public_semantic_collision_group_count"],
        "unresolved_repaired_collision_group_count": analysis["unresolved_repaired_semantic_key_collision_group_count"],
        "pinned_transition_evidence_complete": analysis["incomplete_pinned_transition_evidence_record_count"] == 0,
        "candidate_training_allowed": False,
        "status": report["status"],
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    receipt_path = output_root / "collision-census-receipt.json"
    receipt_sha = _write_create_only(receipt_path, receipt)
    supported = []
    if analysis["legacy_public_semantic_collision_group_count"] and not analysis["unresolved_repaired_semantic_key_collision_group_count"]:
        supported.append("public_rule_semantic_projection")
    unsupported = [
        name for name in (
            "public_rule_metadata_residual",
            "r298_repaired_auxiliary_heads",
            "eight_checklist_provenance_gates",
        ) if name not in supported
    ]
    branch = {
        "schema": BRANCH_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "collision_census_receipt_sha256": receipt_sha,
        "eligible_trainable_branches": [],
        "representationally_supported_but_not_yet_trainable_branches": supported,
        "unsupported_or_unproven_branches": unsupported,
        "unsupported_branches_exact_zero_and_inert": True,
        "targeted_simulator_transition_evidence_required": analysis["incomplete_pinned_transition_evidence_record_count"] > 0,
        "candidate_training_allowed": False,
        "status": "complete_fail_closed_branch_adjudication_pending_targeted_engine_evidence",
        "sealed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    branch_path = output_root / "branch-adjudication-receipt.json"
    branch_sha = _write_create_only(branch_path, branch)
    completion = {
        "schema": "poke_bot.alakazam_recent20_collision_census_bundle/v1",
        "goal_revision": GOAL_REVISION,
        "recent20_manifest_sha256": sha256_file(manifest_path),
        "collision_census_report_sha256": report_sha,
        "collision_census_receipt_sha256": receipt_sha,
        "branch_adjudication_receipt_sha256": branch_sha,
        "status": branch["status"],
    }
    completion_sha = _write_create_only(output_root / "COMPLETE.json", completion)
    return {**completion, "completion_receipt_sha256": completion_sha, "output_root": str(output_root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 20:
        raise Recent20CensusError("workers must be in [1, 20]")
    print(json.dumps(run(manifest_path=args.manifest, output_root=args.output_root, spool_root=args.spool_root, workers=args.workers), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
