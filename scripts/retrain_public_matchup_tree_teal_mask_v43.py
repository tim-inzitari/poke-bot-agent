#!/usr/bin/env python3
"""Append an exact-catalog specialist to the causal public matchup router.

Historical row shards are remapped by archetype name into the staged V6
registry. The independently simulated Spidops calibration archive can be
re-added when those shards predate its route, and the new specialist's labels
come only from its checksum-bound exact public deck catalog. Model inputs remain
cumulative opponent-public card IDs; deck contents, recorded archetypes,
hidden zones, and future observations are never features.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier

from poke_bot.ladder_replay import canonical_deck_sha256
from poke_bot.public_matchup_router import visible_opponent_card_ids
from poke_bot.replay_import import extract_setup_decks
from scripts.retrain_public_matchup_tree_spidops_v41 import (
    _remap_historical_shards,
)
from scripts.train_public_matchup_tree_v32 import (
    SCHEMA,
    UNKNOWN,
    _calibrate_runtime_thresholds,
    _episode_split,
    _expanded_predict_proba,
    _safe_card_ids,
    _tree_payload,
    _weighted_metrics,
    sha256_file,
)


TARGET_ID = "teal-mask-ogerpon-ex"
CATALOG_SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_catalog_bindings(
    target_id: str,
    catalog_path: Path,
    additional: list[str],
) -> list[tuple[str, Path]]:
    target_id = str(target_id).strip()
    if not target_id:
        raise RuntimeError("target specialist ID must be nonempty")
    bindings = [(target_id, catalog_path)]
    for raw_binding in additional:
        bound_id, separator, bound_path = str(raw_binding).partition("=")
        bound_id = bound_id.strip()
        bound_path = bound_path.strip()
        if (
            separator != "="
            or not bound_id
            or not bound_path
            or any(existing_id == bound_id for existing_id, _ in bindings)
        ):
            raise RuntimeError(
                "additional public catalog must be a unique TARGET_ID=PATH binding"
            )
        bindings.append((bound_id, Path(bound_path)))
    return bindings


def _canonical_targets(
    roster_path: Path,
    *,
    target_id: str = TARGET_ID,
) -> tuple[tuple[str, ...], dict[str, str]]:
    payload = json.loads(roster_path.read_text(encoding="utf-8"))
    targets = tuple(str(value) for value in payload.get("expert_ids") or ())
    slots = {
        str(row.get("archetype_id") or "")
        for row in (payload.get("slots") or ())
        if isinstance(row, dict)
        and str(row.get("status") or "") in {"active", "dormant"}
        and str(row.get("archetype_id") or "")
    }
    if (
        payload.get("schema") != "poke_bot.matchup_adapter_roster/v1"
        or len(targets) != len(set(targets))
        or len(targets) != int(payload.get("required_specialist_count") or 0)
        or set(targets) != slots
        or target_id not in targets
    ):
        raise RuntimeError("staged V6 matchup registry identity changed")
    aliases = {
        str(source): str(target)
        for source, target in (payload.get("logical_aliases") or {}).items()
    }
    return targets, aliases


def _catalog(
    path: Path,
    *,
    target_id: str = TARGET_ID,
) -> tuple[dict, frozenset[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    window = dict(payload.get("source_window") or {})
    observed = dict(payload.get("observed_by_day") or {})
    try:
        start = date.fromisoformat(str(window.get("start") or ""))
        end = date.fromisoformat(str(window.get("end") or ""))
    except ValueError as exc:
        raise RuntimeError("public deck catalog date window is invalid") from exc
    dates = [
        (start + timedelta(days=index)).isoformat()
        for index in range((end - start).days + 1)
    ]
    fingerprints = frozenset(
        str(value) for value in payload.get("deck_fingerprints") or ()
    )
    match_facts = payload.get("source_match_facts")
    if match_facts is not None and (
        not isinstance(match_facts, list)
        or len(match_facts) != int(payload.get("source_match_rows") or -1)
        or _canonical_digest(match_facts)
        != payload.get("source_match_facts_sha256")
    ):
        raise RuntimeError("public deck catalog match index changed")
    if (
        payload.get("schema") != CATALOG_SCHEMA
        or payload.get("specialist_id") != target_id
        or int(window.get("days") or 0) != len(dates)
        or sorted(observed) != dates
        or sum(int(value) for value in observed.values())
        != int(payload.get("observed_acting_seat_games") or 0)
        or not fingerprints
    ):
        raise RuntimeError("public deck catalog identity changed")
    return payload, fingerprints, dates


def _public_calibration_rows(
    archive_root: Path,
    *,
    archive_receipt_template: str | None,
    archive_workers: int,
    catalog_path: Path,
    label: int,
    max_card_id: int,
    split_seed: int,
    validation_percent: int,
    target_id: str = TARGET_ID,
):
    catalog, fingerprints, dates = _catalog(
        catalog_path,
        target_id=target_id,
    )
    if archive_workers <= 0:
        raise RuntimeError("archive worker count must be positive")
    catalog_archives = {
        str(row.get("date") or ""): dict(row)
        for row in (catalog.get("source_archives") or ())
        if isinstance(row, dict)
    }
    if archive_receipt_template is None and sorted(catalog_archives) != dates:
        raise RuntimeError(
            "catalog archive evidence does not cover its complete source window"
        )

    match_facts = catalog.get("source_match_facts")
    facts_by_day: dict[str, list[list]] | None = None
    if match_facts is not None:
        facts_by_day = {day: [] for day in dates}
        for raw_fact in match_facts:
            if not isinstance(raw_fact, list) or len(raw_fact) != 4:
                raise RuntimeError("catalog match index row is malformed")
            day, episode_id, seat, fingerprint = raw_fact
            day = str(day)
            if (
                day not in facts_by_day
                or not str(episode_id)
                or int(seat) not in (0, 1)
                or str(fingerprint) not in fingerprints
            ):
                raise RuntimeError("catalog match index row changed")
            facts_by_day[day].append(
                [day, str(episode_id), int(seat), str(fingerprint)]
            )
        observed = dict(catalog.get("observed_by_day") or {})
        if any(
            len(facts_by_day[day]) != int(observed[day])
            for day in dates
        ):
            raise RuntimeError("catalog match index daily counts changed")

    work = [
        (
            day,
            archive_root
            / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            (
                Path(archive_receipt_template.format(date=day))
                if archive_receipt_template is not None
                else None
            ),
            catalog_archives.get(day),
            (facts_by_day.get(day) if facts_by_day is not None else None),
            fingerprints,
            label,
            max_card_id,
            split_seed,
            validation_percent,
        )
        for day in dates
    ]
    with ProcessPoolExecutor(max_workers=archive_workers) as executor:
        daily = list(executor.map(_public_calibration_day, work))

    indices: list[int] = []
    indptr = [0]
    labels: list[int] = []
    weights: list[float] = []
    splits: list[int] = []
    source_days: list[dict] = []
    episodes = 0
    target_seats = 0
    observations = 0
    for result in daily:
        for state, count, split in result["states"]:
            indices.extend(state)
            indptr.append(len(indices))
            labels.append(label)
            weights.append(float(count))
            splits.append(int(split))
        episodes += int(result["matching_episodes"])
        target_seats += int(result["matching_acting_seats"])
        observations += int(result["observations"])
        source_days.append(result["source"])
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.uint8),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(labels), max_card_id + 1),
        dtype=np.uint8,
    )
    return (
        matrix,
        np.asarray(labels, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
        np.asarray(splits, dtype=np.uint8),
        {
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "source_archetype": catalog["source_archetype"],
            "source_window": catalog["source_window"],
            "catalog_observed_acting_seat_games": int(
                catalog["observed_acting_seat_games"]
            ),
            "raw_matching_episodes": episodes,
            "raw_matching_acting_seats": target_seats,
            "public_states": len(labels),
            "observations": observations,
            "input": "cumulative_opponent_public_card_ids_only",
            "label_source": "checksum_bound_exact_public_deck_fingerprint",
            "days": source_days,
            "archive_workers": archive_workers,
        },
    )


def _drop_zero_nnz_rows(
    matrices: list,
    labels: list,
    weights: list,
) -> tuple[list, list, list, dict[str, float | int]]:
    """Remove labeled states that contain no public opponent-card evidence."""

    if not (len(matrices) == len(labels) == len(weights)):
        raise RuntimeError("historical row shard arrays are misaligned")
    filtered_x, filtered_y, filtered_w = [], [], []
    removed_rows = 0
    removed_weight = 0.0
    for matrix, shard_labels, shard_weights in zip(
        matrices, labels, weights, strict=True
    ):
        if (
            matrix.shape[0] != shard_labels.shape[0]
            or matrix.shape[0] != shard_weights.shape[0]
        ):
            raise RuntimeError("historical row shard dimensions are misaligned")
        keep = np.asarray(matrix.getnnz(axis=1)).reshape(-1) > 0
        removed_rows += int((~keep).sum())
        removed_weight += float(np.asarray(shard_weights)[~keep].sum())
        filtered_x.append(matrix[keep])
        filtered_y.append(shard_labels[keep])
        filtered_w.append(shard_weights[keep])
    return (
        filtered_x,
        filtered_y,
        filtered_w,
        {
            "rows": removed_rows,
            "weighted_observations": removed_weight,
        },
    )


def _causal_calibration_rows(
    archive_path: Path,
    *,
    target_id: str,
    label: int,
    max_card_id: int,
    split_seed: int,
    validation_percent: int,
):
    indices: list[int] = []
    indptr = [0]
    labels: list[int] = []
    weights: list[float] = []
    splits: list[int] = []
    episodes = 0
    observations = 0
    with zipfile.ZipFile(archive_path, "r") as stream:
        members = sorted(
            row.filename
            for row in stream.infolist()
            if not row.is_dir() and row.filename.endswith(".json")
        )
        for member in members:
            payload = json.loads(stream.read(member))
            proof = payload.get("router_calibration") or {}
            if (
                proof.get("schema") != "poke_bot.router_calibration_episode/v1"
                or proof.get("target_archetype") != target_id
                or proof.get("causal_public_zones_only") is not True
                or proof.get("hidden_fields_written") is not False
            ):
                raise RuntimeError(
                    f"invalid {target_id} calibration proof: {member}"
                )
            target_seat = int(proof["target_seat"])
            facing_seat = 1 - target_seat
            cumulative: set[int] = set()
            state_counts: dict[tuple[int, ...], int] = {}
            for step in payload.get("steps") or ():
                if not isinstance(step, list) or len(step) != 2:
                    continue
                row = step[facing_seat]
                observation = (
                    row.get("observation") if isinstance(row, dict) else None
                )
                if not isinstance(observation, dict):
                    continue
                cumulative.update(visible_opponent_card_ids(observation))
                state = _safe_card_ids(cumulative, max_card_id)
                state_counts[state] = state_counts.get(state, 0) + 1
                observations += 1
            episode_id = str(payload.get("id") or member)
            split = _episode_split(
                episode_id,
                facing_seat,
                split_seed,
                validation_percent,
            )
            for state, count in state_counts.items():
                indices.extend(state)
                indptr.append(len(indices))
                labels.append(label)
                weights.append(float(count))
                splits.append(split)
            episodes += 1
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.uint8),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(labels), max_card_id + 1),
        dtype=np.uint8,
    )
    return (
        matrix,
        np.asarray(labels, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
        np.asarray(splits, dtype=np.uint8),
        {
            "archive": archive_path.name,
            "archive_digest": sha256_file(archive_path),
            "episodes": episodes,
            "public_states": len(labels),
            "observations": observations,
            "input": "cumulative_opponent_public_card_ids_only",
            "independent_simulation_label": target_id,
        },
    )


def _public_calibration_day(
    task: tuple[
        str,
        Path,
        Path | None,
        dict | None,
        list[list] | None,
        frozenset[str],
        int,
        int,
        int,
        int,
    ],
) -> dict:
    (
        day,
        archive,
        receipt_path,
        catalog_archive,
        match_facts,
        fingerprints,
        _label,
        max_card_id,
        split_seed,
        validation_percent,
    ) = task
    if not archive.is_file():
        raise RuntimeError(f"public replay archive is missing: {archive}")
    receipt_sha256: str | None = None
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_archive = dict(receipt.get("source_archive") or {})
        archive_sha256 = str(receipt_archive.get("sha256") or "")
        if (
            receipt.get("format") != "pokebot-authoritative-visual-day-receipt"
            or receipt.get("source_date") != day
            or Path(str(receipt_archive.get("path") or "")).name != archive.name
            or int(receipt_archive.get("bytes") or -1) != archive.stat().st_size
            or not archive_sha256.startswith("sha256:")
            or len(archive_sha256) != 71
        ):
            raise RuntimeError(
                f"checksum-validated archive receipt changed: {receipt_path}"
            )
        receipt_sha256 = sha256_file(receipt_path)
    else:
        catalog_archive = dict(catalog_archive or {})
        archive_sha256 = str(catalog_archive.get("archive_sha256") or "")
        if (
            catalog_archive.get("date") != day
            or catalog_archive.get("archive") != archive.name
            or not archive_sha256.startswith("sha256:")
            or len(archive_sha256) != 71
            or int(catalog_archive.get("manifest_rows") or 0) <= 0
            or int(catalog_archive.get("json_replays") or 0) <= 0
        ):
            raise RuntimeError(
                f"checksum-bound catalog archive evidence changed: {archive}"
            )
    states: list[tuple[tuple[int, ...], int, int]] = []
    day_episodes = 0
    day_target_seats = 0
    observations = 0
    with zipfile.ZipFile(archive, "r") as stream:
        all_members = sorted(
            row.filename
            for row in stream.infolist()
            if not row.is_dir() and row.filename.endswith(".json")
        )
        indexed_seats: dict[str, set[int]] | None = None
        if match_facts is not None:
            indexed_seats = {}
            for raw_fact in match_facts:
                fact_day, episode_id, seat, fingerprint = raw_fact
                if (
                    str(fact_day) != day
                    or int(seat) not in (0, 1)
                    or str(fingerprint) not in fingerprints
                ):
                    raise RuntimeError("catalog match index row changed")
                indexed_seats.setdefault(str(episode_id), set()).add(int(seat))
            members_by_episode = {
                Path(member).stem: member for member in all_members
            }
            missing_episodes = sorted(set(indexed_seats) - set(members_by_episode))
            if missing_episodes:
                raise RuntimeError(
                    "catalog match index episode is absent from public archive: "
                    + ", ".join(missing_episodes[:3])
                )
            members = [
                members_by_episode[episode_id]
                for episode_id in sorted(indexed_seats)
            ]
        else:
            members = all_members
        for member in members:
            payload = json.loads(stream.read(member))
            decks = extract_setup_decks(payload)
            episode_id = str(payload.get("id") or Path(member).stem)
            if indexed_seats is not None:
                # Catalog match facts are bound to the immutable archive member
                # stem. A replay's internal ``id`` may be a distinct UUID, so
                # it is diagnostic metadata rather than the index key.
                target_indexes = sorted(indexed_seats[Path(member).stem])
                if any(
                    seat >= len(decks)
                    or decks[seat] is None
                    or canonical_deck_sha256(decks[seat]) not in fingerprints
                    for seat in target_indexes
                ):
                    raise RuntimeError(
                        f"catalog match index deck identity changed: {episode_id}"
                    )
            else:
                target_indexes = [
                    seat
                    for seat, cards in enumerate(decks)
                    if cards is not None
                    and canonical_deck_sha256(cards) in fingerprints
                ]
            if not target_indexes:
                continue
            for target_seat in target_indexes:
                facing_seat = 1 - target_seat
                cumulative: set[int] = set()
                state_counts: dict[tuple[int, ...], int] = {}
                for step in payload.get("steps") or ():
                    if not isinstance(step, list) or len(step) != 2:
                        continue
                    row = step[facing_seat]
                    observation = (
                        row.get("observation")
                        if isinstance(row, dict)
                        else None
                    )
                    if not isinstance(observation, dict):
                        continue
                    visible = visible_opponent_card_ids(observation)
                    # Official replay setup frames contain an observation
                    # mapping with no public board.  Treating that empty input
                    # as a positive archetype row would let the private deck
                    # fingerprint label leak into a state where the runtime
                    # must abstain.  Once public evidence exists, retaining the
                    # cumulative state across a temporarily empty frame is
                    # safe and preserves the online router contract.
                    if not visible and not cumulative:
                        continue
                    cumulative.update(visible)
                    state = _safe_card_ids(cumulative, max_card_id)
                    state_counts[state] = state_counts.get(state, 0) + 1
                    observations += 1
                split = _episode_split(
                    episode_id,
                    facing_seat,
                    split_seed,
                    validation_percent,
                )
                states.extend(
                    (state, count, split)
                    for state, count in state_counts.items()
                )
                day_target_seats += 1
            day_episodes += 1
    return {
        "states": states,
        "matching_episodes": day_episodes,
        "matching_acting_seats": day_target_seats,
        "observations": observations,
        "source": {
            "date": day,
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "archive_validation_receipt": (
                str(receipt_path) if receipt_path is not None else None
            ),
            "archive_validation_receipt_sha256": receipt_sha256,
            "archive_validation_source": (
                "authoritative_visual_day_receipt"
                if receipt_path is not None
                else "exact_public_catalog_source_archive_evidence"
            ),
            "episode_selection_source": (
                "checksum_bound_catalog_match_index"
                if match_facts is not None
                else "full_public_archive_scan"
            ),
            "matching_episodes": day_episodes,
            "matching_acting_seats": day_target_seats,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--source-row-root", type=Path, required=True)
    parser.add_argument(
        "--spidops-calibration-archive",
        type=Path,
        help=(
            "Optional independently simulated, proof-bound Spidops archive. "
            "Required when the staged target roster contains Spidops but the "
            "historical source shards predate that route."
        ),
    )
    parser.add_argument(
        "--target-calibration-archive",
        type=Path,
        help=(
            "Optional independently simulated, proof-bound calibration archive "
            "for --target-id."
        ),
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--archive-receipt-template",
        help=(
            "Optional checksum-validated daily receipt path containing {date}. "
            "When omitted, use the exact catalog's complete checksum-bound "
            "source_archives evidence."
        ),
    )
    parser.add_argument("--archive-workers", type=int, default=4)
    parser.add_argument("--public-deck-catalog", type=Path, required=True)
    parser.add_argument(
        "--additional-public-deck-catalog",
        action="append",
        default=[],
        metavar="TARGET_ID=PATH",
        help=(
            "Optional additional checksum-bound exact catalog to retain while "
            "adding the primary target. Repeat once per additional target."
        ),
    )
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-id",
        default=TARGET_ID,
        help=(
            "Exact specialist ID present in both the staged roster and the "
            "public deck catalog."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--max-card-id", type=int, default=4095)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--runtime-precision-floor", type=float, default=0.93)
    args = parser.parse_args()

    source_tree = json.loads(args.source_tree.read_text(encoding="utf-8"))
    if source_tree.get("schema") != SCHEMA:
        raise RuntimeError("source public matchup tree schema changed")
    old_targets = tuple(str(value) for value in source_tree.get("targets") or ())
    target_id = str(args.target_id).strip()
    exact_catalogs = _exact_catalog_bindings(
        target_id,
        args.public_deck_catalog,
        args.additional_public_deck_catalog,
    )
    targets, aliases = _canonical_targets(
        args.roster,
        target_id=target_id,
    )
    missing_exact_targets = sorted(
        bound_id
        for bound_id, _ in exact_catalogs
        if bound_id not in targets
    )
    if missing_exact_targets:
        raise RuntimeError(
            "additional exact-catalog target absent from staged roster: "
            + ", ".join(missing_exact_targets)
        )
    started = time.time()
    train_x, train_y, train_w, val_x, val_y, val_w, sources = (
        _remap_historical_shards(
            args.source_row_root,
            old_targets=old_targets,
            targets=targets,
            aliases=aliases,
        )
    )
    train_x, train_y, train_w, removed_train_empty = _drop_zero_nnz_rows(
        train_x, train_y, train_w
    )
    val_x, val_y, val_w, removed_val_empty = _drop_zero_nnz_rows(
        val_x, val_y, val_w
    )
    if args.spidops_calibration_archive is not None:
        spidops_id = "team-rockets-spidops"
        if spidops_id not in targets:
            raise RuntimeError(
                "Spidops calibration supplied but Spidops is absent from roster"
            )
        (
            spidops_matrix,
            spidops_labels,
            spidops_weights,
            spidops_splits,
            spidops_source,
        ) = _causal_calibration_rows(
            args.spidops_calibration_archive,
            target_id=spidops_id,
            label=targets.index(spidops_id),
            max_card_id=args.max_card_id,
            split_seed=args.seed,
            validation_percent=args.validation_percent,
        )
        spidops_nonempty = (
            np.asarray(spidops_matrix.getnnz(axis=1)).reshape(-1) > 0
        )
        removed_spidops_rows = int((~spidops_nonempty).sum())
        removed_spidops_weight = float(
            spidops_weights[~spidops_nonempty].sum()
        )
        spidops_matrix = spidops_matrix[spidops_nonempty]
        spidops_labels = spidops_labels[spidops_nonempty]
        spidops_weights = spidops_weights[spidops_nonempty]
        spidops_splits = spidops_splits[spidops_nonempty]
        if (
            spidops_labels.size == 0
            or not np.any(spidops_splits == 0)
            or not np.any(spidops_splits == 1)
        ):
            raise RuntimeError("Spidops public calibration split is incomplete")
        spidops_train = spidops_splits == 0
        spidops_val = spidops_splits == 1
        train_x.append(spidops_matrix[spidops_train])
        train_y.append(spidops_labels[spidops_train])
        train_w.append(spidops_weights[spidops_train])
        val_x.append(spidops_matrix[spidops_val])
        val_y.append(spidops_labels[spidops_val])
        val_w.append(spidops_weights[spidops_val])
        spidops_source["empty_public_states_removed"] = {
            "rows": removed_spidops_rows,
            "weighted_observations": removed_spidops_weight,
        }
        sources.append(spidops_source)
    for exact_target_id, exact_catalog_path in exact_catalogs:
        matrix, labels, weights, splits, calibration_source = (
            _public_calibration_rows(
                args.archive_root,
                archive_receipt_template=args.archive_receipt_template,
                archive_workers=args.archive_workers,
                catalog_path=exact_catalog_path,
                label=targets.index(exact_target_id),
                max_card_id=args.max_card_id,
                split_seed=args.seed,
                validation_percent=args.validation_percent,
                target_id=exact_target_id,
            )
        )
        if (
            labels.size == 0
            or not np.any(splits == 0)
            or not np.any(splits == 1)
        ):
            raise RuntimeError(
                f"{exact_target_id} public calibration split is incomplete"
            )
        train_mask = splits == 0
        val_mask = splits == 1
        train_x.append(matrix[train_mask])
        train_y.append(labels[train_mask])
        train_w.append(weights[train_mask])
        val_x.append(matrix[val_mask])
        val_y.append(labels[val_mask])
        val_w.append(weights[val_mask])
        sources.append(calibration_source)
    if args.target_calibration_archive is not None:
        (
            target_matrix,
            target_labels,
            target_weights,
            target_splits,
            target_source,
        ) = _causal_calibration_rows(
            args.target_calibration_archive,
            target_id=target_id,
            label=targets.index(target_id),
            max_card_id=args.max_card_id,
            split_seed=args.seed,
            validation_percent=args.validation_percent,
        )
        target_nonempty = (
            np.asarray(target_matrix.getnnz(axis=1)).reshape(-1) > 0
        )
        removed_target_rows = int((~target_nonempty).sum())
        removed_target_weight = float(
            target_weights[~target_nonempty].sum()
        )
        target_matrix = target_matrix[target_nonempty]
        target_labels = target_labels[target_nonempty]
        target_weights = target_weights[target_nonempty]
        target_splits = target_splits[target_nonempty]
        if (
            target_labels.size == 0
            or not np.any(target_splits == 0)
            or not np.any(target_splits == 1)
        ):
            raise RuntimeError(
                f"{target_id} independent calibration split is incomplete"
            )
        target_train = target_splits == 0
        target_val = target_splits == 1
        train_x.append(target_matrix[target_train])
        train_y.append(target_labels[target_train])
        train_w.append(target_weights[target_train])
        val_x.append(target_matrix[target_val])
        val_y.append(target_labels[target_val])
        val_w.append(target_weights[target_val])
        target_source["empty_public_states_removed"] = {
            "rows": removed_target_rows,
            "weighted_observations": removed_target_weight,
        }
        sources.append(target_source)

    x_train = sparse.vstack(train_x, format="csr")
    y_train = np.concatenate(train_y)
    w_train = np.concatenate(train_w)
    x_val = sparse.vstack(val_x, format="csr")
    y_val = np.concatenate(val_y)
    w_val = np.concatenate(val_w)
    class_names = targets + (UNKNOWN,)
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    prediction = model.predict(x_val)
    probabilities = _expanded_predict_proba(model, x_val, len(class_names))
    validation = _weighted_metrics(
        y_val,
        prediction,
        w_val,
        class_names,
    )
    runtime_calibration = _calibrate_runtime_thresholds(
        y_val,
        prediction,
        probabilities,
        w_val,
        class_names,
        precision_floor=args.runtime_precision_floor,
    )
    payload = {
        "schema": SCHEMA,
        "created_at_unix": time.time(),
        "runtime_enabled": False,
        "activation_status": "validation_required",
        "input_contract": {
            "features": "cumulative binary opponent-public card IDs",
            "public_zones": [
                "active",
                "bench",
                "discard",
                "attached_tools",
                "attached_energy",
                "public_pre_evolution",
            ],
            "uses_our_deck": False,
            "uses_our_archetype": False,
            "prohibited": [
                "opponent_hand",
                "opponent_deck",
                "opponent_prizes",
                "agent_identity",
                "recorded_archetype",
                "future_observations",
            ],
            "oracle_labels_are_inputs": False,
            "unknown_falls_back_to_base_policy": True,
            "empty_public_state_exact_bypass": True,
            "one_route_maximum_per_observation": True,
        },
        "targets": list(targets),
        "unknown_class": UNKNOWN,
        "prediction_contract": {
            "route_output_width": len(targets),
            "route_class_names": list(targets),
            "unknown_is_separate_abstention": True,
            "unknown_class_index": len(targets),
            "adapter_count": len(targets),
        },
        "split": {
            "kind": "episode-and-seat-disjoint-deterministic-hash",
            "seed": args.seed,
            "validation_percent": args.validation_percent,
        },
        "training": {
            "criterion": "entropy",
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "train_public_states": int(x_train.shape[0]),
            "validation_public_states": int(x_val.shape[0]),
            "train_weighted_observations": float(w_train.sum()),
            "validation_weighted_observations": float(w_val.sum()),
            "elapsed_seconds": time.time() - started,
            "historical_label_remap": aliases,
            "historical_empty_public_states_removed": {
                "training": removed_train_empty,
                "validation": removed_val_empty,
            },
        },
        "sources": sources,
        "validation": validation,
        "runtime_calibration": runtime_calibration,
        "calibration_contract": {
            "probability_columns": "expanded_to_canonical_class_indexes",
            "canonical_class_count": len(class_names),
            "fitted_class_indexes": [int(value) for value in model.classes_],
            "missing_fitted_classes_receive_zero_probability": True,
        },
        "tree": _tree_payload(model, class_names),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "public-matchup-tree.json"
    if artifact.exists():
        raise FileExistsError(f"immutable output already exists: {artifact}")
    descriptor, name = tempfile.mkstemp(
        prefix=".public-matchup-tree.",
        dir=output_dir,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, artifact)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema": "poke_bot.public_matchup_decision_tree_receipt/v1",
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "runtime_enabled": False,
        "validation": validation,
        "runtime_calibration": runtime_calibration,
    }
    (output_dir / "PUBLIC_MATCHUP_TREE_READY.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
