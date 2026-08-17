#!/usr/bin/env python3
"""Rebuild row-aligned matchup oracle indexes from pinned raw archives.

The input must already be an Alakazam-only temporal feature manifest whose
opponents have configured adapter routes.  Raw archive members provide the
full opponent deck, agent identity, classifier method, and member digest; no
route is inferred from compact tensors alone.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

from poke_bot.authoritative_visual_trace import _json_digest
from poke_bot.feature_shards import MANIFEST_FORMAT, iter_feature_shard
from poke_bot.ladder_replay import LadderReplayClassifier
from poke_bot.matchup_adapter_activation import normalize_matchup_identity
from poke_bot.matchup_adapters import UNKNOWN_ROUTE, route_for_archetype
from poke_bot.public_matchup_router import (
    PublicMatchupDecisionTree,
    visible_opponent_card_ids,
)
from poke_bot.pure_rl.matchup_adapter_corpus import (
    ORACLE_INDEX_ROW_SCHEMA,
    ORACLE_MANIFEST_SCHEMA,
    PACKAGE_REGISTRY_SCHEMA,
    full_deck_digest,
    sha256_file,
    write_oracle_index,
)
from poke_bot.replay_import import _agent_names, episode_id_of


CLASSIFIER_METHODS = frozenset(
    {
        "representative_exact",
        "registered_signature",
        "artifact_signature",
        "derived_primary_ace",
    }
)


def _bytes_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _bytes_digest(raw)


def opponent_identity(agent_name: str, deck_digest: str) -> tuple[str, str]:
    """Return a collision-resistant archived-agent/deck identity pair."""

    agent = normalize_matchup_identity(agent_name) or "unknown-agent"
    content = _canonical_digest(
        {
            "kind": "official-ladder-archive-agent-and-full-deck/v1",
            "agent": agent,
            "deck_digest": deck_digest,
        }
    )
    return f"expert-ladder:{content[7:31]}", content


def _feature_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = pickle.load(stream)
    if not isinstance(header, dict):
        raise ValueError(f"invalid feature header: {path}")
    return header


def _archive_rows(
    archive: Path,
    classifier: LadderReplayClassifier,
    *,
    archive_digest: str,
    public_tree: PublicMatchupDecisionTree | None = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, str]]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(archive, "r") as stream:
        members = sorted(
            info.filename
            for info in stream.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        )
        for member in members:
            raw = stream.read(member)
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            decks, labels = classifier.classify_episode(payload)
            if len(decks) != 2 or len(labels) != 2:
                continue
            episode_id = episode_id_of(payload)
            agents = _agent_names(payload)
            for seat in (0, 1):
                opponent = 1 - seat
                opponent_label = labels[opponent]
                opponent_deck = decks[opponent]
                if (
                    labels[seat].deck_id != "alakazam"
                    or opponent_deck is None
                    or len(opponent_deck) != 60
                    or route_for_archetype(opponent_label.deck_id) == UNKNOWN_ROUTE
                    or opponent_label.method not in CLASSIFIER_METHODS
                ):
                    continue
                deck_digest = full_deck_digest(opponent_deck)
                opponent_id, content_digest = opponent_identity(
                    agents[opponent], deck_digest
                )
                prior_identity = identities.setdefault(
                    opponent_id,
                    {
                        "content_digest": content_digest,
                        "agent_name": str(agents[opponent]),
                    },
                )
                if prior_identity["content_digest"] != content_digest:
                    raise ValueError("archived opponent identity hash collision")
                key = (episode_id, seat)
                entry = {
                    "schema": ORACLE_INDEX_ROW_SCHEMA,
                    "episode_id": episode_id,
                    "seat": seat,
                    "acting_archetype": "alakazam",
                    "opponent_archetype": str(opponent_label.deck_id),
                    "opponent_id": opponent_id,
                    "opponent_content_digest": content_digest,
                    "opponent_deck_digest": deck_digest,
                    "source_archive_digest": archive_digest,
                    "source_member_digest": _bytes_digest(raw),
                    "classifier_method": str(opponent_label.method),
                    "formal_eval": False,
                    "training_eligible": True,
                }
                if public_tree is not None:
                    cumulative: set[int] = set()
                    predictions_by_env_step: dict[str, dict[str, Any]] = {}
                    for env_step, step in enumerate(payload.get("steps") or ()):
                        if not isinstance(step, list) or len(step) <= seat:
                            continue
                        seat_row = step[seat]
                        if not isinstance(seat_row, dict):
                            continue
                        observation = seat_row.get("observation") or {}
                        select = observation.get("select")
                        current = observation.get("current")
                        action = seat_row.get("action")
                        if select is None or current is None or action is None:
                            continue
                        if int(current.get("yourIndex", seat)) != seat:
                            continue
                        # The immutable feature corpus is the authority for which
                        # historical decision rows survived conversion.  Record
                        # causal public evidence by environment step here, then
                        # align it to those exact rows in ``aligned_rows`` below.
                        # This remains valid across converter revisions that
                        # tightened action-frame filtering after the corpus was
                        # built; ordinal counts are not a stable join key.
                        cumulative.update(visible_opponent_card_ids(observation))
                        prediction = public_tree.predict_card_ids(tuple(cumulative))
                        predictions_by_env_step[str(env_step)] = {
                            "route": int(prediction.route),
                            "confidence": float(prediction.confidence),
                        }
                    entry["_public_predictions_by_env_step"] = predictions_by_env_step
                previous = rows.setdefault(key, entry)
                if previous != entry:
                    raise ValueError(f"ambiguous duplicate episode/seat identity: {key}")
    return rows, identities


def _build_source_oracle_shard(
    task: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Build one independent day shard in a spawn/fork-safe worker."""

    source_row = dict(task["source_row"])
    feature = (
        Path(str(task["feature_parent"])) / str(source_row["path"])
    ).resolve()
    feature_digest = sha256_file(feature)
    if feature_digest != str(source_row.get("sha256") or ""):
        raise ValueError(f"feature digest mismatch: {feature}")
    classifier = LadderReplayClassifier.from_paths(
        Path(str(task["mix"])),
        Path(str(task["representatives"])),
        card_csv=(
            Path(str(task["card_csv"])) if task.get("card_csv") is not None else None
        ),
        additive_registered_ids=tuple(task.get("additive_archetypes") or ()),
    )
    classifier_digest = str(task["classifier_digest"])
    if _json_digest(classifier.contract) != classifier_digest:
        raise ValueError("worker classifier digest changed")
    public_tree = (
        PublicMatchupDecisionTree.from_path(
            Path(str(task["public_tree_path"])), require_runtime_enabled=False
        )
        if task.get("public_tree_path") is not None
        else None
    )
    if public_tree is not None and public_tree.digest != str(
        task.get("public_tree_digest") or ""
    ):
        raise ValueError("worker public-router digest changed")
    header = _feature_header(feature)
    if (
        header.get("required_archetype") != "alakazam"
        or header.get("classifier_sha256") != classifier_digest
        or header.get("opponent_routes_only") is not True
    ):
        raise ValueError("filtered feature lost oracle provenance/selection")
    dates = list(header.get("source_dates") or [])
    if len(dates) != 1:
        raise ValueError("oracle builder requires one source date per shard")
    archive = Path(str(task["archive_dir"])) / str(
        header.get("source_archive") or ""
    )
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_digest = sha256_file(archive)
    if archive_digest != str(header.get("source_archive_sha256") or ""):
        raise ValueError(f"source archive digest mismatch: {archive}")
    available, identities = _archive_rows(
        archive,
        classifier,
        archive_digest=archive_digest,
        public_tree=public_tree,
    )

    def aligned_rows() -> Iterator[dict[str, Any]]:
        count = 0
        for sequence in iter_feature_shard(feature):
            key = (str(sequence.episode_id), int(sequence.seat))
            entry = available.get(key)
            if entry is None:
                raise ValueError(f"raw oracle lacks feature row: {key}")
            if (
                normalize_matchup_identity(sequence.archetype) != "alakazam"
                or normalize_matchup_identity(sequence.opp_archetype)
                != entry["opponent_archetype"]
            ):
                raise ValueError(f"raw/feature matchup mismatch: {key}")
            entry = dict(entry)
            if public_tree is not None:
                raw_predictions = dict(
                    entry.pop("_public_predictions_by_env_step", {}) or {}
                )
                expected_route = route_for_archetype(entry["opponent_archetype"])
                consecutive = 0
                active_indexes: list[int] = []
                for index, decision in enumerate(sequence.decisions):
                    prediction = raw_predictions.get(str(int(decision.env_step)))
                    if prediction is None:
                        raise ValueError(
                            "raw oracle lacks causal public observation for feature "
                            f"row: episode_id={sequence.episode_id!r} "
                            f"seat={sequence.seat} env_step={decision.env_step}"
                        )
                    agrees = bool(
                        int(prediction["route"]) == expected_route
                        and float(prediction["confidence"]) >= 1.0 - 1e-12
                    )
                    consecutive = consecutive + 1 if agrees else 0
                    if consecutive >= 2:
                        active_indexes.append(index)
                segments: list[list[int]] = []
                for index in active_indexes:
                    if segments and segments[-1][1] == index:
                        segments[-1][1] = index + 1
                    else:
                        segments.append([index, index + 1])
                entry.update(
                    {
                        "public_router_digest": public_tree.digest,
                        "public_route_segments": segments,
                        "public_decisions": len(sequence.decisions),
                        "public_confidence_threshold": 1.0,
                        "public_consecutive_required": 2,
                    }
                )
            count += 1
            yield entry
        expected = int((source_row.get("stats") or {}).get("records_kept", -1))
        if count != expected:
            raise ValueError(
                f"feature row count mismatch: expected={expected} actual={count}"
            )

    index_path = Path(str(task["output_dir"])) / f"{feature.stem}.oracle.jsonl"
    metadata = write_oracle_index(
        index_path,
        source_feature_digest=feature_digest,
        classifier_digest=classifier_digest,
        rows=aligned_rows(),
    )
    return (
        {
            "source_feature_digest": feature_digest,
            "source_date": dates[0],
            "path": index_path.name,
            "sha256": metadata["sha256"],
            "records": int(metadata["rows"]),
        },
        identities,
    )


def build_oracle(
    feature_manifest: Path,
    archive_dir: Path,
    output_dir: Path,
    *,
    mix: Path,
    representatives: Path,
    card_csv: Path | None = None,
    additive_archetypes: tuple[str, ...] = (),
    public_tree_path: Path | None = None,
    jobs: int = 1,
) -> Path:
    feature_manifest = feature_manifest.resolve()
    archive_dir = archive_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(feature_manifest.read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("source is not a feature manifest")
    selection = dict(payload.get("selection") or {})
    if (
        selection.get("value") != "alakazam"
        or selection.get("opponent_routes_only") is not True
    ):
        raise ValueError("oracle source must be Alakazam and opponent-route filtered")
    classifier = LadderReplayClassifier.from_paths(
        mix,
        representatives,
        card_csv=card_csv,
        additive_registered_ids=additive_archetypes,
    )
    classifier_digest = _json_digest(classifier.contract)
    public_tree = (
        PublicMatchupDecisionTree.from_path(
            public_tree_path, require_runtime_enabled=False
        )
        if public_tree_path is not None
        else None
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial.", dir=output_dir.parent)
    )
    registry: dict[str, dict[str, str]] = {}
    oracle_shards: list[dict[str, Any]] = []
    try:
        source_rows = list(payload.get("shards") or [])
        if not source_rows:
            raise ValueError("feature manifest has no shards")
        worker_count = max(1, min(int(jobs), len(source_rows)))
        tasks = [
            {
                "source_row": dict(source_row),
                "feature_parent": str(feature_manifest.parent),
                "archive_dir": str(archive_dir),
                "output_dir": str(temporary),
                "mix": str(mix),
                "representatives": str(representatives),
                "card_csv": str(card_csv) if card_csv is not None else None,
                "additive_archetypes": tuple(additive_archetypes),
                "classifier_digest": classifier_digest,
                "public_tree_path": (
                    str(public_tree_path) if public_tree_path is not None else None
                ),
                "public_tree_digest": (
                    public_tree.digest if public_tree is not None else None
                ),
            }
            for source_row in source_rows
        ]
        if worker_count == 1:
            results = map(_build_source_oracle_shard, tasks)
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count
            )
            results = executor.map(_build_source_oracle_shard, tasks)
        try:
            for oracle_shard, identities in results:
                oracle_shards.append(oracle_shard)
                print(
                    json.dumps(
                        {
                            "stage": "causal_oracle_shard",
                            **oracle_shard,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                for opponent_id, identity in identities.items():
                    previous = registry.setdefault(opponent_id, identity)
                    if previous["content_digest"] != identity["content_digest"]:
                        raise ValueError("opponent registry identity conflict")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        package_registry = temporary / "opponent-registry.json"
        package_registry.write_text(
            json.dumps(
                {
                    "schema": PACKAGE_REGISTRY_SCHEMA,
                    "identity_kind": "official-ladder-archive-agent-and-full-deck/v1",
                    "packages": [
                        {
                            "opponent_id": opponent_id,
                            "content_digest": row["content_digest"],
                            "aliases": [],
                            "agent_name": row["agent_name"],
                        }
                        for opponent_id, row in sorted(registry.items())
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        oracle_manifest = temporary / "oracle-manifest.json"
        oracle_manifest.write_text(
            json.dumps(
                {
                    "schema": ORACLE_MANIFEST_SCHEMA,
                    "identity_kind": "official-ladder-archive-agent-and-full-deck/v1",
                    "source_feature_manifest_digest": sha256_file(feature_manifest),
                    "classifier_digest": classifier_digest,
                    "package_registry_digest": sha256_file(package_registry),
                    "public_router_digest": (
                        public_tree.digest if public_tree is not None else None
                    ),
                    "shards": sorted(
                        oracle_shards, key=lambda row: str(row["source_date"])
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return output_dir / oracle_manifest.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    parser.add_argument("--card-csv", type=Path, default=None)
    parser.add_argument("--additive-archetype", action="append", default=[])
    parser.add_argument("--public-tree", type=Path, default=None)
    parser.add_argument(
        "--jobs", type=int, default=max(1, min(6, os.cpu_count() or 1))
    )
    args = parser.parse_args()
    manifest = build_oracle(
        args.feature_manifest,
        args.archive_dir,
        args.output_dir,
        mix=args.mix,
        representatives=args.representatives,
        card_csv=args.card_csv,
        additive_archetypes=tuple(args.additive_archetype),
        public_tree_path=args.public_tree,
        jobs=args.jobs,
    )
    print(json.dumps(json.loads(manifest.read_text()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
