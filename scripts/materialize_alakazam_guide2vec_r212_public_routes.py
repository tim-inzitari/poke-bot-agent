#!/usr/bin/env python3
"""Materialize sealed r195 public-route sidecars for Guide2Vec r212.

There are two bounded producer modes:

``--feature-shard``
    Replays one locally available protected feature shard and its raw daily ZIP.

``--raw-classifier-query``
    Rebuilds the selected Alakazam source identities from the raw daily ZIP,
    exact pinned ladder classifier, and authoritative visual trace.  It is for
    an archive-owning host (Elmo) which deliberately does not need a 0.9 GiB
    feature shard.  Each output row includes an f32 board/action alignment
    digest that Blackwell rechecks against its protected compact sequence.

``--write-heldout-manifest``
    Verifies two transferred day sidecars and writes one no-clobber portable
    manifest beside them.  The trainer resolves each relative day path against
    that manifest's directory and requires the declared file SHA-256 and raw
    archive binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.guide2vec_public_routes import (
    ALIGNMENT_CONTRACT,
    ROUTE_ALGORITHM,
    ROUTE_RECONSTRUCTION_SCHEMA,
    ROUTE_SIDECAR_FOOTER_SCHEMA,
    ROUTE_SIDECAR_FORMAT,
    ROUTE_SIDECAR_HEADER_SCHEMA,
    ROUTE_SIDECAR_ROW_SCHEMA,
    TURN_ORDER_SHORT_CIRCUIT_CONTRACT,
    PublicRouteReconstructionError,
    ProducerCodeBinding,
    RuntimeCodeBinding,
    UNKNOWN_ROUTE,
    bind_sidecar_producer_code,
    compact_alignment_sha256,
    materialize_day_sidecar,
    reconstruct_public_routes_from_raw_member,
    verify_imported_runtime_public_router,
)
from poke_bot.public_matchup_router import PublicMatchupDecisionTree


R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_ENTRYPOINT_MEMBER = "./main.py"
R195_ENTRYPOINT_SHA256 = (
    "sha256:02b6ea8b565e0bb66aed14719cc80636c388742d3af40408a3eb458baa4bd8d7"
)
R195_ROUTER_MEMBER = "./poke_bot/public_matchup_router.py"
R195_ROUTER_SHA256 = (
    "sha256:98b1f6cc871ea56f295aaed9c1fbaad46fbe64036f1ae12d2de31f0f787c4a6a"
)
R195_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
CLASSIFIER_SHA256 = (
    "sha256:04bd15eac4fe1ce2cd5010f198d89201884d21f2968f04cf8ee66e49773f8011"
)
SIDECAR_MANIFEST_SCHEMA = (
    "poke_bot.guide2vec_r212_r195_public_route_sidecar_manifest/v1"
)
RAW_QUERY_SCHEMA = "poke_bot.guide2vec_r212_r195_raw_classifier_query/v1"
HELDOUT_DATES = ("2026-07-22", "2026-07-23")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 71 or not text.startswith("sha256:"):
        raise PublicRouteReconstructionError(f"{label} is not an exact SHA-256")
    try:
        int(text.split(":", 1)[1], 16)
    except ValueError as exc:
        raise PublicRouteReconstructionError(f"{label} is not an exact SHA-256") from exc
    return text


def _runtime_code(args: argparse.Namespace) -> RuntimeCodeBinding:
    bundle = args.submission_bundle.expanduser().resolve()
    bundle_sha = _require_sha256(args.submission_bundle_sha256, label="r195 bundle")
    if not bundle.is_file() or _sha256_file(bundle) != bundle_sha:
        raise PublicRouteReconstructionError("exact r195 submission bundle is unavailable")
    # The module's Raw resolver checks both tar members.  Query mode must
    # independently do the same with a small local tar reader rather than bind
    # a workspace/preimage lookalike.
    import tarfile

    member_hashes = {
        args.submission_entrypoint_member: _require_sha256(
            args.submission_entrypoint_sha256, label="r195 main.py"
        ),
        args.public_matchup_router_member: _require_sha256(
            args.public_matchup_router_sha256, label="r195 router"
        ),
    }
    with tarfile.open(bundle, "r:gz") as archive:
        for member_name, expected in member_hashes.items():
            found = [entry for entry in archive.getmembers() if entry.name == member_name]
            if len(found) != 1 or not found[0].isfile():
                raise PublicRouteReconstructionError(
                    f"exact r195 bundle lacks member {member_name!r}"
                )
            handle = archive.extractfile(found[0])
            if handle is None or "sha256:" + hashlib.sha256(handle.read()).hexdigest() != expected:
                raise PublicRouteReconstructionError(
                    f"exact r195 bundle member digest differs: {member_name}"
                )
    verify_imported_runtime_public_router(
        expected_sha256=args.public_matchup_router_sha256
    )
    return RuntimeCodeBinding(
        submission_bundle_sha256=bundle_sha,
        submission_entrypoint_member=args.submission_entrypoint_member,
        submission_entrypoint_sha256=args.submission_entrypoint_sha256,
        public_matchup_router_member=args.public_matchup_router_member,
        public_matchup_router_sha256=args.public_matchup_router_sha256,
    )


def _load_tree(args: argparse.Namespace) -> tuple[PublicMatchupDecisionTree, str, frozenset[int]]:
    path = args.matchup_tree.expanduser().resolve()
    expected = _require_sha256(args.matchup_tree_sha256, label="r195 matchup tree")
    if not path.is_file() or _sha256_file(path) != expected:
        raise PublicRouteReconstructionError("exact r195 public matchup tree is unavailable")
    tree = PublicMatchupDecisionTree.from_path(path, require_runtime_enabled=True)
    if tree.digest != expected:
        raise PublicRouteReconstructionError("loaded r195 matchup tree digest drifted")
    if int(tree.unknown_route) != UNKNOWN_ROUTE:
        raise PublicRouteReconstructionError("r195 matchup tree bypass route drifted")
    allowed = frozenset(
        int(tree.route_physical_slots[tree.targets.index(target)])
        for target in tree.runtime_accepted_archetype_ids
    )
    supplied = frozenset(args.allowed_physical_slot)
    if supplied != allowed:
        raise PublicRouteReconstructionError(
            "declared V6 physical slots differ from exact r195 runtime tree"
        )
    return tree, expected, allowed


def _archive_binding(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    day = str(args.date)
    archive = args.archive.expanduser().resolve()
    expected_name = f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    if archive.name != expected_name or not archive.is_file():
        raise PublicRouteReconstructionError("raw archive name/path is not the exact source day")
    digest = _sha256_file(archive)
    expected = _require_sha256(args.raw_archive_sha256, label="raw archive")
    if digest != expected:
        raise PublicRouteReconstructionError("raw archive digest differs from declared source")
    return archive, {
        "source_date": day,
        "archive_name": archive.name,
        "sha256": digest,
        "bytes": int(archive.stat().st_size),
    }


def _no_clobber_sidecar(
    *, header: Mapping[str, Any], rows: list[Mapping[str, Any]], projection: Mapping[str, Any], output_dir: Path
) -> tuple[Path, str]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / (
        f".r212-public-routes-{header['source_date']}.partial.{os.getpid()}.{time.time_ns()}.jsonl"
    )
    try:
        row_hasher = hashlib.sha256()
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(dict(header)).decode("utf-8"))
            for row in rows:
                encoded = _canonical_json(dict(row))
                handle.write(encoded.decode("utf-8"))
                row_hasher.update(encoded)
            footer = {
                "schema": ROUTE_SIDECAR_FOOTER_SCHEMA,
                "rows": len(rows),
                "rows_sha256": "sha256:" + row_hasher.hexdigest(),
                "projection": dict(projection),
            }
            handle.write(_canonical_json(footer).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_file(partial)
        final = destination / (
            f"r212-public-routes-{header['source_date']}-{digest.split(':', 1)[1]}.jsonl"
        )
        try:
            os.link(partial, final)
        except FileExistsError:
            if _sha256_file(final) != digest:
                raise PublicRouteReconstructionError("sidecar content-address collision")
        return final, digest
    finally:
        partial.unlink(missing_ok=True)


def _raw_query_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    # The feature-shard producer must remain runnable on an isolated archive
    # host with no classifier/card-data assets.  Keep this unsupported,
    # fail-closed query-only dependency surface lazy.
    from poke_bot import paths
    from poke_bot.authoritative_visual_trace import (
        VISUAL_TRACE_SCHEMA,
        VisualTraceError,
        _guide_targets_enabled,
        _json_digest,
        _record_to_temporal_sequence,
        convert_visual_episode,
    )
    from poke_bot.feature_shards import compact_temporal_expert_sequence
    from poke_bot.ladder_replay import LadderReplayClassifier

    archive, raw_archive = _archive_binding(args)
    producer_code = bind_sidecar_producer_code(materializer_cli_path=Path(__file__))
    runtime_code = _runtime_code(args)
    tree, tree_sha, allowed = _load_tree(args)
    card_csv = (
        args.card_csv.expanduser().resolve()
        if args.card_csv is not None
        else paths.en_card_data_path()
    )
    classifier = LadderReplayClassifier.from_paths(
        args.mix.expanduser().resolve(),
        args.representatives.expanduser().resolve(),
        card_csv=card_csv,
        additive_registered_ids=("alakazam",),
    )
    classifier_sha = _json_digest(classifier.contract)
    expected_classifier = _require_sha256(args.classifier_sha256, label="classifier")
    if classifier_sha != expected_classifier:
        raise PublicRouteReconstructionError(
            "exact raw classifier contract differs from the r212 source classifier"
        )
    max_context = int(args.max_context)
    if max_context != 320:
        raise PublicRouteReconstructionError("r212 raw query must use exact max_context=320")
    source_query = {
        "schema": RAW_QUERY_SCHEMA,
        "classifier_sha256": classifier_sha,
        "mix_sha256": _sha256_file(args.mix.expanduser().resolve()),
        "representatives_sha256": _sha256_file(args.representatives.expanduser().resolve()),
        "card_csv_sha256": _sha256_file(card_csv),
        "required_archetype": "alakazam",
        "max_context": max_context,
        "visual_trace_schema": VISUAL_TRACE_SCHEMA,
        "expected_records": int(args.expected_records),
        "expected_decisions": int(args.expected_decisions),
        "selection_order": "zip_member_lexicographic_then_convert_visual_episode_seat_order/v1",
    }
    feature_shard_sha = _require_sha256(
        args.feature_shard_sha256, label="declared protected feature shard"
    )
    header: dict[str, Any] = {
        "schema": ROUTE_SIDECAR_HEADER_SCHEMA,
        "format": ROUTE_SIDECAR_FORMAT,
        "source_date": str(args.date),
        "source_feature_shard_sha256": feature_shard_sha,
        "raw_archive": raw_archive,
        "runtime_public_tree_sha256": tree_sha,
        "runtime_code": runtime_code.as_dict(),
        "producer_code": producer_code.binding.as_dict(),
        "allowed_physical_slots": sorted(allowed),
        "algorithm": ROUTE_ALGORITHM,
        "alignment_contract": ALIGNMENT_CONTRACT,
        "compact_source_routes_ignored": True,
        "oracle_route_used": False,
        "source_query": source_query,
    }
    rows: list[dict[str, Any]] = []
    active_observations = 0
    turn_order_short_circuits = 0
    game_resets = 0
    routed = 0
    bypassed = 0
    rejected = 0
    source = f"pokemon-tcg-ai-battle-episodes-{args.date}"
    with zipfile.ZipFile(archive, "r") as zip_handle:
        members = sorted(
            info.filename
            for info in zip_handle.infolist()
            if not info.is_dir() and info.filename.endswith(".json")
        )
        for member in members:
            raw = zip_handle.read(member)
            try:
                payload = json.loads(raw)
                converted = convert_visual_episode(
                    payload,
                    classifier,
                    source=source,
                    required_archetype="alakazam",
                )
                # Re-run the exact sealed materializer's compact conversion;
                # this preserves its rejected-member predicate as well as the
                # final last-320 selection/order and frozen sparse vectors.
                compact_records = []
                with _guide_targets_enabled("alakazam"):
                    for record in converted.records:
                        sequence, _details = _record_to_temporal_sequence(
                            record, max_context=max_context
                        )
                        compact_temporal_expert_sequence(sequence)
                        compact_records.append((record, sequence))
            except VisualTraceError:
                # Match the sealed authoritative materializer's rejected
                # member behavior. Exact selected totals below make this a
                # fail-closed source-query predicate rather than a silent drop.
                rejected += 1
                continue
            for record, sequence in compact_records:
                decisions = list(sequence.decisions)
                env_steps = [decision.env_step for decision in decisions]
                actions = [list(decision.action) for decision in decisions]
                if any(
                    type(step) is not int
                    or not action
                    or any(type(item) is not int for item in action)
                    for step, action in zip(env_steps, actions)
                ):
                    raise PublicRouteReconstructionError("raw classifier record action/env-step is malformed")
                if any(decision.action_token is None for decision in decisions):
                    raise PublicRouteReconstructionError(
                        "raw classifier compact sequence lacks shifted action token"
                    )
                alignment = compact_alignment_sha256(
                    env_steps=env_steps,
                    boards=[decision.board for decision in decisions],
                    action_tokens=[decision.action_token for decision in decisions],
                )
                raw_boards: list[Any] = [None] * len(decisions)
                raw_tokens: list[Any] = [None] * len(decisions)

                def validate_target(index: int, observation: Mapping[str, Any], raw_action: list[int]) -> None:
                    if raw_action != actions[index]:
                        raise PublicRouteReconstructionError(
                            "raw classifier record shifted action differs from raw episode"
                        )
                    # The compact materializer has already featurized this
                    # exact masked trace. Rebuild no alternate target here;
                    # the sidecar consumer verifies this vector digest against
                    # its protected compact sequence before use.
                    raw_boards[index] = decisions[index].board
                    raw_tokens[index] = decisions[index].action_token

                result = reconstruct_public_routes_from_raw_member(
                    raw_member=raw,
                    episode_id=str(record.get("episode_id") or ""),
                    seat=record.get("seat"),
                    env_steps=env_steps,
                    compact_alignment_sha256=alignment,
                    tree=tree,
                    allowed_physical_slots=allowed,
                    target_validator=validate_target,
                )
                if any(value is None for value in raw_boards + raw_tokens) or compact_alignment_sha256(
                    env_steps=env_steps, boards=raw_boards, action_tokens=raw_tokens
                ) != alignment:
                    raise PublicRouteReconstructionError(
                        "raw classifier board/action feature projection differs from its compact digest"
                    )
                row = result.as_sidecar_row()
                rows.append(row)
                active_observations += result.active_observations
                turn_order_short_circuits += result.turn_order_short_circuits
                game_resets += result.game_resets
                routed += sum(int(route != UNKNOWN_ROUTE) for route in result.routes)
                bypassed += sum(int(route == UNKNOWN_ROUTE) for route in result.routes)
    if len(rows) != int(args.expected_records) or routed + bypassed != int(args.expected_decisions):
        raise PublicRouteReconstructionError(
            "raw classifier query selected record/decision totals differ from the sealed r212 source"
        )
    member_route_hasher = hashlib.sha256()
    for row in rows:
        member_route_hasher.update(_canonical_json(row))
    projection = {
        "schema": ROUTE_RECONSTRUCTION_SCHEMA,
        "source_date": str(args.date),
        "source_feature_shard_sha256": feature_shard_sha,
        "raw_archive": raw_archive,
        "runtime_public_tree_sha256": tree_sha,
        "runtime_code": runtime_code.as_dict(),
        "producer_code": producer_code.binding.as_dict(),
        "allowed_physical_slots": sorted(allowed),
        "algorithm": ROUTE_ALGORITHM,
        "alignment_contract": ALIGNMENT_CONTRACT,
        "records": len(rows),
        "decisions": routed + bypassed,
        "active_observations": active_observations,
        "turn_order_short_circuits": turn_order_short_circuits,
        "game_resets": game_resets,
        "routed_decisions": routed,
        "bypassed_decisions": bypassed,
        "member_route_sha256": "sha256:" + member_route_hasher.hexdigest(),
        "compact_source_routes_ignored": True,
        "oracle_route_used": False,
        "source_query": source_query,
    }
    sidecar, sidecar_sha = _no_clobber_sidecar(
        header=header, rows=rows, projection=projection, output_dir=args.output_dir
    )
    producer_code.assert_unchanged()
    return {
        "sidecar": str(sidecar),
        "sha256": sidecar_sha,
        "projection": projection,
        "rejected_members": rejected,
    }


def _parse_day_path(value: str) -> tuple[str, Path]:
    day, marker, raw_path = str(value).partition("=")
    if marker != "=" or day not in HELDOUT_DATES or not raw_path:
        raise ValueError("--manifest-day must use YYYY-MM-DD=SIDECAR_PATH for Jul22/Jul23")
    return day, Path(raw_path).expanduser().resolve()


def _read_complete_sidecar(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256_file(path)
    hasher = hashlib.sha256()
    with path.open("r", encoding="utf-8") as handle:
        try:
            header = json.loads(handle.readline())
        except json.JSONDecodeError as exc:
            raise PublicRouteReconstructionError("sidecar manifest input header is invalid") from exc
        while True:
            line = handle.readline()
            if not line:
                raise PublicRouteReconstructionError("sidecar manifest input lacks footer")
            row = json.loads(line)
            if isinstance(row, dict) and row.get("schema") == ROUTE_SIDECAR_FOOTER_SCHEMA:
                footer = row
                break
            if not isinstance(row, dict) or row.get("schema") != ROUTE_SIDECAR_ROW_SCHEMA:
                raise PublicRouteReconstructionError("sidecar manifest input has invalid row")
            hasher.update(_canonical_json(row))
        if handle.read(1):
            raise PublicRouteReconstructionError("sidecar manifest input has trailing data")
    if (
        not isinstance(header, dict)
        or header.get("schema") != ROUTE_SIDECAR_HEADER_SCHEMA
        or header.get("format") != ROUTE_SIDECAR_FORMAT
        or not isinstance(footer, dict)
        or int(footer.get("rows", -1)) < 1
        or footer.get("rows_sha256") != "sha256:" + hasher.hexdigest()
        or not isinstance(footer.get("projection"), dict)
    ):
        raise PublicRouteReconstructionError("sidecar manifest input footer/header accounting is invalid")
    return header, dict(footer["projection"]), digest


def _write_manifest(args: argparse.Namespace) -> dict[str, Any]:
    declared = dict(_parse_day_path(value) for value in args.manifest_day)
    if tuple(sorted(declared)) != HELDOUT_DATES:
        raise PublicRouteReconstructionError("heldout sidecar manifest requires exactly Jul22 and Jul23")
    reference: dict[str, Any] | None = None
    days: dict[str, Any] = {}
    for day in HELDOUT_DATES:
        path = declared[day]
        header, projection, digest = _read_complete_sidecar(path)
        if header.get("source_date") != day or projection.get("source_date") != day:
            raise PublicRouteReconstructionError("sidecar date/projection drifted before manifest")
        required = {
            "runtime_public_tree_sha256": header.get("runtime_public_tree_sha256"),
            "runtime_code": header.get("runtime_code"),
            "producer_code": header.get("producer_code"),
            "algorithm": header.get("algorithm"),
            "alignment_contract": header.get("alignment_contract"),
        }
        ProducerCodeBinding.from_mapping(
            header.get("producer_code"), label="heldout sidecar producer code"
        )
        if reference is None:
            reference = required
        elif required != reference:
            raise PublicRouteReconstructionError("heldout sidecars do not share exact r195 runtime identity")
        raw_archive = header.get("raw_archive")
        if (
            not isinstance(raw_archive, dict)
            or projection.get("raw_archive") != raw_archive
            or projection.get("producer_code") != header.get("producer_code")
        ):
            raise PublicRouteReconstructionError("sidecar raw archive provenance drifted")
        days[day] = {
            "path": path.name,
            "sha256": digest,
            "raw_archive": raw_archive,
            "source_feature_shard_sha256": header.get("source_feature_shard_sha256"),
        }
    assert reference is not None
    manifest = {
        "schema": SIDECAR_MANIFEST_SCHEMA,
        "days": days,
        **reference,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(manifest)
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    destination = output_dir / f"r212-public-route-sidecar-manifest-{digest.split(':', 1)[1]}.json"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(encoded.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if _sha256_file(destination) != digest:
            raise PublicRouteReconstructionError("sidecar manifest content-address collision")
    return {"manifest": str(destination), "sha256": digest, "payload": manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--raw-archive-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path)
    parser.add_argument("--matchup-tree-sha256", default=R195_TREE_SHA256)
    parser.add_argument("--submission-bundle", type=Path)
    parser.add_argument("--submission-bundle-sha256", default=R195_BUNDLE_SHA256)
    parser.add_argument("--submission-entrypoint-member", default=R195_ENTRYPOINT_MEMBER)
    parser.add_argument("--submission-entrypoint-sha256", default=R195_ENTRYPOINT_SHA256)
    parser.add_argument("--public-matchup-router-member", default=R195_ROUTER_MEMBER)
    parser.add_argument("--public-matchup-router-sha256", default=R195_ROUTER_SHA256)
    parser.add_argument("--allowed-physical-slot", type=int, action="append", default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--feature-shard", type=Path)
    mode.add_argument("--raw-classifier-query", action="store_true")
    mode.add_argument("--write-heldout-manifest", action="store_true")
    parser.add_argument("--feature-shard-sha256")
    parser.add_argument("--raw-archive-root", type=Path)
    parser.add_argument("--mix", type=Path, default=ROOT / "data/training_mixes/top_ladder.v1.json")
    parser.add_argument(
        "--representatives",
        type=Path,
        default=ROOT / "data/training_mixes/top_ladder_representatives.v1.json",
    )
    parser.add_argument("--card-csv", type=Path)
    parser.add_argument("--classifier-sha256", default=CLASSIFIER_SHA256)
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-decisions", type=int)
    parser.add_argument("--manifest-day", action="append", default=[])
    return parser


def _require_common(args: argparse.Namespace) -> None:
    if args.write_heldout_manifest:
        return
    required = (args.date, args.matchup_tree, args.submission_bundle)
    if any(value is None for value in required):
        raise ValueError("day materialization requires date/tree/exact bundle arguments")
    if not args.allowed_physical_slot:
        raise ValueError("day materialization requires every exact runtime-accepted --allowed-physical-slot")
    if args.feature_shard is not None:
        if args.archive is not None or args.raw_archive_sha256 is not None:
            raise ValueError(
                "--feature-shard mode does not accept --archive/--raw-archive-sha256; "
                "the protected shard header and --raw-archive-root bind the raw ZIP"
            )
        return
    if args.archive is None or args.raw_archive_sha256 is None:
        raise ValueError("raw-classifier-query requires --archive and --raw-archive-sha256")


def main() -> int:
    args = _parser().parse_args()
    if args.write_heldout_manifest:
        result = _write_manifest(args)
    else:
        _require_common(args)
        if args.feature_shard is not None:
            if args.feature_shard_sha256 is None or args.raw_archive_root is None:
                raise ValueError("--feature-shard mode requires --feature-shard-sha256 and --raw-archive-root")
            result_path, digest, projection = materialize_day_sidecar(
                source_date=args.date,
                feature_shard_path=args.feature_shard,
                feature_shard_sha256=args.feature_shard_sha256,
                raw_archive_root=args.raw_archive_root,
                matchup_tree_path=args.matchup_tree,
                matchup_tree_sha256=args.matchup_tree_sha256,
                submission_bundle_path=args.submission_bundle,
                submission_bundle_sha256=args.submission_bundle_sha256,
                submission_entrypoint_member=args.submission_entrypoint_member,
                submission_entrypoint_sha256=args.submission_entrypoint_sha256,
                public_matchup_router_member=args.public_matchup_router_member,
                public_matchup_router_sha256=args.public_matchup_router_sha256,
                allowed_physical_slots=frozenset(args.allowed_physical_slot),
                output_dir=args.output_dir,
                producer_materializer_cli_path=Path(__file__),
            )
            result = {"sidecar": str(result_path), "sha256": digest, "projection": projection}
        else:
            if args.card_csv is None:
                # Raw-classifier query mode alone needs the competition card
                # catalogue.  Feature-shard mode must remain self-contained on
                # the archive host and must not resolve this unrelated asset.
                args.card_csv = paths.en_card_data_path()
            if (
                args.feature_shard_sha256 is None
                or args.expected_records is None
                or args.expected_decisions is None
            ):
                raise ValueError(
                    "--raw-classifier-query requires feature-shard SHA and expected record/decision totals"
                )
            result = _raw_query_sidecar(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
