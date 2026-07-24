from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from poke_bot import features
from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, DecisionSample, GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
)
from poke_bot.matchup_adapter_activation import adapter_training_ticket
from poke_bot.matchup_adapters import EXPERT_IDS, UNKNOWN_ROUTE, route_for_archetype
from poke_bot.pure_rl import matchup_adapter_corpus as corpus_mod
from poke_bot.pure_rl.matchup_adapter_corpus import (
    ORACLE_INDEX_ROW_SCHEMA,
    ORACLE_MANIFEST_SCHEMA,
    PACKAGE_REGISTRY_SCHEMA,
    STAGED_CORPUS_SCHEMA,
    full_deck_digest,
    iter_staged_split,
    sha256_file,
    stage_matchup_adapter_corpus,
    write_oracle_index,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _sequence(episode: str, archetype: str, *, seat: int = 0) -> GameSequence:
    decisions = [
        DecisionSample(
            board=None,
            options=None,
            action=[0],
            action_combo_index=0,
            action_combos=[[0]],
            env_step=index,
        )
        for index in range(3)
    ]
    return GameSequence(
        episode_id=episode,
        seat=seat,
        archetype="alakazam",
        opp_archetype=archetype,
        deck=[1] * 60,
        value=1.0,
        decisions=decisions,
        source="",
        target_provenance={},
    )


def _oracle_row(sequence: GameSequence, *, formal_eval: bool = False) -> dict:
    return {
        "schema": ORACLE_INDEX_ROW_SCHEMA,
        "episode_id": sequence.episode_id,
        "seat": sequence.seat,
        "acting_archetype": "alakazam",
        "opponent_archetype": sequence.opp_archetype,
        "opponent_id": f"ladder-{sequence.opp_archetype}",
        "opponent_content_digest": _digest(f"package:{sequence.opp_archetype}"),
        "opponent_deck_digest": _digest(f"deck:{sequence.opp_archetype}"),
        "source_archive_digest": _digest("archive:2026-07-20"),
        "source_member_digest": _digest(f"member:{sequence.episode_id}"),
        "classifier_method": "representative_exact",
        "formal_eval": formal_eval,
        "training_eligible": not formal_eval,
    }


def _source_corpus(
    root: Path,
    *,
    add_gate_row: bool = False,
    public_router_digest: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    sequences = [
        _sequence(f"{archetype}-episode-{index}", archetype)
        for archetype in EXPERT_IDS
        for index in range(2)
    ]
    if add_gate_row:
        sequences.append(_sequence("gate-episode", EXPERT_IDS[0]))
    feature = root / "expert.features"
    classifier_digest = _digest("classifier-contract")
    source_archive_digest = _digest("archive:2026-07-20")
    stats = {
        "records_kept": len(sequences),
        "decisions_kept": sum(len(row.decisions) for row in sequences),
    }
    with feature.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                "feature_schema": features.FEATURE_SCHEMA_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "required_archetype": "alakazam",
                "classifier_sha256": classifier_digest,
                "source_archive_sha256": source_archive_digest,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for sequence in sequences:
            pickle.dump(sequence, stream, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    feature_manifest = root / "feature-manifest.json"
    feature_manifest.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "format_version": MANIFEST_FORMAT_VERSION,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
                "shards": [
                    {
                        "path": feature.name,
                        "sha256": sha256_file(feature),
                        "stats": stats,
                    }
                ],
                "totals": stats,
            },
            sort_keys=True,
        )
        + "\n"
    )

    oracle_rows = [_oracle_row(sequence) for sequence in sequences]
    if public_router_digest is not None:
        for row in oracle_rows:
            row.update(
                public_router_digest=public_router_digest,
                public_route_segments=[[1, 3]],
                public_decisions=3,
                public_confidence_threshold=1.0,
                public_consecutive_required=2,
            )
    if add_gate_row:
        oracle_rows[-1].update(
            opponent_id="heldout-agent",
            opponent_content_digest=_digest("heldout-package"),
        )
    oracle_index = root / "expert.oracle.jsonl"
    oracle_meta = write_oracle_index(
        oracle_index,
        source_feature_digest=sha256_file(feature),
        classifier_digest=classifier_digest,
        rows=oracle_rows,
    )
    oracle_manifest = root / "oracle-manifest.json"
    package_registry = root / "package-registry.json"
    package_rows: dict[str, str] = {}
    for row in oracle_rows:
        package_rows.setdefault(
            str(row["opponent_id"]), str(row["opponent_content_digest"])
        )
    package_registry.write_text(
        json.dumps(
            {
                "schema": PACKAGE_REGISTRY_SCHEMA,
                "packages": [
                    {
                        "opponent_id": opponent_id,
                        "content_digest": digest,
                        "aliases": [],
                    }
                    for opponent_id, digest in sorted(package_rows.items())
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    oracle_manifest.write_text(
        json.dumps(
            {
                "schema": ORACLE_MANIFEST_SCHEMA,
                "source_feature_manifest_digest": sha256_file(feature_manifest),
                "classifier_digest": classifier_digest,
                "package_registry_digest": sha256_file(package_registry),
                "shards": [
                    {
                        "source_feature_digest": sha256_file(feature),
                        "path": oracle_index.name,
                        "sha256": oracle_meta["sha256"],
                        "records": len(sequences),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    gate = root / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "active_gate_id": "strong-public-v1",
                "next_gate": {
                    "id": "strong-public-v1",
                    "evaluation": {
                        "formal_eval_disjoint_from_training": True,
                        "package_digest_deduplicated": True,
                    },
                    "roster": [
                        {
                            "opponent_id": "heldout-agent",
                            "content_digest": _digest("heldout-package"),
                        }
                    ],
                    "excluded_aliases": [],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return feature_manifest, oracle_manifest, package_registry, gate


def test_stage_is_route_specific_episode_disjoint_and_runtime_unknown(
    tmp_path: Path,
) -> None:
    feature, oracle, registry, gate = _source_corpus(tmp_path, add_gate_row=True)
    output = tmp_path / "staged"
    manifest_path = stage_matchup_adapter_corpus(
        feature,
        oracle,
        registry,
        gate,
        output,
        val_frac=0.5,
        seed=7,
        min_available_bytes=0,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == STAGED_CORPUS_SCHEMA
    assert manifest["runtime_routes_enabled"] is False
    assert manifest["totals"]["selected_sequences"] == 2 * len(EXPERT_IDS)
    assert manifest["totals"]["gate_or_ineligible_excluded"] == 1
    assert len(manifest["shards"]) == 2 * len(EXPERT_IDS)

    train = list(iter_staged_split(manifest_path, "train"))
    val = list(iter_staged_split(manifest_path, "val"))
    assert len(train) == len(val) == len(EXPERT_IDS)
    assert {row.episode_id for row in train}.isdisjoint(
        {row.episode_id for row in val}
    )
    assert {row.opp_archetype for row in train} == set(EXPERT_IDS)
    for sequence in train + val:
        route = route_for_archetype(sequence.opp_archetype)
        ticket = adapter_training_ticket(sequence)
        assert ticket.route == route
        assert sequence.matchup_adapter_training_ticket["identity_kind"] == (
            "raw-archive-full-deck-and-package"
        )
        assert all(
            decision.matchup_adapter_oracle_route == route
            and decision.matchup_adapter_public_route == UNKNOWN_ROUTE
            for decision in sequence.decisions
        )


def test_causal_stage_only_trains_rows_after_public_route_confirmation(
    tmp_path: Path,
) -> None:
    router_digest = _digest("public-router")
    feature, oracle, registry, gate = _source_corpus(
        tmp_path,
        public_router_digest=router_digest,
    )
    manifest_path = stage_matchup_adapter_corpus(
        feature,
        oracle,
        registry,
        gate,
        tmp_path / "causal-staged",
        val_frac=0.5,
        seed=7,
        min_available_bytes=0,
        required_public_router_digest=router_digest,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["causal_public_alignment"] is True
    assert manifest["public_router_digest"] == router_digest
    for sequence in (
        list(iter_staged_split(manifest_path, "train"))
        + list(iter_staged_split(manifest_path, "val"))
    ):
        route = route_for_archetype(sequence.opp_archetype)
        assert [
            decision.matchup_adapter_oracle_route
            for decision in sequence.decisions
        ] == [UNKNOWN_ROUTE, route, route]
        assert all(
            decision.matchup_adapter_public_route == UNKNOWN_ROUTE
            for decision in sequence.decisions
        )
        ticket = sequence.matchup_adapter_training_ticket
        assert ticket["oracle_route_rows"] == 2
        assert ticket["public_route_segments"] == [[1, 3]]


def test_stage_is_deterministic_and_reopens_sources_without_retaining_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature, oracle, registry, gate = _source_corpus(tmp_path)
    original = corpus_mod.iter_feature_shard
    passes = 0

    def tracked(path: Path):
        nonlocal passes
        passes += 1
        yield from original(path)

    monkeypatch.setattr(corpus_mod, "iter_feature_shard", tracked)
    first = stage_matchup_adapter_corpus(
        feature,
        oracle,
        registry,
        gate,
        tmp_path / "first",
        val_frac=0.5,
        seed=99,
        min_available_bytes=0,
    )
    assert passes == 2
    second = stage_matchup_adapter_corpus(
        feature,
        oracle,
        registry,
        gate,
        tmp_path / "second",
        val_frac=0.5,
        seed=99,
        min_available_bytes=0,
    )
    first_payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())
    assert first_payload == second_payload
    assert [row["sha256"] for row in first_payload["shards"]] == [
        row["sha256"] for row in second_payload["shards"]
    ]


def test_misaligned_or_incomplete_oracle_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    feature, oracle, registry, gate = _source_corpus(tmp_path)
    oracle_payload = json.loads(oracle.read_text())
    index = tmp_path / oracle_payload["shards"][0]["path"]
    lines = index.read_text().splitlines()
    row = json.loads(lines[1])
    row["episode_id"] = "wrong-episode"
    lines[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    index.write_text("\n".join(lines) + "\n")
    oracle_payload["shards"][0]["sha256"] = sha256_file(index)
    oracle.write_text(json.dumps(oracle_payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="alignment mismatch"):
        stage_matchup_adapter_corpus(
            feature,
            oracle,
            registry,
            gate,
            tmp_path / "misaligned",
            val_frac=0.5,
            min_available_bytes=0,
        )

    # A compact feature manifest by itself is deliberately insufficient: the
    # missing raw-archive sidecar is a hard launch blocker, not a guessed label.
    with pytest.raises(FileNotFoundError):
        stage_matchup_adapter_corpus(
            feature,
            tmp_path / "missing-oracle.json",
            registry,
            gate,
            tmp_path / "missing",
            val_frac=0.5,
            min_available_bytes=0,
        )


def test_oracle_writer_rejects_missing_full_deck_or_package_digest(
    tmp_path: Path,
) -> None:
    sequence = _sequence("episode", EXPERT_IDS[0])
    row = _oracle_row(sequence)
    row.pop("opponent_deck_digest")
    with pytest.raises(ValueError, match="full-deck digest"):
        write_oracle_index(
            tmp_path / "bad.jsonl",
            source_feature_digest=_digest("features"),
            classifier_digest=_digest("classifier"),
            rows=[row],
        )

    deck = list(range(60))
    assert full_deck_digest(deck) == full_deck_digest(reversed(deck))
    with pytest.raises(ValueError, match="exactly 60"):
        full_deck_digest(deck[:-1])
    with pytest.raises(ValueError, match="exact integer card IDs"):
        full_deck_digest([True, *deck[1:]])

    boolean_seat = _oracle_row(sequence)
    boolean_seat["seat"] = True
    with pytest.raises(ValueError, match="exact integer"):
        write_oracle_index(
            tmp_path / "boolean-seat.jsonl",
            source_feature_digest=_digest("features"),
            classifier_digest=_digest("classifier"),
            rows=[boolean_seat],
        )


def test_stager_rejects_package_registry_cross_contamination(tmp_path: Path) -> None:
    feature, oracle, registry, gate = _source_corpus(tmp_path)
    registry_payload = json.loads(registry.read_text())
    registry_payload["packages"][0]["content_digest"] = _digest(
        "different-package"
    )
    registry.write_text(json.dumps(registry_payload, sort_keys=True) + "\n")
    oracle_payload = json.loads(oracle.read_text())
    # Even when an attacker/rebuild updates the outer registry checksum, every
    # row must still resolve to the exact package digest pinned by the registry.
    oracle_payload["package_registry_digest"] = sha256_file(registry)
    oracle.write_text(json.dumps(oracle_payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="contradicts the pinned package registry"):
        stage_matchup_adapter_corpus(
            feature,
            oracle,
            registry,
            gate,
            tmp_path / "contaminated",
            val_frac=0.5,
            min_available_bytes=0,
        )


def test_stager_rejects_ambiguous_normalized_registry_aliases(
    tmp_path: Path,
) -> None:
    feature, oracle, registry, gate = _source_corpus(tmp_path)
    registry_payload = json.loads(registry.read_text())
    registry_payload["packages"][0]["aliases"] = [" Alias ", "alias"]
    registry.write_text(json.dumps(registry_payload, sort_keys=True) + "\n")
    oracle_payload = json.loads(oracle.read_text())
    oracle_payload["package_registry_digest"] = sha256_file(registry)
    oracle.write_text(json.dumps(oracle_payload, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="repeats an identity"):
        stage_matchup_adapter_corpus(
            feature,
            oracle,
            registry,
            gate,
            tmp_path / "duplicate-alias",
            val_frac=0.5,
            min_available_bytes=0,
        )
