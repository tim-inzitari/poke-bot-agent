"""Focused tests for the parent-independent Alakazam RTP r197 corpus."""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from poke_bot.authoritative_visual_trace import VisualEpisodeResult
from poke_bot.feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION
from poke_bot.recursive_turn_planner import r197_corpus
from poke_bot.recursive_turn_planner.r197_corpus import (
    ACTION_SPACE_TOO_LARGE_SCHEMA,
    MAX_ACTION_COMBOS,
    R197CorpusError,
    ROW_SCHEMA,
    complete_action_space_fingerprint,
    deterministic_episode_split,
    iter_complete_action_rows,
    materialize_r197_complete_action_corpus,
    sha256_file,
    verify_r197_complete_action_manifest,
)


DAY = "2026-08-01"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _raw_episode(episode_id: str, deck: list[int]) -> dict[str, Any]:
    """The minimum raw shape needed for exact ID/deck revalidation.

    The fixture conversion function below represents the separately tested,
    production visual-trace converter.  The raw zip deliberately also contains
    an invalid unselected member, proving the materializer reads no episode
    outside the protected pointer identities.
    """

    return {
        "id": episode_id,
        "steps": [
            [
                {"action": list(deck)},
                {"action": list(range(500, 560))},
            ]
        ],
    }


def _normal_observation() -> dict[str, Any]:
    return {
        "current": {"yourIndex": 0},
        "select": {
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": "Yes"}, {"type": "No"}],
        },
    }


def _too_large_observation() -> dict[str, Any]:
    return {
        "current": {"yourIndex": 0},
        "select": {
            "minCount": 2,
            "maxCount": 2,
            "option": [{"type": "Choice", "index": index} for index in range(50)],
        },
    }


class _UnknownTargetClassifier:
    """Current-rule fixture: exact protected target is no longer recognized."""

    def classify_episode(self, payload: dict[str, Any]):
        decks = [
            list(payload["steps"][0][0]["action"]),
            list(payload["steps"][0][1]["action"]),
        ]
        return decks, [
            SimpleNamespace(deck_id="unknown", method="fixture_current_rule"),
            SimpleNamespace(
                deck_id="marnie-s-grimmsnarl-ex", method="fixture_current_rule"
            ),
        ]


def _fake_visual_conversion(payload, classifier, *, source: str, required_archetype: str):
    assert required_archetype == "alakazam"
    decks, labels = classifier.classify_episode(payload)
    assert labels[0].deck_id == "alakazam"
    # The r197 wrapper must not alter the opponent's current-rule label.
    assert labels[1].deck_id == "marnie-s-grimmsnarl-ex"
    episode_id = str(payload["id"])
    deck = list(decks[0])
    return VisualEpisodeResult(
        records=[
            {
                "episode_id": episode_id,
                "seat": 0,
                "archetype": "alakazam",
                "opp_archetype": str(labels[1].deck_id),
                "deck": deck,
                "value": 1.0,
                "source": source,
                "info_set_ok": True,
                "steps": [
                    {
                        "env_step": 7,
                        "observation": _normal_observation(),
                        "action": [1],
                    },
                    {
                        "env_step": 8,
                        "observation": _too_large_observation(),
                        "action": [0, 1],
                    },
                ],
            }
        ],
        stats={},
    )


def _find_split_pair() -> tuple[str, str]:
    train = None
    heldout = None
    for number in range(10_000):
        episode_id = f"fixture-{number}"
        split = deterministic_episode_split(
            episode_id, heldout_fraction="0.50"
        )["split"]
        if split == "train" and train is None:
            train = episode_id
        if split == "heldout" and heldout is None:
            heldout = episode_id
        if train is not None and heldout is not None:
            return train, heldout
    raise AssertionError("could not obtain deterministic test split pair")


@pytest.fixture()
def r197_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    deck = [42] * 60
    train_episode, heldout_episode = _find_split_pair()
    archive = archive_root / f"pokemon-tcg-ai-battle-episodes-{DAY}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as source:
        source.writestr(f"{train_episode}.json", json.dumps(_raw_episode(train_episode, deck)))
        source.writestr(
            f"{heldout_episode}.json", json.dumps(_raw_episode(heldout_episode, deck))
        )
        # If the implementation scans the archive instead of the pointer,
        # it will attempt to parse this member and fail the fixture.
        source.writestr("unselected.json", "this is intentionally not JSON")
    archive_sha256 = sha256_file(archive)
    # Production code has the exact immutable Aug1--5 map.  This focused
    # fixture replaces it with its temp archive identity so it exercises the
    # same hard-binding path without copying production raw data into tests.
    monkeypatch.setattr(r197_corpus, "R197_RAW_ARCHIVE_SHA256_BY_DAY", {DAY: archive_sha256})

    shard = tmp_path / "protected.features"
    with shard.open("wb") as stream:
        pickle.dump(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "source_dates": [DAY],
                "required_archetype": "alakazam",
                "source_archive": archive.name,
                "source_archive_sha256": archive_sha256,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for episode_id in (train_episode, heldout_episode):
            pickle.dump(
                SimpleNamespace(
                    episode_id=episode_id,
                    seat=0,
                    archetype="alakazam",
                    deck=deck,
                ),
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "stats": {"records_kept": 2},
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    manifest = tmp_path / "protected-manifest.json"
    _write_json(
        manifest,
        {
            "shards": [
                {
                    "path": shard.name,
                    "sha256": sha256_file(shard),
                    "source_dates": [DAY],
                }
            ]
        },
    )
    pointer = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    _write_json(
        pointer,
        {
            "schema": "poke_bot.pinned_expert_corpus/v1",
            "protected": True,
            "specialist_id": "alakazam",
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest),
        },
    )
    return {
        "archive_root": archive_root,
        "pointer": pointer,
        "pointer_sha256": sha256_file(pointer),
        "output": tmp_path / "output",
        "train_episode": train_episode,
        "heldout_episode": heldout_episode,
    }


def test_materializer_writes_complete_actions_and_separate_overflow_audit(
    r197_inputs: dict[str, Any],
) -> None:
    assert MAX_ACTION_COMBOS == 1024
    receipt = materialize_r197_complete_action_corpus(
        r197_inputs["pointer"],
        r197_inputs["archive_root"],
        r197_inputs["output"],
        heldout_fraction="0.50",
        expected_pointer_sha256=r197_inputs["pointer_sha256"],
        classifier=_UnknownTargetClassifier(),
        convert_episode=_fake_visual_conversion,
    )

    assert receipt["parent_independent"] is True
    assert receipt["eligibility"]["kaggle_replay_eligible"] is False
    assert receipt["status"] == "completed_with_action_space_too_large_exclusions"
    assert receipt["action_space"]["max_action_combos"] == MAX_ACTION_COMBOS
    assert receipt["counts"]["complete_action_rows"] == 2
    assert receipt["counts"]["action_space_too_large_rows"] == 2
    assert receipt["protected_identity_exact"]["counts"] == {
        "verified_records": 2,
        "unknown_label_overrides": 2,
        "current_rule_alakazam_records": 0,
    }
    assert verify_r197_complete_action_manifest(
        r197_inputs["output"], archive_root=r197_inputs["archive_root"]
    )["manifest"]["sha256"] == receipt["manifest"]["sha256"]

    train = list(iter_complete_action_rows(r197_inputs["output"], "train"))
    heldout = list(iter_complete_action_rows(r197_inputs["output"], "heldout"))
    assert {row["episode_id"] for row in train} == {r197_inputs["train_episode"]}
    assert {row["episode_id"] for row in heldout} == {r197_inputs["heldout_episode"]}
    assert not ({row["episode_id"] for row in train} & {row["episode_id"] for row in heldout})
    assert list(
        iter_complete_action_rows(
            r197_inputs["output"],
            "train",
            episode_ids={r197_inputs["train_episode"]},
        )
    ) == train
    assert not list(
        iter_complete_action_rows(
            r197_inputs["output"],
            "train",
            episode_ids={r197_inputs["heldout_episode"]},
        )
    )

    for row in train + heldout:
        assert row["schema"] == ROW_SCHEMA
        assert row["legal_actions"] == [[0], [1]]
        assert row["action"] == [1]
        assert row["selected_action_index"] == 1
        assert row["factorized_prefix_substitution"] is False
        assert "evaluator_targets" not in row
        assert row["protected_identity_exact"]["identity_bound_label_override"] is True
        assert row["protected_identity_exact"]["current_rule_target_label"] == "unknown"
        assert row["protected_identity_exact"]["opponent_current_rule_label"] == (
            "marnie-s-grimmsnarl-ex"
        )
        assert row["opp_archetype"] == "marnie-s-grimmsnarl-ex"
        assert row["action_space_fingerprint"] == complete_action_space_fingerprint(
            row["observation"], row["legal_actions"]
        )
        with pytest.raises(R197CorpusError, match="fixed at 1024"):
            complete_action_space_fingerprint(
                row["observation"], row["legal_actions"], max_action_combos=256
            )

    audit_path = r197_inputs["output"] / "action-space-too-large.jsonl"
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(audit_rows) == 2
    assert all(row["schema"] == ACTION_SPACE_TOO_LARGE_SCHEMA for row in audit_rows)
    assert all(row["complete_ordered_action_count"] == 2450 for row in audit_rows)
    assert all(row["factorized_prefix_substitution"] is False for row in audit_rows)
    assert all("legal_actions" not in row for row in audit_rows)

    # A content-addressed target is safely idempotent only when its receipt and
    # raw inputs still match.  This second call performs verification, not a
    # rewrite.
    reused = materialize_r197_complete_action_corpus(
        r197_inputs["pointer"],
        r197_inputs["archive_root"],
        r197_inputs["output"],
        heldout_fraction="0.50",
        expected_pointer_sha256=r197_inputs["pointer_sha256"],
        classifier=_UnknownTargetClassifier(),
        convert_episode=_fake_visual_conversion,
    )
    assert reused["manifest"]["sha256"] == receipt["manifest"]["sha256"]


def test_recorded_action_must_be_in_complete_ordered_support(
    r197_inputs: dict[str, Any],
) -> None:
    def invalid_action_conversion(payload, classifier, *, source, required_archetype):
        result = _fake_visual_conversion(
            payload, classifier, source=source, required_archetype=required_archetype
        )
        result.records[0]["steps"] = [
            {
                "env_step": 7,
                "observation": _normal_observation(),
                # Two legal options are indexed 0 and 1 only.
                "action": [2],
            }
        ]
        return result

    with pytest.raises(R197CorpusError, match="absent from canonical complete ordered support"):
        materialize_r197_complete_action_corpus(
            r197_inputs["pointer"],
            r197_inputs["archive_root"],
            r197_inputs["output"],
            heldout_fraction="0.50",
            classifier=_UnknownTargetClassifier(),
            convert_episode=invalid_action_conversion,
        )
    assert not r197_inputs["output"].exists()


def test_verifier_rejects_declared_jsonl_row_count_mismatch(
    r197_inputs: dict[str, Any],
) -> None:
    materialize_r197_complete_action_corpus(
        r197_inputs["pointer"],
        r197_inputs["archive_root"],
        r197_inputs["output"],
        heldout_fraction="0.50",
        classifier=_UnknownTargetClassifier(),
        convert_episode=_fake_visual_conversion,
    )
    manifest_path = r197_inputs["output"] / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["train"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(R197CorpusError, match="row count changed"):
        verify_r197_complete_action_manifest(r197_inputs["output"])


def test_materializer_rejects_nearby_raw_archive_with_wrong_digest(
    r197_inputs: dict[str, Any],
) -> None:
    archive = r197_inputs["archive_root"] / f"pokemon-tcg-ai-battle-episodes-{DAY}.zip"
    # The selected episode IDs/decks could still be reproduced in a nearby zip;
    # the owner-pinned bytes, rather than those plausible identities, decide
    # whether it is admissible.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as source:
        source.writestr("nearby.json", json.dumps({"id": "nearby", "steps": []}))
    with pytest.raises(R197CorpusError, match="raw archive digest disagrees"):
        materialize_r197_complete_action_corpus(
            r197_inputs["pointer"],
            r197_inputs["archive_root"],
            r197_inputs["output"],
            heldout_fraction="0.50",
            classifier=_UnknownTargetClassifier(),
            convert_episode=_fake_visual_conversion,
        )
    assert not r197_inputs["output"].exists()


def test_protected_identity_unknown_override_is_exact_and_preserves_opponent() -> None:
    episode_id = "89228866"
    opponent_deck = list(range(100, 160))
    target_deck = list(range(200, 260))
    payload = {
        "id": episode_id,
        "steps": [[{"action": opponent_deck}, {"action": target_deck}]],
    }

    class ProductionLikeClassifier:
        def classify_episode(self, _payload: dict[str, Any]):
            return [opponent_deck, target_deck], [
                SimpleNamespace(
                    deck_id="marnie-s-grimmsnarl-ex", method="current_rule"
                ),
                SimpleNamespace(deck_id="unknown", method="current_rule"),
            ]

    identity = {
        "episode_id": episode_id,
        "seat": 1,
        "day": DAY,
        "deck": target_deck,
        "source_shard": "fixture.features",
    }
    bound = r197_corpus._ProtectedIdentityClassifier(ProductionLikeClassifier(), identity)
    bound.prepare(payload)
    _decks, labels = bound.classify_episode(payload)
    provenance = bound.protected_identity_exact_provenance()

    assert labels[0].deck_id == "marnie-s-grimmsnarl-ex"
    assert labels[1].deck_id == "alakazam"
    assert provenance["episode_id"] == episode_id
    assert provenance["seat"] == 1
    assert provenance["current_rule_target_label"] == "unknown"
    assert provenance["identity_bound_label_override"] is True
    assert provenance["opponent_current_rule_label"] == "marnie-s-grimmsnarl-ex"
    assert provenance["opponent_label_preserved"] is True


def test_protected_identity_override_rejects_recognized_other_and_deck_mismatch() -> None:
    episode_id = "89228866"
    opponent_deck = list(range(100, 160))
    target_deck = list(range(200, 260))
    payload = {
        "id": episode_id,
        "steps": [[{"action": opponent_deck}, {"action": target_deck}]],
    }
    identity = {
        "episode_id": episode_id,
        "seat": 1,
        "day": DAY,
        "deck": target_deck,
        "source_shard": "fixture.features",
    }

    class RecognizedOtherClassifier:
        def classify_episode(self, _payload: dict[str, Any]):
            return [opponent_deck, target_deck], [
                SimpleNamespace(deck_id="marnie-s-grimmsnarl-ex", method="current_rule"),
                SimpleNamespace(deck_id="marnie-s-grimmsnarl-ex", method="current_rule"),
            ]

    with pytest.raises(R197CorpusError, match="different archetype"):
        r197_corpus._ProtectedIdentityClassifier(
            RecognizedOtherClassifier(), identity
        ).prepare(payload)

    class MismatchedDeckClassifier:
        def classify_episode(self, _payload: dict[str, Any]):
            return [opponent_deck, target_deck[:-1] + [999]], [
                SimpleNamespace(deck_id="marnie-s-grimmsnarl-ex", method="current_rule"),
                SimpleNamespace(deck_id="unknown", method="current_rule"),
            ]

    with pytest.raises(R197CorpusError, match="target deck disagrees"):
        r197_corpus._ProtectedIdentityClassifier(
            MismatchedDeckClassifier(), identity
        ).prepare(payload)


def test_identity_bound_override_rejects_converter_opponent_label_mutation() -> None:
    identity = {
        "episode_id": "89228866",
        "seat": 1,
        "day": DAY,
        "deck": list(range(200, 260)),
        "source_shard": "fixture.features",
    }
    provenance = {
        "schema": r197_corpus.PROTECTED_IDENTITY_EXACT_SCHEMA,
        "selection": "protected_pointer_verified_episode_seat_deck",
        "episode_id": "89228866",
        "seat": 1,
        "deck_sha256": r197_corpus.canonical_json_sha256(identity["deck"]),
        "raw_episode_deck_verified": True,
        "current_rule_target_label": "unknown",
        "effective_conversion_target_label": "alakazam",
        "identity_bound_label_override": True,
        "classification_mode": "protected_identity_exact_unknown_override",
        "opponent_current_rule_label": "marnie-s-grimmsnarl-ex",
        "opponent_label_preserved": True,
    }
    with pytest.raises(R197CorpusError, match="preserved opponent"):
        r197_corpus._assert_preserved_opponent_label(
            {"opp_archetype": "other-archetype"}, provenance
        )


def test_verified_member_rejects_embedded_episode_id_mismatch(tmp_path: Path) -> None:
    episode_id = "89228866"
    deck = list(range(60))
    archive_path = tmp_path / f"pokemon-tcg-ai-battle-episodes-{DAY}.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            f"{episode_id}.json",
            json.dumps(
                {
                    "id": "other-episode",
                    "steps": [[{"action": deck}, {"action": deck}]],
                }
            ),
        )
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(R197CorpusError, match="raw episode identity changed"):
            r197_corpus._validated_source_payload(
                archive,
                members=r197_corpus._exact_member_index(archive),
                episode_id=episode_id,
                seat=1,
                deck=deck,
            )


def test_verifier_and_row_reader_require_protected_identity_provenance(
    r197_inputs: dict[str, Any],
) -> None:
    materialize_r197_complete_action_corpus(
        r197_inputs["pointer"],
        r197_inputs["archive_root"],
        r197_inputs["output"],
        heldout_fraction="0.50",
        classifier=_UnknownTargetClassifier(),
        convert_episode=_fake_visual_conversion,
    )
    manifest_path = r197_inputs["output"] / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protected_identity_exact"]["counts"]["unknown_label_overrides"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(R197CorpusError, match="protected identity summary"):
        verify_r197_complete_action_manifest(r197_inputs["output"])

    # Restore the manifest so verify=False isolates the per-row structural
    # guard rather than the sealed-manifest guard.
    manifest["protected_identity_exact"]["counts"]["unknown_label_overrides"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    train_path = r197_inputs["output"] / "train.complete-actions.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["protected_identity_exact"]["seat"] = 1
    rows[0]["target_provenance"]["protected_identity_exact"] = rows[0][
        "protected_identity_exact"
    ]
    row_without_fingerprint = dict(rows[0])
    row_without_fingerprint.pop("row_fingerprint")
    rows[0]["row_fingerprint"] = r197_corpus.canonical_json_sha256(
        row_without_fingerprint
    )
    train_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(R197CorpusError, match="protected identity exact provenance"):
        list(iter_complete_action_rows(r197_inputs["output"], "train", verify=False))
