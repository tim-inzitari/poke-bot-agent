from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.expert_rehearsal import (
    REHEARSAL_RECEIPT_SCHEMA_VERSION,
    ExpertManifestIdentity,
    _validate_receipt,
    canonical_rehearsal_loss_weights,
    resolve_expert_manifest,
)


REQUIRED_TARGETS = (
    "temporal_action_rows",
    "opponent_hand_rows",
    "opponent_remainder_rows",
    "opponent_private_prize_rows",
    "lethal_threat_rows",
    "prize_race_rows",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _loss_weights() -> dict[str, float]:
    return {
        "value": 1.0,
        "archetype": 0.05,
        "opponent_hand": 0.05,
        "opponent_hidden_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "alakazam_guide": 0.05,
    }


def _protected_manifest(tmp_path: Path) -> tuple[Path, ExpertManifestIdentity]:
    decisions = 500
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "pokebot-bootstrap-feature-manifest",
                "format_version": 1,
                "dates": ["2026-07-20"],
                "compact_mode": "temporal-expert-v1",
                "max_context": 320,
                "selection": {
                    "value": "alakazam",
                    "seat_semantics": "acting_seat_only",
                },
                "quality_gates": {
                    "passed": True,
                    "hidden_targets_are_aux_only": True,
                },
                "totals": {
                    "records_kept": 10,
                    "decisions_kept": decisions,
                    "target_coverage": {
                        name: decisions for name in REQUIRED_TARGETS
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    pointer.write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": manifest.name,
                "manifest_sha256": _sha256(manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    identity = resolve_expert_manifest(
        pointer,
        min_decisions=100,
        require_protected=True,
        required_archetype="alakazam",
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=REQUIRED_TARGETS,
    )
    return pointer, identity


def test_exact_manifest_contract_fails_closed(tmp_path: Path) -> None:
    pointer, identity = _protected_manifest(tmp_path)
    assert identity.decisions == 500

    payload = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    payload["max_context"] = 319
    Path(identity.path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["manifest_sha256"] = _sha256(Path(identity.path))
    pointer.write_text(json.dumps(pointer_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max-context"):
        resolve_expert_manifest(
            pointer,
            require_protected=True,
            required_archetype="alakazam",
            required_compact_mode="temporal-expert-v1",
            required_max_context=320,
            required_target_coverage=REQUIRED_TARGETS,
        )


def test_receipt_binds_manifest_losses_and_split_seed(tmp_path: Path) -> None:
    _pointer, identity = _protected_manifest(tmp_path)
    checkpoint = tmp_path / "expert.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    receipt = {
        "schema": REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "before_iteration": 10,
        "parent_digest": "sha256:" + "a" * 64,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": _sha256(checkpoint),
        "manifest": identity.as_dict(),
        "epochs": 5,
        "learning_rate": 2e-5,
        "loss_weights": _loss_weights(),
        "corpus_split_seed": 5_000_123,
    }
    validated = _validate_receipt(
        receipt,
        before_iteration=10,
        parent_digest=receipt["parent_digest"],
        epochs=5,
        learning_rate=2e-5,
        manifest_identity=identity,
        loss_weights=_loss_weights(),
        corpus_split_seed=5_000_123,
    )
    assert validated["reused"] is True

    stale = dict(receipt)
    stale["schema"] = 1
    with pytest.raises(RuntimeError, match="schema"):
        _validate_receipt(
            stale,
            before_iteration=10,
            parent_digest=receipt["parent_digest"],
            epochs=5,
            learning_rate=2e-5,
            manifest_identity=identity,
            loss_weights=_loss_weights(),
            corpus_split_seed=5_000_123,
        )

    changed = _loss_weights()
    changed["opponent_hand"] = 0.1
    with pytest.raises(RuntimeError, match="loss-weight"):
        _validate_receipt(
            receipt,
            before_iteration=10,
            parent_digest=receipt["parent_digest"],
            epochs=5,
            learning_rate=2e-5,
            manifest_identity=identity,
            loss_weights=changed,
            corpus_split_seed=5_000_123,
        )
    with pytest.raises(RuntimeError, match="split-seed"):
        _validate_receipt(
            receipt,
            before_iteration=10,
            parent_digest=receipt["parent_digest"],
            epochs=5,
            learning_rate=2e-5,
            manifest_identity=identity,
            loss_weights=_loss_weights(),
            corpus_split_seed=5_000_124,
        )


def test_loss_contract_requires_every_head() -> None:
    incomplete = _loss_weights()
    incomplete.pop("opponent_hand")
    with pytest.raises(ValueError, match="loss keys"):
        canonical_rehearsal_loss_weights(incomplete)
