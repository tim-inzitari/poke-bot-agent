from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.artifact_retention import apply_artifact_retention
from poke_bot.pure_rl.model_registry import freeze_model, sha256, verify_frozen_model


def _evidence(digest: str) -> dict:
    return {
        "iteration": 9,
        "checkpoint_digest": digest,
        "games": 1000,
        "win_rate": 0.40,
        "confidence_lower": 0.37,
        "confidence_upper": 0.43,
        "audit": {
            "passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "valid_games": 1000,
            "checkpoint_digest": digest,
        },
    }


def _unharden(family: Path) -> None:
    for directory in (family, family / "versions", *list((family / "versions").glob("*"))):
        if directory.exists():
            directory.chmod(0o755)
    for path in family.rglob("*"):
        if path.is_file():
            path.chmod(0o644)


def test_deck_agnostic_core_is_write_once_exact_and_outside_retention(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"deck-agnostic-core")
    digest = sha256(checkpoint)
    registry = tmp_path / "registry"
    try:
        manifest = freeze_model(
            registry_root=registry,
            family="deck_agnostic_core",
            display_name="Deck Agnostic Core",
            checkpoint=checkpoint,
            expected_digest=digest,
            provenance={"source_run": "core", "source_iteration": 9},
            evidence=_evidence(digest),
            require_exact_heldout=True,
        )
        family = registry / "deck_agnostic_core"
        assert manifest["display_name"] == "Deck Agnostic Core"
        assert manifest["automatic_pruning_allowed"] is False
        assert sha256(family / "model.pt") == digest
        assert verify_frozen_model(family)["checkpoint_digest"] == digest
        assert json.loads((family / "PROTECTED_DO_NOT_PRUNE.json").read_text())[
            "automatic_pruning_allowed"
        ] is False
        with pytest.raises(RuntimeError, match="protected model registry"):
            apply_artifact_retention(
                family,
                {},
                completed_iteration=99,
                replay_window_shards=1,
            )
        # Idempotent with the same exact identity.
        assert freeze_model(
            registry_root=registry,
            family="deck_agnostic_core",
            display_name="Deck Agnostic Core",
            checkpoint=checkpoint,
            expected_digest=digest,
            provenance={},
            evidence=_evidence(digest),
            require_exact_heldout=True,
        )["checkpoint_digest"] == digest
    finally:
        family = registry / "deck_agnostic_core"
        if family.exists():
            _unharden(family)


def test_deck_agnostic_core_rejects_replacement_or_weak_audit(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    checkpoint = tmp_path / "one.pt"
    checkpoint.write_bytes(b"one")
    digest = sha256(checkpoint)
    weak = _evidence(digest)
    weak["audit"]["valid_games"] = 999
    with pytest.raises(ValueError, match="not exact/complete"):
        freeze_model(
            registry_root=registry,
            family="deck_agnostic_core",
            display_name="Deck Agnostic Core",
            checkpoint=checkpoint,
            expected_digest=digest,
            provenance={},
            evidence=weak,
            require_exact_heldout=True,
            harden_permissions=False,
        )

    freeze_model(
        registry_root=registry,
        family="deck_agnostic_core",
        display_name="Deck Agnostic Core",
        checkpoint=checkpoint,
        expected_digest=digest,
        provenance={},
        evidence=_evidence(digest),
        require_exact_heldout=True,
        harden_permissions=False,
    )
    replacement = tmp_path / "two.pt"
    replacement.write_bytes(b"two")
    with pytest.raises(RuntimeError, match="another digest"):
        freeze_model(
            registry_root=registry,
            family="deck_agnostic_core",
            display_name="Deck Agnostic Core",
            checkpoint=replacement,
            expected_digest=sha256(replacement),
            provenance={},
        )
