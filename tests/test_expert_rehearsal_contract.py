from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.pure_rl.expert_rehearsal import (
    ARCHETYPE_FAMILY_REHEARSAL_RECEIPT_SCHEMA_VERSION,
    EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION,
    OPTION_CONDITIONED_REHEARSAL_RECEIPT_SCHEMA_VERSION,
    REHEARSAL_RECEIPT_SCHEMA_VERSION,
    ExpertManifestIdentity,
    _validate_receipt,
    canonical_checkpoint_rehearsal_loss_weights,
    canonical_expanded_rehearsal_contract,
    canonical_rehearsal_loss_weights,
    rehearsal_epochs_for_iteration,
    resolve_expert_manifest,
)
from poke_bot.strategic_heads import (
    EXPANDED_STRATEGIC_SCHEMA,
    TARGET_SCHEMA_DIGEST,
)
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS


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


def test_one_time_large_refresh_preserves_ordinary_cadence() -> None:
    assert rehearsal_epochs_for_iteration(
        15,
        ordinary_epochs=5,
        one_time_before=15,
        one_time_epochs=25,
    ) == 25
    assert rehearsal_epochs_for_iteration(
        20,
        ordinary_epochs=5,
        one_time_before=15,
        one_time_epochs=25,
    ) == 5


@pytest.mark.parametrize(
    ("target", "epochs"),
    [(-1, 25), (15, 0), (-2, 25), (15, -1)],
)
def test_one_time_large_refresh_requires_complete_positive_binding(
    target: int, epochs: int
) -> None:
    with pytest.raises(ValueError, match="one-time expert rehearsal"):
        rehearsal_epochs_for_iteration(
            15,
            ordinary_epochs=5,
            one_time_before=target,
            one_time_epochs=epochs,
        )


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


def test_direct_manifest_requires_exact_sibling_protected_pointer(tmp_path: Path) -> None:
    pointer, identity = _protected_manifest(tmp_path)
    manifest = Path(identity.path)

    direct_identity = resolve_expert_manifest(
        manifest,
        min_decisions=100,
        require_protected=True,
        required_archetype="alakazam",
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=REQUIRED_TARGETS,
    )
    assert direct_identity.digest == identity.digest
    assert direct_identity.path == identity.path

    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["manifest_sha256"] = "sha256:" + "0" * 64
    pointer.write_text(json.dumps(pointer_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sibling protected pointer is invalid"):
        resolve_expert_manifest(manifest, require_protected=True)


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


def _expanded_schedule_contract() -> dict:
    return {
        "schema": "poke_bot.expanded_head_schedule/v1",
        "target_schema": EXPANDED_STRATEGIC_SCHEMA,
        "target_schema_digest": TARGET_SCHEMA_DIGEST,
        "schedule_digest": "sha256:" + "b" * 64,
        "loss_weights": {
            name: (0.1 if name == "action_q" else 0.0)
            for name in EXPANDED_HEAD_IDS
        },
        "runtime_enabled_heads": [],
    }


def test_expanded_receipt_binds_shadow_training_and_digests(
    tmp_path: Path,
) -> None:
    _pointer, identity = _protected_manifest(tmp_path)
    output = tmp_path / "expanded-expert.pt"
    output.write_bytes(b"immutable-expanded-checkpoint")
    expanded = canonical_expanded_rehearsal_contract(
        _expanded_schedule_contract()
    )
    training = {
        "schema": "poke_bot.expanded_head_training/v1",
        "target_schema_version": expanded["target_schema"],
        "target_schema_digest": expanded["target_schema_digest"],
        "schedule_digest": expanded["schedule_digest"],
        "loss_weights": expanded["loss_weights"],
        "gradient_enabled_heads": ["action_q"],
        "runtime_enabled_heads": [],
        "heads": {
            "action_q": {
                "present": True,
                "trained_this_epoch": True,
                "gradient_enabled": True,
                "train_loss": 0.4,
                "validation_loss": 0.5,
                "train_labeled_rows": 400,
                "validation_labeled_rows": 100,
            }
        },
    }
    receipt = {
        "schema": EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "before_iteration": 10,
        "parent_digest": "sha256:" + "a" * 64,
        "checkpoint": str(output),
        "checkpoint_digest": _sha256(output),
        "manifest": identity.as_dict(),
        "epochs": 5,
        "learning_rate": 2e-5,
        "loss_weights": _loss_weights(),
        "corpus_split_seed": 5_000_123,
        "expanded_head_training": training,
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
        expanded_head_contract=expanded,
    )
    assert validated["reused"] is True

    drifted = {
        **receipt,
        "expanded_head_training": {
            **training,
            "target_schema_digest": "sha256:" + "c" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="target digest"):
        _validate_receipt(
            drifted,
            before_iteration=10,
            parent_digest=receipt["parent_digest"],
            epochs=5,
            learning_rate=2e-5,
            manifest_identity=identity,
            loss_weights=_loss_weights(),
            corpus_split_seed=5_000_123,
            expanded_head_contract=expanded,
        )

    runtime_enabled = {
        **_expanded_schedule_contract(),
        "runtime_enabled_heads": ["action_q"],
    }
    with pytest.raises(ValueError, match="shadow-only"):
        canonical_expanded_rehearsal_contract(runtime_enabled)


def test_expanded_receipt_allows_masked_or_train_only_labels(
    tmp_path: Path,
) -> None:
    _pointer, identity = _protected_manifest(tmp_path)
    output = tmp_path / "expanded-masked-expert.pt"
    output.write_bytes(b"immutable-expanded-masked-checkpoint")
    expanded = canonical_expanded_rehearsal_contract(
        _expanded_schedule_contract()
    )
    training = {
        "schema": "poke_bot.expanded_head_training/v1",
        "target_schema_version": expanded["target_schema"],
        "target_schema_digest": expanded["target_schema_digest"],
        "schedule_digest": expanded["schedule_digest"],
        "loss_weights": expanded["loss_weights"],
        "gradient_enabled_heads": ["action_q"],
        "runtime_enabled_heads": [],
        "heads": {
            "action_q": {
                "present": True,
                "trained": True,
                # Legacy checkpoints required validation coverage before
                # setting this telemetry bit. Finite train-pass evidence is
                # authoritative when a deterministic split has no val labels.
                "trained_this_epoch": False,
                "gradient_enabled": True,
                "train_loss": 0.4,
                "validation_loss": 0.0,
                "train_labeled_rows": 400,
                "validation_labeled_rows": 0,
            }
        },
    }
    receipt = {
        "schema": EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "before_iteration": 10,
        "parent_digest": "sha256:" + "a" * 64,
        "checkpoint": str(output),
        "checkpoint_digest": _sha256(output),
        "manifest": identity.as_dict(),
        "epochs": 5,
        "learning_rate": 2e-5,
        "loss_weights": _loss_weights(),
        "corpus_split_seed": 5_000_123,
        "expanded_head_training": training,
    }
    assert _validate_receipt(
        receipt,
        before_iteration=10,
        parent_digest=receipt["parent_digest"],
        epochs=5,
        learning_rate=2e-5,
        manifest_identity=identity,
        loss_weights=_loss_weights(),
        corpus_split_seed=5_000_123,
        expanded_head_contract=expanded,
    )["reused"] is True


def test_expanded_checkpoint_loss_contract_accepts_bound_nested_weights() -> None:
    expanded = canonical_expanded_rehearsal_contract(
        _expanded_schedule_contract()
    )
    checkpoint_losses = {
        **_loss_weights(),
        "expanded_strategic": expanded["loss_weights"],
    }
    assert canonical_checkpoint_rehearsal_loss_weights(
        checkpoint_losses,
        expanded,
    ) == _loss_weights()

    changed = {
        **checkpoint_losses,
        "expanded_strategic": {
            **expanded["loss_weights"],
            "action_q": 0.2,
        },
    }
    with pytest.raises(ValueError, match="strategic weights mismatch"):
        canonical_checkpoint_rehearsal_loss_weights(changed, expanded)

    missing = dict(_loss_weights())
    with pytest.raises(ValueError, match="missing strategic weights"):
        canonical_checkpoint_rehearsal_loss_weights(missing, expanded)


def test_option_conditioned_checkpoint_and_receipt_bind_combo_weight(
    tmp_path: Path,
) -> None:
    checkpoint_losses = {**_loss_weights(), "combo_state": 0.025}
    assert canonical_checkpoint_rehearsal_loss_weights(
        checkpoint_losses,
        option_conditioned_loss_weights={"combo_state": 0.025},
    ) == _loss_weights()
    with pytest.raises(ValueError, match="option-conditioned"):
        canonical_checkpoint_rehearsal_loss_weights(
            checkpoint_losses,
            option_conditioned_loss_weights={"combo_state": 0.05},
        )

    _pointer, identity = _protected_manifest(tmp_path)
    output = tmp_path / "combo-expert.pt"
    output.write_bytes(b"immutable-combo-checkpoint")
    receipt = {
        "schema": OPTION_CONDITIONED_REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "before_iteration": 5,
        "parent_digest": "sha256:" + "a" * 64,
        "checkpoint": str(output),
        "checkpoint_digest": _sha256(output),
        "manifest": identity.as_dict(),
        "epochs": 5,
        "learning_rate": 2e-5,
        "loss_weights": _loss_weights(),
        "option_conditioned_loss_weights": {"combo_state": 0.025},
        "corpus_split_seed": 5_000_123,
    }
    assert _validate_receipt(
        receipt,
        before_iteration=5,
        parent_digest=receipt["parent_digest"],
        epochs=5,
        learning_rate=2e-5,
        manifest_identity=identity,
        loss_weights=_loss_weights(),
        option_conditioned_loss_weights={"combo_state": 0.025},
        corpus_split_seed=5_000_123,
    )["reused"] is True


def test_family_residual_receipt_binds_effective_expanded_heads(
    tmp_path: Path,
) -> None:
    _pointer, identity = _protected_manifest(tmp_path)
    output = tmp_path / "family-residual-expert.pt"
    output.write_bytes(b"immutable-family-residual-checkpoint")
    expanded = canonical_expanded_rehearsal_contract(
        _expanded_schedule_contract()
    )
    residuals = {
        "core_setup_continuity": 0.0,
        "resource_attack_readiness": 0.0125,
        "long_horizon_prize_pressure": 0.0,
    }
    effective = dict(expanded["loss_weights"])
    effective["action_resource"] += 0.0125
    effective["resource_forecast"] += 0.0125
    enabled = [
        name for name, weight in effective.items() if float(weight) > 0.0
    ]
    heads = {
        name: {
            "present": True,
            "trained_this_epoch": True,
            "gradient_enabled": True,
            "train_loss": 0.4,
            "validation_loss": 0.5,
            "train_labeled_rows": 400,
            "validation_labeled_rows": 100,
        }
        for name in enabled
    }
    training = {
        "schema": "poke_bot.expanded_head_training/v1",
        "target_schema_version": expanded["target_schema"],
        "target_schema_digest": expanded["target_schema_digest"],
        "schedule_digest": expanded["schedule_digest"],
        "loss_weights": expanded["loss_weights"],
        "effective_loss_weights": effective,
        "archetype_residual_loss_weights": residuals,
        "gradient_enabled_heads": enabled,
        "runtime_enabled_heads": [],
        "heads": heads,
    }
    receipt = {
        "schema": ARCHETYPE_FAMILY_REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "before_iteration": 10,
        "parent_digest": "sha256:" + "a" * 64,
        "checkpoint": str(output),
        "checkpoint_digest": _sha256(output),
        "manifest": identity.as_dict(),
        "epochs": 5,
        "learning_rate": 2e-5,
        "loss_weights": _loss_weights(),
        "archetype_residual_loss_weights": residuals,
        "corpus_split_seed": 5_000_123,
        "expanded_head_training": training,
    }
    assert _validate_receipt(
        receipt,
        before_iteration=10,
        parent_digest=receipt["parent_digest"],
        epochs=5,
        learning_rate=2e-5,
        manifest_identity=identity,
        loss_weights=_loss_weights(),
        corpus_split_seed=5_000_123,
        expanded_head_contract=expanded,
        archetype_residual_loss_weights=residuals,
    )["reused"] is True

    drifted = dict(receipt)
    drifted["archetype_residual_loss_weights"] = {
        **residuals,
        "resource_attack_readiness": 0.025,
    }
    with pytest.raises(RuntimeError, match="residual"):
        _validate_receipt(
            drifted,
            before_iteration=10,
            parent_digest=receipt["parent_digest"],
            epochs=5,
            learning_rate=2e-5,
            manifest_identity=identity,
            loss_weights=_loss_weights(),
            corpus_split_seed=5_000_123,
            expanded_head_contract=expanded,
            archetype_residual_loss_weights=residuals,
        )
