"""Receipt-gated zero-safe migration coverage for the r258 successor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot import own_deck_successor as successor
from poke_bot.model import build_model
from poke_bot.own_deck_migration import (
    MIGRATION_SCHEMA,
    SUCCESSOR_TENSOR_PREFIXES,
    OwnDeckMigrationError,
    materialize_own_deck_successor,
    verify_own_deck_successor_checkpoint,
)
from poke_bot.train import load_model_from_checkpoint


@pytest.fixture(autouse=True)
def _stub_feature_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the synthetic checkpoint independent of a local cg install."""

    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "encoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)


def _cfg() -> config.ModelConfig:
    return config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=1,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        temporal_pos="rope",
        decision_context="history",
        kv_cache=True,
        expanded_heads_enabled=False,
        matchup_adapters_enabled=False,
        decision_fusion_enabled=False,
        dropout=0.0,
    )


def _parent(path: Path) -> Path:
    torch.manual_seed(258)
    cfg = _cfg()
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.float().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    payload = checkpoint.build_checkpoint(
        model=model,
        optimizer=optimizer,
        model_config=cfg,
        step=12,
        epoch=7,
        rl_iteration=10,
        archetype_id="alakazam",
        model_id="alakazam-r241-terminal-test",
        extra={"synthetic": True},
    )
    return checkpoint.atomic_torch_save(payload, path)


def _refresh_receipt(
    manifest: successor.OwnDeckSuccessorManifest,
    parent_digest: str,
) -> dict[str, object]:
    return successor.seal_receipt(
        {
            "schema": successor.REFRESH_COMPLETION_SCHEMA,
            "status": "completed",
            "specialist_id": "alakazam",
            "terminal_completion": True,
            "frozen": True,
            "registered": True,
            "candidate_id": successor.CANDIDATE_ID,
            "manifest_sha256": manifest.identity.sha256,
            "immutable_refresh_lineage": {
                "id": "alakazam-r241-terminal-refresh",
                "sha256": "sha256:" + "1" * 64,
            },
            "completed_refresh_boundary": {
                "id": "iter_00009-terminal",
                "sha256": "sha256:" + "2" * 64,
            },
            "checkpoint": {"id": "expert_before_iter_00010.pt", "sha256": parent_digest},
            "source_receipt_chain": {
                "integrity_verified": True,
                "sha256": "sha256:" + "4" * 64,
            },
            "runtime_receipt_chain": {
                "integrity_verified": True,
                "sha256": "sha256:" + "5" * 64,
            },
        }
    )


def _stage_receipts(
    manifest: successor.OwnDeckSuccessorManifest,
) -> dict[successor.OwnDeckSuccessorStage, dict[str, object]]:
    receipts: dict[successor.OwnDeckSuccessorStage, dict[str, object]] = {}
    prior: dict[str, str] = {}
    for index, stage in enumerate(manifest.pre_refresh_stages):
        receipt = successor.seal_receipt(
            {
                "schema": successor.STAGE_RECEIPT_SCHEMA,
                "candidate_id": successor.CANDIDATE_ID,
                "owner_decision_revision": successor.OWNER_DECISION_REVISION,
                "manifest_sha256": manifest.identity.sha256,
                "stage_id": stage.value,
                "status": successor.OwnDeckSuccessorStageStatus.PASSED.value,
                "source_sha256s": {
                    "fixture": "sha256:" + f"{index + 1:x}" * 64
                },
                "test_command_or_fixture_identity": f"synthetic-{stage.value}",
                "test_result": "passed",
                "public_information_audit": True,
                "direct_policy_audit": True,
                "r241_nonmutation_audit": True,
                "prior_stage_receipt_sha256s": dict(prior),
            }
        )
        receipts[stage] = receipt
        prior[stage.value] = str(receipt["receipt_sha256"])
    return receipts


def _materialize(tmp_path: Path):
    parent = _parent(tmp_path / "parent.pt")
    parent_digest = checkpoint.checkpoint_digest(parent)
    manifest = successor.load_canonical_manifest()
    refresh = _refresh_receipt(manifest, parent_digest)
    stages = _stage_receipts(manifest)
    child = tmp_path / "successor.pt"
    receipt = tmp_path / "migration.json"
    result = materialize_own_deck_successor(
        parent_checkpoint=parent,
        expected_parent_sha256=parent_digest,
        output_checkpoint=child,
        receipt_path=receipt,
        refresh_completion_receipt=refresh,
        stage_receipts=stages,
        ledger_width=12,
    )
    return parent, parent_digest, child, receipt, result, refresh, stages


def test_materializes_exact_dormant_child_and_post_refresh_receipt(tmp_path: Path) -> None:
    parent, parent_digest, child, receipt, result, refresh, stages = _materialize(tmp_path)

    assert parent.exists()
    assert child.exists()
    assert receipt.exists()
    assert child.stat().st_mode & 0o777 == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert result.checkpoint_sha256 == checkpoint.checkpoint_digest(child)
    assert result.verification.optimizer_existing_state_preserved is True
    assert result.verification.optimizer_new_parameters_fresh is True
    assert result.verification.added_tensor_keys
    assert all(
        any(key.startswith(prefix) for prefix in SUCCESSOR_TENSOR_PREFIXES)
        for key in result.verification.added_tensor_keys
    )
    assert checkpoint.checkpoint_digest(parent) == parent_digest

    parent_payload = checkpoint.load_checkpoint(parent, map_location="cpu")
    child_payload = checkpoint.load_checkpoint(child, map_location="cpu")
    for key, value in parent_payload["model_state_dict"].items():
        assert torch.equal(value, child_payload["model_state_dict"][key]), key
    flags = child_payload["model_config"]
    assert flags["own_deck_ledger_enabled"] is True
    assert flags["visible_tutor_completion_head_enabled"] is True
    assert flags["terminal_conversion_head_enabled"] is True
    assert flags["visible_tutor_completion_route_enabled"] is True
    assert flags["terminal_conversion_route_enabled"] is True
    assert flags["own_deck_ledger_runtime_enabled"] is False
    assert flags["visible_tutor_completion_route_runtime_enabled"] is False
    assert flags["terminal_conversion_route_runtime_enabled"] is False
    reloaded = load_model_from_checkpoint(child, device=torch.device("cpu"))
    assert reloaded.own_deck_ledger_enabled is True
    assert reloaded.own_deck_ledger_runtime_enabled is False
    assert reloaded.visible_tutor_completion_head is not None
    assert reloaded.terminal_conversion_head is not None
    migration = child_payload["extra"]["own_deck_successor_migration"]
    assert migration["schema"] == MIGRATION_SCHEMA
    assert migration["runtime_routes_enabled"] is False
    assert migration["physical_training_routes_enabled"] is True
    receipt_payload = json.loads(receipt.read_text())
    assert receipt_payload["schema"] == successor.POST_REFRESH_RECEIPT_SCHEMA
    assert receipt_payload["kind"] == "isolated_migration"
    assert receipt_payload["child_checkpoint"]["sha256"] == result.checkpoint_sha256
    assert successor.validate_post_refresh_receipt(
        receipt_payload,
        kind="isolated_migration",
        manifest=successor.load_canonical_manifest(),
        refresh_completion=successor.validate_refresh_completion_receipt(refresh),
        stage_receipts=successor.validate_prior_stage_receipts(stages),
    ).sha256 == result.receipt_sha256


def test_missing_or_forged_receipts_fail_before_child_publication(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "parent.pt")
    digest = checkpoint.checkpoint_digest(parent)
    manifest = successor.load_canonical_manifest()
    stages = _stage_receipts(manifest)
    child = tmp_path / "child.pt"
    receipt = tmp_path / "receipt.json"

    with pytest.raises(successor.OwnDeckSuccessorGateError):
        materialize_own_deck_successor(
            parent_checkpoint=parent,
            expected_parent_sha256=digest,
            output_checkpoint=child,
            receipt_path=receipt,
            refresh_completion_receipt={},
            stage_receipts=stages,
        )
    assert not child.exists()
    assert not receipt.exists()

    forged = _refresh_receipt(manifest, "sha256:" + "f" * 64)
    with pytest.raises(OwnDeckMigrationError, match="does not bind"):
        materialize_own_deck_successor(
            parent_checkpoint=parent,
            expected_parent_sha256=digest,
            output_checkpoint=child,
            receipt_path=receipt,
            refresh_completion_receipt=forged,
            stage_receipts=stages,
        )
    assert not child.exists()
    assert not receipt.exists()


def test_rejects_unexpected_or_changed_tensor_keys(tmp_path: Path) -> None:
    parent, parent_digest, child, _receipt, _result, _refresh, _stages = _materialize(tmp_path)
    payload = checkpoint.load_checkpoint(child, map_location="cpu")
    payload["model_state_dict"]["unexpected.successor.weight"] = torch.ones(1)
    forged = checkpoint.atomic_torch_save(payload, tmp_path / "forged-extra.pt")
    with pytest.raises(OwnDeckMigrationError, match="strictly load|state inventory"):
        verify_own_deck_successor_checkpoint(
            parent_checkpoint=parent,
            child_checkpoint=forged,
            expected_parent_sha256=parent_digest,
            expected_child_sha256=checkpoint.checkpoint_digest(forged),
            ledger_width=12,
        )

    payload = checkpoint.load_checkpoint(child, map_location="cpu")
    inherited = next(iter(checkpoint.load_checkpoint(parent, map_location="cpu")["model_state_dict"]))
    payload["model_state_dict"][inherited] = payload["model_state_dict"][inherited] + 1
    forged = checkpoint.atomic_torch_save(payload, tmp_path / "forged-drift.pt")
    with pytest.raises(OwnDeckMigrationError, match="inherited tensor changed"):
        verify_own_deck_successor_checkpoint(
            parent_checkpoint=parent,
            child_checkpoint=forged,
            expected_parent_sha256=parent_digest,
            expected_child_sha256=checkpoint.checkpoint_digest(forged),
            ledger_width=12,
        )


def test_rejects_partial_new_prefix_and_nonzero_zero_safe_route(tmp_path: Path) -> None:
    parent, parent_digest, child, _receipt, _result, _refresh, _stages = _materialize(tmp_path)
    payload = checkpoint.load_checkpoint(child, map_location="cpu")
    missing = "terminal_conversion_route.network.2.bias"
    payload["model_state_dict"].pop(missing)
    forged = checkpoint.atomic_torch_save(payload, tmp_path / "forged-missing.pt")
    with pytest.raises(OwnDeckMigrationError, match="state inventory|strictly load"):
        verify_own_deck_successor_checkpoint(
            parent_checkpoint=parent,
            child_checkpoint=forged,
            expected_parent_sha256=parent_digest,
            expected_child_sha256=checkpoint.checkpoint_digest(forged),
            ledger_width=12,
        )

    payload = checkpoint.load_checkpoint(child, map_location="cpu")
    payload["model_state_dict"]["terminal_conversion_route.network.2.bias"].fill_(1.0)
    forged = checkpoint.atomic_torch_save(payload, tmp_path / "forged-route.pt")
    with pytest.raises(OwnDeckMigrationError, match="zero-safe projection is nonzero"):
        verify_own_deck_successor_checkpoint(
            parent_checkpoint=parent,
            child_checkpoint=forged,
            expected_parent_sha256=parent_digest,
            expected_child_sha256=checkpoint.checkpoint_digest(forged),
            ledger_width=12,
        )
