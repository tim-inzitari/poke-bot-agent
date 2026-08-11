"""R260 pre-start successor stays distinct from the legacy peak-r195 audit."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint, config, features
from poke_bot.model import build_model
from poke_bot.r241_own_deck_successor import (
    R241OwnDeckSuccessorError,
    load_r260_owner_contract,
    materialize_r260_own_deck_successor,
    validate_r260_sidecar_binding,
)


@pytest.fixture(autouse=True)
def _small_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("card_vocab_size", "attack_vocab_size", "encoder_vocab_size", "decoder_vocab_size"):
        monkeypatch.setattr(features, name, lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)


def _parent(path: Path) -> Path:
    cfg = config.ModelConfig(d_model=16, spatial_layers=1, temporal_layers=1, option_decoder_layers=1, n_heads=4, ff_dim=32, max_context=8, temporal_pos="rope", decision_context="history", kv_cache=True, expanded_heads_enabled=False, matchup_adapters_enabled=False, decision_fusion_enabled=False, dropout=0.0)
    model = build_model(cfg, device=torch.device("cpu"), aux_archetype_classes=3, encoder_vocab=64, decoder_vocab=64, belief_card_vocab=64)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.float().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    return checkpoint.atomic_torch_save(checkpoint.build_checkpoint(model=model, optimizer=optimizer, model_config=cfg, step=1, epoch=1, rl_iteration=0, archetype_id="alakazam", model_id="synthetic-r195"), path)


def _owner(path: Path, parent: Path) -> object:
    value = {
        "latest_owner_clarification_revision": 262,
        "parent": {"checkpoint": str(parent), "checkpoint_sha256": checkpoint.checkpoint_digest(parent), "checkpoint_bytes": parent.stat().st_size},
        "own_deck_head_structure_import": {
            "owner_revision": 260,
            "expert_corpus": {"source_manifest_sha256": "sha256:" + "1" * 64, "source_window_receipt_sha256": "sha256:" + "2" * 64, "day_count": 20, "validated_episode_count": 91253, "source_archive_bytes": 14842033482, "partial_or_unreceipted_side_store_training_eligible": False, "derived_side_store_root": "/sealed"},
            "training_placement": {
                "owner_revision": 262,
                "sole_managed_training_host": "inzi",
                "elmo_role": "read_only_source_preprocessing_and_bounded_disposable_parity_only",
                "elmo_may_train_learner": False,
                "canonical_inzi_training_root": "/inzi/final",
                "inzi_prefix_staging_root": "/inzi/final-staging-09848f04",
                "prefix_transfer_while_elmo_builder_runs": True,
                "prefix_transfer_scope": "committed_non_dot_daily_directories_only",
                "per_day_transfer": "create_only_byte_identical_rehash_and_read_only_seal",
                "partial_staging_root_training_eligible": False,
                "final_promotion": "atomic_only_after_20_of_20_join_parity_and_transport_receipts_pass",
                "trainer_may_consume_elmo_mnt_main_path": False,
                "trainer_input": "local_inzi_disk_backed_exact_four_key_streaming_index_only",
                "healthy_r259_service_may_be_stopped_restarted_or_reconfigured": False,
            },
            "architecture": {"shared_adapter_width": 128, "option_feature_dim": 8, "visible_tutor_completion_output_dim": 7, "terminal_conversion_output_dim": 6, "typed_option_route_width": 16, "typed_option_route_aggregate_delta_cap": 1.0, "visible_tutor_completion_loss_weight": 0.025, "terminal_conversion_loss_weight": 0.025, "new_tensor_prefixes": ["own_deck_ledger_adapter.", "own_deck_ledger_option_adapter.", "visible_tutor_completion_head.", "terminal_conversion_head.", "visible_tutor_completion_route.", "terminal_conversion_route."]},
            "migration": {"zero_safe_final_projection_keys": ["own_deck_ledger_adapter.output.weight", "own_deck_ledger_adapter.output.bias", "own_deck_ledger_option_adapter.network.3.weight", "own_deck_ledger_option_adapter.network.3.bias", "visible_tutor_completion_route.network.2.weight", "visible_tutor_completion_route.network.2.bias", "terminal_conversion_route.network.2.weight", "terminal_conversion_route.network.2.bias"]},
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return load_r260_owner_contract(path, expected_sha256=None)


def _closure() -> dict[str, object]:
    return {"schema": "poke_bot.r241_own_deck_successor_source_closure/v1", "status": "sealed", "owner_contract_sha256": "", "source_tree_sha256": "sha256:" + "3" * 64, "closure_receipt_sha256": "sha256:" + "4" * 64, "derived_from_r259_runtime_tree": False, "unlisted_pycache_present": False}


def test_materializes_r260_child_with_fresh_closure(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "r195.pt")
    owner = _owner(tmp_path / "owner.json", parent)
    closure = _closure()
    closure["owner_contract_sha256"] = owner.sha256
    result = materialize_r260_own_deck_successor(parent_checkpoint=parent, output_checkpoint=tmp_path / "child.pt", receipt_path=tmp_path / "migration.json", source_closure=closure, owner_contract=owner)
    assert result.checkpoint.path.exists()
    assert result.receipt_path.exists()
    payload = checkpoint.load_checkpoint(result.checkpoint.path, map_location="cpu")
    assert payload["extra"]["r241_own_deck_successor_migration"]["runtime_routes_enabled"] is False
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["source_closure"]["source_tree_sha256"] == closure["source_tree_sha256"]


def test_rejects_partial_side_store_even_with_correct_source(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "r195.pt")
    owner = _owner(tmp_path / "owner.json", parent)
    binding = {"schema": "poke_bot.r241_own_deck_sidecar_binding/v1", "status": "complete_training_eligible", "owner_contract_sha256": owner.sha256, "source_manifest_sha256": owner.source_manifest_sha256, "source_window_receipt_sha256": owner.source_window_receipt_sha256, "day_count": 14, "validated_episode_count": 91253, "source_archive_bytes": 14842033482, "daily_sidecar_meta_receipt_sha256s": {}, "partial_or_unreceipted_side_store_training_eligible": False}
    binding["binding_sha256"] = "sha256:" + hashlib.sha256((json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    with pytest.raises(R241OwnDeckSuccessorError):
        validate_r260_sidecar_binding(binding, owner_contract=owner)
