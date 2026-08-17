from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poke_bot import config, features
from poke_bot.model import EXPANDED_HEAD_NAMES, build_model
from poke_bot.strategic_schedule import EXPANDED_HEAD_IDS
from poke_bot.train import (
    R241_PEAK_R195_ACTIVE_EXPANDED_MODULES,
    R241_PEAK_R195_ACTIVE_SOURCES,
    R241_PEAK_R195_EXPANDED_MODULES,
    R241_PEAK_R195_LIVE_FUSION_SCHEMA,
    R241_PEAK_R195_PHYSICAL_SOURCES,
    R241_PEAK_R195_TRAINING_CONTRACT_SCHEMA,
    canonical_r241_peak_r195_training_contract,
    validate_r241_peak_r195_live_fusion_record,
)
from scripts import train_pure_rl


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _options(count: int) -> features.SparseVector:
    result = features.SparseVector()
    for index in range(count):
        result.word_start()
        result.add(index + 1, 1.0)
    return result


def _contract() -> dict:
    ordinary = {
        "value": 1.0,
        "archetype": 0.05,
        "opponent_hand": 0.05,
        "opponent_remainder": 0.05,
        "lethal_threat": 0.025,
        "prize_race": 0.025,
        "setup_board_outcome": 0.025,
        "combo_state": 0.0,
        "guide": 0.05,
    }
    return {
        "schema": R241_PEAK_R195_TRAINING_CONTRACT_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "owner_decision_revision": 241,
        "owner_clarification_revision": 251,
        "fixed_cycle_updates": 10,
        "anchor_parent_sha256": _sha("a"),
        "serialized_model_config_sha256": _sha("b"),
        "serialized_state_schema_sha256": _sha("c"),
        "trainable_parameter_count": 10_750_146,
        "preservation_receipt": {
            "path": "/immutable/peak-r195-preservation-v6.json",
            "digest": _sha("d"),
            "size_bytes": 100,
        },
        "owner_contract": {
            "path": "/immutable/alakazam-new-list-direct-policy-r241.json",
            "digest": "sha256:2f9ca8fc0d4cb2a7c6acbc12ecce3e96143a2c9e318e198276ea0dd66bb30c7d",
            "size_bytes": 103,
        },
        "head_role_map": {
            "path": "/immutable/head-role-map.json",
            "digest": _sha("e"),
            "size_bytes": 101,
        },
        "source_snapshot": {
            "schema": "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1",
            "status": "authenticated_immutable_source_snapshot",
            "authenticated": True,
            "root": "/immutable/r241-source",
            "source_execution_root": "/immutable/r241-source",
            "manifest": "/immutable/r241-source/r241-source-snapshot-manifest.json",
            "manifest_sha256": _sha("f"),
            "source_tree_sha256": _sha("1"),
            "file_inventory_sha256": _sha("2"),
            "owner_contract_sha256": "sha256:2f9ca8fc0d4cb2a7c6acbc12ecce3e96143a2c9e318e198276ea0dd66bb30c7d",
            "outputs_root": "/external/outputs",
            "host": "inzi",
        },
        "baseline_adapter_roster": {
            "path": "/immutable/matchup_adapter_roster.json",
            "digest": _sha("3"),
            "size_bytes": 102,
        },
        "adapter_slot_migration": {
            "schema": "poke_bot.alakazam_new_list_direct_policy_r241_adapter_slot_migration/v1",
            "status": "no_slot_change",
            "parent_slot_registry_sha256": _sha("4"),
            "candidate_slot_registry_sha256": _sha("4"),
            "retained_slot_count": 20,
            "retained_slot_tensor_inventory_sha256": _sha("5"),
            "existing_slots_byte_immutable": True,
            "new_slots": [],
            "new_slot_proofs": [],
        },
        "checkpoint_audit_fingerprint_sha256": _sha("6"),
        "physical_source_names": list(R241_PEAK_R195_PHYSICAL_SOURCES),
        "active_non_combo_source_names": list(R241_PEAK_R195_ACTIVE_SOURCES),
        "disabled_source_names": ["combo_state"],
        "combo_state_loss_weight": 0.0,
        "combo_state_fusion_route_enabled": False,
        "ordinary_rl_loss_profile": ordinary,
        "expert_refresh_loss_profile": {**ordinary, "guide": 0.0},
        "expanded_head_loss_weights": {
            name: 0.01 for name in EXPANDED_HEAD_IDS
        },
        "matchup_adapter": {
            "isolated_fit": {
                "serialized_runtime_enabled": False,
                "serialized_training_enabled": False,
                "main_optimizer_included": False,
                "base_frozen_during_fit": True,
                "optimizer_scope": "matchup_adapter_bank_only",
                "continuation_state_required": True,
            },
            "external_runtime_activation": {
                "collection_enabled": True,
                "terminal_package_enabled": True,
                "public_tree_sha256": _sha("7"),
                "activation_receipt_sha256": _sha("8"),
            },
        },
    }


def _sidecar(contract: dict, *, phase: str = "rl", boundary: int = 0) -> dict:
    return {
        "schema": R241_PEAK_R195_LIVE_FUSION_SCHEMA,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "owner_decision_revision": 241,
        "owner_clarification_revision": 251,
        "fixed_cycle_updates": 10,
        "phase": phase,
        "boundary_iteration": boundary,
        "anchor_parent_sha256": contract["anchor_parent_sha256"],
        "preservation_receipt": copy.deepcopy(contract["preservation_receipt"]),
        "owner_contract": copy.deepcopy(contract["owner_contract"]),
        "head_role_map": copy.deepcopy(contract["head_role_map"]),
        "source_snapshot": copy.deepcopy(contract["source_snapshot"]),
        "baseline_adapter_roster": copy.deepcopy(
            contract["baseline_adapter_roster"]
        ),
        "adapter_slot_migration": copy.deepcopy(contract["adapter_slot_migration"]),
        "checkpoint_audit_fingerprint_sha256": contract[
            "checkpoint_audit_fingerprint_sha256"
        ],
        "physical_source_names": list(R241_PEAK_R195_PHYSICAL_SOURCES),
        "active_non_combo_source_names": list(R241_PEAK_R195_ACTIVE_SOURCES),
        "disabled_source_names": ["combo_state"],
        "physical_head_count": 19,
        "active_non_combo_head_count": 18,
        "combo_state": {
            "head_present": True,
            "loss_weight": 0.0,
            "fusion_route_enabled": False,
        },
        "expanded_head_inventory": {
            "runtime_enabled_heads": list(R241_PEAK_R195_ACTIVE_EXPANDED_MODULES),
            "runtime_disabled_heads": ["combo_state_head"],
            "modules": {name: {} for name in R241_PEAK_R195_EXPANDED_MODULES},
        },
        "decision_fusion_inventory": {
            "schema": "poke_bot.causal_decision_fusion/v3",
            "runtime_enabled": True,
            "required_heads": list(R241_PEAK_R195_PHYSICAL_SOURCES),
            "active_required_heads": list(R241_PEAK_R195_ACTIVE_SOURCES),
            "dedicated_routes": {
                "active_route_names": list(R241_PEAK_R195_ACTIVE_SOURCES),
                "disabled_route_names": ["combo_state"],
            },
        },
        "ordinary_optimizer": {
            "all_active_source_parameters_included": True,
            "all_active_route_parameters_included": True,
            "source_parameter_counts": {
                name: 1 for name in R241_PEAK_R195_ACTIVE_SOURCES
            },
            "route_parameter_counts": {
                name: 1 for name in R241_PEAK_R195_ACTIVE_SOURCES
            },
        },
        "matchup_adapter_training_checkpoint": {
            "runtime_enabled": False,
            "training_enabled": False,
            "main_optimizer_included": False,
            "isolated_fit_preserved": True,
        },
        "external_matchup_adapter_runtime_activation": copy.deepcopy(
            contract["matchup_adapter"]["external_runtime_activation"]
        ),
        "fused_policy_authoritative": True,
        "guide_training_only": True,
    }


def test_r241_training_contract_is_r251_v2_derived_and_no_slot_change() -> None:
    contract = _contract()

    assert canonical_r241_peak_r195_training_contract(contract) == contract

    stale = copy.deepcopy(contract)
    stale["owner_clarification_revision"] = 245
    with pytest.raises(ValueError, match="owner_clarification_revision"):
        canonical_r241_peak_r195_training_contract(stale)

    migrated = copy.deepcopy(contract)
    migrated["adapter_slot_migration"]["status"] = "append_only_slot_addition"
    migrated["adapter_slot_migration"]["new_slots"] = [{"slot": 20}]
    with pytest.raises(ValueError, match="no-slot-change"):
        canonical_r241_peak_r195_training_contract(migrated)


def test_r241_combo_off_policy_gradient_reaches_all_18_live_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is a model-gradient unit test, not a simulator integration test.
    # Keep it runnable on CI hosts that intentionally do not mount libcg.
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 32)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 16)
    model = build_model(
        config.ModelConfig(
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
            expanded_heads_enabled=True,
            setup_board_outcome_head_enabled=True,
            combo_state_head_enabled=True,
            combo_state_route_enabled=False,
            decision_fusion_enabled=True,
            decision_fusion_runtime_enabled=True,
            decision_fusion_dedicated_routes_enabled=True,
            decision_fusion_dedicated_routes_runtime_enabled=True,
            decision_fusion_typed_output_centered_routes_enabled=True,
            decision_fusion_action_type_reliability_cap=0.25,
            decision_fusion_width=8,
            dropout=0.0,
        ),
        aux_archetype_classes=4,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=32,
    )
    fusion = model.decision_fusion
    assert fusion is not None
    assert tuple(fusion.required_heads) == R241_PEAK_R195_PHYSICAL_SOURCES
    with torch.no_grad():
        for route in fusion.dedicated_routes.values():
            route.network[-1].weight.fill_(0.05)
            route.network[-1].bias.zero_()

    model.train()
    spatial = torch.randn(4, features.NUM_BOARD_TOKENS, model.d_model)
    state = torch.randn(4, model.d_model)
    logits = model.decode_options(
        [_options(4), _options(4), _options(4), _options(4)],
        spatial,
        state,
        n_options=[4, 4, 4, 4],
    )
    torch.nn.functional.cross_entropy(
        logits,
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    ).backward()

    active_modules = (
        model.value_head,
        model.aux_head,
        model.opp_hand_head,
        model.opp_remainder_head,
        model.lethal_threat_head,
        model.prize_race_head,
        *(getattr(model, name) for name in EXPANDED_HEAD_NAMES),
        model.setup_board_outcome_head,
    )
    assert len(active_modules) == 18
    for module in active_modules:
        assert module is not None
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in module.parameters()
        )
    assert model.combo_state_head is not None
    assert all(parameter.grad is None for parameter in model.combo_state_head.parameters())
    for name in R241_PEAK_R195_ACTIVE_SOURCES:
        assert any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
            for parameter in fusion.dedicated_routes[name].parameters()
        ), name
    assert all(
        parameter.grad is None
        for parameter in fusion.dedicated_routes["combo_state"].parameters()
    )


def test_r241_live_fusion_sidecar_binds_source_roster_and_all_18_routes() -> None:
    contract = _contract()
    sidecar = _sidecar(contract, phase="expert_refresh", boundary=10)

    assert validate_r241_peak_r195_live_fusion_record(
        sidecar,
        contract=contract,
        phase="expert_refresh",
        boundary_iteration=10,
    ) == sidecar

    drifted = copy.deepcopy(sidecar)
    drifted["source_snapshot"]["source_tree_sha256"] = _sha("9")
    with pytest.raises(ValueError, match="source_snapshot"):
        validate_r241_peak_r195_live_fusion_record(drifted, contract=contract)

    missing_route = copy.deepcopy(sidecar)
    missing_route["ordinary_optimizer"]["route_parameter_counts"].pop("value")
    with pytest.raises(ValueError, match="optimizer proof"):
        validate_r241_peak_r195_live_fusion_record(missing_route, contract=contract)


def test_r241_completion_revalidates_the_boundary_five_live_fusion_receipt(
    tmp_path: Path,
) -> None:
    contract = _contract()
    commits = tmp_path / "commits"
    rehearsals = tmp_path / "rehearsals"
    commits.mkdir()
    rehearsals.mkdir()
    history = []
    for iteration in range(10):
        history.append({"iteration": iteration, "completed": True})
        (commits / f"iter_{iteration:05d}.json").write_text(
            json.dumps(
                {
                    "last_completed_iteration": iteration,
                    "next_iteration": iteration + 1,
                }
            ),
            encoding="utf-8",
        )
    refresh_path = rehearsals / "before_iter_00005.json"
    refresh = {
        "before_iteration": 5,
        "epochs": 5,
        "peak_r195_live_fusion": _sidecar(
            contract,
            phase="expert_refresh",
            boundary=5,
        ),
    }
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")
    collection_path = train_pure_rl._collection_receipt_path(tmp_path, 5)
    collection_path.parent.mkdir(parents=True)
    collection_path.write_text(
        json.dumps({"iteration": 5, "checkpoint_digest": _sha("7")}),
        encoding="utf-8",
    )
    refresh["checkpoint_digest"] = _sha("7")
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")
    state = {
        "last_completed_iteration": 9,
        "next_iteration": 10,
        "history": history,
    }

    assert train_pure_rl._assert_fixed_cycle_completion(
        tmp_path,
        state,
        updates=10,
        r241_peak_r195_training_contract=contract,
    )["updates_completed"] == 10

    refresh["peak_r195_live_fusion"]["owner_contract"]["digest"] = _sha("9")
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")
    with pytest.raises(ValueError, match="owner_contract"):
        train_pure_rl._assert_fixed_cycle_completion(
            tmp_path,
            state,
            updates=10,
            r241_peak_r195_training_contract=contract,
        )


def test_r241_iter_five_collection_is_bound_to_refreshed_checkpoint() -> None:
    contract = _contract()
    refresh = {
        "before_iteration": 5,
        "epochs": 5,
        "checkpoint_digest": _sha("7"),
        "checkpoint_identity": {"digest": _sha("7")},
        "peak_r195_live_fusion": _sidecar(
            contract,
            phase="expert_refresh",
            boundary=5,
        ),
    }

    train_pure_rl._assert_r241_precollection_refresh_binding(
        iteration=5,
        collection_behavior_digest=_sha("7"),
        rehearsal_record=refresh,
        r241_peak_r195_training_contract=contract,
    )
    with pytest.raises(RuntimeError, match="exact checkpoint"):
        train_pure_rl._assert_r241_precollection_refresh_binding(
            iteration=5,
            collection_behavior_digest=_sha("8"),
            rehearsal_record=refresh,
            r241_peak_r195_training_contract=contract,
        )


def test_r241_initial_learner_override_must_match_sealed_anchor_bytes(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.pt"
    matching = tmp_path / "matching.pt"
    drifted = tmp_path / "drifted.pt"
    parent.write_bytes(b"sealed-r195-anchor")
    matching.write_bytes(parent.read_bytes())
    drifted.write_bytes(b"later-r241-successor")
    sealed = train_pure_rl._r241_file_identity(parent)

    assert (
        train_pure_rl._assert_r241_initial_learner_anchor(
            None,
            sealed_parent=sealed,
        )
        is None
    )
    assert train_pure_rl._assert_r241_initial_learner_anchor(
        matching,
        sealed_parent=sealed,
    )["digest"] == sealed["digest"]
    with pytest.raises(RuntimeError, match="sealed r195 anchor"):
        train_pure_rl._assert_r241_initial_learner_anchor(
            drifted,
            sealed_parent=sealed,
        )


def test_r241_loader_rejects_self_asserted_v1_before_checkpoint_access(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "peak-r195-v1.json"
    receipt.write_text(
        '{"schema":"poke_bot.alakazam_new_list_direct_policy_r241_peak_r195_preservation/v1"}\n',
        encoding="utf-8",
    )
    digest = "sha256:" + hashlib.sha256(receipt.read_bytes()).hexdigest()
    args = SimpleNamespace(
        fixed_cycle_updates=10,
        r241_peak_r195_preservation_receipt=receipt,
        r241_peak_r195_preservation_receipt_sha256=digest,
    )

    with pytest.raises(RuntimeError, match="v2/r251"):
        train_pure_rl._load_r241_peak_r195_training_contract(args)
