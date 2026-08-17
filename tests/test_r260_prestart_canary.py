"""Focused offline tests for the r260 pre-start canary artifact boundary."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest
import torch

from poke_bot import checkpoint
from poke_bot import own_deck_migration as migration
from poke_bot.own_deck_successor import canonical_json, seal_receipt
from poke_bot.r241_own_deck_successor import (
    R260_MIGRATION_KIND,
    R260_MIGRATION_SCHEMA,
    FileIdentity,
    R241OwnDeckSuccessorError,
    R260OwnerContract,
    validate_r260_canary_activation,
)
from poke_bot.r260_prestart_canary import (
    RUNTIME_GATE_FIELDS,
    CanaryStep,
    R260PrestartCanaryError,
    create_r260_bounded_influence_receipt,
    create_r260_canary_activation_receipt,
    create_r260_local_elmo_replay_parity_receipt,
    create_r260_runtime_activation_config,
    create_r260_source_disjoint_evaluation_receipt,
    file_identity,
    prepare_r260_prestart_canary_config,
    run_bounded_deterministic_expert_canary,
    validate_r260_runtime_activation_config,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


class _ToyCanaryModel(torch.nn.Module):
    """Small model with the exact r260 tensor-prefix surface."""

    def __init__(self) -> None:
        super().__init__()
        self.own_deck_ledger_adapter = torch.nn.Linear(2, 2)
        self.own_deck_ledger_option_adapter = torch.nn.Linear(2, 2)
        self.visible_tutor_completion_head = torch.nn.Linear(2, 2)
        self.terminal_conversion_head = torch.nn.Linear(2, 2)
        self.tactical_sequence_outcome_head = torch.nn.Linear(2, 2)
        self.tactical_sequence_outcome_route = torch.nn.Linear(2, 2)
        self.visible_tutor_completion_route = torch.nn.Linear(2, 2)
        self.terminal_conversion_route = torch.nn.Linear(2, 2)
        self.inherited_route = torch.nn.Linear(2, 2)

    def loss(self) -> torch.Tensor:
        sample = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        paths = (
            self.own_deck_ledger_adapter(sample),
            self.own_deck_ledger_option_adapter(sample),
            self.visible_tutor_completion_head(sample),
            self.terminal_conversion_head(sample),
            self.tactical_sequence_outcome_head(sample),
            self.tactical_sequence_outcome_route(sample),
            self.visible_tutor_completion_route(sample),
            self.terminal_conversion_route(sample),
            self.inherited_route(sample),
        )
        return sum(value.square().mean() for value in paths)


def _contract(tmp_path: Path) -> R260OwnerContract:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"immutable r195 parent")
    parent_identity = file_identity(parent)
    inzi_root = tmp_path / "inzi-final"
    inzi_root.mkdir()
    return R260OwnerContract(
        path=tmp_path / "owner-contract.json",
        sha256=_sha("a"),
        parent=FileIdentity(
            path=parent,
            sha256=parent_identity["sha256"],
            size_bytes=parent_identity["size_bytes"],
        ),
        source_manifest_sha256=_sha("b"),
        source_window_receipt_sha256=_sha("c"),
        side_store_root="/elmo-side-store",
        inzi_training_root=str(inzi_root),
        inzi_prefix_staging_root=str(tmp_path / "inzi-final-staging-09848f04"),
    )


def _child_checkpoint(tmp_path: Path) -> tuple[_ToyCanaryModel, Path]:
    model = _ToyCanaryModel()
    path = tmp_path / "migration-child.pt"
    checkpoint.atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "own_deck_ledger_enabled": True,
                "visible_tutor_completion_head_enabled": True,
                "terminal_conversion_head_enabled": True,
                "tactical_sequence_outcome_head_enabled": True,
                "visible_tutor_completion_route_enabled": True,
                "terminal_conversion_route_enabled": True,
                "tactical_sequence_outcome_route_enabled": False,
                **{field: False for field in RUNTIME_GATE_FIELDS},
            },
            "extra": {},
            "provenance": {},
        },
        path,
    )
    return model, path


def _migration_receipt(contract: R260OwnerContract, child: Path) -> dict[str, object]:
    child_identity = file_identity(child)
    return seal_receipt(
        {
            "schema": R260_MIGRATION_SCHEMA,
            "kind": R260_MIGRATION_KIND,
            "status": "passed",
            "owner_contract": {"path": str(contract.path), "sha256": contract.sha256},
            "parent_checkpoint": contract.parent.as_dict(),
            "child_checkpoint": child_identity,
            "verification": {
                "zero_safe_final_projection_keys": list(
                    migration.ZERO_SAFE_FINAL_PROJECTION_KEYS
                ),
                "parent_behavior_exact_for_absent_and_valid_public_ledger": True,
            },
            "runtime_authority": {
                "own_deck_ledger_runtime_enabled": False,
                "visible_tutor_completion_route_runtime_enabled": False,
                "terminal_conversion_route_runtime_enabled": False,
                "tactical_sequence_outcome_route_runtime_enabled": False,
                "selector_change_authorized": False,
                "serving_eligible": False,
            },
        }
    )


def _binding_digest(value: dict[str, object], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return "sha256:" + hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _identity(path: str, character: str) -> dict[str, object]:
    return {"path": path, "sha256": _sha(character), "size_bytes": 1}


def _sidecar_binding(contract: R260OwnerContract) -> dict[str, object]:
    first = date(2026, 7, 22)
    daily = {
        (first + timedelta(days=index)).isoformat(): _identity(
            f"/receipts/day-{index}.json", f"{index:x}"[-1]
        )
        for index in range(20)
    }
    # The strict validator needs each daily digest to be distinct.  Hex digits
    # repeat after sixteen values, so replace the final four with other values.
    for index, day in enumerate(sorted(daily)):
        daily[day]["sha256"] = "sha256:" + f"{index:064x}"
    terminal = {
        name: _identity(f"/receipts/{name}.json", f"{index + 20:064x}"[-1])
        for index, name in enumerate(
            (
                "completion",
                "join",
                "schema",
                "count",
                "digest",
                "causal_local_remote_parity",
            )
        )
    }
    for index, name in enumerate(sorted(terminal)):
        terminal[name]["sha256"] = "sha256:" + f"{index + 40:064x}"
    value: dict[str, object] = {
        "schema": "poke_bot.r241_own_deck_sidecar_binding/v1",
        "status": "complete_training_eligible",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "day_count": 20,
        "validated_episode_count": 91253,
        "source_archive_bytes": 14842033482,
        "daily_sidecar_meta_receipts": daily,
        "terminal_receipts": terminal,
        "daily_build_identity": {
            "sidecar_build_code_sha256": _sha("d"),
            "source_snapshot_tree_sha256": _sha("e"),
            "container_image_id": _sha("f"),
            "archive_native_classifier_sha256": _sha("1"),
        },
        "partial_or_unreceipted_side_store_training_eligible": False,
    }
    value["binding_sha256"] = _binding_digest(value, "binding_sha256")
    return value


def _inzi_binding(
    contract: R260OwnerContract, sidecar: dict[str, object]
) -> dict[str, object]:
    joined = Path(contract.inzi_training_root) / "joined" / "ledger.jsonl.gz"
    joined.parent.mkdir()
    joined.write_bytes(b"sealed Inzi joined dataset")
    joined.chmod(0o444)
    joined_identity = file_identity(joined, immutable=True)
    value: dict[str, object] = {
        "schema": "poke_bot.r241_own_deck_inzi_dataset_binding/v2",
        "status": "complete_transport_ready",
        "owner_contract_sha256": contract.sha256,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "sidecar_binding_sha256": sidecar["binding_sha256"],
        # The aggregate's semantic content digest is intentionally distinct
        # from the SHA-256 of the immutable JSON file that carries it.
        "sidecar_binding_file_sha256": _sha("2"),
        "sidecar_binding": _identity("/receipts/aggregate.json", "2"),
        "transport_kind": "create_only_copy",
        "source_join_identity_kind": "elmo_materialized_joined_dataset",
        "source_joined_dataset_sha256": joined_identity["sha256"],
        "inzi_joined_dataset_sha256": joined_identity["sha256"],
        "join_receipt_sha256": _sha("4"),
        "schema_receipt_sha256": _sha("5"),
        "count_receipt_sha256": _sha("6"),
        "digest_receipt_sha256": _sha("7"),
        "causal_local_remote_parity_receipt_sha256": _sha("8"),
        "transport_receipt_sha256": _sha("9"),
        "day_count": 20,
        "validated_episode_count": 91253,
        "source_archive_bytes": 14842033482,
        "inzi_joined_dataset": joined_identity,
        "inzi_sidecar_root": contract.inzi_training_root,
    }
    value["binding_sha256"] = _binding_digest(value, "binding_sha256")
    return value


class _ToyInziStreamingIndex:
    def __init__(
        self,
        path: Path,
        *,
        source_manifest_sha256: str,
        daily_meta_sha256s: dict[str, str],
    ) -> None:
        self.path = path
        self.source_manifest_sha256 = source_manifest_sha256
        self.daily_meta_sha256s = daily_meta_sha256s
        self.calls = 0

    def assert_verified(
        self,
        *,
        expected_source_manifest_sha256: str,
        daily_meta_sha256s: dict[str, str],
    ) -> None:
        self.calls += 1
        if (
            expected_source_manifest_sha256 != self.source_manifest_sha256
            or daily_meta_sha256s != self.daily_meta_sha256s
        ):
            raise RuntimeError("Inzi index provenance mismatch")


def _inzi_streaming_index(
    contract: R260OwnerContract, sidecar: dict[str, object]
) -> tuple[_ToyInziStreamingIndex, dict[str, object], dict[str, object]]:
    daily = {
        str(day): str(row["sha256"])
        for day, row in dict(sidecar["daily_sidecar_meta_receipts"]).items()
    }
    path = Path(contract.inzi_training_root) / "r260-four-key-index.sqlite3"
    path.write_bytes(b"sealed immutable four-key Inzi index")
    path.chmod(0o444)
    identity = file_identity(path, immutable=True)
    provenance: dict[str, object] = {
        "schema": "poke_bot.r260_inzi_sidecar_index/v1",
        "source_manifest_sha256": contract.source_manifest_sha256,
        "daily_meta_sha256s": daily,
    }
    return (
        _ToyInziStreamingIndex(
            path,
            source_manifest_sha256=contract.source_manifest_sha256,
            daily_meta_sha256s=daily,
        ),
        identity,
        provenance,
    )


def _coverage() -> dict[str, int]:
    return {
        "public_rows": 2,
        "ledger_rows": 2,
        "visible_tutor_labeled_rows": 1,
        "visible_tutor_masked_rows": 1,
        "terminal_labeled_rows": 1,
        "terminal_masked_rows": 1,
    }


def _calibration() -> dict[str, float | int]:
    return {
        "visible_tutor_brier_sum": 0.25,
        "visible_tutor_brier_count": 1,
        "terminal_brier_sum": 0.5,
        "terminal_brier_count": 1,
        "terminal_ece_sum": 0.1,
        "terminal_ece_count": 1,
    }


def _step(
    model: torch.nn.Module, _index: int, inzi_streaming_index: _ToyInziStreamingIndex
) -> CanaryStep:
    assert isinstance(model, _ToyCanaryModel)
    assert inzi_streaming_index.path.is_file()
    return CanaryStep(
        loss=model.loss(),
        source_ids=("train-a", "train-b"),
        coverage=_coverage(),
        calibration=_calibration(),
        public_information_only=True,
        direct_policy_only=True,
        no_search_or_rtp=True,
        no_hidden_state=True,
    )


def test_full_r260_prestart_canary_chain_is_create_only_and_overlay_gated(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    model, child = _child_checkpoint(tmp_path)
    migration_receipt = _migration_receipt(contract, child)
    sidecar = _sidecar_binding(contract)
    inzi = _inzi_binding(contract, sidecar)
    inzi_index, inzi_index_identity, inzi_index_provenance = _inzi_streaming_index(
        contract, sidecar
    )
    config_path = tmp_path / "canary-config.json"
    _config = prepare_r260_prestart_canary_config(
        migration_receipt=migration_receipt,
        sidecar_binding=sidecar,
        inzi_dataset_binding=inzi,
        owner_contract=contract,
        training_source_ids=("train-a", "train-b"),
        evaluation_source_ids=("heldout-a",),
        inherited_route_prefixes=("inherited_route.",),
        inzi_streaming_index=inzi_index_identity,
        inzi_streaming_index_provenance=inzi_index_provenance,
        output_path=config_path,
        max_steps=2,
    )
    assert config_path.stat().st_mode & 0o222 == 0
    result = run_bounded_deterministic_expert_canary(
        canary_config=config_path,
        migration_receipt=migration_receipt,
        owner_contract=contract,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        step_builder=_step,
        inzi_streaming_index=inzi_index,
        migration_child_checkpoint=child,
        output_checkpoint=tmp_path / "canary.pt",
        output_receipt=tmp_path / "canary.json",
    )
    assert result.executed_steps == 2
    assert inzi_index.calls == 1
    payload = checkpoint.load_checkpoint(result.checkpoint["path"], map_location="cpu")
    assert all(payload["model_config"][field] is True for field in RUNTIME_GATE_FIELDS)
    assert result.checkpoint["sha256"] != file_identity(child)["sha256"]
    evaluation = create_r260_source_disjoint_evaluation_receipt(
        canary_config=config_path,
        canary_receipt=tmp_path / "canary.json",
        owner_contract=contract,
        evaluation_source_ids=("heldout-a",),
        coverage=_coverage(),
        calibration=_calibration(),
        output_path=tmp_path / "evaluation.json",
    )
    assert evaluation["status"] == "passed_source_disjoint_factual_evaluation"
    parity = create_r260_local_elmo_replay_parity_receipt(
        canary_config=config_path,
        canary_receipt=tmp_path / "canary.json",
        owner_contract=contract,
        local_feature_digests={"row-1": _sha("1"), "row-2": _sha("2")},
        elmo_feature_digests={"row-1": _sha("1"), "row-2": _sha("2")},
        replay_feature_digests={"row-1": _sha("1"), "row-2": _sha("2")},
        output_path=tmp_path / "parity.json",
    )
    assert parity["status"] == "passed_local_elmo_replay_parity"
    influence = create_r260_bounded_influence_receipt(
        canary_config=config_path,
        canary_receipt=tmp_path / "canary.json",
        owner_contract=contract,
        baseline_policy_logits=torch.tensor([[0.0, 0.0]]),
        runtime_policy_logits=torch.tensor([[0.25, 0.0]]),
        output_path=tmp_path / "influence.json",
    )
    assert influence["max_abs_logit_delta"] == pytest.approx(0.25)
    activation_config_path = tmp_path / "runtime-config.json"
    activation_config = create_r260_runtime_activation_config(
        canary_config=config_path,
        canary_receipt=tmp_path / "canary.json",
        evaluation_receipt=tmp_path / "evaluation.json",
        parity_receipt=tmp_path / "parity.json",
        influence_receipt=tmp_path / "influence.json",
        owner_contract=contract,
        output_path=activation_config_path,
    )
    validated = validate_r260_runtime_activation_config(
        activation_config_path,
        canary_config=config_path,
        owner_contract=contract,
        verify_files=True,
    )
    assert validated == activation_config
    assert all(
        activation_config["runtime_gates"][field] is True
        for field in RUNTIME_GATE_FIELDS
    )
    activation = create_r260_canary_activation_receipt(
        runtime_activation_config=activation_config_path,
        canary_config=config_path,
        owner_contract=contract,
        output_path=tmp_path / "activation.json",
    )
    assert set(activation) == {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "migration_receipt_sha256",
        "canary_checkpoint",
        "evidence_receipts",
        "runtime_gates",
    }
    validate_r260_canary_activation(
        activation,
        migration_receipt=migration_receipt,
        owner_contract=contract,
        require_local_checkpoint=True,
    )
    forged = dict(activation)
    forged["canary_checkpoint"] = dict(migration_receipt["child_checkpoint"])
    forged.pop("receipt_sha256")
    forged = seal_receipt(forged)
    with pytest.raises(R241OwnDeckSuccessorError, match="zero-safe migration child"):
        validate_r260_canary_activation(
            forged,
            migration_receipt=migration_receipt,
            owner_contract=contract,
        )


def test_canary_rejects_search_and_never_publishes_a_checkpoint(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    model, child = _child_checkpoint(tmp_path)
    migration_receipt = _migration_receipt(contract, child)
    sidecar = _sidecar_binding(contract)
    inzi = _inzi_binding(contract, sidecar)
    inzi_index, inzi_index_identity, inzi_index_provenance = _inzi_streaming_index(
        contract, sidecar
    )
    config_path = tmp_path / "config.json"
    prepare_r260_prestart_canary_config(
        migration_receipt=migration_receipt,
        sidecar_binding=sidecar,
        inzi_dataset_binding=inzi,
        owner_contract=contract,
        training_source_ids=("train-a", "train-b"),
        evaluation_source_ids=("heldout-a",),
        inherited_route_prefixes=("inherited_route.",),
        inzi_streaming_index=inzi_index_identity,
        inzi_streaming_index_provenance=inzi_index_provenance,
        output_path=config_path,
        max_steps=2,
    )

    def illegal_step(
        current: torch.nn.Module,
        index: int,
        current_inzi_index: _ToyInziStreamingIndex,
    ) -> CanaryStep:
        step = _step(current, index, current_inzi_index)
        return replace(step, no_search_or_rtp=False)

    output = tmp_path / "should-not-exist.pt"
    with pytest.raises(R260PrestartCanaryError, match="public direct-policy boundary"):
        run_bounded_deterministic_expert_canary(
            canary_config=config_path,
            migration_receipt=migration_receipt,
            owner_contract=contract,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            step_builder=illegal_step,
            inzi_streaming_index=inzi_index,
            migration_child_checkpoint=child,
            output_checkpoint=output,
            output_receipt=tmp_path / "should-not-exist.json",
        )
    assert not output.exists()


def test_canary_rehashes_and_rejects_bad_inzi_index_before_checkpoint(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    model, child = _child_checkpoint(tmp_path)
    migration_receipt = _migration_receipt(contract, child)
    sidecar = _sidecar_binding(contract)
    inzi = _inzi_binding(contract, sidecar)
    index, index_identity, index_provenance = _inzi_streaming_index(contract, sidecar)
    config_path = tmp_path / "config.json"
    prepare_r260_prestart_canary_config(
        migration_receipt=migration_receipt,
        sidecar_binding=sidecar,
        inzi_dataset_binding=inzi,
        owner_contract=contract,
        training_source_ids=("train-a", "train-b"),
        evaluation_source_ids=("heldout-a",),
        inherited_route_prefixes=("inherited_route.",),
        inzi_streaming_index=index_identity,
        inzi_streaming_index_provenance=index_provenance,
        output_path=config_path,
        max_steps=2,
    )
    bad_index = _ToyInziStreamingIndex(
        index.path,
        source_manifest_sha256=_sha("f"),
        daily_meta_sha256s=index.daily_meta_sha256s,
    )
    output = tmp_path / "should-not-exist.pt"
    receipt = tmp_path / "should-not-exist.json"
    with pytest.raises(R260PrestartCanaryError, match="index provenance failed"):
        run_bounded_deterministic_expert_canary(
            canary_config=config_path,
            migration_receipt=migration_receipt,
            owner_contract=contract,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            step_builder=_step,
            inzi_streaming_index=bad_index,
            migration_child_checkpoint=child,
            output_checkpoint=output,
            output_receipt=receipt,
        )
    assert not output.exists()
    assert not receipt.exists()


def test_config_rejects_overlapping_train_and_evaluation_sources(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    _model, child = _child_checkpoint(tmp_path)
    sidecar = _sidecar_binding(contract)
    _inzi_index, inzi_index_identity, inzi_index_provenance = _inzi_streaming_index(
        contract, sidecar
    )
    with pytest.raises(R260PrestartCanaryError, match="sources overlap"):
        prepare_r260_prestart_canary_config(
            migration_receipt=_migration_receipt(contract, child),
            sidecar_binding=sidecar,
            inzi_dataset_binding=_inzi_binding(contract, sidecar),
            owner_contract=contract,
            training_source_ids=("shared",),
            evaluation_source_ids=("shared",),
            inherited_route_prefixes=("inherited_route.",),
            inzi_streaming_index=inzi_index_identity,
            inzi_streaming_index_provenance=inzi_index_provenance,
            output_path=tmp_path / "no-config.json",
            max_steps=2,
        )


@pytest.mark.parametrize(
    ("bad_path", "error"),
    (
        ("staging", "prefix-staging"),
        ("/mnt/Main/elmo/index.sqlite3", "Elmo /mnt/Main"),
    ),
)
def test_config_rejects_nonfinal_inzi_streaming_index(
    tmp_path: Path, bad_path: str, error: str
) -> None:
    contract = _contract(tmp_path)
    _model, child = _child_checkpoint(tmp_path)
    sidecar = _sidecar_binding(contract)
    inzi = _inzi_binding(contract, sidecar)
    _index, identity, provenance = _inzi_streaming_index(contract, sidecar)
    forged_index = dict(identity)
    forged_index["path"] = (
        str(Path(contract.inzi_prefix_staging_root) / "index.sqlite3")
        if bad_path == "staging"
        else bad_path
    )
    with pytest.raises(R260PrestartCanaryError, match=error):
        prepare_r260_prestart_canary_config(
            migration_receipt=_migration_receipt(contract, child),
            sidecar_binding=sidecar,
            inzi_dataset_binding=inzi,
            owner_contract=contract,
            training_source_ids=("train-a",),
            evaluation_source_ids=("heldout-a",),
            inherited_route_prefixes=("inherited_route.",),
            inzi_streaming_index=forged_index,
            inzi_streaming_index_provenance=provenance,
            output_path=tmp_path / "nonfinal-config.json",
            max_steps=2,
        )
