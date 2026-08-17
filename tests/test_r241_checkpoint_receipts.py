"""Focused fail-closed coverage for r241 checkpoint-derived receipt boundaries."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from poke_bot import config
from poke_bot import r241_checkpoint_receipts as receipts


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _h10_source_binding_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    source_root = tmp_path / "alakazam-new-list-direct-r241-src-active"
    source_root.mkdir()
    source_manifest = source_root / "r241-source-snapshot-manifest.json"
    _write_json(source_manifest, {"source": "active"})
    source_snapshot = {
        "root": str(source_root),
        "manifest": str(source_manifest),
        "manifest_sha256": receipts.sha256_file(source_manifest),
    }
    offline_preflight = {
        "source_snapshot_root": str(source_root),
        "source_snapshot_manifest": str(source_manifest),
        "source_snapshot_manifest_sha256": receipts.sha256_file(source_manifest),
        "native_function_calls": 0,
        "search_calls_made": 0,
        "simulator_battles_started": 0,
        "model_weights_loaded": False,
        "baseline_package_main_imported": False,
    }
    adapter_receipt = tmp_path / "marnie-h10-direct-policy-adapter-r251-v8.json"
    _write_json(adapter_receipt, {"offline_preflight": offline_preflight})
    return adapter_receipt, source_snapshot, offline_preflight


def _audit_fixture(tmp_path: Path) -> tuple[dict[str, object], receipts.FileIdentity, tuple[str, ...]]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity only")
    identity = receipts.file_identity(checkpoint, label="test checkpoint")
    _head_map, expected_heads, route_ids = receipts._head_contract()
    audit: dict[str, object] = {
        "schema": receipts.R241_CHECKPOINT_AUDIT_SCHEMA,
        "checkpoint": identity.as_dict(),
        "heads": {
            "architecture_present_head_count": 19,
            "non_combo_head_count": 18,
            "non_combo_route_count": 18,
            "active_non_combo_head_names": list(expected_heads),
            "active_non_combo_route_names": list(expected_heads),
            "active_non_combo_fusion_route_ids": list(route_ids),
            "active_non_combo_fusion_route_ids_sha256": receipts.sha256_bytes(
                receipts.canonical_json(list(route_ids))
            ),
            "every_non_combo_head_trainable": True,
            "every_non_combo_fusion_route_enabled": True,
            "combo_state": {
                "head_present": True,
                "physical_route_present": True,
                "loss_weight": 0.0,
                "route_enabled": False,
            },
        },
        "sorted_tensor_inventory": {
            "structural_sha256": "sha256:" + "1" * 64,
            "content_sha256": "sha256:" + "2" * 64,
        },
        "head_role_map_route_ids_sha256": receipts.sha256_bytes(
            receipts.canonical_json(list(route_ids))
        ),
        "matchup_adapter": {
            "checkpoint_dormant_state": {
                "runtime_enabled": False,
                "training_enabled": False,
                "ordinary_optimizer_included": False,
            },
            "activation_provenance": {
                "matchup_adapter_bank_preserved": True,
                "matchup_adapter_training_enabled": True,
                "matchup_adapter_runtime_enabled": True,
                "matchup_adapter_checkpoint_runtime_enabled": False,
                "matchup_adapter_checkpoint_training_enabled": False,
                "matchup_adapter_checkpoint_main_optimizer_included": False,
                "matchup_adapter_isolated_bank_only_optimizer": True,
                "matchup_adapter_isolated_fit_continuation_required": True,
                "matchup_adapter_external_collection_runtime_enabled": True,
                "matchup_adapter_external_terminal_runtime_enabled": True,
            },
            "isolated_optimizer": {
                "parameter_count": 256,
                "state_count": 1,
                "parameter_name_inventory_sha256": "sha256:" + "3" * 64,
            },
        },
        "runtime_smoke": {
            "model_reconstructed": True,
            "adapter_runtime_enabled_for_smoke": True,
            "adapter_output_changed": True,
            "action_selector": "direct_policy_only",
            "mcts_calls": 0,
            "rtp_calls": 0,
            "search_calls": 0,
        },
    }
    audit["audit_fingerprint_sha256"] = receipts.sha256_bytes(
        receipts.canonical_json(audit)
    )
    return audit, identity, expected_heads


def test_h10_adapter_receipt_binds_the_exact_active_source_snapshot(
    tmp_path: Path,
) -> None:
    adapter_receipt, source_snapshot, _offline = _h10_source_binding_fixture(tmp_path)

    identity = receipts.validate_r241_h10_adapter_source_binding(
        adapter_receipt,
        source_snapshot=source_snapshot,
    )

    assert identity == receipts.file_identity(
        adapter_receipt, label="expected H10 adapter receipt"
    ).as_dict()


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("source_snapshot_root", "/stale/r241-source"),
        ("source_snapshot_manifest", "/stale/r241-source/manifest.json"),
        ("source_snapshot_manifest_sha256", "sha256:" + "f" * 64),
    ),
)
def test_h10_adapter_receipt_rejects_stale_source_lineage(
    tmp_path: Path,
    field: str,
    stale_value: str,
) -> None:
    adapter_receipt, source_snapshot, offline = _h10_source_binding_fixture(tmp_path)
    offline[field] = stale_value
    _write_json(adapter_receipt, {"offline_preflight": offline})

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="stale source snapshot",
    ):
        receipts.validate_r241_h10_adapter_source_binding(
            adapter_receipt,
            source_snapshot=source_snapshot,
        )


@pytest.mark.parametrize(
    "inactive_filename",
    (
        "marnie-h10-direct-policy-adapter-r251.json",
        "marnie-h10-direct-policy-adapter-r251-v2.json",
        "marnie-h10-direct-policy-adapter-r251-v3.json",
        "marnie-h10-direct-policy-adapter-r251-v4.json",
        "marnie-h10-direct-policy-adapter-r251-v5.json",
        "marnie-h10-direct-policy-adapter-r251-v6.json",
        "marnie-h10-direct-policy-adapter-r251-v7.json",
    ),
)
def test_h10_adapter_receipt_rejects_the_inactive_predecessor_path(
    tmp_path: Path, inactive_filename: str,
) -> None:
    _adapter_receipt, source_snapshot, offline = _h10_source_binding_fixture(tmp_path)
    legacy = tmp_path / inactive_filename
    _write_json(legacy, {"offline_preflight": offline})

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="predeclared successor path",
    ):
        receipts.validate_r241_h10_adapter_source_binding(
            legacy,
            source_snapshot=source_snapshot,
        )


@pytest.mark.parametrize("missing_field", ("root", "manifest"))
def test_h10_adapter_binding_rejects_missing_active_source_paths(
    tmp_path: Path,
    missing_field: str,
) -> None:
    adapter_receipt, source_snapshot, _offline = _h10_source_binding_fixture(tmp_path)
    source_snapshot.pop(missing_field)

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="omits the active source root or manifest",
    ):
        receipts.validate_r241_h10_adapter_source_binding(
            adapter_receipt,
            source_snapshot=source_snapshot,
        )


def test_checkpoint_audit_rejects_self_asserted_v1_and_combo_reenable(
    tmp_path: Path,
) -> None:
    audit, identity, expected_heads = _audit_fixture(tmp_path)
    receipts._validate_checkpoint_audit(
        audit,
        expected_checkpoint=identity,
        expected_heads=expected_heads,
        label="fixture",
    )

    v1 = dict(audit)
    v1["schema"] = "poke_bot.alakazam_new_list_direct_policy_r241_checkpoint_audit/v1"
    v1.pop("audit_fingerprint_sha256")
    v1["audit_fingerprint_sha256"] = receipts.sha256_bytes(receipts.canonical_json(v1))
    with pytest.raises(receipts.R241CheckpointReceiptError, match="self-asserted v1"):
        receipts._validate_checkpoint_audit(
            v1,
            expected_checkpoint=identity,
            expected_heads=expected_heads,
            label="fixture",
        )

    combo_on = json.loads(json.dumps(audit))
    combo_on["heads"]["combo_state"]["route_enabled"] = True
    combo_on.pop("audit_fingerprint_sha256")
    combo_on["audit_fingerprint_sha256"] = receipts.sha256_bytes(
        receipts.canonical_json(combo_on)
    )
    with pytest.raises(receipts.R241CheckpointReceiptError, match="19/18"):
        receipts._validate_checkpoint_audit(
            combo_on,
            expected_checkpoint=identity,
            expected_heads=expected_heads,
            label="fixture",
        )


def test_peak_validator_rejects_a_v1_receipt_before_host_or_checkpoint_use(
    tmp_path: Path,
) -> None:
    receipt = {
        "schema": "poke_bot.alakazam_new_list_direct_policy_r241_peak_r195_preservation/v1",
        "revision": 241,
    }
    receipt["receipt_fingerprint_sha256"] = receipts.sha256_bytes(
        receipts.canonical_json(receipt)
    )
    path = tmp_path / "peak-r195-preservation-v6.json"
    _write_json(path, receipt)
    with pytest.raises(receipts.R241CheckpointReceiptError, match="generated v2 evidence"):
        receipts.validate_peak_r195_preservation_receipt(
            receipt_path=path,
            parent_checkpoint=tmp_path / "does-not-matter.pt",
            learner_matchup_tree=tmp_path / "tree.json",
            h10_matchup_tree=tmp_path / "h10.json",
            official_cg_root=tmp_path / "r236",
            environment={},
        )


@pytest.mark.parametrize(
    "inactive_filename",
    (
        "peak-r195-preservation.json",
        "peak-r195-preservation-v2.json",
        "peak-r195-preservation-v3.json",
        "peak-r195-preservation-v4.json",
        "peak-r195-preservation-v5.json",
    ),
)
def test_peak_receipt_paths_reject_the_inactive_predecessor_lineage(
    tmp_path: Path, inactive_filename: str,
) -> None:
    inactive = tmp_path / inactive_filename
    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="predeclared successor path",
    ):
        receipts.generate_peak_r195_preservation_receipt(
            output_path=inactive,
            contract_path="ignored",
            parent_checkpoint="ignored",
            learner_matchup_tree="ignored",
            h10_matchup_tree="ignored",
            official_cg_root="ignored",
            environment={},
            expert_window_receipt="ignored",
            protected_expert_pointer="ignored",
            h10_adapter_receipt="ignored",
            active_gate_contract="ignored",
            frozen_specialist_registry="ignored",
            research_control_registry="ignored",
            adapter_training_activation="ignored",
            source_snapshot_root="ignored",
            source_snapshot_manifest="ignored",
            source_outputs_root="ignored",
            source_snapshot_host="inzi",
        )
    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="predeclared successor path",
    ):
        receipts.validate_peak_r195_preservation_receipt(
            receipt_path=inactive,
            parent_checkpoint="ignored",
            learner_matchup_tree="ignored",
            h10_matchup_tree="ignored",
            official_cg_root="ignored",
            environment={},
        )


def test_terminal_validator_rejects_v1_before_checkpoint_or_refresh_use(
    tmp_path: Path,
) -> None:
    model = {
        "schema": "poke_bot.alakazam_new_list_direct_policy_r241_model_runtime_activation/v1"
    }
    matchup = {
        "schema": "poke_bot.alakazam_new_list_direct_policy_r241_matchup_runtime_activation/v1"
    }
    for name, payload in (("model.json", model), ("matchup.json", matchup)):
        payload["receipt_fingerprint_sha256"] = receipts.sha256_bytes(
            receipts.canonical_json(payload)
        )
        _write_json(tmp_path / name, payload)
    with pytest.raises(receipts.R241CheckpointReceiptError, match="generated v2 evidence"):
        receipts.validate_terminal_runtime_receipts(
            model_receipt_path=tmp_path / "model.json",
            matchup_receipt_path=tmp_path / "matchup.json",
            r195_parent_checkpoint=tmp_path / "does-not-matter.pt",
            terminal_parent_checkpoint=tmp_path / "iter_00009.pt",
            terminal_checkpoint=tmp_path / "expert_before_iter_00010.pt",
            terminal_refresh_receipt=tmp_path / "terminal_expert_refresh.json",
            terminal_rehearsal_receipt=tmp_path / "before_iter_00010.json",
            learner_matchup_tree=tmp_path / "tree.json",
            h10_matchup_tree=tmp_path / "h10.json",
            official_cg_root=tmp_path / "r236",
            environment={},
        )


def test_r241_cycle_exposes_only_no_slot_change_policy() -> None:
    assert receipts.IMMUTABLE_ADAPTER_SLOT_PREFIX == 20
    assert receipts.BASELINE_ADAPTER_ROSTER_SHA256 == (
        "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
    )


def _real_r195_reconstruction_config_shape() -> tuple[dict[str, object], dict[str, object]]:
    """Return a real current dataclass shape and its historical r195 view."""

    live = dataclasses.asdict(config.ModelConfig())
    # The regression must be independent of any test runner environment that
    # happens to set a successor knob.  The actual historical reconstruction
    # explicitly supplies these inert values before strict model loading.
    live.update(receipts.R195_LIVE_ONLY_SUCCESSOR_MODEL_CONFIG_DEFAULTS)
    serialized = dict(live)
    for field in receipts.R195_LIVE_ONLY_SUCCESSOR_MODEL_CONFIG_DEFAULTS:
        serialized.pop(field)
    return serialized, live


def _adapter_activation_portability_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    """Create byte-identical Inzi and Elmo provenance-copy locations."""

    inzi = (
        tmp_path
        / "home/inzi/poke-bot-agent/outputs/final_format_alakazam_r79"
        / "runtime/authorization/matchup_adapter_bootstrap_authorization.json"
    )
    elmo = (
        tmp_path
        / "mnt/Main/main/poke-bot-agent/outputs/pure_rl"
        / "alakazam_new_list_direct_policy_r241/runtime/provenance/r195"
        / "matchup_adapter_bootstrap_authorization.json"
    )
    for path in (inzi, elmo):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable EA38 adapter authorization fixture\n")
    return inzi, elmo, {
        "activation_receipt": str(inzi),
        "activation_receipt_digest": receipts.sha256_file(inzi),
    }


def _patch_adapter_activation_identity(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    monkeypatch.setattr(
        receipts,
        "PARENT_R195_ADAPTER_ACTIVATION_SHA256",
        receipts.sha256_file(path),
    )
    monkeypatch.setattr(
        receipts,
        "PARENT_R195_ADAPTER_ACTIVATION_SIZE_BYTES",
        path.stat().st_size,
    )


@pytest.mark.parametrize("host", ("inzi", "elmo"))
def test_adapter_activation_accepts_the_exact_inzi_or_elmo_local_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    inzi, elmo, fit = _adapter_activation_portability_fixture(tmp_path)
    _patch_adapter_activation_identity(monkeypatch, inzi)
    selected = inzi if host == "inzi" else elmo
    identity, declared = receipts._isolated_adapter_activation_identity(
        fit,
        adapter_training_activation=None if host == "inzi" else selected,
    )

    assert identity == receipts.file_identity(
        selected,
        label="expected host-local adapter activation",
    )
    assert declared == {
        "path": str(inzi),
        "sha256": receipts.sha256_file(inzi),
        "expected_size_bytes": inzi.stat().st_size,
    }


def test_adapter_activation_rejects_a_wrong_sha_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inzi, elmo, fit = _adapter_activation_portability_fixture(tmp_path)
    _patch_adapter_activation_identity(monkeypatch, inzi)
    elmo.write_bytes(b"tampered! EA38 adapter authorization fixture\n")

    with pytest.raises(receipts.R241CheckpointReceiptError, match="checksum mismatch"):
        receipts._isolated_adapter_activation_identity(
            fit,
            adapter_training_activation=elmo,
        )


def test_adapter_activation_rejects_a_wrong_size_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inzi, elmo, fit = _adapter_activation_portability_fixture(tmp_path)
    elmo.write_bytes(elmo.read_bytes() + b"x")
    fit["activation_receipt_digest"] = receipts.sha256_file(elmo)
    _patch_adapter_activation_identity(monkeypatch, elmo)
    monkeypatch.setattr(
        receipts,
        "PARENT_R195_ADAPTER_ACTIVATION_SIZE_BYTES",
        inzi.stat().st_size,
    )

    with pytest.raises(receipts.R241CheckpointReceiptError, match="size mismatch"):
        receipts._isolated_adapter_activation_identity(
            fit,
            adapter_training_activation=elmo,
        )


def test_adapter_activation_rejects_a_missing_host_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inzi, _elmo, fit = _adapter_activation_portability_fixture(tmp_path)
    _patch_adapter_activation_identity(monkeypatch, inzi)

    with pytest.raises(receipts.R241CheckpointReceiptError, match="regular, non-symlink"):
        receipts._isolated_adapter_activation_identity(
            fit,
            adapter_training_activation=tmp_path / "missing-elmo-copy.json",
        )


def test_adapter_activation_rejects_a_non_ea38_embedded_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inzi, elmo, fit = _adapter_activation_portability_fixture(tmp_path)
    _patch_adapter_activation_identity(monkeypatch, inzi)
    elmo.write_bytes(b"different EA38 adapter authorization fixture\n")
    fit["activation_receipt_digest"] = receipts.sha256_file(elmo)

    with pytest.raises(receipts.R241CheckpointReceiptError, match="exact EA38 proof"):
        receipts._isolated_adapter_activation_identity(
            fit,
            adapter_training_activation=elmo,
        )


def test_peak_audit_recomputation_uses_the_receipt_local_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = tmp_path / "r195-parent.pt"
    parent_path.write_bytes(b"r195")
    parent = receipts.file_identity(parent_path, label="r195 parent")
    activation_path = tmp_path / "elmo-provenance/ea38.json"
    activation_path.parent.mkdir()
    activation_path.write_bytes(b"ea38")
    activation = receipts.file_identity(activation_path, label="local activation")
    captured: dict[str, object] = {}

    def audit(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"audit_fingerprint_sha256": "sha256:" + "a" * 64}

    monkeypatch.setattr(receipts, "audit_checkpoint", audit)

    result = receipts._recompute_peak_checkpoint_audit(
        parent=parent,
        policy=receipts.DEFAULT_POLICY,
        environment={"CG_LIB_PATH": "/sealed/cg"},
        training_activation=activation,
    )

    assert result["audit_fingerprint_sha256"] == "sha256:" + "a" * 64
    assert captured["adapter_training_activation"] == activation.path


def test_runtime_model_config_accepts_the_real_r195_backfill_shape() -> None:
    serialized, live = _real_r195_reconstruction_config_shape()

    assert set(live) - set(serialized) == set(
        receipts.R195_LIVE_ONLY_SUCCESSOR_MODEL_CONFIG_DEFAULTS
    )
    receipts._assert_runtime_model_config_matches_serialized(
        serialized=serialized,
        live=live,
    )


@pytest.mark.parametrize(
    ("field", "non_default"),
    (
        ("own_deck_ledger_enabled", True),
        ("own_deck_ledger_runtime_enabled", True),
        ("own_deck_ledger_width", 129),
        ("own_deck_ledger_option_feature_dim", 9),
        ("visible_tutor_completion_head_enabled", True),
        ("terminal_conversion_head_enabled", True),
        ("visible_tutor_completion_route_enabled", True),
        ("visible_tutor_completion_route_runtime_enabled", True),
        ("terminal_conversion_route_enabled", True),
        ("terminal_conversion_route_runtime_enabled", True),
    ),
)
def test_runtime_model_config_rejects_each_nondefault_r195_backfill(
    field: str,
    non_default: object,
) -> None:
    serialized, live = _real_r195_reconstruction_config_shape()
    live[field] = non_default

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="successor-only model_config default",
    ):
        receipts._assert_runtime_model_config_matches_serialized(
            serialized=serialized,
            live=live,
        )


def test_runtime_model_config_rejects_any_other_live_extra_field() -> None:
    serialized, live = _real_r195_reconstruction_config_shape()
    live["unattested_runtime_model_config_field"] = False

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="runtime reconstruction changed model_config",
    ):
        receipts._assert_runtime_model_config_matches_serialized(
            serialized=serialized,
            live=live,
        )


def test_runtime_model_config_does_not_erase_a_checkpoint_field() -> None:
    serialized, live = _real_r195_reconstruction_config_shape()
    serialized["own_deck_ledger_enabled"] = True

    with pytest.raises(
        receipts.R241CheckpointReceiptError,
        match="runtime reconstruction changed model_config",
    ):
        receipts._assert_runtime_model_config_matches_serialized(
            serialized=serialized,
            live=live,
        )


def test_peak_generator_serializes_the_real_protected_pointer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pointer validator returns ``FileIdentity``, never a JSON row."""

    def identity(name: str) -> receipts.FileIdentity:
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        return receipts.file_identity(path, label=name)

    contract = identity("contract.json")
    checkpoint = identity("checkpoint.pt")
    roster = identity("roster.json")
    learner_tree = identity("learner-tree.json")
    h10_tree = identity("h10-tree.json")
    expert_window = identity("expert-window.json")
    protected_pointer = identity("PROTECTED_EXPERT_CORPUS.json")
    adapter = identity("marnie-h10-direct-policy-adapter-r251-v8.json")
    activation = identity("adapter-activation.json")
    public = identity("public-contract.json")
    output = identity("peak-r195-preservation-v6.json")

    audit = {
        "checkpoint": checkpoint.as_dict(),
        "heads": {},
        "matchup_adapter": {},
        "runtime_smoke": {},
    }
    source_snapshot = {
        "owner_contract_sha256": contract.sha256,
        "host": "inzi",
        "root": "/read-only/source",
        "manifest": "/read-only/source/r241-source-snapshot-manifest.json",
    }
    baseline_roster = {"slots": []}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        receipts,
        "_validate_contract",
        lambda *_args, **_kwargs: (
            contract,
            {
                "latest_owner_clarification_revision": receipts.R241_OWNER_CLARIFICATION_REVISION,
                "expert_soft_refresh": {
                    "exact_window_evidence_binding": {
                        "canonical_manifest_sha256": expert_window.sha256
                    }
                },
            },
        ),
    )
    monkeypatch.setattr(
        receipts,
        "_assert_direct_environment",
        lambda *_args, **_kwargs: {"action_selector": "direct_policy_only"},
    )
    def audit_checkpoint(*_args: object, **kwargs: object) -> dict[str, object]:
        captured["adapter_training_activation"] = kwargs.get(
            "adapter_training_activation"
        )
        return audit

    monkeypatch.setattr(receipts, "audit_checkpoint", audit_checkpoint)
    monkeypatch.setattr(
        receipts,
        "_baseline_adapter_roster",
        lambda *_args, **_kwargs: (roster, baseline_roster),
    )
    monkeypatch.setattr(
        receipts,
        "_load_checkpoint_payload",
        lambda *_args, **_kwargs: (checkpoint, {}),
    )
    monkeypatch.setattr(
        receipts,
        "_slot_registry_from_payload",
        lambda *_args, **_kwargs: baseline_roster,
    )
    monkeypatch.setattr(
        receipts,
        "authenticated_source_snapshot_provenance",
        lambda **_kwargs: source_snapshot,
    )
    monkeypatch.setattr(
        receipts,
        "_validate_tree",
        lambda _path, *, label, **_kwargs: {
            "r195 learner matchup tree": learner_tree,
            "H10 Marnie matchup tree": h10_tree,
        }[label],
    )
    monkeypatch.setattr(
        receipts,
        "_validate_expert_window",
        lambda *_args, **_kwargs: expert_window,
    )
    monkeypatch.setattr(
        receipts,
        "validate_r241_protected_expert_pointer",
        lambda *_args, **_kwargs: protected_pointer,
    )
    monkeypatch.setattr(
        receipts,
        "validate_r241_h10_adapter_source_binding",
        lambda *_args, **_kwargs: adapter.as_dict(),
    )
    monkeypatch.setattr(
        receipts,
        "_path_binding",
        lambda path, **_kwargs: (
            activation.as_dict() if str(path) == "activation" else public.as_dict()
        ),
    )
    monkeypatch.setattr(
        receipts,
        "audit_append_only_adapter_slot_migration",
        lambda **_kwargs: {"status": "no_slot_change"},
    )

    def capture_immutable(
        _path: Path | str,
        payload: dict[str, object],
        *,
        label: str,
    ) -> receipts.FileIdentity:
        assert label == "r241 peak-r195 preservation receipt"
        captured["bytes"] = receipts.canonical_json(payload)
        return output

    monkeypatch.setattr(receipts, "_immutable_json", capture_immutable)

    result = receipts.generate_peak_r195_preservation_receipt(
        output_path=output.path,
        contract_path="contract",
        parent_checkpoint="checkpoint",
        learner_matchup_tree="learner",
        h10_matchup_tree="h10",
        official_cg_root="cg-r236",
        environment={},
        expert_window_receipt="expert-window",
        protected_expert_pointer="protected-pointer",
        h10_adapter_receipt="adapter",
        active_gate_contract="public-a",
        frozen_specialist_registry="public-b",
        research_control_registry="public-c",
        adapter_training_activation="activation",
        source_snapshot_root="source",
        source_snapshot_manifest="manifest",
        source_outputs_root="outputs",
        source_snapshot_host="inzi",
        baseline_adapter_roster="roster",
    )

    emitted = json.loads(captured["bytes"])
    assert emitted["expert_window"]["protected_pointer"] == protected_pointer.as_dict()
    assert result["expert_window"]["protected_pointer"] == protected_pointer.as_dict()
    assert captured["adapter_training_activation"] == "activation"


def test_terminal_refresh_auditor_rejects_non_five_epoch_or_sidecar_drift(
    tmp_path: Path,
) -> None:
    """Boundary-10 provenance cannot be fabricated from terminal filenames."""

    parent_path = tmp_path / "iter_00009.pt"
    terminal_path = tmp_path / "expert_before_iter_00010.pt"
    contract_path = tmp_path / "owner-contract.json"
    roster_path = tmp_path / "matchup_adapter_roster.json"
    for path, body in (
        (parent_path, b"iter-nine"),
        (terminal_path, b"terminal"),
        (contract_path, b"owner"),
        (roster_path, b"roster"),
    ):
        path.write_bytes(body)
    parent = receipts.file_identity(parent_path, label="terminal parent")
    terminal = receipts.file_identity(terminal_path, label="terminal output")
    contract = receipts.file_identity(contract_path, label="owner contract")
    roster = receipts.file_identity(roster_path, label="adapter roster")
    source_snapshot = {
        "schema": receipts.R241_SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "root": "/read-only/source",
        "source_execution_root": "/read-only/source",
        "manifest": "/read-only/source/r241-source-snapshot-manifest.json",
        "manifest_sha256": "sha256:" + "a" * 64,
        "source_tree_sha256": "sha256:" + "b" * 64,
        "owner_contract_sha256": contract.sha256,
        "file_inventory_sha256": "sha256:" + "c" * 64,
        "outputs_root": "/external/outputs",
        "host": "inzi",
    }
    slot_migration = {
        "schema": receipts.R241_ADAPTER_SLOT_MIGRATION_SCHEMA,
        "status": "no_slot_change",
        "parent_slot_registry_sha256": "sha256:" + "d" * 64,
        "candidate_slot_registry_sha256": "sha256:" + "d" * 64,
        "retained_slot_count": 20,
        "retained_slot_tensor_inventory_sha256": "sha256:" + "e" * 64,
        "existing_slots_byte_immutable": True,
        "new_slots": [],
        "new_slot_proofs": [],
    }
    fusion = {
        "schema": receipts.R241_PEAK_R195_LIVE_FUSION_SCHEMA,
        "candidate_id": receipts.R241_CANDIDATE_ID,
        "owner_decision_revision": 241,
            "owner_clarification_revision": receipts.R241_OWNER_CLARIFICATION_REVISION,
        "fixed_cycle_updates": 10,
        "phase": "expert_refresh",
        "boundary_iteration": 10,
        "physical_head_count": 19,
        "active_non_combo_head_count": 18,
        "combo_state": {
            "head_present": True,
            "loss_weight": 0.0,
            "fusion_route_enabled": False,
        },
        "checkpoint_audit_fingerprint_sha256": "sha256:" + "f" * 64,
        "source_snapshot": source_snapshot,
        "adapter_slot_migration": slot_migration,
        "owner_contract": contract.as_dict(),
        "baseline_adapter_roster": roster.as_dict(),
    }
    rehearsal = {
        "schema": 5,
        "before_iteration": 10,
        "epochs": 5,
        "parent_digest": parent.sha256,
        "checkpoint": str(terminal.path),
        "checkpoint_digest": terminal.sha256,
        "peak_r195_live_fusion": json.loads(json.dumps(fusion)),
    }
    rehearsal_path = tmp_path / "before_iter_00010.json"
    _write_json(rehearsal_path, rehearsal)
    refresh = {
        "schema": receipts.TERMINAL_EXPERT_REFRESH_SCHEMA,
        "before_iteration": 10,
        "rl_updates_completed": 10,
        "epochs_completed": 5,
        "parent": parent.as_dict(),
        "refreshed": terminal.as_dict(),
        "expert_rehearsal": json.loads(json.dumps(rehearsal)),
        "next_collection_started": False,
    }
    refresh_path = tmp_path / "terminal_expert_refresh.json"
    _write_json(refresh_path, refresh)
    kwargs = {
        "terminal_refresh_receipt": refresh_path,
        "terminal_rehearsal_receipt": rehearsal_path,
        "terminal_parent": parent,
        "terminal_checkpoint": terminal,
        "terminal_payload": {"extra": {"peak_r195_live_fusion": fusion}},
        "root_parent_audit": {"audit_fingerprint_sha256": fusion["checkpoint_audit_fingerprint_sha256"]},
        "contract_identity": contract,
        "source_snapshot": source_snapshot,
        "baseline_adapter_roster": roster,
        "slot_migration": slot_migration,
    }
    proof = receipts._validate_terminal_refresh_boundary(**kwargs)
    assert proof["epochs_completed"] == 5

    refresh["epochs_completed"] = 4
    _write_json(refresh_path, refresh)
    with pytest.raises(receipts.R241CheckpointReceiptError, match="five-epoch"):
        receipts._validate_terminal_refresh_boundary(**kwargs)

    refresh["epochs_completed"] = 5
    rehearsal["peak_r195_live_fusion"]["boundary_iteration"] = 9
    _write_json(rehearsal_path, rehearsal)
    _write_json(refresh_path, {**refresh, "expert_rehearsal": rehearsal})
    with pytest.raises(receipts.R241CheckpointReceiptError, match="sidecars drifted"):
        receipts._validate_terminal_refresh_boundary(**kwargs)
