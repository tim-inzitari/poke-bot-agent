"""R260 pre-start OwnDeckLedger successor for the protected r241 lineage.

This deliberately does *not* relax the r241 peak-r195 preservation auditor or
reuse the post-refresh r258 activation gate.  It is an additive, offline
artifact builder for the one owner-authorized pre-start boundary.  The r195
parent remains the object audited by the legacy contract; this module audits
the new child as a separate immutable lineage.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from . import checkpoint
from . import own_deck_migration as migration
from .own_deck_successor import receipt_digest, seal_receipt


R260_OWNER_REVISION = 260
# Revision 262 changes placement and transport semantics while preserving the
# r260 architecture/migration import itself.  Keep those two assertions
# distinct: treating the latest clarification as the head-import revision
# would either reject the canonical owner file or accidentally rewrite the
# imported architecture provenance.
R260_OWNER_CONTRACT_REVISION = 262
R260_OWNER_CONTRACT_PATH = Path("state/alakazam-new-list-direct-policy-r241.json")
R260_OWNER_CONTRACT_SHA256 = (
    "sha256:57cbc0ac7ca7ee3791f7257899a16f6f0642749effa218323368e35940cdc202"
)
R260_MIGRATION_SCHEMA = "poke_bot.r241_own_deck_successor_migration/v1"
R260_SIDECAR_BINDING_SCHEMA = "poke_bot.r241_own_deck_sidecar_binding/v1"
R260_SOURCE_CLOSURE_SCHEMA = "poke_bot.r241_own_deck_successor_source_closure/v1"
R260_INZI_DATASET_BINDING_SCHEMA = "poke_bot.r241_own_deck_inzi_dataset_binding/v1"
R260_CANARY_ACTIVATION_SCHEMA = "poke_bot.r241_own_deck_canary_activation/v1"
R260_MIGRATION_KIND = "pre_start_zero_safe_import"


class R241OwnDeckSuccessorError(RuntimeError):
    """The r260 successor evidence is incomplete or an immutable invariant moved."""


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class R260OwnerContract:
    path: Path
    sha256: str
    parent: FileIdentity
    source_manifest_sha256: str
    source_window_receipt_sha256: str
    side_store_root: str
    inzi_training_root: str
    inzi_prefix_staging_root: str


@dataclass(frozen=True)
class R260MigrationResult:
    checkpoint: FileIdentity
    receipt_path: Path
    receipt_sha256: str
    added_tensor_keys: tuple[str, ...]


def validate_r260_source_closure(
    value: Mapping[str, Any] | Path | str,
    *,
    owner_contract: R260OwnerContract | None = None,
) -> dict[str, Any]:
    """Require a fresh, sealed r260 source closure rather than a r259 runtime tree."""

    contract = owner_contract or load_r260_owner_contract()
    if isinstance(value, (Path, str)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R241OwnDeckSuccessorError("r260 source closure is unreadable") from exc
    closure = dict(_mapping(value, label="r260 source closure"))
    if closure.get("schema") != R260_SOURCE_CLOSURE_SCHEMA or closure.get("status") != "sealed":
        raise R241OwnDeckSuccessorError("r260 source closure is not sealed")
    if closure.get("owner_contract_sha256") != contract.sha256:
        raise R241OwnDeckSuccessorError("r260 source closure owner contract mismatch")
    _require_sha(closure.get("source_tree_sha256"), label="r260 source closure tree")
    _require_sha(closure.get("closure_receipt_sha256"), label="r260 source closure receipt")
    if closure.get("derived_from_r259_runtime_tree") is not False or closure.get("unlisted_pycache_present") is not False:
        raise R241OwnDeckSuccessorError("r260 source closure is not a fresh sealed source tree")
    return closure


def validate_r260_migration_receipt(
    value: Mapping[str, Any] | Path | str,
    *,
    owner_contract: R260OwnerContract | None = None,
    require_local_child: bool = False,
) -> dict[str, Any]:
    """Validate the r260 child receipt independently of legacy r195 auditing."""

    contract = owner_contract or load_r260_owner_contract()
    if isinstance(value, (Path, str)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R241OwnDeckSuccessorError("r260 migration receipt is unreadable") from exc
    receipt = dict(_mapping(value, label="r260 migration receipt"))
    if receipt.get("schema") != R260_MIGRATION_SCHEMA or receipt.get("kind") != R260_MIGRATION_KIND or receipt.get("status") != "passed":
        raise R241OwnDeckSuccessorError("r260 migration receipt is invalid")
    if receipt.get("receipt_sha256") != receipt_digest(receipt):
        raise R241OwnDeckSuccessorError("r260 migration receipt digest mismatch")
    owner = _mapping(receipt.get("owner_contract"), label="r260 receipt owner")
    if owner.get("sha256") != contract.sha256:
        raise R241OwnDeckSuccessorError("r260 migration receipt owner mismatch")
    parent = _mapping(receipt.get("parent_checkpoint"), label="r260 receipt parent")
    if parent.get("sha256") != contract.parent.sha256 or parent.get("size_bytes") != contract.parent.size_bytes:
        raise R241OwnDeckSuccessorError("r260 migration receipt parent identity mismatch")
    child = _mapping(receipt.get("child_checkpoint"), label="r260 receipt child")
    _require_sha(child.get("sha256"), label="r260 receipt child")
    size = child.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise R241OwnDeckSuccessorError("r260 migration receipt child size is invalid")
    verification = _mapping(receipt.get("verification"), label="r260 receipt verification")
    if tuple(verification.get("zero_safe_final_projection_keys") or ()) != migration.ZERO_SAFE_FINAL_PROJECTION_KEYS or verification.get("parent_behavior_exact_for_absent_and_valid_public_ledger") is not True:
        raise R241OwnDeckSuccessorError("r260 migration receipt zero-safe proof changed")
    runtime = _mapping(receipt.get("runtime_authority"), label="r260 receipt runtime")
    if any(runtime.get(key) is not False for key in ("own_deck_ledger_runtime_enabled", "visible_tutor_completion_route_runtime_enabled", "terminal_conversion_route_runtime_enabled", "selector_change_authorized", "serving_eligible")):
        raise R241OwnDeckSuccessorError("r260 migration receipt grants runtime authority")
    if require_local_child:
        observed = _file_identity(str(child.get("path") or ""), label="r260 child checkpoint")
        if observed.sha256 != child["sha256"] or observed.size_bytes != size:
            raise R241OwnDeckSuccessorError("r260 child checkpoint FileIdentity mismatch")
        parent_path = Path(str(parent.get("path") or ""))
        observed_parent = _file_identity(parent_path, label="r260 r195 parent checkpoint")
        _assert_parent_identity(observed_parent, contract)
        _verify_child(
            parent_path=observed_parent.path,
            child_path=observed.path,
            contract=contract,
            ledger_width=128,
        )
    return receipt


def validate_r260_canary_activation(
    value: Mapping[str, Any] | Path | str,
    *,
    migration_receipt: Mapping[str, Any],
    owner_contract: R260OwnerContract | None = None,
    require_local_checkpoint: bool = False,
) -> dict[str, Any]:
    """Validate the only transition from the dormant migration to update zero.

    The migration child is deliberately runtime-off.  A separate immutable
    canary checkpoint is the actual initial learner only after finite-gradient,
    source-disjoint evaluation, local/remote parity, and bounded-influence
    evidence are all rehashed and its serialized runtime flags are true.
    """

    contract = owner_contract or load_r260_owner_contract()
    if isinstance(value, (Path, str)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R241OwnDeckSuccessorError("r260 canary activation receipt is unreadable") from exc
    receipt = dict(_mapping(value, label="r260 canary activation receipt"))
    required = {
        "schema", "status", "receipt_sha256", "owner_contract_sha256",
        "migration_receipt_sha256", "canary_checkpoint", "evidence_receipts",
        "runtime_gates",
    }
    if set(receipt) != required or receipt.get("schema") != R260_CANARY_ACTIVATION_SCHEMA or receipt.get("status") != "passed":
        raise R241OwnDeckSuccessorError("r260 canary activation receipt shape/status is invalid")
    if receipt.get("receipt_sha256") != receipt_digest(receipt):
        raise R241OwnDeckSuccessorError("r260 canary activation receipt digest mismatch")
    if receipt.get("owner_contract_sha256") != contract.sha256:
        raise R241OwnDeckSuccessorError("r260 canary activation owner contract mismatch")
    if receipt.get("migration_receipt_sha256") != migration_receipt.get("receipt_sha256"):
        raise R241OwnDeckSuccessorError("r260 canary activation migration binding mismatch")
    evidence = _mapping(receipt.get("evidence_receipts"), label="r260 canary evidence")
    names = ("finite_gradient", "source_disjoint_evaluation", "local_remote_parity", "bounded_influence")
    if set(evidence) != set(names):
        raise R241OwnDeckSuccessorError("r260 canary evidence inventory changed")
    for name in names:
        _identity_row(evidence[name], label=f"r260 canary evidence {name}", verify_file=require_local_checkpoint)
    gates = _mapping(receipt.get("runtime_gates"), label="r260 canary runtime gates")
    if any(gates.get(key) is not True for key in ("own_deck_ledger_runtime_enabled", "visible_tutor_completion_route_runtime_enabled", "terminal_conversion_route_runtime_enabled")):
        raise R241OwnDeckSuccessorError("r260 canary does not enable every required runtime route")
    child = _identity_row(receipt.get("canary_checkpoint"), label="r260 canary checkpoint", verify_file=require_local_checkpoint)
    migration_child = _identity_row(
        _mapping(migration_receipt.get("child_checkpoint"), label="r260 migration child"),
        label="r260 migration child",
        verify_file=False,
    )
    if (
        child["sha256"] == migration_child["sha256"]
        and child["size_bytes"] == migration_child["size_bytes"]
    ):
        raise R241OwnDeckSuccessorError(
            "r260 canary checkpoint reuses the zero-safe migration child"
        )
    if require_local_checkpoint:
        payload = migration._load_checkpoint_payload(Path(child["path"]), label="r260 canary checkpoint")
        cfg = _mapping(payload.get("model_config"), label="r260 canary model config")
        if any(cfg.get(key) is not True for key in ("own_deck_ledger_enabled", "visible_tutor_completion_head_enabled", "terminal_conversion_head_enabled", "own_deck_ledger_runtime_enabled", "visible_tutor_completion_route_runtime_enabled", "terminal_conversion_route_runtime_enabled")):
            raise R241OwnDeckSuccessorError("r260 canary checkpoint config is not runtime-enabled")
    return receipt


def resolve_r260_runtime_initial_learner(
    value: Mapping[str, Any] | Path | str,
    *,
    migration_receipt: Mapping[str, Any],
    owner_contract: R260OwnerContract | None = None,
) -> dict[str, Any]:
    """Return the locally rehashed runtime-enabled learner for overlay v2."""
    receipt = validate_r260_canary_activation(
        value,
        migration_receipt=migration_receipt,
        owner_contract=owner_contract,
        require_local_checkpoint=True,
    )
    row = _identity_row(
        receipt["canary_checkpoint"],
        label="r260 runtime initial learner",
        verify_file=True,
    )
    payload = migration._load_checkpoint_payload(
        Path(row["path"]), label="r260 runtime initial learner"
    )
    cfg = dict(_mapping(payload.get("model_config"), label="r260 runtime model config"))
    return {"checkpoint": row, "model_config": cfg, "activation_receipt_sha256": receipt["receipt_sha256"]}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 71 or not text.startswith("sha256:") or any(c not in "0123456789abcdef" for c in text[7:]):
        raise R241OwnDeckSuccessorError(f"{label} must be a sha256 identity")
    return text


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R241OwnDeckSuccessorError(f"{label} must be a mapping")
    return value


def _file_identity(path: Path | str, *, label: str) -> FileIdentity:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R241OwnDeckSuccessorError(f"{label} must be a regular non-symlink file")
    resolved = candidate.resolve()
    return FileIdentity(
        path=resolved,
        sha256=checkpoint.checkpoint_digest(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _identity_row(value: object, *, label: str, verify_file: bool) -> dict[str, Any]:
    row = dict(_mapping(value, label=label))
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise R241OwnDeckSuccessorError(f"{label} must be an exact FileIdentity")
    digest = _require_sha(row.get("sha256"), label=label)
    size = row.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise R241OwnDeckSuccessorError(f"{label} size is invalid")
    if not str(row.get("path") or ""):
        raise R241OwnDeckSuccessorError(f"{label} path is missing")
    if verify_file:
        observed = _file_identity(str(row["path"]), label=label)
        if observed.sha256 != digest or observed.size_bytes != size:
            raise R241OwnDeckSuccessorError(f"{label} FileIdentity mismatch")
    return {"path": str(row["path"]), "sha256": digest, "size_bytes": size}


def load_r260_owner_contract(
    path: Path | str = R260_OWNER_CONTRACT_PATH,
    *,
    expected_sha256: str | None = R260_OWNER_CONTRACT_SHA256,
) -> R260OwnerContract:
    """Load the sole canonical r260 authority, optionally test-injected by SHA."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R241OwnDeckSuccessorError("r260 owner contract must be a regular file")
    raw = candidate.read_bytes()
    actual_sha = _sha256_bytes(raw)
    if expected_sha256 is not None and actual_sha != _require_sha(expected_sha256, label="owner contract"):
        raise R241OwnDeckSuccessorError("r260 owner contract digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241OwnDeckSuccessorError("r260 owner contract is invalid JSON") from exc
    top = _mapping(value, label="r260 owner contract")
    if (
        int(top.get("latest_owner_clarification_revision", -1))
        != R260_OWNER_CONTRACT_REVISION
    ):
        raise R241OwnDeckSuccessorError("r260 owner revision is not canonical")
    imported = _mapping(top.get("own_deck_head_structure_import"), label="head import")
    if int(imported.get("owner_revision", -1)) != R260_OWNER_REVISION:
        raise R241OwnDeckSuccessorError("r260 head import revision changed")
    architecture = _mapping(imported.get("architecture"), label="r260 architecture")
    exact_architecture = {
        "shared_adapter_width": 128,
        "option_feature_dim": 8,
        "visible_tutor_completion_output_dim": 7,
        "terminal_conversion_output_dim": 6,
        "typed_option_route_width": 16,
        "typed_option_route_aggregate_delta_cap": 1.0,
        "visible_tutor_completion_loss_weight": 0.025,
        "terminal_conversion_loss_weight": 0.025,
    }
    for key, expected in exact_architecture.items():
        if architecture.get(key) != expected:
            raise R241OwnDeckSuccessorError(f"r260 architecture changed {key}")
    if tuple(architecture.get("new_tensor_prefixes") or ()) != migration.SUCCESSOR_TENSOR_PREFIXES:
        raise R241OwnDeckSuccessorError("r260 tensor-prefix inventory changed")
    migration_contract = _mapping(imported.get("migration"), label="r260 migration")
    if tuple(migration_contract.get("zero_safe_final_projection_keys") or ()) != migration.ZERO_SAFE_FINAL_PROJECTION_KEYS:
        raise R241OwnDeckSuccessorError("r260 zero-safe projection inventory changed")
    corpus = _mapping(imported.get("expert_corpus"), label="r260 expert corpus")
    if (
        int(corpus.get("day_count", -1)) != 20
        or int(corpus.get("validated_episode_count", -1)) != 91_253
        or int(corpus.get("source_archive_bytes", -1)) != 14_842_033_482
        or corpus.get("partial_or_unreceipted_side_store_training_eligible") is not False
    ):
        raise R241OwnDeckSuccessorError("r260 corpus completeness contract changed")
    placement = _mapping(imported.get("training_placement"), label="r260 training placement")
    if set(placement) != {
        "owner_revision",
        "sole_managed_training_host",
        "elmo_role",
        "elmo_may_train_learner",
        "canonical_inzi_training_root",
        "inzi_prefix_staging_root",
        "prefix_transfer_while_elmo_builder_runs",
        "prefix_transfer_scope",
        "per_day_transfer",
        "partial_staging_root_training_eligible",
        "final_promotion",
        "trainer_may_consume_elmo_mnt_main_path",
        "trainer_input",
        "healthy_r259_service_may_be_stopped_restarted_or_reconfigured",
    }:
        raise R241OwnDeckSuccessorError("r260 training placement key shape changed")
    if (
        int(placement.get("owner_revision", -1)) != R260_OWNER_CONTRACT_REVISION
        or placement.get("sole_managed_training_host") != "inzi"
        or placement.get("elmo_role")
        != "read_only_source_preprocessing_and_bounded_disposable_parity_only"
        or placement.get("elmo_may_train_learner") is not False
        or placement.get("prefix_transfer_while_elmo_builder_runs") is not True
        or placement.get("prefix_transfer_scope")
        != "committed_non_dot_daily_directories_only"
        or placement.get("per_day_transfer")
        != "create_only_byte_identical_rehash_and_read_only_seal"
        or placement.get("partial_staging_root_training_eligible") is not False
        or placement.get("final_promotion")
        != "atomic_only_after_20_of_20_join_parity_and_transport_receipts_pass"
        or placement.get("trainer_may_consume_elmo_mnt_main_path") is not False
        or placement.get("trainer_input")
        != "local_inzi_disk_backed_exact_four_key_streaming_index_only"
        or placement.get("healthy_r259_service_may_be_stopped_restarted_or_reconfigured")
        is not False
    ):
        raise R241OwnDeckSuccessorError("r260 Inzi-only training placement changed")
    inzi_root = str(placement.get("canonical_inzi_training_root") or "")
    staging_root = str(placement.get("inzi_prefix_staging_root") or "")
    if (
        not inzi_root.startswith("/")
        or not staging_root.startswith("/")
        or inzi_root == staging_root
        or "-staging-" not in staging_root
        or "/mnt/Main/" in inzi_root
    ):
        raise R241OwnDeckSuccessorError("r260 Inzi training roots are invalid")
    parent = _mapping(top.get("parent"), label="r260 parent")
    parent_path = Path(str(parent.get("checkpoint") or "")).expanduser()
    parent_sha = _require_sha(parent.get("checkpoint_sha256"), label="r260 parent")
    parent_size = parent.get("checkpoint_bytes")
    if isinstance(parent_size, bool) or not isinstance(parent_size, int) or parent_size <= 0:
        raise R241OwnDeckSuccessorError("r260 parent byte identity is invalid")
    return R260OwnerContract(
        path=candidate.resolve(),
        sha256=actual_sha,
        parent=FileIdentity(parent_path, parent_sha, parent_size),
        source_manifest_sha256=_require_sha(corpus.get("source_manifest_sha256"), label="r260 source manifest"),
        source_window_receipt_sha256=_require_sha(corpus.get("source_window_receipt_sha256"), label="r260 source window"),
        side_store_root=str(corpus.get("derived_side_store_root") or ""),
        inzi_training_root=inzi_root,
        inzi_prefix_staging_root=staging_root,
    )


def _assert_parent_identity(parent: FileIdentity, contract: R260OwnerContract) -> None:
    if parent.sha256 != contract.parent.sha256 or parent.size_bytes != contract.parent.size_bytes:
        raise R241OwnDeckSuccessorError("r260 protected r195 parent FileIdentity mismatch")


def _verify_child(
    *, parent_path: Path, child_path: Path, contract: R260OwnerContract, ledger_width: int
) -> tuple[str, ...]:
    """Verify actual tensors, optimizer append-only state, and two forward parities."""

    parent_payload = migration._load_checkpoint_payload(parent_path, label="r195 parent")
    child_payload = migration._load_checkpoint_payload(child_path, label="r260 child")
    migration._require_alakazam_parent(parent_payload)
    parent_state = migration._tensor_state(parent_payload.get("model_state_dict"), label="r195 state")
    child_state = migration._tensor_state(child_payload.get("model_state_dict"), label="r260 state")
    migration._reject_parent_successor_keys(parent_state)
    parent_cfg = migration._parent_model_config(parent_payload.get("model_config"))
    target_cfg = migration.successor_model_config(parent_payload.get("model_config") or {}, ledger_width=ledger_width)
    if child_payload.get("model_config") != asdict(target_cfg):
        raise R241OwnDeckSuccessorError("r260 child model config drifted")
    parent_model = migration._build_model(migration._default_model_factory, parent_cfg, parent_state, label="r195 parent")
    child_model = migration._build_model(migration._default_model_factory, target_cfg, parent_state, label="r260 child")
    migration._strict_load(parent_model, parent_state, label="r195 parent")
    expected_child = migration._tensor_state(child_model.state_dict(), label="r260 expected state")
    added = migration._validate_state_key_delta(parent_state, expected_child)
    if set(child_state) != set(expected_child):
        raise R241OwnDeckSuccessorError("r260 child tensor inventory drifted")
    migration._assert_inherited_tensor_identity(parent_state, child_state)
    migration._strict_load(child_model, child_state, label="r260 child")
    migration._assert_successor_physical_contract(child_model, target_cfg, child_state)
    migration._assert_fusion_inventory_unchanged(parent_model, child_model)
    expected_optimizer, _ = migration._expanded_optimizer_state(
        parent_payload.get("optimizer_state_dict"), parent_model=parent_model, child_model=child_model
    )
    migration._assert_nested_exact(expected_optimizer, child_payload.get("optimizer_state_dict"), label="r260 optimizer")
    metadata = _mapping(_mapping(child_payload.get("extra"), label="r260 extra").get("r241_own_deck_successor_migration"), label="r260 migration metadata")
    if (
        metadata.get("schema") != R260_MIGRATION_SCHEMA
        or metadata.get("parent_checkpoint_sha256") != contract.parent.sha256
        or metadata.get("owner_contract_sha256") != contract.sha256
        or metadata.get("all_inherited_tensors_bit_identical") is not True
        or metadata.get("runtime_routes_enabled") is not False
        or metadata.get("physical_training_routes_enabled") is not True
        or tuple(metadata.get("added_tensor_keys") or ()) != tuple(added)
    ):
        raise R241OwnDeckSuccessorError("r260 child migration metadata drifted")
    parent_outputs, absent_outputs, valid_outputs = migration._default_parity_probe(parent_model.eval(), child_model.eval())
    migration._assert_output_parity(parent_outputs, absent_outputs, label="r260 absent ledger")
    migration._assert_output_parity(parent_outputs, valid_outputs, label="r260 valid ledger")
    return tuple(added)


def materialize_r260_own_deck_successor(
    *,
    parent_checkpoint: Path | str,
    output_checkpoint: Path | str,
    receipt_path: Path | str,
    source_closure: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract | None = None,
) -> R260MigrationResult:
    """Create the only permitted pre-start child, without changing any authority."""

    contract = owner_contract or load_r260_owner_contract()
    closure = validate_r260_source_closure(source_closure, owner_contract=contract)
    parent = _file_identity(parent_checkpoint, label="r195 parent checkpoint")
    _assert_parent_identity(parent, contract)
    output = migration._new_output_path(output_checkpoint, label="r260 successor checkpoint")
    receipt = migration._new_output_path(receipt_path, label="r260 migration receipt")
    if output == parent.path or receipt == parent.path or output == receipt:
        raise R241OwnDeckSuccessorError("r260 outputs must be distinct from the protected parent and each other")
    parent_payload = migration._load_checkpoint_payload(parent.path, label="r195 parent")
    migration._require_alakazam_parent(parent_payload)
    parent_state = migration._tensor_state(parent_payload.get("model_state_dict"), label="r195 state")
    migration._reject_parent_successor_keys(parent_state)
    parent_cfg = migration._parent_model_config(parent_payload.get("model_config"))
    child_cfg = migration.successor_model_config(parent_payload.get("model_config") or {}, ledger_width=128)
    parent_model = migration._build_model(migration._default_model_factory, parent_cfg, parent_state, label="r195 parent")
    child_model = migration._build_model(migration._default_model_factory, child_cfg, parent_state, label="r260 child")
    migration._strict_load(parent_model, parent_state, label="r195 parent")
    initial_child = migration._tensor_state(child_model.state_dict(), label="r260 initial state")
    added = migration._validate_state_key_delta(parent_state, initial_child)
    child_state = {key: value.detach().cpu().clone() for key, value in initial_child.items()}
    child_state.update({key: value.detach().cpu().clone() for key, value in parent_state.items()})
    migration._strict_load(child_model, child_state, label="r260 child")
    migration._assert_successor_physical_contract(child_model, child_cfg, child_state)
    migration._assert_inherited_tensor_identity(parent_state, child_state)
    migration._assert_fusion_inventory_unchanged(parent_model, child_model)
    child_optimizer, optimizer = migration._expanded_optimizer_state(parent_payload.get("optimizer_state_dict"), parent_model=parent_model, child_model=child_model)
    payload = copy.deepcopy(parent_payload)
    payload["model_state_dict"] = child_state
    payload["model_config"] = asdict(child_cfg)
    payload["optimizer_state_dict"] = child_optimizer
    payload["model_id"] = f"{parent_payload.get('model_id') or 'alakazam'}.r241_own_deck_r260"
    extra = dict(payload.get("extra") or {})
    if "r241_own_deck_successor_migration" in extra:
        raise R241OwnDeckSuccessorError("parent already contains an r260 migration")
    extra["r241_own_deck_successor_migration"] = {
        "schema": R260_MIGRATION_SCHEMA,
        "kind": R260_MIGRATION_KIND,
        "owner_contract_sha256": contract.sha256,
        "parent_checkpoint_sha256": parent.sha256,
        "parent_checkpoint_size_bytes": parent.size_bytes,
        "all_inherited_tensors_bit_identical": True,
        "expected_new_tensor_prefixes": list(migration.SUCCESSOR_TENSOR_PREFIXES),
        "added_tensor_keys": list(added),
        "zero_safe_final_projection_keys": list(migration.ZERO_SAFE_FINAL_PROJECTION_KEYS),
        "physical_training_routes_enabled": True,
        "runtime_routes_enabled": False,
        "optimizer": optimizer,
        "serving_eligible": False,
        "selector_change_authorized": False,
        "package_or_submission_authorized": False,
    }
    payload["extra"] = extra
    provenance = dict(payload.get("provenance") or {})
    provenance["r241_own_deck_successor"] = {
        "schema": R260_MIGRATION_SCHEMA,
        "owner_contract_sha256": contract.sha256,
        "ledger_inventory": migration._inventory(child_model, "own_deck_ledger_inventory"),
        "option_head_inventory": migration._inventory(child_model, "own_deck_option_head_inventory"),
        "runtime_enabled": False,
    }
    payload["provenance"] = provenance
    temporary = migration._save_private_checkpoint(payload, output)
    try:
        migration._publish_immutable_file(temporary, output, label="r260 successor checkpoint")
    finally:
        temporary.unlink(missing_ok=True)
    child = _file_identity(output, label="r260 successor checkpoint")
    added = _verify_child(parent_path=parent.path, child_path=child.path, contract=contract, ledger_width=128)
    if _file_identity(parent.path, label="r195 parent checkpoint") != parent:
        raise R241OwnDeckSuccessorError("r195 parent changed during r260 migration")
    receipt_payload = seal_receipt({
        "schema": R260_MIGRATION_SCHEMA,
        "kind": R260_MIGRATION_KIND,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_contract": {"path": str(contract.path), "sha256": contract.sha256},
        "parent_checkpoint": parent.as_dict(),
        "child_checkpoint": child.as_dict(),
        "source_manifest_sha256": contract.source_manifest_sha256,
        "source_window_receipt_sha256": contract.source_window_receipt_sha256,
        "source_closure": {
            "source_tree_sha256": closure["source_tree_sha256"],
            "closure_receipt_sha256": closure["closure_receipt_sha256"],
        },
        "verification": {"added_tensor_keys": list(added), "zero_safe_final_projection_keys": list(migration.ZERO_SAFE_FINAL_PROJECTION_KEYS), "parent_behavior_exact_for_absent_and_valid_public_ledger": True},
        "runtime_authority": {"own_deck_ledger_runtime_enabled": False, "visible_tutor_completion_route_runtime_enabled": False, "terminal_conversion_route_runtime_enabled": False, "selector_change_authorized": False, "serving_eligible": False},
    })
    migration._write_exclusive_json(receipt, receipt_payload)
    return R260MigrationResult(child, receipt, str(receipt_payload["receipt_sha256"]), added)


def validate_r260_sidecar_binding(
    value: Mapping[str, Any] | Path | str,
    *,
    owner_contract: R260OwnerContract | None = None,
    verify_daily_receipt_files: bool = True,
) -> dict[str, Any]:
    """Accept only an explicit 20/20 r259 completion binding, never a prefix."""

    contract = owner_contract or load_r260_owner_contract()
    if isinstance(value, (Path, str)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R241OwnDeckSuccessorError("r260 side-store binding is unreadable") from exc
    binding = dict(_mapping(value, label="r260 side-store binding"))
    required_keys = {
        "schema", "status", "binding_sha256", "owner_contract_sha256",
        "source_manifest_sha256", "source_window_receipt_sha256", "day_count",
        "validated_episode_count", "source_archive_bytes",
        "daily_sidecar_meta_receipts", "terminal_receipts",
        "daily_build_identity", "partial_or_unreceipted_side_store_training_eligible",
    }
    if set(binding) != required_keys:
        raise R241OwnDeckSuccessorError("r260 side-store binding key shape changed")
    if binding.get("schema") != R260_SIDECAR_BINDING_SCHEMA or binding.get("status") != "complete_training_eligible":
        raise R241OwnDeckSuccessorError("r260 side-store binding is not complete")
    if binding.get("owner_contract_sha256") != contract.sha256:
        raise R241OwnDeckSuccessorError("r260 side-store binding owner contract mismatch")
    binding_sha = _require_sha(binding.get("binding_sha256"), label="r260 side-store binding")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    if _sha256_bytes((json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")) != binding_sha:
        raise R241OwnDeckSuccessorError("r260 side-store binding digest mismatch")
    for key, expected in (("source_manifest_sha256", contract.source_manifest_sha256), ("source_window_receipt_sha256", contract.source_window_receipt_sha256)):
        if binding.get(key) != expected:
            raise R241OwnDeckSuccessorError(f"r260 side-store binding changed {key}")
    if int(binding.get("day_count", -1)) != 20 or int(binding.get("validated_episode_count", -1)) != 91_253 or int(binding.get("source_archive_bytes", -1)) != 14_842_033_482:
        raise R241OwnDeckSuccessorError("r260 side-store binding count drifted")
    daily = _mapping(binding.get("daily_sidecar_meta_receipts"), label="r260 daily sidecars")
    expected_days = tuple(
        (date(2026, 7, 22) + timedelta(days=index)).isoformat()
        for index in range(20)
    )
    if tuple(sorted(str(key) for key in daily)) != expected_days:
        raise R241OwnDeckSuccessorError("r260 side-store does not bind the exact 20 calendar days")
    daily_rows = [
        _identity_row(daily[day], label=f"r260 daily sidecar {day}", verify_file=verify_daily_receipt_files)
        for day in expected_days
    ]
    if len({row["sha256"] for row in daily_rows}) != 20:
        raise R241OwnDeckSuccessorError("r260 side-store does not bind 20 unique daily receipt identities")
    terminal = _mapping(binding.get("terminal_receipts"), label="r260 terminal receipts")
    terminal_names = (
        "completion",
        "join",
        "schema",
        "count",
        "digest",
        "causal_local_remote_parity",
    )
    if set(terminal) != set(terminal_names):
        raise R241OwnDeckSuccessorError("r260 terminal receipt inventory changed")
    terminal_rows = {
        name: _identity_row(
            terminal[name],
            label=f"r260 terminal receipt {name}",
            verify_file=verify_daily_receipt_files,
        )
        for name in terminal_names
    }
    if len({row["sha256"] for row in terminal_rows.values()}) != len(terminal_rows):
        raise R241OwnDeckSuccessorError("r260 terminal receipts are not distinct")
    build = _mapping(binding.get("daily_build_identity"), label="r260 daily build identity")
    if set(build) != {
        "sidecar_build_code_sha256",
        "source_snapshot_tree_sha256",
        "container_image_id",
        "archive_native_classifier_sha256",
    }:
        raise R241OwnDeckSuccessorError("r260 daily build identity fields changed")
    for key, item in build.items():
        _require_sha(item, label=f"r260 daily build {key}")
    if verify_daily_receipt_files:
        observed_build: set[tuple[str, str, str, str]] = set()
        for day, row in zip(expected_days, daily_rows):
            try:
                raw = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
                source = _mapping(_mapping(raw, label=f"daily {day}").get("source"), label=f"daily {day} source")
                manifest = _mapping(source.get("manifest"), label=f"daily {day} manifest")
                raw_build = _mapping(_mapping(raw, label=f"daily {day}").get("build"), label=f"daily {day} build")
                code_sha = _sha256_bytes((json.dumps({"r259_build_code": dict(sorted(_mapping(raw_build.get("code"), label=f"daily {day} code").items()))}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                snapshot = _mapping(raw_build.get("source_snapshot"), label=f"daily {day} snapshot")
                image = _mapping(raw_build.get("image"), label=f"daily {day} image")
                classifier_sha = _sha256_bytes((json.dumps(_mapping(raw_build.get("classifier"), label=f"daily {day} classifier"), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise R241OwnDeckSuccessorError(f"r260 daily sidecar {day} is not readable JSON") from exc
            if raw.get("day") != day or manifest.get("sha256") != contract.source_manifest_sha256:
                raise R241OwnDeckSuccessorError(f"r260 daily sidecar {day} source identity drifted")
            observed_build.add((code_sha, str(snapshot.get("tree_sha256") or ""), str(image.get("id") or ""), classifier_sha))
        if observed_build != {tuple(build[key] for key in ("sidecar_build_code_sha256", "source_snapshot_tree_sha256", "container_image_id", "archive_native_classifier_sha256"))}:
            raise R241OwnDeckSuccessorError("r260 daily sidecars do not share the sealed build identity")
    if binding.get("partial_or_unreceipted_side_store_training_eligible") is not False:
        raise R241OwnDeckSuccessorError("r260 binding permits an unreceipted side store")
    return binding


def validate_r260_inzi_dataset_binding(
    value: Mapping[str, Any] | Path | str,
    *,
    sidecar_binding: Mapping[str, Any],
    owner_contract: R260OwnerContract | None = None,
    require_local_dataset: bool = False,
) -> dict[str, Any]:
    """Bind Elmo's complete join to an immutable, Inzi-consumable copy.

    A completed Elmo side-store is not itself permission for an Inzi trainer to
    dereference an Elmo NAS path.  The binding requires a create-only copy (or
    a deliberately bounded remote staging object) with byte-identical joined
    dataset identity and a separate local/remote causal-parity receipt.
    """

    contract = owner_contract or load_r260_owner_contract()
    if isinstance(value, (Path, str)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R241OwnDeckSuccessorError("r260 Inzi dataset binding is unreadable") from exc
    binding = dict(_mapping(value, label="r260 Inzi dataset binding"))
    required_keys = {
        "schema", "status", "binding_sha256", "owner_contract_sha256",
        "source_manifest_sha256", "source_window_receipt_sha256",
        "sidecar_binding_sha256", "sidecar_binding_file_sha256", "sidecar_binding", "transport_kind",
        "elmo_joined_dataset_sha256", "inzi_joined_dataset_sha256",
        "join_receipt_sha256", "schema_receipt_sha256", "count_receipt_sha256",
        "digest_receipt_sha256", "causal_local_remote_parity_receipt_sha256",
        "transport_receipt_sha256", "day_count", "validated_episode_count",
        "source_archive_bytes", "inzi_joined_dataset", "inzi_sidecar_root",
    }
    if set(binding) != required_keys:
        raise R241OwnDeckSuccessorError("r260 Inzi dataset binding key shape changed")
    if binding.get("schema") != R260_INZI_DATASET_BINDING_SCHEMA or binding.get("status") != "complete_transport_ready":
        raise R241OwnDeckSuccessorError("r260 Inzi dataset transport is incomplete")
    binding_sha = _require_sha(binding.get("binding_sha256"), label="r260 Inzi dataset binding")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    if _sha256_bytes((json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")) != binding_sha:
        raise R241OwnDeckSuccessorError("r260 Inzi dataset binding digest mismatch")
    if binding.get("owner_contract_sha256") != contract.sha256:
        raise R241OwnDeckSuccessorError("r260 Inzi dataset binding owner mismatch")
    for key in ("source_manifest_sha256", "source_window_receipt_sha256"):
        if binding.get(key) != getattr(contract, key):
            raise R241OwnDeckSuccessorError(f"r260 Inzi dataset binding changed {key}")
    if binding.get("sidecar_binding_sha256") != sidecar_binding.get("binding_sha256"):
        raise R241OwnDeckSuccessorError("r260 Inzi dataset does not bind the complete side-store")
    _require_sha(
        binding.get("sidecar_binding_file_sha256"),
        label="r260 Inzi sidecar aggregate file",
    )
    sidecar_file = _identity_row(
        binding.get("sidecar_binding"),
        label="r260 sidecar aggregate binding",
        verify_file=require_local_dataset,
    )
    if sidecar_file["sha256"] != binding["sidecar_binding_file_sha256"]:
        raise R241OwnDeckSuccessorError("r260 Inzi dataset sidecar aggregate FileIdentity drifted")
    if binding.get("transport_kind") not in {"create_only_copy", "bounded_remote_staging"}:
        raise R241OwnDeckSuccessorError("r260 Inzi dataset transport kind is not approved")
    if binding.get("elmo_joined_dataset_sha256") != binding.get("inzi_joined_dataset_sha256"):
        raise R241OwnDeckSuccessorError("r260 Inzi joined dataset differs from Elmo")
    for key in (
        "elmo_joined_dataset_sha256",
        "inzi_joined_dataset_sha256",
        "join_receipt_sha256",
        "schema_receipt_sha256",
        "count_receipt_sha256",
        "digest_receipt_sha256",
        "causal_local_remote_parity_receipt_sha256",
        "transport_receipt_sha256",
    ):
        _require_sha(binding.get(key), label=f"r260 Inzi {key}")
    for key in ("day_count", "validated_episode_count", "source_archive_bytes"):
        expected = {"day_count": 20, "validated_episode_count": 91_253, "source_archive_bytes": 14_842_033_482}[key]
        if int(binding.get(key, -1)) != expected:
            raise R241OwnDeckSuccessorError(f"r260 Inzi dataset changed {key}")
    dataset = _identity_row(binding.get("inzi_joined_dataset"), label="r260 Inzi joined dataset identity", verify_file=require_local_dataset)
    size = dataset["size_bytes"]
    if dataset.get("sha256") != binding.get("inzi_joined_dataset_sha256"):
        raise R241OwnDeckSuccessorError("r260 Inzi joined dataset identity drifted")
    path_text = str(dataset.get("path") or "")
    if not path_text or path_text.startswith(contract.side_store_root):
        raise R241OwnDeckSuccessorError("r260 trainer may not consume the Elmo side-store path directly")
    root_text = str(binding.get("inzi_sidecar_root") or "")
    if not root_text or root_text.startswith(contract.side_store_root):
        raise R241OwnDeckSuccessorError("r260 Inzi sidecar root is invalid or points to Elmo")
    configured_root = Path(contract.inzi_training_root).expanduser()
    configured_staging = Path(contract.inzi_prefix_staging_root).expanduser()
    candidate_root = Path(root_text).expanduser()
    if candidate_root != configured_root or candidate_root == configured_staging:
        raise R241OwnDeckSuccessorError(
            "r260 trainer may consume only the final Inzi dataset root, never prefix staging"
        )
    try:
        Path(path_text).expanduser().relative_to(configured_root)
    except ValueError as exc:
        raise R241OwnDeckSuccessorError(
            "r260 Inzi joined dataset is outside the final Inzi training root"
        ) from exc
    if require_local_dataset:
        root = candidate_root
        if root.is_symlink() or not root.is_dir():
            raise R241OwnDeckSuccessorError("r260 Inzi sidecar root is not a local directory")
    return binding
