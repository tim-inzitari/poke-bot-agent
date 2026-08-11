"""Receipt-gated, zero-safe checkpoint migration for the r258 successor.

This module is intentionally an *artifact* builder.  It has no selector,
service-manager, trainer-launch, or runtime-activation code.  A caller must
present the canonical post-refresh evidence chain before it will create an
immutable child checkpoint.

The child is deliberately a direct-policy no-op at materialization time:

* every inherited model tensor is copied byte-for-byte;
* the OwnDeckLedger adapter, visible-tutor head, terminal-conversion head, and
  their two policy routes are physically present;
* the successor training routes are physically enabled, while every successor
  runtime route remains disabled; and
* a real model forward is compared for absent and valid public-ledger inputs
  before publication.

The resulting receipt uses the canonical r258 post-refresh ``isolated_migration``
shape, so a later training-canary gate can consume it without trusting a
hand-written migration claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import checkpoint, config, features
from .features import SparseVector
from .own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger
from .own_deck_successor import (
    CANDIDATE_ID,
    OWNER_DECISION_REVISION,
    POST_REFRESH_RECEIPT_SCHEMA,
    OwnDeckSuccessorOperation,
    OwnDeckSuccessorStage,
    load_canonical_manifest,
    require_successor_operation,
    seal_receipt,
    validate_prior_stage_receipts,
    validate_refresh_completion_receipt,
)

MIGRATION_SCHEMA = "poke_bot.alakazam_own_deck_successor_checkpoint_migration/v1"
MIGRATION_PROVENANCE_SCHEMA = (
    "poke_bot.alakazam_own_deck_successor_checkpoint_provenance/v1"
)
MIGRATION_KIND = "isolated_migration"

# These prefixes are deliberately exhaustive.  A new key outside this list is
# not a harmless implementation detail: it would be an unreviewed architecture
# change mixed into a protected parent checkpoint.
SUCCESSOR_TENSOR_PREFIXES: tuple[str, ...] = (
    "own_deck_ledger_adapter.",
    "own_deck_ledger_option_adapter.",
    "visible_tutor_completion_head.",
    "terminal_conversion_head.",
    "visible_tutor_completion_route.",
    "terminal_conversion_route.",
)

SUCCESSOR_RUNTIME_CONFIG_FIELDS: tuple[str, ...] = (
    "own_deck_ledger_runtime_enabled",
    "visible_tutor_completion_route_runtime_enabled",
    "terminal_conversion_route_runtime_enabled",
)

SUCCESSOR_PHYSICAL_CONFIG_FIELDS: tuple[str, ...] = (
    "own_deck_ledger_enabled",
    "visible_tutor_completion_head_enabled",
    "terminal_conversion_head_enabled",
    "visible_tutor_completion_route_enabled",
    "terminal_conversion_route_enabled",
)

# The final output projections are the mathematical no-op fences.  The heads
# themselves may have ordinary fresh initializers because neither can alter the
# policy until its separately zeroed route is trained and receipt-activated.
ZERO_SAFE_FINAL_PROJECTION_KEYS: tuple[str, ...] = (
    "own_deck_ledger_adapter.output.weight",
    "own_deck_ledger_adapter.output.bias",
    "own_deck_ledger_option_adapter.network.3.weight",
    "own_deck_ledger_option_adapter.network.3.bias",
    "visible_tutor_completion_route.network.2.weight",
    "visible_tutor_completion_route.network.2.bias",
    "terminal_conversion_route.network.2.weight",
    "terminal_conversion_route.network.2.bias",
)


class OwnDeckMigrationError(RuntimeError):
    """A protected parent, successor child, or receipt chain is invalid."""


ModelFactory = Callable[[config.ModelConfig, Mapping[str, torch.Tensor]], nn.Module]
ParityProbe = Callable[
    [nn.Module, nn.Module],
    tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
]


@dataclass(frozen=True)
class OwnDeckMigrationVerification:
    """Verified facts about one zero-safe successor child."""

    parent_checkpoint_sha256: str
    child_checkpoint_sha256: str
    inherited_tensor_count: int
    added_tensor_keys: tuple[str, ...]
    zero_safe_final_projection_keys: tuple[str, ...]
    parent_output_keys: tuple[str, ...]
    valid_ledger_output_keys: tuple[str, ...]
    successor_parameter_count: int
    optimizer_existing_state_preserved: bool
    optimizer_new_parameters_fresh: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "child_checkpoint_sha256": self.child_checkpoint_sha256,
            "inherited_tensor_count": self.inherited_tensor_count,
            "added_tensor_keys": list(self.added_tensor_keys),
            "zero_safe_final_projection_keys": list(
                self.zero_safe_final_projection_keys
            ),
            "parent_output_keys": list(self.parent_output_keys),
            "valid_ledger_output_keys": list(self.valid_ledger_output_keys),
            "successor_parameter_count": self.successor_parameter_count,
            "optimizer_existing_state_preserved": self.optimizer_existing_state_preserved,
            "optimizer_new_parameters_fresh": self.optimizer_new_parameters_fresh,
        }


@dataclass(frozen=True)
class OwnDeckMigrationResult:
    """Immutable paths and receipt identity emitted by a successful migration."""

    checkpoint_path: Path
    receipt_path: Path
    checkpoint_sha256: str
    receipt_sha256: str
    verification: OwnDeckMigrationVerification

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MIGRATION_SCHEMA,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "receipt_path": str(self.receipt_path),
            "receipt_sha256": self.receipt_sha256,
            "verification": self.verification.as_dict(),
        }


def successor_model_config(
    parent_config: Mapping[str, Any],
    *,
    ledger_width: int = 128,
) -> config.ModelConfig:
    """Return the only allowed r258 additive model configuration.

    Parent fields are retained exactly where they affect the inherited
    architecture.  The six successor modules are the only physical additions;
    their three runtime switches remain false.  We reject a parent that already
    has any of these modules instead of trying to stack an ambiguous migration.
    """

    if isinstance(ledger_width, bool):
        raise OwnDeckMigrationError("own-deck ledger width must be a positive integer")
    try:
        normalized_width = int(ledger_width)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OwnDeckMigrationError(
            "own-deck ledger width must be a positive integer"
        ) from exc
    if normalized_width <= 0:
        raise OwnDeckMigrationError("own-deck ledger width must be a positive integer")
    if not isinstance(parent_config, Mapping):
        raise OwnDeckMigrationError("parent checkpoint lacks a model_config mapping")
    known = set(config.ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(parent_config) - known)
    if unknown:
        raise OwnDeckMigrationError(
            "parent model_config contains unsupported fields: " + ", ".join(unknown)
        )

    # A legacy parent must never inherit successor flags from its process
    # environment merely because a field happened not to be serialized.
    values = asdict(config.ModelConfig())
    values.update(dict(parent_config))
    legacy_defaults = {
        "own_deck_ledger_enabled": False,
        "own_deck_ledger_runtime_enabled": False,
        "own_deck_ledger_width": 128,
        "own_deck_ledger_option_feature_dim": OPTION_FEATURE_DIM,
        "visible_tutor_completion_head_enabled": False,
        "terminal_conversion_head_enabled": False,
        "visible_tutor_completion_route_enabled": False,
        "visible_tutor_completion_route_runtime_enabled": False,
        "terminal_conversion_route_enabled": False,
        "terminal_conversion_route_runtime_enabled": False,
    }
    for field, default in legacy_defaults.items():
        if field not in parent_config:
            values[field] = default
    already_present = [
        field
        for field in (*SUCCESSOR_PHYSICAL_CONFIG_FIELDS, *SUCCESSOR_RUNTIME_CONFIG_FIELDS)
        if bool(values.get(field, False))
    ]
    if already_present:
        raise OwnDeckMigrationError(
            "parent already carries own-deck successor architecture or runtime flags: "
            + ", ".join(already_present)
        )

    values.update(
        {
            "own_deck_ledger_enabled": True,
            "own_deck_ledger_runtime_enabled": False,
            "own_deck_ledger_width": normalized_width,
            "own_deck_ledger_option_feature_dim": OPTION_FEATURE_DIM,
            "visible_tutor_completion_head_enabled": True,
            "terminal_conversion_head_enabled": True,
            "visible_tutor_completion_route_enabled": True,
            "visible_tutor_completion_route_runtime_enabled": False,
            "terminal_conversion_route_enabled": True,
            "terminal_conversion_route_runtime_enabled": False,
        }
    )
    try:
        return config.ModelConfig(**values)
    except (TypeError, ValueError) as exc:
        raise OwnDeckMigrationError(
            f"parent model_config cannot form the r258 successor: {exc}"
        ) from exc


def materialize_own_deck_successor(
    *,
    parent_checkpoint: Path | str,
    expected_parent_sha256: str,
    output_checkpoint: Path | str,
    receipt_path: Path | str,
    refresh_completion_receipt: Mapping[str, Any] | Path | str,
    stage_receipts: Mapping[
        OwnDeckSuccessorStage | str, Mapping[str, Any] | Path | str
    ],
    ledger_width: int = 128,
    model_factory: ModelFactory | None = None,
    parity_probe: ParityProbe | None = None,
) -> OwnDeckMigrationResult:
    """Create one immutable dormant child after the canonical migration gate.

    ``expected_parent_sha256`` is required deliberately.  The parent is not a
    generic checkpoint input; it is a protected, receipt-bound terminal refresh
    artifact.  This function neither looks up a selector nor derives a parent
    from a mutable directory name.
    """

    # This is the canonical authorization decision.  It is intentionally made
    # before opening the checkpoint, so a missing/forged receipt cannot cause a
    # best-effort model conversion.
    gate = require_successor_operation(
        OwnDeckSuccessorOperation.ISOLATED_MIGRATION,
        refresh_completion_receipt=refresh_completion_receipt,
        stage_receipts=stage_receipts,
    )
    manifest = load_canonical_manifest()
    refresh = validate_refresh_completion_receipt(
        refresh_completion_receipt, manifest=manifest
    )
    validated_stages = validate_prior_stage_receipts(stage_receipts, manifest=manifest)
    if gate.manifest_sha256 != manifest.identity.sha256 or (
        gate.refresh_completion_sha256 != refresh.sha256
    ):
        raise OwnDeckMigrationError("canonical migration gate identity drifted")

    parent = _regular_file(parent_checkpoint, label="protected parent checkpoint")
    output = _new_output_path(output_checkpoint, label="successor checkpoint")
    receipt = _new_output_path(receipt_path, label="successor migration receipt")
    if output == parent or receipt == parent:
        raise OwnDeckMigrationError("successor outputs must not alias the protected parent")
    if output == receipt:
        raise OwnDeckMigrationError("checkpoint and receipt paths must be distinct")
    expected_parent_digest = _require_sha256(
        expected_parent_sha256, label="expected parent checkpoint"
    )
    parent_digest = checkpoint.checkpoint_digest(parent)
    if parent_digest != expected_parent_digest:
        raise OwnDeckMigrationError("protected parent checkpoint digest mismatch")
    if refresh.checkpoint_sha256 != parent_digest:
        raise OwnDeckMigrationError(
            "refresh-completion receipt does not bind the protected parent checkpoint"
        )

    parent_payload = _load_checkpoint_payload(parent, label="protected parent checkpoint")
    _require_alakazam_parent(parent_payload)
    parent_state = _tensor_state(
        parent_payload.get("model_state_dict"), label="parent model_state_dict"
    )
    _reject_parent_successor_keys(parent_state)
    target_cfg = successor_model_config(
        _mapping(parent_payload.get("model_config"), label="parent model_config"),
        ledger_width=ledger_width,
    )
    factory = model_factory or _default_model_factory
    parent_cfg = _parent_model_config(parent_payload.get("model_config"))
    parent_model = _build_model(factory, parent_cfg, parent_state, label="parent")
    _strict_load(parent_model, parent_state, label="parent")
    target_model = _build_model(factory, target_cfg, parent_state, label="successor")

    target_initial = _tensor_state(
        target_model.state_dict(), label="successor model_state_dict"
    )
    added_keys = _validate_state_key_delta(parent_state, target_initial)
    child_state = {
        key: value.detach().cpu().clone()
        for key, value in target_initial.items()
    }
    for key, value in parent_state.items():
        child_state[key] = value.detach().cpu().clone()
    _strict_load(target_model, child_state, label="successor")
    _assert_successor_physical_contract(target_model, target_cfg, child_state)
    _assert_inherited_tensor_identity(parent_state, child_state)
    _assert_fusion_inventory_unchanged(parent_model, target_model)

    parent_optimizer = parent_payload.get("optimizer_state_dict")
    child_optimizer, optimizer_summary = _expanded_optimizer_state(
        parent_optimizer,
        parent_model=parent_model,
        child_model=target_model,
    )

    child_payload = copy.deepcopy(parent_payload)
    child_payload["model_state_dict"] = child_state
    child_payload["model_config"] = asdict(target_cfg)
    child_payload["optimizer_state_dict"] = child_optimizer
    # A separate logical model ID makes accidental selector substitution more
    # obvious while retaining the parent as immutable provenance.
    parent_model_id = str(parent_payload.get("model_id") or "alakazam")
    child_payload["model_id"] = f"{parent_model_id}.own_deck_ledger_r258"
    extra = dict(child_payload.get("extra") or {})
    if "own_deck_successor_migration" in extra:
        raise OwnDeckMigrationError("parent already has an own-deck migration record")
    stage_digests = {stage.value: parsed.sha256 for stage, parsed in validated_stages.items()}
    extra["own_deck_successor_migration"] = {
        "schema": MIGRATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "kind": MIGRATION_KIND,
        "parent_checkpoint_sha256": parent_digest,
        "refresh_completion_receipt_sha256": refresh.sha256,
        "manifest_sha256": manifest.identity.sha256,
        "prior_stage_receipt_sha256s": stage_digests,
        "all_inherited_tensors_bit_identical": True,
        "expected_new_tensor_prefixes": list(SUCCESSOR_TENSOR_PREFIXES),
        "added_tensor_keys": list(added_keys),
        "zero_safe_final_projection_keys": list(ZERO_SAFE_FINAL_PROJECTION_KEYS),
        "physical_training_routes_enabled": True,
        "runtime_routes_enabled": False,
        "serving_eligible": False,
        "selector_change_authorized": False,
        "package_or_submission_authorized": False,
        "optimizer": optimizer_summary,
    }
    child_payload["extra"] = extra
    provenance = dict(child_payload.get("provenance") or {})
    if "own_deck_successor" in provenance:
        raise OwnDeckMigrationError("parent already has own-deck successor provenance")
    provenance["own_deck_successor"] = {
        "schema": MIGRATION_PROVENANCE_SCHEMA,
        "migration_schema": MIGRATION_SCHEMA,
        "parent_checkpoint_sha256": parent_digest,
        "successor_runtime_enabled": False,
        "training_routes_physical": True,
        "ledger_inventory": _inventory(target_model, "own_deck_ledger_inventory"),
        "option_head_inventory": _inventory(target_model, "own_deck_option_head_inventory"),
    }
    child_payload["provenance"] = provenance

    # Save only to a private temporary path, reconstruct it, and run the full
    # tensor/forward verifier before publishing an immutable hard link.
    temporary = _save_private_checkpoint(child_payload, output)
    try:
        preliminary_digest = checkpoint.checkpoint_digest(temporary)
        verification = verify_own_deck_successor_checkpoint(
            parent_checkpoint=parent,
            child_checkpoint=temporary,
            expected_parent_sha256=parent_digest,
            expected_child_sha256=preliminary_digest,
            ledger_width=ledger_width,
            model_factory=factory,
            parity_probe=parity_probe,
        )
        _publish_immutable_file(temporary, output, label="successor checkpoint")
    finally:
        temporary.unlink(missing_ok=True)

    child_digest = checkpoint.checkpoint_digest(output)
    if child_digest != preliminary_digest:
        raise OwnDeckMigrationError("published child digest changed during publication")
    if checkpoint.checkpoint_digest(parent) != parent_digest:
        raise OwnDeckMigrationError("protected parent changed during migration")

    receipt_payload = seal_receipt(
        {
            "schema": POST_REFRESH_RECEIPT_SCHEMA,
            "kind": MIGRATION_KIND,
            "status": "passed",
            "candidate_id": CANDIDATE_ID,
            "owner_decision_revision": OWNER_DECISION_REVISION,
            "manifest_sha256": manifest.identity.sha256,
            "refresh_completion_receipt_sha256": refresh.sha256,
            "prior_stage_receipt_sha256s": stage_digests,
            "depends_on_receipt_sha256s": {},
            "migration_schema": MIGRATION_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_checkpoint": {
                "path": str(parent),
                "sha256": parent_digest,
            },
            "child_checkpoint": {
                "path": str(output),
                "sha256": child_digest,
            },
            "runtime_authority": {
                "own_deck_ledger_runtime_enabled": False,
                "visible_tutor_completion_route_runtime_enabled": False,
                "terminal_conversion_route_runtime_enabled": False,
                "selector_change_authorized": False,
                "package_or_submission_authorized": False,
                "serving_eligible": False,
            },
            "verification": verification.as_dict(),
        }
    )
    _write_exclusive_json(receipt, receipt_payload)
    if checkpoint.checkpoint_digest(parent) != parent_digest:
        raise OwnDeckMigrationError("protected parent changed before receipt publication")
    return OwnDeckMigrationResult(
        checkpoint_path=output,
        receipt_path=receipt,
        checkpoint_sha256=child_digest,
        receipt_sha256=str(receipt_payload["receipt_sha256"]),
        verification=verification,
    )


def verify_own_deck_successor_checkpoint(
    *,
    parent_checkpoint: Path | str,
    child_checkpoint: Path | str,
    expected_parent_sha256: str,
    expected_child_sha256: str | None = None,
    ledger_width: int = 128,
    model_factory: ModelFactory | None = None,
    parity_probe: ParityProbe | None = None,
) -> OwnDeckMigrationVerification:
    """Fail closed unless a child is an exact dormant r258 extension.

    This verifies the physical state rather than trusting the child metadata:
    unexpected model-state keys, omitted successor keys, partial optimizer
    state, and nonzero zero-safe projections are all terminal errors.
    """

    parent = _regular_file(parent_checkpoint, label="protected parent checkpoint")
    child = _regular_file(child_checkpoint, label="successor child checkpoint")
    if parent == child:
        raise OwnDeckMigrationError("child checkpoint aliases its protected parent")
    expected_parent_digest = _require_sha256(
        expected_parent_sha256, label="expected parent checkpoint"
    )
    parent_digest = checkpoint.checkpoint_digest(parent)
    if parent_digest != expected_parent_digest:
        raise OwnDeckMigrationError("protected parent checkpoint digest mismatch")
    child_digest = checkpoint.checkpoint_digest(child)
    if expected_child_sha256 is not None:
        _require_sha256(expected_child_sha256, label="expected child checkpoint")
        if child_digest != expected_child_sha256:
            raise OwnDeckMigrationError("successor child checkpoint digest mismatch")

    parent_payload = _load_checkpoint_payload(parent, label="protected parent checkpoint")
    child_payload = _load_checkpoint_payload(child, label="successor child checkpoint")
    _require_alakazam_parent(parent_payload)
    parent_state = _tensor_state(
        parent_payload.get("model_state_dict"), label="parent model_state_dict"
    )
    child_state = _tensor_state(
        child_payload.get("model_state_dict"), label="child model_state_dict"
    )
    _reject_parent_successor_keys(parent_state)
    parent_cfg = _parent_model_config(parent_payload.get("model_config"))
    target_cfg = successor_model_config(
        _mapping(parent_payload.get("model_config"), label="parent model_config"),
        ledger_width=ledger_width,
    )
    if child_payload.get("model_config") != asdict(target_cfg):
        raise OwnDeckMigrationError("child model_config is not the exact r258 successor config")
    factory = model_factory or _default_model_factory
    parent_model = _build_model(factory, parent_cfg, parent_state, label="parent")
    child_model = _build_model(factory, target_cfg, parent_state, label="successor")
    _strict_load(parent_model, parent_state, label="parent")
    expected_child_state = _tensor_state(
        child_model.state_dict(), label="reconstructed successor model_state_dict"
    )
    added_keys = _validate_state_key_delta(parent_state, expected_child_state)
    if set(child_state) != set(expected_child_state):
        missing = sorted(set(expected_child_state) - set(child_state))
        unexpected = sorted(set(child_state) - set(expected_child_state))
        raise OwnDeckMigrationError(
            "child state inventory differs from successor architecture "
            f"(missing={missing}; unexpected={unexpected})"
        )
    _assert_inherited_tensor_identity(parent_state, child_state)
    # Check inherited identity before loading the child.  A corrupted feature
    # schema buffer can otherwise make PyTorch reject the checkpoint first and
    # hide the more useful invariant failure: a protected parent tensor drifted.
    _strict_load(child_model, child_state, label="successor")
    _assert_successor_physical_contract(child_model, target_cfg, child_state)
    _assert_fusion_inventory_unchanged(parent_model, child_model)

    expected_optimizer, optimizer_summary = _expanded_optimizer_state(
        parent_payload.get("optimizer_state_dict"),
        parent_model=parent_model,
        child_model=child_model,
    )
    # PyTorch optimizer state includes tensors, so Python mapping equality is
    # not usable.  Compare it with a serialization-independent exact recursive
    # tensor comparator instead.
    _assert_nested_exact(
        expected_optimizer,
        child_payload.get("optimizer_state_dict"),
        label="child optimizer state",
    )

    _verify_migration_metadata(
        child_payload,
        parent_digest=parent_digest,
        child_digest=child_digest,
        added_keys=added_keys,
    )
    parent_outputs, child_absent, child_valid = (
        (parity_probe or _default_parity_probe)(parent_model.eval(), child_model.eval())
    )
    _assert_output_parity(parent_outputs, child_absent, label="absent ledger")
    _assert_output_parity(parent_outputs, child_valid, label="valid ledger")
    if checkpoint.checkpoint_digest(parent) != parent_digest:
        raise OwnDeckMigrationError("protected parent changed during verification")
    return OwnDeckMigrationVerification(
        parent_checkpoint_sha256=parent_digest,
        child_checkpoint_sha256=child_digest,
        inherited_tensor_count=len(parent_state),
        added_tensor_keys=tuple(added_keys),
        zero_safe_final_projection_keys=ZERO_SAFE_FINAL_PROJECTION_KEYS,
        parent_output_keys=tuple(sorted(parent_outputs)),
        valid_ledger_output_keys=tuple(sorted(child_valid)),
        successor_parameter_count=sum(
            int(parameter.numel())
            for name, parameter in child_model.named_parameters()
            if _has_successor_prefix(name)
        ),
        optimizer_existing_state_preserved=bool(
            optimizer_summary["existing_state_preserved"]
        ),
        optimizer_new_parameters_fresh=bool(optimizer_summary["new_parameters_fresh"]),
    )


def _parent_model_config(value: object) -> config.ModelConfig:
    """Parse a parent config without allowing successor env defaults to leak in."""

    raw = _mapping(value, label="parent model_config")
    known = set(config.ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(raw) - known)
    if unknown:
        raise OwnDeckMigrationError(
            "parent model_config contains unsupported fields: " + ", ".join(unknown)
        )
    values = asdict(config.ModelConfig())
    values.update(dict(raw))
    for field, default in {
        "own_deck_ledger_enabled": False,
        "own_deck_ledger_runtime_enabled": False,
        "own_deck_ledger_width": 128,
        "own_deck_ledger_option_feature_dim": OPTION_FEATURE_DIM,
        "visible_tutor_completion_head_enabled": False,
        "terminal_conversion_head_enabled": False,
        "visible_tutor_completion_route_enabled": False,
        "visible_tutor_completion_route_runtime_enabled": False,
        "terminal_conversion_route_enabled": False,
        "terminal_conversion_route_runtime_enabled": False,
    }.items():
        if field not in raw:
            values[field] = default
    try:
        return config.ModelConfig(**values)
    except (TypeError, ValueError) as exc:
        raise OwnDeckMigrationError(f"parent model_config is invalid: {exc}") from exc


def _default_model_factory(
    cfg: config.ModelConfig,
    state: Mapping[str, torch.Tensor],
) -> nn.Module:
    """Reconstruct the ordinary model using architecture-defining state shapes."""

    from .model import build_model

    try:
        aux_classes = int(state["aux_head.3.weight"].shape[0])
        dense = bool(getattr(cfg, "dense_card2vec", False)) or (
            "card2vec.card_emb.weight" in state
        )
        if dense:
            encoder_vocab = int(
                state.get("card2vec.board_kind", torch.empty(0)).shape[0]
                or features.encoder_vocab_size()
            )
            decoder_vocab = int(
                state.get("card2vec.option_kind", torch.empty(0)).shape[0]
                or features.decoder_vocab_size()
            )
        else:
            encoder_vocab = int(state["board_bag.weight"].shape[0])
            decoder_vocab = int(state["option_bag.weight"].shape[0])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OwnDeckMigrationError(
            "checkpoint lacks architecture-defining tensors required for migration"
        ) from exc
    belief_widths = []
    for name in ("opp_hand_head.weight", "opp_remainder_head.weight"):
        value = state.get(name)
        if value is not None:
            if value.ndim != 2 or int(value.shape[0]) <= 0:
                raise OwnDeckMigrationError(f"checkpoint {name} has invalid shape")
            belief_widths.append(int(value.shape[0]))
    if len(set(belief_widths)) > 1:
        raise OwnDeckMigrationError("checkpoint belief head vocabularies disagree")
    belief_card_vocab = next(iter(belief_widths), int(features.card_vocab_size()))
    try:
        return build_model(
            cfg,
            device=torch.device("cpu"),
            aux_archetype_classes=aux_classes,
            encoder_vocab=encoder_vocab,
            decoder_vocab=decoder_vocab,
            belief_card_vocab=belief_card_vocab,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OwnDeckMigrationError(f"cannot reconstruct checkpoint model: {exc}") from exc


def _build_model(
    factory: ModelFactory,
    cfg: config.ModelConfig,
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> nn.Module:
    try:
        model = factory(cfg, state)
    except OwnDeckMigrationError:
        raise
    except Exception as exc:  # pragma: no cover - preserves original context for custom factories.
        raise OwnDeckMigrationError(f"{label} model factory failed: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise OwnDeckMigrationError(f"{label} model factory did not return torch.nn.Module")
    return model.cpu()


def _strict_load(model: nn.Module, state: Mapping[str, torch.Tensor], *, label: str) -> None:
    try:
        incompatible = model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OwnDeckMigrationError(f"{label} state does not strictly load: {exc}") from exc
    if list(incompatible.missing_keys) or list(incompatible.unexpected_keys):
        raise OwnDeckMigrationError(
            f"{label} strict state load returned missing/unexpected keys"
        )


def _validate_state_key_delta(
    parent_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> list[str]:
    parent_keys = set(parent_state)
    target_keys = set(target_state)
    missing = sorted(parent_keys - target_keys)
    if missing:
        raise OwnDeckMigrationError(
            "successor architecture omits inherited tensor keys: " + ", ".join(missing)
        )
    added = sorted(target_keys - parent_keys)
    expected = sorted(key for key in target_keys if _has_successor_prefix(key))
    if added != expected:
        unexpected = sorted(set(added) - set(expected))
        missing_expected = sorted(set(expected) - set(added))
        raise OwnDeckMigrationError(
            "successor state delta is not exactly the approved own-deck tensor set "
            f"(unexpected={unexpected}; missing={missing_expected})"
        )
    if not added:
        raise OwnDeckMigrationError("successor architecture materialized no own-deck tensors")
    for prefix in SUCCESSOR_TENSOR_PREFIXES:
        if not any(key.startswith(prefix) for key in added):
            raise OwnDeckMigrationError(f"successor is missing required tensor prefix {prefix}")
    return added


def _assert_successor_physical_contract(
    model: nn.Module,
    cfg: config.ModelConfig,
    state: Mapping[str, torch.Tensor],
) -> None:
    for field in SUCCESSOR_PHYSICAL_CONFIG_FIELDS:
        if getattr(cfg, field, None) is not True:
            raise OwnDeckMigrationError(f"successor physical config {field} is not enabled")
    for field in SUCCESSOR_RUNTIME_CONFIG_FIELDS:
        if getattr(cfg, field, None) is not False:
            raise OwnDeckMigrationError(f"successor runtime config {field} is not disabled")
    required_modules = (
        "own_deck_ledger_adapter",
        "own_deck_ledger_option_adapter",
        "visible_tutor_completion_head",
        "terminal_conversion_head",
        "visible_tutor_completion_route",
        "terminal_conversion_route",
    )
    for name in required_modules:
        if getattr(model, name, None) is None:
            raise OwnDeckMigrationError(f"successor lacks required physical module {name}")
    for key in ZERO_SAFE_FINAL_PROJECTION_KEYS:
        tensor = state.get(key)
        if tensor is None:
            raise OwnDeckMigrationError(f"successor lacks zero-safe projection {key}")
        if int(tensor.detach().count_nonzero().item()) != 0:
            raise OwnDeckMigrationError(f"successor zero-safe projection is nonzero: {key}")


def _assert_fusion_inventory_unchanged(parent: nn.Module, child: nn.Module) -> None:
    parent_inventory = _inventory(parent, "decision_fusion_inventory")
    child_inventory = _inventory(child, "decision_fusion_inventory")
    if parent_inventory != child_inventory:
        raise OwnDeckMigrationError(
            "successor changed the inherited Fusion inventory or denominator"
        )


def _inventory(model: nn.Module, name: str) -> Mapping[str, Any]:
    method = getattr(model, name, None)
    if not callable(method):
        raise OwnDeckMigrationError(f"model lacks required inventory method {name}")
    value = method()
    if not isinstance(value, Mapping):
        raise OwnDeckMigrationError(f"model inventory {name} is not a mapping")
    return copy.deepcopy(dict(value))


def _expanded_optimizer_state(
    value: object,
    *,
    parent_model: nn.Module,
    child_model: nn.Module,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append only successor params; preserve every existing Adam state byte."""

    if not isinstance(value, Mapping):
        raise OwnDeckMigrationError("protected parent lacks optimizer_state_dict")
    parent_optimizer = copy.deepcopy(dict(value))
    groups = parent_optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise OwnDeckMigrationError("migration requires one canonical optimizer parameter group")
    parent_names = [
        name for name, parameter in parent_model.named_parameters() if parameter.requires_grad
    ]
    child_names = [
        name for name, parameter in child_model.named_parameters() if parameter.requires_grad
    ]
    if child_names[: len(parent_names)] != parent_names:
        raise OwnDeckMigrationError("successor changed inherited trainable parameter order")
    added_names = child_names[len(parent_names) :]
    if not added_names or any(not _has_successor_prefix(name) for name in added_names):
        raise OwnDeckMigrationError(
            "successor parameters are not one approved appended own-deck suffix"
        )
    original_ids = list(groups[0].get("params") or [])
    if (
        len(original_ids) != len(parent_names)
        or len(set(original_ids)) != len(original_ids)
        or any(isinstance(identifier, bool) or not isinstance(identifier, int) for identifier in original_ids)
    ):
        raise OwnDeckMigrationError("parent optimizer parameter order is not canonical")
    state = parent_optimizer.get("state")
    if not isinstance(state, Mapping):
        raise OwnDeckMigrationError("parent optimizer state is not a mapping")
    if any(identifier not in set(original_ids) for identifier in state):
        raise OwnDeckMigrationError("parent optimizer has state for an unknown parameter")
    next_identifier = max(original_ids, default=-1) + 1
    added_ids = list(range(next_identifier, next_identifier + len(added_names)))
    group = dict(groups[0])
    group["params"] = [*original_ids, *added_ids]
    parent_optimizer["param_groups"] = [group]
    parent_optimizer["state"] = copy.deepcopy(dict(state))
    return parent_optimizer, {
        "existing_state_preserved": True,
        "new_parameters_fresh": True,
        "existing_trainable_parameter_count": len(parent_names),
        "added_trainable_parameter_count": len(added_names),
        "added_parameter_names": added_names,
    }


def _assert_inherited_tensor_identity(
    parent_state: Mapping[str, torch.Tensor], child_state: Mapping[str, torch.Tensor]
) -> None:
    for key, parent in parent_state.items():
        child = child_state.get(key)
        if child is None:
            raise OwnDeckMigrationError(f"successor omitted inherited tensor {key}")
        if _tensor_digest(parent) != _tensor_digest(child):
            raise OwnDeckMigrationError(f"inherited tensor changed during migration: {key}")


def _default_parity_probe(
    parent: nn.Module,
    child: nn.Module,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Run an actual minimal model forward with absent and valid ledger data."""

    if not all(hasattr(model, "forward") for model in (parent, child)):
        raise OwnDeckMigrationError("no forward contract is available for zero-safe parity")
    board = SparseVector()
    for _ in range(int(features.NUM_BOARD_TOKENS)):
        board.word_start()
    options = SparseVector()
    for _ in range(2):
        options.word_start()
    snapshot = OwnDeckLedger([1] * 60).observe(
        {
            "current": {
                "yourIndex": 0,
                "looking": [],
                "players": [
                    {
                        "hand": [{"id": 1, "serial": 1}],
                        "active": [],
                        "bench": [],
                        "discard": [],
                        "prize": [None] * 6,
                        "deckCount": 53,
                    },
                    {"hand": [], "active": [], "bench": [], "discard": [], "prize": []},
                ],
            },
            "select": {"deck": [], "option": []},
        }
    )
    if not snapshot.integrity_ok or snapshot.fail_closed:
        raise OwnDeckMigrationError("internal valid-ledger fixture became invalid")
    # Use a genuinely nonzero, schema-valid option row.  That proves both the
    # shared snapshot and the visible-option adapter remain serving-neutral
    # while their runtime switches are false; all-zero rows would not exercise
    # the latter guard.
    option_row = snapshot.features_for_card(1)
    if len(option_row) != OPTION_FEATURE_DIM or not any(option_row):
        raise OwnDeckMigrationError("internal valid-ledger option fixture became empty")
    option_rows = (option_row, option_row)
    try:
        with torch.inference_mode():
            parent_outputs = parent(board, options, n_options=[2])
            child_absent = child(board, options, n_options=[2])
            child_valid = child(
                board,
                options,
                n_options=[2],
                ledger_snapshots=snapshot,
                ledger_option_features=option_rows,
            )
    except Exception as exc:
        raise OwnDeckMigrationError(
            f"zero-safe policy/value/existing-head parity forward failed: {exc}"
        ) from exc
    for name, output in (
        ("parent", parent_outputs),
        ("child absent", child_absent),
        ("child valid ledger", child_valid),
    ):
        if not isinstance(output, Mapping):
            raise OwnDeckMigrationError(f"{name} forward did not return a mapping")
    return parent_outputs, child_absent, child_valid


def _assert_output_parity(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    label: str,
) -> None:
    missing = sorted(set(parent) - set(child))
    if missing:
        raise OwnDeckMigrationError(
            f"{label} forward omitted parent output(s): " + ", ".join(missing)
        )
    for key, expected in parent.items():
        actual = child[key]
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or _tensor_digest(expected) != _tensor_digest(actual):
                raise OwnDeckMigrationError(
                    f"{label} changed policy/value/existing-head output {key}"
                )
        elif actual != expected:
            raise OwnDeckMigrationError(f"{label} changed existing output {key}")


def _verify_migration_metadata(
    child_payload: Mapping[str, Any],
    *,
    parent_digest: str,
    child_digest: str,
    added_keys: Sequence[str],
) -> None:
    extra = _mapping(child_payload.get("extra"), label="child extra")
    migration = _mapping(extra.get("own_deck_successor_migration"), label="migration metadata")
    expected = {
        "schema": MIGRATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "kind": MIGRATION_KIND,
        "parent_checkpoint_sha256": parent_digest,
        "all_inherited_tensors_bit_identical": True,
        "physical_training_routes_enabled": True,
        "runtime_routes_enabled": False,
        "serving_eligible": False,
        "selector_change_authorized": False,
        "package_or_submission_authorized": False,
    }
    for field, expected_value in expected.items():
        if migration.get(field) != expected_value:
            raise OwnDeckMigrationError(f"migration metadata changed {field}")
    if migration.get("expected_new_tensor_prefixes") != list(SUCCESSOR_TENSOR_PREFIXES):
        raise OwnDeckMigrationError("migration metadata has an invalid tensor-prefix inventory")
    if migration.get("added_tensor_keys") != list(added_keys):
        raise OwnDeckMigrationError("migration metadata has an invalid added-tensor inventory")
    if migration.get("zero_safe_final_projection_keys") != list(
        ZERO_SAFE_FINAL_PROJECTION_KEYS
    ):
        raise OwnDeckMigrationError("migration metadata has an invalid zero-safe inventory")
    if not str(child_digest).startswith("sha256:"):
        raise OwnDeckMigrationError("child checkpoint digest is invalid")


def _tensor_state(value: object, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise OwnDeckMigrationError(f"{label} must be a nonempty tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key:
            raise OwnDeckMigrationError(f"{label} has an invalid tensor key")
        if not isinstance(tensor, torch.Tensor):
            raise OwnDeckMigrationError(f"{label} {key} is not a tensor")
        if tensor.layout != torch.strided:
            raise OwnDeckMigrationError(f"{label} {key} has unsupported tensor layout")
        result[key] = tensor.detach().cpu()
    return result


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu()
    if value.layout != torch.strided:
        raise OwnDeckMigrationError("unsupported non-strided tensor in identity comparison")
    # Include dtype/shape/stride and the contiguous raw storage bytes.  This is
    # stronger than numerical allclose and catches every floating-point bit.
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(repr(tuple(value.stride())).encode("ascii"))
    # ``view(dtype)`` cannot reinterpret a zero-dimensional scalar directly
    # on current PyTorch builds; flattening preserves its one scalar element
    # and also handles empty tensors without a special case.
    raw = value.contiguous().reshape(-1).view(torch.uint8)
    digest.update(raw.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _assert_nested_exact(expected: object, actual: object, *, label: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or _tensor_digest(expected) != _tensor_digest(actual):
            raise OwnDeckMigrationError(f"{label} tensor differs")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise OwnDeckMigrationError(f"{label} mapping keys differ")
        for key in expected:
            _assert_nested_exact(expected[key], actual[key], label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(expected) != len(actual):
            raise OwnDeckMigrationError(f"{label} sequence differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_nested_exact(left, right, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise OwnDeckMigrationError(f"{label} value differs")


def _reject_parent_successor_keys(state: Mapping[str, torch.Tensor]) -> None:
    present = sorted(key for key in state if _has_successor_prefix(key))
    if present:
        raise OwnDeckMigrationError(
            "protected parent already contains own-deck successor tensors: "
            + ", ".join(present)
        )


def _has_successor_prefix(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in SUCCESSOR_TENSOR_PREFIXES)


def _require_alakazam_parent(payload: Mapping[str, Any]) -> None:
    archetype = str(payload.get("archetype_id") or "").strip().casefold()
    if archetype != "alakazam":
        raise OwnDeckMigrationError("protected parent checkpoint is not an Alakazam lineage")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnDeckMigrationError(f"{label} must be a mapping")
    return value


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise OwnDeckMigrationError(f"{label} must be a regular non-symlink file")
    return candidate.resolve()


def _new_output_path(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"{label} is immutable and already exists: {candidate}")
    return candidate


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 71 or not text.startswith("sha256:") or any(
        char not in "0123456789abcdef" for char in text[7:]
    ):
        raise OwnDeckMigrationError(f"{label} digest is not sha256")
    return text


def _load_checkpoint_payload(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise OwnDeckMigrationError(f"{label} cannot be decoded") from exc
    if not isinstance(payload, dict):
        raise OwnDeckMigrationError(f"{label} must contain a checkpoint mapping")
    return payload


def _save_private_checkpoint(payload: Mapping[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.own-deck-migration.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # The hard link used for publication retains the source inode mode.
        # Lock the private inode before it has a public name so the immutable
        # child is never briefly published owner-writable.
        os.chmod(temporary, 0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_immutable_file(source: Path, destination: Path, *, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} is immutable and already exists: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise OwnDeckMigrationError(f"cannot atomically publish {label}: {exc}") from exc
    _fsync_directory(destination.parent)


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"successor migration receipt is immutable: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.own-deck-receipt.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        # Link publication makes the complete receipt visible in one namespace
        # operation.  No reader can observe a partial self-digest or checksum.
        os.link(temporary, path)
    finally:
        # This is only the private temporary inode created above.  The linked
        # immutable receipt, if publication succeeded, remains untouched.
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
