"""Fail-closed preparation contract for the post-r241 OwnDeckLedger train.

This is deliberately a *planning* boundary.  It validates immutable evidence,
derives the only allowed imbalance weights from materialized sidecar counts,
and writes a content-addressed next-train plan.  It never invokes a trainer,
loads a model, starts a service, changes a selector, builds a package, or
submits anything; it may inspect the staged ``TrainConfig`` ABI only.

The r258/r259 owner contract has two separate barriers which must both hold:

* :mod:`poke_bot.own_deck_successor` must permit the post-refresh training
  canary through a valid terminal-refresh, full-stage, and migration chain;
* the r259 replay sidecar must have one complete receipt-backed daily metadata
  record for every day in the protected twenty-day source window, plus exact
  join and local/remote parity evidence.

The emitted plan intentionally enables the *physical* ledger and typed
tutor/terminal heads/routes for a future isolated training process, while all
runtime/action gates remain false.  A prepared plan is not an execution
authorization; the actual canary still owns a separate immutable receipt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from . import own_deck_successor as successor
from .own_deck_ledger import (
    OPTION_FEATURE_DIM,
    OWN_DECK_LEDGER_SCHEMA,
    OWN_DECK_LEDGER_SCHEMA_VERSION,
)
from .own_deck_rollout_store import (
    ARCHIVE_RECEIPT_SCHEMA,
    DAILY_SHARD_NAME,
    EXPECTED_VERSIONED_RECEIPT_SHA256,
    OWN_DECK_ROLLOUT_DAILY_META_SCHEMA,
    OWN_DECK_ROLLOUT_DAILY_META_VERSION,
    OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
    OWN_DECK_ROLLOUT_SIDECAR_VERSION,
)
from .own_deck_rollout_store import (
    OWNER_DECISION_REVISION as R259_OWNER_DECISION_REVISION,
)
from .own_deck_rollout_store import (
    canonical_json_bytes as sidecar_canonical_json_bytes,
)
from .own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
    TERMINAL_CONVERSION_CLASSES,
    TERMINAL_CONVERSION_SCALAR_TARGET_NAMES,
    VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES,
)

NEXT_TRAIN_PLAN_SCHEMA: Final = "poke_bot.alakazam_own_deck_next_train_plan/v1"
NEXT_TRAIN_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_own_deck_next_train_receipt/v1"
)
# Keep this planning module importable in a receipt-only environment where
# PyTorch is intentionally absent.  These are the stable on-disk ABI values
# from own_deck_promotion_metrics.py / TrainConfig, rechecked against train.py
# during plan preparation below.
OWN_DECK_PROMOTION_METRICS_SCHEMA: Final = "poke_bot.own_deck_promotion_metrics/v1"
DEFAULT_CLOSEOUT_THRESHOLD: Final[float] = 0.5
DEFAULT_TERMINAL_ECE_BINS: Final[int] = 10
MISSED_EXPERT_CLOSEOUT_BASIS: Final = (
    "policy_top1_vs_observed_expert_selected_option"
)
# This is intentionally imported from the r259 builder rather than duplicated:
# a plan cannot quietly accept an early/provisional daily-meta spelling after
# the atomic side-store contract has landed.
SIDE_STORE_DAILY_META_SCHEMA: Final = OWN_DECK_ROLLOUT_DAILY_META_SCHEMA
SIDE_STORE_DAILY_META_VERSION: Final = OWN_DECK_ROLLOUT_DAILY_META_VERSION
# ``dataset.py`` carries the compact provenance alongside every joined
# ``GameSequence`` and seals a *separate* detached receipt once an exact-key
# materialization has succeeded.  Keep these names distinct: accepting the
# provenance object by itself would let a caller bypass the immutable receipt
# that binds it to the model/code/dataset preparation context.
SIDE_STORE_JOIN_PROVENANCE_SCHEMA: Final = "poke_bot.own_deck_rollout_sidecar_join/v1"
SIDE_STORE_JOIN_RECEIPT_SCHEMA: Final = (
    "poke_bot.own_deck_rollout_store_join_receipt/v1"
)
SIDE_STORE_PARITY_RECEIPT_SCHEMA: Final = (
    "poke_bot.own_deck_rollout_store_parity_receipt/v1"
)
METRIC_SUPPORT_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_own_deck_metric_support_receipt/v1"
)
MIGRATION_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_own_deck_successor_checkpoint_migration/v1"
)
SIDE_STORE_DATASET_IDENTITY_SCHEMA: Final = (
    "poke_bot.own_deck_rollout_store_dataset_identity/v1"
)
WEIGHT_DERIVATION_SCHEMA: Final = "poke_bot.own_deck_label_weight_derivation/v1"

# `TrainConfig` rejects values above 32.0.  This is therefore an ABI-aligned
# safety bound rather than a hand-picked imbalance recipe.  The lower bound is
# its reciprocal, avoiding zero class/BCE weights while preserving a bounded
# ratio.  Missing-support classes/facts stay neutral rather than receiving an
# invented prior.
TRAIN_MAX_REWEIGHT: Final[float] = 32.0
TRAIN_MIN_REWEIGHT: Final[float] = 1.0 / TRAIN_MAX_REWEIGHT
R258_TACTICAL_AUXILIARY_BUDGET: Final[float] = 0.05
R258_TUTOR_LOSS_WEIGHT: Final[float] = R258_TACTICAL_AUXILIARY_BUDGET / 2.0
R258_TERMINAL_LOSS_WEIGHT: Final[float] = R258_TACTICAL_AUXILIARY_BUDGET / 2.0

_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUIRED_TRAIN_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "visible_tutor_completion_loss_weight",
    "terminal_conversion_loss_weight",
    "collect_own_deck_promotion_metrics",
    "own_deck_promotion_metrics_closeout_threshold",
    "own_deck_promotion_metrics_terminal_ece_bins",
    "visible_tutor_completion_class_weights",
    "visible_tutor_completion_positive_weight",
    "terminal_conversion_class_weights",
    "terminal_conversion_positive_weight",
)
_REQUIRED_MODEL_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "own_deck_ledger_enabled",
    "own_deck_ledger_runtime_enabled",
    "own_deck_ledger_width",
    "own_deck_ledger_option_feature_dim",
    "visible_tutor_completion_head_enabled",
    "terminal_conversion_head_enabled",
    "visible_tutor_completion_route_enabled",
    "visible_tutor_completion_route_runtime_enabled",
    "terminal_conversion_route_enabled",
    "terminal_conversion_route_runtime_enabled",
)


class OwnDeckNextTrainContractError(ValueError):
    """Raised when a next-train input is not receipt-bound and fail-closed."""


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """An immutable content identity recorded in the prepared plan."""

    role: str
    identity: str
    sha256: str
    path: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "role": self.role,
            "id": self.identity,
            "sha256": self.sha256,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class BinaryCounts:
    """Observed scalar-label support; no counterfactual legal rows are used."""

    positive: int
    negative: int

    @property
    def total(self) -> int:
        return self.positive + self.negative

    def as_dict(self) -> dict[str, int]:
        return {"positive": self.positive, "negative": self.negative}


@dataclass(frozen=True, slots=True)
class SupervisionLabelCounts:
    """Aggregate selected-action factual labels from r259 sidecars."""

    terminal_class_counts: tuple[int, ...]
    terminal_scalars: tuple[tuple[str, BinaryCounts], ...]
    tutor_terminal_class_counts: tuple[int, ...]
    tutor_scalars: tuple[tuple[str, BinaryCounts], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "terminal_conversion": {
                "terminal_class": _class_counts_as_dict(self.terminal_class_counts),
                "scalars": {
                    name: count.as_dict() for name, count in self.terminal_scalars
                },
            },
            "visible_tutor_completion": {
                "same_actor_terminal_class": _class_counts_as_dict(
                    self.tutor_terminal_class_counts
                ),
                "scalars": {
                    name: count.as_dict() for name, count in self.tutor_scalars
                },
            },
        }


@dataclass(frozen=True, slots=True)
class DerivedSupervisionWeights:
    """Deterministic bounded weights generated solely from observed counts."""

    visible_tutor_completion_class_weights: tuple[float, ...]
    visible_tutor_completion_positive_weight: float
    terminal_conversion_class_weights: tuple[float, ...]
    terminal_conversion_positive_weight: float
    evidence: Mapping[str, Any]

    def train_config(self) -> dict[str, Any]:
        return {
            "visible_tutor_completion_loss_weight": R258_TUTOR_LOSS_WEIGHT,
            "terminal_conversion_loss_weight": R258_TERMINAL_LOSS_WEIGHT,
            # Offline/shadow-only factual telemetry is deliberately enabled
            # for the successor train.  The collector is detached from the
            # optimizer and has no runtime route authority.
            "collect_own_deck_promotion_metrics": True,
            "own_deck_promotion_metrics_closeout_threshold": float(
                DEFAULT_CLOSEOUT_THRESHOLD
            ),
            "own_deck_promotion_metrics_terminal_ece_bins": int(
                DEFAULT_TERMINAL_ECE_BINS
            ),
            "visible_tutor_completion_class_weights": list(
                self.visible_tutor_completion_class_weights
            ),
            "visible_tutor_completion_positive_weight": float(
                self.visible_tutor_completion_positive_weight
            ),
            "terminal_conversion_class_weights": list(
                self.terminal_conversion_class_weights
            ),
            "terminal_conversion_positive_weight": float(
                self.terminal_conversion_positive_weight
            ),
        }


@dataclass(frozen=True, slots=True)
class DailySidecarMeta:
    """One verified daily sidecar metadata receipt."""

    source_day: str
    meta_sha256: str
    shard_sha256: str
    record_count: int
    source_manifest_sha256: str
    sidecar_build_code_sha256: str
    sidecar_build_code_identities: tuple[tuple[str, str], ...]
    source_snapshot_tree_sha256: str
    image_id: str
    classifier_sha256: str
    label_counts: SupervisionLabelCounts

    def dataset_row(self) -> dict[str, Any]:
        return {
            "source_day": self.source_day,
            "daily_meta_sha256": self.meta_sha256,
            "daily_shard_sha256": self.shard_sha256,
            "record_count": self.record_count,
            "classifier_sha256": self.classifier_sha256,
            "label_counts": self.label_counts.as_dict(),
        }


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON suitable for semantic receipts."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OwnDeckNextTrainContractError("value is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    """Return a normalized raw-byte SHA-256 identity."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without following a symlink."""

    candidate = _regular_file(path, label="content")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash a source tree deterministically, rejecting symlinks and special files.

    A content-addressed source snapshot normally supplies a directory.  The
    hash covers every regular file under it in lexical relative-path order and
    rejects symlinks rather than allowing a later target substitution.  Git
    metadata and Python cache files are intentionally excluded; neither is
    executable source for the staged successor.
    """

    root = Path(path)
    try:
        info = root.lstat()
    except OSError as exc:
        raise OwnDeckNextTrainContractError(f"code path is unreadable: {root}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise OwnDeckNextTrainContractError("code path may not be a symlink")
    if stat.S_ISREG(info.st_mode):
        return sha256_file(root)
    if not stat.S_ISDIR(info.st_mode):
        raise OwnDeckNextTrainContractError("code path must be a regular file or directory")

    rows: list[tuple[str, str]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(root)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if candidate.suffix == ".pyc":
            continue
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise OwnDeckNextTrainContractError(
                f"code tree entry is unreadable: {candidate}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise OwnDeckNextTrainContractError(
                f"code tree may not contain symlink: {candidate}"
            )
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise OwnDeckNextTrainContractError(
                f"code tree may contain only regular files: {candidate}"
            )
        rows.append((relative.as_posix(), sha256_file(candidate)))
    if not rows:
        raise OwnDeckNextTrainContractError("code tree has no source files")
    return sha256_bytes(canonical_json_bytes({"files": rows}))


def receipt_digest(payload: Mapping[str, Any], *, field: str = "receipt_sha256") -> str:
    """Return a detached semantic digest for a receipt-like mapping."""

    detached = dict(payload)
    detached.pop(field, None)
    return sha256_bytes(canonical_json_bytes(detached))


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical copy with a detached ``receipt_sha256`` field."""

    sealed = json.loads(canonical_json_bytes(dict(payload)).decode("utf-8"))
    sealed.pop("receipt_sha256", None)
    sealed["receipt_sha256"] = receipt_digest(sealed)
    return sealed


def daily_meta_digest(payload: Mapping[str, Any]) -> str:
    """Return the r259 builder's exact detached ``meta_sha256`` digest.

    The side-store canonical JSON intentionally has no terminal newline, unlike
    this plan's receipt encoding.  Reusing its encoder prevents a superficial
    formatting difference from making a valid immutable daily shard appear
    forged (or vice versa).
    """

    detached = dict(payload)
    detached.pop("meta_sha256", None)
    try:
        encoded = sidecar_canonical_json_bytes(detached)
    except Exception as exc:  # pragma: no cover - source encoder is strict.
        raise OwnDeckNextTrainContractError(
            "daily sidecar metadata is not canonical JSON"
        ) from exc
    return sha256_bytes(encoded)


def sidecar_join_meta_identity(
    *, source_manifest_sha256: str, daily_meta_sha256s: Mapping[str, str]
) -> str:
    """Mirror dataset.py's immutable sidecar-meta provenance identity."""

    _require_sha256(source_manifest_sha256, label="join source manifest")
    normalized = _sha_mapping(daily_meta_sha256s, label="join daily metadata")
    if not normalized:
        raise OwnDeckNextTrainContractError("join daily metadata is empty")
    payload = {
        "schema": SIDE_STORE_JOIN_PROVENANCE_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "daily_meta_sha256s": dict(sorted(normalized.items())),
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - prevalidated map
        raise OwnDeckNextTrainContractError(
            "join sidecar identity is not canonical JSON"
        ) from exc
    return sha256_bytes(encoded)


def expected_sidecar_days(
    contract: successor.ElmoOwnDeckSideStoreContract | None = None,
) -> tuple[str, ...]:
    """Return the exact r259 source dates, inclusive and gap-free."""

    side_store = contract or successor.load_canonical_manifest().elmo_side_store
    try:
        start = date.fromisoformat(side_store.source_window.start_date)
        end = date.fromisoformat(side_store.source_window.end_date)
    except ValueError as exc:  # pragma: no cover - canonical manifest already checks.
        raise OwnDeckNextTrainContractError("r259 side-store date window is invalid") from exc
    if end < start:
        raise OwnDeckNextTrainContractError("r259 side-store date window is reversed")
    values = tuple(
        (start + timedelta(days=index)).isoformat()
        for index in range((end - start).days + 1)
    )
    if len(values) != int(side_store.source_window.day_count):
        raise OwnDeckNextTrainContractError("r259 side-store day-count contract changed")
    return values


def training_model_config() -> dict[str, Any]:
    """Return the future-only physical architecture with every serving gate off."""

    return {
        "own_deck_ledger_enabled": True,
        "own_deck_ledger_runtime_enabled": False,
        "own_deck_ledger_width": 128,
        "own_deck_ledger_option_feature_dim": OPTION_FEATURE_DIM,
        "visible_tutor_completion_head_enabled": True,
        "terminal_conversion_head_enabled": True,
        "visible_tutor_completion_route_enabled": True,
        "visible_tutor_completion_route_runtime_enabled": False,
        "terminal_conversion_route_enabled": True,
        "terminal_conversion_route_runtime_enabled": False,
    }


def derive_supervision_weights(
    counts: SupervisionLabelCounts,
    *,
    max_reweight: float = TRAIN_MAX_REWEIGHT,
) -> DerivedSupervisionWeights:
    """Derive bounded CE/BCE weights from r259 factual label support only.

    Categorical classes use inverse observed support normalized to expected
    mean weight one over observed rows.  BCE ``pos_weight`` is observed
    negatives divided by observed positives.  An absent class or missing
    binary complement is assigned *neutral* one because there is no observed
    support from which to infer an imbalance; it cannot fabricate a prior.
    """

    cap = _positive_finite(max_reweight, label="maximum reweight")
    if cap < 1.0:
        raise OwnDeckNextTrainContractError("maximum reweight must be at least 1")
    floor = 1.0 / cap
    terminal_classes = _derive_class_weights(
        counts.terminal_class_counts, floor=floor, cap=cap
    )
    tutor_classes = _derive_class_weights(
        counts.tutor_terminal_class_counts, floor=floor, cap=cap
    )
    terminal_scalars = dict(counts.terminal_scalars)
    tutor_scalars = dict(counts.tutor_scalars)
    terminal_positive = _aggregate_positive_weight(
        terminal_scalars, floor=floor, cap=cap
    )
    tutor_positive = _aggregate_positive_weight(tutor_scalars, floor=floor, cap=cap)
    evidence = {
        "schema": WEIGHT_DERIVATION_SCHEMA,
        "method": "inverse_observed_support_and_observed_negative_positive_ratio",
        "source": "r259 daily sidecar factual selected-action label counts only",
        "counterfactual_or_imputed_rows_used": False,
        "max_reweight": cap,
        "min_reweight": floor,
        "terminal_conversion": {
            "class_counts": _class_counts_as_dict(counts.terminal_class_counts),
            "class_weights": list(terminal_classes),
            "scalar_counts": {name: value.as_dict() for name, value in terminal_scalars.items()},
            "aggregate_positive_weight": terminal_positive,
        },
        "visible_tutor_completion": {
            "class_counts": _class_counts_as_dict(counts.tutor_terminal_class_counts),
            "class_weights": list(tutor_classes),
            "scalar_counts": {name: value.as_dict() for name, value in tutor_scalars.items()},
            "aggregate_positive_weight": tutor_positive,
        },
        "zero_support_rule": "neutral_1.0_no_invented_prior",
    }
    return DerivedSupervisionWeights(
        visible_tutor_completion_class_weights=tutor_classes,
        visible_tutor_completion_positive_weight=tutor_positive,
        terminal_conversion_class_weights=terminal_classes,
        terminal_conversion_positive_weight=terminal_positive,
        evidence=evidence,
    )


def prepare_next_train_plan(
    *,
    refresh_completion_receipt: Mapping[str, Any] | str | Path,
    stage_receipts: Mapping[
        successor.OwnDeckSuccessorStage | str, Mapping[str, Any] | str | Path
    ],
    migration_receipt: Mapping[str, Any] | str | Path,
    daily_meta_receipts: Sequence[Mapping[str, Any] | str | Path],
    join_receipt: Mapping[str, Any] | str | Path,
    parity_receipt: Mapping[str, Any] | str | Path,
    metric_support_receipt: Mapping[str, Any] | str | Path,
    source_manifest_identity: Mapping[str, Any] | str | Path,
    model_identity: Mapping[str, Any] | str | Path,
    code_identity: Mapping[str, Any] | str | Path,
    plan_id: str = "alakazam-own-deck-next-train-r258-r259",
) -> dict[str, Any]:
    """Build a dormant, receipt-bound configuration for the next train.

    This function is pure except for optional read-only input file hashing.  It
    returns a JSON-ready plan; use :func:`write_next_train_plan` to publish it
    immutably.  It deliberately does not call a trainer or any service API.
    """

    if not isinstance(plan_id, str) or not plan_id.strip() or "\x00" in plan_id:
        raise OwnDeckNextTrainContractError("plan id must be a non-empty string")
    manifest = successor.load_canonical_manifest()
    source = _content_identity(
        source_manifest_identity, role="expert_source_manifest", allow_tree=False
    )
    _validate_source_identity(source, manifest=manifest)
    code = _content_identity(code_identity, role="successor_code", allow_tree=True)

    # The canonical r258 guard is the authority boundary.  Passing daily
    # receipts alone cannot create a training canary permission.
    gate = successor.evaluate_successor_gate(
        successor.OwnDeckSuccessorOperation.TRAINING_CANARY,
        refresh_completion_receipt=refresh_completion_receipt,
        stage_receipts=stage_receipts,
        post_refresh_receipts={
            successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION: migration_receipt
        },
    )
    if not gate.allowed:
        raise OwnDeckNextTrainContractError(
            "canonical post-refresh training gate denied: " + gate.reason
        )
    refresh = successor.validate_refresh_completion_receipt(
        refresh_completion_receipt, manifest=manifest
    )
    validated_stages = successor.validate_prior_stage_receipts(
        stage_receipts, manifest=manifest
    )
    migration = successor.validate_post_refresh_receipt(
        migration_receipt,
        kind=successor.OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
        manifest=manifest,
        refresh_completion=refresh,
        stage_receipts=validated_stages,
        required_dependencies={},
    )
    model = _content_identity(
        model_identity, role="migrated_successor_checkpoint", allow_tree=False
    )
    migrated_checkpoint_sha256 = _validate_migration_checkpoint_binding(
        migration, refresh=refresh
    )
    if model.sha256 != migrated_checkpoint_sha256:
        raise OwnDeckNextTrainContractError(
            "model digest does not match the isolated migration child checkpoint"
        )

    daily = validate_daily_sidecar_receipts(
        daily_meta_receipts,
        manifest=manifest,
    )
    labels = aggregate_label_counts(daily)
    weights = derive_supervision_weights(labels)
    dataset_sha256 = sidecar_dataset_sha256(daily)
    join = _validate_join_receipt(
        join_receipt,
        manifest=manifest,
        source=source,
        model=model,
        code=code,
        daily=daily,
        dataset_sha256=dataset_sha256,
        migration_receipt_sha256=migration.sha256,
    )
    # This is the digest of the normalized four-part-key provenance, not a
    # hash of the detached receipt wrapper.  It is the value dataset.py
    # persists in the compact join receipt and the independent parity record.
    join_provenance_sha256 = str(join["join_provenance_sha256"])
    _validate_parity_receipt(
        parity_receipt,
        manifest=manifest,
        source=source,
        model=model,
        code=code,
        daily=daily,
        dataset_sha256=dataset_sha256,
        migration_receipt_sha256=migration.sha256,
        join_receipt_sha256=str(join["receipt_sha256"]),
        join_provenance_sha256=join_provenance_sha256,
    )
    parity = _load_receipt(parity_receipt, label="sidecar parity receipt")
    _validate_metric_support_receipt(
        metric_support_receipt,
        manifest=manifest,
        source=source,
        model=model,
        code=code,
        daily=daily,
        labels=labels,
        dataset_sha256=dataset_sha256,
        migration_receipt_sha256=migration.sha256,
        join_receipt_sha256=str(join["receipt_sha256"]),
        join_provenance_sha256=join_provenance_sha256,
        parity_receipt_sha256=parity["receipt_sha256"],
    )
    metric_support = _load_receipt(
        metric_support_receipt, label="promotion metric support receipt"
    )

    model_config = training_model_config()
    _validate_config_field_contract(model_config, weights.train_config())
    plan_base: dict[str, Any] = {
        "schema": NEXT_TRAIN_PLAN_SCHEMA,
        "status": "prepared_dormant_next_train_only",
        "plan_id": plan_id,
        "candidate_id": successor.CANDIDATE_ID,
        "owner_decision_revision": successor.OWNER_DECISION_REVISION,
        "owner_clarification_revision": successor.LATEST_OWNER_CLARIFICATION_REVISION,
        "manifest_sha256": manifest.identity.sha256,
        "gate": {
            "operation": successor.OwnDeckSuccessorOperation.TRAINING_CANARY.value,
            "allowed": True,
            "reason": gate.reason,
            "refresh_completion_receipt_sha256": refresh.sha256,
            "migration_receipt_sha256": migration.sha256,
            "refresh_parent_checkpoint_sha256": refresh.checkpoint_sha256,
            "migration_child_checkpoint_sha256": migrated_checkpoint_sha256,
            "prior_stage_receipt_sha256s": {
                stage.value: receipt.sha256
                for stage, receipt in validated_stages.items()
            },
        },
        "identities": {
            "source_manifest": source.as_dict(),
            "model_checkpoint": model.as_dict(),
            "successor_code": code.as_dict(),
            "sidecar_build": _sidecar_build_identity(daily),
            "sidecar_dataset": {
                "schema": SIDE_STORE_DATASET_IDENTITY_SCHEMA,
                "sha256": dataset_sha256,
                "source_window": list(expected_sidecar_days(manifest.elmo_side_store)),
                "daily_meta_sha256s": {
                    item.source_day: item.meta_sha256 for item in daily
                },
                "daily_shard_sha256s": {
                    item.source_day: item.shard_sha256 for item in daily
                },
                "record_count": sum(item.record_count for item in daily),
            },
            "join_sidecar_meta_identity": join["sidecar_meta_identity"],
            "join_receipt_sha256": join["receipt_sha256"],
            "join_provenance_sha256": join_provenance_sha256,
            "parity_receipt_sha256": parity["receipt_sha256"],
            "metric_support_receipt_sha256": metric_support["receipt_sha256"],
        },
        "sidecar_contract": {
            "source_access": "read_only",
            "active_r241_training_eligible": False,
            "evaluation_or_kaggle_training_eligible": False,
            "joined_training_dataset_materialized_by_this_plan": False,
            "detached_exact_key_join_receipt_validated": True,
            "record_key": list(manifest.elmo_side_store.record_key),
            "ledger_schema": {
                "schema": OWN_DECK_LEDGER_SCHEMA,
                "version": OWN_DECK_LEDGER_SCHEMA_VERSION,
            },
            "supervision_schema": {
                "schema": OWN_DECK_SUPERVISION_SCHEMA,
                "version": OWN_DECK_SUPERVISION_VERSION,
            },
            "daily_records": [item.dataset_row() for item in daily],
            "aggregate_label_counts": labels.as_dict(),
        },
        "model_config": model_config,
        "train_config": {
            **weights.train_config(),
            "loss_budget": {
                "schema": "poke_bot.alakazam_own_deck_auxiliary_budget/v1",
                "total": R258_TACTICAL_AUXILIARY_BUDGET,
                "allocation": "equal_across_visible_tutor_and_terminal_conversion",
                "visible_tutor_completion": R258_TUTOR_LOSS_WEIGHT,
                "terminal_conversion": R258_TERMINAL_LOSS_WEIGHT,
            },
            "class_weight_derivation": dict(weights.evidence),
        },
        "runtime_gates": {
            "own_deck_ledger_runtime_enabled": False,
            "visible_tutor_completion_route_runtime_enabled": False,
            "terminal_conversion_route_runtime_enabled": False,
            "runtime_action_authority": False,
        },
        "required_future_metrics": {
            "schema": OWN_DECK_PROMOTION_METRICS_SCHEMA,
            "scope": "observed_expert_labels_only_not_counterfactual_legal_actions",
            "visible_tutor": [
                "visible_tutor_observed_menu_expert_top1.agreement",
            ],
            "terminal_conversion": [
                "selected_option_factual_recall.own_win.recall",
                "selected_option_factual_recall.prize_closeout.recall",
                "selected_option_factual_recall.opponent_knockout.recall",
                "terminal_multiclass.brier",
                "terminal_multiclass.ece",
                "missed_expert_closeout.miss_rate",
                f"missed_expert_closeout.basis={MISSED_EXPERT_CLOSEOUT_BASIS}",
            ],
            "explicitly_not_claimed": [
                "counterfactual_unselected_legal_action_lethality",
                "hidden_deck_or_prize_identity",
                "search_or_rollout_value",
            ],
        },
        "execution_readiness": {
            "ready": False,
            "this_plan_materializes_no_joined_dataset": True,
            "this_plan_starts_no_trainer_or_service": True,
            "validated_join_receipt_is_not_training_authority": True,
            "next_authority_is_a_separate_training_canary_receipt": True,
        },
        "authority": _inert_authority(),
        "next_required_receipt": {
            "schema": successor.POST_REFRESH_RECEIPT_SCHEMA,
            "kind": successor.OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY.value,
            "separate_from_this_plan": True,
            "required_before_any_following_evaluation_or_runtime_activation": True,
        },
    }
    plan_sha256 = sha256_bytes(canonical_json_bytes(plan_base))
    receipt = seal_receipt(
        {
            "schema": NEXT_TRAIN_RECEIPT_SCHEMA,
            "status": "prepared_dormant_next_train_only",
            "plan_sha256": plan_sha256,
            "candidate_id": successor.CANDIDATE_ID,
            "manifest_sha256": manifest.identity.sha256,
            "source_manifest_sha256": source.sha256,
            "model_checkpoint_sha256": model.sha256,
            "refresh_completion_receipt_sha256": refresh.sha256,
            "migration_receipt_sha256": migration.sha256,
            "refresh_parent_checkpoint_sha256": refresh.checkpoint_sha256,
            "migration_child_checkpoint_sha256": migrated_checkpoint_sha256,
            "successor_code_sha256": code.sha256,
            "sidecar_build_code_sha256": daily[0].sidecar_build_code_sha256,
            "sidecar_source_snapshot_tree_sha256": daily[0].source_snapshot_tree_sha256,
            "sidecar_container_image_id": daily[0].image_id,
            "sidecar_archive_native_classifier_sha256": daily[0].classifier_sha256,
            "sidecar_dataset_sha256": dataset_sha256,
            "join_sidecar_meta_identity": join["sidecar_meta_identity"],
            "join_receipt_sha256": join["receipt_sha256"],
            "join_provenance_sha256": join_provenance_sha256,
            "parity_receipt_sha256": parity["receipt_sha256"],
            "metric_support_receipt_sha256": metric_support["receipt_sha256"],
            "joined_training_dataset_materialized": False,
            "training_execution_started": False,
            "managed_service_action_taken": False,
            "selector_change": False,
            "package_creation": False,
            "submission": False,
        }
    )
    plan = {**plan_base, "plan_sha256": plan_sha256, "receipt": receipt}
    validate_next_train_plan(plan)
    return plan


def validate_daily_sidecar_receipts(
    values: Sequence[Mapping[str, Any] | str | Path],
    *,
    manifest: successor.OwnDeckSuccessorManifest | None = None,
) -> tuple[DailySidecarMeta, ...]:
    """Validate one complete, nonduplicated r259 sidecar meta receipt per day."""

    contract = manifest or successor.load_canonical_manifest()
    expected_dates = expected_sidecar_days(contract.elmo_side_store)
    parsed: dict[str, DailySidecarMeta] = {}
    for value in values:
        meta = _parse_daily_meta(value, manifest=contract)
        if meta.source_day in parsed:
            raise OwnDeckNextTrainContractError(
                f"duplicate daily sidecar metadata for {meta.source_day}"
            )
        parsed[meta.source_day] = meta
    if tuple(sorted(parsed)) != expected_dates:
        missing = sorted(set(expected_dates) - set(parsed))
        extra = sorted(set(parsed) - set(expected_dates))
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if extra:
            parts.append("unexpected=" + ",".join(extra))
        raise OwnDeckNextTrainContractError(
            "sidecar daily metadata is incomplete for the exact r259 window ("
            + "; ".join(parts)
            + ")"
        )
    if sum(item.record_count for item in parsed.values()) <= 0:
        raise OwnDeckNextTrainContractError("sidecar dataset has no acting-seat records")
    ordered = tuple(parsed[day] for day in expected_dates)
    if len({item.sidecar_build_code_sha256 for item in ordered}) != 1:
        raise OwnDeckNextTrainContractError(
            "daily sidecars do not share one exact r259 build-code identity"
        )
    if len({item.source_snapshot_tree_sha256 for item in ordered}) != 1:
        raise OwnDeckNextTrainContractError(
            "daily sidecars do not share one exact source snapshot identity"
        )
    if len({item.image_id for item in ordered}) != 1:
        raise OwnDeckNextTrainContractError(
            "daily sidecars do not share one exact container image identity"
        )
    if len({item.classifier_sha256 for item in ordered}) != 1:
        raise OwnDeckNextTrainContractError(
            "daily sidecars do not share one exact archive-native classifier identity"
        )
    return ordered


def aggregate_label_counts(daily: Sequence[DailySidecarMeta]) -> SupervisionLabelCounts:
    """Sum sidecar count receipts without reading or relabelling data rows."""

    if not daily:
        raise OwnDeckNextTrainContractError("cannot aggregate zero daily sidecars")
    n_classes = len(TERMINAL_CONVERSION_CLASSES)
    terminal_classes = [0] * n_classes
    tutor_classes = [0] * n_classes
    terminal_scalars = {
        name: BinaryCounts(0, 0) for name in TERMINAL_CONVERSION_SCALAR_TARGET_NAMES
    }
    tutor_scalars = {
        name: BinaryCounts(0, 0)
        for name in VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES
    }
    for item in daily:
        for index, count in enumerate(item.label_counts.terminal_class_counts):
            terminal_classes[index] += count
        for index, count in enumerate(item.label_counts.tutor_terminal_class_counts):
            tutor_classes[index] += count
        for name, count in item.label_counts.terminal_scalars:
            prior = terminal_scalars[name]
            terminal_scalars[name] = BinaryCounts(
                prior.positive + count.positive, prior.negative + count.negative
            )
        for name, count in item.label_counts.tutor_scalars:
            prior = tutor_scalars[name]
            tutor_scalars[name] = BinaryCounts(
                prior.positive + count.positive, prior.negative + count.negative
            )
    return SupervisionLabelCounts(
        terminal_class_counts=tuple(terminal_classes),
        terminal_scalars=tuple((name, terminal_scalars[name]) for name in terminal_scalars),
        tutor_terminal_class_counts=tuple(tutor_classes),
        tutor_scalars=tuple((name, tutor_scalars[name]) for name in tutor_scalars),
    )


def sidecar_dataset_sha256(daily: Sequence[DailySidecarMeta]) -> str:
    """Derive one semantic dataset identity from every complete daily receipt."""

    if not daily:
        raise OwnDeckNextTrainContractError("cannot digest zero daily sidecars")
    source = {item.source_manifest_sha256 for item in daily}
    code = {item.sidecar_build_code_sha256 for item in daily}
    source_snapshot = {item.source_snapshot_tree_sha256 for item in daily}
    image = {item.image_id for item in daily}
    classifier = {item.classifier_sha256 for item in daily}
    if (
        len(source) != 1
        or len(code) != 1
        or len(source_snapshot) != 1
        or len(image) != 1
        or len(classifier) != 1
    ):
        raise OwnDeckNextTrainContractError(
            "daily sidecars do not share source/build identity"
        )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema": SIDE_STORE_DATASET_IDENTITY_SCHEMA,
                "source_manifest_sha256": next(iter(source)),
                "sidecar_build_code_sha256": next(iter(code)),
                "source_snapshot_tree_sha256": next(iter(source_snapshot)),
                "container_image_id": next(iter(image)),
                "archive_native_classifier_sha256": next(iter(classifier)),
                "days": [item.dataset_row() for item in daily],
            }
        )
    )


def _sidecar_build_identity(daily: Sequence[DailySidecarMeta]) -> dict[str, Any]:
    """Project the one exact r259 build identity shared by all daily shards."""

    if not daily:
        raise OwnDeckNextTrainContractError("cannot project a zero-day sidecar build")
    first = daily[0]
    expected_codes = first.sidecar_build_code_identities
    for item in daily[1:]:
        if (
            item.sidecar_build_code_sha256 != first.sidecar_build_code_sha256
            or item.sidecar_build_code_identities != expected_codes
            or item.source_snapshot_tree_sha256 != first.source_snapshot_tree_sha256
            or item.image_id != first.image_id
            or item.classifier_sha256 != first.classifier_sha256
        ):
            raise OwnDeckNextTrainContractError(
                "daily sidecars do not share an exact r259 build identity"
            )
    return {
        "sidecar_build_code_sha256": first.sidecar_build_code_sha256,
        "code_identities": dict(first.sidecar_build_code_identities),
        "source_snapshot_tree_sha256": first.source_snapshot_tree_sha256,
        "container_image_id": first.image_id,
        "archive_native_classifier_sha256": first.classifier_sha256,
    }


def validate_next_train_plan(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate a prepared plan/receipt without granting execution authority."""

    plan = _load_json_object(value, label="next-train plan")
    _require_exact(plan.get("schema"), NEXT_TRAIN_PLAN_SCHEMA, label="plan schema")
    _require_exact(
        plan.get("status"), "prepared_dormant_next_train_only", label="plan status"
    )
    _require_exact(plan.get("candidate_id"), successor.CANDIDATE_ID, label="candidate")
    _require_sha256(plan.get("manifest_sha256"), label="plan manifest digest")
    _require_sha256(plan.get("plan_sha256"), label="plan digest")
    base = dict(plan)
    declared_plan_sha = base.pop("plan_sha256")
    receipt = base.pop("receipt", None)
    if sha256_bytes(canonical_json_bytes(base)) != declared_plan_sha:
        raise OwnDeckNextTrainContractError("next-train plan fingerprint mismatch")
    if not isinstance(receipt, Mapping):
        raise OwnDeckNextTrainContractError("next-train plan lacks receipt")
    _require_exact(receipt.get("schema"), NEXT_TRAIN_RECEIPT_SCHEMA, label="receipt schema")
    _require_exact(
        receipt.get("plan_sha256"), declared_plan_sha, label="receipt plan binding"
    )
    _validate_self_receipt(receipt, label="next-train receipt")
    for key in (
        "joined_training_dataset_materialized",
        "training_execution_started",
        "managed_service_action_taken",
        "selector_change",
        "package_creation",
        "submission",
    ):
        _require_exact(receipt.get(key), False, label=f"next-train receipt {key}")
    gate = _mapping(plan.get("gate"), label="plan gate")
    _require_exact(
        gate.get("operation"),
        successor.OwnDeckSuccessorOperation.TRAINING_CANARY.value,
        label="plan gate operation",
    )
    _require_exact(gate.get("allowed"), True, label="plan gate allowed")
    for key in (
        "refresh_completion_receipt_sha256",
        "migration_receipt_sha256",
        "refresh_parent_checkpoint_sha256",
        "migration_child_checkpoint_sha256",
    ):
        _require_sha256(gate.get(key), label=f"plan gate {key}")
    authority = _mapping(plan.get("authority"), label="plan authority")
    for key, value_expected in _inert_authority().items():
        _require_exact(authority.get(key), value_expected, label=f"plan authority {key}")
    model_config = _mapping(plan.get("model_config"), label="plan model config")
    _validate_training_model_config(model_config)
    runtime = _mapping(plan.get("runtime_gates"), label="plan runtime gates")
    for name in (
        "own_deck_ledger_runtime_enabled",
        "visible_tutor_completion_route_runtime_enabled",
        "terminal_conversion_route_runtime_enabled",
        "runtime_action_authority",
    ):
        _require_exact(runtime.get(name), False, label=f"runtime gate {name}")
    train_config = _mapping(plan.get("train_config"), label="plan train config")
    _validate_train_config_values(train_config)
    identities = _mapping(plan.get("identities"), label="plan identities")
    identity_rows: dict[str, dict[str, Any]] = {}
    for name in ("source_manifest", "model_checkpoint", "successor_code"):
        row = _mapping(identities.get(name), label=f"plan identity {name}")
        _require_sha256(row.get("sha256"), label=f"plan identity {name} digest")
        identity_rows[name] = row
    dataset = _mapping(identities.get("sidecar_dataset"), label="plan sidecar dataset")
    _require_exact(
        dataset.get("schema"), SIDE_STORE_DATASET_IDENTITY_SCHEMA, label="dataset schema"
    )
    _require_sha256(dataset.get("sha256"), label="dataset digest")
    build = _mapping(identities.get("sidecar_build"), label="plan sidecar build")
    _require_sha256(
        build.get("sidecar_build_code_sha256"), label="sidecar build code digest"
    )
    _require_sha256(
        build.get("source_snapshot_tree_sha256"), label="sidecar source snapshot digest"
    )
    _require_sha256(build.get("container_image_id"), label="sidecar image digest")
    _require_sha256(
        build.get("archive_native_classifier_sha256"),
        label="sidecar archive-native classifier digest",
    )
    _sha_mapping(build.get("code_identities"), label="sidecar build code identities")
    _require_sha256(
        identities.get("join_sidecar_meta_identity"), label="join metadata identity"
    )
    _require_sha256(
        identities.get("join_receipt_sha256"), label="join receipt digest"
    )
    _require_sha256(
        identities.get("join_provenance_sha256"), label="join provenance digest"
    )
    _require_sha256(
        identities.get("parity_receipt_sha256"), label="parity receipt digest"
    )
    _require_sha256(
        identities.get("metric_support_receipt_sha256"),
        label="metric support receipt digest",
    )
    # The receipt is not merely a statement that this plan existed: it must
    # duplicate every immutable identity from the plan/gate so an offline
    # reader can detect an internally self-consistent but cross-bound plan.
    for key, expected, label in (
        ("manifest_sha256", plan["manifest_sha256"], "receipt manifest binding"),
        (
            "source_manifest_sha256",
            identity_rows["source_manifest"]["sha256"],
            "receipt source binding",
        ),
        (
            "model_checkpoint_sha256",
            identity_rows["model_checkpoint"]["sha256"],
            "receipt model binding",
        ),
        (
            "successor_code_sha256",
            identity_rows["successor_code"]["sha256"],
            "receipt code binding",
        ),
        (
            "refresh_completion_receipt_sha256",
            gate["refresh_completion_receipt_sha256"],
            "receipt refresh binding",
        ),
        (
            "migration_receipt_sha256",
            gate["migration_receipt_sha256"],
            "receipt migration binding",
        ),
        (
            "refresh_parent_checkpoint_sha256",
            gate["refresh_parent_checkpoint_sha256"],
            "receipt refresh checkpoint binding",
        ),
        (
            "migration_child_checkpoint_sha256",
            gate["migration_child_checkpoint_sha256"],
            "receipt migration checkpoint binding",
        ),
        (
            "sidecar_build_code_sha256",
            build["sidecar_build_code_sha256"],
            "receipt sidecar code binding",
        ),
        (
            "sidecar_source_snapshot_tree_sha256",
            build["source_snapshot_tree_sha256"],
            "receipt sidecar source snapshot binding",
        ),
        (
            "sidecar_container_image_id",
            build["container_image_id"],
            "receipt sidecar image binding",
        ),
        (
            "sidecar_archive_native_classifier_sha256",
            build["archive_native_classifier_sha256"],
            "receipt sidecar classifier binding",
        ),
        ("sidecar_dataset_sha256", dataset["sha256"], "receipt dataset binding"),
        (
            "join_sidecar_meta_identity",
            identities["join_sidecar_meta_identity"],
            "receipt join metadata binding",
        ),
        (
            "join_receipt_sha256",
            identities["join_receipt_sha256"],
            "receipt join receipt binding",
        ),
        (
            "join_provenance_sha256",
            identities["join_provenance_sha256"],
            "receipt join provenance binding",
        ),
        (
            "parity_receipt_sha256",
            identities["parity_receipt_sha256"],
            "receipt parity binding",
        ),
        (
            "metric_support_receipt_sha256",
            identities["metric_support_receipt_sha256"],
            "receipt metric-support binding",
        ),
    ):
        _require_exact(receipt.get(key), expected, label=label)
    _require_exact(
        receipt.get("migration_child_checkpoint_sha256"),
        identity_rows["model_checkpoint"]["sha256"],
        label="receipt model/migration child binding",
    )
    readiness = _mapping(plan.get("execution_readiness"), label="execution readiness")
    for key, expected in {
        "ready": False,
        "this_plan_materializes_no_joined_dataset": True,
        "this_plan_starts_no_trainer_or_service": True,
        "validated_join_receipt_is_not_training_authority": True,
        "next_authority_is_a_separate_training_canary_receipt": True,
    }.items():
        _require_exact(readiness.get(key), expected, label=f"execution readiness {key}")
    return plan


def write_next_train_plan(path: str | Path, plan: Mapping[str, Any]) -> Path:
    """Publish a validated plan atomically without replacing an existing one."""

    validated = validate_next_train_plan(plan)
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"next-train plan already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(validated)
    fd, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish only a read-only inode.  Hard-link publication below is
        # no-overwrite and atomic, so a completed plan is never briefly
        # mutable at its public path.
        os.chmod(temp, 0o444)
        # `link` is a no-overwrite atomic publish: an existing target causes
        # FileExistsError rather than silently replacing an immutable plan.
        try:
            os.link(temp, target)
        except FileExistsError:
            raise FileExistsError(f"next-train plan already exists: {target}") from None
        except OSError as exc:
            raise OwnDeckNextTrainContractError(
                f"atomic immutable plan publish failed: {exc}"
            ) from exc
        _fsync_directory(target.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return target


def _parse_daily_meta(
    value: Mapping[str, Any] | str | Path,
    *,
    manifest: successor.OwnDeckSuccessorManifest,
) -> DailySidecarMeta:
    raw = _load_receipt(value, label="daily sidecar metadata")
    _require_exact(
        raw.get("schema"), SIDE_STORE_DAILY_META_SCHEMA, label="daily metadata schema"
    )
    _require_exact(
        raw.get("version"), SIDE_STORE_DAILY_META_VERSION, label="daily metadata version"
    )
    _require_exact(
        raw.get("owner_decision_revision"),
        R259_OWNER_DECISION_REVISION,
        label="daily metadata owner revision",
    )
    _require_exact(
        raw.get("status"), "complete_immutable_sidecar", label="daily metadata status"
    )
    _validate_daily_meta_self_digest(raw)
    source_day = _string_from(raw, ("day",), label="daily source day")
    try:
        date.fromisoformat(source_day)
    except ValueError as exc:
        raise OwnDeckNextTrainContractError("daily source day is not ISO-8601") from exc
    source = _mapping(raw.get("source"), label="daily source identity")
    source_manifest = _mapping(source.get("manifest"), label="daily source manifest")
    source_manifest_sha = _require_sha256(
        source_manifest.get("sha256"), label="daily source manifest"
    )
    if source_manifest_sha != manifest.elmo_side_store.source_manifest_sha256:
        raise OwnDeckNextTrainContractError("daily source manifest digest differs from r259")
    _require_exact(
        source_manifest.get("schema"),
        ARCHIVE_RECEIPT_SCHEMA,
        label="daily source manifest schema",
    )
    _require_exact(
        source_manifest.get("original_path"),
        manifest.elmo_side_store.source_manifest,
        label="daily source manifest original path",
    )
    _nonempty_string(
        source_manifest.get("locked_path"), label="daily source manifest locked path"
    )
    _require_exact(
        source_manifest.get("window_start"),
        manifest.elmo_side_store.source_window.start_date,
        label="daily source window start",
    )
    _require_exact(
        source_manifest.get("window_end"),
        manifest.elmo_side_store.source_window.end_date,
        label="daily source window end",
    )
    _require_exact(
        source_manifest.get("days"),
        manifest.elmo_side_store.source_window.day_count,
        label="daily source window count",
    )
    _require_exact(
        source_manifest.get("total_episodes"),
        manifest.elmo_side_store.source_window.validated_episode_count,
        label="daily source episode count",
    )
    versioned_receipt = _mapping(
        source.get("versioned_receipt"), label="daily versioned archive receipt"
    )
    _require_exact(
        versioned_receipt.get("sha256"),
        EXPECTED_VERSIONED_RECEIPT_SHA256,
        label="daily versioned archive receipt digest",
    )
    versioned_path = _nonempty_string(
        versioned_receipt.get("path"), label="daily versioned archive receipt path"
    )
    _require_exact(
        versioned_receipt.get("original_path"),
        versioned_path,
        label="daily versioned archive receipt original path",
    )
    _nonempty_string(
        versioned_receipt.get("locked_path"),
        label="daily versioned archive receipt locked path",
    )
    archive = _mapping(source.get("archive"), label="daily source archive")
    _require_exact(archive.get("date"), source_day, label="daily source archive day")
    _require_sha256(archive.get("sha256"), label="daily source archive digest")
    _nonnegative_int(archive.get("bytes"), label="daily source archive bytes")
    _nonnegative_int(
        archive.get("validated_episode_count"), label="daily source archive record count"
    )
    shard_sha = _require_sha256(raw.get("shard_sha256"), label="daily shard")
    shard = _mapping(raw.get("shard"), label="daily shard metadata")
    _require_exact(shard.get("path"), DAILY_SHARD_NAME, label="daily shard path")
    _require_exact(shard.get("sha256"), shard_sha, label="daily shard metadata digest")
    _require_exact(shard.get("compression"), "gzip", label="daily shard compression")
    _require_exact(shard.get("format"), "jsonl", label="daily shard format")
    _require_exact(
        shard.get("row_schema"), OWN_DECK_ROLLOUT_SIDECAR_SCHEMA, label="daily row schema"
    )
    _require_exact(
        shard.get("row_version"), OWN_DECK_ROLLOUT_SIDECAR_VERSION, label="daily row version"
    )
    _nonnegative_int(shard.get("bytes"), label="daily shard bytes")
    _require_sha256(raw.get("rows_sha256"), label="daily row digest")
    _require_sha256(raw.get("source_records_sha256"), label="daily source record digest")
    record_count = _nonnegative_int(raw.get("row_count"), label="daily row count")
    _nonnegative_int(raw.get("source_record_count"), label="daily source record count")
    build = _mapping(raw.get("build"), label="daily build identity")
    _require_exact(
        build.get("mode"), "archive_native", label="daily archive-native source mode"
    )
    snapshot = _mapping(build.get("source_snapshot"), label="daily source snapshot")
    source_snapshot_tree_sha = _require_sha256(
        snapshot.get("tree_sha256"), label="daily source snapshot tree"
    )
    _require_exact(
        snapshot.get("path"), manifest.elmo_side_store.source_snapshot_root,
        label="daily source snapshot path",
    )
    image = _mapping(build.get("image"), label="daily container image")
    _require_exact(
        image.get("tag"), manifest.elmo_side_store.container_image, label="daily image tag"
    )
    image_id = _require_sha256(image.get("id"), label="daily image id")
    _require_exact(
        image_id, manifest.elmo_side_store.container_image_id, label="daily image identity"
    )
    code_identities = _sha_mapping(build.get("code"), label="daily build code")
    if not code_identities:
        raise OwnDeckNextTrainContractError("daily build code identities are empty")
    code_identity_rows = tuple(sorted(code_identities.items()))
    code_sha = sha256_bytes(
        canonical_json_bytes({"r259_build_code": dict(code_identity_rows)})
    )
    _require_exact(
        build.get("protected_stream_sha256"),
        None,
        label="daily archive-native protected-stream identity",
    )
    classifier_sha = _archive_native_classifier_sha256(build.get("classifier"))
    eligibility = _mapping(raw.get("training_eligibility"), label="daily eligibility")
    _require_exact(eligibility.get("active_r241"), False, label="daily r241 eligibility")
    _require_exact(eligibility.get("sidecar_only"), True, label="daily sidecar-only eligibility")
    _require_exact(
        eligibility.get("successor"),
        "pending_refresh_join_parity_receipt",
        label="daily successor eligibility",
    )
    label_counts_raw = _mapping(raw.get("label_counts"), label="daily label counts")
    labels = _parse_label_counts(label_counts_raw)
    return DailySidecarMeta(
        source_day=source_day,
        meta_sha256=str(raw["meta_sha256"]),
        shard_sha256=shard_sha,
        record_count=record_count,
        source_manifest_sha256=source_manifest_sha,
        sidecar_build_code_sha256=code_sha,
        sidecar_build_code_identities=code_identity_rows,
        source_snapshot_tree_sha256=source_snapshot_tree_sha,
        image_id=image_id,
        classifier_sha256=classifier_sha,
        label_counts=labels,
    )


def _parse_label_counts(value: Mapping[str, Any]) -> SupervisionLabelCounts:
    terminal = _mapping(value.get("terminal_conversion"), label="terminal label counts")
    tutor = _mapping(
        value.get("visible_tutor_completion"), label="visible tutor label counts"
    )
    terminal_classes = _class_counts_from(
        terminal,
        keys=("terminal_class", "categorical_class_counts", "terminal_class_counts"),
        label="terminal class counts",
    )
    tutor_classes = _class_counts_from(
        tutor,
        keys=(
            "same_actor_terminal_class",
            "categorical_class_counts",
            "same_actor_terminal_class_counts",
        ),
        label="tutor terminal class counts",
    )
    terminal_scalars = _scalar_counts_from(
        terminal, TERMINAL_CONVERSION_SCALAR_TARGET_NAMES, label="terminal scalar counts"
    )
    tutor_scalars = _scalar_counts_from(
        tutor, VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES, label="tutor scalar counts"
    )
    return SupervisionLabelCounts(
        terminal_class_counts=terminal_classes,
        terminal_scalars=terminal_scalars,
        tutor_terminal_class_counts=tutor_classes,
        tutor_scalars=tutor_scalars,
    )


def _archive_native_classifier_sha256(value: object) -> str:
    """Validate and digest the pinned archive-native seat classifier identity.

    Archive-native r259 construction needs all three file identities plus the
    classifier's own immutable contract.  The daily metadata self-digest
    already protects the complete JSON object; retaining this derived identity
    makes cross-day drift explicit in the successor dataset/plan receipt.
    """

    classifier = _mapping(value, label="daily archive-native classifier")
    classifier_contract = _mapping(
        classifier.get("contract"), label="daily archive-native classifier contract"
    )
    if not classifier_contract:
        raise OwnDeckNextTrainContractError(
            "daily archive-native classifier contract is empty"
        )
    for name in ("mix", "representatives", "card_csv"):
        artifact = _mapping(
            classifier.get(name), label=f"daily archive-native classifier {name}"
        )
        _nonempty_string(
            artifact.get("path"), label=f"daily archive-native classifier {name} path"
        )
        _require_sha256(
            artifact.get("sha256"),
            label=f"daily archive-native classifier {name} digest",
        )
    return sha256_bytes(canonical_json_bytes(classifier))


def _validate_join_receipt(
    value: Mapping[str, Any] | str | Path,
    *,
    manifest: successor.OwnDeckSuccessorManifest,
    source: ContentIdentity,
    model: ContentIdentity,
    code: ContentIdentity,
    daily: Sequence[DailySidecarMeta],
    dataset_sha256: str,
    migration_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate dataset.py's sealed exact-key join receipt and provenance.

    The compact provenance is copied into dataset conversion stats and every
    sequence's ``target_provenance``.  It is intentionally insufficient on
    its own: the detached immutable receipt must also bind that materialized
    four-part-key join to the exact successor model, code, migration, and
    semantic sidecar dataset used for this dormant plan.
    """

    raw = _load_receipt(value, label="sidecar join receipt")
    _require_exact(raw.get("schema"), SIDE_STORE_JOIN_RECEIPT_SCHEMA, label="join schema")
    if raw.get("status") not in {"passed", "complete", "completed"}:
        raise OwnDeckNextTrainContractError("sidecar join receipt is not complete")
    _validate_self_receipt(raw, label="sidecar join receipt")
    for key, expected, label in (
        ("manifest_sha256", manifest.identity.sha256, "join manifest binding"),
        ("source_manifest_sha256", source.sha256, "join source binding"),
        ("code_sha256", code.sha256, "join code binding"),
        ("training_code_sha256", code.sha256, "join training code binding"),
        ("model_sha256", model.sha256, "join model binding"),
        ("sidecar_dataset_sha256", dataset_sha256, "join dataset binding"),
        (
            "migration_receipt_sha256",
            migration_receipt_sha256,
            "join migration binding",
        ),
        (
            "join_provenance_schema",
            SIDE_STORE_JOIN_PROVENANCE_SCHEMA,
            "join provenance schema",
        ),
    ):
        _require_exact(raw.get(key), expected, label=label)
    expected_daily = {item.source_day: item.meta_sha256 for item in daily}
    if _sha_mapping(raw.get("daily_meta_sha256s"), label="join daily metadata") != expected_daily:
        raise OwnDeckNextTrainContractError("join receipt does not bind every daily metadata receipt")
    _require_exact(
        raw.get("sidecar_meta_identity"),
        sidecar_join_meta_identity(
            source_manifest_sha256=source.sha256,
            daily_meta_sha256s=expected_daily,
        ),
        label="join sidecar metadata identity",
    )
    _require_exact(
        tuple(raw.get("record_key") or ()),
        tuple(manifest.elmo_side_store.record_key),
        label="join canonical record key",
    )
    records = sum(item.record_count for item in daily)
    _require_exact(
        _nonnegative_int_from(
            raw, ("sidecar_record_count",), label="join sidecar record count"
        ),
        records,
        label="join sidecar record count",
    )
    _require_exact(
        _nonnegative_int_from(raw, ("joined_decision_count",), label="join decision count"),
        records,
        label="join decision count",
    )
    _require_exact(
        _nonnegative_int_from(
            raw, ("unmatched_record_count",), label="join unmatched record count"
        ),
        0,
        label="join unmatched record count",
    )
    _require_exact(
        _nonnegative_int_from(
            raw, ("duplicate_key_count",), label="join duplicate record count"
        ),
        0,
        label="join duplicate record count",
    )
    _require_exact(
        raw.get("one_to_one_coverage"), True, label="join one-to-one coverage"
    )
    _require_exact(
        raw.get("canonical_record_key_coverage"),
        True,
        label="join canonical record-key coverage",
    )
    _require_exact(
        _nonnegative_int_from(
            raw,
            ("observation_fingerprint_parity_count",),
            label="join observation-fingerprint parity",
        ),
        records,
        label="join observation-fingerprint parity",
    )
    raw_parity = _nonnegative_int_from(
        raw, ("raw_reconstruction_parity_count",), label="join raw reconstruction parity"
    )
    if raw_parity > records:
        raise OwnDeckNextTrainContractError(
            "join raw-reconstruction parity exceeds joined decisions"
        )
    _require_exact(
        raw.get("active_r241_training_eligible"),
        False,
        label="join active r241 eligibility",
    )
    # Match dataset._validated_sidecar_join_provenance exactly.  The record
    # key is deliberately four-part (including the public observation
    # fingerprint), while raw reconstruction parity is only supplemental: a
    # pre-featurized shard can have zero raw snapshots and still prove the
    # fully validated public-key/target-provenance join.
    normalized_provenance = {
        "schema": SIDE_STORE_JOIN_PROVENANCE_SCHEMA,
        "source_manifest_sha256": source.sha256,
        "daily_meta_sha256s": dict(sorted(expected_daily.items())),
        "sidecar_meta_identity": sidecar_join_meta_identity(
            source_manifest_sha256=source.sha256,
            daily_meta_sha256s=expected_daily,
        ),
        "record_key": list(manifest.elmo_side_store.record_key),
        "sidecar_record_count": records,
        "joined_decision_count": records,
        "unmatched_record_count": 0,
        "duplicate_key_count": 0,
        "observation_fingerprint_parity_count": records,
        "raw_reconstruction_parity_count": raw_parity,
        "one_to_one_coverage": True,
        "canonical_record_key_coverage": True,
        "active_r241_training_eligible": False,
    }
    _require_exact(
        raw.get("join_provenance_sha256"),
        sha256_bytes(canonical_json_bytes(normalized_provenance)),
        label="join authoritative target-provenance digest",
    )
    return raw


def _validate_parity_receipt(
    value: Mapping[str, Any] | str | Path,
    *,
    manifest: successor.OwnDeckSuccessorManifest,
    source: ContentIdentity,
    model: ContentIdentity,
    code: ContentIdentity,
    daily: Sequence[DailySidecarMeta],
    dataset_sha256: str,
    migration_receipt_sha256: str,
    join_receipt_sha256: str,
    join_provenance_sha256: str,
) -> None:
    raw = _load_receipt(value, label="sidecar parity receipt")
    _require_exact(raw.get("schema"), SIDE_STORE_PARITY_RECEIPT_SCHEMA, label="parity schema")
    if raw.get("status") not in {"passed", "complete", "completed"}:
        raise OwnDeckNextTrainContractError("sidecar parity receipt is not passed")
    _validate_self_receipt(raw, label="sidecar parity receipt")
    for key, expected, label in (
        ("manifest_sha256", manifest.identity.sha256, "parity manifest binding"),
        ("source_manifest_sha256", source.sha256, "parity source binding"),
        ("model_sha256", model.sha256, "parity model binding"),
        ("migration_receipt_sha256", migration_receipt_sha256, "parity migration binding"),
        ("training_code_sha256", code.sha256, "parity training code binding"),
        (
            "sidecar_build_code_sha256",
            daily[0].sidecar_build_code_sha256,
            "parity sidecar build code binding",
        ),
        ("sidecar_dataset_sha256", dataset_sha256, "parity dataset binding"),
        ("join_receipt_sha256", join_receipt_sha256, "parity join receipt binding"),
        ("join_provenance_sha256", join_provenance_sha256, "parity join binding"),
    ):
        _require_exact(raw.get(key), expected, label=label)
    expected_daily = {item.source_day: item.meta_sha256 for item in daily}
    if _sha_mapping(raw.get("daily_meta_sha256s"), label="parity daily metadata") != expected_daily:
        raise OwnDeckNextTrainContractError("parity receipt does not bind every daily metadata receipt")
    expected_shards = {item.source_day: item.shard_sha256 for item in daily}
    if _sha_mapping(raw.get("daily_shard_sha256s"), label="parity daily shards") != expected_shards:
        raise OwnDeckNextTrainContractError("parity receipt does not bind every daily shard")
    for key in (
        "exact_key_join_parity",
        "local_remote_parity",
        "ledger_parity",
        "supervision_parity",
        "public_information_only",
        "direct_policy_only",
    ):
        _require_exact(raw.get(key), True, label=f"parity {key}")


def _validate_metric_support_receipt(
    value: Mapping[str, Any] | str | Path,
    *,
    manifest: successor.OwnDeckSuccessorManifest,
    source: ContentIdentity,
    model: ContentIdentity,
    code: ContentIdentity,
    daily: Sequence[DailySidecarMeta],
    labels: SupervisionLabelCounts,
    dataset_sha256: str,
    migration_receipt_sha256: str,
    join_receipt_sha256: str,
    join_provenance_sha256: str,
    parity_receipt_sha256: str,
) -> None:
    """Require support for promotion-quality metrics before a plan is ready.

    This is deliberately a *support* receipt, not an evaluation result.  It
    proves that the exact factual sidecar has enough identified rows to later
    compute the required selected-expert metrics, without claiming an
    unobserved counterfactual closeout label or a canary outcome today.
    """

    raw = _load_receipt(value, label="promotion metric support receipt")
    _require_exact(
        raw.get("schema"), METRIC_SUPPORT_RECEIPT_SCHEMA, label="metric support schema"
    )
    if raw.get("status") not in {"passed", "complete", "completed"}:
        raise OwnDeckNextTrainContractError("promotion metric support receipt is not passed")
    _validate_self_receipt(raw, label="promotion metric support receipt")
    for key, expected, label in (
        ("manifest_sha256", manifest.identity.sha256, "metric support manifest binding"),
        ("source_manifest_sha256", source.sha256, "metric support source binding"),
        ("model_sha256", model.sha256, "metric support model binding"),
        (
            "migration_receipt_sha256",
            migration_receipt_sha256,
            "metric support migration binding",
        ),
        ("training_code_sha256", code.sha256, "metric support training code binding"),
        (
            "sidecar_build_code_sha256",
            daily[0].sidecar_build_code_sha256,
            "metric support sidecar build code binding",
        ),
        ("sidecar_dataset_sha256", dataset_sha256, "metric support dataset binding"),
        (
            "join_receipt_sha256",
            join_receipt_sha256,
            "metric support join receipt binding",
        ),
        (
            "join_provenance_sha256",
            join_provenance_sha256,
            "metric support join binding",
        ),
        ("parity_receipt_sha256", parity_receipt_sha256, "metric support parity binding"),
    ):
        _require_exact(raw.get(key), expected, label=label)
    expected_daily = {item.source_day: item.meta_sha256 for item in daily}
    if _sha_mapping(
        raw.get("daily_meta_sha256s"), label="metric support daily metadata"
    ) != expected_daily:
        raise OwnDeckNextTrainContractError(
            "metric support receipt does not bind every daily metadata receipt"
        )
    expected_shards = {item.source_day: item.shard_sha256 for item in daily}
    if _sha_mapping(
        raw.get("daily_shard_sha256s"), label="metric support daily shards"
    ) != expected_shards:
        raise OwnDeckNextTrainContractError(
            "metric support receipt does not bind every daily shard"
        )
    _require_exact(
        raw.get("metric_schema"),
        OWN_DECK_PROMOTION_METRICS_SCHEMA,
        label="metric support telemetry schema",
    )
    _require_exact(
        raw.get("missed_expert_closeout_basis"),
        MISSED_EXPERT_CLOSEOUT_BASIS,
        label="metric support missed-closeout basis",
    )
    for key in (
        "observed_selected_action_labels_only",
        "counterfactual_legal_action_labels_absent",
        "hidden_deck_or_prize_labels_absent",
    ):
        _require_exact(raw.get(key), True, label=f"metric support {key}")
    support = _mapping(raw.get("metric_support"), label="metric support counts")
    terminal_scalars = dict(labels.terminal_scalars)
    tutor_scalars = dict(labels.tutor_scalars)
    required = {
        "visible_tutor_observed_menu_expert_top1_denominator": tutor_scalars[
            "selected_from_visible_deck"
        ].positive,
        "terminal_multiclass_brier_ece_denominator": sum(labels.terminal_class_counts),
        "selected_option_factual_recall_own_win_denominator": labels.terminal_class_counts[
            TERMINAL_CONVERSION_CLASSES.index("own_win")
        ],
        "selected_option_factual_recall_prize_closeout_denominator": terminal_scalars[
            "prize_closeout"
        ].positive,
        "selected_option_factual_recall_opponent_knockout_denominator": terminal_scalars[
            "opponent_knockout"
        ].positive,
    }
    for key, expected in required.items():
        _require_exact(
            _nonnegative_int_from(support, (key,), label=f"metric support {key}"),
            expected,
            label=f"metric support {key}",
        )
    # The aggregate daily counters cannot reconstruct the union of own-win,
    # prize-closeout, and KO facts because those labels can overlap.  The
    # exact-key join materializer must count that factual union itself, and
    # bind it here rather than pretending prize-closeout alone is the rate's
    # denominator.
    closeout_denominator = _nonnegative_int_from(
        support,
        ("missed_observed_expert_closeout_denominator",),
        label="metric support missed observed expert closeout denominator",
    )
    if closeout_denominator > sum(labels.terminal_class_counts):
        raise OwnDeckNextTrainContractError(
            "metric support closeout denominator exceeds terminal factual rows"
        )
    if any(value <= 0 for value in required.values()) or closeout_denominator <= 0:
        raise OwnDeckNextTrainContractError(
            "metric support is incomplete for required observed-expert promotion metrics"
        )


def _content_identity(
    value: Mapping[str, Any] | str | Path,
    *,
    role: str,
    allow_tree: bool,
) -> ContentIdentity:
    if isinstance(value, (str, Path)):
        path = Path(value)
        digest = sha256_tree(path) if allow_tree else sha256_file(path)
        return ContentIdentity(role=role, identity=path.name, sha256=digest, path=str(path))
    raw = _mapping(value, label=f"{role} identity")
    identity = raw.get("id", raw.get("identity", role))
    if not isinstance(identity, str) or not identity.strip():
        raise OwnDeckNextTrainContractError(f"{role} identity id is invalid")
    digest = raw.get("sha256")
    _require_sha256(digest, label=f"{role} identity digest")
    path_value = raw.get("path")
    if path_value is not None and (not isinstance(path_value, str) or not path_value.strip()):
        raise OwnDeckNextTrainContractError(f"{role} identity path is invalid")
    if isinstance(path_value, str) and Path(path_value).exists():
        actual = sha256_tree(path_value) if allow_tree else sha256_file(path_value)
        if actual != digest:
            raise OwnDeckNextTrainContractError(f"{role} identity path digest mismatch")
    return ContentIdentity(role=role, identity=identity, sha256=str(digest), path=path_value)


def _validate_source_identity(
    value: ContentIdentity,
    *,
    manifest: successor.OwnDeckSuccessorManifest,
) -> None:
    expected = manifest.elmo_side_store
    if value.sha256 != expected.source_manifest_sha256:
        raise OwnDeckNextTrainContractError("source manifest digest differs from r259")
    if value.path is not None and value.path != expected.source_manifest:
        raise OwnDeckNextTrainContractError("source manifest path differs from r259")


def _validate_migration_checkpoint_binding(
    migration: successor.PostRefreshReceipt,
    *,
    refresh: successor.RefreshCompletionReceipt,
) -> str:
    """Require a real zero-safe migration artifact, not only a generic gate row."""

    raw = _mapping(migration.payload, label="isolated migration receipt")
    _require_exact(
        raw.get("migration_schema"),
        MIGRATION_RECEIPT_SCHEMA,
        label="migration artifact schema",
    )
    parent = _mapping(raw.get("parent_checkpoint"), label="migration parent checkpoint")
    child = _mapping(raw.get("child_checkpoint"), label="migration child checkpoint")
    _require_exact(
        parent.get("sha256"),
        refresh.checkpoint_sha256,
        label="migration parent refresh binding",
    )
    child_sha = _require_sha256(child.get("sha256"), label="migration child checkpoint")
    runtime = _mapping(raw.get("runtime_authority"), label="migration runtime authority")
    for key in (
        "own_deck_ledger_runtime_enabled",
        "visible_tutor_completion_route_runtime_enabled",
        "terminal_conversion_route_runtime_enabled",
        "selector_change_authorized",
        "package_or_submission_authorized",
        "serving_eligible",
    ):
        _require_exact(runtime.get(key), False, label=f"migration runtime authority {key}")
    verification = _mapping(raw.get("verification"), label="migration verification")
    _require_exact(
        verification.get("child_checkpoint_sha256"),
        child_sha,
        label="migration verification child binding",
    )
    _nonnegative_int(
        verification.get("inherited_tensor_count"), label="migration inherited tensor count"
    )
    added = verification.get("added_tensor_keys")
    if not isinstance(added, list) or not added or not all(
        isinstance(key, str) and key for key in added
    ):
        raise OwnDeckNextTrainContractError(
            "migration verification does not prove physical successor tensors"
        )
    return child_sha


def _validate_config_field_contract(
    model_config: Mapping[str, Any], train_config: Mapping[str, Any]
) -> None:
    _validate_training_model_config(model_config)
    _validate_train_config_values(train_config)
    # Prefer the live dataclasses.  Receipt-only environments intentionally do
    # not install PyTorch, so a missing torch dependency falls back to an AST
    # check of the same staged source rather than weakening the ABI check.
    try:
        from .config import ModelConfig
        from .train import TrainConfig
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise OwnDeckNextTrainContractError(
                "cannot verify successor config fields against training implementation"
            ) from exc
        model_fields = _dataclass_fields_from_source("config.py", "ModelConfig")
        train_fields = _dataclass_fields_from_source("train.py", "TrainConfig")
    except Exception as exc:  # pragma: no cover - integration-only failure path.
        raise OwnDeckNextTrainContractError(
            "cannot verify successor config fields against training implementation"
        ) from exc
    else:
        model_fields = set(getattr(ModelConfig, "__dataclass_fields__", {}))
        train_fields = set(getattr(TrainConfig, "__dataclass_fields__", {}))
    missing_model = set(_REQUIRED_MODEL_CONFIG_FIELDS) - model_fields
    missing_train = set(_REQUIRED_TRAIN_CONFIG_FIELDS) - train_fields
    if missing_model or missing_train:
        raise OwnDeckNextTrainContractError(
            "training implementation config fields drifted "
            f"(model={sorted(missing_model)}, train={sorted(missing_train)})"
        )


def _dataclass_fields_from_source(filename: str, class_name: str) -> set[str]:
    """Read annotated dataclass fields without importing torch-heavy train.py."""

    path = Path(__file__).with_name(filename)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:  # pragma: no cover - staged source failure.
        raise OwnDeckNextTrainContractError(
            f"cannot statically inspect {filename} for {class_name}"
        ) from exc
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        return {
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
    raise OwnDeckNextTrainContractError(
        f"cannot find {class_name} in staged {filename}"
    )


def _validate_training_model_config(value: Mapping[str, Any]) -> None:
    expected = training_model_config()
    if dict(value) != expected:
        raise OwnDeckNextTrainContractError(
            "next-train model config must enable physical successor modules and keep every runtime gate false"
        )


def _validate_train_config_values(value: Mapping[str, Any]) -> None:
    for field in _REQUIRED_TRAIN_CONFIG_FIELDS:
        if field not in value:
            raise OwnDeckNextTrainContractError(f"next-train config lacks {field}")
    _require_exact(
        value.get("collect_own_deck_promotion_metrics"),
        True,
        label="next-train factual promotion telemetry",
    )
    _require_exact(
        value.get("own_deck_promotion_metrics_closeout_threshold"),
        float(DEFAULT_CLOSEOUT_THRESHOLD),
        label="next-train closeout telemetry threshold",
    )
    _require_exact(
        value.get("own_deck_promotion_metrics_terminal_ece_bins"),
        int(DEFAULT_TERMINAL_ECE_BINS),
        label="next-train terminal telemetry ECE bins",
    )
    for name in (
        "visible_tutor_completion_loss_weight",
        "terminal_conversion_loss_weight",
    ):
        parsed = _positive_finite(value.get(name), label=name)
        if parsed > R258_TACTICAL_AUXILIARY_BUDGET:
            raise OwnDeckNextTrainContractError(f"{name} exceeds the r258 auxiliary budget")
    for name in (
        "visible_tutor_completion_class_weights",
        "terminal_conversion_class_weights",
    ):
        raw = value.get(name)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != len(TERMINAL_CONVERSION_CLASSES):
            raise OwnDeckNextTrainContractError(f"{name} must be a four-class vector")
        for index, weight in enumerate(raw):
            parsed = _positive_finite(weight, label=f"{name}[{index}]")
            if not TRAIN_MIN_REWEIGHT <= parsed <= TRAIN_MAX_REWEIGHT:
                raise OwnDeckNextTrainContractError(f"{name}[{index}] is outside train.py's reweight bound")
    for name in (
        "visible_tutor_completion_positive_weight",
        "terminal_conversion_positive_weight",
    ):
        parsed = _positive_finite(value.get(name), label=name)
        if not TRAIN_MIN_REWEIGHT <= parsed <= TRAIN_MAX_REWEIGHT:
            raise OwnDeckNextTrainContractError(f"{name} is outside train.py's reweight bound")


def _derive_class_weights(
    counts: Sequence[int], *, floor: float, cap: float
) -> tuple[float, ...]:
    normalized = tuple(_nonnegative_int(value, label="class count") for value in counts)
    if len(normalized) != len(TERMINAL_CONVERSION_CLASSES):
        raise OwnDeckNextTrainContractError("class count width does not match terminal ABI")
    total = sum(normalized)
    positive_classes = sum(1 for count in normalized if count > 0)
    if total <= 0 or positive_classes <= 0:
        return (1.0,) * len(normalized)
    # The formula is exactly N/(K*n_i) for supported classes.  Unsupported
    # classes never appear in CE and stay neutral, avoiding an invented prior.
    result = [1.0] * len(normalized)
    for index, count in enumerate(normalized):
        if count > 0:
            result[index] = _clamp(float(total) / (float(positive_classes) * float(count)), floor, cap)
    return tuple(result)


def _aggregate_positive_weight(
    counts: Mapping[str, BinaryCounts], *, floor: float, cap: float
) -> float:
    positives = sum(row.positive for row in counts.values())
    negatives = sum(row.negative for row in counts.values())
    if positives <= 0 or negatives <= 0:
        return 1.0
    return _clamp(float(negatives) / float(positives), floor, cap)


def _class_counts_from(
    row: Mapping[str, Any], *, keys: Sequence[str], label: str
) -> tuple[int, ...]:
    raw: object | None = None
    for key in keys:
        if key in row:
            raw = row[key]
            break
    if raw is None:
        raise OwnDeckNextTrainContractError(f"{label} is absent")
    if isinstance(raw, Mapping):
        result = tuple(_nonnegative_int(raw.get(name), label=f"{label}.{name}") for name in TERMINAL_CONVERSION_CLASSES)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = tuple(_nonnegative_int(value, label=label) for value in raw)
    else:
        raise OwnDeckNextTrainContractError(f"{label} must be a mapping or vector")
    if len(result) != len(TERMINAL_CONVERSION_CLASSES):
        raise OwnDeckNextTrainContractError(f"{label} has wrong class width")
    return result


def _scalar_counts_from(
    row: Mapping[str, Any], names: Sequence[str], *, label: str
) -> tuple[tuple[str, BinaryCounts], ...]:
    source = row.get("scalars", row.get("scalar_counts", row))
    source_map = _mapping(source, label=label)
    result: list[tuple[str, BinaryCounts]] = []
    for name in names:
        raw = _mapping(source_map.get(name), label=f"{label}.{name}")
        # The r259 side-store exposes typed ``labeled`` and ``positive``
        # counters.  Negatives are derived from those factual observations;
        # accepting a separately asserted negative count would weaken the
        # receipt's no-imputed-label guarantee.
        labeled = _nonnegative_int_from(raw, ("labeled",), label=f"{label}.{name}.labeled")
        positive = _nonnegative_int_from(raw, ("positive",), label=f"{label}.{name}.positive")
        if positive > labeled:
            raise OwnDeckNextTrainContractError(
                f"{label}.{name}.positive exceeds factual labeled support"
            )
        negative = labeled - positive
        result.append((name, BinaryCounts(positive, negative)))
    return tuple(result)


def _load_receipt(value: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    return _load_json_object(value, label=label)


def _load_json_object(value: Mapping[str, Any] | str | Path, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = _regular_file(value, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnDeckNextTrainContractError(f"{label} is not readable JSON") from exc
    return _mapping(payload, label=label)


def _validate_self_receipt(value: Mapping[str, Any], *, label: str) -> None:
    digest = value.get("receipt_sha256")
    _require_sha256(digest, label=f"{label} digest")
    if digest != receipt_digest(value):
        raise OwnDeckNextTrainContractError(f"{label} fingerprint mismatch")


def _validate_daily_meta_self_digest(value: Mapping[str, Any]) -> None:
    digest = value.get("meta_sha256")
    _require_sha256(digest, label="daily metadata digest")
    if digest != daily_meta_digest(value):
        raise OwnDeckNextTrainContractError("daily sidecar metadata fingerprint mismatch")


def _inert_authority() -> dict[str, bool]:
    return {
        "training_service_start_authorized": False,
        "gradient_updates_authorized": False,
        "managed_service_action_authorized": False,
        "runtime_action_authority": False,
        "selector_change_authorized": False,
        "package_creation_authorized": False,
        "submission_authorized": False,
        "promotion_authorized": False,
        "evaluation_games_training_eligible": False,
        "active_r241_training_eligible": False,
    }


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise OwnDeckNextTrainContractError(f"{label} path is unreadable: {candidate}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OwnDeckNextTrainContractError(f"{label} path must be a regular non-symlink file")
    return candidate.resolve()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnDeckNextTrainContractError(f"{label} must be an object")
    return dict(value)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise OwnDeckNextTrainContractError(f"{label} must be a lowercase sha256 identity")
    return value


def _require_exact(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise OwnDeckNextTrainContractError(f"{label} does not match the canonical contract")


def _string_from(row: Mapping[str, Any], keys: Sequence[str], *, label: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise OwnDeckNextTrainContractError(f"{label} is absent")


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OwnDeckNextTrainContractError(f"{label} must be a non-empty string")
    return value


def _sha_from(row: Mapping[str, Any], keys: Sequence[str], *, label: str) -> str:
    for key in keys:
        if key in row:
            return _require_sha256(row[key], label=label)
    raise OwnDeckNextTrainContractError(f"{label} is absent")


def _nonnegative_int_from(row: Mapping[str, Any], keys: Sequence[str], *, label: str) -> int:
    for key in keys:
        if key in row:
            return _nonnegative_int(row[key], label=label)
    raise OwnDeckNextTrainContractError(f"{label} is absent")


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OwnDeckNextTrainContractError(f"{label} must be a nonnegative integer")
    return int(value)


def _positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise OwnDeckNextTrainContractError(f"{label} must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OwnDeckNextTrainContractError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise OwnDeckNextTrainContractError(f"{label} must be a finite positive number")
    return parsed


def _sha_mapping(value: object, *, label: str) -> dict[str, str]:
    row = _mapping(value, label=label)
    result: dict[str, str] = {}
    for key, digest in row.items():
        if not isinstance(key, str) or not key:
            raise OwnDeckNextTrainContractError(f"{label} key is invalid")
        result[key] = _require_sha256(digest, label=f"{label}.{key}")
    return result


def _class_counts_as_dict(values: Sequence[int]) -> dict[str, int]:
    return {name: int(values[index]) for index, name in enumerate(TERMINAL_CONVERSION_CLASSES)}


def _clamp(value: float, floor: float, cap: float) -> float:
    return max(float(floor), min(float(cap), float(value)))


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - uncommon platform limitation.
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystem may not support it.
        pass
    finally:
        os.close(fd)


__all__ = [
    "NEXT_TRAIN_PLAN_SCHEMA",
    "NEXT_TRAIN_RECEIPT_SCHEMA",
    "SIDE_STORE_JOIN_PROVENANCE_SCHEMA",
    "SIDE_STORE_JOIN_RECEIPT_SCHEMA",
    "BinaryCounts",
    "ContentIdentity",
    "DailySidecarMeta",
    "DerivedSupervisionWeights",
    "OwnDeckNextTrainContractError",
    "SupervisionLabelCounts",
    "aggregate_label_counts",
    "canonical_json_bytes",
    "daily_meta_digest",
    "derive_supervision_weights",
    "expected_sidecar_days",
    "prepare_next_train_plan",
    "receipt_digest",
    "seal_receipt",
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
    "sidecar_dataset_sha256",
    "sidecar_join_meta_identity",
    "training_model_config",
    "validate_daily_sidecar_receipts",
    "validate_next_train_plan",
    "write_next_train_plan",
]
