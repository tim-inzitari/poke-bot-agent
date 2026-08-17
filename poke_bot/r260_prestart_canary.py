"""Receipt-gated, bounded r260 OwnDeckLedger pre-start canary.

This module is deliberately isolated from the managed r241 trainer, runtime
registry, service definitions, and launchers.  It has one narrow job: turn a
validated zero-safe migration child plus a completed Elmo/Inzi expert-data
binding into immutable *offline* evidence.  The only mutable computation is a
bounded deterministic optimizer loop supplied by the caller; it cannot start a
service, select a checkpoint, build a package, or submit anything.

The evidence sequence is intentionally split so no one receipt can quietly
stand in for another:

* the canary receipt proves finite/nonzero gradients, factual mask coverage,
  calibration support, public-information-only inputs, and a runtime-enabled
  child checkpoint;
* the evaluation receipt proves source-disjoint factual evaluation;
* the parity receipt proves equal local, Elmo, and replay feature digests;
* the influence receipt proves that the newly trained path has a finite,
  bounded policy effect; and
* the immutable runtime configuration binds all of those facts while retaining
  the external-overlay requirement for any managed-service action.

All writers are create-only.  A zero-safe migration child has all three new
runtime gates disabled and is explicitly rejected as a launch candidate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import torch
from torch import nn

from . import checkpoint
from . import own_deck_migration as migration
from .own_deck_successor import canonical_json, receipt_digest, seal_receipt
from .r241_own_deck_successor import (
    R260_CANARY_ACTIVATION_SCHEMA,
    R260_MIGRATION_KIND,
    R260_MIGRATION_SCHEMA,
    R241OwnDeckSuccessorError,
    R260OwnerContract,
    validate_r260_inzi_dataset_binding,
    validate_r260_migration_receipt,
    validate_r260_sidecar_binding,
)

R260_PRESTART_CANARY_CONFIG_SCHEMA: Final = (
    "poke_bot.r241_own_deck_prestart_canary_config/v1"
)
R260_PRESTART_CANARY_RECEIPT_SCHEMA: Final = (
    "poke_bot.r241_own_deck_prestart_canary_receipt/v1"
)
R260_PRESTART_EVALUATION_RECEIPT_SCHEMA: Final = (
    "poke_bot.r241_own_deck_prestart_evaluation_receipt/v1"
)
R260_PRESTART_PARITY_RECEIPT_SCHEMA: Final = (
    "poke_bot.r241_own_deck_prestart_parity_receipt/v1"
)
R260_PRESTART_INFLUENCE_RECEIPT_SCHEMA: Final = (
    "poke_bot.r241_own_deck_prestart_influence_receipt/v1"
)
R260_RUNTIME_ACTIVATION_CONFIG_SCHEMA: Final = (
    "poke_bot.r241_own_deck_runtime_activation_config/v1"
)
R260_INZI_STREAMING_INDEX_SCHEMA: Final = "poke_bot.r260_inzi_sidecar_index/v1"

MAX_CANARY_STEPS: Final = 32
MIN_CANARY_STEPS: Final = 2
R260_ROUTE_DELTA_CAP: Final = 1.0
R260_TUTOR_LOSS_WEIGHT: Final = 0.025
R260_TERMINAL_LOSS_WEIGHT: Final = 0.025
R274_TACTICAL_LOSS_WEIGHT: Final = 0.025
R274_BOOTSTRAP_TRAINABLE_PREFIXES: Final[tuple[str, ...]] = (
    migration.SUCCESSOR_TENSOR_PREFIXES
)
R260_INZI_TRAINER_INPUT: Final = (
    "local_inzi_disk_backed_exact_four_key_streaming_index_only"
)

RUNTIME_GATE_FIELDS: Final[tuple[str, ...]] = (
    "own_deck_ledger_runtime_enabled",
    "visible_tutor_completion_route_runtime_enabled",
    "terminal_conversion_route_runtime_enabled",
)
PHYSICAL_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "own_deck_ledger_enabled",
    "visible_tutor_completion_head_enabled",
    "terminal_conversion_head_enabled",
    "visible_tutor_completion_route_enabled",
    "terminal_conversion_route_enabled",
)
REQUIRED_COVERAGE_FIELDS: Final[tuple[str, ...]] = (
    "public_rows",
    "ledger_rows",
    "visible_tutor_labeled_rows",
    "visible_tutor_masked_rows",
    "terminal_labeled_rows",
    "terminal_masked_rows",
)
REQUIRED_CALIBRATION_FIELDS: Final[tuple[str, ...]] = (
    "visible_tutor_brier_sum",
    "visible_tutor_brier_count",
    "terminal_brier_sum",
    "terminal_brier_count",
    "terminal_ece_sum",
    "terminal_ece_count",
)
DATA_AUDIT_FIELDS: Final[tuple[str, ...]] = (
    "public_information_only",
    "direct_policy_only",
    "no_search_or_rtp",
    "no_hidden_state",
    "evaluation_or_kaggle_replay_used",
)
ACTIVATION_EVIDENCE_NAMES: Final[tuple[str, ...]] = (
    "finite_gradient",
    "source_disjoint_evaluation",
    "local_remote_parity",
    "bounded_influence",
)


class R260PrestartCanaryError(RuntimeError):
    """A purported r260 canary/evidence artifact is incomplete or unsafe."""


@dataclass(frozen=True)
class CanaryStep:
    """One factual, caller-supplied offline optimizer step.

    ``loss`` must be connected to the supplied model.  The module deliberately
    does not invent a loss or labels: callers keep the typed tutor/terminal
    target construction in their data layer and report only factual coverage
    and calibration sufficient statistics here.
    """

    loss: torch.Tensor
    source_ids: tuple[str, ...]
    coverage: Mapping[str, int]
    calibration: Mapping[str, float | int]
    public_information_only: bool
    direct_policy_only: bool
    no_search_or_rtp: bool
    no_hidden_state: bool
    evaluation_or_kaggle_replay_used: bool = False


class InziStreamingIndex(Protocol):
    """The narrow, verified index surface a canary may consume on Inzi."""

    path: Path

    def assert_verified(
        self,
        *,
        expected_source_manifest_sha256: str,
        daily_meta_sha256s: Mapping[str, str],
    ) -> None: ...


CanaryStepBuilder = Callable[[nn.Module, int, InziStreamingIndex], CanaryStep]


@dataclass(frozen=True)
class R260CanaryRunResult:
    """The create-only artifacts emitted by a successful bounded canary."""

    checkpoint: dict[str, Any]
    receipt: dict[str, Any]
    executed_steps: int


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    detached = dict(value)
    detached.pop(field, None)
    return _sha256_bytes(canonical_json(detached))


def _require_sha(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise R260PrestartCanaryError(f"{label} must be a SHA-256 identity")
    return text


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R260PrestartCanaryError(f"{label} must be an object")
    return value


def _regular_file(path: Path | str, *, label: str, immutable: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R260PrestartCanaryError(f"{label} must be a regular non-symlink file")
    resolved = candidate.resolve()
    if immutable and stat.S_IMODE(resolved.stat().st_mode) & 0o222:
        raise R260PrestartCanaryError(f"{label} must be immutable (no write bits)")
    return resolved


def file_identity(path: Path | str, *, immutable: bool = False) -> dict[str, Any]:
    """Return a strict `FileIdentity` record without following a symlink."""

    candidate = _regular_file(path, label="artifact", immutable=immutable)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(candidate),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": int(candidate.stat().st_size),
    }


def _validate_file_identity(
    value: object,
    *,
    label: str,
    verify_file: bool,
    immutable: bool = False,
) -> dict[str, Any]:
    row = dict(_mapping(value, label=label))
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise R260PrestartCanaryError(f"{label} must be an exact FileIdentity")
    path = str(row.get("path") or "")
    if not path:
        raise R260PrestartCanaryError(f"{label} path is missing")
    digest = _require_sha(row.get("sha256"), label=label)
    size = row.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise R260PrestartCanaryError(f"{label} size is invalid")
    result = {"path": path, "sha256": digest, "size_bytes": int(size)}
    if verify_file:
        observed = file_identity(path, immutable=immutable)
        if observed != result:
            raise R260PrestartCanaryError(f"{label} FileIdentity mismatch")
    return result


def _write_immutable_bytes(path: Path | str, payload: bytes, *, label: str) -> Path:
    destination = Path(path).expanduser()
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise R260PrestartCanaryError(f"{label} parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} already exists: {destination}")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.r260.", dir=parent
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)
    return destination


def _write_immutable_json(
    path: Path | str, payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    sealed = json.loads(canonical_json(dict(payload)).decode("utf-8"))
    _write_immutable_bytes(path, canonical_json(sealed), label=label)
    return file_identity(path, immutable=True)


def _load_immutable_json(
    path: Path | str, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _regular_file(path, label=label, immutable=True)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R260PrestartCanaryError(f"{label} is not readable JSON") from exc
    return dict(_mapping(value, label=label)), file_identity(candidate, immutable=True)


def _sealed_config(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("config_sha256", None)
    payload["config_sha256"] = _canonical_digest(payload, field="config_sha256")
    return payload


def _normalise_source_ids(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise R260PrestartCanaryError(f"{label} must be a sequence of opaque IDs")
    rows = tuple(str(item).strip() for item in value)
    if not rows or any(not item for item in rows) or len(set(rows)) != len(rows):
        raise R260PrestartCanaryError(f"{label} must be nonempty and duplicate-free")
    return rows


def _normalise_prefixes(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise R260PrestartCanaryError("inherited route prefixes must be a sequence")
    prefixes = tuple(str(item).strip() for item in value)
    if not prefixes or any(not item or not item.endswith(".") for item in prefixes):
        raise R260PrestartCanaryError(
            "inherited route prefixes must be nonempty dotted prefixes"
        )
    if len(set(prefixes)) != len(prefixes):
        raise R260PrestartCanaryError("inherited route prefixes must be unique")
    if any(
        prefix.startswith(migration.SUCCESSOR_TENSOR_PREFIXES) for prefix in prefixes
    ):
        raise R260PrestartCanaryError(
            "new successor prefixes cannot pose as inherited routes"
        )
    return prefixes


def _normalise_limits(*, seed: int, max_steps: int) -> tuple[int, int]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise R260PrestartCanaryError("canary seed must be an integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise R260PrestartCanaryError("canary max_steps must be an integer")
    if not MIN_CANARY_STEPS <= max_steps <= MAX_CANARY_STEPS:
        raise R260PrestartCanaryError(
            f"canary max_steps must be in [{MIN_CANARY_STEPS}, {MAX_CANARY_STEPS}]"
        )
    return int(seed), int(max_steps)


def _normalised_absolute_path(value: object, *, label: str) -> Path:
    """Normalise lexically without dereferencing a potentially hostile link."""

    text = str(value or "").strip()
    candidate = Path(text).expanduser()
    if not text or not candidate.is_absolute():
        raise R260PrestartCanaryError(f"{label} must be an absolute path")
    return Path(os.path.normpath(str(candidate)))


def _assert_final_inzi_path(
    value: object,
    *,
    owner_contract: R260OwnerContract,
    label: str,
) -> Path:
    """Accept a path only under the canonical final Inzi root.

    This is intentionally lexical rather than ``resolve()`` based: config
    validation must not follow an untrusted symlink simply to decide whether a
    path is eligible.  Actual artifact verification separately rejects direct
    symlink files.
    """

    candidate = _normalised_absolute_path(value, label=label)
    final_root = _normalised_absolute_path(
        owner_contract.inzi_training_root, label="canonical Inzi training root"
    )
    staging_root = _normalised_absolute_path(
        owner_contract.inzi_prefix_staging_root, label="Inzi prefix staging root"
    )
    if candidate == staging_root or staging_root in candidate.parents:
        raise R260PrestartCanaryError(
            f"{label} is in the Inzi prefix-staging root and cannot train"
        )
    if str(candidate).startswith("/mnt/Main/") or str(candidate) == "/mnt/Main":
        raise R260PrestartCanaryError(f"{label} points at an Elmo /mnt/Main path")
    try:
        candidate.relative_to(final_root)
    except ValueError as exc:
        raise R260PrestartCanaryError(
            f"{label} must be under the final Inzi training root"
        ) from exc
    return candidate


def _assert_local_non_symlink_inzi_path(
    value: object,
    *,
    owner_contract: R260OwnerContract,
    label: str,
) -> None:
    """Reject a local final-root artifact reached through any symlink segment."""

    candidate = _assert_final_inzi_path(
        value, owner_contract=owner_contract, label=label
    )
    final_root = _normalised_absolute_path(
        owner_contract.inzi_training_root, label="canonical Inzi training root"
    )
    if final_root.is_symlink() or not final_root.is_dir():
        raise R260PrestartCanaryError(
            "final Inzi training root is not a local directory"
        )
    relative = candidate.relative_to(final_root)
    cursor = final_root
    for segment in relative.parts:
        cursor = cursor / segment
        if cursor.is_symlink():
            raise R260PrestartCanaryError(f"{label} traverses a symlink")


def _daily_meta_sha256s(value: Mapping[str, Any]) -> dict[str, str]:
    """Extract the 20 embedded daily semantic identities for the Inzi index.

    The sidecar binding's ``sha256`` is the immutable file identity of
    ``meta.json``.  The streaming reader is keyed by the independently
    self-verified ``meta_sha256`` inside that file; treating the outer file
    digest as the inner semantic digest makes a valid transferred corpus
    impossible to index.
    """

    rows = _mapping(
        value.get("daily_sidecar_meta_receipts"), label="daily sidecar receipts"
    )
    if len(rows) != 20:
        raise R260PrestartCanaryError(
            "r260 Inzi index requires exactly 20 daily receipts"
        )
    result: dict[str, str] = {}
    for day, row in rows.items():
        day_text = str(day).strip()
        if not day_text or day_text in result:
            raise R260PrestartCanaryError("r260 Inzi daily receipt days are invalid")
        receipt = _mapping(row, label=f"daily sidecar receipt {day_text}")
        path = Path(str(receipt.get("path") or "")).expanduser()
        if not path.exists():
            # Portable config validation may run from a host that has only the
            # immutable binding.  In that case the independently supplied
            # streaming-index provenance still has to match this map exactly.
            result[day_text] = _require_sha(
                receipt.get("sha256"),
                label=f"daily sidecar receipt {day_text}",
            )
            continue
        path = _regular_file(path, label=f"daily sidecar meta {day_text}")
        try:
            meta = _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                label=f"daily sidecar meta {day_text}",
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R260PrestartCanaryError(
                f"daily sidecar meta {day_text} is unreadable"
            ) from exc
        result[day_text] = _require_sha(
            meta.get("meta_sha256"),
            label=f"daily sidecar semantic identity {day_text}",
        )
    return dict(sorted(result.items()))


def _normalise_daily_meta_sha256s(value: object) -> dict[str, str]:
    rows = _mapping(value, label="Inzi streaming-index daily metadata")
    if len(rows) != 20:
        raise R260PrestartCanaryError(
            "Inzi streaming index requires exactly 20 daily metadata rows"
        )
    result: dict[str, str] = {}
    for day, digest in rows.items():
        day_text = str(day).strip()
        if not day_text or day_text in result:
            raise R260PrestartCanaryError(
                "Inzi streaming-index daily metadata keys are invalid"
            )
        result[day_text] = _require_sha(
            digest, label=f"Inzi streaming-index daily metadata {day_text}"
        )
    return dict(sorted(result.items()))


def _validate_inzi_execution(
    value: object,
    *,
    owner_contract: R260OwnerContract,
    verify_files: bool,
    expected_joined_dataset: Mapping[str, Any] | None = None,
    expected_daily_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the final-root-only typed index binding in a canary config."""

    execution = dict(_mapping(value, label="Inzi canary execution"))
    required = {
        "managed_training_host",
        "training_root",
        "trainer_input",
        "inzi_joined_dataset",
        "streaming_index",
        "streaming_index_provenance",
    }
    if set(execution) != required:
        raise R260PrestartCanaryError("Inzi canary execution key inventory changed")
    root = _normalised_absolute_path(
        execution.get("training_root"), label="Inzi training root"
    )
    canonical_root = _normalised_absolute_path(
        owner_contract.inzi_training_root, label="canonical Inzi training root"
    )
    if (
        execution.get("managed_training_host") != "inzi"
        or root != canonical_root
        or execution.get("trainer_input") != R260_INZI_TRAINER_INPUT
    ):
        raise R260PrestartCanaryError(
            "r260 canary is not pinned to Inzi final-root input"
        )
    joined_dataset = _validate_file_identity(
        execution.get("inzi_joined_dataset"),
        label="final Inzi joined dataset",
        verify_file=verify_files,
        immutable=verify_files,
    )
    streaming_index = _validate_file_identity(
        execution.get("streaming_index"),
        label="Inzi streaming index",
        verify_file=verify_files,
        immutable=verify_files,
    )
    _assert_final_inzi_path(
        joined_dataset["path"],
        owner_contract=owner_contract,
        label="final Inzi joined dataset",
    )
    _assert_final_inzi_path(
        streaming_index["path"],
        owner_contract=owner_contract,
        label="Inzi streaming index",
    )
    if verify_files:
        _assert_local_non_symlink_inzi_path(
            joined_dataset["path"],
            owner_contract=owner_contract,
            label="final Inzi joined dataset",
        )
        _assert_local_non_symlink_inzi_path(
            streaming_index["path"],
            owner_contract=owner_contract,
            label="Inzi streaming index",
        )
    if (
        expected_joined_dataset is not None
        and joined_dataset
        != _validate_file_identity(
            expected_joined_dataset,
            label="Inzi dataset binding joined dataset",
            verify_file=verify_files,
            immutable=verify_files,
        )
    ):
        raise R260PrestartCanaryError(
            "canary index does not bind the completed Inzi joined dataset"
        )
    provenance = dict(
        _mapping(
            execution.get("streaming_index_provenance"),
            label="Inzi streaming-index provenance",
        )
    )
    if set(provenance) != {
        "schema",
        "source_manifest_sha256",
        "daily_meta_sha256s",
    }:
        raise R260PrestartCanaryError(
            "Inzi streaming-index provenance key inventory changed"
        )
    daily_metadata = _normalise_daily_meta_sha256s(provenance.get("daily_meta_sha256s"))
    if (
        provenance.get("schema") != R260_INZI_STREAMING_INDEX_SCHEMA
        or provenance.get("source_manifest_sha256")
        != owner_contract.source_manifest_sha256
        or (
            expected_daily_metadata is not None
            and daily_metadata != dict(sorted(expected_daily_metadata.items()))
        )
    ):
        raise R260PrestartCanaryError("Inzi streaming-index provenance drifted")
    return {
        "managed_training_host": "inzi",
        "training_root": str(canonical_root),
        "trainer_input": R260_INZI_TRAINER_INPUT,
        "inzi_joined_dataset": joined_dataset,
        "streaming_index": streaming_index,
        "streaming_index_provenance": {
            "schema": R260_INZI_STREAMING_INDEX_SCHEMA,
            "source_manifest_sha256": owner_contract.source_manifest_sha256,
            "daily_meta_sha256s": daily_metadata,
        },
    }


def _verify_runtime_streaming_index(
    index: InziStreamingIndex,
    *,
    execution: Mapping[str, Any],
) -> None:
    """Rehash and ask the live index to re-open its sealed provenance."""

    path = getattr(index, "path", None)
    assertion = getattr(index, "assert_verified", None)
    if path is None or not callable(assertion):
        raise R260PrestartCanaryError("canary requires a typed Inzi streaming index")
    expected_index = _validate_file_identity(
        execution.get("streaming_index"),
        label="sealed Inzi streaming index",
        verify_file=False,
    )
    observed_index = file_identity(path, immutable=True)
    if observed_index != expected_index:
        raise R260PrestartCanaryError(
            "runtime Inzi streaming index FileIdentity mismatch"
        )
    provenance = _mapping(
        execution.get("streaming_index_provenance"),
        label="Inzi streaming-index provenance",
    )
    try:
        assertion(
            expected_source_manifest_sha256=provenance["source_manifest_sha256"],
            daily_meta_sha256s=dict(provenance["daily_meta_sha256s"]),
        )
    except Exception as exc:  # Index-specific errors are deliberately fail-closed.
        raise R260PrestartCanaryError(
            "runtime Inzi streaming index provenance failed"
        ) from exc


def _validate_child_runtime_off(payload: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(
        _mapping(payload.get("model_config"), label="migration child model_config")
    )
    if any(cfg.get(field) is not True for field in PHYSICAL_CONFIG_FIELDS):
        raise R260PrestartCanaryError(
            "migration child lacks physical OwnDeckLedger modules"
        )
    if any(cfg.get(field) is not False for field in RUNTIME_GATE_FIELDS):
        raise R260PrestartCanaryError(
            "zero-safe migration child is already runtime-enabled"
        )
    return cfg


def _validate_bindings(
    *,
    migration_receipt: Mapping[str, Any] | Path | str,
    sidecar_binding: Mapping[str, Any] | Path | str,
    inzi_dataset_binding: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    verify_local_evidence: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use the r260 lineage validators before building any canary config."""

    try:
        migration_receipt_value = validate_r260_migration_receipt(
            migration_receipt,
            owner_contract=owner_contract,
            require_local_child=verify_local_evidence,
        )
        sidecar_value = validate_r260_sidecar_binding(
            sidecar_binding,
            owner_contract=owner_contract,
            verify_daily_receipt_files=verify_local_evidence,
        )
        inzi_value = validate_r260_inzi_dataset_binding(
            inzi_dataset_binding,
            sidecar_binding=sidecar_value,
            owner_contract=owner_contract,
            require_local_dataset=verify_local_evidence,
        )
    except R241OwnDeckSuccessorError as exc:
        raise R260PrestartCanaryError(str(exc)) from exc
    return migration_receipt_value, sidecar_value, inzi_value


def prepare_r260_prestart_canary_config(
    *,
    migration_receipt: Mapping[str, Any] | Path | str,
    sidecar_binding: Mapping[str, Any] | Path | str,
    inzi_dataset_binding: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    training_source_ids: Sequence[str],
    evaluation_source_ids: Sequence[str],
    inherited_route_prefixes: Sequence[str],
    inzi_streaming_index: Mapping[str, Any],
    inzi_streaming_index_provenance: Mapping[str, Any],
    output_path: Path | str,
    seed: int = 260,
    max_steps: int = 4,
    verify_local_evidence: bool = False,
) -> dict[str, Any]:
    """Write the create-only bounded expert-canary configuration.

    This function validates the completed Elmo side-store and its separate Inzi
    binding before it writes anything.  The configuration has zero service,
    selector, package, and submission authority.
    """

    migration_value, sidecar_value, inzi_value = _validate_bindings(
        migration_receipt=migration_receipt,
        sidecar_binding=sidecar_binding,
        inzi_dataset_binding=inzi_dataset_binding,
        owner_contract=owner_contract,
        verify_local_evidence=verify_local_evidence,
    )
    train_ids = _normalise_source_ids(training_source_ids, label="training source IDs")
    eval_ids = _normalise_source_ids(
        evaluation_source_ids, label="evaluation source IDs"
    )
    if set(train_ids) & set(eval_ids):
        raise R260PrestartCanaryError("canary training and evaluation sources overlap")
    routes = _normalise_prefixes(inherited_route_prefixes)
    seed_value, step_limit = _normalise_limits(seed=seed, max_steps=max_steps)
    if (
        migration_value.get("schema") != R260_MIGRATION_SCHEMA
        or migration_value.get("kind") != R260_MIGRATION_KIND
    ):
        raise R260PrestartCanaryError("canary requires the r260 zero-safe migration")
    inzi_execution = _validate_inzi_execution(
        {
            "managed_training_host": "inzi",
            "training_root": owner_contract.inzi_training_root,
            "trainer_input": R260_INZI_TRAINER_INPUT,
            "inzi_joined_dataset": inzi_value["inzi_joined_dataset"],
            "streaming_index": inzi_streaming_index,
            "streaming_index_provenance": inzi_streaming_index_provenance,
        },
        owner_contract=owner_contract,
        verify_files=verify_local_evidence,
        expected_joined_dataset=_mapping(
            inzi_value.get("inzi_joined_dataset"), label="Inzi joined dataset"
        ),
        expected_daily_metadata=_daily_meta_sha256s(sidecar_value),
    )
    config = _sealed_config(
        {
            "schema": R260_PRESTART_CANARY_CONFIG_SCHEMA,
            "status": "sealed_prestart_deterministic_expert_canary",
            "owner_contract_sha256": owner_contract.sha256,
            "migration_receipt_sha256": migration_value["receipt_sha256"],
            "migration_child_checkpoint": dict(migration_value["child_checkpoint"]),
            "sidecar_binding_sha256": sidecar_value["binding_sha256"],
            "inzi_dataset_binding_sha256": inzi_value["binding_sha256"],
            "inzi_execution": inzi_execution,
            "training_source_ids": list(train_ids),
            "evaluation_source_ids": list(eval_ids),
            "bounded_execution": {
                "seed": seed_value,
                "min_steps": MIN_CANARY_STEPS,
                "max_steps": step_limit,
                "deterministic": True,
            },
            "gradient_requirements": {
                "new_parameter_prefixes": list(R274_BOOTSTRAP_TRAINABLE_PREFIXES),
                "inherited_route_prefixes": list(routes),
                "finite_and_nonzero_required": True,
            },
            "coverage_requirements": {
                "fields": list(REQUIRED_COVERAGE_FIELDS),
                "both_labeled_and_masked_rows_required": True,
            },
            "calibration_requirements": {
                "fields": list(REQUIRED_CALIBRATION_FIELDS),
                "factual_selected_option_only": True,
            },
            "loss_weights": {
                "visible_tutor_completion": R260_TUTOR_LOSS_WEIGHT,
                "terminal_conversion": R260_TERMINAL_LOSS_WEIGHT,
                "tactical_sequence_outcome": R274_TACTICAL_LOSS_WEIGHT,
                "total_auxiliary": (
                    R260_TUTOR_LOSS_WEIGHT
                    + R260_TERMINAL_LOSS_WEIGHT
                    + R274_TACTICAL_LOSS_WEIGHT
                ),
            },
            "input_migration_child_runtime_gates": {
                field: False for field in RUNTIME_GATE_FIELDS
            },
            "output_runtime_gates_after_evidence": {
                field: True for field in RUNTIME_GATE_FIELDS
            },
            "invariants": {
                "public_information_only": True,
                "direct_policy_only": True,
                "mcts_rtp_or_tree_search": False,
                "hidden_deck_prize_or_opponent_private_state": False,
                "evaluation_or_kaggle_replay_training": False,
                "zero_safe_migration_child_may_launch_directly": False,
            },
            "authority": {
                "managed_service_start": False,
                "selector_change": False,
                "runtime_activation_without_external_overlay": False,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(output_path, config, label="r260 canary config")
    return config


def validate_r260_prestart_canary_config(
    value: Mapping[str, Any] | Path | str,
    *,
    owner_contract: R260OwnerContract,
    verify_file: bool = False,
) -> dict[str, Any]:
    """Strictly validate a previously sealed canary configuration."""

    if isinstance(value, (Path, str)):
        config, _ = _load_immutable_json(value, label="r260 canary config")
    else:
        config = dict(_mapping(value, label="r260 canary config"))
    required = {
        "schema",
        "status",
        "config_sha256",
        "owner_contract_sha256",
        "migration_receipt_sha256",
        "migration_child_checkpoint",
        "sidecar_binding_sha256",
        "inzi_dataset_binding_sha256",
        "inzi_execution",
        "training_source_ids",
        "evaluation_source_ids",
        "bounded_execution",
        "gradient_requirements",
        "coverage_requirements",
        "calibration_requirements",
        "loss_weights",
        "input_migration_child_runtime_gates",
        "output_runtime_gates_after_evidence",
        "invariants",
        "authority",
    }
    if set(config) != required:
        raise R260PrestartCanaryError("r260 canary config key inventory changed")
    if (
        config.get("schema") != R260_PRESTART_CANARY_CONFIG_SCHEMA
        or config.get("status") != "sealed_prestart_deterministic_expert_canary"
        or config.get("config_sha256")
        != _canonical_digest(config, field="config_sha256")
    ):
        raise R260PrestartCanaryError(
            "r260 canary config schema/status/digest is invalid"
        )
    if config.get("owner_contract_sha256") != owner_contract.sha256:
        raise R260PrestartCanaryError("r260 canary config owner contract mismatch")
    _require_sha(config.get("migration_receipt_sha256"), label="migration receipt")
    _require_sha(config.get("sidecar_binding_sha256"), label="sidecar binding")
    _require_sha(
        config.get("inzi_dataset_binding_sha256"), label="Inzi dataset binding"
    )
    _validate_file_identity(
        config.get("migration_child_checkpoint"),
        label="migration child checkpoint",
        verify_file=verify_file,
    )
    _validate_inzi_execution(
        config.get("inzi_execution"),
        owner_contract=owner_contract,
        verify_files=verify_file,
    )
    train_ids = _normalise_source_ids(
        config.get("training_source_ids") or (), label="training source IDs"
    )
    eval_ids = _normalise_source_ids(
        config.get("evaluation_source_ids") or (), label="evaluation source IDs"
    )
    if set(train_ids) & set(eval_ids):
        raise R260PrestartCanaryError("r260 config training/evaluation sources overlap")
    bounded = _mapping(config.get("bounded_execution"), label="bounded execution")
    if set(bounded) != {"seed", "min_steps", "max_steps", "deterministic"}:
        raise R260PrestartCanaryError("r260 bounded execution inventory changed")
    _normalise_limits(seed=bounded.get("seed"), max_steps=bounded.get("max_steps"))
    if (
        bounded.get("min_steps") != MIN_CANARY_STEPS
        or bounded.get("deterministic") is not True
    ):
        raise R260PrestartCanaryError("r260 canary is not bounded deterministic")
    gradients = _mapping(
        config.get("gradient_requirements"), label="gradient requirements"
    )
    if set(gradients) != {
        "new_parameter_prefixes",
        "inherited_route_prefixes",
        "finite_and_nonzero_required",
    }:
        raise R260PrestartCanaryError("r260 gradient requirement inventory changed")
    if (
        tuple(gradients.get("new_parameter_prefixes") or ())
        != R274_BOOTSTRAP_TRAINABLE_PREFIXES
    ):
        raise R260PrestartCanaryError("r260 successor gradient prefixes changed")
    _normalise_prefixes(gradients.get("inherited_route_prefixes") or ())
    if gradients.get("finite_and_nonzero_required") is not True:
        raise R260PrestartCanaryError("r260 canary permits zero/nonfinite gradients")
    coverage = _mapping(
        config.get("coverage_requirements"), label="coverage requirements"
    )
    if (
        set(coverage) != {"fields", "both_labeled_and_masked_rows_required"}
        or tuple(coverage.get("fields") or ()) != REQUIRED_COVERAGE_FIELDS
        or coverage.get("both_labeled_and_masked_rows_required") is not True
    ):
        raise R260PrestartCanaryError("r260 coverage requirements changed")
    calibration = _mapping(
        config.get("calibration_requirements"), label="calibration requirements"
    )
    if (
        set(calibration) != {"fields", "factual_selected_option_only"}
        or tuple(calibration.get("fields") or ()) != REQUIRED_CALIBRATION_FIELDS
        or calibration.get("factual_selected_option_only") is not True
    ):
        raise R260PrestartCanaryError("r260 calibration requirements changed")
    losses = _mapping(config.get("loss_weights"), label="loss weights")
    if losses != {
        "visible_tutor_completion": R260_TUTOR_LOSS_WEIGHT,
        "terminal_conversion": R260_TERMINAL_LOSS_WEIGHT,
        "tactical_sequence_outcome": R274_TACTICAL_LOSS_WEIGHT,
        "total_auxiliary": (
            R260_TUTOR_LOSS_WEIGHT
            + R260_TERMINAL_LOSS_WEIGHT
            + R274_TACTICAL_LOSS_WEIGHT
        ),
    }:
        raise R260PrestartCanaryError("r260 auxiliary loss budget changed")
    input_gates = _mapping(
        config.get("input_migration_child_runtime_gates"), label="input gates"
    )
    output_gates = _mapping(
        config.get("output_runtime_gates_after_evidence"), label="output gates"
    )
    if (
        set(input_gates) != set(RUNTIME_GATE_FIELDS)
        or set(output_gates) != set(RUNTIME_GATE_FIELDS)
        or any(input_gates.get(field) is not False for field in RUNTIME_GATE_FIELDS)
        or any(output_gates.get(field) is not True for field in RUNTIME_GATE_FIELDS)
    ):
        raise R260PrestartCanaryError("r260 runtime gate plan changed")
    invariants = _mapping(config.get("invariants"), label="invariants")
    expected_invariants = {
        "public_information_only": True,
        "direct_policy_only": True,
        "mcts_rtp_or_tree_search": False,
        "hidden_deck_prize_or_opponent_private_state": False,
        "evaluation_or_kaggle_replay_training": False,
        "zero_safe_migration_child_may_launch_directly": False,
    }
    if invariants != expected_invariants:
        raise R260PrestartCanaryError("r260 canary invariants changed")
    authority = _mapping(config.get("authority"), label="authority")
    if authority != {
        "managed_service_start": False,
        "selector_change": False,
        "runtime_activation_without_external_overlay": False,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError("r260 canary config grants operational authority")
    return config


def _coverage(value: Mapping[str, int]) -> dict[str, int]:
    row = dict(_mapping(value, label="coverage"))
    if set(row) != set(REQUIRED_COVERAGE_FIELDS):
        raise R260PrestartCanaryError("coverage inventory changed")
    result: dict[str, int] = {}
    for key in REQUIRED_COVERAGE_FIELDS:
        raw = row[key]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise R260PrestartCanaryError(f"coverage {key} is invalid")
        result[key] = int(raw)
    return result


def _calibration(value: Mapping[str, float | int]) -> dict[str, float | int]:
    row = dict(_mapping(value, label="calibration"))
    if set(row) != set(REQUIRED_CALIBRATION_FIELDS):
        raise R260PrestartCanaryError("calibration inventory changed")
    result: dict[str, float | int] = {}
    for key in REQUIRED_CALIBRATION_FIELDS:
        raw = row[key]
        if key.endswith("_count"):
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise R260PrestartCanaryError(f"calibration {key} is invalid")
            result[key] = int(raw)
        else:
            if isinstance(raw, bool):
                raise R260PrestartCanaryError(f"calibration {key} is invalid")
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise R260PrestartCanaryError(f"calibration {key} is invalid") from exc
            if not math.isfinite(numeric) or numeric < 0.0:
                raise R260PrestartCanaryError(f"calibration {key} is invalid")
            result[key] = numeric
    return result


def _merge_coverage(rows: Sequence[Mapping[str, int]]) -> dict[str, int]:
    totals = {key: 0 for key in REQUIRED_COVERAGE_FIELDS}
    for row in rows:
        for key, value in _coverage(row).items():
            totals[key] += value
    if any(totals[key] <= 0 for key in REQUIRED_COVERAGE_FIELDS):
        raise R260PrestartCanaryError("canary lacks factual mask/coverage evidence")
    return totals


def _merge_calibration(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    totals: dict[str, float | int] = {
        key: (0 if key.endswith("_count") else 0.0)
        for key in REQUIRED_CALIBRATION_FIELDS
    }
    for row in rows:
        for key, value in _calibration(row).items():
            totals[key] = totals[key] + value  # type: ignore[operator]
    if (
        int(totals["visible_tutor_brier_count"]) <= 0
        or int(totals["terminal_brier_count"]) <= 0
        or int(totals["terminal_ece_count"]) <= 0
    ):
        raise R260PrestartCanaryError("canary lacks factual calibration support")
    if (
        float(totals["visible_tutor_brier_sum"])
        > float(totals["visible_tutor_brier_count"])
        or float(totals["terminal_brier_sum"]) > float(totals["terminal_brier_count"])
        or float(totals["terminal_ece_sum"]) > float(totals["terminal_ece_count"])
    ):
        raise R260PrestartCanaryError(
            "canary calibration metric exceeds factual support"
        )
    totals["visible_tutor_brier"] = float(totals["visible_tutor_brier_sum"]) / int(
        totals["visible_tutor_brier_count"]
    )
    totals["terminal_brier"] = float(totals["terminal_brier_sum"]) / int(
        totals["terminal_brier_count"]
    )
    totals["terminal_ece"] = float(totals["terminal_ece_sum"]) / int(
        totals["terminal_ece_count"]
    )
    return totals


def _validate_aggregated_calibration(
    value: Mapping[str, Any],
) -> dict[str, float | int]:
    """Validate mergeable calibration totals plus their derived rates."""

    row = dict(_mapping(value, label="aggregated calibration"))
    expected = set(REQUIRED_CALIBRATION_FIELDS) | {
        "visible_tutor_brier",
        "terminal_brier",
        "terminal_ece",
    }
    if set(row) != expected:
        raise R260PrestartCanaryError("aggregated calibration inventory changed")
    base = {key: row[key] for key in REQUIRED_CALIBRATION_FIELDS}
    computed = _merge_calibration([base])
    for key in ("visible_tutor_brier", "terminal_brier", "terminal_ece"):
        try:
            observed = float(row[key])
        except (TypeError, ValueError) as exc:
            raise R260PrestartCanaryError(
                f"aggregated calibration {key} is invalid"
            ) from exc
        if not math.isfinite(observed) or not math.isclose(
            observed, float(computed[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise R260PrestartCanaryError(f"aggregated calibration {key} drifted")
    return computed


def _gradient_snapshot(
    model: nn.Module, prefixes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    named = tuple(model.named_parameters())
    result: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        matches = [
            (name, parameter) for name, parameter in named if name.startswith(prefix)
        ]
        if not matches:
            raise R260PrestartCanaryError(
                f"model lacks required gradient prefix {prefix}"
            )
        norms: dict[str, float] = {}
        finite = True
        for name, parameter in matches:
            gradient = parameter.grad
            if gradient is None:
                continue
            detached = gradient.detach()
            if not torch.isfinite(detached).all():
                finite = False
            norms[name] = float(detached.float().norm().cpu())
        result[prefix] = {
            "parameter_names": [name for name, _ in matches],
            "gradient_norms": norms,
            "finite": finite,
        }
    return result


def _merge_gradients(
    snapshots: Sequence[Mapping[str, Mapping[str, Any]]], prefixes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        names: set[str] = set()
        total_norms: dict[str, float] = {}
        finite = True
        for snapshot in snapshots:
            row = _mapping(snapshot.get(prefix), label=f"gradient {prefix}")
            finite = finite and row.get("finite") is True
            for name in row.get("parameter_names") or ():
                names.add(str(name))
            for name, norm in _mapping(
                row.get("gradient_norms"), label="gradient norms"
            ).items():
                numeric = float(norm)
                if not math.isfinite(numeric) or numeric < 0.0:
                    finite = False
                total_norms[str(name)] = total_norms.get(str(name), 0.0) + numeric
        if (
            not finite
            or not total_norms
            or not any(value > 0.0 for value in total_norms.values())
        ):
            raise R260PrestartCanaryError(
                f"required finite nonzero gradient is absent for {prefix}"
            )
        merged[prefix] = {
            "parameter_names": sorted(names),
            "gradient_norms": dict(sorted(total_norms.items())),
            "finite": True,
            "nonzero": True,
        }
    return merged


def _model_state_exact(model: nn.Module, state: Mapping[str, Any]) -> None:
    observed = model.state_dict()
    if set(observed) != set(state):
        raise R260PrestartCanaryError(
            "provided model does not match migration child state keys"
        )
    for name, tensor in observed.items():
        source = state[name]
        if not isinstance(source, torch.Tensor) or not torch.equal(
            tensor.detach().cpu(), source.detach().cpu()
        ):
            raise R260PrestartCanaryError(
                "provided model does not exactly load the migration child"
            )


def _assert_new_output_path(path: Path | str, *, label: str) -> Path:
    """Preflight a create-only destination before any optimizer mutation."""

    destination = Path(path).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} already exists: {destination}")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise R260PrestartCanaryError(f"{label} parent must be an existing directory")
    return destination


@contextmanager
def _deterministic_canary_execution(seed: int) -> Any:
    """Temporarily make the bounded local canary reproducible and fail closed.

    The surrounding trainer's random state and deterministic-algorithm setting
    are restored even when a factual gate fails.  A nondeterministic kernel is
    therefore a canary failure rather than silently becoming evidence.
    """

    random_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms)
        random.setstate(random_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _write_immutable_checkpoint(
    path: Path | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    destination = Path(path).expanduser()
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise R260PrestartCanaryError(
            "canary checkpoint parent must be an existing directory"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"r260 canary checkpoint already exists: {destination}")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.r260.", dir=parent
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink(missing_ok=True)
    return file_identity(destination, immutable=True)


def _load_checkpoint(
    path: Path | str, *, label: str, immutable: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _regular_file(path, label=label, immutable=immutable)
    try:
        payload = checkpoint.load_checkpoint(candidate, map_location="cpu")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise R260PrestartCanaryError(f"{label} is not a readable checkpoint") from exc
    return dict(_mapping(payload, label=label)), file_identity(
        candidate, immutable=immutable
    )


def run_bounded_deterministic_expert_canary(
    *,
    canary_config: Mapping[str, Any] | Path | str,
    migration_receipt: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step_builder: CanaryStepBuilder,
    inzi_streaming_index: InziStreamingIndex,
    migration_child_checkpoint: Path | str,
    output_checkpoint: Path | str,
    output_receipt: Path | str,
) -> R260CanaryRunResult:
    """Run the finite offline canary and publish only create-only artifacts.

    The caller supplies a model which must exactly match the zero-safe child
    checkpoint, an optimizer, and a factual loss builder.  This keeps model- and
    dataset-specific code out of the activation layer while the layer itself
    audits the difficult invariants after every backward pass.
    """

    if not isinstance(canary_config, (Path, str)):
        raise R260PrestartCanaryError(
            "canary execution requires an immutable config file"
        )
    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract, verify_file=True
    )
    try:
        migration_value = validate_r260_migration_receipt(
            migration_receipt, owner_contract=owner_contract, require_local_child=False
        )
    except R241OwnDeckSuccessorError as exc:
        raise R260PrestartCanaryError(str(exc)) from exc
    if migration_value.get("receipt_sha256") != config["migration_receipt_sha256"]:
        raise R260PrestartCanaryError("canary config/migration receipt mismatch")
    output_checkpoint_path = _assert_new_output_path(
        output_checkpoint, label="r260 canary checkpoint"
    )
    output_receipt_path = _assert_new_output_path(
        output_receipt, label="r260 canary receipt"
    )
    if output_checkpoint_path == output_receipt_path:
        raise R260PrestartCanaryError(
            "canary checkpoint and receipt destinations must differ"
        )
    child_payload, child_identity = _load_checkpoint(
        migration_child_checkpoint, label="zero-safe migration child"
    )
    expected_child = _validate_file_identity(
        migration_value.get("child_checkpoint"),
        label="migration receipt child checkpoint",
        verify_file=False,
    )
    if child_identity != expected_child or child_identity != _validate_file_identity(
        config.get("migration_child_checkpoint"),
        label="canary config child checkpoint",
        verify_file=False,
    ):
        raise R260PrestartCanaryError("canary child checkpoint identity mismatch")
    child_config = _validate_child_runtime_off(child_payload)
    child_state = _mapping(
        child_payload.get("model_state_dict"), label="migration child state"
    )
    _model_state_exact(model, child_state)
    _verify_runtime_streaming_index(
        inzi_streaming_index,
        execution=_mapping(config["inzi_execution"], label="Inzi canary execution"),
    )
    bounded = _mapping(config["bounded_execution"], label="bounded execution")
    seed = int(bounded["seed"])
    max_steps = int(bounded["max_steps"])
    routes = tuple(
        _mapping(config["gradient_requirements"], label="gradient requirements")[
            "inherited_route_prefixes"
        ]
    )
    prefixes = R274_BOOTSTRAP_TRAINABLE_PREFIXES + tuple(
        str(item) for item in routes
    )
    snapshots: list[dict[str, dict[str, Any]]] = []
    coverage_rows: list[Mapping[str, int]] = []
    calibration_rows: list[Mapping[str, float | int]] = []
    observed_sources: set[str] = set()
    training_sources = {str(item) for item in config["training_source_ids"]}
    with _deterministic_canary_execution(seed):
        model.train()
        for step_index in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            step = step_builder(model, step_index, inzi_streaming_index)
            if not isinstance(step, CanaryStep):
                raise R260PrestartCanaryError(
                    "canary step builder must return CanaryStep"
                )
            if not isinstance(step.loss, torch.Tensor) or not step.loss.requires_grad:
                raise R260PrestartCanaryError(
                    "canary loss must be a differentiable tensor"
                )
            if step.loss.numel() != 1 or not torch.isfinite(step.loss.detach()).all():
                raise R260PrestartCanaryError("canary loss must be finite scalar")
            source_ids = _normalise_source_ids(
                step.source_ids, label="canary step source IDs"
            )
            if not set(source_ids) <= training_sources:
                raise R260PrestartCanaryError("canary step used a non-training source")
            if (
                step.public_information_only is not True
                or step.direct_policy_only is not True
                or step.no_search_or_rtp is not True
                or step.no_hidden_state is not True
                or step.evaluation_or_kaggle_replay_used is not False
            ):
                raise R260PrestartCanaryError(
                    "canary step violates the public direct-policy boundary"
                )
            step.loss.backward()
            snapshots.append(_gradient_snapshot(model, prefixes))
            coverage_rows.append(step.coverage)
            calibration_rows.append(step.calibration)
            optimizer.step()
            observed_sources.update(source_ids)
    if observed_sources != training_sources:
        raise R260PrestartCanaryError(
            "canary did not consume every sealed training source"
        )
    gradients = _merge_gradients(snapshots, prefixes)
    coverage = _merge_coverage(coverage_rows)
    calibration = _merge_calibration(calibration_rows)
    output_payload = copy.deepcopy(child_payload)
    output_payload["model_state_dict"] = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    output_model_config = dict(child_config)
    output_model_config.update({field: True for field in RUNTIME_GATE_FIELDS})
    output_payload["model_config"] = output_model_config
    extra = dict(output_payload.get("extra") or {})
    if "r260_prestart_canary" in extra:
        raise R260PrestartCanaryError(
            "migration child already has r260 canary metadata"
        )
    extra["r260_prestart_canary"] = {
        "schema": R260_PRESTART_CANARY_RECEIPT_SCHEMA,
        "owner_contract_sha256": owner_contract.sha256,
        "canary_config_sha256": config["config_sha256"],
        "migration_receipt_sha256": migration_value["receipt_sha256"],
        "migration_child_checkpoint_sha256": child_identity["sha256"],
        "inzi_joined_dataset_sha256": _mapping(
            config["inzi_execution"], label="Inzi canary execution"
        )["inzi_joined_dataset"]["sha256"],
        "inzi_streaming_index_sha256": _mapping(
            config["inzi_execution"], label="Inzi canary execution"
        )["streaming_index"]["sha256"],
        "bounded_steps": max_steps,
        "runtime_gates": {field: True for field in RUNTIME_GATE_FIELDS},
        "managed_service_start_authorized": False,
        "selector_change_authorized": False,
        "package_or_submission_authorized": False,
    }
    output_payload["extra"] = extra
    provenance = dict(output_payload.get("provenance") or {})
    provenance["r260_prestart_canary"] = {
        "public_information_only": True,
        "direct_policy_only": True,
        "no_search_or_rtp": True,
        "source_ids_sha256": _sha256_bytes(canonical_json(sorted(observed_sources))),
    }
    output_payload["provenance"] = provenance
    canary_checkpoint = _write_immutable_checkpoint(
        output_checkpoint_path, output_payload
    )
    receipt = seal_receipt(
        {
            "schema": R260_PRESTART_CANARY_RECEIPT_SCHEMA,
            "status": "passed",
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "migration_receipt_sha256": migration_value["receipt_sha256"],
            "inzi_dataset_binding_sha256": config["inzi_dataset_binding_sha256"],
            "inzi_execution": dict(config["inzi_execution"]),
            "migration_child_checkpoint": child_identity,
            "canary_checkpoint": canary_checkpoint,
            "bounded_execution": {
                "seed": seed,
                "executed_steps": max_steps,
                "max_steps": max_steps,
                "deterministic": True,
            },
            "gradient_audit": gradients,
            "coverage": coverage,
            "calibration": calibration,
            "data_audit": {
                "public_information_only": True,
                "direct_policy_only": True,
                "no_search_or_rtp": True,
                "no_hidden_state": True,
                "evaluation_or_kaggle_replay_used": False,
            },
            "runtime_checkpoint_config": {field: True for field in RUNTIME_GATE_FIELDS},
            "authority": {
                "managed_service_start": False,
                "selector_change": False,
                "external_overlay_required": True,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(output_receipt_path, receipt, label="r260 canary receipt")
    return R260CanaryRunResult(
        checkpoint=canary_checkpoint,
        receipt=receipt,
        executed_steps=max_steps,
    )


def _canary_receipt(
    value: Mapping[str, Any] | Path | str,
    *,
    config: Mapping[str, Any],
    owner_contract: R260OwnerContract,
    verify_files: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    file_row: dict[str, Any] | None = None
    if isinstance(value, (Path, str)):
        receipt, file_row = _load_immutable_json(value, label="r260 canary receipt")
    else:
        receipt = dict(_mapping(value, label="r260 canary receipt"))
    required = {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "canary_config_sha256",
        "migration_receipt_sha256",
        "inzi_dataset_binding_sha256",
        "inzi_execution",
        "migration_child_checkpoint",
        "canary_checkpoint",
        "bounded_execution",
        "gradient_audit",
        "coverage",
        "calibration",
        "data_audit",
        "runtime_checkpoint_config",
        "authority",
    }
    if set(receipt) != required:
        raise R260PrestartCanaryError("r260 canary receipt key inventory changed")
    if (
        receipt.get("schema") != R260_PRESTART_CANARY_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("receipt_sha256") != receipt_digest(receipt)
        or receipt.get("owner_contract_sha256") != owner_contract.sha256
        or receipt.get("canary_config_sha256") != config["config_sha256"]
        or receipt.get("migration_receipt_sha256") != config["migration_receipt_sha256"]
        or receipt.get("inzi_dataset_binding_sha256")
        != config["inzi_dataset_binding_sha256"]
        or receipt.get("inzi_execution") != config["inzi_execution"]
        or receipt.get("migration_child_checkpoint")
        != config["migration_child_checkpoint"]
    ):
        raise R260PrestartCanaryError("r260 canary receipt identity drifted")
    bounded = _mapping(
        receipt.get("bounded_execution"), label="canary bounded execution"
    )
    config_bounded = _mapping(
        config["bounded_execution"], label="config bounded execution"
    )
    if (
        set(bounded) != {"seed", "executed_steps", "max_steps", "deterministic"}
        or bounded.get("seed") != config_bounded["seed"]
        or bounded.get("max_steps") != config_bounded["max_steps"]
        or bounded.get("executed_steps") != config_bounded["max_steps"]
        or bounded.get("deterministic") is not True
    ):
        raise R260PrestartCanaryError("r260 canary execution drifted")
    gradients = _mapping(receipt.get("gradient_audit"), label="gradient audit")
    expected_prefixes = R274_BOOTSTRAP_TRAINABLE_PREFIXES + tuple(
        _mapping(config["gradient_requirements"], label="gradient requirements")[
            "inherited_route_prefixes"
        ]
    )
    if set(gradients) != set(expected_prefixes):
        raise R260PrestartCanaryError("r260 gradient receipt prefix inventory changed")
    for prefix in expected_prefixes:
        row = _mapping(gradients[prefix], label=f"gradient audit {prefix}")
        norms = _mapping(row.get("gradient_norms"), label=f"gradient norms {prefix}")
        if (
            row.get("finite") is not True
            or row.get("nonzero") is not True
            or not norms
            or not any(
                float(value) > 0.0 and math.isfinite(float(value))
                for value in norms.values()
            )
        ):
            raise R260PrestartCanaryError(f"r260 gradient receipt fails {prefix}")
    _merge_coverage([_mapping(receipt.get("coverage"), label="coverage")])
    _validate_aggregated_calibration(
        _mapping(receipt.get("calibration"), label="calibration")
    )
    if _mapping(receipt.get("data_audit"), label="data audit") != {
        "public_information_only": True,
        "direct_policy_only": True,
        "no_search_or_rtp": True,
        "no_hidden_state": True,
        "evaluation_or_kaggle_replay_used": False,
    }:
        raise R260PrestartCanaryError("r260 canary data boundary changed")
    gates = _mapping(
        receipt.get("runtime_checkpoint_config"), label="runtime checkpoint config"
    )
    if set(gates) != set(RUNTIME_GATE_FIELDS) or any(
        gates[field] is not True for field in RUNTIME_GATE_FIELDS
    ):
        raise R260PrestartCanaryError(
            "r260 canary checkpoint does not enable all runtime gates"
        )
    checkpoint_row = _validate_file_identity(
        receipt.get("canary_checkpoint"),
        label="canary checkpoint",
        verify_file=verify_files,
        immutable=verify_files,
    )
    migration_child = _validate_file_identity(
        config["migration_child_checkpoint"],
        label="migration child checkpoint",
        verify_file=False,
    )
    if checkpoint_row["sha256"] == migration_child["sha256"]:
        raise R260PrestartCanaryError(
            "zero-safe migration child cannot be a canary checkpoint"
        )
    if verify_files:
        payload, _ = _load_checkpoint(
            checkpoint_row["path"], label="canary checkpoint", immutable=True
        )
        runtime_cfg = _mapping(
            payload.get("model_config"), label="canary checkpoint model_config"
        )
        if any(
            runtime_cfg.get(field) is not True
            for field in PHYSICAL_CONFIG_FIELDS + RUNTIME_GATE_FIELDS
        ):
            raise R260PrestartCanaryError(
                "canary checkpoint serializes disabled OwnDeck gates"
            )
        metadata = _mapping(
            _mapping(payload.get("extra"), label="canary checkpoint extra").get(
                "r260_prestart_canary"
            ),
            label="canary checkpoint metadata",
        )
        expected_metadata = {
            "schema": R260_PRESTART_CANARY_RECEIPT_SCHEMA,
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "migration_receipt_sha256": config["migration_receipt_sha256"],
            "migration_child_checkpoint_sha256": migration_child["sha256"],
            "inzi_joined_dataset_sha256": _mapping(
                config["inzi_execution"], label="Inzi canary execution"
            )["inzi_joined_dataset"]["sha256"],
            "inzi_streaming_index_sha256": _mapping(
                config["inzi_execution"], label="Inzi canary execution"
            )["streaming_index"]["sha256"],
            "bounded_steps": config_bounded["max_steps"],
            "runtime_gates": {field: True for field in RUNTIME_GATE_FIELDS},
            "managed_service_start_authorized": False,
            "selector_change_authorized": False,
            "package_or_submission_authorized": False,
        }
        if dict(metadata) != expected_metadata:
            raise R260PrestartCanaryError("canary checkpoint metadata drifted")
    authority = _mapping(receipt.get("authority"), label="canary authority")
    if authority != {
        "managed_service_start": False,
        "selector_change": False,
        "external_overlay_required": True,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError("canary receipt grants operational authority")
    return receipt, file_row


def create_r260_source_disjoint_evaluation_receipt(
    *,
    canary_config: Mapping[str, Any] | Path | str,
    canary_receipt: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    evaluation_source_ids: Sequence[str],
    coverage: Mapping[str, int],
    calibration: Mapping[str, float | int],
    output_path: Path | str,
) -> dict[str, Any]:
    """Seal a factual held-out evaluation with exact source-disjointness."""

    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract
    )
    receipt, _ = _canary_receipt(
        canary_receipt,
        config=config,
        owner_contract=owner_contract,
        verify_files=isinstance(canary_receipt, (Path, str)),
    )
    ids = _normalise_source_ids(evaluation_source_ids, label="evaluation source IDs")
    if tuple(ids) != tuple(config["evaluation_source_ids"]):
        raise R260PrestartCanaryError(
            "evaluation source inventory differs from sealed config"
        )
    if set(ids) & set(config["training_source_ids"]):
        raise R260PrestartCanaryError(
            "evaluation is not source-disjoint from canary training"
        )
    metric_coverage = _merge_coverage([coverage])
    metric_calibration = _merge_calibration([calibration])
    payload = seal_receipt(
        {
            "schema": R260_PRESTART_EVALUATION_RECEIPT_SCHEMA,
            "status": "passed_source_disjoint_factual_evaluation",
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "canary_checkpoint": dict(receipt["canary_checkpoint"]),
            "training_source_ids_sha256": _sha256_bytes(
                canonical_json(sorted(config["training_source_ids"]))
            ),
            "evaluation_source_ids": list(ids),
            "coverage": metric_coverage,
            "calibration": metric_calibration,
            "data_audit": {
                "public_information_only": True,
                "direct_policy_only": True,
                "no_search_or_rtp": True,
                "no_hidden_state": True,
                "evaluation_or_kaggle_replay_used": False,
            },
            "authority": {
                "training_data_eligible": False,
                "managed_service_start": False,
                "selector_change": False,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(
        output_path, payload, label="r260 source-disjoint evaluation receipt"
    )
    return payload


def _evaluation_receipt(
    value: Mapping[str, Any] | Path | str,
    *,
    config: Mapping[str, Any],
    canary_receipt: Mapping[str, Any],
    owner_contract: R260OwnerContract,
    verify_files: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    file_row: dict[str, Any] | None = None
    if isinstance(value, (Path, str)):
        receipt, file_row = _load_immutable_json(value, label="r260 evaluation receipt")
    else:
        receipt = dict(_mapping(value, label="r260 evaluation receipt"))
    required = {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "canary_config_sha256",
        "canary_checkpoint",
        "training_source_ids_sha256",
        "evaluation_source_ids",
        "coverage",
        "calibration",
        "data_audit",
        "authority",
    }
    if set(receipt) != required:
        raise R260PrestartCanaryError("evaluation receipt key inventory changed")
    if (
        receipt.get("schema") != R260_PRESTART_EVALUATION_RECEIPT_SCHEMA
        or receipt.get("status") != "passed_source_disjoint_factual_evaluation"
        or receipt.get("receipt_sha256") != receipt_digest(receipt)
        or receipt.get("owner_contract_sha256") != owner_contract.sha256
        or receipt.get("canary_config_sha256") != config["config_sha256"]
        or receipt.get("canary_checkpoint") != canary_receipt["canary_checkpoint"]
        or receipt.get("training_source_ids_sha256")
        != _sha256_bytes(canonical_json(sorted(config["training_source_ids"])))
    ):
        raise R260PrestartCanaryError("evaluation receipt identity drifted")
    ids = _normalise_source_ids(
        receipt.get("evaluation_source_ids") or (), label="evaluation source IDs"
    )
    if tuple(ids) != tuple(config["evaluation_source_ids"]) or set(ids) & set(
        config["training_source_ids"]
    ):
        raise R260PrestartCanaryError("evaluation source-disjointness failed")
    _merge_coverage([_mapping(receipt.get("coverage"), label="evaluation coverage")])
    _validate_aggregated_calibration(
        _mapping(receipt.get("calibration"), label="evaluation calibration")
    )
    if _mapping(receipt.get("data_audit"), label="evaluation audit") != {
        "public_information_only": True,
        "direct_policy_only": True,
        "no_search_or_rtp": True,
        "no_hidden_state": True,
        "evaluation_or_kaggle_replay_used": False,
    }:
        raise R260PrestartCanaryError("evaluation data audit changed")
    if _mapping(receipt.get("authority"), label="evaluation authority") != {
        "training_data_eligible": False,
        "managed_service_start": False,
        "selector_change": False,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError("evaluation receipt grants authority")
    return receipt, file_row


def create_r260_local_elmo_replay_parity_receipt(
    *,
    canary_config: Mapping[str, Any] | Path | str,
    canary_receipt: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    local_feature_digests: Mapping[str, str],
    elmo_feature_digests: Mapping[str, str],
    replay_feature_digests: Mapping[str, str],
    output_path: Path | str,
) -> dict[str, Any]:
    """Seal exact public feature parity across local, Elmo, and replay paths."""

    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract
    )
    receipt, _ = _canary_receipt(
        canary_receipt,
        config=config,
        owner_contract=owner_contract,
        verify_files=isinstance(canary_receipt, (Path, str)),
    )
    local = _normalise_digest_map(local_feature_digests, label="local parity rows")
    elmo = _normalise_digest_map(elmo_feature_digests, label="Elmo parity rows")
    replay = _normalise_digest_map(replay_feature_digests, label="replay parity rows")
    if local != elmo or local != replay:
        raise R260PrestartCanaryError("local/Elmo/replay feature parity mismatch")
    payload = seal_receipt(
        {
            "schema": R260_PRESTART_PARITY_RECEIPT_SCHEMA,
            "status": "passed_local_elmo_replay_parity",
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "canary_checkpoint": dict(receipt["canary_checkpoint"]),
            "feature_digests": local,
            "data_audit": {
                "public_information_only": True,
                "direct_policy_only": True,
                "no_search_or_rtp": True,
                "no_hidden_state": True,
                "evaluation_or_kaggle_replay_used": False,
            },
            "authority": {
                "training_data_eligible": False,
                "managed_service_start": False,
                "selector_change": False,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(
        output_path, payload, label="r260 local/Elmo/replay parity receipt"
    )
    return payload


def _normalise_digest_map(value: Mapping[str, str], *, label: str) -> dict[str, str]:
    mapping = dict(_mapping(value, label=label))
    if not mapping:
        raise R260PrestartCanaryError(f"{label} cannot be empty")
    result: dict[str, str] = {}
    for key, digest in mapping.items():
        name = str(key).strip()
        if not name or name in result:
            raise R260PrestartCanaryError(f"{label} has invalid record IDs")
        result[name] = _require_sha(digest, label=f"{label} {name}")
    return dict(sorted(result.items()))


def _parity_receipt(
    value: Mapping[str, Any] | Path | str,
    *,
    config: Mapping[str, Any],
    canary_receipt: Mapping[str, Any],
    owner_contract: R260OwnerContract,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    file_row: dict[str, Any] | None = None
    if isinstance(value, (Path, str)):
        receipt, file_row = _load_immutable_json(value, label="r260 parity receipt")
    else:
        receipt = dict(_mapping(value, label="r260 parity receipt"))
    required = {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "canary_config_sha256",
        "canary_checkpoint",
        "feature_digests",
        "data_audit",
        "authority",
    }
    if set(receipt) != required:
        raise R260PrestartCanaryError("parity receipt key inventory changed")
    if (
        receipt.get("schema") != R260_PRESTART_PARITY_RECEIPT_SCHEMA
        or receipt.get("status") != "passed_local_elmo_replay_parity"
        or receipt.get("receipt_sha256") != receipt_digest(receipt)
        or receipt.get("owner_contract_sha256") != owner_contract.sha256
        or receipt.get("canary_config_sha256") != config["config_sha256"]
        or receipt.get("canary_checkpoint") != canary_receipt["canary_checkpoint"]
    ):
        raise R260PrestartCanaryError("parity receipt identity drifted")
    _normalise_digest_map(
        receipt.get("feature_digests"), label="parity feature digests"
    )
    if _mapping(receipt.get("data_audit"), label="parity audit") != {
        "public_information_only": True,
        "direct_policy_only": True,
        "no_search_or_rtp": True,
        "no_hidden_state": True,
        "evaluation_or_kaggle_replay_used": False,
    }:
        raise R260PrestartCanaryError("parity data audit changed")
    if _mapping(receipt.get("authority"), label="parity authority") != {
        "training_data_eligible": False,
        "managed_service_start": False,
        "selector_change": False,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError("parity receipt grants authority")
    return receipt, file_row


def create_r260_bounded_influence_receipt(
    *,
    canary_config: Mapping[str, Any] | Path | str,
    canary_receipt: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    baseline_policy_logits: torch.Tensor,
    runtime_policy_logits: torch.Tensor,
    output_path: Path | str,
    aggregate_delta_cap: float = R260_ROUTE_DELTA_CAP,
) -> dict[str, Any]:
    """Seal a finite, nonzero policy change bounded by the typed route cap."""

    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract
    )
    receipt, _ = _canary_receipt(
        canary_receipt,
        config=config,
        owner_contract=owner_contract,
        verify_files=isinstance(canary_receipt, (Path, str)),
    )
    if (
        baseline_policy_logits.shape != runtime_policy_logits.shape
        or baseline_policy_logits.numel() <= 0
    ):
        raise R260PrestartCanaryError(
            "bounded influence logits must have the same nonempty shape"
        )
    if (
        not torch.isfinite(baseline_policy_logits).all()
        or not torch.isfinite(runtime_policy_logits).all()
    ):
        raise R260PrestartCanaryError("bounded influence logits must be finite")
    try:
        cap = float(aggregate_delta_cap)
    except (TypeError, ValueError) as exc:
        raise R260PrestartCanaryError("aggregate delta cap must be numeric") from exc
    if not math.isfinite(cap) or cap <= 0.0 or cap > R260_ROUTE_DELTA_CAP:
        raise R260PrestartCanaryError("aggregate delta cap exceeds r260 contract")
    delta = (
        runtime_policy_logits.detach().float().cpu()
        - baseline_policy_logits.detach().float().cpu()
    ).abs()
    maximum = float(delta.max())
    mean = float(delta.mean())
    if maximum <= 0.0 or maximum > cap:
        raise R260PrestartCanaryError(
            "new route influence is absent or exceeds the bound"
        )
    payload = seal_receipt(
        {
            "schema": R260_PRESTART_INFLUENCE_RECEIPT_SCHEMA,
            "status": "passed_bounded_nonzero_influence",
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "canary_checkpoint": dict(receipt["canary_checkpoint"]),
            "aggregate_delta_cap": cap,
            "max_abs_logit_delta": maximum,
            "mean_abs_logit_delta": mean,
            "tensor_shape": list(delta.shape),
            "data_audit": {
                "public_information_only": True,
                "direct_policy_only": True,
                "no_search_or_rtp": True,
                "no_hidden_state": True,
                "evaluation_or_kaggle_replay_used": False,
            },
            "authority": {
                "training_data_eligible": False,
                "managed_service_start": False,
                "selector_change": False,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(output_path, payload, label="r260 bounded-influence receipt")
    return payload


def _influence_receipt(
    value: Mapping[str, Any] | Path | str,
    *,
    config: Mapping[str, Any],
    canary_receipt: Mapping[str, Any],
    owner_contract: R260OwnerContract,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    file_row: dict[str, Any] | None = None
    if isinstance(value, (Path, str)):
        receipt, file_row = _load_immutable_json(value, label="r260 influence receipt")
    else:
        receipt = dict(_mapping(value, label="r260 influence receipt"))
    required = {
        "schema",
        "status",
        "receipt_sha256",
        "owner_contract_sha256",
        "canary_config_sha256",
        "canary_checkpoint",
        "aggregate_delta_cap",
        "max_abs_logit_delta",
        "mean_abs_logit_delta",
        "tensor_shape",
        "data_audit",
        "authority",
    }
    if set(receipt) != required:
        raise R260PrestartCanaryError("influence receipt key inventory changed")
    if (
        receipt.get("schema") != R260_PRESTART_INFLUENCE_RECEIPT_SCHEMA
        or receipt.get("status") != "passed_bounded_nonzero_influence"
        or receipt.get("receipt_sha256") != receipt_digest(receipt)
        or receipt.get("owner_contract_sha256") != owner_contract.sha256
        or receipt.get("canary_config_sha256") != config["config_sha256"]
        or receipt.get("canary_checkpoint") != canary_receipt["canary_checkpoint"]
    ):
        raise R260PrestartCanaryError("influence receipt identity drifted")
    cap = float(receipt.get("aggregate_delta_cap"))
    maximum = float(receipt.get("max_abs_logit_delta"))
    mean = float(receipt.get("mean_abs_logit_delta"))
    if (
        not math.isfinite(cap)
        or not math.isfinite(maximum)
        or not math.isfinite(mean)
        or not 0.0 < cap <= R260_ROUTE_DELTA_CAP
        or not 0.0 < maximum <= cap
        or not 0.0 <= mean <= maximum
        or not isinstance(receipt.get("tensor_shape"), list)
        or not receipt["tensor_shape"]
    ):
        raise R260PrestartCanaryError("influence receipt bounds are invalid")
    if _mapping(receipt.get("data_audit"), label="influence audit") != {
        "public_information_only": True,
        "direct_policy_only": True,
        "no_search_or_rtp": True,
        "no_hidden_state": True,
        "evaluation_or_kaggle_replay_used": False,
    }:
        raise R260PrestartCanaryError("influence data audit changed")
    if _mapping(receipt.get("authority"), label="influence authority") != {
        "training_data_eligible": False,
        "managed_service_start": False,
        "selector_change": False,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError("influence receipt grants authority")
    return receipt, file_row


def create_r260_runtime_activation_config(
    *,
    canary_config: Mapping[str, Any] | Path | str,
    canary_receipt: Path | str,
    evaluation_receipt: Path | str,
    parity_receipt: Path | str,
    influence_receipt: Path | str,
    owner_contract: R260OwnerContract,
    output_path: Path | str,
) -> dict[str, Any]:
    """Bind all evidence into immutable runtime flags, still overlay-only."""

    if not isinstance(canary_config, (Path, str)):
        raise R260PrestartCanaryError(
            "runtime activation requires an immutable canary config file"
        )
    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract
    )
    canary, canary_file = _canary_receipt(
        canary_receipt, config=config, owner_contract=owner_contract, verify_files=True
    )
    assert canary_file is not None
    evaluation, evaluation_file = _evaluation_receipt(
        evaluation_receipt,
        config=config,
        canary_receipt=canary,
        owner_contract=owner_contract,
        verify_files=True,
    )
    parity, parity_file = _parity_receipt(
        parity_receipt,
        config=config,
        canary_receipt=canary,
        owner_contract=owner_contract,
    )
    influence, influence_file = _influence_receipt(
        influence_receipt,
        config=config,
        canary_receipt=canary,
        owner_contract=owner_contract,
    )
    del evaluation, parity, influence
    assert (
        evaluation_file is not None
        and parity_file is not None
        and influence_file is not None
    )
    payload = _sealed_config(
        {
            "schema": R260_RUNTIME_ACTIVATION_CONFIG_SCHEMA,
            "status": "runtime_gates_ready_external_overlay_required",
            "owner_contract_sha256": owner_contract.sha256,
            "canary_config_sha256": config["config_sha256"],
            "migration_receipt_sha256": config["migration_receipt_sha256"],
            "canary_checkpoint": dict(canary["canary_checkpoint"]),
            "evidence_receipts": {
                "finite_gradient": canary_file,
                "source_disjoint_evaluation": evaluation_file,
                "local_remote_parity": parity_file,
                "bounded_influence": influence_file,
            },
            "runtime_gates": {field: True for field in RUNTIME_GATE_FIELDS},
            "external_activation_overlay_required": True,
            "authority": {
                "managed_service_start": False,
                "selector_change": False,
                "package_creation": False,
                "submission": False,
            },
        }
    )
    _write_immutable_json(output_path, payload, label="r260 runtime activation config")
    return payload


def validate_r260_runtime_activation_config(
    value: Mapping[str, Any] | Path | str,
    *,
    canary_config: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Fail closed unless immutable runtime configuration proves all gates."""

    config = validate_r260_prestart_canary_config(
        canary_config, owner_contract=owner_contract
    )
    if isinstance(value, (Path, str)):
        activation, _ = _load_immutable_json(
            value, label="r260 runtime activation config"
        )
    else:
        activation = dict(_mapping(value, label="r260 runtime activation config"))
    required = {
        "schema",
        "status",
        "config_sha256",
        "owner_contract_sha256",
        "canary_config_sha256",
        "migration_receipt_sha256",
        "canary_checkpoint",
        "evidence_receipts",
        "runtime_gates",
        "external_activation_overlay_required",
        "authority",
    }
    if set(activation) != required:
        raise R260PrestartCanaryError("runtime activation config key inventory changed")
    if (
        activation.get("schema") != R260_RUNTIME_ACTIVATION_CONFIG_SCHEMA
        or activation.get("status") != "runtime_gates_ready_external_overlay_required"
        or activation.get("config_sha256")
        != _canonical_digest(activation, field="config_sha256")
        or activation.get("owner_contract_sha256") != owner_contract.sha256
        or activation.get("canary_config_sha256") != config["config_sha256"]
        or activation.get("migration_receipt_sha256")
        != config["migration_receipt_sha256"]
    ):
        raise R260PrestartCanaryError("runtime activation config identity drifted")
    runtime_gates = _mapping(activation.get("runtime_gates"), label="runtime gates")
    if set(runtime_gates) != set(RUNTIME_GATE_FIELDS) or any(
        runtime_gates.get(field) is not True for field in RUNTIME_GATE_FIELDS
    ):
        raise R260PrestartCanaryError(
            "runtime activation config has disabled OwnDeck gates"
        )
    if activation.get("external_activation_overlay_required") is not True:
        raise R260PrestartCanaryError(
            "runtime activation config bypasses external overlay"
        )
    authority = _mapping(
        activation.get("authority"), label="runtime activation authority"
    )
    if authority != {
        "managed_service_start": False,
        "selector_change": False,
        "package_creation": False,
        "submission": False,
    }:
        raise R260PrestartCanaryError(
            "runtime activation config grants service authority"
        )
    evidence = _mapping(
        activation.get("evidence_receipts"), label="activation evidence"
    )
    if set(evidence) != set(ACTIVATION_EVIDENCE_NAMES):
        raise R260PrestartCanaryError("runtime activation evidence inventory changed")
    for name in ACTIVATION_EVIDENCE_NAMES:
        _validate_file_identity(
            evidence[name],
            label=f"activation evidence {name}",
            verify_file=verify_files,
            immutable=verify_files,
        )
    # Require a structurally valid *distinct* trained child without trusting a
    # caller-supplied string.  Its full identity is additionally checked in the
    # canary receipt at producer time.
    candidate = _validate_file_identity(
        activation.get("canary_checkpoint"),
        label="runtime canary checkpoint",
        verify_file=verify_files,
        immutable=verify_files,
    )
    migration_child = _validate_file_identity(
        config["migration_child_checkpoint"],
        label="migration child checkpoint",
        verify_file=False,
    )
    if candidate["sha256"] == migration_child["sha256"]:
        raise R260PrestartCanaryError(
            "zero-safe migration child cannot be runtime activated"
        )
    if verify_files:
        payload, _ = _load_checkpoint(
            candidate["path"], label="runtime canary checkpoint", immutable=True
        )
        cfg = _mapping(
            payload.get("model_config"), label="runtime canary checkpoint config"
        )
        if any(
            cfg.get(field) is not True
            for field in PHYSICAL_CONFIG_FIELDS + RUNTIME_GATE_FIELDS
        ):
            raise R260PrestartCanaryError(
                "runtime checkpoint is missing enabled OwnDeck gates"
            )
        canary, canary_file = _canary_receipt(
            evidence["finite_gradient"]["path"],
            config=config,
            owner_contract=owner_contract,
            verify_files=True,
        )
        if (
            canary_file != evidence["finite_gradient"]
            or canary["canary_checkpoint"] != candidate
        ):
            raise R260PrestartCanaryError(
                "runtime activation finite-gradient evidence drifted"
            )
        evaluation, evaluation_file = _evaluation_receipt(
            evidence["source_disjoint_evaluation"]["path"],
            config=config,
            canary_receipt=canary,
            owner_contract=owner_contract,
            verify_files=True,
        )
        parity, parity_file = _parity_receipt(
            evidence["local_remote_parity"]["path"],
            config=config,
            canary_receipt=canary,
            owner_contract=owner_contract,
        )
        influence, influence_file = _influence_receipt(
            evidence["bounded_influence"]["path"],
            config=config,
            canary_receipt=canary,
            owner_contract=owner_contract,
        )
        del evaluation, parity, influence
        if (
            evaluation_file != evidence["source_disjoint_evaluation"]
            or parity_file != evidence["local_remote_parity"]
            or influence_file != evidence["bounded_influence"]
        ):
            raise R260PrestartCanaryError(
                "runtime activation evidence FileIdentity drifted"
            )
    return activation


def create_r260_canary_activation_receipt(
    *,
    runtime_activation_config: Mapping[str, Any] | Path | str,
    canary_config: Mapping[str, Any] | Path | str,
    owner_contract: R260OwnerContract,
    output_path: Path | str,
) -> dict[str, Any]:
    """Produce the compact v1 receipt consumed by the r241 launch boundary.

    The r241 successor validator intentionally expects this exact eight-field
    top-level shape.  Detailed coverage and calibration live in the immutable
    finite-gradient canary receipt referenced here.
    """

    if not isinstance(runtime_activation_config, (Path, str)) or not isinstance(
        canary_config, (Path, str)
    ):
        raise R260PrestartCanaryError(
            "compact activation requires immutable config files"
        )
    activation = validate_r260_runtime_activation_config(
        runtime_activation_config,
        canary_config=canary_config,
        owner_contract=owner_contract,
        verify_files=True,
    )
    payload = seal_receipt(
        {
            "schema": R260_CANARY_ACTIVATION_SCHEMA,
            "status": "passed",
            "owner_contract_sha256": owner_contract.sha256,
            "migration_receipt_sha256": activation["migration_receipt_sha256"],
            "canary_checkpoint": dict(activation["canary_checkpoint"]),
            "evidence_receipts": dict(activation["evidence_receipts"]),
            "runtime_gates": dict(activation["runtime_gates"]),
        }
    )
    _write_immutable_json(output_path, payload, label="r260 canary activation receipt")
    return payload


__all__ = [
    "ACTIVATION_EVIDENCE_NAMES",
    "MAX_CANARY_STEPS",
    "MIN_CANARY_STEPS",
    "R260_INZI_STREAMING_INDEX_SCHEMA",
    "R260_INZI_TRAINER_INPUT",
    "R260_PRESTART_CANARY_CONFIG_SCHEMA",
    "R260_PRESTART_CANARY_RECEIPT_SCHEMA",
    "R260_PRESTART_EVALUATION_RECEIPT_SCHEMA",
    "R260_PRESTART_INFLUENCE_RECEIPT_SCHEMA",
    "R260_PRESTART_PARITY_RECEIPT_SCHEMA",
    "R260_RUNTIME_ACTIVATION_CONFIG_SCHEMA",
    "RUNTIME_GATE_FIELDS",
    "CanaryStep",
    "InziStreamingIndex",
    "R260CanaryRunResult",
    "R260PrestartCanaryError",
    "create_r260_bounded_influence_receipt",
    "create_r260_canary_activation_receipt",
    "create_r260_local_elmo_replay_parity_receipt",
    "create_r260_runtime_activation_config",
    "create_r260_source_disjoint_evaluation_receipt",
    "file_identity",
    "prepare_r260_prestart_canary_config",
    "run_bounded_deterministic_expert_canary",
    "validate_r260_prestart_canary_config",
    "validate_r260_runtime_activation_config",
]
