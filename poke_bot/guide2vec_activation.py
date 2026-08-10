"""Fail-closed future activation boundary for generic Guide2Vec training.

Revision 226 has no semantic causal-partition proof that binds actual stage
rows to the exact train/validation/heldout chunk sets.  Consequently this
module cannot issue a training grant yet: it first canonically reloads a
manifest with ``require_ready=True`` and then raises before it reads a receipt.

The frozen grant shape and chunk-set helpers are retained only as the typed
future interface for the trainer.  They are not a current authority mechanism,
do not write a receipt, start a service, load chunks, apply gradients, or
publish/attach a candidate.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from poke_bot.guide2vec_contracts import (
    Guide2VecContractError,
    canonical_json_sha256,
    load_and_validate_training_manifest,
)

ACTIVATION_RECEIPT_SCHEMA: Final = "poke_bot.guide2vec_training_activation_receipt/v1"
ACTIVATION_RECEIPT_STATUS: Final = "future_owner_authorized_single_offline_action"
JOINED_CHUNK_REF_SCHEMA: Final = "poke_bot.guide2vec_joined_chunk_ref/v1"
PARTITIONS: Final[tuple[str, ...]] = ("train", "validation", "heldout")
SEALED_CHUNK_SET_SCHEMA: Final = "poke_bot.guide2vec_sealed_chunk_set/v1"
SINGLE_ACTION_KIND: Final = "offline_guide2vec_head_train_gradient_candidate"

_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GRANT_CONSTRUCTION_TOKEN = object()
_ISSUED_GRANT_DIGESTS: dict[int, str] = {}
_CONSUMED_GRANT_DIGESTS: set[str] = set()
_GRANT_LOCK = threading.Lock()


class Guide2VecActivationError(ValueError):
    """Raised when Guide2Vec has no canonically provable activation boundary."""


@dataclass(frozen=True, slots=True)
class VerifiedContentRef:
    """Future receipt content reference shape; no r226 receipt is accepted."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SealedChunkSet:
    """Ordered immutable latent/label pair identities for one partition."""

    partition: str
    records: tuple[Mapping[str, str], ...]
    sha256: str


@dataclass(frozen=True, slots=True, init=False)
class Guide2VecTrainingGrant:
    """Future-only one-action grant; no instance can be issued under r226.

    Python private names are not an authorization boundary.  The trainer must
    obtain any future instance by internally calling
    :func:`validate_activation_receipt`; this class additionally rejects an
    instance not registered by that function.
    """

    receipt_path: Path
    receipt_sha256: str
    owner_contract_revision: int
    owner_contract: VerifiedContentRef
    training_manifest_sha256: str
    guide_label_overlay_sha256: str
    guide_contract: VerifiedContentRef
    compatibility: Mapping[str, str]
    causal_split_manifest_sha256: str
    causal_split_receipt: VerifiedContentRef
    source_snapshot: VerifiedContentRef
    host_noninterference_capability_receipt: VerifiedContentRef
    output_root: Path
    output_root_identity_sha256: str
    partition_chunk_sets: Mapping[str, SealedChunkSet]

    def __init__(
        self,
        *,
        _token: object,
        receipt_path: Path,
        receipt_sha256: str,
        owner_contract_revision: int,
        owner_contract: VerifiedContentRef,
        training_manifest_sha256: str,
        guide_label_overlay_sha256: str,
        guide_contract: VerifiedContentRef,
        compatibility: Mapping[str, str],
        causal_split_manifest_sha256: str,
        causal_split_receipt: VerifiedContentRef,
        source_snapshot: VerifiedContentRef,
        host_noninterference_capability_receipt: VerifiedContentRef,
        output_root: Path,
        output_root_identity_sha256: str,
        partition_chunk_sets: Mapping[str, SealedChunkSet],
    ) -> None:
        if _token is not _GRANT_CONSTRUCTION_TOKEN:
            raise TypeError(
                "Guide2VecTrainingGrant is internal; use validate_activation_receipt()"
            )
        object.__setattr__(self, "receipt_path", receipt_path)
        object.__setattr__(self, "receipt_sha256", receipt_sha256)
        object.__setattr__(self, "owner_contract_revision", owner_contract_revision)
        object.__setattr__(self, "owner_contract", owner_contract)
        object.__setattr__(self, "training_manifest_sha256", training_manifest_sha256)
        object.__setattr__(self, "guide_label_overlay_sha256", guide_label_overlay_sha256)
        object.__setattr__(self, "guide_contract", guide_contract)
        object.__setattr__(self, "compatibility", MappingProxyType(dict(compatibility)))
        object.__setattr__(self, "causal_split_manifest_sha256", causal_split_manifest_sha256)
        object.__setattr__(self, "causal_split_receipt", causal_split_receipt)
        object.__setattr__(self, "source_snapshot", source_snapshot)
        object.__setattr__(
            self,
            "host_noninterference_capability_receipt",
            host_noninterference_capability_receipt,
        )
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(
            self,
            "output_root_identity_sha256",
            output_root_identity_sha256,
        )
        object.__setattr__(
            self,
            "partition_chunk_sets",
            MappingProxyType(dict(partition_chunk_sets)),
        )

    def validate_for(
        self,
        *,
        training_manifest_sha256: str,
        guide_manifest_sha256: str,
        base_identity_sha256: str,
        partition_chunks: Mapping[str, Sequence[Mapping[str, object]]],
        output_dir: str | Path,
    ) -> None:
        """Future trainer input comparison; unreachable without a valid issuer."""

        self._require_issued()
        if _require_sha256(
            training_manifest_sha256, label="trainer training_manifest_sha256"
        ) != self.training_manifest_sha256:
            raise Guide2VecActivationError("trainer training manifest does not match grant")
        if _require_sha256(
            guide_manifest_sha256, label="trainer guide_manifest_sha256"
        ) != self.guide_label_overlay_sha256:
            raise Guide2VecActivationError("trainer guide overlay does not match grant")
        if _require_sha256(
            base_identity_sha256, label="trainer base_identity_sha256"
        ) != self.compatibility.get("frozen_base_identity_sha256"):
            raise Guide2VecActivationError("trainer frozen base identity does not match grant")
        actual_sets = _normalize_trainer_partition_chunks(partition_chunks)
        for partition in PARTITIONS:
            expected = self.partition_chunk_sets.get(partition)
            if expected is None:
                raise Guide2VecActivationError(f"grant lacks {partition} chunk set")
            actual = actual_sets[partition]
            if actual.records != expected.records or actual.sha256 != expected.sha256:
                raise Guide2VecActivationError(
                    f"trainer {partition} chunk set does not match grant"
                )
        if _resolve_output_root(output_dir) != self.output_root:
            raise Guide2VecActivationError("trainer output_dir does not match granted output root")

    def consume(self) -> None:
        """Future in-process one-use claim; durable consumption remains required."""

        self._require_issued()
        with _GRANT_LOCK:
            if self.receipt_sha256 in _CONSUMED_GRANT_DIGESTS:
                raise Guide2VecActivationError("Guide2Vec training grant was already consumed")
            _CONSUMED_GRANT_DIGESTS.add(self.receipt_sha256)

    def _require_issued(self) -> None:
        with _GRANT_LOCK:
            if _ISSUED_GRANT_DIGESTS.get(id(self)) != self.receipt_sha256:
                raise Guide2VecActivationError(
                    "Guide2Vec training grant was not issued by canonical receipt validation"
                )


def joined_chunk_pair_sha256(*, latent_sha256: str, label_sha256: str) -> str:
    """Return the content identity shared with ``JoinedChunkRef``."""

    latent = _require_sha256(latent_sha256, label="latent_sha256")
    label = _require_sha256(label_sha256, label="label_sha256")
    return canonical_json_sha256(
        {
            "schema": JOINED_CHUNK_REF_SCHEMA,
            "latent_sha256": latent,
            "label_sha256": label,
        }
    )


def sealed_chunk_set_sha256(
    *, partition: str, records: Sequence[Mapping[str, object]]
) -> str:
    """Return a canonical identity for exact ordered chunk-pair records."""

    _require_partition(partition)
    normalized = _normalize_chunk_records(records, label=f"{partition} chunks")
    return canonical_json_sha256(
        {
            "schema": SEALED_CHUNK_SET_SCHEMA,
            "partition": partition,
            "chunks": [dict(record) for record in normalized],
        }
    )


def validate_activation_receipt(
    *,
    training_manifest_path: str | Path,
    receipt_path: str | Path,
) -> Guide2VecTrainingGrant:
    """Fail closed until a future proof binds split rows to sealed chunks.

    The canonical loader is deliberately invoked before receipt I/O, so neither
    a forged ``ResolvedTrainingManifest`` nor a receipt-shaped JSON object can
    bypass r226's absent semantic split proof.
    """

    try:
        load_and_validate_training_manifest(training_manifest_path, require_ready=True)
    except Guide2VecContractError as exc:
        raise Guide2VecActivationError(
            "Guide2Vec activation requires a canonically revalidated semantic-ready "
            "training manifest"
        ) from exc
    raise Guide2VecActivationError(
        "Guide2Vec activation receipt was not opened and no grant was issued: "
        "future causal partition proof-to-chunk-set binding is not implemented "
        f"for manifest {Path(training_manifest_path)} (receipt {Path(receipt_path)})"
    )


def _normalize_trainer_partition_chunks(
    value: Mapping[str, Sequence[Mapping[str, object]]],
) -> Mapping[str, SealedChunkSet]:
    _strict_mapping(value, fields=PARTITIONS, label="trainer partition_chunks")
    return MappingProxyType(
        {
            partition: SealedChunkSet(
                partition=partition,
                records=(
                    records := _normalize_chunk_records(
                        value[partition], label=f"trainer {partition} chunks"
                    )
                ),
                sha256=sealed_chunk_set_sha256(partition=partition, records=records),
            )
            for partition in PARTITIONS
        }
    )


def _normalize_chunk_records(
    records: Sequence[Mapping[str, object]], *, label: str
) -> tuple[Mapping[str, str], ...]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise Guide2VecActivationError(f"{label} must be a sequence of chunk records")
    result: list[Mapping[str, str]] = []
    seen_pairs: set[str] = set()
    for index, record in enumerate(records):
        payload = _strict_mapping(
            record,
            fields=("latent_sha256", "label_sha256", "pair_sha256"),
            label=f"{label}[{index}]",
        )
        latent = _require_sha256(payload["latent_sha256"], label=f"{label}[{index}].latent")
        label_digest = _require_sha256(
            payload["label_sha256"], label=f"{label}[{index}].label"
        )
        pair = _require_sha256(payload["pair_sha256"], label=f"{label}[{index}].pair")
        if pair != joined_chunk_pair_sha256(
            latent_sha256=latent,
            label_sha256=label_digest,
        ):
            raise Guide2VecActivationError(f"{label}[{index}].pair_sha256 mismatch")
        if pair in seen_pairs:
            raise Guide2VecActivationError(f"{label} contains a duplicate pair")
        seen_pairs.add(pair)
        result.append(
            MappingProxyType(
                {
                    "latent_sha256": latent,
                    "label_sha256": label_digest,
                    "pair_sha256": pair,
                }
            )
        )
    if not result:
        raise Guide2VecActivationError(f"{label} cannot be empty")
    return tuple(result)


def _resolve_output_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise Guide2VecActivationError("Guide2Vec output root cannot be a filesystem root")
    return resolved


def _strict_mapping(
    value: object, *, fields: Sequence[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Guide2VecActivationError(f"{label} must be an object")
    expected = set(fields)
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing or unknown:
        raise Guide2VecActivationError(
            f"{label} fields changed: missing={missing} unknown={unknown}"
        )
    return value


def _require_partition(value: object) -> str:
    if not isinstance(value, str) or value not in PARTITIONS:
        raise Guide2VecActivationError(f"partition must be one of {list(PARTITIONS)!r}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Guide2VecActivationError(f"{label} must be a canonical lowercase sha256 digest")
    return value


__all__ = [
    "ACTIVATION_RECEIPT_SCHEMA",
    "ACTIVATION_RECEIPT_STATUS",
    "JOINED_CHUNK_REF_SCHEMA",
    "PARTITIONS",
    "SEALED_CHUNK_SET_SCHEMA",
    "SINGLE_ACTION_KIND",
    "Guide2VecActivationError",
    "Guide2VecTrainingGrant",
    "SealedChunkSet",
    "VerifiedContentRef",
    "joined_chunk_pair_sha256",
    "sealed_chunk_set_sha256",
    "validate_activation_receipt",
]
