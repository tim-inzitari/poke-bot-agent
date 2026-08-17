"""Strict, guide-agnostic contracts for the dormant Guide2Vec pipeline.

The r226 pipeline deliberately separates guide-independent frozen latents from
guide-versioned labels.  This module is the small, stdlib-only boundary that
validates the JSON manifests joining those artifacts.  It does *not* materialize
latents, load a model, train a head, or grant runtime/training authority.

Content references are byte-addressed (their ``sha256`` is the digest of the
referenced file's raw bytes).  The canonical JSON helpers are for semantic
identities inside manifests and always use canonical UTF-8 JSON followed by a
single newline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

STAGE_INDEX_SCHEMA: Final = "poke_bot.guide2vec_stage_index/v1"
FROZEN_LATENT_STORE_SCHEMA: Final = "poke_bot.guide2vec_frozen_latent_store/v1"
GUIDE_LABEL_OVERLAY_SCHEMA: Final = "poke_bot.guide2vec_label_overlay/v1"
CAUSAL_SPLIT_SCHEMA: Final = "poke_bot.guide2vec_causal_split/v1"
CAUSAL_PARTITION_PAYLOAD_SCHEMA: Final = (
    "poke_bot.guide2vec_causal_partition_payload/v1"
)
TRAINING_MANIFEST_SCHEMA: Final = "poke_bot.guide2vec_training_manifest/v1"

# This order is part of the r226 ABI.  Do not derive it from a guide or accept
# an equivalent reordering: the tuple is the stable stage-to-label join key.
STAGE_JOIN_KEY_FIELDS: Final[tuple[str, ...]] = (
    "source_day",
    "episode_id",
    "seat",
    "decision_index",
    "factorized_stage_index",
    "policy_visible_observation_sha256",
    "legal_options_sha256",
)

FROZEN_LATENT_TENSORS: Final[tuple[str, ...]] = (
    "state_vec",
    "option_hidden",
    "base_logits",
    "option_offsets",
)
GUIDE_LABEL_FIELDS: Final[tuple[str, ...]] = (
    "guide_target_index",
    "guide_confidence",
)

# Chunks store the two identity digests as raw bytes, not hex strings.  A
# materializer can make a [rows, 32] uint8 tensor by stacking
# ``digest_tensor_bytes(...)`` for each row.
DIGEST_TENSOR_LAYOUT_FIELDS: Final[tuple[str, ...]] = (
    "dtype",
    "shape",
    "encoding",
    "stage_key_digest",
    "legal_option_digest",
)
DIGEST_TENSOR_LAYOUT: Final[Mapping[str, object]] = MappingProxyType(
    {
        "dtype": "uint8",
        "shape": ("rows", 32),
        "encoding": "raw_sha256_digest_bytes",
        "stage_key_digest": "stage_key_sha256",
        "legal_option_digest": "legal_options_sha256",
    }
)

# These fields bind an overlay/split/head plan to exactly one reusable frozen
# latent representation.  Their values are opaque content identities here.
COMPATIBILITY_FIELDS: Final[tuple[str, ...]] = (
    "frozen_base_identity_sha256",
    "latent_abi_sha256",
    "adapter_runtime_context_sha256",
    "extractor_code_sha256",
    "dtype_policy_sha256",
    "stage_index_sha256",
    "legal_option_key_sha256",
)

TRAINING_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "stage_index_manifest",
    "frozen_latent_store_manifest",
    "guide_label_overlay_manifest",
    "causal_split_manifest",
)
TRAINING_INPUT_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "stage_index_sha256",
    "frozen_latent_store_sha256",
    "guide_label_overlay_sha256",
    "causal_split_sha256",
)
_INPUT_IDENTITY_BY_REF: Final[Mapping[str, str]] = MappingProxyType(
    {
        "stage_index_manifest": "stage_index_sha256",
        "frozen_latent_store_manifest": "frozen_latent_store_sha256",
        "guide_label_overlay_manifest": "guide_label_overlay_sha256",
        "causal_split_manifest": "causal_split_sha256",
    }
)

# Artifact manifests must never themselves start a service, train, publish, or
# attach a candidate.  A future owner activation receipt is intentionally a
# separate contract and is not accepted as authority by this module.
INERT_AUTHORITY_FIELDS: Final[tuple[str, ...]] = (
    "training_service_start_authorized",
    "gradient_updates_authorized",
    "candidate_publication_authorized",
    "runtime_attachment_authorized",
    "bo1000_authorized",
    "mcts_change_authorized",
    "rtp_authorized",
    "selector_change_authorized",
    "serving_authorized",
    "promotion_authorized",
    "kaggle_api_or_submission_authorized",
    "evaluation_games_training_eligible",
)

_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")


class Guide2VecContractError(ValueError):
    """Raised when a Guide2Vec artifact manifest is not fail-closed valid."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the canonical JSON bytes used for semantic manifest identities.

    File content references intentionally use :func:`sha256_file` instead:
    they bind the exact raw file bytes, including any non-canonical formatting.
    """

    try:
        normalized = _json_compatible(value, label="canonical JSON value")
        return (
            json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Guide2VecContractError("value is not canonical JSON") from exc


def canonical_json_sha256(value: object) -> str:
    """Return ``sha256:…`` for :func:`canonical_json_bytes` of ``value``."""

    return sha256_bytes(canonical_json_bytes(value))


def stage_key_sha256(mapping: Mapping[str, object]) -> str:
    """Return the canonical r226 stage-key digest for exactly seven fields.

    ``mapping`` must be the key projection itself, rather than a larger row:
    rejecting extra fields prevents hidden/non-causal data from silently
    changing a join identity.  The two fingerprint fields are canonical
    SHA-256 identities and the remaining values must be canonical JSON.
    """

    payload = _strict_mapping(mapping, fields=STAGE_JOIN_KEY_FIELDS, label="stage key")
    _require_nonempty_string(payload["source_day"], label="stage key source_day")
    episode_id = payload["episode_id"]
    if not isinstance(episode_id, str) or not episode_id:
        raise Guide2VecContractError("stage key episode_id must be a non-empty string")
    _require_nonnegative_int(payload["seat"], label="stage key seat")
    _require_nonnegative_int(payload["decision_index"], label="stage key decision_index")
    _require_nonnegative_int(
        payload["factorized_stage_index"], label="stage key factorized_stage_index"
    )
    _require_sha256(
        payload["policy_visible_observation_sha256"],
        label="stage key policy_visible_observation_sha256",
    )
    _require_sha256(payload["legal_options_sha256"], label="stage key legal_options_sha256")
    return canonical_json_sha256(
        {field: _json_compatible(payload[field], label=f"stage key {field}") for field in STAGE_JOIN_KEY_FIELDS}
    )


def digest_tensor_bytes(value: str) -> bytes:
    """Convert one canonical digest to its exact 32 raw bytes for uint8 chunks."""

    digest = _require_sha256(value, label="digest tensor value")
    raw = bytes.fromhex(digest.removeprefix("sha256:"))
    if len(raw) != 32:  # Defensive: _require_sha256 already establishes this.
        raise Guide2VecContractError("digest tensor value is not 32 bytes")
    return raw


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return the canonical lowercase SHA-256 content digest for bytes."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise Guide2VecContractError("sha256_bytes requires bytes-like input")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the raw-byte SHA-256 digest of a regular content file."""

    candidate = Path(path)
    if not candidate.is_file():
        raise Guide2VecContractError(f"content reference is not a file: {candidate}")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Guide2VecContractError(
            f"could not read content reference: {candidate}"
        ) from exc
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ContentRef:
    """A raw-byte content reference resolved relative to its parent manifest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip() or "\x00" in self.path:
            raise Guide2VecContractError("content reference path must be a non-empty string")
        _require_sha256(self.sha256, label="content reference sha256")

    @classmethod
    def from_mapping(cls, value: object, *, label: str = "content reference") -> ContentRef:
        payload = _strict_mapping(value, fields=("path", "sha256"), label=label)
        path = payload["path"]
        digest = payload["sha256"]
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise Guide2VecContractError(f"{label}.path must be a non-empty string")
        return cls(path=path, sha256=_require_sha256(digest, label=f"{label}.sha256"))

    def resolve(self, parent_directory: str | Path) -> Path:
        """Resolve a relative reference against the manifest that contains it."""

        candidate = Path(self.path)
        return candidate if candidate.is_absolute() else Path(parent_directory) / candidate

    def as_dict(self) -> dict[str, str]:
        """Return the strict two-field JSON representation."""

        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Separate byte sealing from split proof and future training authority."""

    status: str
    manifest_sha256: str
    declared_inputs: tuple[str, ...]
    resolved_inputs: tuple[str, ...]
    unresolved_inputs: tuple[str, ...]
    content_refs_sealed: bool
    split_semantics_proven: bool
    split_proof_required_schema: str
    blocking_reasons: tuple[str, ...]
    training_authorized: bool
    separate_activation_receipt_required: bool

    @property
    def content_sealed(self) -> bool:
        """Alias for byte-level sealing of every declared content reference."""

        return self.content_refs_sealed

    @property
    def data_ready(self) -> bool:
        """Whether sealed contents also have a validated semantic split proof.

        The dormant r226 schema does not yet define a streamable causal split
        proof, so this is deliberately false even for a content-sealed ready
        manifest.  Raw hashes and boolean guards alone cannot prove whole-day,
        whole-episode, or stage-level partition disjointness.
        """

        return self.content_refs_sealed and self.split_semantics_proven

    @property
    def ready(self) -> bool:
        """Alias for true data readiness, never merely content sealing."""

        return self.data_ready


@dataclass(frozen=True, slots=True)
class ResolvedContentRef:
    """One raw-byte verified content reference reached from a ready manifest.

    ``path`` is an observation recorded at validation time, never a reusable
    authorization to consume bytes later.  A future training boundary must
    independently open and verify its own immutable inputs.
    """

    role: str
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class ResolvedTrainingManifest:
    """An immutable validated training declaration and its resolved JSON inputs.

    ``readiness.content_refs_sealed`` means only that every declared file was
    byte-verified.  ``readiness.data_ready`` remains false until the named
    causal-partition proof schema is checked.  ``training_authorized`` is
    always false in r226; a separate later activation receipt must be checked
    by a future launch boundary.
    """

    manifest_path: Path
    manifest_sha256: str
    training_manifest: Mapping[str, Any]
    stage_index_manifest: Mapping[str, Any] | None
    frozen_latent_store_manifest: Mapping[str, Any] | None
    guide_label_overlay_manifest: Mapping[str, Any] | None
    causal_split_manifest: Mapping[str, Any] | None
    resolved_content: tuple[ResolvedContentRef, ...]
    readiness: ReadinessReport

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Alias for the immutable root training manifest."""

        return self.training_manifest


@dataclass(frozen=True, slots=True)
class _StageIndexDeclaration:
    legal_option_key_sha256: str
    stage_count: int
    stage_rows: ContentRef


@dataclass(frozen=True, slots=True)
class _ArtifactDeclaration:
    compatibility: Mapping[str, str]
    stage_index_manifest: ContentRef
    stage_count: int
    nested_content: tuple[tuple[str, ContentRef], ...]


@dataclass(frozen=True, slots=True)
class _TrainingDeclaration:
    status: str
    compatibility: Mapping[str, str | None]
    inputs: Mapping[str, ContentRef | None]
    input_identities: Mapping[str, str | None]
    stage_count: int | None


def validate_stage_index_manifest(payload: object) -> Mapping[str, Any]:
    """Strictly validate and immutably copy a stage-index JSON manifest."""

    _parse_stage_index_manifest(payload)
    return _freeze_json(payload)


def validate_frozen_latent_store_manifest(payload: object) -> Mapping[str, Any]:
    """Strictly validate and immutably copy a frozen-latent-store manifest."""

    _parse_frozen_latent_store_manifest(payload)
    return _freeze_json(payload)


def validate_guide_label_overlay_manifest(payload: object) -> Mapping[str, Any]:
    """Strictly validate and immutably copy a guide label-overlay manifest."""

    _parse_guide_label_overlay_manifest(payload)
    return _freeze_json(payload)


def validate_causal_split_manifest(payload: object) -> Mapping[str, Any]:
    """Strictly validate and immutably copy a causal split manifest."""

    _parse_causal_split_manifest(payload)
    return _freeze_json(payload)


def validate_training_manifest(payload: object) -> Mapping[str, Any]:
    """Strictly validate and immutably copy a dormant or ready declaration.

    This validates only the declaration itself.  Use
    :func:`load_and_validate_training_manifest` to hash-check its referenced
    JSON manifests when its status is ``"ready"``.
    """

    _parse_training_manifest(payload)
    return _freeze_json(payload)


def load_and_validate_training_manifest(
    path: str | Path,
    *,
    require_ready: bool = False,
) -> ResolvedTrainingManifest:
    """Load a strict training declaration and, when ready, resolve its inputs.

    A ``"dormant"`` declaration is intentionally useful before a future guide
    exists: it validates without opening any of its unresolved content refs.
    It never reports content sealing and ``require_ready=True`` rejects it.

    A ``"ready"`` declaration hash-verifies exactly one stage-index, frozen
    latent-store, guide-label-overlay, and causal-split JSON manifest.  Their
    common identities, stage-index bytes, legal-option key, stable join key,
    stage counts, and every nested content ref must all agree.  A current
    ``"ready"`` manifest is therefore content-sealed, not training-ready:
    passing ``require_ready=True`` fails until a future
    ``causal_partition_payload/v1`` semantic split proof is implemented and
    validated.  It never authorizes a training service.
    """

    manifest_path = Path(path)
    raw_manifest, payload = _read_json_object(manifest_path, label="training manifest")
    manifest_sha256 = sha256_bytes(raw_manifest)
    declaration = _parse_training_manifest(payload)
    frozen_root = _freeze_json(payload)

    declared_inputs = tuple(
        name for name in TRAINING_INPUT_FIELDS if declaration.inputs[name] is not None
    )
    if declaration.status == "dormant":
        if require_ready:
            raise Guide2VecContractError(
                "training manifest is dormant: content refs are not sealed and "
                f"semantic split proof is required: {CAUSAL_PARTITION_PAYLOAD_SCHEMA}"
            )
        return ResolvedTrainingManifest(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            training_manifest=frozen_root,
            stage_index_manifest=None,
            frozen_latent_store_manifest=None,
            guide_label_overlay_manifest=None,
            causal_split_manifest=None,
            resolved_content=(),
            readiness=ReadinessReport(
                status="dormant",
                manifest_sha256=manifest_sha256,
                declared_inputs=declared_inputs,
                resolved_inputs=(),
                unresolved_inputs=TRAINING_INPUT_FIELDS,
                content_refs_sealed=False,
                split_semantics_proven=False,
                split_proof_required_schema=CAUSAL_PARTITION_PAYLOAD_SCHEMA,
                blocking_reasons=(
                    f"required semantic split proof: {CAUSAL_PARTITION_PAYLOAD_SCHEMA}",
                    "required separate owner activation receipt",
                ),
                training_authorized=False,
                separate_activation_receipt_required=True,
            ),
        )

    # A ready manifest cannot have nullable refs or identities; this was
    # checked by _parse_training_manifest before any nested file is opened.
    stage_path, stage_payload, stage_sha256 = _read_referenced_json_manifest(
        declaration.inputs["stage_index_manifest"],
        parent_directory=manifest_path.parent,
        label="training.inputs.stage_index_manifest",
    )
    stage = _parse_stage_index_manifest(stage_payload)

    frozen_path, frozen_payload, _ = _read_referenced_json_manifest(
        declaration.inputs["frozen_latent_store_manifest"],
        parent_directory=manifest_path.parent,
        label="training.inputs.frozen_latent_store_manifest",
    )
    frozen = _parse_frozen_latent_store_manifest(frozen_payload)

    overlay_path, overlay_payload, _ = _read_referenced_json_manifest(
        declaration.inputs["guide_label_overlay_manifest"],
        parent_directory=manifest_path.parent,
        label="training.inputs.guide_label_overlay_manifest",
    )
    overlay = _parse_guide_label_overlay_manifest(overlay_payload)

    split_path, split_payload, _ = _read_referenced_json_manifest(
        declaration.inputs["causal_split_manifest"],
        parent_directory=manifest_path.parent,
        label="training.inputs.causal_split_manifest",
    )
    split = _parse_causal_split_manifest(split_payload)

    _require_exact_compatibility(
        expected=declaration.compatibility,
        actual=frozen.compatibility,
        label="frozen latent store",
    )
    _require_exact_compatibility(
        expected=declaration.compatibility,
        actual=overlay.compatibility,
        label="guide label overlay",
    )
    _require_exact_compatibility(
        expected=declaration.compatibility,
        actual=split.compatibility,
        label="causal split",
    )
    _require_equal(
        declaration.compatibility["stage_index_sha256"],
        stage_sha256,
        label="training stage_index_sha256",
    )
    _require_equal(
        declaration.compatibility["legal_option_key_sha256"],
        stage.legal_option_key_sha256,
        label="training legal_option_key_sha256",
    )
    _require_equal(declaration.stage_count, stage.stage_count, label="training stage_count")
    _require_equal(frozen.stage_count, stage.stage_count, label="frozen stage_count")
    _require_equal(overlay.stage_count, stage.stage_count, label="overlay stage_count")
    _require_equal(split.stage_count, stage.stage_count, label="causal split stage_count")

    # Each artifact declares its own stage-index ref.  Verify the raw JSON at
    # every declared path rather than trusting a matching digest string alone.
    _validate_artifact_stage_ref(
        frozen.stage_index_manifest,
        parent_directory=frozen_path.parent,
        expected_stage_sha256=stage_sha256,
        expected_stage=stage,
        label="frozen latent store",
    )
    _validate_artifact_stage_ref(
        overlay.stage_index_manifest,
        parent_directory=overlay_path.parent,
        expected_stage_sha256=stage_sha256,
        expected_stage=stage,
        label="guide label overlay",
    )
    _validate_artifact_stage_ref(
        split.stage_index_manifest,
        parent_directory=split_path.parent,
        expected_stage_sha256=stage_sha256,
        expected_stage=stage,
        label="causal split",
    )

    # A ready declaration is a sealed data boundary, not merely a collection
    # of hash-checked JSON envelopes.  Verify every payload/guide/split ref
    # without deserializing model or tensor bytes.
    resolved_content = (
        _verify_content_ref(
            stage.stage_rows,
            parent_directory=stage_path.parent,
            role="stage_index.stage_rows",
        ),
        *_verify_nested_content(
            frozen.nested_content,
            parent_directory=frozen_path.parent,
            prefix="frozen_latent_store",
        ),
        *_verify_nested_content(
            overlay.nested_content,
            parent_directory=overlay_path.parent,
            prefix="guide_label_overlay",
        ),
        *_verify_nested_content(
            split.nested_content,
            parent_directory=split_path.parent,
            prefix="causal_split",
        ),
    )

    result = ResolvedTrainingManifest(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        training_manifest=frozen_root,
        stage_index_manifest=_freeze_json(stage_payload),
        frozen_latent_store_manifest=_freeze_json(frozen_payload),
        guide_label_overlay_manifest=_freeze_json(overlay_payload),
        causal_split_manifest=_freeze_json(split_payload),
        resolved_content=resolved_content,
        readiness=ReadinessReport(
            status="ready",
            manifest_sha256=manifest_sha256,
            declared_inputs=declared_inputs,
            resolved_inputs=TRAINING_INPUT_FIELDS,
            unresolved_inputs=(),
            content_refs_sealed=True,
            split_semantics_proven=False,
            split_proof_required_schema=CAUSAL_PARTITION_PAYLOAD_SCHEMA,
            blocking_reasons=(
                f"required semantic split proof: {CAUSAL_PARTITION_PAYLOAD_SCHEMA}",
                "required separate owner activation receipt",
            ),
            training_authorized=False,
            separate_activation_receipt_required=True,
        ),
    )
    if require_ready:
        raise Guide2VecContractError(
            "training manifest content is sealed but semantic split proof is "
            f"required before readiness: {CAUSAL_PARTITION_PAYLOAD_SCHEMA}"
        )
    return result


def _parse_stage_index_manifest(payload: object) -> _StageIndexDeclaration:
    value = _strict_mapping(
        payload,
        fields=(
            "schema",
            "source_identity_sha256",
            "legal_option_key_sha256",
            "stage_join_key_fields",
            "digest_tensor_layout",
            "stage_count",
            "stage_rows",
            "authority",
        ),
        label="stage index manifest",
    )
    _require_schema(value["schema"], STAGE_INDEX_SCHEMA, label="stage index manifest")
    _require_sha256(value["source_identity_sha256"], label="stage index source_identity_sha256")
    legal_option_key_sha256 = _require_sha256(
        value["legal_option_key_sha256"],
        label="stage index legal_option_key_sha256",
    )
    _require_stage_join_key(value["stage_join_key_fields"], label="stage index")
    _require_digest_tensor_layout(value["digest_tensor_layout"], label="stage index")
    stage_count = _require_positive_int(value["stage_count"], label="stage index stage_count")
    stage_rows = ContentRef.from_mapping(value["stage_rows"], label="stage index stage_rows")
    _require_inert_authority(value["authority"], label="stage index authority")
    return _StageIndexDeclaration(
        legal_option_key_sha256=legal_option_key_sha256,
        stage_count=stage_count,
        stage_rows=stage_rows,
    )


def _parse_frozen_latent_store_manifest(payload: object) -> _ArtifactDeclaration:
    value = _strict_mapping(
        payload,
        fields=(
            "schema",
            "compatibility",
            "stage_join_key_fields",
            "digest_tensor_layout",
            "stage_index_manifest",
            "stage_count",
            "latent_payload",
            "tensors",
            "guide_labels_embedded",
            "authority",
        ),
        label="frozen latent store manifest",
    )
    _require_schema(
        value["schema"], FROZEN_LATENT_STORE_SCHEMA, label="frozen latent store manifest"
    )
    compatibility = _parse_compatibility(
        value["compatibility"], allow_none=False, label="frozen latent store compatibility"
    )
    _require_stage_join_key(value["stage_join_key_fields"], label="frozen latent store")
    _require_digest_tensor_layout(value["digest_tensor_layout"], label="frozen latent store")
    stage_ref = ContentRef.from_mapping(
        value["stage_index_manifest"], label="frozen latent store stage_index_manifest"
    )
    _require_equal(
        compatibility["stage_index_sha256"],
        stage_ref.sha256,
        label="frozen latent store stage_index_sha256",
    )
    stage_count = _require_positive_int(
        value["stage_count"], label="frozen latent store stage_count"
    )
    latent_payload = ContentRef.from_mapping(
        value["latent_payload"], label="frozen latent store latent_payload"
    )
    _require_exact_string_list(
        value["tensors"], FROZEN_LATENT_TENSORS, label="frozen latent store tensors"
    )
    if type(value["guide_labels_embedded"]) is not bool or value["guide_labels_embedded"]:
        raise Guide2VecContractError(
            "frozen latent store guide_labels_embedded must be exactly false"
        )
    _require_inert_authority(value["authority"], label="frozen latent store authority")
    return _ArtifactDeclaration(
        compatibility=_freeze_mapping(compatibility),
        stage_index_manifest=stage_ref,
        stage_count=stage_count,
        nested_content=(("latent_payload", latent_payload),),
    )


def _parse_guide_label_overlay_manifest(payload: object) -> _ArtifactDeclaration:
    value = _strict_mapping(
        payload,
        fields=(
            "schema",
            "compatibility",
            "stage_join_key_fields",
            "digest_tensor_layout",
            "stage_index_manifest",
            "stage_count",
            "label_payload",
            "label_fields",
            "guide_contract",
            "teacher_semantics",
            "teacher_dependencies",
            "authority",
        ),
        label="guide label overlay manifest",
    )
    _require_schema(
        value["schema"], GUIDE_LABEL_OVERLAY_SCHEMA, label="guide label overlay manifest"
    )
    compatibility = _parse_compatibility(
        value["compatibility"], allow_none=False, label="guide label overlay compatibility"
    )
    _require_stage_join_key(value["stage_join_key_fields"], label="guide label overlay")
    _require_digest_tensor_layout(value["digest_tensor_layout"], label="guide label overlay")
    stage_ref = ContentRef.from_mapping(
        value["stage_index_manifest"], label="guide label overlay stage_index_manifest"
    )
    _require_equal(
        compatibility["stage_index_sha256"],
        stage_ref.sha256,
        label="guide label overlay stage_index_sha256",
    )
    stage_count = _require_positive_int(
        value["stage_count"], label="guide label overlay stage_count"
    )
    label_payload = ContentRef.from_mapping(
        value["label_payload"], label="guide label overlay label_payload"
    )
    _require_exact_string_list(
        value["label_fields"], GUIDE_LABEL_FIELDS, label="guide label overlay label_fields"
    )
    guide_contract = ContentRef.from_mapping(
        value["guide_contract"], label="guide label overlay guide_contract"
    )
    teacher_semantics = ContentRef.from_mapping(
        value["teacher_semantics"], label="guide label overlay teacher_semantics"
    )
    teacher_dependencies = _parse_content_ref_list(
        value["teacher_dependencies"], label="guide label overlay teacher_dependencies"
    )
    _require_inert_authority(value["authority"], label="guide label overlay authority")
    return _ArtifactDeclaration(
        compatibility=_freeze_mapping(compatibility),
        stage_index_manifest=stage_ref,
        stage_count=stage_count,
        nested_content=(
            ("label_payload", label_payload),
            ("guide_contract", guide_contract),
            ("teacher_semantics", teacher_semantics),
            *tuple(
                (f"teacher_dependencies[{index}]", ref)
                for index, ref in enumerate(teacher_dependencies)
            ),
        ),
    )


def _parse_causal_split_manifest(payload: object) -> _ArtifactDeclaration:
    value = _strict_mapping(
        payload,
        fields=(
            "schema",
            "compatibility",
            "stage_join_key_fields",
            "digest_tensor_layout",
            "stage_index_manifest",
            "stage_count",
            "partition_payload",
            "partition_guards",
            "authority",
        ),
        label="causal split manifest",
    )
    _require_schema(value["schema"], CAUSAL_SPLIT_SCHEMA, label="causal split manifest")
    compatibility = _parse_compatibility(
        value["compatibility"], allow_none=False, label="causal split compatibility"
    )
    _require_stage_join_key(value["stage_join_key_fields"], label="causal split")
    _require_digest_tensor_layout(value["digest_tensor_layout"], label="causal split")
    stage_ref = ContentRef.from_mapping(
        value["stage_index_manifest"], label="causal split stage_index_manifest"
    )
    _require_equal(
        compatibility["stage_index_sha256"],
        stage_ref.sha256,
        label="causal split stage_index_sha256",
    )
    stage_count = _require_positive_int(value["stage_count"], label="causal split stage_count")
    partition_payload = ContentRef.from_mapping(
        value["partition_payload"], label="causal split partition_payload"
    )
    _require_partition_guards(value["partition_guards"], label="causal split partition_guards")
    _require_inert_authority(value["authority"], label="causal split authority")
    return _ArtifactDeclaration(
        compatibility=_freeze_mapping(compatibility),
        stage_index_manifest=stage_ref,
        stage_count=stage_count,
        nested_content=(("partition_payload", partition_payload),),
    )


def _parse_training_manifest(payload: object) -> _TrainingDeclaration:
    value = _strict_mapping(
        payload,
        fields=(
            "schema",
            "status",
            "compatibility",
            "stage_join_key_fields",
            "digest_tensor_layout",
            "stage_count",
            "inputs",
            "input_identities",
            "authority",
            "activation_receipt",
        ),
        label="training manifest",
    )
    _require_schema(value["schema"], TRAINING_MANIFEST_SCHEMA, label="training manifest")
    status = value["status"]
    if not isinstance(status, str) or status not in {"dormant", "ready"}:
        raise Guide2VecContractError("training manifest status must be 'dormant' or 'ready'")
    compatibility = _parse_compatibility(
        value["compatibility"],
        allow_none=status == "dormant",
        label="training manifest compatibility",
    )
    _require_stage_join_key(value["stage_join_key_fields"], label="training manifest")
    _require_digest_tensor_layout(value["digest_tensor_layout"], label="training manifest")
    stage_count = _require_optional_positive_int(
        value["stage_count"], label="training manifest stage_count"
    )
    inputs = _parse_training_inputs(value["inputs"])
    input_identities = _parse_training_input_identities(value["input_identities"])
    _require_inert_authority(value["authority"], label="training manifest authority")
    if value["activation_receipt"] is not None:
        raise Guide2VecContractError(
            "activation_receipt must be null; owner activation is a separate later contract"
        )

    for ref_name, identity_name in _INPUT_IDENTITY_BY_REF.items():
        ref = inputs[ref_name]
        identity = input_identities[identity_name]
        if ref is None and identity is not None:
            raise Guide2VecContractError(
                f"training {identity_name} is set without {ref_name}"
            )
        if ref is not None and identity != ref.sha256:
            raise Guide2VecContractError(
                f"training {identity_name} must equal {ref_name}.sha256"
            )

    stage_ref = inputs["stage_index_manifest"]
    stage_identity = compatibility["stage_index_sha256"]
    if stage_ref is None and stage_identity is not None:
        raise Guide2VecContractError(
            "training compatibility.stage_index_sha256 is set without stage_index_manifest"
        )
    if stage_ref is not None and stage_identity is not None and stage_identity != stage_ref.sha256:
        raise Guide2VecContractError(
            "training compatibility.stage_index_sha256 must equal stage_index_manifest.sha256"
        )

    if status == "ready":
        missing_refs = [name for name in TRAINING_INPUT_FIELDS if inputs[name] is None]
        missing_identities = [
            name for name in TRAINING_INPUT_IDENTITY_FIELDS if input_identities[name] is None
        ]
        missing_compatibility = [
            name for name in COMPATIBILITY_FIELDS if compatibility[name] is None
        ]
        if missing_refs or missing_identities or missing_compatibility or stage_count is None:
            raise Guide2VecContractError(
                "ready training manifest requires every input, input identity, "
                "compatibility identity, and stage_count: "
                f"missing_refs={missing_refs} missing_identities={missing_identities} "
                f"missing_compatibility={missing_compatibility} "
                f"missing_stage_count={stage_count is None}"
            )

    return _TrainingDeclaration(
        status=status,
        compatibility=_freeze_mapping(compatibility),
        inputs=_freeze_mapping(inputs),
        input_identities=_freeze_mapping(input_identities),
        stage_count=stage_count,
    )


def _parse_compatibility(
    value: object,
    *,
    allow_none: bool,
    label: str,
) -> dict[str, str | None]:
    payload = _strict_mapping(value, fields=COMPATIBILITY_FIELDS, label=label)
    result: dict[str, str | None] = {}
    for field in COMPATIBILITY_FIELDS:
        raw = payload[field]
        if raw is None:
            if not allow_none:
                raise Guide2VecContractError(f"{label}.{field} is required")
            result[field] = None
        else:
            result[field] = _require_sha256(raw, label=f"{label}.{field}")
    return result


def _parse_training_inputs(value: object) -> dict[str, ContentRef | None]:
    payload = _strict_mapping(value, fields=TRAINING_INPUT_FIELDS, label="training inputs")
    result: dict[str, ContentRef | None] = {}
    for field in TRAINING_INPUT_FIELDS:
        raw = payload[field]
        result[field] = (
            None
            if raw is None
            else ContentRef.from_mapping(raw, label=f"training inputs.{field}")
        )
    return result


def _parse_training_input_identities(value: object) -> dict[str, str | None]:
    payload = _strict_mapping(
        value, fields=TRAINING_INPUT_IDENTITY_FIELDS, label="training input_identities"
    )
    result: dict[str, str | None] = {}
    for field in TRAINING_INPUT_IDENTITY_FIELDS:
        raw = payload[field]
        result[field] = None if raw is None else _require_sha256(raw, label=field)
    return result


def _validate_artifact_stage_ref(
    ref: ContentRef,
    *,
    parent_directory: Path,
    expected_stage_sha256: str,
    expected_stage: _StageIndexDeclaration,
    label: str,
) -> None:
    _, payload, digest = _read_referenced_json_manifest(
        ref,
        parent_directory=parent_directory,
        label=f"{label}.stage_index_manifest",
    )
    stage = _parse_stage_index_manifest(payload)
    _require_equal(digest, expected_stage_sha256, label=f"{label} stage-index bytes")
    _require_equal(
        stage.legal_option_key_sha256,
        expected_stage.legal_option_key_sha256,
        label=f"{label} stage-index legal option key",
    )
    _require_equal(stage.stage_count, expected_stage.stage_count, label=f"{label} stage-index count")


def _verify_nested_content(
    refs: Sequence[tuple[str, ContentRef]],
    *,
    parent_directory: Path,
    prefix: str,
) -> tuple[ResolvedContentRef, ...]:
    return tuple(
        _verify_content_ref(
            ref,
            parent_directory=parent_directory,
            role=f"{prefix}.{name}",
        )
        for name, ref in refs
    )


def _verify_content_ref(
    ref: ContentRef,
    *,
    parent_directory: Path,
    role: str,
) -> ResolvedContentRef:
    """Raw-hash a sealed payload through one stable open file descriptor."""

    path = ref.resolve(parent_directory)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise Guide2VecContractError(f"content reference is not a file: {path}")
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
            after = os.fstat(stream.fileno())
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise Guide2VecContractError(f"content reference is not a file: {path}") from exc
    except OSError as exc:
        raise Guide2VecContractError(f"could not read content reference: {path}") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or byte_count != after.st_size:
        raise Guide2VecContractError(
            f"content reference changed while being verified: {path}"
        )
    actual_sha256 = "sha256:" + digest.hexdigest()
    _require_equal(actual_sha256, ref.sha256, label=f"{role} content digest")
    return ResolvedContentRef(
        role=role,
        path=path,
        sha256=actual_sha256,
        bytes=byte_count,
    )


def _read_referenced_json_manifest(
    ref: ContentRef | None,
    *,
    parent_directory: Path,
    label: str,
) -> tuple[Path, Mapping[str, Any], str]:
    if ref is None:
        raise Guide2VecContractError(f"{label} is required")
    path = ref.resolve(parent_directory)
    raw, payload = _read_json_object(path, label=label)
    actual_sha256 = sha256_bytes(raw)
    _require_equal(actual_sha256, ref.sha256, label=f"{label} content digest")
    return path, payload, actual_sha256


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise Guide2VecContractError(f"could not read {label}: {path}") from exc
    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, Guide2VecContractError) as exc:
        raise Guide2VecContractError(f"{label} is not strict UTF-8 JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Guide2VecContractError(f"{label} must be a JSON object: {path}")
    return raw, payload


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Guide2VecContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise Guide2VecContractError(f"non-standard JSON constant: {value}")


def _strict_mapping(
    value: object,
    *,
    fields: Sequence[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Guide2VecContractError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise Guide2VecContractError(f"{label} has a non-string field name")
    expected = set(fields)
    actual = set(value)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing or unknown:
        raise Guide2VecContractError(
            f"{label} fields changed: missing={missing} unknown={unknown}"
        )
    return value


def _require_schema(value: object, expected: str, *, label: str) -> None:
    if value != expected or not isinstance(value, str):
        raise Guide2VecContractError(f"{label}.schema must be exactly {expected!r}")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Guide2VecContractError(f"{label} must be a canonical lowercase sha256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Guide2VecContractError(f"{label} must be a positive exact integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Guide2VecContractError(f"{label} must be a non-negative exact integer")
    return value


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Guide2VecContractError(f"{label} must be a non-empty string")
    return value


def _require_optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, label=label)


def _require_stage_join_key(value: object, *, label: str) -> None:
    _require_exact_string_list(value, STAGE_JOIN_KEY_FIELDS, label=f"{label} stage_join_key_fields")


def _require_digest_tensor_layout(value: object, *, label: str) -> None:
    """Require the compact raw SHA-256 ``uint8 [rows, 32]`` representation."""

    payload = _strict_mapping(value, fields=DIGEST_TENSOR_LAYOUT_FIELDS, label=f"{label} digest_tensor_layout")
    if payload["dtype"] != DIGEST_TENSOR_LAYOUT["dtype"]:
        raise Guide2VecContractError(f"{label} digest_tensor_layout.dtype must be 'uint8'")
    if not isinstance(payload["shape"], list) or tuple(payload["shape"]) != (
        "rows",
        32,
    ):
        raise Guide2VecContractError(
            f"{label} digest_tensor_layout.shape must be ['rows', 32]"
        )
    for field in ("encoding", "stage_key_digest", "legal_option_digest"):
        if payload[field] != DIGEST_TENSOR_LAYOUT[field]:
            raise Guide2VecContractError(
                f"{label} digest_tensor_layout.{field} must be "
                f"{DIGEST_TENSOR_LAYOUT[field]!r}"
            )


def _require_exact_string_list(
    value: object,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise Guide2VecContractError(f"{label} must be an exact JSON string list")
    if tuple(value) != tuple(expected):
        raise Guide2VecContractError(f"{label} must equal {list(expected)!r}")


def _parse_content_ref_list(value: object, *, label: str) -> tuple[ContentRef, ...]:
    if not isinstance(value, list):
        raise Guide2VecContractError(f"{label} must be a JSON list")
    seen: set[tuple[str, str]] = set()
    refs: list[ContentRef] = []
    for index, item in enumerate(value):
        ref = ContentRef.from_mapping(item, label=f"{label}[{index}]")
        identity = (ref.path, ref.sha256)
        if identity in seen:
            raise Guide2VecContractError(f"{label} has a duplicate content reference")
        seen.add(identity)
        refs.append(ref)
    return tuple(refs)


def _require_partition_guards(value: object, *, label: str) -> None:
    payload = _strict_mapping(
        value,
        fields=(
            "whole_episode_disjoint",
            "whole_source_day_disjoint",
            "heldout_sealed_until_candidate_selection",
        ),
        label=label,
    )
    for field in (
        "whole_episode_disjoint",
        "whole_source_day_disjoint",
        "heldout_sealed_until_candidate_selection",
    ):
        if type(payload[field]) is not bool or not payload[field]:
            raise Guide2VecContractError(f"{label}.{field} must be exactly true")


def _require_inert_authority(value: object, *, label: str) -> None:
    payload = _strict_mapping(value, fields=INERT_AUTHORITY_FIELDS, label=label)
    for field in INERT_AUTHORITY_FIELDS:
        if type(payload[field]) is not bool or payload[field]:
            raise Guide2VecContractError(f"{label}.{field} must be exactly false")


def _require_exact_compatibility(
    *,
    expected: Mapping[str, str | None],
    actual: Mapping[str, str],
    label: str,
) -> None:
    for field in COMPATIBILITY_FIELDS:
        if expected[field] is None:
            raise Guide2VecContractError(
                f"ready training compatibility.{field} unexpectedly missing"
            )
        _require_equal(
            expected[field], actual[field], label=f"{label} compatibility.{field}"
        )


def _require_equal(left: object, right: object, *, label: str) -> None:
    if left != right:
        raise Guide2VecContractError(f"{label} mismatch: {left!r} != {right!r}")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _freeze_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_compatible(value: object, *, label: str) -> Any:
    """Recursively normalize standard JSON types while rejecting lossy values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise Guide2VecContractError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Guide2VecContractError(f"{label} has a non-string object key")
            normalized[key] = _json_compatible(item, label=label)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item, label=label) for item in value]
    raise Guide2VecContractError(f"{label} contains unsupported type {type(value).__name__}")


__all__ = [
    "CAUSAL_PARTITION_PAYLOAD_SCHEMA",
    "CAUSAL_SPLIT_SCHEMA",
    "COMPATIBILITY_FIELDS",
    "DIGEST_TENSOR_LAYOUT",
    "DIGEST_TENSOR_LAYOUT_FIELDS",
    "FROZEN_LATENT_STORE_SCHEMA",
    "FROZEN_LATENT_TENSORS",
    "GUIDE_LABEL_FIELDS",
    "GUIDE_LABEL_OVERLAY_SCHEMA",
    "INERT_AUTHORITY_FIELDS",
    "STAGE_INDEX_SCHEMA",
    "STAGE_JOIN_KEY_FIELDS",
    "TRAINING_INPUT_FIELDS",
    "TRAINING_INPUT_IDENTITY_FIELDS",
    "TRAINING_MANIFEST_SCHEMA",
    "ContentRef",
    "Guide2VecContractError",
    "ReadinessReport",
    "ResolvedContentRef",
    "ResolvedTrainingManifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "digest_tensor_bytes",
    "load_and_validate_training_manifest",
    "sha256_bytes",
    "sha256_file",
    "stage_key_sha256",
    "validate_causal_split_manifest",
    "validate_frozen_latent_store_manifest",
    "validate_guide_label_overlay_manifest",
    "validate_stage_index_manifest",
    "validate_training_manifest",
]
