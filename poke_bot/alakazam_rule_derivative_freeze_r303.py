"""Revision-5 schema-freeze and materialization preflight for the rule derivative.

This is an additive, receipt-only bridge.  It deliberately does not import the
live policy, train a tensor, start a worker, stage bytes to Inzi, or update a
selector.  Its narrow role is to bind the already-versioned public-rule,
target, checklist, and Card2Vec-adjacent surfaces into the canonical
pre-refeaturization artifacts required by the rev5/r303 contract.

The historical r4 catalog can be inspected here only through the checklist
owner's explicit consumer-migration boundary.  Until that migration is pinned
by its owner, this module cannot issue a frozen schema manifest.  Likewise,
selected-action targets remain diagnostic/all-masked until the target owner's
post-handoff sealed trajectory validator exists; re-featurization may still
record their public provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .alakazam_checklist_provenance_r298 import (
    R298_CHECKLIST_PROVENANCE_SCHEMA,
    R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA,
    R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA,
    ChecklistProvenanceError,
    checklist_provenance_schema_manifest_r298,
    revision_5_consumer_migration_schema_manifest_r298,
    validate_pinned_public_catalog_r298,
    validate_revision_5_consumer_migration_r298,
    validate_sealed_public_catalog_r298,
)
from .alakazam_collision_census_r298 import (
    CANONICAL_R236_LIBCG_SHA256,
    CANONICAL_R236_LIBCG_SIZE_BYTES,
    EXACT_NEW_LIST_MULTISET_SHA256,
    MECHANICS_ATTACHMENT_SHA256,
    OWNER_GOAL_SHA256,
    R274_EXACT_FEATURES_SOURCE_SHA256,
    R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
    R298_RAW_CORPUS_RECEIPT_SCHEMA,
    R298_ZERO_BYPASS_RECEIPT_SCHEMA,
    R298_OWNER_REVISION,
    RULE_DERIVATIVE_CONTRACT_SHA256,
    RULE_DERIVATIVE_GATEWAY_SHA256,
    REVISION_5_GOAL_REVISION,
    REVISION_5_ROOT_HANDOFF_REVISION,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    revision_5_predecessor_classification,
    validate_frozen_schema_gate,
    validate_raw_corpus_manifest,
    validate_revision_5_predecessor_classification,
)
from .alakazam_rule_aux_heads_r298 import load_r298_aux_heads_config, r298_aux_heads_schema_manifest
from .alakazam_rule_derivative_handoff_r303 import (
    R303_CONTRACT_PATH,
    R303_CONTRACT_SHA256,
    R303_GOAL_GATEWAY_SHA256,
    R303_GOAL_REVISION,
    R303_ROOT_OWNER_REVISION,
    R303HandoffError,
    load_r303_contract,
    validate_handoff_receipt,
)
from .alakazam_rule_derivative_model_r298 import semantic_projection_schema_manifest
from .alakazam_simulator_rule_targets_r298 import (
    R298_RULE_TARGET_SCHEMA,
    assert_r298_rule_target_schema_binding,
    r298_rule_target_schema_manifest,
)


R303_FREEZE_PRODUCER_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_freeze_producer/v1"
R303_SEALED_CATALOG_BINDING_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_sealed_catalog_binding/v1"
)
R303_MATERIALIZATION_PREFLIGHT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_materialization_preflight/v1"
)
R303_SCHEMA_FREEZE_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_schema_freeze_receipt/v1"
)
R303_SCHEMA_FREEZE_BUNDLE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_schema_freeze_bundle/v1"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
_SHA256_PREFIX: Final = "sha256:"

_NEW_BRANCHES: Final = (
    "public_rule_semantic_projection",
    "public_rule_metadata_residual",
    "r298_repaired_auxiliary_heads",
    "eight_checklist_provenance_gates",
)
_MATERIALIZED_RECORD_SURFACES: Final = (
    "legacy_feature_signatures",
    "public_rule_representation",
    "selected_action_targets_diagnostic_all_masks_false_without_sealed_validator",
    "strict_checklist_provenance_trace",
    "source_day_group_seed_split_provenance",
)


class R303FreezeError(RuntimeError):
    """A proposed revision-5 freeze has incomplete or stale evidence."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R303FreezeError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise R303FreezeError(f"{field} must be a list")
    return list(value)


def _sha(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except Exception as exc:
        raise R303FreezeError(str(exc)) from exc


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise R303FreezeError(f"{field} must be a positive exact integer")
    return value


def _utc_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise R303FreezeError(f"{field} must be a nonempty UTC timestamp string")
    if not value.endswith("Z"):
        raise R303FreezeError(f"{field} must use an explicit UTC Z suffix")
    return value


def _canonical_digest(value: Any) -> str:
    try:
        return canonical_sha256(value)
    except Exception as exc:
        raise R303FreezeError("value is not canonical JSON") from exc


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise R303FreezeError(f"cannot read immutable source: {path}") from exc
    return _SHA256_PREFIX + digest.hexdigest()


def _regular_file_identity(path: Path | str, *, field: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise R303FreezeError(f"{field} must be a regular non-symlink file")
    resolved = target.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _source_identity(module_or_path: str | Path, *, field: str) -> dict[str, Any]:
    return _regular_file_identity(module_or_path, field=field)


def _r303_contract() -> Mapping[str, Any]:
    """Open the rev5 contract through the canonical handoff owner."""

    try:
        contract = load_r303_contract()
    except (R303HandoffError, OSError) as exc:
        raise R303FreezeError("canonical rev5/r303 contract is unavailable") from exc
    if (
        contract.get("goal_revision"),
        contract.get("root_handoff_revision"),
    ) != (R303_GOAL_REVISION, R303_ROOT_OWNER_REVISION):
        raise R303FreezeError("canonical contract has a foreign goal revision")
    return contract


def _validate_r303_authority() -> None:
    _r303_contract()
    if (
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        R303_CONTRACT_SHA256,
        R303_GOAL_GATEWAY_SHA256,
    ) != (
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        RULE_DERIVATIVE_GATEWAY_SHA256,
    ):
        raise R303FreezeError("r303 and collision-census authority bindings disagree")


def _validate_assertions(assertions: Mapping[str, Any]) -> dict[str, bool]:
    """Validate the checklist owner's closed migration assertion inventory."""

    schema = revision_5_consumer_migration_schema_manifest_r298()
    definition = _mapping(schema.get("definition"), field="checklist migration schema definition")
    required = _rows(definition.get("required_assertions"), field="checklist migration assertions")
    if not required or any(not isinstance(name, str) or not name for name in required):
        raise R303FreezeError("checklist migration assertion schema is malformed")
    if set(assertions) != set(required):
        raise R303FreezeError("checklist migration assertion inventory drifted")
    result: dict[str, bool] = {}
    for name in required:
        if assertions.get(name) is not True:
            raise R303FreezeError(f"checklist migration assertion is not passed: {name}")
        result[name] = True
    return result


def build_revision_5_consumer_migration_payload(
    *,
    consumer_rebind_assertions: Mapping[str, Any],
    validated_at_utc: str,
) -> dict[str, Any]:
    """Build the owner-pinnable, non-circular r5 catalog migration payload.

    This is intentionally not an authority issuer: callers provide the closed
    all-true assertion set after completing their evidence checks, then the
    checklist owner alone pins the resulting immutable file/payload hashes in
    its config.  The payload has no config/schema/zero-bypass digest, avoiding
    the documented hash cycle.
    """

    _validate_r303_authority()
    assertions = _validate_assertions(_mapping(consumer_rebind_assertions, field="migration assertions"))
    timestamp = _utc_text(validated_at_utc, field="migration validated_at_utc")
    schema = revision_5_consumer_migration_schema_manifest_r298()
    definition = _mapping(schema.get("definition"), field="checklist migration schema definition")
    required_fields = set(_rows(definition.get("required_receipt_fields"), field="migration receipt fields"))
    payload = {
        "schema": R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA,
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "r241_typed_source_path": "state/alakazam-new-list-direct-policy-r241.json",
        "r241_typed_source_sha256": "sha256:8d83f5e9eafc8e554f33dcbfbda7e1b337b8f65dc2e49109be4acdd829850c1e",
        "predecessor_goal_revision": 4,
        "predecessor_gateway_sha256": "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8",
        "predecessor_contract_sha256": "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2",
        "predecessor_catalog_sealer_receipt_sha256": "sha256:9ca7534281dd08d1804babe223fa0779a84a1f5a7ee6f609497d595dca10b4d1",
        "predecessor_catalog_sha256": "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6",
        "predecessor_fixed_vectors_sha256": "sha256:2e1a817dac1d17d0131056ead9669b76803b41f5864d4e9cd82d7f34a9180e09",
        "catalog_classification": "immutable_revision_4_predecessor_evidence_not_revision_5_authority_without_this_receipt",
        "blind_goal_or_contract_hash_substitution_allowed": False,
        "revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze": False,
        "consumer_rebind_assertions": assertions,
        "default_zero_and_inert": True,
        "runtime_wired": False,
        "production_or_inzi_authority": False,
        "validated_at_utc": timestamp,
    }
    if set(payload) != required_fields:
        raise R303FreezeError("constructed checklist migration payload drifted from owner schema")
    return payload


def validate_revision_5_consumer_migration_payload_for_issue(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate shape/content before create-only emission, without pinning it.

    The checklist owner's :func:`validate_revision_5_consumer_migration_r298`
    additionally requires the config anchor and is the only function that
    grants this payload downstream authority.
    """

    row = _mapping(payload, field="revision-5 consumer migration payload")
    assertions = _mapping(row.get("consumer_rebind_assertions"), field="migration assertions")
    expected = build_revision_5_consumer_migration_payload(
        consumer_rebind_assertions=assertions,
        validated_at_utc=_utc_text(row.get("validated_at_utc"), field="migration validated_at_utc"),
    )
    if dict(row) != expected:
        raise R303FreezeError("revision-5 consumer migration payload is stale or foreign")
    return expected


def write_revision_5_consumer_migration_create_only(
    path: Path | str, payload: Mapping[str, Any]
) -> str:
    """Write a proposed migration receipt exactly once for checklist-owner pinning."""

    row = validate_revision_5_consumer_migration_payload_for_issue(payload)
    return _write_create_only_json(Path(path), row)


def _adapter_config_binding() -> dict[str, Any]:
    """Require the public adapter owner to have migrated its own config to r5."""

    try:
        from . import alakazam_public_rule_adapter_r298 as adapter

        config = adapter.load_public_rule_adapter_config()
        config_path = Path(adapter.DEFAULT_CONFIG_PATH)
    except Exception as exc:
        raise R303FreezeError(
            "public-rule adapter is not yet revision-5 schema-bound"
        ) from exc
    authority = _mapping(config.get("authority"), field="public-rule adapter authority")
    if (
        authority.get("canonical_goal"),
        authority.get("canonical_contract"),
        authority.get("canonical_goal_sha256"),
        authority.get("canonical_contract_sha256"),
        authority.get("root_handoff_revision"),
    ) != (
        "goals/alakazam-elmo-rule-derivative/GOAL.md",
        "goals/alakazam-elmo-rule-derivative/contract.json",
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_SHA256,
        R303_ROOT_OWNER_REVISION,
    ):
        raise R303FreezeError("public-rule adapter authority is not revision-5/r303")
    runtime = _mapping(config.get("runtime"), field="public-rule adapter runtime")
    gates = _mapping(runtime.get("zero_gates"), field="public-rule adapter zero gates")
    if (
        runtime.get("enabled_default") is not False
        or runtime.get("runtime_wired") is not False
        or runtime.get("production_or_inzi_authority") is not False
        or any(value != 0.0 for value in gates.values())
    ):
        raise R303FreezeError("public-rule adapter is not exact-zero and unwired")
    source = _source_identity(adapter.__file__, field="public-rule adapter module")
    config_identity = _source_identity(config_path, field="public-rule adapter config")
    return {
        "schema": str(config.get("schema")),
        "config": config_identity,
        "config_payload_sha256": _canonical_digest(config),
        "module": source,
        "representation_schema": getattr(adapter, "PUBLIC_RULE_REPRESENTATION_SCHEMA", None),
        "adapter_schema": getattr(adapter, "PUBLIC_RULE_ADAPTER_SCHEMA", None),
        "runtime_wired": False,
        "all_gates_exact_zero": True,
    }


def build_r303_sealed_catalog_binding(
    *,
    catalog_sealer_receipt_path: Path | str,
    revision_5_consumer_migration_receipt_path: Path | str,
) -> dict[str, Any]:
    """Bind the immutable catalog only after checklist-owner r5 pinning.

    The checklist APIs reopen the r4 sealer receipt/catalog/vector bytes and
    then prove that the supplied r5 migration receipt is the owner-pinned
    immutable artifact.  No raw CSV, old MetadataCatalog, or guide prose can
    enter this binding.
    """

    _validate_r303_authority()
    try:
        predecessor = validate_sealed_public_catalog_r298(catalog_sealer_receipt_path)
        migration = validate_revision_5_consumer_migration_r298(
            revision_5_consumer_migration_receipt_path
        )
        catalog = validate_pinned_public_catalog_r298(
            catalog_sealer_receipt_path,
            revision_5_consumer_migration_receipt_path=revision_5_consumer_migration_receipt_path,
        )
    except ChecklistProvenanceError as exc:
        raise R303FreezeError(
            "sealed public catalog is not eligible for revision-5 schema freeze"
        ) from exc
    if catalog.sha256 != predecessor.catalog.sha256:
        raise R303FreezeError("checklist-pinned catalog identity disagrees with sealer evidence")
    predecessor_row = predecessor.to_dict()
    migration_row = migration.to_dict()
    catalog_row = catalog.to_dict()
    return {
        "schema": R303_SEALED_CATALOG_BINDING_SCHEMA,
        "status": "passed_revision_5_pinned_catalog_consumer",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "canonical_libcg_sha256": CANONICAL_R236_LIBCG_SHA256,
        "canonical_libcg_size_bytes": CANONICAL_R236_LIBCG_SIZE_BYTES,
        "catalog_predecessor_evidence": predecessor_row,
        "pinned_catalog": catalog_row,
        "revision_5_consumer_migration": migration_row,
        "catalog_file_sha256": catalog.sha256,
        "catalog_file_size_bytes": catalog.size_bytes,
        "fixed_vectors_file_sha256": predecessor.fixed_vectors.sha256,
        "fixed_vectors_file_size_bytes": predecessor.fixed_vectors.size_bytes,
        "catalog_semantic_sha256": predecessor.catalog_semantic_sha256,
        "structured_rule_vectors_sha256": "sha256:6602af831128364a7a33ea3e691a73e4cac7e39ad037855ada7b13c80f87ebb1",
        "engine_cards_sha256": "sha256:5e92642577d5e61324e4d4095b883fb7926e4d7e822967b48778ecd1996998f4",
        "engine_attacks_sha256": "sha256:97834eb17429fbacfedcf563d43753b129103c0a1d1941d1ee9c9af36392be3f",
        "card2vec_preserved_unchanged": True,
        "structured_residual_additive_after_card2vec": True,
        "csv_or_text_hash_mechanics_authority": False,
        "raw_mapping_or_legacy_metadata_eligible": False,
        "runtime_wired": False,
        "training_or_inzi_authority": False,
    }


def _validate_catalog_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(binding, field="revision-5 sealed catalog binding")
    required = {
        "schema",
        "status",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "canonical_libcg_sha256",
        "canonical_libcg_size_bytes",
        "catalog_predecessor_evidence",
        "pinned_catalog",
        "revision_5_consumer_migration",
        "catalog_file_sha256",
        "catalog_file_size_bytes",
        "fixed_vectors_file_sha256",
        "fixed_vectors_file_size_bytes",
        "catalog_semantic_sha256",
        "structured_rule_vectors_sha256",
        "engine_cards_sha256",
        "engine_attacks_sha256",
        "card2vec_preserved_unchanged",
        "structured_residual_additive_after_card2vec",
        "csv_or_text_hash_mechanics_authority",
        "raw_mapping_or_legacy_metadata_eligible",
        "runtime_wired",
        "training_or_inzi_authority",
    }
    if set(row) != required:
        raise R303FreezeError("sealed catalog binding field inventory drifted")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("canonical_libcg_sha256"),
        row.get("canonical_libcg_size_bytes"),
    ) != (
        R303_SEALED_CATALOG_BINDING_SCHEMA,
        "passed_revision_5_pinned_catalog_consumer",
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        CANONICAL_R236_LIBCG_SHA256,
        CANONICAL_R236_LIBCG_SIZE_BYTES,
    ):
        raise R303FreezeError("sealed catalog binding authority drifted")
    for field in (
        "catalog_file_sha256",
        "fixed_vectors_file_sha256",
        "catalog_semantic_sha256",
        "structured_rule_vectors_sha256",
        "engine_cards_sha256",
        "engine_attacks_sha256",
    ):
        _sha(row.get(field), field=f"sealed catalog binding.{field}")
    _positive_int(row.get("catalog_file_size_bytes"), field="catalog file size")
    _positive_int(row.get("fixed_vectors_file_size_bytes"), field="catalog vectors file size")
    for field, expected in (
        ("card2vec_preserved_unchanged", True),
        ("structured_residual_additive_after_card2vec", True),
        ("csv_or_text_hash_mechanics_authority", False),
        ("raw_mapping_or_legacy_metadata_eligible", False),
        ("runtime_wired", False),
        ("training_or_inzi_authority", False),
    ):
        if row.get(field) is not expected:
            raise R303FreezeError(f"sealed catalog binding.{field} is unsafe")
    predecessor = _mapping(row.get("catalog_predecessor_evidence"), field="catalog predecessor evidence")
    if predecessor.get("predecessor_only") is not True or predecessor.get("revision_5_mechanics_authority") is not False:
        raise R303FreezeError("catalog predecessor evidence was promoted directly")
    migration = _mapping(row.get("revision_5_consumer_migration"), field="catalog migration")
    receipt = _mapping(migration.get("receipt"), field="catalog migration receipt")
    if receipt.get("sha256") is None:
        raise R303FreezeError("catalog migration has no immutable receipt identity")
    if migration.get("predecessor_catalog_sha256") != row.get("catalog_file_sha256"):
        raise R303FreezeError("catalog migration does not bind catalog bytes")
    if migration.get("predecessor_fixed_vectors_sha256") != row.get("fixed_vectors_file_sha256"):
        raise R303FreezeError("catalog migration does not bind vector bytes")
    return dict(row)


def _target_and_auxiliary_binding() -> dict[str, Any]:
    """Load the target owner APIs, retaining the all-mask pre-handoff state."""

    try:
        target = r298_rule_target_schema_manifest()
        assert_r298_rule_target_schema_binding(target, require_current_goal_files=True)
        auxiliary_config = load_r298_aux_heads_config()
        auxiliary_schema = r298_aux_heads_schema_manifest()
    except Exception as exc:
        raise R303FreezeError("selected-action target/auxiliary schema binding is stale") from exc
    target = dict(_mapping(target, field="target schema manifest"))
    auxiliary_config = dict(_mapping(auxiliary_config, field="auxiliary-head config"))
    auxiliary_schema = dict(_mapping(auxiliary_schema, field="auxiliary-head schema"))
    target_digest = target.get("schema_digest")
    if not isinstance(target_digest, str):
        raise R303FreezeError("target schema lacks its digest")
    target_digest = _sha(target_digest, field="target schema digest")
    target_contract = _mapping(auxiliary_config.get("target_contract"), field="auxiliary target contract")
    if target_contract.get("schema") != R298_RULE_TARGET_SCHEMA or target_contract.get("schema_digest") != target_digest:
        raise R303FreezeError("auxiliary config does not bind the current target schema")
    runtime = _mapping(auxiliary_config.get("runtime"), field="auxiliary runtime")
    gates = _mapping(auxiliary_config.get("gates"), field="auxiliary gates")
    if (
        runtime.get("runtime_wired") is not False
        or runtime.get("enabled_default") is not False
        or runtime.get("policy_gate") != 0.0
        or any(value != 0.0 for value in gates.values())
    ):
        raise R303FreezeError("auxiliary target path is not exact-zero and unwired")
    if target_contract.get("observed_trajectory") != (
        "diagnostic_target_only_all_trainable_masks_false_without_exact_r5_sealed_validator"
    ):
        raise R303FreezeError("target config does not enforce all-mask pre-handoff materialization")
    return {
        "schema": R298_RULE_TARGET_SCHEMA,
        "schema_digest": target_digest,
        "manifest": target,
        "auxiliary_config": auxiliary_config,
        "auxiliary_schema": auxiliary_schema,
        "all_trainable_selected_action_masks_false_before_handoff": True,
        "opponent_belief_masked_unavailable": True,
        "policy_feature_eligible": False,
        "runtime_wired": False,
    }


def _checklist_binding(
    *, revision_5_consumer_migration_receipt_path: Path | str
) -> dict[str, Any]:
    """Require the checklist owner's actual pinned migration anchor."""

    try:
        migration = validate_revision_5_consumer_migration_r298(
            revision_5_consumer_migration_receipt_path
        )
        manifest = checklist_provenance_schema_manifest_r298()
    except ChecklistProvenanceError as exc:
        raise R303FreezeError(
            "checklist revision-5 consumer migration anchor is not ready"
        ) from exc
    manifest = dict(_mapping(manifest, field="checklist provenance schema manifest"))
    definition = _mapping(manifest.get("schema_definition"), field="checklist schema definition")
    if (
        definition.get("schema"),
        definition.get("canonical_goal_sha256"),
        definition.get("canonical_contract_sha256"),
        definition.get("q3_bench_only"),
        definition.get("q5_q6_trace_only_exact_zero"),
    ) != (
        R298_CHECKLIST_PROVENANCE_SCHEMA,
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_SHA256,
        True,
        True,
    ):
        raise R303FreezeError("checklist schema lacks rev5 public-information gates")
    anchor = _mapping(
        definition.get("revision_5_consumer_migration_immutable_anchor"),
        field="checklist migration anchor",
    )
    if anchor.get("status") != "pinned_immutable_artifact":
        raise R303FreezeError("checklist migration anchor is not immutable")
    schema_digest = _sha(manifest.get("schema_sha256"), field="checklist schema digest")
    return {
        "schema_manifest": manifest,
        "schema_sha256": schema_digest,
        "migration": migration.to_dict(),
        "strict_q1_q2_q8_disposition": definition.get("strict_mechanics_current_disposition"),
        "q3_bench_only": True,
        "q5_q6_trace_only_exact_zero": True,
        "runtime_wired": False,
    }


def build_r303_frozen_schema_manifest(
    *,
    catalog_binding: Mapping[str, Any],
    revision_5_consumer_migration_receipt_path: Path | str,
) -> dict[str, Any]:
    """Build the canonical census-compatible rev5 schema freeze manifest.

    It is deliberately a builder only.  :func:`write_r303_frozen_schema_manifest_create_only`
    re-runs all bindings immediately before writing, refusing an old in-memory
    result after any owner changes a schema or pinned config.
    """

    _validate_r303_authority()
    catalog = _validate_catalog_binding(catalog_binding)
    checklist = _checklist_binding(
        revision_5_consumer_migration_receipt_path=revision_5_consumer_migration_receipt_path
    )
    target = _target_and_auxiliary_binding()
    adapter = _adapter_config_binding()
    legacy_features = _source_identity(PROJECT_ROOT / "poke_bot" / "features.py", field="legacy features")
    if legacy_features["sha256"] != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise R303FreezeError("legacy features source is not the exact r274 ABI")
    projection = semantic_projection_schema_manifest()
    if (
        projection.get("default_zero_and_inert") is not True
        or projection.get("runtime_wired") is not False
        or _mapping(projection.get("feature_contract"), field="semantic projection feature contract").get(
            "structured_residual_additive_after_or_alongside_card2vec"
        )
        is not True
    ):
        raise R303FreezeError("semantic projection violates the frozen Card2Vec additive boundary")
    predecessor = revision_5_predecessor_classification(
        [
            {
                "receipt_sha256": _sha(
                    _mapping(catalog["catalog_predecessor_evidence"], field="catalog predecessor evidence").get(
                        "sealer_receipt", {}
                    ).get("sha256"),
                    field="catalog predecessor receipt",
                ),
                "schema": R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA,
                "classification": "immutable_predecessor_evidence_not_revision_5_authority",
                "checksum_identical_bytes_reused": True,
                "satisfies_revision_5_schema_freeze": False,
            }
        ]
    )
    feature_schema = {
        "schema": R303_FREEZE_PRODUCER_SCHEMA,
        "legacy_r274_feature_source": legacy_features,
        "public_rule_adapter": adapter,
        "semantic_projection": projection,
        "sealed_catalog_binding_sha256": _canonical_digest(catalog),
        "card2vec": {
            "preserved_untouched": True,
            "new_structured_residual_additive_after_or_alongside": True,
            "legacy_text_hash_exact_mechanics_authority": False,
        },
        "policy_information_set": {
            "opponent_hand_identities": "forbidden",
            "opponent_deck_order": "forbidden",
            "unrevealed_prize_identities": "forbidden",
            "terminal_or_future_transition": "target_only",
        },
    }
    target_schema = {
        "target": target,
        "selected_action_training_state": "diagnostic_only_all_trainable_masks_false_until_sealed_r5_ledger_schema_freeze_and_handoff_validator",
    }
    checklist_schema = checklist["schema_manifest"]
    return {
        "schema": R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "version": 2,
        "status": "frozen_zero_inert_before_refeaturization",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "goal_contract_path": R303_CONTRACT_PATH,
        "exact_goal_contract_sha256": R303_CONTRACT_SHA256,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "rule_derivative_contract_sha256": R303_CONTRACT_SHA256,
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "revision_5_predecessor_classification": predecessor,
        "feature_schema": feature_schema,
        "feature_schema_sha256": _canonical_digest(feature_schema),
        "target_schema": target_schema,
        "target_schema_sha256": target["schema_digest"],
        "checklist_provenance_schema": checklist_schema,
        "checklist_provenance_schema_sha256": checklist["schema_sha256"],
        "sealed_structured_catalog": catalog,
        "sealed_structured_catalog_binding_sha256": _canonical_digest(catalog),
        "canonical_simulator": {
            "linux_x86_64_sha256": CANONICAL_R236_LIBCG_SHA256,
            "linux_x86_64_size_bytes": CANONICAL_R236_LIBCG_SIZE_BYTES,
        },
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "new_branch_inventory": list(_NEW_BRANCHES),
        "new_branch_state": {
            name: {
                "default_gate": 0.0,
                "runtime_wired": False,
                "training_eligible_before_census_and_handoff": False,
            }
            for name in _NEW_BRANCHES
        },
        "selected_action_target_materialization": {
            "records_emit_diagnostic_targets": True,
            "all_trainable_masks_false_without_exact_sealed_r5_validator": True,
            "opponent_belief_masked_unavailable": True,
            "post_handoff_validator_required_for_nonzero_masks": True,
        },
        "checklist_contract": {
            "q3_bench_only": True,
            "q5_q6_trace_only_exact_zero": True,
            "q1_q2_q8_unavailable_zero_without_immutable_owner_pinned_simulator_witness": True,
            "strict_mechanics_current_disposition": checklist[
                "strict_q1_q2_q8_disposition"
            ],
        },
        "default_zero_and_inert_attestation": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "runtime_wired": False,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def validate_r303_frozen_schema_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the richer r303 fields in addition to the census gate ABI."""

    row = _mapping(manifest, field="r303 frozen schema manifest")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_contract_path"),
        row.get("exact_goal_contract_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("rule_derivative_gateway_sha256"),
        row.get("rule_derivative_contract_sha256"),
    ) != (
        R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "frozen_zero_inert_before_refeaturization",
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_SHA256,
    ):
        raise R303FreezeError("r303 frozen schema manifest authority drifted")
    validate_revision_5_predecessor_classification(
        _mapping(row.get("revision_5_predecessor_classification"), field="predecessor classification")
    )
    branches = _rows(row.get("new_branch_inventory"), field="r303 branch inventory")
    if branches != list(_NEW_BRANCHES):
        raise R303FreezeError("r303 new branch inventory drifted")
    for field in (
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    ):
        _sha(row.get(field), field=f"r303 frozen schema {field}")
    if row.get("target_schema_sha256") != _mapping(
        _mapping(row.get("target_schema"), field="target schema").get("target"), field="target binding"
    ).get("schema_digest"):
        raise R303FreezeError("r303 target schema digest does not match the target binding")
    policy = _mapping(row.get("selected_action_target_materialization"), field="selected-action materialization")
    if (
        policy.get("records_emit_diagnostic_targets") is not True
        or policy.get("all_trainable_masks_false_without_exact_sealed_r5_validator") is not True
        or policy.get("opponent_belief_masked_unavailable") is not True
        or policy.get("post_handoff_validator_required_for_nonzero_masks") is not True
    ):
        raise R303FreezeError("frozen schema permits selected-action supervision too early")
    checklist = _mapping(row.get("checklist_contract"), field="checklist contract")
    if (
        checklist.get("q3_bench_only") is not True
        or checklist.get("q5_q6_trace_only_exact_zero") is not True
        or checklist.get("q1_q2_q8_unavailable_zero_without_immutable_owner_pinned_simulator_witness")
        is not True
    ):
        raise R303FreezeError("frozen schema weakens checklist safety gates")
    catalog = _validate_catalog_binding(
        _mapping(row.get("sealed_structured_catalog"), field="sealed structured catalog")
    )
    if row.get("sealed_structured_catalog_binding_sha256") != _canonical_digest(catalog):
        raise R303FreezeError("frozen schema catalog binding digest drifted")
    runtime = _mapping(row.get("runtime_authority"), field="runtime authority")
    if runtime != {
        "elmo_only": True,
        "create_only": True,
        "training_allowed": False,
        "inzi_authority": False,
        "production_authority": False,
    }:
        raise R303FreezeError("frozen schema has expanded runtime authority")
    if (
        row.get("default_zero_and_inert_attestation") is not True
        or row.get("layer_off_bit_identical_baseline_logits") is not True
        or row.get("layer_off_identical_legal_choice") is not True
        or row.get("runtime_wired") is not False
    ):
        raise R303FreezeError("frozen schema does not preserve exact layer-off safety")
    return dict(row)


def write_r303_frozen_schema_manifest_create_only(
    path: Path | str,
    manifest: Mapping[str, Any],
    *,
    catalog_sealer_receipt_path: Path | str,
    revision_5_consumer_migration_receipt_path: Path | str,
) -> str:
    """Rebuild and compare immediately before create-only schema publication."""

    row = validate_r303_frozen_schema_manifest(manifest)
    live_catalog = build_r303_sealed_catalog_binding(
        catalog_sealer_receipt_path=catalog_sealer_receipt_path,
        revision_5_consumer_migration_receipt_path=revision_5_consumer_migration_receipt_path,
    )
    current = build_r303_frozen_schema_manifest(
        catalog_binding=live_catalog,
        revision_5_consumer_migration_receipt_path=revision_5_consumer_migration_receipt_path,
    )
    if _canonical_digest(row) != _canonical_digest(current):
        raise R303FreezeError("frozen schema manifest became stale before create-only write")
    return _write_create_only_json(Path(path), row)


def _byte_identity(value: Any) -> str:
    """Hash observed parity data without coercing floating-point bytes."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            header = canonical_json_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
            return _SHA256_PREFIX + hashlib.sha256(header + b"\0" + raw).hexdigest()
    except ModuleNotFoundError:
        pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _SHA256_PREFIX + hashlib.sha256(bytes(value)).hexdigest()
    return _canonical_digest(value)


def _checkpoint_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _mapping(value, field="baseline checkpoint identity")
    expected = {"path", "sha256", "size_bytes"}
    if set(row) != expected:
        raise R303FreezeError("baseline checkpoint identity field inventory drifted")
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise R303FreezeError("baseline checkpoint identity path is malformed")
    return {
        "path": path,
        "sha256": _sha(row.get("sha256"), field="baseline checkpoint sha256"),
        "size_bytes": _positive_int(row.get("size_bytes"), field="baseline checkpoint size"),
    }


def make_r303_zero_bypass_receipt(
    *,
    schema_manifest: Mapping[str, Any],
    baseline_checkpoint_identity: Mapping[str, Any],
    baseline_logits: Any,
    layer_off_logits: Any,
    baseline_legal_choice: Any,
    layer_off_legal_choice: Any,
    evidence_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Make an exact-layer-off receipt for the rev5 schema manifest.

    A mismatch returns a failed receipt rather than claiming a freeze.  The
    caller must not write a failed receipt as the canonical schema gate.
    """

    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    checkpoint = _checkpoint_identity(baseline_checkpoint_identity)
    evidence = dict(_mapping(evidence_provenance, field="zero-bypass evidence provenance"))
    baseline_identity = _byte_identity(baseline_logits)
    layer_off_identity = _byte_identity(layer_off_logits)
    same_logits = baseline_identity == layer_off_identity
    same_choice = baseline_legal_choice == layer_off_legal_choice
    status = "passed_exact_baseline_logits" if same_logits and same_choice else "failed_parity"
    return {
        "schema": R298_ZERO_BYPASS_RECEIPT_SCHEMA,
        "version": 2,
        "status": status,
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "rule_derivative_contract_sha256": R303_CONTRACT_SHA256,
        "revision_5_predecessor_classification": manifest["revision_5_predecessor_classification"],
        "frozen_schema_manifest_sha256": _canonical_digest(manifest),
        "baseline_r274_checkpoint_identity": checkpoint,
        "baseline_logits_byte_identity_sha256": baseline_identity,
        "layer_off_logits_byte_identity_sha256": layer_off_identity,
        "layer_off_bit_identical_baseline_logits": same_logits,
        "baseline_legal_choice": baseline_legal_choice,
        "layer_off_legal_choice": layer_off_legal_choice,
        "layer_off_identical_legal_choice": same_choice,
        "all_new_routes_exact_zero_or_bypassed": True,
        "checklist_provenance_schema_frozen": True,
        "selected_action_targets_all_trainable_masks_false": True,
        "card2vec_unchanged_and_structured_residual_additive": True,
        "runtime_wired": False,
        "evidence_provenance": evidence,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def validate_r303_zero_bypass_receipt(
    schema_manifest: Mapping[str, Any], zero_bypass_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate extra r303 parity evidence and the shared census gate ABI."""

    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    receipt = _mapping(zero_bypass_receipt, field="r303 zero-bypass receipt")
    try:
        validate_frozen_schema_gate(manifest, receipt)
    except Exception as exc:
        raise R303FreezeError("canonical frozen-schema/zero-bypass gate failed") from exc
    checkpoint = _checkpoint_identity(
        _mapping(receipt.get("baseline_r274_checkpoint_identity"), field="zero-bypass checkpoint")
    )
    if receipt.get("baseline_logits_byte_identity_sha256") != receipt.get(
        "layer_off_logits_byte_identity_sha256"
    ):
        raise R303FreezeError("zero-bypass logit byte identities differ")
    for field, expected in (
        ("selected_action_targets_all_trainable_masks_false", True),
        ("card2vec_unchanged_and_structured_residual_additive", True),
        ("runtime_wired", False),
    ):
        if receipt.get(field) is not expected:
            raise R303FreezeError(f"zero-bypass receipt {field} is unsafe")
    runtime = _mapping(receipt.get("runtime_authority"), field="zero-bypass runtime authority")
    if runtime != {
        "elmo_only": True,
        "create_only": True,
        "training_allowed": False,
        "inzi_authority": False,
        "production_authority": False,
    }:
        raise R303FreezeError("zero-bypass receipt has expanded runtime authority")
    result = dict(receipt)
    result["baseline_r274_checkpoint_identity"] = checkpoint
    return result


def write_r303_zero_bypass_receipt_create_only(
    path: Path | str,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> str:
    """Write only a passing, current zero-bypass receipt once."""

    row = validate_r303_zero_bypass_receipt(schema_manifest, zero_bypass_receipt)
    if row.get("status") != "passed_exact_baseline_logits":
        raise R303FreezeError("cannot publish a failed zero-bypass receipt")
    return _write_create_only_json(Path(path), row)


def build_r303_schema_freeze_receipt(
    *,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Build the exact contract schema-freeze receipt for later r303 readers."""

    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    bypass = validate_r303_zero_bypass_receipt(manifest, zero_bypass_receipt)
    timestamp = _utc_text(frozen_at_utc, field="schema freeze timestamp")
    catalog = _validate_catalog_binding(
        _mapping(manifest.get("sealed_structured_catalog"), field="sealed structured catalog")
    )
    receipt = {
        "schema": R303_SCHEMA_FREEZE_RECEIPT_SCHEMA,
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "feature_schema_id": str(_mapping(manifest.get("feature_schema"), field="feature schema").get("schema")),
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "target_schema_id": R298_RULE_TARGET_SCHEMA,
        "target_schema_sha256": manifest["target_schema_sha256"],
        "checklist_provenance_schema_id": R298_CHECKLIST_PROVENANCE_SCHEMA,
        "checklist_provenance_schema_sha256": manifest["checklist_provenance_schema_sha256"],
        "public_catalog_manifest_sha256": catalog["catalog_file_sha256"],
        "canonical_simulator_sha256": CANONICAL_R236_LIBCG_SHA256,
        "new_branch_inventory": list(_NEW_BRANCHES),
        "census_supported_branch_inventory": [],
        "unsupported_zero_inert_branch_inventory": list(_NEW_BRANCHES),
        "q3_bench_only": True,
        "q5_q6_trace_only_zero": True,
        "public_information_contract_passed": True,
        "zero_bypass_receipt_sha256": _canonical_digest(bypass),
        "layer_off_bit_identical_baseline_logits": True,
        "frozen_at_utc": timestamp,
    }
    try:
        validate_handoff_receipt("schema_freeze", receipt)
    except Exception as exc:
        raise R303FreezeError("schema-freeze receipt does not satisfy the rev5 contract") from exc
    return receipt


def validate_r303_schema_freeze_receipt(
    *,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-link the contract receipt to its schema and zero-bypass bytes."""

    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    bypass = validate_r303_zero_bypass_receipt(manifest, zero_bypass_receipt)
    receipt = _mapping(schema_freeze_receipt, field="r303 schema-freeze receipt")
    try:
        checked = validate_handoff_receipt("schema_freeze", receipt)
    except Exception as exc:
        raise R303FreezeError("schema-freeze receipt is not contract-valid") from exc
    expected = {
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "target_schema_sha256": manifest["target_schema_sha256"],
        "checklist_provenance_schema_sha256": manifest["checklist_provenance_schema_sha256"],
        "public_catalog_manifest_sha256": _mapping(
            manifest["sealed_structured_catalog"], field="sealed catalog"
        )["catalog_file_sha256"],
        "canonical_simulator_sha256": CANONICAL_R236_LIBCG_SHA256,
        "zero_bypass_receipt_sha256": _canonical_digest(bypass),
    }
    if any(checked.get(field) != value for field, value in expected.items()):
        raise R303FreezeError("schema-freeze receipt is not cross-linked to current freeze evidence")
    if (
        checked.get("new_branch_inventory") != list(_NEW_BRANCHES)
        or checked.get("census_supported_branch_inventory") != []
        or checked.get("unsupported_zero_inert_branch_inventory") != list(_NEW_BRANCHES)
    ):
        raise R303FreezeError("schema-freeze receipt claims unsupported branch eligibility")
    return dict(checked)


def write_r303_schema_freeze_receipt_create_only(
    path: Path | str,
    *,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
) -> str:
    """Persist the contract schema-freeze receipt once, after cross-linking."""

    row = validate_r303_schema_freeze_receipt(
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
        schema_freeze_receipt=schema_freeze_receipt,
    )
    return _write_create_only_json(Path(path), row)


def build_r303_schema_freeze_bundle(
    *,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a non-authorizing index for the three immutable freeze objects."""

    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    bypass = validate_r303_zero_bypass_receipt(manifest, zero_bypass_receipt)
    receipt = validate_r303_schema_freeze_receipt(
        schema_manifest=manifest,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=schema_freeze_receipt,
    )
    return {
        "schema": R303_SCHEMA_FREEZE_BUNDLE_SCHEMA,
        "status": "frozen_zero_inert_ready_for_elmo_refeaturization_only",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "frozen_schema_manifest_sha256": _canonical_digest(manifest),
        "zero_bypass_receipt_sha256": _canonical_digest(bypass),
        "schema_freeze_receipt_sha256": _canonical_digest(receipt),
        "selected_action_target_mode": "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def validate_r303_schema_freeze_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a read-only index of the three rev5 freeze artifacts."""

    row = _mapping(bundle, field="r303 schema-freeze bundle")
    expected_fields = {
        "schema",
        "status",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "schema_freeze_receipt_sha256",
        "selected_action_target_mode",
        "runtime_authority",
    }
    if set(row) != expected_fields:
        raise R303FreezeError("schema-freeze bundle field inventory drifted")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("selected_action_target_mode"),
    ) != (
        R303_SCHEMA_FREEZE_BUNDLE_SCHEMA,
        "frozen_zero_inert_ready_for_elmo_refeaturization_only",
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
    ):
        raise R303FreezeError("schema-freeze bundle authority or target safety drifted")
    for field in (
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "schema_freeze_receipt_sha256",
    ):
        _sha(row.get(field), field=f"schema-freeze bundle.{field}")
    if _mapping(row.get("runtime_authority"), field="schema-freeze bundle authority") != {
        "elmo_only": True,
        "create_only": True,
        "training_allowed": False,
        "inzi_authority": False,
        "production_authority": False,
    }:
        raise R303FreezeError("schema-freeze bundle expands runtime authority")
    return dict(row)


def write_r303_schema_freeze_bundle_create_only(
    path: Path | str, bundle: Mapping[str, Any]
) -> str:
    """Persist only a structurally safe, non-authorizing freeze index."""

    return _write_create_only_json(
        Path(path), validate_r303_schema_freeze_bundle(bundle)
    )


def build_r303_materialization_preflight(
    *,
    raw_manifest: Mapping[str, Any],
    raw_corpus_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate prerequisites for the exact 30-day, diagnostic-only pass.

    This authorizes no training.  It is the handoff from freeze tooling to a
    future bounded Elmo record writer and makes the all-masked selected-action
    target condition explicit rather than relying on a caller default.
    """

    try:
        validate_raw_corpus_manifest(raw_manifest)
    except Exception as exc:
        raise R303FreezeError("raw corpus is not the exact validated 30-day manifest") from exc
    raw_receipt = _mapping(raw_corpus_receipt, field="raw corpus receipt")
    if (
        raw_receipt.get("schema"),
        raw_receipt.get("status"),
        raw_receipt.get("goal_revision"),
        raw_receipt.get("root_handoff_revision"),
        raw_receipt.get("rule_derivative_gateway_sha256"),
        raw_receipt.get("rule_derivative_contract_sha256"),
        raw_receipt.get("raw_expert_corpus_manifest_sha256"),
    ) != (
        R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "passed",
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        R303_GOAL_GATEWAY_SHA256,
        R303_CONTRACT_SHA256,
        _canonical_digest(raw_manifest),
    ):
        raise R303FreezeError("raw corpus receipt does not bind the exact r5 30-day manifest")
    manifest = validate_r303_frozen_schema_manifest(schema_manifest)
    bypass = validate_r303_zero_bypass_receipt(manifest, zero_bypass_receipt)
    freeze_receipt = validate_r303_schema_freeze_receipt(
        schema_manifest=manifest,
        zero_bypass_receipt=bypass,
        schema_freeze_receipt=schema_freeze_receipt,
    )
    return {
        "schema": R303_MATERIALIZATION_PREFLIGHT_SCHEMA,
        "status": "ready_for_exact_30_day_elmo_refeaturization_diagnostic_targets_only",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "raw_expert_corpus_manifest_sha256": _canonical_digest(raw_manifest),
        "raw_expert_corpus_receipt_sha256": _canonical_digest(raw_receipt),
        "frozen_schema_manifest_sha256": _canonical_digest(manifest),
        "zero_bypass_receipt_sha256": _canonical_digest(bypass),
        "schema_freeze_receipt_sha256": _canonical_digest(freeze_receipt),
        "utc_window": ["2026-07-13", "2026-08-11"],
        "utc_partition_count": 30,
        "required_record_surfaces": list(_MATERIALIZED_RECORD_SURFACES),
        "selected_action_target_mode": "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "opponent_belief_target_mode": "unavailable_masked_no_loss",
        "source_day_group_seed_disjoint_split_manifest_required": True,
        "physical_shard_max_bytes": 96 * 1024 * 1024,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def validate_r303_materialization_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a preflight that quietly broadens the diagnostic-only pass."""

    row = _mapping(preflight, field="r303 materialization preflight")
    expected_fields = {
        "schema",
        "status",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "schema_freeze_receipt_sha256",
        "utc_window",
        "utc_partition_count",
        "required_record_surfaces",
        "selected_action_target_mode",
        "opponent_belief_target_mode",
        "source_day_group_seed_disjoint_split_manifest_required",
        "physical_shard_max_bytes",
        "runtime_authority",
    }
    if set(row) != expected_fields:
        raise R303FreezeError("materialization preflight field inventory drifted")
    if (
        row.get("schema"),
        row.get("status"),
        row.get("goal_contract_path"),
        row.get("goal_contract_sha256"),
        row.get("goal_revision"),
        row.get("root_handoff_revision"),
        row.get("utc_window"),
        row.get("utc_partition_count"),
        row.get("selected_action_target_mode"),
        row.get("opponent_belief_target_mode"),
        row.get("source_day_group_seed_disjoint_split_manifest_required"),
        row.get("physical_shard_max_bytes"),
    ) != (
        R303_MATERIALIZATION_PREFLIGHT_SCHEMA,
        "ready_for_exact_30_day_elmo_refeaturization_diagnostic_targets_only",
        R303_CONTRACT_PATH,
        R303_CONTRACT_SHA256,
        R303_GOAL_REVISION,
        R303_ROOT_OWNER_REVISION,
        ["2026-07-13", "2026-08-11"],
        30,
        "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "unavailable_masked_no_loss",
        True,
        96 * 1024 * 1024,
    ):
        raise R303FreezeError("materialization preflight violates rev5 boundaries")
    if list(row.get("required_record_surfaces", ())) != list(_MATERIALIZED_RECORD_SURFACES):
        raise R303FreezeError("materialization preflight record surface drifted")
    for field in (
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "schema_freeze_receipt_sha256",
    ):
        _sha(row.get(field), field=f"materialization preflight.{field}")
    if _mapping(row.get("runtime_authority"), field="materialization preflight authority") != {
        "elmo_only": True,
        "create_only": True,
        "training_allowed": False,
        "inzi_authority": False,
        "production_authority": False,
    }:
        raise R303FreezeError("materialization preflight expands runtime authority")
    return dict(row)


def write_r303_materialization_preflight_create_only(
    path: Path | str, preflight: Mapping[str, Any]
) -> str:
    """Persist a validated preflight receipt without starting materialization."""

    return _write_create_only_json(
        Path(path), validate_r303_materialization_preflight(preflight)
    )


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically create one canonical JSON artifact without overwrite rights."""

    if path.is_symlink() or path.exists():
        raise R303FreezeError(f"create-only artifact already exists: {path}")
    parent = path.parent
    if parent.is_symlink():
        raise R303FreezeError("create-only artifact parent must not be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise R303FreezeError("create-only artifact parent is not a directory")
    encoded = canonical_json_bytes(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise R303FreezeError(f"create-only artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    actual = _sha256_file(path)
    if actual != _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest():
        raise R303FreezeError("create-only artifact changed while being written")
    return actual


__all__ = [
    "R303_FREEZE_PRODUCER_SCHEMA",
    "R303_MATERIALIZATION_PREFLIGHT_SCHEMA",
    "R303_SCHEMA_FREEZE_BUNDLE_SCHEMA",
    "R303_SCHEMA_FREEZE_RECEIPT_SCHEMA",
    "R303_SEALED_CATALOG_BINDING_SCHEMA",
    "R303FreezeError",
    "build_r303_frozen_schema_manifest",
    "build_r303_materialization_preflight",
    "build_r303_schema_freeze_bundle",
    "build_r303_schema_freeze_receipt",
    "build_r303_sealed_catalog_binding",
    "build_revision_5_consumer_migration_payload",
    "make_r303_zero_bypass_receipt",
    "validate_r303_frozen_schema_manifest",
    "validate_r303_materialization_preflight",
    "validate_r303_schema_freeze_receipt",
    "validate_r303_schema_freeze_bundle",
    "validate_r303_zero_bypass_receipt",
    "validate_revision_5_consumer_migration_payload_for_issue",
    "write_r303_frozen_schema_manifest_create_only",
    "write_r303_materialization_preflight_create_only",
    "write_r303_schema_freeze_receipt_create_only",
    "write_r303_schema_freeze_bundle_create_only",
    "write_r303_zero_bypass_receipt_create_only",
    "write_revision_5_consumer_migration_create_only",
]
