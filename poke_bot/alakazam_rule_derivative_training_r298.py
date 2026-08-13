"""Create-only materialization and frozen-derivative guards for r298.

This module deliberately sits beside, rather than inside, the r195/r241/r274
policy.  It has no runtime import from ``model.py`` or ``agent.py`` and never
loads, rewrites, selects, or publishes a parent checkpoint.  Its jobs are
limited to the isolated Elmo study:

* freeze and identify the new public-rule/target/checklist schema before the
  thirty-day pass;
* materialize public-only, re-featurized raw records from the sealed thirty
  UTC-day archive;
* make source/day/group/seed-disjoint split manifests; and
* make it mechanically impossible to describe a training/candidate attempt as
  authorized before the schema, corpus, census, and zero-bypass receipts pass.

The source corpus is intentionally not the historic r274 20-day pack.  A
caller that presents it (or any subset) gets a hard failure rather than a
helpful-looking partial dataset.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import math
import os
import resource
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from .alakazam_collision_census_r298 import (
    CANONICAL_R236_LIBCG_SHA256,
    CANONICAL_R236_LIBCG_SIZE_BYTES,
    EXACT_NEW_LIST_MULTISET_SHA256,
    R274_EXACT_FEATURES_SOURCE_SHA256,
    R298_COLLISION_RECEIPT_SCHEMA,
    R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
    R298_RAW_CORPUS_MANIFEST_SCHEMA,
    R298_RAW_CORPUS_RECEIPT_SCHEMA,
    R298_ZERO_BYPASS_RECEIPT_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    current_feature_token_hashes,
    recorded_episode_frame_coverage,
    require_sha256,
    stage_descriptors_from_recorded_episode,
    validate_frozen_schema_gate,
    validate_raw_corpus_manifest,
)
from .alakazam_public_rule_adapter_r298 import (
    PUBLIC_RULE_ADAPTER_SCHEMA,
    PUBLIC_RULE_REPRESENTATION_SCHEMA,
    build_public_rule_representation,
    sanitize_public_observation,
)
from .alakazam_simulator_rule_targets_r298 import (
    R298_RULE_TARGET_SCHEMA,
    compile_simulator_rule_targets,
)
from .alakazam_checklist_provenance_r298 import (
    R298_CHECKLIST_PROVENANCE_SCHEMA as CANONICAL_R298_CHECKLIST_PROVENANCE_SCHEMA,
    checklist_provenance_schema_manifest_r298,
    filter_r288_checklist_trace_r298,
)


R298_REVISION: Final = 298
R298_GOAL_PATH: Final = "goals/alakazam-elmo-rule-derivative/GOAL.md"
R298_CONTRACT_PATH: Final = "goals/alakazam-elmo-rule-derivative/contract.json"
R298_GOAL_SHA256: Final = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
R298_CONTRACT_SHA256: Final = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
R298_OWNER_ATTACHMENT_SHA256: Final = (
    "sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f"
)
R298_MECHANICS_ATTACHMENT_SHA256: Final = (
    "sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb"
)

R298_REFEATURE_RECORD_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_refeatured_record/v1"
)
R298_REFEATURE_MANIFEST_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_refeature_manifest/v1"
)
R298_REFEATURE_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_refeature_receipt/v1"
)
R298_SPLIT_MANIFEST_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_disjoint_split_manifest/v1"
)
# Alias the dedicated checklist owner's schema; do not fork a second local
# checklist ABI in the materializer.
R298_CHECKLIST_PROVENANCE_SCHEMA: Final = CANONICAL_R298_CHECKLIST_PROVENANCE_SCHEMA
R298_STRICT_MECHANICS_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_pinned_public_mechanics/v1"
)
R298_TRAINING_PLAN_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_training_plan/v1"
)
R298_CANDIDATE_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_candidate_receipt/v1"
)
R298_SEALED_CATALOG_BINDING_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_sealed_catalog_binding/v1"
)
R298_TENSOR_AUDIT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_frozen_tensor_audit/v1"
)
R298_CARD2VEC_EVIDENCE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_card2vec_non_regression_receipt/v1"
)
R298_TRAINING_GATE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_training_gate_report/v1"
)
R298_BRANCH_DESCRIPTOR_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_branch_descriptor/v1"
)

R298_RAW_WINDOW_START: Final = "2026-07-13"
R298_RAW_WINDOW_END: Final = "2026-08-11"
R298_RAW_DAY_COUNT: Final = 30
R298_EXPERIMENT_RAM_BYTES: Final = 96 * 1024**3

# The canonical goal deliberately parameterizes the actual r274 checkpoint at
# receipt/launch time.  A historical Elmo path or digest is useful operator
# context, but it must never become an unowned hard-coded baseline substitute.

CHECKLIST_CHANNELS: Final = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)
STRICT_PINNED_RULE_CHANNELS: Final = frozenset(
    {
        "ko_hand_threshold",
        "safe_spend_above_threshold",
        "terminal_before_forced_draw",
    }
)
TRACE_ONLY_ZERO_CHANNELS: Final = frozenset(
    {"bench_prize_exposure", "immediate_disruption_outcome"}
)


class RuleDerivativeTrainingError(RuntimeError):
    """An isolated r298 artifact lacks evidence required by the contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strict_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuleDerivativeTrainingError(f"{field} must be an object")
    return value


def _strict_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise RuleDerivativeTrainingError(f"{field} must be a list")
    return list(value)


def _exact_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleDerivativeTrainingError(f"{field} must be an exact integer")
    if minimum is not None and value < minimum:
        raise RuleDerivativeTrainingError(f"{field} is below {minimum}")
    return value


def _utc_days() -> tuple[str, ...]:
    start = date.fromisoformat(R298_RAW_WINDOW_START)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(R298_RAW_DAY_COUNT))


def _require_sha(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except Exception as exc:
        raise RuleDerivativeTrainingError(str(exc)) from exc


def _public_digest(value: Any) -> str:
    """Content-address a public/provenance value without stringifying objects."""

    try:
        # Cross-owner r298 receipts use the collision producer's canonical
        # UTF-8/no-ASCII-escape/newline encoding.  Do not silently use this
        # module's older internal JSON helper for a schema/raw/receipt binding.
        return canonical_sha256(value)
    except Exception as exc:
        raise RuleDerivativeTrainingError("value is not canonical JSON") from exc


def _assert_contract_identity(
    manifest: Mapping[str, Any],
    *,
    field: str = "schema manifest",
) -> None:
    contract_identity = manifest.get(
        "rule_derivative_contract_sha256", manifest.get("goal_contract_sha256")
    )
    if contract_identity != R298_CONTRACT_SHA256:
        raise RuleDerivativeTrainingError(f"{field} binds a stale or foreign contract")
    goal_identity = manifest.get(
        "rule_derivative_goal_sha256", manifest.get("rule_derivative_gateway_sha256")
    )
    if goal_identity != R298_GOAL_SHA256:
        raise RuleDerivativeTrainingError(f"{field} binds a stale or foreign goal gateway")


def _module_source_identity(module_file: str | Path) -> dict[str, Any]:
    path = Path(module_file).resolve()
    if not path.is_file():
        raise RuleDerivativeTrainingError(f"schema source is unavailable: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _target_schema_manifest() -> Mapping[str, Any]:
    """Ask the target owner for its own canonical manifest rather than copy it."""

    try:
        from .alakazam_simulator_rule_targets_r298 import r298_rule_target_schema_manifest

        result = r298_rule_target_schema_manifest()
    except (ImportError, AttributeError) as exc:
        raise RuleDerivativeTrainingError(
            "r298 selected-action target schema manifest is unavailable"
        ) from exc
    return _strict_mapping(result, field="r298 target schema manifest")


def _target_schema_digest() -> str:
    target = _target_schema_manifest()
    digest = target.get(
        "schema_sha256", target.get("schema_digest", target.get("digest"))
    )
    if isinstance(digest, str) and digest.startswith("sha256:"):
        return _require_sha(digest, field="target schema digest")
    return _public_digest(target)


def checklist_provenance_schema() -> dict[str, Any]:
    """Return the checklist owner's checksum-bindable canonical manifest."""

    try:
        manifest = checklist_provenance_schema_manifest_r298()
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            "canonical r298 checklist provenance schema is unavailable"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise RuleDerivativeTrainingError("canonical r298 checklist schema is malformed")
    definition = _strict_mapping(manifest.get("schema_definition"), field="checklist schema definition")
    if definition.get("schema") != R298_CHECKLIST_PROVENANCE_SCHEMA:
        raise RuleDerivativeTrainingError("canonical r298 checklist schema drifted")
    if tuple(definition.get("channels", ())) != CHECKLIST_CHANNELS:
        raise RuleDerivativeTrainingError("canonical r298 checklist channel order drifted")
    if definition.get("q3_bench_only") is not True or definition.get("q5_q6_trace_only_exact_zero") is not True:
        raise RuleDerivativeTrainingError("canonical r298 checklist safety gates drifted")
    return dict(manifest)


def _checklist_provenance_schema_digest(manifest: Mapping[str, Any]) -> str:
    digest = manifest.get("schema_sha256")
    if not isinstance(digest, str):
        raise RuleDerivativeTrainingError("canonical r298 checklist schema digest is missing")
    return _require_sha(digest, field="canonical checklist provenance schema digest")


def build_frozen_schema_manifest(
    *, catalog_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build the immutable schema description required before re-featurizing.

    The caller must still write this object create-only.  The manifest does
    not claim a benchmark or a trained candidate; it only identifies the
    default-off code/data surface the census is allowed to inspect.
    """

    from . import alakazam_public_rule_adapter_r298 as public_adapter
    from . import alakazam_rule_aux_heads_r298 as aux_heads
    from . import alakazam_rule_derivative_model_r298 as semantic_projection
    from . import alakazam_simulator_rule_targets_r298 as rule_targets

    feature_sources = {
        "legacy_r274_features": _module_source_identity(
            Path(__file__).with_name("features.py")
        ),
        "public_rule_adapter": _module_source_identity(public_adapter.__file__),
        "semantic_projection": _module_source_identity(semantic_projection.__file__),
        "repaired_auxiliary_heads": _module_source_identity(aux_heads.__file__),
    }
    if feature_sources["legacy_r274_features"]["sha256"] != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise RuleDerivativeTrainingError("legacy feature source does not match exact r274 ABI")
    target_manifest = _target_schema_manifest()
    try:
        # The target source digest alone is not enough: its companion
        # auxiliary-head config owns the frozen static target/aux schema
        # binding.  Fail schema freeze if that owner has not atomically
        # regenerated its config after a target change.
        aux_config = aux_heads.load_r298_aux_heads_config()
        rule_targets.assert_r298_rule_target_schema_binding(
            target_manifest,
            require_current_goal_files=True,
        )
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            "r298 repaired auxiliary-head/selected-target schema binding is not current"
        ) from exc
    aux_target = _strict_mapping(aux_config.get("target_contract"), field="aux target contract")
    if aux_target.get("schema_digest") != _target_schema_digest():
        raise RuleDerivativeTrainingError("auxiliary-head config does not bind the current target schema")
    aux_static = _strict_mapping(
        aux_config.get("frozen_schema_manifest"), field="aux static schema binding"
    )
    static_path_value = aux_static.get("path")
    if not isinstance(static_path_value, str):
        raise RuleDerivativeTrainingError("auxiliary-head static schema path is missing")
    static_path = Path(__file__).resolve().parents[1] / static_path_value
    static_identity = _module_source_identity(static_path)
    if aux_static.get("sha256") != static_identity["sha256"]:
        raise RuleDerivativeTrainingError("auxiliary-head static schema digest is stale")
    checklist_schema = checklist_provenance_schema()
    feature_layout = {
        "schema": R298_REFEATURE_RECORD_SCHEMA,
        "legacy_feature_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
        "public_representation_schema": PUBLIC_RULE_REPRESENTATION_SCHEMA,
        "public_adapter_schema": PUBLIC_RULE_ADAPTER_SCHEMA,
        "semantic_projection": semantic_projection.semantic_projection_schema_manifest(),
        "sources": feature_sources,
    }
    catalog_surface: dict[str, Any]
    if catalog_binding is None:
        catalog_surface = {
            "status": "catalog_binding_required_before_refeaturization",
            "binding": None,
        }
    else:
        binding = _strict_mapping(catalog_binding, field="sealed catalog binding")
        if binding.get("schema") != R298_SEALED_CATALOG_BINDING_SCHEMA:
            raise RuleDerivativeTrainingError("frozen schema catalog binding schema drifted")
        if binding.get("pinned_libcg_sha256") != CANONICAL_R236_LIBCG_SHA256:
            raise RuleDerivativeTrainingError("frozen schema catalog binding libcg drifted")
        if binding.get("card2vec_preserved_unchanged") is not True or binding.get("csv_or_text_hash_mechanics_authority") is not False:
            raise RuleDerivativeTrainingError("frozen schema catalog binding violates Card2Vec/mechanics boundary")
        if binding.get("adapter_catalog_kind") != "sealed_public_catalog" or binding.get("adapter_catalog_eligible") is not True:
            raise RuleDerivativeTrainingError("frozen schema catalog binding is not adapter-sealed/eligible")
        catalog_surface = {"status": "sealed_catalog_bound", "binding": dict(binding)}
    return {
        "schema": R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "version": 1,
        "status": "frozen_zero_inert_before_refeaturization",
        "revision": R298_REVISION,
        "goal_contract_path": R298_CONTRACT_PATH,
        "exact_goal_contract_sha256": R298_CONTRACT_SHA256,
        "rule_derivative_contract_sha256": R298_CONTRACT_SHA256,
        "rule_derivative_goal_sha256": R298_GOAL_SHA256,
        "owner_goal_sha256": R298_OWNER_ATTACHMENT_SHA256,
        "mechanics_attachment_sha256": R298_MECHANICS_ATTACHMENT_SHA256,
        "feature_schema": feature_layout,
        "feature_schema_sha256": _public_digest(feature_layout),
        "target_schema": dict(target_manifest),
        "target_schema_sha256": _target_schema_digest(),
        "repaired_auxiliary_heads_config": dict(aux_config),
        "repaired_auxiliary_heads_static_schema": static_identity,
        "checklist_provenance_schema": checklist_schema,
        "checklist_provenance_schema_sha256": _checklist_provenance_schema_digest(
            checklist_schema
        ),
        "sealed_structured_catalog": catalog_surface,
        "sealed_structured_catalog_binding_sha256": (
            None if catalog_binding is None else _public_digest(catalog_surface["binding"])
        ),
        # The collision producer deliberately consumes an ordered list of
        # stable names.  Keep richer metadata in the separately typed
        # descriptors field instead of changing that cross-owner ABI.
        "new_branch_inventory": [
            "public_rule_semantic_projection",
            "public_rule_metadata_residual",
            "r298_repaired_auxiliary_heads",
            "eight_checklist_provenance_gates",
        ],
        "new_branch_descriptor_schema": R298_BRANCH_DESCRIPTOR_SCHEMA,
        "new_branch_descriptors": [
            {
                "name": "public_rule_semantic_projection",
                "trainable_after_census_only": True,
                "default_gate": 0.0,
                "runtime_wired": False,
            },
            {
                "name": "public_rule_metadata_residual",
                "trainable_after_census_only": True,
                "default_gate": 0.0,
                "runtime_wired": False,
            },
            {
                "name": "r298_repaired_auxiliary_heads",
                "trainable_after_census_only": True,
                "policy_gate": 0.0,
                "runtime_wired": False,
            },
            {
                "name": "eight_checklist_provenance_gates",
                "trainable_after_census_only": True,
                "q5_q6_exact_zero": True,
                "runtime_wired": False,
            },
        ],
        "default_zero_and_inert_attestation": True,
        "default_zero_and_inert_details": {
            "all_new_routes_default_zero_or_complete_bypass": True,
            "q5_q6_trace_only_exact_zero": True,
            "candidate_training_before_census": False,
            "production_or_inzi_wiring": False,
        },
        # These are a static implementation assertion.  The separately
        # checksum-bound zero-bypass receipt below is still required to prove
        # the assertion against receipt-bound baseline logits/legal choices.
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "rule_derivative_gateway_sha256": R298_GOAL_SHA256,
        "runtime_wired": False,
        "canonical_simulator": {
            "linux_x86_64_sha256": CANONICAL_R236_LIBCG_SHA256,
            "linux_x86_64_size_bytes": CANONICAL_R236_LIBCG_SIZE_BYTES,
        },
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "production_authority": False,
            "inzi_authority": False,
        },
    }


def write_frozen_schema_manifest_create_only(
    path: Path | str, manifest: Mapping[str, Any]
) -> str:
    """Persist a checksum-current schema freeze exactly once.

    The manifest is regenerated immediately before the write and compared to
    the caller's payload.  This refuses an old in-memory manifest if a target,
    checklist, or auxiliary-head owner changed its checked-in schema while an
    operator was preparing a materialization command.
    """

    row = _strict_mapping(manifest, field="frozen schema manifest")
    catalog_surface = _strict_mapping(
        row.get("sealed_structured_catalog"), field="sealed catalog schema surface"
    )
    binding = catalog_surface.get("binding")
    if not isinstance(binding, Mapping):
        raise RuleDerivativeTrainingError("schema freeze requires a sealed catalog binding")
    current = build_frozen_schema_manifest(catalog_binding=binding)
    if _public_digest(row) != _public_digest(current):
        raise RuleDerivativeTrainingError(
            "schema manifest is stale; regenerate only after all owner schema bindings pass"
        )
    target = Path(path)
    if target.parent.is_symlink():
        raise RuleDerivativeTrainingError("frozen schema manifest parent cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_create_only_json(target, row)
    return _sha256_file(target)


def _byte_identity(value: Any) -> str:
    """Hash an observed logit payload without coercing its floating bytes."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            header = _canonical_json(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            ).encode("utf-8")
            return "sha256:" + hashlib.sha256(header + b"\\0" + raw).hexdigest()
    except ModuleNotFoundError:
        pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "sha256:" + hashlib.sha256(bytes(value)).hexdigest()
    return _public_digest(value)


def make_zero_bypass_receipt(
    *,
    schema_manifest: Mapping[str, Any],
    baseline_checkpoint_sha256: str,
    baseline_checkpoint_size_bytes: int,
    baseline_logits: Any,
    layer_off_logits: Any,
    baseline_legal_choice: Any,
    layer_off_legal_choice: Any,
    evidence_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Attest a real observed disabled-path parity probe.

    It requires an explicit receipt-bound r274 identity.  The canonical goal
    parameterizes that identity at launch, so this helper must not substitute
    a magic historical checkpoint digest on the owner's behalf.
    """

    if schema_manifest.get("schema") != R298_FROZEN_SCHEMA_MANIFEST_SCHEMA:
        raise RuleDerivativeTrainingError("zero-bypass receipt requires frozen r298 schema")
    _assert_contract_identity(schema_manifest)
    baseline_checkpoint_sha256 = _require_sha(
        baseline_checkpoint_sha256, field="receipt-bound r274 baseline checkpoint"
    )
    baseline_checkpoint_size_bytes = _exact_int(
        baseline_checkpoint_size_bytes, field="receipt-bound r274 baseline checkpoint size", minimum=1
    )
    baseline_identity = _byte_identity(baseline_logits)
    layer_off_identity = _byte_identity(layer_off_logits)
    same_logits = baseline_identity == layer_off_identity
    same_choice = baseline_legal_choice == layer_off_legal_choice
    status = "passed_exact_baseline_logits" if same_logits and same_choice else "failed_parity"
    return {
        "schema": R298_ZERO_BYPASS_RECEIPT_SCHEMA,
        "version": 1,
        "status": status,
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "rule_derivative_contract_sha256": R298_CONTRACT_SHA256,
        "rule_derivative_goal_sha256": R298_GOAL_SHA256,
        "frozen_schema_manifest_sha256": _public_digest(schema_manifest),
        "baseline_r274_checkpoint_sha256": baseline_checkpoint_sha256,
        "baseline_r274_checkpoint_size_bytes": baseline_checkpoint_size_bytes,
        "baseline_logits_byte_identity_sha256": baseline_identity,
        "layer_off_logits_byte_identity_sha256": layer_off_identity,
        "layer_off_bit_identical_baseline_logits": same_logits,
        "baseline_legal_choice": baseline_legal_choice,
        "layer_off_legal_choice": layer_off_legal_choice,
        "layer_off_identical_legal_choice": same_choice,
        "all_new_routes_exact_zero_or_bypassed": True,
        "checklist_provenance_schema_frozen": True,
        "runtime_wired": False,
        "evidence_provenance": dict(evidence_provenance),
        "runtime_authority": {
            "elmo_only": True,
            "production_authority": False,
            "inzi_authority": False,
            "training_authority": False,
        },
    }


def validate_raw_corpus_binding(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    """Require the exact validated/deduplicated thirty-day corpus proof."""

    # Let the raw-corpus owner validate the complete typed shape first.  This
    # includes the exact two immutable source-receipt identities, per-day
    # coverage, raw ZIP/member counts, and episode-level dedup proof.  Do not
    # reproduce a weakened subset of that ABI here.
    try:
        validate_raw_corpus_manifest(manifest)
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            "raw manifest fails the canonical r298 30-day provenance validator"
        ) from exc

    if manifest.get("schema") != R298_RAW_CORPUS_MANIFEST_SCHEMA:
        raise RuleDerivativeTrainingError("raw manifest is not the r298 30-day manifest")
    if manifest.get("status") != "sealed_create_only_raw_zip_inventory":
        raise RuleDerivativeTrainingError("raw manifest is not sealed create-only inventory")
    if receipt.get("schema") != R298_RAW_CORPUS_RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise RuleDerivativeTrainingError("raw corpus receipt has not passed")
    if receipt.get("raw_expert_corpus_manifest_sha256") != _public_digest(manifest):
        raise RuleDerivativeTrainingError("raw receipt does not bind the supplied manifest")
    exact_window = (
        manifest.get("window_start_utc"),
        manifest.get("window_end_utc"),
        manifest.get("distinct_utc_day_count"),
    )
    if exact_window != (R298_RAW_WINDOW_START, R298_RAW_WINDOW_END, R298_RAW_DAY_COUNT):
        raise RuleDerivativeTrainingError("raw source is not exact contiguous 2026-07-13..2026-08-11")
    archives = _strict_list(manifest.get("archives"), field="raw manifest archives")
    if len(archives) != R298_RAW_DAY_COUNT:
        raise RuleDerivativeTrainingError("raw manifest does not contain exactly thirty archive rows")
    dates = [
        _strict_mapping(row, field="raw archive row").get("date") for row in archives
    ]
    if dates != list(_utc_days()):
        raise RuleDerivativeTrainingError("raw manifest has missing, duplicate, or non-UTC day partitions")
    archive_shas = [
        _require_sha(_strict_mapping(row, field="raw archive row").get("sha256"), field="archive digest")
        for row in archives
    ]
    if len(set(archive_shas)) != R298_RAW_DAY_COUNT:
        raise RuleDerivativeTrainingError("raw manifest reuses an archive source across partitions")
    dedup = _strict_mapping(manifest.get("deduplication"), field="raw deduplication")
    required_dedup = {
        "unique_date_count": R298_RAW_DAY_COUNT,
        "unique_archive_sha256_count": R298_RAW_DAY_COUNT,
        "unique_date_dataset_source_count": R298_RAW_DAY_COUNT,
        "duplicate_day_or_source_or_archive_permitted": False,
    }
    if any(dedup.get(name) != expected for name, expected in required_dedup.items()):
        raise RuleDerivativeTrainingError("raw day/source deduplication proof is incomplete")
    episode_dedup = _strict_mapping(
        manifest.get("episode_deduplication"), field="episode deduplication"
    )
    required_episode_dedup = {
        "episode_identity_algorithm": "payload.id_plus_canonical_content_sha256",
        "duplicate_episode_identity_count": 0,
        "duplicate_episode_id_with_distinct_content_count": 0,
        "excluded_duplicate_mapping": [],
    }
    if any(episode_dedup.get(name) != expected for name, expected in required_episode_dedup.items()):
        raise RuleDerivativeTrainingError("raw episode deduplication proof is incomplete")
    _require_sha(
        episode_dedup.get("episode_identity_inventory_sha256"),
        field="episode identity inventory",
    )
    source_provenance = _strict_list(
        manifest.get("source_manifest_provenance"), field="raw source receipt provenance"
    )
    source_coverage = _strict_list(
        manifest.get("source_receipt_day_coverage"), field="raw source receipt coverage"
    )
    if receipt.get("source_manifest_provenance_sha256") != _public_digest(source_provenance):
        raise RuleDerivativeTrainingError("raw receipt does not bind source receipt provenance")
    if receipt.get("source_receipt_day_coverage_sha256") != _public_digest(source_coverage):
        raise RuleDerivativeTrainingError("raw receipt does not bind per-day source receipt coverage")
    if receipt.get("episode_deduplication_sha256") != _public_digest(episode_dedup):
        raise RuleDerivativeTrainingError("raw receipt does not bind episode deduplication")
    if receipt.get("completed_raw_zip_member_count") != manifest.get("total_raw_zip_json_members") or receipt.get(
        "completed_validated_episode_count"
    ) != manifest.get("total_validated_episodes"):
        raise RuleDerivativeTrainingError("raw receipt does not bind complete raw ZIP counts")
    source_disjointness = _strict_mapping(
        receipt.get("source_disjointness"), field="raw receipt source disjointness"
    )
    required_source_disjointness = {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
        "training_eligible": False,
    }
    if any(source_disjointness.get(name) is not expected for name, expected in required_source_disjointness.items()):
        raise RuleDerivativeTrainingError("raw receipt source/day/group provenance is incomplete")
    if receipt.get("distinct_utc_day_count") != R298_RAW_DAY_COUNT:
        raise RuleDerivativeTrainingError("raw receipt does not attest thirty UTC days")
    _assert_contract_identity(receipt, field="raw corpus receipt")


def verify_raw_source_receipt_files(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    """Re-open immutable source receipts before materialization on Elmo.

    The pure canonical validator proves the signed layout.  This additional
    materializer-side check proves that the physical receipt files named by
    that layout still hash to their recorded identities and actually cover
    the exact raw archive identity for every claimed day.  It is deliberately
    read-only and fails closed; it never recollects or substitutes a source.
    """

    validate_raw_corpus_binding(manifest, receipt)
    archive_by_day = {
        str(row["date"]): _strict_mapping(row, field="manifest archive row")
        for row in _strict_list(manifest.get("archives"), field="raw manifest archives")
    }
    source_rows: dict[str, Mapping[str, Any]] = {}
    for raw in _strict_list(
        manifest.get("source_manifest_provenance"), field="raw source receipt provenance"
    ):
        row = _strict_mapping(raw, field="raw source receipt provenance row")
        digest = _require_sha(row.get("sha256"), field="raw source receipt digest")
        path_value = row.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RuleDerivativeTrainingError("raw source receipt path is missing")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise RuleDerivativeTrainingError("raw source receipt physical identity drifted")
        payload = _read_json(path, field="raw source receipt")
        if payload.get("schema") != row.get("schema") or payload.get("status") != "ready":
            raise RuleDerivativeTrainingError("raw source receipt schema/status drifted")
        archives = _strict_list(payload.get("archives"), field="raw source receipt archives")
        by_day: dict[str, Mapping[str, Any]] = {}
        for archive in archives:
            archive_row = _strict_mapping(archive, field="raw source receipt archive")
            day = archive_row.get("date")
            if isinstance(day, str) and day not in by_day:
                by_day[day] = archive_row
        source_rows[digest] = by_day
    for raw in _strict_list(
        manifest.get("source_receipt_day_coverage"), field="raw source receipt coverage"
    ):
        coverage = _strict_mapping(raw, field="raw source day coverage")
        day = coverage.get("date")
        target = archive_by_day.get(str(day))
        if not isinstance(day, str) or target is None:
            raise RuleDerivativeTrainingError("raw source coverage has an unknown day")
        identities = _strict_list(coverage.get("source_receipt_sha256s"), field="source receipt identities")
        for identity in identities:
            source = source_rows.get(_require_sha(identity, field="covered source receipt"), {}).get(day)
            if source is None:
                raise RuleDerivativeTrainingError("raw source receipt does not cover its claimed day")
            for field in ("date", "dataset_slug", "path", "sha256", "bytes"):
                if source.get(field) != target.get(field):
                    raise RuleDerivativeTrainingError(
                        f"raw source receipt/archive identity drifted for {day}:{field}"
                    )
            if source.get("validated") is not True:
                raise RuleDerivativeTrainingError("raw source receipt day is not validated")


@dataclass
class ExperimentLease:
    """An explicit create-only lease serializing memory-heavy r298 jobs.

    The lease is intentionally a small local file, not a service or a host
    tuning knob.  It never steals a pre-existing lease: a stale-looking lease
    remains an operator-visible blocker rather than something this tool tries
    to delete or override.
    """

    path: Path
    token: str
    acquired: bool = False

    @classmethod
    def acquire(cls, path: Path | str, *, purpose: str) -> "ExperimentLease":
        target = Path(path)
        token = _public_digest(
            {"pid": os.getpid(), "purpose": purpose, "monotonic": time.monotonic_ns()}
        )
        payload = {
            "schema": "poke_bot.alakazam_rule_derivative_r298_experiment_lease/v1",
            "pid": os.getpid(),
            "purpose": purpose,
            "token": token,
            "acquired_monotonic_ns": time.monotonic_ns(),
            "memory_heavy_phase_concurrency_exact": 1,
            "ram_hard_ceiling_bytes": R298_EXPERIMENT_RAM_BYTES,
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise RuleDerivativeTrainingError(
                f"memory-heavy r298 lease is already held: {target}; do not bypass it"
            ) from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((_canonical_json(payload) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return cls(path=target, token=token, acquired=True)

    def assert_held(self) -> None:
        if not self.acquired:
            raise RuleDerivativeTrainingError("r298 resource lease is not held")
        payload = _read_json(self.path, field="experiment lease")
        if payload.get("token") != self.token or payload.get("pid") != os.getpid():
            raise RuleDerivativeTrainingError("r298 resource lease identity changed")

    def release(self) -> None:
        """Release only the exact file this process created.

        This is deliberately opt-in and is not invoked by a failed job.  A
        failed run leaves its lease/artifacts for inspection instead of making
        a later run look as if no contention occurred.
        """

        self.assert_held()
        payload = _read_json(self.path, field="experiment lease")
        if payload.get("token") != self.token:
            raise RuleDerivativeTrainingError("refusing to release another job's lease")
        self.path.unlink()
        self.acquired = False


@dataclass
class ResourceTelemetry:
    """Bounded aggregate telemetry for one exclusive r298 experiment lease.

    Materialization deliberately uses threads rather than worker processes, so
    one leased root process is normally the whole memory-heavy experiment.  We
    still sample its live descendant tree instead of asserting that fact from a
    configuration boolean.  Threads are intentionally not summed: doing so
    would double-count the same address space and defeat the contract's
    non-double-counting requirement.
    """

    _start_wall: float = 0.0
    _start_cpu: float = 0.0
    _peak_rss: int = 0
    _peak_aggregate_rss: int = 0
    _peak_descendant_count: int = 0
    _observed_process_ids: set[int] | None = None
    _peak_gpu_by_device: dict[str, int] | None = None
    _peak_gpu_util_by_device: dict[str, int] | None = None
    lease: ExperimentLease | None = None
    require_exclusive_lease: bool = False

    def __post_init__(self) -> None:
        if self.require_exclusive_lease:
            if self.lease is None:
                raise RuleDerivativeTrainingError("memory-heavy materialization requires an exclusive r298 lease")
            self.lease.assert_held()
        self._start_wall = time.perf_counter()
        current = resource.getrusage(resource.RUSAGE_SELF)
        self._start_cpu = current.ru_utime + current.ru_stime
        self._peak_rss = self._rss_bytes()
        self._peak_aggregate_rss = self._peak_rss
        self._observed_process_ids = {os.getpid()}
        self._peak_gpu_by_device = {}
        self._peak_gpu_util_by_device = {}
        self.sample()

    @staticmethod
    def _rss_bytes() -> int:
        # Elmo is Linux.  The platform label in the receipt prevents a caller
        # from mistaking macOS's byte unit for Linux's KiB unit.
        scale = 1024 if os.name == "posix" and Path("/proc").exists() else 1
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale

    @staticmethod
    def _linux_process_tree_rss(root_pid: int) -> tuple[int, set[int]]:
        """Return current RSS for root plus descendants without double count.

        The function is intentionally best-effort on non-Linux development
        hosts.  Receipt-bearing Elmo runs are Linux; failure to inspect /proc
        there is reported explicitly rather than converted into a claim that a
        multi-process experiment was serialized.
        """

        proc = Path("/proc")
        if not proc.is_dir():
            return 0, {root_pid}
        parents: dict[int, list[int]] = defaultdict(list)
        rss: dict[int, int] = {}
        for child in proc.iterdir():
            if not child.name.isdigit():
                continue
            pid = int(child.name)
            try:
                stat_raw = (child / "stat").read_text(encoding="utf-8")
                # ``comm`` is parenthesized and may contain spaces.  Parse
                # from its closing parenthesis, where the remaining fields
                # begin ``state ppid ...``.
                closing = stat_raw.rfind(")")
                stat_fields = stat_raw[closing + 1 :].split()
                ppid = int(stat_fields[1])
                status = (child / "status").read_text(encoding="utf-8")
            except (OSError, ValueError, IndexError):
                continue
            kb = 0
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    pieces = line.split()
                    if len(pieces) >= 2:
                        try:
                            kb = int(pieces[1])
                        except ValueError:
                            kb = 0
                    break
            parents[ppid].append(pid)
            rss[pid] = max(0, kb) * 1024
        seen: set[int] = set()
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            pending.extend(parents.get(pid, ()))
        return sum(rss.get(pid, 0) for pid in seen), seen

    @staticmethod
    def _gpu_sample() -> tuple[dict[str, int], dict[str, int]]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return {}, {}
        if result.returncode != 0:
            return {}, {}
        memory: dict[str, int] = {}
        utilization: dict[str, int] = {}
        for line in result.stdout.splitlines():
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) != 3:
                continue
            try:
                memory[pieces[0]] = int(pieces[1]) * 1024 * 1024
                utilization[pieces[0]] = int(pieces[2])
            except ValueError:
                continue
        return memory, utilization

    def sample(self) -> None:
        if self.require_exclusive_lease:
            assert self.lease is not None
            self.lease.assert_held()
        self._peak_rss = max(self._peak_rss, self._rss_bytes())
        tree_rss, process_ids = self._linux_process_tree_rss(os.getpid())
        # ``ru_maxrss`` preserves this process's historical high-water mark;
        # the process-tree value catches live child workers.  Their maximum is
        # a conservative non-double-counted aggregate for this bounded runner.
        aggregate_rss = max(self._peak_rss, tree_rss)
        self._peak_aggregate_rss = max(self._peak_aggregate_rss, aggregate_rss)
        self._peak_descendant_count = max(self._peak_descendant_count, max(0, len(process_ids) - 1))
        assert self._observed_process_ids is not None
        self._observed_process_ids.update(process_ids)
        if self._peak_aggregate_rss > R298_EXPERIMENT_RAM_BYTES:
            raise RuleDerivativeTrainingError("isolated experiment exceeded the 96-GiB RAM ceiling")
        memory, utilization = self._gpu_sample()
        assert self._peak_gpu_by_device is not None and self._peak_gpu_util_by_device is not None
        for device, value in memory.items():
            self._peak_gpu_by_device[device] = max(value, self._peak_gpu_by_device.get(device, 0))
        for device, value in utilization.items():
            self._peak_gpu_util_by_device[device] = max(value, self._peak_gpu_util_by_device.get(device, 0))

    def final(self) -> dict[str, Any]:
        self.sample()
        current = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = current.ru_utime + current.ru_stime - self._start_cpu
        return {
            "experiment_ram_measurement_method": (
                "exclusive_r298_lease_plus_aggregate_process_tree_rss_linux_bytes"
                if self.require_exclusive_lease
                else "self_ru_maxrss_linux_kib_to_bytes_nonreceipt_smoke"
            ),
            "experiment_ram_peak_bytes": self._peak_aggregate_rss,
            "experiment_ram_hard_ceiling_bytes": R298_EXPERIMENT_RAM_BYTES,
            "experiment_ram_accounting_scope": (
                "leased root process plus observed live descendants; threads are not double-counted"
                if self.require_exclusive_lease
                else "local smoke only; not a materialization/training receipt"
            ),
            "aggregate_experiment_root_pid": os.getpid(),
            "aggregate_experiment_process_ids_observed": sorted(self._observed_process_ids or ()),
            "aggregate_experiment_descendant_peak_count": self._peak_descendant_count,
            "aggregate_experiment_peak_bytes": self._peak_aggregate_rss,
            "cpu_process_or_utilization_peak": {
                "self_cpu_seconds": cpu_seconds,
                "wall_seconds": time.perf_counter() - self._start_wall,
            },
            "gpu_memory_peak_bytes_per_device_or_not_applicable": dict(
                sorted((self._peak_gpu_by_device or {}).items())
            ) or "not_applicable_or_nvidia_smi_unavailable",
            "gpu_utilization_peak_per_device_or_not_applicable": dict(
                sorted((self._peak_gpu_util_by_device or {}).items())
            ) or "not_applicable_or_nvidia_smi_unavailable",
            "one_memory_heavy_phase_at_a_time": True,
            "memory_heavy_phase_concurrency_exact": 1,
            "exclusive_lease": (
                {"path": str(self.lease.path), "token": self.lease.token}
                if self.require_exclusive_lease and self.lease is not None
                else "not_required_for_smoke"
            ),
            "managed_service_change": False,
            "zfs_arc_l2arc_tuning": False,
        }


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one immutable JSON artifact, never replacing an earlier run."""

    encoded = (_canonical_json(value) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuleDerivativeTrainingError(f"create-only artifact already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The incomplete file remains a visible failed attempt rather than
        # being silently overwritten on retry.  It has no completion marker.
        raise


def _read_json(path: Path | str, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleDerivativeTrainingError(f"cannot read {field}: {path}") from exc
    return _strict_mapping(value, field=field)


def _archive_for_day(manifest: Mapping[str, Any], day: str) -> Mapping[str, Any]:
    for row in _strict_list(manifest.get("archives"), field="raw manifest archives"):
        mapped = _strict_mapping(row, field="raw archive row")
        if mapped.get("date") == day:
            return mapped
    raise RuleDerivativeTrainingError(f"sealed manifest has no archive for {day}")


def _archive_path(archive: Mapping[str, Any], *, archive_root: Path | None) -> Path:
    raw = archive.get("path")
    if not isinstance(raw, str) or not raw:
        raise RuleDerivativeTrainingError("raw archive path is missing")
    source = Path(raw)
    return (archive_root / source.name) if archive_root is not None else source


def _verify_archive(archive: Mapping[str, Any], path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuleDerivativeTrainingError(f"sealed archive is unavailable or not regular: {path}")
    expected_digest = _require_sha(archive.get("sha256"), field="archive digest")
    expected_size = _exact_int(archive.get("bytes"), field="archive byte count", minimum=1)
    if path.stat().st_size != expected_size:
        raise RuleDerivativeTrainingError(f"sealed archive byte count drifted: {path.name}")
    if _sha256_file(path) != expected_digest:
        raise RuleDerivativeTrainingError(f"sealed archive digest drifted: {path.name}")
    try:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            json_names = [name for name in names if name.endswith(".json") and not name.endswith("/")]
            if len(names) != len(json_names) or len(names) != len(set(names)):
                raise RuleDerivativeTrainingError(f"raw archive member layout drifted: {path.name}")
            if len(json_names) != _exact_int(
                archive.get("zip_json_member_count"), field="archive member count", minimum=1
            ):
                raise RuleDerivativeTrainingError(f"raw archive member count drifted: {path.name}")
    except zipfile.BadZipFile as exc:
        raise RuleDerivativeTrainingError(f"sealed archive is not a ZIP: {path}") from exc


def _policy_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project an observation onto the only fields a policy may persist/use."""

    try:
        result = sanitize_public_observation(observation)
    except Exception as exc:
        raise RuleDerivativeTrainingError(f"cannot sanitize public observation: {exc}") from exc
    current = result.get("current")
    if not isinstance(current, Mapping):
        raise RuleDerivativeTrainingError("sanitized observation lacks current state")
    trimmed_current = dict(current)
    for key in (
        "result",
        "resultReason",
        "reason",
        "future",
        "futureState",
        "future_state",
        "futureTransitions",
        "future_transitions",
        "nextState",
        "next_state",
        "successor",
    ):
        trimmed_current.pop(key, None)
    result["current"] = trimmed_current
    for key in (
        "future",
        "futureState",
        "future_state",
        "futureTransitions",
        "future_transitions",
        "nextState",
        "next_state",
        "successor",
    ):
        result.pop(key, None)
    return result


def _episode_seed_identity(payload: Mapping[str, Any]) -> str:
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping) or "seed" not in configuration:
        raise RuleDerivativeTrainingError("raw episode lacks explicit configuration.seed for split proof")
    seed = configuration.get("seed")
    if isinstance(seed, (Mapping, list, tuple, set)) or seed is None:
        raise RuleDerivativeTrainingError("raw episode configuration.seed is malformed")
    return _public_digest({"seed": seed})


def _episode_identity(payload: Mapping[str, Any]) -> str:
    value = payload.get("id")
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise RuleDerivativeTrainingError("raw episode lacks stable payload.id")
    return _public_digest({"payload_id": str(value)})


def _extract_decks(payload: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    """Read setup decks only to call legacy diagnostics, never persist cards."""

    try:
        steps = _strict_list(payload.get("steps"), field="raw episode steps")
        first = _strict_list(steps[0], field="raw episode initial step")
        visual = _strict_mapping(first[0], field="raw episode visual").get("visualize")
        trace = _strict_list(visual, field="raw episode visualizer trace")
        setup = _strict_mapping(trace[0], field="raw setup trace row").get("action")
        rows = _strict_list(setup, field="raw setup decks")
    except (IndexError, RuleDerivativeTrainingError):
        return {}
    if len(rows) != 2:
        return {}
    output: dict[int, tuple[int, ...]] = {}
    for seat, deck in enumerate(rows):
        if not isinstance(deck, (list, tuple)) or isinstance(deck, (str, bytes)):
            continue
        try:
            output[seat] = tuple(_exact_int(card, field="setup card", minimum=1) for card in deck)
        except RuleDerivativeTrainingError:
            continue
    return output


def validate_strict_mechanics_provenance(
    provenance: Mapping[str, Any] | None,
) -> tuple[bool, str, str | None]:
    """Authorize Q1/Q2/Q8 only from receipt-bound public simulator/catalog facts.

    This intentionally rejects a guide identifier, text hash, or loose CSV as
    a substitute.  A valid receipt still does *not* fabricate a rule answer:
    it only permits a separately supplied strict public evaluator to emit one.
    """

    if provenance is None:
        return False, "strict_pinned_mechanics_receipt_not_supplied", None
    try:
        row = _strict_mapping(provenance, field="strict mechanics provenance")
    except RuleDerivativeTrainingError:
        return False, "strict_pinned_mechanics_receipt_malformed", None
    if row.get("schema") != R298_STRICT_MECHANICS_RECEIPT_SCHEMA:
        return False, "strict_pinned_mechanics_receipt_schema_mismatch", None
    if row.get("canonical_libcg_sha256") != CANONICAL_R236_LIBCG_SHA256:
        return False, "pinned_libcg_identity_missing_or_mismatched", None
    try:
        _require_sha(row.get("legal_option_surface_receipt_sha256"), field="legal option receipt")
        _require_sha(row.get("public_catalog_sha256"), field="public catalog digest")
    except RuleDerivativeTrainingError:
        return False, "public_catalog_or_legal_option_receipt_missing", None
    if row.get("public_catalog_provenance") != "pinned_public_catalog_or_simulator_mechanics":
        return False, "public_catalog_provenance_not_pinned", None
    if row.get("guide_or_unpinned_csv_mechanics_authority") is not False:
        return False, "guide_or_unpinned_csv_mechanics_not_allowed", None
    if row.get("q1_q2_q8_authorized") is not True:
        return False, "strict_q1_q2_q8_authorization_not_granted", None
    return True, "pinned_libcg_and_public_catalog_receipted", _public_digest(row)


def load_sealed_structured_catalog(
    catalog_path: Path | str,
    receipt_path: Path | str,
    *,
    fixed_vectors_path: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Read the owner-sealed structured catalog without reinterpreting Card2Vec.

    The function validates immutable file identities and the receipt's explicit
    statement that CSV/text hashes are not mechanics authority.  It does not
    train, mutate, or replace the legacy Card2Vec vectors.  Crucially, it
    returns the public-rule adapter's ``SealedPublicCatalog`` object instead of
    a lookalike mapping: the adapter and target compiler deliberately reject
    raw mappings so a caller cannot make an arbitrary CSV-derived catalog
    feature-bearing by adding a provenance key.
    """

    catalog_file = Path(catalog_path)
    receipt_file = Path(receipt_path)
    if catalog_file.is_symlink() or receipt_file.is_symlink():
        raise RuleDerivativeTrainingError("sealed catalog/receipt must not be symlinks")
    receipt = _read_json(receipt_file, field="sealed structured catalog receipt")
    if receipt.get("schema") != "poke_bot.alakazam_public_catalog_r298_receipt/v1":
        raise RuleDerivativeTrainingError("sealed catalog receipt schema drifted")
    if receipt.get("status") != "passed_elmo_only_nonproduction":
        raise RuleDerivativeTrainingError("sealed catalog receipt is not passed")
    if receipt.get("goal_sha256") != R298_GOAL_SHA256 or receipt.get("contract_sha256") != R298_CONTRACT_SHA256:
        raise RuleDerivativeTrainingError("sealed catalog receipt binds stale goal/contract")
    if receipt.get("libcg_sha256") != CANONICAL_R236_LIBCG_SHA256 or receipt.get("libcg_size_bytes") != CANONICAL_R236_LIBCG_SIZE_BYTES:
        raise RuleDerivativeTrainingError("sealed catalog receipt libcg identity drifted")
    if receipt.get("catalog_schema") != "poke_bot.alakazam_public_catalog_r298/v1":
        raise RuleDerivativeTrainingError("sealed catalog payload schema drifted")
    facts = _strict_mapping(receipt.get("facts"), field="sealed catalog facts")
    if facts.get("mechanics_from_pinned_engine") is not True or facts.get("csv_used_as_mechanics_authority") is not False or facts.get("text_hash_used_as_rules_parser") is not False or facts.get("text_or_name_hash_in_new_residual_vectors") is not False:
        raise RuleDerivativeTrainingError("sealed catalog does not prove pinned structured mechanics provenance")
    if catalog_file.is_symlink() or not catalog_file.is_file():
        raise RuleDerivativeTrainingError("sealed catalog file is unavailable")
    if catalog_file.stat().st_size != _exact_int(receipt.get("catalog_file_size_bytes"), field="catalog size", minimum=1):
        raise RuleDerivativeTrainingError("sealed catalog file size drifted")
    if _sha256_file(catalog_file) != _require_sha(receipt.get("catalog_file_sha256"), field="catalog digest"):
        raise RuleDerivativeTrainingError("sealed catalog file digest drifted")
    vectors_file = (
        Path(fixed_vectors_path)
        if fixed_vectors_path is not None
        else catalog_file.with_name("fixed-vectors.pt")
    )
    if vectors_file.is_symlink() or not vectors_file.is_file():
        raise RuleDerivativeTrainingError("sealed structured residual vectors are unavailable")
    if vectors_file.stat().st_size != _exact_int(
        receipt.get("fixed_vectors_file_size_bytes"), field="structured vector size", minimum=1
    ):
        raise RuleDerivativeTrainingError("sealed structured residual vector size drifted")
    if _sha256_file(vectors_file) != _require_sha(
        receipt.get("fixed_vectors_file_sha256"), field="structured vector digest"
    ):
        raise RuleDerivativeTrainingError("sealed structured residual vector digest drifted")
    try:
        from .alakazam_public_rule_adapter_r298 import (
            is_public_catalog_eligible,
            load_sealed_public_catalog,
        )

        sealed_catalog = load_sealed_public_catalog(
            catalog_file,
            receipt_file,
            vectors_path=vectors_file,
        )
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            "sealed catalog is not eligible through the public-rule adapter boundary"
        ) from exc
    if not is_public_catalog_eligible(sealed_catalog):
        raise RuleDerivativeTrainingError("sealed catalog eligibility changed while loading")
    payload = _read_json(catalog_file, field="sealed structured catalog")
    cards = payload.get("cards")
    attacks = payload.get("attacks")
    if not isinstance(cards, list) or not isinstance(attacks, list):
        raise RuleDerivativeTrainingError("sealed catalog lacks cards or attacks")
    metadata_provenance = _strict_mapping(
        receipt.get("metadata_provenance"), field="sealed catalog metadata provenance"
    )
    for field in ("engine_cards_sha256", "engine_attacks_sha256"):
        _require_sha(metadata_provenance.get(field), field=f"sealed catalog {field}")
    payload_provenance = _strict_mapping(payload.get("provenance"), field="sealed catalog payload provenance")
    for field in ("engine_cards_sha256", "engine_attacks_sha256"):
        if payload_provenance.get(field) != metadata_provenance.get(field):
            raise RuleDerivativeTrainingError("catalog/receipt engine metadata provenance drifted")
    binding = {
        "schema": R298_SEALED_CATALOG_BINDING_SCHEMA,
        "catalog_path": str(catalog_file.resolve()),
        "catalog_sha256": receipt["catalog_file_sha256"],
        "catalog_size_bytes": receipt["catalog_file_size_bytes"],
        "catalog_semantic_sha256": receipt.get("catalog_semantic_sha256"),
        "receipt_path": str(receipt_file.resolve()),
        "receipt_sha256": _sha256_file(receipt_file),
        "receipt_payload_sha256": receipt.get("receipt_payload_sha256"),
        "fixed_vectors_file_sha256": receipt.get("fixed_vectors_file_sha256"),
        "fixed_vectors_file_size_bytes": receipt.get("fixed_vectors_file_size_bytes"),
        "fixed_vectors_path": str(vectors_file.resolve()),
        "structured_rule_vectors_sha256": facts.get("structured_rule_vectors_sha256"),
        "engine_cards_sha256": metadata_provenance.get("engine_cards_sha256"),
        "engine_attacks_sha256": metadata_provenance.get("engine_attacks_sha256"),
        "card2vec_preserved_unchanged": facts.get("card2vec_preserved_unchanged") is True,
        "csv_or_text_hash_mechanics_authority": False,
        "pinned_libcg_sha256": CANONICAL_R236_LIBCG_SHA256,
        "adapter_catalog_kind": "sealed_public_catalog",
        "adapter_catalog_eligible": True,
    }
    return sealed_catalog, binding


def build_strict_checklist_trace(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    *,
    deck: Iterable[int] | None = None,
    strict_mechanics_provenance: Mapping[str, Any] | None = None,
    legacy_evaluator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the canonical r298 provenance filter over a diagnostic r288 trace.

    This compatibility wrapper does not own checklist semantics.  The old
    evaluator may supply diagnostic public rows, but the separately owned
    r298 filter is the only code that determines whether Q1/Q2/Q8 provenance
    is sufficient and it always emits an exact-zero residual.  Its output is
    persisted byte-for-byte as the materialized record's checklist field.
    """

    width = len(candidates)
    legacy_payload: Mapping[str, Any] | None = None
    if legacy_evaluator is None and deck is not None:
        try:
            from .alakazam_turn_checklist_logit_layer import evaluate_turn_checklist

            legacy_evaluator = evaluate_turn_checklist
        except ImportError:
            legacy_evaluator = None
    if legacy_evaluator is not None and deck is not None:
        try:
            evaluated = legacy_evaluator(observation, candidates, tuple(deck))
            if hasattr(evaluated, "to_dict"):
                evaluated = evaluated.to_dict()
            if isinstance(evaluated, Mapping):
                legacy_payload = evaluated
        except Exception:
            legacy_payload = None
    if legacy_payload is None:
        zeroes = [0.0] * width
        legacy_payload = {
            "channel_names": list(CHECKLIST_CHANNELS),
            "channels": [
                {
                    "name": name,
                    "raw": list(zeroes),
                    "normalized": list(zeroes),
                    "option_availability": [False] * width,
                    "available": False,
                    "reason": "r298_legacy_diagnostic_unavailable",
                }
                for name in CHECKLIST_CHANNELS
            ],
            # Absence is intentionally not treated as a Bench-only proof.
            "facts": {"bench_only": False, "classification": "unavailable"},
            "residuals": list(zeroes),
        }
    # A caller may supply the canonical strict provenance record.  Anything
    # else (including the prior ad-hoc receipt shape) fails closed inside the
    # canonical filter and leaves Q1/Q2/Q8 explicitly unavailable/zero.
    provenance: Mapping[str, Any] = (
        strict_mechanics_provenance
        if isinstance(strict_mechanics_provenance, Mapping)
        else {}
    )
    try:
        filtered = filter_r288_checklist_trace_r298(
            legacy_payload,
            provenance=provenance,
        )
        result = filtered.to_dict()
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            f"canonical r298 checklist provenance filter failed: {type(exc).__name__}"
        ) from exc
    if result.get("schema") != "poke_bot.alakazam_checklist_provenance_result_r298/v1":
        raise RuleDerivativeTrainingError("canonical r298 checklist filter result schema drifted")
    if result.get("runtime_wired") is not False or result.get("final_residual_gate") != 0.0:
        raise RuleDerivativeTrainingError("canonical r298 checklist filter is not exact-zero/inert")
    if len(result.get("channels", ())) != len(CHECKLIST_CHANNELS):
        raise RuleDerivativeTrainingError("canonical r298 checklist filter omitted a channel")
    return result


MAX_PHYSICAL_SHARD_BYTES: Final = 96 * 1024 * 1024
_SHARD_TARGET_BYTES: Final = 92 * 1024 * 1024
_SHARD_BUFFER_BYTES: Final = 1024 * 1024


@dataclass(frozen=True)
class PhysicalShardIdentity:
    """A content-addressed physical shard below the transfer gateway limit."""

    relative_path: str
    sha256: str
    size_bytes: int
    record_count: int
    uncompressed_jsonl_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "uncompressed_jsonl_bytes": self.uncompressed_jsonl_bytes,
            "physical_size_cap_bytes": MAX_PHYSICAL_SHARD_BYTES,
        }


class _BoundedGzipShardWriter:
    """Write concatenated-gzip JSONL members with a hard physical size cap.

    Every flushed batch is one deterministic gzip member.  Concatenated gzip
    members are a standard gzip stream, so readers can stream them normally.
    This gives exact pre-append byte accounting without retaining a whole day
    in memory or risking a 100-MB gateway rejection after compression.
    """

    def __init__(self, root: Path, *, day: str) -> None:
        self._root = root
        self._day = day
        self._root.mkdir(parents=True, exist_ok=False)
        self._pending: list[bytes] = []
        self._pending_bytes = 0
        self._part = 0
        self._stream: Any | None = None
        self._current_path: Path | None = None
        self._current_physical = 0
        self._current_uncompressed = 0
        self._current_records = 0
        self._current_hash: Any | None = None
        self._identities: list[PhysicalShardIdentity] = []

    def _open_part(self) -> None:
        if self._stream is not None:
            return
        self._current_path = self._root / f"part-{self._part:05d}.jsonl.gz"
        self._part += 1
        self._stream = self._current_path.open("xb")
        self._current_physical = 0
        self._current_uncompressed = 0
        self._current_records = 0
        self._current_hash = hashlib.sha256()

    def _close_part(self) -> None:
        if self._stream is None:
            return
        assert self._current_path is not None and self._current_hash is not None
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        if self._current_physical <= 0:
            # No empty physical shards are allowed in a completion marker.
            raise RuleDerivativeTrainingError("attempted to seal an empty re-featurization shard")
        if self._current_physical > MAX_PHYSICAL_SHARD_BYTES:
            raise RuleDerivativeTrainingError("refeature shard exceeded the 96-MiB physical cap")
        self._identities.append(
            PhysicalShardIdentity(
                relative_path=str(self._current_path.relative_to(self._root.parent.parent)),
                sha256="sha256:" + self._current_hash.hexdigest(),
                size_bytes=self._current_physical,
                record_count=self._current_records,
                uncompressed_jsonl_bytes=self._current_uncompressed,
            )
        )
        self._stream = None
        self._current_path = None
        self._current_hash = None

    @staticmethod
    def _gzip_member(rows: Sequence[bytes]) -> bytes:
        # mtime=0 keeps byte identity deterministic across resumable workers.
        return gzip.compress(b"".join(rows), compresslevel=6, mtime=0)

    def _append_batch(self, rows: Sequence[bytes]) -> None:
        if not rows:
            return
        member = self._gzip_member(rows)
        if len(member) > MAX_PHYSICAL_SHARD_BYTES:
            if len(rows) == 1:
                raise RuleDerivativeTrainingError("one re-featurized record exceeds physical shard cap")
            midpoint = len(rows) // 2
            self._append_batch(rows[:midpoint])
            self._append_batch(rows[midpoint:])
            return
        self._open_part()
        # Target lower than the hard cap, but rotate based on exact compressed
        # bytes.  A nearly full part receives no additional gzip member.
        if self._current_physical and (
            self._current_physical + len(member) > _SHARD_TARGET_BYTES
            or self._current_physical + len(member) > MAX_PHYSICAL_SHARD_BYTES
        ):
            self._close_part()
            self._open_part()
        if self._current_physical + len(member) > MAX_PHYSICAL_SHARD_BYTES:
            # Only possible when an initially empty part has a huge member;
            # it was already checked above, so preserve the invariant loudly.
            raise RuleDerivativeTrainingError("cannot keep re-feature shard below 96 MiB")
        assert self._stream is not None and self._current_hash is not None
        self._stream.write(member)
        self._current_hash.update(member)
        self._current_physical += len(member)
        self._current_uncompressed += sum(len(row) for row in rows)
        self._current_records += len(rows)

    def write(self, record: Mapping[str, Any]) -> None:
        row = canonical_json_bytes(record)
        if len(row) > MAX_PHYSICAL_SHARD_BYTES:
            raise RuleDerivativeTrainingError("one re-feature record exceeds the physical shard cap")
        self._pending.append(row)
        self._pending_bytes += len(row)
        if self._pending_bytes >= _SHARD_BUFFER_BYTES:
            self.flush()

    def flush(self) -> None:
        self._append_batch(self._pending)
        self._pending = []
        self._pending_bytes = 0

    def close(self) -> tuple[PhysicalShardIdentity, ...]:
        self.flush()
        self._close_part()
        if not self._identities:
            raise RuleDerivativeTrainingError("day materializer produced no records")
        return tuple(self._identities)


def _run_contract_payload(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    strict_mechanics_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Immutable common identity for independently resumable day workers."""

    validate_raw_corpus_binding(raw_manifest, raw_receipt)
    try:
        frozen_schema_sha, bypass_sha = validate_frozen_schema_gate(
            schema_manifest, zero_bypass_receipt
        )
    except Exception as exc:
        raise RuleDerivativeTrainingError(f"frozen schema/zero-bypass gate failed: {exc}") from exc
    _assert_contract_identity(schema_manifest)
    catalog_surface = _strict_mapping(
        schema_manifest.get("sealed_structured_catalog"), field="frozen schema sealed catalog"
    )
    binding = catalog_surface.get("binding")
    if catalog_surface.get("status") != "sealed_catalog_bound" or not isinstance(binding, Mapping):
        raise RuleDerivativeTrainingError("re-featurization requires a schema-bound sealed structured catalog")
    if binding.get("schema") != R298_SEALED_CATALOG_BINDING_SCHEMA:
        raise RuleDerivativeTrainingError("schema-bound sealed catalog binding drifted")
    if schema_manifest.get("sealed_structured_catalog_binding_sha256") != _public_digest(binding):
        raise RuleDerivativeTrainingError("schema-bound sealed catalog digest drifted")
    authorized, reason, mechanics_sha = validate_strict_mechanics_provenance(
        strict_mechanics_provenance
    )
    return {
        "schema": "poke_bot.alakazam_rule_derivative_r298_materialization_run/v1",
        "revision": R298_REVISION,
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "raw_manifest_sha256": _public_digest(raw_manifest),
        "raw_receipt_sha256": _public_digest(raw_receipt),
        "frozen_schema_manifest_sha256": frozen_schema_sha,
        "zero_bypass_receipt_sha256": bypass_sha,
        "strict_mechanics_provenance": {
            "authorized": authorized,
            "reason": reason,
            "receipt_sha256": mechanics_sha,
        },
        "sealed_structured_catalog_binding_sha256": _public_digest(binding),
        "physical_shard_cap_bytes": MAX_PHYSICAL_SHARD_BYTES,
        "resumable_complete_day_markers": True,
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def initialize_refeature_root(
    output_root: Path | str,
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    strict_mechanics_provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Create an empty, resumable materialization root exactly once."""

    target = Path(output_root)
    if target.exists():
        raise RuleDerivativeTrainingError(f"create-only materialization root exists: {target}")
    # Bind physical immutable source receipts before writing any run marker.
    # This is read-only and makes a missing/foreign receipt a hard blocker,
    # rather than allowing a later day worker to quietly use a partial window.
    verify_raw_source_receipt_files(raw_manifest, raw_receipt)
    run = _run_contract_payload(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
        strict_mechanics_provenance=strict_mechanics_provenance,
    )
    target.mkdir(parents=True, exist_ok=False)
    _write_create_only_json(target / "RUN_CONTRACT.json", run)
    _write_create_only_json(
        target / "IN_PROGRESS.json",
        {
            "schema": "poke_bot.alakazam_rule_derivative_r298_completion_marker/v1",
            "status": "in_progress_no_day_completion_implied",
            "run_contract_sha256": _public_digest(run),
            "create_only": True,
        },
    )
    return run


def load_refeature_run(output_root: Path | str) -> Mapping[str, Any]:
    root = Path(output_root)
    run = _read_json(root / "RUN_CONTRACT.json", field="materialization run contract")
    if run.get("schema") != "poke_bot.alakazam_rule_derivative_r298_materialization_run/v1":
        raise RuleDerivativeTrainingError("materialization root has foreign run schema")
    if run.get("goal_contract_sha256") != R298_CONTRACT_SHA256:
        raise RuleDerivativeTrainingError("materialization root binds stale contract")
    if run.get("physical_shard_cap_bytes") != MAX_PHYSICAL_SHARD_BYTES:
        raise RuleDerivativeTrainingError("materialization root has wrong physical shard cap")
    return run


def _full_actions_by_step(descriptors: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    """Recover the one recorded complete action from factorized stage rows."""

    selected: dict[int, tuple[int, list[int]]] = {}
    for descriptor in descriptors:
        source = _strict_mapping(descriptor.get("source"), field="stage source")
        step = _exact_int(source.get("env_step"), field="stage env step", minimum=0)
        candidate_index = _exact_int(
            descriptor.get("selected_candidate_index"),
            field="selected factorized candidate index",
            minimum=0,
        )
        candidates = _strict_list(descriptor.get("candidates"), field="stage candidates")
        if candidate_index >= len(candidates):
            raise RuleDerivativeTrainingError("selected factorized candidate is absent")
        action = [
            _exact_int(value, field="selected factorized option", minimum=0)
            for value in _strict_list(candidates[candidate_index], field="selected factorized action")
        ]
        stage = _exact_int(source.get("factorized_stage"), field="factorized stage", minimum=0)
        prior = selected.get(step)
        # The latest stage is the full recorded action.  It may be an explicit
        # stop row whose action length equals the preceding prefix, so stage
        # order rather than action length is authoritative.
        if prior is None or stage > prior[0]:
            selected[step] = (stage, action)
    return {step: action for step, (_stage, action) in selected.items()}


def _selected_action_target(
    observation: Mapping[str, Any],
    action: Sequence[int],
    *,
    metadata_catalog: Any,
    prompt_chain: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compile only the observed full action; never synthesize alternatives.

    This historical/pre-handoff materializer has no receipt-owned revision-5
    selected-action ledger validator.  It therefore *first* requests the
    target compiler's strict trainable boundary, then deliberately falls back
    to a diagnostic-only compile when that immutable authority is absent.
    The fallback vectors are explicitly all-masked.  A future r303
    post-handoff materializer must supply the target owner's sealed validator
    to the same strict call; a callback or raw receipt mapping cannot arm a
    label here.
    """

    decision: dict[str, Any] = {"observation": dict(observation), "action": list(action)}
    if prompt_chain is not None:
        decision["prompt_chain"] = prompt_chain
    try:
        compiled = compile_simulator_rule_targets(
            decision,
            metadata_catalog=metadata_catalog,
            trajectory_receipt_validator=None,
            require_trainable_trajectory_receipt=True,
            strict=True,
        )
    except Exception as exc:
        # Retain an audit-only selected transition record where possible, but
        # never let it become supervised training evidence.  The target
        # compiler itself clears every mask for this no-ledger branch.
        try:
            compiled = compile_simulator_rule_targets(
                decision,
                metadata_catalog=metadata_catalog,
                trajectory_receipt_validator=None,
                require_trainable_trajectory_receipt=False,
                strict=False,
            )
            if isinstance(compiled, Mapping):
                compiled = dict(compiled)
                compiled["materializer_trainability"] = {
                    "available": False,
                    "reason": (
                        "sealed_revision_5_selected_action_trajectory_receipt_required:"
                        f"{type(exc).__name__}"
                    ),
                    "all_training_masks_forced_false": True,
                }
        except Exception:
            compiled = {
                "schema": R298_RULE_TARGET_SCHEMA,
                "status": "unavailable",
                "target_only": True,
                "reason": f"target_compiler_exception:{type(exc).__name__}",
                "materializer_trainability": {
                    "available": False,
                    "reason": "sealed_revision_5_selected_action_trajectory_receipt_required",
                    "all_training_masks_forced_false": True,
                },
            }
    if not isinstance(compiled, Mapping):
        raise RuleDerivativeTrainingError("target compiler returned non-object")
    provenance = compiled.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("catalog_metadata") == "explicit_test_fixture_only":
        raise RuleDerivativeTrainingError(
            "test-only catalog provenance is forbidden in r298 materialization"
        )
    public_target = dict(compiled)
    # The corpus is a policy/public-representation artifact.  Even though the
    # target contract has a separately typed belief sidecar, this materializer
    # does not persist an accidental raw privileged target without a later
    # explicit target-only data contract.
    public_target.pop("privileged_belief_targets", None)
    public_target["target_only"] = True
    public_target["policy_feature_eligible"] = False
    vectors: dict[str, Any] | None = None
    try:
        from .alakazam_simulator_rule_targets_r298 import rule_head_target_vectors

        candidate_vectors = rule_head_target_vectors(
            compiled, trajectory_receipt_validator=None
        )
        if isinstance(candidate_vectors, Mapping):
            vectors = dict(candidate_vectors)
            vectors.pop("opponent_belief", None)
            if vectors.get("target_training_eligible") is not False:
                raise RuleDerivativeTrainingError(
                    "pre-handoff materializer cannot emit trainable selected-action targets"
                )
    except Exception:
        vectors = None
    return public_target, vectors


def _safe_stage_source(
    *,
    day: str,
    archive: Mapping[str, Any],
    member: str,
    member_sha256: str,
    episode_group: str,
    seed_identity: str,
    descriptor_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep split provenance while excluding an opponent deck or raw cards."""

    return {
        "source_id": _require_sha(archive.get("sha256"), field="source archive digest"),
        "utc_day_partition": day,
        "group_id": episode_group,
        "seed_id": seed_identity,
        "archive_member": member,
        "archive_member_sha256": member_sha256,
        "env_step": _exact_int(descriptor_source.get("env_step"), field="env step", minimum=0),
        "acting_seat": _exact_int(descriptor_source.get("acting_seat"), field="acting seat", minimum=0),
        "factorized_stage": _exact_int(
            descriptor_source.get("factorized_stage"), field="factorized stage", minimum=0
        ),
        # This digest is stratification/provenance only; no record field makes
        # either deck multiset available to a policy feature route.
        "acting_deck_multiset_sha256": descriptor_source.get("acting_deck_multiset_sha256"),
        "source_stratification_only": True,
    }


def _catalog_for_refeaturization(
    metadata_catalog: Any,
    *,
    schema_manifest: Mapping[str, Any],
) -> Any:
    """Default-deny arbitrary catalog mappings for the new structured path."""

    surface = _strict_mapping(
        schema_manifest.get("sealed_structured_catalog"), field="sealed catalog schema surface"
    )
    binding = _strict_mapping(surface.get("binding"), field="sealed catalog binding")
    if surface.get("status") != "sealed_catalog_bound":
        raise RuleDerivativeTrainingError("refeaturization schema has no sealed catalog")
    if binding.get("schema") != R298_SEALED_CATALOG_BINDING_SCHEMA:
        raise RuleDerivativeTrainingError("sealed catalog binding schema drifted")
    if metadata_catalog is None:
        # Catalog metadata is an optional parallel adapter input.  Missing it
        # leaves the adapter unavailable/masked rather than substituting CSV.
        return None
    try:
        from .alakazam_public_rule_adapter_r298 import is_public_catalog_eligible
    except ImportError as exc:
        raise RuleDerivativeTrainingError("public-rule sealed catalog API is unavailable") from exc
    if not is_public_catalog_eligible(metadata_catalog):
        raise RuleDerivativeTrainingError(
            "provided catalog is not the adapter's exact sealed public catalog"
        )
    # The sealed object has no arbitrary mutable mapping provenance.  Compare
    # its immutable paths and file identities against the schema-freeze
    # binding, then let the adapter/target owner independently revalidate it.
    expected_paths = {
        "catalog_path": "catalog_path",
        "receipt_path": "receipt_path",
        "fixed_vectors_path": "vectors_path",
    }
    for binding_field, object_field in expected_paths.items():
        supplied = getattr(metadata_catalog, object_field, None)
        expected = binding.get(binding_field)
        if supplied is None or not isinstance(expected, str) or str(Path(supplied).resolve()) != expected:
            raise RuleDerivativeTrainingError(
                "provided sealed catalog path does not match frozen schema binding"
            )
    provenance = getattr(metadata_catalog, "provenance", None)
    for binding_field, provenance_field in (
        ("catalog_sha256", "catalog_file_sha256"),
        ("receipt_sha256", "receipt_file_sha256"),
        ("fixed_vectors_file_sha256", "vectors_file_sha256"),
        ("structured_rule_vectors_sha256", "structured_vectors_sha256"),
        ("engine_cards_sha256", "engine_cards_sha256"),
        ("engine_attacks_sha256", "engine_attacks_sha256"),
    ):
        if getattr(provenance, provenance_field, None) != binding.get(binding_field):
            raise RuleDerivativeTrainingError(
                "provided sealed catalog identity does not match frozen schema binding"
            )
    return metadata_catalog


def _write_group_provenance(stream: Any, *, source: Mapping[str, Any]) -> None:
    # ``group_stream`` is deliberately a text JSONL stream so it can be
    # checked without decompressing data shards.  ``canonical_json_bytes``
    # owns record determinism; decode only at this file boundary.
    stream.write(
        canonical_json_bytes(
            {
                "schema": "poke_bot.alakazam_rule_derivative_r298_episode_split_unit/v1",
                "source_id": source["source_id"],
                "utc_day_partition": source["utc_day_partition"],
                "group_id": source["group_id"],
                "seed_id": source["seed_id"],
            }
        ).decode("utf-8")
    )


def _verify_legacy_feature_source() -> None:
    feature_path = Path(__file__).with_name("features.py")
    if _sha256_file(feature_path) != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise RuleDerivativeTrainingError("imported legacy features source is not exact r274 ABI")


def materialize_refeature_day(
    output_root: Path | str,
    *,
    day: str,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    strict_mechanics_provenance: Mapping[str, Any] | None = None,
    archive_root: Path | None = None,
    metadata_catalog: Any = None,
    token_builder: Callable[[Mapping[str, Any], list[list[int]]], Any] | None = None,
    prompt_chain_resolver: Callable[[Mapping[str, Any], int, Sequence[int], Mapping[str, Any]], Any] | None = None,
    telemetry: ResourceTelemetry | None = None,
) -> Mapping[str, Any]:
    """Materialize one complete UTC partition into capped physical shards.

    A partial episode/day is not accepted.  This allows bounded parallelism at
    complete-day boundaries while keeping the final merge deterministic and
    source/day/group/seed provenance exact.
    """

    if day not in _utc_days():
        raise RuleDerivativeTrainingError("requested day is outside exact 30-day UTC window")
    run = load_refeature_run(output_root)
    expected_run = _run_contract_payload(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
        strict_mechanics_provenance=strict_mechanics_provenance,
    )
    if _public_digest(run) != _public_digest(expected_run):
        raise RuleDerivativeTrainingError("materialization root inputs do not match its immutable run contract")
    if telemetry is None:
        raise RuleDerivativeTrainingError("day materialization requires receipt-bound resource telemetry")
    if not telemetry.require_exclusive_lease or telemetry.lease is None:
        raise RuleDerivativeTrainingError(
            "day materialization requires an aggregate exclusive r298 resource lease"
        )
    telemetry.sample()
    _verify_legacy_feature_source()
    effective_catalog = _catalog_for_refeaturization(
        metadata_catalog, schema_manifest=schema_manifest
    )
    root = Path(output_root)
    day_root = root / f"day={day}"
    complete_path = day_root / "DAY_COMPLETE.json"
    if complete_path.exists():
        completed = _read_json(complete_path, field="day completion marker")
        if completed.get("run_contract_sha256") != _public_digest(run):
            raise RuleDerivativeTrainingError("existing day marker belongs to a different materialization run")
        return completed
    if day_root.exists():
        raise RuleDerivativeTrainingError(
            f"incomplete day output already exists: {day_root}; inspect rather than overwrite"
        )
    archive = _archive_for_day(raw_manifest, day)
    archive_path = _archive_path(archive, archive_root=archive_root)
    _verify_archive(archive, archive_path)
    day_root.mkdir(parents=False, exist_ok=False)
    writer = _BoundedGzipShardWriter(day_root / "records", day=day)
    group_path = day_root / "episode_split_units.jsonl"
    raw_episode_count = 0
    actor_visible_frame_count = 0
    forced_frame_count = 0
    factorized_stage_count = 0
    record_count = 0
    seen_episode_groups: set[str] = set()
    source_member_inventory = hashlib.sha256()
    deck_distribution: Counter[str] = Counter()
    try:
        with group_path.open("x", encoding="utf-8") as group_stream:
            with zipfile.ZipFile(archive_path) as bundle:
                for member in sorted(bundle.namelist()):
                    try:
                        raw = bundle.read(member)
                        payload = json.loads(raw)
                    except (KeyError, OSError, json.JSONDecodeError) as exc:
                        raise RuleDerivativeTrainingError(
                            f"cannot decode sealed raw episode {archive_path.name}:{member}"
                        ) from exc
                    if not isinstance(payload, Mapping):
                        raise RuleDerivativeTrainingError("raw episode is not an object")
                    episode_group = _episode_identity(payload)
                    if episode_group in seen_episode_groups:
                        raise RuleDerivativeTrainingError("duplicate episode group inside validated day")
                    seen_episode_groups.add(episode_group)
                    seed_identity = _episode_seed_identity(payload)
                    member_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
                    source_member_inventory.update(
                        canonical_json_bytes(
                            {
                                "member": member,
                                "member_sha256": member_sha256,
                                "episode_group": episode_group,
                                "seed_id": seed_identity,
                            }
                        )
                    )
                    coverage = recorded_episode_frame_coverage(payload)
                    actor_visible_frame_count += coverage["actor_visible_selection_frame_count"]
                    forced_frame_count += coverage["forced_selection_frame_count"]
                    root_source = {
                        "source_archive_sha256": archive.get("sha256"),
                        "source_archive_date": day,
                        "source_member": member,
                        "raw_transition_target_only": True,
                    }
                    descriptors = stage_descriptors_from_recorded_episode(payload, source=root_source)
                    represented_steps = {
                        _exact_int(
                            _strict_mapping(descriptor.get("source"), field="stage source").get("env_step"),
                            field="represented env step",
                            minimum=0,
                        )
                        for descriptor in descriptors
                    }
                    if len(represented_steps) != coverage["actor_visible_selection_frame_count"]:
                        raise RuleDerivativeTrainingError("actor-visible selection frame omitted in re-featurization")
                    full_actions = _full_actions_by_step(descriptors)
                    decks = _extract_decks(payload)
                    episode_source_written = False
                    target_cache: dict[int, tuple[dict[str, Any], dict[str, Any] | None]] = {}
                    for descriptor in descriptors:
                        descriptor_source = _strict_mapping(descriptor.get("source"), field="stage source")
                        source = _safe_stage_source(
                            day=day,
                            archive=archive,
                            member=member,
                            member_sha256=member_sha256,
                            episode_group=episode_group,
                            seed_identity=seed_identity,
                            descriptor_source=descriptor_source,
                        )
                        if not episode_source_written:
                            _write_group_provenance(group_stream, source=source)
                            episode_source_written = True
                        deck_hash = source.get("acting_deck_multiset_sha256")
                        if isinstance(deck_hash, str):
                            deck_distribution[deck_hash] += 1
                        raw_observation = _strict_mapping(
                            descriptor.get("observation"), field="stage observation"
                        )
                        # Preserve the raw libcg observation only in this
                        # transient call boundary.  The adapter needs a
                        # serial-only SKILL/source locator to resolve a
                        # visible public area/slot binding before it strips
                        # serials; handing it an already-sanitized mapping
                        # can make two legal SKILL options collapse.  Nothing
                        # raw is persisted below: ``public_observation`` and
                        # the adapter representation are both sanitized.
                        persisted_observation = _policy_public_observation(raw_observation)
                        candidates = [
                            [
                                _exact_int(option, field="candidate option", minimum=0)
                                for option in _strict_list(candidate, field="candidate action")
                            ]
                            for candidate in _strict_list(descriptor.get("candidates"), field="stage candidates")
                        ]
                        selected_index = _exact_int(
                            descriptor.get("selected_candidate_index"),
                            field="selected candidate index",
                            minimum=0,
                        )
                        if selected_index >= len(candidates):
                            raise RuleDerivativeTrainingError("selected candidate index is out of range")
                        step = source["env_step"]
                        selected_action = full_actions.get(step)
                        if selected_action is None:
                            raise RuleDerivativeTrainingError("factorized stage has no recorded complete action")
                        representation = build_public_rule_representation(
                            raw_observation, metadata_catalog=effective_catalog
                        )
                        try:
                            legacy_hashes = current_feature_token_hashes(
                                raw_observation, candidates, token_builder=token_builder
                            )
                        except Exception as exc:
                            raise RuleDerivativeTrainingError(
                                f"exact r274 legacy feature signature failed: {exc}"
                            ) from exc
                        if len(legacy_hashes) != len(candidates):
                            raise RuleDerivativeTrainingError("legacy feature signature candidate alignment failed")
                        if step not in target_cache:
                            prompt_chain = None
                            if prompt_chain_resolver is not None:
                                prompt_chain = prompt_chain_resolver(
                                    payload, step, selected_action, raw_observation
                                )
                            target_cache[step] = _selected_action_target(
                                raw_observation,
                                selected_action,
                                metadata_catalog=effective_catalog,
                                prompt_chain=prompt_chain,
                            )
                        selected_target, head_vectors = target_cache[step]
                        checklist = build_strict_checklist_trace(
                            persisted_observation,
                            candidates,
                            deck=decks.get(source["acting_seat"]),
                            strict_mechanics_provenance=strict_mechanics_provenance,
                        )
                        record = {
                            "schema": R298_REFEATURE_RECORD_SCHEMA,
                            "revision": R298_REVISION,
                            "record_id": _public_digest(
                                {
                                    "raw_manifest": run["raw_manifest_sha256"],
                                    "source": source,
                                    "public_observation": representation.public_observation_hash,
                                    "stage_prefix": descriptor.get("stage_prefix"),
                                }
                            ),
                            "source": source,
                            "materialization_bindings": {
                                "validated_30_day_raw_manifest_sha256": run["raw_manifest_sha256"],
                                "frozen_schema_manifest_sha256": run["frozen_schema_manifest_sha256"],
                                "feature_schema_sha256": schema_manifest.get("feature_schema_sha256"),
                                "target_schema_sha256": schema_manifest.get("target_schema_sha256"),
                                "checklist_provenance_schema_sha256": schema_manifest.get("checklist_provenance_schema_sha256"),
                                "sealed_structured_catalog_binding_sha256": run["sealed_structured_catalog_binding_sha256"],
                            },
                            "public_observation": persisted_observation,
                            "public_rule_representation": representation.to_dict(),
                            "legacy_feature_signatures": {
                                "source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
                                "candidate_token_sha256": list(legacy_hashes),
                            },
                            "factorized_stage": {
                                "candidates": candidates,
                                "stage_prefix": list(descriptor.get("stage_prefix", [])),
                                "selected_candidate_index": selected_index,
                                "selected_complete_action": list(selected_action),
                            },
                            "selected_action_targets": selected_target,
                            "rule_head_target_vectors": head_vectors,
                            "strict_checklist_trace": checklist,
                            "all_new_policy_routes_exact_zero_or_unwired": True,
                            "target_only_fields_never_policy_features": True,
                        }
                        writer.write(record)
                        factorized_stage_count += 1
                        record_count += 1
                        if record_count % 64 == 0:
                            telemetry.sample()
                    raw_episode_count += 1
                    telemetry.sample()
        shards = writer.close()
    except Exception:
        # Deliberately leave day_root without DAY_COMPLETE.json.  It is not
        # mergeable and cannot be overwritten by a later run.
        with contextlib.suppress(Exception):
            if writer._stream is not None:
                writer._stream.close()
        raise
    group_sha = _sha256_file(group_path)
    marker = {
        "schema": "poke_bot.alakazam_rule_derivative_r298_day_completion/v1",
        "status": "complete_create_only_day",
        "run_contract_sha256": _public_digest(run),
        "raw_manifest_sha256": run["raw_manifest_sha256"],
        "utc_day_partition": day,
        "source_archive_sha256": archive.get("sha256"),
        "source_archive_size_bytes": archive.get("bytes"),
        "raw_episode_count": raw_episode_count,
        "actor_visible_selection_frame_count": actor_visible_frame_count,
        "forced_selection_frame_count": forced_frame_count,
        "factorized_stage_count": factorized_stage_count,
        "record_count": record_count,
        "episode_split_units": {
            "relative_path": str(group_path.relative_to(root)),
            "sha256": group_sha,
            "size_bytes": group_path.stat().st_size,
            "count": raw_episode_count,
        },
        "source_member_inventory_sha256": "sha256:" + source_member_inventory.hexdigest(),
        "acting_deck_distribution": dict(sorted(deck_distribution.items())),
        "physical_shards": [identity.to_dict() for identity in shards],
        "physical_shard_count": len(shards),
        "all_physical_shards_at_or_below_96_mib": all(
            identity.size_bytes <= MAX_PHYSICAL_SHARD_BYTES for identity in shards
        ),
        "all_actor_visible_and_forced_frames_included": True,
        "resource_observation": telemetry.final(),
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }
    _write_create_only_json(complete_path, marker)
    return marker


def bounded_day_parallel_plan(*, lane_count: int) -> dict[str, Any]:
    """Deterministically assign complete days to bounded in-process lanes.

    The plan is intentionally data-only.  A caller may use one process with
    bounded worker threads under one :class:`ExperimentLease`; it must not
    fork unaccounted memory-heavy jobs.  Four lanes is also compatible with
    the later quarantined transfer cap, while the default remains one.
    """

    if isinstance(lane_count, bool) or not isinstance(lane_count, int) or not 1 <= lane_count <= 4:
        raise RuleDerivativeTrainingError("materialization lane count must be an integer from 1 through 4")
    assignments = {str(index): [] for index in range(lane_count)}
    for index, day in enumerate(_utc_days()):
        assignments[str(index % lane_count)].append(day)
    return {
        "schema": "poke_bot.alakazam_rule_derivative_r298_bounded_day_parallel_plan/v1",
        "lane_count": lane_count,
        "complete_day_boundary_only": True,
        "aggregate_ram_accounted_by": "single_exclusive_lease_single_process",
        "per_lane_assignment": assignments,
        "all_days_exactly_once": sorted(
            day for rows in assignments.values() for day in rows
        ) == list(_utc_days()),
        "training_or_runtime_authority": False,
    }


def _day_marker(root: Path, day: str, *, run: Mapping[str, Any]) -> Mapping[str, Any]:
    marker = _read_json(root / f"day={day}" / "DAY_COMPLETE.json", field=f"{day} completion marker")
    if marker.get("schema") != "poke_bot.alakazam_rule_derivative_r298_day_completion/v1":
        raise RuleDerivativeTrainingError(f"day {day} completion marker schema drifted")
    if marker.get("status") != "complete_create_only_day":
        raise RuleDerivativeTrainingError(f"day {day} is not complete")
    if marker.get("run_contract_sha256") != _public_digest(run):
        raise RuleDerivativeTrainingError(f"day {day} belongs to a different run")
    if marker.get("utc_day_partition") != day:
        raise RuleDerivativeTrainingError(f"day {day} marker has wrong partition")
    if marker.get("all_physical_shards_at_or_below_96_mib") is not True:
        raise RuleDerivativeTrainingError(f"day {day} has over-limit physical shards")
    if marker.get("all_actor_visible_and_forced_frames_included") is not True:
        raise RuleDerivativeTrainingError(f"day {day} omitted actor-visible/forced frames")
    return marker


def _validate_physical_shard(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    relative = row.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuleDerivativeTrainingError("shard relative path is invalid")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RuleDerivativeTrainingError(f"physical shard is unavailable: {path}")
    expected_size = _exact_int(row.get("size_bytes"), field="physical shard size", minimum=1)
    if expected_size > MAX_PHYSICAL_SHARD_BYTES or path.stat().st_size != expected_size:
        raise RuleDerivativeTrainingError("physical shard size is invalid or drifted")
    expected_sha = _require_sha(row.get("sha256"), field="physical shard digest")
    if _sha256_file(path) != expected_sha:
        raise RuleDerivativeTrainingError("physical shard digest drifted")
    # Read a bounded header to prove it is actually a streamed r298 record
    # object, without loading the day into memory.  Full record validation is
    # performed by the collision/corpus pass, not by transfer planning.
    try:
        with gzip.open(path, "rb") as stream:
            first = stream.readline(MAX_PHYSICAL_SHARD_BYTES)
    except OSError as exc:
        raise RuleDerivativeTrainingError("physical shard is not valid gzip JSONL") from exc
    try:
        header = json.loads(first)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuleDerivativeTrainingError("physical shard lacks JSONL header") from exc
    if not isinstance(header, Mapping) or header.get("schema") != R298_REFEATURE_RECORD_SCHEMA:
        raise RuleDerivativeTrainingError("physical shard header schema drifted")
    return {
        "relative_path": relative,
        "sha256": expected_sha,
        "size_bytes": expected_size,
        "record_count": _exact_int(row.get("record_count"), field="shard record count", minimum=1),
        "uncompressed_jsonl_bytes": _exact_int(
            row.get("uncompressed_jsonl_bytes"), field="shard JSONL bytes", minimum=1
        ),
    }


def _read_split_units(root: Path, marker: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    identity = _strict_mapping(marker.get("episode_split_units"), field="day split unit identity")
    relative = identity.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuleDerivativeTrainingError("split unit path is invalid")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise RuleDerivativeTrainingError("split unit index is unavailable")
    if path.stat().st_size != _exact_int(identity.get("size_bytes"), field="split unit size", minimum=1):
        raise RuleDerivativeTrainingError("split unit index size drifted")
    if _sha256_file(path) != _require_sha(identity.get("sha256"), field="split unit digest"):
        raise RuleDerivativeTrainingError("split unit index digest drifted")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuleDerivativeTrainingError("split unit index has malformed JSONL") from exc
            mapped = _strict_mapping(row, field=f"split unit {line_number}")
            if mapped.get("schema") != "poke_bot.alakazam_rule_derivative_r298_episode_split_unit/v1":
                raise RuleDerivativeTrainingError("split unit schema drifted")
            yield mapped


def build_disjoint_split_manifest(
    output_root: Path | str,
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one simultaneous source/day/group/seed-disjoint split manifest."""

    root = Path(output_root)
    run = load_refeature_run(root)
    expected = _run_contract_payload(
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
        strict_mechanics_provenance=None,
    )
    # The strict catalog receipt is intentionally not needed to prove split
    # disjointness.  Compare the invariants that are shared by either state.
    for key in (
        "goal_contract_sha256",
        "raw_manifest_sha256",
        "raw_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
    ):
        if run.get(key) != expected.get(key):
            raise RuleDerivativeTrainingError("split inputs do not match materialization run")
    partitions = {
        "train": set(_utc_days()[:20]),
        "validation": set(_utc_days()[20:25]),
        "evaluation": set(_utc_days()[25:]),
    }
    if set().union(*partitions.values()) != set(_utc_days()) or sum(len(value) for value in partitions.values()) != R298_RAW_DAY_COUNT:
        raise RuleDerivativeTrainingError("fixed split day assignment is incomplete")
    membership: dict[str, dict[str, set[str]]] = {
        name: {"source": set(), "day": set(), "group": set(), "seed": set()}
        for name in partitions
    }
    unit_count: Counter[str] = Counter()
    seen_units: set[tuple[str, str, str, str]] = set()
    day_markers: list[Mapping[str, Any]] = []
    for day in _utc_days():
        marker = _day_marker(root, day, run=run)
        day_markers.append(marker)
        partition = next(name for name, days in partitions.items() if day in days)
        for unit in _read_split_units(root, marker):
            source = _require_sha(unit.get("source_id"), field="split source id")
            unit_day = unit.get("utc_day_partition")
            group = _require_sha(unit.get("group_id"), field="split group id")
            seed = _require_sha(unit.get("seed_id"), field="split seed id")
            if unit_day != day:
                raise RuleDerivativeTrainingError("split unit appears under the wrong day")
            compound = (source, unit_day, group, seed)
            if compound in seen_units:
                raise RuleDerivativeTrainingError("duplicate episode split unit in materialized corpus")
            seen_units.add(compound)
            membership[partition]["source"].add(source)
            membership[partition]["day"].add(day)
            membership[partition]["group"].add(group)
            membership[partition]["seed"].add(seed)
            unit_count[partition] += 1
    if not seen_units:
        raise RuleDerivativeTrainingError("refeaturization has no split units")
    overlaps: dict[str, dict[str, list[str]]] = {}
    names = tuple(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for dimension in ("source", "day", "group", "seed"):
                overlap = membership[left][dimension] & membership[right][dimension]
                if overlap:
                    overlaps[f"{left}__{right}__{dimension}"] = sorted(overlap)
    if overlaps:
        raise RuleDerivativeTrainingError(
            "source/day/group/seed split overlap detected: " + ", ".join(sorted(overlaps))
        )
    summary: dict[str, Any] = {}
    for name in names:
        summary[name] = {
            "utc_day_partitions": sorted(membership[name]["day"]),
            "unit_count": unit_count[name],
            "source_count": len(membership[name]["source"]),
            "group_count": len(membership[name]["group"]),
            "seed_count": len(membership[name]["seed"]),
            "source_identity_sha256": _public_digest(sorted(membership[name]["source"])),
            "group_identity_sha256": _public_digest(sorted(membership[name]["group"])),
            "seed_identity_sha256": _public_digest(sorted(membership[name]["seed"])),
        }
    return {
        "schema": R298_SPLIT_MANIFEST_SCHEMA,
        "version": 1,
        "status": "passed_source_day_group_seed_disjoint",
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "validated_30_day_raw_manifest_sha256": run["raw_manifest_sha256"],
        "raw_corpus_receipt_sha256": run["raw_receipt_sha256"],
        "frozen_schema_manifest_sha256": run["frozen_schema_manifest_sha256"],
        "zero_bypass_receipt_sha256": run["zero_bypass_receipt_sha256"],
        "disjoint_dimensions": ["source", "utc_day_partition", "group", "seed"],
        "partitioning": summary,
        "cross_split_overlap": {},
        "all_30_utc_days_assigned_exactly_once": True,
        "source_day_group_seed_disjoint": True,
        "runtime_authority": False,
    }


def merge_refeatured_days(
    output_root: Path | str,
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    telemetry: ResourceTelemetry,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Deterministically merge all thirty completed days without copying bytes."""

    if not telemetry.require_exclusive_lease or telemetry.lease is None:
        raise RuleDerivativeTrainingError(
            "deterministic merge requires an aggregate exclusive r298 resource lease"
        )
    telemetry.sample()
    root = Path(output_root)
    run = load_refeature_run(root)
    if (root / "REFEATURE_MANIFEST.json").exists() or (root / "REFEATURE_RECEIPT.json").exists():
        raise RuleDerivativeTrainingError("final materialization artifacts already exist; never replace them")
    split = build_disjoint_split_manifest(
        root,
        raw_manifest=raw_manifest,
        raw_receipt=raw_receipt,
        schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt,
    )
    day_rows: list[dict[str, Any]] = []
    all_shards: list[dict[str, Any]] = []
    total_records = 0
    total_episodes = 0
    total_actor_frames = 0
    total_forced_frames = 0
    for day in _utc_days():
        marker = _day_marker(root, day, run=run)
        physical_rows = [
            _validate_physical_shard(root, _strict_mapping(row, field="physical shard"))
            for row in _strict_list(marker.get("physical_shards"), field="day physical shards")
        ]
        if sum(row["record_count"] for row in physical_rows) != marker.get("record_count"):
            raise RuleDerivativeTrainingError("day physical shard records do not match day marker")
        day_rows.append(
            {
                "utc_day_partition": day,
                "day_completion_sha256": _public_digest(marker),
                "source_archive_sha256": marker.get("source_archive_sha256"),
                "raw_episode_count": marker.get("raw_episode_count"),
                "factorized_stage_count": marker.get("factorized_stage_count"),
                "record_count": marker.get("record_count"),
                "physical_shards": physical_rows,
            }
        )
        for shard_index, shard in enumerate(physical_rows):
            all_shards.append(
                {
                    "logical_id": f"{day}:{shard_index:05d}",
                    "utc_day_partition": day,
                    **shard,
                }
            )
        total_records += int(marker["record_count"])
        total_episodes += int(marker["raw_episode_count"])
        total_actor_frames += int(marker["actor_visible_selection_frame_count"])
        total_forced_frames += int(marker["forced_selection_frame_count"])
        telemetry.sample()
    frozen_schema = _strict_mapping(schema_manifest, field="frozen schema manifest")
    manifest = {
        "schema": R298_REFEATURE_MANIFEST_SCHEMA,
        "version": 1,
        "status": "complete_30_day_create_only_refeaturization_pending_census_gate",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "validated_30_day_raw_manifest_sha256": run["raw_manifest_sha256"],
        "raw_corpus_receipt_sha256": run["raw_receipt_sha256"],
        "frozen_schema_manifest_sha256": run["frozen_schema_manifest_sha256"],
        "zero_bypass_receipt_sha256": run["zero_bypass_receipt_sha256"],
        "feature_schema_sha256": frozen_schema.get("feature_schema_sha256"),
        "target_schema_sha256": frozen_schema.get("target_schema_sha256"),
        "checklist_provenance_schema_sha256": frozen_schema.get("checklist_provenance_schema_sha256"),
        "legacy_feature_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "utc_day_partitions": list(_utc_days()),
        "days": day_rows,
        "physical_shards": all_shards,
        "physical_shard_count": len(all_shards),
        "all_physical_shards_at_or_below_96_mib": all(
            row["size_bytes"] <= MAX_PHYSICAL_SHARD_BYTES for row in all_shards
        ),
        "total_raw_episode_count": total_episodes,
        "total_refeature_record_count": total_records,
        "total_actor_visible_selection_frame_count": total_actor_frames,
        "total_forced_selection_frame_count": total_forced_frames,
        "source_day_group_seed_split_manifest_sha256": _public_digest(split),
        "card2vec": {
            "legacy_feature_source_preserved": True,
            "structured_residual_additive_only": True,
            "text_hash_exact_mechanics_authority": False,
            "bit_identity_receipt_required_before_candidate": True,
        },
        "training_eligible": False,
        "reason": "complete_collision_census_and_branch_support_receipt_required",
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "inzi_loading_or_execution_authority": False,
            "production_authority": False,
        },
    }
    receipt = {
        "schema": R298_REFEATURE_RECEIPT_SCHEMA,
        "version": 1,
        "status": "passed_complete_30_day_refeaturization_pending_census",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "refeature_manifest_sha256": _public_digest(manifest),
        "validated_30_day_raw_manifest_sha256": run["raw_manifest_sha256"],
        "frozen_schema_manifest_sha256": run["frozen_schema_manifest_sha256"],
        "source_day_group_seed_split_manifest_sha256": _public_digest(split),
        "all_30_utc_days_complete": True,
        "all_actor_visible_and_forced_frames_included": True,
        "resource_observation": telemetry.final(),
        "candidate_training_allowed": False,
        "bo250_allowed": False,
        "runtime_authority": {
            "elmo_only": True,
            "inzi_loading_or_execution_authority": False,
            "production_authority": False,
        },
    }
    _write_create_only_json(root / "SPLIT_MANIFEST.json", split)
    _write_create_only_json(root / "REFEATURE_MANIFEST.json", manifest)
    _write_create_only_json(root / "REFEATURE_RECEIPT.json", receipt)
    _write_create_only_json(
        root / "COMPLETE.json",
        {
            "schema": "poke_bot.alakazam_rule_derivative_r298_completion_marker/v1",
            "status": "complete_pending_census_gate",
            "run_contract_sha256": _public_digest(run),
            "refeature_manifest_sha256": _public_digest(manifest),
            "refeature_receipt_sha256": _public_digest(receipt),
            "create_only": True,
        },
    )
    return manifest, receipt, split


# ---------------------------------------------------------------------------
# Post-census candidate evidence.  These helpers deliberately materialize
# receipts only; they do not instantiate a trainer, mutate a checkpoint, or
# make any policy route runtime-visible.

R298_BRANCH_SUPPORT_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_branch_support_receipt/v1"
)
R298_RESOURCE_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_aggregate_resource_receipt/v1"
)
R298_CHECKPOINT_INSPECTION_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r298_safe_checkpoint_inspection/v1"
)
REQUIRED_FROZEN_COMPONENTS: Final = (
    "neural_backbone",
    "existing_heads",
    "fusion",
    "own_deck_routes",
    "matchup_adapters",
    "card2vec",
)


def _checkpoint_identity(
    value: Mapping[str, Any],
    *,
    field: str,
    require_local_file: bool,
) -> dict[str, Any]:
    """Normalize a receipt-bound checkpoint identity without substituting one.

    The contract intentionally leaves the r241/r274 checkpoint parameterized
    at launch.  Thus this accepts only a caller-provided identity and, for an
    actual candidate receipt, re-hashes the supplied local immutable file.
    """

    row = _strict_mapping(value, field=field)
    digest = _require_sha(row.get("sha256"), field=f"{field} sha256")
    size = _exact_int(row.get("size_bytes"), field=f"{field} size", minimum=1)
    result: dict[str, Any] = {"sha256": digest, "size_bytes": size}
    path_value = row.get("path")
    if path_value is None and require_local_file:
        raise RuleDerivativeTrainingError(f"{field} must provide an immutable checkpoint path")
    if path_value is not None:
        if not isinstance(path_value, str) or not path_value:
            raise RuleDerivativeTrainingError(f"{field} path is malformed")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise RuleDerivativeTrainingError(f"{field} is not a regular checkpoint file")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise RuleDerivativeTrainingError(f"{field} file identity drifted")
        result["path"] = str(path.resolve())
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def immutable_artifact_identity(
    path: Path | str,
    *,
    trusted_roots: Sequence[Path | str],
    field: str,
) -> dict[str, Any]:
    """Reopen/hash a regular artifact beneath an explicit trusted root.

    A pathname supplied in a receipt is not a trust boundary.  This helper
    rejects symlinks and arbitrary filesystem locations, fingerprints the
    exact byte object before and after use, and keeps the root list visible in
    the result.  It is suitable for parent/candidate checkpoints and the
    separately versioned derivative sidecar artifact.
    """

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RuleDerivativeTrainingError(f"{field} must be a regular non-symlink file")
    resolved = target.resolve()
    roots: list[Path] = []
    for raw_root in trusted_roots:
        root = Path(raw_root)
        if root.is_symlink() or not root.is_dir():
            raise RuleDerivativeTrainingError(f"trusted {field} root is not a regular directory")
        roots.append(root.resolve())
    if not roots or not any(_path_is_within(resolved, root) for root in roots):
        raise RuleDerivativeTrainingError(f"{field} lies outside the declared trusted artifact roots")
    before = target.stat()
    digest = _sha256_file(target)
    identity = {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": before.st_size,
        "trusted_root_paths": [str(root) for root in sorted(roots)],
    }
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or _sha256_file(target) != digest:
        raise RuleDerivativeTrainingError(f"{field} changed while its identity was inspected")
    return identity


def _require_same_trusted_artifact_identity(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    field: str,
) -> None:
    """Compare an attested artifact identity with a fresh trusted-root read.

    A SHA/size pair alone is insufficient here: a hand-authored audit could
    point at an arbitrary local path with matching-looking scalar fields.  The
    strict candidate path therefore binds pathname, bytes, and the exact
    declared trusted roots together.
    """

    expected_row = _strict_mapping(expected, field=f"{field} expected identity")
    observed_row = _strict_mapping(observed, field=f"{field} observed identity")
    for key in ("path", "sha256", "size_bytes"):
        if expected_row.get(key) != observed_row.get(key):
            raise RuleDerivativeTrainingError(f"{field} trusted artifact identity drifted")
    expected_roots = _strict_list(
        expected_row.get("trusted_root_paths"), field=f"{field} expected trusted roots"
    )
    observed_roots = _strict_list(
        observed_row.get("trusted_root_paths"), field=f"{field} observed trusted roots"
    )
    if (
        not expected_roots
        or any(not isinstance(root, str) or not root for root in expected_roots)
        or expected_roots != observed_roots
    ):
        raise RuleDerivativeTrainingError(f"{field} trusted root binding drifted")


def _trusted_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    *,
    trusted_roots: Sequence[Path | str],
    field: str,
) -> dict[str, Any]:
    """Require and rehash a checkpoint beneath an explicit trusted root."""

    declared = _checkpoint_identity(checkpoint, field=field, require_local_file=True)
    path_value = declared.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise RuleDerivativeTrainingError(f"{field} must provide an immutable checkpoint path")
    observed = immutable_artifact_identity(
        path_value, trusted_roots=trusted_roots, field=field
    )
    if observed["sha256"] != declared["sha256"] or observed["size_bytes"] != declared["size_bytes"]:
        raise RuleDerivativeTrainingError(f"{field} declared identity does not match rehashed bytes")
    return observed


def inspect_checkpoint_state_dict_create_only(
    checkpoint_path: Path | str,
    *,
    trusted_roots: Sequence[Path | str],
    checkpoint_role: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Safely derive a tensor state-dict from checkpoint bytes for an audit.

    This is inspection only: it uses PyTorch's ``weights_only=True`` loader,
    never imports a model class, never calls ``load_state_dict``, and rejects
    old PyTorch versions that would require unsafe pickle deserialization.
    Both file identity checks surround the load, so a parent/candidate cannot
    be swapped after an initial SHA check.
    """

    if checkpoint_role not in {"baseline_r274", "candidate"}:
        raise RuleDerivativeTrainingError("checkpoint inspection role must be baseline_r274 or candidate")
    identity = immutable_artifact_identity(
        checkpoint_path,
        trusted_roots=trusted_roots,
        field=f"{checkpoint_role} checkpoint",
    )
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - local audit host
        raise RuleDerivativeTrainingError(
            "safe checkpoint tensor inspection requires torch with weights_only support"
        ) from exc
    source = Path(identity["path"])
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # Do not fall back to unsafe pickle loading on an old runtime.
        raise RuleDerivativeTrainingError(
            "installed torch lacks safe weights_only checkpoint loading"
        ) from exc
    except Exception as exc:
        raise RuleDerivativeTrainingError(
            f"cannot safely inspect {checkpoint_role} checkpoint bytes"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuleDerivativeTrainingError("safe checkpoint payload is not a mapping")
    direct_tensor_map = all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in payload.items())
    wrappers = {
        key: value
        for key, value in payload.items()
        if key in {"state_dict", "model_state_dict", "policy_state_dict"}
        and isinstance(value, Mapping)
    }
    if direct_tensor_map:
        state_dict = payload
        wrapper_key = "direct_tensor_mapping"
    elif len(wrappers) == 1:
        wrapper_key, state_dict = next(iter(wrappers.items()))
        if not state_dict or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state_dict.items()
        ):
            raise RuleDerivativeTrainingError("checkpoint state-dict wrapper contains non-tensor values")
    else:
        raise RuleDerivativeTrainingError(
            "checkpoint must contain one explicit tensor-only state-dict wrapper"
        )
    # Re-hash after deserialization; a completed audit uses only stable bytes.
    after = immutable_artifact_identity(
        source,
        trusted_roots=trusted_roots,
        field=f"{checkpoint_role} checkpoint",
    )
    if after["sha256"] != identity["sha256"] or after["size_bytes"] != identity["size_bytes"]:
        raise RuleDerivativeTrainingError("checkpoint identity changed during safe tensor inspection")
    state = dict(state_dict)
    inspection = {
        "schema": R298_CHECKPOINT_INSPECTION_SCHEMA,
        "version": 1,
        "status": "passed_safe_weights_only_tensor_inspection",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "checkpoint_role": checkpoint_role,
        "checkpoint_identity": identity,
        "checkpoint_loader": "torch.load(map_location=cpu,weights_only=True)",
        "checkpoint_payload_state_dict_location": wrapper_key,
        "tensor_count": len(state),
        "model_import_or_load_state_dict_called": False,
        "checkpoint_mutated": False,
        "runtime_wired": False,
    }
    inspection["inspection_payload_sha256"] = _public_digest(inspection)
    return state, inspection


def _tensor_inventory(
    state_dict: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    """Make a byte-identity inventory for tensors/buffers only.

    State-dict values must be actual tensors or raw immutable byte buffers.
    Accepting arbitrary JSON objects here would make a future caller prove
    frozen model identity by a lossy serialization rather than its real bytes.
    """

    row = _strict_mapping(state_dict, field=field)
    entries: list[dict[str, Any]] = []
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - exercised on audit hosts
        torch = None  # type: ignore[assignment]
    for name in sorted(row):
        if not isinstance(name, str) or not name:
            raise RuleDerivativeTrainingError(f"{field} contains an invalid tensor name")
        value = row[name]
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            entries.append(
                {
                    "name": name,
                    "kind": "torch_tensor",
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "byte_identity_sha256": _byte_identity(tensor),
                }
            )
        elif isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            entries.append(
                {
                    "name": name,
                    "kind": "raw_buffer",
                    "dtype": "uint8",
                    "shape": [len(raw)],
                    "byte_identity_sha256": _byte_identity(raw),
                }
            )
        else:
            raise RuleDerivativeTrainingError(
                f"{field}.{name} is not an auditable tensor/buffer"
            )
    if not entries:
        raise RuleDerivativeTrainingError(f"{field} has no tensors or buffers")
    inventory = {
        "schema": "poke_bot.alakazam_rule_derivative_r298_tensor_inventory/v1",
        "tensor_count": len(entries),
        "tensors": entries,
    }
    inventory["inventory_sha256"] = _public_digest(inventory)
    return inventory


def _name_set(value: Iterable[str], *, field: str, nonempty: bool = True) -> set[str]:
    names = set(value)
    if (nonempty and not names) or any(not isinstance(name, str) or not name for name in names):
        raise RuleDerivativeTrainingError(f"{field} must contain nonempty tensor names")
    return names


def build_frozen_tensor_audit(
    *,
    baseline_state_dict: Mapping[str, Any],
    candidate_state_dict: Mapping[str, Any],
    baseline_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    frozen_component_tensor_names: Mapping[str, Sequence[str]],
    trainable_new_tensor_names: Sequence[str],
    require_local_checkpoint_files: bool = False,
) -> dict[str, Any]:
    """Prove every protected parent tensor is byte-identical in a candidate.

    The caller explicitly maps state-dict names to every frozen subsystem.
    That prevents a convenient but invalid assertion such as ``all frozen``
    with no evidence that Card2Vec, Fusion, OwnDeck, or adapters were in the
    compared checkpoint.  Candidate-only tensors are allowed exclusively when
    named in the trainable r298 sidecar inventory.
    """

    baseline_identity = _checkpoint_identity(
        baseline_checkpoint,
        field="baseline checkpoint",
        require_local_file=require_local_checkpoint_files,
    )
    candidate_identity = _checkpoint_identity(
        candidate_checkpoint,
        field="candidate checkpoint",
        require_local_file=require_local_checkpoint_files,
    )
    parent = _tensor_inventory(baseline_state_dict, field="baseline state dict")
    candidate = _tensor_inventory(candidate_state_dict, field="candidate state dict")
    parent_rows = {str(row["name"]): row for row in parent["tensors"]}
    candidate_rows = {str(row["name"]): row for row in candidate["tensors"]}
    component_rows = _strict_mapping(
        frozen_component_tensor_names, field="frozen component tensor names"
    )
    if set(component_rows) != set(REQUIRED_FROZEN_COMPONENTS):
        raise RuleDerivativeTrainingError(
            "frozen tensor audit must explicitly cover backbone, heads, Fusion, "
            "OwnDeck, matchup adapters, and Card2Vec"
        )
    component_inventory: dict[str, list[str]] = {}
    protected_names: set[str] = set()
    for component in REQUIRED_FROZEN_COMPONENTS:
        names = _name_set(
            _strict_list(component_rows[component], field=f"{component} tensor names"),
            field=f"{component} tensor names",
        )
        if not names <= set(parent_rows):
            raise RuleDerivativeTrainingError(
                f"{component} lists a tensor absent from the baseline state dict"
            )
        component_inventory[component] = sorted(names)
        protected_names.update(names)
    # A complete frozen-backbone derivative must protect every parent state
    # value.  The component map makes that exhaustive claim inspectable.
    if protected_names != set(parent_rows):
        omitted = sorted(set(parent_rows) - protected_names)
        raise RuleDerivativeTrainingError(
            "frozen component inventory omits baseline tensors: " + ", ".join(omitted[:8])
        )
    new_names = _name_set(
        trainable_new_tensor_names,
        field="new trainable tensor names",
        nonempty=False,
    )
    actual_new = set(candidate_rows) - set(parent_rows)
    if actual_new != new_names:
        raise RuleDerivativeTrainingError(
            "candidate tensor inventory has undeclared or missing new sidecar tensors"
        )
    overlap = new_names & set(parent_rows)
    if overlap:
        raise RuleDerivativeTrainingError("new trainable tensor names overlap the frozen parent")
    pairs: list[dict[str, Any]] = []
    changed: list[str] = []
    for name in sorted(protected_names):
        candidate_row = candidate_rows.get(name)
        if candidate_row is None:
            raise RuleDerivativeTrainingError("candidate checkpoint omits a frozen parent tensor")
        parent_row = parent_rows[name]
        same = (
            parent_row["kind"] == candidate_row["kind"]
            and parent_row["dtype"] == candidate_row["dtype"]
            and parent_row["shape"] == candidate_row["shape"]
            and parent_row["byte_identity_sha256"] == candidate_row["byte_identity_sha256"]
        )
        if not same:
            changed.append(name)
        pairs.append(
            {
                "name": name,
                "parent_byte_identity_sha256": parent_row["byte_identity_sha256"],
                "candidate_byte_identity_sha256": candidate_row["byte_identity_sha256"],
                "bit_identical": same,
            }
        )
    if changed:
        raise RuleDerivativeTrainingError(
            "candidate modifies protected tensors: " + ", ".join(changed[:8])
        )
    audit = {
        "schema": R298_TENSOR_AUDIT_SCHEMA,
        "version": 1,
        "status": "passed_complete_frozen_tensor_bit_identity",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "baseline_checkpoint": baseline_identity,
        "candidate_checkpoint": candidate_identity,
        "complete_parent_tensor_inventory_sha256": parent["inventory_sha256"],
        "complete_candidate_tensor_inventory_sha256": candidate["inventory_sha256"],
        "parent_tensor_count": parent["tensor_count"],
        "candidate_tensor_count": candidate["tensor_count"],
        "protected_tensor_count": len(protected_names),
        "protected_tensor_pairs": pairs,
        "frozen_component_tensor_names": component_inventory,
        "changed_protected_tensor_names": [],
        "candidate_new_trainable_tensor_names": sorted(new_names),
        "inventory_complete": True,
        "all_frozen_tensor_bit_identity": True,
        "parent_checkpoint_rewrite": False,
        "runtime_wired": False,
        "runtime_authority": {
            "elmo_only": True,
            "inzi_authority": False,
            "production_authority": False,
        },
    }
    audit["audit_payload_sha256"] = _public_digest(audit)
    return audit


def inspect_and_build_frozen_tensor_audit_create_only(
    *,
    baseline_checkpoint_path: Path | str,
    candidate_checkpoint_path: Path | str,
    trusted_roots: Sequence[Path | str],
    frozen_component_tensor_names: Mapping[str, Sequence[str]],
    trainable_new_tensor_names: Sequence[str],
) -> dict[str, Any]:
    """Produce frozen-tensor evidence directly from immutable checkpoint bytes.

    This is the only intended producer for a strict candidate handoff: callers
    do not provide a hand-written digest map.  The output retains both safe
    loader inspection receipts so a portable validator can identify the exact
    parent/candidate byte objects it must re-hash at the trusted root.
    """

    parent_state, parent_inspection = inspect_checkpoint_state_dict_create_only(
        baseline_checkpoint_path,
        trusted_roots=trusted_roots,
        checkpoint_role="baseline_r274",
    )
    candidate_state, candidate_inspection = inspect_checkpoint_state_dict_create_only(
        candidate_checkpoint_path,
        trusted_roots=trusted_roots,
        checkpoint_role="candidate",
    )
    parent_identity = _strict_mapping(
        parent_inspection.get("checkpoint_identity"), field="baseline checkpoint inspection identity"
    )
    candidate_identity = _strict_mapping(
        candidate_inspection.get("checkpoint_identity"), field="candidate checkpoint inspection identity"
    )
    audit = build_frozen_tensor_audit(
        baseline_state_dict=parent_state,
        candidate_state_dict=candidate_state,
        baseline_checkpoint=parent_identity,
        candidate_checkpoint=candidate_identity,
        frozen_component_tensor_names=frozen_component_tensor_names,
        trainable_new_tensor_names=trainable_new_tensor_names,
        require_local_checkpoint_files=True,
    )
    # Reconstruct after adding producer evidence so the checksum is the one
    # validators consume; the base audit itself already proved byte pairs.
    enriched = dict(audit)
    enriched.pop("audit_payload_sha256", None)
    enriched["baseline_checkpoint_inspection_sha256"] = _public_digest(parent_inspection)
    enriched["candidate_checkpoint_inspection_sha256"] = _public_digest(candidate_inspection)
    enriched["checkpoint_inspection_producer"] = {
        "schema": R298_CHECKPOINT_INSPECTION_SCHEMA,
        "safe_weights_only_required": True,
        "derives_state_dict_from_checkpoint_bytes": True,
        "baseline": parent_inspection,
        "candidate": candidate_inspection,
    }
    enriched["audit_payload_sha256"] = _public_digest(enriched)
    return enriched


def _validated_tensor_audit(
    value: Mapping[str, Any], *, require_checkpoint_byte_producer: bool = False
) -> Mapping[str, Any]:
    audit = _strict_mapping(value, field="frozen tensor audit")
    if audit.get("schema") != R298_TENSOR_AUDIT_SCHEMA or audit.get("status") != "passed_complete_frozen_tensor_bit_identity":
        raise RuleDerivativeTrainingError("frozen tensor audit is not a passed r298 audit")
    expected = dict(audit)
    claimed = expected.pop("audit_payload_sha256", None)
    if claimed != _public_digest(expected):
        raise RuleDerivativeTrainingError("frozen tensor audit checksum drifted")
    if audit.get("inventory_complete") is not True or audit.get("all_frozen_tensor_bit_identity") is not True:
        raise RuleDerivativeTrainingError("frozen tensor audit does not prove complete bit identity")
    if audit.get("changed_protected_tensor_names") != []:
        raise RuleDerivativeTrainingError("frozen tensor audit reports protected changes")
    components = _strict_mapping(audit.get("frozen_component_tensor_names"), field="audit components")
    if set(components) != set(REQUIRED_FROZEN_COMPONENTS):
        raise RuleDerivativeTrainingError("frozen tensor audit component coverage is incomplete")
    for field in (
        "complete_parent_tensor_inventory_sha256",
        "complete_candidate_tensor_inventory_sha256",
    ):
        _require_sha(audit.get(field), field=f"frozen tensor audit {field}")
    if require_checkpoint_byte_producer:
        producer = _strict_mapping(
            audit.get("checkpoint_inspection_producer"),
            field="checkpoint inspection producer",
        )
        if (
            producer.get("schema") != R298_CHECKPOINT_INSPECTION_SCHEMA
            or producer.get("safe_weights_only_required") is not True
            or producer.get("derives_state_dict_from_checkpoint_bytes") is not True
        ):
            raise RuleDerivativeTrainingError(
                "strict frozen tensor audit lacks a safe checkpoint-byte inspection producer"
            )
        for role, expected_identity in (
            ("baseline", audit.get("baseline_checkpoint")),
            ("candidate", audit.get("candidate_checkpoint")),
        ):
            inspection = _strict_mapping(producer.get(role), field=f"{role} checkpoint inspection")
            identity = _strict_mapping(
                inspection.get("checkpoint_identity"), field=f"{role} checkpoint identity"
            )
            audit_identity = _strict_mapping(
                expected_identity, field=f"audit {role} checkpoint"
            )
            for key in ("path", "sha256", "size_bytes"):
                if identity.get(key) != audit_identity.get(key):
                    raise RuleDerivativeTrainingError(
                        "checkpoint-byte inspection identity does not match tensor audit"
                    )
            roots = _strict_list(
                identity.get("trusted_root_paths"),
                field=f"{role} checkpoint inspection trusted roots",
            )
            audit_roots = _strict_list(
                audit_identity.get("trusted_root_paths"),
                field=f"audit {role} checkpoint trusted roots",
            )
            if (
                not roots
                or any(not isinstance(root, str) or not root for root in roots)
                or roots != audit_roots
            ):
                raise RuleDerivativeTrainingError(
                    "checkpoint-byte inspection lacks an exact trusted-root binding"
                )
            if inspection.get("checkpoint_loader") != "torch.load(map_location=cpu,weights_only=True)":
                raise RuleDerivativeTrainingError("checkpoint inspection did not use safe weights-only loading")
    return audit


def build_card2vec_nonregression_receipt(
    *,
    tensor_audit: Mapping[str, Any],
    catalog_binding: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    baseline_card2vec_input_schema_sha256: str,
    candidate_card2vec_input_schema_sha256: str,
    baseline_card2vec_input_values_byte_identity_sha256: str,
    candidate_card2vec_input_values_byte_identity_sha256: str,
    baseline_card2vec_outputs: Any,
    candidate_card2vec_outputs_with_new_path_off: Any,
) -> dict[str, Any]:
    """Build evidence for the complete frozen Card2Vec non-regression fence."""

    audit = _validated_tensor_audit(tensor_audit)
    try:
        validate_frozen_schema_gate(schema_manifest, zero_bypass_receipt)
    except Exception as exc:
        raise RuleDerivativeTrainingError("Card2Vec receipt requires a passed frozen zero-bypass gate") from exc
    binding = _strict_mapping(catalog_binding, field="sealed structured catalog binding")
    if binding.get("schema") != R298_SEALED_CATALOG_BINDING_SCHEMA:
        raise RuleDerivativeTrainingError("Card2Vec receipt catalog binding schema drifted")
    if binding.get("card2vec_preserved_unchanged") is not True:
        raise RuleDerivativeTrainingError("sealed catalog does not attest unchanged Card2Vec")
    if binding.get("csv_or_text_hash_mechanics_authority") is not False:
        raise RuleDerivativeTrainingError("new Card2Vec receipt would use text/CSV as mechanics authority")
    component_names = _strict_list(
        _strict_mapping(audit.get("frozen_component_tensor_names"), field="audit components").get("card2vec"),
        field="Card2Vec tensor names",
    )
    pairs = {
        row.get("name"): row
        for row in _strict_list(audit.get("protected_tensor_pairs"), field="protected tensor pairs")
        if isinstance(row, Mapping)
    }
    if not component_names or any(
        not isinstance(pairs.get(name), Mapping) or pairs[name].get("bit_identical") is not True
        for name in component_names
    ):
        raise RuleDerivativeTrainingError("Card2Vec tensors are not completely bit-identical")
    input_schema_before = _require_sha(
        baseline_card2vec_input_schema_sha256, field="baseline Card2Vec input schema"
    )
    input_schema_after = _require_sha(
        candidate_card2vec_input_schema_sha256, field="candidate Card2Vec input schema"
    )
    input_values_before = _require_sha(
        baseline_card2vec_input_values_byte_identity_sha256,
        field="baseline Card2Vec input values",
    )
    input_values_after = _require_sha(
        candidate_card2vec_input_values_byte_identity_sha256,
        field="candidate Card2Vec input values",
    )
    output_before = _byte_identity(baseline_card2vec_outputs)
    output_after = _byte_identity(candidate_card2vec_outputs_with_new_path_off)
    if input_schema_before != input_schema_after or input_values_before != input_values_after:
        raise RuleDerivativeTrainingError("Card2Vec input schema or values changed")
    if output_before != output_after:
        raise RuleDerivativeTrainingError("Card2Vec output changed with the new path off")
    receipt = {
        "schema": R298_CARD2VEC_EVIDENCE_SCHEMA,
        "version": 1,
        "status": "passed_complete_card2vec_non_regression",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "frozen_schema_manifest_sha256": _public_digest(schema_manifest),
        "zero_bypass_receipt_sha256": _public_digest(zero_bypass_receipt),
        "frozen_tensor_audit_sha256": _public_digest(audit),
        "card2vec_tensor_names": list(component_names),
        "card2vec_parameter_and_buffer_bit_identity": True,
        "card2vec_input_schema_and_values_unchanged": True,
        "baseline_card2vec_input_schema_sha256": input_schema_before,
        "card2vec_input_values_byte_identity_sha256": input_values_before,
        "card2vec_runtime_output_and_behavior_parity_with_new_path_off": True,
        "baseline_card2vec_output_byte_identity_sha256": output_before,
        "new_path_off_card2vec_output_byte_identity_sha256": output_after,
        "sealed_structured_catalog_binding_sha256": _public_digest(binding),
        "structured_rule_vectors_sha256": binding.get("structured_rule_vectors_sha256"),
        "new_structured_residual_is_additive_alongside_or_after_card2vec": True,
        "new_structured_residual_replaces_card2vec": False,
        "new_structured_residual_zero_initialized_and_default_off": True,
        "no_text_hash_exact_mechanics_or_target_provenance_in_new_residual": True,
        "runtime_wired": False,
        "inzi_or_production_authority": False,
    }
    receipt["receipt_payload_sha256"] = _public_digest(receipt)
    return receipt


def validate_aggregate_resource_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require evidence-backed aggregate Elmo resource telemetry, not a flag."""

    row = _strict_mapping(value, field="aggregate resource receipt")
    if row.get("schema") != R298_RESOURCE_RECEIPT_SCHEMA or row.get("status") != "passed":
        raise RuleDerivativeTrainingError("aggregate resource receipt has not passed")
    if row.get("goal_contract_sha256") != R298_CONTRACT_SHA256 or row.get("goal_gateway_sha256") != R298_GOAL_SHA256:
        raise RuleDerivativeTrainingError("aggregate resource receipt binds stale authority")
    if row.get("execution_host_role") != "elmo" or row.get("production_or_inzi_authority") is not False:
        raise RuleDerivativeTrainingError("resource receipt is not an Elmo-only nonproduction job")
    method = row.get("experiment_ram_measurement_method")
    if not isinstance(method, str) or "aggregate" not in method:
        raise RuleDerivativeTrainingError("resource receipt does not report aggregate RAM accounting")
    peak = _exact_int(row.get("experiment_ram_peak_bytes"), field="aggregate experiment RAM peak", minimum=0)
    if peak > R298_EXPERIMENT_RAM_BYTES:
        raise RuleDerivativeTrainingError("resource receipt exceeds the 96-GiB experiment RAM cap")
    if row.get("memory_heavy_phase_concurrency_exact") != 1:
        raise RuleDerivativeTrainingError("resource receipt does not serialize memory-heavy phases")
    for field in (
        "cpu_process_or_utilization_peak",
        "gpu_memory_peak_bytes_per_device_or_not_applicable",
        "gpu_utilization_peak_per_device_or_not_applicable",
    ):
        if field not in row:
            raise RuleDerivativeTrainingError(f"resource receipt misses {field}")
    payload = dict(row)
    claimed = payload.pop("receipt_payload_sha256", None)
    if claimed != _public_digest(payload):
        raise RuleDerivativeTrainingError("aggregate resource receipt checksum drifted")
    return row


def build_training_gate_report(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    refeature_manifest: Mapping[str, Any] | None = None,
    refeature_receipt: Mapping[str, Any] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    collision_receipt: Mapping[str, Any] | None = None,
    branch_support_receipt: Mapping[str, Any] | None = None,
    transfer_parity_receipt: Mapping[str, Any] | None = None,
    aggregate_resource_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the receipt gates without launching or authorizing training.

    Missing inputs produce an explicit blocked report.  A passing report is a
    narrow prerequisite for a later candidate receipt; it is never a training
    command and it cannot turn a zero-gated route on.
    """

    blockers: list[str] = []
    try:
        validate_raw_corpus_binding(raw_manifest, raw_receipt)
    except Exception as exc:
        blockers.append(f"raw_corpus:{type(exc).__name__}")
    try:
        frozen_sha, zero_sha = validate_frozen_schema_gate(schema_manifest, zero_bypass_receipt)
    except Exception as exc:
        blockers.append(f"frozen_schema_or_zero_bypass:{type(exc).__name__}")
        frozen_sha = None
        zero_sha = None
    raw_sha = _public_digest(raw_manifest)
    refeature_sha: str | None = None
    split_sha: str | None = None
    collision_sha: str | None = None
    branch_sha: str | None = None
    transfer_sha: str | None = None
    resource_sha: str | None = None
    if refeature_manifest is None or refeature_receipt is None or split_manifest is None:
        blockers.append("complete_30_day_refeaturization_and_disjoint_splits_missing")
    else:
        refeature_sha = _public_digest(refeature_manifest)
        split_sha = _public_digest(split_manifest)
        if (
            refeature_manifest.get("schema") != R298_REFEATURE_MANIFEST_SCHEMA
            or refeature_manifest.get("status") != "complete_30_day_create_only_refeaturization_pending_census_gate"
            or refeature_manifest.get("validated_30_day_raw_manifest_sha256") != raw_sha
            or refeature_manifest.get("frozen_schema_manifest_sha256") != frozen_sha
            or refeature_manifest.get("zero_bypass_receipt_sha256") != zero_sha
        ):
            blockers.append("refeaturization_manifest_binding_or_status_invalid")
        if (
            refeature_receipt.get("schema") != R298_REFEATURE_RECEIPT_SCHEMA
            or refeature_receipt.get("status") != "passed_complete_30_day_refeaturization_pending_census"
            or refeature_receipt.get("refeature_manifest_sha256") != refeature_sha
            or refeature_receipt.get("candidate_training_allowed") is not False
        ):
            blockers.append("refeaturization_receipt_invalid")
        if (
            split_manifest.get("schema") != R298_SPLIT_MANIFEST_SCHEMA
            or split_manifest.get("status") != "passed_source_day_group_seed_disjoint"
            or split_manifest.get("validated_30_day_raw_manifest_sha256") != raw_sha
            or split_manifest.get("source_day_group_seed_disjoint") is not True
            or split_manifest.get("all_30_utc_days_assigned_exactly_once") is not True
            or split_manifest.get("cross_split_overlap") != {}
        ):
            blockers.append("source_day_group_seed_split_receipt_invalid")
    if collision_receipt is None:
        blockers.append("complete_collision_census_receipt_missing")
    else:
        collision_sha = _public_digest(collision_receipt)
        gate = collision_receipt.get("frozen_schema_gate")
        if (
            collision_receipt.get("schema") != R298_COLLISION_RECEIPT_SCHEMA
            or collision_receipt.get("pass") is not True
            or collision_receipt.get("raw_expert_corpus_manifest_sha256") != raw_sha
            or not isinstance(gate, Mapping)
            or gate.get("frozen_schema_manifest_sha256") != frozen_sha
            or gate.get("zero_bypass_receipt_sha256") != zero_sha
        ):
            blockers.append("collision_census_receipt_invalid")
    if branch_support_receipt is None:
        blockers.append("census_branch_support_receipt_missing")
    else:
        branch_sha = _public_digest(branch_support_receipt)
        eligible = branch_support_receipt.get("eligible_trainable_branches")
        if (
            branch_support_receipt.get("schema") != R298_BRANCH_SUPPORT_RECEIPT_SCHEMA
            or branch_support_receipt.get("status") != "passed_census_supported_branch_adjudication"
            or branch_support_receipt.get("collision_receipt_sha256") != collision_sha
            or branch_support_receipt.get("frozen_schema_manifest_sha256") != frozen_sha
            or branch_support_receipt.get("unsupported_branches_exact_zero_and_inert") is not True
            or not isinstance(eligible, Sequence)
            or isinstance(eligible, (str, bytes, bytearray))
        ):
            blockers.append("census_branch_support_receipt_invalid")
    if transfer_parity_receipt is None:
        blockers.append("final_quarantined_inzi_shard_parity_receipt_missing")
    else:
        transfer_sha = _public_digest(transfer_parity_receipt)
        payload = dict(transfer_parity_receipt)
        claimed = payload.pop("receipt_payload_sha256", None)
        if (
            transfer_parity_receipt.get("schema")
            != "poke_bot.alakazam_elmo_refeaturization_transfer_parity_receipt/v1"
            or transfer_parity_receipt.get("status") != "passed_complete_source_remote_parity"
            or transfer_parity_receipt.get("goal_contract_sha256") != R298_CONTRACT_SHA256
            or transfer_parity_receipt.get("goal_revision") != 4
            or transfer_parity_receipt.get("source_remote_parity_passed") is not True
            or transfer_parity_receipt.get("inzi_loading_or_execution_authority") is not False
            or claimed != _public_digest(payload)
        ):
            blockers.append("final_quarantined_inzi_shard_parity_receipt_invalid")
    if aggregate_resource_receipt is None:
        blockers.append("aggregate_elmo_resource_receipt_missing")
    else:
        try:
            validate_aggregate_resource_receipt(aggregate_resource_receipt)
            resource_sha = _public_digest(aggregate_resource_receipt)
        except Exception as exc:
            blockers.append(f"aggregate_elmo_resource_receipt_invalid:{type(exc).__name__}")
    report = {
        "schema": R298_TRAINING_GATE_SCHEMA,
        "version": 1,
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "status": (
            "passed_all_receipt_gates_candidate_materialization_may_be_considered"
            if not blockers
            else "blocked_receipt_gates_incomplete_or_invalid"
        ),
        "validated_30_day_raw_manifest_sha256": raw_sha,
        "frozen_schema_manifest_sha256": frozen_sha,
        "zero_bypass_receipt_sha256": zero_sha,
        "refeature_manifest_sha256": refeature_sha,
        "split_manifest_sha256": split_sha,
        "collision_receipt_sha256": collision_sha,
        "branch_support_receipt_sha256": branch_sha,
        "transfer_parity_receipt_sha256": transfer_sha,
        "aggregate_resource_receipt_sha256": resource_sha,
        "blockers": blockers,
        "candidate_training_allowed": not blockers,
        "bo250_allowed": False,
        "eligible_branches_may_be_nonzero_only_after_separate_candidate_training": not blockers,
        "runtime_wired": False,
        "elmo_only_nonproduction": True,
        "inzi_or_production_authority": False,
    }
    report["report_payload_sha256"] = _public_digest(report)
    return report


def _validated_training_gate(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = _strict_mapping(value, field="training gate report")
    payload = dict(report)
    claimed = payload.pop("report_payload_sha256", None)
    if claimed != _public_digest(payload):
        raise RuleDerivativeTrainingError("training gate report checksum drifted")
    if (
        report.get("schema") != R298_TRAINING_GATE_SCHEMA
        or report.get("status") != "passed_all_receipt_gates_candidate_materialization_may_be_considered"
        or report.get("candidate_training_allowed") is not True
        or report.get("blockers") != []
    ):
        raise RuleDerivativeTrainingError("candidate receipt requires a passed complete training gate report")
    return report


def build_candidate_validation_receipt(
    *,
    candidate_checkpoint: Mapping[str, Any],
    baseline_r274_checkpoint: Mapping[str, Any],
    tensor_audit: Mapping[str, Any],
    card2vec_nonregression_receipt: Mapping[str, Any],
    training_gate_report: Mapping[str, Any],
    aggregate_resource_receipt: Mapping[str, Any],
    derivative_sidecar_path: Path | str,
    trusted_artifact_roots: Sequence[Path | str],
    candidate_id: str = "alakazam-elmo-public-rule-derivative-g1",
    baseline_r274_candidate_id: str = "alakazam-new-list-direct-policy-r274",
    require_local_checkpoint_files: bool = True,
) -> dict[str, Any]:
    """Make a receipt for an already-created candidate; never create one.

    This does not run training and cannot be used before the input candidate
    file already exists.  It simply rejects candidate evidence unless every
    frozen, Card2Vec, corpus/census, transfer, resource, and zero-bypass gate
    has independently passed.
    """

    if not isinstance(candidate_id, str) or not candidate_id:
        raise RuleDerivativeTrainingError("candidate id is malformed")
    if not isinstance(baseline_r274_candidate_id, str) or not baseline_r274_candidate_id:
        raise RuleDerivativeTrainingError("baseline candidate id is malformed")
    if not require_local_checkpoint_files:
        raise RuleDerivativeTrainingError(
            "strict candidate validation always requires trusted local checkpoint bytes"
        )
    candidate = _trusted_checkpoint_identity(
        candidate_checkpoint,
        trusted_roots=trusted_artifact_roots,
        field="candidate checkpoint",
    )
    baseline = _trusted_checkpoint_identity(
        baseline_r274_checkpoint,
        trusted_roots=trusted_artifact_roots,
        field="baseline r274 checkpoint",
    )
    audit = _validated_tensor_audit(
        tensor_audit, require_checkpoint_byte_producer=True
    )
    _require_same_trusted_artifact_identity(
        _strict_mapping(audit.get("candidate_checkpoint"), field="audit candidate checkpoint"),
        candidate,
        field="candidate checkpoint audit",
    )
    _require_same_trusted_artifact_identity(
        _strict_mapping(audit.get("baseline_checkpoint"), field="audit baseline checkpoint"),
        baseline,
        field="baseline checkpoint audit",
    )
    card2vec = _strict_mapping(card2vec_nonregression_receipt, field="Card2Vec receipt")
    card_payload = dict(card2vec)
    claimed_card = card_payload.pop("receipt_payload_sha256", None)
    if (
        card2vec.get("schema") != R298_CARD2VEC_EVIDENCE_SCHEMA
        or card2vec.get("status") != "passed_complete_card2vec_non_regression"
        or claimed_card != _public_digest(card_payload)
        or card2vec.get("frozen_tensor_audit_sha256") != _public_digest(audit)
        or card2vec.get("new_structured_residual_replaces_card2vec") is not False
    ):
        raise RuleDerivativeTrainingError("Card2Vec non-regression receipt is invalid")
    gate = _validated_training_gate(training_gate_report)
    resource = validate_aggregate_resource_receipt(aggregate_resource_receipt)
    if gate.get("aggregate_resource_receipt_sha256") != _public_digest(resource):
        raise RuleDerivativeTrainingError("training gate does not bind candidate resource receipt")
    derivative_sidecar_identity = immutable_artifact_identity(
        derivative_sidecar_path,
        trusted_roots=trusted_artifact_roots,
        field="derivative sidecar artifact",
    )
    producer = _strict_mapping(
        audit.get("checkpoint_inspection_producer"), field="checkpoint inspection producer"
    )
    baseline_inspection_identity = _strict_mapping(
        _strict_mapping(producer.get("baseline"), field="baseline inspection").get("checkpoint_identity"),
        field="baseline inspection identity",
    )
    candidate_inspection_identity = _strict_mapping(
        _strict_mapping(producer.get("candidate"), field="candidate inspection").get("checkpoint_identity"),
        field="candidate inspection identity",
    )
    _require_same_trusted_artifact_identity(
        baseline_inspection_identity, baseline, field="baseline inspection"
    )
    _require_same_trusted_artifact_identity(
        candidate_inspection_identity, candidate, field="candidate inspection"
    )
    receipt = {
        "schema": R298_CANDIDATE_RECEIPT_SCHEMA,
        "version": 1,
        "status": "passed_frozen_backbone_elmo_only_candidate_validation",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "candidate_id": candidate_id,
        "candidate_checkpoint_sha256": candidate["sha256"],
        "candidate_checkpoint_size_bytes": candidate["size_bytes"],
        "candidate_checkpoint_path": candidate["path"],
        "candidate_checkpoint_identity": dict(candidate),
        "frozen_backbone_true": True,
        "existing_heads_frozen_true": True,
        "fusion_frozen_true": True,
        "own_deck_routes_frozen_true": True,
        "matchup_adapters_frozen_true": True,
        "baseline_r274_candidate_id": baseline_r274_candidate_id,
        "baseline_r274_checkpoint_sha256": baseline["sha256"],
        "baseline_r274_checkpoint_size_bytes": baseline["size_bytes"],
        "baseline_r274_checkpoint_path": baseline["path"],
        "baseline_r274_checkpoint_identity": dict(baseline),
        "frozen_tensor_audit_checkpoint_inspection_producer": dict(producer),
        "derivative_sidecar_identity": derivative_sidecar_identity,
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "all_frozen_tensor_bit_identity": True,
        "frozen_tensor_audit_sha256": _public_digest(audit),
        "card2vec_non_regression_receipt_sha256": _public_digest(card2vec),
        "training_gate_report_sha256": _public_digest(gate),
        "aggregate_resource_receipt_sha256": _public_digest(resource),
        "validated_30_day_raw_manifest_sha256": gate.get("validated_30_day_raw_manifest_sha256"),
        "frozen_schema_manifest_sha256": gate.get("frozen_schema_manifest_sha256"),
        "zero_bypass_receipt_sha256": gate.get("zero_bypass_receipt_sha256"),
        "collision_receipt_sha256": gate.get("collision_receipt_sha256"),
        "branch_support_receipt_sha256": gate.get("branch_support_receipt_sha256"),
        "transfer_parity_receipt_sha256": gate.get("transfer_parity_receipt_sha256"),
        "elmo_only_nonproduction_true": True,
        "runtime_wired": False,
        "inzi_or_production_authority": False,
        "candidate_training_or_bo250_authority": False,
    }
    receipt["receipt_payload_sha256"] = _public_digest(receipt)
    return receipt


def validate_candidate_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    trusted_artifact_roots: Sequence[Path | str],
) -> Mapping[str, Any]:
    """Validate a candidate receipt by reopening all byte-bearing artifacts.

    There is intentionally no portable ``shape-only`` success mode for a
    candidate receipt.  Parent/candidate checkpoint and sidecar evidence must
    be rehashed beneath caller-supplied trusted roots, then matched against the
    safe checkpoint-byte producer embedded in the receipt.
    """

    row = _strict_mapping(receipt, field="candidate validation receipt")
    payload = dict(row)
    claimed = payload.pop("receipt_payload_sha256", None)
    if claimed != _public_digest(payload):
        raise RuleDerivativeTrainingError("candidate receipt checksum drifted")
    required = {
        "schema": R298_CANDIDATE_RECEIPT_SCHEMA,
        "status": "passed_frozen_backbone_elmo_only_candidate_validation",
        "goal_contract_path": R298_CONTRACT_PATH,
        "goal_contract_sha256": R298_CONTRACT_SHA256,
        "goal_gateway_sha256": R298_GOAL_SHA256,
        "frozen_backbone_true": True,
        "existing_heads_frozen_true": True,
        "fusion_frozen_true": True,
        "own_deck_routes_frozen_true": True,
        "matchup_adapters_frozen_true": True,
        "all_frozen_tensor_bit_identity": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "elmo_only_nonproduction_true": True,
        "runtime_wired": False,
        "inzi_or_production_authority": False,
        "candidate_training_or_bo250_authority": False,
    }
    if any(row.get(field) != expected for field, expected in required.items()):
        raise RuleDerivativeTrainingError("candidate receipt violates frozen/nonproduction contract")
    for prefix in ("candidate", "baseline_r274"):
        digest = _require_sha(row.get(f"{prefix}_checkpoint_sha256"), field=f"{prefix} checkpoint SHA")
        size = _exact_int(row.get(f"{prefix}_checkpoint_size_bytes"), field=f"{prefix} checkpoint size", minimum=1)
        identity = _strict_mapping(row.get(f"{prefix}_checkpoint_identity"), field=f"{prefix} checkpoint identity")
        if identity.get("sha256") != digest or identity.get("size_bytes") != size:
            raise RuleDerivativeTrainingError("candidate receipt checkpoint scalar/identity mismatch")
        if not isinstance(identity.get("path"), str) or not identity.get("path"):
            raise RuleDerivativeTrainingError("candidate receipt checkpoint identity lacks a path")
        if row.get(f"{prefix}_checkpoint_path") != identity.get("path"):
            raise RuleDerivativeTrainingError("candidate receipt checkpoint path/identity mismatch")
    sidecar = _strict_mapping(row.get("derivative_sidecar_identity"), field="derivative sidecar identity")
    _require_sha(sidecar.get("sha256"), field="derivative sidecar SHA")
    _exact_int(sidecar.get("size_bytes"), field="derivative sidecar size", minimum=1)
    if not isinstance(sidecar.get("path"), str) or not sidecar.get("path"):
        raise RuleDerivativeTrainingError("candidate receipt sidecar identity lacks a path")
    for field in (
        "frozen_tensor_audit_sha256",
        "card2vec_non_regression_receipt_sha256",
        "training_gate_report_sha256",
        "aggregate_resource_receipt_sha256",
        "validated_30_day_raw_manifest_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "collision_receipt_sha256",
        "branch_support_receipt_sha256",
        "transfer_parity_receipt_sha256",
    ):
        _require_sha(row.get(field), field=f"candidate receipt {field}")
    if not trusted_artifact_roots:
        raise RuleDerivativeTrainingError("candidate receipt requires explicit trusted artifact roots")
    observed_checkpoints: dict[str, Mapping[str, Any]] = {}
    for field, label in (
        ("candidate_checkpoint_identity", "candidate checkpoint"),
        ("baseline_r274_checkpoint_identity", "baseline r274 checkpoint"),
    ):
        expected = _strict_mapping(row.get(field), field=field)
        observed = immutable_artifact_identity(
            expected["path"], trusted_roots=trusted_artifact_roots, field=label
        )
        _require_same_trusted_artifact_identity(expected, observed, field=label)
        observed_checkpoints[field] = observed
    observed_sidecar = immutable_artifact_identity(
        sidecar["path"], trusted_roots=trusted_artifact_roots, field="derivative sidecar"
    )
    _require_same_trusted_artifact_identity(sidecar, observed_sidecar, field="derivative sidecar")
    producer = _strict_mapping(
        row.get("frozen_tensor_audit_checkpoint_inspection_producer"),
        field="candidate receipt checkpoint inspection producer",
    )
    if (
        producer.get("schema") != R298_CHECKPOINT_INSPECTION_SCHEMA
        or producer.get("safe_weights_only_required") is not True
        or producer.get("derives_state_dict_from_checkpoint_bytes") is not True
    ):
        raise RuleDerivativeTrainingError("candidate receipt lacks safe checkpoint inspection evidence")
    for producer_role, receipt_field in (
        ("candidate", "candidate_checkpoint_identity"),
        ("baseline", "baseline_r274_checkpoint_identity"),
    ):
        inspection = _strict_mapping(
            producer.get(producer_role), field=f"candidate receipt {producer_role} inspection"
        )
        if inspection.get("checkpoint_loader") != "torch.load(map_location=cpu,weights_only=True)":
            raise RuleDerivativeTrainingError("candidate receipt inspection used an unsafe loader")
        inspected_identity = _strict_mapping(
            inspection.get("checkpoint_identity"),
            field=f"candidate receipt {producer_role} inspection identity",
        )
        _require_same_trusted_artifact_identity(
            inspected_identity,
            observed_checkpoints[receipt_field],
            field=f"candidate receipt {producer_role} inspection",
        )
    return row


def write_frozen_tensor_audit_create_only(path: Path | str, audit: Mapping[str, Any]) -> str:
    """Write only a checkpoint-byte-derived frozen-tensor evidence artifact."""

    _validated_tensor_audit(audit, require_checkpoint_byte_producer=True)
    target = Path(path)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuleDerivativeTrainingError("frozen tensor audit parent must be a regular existing directory")
    _write_create_only_json(target, audit)
    return _sha256_file(target)


def write_card2vec_nonregression_receipt_create_only(
    path: Path | str, receipt: Mapping[str, Any]
) -> str:
    """Persist one checksum-bound Card2Vec evidence receipt without overwrite."""

    row = _strict_mapping(receipt, field="Card2Vec non-regression receipt")
    payload = dict(row)
    claimed = payload.pop("receipt_payload_sha256", None)
    if (
        row.get("schema") != R298_CARD2VEC_EVIDENCE_SCHEMA
        or row.get("status") != "passed_complete_card2vec_non_regression"
        or claimed != _public_digest(payload)
    ):
        raise RuleDerivativeTrainingError("Card2Vec non-regression receipt is invalid")
    target = Path(path)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuleDerivativeTrainingError("Card2Vec receipt parent must be a regular existing directory")
    _write_create_only_json(target, row)
    return _sha256_file(target)


def write_candidate_validation_receipt_create_only(
    path: Path | str,
    receipt: Mapping[str, Any],
    *,
    trusted_artifact_roots: Sequence[Path | str],
) -> str:
    """Persist a strict, re-hashed candidate receipt exactly once."""

    validate_candidate_validation_receipt(
        receipt, trusted_artifact_roots=trusted_artifact_roots
    )
    target = Path(path)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise RuleDerivativeTrainingError("candidate receipt parent must be a regular existing directory")
    _write_create_only_json(target, receipt)
    return _sha256_file(target)
