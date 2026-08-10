"""Static contract coverage for the dormant guide-agnostic Guide2Vec pipeline.

Revision 226 deliberately leaves the existing r212 experiment byte-for-byte
preserved but retires its launch authority.  These tests cover only the typed
owner contract and isolated fail-closed boundaries.  They use only temporary
synthetic manifests/chunks and an in-memory head; they do not apply gradients,
start a service, attach runtime, or touch any BO1000 runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from poke_bot.guide2vec import FrozenBaseIdentity, Guide2VecConfig, Guide2VecHead
from poke_bot.guide2vec_activation import (
    Guide2VecActivationError,
    Guide2VecTrainingGrant,
    SealedChunkSet,
    VerifiedContentRef,
    joined_chunk_pair_sha256,
    sealed_chunk_set_sha256,
    validate_activation_receipt,
)
from poke_bot.guide2vec_contracts import (
    CAUSAL_PARTITION_PAYLOAD_SCHEMA,
    CAUSAL_SPLIT_SCHEMA,
    COMPATIBILITY_FIELDS,
    DIGEST_TENSOR_LAYOUT,
    FROZEN_LATENT_STORE_SCHEMA,
    FROZEN_LATENT_TENSORS,
    GUIDE_LABEL_FIELDS,
    GUIDE_LABEL_OVERLAY_SCHEMA,
    INERT_AUTHORITY_FIELDS,
    STAGE_INDEX_SCHEMA,
    STAGE_JOIN_KEY_FIELDS,
    TRAINING_INPUT_FIELDS,
    TRAINING_INPUT_IDENTITY_FIELDS,
    TRAINING_MANIFEST_SCHEMA,
    Guide2VecContractError,
    digest_tensor_bytes,
    load_and_validate_training_manifest,
    stage_key_sha256,
    validate_training_manifest,
)
from poke_bot.guide2vec_training import (
    LABEL_CHUNK_SCHEMA,
    LATENT_CHUNK_SCHEMA,
    ContentChunkRef,
    FrozenGuide2VecRuntime,
    Guide2VecTrainingError,
    JoinedChunkRef,
    load_joined_chunk,
    load_latent_chunk,
    rerank_with_frozen_runtime,
    train_from_joined_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "state/guide2vec-general-training-pipeline-r226.json"
R212_PATH = ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json"
REQUIREMENTS_PATH = ROOT / "ops/current_goal_requirements.json"

R226_SHA256 = "sha256:5a1a3283e3097678c4feb02f468e901caf849548307f132bbddb339977958473"
R212_SHA256 = "sha256:aa9c7b8158c91d183c092b92bab3047c7bd7af705d539c68cdd3e9c206c0c2b9"


def _pipeline() -> dict[str, object]:
    value = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _overrides() -> dict[str, object]:
    value = json.loads(REQUIREMENTS_PATH.read_text(encoding="utf-8"))
    overrides = value["current_owner_overrides"]
    assert isinstance(overrides, dict)
    return overrides


def test_r226_typed_source_goal_and_projection_are_checksum_bound() -> None:
    pipeline = _pipeline()
    overrides = _overrides()
    projection = overrides["guide2vec_general_training_pipeline_r226"]

    assert "sha256:" + hashlib.sha256(PIPELINE_PATH.read_bytes()).hexdigest() == R226_SHA256
    assert pipeline["schema"] == "poke_bot.guide2vec_general_training_pipeline_r226/v1"
    assert pipeline["owner_decision_revision"] == 226
    assert pipeline["status"] == "staged_dormant_awaiting_future_guide"
    assert pipeline["purpose"] == (
        "Prepare a reusable numeric Guide2Vec training pipeline without launching "
        "training until a future checksum-bound guide revision is ready."
    )

    assert projection == {
        "goal_revision": 226,
        "status": "staged_dormant_awaiting_future_guide",
        "typed_source": str(PIPELINE_PATH.relative_to(ROOT)),
        "typed_source_sha256": R226_SHA256,
        "typed_schema": "poke_bot.guide2vec_general_training_pipeline_r226/v1",
        "strategy": "reusable_frozen_numeric_latent_store_plus_versioned_guide_label_overlay",
        "language_model_or_text_corpus_used": False,
        "card2vec_or_word2vec_retraining_required": False,
        "guide_independent_latent_store_required": True,
        "guide_labels_must_be_separate_from_latent_store": True,
        "exact_causal_stage_and_legal_option_join_required": True,
        "future_guide_update_may_reuse_latents_only_after_exact_base_abi_runtime_and_stage_identity_match": True,
        "current_guide_manifest": None,
        "current_guide_ready": False,
        "pipeline_implementation_authorized": True,
        "read_only_manifest_check_authorized": True,
        "latent_store_materialization_authorized_now": False,
        "training_service_start_authorized": False,
        "gradient_updates_authorized": False,
        "candidate_publication_authorized": False,
        "runtime_attachment_authorized": False,
        "bo1000_authorized": False,
        "mcts_change_authorized": False,
        "rtp_authorized": False,
        "selector_change_authorized": False,
        "serving_authorized": False,
        "promotion_authorized": False,
        "kaggle_api_or_submission_authorized": False,
        "evaluation_games_training_eligible": False,
        "existing_mcts_or_bo1000_service_or_output_mutation_allowed": False,
    }

    goal = " ".join((ROOT / "GOAL.md").read_text(encoding="utf-8").split())
    assert int(goal.split("Revision: `", 1)[1].split("`", 1)[0]) >= 226
    assert "| 226-GUIDE2VEC |" in goal
    assert str(PIPELINE_PATH.relative_to(ROOT)) in goal
    assert "r212 launch authority, training, gradients, candidate publication" in goal


def test_r226_keeps_labels_separate_and_binds_the_exact_causal_stage_join() -> None:
    design = _pipeline()["design"]
    assert isinstance(design, dict)
    artifacts = design["artifacts"]
    assert isinstance(artifacts, dict)
    stage_index = artifacts["stage_index"]
    latents = artifacts["frozen_latent_store"]
    labels = artifacts["guide_label_overlay"]
    training_manifest = artifacts["training_manifest"]

    assert stage_index == {
        "schema": "poke_bot.guide2vec_stage_index/v1",
        "purpose": (
            "Stable causal decision-stage identities and exact legal-option ordering "
            "independent of guide revision."
        ),
    }
    assert latents == {
        "schema": "poke_bot.guide2vec_frozen_latent_store/v1",
        "guide_labels_embedded": False,
        "required_tensors": [
            "state_vec",
            "option_hidden",
            "base_logits",
            "option_offsets",
        ],
        "reuse_requires_exact_identity": [
            "frozen_base_identity_sha256",
            "latent_abi_sha256",
            "adapter_runtime_context_sha256",
            "stage_index_sha256",
            "extractor_code_sha256",
            "dtype_policy_sha256",
        ],
    }
    assert labels == {
        "schema": "poke_bot.guide2vec_label_overlay/v1",
        "required_labels": ["guide_target_index", "guide_confidence"],
        "stage_join_key_fields": [
            "source_day",
            "episode_id",
            "seat",
            "decision_index",
            "factorized_stage_index",
            "policy_visible_observation_sha256",
            "legal_options_sha256",
        ],
        "target_must_index_the_bound_legal_option_order": True,
        "teacher_semantics_sha256_required": True,
        "guide_contract_and_every_teacher_dependency_sha256_required": True,
    }
    assert training_manifest == {
        "schema": "poke_bot.guide2vec_training_manifest/v1",
        "binds_stage_index_latents_labels_split_base_and_guide": True,
        "whole_episode_and_source_day_disjoint_partitions_required": True,
        "heldout_sealed_until_candidate_selection": True,
    }


def test_r226_guide_update_reuses_latents_only_when_every_identity_matches() -> None:
    pipeline = _pipeline()
    design = pipeline["design"]
    readiness = pipeline["future_guide_readiness"]
    assert isinstance(design, dict)
    assert isinstance(readiness, dict)

    assert design["strategy"] == (
        "frozen_numeric_state_and_legal_option_latents_plus_versioned_guide_label_overlay"
    )
    assert design["language_model_or_text_corpus_used"] is False
    assert design["card2vec_or_word2vec_retraining_required"] is False
    assert design["frozen_base_feature_extraction_is_guide_independent"] is True
    assert design["guide_update_requires_base_latent_reextraction_when_base_abi_is_unchanged"] is False
    assert design["guide_update_requires_new_label_overlay"] is True
    assert design["guide_update_requires_new_small_head_candidate"] is True

    assert readiness["current_guide_manifest"] is None
    assert readiness["current_guide_ready"] is False
    assert readiness["warm_start_from_prior_guide_head_allowed"] is False
    assert readiness["warm_start_requires_separate_target_compatibility_receipt"] is True
    assert readiness["required_before_training_authority"] == [
        "immutable_guide_contract_and_teacher_semantics_receipt",
        "causal_stage_index_receipt",
        "exact_legal_option_label_alignment_receipt",
        "whole_episode_and_source_day_split_receipt",
        "guide_independent_latent_store_identity_receipt",
        "frozen_base_and_runtime_context_receipt",
        "dedicated_host_noninterference_receipt",
        "content_addressed_source_and_output_receipt",
        "separate_explicit_owner_training_launch_authorization",
    ]


def test_r226_is_dormant_and_cannot_grant_runtime_or_search_authority() -> None:
    pipeline = _pipeline()
    authority = pipeline["authority"]
    non_interference = pipeline["non_interference"]
    head = pipeline["head"]
    assert isinstance(authority, dict)
    assert isinstance(non_interference, dict)
    assert isinstance(head, dict)

    assert authority["pipeline_implementation_authorized"] is True
    assert authority["read_only_manifest_check_authorized"] is True
    assert authority["label_overlay_materialization_authorized_after_future_guide_is_sealed"] is True
    assert all(
        authority[key] is False
        for key in (
            "latent_store_materialization_authorized_now",
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
    )
    assert all(value is False for value in non_interference.values())

    assert head == {
        "implementation": "poke_bot.guide2vec.Guide2VecHead",
        "architecture": "legal_option_conditioned_ranker_plus_explicit_eligibility_head",
        "default_d_model": 96,
        "default_parameter_count": 155468,
        "parameter_count_min": 100000,
        "parameter_count_max": 500000,
        "maximum_normalized_logit_bonus": 0.05,
        "illegal_or_padded_option_authority": False,
        "exact_base_fallback_required": True,
        "base_model_trainable": False,
        "matchup_adapter_trainable": False,
        "only_guide2vec_head_parameters_may_receive_gradients": True,
    }


def test_r226_retires_r212_launches_without_rewriting_r212_evidence() -> None:
    pipeline = _pipeline()
    supersession = pipeline["supersession"]
    overrides = _overrides()
    r212_projection = overrides["alakazam_guide2vec_no_mcts_bo1000_r212"]
    assert isinstance(supersession, dict)
    assert isinstance(r212_projection, dict)

    assert supersession == {
        "retired_experiment": str(R212_PATH.relative_to(ROOT)),
        "r212_training_launch_authority_retired": True,
        "r212_bo1000_launch_authority_retired": True,
        "r212_service_must_remain_unlinked_and_unstarted": True,
        "r212_artifacts_and_receipts_preserved_for_audit": True,
        "r212_files_may_be_deleted_or_rewritten": False,
        "mcts_contracts_or_running_mcts_work_changed": False,
    }
    assert "sha256:" + hashlib.sha256(R212_PATH.read_bytes()).hexdigest() == R212_SHA256
    assert r212_projection["latest_owner_clarification_revision"] == 226
    assert r212_projection["status"] == "abandoned_unlaunched_preserved_by_r226_general_pipeline"
    assert r212_projection["superseded_by_typed_source"] == str(
        PIPELINE_PATH.relative_to(ROOT)
    )
    assert r212_projection["training_or_bo1000_launch_authority_retired"] is True
    assert all(
        r212_projection[key] is False
        for key in (
            "guide2vec_head_gradient_updates_authorized",
            "dedicated_guide2vec_training_service_start_authorized_after_preflight",
            "immutable_guide2vec_artifact_publication_authorized",
            "exact_no_mcts_bo1000_authorized_after_prerequisites",
        )
    )


def _sha256(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw_json_write(path: Path, payload: object) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _raw_bytes_write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _inert_authority() -> dict[str, bool]:
    return {field: False for field in INERT_AUTHORITY_FIELDS}


def _digest_tensor_layout() -> dict[str, object]:
    return {
        "dtype": DIGEST_TENSOR_LAYOUT["dtype"],
        "shape": list(DIGEST_TENSOR_LAYOUT["shape"]),
        "encoding": DIGEST_TENSOR_LAYOUT["encoding"],
        "stage_key_digest": DIGEST_TENSOR_LAYOUT["stage_key_digest"],
        "legal_option_digest": DIGEST_TENSOR_LAYOUT["legal_option_digest"],
    }


def _content_ref(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _dormant_training_manifest() -> dict[str, object]:
    input_digests = {
        identity: _sha256(f"unresolved-{identity}")
        for identity in TRAINING_INPUT_IDENTITY_FIELDS
    }
    return {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "status": "dormant",
        "compatibility": {field: None for field in COMPATIBILITY_FIELDS},
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_count": None,
        "inputs": {
            ref_name: _content_ref(
                f"must-not-be-opened/{ref_name}.json",
                input_digests[identity_name],
            )
            for ref_name, identity_name in zip(
                TRAINING_INPUT_FIELDS, TRAINING_INPUT_IDENTITY_FIELDS, strict=True
            )
        },
        "input_identities": input_digests,
        "authority": _inert_authority(),
        "activation_receipt": None,
    }


def _ready_manifest_tree(
    directory: Path,
    *,
    guide_revision: str,
    base_revision: str,
) -> dict[str, object]:
    """Write one internally compatible, content-sealed manifest fixture."""

    stage_count = 3
    legal_option_key_sha256 = _sha256("legal-options-v1")
    content_paths = {
        "stage_rows": directory / "stage-rows.jsonl",
        "latent_payload": directory / "frozen-latents.pt",
        "label_payload": directory / "guide-labels.pt",
        "guide_contract": directory / "future-guide-contract.json",
        "teacher_semantics": directory / "future-teacher-semantics.json",
        "teacher_dependency": directory / "future-teacher-dependency.json",
        "partition_payload": directory / "causal-partitions.json",
    }
    content_digests = {
        "stage_rows": _raw_bytes_write(
            content_paths["stage_rows"], b'{"stage":"causal-stage-v1"}\n'
        ),
        "latent_payload": _raw_bytes_write(
            content_paths["latent_payload"], f"frozen-latents:{base_revision}\n".encode()
        ),
        "label_payload": _raw_bytes_write(
            content_paths["label_payload"], f"guide-labels:{guide_revision}\n".encode()
        ),
        "guide_contract": _raw_bytes_write(
            content_paths["guide_contract"], f"guide-contract:{guide_revision}\n".encode()
        ),
        "teacher_semantics": _raw_bytes_write(
            content_paths["teacher_semantics"], f"teacher:{guide_revision}\n".encode()
        ),
        "teacher_dependency": _raw_bytes_write(
            content_paths["teacher_dependency"], f"teacher-dependency:{guide_revision}\n".encode()
        ),
        "partition_payload": _raw_bytes_write(
            content_paths["partition_payload"], b"partition-v1\n"
        ),
    }
    stage_payload = {
        "schema": STAGE_INDEX_SCHEMA,
        "source_identity_sha256": _sha256("causal-source-v1"),
        "legal_option_key_sha256": legal_option_key_sha256,
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_count": stage_count,
        "stage_rows": _content_ref(content_paths["stage_rows"].name, content_digests["stage_rows"]),
        "authority": _inert_authority(),
    }
    stage_path = directory / "stage-index.json"
    stage_sha256 = _raw_json_write(stage_path, stage_payload)

    compatibility = {
        field: _sha256(f"{base_revision}:{field}") for field in COMPATIBILITY_FIELDS
    }
    compatibility["stage_index_sha256"] = stage_sha256
    compatibility["legal_option_key_sha256"] = legal_option_key_sha256
    stage_ref = _content_ref(stage_path.name, stage_sha256)

    frozen_payload = {
        "schema": FROZEN_LATENT_STORE_SCHEMA,
        "compatibility": compatibility,
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_index_manifest": stage_ref,
        "stage_count": stage_count,
        "latent_payload": _content_ref(
            content_paths["latent_payload"].name, content_digests["latent_payload"]
        ),
        "tensors": list(FROZEN_LATENT_TENSORS),
        "guide_labels_embedded": False,
        "authority": _inert_authority(),
    }
    frozen_path = directory / "frozen-latents.json"
    frozen_sha256 = _raw_json_write(frozen_path, frozen_payload)

    label_payload = {
        "schema": GUIDE_LABEL_OVERLAY_SCHEMA,
        "compatibility": compatibility,
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_index_manifest": stage_ref,
        "stage_count": stage_count,
        "label_payload": _content_ref(
            content_paths["label_payload"].name, content_digests["label_payload"]
        ),
        "label_fields": list(GUIDE_LABEL_FIELDS),
        "guide_contract": _content_ref(
            content_paths["guide_contract"].name, content_digests["guide_contract"]
        ),
        "teacher_semantics": _content_ref(
            content_paths["teacher_semantics"].name, content_digests["teacher_semantics"]
        ),
        "teacher_dependencies": [
            _content_ref(
                content_paths["teacher_dependency"].name,
                content_digests["teacher_dependency"],
            )
        ],
        "authority": _inert_authority(),
    }
    labels_path = directory / "guide-labels.json"
    labels_sha256 = _raw_json_write(labels_path, label_payload)

    split_payload = {
        "schema": CAUSAL_SPLIT_SCHEMA,
        "compatibility": compatibility,
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_index_manifest": stage_ref,
        "stage_count": stage_count,
        "partition_payload": _content_ref(
            content_paths["partition_payload"].name, content_digests["partition_payload"]
        ),
        "partition_guards": {
            "whole_episode_disjoint": True,
            "whole_source_day_disjoint": True,
            "heldout_sealed_until_candidate_selection": True,
        },
        "authority": _inert_authority(),
    }
    split_path = directory / "causal-split.json"
    split_sha256 = _raw_json_write(split_path, split_payload)

    input_refs = {
        "stage_index_manifest": stage_ref,
        "frozen_latent_store_manifest": _content_ref(frozen_path.name, frozen_sha256),
        "guide_label_overlay_manifest": _content_ref(labels_path.name, labels_sha256),
        "causal_split_manifest": _content_ref(split_path.name, split_sha256),
    }
    input_identities = {
        "stage_index_sha256": stage_sha256,
        "frozen_latent_store_sha256": frozen_sha256,
        "guide_label_overlay_sha256": labels_sha256,
        "causal_split_sha256": split_sha256,
    }
    training_payload = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "status": "ready",
        "compatibility": compatibility,
        "stage_join_key_fields": list(STAGE_JOIN_KEY_FIELDS),
        "digest_tensor_layout": _digest_tensor_layout(),
        "stage_count": stage_count,
        "inputs": input_refs,
        "input_identities": input_identities,
        "authority": _inert_authority(),
        "activation_receipt": None,
    }
    training_path = directory / "training-manifest.json"
    training_sha256 = _raw_json_write(training_path, training_payload)
    return {
        "training_path": training_path,
        "training_payload": training_payload,
        "training_sha256": training_sha256,
        "stage_sha256": stage_sha256,
        "frozen_path": frozen_path,
        "frozen_payload": frozen_payload,
        "frozen_sha256": frozen_sha256,
        "labels_sha256": labels_sha256,
        "split_sha256": split_sha256,
        "compatibility": compatibility,
        "content_paths": content_paths,
        "content_digests": content_digests,
    }


def test_dormant_manifest_never_opens_future_refs_or_authorizes_training(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dormant-training-manifest.json"
    _raw_json_write(manifest_path, _dormant_training_manifest())

    resolved = load_and_validate_training_manifest(manifest_path)

    assert resolved.readiness.status == "dormant"
    assert resolved.readiness.content_refs_sealed is False
    assert resolved.readiness.split_semantics_proven is False
    assert resolved.readiness.data_ready is False
    assert resolved.readiness.ready is False
    assert resolved.readiness.split_proof_required_schema == CAUSAL_PARTITION_PAYLOAD_SCHEMA
    assert resolved.readiness.blocking_reasons[0].endswith(CAUSAL_PARTITION_PAYLOAD_SCHEMA)
    assert resolved.readiness.training_authorized is False
    assert resolved.readiness.separate_activation_receipt_required is True
    assert resolved.readiness.resolved_inputs == ()
    assert resolved.readiness.unresolved_inputs == TRAINING_INPUT_FIELDS
    assert not (tmp_path / "must-not-be-opened").exists()

    with pytest.raises(Guide2VecContractError, match="dormant"):
        load_and_validate_training_manifest(manifest_path, require_ready=True)


def test_ready_manifest_binds_every_base_abi_and_stage_identity(tmp_path: Path) -> None:
    fixture = _ready_manifest_tree(
        tmp_path / "valid", guide_revision="guide-v1", base_revision="base-v1"
    )
    resolved = load_and_validate_training_manifest(fixture["training_path"])

    assert resolved.manifest_sha256 == fixture["training_sha256"]
    assert resolved.readiness.status == "ready"
    assert resolved.readiness.content_refs_sealed is True
    assert resolved.readiness.split_semantics_proven is False
    assert resolved.readiness.data_ready is False
    assert resolved.readiness.ready is False
    assert resolved.readiness.split_proof_required_schema == CAUSAL_PARTITION_PAYLOAD_SCHEMA
    assert resolved.readiness.blocking_reasons[0].endswith(CAUSAL_PARTITION_PAYLOAD_SCHEMA)
    assert "separate owner activation receipt" in resolved.readiness.blocking_reasons[1]
    assert resolved.readiness.training_authorized is False
    assert resolved.readiness.resolved_inputs == TRAINING_INPUT_FIELDS
    assert resolved.frozen_latent_store_manifest is not None
    assert resolved.guide_label_overlay_manifest is not None
    assert [content.role for content in resolved.resolved_content] == [
        "stage_index.stage_rows",
        "frozen_latent_store.latent_payload",
        "guide_label_overlay.label_payload",
        "guide_label_overlay.guide_contract",
        "guide_label_overlay.teacher_semantics",
        "guide_label_overlay.teacher_dependencies[0]",
        "causal_split.partition_payload",
    ]

    with pytest.raises(Guide2VecContractError, match="causal_partition_payload/v1"):
        load_and_validate_training_manifest(fixture["training_path"], require_ready=True)

    # A replacement frozen store with a different ABI is not reusable under the
    # original training manifest, even though its stage index still matches.
    changed_frozen = dict(fixture["frozen_payload"])
    changed_compatibility = dict(fixture["compatibility"])
    changed_compatibility["latent_abi_sha256"] = _sha256("base-v2:latent_abi_sha256")
    changed_frozen["compatibility"] = changed_compatibility
    changed_frozen_sha256 = _raw_json_write(fixture["frozen_path"], changed_frozen)
    changed_training = dict(fixture["training_payload"])
    changed_inputs = dict(changed_training["inputs"])
    changed_inputs["frozen_latent_store_manifest"] = _content_ref(
        Path(fixture["frozen_path"]).name, changed_frozen_sha256
    )
    changed_training["inputs"] = changed_inputs
    changed_identities = dict(changed_training["input_identities"])
    changed_identities["frozen_latent_store_sha256"] = changed_frozen_sha256
    changed_training["input_identities"] = changed_identities
    _raw_json_write(fixture["training_path"], changed_training)

    with pytest.raises(Guide2VecContractError, match="frozen latent store compatibility.latent_abi_sha256"):
        load_and_validate_training_manifest(fixture["training_path"], require_ready=True)


def test_guide_revision_changes_only_overlay_and_training_identity_when_base_is_stable(
    tmp_path: Path,
) -> None:
    first = _ready_manifest_tree(
        tmp_path / "guide-v1", guide_revision="guide-v1", base_revision="base-v1"
    )
    second = _ready_manifest_tree(
        tmp_path / "guide-v2", guide_revision="guide-v2", base_revision="base-v1"
    )

    assert first["stage_sha256"] == second["stage_sha256"]
    assert first["frozen_sha256"] == second["frozen_sha256"]
    assert first["labels_sha256"] != second["labels_sha256"]
    assert first["training_sha256"] != second["training_sha256"]
    first_readiness = load_and_validate_training_manifest(first["training_path"]).readiness
    second_readiness = load_and_validate_training_manifest(second["training_path"]).readiness
    assert first_readiness.content_sealed is True
    assert second_readiness.content_sealed is True
    assert first_readiness.data_ready is False
    assert second_readiness.data_ready is False


def test_ready_manifest_fails_closed_for_missing_or_tampered_nested_content(tmp_path: Path) -> None:
    missing = _ready_manifest_tree(
        tmp_path / "missing", guide_revision="guide-v1", base_revision="base-v1"
    )
    missing_paths = missing["content_paths"]
    assert isinstance(missing_paths, dict)
    Path(missing_paths["teacher_semantics"]).unlink()
    with pytest.raises(Guide2VecContractError, match="content reference is not a file"):
        load_and_validate_training_manifest(missing["training_path"])

    tampered = _ready_manifest_tree(
        tmp_path / "tampered", guide_revision="guide-v1", base_revision="base-v1"
    )
    tampered_paths = tampered["content_paths"]
    assert isinstance(tampered_paths, dict)
    Path(tampered_paths["label_payload"]).write_bytes(b"tampered-label-bytes\n")
    with pytest.raises(
        Guide2VecContractError,
        match="guide_label_overlay.label_payload content digest mismatch",
    ):
        load_and_validate_training_manifest(tampered["training_path"])


def test_manifest_contract_rejects_reordered_stage_key_or_any_authority_grant() -> None:
    invalid = _dormant_training_manifest()
    invalid["stage_join_key_fields"] = list(reversed(STAGE_JOIN_KEY_FIELDS))
    with pytest.raises(Guide2VecContractError, match="stage_join_key_fields"):
        validate_training_manifest(invalid)

    invalid = _dormant_training_manifest()
    authority = dict(invalid["authority"])
    authority["mcts_change_authorized"] = True
    invalid["authority"] = authority
    with pytest.raises(Guide2VecContractError, match="mcts_change_authorized"):
        validate_training_manifest(invalid)


def test_stage_key_digest_is_exact_and_can_be_used_as_raw_chunk_bytes() -> None:
    stage_key = {
        "source_day": "2026-08-10",
        "episode_id": "episode-123",
        "seat": 1,
        "decision_index": 4,
        "factorized_stage_index": 2,
        "policy_visible_observation_sha256": _sha256("observation-v1"),
        "legal_options_sha256": _sha256("legal-options-v1"),
    }
    digest = stage_key_sha256(stage_key)
    assert len(digest_tensor_bytes(digest)) == 32

    changed_legal_options = dict(stage_key)
    changed_legal_options["legal_options_sha256"] = _sha256("legal-options-v2")
    assert stage_key_sha256(changed_legal_options) != digest

    with pytest.raises(Guide2VecContractError, match="fields changed"):
        stage_key_sha256({**stage_key, "noncausal_extra": "forbidden"})


def test_training_entrypoint_refuses_an_unverified_grant_before_any_chunk_io(tmp_path: Path) -> None:
    absent_paths: list[Path] = []

    def absent_pair(partition: str) -> JoinedChunkRef:
        latent_path = tmp_path / f"must-not-open-{partition}-latents.pt"
        label_path = tmp_path / f"must-not-open-{partition}-labels.pt"
        absent_paths.extend((latent_path, label_path))
        return JoinedChunkRef(
            latent=ContentChunkRef(
                path=latent_path, sha256=_sha256(f"not-a-real-latent:{partition}")
            ),
            labels=ContentChunkRef(
                path=label_path, sha256=_sha256(f"not-a-real-label:{partition}")
            ),
        )

    train_pair = absent_pair("train")
    validation_pair = absent_pair("validation")
    heldout_pair = absent_pair("heldout")
    output_dir = tmp_path / "must-not-create-output"
    dormant_manifest_path = tmp_path / "dormant-training-manifest.json"
    activation_receipt_path = tmp_path / "must-not-open-activation-receipt.json"
    _raw_json_write(dormant_manifest_path, _dormant_training_manifest())
    base_identity = FrozenBaseIdentity(
        submission_id=1,
        checkpoint_sha256=_sha256("checkpoint"),
        checkpoint_bytes=1,
        bundle_sha256=_sha256("bundle"),
        model_config_sha256=_sha256("model-config"),
        feature_schema_sha256=_sha256("feature-schema"),
    )

    with pytest.raises(Guide2VecTrainingError, match="semantically-ready activation grant"):
        train_from_joined_chunks(
            train_chunks=[train_pair],
            validation_chunks=[validation_pair],
            heldout_chunks=[heldout_pair],
            base_identity=base_identity,
            guide_manifest_sha256=_sha256("guide-manifest"),
            training_manifest_path=dormant_manifest_path,
            activation_receipt_path=activation_receipt_path,
            config=None,
            output_dir=output_dir,
        )

    assert all(not path.exists() for path in absent_paths)
    assert not activation_receipt_path.exists()
    assert not output_dir.exists()


def test_r226_activation_cannot_issue_a_grant_before_opening_any_receipt(
    tmp_path: Path,
) -> None:
    missing_receipt = tmp_path / "must-not-be-opened-activation-receipt.json"

    dormant_path = tmp_path / "dormant-manifest.json"
    _raw_json_write(dormant_path, _dormant_training_manifest())
    with pytest.raises(Guide2VecActivationError, match="canonically revalidated semantic-ready"):
        validate_activation_receipt(
            training_manifest_path=dormant_path,
            receipt_path=missing_receipt,
        )

    sealed_fixture = _ready_manifest_tree(
        tmp_path / "sealed", guide_revision="guide-v1", base_revision="base-v1"
    )
    sealed = load_and_validate_training_manifest(sealed_fixture["training_path"])
    assert sealed.readiness.content_sealed is True
    assert sealed.readiness.data_ready is False
    with pytest.raises(Guide2VecActivationError, match="canonically revalidated semantic-ready"):
        validate_activation_receipt(
            training_manifest_path=sealed_fixture["training_path"],
            receipt_path=missing_receipt,
        )

    assert not missing_receipt.exists()


def test_training_grant_is_opaque_and_chunk_set_identities_fail_closed() -> None:
    content = VerifiedContentRef(path=Path("unopened.json"), sha256=_sha256("unopened"))
    compatibility = {field: _sha256(f"compatibility:{field}") for field in COMPATIBILITY_FIELDS}
    partition_chunk_sets = {
        partition: SealedChunkSet(
            partition=partition,
            records=(),
            sha256=_sha256(f"unopened-set:{partition}"),
        )
        for partition in ("train", "validation", "heldout")
    }
    with pytest.raises(TypeError, match="use validate_activation_receipt"):
        Guide2VecTrainingGrant(
            _token=object(),
            receipt_path=Path("unopened-receipt.json"),
            receipt_sha256=_sha256("receipt"),
            owner_contract_revision=227,
            owner_contract=content,
            training_manifest_sha256=_sha256("manifest"),
            guide_label_overlay_sha256=_sha256("overlay"),
            guide_contract=content,
            compatibility=compatibility,
            causal_split_manifest_sha256=_sha256("split"),
            causal_split_receipt=content,
            source_snapshot=content,
            host_noninterference_capability_receipt=content,
            output_root=Path("unopened-output"),
            output_root_identity_sha256=_sha256("output"),
            partition_chunk_sets=partition_chunk_sets,
        )

    record = {
        "latent_sha256": _sha256("latent"),
        "label_sha256": _sha256("labels"),
        "pair_sha256": joined_chunk_pair_sha256(
            latent_sha256=_sha256("latent"), label_sha256=_sha256("labels")
        ),
    }
    train_identity = sealed_chunk_set_sha256(partition="train", records=[record])
    assert train_identity == sealed_chunk_set_sha256(partition="train", records=[record])

    changed_record = dict(record)
    changed_record["label_sha256"] = _sha256("different-labels")
    changed_record["pair_sha256"] = joined_chunk_pair_sha256(
        latent_sha256=changed_record["latent_sha256"],
        label_sha256=changed_record["label_sha256"],
    )
    assert sealed_chunk_set_sha256(partition="train", records=[changed_record]) != train_identity

    forged_pair = dict(record)
    forged_pair["pair_sha256"] = _sha256("forged-pair")
    with pytest.raises(Guide2VecActivationError, match="pair_sha256 mismatch"):
        sealed_chunk_set_sha256(partition="train", records=[forged_pair])


def test_explicit_always_abstain_beats_a_finite_float32_sigmoid_one() -> None:
    """A threshold of one alone cannot suppress a finite saturated sigmoid."""

    config = Guide2VecConfig(min_eligibility=1.0)
    head = Guide2VecHead(config)
    forward_calls = 0

    def saturated_forward(
        state_vec: torch.Tensor,
        option_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        n_options: int | list[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del state_vec, option_hidden, n_options
        nonlocal forward_calls
        forward_calls += 1
        guide_scores = torch.tensor([[0.0, 1.0]], dtype=base_logits.dtype)
        # This is finite in float32 but sigmoid-rounds to exactly 1.0.
        eligibility_logits = torch.full((1,), 20.0, dtype=base_logits.dtype)
        return guide_scores, eligibility_logits

    head.forward = saturated_forward  # type: ignore[method-assign]
    identity = FrozenBaseIdentity(
        submission_id=1,
        checkpoint_sha256=_sha256("checkpoint"),
        checkpoint_bytes=1,
        bundle_sha256=_sha256("bundle"),
        model_config_sha256=_sha256("model-config"),
        feature_schema_sha256=_sha256("feature-schema"),
    )
    state = torch.zeros((1, 96), dtype=torch.float32)
    hidden = torch.zeros((1, 2, 96), dtype=torch.float32)
    base_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    assert torch.sigmoid(torch.tensor(20.0, dtype=torch.float32)).item() == 1.0
    direct = head.rerank(
        state,
        hidden,
        base_logits,
        n_options=2,
        expected_base_identity=identity,
        observed_base_identity=identity,
    )
    assert direct.applied.tolist() == [True]
    assert forward_calls == 1

    runtime = FrozenGuide2VecRuntime(
        guide2vec_config=config,
        calibration_status="abstain_all_no_validation_labels",
        calibrated_threshold=1.0,
        always_abstain=True,
    )
    abstained = rerank_with_frozen_runtime(
        head,
        runtime,
        state,
        hidden,
        base_logits,
        n_options=2,
        expected_base_identity=identity,
        observed_base_identity=identity,
    )

    assert forward_calls == 1  # The frozen gate must precede ``head.forward``.
    assert abstained.applied.tolist() == [False]
    assert abstained.fallback.tolist() == [True]
    assert torch.equal(abstained.adjusted_logits, base_logits)
    assert abstained.reasons == ("configured_always_abstain",)


def _torch_chunk_ref(path: Path, payload: dict[str, object]) -> ContentChunkRef:
    torch.save(payload, path)
    return ContentChunkRef(
        path=path,
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _joined_chunk_payloads() -> tuple[dict[str, object], dict[str, object]]:
    stage_key_digest = torch.arange(96, dtype=torch.uint8).reshape(3, 32)
    legal_options_digest = stage_key_digest + 96
    latent = {
        "schema": LATENT_CHUNK_SCHEMA,
        "stage_key_digest": stage_key_digest,
        "legal_options_digest": legal_options_digest,
        "state_vec": torch.zeros((3, 96), dtype=torch.float32),
        "option_hidden": torch.zeros((6, 96), dtype=torch.float32),
        "base_logits": torch.zeros(6, dtype=torch.float32),
        "option_offsets": torch.tensor([0, 2, 3, 6], dtype=torch.int64),
    }
    labels = {
        "schema": LABEL_CHUNK_SCHEMA,
        "stage_key_digest": stage_key_digest.clone(),
        "legal_options_digest": legal_options_digest.clone(),
        "guide_target_index": torch.tensor([1, -1, 2], dtype=torch.int64),
        "guide_confidence": torch.tensor([0.8, 0.0, 1.0], dtype=torch.float32),
    }
    return latent, labels


def test_chunk_join_requires_exact_stage_and_legal_option_fingerprints(tmp_path: Path) -> None:
    latent_payload, label_payload = _joined_chunk_payloads()
    latent = _torch_chunk_ref(tmp_path / "latents.pt", latent_payload)
    labels = _torch_chunk_ref(tmp_path / "labels.pt", label_payload)

    joined = load_joined_chunk(JoinedChunkRef(latent=latent, labels=labels))
    assert joined.rows == 3
    assert joined.latent.counts.tolist() == [2, 1, 3]

    mismatched_labels = dict(label_payload)
    mismatched_legal_options = label_payload["legal_options_digest"].clone()
    mismatched_legal_options[1, 0] += 1
    mismatched_labels["legal_options_digest"] = mismatched_legal_options
    bad_labels = _torch_chunk_ref(tmp_path / "labels-wrong-order.pt", mismatched_labels)
    with pytest.raises(Guide2VecTrainingError, match="legal_options_digest does not exactly join"):
        load_joined_chunk(JoinedChunkRef(latent=latent, labels=bad_labels))

    bad_target = dict(label_payload)
    escaped_target = label_payload["guide_target_index"].clone()
    escaped_target[0] = 2  # Row zero has only two legal options, indexed 0 and 1.
    bad_target["guide_target_index"] = escaped_target
    escaped_labels = _torch_chunk_ref(tmp_path / "labels-escaped-target.pt", bad_target)
    with pytest.raises(Guide2VecTrainingError, match="escaped its bound legal option order"):
        load_joined_chunk(JoinedChunkRef(latent=latent, labels=escaped_labels))


def test_chunk_digest_is_checked_before_any_torch_deserialization(tmp_path: Path) -> None:
    invalid_chunk = tmp_path / "not-a-torch-payload.pt"
    invalid_chunk.write_bytes(b"this is deliberately not a Torch artifact")
    wrong_ref = ContentChunkRef(path=invalid_chunk, sha256=_sha256("different-bytes"))

    with pytest.raises(Guide2VecTrainingError, match="chunk digest mismatch"):
        load_latent_chunk(wrong_ref)
