"""Fail-closed validation receipts for the isolated Alakazam r298 study.

This module deliberately owns *evidence binding*, not a policy path.  It is
used by the disposable Elmo-only runner to combine independently produced
collision, representation, target, trace, and engine-test artifacts into one
receipt that later diagnostics can consume.  It cannot enable a model, alter a
checkpoint, select an action, or touch a managed service.

The distinction between a test fixture and a pinned-simulator execution is
important here.  Fixture tests are useful regressions, but a receipt marked
``passed_elmo_only_nonproduction`` requires an explicit actual pinned-engine
evidence record.  This prevents paper-rule or hand-written fixture claims from
being relabelled as competition-simulator proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


R298_VALIDATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_validation_receipt/v1"
)
R298_PHASE_EVIDENCE_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_phase_evidence_manifest/v1"
)
R298_ENGINE_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_engine_evidence/v1"
)
R298_RAW_EXPERT_CORPUS_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_raw_expert_corpus_manifest/v1"
)
R298_RESOURCE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_resource_receipt/v1"
)
R298_FROZEN_TENSOR_IDENTITY_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_frozen_tensor_identity_evidence/v1"
)
R298_CHECKLIST_PROVENANCE_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_checklist_provenance_evidence/v1"
)
R298_CANDIDATE_ACTION_TIME_CHECKLIST_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_candidate_action_time_checklist_receipt/v1"
)
R298_SCHEMA_FREEZE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_schema_freeze_receipt/v1"
)
R298_COLLISION_RAW_CORPUS_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_manifest/v1"
)
R298_COLLISION_RAW_CORPUS_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_receipt/v1"
)
R298_COLLISION_CENSUS_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_collision_census_r298_receipt/v1"
)
R298_COLLISION_FROZEN_SCHEMA_MANIFEST_SCHEMA = (
    "poke_bot.alakazam_collision_census_r298_frozen_schema_manifest/v1"
)
R298_COLLISION_ZERO_BYPASS_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_collision_census_r298_zero_bypass_receipt/v1"
)
R298_REFEATURIZATION_SPLIT_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_refeaturization_split_receipt/v1"
)
R298_MATERIALIZATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_materialization_receipt/v1"
)
R298_CANDIDATE_TRAINING_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_candidate_training_receipt/v1"
)
R298_OWNER_REVISION = 298
R298_RAW_EXPERT_CORPUS_DAYS = 30
R298_RAW_EXPERT_CORPUS_START_UTC = "2026-07-13"
R298_RAW_EXPERT_CORPUS_END_UTC = "2026-08-11"
R298_EXPERIMENT_RAM_HARD_CEILING_BYTES = 103_079_215_104
R298_MATERIALIZATION_SHARD_MAX_BYTES = 96 * 1024 * 1024
R298_TRUSTED_ELMO_OUTPUT_ROOT = (
    "/srv/poke-bot-agent/outputs/experiments/"
    "alakazam-elmo-rule-derivative-g1"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
R298_CANONICAL_ELMO_EXECUTION_HOSTNAME = "truenas"
R298_ELMO_EXECUTION_HOST_VERIFICATION = (
    "exact_socket_hostname_and_systemd_detect_virt_non_container"
)

OWNER_GOAL_ATTACHMENT_SHA256 = (
    "sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f"
)
MECHANICS_ATTACHMENT_SHA256 = (
    "sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb"
)
CANONICAL_LIBCG_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
DEDICATED_GOAL_GATEWAY_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
DEDICATED_GOAL_CONTRACT_SHA256 = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
CANONICAL_LIBCG_CONTRACT_SHA256 = (
    "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"
)
R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
EXACT_NEW_LIST_MULTISET_SHA256 = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)

PHASE_NAMES: tuple[str, ...] = (
    "collision_census",
    "public_rule_adapter",
    "simulator_targets",
    "eight_channel_trace_contract",
    "engine_metamorphic_tests",
    "layer_off_baseline_parity",
    "information_set_hidden_invariance",
)

# Every artifact that can make the aggregate receipt eligible has its own
# measured Elmo envelope.  These names deliberately include work that happens
# between the seven logical validation lanes (notably re-featurization and the
# rev4 quarantine transfer) so a pleasant-looking phase report cannot hide an
# unmeasured memory-heavy job.
RESOURCE_JOB_NAMES: tuple[str, ...] = (
    "schema_freeze",
    "raw_expert_corpus",
    "refeaturization",
    "collision_census",
    "engine_metamorphic_tests",
    "transfer_staging",
    "candidate_training",
)

PHASE_REQUIRED_ASSERTIONS: Mapping[str, frozenset[str]] = {
    "collision_census": frozenset(
        {
            "raw_schema_field_classification",
            "public_collision_classification",
            "frequency_and_action_change_risk",
        }
    ),
    "public_rule_adapter": frozenset(
        {
            "number_semantics",
            "skill_source_binding",
            "permutation_equivariance",
            "hidden_sanitization",
        }
    ),
    "simulator_targets": frozenset(
        {
            "legality_from_option_list_only",
            "prompt_chain_credit",
            "prize_yield",
            "deckout_timing",
        }
    ),
    "eight_channel_trace_contract": frozenset(
        {
            "all_scored_stages_and_forced_turns",
            "q3_bench_only",
            "q5_q6_trace_only_zero",
            "q1_q2_q8_pinned_provenance_or_zero_unavailable",
        }
    ),
    "engine_metamorphic_tests": frozenset(
        {
            "pinned_engine_execution",
            "fixture_regressions_not_relabelled_as_engine_proof",
        }
    ),
    "layer_off_baseline_parity": frozenset(
        {"exact_logits", "legal_choice"}
    ),
    "information_set_hidden_invariance": frozenset(
        {
            "sanitized_features_bit_identical",
            "policy_logits_bit_identical",
            "public_belief_decisions_bit_identical",
        }
    ),
}

ENGINE_CASE_NAMES: tuple[str, ...] = (
    "number_4_vs_5_vs_larger",
    "distinct_same_card_skill_sources",
    "duplicate_attack_payloads_with_different_simulator_meanings",
    "yes_no_prompts_with_different_context_card_or_effect",
    "energy_count_and_remaining_energy_damage_budgets",
    "max_hp_evolution_stack_appear_this_turn_and_once_per_turn_flags",
    "special_energy_typed_units",
    "ordinary_pokemon_ordinary_ex_mega_ex_prize_yields",
    "explicit_prize_reduction_effects",
    "deck_zero_attack_selectability",
    "full_bench_attack_selectability",
    "opponent_hand_count_zero_attack_selectability",
    "nullifying_zero_automatic_left_to_right_resolution",
    "simultaneous_knockout_prize_prompt_order_and_double_closeout_draw",
    "attack_to_knockout_to_prize_to_active_promotion_credit_assignment",
    "hidden_state_invariance",
    "legal_masking_option_permutation_padding_and_finite_residual_cap",
    "layer_off_exact_baseline_logits",
)

ATTESTATION_NAMES: tuple[str, ...] = PHASE_NAMES
AUTHORITY_NAMES: tuple[str, ...] = (
    "training",
    "replay",
    "production",
    "promotion",
    "selector",
    "kaggle",
    "submission",
    "elmo_readmission",
)
DIRECT_POLICY_BOUNDARY_NAMES: tuple[str, ...] = (
    "rtp",
    "search",
    "mcts",
    "rollout",
    "hidden_information_inference",
)
CANONICAL_SOURCE_NAMES: tuple[str, ...] = (
    "goal_gateway",
    "goal_contract",
    "canonical_libcg_contract",
)
CANONICAL_SOURCE_SHA256S: Mapping[str, str] = {
    "goal_gateway": DEDICATED_GOAL_GATEWAY_SHA256,
    "goal_contract": DEDICATED_GOAL_CONTRACT_SHA256,
    "canonical_libcg_contract": CANONICAL_LIBCG_CONTRACT_SHA256,
}
CANONICAL_SOURCE_RELATIVE_PATHS: Mapping[str, Path] = {
    "goal_gateway": Path("goals/alakazam-elmo-rule-derivative/GOAL.md"),
    "goal_contract": Path("goals/alakazam-elmo-rule-derivative/contract.json"),
    "canonical_libcg_contract": Path("state/canonical-libcg-r236.json"),
}
EVALUATION_DISJOINTNESS_NAMES: tuple[str, ...] = (
    "candidate_training_source_disjoint",
    "benchmark_seed_identities_excluded",
)


class R298ValidationError(ValueError):
    """A proposed r298 validation artifact is incomplete or unsafe."""


@dataclass(frozen=True)
class FileIdentity:
    """Stable identity for an immutable source or evidence artifact."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def canonical_json_bytes(payload: object) -> bytes:
    """The sole content-addressing encoding used by r298 receipts."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise R298ValidationError(f"{label} must be a sha256 digest")
    encoded = value.removeprefix("sha256:")
    if len(encoded) != 64 or any(character not in "0123456789abcdef" for character in encoded):
        raise R298ValidationError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R298ValidationError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _require_bool(value: object, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise R298ValidationError(f"{label} must be a bool")
    result = bool(value)
    if expected is not None and result is not expected:
        raise R298ValidationError(f"{label} must be {expected}")
    return result


def _require_nonempty_string(value: object, *, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise R298ValidationError(f"{label} must be a nonempty string")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R298ValidationError(f"{label} must be an object")
    return value


def _string_set(value: object, *, label: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise R298ValidationError(f"{label} must be a list of nonempty strings")
    return frozenset(value)


def _unique_strings(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise R298ValidationError(f"{label} must be a list of nonempty strings")
    normalized = [_require_nonempty_string(item, label=f"{label}[]") for item in value]
    if not normalized and not allow_empty:
        raise R298ValidationError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise R298ValidationError(f"{label} must not contain duplicates")
    return normalized


def _require_percent(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R298ValidationError(f"{label} must be a numeric percentage")
    result = float(value)
    if not 0.0 <= result <= 100.0:
        raise R298ValidationError(f"{label} must be between 0 and 100")
    return result


def _required_raw_utc_dates() -> tuple[str, ...]:
    start = date.fromisoformat(R298_RAW_EXPERT_CORPUS_START_UTC)
    return tuple(
        (start + timedelta(days=offset)).isoformat()
        for offset in range(R298_RAW_EXPERT_CORPUS_DAYS)
    )


def file_identity(path: Path | str, *, label: str) -> FileIdentity:
    """Read a regular, non-symlink artifact without following the input path."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R298ValidationError(f"{label} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    try:
        details = resolved.stat()
    except OSError as exc:
        raise R298ValidationError(f"{label} is unreadable: {resolved}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise R298ValidationError(f"{label} is not a regular file: {resolved}")
    return FileIdentity(
        path=str(resolved),
        sha256=sha256_file(resolved),
        size_bytes=int(details.st_size),
    )


def read_json_object(path: Path | str, *, label: str) -> tuple[FileIdentity, dict[str, Any]]:
    identity = file_identity(path, label=label)
    try:
        payload = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R298ValidationError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise R298ValidationError(f"{label} must contain a JSON object")
    return identity, payload


def _identity_from_mapping(value: object, *, label: str) -> FileIdentity:
    row = _mapping(value, label=label)
    expected = {"path", "sha256", "size_bytes"}
    if set(row) != expected:
        raise R298ValidationError(f"{label} keys must be exactly {sorted(expected)}")
    path = row["path"]
    if not isinstance(path, str) or not path:
        raise R298ValidationError(f"{label}.path must be a nonempty string")
    return FileIdentity(
        path=path,
        sha256=_require_sha256(row["sha256"], label=f"{label}.sha256"),
        size_bytes=_require_exact_int(row["size_bytes"], label=f"{label}.size_bytes", minimum=1),
    )


def _validate_identity_on_disk(identity: FileIdentity, *, label: str) -> FileIdentity:
    actual = file_identity(identity.path, label=label)
    if actual != identity:
        raise R298ValidationError(f"{label} identity drifted after manifest creation")
    return actual


def _validate_attachments(value: object) -> dict[str, Any]:
    row = _mapping(value, label="source attachments")
    expected = {"source_attachment", "mechanics_attachment"}
    if set(row) != expected:
        raise R298ValidationError(
            "source attachments must contain current owner goal and mechanics attachment"
        )
    goal = _mapping(row["source_attachment"], label="source_attachment")
    mechanics = _mapping(row["mechanics_attachment"], label="mechanics_attachment")
    if _require_sha256(goal.get("sha256"), label="source_attachment.sha256") != OWNER_GOAL_ATTACHMENT_SHA256:
        raise R298ValidationError("source attachment is not the current r298 owner goal")
    if _require_sha256(mechanics.get("sha256"), label="mechanics_attachment.sha256") != MECHANICS_ATTACHMENT_SHA256:
        raise R298ValidationError("mechanics attachment is not the r298 simulator-rules source")
    return {
        "source_attachment": {"sha256": OWNER_GOAL_ATTACHMENT_SHA256},
        "mechanics_attachment": {"sha256": MECHANICS_ATTACHMENT_SHA256},
    }


def _validate_collision_provenance(payload: Mapping[str, Any], *, label: str) -> None:
    """Check the dedicated-gateway bindings shared by collision artifacts."""

    if payload.get("owner_goal_sha256") != OWNER_GOAL_ATTACHMENT_SHA256:
        raise R298ValidationError(f"{label} owner attachment binding drifted")
    if payload.get("rule_derivative_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError(f"{label} dedicated contract binding drifted")
    if payload.get("rule_derivative_gateway_sha256") != DEDICATED_GOAL_GATEWAY_SHA256:
        raise R298ValidationError(f"{label} dedicated gateway binding drifted")
    if payload.get("mechanics_attachment_sha256") != MECHANICS_ATTACHMENT_SHA256:
        raise R298ValidationError(f"{label} mechanics attachment binding drifted")


def _validate_elmo_execution_identity(value: object, *, label: str) -> dict[str, Any]:
    """Require the producer's non-container TrueNAS/Elmo identity verbatim.

    A caller-controlled ``host: elmo`` label is not an execution boundary.  The
    collision runner emits this object after checking the socket hostname and
    systemd virtualization state; aggregate validation preserves the exact
    producer contract and rejects a substituted remote/container invocation.
    """

    row = _mapping(value, label=label)
    expected = {
        "execution_host_role",
        "canonical_execution_hostname",
        "execution_hostname",
        "execution_fqdn",
        "host_verification",
        "container_execution_permitted",
    }
    if set(row) != expected:
        raise R298ValidationError(f"{label} field inventory changed")
    if (
        row.get("execution_host_role"),
        row.get("canonical_execution_hostname"),
        row.get("execution_hostname"),
        row.get("host_verification"),
        row.get("container_execution_permitted"),
    ) != (
        "elmo",
        R298_CANONICAL_ELMO_EXECUTION_HOSTNAME,
        R298_CANONICAL_ELMO_EXECUTION_HOSTNAME,
        R298_ELMO_EXECUTION_HOST_VERIFICATION,
        False,
    ):
        raise R298ValidationError(f"{label} is not the exact non-container Elmo execution identity")
    fqdn = _require_nonempty_string(
        row.get("execution_fqdn"), label=f"{label}.execution_fqdn", maximum=1024
    )
    return {
        "execution_host_role": "elmo",
        "canonical_execution_hostname": R298_CANONICAL_ELMO_EXECUTION_HOSTNAME,
        "execution_hostname": R298_CANONICAL_ELMO_EXECUTION_HOSTNAME,
        "execution_fqdn": fqdn,
        "host_verification": R298_ELMO_EXECUTION_HOST_VERIFICATION,
        "container_execution_permitted": False,
    }


def _validate_collision_raw_manifest(
    manifest: Mapping[str, Any],
    *,
    identity: FileIdentity | None,
    verify_artifacts: bool,
    verify_archive_paths: bool,
) -> dict[str, Any]:
    """Validate the collision-census team's one canonical 30-day manifest.

    This intentionally does not accept the earlier proposed generic manifest
    schema.  The census runner is the sole raw-archive authority and carries
    the ZIP member/deduplication inventory needed to show that no 20-day or
    blended-window fallback was silently substituted.
    """

    if manifest.get("schema") != R298_COLLISION_RAW_CORPUS_MANIFEST_SCHEMA:
        raise R298ValidationError("raw corpus manifest must use the collision-census schema")
    if manifest.get("status") != "sealed_create_only_raw_zip_inventory":
        raise R298ValidationError("raw corpus manifest is not sealed create-only inventory")
    if (
        manifest.get("window_start_utc"),
        manifest.get("window_end_utc"),
        manifest.get("distinct_utc_day_count"),
    ) != (
        R298_RAW_EXPERT_CORPUS_START_UTC,
        R298_RAW_EXPERT_CORPUS_END_UTC,
        R298_RAW_EXPERT_CORPUS_DAYS,
    ):
        raise R298ValidationError("raw corpus manifest does not bind the exact 30-day UTC window")
    if manifest.get("recollection_authorized") is not False or manifest.get("archive_mutation_authorized") is not False:
        raise R298ValidationError("raw corpus manifest permits collection or archive mutation")

    deduplication = _mapping(manifest.get("deduplication"), label="raw corpus manifest deduplication")
    expected_deduplication = {
        "unique_date_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "unique_archive_sha256_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "unique_date_dataset_source_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "duplicate_day_or_source_or_archive_permitted": False,
    }
    if dict(deduplication) != expected_deduplication:
        raise R298ValidationError("raw corpus manifest day/archive/source deduplication drifted")
    episode_dedup = _mapping(
        manifest.get("episode_deduplication"), label="raw corpus manifest episode deduplication"
    )
    required_episode = {
        "episode_identity_algorithm": "payload.id_plus_canonical_content_sha256",
        "duplicate_episode_identity_count": 0,
        "duplicate_episode_id_with_distinct_content_count": 0,
        "excluded_duplicate_mapping": [],
    }
    for name, expected in required_episode.items():
        if episode_dedup.get(name) != expected:
            raise R298ValidationError(f"raw corpus episode deduplication {name} drifted")
    unique_episode_identity_count = _require_exact_int(
        episode_dedup.get("unique_episode_identity_count"),
        label="raw corpus unique episode identity count",
        minimum=1,
    )
    unique_episode_id_count = _require_exact_int(
        episode_dedup.get("unique_episode_id_count"),
        label="raw corpus unique episode ID count",
        minimum=1,
    )
    if unique_episode_identity_count != unique_episode_id_count:
        raise R298ValidationError("raw corpus episode identity and ID counts differ")
    episode_inventory = _require_sha256(
        episode_dedup.get("episode_identity_inventory_sha256"),
        label="raw corpus episode identity inventory",
    )

    archives = manifest.get("archives")
    if not isinstance(archives, (list, tuple)) or len(archives) != R298_RAW_EXPERT_CORPUS_DAYS:
        raise R298ValidationError("raw corpus manifest must enumerate exactly 30 archives")
    expected_dates = _required_raw_utc_dates()
    normalized_archives: list[dict[str, Any]] = []
    total_bytes = 0
    total_members = 0
    total_validated = 0
    seen_archives: set[str] = set()
    seen_source_ids: set[tuple[str, str]] = set()
    for index, raw_archive in enumerate(archives):
        archive = _mapping(raw_archive, label=f"raw corpus archive[{index}]")
        expected_archive_keys = {
            "date",
            "dataset_slug",
            "path",
            "sha256",
            "bytes",
            "zip_json_member_count",
            "validated_episode_count",
            "index_episode_count",
            "source_discrepancy",
        }
        if set(archive) != expected_archive_keys:
            raise R298ValidationError(f"raw corpus archive[{index}] fields drifted")
        utc_date = _require_nonempty_string(archive.get("date"), label=f"raw corpus archive[{index}].date")
        if utc_date != expected_dates[index]:
            raise R298ValidationError("raw corpus archive dates are not the exact contiguous UTC window")
        dataset_slug = _require_nonempty_string(
            archive.get("dataset_slug"), label=f"raw corpus archive[{index}].dataset_slug"
        )
        path = _require_nonempty_string(archive.get("path"), label=f"raw corpus archive[{index}].path")
        digest = _require_sha256(archive.get("sha256"), label=f"raw corpus archive[{index}].sha256")
        if digest in seen_archives or (utc_date, dataset_slug) in seen_source_ids:
            raise R298ValidationError("raw corpus archive/source identities are duplicated")
        seen_archives.add(digest)
        seen_source_ids.add((utc_date, dataset_slug))
        byte_count = _require_exact_int(
            archive.get("bytes"), label=f"raw corpus archive[{index}].bytes", minimum=1
        )
        member_count = _require_exact_int(
            archive.get("zip_json_member_count"),
            label=f"raw corpus archive[{index}].zip_json_member_count",
            minimum=1,
        )
        validated_count = _require_exact_int(
            archive.get("validated_episode_count"),
            label=f"raw corpus archive[{index}].validated_episode_count",
            minimum=1,
        )
        index_count = _require_exact_int(
            archive.get("index_episode_count"),
            label=f"raw corpus archive[{index}].index_episode_count",
            minimum=1,
        )
        if member_count != validated_count:
            raise R298ValidationError("raw corpus archive member and validated episode counts differ")
        if verify_archive_paths:
            path_identity = file_identity(path, label=f"raw corpus archive[{index}]")
            if path_identity.sha256 != digest or path_identity.size_bytes != byte_count:
                raise R298ValidationError(f"raw corpus archive[{index}] physical identity drifted")
        total_bytes += byte_count
        total_members += member_count
        total_validated += validated_count
        normalized_archives.append(
            {
                "date": utc_date,
                "dataset_slug": dataset_slug,
                "path": path,
                "sha256": digest,
                "bytes": byte_count,
                "zip_json_member_count": member_count,
                "validated_episode_count": validated_count,
                "index_episode_count": index_count,
                "source_discrepancy": archive["source_discrepancy"],
            }
        )
    if manifest.get("total_bytes") != total_bytes:
        raise R298ValidationError("raw corpus total bytes do not match archive inventory")
    if manifest.get("total_raw_zip_json_members") != total_members:
        raise R298ValidationError("raw corpus total member count does not match archive inventory")
    if manifest.get("total_validated_episodes") != total_validated:
        raise R298ValidationError("raw corpus validated episode count does not match archive inventory")
    if total_validated != unique_episode_id_count:
        raise R298ValidationError("raw corpus unique episode count does not match validated archive count")
    if _require_exact_int(
        episode_dedup.get("raw_zip_member_count_observed"),
        label="raw corpus observed raw ZIP member count",
        minimum=1,
    ) != total_members:
        raise R298ValidationError("raw corpus deduplication did not scan every raw ZIP member")

    provenance = manifest.get("source_manifest_provenance")
    if not isinstance(provenance, (list, tuple)) or len(provenance) != 2:
        raise R298ValidationError("raw corpus must bind both source-manifest provenance receipts")
    provenance_digests = {
        _require_sha256(
            _mapping(row, label="raw corpus source provenance").get("sha256"),
            label="raw corpus source provenance sha256",
        )
        for row in provenance
    }
    if provenance_digests != {
        "sha256:e29bfe45b092e291924300b12600be7f68f3ace59a931a5edff7f3a09150d8ee",
        "sha256:8fbf78f3005b4e3d8b01474026e393fed778ea54ea43967b781e058f8a4f3f51",
    }:
        raise R298ValidationError("raw corpus source provenance receipts drifted")
    coverage = manifest.get("source_receipt_day_coverage")
    if not isinstance(coverage, (list, tuple)) or len(coverage) != R298_RAW_EXPERT_CORPUS_DAYS:
        raise R298ValidationError("raw corpus source receipt day coverage is incomplete")
    expected_source_receipts = provenance_digests
    covered_source_receipts: set[str] = set()
    for index, raw_coverage in enumerate(coverage):
        day = _mapping(raw_coverage, label=f"raw corpus source receipt coverage[{index}]")
        if set(day) != {"date", "source_receipt_sha256s"} or day.get("date") != expected_dates[index]:
            raise R298ValidationError("raw corpus source receipt coverage is noncontiguous")
        source_receipts = day.get("source_receipt_sha256s")
        if not isinstance(source_receipts, (list, tuple)) or not source_receipts:
            raise R298ValidationError("raw corpus source receipt coverage is malformed")
        normalized_source_receipts = [
            _require_sha256(value, label=f"raw corpus source receipt coverage[{index}]")
            for value in source_receipts
        ]
        if len(set(normalized_source_receipts)) != len(normalized_source_receipts):
            raise R298ValidationError("raw corpus source receipt coverage repeats a receipt")
        if not set(normalized_source_receipts).issubset(expected_source_receipts):
            raise R298ValidationError("raw corpus source receipt coverage names an unknown receipt")
        covered_source_receipts.update(normalized_source_receipts)
    if covered_source_receipts != expected_source_receipts:
        raise R298ValidationError("raw corpus source receipt coverage omits a required receipt")
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="raw corpus manifest")
    return {
        "manifest_sha256": canonical_digest(manifest),
        "manifest": None if identity is None else identity.to_dict(),
        "window_start_utc": R298_RAW_EXPERT_CORPUS_START_UTC,
        "window_end_utc": R298_RAW_EXPERT_CORPUS_END_UTC,
        "utc_partition_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "distinct_utc_partitions": True,
        "contiguous_window": True,
        "archive_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "archive_total_bytes": total_bytes,
        "completed_raw_zip_member_count": total_members,
        "completed_validated_episode_count": total_validated,
        "unique_episode_identity_count": unique_episode_identity_count,
        "episode_identity_inventory_sha256": episode_inventory,
        "archive_identities": normalized_archives,
    }


def _validate_collision_raw_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest_sha256: str,
    source_manifest_provenance_sha256: str,
    source_receipt_day_coverage_sha256: str,
    episode_deduplication_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    if receipt.get("schema") != R298_COLLISION_RAW_CORPUS_RECEIPT_SCHEMA:
        raise R298ValidationError("raw corpus receipt schema mismatch")
    if receipt.get("status") != "passed":
        raise R298ValidationError("raw corpus receipt is not passed")
    _validate_collision_provenance(receipt, label="raw corpus receipt")
    if receipt.get("raw_expert_corpus_manifest_sha256") != manifest_sha256:
        raise R298ValidationError("raw corpus receipt does not bind the supplied manifest")
    if (
        receipt.get("window_start_utc"),
        receipt.get("window_end_utc"),
        receipt.get("distinct_utc_day_count"),
    ) != (
        R298_RAW_EXPERT_CORPUS_START_UTC,
        R298_RAW_EXPERT_CORPUS_END_UTC,
        R298_RAW_EXPERT_CORPUS_DAYS,
    ):
        raise R298ValidationError("raw corpus receipt does not bind exact 30-day window")
    if receipt.get("recollection_authorized") is not False:
        raise R298ValidationError("raw corpus receipt permits recollection")
    if _require_sha256(
        receipt.get("source_manifest_provenance_sha256"),
        label="raw corpus receipt source manifest provenance",
    ) != source_manifest_provenance_sha256:
        raise R298ValidationError("raw corpus receipt does not bind source provenance inventory")
    if _require_sha256(
        receipt.get("source_receipt_day_coverage_sha256"),
        label="raw corpus receipt source receipt day coverage",
    ) != source_receipt_day_coverage_sha256:
        raise R298ValidationError("raw corpus receipt does not bind per-day source coverage")
    if _require_sha256(
        receipt.get("episode_deduplication_sha256"),
        label="raw corpus receipt episode deduplication",
    ) != episode_deduplication_sha256:
        raise R298ValidationError("raw corpus receipt does not bind episode deduplication proof")
    disjoint = _mapping(receipt.get("source_disjointness"), label="raw corpus source disjointness")
    for name, expected in {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
        "training_eligible": False,
    }.items():
        _require_bool(disjoint.get(name), label=f"raw corpus source disjointness.{name}", expected=expected)
    authority = _mapping(receipt.get("runtime_authority"), label="raw corpus runtime authority")
    for name, expected in {
        "elmo_only": True,
        "create_only": True,
        "production_activation": False,
        "inzi_mutation": False,
        "archive_mutation": False,
    }.items():
        _require_bool(authority.get(name), label=f"raw corpus authority.{name}", expected=expected)
    execution_identity = _validate_elmo_execution_identity(
        receipt.get("execution_identity"), label="raw corpus receipt execution_identity"
    )
    resource_observation = _mapping(
        receipt.get("resource_observation"), label="raw corpus receipt resource observation"
    )
    resource_execution_identity = _validate_elmo_execution_identity(
        resource_observation.get("execution_identity"),
        label="raw corpus receipt resource observation execution_identity",
    )
    if resource_execution_identity != execution_identity:
        raise R298ValidationError("raw corpus receipt resource execution identity differs from run identity")
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="raw corpus receipt")
    return {
        "receipt_sha256": canonical_digest(receipt),
        "receipt": None if identity is None else identity.to_dict(),
        "source_disjointness": {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
            "training_eligible": False,
        },
        "execution_identity": execution_identity,
    }


def validate_collision_raw_corpus_evidence(
    manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    *,
    manifest_identity: FileIdentity | None,
    raw_receipt_identity: FileIdentity | None,
    verify_artifacts: bool,
    verify_archive_paths: bool,
) -> dict[str, Any]:
    """Bridge only the census-owned manifest/receipt into r298 validation."""

    normalized_manifest = _validate_collision_raw_manifest(
        manifest,
        identity=manifest_identity,
        verify_artifacts=verify_artifacts,
        verify_archive_paths=verify_archive_paths,
    )
    normalized_receipt = _validate_collision_raw_receipt(
        raw_receipt,
        manifest_sha256=str(normalized_manifest["manifest_sha256"]),
        source_manifest_provenance_sha256=canonical_digest(
            manifest.get("source_manifest_provenance")
        ),
        source_receipt_day_coverage_sha256=canonical_digest(
            manifest.get("source_receipt_day_coverage")
        ),
        episode_deduplication_sha256=canonical_digest(manifest.get("episode_deduplication")),
        identity=raw_receipt_identity,
        verify_artifacts=verify_artifacts,
    )
    return {
        **normalized_manifest,
        # This spelling is consumed by the BO gate and deliberately names the
        # property proved by the manifest, rather than treating an arbitrary
        # file digest as a sufficient corpus claim.
        "validated_deduplicated_manifest_sha256": normalized_manifest["manifest_sha256"],
        "raw_corpus_receipt_sha256": normalized_receipt["receipt_sha256"],
        "raw_corpus_receipt": normalized_receipt["receipt"],
        "raw_source_disjointness": normalized_receipt["source_disjointness"],
        "twenty_day_or_subset_fallback": False,
        "execution_identity": normalized_receipt["execution_identity"],
    }


def validate_refeaturization_split_receipt(
    payload: Mapping[str, Any],
    *,
    raw_manifest_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate source/day/group-disjoint 30-day re-featurization splits."""

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "raw_expert_corpus_manifest_sha256",
        "splits",
        "fallbacks",
    }
    if set(payload) != expected:
        raise R298ValidationError("re-featurization split receipt key inventory changed")
    if payload.get("schema") != R298_REFEATURIZATION_SPLIT_RECEIPT_SCHEMA:
        raise R298ValidationError("re-featurization split receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("re-featurization split receipt owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("re-featurization split receipt is not passed")
    if payload.get("host") != "elmo":
        raise R298ValidationError("re-featurization split receipt must be Elmo-only")
    _validate_attachments(payload.get("source_attachments"))
    if _require_sha256(
        payload.get("raw_expert_corpus_manifest_sha256"),
        label="re-featurization raw manifest",
    ) != raw_manifest_sha256:
        raise R298ValidationError("re-featurization split receipt binds a different raw manifest")
    fallbacks = _mapping(payload.get("fallbacks"), label="re-featurization split fallbacks")
    if set(fallbacks) != {"twenty_day_or_subset_fallback", "adjacent_window_fallback"}:
        raise R298ValidationError("re-featurization split fallback fields changed")
    for name in fallbacks:
        _require_bool(fallbacks[name], label=f"re-featurization fallback {name}", expected=False)

    raw_splits = _mapping(payload.get("splits"), label="re-featurization splits")
    if set(raw_splits) != {"train", "validation", "evaluation"}:
        raise R298ValidationError("re-featurization splits must be train/validation/evaluation")
    expected_dates = set(_required_raw_utc_dates())
    grouped: dict[str, list[set[str]]] = {"source_ids": [], "utc_dates": [], "group_ids": []}
    normalized_splits: dict[str, Any] = {}
    for split_name in ("train", "validation", "evaluation"):
        split = _mapping(raw_splits[split_name], label=f"re-featurization split {split_name}")
        if set(split) != {"source_ids", "utc_dates", "group_ids", "row_count"}:
            raise R298ValidationError(f"re-featurization split {split_name} fields changed")
        values = {
            field: _unique_strings(
                split[field], label=f"re-featurization split {split_name}.{field}"
            )
            for field in ("source_ids", "utc_dates", "group_ids")
        }
        if not set(values["utc_dates"]).issubset(expected_dates):
            raise R298ValidationError(f"re-featurization split {split_name} includes non-window dates")
        row_count = _require_exact_int(
            split.get("row_count"), label=f"re-featurization split {split_name}.row_count", minimum=1
        )
        normalized_splits[split_name] = {**values, "row_count": row_count}
        for field, values_for_field in values.items():
            grouped[field].append(set(values_for_field))
    for field, sets in grouped.items():
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise R298ValidationError(f"re-featurization {field} overlap across splits")
    if set().union(*grouped["utc_dates"]) != expected_dates:
        raise R298ValidationError("re-featurization split days do not cover exact 30-day window")
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="re-featurization split receipt")
    return {
        "schema": R298_REFEATURIZATION_SPLIT_RECEIPT_SCHEMA,
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha256,
        "source_day_group_disjoint": True,
        "splits": normalized_splits,
        "receipt": None if identity is None else identity.to_dict(),
        "receipt_sha256": canonical_digest(payload),
    }


def validate_materialization_receipt(
    payload: Mapping[str, Any],
    *,
    raw_manifest_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
    verify_shard_paths: bool,
) -> dict[str, Any]:
    """Validate transfer-ready Elmo materialization without authorizing transfer.

    This is deliberately a *preparation* proof.  It proves a bounded,
    content-addressed, deterministic, resumable shard layout can be handed to
    a future owner decision.  It rejects any receipt that claims Inzi writes,
    transmission, service mutation, or a move to a production root.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "raw_expert_corpus_manifest_sha256",
        "shards",
        "complete_manifest",
        "deterministic_merge",
        "resumption_and_parallel_safety",
        "authority",
    }
    if set(payload) != expected:
        raise R298ValidationError("materialization receipt key inventory changed")
    if payload.get("schema") != R298_MATERIALIZATION_RECEIPT_SCHEMA:
        raise R298ValidationError("materialization receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("materialization receipt owner revision mismatch")
    if payload.get("status") != "passed_transfer_ready_no_transfer" or payload.get("passed") is not True:
        raise R298ValidationError("materialization receipt is not transfer-ready/no-transfer pass")
    if payload.get("host") != "elmo":
        raise R298ValidationError("materialization receipt must be Elmo-only")
    _validate_attachments(payload.get("source_attachments"))
    if _require_sha256(
        payload.get("raw_expert_corpus_manifest_sha256"),
        label="materialization raw corpus manifest",
    ) != raw_manifest_sha256:
        raise R298ValidationError("materialization receipt binds a different raw corpus manifest")

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, (list, tuple)) or not raw_shards:
        raise R298ValidationError("materialization receipt requires nonempty physical shard inventory")
    normalized_shards: list[dict[str, Any]] = []
    shard_ids: set[str] = set()
    shard_digests: set[str] = set()
    covered_dates: set[str] = set()
    expected_dates = set(_required_raw_utc_dates())
    for index, raw_shard in enumerate(raw_shards):
        shard = _mapping(raw_shard, label=f"materialization shard[{index}]")
        expected_shard = {
            "shard_id",
            "relative_path",
            "sha256",
            "content_sha256",
            "size_bytes",
            "utc_dates",
        }
        if set(shard) != expected_shard:
            raise R298ValidationError(f"materialization shard[{index}] fields changed")
        shard_id = _require_identifier(shard.get("shard_id"), label=f"materialization shard[{index}].shard_id")
        relative_path = _require_nonempty_string(
            shard.get("relative_path"), label=f"materialization shard[{index}].relative_path"
        )
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise R298ValidationError("materialization shard path must be relative and containment-safe")
        digest = _require_sha256(shard.get("sha256"), label=f"materialization shard[{index}].sha256")
        content_digest = _require_sha256(
            shard.get("content_sha256"), label=f"materialization shard[{index}].content_sha256"
        )
        if digest != content_digest:
            raise R298ValidationError("materialization shard is not content-addressed by its physical digest")
        size = _require_exact_int(
            shard.get("size_bytes"), label=f"materialization shard[{index}].size_bytes", minimum=1
        )
        if size > R298_MATERIALIZATION_SHARD_MAX_BYTES:
            raise R298ValidationError("materialization shard exceeds 96-MiB physical cap")
        dates = _unique_strings(shard.get("utc_dates"), label=f"materialization shard[{index}].utc_dates")
        if not set(dates).issubset(expected_dates):
            raise R298ValidationError("materialization shard covers a non-window UTC date")
        if shard_id in shard_ids or digest in shard_digests:
            raise R298ValidationError("materialization shard IDs/content digests must be unique")
        if covered_dates & set(dates):
            raise R298ValidationError("materialization shard UTC partition coverage overlaps")
        shard_ids.add(shard_id)
        shard_digests.add(digest)
        covered_dates.update(dates)
        if verify_shard_paths:
            physical = file_identity(relative, label=f"materialization shard[{index}]")
            if physical.sha256 != digest or physical.size_bytes != size:
                raise R298ValidationError(f"materialization shard[{index}] physical identity drifted")
        normalized_shards.append(
            {
                "shard_id": shard_id,
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "content_sha256": digest,
                "size_bytes": size,
                "utc_dates": sorted(dates),
            }
        )
    if covered_dates != expected_dates:
        raise R298ValidationError("materialization shard inventory does not cover exact 30 UTC dates")

    complete_manifest = _mapping(
        payload.get("complete_manifest"), label="materialization complete manifest"
    )
    if set(complete_manifest) != {"sha256", "shard_sha256s", "complete"}:
        raise R298ValidationError("materialization complete manifest fields changed")
    complete_manifest_sha256 = _require_sha256(
        complete_manifest.get("sha256"), label="materialization complete manifest sha256"
    )
    manifest_shards = _unique_strings(
        complete_manifest.get("shard_sha256s"), label="materialization complete manifest shard sha256s"
    )
    for digest in manifest_shards:
        _require_sha256(digest, label="materialization complete manifest shard sha256")
    if set(manifest_shards) != shard_digests:
        raise R298ValidationError("materialization complete manifest shard inventory differs")
    _require_bool(complete_manifest.get("complete"), label="materialization complete manifest", expected=True)

    merge = _mapping(payload.get("deterministic_merge"), label="materialization deterministic merge")
    if set(merge) != {"algorithm", "input_shard_sha256s", "output_sha256", "deterministic"}:
        raise R298ValidationError("materialization merge fields changed")
    merge_algorithm = _require_nonempty_string(merge.get("algorithm"), label="materialization merge algorithm")
    merge_inputs = _unique_strings(
        merge.get("input_shard_sha256s"), label="materialization merge input shard sha256s"
    )
    for digest in merge_inputs:
        _require_sha256(digest, label="materialization merge input shard sha256")
    if set(merge_inputs) != shard_digests:
        raise R298ValidationError("materialization merge does not consume complete shard inventory")
    output_digest = _require_sha256(
        merge.get("output_sha256"), label="materialization merge output sha256"
    )
    _require_bool(merge.get("deterministic"), label="materialization deterministic merge", expected=True)

    resume = _mapping(
        payload.get("resumption_and_parallel_safety"),
        label="materialization resumption/parallel safety",
    )
    expected_resume = {
        "create_only",
        "resumable",
        "parallel_safe",
        "completed_shard_idempotent",
        "resume_manifest_sha256",
    }
    if set(resume) != expected_resume:
        raise R298ValidationError("materialization resumption fields changed")
    for name in (
        "create_only",
        "resumable",
        "parallel_safe",
        "completed_shard_idempotent",
    ):
        _require_bool(resume.get(name), label=f"materialization {name}", expected=True)
    if _require_sha256(
        resume.get("resume_manifest_sha256"), label="materialization resume manifest"
    ) != complete_manifest_sha256:
        raise R298ValidationError("materialization resume manifest differs from complete manifest")

    authority = _mapping(payload.get("authority"), label="materialization authority")
    expected_authority = {
        "inzi_side_effects",
        "transfer_to_inzi",
        "production_mutation",
        "managed_service_mutation",
    }
    if set(authority) != expected_authority:
        raise R298ValidationError("materialization authority fields changed")
    for name in expected_authority:
        _require_bool(authority.get(name), label=f"materialization authority.{name}", expected=False)
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="materialization receipt")
    return {
        "schema": R298_MATERIALIZATION_RECEIPT_SCHEMA,
        "status": "passed_transfer_ready_no_transfer",
        "receipt": None if identity is None else identity.to_dict(),
        "receipt_sha256": canonical_digest(payload),
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha256,
        "shards": normalized_shards,
        "complete_manifest_sha256": complete_manifest_sha256,
        "deterministic_merge": {
            "algorithm": merge_algorithm,
            "output_sha256": output_digest,
            "deterministic": True,
        },
        "resumable_parallel_safe": True,
        "transfer_to_inzi": False,
        "inzi_side_effects": False,
    }


def validate_candidate_training_receipt(
    payload: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    raw_manifest_sha256: str,
    split_receipt_sha256: str,
    collision_census_sha256: str,
    schema_freeze_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Require census-supported, frozen-backbone Elmo-only candidate training."""

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "candidate",
        "inputs",
        "frozen_modules",
        "branch_support",
        "trained_branch_ids",
        "authority",
    }
    if set(payload) != expected:
        raise R298ValidationError("candidate training receipt key inventory changed")
    if payload.get("schema") != R298_CANDIDATE_TRAINING_RECEIPT_SCHEMA:
        raise R298ValidationError("candidate training receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("candidate training receipt owner revision mismatch")
    if payload.get("status") != "passed_census_supported_elmo_only" or payload.get("passed") is not True:
        raise R298ValidationError("candidate training receipt is not a census-supported pass")
    if payload.get("host") != "elmo":
        raise R298ValidationError("candidate training receipt must be Elmo-only")
    _validate_attachments(payload.get("source_attachments"))
    candidate_row = _mapping(payload.get("candidate"), label="candidate training candidate")
    if set(candidate_row) != {"candidate_id", "checkpoint_sha256", "checkpoint_size_bytes"}:
        raise R298ValidationError("candidate training candidate fields changed")
    if candidate_row.get("candidate_id") != candidate.get("candidate_id"):
        raise R298ValidationError("candidate training receipt binds a different candidate ID")
    if candidate_row.get("checkpoint_sha256") != candidate.get("checkpoint_sha256"):
        raise R298ValidationError("candidate training receipt binds a different candidate checkpoint")
    if candidate_row.get("checkpoint_size_bytes") != candidate.get("checkpoint_size_bytes"):
        raise R298ValidationError("candidate training receipt binds a different candidate checkpoint size")

    inputs = _mapping(payload.get("inputs"), label="candidate training inputs")
    expected_inputs = {
        "raw_expert_corpus_manifest_sha256",
        "refeaturization_split_receipt_sha256",
        "collision_census_receipt_sha256",
        "schema_freeze_manifest_sha256",
    }
    if set(inputs) != expected_inputs:
        raise R298ValidationError("candidate training input fields changed")
    expected_input_values = {
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha256,
        "refeaturization_split_receipt_sha256": split_receipt_sha256,
        "collision_census_receipt_sha256": collision_census_sha256,
        "schema_freeze_manifest_sha256": schema_freeze_sha256,
    }
    for name, expected_value in expected_input_values.items():
        if _require_sha256(inputs.get(name), label=f"candidate training inputs.{name}") != expected_value:
            raise R298ValidationError(f"candidate training input {name} does not bind required evidence")

    frozen_modules = _mapping(payload.get("frozen_modules"), label="candidate training frozen modules")
    expected_frozen = {
        "neural_backbone",
        "existing_heads",
        "fusion",
        "own_deck_routes",
        "matchup_adapters",
    }
    if set(frozen_modules) != expected_frozen:
        raise R298ValidationError("candidate training frozen-module inventory changed")
    for name in expected_frozen:
        _require_bool(frozen_modules[name], label=f"candidate training frozen {name}", expected=True)

    support = _mapping(payload.get("branch_support"), label="candidate training branch support")
    expected_support = {
        "collision_census_receipt_sha256",
        "branch_support_report_sha256",
        "supported_branch_ids",
        "unsupported_branch_ids",
        "unsupported_branches_exact_zero_inert",
    }
    if set(support) != expected_support:
        raise R298ValidationError("candidate training branch-support fields changed")
    if _require_sha256(
        support.get("collision_census_receipt_sha256"),
        label="candidate training branch support collision census",
    ) != collision_census_sha256:
        raise R298ValidationError("candidate training branch support binds wrong collision census")
    support_report = _require_sha256(
        support.get("branch_support_report_sha256"),
        label="candidate training branch support report",
    )
    supported = _unique_strings(
        support.get("supported_branch_ids"), label="candidate training supported branch IDs"
    )
    unsupported = _unique_strings(
        support.get("unsupported_branch_ids"),
        label="candidate training unsupported branch IDs",
        allow_empty=True,
    )
    if set(supported) & set(unsupported):
        raise R298ValidationError("candidate training support/unsupported branch IDs overlap")
    _require_bool(
        support.get("unsupported_branches_exact_zero_inert"),
        label="candidate training unsupported branches zero/inert",
        expected=True,
    )
    trained = _unique_strings(
        payload.get("trained_branch_ids"), label="candidate training trained branch IDs"
    )
    if not set(trained).issubset(set(supported)):
        raise R298ValidationError("candidate training includes a branch not supported by census")

    authority = _mapping(payload.get("authority"), label="candidate training authority")
    expected_authority = {
        "isolated_elmo_training",
        "inzi_mutation",
        "production_activation",
        "checkpoint_replacement",
        "selector_mutation",
        "submission",
    }
    if set(authority) != expected_authority:
        raise R298ValidationError("candidate training authority fields changed")
    _require_bool(authority.get("isolated_elmo_training"), label="candidate training isolated Elmo training", expected=True)
    for name in expected_authority - {"isolated_elmo_training"}:
        _require_bool(authority.get(name), label=f"candidate training authority.{name}", expected=False)
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="candidate training receipt")
    return {
        "schema": R298_CANDIDATE_TRAINING_RECEIPT_SCHEMA,
        "status": "passed_census_supported_elmo_only",
        "receipt": None if identity is None else identity.to_dict(),
        "receipt_sha256": canonical_digest(payload),
        "candidate_id": candidate["candidate_id"],
        "raw_expert_corpus_manifest_sha256": raw_manifest_sha256,
        "refeaturization_split_receipt_sha256": split_receipt_sha256,
        "collision_census_receipt_sha256": collision_census_sha256,
        "schema_freeze_manifest_sha256": schema_freeze_sha256,
        "branch_support_report_sha256": support_report,
        "trained_branch_ids": trained,
        "supported_branch_ids": supported,
        "unsupported_branch_ids": unsupported,
        "unsupported_branches_exact_zero_inert": True,
        "frozen_modules": {name: True for name in sorted(expected_frozen)},
        "inzi_mutation": False,
        "production_activation": False,
    }


def validate_raw_expert_corpus_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate the sole allowed 30-day Elmo raw-expert corpus window.

    This intentionally retains the per-day episode and content identities in
    the manifest rather than accepting a Boolean such as ``deduplicated``.
    The validator can therefore prove there are no repeated IDs or payload
    digests across calendar days, rather than trusting a reporting claim.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "window",
        "partitions",
        "totals",
        "split_disjointness",
        "fallbacks",
    }
    if set(payload) != expected:
        raise R298ValidationError("raw expert corpus manifest key inventory changed")
    if payload.get("schema") != R298_RAW_EXPERT_CORPUS_MANIFEST_SCHEMA:
        raise R298ValidationError("raw expert corpus manifest schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("raw expert corpus manifest owner revision mismatch")
    if payload.get("status") != "passed_validated_deduplicated_30_day" or payload.get("passed") is not True:
        raise R298ValidationError("raw expert corpus manifest is not a validated 30-day pass")
    if payload.get("host") != "elmo":
        raise R298ValidationError("raw expert corpus manifest must be produced on Elmo")
    _validate_attachments(payload.get("source_attachments"))

    window = _mapping(payload.get("window"), label="raw expert corpus window")
    expected_window = {
        "start_utc",
        "end_utc",
        "inclusive",
        "contiguous",
        "utc_partition_count",
    }
    if set(window) != expected_window:
        raise R298ValidationError("raw expert corpus window fields are incomplete")
    if window.get("start_utc") != R298_RAW_EXPERT_CORPUS_START_UTC:
        raise R298ValidationError("raw expert corpus window start must be 2026-07-13 UTC")
    if window.get("end_utc") != R298_RAW_EXPERT_CORPUS_END_UTC:
        raise R298ValidationError("raw expert corpus window end must be 2026-08-11 UTC")
    _require_bool(window.get("inclusive"), label="raw expert corpus window.inclusive", expected=True)
    _require_bool(window.get("contiguous"), label="raw expert corpus window.contiguous", expected=True)
    if _require_exact_int(
        window.get("utc_partition_count"),
        label="raw expert corpus window.utc_partition_count",
        minimum=1,
    ) != R298_RAW_EXPERT_CORPUS_DAYS:
        raise R298ValidationError("raw expert corpus must contain exactly 30 UTC partitions")

    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, (list, tuple)):
        raise R298ValidationError("raw expert corpus partitions must be a list")
    if len(raw_partitions) != R298_RAW_EXPERT_CORPUS_DAYS:
        raise R298ValidationError("raw expert corpus manifest must enumerate exactly 30 partitions")
    expected_dates = _required_raw_utc_dates()
    seen_dates: list[str] = []
    all_episode_ids: list[str] = []
    all_content_digests: list[str] = []
    normalized_partitions: list[dict[str, Any]] = []
    for index, raw_partition in enumerate(raw_partitions):
        partition = _mapping(raw_partition, label=f"raw expert corpus partition[{index}]")
        expected_partition = {
            "utc_date",
            "archive",
            "archive_verified",
            "source_provenance",
            "episode_ids",
            "content_digests",
        }
        if set(partition) != expected_partition:
            raise R298ValidationError(
                f"raw expert corpus partition[{index}] fields are incomplete"
            )
        utc_date = _require_nonempty_string(
            partition.get("utc_date"), label=f"raw expert corpus partition[{index}].utc_date"
        )
        try:
            parsed_date = date.fromisoformat(utc_date)
        except ValueError as exc:
            raise R298ValidationError(
                f"raw expert corpus partition[{index}].utc_date is not ISO UTC date"
            ) from exc
        if parsed_date.isoformat() != utc_date:
            raise R298ValidationError(
                f"raw expert corpus partition[{index}].utc_date is not canonical ISO date"
            )
        seen_dates.append(utc_date)
        archive = _identity_from_mapping(
            partition.get("archive"), label=f"raw expert corpus partition[{index}].archive"
        )
        if verify_artifacts:
            _validate_identity_on_disk(
                archive, label=f"raw expert corpus partition[{index}].archive"
            )
        _require_bool(
            partition.get("archive_verified"),
            label=f"raw expert corpus partition[{index}].archive_verified",
            expected=True,
        )
        provenance = _mapping(
            partition.get("source_provenance"),
            label=f"raw expert corpus partition[{index}].source_provenance",
        )
        if set(provenance) != {"kind", "source_id"}:
            raise R298ValidationError(
                f"raw expert corpus partition[{index}].source_provenance fields changed"
            )
        if provenance.get("kind") != "existing_elmo_raw_expert_archive":
            raise R298ValidationError(
                f"raw expert corpus partition[{index}] is not an existing Elmo raw archive"
            )
        source_id = _require_nonempty_string(
            provenance.get("source_id"),
            label=f"raw expert corpus partition[{index}].source_provenance.source_id",
        )
        episode_ids = _unique_strings(
            partition.get("episode_ids"),
            label=f"raw expert corpus partition[{index}].episode_ids",
        )
        content_digests = [
            _require_sha256(
                digest,
                label=f"raw expert corpus partition[{index}].content_digests[]",
            )
            for digest in partition.get("content_digests", ())
        ]
        if not content_digests:
            raise R298ValidationError(
                f"raw expert corpus partition[{index}].content_digests must not be empty"
            )
        if len(set(content_digests)) != len(content_digests):
            raise R298ValidationError(
                f"raw expert corpus partition[{index}].content_digests must be unique"
            )
        if len(episode_ids) != len(content_digests):
            raise R298ValidationError(
                f"raw expert corpus partition[{index}] episode/content counts differ"
            )
        all_episode_ids.extend(episode_ids)
        all_content_digests.extend(content_digests)
        normalized_partitions.append(
            {
                "utc_date": utc_date,
                "archive": archive.to_dict(),
                "archive_verified": True,
                "source_provenance": {
                    "kind": "existing_elmo_raw_expert_archive",
                    "source_id": source_id,
                },
                "episode_ids": episode_ids,
                "content_digests": content_digests,
            }
        )
    if tuple(sorted(seen_dates)) != expected_dates:
        raise R298ValidationError(
            "raw expert corpus days must be the exact 2026-07-13 through 2026-08-11 UTC window"
        )
    if len(set(all_episode_ids)) != len(all_episode_ids):
        raise R298ValidationError("raw expert corpus has duplicate episode IDs across days")
    if len(set(all_content_digests)) != len(all_content_digests):
        raise R298ValidationError("raw expert corpus has duplicate content digests across days")

    totals = _mapping(payload.get("totals"), label="raw expert corpus totals")
    expected_totals = {
        "partition_count",
        "episode_count",
        "content_digest_count",
        "unique_episode_id_count",
        "unique_content_digest_count",
    }
    if set(totals) != expected_totals:
        raise R298ValidationError("raw expert corpus totals fields are incomplete")
    expected_counts = {
        "partition_count": R298_RAW_EXPERT_CORPUS_DAYS,
        "episode_count": len(all_episode_ids),
        "content_digest_count": len(all_content_digests),
        "unique_episode_id_count": len(all_episode_ids),
        "unique_content_digest_count": len(all_content_digests),
    }
    for name, expected_count in expected_counts.items():
        actual_count = _require_exact_int(
            totals.get(name), label=f"raw expert corpus totals.{name}", minimum=1
        )
        if actual_count != expected_count:
            raise R298ValidationError(
                f"raw expert corpus totals.{name} does not match enumerated evidence"
            )

    fallbacks = _mapping(payload.get("fallbacks"), label="raw expert corpus fallbacks")
    if set(fallbacks) != {
        "twenty_day_fallback_used",
        "subset_or_adjacent_window_fallback_used",
        "recollection_without_absence_proof",
    }:
        raise R298ValidationError("raw expert corpus fallback fields are incomplete")
    for name in fallbacks:
        _require_bool(fallbacks[name], label=f"raw expert corpus fallbacks.{name}", expected=False)

    split_disjointness = _mapping(
        payload.get("split_disjointness"), label="raw expert corpus split disjointness"
    )
    expected_split_keys = {"simultaneously_disjoint_by", "train", "validation", "evaluation"}
    if set(split_disjointness) != expected_split_keys:
        raise R298ValidationError("raw expert corpus split disjointness fields are incomplete")
    if tuple(split_disjointness["simultaneously_disjoint_by"]) != (
        "source",
        "utc_day_partition",
        "group",
    ):
        raise R298ValidationError("raw expert corpus split axes must be source/day/group")
    normalized_splits: dict[str, dict[str, list[str]]] = {}
    split_values: dict[str, list[set[str]]] = {
        "source_ids": [],
        "utc_dates": [],
        "group_ids": [],
    }
    for split_name in ("train", "validation", "evaluation"):
        split = _mapping(
            split_disjointness[split_name],
            label=f"raw expert corpus split {split_name}",
        )
        if set(split) != {"source_ids", "utc_dates", "group_ids"}:
            raise R298ValidationError(
                f"raw expert corpus split {split_name} fields are incomplete"
            )
        normalized_split = {
            field: _unique_strings(
                split[field], label=f"raw expert corpus split {split_name}.{field}"
            )
            for field in ("source_ids", "utc_dates", "group_ids")
        }
        if not set(normalized_split["utc_dates"]).issubset(set(expected_dates)):
            raise R298ValidationError(
                f"raw expert corpus split {split_name} has a non-window UTC day"
            )
        normalized_splits[split_name] = normalized_split
        for field, values in normalized_split.items():
            split_values[field].append(set(values))
    for field, sets in split_values.items():
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise R298ValidationError(
                f"raw expert corpus {field} are not disjoint across train/validation/evaluation"
            )
    if set().union(*split_values["utc_dates"]) != set(expected_dates):
        raise R298ValidationError(
            "raw expert corpus split UTC days must cover the exact 30-day window"
        )

    normalized: dict[str, Any] = {
        "schema": R298_RAW_EXPERT_CORPUS_MANIFEST_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed_validated_deduplicated_30_day",
        "host": "elmo",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "window": {
            "start_utc": R298_RAW_EXPERT_CORPUS_START_UTC,
            "end_utc": R298_RAW_EXPERT_CORPUS_END_UTC,
            "inclusive": True,
            "contiguous": True,
            "utc_partition_count": R298_RAW_EXPERT_CORPUS_DAYS,
        },
        "partitions": normalized_partitions,
        "totals": expected_counts,
        "split_disjointness": {
            "simultaneously_disjoint_by": ["source", "utc_day_partition", "group"],
            **normalized_splits,
        },
        "fallbacks": {
            "twenty_day_fallback_used": False,
            "subset_or_adjacent_window_fallback_used": False,
            "recollection_without_absence_proof": False,
        },
    }
    if manifest_identity is not None:
        if verify_artifacts:
            _validate_identity_on_disk(manifest_identity, label="raw expert corpus manifest")
        normalized["manifest"] = manifest_identity.to_dict()
    return normalized


def validate_resource_receipt(
    payload: Mapping[str, Any],
    *,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    """Validate one isolated-Elmo resource envelope receipt.

    Resource accounting is part of the experiment boundary, not a dashboard
    nicety.  A receipt has to show its measured aggregate RAM peak stayed
    under 96 GiB and that it neither overlapped another memory-heavy phase nor
    changed ZFS/ARC/L2ARC or a managed service.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "phase",
        "resource_envelope",
        "peaks",
        "boundaries",
    }
    if set(payload) != expected:
        raise R298ValidationError("r298 resource receipt key inventory changed")
    if payload.get("schema") != R298_RESOURCE_RECEIPT_SCHEMA:
        raise R298ValidationError("r298 resource receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("r298 resource receipt owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("r298 resource receipt is not passed")
    if payload.get("host") != "elmo":
        raise R298ValidationError("r298 resource receipt must be produced on Elmo")
    _validate_attachments(payload.get("source_attachments"))
    phase = _require_nonempty_string(payload.get("phase"), label="resource receipt phase")
    if phase not in RESOURCE_JOB_NAMES:
        raise R298ValidationError("resource receipt phase is not an r298 isolated job")
    if expected_phase is not None and phase != expected_phase:
        raise R298ValidationError("resource receipt phase does not match its evidence lane")

    envelope = _mapping(payload.get("resource_envelope"), label="resource receipt envelope")
    if set(envelope) != {
        "experiment_ram_hard_ceiling_bytes",
        "experiment_ram_measurement_method",
        "memory_heavy",
        "concurrent_memory_heavy_phases",
        "all_elmo_cpu_gpu_permitted",
    }:
        raise R298ValidationError("resource receipt envelope fields are incomplete")
    if _require_exact_int(
        envelope.get("experiment_ram_hard_ceiling_bytes"),
        label="resource receipt experiment RAM hard ceiling",
        minimum=1,
    ) != R298_EXPERIMENT_RAM_HARD_CEILING_BYTES:
        raise R298ValidationError("resource receipt has the wrong 96-GiB RAM ceiling")
    ram_method = _require_nonempty_string(
        envelope.get("experiment_ram_measurement_method"),
        label="resource receipt experiment RAM measurement method",
    )
    memory_heavy = _require_bool(
        envelope.get("memory_heavy"), label="resource receipt memory_heavy"
    )
    concurrent_memory_heavy = _require_exact_int(
        envelope.get("concurrent_memory_heavy_phases"),
        label="resource receipt concurrent memory-heavy phases",
        minimum=0,
    )
    if concurrent_memory_heavy > 1 or (
        memory_heavy and concurrent_memory_heavy != 1
    ):
        raise R298ValidationError(
            "resource receipt violates one-memory-heavy-phase-at-a-time boundary"
        )
    _require_bool(
        envelope.get("all_elmo_cpu_gpu_permitted"),
        label="resource receipt all Elmo CPU/GPU permitted",
        expected=True,
    )

    peaks = _mapping(payload.get("peaks"), label="resource receipt peaks")
    if set(peaks) != {"experiment_ram_peak_bytes", "cpu", "gpu", "io"}:
        raise R298ValidationError("resource receipt peak fields are incomplete")
    ram_peak = _require_exact_int(
        peaks.get("experiment_ram_peak_bytes"),
        label="resource receipt experiment RAM peak",
        minimum=0,
    )
    if ram_peak > R298_EXPERIMENT_RAM_HARD_CEILING_BYTES:
        raise R298ValidationError("resource receipt exceeded the 96-GiB experiment RAM ceiling")
    cpu = _mapping(peaks.get("cpu"), label="resource receipt CPU peaks")
    if set(cpu) != {"peak_processes", "peak_utilization_percent"}:
        raise R298ValidationError("resource receipt CPU peak fields are incomplete")
    cpu_processes = _require_exact_int(
        cpu.get("peak_processes"), label="resource receipt CPU peak processes", minimum=0
    )
    cpu_utilization = _require_percent(
        cpu.get("peak_utilization_percent"),
        label="resource receipt CPU peak utilization",
    )

    raw_gpu = peaks.get("gpu")
    normalized_gpu: Any
    if raw_gpu == "not_applicable":
        normalized_gpu = "not_applicable"
    else:
        gpu = _mapping(raw_gpu, label="resource receipt GPU peaks")
        if not gpu:
            raise R298ValidationError("resource receipt GPU peaks must name a device or not_applicable")
        normalized_gpu = {}
        for device, device_peaks in gpu.items():
            device_name = _require_nonempty_string(
                device, label="resource receipt GPU device", maximum=256
            )
            row = _mapping(
                device_peaks, label=f"resource receipt GPU peaks {device_name}"
            )
            if set(row) != {"memory_peak_bytes", "utilization_peak_percent"}:
                raise R298ValidationError(
                    f"resource receipt GPU peak fields changed for {device_name}"
                )
            normalized_gpu[device_name] = {
                "memory_peak_bytes": _require_exact_int(
                    row.get("memory_peak_bytes"),
                    label=f"resource receipt GPU memory peak {device_name}",
                    minimum=0,
                ),
                "utilization_peak_percent": _require_percent(
                    row.get("utilization_peak_percent"),
                    label=f"resource receipt GPU utilization {device_name}",
                ),
            }
    io = _mapping(peaks.get("io"), label="resource receipt I/O peaks")
    expected_io = {
        "read_bytes",
        "write_bytes",
        "peak_read_bytes_per_second",
        "peak_write_bytes_per_second",
    }
    if set(io) != expected_io:
        raise R298ValidationError("resource receipt I/O peak fields are incomplete")
    normalized_io = {
        name: _require_exact_int(
            io.get(name), label=f"resource receipt I/O {name}", minimum=0
        )
        for name in sorted(expected_io)
    }

    boundaries = _mapping(payload.get("boundaries"), label="resource receipt boundaries")
    expected_boundaries = {
        "zfs_arc_l2arc_tuning_or_reconfiguration",
        "managed_service_mutation",
        "inzi_or_production_mutation",
    }
    if set(boundaries) != expected_boundaries:
        raise R298ValidationError("resource receipt boundary fields are incomplete")
    for name in expected_boundaries:
        _require_bool(boundaries[name], label=f"resource receipt boundary {name}", expected=False)

    return {
        "schema": R298_RESOURCE_RECEIPT_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed",
        "host": "elmo",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "phase": phase,
        "resource_envelope": {
            "experiment_ram_hard_ceiling_bytes": R298_EXPERIMENT_RAM_HARD_CEILING_BYTES,
            "experiment_ram_measurement_method": ram_method,
            "memory_heavy": memory_heavy,
            "concurrent_memory_heavy_phases": concurrent_memory_heavy,
            "all_elmo_cpu_gpu_permitted": True,
        },
        "peaks": {
            "experiment_ram_peak_bytes": ram_peak,
            "cpu": {
                "peak_processes": cpu_processes,
                "peak_utilization_percent": cpu_utilization,
            },
            "gpu": normalized_gpu,
            "io": normalized_io,
        },
        "boundaries": {name: False for name in sorted(expected_boundaries)},
    }


def validate_schema_freeze_receipt(
    payload: Mapping[str, Any],
    *,
    receipt_identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate the rev-3 pre-corpus schema-freeze / zero-bypass gate."""

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "goal_contract_sha256",
        "schemas",
        "new_branch_inventory",
        "default_zero_and_inert_attestation",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    }
    if set(payload) != expected:
        raise R298ValidationError("schema freeze receipt key inventory changed")
    if payload.get("schema") != R298_SCHEMA_FREEZE_RECEIPT_SCHEMA:
        raise R298ValidationError("schema freeze receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("schema freeze receipt owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("schema freeze receipt is not passed")
    if payload.get("host") != "elmo":
        raise R298ValidationError("schema freeze receipt must be produced on Elmo")
    _validate_attachments(payload.get("source_attachments"))
    if _require_sha256(
        payload.get("goal_contract_sha256"), label="schema freeze receipt goal contract"
    ) != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("schema freeze receipt does not bind the dedicated rev-3 contract")
    schemas = _mapping(payload.get("schemas"), label="schema freeze receipt schemas")
    expected_schemas = {
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    }
    if set(schemas) != expected_schemas:
        raise R298ValidationError("schema freeze receipt schema identities are incomplete")
    normalized_schemas = {
        name: _require_sha256(
            schemas[name], label=f"schema freeze receipt schemas.{name}"
        )
        for name in sorted(expected_schemas)
    }
    branches = _unique_strings(
        payload.get("new_branch_inventory"),
        label="schema freeze receipt new branch inventory",
    )
    _require_bool(
        payload.get("default_zero_and_inert_attestation"),
        label="schema freeze receipt default zero/inert",
        expected=True,
    )
    _require_bool(
        payload.get("layer_off_bit_identical_baseline_logits"),
        label="schema freeze receipt layer-off logits",
        expected=True,
    )
    _require_bool(
        payload.get("layer_off_identical_legal_choice"),
        label="schema freeze receipt layer-off legal choice",
        expected=True,
    )
    if receipt_identity is not None and verify_artifacts:
        _validate_identity_on_disk(receipt_identity, label="schema freeze receipt")
    normalized: dict[str, Any] = {
        "schema": R298_SCHEMA_FREEZE_RECEIPT_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed",
        "host": "elmo",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "goal_contract_sha256": DEDICATED_GOAL_CONTRACT_SHA256,
        "schemas": normalized_schemas,
        "new_branch_inventory": branches,
        "default_zero_and_inert_attestation": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
    }
    if receipt_identity is not None:
        normalized["receipt"] = receipt_identity.to_dict()
    return normalized


def validate_collision_schema_freeze_evidence(
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    *,
    schema_manifest_identity: FileIdentity | None,
    zero_bypass_identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Bridge the producer-owned rev-3 schema-freeze/zero-bypass gate.

    The collision runner consumes these exact two artifacts before it is even
    allowed to re-featurize the 30-day corpus.  Keeping the bridge strict here
    makes the final r298 receipt demonstrate that ordering instead of merely
    restating it in a phase assertion.
    """

    if schema_manifest.get("schema") != R298_COLLISION_FROZEN_SCHEMA_MANIFEST_SCHEMA:
        raise R298ValidationError("schema-freeze manifest must use collision-census producer schema")
    if schema_manifest.get("status") != "frozen_zero_inert_before_refeaturization":
        raise R298ValidationError("schema-freeze manifest is not frozen zero/inert")
    _validate_collision_provenance(schema_manifest, label="schema-freeze manifest")
    required_schema_fields = {
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    }
    normalized_schemas = {
        name: _require_sha256(
            schema_manifest.get(name), label=f"schema-freeze manifest {name}"
        )
        for name in sorted(required_schema_fields)
    }
    branches = _unique_strings(
        schema_manifest.get("new_branch_inventory"),
        label="schema-freeze manifest new branch inventory",
    )
    if schema_manifest.get("default_zero_and_inert_attestation") is not True:
        raise R298ValidationError("schema-freeze manifest does not attest default zero/inert branches")
    if schema_manifest.get("runtime_wired") is not False:
        raise R298ValidationError("schema-freeze manifest is wired to runtime")
    manifest_sha256 = canonical_digest(schema_manifest)
    if schema_manifest_identity is not None and verify_artifacts:
        _validate_identity_on_disk(schema_manifest_identity, label="schema-freeze manifest")

    if zero_bypass_receipt.get("schema") != R298_COLLISION_ZERO_BYPASS_RECEIPT_SCHEMA:
        raise R298ValidationError("zero-bypass receipt must use collision-census producer schema")
    if zero_bypass_receipt.get("status") != "passed_exact_baseline_logits":
        raise R298ValidationError("zero-bypass receipt is not an exact-logit pass")
    _validate_collision_provenance(zero_bypass_receipt, label="zero-bypass receipt")
    if zero_bypass_receipt.get("frozen_schema_manifest_sha256") != manifest_sha256:
        raise R298ValidationError("zero-bypass receipt does not bind frozen schema manifest")
    for name in (
        "all_new_routes_exact_zero_or_bypassed",
        "checklist_provenance_schema_frozen",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    ):
        _require_bool(zero_bypass_receipt.get(name), label=f"zero-bypass receipt {name}", expected=True)
    if zero_bypass_identity is not None and verify_artifacts:
        _validate_identity_on_disk(zero_bypass_identity, label="zero-bypass receipt")
    return {
        "schema_manifest_sha256": manifest_sha256,
        "schema_manifest": (
            None if schema_manifest_identity is None else schema_manifest_identity.to_dict()
        ),
        "zero_bypass_receipt_sha256": canonical_digest(zero_bypass_receipt),
        "zero_bypass_receipt": (
            None if zero_bypass_identity is None else zero_bypass_identity.to_dict()
        ),
        "schemas": normalized_schemas,
        "new_branch_inventory": branches,
        "default_zero_and_inert_attestation": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
    }


def validate_collision_census_evidence(
    payload: Mapping[str, Any],
    *,
    raw_corpus: Mapping[str, Any],
    schema_freeze: Mapping[str, Any],
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate the completed full-corpus collision census gate.

    Only a no-actionable-collision result can support later nonzero training or
    BO work.  `blocked`, sample, partial, and collision-found reports are
    deliberately valid diagnostics but not valid completion evidence here.
    """

    if payload.get("schema") != R298_COLLISION_CENSUS_RECEIPT_SCHEMA:
        raise R298ValidationError("collision census receipt schema mismatch")
    if payload.get("status") != "passed_no_actionable_public_semantic_collision" or payload.get("pass") is not True:
        raise R298ValidationError("collision census has not passed complete no-actionable-collision gate")
    _validate_collision_provenance(payload, label="collision census receipt")
    if payload.get("exact_new_list_multiset_sha256") != EXACT_NEW_LIST_MULTISET_SHA256:
        raise R298ValidationError("collision census does not bind exact new-list multiset")
    if payload.get("raw_expert_corpus_manifest_sha256") != raw_corpus.get("manifest_sha256"):
        raise R298ValidationError("collision census binds a different raw corpus manifest")
    if payload.get("raw_expert_corpus_receipt_sha256") != raw_corpus.get("raw_corpus_receipt_sha256"):
        raise R298ValidationError("collision census binds a different raw corpus receipt")
    raw = _mapping(payload.get("raw_expert_corpus"), label="collision census raw corpus")
    if (
        raw.get("window_start_utc"),
        raw.get("window_end_utc"),
        raw.get("distinct_utc_day_count"),
    ) != (
        R298_RAW_EXPERT_CORPUS_START_UTC,
        R298_RAW_EXPERT_CORPUS_END_UTC,
        R298_RAW_EXPERT_CORPUS_DAYS,
    ):
        raise R298ValidationError("collision census does not bind exact 30-day window")
    source_disjoint = _mapping(raw.get("source_disjointness"), label="collision census source disjointness")
    for name, expected in {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
    }.items():
        _require_bool(source_disjoint.get(name), label=f"collision census source disjointness.{name}", expected=expected)
    pinned = _mapping(payload.get("pinned_simulator"), label="collision census pinned simulator")
    if pinned.get("libcg_sha256") != CANONICAL_LIBCG_SHA256:
        raise R298ValidationError("collision census does not bind canonical libcg")
    for name, expected in {
        "complete_evidence_required_for_pass": True,
        "hidden_branching_permitted": False,
    }.items():
        _require_bool(pinned.get(name), label=f"collision census pinned simulator.{name}", expected=expected)
    authority = _mapping(payload.get("runtime_authority"), label="collision census authority")
    for name, expected in {
        "elmo_only": True,
        "create_only": True,
        "training_eligible": False,
        "production_activation": False,
        "inzi_mutation": False,
        "learned_baseline_mutation": False,
    }.items():
        _require_bool(authority.get(name), label=f"collision census authority.{name}", expected=expected)
    execution_identity = _validate_elmo_execution_identity(
        payload.get("execution_identity"), label="collision census execution_identity"
    )
    resource_observation = _mapping(
        payload.get("resource_observation"), label="collision census resource observation"
    )
    resource_execution_identity = _validate_elmo_execution_identity(
        resource_observation.get("execution_identity"),
        label="collision census resource observation execution_identity",
    )
    if resource_execution_identity != execution_identity:
        raise R298ValidationError("collision census resource execution identity differs from run identity")
    raw_execution_identity = raw_corpus.get("execution_identity")
    if raw_execution_identity is not None and execution_identity != raw_execution_identity:
        raise R298ValidationError("collision census execution identity differs from raw-corpus execution identity")
    frozen_gate = _mapping(payload.get("frozen_schema_gate"), label="collision census frozen schema gate")
    if frozen_gate.get("frozen_schema_manifest_sha256") != schema_freeze.get("schema_manifest_sha256"):
        raise R298ValidationError("collision census does not bind schema-freeze manifest")
    if frozen_gate.get("zero_bypass_receipt_sha256") != schema_freeze.get("zero_bypass_receipt_sha256"):
        raise R298ValidationError("collision census does not bind zero-bypass receipt")
    _require_bool(
        frozen_gate.get("frozen_before_refeaturization"),
        label="collision census frozen schema gate ordering",
        expected=True,
    )
    re_featurization = _mapping(payload.get("re_featurization"), label="collision census re-featurization")
    _require_bool(
        re_featurization.get("full_30_day_complete"),
        label="collision census full 30-day re-featurization",
        expected=True,
    )
    if _require_exact_int(
        re_featurization.get("raw_episode_count"),
        label="collision census re-featurization raw episode count",
        minimum=1,
    ) != _require_exact_int(
        raw.get("completed_validated_episode_count"),
        label="collision census raw validated episode count",
        minimum=1,
    ):
        raise R298ValidationError("collision census did not re-featurize every validated raw episode")
    frame_coverage = _mapping(payload.get("frame_coverage"), label="collision census frame coverage")
    _require_bool(
        frame_coverage.get("all_actor_visible_and_forced_frames_included"),
        label="collision census frame coverage",
        expected=True,
    )
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="collision census receipt")
    return {
        "receipt_sha256": canonical_digest(payload),
        "receipt": None if identity is None else identity.to_dict(),
        "status": "passed_no_actionable_public_semantic_collision",
        "full_30_day_complete": True,
        "raw_episode_count": int(re_featurization["raw_episode_count"]),
        "factorized_stage_count": _require_exact_int(
            re_featurization.get("factorized_stage_count"),
            label="collision census factorized stage count",
            minimum=1,
        ),
        "execution_identity": execution_identity,
    }


def validate_frozen_tensor_identity_evidence(
    payload: Mapping[str, Any],
    *,
    evidence_identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Require actual per-tensor parent/candidate equality evidence.

    `frozen_backbone=true` is only an assertion.  This evidence maps every
    protected tensor path to its parent and derivative content identity and
    proves byte equality before a candidate can be benchmarked.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "frozen_tensor_identity_map",
        "all_frozen_tensor_bit_identity",
    }
    if set(payload) != expected:
        raise R298ValidationError("frozen tensor evidence key inventory changed")
    if payload.get("schema") != R298_FROZEN_TENSOR_IDENTITY_EVIDENCE_SCHEMA:
        raise R298ValidationError("frozen tensor evidence schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("frozen tensor evidence owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("frozen tensor evidence is not passed")
    if payload.get("host") != "elmo":
        raise R298ValidationError("frozen tensor evidence must be produced on Elmo")
    _validate_attachments(payload.get("source_attachments"))
    parent = _require_sha256(
        payload.get("parent_checkpoint_sha256"), label="frozen tensor parent checkpoint"
    )
    candidate = _require_sha256(
        payload.get("candidate_checkpoint_sha256"), label="frozen tensor candidate checkpoint"
    )
    _require_bool(
        payload.get("all_frozen_tensor_bit_identity"),
        label="all frozen tensor bit identity",
        expected=True,
    )
    raw_map = _mapping(
        payload.get("frozen_tensor_identity_map"), label="frozen tensor identity map"
    )
    if not raw_map:
        raise R298ValidationError("frozen tensor identity map must not be empty")
    normalized_map: dict[str, str] = {}
    for tensor_path, digest_pair in raw_map.items():
        path = _require_nonempty_string(tensor_path, label="frozen tensor path")
        pair = _mapping(digest_pair, label=f"frozen tensor identity {path}")
        if set(pair) != {"parent_sha256", "candidate_sha256"}:
            raise R298ValidationError(f"frozen tensor identity {path} fields changed")
        parent_digest = _require_sha256(
            pair.get("parent_sha256"), label=f"frozen tensor {path} parent digest"
        )
        candidate_digest = _require_sha256(
            pair.get("candidate_sha256"), label=f"frozen tensor {path} candidate digest"
        )
        if parent_digest != candidate_digest:
            raise R298ValidationError(f"frozen tensor {path} is not bit-identical")
        normalized_map[path] = parent_digest
    if evidence_identity is not None and verify_artifacts:
        _validate_identity_on_disk(evidence_identity, label="frozen tensor evidence")
    normalized: dict[str, Any] = {
        "schema": R298_FROZEN_TENSOR_IDENTITY_EVIDENCE_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed",
        "host": "elmo",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "parent_checkpoint_sha256": parent,
        "candidate_checkpoint_sha256": candidate,
        "frozen_tensor_identity_map": normalized_map,
        "all_frozen_tensor_bit_identity": True,
    }
    if evidence_identity is not None:
        normalized["evidence"] = evidence_identity.to_dict()
    return normalized


def validate_engine_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate actual pinned-simulator coverage, rejecting fixture-only claims."""

    if payload.get("schema") != R298_ENGINE_EVIDENCE_SCHEMA:
        raise R298ValidationError("engine evidence schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("engine evidence owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("engine evidence is not a passed execution")
    if payload.get("host") != "elmo":
        raise R298ValidationError("engine evidence must be produced on Elmo")
    _validate_attachments(payload.get("source_attachments"))
    execution = _mapping(payload.get("execution"), label="engine evidence execution")
    for name, expected in {
        "engine_backed": True,
        "actual_pinned_competition_simulator": True,
        "fixture_only": False,
        "legal_option_list_authoritative": True,
    }.items():
        _require_bool(execution.get(name), label=f"engine execution {name}", expected=expected)
    if execution.get("mode") != "pinned_competition_simulator":
        raise R298ValidationError("engine evidence execution mode is not pinned competition simulator")
    simulator = _mapping(payload.get("simulator"), label="engine evidence simulator")
    if _require_sha256(simulator.get("libcg_sha256"), label="simulator.libcg_sha256") != CANONICAL_LIBCG_SHA256:
        raise R298ValidationError("engine evidence does not bind canonical r236 libcg")
    if simulator.get("kaggle_environments_version") != "1.32.6":
        raise R298ValidationError("engine evidence does not bind Kaggle Environments 1.32.6")
    case_results = _mapping(payload.get("case_results"), label="engine evidence case_results")
    if set(case_results) != set(ENGINE_CASE_NAMES):
        raise R298ValidationError("engine evidence case inventory differs from r298 Phase 5")
    for name in ENGINE_CASE_NAMES:
        _require_bool(case_results.get(name), label=f"engine case {name}", expected=True)
    witnesses = _mapping(payload.get("case_witnesses"), label="engine evidence case_witnesses")
    if set(witnesses) != set(ENGINE_CASE_NAMES):
        raise R298ValidationError("engine evidence witness inventory differs from r298 Phase 5")
    normalized_witnesses: dict[str, str] = {}
    for name in ENGINE_CASE_NAMES:
        witness = witnesses[name]
        if not isinstance(witness, str) or not witness:
            raise R298ValidationError(f"engine case {name} has no immutable witness")
        normalized_witnesses[name] = witness
    summary = _mapping(payload.get("test_summary"), label="engine evidence test_summary")
    _require_exact_int(summary.get("failed"), label="engine evidence failed tests", minimum=0)
    if int(summary["failed"]) != 0:
        raise R298ValidationError("engine evidence has failed tests")
    if _require_exact_int(summary.get("passed"), label="engine evidence passed tests", minimum=1) < len(ENGINE_CASE_NAMES):
        raise R298ValidationError("engine evidence cannot cover every required Phase 5 case")
    return {
        "schema": R298_ENGINE_EVIDENCE_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed",
        "passed": True,
        "host": "elmo",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "execution": {
            "engine_backed": True,
            "actual_pinned_competition_simulator": True,
            "fixture_only": False,
            "legal_option_list_authoritative": True,
            "mode": "pinned_competition_simulator",
        },
        "simulator": {
            "libcg_sha256": CANONICAL_LIBCG_SHA256,
            "kaggle_environments_version": "1.32.6",
        },
        "case_results": {name: True for name in ENGINE_CASE_NAMES},
        "case_witnesses": normalized_witnesses,
        "test_summary": {
            "passed": int(summary["passed"]),
            "failed": 0,
        },
    }


def validate_phase_evidence_manifest(
    payload: Mapping[str, Any],
    *,
    verify_artifacts: bool,
    engine_evidence_identity: FileIdentity | None = None,
) -> dict[str, Any]:
    """Validate the seven evidence lanes needed for a non-placeholder receipt.

    Every phase points to an immutable artifact and names the assertions it
    proves.  The runner re-hashes each artifact.  Receipt readers can validate
    the same structure without requiring an Elmo-mounted artifact archive.
    """

    if payload.get("schema") != R298_PHASE_EVIDENCE_MANIFEST_SCHEMA:
        raise R298ValidationError("r298 phase evidence manifest schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("r298 phase evidence manifest owner revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise R298ValidationError("r298 phase evidence manifest is not passed")
    _validate_attachments(payload.get("source_attachments"))
    phases = _mapping(payload.get("phases"), label="r298 phase evidence manifest phases")
    if set(phases) != set(PHASE_NAMES):
        raise R298ValidationError("r298 phase evidence manifest has an incomplete phase inventory")
    normalized: dict[str, Any] = {}
    for name in PHASE_NAMES:
        row = _mapping(phases[name], label=f"r298 phase evidence {name}")
        if row.get("passed") is not True:
            raise R298ValidationError(f"r298 phase evidence {name} is not passed")
        identity = _identity_from_mapping(row.get("artifact"), label=f"r298 phase evidence {name}.artifact")
        if verify_artifacts:
            _validate_identity_on_disk(identity, label=f"r298 phase evidence {name}.artifact")
        assertions = _string_set(row.get("assertions"), label=f"r298 phase evidence {name}.assertions")
        missing = PHASE_REQUIRED_ASSERTIONS[name] - assertions
        if missing:
            raise R298ValidationError(
                f"r298 phase evidence {name} omits required assertions: {sorted(missing)}"
            )
        if name == "engine_metamorphic_tests" and engine_evidence_identity is not None:
            if identity != engine_evidence_identity:
                raise R298ValidationError(
                    "engine/metamorphic phase must bind the exact actual engine evidence artifact"
                )
        normalized[name] = {
            "passed": True,
            "artifact": identity.to_dict(),
            "assertions": sorted(assertions),
        }
    return {
        "schema": R298_PHASE_EVIDENCE_MANIFEST_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed",
        "source_attachments": _validate_attachments(payload["source_attachments"]),
        "phases": normalized,
    }


def _require_utc_timestamp(value: object, *, label: str) -> str:
    """Accept only an explicit, timezone-qualified ISO-8601 timestamp."""

    result = _require_nonempty_string(value, label=label, maximum=128)
    if not result.endswith("Z"):
        raise R298ValidationError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(result.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise R298ValidationError(f"{label} is not an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise R298ValidationError(f"{label} must include a UTC offset")
    return result


def _artifact_row(
    value: object,
    *,
    label: str,
    verify_artifacts: bool,
) -> tuple[Mapping[str, Any], FileIdentity]:
    """Read one `{payload, identity}` boundary without accepting aliases."""

    row = _mapping(value, label=label)
    if set(row) != {"payload", "identity"}:
        raise R298ValidationError(f"{label} must contain exactly payload and identity")
    payload = _mapping(row.get("payload"), label=f"{label}.payload")
    identity = _identity_from_mapping(row.get("identity"), label=f"{label}.identity")
    if verify_artifacts:
        _validate_identity_on_disk(identity, label=label)
    return payload, identity


def _validate_resource_evidence_bundle(
    value: object,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Bind every isolated r298 job to an independently measured envelope."""

    rows = _mapping(value, label="r298 resource evidence bundle")
    if set(rows) != set(RESOURCE_JOB_NAMES):
        raise R298ValidationError("r298 resource evidence must cover every isolated job")
    normalized_jobs: dict[str, Any] = {}
    max_ram = 0
    for job in RESOURCE_JOB_NAMES:
        payload, identity = _artifact_row(
            rows[job], label=f"r298 resource evidence {job}", verify_artifacts=verify_artifacts
        )
        normalized = validate_resource_receipt(payload, expected_phase=job)
        normalized_jobs[job] = {
            "receipt": identity.to_dict(),
            "receipt_sha256": canonical_digest(payload),
            "phase": job,
            "experiment_ram_peak_bytes": normalized["peaks"]["experiment_ram_peak_bytes"],
            "cpu": normalized["peaks"]["cpu"],
            "gpu": normalized["peaks"]["gpu"],
            "io": normalized["peaks"]["io"],
            "memory_heavy": normalized["resource_envelope"]["memory_heavy"],
            "concurrent_memory_heavy_phases": normalized["resource_envelope"][
                "concurrent_memory_heavy_phases"
            ],
            "measurement_method": normalized["resource_envelope"][
                "experiment_ram_measurement_method"
            ],
        }
        max_ram = max(max_ram, int(normalized["peaks"]["experiment_ram_peak_bytes"]))
    return {
        "schema": "poke_bot.alakazam_simulator_rules_r298_resource_evidence_bundle/v1",
        "host": "elmo",
        "experiment_ram_hard_ceiling_bytes": R298_EXPERIMENT_RAM_HARD_CEILING_BYTES,
        "memory_heavy_phase_concurrency": 1,
        "measured_peak_receipt": True,
        "managed_service_changed": False,
        "zfs_arc_l2arc_tuning_or_reconfiguration": False,
        "max_experiment_ram_peak_bytes": max_ram,
        "jobs": normalized_jobs,
    }


_TRANSFER_PER_SHARD_SCHEMA = (
    "poke_bot.alakazam_elmo_refeaturization_shard_transfer_receipt/v1"
)
_TRANSFER_PARITY_SCHEMA = (
    "poke_bot.alakazam_elmo_refeaturization_transfer_parity_receipt/v1"
)
_TRANSFER_DESTINATION_DIRECTORY = (
    "/home/pokebot/poke-bot-agent/outputs/quarantine/"
    "alakazam-elmo-rule-derivative/g4-refeaturized-census-shards"
)
_TRANSFER_PER_SHARD_FIELDS = frozenset(
    {
        "schema",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "validated_30_day_raw_manifest_sha256",
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
        "source_shard_logical_id",
        "content_addressed_filename",
        "source_sha256",
        "source_size_bytes",
        "source_schema_validation_passed",
        "destination_host",
        "destination_directory",
        "destination_path",
        "destination_sha256",
        "destination_size_bytes",
        "destination_validation_passed",
        "transfer_disposition_copied_resumed_or_skipped_exact",
        "parallel_lane_id",
        "started_at_utc",
        "validated_at_utc",
        "conflict_refused",
        "inzi_loading_or_execution_authority",
    }
)
_TRANSFER_PARITY_FIELDS = frozenset(
    {
        "schema",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "destination_directory",
        "source_manifest_sha256",
        "destination_manifest_sha256",
        "source_and_destination_object_count",
        "content_addressed_filename_set",
        "per_object_source_sha256_and_size",
        "per_object_destination_sha256_and_size",
        "schema_binding",
        "validated_30_day_raw_manifest_binding",
        "ordered_per_shard_receipt_sha256s",
        "skip_resume_and_conflict_counts",
        "parallel_lane_maximum_observed",
        "source_remote_parity_passed",
        "inzi_loading_or_execution_authority",
    }
)


def _validate_transfer_per_shard_receipt(
    payload: Mapping[str, Any],
    *,
    raw_manifest_sha256: str,
    schema_freeze: Mapping[str, Any],
    identity: FileIdentity,
    verify_artifacts: bool,
) -> dict[str, Any]:
    if set(payload) != _TRANSFER_PER_SHARD_FIELDS:
        raise R298ValidationError("transfer per-shard receipt key inventory changed")
    if payload.get("schema") != _TRANSFER_PER_SHARD_SCHEMA:
        raise R298ValidationError("transfer per-shard receipt schema mismatch")
    if payload.get("goal_contract_path") != "goals/alakazam-elmo-rule-derivative/contract.json":
        raise R298ValidationError("transfer per-shard receipt contract path mismatch")
    if payload.get("goal_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("transfer per-shard receipt contract digest mismatch")
    if payload.get("goal_revision") != 4:
        raise R298ValidationError("transfer per-shard receipt goal revision mismatch")
    if _require_sha256(
        payload.get("validated_30_day_raw_manifest_sha256"),
        label="transfer per-shard raw manifest",
    ) != raw_manifest_sha256:
        raise R298ValidationError("transfer per-shard receipt binds a different raw manifest")
    for receipt_field, frozen_field in (
        ("feature_schema_sha256", "feature_schema_sha256"),
        ("target_schema_sha256", "target_schema_sha256"),
        ("checklist_provenance_schema_sha256", "checklist_provenance_schema_sha256"),
    ):
        if _require_sha256(payload.get(receipt_field), label=f"transfer {receipt_field}") != schema_freeze[
            "schemas"
        ][frozen_field]:
            raise R298ValidationError(f"transfer per-shard receipt does not bind frozen {receipt_field}")
    logical_id = _require_identifier(
        payload.get("source_shard_logical_id"), label="transfer source shard logical id"
    )
    source_sha = _require_sha256(payload.get("source_sha256"), label="transfer source sha256")
    filename = _require_nonempty_string(
        payload.get("content_addressed_filename"), label="transfer content-addressed filename"
    )
    expected_filename = f"sha256-{source_sha.removeprefix('sha256:')}.refeaturization-census.shard"
    if filename != expected_filename:
        raise R298ValidationError("transfer filename is not the full lowercase content address")
    if Path(identity.path).name != f"sha256-{source_sha.removeprefix('sha256:')}.transfer-receipt.json":
        raise R298ValidationError("transfer receipt filename is not bound to its shard digest")
    source_size = _require_exact_int(
        payload.get("source_size_bytes"), label="transfer source size", minimum=1
    )
    if source_size > R298_MATERIALIZATION_SHARD_MAX_BYTES:
        raise R298ValidationError("transfer source shard exceeds strict 96-MiB cap")
    _require_bool(
        payload.get("source_schema_validation_passed"),
        label="transfer source schema validation",
        expected=True,
    )
    if payload.get("destination_host") != "inzi" or payload.get("destination_directory") != _TRANSFER_DESTINATION_DIRECTORY:
        raise R298ValidationError("transfer destination is not the quarantined Inzi directory")
    destination_path = _require_nonempty_string(
        payload.get("destination_path"), label="transfer destination path"
    )
    if destination_path != f"{_TRANSFER_DESTINATION_DIRECTORY}/{filename}":
        raise R298ValidationError("transfer destination path is not the content-addressed final path")
    if _require_sha256(payload.get("destination_sha256"), label="transfer destination sha256") != source_sha:
        raise R298ValidationError("transfer destination digest does not match source")
    if _require_exact_int(
        payload.get("destination_size_bytes"), label="transfer destination size", minimum=1
    ) != source_size:
        raise R298ValidationError("transfer destination size does not match source")
    _require_bool(
        payload.get("destination_validation_passed"),
        label="transfer destination validation",
        expected=True,
    )
    if payload.get("transfer_disposition_copied_resumed_or_skipped_exact") not in {
        "copied",
        "resumed",
        "skipped_exact",
    }:
        raise R298ValidationError("transfer disposition is not create-only/resumable")
    lane = _require_exact_int(payload.get("parallel_lane_id"), label="transfer lane", minimum=1)
    if lane > 4:
        raise R298ValidationError("transfer used more than four lanes")
    _require_utc_timestamp(payload.get("started_at_utc"), label="transfer started_at_utc")
    _require_utc_timestamp(payload.get("validated_at_utc"), label="transfer validated_at_utc")
    _require_bool(payload.get("conflict_refused"), label="transfer conflict_refused", expected=False)
    _require_bool(
        payload.get("inzi_loading_or_execution_authority"),
        label="transfer Inzi loading/execution authority",
        expected=False,
    )
    if verify_artifacts:
        _validate_identity_on_disk(identity, label=f"transfer receipt {logical_id}")
    return dict(payload)


def validate_transfer_staging_evidence(
    payload: Mapping[str, Any],
    *,
    raw_manifest_sha256: str,
    schema_freeze: Mapping[str, Any],
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate rev4's narrow quarantine-transfer exception end to end.

    No receipt may replace a shard or grant a process on Inzi access to staged
    bytes.  The only accepted disposition is a byte-exact, verified object in
    the one contract-named quarantine directory.
    """

    expected = {
        "per_shard_receipts",
        "per_shard_receipt_identities",
        "parity_receipt",
        "parity_receipt_identity",
    }
    if set(payload) != expected:
        raise R298ValidationError("transfer staging evidence key inventory changed")
    receipts = payload.get("per_shard_receipts")
    raw_identities = payload.get("per_shard_receipt_identities")
    if not isinstance(receipts, (list, tuple)) or not receipts:
        raise R298ValidationError("transfer staging requires at least one completed shard receipt")
    if not isinstance(raw_identities, (list, tuple)) or len(raw_identities) != len(receipts):
        raise R298ValidationError("transfer staging receipt identities do not cover every shard")
    normalized_receipts: list[dict[str, Any]] = []
    normalized_identities: list[dict[str, Any]] = []
    source_shas: set[str] = set()
    logical_ids: set[str] = set()
    filenames: set[str] = set()
    lanes: set[int] = set()
    for index, raw in enumerate(receipts):
        receipt = _mapping(raw, label=f"transfer shard receipt[{index}]")
        identity = _identity_from_mapping(
            raw_identities[index], label=f"transfer shard receipt identity[{index}]"
        )
        normalized = _validate_transfer_per_shard_receipt(
            receipt,
            raw_manifest_sha256=raw_manifest_sha256,
            schema_freeze=schema_freeze,
            identity=identity,
            verify_artifacts=verify_artifacts,
        )
        source_sha = str(normalized["source_sha256"])
        logical_id = str(normalized["source_shard_logical_id"])
        filename = str(normalized["content_addressed_filename"])
        if source_sha in source_shas or logical_id in logical_ids or filename in filenames:
            raise R298ValidationError("transfer staging contains duplicate source shard evidence")
        source_shas.add(source_sha)
        logical_ids.add(logical_id)
        filenames.add(filename)
        lanes.add(int(normalized["parallel_lane_id"]))
        normalized_receipts.append(normalized)
        normalized_identities.append(identity.to_dict())

    parity = _mapping(payload.get("parity_receipt"), label="transfer parity receipt")
    parity_identity = _identity_from_mapping(
        payload.get("parity_receipt_identity"), label="transfer parity receipt identity"
    )
    if set(parity) != _TRANSFER_PARITY_FIELDS:
        raise R298ValidationError("transfer parity receipt key inventory changed")
    if parity.get("schema") != _TRANSFER_PARITY_SCHEMA:
        raise R298ValidationError("transfer parity receipt schema mismatch")
    if parity.get("goal_contract_path") != "goals/alakazam-elmo-rule-derivative/contract.json" or parity.get(
        "goal_contract_sha256"
    ) != DEDICATED_GOAL_CONTRACT_SHA256 or parity.get("goal_revision") != 4:
        raise R298ValidationError("transfer parity receipt canonical contract binding changed")
    if Path(parity_identity.path).name != "transfer-parity-receipt.json":
        raise R298ValidationError("transfer parity receipt filename changed")
    if parity.get("destination_directory") != _TRANSFER_DESTINATION_DIRECTORY:
        raise R298ValidationError("transfer parity destination changed")
    _require_sha256(parity.get("source_manifest_sha256"), label="transfer parity source manifest")
    _require_sha256(parity.get("destination_manifest_sha256"), label="transfer parity destination manifest")
    if _require_exact_int(
        parity.get("source_and_destination_object_count"),
        label="transfer parity object count",
        minimum=1,
    ) != len(normalized_receipts):
        raise R298ValidationError("transfer parity object count differs from per-shard receipts")
    parity_filenames = _unique_strings(
        parity.get("content_addressed_filename_set"), label="transfer parity filenames"
    )
    if set(parity_filenames) != filenames:
        raise R298ValidationError("transfer parity filename set differs from per-shard receipts")

    def _object_map(value: object, *, label: str) -> dict[str, dict[str, Any]]:
        row = _mapping(value, label=label)
        result: dict[str, dict[str, Any]] = {}
        for filename, raw_row in row.items():
            name = _require_nonempty_string(filename, label=f"{label} filename")
            object_row = _mapping(raw_row, label=f"{label}.{name}")
            if set(object_row) != {"sha256", "size_bytes"}:
                raise R298ValidationError(f"{label}.{name} must contain sha256 and size_bytes")
            result[name] = {
                "sha256": _require_sha256(object_row.get("sha256"), label=f"{label}.{name}.sha256"),
                "size_bytes": _require_exact_int(
                    object_row.get("size_bytes"), label=f"{label}.{name}.size_bytes", minimum=1
                ),
            }
        return result

    source_objects = _object_map(
        parity.get("per_object_source_sha256_and_size"), label="transfer parity source objects"
    )
    destination_objects = _object_map(
        parity.get("per_object_destination_sha256_and_size"), label="transfer parity destination objects"
    )
    expected_objects = {
        str(row["content_addressed_filename"]): {
            "sha256": str(row["source_sha256"]),
            "size_bytes": int(row["source_size_bytes"]),
        }
        for row in normalized_receipts
    }
    if source_objects != expected_objects or destination_objects != expected_objects:
        raise R298ValidationError("transfer parity per-object identities differ from shard receipts")
    schema_binding = _mapping(parity.get("schema_binding"), label="transfer parity schema binding")
    if set(schema_binding) != {
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    }:
        raise R298ValidationError("transfer parity schema binding inventory changed")
    for name, expected_digest in schema_freeze["schemas"].items():
        if _require_sha256(schema_binding.get(name), label=f"transfer parity {name}") != expected_digest:
            raise R298ValidationError("transfer parity does not bind frozen schemas")
    if _require_sha256(
        parity.get("validated_30_day_raw_manifest_binding"),
        label="transfer parity raw manifest binding",
    ) != raw_manifest_sha256:
        raise R298ValidationError("transfer parity binds a different raw manifest")
    receipt_digests = _unique_strings(
        parity.get("ordered_per_shard_receipt_sha256s"), label="transfer parity receipt digests"
    )
    for digest in receipt_digests:
        _require_sha256(digest, label="transfer parity receipt digest")
    if receipt_digests != [identity["sha256"] for identity in normalized_identities]:
        raise R298ValidationError("transfer parity does not bind ordered physical shard receipts")
    counts = _mapping(parity.get("skip_resume_and_conflict_counts"), label="transfer parity disposition counts")
    if set(counts) != {"copied", "resumed", "skipped_exact", "conflict_refused"}:
        raise R298ValidationError("transfer parity disposition count inventory changed")
    normalized_counts = {
        name: _require_exact_int(counts.get(name), label=f"transfer parity {name}", minimum=0)
        for name in sorted(counts)
    }
    if normalized_counts["conflict_refused"] != 0:
        raise R298ValidationError("transfer parity cannot pass with a conflicting final object")
    actual_counts = {
        value: sum(
            1 for receipt in normalized_receipts
            if receipt["transfer_disposition_copied_resumed_or_skipped_exact"] == value
        )
        for value in ("copied", "resumed", "skipped_exact")
    }
    if any(normalized_counts[name] != actual_counts[name] for name in actual_counts):
        raise R298ValidationError("transfer parity disposition counts differ from shard receipts")
    max_lane = _require_exact_int(
        parity.get("parallel_lane_maximum_observed"), label="transfer parity max lane", minimum=1
    )
    if max_lane > 4 or max(lanes) != max_lane:
        raise R298ValidationError("transfer parity lane accounting differs from shard receipts")
    _require_bool(
        parity.get("source_remote_parity_passed"), label="transfer source/remote parity", expected=True
    )
    _require_bool(
        parity.get("inzi_loading_or_execution_authority"),
        label="transfer parity Inzi loading/execution authority",
        expected=False,
    )
    if verify_artifacts:
        _validate_identity_on_disk(parity_identity, label="transfer parity receipt")
    return {
        "per_shard_receipts": normalized_receipts,
        "per_shard_receipt_identities": normalized_identities,
        "parity_receipt": dict(parity),
        "parity_receipt_identity": parity_identity.to_dict(),
        "source_remote_parity_passed": True,
        "inzi_loading_or_execution_authority": False,
        "transfer_to_inzi_quarantine_only": True,
    }


def validate_checklist_provenance_evidence(
    payload: Mapping[str, Any],
    *,
    schema_freeze: Mapping[str, Any],
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate the currently available *inert* strict-filter stage record.

    This intentionally does **not** promote an invocation of the isolated
    diagnostic filter into candidate action-time wiring.  The r298 checklist
    module records ``runtime_wired:false`` by design.  A future separately
    authorized candidate materializer must supply a different receipt before
    BO can claim a sealed action-time wrapper.
    """

    if payload.get("schema") == R298_CANDIDATE_ACTION_TIME_CHECKLIST_RECEIPT_SCHEMA:
        return validate_candidate_action_time_checklist_evidence(
            payload,
            schema_freeze=schema_freeze,
            identity=identity,
            verify_artifacts=verify_artifacts,
        )
    try:
        from .alakazam_checklist_provenance_r298 import (
            CHANNEL_NAMES,
            R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA,
            R298_CHECKLIST_STAGE_RECEIPT_SCHEMA,
            load_checklist_provenance_config_r298,
        )
    except ImportError as exc:  # pragma: no cover - broken isolated install.
        raise R298ValidationError("strict r298 checklist provenance module is unavailable") from exc
    config = load_checklist_provenance_config_r298()
    if config.canonical_contract_sha256 != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("strict checklist config binds stale canonical contract")
    if payload.get("schema") != R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA:
        raise R298ValidationError("checklist evidence must be a strict-filter result")
    if payload.get("runtime_wired") is not False or payload.get("final_residual_gate") != 0.0:
        raise R298ValidationError("strict checklist evidence is not inert exact-zero output")
    if payload.get("phase_4_provenance_assertion") != "q1_q2_q8_pinned_provenance_or_zero_unavailable":
        raise R298ValidationError("checklist evidence lacks Phase-4 provenance assertion")
    channels = payload.get("channels")
    if not isinstance(channels, (list, tuple)) or [row.get("name") if isinstance(row, Mapping) else None for row in channels] != list(CHANNEL_NAMES):
        raise R298ValidationError("checklist evidence does not retain the exact eight-channel order")
    for row in channels:
        channel = _mapping(row, label="checklist result channel")
        if channel.get("applied_gate") != 0.0 or channel.get("final_residual") != 0.0:
            raise R298ValidationError("checklist result has a nonzero residual gate")
    stage = _mapping(payload.get("stage_receipt"), label="checklist stage receipt")
    if stage.get("schema") != R298_CHECKLIST_STAGE_RECEIPT_SCHEMA:
        raise R298ValidationError("checklist stage receipt schema mismatch")
    for name in (
        "sealed_r298_wrapper_invoked",
        "acting_player_public_information_only",
        "strict_provenance_filter_applied",
        "q1_q2_q8_missing_or_mismatch_unavailable_and_zero",
        "q5_q6_final_residual_exact_zero",
        "guide_support_final_residual_exact_zero",
        "all_final_residuals_exact_zero",
    ):
        _require_bool(stage.get(name), label=f"checklist stage {name}", expected=True)
    for name in ("legacy_r288_guide_residual_included", "runtime_wired", "production_or_inzi_authority"):
        _require_bool(stage.get(name), label=f"checklist stage {name}", expected=False)
    _require_sha256(stage.get("stage_receipt_sha256"), label="checklist stage receipt digest")
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="checklist strict-filter result")
    if schema_freeze["schemas"]["checklist_provenance_schema_sha256"] == "":  # defensive typed linkage
        raise R298ValidationError("frozen checklist schema digest is empty")
    return {
        "schema": R298_CHECKLIST_PROVENANCE_EVIDENCE_SCHEMA,
        "status": "inert_strict_filter_only",
        "evidence": None if identity is None else identity.to_dict(),
        "evidence_sha256": canonical_digest(payload),
        "strict_provenance_config": config.to_dict(),
        "candidate_action_time_wrapper_sealed": False,
        "acting_player_public_information_only": True,
        "strict_q1_q2_q8_provenance_enforced": True,
        "q5_q6_exact_zero": True,
        "legacy_r288_runtime_residual": 0.0,
        "candidate_checkpoint_sha256": None,
    }


def validate_candidate_action_time_checklist_evidence(
    payload: Mapping[str, Any],
    *,
    schema_freeze: Mapping[str, Any],
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate future benchmark-only action-time wrapper evidence.

    This schema exists specifically to avoid relabelling the current inert
    trace filter as policy wiring.  It is still not production authority: the
    receipt may prove the wrapper was invoked for the isolated Elmo candidate
    benchmark, and nothing more.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "host",
        "source_attachments",
        "goal_contract_sha256",
        "frozen_checklist_provenance_schema_sha256",
        "candidate_checkpoint_sha256",
        "strict_provenance_config",
        "stage_receipts",
        "candidate_action_time_wrapper_sealed",
        "acting_player_public_information_only",
        "strict_q1_q2_q8_provenance_enforced",
        "q5_q6_exact_zero",
        "legacy_r288_runtime_residual",
        "isolated_elmo_benchmark_only",
        "production_or_inzi_runtime_authority",
    }
    if set(payload) != expected:
        raise R298ValidationError("candidate action-time checklist receipt key inventory changed")
    if payload.get("schema") != R298_CANDIDATE_ACTION_TIME_CHECKLIST_RECEIPT_SCHEMA:
        raise R298ValidationError("candidate action-time checklist receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION or payload.get("status") != "passed" or payload.get("host") != "elmo":
        raise R298ValidationError("candidate action-time checklist receipt is not passed Elmo evidence")
    _validate_attachments(payload.get("source_attachments"))
    if payload.get("goal_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("candidate action-time checklist receipt binds stale contract")
    if _require_sha256(
        payload.get("frozen_checklist_provenance_schema_sha256"),
        label="candidate action-time checklist frozen schema",
    ) != schema_freeze["schemas"]["checklist_provenance_schema_sha256"]:
        raise R298ValidationError("candidate action-time checklist binds a different frozen schema")
    _require_sha256(
        payload.get("candidate_checkpoint_sha256"), label="candidate action-time checklist checkpoint"
    )
    config = _identity_from_mapping(
        payload.get("strict_provenance_config"), label="candidate action-time checklist config"
    )
    stage_receipts = payload.get("stage_receipts")
    if not isinstance(stage_receipts, (list, tuple)) or not stage_receipts:
        raise R298ValidationError("candidate action-time checklist receipt needs immutable stage receipts")
    normalized_stage_receipts: list[dict[str, Any]] = []
    seen_stage_digests: set[str] = set()
    for index, raw_identity in enumerate(stage_receipts):
        stage_identity = _identity_from_mapping(
            raw_identity, label=f"candidate action-time stage receipt[{index}]"
        )
        if stage_identity.sha256 in seen_stage_digests:
            raise R298ValidationError("candidate action-time checklist repeats a stage receipt")
        seen_stage_digests.add(stage_identity.sha256)
        if verify_artifacts:
            _, stage = read_json_object(stage_identity.path, label=f"candidate action-time stage receipt[{index}]")
            if file_identity(stage_identity.path, label=f"candidate action-time stage receipt[{index}]") != stage_identity:
                raise R298ValidationError("candidate action-time stage receipt identity drifted")
            if stage.get("schema") != "poke_bot.alakazam_checklist_stage_receipt_r298/v1":
                raise R298ValidationError("candidate action-time stage receipt schema mismatch")
            for field in (
                "sealed_r298_wrapper_invoked",
                "acting_player_public_information_only",
                "strict_provenance_filter_applied",
                "q1_q2_q8_missing_or_mismatch_unavailable_and_zero",
                "q5_q6_final_residual_exact_zero",
                "guide_support_final_residual_exact_zero",
                "all_final_residuals_exact_zero",
            ):
                _require_bool(stage.get(field), label=f"candidate action-time stage {field}", expected=True)
            for field in ("legacy_r288_guide_residual_included", "production_or_inzi_authority"):
                _require_bool(stage.get(field), label=f"candidate action-time stage {field}", expected=False)
        normalized_stage_receipts.append(stage_identity.to_dict())
    for name in (
        "candidate_action_time_wrapper_sealed",
        "acting_player_public_information_only",
        "strict_q1_q2_q8_provenance_enforced",
        "q5_q6_exact_zero",
        "isolated_elmo_benchmark_only",
    ):
        _require_bool(payload.get(name), label=f"candidate action-time checklist {name}", expected=True)
    if payload.get("legacy_r288_runtime_residual") != 0.0:
        raise R298ValidationError("candidate action-time checklist includes legacy r288 residual")
    _require_bool(
        payload.get("production_or_inzi_runtime_authority"),
        label="candidate action-time checklist production/Inzi authority",
        expected=False,
    )
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="candidate action-time checklist receipt")
    return {
        "schema": R298_CHECKLIST_PROVENANCE_EVIDENCE_SCHEMA,
        "status": "passed",
        "evidence": None if identity is None else identity.to_dict(),
        "evidence_sha256": canonical_digest(payload),
        "strict_provenance_config": config.to_dict(),
        "candidate_action_time_wrapper_sealed": True,
        "acting_player_public_information_only": True,
        "strict_q1_q2_q8_provenance_enforced": True,
        "q5_q6_exact_zero": True,
        "legacy_r288_runtime_residual": 0.0,
        "stage_receipts": normalized_stage_receipts,
        "candidate_checkpoint_sha256": _require_sha256(
            payload.get("candidate_checkpoint_sha256"),
            label="candidate action-time checklist checkpoint",
        ),
    }


R298_FROZEN_TENSOR_AUDIT_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_r298_frozen_tensor_audit/v1"
)
R298_CARD2VEC_NONREGRESSION_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_r298_card2vec_nonregression_receipt/v1"
)
R298_CANDIDATE_TRAINING_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_r298_candidate_receipt/v1"
)


def validate_frozen_tensor_audit(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate complete (not sampled) frozen-tensor byte equality evidence."""

    required = {
        "schema",
        "status",
        "goal_contract_sha256",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "parent_tensor_inventory_sha256",
        "candidate_tensor_inventory_sha256",
        "protected_tensor_count",
        "tensor_digest_pairs",
        "changed_tensors",
        "inventory_complete",
        "all_frozen_tensor_bit_identity",
    }
    if not required.issubset(payload):
        raise R298ValidationError("frozen tensor audit omits required complete-inventory evidence")
    if payload.get("schema") != R298_FROZEN_TENSOR_AUDIT_SCHEMA or payload.get("status") != "passed":
        raise R298ValidationError("frozen tensor audit schema/status mismatch")
    if payload.get("goal_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("frozen tensor audit binds stale contract")
    if _require_sha256(payload.get("parent_checkpoint_sha256"), label="frozen tensor parent checkpoint") != parent_checkpoint_sha256:
        raise R298ValidationError("frozen tensor audit binds wrong parent checkpoint")
    if _require_sha256(payload.get("candidate_checkpoint_sha256"), label="frozen tensor candidate checkpoint") != candidate_checkpoint_sha256:
        raise R298ValidationError("frozen tensor audit binds wrong candidate checkpoint")
    parent_inventory = _require_sha256(
        payload.get("parent_tensor_inventory_sha256"), label="frozen tensor parent inventory"
    )
    candidate_inventory = _require_sha256(
        payload.get("candidate_tensor_inventory_sha256"), label="frozen tensor candidate inventory"
    )
    _require_bool(payload.get("inventory_complete"), label="frozen tensor audit inventory_complete", expected=True)
    _require_bool(
        payload.get("all_frozen_tensor_bit_identity"),
        label="frozen tensor audit all_frozen_tensor_bit_identity",
        expected=True,
    )
    if payload.get("changed_tensors") != []:
        raise R298ValidationError("frozen tensor audit reports changed protected tensors")
    count = _require_exact_int(
        payload.get("protected_tensor_count"), label="frozen protected tensor count", minimum=1
    )
    pairs = payload.get("tensor_digest_pairs")
    if not isinstance(pairs, (list, tuple)) or len(pairs) != count:
        raise R298ValidationError("frozen tensor audit does not enumerate complete protected tensor inventory")
    names: set[str] = set()
    normalized_pairs: list[dict[str, str]] = []
    for index, raw_pair in enumerate(pairs):
        pair = _mapping(raw_pair, label=f"frozen tensor digest pair[{index}]")
        if set(pair) != {"name", "parent_sha256", "candidate_sha256"}:
            raise R298ValidationError("frozen tensor digest pair key inventory changed")
        name = _require_nonempty_string(pair.get("name"), label="frozen tensor name")
        parent = _require_sha256(pair.get("parent_sha256"), label="frozen tensor parent digest")
        candidate = _require_sha256(pair.get("candidate_sha256"), label="frozen tensor candidate digest")
        if name in names or parent != candidate:
            raise R298ValidationError("frozen tensor audit is incomplete or non-identical")
        names.add(name)
        normalized_pairs.append({"name": name, "sha256": parent})
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="frozen tensor audit")
    return {
        "schema": R298_FROZEN_TENSOR_AUDIT_SCHEMA,
        "status": "passed",
        "evidence": None if identity is None else identity.to_dict(),
        "evidence_sha256": canonical_digest(payload),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "parent_tensor_inventory_sha256": parent_inventory,
        "candidate_tensor_inventory_sha256": candidate_inventory,
        "protected_tensor_count": count,
        "tensor_digest_pairs": normalized_pairs,
        "all_frozen_tensor_bit_identity": True,
        "inventory_complete": True,
    }


def validate_card2vec_nonregression_evidence(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Bind all five rev4 Card2Vec non-regression guarantees to an artifact."""

    expected = {
        "schema",
        "status",
        "goal_contract_sha256",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "parameter_and_buffer_bit_identity",
        "input_schema_and_values_unchanged",
        "layer_off_runtime_output_and_behavior_parity",
        "structured_residual_relationship",
        "legacy_text_hash_exact_mechanics_or_target_provenance",
    }
    if set(payload) != expected:
        raise R298ValidationError("Card2Vec non-regression evidence key inventory changed")
    if payload.get("schema") != R298_CARD2VEC_NONREGRESSION_EVIDENCE_SCHEMA or payload.get("status") != "passed":
        raise R298ValidationError("Card2Vec non-regression evidence schema/status mismatch")
    if payload.get("goal_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256:
        raise R298ValidationError("Card2Vec evidence binds stale contract")
    if _require_sha256(payload.get("parent_checkpoint_sha256"), label="Card2Vec parent checkpoint") != parent_checkpoint_sha256:
        raise R298ValidationError("Card2Vec evidence binds wrong parent checkpoint")
    if _require_sha256(payload.get("candidate_checkpoint_sha256"), label="Card2Vec candidate checkpoint") != candidate_checkpoint_sha256:
        raise R298ValidationError("Card2Vec evidence binds wrong candidate checkpoint")
    for name in (
        "parameter_and_buffer_bit_identity",
        "input_schema_and_values_unchanged",
        "layer_off_runtime_output_and_behavior_parity",
    ):
        _require_bool(payload.get(name), label=f"Card2Vec {name}", expected=True)
    if payload.get("structured_residual_relationship") != "additive_alongside_or_after_card2vec":
        raise R298ValidationError("structured residual is not additive alongside/after Card2Vec")
    _require_bool(
        payload.get("legacy_text_hash_exact_mechanics_or_target_provenance"),
        label="Card2Vec legacy text hash exact authority",
        expected=False,
    )
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="Card2Vec non-regression evidence")
    return {
        "schema": R298_CARD2VEC_NONREGRESSION_EVIDENCE_SCHEMA,
        "status": "passed",
        "evidence": None if identity is None else identity.to_dict(),
        "evidence_sha256": canonical_digest(payload),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "parameter_and_buffer_bit_identity": True,
        "input_schema_and_values_unchanged": True,
        "layer_off_runtime_output_and_behavior_parity": True,
        "structured_residual_relationship": "additive_alongside_or_after_card2vec",
        "legacy_text_hash_exact_mechanics_or_target_provenance": False,
    }


def validate_candidate_training_evidence(
    payload: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    baseline_r274: Mapping[str, Any],
    schema_freeze: Mapping[str, Any],
    raw_corpus: Mapping[str, Any],
    refeaturization: Mapping[str, Any],
    collision_census: Mapping[str, Any],
    transfer_staging: Mapping[str, Any],
    frozen_tensor: Mapping[str, Any],
    card2vec: Mapping[str, Any],
    identity: FileIdentity | None,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Require the Phase-D receipt to bind every prior rev4 gate explicitly."""

    required = {
        "schema",
        "status",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_gateway_sha256",
        "candidate_id",
        "candidate_checkpoint_sha256",
        "candidate_checkpoint_size_bytes",
        "derivative_sidecar_sha256",
        "derivative_sidecar_size_bytes",
        "baseline_r274_candidate_id",
        "baseline_r274_checkpoint_sha256",
        "baseline_r274_checkpoint_size_bytes",
        "exact_new_list_canonical_multiset_sha256",
        "frozen_backbone_true",
        "all_frozen_tensor_bit_identity",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
        "schema_freeze_manifest_sha256",
        "raw_expert_corpus_manifest_sha256",
        "refeaturization_split_receipt_sha256",
        "collision_census_receipt_sha256",
        "transfer_parity_receipt_sha256",
        "card2vec_nonregression_receipt_sha256",
        "frozen_tensor_audit_receipt_sha256",
        "census_supported_trainable_branches",
        "resource_observation",
        "elmo_only_nonproduction_true",
    }
    if not required.issubset(payload):
        raise R298ValidationError("candidate training receipt omits required rev4 fields")
    if payload.get("schema") != R298_CANDIDATE_TRAINING_EVIDENCE_SCHEMA or payload.get("status") != "passed":
        raise R298ValidationError("candidate training receipt schema/status mismatch")
    if (
        payload.get("goal_contract_path") != "goals/alakazam-elmo-rule-derivative/contract.json"
        or payload.get("goal_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256
        or payload.get("goal_gateway_sha256") != DEDICATED_GOAL_GATEWAY_SHA256
    ):
        raise R298ValidationError("candidate training receipt canonical contract binding changed")
    bindings = {
        "candidate_id": candidate["candidate_id"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_checkpoint_size_bytes": candidate["checkpoint_size_bytes"],
        "baseline_r274_candidate_id": baseline_r274["candidate_id"],
        "baseline_r274_checkpoint_sha256": baseline_r274["checkpoint_sha256"],
        "baseline_r274_checkpoint_size_bytes": baseline_r274["checkpoint_size_bytes"],
        "exact_new_list_canonical_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "schema_freeze_manifest_sha256": schema_freeze["schema_manifest_sha256"],
        "raw_expert_corpus_manifest_sha256": raw_corpus["manifest_sha256"],
        "refeaturization_split_receipt_sha256": refeaturization["receipt_sha256"],
        "collision_census_receipt_sha256": collision_census["receipt_sha256"],
        "transfer_parity_receipt_sha256": transfer_staging["parity_receipt_identity"]["sha256"],
        "card2vec_nonregression_receipt_sha256": card2vec["evidence"]["sha256"],
        "frozen_tensor_audit_receipt_sha256": frozen_tensor["evidence"]["sha256"],
    }
    for name, expected_value in bindings.items():
        actual = payload.get(name)
        if isinstance(expected_value, str):
            if _require_sha256(actual, label=f"candidate training {name}") != expected_value if name.endswith("sha256") else actual != expected_value:
                raise R298ValidationError(f"candidate training receipt binds wrong {name}")
        elif actual != expected_value:
            raise R298ValidationError(f"candidate training receipt binds wrong {name}")
    _require_sha256(payload.get("derivative_sidecar_sha256"), label="candidate derivative sidecar")
    _require_exact_int(payload.get("derivative_sidecar_size_bytes"), label="candidate derivative sidecar size", minimum=1)
    for name in (
        "frozen_backbone_true",
        "all_frozen_tensor_bit_identity",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
        "elmo_only_nonproduction_true",
    ):
        _require_bool(payload.get(name), label=f"candidate training {name}", expected=True)
    branches = _unique_strings(
        payload.get("census_supported_trainable_branches"),
        label="candidate census-supported trainable branches",
    )
    frozen_branches = set(schema_freeze["new_branch_inventory"])
    if not set(branches).issubset(frozen_branches):
        raise R298ValidationError("candidate training names an unfrozen trainable branch")
    resource = _mapping(payload.get("resource_observation"), label="candidate training resource observation")
    if set(resource) != {"experiment_ram_peak_bytes", "memory_heavy_phase_concurrency", "host"}:
        raise R298ValidationError("candidate training resource observation fields changed")
    if resource.get("host") != "elmo" or _require_exact_int(
        resource.get("experiment_ram_peak_bytes"), label="candidate training RAM peak", minimum=0
    ) > R298_EXPERIMENT_RAM_HARD_CEILING_BYTES or resource.get("memory_heavy_phase_concurrency") != 1:
        raise R298ValidationError("candidate training violates Elmo resource envelope")
    if identity is not None and verify_artifacts:
        _validate_identity_on_disk(identity, label="candidate training receipt")
    return {
        "schema": R298_CANDIDATE_TRAINING_EVIDENCE_SCHEMA,
        "status": "passed",
        "receipt": None if identity is None else identity.to_dict(),
        "receipt_sha256": canonical_digest(payload),
        "candidate_id": candidate["candidate_id"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_checkpoint_size_bytes": candidate["checkpoint_size_bytes"],
        "baseline_r274_checkpoint_sha256": baseline_r274["checkpoint_sha256"],
        "schema_freeze_manifest_sha256": schema_freeze["schema_manifest_sha256"],
        "raw_expert_corpus_manifest_sha256": raw_corpus["manifest_sha256"],
        "refeaturization_split_receipt_sha256": refeaturization["receipt_sha256"],
        "collision_census_receipt_sha256": collision_census["receipt_sha256"],
        "transfer_parity_receipt_sha256": transfer_staging["parity_receipt_identity"]["sha256"],
        "card2vec_nonregression_receipt_sha256": card2vec["evidence"]["sha256"],
        "frozen_tensor_audit_receipt_sha256": frozen_tensor["evidence"]["sha256"],
        "census_supported_trainable_branches": branches,
        "elmo_only_nonproduction_true": True,
        "all_frozen_tensor_bit_identity": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
    }


def _validate_aggregate_evidence(
    value: object,
    *,
    candidate: Mapping[str, Any],
    baseline_r274: Mapping[str, Any],
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Validate all ordered artifact lanes before issuing an aggregate pass."""

    evidence = _mapping(value, label="r298 aggregate evidence")
    expected = {
        "schema_freeze",
        "raw_expert_corpus",
        "refeaturization_splits",
        "collision_census",
        "transfer_staging",
        "resource_receipts",
        "checklist_provenance",
        "frozen_tensor_identity",
        "card2vec_non_regression",
        "candidate_training",
        "engine_evidence",
        "phase_evidence",
    }
    if set(evidence) != expected:
        raise R298ValidationError("r298 aggregate evidence inventory changed")

    freeze_rows = _mapping(evidence["schema_freeze"], label="r298 schema-freeze evidence")
    if set(freeze_rows) != {"schema_manifest", "zero_bypass_receipt"}:
        raise R298ValidationError("schema-freeze evidence must bind manifest and zero-bypass receipt")
    freeze_manifest, freeze_identity = _artifact_row(
        freeze_rows["schema_manifest"],
        label="schema-freeze manifest artifact",
        verify_artifacts=verify_artifacts,
    )
    zero_bypass, zero_bypass_identity = _artifact_row(
        freeze_rows["zero_bypass_receipt"],
        label="zero-bypass artifact",
        verify_artifacts=verify_artifacts,
    )
    schema_freeze = validate_collision_schema_freeze_evidence(
        freeze_manifest,
        zero_bypass,
        schema_manifest_identity=freeze_identity,
        zero_bypass_identity=zero_bypass_identity,
        verify_artifacts=verify_artifacts,
    )

    raw_rows = _mapping(evidence["raw_expert_corpus"], label="r298 raw corpus evidence")
    if set(raw_rows) != {"manifest", "receipt"}:
        raise R298ValidationError("raw corpus evidence must bind manifest and receipt")
    raw_manifest, raw_manifest_identity = _artifact_row(
        raw_rows["manifest"], label="raw corpus manifest artifact", verify_artifacts=verify_artifacts
    )
    raw_receipt, raw_receipt_identity = _artifact_row(
        raw_rows["receipt"], label="raw corpus receipt artifact", verify_artifacts=verify_artifacts
    )
    raw_corpus = validate_collision_raw_corpus_evidence(
        raw_manifest,
        raw_receipt,
        manifest_identity=raw_manifest_identity,
        raw_receipt_identity=raw_receipt_identity,
        verify_artifacts=verify_artifacts,
        verify_archive_paths=verify_artifacts,
    )

    ref_payload, ref_identity = _artifact_row(
        evidence["refeaturization_splits"],
        label="re-featurization split receipt artifact",
        verify_artifacts=verify_artifacts,
    )
    refeaturization = validate_refeaturization_split_receipt(
        ref_payload,
        raw_manifest_sha256=str(raw_corpus["manifest_sha256"]),
        identity=ref_identity,
        verify_artifacts=verify_artifacts,
    )

    census_payload, census_identity = _artifact_row(
        evidence["collision_census"],
        label="collision census receipt artifact",
        verify_artifacts=verify_artifacts,
    )
    collision_census = validate_collision_census_evidence(
        census_payload,
        raw_corpus=raw_corpus,
        schema_freeze=schema_freeze,
        identity=census_identity,
        verify_artifacts=verify_artifacts,
    )

    transfer = validate_transfer_staging_evidence(
        _mapping(evidence["transfer_staging"], label="transfer staging evidence"),
        raw_manifest_sha256=str(raw_corpus["manifest_sha256"]),
        schema_freeze=schema_freeze,
        verify_artifacts=verify_artifacts,
    )
    resources = _validate_resource_evidence_bundle(
        evidence["resource_receipts"], verify_artifacts=verify_artifacts
    )

    checklist_payload, checklist_identity = _artifact_row(
        evidence["checklist_provenance"],
        label="checklist provenance evidence artifact",
        verify_artifacts=verify_artifacts,
    )
    checklist = validate_checklist_provenance_evidence(
        checklist_payload,
        schema_freeze=schema_freeze,
        identity=checklist_identity,
        verify_artifacts=verify_artifacts,
    )
    if checklist["candidate_action_time_wrapper_sealed"] is not True:
        raise R298ValidationError(
            "strict checklist filter is inert; no sealed candidate action-time wrapper evidence exists"
        )
    if checklist.get("candidate_checkpoint_sha256") != candidate["checkpoint_sha256"]:
        raise R298ValidationError(
            "sealed checklist wrapper does not bind the exact candidate checkpoint"
        )
    frozen_payload, frozen_identity = _artifact_row(
        evidence["frozen_tensor_identity"],
        label="frozen tensor audit artifact",
        verify_artifacts=verify_artifacts,
    )
    frozen = validate_frozen_tensor_audit(
        frozen_payload,
        parent_checkpoint_sha256=str(baseline_r274["checkpoint_sha256"]),
        candidate_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        identity=frozen_identity,
        verify_artifacts=verify_artifacts,
    )
    card2vec_payload, card2vec_identity = _artifact_row(
        evidence["card2vec_non_regression"],
        label="Card2Vec non-regression artifact",
        verify_artifacts=verify_artifacts,
    )
    card2vec = validate_card2vec_nonregression_evidence(
        card2vec_payload,
        parent_checkpoint_sha256=str(baseline_r274["checkpoint_sha256"]),
        candidate_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        identity=card2vec_identity,
        verify_artifacts=verify_artifacts,
    )
    candidate_payload, candidate_identity = _artifact_row(
        evidence["candidate_training"],
        label="candidate training receipt artifact",
        verify_artifacts=verify_artifacts,
    )
    candidate_training = validate_candidate_training_evidence(
        candidate_payload,
        candidate=candidate,
        baseline_r274=baseline_r274,
        schema_freeze=schema_freeze,
        raw_corpus=raw_corpus,
        refeaturization=refeaturization,
        collision_census=collision_census,
        transfer_staging=transfer,
        frozen_tensor=frozen,
        card2vec=card2vec,
        identity=candidate_identity,
        verify_artifacts=verify_artifacts,
    )

    engine_payload, engine_identity = _artifact_row(
        evidence["engine_evidence"], label="engine evidence artifact", verify_artifacts=verify_artifacts
    )
    engine = validate_engine_evidence(engine_payload)
    engine["evidence"] = engine_identity.to_dict()
    engine["evidence_sha256"] = canonical_digest(engine_payload)
    phase_payload, phase_identity = _artifact_row(
        evidence["phase_evidence"], label="phase evidence manifest artifact", verify_artifacts=verify_artifacts
    )
    phase = validate_phase_evidence_manifest(
        phase_payload,
        verify_artifacts=verify_artifacts,
        engine_evidence_identity=engine_identity,
    )
    phase["manifest"] = phase_identity.to_dict()
    phase["manifest_sha256"] = canonical_digest(phase_payload)
    if phase["phases"]["collision_census"]["artifact"] != collision_census["receipt"]:
        raise R298ValidationError("collision phase does not bind the exact census receipt artifact")

    raw_corpus["source_day_group_disjoint"] = bool(
        refeaturization["source_day_group_disjoint"]
    )
    return {
        "schema_freeze": schema_freeze,
        "raw_expert_corpus": raw_corpus,
        "refeaturization_splits": refeaturization,
        "collision_census": collision_census,
        "transfer_staging": transfer,
        "resource_receipt": resources,
        "checklist_provenance": checklist,
        "frozen_tensor_identity": frozen,
        "card2vec_non_regression": card2vec,
        "candidate_training": candidate_training,
        "engine_evidence": engine,
        "phase_evidence": phase,
    }


def _validate_baseline(value: object) -> dict[str, Any]:
    baseline = _mapping(value, label="r298 baseline")
    expected = {
        "r195_checkpoint_sha256",
        "r274_checkpoint_sha256",
        "exact_new_list_multiset_sha256",
        "r195_immutable",
        "r241_r274_baseline_preserved",
    }
    if set(baseline) != expected:
        raise R298ValidationError("r298 baseline fields are incomplete")
    if _require_sha256(baseline["r195_checkpoint_sha256"], label="baseline.r195_checkpoint_sha256") != R195_CHECKPOINT_SHA256:
        raise R298ValidationError("r298 baseline does not bind immutable r195")
    r274 = _require_sha256(baseline["r274_checkpoint_sha256"], label="baseline.r274_checkpoint_sha256")
    if baseline["exact_new_list_multiset_sha256"] != EXACT_NEW_LIST_MULTISET_SHA256:
        raise R298ValidationError("r298 baseline does not bind the exact new list")
    _require_bool(baseline["r195_immutable"], label="baseline.r195_immutable", expected=True)
    _require_bool(
        baseline["r241_r274_baseline_preserved"],
        label="baseline.r241_r274_baseline_preserved",
        expected=True,
    )
    return {
        "r195_checkpoint_sha256": R195_CHECKPOINT_SHA256,
        "r274_checkpoint_sha256": r274,
        "exact_new_list_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "r195_immutable": True,
        "r241_r274_baseline_preserved": True,
    }


def _require_identifier(value: object, *, label: str) -> str:
    """Require a display-safe immutable candidate label, never a path."""

    if not isinstance(value, str) or not value or len(value) > 160:
        raise R298ValidationError(f"{label} must be a nonempty identifier")
    if any(character.isspace() for character in value):
        raise R298ValidationError(f"{label} must not contain whitespace")
    return value


def _validate_baseline_r274(
    value: object,
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind BO comparisons to the exact r274 checkpoint rather than a CLI label."""

    row = _mapping(value, label="r298 baseline_r274")
    expected = {"candidate_id", "checkpoint_sha256", "checkpoint_size_bytes"}
    if set(row) != expected:
        raise R298ValidationError("r298 baseline_r274 fields are incomplete")
    checkpoint = _require_sha256(
        row["checkpoint_sha256"], label="baseline_r274.checkpoint_sha256"
    )
    if checkpoint != baseline["r274_checkpoint_sha256"]:
        raise R298ValidationError(
            "baseline_r274 checkpoint does not match the baseline r274 binding"
        )
    return {
        "candidate_id": _require_identifier(
            row["candidate_id"], label="baseline_r274.candidate_id"
        ),
        "checkpoint_sha256": checkpoint,
        "checkpoint_size_bytes": _require_exact_int(
            row["checkpoint_size_bytes"],
            label="baseline_r274.checkpoint_size_bytes",
            minimum=1,
        ),
    }


def _validate_candidate(value: object, *, baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _mapping(value, label="r298 candidate")
    expected = {
        "candidate_id",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "derivative_identity_sha256",
        "frozen_backbone",
        "zero_gated_layer_off_baseline_parity",
    }
    if set(candidate) != expected:
        raise R298ValidationError("r298 candidate fields are incomplete")
    checkpoint = _require_sha256(candidate["checkpoint_sha256"], label="candidate.checkpoint_sha256")
    # A parallel derivative can legitimately retain its frozen r274 checkpoint
    # byte identity while binding a new zero-gated sidecar.  The separate
    # derivative identity is therefore mandatory rather than guessed from
    # checkpoint inequality.
    _require_exact_int(candidate["checkpoint_size_bytes"], label="candidate.checkpoint_size_bytes", minimum=1)
    derivative = _require_sha256(candidate["derivative_identity_sha256"], label="candidate.derivative_identity_sha256")
    _require_bool(candidate["frozen_backbone"], label="candidate.frozen_backbone", expected=True)
    _require_bool(
        candidate["zero_gated_layer_off_baseline_parity"],
        label="candidate.zero_gated_layer_off_baseline_parity",
        expected=True,
    )
    return {
        "candidate_id": _require_identifier(
            candidate["candidate_id"], label="candidate.candidate_id"
        ),
        "checkpoint_sha256": checkpoint,
        "checkpoint_size_bytes": int(candidate["checkpoint_size_bytes"]),
        "derivative_identity_sha256": derivative,
        "frozen_backbone": True,
        "zero_gated_layer_off_baseline_parity": True,
        "baseline_r274_checkpoint_sha256": str(baseline["r274_checkpoint_sha256"]),
    }


def _validate_evaluation_disjointness(value: object) -> dict[str, Any]:
    """Require an explicit candidate/training and BO-seed separation claim.

    A benchmark root may deterministically derive multiple cohorts.  Every
    derived cohort seed identity must be listed here before a BO runner may
    consume the receipt.  The receipt cannot infer this from a mutable CLI
    seed or a friendly experiment name.
    """

    row = _mapping(value, label="r298 evaluation_disjointness")
    if set(row) != set(EVALUATION_DISJOINTNESS_NAMES):
        raise R298ValidationError("r298 evaluation disjointness inventory changed")
    _require_bool(
        row["candidate_training_source_disjoint"],
        label="evaluation_disjointness.candidate_training_source_disjoint",
        expected=True,
    )
    seeds = row["benchmark_seed_identities_excluded"]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise R298ValidationError(
            "evaluation_disjointness.benchmark_seed_identities_excluded must be nonempty"
        )
    normalized_seeds = [
        _require_sha256(
            seed,
            label="evaluation_disjointness.benchmark_seed_identities_excluded[]",
        )
        for seed in seeds
    ]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise R298ValidationError(
            "evaluation_disjointness.benchmark_seed_identities_excluded must be unique"
        )
    return {
        "candidate_training_source_disjoint": True,
        "benchmark_seed_identities_excluded": sorted(normalized_seeds),
    }


def _all_false_mapping(value: object, *, label: str, names: Sequence[str]) -> dict[str, bool]:
    row = _mapping(value, label=label)
    if set(row) != set(names):
        raise R298ValidationError(f"{label} inventory differs from the r298 contract")
    for name in names:
        _require_bool(row[name], label=f"{label}.{name}", expected=False)
    return {name: False for name in names}


def _validate_canonical_sources(
    value: object,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    """Bind the receipt to the rev-2 dedicated gateway, contract, and libcg.

    A generic file identity is not sufficient: an arbitrary JSON file can be
    content-addressed just as well as the sole typed goal contract.  Build
    paths re-hash the physical sources; portable receipt inspection still
    verifies the pinned identities below.
    """

    sources = _mapping(value, label="r298 canonical sources")
    if set(sources) != set(CANONICAL_SOURCE_NAMES):
        raise R298ValidationError("r298 canonical source inventory is incomplete")
    normalized: dict[str, Any] = {}
    for name in CANONICAL_SOURCE_NAMES:
        identity = _identity_from_mapping(
            sources[name], label=f"canonical source {name}"
        )
        expected_sha256 = CANONICAL_SOURCE_SHA256S[name]
        if identity.sha256 != expected_sha256:
            raise R298ValidationError(
                f"canonical source {name} does not bind the current dedicated r298 source"
            )
        if verify_artifacts:
            _validate_identity_on_disk(identity, label=f"canonical source {name}")
        normalized[name] = identity.to_dict()
    return normalized


def _portable_identity(value: object, *, label: str) -> dict[str, Any]:
    """Validate a receipt-carried identity without reopening a remote mount."""

    return _identity_from_mapping(value, label=label).to_dict()


def _validate_portable_schema_freeze(value: object) -> dict[str, Any]:
    row = _mapping(value, label="portable schema-freeze evidence")
    expected = {
        "schema_manifest_sha256",
        "schema_manifest",
        "zero_bypass_receipt_sha256",
        "zero_bypass_receipt",
        "schemas",
        "new_branch_inventory",
        "default_zero_and_inert_attestation",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    }
    if set(row) != expected:
        raise R298ValidationError("portable schema-freeze evidence inventory changed")
    schema_manifest_sha = _require_sha256(
        row.get("schema_manifest_sha256"), label="portable schema-freeze manifest"
    )
    zero_bypass_sha = _require_sha256(
        row.get("zero_bypass_receipt_sha256"), label="portable zero-bypass receipt"
    )
    schema_manifest = _portable_identity(row.get("schema_manifest"), label="portable schema manifest identity")
    zero_bypass = _portable_identity(row.get("zero_bypass_receipt"), label="portable zero-bypass identity")
    if schema_manifest["sha256"] != schema_manifest_sha or zero_bypass["sha256"] != zero_bypass_sha:
        raise R298ValidationError("portable schema-freeze digest does not match physical artifact identity")
    schemas = _mapping(row.get("schemas"), label="portable frozen schema digests")
    expected_schemas = {
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    }
    if set(schemas) != expected_schemas:
        raise R298ValidationError("portable frozen schema digest inventory changed")
    normalized_schemas = {
        name: _require_sha256(schemas[name], label=f"portable frozen {name}")
        for name in sorted(expected_schemas)
    }
    branches = _unique_strings(
        row.get("new_branch_inventory"), label="portable frozen branch inventory"
    )
    for name in (
        "default_zero_and_inert_attestation",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    ):
        _require_bool(row.get(name), label=f"portable schema freeze {name}", expected=True)
    return {
        "schema_manifest_sha256": schema_manifest_sha,
        "schema_manifest": schema_manifest,
        "zero_bypass_receipt_sha256": zero_bypass_sha,
        "zero_bypass_receipt": zero_bypass,
        "schemas": normalized_schemas,
        "new_branch_inventory": branches,
        "default_zero_and_inert_attestation": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
    }


def _validate_portable_raw_corpus(
    value: object,
    *,
    schema_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    del schema_freeze  # Raw receipt predates/does not mutate frozen schema, cross-link occurs in census.
    row = _mapping(value, label="portable raw corpus evidence")
    required = {
        "manifest_sha256",
        "manifest",
        "validated_deduplicated_manifest_sha256",
        "window_start_utc",
        "window_end_utc",
        "utc_partition_count",
        "distinct_utc_partitions",
        "contiguous_window",
        "archive_count",
        "archive_total_bytes",
        "completed_raw_zip_member_count",
        "completed_validated_episode_count",
        "unique_episode_identity_count",
        "episode_identity_inventory_sha256",
        "archive_identities",
        "raw_corpus_receipt_sha256",
        "raw_corpus_receipt",
        "raw_source_disjointness",
        "twenty_day_or_subset_fallback",
        "source_day_group_disjoint",
        "execution_identity",
    }
    if set(row) != required:
        raise R298ValidationError("portable raw corpus evidence inventory changed")
    manifest_sha = _require_sha256(row.get("manifest_sha256"), label="portable raw manifest")
    if _require_sha256(
        row.get("validated_deduplicated_manifest_sha256"), label="portable validated raw manifest"
    ) != manifest_sha:
        raise R298ValidationError("portable raw manifest has a mismatched validated/deduplicated digest")
    manifest = _portable_identity(row.get("manifest"), label="portable raw manifest identity")
    raw_receipt = _portable_identity(row.get("raw_corpus_receipt"), label="portable raw receipt identity")
    raw_receipt_sha = _require_sha256(
        row.get("raw_corpus_receipt_sha256"), label="portable raw corpus receipt"
    )
    if manifest["sha256"] != manifest_sha or raw_receipt["sha256"] != raw_receipt_sha:
        raise R298ValidationError("portable raw corpus artifact identities do not match bound digests")
    if (
        row.get("window_start_utc"),
        row.get("window_end_utc"),
        row.get("utc_partition_count"),
    ) != (
        R298_RAW_EXPERT_CORPUS_START_UTC,
        R298_RAW_EXPERT_CORPUS_END_UTC,
        R298_RAW_EXPERT_CORPUS_DAYS,
    ):
        raise R298ValidationError("portable raw corpus does not bind exact 30-day window")
    for name in ("distinct_utc_partitions", "contiguous_window", "source_day_group_disjoint"):
        _require_bool(row.get(name), label=f"portable raw corpus {name}", expected=True)
    _require_bool(
        row.get("twenty_day_or_subset_fallback"),
        label="portable raw corpus twenty-day fallback",
        expected=False,
    )
    if _require_exact_int(row.get("archive_count"), label="portable raw archive count", minimum=1) != 30:
        raise R298ValidationError("portable raw corpus archive count is not 30")
    for name in (
        "archive_total_bytes",
        "completed_raw_zip_member_count",
        "completed_validated_episode_count",
        "unique_episode_identity_count",
    ):
        _require_exact_int(row.get(name), label=f"portable raw corpus {name}", minimum=1)
    _require_sha256(row.get("episode_identity_inventory_sha256"), label="portable raw episode inventory")
    archives = row.get("archive_identities")
    if not isinstance(archives, (list, tuple)) or len(archives) != 30:
        raise R298ValidationError("portable raw archive identity inventory is incomplete")
    dates = []
    digests: set[str] = set()
    for index, archive in enumerate(archives):
        item = _mapping(archive, label=f"portable raw archive[{index}]")
        expected_archive = {
            "date",
            "dataset_slug",
            "path",
            "sha256",
            "bytes",
            "zip_json_member_count",
            "validated_episode_count",
            "index_episode_count",
            "source_discrepancy",
        }
        if set(item) != expected_archive:
            raise R298ValidationError("portable raw archive inventory fields changed")
        dates.append(_require_nonempty_string(item.get("date"), label="portable raw archive date"))
        digest = _require_sha256(item.get("sha256"), label="portable raw archive sha256")
        if digest in digests:
            raise R298ValidationError("portable raw archive digest duplicated")
        digests.add(digest)
        _require_exact_int(item.get("bytes"), label="portable raw archive bytes", minimum=1)
    if tuple(dates) != _required_raw_utc_dates():
        raise R298ValidationError("portable raw archive dates are not exact contiguous UTC window")
    disjoint = _mapping(row.get("raw_source_disjointness"), label="portable raw source disjointness")
    expected_disjoint = {
        "archive_date_source_sha256_unique": True,
        "episode_identity_unique": True,
        "episode_id_content_unique": True,
        "source_window_blending_permitted": False,
        "training_eligible": False,
    }
    if dict(disjoint) != expected_disjoint:
        raise R298ValidationError("portable raw source disjointness changed")
    _validate_elmo_execution_identity(
        row.get("execution_identity"), label="portable raw corpus execution identity"
    )
    return dict(row)


def _validate_portable_refeaturization(
    value: object,
    *,
    raw_corpus: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, label="portable refeaturization split receipt")
    expected = {
        "schema",
        "raw_expert_corpus_manifest_sha256",
        "source_day_group_disjoint",
        "splits",
        "receipt",
        "receipt_sha256",
    }
    if set(row) != expected or row.get("schema") != R298_REFEATURIZATION_SPLIT_RECEIPT_SCHEMA:
        raise R298ValidationError("portable refeaturization receipt schema/inventory changed")
    if _require_sha256(
        row.get("raw_expert_corpus_manifest_sha256"), label="portable refeaturization raw manifest"
    ) != raw_corpus["manifest_sha256"]:
        raise R298ValidationError("portable refeaturization binds a different raw manifest")
    _require_bool(row.get("source_day_group_disjoint"), label="portable re-featurization disjointness", expected=True)
    receipt = _portable_identity(row.get("receipt"), label="portable refeaturization receipt identity")
    if _require_sha256(row.get("receipt_sha256"), label="portable refeaturization receipt") != receipt["sha256"]:
        raise R298ValidationError("portable refeaturization receipt digest does not match artifact identity")
    splits = _mapping(row.get("splits"), label="portable re-featurization splits")
    if set(splits) != {"train", "validation", "evaluation"}:
        raise R298ValidationError("portable refeaturization split inventory changed")
    grouped: dict[str, list[set[str]]] = {"source_ids": [], "utc_dates": [], "group_ids": []}
    for split_name in ("train", "validation", "evaluation"):
        split = _mapping(splits[split_name], label=f"portable split {split_name}")
        if set(split) != {"source_ids", "utc_dates", "group_ids", "row_count"}:
            raise R298ValidationError("portable refeaturization split fields changed")
        for field in grouped:
            values = _unique_strings(split.get(field), label=f"portable split {split_name}.{field}")
            if field == "utc_dates" and not set(values).issubset(set(_required_raw_utc_dates())):
                raise R298ValidationError("portable split contains a non-window UTC date")
            grouped[field].append(set(values))
        _require_exact_int(split.get("row_count"), label=f"portable split {split_name}.row_count", minimum=1)
    for field, sets in grouped.items():
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise R298ValidationError(f"portable split {field} overlaps")
    if set().union(*grouped["utc_dates"]) != set(_required_raw_utc_dates()):
        raise R298ValidationError("portable splits do not cover exact raw window")
    return dict(row)


def _validate_portable_collision_census(
    value: object,
    *,
    raw_corpus: Mapping[str, Any],
    schema_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, label="portable collision census")
    expected = {
        "receipt_sha256",
        "receipt",
        "status",
        "full_30_day_complete",
        "raw_episode_count",
        "factorized_stage_count",
        "execution_identity",
    }
    if set(row) != expected:
        raise R298ValidationError("portable collision census inventory changed")
    receipt = _portable_identity(row.get("receipt"), label="portable collision census receipt identity")
    if _require_sha256(row.get("receipt_sha256"), label="portable collision census receipt") != receipt["sha256"]:
        raise R298ValidationError("portable collision census digest does not match artifact identity")
    if row.get("status") != "passed_no_actionable_public_semantic_collision":
        raise R298ValidationError("portable collision census did not pass no-collision gate")
    _require_bool(row.get("full_30_day_complete"), label="portable census full 30-day", expected=True)
    if _require_exact_int(row.get("raw_episode_count"), label="portable census raw episode count", minimum=1) != raw_corpus[
        "completed_validated_episode_count"
    ]:
        raise R298ValidationError("portable collision census raw episode count differs from raw receipt")
    _require_exact_int(row.get("factorized_stage_count"), label="portable census factorized stage count", minimum=1)
    _validate_elmo_execution_identity(
        row.get("execution_identity"), label="portable collision census execution identity"
    )
    # The field has no direct appearance in the compact census summary, but a
    # caller cannot omit schema freeze from the aggregate because its typed
    # receipt is independently present and bound by the candidate receipt.
    _require_sha256(schema_freeze["schema_manifest_sha256"], label="portable frozen schema manifest")
    return dict(row)


def _validate_portable_transfer_staging(
    value: object,
    *,
    raw_corpus: Mapping[str, Any],
    schema_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    # This compact result intentionally retains the actual per-shard/parity
    # receipt payloads, allowing portable consumers to re-check every
    # source/destination binding rather than trusting a Boolean summary.
    return validate_transfer_staging_evidence(
        _mapping(value, label="portable transfer staging"),
        raw_manifest_sha256=str(raw_corpus["manifest_sha256"]),
        schema_freeze=schema_freeze,
        verify_artifacts=False,
    )


def _validate_portable_resource_bundle(value: object) -> dict[str, Any]:
    row = _mapping(value, label="portable resource evidence bundle")
    expected = {
        "schema",
        "host",
        "experiment_ram_hard_ceiling_bytes",
        "memory_heavy_phase_concurrency",
        "measured_peak_receipt",
        "managed_service_changed",
        "zfs_arc_l2arc_tuning_or_reconfiguration",
        "max_experiment_ram_peak_bytes",
        "jobs",
    }
    if set(row) != expected or row.get("schema") != "poke_bot.alakazam_simulator_rules_r298_resource_evidence_bundle/v1":
        raise R298ValidationError("portable resource evidence bundle schema/inventory changed")
    if row.get("host") != "elmo":
        raise R298ValidationError("portable resource evidence is not Elmo-only")
    if _require_exact_int(
        row.get("experiment_ram_hard_ceiling_bytes"), label="portable RAM ceiling", minimum=1
    ) != R298_EXPERIMENT_RAM_HARD_CEILING_BYTES:
        raise R298ValidationError("portable resource RAM ceiling changed")
    if row.get("memory_heavy_phase_concurrency") != 1 or row.get("measured_peak_receipt") is not True:
        raise R298ValidationError("portable resource concurrency/measurement proof is missing")
    for name in ("managed_service_changed", "zfs_arc_l2arc_tuning_or_reconfiguration"):
        _require_bool(row.get(name), label=f"portable resource {name}", expected=False)
    if _require_exact_int(row.get("max_experiment_ram_peak_bytes"), label="portable max RAM peak", minimum=0) > R298_EXPERIMENT_RAM_HARD_CEILING_BYTES:
        raise R298ValidationError("portable resource evidence exceeded 96-GiB cap")
    jobs = _mapping(row.get("jobs"), label="portable resource jobs")
    if set(jobs) != set(RESOURCE_JOB_NAMES):
        raise R298ValidationError("portable resource job inventory changed")
    for job in RESOURCE_JOB_NAMES:
        item = _mapping(jobs[job], label=f"portable resource job {job}")
        expected_item = {
            "receipt",
            "receipt_sha256",
            "phase",
            "experiment_ram_peak_bytes",
            "cpu",
            "gpu",
            "io",
            "memory_heavy",
            "concurrent_memory_heavy_phases",
            "measurement_method",
        }
        if set(item) != expected_item or item.get("phase") != job:
            raise R298ValidationError("portable resource job inventory/phase changed")
        identity = _portable_identity(item.get("receipt"), label=f"portable resource job {job} receipt")
        if _require_sha256(item.get("receipt_sha256"), label=f"portable resource job {job} digest") != identity["sha256"]:
            raise R298ValidationError("portable resource job receipt digest mismatch")
        if _require_exact_int(item.get("experiment_ram_peak_bytes"), label=f"portable resource job {job} RAM", minimum=0) > R298_EXPERIMENT_RAM_HARD_CEILING_BYTES:
            raise R298ValidationError("portable resource job exceeded RAM cap")
        if _require_exact_int(item.get("concurrent_memory_heavy_phases"), label=f"portable resource job {job} concurrency", minimum=0) > 1:
            raise R298ValidationError("portable resource job exceeds memory-heavy concurrency")
        _require_nonempty_string(item.get("measurement_method"), label=f"portable resource job {job} measurement")
    return dict(row)


def _validate_portable_checklist(
    value: object,
    *,
    candidate_checkpoint_sha256: str,
) -> dict[str, Any]:
    row = _mapping(value, label="portable checklist provenance")
    expected = {
        "schema",
        "status",
        "evidence",
        "evidence_sha256",
        "strict_provenance_config",
        "candidate_action_time_wrapper_sealed",
        "acting_player_public_information_only",
        "strict_q1_q2_q8_provenance_enforced",
        "q5_q6_exact_zero",
        "legacy_r288_runtime_residual",
        "candidate_checkpoint_sha256",
        "stage_receipts",
    }
    if set(row) != expected or row.get("schema") != R298_CHECKLIST_PROVENANCE_EVIDENCE_SCHEMA:
        raise R298ValidationError("portable checklist provenance schema/inventory changed")
    if row.get("status") != "passed":
        raise R298ValidationError("portable checklist provenance is not a passed action-time artifact")
    evidence = _portable_identity(row.get("evidence"), label="portable checklist evidence identity")
    if _require_sha256(row.get("evidence_sha256"), label="portable checklist evidence digest") != evidence["sha256"]:
        raise R298ValidationError("portable checklist evidence digest mismatch")
    config = _mapping(row.get("strict_provenance_config"), label="portable checklist config")
    if config.get("canonical_contract_sha256") != DEDICATED_GOAL_CONTRACT_SHA256 or config.get("runtime_wired") is not False:
        raise R298ValidationError("portable checklist config does not bind inert strict r298 authority")
    for name in (
        "candidate_action_time_wrapper_sealed",
        "acting_player_public_information_only",
        "strict_q1_q2_q8_provenance_enforced",
        "q5_q6_exact_zero",
    ):
        _require_bool(row.get(name), label=f"portable checklist {name}", expected=True)
    if row.get("legacy_r288_runtime_residual") != 0.0:
        raise R298ValidationError("portable checklist includes a legacy r288 residual")
    if _require_sha256(
        row.get("candidate_checkpoint_sha256"), label="portable checklist candidate checkpoint"
    ) != candidate_checkpoint_sha256:
        raise R298ValidationError("portable checklist does not bind the exact candidate checkpoint")
    stage_receipts = row.get("stage_receipts")
    if not isinstance(stage_receipts, (list, tuple)) or not stage_receipts:
        raise R298ValidationError("portable checklist omits action-time stage receipt identities")
    seen_stage_receipts: set[str] = set()
    for index, stage_receipt in enumerate(stage_receipts):
        identity = _portable_identity(
            stage_receipt, label=f"portable checklist stage receipt[{index}]"
        )
        if identity["sha256"] in seen_stage_receipts:
            raise R298ValidationError("portable checklist repeats an action-time stage receipt")
        seen_stage_receipts.add(identity["sha256"])
    return dict(row)


def _validate_portable_frozen_tensor(
    value: object,
    *,
    parent_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
) -> dict[str, Any]:
    row = _mapping(value, label="portable frozen tensor audit")
    expected = {
        "schema",
        "status",
        "evidence",
        "evidence_sha256",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "parent_tensor_inventory_sha256",
        "candidate_tensor_inventory_sha256",
        "protected_tensor_count",
        "tensor_digest_pairs",
        "all_frozen_tensor_bit_identity",
        "inventory_complete",
    }
    if set(row) != expected or row.get("schema") != R298_FROZEN_TENSOR_AUDIT_SCHEMA or row.get("status") != "passed":
        raise R298ValidationError("portable frozen tensor audit schema/inventory changed")
    evidence = _portable_identity(row.get("evidence"), label="portable frozen tensor evidence")
    if _require_sha256(row.get("evidence_sha256"), label="portable frozen tensor digest") != evidence["sha256"]:
        raise R298ValidationError("portable frozen tensor evidence identity mismatch")
    if row.get("parent_checkpoint_sha256") != parent_checkpoint_sha256 or row.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256:
        raise R298ValidationError("portable frozen tensor checkpoint binding changed")
    for name in ("parent_tensor_inventory_sha256", "candidate_tensor_inventory_sha256"):
        _require_sha256(row.get(name), label=f"portable frozen {name}")
    _require_bool(row.get("all_frozen_tensor_bit_identity"), label="portable frozen identity", expected=True)
    _require_bool(row.get("inventory_complete"), label="portable frozen inventory complete", expected=True)
    count = _require_exact_int(row.get("protected_tensor_count"), label="portable frozen tensor count", minimum=1)
    pairs = row.get("tensor_digest_pairs")
    if not isinstance(pairs, (list, tuple)) or len(pairs) != count:
        raise R298ValidationError("portable frozen tensor map is incomplete")
    names: set[str] = set()
    for item in pairs:
        pair = _mapping(item, label="portable frozen tensor pair")
        if set(pair) != {"name", "sha256"}:
            raise R298ValidationError("portable frozen tensor pair inventory changed")
        name = _require_nonempty_string(pair.get("name"), label="portable frozen tensor name")
        if name in names:
            raise R298ValidationError("portable frozen tensor pair repeats a name")
        names.add(name)
        _require_sha256(pair.get("sha256"), label="portable frozen tensor digest")
    return dict(row)


def _validate_portable_card2vec(
    value: object,
    *,
    parent_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
) -> dict[str, Any]:
    row = _mapping(value, label="portable Card2Vec non-regression evidence")
    expected = {
        "schema",
        "status",
        "evidence",
        "evidence_sha256",
        "parent_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "parameter_and_buffer_bit_identity",
        "input_schema_and_values_unchanged",
        "layer_off_runtime_output_and_behavior_parity",
        "structured_residual_relationship",
        "legacy_text_hash_exact_mechanics_or_target_provenance",
    }
    if set(row) != expected or row.get("schema") != R298_CARD2VEC_NONREGRESSION_EVIDENCE_SCHEMA or row.get("status") != "passed":
        raise R298ValidationError("portable Card2Vec evidence schema/inventory changed")
    evidence = _portable_identity(row.get("evidence"), label="portable Card2Vec evidence identity")
    if _require_sha256(row.get("evidence_sha256"), label="portable Card2Vec evidence digest") != evidence["sha256"]:
        raise R298ValidationError("portable Card2Vec evidence identity mismatch")
    if row.get("parent_checkpoint_sha256") != parent_checkpoint_sha256 or row.get("candidate_checkpoint_sha256") != candidate_checkpoint_sha256:
        raise R298ValidationError("portable Card2Vec checkpoint binding changed")
    for name in (
        "parameter_and_buffer_bit_identity",
        "input_schema_and_values_unchanged",
        "layer_off_runtime_output_and_behavior_parity",
    ):
        _require_bool(row.get(name), label=f"portable Card2Vec {name}", expected=True)
    if row.get("structured_residual_relationship") != "additive_alongside_or_after_card2vec":
        raise R298ValidationError("portable Card2Vec residual is not additive")
    _require_bool(
        row.get("legacy_text_hash_exact_mechanics_or_target_provenance"),
        label="portable Card2Vec legacy text authority",
        expected=False,
    )
    return dict(row)


def _validate_portable_candidate_training(
    value: object,
    *,
    candidate: Mapping[str, Any],
    baseline_r274: Mapping[str, Any],
    schema_freeze: Mapping[str, Any],
    raw_corpus: Mapping[str, Any],
    refeaturization: Mapping[str, Any],
    collision_census: Mapping[str, Any],
    transfer_staging: Mapping[str, Any],
    frozen_tensor: Mapping[str, Any],
    card2vec: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, label="portable candidate training receipt")
    expected = {
        "schema",
        "status",
        "receipt",
        "receipt_sha256",
        "candidate_id",
        "candidate_checkpoint_sha256",
        "candidate_checkpoint_size_bytes",
        "baseline_r274_checkpoint_sha256",
        "schema_freeze_manifest_sha256",
        "raw_expert_corpus_manifest_sha256",
        "refeaturization_split_receipt_sha256",
        "collision_census_receipt_sha256",
        "transfer_parity_receipt_sha256",
        "card2vec_nonregression_receipt_sha256",
        "frozen_tensor_audit_receipt_sha256",
        "census_supported_trainable_branches",
        "elmo_only_nonproduction_true",
        "all_frozen_tensor_bit_identity",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    }
    if set(row) != expected or row.get("schema") != R298_CANDIDATE_TRAINING_EVIDENCE_SCHEMA or row.get("status") != "passed":
        raise R298ValidationError("portable candidate training schema/inventory changed")
    receipt = _portable_identity(row.get("receipt"), label="portable candidate training receipt identity")
    if _require_sha256(row.get("receipt_sha256"), label="portable candidate training digest") != receipt["sha256"]:
        raise R298ValidationError("portable candidate training receipt identity mismatch")
    expected_bindings = {
        "candidate_id": candidate["candidate_id"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "candidate_checkpoint_size_bytes": candidate["checkpoint_size_bytes"],
        "baseline_r274_checkpoint_sha256": baseline_r274["checkpoint_sha256"],
        "schema_freeze_manifest_sha256": schema_freeze["schema_manifest_sha256"],
        "raw_expert_corpus_manifest_sha256": raw_corpus["manifest_sha256"],
        "refeaturization_split_receipt_sha256": refeaturization["receipt_sha256"],
        "collision_census_receipt_sha256": collision_census["receipt_sha256"],
        "transfer_parity_receipt_sha256": transfer_staging["parity_receipt_identity"]["sha256"],
        "card2vec_nonregression_receipt_sha256": card2vec["evidence"]["sha256"],
        "frozen_tensor_audit_receipt_sha256": frozen_tensor["evidence"]["sha256"],
    }
    for name, expected_value in expected_bindings.items():
        if row.get(name) != expected_value:
            raise R298ValidationError(f"portable candidate training binds wrong {name}")
    for name in (
        "elmo_only_nonproduction_true",
        "all_frozen_tensor_bit_identity",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    ):
        _require_bool(row.get(name), label=f"portable candidate {name}", expected=True)
    branches = _unique_strings(
        row.get("census_supported_trainable_branches"), label="portable candidate supported branches"
    )
    if not set(branches).issubset(set(schema_freeze["new_branch_inventory"])):
        raise R298ValidationError("portable candidate trains an unfrozen branch")
    return dict(row)


def _validate_portable_engine(value: object) -> dict[str, Any]:
    row = _mapping(value, label="portable engine evidence")
    expected = {
        "schema",
        "owner_revision",
        "status",
        "passed",
        "host",
        "source_attachments",
        "execution",
        "simulator",
        "case_results",
        "case_witnesses",
        "test_summary",
        "evidence",
        "evidence_sha256",
    }
    if set(row) != expected:
        raise R298ValidationError("portable engine evidence inventory changed")
    evidence = _portable_identity(row.get("evidence"), label="portable engine evidence identity")
    if _require_sha256(row.get("evidence_sha256"), label="portable engine evidence digest") != evidence["sha256"]:
        raise R298ValidationError("portable engine evidence identity mismatch")
    normalized = validate_engine_evidence({key: row[key] for key in expected - {"evidence", "evidence_sha256"}})
    return {**normalized, "evidence": evidence, "evidence_sha256": row["evidence_sha256"]}


def _validate_portable_phase_evidence(
    value: object,
    *,
    engine: Mapping[str, Any],
    collision_census: Mapping[str, Any],
) -> dict[str, Any]:
    row = _mapping(value, label="portable phase evidence")
    expected = {
        "schema",
        "owner_revision",
        "status",
        "source_attachments",
        "phases",
        "manifest",
        "manifest_sha256",
    }
    if set(row) != expected:
        raise R298ValidationError("portable phase evidence inventory changed")
    manifest = _portable_identity(row.get("manifest"), label="portable phase manifest identity")
    if _require_sha256(row.get("manifest_sha256"), label="portable phase manifest digest") != manifest["sha256"]:
        raise R298ValidationError("portable phase manifest identity mismatch")
    normalized = validate_phase_evidence_manifest(
        {key: row[key] for key in expected - {"manifest", "manifest_sha256"}},
        verify_artifacts=False,
        engine_evidence_identity=None,
    )
    phases = normalized["phases"]
    if phases["engine_metamorphic_tests"]["artifact"] != engine["evidence"]:
        raise R298ValidationError("portable engine phase does not bind exact engine evidence artifact")
    if phases["collision_census"]["artifact"] != collision_census["receipt"]:
        raise R298ValidationError("portable collision phase does not bind exact census artifact")
    return {**normalized, "manifest": manifest, "manifest_sha256": row["manifest_sha256"]}


def build_validation_receipt(
    *,
    canonical_sources: Mapping[str, Any],
    baseline: Mapping[str, Any],
    baseline_r274: Mapping[str, Any],
    candidate: Mapping[str, Any],
    evaluation_disjointness: Mapping[str, Any],
    aggregate_evidence: Mapping[str, Any],
    host: str,
) -> dict[str, Any]:
    """Compile a receipt only after re-hashing every primary artifact on Elmo.

    This is intentionally not a convenience constructor: it refuses all
    incomplete evidence, including the currently expected state where the
    strict checklist filter is still inert rather than candidate action-time
    wiring.  Portable consumers use :func:`validate_validation_receipt` below
    and re-check all immutable identities and cross-links.
    """

    if host != "elmo":
        raise R298ValidationError("r298 validation receipt may be compiled only on Elmo")
    normalized_sources = _validate_canonical_sources(
        canonical_sources, verify_artifacts=True
    )
    normalized_baseline = _validate_baseline(baseline)
    normalized_baseline_r274 = _validate_baseline_r274(
        baseline_r274,
        baseline=normalized_baseline,
    )
    normalized_candidate = _validate_candidate(candidate, baseline=normalized_baseline)
    normalized_evaluation_disjointness = _validate_evaluation_disjointness(
        evaluation_disjointness
    )
    normalized_evidence = _validate_aggregate_evidence(
        aggregate_evidence,
        candidate=normalized_candidate,
        baseline_r274=normalized_baseline_r274,
        verify_artifacts=True,
    )
    attestations = {name: True for name in ATTESTATION_NAMES}
    receipt: dict[str, Any] = {
        "schema": R298_VALIDATION_RECEIPT_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed_elmo_only_nonproduction",
        "host": "elmo",
        "source_attachment": {"sha256": OWNER_GOAL_ATTACHMENT_SHA256},
        "mechanics_attachment": {"sha256": MECHANICS_ATTACHMENT_SHA256},
        # Kept as an explicit top-level field for the BO runner.  It must
        # exactly match the same immutable GOAL.md identity in the complete
        # canonical-source inventory below.
        "goal_contract": normalized_sources["goal_gateway"],
        "canonical_sources": normalized_sources,
        "baseline": normalized_baseline,
        "baseline_r274": normalized_baseline_r274,
        "candidate": normalized_candidate,
        "evaluation_disjointness": normalized_evaluation_disjointness,
        "attestations": attestations,
        "authority": {name: False for name in AUTHORITY_NAMES},
        "direct_policy_boundary": {
            name: False for name in DIRECT_POLICY_BOUNDARY_NAMES
        },
        "schema_freeze": normalized_evidence["schema_freeze"],
        "raw_expert_corpus": normalized_evidence["raw_expert_corpus"],
        "refeaturization_splits": normalized_evidence["refeaturization_splits"],
        "collision_census": normalized_evidence["collision_census"],
        "transfer_staging": normalized_evidence["transfer_staging"],
        "resource_receipt": normalized_evidence["resource_receipt"],
        "checklist_provenance": normalized_evidence["checklist_provenance"],
        "frozen_tensor_identity": normalized_evidence["frozen_tensor_identity"],
        "card2vec_non_regression": normalized_evidence["card2vec_non_regression"],
        "candidate_training": normalized_evidence["candidate_training"],
        "phase_evidence": normalized_evidence["phase_evidence"],
        "engine_evidence": normalized_evidence["engine_evidence"],
    }
    receipt["receipt_sha256"] = canonical_digest(receipt)
    return receipt


def validate_validation_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a compiled r298 receipt without reopening Elmo data archives.

    The builder has already re-hashed all live artifacts.  This portable path
    still revalidates exact schema, digest, source, evidence identity and
    cross-link claims; it never accepts an old sparse rev3-style receipt.
    """

    expected = {
        "schema",
        "owner_revision",
        "status",
        "host",
        "source_attachment",
        "mechanics_attachment",
        "goal_contract",
        "canonical_sources",
        "baseline",
        "baseline_r274",
        "candidate",
        "evaluation_disjointness",
        "attestations",
        "authority",
        "direct_policy_boundary",
        "schema_freeze",
        "raw_expert_corpus",
        "refeaturization_splits",
        "collision_census",
        "transfer_staging",
        "resource_receipt",
        "checklist_provenance",
        "frozen_tensor_identity",
        "card2vec_non_regression",
        "candidate_training",
        "phase_evidence",
        "engine_evidence",
        "receipt_sha256",
    }
    if set(payload) != expected:
        raise R298ValidationError("r298 validation receipt key inventory changed")
    if payload.get("schema") != R298_VALIDATION_RECEIPT_SCHEMA:
        raise R298ValidationError("r298 validation receipt schema mismatch")
    if payload.get("owner_revision") != R298_OWNER_REVISION:
        raise R298ValidationError("r298 validation receipt owner revision mismatch")
    if payload.get("status") != "passed_elmo_only_nonproduction" or payload.get("host") != "elmo":
        raise R298ValidationError("r298 receipt is not an Elmo-only passed non-production receipt")
    _validate_attachments(
        {
            "source_attachment": payload.get("source_attachment"),
            "mechanics_attachment": payload.get("mechanics_attachment"),
        }
    )
    normalized_sources = _validate_canonical_sources(
        payload.get("canonical_sources"), verify_artifacts=False
    )
    goal_contract = _identity_from_mapping(
        payload.get("goal_contract"), label="r298 goal_contract"
    ).to_dict()
    if goal_contract != normalized_sources["goal_gateway"]:
        raise R298ValidationError(
            "r298 goal_contract must exactly match canonical_sources.goal_gateway"
        )
    normalized_baseline = _validate_baseline(payload.get("baseline"))
    normalized_baseline_r274 = _validate_baseline_r274(
        payload.get("baseline_r274"), baseline=normalized_baseline
    )
    normalized_candidate = _validate_candidate(payload.get("candidate"), baseline=normalized_baseline)
    normalized_evaluation_disjointness = _validate_evaluation_disjointness(
        payload.get("evaluation_disjointness")
    )
    attestations = _mapping(payload.get("attestations"), label="r298 attestations")
    if set(attestations) != set(ATTESTATION_NAMES):
        raise R298ValidationError("r298 attestation inventory changed")
    for name in ATTESTATION_NAMES:
        _require_bool(attestations[name], label=f"r298 attestation {name}", expected=True)
    authority = _all_false_mapping(
        payload.get("authority"), label="r298 authority", names=AUTHORITY_NAMES
    )
    boundary = _all_false_mapping(
        payload.get("direct_policy_boundary"),
        label="r298 direct policy boundary",
        names=DIRECT_POLICY_BOUNDARY_NAMES,
    )
    normalized_schema_freeze = _validate_portable_schema_freeze(
        payload.get("schema_freeze")
    )
    normalized_raw_corpus = _validate_portable_raw_corpus(
        payload.get("raw_expert_corpus"), schema_freeze=normalized_schema_freeze
    )
    normalized_refeaturization = _validate_portable_refeaturization(
        payload.get("refeaturization_splits"), raw_corpus=normalized_raw_corpus
    )
    normalized_census = _validate_portable_collision_census(
        payload.get("collision_census"),
        raw_corpus=normalized_raw_corpus,
        schema_freeze=normalized_schema_freeze,
    )
    normalized_transfer = _validate_portable_transfer_staging(
        payload.get("transfer_staging"),
        raw_corpus=normalized_raw_corpus,
        schema_freeze=normalized_schema_freeze,
    )
    normalized_resource = _validate_portable_resource_bundle(payload.get("resource_receipt"))
    normalized_checklist = _validate_portable_checklist(
        payload.get("checklist_provenance"),
        candidate_checkpoint_sha256=normalized_candidate["checkpoint_sha256"],
    )
    if normalized_checklist["candidate_action_time_wrapper_sealed"] is not True:
        raise R298ValidationError("aggregate receipt lacks sealed candidate action-time checklist wrapper")
    normalized_frozen = _validate_portable_frozen_tensor(
        payload.get("frozen_tensor_identity"),
        parent_checkpoint_sha256=normalized_baseline_r274["checkpoint_sha256"],
        candidate_checkpoint_sha256=normalized_candidate["checkpoint_sha256"],
    )
    normalized_card2vec = _validate_portable_card2vec(
        payload.get("card2vec_non_regression"),
        parent_checkpoint_sha256=normalized_baseline_r274["checkpoint_sha256"],
        candidate_checkpoint_sha256=normalized_candidate["checkpoint_sha256"],
    )
    normalized_candidate_training = _validate_portable_candidate_training(
        payload.get("candidate_training"),
        candidate=normalized_candidate,
        baseline_r274=normalized_baseline_r274,
        schema_freeze=normalized_schema_freeze,
        raw_corpus=normalized_raw_corpus,
        refeaturization=normalized_refeaturization,
        collision_census=normalized_census,
        transfer_staging=normalized_transfer,
        frozen_tensor=normalized_frozen,
        card2vec=normalized_card2vec,
    )
    normalized_engine = _validate_portable_engine(payload.get("engine_evidence"))
    normalized_phases = _validate_portable_phase_evidence(
        payload.get("phase_evidence"),
        engine=normalized_engine,
        collision_census=normalized_census,
    )
    expected_without_digest = {
        "schema": R298_VALIDATION_RECEIPT_SCHEMA,
        "owner_revision": R298_OWNER_REVISION,
        "status": "passed_elmo_only_nonproduction",
        "host": "elmo",
        "source_attachment": {"sha256": OWNER_GOAL_ATTACHMENT_SHA256},
        "mechanics_attachment": {"sha256": MECHANICS_ATTACHMENT_SHA256},
        "goal_contract": goal_contract,
        "canonical_sources": normalized_sources,
        "baseline": normalized_baseline,
        "baseline_r274": normalized_baseline_r274,
        "candidate": normalized_candidate,
        "evaluation_disjointness": normalized_evaluation_disjointness,
        "attestations": {name: True for name in ATTESTATION_NAMES},
        "authority": authority,
        "direct_policy_boundary": boundary,
        "schema_freeze": normalized_schema_freeze,
        "raw_expert_corpus": normalized_raw_corpus,
        "refeaturization_splits": normalized_refeaturization,
        "collision_census": normalized_census,
        "transfer_staging": normalized_transfer,
        "resource_receipt": normalized_resource,
        "checklist_provenance": normalized_checklist,
        "frozen_tensor_identity": normalized_frozen,
        "card2vec_non_regression": normalized_card2vec,
        "candidate_training": normalized_candidate_training,
        "phase_evidence": normalized_phases,
        "engine_evidence": normalized_engine,
    }
    if payload.get("receipt_sha256") != canonical_digest(expected_without_digest):
        raise R298ValidationError("r298 validation receipt digest mismatch")
    return dict(payload)


def _trusted_root_path(value: Path | str) -> Path:
    """Resolve the one approved Elmo evidence root without following aliases."""

    root = Path(value).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise R298ValidationError("trusted Elmo output root must be a physical directory")
    resolved = root.resolve()
    if not resolved.is_dir():  # pragma: no cover - defensive filesystem race guard.
        raise R298ValidationError("trusted Elmo output root disappeared")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_trusted_readonly_identity(
    value: object,
    *,
    label: str,
    trusted_root: Path,
) -> tuple[FileIdentity, dict[str, Any] | None]:
    """Re-open one aggregate artifact and prove it lives beneath the root.

    Portable receipt validation deliberately permits an identity mapping from a
    remote archive.  BO authorization may not.  This helper turns every such
    mapping into a physical regular file, rejects symlinked path components or
    writable evidence, and compares the claimed SHA/size with bytes read now.
    """

    claimed = _identity_from_mapping(value, label=label)
    candidate = Path(claimed.path).expanduser()
    if candidate.is_symlink():
        raise R298ValidationError(f"{label} may not be a symlink")
    resolved = candidate.resolve()
    if not _is_within(resolved, trusted_root):
        raise R298ValidationError(f"{label} must be under the trusted Elmo output root")
    # Refuse a symlink in any component between the immutable root and leaf.
    relative = resolved.relative_to(trusted_root)
    cursor = trusted_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise R298ValidationError(f"{label} contains a symlinked path component")
    actual = _validate_identity_on_disk(claimed, label=label)
    try:
        mode = Path(actual.path).stat().st_mode
    except OSError as exc:  # pragma: no cover - checked in _validate_identity_on_disk.
        raise R298ValidationError(f"{label} cannot be statted") from exc
    if mode & 0o222:
        raise R298ValidationError(f"{label} must be immutable/read-only before BO authorization")
    parsed: dict[str, Any] | None = None
    try:
        loaded = json.loads(Path(actual.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        loaded = None
    if loaded is not None:
        if not isinstance(loaded, dict):
            raise R298ValidationError(f"{label} JSON artifact must contain an object")
        parsed = loaded
    return actual, parsed


def _validate_exact_canonical_source_files(
    canonical_sources: Mapping[str, Any],
) -> None:
    """Rehash the three sole canonical sources at their exact repo paths."""

    for name in CANONICAL_SOURCE_NAMES:
        expected_path = (PROJECT_ROOT / CANONICAL_SOURCE_RELATIVE_PATHS[name]).resolve()
        claimed = _identity_from_mapping(canonical_sources.get(name), label=f"canonical source {name}")
        if Path(claimed.path).expanduser().resolve() != expected_path:
            raise R298ValidationError(f"canonical source {name} path is not the exact canonical path")
        actual = _validate_identity_on_disk(claimed, label=f"canonical source {name}")
        if actual.sha256 != CANONICAL_SOURCE_SHA256S[name]:
            raise R298ValidationError(f"canonical source {name} bytes no longer match the pinned digest")


def _trusted_json_artifact(
    identity: object,
    *,
    label: str,
    trusted_root: Path,
) -> tuple[FileIdentity, dict[str, Any]]:
    actual, payload = _validate_trusted_readonly_identity(
        identity, label=label, trusted_root=trusted_root
    )
    if payload is None:
        raise R298ValidationError(f"{label} must be a JSON evidence artifact")
    return actual, payload


def validate_trusted_validation_receipt_file(
    path: Path | str,
    *,
    trusted_output_root: Path | str = R298_TRUSTED_ELMO_OUTPUT_ROOT,
) -> FileIdentity:
    """Authorize only a physically revalidated aggregate receipt for BO.

    This is intentionally separate from :func:`validate_validation_receipt`:
    the latter is a portable *structural* parser useful for reports and must
    never be used to launch a benchmark.  This verifier opens the aggregate,
    every subordinate receipt, the exact canonical sources, and raw archives
    again.  It also requires read-only, non-symlink evidence below the one
    sealed Elmo output root.  It performs no simulator, training, Inzi, or
    service action.
    """

    trusted_root = _trusted_root_path(trusted_output_root)
    aggregate_identity, aggregate = _trusted_json_artifact(
        file_identity(path, label="r298 aggregate validation receipt").to_dict(),
        label="r298 aggregate validation receipt",
        trusted_root=trusted_root,
    )
    # First prove the nested structural/cross-link contract.  This means the
    # subsequent file traversal is driven by a fixed schema rather than loose
    # field probing.
    receipt = validate_validation_receipt(aggregate)
    _validate_exact_canonical_source_files(
        _mapping(receipt.get("canonical_sources"), label="trusted canonical sources")
    )

    schema_freeze = _mapping(receipt["schema_freeze"], label="trusted schema freeze")
    schema_identity, schema_payload = _trusted_json_artifact(
        schema_freeze["schema_manifest"],
        label="trusted frozen-schema manifest",
        trusted_root=trusted_root,
    )
    zero_identity, zero_payload = _trusted_json_artifact(
        schema_freeze["zero_bypass_receipt"],
        label="trusted zero-bypass receipt",
        trusted_root=trusted_root,
    )
    trusted_schema = validate_collision_schema_freeze_evidence(
        schema_payload,
        zero_payload,
        schema_manifest_identity=schema_identity,
        zero_bypass_identity=zero_identity,
        verify_artifacts=True,
    )
    if trusted_schema != schema_freeze:
        raise R298ValidationError("trusted schema-freeze artifacts do not reproduce aggregate summary")

    raw = _mapping(receipt["raw_expert_corpus"], label="trusted raw corpus")
    raw_manifest_identity, raw_manifest = _trusted_json_artifact(
        raw["manifest"], label="trusted raw corpus manifest", trusted_root=trusted_root
    )
    raw_receipt_identity, raw_receipt = _trusted_json_artifact(
        raw["raw_corpus_receipt"], label="trusted raw corpus receipt", trusted_root=trusted_root
    )
    trusted_raw = validate_collision_raw_corpus_evidence(
        raw_manifest,
        raw_receipt,
        manifest_identity=raw_manifest_identity,
        raw_receipt_identity=raw_receipt_identity,
        verify_artifacts=True,
        verify_archive_paths=True,
    )
    # The split proof is produced after raw validation and is the only place
    # that can add the source/day/group-disjoint assertion to the compact row.
    ref = _mapping(receipt["refeaturization_splits"], label="trusted refeaturization")
    ref_identity, ref_payload = _trusted_json_artifact(
        ref["receipt"], label="trusted refeaturization receipt", trusted_root=trusted_root
    )
    trusted_ref = validate_refeaturization_split_receipt(
        ref_payload,
        raw_manifest_sha256=str(trusted_raw["manifest_sha256"]),
        identity=ref_identity,
        verify_artifacts=True,
    )
    trusted_raw["source_day_group_disjoint"] = bool(trusted_ref["source_day_group_disjoint"])
    if trusted_raw != raw:
        raise R298ValidationError("trusted raw artifacts do not reproduce aggregate summary")
    if trusted_ref != ref:
        raise R298ValidationError("trusted re-featurization artifact does not reproduce aggregate summary")

    census = _mapping(receipt["collision_census"], label="trusted collision census")
    census_identity, census_payload = _trusted_json_artifact(
        census["receipt"], label="trusted collision census receipt", trusted_root=trusted_root
    )
    trusted_census = validate_collision_census_evidence(
        census_payload,
        raw_corpus=trusted_raw,
        schema_freeze=trusted_schema,
        identity=census_identity,
        verify_artifacts=True,
    )
    if trusted_census != census:
        raise R298ValidationError("trusted census artifact does not reproduce aggregate summary")

    transfer = _mapping(receipt["transfer_staging"], label="trusted transfer staging")
    trusted_per_shards: list[dict[str, Any]] = []
    for index, identity in enumerate(transfer["per_shard_receipt_identities"]):
        actual, payload = _trusted_json_artifact(
            identity,
            label=f"trusted transfer shard receipt[{index}]",
            trusted_root=trusted_root,
        )
        if actual.to_dict() != identity or payload != transfer["per_shard_receipts"][index]:
            raise R298ValidationError("trusted transfer shard receipt differs from aggregate copy")
        trusted_per_shards.append(payload)
    parity_identity, parity_payload = _trusted_json_artifact(
        transfer["parity_receipt_identity"],
        label="trusted transfer parity receipt",
        trusted_root=trusted_root,
    )
    if parity_identity.to_dict() != transfer["parity_receipt_identity"] or parity_payload != transfer["parity_receipt"]:
        raise R298ValidationError("trusted transfer parity receipt differs from aggregate copy")
    trusted_transfer = validate_transfer_staging_evidence(
        {
            "per_shard_receipts": trusted_per_shards,
            "per_shard_receipt_identities": transfer["per_shard_receipt_identities"],
            "parity_receipt": parity_payload,
            "parity_receipt_identity": parity_identity.to_dict(),
        },
        raw_manifest_sha256=str(trusted_raw["manifest_sha256"]),
        schema_freeze=trusted_schema,
        verify_artifacts=True,
    )
    if trusted_transfer != transfer:
        raise R298ValidationError("trusted transfer evidence does not reproduce aggregate summary")

    resource = _mapping(receipt["resource_receipt"], label="trusted resource evidence")
    resource_jobs = _mapping(resource["jobs"], label="trusted resource jobs")
    resource_rows: dict[str, Any] = {}
    for name in RESOURCE_JOB_NAMES:
        job = _mapping(resource_jobs[name], label=f"trusted resource job {name}")
        resource_identity, resource_payload = _trusted_json_artifact(
            job["receipt"], label=f"trusted resource receipt {name}", trusted_root=trusted_root
        )
        if resource_identity.to_dict() != job["receipt"]:
            raise R298ValidationError("trusted resource receipt identity differs from aggregate copy")
        resource_rows[name] = {"payload": resource_payload, "identity": resource_identity.to_dict()}
    trusted_resources = _validate_resource_evidence_bundle(resource_rows, verify_artifacts=True)
    if trusted_resources != resource:
        raise R298ValidationError("trusted resource receipts do not reproduce aggregate summary")

    checklist = _mapping(receipt["checklist_provenance"], label="trusted checklist provenance")
    checklist_identity, checklist_payload = _trusted_json_artifact(
        checklist["evidence"], label="trusted checklist evidence", trusted_root=trusted_root
    )
    # Stage receipts are typed immutable evidence too—not mere SHA strings.
    for index, identity in enumerate(checklist.get("stage_receipts", ())):
        _trusted_json_artifact(
            identity,
            label=f"trusted action-time checklist stage receipt[{index}]",
            trusted_root=trusted_root,
        )
    trusted_checklist = validate_checklist_provenance_evidence(
        checklist_payload,
        schema_freeze=trusted_schema,
        identity=checklist_identity,
        verify_artifacts=True,
    )
    if trusted_checklist != checklist:
        raise R298ValidationError("trusted checklist evidence does not reproduce aggregate summary")
    if trusted_checklist.get("candidate_checkpoint_sha256") != receipt["candidate"]["checkpoint_sha256"]:
        raise R298ValidationError("trusted checklist evidence binds a different candidate checkpoint")

    baseline = _mapping(receipt["baseline_r274"], label="trusted baseline r274")
    candidate = _mapping(receipt["candidate"], label="trusted candidate")
    frozen = _mapping(receipt["frozen_tensor_identity"], label="trusted frozen tensor audit")
    frozen_identity, frozen_payload = _trusted_json_artifact(
        frozen["evidence"], label="trusted frozen tensor audit", trusted_root=trusted_root
    )
    trusted_frozen = validate_frozen_tensor_audit(
        frozen_payload,
        parent_checkpoint_sha256=str(baseline["checkpoint_sha256"]),
        candidate_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        identity=frozen_identity,
        verify_artifacts=True,
    )
    if trusted_frozen != frozen:
        raise R298ValidationError("trusted frozen-tensor evidence does not reproduce aggregate summary")
    card2vec = _mapping(receipt["card2vec_non_regression"], label="trusted Card2Vec evidence")
    card2vec_identity, card2vec_payload = _trusted_json_artifact(
        card2vec["evidence"], label="trusted Card2Vec evidence", trusted_root=trusted_root
    )
    trusted_card2vec = validate_card2vec_nonregression_evidence(
        card2vec_payload,
        parent_checkpoint_sha256=str(baseline["checkpoint_sha256"]),
        candidate_checkpoint_sha256=str(candidate["checkpoint_sha256"]),
        identity=card2vec_identity,
        verify_artifacts=True,
    )
    if trusted_card2vec != card2vec:
        raise R298ValidationError("trusted Card2Vec evidence does not reproduce aggregate summary")

    candidate_training = _mapping(receipt["candidate_training"], label="trusted candidate training")
    candidate_training_identity, candidate_training_payload = _trusted_json_artifact(
        candidate_training["receipt"],
        label="trusted candidate training receipt",
        trusted_root=trusted_root,
    )
    trusted_training = validate_candidate_training_evidence(
        candidate_training_payload,
        candidate=candidate,
        baseline_r274=baseline,
        schema_freeze=trusted_schema,
        raw_corpus=trusted_raw,
        refeaturization=trusted_ref,
        collision_census=trusted_census,
        transfer_staging=trusted_transfer,
        frozen_tensor=trusted_frozen,
        card2vec=trusted_card2vec,
        identity=candidate_training_identity,
        verify_artifacts=True,
    )
    if trusted_training != candidate_training:
        raise R298ValidationError("trusted candidate-training evidence does not reproduce aggregate summary")

    engine = _mapping(receipt["engine_evidence"], label="trusted engine evidence")
    engine_identity, engine_payload = _trusted_json_artifact(
        engine["evidence"], label="trusted engine evidence", trusted_root=trusted_root
    )
    trusted_engine = validate_engine_evidence(engine_payload)
    trusted_engine["evidence"] = engine_identity.to_dict()
    trusted_engine["evidence_sha256"] = canonical_digest(engine_payload)
    if trusted_engine != engine:
        raise R298ValidationError("trusted engine evidence does not reproduce aggregate summary")
    phases = _mapping(receipt["phase_evidence"], label="trusted phase evidence")
    phase_identity, phase_payload = _trusted_json_artifact(
        phases["manifest"], label="trusted phase-evidence manifest", trusted_root=trusted_root
    )
    trusted_phases = validate_phase_evidence_manifest(
        phase_payload, verify_artifacts=True, engine_evidence_identity=engine_identity
    )
    trusted_phases["manifest"] = phase_identity.to_dict()
    trusted_phases["manifest_sha256"] = canonical_digest(phase_payload)
    if trusted_phases["phases"]["collision_census"]["artifact"] != trusted_census["receipt"]:
        raise R298ValidationError("trusted phase evidence does not bind the trusted census receipt")
    if trusted_phases != phases:
        raise R298ValidationError("trusted phase evidence does not reproduce aggregate summary")
    return aggregate_identity


def write_create_only_json(path: Path | str, payload: Mapping[str, Any]) -> FileIdentity:
    """Atomically write one immutable receipt, refusing an existing target."""

    output = Path(path).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"r298 validation receipt already exists: {output}")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise R298ValidationError("r298 validation receipt parent must be a physical directory")
    encoded = canonical_json_bytes(dict(payload))
    descriptor: int | None = None
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    identity = file_identity(output, label="written r298 validation receipt")
    if identity.sha256 != sha256_file(output):  # pragma: no cover - defensive I/O race guard
        raise R298ValidationError("r298 validation receipt identity drifted after write")
    return identity


__all__ = [
    "ATTESTATION_NAMES",
    "AUTHORITY_NAMES",
    "CANONICAL_LIBCG_SHA256",
    "CANONICAL_SOURCE_NAMES",
    "DIRECT_POLICY_BOUNDARY_NAMES",
    "ENGINE_CASE_NAMES",
    "EVALUATION_DISJOINTNESS_NAMES",
    "EXACT_NEW_LIST_MULTISET_SHA256",
    "FileIdentity",
    "MECHANICS_ATTACHMENT_SHA256",
    "OWNER_GOAL_ATTACHMENT_SHA256",
    "PHASE_NAMES",
    "PHASE_REQUIRED_ASSERTIONS",
    "R195_CHECKPOINT_SHA256",
    "R298_ENGINE_EVIDENCE_SCHEMA",
    "R298_OWNER_REVISION",
    "R298_PHASE_EVIDENCE_MANIFEST_SCHEMA",
    "R298_TRUSTED_ELMO_OUTPUT_ROOT",
    "R298_VALIDATION_RECEIPT_SCHEMA",
    "R298ValidationError",
    "build_validation_receipt",
    "canonical_digest",
    "canonical_json_bytes",
    "file_identity",
    "read_json_object",
    "sha256_file",
    "validate_engine_evidence",
    "validate_phase_evidence_manifest",
    "validate_trusted_validation_receipt_file",
    "validate_validation_receipt",
    "write_create_only_json",
]
