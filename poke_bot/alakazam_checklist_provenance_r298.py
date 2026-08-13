"""Strict, inert r298 provenance boundary for the eight-turn checklist.

This module is intentionally *not* a policy layer.  It does not import the
legacy r288 checklist, card CSVs, guide prose, learned heads, or a simulator
wrapper.  It validates receipts and filters an already-produced r288-shaped
trace into a separately versioned, exact-zero diagnostic record for the
isolated Elmo experiment.

The narrow purpose is to stop a useful historical trace from being mistaken
for exact mechanics authority.  In particular, Q1/Q2/Q8 may retain diagnostic
evidence only when an explicit revision-5 consumer-migration receipt, an
immutable pinned public-catalog artifact, an owner-pinned simulator fact
witness, the canonical r236 libcg identity, the exact legal-option binding,
and every required fact authorization agree.  The prior revision-4 catalog is
predecessor evidence only; it cannot be blindly substituted.  Otherwise those
rows are explicitly unavailable and zero.  Q3 remains Bench-only; Q5/Q6 are
always trace-only with zero final residuals.  Every output residual is zero
while the r298 derivative remains default-off.

Nothing in this file is wired into production, Inzi, a selector, training, or
the existing r288 layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional


R298_CHECKLIST_PROVENANCE_SCHEMA = "poke_bot.alakazam_checklist_provenance_r298/v1"
R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA = (
    "poke_bot.alakazam_checklist_provenance_result_r298/v1"
)
R298_CHECKLIST_STAGE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_checklist_stage_receipt_r298/v1"
)
R298_CHECKLIST_PROVENANCE_CONFIG_SCHEMA = (
    "poke_bot.alakazam_checklist_provenance_r298_config/v1"
)
R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_public_catalog_r298_receipt/v1"
)
R298_PINNED_PUBLIC_CATALOG_SCHEMA = "poke_bot.alakazam_public_catalog_r298/v1"
R298_SIMULATOR_FACT_WITNESS_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_checklist_simulator_fact_witness_r298/v1"
)
R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_checklist_provenance_r298_consumer_migration/v1"
)
R298_ENGINE_EVIDENCE_SCHEMA = (
    "poke_bot.alakazam_simulator_rules_r298_engine_evidence/v1"
)
R298_CHECKLIST_PROVENANCE_REVISION = 298

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "policy_layers"
    / "alakazam-checklist-provenance-r298.json"
)
CANONICAL_GOAL_PATH = PROJECT_ROOT / "goals" / "alakazam-elmo-rule-derivative" / "GOAL.md"
CANONICAL_CONTRACT_PATH = (
    PROJECT_ROOT / "goals" / "alakazam-elmo-rule-derivative" / "contract.json"
)
CANONICAL_LIBCG_CONTRACT_PATH = PROJECT_ROOT / "state" / "canonical-libcg-r236.json"

# These values are duplicated in the checked configuration so a changed
# canonical source cannot silently be accepted through a caller-provided
# provenance record.  ``load_checklist_provenance_config_r298`` re-hashes the
# canonical gateway, contract, typed r241 source, and r236 source every time
# it opens a config.
CANONICAL_GOAL_SHA256 = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)
CANONICAL_CONTRACT_SHA256 = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
CANONICAL_R241_TYPED_SOURCE_PATH = (
    PROJECT_ROOT / "state" / "alakazam-new-list-direct-policy-r241.json"
)
CANONICAL_R241_TYPED_SOURCE_SHA256 = (
    "sha256:8d83f5e9eafc8e554f33dcbfbda7e1b337b8f65dc2e49109be4acdd829850c1e"
)
CANONICAL_LIBCG_CONTRACT_SHA256 = (
    "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"
)
CANONICAL_LIBCG_BINARY_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
CANONICAL_LIBCG_MEMBER = "cg/libcg.so"
CANONICAL_LIBCG_SIZE_BYTES = 1_342_400
CANONICAL_KAGGLE_ENVIRONMENTS_VERSION = "1.32.6"
CATALOG_SEALER_SOURCE_PATH = PROJECT_ROOT / "scripts" / "seal_alakazam_public_catalog_r298.py"
CATALOG_SEALER_SOURCE_SHA256 = (
    "sha256:dfe952531ad372b970522a11d9c60d1a18375caeb006fe0410989967dba66b4d"
)
CATALOG_SEALER_CSV_IDENTITY_SHA256 = (
    "sha256:408bc978661c8b0628e5f17b27693dc8da9c732472168f5574999be4774031c1"
)
# The only sealed catalog currently available was created against the revision
# 4 gateway/contract.  Revision 5 explicitly preserves these bytes as
# historical evidence but forbids blind hash substitution.  The identities
# below are therefore useful only to validate the predecessor artifact; they
# never authorize Q1/Q2/Q8 under the revision-5 consumer.
PREDECESSOR_GOAL_REVISION = 4
PREDECESSOR_GATEWAY_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
PREDECESSOR_CONTRACT_SHA256 = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
CATALOG_SEALER_IMMUTABLE_ARTIFACT_PINS: Mapping[str, str | int] = MappingProxyType(
    {
        # These are the actual create-only Elmo sealer outputs.  A receipt that
        # merely claims the r298 envelope is not an authority root.
        "receipt_file_sha256": "sha256:9ca7534281dd08d1804babe223fa0779a84a1f5a7ee6f609497d595dca10b4d1",
        "receipt_file_size_bytes": 3_556,
        "receipt_payload_sha256": "sha256:89271e51000fae3abeb4227d1b870b4976f592a571edbbef83f4aee343c9355d",
        "ready_file_sha256": "sha256:b0e14b770ca8fd5152583769bed79d2f676fa8f7c931473d548ebead6fc4bffb",
        "ready_file_size_bytes": 574,
        "catalog_file_sha256": "sha256:4d1c35124cdeeddcaca34a7d0ab3f2fc94e4257fe4578a03c8608ac561d00df6",
        "catalog_file_size_bytes": 707_563,
        "catalog_semantic_sha256": "sha256:bae13f773d27f305dff4774b9ddae32e0c02da8023fc8f142be7a8634c3fd3a9",
        "fixed_vectors_file_sha256": "sha256:2e1a817dac1d17d0131056ead9669b76803b41f5864d4e9cd82d7f34a9180e09",
        "fixed_vectors_file_size_bytes": 451_416,
        "structured_rule_vectors_sha256": "sha256:6602af831128364a7a33ea3e691a73e4cac7e39ad037855ada7b13c80f87ebb1",
        "engine_cards_sha256": "sha256:5e92642577d5e61324e4d4095b883fb7926e4d7e822967b48778ecd1996998f4",
        "engine_attacks_sha256": "sha256:97834eb17429fbacfedcf563d43753b129103c0a1d1941d1ee9c9af36392be3f",
    }
)
REV5_CONSUMER_MIGRATION_IMMUTABLE_ANCHOR: Mapping[str, str | None] = MappingProxyType(
    {
        # Create-only Elmo revision-5 consumer migration receipt.  It binds
        # the historical revision-4 catalog as predecessor evidence while the
        # final schema-freeze receipt separately binds the resulting config
        # and zero-bypass state.
        "status": "pinned_immutable_artifact",
        "receipt_file_sha256": "sha256:d33aa897f244c18203bf3851b70b7269cae928fc6e8e2447e5a363b8817f77de",
        "receipt_payload_sha256": "sha256:d33aa897f244c18203bf3851b70b7269cae928fc6e8e2447e5a363b8817f77de",
    }
)
SIMULATOR_FACT_WITNESS_IMMUTABLE_ANCHOR: Mapping[str, str | None] = MappingProxyType(
    {
        # No owner-issued, immutable public-frame fact witness has been frozen
        # yet.  Do not promote a self-consistent local JSON chain into one.
        "status": "not_issued_fail_closed",
        "receipt_file_sha256": None,
        "receipt_payload_sha256": None,
        "engine_evidence_file_sha256": None,
        "engine_evidence_payload_sha256": None,
    }
)
R298_EXPERIMENT_RAM_BYTES = 96 * 1024**3

PHASE_4_ASSERTION = "q1_q2_q8_pinned_provenance_or_zero_unavailable"
REV5_CONSUMER_MIGRATION_ASSERTIONS: tuple[str, ...] = (
    "read_revision_5_gateway_and_contract_completely",
    "revision_4_catalog_classified_immutable_predecessor_evidence",
    "no_blind_goal_or_contract_hash_substitution",
    "revision_5_config_goal_contract_and_r241_bindings_validated",
    "zero_inert_layer_off_and_card2vec_tests_passed",
    "unit_engine_information_set_layer_off_and_contract_binding_tests_passed",
    PHASE_4_ASSERTION,
    "revision_5_validation_receipt_checksum_bound_by_consumer",
)

CHANNEL_NAMES: tuple[str, ...] = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "replacement_alakazam_line",
    "unavoidable_draws_before_attack",
    "bench_prize_exposure",
    "immediate_disruption_outcome",
    "unknown_prize_robust_line",
    "terminal_before_forced_draw",
)
STRICT_CHANNELS: tuple[str, ...] = (
    "ko_hand_threshold",
    "safe_spend_above_threshold",
    "terminal_before_forced_draw",
)
TRACE_ONLY_CHANNELS: tuple[str, ...] = (
    "bench_prize_exposure",
    "immediate_disruption_outcome",
)
GUIDE_SUPPORT_CHANNEL = "guide_support"

STRICT_REQUIRED_FACT_KINDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ko_hand_threshold": (
            "legal_attack_option",
            "damage_formula",
            "defender_remaining_hp",
            "attack_effect_prevention",
        ),
        "safe_spend_above_threshold": (
            "post_cost_hand_delta",
            "acting_hand_count",
            "ko_hand_threshold",
        ),
        "terminal_before_forced_draw": (
            "legal_attack_option",
            "knockout_resolution",
            "terminal_prize_or_last_pokemon",
            "forced_draw_timing",
        ),
    }
)
FACT_AUTHORITY_BY_KIND: Mapping[str, str] = MappingProxyType(
    {
        "legal_attack_option": "pinned_libcg_legal_option",
        "damage_formula": "receipt_bound_simulator_mechanics",
        "defender_remaining_hp": "public_observation",
        "attack_effect_prevention": "receipt_bound_public_catalog",
        "post_cost_hand_delta": "receipt_bound_simulator_mechanics",
        "acting_hand_count": "public_observation",
        "ko_hand_threshold": "receipt_bound_simulator_mechanics",
        "knockout_resolution": "receipt_bound_simulator_mechanics",
        "terminal_prize_or_last_pokemon": "receipt_bound_public_catalog",
        "forced_draw_timing": "receipt_bound_simulator_mechanics",
    }
)
Q3_CLASSIFICATIONS: tuple[str, ...] = (
    "ready",
    "completable",
    "not_live",
    "unavailable",
)

_SHA256_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_AUTHORITY_TOKENS = frozenset({"guide", "prose", "csv", "text_hash"})


class ChecklistProvenanceError(ValueError):
    """The isolated r298 provenance schema cannot be trusted."""


def canonical_json_bytes_r298(payload: object) -> bytes:
    """Return the one canonical encoding used by r298 receipt digests."""

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


def canonical_digest_r298(payload: object) -> str:
    """Content-address a JSON-safe r298 payload."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes_r298(payload)).hexdigest()


def _receipt_digest(payload: Mapping[str, Any]) -> str:
    """Digest a receipt without its self-referential digest field."""

    material = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return canonical_digest_r298(material)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ChecklistProvenanceError(f"{label} must be a sha256 digest")
    encoded = value.removeprefix("sha256:")
    if len(encoded) != 64 or any(character not in _SHA256_HEX for character in encoded):
        raise ChecklistProvenanceError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChecklistProvenanceError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _require_bool(value: object, *, label: str, expected: Optional[bool] = None) -> bool:
    if type(value) is not bool:
        raise ChecklistProvenanceError(f"{label} must be a bool")
    result = bool(value)
    if expected is not None and result is not expected:
        raise ChecklistProvenanceError(f"{label} must be {expected}")
    return result


def _require_string(value: object, *, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ChecklistProvenanceError(f"{label} must be a nonempty string")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ChecklistProvenanceError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ChecklistProvenanceError(f"{label} must be an array")
    return value


def sha256_file_r298(path: Path | str) -> str:
    """Hash an immutable artifact without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class FileIdentityR298:
    """Exact identity for a regular, non-symlink evidence file."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def file_identity_r298(path: Path | str, *, label: str = "artifact") -> FileIdentityR298:
    """Return a verified identity, refusing symlinks and special files."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ChecklistProvenanceError(
            f"{label} must be a regular non-symlink file: {candidate}"
        )
    resolved = candidate.resolve()
    try:
        details = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise ChecklistProvenanceError(f"{label} is unreadable: {resolved}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise ChecklistProvenanceError(f"{label} is not a regular file: {resolved}")
    return FileIdentityR298(
        path=str(resolved),
        sha256=sha256_file_r298(resolved),
        size_bytes=int(details.st_size),
    )


def _identity_from_mapping(value: object, *, label: str) -> FileIdentityR298:
    row = _mapping(value, label=label)
    expected = {"path", "sha256", "size_bytes"}
    if set(row) != expected:
        raise ChecklistProvenanceError(f"{label} keys must be exactly {sorted(expected)}")
    path = _require_string(row.get("path"), label=f"{label}.path")
    return FileIdentityR298(
        path=path,
        sha256=_require_sha256(row.get("sha256"), label=f"{label}.sha256"),
        size_bytes=_require_exact_int(row.get("size_bytes"), label=f"{label}.size_bytes", minimum=1),
    )


def _verify_identity(identity: FileIdentityR298, *, label: str) -> FileIdentityR298:
    actual = file_identity_r298(identity.path, label=label)
    if actual != identity:
        raise ChecklistProvenanceError(f"{label} identity drifted")
    return actual


def _read_json_identity(identity: FileIdentityR298, *, label: str) -> Mapping[str, Any]:
    _verify_identity(identity, label=label)
    try:
        payload = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChecklistProvenanceError(f"{label} is not readable JSON") from exc
    return _mapping(payload, label=label)


@dataclass(frozen=True)
class R298ChecklistProvenanceConfig:
    """Frozen source bindings and fact authorities for the isolated filter."""

    path: str
    sha256: str
    canonical_goal_sha256: str
    canonical_contract_sha256: str
    canonical_r241_typed_source_sha256: str
    canonical_libcg_contract_sha256: str
    canonical_libcg_sha256: str
    canonical_libcg_member: str
    canonical_libcg_size_bytes: int
    canonical_kaggle_environments_version: str
    catalog_sealer_source_path: str
    catalog_sealer_source_sha256: str
    catalog_sealer_receipt_schema: str
    catalog_schema: str
    catalog_sealer_receipt_filename: str
    catalog_sealer_catalog_filename: str
    catalog_sealer_fixed_vectors_filename: str
    catalog_sealer_required_host: str
    catalog_sealer_allowed_hostnames: tuple[str, ...]
    catalog_sealer_resource_peak_schema: str
    catalog_sealer_resource_peak_required_fields: tuple[str, ...]
    catalog_sealer_csv_identity_sha256: str
    catalog_sealer_immutable_artifact_pins: Mapping[str, str | int]
    predecessor_goal_revision: int
    predecessor_gateway_sha256: str
    predecessor_contract_sha256: str
    revision_5_consumer_migration_schema: str
    revision_5_consumer_migration_anchor: Mapping[str, str | None]
    revision_5_consumer_migration_required_assertions: tuple[str, ...]
    simulator_fact_witness_schema: str
    engine_evidence_schema: str
    simulator_fact_witness_status: str
    simulator_fact_witness_execution_kind: str
    simulator_fact_witness_required_host: str
    simulator_fact_witness_public_information_attestation: Mapping[str, bool]
    simulator_fact_witness_immutable_anchor: Mapping[str, str | None]
    strict_required_fact_kinds: Mapping[str, tuple[str, ...]]
    fact_authority_by_kind: Mapping[str, str]
    q3_classifications: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R298_CHECKLIST_PROVENANCE_CONFIG_SCHEMA,
            "revision": R298_CHECKLIST_PROVENANCE_REVISION,
            "path": self.path,
            "sha256": self.sha256,
            "canonical_goal_sha256": self.canonical_goal_sha256,
            "canonical_contract_sha256": self.canonical_contract_sha256,
            "canonical_r241_typed_source_sha256": self.canonical_r241_typed_source_sha256,
            "canonical_libcg_contract_sha256": self.canonical_libcg_contract_sha256,
            "canonical_libcg_sha256": self.canonical_libcg_sha256,
            "canonical_libcg_member": self.canonical_libcg_member,
            "canonical_libcg_size_bytes": self.canonical_libcg_size_bytes,
            "canonical_kaggle_environments_version": self.canonical_kaggle_environments_version,
            "catalog_sealer": {
                "source": self.catalog_sealer_source_path,
                "source_sha256": self.catalog_sealer_source_sha256,
                "receipt_schema": self.catalog_sealer_receipt_schema,
                "catalog_schema": self.catalog_schema,
                "receipt_filename": self.catalog_sealer_receipt_filename,
                "catalog_filename": self.catalog_sealer_catalog_filename,
                "fixed_vectors_filename": self.catalog_sealer_fixed_vectors_filename,
                "required_execution_host": self.catalog_sealer_required_host,
                "allowed_execution_hostnames": list(self.catalog_sealer_allowed_hostnames),
                "resource_peak_schema": self.catalog_sealer_resource_peak_schema,
                "resource_peak_required_fields": list(
                    self.catalog_sealer_resource_peak_required_fields
                ),
                "csv_identity_join_sha256": self.catalog_sealer_csv_identity_sha256,
                "immutable_artifact_pins": dict(
                    self.catalog_sealer_immutable_artifact_pins
                ),
                "historical_predecessor_only": True,
                "eligible_without_revision_5_consumer_migration": False,
            },
            "revision_5_consumer_migration": {
                "schema": self.revision_5_consumer_migration_schema,
                "predecessor_goal_revision": self.predecessor_goal_revision,
                "predecessor_gateway_sha256": self.predecessor_gateway_sha256,
                "predecessor_contract_sha256": self.predecessor_contract_sha256,
                "consumer_goal_revision": 5,
                "consumer_root_handoff_revision": 303,
                "consumer_r241_typed_source_sha256": self.canonical_r241_typed_source_sha256,
                "catalog_reuse_policy": "only_after_explicit_checksum_bound_revision_5_consumer_validation",
                "revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze": False,
                "immutable_anchor": dict(self.revision_5_consumer_migration_anchor),
                "required_assertions": list(
                    self.revision_5_consumer_migration_required_assertions
                ),
            },
            "simulator_fact_witness": {
                "receipt_schema": self.simulator_fact_witness_schema,
                "engine_evidence_schema": self.engine_evidence_schema,
                "status": self.simulator_fact_witness_status,
                "execution_kind": self.simulator_fact_witness_execution_kind,
                "required_execution_host": self.simulator_fact_witness_required_host,
                "required_public_information_attestation": dict(
                    self.simulator_fact_witness_public_information_attestation
                ),
                "immutable_anchor": dict(self.simulator_fact_witness_immutable_anchor),
            },
            "strict_required_fact_kinds": {
                name: list(values)
                for name, values in self.strict_required_fact_kinds.items()
            },
            "fact_authority_by_kind": dict(self.fact_authority_by_kind),
            "q3_classifications": list(self.q3_classifications),
            "runtime_wired": False,
            "strict_filter_output_residual_gate": 0.0,
        }


def _read_json_file(path: Path, *, label: str) -> tuple[FileIdentityR298, Mapping[str, Any]]:
    identity = file_identity_r298(path, label=label)
    try:
        payload = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChecklistProvenanceError(f"{label} is not readable JSON") from exc
    return identity, _mapping(payload, label=label)


def _validate_canonical_source_bindings(authority: Mapping[str, Any]) -> None:
    """Fail closed if a config records stale source bytes or paths."""

    expected_paths = {
        "canonical_goal": CANONICAL_GOAL_PATH,
        "canonical_contract": CANONICAL_CONTRACT_PATH,
        "canonical_r241_typed_source": CANONICAL_R241_TYPED_SOURCE_PATH,
        "canonical_libcg_contract": CANONICAL_LIBCG_CONTRACT_PATH,
    }
    expected_digests = {
        "canonical_goal_sha256": CANONICAL_GOAL_SHA256,
        "canonical_contract_sha256": CANONICAL_CONTRACT_SHA256,
        "canonical_r241_typed_source_sha256": CANONICAL_R241_TYPED_SOURCE_SHA256,
        "canonical_libcg_contract_sha256": CANONICAL_LIBCG_CONTRACT_SHA256,
    }
    for key, path in expected_paths.items():
        configured = _require_string(authority.get(key), label=f"config authority.{key}")
        if Path(configured) != path.relative_to(PROJECT_ROOT):
            raise ChecklistProvenanceError(f"config authority.{key} path changed")
    for key, expected in expected_digests.items():
        configured = _require_sha256(authority.get(key), label=f"config authority.{key}")
        if configured != expected:
            raise ChecklistProvenanceError(f"config authority.{key} is not frozen r298 identity")
    actual = {
        "canonical_goal_sha256": sha256_file_r298(CANONICAL_GOAL_PATH),
        "canonical_contract_sha256": sha256_file_r298(CANONICAL_CONTRACT_PATH),
        "canonical_r241_typed_source_sha256": sha256_file_r298(
            CANONICAL_R241_TYPED_SOURCE_PATH
        ),
        "canonical_libcg_contract_sha256": sha256_file_r298(CANONICAL_LIBCG_CONTRACT_PATH),
    }
    for key, expected in expected_digests.items():
        if actual[key] != expected:
            raise ChecklistProvenanceError(
                f"canonical source drifted: {key}; all checklist authority is unavailable"
            )


def load_checklist_provenance_config_r298(
    config_path: Path | str | None = None,
) -> R298ChecklistProvenanceConfig:
    """Load and verify the frozen, default-zero r298 checklist boundary config.

    This deliberately validates the on-disk dedicated goal, sole contract, and
    canonical r236 source before returning.  A stale config is not a weaker
    mode; it is an error that callers can turn into an unavailable zero row.
    """

    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    identity, payload = _read_json_file(path, label="r298 checklist provenance config")
    if payload.get("schema") != R298_CHECKLIST_PROVENANCE_CONFIG_SCHEMA:
        raise ChecklistProvenanceError("r298 checklist provenance config schema mismatch")
    if payload.get("revision") != R298_CHECKLIST_PROVENANCE_REVISION:
        raise ChecklistProvenanceError("r298 checklist provenance config revision mismatch")
    authority = _mapping(payload.get("authority"), label="config authority")
    _validate_canonical_source_bindings(authority)

    canonical_libcg = _mapping(payload.get("canonical_libcg"), label="config canonical_libcg")
    if canonical_libcg.get("sha256") != CANONICAL_LIBCG_BINARY_SHA256:
        raise ChecklistProvenanceError("config canonical libcg binary digest changed")
    if canonical_libcg.get("member") != CANONICAL_LIBCG_MEMBER:
        raise ChecklistProvenanceError("config canonical libcg member changed")
    if canonical_libcg.get("size_bytes") != CANONICAL_LIBCG_SIZE_BYTES:
        raise ChecklistProvenanceError("config canonical libcg size changed")
    if canonical_libcg.get("kaggle_environments_version") != CANONICAL_KAGGLE_ENVIRONMENTS_VERSION:
        raise ChecklistProvenanceError("config canonical libcg version changed")
    if (
        canonical_libcg.get("legal_option_authority")
        != "pinned_libcg_emitted_complete_ordered_legal_options"
    ):
        raise ChecklistProvenanceError("config legal option authority changed")

    catalog_sealer = _mapping(payload.get("catalog_sealer"), label="config catalog_sealer")
    expected_sealer = {
        "source": str(CATALOG_SEALER_SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": CATALOG_SEALER_SOURCE_SHA256,
        "receipt_schema": R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA,
        "catalog_schema": R298_PINNED_PUBLIC_CATALOG_SCHEMA,
        "receipt_filename": "receipt.json",
        "catalog_filename": "catalog.json",
        "fixed_vectors_filename": "fixed-vectors.pt",
        "required_execution_host": "elmo",
        "allowed_execution_hostnames": ["elmo", "truenas"],
        "resource_peak_schema": "poke_bot.alakazam_elmo_rule_derivative_resource_peak/v1",
        "resource_peak_required_fields": [
            "experiment_ram_measurement_method",
            "experiment_ram_peak_bytes",
            "cpu_process_or_utilization_peak",
            "gpu_memory_peak_bytes_per_device_or_not_applicable",
            "gpu_utilization_peak_per_device_or_not_applicable",
        ],
        "csv_identity_join_sha256": CATALOG_SEALER_CSV_IDENTITY_SHA256,
        "immutable_artifact_pins": dict(CATALOG_SEALER_IMMUTABLE_ARTIFACT_PINS),
        "historical_predecessor_only": True,
        "eligible_without_revision_5_consumer_migration": False,
    }
    if dict(catalog_sealer) != expected_sealer:
        raise ChecklistProvenanceError("config catalog sealer contract changed")
    if sha256_file_r298(CATALOG_SEALER_SOURCE_PATH) != CATALOG_SEALER_SOURCE_SHA256:
        raise ChecklistProvenanceError("catalog sealer source drifted")

    migration = _mapping(
        payload.get("revision_5_consumer_migration"),
        label="config revision_5_consumer_migration",
    )
    expected_migration = {
        "schema": R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA,
        "predecessor_goal_revision": PREDECESSOR_GOAL_REVISION,
        "predecessor_gateway_sha256": PREDECESSOR_GATEWAY_SHA256,
        "predecessor_contract_sha256": PREDECESSOR_CONTRACT_SHA256,
        "consumer_goal_revision": 5,
        "consumer_root_handoff_revision": 303,
        "consumer_r241_typed_source_sha256": CANONICAL_R241_TYPED_SOURCE_SHA256,
        "catalog_reuse_policy": "only_after_explicit_checksum_bound_revision_5_consumer_validation",
        "revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze": False,
        "immutable_anchor": dict(REV5_CONSUMER_MIGRATION_IMMUTABLE_ANCHOR),
        "required_assertions": list(REV5_CONSUMER_MIGRATION_ASSERTIONS),
    }
    if dict(migration) != expected_migration:
        raise ChecklistProvenanceError("config revision-5 consumer migration contract changed")

    fact_witness = _mapping(
        payload.get("simulator_fact_witness"), label="config simulator_fact_witness"
    )
    expected_public_attestation = {
        "acting_player_public_information_only": True,
        "opponent_hand_identities_used": False,
        "opponent_deck_order_used": False,
        "unrevealed_prize_identities_used": False,
        "future_transition_used": False,
    }
    expected_witness = {
        "receipt_schema": R298_SIMULATOR_FACT_WITNESS_RECEIPT_SCHEMA,
        "engine_evidence_schema": R298_ENGINE_EVIDENCE_SCHEMA,
        "status": "passed_elmo_only_nonproduction",
        "execution_kind": "pinned_libcg_r236_public_frame",
        "required_execution_host": "elmo",
        "required_public_information_attestation": expected_public_attestation,
        "immutable_anchor": dict(SIMULATOR_FACT_WITNESS_IMMUTABLE_ANCHOR),
    }
    if dict(fact_witness) != expected_witness:
        raise ChecklistProvenanceError("config simulator fact witness contract changed")

    runtime = _mapping(payload.get("runtime"), label="config runtime")
    for name, expected in {
        "enabled_default": False,
        "runtime_wired": False,
        "production_or_inzi_authority": False,
        "guide_or_prose_mechanics_authority": False,
        "unpinned_csv_mechanics_authority": False,
    }.items():
        _require_bool(runtime.get(name), label=f"config runtime.{name}", expected=expected)
    if runtime.get("strict_filter_output_residual_gate") != 0.0:
        raise ChecklistProvenanceError("r298 checklist final residual gate must remain exact zero")

    strict_rows = _mapping(payload.get("strict_channels"), label="config strict_channels")
    expected_strict_rows = {
        channel: {"required_fact_kinds": list(STRICT_REQUIRED_FACT_KINDS[channel])}
        for channel in STRICT_CHANNELS
    }
    if dict(strict_rows) != expected_strict_rows:
        raise ChecklistProvenanceError("config strict channel inventory changed")
    strict_required = dict(STRICT_REQUIRED_FACT_KINDS)

    fact_authority = _mapping(payload.get("fact_authority_by_kind"), label="config fact_authority_by_kind")
    if dict(fact_authority) != dict(FACT_AUTHORITY_BY_KIND):
        raise ChecklistProvenanceError("config fact authority inventory changed")
    normalized_authority: dict[str, str] = {}
    for kind, expected_authority in FACT_AUTHORITY_BY_KIND.items():
        value = _require_string(fact_authority[kind], label=f"config fact authority {kind}")
        if any(token in value.casefold() for token in _FORBIDDEN_AUTHORITY_TOKENS):
            raise ChecklistProvenanceError("guide/prose/CSV cannot become mechanics authority")
        if value != expected_authority:
            raise ChecklistProvenanceError("config fact authority changed")
        normalized_authority[kind] = value

    trace_only = _mapping(payload.get("trace_only_channels"), label="config trace_only_channels")
    if dict(trace_only) != {
        "bench_prize_exposure": 0.0,
        "immediate_disruption_outcome": 0.0,
        "guide_support": 0.0,
    }:
        raise ChecklistProvenanceError("Q5/Q6/guide must remain fixed-zero trace-only")
    q3 = _mapping(payload.get("q3"), label="config q3")
    _require_bool(
        q3.get("bench_only_required"),
        label="config q3.bench_only_required",
        expected=True,
    )
    classifications = _sequence(q3.get("accepted_classifications"), label="config q3 classifications")
    if tuple(classifications) != Q3_CLASSIFICATIONS:
        raise ChecklistProvenanceError("config q3 classifications changed")
    if payload.get("phase_4_assertion_required") != PHASE_4_ASSERTION:
        raise ChecklistProvenanceError("config phase-4 provenance assertion changed")
    return R298ChecklistProvenanceConfig(
        path=identity.path,
        sha256=identity.sha256,
        canonical_goal_sha256=CANONICAL_GOAL_SHA256,
        canonical_contract_sha256=CANONICAL_CONTRACT_SHA256,
        canonical_r241_typed_source_sha256=CANONICAL_R241_TYPED_SOURCE_SHA256,
        canonical_libcg_contract_sha256=CANONICAL_LIBCG_CONTRACT_SHA256,
        canonical_libcg_sha256=CANONICAL_LIBCG_BINARY_SHA256,
        canonical_libcg_member=CANONICAL_LIBCG_MEMBER,
        canonical_libcg_size_bytes=CANONICAL_LIBCG_SIZE_BYTES,
        canonical_kaggle_environments_version=CANONICAL_KAGGLE_ENVIRONMENTS_VERSION,
        catalog_sealer_source_path=expected_sealer["source"],
        catalog_sealer_source_sha256=CATALOG_SEALER_SOURCE_SHA256,
        catalog_sealer_receipt_schema=R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA,
        catalog_schema=R298_PINNED_PUBLIC_CATALOG_SCHEMA,
        catalog_sealer_receipt_filename="receipt.json",
        catalog_sealer_catalog_filename="catalog.json",
        catalog_sealer_fixed_vectors_filename="fixed-vectors.pt",
        catalog_sealer_required_host="elmo",
        catalog_sealer_allowed_hostnames=("elmo", "truenas"),
        catalog_sealer_resource_peak_schema=(
            "poke_bot.alakazam_elmo_rule_derivative_resource_peak/v1"
        ),
        catalog_sealer_resource_peak_required_fields=(
            "experiment_ram_measurement_method",
            "experiment_ram_peak_bytes",
            "cpu_process_or_utilization_peak",
            "gpu_memory_peak_bytes_per_device_or_not_applicable",
            "gpu_utilization_peak_per_device_or_not_applicable",
        ),
        catalog_sealer_csv_identity_sha256=CATALOG_SEALER_CSV_IDENTITY_SHA256,
        catalog_sealer_immutable_artifact_pins=MappingProxyType(
            dict(CATALOG_SEALER_IMMUTABLE_ARTIFACT_PINS)
        ),
        predecessor_goal_revision=PREDECESSOR_GOAL_REVISION,
        predecessor_gateway_sha256=PREDECESSOR_GATEWAY_SHA256,
        predecessor_contract_sha256=PREDECESSOR_CONTRACT_SHA256,
        revision_5_consumer_migration_schema=R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA,
        revision_5_consumer_migration_anchor=MappingProxyType(
            dict(REV5_CONSUMER_MIGRATION_IMMUTABLE_ANCHOR)
        ),
        revision_5_consumer_migration_required_assertions=(
            REV5_CONSUMER_MIGRATION_ASSERTIONS
        ),
        simulator_fact_witness_schema=R298_SIMULATOR_FACT_WITNESS_RECEIPT_SCHEMA,
        engine_evidence_schema=R298_ENGINE_EVIDENCE_SCHEMA,
        simulator_fact_witness_status="passed_elmo_only_nonproduction",
        simulator_fact_witness_execution_kind="pinned_libcg_r236_public_frame",
        simulator_fact_witness_required_host="elmo",
        simulator_fact_witness_public_information_attestation=MappingProxyType(
            expected_public_attestation
        ),
        simulator_fact_witness_immutable_anchor=MappingProxyType(
            dict(SIMULATOR_FACT_WITNESS_IMMUTABLE_ANCHOR)
        ),
        strict_required_fact_kinds=MappingProxyType(strict_required),
        fact_authority_by_kind=MappingProxyType(normalized_authority),
        q3_classifications=tuple(classifications),
    )


def _canonical_libcg_payload(config: R298ChecklistProvenanceConfig) -> dict[str, Any]:
    return {
        "canonical_libcg_contract_sha256": config.canonical_libcg_contract_sha256,
        "sha256": config.canonical_libcg_sha256,
        "member": config.canonical_libcg_member,
        "size_bytes": config.canonical_libcg_size_bytes,
        "kaggle_environments_version": config.canonical_kaggle_environments_version,
    }


def _sealer_canonical_digest(payload: Mapping[str, Any]) -> str:
    """Match ``seal_alakazam_public_catalog_r298.py``'s receipt encoding."""

    return "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_finite_number(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ChecklistProvenanceError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ChecklistProvenanceError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ChecklistProvenanceError(f"{label} must be finite and >= {minimum}")
    return result


@dataclass(frozen=True)
class SealedPublicCatalogEvidenceR298:
    """A validated revision-4 catalog retained as revision-5 predecessor evidence."""

    sealer_receipt: FileIdentityR298
    sealer_receipt_payload_sha256: str
    catalog: FileIdentityR298
    fixed_vectors: FileIdentityR298
    catalog_semantic_sha256: str
    execution_host: str
    predecessor_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sealer_receipt": self.sealer_receipt.to_dict(),
            "sealer_receipt_payload_sha256": self.sealer_receipt_payload_sha256,
            "catalog": self.catalog.to_dict(),
            "fixed_vectors": self.fixed_vectors.to_dict(),
            "catalog_semantic_sha256": self.catalog_semantic_sha256,
            "execution_host": self.execution_host,
            "predecessor_only": bool(self.predecessor_only),
            "revision_5_mechanics_authority": False,
        }


def _validate_pinned_public_catalog_payload(
    identity: FileIdentityR298,
    *,
    config: R298ChecklistProvenanceConfig,
) -> Mapping[str, Any]:
    """Validate the sealer's structured catalog—not a self-claimed envelope."""

    catalog = _read_json_identity(identity, label="pinned public catalog")
    if catalog.get("schema") != config.catalog_schema:
        raise ChecklistProvenanceError("pinned public catalog schema mismatch")
    if catalog.get("revision") != R298_CHECKLIST_PROVENANCE_REVISION:
        raise ChecklistProvenanceError("pinned public catalog revision mismatch")
    if catalog.get("status") != "sealed_public_simulator_catalog":
        raise ChecklistProvenanceError("pinned public catalog is not sealed")
    authority = _mapping(catalog.get("authority"), label="pinned public catalog authority")
    expected_authority = {
        "mechanics": "pinned_official_libcg_r236",
        "csv_role": "provenance_checked_identity_join_only",
        "card_text_rules_parser": False,
        "card2vec_preserved_unchanged": True,
        "legacy_text_derived_embeddings_preserved_but_not_rule_authority": True,
        "new_residual_vector_source": "structured_engine_fields_only",
        "policy_or_runtime_authority": False,
    }
    if dict(authority) != expected_authority:
        raise ChecklistProvenanceError("pinned public catalog authority mismatch")
    source = _mapping(catalog.get("source"), label="pinned public catalog source")
    expected_source = {
        "goal_sha256": config.predecessor_gateway_sha256,
        "contract_sha256": config.predecessor_contract_sha256,
        "libcg_sha256": config.canonical_libcg_sha256,
        "libcg_size_bytes": config.canonical_libcg_size_bytes,
        "csv_sha256": config.catalog_sealer_csv_identity_sha256,
        "sealer_source_sha256": config.catalog_sealer_source_sha256,
    }
    if dict(source) != expected_source:
        raise ChecklistProvenanceError(
            "historical predecessor catalog source binding mismatch"
        )
    _mapping(catalog.get("provenance"), label="pinned public catalog provenance")
    cards = _sequence(catalog.get("cards"), label="pinned public catalog cards")
    attacks = _sequence(catalog.get("attacks"), label="pinned public catalog attacks")
    if not cards or not attacks:
        raise ChecklistProvenanceError("pinned public catalog must contain engine cards and attacks")
    return catalog


def _validate_catalog_sealer_resource_receipt(
    row: Mapping[str, Any],
    *,
    config: R298ChecklistProvenanceConfig,
) -> str:
    expected_keys = {
        "schema",
        "host",
        "started_at_utc",
        "elapsed_seconds",
        "experiment_ram_measurement_method",
        "experiment_ram_peak_bytes",
        "aggregate_experiment_ram_limit_bytes",
        "cpu_process_or_utilization_peak",
        "gpu_memory_peak_bytes_per_device_or_not_applicable",
        "gpu_utilization_peak_per_device_or_not_applicable",
        "zfs_arc_or_l2arc_tuning_performed",
    }
    if set(row) != expected_keys:
        raise ChecklistProvenanceError("catalog sealer resource receipt fields changed")
    if row.get("schema") != config.catalog_sealer_resource_peak_schema:
        raise ChecklistProvenanceError("catalog sealer resource receipt schema mismatch")
    host = _require_string(row.get("host"), label="catalog sealer resource host")
    if host not in config.catalog_sealer_allowed_hostnames:
        raise ChecklistProvenanceError("catalog sealer host is not an allowed Elmo worker")
    _require_string(row.get("started_at_utc"), label="catalog sealer started_at_utc")
    _require_finite_number(row.get("elapsed_seconds"), label="catalog sealer elapsed_seconds")
    method = _require_string(
        row.get("experiment_ram_measurement_method"),
        label="catalog sealer experiment RAM measurement method",
    )
    if method != "linux_getrusage_ru_maxrss_single_isolated_process_no_children":
        raise ChecklistProvenanceError("catalog sealer RAM measurement method mismatch")
    peak = _require_exact_int(
        row.get("experiment_ram_peak_bytes"), label="catalog sealer experiment RAM peak"
    )
    if peak > R298_EXPERIMENT_RAM_BYTES:
        raise ChecklistProvenanceError("catalog sealer exceeded the 96-GiB experiment RAM ceiling")
    if row.get("aggregate_experiment_ram_limit_bytes") != R298_EXPERIMENT_RAM_BYTES:
        raise ChecklistProvenanceError("catalog sealer resource envelope mismatch")
    cpu = _mapping(
        row.get("cpu_process_or_utilization_peak"), label="catalog sealer CPU/process peak"
    )
    if set(cpu) != {"measurement", "process_count_peak"}:
        raise ChecklistProvenanceError("catalog sealer CPU/process peak fields changed")
    if cpu.get("measurement") != "single_sealing_process":
        raise ChecklistProvenanceError("catalog sealer CPU/process measurement mismatch")
    if _require_exact_int(cpu.get("process_count_peak"), label="catalog sealer process peak") != 1:
        raise ChecklistProvenanceError("catalog sealer must be a single isolated process")
    for name in config.catalog_sealer_resource_peak_required_fields[3:]:
        if row.get(name) != "not_applicable_cpu_catalog_seal":
            raise ChecklistProvenanceError(f"catalog sealer {name} mismatch")
    _require_bool(
        row.get("zfs_arc_or_l2arc_tuning_performed"),
        label="catalog sealer zfs tuning",
        expected=False,
    )
    return host


def validate_sealed_public_catalog_r298(
    sealer_receipt_path: Path | str,
    *,
    config_path: Path | str | None = None,
) -> SealedPublicCatalogEvidenceR298:
    """Reopen the actual revision-4 catalog receipt as predecessor evidence.

    A catalog JSON is never authority by itself.  This validates the original
    create-only r4 sealer receipt, its exact source identity, sibling
    catalog/vector file hashes and sizes, and Elmo resource evidence.  The
    returned artifact remains historical evidence only; a separate pinned
    revision-5 consumer-migration receipt is required before it can be used by
    Q1/Q2/Q8 provenance.
    """

    config = load_checklist_provenance_config_r298(config_path)
    receipt_identity = file_identity_r298(sealer_receipt_path, label="catalog sealer receipt")
    if Path(receipt_identity.path).name != config.catalog_sealer_receipt_filename:
        raise ChecklistProvenanceError("catalog sealer receipt filename is not canonical")
    pins = config.catalog_sealer_immutable_artifact_pins
    if receipt_identity.sha256 != pins["receipt_file_sha256"]:
        raise ChecklistProvenanceError(
            "catalog sealer receipt is not the immutable r298 sealed artifact"
        )
    if receipt_identity.size_bytes != pins["receipt_file_size_bytes"]:
        raise ChecklistProvenanceError(
            "catalog sealer receipt size is not the immutable r298 sealed artifact"
        )
    receipt = _read_json_identity(receipt_identity, label="catalog sealer receipt")
    expected_keys = {
        "schema",
        "revision",
        "status",
        "execution_host_role",
        "execution_hostname",
        "created_at_utc",
        "goal_sha256",
        "contract_sha256",
        "libcg_sha256",
        "libcg_size_bytes",
        "csv_sha256",
        "catalog_schema",
        "catalog_semantic_sha256",
        "catalog_file_sha256",
        "catalog_file_size_bytes",
        "fixed_vectors_file_sha256",
        "fixed_vectors_file_size_bytes",
        "metadata_provenance",
        "facts",
        "resource_peak_receipt",
        "authority",
        "receipt_payload_sha256",
    }
    if set(receipt) != expected_keys:
        raise ChecklistProvenanceError("catalog sealer receipt fields changed")
    if receipt.get("schema") != config.catalog_sealer_receipt_schema:
        raise ChecklistProvenanceError("catalog sealer receipt schema mismatch")
    if receipt.get("revision") != R298_CHECKLIST_PROVENANCE_REVISION:
        raise ChecklistProvenanceError("catalog sealer receipt revision mismatch")
    if receipt.get("status") != "passed_elmo_only_nonproduction":
        raise ChecklistProvenanceError("catalog sealer receipt did not pass on Elmo")
    if receipt.get("execution_host_role") != config.catalog_sealer_required_host:
        raise ChecklistProvenanceError("catalog sealer logical execution role is not Elmo")
    hostname = _require_string(
        receipt.get("execution_hostname"), label="catalog sealer execution hostname"
    )
    if hostname not in config.catalog_sealer_allowed_hostnames:
        raise ChecklistProvenanceError("catalog sealer physical hostname is not an allowed Elmo worker")
    _require_string(receipt.get("created_at_utc"), label="catalog sealer created_at_utc")
    for name, expected in {
        "goal_sha256": config.predecessor_gateway_sha256,
        "contract_sha256": config.predecessor_contract_sha256,
        "libcg_sha256": config.canonical_libcg_sha256,
        "libcg_size_bytes": config.canonical_libcg_size_bytes,
        "csv_sha256": config.catalog_sealer_csv_identity_sha256,
        "catalog_schema": config.catalog_schema,
    }.items():
        if receipt.get(name) != expected:
            raise ChecklistProvenanceError(f"catalog sealer receipt {name} mismatch")
    if receipt.get("receipt_payload_sha256") != _sealer_canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    ):
        raise ChecklistProvenanceError("catalog sealer receipt self-digest mismatch")
    if receipt.get("receipt_payload_sha256") != pins["receipt_payload_sha256"]:
        raise ChecklistProvenanceError("catalog sealer receipt payload is not the pinned artifact")
    for name in (
        "catalog_file_sha256",
        "catalog_file_size_bytes",
        "catalog_semantic_sha256",
        "fixed_vectors_file_sha256",
        "fixed_vectors_file_size_bytes",
    ):
        if receipt.get(name) != pins[name]:
            raise ChecklistProvenanceError(f"catalog sealer receipt {name} is not pinned")
    _mapping(receipt.get("metadata_provenance"), label="catalog sealer metadata provenance")
    facts = _mapping(receipt.get("facts"), label="catalog sealer facts")
    expected_fact_flags = {
        "mechanics_from_pinned_engine": True,
        "csv_used_as_mechanics_authority": False,
        "text_hash_used_as_rules_parser": False,
        "text_or_name_hash_in_new_residual_vectors": False,
        "card2vec_preserved_unchanged": True,
    }
    for name, expected in expected_fact_flags.items():
        _require_bool(facts.get(name), label=f"catalog sealer facts.{name}", expected=expected)
    _require_exact_int(facts.get("card_count"), label="catalog sealer card count", minimum=1)
    _require_exact_int(facts.get("attack_count"), label="catalog sealer attack count", minimum=1)
    _require_sha256(
        facts.get("structured_rule_vectors_sha256"), label="catalog sealer structured vectors digest"
    )
    if facts.get("structured_rule_vectors_sha256") != pins["structured_rule_vectors_sha256"]:
        raise ChecklistProvenanceError("catalog sealer structured vectors are not pinned")
    authority = _mapping(receipt.get("authority"), label="catalog sealer authority")
    if dict(authority) != {
        "elmo_only": True,
        "create_only": True,
        "training_eligibility_by_itself": False,
        "production_or_inzi": False,
        "runtime_activation": False,
    }:
        raise ChecklistProvenanceError("catalog sealer authority boundary mismatch")
    host = _validate_catalog_sealer_resource_receipt(
        _mapping(receipt.get("resource_peak_receipt"), label="catalog sealer resource receipt"),
        config=config,
    )
    if host != hostname:
        raise ChecklistProvenanceError("catalog sealer receipt/resource physical host mismatch")
    root = Path(receipt_identity.path).parent
    ready = file_identity_r298(root / "READY.json", label="catalog sealer ready marker")
    if (
        ready.sha256 != pins["ready_file_sha256"]
        or ready.size_bytes != pins["ready_file_size_bytes"]
    ):
        raise ChecklistProvenanceError("catalog sealer ready marker is not the immutable r298 artifact")
    ready_payload = _read_json_identity(ready, label="catalog sealer ready marker")
    expected_ready = {
        "schema": "poke_bot.alakazam_public_catalog_r298_ready/v1",
        "status": "ready_for_r298_schema_freeze_and_refeaturization",
        "catalog_file_sha256": pins["catalog_file_sha256"],
        "vectors_file_sha256": pins["fixed_vectors_file_sha256"],
        "receipt_file_sha256": pins["receipt_file_sha256"],
        "receipt_payload_sha256": pins["receipt_payload_sha256"],
        "training_or_runtime_authority": False,
    }
    if dict(ready_payload) != expected_ready:
        raise ChecklistProvenanceError("catalog sealer ready marker payload mismatch")
    catalog = file_identity_r298(root / config.catalog_sealer_catalog_filename, label="sealed catalog")
    vectors = file_identity_r298(
        root / config.catalog_sealer_fixed_vectors_filename, label="sealed catalog vectors"
    )
    if catalog.sha256 != receipt.get("catalog_file_sha256") or catalog.size_bytes != receipt.get("catalog_file_size_bytes"):
        raise ChecklistProvenanceError("catalog sealer catalog file identity mismatch")
    if vectors.sha256 != receipt.get("fixed_vectors_file_sha256") or vectors.size_bytes != receipt.get("fixed_vectors_file_size_bytes"):
        raise ChecklistProvenanceError("catalog sealer vector file identity mismatch")
    if (
        catalog.sha256 != pins["catalog_file_sha256"]
        or catalog.size_bytes != pins["catalog_file_size_bytes"]
    ):
        raise ChecklistProvenanceError("catalog output is not the immutable r298 sealed artifact")
    if (
        vectors.sha256 != pins["fixed_vectors_file_sha256"]
        or vectors.size_bytes != pins["fixed_vectors_file_size_bytes"]
    ):
        raise ChecklistProvenanceError("catalog vectors are not the immutable r298 sealed artifact")
    catalog_payload = _validate_pinned_public_catalog_payload(catalog, config=config)
    semantic = _sealer_canonical_digest(catalog_payload)
    if receipt.get("catalog_semantic_sha256") != semantic:
        raise ChecklistProvenanceError("catalog sealer catalog semantic digest mismatch")
    if semantic != pins["catalog_semantic_sha256"]:
        raise ChecklistProvenanceError("catalog semantic payload is not the immutable r298 artifact")
    metadata = _mapping(receipt.get("metadata_provenance"), label="catalog sealer metadata provenance")
    provenance = _mapping(catalog_payload.get("provenance"), label="pinned public catalog provenance")
    if dict(metadata) != dict(provenance):
        raise ChecklistProvenanceError("catalog sealer metadata/catalog provenance mismatch")
    for name in ("engine_cards_sha256", "engine_attacks_sha256"):
        if provenance.get(name) != pins[name]:
            raise ChecklistProvenanceError(f"catalog provenance {name} is not pinned")
    return SealedPublicCatalogEvidenceR298(
        sealer_receipt=receipt_identity,
        sealer_receipt_payload_sha256=_require_sha256(
            receipt.get("receipt_payload_sha256"), label="catalog sealer receipt digest"
        ),
        catalog=catalog,
        fixed_vectors=vectors,
        catalog_semantic_sha256=semantic,
        execution_host=host,
        predecessor_only=True,
    )


def validate_pinned_public_catalog_r298(
    sealer_receipt_path: Path | str,
    *,
    revision_5_consumer_migration_receipt_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> FileIdentityR298:
    """Return a catalog only after an explicit pinned r5 consumer migration.

    Calling this legacy-looking convenience name without the r5 migration is
    deliberately an error, rather than an ambiguous proof of mechanics
    authority.  Use :func:`validate_sealed_public_catalog_r298` only when the
    caller needs to inspect the revision-4 predecessor as historical evidence.
    """

    catalog = validate_sealed_public_catalog_r298(
        sealer_receipt_path, config_path=config_path
    )
    if revision_5_consumer_migration_receipt_path is None:
        raise ChecklistProvenanceError(
            "revision-4 catalog is predecessor evidence only; an explicit revision-5 consumer migration receipt is required"
        )
    migration = validate_revision_5_consumer_migration_r298(
        revision_5_consumer_migration_receipt_path, config_path=config_path
    )
    if (
        migration.predecessor_catalog_sealer_receipt_sha256 != catalog.sealer_receipt.sha256
        or migration.predecessor_catalog_sha256 != catalog.catalog.sha256
        or migration.predecessor_fixed_vectors_sha256 != catalog.fixed_vectors.sha256
    ):
        raise ChecklistProvenanceError(
            "revision-5 consumer migration does not bind the requested predecessor catalog"
        )
    return catalog.catalog


@dataclass(frozen=True)
class Revision5ConsumerMigrationEvidenceR298:
    """The explicit r5 rebind that may consume an r4 catalog predecessor."""

    receipt: FileIdentityR298
    receipt_payload_sha256: str
    predecessor_catalog_sealer_receipt_sha256: str
    predecessor_catalog_sha256: str
    predecessor_fixed_vectors_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "receipt_payload_sha256": self.receipt_payload_sha256,
            "predecessor_catalog_sealer_receipt_sha256": (
                self.predecessor_catalog_sealer_receipt_sha256
            ),
            "predecessor_catalog_sha256": self.predecessor_catalog_sha256,
            "predecessor_fixed_vectors_sha256": self.predecessor_fixed_vectors_sha256,
            "predecessor_only_catalog_rebound_by_explicit_revision_5_consumer_validation": True,
        }


def _validate_revision_5_consumer_migration_payload(
    identity: FileIdentityR298,
    *,
    config: R298ChecklistProvenanceConfig,
) -> Revision5ConsumerMigrationEvidenceR298:
    """Validate an owner-pinned r5 receipt instead of trusting its claims.

    The contract deliberately has no self-referential receipt digest field.
    The checked config holds both the immutable file digest and the canonical
    payload digest that a later consuming/schema-freeze receipt must bind.
    """

    anchor = config.revision_5_consumer_migration_anchor
    if anchor.get("status") != "pinned_immutable_artifact":
        raise ChecklistProvenanceError(
            "no immutable revision-5 consumer migration receipt is issued"
        )
    for field in ("receipt_file_sha256", "receipt_payload_sha256"):
        _require_sha256(anchor.get(field), label=f"revision-5 migration anchor.{field}")
    if identity.sha256 != anchor["receipt_file_sha256"]:
        raise ChecklistProvenanceError(
            "revision-5 consumer migration receipt is not the immutable pinned artifact"
        )
    receipt = _read_json_identity(identity, label="revision-5 consumer migration receipt")
    expected_keys = {
        "schema",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "r241_typed_source_path",
        "r241_typed_source_sha256",
        "predecessor_goal_revision",
        "predecessor_gateway_sha256",
        "predecessor_contract_sha256",
        "predecessor_catalog_sealer_receipt_sha256",
        "predecessor_catalog_sha256",
        "predecessor_fixed_vectors_sha256",
        "catalog_classification",
        "blind_goal_or_contract_hash_substitution_allowed",
        "revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze",
        "consumer_rebind_assertions",
        "default_zero_and_inert",
        "runtime_wired",
        "production_or_inzi_authority",
        "validated_at_utc",
    }
    if set(receipt) != expected_keys:
        raise ChecklistProvenanceError("revision-5 consumer migration receipt fields changed")
    if receipt.get("schema") != config.revision_5_consumer_migration_schema:
        raise ChecklistProvenanceError("revision-5 consumer migration receipt schema mismatch")
    if receipt.get("goal_contract_path") != str(CANONICAL_CONTRACT_PATH.relative_to(PROJECT_ROOT)):
        raise ChecklistProvenanceError("revision-5 consumer migration contract path mismatch")
    if receipt.get("goal_contract_sha256") != config.canonical_contract_sha256:
        raise ChecklistProvenanceError("revision-5 consumer migration contract binding mismatch")
    if receipt.get("goal_revision") != 5 or receipt.get("root_handoff_revision") != 303:
        raise ChecklistProvenanceError("revision-5 consumer migration revision binding mismatch")
    if receipt.get("r241_typed_source_path") != str(
        CANONICAL_R241_TYPED_SOURCE_PATH.relative_to(PROJECT_ROOT)
    ):
        raise ChecklistProvenanceError("revision-5 consumer migration r241 path mismatch")
    if receipt.get("r241_typed_source_sha256") != config.canonical_r241_typed_source_sha256:
        raise ChecklistProvenanceError("revision-5 consumer migration r241 binding mismatch")
    for field, expected in {
        "predecessor_goal_revision": config.predecessor_goal_revision,
        "predecessor_gateway_sha256": config.predecessor_gateway_sha256,
        "predecessor_contract_sha256": config.predecessor_contract_sha256,
        "predecessor_catalog_sealer_receipt_sha256": (
            config.catalog_sealer_immutable_artifact_pins["receipt_file_sha256"]
        ),
        "predecessor_catalog_sha256": (
            config.catalog_sealer_immutable_artifact_pins["catalog_file_sha256"]
        ),
        "predecessor_fixed_vectors_sha256": (
            config.catalog_sealer_immutable_artifact_pins["fixed_vectors_file_sha256"]
        ),
    }.items():
        if receipt.get(field) != expected:
            raise ChecklistProvenanceError(
                f"revision-5 consumer migration {field} mismatch"
            )
    if receipt.get("catalog_classification") != (
        "immutable_revision_4_predecessor_evidence_not_revision_5_authority_without_this_receipt"
    ):
        raise ChecklistProvenanceError(
            "revision-5 consumer migration catalog classification mismatch"
        )
    _require_bool(
        receipt.get("blind_goal_or_contract_hash_substitution_allowed"),
        label="revision-5 consumer migration blind substitution",
        expected=False,
    )
    _require_bool(
        receipt.get("revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze"),
        label="revision-5 consumer migration predecessor schema-freeze status",
        expected=False,
    )
    assertions = _mapping(
        receipt.get("consumer_rebind_assertions"),
        label="revision-5 consumer migration assertions",
    )
    if set(assertions) != set(config.revision_5_consumer_migration_required_assertions):
        raise ChecklistProvenanceError("revision-5 consumer migration assertion inventory changed")
    for name in config.revision_5_consumer_migration_required_assertions:
        _require_bool(
            assertions.get(name),
            label=f"revision-5 consumer migration assertion {name}",
            expected=True,
        )
    for field in ("default_zero_and_inert", "runtime_wired", "production_or_inzi_authority"):
        _require_bool(
            receipt.get(field),
            label=f"revision-5 consumer migration {field}",
            expected=False if field != "default_zero_and_inert" else True,
        )
    _require_string(receipt.get("validated_at_utc"), label="revision-5 consumer migration timestamp")
    payload_digest = canonical_digest_r298(receipt)
    if payload_digest != anchor["receipt_payload_sha256"]:
        raise ChecklistProvenanceError(
            "revision-5 consumer migration receipt payload is not pinned"
        )
    return Revision5ConsumerMigrationEvidenceR298(
        receipt=identity,
        receipt_payload_sha256=payload_digest,
        predecessor_catalog_sealer_receipt_sha256=str(
            config.catalog_sealer_immutable_artifact_pins["receipt_file_sha256"]
        ),
        predecessor_catalog_sha256=str(
            config.catalog_sealer_immutable_artifact_pins["catalog_file_sha256"]
        ),
        predecessor_fixed_vectors_sha256=str(
            config.catalog_sealer_immutable_artifact_pins["fixed_vectors_file_sha256"]
        ),
    )


def validate_revision_5_consumer_migration_r298(
    migration_receipt_path: Path | str,
    *,
    config_path: Path | str | None = None,
) -> Revision5ConsumerMigrationEvidenceR298:
    """Validate the sole explicit r5 rebind for the historical r4 catalog.

    It intentionally fails until an owner-pinned receipt is frozen in the
    checked config.  A valid-looking local migration JSON is never enough.
    """

    config = load_checklist_provenance_config_r298(config_path)
    identity = file_identity_r298(
        migration_receipt_path, label="revision-5 consumer migration receipt"
    )
    return _validate_revision_5_consumer_migration_payload(identity, config=config)


def revision_5_consumer_migration_schema_manifest_r298(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Describe the non-circular r5 predecessor-consumer migration boundary.

    This is a schema description, not a receipt builder.  The owner-issued
    receipt deliberately binds canonical r5 sources and exact predecessor
    bytes, but not a mutable candidate/final config SHA.  After its immutable
    identity is pinned in this config, the *separate* schema-freeze receipt
    binds the resulting final config/schema digest and zero-bypass result.
    """

    config = load_checklist_provenance_config_r298(config_path)
    receipt_fields = [
        "schema",
        "goal_contract_path",
        "goal_contract_sha256",
        "goal_revision",
        "root_handoff_revision",
        "r241_typed_source_path",
        "r241_typed_source_sha256",
        "predecessor_goal_revision",
        "predecessor_gateway_sha256",
        "predecessor_contract_sha256",
        "predecessor_catalog_sealer_receipt_sha256",
        "predecessor_catalog_sha256",
        "predecessor_fixed_vectors_sha256",
        "catalog_classification",
        "blind_goal_or_contract_hash_substitution_allowed",
        "revision_4_catalog_receipt_alone_satisfies_revision_5_schema_freeze",
        "consumer_rebind_assertions",
        "default_zero_and_inert",
        "runtime_wired",
        "production_or_inzi_authority",
        "validated_at_utc",
    ]
    definition = {
        "schema": R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA,
        "consumer_goal_revision": 5,
        "consumer_root_handoff_revision": 303,
        "canonical_goal_sha256": config.canonical_goal_sha256,
        "canonical_contract_sha256": config.canonical_contract_sha256,
        "canonical_r241_typed_source_sha256": config.canonical_r241_typed_source_sha256,
        "canonical_libcg_contract_sha256": config.canonical_libcg_contract_sha256,
        "predecessor_goal_revision": config.predecessor_goal_revision,
        "predecessor_gateway_sha256": config.predecessor_gateway_sha256,
        "predecessor_contract_sha256": config.predecessor_contract_sha256,
        "predecessor_catalog_sealer_receipt_sha256": (
            config.catalog_sealer_immutable_artifact_pins["receipt_file_sha256"]
        ),
        "predecessor_catalog_sha256": (
            config.catalog_sealer_immutable_artifact_pins["catalog_file_sha256"]
        ),
        "predecessor_fixed_vectors_sha256": (
            config.catalog_sealer_immutable_artifact_pins["fixed_vectors_file_sha256"]
        ),
        "required_receipt_fields": receipt_fields,
        "required_assertions": list(config.revision_5_consumer_migration_required_assertions),
        "receipt_may_bind_candidate_or_final_checklist_config_sha256": False,
        "receipt_may_bind_candidate_or_final_checklist_schema_sha256": False,
        "receipt_may_bind_zero_bypass_receipt_sha256": False,
        "reason": "avoid_config_receipt_hash_cycle",
        "after_receipt_identity_is_pinned": (
            "freeze_final_checklist_config_and_schema_then_bind_both_migration_receipt_and_final_schema_in_separate_zero_bypass_receipt"
        ),
        "current_anchor": dict(config.revision_5_consumer_migration_anchor),
        "runtime_wired": False,
        "production_or_inzi_authority": False,
    }
    return {
        "schema": "poke_bot.alakazam_checklist_provenance_r298_consumer_migration_schema_manifest/v1",
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "definition": definition,
        "schema_sha256": canonical_digest_r298(definition),
    }


def build_public_catalog_receipt_r298(*_: Any, **__: Any) -> None:
    """Refuse local manufacture of a catalog authority receipt.

    The only accepted catalog receipt is the create-only output of
    ``scripts/seal_alakazam_public_catalog_r298.py``.  Keeping this symbol as
    an explicit failure prevents legacy callers from silently turning a JSON
    fixture, guide, or CSV into an apparently sealed authority record.
    """

    raise ChecklistProvenanceError(
        "local catalog receipt construction is forbidden; use the r298 catalog sealer receipt"
    )


def build_simulator_mechanics_receipt_r298(*_: Any, **__: Any) -> None:
    """Refuse local manufacture of simulator mechanics authority.

    Per-frame mechanics must arrive through a separately sealed simulator fact
    witness.  This function intentionally cannot create one.
    """

    raise ChecklistProvenanceError(
        "local simulator mechanics receipt construction is forbidden; use a simulator fact witness"
    )


@dataclass(frozen=True)
class ChecklistChannelProvenanceStatus:
    """Authorization result for one strict mechanics-dependent channel."""

    name: str
    authorized: bool
    reason: str
    required_fact_kinds: tuple[str, ...]
    authorized_fact_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": bool(self.authorized),
            "status": "available" if self.authorized else "unavailable",
            "authorized": bool(self.authorized),
            "reason": self.reason,
            "required_fact_kinds": list(self.required_fact_kinds),
            "authorized_fact_kinds": list(self.authorized_fact_kinds),
        }


@dataclass(frozen=True)
class ChecklistProvenanceValidation:
    """Fail-closed validation result, safe to serialize into a trace record."""

    valid: bool
    global_valid: bool
    reason: str
    config: Optional[R298ChecklistProvenanceConfig]
    channel_statuses: tuple[ChecklistChannelProvenanceStatus, ...]
    sealed_public_catalog: Optional[Mapping[str, Any]] = None
    revision_5_consumer_migration: Optional[Mapping[str, Any]] = None
    simulator_fact_witness: Optional[Mapping[str, Any]] = None

    @property
    def channel_status(self) -> Mapping[str, ChecklistChannelProvenanceStatus]:
        return MappingProxyType({row.name: row for row in self.channel_statuses})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R298_CHECKLIST_PROVENANCE_SCHEMA,
            "revision": R298_CHECKLIST_PROVENANCE_REVISION,
            "valid": bool(self.valid),
            "global_valid": bool(self.global_valid),
            "reason": self.reason,
            "config": None if self.config is None else self.config.to_dict(),
            "sealed_public_catalog": (
                None
                if self.sealed_public_catalog is None
                else dict(self.sealed_public_catalog)
            ),
            "revision_5_consumer_migration": (
                None
                if self.revision_5_consumer_migration is None
                else dict(self.revision_5_consumer_migration)
            ),
            "simulator_fact_witness": (
                None
                if self.simulator_fact_witness is None
                else dict(self.simulator_fact_witness)
            ),
            "channel_status": {
                row.name: row.to_dict() for row in self.channel_statuses
            },
            "phase_4_provenance_assertion": PHASE_4_ASSERTION,
            "runtime_wired": False,
            "final_residual_gate": 0.0,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def _unavailable_validation(
    reason: str,
    *,
    config: Optional[R298ChecklistProvenanceConfig] = None,
) -> ChecklistProvenanceValidation:
    statuses = tuple(
        ChecklistChannelProvenanceStatus(
            name=name,
            authorized=False,
            reason=reason,
            required_fact_kinds=(
                () if config is None else config.strict_required_fact_kinds.get(name, ())
            ),
            authorized_fact_kinds=(),
        )
        for name in STRICT_CHANNELS
    )
    return ChecklistProvenanceValidation(
        valid=False,
        global_valid=False,
        reason=reason,
        config=config,
        channel_statuses=statuses,
    )


def _validate_canonical_libcg_record(
    row: Mapping[str, Any],
    *,
    config: R298ChecklistProvenanceConfig,
    expected_libcg_sha256: Optional[str],
    expected_canonical_libcg_sha256: Optional[str],
) -> None:
    expected = _canonical_libcg_payload(config)
    if dict(row) != expected:
        raise ChecklistProvenanceError("canonical libcg receipt/digest mismatch")
    if expected_libcg_sha256 is not None and expected_libcg_sha256 != config.canonical_libcg_sha256:
        raise ChecklistProvenanceError("caller expected a different libcg binary digest")
    if (
        expected_canonical_libcg_sha256 is not None
        and expected_canonical_libcg_sha256 != config.canonical_libcg_contract_sha256
    ):
        raise ChecklistProvenanceError("caller expected a different canonical libcg receipt digest")


def _validate_digest_binding(
    row: Mapping[str, Any],
    *,
    expected_public_observation_sha256: Optional[str],
    expected_legal_option_order_sha256: Optional[str],
    expected_legal_option_multiset_sha256: Optional[str],
    expected_trace_evidence_sha256: Optional[str],
    expected_option_count: Optional[int],
    config: R298ChecklistProvenanceConfig,
) -> dict[str, Any]:
    expected_keys = {
        "authority",
        "canonical_libcg_sha256",
        "public_observation_sha256",
        "legal_option_order_sha256",
        "legal_option_multiset_sha256",
        "legal_option_count",
    }
    if set(row) != expected_keys:
        raise ChecklistProvenanceError("legal-option binding fields changed")
    if row.get("authority") != "pinned_libcg_emitted_complete_ordered_legal_options":
        raise ChecklistProvenanceError("legal options are not emitted by pinned libcg")
    if row.get("canonical_libcg_sha256") != config.canonical_libcg_sha256:
        raise ChecklistProvenanceError("legal-option binding uses another libcg")
    normalized = {
        "public_observation_sha256": _require_sha256(
            row.get("public_observation_sha256"), label="legal-option public observation digest"
        ),
        "legal_option_order_sha256": _require_sha256(
            row.get("legal_option_order_sha256"), label="legal-option ordered digest"
        ),
        "legal_option_multiset_sha256": _require_sha256(
            row.get("legal_option_multiset_sha256"), label="legal-option multiset digest"
        ),
        "legal_option_count": _require_exact_int(
            row.get("legal_option_count"), label="legal-option count", minimum=0
        ),
    }
    expected_pairs = {
        "public_observation_sha256": expected_public_observation_sha256,
        "legal_option_order_sha256": expected_legal_option_order_sha256,
        "legal_option_multiset_sha256": expected_legal_option_multiset_sha256,
        "legal_option_count": expected_option_count,
    }
    for name, expected_value in expected_pairs.items():
        if expected_value is not None and normalized[name] != expected_value:
            raise ChecklistProvenanceError(f"{name} binding mismatch")
    # The trace digest is checked by the caller's trace_binding.  It is passed
    # through here solely so the function owns all legal-stage expected values.
    if expected_trace_evidence_sha256 is not None:
        _require_sha256(expected_trace_evidence_sha256, label="expected trace evidence digest")
    return normalized


def _validate_trace_binding(
    row: Mapping[str, Any],
    *,
    legal: Mapping[str, Any],
    expected_trace_evidence_sha256: Optional[str],
) -> None:
    expected_keys = {
        "r288_trace_sha256",
        "public_observation_sha256",
        "legal_option_order_sha256",
        "legal_option_multiset_sha256",
        "legal_option_count",
    }
    if set(row) != expected_keys:
        raise ChecklistProvenanceError("trace binding fields changed")
    trace_digest = _require_sha256(row.get("r288_trace_sha256"), label="trace evidence digest")
    if expected_trace_evidence_sha256 is not None and trace_digest != expected_trace_evidence_sha256:
        raise ChecklistProvenanceError("trace evidence digest mismatch")
    for name in (
        "public_observation_sha256",
        "legal_option_order_sha256",
        "legal_option_multiset_sha256",
        "legal_option_count",
    ):
        if row.get(name) != legal.get(name):
            raise ChecklistProvenanceError(f"trace binding {name} mismatch")


def _fact_has_forbidden_authority(row: Mapping[str, Any]) -> bool:
    for field in ("authority", "source", "provenance", "mechanics_authority"):
        value = row.get(field)
        if isinstance(value, str):
            normalized = value.casefold().replace("-", "_")
            if any(token in normalized for token in _FORBIDDEN_AUTHORITY_TOKENS):
                return True
    return False


def _public_fact_value(value: Any, *, label: str) -> None:
    """Reject obvious hidden/future payloads before they become witness facts."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            token = str(key).casefold().replace("-", "_")
            if any(
                forbidden in token
                for forbidden in (
                    "opponent_hand",
                    "opponent_deck",
                    "deck_order",
                    "unrevealed_prize",
                    "hidden_prize",
                    "future_transition",
                    "terminal_result",
                )
            ):
                raise ChecklistProvenanceError(f"{label} contains a forbidden hidden/future field")
            _public_fact_value(nested, label=label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _public_fact_value(nested, label=label)
    else:
        # ``canonical_digest_r298`` below additionally rejects unsupported
        # non-JSON values and NaN/Infinity.
        canonical_digest_r298({"public_fact_value": value})


@dataclass(frozen=True)
class SimulatorFactWitnessEvidenceR298:
    """One file-bound public-frame simulator fact witness."""

    receipt: FileIdentityR298
    receipt_payload_sha256: str
    engine_evidence: FileIdentityR298
    channel_fact_digests: Mapping[str, Mapping[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_dict(),
            "receipt_payload_sha256": self.receipt_payload_sha256,
            "engine_evidence": self.engine_evidence.to_dict(),
            "channel_fact_digests": {
                channel: dict(facts)
                for channel, facts in self.channel_fact_digests.items()
            },
        }


def _validate_engine_evidence_for_witness(
    identity: FileIdentityR298,
    *,
    config: R298ChecklistProvenanceConfig,
) -> Mapping[str, str]:
    """Require an actual pinned-engine evidence artifact, not a fixture flag."""

    payload = _read_json_identity(identity, label="simulator fact witness engine evidence")
    if payload.get("schema") != config.engine_evidence_schema:
        raise ChecklistProvenanceError("engine evidence schema mismatch")
    if payload.get("owner_revision") != R298_CHECKLIST_PROVENANCE_REVISION:
        raise ChecklistProvenanceError("engine evidence revision mismatch")
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise ChecklistProvenanceError("engine evidence did not pass")
    if payload.get("host") != config.simulator_fact_witness_required_host:
        raise ChecklistProvenanceError("engine evidence did not execute on Elmo")
    execution = _mapping(payload.get("execution"), label="engine evidence execution")
    for name, expected in {
        "engine_backed": True,
        "actual_pinned_competition_simulator": True,
        "fixture_only": False,
        "legal_option_list_authoritative": True,
    }.items():
        _require_bool(execution.get(name), label=f"engine evidence {name}", expected=expected)
    if execution.get("mode") != "pinned_competition_simulator":
        raise ChecklistProvenanceError("engine evidence mode is not pinned libcg")
    simulator = _mapping(payload.get("simulator"), label="engine evidence simulator")
    if simulator.get("libcg_sha256") != config.canonical_libcg_sha256:
        raise ChecklistProvenanceError("engine evidence libcg digest mismatch")
    if simulator.get("kaggle_environments_version") != config.canonical_kaggle_environments_version:
        raise ChecklistProvenanceError("engine evidence Kaggle Environments version mismatch")
    cases = _mapping(payload.get("case_results"), label="engine evidence cases")
    witnesses = _mapping(payload.get("case_witnesses"), label="engine evidence case witnesses")
    if not cases or set(cases) != set(witnesses):
        raise ChecklistProvenanceError("engine evidence case/witness inventory mismatch")
    normalized: dict[str, str] = {}
    for name in sorted(cases):
        _require_bool(cases[name], label=f"engine evidence case {name}", expected=True)
        normalized[name] = _require_string(
            witnesses[name], label=f"engine evidence witness {name}"
        )
    summary = _mapping(payload.get("test_summary"), label="engine evidence summary")
    if _require_exact_int(summary.get("failed"), label="engine evidence failures") != 0:
        raise ChecklistProvenanceError("engine evidence has failed cases")
    _require_exact_int(summary.get("passed"), label="engine evidence passes", minimum=1)
    return MappingProxyType(normalized)


def _validate_simulator_fact_witness(
    identity: FileIdentityR298,
    *,
    catalog: SealedPublicCatalogEvidenceR298,
    legal: Mapping[str, Any],
    trace_evidence_sha256: str,
    config: R298ChecklistProvenanceConfig,
) -> SimulatorFactWitnessEvidenceR298:
    """Validate fact values against an actual sealer and engine witness chain."""

    # A self-digested JSON file can establish internal consistency but cannot
    # establish that pinned libcg actually executed the public frame.  The
    # current frozen schema has no owner-issued witness anchor, so every
    # mechanics-dependent channel must remain unavailable.  A future schema
    # revision may add a concrete immutable file/payload identity only through
    # a new frozen config/receipt boundary; callers cannot opt into it here.
    anchor = config.simulator_fact_witness_immutable_anchor
    if anchor.get("status") != "pinned_immutable_artifact":
        raise ChecklistProvenanceError(
            "no immutable owner-pinned simulator fact witness is issued"
        )
    # Kept deliberately unreachable under the current frozen config.  These
    # checks make an explicit future code/config revision prove all four roots
    # rather than weakening the present fail-closed behavior.
    required_anchor_fields = (
        "receipt_file_sha256",
        "receipt_payload_sha256",
        "engine_evidence_file_sha256",
        "engine_evidence_payload_sha256",
    )
    for field in required_anchor_fields:
        _require_sha256(
            anchor.get(field), label=f"simulator fact witness immutable anchor.{field}"
        )
    witness = _read_json_identity(identity, label="simulator fact witness")
    if identity.sha256 != anchor["receipt_file_sha256"]:
        raise ChecklistProvenanceError("simulator fact witness is not the immutable pinned artifact")
    expected_keys = {
        "schema",
        "revision",
        "status",
        "created_at_utc",
        "goal_gateway_sha256",
        "goal_contract_sha256",
        "canonical_libcg",
        "catalog_sealer_receipt",
        "catalog",
        "fixed_vectors",
        "catalog_sealer_receipt_payload_sha256",
        "engine_evidence",
        "execution",
        "public_input_binding",
        "public_information_attestation",
        "channel_facts",
        "phase_4_assertions",
        "guide_or_prose_mechanics_authority",
        "unpinned_csv_mechanics_authority",
        "runtime_wired",
        "witness_payload_sha256",
    }
    if set(witness) != expected_keys:
        raise ChecklistProvenanceError("simulator fact witness fields changed")
    if witness.get("schema") != config.simulator_fact_witness_schema:
        raise ChecklistProvenanceError("simulator fact witness schema mismatch")
    if witness.get("revision") != R298_CHECKLIST_PROVENANCE_REVISION:
        raise ChecklistProvenanceError("simulator fact witness revision mismatch")
    if witness.get("status") != config.simulator_fact_witness_status:
        raise ChecklistProvenanceError("simulator fact witness status mismatch")
    _require_string(witness.get("created_at_utc"), label="simulator fact witness created_at_utc")
    if witness.get("goal_gateway_sha256") != config.canonical_goal_sha256:
        raise ChecklistProvenanceError("simulator fact witness goal binding mismatch")
    if witness.get("goal_contract_sha256") != config.canonical_contract_sha256:
        raise ChecklistProvenanceError("simulator fact witness contract binding mismatch")
    if witness.get("canonical_libcg") != _canonical_libcg_payload(config):
        raise ChecklistProvenanceError("simulator fact witness libcg binding mismatch")
    if _identity_from_mapping(
        witness.get("catalog_sealer_receipt"), label="fact witness catalog sealer receipt"
    ) != catalog.sealer_receipt:
        raise ChecklistProvenanceError("simulator fact witness sealer receipt binding mismatch")
    if _identity_from_mapping(witness.get("catalog"), label="fact witness catalog") != catalog.catalog:
        raise ChecklistProvenanceError("simulator fact witness catalog binding mismatch")
    if _identity_from_mapping(witness.get("fixed_vectors"), label="fact witness vectors") != catalog.fixed_vectors:
        raise ChecklistProvenanceError("simulator fact witness vector binding mismatch")
    if witness.get("catalog_sealer_receipt_payload_sha256") != catalog.sealer_receipt_payload_sha256:
        raise ChecklistProvenanceError("simulator fact witness sealer payload binding mismatch")
    engine_identity = _identity_from_mapping(
        witness.get("engine_evidence"), label="fact witness engine evidence"
    )
    if engine_identity.sha256 != anchor["engine_evidence_file_sha256"]:
        raise ChecklistProvenanceError("simulator engine evidence is not the immutable pinned artifact")
    engine_case_witnesses = _validate_engine_evidence_for_witness(
        engine_identity, config=config
    )
    engine_payload = _read_json_identity(
        engine_identity, label="simulator fact witness engine evidence"
    )
    if canonical_digest_r298(engine_payload) != anchor["engine_evidence_payload_sha256"]:
        raise ChecklistProvenanceError("simulator engine evidence payload is not pinned")
    execution = _mapping(witness.get("execution"), label="fact witness execution")
    expected_execution = {
        "host": config.simulator_fact_witness_required_host,
        "execution_kind": config.simulator_fact_witness_execution_kind,
        "actual_pinned_libcg_execution": True,
        "legal_options_emitted_by_libcg": True,
    }
    if dict(execution) != expected_execution:
        raise ChecklistProvenanceError("simulator fact witness execution binding mismatch")
    binding = _mapping(witness.get("public_input_binding"), label="fact witness public input")
    expected_binding = {
        "r288_trace_sha256": trace_evidence_sha256,
        "public_observation_sha256": legal["public_observation_sha256"],
        "legal_option_order_sha256": legal["legal_option_order_sha256"],
        "legal_option_multiset_sha256": legal["legal_option_multiset_sha256"],
        "legal_option_count": legal["legal_option_count"],
    }
    if dict(binding) != expected_binding:
        raise ChecklistProvenanceError("simulator fact witness public input binding mismatch")
    public_attestation = _mapping(
        witness.get("public_information_attestation"), label="fact witness public information"
    )
    if dict(public_attestation) != dict(config.simulator_fact_witness_public_information_attestation):
        raise ChecklistProvenanceError("simulator fact witness public-information boundary mismatch")
    assertions = _mapping(witness.get("phase_4_assertions"), label="fact witness phase-4 assertions")
    _require_bool(assertions.get(PHASE_4_ASSERTION), label=f"fact witness {PHASE_4_ASSERTION}", expected=True)
    for field in (
        "guide_or_prose_mechanics_authority",
        "unpinned_csv_mechanics_authority",
        "runtime_wired",
    ):
        _require_bool(witness.get(field), label=f"fact witness {field}", expected=False)
    if witness.get("witness_payload_sha256") != canonical_digest_r298(
        {key: value for key, value in witness.items() if key != "witness_payload_sha256"}
    ):
        raise ChecklistProvenanceError("simulator fact witness self-digest mismatch")
    if witness.get("witness_payload_sha256") != anchor["receipt_payload_sha256"]:
        raise ChecklistProvenanceError("simulator fact witness payload is not pinned")
    raw_channels = _mapping(witness.get("channel_facts"), label="fact witness channel facts")
    if set(raw_channels) != set(STRICT_CHANNELS):
        raise ChecklistProvenanceError("simulator fact witness channel inventory mismatch")
    channel_digests: dict[str, Mapping[str, str]] = {}
    for channel in STRICT_CHANNELS:
        required = config.strict_required_fact_kinds[channel]
        raw_facts = _sequence(raw_channels[channel], label=f"fact witness {channel} facts")
        if len(raw_facts) != len(required):
            raise ChecklistProvenanceError("simulator fact witness fact count mismatch")
        seen: set[str] = set()
        digests: dict[str, str] = {}
        for index, raw_fact in enumerate(raw_facts):
            fact = _mapping(raw_fact, label=f"fact witness {channel} fact[{index}]")
            expected_keys = {
                "kind",
                "authority",
                "value",
                "value_sha256",
                "public_observation_sha256",
                "engine_case_witness",
            }
            kind = _require_string(fact.get("kind"), label=f"fact witness {channel} kind")
            authority = config.fact_authority_by_kind.get(kind)
            if authority is None or kind in seen or _fact_has_forbidden_authority(fact):
                raise ChecklistProvenanceError("simulator fact witness fact authority is invalid")
            if fact.get("authority") != authority:
                raise ChecklistProvenanceError("simulator fact witness fact authority mismatch")
            if fact.get("public_observation_sha256") != legal["public_observation_sha256"]:
                raise ChecklistProvenanceError("simulator fact witness fact public binding mismatch")
            _public_fact_value(fact.get("value"), label=f"fact witness {channel}.{kind}")
            expected_digest = canonical_digest_r298(
                {"channel": channel, "kind": kind, "value": fact.get("value")}
            )
            if fact.get("value_sha256") != expected_digest:
                raise ChecklistProvenanceError("simulator fact witness fact value digest mismatch")
            engine_case = _require_string(
                fact.get("engine_case_witness"), label=f"fact witness {channel} engine case witness"
            )
            if engine_case not in set(engine_case_witnesses.values()):
                raise ChecklistProvenanceError("simulator fact witness lacks an engine case witness")
            if authority == "pinned_libcg_legal_option":
                expected_keys |= {"canonical_libcg_sha256", "legal_option_order_sha256"}
                if fact.get("canonical_libcg_sha256") != config.canonical_libcg_sha256:
                    raise ChecklistProvenanceError("simulator fact witness legal fact libcg mismatch")
                if fact.get("legal_option_order_sha256") != legal["legal_option_order_sha256"]:
                    raise ChecklistProvenanceError("simulator fact witness legal fact options mismatch")
            elif authority == "receipt_bound_public_catalog":
                expected_keys |= {
                    "catalog_sha256",
                    "catalog_sealer_receipt_payload_sha256",
                }
                if fact.get("catalog_sha256") != catalog.catalog.sha256:
                    raise ChecklistProvenanceError("simulator fact witness catalog fact digest mismatch")
                if fact.get("catalog_sealer_receipt_payload_sha256") != catalog.sealer_receipt_payload_sha256:
                    raise ChecklistProvenanceError("simulator fact witness catalog receipt mismatch")
            elif authority == "receipt_bound_simulator_mechanics":
                expected_keys |= {"engine_evidence_sha256"}
                if fact.get("engine_evidence_sha256") != engine_identity.sha256:
                    raise ChecklistProvenanceError("simulator fact witness engine fact binding mismatch")
            elif authority != "public_observation":
                raise ChecklistProvenanceError("simulator fact witness unsupported authority")
            if set(fact) != expected_keys:
                raise ChecklistProvenanceError("simulator fact witness fact fields changed")
            seen.add(kind)
            digests[kind] = expected_digest
        if set(seen) != set(required):
            raise ChecklistProvenanceError("simulator fact witness misses required fact kinds")
        channel_digests[channel] = MappingProxyType(digests)
    return SimulatorFactWitnessEvidenceR298(
        receipt=identity,
        receipt_payload_sha256=_require_sha256(
            witness.get("witness_payload_sha256"), label="simulator fact witness digest"
        ),
        engine_evidence=engine_identity,
        channel_fact_digests=MappingProxyType(channel_digests),
    )


def _validate_channel_evidence(
    channel: str,
    row: Mapping[str, Any],
    *,
    config: R298ChecklistProvenanceConfig,
    fact_witness: SimulatorFactWitnessEvidenceR298,
) -> ChecklistChannelProvenanceStatus:
    required = config.strict_required_fact_kinds[channel]
    try:
        if set(row) != {"status", "witness_fact_digests"}:
            raise ChecklistProvenanceError("channel evidence fields changed")
        if row.get("status") != "authorized":
            raise ChecklistProvenanceError("channel evidence is not authorized")
        digests = _mapping(row.get("witness_fact_digests"), label=f"{channel} witness facts")
        expected = fact_witness.channel_fact_digests.get(channel)
        if expected is None or dict(digests) != dict(expected):
            raise ChecklistProvenanceError("channel evidence does not bind exact simulator witness facts")
        return ChecklistChannelProvenanceStatus(
            name=channel,
            authorized=True,
            reason="pinned_libcg_and_receipt_bound_public_mechanics",
            required_fact_kinds=required,
            authorized_fact_kinds=tuple(required),
        )
    except ChecklistProvenanceError as exc:
        return ChecklistChannelProvenanceStatus(
            name=channel,
            authorized=False,
            reason=f"r298_provenance_unavailable:{exc}",
            required_fact_kinds=required,
            authorized_fact_kinds=(),
        )


def validate_checklist_provenance_r298(
    provenance: Mapping[str, Any] | Any,
    *,
    expected_libcg_sha256: Optional[str] = None,
    expected_canonical_libcg_sha256: Optional[str] = None,
    expected_canonical_libcg_receipt_sha256: Optional[str] = None,
    expected_public_observation_sha256: Optional[str] = None,
    expected_legal_option_order_sha256: Optional[str] = None,
    expected_legal_option_multiset_sha256: Optional[str] = None,
    expected_catalog_sha256: Optional[str] = None,
    expected_simulator_mechanics_receipt_sha256: Optional[str] = None,
    expected_simulator_fact_witness_receipt_sha256: Optional[str] = None,
    expected_trace_evidence_sha256: Optional[str] = None,
    expected_option_count: Optional[int] = None,
    config_path: Path | str | None = None,
) -> ChecklistProvenanceValidation:
    """Validate strict Q1/Q2/Q8 mechanics provenance without raising on input.

    Invalid, stale, missing, forged, guide-backed, CSV-only, or mismatched
    evidence returns an explicit unavailable result.  This behavior lets a
    materializer preserve an eight-channel record while safely emitting zeros.
    """

    try:
        config = load_checklist_provenance_config_r298(config_path)
    except ChecklistProvenanceError as exc:
        return _unavailable_validation(f"r298_provenance_unavailable:{exc}")
    try:
        record = _mapping(provenance, label="checklist provenance")
        expected_record_keys = {
            "schema",
            "revision",
            "mode",
            "goal_gateway_sha256",
            "goal_contract_sha256",
            "config_sha256",
            "canonical_libcg",
            "authority_sources",
            "guide_or_prose_mechanics_authority",
            "unpinned_csv_mechanics_authority",
            "runtime_wired",
            "legal_options",
            "catalog_sealer_receipt",
            "revision_5_consumer_migration_receipt",
            "simulator_fact_witness_receipt",
            "trace_binding",
            "phase_4_assertions",
            "channel_evidence",
        }
        if set(record) != expected_record_keys:
            raise ChecklistProvenanceError("checklist provenance field inventory changed")
        if record.get("schema") != R298_CHECKLIST_PROVENANCE_SCHEMA:
            raise ChecklistProvenanceError("checklist provenance schema mismatch")
        if record.get("revision") != R298_CHECKLIST_PROVENANCE_REVISION:
            raise ChecklistProvenanceError("checklist provenance revision mismatch")
        if record.get("mode") != "strict_pinned_public_mechanics":
            raise ChecklistProvenanceError("checklist provenance mode is not strict")
        if record.get("goal_gateway_sha256") != config.canonical_goal_sha256:
            raise ChecklistProvenanceError("checklist provenance goal binding mismatch")
        if record.get("goal_contract_sha256") != config.canonical_contract_sha256:
            raise ChecklistProvenanceError("checklist provenance contract binding mismatch")
        if record.get("config_sha256") != config.sha256:
            raise ChecklistProvenanceError("checklist provenance frozen config binding mismatch")
        for field in (
            "guide_or_prose_mechanics_authority",
            "unpinned_csv_mechanics_authority",
            "runtime_wired",
        ):
            _require_bool(record.get(field), label=f"checklist provenance {field}", expected=False)
        sources = _sequence(record.get("authority_sources"), label="checklist provenance authority sources")
        expected_sources = {
            "pinned_libcg_legal_options",
            "receipt_bound_public_catalog",
            "receipt_bound_simulator_mechanics",
            "public_observation",
        }
        if set(sources) != expected_sources:
            raise ChecklistProvenanceError("checklist provenance authority source inventory changed")
        if any(
            not isinstance(source, str)
            or any(token in source.casefold() for token in _FORBIDDEN_AUTHORITY_TOKENS)
            for source in sources
        ):
            raise ChecklistProvenanceError("guide/prose/CSV authority source is forbidden")
        canonical_expected = expected_canonical_libcg_sha256
        if expected_canonical_libcg_receipt_sha256 is not None:
            if canonical_expected is not None and canonical_expected != expected_canonical_libcg_receipt_sha256:
                raise ChecklistProvenanceError("caller supplied conflicting canonical libcg expected digests")
            canonical_expected = expected_canonical_libcg_receipt_sha256
        _validate_canonical_libcg_record(
            _mapping(record.get("canonical_libcg"), label="checklist provenance canonical_libcg"),
            config=config,
            expected_libcg_sha256=expected_libcg_sha256,
            expected_canonical_libcg_sha256=canonical_expected,
        )
        legal = _validate_digest_binding(
            _mapping(record.get("legal_options"), label="checklist provenance legal_options"),
            expected_public_observation_sha256=expected_public_observation_sha256,
            expected_legal_option_order_sha256=expected_legal_option_order_sha256,
            expected_legal_option_multiset_sha256=expected_legal_option_multiset_sha256,
            expected_trace_evidence_sha256=expected_trace_evidence_sha256,
            expected_option_count=expected_option_count,
            config=config,
        )
        _validate_trace_binding(
            _mapping(record.get("trace_binding"), label="checklist provenance trace binding"),
            legal=legal,
            expected_trace_evidence_sha256=expected_trace_evidence_sha256,
        )
        trace_digest = _require_sha256(
            _mapping(record["trace_binding"], label="checklist provenance trace binding").get(
                "r288_trace_sha256"
            ),
            label="checklist provenance trace digest",
        )
        catalog_identity = _identity_from_mapping(
            record.get("catalog_sealer_receipt"),
            label="checklist provenance catalog sealer receipt",
        )
        migration_identity = _identity_from_mapping(
            record.get("revision_5_consumer_migration_receipt"),
            label="checklist provenance revision-5 consumer migration receipt",
        )
        migration = _validate_revision_5_consumer_migration_payload(
            migration_identity,
            config=config,
        )
        catalog = validate_sealed_public_catalog_r298(catalog_identity.path, config_path=config_path)
        if catalog.sealer_receipt != catalog_identity:
            raise ChecklistProvenanceError("catalog sealer receipt file identity mismatch")
        if not catalog.predecessor_only:
            raise ChecklistProvenanceError("catalog must be classified as revision-4 predecessor evidence")
        if (
            migration.predecessor_catalog_sealer_receipt_sha256
            != catalog.sealer_receipt.sha256
            or migration.predecessor_catalog_sha256 != catalog.catalog.sha256
            or migration.predecessor_fixed_vectors_sha256 != catalog.fixed_vectors.sha256
        ):
            raise ChecklistProvenanceError(
                "revision-5 consumer migration does not bind the consumed predecessor catalog"
            )
        if (
            expected_catalog_sha256 is not None
            and catalog.catalog.sha256 != expected_catalog_sha256
        ):
            raise ChecklistProvenanceError("public catalog digest mismatch")
        witness_identity = _identity_from_mapping(
            record.get("simulator_fact_witness_receipt"),
            label="checklist provenance simulator fact witness receipt",
        )
        witness = _validate_simulator_fact_witness(
            witness_identity,
            catalog=catalog,
            legal=legal,
            trace_evidence_sha256=trace_digest,
            config=config,
        )
        expected_witness = (
            expected_simulator_fact_witness_receipt_sha256
            if expected_simulator_fact_witness_receipt_sha256 is not None
            else expected_simulator_mechanics_receipt_sha256
        )
        if expected_witness is not None and expected_witness not in {
            witness.receipt.sha256,
            witness.receipt_payload_sha256,
        }:
            raise ChecklistProvenanceError("simulator fact witness receipt digest mismatch")
        assertions = _mapping(record.get("phase_4_assertions"), label="checklist provenance phase-4 assertions")
        _require_bool(assertions.get(PHASE_4_ASSERTION), label=f"checklist provenance {PHASE_4_ASSERTION}", expected=True)
        evidence = _mapping(record.get("channel_evidence"), label="checklist provenance channel evidence")
        if set(evidence) != set(STRICT_CHANNELS):
            raise ChecklistProvenanceError("strict channel evidence inventory changed")
        statuses = tuple(
            _validate_channel_evidence(
                channel,
                _mapping(evidence[channel], label=f"checklist provenance {channel}"),
                config=config,
                fact_witness=witness,
            )
            for channel in STRICT_CHANNELS
        )
        all_authorized = all(row.authorized for row in statuses)
        return ChecklistProvenanceValidation(
            valid=all_authorized,
            global_valid=all_authorized,
            reason=(
                "pinned_libcg_and_receipt_bound_public_mechanics"
                if all_authorized
                else "one_or_more_strict_channels_unavailable"
            ),
            config=config,
            channel_statuses=statuses,
            sealed_public_catalog=MappingProxyType(catalog.to_dict()),
            revision_5_consumer_migration=MappingProxyType(migration.to_dict()),
            simulator_fact_witness=MappingProxyType(witness.to_dict()),
        )
    except ChecklistProvenanceError as exc:
        return _unavailable_validation(f"r298_provenance_unavailable:{exc}", config=config)


@dataclass(frozen=True)
class _ParsedChannel:
    name: str
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    option_availability: tuple[bool, ...]
    available: bool
    reason: str


@dataclass(frozen=True)
class _ParsedTrace:
    channels: Mapping[str, _ParsedChannel]
    facts: Mapping[str, Any]
    width: int


def _finite_vector(value: object, *, label: str, width: Optional[int] = None) -> tuple[float, ...]:
    values = _sequence(value, label=label)
    if width is not None and len(values) != width:
        raise ChecklistProvenanceError(f"{label} has the wrong candidate width")
    result: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            raise ChecklistProvenanceError(f"{label}[{index}] must be a finite number")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ChecklistProvenanceError(f"{label}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ChecklistProvenanceError(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _bool_vector(value: object, *, label: str, width: int) -> tuple[bool, ...]:
    values = _sequence(value, label=label)
    if len(values) != width or any(type(item) is not bool for item in values):
        raise ChecklistProvenanceError(f"{label} must be bools aligned to candidates")
    return tuple(bool(item) for item in values)


def _trace_mapping(trace_or_payload: Any) -> Mapping[str, Any]:
    if hasattr(trace_or_payload, "to_dict") and callable(trace_or_payload.to_dict):
        trace_or_payload = trace_or_payload.to_dict()
    return _mapping(trace_or_payload, label="r288 checklist trace")


def _parse_r288_trace(trace_or_payload: Any) -> _ParsedTrace:
    trace = _trace_mapping(trace_or_payload)
    names = _sequence(trace.get("channel_names"), label="r288 channel names")
    if tuple(names) != CHANNEL_NAMES:
        raise ChecklistProvenanceError("r288 trace channel order/inventory mismatch")
    channel_rows = _sequence(trace.get("channels"), label="r288 channel rows")
    if len(channel_rows) != len(CHANNEL_NAMES):
        raise ChecklistProvenanceError("r288 trace channel count mismatch")
    width: Optional[int] = None
    residuals = trace.get("residuals")
    if residuals is not None:
        width = len(_finite_vector(residuals, label="r288 residuals"))
    parsed: dict[str, _ParsedChannel] = {}
    for index, raw_row in enumerate(channel_rows):
        row = _mapping(raw_row, label=f"r288 channel[{index}]")
        name = row.get("name")
        if name != CHANNEL_NAMES[index] or name in parsed:
            raise ChecklistProvenanceError("r288 channel names are not exact/aligned")
        raw = _finite_vector(row.get("raw"), label=f"r288 {name}.raw", width=width)
        if width is None:
            width = len(raw)
        normalized = _finite_vector(
            row.get("normalized"), label=f"r288 {name}.normalized", width=width
        )
        mask = _bool_vector(
            row.get("option_availability"), label=f"r288 {name}.option_availability", width=width
        )
        available = _require_bool(row.get("available"), label=f"r288 {name}.available")
        reason = _require_string(row.get("reason"), label=f"r288 {name}.reason")
        parsed[name] = _ParsedChannel(name, raw, normalized, mask, available, reason)
    facts = trace.get("facts", {})
    return _ParsedTrace(
        channels=MappingProxyType(parsed),
        facts=_mapping(facts, label="r288 trace facts"),
        width=0 if width is None else width,
    )


def checklist_trace_evidence_sha256_r298(trace_or_payload: Any) -> str:
    """Digest only aligned r288 evidence fields, never guide prose or facts."""

    parsed = _parse_r288_trace(trace_or_payload)
    material = {
        "channel_names": list(CHANNEL_NAMES),
        "channels": [
            {
                "name": row.name,
                "raw": list(row.raw),
                "normalized": list(row.normalized),
                "option_availability": list(row.option_availability),
                "available": row.available,
                "reason": row.reason,
            }
            for row in (parsed.channels[name] for name in CHANNEL_NAMES)
        ],
    }
    return canonical_digest_r298(material)


def build_checklist_provenance_r298(
    trace_or_payload: Any,
    *,
    public_observation_sha256: str,
    legal_option_order_sha256: str,
    legal_option_multiset_sha256: str,
    catalog_sealer_receipt_path: Path | str,
    revision_5_consumer_migration_receipt_path: Path | str,
    simulator_fact_witness_receipt_path: Path | str,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a record only from pre-existing sealer and simulator witnesses.

    This helper cannot manufacture a catalog receipt, a mechanics receipt, or
    per-fact evidence.  It only joins a caller's public-frame digests to an
    actual r4 catalog sealer output, an explicit pinned r5 consumer migration,
    and a separately file-bound simulator fact witness, then revalidates the
    resulting record before returning it.
    """

    config = load_checklist_provenance_config_r298(config_path)
    parsed = _parse_r288_trace(trace_or_payload)
    public_observation_sha256 = _require_sha256(
        public_observation_sha256, label="public observation digest"
    )
    legal_option_order_sha256 = _require_sha256(
        legal_option_order_sha256, label="legal option order digest"
    )
    legal_option_multiset_sha256 = _require_sha256(
        legal_option_multiset_sha256, label="legal option multiset digest"
    )
    evidence_digest = checklist_trace_evidence_sha256_r298(trace_or_payload)
    legal = {
        "authority": "pinned_libcg_emitted_complete_ordered_legal_options",
        "canonical_libcg_sha256": config.canonical_libcg_sha256,
        "public_observation_sha256": public_observation_sha256,
        "legal_option_order_sha256": legal_option_order_sha256,
        "legal_option_multiset_sha256": legal_option_multiset_sha256,
        "legal_option_count": parsed.width,
    }
    catalog = validate_sealed_public_catalog_r298(
        catalog_sealer_receipt_path, config_path=config_path
    )
    migration = validate_revision_5_consumer_migration_r298(
        revision_5_consumer_migration_receipt_path, config_path=config_path
    )
    if (
        migration.predecessor_catalog_sealer_receipt_sha256 != catalog.sealer_receipt.sha256
        or migration.predecessor_catalog_sha256 != catalog.catalog.sha256
        or migration.predecessor_fixed_vectors_sha256 != catalog.fixed_vectors.sha256
    ):
        raise ChecklistProvenanceError(
            "revision-5 consumer migration does not bind the consumed predecessor catalog"
        )
    witness_identity = file_identity_r298(
        simulator_fact_witness_receipt_path, label="simulator fact witness receipt"
    )
    witness = _validate_simulator_fact_witness(
        witness_identity,
        catalog=catalog,
        legal={
            name: legal[name]
            for name in (
                "public_observation_sha256",
                "legal_option_order_sha256",
                "legal_option_multiset_sha256",
                "legal_option_count",
            )
        },
        trace_evidence_sha256=evidence_digest,
        config=config,
    )
    record = {
        "schema": R298_CHECKLIST_PROVENANCE_SCHEMA,
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "mode": "strict_pinned_public_mechanics",
        "goal_gateway_sha256": config.canonical_goal_sha256,
        "goal_contract_sha256": config.canonical_contract_sha256,
        "config_sha256": config.sha256,
        "canonical_libcg": _canonical_libcg_payload(config),
        "authority_sources": [
            "pinned_libcg_legal_options",
            "receipt_bound_public_catalog",
            "receipt_bound_simulator_mechanics",
            "public_observation",
        ],
        "guide_or_prose_mechanics_authority": False,
        "unpinned_csv_mechanics_authority": False,
        "runtime_wired": False,
        "legal_options": legal,
        "catalog_sealer_receipt": catalog.sealer_receipt.to_dict(),
        "revision_5_consumer_migration_receipt": migration.receipt.to_dict(),
        "simulator_fact_witness_receipt": witness.receipt.to_dict(),
        "trace_binding": {
            "r288_trace_sha256": evidence_digest,
            **{
                name: legal[name]
                for name in (
                    "public_observation_sha256",
                    "legal_option_order_sha256",
                    "legal_option_multiset_sha256",
                    "legal_option_count",
                )
            },
        },
        "phase_4_assertions": {PHASE_4_ASSERTION: True},
        "channel_evidence": {
            channel: {
                "status": "authorized",
                "witness_fact_digests": dict(witness.channel_fact_digests[channel]),
            }
            for channel in STRICT_CHANNELS
        },
    }
    validated = validate_checklist_provenance_r298(
        record,
        expected_trace_evidence_sha256=evidence_digest,
        expected_option_count=parsed.width,
        config_path=config_path,
    )
    if not validated.valid:
        raise ChecklistProvenanceError(
            "strict provenance construction did not validate: " + validated.reason
        )
    return record


def _zeroes(width: int) -> list[float]:
    return [0.0] * width


def _false_mask(width: int) -> list[bool]:
    return [False] * width


def _source_channel_payload(row: _ParsedChannel) -> dict[str, Any]:
    return {
        "raw": list(row.raw),
        "normalized": list(row.normalized),
        "option_availability": list(row.option_availability),
        "available": bool(row.available),
        "reason": row.reason,
    }


def _q3_bench_only_proven(parsed: _ParsedTrace, config: Optional[R298ChecklistProvenanceConfig]) -> tuple[bool, str]:
    if config is None:
        return False, "r298_q3_bench_only_not_proven:configuration_unavailable"
    facts = parsed.facts
    if facts.get("bench_only") is not True:
        return False, "r298_q3_bench_only_not_proven:bench_only_missing_or_false"
    classification = facts.get("classification")
    if classification not in config.q3_classifications:
        return False, "r298_q3_bench_only_not_proven:classification_unavailable"
    return True, "bench_only_public_replacement_classification"


@dataclass(frozen=True)
class ChecklistProvenanceFilterResult:
    """A mapping-like, exact-zero r298 filtered checklist trace."""

    payload: Mapping[str, Any]

    @property
    def residuals(self) -> tuple[float, ...]:
        values = self.payload.get("residuals", ())
        return tuple(float(value) for value in values)

    @property
    def active(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        # The payload contains only plain JSON-safe containers.  A JSON round
        # trip makes callers unable to mutate a frozen result by aliasing it.
        return json.loads(json.dumps(dict(self.payload), sort_keys=True, allow_nan=False))

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def _empty_filter_result(
    reason: str,
    *,
    validation: Optional[ChecklistProvenanceValidation] = None,
    width: int = 0,
) -> ChecklistProvenanceFilterResult:
    zeros = _zeroes(width)
    channels = [
        {
            "name": name,
            "raw": list(zeros),
            "normalized": list(zeros),
            "option_availability": _false_mask(width),
            "available": False,
            "status": "unavailable",
            "reason": reason,
            "provenance_eligible": False,
            "trace_only": name in TRACE_ONLY_CHANNELS,
            "applied_gate": 0.0,
            "final_residual": 0.0,
            "final_residual_vector": list(zeros),
        }
        for name in CHANNEL_NAMES
    ]
    payload: dict[str, Any] = {
        "schema": R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA,
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "channel_names": list(CHANNEL_NAMES),
        "channels": channels,
        "guide_support": {
            "name": GUIDE_SUPPORT_CHANNEL,
            "raw": list(zeros),
            "normalized": list(zeros),
            "option_availability": _false_mask(width),
            "available": False,
            "status": "unavailable",
            "reason": "r298_guide_support_trace_only_fixed_zero",
            "trace_only": True,
            "applied_gate": 0.0,
            "final_residual": 0.0,
            "final_residual_vector": list(zeros),
        },
        "residuals": list(zeros),
        "available": False,
        "active": False,
        "runtime_wired": False,
        "final_residual_gate": 0.0,
        "reason": reason,
        "phase_4_provenance_assertion": PHASE_4_ASSERTION,
        "provenance": (
            _unavailable_validation(reason).to_dict()
            if validation is None
            else validation.to_dict()
        ),
    }
    payload["stage_receipt"] = _stage_receipt_payload(
        source_trace_evidence_sha256=None,
        channels=channels,
        width=width,
    )
    return ChecklistProvenanceFilterResult(MappingProxyType(payload))


def _stage_receipt_payload(
    *,
    source_trace_evidence_sha256: Optional[str],
    channels: Sequence[Mapping[str, Any]],
    width: int,
) -> dict[str, Any]:
    """Make an action-stage, zero-bypass-attestable wrapper receipt.

    This is a record of this explicit function invocation over one candidate
    stage; it is not a policy callback and does not claim a wall-clock game
    action happened.  A materializer can bind it to an actor-visible frame
    using the public-observation and legal-option digests in its provenance.
    """

    by_name = {row.get("name"): row for row in channels}
    strict_zero_unavailable = all(
        isinstance(by_name.get(name), Mapping)
        and (
            by_name[name].get("available") is True
            or (
                by_name[name].get("status") == "unavailable"
                and by_name[name].get("final_residual") == 0.0
                and by_name[name].get("final_residual_vector") == _zeroes(width)
            )
        )
        for name in STRICT_CHANNELS
    )
    q5_q6_zero = all(
        isinstance(by_name.get(name), Mapping)
        and by_name[name].get("trace_only") is True
        and by_name[name].get("applied_gate") == 0.0
        and by_name[name].get("final_residual") == 0.0
        and by_name[name].get("final_residual_vector") == _zeroes(width)
        for name in TRACE_ONLY_CHANNELS
    )
    receipt: dict[str, Any] = {
        "schema": R298_CHECKLIST_STAGE_RECEIPT_SCHEMA,
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "stage_kind": "explicit_candidate_stage_trace_filter",
        "sealed_r298_wrapper_invoked": True,
        "acting_player_public_information_only": True,
        "input_surface": "r288_trace_evidence_digest_plus_strict_public_receipts_only",
        "source_trace_evidence_sha256": source_trace_evidence_sha256,
        "strict_provenance_filter_applied": True,
        "q1_q2_q8_missing_or_mismatch_unavailable_and_zero": strict_zero_unavailable,
        "q5_q6_final_residual_exact_zero": q5_q6_zero,
        "guide_support_final_residual_exact_zero": True,
        "legacy_r288_guide_residual_included": False,
        "all_final_residuals_exact_zero": True,
        "runtime_wired": False,
        "production_or_inzi_authority": False,
        "phase_4_provenance_assertion": PHASE_4_ASSERTION,
    }
    receipt["stage_receipt_sha256"] = canonical_digest_r298(receipt)
    return receipt


def filter_r288_checklist_trace_r298(
    trace_or_payload: Any,
    *,
    provenance: Mapping[str, Any] | Any,
    config_path: Path | str | None = None,
) -> ChecklistProvenanceFilterResult:
    """Filter an r288 trace through the strict r298 evidence boundary.

    This is a diagnostic-only transform.  It never returns a nonzero final
    residual, and never changes a policy logit.  Strict rows Q1/Q2/Q8 are
    marked unavailable and zeroed unless their complete mechanics provenance
    validates against the source trace's exact option-aligned digest.
    """

    try:
        parsed = _parse_r288_trace(trace_or_payload)
        evidence_digest = checklist_trace_evidence_sha256_r298(trace_or_payload)
    except ChecklistProvenanceError as exc:
        return _empty_filter_result(f"r298_trace_unavailable:{exc}")
    validation = validate_checklist_provenance_r298(
        provenance,
        expected_trace_evidence_sha256=evidence_digest,
        expected_option_count=parsed.width,
        config_path=config_path,
    )
    config = validation.config
    zeros = _zeroes(parsed.width)
    strict_status = validation.channel_status
    output_channels: list[dict[str, Any]] = []
    any_available = False
    for name in CHANNEL_NAMES:
        source = parsed.channels[name]
        trace_only = name in TRACE_ONLY_CHANNELS
        eligible = True
        reason = source.reason
        if name in STRICT_CHANNELS:
            status = strict_status.get(name)
            eligible = bool(status and status.authorized)
            if not eligible:
                reason = status.reason if status is not None else "r298_provenance_unavailable"
        elif name == "replacement_alakazam_line":
            eligible, reason = _q3_bench_only_proven(parsed, config)
        # A source row that labels its own evidence unavailable cannot retain
        # a diagnostic preference merely because another receipt validates the
        # general mechanics family.  This is especially important for Q7:
        # unobserved prize identity is never upgraded by a route receipt.
        if not source.available and not trace_only:
            eligible = False
            reason = "r298_source_evidence_unavailable:" + source.reason
        if eligible:
            raw = list(source.raw)
            normalized = list(source.normalized)
            mask = list(source.option_availability)
            available = bool(source.available)
            status_text = "available" if available else "unavailable"
        else:
            raw = list(zeros)
            normalized = list(zeros)
            mask = _false_mask(parsed.width)
            available = False
            status_text = "unavailable"
        if trace_only:
            # Q5/Q6 remain useful factual trace rows, but no caller can mistake
            # them for an enabled residual.  Their raw evidence is retained if
            # structurally valid; final residual remains exactly zero.
            raw = list(source.raw)
            normalized = list(source.normalized)
            mask = list(source.option_availability)
            available = bool(source.available)
            status_text = "available" if available else "unavailable"
            reason = "r298_trace_only_fixed_zero:" + source.reason
        any_available = any_available or available
        output_channels.append(
            {
                "name": name,
                "raw": raw,
                "normalized": normalized,
                "option_availability": mask,
                "available": available,
                "status": status_text,
                "reason": reason,
                "provenance_eligible": bool(eligible and not trace_only),
                "trace_only": trace_only,
                "source": _source_channel_payload(source),
                "applied_gate": 0.0,
                # Scalar is deliberately repeated with a vector for consumers
                # that operate at different trace granularities.
                "final_residual": 0.0,
                "final_residual_vector": list(zeros),
            }
        )
    payload: dict[str, Any] = {
        "schema": R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA,
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "channel_names": list(CHANNEL_NAMES),
        "channels": output_channels,
        "guide_support": {
            "name": GUIDE_SUPPORT_CHANNEL,
            "raw": list(zeros),
            "normalized": list(zeros),
            "option_availability": _false_mask(parsed.width),
            "available": False,
            "status": "unavailable",
            "reason": "r298_guide_support_trace_only_fixed_zero",
            "trace_only": True,
            "applied_gate": 0.0,
            "final_residual": 0.0,
            "final_residual_vector": list(zeros),
        },
        "residuals": list(zeros),
        "available": any_available,
        "active": False,
        "runtime_wired": False,
        "final_residual_gate": 0.0,
        "reason": "r298_strict_provenance_filtered_zero_gated",
        "source_trace_evidence_sha256": evidence_digest,
        "phase_4_provenance_assertion": PHASE_4_ASSERTION,
        "provenance": validation.to_dict(),
        "facts": {
            "r298_checklist_provenance": validation.to_dict(),
            "r298_source_trace_evidence_sha256": evidence_digest,
            "q3_bench_only": output_channels[2]["provenance_eligible"],
            "q5_q6_trace_only_zero": True,
            "q1_q2_q8_pinned_provenance_or_zero_unavailable": True,
        },
    }
    payload["stage_receipt"] = _stage_receipt_payload(
        source_trace_evidence_sha256=evidence_digest,
        channels=output_channels,
        width=parsed.width,
    )
    return ChecklistProvenanceFilterResult(MappingProxyType(payload))


def checklist_provenance_schema_manifest_r298(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return the frozen, checksum-bindable isolated checklist schema."""

    config = load_checklist_provenance_config_r298(config_path)
    definition = {
        "schema": R298_CHECKLIST_PROVENANCE_SCHEMA,
        "result_schema": R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA,
        "stage_receipt_schema": R298_CHECKLIST_STAGE_RECEIPT_SCHEMA,
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "canonical_goal_sha256": config.canonical_goal_sha256,
        "canonical_contract_sha256": config.canonical_contract_sha256,
        "canonical_r241_typed_source_sha256": config.canonical_r241_typed_source_sha256,
        "canonical_libcg_contract_sha256": config.canonical_libcg_contract_sha256,
        "canonical_libcg_sha256": config.canonical_libcg_sha256,
        "channels": list(CHANNEL_NAMES),
        "strict_mechanics_channels": list(STRICT_CHANNELS),
        "strict_required_fact_kinds": {
            name: list(kinds)
            for name, kinds in config.strict_required_fact_kinds.items()
        },
        "fact_authority_by_kind": dict(config.fact_authority_by_kind),
        "legal_options": "complete_ordered_list_emitted_by_pinned_libcg",
        "public_catalog": "revision_4_sealer_receipt_is_historical_predecessor_only",
        "simulator_mechanics": "file_bound_public_frame_simulator_fact_witness",
        "catalog_sealer_source_sha256": config.catalog_sealer_source_sha256,
        "catalog_sealer_receipt_schema": config.catalog_sealer_receipt_schema,
        "catalog_sealer_immutable_artifact_pins": dict(
            config.catalog_sealer_immutable_artifact_pins
        ),
        "revision_5_consumer_migration_schema": config.revision_5_consumer_migration_schema,
        "revision_5_consumer_migration_immutable_anchor": dict(
            config.revision_5_consumer_migration_anchor
        ),
        "revision_5_consumer_migration_required_assertions": list(
            config.revision_5_consumer_migration_required_assertions
        ),
        "revision_4_catalog_reuse_without_explicit_revision_5_migration": False,
        "simulator_fact_witness_schema": config.simulator_fact_witness_schema,
        "engine_evidence_schema": config.engine_evidence_schema,
        "simulator_fact_witness_immutable_anchor": dict(
            config.simulator_fact_witness_immutable_anchor
        ),
        "strict_mechanics_current_disposition": (
            "q1_q2_q8_unavailable_zero_until_pinned_revision_5_consumer_migration_and_immutable_owner_pinned_simulator_witness"
        ),
        "legacy_guide_prose_or_unpinned_csv_mechanics_authority": False,
        "q3_bench_only": True,
        "q5_q6_trace_only_exact_zero": True,
        "unknown_hidden_or_malformed_evidence_final_residual": 0.0,
        "default_zero_and_inert": True,
        "runtime_wired": False,
        "phase_4_provenance_assertion": PHASE_4_ASSERTION,
    }
    return {
        "schema": "poke_bot.alakazam_checklist_provenance_r298_schema_manifest/v1",
        "revision": R298_CHECKLIST_PROVENANCE_REVISION,
        "config": config.to_dict(),
        "schema_definition": definition,
        "schema_sha256": canonical_digest_r298(definition),
        "default_zero_and_inert": True,
        "runtime_wired": False,
        "production_or_inzi_authority": False,
    }


__all__ = [
    "CANONICAL_CONTRACT_SHA256",
    "CANONICAL_GOAL_SHA256",
    "CANONICAL_LIBCG_BINARY_SHA256",
    "CANONICAL_LIBCG_CONTRACT_SHA256",
    "CANONICAL_R241_TYPED_SOURCE_SHA256",
    "CHANNEL_NAMES",
    "ChecklistChannelProvenanceStatus",
    "ChecklistProvenanceError",
    "ChecklistProvenanceFilterResult",
    "ChecklistProvenanceValidation",
    "DEFAULT_CONFIG_PATH",
    "FileIdentityR298",
    "GUIDE_SUPPORT_CHANNEL",
    "PHASE_4_ASSERTION",
    "R298_CHECKLIST_PROVENANCE_CONFIG_SCHEMA",
    "R298_CHECKLIST_PROVENANCE_RESULT_SCHEMA",
    "R298_CHECKLIST_PROVENANCE_REVISION",
    "R298_CHECKLIST_PROVENANCE_SCHEMA",
    "R298_CHECKLIST_STAGE_RECEIPT_SCHEMA",
    "R298_ENGINE_EVIDENCE_SCHEMA",
    "R298_PINNED_PUBLIC_CATALOG_SCHEMA",
    "R298_REV5_CONSUMER_MIGRATION_RECEIPT_SCHEMA",
    "R298_SEALED_PUBLIC_CATALOG_RECEIPT_SCHEMA",
    "R298_SIMULATOR_FACT_WITNESS_RECEIPT_SCHEMA",
    "SealedPublicCatalogEvidenceR298",
    "Revision5ConsumerMigrationEvidenceR298",
    "SimulatorFactWitnessEvidenceR298",
    "STRICT_CHANNELS",
    "TRACE_ONLY_CHANNELS",
    "build_checklist_provenance_r298",
    "build_public_catalog_receipt_r298",
    "build_simulator_mechanics_receipt_r298",
    "canonical_digest_r298",
    "canonical_json_bytes_r298",
    "checklist_provenance_schema_manifest_r298",
    "checklist_trace_evidence_sha256_r298",
    "file_identity_r298",
    "load_checklist_provenance_config_r298",
    "revision_5_consumer_migration_schema_manifest_r298",
    "sha256_file_r298",
    "validate_checklist_provenance_r298",
    "validate_pinned_public_catalog_r298",
    "validate_revision_5_consumer_migration_r298",
    "validate_sealed_public_catalog_r298",
    "filter_r288_checklist_trace_r298",
]
