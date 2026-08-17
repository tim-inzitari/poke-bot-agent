"""Create-only rev5/r303 public-rule record materialization.

This module is deliberately separate from the historical r298 materializer.
It consumes the r303 frozen-schema/preflight boundary, emits only a
public-information representation, and leaves every selected-action training
mask false until a later sealed post-handoff trajectory validator exists.
Nothing here loads a checkpoint, trains a model, starts a service, stages to
Inzi, or changes an existing policy route.
"""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
import resource
import socket
import stat
import threading
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from .alakazam_checklist_provenance_r298 import (
    CHANNEL_NAMES,
    filter_r288_checklist_trace_r298,
)
from .alakazam_collision_census_r298 import (
    R274_EXACT_FEATURES_SOURCE_SHA256,
    canonical_json_bytes,
    canonical_sha256,
    current_feature_token_hashes,
    recorded_episode_frame_coverage,
    sha256_file,
    stage_descriptors_from_recorded_episode,
    validate_raw_corpus_manifest,
)
from .alakazam_public_rule_adapter_r298 import (
    build_public_rule_representation,
    is_public_catalog_eligible,
    load_sealed_public_catalog,
    sanitize_public_observation,
)
from .alakazam_rule_derivative_freeze_r303 import (
    R303_CONTRACT_PATH,
    R303_CONTRACT_SHA256,
    R303_GOAL_GATEWAY_SHA256,
    R303_GOAL_REVISION,
    R303_MATERIALIZATION_PREFLIGHT_SCHEMA,
    R303_ROOT_OWNER_REVISION,
    R303FreezeError,
    validate_r303_frozen_schema_manifest,
    validate_r303_materialization_preflight,
    validate_r303_schema_freeze_receipt,
    validate_r303_zero_bypass_receipt,
)
from .alakazam_rule_derivative_corpus_scope_rev6 import (
    REV6_CONTRACT_SHA256,
    REV6_GOAL_GATEWAY_SHA256,
    REV6_GOAL_REVISION,
    REV6_PREDECESSOR_CONTRACT_SHA256,
    REV6_PREDECESSOR_GATEWAY_SHA256,
)
from .alakazam_rule_derivative_predecessor_compat_rev7 import (
    REV7_CENSUS_PHYSICAL_INVENTORY_KEY,
    build_revision_6_scope_migration_under_revision_7 as build_corpus_scope_migration_receipt,
    validate_revision_6_scope_migration_under_revision_7,
    write_revision_6_scope_migration_under_revision_7_create_only as write_corpus_scope_migration_receipt_create_only,
)
from .alakazam_simulator_rule_targets_r298 import (
    compile_simulator_rule_targets,
    rule_head_target_vectors,
)


R303_RECORD_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_refeatured_record/v1"
R303_RUN_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_materialization_run/v1"
R303_DAY_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_day_completion/v1"
R303_SHARD_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_public_rule_shard/v1"
R303_SPLIT_UNIT_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_episode_split_unit/v1"
R303_SPLIT_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_split_manifest/v1"
R303_MANIFEST_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_refeatured_manifest/v1"
R303_RECEIPT_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_refeatured_receipt/v1"
R303_COMPLETE_SCHEMA: Final = "poke_bot.alakazam_rule_derivative_r303_completion/v1"
R303_LIST_VARIANT_STRATA_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_list_variant_strata_manifest/v1"
)
R303_SCOPE_INVENTORY_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_revision_6_scope_inventory/v1"
)
R303_SCOPE_SPLIT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_revision_6_scope_split_manifest/v1"
)
R303_PROCESS_POOL_PROBE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_process_pool_startup_probe/v1"
)
R303_PROCESS_POOL_EXECUTION_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_process_pool_execution/v1"
)
R303_SCOPE_FILTER_TEST_PATH: Final = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "test_alakazam_rule_derivative_materialize_r303.py"
)

MAX_PHYSICAL_SHARD_BYTES: Final = 96 * 1024 * 1024
TARGET_PHYSICAL_SHARD_BYTES: Final = 92 * 1024 * 1024
BUFFER_BYTES: Final = 1024 * 1024
EXPERIMENT_RAM_LIMIT_BYTES: Final = 96 * 1024**3
MAX_WHOLE_DAY_WORKERS: Final = 24
DEFAULT_LEASE_PATH: Final = Path(
    "/srv/poke-bot-agent/outputs/quarantine/"
    "alakazam-elmo-rule-derivative/r298-experiment-lease/.r298-memory-heavy-phase.lock"
)
CANONICAL_ELMO_HOSTNAME: Final = "truenas"


class R303MaterializationError(RuntimeError):
    """The r303 diagnostic-only materializer cannot safely continue."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R303MaterializationError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise R303MaterializationError(f"{field} must be a list")
    return list(value)


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise R303MaterializationError(f"{field} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise R303MaterializationError(f"{field} must be hexadecimal") from exc
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise R303MaterializationError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _sha256_file(path: Path) -> str:
    try:
        return sha256_file(path)
    except Exception as exc:
        raise R303MaterializationError(f"cannot hash {path}") from exc


def _read_json(path: Path | str, *, field: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise R303MaterializationError(f"{field} must be a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303MaterializationError(f"cannot read {field}: {source}") from exc
    return dict(_mapping(value, field=field))


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise R303MaterializationError(f"create-only artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_file(path)


def _utc_days() -> tuple[str, ...]:
    start = date(2026, 7, 13)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(30))


def _assert_exact_feature_source() -> None:
    path = Path(__file__).with_name("features.py")
    if _sha256_file(path) != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise R303MaterializationError("legacy feature source is not the frozen r274 ABI")


def _validate_inputs(
    *,
    raw_manifest: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
    preflight: Mapping[str, Any],
    corpus_scope_migration_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind immutable rev5 freeze bytes as rev6 predecessor evidence.

    Revision 6 intentionally forbids blind re-freezing/re-hashing the r5
    schema and parity artifacts.  The old r303 validators re-open today's
    gateway and therefore correctly reject an r6 workspace; use their already
    sealed identities here, then require the contract-owner's r6 migration
    receipt before any row writer is initialized.
    """

    try:
        validate_raw_corpus_manifest(raw_manifest)
    except Exception as exc:
        raise R303MaterializationError("raw corpus is not the immutable exact 30-day predecessor") from exc
    manifest = dict(_mapping(schema_manifest, field="revision-5 frozen schema predecessor"))
    bypass = dict(_mapping(zero_bypass_receipt, field="revision-5 zero-bypass predecessor"))
    freeze = dict(_mapping(schema_freeze_receipt, field="revision-5 schema-freeze predecessor"))
    checked = dict(_mapping(preflight, field="revision-5 materialization preflight predecessor"))
    if (
        manifest.get("exact_goal_contract_sha256"),
        manifest.get("rule_derivative_gateway_sha256"),
        manifest.get("rule_derivative_contract_sha256"),
        manifest.get("goal_revision"),
    ) != (R303_CONTRACT_SHA256, R303_GOAL_GATEWAY_SHA256, R303_CONTRACT_SHA256, R303_GOAL_REVISION):
        raise R303MaterializationError("frozen schema is not the immutable rev5/r303 predecessor")
    if (
        bypass.get("status"),
        bypass.get("goal_contract_sha256"),
        bypass.get("rule_derivative_gateway_sha256"),
        bypass.get("frozen_schema_manifest_sha256"),
        bypass.get("all_new_routes_exact_zero_or_bypassed"),
        bypass.get("layer_off_bit_identical_baseline_logits"),
        bypass.get("layer_off_identical_legal_choice"),
    ) != (
        "passed_exact_baseline_logits", R303_CONTRACT_SHA256, R303_GOAL_GATEWAY_SHA256,
        canonical_sha256(manifest), True, True, True,
    ):
        raise R303MaterializationError("zero-bypass predecessor does not bind frozen schema/parity")
    if (
        freeze.get("goal_contract_sha256"), freeze.get("goal_revision"),
        freeze.get("feature_schema_sha256"), freeze.get("target_schema_sha256"),
        freeze.get("checklist_provenance_schema_sha256"), freeze.get("zero_bypass_receipt_sha256"),
    ) != (
        R303_CONTRACT_SHA256, R303_GOAL_REVISION, manifest.get("feature_schema_sha256"),
        manifest.get("target_schema_sha256"), manifest.get("checklist_provenance_schema_sha256"), canonical_sha256(bypass),
    ):
        raise R303MaterializationError("schema-freeze predecessor does not cross-link frozen bytes")
    expected = {
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_manifest),
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_receipt),
        "frozen_schema_manifest_sha256": canonical_sha256(manifest),
        "zero_bypass_receipt_sha256": canonical_sha256(bypass),
        "schema_freeze_receipt_sha256": canonical_sha256(freeze),
    }
    for field, digest in expected.items():
        if checked.get(field) != digest:
            raise R303MaterializationError(f"preflight {field} does not bind supplied bytes")
    if (
        checked.get("schema"), checked.get("status"), checked.get("goal_contract_sha256"),
        checked.get("goal_revision"), checked.get("selected_action_target_mode"),
        checked.get("opponent_belief_target_mode"),
    ) != (
        R303_MATERIALIZATION_PREFLIGHT_SCHEMA,
        "ready_for_exact_30_day_elmo_refeaturization_diagnostic_targets_only",
        R303_CONTRACT_SHA256, R303_GOAL_REVISION,
        "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "unavailable_masked_no_loss",
    ):
        raise R303MaterializationError("materialization preflight schema drifted")
    if corpus_scope_migration_receipt is None:
        raise R303MaterializationError("revision-6 corpus-scope migration receipt is required before row materialization")
    try:
        migration = validate_revision_6_scope_migration_under_revision_7(
            corpus_scope_migration_receipt
        )
    except Exception as exc:
        raise R303MaterializationError("revision-6 corpus-scope migration receipt failed") from exc
    if migration.get("validated_30_day_raw_manifest_sha256") != expected["raw_expert_corpus_manifest_sha256"]:
        raise R303MaterializationError("revision-6 migration receipt binds a different raw manifest")
    if migration.get("feature_schema_sha256") != manifest.get("feature_schema_sha256"):
        raise R303MaterializationError("revision-6 migration feature schema mismatch")
    if migration.get("target_schema_sha256") != manifest.get("target_schema_sha256"):
        raise R303MaterializationError("revision-6 migration target schema mismatch")
    if migration.get("checklist_provenance_schema_sha256") != manifest.get("checklist_provenance_schema_sha256"):
        raise R303MaterializationError("revision-6 migration checklist schema mismatch")
    if migration.get("zero_bypass_receipt_sha256") != expected["zero_bypass_receipt_sha256"]:
        raise R303MaterializationError("revision-6 migration zero-bypass mismatch")
    return {
        "raw_manifest": dict(raw_manifest),
        "raw_receipt": dict(raw_receipt),
        "schema_manifest": manifest,
        "zero_bypass_receipt": bypass,
        "schema_freeze_receipt": freeze,
        "preflight": checked,
        "corpus_scope_migration_receipt": migration,
        **expected,
    }


def _catalog_binding(schema_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    surface = _mapping(schema_manifest.get("sealed_structured_catalog"), field="sealed catalog surface")
    required = {
        "catalog_file_sha256",
        "fixed_vectors_file_sha256",
        "catalog_file_size_bytes",
        "fixed_vectors_file_size_bytes",
        "structured_residual_additive_after_card2vec",
        "card2vec_preserved_unchanged",
        "training_or_inzi_authority",
        "runtime_wired",
    }
    if not required.issubset(surface):
        raise R303MaterializationError("frozen schema has incomplete sealed catalog binding")
    if (
        surface.get("structured_residual_additive_after_card2vec") is not True
        or surface.get("card2vec_preserved_unchanged") is not True
        or surface.get("training_or_inzi_authority") is not False
        or surface.get("runtime_wired") is not False
    ):
        raise R303MaterializationError("sealed catalog binding violates frozen boundary")
    return surface


def load_r303_sealed_catalog(
    *,
    catalog_path: Path | str,
    catalog_receipt_path: Path | str,
    catalog_vectors_path: Path | str,
    schema_manifest: Mapping[str, Any],
) -> Any:
    """Load only the exact historical catalog bytes approved by the r303 freeze."""

    binding = _catalog_binding(schema_manifest)
    try:
        catalog = load_sealed_public_catalog(
            catalog_path, catalog_receipt_path, vectors_path=catalog_vectors_path
        )
    except Exception as exc:
        raise R303MaterializationError("sealed public catalog cannot be reopened") from exc
    if not is_public_catalog_eligible(catalog):
        raise R303MaterializationError("sealed public catalog is not adapter eligible")
    provenance = getattr(catalog, "provenance", None)
    for schema_field, provenance_field in (
        ("catalog_file_sha256", "catalog_file_sha256"),
        ("fixed_vectors_file_sha256", "vectors_file_sha256"),
    ):
        if getattr(provenance, provenance_field, None) != binding.get(schema_field):
            raise R303MaterializationError("sealed catalog bytes do not match frozen schema")
    for path, expected_size, label in (
        (Path(catalog_path), binding.get("catalog_file_size_bytes"), "catalog"),
        (Path(catalog_vectors_path), binding.get("fixed_vectors_file_size_bytes"), "catalog vectors"),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != _integer(
            expected_size, field=f"frozen {label} size", minimum=1
        ):
            raise R303MaterializationError(f"sealed {label} file identity drifted")
    return catalog


def _public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = dict(sanitize_public_observation(observation))
    except Exception as exc:
        raise R303MaterializationError("cannot sanitize actor-visible observation") from exc
    current = result.get("current")
    if not isinstance(current, Mapping):
        raise R303MaterializationError("sanitized observation lacks public current state")
    trimmed = dict(current)
    for name in (
        "result", "resultReason", "reason", "future", "futureState", "future_state",
        "futureTransitions", "future_transitions", "nextState", "next_state", "successor",
    ):
        trimmed.pop(name, None)
        result.pop(name, None)
    result["current"] = trimmed
    return result


def _digest_identity(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        raise R303MaterializationError(f"{field} is not a stable scalar")
    return canonical_sha256({field: str(value)})


def _episode_source(payload: Mapping[str, Any], archive: Mapping[str, Any], *, day: str, member: str, raw: bytes) -> tuple[str, str, str]:
    identifier = payload.get("id")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise R303MaterializationError("raw episode lacks configuration")
    return (
        _digest_identity(identifier, field="payload_id"),
        _digest_identity(configuration.get("seed"), field="seed"),
        "sha256:" + hashlib.sha256(raw).hexdigest(),
    )


def _extract_decks(payload: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    """Read same-seat setup cards transiently for rev6 eligibility only.

    Raw card IDs never enter a materialized record; only the literal
    card-743 result and canonical multiset digest may be retained.
    """

    try:
        steps = _rows(payload.get("steps"), field="episode steps")
        first = _rows(steps[0], field="episode setup step")
        visual = _mapping(first[0], field="episode setup visual").get("visualize")
        trace = _rows(visual, field="episode visual trace")
        setup = _mapping(trace[0], field="episode setup trace").get("action")
        decks = _rows(setup, field="episode setup decks")
    except (IndexError, R303MaterializationError):
        return {}
    if len(decks) != 2:
        return {}
    result: dict[int, tuple[int, ...]] = {}
    for seat, deck in enumerate(decks):
        try:
            cards = tuple(_integer(card, field="setup card", minimum=1) for card in _rows(deck, field="setup deck"))
        except R303MaterializationError:
            continue
        result[seat] = cards
    return result


def _acting_seat_eligibility(
    decks: Mapping[int, tuple[int, ...]], acting_seat: int
) -> tuple[bool, str, str | None]:
    """Implement the rev6 literal *same-seat* 743 predicate.

    This deliberately does not use an archetype label, pilot-list digest, or
    any opponent deck fact.  A card-743 tech copy qualifies; a qualifying
    opponent does not make the acting seat eligible.
    """

    deck = decks.get(acting_seat)
    if deck is None:
        return False, "missing_or_malformed_same_acting_seat_setup_deck", None
    digest = "sha256:" + hashlib.sha256(
        json.dumps(sorted(deck), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if 743 not in deck:
        return False, "same_acting_seat_setup_deck_lacks_card_743", digest
    return True, "eligible_literal_same_acting_seat_setup_deck_contains_card_743", digest


def _full_actions_by_step(descriptors: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    actions: dict[int, tuple[int, list[int]]] = {}
    for descriptor in descriptors:
        source = _mapping(descriptor.get("source"), field="stage source")
        step = _integer(source.get("env_step"), field="environment step", minimum=0)
        stage = _integer(source.get("factorized_stage"), field="factorized stage", minimum=0)
        candidates = _rows(descriptor.get("candidates"), field="stage candidates")
        chosen = _integer(descriptor.get("selected_candidate_index"), field="selected candidate", minimum=0)
        if chosen >= len(candidates):
            raise R303MaterializationError("selected candidate is absent")
        action = [_integer(item, field="action option", minimum=0) for item in _rows(candidates[chosen], field="selected action")]
        previous = actions.get(step)
        if previous is None or stage > previous[0]:
            actions[step] = (stage, action)
    return {step: action for step, (_stage, action) in actions.items()}


def _safe_source(
    *, archive: Mapping[str, Any], day: str, member: str, member_sha256: str,
    group_id: str, seed_id: str, descriptor_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": _sha(archive.get("sha256"), field="source archive digest"),
        "utc_day_partition": day,
        "group_id": group_id,
        "seed_id": seed_id,
        "archive_member": member,
        "archive_member_sha256": member_sha256,
        "env_step": _integer(descriptor_source.get("env_step"), field="environment step", minimum=0),
        "acting_seat": _integer(descriptor_source.get("acting_seat"), field="acting seat", minimum=0),
        "factorized_stage": _integer(descriptor_source.get("factorized_stage"), field="factorized stage", minimum=0),
        "acting_deck_multiset_sha256": descriptor_source.get("acting_deck_multiset_sha256"),
        "source_stratification_only": True,
    }


def _zero_legacy_trace(width: int) -> dict[str, Any]:
    zeroes = [0.0] * width
    return {
        "channel_names": list(CHANNEL_NAMES),
        "channels": [
            {
                "name": name,
                "raw": list(zeroes),
                "normalized": list(zeroes),
                "option_availability": [False] * width,
                "available": False,
                "reason": "r303_legacy_diagnostic_unavailable",
            }
            for name in CHANNEL_NAMES
        ],
        "facts": {"bench_only": False, "classification": "unavailable"},
        "residuals": list(zeroes),
    }


def _checklist_trace(observation: Mapping[str, Any], candidates: Sequence[Sequence[int]], *, deck: Iterable[int] | None) -> dict[str, Any]:
    """Emit all eight rows, with strict mechanics unavailable/zero pre-witness."""

    legacy: Mapping[str, Any] = _zero_legacy_trace(len(candidates))
    if deck is not None:
        try:
            from .alakazam_turn_checklist_logit_layer import evaluate_turn_checklist

            evaluated = evaluate_turn_checklist(observation, candidates, tuple(deck))
            if hasattr(evaluated, "to_dict"):
                evaluated = evaluated.to_dict()
            if isinstance(evaluated, Mapping):
                legacy = evaluated
        except Exception:
            pass
    try:
        trace = filter_r288_checklist_trace_r298(legacy, provenance={}).to_dict()
    except Exception as exc:
        raise R303MaterializationError("canonical strict checklist filter failed") from exc
    if trace.get("runtime_wired") is not False or trace.get("final_residual_gate") != 0.0:
        raise R303MaterializationError("checklist trace is not exact-zero/inert")
    if len(trace.get("channels", ())) != 8:
        raise R303MaterializationError("checklist trace omitted a requested channel")
    return trace


def _diagnostic_selected_target(
    observation: Mapping[str, Any], action: Sequence[int], *, catalog: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile a realized target diagnostically; never arm a training mask."""

    decision = {"observation": dict(observation), "action": list(action)}
    try:
        compiled = compile_simulator_rule_targets(
            decision,
            metadata_catalog=catalog,
            trajectory_receipt_validator=None,
            require_trainable_trajectory_receipt=True,
            strict=True,
        )
    except Exception as strict_error:
        try:
            compiled = compile_simulator_rule_targets(
                decision,
                metadata_catalog=catalog,
                trajectory_receipt_validator=None,
                require_trainable_trajectory_receipt=False,
                strict=False,
            )
        except Exception as diagnostic_error:
            compiled = {
                "schema": "poke_bot.alakazam_simulator_rule_targets/v1",
                "status": "unavailable",
                "target_only": True,
                "reason": f"diagnostic_target_compiler_exception:{type(diagnostic_error).__name__}",
            }
        if isinstance(compiled, Mapping):
            compiled = dict(compiled)
            compiled["materializer_trainability"] = {
                "available": False,
                "reason": f"sealed_r5_trajectory_ledger_required:{type(strict_error).__name__}",
                "all_training_masks_forced_false": True,
            }
    if not isinstance(compiled, Mapping):
        raise R303MaterializationError("selected-action compiler returned a non-object")
    target = dict(compiled)
    target.pop("privileged_belief_targets", None)
    target.pop("opponent_belief", None)
    target["target_only"] = True
    target["policy_feature_eligible"] = False
    target["training_eligible"] = False
    try:
        vectors = dict(rule_head_target_vectors(compiled, trajectory_receipt_validator=None))
    except Exception:
        vectors = {"target_training_eligible": False, "unavailable": True}
    vectors.pop("opponent_belief", None)
    if vectors.get("target_training_eligible") is not False:
        raise R303MaterializationError("pre-handoff record attempted to emit a trainable target mask")
    vectors["target_training_eligible"] = False
    vectors["opponent_belief_target_mode"] = "unavailable_masked_no_loss"
    return target, vectors


@dataclass(frozen=True)
class R303ShardIdentity:
    relative_path: str
    sha256: str
    size_bytes: int
    record_count: int
    uncompressed_jsonl_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R303_SHARD_SCHEMA,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
            "uncompressed_jsonl_bytes": self.uncompressed_jsonl_bytes,
            "physical_size_cap_bytes": MAX_PHYSICAL_SHARD_BYTES,
        }


class _ShardWriter:
    """Bounded deterministic gzip JSONL writer with content-addressed publication."""

    def __init__(self, root: Path, *, day: str) -> None:
        self.root = root
        self.day = day
        self.directory = root / "shards"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.pending: list[bytes] = []
        self.pending_bytes = 0
        self.part = 0
        self.stream: Any | None = None
        self.temp_path: Path | None = None
        self.current_hash: Any | None = None
        self.current_size = 0
        self.current_records = 0
        self.current_uncompressed = 0
        self.identities: list[R303ShardIdentity] = []

    def _open(self) -> None:
        if self.stream is not None:
            return
        self.temp_path = self.directory / f".private-{self.day}-{self.part:05d}.part"
        self.part += 1
        self.stream = self.temp_path.open("xb")
        self.current_hash = hashlib.sha256()
        self.current_size = self.current_records = self.current_uncompressed = 0

    @staticmethod
    def _member(rows: Sequence[bytes]) -> bytes:
        return gzip.compress(b"".join(rows), compresslevel=6, mtime=0)

    def _close(self) -> None:
        if self.stream is None:
            return
        assert self.temp_path is not None and self.current_hash is not None
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if not 0 < self.current_size <= MAX_PHYSICAL_SHARD_BYTES:
            raise R303MaterializationError("invalid r303 physical shard size")
        digest = "sha256:" + self.current_hash.hexdigest()
        final = self.directory / f"sha256-{digest[7:]}.r303-public-rule-records.jsonl.gz"
        if final.exists():
            if final.is_symlink() or final.stat().st_size != self.current_size or _sha256_file(final) != digest:
                raise R303MaterializationError("content-addressed shard conflict; refusing replacement")
            self.temp_path.unlink()
        else:
            try:
                os.link(self.temp_path, final)
            except FileExistsError:
                if final.is_symlink() or final.stat().st_size != self.current_size or _sha256_file(final) != digest:
                    raise R303MaterializationError("raced content-addressed shard conflict")
            finally:
                with contextlib.suppress(FileNotFoundError):
                    self.temp_path.unlink()
        self.identities.append(
            R303ShardIdentity(
                relative_path=str(final.relative_to(self.root.parent)),
                sha256=digest,
                size_bytes=self.current_size,
                record_count=self.current_records,
                uncompressed_jsonl_bytes=self.current_uncompressed,
            )
        )
        self.stream = self.temp_path = self.current_hash = None

    def _append(self, rows: Sequence[bytes]) -> None:
        if not rows:
            return
        member = self._member(rows)
        if len(member) > MAX_PHYSICAL_SHARD_BYTES:
            if len(rows) == 1:
                raise R303MaterializationError("one r303 public record exceeds 96 MiB")
            midpoint = len(rows) // 2
            self._append(rows[:midpoint])
            self._append(rows[midpoint:])
            return
        self._open()
        if self.current_size and self.current_size + len(member) > TARGET_PHYSICAL_SHARD_BYTES:
            self._close()
            self._open()
        if self.current_size + len(member) > MAX_PHYSICAL_SHARD_BYTES:
            raise R303MaterializationError("cannot fit gzip member within the physical cap")
        assert self.stream is not None and self.current_hash is not None
        self.stream.write(member)
        self.current_hash.update(member)
        self.current_size += len(member)
        self.current_records += len(rows)
        self.current_uncompressed += sum(len(row) for row in rows)

    def write(self, record: Mapping[str, Any]) -> None:
        row = canonical_json_bytes(record)
        self.pending.append(row)
        self.pending_bytes += len(row)
        if self.pending_bytes >= BUFFER_BYTES:
            self.flush()

    def flush(self) -> None:
        self._append(self.pending)
        self.pending = []
        self.pending_bytes = 0

    def close(self) -> tuple[R303ShardIdentity, ...]:
        self.flush()
        self._close()
        # Rev6 deliberately permits a fully audited day to contribute zero
        # materialized rows when neither acting seat has card 743.  The day
        # still receives a complete marker; an empty shard is never emitted.
        return tuple(self.identities)


class _ElmoMemoryLease:
    """The same one-file exclusive Elmo phase lease used by the census runner."""

    def __init__(self, path: Path = DEFAULT_LEASE_PATH) -> None:
        self.path = path
        self.stream: Any | None = None

    def __enter__(self) -> "_ElmoMemoryLease":
        if socket.gethostname().strip().casefold() != CANONICAL_ELMO_HOSTNAME:
            raise R303MaterializationError("r303 materialization --execute is restricted to canonical Elmo")
        if self.path.is_symlink() or not self.path.is_file():
            raise R303MaterializationError("r303 global experiment lease is not a regular file")
        self.stream = self.path.open("r+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise R303MaterializationError("another r298 memory-heavy phase holds the Elmo lease") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class _Telemetry:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.peak_rss_bytes = 0
        self.samples = 0

    def sample(self) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KiB; Elmo is Linux.  Keep the method receipt explicit.
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(usage.ru_maxrss) * 1024)
        self.samples += 1
        if self.peak_rss_bytes > EXPERIMENT_RAM_LIMIT_BYTES:
            raise R303MaterializationError("r303 materialization exceeded its 96-GiB RAM ceiling")

    def final(self) -> dict[str, Any]:
        self.sample()
        return {
            "experiment_ram_measurement_method": "linux_getrusage_ru_maxrss_single_process_under_exclusive_flock",
            "experiment_ram_peak_bytes": self.peak_rss_bytes,
            "experiment_ram_hard_limit_bytes": EXPERIMENT_RAM_LIMIT_BYTES,
            "cpu_process_or_utilization_peak": "not_sampled_per_core; bounded_single_process_worker_pool",
            "gpu_memory_peak_bytes_per_device_or_not_applicable": "not_applicable_cpu_only",
            "gpu_utilization_peak_per_device_or_not_applicable": "not_applicable_cpu_only",
            "sample_count": self.samples,
            "elapsed_seconds": round(time.monotonic() - self.started, 6),
            "process_id": os.getpid(),
            "one_memory_heavy_phase_exclusive_lease": True,
        }


def _process_pool_catalog_identity(catalog: Any) -> str:
    """Return a public, pickle-safe identity for a worker input catalog."""

    if catalog is None:
        return canonical_sha256({"kind": "no_catalog_for_scope_audit"})
    provenance = getattr(catalog, "provenance", None)
    if provenance is None or not hasattr(provenance, "to_dict"):
        raise R303MaterializationError("process-pool probe received an unsealed catalog")
    try:
        card_vocab = int(getattr(catalog, "card_vocab"))
        attack_vocab = int(getattr(catalog, "attack_vocab"))
        provenance_row = dict(provenance.to_dict())
    except (AttributeError, TypeError, ValueError) as exc:
        raise R303MaterializationError("process-pool catalog surface is malformed") from exc
    return canonical_sha256(
        {
            "kind": f"{type(catalog).__module__}.{type(catalog).__qualname__}",
            "card_vocab": card_vocab,
            "attack_vocab": attack_vocab,
            "provenance": provenance_row,
        }
    )


def _r303_process_pool_probe_worker(
    worker_index: int,
    barrier: Any,
    catalog: Any,
    raw_manifest: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    expected_raw_manifest_sha256: str,
    expected_schema_manifest_sha256: str,
    expected_catalog_identity_sha256: str,
) -> dict[str, Any]:
    """Exercise real process serialization before a whole-day pool begins."""

    if canonical_sha256(raw_manifest) != expected_raw_manifest_sha256:
        raise R303MaterializationError("process-pool raw manifest pickle identity drifted")
    if canonical_sha256(schema_manifest) != expected_schema_manifest_sha256:
        raise R303MaterializationError("process-pool schema manifest pickle identity drifted")
    if _process_pool_catalog_identity(catalog) != expected_catalog_identity_sha256:
        raise R303MaterializationError("process-pool catalog pickle identity drifted")
    try:
        barrier.wait(timeout=60)
    except (threading.BrokenBarrierError, TimeoutError) as exc:
        raise R303MaterializationError("process-pool startup did not establish every worker") from exc
    return {"worker_index": worker_index, "process_id": os.getpid()}


def probe_r303_whole_day_process_pool(
    *,
    workers: int,
    catalog: Any,
    raw_manifest: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that all requested whole-day workers start and deserialize inputs.

    This is intentionally a no-write preflight.  A manager-backed barrier
    keeps each task alive until every requested worker has started, preventing
    a short no-op task set from accidentally reporting one reused child as a
    successful 24-process pool.
    """

    bounded_day_parallel_plan_r303(workers=workers)
    raw_identity = canonical_sha256(raw_manifest)
    schema_identity = canonical_sha256(schema_manifest)
    catalog_identity = _process_pool_catalog_identity(catalog)
    manager: Any | None = None
    try:
        manager = multiprocessing.Manager()
        barrier = manager.Barrier(workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _r303_process_pool_probe_worker,
                    worker_index,
                    barrier,
                    catalog,
                    dict(raw_manifest),
                    dict(schema_manifest),
                    raw_identity,
                    schema_identity,
                    catalog_identity,
                )
                for worker_index in range(workers)
            ]
            results = [future.result() for future in futures]
    except Exception as exc:
        raise R303MaterializationError("whole-day process-pool startup/pickle probe failed") from exc
    finally:
        if manager is not None:
            manager.shutdown()
    by_index = {
        _integer(result.get("worker_index"), field="process-pool probe worker index"): _integer(
            result.get("process_id"), field="process-pool probe process id", minimum=1
        )
        for result in results
        if isinstance(result, Mapping)
    }
    if set(by_index) != set(range(workers)) or len(by_index) != workers:
        raise R303MaterializationError("process-pool probe has incomplete worker identities")
    process_ids = sorted(set(by_index.values()))
    if len(process_ids) != workers:
        raise R303MaterializationError("process-pool probe reused a child instead of starting every worker")
    return {
        "schema": R303_PROCESS_POOL_PROBE_SCHEMA,
        "status": "passed_actual_whole_day_process_startup_and_pickle",
        "worker_count": workers,
        "unique_child_process_count": len(process_ids),
        "worker_process_ids": process_ids,
        "raw_manifest_sha256": raw_identity,
        "schema_manifest_sha256": schema_identity,
        "catalog_input_identity_sha256": catalog_identity,
        "training_allowed": False,
    }


def conservative_r303_process_pool_resource_observation(
    *,
    coordinator: Mapping[str, Any],
    workers: Sequence[Mapping[str, Any]],
    requested_worker_count: int,
    startup_probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Conservatively charge each observed pool process once at peak RSS."""

    bounded_day_parallel_plan_r303(workers=requested_worker_count)
    coordinator_row = dict(_mapping(coordinator, field="pool coordinator telemetry"))
    coordinator_pid = _integer(
        coordinator_row.get("process_id"), field="pool coordinator process id", minimum=1
    )
    coordinator_peak = _integer(
        coordinator_row.get("experiment_ram_peak_bytes"),
        field="pool coordinator peak RSS",
    )
    per_process = {coordinator_pid: coordinator_peak}
    child_ids: set[int] = set()
    for index, resource_row in enumerate(workers):
        resource = _mapping(resource_row, field=f"pool worker telemetry {index}")
        process_id = _integer(
            resource.get("process_id"), field=f"pool worker process id {index}", minimum=1
        )
        peak = _integer(
            resource.get("experiment_ram_peak_bytes"),
            field=f"pool worker peak RSS {index}",
        )
        per_process[process_id] = max(per_process.get(process_id, 0), peak)
        if process_id != coordinator_pid:
            child_ids.add(process_id)
    if requested_worker_count > 1 and len(child_ids) != requested_worker_count:
        raise R303MaterializationError(
            "whole-day execution did not report exactly the requested child-process count"
        )
    if requested_worker_count == 1 and child_ids:
        raise R303MaterializationError("single-worker execution unexpectedly used a child process")
    if startup_probe is not None:
        probe = _mapping(startup_probe, field="process-pool startup probe")
        if (
            probe.get("schema"),
            probe.get("status"),
            probe.get("worker_count"),
            probe.get("unique_child_process_count"),
        ) != (
            R303_PROCESS_POOL_PROBE_SCHEMA,
            "passed_actual_whole_day_process_startup_and_pickle",
            requested_worker_count,
            requested_worker_count,
        ):
            raise R303MaterializationError("process-pool startup probe does not bind this worker count")
    aggregate_peak = sum(per_process.values())
    if aggregate_peak > EXPERIMENT_RAM_LIMIT_BYTES:
        raise R303MaterializationError("r303 conservative aggregate process RSS exceeded 96-GiB ceiling")
    result = dict(coordinator_row)
    result.update(
        {
            "experiment_ram_measurement_method": (
                "conservative_sum_of_unique_coordinator_and_whole_day_process_ru_maxrss_peaks"
            ),
            "experiment_ram_peak_bytes": aggregate_peak,
            "coordinator_process_id": coordinator_pid,
            "coordinator_process_peak_rss_bytes": coordinator_peak,
            "worker_process_count": len(child_ids),
            "worker_peak_rss_charge_bytes": sum(
                peak for process_id, peak in per_process.items() if process_id != coordinator_pid
            ),
            "per_process_peak_rss_bytes": {
                str(process_id): per_process[process_id]
                for process_id in sorted(per_process)
            },
            "requested_whole_day_worker_count": requested_worker_count,
            "process_pool_startup_probe_sha256": (
                canonical_sha256(startup_probe) if startup_probe is not None else None
            ),
            "one_memory_heavy_phase_exclusive_lease": True,
        }
    )
    return result


def _archive_for_day(raw_manifest: Mapping[str, Any], day: str) -> Mapping[str, Any]:
    for row in _rows(raw_manifest.get("archives"), field="raw archives"):
        archive = _mapping(row, field="raw archive")
        if archive.get("date") == day:
            return archive
    raise R303MaterializationError(f"raw manifest omits {day}")


def _archive_path(archive: Mapping[str, Any], archive_root: Path | None) -> Path:
    raw = archive.get("path")
    if not isinstance(raw, str) or not raw:
        raise R303MaterializationError("raw archive path is absent")
    path = Path(raw)
    return (archive_root / path.name) if archive_root is not None else path


def _episode_json_members(bundle: zipfile.ZipFile) -> list[str]:
    """Return the only raw episode members accepted by the daily exporter.

    The validated daily archive may carry its value-free ``manifest.csv``
    companion alongside episode JSON.  It is provenance only, never an
    episode or policy input.  Other auxiliary members remain a hard failure
    so a changed archive layout cannot quietly alter the audited population.
    """

    names = bundle.namelist()
    if len(names) != len(set(names)):
        raise R303MaterializationError("raw archive has duplicate member names")
    members = sorted(
        name for name in names if name.endswith(".json") and not name.endswith("/")
    )
    auxiliary = sorted(name for name in names if name not in set(members))
    if not members or auxiliary not in ([], ["manifest.csv"]):
        raise R303MaterializationError("raw archive has non-episode or missing JSON members")
    return members


def _verify_archive(archive: Mapping[str, Any], path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise R303MaterializationError(f"raw archive is unavailable: {path}")
    if path.stat().st_size != _integer(archive.get("bytes"), field="archive bytes", minimum=1):
        raise R303MaterializationError("raw archive size drifted")
    if _sha256_file(path) != _sha(archive.get("sha256"), field="archive sha256"):
        raise R303MaterializationError("raw archive digest drifted")
    try:
        with zipfile.ZipFile(path) as bundle:
            json_members = _episode_json_members(bundle)
    except zipfile.BadZipFile as exc:
        raise R303MaterializationError("raw archive is not a ZIP") from exc
    expected_count = archive.get("zip_json_member_count", archive.get("raw_zip_member_count_observed"))
    if len(json_members) != _integer(expected_count, field="archive JSON member count", minimum=1):
        raise R303MaterializationError("raw archive member count drifted")


def _partition_scope_units(
    units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build simultaneous rev6 train/validation/evaluation provenance summaries."""

    partitions = {
        "train": set(_utc_days()[:20]),
        "validation": set(_utc_days()[20:25]),
        "evaluation": set(_utc_days()[25:]),
    }
    values: dict[str, dict[str, set[str]]] = {
        name: {"source": set(), "day": set(), "group": set(), "seed": set()}
        for name in partitions
    }
    # A zero-row day is still a required audited partition.  Preserve it in
    # split provenance rather than allowing eligibility filtering to erase a
    # day from the exact 30-day evidence.
    for name, days in partitions.items():
        values[name]["day"].update(days)
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()
    for unit in units:
        day = unit["utc_day_partition"]
        partition = next(name for name, days in partitions.items() if day in days)
        compound = (unit["source_id"], day, unit["group_id"], unit["seed_id"])
        if compound in seen:
            continue
        seen.add(compound)
        values[partition]["source"].add(unit["source_id"])
        values[partition]["day"].add(day)
        values[partition]["group"].add(unit["group_id"])
        values[partition]["seed"].add(unit["seed_id"])
        counts[partition] += 1
    overlaps: dict[str, list[str]] = {}
    names = tuple(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for dimension in ("source", "day", "group", "seed"):
                overlap = values[left][dimension] & values[right][dimension]
                if overlap:
                    overlaps[f"{left}__{right}__{dimension}"] = sorted(overlap)
    if overlaps:
        raise R303MaterializationError("revision-6 scope split has source/day/group/seed overlap")
    summary = {
        name: {
            "utc_day_partitions": sorted(values[name]["day"]),
            "unit_count": counts[name],
            "source_count": len(values[name]["source"]),
            "group_count": len(values[name]["group"]),
            "seed_count": len(values[name]["seed"]),
            "source_identity_sha256": canonical_sha256(sorted(values[name]["source"])),
            "group_identity_sha256": canonical_sha256(sorted(values[name]["group"])),
            "seed_identity_sha256": canonical_sha256(sorted(values[name]["seed"])),
        }
        for name in names
    }
    return summary, overlaps


def _scan_revision_6_scope_day_worker(
    output_root: str,
    day: str,
    archive: Mapping[str, Any],
    archive_root: str | None,
) -> dict[str, Any]:
    """Scan one disjoint UTC archive for the rev6 same-seat predicate.

    The parent holds the one experiment-wide lease.  Workers never acquire it
    themselves: their only mutable output is the unique per-day decision
    index below ``output_root``.  Returning small day summaries keeps the
    deterministic merge in the parent while allowing the expensive ZIP/JSON
    traversal to use Elmo's CPU cores.
    """

    telemetry = _Telemetry()
    archive_path = _archive_path(
        archive, Path(archive_root) if archive_root is not None else None
    )
    _verify_archive(archive, archive_path)
    day_episode_count = day_decisions = day_included = 0
    day_nonqualifying = day_malformed = 0
    seen_groups: set[str] = set()
    units: list[dict[str, str]] = []
    variant_games: Counter[str] = Counter()
    variant_decisions: Counter[str] = Counter()
    day_rows_path = Path(output_root) / f"day={day}.eligible-decision-ids.jsonl"

    try:
        with day_rows_path.open("x", encoding="utf-8") as rows_stream, zipfile.ZipFile(
            archive_path
        ) as bundle:
            for member in _episode_json_members(bundle):
                try:
                    raw = bundle.read(member)
                    payload = json.loads(raw)
                except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise R303MaterializationError(
                        f"cannot decode scope-audit episode {archive_path.name}:{member}"
                    ) from exc
                episode = _mapping(payload, field="scope-audit episode")
                group_id, seed_id, member_sha = _episode_source(
                    episode, archive, day=day, member=member, raw=raw
                )
                if group_id in seen_groups:
                    raise R303MaterializationError(
                        "duplicate episode group in revision-6 scope audit"
                    )
                seen_groups.add(group_id)
                coverage = recorded_episode_frame_coverage(episode)
                descriptors = stage_descriptors_from_recorded_episode(
                    episode,
                    source={
                        "source_archive_sha256": archive.get("sha256"),
                        "source_archive_date": day,
                        "source_member": member,
                        "raw_transition_target_only": True,
                    },
                )
                steps: dict[int, int] = {}
                for descriptor in descriptors:
                    source = _mapping(descriptor.get("source"), field="scope stage source")
                    steps[_integer(source.get("env_step"), field="scope env step")] = _integer(
                        source.get("acting_seat"), field="scope acting seat", minimum=0
                    )
                if len(steps) != _integer(
                    coverage.get("actor_visible_selection_frame_count"),
                    field="scope actor-visible frames",
                ):
                    raise R303MaterializationError(
                        "scope audit omitted an actor-visible decision"
                    )
                decks = _extract_decks(episode)
                # Count an eligible deck variant once per episode even when
                # both seats have that same multiset.  Both seats' decisions
                # remain independently included below.
                eligible_variant_digests_seen: set[str] = set()
                episode_had_included = False
                for step, seat in sorted(steps.items()):
                    eligible, reason, deck_digest = _acting_seat_eligibility(decks, seat)
                    day_decisions += 1
                    if not eligible:
                        if reason == "missing_or_malformed_same_acting_seat_setup_deck":
                            day_malformed += 1
                        else:
                            day_nonqualifying += 1
                        continue
                    assert deck_digest is not None
                    day_included += 1
                    episode_had_included = True
                    variant_decisions[deck_digest] += 1
                    if deck_digest not in eligible_variant_digests_seen:
                        variant_games[deck_digest] += 1
                        eligible_variant_digests_seen.add(deck_digest)
                    rows_stream.write(
                        canonical_json_bytes(
                            {
                                "schema": "poke_bot.alakazam_rule_derivative_r303_revision_6_eligible_decision/v1",
                                "decision_id": canonical_sha256(
                                    {
                                        "source": archive.get("sha256"),
                                        "member_sha256": member_sha,
                                        "group_id": group_id,
                                        "env_step": step,
                                        "acting_seat": seat,
                                        "acting_deck_multiset_sha256": deck_digest,
                                    }
                                ),
                                "source_id": archive.get("sha256"),
                                "utc_day_partition": day,
                                "group_id": group_id,
                                "seed_id": seed_id,
                                "env_step": step,
                                "acting_seat": seat,
                                "acting_deck_multiset_sha256": deck_digest,
                            }
                        ).decode("utf-8")
                    )
                if episode_had_included:
                    units.append(
                        {
                            "source_id": _sha(
                                archive.get("sha256"), field="scope archive sha"
                            ),
                            "utc_day_partition": day,
                            "group_id": group_id,
                            "seed_id": seed_id,
                        }
                    )
                day_episode_count += 1
                telemetry.sample()
    except Exception:
        # Keep an incomplete create-only per-day index visible; it cannot be
        # mistaken for a completed inventory because no parent manifest is
        # emitted when a worker fails.
        raise

    identity = {
        "relative_path": day_rows_path.name,
        "sha256": _sha256_file(day_rows_path),
        "size_bytes": day_rows_path.stat().st_size,
    }
    return {
        "day": {
            "utc_day_partition": day,
            "source_archive_sha256": archive.get("sha256"),
            "raw_episode_count": day_episode_count,
            "actor_visible_decision_count": day_decisions,
            "included_decision_count": day_included,
            "excluded_nonqualifying_decision_count": day_nonqualifying,
            "excluded_missing_or_malformed_setup_deck_decision_count": day_malformed,
            "eligible_decision_index": identity,
        },
        "units": units,
        "variant_games": dict(variant_games),
        "variant_decisions": dict(variant_decisions),
        "resource_observation": telemetry.final(),
    }


def scan_revision_6_scope_inventory(
    output_root: Path | str,
    *,
    raw_manifest: Mapping[str, Any],
    raw_manifest_path: Path | str,
    archive_root: Path | None = None,
    telemetry: _Telemetry | None = None,
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Audit row eligibility over all episodes before materializing features.

    This pass is intentionally much narrower than feature materialization: it
    never invokes a model, feature builder, target compiler, or catalog.  It
    enumerates every raw episode and every actor-visible decision, records no
    cards, and produces only hashed IDs plus exact counts needed for the
    canonical rev6 migration receipt.
    """

    try:
        validate_raw_corpus_manifest(raw_manifest)
    except Exception as exc:
        raise R303MaterializationError("scope audit requires the sealed 30-day raw predecessor") from exc
    root = Path(output_root)
    if root.exists():
        raise R303MaterializationError("revision-6 scope inventory root already exists")
    if telemetry is None:
        raise R303MaterializationError("revision-6 scope audit requires resource telemetry")
    bounded_day_parallel_plan_r303(workers=workers)
    if R303_SCOPE_FILTER_TEST_PATH.is_symlink() or not R303_SCOPE_FILTER_TEST_PATH.is_file():
        raise R303MaterializationError("revision-6 same-seat filter test source is unavailable")
    filter_source_sha256 = _sha256_file(Path(__file__))
    filter_test_sha256 = _sha256_file(R303_SCOPE_FILTER_TEST_PATH)
    root.mkdir(parents=True, exist_ok=False)
    days: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    variant_games: Counter[str] = Counter()
    variant_decisions: Counter[str] = Counter()
    total_episodes = total_decisions = included = excluded_nonqualifying = excluded_malformed = 0
    worker_resource_observations: list[Mapping[str, Any]] = []
    startup_probe: dict[str, Any] | None = None
    try:
        archives_by_day = {
            day: dict(_archive_for_day(raw_manifest, day)) for day in _utc_days()
        }
        if workers > 1:
            startup_probe = probe_r303_whole_day_process_pool(
                workers=workers,
                catalog=None,
                raw_manifest=raw_manifest,
                schema_manifest={},
            )
        if workers == 1:
            completed = [
                _scan_revision_6_scope_day_worker(
                    str(root),
                    day,
                    archives_by_day[day],
                    str(archive_root) if archive_root is not None else None,
                )
                for day in _utc_days()
            ]
        else:
            completed_by_day: dict[str, dict[str, Any]] = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _scan_revision_6_scope_day_worker,
                        str(root),
                        day,
                        archives_by_day[day],
                        str(archive_root) if archive_root is not None else None,
                    ): day
                    for day in _utc_days()
                }
                for future in as_completed(futures):
                    completed_by_day[futures[future]] = future.result()
            completed = [completed_by_day[day] for day in _utc_days()]
        for result in completed:
            day_row = _mapping(result.get("day"), field="scope day result")
            days.append(dict(day_row))
            units.extend(
                dict(_mapping(unit, field="scope eligible episode"))
                for unit in _rows(result.get("units"), field="scope eligible episodes")
            )
            variant_games.update(
                _mapping(result.get("variant_games"), field="scope variant game counts")
            )
            variant_decisions.update(
                _mapping(result.get("variant_decisions"), field="scope variant decision counts")
            )
            total_episodes += _integer(day_row.get("raw_episode_count"), field="scope day episode count")
            total_decisions += _integer(
                day_row.get("actor_visible_decision_count"), field="scope day decision count"
            )
            included += _integer(day_row.get("included_decision_count"), field="scope day included count")
            excluded_nonqualifying += _integer(
                day_row.get("excluded_nonqualifying_decision_count"),
                field="scope day nonqualifying count",
            )
            excluded_malformed += _integer(
                day_row.get("excluded_missing_or_malformed_setup_deck_decision_count"),
                field="scope day malformed count",
            )
            worker_resource = _mapping(
                result.get("resource_observation"), field="scope worker resource observation"
            )
            worker_resource_observations.append(dict(worker_resource))
    except Exception:
        # An incomplete create-only root intentionally remains visible and is
        # never mistaken for a completed migration input.
        raise
    split_summary, overlaps = _partition_scope_units(units)
    if overlaps:
        raise R303MaterializationError("scope split unexpectedly overlaps")
    resource_observation = conservative_r303_process_pool_resource_observation(
        coordinator=telemetry.final(),
        workers=worker_resource_observations,
        requested_worker_count=workers,
        startup_probe=startup_probe,
    )
    variants = {
        "schema": R303_LIST_VARIANT_STRATA_SCHEMA,
        "status": "passed_literal_card_743_acting_seat_stratification",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "required_card_id": 743,
        "eligibility_predicate": "literal_membership_of_card_id_743_in_same_acting_seat_setup_deck",
        "acting_deck_multiset_digest_algorithm": "sha256_of_utf8_canonical_json_numerically_sorted_card_id_array_with_comma_colon_separators",
        "exact_pilot_list_digest_required": False,
        "variant_strata": [{"acting_deck_multiset_sha256": digest, "game_count": variant_games[digest], "row_count": variant_decisions[digest]} for digest in sorted(variant_games)],
        "all_included_rows_have_acting_deck_contains_743": True,
        "all_included_rows_have_acting_deck_multiset_sha256": True,
        "training_allowed": False,
    }
    split = {
        "schema": R303_SCOPE_SPLIT_SCHEMA,
        "status": "passed_source_day_group_seed_disjoint_eligible_rows_only",
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "validated_30_day_raw_manifest_sha256": canonical_sha256(raw_manifest),
        "disjoint_dimensions": ["source", "utc_day_partition", "group", "seed"],
        "partitioning": split_summary, "cross_split_overlap": {},
        "all_30_utc_days_assigned_exactly_once": True,
        "source_day_group_seed_disjoint": True, "training_allowed": False,
    }
    inventory = {
        "schema": R303_SCOPE_INVENTORY_SCHEMA,
        "status": "complete_all_episode_audit_card_743_acting_seat_row_scope_pending_census_receipt",
        "goal_contract_path": R303_CONTRACT_PATH, "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION, "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "predecessor_goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "predecessor_goal_contract_sha256": R303_CONTRACT_SHA256,
        "validated_30_day_raw_manifest_path": str(Path(raw_manifest_path).resolve()),
        "validated_30_day_raw_manifest_sha256": canonical_sha256(raw_manifest),
        "raw_episode_count": total_episodes, "raw_episode_scan_count": total_episodes,
        "all_raw_episodes_scanned": True, "no_episode_deck_seat_or_archetype_prefilter": True,
        "total_actor_visible_decisions_scanned": total_decisions,
        "included_actor_visible_decisions": included,
        "excluded_nonqualifying_acting_seat_decisions": excluded_nonqualifying,
        "excluded_missing_or_malformed_setup_deck_decisions": excluded_malformed,
        "both_qualifying_seats_handled_independently": True,
        "opponent_nonqualifying_decisions_included": False,
        "all_included_rows_have_acting_deck_contains_743": True,
        "all_included_rows_have_acting_deck_multiset_sha256": True,
        "no_included_row_requires_exact_pilot_digest": True,
        "same_seat_card_743_filter_source_sha256": filter_source_sha256,
        "same_seat_card_743_filter_test_sha256": filter_test_sha256,
        "process_pool_startup_probe": startup_probe,
        "days": days,
        "excluded_row_reason_counts": {
            "same_acting_seat_setup_deck_lacks_card_743": excluded_nonqualifying,
            "missing_or_malformed_same_acting_seat_setup_deck": excluded_malformed,
        },
        "list_variant_strata_manifest_sha256": canonical_sha256(variants),
        "source_day_group_disjoint_split_manifest_sha256": canonical_sha256(split),
        "resource_observation": resource_observation,
        "training_allowed": False,
    }
    if total_decisions != included + excluded_nonqualifying + excluded_malformed:
        raise R303MaterializationError("scope inventory decision accounting does not partition all actor-visible frames")
    _write_create_only_json(root / "LIST_VARIANT_STRATA_MANIFEST.json", variants)
    _write_create_only_json(root / "SPLIT_MANIFEST.json", split)
    _write_create_only_json(root / "SCOPE_INVENTORY.json", inventory)
    return inventory, variants, split


def build_revision_6_scope_migration_from_inventory(
    scope_root: Path | str,
    *,
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any],
    all_episode_collision_census_receipt_path: Path | str,
    predecessor_corpus_and_refeaturization_receipt_sha256s: Sequence[str],
    sealed_at_utc: str,
) -> dict[str, Any]:
    """Convert a complete audit-only scope inventory into the canonical receipt.

    The all-episode census receipt is reopened here.  Its canonical parsed-
    payload digest fills the closed migration field shared with the r5 bridge,
    while its physical file digest is retained separately in the reused-
    artifact inventory.  Callers cannot satisfy either domain with a copied
    digest string.  Rev5 freeze bytes are immutable predecessor evidence only.
    """

    root = Path(scope_root)
    inventory = _read_json(root / "SCOPE_INVENTORY.json", field="revision-6 scope inventory")
    variants = _read_json(root / "LIST_VARIANT_STRATA_MANIFEST.json", field="revision-6 list strata")
    split = _read_json(root / "SPLIT_MANIFEST.json", field="revision-6 split manifest")
    if inventory.get("list_variant_strata_manifest_sha256") != canonical_sha256(variants):
        raise R303MaterializationError("scope inventory list-variant identity drifted")
    if inventory.get("source_day_group_disjoint_split_manifest_sha256") != canonical_sha256(split):
        raise R303MaterializationError("scope inventory split identity drifted")
    census_path = Path(all_episode_collision_census_receipt_path)
    census = _read_json(census_path, field="all-episode collision census receipt")
    census_file_sha = _sha256_file(census_path)
    census_canonical_sha = canonical_sha256(census)
    if census.get("goal_revision") not in {R303_GOAL_REVISION, REV6_GOAL_REVISION}:
        raise R303MaterializationError("all-episode census receipt has an unsupported authority revision")
    frozen = _mapping(schema_manifest, field="revision-5 frozen schema")
    zero = _mapping(zero_bypass_receipt, field="revision-5 zero-bypass receipt")
    freeze = _mapping(schema_freeze_receipt, field="revision-5 schema-freeze receipt")
    if zero.get("frozen_schema_manifest_sha256") != canonical_sha256(frozen):
        raise R303MaterializationError("zero-bypass predecessor does not bind frozen schema")
    if freeze.get("zero_bypass_receipt_sha256") != canonical_sha256(zero):
        raise R303MaterializationError("schema-freeze predecessor does not bind zero-bypass")
    excluded = _mapping(inventory.get("excluded_row_reason_counts"), field="scope excluded reasons")
    split_sha = canonical_sha256(split)
    reusable = {
        "revision_5_frozen_schema_manifest": canonical_sha256(frozen),
        "revision_5_zero_bypass_receipt": canonical_sha256(zero),
        "revision_5_schema_freeze_receipt": canonical_sha256(freeze),
        "all_episode_raw_manifest": _sha(inventory.get("validated_30_day_raw_manifest_sha256"), field="raw manifest sha"),
        REV7_CENSUS_PHYSICAL_INVENTORY_KEY: census_file_sha,
    }
    rematerialized = {
        "revision_6_scope_inventory": canonical_sha256(inventory),
        "revision_6_list_variant_strata": canonical_sha256(variants),
        "revision_6_scope_split_manifest": split_sha,
        "revision_6_same_seat_card_743_filter_source": _sha(
            inventory.get("same_seat_card_743_filter_source_sha256"),
            field="scope filter source sha",
        ),
        "revision_6_same_seat_card_743_filter_test": _sha(
            inventory.get("same_seat_card_743_filter_test_sha256"),
            field="scope filter test sha",
        ),
    }
    try:
        return build_corpus_scope_migration_receipt(
            validated_30_day_raw_manifest_path=str(inventory["validated_30_day_raw_manifest_path"]),
            validated_30_day_raw_manifest_sha256=_sha(inventory.get("validated_30_day_raw_manifest_sha256"), field="raw manifest sha"),
            raw_episode_count=_integer(inventory.get("raw_episode_count"), field="raw episode count", minimum=1),
            raw_episode_scan_count=_integer(inventory.get("raw_episode_scan_count"), field="raw episode scan count", minimum=1),
            all_episode_collision_census_receipt_sha256=census_canonical_sha,
            feature_schema_sha256=_sha(frozen.get("feature_schema_sha256"), field="feature schema sha"),
            target_schema_sha256=_sha(frozen.get("target_schema_sha256"), field="target schema sha"),
            checklist_provenance_schema_sha256=_sha(frozen.get("checklist_provenance_schema_sha256"), field="checklist schema sha"),
            zero_bypass_receipt_sha256=canonical_sha256(zero),
            predecessor_schema_freeze_receipt_sha256=canonical_sha256(freeze),
            predecessor_corpus_and_refeaturization_receipt_sha256s=predecessor_corpus_and_refeaturization_receipt_sha256s,
            total_actor_visible_decisions_scanned=_integer(inventory.get("total_actor_visible_decisions_scanned"), field="scanned decisions"),
            included_actor_visible_decisions=_integer(inventory.get("included_actor_visible_decisions"), field="included decisions"),
            excluded_nonqualifying_acting_seat_decisions=_integer(excluded.get("same_acting_seat_setup_deck_lacks_card_743"), field="excluded nonqualifying"),
            excluded_missing_or_malformed_setup_deck_decisions=_integer(excluded.get("missing_or_malformed_same_acting_seat_setup_deck"), field="excluded malformed"),
            included_row_manifest_sha256=canonical_sha256(inventory),
            excluded_row_reason_counts_sha256=canonical_sha256(excluded),
            list_variant_strata_manifest_sha256=canonical_sha256(variants),
            source_day_group_disjoint_split_manifest_sha256s={"train": split_sha, "validation": split_sha, "evaluation": split_sha},
            reused_artifact_sha256_inventory=reusable,
            filtered_or_rematerialized_artifact_sha256_inventory=rematerialized,
            sealed_at_utc=sealed_at_utc,
        )
    except Exception as exc:
        raise R303MaterializationError("canonical revision-6 migration builder rejected scope evidence") from exc


def _run_payload(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": R303_RUN_SCHEMA,
        "status": "initialized_exact_30_day_diagnostic_only",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "rule_derivative_gateway_sha256": REV6_GOAL_GATEWAY_SHA256,
        "predecessor_goal_contract_sha256": R303_CONTRACT_SHA256,
        "predecessor_goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "raw_expert_corpus_manifest_sha256": inputs["raw_expert_corpus_manifest_sha256"],
        "raw_expert_corpus_receipt_sha256": inputs["raw_expert_corpus_receipt_sha256"],
        "frozen_schema_manifest_sha256": inputs["frozen_schema_manifest_sha256"],
        "zero_bypass_receipt_sha256": inputs["zero_bypass_receipt_sha256"],
        "schema_freeze_receipt_sha256": inputs["schema_freeze_receipt_sha256"],
        "materialization_preflight_sha256": canonical_sha256(inputs["preflight"]),
        "revision_6_corpus_scope_migration_receipt_sha256": canonical_sha256(inputs["corpus_scope_migration_receipt"]),
        "legacy_feature_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
        "utc_day_partitions": list(_utc_days()),
        "selected_action_target_mode": "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
        "opponent_belief_target_mode": "unavailable_masked_no_loss",
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_allowed": False,
            "inzi_authority": False,
            "production_authority": False,
        },
    }


def initialize_r303_materialization(
    output_root: Path | str,
    *, raw_manifest: Mapping[str, Any], raw_receipt: Mapping[str, Any],
    schema_manifest: Mapping[str, Any], zero_bypass_receipt: Mapping[str, Any],
    schema_freeze_receipt: Mapping[str, Any], preflight: Mapping[str, Any],
    corpus_scope_migration_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = _validate_inputs(
        raw_manifest=raw_manifest, raw_receipt=raw_receipt, schema_manifest=schema_manifest,
        zero_bypass_receipt=zero_bypass_receipt, schema_freeze_receipt=schema_freeze_receipt,
        preflight=preflight, corpus_scope_migration_receipt=corpus_scope_migration_receipt,
    )
    _assert_exact_feature_source()
    root = Path(output_root)
    if root.exists():
        raise R303MaterializationError(f"create-only materialization root already exists: {root}")
    payload = _run_payload(inputs)
    root.mkdir(parents=True, exist_ok=False)
    _write_create_only_json(root / "RUN_CONTRACT.json", payload)
    _write_create_only_json(
        root / "IN_PROGRESS.json",
        {
            "schema": R303_COMPLETE_SCHEMA,
            "status": "in_progress_no_day_completion_implied",
            "run_contract_sha256": canonical_sha256(payload),
            "create_only": True,
        },
    )
    return payload


def load_r303_materialization_run(output_root: Path | str) -> dict[str, Any]:
    run = _read_json(Path(output_root) / "RUN_CONTRACT.json", field="r303 materialization run")
    if (
        run.get("schema"), run.get("goal_contract_sha256"), run.get("goal_revision"),
        run.get("root_handoff_revision"), run.get("selected_action_target_mode"),
    ) != (
        R303_RUN_SCHEMA, REV6_CONTRACT_SHA256, REV6_GOAL_REVISION, R303_ROOT_OWNER_REVISION,
        "diagnostic_all_masks_false_until_exact_post_handoff_sealed_validator",
    ):
        raise R303MaterializationError("r303 materialization run contract drifted")
    return run


def materialize_r303_day(
    output_root: Path | str, *, day: str, raw_manifest: Mapping[str, Any],
    schema_manifest: Mapping[str, Any], catalog: Any, archive_root: Path | None = None,
    token_builder: Callable[[Mapping[str, Any], list[list[int]]], Any] | None = None,
    telemetry: _Telemetry | None = None,
) -> dict[str, Any]:
    """Materialize every factorized stage from one complete day into capped shards."""

    if day not in _utc_days():
        raise R303MaterializationError("day is outside the exact 30-day window")
    if telemetry is None:
        raise R303MaterializationError("day materialization requires resource telemetry")
    root = Path(output_root)
    run = load_r303_materialization_run(root)
    if run.get("raw_expert_corpus_manifest_sha256") != canonical_sha256(raw_manifest):
        raise R303MaterializationError("day input raw manifest does not bind the initialized run")
    if run.get("frozen_schema_manifest_sha256") != canonical_sha256(schema_manifest):
        raise R303MaterializationError("day input frozen schema does not bind the initialized run")
    if not is_public_catalog_eligible(catalog):
        raise R303MaterializationError("day materialization received a non-sealed catalog")
    day_root = root / f"day={day}"
    marker_path = day_root / "DAY_COMPLETE.json"
    if marker_path.exists():
        marker = _read_json(marker_path, field="day completion marker")
        if marker.get("run_contract_sha256") != canonical_sha256(run):
            raise R303MaterializationError("existing day marker belongs to another run")
        return marker
    if day_root.exists():
        raise R303MaterializationError("incomplete day root exists; refusing overwrite")
    archive = _archive_for_day(raw_manifest, day)
    archive_path = _archive_path(archive, archive_root)
    _verify_archive(archive, archive_path)
    day_root.mkdir(parents=False, exist_ok=False)
    writer = _ShardWriter(day_root, day=day)
    group_path = day_root / "episode_split_units.jsonl"
    episode_count = eligible_episode_count = actor_frame_count = forced_frame_count = stage_count = record_count = 0
    included_actor_decision_count = 0
    excluded_nonqualifying_decision_count = 0
    excluded_missing_setup_deck_decision_count = 0
    seen_groups: set[str] = set()
    source_members = hashlib.sha256()
    deck_distribution: Counter[str] = Counter()
    variant_game_counts: Counter[str] = Counter()
    variant_row_counts: Counter[str] = Counter()
    try:
        with group_path.open("x", encoding="utf-8") as group_stream, zipfile.ZipFile(archive_path) as bundle:
            for member in _episode_json_members(bundle):
                try:
                    raw = bundle.read(member)
                    payload = json.loads(raw)
                except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise R303MaterializationError(f"cannot decode raw episode {archive_path.name}:{member}") from exc
                episode = _mapping(payload, field="raw episode")
                group_id, seed_id, member_sha = _episode_source(episode, archive, day=day, member=member, raw=raw)
                if group_id in seen_groups:
                    raise R303MaterializationError("duplicate episode group inside validated day")
                seen_groups.add(group_id)
                source_members.update(canonical_json_bytes({"member": member, "member_sha256": member_sha, "group_id": group_id, "seed_id": seed_id}))
                coverage = recorded_episode_frame_coverage(episode)
                actor_frame_count += _integer(coverage.get("actor_visible_selection_frame_count"), field="actor frame count")
                forced_frame_count += _integer(coverage.get("forced_selection_frame_count"), field="forced frame count")
                descriptors = stage_descriptors_from_recorded_episode(
                    episode,
                    source={"source_archive_sha256": archive.get("sha256"), "source_archive_date": day, "source_member": member, "raw_transition_target_only": True},
                )
                represented_steps = {
                    _integer(_mapping(item.get("source"), field="stage source").get("env_step"), field="represented step")
                    for item in descriptors
                }
                if len(represented_steps) != coverage["actor_visible_selection_frame_count"]:
                    raise R303MaterializationError("actor-visible frame omitted from r303 materialization")
                actions = _full_actions_by_step(descriptors)
                decks = _extract_decks(episode)
                group_written = False
                target_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
                eligibility_by_seat: dict[int, tuple[bool, str, str | None]] = {}
                # Count decision frames once per environment step, not once
                # per factorized option stage.  This supports the rev6 receipt
                # independently of the number of factorized rows emitted.
                unique_steps: dict[int, int] = {}
                for stage_descriptor in descriptors:
                    stage_source = _mapping(stage_descriptor.get("source"), field="stage source")
                    unique_steps[_integer(stage_source.get("env_step"), field="environment step")] = _integer(
                        stage_source.get("acting_seat"), field="acting seat", minimum=0
                    )
                if len(unique_steps) != coverage["actor_visible_selection_frame_count"]:
                    raise R303MaterializationError("r303 decision-scope accounting drifted from all-episode coverage")
                # See the scope-audit equivalent above: both qualifying seats
                # remain independent for row eligibility, but an identical
                # deck variant appears in this game only once in game counts.
                eligible_variant_digests_counted: set[str] = set()
                for _step, seat in unique_steps.items():
                    eligible, reason, digest = _acting_seat_eligibility(decks, seat)
                    eligibility_by_seat[seat] = (eligible, reason, digest)
                    if eligible:
                        included_actor_decision_count += 1
                        assert digest is not None
                        if digest not in eligible_variant_digests_counted:
                            variant_game_counts[digest] += 1
                            eligible_variant_digests_counted.add(digest)
                    elif reason == "missing_or_malformed_same_acting_seat_setup_deck":
                        excluded_missing_setup_deck_decision_count += 1
                    else:
                        excluded_nonqualifying_decision_count += 1
                for descriptor in descriptors:
                    descriptor_source = _mapping(descriptor.get("source"), field="stage source")
                    source = _safe_source(
                        archive=archive, day=day, member=member, member_sha256=member_sha,
                        group_id=group_id, seed_id=seed_id, descriptor_source=descriptor_source,
                    )
                    eligible, exclusion_reason, deck_variant_digest = eligibility_by_seat.get(
                        source["acting_seat"],
                        (False, "missing_or_malformed_same_acting_seat_setup_deck", None),
                    )
                    # Rev6 scopes only re-featurized rows.  The all-episode
                    # audit/census happens in its separate full pass; this
                    # filter never influences raw audit enumeration.
                    if not eligible:
                        continue
                    if deck_variant_digest is None:
                        raise R303MaterializationError("eligible acting seat lacks a variant digest")
                    if not group_written:
                        group_stream.write(canonical_json_bytes({
                            "schema": R303_SPLIT_UNIT_SCHEMA,
                            "source_id": source["source_id"], "utc_day_partition": day,
                            "group_id": group_id, "seed_id": seed_id,
                        }).decode("utf-8"))
                        group_written = True
                    source["acting_deck_multiset_sha256"] = deck_variant_digest
                    deck_distribution[deck_variant_digest] += 1
                    variant_row_counts[deck_variant_digest] += 1
                    raw_observation = _mapping(descriptor.get("observation"), field="raw public observation")
                    public_observation = _public_observation(raw_observation)
                    candidates = [
                        [_integer(option, field="candidate option", minimum=0) for option in _rows(row, field="candidate")]
                        for row in _rows(descriptor.get("candidates"), field="stage candidates")
                    ]
                    chosen = _integer(descriptor.get("selected_candidate_index"), field="selected candidate", minimum=0)
                    if chosen >= len(candidates):
                        raise R303MaterializationError("selected candidate is out of range")
                    step = source["env_step"]
                    selected_action = actions.get(step)
                    if selected_action is None:
                        raise R303MaterializationError("factorized stage lacks its selected complete action")
                    representation = build_public_rule_representation(raw_observation, metadata_catalog=catalog)
                    try:
                        legacy_tokens = current_feature_token_hashes(raw_observation, candidates, token_builder=token_builder)
                    except Exception as exc:
                        raise R303MaterializationError("exact r274 legacy token ABI failed") from exc
                    if len(legacy_tokens) != len(candidates):
                        raise R303MaterializationError("legacy token/candidate alignment drifted")
                    if step not in target_cache:
                        target_cache[step] = _diagnostic_selected_target(raw_observation, selected_action, catalog=catalog)
                    selected_target, target_vectors = target_cache[step]
                    checklist = _checklist_trace(public_observation, candidates, deck=decks.get(source["acting_seat"]))
                    record = {
                        "schema": R303_RECORD_SCHEMA,
                        "record_id": canonical_sha256({
                            "raw_manifest": run["raw_expert_corpus_manifest_sha256"],
                            "source": source,
                            "public_observation": representation.public_observation_hash,
                            "stage_prefix": list(descriptor.get("stage_prefix", [])),
                        }),
                        "source": source,
                        "materialization_bindings": {
                            "raw_expert_corpus_manifest_sha256": run["raw_expert_corpus_manifest_sha256"],
                            "frozen_schema_manifest_sha256": run["frozen_schema_manifest_sha256"],
                            "zero_bypass_receipt_sha256": run["zero_bypass_receipt_sha256"],
                            "schema_freeze_receipt_sha256": run["schema_freeze_receipt_sha256"],
                            "materialization_preflight_sha256": run["materialization_preflight_sha256"],
                            "legacy_feature_source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256,
                        },
                        "public_observation": public_observation,
                        "public_rule_representation": representation.to_dict(),
                        "legacy_feature_signatures": {"source_sha256": R274_EXACT_FEATURES_SOURCE_SHA256, "candidate_token_sha256": list(legacy_tokens)},
                        "factorized_stage": {
                            "candidates": candidates, "stage_prefix": list(descriptor.get("stage_prefix", [])),
                            "selected_candidate_index": chosen, "selected_complete_action": list(selected_action),
                        },
                        "selected_action_targets": selected_target,
                        "rule_head_target_vectors": target_vectors,
                        "strict_checklist_trace": checklist,
                        "all_new_policy_routes_exact_zero_or_unwired": True,
                        "target_only_fields_never_policy_features": True,
                    }
                    writer.write(record)
                    stage_count += 1
                    record_count += 1
                    if record_count % 64 == 0:
                        telemetry.sample()
                if group_written:
                    eligible_episode_count += 1
                episode_count += 1
                telemetry.sample()
        shards = writer.close()
    except Exception:
        with contextlib.suppress(Exception):
            if writer.stream is not None:
                writer.stream.close()
        raise
    marker = {
        "schema": R303_DAY_SCHEMA,
        "status": "complete_create_only_diagnostic_targets_only",
        "run_contract_sha256": canonical_sha256(run),
        "utc_day_partition": day,
        "source_archive_sha256": archive.get("sha256"),
        "source_archive_size_bytes": archive.get("bytes"),
        "raw_episode_count": episode_count,
        "actor_visible_selection_frame_count": actor_frame_count,
        "forced_selection_frame_count": forced_frame_count,
        "factorized_stage_count": stage_count,
        "record_count": record_count,
        "episode_split_units": {"relative_path": str(group_path.relative_to(root)), "sha256": _sha256_file(group_path), "size_bytes": group_path.stat().st_size, "count": eligible_episode_count},
        "source_member_inventory_sha256": "sha256:" + source_members.hexdigest(),
        "acting_deck_distribution": dict(sorted(deck_distribution.items())),
        "revision_6_scope": {
            "eligibility_card_id": 743,
            "eligibility_predicate": "literal_membership_of_card_id_743_in_same_acting_seat_setup_deck",
            "actor_visible_decisions_scanned": actor_frame_count,
            "included_actor_visible_decisions": included_actor_decision_count,
            "excluded_nonqualifying_acting_seat_decisions": excluded_nonqualifying_decision_count,
            "excluded_missing_or_malformed_setup_deck_decisions": excluded_missing_setup_deck_decision_count,
            "both_qualifying_seats_handled_independently": True,
            "opponent_nonqualifying_decisions_included": False,
            "all_included_rows_have_acting_deck_contains_743": True,
            "all_included_rows_have_acting_deck_multiset_sha256": True,
            "no_included_row_requires_exact_pilot_digest": True,
            "list_variant_game_counts": dict(sorted(variant_game_counts.items())),
            "list_variant_row_counts": dict(sorted(variant_row_counts.items())),
        },
        "physical_shards": [item.to_dict() for item in shards],
        "physical_shard_count": len(shards),
        "all_physical_shards_at_or_below_96_mib": True,
        "all_actor_visible_and_forced_frames_scanned_before_eligibility_filter": True,
        "selected_action_training_eligible": False,
        "resource_observation": telemetry.final(),
        "runtime_authority": {"elmo_only": True, "create_only": True, "training_allowed": False, "inzi_authority": False, "production_authority": False},
    }
    _write_create_only_json(marker_path, marker)
    return marker


def _read_day_marker(root: Path, day: str, run: Mapping[str, Any]) -> dict[str, Any]:
    marker = _read_json(root / f"day={day}" / "DAY_COMPLETE.json", field=f"{day} completion marker")
    if (
        marker.get("schema"), marker.get("status"), marker.get("run_contract_sha256"),
        marker.get("utc_day_partition"), marker.get("all_physical_shards_at_or_below_96_mib"),
        marker.get("all_actor_visible_and_forced_frames_scanned_before_eligibility_filter"), marker.get("selected_action_training_eligible"),
    ) != (
        R303_DAY_SCHEMA, "complete_create_only_diagnostic_targets_only", canonical_sha256(run), day, True, True, False,
    ):
        raise R303MaterializationError(f"day {day} completion marker drifted")
    return marker


def _validate_shard(root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    relative = identity.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise R303MaterializationError("shard relative path is invalid")
    path = root / relative
    expected_size = _integer(identity.get("size_bytes"), field="shard bytes", minimum=1)
    expected_sha = _sha(identity.get("sha256"), field="shard sha")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size or expected_size > MAX_PHYSICAL_SHARD_BYTES:
        raise R303MaterializationError("shard file identity is invalid")
    if _sha256_file(path) != expected_sha:
        raise R303MaterializationError("shard digest drifted")
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = _mapping(json.loads(line), field="shard record")
                if row.get("schema") != R303_RECORD_SCHEMA:
                    raise R303MaterializationError("shard contains a foreign record schema")
                vectors = _mapping(row.get("rule_head_target_vectors"), field="rule vectors")
                if vectors.get("target_training_eligible") is not False:
                    raise R303MaterializationError("shard contains a pre-handoff trainable target")
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303MaterializationError("shard gzip stream is malformed") from exc
    if count != _integer(identity.get("record_count"), field="shard record count", minimum=1):
        raise R303MaterializationError("shard full stream record count drifted")
    return dict(identity)


def _split_manifest(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    partitions = {"train": set(_utc_days()[:20]), "validation": set(_utc_days()[20:25]), "evaluation": set(_utc_days()[25:])}
    membership = {name: {"source": set(), "day": set(), "group": set(), "seed": set()} for name in partitions}
    counts: Counter[str] = Counter()
    for day in _utc_days():
        marker = _read_day_marker(root, day, run)
        unit = _mapping(marker.get("episode_split_units"), field="episode split unit identity")
        path = root / str(unit.get("relative_path"))
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != _sha(unit.get("sha256"), field="split unit sha"):
            raise R303MaterializationError("episode split unit identity drifted")
        partition = next(name for name, days in partitions.items() if day in days)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _mapping(json.loads(line), field="episode split unit")
            if row.get("schema") != R303_SPLIT_UNIT_SCHEMA or row.get("utc_day_partition") != day:
                raise R303MaterializationError("episode split unit schema/day drifted")
            for dimension, field in (("source", "source_id"), ("group", "group_id"), ("seed", "seed_id")):
                membership[partition][dimension].add(_sha(row.get(field), field=field))
            membership[partition]["day"].add(day)
            counts[partition] += 1
    overlaps: dict[str, list[str]] = {}
    names = tuple(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            for dimension in ("source", "day", "group", "seed"):
                values = membership[left][dimension] & membership[right][dimension]
                if values:
                    overlaps[f"{left}__{right}__{dimension}"] = sorted(values)
    if overlaps:
        raise R303MaterializationError("source/day/group/seed split overlap detected")
    return {
        "schema": R303_SPLIT_SCHEMA, "status": "passed_source_day_group_seed_disjoint",
        "goal_contract_sha256": REV6_CONTRACT_SHA256, "goal_revision": REV6_GOAL_REVISION,
        "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "validated_30_day_raw_manifest_sha256": run["raw_expert_corpus_manifest_sha256"],
        "disjoint_dimensions": ["source", "utc_day_partition", "group", "seed"],
        "partitioning": {
            name: {
                "utc_day_partitions": sorted(membership[name]["day"]), "unit_count": counts[name],
                "source_count": len(membership[name]["source"]), "group_count": len(membership[name]["group"]), "seed_count": len(membership[name]["seed"]),
                "source_identity_sha256": canonical_sha256(sorted(membership[name]["source"])),
                "group_identity_sha256": canonical_sha256(sorted(membership[name]["group"])),
                "seed_identity_sha256": canonical_sha256(sorted(membership[name]["seed"])),
            } for name in names
        },
        "cross_split_overlap": {}, "all_30_utc_days_assigned_exactly_once": True,
        "source_day_group_seed_disjoint": True, "training_allowed": False,
    }


def _list_variant_strata_manifest(root: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate rev6 list-variant provenance without exposing deck contents."""

    game_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for day in _utc_days():
        marker = _read_day_marker(root, day, run)
        scope = _mapping(marker.get("revision_6_scope"), field="revision 6 day scope")
        if (
            scope.get("eligibility_card_id"),
            scope.get("eligibility_predicate"),
            scope.get("all_included_rows_have_acting_deck_contains_743"),
            scope.get("all_included_rows_have_acting_deck_multiset_sha256"),
            scope.get("no_included_row_requires_exact_pilot_digest"),
        ) != (
            743,
            "literal_membership_of_card_id_743_in_same_acting_seat_setup_deck",
            True,
            True,
            True,
        ):
            raise R303MaterializationError("day marker has an invalid revision-6 acting-seat scope")
        for digest, count in _mapping(scope.get("list_variant_game_counts"), field="variant game counts").items():
            game_counts[_sha(digest, field="variant digest")] += _integer(count, field="variant game count")
        for digest, count in _mapping(scope.get("list_variant_row_counts"), field="variant row counts").items():
            row_counts[_sha(digest, field="variant digest")] += _integer(count, field="variant row count")
    if set(game_counts) != set(row_counts):
        raise R303MaterializationError("list-variant game/row strata disagree")
    return {
        "schema": R303_LIST_VARIANT_STRATA_SCHEMA,
        "status": "passed_literal_card_743_acting_seat_stratification",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION,
        "required_card_id": 743,
        "eligibility_predicate": "literal_membership_of_card_id_743_in_same_acting_seat_setup_deck",
        "acting_deck_multiset_digest_algorithm": "sha256_of_utf8_canonical_json_numerically_sorted_card_id_array_with_comma_colon_separators",
        "exact_pilot_list_digest_required": False,
        "variant_strata": [
            {
                "acting_deck_multiset_sha256": digest,
                "game_count": game_counts[digest],
                "row_count": row_counts[digest],
            }
            for digest in sorted(game_counts)
        ],
        "all_included_rows_have_acting_deck_contains_743": True,
        "all_included_rows_have_acting_deck_multiset_sha256": True,
        "training_allowed": False,
    }


def build_revision_6_scope_migration_from_materialization(
    output_root: Path | str,
    *,
    raw_manifest_path: Path | str,
    schema_manifest: Mapping[str, Any],
    all_episode_collision_census_receipt_sha256: str,
    predecessor_corpus_and_refeaturization_receipt_sha256s: Sequence[str],
    reused_artifact_sha256_inventory: Mapping[str, str],
    filtered_or_rematerialized_artifact_sha256_inventory: Mapping[str, str],
    sealed_at_utc: str,
) -> dict[str, Any]:
    """Build the canonical rev6 receipt from measured, completed local outputs.

    This wrapper intentionally delegates the closed field inventory and all
    rev6 authority checks to the contract owner's producer.  It only derives
    counts/digests from the complete materialization markers; it never rebinds
    or mutates predecessor freeze/parity bytes.
    """

    root = Path(output_root)
    run = load_r303_materialization_run(root)
    manifest = _read_json(root / "REFEATURE_MANIFEST.json", field="r303 refeature manifest")
    split = _read_json(root / "SPLIT_MANIFEST.json", field="r303 split manifest")
    variants = _read_json(root / "LIST_VARIANT_STRATA_MANIFEST.json", field="r303 list variant strata")
    if manifest.get("source_day_group_seed_split_manifest_sha256") != canonical_sha256(split):
        raise R303MaterializationError("r303 split manifest digest does not bind refeature manifest")
    scope = _mapping(manifest.get("revision_6_corpus_scope"), field="r303 revision-6 corpus scope")
    total_raw_episodes = _integer(manifest.get("total_raw_episode_count"), field="raw episode count", minimum=1)
    excluded_reasons = {
        "same_acting_seat_setup_deck_lacks_card_743": _integer(
            scope.get("excluded_nonqualifying_acting_seat_decisions"), field="excluded nonqualifying decisions"
        ),
        "missing_or_malformed_same_acting_seat_setup_deck": _integer(
            scope.get("excluded_missing_or_malformed_setup_deck_decisions"), field="excluded malformed decisions"
        ),
    }
    split_sha = canonical_sha256(split)
    # The contract requires one identity per split.  The current materializer
    # emits one simultaneous immutable split object, so use its digest for all
    # three named projections rather than pretending three independent files
    # exist.  The semantic partitioning remains explicitly inside that object.
    named_splits = {"train": split_sha, "validation": split_sha, "evaluation": split_sha}
    frozen_schema = _mapping(schema_manifest, field="revision-5 frozen schema predecessor")
    try:
        receipt = build_corpus_scope_migration_receipt(
            validated_30_day_raw_manifest_path=str(Path(raw_manifest_path).resolve()),
            validated_30_day_raw_manifest_sha256=run["raw_expert_corpus_manifest_sha256"],
            raw_episode_count=total_raw_episodes,
            raw_episode_scan_count=total_raw_episodes,
            all_episode_collision_census_receipt_sha256=all_episode_collision_census_receipt_sha256,
            feature_schema_sha256=_sha(frozen_schema.get("feature_schema_sha256"), field="feature schema sha"),
            target_schema_sha256=_sha(
                frozen_schema.get("target_schema_sha256"), field="target schema sha"
            ),
            checklist_provenance_schema_sha256=_sha(
                frozen_schema.get("checklist_provenance_schema_sha256"), field="checklist schema sha"
            ),
            zero_bypass_receipt_sha256=run["zero_bypass_receipt_sha256"],
            predecessor_schema_freeze_receipt_sha256=run["schema_freeze_receipt_sha256"],
            predecessor_corpus_and_refeaturization_receipt_sha256s=predecessor_corpus_and_refeaturization_receipt_sha256s,
            total_actor_visible_decisions_scanned=_integer(scope.get("total_actor_visible_decisions_scanned"), field="total scanned decisions"),
            included_actor_visible_decisions=_integer(scope.get("included_actor_visible_decisions"), field="included decisions"),
            excluded_nonqualifying_acting_seat_decisions=excluded_reasons["same_acting_seat_setup_deck_lacks_card_743"],
            excluded_missing_or_malformed_setup_deck_decisions=excluded_reasons["missing_or_malformed_same_acting_seat_setup_deck"],
            included_row_manifest_sha256=canonical_sha256(manifest),
            excluded_row_reason_counts_sha256=canonical_sha256(excluded_reasons),
            list_variant_strata_manifest_sha256=canonical_sha256(variants),
            source_day_group_disjoint_split_manifest_sha256s=named_splits,
            reused_artifact_sha256_inventory=reused_artifact_sha256_inventory,
            filtered_or_rematerialized_artifact_sha256_inventory=filtered_or_rematerialized_artifact_sha256_inventory,
            sealed_at_utc=sealed_at_utc,
        )
    except Exception as exc:
        raise R303MaterializationError("canonical revision-6 scope receipt builder rejected measured evidence") from exc
    return receipt


def merge_r303_materialized_days(output_root: Path | str, *, telemetry: _Telemetry) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(output_root)
    run = load_r303_materialization_run(root)
    if any((root / name).exists() for name in ("SPLIT_MANIFEST.json", "REFEATURE_MANIFEST.json", "REFEATURE_RECEIPT.json", "COMPLETE.json")):
        raise R303MaterializationError("final r303 artifacts already exist; refusing replacement")
    split = _split_manifest(root, run)
    variants = _list_variant_strata_manifest(root, run)
    days: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    totals = Counter()
    scope_totals = Counter()
    for day in _utc_days():
        marker = _read_day_marker(root, day, run)
        rows = [_validate_shard(root, _mapping(row, field="physical shard")) for row in _rows(marker.get("physical_shards"), field="physical shards")]
        if sum(_integer(row.get("record_count"), field="shard record count", minimum=1) for row in rows) != marker.get("record_count"):
            raise R303MaterializationError("day shard records do not match marker")
        days.append({"utc_day_partition": day, "day_completion_sha256": canonical_sha256(marker), "source_archive_sha256": marker.get("source_archive_sha256"), "raw_episode_count": marker.get("raw_episode_count"), "factorized_stage_count": marker.get("factorized_stage_count"), "record_count": marker.get("record_count"), "physical_shards": rows})
        for index, row in enumerate(rows):
            shards.append({"logical_id": f"{day}:{index:05d}", "utc_day_partition": day, **row})
        totals["episodes"] += _integer(marker.get("raw_episode_count"), field="episode count")
        totals["records"] += _integer(marker.get("record_count"), field="record count")
        totals["actor_frames"] += _integer(marker.get("actor_visible_selection_frame_count"), field="actor frames")
        totals["forced_frames"] += _integer(marker.get("forced_selection_frame_count"), field="forced frames")
        scope = _mapping(marker.get("revision_6_scope"), field="revision 6 day scope")
        for total_key, source_key in (
            ("scanned_decisions", "actor_visible_decisions_scanned"),
            ("included_decisions", "included_actor_visible_decisions"),
            ("excluded_nonqualifying", "excluded_nonqualifying_acting_seat_decisions"),
            ("excluded_malformed", "excluded_missing_or_malformed_setup_deck_decisions"),
        ):
            scope_totals[total_key] += _integer(scope.get(source_key), field=source_key)
        telemetry.sample()
    if scope_totals["scanned_decisions"] != totals["actor_frames"]:
        raise R303MaterializationError("revision-6 scan count does not cover every actor-visible frame")
    if scope_totals["scanned_decisions"] != (
        scope_totals["included_decisions"]
        + scope_totals["excluded_nonqualifying"]
        + scope_totals["excluded_malformed"]
    ):
        raise R303MaterializationError("revision-6 inclusion/exclusion reconciliation failed")
    manifest = {
        "schema": R303_MANIFEST_SCHEMA, "status": "complete_30_day_create_only_diagnostic_targets_only_pending_census_gate",
        "goal_contract_path": R303_CONTRACT_PATH, "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION, "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "rule_derivative_gateway_sha256": REV6_GOAL_GATEWAY_SHA256, **{key: run[key] for key in (
            "raw_expert_corpus_manifest_sha256", "raw_expert_corpus_receipt_sha256", "frozen_schema_manifest_sha256", "zero_bypass_receipt_sha256", "schema_freeze_receipt_sha256", "materialization_preflight_sha256", "legacy_feature_source_sha256",
        )},
        "utc_day_partitions": list(_utc_days()), "days": days, "physical_shards": shards,
        "physical_shard_count": len(shards), "all_physical_shards_at_or_below_96_mib": True,
        "total_raw_episode_count": totals["episodes"], "total_refeature_record_count": totals["records"],
        "total_actor_visible_selection_frame_count": totals["actor_frames"], "total_forced_selection_frame_count": totals["forced_frames"],
        "source_day_group_seed_split_manifest_sha256": canonical_sha256(split),
        "revision_6_corpus_scope": {
            "eligibility_card_id": 743,
            "eligibility_predicate": "literal_membership_of_card_id_743_in_same_acting_seat_setup_deck",
            "total_actor_visible_decisions_scanned": scope_totals["scanned_decisions"],
            "included_actor_visible_decisions": scope_totals["included_decisions"],
            "excluded_nonqualifying_acting_seat_decisions": scope_totals["excluded_nonqualifying"],
            "excluded_missing_or_malformed_setup_deck_decisions": scope_totals["excluded_malformed"],
            "both_qualifying_seats_handled_independently": True,
            "opponent_nonqualifying_decisions_included": False,
            "list_variant_strata_manifest_sha256": canonical_sha256(variants),
        },
        "selected_action_training_eligible": False,
        "reason": "complete_collision_census_branch_adjudication_and_handoff_receipts_required_before_training",
        "runtime_authority": {"elmo_only": True, "create_only": True, "training_allowed": False, "inzi_authority": False, "production_authority": False},
    }
    receipt = {
        "schema": R303_RECEIPT_SCHEMA, "status": "passed_complete_30_day_r303_refeaturization_diagnostic_targets_only_pending_census",
        "goal_contract_path": R303_CONTRACT_PATH, "goal_contract_sha256": REV6_CONTRACT_SHA256,
        "goal_revision": REV6_GOAL_REVISION, "root_handoff_revision": R303_ROOT_OWNER_REVISION,
        "refeature_manifest_sha256": canonical_sha256(manifest),
        "validated_30_day_raw_manifest_sha256": run["raw_expert_corpus_manifest_sha256"],
        "source_day_group_seed_split_manifest_sha256": canonical_sha256(split),
        "all_30_utc_days_complete": True, "all_actor_visible_and_forced_frames_scanned_before_eligibility_filter": True,
        "selected_action_training_eligible": False, "resource_observation": telemetry.final(),
        "candidate_training_allowed": False, "bo250_allowed": False,
        "runtime_authority": {"elmo_only": True, "inzi_authority": False, "production_authority": False},
    }
    _write_create_only_json(root / "SPLIT_MANIFEST.json", split)
    _write_create_only_json(root / "LIST_VARIANT_STRATA_MANIFEST.json", variants)
    _write_create_only_json(root / "REFEATURE_MANIFEST.json", manifest)
    _write_create_only_json(root / "REFEATURE_RECEIPT.json", receipt)
    _write_create_only_json(root / "COMPLETE.json", {"schema": R303_COMPLETE_SCHEMA, "status": "complete_pending_census_gate", "run_contract_sha256": canonical_sha256(run), "refeature_manifest_sha256": canonical_sha256(manifest), "refeature_receipt_sha256": canonical_sha256(receipt), "create_only": True})
    return manifest, receipt, split


def bounded_day_parallel_plan_r303(*, workers: int) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= MAX_WHOLE_DAY_WORKERS:
        raise R303MaterializationError(
            f"r303 worker count must be 1 through {MAX_WHOLE_DAY_WORKERS}"
        )
    lanes = {str(index): [] for index in range(workers)}
    for index, day in enumerate(_utc_days()):
        lanes[str(index % workers)].append(day)
    return {
        "schema": "poke_bot.alakazam_rule_derivative_r303_bounded_day_parallel_plan/v1",
        "worker_count": workers,
        "maximum_whole_day_workers": MAX_WHOLE_DAY_WORKERS,
        "worker_execution": "bounded_whole_day_process_pool",
        "one_memory_heavy_phase_exclusive_lease": True,
        "per_lane_assignment": lanes,
        "all_days_exactly_once": (
            sorted(day for days in lanes.values() for day in days) == list(_utc_days())
        ),
        "training_allowed": False,
    }


def _materialize_r303_day_worker(
    output_root: str,
    day: str,
    raw_manifest: Mapping[str, Any],
    schema_manifest: Mapping[str, Any],
    catalog: Any,
    archive_root: str | None,
) -> dict[str, Any]:
    """Run one disjoint UTC day inside the parent-held Elmo phase lease."""

    return materialize_r303_day(
        output_root,
        day=day,
        raw_manifest=raw_manifest,
        schema_manifest=schema_manifest,
        catalog=catalog,
        archive_root=Path(archive_root) if archive_root else None,
        telemetry=_Telemetry(),
    )


def materialize_all_r303_days(
    output_root: Path | str, *, raw_manifest: Mapping[str, Any], schema_manifest: Mapping[str, Any], catalog: Any,
    archive_root: Path | None = None, workers: int = 1,
) -> list[dict[str, Any]]:
    """One leased Elmo phase with bounded processes over disjoint whole days."""

    bounded_day_parallel_plan_r303(workers=workers)
    telemetry = _Telemetry()
    root = Path(output_root)
    with _ElmoMemoryLease():
        startup_probe = (
            probe_r303_whole_day_process_pool(
                workers=workers,
                catalog=catalog,
                raw_manifest=raw_manifest,
                schema_manifest=schema_manifest,
            )
            if workers > 1
            else None
        )
        if workers == 1:
            completed = [
                materialize_r303_day(
                    output_root,
                    day=day,
                    raw_manifest=raw_manifest,
                    schema_manifest=schema_manifest,
                    catalog=catalog,
                    archive_root=archive_root,
                    telemetry=telemetry,
                )
                for day in _utc_days()
            ]
        else:
            results: dict[str, dict[str, Any]] = {}
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _materialize_r303_day_worker,
                        str(output_root),
                        day,
                        dict(raw_manifest),
                        dict(schema_manifest),
                        catalog,
                        str(archive_root) if archive_root else None,
                    ): day
                    for day in _utc_days()
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            completed = [results[day] for day in _utc_days()]
        worker_resources = [
            dict(_mapping(marker.get("resource_observation"), field="day worker resource observation"))
            for marker in completed
        ]
        resource_observation = conservative_r303_process_pool_resource_observation(
            coordinator=telemetry.final(),
            workers=worker_resources,
            requested_worker_count=workers,
            startup_probe=startup_probe,
        )
        execution_path = root / "PROCESS_POOL_EXECUTION.json"
        if execution_path.exists():
            raise R303MaterializationError(
                "whole-day process-pool execution receipt already exists; refusing replacement"
            )
        run = load_r303_materialization_run(root)
        _write_create_only_json(
            execution_path,
            {
                "schema": R303_PROCESS_POOL_EXECUTION_SCHEMA,
                "status": "passed_bounded_whole_day_process_pool",
                "run_contract_sha256": canonical_sha256(run),
                "worker_count": workers,
                "parallel_plan": bounded_day_parallel_plan_r303(workers=workers),
                "process_pool_startup_probe": startup_probe,
                "completed_day_markers_sha256": [
                    canonical_sha256(marker) for marker in completed
                ],
                "resource_observation": resource_observation,
                "training_allowed": False,
            },
        )
        return completed


__all__ = [
    "MAX_PHYSICAL_SHARD_BYTES", "R303MaterializationError", "R303_RECORD_SCHEMA",
    "R303_MANIFEST_SCHEMA", "R303_RECEIPT_SCHEMA", "bounded_day_parallel_plan_r303",
    "probe_r303_whole_day_process_pool", "conservative_r303_process_pool_resource_observation",
    "initialize_r303_materialization", "load_r303_materialization_run", "load_r303_sealed_catalog",
    "materialize_r303_day", "materialize_all_r303_days", "merge_r303_materialized_days",
    "scan_revision_6_scope_inventory", "build_revision_6_scope_migration_from_inventory",
    "write_corpus_scope_migration_receipt_create_only",
]
