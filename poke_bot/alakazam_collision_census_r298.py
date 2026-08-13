"""Offline r298 public-representation collision census.

This module is deliberately an *audit* surface, not a policy or simulator
adapter.  It can re-featurize recorded factorized legal stages with the exact
current ``features.build_option_tokens`` ABI, preserve a complete public
semantic description of each candidate, and compare that description with
sealed pinned-simulator transition evidence.

The important boundary is information-set safety.  Raw replay material may
contain a realized hidden world, but it is never used here to make an
alternative branch.  A recorded realized transition is useful inventory
evidence for the selected option only.  A complete collision conclusion
requires separately supplied, public-deterministic pinned-simulator evidence
for every candidate.  Missing evidence remains explicit and blocks a passing
receipt instead of being filled with a guessed Search rollout.

Nothing in this file starts a game, starts a search, loads a checkpoint, or
changes a learned tensor.  The command wrapper is create-only and default-off.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


R298_COLLISION_CENSUS_SCHEMA: Final = "poke_bot.alakazam_collision_census_r298/v1"
R298_COLLISION_RECORD_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_option_record/v1"
)
R298_COLLISION_REPORT_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_report/v1"
)
R298_COLLISION_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_receipt/v1"
)
R298_PHASE_A_INVENTORY_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_phase_a_inventory/v2"
)
R298_RAW_CORPUS_MANIFEST_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_manifest/v1"
)
R298_RAW_CORPUS_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_raw_expert_corpus_receipt/v1"
)
R298_FROZEN_SCHEMA_MANIFEST_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_frozen_schema_manifest/v1"
)
R298_ZERO_BYPASS_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_zero_bypass_receipt/v1"
)
R298_REFEATURED_RECORD_MANIFEST_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_refeatured_record_manifest/v1"
)
R298_REFEATURED_RECORD_SHARD_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_refeatured_record_shard/v1"
)
R298_ENGINE_EVIDENCE_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_engine_transition_evidence/v1"
)
R298_TOKEN_ABI_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_current_option_token_abi/v1"
)
R298_SEMANTIC_OPTION_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_complete_semantic_option_key/v1"
)
R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_rev5_predecessor_classification/v1"
)
R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA: Final = (
    "poke_bot.alakazam_collision_census_r298_rev5_validation_receipt/v1"
)

R298_OWNER_REVISION: Final = 298
REVISION_5_GOAL_REVISION: Final = 5
REVISION_5_ROOT_HANDOFF_REVISION: Final = 303
REVISION_4_GOAL_REVISION: Final = 4
REVISION_4_GATEWAY_SHA256: Final = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
REVISION_4_CONTRACT_SHA256: Final = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)

OWNER_GOAL_SHA256: Final = (
    "sha256:0f440fc71043b4352e6401a3187c9d582c1c5614d76e186095e0eef51017af6f"
)
RULE_DERIVATIVE_CONTRACT_SHA256: Final = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
RULE_DERIVATIVE_GATEWAY_SHA256: Final = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)
MECHANICS_ATTACHMENT_SHA256: Final = (
    "sha256:d3f06071663dde2ae7012da72b407b410c7facd06d09ab723cad05af44ddb2cb"
)
EXACT_NEW_LIST_MULTISET_SHA256: Final = (
    "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
)
R274_EXACT_FEATURES_SOURCE_SHA256: Final = (
    "sha256:ce0eb08ca74e337fe6ee1eaeb678eb42bd7f413bbfab04f8a228c3cfd3ce3db5"
)
RAW_CORPUS_START_UTC: Final = "2026-07-13"
RAW_CORPUS_END_UTC: Final = "2026-08-11"
RAW_CORPUS_EXACT_DAYS: Final = 30
RAW_CORPUS_SOURCE_RECEIPT_SHA256S: Final = (
    "sha256:e29bfe45b092e291924300b12600be7f68f3ace59a931a5edff7f3a09150d8ee",
    "sha256:85b05efac8192d43123450fc296c14fcfa55668c1b54b651037793e6e015fd59",
)
CANONICAL_R236_LIBCG_SHA256: Final = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
CANONICAL_R236_LIBCG_SIZE_BYTES: Final = 1_342_400

COLLISION_INTENTIONAL_HIDDEN: Final = "intentional_hidden_information_equivalence"
COLLISION_INTENTIONAL_PERMUTATION: Final = "intentional_permutation_equivalence"
COLLISION_SIMULATOR_LEGALITY_PROXY: Final = "simulator_legality_proxy"
COLLISION_GENUINE_PUBLIC_NONIDENTIFIABILITY: Final = (
    "genuine_public_state_option_non_identifiability"
)
COLLISION_CLASSES: Final = (
    COLLISION_INTENTIONAL_HIDDEN,
    COLLISION_INTENTIONAL_PERMUTATION,
    COLLISION_SIMULATOR_LEGALITY_PROXY,
    COLLISION_GENUINE_PUBLIC_NONIDENTIFIABILITY,
)

STATUS_INVENTORY_ONLY: Final = "inventory_only_not_a_collision_verdict"
STATUS_BLOCKED_EVIDENCE: Final = "blocked_incomplete_pinned_simulator_evidence"
STATUS_FAILED_COLLISION: Final = "failed_actionable_public_semantic_collision"
STATUS_PASSED: Final = "passed_no_actionable_public_semantic_collision"

MAX_EXACT_NUMBER: Final = 65_535
MAX_EXACT_SELECTION_COUNT: Final = 65_535


class CollisionCensusError(ValueError):
    """An r298 collision input or receipt cannot be trusted."""


class CollisionCensusFailure(CollisionCensusError):
    """A complete audit found an actionable public semantic collision."""


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _exact_int(value: object, *, field: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if not _is_exact_int(value):
        raise CollisionCensusError(f"{field} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise CollisionCensusError(f"{field} is below {minimum}")
    if maximum is not None and result > maximum:
        raise CollisionCensusError(f"{field} is above {maximum}")
    return result


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CollisionCensusError(f"{field} must be a lowercase sha256 digest")
    suffix = value[7:]
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise CollisionCensusError(f"{field} must be a lowercase sha256 digest")
    return value


def revision_5_predecessor_classification(
    consumed_predecessor_receipts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Classify every revision-4 receipt used by a fresh revision-5 artifact.

    Revision 5 deliberately permits checksum-identical schema/catalog *bytes*
    to be reused, but never lets a revision-4 receipt become a revision-5
    authority by changing two digest strings.  A new consumer must therefore
    name each such receipt and state that it is historical evidence only.
    Empty input is meaningful: the fresh artifact did not consume any older
    receipt, so no predecessor can be silently inferred later.
    """

    if not isinstance(
        consumed_predecessor_receipts,
        Sequence,
    ) or isinstance(consumed_predecessor_receipts, (str, bytes, bytearray)):
        raise CollisionCensusError("revision-5 predecessor receipt inventory must be a sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_item_fields = {
        "receipt_sha256",
        "schema",
        "classification",
        "checksum_identical_bytes_reused",
        "satisfies_revision_5_schema_freeze",
    }
    for index, raw in enumerate(consumed_predecessor_receipts):
        if not isinstance(raw, Mapping) or set(raw) != expected_item_fields:
            raise CollisionCensusError(
                f"revision-5 predecessor receipt {index} has an invalid field inventory"
            )
        digest = require_sha256(
            raw.get("receipt_sha256"),
            field=f"revision-5 predecessor receipt {index} digest",
        )
        if digest in seen:
            raise CollisionCensusError("revision-5 predecessor receipt inventory repeats a digest")
        seen.add(digest)
        schema = raw.get("schema")
        if not isinstance(schema, str) or not schema:
            raise CollisionCensusError(
                f"revision-5 predecessor receipt {index} schema is malformed"
            )
        if raw.get("classification") != "immutable_predecessor_evidence_not_revision_5_authority":
            raise CollisionCensusError(
                f"revision-5 predecessor receipt {index} is not classified as historical evidence"
            )
        if type(raw.get("checksum_identical_bytes_reused")) is not bool:
            raise CollisionCensusError(
                f"revision-5 predecessor receipt {index} checksum-reuse flag is malformed"
            )
        if raw.get("satisfies_revision_5_schema_freeze") is not False:
            raise CollisionCensusError(
                f"revision-5 predecessor receipt {index} cannot satisfy the revision-5 schema freeze"
            )
        normalized.append(
            {
                "receipt_sha256": digest,
                "schema": schema,
                "classification": "immutable_predecessor_evidence_not_revision_5_authority",
                "checksum_identical_bytes_reused": raw["checksum_identical_bytes_reused"],
                "satisfies_revision_5_schema_freeze": False,
            }
        )
    return {
        "schema": R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA,
        "predecessor_goal_revision": REVISION_4_GOAL_REVISION,
        "predecessor_gateway_sha256": REVISION_4_GATEWAY_SHA256,
        "predecessor_contract_sha256": REVISION_4_CONTRACT_SHA256,
        "revision_4_receipts_are_immutable_historical_evidence": True,
        "revision_4_receipts_satisfy_revision_5_schema_freeze": False,
        "revision_4_schema_or_catalog_bytes_reusable_only_when_checksum_identical": True,
        "consumed_predecessor_receipts": normalized,
    }


def validate_revision_5_predecessor_classification(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a rev5 artifact explicitly classifies r4 evidence."""

    expected_fields = {
        "schema",
        "predecessor_goal_revision",
        "predecessor_gateway_sha256",
        "predecessor_contract_sha256",
        "revision_4_receipts_are_immutable_historical_evidence",
        "revision_4_receipts_satisfy_revision_5_schema_freeze",
        "revision_4_schema_or_catalog_bytes_reusable_only_when_checksum_identical",
        "consumed_predecessor_receipts",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CollisionCensusError("revision-5 predecessor classification field inventory drifted")
    if (
        value.get("schema"),
        value.get("predecessor_goal_revision"),
        value.get("predecessor_gateway_sha256"),
        value.get("predecessor_contract_sha256"),
        value.get("revision_4_receipts_are_immutable_historical_evidence"),
        value.get("revision_4_receipts_satisfy_revision_5_schema_freeze"),
        value.get("revision_4_schema_or_catalog_bytes_reusable_only_when_checksum_identical"),
    ) != (
        R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA,
        REVISION_4_GOAL_REVISION,
        REVISION_4_GATEWAY_SHA256,
        REVISION_4_CONTRACT_SHA256,
        True,
        False,
        True,
    ):
        raise CollisionCensusError("revision-5 predecessor classification provenance drifted")
    receipts = value.get("consumed_predecessor_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        raise CollisionCensusError("revision-5 predecessor receipt inventory is malformed")
    return revision_5_predecessor_classification(
        [dict(row) if isinstance(row, Mapping) else row for row in receipts]
    )


def merge_revision_5_predecessor_classifications(
    *classifications: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge historical-evidence inventories from fresh rev5 inputs safely.

    A frozen schema manifest and its bypass receipt may each describe the
    same revision-4 evidence.  The final census bridge carries one deduped
    classification so later consumers cannot lose that provenance or count it
    twice.  A same-digest/different-description pair is ambiguous and fails.
    """

    by_digest: dict[str, dict[str, Any]] = {}
    for value in classifications:
        normalized = validate_revision_5_predecessor_classification(value)
        receipts = normalized["consumed_predecessor_receipts"]
        assert isinstance(receipts, list)  # canonical builder above
        for row in receipts:
            assert isinstance(row, dict)  # canonical builder above
            digest = str(row["receipt_sha256"])
            existing = by_digest.get(digest)
            if existing is not None and existing != row:
                raise CollisionCensusError(
                    "revision-5 predecessor receipt has conflicting classifications"
                )
            by_digest[digest] = row
    return revision_5_predecessor_classification(
        [by_digest[digest] for digest in sorted(by_digest)]
    )


def _canonical_value(value: Any, *, field: str = "value") -> Any:
    """Return a JSON-native, finite, deterministic value or reject it.

    The census must never stringify arbitrary runtime objects: doing so makes
    an implementation-version-dependent token or semantic key look canonical.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CollisionCensusError(f"{field} contains a non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CollisionCensusError(f"{field} has a non-string mapping key")
            result[key] = _canonical_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    raise CollisionCensusError(f"{field} is not canonical JSON data: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                _canonical_value(value),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise CollisionCensusError("value cannot be serialized canonically") from exc


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CollisionCensusError(f"cannot hash {path}") from exc
    return "sha256:" + digest.hexdigest()


def _action_indices(value: object, *, field: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CollisionCensusError(f"{field} must be a sequence of option indices")
    indices = [
        _exact_int(index, field=f"{field}[{position}]", minimum=0)
        for position, index in enumerate(value)
    ]
    if len(set(indices)) != len(indices):
        raise CollisionCensusError(f"{field} repeats an option index")
    return indices


def _select_mapping(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise CollisionCensusError("observation.select must be an object")
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes, bytearray)):
        raise CollisionCensusError("observation.select.option must be a sequence")
    return select


def _select_bounds(select: Mapping[str, Any]) -> tuple[int, int, int]:
    options = select.get("option")
    assert isinstance(options, Sequence)
    n_options = len(options)
    minimum = _exact_int(
        select.get("minCount"),
        field="select.minCount",
        minimum=0,
        maximum=MAX_EXACT_SELECTION_COUNT,
    )
    maximum = _exact_int(
        select.get("maxCount"),
        field="select.maxCount",
        minimum=0,
        maximum=MAX_EXACT_SELECTION_COUNT,
    )
    if not 0 <= minimum <= maximum <= n_options:
        raise CollisionCensusError("select minCount/maxCount are outside legal option bounds")
    return n_options, minimum, maximum


def canonical_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Use the r298 projection that owns the frozen public-information ABI.

    This must be a projection, not an older mask-with-deletions helper: raw
    replay can retain search logs, private aliases, actor deck order, and
    target-only post-transition payloads in locations an r259 sanitizer never
    knew existed.  The r298 adapter is the canonical information-set owner for
    both the derivative and this census.
    """

    if not isinstance(observation, Mapping):
        raise CollisionCensusError("observation must be an object")
    representation = _normalized_r298_public_representation(observation)
    return {
        "schema": "poke_bot.alakazam_collision_census_r298_normalized_public_observation/v1",
        "public_observation_hash": representation.public_observation_hash,
        "state": _canonical_value(representation.state, field="normalized public state"),
        "selection": _canonical_value(
            representation.selection, field="normalized public selection"
        ),
        "canonical_option_multiset_hash": representation.canonical_option_multiset_hash,
        "option_semantic_multiset": sorted(
            option.semantic_key_sha256 for option in representation.options
        ),
    }


def _normalized_r298_public_representation(observation: Mapping[str, Any]) -> Any:
    """Build the one slot-normalized r298 representation used by this audit.

    In particular, raw visible card serials are only an internal lookup aid
    inside the adapter.  They cannot participate in a public collision key:
    stable within-observation source/area/slot bindings are the required ABI.
    """

    if not isinstance(observation, Mapping):
        raise CollisionCensusError("observation must be an object")
    try:
        from .alakazam_public_rule_adapter_r298 import build_public_rule_representation

        representation = build_public_rule_representation(observation)
    except Exception as exc:
        raise CollisionCensusError("could not reproduce the r298 public observation projection") from exc
    for field in (
        "public_observation_hash",
        "semantic_token_hash",
        "canonical_option_multiset_hash",
        "state",
        "selection",
        "options",
    ):
        if not hasattr(representation, field):
            raise CollisionCensusError("r298 public representation is structurally incomplete")
    require_sha256(representation.public_observation_hash, field="r298 normalized public hash")
    return representation


def canonical_public_observation_hash(observation: Mapping[str, Any]) -> str:
    return _normalized_r298_public_representation(observation).public_observation_hash


def canonical_public_current_hash(current: Mapping[str, Any], *, actor: int) -> str:
    """Hash a post-transition ``current`` state without retaining hidden cards.

    Visualizer replay rows contain a full post-action state.  This projection is
    target-side provenance only and deliberately does not turn that full state
    into a search or policy input.
    """

    actor = _exact_int(actor, field="actor", minimum=0, maximum=1)
    if not isinstance(current, Mapping):
        raise CollisionCensusError("successor current must be an object")
    if _exact_int(
        current.get("yourIndex"),
        field="successor current.yourIndex",
        minimum=0,
        maximum=1,
    ) != actor:
        raise CollisionCensusError("successor current actor does not match transition acting seat")
    players = current.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes, bytearray)) or len(players) != 2:
        raise CollisionCensusError("successor current must contain exactly two players")
    # Reuse the exact slot-normalized representation that owns the policy
    # information set.  A successor has no decision prompt; this explicit
    # empty prompt exists only to satisfy the representation's structural
    # boundary.  It neither invents an option nor evaluates a policy.
    representation = _normalized_r298_public_representation(
        {
            "current": current,
            "select": {
                "type": "End",
                "context": "Main",
                "minCount": 0,
                "maxCount": 0,
                "option": [],
            },
        }
    )
    return canonical_sha256(
        {
            "schema": "poke_bot.alakazam_collision_census_r298_normalized_successor/v1",
            "state": _canonical_value(
                representation.state,
                field="normalized successor public state",
            ),
            "acting_seat": actor,
        }
    )


def _token_word_hash(index: Sequence[Any], value: Sequence[Any], start: int, end: int) -> str:
    if start < 0 or end < start or end > len(index) or len(index) != len(value):
        raise CollisionCensusError("SparseVector word offsets are invalid")
    exact_index: list[int] = []
    values_f32_le_hex: list[str] = []
    for position in range(start, end):
        feature_index = _exact_int(index[position], field=f"sparse index {position}", minimum=0)
        raw_value = value[position]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise CollisionCensusError(f"sparse value {position} is not numeric")
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise CollisionCensusError(f"sparse value {position} is non-finite")
        # ``pack_sparse_vectors`` turns values into torch.float32.  Hash those
        # exact bytes rather than Python's wider transient float.
        values_f32_le_hex.append(struct.pack("<f", numeric).hex())
        exact_index.append(feature_index)
    return canonical_sha256(
        {
            "schema": R298_TOKEN_ABI_SCHEMA,
            "index": exact_index,
            "value_f32_le_hex": values_f32_le_hex,
        }
    )


def current_feature_token_hashes(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    *,
    token_builder: Callable[[Mapping[str, Any], list[list[int]]], Any] | None = None,
) -> list[str]:
    """Hash the current-model token for each factorized candidate.

    The builder is injectable for static tests.  The normal path imports the
    current feature ABI lazily, so merely importing this offline audit module
    does not import the native simulator or torch.
    """

    actions = [_action_indices(candidate, field=f"candidate[{position}]") for position, candidate in enumerate(candidates)]
    if not actions:
        raise CollisionCensusError("a factorized stage must contain at least one candidate")
    if token_builder is None:
        try:
            from . import features

            token_builder = features.build_option_tokens
        except Exception as exc:
            raise CollisionCensusError("current option-token ABI is unavailable") from exc
    try:
        sparse = token_builder(observation, actions)
        index = getattr(sparse, "index")
        value = getattr(sparse, "value")
        offsets = getattr(sparse, "offset")
    except Exception as exc:
        raise CollisionCensusError("current option-token ABI rejected the public observation") from exc
    if not isinstance(index, Sequence) or isinstance(index, (str, bytes, bytearray)):
        raise CollisionCensusError("SparseVector.index is malformed")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CollisionCensusError("SparseVector.value is malformed")
    if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes, bytearray)):
        raise CollisionCensusError("SparseVector.offset is malformed")
    if len(offsets) != len(actions):
        raise CollisionCensusError("current token builder emitted the wrong word count")
    result: list[str] = []
    for word, raw_start in enumerate(offsets):
        start = _exact_int(raw_start, field=f"SparseVector.offset[{word}]", minimum=0)
        raw_end = offsets[word + 1] if word + 1 < len(offsets) else len(index)
        end = _exact_int(raw_end, field=f"SparseVector.word_end[{word}]", minimum=0)
        result.append(_token_word_hash(index, value, start, end))
    return result


def complete_semantic_option_key(
    observation: Mapping[str, Any],
    candidate_action: Sequence[int],
    *,
    stage_prefix: Sequence[int] = (),
    stable_simulator_discriminator: Mapping[str, Any] | None = None,
    normalized_representation: Any | None = None,
) -> dict[str, Any]:
    """Return a complete public semantic key for one factorized candidate.

    A raw option index, candidate ordinal, and physical-card serial are
    intentionally absent.  The r298 adapter owns all public source/target and
    attachment binding normalization.  Its locators use within-observation
    owner/area/slot/card bindings, so a globally consistent serial renumbering
    cannot split a collision group.
    """

    select = _select_mapping(observation)
    n_options, _minimum, maximum = _select_bounds(select)
    representation = (
        _normalized_r298_public_representation(observation)
        if normalized_representation is None
        else normalized_representation
    )
    for field in (
        "public_observation_hash",
        "canonical_option_multiset_hash",
        "selection",
        "options",
    ):
        if not hasattr(representation, field):
            raise CollisionCensusError("normalized representation is structurally incomplete")
    if len(representation.options) != n_options:
        raise CollisionCensusError("normalized representation no longer aligns to legal option menu")
    require_sha256(
        representation.public_observation_hash,
        field="normalized public observation hash",
    )
    require_sha256(
        representation.canonical_option_multiset_hash,
        field="normalized canonical option multiset hash",
    )
    prefix = _action_indices(stage_prefix, field="stage_prefix")
    candidate = _action_indices(candidate_action, field="candidate_action")
    if len(prefix) > maximum:
        raise CollisionCensusError("stage_prefix exceeds select.maxCount")
    if candidate[: len(prefix)] != prefix:
        raise CollisionCensusError("candidate action does not retain the factorized prefix")
    if len(candidate) not in {len(prefix), len(prefix) + 1}:
        raise CollisionCensusError("factorized candidate must append one option or STOP")
    if len(candidate) > maximum or any(index >= n_options for index in candidate):
        raise CollisionCensusError("candidate action is outside the legal option set")
    prefix_semantics = [
        _canonical_value(
            representation.options[index].semantic,
            field=f"normalized option semantic[{index}]",
        )
        for index in prefix
    ]
    candidate_semantics = [
        _canonical_value(
            representation.options[index].semantic,
            field=f"normalized option semantic[{index}]",
        )
        for index in candidate
    ]
    key: dict[str, Any] = {
        "schema": R298_SEMANTIC_OPTION_SCHEMA,
        "normalized_public_observation_hash": representation.public_observation_hash,
        "normalized_selection": _canonical_value(
            representation.selection,
            field="normalized selection",
        ),
        "normalized_canonical_option_multiset_hash": representation.canonical_option_multiset_hash,
        "factorized_stage": {
            "kind": "stop" if len(candidate) == len(prefix) else "append",
            "prefix_semantics": prefix_semantics,
            "candidate_semantics": candidate_semantics,
        },
    }
    if stable_simulator_discriminator is not None:
        # The adapter has already placed a verified opaque discriminator in
        # each normalized option *only* where identical semantic rows need one.
        # An external raw payload could smuggle serials or ordinals back into a
        # grouping key, so this audit surface rejects it rather than merging it.
        raise CollisionCensusError(
            "external simulator discriminator is forbidden; use r298 normalized option semantics"
        )
    return key


def semantic_option_key_sha256(key: Mapping[str, Any]) -> str:
    if key.get("schema") != R298_SEMANTIC_OPTION_SCHEMA:
        raise CollisionCensusError("semantic option key schema drifted")
    return canonical_sha256(key)


def action_key_sha256(action: Sequence[int]) -> str:
    return canonical_sha256({"action": _action_indices(action, field="action")})


def _raw_candidate_option_audit_payload(
    observation: Mapping[str, Any],
    candidate: Sequence[int],
) -> list[Any]:
    """Return raw replay payloads solely as non-grouping audit provenance.

    Physical serials are useful for replay diagnosis, but are expressly not a
    public semantic identity.  Keep them in this named sidecar rather than
    allowing them to leak into ``complete_semantic_option_key`` or the public
    observation collision key.
    """

    select = _select_mapping(observation)
    options = select.get("option")
    assert isinstance(options, Sequence)
    return [
        _canonical_value(options[index], field=f"raw candidate option[{index}]")
        for index in candidate
    ]


def _normalise_transition_evidence(
    transition: Mapping[str, Any] | None,
    *,
    selected: bool,
) -> dict[str, Any]:
    """Normalize a transition witness without ever inventing a successor."""

    if transition is None:
        return {
            "evidence_status": "missing",
            "simulator_successor_event_chain_hash": None,
            "simulator_outcome_distribution_hash": None,
            "transition_scope": None,
            "missing_reason": "no_pinned_simulator_transition_evidence",
        }
    if not isinstance(transition, Mapping):
        raise CollisionCensusError("transition evidence must be an object")
    schema = transition.get("schema")
    if schema != R298_ENGINE_EVIDENCE_SCHEMA:
        raise CollisionCensusError("transition evidence schema drifted")
    source_kind = transition.get("source_kind")
    if source_kind not in {
        "pinned_simulator_public_deterministic_branch",
        "pinned_simulator_recorded_realized_transition",
    }:
        raise CollisionCensusError("transition evidence source_kind is not permitted")
    pinned_binary = require_sha256(
        transition.get("pinned_simulator_binary_sha256"),
        field="transition pinned simulator binary",
    )
    if pinned_binary != CANONICAL_R236_LIBCG_SHA256:
        raise CollisionCensusError("transition evidence uses a noncanonical libcg binary")
    source_receipt = require_sha256(
        transition.get("source_receipt_sha256"),
        field="transition source receipt",
    )
    scope = transition.get("transition_scope")
    if source_kind == "pinned_simulator_public_deterministic_branch":
        if scope != "public_deterministic_branch" or transition.get("hidden_branching_used") is not False:
            raise CollisionCensusError("public branch evidence lacks a no-hidden-branching attestation")
    else:
        if not selected:
            raise CollisionCensusError("a realized transition may only witness the recorded selected option")
        if scope != "realized_true_environment_trajectory":
            raise CollisionCensusError("recorded transition scope is invalid")
    successor_hash = transition.get("successor_public_observation_hash")
    if successor_hash is None and "successor_public_observation" in transition:
        successor_hash = canonical_public_observation_hash(transition["successor_public_observation"])
    if successor_hash is None and "successor_public_current" in transition:
        current = transition["successor_public_current"]
        if not isinstance(current, Mapping):
            raise CollisionCensusError("successor_public_current must be an object")
        actor = _exact_int(transition.get("acting_seat"), field="transition acting_seat", minimum=0, maximum=1)
        successor_hash = canonical_public_current_hash(current, actor=actor)
    if successor_hash is not None:
        successor_hash = require_sha256(successor_hash, field="successor public hash")
    event_chain = transition.get("public_event_chain")
    event_chain_hash = canonical_sha256(event_chain) if event_chain is not None else None
    successor_event_hash = (
        canonical_sha256(
            {
                "successor_public_observation_hash": successor_hash,
                "public_event_chain_hash": event_chain_hash,
            }
        )
        if successor_hash is not None and event_chain_hash is not None
        else None
    )
    outcome_distribution = transition.get("public_outcome_distribution")
    outcome_hash = canonical_sha256(outcome_distribution) if outcome_distribution is not None else None
    complete = (
        source_kind == "pinned_simulator_public_deterministic_branch"
        and successor_event_hash is not None
        and outcome_hash is not None
    )
    target_private = transition.get("target_only_private_world_sha256")
    if target_private is not None:
        target_private = require_sha256(
            target_private,
            field="target-only private world hash",
        )
    result = {
        "evidence_status": "complete" if complete else "partial",
        "source_kind": source_kind,
        "transition_scope": scope,
        "pinned_simulator_binary_sha256": pinned_binary,
        "source_receipt_sha256": source_receipt,
        "simulator_successor_event_chain_hash": successor_event_hash,
        "simulator_outcome_distribution_hash": outcome_hash,
        "target_only_private_world_sha256": target_private,
        "intentional_hidden_information_equivalence_attested": transition.get(
            "intentional_hidden_information_equivalence_attested", False
        ) is True,
        "intentional_permutation_equivalence_attested": transition.get(
            "intentional_permutation_equivalence_attested", False
        ) is True,
        "simulator_legality_proxy_attested": transition.get(
            "simulator_legality_proxy_attested", False
        ) is True,
    }
    if not complete:
        result["missing_reason"] = transition.get(
            "missing_reason",
            "transition lacks public successor/event-chain or outcome distribution",
        )
    return result


def build_stage_option_records(
    observation: Mapping[str, Any],
    candidates: Sequence[Sequence[int]],
    *,
    stage_prefix: Sequence[int] = (),
    selected_candidate_index: int | None = None,
    transition_by_action: Mapping[str, Mapping[str, Any]] | None = None,
    token_builder: Callable[[Mapping[str, Any], list[list[int]]], Any] | None = None,
    public_hash_provider: Callable[[Mapping[str, Any]], str] = canonical_public_observation_hash,
    source: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one auditable record per legal factorized candidate.

    ``transition_by_action`` is keyed by :func:`action_key_sha256`.  Its data
    must originate from a separately sealed pinned-engine probe; raw replay
    extraction supplies only the selected realized trajectory and never an
    alternate hidden-world branch.
    """

    if not isinstance(observation, Mapping):
        raise CollisionCensusError("observation must be an object")
    stage_candidates = [
        _action_indices(candidate, field=f"candidate[{index}]")
        for index, candidate in enumerate(candidates)
    ]
    if not stage_candidates:
        raise CollisionCensusError("factorized stage has no candidates")
    if selected_candidate_index is not None:
        selected_candidate_index = _exact_int(
            selected_candidate_index,
            field="selected candidate index",
            minimum=0,
            maximum=len(stage_candidates) - 1,
        )
    # The r298 adapter, rather than a generic raw-observation hasher, owns
    # collision grouping.  Retain the injectable argument only as a strict
    # compatibility assertion: a caller may prove it agrees, never override
    # the normalized slot-based identity with a raw-serial-dependent one.
    normalized_representation = _normalized_r298_public_representation(observation)
    public_hash = require_sha256(
        normalized_representation.public_observation_hash,
        field="canonical public observation hash",
    )
    supplied_public_hash = (
        public_hash
        if public_hash_provider is canonical_public_observation_hash
        else require_sha256(
            public_hash_provider(observation),
            field="supplied canonical public observation hash",
        )
    )
    if supplied_public_hash != public_hash:
        raise CollisionCensusError(
            "public hash provider disagrees with normalized r298 public representation"
        )
    require_sha256(
        normalized_representation.semantic_token_hash,
        field="normalized public representation semantic token hash",
    )
    require_sha256(
        normalized_representation.canonical_option_multiset_hash,
        field="normalized canonical option multiset hash",
    )
    token_hashes = current_feature_token_hashes(
        observation,
        stage_candidates,
        token_builder=token_builder,
    )
    source_surface = _canonical_value(source or {}, field="option record source")
    if not isinstance(source_surface, dict):  # pragma: no cover - guarded above
        raise CollisionCensusError("option record source must be an object")
    records: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(stage_candidates):
        selected = candidate_index == selected_candidate_index
        action_hash = action_key_sha256(candidate)
        transition = (transition_by_action or {}).get(action_hash)
        semantics = complete_semantic_option_key(
            observation,
            candidate,
            stage_prefix=stage_prefix,
            normalized_representation=normalized_representation,
        )
        transition_surface = _normalise_transition_evidence(transition, selected=selected)
        # Private-world fingerprints are purpose-limited target provenance;
        # they are deliberately not part of the model or semantic key.
        record: dict[str, Any] = {
            "schema": R298_COLLISION_RECORD_SCHEMA,
            "canonical_public_observation_hash": public_hash,
            "current_feature_token_hash": token_hashes[candidate_index],
            "legacy_current_feature_token_hash": token_hashes[candidate_index],
            "complete_semantic_option_key": semantics,
            "complete_semantic_option_key_sha256": semantic_option_key_sha256(semantics),
            "new_complete_semantic_option_key_sha256": semantic_option_key_sha256(semantics),
            "public_representation_schema": str(normalized_representation.schema),
            "normalized_public_rule_semantic_token_hash": (
                normalized_representation.semantic_token_hash
            ),
            "normalized_canonical_option_multiset_hash": (
                normalized_representation.canonical_option_multiset_hash
            ),
            # This field remains raw replay provenance only.  It can contain
            # serials, raw source aliases, and presentation details, none of
            # which participates in a collision group or semantic key.
            "complete_raw_option_payload": _raw_candidate_option_audit_payload(
                observation,
                candidate,
            ),
            "complete_raw_option_payload_audit_only": True,
            "candidate_action": candidate,
            "candidate_action_sha256": action_hash,
            "factorized_stage_prefix": _action_indices(stage_prefix, field="stage_prefix"),
            "selected_candidate": selected,
            "transition": transition_surface,
            "source": source_surface,
        }
        records.append(record)
    return records


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    if record.get("schema") != R298_COLLISION_RECORD_SCHEMA:
        raise CollisionCensusError("collision record schema drifted")
    return (
        require_sha256(
            record.get("canonical_public_observation_hash"),
            field="record public observation hash",
        ),
        require_sha256(record.get("current_feature_token_hash"), field="record token hash"),
    )


def _record_transition(record: Mapping[str, Any]) -> Mapping[str, Any]:
    transition = record.get("transition")
    if not isinstance(transition, Mapping):
        raise CollisionCensusError("record transition is malformed")
    return transition


def _collision_class(records: Sequence[Mapping[str, Any]]) -> str:
    semantic_hashes = {
        require_sha256(
            record.get("complete_semantic_option_key_sha256"),
            field="record semantic key hash",
        )
        for record in records
    }
    transitions = [_record_transition(record) for record in records]
    private_worlds = {
        value
        for transition in transitions
        if (value := transition.get("target_only_private_world_sha256")) is not None
    }
    hidden_attested = all(
        transition.get("intentional_hidden_information_equivalence_attested") is True
        for transition in transitions
    )
    if len(semantic_hashes) == 1 and len(private_worlds) > 1 and hidden_attested:
        return COLLISION_INTENTIONAL_HIDDEN
    permutation_attested = all(
        transition.get("intentional_permutation_equivalence_attested") is True
        for transition in transitions
    )
    if len(semantic_hashes) == 1 and permutation_attested:
        return COLLISION_INTENTIONAL_PERMUTATION
    legality_proxy_attested = all(
        transition.get("simulator_legality_proxy_attested") is True
        for transition in transitions
    )
    if legality_proxy_attested:
        return COLLISION_SIMULATOR_LEGALITY_PROXY
    return COLLISION_GENUINE_PUBLIC_NONIDENTIFIABILITY


def _action_change_risk(
    records: Sequence[Mapping[str, Any]],
    *,
    divergent: bool,
    incomplete: bool,
    collision_class: str,
) -> str:
    selected_count = sum(record.get("selected_candidate") is True for record in records)
    if divergent and collision_class != COLLISION_INTENTIONAL_HIDDEN:
        return "realized_selected_action" if selected_count else "potential_action_change"
    if incomplete:
        return "unproven_missing_pinned_successor_evidence"
    if collision_class == COLLISION_GENUINE_PUBLIC_NONIDENTIFIABILITY:
        return "latent_public_semantic_alias"
    return "none_observed"


def analyze_collision_records(
    records: Iterable[Mapping[str, Any]],
    *,
    decision_count: int | None = None,
    inventory_only: bool = False,
) -> dict[str, Any]:
    """Classify equal-current-token records and calculate action-change risk.

    Equal encoding is examined within the exact canonical public observation.
    A hidden-world equivalence can only be declared with an explicit typed
    attestation and distinct target-only private-world hashes; it is never
    inferred from a missing successor.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        grouped[_record_key(record)].append(record)
    groups: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    actionable_failure_count = 0
    incomplete_record_count = 0
    selected_in_collision_count = 0
    selected_divergent_count = 0
    for (public_hash, token_hash), entries in sorted(grouped.items()):
        if len(entries) < 2:
            transition = _record_transition(entries[0])
            if transition.get("evidence_status") != "complete":
                incomplete_record_count += 1
            continue
        transitions = [_record_transition(entry) for entry in entries]
        successor_hashes = {
            value
            for transition in transitions
            if (value := transition.get("simulator_successor_event_chain_hash")) is not None
        }
        outcome_hashes = {
            value
            for transition in transitions
            if (value := transition.get("simulator_outcome_distribution_hash")) is not None
        }
        incomplete = any(transition.get("evidence_status") != "complete" for transition in transitions)
        incomplete_record_count += sum(
            transition.get("evidence_status") != "complete" for transition in transitions
        )
        divergent_successor = len(successor_hashes) > 1
        divergent_outcome = len(outcome_hashes) > 1
        divergent = divergent_successor or divergent_outcome
        collision_class = _collision_class(entries)
        selected_count = sum(entry.get("selected_candidate") is True for entry in entries)
        selected_in_collision_count += selected_count
        if divergent and selected_count:
            selected_divergent_count += selected_count
        risk = _action_change_risk(
            entries,
            divergent=divergent,
            incomplete=incomplete,
            collision_class=collision_class,
        )
        # A hidden-world equivalence is intentionally non-actionable under the
        # stated information-set contract.  Any other divergent equal model
        # encoding is an r298 failure, even if someone labelled it a proxy.
        actionable_failure = divergent and collision_class != COLLISION_INTENTIONAL_HIDDEN
        actionable_failure_count += int(actionable_failure)
        class_counter[collision_class] += 1
        risk_counter[risk] += 1
        groups.append(
            {
                "canonical_public_observation_hash": public_hash,
                "current_feature_token_hash": token_hash,
                "record_count": len(entries),
                "semantic_option_key_count": len(
                    {
                        entry.get("complete_semantic_option_key_sha256")
                        for entry in entries
                    }
                ),
                "selected_candidate_record_count": selected_count,
                "known_successor_event_chain_count": len(successor_hashes),
                "known_outcome_distribution_count": len(outcome_hashes),
                "incomplete_pinned_evidence_record_count": sum(
                    transition.get("evidence_status") != "complete"
                    for transition in transitions
                ),
                "collision_class": collision_class,
                "successor_divergent": divergent_successor,
                "outcome_distribution_divergent": divergent_outcome,
                "action_change_risk": risk,
                "actionable_failure": actionable_failure,
            }
        )
    effective_decisions = decision_count if decision_count is not None else len(
        {
            canonical_sha256(
                {
                    "public": record.get("canonical_public_observation_hash"),
                    "source": record.get("source"),
                    "prefix": record.get("factorized_stage_prefix"),
                }
            )
            for record in all_records
        }
    )
    effective_decisions = _exact_int(
        effective_decisions,
        field="decision_count",
        minimum=0,
    )
    if inventory_only:
        status = STATUS_INVENTORY_ONLY
    elif actionable_failure_count:
        status = STATUS_FAILED_COLLISION
    elif incomplete_record_count:
        status = STATUS_BLOCKED_EVIDENCE
    else:
        status = STATUS_PASSED
    return {
        "schema": R298_COLLISION_REPORT_SCHEMA,
        "status": status,
        "collision_group_count": len(groups),
        "collision_record_count": sum(group["record_count"] for group in groups),
        "all_option_record_count": len(all_records),
        "decision_count": effective_decisions,
        "collision_frequency_per_decision": (
            len(groups) / effective_decisions if effective_decisions else 0.0
        ),
        "classification_frequency": {
            collision_class: class_counter.get(collision_class, 0)
            for collision_class in COLLISION_CLASSES
        },
        "action_change_risk_frequency": dict(sorted(risk_counter.items())),
        "selected_action_in_collision_record_count": selected_in_collision_count,
        "selected_action_in_divergent_collision_record_count": selected_divergent_count,
        "actionable_failure_group_count": actionable_failure_count,
        "incomplete_pinned_evidence_record_count": incomplete_record_count,
        "groups": groups,
    }


_TARGET_ONLY_PATH_TOKENS: Final = frozenset(
    {
        "deck",
        "deckorder",
        "deck_order",
        "truedeck",
        "true_deck",
        "hiddendeck",
        "hidden_deck",
        "hiddenstate",
        "hidden_state",
        "privatestate",
        "private_state",
        "prizeorder",
        "prize_order",
        "hiddenprize",
        "hidden_prize",
        "rawpayload",
        "raw_payload",
        "searchbegininput",
        "search_begin_input",
    }
)

FIELD_DIRECT_PUBLIC_LIBCG: Final = "direct_public_libcg_observation_or_option"
FIELD_IMMUTABLE_PUBLIC_CARD_CATALOG: Final = "immutable_public_card_catalog_metadata"
FIELD_TARGET_RECORDED_TRANSITION: Final = "target_derived_from_recorded_transition"
FIELD_TARGET_FRESH_SIMULATOR: Final = "target_requiring_fresh_simulator_execution"
FIELD_INTENTIONALLY_HIDDEN: Final = "intentionally_hidden_information"
FIELD_CLASSIFICATIONS: Final = (
    FIELD_DIRECT_PUBLIC_LIBCG,
    FIELD_IMMUTABLE_PUBLIC_CARD_CATALOG,
    FIELD_TARGET_RECORDED_TRANSITION,
    FIELD_TARGET_FRESH_SIMULATOR,
    FIELD_INTENTIONALLY_HIDDEN,
)
PHASE_A_RAW_REPLAY_INVENTORY_SCOPE: Final = "all_raw_replay_observations"
PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE: Final = (
    "all_masked_actor_visible_selection_observations"
)


def classify_raw_observation_path(path: Sequence[str], *, actor: int | None) -> str:
    """Classify an observed raw field into the exact Phase-A five-way ABI.

    This function only describes fields actually present in the raw episode.
    A caller that needs a simulator-only target must put an explicit catalog
    declaration in the inventory; it may not reclassify a missing raw field as
    though it was observed.
    """

    lowered = [part.casefold() for part in path]
    if any(part in _TARGET_ONLY_PATH_TOKENS for part in lowered):
        return FIELD_INTENTIONALLY_HIDDEN
    # opponent hand contents and all face-down prize identities are privileged.
    if len(lowered) >= 4 and lowered[:2] == ["current", "players"]:
        try:
            seat = int(lowered[2])
        except ValueError:
            seat = -1
        if seat in (0, 1) and len(lowered) >= 4:
            zone = lowered[3]
            # Setup-time outer observations may have ``current: null`` and
            # therefore no policy actor.  A no-actor schema inventory must
            # never infer which hand is visible, so it classifies every hand
            # identity conservatively as hidden.  It records only the path
            # and type, never a value.
            if zone == "hand" and (actor is None or seat != actor):
                return FIELD_INTENTIONALLY_HIDDEN
            if zone == "prize":
                return FIELD_INTENTIONALLY_HIDDEN
    # Trace ``current`` is an exact post-action target, not a legal policy
    # input; its fields must never be promoted just because they are retained.
    if lowered and lowered[0] in {"transition_after", "successor", "post_action"}:
        return FIELD_TARGET_RECORDED_TRANSITION
    return FIELD_DIRECT_PUBLIC_LIBCG


def _walk_schema(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    classification_path: tuple[str, ...] | None = None,
) -> Iterable[tuple[tuple[str, ...], tuple[str, ...], str]]:
    """Yield a normalized schema path and a concrete visibility path.

    Schema output intentionally collapses all sequence positions to ``[]`` so
    it stays bounded across the 30-day corpus.  Visibility classification must
    retain the concrete player-seat index, however: otherwise an opponent hand
    and an acting-player hand collapse into one apparently public path.  The
    concrete path is transient and never written to the inventory artifact.
    """

    if classification_path is None:
        classification_path = path
    if isinstance(value, Mapping):
        yield path, classification_path, "object"
        for key, item in value.items():
            if isinstance(key, str):
                yield from _walk_schema(
                    item,
                    path=(*path, key),
                    classification_path=(*classification_path, key),
                )
            else:
                yield (
                    (*path, "<non_string_key>"),
                    (*classification_path, "<non_string_key>"),
                    type(key).__name__,
                )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        yield path, classification_path, "array"
        for index, item in enumerate(value):
            yield from _walk_schema(
                item,
                path=(*path, "[]"),
                classification_path=(*classification_path, str(index)),
            )
    else:
        yield path, classification_path, type(value).__name__


def inventory_raw_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    inventory_scope: str = PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
) -> dict[str, Any]:
    """Build a bounded schema/visibility inventory before re-featurization.

    ``current: null`` is a legitimate setup-time raw outer observation.  It
    has no actor-visible policy surface, but remains important raw-schema
    evidence.  Such observations are included with a conservative hidden-hand
    classification.  Other malformed observations are counted as rejected so
    a receipt cannot quietly claim full coverage.
    """

    if not isinstance(inventory_scope, str) or not inventory_scope:
        raise CollisionCensusError("Phase A inventory scope must be a nonempty string")

    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    field_classifications: dict[str, Counter[str]] = defaultdict(Counter)
    classification_counts: Counter[str] = Counter()
    observation_count = 0
    rejected_count = 0
    actor_context_observation_count = 0
    no_actor_context_observation_count = 0
    for observation in observations:
        observation_count += 1
        if not isinstance(observation, Mapping) or "current" not in observation:
            rejected_count += 1
            continue
        current = observation.get("current")
        actor: int | None
        if current is None:
            actor = None
            no_actor_context_observation_count += 1
        elif isinstance(current, Mapping):
            try:
                actor = _exact_int(
                    current.get("yourIndex"),
                    field="observation current.yourIndex",
                    minimum=0,
                    maximum=1,
                )
            except CollisionCensusError:
                rejected_count += 1
                continue
            actor_context_observation_count += 1
        else:
            rejected_count += 1
            continue
        for path, concrete_path, value_type in _walk_schema(observation):
            rendered = ".".join(path) if path else "<root>"
            field_types[rendered][value_type] += 1
            classification = classify_raw_observation_path(concrete_path, actor=actor)
            classification_counts[classification] += 1
            field_classifications[rendered][classification] += 1
    fields = [
        {
            "path": path,
            "types": dict(sorted(types.items())),
            "classification_occurrences": dict(
                sorted(field_classifications[path].items())
            ),
        }
        for path, types in sorted(field_types.items())
    ]
    return {
        "schema": R298_PHASE_A_INVENTORY_SCHEMA,
        "inventory_scope": inventory_scope,
        "raw_observation_count": observation_count,
        "rejected_observation_count": rejected_count,
        "actor_context_observation_count": actor_context_observation_count,
        "no_actor_context_observation_count": no_actor_context_observation_count,
        "field_classification_occurrences": {
            classification: classification_counts.get(classification, 0)
            for classification in FIELD_CLASSIFICATIONS
        },
        "field_schema_sha256": canonical_sha256(fields),
        "fields": fields,
        "non_observation_field_catalog": phase_a_field_catalog(),
    }


def phase_a_field_catalog() -> list[dict[str, Any]]:
    """Return the non-observation field contract used to complete Phase A.

    Raw replay cannot demonstrate fields that only become available after a
    fresh public-deterministic simulator execution, nor can it contain the
    immutable card catalog.  Listing those surfaces explicitly prevents a
    report from quietly treating absent data as observed or supported.
    """

    return [
        {
            "field": "pinned_card_catalog.card_type/pokemon_type/stage/evolvesFrom/hp/retreat/ex/megaEx/tera/ACE_SPEC/weakness/resistance/attack_cost/damage/skill_identity",
            "classification": FIELD_IMMUTABLE_PUBLIC_CARD_CATALOG,
            "availability_requirement": "checksum-bound pinned catalog receipt",
            "policy_input_authority": "future_zero_gated_adapter_only",
        },
        {
            "field": "selected_action.adjacent_public_successor_and_prompt_chain",
            "classification": FIELD_TARGET_RECORDED_TRANSITION,
            "availability_requirement": "aligned visualizer trace row",
            "policy_input_authority": "target_only",
        },
        {
            "field": "unchosen_legal_option.successor_event_chain_and_outcome_distribution",
            "classification": FIELD_TARGET_FRESH_SIMULATOR,
            "availability_requirement": "separate pinned-libcg public-deterministic branch receipt",
            "policy_input_authority": "target_only; no true-hidden-world branch",
        },
        {
            "field": "opponent_hand_identity/deck_order/unrevealed_prize_identity",
            "classification": FIELD_INTENTIONALLY_HIDDEN,
            "availability_requirement": "separately typed belief target only when retained",
            "policy_input_authority": "forbidden",
        },
    ]


def validate_phase_a_inventory(inventory: Mapping[str, Any]) -> None:
    """Reject a receipt whose field classification is partial or ambiguous."""

    if inventory.get("schema") != R298_PHASE_A_INVENTORY_SCHEMA:
        raise CollisionCensusError("phase A inventory schema drifted")
    scope = inventory.get("inventory_scope")
    if not isinstance(scope, str) or not scope:
        raise CollisionCensusError("Phase A inventory scope is malformed")
    observation_count = _exact_int(
        inventory.get("raw_observation_count"),
        field="Phase A raw observation count",
        minimum=0,
    )
    rejected_count = _exact_int(
        inventory.get("rejected_observation_count"),
        field="Phase A rejected observation count",
        minimum=0,
    )
    actor_context_count = _exact_int(
        inventory.get("actor_context_observation_count"),
        field="Phase A actor-context observation count",
        minimum=0,
    )
    no_actor_context_count = _exact_int(
        inventory.get("no_actor_context_observation_count"),
        field="Phase A no-actor-context observation count",
        minimum=0,
    )
    if actor_context_count + no_actor_context_count + rejected_count != observation_count:
        raise CollisionCensusError("Phase A observation-context counts do not reconcile")
    if rejected_count:
        raise CollisionCensusError("Phase A inventory contains rejected raw observations")
    occurrences = inventory.get("field_classification_occurrences")
    if not isinstance(occurrences, Mapping):
        raise CollisionCensusError("Phase A inventory has no classification counts")
    if set(occurrences) != set(FIELD_CLASSIFICATIONS):
        raise CollisionCensusError("Phase A inventory classification keys are incomplete")
    for classification, count in occurrences.items():
        _exact_int(count, field=f"Phase A classification count {classification}", minimum=0)
    fields = inventory.get("fields")
    if not isinstance(fields, list):
        raise CollisionCensusError("Phase A inventory has no field schema")
    if inventory.get("field_schema_sha256") != canonical_sha256(fields):
        raise CollisionCensusError("Phase A inventory field schema digest drifted")
    summed_occurrences: Counter[str] = Counter()
    for field in fields:
        if not isinstance(field, Mapping):
            raise CollisionCensusError("Phase A inventory field row is malformed")
        if not isinstance(field.get("path"), str) or not isinstance(field.get("types"), Mapping):
            raise CollisionCensusError("Phase A inventory field row has malformed schema")
        row_occurrences = field.get("classification_occurrences")
        if not isinstance(row_occurrences, Mapping) or not set(row_occurrences).issubset(
            FIELD_CLASSIFICATIONS
        ):
            raise CollisionCensusError("Phase A inventory field classifications are malformed")
        for classification, count in row_occurrences.items():
            summed_occurrences[classification] += _exact_int(
                count,
                field=f"Phase A field classification count {classification}",
                minimum=0,
            )
    if {
        classification: summed_occurrences.get(classification, 0)
        for classification in FIELD_CLASSIFICATIONS
    } != {
        classification: occurrences[classification]
        for classification in FIELD_CLASSIFICATIONS
    }:
        raise CollisionCensusError("Phase A field classifications do not reconcile")
    catalog = inventory.get("non_observation_field_catalog")
    if not isinstance(catalog, list):
        raise CollisionCensusError("Phase A inventory has no non-observation field catalog")
    catalog_classes = {
        row.get("classification")
        for row in catalog
        if isinstance(row, Mapping)
    }
    missing = set(FIELD_CLASSIFICATIONS) - set(occurrences)
    if missing:
        raise CollisionCensusError("Phase A inventory misses a required classification")
    # Direct-public fields are observed in any valid public trace.  The other
    # four classes may have zero observed raw occurrences, but must be named in
    # the explicit catalog rather than omitted.
    if occurrences.get(FIELD_DIRECT_PUBLIC_LIBCG, 0) <= 0:
        raise CollisionCensusError("Phase A inventory did not observe public libcg fields")
    if not {
        FIELD_IMMUTABLE_PUBLIC_CARD_CATALOG,
        FIELD_TARGET_RECORDED_TRANSITION,
        FIELD_TARGET_FRESH_SIMULATOR,
        FIELD_INTENTIONALLY_HIDDEN,
    }.issubset(catalog_classes | {
        FIELD_INTENTIONALLY_HIDDEN
        if occurrences.get(FIELD_INTENTIONALLY_HIDDEN, 0) > 0
        else ""
    }):
        raise CollisionCensusError("Phase A field catalog is incomplete")


def build_raw_corpus_manifest(
    archives: Sequence[Mapping[str, Any]],
    *,
    archive_member_counts: Mapping[str, int],
    archive_bytes_actual: Mapping[str, int],
    source_manifest_provenance: Sequence[Mapping[str, Any]] = (),
    episode_deduplication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal the exact latest 30 contiguous UTC day identities.

    Every row uses the raw ZIP's own digest, size, member count and original
    source-discrepancy record.  Duplicate day, archive digest, or source
    identity is a hard error: a blended window must never look complete.
    """

    if len(archives) != RAW_CORPUS_EXACT_DAYS:
        raise CollisionCensusError("raw corpus must contain exactly thirty archives")
    normalized: list[dict[str, Any]] = []
    source_receipt_day_coverage: list[dict[str, Any]] = []
    dates: set[str] = set()
    archive_digests: set[str] = set()
    source_identities: set[tuple[str, str]] = set()
    for raw in archives:
        if not isinstance(raw, Mapping):
            raise CollisionCensusError("raw corpus archive row is malformed")
        date = raw.get("date")
        slug = raw.get("dataset_slug")
        digest = require_sha256(raw.get("sha256"), field="raw corpus archive digest")
        path = raw.get("path")
        if not isinstance(date, str) or not isinstance(slug, str) or not isinstance(path, str):
            raise CollisionCensusError("raw corpus archive identity is malformed")
        if raw.get("validated") is not True:
            raise CollisionCensusError(f"raw corpus day {date} is not validated")
        raw_source_receipts = raw.get("source_receipt_sha256s")
        if not isinstance(raw_source_receipts, Sequence) or isinstance(
            raw_source_receipts, (str, bytes, bytearray)
        ):
            raise CollisionCensusError(f"raw corpus day {date} lacks source-receipt coverage")
        source_receipts = sorted(
            {
                require_sha256(value, field=f"raw corpus source receipt {date}")
                for value in raw_source_receipts
            }
        )
        if not source_receipts or not set(source_receipts).issubset(
            set(RAW_CORPUS_SOURCE_RECEIPT_SHA256S)
        ):
            raise CollisionCensusError(f"raw corpus day {date} has an unknown source receipt")
        if date in dates or digest in archive_digests or (date, slug) in source_identities:
            raise CollisionCensusError("raw corpus has duplicate day/source/archive identity")
        member_count = _exact_int(
            archive_member_counts.get(digest),
            field=f"raw ZIP member count {date}",
            minimum=1,
        )
        actual_bytes = _exact_int(
            archive_bytes_actual.get(digest),
            field=f"raw ZIP bytes {date}",
            minimum=1,
        )
        expected_bytes = _exact_int(raw.get("bytes"), field=f"receipt ZIP bytes {date}", minimum=1)
        if actual_bytes != expected_bytes:
            raise CollisionCensusError(f"raw ZIP byte count drifted for {date}")
        validated_count = _exact_int(
            raw.get("validated_episode_count"),
            field=f"validated episode count {date}",
            minimum=1,
        )
        if member_count != validated_count:
            raise CollisionCensusError(f"raw ZIP member count disagrees with validation for {date}")
        dates.add(date)
        archive_digests.add(digest)
        source_identities.add((date, slug))
        normalized.append(
            {
                "date": date,
                "dataset_slug": slug,
                "path": path,
                "sha256": digest,
                "bytes": actual_bytes,
                "zip_json_member_count": member_count,
                "validated_episode_count": validated_count,
                "index_episode_count": _exact_int(
                    raw.get("index_episode_count"),
                    field=f"index episode count {date}",
                    minimum=1,
                ),
                "source_discrepancy": _canonical_value(
                    raw.get("source_discrepancy"),
                    field=f"source discrepancy {date}",
                ),
            }
        )
        source_receipt_day_coverage.append(
            {
                "date": date,
                "source_receipt_sha256s": source_receipts,
            }
        )
    normalized.sort(key=lambda row: row["date"])
    expected_dates = [
        f"2026-07-{day:02d}" for day in range(13, 32)
    ] + [f"2026-08-{day:02d}" for day in range(1, 12)]
    actual_dates = [row["date"] for row in normalized]
    if actual_dates != expected_dates:
        raise CollisionCensusError(
            "raw corpus must be the owner-selected contiguous 2026-07-13..2026-08-11 window"
        )
    provenance = [_canonical_value(row, field="source manifest provenance") for row in source_manifest_provenance]
    if len(provenance) != 2:
        raise CollisionCensusError("raw corpus must bind both overlapping latest20 source receipts")
    receipt_digests = {
        require_sha256(row.get("sha256"), field="source manifest provenance digest")
        for row in provenance
        if isinstance(row, Mapping)
    }
    expected_receipts = set(RAW_CORPUS_SOURCE_RECEIPT_SHA256S)
    if receipt_digests != expected_receipts:
        raise CollisionCensusError("raw corpus source receipt identities drifted")
    for row in provenance:
        if not isinstance(row, Mapping):
            raise CollisionCensusError("raw corpus source receipt provenance is malformed")
        if not isinstance(row.get("path"), str) or not row["path"]:
            raise CollisionCensusError("raw corpus source receipt provenance lacks a path")
        if row.get("status") != "ready":
            raise CollisionCensusError("raw corpus source receipt is not a ready immutable receipt")
    source_receipt_day_coverage.sort(key=lambda row: row["date"])
    if [row["date"] for row in source_receipt_day_coverage] != expected_dates:
        raise CollisionCensusError("raw corpus source receipt coverage is incomplete")
    if set().union(
        *(set(row["source_receipt_sha256s"]) for row in source_receipt_day_coverage)
    ) != expected_receipts:
        raise CollisionCensusError("raw corpus source receipt coverage omits a required receipt")
    if episode_deduplication is None:
        raise CollisionCensusError("raw corpus requires episode-level deduplication proof")
    episode_dedup = _canonical_value(episode_deduplication, field="episode deduplication")
    if not isinstance(episode_dedup, dict):
        raise CollisionCensusError("episode deduplication proof must be an object")
    if episode_dedup.get("episode_identity_algorithm") not in {
        "payload.id_plus_canonical_content_sha256",
        "trusted_existing_daily_validation_receipts_owner_authorized",
    }:
        raise CollisionCensusError("episode deduplication authority is unknown")
    required_episode_dedup = {
        "duplicate_episode_identity_count": 0,
        "duplicate_episode_id_with_distinct_content_count": 0,
        "excluded_duplicate_mapping": [],
    }
    for field, expected in required_episode_dedup.items():
        if episode_dedup.get(field) != expected:
            raise CollisionCensusError(f"episode deduplication proof {field} drifted")
    _exact_int(
        episode_dedup.get("unique_episode_identity_count"),
        field="unique episode identity count",
        minimum=1,
    )
    _exact_int(
        episode_dedup.get("unique_episode_id_count"),
        field="unique episode id count",
        minimum=1,
    )
    require_sha256(
        episode_dedup.get("episode_identity_inventory_sha256"),
        field="episode identity inventory",
    )
    total_members = sum(row["zip_json_member_count"] for row in normalized)
    if _exact_int(
        episode_dedup.get("raw_zip_member_count_observed"),
        field="observed raw ZIP member count",
        minimum=1,
    ) != total_members:
        raise CollisionCensusError("episode deduplication did not scan every raw ZIP member")
    if episode_dedup.get("unique_episode_identity_count") != total_members or episode_dedup.get(
        "unique_episode_id_count"
    ) != total_members:
        raise CollisionCensusError("episode deduplication counts do not cover all raw ZIP members")
    return {
        "schema": R298_RAW_CORPUS_MANIFEST_SCHEMA,
        "status": "sealed_create_only_raw_zip_inventory",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "window_start_utc": RAW_CORPUS_START_UTC,
        "window_end_utc": RAW_CORPUS_END_UTC,
        "distinct_utc_day_count": RAW_CORPUS_EXACT_DAYS,
        "deduplication": {
            "unique_date_count": len(dates),
            "unique_archive_sha256_count": len(archive_digests),
            "unique_date_dataset_source_count": len(source_identities),
            "duplicate_day_or_source_or_archive_permitted": False,
        },
        "archives": normalized,
        "total_bytes": sum(row["bytes"] for row in normalized),
        "total_raw_zip_json_members": total_members,
        "total_validated_episodes": sum(row["validated_episode_count"] for row in normalized),
        "source_manifest_provenance": provenance,
        "source_receipt_day_coverage": source_receipt_day_coverage,
        "episode_deduplication": episode_dedup,
        "source_discrepancy_count": sum(row["source_discrepancy"] is not None for row in normalized),
        "recollection_authorized": False,
        "archive_mutation_authorized": False,
    }


def validate_raw_corpus_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the strict 30-day manifest shape before consuming it.

    This is intentionally structural and content-addressable: callers that
    have access to the source receipts or ZIPs additionally verify those
    physical paths.  Keeping that I/O out of the pure validator makes it
    usable by downstream gates without accidentally opening an archive or
    turning a receipt validation into a collection path.
    """

    if manifest.get("schema") != R298_RAW_CORPUS_MANIFEST_SCHEMA:
        raise CollisionCensusError("raw corpus manifest schema drifted")
    if manifest.get("status") != "sealed_create_only_raw_zip_inventory":
        raise CollisionCensusError("raw corpus manifest is not a sealed inventory")
    if (
        manifest.get("owner_revision"),
        manifest.get("goal_revision"),
        manifest.get("root_handoff_revision"),
        manifest.get("rule_derivative_gateway_sha256"),
        manifest.get("rule_derivative_contract_sha256"),
    ) != (
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
    ):
        raise CollisionCensusError("raw corpus manifest revision-5 authority drifted")
    predecessor = validate_revision_5_predecessor_classification(
        manifest.get("revision_5_predecessor_classification")
    )
    if predecessor != revision_5_predecessor_classification():
        raise CollisionCensusError(
            "raw corpus manifest may not consume revision-4 receipts as current authority"
        )
    if (
        manifest.get("window_start_utc"),
        manifest.get("window_end_utc"),
        manifest.get("distinct_utc_day_count"),
    ) != (RAW_CORPUS_START_UTC, RAW_CORPUS_END_UTC, RAW_CORPUS_EXACT_DAYS):
        raise CollisionCensusError("raw corpus manifest does not bind the exact 30-day window")
    dedupe = manifest.get("deduplication")
    if not isinstance(dedupe, Mapping) or any(
        dedupe.get(field) != RAW_CORPUS_EXACT_DAYS
        for field in (
            "unique_date_count",
            "unique_archive_sha256_count",
            "unique_date_dataset_source_count",
        )
    ) or dedupe.get("duplicate_day_or_source_or_archive_permitted") is not False:
        raise CollisionCensusError("raw corpus manifest deduplication proof is incomplete")
    archives = manifest.get("archives")
    if not isinstance(archives, Sequence) or isinstance(archives, (str, bytes, bytearray)):
        raise CollisionCensusError("raw corpus manifest archives are malformed")
    expected_dates = [
        f"2026-07-{day:02d}" for day in range(13, 32)
    ] + [f"2026-08-{day:02d}" for day in range(1, 12)]
    if len(archives) != RAW_CORPUS_EXACT_DAYS:
        raise CollisionCensusError("raw corpus manifest lacks exactly thirty archive rows")
    seen_digests: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    total_members = 0
    total_bytes = 0
    total_validated = 0
    for index, raw in enumerate(archives):
        if not isinstance(raw, Mapping):
            raise CollisionCensusError("raw corpus manifest archive row is malformed")
        expected_keys = {
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
        if set(raw) != expected_keys:
            raise CollisionCensusError("raw corpus manifest archive row fields drifted")
        if raw.get("date") != expected_dates[index]:
            raise CollisionCensusError("raw corpus manifest dates are incomplete or noncontiguous")
        slug = raw.get("dataset_slug")
        path = raw.get("path")
        if not isinstance(slug, str) or not slug or not isinstance(path, str) or not path:
            raise CollisionCensusError("raw corpus manifest archive identity is malformed")
        digest = require_sha256(raw.get("sha256"), field="raw corpus manifest archive digest")
        if digest in seen_digests or (str(raw["date"]), slug) in seen_sources:
            raise CollisionCensusError("raw corpus manifest archive/source identity repeats")
        seen_digests.add(digest)
        seen_sources.add((str(raw["date"]), slug))
        byte_count = _exact_int(raw.get("bytes"), field="raw corpus manifest archive bytes", minimum=1)
        member_count = _exact_int(
            raw.get("zip_json_member_count"),
            field="raw corpus manifest member count",
            minimum=1,
        )
        validated_count = _exact_int(
            raw.get("validated_episode_count"),
            field="raw corpus manifest validated count",
            minimum=1,
        )
        _exact_int(raw.get("index_episode_count"), field="raw corpus manifest index count", minimum=1)
        if member_count != validated_count:
            raise CollisionCensusError("raw corpus manifest member/validated count differs")
        total_bytes += byte_count
        total_members += member_count
        total_validated += validated_count
    if (
        manifest.get("total_bytes"),
        manifest.get("total_raw_zip_json_members"),
        manifest.get("total_validated_episodes"),
    ) != (total_bytes, total_members, total_validated):
        raise CollisionCensusError("raw corpus manifest totals do not match its archive inventory")
    provenance = manifest.get("source_manifest_provenance")
    if not isinstance(provenance, Sequence) or isinstance(provenance, (str, bytes, bytearray)):
        raise CollisionCensusError("raw corpus manifest source provenance is malformed")
    provenance_by_digest: dict[str, Mapping[str, Any]] = {}
    for raw in provenance:
        if not isinstance(raw, Mapping):
            raise CollisionCensusError("raw corpus manifest source provenance row is malformed")
        digest = require_sha256(raw.get("sha256"), field="source provenance digest")
        if digest in provenance_by_digest or not isinstance(raw.get("path"), str) or not raw["path"]:
            raise CollisionCensusError("raw corpus manifest source provenance is ambiguous")
        if raw.get("status") != "ready":
            raise CollisionCensusError("raw corpus source receipt is not ready")
        provenance_by_digest[digest] = raw
    expected_receipts = set(RAW_CORPUS_SOURCE_RECEIPT_SHA256S)
    if set(provenance_by_digest) != expected_receipts:
        raise CollisionCensusError("raw corpus manifest source receipt set drifted")
    coverage = manifest.get("source_receipt_day_coverage")
    if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes, bytearray)):
        raise CollisionCensusError("raw corpus manifest source receipt day coverage is malformed")
    if len(coverage) != RAW_CORPUS_EXACT_DAYS:
        raise CollisionCensusError("raw corpus manifest source receipt coverage has wrong day count")
    covered_receipts: set[str] = set()
    for index, raw in enumerate(coverage):
        if not isinstance(raw, Mapping) or raw.get("date") != expected_dates[index]:
            raise CollisionCensusError("raw corpus source receipt coverage is noncontiguous")
        values = raw.get("source_receipt_sha256s")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise CollisionCensusError("raw corpus source receipt coverage is malformed")
        row_receipts = [require_sha256(value, field="source receipt day coverage") for value in values]
        if not row_receipts or len(set(row_receipts)) != len(row_receipts) or not set(row_receipts).issubset(expected_receipts):
            raise CollisionCensusError("raw corpus source receipt coverage is ambiguous")
        covered_receipts.update(row_receipts)
    if covered_receipts != expected_receipts:
        raise CollisionCensusError("raw corpus source receipt coverage misses a required provenance source")
    episode_dedup = manifest.get("episode_deduplication")
    if not isinstance(episode_dedup, Mapping) or (
        episode_dedup.get("duplicate_episode_identity_count"),
        episode_dedup.get("duplicate_episode_id_with_distinct_content_count"),
        episode_dedup.get("excluded_duplicate_mapping"),
        episode_dedup.get("raw_zip_member_count_observed"),
    ) != (
        0,
        0,
        [],
        total_members,
    ):
        raise CollisionCensusError("raw corpus manifest episode deduplication proof is incomplete")
    if episode_dedup.get("episode_identity_algorithm") not in {
        "payload.id_plus_canonical_content_sha256",
        "trusted_existing_daily_validation_receipts_owner_authorized",
    }:
        raise CollisionCensusError("raw corpus manifest episode authority is unknown")
    if (
        episode_dedup.get("unique_episode_identity_count"),
        episode_dedup.get("unique_episode_id_count"),
    ) != (total_members, total_members):
        raise CollisionCensusError("raw corpus manifest episode deduplication count drifted")
    require_sha256(
        episode_dedup.get("episode_identity_inventory_sha256"),
        field="raw corpus episode identity inventory",
    )
    if manifest.get("recollection_authorized") is not False or manifest.get("archive_mutation_authorized") is not False:
        raise CollisionCensusError("raw corpus manifest permits recollection or archive mutation")


def make_raw_corpus_receipt(
    manifest: Mapping[str, Any],
    *,
    run_identity_sha256: str,
    resource_observation: Mapping[str, Any],
) -> dict[str, Any]:
    validate_raw_corpus_manifest(manifest)
    episode_dedup = manifest.get("episode_deduplication")
    assert isinstance(episode_dedup, Mapping)  # validated above
    provenance = manifest.get("source_manifest_provenance")
    assert isinstance(provenance, Sequence)  # validated above
    resource = _canonical_value(resource_observation, field="resource observation")
    if not isinstance(resource, dict):
        raise CollisionCensusError("resource observation must be an object")
    return {
        "schema": R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "status": "passed",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "raw_expert_corpus_manifest_sha256": canonical_sha256(manifest),
        "window_start_utc": RAW_CORPUS_START_UTC,
        "window_end_utc": RAW_CORPUS_END_UTC,
        "distinct_utc_day_count": RAW_CORPUS_EXACT_DAYS,
        "completed_raw_zip_member_count": manifest.get("total_raw_zip_json_members"),
        "completed_validated_episode_count": manifest.get("total_validated_episodes"),
        "recollection_authorized": False,
        "source_disjointness": {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
            "training_eligible": False,
        },
        "source_manifest_provenance_sha256": canonical_sha256(provenance),
        "source_receipt_day_coverage_sha256": canonical_sha256(
            manifest["source_receipt_day_coverage"]
        ),
        "episode_deduplication_sha256": canonical_sha256(episode_dedup),
        "resource_observation": resource,
        "run_identity_sha256": require_sha256(run_identity_sha256, field="raw corpus run identity"),
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "production_activation": False,
            "inzi_mutation": False,
            "archive_mutation": False,
        },
    }


def validate_frozen_schema_gate(
    schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> tuple[str, str]:
    """Verify the rev3 implementation-schema freeze before re-featurization.

    The census does not own Phase B's implementation files, but it refuses to
    credit a full r298 corpus pass without a separately immutable schema
    manifest and a zero-bypass attestation.  This leaves zero/inert repairs
    unblocked while preventing a later corpus receipt from laundering an
    unverified nonzero path.
    """

    if schema_manifest.get("schema") != R298_FROZEN_SCHEMA_MANIFEST_SCHEMA:
        raise CollisionCensusError("frozen schema manifest schema drifted")
    if schema_manifest.get("status") != "frozen_zero_inert_before_refeaturization":
        raise CollisionCensusError("frozen schema manifest is not zero/inert")
    if (
        schema_manifest.get("owner_revision"),
        schema_manifest.get("goal_revision"),
        schema_manifest.get("root_handoff_revision"),
    ) != (
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
    ):
        raise CollisionCensusError("frozen schema manifest revision-5 authority drifted")
    if schema_manifest.get("owner_goal_sha256") != OWNER_GOAL_SHA256 or (
        schema_manifest.get("rule_derivative_contract_sha256")
        != RULE_DERIVATIVE_CONTRACT_SHA256
    ):
        raise CollisionCensusError("frozen schema manifest provenance drifted")
    if schema_manifest.get("rule_derivative_gateway_sha256") != RULE_DERIVATIVE_GATEWAY_SHA256:
        raise CollisionCensusError("frozen schema manifest gateway binding drifted")
    if schema_manifest.get("mechanics_attachment_sha256") != MECHANICS_ATTACHMENT_SHA256:
        raise CollisionCensusError("frozen schema manifest mechanics binding drifted")
    validate_revision_5_predecessor_classification(
        schema_manifest.get("revision_5_predecessor_classification")
    )
    for field in (
        "feature_schema_sha256",
        "target_schema_sha256",
        "checklist_provenance_schema_sha256",
    ):
        require_sha256(schema_manifest.get(field), field=f"frozen schema manifest {field}")
    branches = schema_manifest.get("new_branch_inventory")
    if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes, bytearray)):
        raise CollisionCensusError("frozen schema manifest new-branch inventory is malformed")
    if not branches or any(not isinstance(branch, str) or not branch for branch in branches):
        raise CollisionCensusError("frozen schema manifest must name each new branch")
    if len(set(branches)) != len(branches):
        raise CollisionCensusError("frozen schema manifest new-branch inventory repeats a branch")
    if schema_manifest.get("runtime_wired") is not False:
        raise CollisionCensusError("frozen schema manifest permits a runtime wiring")
    for field in (
        "default_zero_and_inert_attestation",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
    ):
        if schema_manifest.get(field) is not True:
            raise CollisionCensusError(f"frozen schema manifest {field} did not pass")
    manifest_digest = canonical_sha256(schema_manifest)
    if zero_bypass_receipt.get("schema") != R298_ZERO_BYPASS_RECEIPT_SCHEMA:
        raise CollisionCensusError("zero-bypass receipt schema drifted")
    if zero_bypass_receipt.get("status") != "passed_exact_baseline_logits":
        raise CollisionCensusError("zero-bypass receipt did not pass")
    if (
        zero_bypass_receipt.get("owner_revision"),
        zero_bypass_receipt.get("goal_revision"),
        zero_bypass_receipt.get("root_handoff_revision"),
        zero_bypass_receipt.get("rule_derivative_gateway_sha256"),
        zero_bypass_receipt.get("rule_derivative_contract_sha256"),
    ) != (
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
    ):
        raise CollisionCensusError("zero-bypass receipt revision-5 authority drifted")
    validate_revision_5_predecessor_classification(
        zero_bypass_receipt.get("revision_5_predecessor_classification")
    )
    if zero_bypass_receipt.get("frozen_schema_manifest_sha256") != manifest_digest:
        raise CollisionCensusError("zero-bypass receipt does not bind frozen schema manifest")
    if zero_bypass_receipt.get("all_new_routes_exact_zero_or_bypassed") is not True:
        raise CollisionCensusError("zero-bypass receipt permits a nonzero new route")
    if zero_bypass_receipt.get("checklist_provenance_schema_frozen") is not True:
        raise CollisionCensusError("zero-bypass receipt lacks checklist provenance schema")
    if zero_bypass_receipt.get("layer_off_bit_identical_baseline_logits") is not True:
        raise CollisionCensusError("zero-bypass receipt lacks exact layer-off logit parity")
    if zero_bypass_receipt.get("layer_off_identical_legal_choice") is not True:
        raise CollisionCensusError("zero-bypass receipt lacks exact layer-off legal-choice parity")
    if zero_bypass_receipt.get("runtime_wired") is not False:
        raise CollisionCensusError("zero-bypass receipt permits a runtime wiring")
    return manifest_digest, canonical_sha256(zero_bypass_receipt)


def frozen_schema_gate_contract() -> dict[str, Any]:
    """Return the immutable producer-side pre-refeaturization gate shape.

    Other r298 receipts may *bridge* this shape, but a census cannot replace
    it with a generic declaration.  Keeping the required field list here
    avoids a stale runner silently accepting an earlier schema-freeze format.
    """

    return {
        "frozen_schema_manifest_schema": R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "zero_bypass_receipt_schema": R298_ZERO_BYPASS_RECEIPT_SCHEMA,
        "manifest_required_fields": [
            "owner_goal_sha256",
            "owner_revision:298",
            "goal_revision:5",
            "root_handoff_revision:303",
            "rule_derivative_contract_sha256",
            "rule_derivative_gateway_sha256",
            "revision_5_predecessor_classification",
            "mechanics_attachment_sha256",
            "feature_schema_sha256",
            "target_schema_sha256",
            "checklist_provenance_schema_sha256",
            "new_branch_inventory",
            "runtime_wired:false",
            "default_zero_and_inert_attestation:true",
            "layer_off_bit_identical_baseline_logits:true",
            "layer_off_identical_legal_choice:true",
        ],
        "zero_bypass_required_fields": [
            "frozen_schema_manifest_sha256",
            "owner_revision:298",
            "goal_revision:5",
            "root_handoff_revision:303",
            "rule_derivative_contract_sha256",
            "rule_derivative_gateway_sha256",
            "revision_5_predecessor_classification",
            "all_new_routes_exact_zero_or_bypassed:true",
            "checklist_provenance_schema_frozen:true",
            "layer_off_bit_identical_baseline_logits:true",
            "layer_off_identical_legal_choice:true",
            "runtime_wired:false",
        ],
    }


def _raw_visual_alignment_surface(
    observation: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    """Normalize the one ephemeral raw field absent from visualizer ``obs``.

    Raw actor observations retain ``search_begin_input`` for the legacy search
    runtime, while authoritative visualizer rows intentionally omit it.  It is
    private/non-policy state, so it is removed *only* for exact trace joining;
    no other field is relaxed.  This mirrors the sealed visual-trace reader's
    `_without_search_token` check without importing its training stack.
    """

    if not isinstance(observation, Mapping):
        raise CollisionCensusError(f"{field} must be an object")
    surface = dict(observation)
    surface.pop("search_begin_input", None)
    normalized = _canonical_value(surface, field=field)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guard above
        raise CollisionCensusError(f"{field} normalization is malformed")
    return normalized


def raw_observations_from_recorded_episode(
    payload: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    """Yield every raw two-seat outer observation for schema inventory only.

    This is deliberately stricter than the visual-trace reader: Phase A must
    inventory the complete recorded raw observation surface, including
    setup-time ``current: null`` outers and the inactive seat, rather than
    merely the masked policy frames that happen to be re-featurized.  The
    yielded mappings are transient; callers must reduce them to paths/types
    before artifact emission and must never persist their values.
    """

    if not isinstance(payload, Mapping):
        raise CollisionCensusError("raw episode must be an object")
    steps = payload.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)) or not steps:
        raise CollisionCensusError("raw episode has no steps")
    for step_index, step in enumerate(steps):
        if (
            not isinstance(step, Sequence)
            or isinstance(step, (str, bytes, bytearray))
            or len(step) != 2
        ):
            raise CollisionCensusError(
                f"raw episode step {step_index} does not contain exactly two outer states"
            )
        for seat, outer in enumerate(step):
            if not isinstance(outer, Mapping):
                raise CollisionCensusError(
                    f"raw episode step {step_index} seat {seat} outer state is malformed"
                )
            observation = outer.get("observation")
            if not isinstance(observation, Mapping):
                raise CollisionCensusError(
                    f"raw episode step {step_index} seat {seat} lacks a raw observation"
                )
            yield observation


def _matched_visual_row_observation(
    *,
    row: Mapping[str, Any],
    steps: Sequence[Any],
    trace_index: int,
) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    """Resolve a visual row to its unique raw actor observation by content.

    A trace row's post-action ``current.yourIndex`` can differ from the seat
    that owned the pre-action selection (notably during setup and automatic
    prompts).  Joining by that post-action actor silently mislabels legal
    actions.  The authoritative masked observation is the unambiguous bridge.
    """

    if trace_index >= len(steps):
        raise CollisionCensusError(f"raw step {trace_index} does not align with visual trace")
    step = steps[trace_index]
    if not isinstance(step, Sequence) or isinstance(step, (str, bytes, bytearray)) or len(step) != 2:
        raise CollisionCensusError(f"raw step {trace_index} does not align with visual trace")
    visual_observation = row.get("obs")
    if not isinstance(visual_observation, Mapping):
        raise CollisionCensusError(f"visual trace row {trace_index} lacks masked observation")
    if "search_begin_input" in visual_observation:
        raise CollisionCensusError(
            f"visual trace row {trace_index} exposes a search-begin input"
        )
    visual_surface = _raw_visual_alignment_surface(
        visual_observation,
        field=f"visual trace {trace_index} observation",
    )
    matches: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    for seat, outer in enumerate(step):
        if not isinstance(outer, Mapping):
            continue
        raw_observation = outer.get("observation")
        if not isinstance(raw_observation, Mapping):
            continue
        if _raw_visual_alignment_surface(
            raw_observation,
            field=f"raw step {trace_index} observation[{seat}]",
        ) == visual_surface:
            matches.append((seat, outer, raw_observation))
    if len(matches) != 1:
        raise CollisionCensusError(
            f"visual trace row {trace_index} has no unique matching raw masked observation"
        )
    return matches[0]


def _aligned_visual_row_actions(
    *,
    row: Mapping[str, Any],
    steps: Sequence[Any],
    trace_index: int,
) -> Sequence[Any]:
    """Prove the trace's selected actions match the next raw simulator step.

    The masked pre-action observation lives at ``steps[index]`` while its
    selected actions are recorded in the environment result at
    ``steps[index + 1]``.  Keeping those joins separate prevents an apparently
    legal action from being paired with the wrong prompt during a seat handoff.
    """

    action = row.get("action")
    if (
        not isinstance(action, Sequence)
        or isinstance(action, (str, bytes, bytearray))
        or len(action) != 2
    ):
        raise CollisionCensusError(f"visual trace row {trace_index} lacks two-seat action")
    next_index = trace_index + 1
    if next_index >= len(steps):
        raise CollisionCensusError(
            f"visual trace row {trace_index} lacks next raw action step"
        )
    next_step = steps[next_index]
    if (
        not isinstance(next_step, Sequence)
        or isinstance(next_step, (str, bytes, bytearray))
        or len(next_step) != 2
        or any(not isinstance(outer, Mapping) for outer in next_step)
    ):
        raise CollisionCensusError(
            f"visual trace row {trace_index} next raw action step is malformed"
        )
    expected = [outer.get("action") for outer in next_step]
    if list(action) != expected:
        raise CollisionCensusError(
            f"visual trace row {trace_index} action does not align with next raw step"
        )
    return action


def stage_descriptors_from_recorded_episode(
    payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    max_stages: int | None = None,
) -> list[dict[str, Any]]:
    """Extract public recorded factorized stages without branching hidden state.

    This intentionally mirrors the alignment discipline in
    :mod:`authoritative_visual_trace`, but returns no private target payload.
    Only the actually selected candidate carries a partial realized-transition
    marker; alternate candidates remain without successor evidence until a
    permitted public-deterministic engine probe supplies it.
    """

    if not isinstance(payload, Mapping):
        raise CollisionCensusError("raw episode must be an object")
    steps = payload.get("steps")
    if not isinstance(steps, Sequence) or not steps:
        raise CollisionCensusError("raw episode has no steps")
    first = steps[0]
    if not isinstance(first, Sequence) or not first or not isinstance(first[0], Mapping):
        raise CollisionCensusError("raw episode step zero is malformed")
    visual = first[0].get("visualize")
    trace = visual if isinstance(visual, Sequence) and not isinstance(visual, (str, bytes, bytearray)) else None
    if not trace:
        raise CollisionCensusError("raw episode lacks a visualizer trace")
    source_surface = _canonical_value(source, field="raw stage source")
    if not isinstance(source_surface, dict):  # pragma: no cover - guarded above
        raise CollisionCensusError("raw stage source must be an object")
    setup = trace[0].get("action") if isinstance(trace[0], Mapping) else None
    own_deck_fingerprints: dict[int, str] = {}
    own_deck_is_alakazam_archetype: dict[int, bool] = {}
    own_deck_contains_card_743: dict[int, bool] = {}
    if isinstance(setup, Sequence) and not isinstance(setup, (str, bytes, bytearray)) and len(setup) == 2:
        for seat, deck in enumerate(setup):
            if isinstance(deck, Sequence) and not isinstance(deck, (str, bytes, bytearray)):
                cards = [
                    _exact_int(card, field=f"setup deck {seat}", minimum=1)
                    for card in deck
                ]
                # Use the same canonical 60-card multiset encoding as the
                # r241/r274 deck contract.  This makes the actor-deck identity
                # directly comparable with EXACT_NEW_LIST_MULTISET_SHA256 and
                # permits the re-featurizer to retain only Alakazam decisions.
                own_deck_fingerprints[seat] = "sha256:" + hashlib.sha256(
                    json.dumps(sorted(cards), separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                # Persist only the archetype-membership predicate, never the
                # ordered setup deck.  In this corpus Alakazam is not used as
                # a tech inclusion, so any list containing card 743 belongs to
                # the Alakazam archetype; its multiset digest still preserves
                # list-variant stratification.
                contains_card_743 = 743 in cards
                own_deck_contains_card_743[seat] = contains_card_743
                # Kept only as an older audit label.  Eligibility consumers
                # must use the explicit literal membership field below; the
                # revision-6 contract expressly rejects an archetype-label or
                # exact-list interpretation.
                own_deck_is_alakazam_archetype[seat] = contains_card_743
    descriptors: list[dict[str, Any]] = []
    # ``trace[index]`` carries the post-action full ``current`` (and hence its
    # acting seat), but its masked ``obs``/``select`` and selected action live
    # in the *other* seat's payload at ``steps[index]``.  The raw corpus has
    # intervening setup/automatic rows where the post-action actor differs
    # from the pre-action selected seat; resolve the exact matching seat by
    # masked-observation identity instead of assuming either offset/seat.
    for index in range(1, len(trace)):
        if max_stages is not None and len(descriptors) >= max_stages:
            break
        row = trace[index]
        if not isinstance(row, Mapping):
            raise CollisionCensusError(f"visual trace row {index} is malformed")
        visual_observation = row.get("obs")
        if not isinstance(visual_observation, Mapping):
            raise CollisionCensusError(f"visual trace row {index} lacks masked observation")
        current = row.get("current")
        if not isinstance(current, Mapping):
            # A non-policy trace row may lack a post-action current state, but
            # a visible selection without one would otherwise be silently
            # omitted from both coverage and re-featurization.
            if visual_observation.get("select") is not None:
                raise CollisionCensusError(
                    f"visual trace row {index} selection lacks post-action current"
                )
            continue
        actor = _exact_int(current.get("yourIndex"), field=f"trace {index} actor", minimum=0, maximum=1)
        # The raw visualizer contract defines this row's full ``current`` as
        # the authoritative post-action state for the masked observation at
        # the same trace index.  Retain only its public projection hash as a
        # target-only adjacent successor witness; no full state or hidden
        # identity enters the option feature/semantic key.
        adjacent_public_successor_hash = canonical_public_current_hash(current, actor=actor)
        selected_seat, _outer, _raw_observation = _matched_visual_row_observation(
            row=row,
            steps=steps,
            trace_index=index,
        )
        raw_action = _aligned_visual_row_actions(
            row=row,
            steps=steps,
            trace_index=index,
        )
        # Use the masked visual input, not the raw outer wrapper.  The two are
        # exact after removing only search_begin_input for trace matching, and
        # this keeps that private legacy runtime token out of every r298 token
        # and semantic computation.
        observation = dict(visual_observation)
        pre_action_current = observation.get("current")
        if not isinstance(pre_action_current, Mapping):
            raise CollisionCensusError(
                f"visual trace row {index} masked observation lacks current state"
            )
        pre_action_seat = _exact_int(
            pre_action_current.get("yourIndex"),
            field=f"trace {index} masked-observation actor",
            minimum=0,
            maximum=1,
        )
        if pre_action_seat != selected_seat:
            raise CollisionCensusError(
                f"visual trace row {index} matched raw seat disagrees with masked-observation actor"
            )
        # A visualizer row is not necessarily a factorized policy decision.
        # The raw corpus encodes ordinary post-action/non-selection rows as
        # ``select: null``.  Match ``recorded_episode_frame_coverage`` exactly:
        # skip those rows, while treating any *present* malformed selection as
        # a fail-closed schema error.
        if observation.get("select") is None:
            continue
        action = _action_indices(raw_action[selected_seat], field=f"trace {index} action")
        select = _select_mapping(observation)
        n_options, minimum, maximum = _select_bounds(select)
        if not minimum <= len(action) <= maximum or any(item >= n_options for item in action):
            raise CollisionCensusError(f"visual trace row {index} action is not legal")
        # Pure-Python factorized support, equivalent to the current feature
        # helper but retained here so Phase A schema inventory does not require
        # importing cg/libcg before its receipt is verified.
        prefix: list[int] = []
        stage_number = 0
        for next_option in action:
            candidates = [prefix + [option] for option in range(n_options) if option not in prefix]
            if len(prefix) >= minimum:
                candidates.append(list(prefix))
            selected_index = candidates.index(prefix + [next_option])
            descriptors.append(
                {
                    "observation": observation,
                    "candidates": candidates,
                    "stage_prefix": list(prefix),
                    "selected_candidate_index": selected_index,
                    "source": {
                        **source_surface,
                        "episode_id": payload.get("id"),
                        "env_step": index,
                        "acting_seat": selected_seat,
                        "factorized_stage": stage_number,
                        "acting_deck_multiset_sha256": own_deck_fingerprints.get(selected_seat),
                        "acting_seat_setup_deck_contains_card_743": own_deck_contains_card_743.get(selected_seat, False),
                        "opponent_deck_multiset_sha256": own_deck_fingerprints.get(1 - selected_seat),
                        "acting_deck_is_alakazam_archetype": own_deck_is_alakazam_archetype.get(selected_seat, False),
                        "adjacent_public_successor": {
                            "availability": "available_target_only_visualizer_post_action_current",
                            "canonical_public_current_hash": adjacent_public_successor_hash,
                        },
                    },
                }
            )
            prefix.append(next_option)
            stage_number += 1
        if len(prefix) < maximum:
            candidates = [prefix + [option] for option in range(n_options) if option not in prefix]
            if len(prefix) >= minimum:
                candidates.append(list(prefix))
            descriptors.append(
                {
                    "observation": observation,
                    "candidates": candidates,
                    "stage_prefix": list(prefix),
                    "selected_candidate_index": candidates.index(prefix),
                    "source": {
                        **source_surface,
                        "episode_id": payload.get("id"),
                        "env_step": index,
                        "acting_seat": selected_seat,
                        "factorized_stage": stage_number,
                        "acting_deck_multiset_sha256": own_deck_fingerprints.get(selected_seat),
                        "acting_seat_setup_deck_contains_card_743": own_deck_contains_card_743.get(selected_seat, False),
                        "opponent_deck_multiset_sha256": own_deck_fingerprints.get(1 - selected_seat),
                        "acting_deck_is_alakazam_archetype": own_deck_is_alakazam_archetype.get(selected_seat, False),
                        "adjacent_public_successor": {
                            "availability": "available_target_only_visualizer_post_action_current",
                            "canonical_public_current_hash": adjacent_public_successor_hash,
                        },
                    },
                }
            )
        elif not action and minimum == 0 and maximum == 0:
            # A zero-width legal selection is still an actor-visible forced
            # frame.  It has one legal STOP action, but no appended option
            # stage above could represent it.  Emit that explicit factorized
            # STOP row so Phase A's forced-frame coverage proof does not
            # silently omit this legal simulator decision.
            descriptors.append(
                {
                    "observation": observation,
                    "candidates": [[]],
                    "stage_prefix": [],
                    "selected_candidate_index": 0,
                    "source": {
                        **source_surface,
                        "episode_id": payload.get("id"),
                        "env_step": index,
                        "acting_seat": selected_seat,
                        "factorized_stage": 0,
                        "acting_deck_multiset_sha256": own_deck_fingerprints.get(selected_seat),
                        "acting_seat_setup_deck_contains_card_743": own_deck_contains_card_743.get(selected_seat, False),
                        "opponent_deck_multiset_sha256": own_deck_fingerprints.get(1 - selected_seat),
                        "acting_deck_is_alakazam_archetype": own_deck_is_alakazam_archetype.get(selected_seat, False),
                        "adjacent_public_successor": {
                            "availability": "available_target_only_visualizer_post_action_current",
                            "canonical_public_current_hash": adjacent_public_successor_hash,
                        },
                    },
                }
            )
    return descriptors


def recorded_episode_frame_coverage(payload: Mapping[str, Any]) -> dict[str, int]:
    """Count every actor-visible/forced selection frame in one raw episode.

    The function is intentionally independent of feature construction, so the
    full re-featurizer can prove that it did not silently skip a visible prompt
    when one row happens to be malformed.  Any alignment problem raises rather
    than converting an omitted frame into a lower coverage number.
    """

    if not isinstance(payload, Mapping):
        raise CollisionCensusError("raw episode must be an object")
    steps = payload.get("steps")
    if not isinstance(steps, Sequence) or not steps:
        raise CollisionCensusError("raw episode has no steps")
    first = steps[0]
    if not isinstance(first, Sequence) or not first or not isinstance(first[0], Mapping):
        raise CollisionCensusError("raw episode step zero is malformed")
    visual = first[0].get("visualize")
    trace = visual if isinstance(visual, Sequence) and not isinstance(visual, (str, bytes, bytearray)) else None
    if not trace:
        raise CollisionCensusError("raw episode lacks a visualizer trace")
    actor_visible_frames = 0
    forced_frames = 0
    # Keep this exactly aligned with ``stage_descriptors_from_recorded_episode``:
    # determine the pre-action owner from the unique masked-observation match,
    # not the post-action actor in visualizer ``current``.
    for index in range(1, len(trace)):
        row = trace[index]
        if not isinstance(row, Mapping):
            raise CollisionCensusError(f"visual trace row {index} is malformed")
        visual_observation = row.get("obs")
        if not isinstance(visual_observation, Mapping):
            raise CollisionCensusError(f"visual trace row {index} lacks masked observation")
        current = row.get("current")
        if not isinstance(current, Mapping):
            if visual_observation.get("select") is not None:
                raise CollisionCensusError(
                    f"visual trace row {index} selection lacks post-action current"
                )
            continue
        selected_seat, _outer, _raw_observation = _matched_visual_row_observation(
            row=row,
            steps=steps,
            trace_index=index,
        )
        raw_action = _aligned_visual_row_actions(
            row=row,
            steps=steps,
            trace_index=index,
        )
        observation = visual_observation
        # No select means there is no factorized policy decision at this row.
        if observation.get("select") is None:
            continue
        select = _select_mapping(observation)
        n_options, minimum, maximum = _select_bounds(select)
        action = _action_indices(raw_action[selected_seat], field=f"trace {index} action")
        if not minimum <= len(action) <= maximum or any(item >= n_options for item in action):
            raise CollisionCensusError(f"visual trace row {index} action is not legal")
        actor_visible_frames += 1
        # A forced frame has exactly one complete ordered legal action.  The
        # product is calculated directly rather than guessing from one option.
        action_count = sum(
            math.perm(n_options, count)
            for count in range(minimum, maximum + 1)
        )
        if action_count == 1:
            forced_frames += 1
    return {
        "actor_visible_selection_frame_count": actor_visible_frames,
        "forced_selection_frame_count": forced_frames,
    }


def make_receipt(
    *,
    report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    raw_expert_corpus_manifest: Mapping[str, Any],
    raw_expert_corpus_receipt: Mapping[str, Any],
    frozen_schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
    current_token_abi_source_sha256: str,
    run_identity_sha256: str,
    raw_episode_count: int,
    public_matchup_distribution: Mapping[str, int],
    acting_deck_distribution: Mapping[str, int],
    frame_coverage: Mapping[str, Any],
    re_featurization: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the strict r298 handoff receipt consumed by later gates."""

    if report.get("schema") != R298_COLLISION_REPORT_SCHEMA:
        raise CollisionCensusError("collision report schema drifted")
    if report.get("status") not in {STATUS_PASSED, STATUS_FAILED_COLLISION, STATUS_BLOCKED_EVIDENCE}:
        raise CollisionCensusError("a final collision receipt requires a complete non-inventory report")
    if inventory.get("schema") != R298_PHASE_A_INVENTORY_SCHEMA:
        raise CollisionCensusError("phase A inventory schema drifted")
    validate_phase_a_inventory(inventory)
    if inventory.get("inventory_scope") != PHASE_A_RAW_REPLAY_INVENTORY_SCOPE:
        raise CollisionCensusError("collision receipt requires complete raw replay schema inventory")
    validate_raw_corpus_manifest(raw_expert_corpus_manifest)
    if raw_expert_corpus_receipt.get("schema") != R298_RAW_CORPUS_RECEIPT_SCHEMA:
        raise CollisionCensusError("census does not bind an r298 raw expert corpus receipt")
    if raw_expert_corpus_receipt.get("status") != "passed":
        raise CollisionCensusError("census raw expert corpus receipt did not pass")
    if (
        raw_expert_corpus_receipt.get("owner_revision"),
        raw_expert_corpus_receipt.get("goal_revision"),
        raw_expert_corpus_receipt.get("root_handoff_revision"),
        raw_expert_corpus_receipt.get("rule_derivative_gateway_sha256"),
        raw_expert_corpus_receipt.get("rule_derivative_contract_sha256"),
    ) != (
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
    ):
        raise CollisionCensusError("census raw receipt does not bind revision-5 authority")
    validate_revision_5_predecessor_classification(
        raw_expert_corpus_receipt.get("revision_5_predecessor_classification")
    )
    if raw_expert_corpus_receipt.get("distinct_utc_day_count") != RAW_CORPUS_EXACT_DAYS:
        raise CollisionCensusError("census source is not exactly thirty distinct UTC days")
    if raw_expert_corpus_receipt.get("raw_expert_corpus_manifest_sha256") != canonical_sha256(
        raw_expert_corpus_manifest
    ):
        raise CollisionCensusError("census raw receipt does not bind its supplied raw manifest")
    if raw_expert_corpus_receipt.get("source_manifest_provenance_sha256") != canonical_sha256(
        raw_expert_corpus_manifest["source_manifest_provenance"]
    ) or raw_expert_corpus_receipt.get("source_receipt_day_coverage_sha256") != canonical_sha256(
        raw_expert_corpus_manifest["source_receipt_day_coverage"]
    ) or raw_expert_corpus_receipt.get("episode_deduplication_sha256") != canonical_sha256(
        raw_expert_corpus_manifest["episode_deduplication"]
    ):
        raise CollisionCensusError("census raw receipt misses strict provenance or dedup bindings")
    frozen_schema_manifest_sha256, zero_bypass_receipt_sha256 = validate_frozen_schema_gate(
        frozen_schema_manifest,
        zero_bypass_receipt,
    )
    raw_corpus_receipt_digest = canonical_sha256(raw_expert_corpus_receipt)
    feature_digest = require_sha256(
        current_token_abi_source_sha256,
        field="current token ABI source",
    )
    if feature_digest != R274_EXACT_FEATURES_SOURCE_SHA256:
        raise CollisionCensusError("census input does not bind the exact r274 feature ABI")
    coverage_surface = _canonical_value(
        frame_coverage,
        field="collision receipt frame coverage",
    )
    if not isinstance(coverage_surface, dict):  # pragma: no cover - canonical guard
        raise CollisionCensusError("collision receipt frame coverage must be an object")
    actor_visible_frame_count = _exact_int(
        coverage_surface.get("actor_visible_selection_frame_count"),
        field="collision receipt actor-visible frame count",
        minimum=0,
    )
    processed_actor_visible_frame_count = _exact_int(
        coverage_surface.get("processed_actor_visible_selection_frame_count"),
        field="collision receipt processed actor-visible frame count",
        minimum=0,
    )
    forced_frame_count = _exact_int(
        coverage_surface.get("forced_selection_frame_count"),
        field="collision receipt forced selection frame count",
        minimum=0,
    )
    if (
        processed_actor_visible_frame_count != actor_visible_frame_count
        or forced_frame_count > actor_visible_frame_count
        or coverage_surface.get("all_actor_visible_and_forced_frames_included") is not True
    ):
        raise CollisionCensusError("collision receipt actor-visible/forced frame coverage is incomplete")
    refeaturization_surface = _canonical_value(
        re_featurization,
        field="collision receipt re-featurization",
    )
    if not isinstance(refeaturization_surface, dict):  # pragma: no cover - canonical guard
        raise CollisionCensusError("collision receipt re-featurization must be an object")
    if (
        refeaturization_surface.get("raw_schema_inventory_sha256"),
        refeaturization_surface.get("raw_schema_inventory_scope"),
        refeaturization_surface.get("raw_schema_inventory_rejected_observation_count"),
        refeaturization_surface.get("actor_visible_selection_inventory_scope"),
        refeaturization_surface.get("actor_visible_selection_inventory_observation_count"),
        refeaturization_surface.get("raw_observation_values_persisted"),
    ) != (
        canonical_sha256(inventory),
        PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
        0,
        PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE,
        processed_actor_visible_frame_count,
        False,
    ):
        raise CollisionCensusError("collision receipt re-featurization inventory bindings are incomplete")
    if refeaturization_surface.get("raw_outer_observation_count") != inventory.get(
        "raw_observation_count"
    ):
        raise CollisionCensusError("collision receipt raw outer-observation coverage drifted")
    require_sha256(
        refeaturization_surface.get("actor_visible_selection_inventory_sha256"),
        field="collision receipt actor-visible selection inventory",
    )
    return {
        "schema": R298_COLLISION_RECEIPT_SCHEMA,
        "status": report.get("status"),
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "exact_new_list_multiset_sha256": EXACT_NEW_LIST_MULTISET_SHA256,
        "raw_expert_corpus_receipt_sha256": raw_corpus_receipt_digest,
        "raw_expert_corpus_manifest_sha256": require_sha256(
            raw_expert_corpus_receipt.get("raw_expert_corpus_manifest_sha256"),
            field="raw expert corpus manifest",
        ),
        "raw_expert_corpus": {
            "window_start_utc": RAW_CORPUS_START_UTC,
            "window_end_utc": RAW_CORPUS_END_UTC,
            "distinct_utc_day_count": RAW_CORPUS_EXACT_DAYS,
            "completed_raw_zip_member_count": raw_expert_corpus_receipt.get(
                "completed_raw_zip_member_count"
            ),
            "completed_validated_episode_count": raw_expert_corpus_receipt.get(
                "completed_validated_episode_count"
            ),
            "source_disjointness": raw_expert_corpus_receipt.get("source_disjointness"),
        },
        "frozen_schema_gate": {
            "frozen_schema_manifest_sha256": frozen_schema_manifest_sha256,
            "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
            "frozen_before_refeaturization": True,
            "runtime_wired": False,
        },
        "pinned_simulator": {
            "libcg_sha256": CANONICAL_R236_LIBCG_SHA256,
            "libcg_size_bytes": CANONICAL_R236_LIBCG_SIZE_BYTES,
            "engine_transition_evidence_schema": R298_ENGINE_EVIDENCE_SCHEMA,
            "complete_evidence_required_for_pass": True,
            "hidden_branching_permitted": False,
        },
        "current_token_abi": {
            "schema": R298_TOKEN_ABI_SCHEMA,
            "features_source_sha256": feature_digest,
        },
        "phase_a_inventory_sha256": canonical_sha256(inventory),
        "phase_a_raw_observation_inventory": {
            "inventory_scope": PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
            "raw_observation_count": inventory.get("raw_observation_count"),
            "actor_context_observation_count": inventory.get(
                "actor_context_observation_count"
            ),
            "no_actor_context_observation_count": inventory.get(
                "no_actor_context_observation_count"
            ),
            "rejected_observation_count": inventory.get("rejected_observation_count"),
            "raw_values_persisted": False,
            "complete_raw_outer_observation_inventory_required": True,
        },
        "frame_coverage": coverage_surface,
        "re_featurization": refeaturization_surface,
        "collision_report_sha256": canonical_sha256(report),
        "run_identity_sha256": require_sha256(run_identity_sha256, field="run identity"),
        "raw_episode_count": _exact_int(raw_episode_count, field="raw episode count", minimum=0),
        "selected_action_involvement": {
            "selected_action_in_collision_record_count": report.get(
                "selected_action_in_collision_record_count"
            ),
            "selected_action_in_divergent_collision_record_count": report.get(
                "selected_action_in_divergent_collision_record_count"
            ),
        },
        "public_matchup_distribution": {
            str(key): _exact_int(value, field=f"public matchup count {key}", minimum=0)
            for key, value in sorted(public_matchup_distribution.items())
        },
        "acting_deck_multiset_distribution": {
            str(key): _exact_int(value, field=f"acting deck count {key}", minimum=0)
            for key, value in sorted(acting_deck_distribution.items())
        },
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_eligible": False,
            "production_activation": False,
            "inzi_mutation": False,
            "learned_baseline_mutation": False,
        },
        "pass": report.get("status") == STATUS_PASSED,
    }


def make_revision_5_census_validation_receipt(
    *,
    census_receipt: Mapping[str, Any],
    raw_expert_corpus_manifest: Mapping[str, Any],
    raw_expert_corpus_receipt: Mapping[str, Any],
    frozen_schema_manifest: Mapping[str, Any],
    zero_bypass_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a typed rev5-only validation bridge for the r298 census.

    This is deliberately narrower than a future training/handoff receipt.  It
    proves only that a completed r298 raw-corpus/collision pass is governed by
    the final revision-5 authority and that no revision-4 receipt has been
    promoted by blind digest substitution.  It grants no training, staging,
    service, package, queue, or fleet authority.
    """

    validate_raw_corpus_manifest(raw_expert_corpus_manifest)
    frozen_schema_manifest_sha256, zero_bypass_receipt_sha256 = validate_frozen_schema_gate(
        frozen_schema_manifest,
        zero_bypass_receipt,
    )
    if census_receipt.get("schema") != R298_COLLISION_RECEIPT_SCHEMA:
        raise CollisionCensusError("revision-5 validation requires an r298 collision receipt")
    if census_receipt.get("status") != STATUS_PASSED or census_receipt.get("pass") is not True:
        raise CollisionCensusError("revision-5 validation requires a passed complete collision census")
    if (
        census_receipt.get("owner_revision"),
        census_receipt.get("goal_revision"),
        census_receipt.get("root_handoff_revision"),
        census_receipt.get("rule_derivative_gateway_sha256"),
        census_receipt.get("rule_derivative_contract_sha256"),
    ) != (
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
    ):
        raise CollisionCensusError("collision receipt does not bind final revision-5 authority")
    census_predecessor = validate_revision_5_predecessor_classification(
        census_receipt.get("revision_5_predecessor_classification")
    )
    if census_predecessor != revision_5_predecessor_classification():
        # The actual census pass should use fresh r5 schema/bypass artifacts.
        # Any historical receipt it consumes belongs in the new r5 schema
        # artifact, not in a completed collision verdict by itself.
        raise CollisionCensusError(
            "completed collision census cannot treat revision-4 receipts as active authority"
        )
    raw_inventory_proof = census_receipt.get("phase_a_raw_observation_inventory")
    if not isinstance(raw_inventory_proof, Mapping) or (
        raw_inventory_proof.get("inventory_scope"),
        raw_inventory_proof.get("rejected_observation_count"),
        raw_inventory_proof.get("raw_values_persisted"),
        raw_inventory_proof.get("complete_raw_outer_observation_inventory_required"),
    ) != (
        PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
        0,
        False,
        True,
    ):
        raise CollisionCensusError(
            "revision-5 validation requires complete value-free raw Phase A inventory"
        )
    for field in (
        "raw_observation_count",
        "actor_context_observation_count",
        "no_actor_context_observation_count",
    ):
        _exact_int(raw_inventory_proof.get(field), field=f"revision-5 raw inventory {field}", minimum=0)
    if (
        raw_inventory_proof["actor_context_observation_count"]
        + raw_inventory_proof["no_actor_context_observation_count"]
        != raw_inventory_proof["raw_observation_count"]
    ):
        raise CollisionCensusError("revision-5 raw Phase A inventory counts do not reconcile")
    frame_coverage = census_receipt.get("frame_coverage")
    if not isinstance(frame_coverage, Mapping):
        raise CollisionCensusError("revision-5 validation requires actor-visible frame coverage")
    actor_visible_frame_count = _exact_int(
        frame_coverage.get("actor_visible_selection_frame_count"),
        field="revision-5 actor-visible frame count",
        minimum=0,
    )
    processed_actor_visible_frame_count = _exact_int(
        frame_coverage.get("processed_actor_visible_selection_frame_count"),
        field="revision-5 processed actor-visible frame count",
        minimum=0,
    )
    forced_frame_count = _exact_int(
        frame_coverage.get("forced_selection_frame_count"),
        field="revision-5 forced selection frame count",
        minimum=0,
    )
    if (
        processed_actor_visible_frame_count != actor_visible_frame_count
        or forced_frame_count > actor_visible_frame_count
        or frame_coverage.get("all_actor_visible_and_forced_frames_included") is not True
    ):
        raise CollisionCensusError("revision-5 actor-visible/forced frame coverage is incomplete")
    refeaturization = census_receipt.get("re_featurization")
    if not isinstance(refeaturization, Mapping) or (
        refeaturization.get("raw_schema_inventory_sha256"),
        refeaturization.get("raw_schema_inventory_scope"),
        refeaturization.get("raw_schema_inventory_rejected_observation_count"),
        refeaturization.get("actor_visible_selection_inventory_scope"),
        refeaturization.get("actor_visible_selection_inventory_observation_count"),
        refeaturization.get("raw_observation_values_persisted"),
    ) != (
        census_receipt.get("phase_a_inventory_sha256"),
        PHASE_A_RAW_REPLAY_INVENTORY_SCOPE,
        0,
        PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE,
        processed_actor_visible_frame_count,
        False,
    ):
        raise CollisionCensusError("revision-5 re-featurization inventory bindings are incomplete")
    if refeaturization.get("raw_outer_observation_count") != raw_inventory_proof[
        "raw_observation_count"
    ]:
        raise CollisionCensusError("revision-5 raw outer-observation coverage drifted")
    actor_selection_inventory_sha256 = require_sha256(
        refeaturization.get("actor_visible_selection_inventory_sha256"),
        field="revision-5 actor-visible selection inventory",
    )
    if raw_expert_corpus_receipt.get("schema") != R298_RAW_CORPUS_RECEIPT_SCHEMA:
        raise CollisionCensusError("revision-5 validation raw corpus receipt schema drifted")
    if raw_expert_corpus_receipt.get("status") != "passed":
        raise CollisionCensusError("revision-5 validation raw corpus receipt did not pass")
    if raw_expert_corpus_receipt.get("raw_expert_corpus_manifest_sha256") != canonical_sha256(
        raw_expert_corpus_manifest
    ):
        raise CollisionCensusError("revision-5 validation raw corpus manifest binding drifted")
    raw_predecessor = validate_revision_5_predecessor_classification(
        raw_expert_corpus_manifest.get("revision_5_predecessor_classification")
    )
    raw_receipt_predecessor = validate_revision_5_predecessor_classification(
        raw_expert_corpus_receipt.get("revision_5_predecessor_classification")
    )
    if (
        raw_predecessor != revision_5_predecessor_classification()
        or raw_receipt_predecessor != revision_5_predecessor_classification()
    ):
        raise CollisionCensusError(
            "fresh raw-corpus evidence cannot consume revision-4 authority"
        )
    if census_receipt.get("raw_expert_corpus_receipt_sha256") != canonical_sha256(
        raw_expert_corpus_receipt
    ):
        raise CollisionCensusError("revision-5 validation collision/raw receipt binding drifted")
    frozen_gate = census_receipt.get("frozen_schema_gate")
    if not isinstance(frozen_gate, Mapping) or (
        frozen_gate.get("frozen_schema_manifest_sha256"),
        frozen_gate.get("zero_bypass_receipt_sha256"),
        frozen_gate.get("frozen_before_refeaturization"),
        frozen_gate.get("runtime_wired"),
    ) != (
        frozen_schema_manifest_sha256,
        zero_bypass_receipt_sha256,
        True,
        False,
    ):
        raise CollisionCensusError("revision-5 validation frozen-schema gate binding drifted")
    predecessor = merge_revision_5_predecessor_classifications(
        census_predecessor,
        raw_predecessor,
        raw_receipt_predecessor,
        validate_revision_5_predecessor_classification(
            frozen_schema_manifest.get("revision_5_predecessor_classification")
        ),
        validate_revision_5_predecessor_classification(
            zero_bypass_receipt.get("revision_5_predecessor_classification")
        ),
    )
    return {
        "schema": R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
        "status": "passed_complete_r298_census_under_revision_5",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "revision_5_predecessor_classification": predecessor,
        "raw_expert_corpus_manifest_sha256": canonical_sha256(raw_expert_corpus_manifest),
        "raw_expert_corpus_receipt_sha256": canonical_sha256(raw_expert_corpus_receipt),
        "frozen_schema_manifest_sha256": frozen_schema_manifest_sha256,
        "zero_bypass_receipt_sha256": zero_bypass_receipt_sha256,
        "collision_census_receipt_sha256": canonical_sha256(census_receipt),
        "collision_report_sha256": require_sha256(
            census_receipt.get("collision_report_sha256"),
            field="revision-5 validation collision report",
        ),
        "phase_a_inventory_sha256": require_sha256(
            census_receipt.get("phase_a_inventory_sha256"),
            field="revision-5 validation Phase A inventory",
        ),
        "actor_visible_selection_inventory_sha256": actor_selection_inventory_sha256,
        "frame_coverage_sha256": canonical_sha256(frame_coverage),
        "runtime_authority": {
            "elmo_only": True,
            "create_only": True,
            "training_eligible": False,
            "inzi_mutation": False,
            "runtime_or_service_change": False,
            "handoff_activation": False,
        },
        "validation_scope": "raw_corpus_and_collision_census_only_not_training_or_handoff",
    }


def validate_revision_5_census_validation_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a portable typed rev5 census bridge before a later aggregator.

    The caller still has to re-open/checksum the referenced artifacts.  This
    structural verifier prevents a generic r298 JSON report from being treated
    as the revision-5 migration evidence required by the canonical contract.
    """

    expected_fields = {
        "schema",
        "status",
        "owner_revision",
        "goal_revision",
        "root_handoff_revision",
        "rule_derivative_gateway_sha256",
        "rule_derivative_contract_sha256",
        "revision_5_predecessor_classification",
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "collision_census_receipt_sha256",
        "collision_report_sha256",
        "phase_a_inventory_sha256",
        "actor_visible_selection_inventory_sha256",
        "frame_coverage_sha256",
        "runtime_authority",
        "validation_scope",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CollisionCensusError("revision-5 census validation receipt field inventory drifted")
    if (
        value.get("schema"),
        value.get("status"),
        value.get("owner_revision"),
        value.get("goal_revision"),
        value.get("root_handoff_revision"),
        value.get("rule_derivative_gateway_sha256"),
        value.get("rule_derivative_contract_sha256"),
        value.get("validation_scope"),
    ) != (
        R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
        "passed_complete_r298_census_under_revision_5",
        R298_OWNER_REVISION,
        REVISION_5_GOAL_REVISION,
        REVISION_5_ROOT_HANDOFF_REVISION,
        RULE_DERIVATIVE_GATEWAY_SHA256,
        RULE_DERIVATIVE_CONTRACT_SHA256,
        "raw_corpus_and_collision_census_only_not_training_or_handoff",
    ):
        raise CollisionCensusError("revision-5 census validation receipt authority drifted")
    predecessor = validate_revision_5_predecessor_classification(
        value.get("revision_5_predecessor_classification")
    )
    for field in (
        "raw_expert_corpus_manifest_sha256",
        "raw_expert_corpus_receipt_sha256",
        "frozen_schema_manifest_sha256",
        "zero_bypass_receipt_sha256",
        "collision_census_receipt_sha256",
        "collision_report_sha256",
        "phase_a_inventory_sha256",
        "actor_visible_selection_inventory_sha256",
        "frame_coverage_sha256",
    ):
        require_sha256(value.get(field), field=f"revision-5 census validation {field}")
    authority = value.get("runtime_authority")
    if not isinstance(authority, Mapping) or authority != {
        "elmo_only": True,
        "create_only": True,
        "training_eligible": False,
        "inzi_mutation": False,
        "runtime_or_service_change": False,
        "handoff_activation": False,
    }:
        raise CollisionCensusError("revision-5 census validation receipt authority drifted")
    return _canonical_value(value, field="revision-5 census validation receipt")


__all__ = [
    "CANONICAL_R236_LIBCG_SHA256",
    "CANONICAL_R236_LIBCG_SIZE_BYTES",
    "COLLISION_CLASSES",
    "COLLISION_GENUINE_PUBLIC_NONIDENTIFIABILITY",
    "COLLISION_INTENTIONAL_HIDDEN",
    "COLLISION_INTENTIONAL_PERMUTATION",
    "COLLISION_SIMULATOR_LEGALITY_PROXY",
    "CollisionCensusError",
    "CollisionCensusFailure",
    "EXACT_NEW_LIST_MULTISET_SHA256",
    "MECHANICS_ATTACHMENT_SHA256",
    "OWNER_GOAL_SHA256",
    "R298_OWNER_REVISION",
    "REVISION_5_GOAL_REVISION",
    "REVISION_5_ROOT_HANDOFF_REVISION",
    "REVISION_4_GOAL_REVISION",
    "REVISION_4_GATEWAY_SHA256",
    "REVISION_4_CONTRACT_SHA256",
    "RULE_DERIVATIVE_CONTRACT_SHA256",
    "RULE_DERIVATIVE_GATEWAY_SHA256",
    "R298_COLLISION_RECEIPT_SCHEMA",
    "R298_COLLISION_CENSUS_SCHEMA",
    "R298_COLLISION_RECORD_SCHEMA",
    "R298_COLLISION_REPORT_SCHEMA",
    "R298_ENGINE_EVIDENCE_SCHEMA",
    "R298_PHASE_A_INVENTORY_SCHEMA",
    "R298_RAW_CORPUS_MANIFEST_SCHEMA",
    "R298_RAW_CORPUS_RECEIPT_SCHEMA",
    "R298_FROZEN_SCHEMA_MANIFEST_SCHEMA",
    "R298_ZERO_BYPASS_RECEIPT_SCHEMA",
    "R298_REFEATURED_RECORD_MANIFEST_SCHEMA",
    "R298_REFEATURED_RECORD_SHARD_SCHEMA",
    "R298_REV5_PREDECESSOR_CLASSIFICATION_SCHEMA",
    "R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA",
    "R274_EXACT_FEATURES_SOURCE_SHA256",
    "RAW_CORPUS_END_UTC",
    "RAW_CORPUS_EXACT_DAYS",
    "RAW_CORPUS_START_UTC",
    "RAW_CORPUS_SOURCE_RECEIPT_SHA256S",
    "FIELD_CLASSIFICATIONS",
    "FIELD_DIRECT_PUBLIC_LIBCG",
    "FIELD_IMMUTABLE_PUBLIC_CARD_CATALOG",
    "FIELD_INTENTIONALLY_HIDDEN",
    "FIELD_TARGET_FRESH_SIMULATOR",
    "FIELD_TARGET_RECORDED_TRANSITION",
    "PHASE_A_RAW_REPLAY_INVENTORY_SCOPE",
    "PHASE_A_ACTOR_SELECTION_INVENTORY_SCOPE",
    "STATUS_BLOCKED_EVIDENCE",
    "STATUS_FAILED_COLLISION",
    "STATUS_INVENTORY_ONLY",
    "STATUS_PASSED",
    "action_key_sha256",
    "analyze_collision_records",
    "build_stage_option_records",
    "canonical_json_bytes",
    "canonical_public_current_hash",
    "canonical_public_observation",
    "canonical_public_observation_hash",
    "canonical_sha256",
    "classify_raw_observation_path",
    "complete_semantic_option_key",
    "current_feature_token_hashes",
    "inventory_raw_observations",
    "raw_observations_from_recorded_episode",
    "make_raw_corpus_receipt",
    "build_raw_corpus_manifest",
    "validate_raw_corpus_manifest",
    "make_receipt",
    "make_revision_5_census_validation_receipt",
    "phase_a_field_catalog",
    "require_sha256",
    "semantic_option_key_sha256",
    "sha256_file",
    "stage_descriptors_from_recorded_episode",
    "validate_phase_a_inventory",
    "validate_frozen_schema_gate",
    "revision_5_predecessor_classification",
    "merge_revision_5_predecessor_classifications",
    "validate_revision_5_predecessor_classification",
    "validate_revision_5_census_validation_receipt",
    "frozen_schema_gate_contract",
]
