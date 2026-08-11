"""Fail-closed staging guard for the dormant r258 OwnDeckLedger successor.

This module intentionally has no trainer, model, simulator, service-manager, or
selector imports.  It only validates immutable JSON evidence and answers the
narrow question: which *future* r258 operation is eligible to be staged?

The answer is deliberately conservative:

* the canonical r258 manifest is the only contract accepted by the gate;
* isolated successor build and offline-test work are allowed while the refresh
  is healthy, because the manifest explicitly grants that authority;
* r259 additionally permits only the checksum-pinned, read-only Elmo
  OwnDeckLedger side-store materialization contract; it is not an active-r241
  training or existing-service authority;
* every migration, canary, evaluation, runtime activation, and promotion path
  is denied until a terminal Alakazam-refresh receipt and the appropriate
  receipt chain are independently valid; and
* package, submission, and selector operations remain denied by this manifest.

Nothing here performs an operation after it is found eligible.  A caller must
still use the separately authorized receipt-backed workflow that owns that
operation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST_PATH = ROOT / "state/alakazam-own-deck-ledger-successor-r258.json"

MANIFEST_SCHEMA = "poke_bot.alakazam_own_deck_ledger_successor_r258/v1"
STAGE_RECEIPT_SCHEMA = "poke_bot.alakazam_own_deck_ledger_stage_receipt/v1"
REFRESH_COMPLETION_SCHEMA = "poke_bot.alakazam_refresh_completion/v1"
POST_REFRESH_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_own_deck_ledger_post_refresh_receipt/v1"
)

OWNER_DECISION_REVISION = 258
LATEST_OWNER_CLARIFICATION_REVISION = 259
CANDIDATE_ID = "alakazam-own-deck-ledger-successor-r258"
_SHA256_PREFIX = "sha256:"

ELMO_SIDE_STORE_OWNER_CLARIFICATION_REVISION = 259
ELMO_SIDE_STORE_STATUS = "authorized_for_build_and_managed_start"
ELMO_SIDE_STORE_PURPOSE = "next_training_lineage_only"
ELMO_SIDE_STORE_HOST = "elmo"
ELMO_SIDE_STORE_SOURCE_MANIFEST = (
    "/mnt/Main/main/poke-bot-agent/archive/"
    "expert-r241-20260722-20260810/current.json"
)
ELMO_SIDE_STORE_SOURCE_MANIFEST_SHA256 = (
    "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
)
ELMO_SIDE_STORE_SOURCE_SNAPSHOT_ROOT = (
    "/home/admin/pokebot-own-deck-r259-src-de844af19ca6"
)
ELMO_SIDE_STORE_SEALED_RUNTIME_SNAPSHOT_ROOT = (
    "/var/lib/pokebot-own-deck-r259-src-de844af19ca6"
)
ELMO_SIDE_STORE_SEALED_RUNTIME_SOURCE_LOCK = (
    "/etc/pokebot-own-deck-r259/de844af19ca6/source-tree.lock.json"
)
ELMO_SIDE_STORE_CONTROLLER_EXPECTED_INVENTORY = (
    "/etc/pokebot-own-deck-r259/de844af19ca6/expected-file-inventory.json"
)
ELMO_SIDE_STORE_OUTPUT_ROOT = (
    "/mnt/Main/main/poke-bot-agent/archive/"
    "expert-r258-own-deck-ledger-sidecar/2026-07-22_2026-08-10"
)
ELMO_SIDE_STORE_MANAGED_SERVICE = "pokebot-own-deck-rollout-store-r259.service"
ELMO_SIDE_STORE_CONTAINER_IMAGE = "poke-bot-truenas-worker:matchup-v33-runtime"
ELMO_SIDE_STORE_CONTAINER_IMAGE_ID = (
    "sha256:74d66c41fda841e96ee89e88fab1fa800b82ab8c6a06cabdff146803a1b05a0f"
)
ELMO_SIDE_STORE_CLASSIFIER_ASSET_ROOT = (
    "/home/admin/pokebot-expert-src-v6-strategic/data/training_mixes"
)
ELMO_SIDE_STORE_SOURCE_WINDOW = (
    "2026-07-22",
    "2026-08-10",
    20,
    91_253,
)
ELMO_SIDE_STORE_SOURCE_ACCESS = "read_only"
ELMO_SIDE_STORE_RECORD_SCOPE = (
    "acting-seat records whose exact starting deck is classified as Alakazam"
)
ELMO_SIDE_STORE_RECORD_KEY = (
    "episode_id",
    "seat",
    "env_step",
    "observation_fingerprint",
)
ELMO_SIDE_STORE_STORAGE_CONTRACT = (
    "daily atomic idempotent gzip JSONL sidecars plus immutable metadata and sha256 receipts"
)


class OwnDeckSuccessorError(RuntimeError):
    """The r258 manifest or an evidence receipt is malformed or untrusted."""


class OwnDeckSuccessorGateError(OwnDeckSuccessorError):
    """A requested successor operation is not receipt-authorized."""


class OwnDeckSuccessorStage(str, Enum):
    """The exact ordered r258 implementation and release stages."""

    LEDGER_CORE = "ledger_core"
    SHARED_ALL_HEAD_INPUT = "shared_all_head_input"
    VISIBLE_TUTOR_LEARNING = "visible_tutor_learning"
    TERMINAL_CONVERSION = "terminal_conversion"
    DATASET_RUNTIME_REMOTE_PARITY = "dataset_runtime_remote_parity"
    ELMO_EXPERT_ROLLOUT_SIDE_STORE = "elmo_expert_rollout_side_store"
    SHADOW_ABLATIONS = "shadow_ablations"
    POST_REFRESH_TRAINING_CANARY_EVAL_PROMOTION = (
        "post_refresh_training_canary_eval_promotion"
    )


class OwnDeckSuccessorStageStatus(str, Enum):
    """Typed values used by the canonical manifest and accepted receipts."""

    STAGED_DORMANT = "staged_dormant"
    BLOCKED_PENDING_REFRESH_COMPLETION_AND_PRIOR_STAGE_RECEIPTS = (
        "blocked_pending_refresh_completion_and_prior_stage_receipts"
    )
    AUTHORIZED_BUILDING = "authorized_building"
    PASSED = "passed"


class ElmoOwnDeckSideStoreStatus(str, Enum):
    """The one r259 side-store lifecycle status accepted before refresh completion."""

    AUTHORIZED_FOR_BUILD_AND_MANAGED_START = ELMO_SIDE_STORE_STATUS


class OwnDeckSuccessorOperation(str, Enum):
    """Operations the guard can classify without performing any of them."""

    ISOLATED_BUILD = "isolated_build"
    OFFLINE_TEST = "offline_test"
    ELMO_SIDE_STORE_MATERIALIZATION = "elmo_side_store_materialization"
    ISOLATED_MIGRATION = "isolated_migration"
    TRAINING_CANARY = "training_canary"
    SOURCE_DISJOINT_EVALUATION = "source_disjoint_evaluation"
    RUNTIME_ACTIVATION = "runtime_activation"
    PROMOTION = "promotion"
    SELECTOR_CHANGE = "selector_change"
    PACKAGE_CREATION = "package_creation"
    SUBMISSION = "submission"


class OwnDeckSuccessorPostRefreshReceiptKind(str, Enum):
    """Distinct post-refresh receipts; none substitutes for another."""

    ISOLATED_MIGRATION = "isolated_migration"
    TRAINING_CANARY = "training_canary"
    SOURCE_DISJOINT_EVALUATION = "source_disjoint_evaluation"
    RUNTIME_ACTIVATION = "runtime_activation"
    PROMOTION = "promotion"


_STAGE_ORDER = tuple(OwnDeckSuccessorStage)
_PRE_REFRESH_STAGES = _STAGE_ORDER[:-1]
_EXPECTED_STAGE_STATUS = {
    **{
        stage: OwnDeckSuccessorStageStatus.STAGED_DORMANT
        for stage in _PRE_REFRESH_STAGES
    },
    OwnDeckSuccessorStage.POST_REFRESH_TRAINING_CANARY_EVAL_PROMOTION: (
        OwnDeckSuccessorStageStatus.BLOCKED_PENDING_REFRESH_COMPLETION_AND_PRIOR_STAGE_RECEIPTS
    ),
}
_EXPECTED_STAGE_STATUS[
    OwnDeckSuccessorStage.ELMO_EXPERT_ROLLOUT_SIDE_STORE
] = OwnDeckSuccessorStageStatus.AUTHORIZED_BUILDING
_OPERATION_ALIASES = {
    "build": OwnDeckSuccessorOperation.ISOLATED_BUILD,
    "isolated_build": OwnDeckSuccessorOperation.ISOLATED_BUILD,
    "test": OwnDeckSuccessorOperation.OFFLINE_TEST,
    "offline_test": OwnDeckSuccessorOperation.OFFLINE_TEST,
    "elmo_side_store_materialization": (
        OwnDeckSuccessorOperation.ELMO_SIDE_STORE_MATERIALIZATION
    ),
    "side_store_materialization": (
        OwnDeckSuccessorOperation.ELMO_SIDE_STORE_MATERIALIZATION
    ),
    "elmo_side_store": OwnDeckSuccessorOperation.ELMO_SIDE_STORE_MATERIALIZATION,
    "side_store": OwnDeckSuccessorOperation.ELMO_SIDE_STORE_MATERIALIZATION,
    "migration": OwnDeckSuccessorOperation.ISOLATED_MIGRATION,
    "isolated_migration": OwnDeckSuccessorOperation.ISOLATED_MIGRATION,
    "training": OwnDeckSuccessorOperation.TRAINING_CANARY,
    "training_canary": OwnDeckSuccessorOperation.TRAINING_CANARY,
    "evaluation": OwnDeckSuccessorOperation.SOURCE_DISJOINT_EVALUATION,
    "source_disjoint_evaluation": OwnDeckSuccessorOperation.SOURCE_DISJOINT_EVALUATION,
    "runtime": OwnDeckSuccessorOperation.RUNTIME_ACTIVATION,
    "runtime_activation": OwnDeckSuccessorOperation.RUNTIME_ACTIVATION,
    "activation": OwnDeckSuccessorOperation.RUNTIME_ACTIVATION,
    "promotion": OwnDeckSuccessorOperation.PROMOTION,
    "selector": OwnDeckSuccessorOperation.SELECTOR_CHANGE,
    "selector_change": OwnDeckSuccessorOperation.SELECTOR_CHANGE,
    "package": OwnDeckSuccessorOperation.PACKAGE_CREATION,
    "package_creation": OwnDeckSuccessorOperation.PACKAGE_CREATION,
    "submission": OwnDeckSuccessorOperation.SUBMISSION,
}


@dataclass(frozen=True)
class ManifestIdentity:
    """Validated immutable identity for the one canonical owner contract."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class ElmoOwnDeckSideStoreWindow:
    """The immutable source-corpus range accepted by r259."""

    start_date: str
    end_date: str
    day_count: int
    validated_episode_count: int


@dataclass(frozen=True)
class ElmoOwnDeckSideStoreContract:
    """The narrowly authorized r259 Elmo-only materialization contract.

    This is an identity projection, not a service-control object.  Validating
    it never starts the service; it merely ensures a caller cannot substitute
    another host, corpus, source snapshot, output root, or service identity.
    """

    owner_clarification_revision: int
    status: ElmoOwnDeckSideStoreStatus
    purpose: str
    host: str
    source_manifest: str
    source_manifest_sha256: str
    source_window: ElmoOwnDeckSideStoreWindow
    source_access: str
    record_scope: str
    source_snapshot_root: str
    sealed_runtime_snapshot_root: str
    sealed_runtime_source_lock: str
    controller_expected_inventory: str
    output_root: str
    managed_service: str
    container_image: str
    container_image_id: str
    classifier_asset_root: str
    record_key: tuple[str, ...]
    storage_contract: str
    source_corpus_mutation_allowed: bool
    current_r241_training_eligible: bool
    post_refresh_successor_training_eligible_after_join_and_parity_receipt: bool


@dataclass(frozen=True)
class OwnDeckSuccessorManifest:
    """The small typed projection of the canonical r258 manifest."""

    identity: ManifestIdentity
    stages: tuple[OwnDeckSuccessorStage, ...]
    elmo_side_store: ElmoOwnDeckSideStoreContract
    raw: Mapping[str, Any]

    @property
    def pre_refresh_stages(self) -> tuple[OwnDeckSuccessorStage, ...]:
        return self.stages[:-1]


@dataclass(frozen=True)
class ReceiptIdentity:
    """A validated evidence receipt and its content digest."""

    sha256: str
    payload: Mapping[str, Any]
    path: Path | None = None


@dataclass(frozen=True)
class StageReceipt(ReceiptIdentity):
    """A receipt proving one pre-refresh r258 implementation stage passed."""

    stage: OwnDeckSuccessorStage = OwnDeckSuccessorStage.LEDGER_CORE


@dataclass(frozen=True)
class RefreshCompletionReceipt(ReceiptIdentity):
    """A terminal Alakazam-refresh completion receipt accepted by r258."""

    checkpoint_sha256: str = ""


@dataclass(frozen=True)
class PostRefreshReceipt(ReceiptIdentity):
    """A distinct receipt in the post-refresh migration-to-promotion chain."""

    kind: OwnDeckSuccessorPostRefreshReceiptKind = (
        OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION
    )


@dataclass(frozen=True)
class OwnDeckSuccessorGateDecision:
    """Pure gate result; it has no authority to carry out the operation."""

    operation: OwnDeckSuccessorOperation
    allowed: bool
    reason: str
    manifest_sha256: str | None
    refresh_completion_sha256: str | None = None
    accepted_stage_receipts: tuple[OwnDeckSuccessorStage, ...] = ()


def canonical_json(value: object) -> bytes:
    """Return deterministic JSON bytes suitable for receipt fingerprints."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OwnDeckSuccessorError("value is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    """Return a typed SHA-256 digest for immutable JSON or file bytes."""

    return _SHA256_PREFIX + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash one regular evidence file without following a symlink."""

    source = _regular_file(path, label="receipt source")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def receipt_digest(payload: Mapping[str, Any]) -> str:
    """Digest a receipt after removing its optional self-fingerprint field."""

    detached = dict(payload)
    detached.pop("receipt_sha256", None)
    return sha256_bytes(canonical_json(detached))


def stage_receipt_digest(payload: Mapping[str, Any]) -> str:
    """Return the detached canonical digest used by an r258 stage receipt."""

    return receipt_digest(payload)


def seal_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical-copy receipt with its detached SHA-256 fingerprint.

    This helper is intentionally side-effect free.  It is useful for offline
    fixtures and future create-only receipt writers, but cannot authorize a
    service, selector, training run, package, or submission.
    """

    try:
        sealed = json.loads(canonical_json(dict(payload)).decode("utf-8"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnDeckSuccessorError("receipt cannot be canonically sealed") from exc
    if not isinstance(sealed, dict):  # Defensive; canonical JSON above starts from a dict.
        raise OwnDeckSuccessorError("receipt must be an object")
    sealed.pop("receipt_sha256", None)
    sealed["receipt_sha256"] = receipt_digest(sealed)
    return sealed


def load_canonical_manifest(
    path: Path | str = CANONICAL_MANIFEST_PATH,
) -> OwnDeckSuccessorManifest:
    """Load the one canonical r258 manifest; aliases and symlinks fail closed."""

    source = _regular_file(path, label="r258 canonical manifest")
    expected = CANONICAL_MANIFEST_PATH.resolve()
    if source != expected:
        raise OwnDeckSuccessorError(
            f"r258 manifest must be the canonical path: {CANONICAL_MANIFEST_PATH}"
        )
    payload = _read_json_object(source, label="r258 canonical manifest")
    return validate_canonical_manifest(
        payload,
        identity=ManifestIdentity(path=source, sha256=sha256_file(source)),
    )


def validate_canonical_manifest(
    payload: Mapping[str, Any],
    *,
    identity: ManifestIdentity | None = None,
) -> OwnDeckSuccessorManifest:
    """Validate r258 intent itself, independently of mutable runtime state."""

    manifest = _mapping(payload, label="r258 manifest")
    _require_exact(manifest.get("schema"), MANIFEST_SCHEMA, label="manifest schema")
    _require_exact(manifest.get("canonical"), True, label="manifest canonical flag")
    _require_exact(
        manifest.get("owner_decision_revision"),
        OWNER_DECISION_REVISION,
        label="manifest owner revision",
    )
    _require_exact(
        manifest.get("latest_owner_clarification_revision"),
        LATEST_OWNER_CLARIFICATION_REVISION,
        label="manifest latest owner clarification revision",
    )
    _require_exact(
        manifest.get("latest_owner_clarification_recorded_at_utc"),
        "2026-08-11T06:39:05Z",
        label="manifest latest owner clarification timestamp",
    )
    _require_exact(
        manifest.get("candidate_id"), CANDIDATE_ID, label="manifest candidate identity"
    )
    _require_exact(
        manifest.get("specialist_id"), "alakazam", label="manifest specialist"
    )
    _require_exact(
        manifest.get("status"),
        OwnDeckSuccessorStageStatus.STAGED_DORMANT.value,
        label="manifest status",
    )

    lineage = _mapping(manifest.get("lineage"), label="manifest lineage")
    _require_exact(
        lineage.get("predecessor_manifest"),
        "state/alakazam-new-list-direct-policy-r241.json",
        label="manifest predecessor",
    )
    _require_exact(
        lineage.get("predecessor_revision"), 241, label="manifest predecessor revision"
    )
    _require_exact(
        lineage.get("predecessor_is_immutable"),
        True,
        label="manifest predecessor immutability",
    )
    _require_exact(
        lineage.get("successor_isolated_until_activation"),
        True,
        label="manifest successor isolation",
    )
    _require_exact(
        lineage.get("active_r241_mutation_allowed"),
        False,
        label="manifest active-r241 mutation authority",
    )

    authority = _mapping(manifest.get("authority"), label="manifest authority")
    _require_exact(
        authority.get("isolated_successor_build_and_offline_test_now"),
        True,
        label="isolated build/test authority",
    )
    for field in (
        "new_elmo_side_store_managed_service_start",
        "elmo_expert_rollout_side_store_materialization_now",
    ):
        _require_exact(authority.get(field), True, label=f"manifest authority {field}")
    for field in (
        "active_lineage_implementation_before_refresh_completion_receipt",
        "active_lineage_activation_before_refresh_completion_receipt",
        "r241_training",
        "existing_managed_service_start_stop_restart_or_reconfigure",
        "elmo_side_store_may_feed_active_r241",
        "selector_change",
        "runtime_package_creation_or_publication",
        "submission",
        "automatic_promotion",
    ):
        _require_exact(authority.get(field), False, label=f"manifest authority {field}")
    _require_exact(
        authority.get("post_refresh_isolated_migration_training_and_evaluation"),
        True,
        label="post-refresh staged authority",
    )

    dependency = _mapping(
        manifest.get("refresh_completion_dependency"),
        label="refresh completion dependency",
    )
    _require_exact(dependency.get("required"), True, label="refresh completion required")
    _require_exact(
        dependency.get("receipt_schema"),
        REFRESH_COMPLETION_SCHEMA,
        label="refresh completion schema",
    )
    _require_exact(
        dependency.get("receipt_status"),
        "not_yet_proven",
        label="refresh completion status",
    )
    _require_exact(
        dependency.get("absence_fails_closed"),
        True,
        label="refresh completion absence behavior",
    )
    required_before = _string_set(
        dependency.get("required_before"), label="refresh required-before operations"
    )
    if {
        "active_lineage_implementation",
        "runtime_activation",
        "training_canary",
        "evaluation_promotion",
    } - required_before:
        raise OwnDeckSuccessorError("refresh dependency omits a protected operation")

    invariants = _mapping(manifest.get("invariants"), label="manifest invariants")
    for field in (
        "direct_policy_only",
        "no_mcts",
        "no_recursive_turn_planner",
        "no_search_target_generation",
        "no_hidden_snapshot_labels",
        "no_guide_runtime_action_authority",
        "no_selector_or_submission_authority",
        "all_existing_decision_heads_remain_present_and_trainable_when_a_successor_is_explicitly_activated",
        "combo_head_status_must_not_be_changed_by_this_manifest",
        "r241_exact_deck_guide_and_public_matchup_selector_remain_unchanged",
        "source_manifest_and_runtime_receipts_remain_authoritative",
        "successor_outputs_are_bounded_option_conditioned_inputs_not_action_overrides",
        "unknown_deck_or_prize_identity_is_never_materialized",
    ):
        _require_exact(invariants.get(field), True, label=f"manifest invariant {field}")

    stages = _validate_manifest_stages(manifest.get("stages"))
    elmo_side_store = validate_elmo_side_store_contract(
        manifest.get("elmo_expert_rollout_side_store")
    )
    _validate_receipt_plan(manifest.get("receipt_plan"))

    if identity is None:
        # A mapping can be schema-validated for focused tests, but is not a
        # gate-eligible canonical owner source until it is loaded from the path.
        identity = ManifestIdentity(
            path=CANONICAL_MANIFEST_PATH.resolve(),
            sha256=sha256_bytes(canonical_json(manifest)),
        )
    return OwnDeckSuccessorManifest(
        identity=identity,
        stages=stages,
        elmo_side_store=elmo_side_store,
        raw=manifest,
    )


def validate_elmo_side_store_contract(value: object) -> ElmoOwnDeckSideStoreContract:
    """Validate the one r259 materialization contract exactly.

    The contract is deliberately data-only.  Passing validation does not start
    its service, touch the r241 corpus, or make the resulting records eligible
    for r241 or pre-refresh successor training.
    """

    side_store = _mapping(value, label="r259 Elmo side-store contract")
    _require_exact(
        side_store.get("owner_clarification_revision"),
        ELMO_SIDE_STORE_OWNER_CLARIFICATION_REVISION,
        label="Elmo side-store owner clarification revision",
    )
    _require_exact(
        side_store.get("status"),
        ElmoOwnDeckSideStoreStatus.AUTHORIZED_FOR_BUILD_AND_MANAGED_START.value,
        label="Elmo side-store status",
    )
    _require_exact(
        side_store.get("purpose"),
        ELMO_SIDE_STORE_PURPOSE,
        label="Elmo side-store purpose",
    )
    _require_exact(side_store.get("host"), ELMO_SIDE_STORE_HOST, label="Elmo side-store host")
    _require_exact(
        side_store.get("source_manifest"),
        ELMO_SIDE_STORE_SOURCE_MANIFEST,
        label="Elmo side-store source manifest",
    )
    _require_exact(
        side_store.get("source_manifest_sha256"),
        ELMO_SIDE_STORE_SOURCE_MANIFEST_SHA256,
        label="Elmo side-store source manifest digest",
    )
    _sha256_text(
        side_store.get("source_manifest_sha256"),
        label="Elmo side-store source manifest digest",
    )

    source_window = _mapping(
        side_store.get("source_window"), label="Elmo side-store source window"
    )
    expected_start, expected_end, expected_days, expected_episodes = (
        ELMO_SIDE_STORE_SOURCE_WINDOW
    )
    _require_exact(
        source_window.get("start_date"),
        expected_start,
        label="Elmo side-store source start date",
    )
    _require_exact(
        source_window.get("end_date"),
        expected_end,
        label="Elmo side-store source end date",
    )
    _require_exact(
        source_window.get("day_count"),
        expected_days,
        label="Elmo side-store source day count",
    )
    _require_exact(
        source_window.get("validated_episode_count"),
        expected_episodes,
        label="Elmo side-store validated episode count",
    )

    _require_exact(
        side_store.get("source_access"),
        ELMO_SIDE_STORE_SOURCE_ACCESS,
        label="Elmo side-store source access",
    )
    _require_exact(
        side_store.get("record_scope"),
        ELMO_SIDE_STORE_RECORD_SCOPE,
        label="Elmo side-store record scope",
    )
    _require_exact(
        side_store.get("source_snapshot_root"),
        ELMO_SIDE_STORE_SOURCE_SNAPSHOT_ROOT,
        label="Elmo side-store source snapshot root",
    )
    _require_exact(
        side_store.get("sealed_runtime_snapshot_root"),
        ELMO_SIDE_STORE_SEALED_RUNTIME_SNAPSHOT_ROOT,
        label="Elmo side-store sealed runtime snapshot root",
    )
    _require_exact(
        side_store.get("sealed_runtime_source_lock"),
        ELMO_SIDE_STORE_SEALED_RUNTIME_SOURCE_LOCK,
        label="Elmo side-store sealed runtime source lock",
    )
    _require_exact(
        side_store.get("controller_expected_inventory"),
        ELMO_SIDE_STORE_CONTROLLER_EXPECTED_INVENTORY,
        label="Elmo side-store controller expected inventory",
    )
    _require_exact(
        side_store.get("output_root"),
        ELMO_SIDE_STORE_OUTPUT_ROOT,
        label="Elmo side-store output root",
    )
    _require_exact(
        side_store.get("managed_service"),
        ELMO_SIDE_STORE_MANAGED_SERVICE,
        label="Elmo side-store managed service",
    )
    _require_exact(
        side_store.get("container_image"),
        ELMO_SIDE_STORE_CONTAINER_IMAGE,
        label="Elmo side-store container image",
    )
    _require_exact(
        side_store.get("container_image_id"),
        ELMO_SIDE_STORE_CONTAINER_IMAGE_ID,
        label="Elmo side-store container image digest",
    )
    _sha256_text(
        side_store.get("container_image_id"),
        label="Elmo side-store container image digest",
    )
    _require_exact(
        side_store.get("classifier_asset_root"),
        ELMO_SIDE_STORE_CLASSIFIER_ASSET_ROOT,
        label="Elmo side-store classifier asset root",
    )

    raw_record_key = side_store.get("record_key")
    if (
        not isinstance(raw_record_key, list)
        or not all(type(field) is str and field for field in raw_record_key)
        or tuple(raw_record_key) != ELMO_SIDE_STORE_RECORD_KEY
    ):
        raise OwnDeckSuccessorError(
            "Elmo side-store record key is not the exact r259 identity key"
        )
    _require_exact(
        side_store.get("storage_contract"),
        ELMO_SIDE_STORE_STORAGE_CONTRACT,
        label="Elmo side-store storage contract",
    )
    _require_exact(
        side_store.get("source_corpus_mutation_allowed"),
        False,
        label="Elmo side-store source corpus mutation authority",
    )
    _require_exact(
        side_store.get("current_r241_training_eligible"),
        False,
        label="Elmo side-store active-r241 training eligibility",
    )
    _require_exact(
        side_store.get(
            "post_refresh_successor_training_eligible_after_join_and_parity_receipt"
        ),
        True,
        label="Elmo side-store post-refresh successor eligibility",
    )

    return ElmoOwnDeckSideStoreContract(
        owner_clarification_revision=ELMO_SIDE_STORE_OWNER_CLARIFICATION_REVISION,
        status=ElmoOwnDeckSideStoreStatus.AUTHORIZED_FOR_BUILD_AND_MANAGED_START,
        purpose=ELMO_SIDE_STORE_PURPOSE,
        host=ELMO_SIDE_STORE_HOST,
        source_manifest=ELMO_SIDE_STORE_SOURCE_MANIFEST,
        source_manifest_sha256=ELMO_SIDE_STORE_SOURCE_MANIFEST_SHA256,
        source_window=ElmoOwnDeckSideStoreWindow(
            start_date=expected_start,
            end_date=expected_end,
            day_count=expected_days,
            validated_episode_count=expected_episodes,
        ),
        source_access=ELMO_SIDE_STORE_SOURCE_ACCESS,
        record_scope=ELMO_SIDE_STORE_RECORD_SCOPE,
        source_snapshot_root=ELMO_SIDE_STORE_SOURCE_SNAPSHOT_ROOT,
        sealed_runtime_snapshot_root=ELMO_SIDE_STORE_SEALED_RUNTIME_SNAPSHOT_ROOT,
        sealed_runtime_source_lock=ELMO_SIDE_STORE_SEALED_RUNTIME_SOURCE_LOCK,
        controller_expected_inventory=ELMO_SIDE_STORE_CONTROLLER_EXPECTED_INVENTORY,
        output_root=ELMO_SIDE_STORE_OUTPUT_ROOT,
        managed_service=ELMO_SIDE_STORE_MANAGED_SERVICE,
        container_image=ELMO_SIDE_STORE_CONTAINER_IMAGE,
        container_image_id=ELMO_SIDE_STORE_CONTAINER_IMAGE_ID,
        classifier_asset_root=ELMO_SIDE_STORE_CLASSIFIER_ASSET_ROOT,
        record_key=ELMO_SIDE_STORE_RECORD_KEY,
        storage_contract=ELMO_SIDE_STORE_STORAGE_CONTRACT,
        source_corpus_mutation_allowed=False,
        current_r241_training_eligible=False,
        post_refresh_successor_training_eligible_after_join_and_parity_receipt=True,
    )


def validate_refresh_completion_receipt(
    receipt: Mapping[str, Any] | Path | str,
    *,
    manifest: OwnDeckSuccessorManifest | None = None,
) -> RefreshCompletionReceipt:
    """Validate the terminal refresh proof required before any active work.

    The manifest deliberately names required evidence categories rather than a
    live service path.  This validator fixes a narrow, auditable receipt shape
    for r258: immutable lineage, completed boundary, checkpoint, and separate
    source/runtime chain proofs must all be present and digest-bound.
    """

    contract = manifest or load_canonical_manifest()
    identity = _load_receipt(receipt, label="Alakazam refresh completion receipt")
    payload = identity.payload
    _require_exact(
        payload.get("schema"), REFRESH_COMPLETION_SCHEMA, label="refresh receipt schema"
    )
    _require_exact(payload.get("status"), "completed", label="refresh receipt status")
    _require_exact(
        payload.get("specialist_id"), "alakazam", label="refresh receipt specialist"
    )
    _require_exact(
        payload.get("terminal_completion"), True, label="refresh terminal completion"
    )
    _require_exact(payload.get("frozen"), True, label="refresh frozen state")
    _require_exact(payload.get("registered"), True, label="refresh registration state")
    _require_exact(
        payload.get("candidate_id"), CANDIDATE_ID, label="refresh candidate identity"
    )
    _require_exact(
        payload.get("manifest_sha256"),
        contract.identity.sha256,
        label="refresh manifest binding",
    )
    _validate_receipt_self_digest(payload, label="refresh completion receipt")
    _identity_object(
        payload.get("immutable_refresh_lineage"), label="immutable refresh lineage"
    )
    _identity_object(
        payload.get("completed_refresh_boundary"), label="completed refresh boundary"
    )
    checkpoint = _identity_object(payload.get("checkpoint"), label="refresh checkpoint")
    _chain_integrity(
        payload.get("source_receipt_chain"), label="refresh source receipt chain"
    )
    _chain_integrity(
        payload.get("runtime_receipt_chain"), label="refresh runtime receipt chain"
    )
    return RefreshCompletionReceipt(
        sha256=identity.sha256,
        payload=payload,
        path=identity.path,
        checkpoint_sha256=str(checkpoint["sha256"]),
    )


def validate_stage_receipt(
    receipt: Mapping[str, Any] | Path | str,
    *,
    manifest: OwnDeckSuccessorManifest | None = None,
    expected_stage: OwnDeckSuccessorStage | str | None = None,
) -> StageReceipt:
    """Validate one pre-refresh stage receipt without granting runtime authority."""

    contract = manifest or load_canonical_manifest()
    identity = _load_receipt(receipt, label="r258 stage receipt")
    payload = identity.payload
    _require_exact(payload.get("schema"), STAGE_RECEIPT_SCHEMA, label="stage receipt schema")
    _require_exact(
        payload.get("candidate_id"), CANDIDATE_ID, label="stage receipt candidate identity"
    )
    _require_exact(
        payload.get("owner_decision_revision"),
        OWNER_DECISION_REVISION,
        label="stage receipt owner revision",
    )
    _require_exact(
        payload.get("manifest_sha256"),
        contract.identity.sha256,
        label="stage receipt manifest binding",
    )
    _require_exact(
        payload.get("status"),
        OwnDeckSuccessorStageStatus.PASSED.value,
        label="stage receipt status",
    )
    stage = _coerce_stage(payload.get("stage_id"), label="stage receipt stage")
    if stage not in contract.pre_refresh_stages:
        raise OwnDeckSuccessorError(
            "the post-refresh release stage cannot be represented by a pre-refresh stage receipt"
        )
    if expected_stage is not None and stage != _coerce_stage(
        expected_stage, label="expected stage"
    ):
        raise OwnDeckSuccessorError("stage receipt does not bind the expected stage")
    _digest_mapping(payload.get("source_sha256s"), label="stage receipt source digests")
    command = payload.get("test_command_or_fixture_identity")
    if not isinstance(command, str) or not command.strip():
        raise OwnDeckSuccessorError("stage receipt test command or fixture identity is absent")
    if payload.get("test_result") not in (True, "passed"):
        raise OwnDeckSuccessorError("stage receipt test result is not passed")
    for field in (
        "public_information_audit",
        "direct_policy_audit",
        "r241_nonmutation_audit",
    ):
        _require_exact(payload.get(field), True, label=f"stage receipt {field}")
    _validate_receipt_self_digest(payload, label="stage receipt")
    return StageReceipt(
        sha256=identity.sha256,
        payload=payload,
        path=identity.path,
        stage=stage,
    )


def validate_prior_stage_receipts(
    receipts: Mapping[OwnDeckSuccessorStage | str, Mapping[str, Any] | Path | str],
    *,
    manifest: OwnDeckSuccessorManifest | None = None,
) -> dict[OwnDeckSuccessorStage, StageReceipt]:
    """Validate every pre-refresh receipt and its digest dependency chain."""

    contract = manifest or load_canonical_manifest()
    normalized: dict[OwnDeckSuccessorStage, Mapping[str, Any] | Path | str] = {}
    for raw_stage, raw_receipt in receipts.items():
        stage = _coerce_stage(raw_stage, label="stage receipt mapping key")
        if stage in normalized:
            raise OwnDeckSuccessorError(f"duplicate stage receipt: {stage.value}")
        normalized[stage] = raw_receipt
    expected = set(contract.pre_refresh_stages)
    if set(normalized) != expected:
        missing = sorted(stage.value for stage in expected - set(normalized))
        extra = sorted(stage.value for stage in set(normalized) - expected)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("unexpected=" + ",".join(extra))
        raise OwnDeckSuccessorError("prior stage receipts are incomplete (" + "; ".join(detail) + ")")

    validated: dict[OwnDeckSuccessorStage, StageReceipt] = {}
    for stage in contract.pre_refresh_stages:
        parsed = validate_stage_receipt(
            normalized[stage], manifest=contract, expected_stage=stage
        )
        declared_prior = _digest_mapping(
            parsed.payload.get("prior_stage_receipt_sha256s", {}),
            label=f"{stage.value} prior stage receipt digests",
            permit_empty=True,
        )
        expected_prior = {prior.value: validated[prior].sha256 for prior in validated}
        if declared_prior != expected_prior:
            raise OwnDeckSuccessorError(
                f"{stage.value} prior-stage receipt chain does not match validated evidence"
            )
        validated[stage] = parsed
    return validated


def validate_post_refresh_receipt(
    receipt: Mapping[str, Any] | Path | str,
    *,
    kind: OwnDeckSuccessorPostRefreshReceiptKind | str,
    manifest: OwnDeckSuccessorManifest | None = None,
    refresh_completion: RefreshCompletionReceipt,
    stage_receipts: Mapping[OwnDeckSuccessorStage, StageReceipt],
    required_dependencies: Mapping[
        OwnDeckSuccessorPostRefreshReceiptKind, PostRefreshReceipt
    ] | None = None,
) -> PostRefreshReceipt:
    """Validate a distinct post-refresh receipt and its evidence bindings."""

    contract = manifest or load_canonical_manifest()
    receipt_kind = _coerce_post_refresh_kind(kind, label="post-refresh receipt kind")
    identity = _load_receipt(receipt, label=f"{receipt_kind.value} receipt")
    payload = identity.payload
    _require_exact(
        payload.get("schema"), POST_REFRESH_RECEIPT_SCHEMA, label="post-refresh receipt schema"
    )
    _require_exact(
        payload.get("kind"), receipt_kind.value, label="post-refresh receipt kind"
    )
    _require_exact(
        payload.get("status"), OwnDeckSuccessorStageStatus.PASSED.value, label="post-refresh receipt status"
    )
    _require_exact(
        payload.get("candidate_id"), CANDIDATE_ID, label="post-refresh candidate identity"
    )
    _require_exact(
        payload.get("owner_decision_revision"),
        OWNER_DECISION_REVISION,
        label="post-refresh owner revision",
    )
    _require_exact(
        payload.get("manifest_sha256"),
        contract.identity.sha256,
        label="post-refresh manifest binding",
    )
    _require_exact(
        payload.get("refresh_completion_receipt_sha256"),
        refresh_completion.sha256,
        label="post-refresh refresh-completion binding",
    )
    expected_stages = {stage.value: parsed.sha256 for stage, parsed in stage_receipts.items()}
    declared_stages = _digest_mapping(
        payload.get("prior_stage_receipt_sha256s"),
        label="post-refresh prior-stage receipt digests",
    )
    if declared_stages != expected_stages:
        raise OwnDeckSuccessorError(
            "post-refresh receipt does not bind the complete validated stage chain"
        )
    _validate_receipt_self_digest(payload, label="post-refresh receipt")

    dependencies = required_dependencies or {}
    declared_dependencies = _digest_mapping(
        payload.get("depends_on_receipt_sha256s", {}),
        label="post-refresh dependency digests",
        permit_empty=True,
    )
    expected_dependencies = {
        dependency_kind.value: dependency.sha256
        for dependency_kind, dependency in dependencies.items()
    }
    if declared_dependencies != expected_dependencies:
        raise OwnDeckSuccessorError(
            "post-refresh receipt dependency chain does not match validated evidence"
        )
    return PostRefreshReceipt(
        sha256=identity.sha256,
        payload=payload,
        path=identity.path,
        kind=receipt_kind,
    )


def evaluate_successor_gate(
    operation: OwnDeckSuccessorOperation | str,
    *,
    refresh_completion_receipt: Mapping[str, Any] | Path | str | None = None,
    stage_receipts: Mapping[OwnDeckSuccessorStage | str, Mapping[str, Any] | Path | str]
    | None = None,
    post_refresh_receipts: Mapping[
        OwnDeckSuccessorPostRefreshReceiptKind | str, Mapping[str, Any] | Path | str
    ]
    | None = None,
    manifest: OwnDeckSuccessorManifest | None = None,
) -> OwnDeckSuccessorGateDecision:
    """Return a fail-closed eligibility decision without executing anything.

    ``isolated_build`` and ``offline_test`` are allowed without a refresh
    receipt.  The only managed-work exception is the separately typed r259
    Elmo side-store materialization contract.  Every migration, training,
    evaluation, runtime, and promotion path still validates evidence in-order
    and returns a denial rather than attempting a partial migration or a
    best-effort runtime activation.
    """

    op = _coerce_operation(operation)
    try:
        contract = load_canonical_manifest()
        if manifest is not None:
            _require_canonical_contract(manifest, canonical=contract)
    except OwnDeckSuccessorError as exc:
        return OwnDeckSuccessorGateDecision(op, False, str(exc), None)

    if op in {
        OwnDeckSuccessorOperation.ISOLATED_BUILD,
        OwnDeckSuccessorOperation.OFFLINE_TEST,
    }:
        return OwnDeckSuccessorGateDecision(
            operation=op,
            allowed=True,
            reason="r258 authorizes isolated successor build and offline tests only",
            manifest_sha256=contract.identity.sha256,
        )

    if op == OwnDeckSuccessorOperation.ELMO_SIDE_STORE_MATERIALIZATION:
        side_store = contract.elmo_side_store
        return OwnDeckSuccessorGateDecision(
            operation=op,
            allowed=True,
            reason=(
                "r259 authorizes only the exact typed Elmo read-only side-store "
                f"materialization contract ({side_store.managed_service}); it grants "
                "no active-r241, existing-service, training, runtime, selector, "
                "package, or submission authority"
            ),
            manifest_sha256=contract.identity.sha256,
        )

    if op in {
        OwnDeckSuccessorOperation.SELECTOR_CHANGE,
        OwnDeckSuccessorOperation.PACKAGE_CREATION,
        OwnDeckSuccessorOperation.SUBMISSION,
    }:
        return OwnDeckSuccessorGateDecision(
            operation=op,
            allowed=False,
            reason=(
                f"r258 does not authorize {op.value}; selector, package, and submission "
                "authority remain independently denied"
            ),
            manifest_sha256=contract.identity.sha256,
        )

    if refresh_completion_receipt is None:
        return _denied_missing_refresh(op, contract)
    if stage_receipts is None:
        return OwnDeckSuccessorGateDecision(
            op,
            False,
            "r258 requires all validated prior stage receipts before active successor work",
            contract.identity.sha256,
        )

    try:
        refresh = validate_refresh_completion_receipt(
            refresh_completion_receipt, manifest=contract
        )
        stages = validate_prior_stage_receipts(stage_receipts, manifest=contract)
    except OwnDeckSuccessorError as exc:
        return OwnDeckSuccessorGateDecision(
            op,
            False,
            str(exc),
            contract.identity.sha256,
        )

    accepted = tuple(stages)
    if op == OwnDeckSuccessorOperation.ISOLATED_MIGRATION:
        return OwnDeckSuccessorGateDecision(
            op,
            True,
            "terminal refresh and every prior r258 stage receipt are valid; isolated migration may be staged",
            contract.identity.sha256,
            refresh.sha256,
            accepted,
        )

    supplied_post = post_refresh_receipts or {}
    try:
        _validate_required_post_refresh_chain(
            operation=op,
            receipts=supplied_post,
            manifest=contract,
            refresh_completion=refresh,
            stage_receipts=stages,
        )
    except OwnDeckSuccessorError as exc:
        return OwnDeckSuccessorGateDecision(
            op,
            False,
            str(exc),
            contract.identity.sha256,
            refresh.sha256,
            accepted,
        )

    if op == OwnDeckSuccessorOperation.TRAINING_CANARY:
        reason = "validated isolated migration receipt permits the receipt-bound training canary"
    elif op == OwnDeckSuccessorOperation.SOURCE_DISJOINT_EVALUATION:
        reason = "validated migration and canary receipts permit source-disjoint evaluation"
    elif op == OwnDeckSuccessorOperation.RUNTIME_ACTIVATION:
        reason = "validated migration, canary, evaluation, and distinct activation receipts permit runtime activation"
    elif op == OwnDeckSuccessorOperation.PROMOTION:
        reason = "validated migration, canary, evaluation, activation, and distinct promotion receipts permit promotion"
    else:  # Defensive: all operations are classified above.
        return OwnDeckSuccessorGateDecision(
            op,
            False,
            "unknown r258 operation cannot be authorized",
            contract.identity.sha256,
            refresh.sha256,
            accepted,
        )
    return OwnDeckSuccessorGateDecision(
        op,
        True,
        reason,
        contract.identity.sha256,
        refresh.sha256,
        accepted,
    )


def require_successor_operation(
    operation: OwnDeckSuccessorOperation | str,
    **kwargs: Any,
) -> OwnDeckSuccessorGateDecision:
    """Raise a typed error instead of returning a denied gate decision."""

    decision = evaluate_successor_gate(operation, **kwargs)
    if not decision.allowed:
        raise OwnDeckSuccessorGateError(decision.reason)
    return decision


# Short aliases make this narrow module straightforward to call from an
# eventually-created isolated successor workflow without importing a runtime.
evaluate_gate = evaluate_successor_gate
require_gate = require_successor_operation
validate_manifest = validate_canonical_manifest


def _validate_required_post_refresh_chain(
    *,
    operation: OwnDeckSuccessorOperation,
    receipts: Mapping[
        OwnDeckSuccessorPostRefreshReceiptKind | str, Mapping[str, Any] | Path | str
    ],
    manifest: OwnDeckSuccessorManifest,
    refresh_completion: RefreshCompletionReceipt,
    stage_receipts: Mapping[OwnDeckSuccessorStage, StageReceipt],
) -> dict[OwnDeckSuccessorPostRefreshReceiptKind, PostRefreshReceipt]:
    normalized: dict[
        OwnDeckSuccessorPostRefreshReceiptKind, Mapping[str, Any] | Path | str
    ] = {}
    for raw_kind, raw_receipt in receipts.items():
        kind = _coerce_post_refresh_kind(raw_kind, label="post-refresh receipt mapping key")
        if kind in normalized:
            raise OwnDeckSuccessorError(f"duplicate post-refresh receipt: {kind.value}")
        normalized[kind] = raw_receipt

    required_by_operation = {
        OwnDeckSuccessorOperation.TRAINING_CANARY: (
            OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
        ),
        OwnDeckSuccessorOperation.SOURCE_DISJOINT_EVALUATION: (
            OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
            OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY,
        ),
        OwnDeckSuccessorOperation.RUNTIME_ACTIVATION: (
            OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
            OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY,
            OwnDeckSuccessorPostRefreshReceiptKind.SOURCE_DISJOINT_EVALUATION,
            OwnDeckSuccessorPostRefreshReceiptKind.RUNTIME_ACTIVATION,
        ),
        OwnDeckSuccessorOperation.PROMOTION: (
            OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
            OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY,
            OwnDeckSuccessorPostRefreshReceiptKind.SOURCE_DISJOINT_EVALUATION,
            OwnDeckSuccessorPostRefreshReceiptKind.RUNTIME_ACTIVATION,
            OwnDeckSuccessorPostRefreshReceiptKind.PROMOTION,
        ),
    }
    required = required_by_operation.get(operation)
    if required is None:
        raise OwnDeckSuccessorError(f"{operation.value} has no post-refresh receipt chain")
    missing = [kind.value for kind in required if kind not in normalized]
    if missing:
        raise OwnDeckSuccessorError(
            "missing distinct post-refresh receipt(s): " + ", ".join(missing)
        )

    validated: dict[OwnDeckSuccessorPostRefreshReceiptKind, PostRefreshReceipt] = {}
    for kind in required:
        dependencies = _post_refresh_dependencies(kind, validated)
        validated[kind] = validate_post_refresh_receipt(
            normalized[kind],
            kind=kind,
            manifest=manifest,
            refresh_completion=refresh_completion,
            stage_receipts=stage_receipts,
            required_dependencies=dependencies,
        )
    return validated


def _post_refresh_dependencies(
    kind: OwnDeckSuccessorPostRefreshReceiptKind,
    validated: Mapping[OwnDeckSuccessorPostRefreshReceiptKind, PostRefreshReceipt],
) -> dict[OwnDeckSuccessorPostRefreshReceiptKind, PostRefreshReceipt]:
    ordered = (
        OwnDeckSuccessorPostRefreshReceiptKind.ISOLATED_MIGRATION,
        OwnDeckSuccessorPostRefreshReceiptKind.TRAINING_CANARY,
        OwnDeckSuccessorPostRefreshReceiptKind.SOURCE_DISJOINT_EVALUATION,
        OwnDeckSuccessorPostRefreshReceiptKind.RUNTIME_ACTIVATION,
        OwnDeckSuccessorPostRefreshReceiptKind.PROMOTION,
    )
    position = ordered.index(kind)
    return {prior: validated[prior] for prior in ordered[:position] if prior in validated}


def _denied_missing_refresh(
    operation: OwnDeckSuccessorOperation,
    manifest: OwnDeckSuccessorManifest,
) -> OwnDeckSuccessorGateDecision:
    return OwnDeckSuccessorGateDecision(
        operation,
        False,
        "r258 fails closed: a valid terminal Alakazam-refresh completion receipt is required",
        manifest.identity.sha256,
    )


def _validate_manifest_stages(value: object) -> tuple[OwnDeckSuccessorStage, ...]:
    if not isinstance(value, list):
        raise OwnDeckSuccessorError("manifest stages must be an ordered list")
    stages: list[OwnDeckSuccessorStage] = []
    for index, raw_stage in enumerate(value):
        stage_payload = _mapping(raw_stage, label=f"manifest stage {index}")
        stage = _coerce_stage(stage_payload.get("id"), label=f"manifest stage {index} id")
        expected_status = _EXPECTED_STAGE_STATUS[stage].value
        _require_exact(
            stage_payload.get("status"), expected_status, label=f"manifest stage {stage.value} status"
        )
        required_fields = (
            ("scope", "deliverables", "acceptance")
            if stage in _PRE_REFRESH_STAGES
            else ("scope", "entry_requirements", "acceptance")
        )
        for field in required_fields:
            candidate = stage_payload.get(field)
            if field == "scope":
                if not isinstance(candidate, str) or not candidate.strip():
                    raise OwnDeckSuccessorError(f"manifest stage {stage.value} lacks scope")
            elif not isinstance(candidate, list) or not candidate:
                raise OwnDeckSuccessorError(f"manifest stage {stage.value} lacks {field}")
        stages.append(stage)
    if tuple(stages) != _STAGE_ORDER:
        raise OwnDeckSuccessorError("manifest stages are not the canonical r258 order")
    return tuple(stages)


def _validate_receipt_plan(value: object) -> None:
    receipt_plan = _mapping(value, label="manifest receipt plan")
    _require_exact(
        receipt_plan.get("stage_receipt_schema"), STAGE_RECEIPT_SCHEMA, label="stage receipt schema"
    )
    required = _string_set(
        receipt_plan.get("required_fields"), label="stage receipt required fields"
    )
    required_fields = {
        "stage_id",
        "status",
        "source_sha256s",
        "test_command_or_fixture_identity",
        "test_result",
        "public_information_audit",
        "direct_policy_audit",
        "r241_nonmutation_audit",
    }
    if required_fields - required:
        raise OwnDeckSuccessorError("manifest receipt plan omits a required stage field")
    _require_exact(
        receipt_plan.get("activation_receipt_is_distinct_from_stage_receipts"),
        True,
        label="activation receipt separation",
    )
    _require_exact(
        receipt_plan.get("promotion_receipt_is_distinct_from_activation_receipt"),
        True,
        label="promotion receipt separation",
    )


def _load_receipt(
    receipt: Mapping[str, Any] | Path | str,
    *,
    label: str,
) -> ReceiptIdentity:
    if isinstance(receipt, (str, Path)):
        path = _regular_file(receipt, label=label)
        payload = _read_json_object(path, label=label)
        # Receipt identity is deliberately the detached self-fingerprint, not
        # the transport file hash.  That makes evidence chains insensitive to
        # harmless JSON whitespace while still binding every semantic field.
        return ReceiptIdentity(
            sha256=receipt_digest(payload), payload=payload, path=path
        )
    payload = _mapping(receipt, label=label)
    # Round-trip to avoid retaining a mutable caller-owned nested mapping.
    immutable_payload = json.loads(canonical_json(payload).decode("utf-8"))
    if not isinstance(immutable_payload, dict):  # Defensive: payload is a mapping.
        raise OwnDeckSuccessorError(f"{label} must be a JSON object")
    return ReceiptIdentity(
        sha256=receipt_digest(immutable_payload),
        payload=immutable_payload,
        path=None,
    )


def _validate_receipt_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    declared = _sha256_text(payload.get("receipt_sha256"), label=f"{label} fingerprint")
    computed = receipt_digest(payload)
    if declared != computed:
        raise OwnDeckSuccessorError(
            f"{label} fingerprint does not match its canonical contents"
        )


def _identity_object(value: object, *, label: str) -> Mapping[str, Any]:
    identity = _mapping(value, label=label)
    raw_id = identity.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise OwnDeckSuccessorError(f"{label} identity is absent")
    _sha256_text(identity.get("sha256"), label=f"{label} digest")
    return identity


def _chain_integrity(value: object, *, label: str) -> None:
    chain = _mapping(value, label=label)
    _require_exact(chain.get("integrity_verified"), True, label=f"{label} integrity")
    _sha256_text(chain.get("sha256"), label=f"{label} digest")


def _digest_mapping(
    value: object,
    *,
    label: str,
    permit_empty: bool = False,
) -> dict[str, str]:
    mapping = _mapping(value, label=label)
    if not mapping and not permit_empty:
        raise OwnDeckSuccessorError(f"{label} must not be empty")
    normalized: dict[str, str] = {}
    for key, digest in mapping.items():
        if not isinstance(key, str) or not key:
            raise OwnDeckSuccessorError(f"{label} has an invalid key")
        normalized[key] = _sha256_text(digest, label=f"{label} {key}")
    return normalized


def _coerce_stage(value: object, *, label: str) -> OwnDeckSuccessorStage:
    try:
        return value if isinstance(value, OwnDeckSuccessorStage) else OwnDeckSuccessorStage(str(value))
    except ValueError as exc:
        raise OwnDeckSuccessorError(f"{label} is not an r258 stage") from exc


def _coerce_post_refresh_kind(
    value: object,
    *,
    label: str,
) -> OwnDeckSuccessorPostRefreshReceiptKind:
    try:
        return (
            value
            if isinstance(value, OwnDeckSuccessorPostRefreshReceiptKind)
            else OwnDeckSuccessorPostRefreshReceiptKind(str(value))
        )
    except ValueError as exc:
        raise OwnDeckSuccessorError(f"{label} is not an r258 post-refresh receipt kind") from exc


def _coerce_operation(value: OwnDeckSuccessorOperation | str) -> OwnDeckSuccessorOperation:
    if isinstance(value, OwnDeckSuccessorOperation):
        return value
    normalized = str(value).strip().lower()
    aliased = _OPERATION_ALIASES.get(normalized)
    if aliased is not None:
        return aliased
    try:
        return OwnDeckSuccessorOperation(normalized)
    except ValueError as exc:
        raise OwnDeckSuccessorGateError(f"unknown r258 successor operation: {value}") from exc


def _require_canonical_contract(
    contract: OwnDeckSuccessorManifest,
    *,
    canonical: OwnDeckSuccessorManifest,
) -> None:
    """Reject a caller-supplied projection that differs from the loaded source."""

    if contract.identity.path != CANONICAL_MANIFEST_PATH.resolve():
        raise OwnDeckSuccessorError("successor gate only accepts the canonical r258 manifest")
    if contract.identity != canonical.identity:
        raise OwnDeckSuccessorError("canonical r258 manifest changed after validation")
    if (
        contract.stages != canonical.stages
        or contract.elmo_side_store != canonical.elmo_side_store
    ):
        raise OwnDeckSuccessorError(
            "caller-supplied successor contract does not match the canonical r258 manifest"
        )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, OwnDeckSuccessorError) as exc:
        raise OwnDeckSuccessorError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise OwnDeckSuccessorError(f"{label} must be a JSON object")
    return payload


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Reject same-valued duplicate keys before contract validation sees them."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OwnDeckSuccessorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise OwnDeckSuccessorError(f"{label} must be a regular non-symlink file")
    return candidate.resolve()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnDeckSuccessorError(f"{label} must be an object")
    return value


def _string_set(value: object, *, label: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise OwnDeckSuccessorError(f"{label} must be a non-empty string list")
    return set(value)


def _sha256_text(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 71 or not text.startswith(_SHA256_PREFIX):
        raise OwnDeckSuccessorError(f"{label} is not a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in text[len(_SHA256_PREFIX) :]):
        raise OwnDeckSuccessorError(f"{label} is not a SHA-256 digest")
    return text


def _require_exact(actual: object, expected: object, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise OwnDeckSuccessorError(f"{label} is not the required value")
