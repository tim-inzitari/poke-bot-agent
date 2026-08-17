"""Receipt-only r303 plans for the future r274 → Blackwell handoff.

This module is deliberately *not* a service controller, trainer, package
builder, queue writer, or fleet worker.  It turns the revision-5 contract's
strict receipt inventories into create-only, non-activating handoff plans and
validators.  In particular it cannot pause r274, load the staged corpus,
touch a checkpoint, call systemd, queue a Kaggle submission, or launch a
Blackwell process.

The purpose is to make the eventual clean boundary mechanically explicit:

* every prerequisite receipt is re-opened and checksum-bound;
* old r274 lineage identities are preserved for rollback, never substituted;
* the only permitted future learner is Inzi ``cuda:1`` after a separately
  observed activation receipt; and
* package/queue/fleet consumers can reject incomplete or duplicate-credit
  evidence before any runtime authority is considered.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


R303_GOAL_REVISION: Final = 5
R303_ROOT_OWNER_REVISION: Final = 303
R303_CONTRACT_PATH: Final = "goals/alakazam-elmo-rule-derivative/contract.json"
R303_GOAL_PATH: Final = "goals/alakazam-elmo-rule-derivative/GOAL.md"
R303_CONTRACT_SHA256: Final = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
# Revision-5 consumers must bind the revision-5 goal gateway and exact typed
# contract as a pair.  Revision-4 evidence can be consumed only after explicit
# classification as immutable predecessor evidence; it cannot be substituted
# into a revision-5 handoff plan.
R303_GOAL_GATEWAY_SHA256: Final = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)

R303_HANDOFF_PREFLIGHT_PLAN_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_handoff_preflight_plan/v1"
)
R303_BLACKWELL_LAUNCH_CONTRACT_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_blackwell_launch_contract/v1"
)
R303_FLEET_PREFLIGHT_PLAN_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_fleet_preflight_plan/v1"
)
R303_ROLLBACK_TEMPLATE_SCHEMA: Final = (
    "poke_bot.alakazam_rule_derivative_r303_rollback_template/v1"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_FILE: Final = PROJECT_ROOT / R303_CONTRACT_PATH
GOAL_FILE: Final = PROJECT_ROOT / R303_GOAL_PATH
_SHA256_PREFIX: Final = "sha256:"


class R303HandoffError(RuntimeError):
    """A proposed handoff has insufficient or inconsistent immutable proof."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return _SHA256_PREFIX + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R303HandoffError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise R303HandoffError(f"{field} must be a list")
    return list(value)


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        raise R303HandoffError(f"{field} must be a sha256 identity")
    raw = value.removeprefix(_SHA256_PREFIX)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise R303HandoffError(f"{field} must use lowercase SHA-256 hex")
    return value


def _sha_list(value: Any, *, field: str, nonempty: bool = True) -> list[str]:
    rows = _rows(value, field=field)
    if (nonempty and not rows) or any(not isinstance(item, str) for item in rows):
        raise R303HandoffError(f"{field} must be a {'nonempty ' if nonempty else ''}list of SHA-256 identities")
    return [_sha(item, field=f"{field}[{index}]") for index, item in enumerate(rows)]


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise R303HandoffError(f"{field} must be an exact {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_regular_json(path: Path | str, *, field: str) -> tuple[Path, Mapping[str, Any], str]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise R303HandoffError(f"{field} must be a regular non-symlink file")
    before = target.stat()
    digest = sha256_file(target)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R303HandoffError(f"cannot read {field}") from exc
    after = target.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sha256_file(target) != digest:
        raise R303HandoffError(f"{field} changed while being inspected")
    return target.resolve(), _mapping(payload, field=field), digest


def load_r303_contract() -> Mapping[str, Any]:
    """Open the exact rev5 typed contract; no caller can substitute one."""

    path, contract, digest = _read_regular_json(CONTRACT_FILE, field="r303 canonical contract")
    if path != CONTRACT_FILE.resolve() or digest != R303_CONTRACT_SHA256:
        raise R303HandoffError("r303 contract file identity is stale or foreign")
    if (
        contract.get("goal_id"),
        contract.get("goal_revision"),
        contract.get("root_handoff_revision"),
    ) != ("alakazam-elmo-rule-derivative", R303_GOAL_REVISION, R303_ROOT_OWNER_REVISION):
        raise R303HandoffError("r303 contract goal/revision identity drifted")
    authority = _mapping(contract.get("authority"), field="r303 authority")
    if authority.get("immediate_runtime_service_preemption_training_submission_or_self_play_authority") is not False:
        raise R303HandoffError("r303 contract unexpectedly grants immediate runtime authority")
    handoff = _mapping(contract.get("inzi_blackwell_training_handoff"), field="r303 handoff")
    if handoff.get("status") != "authorized_for_future_clean_boundary_no_runtime_change_now":
        raise R303HandoffError("r303 contract handoff status is not non-immediate")
    if GOAL_FILE.is_symlink() or not GOAL_FILE.is_file() or sha256_file(GOAL_FILE) != R303_GOAL_GATEWAY_SHA256:
        raise R303HandoffError("r303 goal gateway identity changed without a new contract binding")
    return contract


def _contract_handoff(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(contract.get("inzi_blackwell_training_handoff"), field="contract handoff")


def _handoff_receipt_spec(contract: Mapping[str, Any], kind: str) -> tuple[str, tuple[str, ...]]:
    receipt_contract = _mapping(
        _contract_handoff(contract).get("receipt_contract"), field="handoff receipt contract"
    )
    raw = _mapping(receipt_contract.get(kind), field=f"handoff receipt contract {kind}")
    schema = raw.get("schema")
    if not isinstance(schema, str) or not schema:
        raise R303HandoffError(f"handoff receipt contract {kind} lacks schema")
    fields = tuple(_rows(raw.get("required_fields"), field=f"handoff receipt {kind} fields"))
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise R303HandoffError(f"handoff receipt contract {kind} fields are malformed")
    if len(set(fields)) != len(fields):
        raise R303HandoffError(f"handoff receipt contract {kind} repeats fields")
    return schema, fields


def _assert_exact_receipt_inventory(
    receipt: Mapping[str, Any], *, schema: str, fields: Sequence[str], label: str
) -> Mapping[str, Any]:
    row = _mapping(receipt, field=label)
    actual = set(row)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise R303HandoffError(
            f"{label} top-level inventory mismatch; missing={missing!r} extra={extra!r}"
        )
    if row.get("schema") != schema:
        raise R303HandoffError(f"{label} schema mismatch")
    if row.get("goal_contract_path") != R303_CONTRACT_PATH or row.get("goal_contract_sha256") != R303_CONTRACT_SHA256:
        raise R303HandoffError(f"{label} binds a stale contract")
    if row.get("goal_revision") != R303_GOAL_REVISION:
        raise R303HandoffError(f"{label} goal revision mismatch")
    # Every checksum-shaped field in a contract receipt must actually be a
    # lowercase SHA identity.  This prevents a plan from treating arbitrary
    # prose as a cryptographic binding.  ``*_sha256s`` is intentionally
    # handled before ``*_sha256`` so list-valued contract bindings receive the
    # same strict treatment.
    for key, value in row.items():
        if key.endswith("_sha256s"):
            _sha_list(value, field=f"{label}.{key}")
        elif key.endswith("_sha256"):
            _sha(value, field=f"{label}.{key}")
    return row


def _assert_bool(row: Mapping[str, Any], field: str, expected: bool, *, label: str) -> None:
    if row.get(field) is not expected:
        raise R303HandoffError(f"{label}.{field} must be {expected}")


def _validate_parent_receipt(row: Mapping[str, Any]) -> None:
    _assert_bool(row, "safety_approved", True, label="parent receipt")
    _assert_bool(row, "immutable", True, label="parent receipt")
    if row.get("old_lineage_id") != "alakazam-new-list-direct-policy-r274":
        raise R303HandoffError("parent receipt names an unexpected old lineage")
    _positive_int(row.get("old_checkpoint_size_bytes"), field="parent checkpoint size")
    _positive_int(row.get("old_optimizer_state_size_bytes"), field="parent optimizer size")


def _validate_corpus_receipt(row: Mapping[str, Any]) -> None:
    if (row.get("window_start_utc"), row.get("window_end_utc"), row.get("utc_partition_count")) != (
        "2026-07-13", "2026-08-11", 30
    ):
        raise R303HandoffError("corpus receipt does not bind the exact 30-day window")
    _assert_bool(row, "schema_validation_passed", True, label="corpus receipt")
    _assert_bool(row, "source_remote_parity_passed", True, label="corpus receipt")
    _assert_bool(row, "hidden_inputs_absent", True, label="corpus receipt")
    _assert_bool(row, "training_eligible_before_activation", False, label="corpus receipt")
    _positive_int(row.get("object_count"), field="corpus object count")
    _positive_int(row.get("total_games"), field="corpus total games", allow_zero=True)
    _positive_int(row.get("total_decisions"), field="corpus total decisions", allow_zero=True)


def _validate_schema_freeze_receipt(row: Mapping[str, Any]) -> None:
    _assert_bool(row, "q3_bench_only", True, label="schema-freeze receipt")
    _assert_bool(row, "q5_q6_trace_only_zero", True, label="schema-freeze receipt")
    _assert_bool(row, "public_information_contract_passed", True, label="schema-freeze receipt")
    _assert_bool(row, "layer_off_bit_identical_baseline_logits", True, label="schema-freeze receipt")
    branches = _rows(row.get("new_branch_inventory"), field="schema-freeze branch inventory")
    if not branches or any(not isinstance(item, str) or not item for item in branches):
        raise R303HandoffError("schema-freeze receipt branch inventory is malformed")


def _validate_frozen_tensor_receipt(row: Mapping[str, Any]) -> None:
    _assert_bool(row, "card2vec_parameter_and_buffer_bit_identity", True, label="frozen-tensor receipt")
    _assert_bool(
        row,
        "backbone_existing_heads_fusion_own_deck_and_matchup_adapter_bit_identity",
        True,
        label="frozen-tensor receipt",
    )
    _assert_bool(row, "frozen_parameters_absent_from_optimizer", True, label="frozen-tensor receipt")
    _assert_bool(row, "new_path_zero_initialized", True, label="frozen-tensor receipt")
    _assert_bool(row, "layer_off_bit_identical_baseline_logits", True, label="frozen-tensor receipt")
    frozen = _rows(row.get("frozen_module_names"), field="frozen module names")
    required = {
        "neural_backbone",
        "existing_heads",
        "fusion",
        "own_deck_routes",
        "matchup_adapters",
        "card2vec",
    }
    if not required <= set(frozen):
        raise R303HandoffError("frozen-tensor receipt omits a required frozen module")
    trainable = _rows(row.get("trainable_parameter_names"), field="trainable parameter names")
    optimizer = _rows(row.get("optimizer_parameter_names"), field="optimizer parameter names")
    if set(optimizer) != set(trainable):
        raise R303HandoffError("optimizer parameter set differs from explicit trainable set")


def _validate_blackwell_preflight_receipt(row: Mapping[str, Any]) -> None:
    if (
        row.get("host"),
        row.get("device"),
        row.get("device_model"),
    ) != ("inzi", "cuda:1", "NVIDIA RTX PRO 5000 Blackwell"):
        raise R303HandoffError("Blackwell preflight does not bind the authorized device")
    for field in (
        "load_canary_passed",
        "forward_backward_optimizer_canary_passed",
        "finite_loss_and_gradients",
        "frozen_tensor_gradient_norm_exact_zero",
        "frozen_parameters_absent_from_optimizer",
        "layer_off_parity_passed",
        "resource_headroom_passed",
        "no_runtime_or_service_change_performed",
    ):
        _assert_bool(row, field, True, label="Blackwell preflight receipt")
    _positive_int(row.get("device_total_memory_bytes"), field="Blackwell total device memory")
    _positive_int(row.get("device_free_memory_bytes_before"), field="Blackwell free device memory")
    _positive_int(row.get("peak_device_memory_bytes"), field="Blackwell peak device memory", allow_zero=True)
    _positive_int(row.get("peak_host_ram_bytes"), field="Blackwell peak host RAM", allow_zero=True)


def _validate_rollback_plan_receipt(row: Mapping[str, Any]) -> None:
    if (
        row.get("old_lineage_id"),
        row.get("old_trainer_service"),
        row.get("old_submission_boundary_service"),
        row.get("new_lineage_id"),
    ) != (
        "alakazam-new-list-direct-policy-r274",
        "pokebot-alakazam-r274-rl.service",
        "pokebot-alakazam-r274-rl-submission-boundaries.service",
        "alakazam-rule-derivative-g5",
    ):
        raise R303HandoffError("rollback plan has foreign lineage/service identities")
    for field in (
        "shared_kaggle_queue_service_preserved",
        "clean_boundary_required",
        "new_artifacts_preserved_not_deleted",
        "staged_corpus_requarantined_on_rollback",
        "rollback_dry_run_passed",
    ):
        _assert_bool(row, field, True, label="rollback plan receipt")


_EXACT_R274_PAUSE_SERVICES: Final = (
    "pokebot-alakazam-r274-rl.service",
    "pokebot-alakazam-r274-rl-submission-boundaries.service",
)
_SHARED_KAGGLE_QUEUE_SERVICE: Final = "pokebot-kaggle-submission-queue.service"
_R303_DERIVATIVE_LINEAGE: Final = "alakazam-rule-derivative-g5"
_R303_BOOTSTRAP_SERVICE: Final = "pokebot-alakazam-rule-derivative-g5-bootstrap.service"
_R303_TRAINER_SERVICE: Final = "pokebot-alakazam-rule-derivative-g5-rl.service"


def _validate_preemption_readiness_receipt(row: Mapping[str, Any]) -> None:
    if row.get("root_owner_revision") != R303_ROOT_OWNER_REVISION:
        raise R303HandoffError("preemption readiness has an unexpected root handoff revision")
    for field in (
        "current_r274_checkpoint_and_optimizer_sealed",
        "current_r274_collection_and_shard_sealed",
        "current_r274_adapter_and_commit_sealed",
        "no_r274_work_unit_in_flight",
        "shared_kaggle_queue_service_excluded_from_pause",
        "staged_shards_still_training_ineligible",
        "readiness_passed",
    ):
        _assert_bool(row, field, True, label="preemption readiness receipt")
    pauses = _rows(row.get("planned_pause_service_names"), field="preemption readiness pause services")
    if pauses != list(_EXACT_R274_PAUSE_SERVICES):
        raise R303HandoffError("preemption readiness has an unsafe service pause set")


def _validate_handoff_activation_receipt(row: Mapping[str, Any]) -> None:
    if row.get("root_owner_revision") != R303_ROOT_OWNER_REVISION:
        raise R303HandoffError("handoff activation has an unexpected root handoff revision")
    if (
        row.get("old_lineage_id"),
        row.get("old_trainer_service"),
        row.get("old_submission_boundary_service"),
        row.get("new_lineage_id"),
        row.get("new_managed_bootstrap_service"),
    ) != (
        "alakazam-new-list-direct-policy-r274",
        _EXACT_R274_PAUSE_SERVICES[0],
        _EXACT_R274_PAUSE_SERVICES[1],
        _R303_DERIVATIVE_LINEAGE,
        _R303_BOOTSTRAP_SERVICE,
    ):
        raise R303HandoffError("handoff activation has foreign lineage/service identities")
    if row.get("shared_kaggle_queue_service") != _SHARED_KAGGLE_QUEUE_SERVICE:
        raise R303HandoffError("handoff activation does not preserve the shared Kaggle queue service")
    for field in (
        "old_services_paused_via_systemd_user",
        "old_services_inactive_verified",
        "shared_kaggle_queue_service_unchanged",
        "staged_shards_training_eligible",
        "no_concurrent_r274_training_or_collection",
        "no_serving_selector_or_submission_activation",
    ):
        _assert_bool(row, field, True, label="handoff activation receipt")
    boundary = row.get("activation_boundary_id")
    if not isinstance(boundary, str) or not boundary:
        raise R303HandoffError("handoff activation lacks a durable activation boundary id")


_HANDOFF_VALIDATORS: Final = {
    "parent": _validate_parent_receipt,
    "corpus": _validate_corpus_receipt,
    "schema_freeze": _validate_schema_freeze_receipt,
    "frozen_tensors": _validate_frozen_tensor_receipt,
    "blackwell_preflight": _validate_blackwell_preflight_receipt,
    "rollback_plan": _validate_rollback_plan_receipt,
    "preemption_readiness": _validate_preemption_readiness_receipt,
    "activation": _validate_handoff_activation_receipt,
}

_HANDOFF_PREREQUISITE_KINDS: Final = (
    "parent",
    "corpus",
    "schema_freeze",
    "frozen_tensors",
    "blackwell_preflight",
    "rollback_plan",
)


def validate_handoff_receipt(kind: str, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one exact rev5 handoff receipt shape and safety invariants."""

    contract = load_r303_contract()
    schema, fields = _handoff_receipt_spec(contract, kind)
    row = _assert_exact_receipt_inventory(
        receipt, schema=schema, fields=fields, label=f"handoff {kind} receipt"
    )
    validator = _HANDOFF_VALIDATORS.get(kind)
    if validator is not None:
        validator(row)
    return row


@dataclass(frozen=True)
class ReceiptIdentity:
    """Re-opened file identity for a non-self-referential contract receipt."""

    kind: str
    path: str
    sha256: str
    size_bytes: int
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "schema": self.schema,
        }


def inspect_handoff_receipt(
    kind: str,
    path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity]:
    """Re-open an exact handoff receipt and optionally pin it to a root."""

    source, receipt, digest = _read_regular_json(path, field=f"handoff {kind} receipt")
    if trusted_receipt_root is not None:
        root = Path(trusted_receipt_root)
        if root.is_symlink() or not root.is_dir() or not _path_within(source, root.resolve()):
            raise R303HandoffError("handoff receipt lies outside its trusted receipt root")
    checked = validate_handoff_receipt(kind, receipt)
    return checked, ReceiptIdentity(
        kind=kind,
        path=str(source),
        sha256=digest,
        size_bytes=source.stat().st_size,
        schema=str(checked["schema"]),
    )


def inspect_handoff_bundle(
    paths: Mapping[str, Path | str],
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, ReceiptIdentity]]:
    """Validate all six immutable prerequisites for a future clean boundary."""

    required = _HANDOFF_PREREQUISITE_KINDS
    if set(paths) != set(required):
        raise R303HandoffError(
            "handoff bundle must contain exactly: " + ", ".join(required)
        )
    receipts: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, ReceiptIdentity] = {}
    for kind in required:
        receipt, identity = inspect_handoff_receipt(
            kind, paths[kind], trusted_receipt_root=trusted_receipt_root
        )
        receipts[kind] = receipt
        identities[kind] = identity
    parent = receipts["parent"]
    corpus = receipts["corpus"]
    schema = receipts["schema_freeze"]
    frozen = receipts["frozen_tensors"]
    blackwell = receipts["blackwell_preflight"]
    rollback = receipts["rollback_plan"]
    if frozen.get("parent_checkpoint_sha256") != parent.get("old_checkpoint_sha256"):
        raise R303HandoffError("frozen-tensor receipt parent does not equal immutable r274 parent")
    if (
        rollback.get("old_lineage_id") != parent.get("old_lineage_id")
        or rollback.get("old_run_root") != parent.get("old_run_root")
        or rollback.get("old_runtime_registry_path") != parent.get("old_runtime_registry_path")
        or rollback.get("old_runtime_registry_sha256") != parent.get("old_runtime_registry_sha256")
        or rollback.get("old_checkpoint_path") != parent.get("old_checkpoint_path")
        or rollback.get("old_checkpoint_sha256") != parent.get("old_checkpoint_sha256")
        or rollback.get("old_optimizer_state_path") != parent.get("old_optimizer_state_path")
        or rollback.get("old_optimizer_state_sha256") != parent.get("old_optimizer_state_sha256")
        or rollback.get("old_collection_manifest_path") != parent.get("old_collection_manifest_path")
        or rollback.get("old_collection_manifest_sha256") != parent.get("old_collection_manifest_sha256")
    ):
        raise R303HandoffError("rollback plan is not cross-linked to the immutable r274 parent")
    service_definition_hashes = _sha_list(
        rollback.get("old_service_definition_sha256s"),
        field="rollback plan old service definition hashes",
    )
    if {
        parent.get("old_trainer_service_definition_sha256"),
        parent.get("old_submission_boundary_service_definition_sha256"),
    } != set(service_definition_hashes):
        raise R303HandoffError("rollback plan omits or substitutes an r274 service definition")
    if rollback.get("shared_kaggle_queue_service") != _SHARED_KAGGLE_QUEUE_SERVICE:
        raise R303HandoffError("rollback plan does not preserve the shared Kaggle queue service")
    if schema.get("canonical_simulator_sha256") != parent.get("canonical_simulator_sha256"):
        raise R303HandoffError("schema freeze and r274 parent disagree about canonical simulator")
    if corpus.get("staging_root") != _contract_handoff(load_r303_contract())["trusted_roots"]["staged_corpus_root"]:
        raise R303HandoffError("corpus receipt uses a foreign staged-corpus root")
    if blackwell.get("planned_batch_and_optimizer_config_sha256") is None:
        raise R303HandoffError("Blackwell preflight lacks the planned optimizer identity")
    return receipts, identities


def inspect_preemption_readiness_bundle(
    prerequisite_paths: Mapping[str, Path | str],
    readiness_path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, ReceiptIdentity], Mapping[str, Any], ReceiptIdentity]:
    """Validate a readiness receipt against all six immutable prerequisites.

    This is read-only inspection.  It does not pause a service or make staged
    shards usable; its purpose is to reject a readiness document that merely
    names plausible hashes without re-opening the evidence it depends on.
    """

    receipts, identities = inspect_handoff_bundle(
        prerequisite_paths, trusted_receipt_root=trusted_receipt_root
    )
    readiness, readiness_identity = inspect_handoff_receipt(
        "preemption_readiness", readiness_path, trusted_receipt_root=trusted_receipt_root
    )
    for receipt_kind, field in (
        ("parent", "parent_receipt_sha256"),
        ("corpus", "corpus_receipt_sha256"),
        ("schema_freeze", "schema_freeze_receipt_sha256"),
        ("frozen_tensors", "frozen_tensor_receipt_sha256"),
        ("blackwell_preflight", "blackwell_preflight_receipt_sha256"),
        ("rollback_plan", "rollback_plan_receipt_sha256"),
    ):
        if readiness.get(field) != identities[receipt_kind].sha256:
            raise R303HandoffError(f"preemption readiness does not bind {receipt_kind} receipt bytes")
    return receipts, identities, readiness, readiness_identity


def inspect_handoff_activation_bundle(
    prerequisite_paths: Mapping[str, Path | str],
    readiness_path: Path | str,
    activation_path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, ReceiptIdentity], Mapping[str, Any], ReceiptIdentity, Mapping[str, Any], ReceiptIdentity]:
    """Read and cross-link an observed activation receipt without activating."""

    receipts, identities, readiness, readiness_identity = inspect_preemption_readiness_bundle(
        prerequisite_paths,
        readiness_path,
        trusted_receipt_root=trusted_receipt_root,
    )
    activation, activation_identity = inspect_handoff_receipt(
        "activation", activation_path, trusted_receipt_root=trusted_receipt_root
    )
    parent = receipts["parent"]
    corpus = receipts["corpus"]
    if activation.get("readiness_receipt_sha256") != readiness_identity.sha256:
        raise R303HandoffError("handoff activation does not bind readiness receipt bytes")
    for activation_field, parent_field in (
        ("old_parent_checkpoint_sha256", "old_checkpoint_sha256"),
        ("old_optimizer_state_sha256", "old_optimizer_state_sha256"),
        ("old_collection_manifest_sha256", "old_collection_manifest_sha256"),
    ):
        if activation.get(activation_field) != parent.get(parent_field):
            raise R303HandoffError("handoff activation substitutes immutable r274 lineage evidence")
    if activation.get("staged_corpus_receipt_sha256") != identities["corpus"].sha256:
        raise R303HandoffError("handoff activation does not bind the sealed corpus receipt")
    if activation.get("blackwell_preflight_receipt_sha256") != identities["blackwell_preflight"].sha256:
        raise R303HandoffError("handoff activation does not bind Blackwell preflight bytes")
    if activation.get("rollback_plan_receipt_sha256") != identities["rollback_plan"].sha256:
        raise R303HandoffError("handoff activation does not bind rollback plan bytes")
    handoff = _contract_handoff(load_r303_contract())
    roots = _mapping(handoff.get("trusted_roots"), field="handoff trusted roots")
    if (
        activation.get("new_run_root") != roots.get("run_root")
        or activation.get("new_runtime_registry_path") != roots.get("runtime_registry")
    ):
        raise R303HandoffError("handoff activation uses a foreign derivative run/registry path")
    return receipts, identities, readiness, readiness_identity, activation, activation_identity


def _plan_checksum(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("plan_payload_sha256", None)
    return canonical_sha256(payload)


def build_handoff_preflight_plan(
    receipt_paths: Mapping[str, Path | str],
    *,
    trusted_receipt_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a non-activating clean-boundary handoff plan from six receipts."""

    contract = load_r303_contract()
    receipts, identities = inspect_handoff_bundle(
        receipt_paths, trusted_receipt_root=trusted_receipt_root
    )
    handoff = _contract_handoff(contract)
    old = _mapping(handoff.get("old_lineage"), field="old lineage")
    roots = _mapping(handoff.get("trusted_roots"), field="handoff trusted roots")
    plan = {
        "schema": R303_HANDOFF_PREFLIGHT_PLAN_SCHEMA,
        "version": 1,
        "status": "planned_no_runtime_or_service_change",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "required_receipts": {
            kind: identity.to_dict() for kind, identity in sorted(identities.items())
        },
        "immutable_parent_checkpoint_sha256": receipts["parent"]["old_checkpoint_sha256"],
        "immutable_parent_optimizer_sha256": receipts["parent"]["old_optimizer_state_sha256"],
        "staged_corpus_source_remote_parity_receipt_sha256": receipts["corpus"][
            "source_remote_parity_receipt_sha256"
        ],
        "frozen_tensor_receipt_card2vec_bit_identity": receipts["frozen_tensors"][
            "card2vec_parameter_and_buffer_bit_identity"
        ],
        "frozen_tensor_receipt_all_parent_routes_bit_identity": receipts["frozen_tensors"][
            "backbone_existing_heads_fusion_own_deck_and_matchup_adapter_bit_identity"
        ],
        "authorized_future_learner": {
            "host": "inzi",
            "device": "cuda:1",
            "device_model": "NVIDIA RTX PRO 5000 Blackwell",
            "new_lineage_id": "alakazam-rule-derivative-g5",
            "new_managed_bootstrap_service": _R303_BOOTSTRAP_SERVICE,
            "new_managed_self_play_service_after_queue": _R303_TRAINER_SERVICE,
            "run_root": roots["run_root"],
            "checkpoint_root": roots["checkpoint_root"],
            "runtime_registry": roots["runtime_registry"],
        },
        "old_r274_services_to_pause_only_at_future_clean_boundary": [
            old["trainer_service"],
            old["submission_boundary_service"],
        ],
        "shared_kaggle_queue_service_must_remain_unchanged": True,
        "clean_boundary_requirements": {
            "current_r274_checkpoint_optimizer_collection_adapter_and_commit_sealed": True,
            "no_mid_collection_or_shard_pause": True,
            "no_mid_optimizer_adapter_checkpoint_or_commit_pause": True,
            "systemd_user_pause_only_no_direct_signal": True,
            "activation_is_not_performed_by_this_plan": True,
        },
        "staged_shards_training_eligible_now": False,
        "activation_receipt_required_before_blackwell_load_or_training": True,
        "production_serving_selector_authority": False,
        "service_control_performed": False,
        "checkpoint_or_registry_mutated": False,
    }
    plan["plan_payload_sha256"] = _plan_checksum(plan)
    return plan


def validate_handoff_preflight_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate that a plan remains explicitly non-activating and exact."""

    row = _mapping(plan, field="handoff preflight plan")
    payload = dict(row)
    claimed = payload.pop("plan_payload_sha256", None)
    if claimed != canonical_sha256(payload):
        raise R303HandoffError("handoff preflight plan checksum drifted")
    required = {
        "schema": R303_HANDOFF_PREFLIGHT_PLAN_SCHEMA,
        "status": "planned_no_runtime_or_service_change",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "staged_shards_training_eligible_now": False,
        "activation_receipt_required_before_blackwell_load_or_training": True,
        "production_serving_selector_authority": False,
        "service_control_performed": False,
        "checkpoint_or_registry_mutated": False,
    }
    if any(row.get(field) != expected for field, expected in required.items()):
        raise R303HandoffError("handoff preflight plan violates non-activation boundary")
    learner = _mapping(row.get("authorized_future_learner"), field="future learner")
    if (
        learner.get("host"), learner.get("device"), learner.get("device_model")
    ) != ("inzi", "cuda:1", "NVIDIA RTX PRO 5000 Blackwell"):
        raise R303HandoffError("handoff plan has an unauthorized learner target")
    pauses = _rows(row.get("old_r274_services_to_pause_only_at_future_clean_boundary"), field="old services")
    if pauses != [
        "pokebot-alakazam-r274-rl.service",
        "pokebot-alakazam-r274-rl-submission-boundaries.service",
    ]:
        raise R303HandoffError("handoff plan has an unsafe service pause set")
    return row


def build_blackwell_launch_contract(
    preflight_plan: Mapping[str, Any],
    *,
    bootstrap_optimizer_config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe, but never invoke, a post-activation Blackwell bootstrap."""

    plan = validate_handoff_preflight_plan(preflight_plan)
    config = _mapping(bootstrap_optimizer_config_identity, field="bootstrap optimizer config")
    config_sha = _sha(config.get("sha256"), field="bootstrap optimizer config sha")
    config_size = _positive_int(config.get("size_bytes"), field="bootstrap optimizer config size")
    config_path = config.get("path")
    if not isinstance(config_path, str) or not config_path:
        raise R303HandoffError("bootstrap optimizer config needs an immutable path")
    launch = {
        "schema": R303_BLACKWELL_LAUNCH_CONTRACT_SCHEMA,
        "version": 1,
        "status": "planned_pending_clean_handoff_activation",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "handoff_preflight_plan_sha256": canonical_sha256(plan),
        "activation_receipt_required": "poke_bot.alakazam_rule_derivative_inzi_training_handoff_activation_receipt/v1",
        "parent_checkpoint_sha256": plan["immutable_parent_checkpoint_sha256"],
        "frozen_tensor_and_card2vec_identity_required": True,
        "future_host": "inzi",
        "future_device": "cuda:1",
        "future_device_model": "NVIDIA RTX PRO 5000 Blackwell",
        "future_bootstrap_service": _R303_BOOTSTRAP_SERVICE,
        "future_self_play_service_after_durable_queue": _R303_TRAINER_SERVICE,
        "bootstrap_optimizer_config_identity": {
            "path": config_path,
            "sha256": config_sha,
            "size_bytes": config_size,
        },
        "trainable_scope_only": [
            "new_public_rule_adapter",
            "repaired_heads",
            "explicitly_eligible_bounded_gates",
        ],
        "frozen_scope": [
            "neural_backbone",
            "existing_heads",
            "fusion",
            "own_deck_routes",
            "matchup_adapters",
            "card2vec",
        ],
        "required_canaries": [
            "safe_parent_load",
            "forward_backward_optimizer",
            "finite_loss_gradient_update",
            "exact_zero_frozen_gradients",
            "frozen_parameters_absent_from_optimizer",
            "layer_off_logits_and_legal_choice_parity",
        ],
        "no_mid_shard_optimizer_adapter_checkpoint_or_commit_switch": True,
        "staged_corpus_training_eligible_now": False,
        "launch_or_service_control_performed": False,
        "production_serving_selector_authority": False,
    }
    launch["plan_payload_sha256"] = _plan_checksum(launch)
    return launch


def validate_blackwell_launch_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    row = _mapping(contract, field="Blackwell launch contract")
    payload = dict(row)
    claimed = payload.pop("plan_payload_sha256", None)
    if claimed != canonical_sha256(payload):
        raise R303HandoffError("Blackwell launch contract checksum drifted")
    required = {
        "schema": R303_BLACKWELL_LAUNCH_CONTRACT_SCHEMA,
        "status": "planned_pending_clean_handoff_activation",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "future_host": "inzi",
        "future_device": "cuda:1",
        "future_device_model": "NVIDIA RTX PRO 5000 Blackwell",
        "future_bootstrap_service": _R303_BOOTSTRAP_SERVICE,
        "future_self_play_service_after_durable_queue": _R303_TRAINER_SERVICE,
        "staged_corpus_training_eligible_now": False,
        "launch_or_service_control_performed": False,
        "production_serving_selector_authority": False,
        "no_mid_shard_optimizer_adapter_checkpoint_or_commit_switch": True,
    }
    if any(row.get(field) != expected for field, expected in required.items()):
        raise R303HandoffError("Blackwell launch contract violates future-only boundary")
    identity = _mapping(row.get("bootstrap_optimizer_config_identity"), field="bootstrap config identity")
    _sha(identity.get("sha256"), field="bootstrap config identity SHA")
    _positive_int(identity.get("size_bytes"), field="bootstrap config identity size")
    return row


def build_rollback_template(preflight_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Emit a non-executable rollback template tied to the immutable parent."""

    plan = validate_handoff_preflight_plan(preflight_plan)
    template = {
        "schema": R303_ROLLBACK_TEMPLATE_SCHEMA,
        "version": 1,
        "status": "planned_no_rollback_execution",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "handoff_preflight_plan_sha256": canonical_sha256(plan),
        "old_lineage_id": "alakazam-new-list-direct-policy-r274",
        "old_parent_checkpoint_sha256": plan["immutable_parent_checkpoint_sha256"],
        "old_optimizer_state_sha256": plan["immutable_parent_optimizer_sha256"],
        "rollback_triggers": [
            "receipt_identity_or_checksum_failure",
            "blackwell_load_or_resource_failure",
            "nonfinite_loss_gradient_or_update",
            "frozen_tensor_or_layer_off_parity_failure",
            "managed_launch_failure",
            "initial_candidate_validation_failure",
            "fleet_package_or_collection_integrity_failure",
        ],
        "future_action_at_safe_boundary_only": (
            "pause_derivative_managed_units_requarantine_derivative_inputs_"
            "restore_exact_old_registry_then_resume_two_r274_units_via_systemd_user"
        ),
        "old_artifacts_deleted_or_overwritten": False,
        "shared_kaggle_queue_service_touched": False,
        "rollback_executed": False,
    }
    template["plan_payload_sha256"] = _plan_checksum(template)
    return template


def _kaggle_contract_spec(contract: Mapping[str, Any], kind: str) -> tuple[str, tuple[str, ...]]:
    root = _mapping(contract.get("bootstrap_validation_and_kaggle"), field="Kaggle contract")
    receipt_contract = _mapping(root.get("receipt_contract"), field="Kaggle receipt contract")
    raw = _mapping(receipt_contract.get(kind), field=f"Kaggle {kind} contract")
    schema = raw.get("schema")
    fields = tuple(_rows(raw.get("required_fields"), field=f"Kaggle {kind} fields"))
    if not isinstance(schema, str) or not schema:
        raise R303HandoffError(f"Kaggle {kind} schema is missing")
    return schema, fields


def validate_kaggle_receipt(kind: str, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate package/queue/upload receipts without queueing or uploading."""

    contract = load_r303_contract()
    schema, fields = _kaggle_contract_spec(contract, kind)
    row = _assert_exact_receipt_inventory(
        receipt, schema=schema, fields=fields, label=f"Kaggle {kind} receipt"
    )
    kaggle = _mapping(contract.get("bootstrap_validation_and_kaggle"), field="Kaggle contract")
    if kind == "package":
        for field, expected in (
            ("deck_canonical_multiset_sha256", kaggle["exact_new_list_canonical_multiset_sha256"]),
            ("canonical_label", kaggle["canonical_label"]),
            ("rtp_enabled", False),
            ("search_or_mcts_enabled", False),
            ("q3_bench_only", True),
            ("q5_q6_trace_only_zero", True),
            ("package_parity_passed", True),
            ("package_smoke_passed", True),
        ):
            if row.get(field) != expected:
                raise R303HandoffError(f"Kaggle package receipt {field} mismatch")
    elif kind == "queue":
        if row.get("competition") != kaggle["competition"] or row.get("turn_order_preference") != kaggle["turn_order_preference"]:
            raise R303HandoffError("Kaggle queue receipt competition/turn order mismatch")
        if row.get("shared_queue_path") != kaggle["shared_queue_path"]:
            raise R303HandoffError("Kaggle queue receipt has a foreign queue path")
        if row.get("queue_entry_status") not in kaggle["queue_entry_status_sufficient_for_self_play"]:
            raise R303HandoffError("Kaggle queue receipt status is not durable/eligible")
        _assert_bool(row, "single_use_consumed", True, label="Kaggle queue receipt")
        _assert_bool(row, "duplicate_or_retry_created", False, label="Kaggle queue receipt")
        _assert_bool(row, "durable_fsync_or_equivalent_passed", True, label="Kaggle queue receipt")
    return row


def inspect_kaggle_receipt(
    kind: str,
    path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity]:
    """Re-open a package, queue, or upload receipt without touching Kaggle."""

    source, receipt, digest = _read_regular_json(path, field=f"Kaggle {kind} receipt")
    if trusted_receipt_root is not None:
        root = Path(trusted_receipt_root)
        if root.is_symlink() or not root.is_dir() or not _path_within(source, root.resolve()):
            raise R303HandoffError("Kaggle receipt lies outside its trusted receipt root")
    checked = validate_kaggle_receipt(kind, receipt)
    return checked, ReceiptIdentity(
        kind=f"kaggle_{kind}",
        path=str(source),
        sha256=digest,
        size_bytes=source.stat().st_size,
        schema=str(checked["schema"]),
    )


def _candidate_receipt_spec(contract: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    root = _mapping(contract.get("baselines_and_candidate"), field="candidate contract")
    schema = root.get("candidate_validation_receipt_schema")
    fields = tuple(
        _rows(root.get("candidate_validation_receipt_required_fields"), field="candidate receipt fields")
    )
    if not isinstance(schema, str) or not schema or not fields:
        raise R303HandoffError("candidate receipt contract is malformed")
    return schema, fields


def validate_candidate_receipt(kind: str, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the strict candidate receipt contract as a read-only consumer."""

    if kind != "candidate":
        raise R303HandoffError("only the canonical candidate receipt kind is supported")
    contract = load_r303_contract()
    schema, fields = _candidate_receipt_spec(contract)
    row = _assert_exact_receipt_inventory(
        receipt, schema=schema, fields=fields, label="candidate receipt"
    )
    baselines = _mapping(contract.get("baselines_and_candidate"), field="candidate contract")
    if (
        row.get("candidate_id") != baselines["derivative_candidate"]["candidate_id"]
        or row.get("baseline_r274_candidate_id")
        != baselines["r241_r274_same_architecture_baseline"]["candidate_id"]
        or row.get("exact_new_list_canonical_multiset_sha256")
        != baselines["r241_r274_same_architecture_baseline"]["exact_new_list_canonical_multiset_sha256"]
    ):
        raise R303HandoffError("candidate receipt has a foreign candidate/baseline/deck identity")
    for field in (
        "trainable_parameters_changed_and_finite",
        "frozen_backbone_true",
        "layer_off_bit_identical_baseline_logits",
        "layer_off_identical_legal_choice",
        "all_frozen_tensor_bit_identity",
        "public_information_metamorphic_tests_passed",
        "simulator_engine_tests_passed",
        "bootstrap_validation_passed",
    ):
        _assert_bool(row, field, True, label="candidate receipt")
    _assert_bool(row, "production_serving_selector_authority", False, label="candidate receipt")
    _positive_int(row.get("candidate_checkpoint_size_bytes"), field="candidate checkpoint size")
    _positive_int(row.get("baseline_r274_checkpoint_size_bytes"), field="baseline checkpoint size")
    return row


def inspect_candidate_receipt(
    path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity]:
    source, receipt, digest = _read_regular_json(path, field="candidate receipt")
    if trusted_receipt_root is not None:
        root = Path(trusted_receipt_root)
        if root.is_symlink() or not root.is_dir() or not _path_within(source, root.resolve()):
            raise R303HandoffError("candidate receipt lies outside its trusted receipt root")
    checked = validate_candidate_receipt("candidate", receipt)
    return checked, ReceiptIdentity(
        kind="candidate",
        path=str(source),
        sha256=digest,
        size_bytes=source.stat().st_size,
        schema=str(checked["schema"]),
    )


def inspect_kaggle_queue_bundle(
    *,
    prerequisite_paths: Mapping[str, Path | str],
    readiness_path: Path | str,
    activation_path: Path | str,
    candidate_path: Path | str,
    package_path: Path | str,
    queue_path: Path | str,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity, Mapping[str, Any], ReceiptIdentity]:
    """Cross-link a durable queue receipt to the exact candidate/package path.

    The function is a validator only: it cannot create a package or a queue
    entry.  An upload receipt remains optional because a durable accepted or
    quota/spacing-pending queue entry is sufficient for the later fleet gate.
    """

    receipts, identities, _readiness, _readiness_id, activation, activation_id = (
        inspect_handoff_activation_bundle(
            prerequisite_paths,
            readiness_path,
            activation_path,
            trusted_receipt_root=trusted_receipt_root,
        )
    )
    candidate, candidate_id = inspect_candidate_receipt(
        candidate_path, trusted_receipt_root=trusted_receipt_root
    )
    package, package_id = inspect_kaggle_receipt(
        "package", package_path, trusted_receipt_root=trusted_receipt_root
    )
    queue, queue_id = inspect_kaggle_receipt(
        "queue", queue_path, trusted_receipt_root=trusted_receipt_root
    )
    schema = receipts["schema_freeze"]
    parent = receipts["parent"]
    if (
        candidate.get("training_handoff_activation_receipt_sha256") != activation_id.sha256
        or candidate.get("parent_receipt_sha256") != identities["parent"].sha256
        or candidate.get("corpus_receipt_sha256") != identities["corpus"].sha256
        or candidate.get("schema_freeze_receipt_sha256") != identities["schema_freeze"].sha256
        or candidate.get("frozen_tensor_receipt_sha256") != identities["frozen_tensors"].sha256
        or candidate.get("blackwell_preflight_receipt_sha256")
        != identities["blackwell_preflight"].sha256
    ):
        raise R303HandoffError("candidate receipt does not bind the completed handoff evidence")
    candidate_sha = candidate["candidate_checkpoint_sha256"]
    if (
        package.get("candidate_validation_receipt_sha256") != candidate_id.sha256
        or package.get("candidate_checkpoint_sha256") != candidate_sha
        or package.get("parent_checkpoint_sha256") != parent["old_checkpoint_sha256"]
        or package.get("feature_schema_sha256") != schema["feature_schema_sha256"]
        or package.get("target_schema_sha256") != schema["target_schema_sha256"]
        or package.get("checklist_provenance_schema_sha256")
        != schema["checklist_provenance_schema_sha256"]
        or package.get("public_catalog_manifest_sha256")
        != schema["public_catalog_manifest_sha256"]
        or package.get("canonical_simulator_sha256") != schema["canonical_simulator_sha256"]
    ):
        raise R303HandoffError("Kaggle package does not bind candidate/parent/schema/catalog evidence")
    kaggle = _mapping(load_r303_contract().get("bootstrap_validation_and_kaggle"), field="Kaggle contract")
    if (
        queue.get("package_receipt_sha256") != package_id.sha256
        or queue.get("package_sha256") != package["package_sha256"]
        or queue.get("candidate_checkpoint_sha256") != candidate_sha
        or queue.get("deck_canonical_multiset_sha256")
        != package["deck_canonical_multiset_sha256"]
        or queue.get("canonical_label") != package["canonical_label"]
        or queue.get("shared_queue_service") != kaggle["shared_queue_service"]
        or queue.get("preexisting_matching_entry_count") != 0
    ):
        raise R303HandoffError("Kaggle queue receipt has a broken package/candidate/dedup linkage")
    dedup_components = {
        "goal_revision": R303_GOAL_REVISION,
        "candidate_checkpoint_sha256": candidate_sha,
        "package_sha256": package["package_sha256"],
        "deck_canonical_multiset_sha256": package["deck_canonical_multiset_sha256"],
        "competition": kaggle["competition"],
        "single_use_authorization_id": queue["single_use_authorization_id"],
    }
    if queue.get("deduplication_key") != canonical_sha256(dedup_components):
        raise R303HandoffError("Kaggle queue receipt has a non-canonical deduplication key")
    return package, package_id, queue, queue_id


def _fleet_contract_spec(contract: Mapping[str, Any], kind: str) -> tuple[str, tuple[str, ...]]:
    root = _mapping(contract.get("full_available_fleet_self_play"), field="fleet contract")
    receipt_contract = _mapping(root.get("receipt_contract"), field="fleet receipt contract")
    raw = _mapping(receipt_contract.get(kind), field=f"fleet {kind} receipt contract")
    schema = raw.get("schema")
    fields = tuple(_rows(raw.get("required_fields"), field=f"fleet {kind} receipt fields"))
    if not isinstance(schema, str) or not schema or not fields:
        raise R303HandoffError(f"fleet {kind} receipt contract is malformed")
    return schema, fields


def _exact_known_hosts(contract: Mapping[str, Any]) -> list[str]:
    fleet = _mapping(contract.get("full_available_fleet_self_play"), field="fleet contract")
    hosts = _rows(fleet.get("known_candidate_hosts"), field="known candidate hosts")
    if hosts != ["inzi", "elmo", "bert"]:
        raise R303HandoffError("canonical known candidate host inventory drifted")
    return [str(host) for host in hosts]


def _host_bool_map(value: Any, *, field: str, hosts: Sequence[str]) -> Mapping[str, Any]:
    row = _mapping(value, field=field)
    if set(row) != set(hosts) or any(row.get(host) is not True for host in hosts):
        raise R303HandoffError(f"{field} must report every known host as passed")
    return row


def _validate_fleet_inventory_receipt(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    hosts = _exact_known_hosts(contract)
    if _rows(row.get("known_candidate_hosts"), field="fleet inventory hosts") != hosts:
        raise R303HandoffError("fleet inventory does not enumerate every canonical host exactly once")
    _assert_bool(row, "full_available_fleet_included", True, label="fleet inventory receipt")
    ground_truth = _mapping(contract.get("ground_truth"), field="ground truth")
    simulator = _mapping(ground_truth.get("canonical_simulator"), field="canonical simulator")
    if row.get("canonical_simulator_sha256") != simulator["linux_x86_64_sha256"]:
        raise R303HandoffError("fleet inventory simulator identity drifted")
    for field in (
        "per_host_availability_eligibility_and_exclusion_evidence",
        "per_host_os_arch_cpu_ram_and_network_identity",
        "per_host_gpu_device_models_uuids_memory_and_runtime_versions",
        "per_host_simulator_capacity_and_pack_sizes",
        "per_host_resource_caps",
        "per_host_managed_worker_service_names",
    ):
        item = _mapping(row.get(field), field=f"fleet inventory {field}")
        if set(item) != set(hosts):
            raise R303HandoffError(f"fleet inventory {field} omits or adds a host")


def _validate_fleet_package_parity_receipt(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    hosts = _exact_known_hosts(contract)
    _assert_bool(row, "all_available_hosts_checksum_exact", True, label="fleet package parity receipt")
    _host_bool_map(row.get("per_host_engine_smoke_passed"), field="per-host engine smoke", hosts=hosts)
    # The per-host paths and layer-off/action parity details remain typed
    # records rather than assumptions; their host universe must be complete.
    for field in (
        "per_host_package_path_sha256_size_and_member_manifest",
        "per_host_layer_off_and_action_parity",
    ):
        values = _mapping(row.get(field), field=f"fleet package parity {field}")
        if set(values) != set(hosts):
            raise R303HandoffError(f"fleet package parity {field} omits or adds a host")


def _validate_fleet_routing_receipt(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    hosts = _exact_known_hosts(contract)
    if row.get("learner_host") != "inzi":
        raise R303HandoffError("fleet routing moves the learner off Inzi")
    for field in (
        "no_oversubscription",
        "no_unreceipted_fallback",
    ):
        _assert_bool(row, field, True, label="fleet routing receipt")
    _positive_int(row.get("blackwell_training_headroom_bytes"), field="Blackwell training headroom")
    for field in (
        "per_host_simulator_process_count",
        "per_process_environment_count",
        "per_host_policy_leaf_device_uuid_and_worker_ranges",
        "per_host_concurrency_and_resource_caps",
        "capacity_probe_results",
    ):
        values = _mapping(row.get(field), field=f"fleet routing {field}")
        if set(values) != set(hosts):
            raise R303HandoffError(f"fleet routing {field} omits or adds a host")


def _validate_fleet_activation_receipt(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    fleet = _mapping(contract.get("full_available_fleet_self_play"), field="fleet contract")
    kaggle = _mapping(contract.get("bootstrap_validation_and_kaggle"), field="Kaggle contract")
    if (
        row.get("kaggle_queue_entry_status") not in kaggle["queue_entry_status_sufficient_for_self_play"]
        or row.get("new_managed_trainer_service") != fleet["new_managed_trainer_service"]
    ):
        raise R303HandoffError("fleet activation has an ineligible queue state or foreign trainer")
    for field in (
        "old_r274_services_inactive_verified",
        "shared_kaggle_queue_service_unchanged",
        "no_old_r274_training_or_collection",
        "no_serving_selector_change",
    ):
        _assert_bool(row, field, True, label="fleet activation receipt")
    for field in ("global_collection_id", "collection_manifest_path", "global_game_identity_schema", "lease_and_duplicate_suppression_schema"):
        if not isinstance(row.get(field), str) or not row.get(field):
            raise R303HandoffError(f"fleet activation lacks {field}")


def _validate_collection_shard_receipt(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    _positive_int(row.get("size_bytes"), field="collection shard size")
    for field in ("iteration", "duplicate_identical_count", "duplicate_conflict_count", "missing_game_count", "simulator_failure_count"):
        _positive_int(row.get(field), field=f"collection shard {field}", allow_zero=True)
    _assert_bool(row, "selected_action_legality_passed", True, label="collection shard receipt")
    _assert_bool(row, "schema_validation_passed", True, label="collection shard receipt")
    _assert_bool(row, "sealed_append_only", True, label="collection shard receipt")
    if row.get("duplicate_conflict_count") != 0:
        raise R303HandoffError("collection shard contains conflicting duplicate games")
    game_ids = _rows(row.get("ordered_global_game_ids"), field="collection shard game identities")
    if not game_ids or any(not isinstance(game_id, str) or not game_id for game_id in game_ids):
        raise R303HandoffError("collection shard lacks global game identities")
    if len(set(game_ids)) != len(game_ids):
        raise R303HandoffError("collection shard repeats a global game identity")


_FLEET_VALIDATORS: Final = {
    "fleet_inventory": _validate_fleet_inventory_receipt,
    "package_parity": _validate_fleet_package_parity_receipt,
    "routing": _validate_fleet_routing_receipt,
    "activation": _validate_fleet_activation_receipt,
    "collection_shard": _validate_collection_shard_receipt,
}


def validate_fleet_receipt(kind: str, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate one strict fleet receipt without launching fleet work."""

    contract = load_r303_contract()
    schema, fields = _fleet_contract_spec(contract, kind)
    row = _assert_exact_receipt_inventory(
        receipt, schema=schema, fields=fields, label=f"fleet {kind} receipt"
    )
    validator = _FLEET_VALIDATORS.get(kind)
    if validator is None:
        raise R303HandoffError(f"unsupported fleet receipt kind: {kind}")
    validator(row, contract)
    return row


def inspect_fleet_receipt(
    kind: str,
    path: Path | str,
    *,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity]:
    source, receipt, digest = _read_regular_json(path, field=f"fleet {kind} receipt")
    if trusted_receipt_root is not None:
        root = Path(trusted_receipt_root)
        if root.is_symlink() or not root.is_dir() or not _path_within(source, root.resolve()):
            raise R303HandoffError("fleet receipt lies outside its trusted receipt root")
    checked = validate_fleet_receipt(kind, receipt)
    return checked, ReceiptIdentity(
        kind=f"fleet_{kind}",
        path=str(source),
        sha256=digest,
        size_bytes=source.stat().st_size,
        schema=str(checked["schema"]),
    )


def inspect_fleet_activation_bundle(
    *,
    prerequisite_paths: Mapping[str, Path | str],
    readiness_path: Path | str,
    handoff_activation_path: Path | str,
    candidate_path: Path | str,
    package_path: Path | str,
    queue_path: Path | str,
    fleet_inventory_path: Path | str,
    package_parity_path: Path | str,
    routing_path: Path | str,
    fleet_activation_path: Path | str,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, ReceiptIdentity]]:
    """Cross-link all evidence that would precede fleet activation.

    This is deliberately inspection-only.  A successful return means the
    supplied receipts agree; it does not start any managed worker or learner.
    """

    package, package_id, queue, queue_id = inspect_kaggle_queue_bundle(
        prerequisite_paths=prerequisite_paths,
        readiness_path=readiness_path,
        activation_path=handoff_activation_path,
        candidate_path=candidate_path,
        package_path=package_path,
        queue_path=queue_path,
        trusted_receipt_root=trusted_receipt_root,
    )
    candidate, candidate_id = inspect_candidate_receipt(
        candidate_path, trusted_receipt_root=trusted_receipt_root
    )
    _receipts, handoff_ids, _readiness, _readiness_id, handoff_activation, handoff_activation_id = (
        inspect_handoff_activation_bundle(
            prerequisite_paths,
            readiness_path,
            handoff_activation_path,
            trusted_receipt_root=trusted_receipt_root,
        )
    )
    inventory, inventory_id = inspect_fleet_receipt(
        "fleet_inventory", fleet_inventory_path, trusted_receipt_root=trusted_receipt_root
    )
    parity, parity_id = inspect_fleet_receipt(
        "package_parity", package_parity_path, trusted_receipt_root=trusted_receipt_root
    )
    routing, routing_id = inspect_fleet_receipt(
        "routing", routing_path, trusted_receipt_root=trusted_receipt_root
    )
    activation, activation_id = inspect_fleet_receipt(
        "activation", fleet_activation_path, trusted_receipt_root=trusted_receipt_root
    )
    if (
        parity.get("fleet_inventory_receipt_sha256") != inventory_id.sha256
        or parity.get("candidate_checkpoint_sha256") != candidate["candidate_checkpoint_sha256"]
        or routing.get("fleet_inventory_receipt_sha256") != inventory_id.sha256
        or routing.get("package_parity_receipt_sha256") != parity_id.sha256
    ):
        raise R303HandoffError("fleet parity/routing receipts are not cross-linked")
    if (
        activation.get("training_handoff_activation_receipt_sha256") != handoff_activation_id.sha256
        or activation.get("candidate_validation_receipt_sha256") != candidate_id.sha256
        or activation.get("kaggle_queue_receipt_sha256") != queue_id.sha256
        or activation.get("kaggle_queue_entry_status") != queue["queue_entry_status"]
        or activation.get("fleet_inventory_receipt_sha256") != inventory_id.sha256
        or activation.get("package_parity_receipt_sha256") != parity_id.sha256
        or activation.get("routing_receipt_sha256") != routing_id.sha256
    ):
        raise R303HandoffError("fleet activation receipt is not cross-linked to its prerequisites")
    return (
        {
            "package": package,
            "queue": queue,
            "inventory": inventory,
            "package_parity": parity,
            "routing": routing,
            "activation": activation,
            "handoff_activation": handoff_activation,
        },
        {
            "package": package_id,
            "queue": queue_id,
            "candidate": candidate_id,
            "handoff_activation": handoff_activation_id,
            "fleet_inventory": inventory_id,
            "package_parity": parity_id,
            "routing": routing_id,
            "fleet_activation": activation_id,
            **{f"handoff_{kind}": identity for kind, identity in handoff_ids.items()},
        },
    )


def inspect_collection_shard_against_fleet_activation(
    collection_shard_path: Path | str,
    *,
    fleet_activation_path: Path | str,
    trusted_receipt_root: Path | str | None = None,
) -> tuple[Mapping[str, Any], ReceiptIdentity]:
    """Validate one sealed append-only shard against its fleet activation."""

    activation, activation_id = inspect_fleet_receipt(
        "activation", fleet_activation_path, trusted_receipt_root=trusted_receipt_root
    )
    shard, shard_id = inspect_fleet_receipt(
        "collection_shard", collection_shard_path, trusted_receipt_root=trusted_receipt_root
    )
    if (
        shard.get("fleet_activation_receipt_sha256") != activation_id.sha256
        or shard.get("global_collection_id") != activation.get("global_collection_id")
    ):
        raise R303HandoffError("collection shard is not bound to the active fleet collection")
    return shard, shard_id


def build_fleet_preflight_plan(
    *,
    prerequisite_paths: Mapping[str, Path | str],
    readiness_path: Path | str,
    handoff_activation_path: Path | str,
    candidate_path: Path | str,
    package_path: Path | str,
    queue_path: Path | str,
    fleet_inventory_path: Path | str,
    package_parity_path: Path | str,
    routing_path: Path | str,
    trusted_receipt_root: Path | str | None = None,
) -> dict[str, Any]:
    """Prepare a read-only fleet plan after queue durability, never activate it."""

    package, package_id, queue, queue_id = inspect_kaggle_queue_bundle(
        prerequisite_paths=prerequisite_paths,
        readiness_path=readiness_path,
        activation_path=handoff_activation_path,
        candidate_path=candidate_path,
        package_path=package_path,
        queue_path=queue_path,
        trusted_receipt_root=trusted_receipt_root,
    )
    candidate, candidate_id = inspect_candidate_receipt(
        candidate_path, trusted_receipt_root=trusted_receipt_root
    )
    _receipts, _handoff_ids, _readiness, _readiness_id, handoff_activation, handoff_activation_id = (
        inspect_handoff_activation_bundle(
            prerequisite_paths,
            readiness_path,
            handoff_activation_path,
            trusted_receipt_root=trusted_receipt_root,
        )
    )
    inventory, inventory_id = inspect_fleet_receipt(
        "fleet_inventory", fleet_inventory_path, trusted_receipt_root=trusted_receipt_root
    )
    parity, parity_id = inspect_fleet_receipt(
        "package_parity", package_parity_path, trusted_receipt_root=trusted_receipt_root
    )
    routing, routing_id = inspect_fleet_receipt(
        "routing", routing_path, trusted_receipt_root=trusted_receipt_root
    )
    if (
        parity.get("fleet_inventory_receipt_sha256") != inventory_id.sha256
        or parity.get("candidate_checkpoint_sha256") != candidate["candidate_checkpoint_sha256"]
        or routing.get("fleet_inventory_receipt_sha256") != inventory_id.sha256
        or routing.get("package_parity_receipt_sha256") != parity_id.sha256
    ):
        raise R303HandoffError("fleet preflight evidence is not cross-linked")
    plan = {
        "schema": R303_FLEET_PREFLIGHT_PLAN_SCHEMA,
        "version": 1,
        "status": "planned_pending_fleet_activation_no_runtime_or_service_change",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "training_handoff_activation_receipt_sha256": handoff_activation_id.sha256,
        "candidate_validation_receipt_sha256": candidate_id.sha256,
        "kaggle_package_receipt_sha256": package_id.sha256,
        "kaggle_queue_receipt_sha256": queue_id.sha256,
        "kaggle_queue_entry_status": queue["queue_entry_status"],
        "fleet_inventory_receipt_sha256": inventory_id.sha256,
        "package_parity_receipt_sha256": parity_id.sha256,
        "routing_receipt_sha256": routing_id.sha256,
        "frozen_bootstrap_checkpoint_sha256": candidate["candidate_checkpoint_sha256"],
        "package_sha256": package["package_sha256"],
        "known_candidate_hosts": _exact_known_hosts(load_r303_contract()),
        "learner_host": routing["learner_host"],
        "activation_or_worker_service_control_performed": False,
        "collection_or_training_started": False,
        "old_r274_lineage_resumed_or_started": False,
        "production_serving_selector_authority": False,
    }
    plan["plan_payload_sha256"] = _plan_checksum(plan)
    return plan


def validate_fleet_preflight_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    row = _mapping(plan, field="fleet preflight plan")
    payload = dict(row)
    claimed = payload.pop("plan_payload_sha256", None)
    if claimed != canonical_sha256(payload):
        raise R303HandoffError("fleet preflight plan checksum drifted")
    required = {
        "schema": R303_FLEET_PREFLIGHT_PLAN_SCHEMA,
        "status": "planned_pending_fleet_activation_no_runtime_or_service_change",
        "goal_contract_path": R303_CONTRACT_PATH,
        "goal_contract_sha256": R303_CONTRACT_SHA256,
        "goal_gateway_sha256": R303_GOAL_GATEWAY_SHA256,
        "goal_revision": R303_GOAL_REVISION,
        "root_owner_revision": R303_ROOT_OWNER_REVISION,
        "known_candidate_hosts": ["inzi", "elmo", "bert"],
        "learner_host": "inzi",
        "activation_or_worker_service_control_performed": False,
        "collection_or_training_started": False,
        "old_r274_lineage_resumed_or_started": False,
        "production_serving_selector_authority": False,
    }
    if any(row.get(field) != expected for field, expected in required.items()):
        raise R303HandoffError("fleet preflight plan violates future-only activation boundary")
    for field in (
        "training_handoff_activation_receipt_sha256",
        "candidate_validation_receipt_sha256",
        "kaggle_package_receipt_sha256",
        "kaggle_queue_receipt_sha256",
        "fleet_inventory_receipt_sha256",
        "package_parity_receipt_sha256",
        "routing_receipt_sha256",
        "frozen_bootstrap_checkpoint_sha256",
        "package_sha256",
    ):
        _sha(row.get(field), field=f"fleet preflight plan {field}")
    return row


def global_game_identity(
    *,
    lineage_id: str,
    iteration: int,
    phase: str,
    global_game_index: int,
    seed: int,
    seat: int,
    opponent_package_sha256: str,
    simulator_sha256: str,
    candidate_checkpoint_sha256: str,
) -> str:
    """Deterministically derive the one globally unique fleet-game identity."""

    if lineage_id != "alakazam-rule-derivative-g5" or not isinstance(phase, str) or not phase:
        raise R303HandoffError("global game identity has an invalid lineage/phase")
    for field, value in (
        ("iteration", iteration),
        ("global_game_index", global_game_index),
        ("seed", seed),
        ("seat", seat),
    ):
        _positive_int(value, field=field, allow_zero=True)
    for field, value in (
        ("opponent package", opponent_package_sha256),
        ("simulator", simulator_sha256),
        ("candidate checkpoint", candidate_checkpoint_sha256),
    ):
        _sha(value, field=field)
    return canonical_sha256(
        {
            "lineage_id": lineage_id,
            "iteration": iteration,
            "phase": phase,
            "global_game_index": global_game_index,
            "seed": seed,
            "seat": seat,
            "opponent_package_sha256": opponent_package_sha256,
            "simulator_sha256": simulator_sha256,
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        }
    )


def _write_create_only_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.is_symlink() or path.exists():
        raise R303HandoffError(f"create-only r303 artifact already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise R303HandoffError("r303 artifact parent must be a regular existing directory")
    encoded = (_canonical_json(value) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise R303HandoffError(f"create-only r303 artifact already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def write_nonactivating_plan_create_only(path: Path | str, plan: Mapping[str, Any]) -> str:
    """Persist only known r303 *planned* artifacts, never a pass/activation."""

    row = _mapping(plan, field="r303 plan")
    schema = row.get("schema")
    if schema == R303_HANDOFF_PREFLIGHT_PLAN_SCHEMA:
        validate_handoff_preflight_plan(row)
    elif schema == R303_BLACKWELL_LAUNCH_CONTRACT_SCHEMA:
        validate_blackwell_launch_contract(row)
    elif schema == R303_FLEET_PREFLIGHT_PLAN_SCHEMA:
        validate_fleet_preflight_plan(row)
    elif schema == R303_ROLLBACK_TEMPLATE_SCHEMA:
        if row.get("status") != "planned_no_rollback_execution" or row.get("rollback_executed") is not False:
            raise R303HandoffError("rollback template is not explicitly non-executing")
    else:
        raise R303HandoffError("refusing to write an unknown or activating r303 artifact")
    return _write_create_only_json(Path(path), row)


__all__ = [
    "R303_BLACKWELL_LAUNCH_CONTRACT_SCHEMA",
    "R303_CONTRACT_SHA256",
    "R303_GOAL_GATEWAY_SHA256",
    "R303_GOAL_REVISION",
    "R303_HANDOFF_PREFLIGHT_PLAN_SCHEMA",
    "R303HandoffError",
    "R303_FLEET_PREFLIGHT_PLAN_SCHEMA",
    "R303_ROLLBACK_TEMPLATE_SCHEMA",
    "ReceiptIdentity",
    "build_blackwell_launch_contract",
    "build_fleet_preflight_plan",
    "build_handoff_preflight_plan",
    "build_rollback_template",
    "canonical_sha256",
    "global_game_identity",
    "inspect_handoff_bundle",
    "inspect_handoff_activation_bundle",
    "inspect_handoff_receipt",
    "inspect_kaggle_queue_bundle",
    "inspect_kaggle_receipt",
    "inspect_candidate_receipt",
    "inspect_fleet_activation_bundle",
    "inspect_fleet_receipt",
    "inspect_collection_shard_against_fleet_activation",
    "inspect_preemption_readiness_bundle",
    "load_r303_contract",
    "sha256_file",
    "validate_blackwell_launch_contract",
    "validate_candidate_receipt",
    "validate_fleet_preflight_plan",
    "validate_fleet_receipt",
    "validate_handoff_preflight_plan",
    "validate_handoff_receipt",
    "validate_kaggle_receipt",
    "write_nonactivating_plan_create_only",
]
