#!/usr/bin/env python3
"""Create a narrow, local-only r260 pre-start supersession receipt.

This tool deliberately does not contact Inzi, Elmo, Docker, systemd, Kaggle,
or a worker endpoint.  A caller first captures bounded, read-only host
observations and passes those immutable JSON files here.  The receipt fails
closed unless it includes retained service-journal and Docker inventory/event
evidence in addition to current process/listener/filesystem observations.  It
never turns a historical staging artifact into a training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "poke_bot.alakazam_r260_prestart_supersession_gate/v1"
OBSERVATION_SCHEMA = "poke_bot.alakazam_r260_prestart_host_observation/v1"
R241_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241/v1"
R258_SCHEMA = "poke_bot.alakazam_own_deck_ledger_successor_r258/v1"
R195_SCHEMA = "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1"
LEGACY_LEDGER_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_activation_evidence_controller_ledger/v1"
)
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
LATEST_OWNER_CLARIFICATION_REVISION = 262
INZI_TRAINING_ROOT = (
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/r260-own-deck-training-dataset"
)
INZI_PREFIX_STAGING_ROOT = INZI_TRAINING_ROOT + "-staging-09848f04"
SHA256_PREFIX = "sha256:"


class R260PrestartGateError(RuntimeError):
    """The r260 pre-start boundary is not proven."""


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R260PrestartGateError("receipt is not canonical JSON") from exc


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return SHA256_PREFIX + digest.hexdigest()


def _regular_file(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R260PrestartGateError(f"{label} must be a regular non-symlink file: {raw}")
    return raw.resolve()


def _read_json(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    file = _regular_file(path, label=label)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R260PrestartGateError(f"{label} is not readable JSON: {file}") from exc
    if not isinstance(payload, dict):
        raise R260PrestartGateError(f"{label} must contain an object")
    return file, payload


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R260PrestartGateError(f"{label} must be an object")
    return dict(value)


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise R260PrestartGateError(f"{label} must be a list")
    return list(value)


def _valid_sha256(value: object, *, label: str) -> str:
    rendered = str(value or "")
    if not rendered.startswith(SHA256_PREFIX) or len(rendered) != 71:
        raise R260PrestartGateError(f"{label} lacks a canonical SHA-256")
    try:
        int(rendered.removeprefix(SHA256_PREFIX), 16)
    except ValueError as exc:
        raise R260PrestartGateError(f"{label} is not hexadecimal") from exc
    return rendered


def _identity(path: Path, *, label: str) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _validate_contracts(
    *,
    r241: Mapping[str, Any],
    r258: Mapping[str, Any],
    r195: Mapping[str, Any],
    legacy_ledger: Mapping[str, Any],
) -> dict[str, object]:
    imported = _mapping(
        r241.get("own_deck_head_structure_import"), label="r260 own-deck import"
    )
    override = _mapping(imported.get("pre_start_override"), label="r260 pre-start override")
    migration = _mapping(imported.get("migration"), label="r260 migration")
    placement = _mapping(imported.get("training_placement"), label="r262 training placement")
    if (
        r241.get("schema") != R241_SCHEMA
        or r241.get("candidate_id") != CANDIDATE_ID
        or r241.get("latest_owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or imported.get("owner_revision") != 260
        or imported.get("provenance_manifest")
        != "state/alakazam-own-deck-ledger-successor-r258.json"
        or override.get("supersedes_r258_wait_for_r241_completion_only_for_this_successor_boundary")
        is not True
        or migration.get("zero_safe") is not True
        or migration.get("typed_parent_file_identity_keys")
        != ["path", "sha256", "size_bytes"]
        or placement.get("owner_revision") != LATEST_OWNER_CLARIFICATION_REVISION
        or placement.get("sole_managed_training_host") != "inzi"
        or placement.get("elmo_may_train_learner") is not False
        or placement.get("canonical_inzi_training_root") != INZI_TRAINING_ROOT
        or placement.get("inzi_prefix_staging_root") != INZI_PREFIX_STAGING_ROOT
        or placement.get("prefix_transfer_while_elmo_builder_runs") is not True
        or placement.get("partial_staging_root_training_eligible") is not False
        or placement.get("trainer_may_consume_elmo_mnt_main_path") is not False
        or placement.get("healthy_r259_service_may_be_stopped_restarted_or_reconfigured")
        is not False
    ):
        raise R260PrestartGateError("r241 contract does not authorize the narrow r260 pre-start override")
    if (
        r258.get("schema") != R258_SCHEMA
        or r258.get("canonical") is not True
        or r258.get("candidate_id") != "alakazam-own-deck-ledger-successor-r258"
        or r195.get("schema") != R195_SCHEMA
    ):
        raise R260PrestartGateError("r258 or r195 immutable provenance is invalid")
    completion = _mapping(r195.get("completion"), label="r195 completion")
    expected_parent = {
        "path": completion.get("expert_checkpoint"),
        "sha256": completion.get("expert_checkpoint_sha256"),
        "size_bytes": completion.get("expert_checkpoint_bytes"),
    }
    if (
        not isinstance(expected_parent["path"], str)
        or not expected_parent["path"]
        or _valid_sha256(expected_parent["sha256"], label="r195 parent checkpoint")
        != "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
        or expected_parent["size_bytes"] != 127914385
    ):
        raise R260PrestartGateError("r195 terminal parent FileIdentity is invalid")
    boundary = _mapping(legacy_ledger.get("execution_boundary"), label="legacy execution boundary")
    h10 = _mapping(legacy_ledger.get("h10_v8"), label="legacy H10-v8 receipts")
    peak = _mapping(legacy_ledger.get("peak_v6"), label="legacy peak-v6 receipts")
    worker = _mapping(legacy_ledger.get("worker_image_r13"), label="legacy r13 image receipt")
    preflight = _mapping(
        legacy_ledger.get("canonical_worker_preflight"), label="legacy worker preflight"
    )
    expected_false = {
        "environment_installed",
        "unit_installed",
        "service_started",
        "worker_armed",
        "listener_started",
        "training_started",
        "submission_started",
    }
    if (
        legacy_ledger.get("schema") != LEGACY_LEDGER_SCHEMA
        or legacy_ledger.get("candidate_id") != CANDIDATE_ID
        or legacy_ledger.get("status")
        != "activation_overlay_published_and_mirrored_services_not_started"
        or set(boundary) != expected_false
        or any(boundary[key] is not False for key in expected_false)
        or any(_valid_sha256(h10.get(host), label=f"legacy H10-v8 {host}") is None for host in ("inzi_receipt_sha256", "elmo_receipt_sha256"))
        or any(_valid_sha256(peak.get(host), label=f"legacy peak-v6 {host}") is None for host in ("inzi_receipt_sha256", "elmo_receipt_sha256"))
        or _valid_sha256(worker.get("image_id_sha256"), label="legacy r13 image") is None
        or _valid_sha256(worker.get("receipt_sha256"), label="legacy r13 receipt") is None
        or preflight.get("deployment_action") != "not_started"
        or any(
            _valid_sha256(preflight.get(key), label=f"legacy worker preflight {key}") is None
            for key in ("host_sha256", "runtime_sha256", "gameplay_sha256", "manifest_sha256")
        )
    ):
        raise R260PrestartGateError("legacy r241 execution ledger is not an all-false pre-start boundary")
    return {
        "r195_parent": expected_parent,
        "legacy_execution_boundary": {key: False for key in sorted(expected_false)},
        "legacy_quartet": {
            "h10_v8": dict(h10),
            "peak_v6": dict(peak),
            "worker_image_r13": dict(worker),
            "canonical_worker_preflight": dict(preflight),
        },
    }


def _validate_unit(unit: Mapping[str, Any], *, host: str) -> dict[str, object]:
    required = {
        "id",
        "load_state",
        "active_state",
        "sub_state",
        "exec_main_start_timestamp",
        "active_enter_timestamp",
        "invocation_id",
        "n_restarts",
    }
    if set(unit) != required:
        raise R260PrestartGateError(f"{host} unit observation has unsupported fields")
    if (
        not isinstance(unit["id"], str)
        or not unit["id"]
        or unit["load_state"] not in {"loaded", "not-found"}
        or unit["active_state"] != "inactive"
        or unit["sub_state"] != "dead"
        or unit["exec_main_start_timestamp"] not in {None, ""}
        or unit["active_enter_timestamp"] not in {None, ""}
        or unit["invocation_id"] not in {None, ""}
        or unit["n_restarts"] != 0
    ):
        raise R260PrestartGateError(f"{host} unit was started or has untrusted history: {unit.get('id')}")
    return dict(unit)


def _validate_observation(
    observation: Mapping[str, Any], *, expected_host: str, expected_units: set[str]
) -> dict[str, object]:
    if (
        observation.get("schema") != OBSERVATION_SCHEMA
        or observation.get("host") != expected_host
        or not isinstance(observation.get("observed_at_utc"), str)
        or not observation["observed_at_utc"]
        or observation.get("scope") != "managed_r241_execution_topology"
    ):
        raise R260PrestartGateError(f"{expected_host} observation identity is invalid")
    units = [_validate_unit(_mapping(item, label=f"{expected_host} unit"), host=expected_host)
             for item in _list(observation.get("units"), label=f"{expected_host} units")]
    if {str(unit["id"]) for unit in units} != expected_units:
        raise R260PrestartGateError(f"{expected_host} observation omits or adds a managed r241 unit")
    arm_paths = _list(observation.get("arm_paths"), label=f"{expected_host} arm paths")
    if not arm_paths or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("path"), str)
        or not item.get("path")
        or item.get("exists") is not False
        for item in arm_paths
    ):
        raise R260PrestartGateError(f"{expected_host} arm path is present or unproven")
    artifacts = _mapping(observation.get("execution_artifacts"), label=f"{expected_host} artifacts")
    expected_artifacts = {
        "training_updates",
        "checkpoints",
        "commits",
        "loop_state",
        "logs",
        "packages",
        "finalizers",
        "queues",
        "uploads",
        "submissions",
    }
    if set(artifacts) != expected_artifacts or any(
        _list(value, label=f"{expected_host} artifact list") for value in artifacts.values()
    ):
        raise R260PrestartGateError(f"{expected_host} has r241 execution artifacts")
    processes = _mapping(observation.get("processes"), label=f"{expected_host} processes")
    if processes != {"matching_count": 0}:
        raise R260PrestartGateError(f"{expected_host} has an r241 process")
    container_route = _mapping(observation.get("container_route"), label=f"{expected_host} container route")
    if container_route != {
        "managed_service_started": False,
        "arm_present": False,
        "listener_count": 0,
        "direct_daemon_inventory": "verified_no_current_or_retained_r241_container",
        "events_since_r241_staging": "verified_zero_matches",
    }:
        raise R260PrestartGateError(f"{expected_host} container route is not proven unstarted")
    journal = _mapping(observation.get("journal_history"), label=f"{expected_host} journal")
    if (
        journal.get("readable") is not True
        or journal.get("matching_r241_unit_entries") != 0
        or not isinstance(journal.get("coverage_start_utc"), str)
        or not journal["coverage_start_utc"]
    ):
        raise R260PrestartGateError(f"{expected_host} retained r241 service journal is absent or nonempty")
    listener_count = observation.get("listener_count")
    if expected_host == "elmo":
        if listener_count != 0:
            raise R260PrestartGateError("Elmo r241 :8767 listener is present or unproven")
    elif listener_count is not None:
        raise R260PrestartGateError("Inzi observation may not claim an Elmo listener probe")
    selector = observation.get("selector_r241_active_reference")
    if expected_host == "inzi":
        if selector is not False:
            raise R260PrestartGateError("Inzi selector still references r241")
    elif selector is not None:
        raise R260PrestartGateError("Elmo observation may not claim the Inzi selector")
    return {
        "host": expected_host,
        "observed_at_utc": observation["observed_at_utc"],
        "unit_ids": sorted(str(unit["id"]) for unit in units),
        "listener_count": listener_count,
        "selector_r241_active_reference": selector,
        "journal_history": dict(journal),
        "direct_daemon_inventory": "verified_no_current_or_retained_r241_container",
    }


def build_receipt(
    *,
    r241_contract: Path | str,
    r258_contract: Path | str,
    r195_contract: Path | str,
    legacy_execution_ledger: Path | str,
    inzi_observation: Path | str,
    elmo_observation: Path | str,
) -> dict[str, object]:
    """Validate only evidence passed by the caller and return a canonical receipt."""

    r241_path, r241 = _read_json(r241_contract, label="r241 r260 owner contract")
    r258_path, r258 = _read_json(r258_contract, label="r258 provenance contract")
    r195_path, r195 = _read_json(r195_contract, label="r195 parent contract")
    legacy_path, legacy = _read_json(legacy_execution_ledger, label="legacy r241 execution ledger")
    inzi_path, inzi = _read_json(inzi_observation, label="Inzi host observation")
    elmo_path, elmo = _read_json(elmo_observation, label="Elmo host observation")
    contracts = _validate_contracts(r241=r241, r258=r258, r195=r195, legacy_ledger=legacy)
    inzi_summary = _validate_observation(
        inzi,
        expected_host="inzi",
        expected_units={
            "pokebot-alakazam-new-list-direct-r241.service",
            "pokebot-alakazam-new-list-direct-r241-finalize.service",
            "pokebot-alakazam-new-list-direct-r241-submission-queue.service",
            "pokebot-alakazam-new-list-direct-r241-upload.service",
        },
    )
    elmo_summary = _validate_observation(
        elmo,
        expected_host="elmo",
        expected_units={"pokebot-r241-elmo-official-r236-remote-worker.service"},
    )
    return {
        "schema": SCHEMA,
        "revision": 260,
        "latest_owner_clarification_revision": LATEST_OWNER_CLARIFICATION_REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "passed_prestart_supersession_for_managed_r241_topology",
        "scope": {
            "supersession": "r258 post-refresh/no-r241-mutation delay only at the r241 pre-start successor boundary",
            "does_not_authorize": [
                "training_service_start",
                "selector_change",
                "package_creation",
                "upload",
                "submission",
            ],
            "historical_limit": "The receipt asserts the managed r241 topology only across the supplied retained systemd-journal and Docker-event coverage window; it does not claim visibility into unrelated, manually deleted, out-of-band containers outside that retained history.",
        },
        "immutable_inputs": {
            "r241_r260_contract": _identity(r241_path, label="r241 r260 owner contract"),
            "r258_provenance_contract": _identity(r258_path, label="r258 provenance contract"),
            "r195_parent_contract": _identity(r195_path, label="r195 parent contract"),
            "legacy_execution_ledger": _identity(legacy_path, label="legacy r241 execution ledger"),
        },
        "parent_file_identity": contracts["r195_parent"],
        "legacy_execution_boundary": contracts["legacy_execution_boundary"],
        "legacy_quartet": contracts["legacy_quartet"],
        "host_observations": {
            "inzi": {**inzi_summary, "identity": _identity(inzi_path, label="Inzi host observation")},
            "elmo": {**elmo_summary, "identity": _identity(elmo_path, label="Elmo host observation")},
        },
        "required_follow_on_gates": [
            "completed_20_shard_elmo_side_store_join_schema_count_digest_causal_parity_and_completion_receipts",
            "zero_safe_r195_to_r260_migration_receipt",
            "bounded_training_canary_and_runtime_influence_receipts",
            "fresh_source_h10_peak_worker_image_preflight_and_byte_identical_overlay_receipts",
        ],
    }


def validate_sealed_receipt(
    *,
    receipt_path: Path | str,
    expected_sha256: str,
    expected_r241_contract_sha256: str,
) -> dict[str, Any]:
    """Reopen a sealed gate for a later launcher without touching a host.

    A launcher should call this together with its ordinary immutable-source
    checks.  It verifies the receipt's exact digest and the small set of facts
    that make it a pre-start supersession receipt; it does not recreate host
    observations or grant managed-service authority.
    """

    expected_receipt = _valid_sha256(expected_sha256, label="r260 pre-start receipt")
    expected_contract = _valid_sha256(
        expected_r241_contract_sha256, label="r260 r241 owner contract"
    )
    path, receipt = _read_json(receipt_path, label="r260 pre-start receipt")
    if _sha256_file(path) != expected_receipt:
        raise R260PrestartGateError("r260 pre-start receipt checksum mismatch")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("revision") != 260
        or receipt.get("latest_owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("status") != "passed_prestart_supersession_for_managed_r241_topology"
    ):
        raise R260PrestartGateError("r260 pre-start receipt identity is invalid")
    immutable = _mapping(receipt.get("immutable_inputs"), label="r260 immutable inputs")
    if set(immutable) != {
        "r241_r260_contract",
        "r258_provenance_contract",
        "r195_parent_contract",
        "legacy_execution_ledger",
    }:
        raise R260PrestartGateError("r260 pre-start receipt immutable input set drifted")
    for name, identity in immutable.items():
        item = _mapping(identity, label=f"r260 immutable input {name}")
        if set(item) != {"path", "sha256"} or not isinstance(item["path"], str):
            raise R260PrestartGateError(f"r260 immutable input {name} identity is invalid")
        _valid_sha256(item["sha256"], label=f"r260 immutable input {name}")
    if immutable["r241_r260_contract"]["sha256"] != expected_contract:
        raise R260PrestartGateError("r260 pre-start receipt binds a different r241 contract")
    boundary = _mapping(
        receipt.get("legacy_execution_boundary"), label="r260 legacy execution boundary"
    )
    if boundary != {
        "environment_installed": False,
        "listener_started": False,
        "service_started": False,
        "submission_started": False,
        "training_started": False,
        "unit_installed": False,
        "worker_armed": False,
    }:
        raise R260PrestartGateError("r260 legacy execution boundary is not all false")
    observations = _mapping(receipt.get("host_observations"), label="r260 host observations")
    if set(observations) != {"inzi", "elmo"}:
        raise R260PrestartGateError("r260 pre-start receipt host set drifted")
    for host, expected_units in {
        "inzi": {
            "pokebot-alakazam-new-list-direct-r241.service",
            "pokebot-alakazam-new-list-direct-r241-finalize.service",
            "pokebot-alakazam-new-list-direct-r241-submission-queue.service",
            "pokebot-alakazam-new-list-direct-r241-upload.service",
        },
        "elmo": {"pokebot-r241-elmo-official-r236-remote-worker.service"},
    }.items():
        summary = _mapping(observations[host], label=f"r260 {host} observation")
        identity = _mapping(summary.get("identity"), label=f"r260 {host} observation identity")
        journal = _mapping(summary.get("journal_history"), label=f"r260 {host} journal")
        if (
            summary.get("host") != host
            or set(summary.get("unit_ids") or []) != expected_units
            or summary.get("direct_daemon_inventory")
            != "verified_no_current_or_retained_r241_container"
            or journal.get("readable") is not True
            or journal.get("matching_r241_unit_entries") != 0
            or set(identity) != {"path", "sha256"}
            or not isinstance(identity.get("path"), str)
        ):
            raise R260PrestartGateError(f"r260 {host} observation summary is invalid")
        _valid_sha256(identity["sha256"], label=f"r260 {host} observation")
    if observations["elmo"].get("listener_count") != 0:
        raise R260PrestartGateError("r260 Elmo listener evidence is invalid")
    if observations["inzi"].get("selector_r241_active_reference") is not False:
        raise R260PrestartGateError("r260 Inzi selector evidence is invalid")
    return receipt


def _create_only_json(path: Path | str, payload: Mapping[str, object]) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise R260PrestartGateError(f"gate receipt output may not be a symlink: {raw}")
    parent = raw.parent
    if parent.is_symlink() or not parent.is_dir():
        raise R260PrestartGateError(f"gate receipt parent must be a real directory: {parent}")
    target = parent / raw.name
    encoded = _canonical_json(payload)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
            raise R260PrestartGateError(f"gate receipt already exists with different bytes: {target}")
        if target.stat().st_mode & 0o222:
            raise R260PrestartGateError(f"existing gate receipt is writable: {target}")
        return target.resolve()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != encoded:
                raise R260PrestartGateError(f"gate receipt already exists with different bytes: {target}")
        if target.stat().st_mode & 0o222:
            raise R260PrestartGateError(f"gate receipt publication became writable: {target}")
        return target.resolve()
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r241-contract", type=Path, required=True)
    parser.add_argument("--r258-contract", type=Path, required=True)
    parser.add_argument("--r195-contract", type=Path, required=True)
    parser.add_argument("--legacy-execution-ledger", type=Path, required=True)
    parser.add_argument("--inzi-observation", type=Path, required=True)
    parser.add_argument("--elmo-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(
            r241_contract=args.r241_contract,
            r258_contract=args.r258_contract,
            r195_contract=args.r195_contract,
            legacy_execution_ledger=args.legacy_execution_ledger,
            inzi_observation=args.inzi_observation,
            elmo_observation=args.elmo_observation,
        )
        output = _create_only_json(args.output, receipt)
    except R260PrestartGateError as exc:
        print(f"r260 pre-start supersession gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"path": str(output), "sha256": _sha256_file(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
