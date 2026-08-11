"""Offline immutable R244 handle-scoped ``SearchId`` regression preflight.

The official ``libcg`` numeric SearchId namespace belongs to an ``AgentStart``
handle.  Two distinct two-lane handles may therefore both first return raw ID
zero.  This module validates that precise relationship without launching a
child, loading a DSO, using CUDA, or contacting any external service.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any

from poke_bot.r235_kaggle_phase1_preflight import (
    ROOT,
    R225_CANONICAL_SHA256,
    R225_SCHEMA,
    R236_SCHEMA,
    R246_OWNER_REVISION,
    R235PreflightError,
    _regular_file,
    _validate_r225_r238_contract,
    _validate_r236_contract,
    sha256_file,
    write_once_atomic,
)


RECEIPT_SCHEMA = "poke_bot.r244_handle_scoped_search_id_identity_regression_receipt/v1"
RECEIPT_NAME = "official_libcg_handle_scoped_search_id_identity_regression_receipt"
WITNESS_SCHEMA = "poke_bot.r244_handle_scoped_search_id_identity_probe/v1"
R244_OWNER_REVISION = 244
LANE_COUNT = 2


class R244HandleScopedSearchIdError(RuntimeError):
    """The R244 handle-scoped SearchId contract did not validate."""


class R244HandleScopedSearchIdFailure(R244HandleScopedSearchIdError):
    """A write-once failure receipt was published before raising."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any], path: Path) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)
        self.path = path


@dataclass(frozen=True)
class R244HandleScopedSearchIdInputs:
    """Explicit offline inputs; no package name or process is inferred."""

    witness_path: Path
    receipt_path: Path
    r225_contract_path: Path = ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json"
    r236_contract_path: Path = ROOT / "state/canonical-libcg-r236.json"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R244HandleScopedSearchIdError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise R244HandleScopedSearchIdError(f"{label} must be a JSON object")
    return payload


def _require_exact(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise R244HandleScopedSearchIdError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R244HandleScopedSearchIdError(f"{label} must be a nonnegative integer")
    return int(value)


def _validate_r244_contract(payload: Mapping[str, Any]) -> None:
    # R244 is an additive local regression gate, not a separate authority.
    # First require the whole frozen R235/R238/R240/R242/R244 canonical
    # contract projection, then repeat the R244 fields below with errors that
    # identify this standalone receipt's precise boundary.
    try:
        _validate_r225_r238_contract(payload)
    except R235PreflightError as exc:
        raise R244HandleScopedSearchIdError(
            f"r225 fails the canonical R235/R238/R240/R242/R244 preflight contract: {exc}"
        ) from exc
    _require_exact(payload.get("schema"), R225_SCHEMA, label="r225 schema")
    # The standalone receipt keeps its R244 semantic identity, while its
    # canonical source must be the currently active R246 revision.  R246
    # explicitly retains the R244 handle-scoped correction; accepting a
    # historical r244-shaped source here would disconnect this gate from the
    # active replacement-package contract.
    _require_exact(
        payload.get("owner_decision_revision"),
        R246_OWNER_REVISION,
        label="r225 active owner revision",
    )
    _require_exact(
        payload.get("owner_handle_scoped_search_id_revision"),
        R244_OWNER_REVISION,
        label="r225 R244 handle-scoped SearchId revision",
    )
    relationship = payload.get("relationship_to_existing_work")
    if not isinstance(relationship, Mapping):
        raise R244HandleScopedSearchIdError("r225 relationship must be an object")
    for field in (
        "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229",
        "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries",
    ):
        _require_exact(relationship.get(field), True, label=f"r225 {field}")
    local = payload.get("local_preflight")
    if not isinstance(local, Mapping):
        raise R244HandleScopedSearchIdError("r225 local preflight must be an object")
    exact_local = {
        "required_simulator_search_lane_count": 2,
        "required_internal_agent_start_simulator_search_arena_count_per_child": 2,
        "required_search_begin_call_count_per_ambiguous_mcts_decision": 2,
        "required_distinct_internal_agent_start_handle_identity_count": 2,
        "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
        "per_lane_handle_scoped_search_id_chains_required": True,
    }
    for field, expected in exact_local.items():
        _require_exact(local.get(field), expected, label=f"r225 R244 {field}")


def _validate_witness(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return the exact canonical handle-scoped identity projection."""

    _require_exact(payload.get("schema"), WITNESS_SCHEMA, label="R244 witness schema")
    for field, expected in {
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }.items():
        _require_exact(payload.get(field), expected, label=f"R244 witness {field}")

    raw_handles = payload.get("per_lane_handle_identities")
    if not isinstance(raw_handles, list) or len(raw_handles) != LANE_COUNT:
        raise R244HandleScopedSearchIdError("R244 witness must contain exactly two handle identities")
    handles: list[int | str] = []
    for lane, handle in enumerate(raw_handles):
        if isinstance(handle, bool) or not isinstance(handle, (int, str)):
            raise R244HandleScopedSearchIdError(f"R244 handle identity {lane} is malformed")
        if isinstance(handle, str) and not handle:
            raise R244HandleScopedSearchIdError(f"R244 handle identity {lane} is empty")
        handles.append(handle)
    if len(set(handles)) != LANE_COUNT:
        raise R244HandleScopedSearchIdError("R244 witness lacks two distinct raw handles")

    raw_chains = payload.get("per_lane_search_id_chains")
    if not isinstance(raw_chains, list) or len(raw_chains) != LANE_COUNT:
        raise R244HandleScopedSearchIdError("R244 witness must contain exactly two SearchId chains")
    chains: list[list[int]] = []
    for lane, raw_chain in enumerate(raw_chains):
        if not isinstance(raw_chain, list) or not raw_chain:
            raise R244HandleScopedSearchIdError(f"R244 SearchId chain {lane} is empty")
        chains.append(
            [
                _nonnegative_int(value, label=f"R244 SearchId lane {lane} index {index}")
                for index, value in enumerate(raw_chain)
            ]
        )
    first_ids = [chain[0] for chain in chains]
    composites = [
        {"lane_id": lane, "handle_identity": handles[lane], "first_search_id": first_ids[lane]}
        for lane in range(LANE_COUNT)
    ]
    if len({(handles[lane], first_ids[lane]) for lane in range(LANE_COUNT)}) != LANE_COUNT:
        raise R244HandleScopedSearchIdError(
            "R244 witness lacks two distinct handle-scoped first SearchId composites"
        )
    _require_exact(
        payload.get("per_lane_first_search_ids"),
        first_ids,
        label="R244 per-lane first SearchIds",
    )
    _require_exact(
        payload.get("handle_scoped_first_search_id_composite_states"),
        composites,
        label="R244 handle-scoped first SearchId composite states",
    )
    return {
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "per_lane_handle_identities": handles,
        "per_lane_search_id_chains": chains,
        "per_lane_first_search_ids": first_ids,
        "handle_scoped_first_search_id_composite_states": composites,
        "per_lane_handle_first_search_id_composites": composites,
        "distinct_handle_identity_count": 2,
        "distinct_handle_scoped_first_search_id_composite_state_count": 2,
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }


def _assert_regression_rejections(valid_witness: Mapping[str, Any]) -> None:
    duplicate_handle = copy.deepcopy(dict(valid_witness))
    duplicate_handle["per_lane_handle_identities"][1] = duplicate_handle[
        "per_lane_handle_identities"
    ][0]
    try:
        _validate_witness(duplicate_handle)
    except R244HandleScopedSearchIdError:
        pass
    else:  # pragma: no cover - the invariant is tested through the public API
        raise R244HandleScopedSearchIdError("R244 duplicate-handle regression was accepted")

    duplicate_composite = copy.deepcopy(dict(valid_witness))
    duplicate_composite["handle_scoped_first_search_id_composite_states"][1] = copy.deepcopy(
        duplicate_composite["handle_scoped_first_search_id_composite_states"][0]
    )
    try:
        _validate_witness(duplicate_composite)
    except R244HandleScopedSearchIdError:
        pass
    else:  # pragma: no cover - the invariant is tested through the public API
        raise R244HandleScopedSearchIdError("R244 duplicate-composite regression was accepted")


def _failure_receipt(
    *, inputs: R244HandleScopedSearchIdInputs, error: Exception
) -> dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "receipt_name": RECEIPT_NAME,
        "status": "failed",
        "passed": False,
        "immutable": True,
        "write_once": True,
        "execution_mode": "offline_structural_regression",
        "expected_canonical_r225_contract_sha256": R225_CANONICAL_SHA256,
        "requested_witness_path": str(inputs.witness_path),
        "failure": {"type": type(error).__name__, "message": str(error)},
        "child_started": False,
        "cuda_initialized": False,
        "kaggle_api_called": False,
        "kaggle_upload_used": False,
        "managed_service_changed": False,
    }


def run_r244_handle_scoped_search_id_preflight(
    *, inputs: R244HandleScopedSearchIdInputs
) -> dict[str, object]:
    """Validate one offline R244 witness and atomically write its receipt."""

    try:
        witness_path = _regular_file(inputs.witness_path, label="R244 witness")
        r225_path = _regular_file(inputs.r225_contract_path, label="r225 contract")
        r236_path = _regular_file(inputs.r236_contract_path, label="r236 contract")
        r225_sha = sha256_file(r225_path)
        _require_exact(
            r225_sha,
            R225_CANONICAL_SHA256,
            label="final canonical r225 contract digest",
        )
        r225_payload = _read_json_object(r225_path, label="r225 contract")
        _validate_r244_contract(r225_payload)
        r236_payload = _read_json_object(r236_path, label="r236 contract")
        if r236_payload.get("schema") != R236_SCHEMA:
            raise R244HandleScopedSearchIdError("r236 schema mismatch")
        r236 = _validate_r236_contract(r236_payload)
        witness = _read_json_object(witness_path, label="R244 witness")
        projection = _validate_witness(witness)
        _assert_regression_rejections(witness)
        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "receipt_name": RECEIPT_NAME,
            "status": "passed",
            "passed": True,
            "immutable": True,
            "write_once": True,
            "execution_mode": "offline_structural_regression",
            "r225_contract_sha256": r225_sha,
            "expected_canonical_r225_contract_sha256": R225_CANONICAL_SHA256,
            "canonical_libcg_contract_sha256": sha256_file(r236_path),
            "linux_x86_64_libcg_sha256": r236["linux_sha256"],
            "linux_x86_64_libcg_size_bytes": r236["linux_size_bytes"],
            "r225_active_owner_revision": R246_OWNER_REVISION,
            "r244_owner_revision": R244_OWNER_REVISION,
            "official_libcg_handle_scoped_search_id_identity_regression_passed": True,
            **projection,
            "same_raw_first_search_id_on_distinct_handles_accepted": True,
            "duplicate_handle_identity_rejected": True,
            "duplicate_handle_scoped_first_search_id_composite_rejected": True,
            "child_started": False,
            "cuda_initialized": False,
            "kaggle_api_called": False,
            "kaggle_upload_used": False,
            "managed_service_changed": False,
        }
        write_once_atomic(Path(inputs.receipt_path), receipt)
        return receipt
    except Exception as error:
        if isinstance(error, R244HandleScopedSearchIdFailure):
            raise
        failure = _failure_receipt(inputs=inputs, error=error)
        try:
            write_once_atomic(Path(inputs.receipt_path), failure)
        except Exception:
            raise error
        raise R244HandleScopedSearchIdFailure(str(error), receipt=failure, path=Path(inputs.receipt_path)) from error
