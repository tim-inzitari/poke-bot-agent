#!/usr/bin/env python3
"""Verify and create-only copy the sealed Prize-plan-v2 target set to Bert.

The default action is a read-only dry run.  ``--execute`` uses exactly four
independent object lanes.  It transfers only the portable aggregate inventory:
compact target rows, their schemas/receipts, the public Phi fit, the analytic
target transform, the canonical contract, and the complete-action *manifest*.
Raw replay ZIPs, feature packs, and complete-action JSONL payloads are never
eligible transfer objects.

Private resumable partials are prefix-hashed against Elmo before append.  A
partial becomes a final file only after full SHA-256 and size verification and
create-only hard-link promotion.  Existing exact finals are reusable; every
conflict fails closed.  Execution reserves the remaining object bytes plus a
bounded metadata allowance while preserving a 20 GiB Bert free-space floor.
This command never starts training or changes a runtime/service.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from poke_bot import prize_plan_targets_v2 as targets  # noqa: E402
from scripts import transfer_alakazam_action_critic_targets as transport  # noqa: E402


TRANSFER_PLAN_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_transfer_plan/v1"
FILE_RECEIPT_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_file_transfer_receipt/v1"
TARGET_VIEW_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_view/v1"
COMPLETION_SCHEMA = "poke_bot.alakazam_prize_plan_v2_target_transfer_completion/v1"
TARGET_VALUE_TRANSFORM_SCHEMA = "poke_bot.alakazam_prize_plan_target_value_transform/v2"

SOURCE_HOST = "elmo"
LANE_COUNT = 4
OWNER_GOAL_REVISION = 23
REQUIRED_AUTHORITY = "revision_23_prize_plan_v2_h3_actor_canary"
DEFAULT_DISK_FLOOR_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_METADATA_RESERVE_BYTES = 4 * 1024 * 1024
DEFAULT_DESTINATION_ROOT = Path(
    "/Users/tsinzitari/Documents/poke-agent-critic-bootstrap/"
    "recent20-prize-plan-v2-r23"
)
DEFAULT_CONTRACT_PATH = REPOSITORY_ROOT / "goals/alakazam-elmo-rule-derivative/contract.json"
DEFAULT_SOURCE_ROOT = (
    "/mnt/Main/main/poke-bot-agent/outputs/experiments/"
    "alakazam-prize-plan-v2-targets-r23"
)

WINDOW_DAYS = targets.WINDOW_DAYS
SPLIT_BY_DAY = targets.SPLIT_BY_DAY

PORTABLE_ROLE_ALLOWLIST = frozenset(
    {
        "goal_contract",
        "complete_action_overlay_manifest",
        "phi_fit_input_manifest",
        "phi_table",
        "phi_fit_manifest",
        "phi_fit_receipt",
        "phi_fit_artifact",
        "target_value_transform",
        "target_schema",
        "target_shard",
        "target_day_manifest",
        "target_day_receipt",
    }
)
PLAN_ROLE_ALLOWLIST = PORTABLE_ROLE_ALLOWLIST | {
    "target_set_manifest",
    "target_set_receipt",
}

TargetTransferError = transport.TargetTransferError
FileIdentity = transport.FileIdentity
SourceReader = transport.SourceReader
LocalSourceReader = transport.LocalSourceReader
SSHSourceReader = transport.SSHSourceReader


def canonical_bytes(value: Any) -> bytes:
    return transport.canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return transport.sha256_bytes(value)


def sha256_file(path: Path | str, *, limit: int | None = None) -> str:
    return transport.sha256_file(path, limit=limit)


def _require(condition: bool, message: str) -> None:
    transport._require(condition, message)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    return transport._mapping(value, label=label)


def _rows(value: object, *, label: str) -> list[Any]:
    return transport._rows(value, label=label)


def _sha256(value: object, *, label: str) -> str:
    return transport._sha256(value, label=label)


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    return transport._nonnegative_int(value, label=label, allow_zero=allow_zero)


def _safe_relative(value: object, *, label: str) -> str:
    return transport._safe_relative(value, label=label)


def _read_json(
    source: SourceReader,
    path: str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str, int, bytes]:
    return transport._read_json(
        source, path, label=label, expected_sha256=expected_sha256
    )


def _source_member(root: str, relative: object, *, label: str) -> str:
    return transport._source_member(root, relative, label=label)


def _destination_member(root: Path, relative: object, *, label: str) -> Path:
    return transport._destination_member(root, relative, label=label)


def _portable_child(prefix: str, child: object, *, label: str) -> str:
    child_relative = _safe_relative(child, label=label)
    return _safe_relative(
        str(PurePosixPath(prefix).joinpath(*PurePosixPath(child_relative).parts)),
        label=label,
    )


def _content_addressed_json(
    source: SourceReader,
    path: str,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[dict[str, Any], str, int, bytes]:
    result = _read_json(
        source, path, label=label, expected_sha256=expected_sha256
    )
    transport._assert_content_addressed(path, result[3], label=label)
    return result


def _load_current_contract(
    path: Path | str, *, expected_sha256: str
) -> tuple[Path, Mapping[str, Any], str, bytes]:
    expected = _sha256(expected_sha256, label="expected current contract SHA-256")
    identity = transport._regular_local_identity(path, label="current canonical contract")
    if identity.sha256 != expected:
        raise TargetTransferError("current canonical contract SHA-256 mismatch")
    try:
        contract_path, contract, actual = targets._load_goal_contract(
            path, expected_sha256=expected
        )
    except targets.PrizePlanTargetError as exc:
        raise TargetTransferError(str(exc)) from exc
    return contract_path, contract, actual, contract_path.read_bytes()


def _inventory_map(
    manifest: Mapping[str, Any], *, root: str
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    by_path: dict[str, Mapping[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for raw in _rows(manifest.get("portable_objects"), label="portable object inventory"):
        item = _mapping(raw, label="portable object")
        relative = _safe_relative(item.get("relative_path"), label="portable object path")
        if relative in by_path:
            raise TargetTransferError("portable object inventory duplicates a path")
        role = str(item.get("role") or "")
        if role not in PORTABLE_ROLE_ALLOWLIST:
            raise TargetTransferError(f"portable object has ineligible role: {role!r}")
        if relative.lower().endswith((".zip", ".partial", ".part")):
            raise TargetTransferError("raw ZIP or private partial entered portable inventory")
        if role in {"target_shard", "target_schema", "target_day_manifest", "target_day_receipt"}:
            day = str(item.get("utc_day") or "")
            split = str(item.get("split") or "")
            if day not in SPLIT_BY_DAY or split != SPLIT_BY_DAY[day]:
                raise TargetTransferError("portable day object day/split drifted")
            if not relative.startswith(f"days/{day}/"):
                raise TargetTransferError("portable day object escaped its day root")
        elif "utc_day" in item or "split" in item:
            raise TargetTransferError("non-day portable object carries day/split fields")
        digest = _sha256(item.get("sha256"), label="portable object SHA-256")
        size = _positive_int(item.get("size_bytes"), label="portable object size")
        source_path = _source_member(root, relative, label="portable source object")
        entry: dict[str, Any] = {
            "source_path": source_path,
            "destination_relative": relative,
            "sha256": digest,
            "size_bytes": size,
            "role": role,
        }
        if role.startswith("target_") and role != "target_value_transform":
            entry["utc_day"] = str(item["utc_day"])
            entry["split"] = str(item["split"])
        by_path[relative] = item
        entries.append(entry)
    if len(entries) < LANE_COUNT:
        raise TargetTransferError("portable target set has too few objects for four streams")
    return by_path, entries


def _inventory_identity(
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    relative: object,
    expected_role: str,
    expected_sha256: object,
    label: str,
) -> Mapping[str, Any]:
    path = _safe_relative(relative, label=f"{label} path")
    item = inventory.get(path)
    if item is None:
        raise TargetTransferError(f"{label} is absent from portable inventory")
    if item.get("role") != expected_role:
        raise TargetTransferError(f"{label} portable role drifted")
    if _sha256(item.get("sha256"), label=f"{label} inventory SHA-256") != _sha256(
        expected_sha256, label=f"{label} bound SHA-256"
    ):
        raise TargetTransferError(f"{label} portable SHA-256 drifted")
    return item


def _exact_day_map(rows: object, *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _rows(rows, label=label):
        item = _mapping(raw, label=f"{label} item")
        day = str(item.get("utc_day") or "")
        if day not in SPLIT_BY_DAY or day in result:
            raise TargetTransferError(f"{label} day inventory drifted")
        result[day] = item
    if set(result) != set(WINDOW_DAYS):
        raise TargetTransferError(f"{label} is not the exact recent-20 window")
    return result


def _validate_phi_graph(
    *,
    source: SourceReader,
    root: str,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    contract_sha: str,
    contract_revision: int,
) -> dict[str, Any]:
    phi = _mapping(manifest.get("phi_fit"), label="aggregate Phi fit")
    if phi.get("fit_scope") != "sealed_train_split_only":
        raise TargetTransferError("Phi fit is not sealed train-split-only")
    phi_root = _safe_relative(phi.get("portable_root"), label="portable Phi root")
    if phi_root != "bindings/phi_fit":
        raise TargetTransferError("portable Phi root drifted")
    fit_manifest_binding = _mapping(phi.get("fit_manifest"), label="Phi fit manifest binding")
    fit_receipt_binding = _mapping(phi.get("fit_receipt"), label="Phi fit receipt binding")
    table_binding = _mapping(phi.get("frozen_phi_table"), label="Phi table binding")
    fit_manifest_item = _inventory_identity(
        inventory,
        relative=fit_manifest_binding.get("path"),
        expected_role="phi_fit_manifest",
        expected_sha256=fit_manifest_binding.get("sha256"),
        label="Phi fit manifest",
    )
    fit_receipt_item = _inventory_identity(
        inventory,
        relative=fit_receipt_binding.get("path"),
        expected_role="phi_fit_receipt",
        expected_sha256=fit_receipt_binding.get("sha256"),
        label="Phi fit receipt",
    )
    table_item = _inventory_identity(
        inventory,
        relative=table_binding.get("path"),
        expected_role="phi_table",
        expected_sha256=table_binding.get("sha256"),
        label="frozen Phi table",
    )
    if table_binding.get("schema") != targets.PRIZE_PLAN_POTENTIAL_SCHEMA:
        raise TargetTransferError("frozen Phi table schema drifted")
    fit_manifest, fit_manifest_sha, _, _ = _read_json(
        source,
        _source_member(root, fit_manifest_binding["path"], label="Phi fit manifest source"),
        label="Phi fit manifest",
        expected_sha256=str(fit_manifest_binding["sha256"]),
    )
    goal = _mapping(fit_manifest.get("goal_contract"), label="Phi fit contract binding")
    if (
        fit_manifest.get("schema") != targets.PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA
        or fit_manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or fit_manifest.get("fit_scope") != "sealed_train_split_only"
        or goal.get("sha256") != contract_sha
        or goal.get("goal_revision") != contract_revision
        or goal.get("required_authority") != REQUIRED_AUTHORITY
        or goal.get("semantic_owner_goal_revision") != OWNER_GOAL_REVISION
    ):
        raise TargetTransferError("Phi fit manifest authority/split binding drifted")
    fit_input = _mapping(fit_manifest.get("fit_input_manifest"), label="Phi fit input binding")
    fit_input_relative = _portable_child(
        phi_root, fit_input.get("path"), label="portable Phi fit input path"
    )
    fit_input_item = _inventory_identity(
        inventory,
        relative=fit_input_relative,
        expected_role="phi_fit_input_manifest",
        expected_sha256=fit_input.get("sha256"),
        label="Phi fit input manifest",
    )
    if _positive_int(fit_input.get("size_bytes"), label="Phi fit input size") != int(
        fit_input_item["size_bytes"]
    ):
        raise TargetTransferError("Phi fit input size drifted")
    table = _mapping(fit_manifest.get("frozen_phi_table"), label="Phi manifest table binding")
    table_relative = _portable_child(phi_root, table.get("path"), label="portable Phi table path")
    if (
        table_relative != table_binding.get("path")
        or table.get("sha256") != table_binding.get("sha256")
        or table.get("size_bytes") != table_item.get("size_bytes")
        or fit_input.get("sha256") != phi.get("fit_input_manifest_sha256")
        or fit_manifest.get("fit_configuration_sha256")
        != phi.get("fit_configuration_sha256")
    ):
        raise TargetTransferError("aggregate/Phi manifest identity graph drifted")
    try:
        expected_config_sha = targets.canonical_sha256(fit_manifest.get("fit_configuration"))
    except (TypeError, ValueError) as exc:
        raise TargetTransferError("Phi fit configuration is not canonical JSON") from exc
    if expected_config_sha != fit_manifest.get("fit_configuration_sha256"):
        raise TargetTransferError("Phi fit configuration SHA-256 drifted")
    fit_receipt, fit_receipt_sha, _, _ = _read_json(
        source,
        _source_member(root, fit_receipt_binding["path"], label="Phi fit receipt source"),
        label="Phi fit receipt",
        expected_sha256=str(fit_receipt_binding["sha256"]),
    )
    if (
        fit_receipt.get("schema") != targets.PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA
        or fit_receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or fit_receipt.get("goal_contract_sha256") != contract_sha
        or fit_receipt.get("goal_contract_goal_revision") != contract_revision
        or fit_receipt.get("required_authority") != REQUIRED_AUTHORITY
        or fit_receipt.get("phi_fit_manifest_sha256") != fit_manifest_sha
        or fit_receipt.get("fit_input_manifest_sha256") != fit_input.get("sha256")
        or fit_receipt.get("fit_configuration_sha256")
        != fit_manifest.get("fit_configuration_sha256")
        or fit_receipt.get("frozen_phi_table_sha256") != table.get("sha256")
        or fit_receipt.get("fit_scope") != "sealed_train_split_only"
        or fit_receipt.get("validation_evaluation_or_runtime_refit") is not False
        or fit_receipt.get("terminal_z_used_only_for_train_phi_fit") is not True
    ):
        raise TargetTransferError("Phi fit receipt binding drifted")
    return {
        "fit_manifest_relative": str(fit_manifest_binding["path"]),
        "fit_manifest_sha256": fit_manifest_sha,
        "fit_receipt_relative": str(fit_receipt_binding["path"]),
        "fit_receipt_sha256": fit_receipt_sha,
        "fit_input_manifest_relative": fit_input_relative,
        "fit_input_manifest_sha256": str(fit_input["sha256"]),
        "fit_configuration_sha256": str(fit_manifest["fit_configuration_sha256"]),
        "frozen_phi_table_relative": table_relative,
        "frozen_phi_table_sha256": str(table["sha256"]),
        "portable_object_sha256s": {
            "fit_manifest": fit_manifest_item["sha256"],
            "fit_receipt": fit_receipt_item["sha256"],
            "fit_input": fit_input_item["sha256"],
            "table": table_item["sha256"],
        },
    }


def _validate_transform(
    *,
    source: SourceReader,
    root: str,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    binding = _mapping(
        manifest.get("target_value_transform"), label="target value transform binding"
    )
    if binding.get("schema") != TARGET_VALUE_TRANSFORM_SCHEMA:
        raise TargetTransferError("target value transform schema drifted")
    _inventory_identity(
        inventory,
        relative=binding.get("path"),
        expected_role="target_value_transform",
        expected_sha256=binding.get("sha256"),
        label="target value transform",
    )
    document, digest, _, _ = _read_json(
        source,
        _source_member(root, binding["path"], label="target transform source"),
        label="target value transform",
        expected_sha256=str(binding["sha256"]),
    )
    if (
        document.get("schema") != TARGET_VALUE_TRANSFORM_SCHEMA
        or document.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or document.get("formula")
        != "model_target_value=raw_return_value/(1+gamma**h)"
        or document.get("gamma") != 1.0
        or document.get("horizons") != [1, 3, 6, 12]
        or document.get("expected_model_target_range") != [-1.0, 1.0]
        or document.get("clipping") is not False
        or document.get("data_dependent_train_fit") is not False
        or document.get("actor_advantage_scaling")
        != "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform"
    ):
        raise TargetTransferError("target value transform semantics drifted")
    if document.get("source_target_shards") != manifest.get("all_20_target_shards"):
        raise TargetTransferError("target transform source-shard binding drifted")
    return {
        "relative_path": str(binding["path"]),
        "sha256": digest,
        "gamma": 1.0,
        "formula": str(document["formula"]),
        "data_dependent_train_fit": False,
    }


def _validate_target_days(
    *,
    source: SourceReader,
    root: str,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    contract_sha: str,
    contract_revision: int,
    phi_graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    days = _exact_day_map(manifest.get("target_days"), label="target days")
    raw_days = _exact_day_map(
        manifest.get("all_20_raw_episode_zip_sha256s"), label="raw ZIP provenance"
    )
    overlay_days = _exact_day_map(
        manifest.get("all_20_complete_action_overlay_sha256s"),
        label="complete-action overlay provenance",
    )
    shard_days = _exact_day_map(
        manifest.get("all_20_target_shards"), label="aggregate target shards"
    )
    output: list[dict[str, Any]] = []
    for day in WINDOW_DAYS:
        split = SPLIT_BY_DAY[day]
        item = days[day]
        if item.get("split") != split or item.get("day_artifact_root") != f"days/{day}":
            raise TargetTransferError(f"target day root/split drifted for {day}")
        day_manifest_binding = _mapping(item.get("day_manifest"), label="day manifest binding")
        day_receipt_binding = _mapping(item.get("day_receipt"), label="day receipt binding")
        schema_binding = _mapping(item.get("target_schema"), label="day schema binding")
        shard_binding = _mapping(item.get("target_shard"), label="day shard binding")
        for binding, role, label in (
            (day_manifest_binding, "target_day_manifest", "day manifest"),
            (day_receipt_binding, "target_day_receipt", "day receipt"),
            (schema_binding, "target_schema", "day schema"),
            (shard_binding, "target_shard", "day shard"),
        ):
            _inventory_identity(
                inventory,
                relative=binding.get("path"),
                expected_role=role,
                expected_sha256=binding.get("sha256"),
                label=f"{label} {day}",
            )
        day_manifest, day_manifest_sha, _, _ = _read_json(
            source,
            _source_member(root, day_manifest_binding["path"], label="day manifest source"),
            label=f"target day manifest {day}",
            expected_sha256=str(day_manifest_binding["sha256"]),
        )
        day_receipt, day_receipt_sha, _, _ = _read_json(
            source,
            _source_member(root, day_receipt_binding["path"], label="day receipt source"),
            label=f"target day receipt {day}",
            expected_sha256=str(day_receipt_binding["sha256"]),
        )
        goal = _mapping(day_manifest.get("goal_contract"), label="day contract binding")
        phi = _mapping(day_manifest.get("phi_fit_manifest"), label="day Phi binding")
        raw = _mapping(day_manifest.get("raw_episode_zip"), label="day raw ZIP provenance")
        overlay = _mapping(
            day_manifest.get("complete_action_overlay"), label="day overlay provenance"
        )
        shard = _mapping(day_manifest.get("target_shard"), label="day target shard")
        schema = _mapping(day_manifest.get("target_schema"), label="day target schema")
        day_root = f"days/{day}"
        shard_relative = _portable_child(day_root, shard.get("path"), label="day shard path")
        schema_relative = _portable_child(day_root, schema.get("path"), label="day schema path")
        expected_transform = {
            "formula": "model_target_value=raw_return_value/(1+gamma**h)",
            "gamma": 1.0,
            "data_dependent_train_fit": False,
            "clipping": False,
            "expected_model_target_range": [-1.0, 1.0],
            "actor_advantage_scaling": (
                "separate_train_split_only_frozen_sidecar_or_actor_receipt_"
                "not_this_target_transform"
            ),
        }
        aggregate_raw = raw_days[day]
        aggregate_overlay = overlay_days[day]
        aggregate_shard = shard_days[day]
        if (
            day_manifest.get("schema") != targets.PRIZE_PLAN_DAY_MANIFEST_SCHEMA
            or day_manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION
            or day_manifest.get("utc_day") != day
            or day_manifest.get("split") != split
            or day_manifest.get("gamma") != 1.0
            or day_manifest.get("target_value_transform") != expected_transform
            or goal.get("sha256") != contract_sha
            or goal.get("goal_revision") != contract_revision
            or goal.get("required_authority") != REQUIRED_AUTHORITY
            or goal.get("semantic_owner_goal_revision") != OWNER_GOAL_REVISION
            or phi.get("sha256") != phi_graph["fit_manifest_sha256"]
            or phi.get("frozen_phi_table_sha256")
            != phi_graph["frozen_phi_table_sha256"]
            or phi.get("fit_input_manifest_sha256")
            != phi_graph["fit_input_manifest_sha256"]
            or phi.get("fit_configuration_sha256")
            != phi_graph["fit_configuration_sha256"]
            or raw.get("source_archive_sha256_verified") is not True
            or raw.get("sha256") != aggregate_raw.get("sha256")
            or raw.get("size_bytes") != aggregate_raw.get("size_bytes")
            or overlay.get("sha256") != aggregate_overlay.get("sha256")
            or overlay.get("size_bytes") != aggregate_overlay.get("size_bytes")
            or shard_relative != shard_binding.get("path")
            or schema_relative != schema_binding.get("path")
            or shard.get("sha256") != shard_binding.get("sha256")
            or shard.get("size_bytes") != shard_binding.get("size_bytes")
            or shard.get("row_count") != shard_binding.get("row_count")
            or {key: shard_binding.get(key) for key in ("sha256", "size_bytes", "row_count")}
            != {key: aggregate_shard.get(key) for key in ("sha256", "size_bytes", "row_count")}
            or aggregate_shard.get("split") != split
        ):
            raise TargetTransferError(f"target day identity graph drifted for {day}")
        if (
            day_receipt.get("schema") != targets.PRIZE_PLAN_DAY_RECEIPT_SCHEMA
            or day_receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
            or day_receipt.get("goal_contract_sha256") != contract_sha
            or day_receipt.get("goal_contract_goal_revision") != contract_revision
            or day_receipt.get("required_authority") != REQUIRED_AUTHORITY
            or day_receipt.get("day_manifest_sha256") != day_manifest_sha
            or day_receipt.get("phi_fit_manifest_sha256")
            != phi_graph["fit_manifest_sha256"]
            or day_receipt.get("frozen_phi_table_sha256")
            != phi_graph["frozen_phi_table_sha256"]
            or day_receipt.get("gamma") != 1.0
            or day_receipt.get("target_value_transform") != expected_transform
            or day_receipt.get("complete_action_overlay_sha256") != overlay.get("sha256")
            or day_receipt.get("raw_episode_zip_sha256") != raw.get("sha256")
            or day_receipt.get("target_shard_sha256") != shard.get("sha256")
            or day_receipt.get("target_shard_size_bytes") != shard.get("size_bytes")
            or day_receipt.get("target_row_count") != shard.get("row_count")
            or day_receipt.get("coverage") != day_manifest.get("coverage")
        ):
            raise TargetTransferError(f"target day receipt binding drifted for {day}")
        output.append(
            {
                "utc_day": day,
                "split": split,
                "day_artifact_root_relative": day_root,
                "day_manifest_relative": str(day_manifest_binding["path"]),
                "day_manifest_sha256": day_manifest_sha,
                "day_receipt_relative": str(day_receipt_binding["path"]),
                "day_receipt_sha256": day_receipt_sha,
                "target_schema_relative": str(schema_binding["path"]),
                "target_schema_sha256": str(schema_binding["sha256"]),
                "target_shard_relative": str(shard_binding["path"]),
                "target_shard_sha256": str(shard_binding["sha256"]),
                "target_shard_size_bytes": int(shard_binding["size_bytes"]),
                "target_row_count": int(shard_binding["row_count"]),
                "raw_episode_zip_sha256": str(raw["sha256"]),
                "raw_episode_zip_size_bytes": int(raw["size_bytes"]),
                "complete_action_overlay_sha256": str(overlay["sha256"]),
            }
        )
    return output


def build_prize_plan_v2_transfer_plan(
    *,
    source: SourceReader,
    source_root: Path | str,
    destination_root: Path | str,
    target_set_manifest_relative: str,
    expected_target_set_manifest_sha256: str,
    target_set_receipt_relative: str,
    expected_target_set_receipt_sha256: str,
    local_contract_path: Path | str,
    expected_contract_sha256: str,
    disk_floor_bytes: int = DEFAULT_DISK_FLOOR_BYTES,
    metadata_reserve_bytes: int = DEFAULT_METADATA_RESERVE_BYTES,
    test_allow_non_elmo_source: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate a sealed portable r23 target set and make its four-lane plan."""

    if source.host != SOURCE_HOST and not test_allow_non_elmo_source:
        raise TargetTransferError("production Prize-plan-v2 source host must be elmo")
    root = transport._safe_source_root(source_root, label="source target-set root")
    manifest_relative = _safe_relative(
        target_set_manifest_relative, label="target-set manifest relative path"
    )
    receipt_relative = _safe_relative(
        target_set_receipt_relative, label="target-set receipt relative path"
    )
    expected_manifest_sha = _sha256(
        expected_target_set_manifest_sha256, label="expected target-set manifest SHA-256"
    )
    expected_receipt_sha = _sha256(
        expected_target_set_receipt_sha256, label="expected target-set receipt SHA-256"
    )
    contract_path, contract, contract_sha, local_contract_body = _load_current_contract(
        local_contract_path, expected_sha256=expected_contract_sha256
    )
    contract_revision = _positive_int(
        contract.get("goal_revision"), label="current contract goal revision"
    )
    if contract_revision < OWNER_GOAL_REVISION:
        raise TargetTransferError("current contract predates r23")
    manifest_path = _source_member(root, manifest_relative, label="target-set manifest source")
    receipt_path = _source_member(root, receipt_relative, label="target-set receipt source")
    manifest, manifest_sha, manifest_size, manifest_body = _content_addressed_json(
        source,
        manifest_path,
        label="Prize-plan-v2 target-set manifest",
        expected_sha256=expected_manifest_sha,
    )
    receipt, receipt_sha, receipt_size, receipt_body = _content_addressed_json(
        source,
        receipt_path,
        label="Prize-plan-v2 target-set receipt",
        expected_sha256=expected_receipt_sha,
    )
    if (
        manifest.get("schema") != targets.PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA
        or manifest.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or manifest.get("goal_contract_goal_revision") != contract_revision
        or manifest.get("required_authority") != REQUIRED_AUTHORITY
    ):
        raise TargetTransferError("target-set manifest schema/authority drifted")
    publication = _mapping(manifest.get("publication"), label="target-set publication")
    information = _mapping(manifest.get("information_boundary"), label="information boundary")
    if (
        publication.get("create_only") is not True
        or publication.get("atomic_root_no_replace") is not True
        or publication.get("portable_relative_paths_only") is not True
        or information.get("raw_zip_or_feature_or_complete_action_overlay_payload_copied")
        is not False
        or information.get(
            "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed"
        )
        is not False
        or information.get("terminal_z_is_direct_plan_target_or_actor_term") is not False
        or manifest.get("whole_day_episode_and_group_split_disjoint") is not True
    ):
        raise TargetTransferError("target-set publication/information boundary drifted")
    if list(manifest.get("source_days") or []) != list(WINDOW_DAYS):
        raise TargetTransferError("target-set source-day order drifted")
    split_days = _mapping(manifest.get("split_days"), label="target-set split days")
    expected_splits = {
        split: [day for day in WINDOW_DAYS if SPLIT_BY_DAY[day] == split]
        for split in ("train", "validation", "evaluation")
    }
    if {key: list(split_days.get(key) or []) for key in expected_splits} != expected_splits:
        raise TargetTransferError("target-set split days drifted")
    inventory, entries = _inventory_map(manifest, root=root)
    goal_binding = _mapping(manifest.get("goal_contract"), label="target-set contract binding")
    goal_item = _inventory_identity(
        inventory,
        relative=goal_binding.get("path"),
        expected_role="goal_contract",
        expected_sha256=goal_binding.get("sha256"),
        label="portable goal contract",
    )
    if (
        goal_binding.get("sha256") != contract_sha
        or goal_binding.get("goal_revision") != contract_revision
        or goal_binding.get("required_authority") != REQUIRED_AUTHORITY
        or goal_binding.get("semantic_owner_goal_revision") != OWNER_GOAL_REVISION
    ):
        raise TargetTransferError("portable goal contract binding drifted")
    source_contract_path = _source_member(
        root, goal_binding["path"], label="portable goal contract source"
    )
    source_contract_body = source.read_bytes(source_contract_path)
    if source_contract_body != local_contract_body or sha256_bytes(source_contract_body) != contract_sha:
        raise TargetTransferError("portable goal contract is not current canonical bytes")
    overlay_binding = _mapping(
        manifest.get("complete_action_overlay_manifest"),
        label="complete-action overlay manifest binding",
    )
    _inventory_identity(
        inventory,
        relative=overlay_binding.get("path"),
        expected_role="complete_action_overlay_manifest",
        expected_sha256=overlay_binding.get("sha256"),
        label="complete-action overlay manifest",
    )
    if overlay_binding.get("schema") != targets.COMPLETE_ACTION_OVERLAY_SCHEMA.replace(
        "complete_action_overlay/v1", "overlay_manifest/v1"
    ):
        # The explicit literal avoids granting payload eligibility merely due
        # to a similarly named row schema.
        if overlay_binding.get("schema") != "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1":
            raise TargetTransferError("complete-action overlay manifest schema drifted")
    overlay_doc, overlay_sha, _, _ = _read_json(
        source,
        _source_member(root, overlay_binding["path"], label="overlay manifest source"),
        label="complete-action overlay manifest",
        expected_sha256=str(overlay_binding["sha256"]),
    )
    if overlay_doc.get("schema") != "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1":
        raise TargetTransferError("foreign complete-action overlay manifest")
    phi_graph = _validate_phi_graph(
        source=source,
        root=root,
        manifest=manifest,
        inventory=inventory,
        contract_sha=contract_sha,
        contract_revision=contract_revision,
    )
    transform = _validate_transform(
        source=source, root=root, manifest=manifest, inventory=inventory
    )
    target_days = _validate_target_days(
        source=source,
        root=root,
        manifest=manifest,
        inventory=inventory,
        contract_sha=contract_sha,
        contract_revision=contract_revision,
        phi_graph=phi_graph,
    )
    if (
        receipt.get("schema") != targets.PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA
        or receipt.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or receipt.get("goal_contract_sha256") != contract_sha
        or receipt.get("goal_contract_goal_revision") != contract_revision
        or receipt.get("required_authority") != REQUIRED_AUTHORITY
        or receipt.get("target_set_manifest_path") != manifest_relative
        or receipt.get("target_set_manifest_sha256") != manifest_sha
        or receipt.get("phi_fit_manifest_sha256") != phi_graph["fit_manifest_sha256"]
        or receipt.get("phi_fit_receipt_sha256") != phi_graph["fit_receipt_sha256"]
        or receipt.get("frozen_phi_table_sha256") != phi_graph["frozen_phi_table_sha256"]
        or receipt.get("target_value_transform_sha256") != transform["sha256"]
        or receipt.get("complete_action_overlay_manifest_sha256") != overlay_sha
        or receipt.get("day_count") != len(WINDOW_DAYS)
        or receipt.get("coverage") != manifest.get("coverage")
        or receipt.get("whole_day_episode_and_group_split_disjoint") is not True
        or receipt.get("portable_object_count") != len(inventory)
        or receipt.get("raw_zip_or_feature_or_complete_action_overlay_payload_copied")
        is not False
        or receipt.get("terminal_z_used_as_direct_plan_target_or_actor_term") is not False
        or receipt.get("atomic_root_no_replace") is not True
        or not isinstance(receipt.get("sealed_at_unix_seconds"), (int, float))
    ):
        raise TargetTransferError("target-set materialization receipt binding drifted")
    entries.extend(
        [
            {
                "source_path": manifest_path,
                "destination_relative": manifest_relative,
                "sha256": manifest_sha,
                "size_bytes": manifest_size,
                "role": "target_set_manifest",
            },
            {
                "source_path": receipt_path,
                "destination_relative": receipt_relative,
                "sha256": receipt_sha,
                "size_bytes": receipt_size,
                "role": "target_set_receipt",
            },
        ]
    )
    source_paths = [str(item["source_path"]) for item in entries]
    destination_paths = [str(item["destination_relative"]) for item in entries]
    if len(set(source_paths)) != len(entries) or len(set(destination_paths)) != len(entries):
        raise TargetTransferError("transfer inventory duplicates a source/destination object")
    identities = source.identities(source_paths)
    for entry in entries:
        identity = identities.get(str(entry["source_path"]))
        if (
            identity is None
            or identity.sha256 != entry["sha256"]
            or identity.size_bytes != entry["size_bytes"]
        ):
            raise TargetTransferError("portable source object SHA-256 or size drifted")
    assigned, lanes = transport._assign_lpt_lanes(entries)
    destination = Path(destination_root).expanduser().resolve(strict=False)
    floor = _positive_int(disk_floor_bytes, label="Bert disk floor")
    reserve = _positive_int(metadata_reserve_bytes, label="metadata reserve")
    if floor < DEFAULT_DISK_FLOOR_BYTES:
        raise TargetTransferError("Bert disk floor may not be lower than 20 GiB")
    if reserve < DEFAULT_METADATA_RESERVE_BYTES:
        raise TargetTransferError("metadata reserve may not be lower than 4 MiB")
    plan = {
        "schema": TRANSFER_PLAN_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "required_authority": REQUIRED_AUTHORITY,
        "parallel_lanes_exact": LANE_COUNT,
        "source": {
            "host": source.host,
            "read_only": True,
            "target_set_root": root,
            "target_set_manifest_relative": manifest_relative,
            "target_set_manifest_sha256": manifest_sha,
            "target_set_receipt_relative": receipt_relative,
            "target_set_receipt_sha256": receipt_sha,
            "goal_contract_relative": str(goal_binding["path"]),
            "goal_contract_sha256": contract_sha,
            "goal_contract_goal_revision": contract_revision,
            "embedded_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
            "complete_action_overlay_manifest_relative": str(overlay_binding["path"]),
            "complete_action_overlay_manifest_sha256": overlay_sha,
            "phi_fit": phi_graph,
            "target_value_transform": transform,
            "raw_episode_zip_identities": [
                {
                    "utc_day": day,
                    "sha256": row["raw_episode_zip_sha256"],
                    "size_bytes": row["raw_episode_zip_size_bytes"],
                }
                for day, row in zip(WINDOW_DAYS, target_days, strict=True)
            ],
        },
        "local_contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
            "goal_revision": contract_revision,
            "embedded_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
        },
        "destination_root": str(destination),
        "bert_disk_free_floor_bytes": floor,
        "metadata_reserve_bytes": reserve,
        "test_only_non_elmo_source": bool(test_allow_non_elmo_source),
        "target_only": True,
        "raw_episode_zip_objects_transferred": False,
        "feature_pack_objects_transferred": False,
        "complete_action_overlay_payload_objects_transferred": False,
        "canonical_target_set_manifest_remains_byte_identical": True,
        "bert_side_c3_or_advantage_scaling_produced_or_transferred": False,
        "target_days": target_days,
        "entries": assigned,
        "lanes": lanes,
        "total_size_bytes": sum(int(item["size_bytes"]) for item in assigned),
    }
    plan_sha = sha256_bytes(canonical_bytes(plan))
    validate_prize_plan_v2_transfer_plan(plan, expected_sha256=plan_sha)
    return plan, plan_sha


def validate_prize_plan_v2_transfer_plan(
    plan: Mapping[str, Any], *, expected_sha256: str | None = None
) -> str:
    if (
        plan.get("schema") != TRANSFER_PLAN_SCHEMA
        or plan.get("owner_goal_revision") != OWNER_GOAL_REVISION
        or plan.get("required_authority") != REQUIRED_AUTHORITY
        or plan.get("parallel_lanes_exact") != LANE_COUNT
        or plan.get("target_only") is not True
        or plan.get("raw_episode_zip_objects_transferred") is not False
        or plan.get("feature_pack_objects_transferred") is not False
        or plan.get("complete_action_overlay_payload_objects_transferred") is not False
        or plan.get("canonical_target_set_manifest_remains_byte_identical") is not True
        or plan.get("bert_side_c3_or_advantage_scaling_produced_or_transferred") is not False
    ):
        raise TargetTransferError("Prize-plan-v2 transfer plan authority/boundary drifted")
    entries = [_mapping(item, label="transfer entry") for item in _rows(plan.get("entries"), label="entries")]
    lanes = [_mapping(item, label="transfer lane") for item in _rows(plan.get("lanes"), label="lanes")]
    if len(lanes) != LANE_COUNT or {item.get("lane_id") for item in lanes} != set(range(LANE_COUNT)):
        raise TargetTransferError("transfer plan does not have exactly four lanes")
    if any(_positive_int(item.get("entry_count"), label="lane entry count") < 1 for item in lanes):
        raise TargetTransferError("all four transfer lanes must be nonempty")
    if len(entries) < LANE_COUNT:
        raise TargetTransferError("transfer plan has fewer objects than lanes")
    source_paths: set[str] = set()
    destination_paths: set[str] = set()
    lane_counts = {lane_id: 0 for lane_id in range(LANE_COUNT)}
    lane_sizes = {lane_id: 0 for lane_id in range(LANE_COUNT)}
    for entry in entries:
        source_path = transport._safe_source_root(entry.get("source_path"), label="entry source")
        destination = _safe_relative(entry.get("destination_relative"), label="entry destination")
        role = str(entry.get("role") or "")
        if role not in PLAN_ROLE_ALLOWLIST or destination.lower().endswith((".zip", ".part", ".partial")):
            raise TargetTransferError("ineligible object entered transfer plan")
        if source_path in source_paths or destination in destination_paths:
            raise TargetTransferError("transfer plan duplicates an object")
        source_paths.add(source_path)
        destination_paths.add(destination)
        _sha256(entry.get("sha256"), label="entry SHA-256")
        size = _positive_int(entry.get("size_bytes"), label="entry size")
        lane_id = entry.get("lane_id")
        if lane_id not in lane_counts:
            raise TargetTransferError("entry has invalid lane assignment")
        lane_counts[int(lane_id)] += 1
        lane_sizes[int(lane_id)] += size
    for lane in lanes:
        lane_id = int(lane["lane_id"])
        if lane.get("entry_count") != lane_counts[lane_id] or lane.get("total_size_bytes") != lane_sizes[lane_id]:
            raise TargetTransferError("lane summary disagrees with entries")
        expected_sources = {
            str(entry["source_path"]) for entry in entries if entry["lane_id"] == lane_id
        }
        if set(_rows(lane.get("source_paths"), label="lane source paths")) != expected_sources:
            raise TargetTransferError("lane source inventory drifted")
    if plan.get("total_size_bytes") != sum(int(item["size_bytes"]) for item in entries):
        raise TargetTransferError("transfer total size drifted")
    floor = _positive_int(plan.get("bert_disk_free_floor_bytes"), label="Bert disk floor")
    reserve = _positive_int(plan.get("metadata_reserve_bytes"), label="metadata reserve")
    if floor < DEFAULT_DISK_FLOOR_BYTES:
        raise TargetTransferError("Bert disk floor may not be lower than 20 GiB")
    if reserve < DEFAULT_METADATA_RESERVE_BYTES:
        raise TargetTransferError("metadata reserve may not be lower than 4 MiB")
    source_binding = _mapping(plan.get("source"), label="transfer source")
    test_non_elmo = plan.get("test_only_non_elmo_source") is True
    if source_binding.get("host") != SOURCE_HOST and not test_non_elmo:
        raise TargetTransferError("production Prize-plan-v2 source host must be elmo")
    if plan.get("test_only_non_elmo_source") not in (False, True):
        raise TargetTransferError("test-only source marker is malformed")
    digest = sha256_bytes(canonical_bytes(dict(plan)))
    if expected_sha256 is not None and digest != expected_sha256:
        raise TargetTransferError("Prize-plan-v2 transfer plan SHA-256 mismatch")
    return digest


def _entry_receipt_relative(entry: Mapping[str, Any], plan_sha256: str) -> str:
    key = sha256_bytes(
        canonical_bytes(
            {
                "plan_sha256": plan_sha256,
                "source_path": entry["source_path"],
                "destination_relative": entry["destination_relative"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
        )
    ).removeprefix("sha256:")
    return f"transfer/receipts/sha256-{key}.prize-plan-v2-file-transfer-receipt.json"


def _part_path(root: Path, plan_sha256: str, entry: Mapping[str, Any]) -> Path:
    return _destination_member(
        root,
        f"transfer/private-partials/{plan_sha256.removeprefix('sha256:')}/"
        f"{entry['destination_relative']}.part",
        label="private Prize-plan-v2 partial",
    )


def _final_state(root: Path, entry: Mapping[str, Any]) -> tuple[Path, FileIdentity | None]:
    final = _destination_member(root, entry["destination_relative"], label="final destination")
    if not final.exists() and not final.is_symlink():
        return final, None
    return final, transport._regular_local_identity(final, label="existing destination final")


def _verify_or_copy_entry(
    *,
    source: SourceReader,
    root: Path,
    entry: Mapping[str, Any],
    plan_sha256: str,
    disk_floor_bytes: int,
    metadata_reserve_bytes: int,
) -> tuple[dict[str, Any], str]:
    expected_sha = _sha256(entry.get("sha256"), label="entry SHA-256")
    expected_size = _positive_int(entry.get("size_bytes"), label="entry size")
    final, existing = _final_state(root, entry)
    observed_disposition = "skipped_exact"
    if existing is not None:
        if existing.sha256 != expected_sha or existing.size_bytes != expected_size:
            raise TargetTransferError(f"existing final conflicts; refusing overwrite: {final}")
    else:
        part = _part_path(root, plan_sha256, entry)
        have = 0
        if part.exists() or part.is_symlink():
            partial = transport._regular_local_identity(part, label="private transfer partial")
            have = partial.size_bytes
            if have > expected_size:
                raise TargetTransferError("private partial exceeds its source object")
            if sha256_file(part, limit=have) != source.prefix_sha256(
                str(entry["source_path"]), have
            ):
                raise TargetTransferError("private partial prefix does not match source")
        if have < expected_size:
            transport._assert_directory_not_symlink(part.parent, create=True)
            source.append_to_part(str(entry["source_path"]), part)
        completed = transport._regular_local_identity(part, label="completed private partial")
        if completed.sha256 != expected_sha or completed.size_bytes != expected_size:
            raise TargetTransferError("completed private partial identity mismatch")
        if shutil.disk_usage(root.parent).free < disk_floor_bytes + metadata_reserve_bytes:
            raise TargetTransferError(
                "Bert free space fell below the required floor before final promotion"
            )
        transport._assert_directory_not_symlink(final.parent, create=True)
        try:
            os.link(part, final)
        except FileExistsError:
            raced = transport._regular_local_identity(final, label="raced final")
            if raced.sha256 != expected_sha or raced.size_bytes != expected_size:
                raise TargetTransferError("raced final conflicts; refusing overwrite")
        except OSError as exc:
            raise TargetTransferError("create-only promotion failed") from exc
        promoted = transport._regular_local_identity(final, label="promoted final")
        if promoted.sha256 != expected_sha or promoted.size_bytes != expected_size:
            raise TargetTransferError("promoted final identity mismatch")
        observed_disposition = "resumed" if have else "copied"
        directory_descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    receipt = {
        "schema": FILE_RECEIPT_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "required_authority": REQUIRED_AUTHORITY,
        "plan_sha256": plan_sha256,
        "source": {
            "path": str(entry["source_path"]),
            "sha256": expected_sha,
            "size_bytes": expected_size,
        },
        "destination": {
            "relative_path": str(entry["destination_relative"]),
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "regular_non_symlink": True,
        },
        "role": str(entry["role"]),
        "utc_day": entry.get("utc_day"),
        "split": entry.get("split"),
        # Receipt bytes describe immutable parity, not whether this particular
        # invocation copied, resumed, raced, or reused the already-sealed file.
        # That keeps create-only retries and clean idempotent reruns valid.
        "disposition": "verified_exact",
        "source_destination_sha256_size_match": True,
        "raw_zip_feature_or_complete_action_payload_transferred": False,
        "private_partials_not_training_eligible": True,
        "create_only": True,
    }
    relative = _entry_receipt_relative(entry, plan_sha256)
    receipt_sha = transport._write_create_only_or_verify(
        _destination_member(root, relative, label="per-file transfer receipt"),
        canonical_bytes(receipt),
    )
    receipt["observed_invocation_disposition"] = observed_disposition
    return receipt, receipt_sha


def _remaining_copy_bytes(root: Path, plan: Mapping[str, Any], plan_sha256: str) -> int:
    remaining = 0
    for raw in _rows(plan.get("entries"), label="transfer entries"):
        entry = _mapping(raw, label="transfer entry")
        expected_size = _positive_int(entry.get("size_bytes"), label="entry size")
        expected_sha = _sha256(entry.get("sha256"), label="entry SHA-256")
        _, final = _final_state(root, entry)
        if final is not None:
            if final.sha256 != expected_sha or final.size_bytes != expected_size:
                raise TargetTransferError("existing final conflicts; refusing overwrite")
            continue
        part = _part_path(root, plan_sha256, entry)
        if part.exists() or part.is_symlink():
            partial = transport._regular_local_identity(part, label="private transfer partial")
            if partial.size_bytes > expected_size:
                raise TargetTransferError("private partial exceeds source size")
            remaining += expected_size - partial.size_bytes
        else:
            remaining += expected_size
    return remaining


def execute_prize_plan_v2_transfer_plan(
    plan: Mapping[str, Any],
    *,
    source: SourceReader,
    plan_sha256: str | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    test_allow_non_elmo_source: bool = False,
) -> dict[str, Any]:
    """Execute the verified plan with exactly four disjoint object streams."""

    calculated_sha = validate_prize_plan_v2_transfer_plan(
        plan, expected_sha256=plan_sha256
    )
    source_binding = _mapping(plan.get("source"), label="plan source binding")
    if not test_allow_non_elmo_source and (
        source.host != SOURCE_HOST
        or source_binding.get("host") != SOURCE_HOST
        or plan.get("test_only_non_elmo_source") is not False
    ):
        raise TargetTransferError("execution requires the production read-only elmo source")
    if test_allow_non_elmo_source and plan.get("test_only_non_elmo_source") is not True:
        raise TargetTransferError("hermetic execution requires a test-only transfer plan")
    destination = Path(str(plan["destination_root"])).expanduser().resolve(strict=False)
    parent = destination.parent
    transport._assert_directory_not_symlink(parent, create=False)
    if destination.exists() or destination.is_symlink():
        state = destination.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise TargetTransferError("destination root is not a real directory")
    remaining = _remaining_copy_bytes(destination, plan, calculated_sha)
    floor = _positive_int(plan.get("bert_disk_free_floor_bytes"), label="Bert disk floor")
    reserve = _positive_int(plan.get("metadata_reserve_bytes"), label="metadata reserve")
    free = (free_bytes or (lambda path: shutil.disk_usage(path).free))(parent)
    if int(free) < floor + remaining + reserve:
        raise TargetTransferError("Bert free space would fall below the required 20GiB floor")
    transport._verified_source_execution_preflight(source, plan)
    transport._assert_directory_not_symlink(destination, create=True)
    plan_relative = (
        f"transfer/plans/sha256-{calculated_sha.removeprefix('sha256:')}."
        "prize-plan-v2-target-transfer-plan.json"
    )
    transport._write_create_only_or_verify(
        _destination_member(destination, plan_relative, label="transfer plan"),
        canonical_bytes(dict(plan)),
    )
    entries_by_lane: dict[int, list[Mapping[str, Any]]] = {
        lane_id: [] for lane_id in range(LANE_COUNT)
    }
    for raw in _rows(plan.get("entries"), label="transfer entries"):
        entry = _mapping(raw, label="transfer entry")
        entries_by_lane[int(entry["lane_id"])].append(entry)

    def run_lane(lane_id: int) -> list[tuple[dict[str, Any], str]]:
        return [
            _verify_or_copy_entry(
                source=source,
                root=destination,
                entry=entry,
                plan_sha256=calculated_sha,
                disk_floor_bytes=floor,
                metadata_reserve_bytes=reserve,
            )
            for entry in entries_by_lane[lane_id]
        ]

    results: dict[int, list[tuple[dict[str, Any], str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=LANE_COUNT) as pool:
        futures = {
            pool.submit(run_lane, lane_id): lane_id for lane_id in range(LANE_COUNT)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    if set(results) != set(range(LANE_COUNT)):
        raise TargetTransferError("exact four-stream execution did not complete")
    receipt_by_destination: dict[str, dict[str, str]] = {}
    for lane_id in range(LANE_COUNT):
        for receipt, receipt_sha in results[lane_id]:
            relative = str(
                _mapping(receipt.get("destination"), label="receipt destination")[
                    "relative_path"
                ]
            )
            entry = next(
                item
                for item in entries_by_lane[lane_id]
                if item["destination_relative"] == relative
            )
            receipt_by_destination[relative] = {
                "receipt_relative": _entry_receipt_relative(entry, calculated_sha),
                "receipt_sha256": receipt_sha,
            }
    source_binding = _mapping(plan.get("source"), label="plan source binding")
    target_view = {
        "schema": TARGET_VIEW_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "required_authority": REQUIRED_AUTHORITY,
        "status": "verified_target_only_offline_bert_input",
        "target_set_root_relative": ".",
        "canonical_target_set_manifest": {
            "relative_path": source_binding["target_set_manifest_relative"],
            "sha256": source_binding["target_set_manifest_sha256"],
            "remains_byte_identical": True,
        },
        "canonical_target_set_receipt": {
            "relative_path": source_binding["target_set_receipt_relative"],
            "sha256": source_binding["target_set_receipt_sha256"],
        },
        "source_identity_graph": {
            "goal_contract_relative": source_binding["goal_contract_relative"],
            "goal_contract_sha256": source_binding["goal_contract_sha256"],
            "goal_contract_goal_revision": source_binding[
                "goal_contract_goal_revision"
            ],
            "embedded_semantic_owner_goal_revision": OWNER_GOAL_REVISION,
            "complete_action_overlay_manifest_relative": source_binding[
                "complete_action_overlay_manifest_relative"
            ],
            "complete_action_overlay_manifest_sha256": source_binding[
                "complete_action_overlay_manifest_sha256"
            ],
            "phi_fit": source_binding["phi_fit"],
            "target_value_transform": source_binding["target_value_transform"],
            "raw_episode_zip_identities": source_binding["raw_episode_zip_identities"],
        },
        "target_days": [
            dict(item) for item in _rows(plan.get("target_days"), label="target days")
        ],
        "file_receipts": [
            {
                "destination_relative": str(entry["destination_relative"]),
                **receipt_by_destination[str(entry["destination_relative"])],
            }
            for entry in sorted(
                (
                    _mapping(item, label="transfer entry")
                    for item in _rows(plan.get("entries"), label="transfer entries")
                ),
                key=lambda item: str(item["destination_relative"]),
            )
        ],
        "plan_relative": plan_relative,
        "plan_sha256": calculated_sha,
        "parallel_lanes_exact": LANE_COUNT,
        "raw_zip_feature_or_complete_action_payload_transferred": False,
        "private_partials_not_training_eligible": True,
        "bert_side_c3_or_advantage_scaling_produced_or_transferred": False,
        "runtime_or_training_started": False,
        "create_only": True,
    }
    view_body = canonical_bytes(target_view)
    view_sha = sha256_bytes(view_body)
    view_relative = (
        f"transfer/target-view/sha256-{view_sha.removeprefix('sha256:')}."
        "prize-plan-v2-target-view.json"
    )
    transport._write_create_only_or_verify(
        _destination_member(destination, view_relative, label="target view"), view_body
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "owner_goal_revision": OWNER_GOAL_REVISION,
        "required_authority": REQUIRED_AUTHORITY,
        "status": "complete_verified_target_only_transfer",
        "plan_relative": plan_relative,
        "plan_sha256": calculated_sha,
        "target_view_relative": view_relative,
        "target_view_sha256": view_sha,
        "target_set_manifest_sha256": source_binding["target_set_manifest_sha256"],
        "target_set_receipt_sha256": source_binding["target_set_receipt_sha256"],
        "goal_contract_sha256": source_binding["goal_contract_sha256"],
        "phi_fit_manifest_sha256": source_binding["phi_fit"]["fit_manifest_sha256"],
        "phi_fit_receipt_sha256": source_binding["phi_fit"]["fit_receipt_sha256"],
        "frozen_phi_table_sha256": source_binding["phi_fit"][
            "frozen_phi_table_sha256"
        ],
        "target_value_transform_sha256": source_binding["target_value_transform"][
            "sha256"
        ],
        "entry_count": len(_rows(plan.get("entries"), label="transfer entries")),
        "parallel_lanes_exact": LANE_COUNT,
        "source_destination_sha256_size_verified": True,
        "raw_zip_feature_or_complete_action_payload_transferred": False,
        "private_partials_not_training_eligible": True,
        "bert_side_c3_or_advantage_scaling_produced_or_transferred": False,
        "runtime_or_training_started": False,
        "create_only": True,
    }
    completion_body = canonical_bytes(completion)
    completion_sha = sha256_bytes(completion_body)
    completion_relative = (
        f"transfer/completion/sha256-{completion_sha.removeprefix('sha256:')}."
        "prize-plan-v2-target-transfer-completion.json"
    )
    transport._write_create_only_or_verify(
        _destination_member(destination, completion_relative, label="completion receipt"),
        completion_body,
    )
    return {
        "destination_root": str(destination),
        "plan_sha256": calculated_sha,
        "canonical_target_set_manifest_path": str(
            _destination_member(
                destination,
                source_binding["target_set_manifest_relative"],
                label="local target-set manifest",
            )
        ),
        "target_view_path": str(
            _destination_member(destination, view_relative, label="target view")
        ),
        "target_view_sha256": view_sha,
        "completion_path": str(
            _destination_member(destination, completion_relative, label="completion")
        ),
        "completion_sha256": completion_sha,
        "entry_count": completion["entry_count"],
        "remaining_copy_bytes_reserved": remaining,
        "metadata_reserve_bytes": reserve,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-host", default=SOURCE_HOST)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-set-manifest-relative", required=True)
    parser.add_argument("--expected-target-set-manifest-sha256", required=True)
    parser.add_argument("--target-set-receipt-relative", required=True)
    parser.add_argument("--expected-target-set-receipt-sha256", required=True)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument(
        "--bert-disk-floor-bytes", type=int, default=DEFAULT_DISK_FLOOR_BYTES
    )
    parser.add_argument(
        "--metadata-reserve-bytes", type=int, default=DEFAULT_METADATA_RESERVE_BYTES
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="copy only after the same checks shown by the default dry run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        source = SSHSourceReader(args.source_host)
        plan, plan_sha = build_prize_plan_v2_transfer_plan(
            source=source,
            source_root=args.source_root,
            destination_root=args.destination_root,
            target_set_manifest_relative=args.target_set_manifest_relative,
            expected_target_set_manifest_sha256=args.expected_target_set_manifest_sha256,
            target_set_receipt_relative=args.target_set_receipt_relative,
            expected_target_set_receipt_sha256=args.expected_target_set_receipt_sha256,
            local_contract_path=args.contract,
            expected_contract_sha256=args.expected_contract_sha256,
            disk_floor_bytes=args.bert_disk_floor_bytes,
            metadata_reserve_bytes=args.metadata_reserve_bytes,
        )
        if not args.execute:
            print(
                json.dumps(
                    {
                        "phase": "dry_run",
                        "plan_sha256": plan_sha,
                        "entry_count": len(plan["entries"]),
                        "total_size_bytes": plan["total_size_bytes"],
                        "parallel_lanes_exact": LANE_COUNT,
                        "lanes": plan["lanes"],
                        "destination_root": plan["destination_root"],
                        "raw_zip_feature_or_complete_action_payload_transferred": False,
                        "runtime_or_training_started": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = execute_prize_plan_v2_transfer_plan(
            plan, source=source, plan_sha256=plan_sha
        )
    except (TargetTransferError, targets.PrizePlanTargetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"phase": "complete", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
