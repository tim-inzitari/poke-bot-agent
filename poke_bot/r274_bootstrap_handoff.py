"""Receipt-backed r274 bootstrap and tactical-route handoff.

The 25-epoch bootstrap deliberately trains the tactical outcome head and its
physical route through the shadow loss while keeping the route out of policy
logits.  The first submission therefore comes from a runtime-enabled OwnDeck
checkpoint whose tactical route remains off.  Only a successful upload receipt
may authorize the second, config-only checkpoint that enables that learned
route for RL update zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from . import checkpoint


BOOTSTRAP_COMPLETION_SCHEMA = "poke_bot.alakazam_r274_bootstrap_completion/v1"
BOOTSTRAP_SUBMISSION_SCHEMA = "poke_bot.alakazam_r274_bootstrap_submission_checkpoint/v1"
TACTICAL_ACTIVATION_SCHEMA = "poke_bot.alakazam_r274_tactical_route_activation/v1"
R274_HANDOFF_SCHEMA = "poke_bot.alakazam_r274_bootstrap_to_rl_handoff/v1"
R280_GPU_BOOTSTRAP_SCHEMA = "poke_bot.r280_gpu_resident_bootstrap_result/v1"
R280_TACTICAL_REPAIR_SCHEMA = "poke_bot.r280_tactical_bootstrap_train_repair/v1"
R281_ADAPTER_TRAINING_SCHEMA = (
    "poke_bot.r281_bootstrap_matchup_adapter_training/v1"
)

EXPECTED_DATES = tuple(f"2026-07-{day:02d}" for day in range(22, 32)) + tuple(
    f"2026-08-{day:02d}" for day in range(1, 11)
)
EXPECTED_EXPERT_DECISIONS = 2_040_911
EXPECTED_BOOTSTRAP_EPOCHS = 25
MINIMUM_TACTICAL_ROOTS = 1_024
RUNTIME_GATE_FIELDS = (
    "own_deck_ledger_runtime_enabled",
    "visible_tutor_completion_route_runtime_enabled",
    "terminal_conversion_route_runtime_enabled",
)
TACTICAL_ROUTE_FIELDS = (
    "tactical_sequence_outcome_route_enabled",
    "tactical_sequence_outcome_route_runtime_enabled",
)
TACTICAL_PREFIXES = (
    "tactical_sequence_outcome_head.",
    "tactical_sequence_outcome_route.",
)


class R274BootstrapHandoffError(RuntimeError):
    """The bootstrap boundary cannot be promoted without complete evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def file_identity(path: Path | str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R274BootstrapHandoffError(
            f"evidence must be a regular non-symlink file: {candidate}"
        )
    candidate = candidate.resolve()
    return {
        "path": str(candidate),
        "sha256": _sha256_file(candidate),
        "size_bytes": int(candidate.stat().st_size),
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def receipt_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["receipt_sha256"] = receipt_digest(payload)
    return payload


def write_immutable_json(path: Path | str, value: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(value)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != body:
            raise R274BootstrapHandoffError(
                f"immutable receipt already exists with different bytes: {target}"
            )
        return file_identity(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.r274-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return file_identity(target)


def _read_object(path: Path | str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    try:
        value = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R274BootstrapHandoffError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R274BootstrapHandoffError(f"{label} must be a JSON object")
    return value, identity


def _changed_finite_prefix(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    prefix: str,
) -> bool:
    names = [name for name in after if name.startswith(prefix) and name in before]
    return bool(names) and any(
        bool(torch.isfinite(after[name]).all())
        and not torch.equal(before[name], after[name])
        for name in names
    )


def validate_full_expert_manifest(path: Path | str) -> dict[str, Any]:
    manifest, identity = _read_object(path, label="r274 expert manifest")
    rows = manifest.get("source_days")
    if not isinstance(rows, list):
        raise R274BootstrapHandoffError("expert manifest lacks source_days")
    dates = tuple(str(dict(row).get("date") or "") for row in rows)
    decisions = sum(int(dict(row).get("matching_decisions", 0)) for row in rows)
    if dates != EXPECTED_DATES or decisions != EXPECTED_EXPERT_DECISIONS:
        raise R274BootstrapHandoffError(
            "bootstrap manifest is not the exact 20-day/2,040,911-decision corpus"
        )
    if any(
        dict(row).get("source_archive_validated") is not True
        or dict(row).get("source_feature_validated") is not True
        for row in rows
    ):
        raise R274BootstrapHandoffError("expert manifest contains an unvalidated day")
    return {
        **identity,
        "dates": list(dates),
        "days": len(dates),
        "decisions": decisions,
    }


def validate_tactical_overlay(path: Path | str) -> dict[str, Any]:
    overlay, identity = _read_object(path, label="r274 tactical overlay")
    rows = overlay.get("rows")
    roots = int(overlay.get("roots", -1))
    if (
        overlay.get("mode") != "shadow_only"
        or overlay.get("planner_dispatch_authority") is not False
        or not isinstance(rows, list)
        or roots != len(rows)
        or roots < MINIMUM_TACTICAL_ROOTS
    ):
        raise R274BootstrapHandoffError("tactical overlay is not sealed shadow evidence")
    keys = {
        (
            str(row.get("episode_id") or ""),
            int(row.get("seat", -1)),
            int(row.get("env_step", -1)),
            str(row.get("observation_fingerprint") or ""),
        )
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(keys) != roots:
        raise R274BootstrapHandoffError("tactical overlay keys are not unique")
    return {**identity, "roots": roots}


def validate_r280_gpu_bootstrap_result(
    path: Path | str,
    *,
    expected_bootstrap_checkpoint: Path | str,
) -> dict[str, Any]:
    """Validate the immutable r280 result that replaces the streaming receipt."""

    result, identity = _read_object(path, label="r280 GPU bootstrap result")
    expected_output = file_identity(expected_bootstrap_checkpoint)
    counts = dict(result.get("counts") or {})
    residency = dict(result.get("gpu_residency") or {})
    pack = dict(result.get("pack") or {})
    output = dict(result.get("output") or {})
    if (
        result.get("schema") != R280_GPU_BOOTSTRAP_SCHEMA
        or result.get("status") != "completed"
        or int(result.get("epochs_completed", -1)) != EXPECTED_BOOTSTRAP_EPOCHS
        or result.get("receipt_sha256") != receipt_digest(result)
        or output != expected_output
        or int(counts.get("games", -1)) != 26_704
        or int(counts.get("decisions", -1)) != EXPECTED_EXPERT_DECISIONS
        or int(counts.get("samples", 0)) <= 0
        or int(counts.get("options", 0)) <= 0
        or not str(pack.get("sha256") or "").startswith("sha256:")
        or int(pack.get("size_bytes", -1)) != 5_725_073_070
        or residency.get("device") != "cuda:1"
        or residency.get("full_numeric_pack_resident") is not True
        or residency.get("device_side_batch_gather") is not True
        or residency.get("pinned_cpu_fallback_used") is not False
    ):
        raise R274BootstrapHandoffError("r280 GPU bootstrap result is invalid")
    return {**result, "file_identity": identity}


def validate_bootstrap_checkpoint(
    *,
    base_checkpoint: Path | str,
    bootstrap_checkpoint: Path | str,
    expert_manifest: Path | str,
    tactical_overlay: Path | str,
    gpu_bootstrap_result_receipt: Path | str | None = None,
    tactical_repair_receipt: Path | str | None = None,
    adapter_training_receipt: Path | str | None = None,
) -> dict[str, Any]:
    base_identity = file_identity(base_checkpoint)
    bootstrap_identity = file_identity(bootstrap_checkpoint)
    base = checkpoint.load_checkpoint(base_identity["path"], map_location="cpu")
    child = checkpoint.load_checkpoint(bootstrap_identity["path"], map_location="cpu")
    adapter_training: dict[str, Any] | None = None
    tactical_child_identity = bootstrap_identity
    tactical_child = child
    if adapter_training_receipt is None:
        raise R274BootstrapHandoffError(
            "revision-281 bootstrap requires matchup-adapter training evidence"
        )
    if adapter_training_receipt is not None:
        adapter, adapter_identity = _read_object(
            adapter_training_receipt,
            label="r281 bootstrap matchup-adapter training",
        )
        parent_row = dict(adapter.get("parent") or {})
        checkpoint_row = dict(adapter.get("checkpoint") or {})
        eligible_routes = [int(value) for value in adapter.get("eligible_routes") or ()]
        changed_routes = [int(value) for value in adapter.get("changed_routes") or ()]
        inactive_routes = [
            int(value) for value in adapter.get("inactive_routes_bit_identical") or ()
        ]
        per_route = dict(adapter.get("per_route_validation") or {})
        split = dict(adapter.get("source_disjoint") or {})
        if (
            adapter.get("schema") != R281_ADAPTER_TRAINING_SCHEMA
            or adapter.get("status") != "passed"
            or adapter.get("receipt_sha256") != receipt_digest(adapter)
            or checkpoint_row != bootstrap_identity
            or file_identity(str(parent_row.get("path") or "")) != parent_row
            or file_identity(str(checkpoint_row.get("path") or ""))
            != checkpoint_row
            or int(adapter.get("epochs", -1)) != 25
            or int(adapter.get("steps", 0)) <= 0
            or int(adapter.get("rows", 0)) <= 0
            or adapter.get("optimizer_scope") != "matchup_adapter_bank_only"
            or adapter.get("all_non_adapter_tensors_bit_identical") is not True
            or adapter.get("unsupported_slots_remain_dormant") is not True
            or not eligible_routes
            or changed_routes != eligible_routes
            or sorted(eligible_routes + inactive_routes) != list(range(64))
            or set(eligible_routes).intersection(inactive_routes)
            or split.get("train_days") != list(EXPECTED_DATES[:-2])
            or split.get("validation_days") != list(EXPECTED_DATES[-2:])
            or set(per_route) != {str(route) for route in eligible_routes}
        ):
            raise R274BootstrapHandoffError(
                "r281 bootstrap matchup-adapter training receipt is invalid"
            )
        for route in eligible_routes:
            row = dict(per_route[str(route)])
            numeric = (
                "route_on_loss",
                "route_off_loss",
                "route_on_minus_off_loss",
            )
            if (
                int(row.get("train_games", 0)) <= 0
                or int(row.get("train_decisions", 0)) <= 0
                or int(row.get("validation_games", 0)) <= 0
                or int(row.get("validation_decisions", 0)) <= 0
                or any(not math.isfinite(float(row.get(name, math.nan))) for name in numeric)
            ):
                raise R274BootstrapHandoffError(
                    f"r281 adapter route {route} lacks finite heldout evidence"
                )
        adapter_parent = checkpoint.load_checkpoint(
            parent_row["path"], map_location="cpu"
        )
        adapter_parent_state = dict(adapter_parent.get("model_state_dict") or {})
        adapter_child_state = dict(child.get("model_state_dict") or {})
        if (
            int(adapter_parent.get("epoch", -1)) != int(child.get("epoch", -2))
            or int(adapter_parent.get("rl_iteration", -1))
            != int(child.get("rl_iteration", -2))
            or any(
                name not in adapter_child_state
                or not torch.equal(value, adapter_child_state[name])
                for name, value in adapter_parent_state.items()
                if not name.startswith("matchup_adapter_bank.")
            )
        ):
            raise R274BootstrapHandoffError(
                "r281 adapter phase changed counters or non-adapter tensors"
            )
        tactical_child_identity = parent_row
        tactical_child = adapter_parent
        adapter_training = {**adapter, "file_identity": adapter_identity}
    tactical_repair: dict[str, Any] | None = None
    gpu_result_checkpoint: Path | str = bootstrap_identity["path"]
    if tactical_repair_receipt is not None:
        repair, repair_identity = _read_object(
            tactical_repair_receipt, label="r280 tactical bootstrap repair"
        )
        parent_row = dict(repair.get("parent") or {})
        checkpoint_row = dict(repair.get("checkpoint") or {})
        if (
            repair.get("schema") != R280_TACTICAL_REPAIR_SCHEMA
            or repair.get("status") != "passed"
            or repair.get("receipt_sha256") != receipt_digest(repair)
            or checkpoint_row != tactical_child_identity
            or file_identity(str(parent_row.get("path") or "")) != parent_row
            or file_identity(str(checkpoint_row.get("path") or ""))
            != checkpoint_row
            or repair.get("train_days") != ["2026-08-08"]
            or set(repair.get("validation_days") or ())
            != {"2026-08-09", "2026-08-10"}
            or repair.get("source_disjoint") is not True
            or int(repair.get("epochs", -1)) != 25
            or int(repair.get("attached_roots", -1)) < MINIMUM_TACTICAL_ROOTS
            or int(repair.get("labeled_option_rows_seen", 0)) <= 0
            or repair.get("optimizer_scope")
            != "tactical_head_and_shadow_route_only"
            or repair.get("changed_finite_prefixes")
            != {prefix: True for prefix in TACTICAL_PREFIXES}
            or repair.get("all_non_tactical_tensors_bit_identical") is not True
            or repair.get("epoch_counter_unchanged") is not True
            or repair.get("rl_iteration_unchanged") is not True
            or repair.get("planner_dispatch_authority") is not False
            or repair.get("tactical_route_policy_influence") != "exact_zero"
        ):
            raise R274BootstrapHandoffError(
                "r280 tactical bootstrap repair receipt is invalid"
            )
        repair_parent = checkpoint.load_checkpoint(
            parent_row["path"], map_location="cpu"
        )
        repair_parent_state = dict(repair_parent.get("model_state_dict") or {})
        child_state_for_scope = dict(tactical_child.get("model_state_dict") or {})
        if (
            int(repair_parent.get("epoch", -1))
            != int(tactical_child.get("epoch", -2))
            or int(repair_parent.get("rl_iteration", -1))
            != int(tactical_child.get("rl_iteration", -2))
            or any(
                name not in child_state_for_scope
                or not torch.equal(value, child_state_for_scope[name])
                for name, value in repair_parent_state.items()
                if not any(name.startswith(prefix) for prefix in TACTICAL_PREFIXES)
            )
        ):
            raise R274BootstrapHandoffError(
                "r280 tactical repair changed counters or non-tactical tensors"
            )
        gpu_result_checkpoint = parent_row["path"]
        tactical_repair = {**repair, "file_identity": repair_identity}
    if str(child.get("archetype_id") or "").casefold() != "alakazam":
        raise R274BootstrapHandoffError("bootstrap checkpoint is not Alakazam")
    if int(child.get("epoch", -1)) - int(base.get("epoch", -1)) != EXPECTED_BOOTSTRAP_EPOCHS:
        raise R274BootstrapHandoffError("bootstrap checkpoint is not exactly 25 epochs")
    if int(child.get("rl_iteration", -1)) != int(base.get("rl_iteration", -2)):
        raise R274BootstrapHandoffError("bootstrap advanced an RL counter")
    cfg = dict(child.get("model_config") or {})
    required_true = (
        "own_deck_ledger_enabled",
        "visible_tutor_completion_head_enabled",
        "terminal_conversion_head_enabled",
        "tactical_sequence_outcome_head_enabled",
        "visible_tutor_completion_route_enabled",
        "terminal_conversion_route_enabled",
        "tactical_sequence_outcome_route_present",
        "combo_state_head_enabled",
    )
    if any(cfg.get(name) is not True for name in required_true):
        raise R274BootstrapHandoffError("bootstrap lost a required physical head/route")
    if cfg.get("tactical_sequence_outcome_route_enabled") is not False:
        raise R274BootstrapHandoffError("bootstrap tactical route influenced policy")
    if any(cfg.get(name) is not False for name in RUNTIME_GATE_FIELDS + TACTICAL_ROUTE_FIELDS):
        raise R274BootstrapHandoffError("raw bootstrap unexpectedly grants runtime authority")
    child_extra = dict(child.get("extra") or {})
    rehearsal = dict(child_extra.get("r260_streaming_rehearsal") or {})
    r280_training = dict(child_extra.get("r280_gpu_resident_training") or {})
    r280_result: dict[str, Any] | None = None
    representation = "streaming"
    if r280_training:
        if gpu_bootstrap_result_receipt is None:
            raise R274BootstrapHandoffError(
                "r280 bootstrap requires its immutable GPU result receipt"
            )
        r280_result = validate_r280_gpu_bootstrap_result(
            gpu_bootstrap_result_receipt,
            expected_bootstrap_checkpoint=gpu_result_checkpoint,
        )
        expert_rehearsal = dict(child_extra.get("expert_rehearsal") or {})
        packed_manifest = dict(expert_rehearsal.get("manifest") or {})
        packed_counts = dict(packed_manifest.get("counts") or {})
        if (
            r280_training.get("schema") != R280_GPU_BOOTSTRAP_SCHEMA
            or r280_training.get("device") != "cuda:1"
            or r280_training.get("epoch_batch_gather") != "device_side_only"
            or r280_training.get("pinned_cpu_streaming_used") is not False
            or r280_training.get("resident_python_objects_used") is not False
            or r280_training.get("pack_sha256")
            != dict(r280_result.get("pack") or {}).get("sha256")
            or int(r280_training.get("pack_tensor_bytes", 0)) <= 0
            or int(child.get("step", -1)) <= int(base.get("step", -1))
            or int(expert_rehearsal.get("epochs", -1))
            != EXPECTED_BOOTSTRAP_EPOCHS
            or packed_manifest.get("schema")
            != "poke_bot.r279_contiguous_expert_pack_receipt/v1"
            or int(packed_counts.get("games", -1)) != 26_704
            or int(packed_counts.get("decisions", -1))
            != EXPECTED_EXPERT_DECISIONS
        ):
            raise R274BootstrapHandoffError(
                "r280 checkpoint GPU-resident rehearsal evidence is incomplete"
            )
        rehearsal = expert_rehearsal
        representation = "full_gpu_resident_contiguous_pack"
    elif (
        rehearsal.get("schema") != "poke_bot.r260_streaming_expert_rehearsal/v1"
        or int(rehearsal.get("tactical_exact_root_count", -1)) < MINIMUM_TACTICAL_ROOTS
        or int(rehearsal.get("sampled_key_count", -1)) <= 0
        or int(rehearsal.get("sidecar_joined_decision_count", -1)) < 0
        or int(rehearsal.get("sidecar_masked_unjoinable_decision_count", -1)) < 0
        or int(rehearsal.get("sidecar_joined_decision_count", 0))
        + int(rehearsal.get("sidecar_masked_unjoinable_decision_count", 0))
        != int(rehearsal.get("sampled_key_count", -1))
        or int(child.get("step", -1)) != int(base.get("step", -2))
    ):
        raise R274BootstrapHandoffError("bootstrap streaming receipt is incomplete")
    base_state = dict(base.get("model_state_dict") or {})
    child_state = dict(child.get("model_state_dict") or {})
    changed = {
        prefix: _changed_finite_prefix(base_state, child_state, prefix)
        for prefix in TACTICAL_PREFIXES
    }
    if not all(changed.values()):
        raise R274BootstrapHandoffError(
            "bootstrap did not learn both tactical head and shadow route"
        )
    manifest = validate_full_expert_manifest(expert_manifest)
    overlay = validate_tactical_overlay(tactical_overlay)
    return {
        "schema": BOOTSTRAP_COMPLETION_SCHEMA,
        "status": "passed_exact_25_epoch_full_expert_bootstrap",
        "base_checkpoint": base_identity,
        "bootstrap_checkpoint": bootstrap_identity,
        "expert_manifest": manifest,
        "tactical_overlay": overlay,
        "epochs": EXPECTED_BOOTSTRAP_EPOCHS,
        "rl_iteration_before_after": [
            int(base.get("rl_iteration", 0)),
            int(child.get("rl_iteration", 0)),
        ],
        "streaming_rehearsal": rehearsal,
        "training_representation": representation,
        "gpu_bootstrap_result": r280_result,
        "tactical_bootstrap_repair": tactical_repair,
        "matchup_adapter_bootstrap_training": adapter_training,
        "tactical_tensor_learning": changed,
        "tactical_policy_influence": "exact_zero",
        "planner_dispatch_authority": False,
    }


def materialize_bootstrap_submission_checkpoint(
    *,
    base_checkpoint: Path | str,
    bootstrap_checkpoint: Path | str,
    expert_manifest: Path | str,
    tactical_overlay: Path | str,
    gpu_bootstrap_result_receipt: Path | str | None = None,
    tactical_repair_receipt: Path | str | None = None,
    adapter_training_receipt: Path | str | None = None,
    output_checkpoint: Path | str,
    output_receipt: Path | str,
) -> dict[str, Any]:
    completion = validate_bootstrap_checkpoint(
        base_checkpoint=base_checkpoint,
        bootstrap_checkpoint=bootstrap_checkpoint,
        expert_manifest=expert_manifest,
        tactical_overlay=tactical_overlay,
        gpu_bootstrap_result_receipt=gpu_bootstrap_result_receipt,
        tactical_repair_receipt=tactical_repair_receipt,
        adapter_training_receipt=adapter_training_receipt,
    )
    output = Path(output_checkpoint).expanduser().absolute()
    if output.is_symlink():
        raise FileExistsError(output)
    payload = checkpoint.load_checkpoint(bootstrap_checkpoint, map_location="cpu")
    cfg = dict(payload.get("model_config") or {})
    cfg.update({name: True for name in RUNTIME_GATE_FIELDS})
    cfg.update({name: False for name in TACTICAL_ROUTE_FIELDS})
    # The learned adapter bank remains serialized dormant.  Submission serving
    # authority comes only from the checksum-bound packaged public tree.
    cfg["matchup_adapters_enabled"] = False
    payload["model_config"] = cfg
    extra = dict(payload.get("extra") or {})
    if "r274_bootstrap_submission" in extra:
        raise R274BootstrapHandoffError("bootstrap already carries submission metadata")
    extra["r274_bootstrap_submission"] = {
        "schema": BOOTSTRAP_SUBMISSION_SCHEMA,
        "bootstrap_checkpoint_sha256": completion["bootstrap_checkpoint"]["sha256"],
        "own_deck_runtime_routes_enabled": True,
        "tactical_route_enabled": False,
        "tactical_route_submission_influence": "exact_zero",
        "direct_policy_only": True,
        "rtp_enabled": False,
    }
    adapter_training = dict(completion.get("matchup_adapter_bootstrap_training") or {})
    adapter_fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    route_decisions = {
        str(key): int(value)
        for key, value in dict(adapter_fit.get("route_decisions") or {}).items()
    }
    phase_route_decisions: dict[str, int] = {}
    phase_route_sequences: dict[str, int] = {}
    for route in adapter_training.get("eligible_routes") or ():
        validation = dict(
            dict(adapter_training.get("per_route_validation") or {}).get(str(route))
            or {}
        )
        archetype_id = str(validation.get("archetype_id") or "")
        decisions = int(validation.get("train_decisions") or 0) * EXPECTED_BOOTSTRAP_EPOCHS
        sequences = int(validation.get("train_games") or 0) * EXPECTED_BOOTSTRAP_EPOCHS
        if not archetype_id or decisions <= 0 or sequences <= 0:
            raise R274BootstrapHandoffError(
                "adapter submission metadata lacks trained route support"
            )
        route_decisions[archetype_id] = route_decisions.get(archetype_id, 0) + decisions
        phase_route_decisions[archetype_id] = decisions
        phase_route_sequences[archetype_id] = sequences
    route_contract = dict(extra.get("matchup_adapter_config") or {})
    registry = dict(route_contract.get("slot_registry") or {})
    active_ids = [str(value) for value in registry.get("active_expert_ids") or ()]
    trained_ids = [
        archetype_id for archetype_id in active_ids
        if int(route_decisions.get(archetype_id, 0)) > 0
    ]
    dormant_ids = [
        archetype_id for archetype_id in active_ids
        if int(route_decisions.get(archetype_id, 0)) == 0
    ]
    adapter_fit.update(
        {
            "schema": "poke_bot.dormant_matchup_adapter_fit/v1",
            "runtime_enabled": False,
            "base_frozen": True,
            "optimizer_scope": "matchup_adapter_bank_only",
            "epochs": int(adapter_fit.get("epochs") or 0) + EXPECTED_BOOTSTRAP_EPOCHS,
            "steps": int(adapter_fit.get("steps") or 0)
            + int(adapter_training.get("steps") or 0),
            "rows": int(adapter_fit.get("rows") or 0)
            + int(adapter_training.get("rows") or 0),
            "phase_epochs": EXPECTED_BOOTSTRAP_EPOCHS,
            "phase_steps": int(adapter_training.get("steps") or 0),
            "phase_rows": int(adapter_training.get("rows") or 0),
            "phase_route_decisions": phase_route_decisions,
            "phase_route_sequences": phase_route_sequences,
            "route_decisions": route_decisions,
            "trained_archetype_ids": trained_ids,
            "dormant_no_example_archetype_ids": dormant_ids,
            "zero_example_routes_remain_dormant": True,
            "optimizer_state_restored": False,
            "optimizer_reset_reason": "r281_contiguous_bootstrap_new_fit",
        }
    )
    extra["dormant_matchup_adapter_fit"] = adapter_fit
    dormant_bank = dict(extra.get("dormant_matchup_adapter_bank") or {})
    dormant_bank["adapter_config"] = route_contract
    extra["dormant_matchup_adapter_bank"] = dormant_bank
    extra.pop("dormant_matchup_adapter_optimizer_state", None)
    extra["r281_matchup_adapter_optimizer_reset"] = {
        "schema": R281_ADAPTER_TRAINING_SCHEMA,
        "receipt_sha256": adapter_training.get("receipt_sha256"),
        "steps": int(adapter_training.get("steps") or 0),
        "rows": int(adapter_training.get("rows") or 0),
        "optimizer_scope": "matchup_adapter_bank_only",
        "base_frozen": True,
        "continuation_behavior": "fresh_adam_state_on_next_isolated_rl_phase",
    }
    extra["matchup_adapters_runtime_enabled"] = False
    extra["matchup_adapter_training_enabled"] = False
    extra["matchup_adapter_optimizer_included"] = False
    payload["extra"] = extra
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        # A failed post-write reconstruction may leave the deterministic
        # create-only child without its receipt. Adopt it only after proving
        # that its tensors, counters, config, and exact boundary metadata are
        # identical to the artifact this invocation would have created.
        existing = checkpoint.load_checkpoint(output, map_location="cpu")
        expected_state = dict(payload.get("model_state_dict") or {})
        existing_state = dict(existing.get("model_state_dict") or {})
        scalar_keys = ("step", "epoch", "rl_iteration", "archetype_id", "model_id")
        if (
            dict(existing.get("model_config") or {}) != cfg
            or dict(existing.get("extra") or {}).get("r274_bootstrap_submission")
            != extra["r274_bootstrap_submission"]
            or any(existing.get(name) != payload.get(name) for name in scalar_keys)
            or set(existing_state) != set(expected_state)
            or any(
                not torch.equal(existing_state[name], expected_state[name])
                for name in expected_state
            )
        ):
            raise R274BootstrapHandoffError(
                "orphan bootstrap submission checkpoint differs from expected bytes"
            )
    else:
        checkpoint.immutable_torch_save(payload, output)
    # Reconstructing is the authoritative check that the serialized gate
    # combination is legal and does not rely on process-local mutations.
    from .train import load_model_from_checkpoint

    model = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if any(getattr(model, name, None) is not True for name in RUNTIME_GATE_FIELDS):
        raise R274BootstrapHandoffError("submission checkpoint lost OwnDeck runtime gates")
    if any(getattr(model, name, None) is not False for name in TACTICAL_ROUTE_FIELDS):
        raise R274BootstrapHandoffError("submission checkpoint enabled tactical influence")
    receipt = _seal(
        {
            "schema": BOOTSTRAP_SUBMISSION_SCHEMA,
            "status": "ready_for_first_if_allowed_submission",
            "bootstrap_completion": completion,
            "submission_checkpoint": file_identity(output),
            "runtime_gates": {name: True for name in RUNTIME_GATE_FIELDS},
            "tactical_route": {name: False for name in TACTICAL_ROUTE_FIELDS},
            "direct_policy_only": True,
            "rtp_enabled": False,
            "planner_dispatch_authority": False,
            "submission_performed": False,
        }
    )
    write_immutable_json(output_receipt, receipt)
    return receipt


def validate_successful_upload_receipt(
    path: Path | str,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    receipt, identity = _read_object(path, label="r274 bootstrap upload receipt")
    status = str(receipt.get("status") or "")
    if status not in {"submitted", "reconciled_existing_remote_submission"}:
        raise R274BootstrapHandoffError("bootstrap upload has not succeeded")
    checkpoint_row = receipt.get("checkpoint") or receipt.get("terminal_checkpoint")
    if not isinstance(checkpoint_row, Mapping):
        raise R274BootstrapHandoffError("upload receipt lacks checkpoint identity")
    if str(checkpoint_row.get("sha256") or checkpoint_row.get("digest") or "") != str(
        expected_checkpoint_sha256
    ):
        raise R274BootstrapHandoffError("upload receipt checkpoint digest mismatch")
    submission_id = receipt.get("remote_submission_id") or receipt.get("submission_id")
    if isinstance(submission_id, bool) or not isinstance(submission_id, int) or submission_id <= 0:
        raise R274BootstrapHandoffError("upload receipt lacks a remote submission ID")
    if receipt.get("direct_policy_only") is not True or receipt.get("rtp_enabled") is not False:
        raise R274BootstrapHandoffError("upload receipt is not direct-policy NO-RTP")
    return {**receipt, "file_identity": identity}


def materialize_tactical_activation_checkpoint(
    *,
    submission_checkpoint: Path | str,
    bootstrap_submission_receipt: Path | str,
    upload_receipt: Path | str,
    impact: Mapping[str, Any],
    output_checkpoint: Path | str,
    output_receipt: Path | str,
) -> dict[str, Any]:
    submission = file_identity(submission_checkpoint)
    submission_receipt, submission_receipt_identity = _read_object(
        bootstrap_submission_receipt, label="r274 bootstrap submission receipt"
    )
    if (
        submission_receipt.get("schema") != BOOTSTRAP_SUBMISSION_SCHEMA
        or submission_receipt.get("status") != "ready_for_first_if_allowed_submission"
        or submission_receipt.get("receipt_sha256") != receipt_digest(submission_receipt)
        or dict(submission_receipt.get("submission_checkpoint") or {}) != submission
    ):
        raise R274BootstrapHandoffError("bootstrap submission receipt is invalid")
    upload = validate_successful_upload_receipt(
        upload_receipt, expected_checkpoint_sha256=submission["sha256"]
    )
    required_impact = {
        "source_disjoint",
        "support",
        "action_change_rate",
        "top_action_margin_delta",
        "policy_kl_divergence",
        "value_delta",
        "terminal_win_and_public_sme_label_calibration",
        "route_magnitude",
        "latency_ms",
        "max_abs_logit_delta",
    }
    if set(impact) != required_impact:
        raise R274BootstrapHandoffError("tactical impact metric inventory changed")
    numeric = (
        "action_change_rate",
        "top_action_margin_delta",
        "policy_kl_divergence",
        "value_delta",
        "route_magnitude",
        "latency_ms",
        "max_abs_logit_delta",
    )
    if any(not math.isfinite(float(impact[name])) for name in numeric):
        raise R274BootstrapHandoffError("tactical impact contains nonfinite metrics")
    if (
        impact.get("source_disjoint") is not True
        or int(impact.get("support", 0)) <= 0
        or not 0.0 < float(impact["max_abs_logit_delta"]) <= 1.0
        or float(impact["route_magnitude"]) <= 0.0
        or float(impact["policy_kl_divergence"]) < 0.0
        or float(impact["latency_ms"]) < 0.0
    ):
        raise R274BootstrapHandoffError("tactical route lacks bounded nonzero impact")
    output = Path(output_checkpoint).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = checkpoint.load_checkpoint(submission["path"], map_location="cpu")
    cfg = dict(payload.get("model_config") or {})
    if any(cfg.get(name) is not True for name in RUNTIME_GATE_FIELDS):
        raise R274BootstrapHandoffError("submission checkpoint lacks OwnDeck runtime gates")
    if any(cfg.get(name) is not False for name in TACTICAL_ROUTE_FIELDS):
        raise R274BootstrapHandoffError("submission checkpoint was not tactical route-off")
    cfg.update({name: True for name in TACTICAL_ROUTE_FIELDS})
    payload["model_config"] = cfg
    extra = dict(payload.get("extra") or {})
    extra["r274_tactical_route_activation"] = {
        "schema": TACTICAL_ACTIVATION_SCHEMA,
        "bootstrap_submission_checkpoint_sha256": submission["sha256"],
        "bootstrap_upload_receipt_sha256": upload["file_identity"]["sha256"],
        "impact": copy.deepcopy(dict(impact)),
        "planner_dispatch_authority": False,
    }
    payload["extra"] = extra
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.immutable_torch_save(payload, output)
    from .train import load_model_from_checkpoint

    model = load_model_from_checkpoint(output, device=torch.device("cpu"))
    if any(getattr(model, name, None) is not True for name in TACTICAL_ROUTE_FIELDS):
        raise R274BootstrapHandoffError("activated checkpoint did not reconstruct route-on")
    receipt = _seal(
        {
            "schema": TACTICAL_ACTIVATION_SCHEMA,
            "status": "passed_ready_for_rl_update_0",
            "bootstrap_submission_receipt": submission_receipt_identity,
            "bootstrap_submission_checkpoint": submission,
            "bootstrap_upload_receipt": upload["file_identity"],
            "bootstrap_remote_submission_id": int(
                upload.get("remote_submission_id") or upload.get("submission_id")
            ),
            "activated_checkpoint": file_identity(output),
            "runtime_gates": {
                **{name: True for name in RUNTIME_GATE_FIELDS},
                **{name: True for name in TACTICAL_ROUTE_FIELDS},
            },
            "impact": copy.deepcopy(dict(impact)),
            "direct_policy_only": True,
            "rtp_enabled": False,
            "planner_dispatch_authority": False,
        }
    )
    write_immutable_json(output_receipt, receipt)
    return receipt


def validate_handoff_receipt(
    path: Path | str,
    *,
    expected_initial_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    receipt, identity = _read_object(path, label="r274 tactical activation receipt")
    if (
        receipt.get("schema") != TACTICAL_ACTIVATION_SCHEMA
        or receipt.get("status") != "passed_ready_for_rl_update_0"
        or receipt.get("receipt_sha256") != receipt_digest(receipt)
        or receipt.get("direct_policy_only") is not True
        or receipt.get("rtp_enabled") is not False
        or receipt.get("planner_dispatch_authority") is not False
    ):
        raise R274BootstrapHandoffError("r274 tactical activation receipt is invalid")
    activated = dict(receipt.get("activated_checkpoint") or {})
    observed = file_identity(str(activated.get("path") or ""))
    if observed != activated:
        raise R274BootstrapHandoffError("r274 activated checkpoint identity drifted")
    if expected_initial_checkpoint is not None and observed != file_identity(
        expected_initial_checkpoint
    ):
        raise R274BootstrapHandoffError("initial learner is not the activated checkpoint")
    gates = dict(receipt.get("runtime_gates") or {})
    if gates != {
        **{name: True for name in RUNTIME_GATE_FIELDS},
        **{name: True for name in TACTICAL_ROUTE_FIELDS},
    }:
        raise R274BootstrapHandoffError("r274 handoff runtime gates changed")
    return {**receipt, "file_identity": identity}


__all__ = [
    "BOOTSTRAP_COMPLETION_SCHEMA",
    "BOOTSTRAP_SUBMISSION_SCHEMA",
    "TACTICAL_ACTIVATION_SCHEMA",
    "R280_TACTICAL_REPAIR_SCHEMA",
    "R274BootstrapHandoffError",
    "file_identity",
    "materialize_bootstrap_submission_checkpoint",
    "materialize_tactical_activation_checkpoint",
    "receipt_digest",
    "validate_bootstrap_checkpoint",
    "validate_full_expert_manifest",
    "validate_r280_gpu_bootstrap_result",
    "validate_handoff_receipt",
    "validate_successful_upload_receipt",
    "validate_tactical_overlay",
    "write_immutable_json",
]
