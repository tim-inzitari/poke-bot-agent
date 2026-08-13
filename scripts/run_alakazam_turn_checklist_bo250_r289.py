#!/usr/bin/env python3
"""Run the isolated, receipt-bound r289 derivative 3×BO250 on Elmo.

This is deliberately a diagnostic runner, not a trainer, promotion gate,
submission tool, service controller, or production entrypoint.  It stages
verified immutable inputs under a content-addressed output directory and then
plays the three separately reported 125-pair deterministic, seed-matched,
seat-swapped cohorts specified by the dedicated derivative contract:

* A: r298 frozen derivative vs unchanged r241/r274 on the new list;
* B: derivative/new list vs immutable r195 native operational configuration;
* C: derivative vs r195 weights/new list, labelled deck-shift confound.

There is intentionally no unseeded fallback, MCTS, RTP, rollout, remote leaf
backend, hidden-state API, training, Kaggle, production selector, or managed
service authority in this file.  A missing or mismatched receipt invalidates
the run before a game can become outcome-eligible.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import socket
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.alakazam_turn_checklist_bo250_r289 import (  # noqa: E402
    CALIBRATION_ELIGIBLE_GATE_NAMES,
    CANDIDATE_ARM,
    CHECKLIST_CHANNELS,
    CHECKLIST_GATE_NAMES,
    COMPARISON_IDS,
    CONTROL_ARM,
    CONTROL_ARMS,
    DERIVATIVE_GOAL_CONTRACT_SHA256,
    EVALUATION_ID,
    R195_NO_RTP_BUNDLE_SHA256,
    R295_CORRECTED_GUIDE_ATTACHMENT_SHA256,
    R289BO250Error,
    R289_SCHEMA,
    EXPECTED_GATE_CHANNEL_MAP,
    FIXED_TRACE_ONLY_GATES,
    GROUPED_RESIDUAL_AGGREGATION_KIND,
    RESIDUAL_GROUP_GATE_MEMBERS,
    build_run_identity,
    build_three_cohort_run_identity,
    build_schedule,
    canonical_digest,
    canonical_json_bytes,
    compile_report,
    compile_three_comparison_report,
    comparison_evaluation_id,
    derive_comparison_seed_identity,
    empty_checklist_telemetry,
    file_identity,
    load_r289_config,
    make_game_receipt,
    output_index_payload,
    read_exact_new_list_deck,
    read_r195_native_deck,
    schedule_identity,
    stage_verified_copy,
    validate_checklist_config,
    validate_derivative_goal_contract,
    validate_game_receipt,
    validate_r298_collision_census_receipt,
    validate_r298_raw_corpus_receipt,
    validate_r298_validation_receipt,
    validate_required_calibration_receipt,
    validate_r293_overlap_audit_receipt,
    validate_r195_contract,
    validate_schedule,
    validate_seeded_engine_receipt,
    write_create_only_json,
)
from poke_bot.seeded_mirror_harness import (  # noqa: E402
    PairFirstPlayerSeal,
    SeededMirrorGameSpec,
    SeededMirrorHarnessError,
    configure_battle_start_seeded,
    validate_pair_first_player_seal,
)


SCRIPT_SCHEMA = "poke_bot.alakazam_turn_checklist_bo250_r289_runner/v1"
PAIR_SEAL_KIND = "pair_first_player_seal"
TRACE_KIND = "candidate_checklist_stage_trace"
PREFLIGHT_KIND = "runtime_preflight"
SCHEDULE_KIND = "seeded_schedule"

DIRECT_POLICY_FLAGS: dict[str, bool] = {
    "rtp": False,
    "search": False,
    "mcts": False,
    "rollout": False,
    "hidden_information_inference": False,
    "candidate_checklist_layer": True,
    "control_checklist_layer": False,
}

# The runner seals the complete importable ``poke_bot`` Python package, not a
# hand-maintained subset.  A frozen checkpoint can reconstruct through several
# transitive modules, so a partial manifest would leave an accidental package
# mismatch invisible.  The source copies are audit-only; execution still uses
# the checked source tree and is rechecked before report compilation.
RUNTIME_SOURCE_EXTRA_PATHS = ("scripts/run_alakazam_turn_checklist_bo250_r289.py",)


class R289RunnerError(RuntimeError):
    """The r289 runner cannot safely credit the requested diagnostic."""


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise R289RunnerError(f"{label} must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R289RunnerError(f"{label} is unreadable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise R289RunnerError(f"{label} must contain a JSON object")
    return payload


def _path(value: str | Path) -> Path:
    # Do not resolve before ``file_identity`` has rejected a symlink.  Resolving
    # first would turn a symlink input into an apparently ordinary file.
    return Path(os.path.abspath(str(Path(value).expanduser())))


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise R289RunnerError(f"{label} must be a bool")
    return bool(value)


def _require_int(value: object, *, label: str, allowed: set[int] | None = None) -> int:
    if type(value) is not int:
        raise R289RunnerError(f"{label} must be an integer")
    result = int(value)
    if allowed is not None and result not in allowed:
        raise R289RunnerError(f"{label} is not one of {sorted(allowed)}")
    return result


def _content_only(value: Any) -> Any:
    """Strip local provenance recursively before serializing a content address."""

    if isinstance(value, Mapping):
        return {
            str(key): _content_only(item)
            for key, item in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_content_only(item) for item in value]
    if isinstance(value, tuple):
        return [_content_only(item) for item in value]
    return value


def _require_elmo_host() -> str:
    """Reject accidental local/Inzi execution before any staged write or reset."""

    hostname = socket.gethostname().strip().casefold()
    primary = hostname.split(".", 1)[0]
    if primary != "elmo" and not primary.startswith("elmo-"):
        raise R289RunnerError(
            f"r289 is Elmo-only; refusing to run on host {hostname or '<unknown>'}"
        )
    return hostname


def _assert_isolated_output_root(path: Path) -> None:
    """Keep receipt writes out of a broad filesystem/repository root."""

    if path.is_symlink():
        raise R289RunnerError("--output-root may not be a symlink")
    if path == Path(path.anchor):
        raise R289RunnerError("--output-root may not be a filesystem root")
    if path == ROOT:
        raise R289RunnerError("--output-root may not be the source repository root")


def _validate_legal_action(observation: Mapping[str, Any], action: Sequence[int]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise R289RunnerError("engine requested an action without a select payload")
    options = select.get("option")
    if not isinstance(options, list):
        raise R289RunnerError("engine select payload has no legal option list")
    try:
        lower = int(select.get("minCount", 0) or 0)
        upper = min(int(select.get("maxCount", 0) or 0), len(options))
    except (TypeError, ValueError) as exc:
        raise R289RunnerError("engine select count bounds are malformed") from exc
    normalized = list(action)
    if not lower <= len(normalized) <= upper:
        raise R289RunnerError("policy returned an illegal action count")
    if len(set(normalized)) != len(normalized) or any(
        type(index) is not int or not 0 <= index < len(options)
        for index in normalized
    ):
        raise R289RunnerError("policy returned an illegal action index")
    return normalized


def _first_player_from_observation(observation: Mapping[str, Any]) -> int | None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    first = current.get("firstPlayer")
    return int(first) if type(first) is int and first in {0, 1} else None


def _forced_turn_order_action(observation: Mapping[str, Any]) -> list[int] | None:
    from poke_bot.features import forced_go_first_action

    try:
        action = forced_go_first_action(dict(observation))
    except Exception as exc:  # noqa: BLE001 - ambiguous setup is unsafe.
        raise R289RunnerError("engine IsFirst prompt is malformed") from exc
    if action is None:
        return None
    return _validate_legal_action(observation, action)


def _runtime_source_identities() -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    paths = sorted((ROOT / "poke_bot").rglob("*.py"))
    paths.extend(ROOT / relative for relative in RUNTIME_SOURCE_EXTRA_PATHS)
    for source in paths:
        relative = source.relative_to(ROOT).as_posix()
        source = ROOT / relative
        identities[relative] = file_identity(source, label=f"r289 runtime source {relative}")
    return identities


def _assert_runtime_sources_unchanged(expected: Mapping[str, Mapping[str, Any]]) -> None:
    current = _runtime_source_identities()
    if dict(current) != {str(key): dict(value) for key, value in expected.items()}:
        raise R289RunnerError(
            "r289 runtime package/source identity changed during the diagnostic"
        )


def _semantic_digest(payload: Mapping[str, Any], field: str) -> str:
    detached = dict(payload)
    detached.pop(field, None)
    return canonical_digest(detached)


def _prepare_calibrated_runtime_config(
    *,
    base_config_path: Path,
    base_config_identity: Mapping[str, Any],
    calibration_receipt_path: Path,
    calibration_receipt_identity: Mapping[str, Any],
    calibration_artifact_path: Path,
    candidate_checkpoint_sha256: str,
    comparison_seed_identities: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Retired legacy r288 activation path.

    Rev4 forbids treating guide/CSV-derived r288 evidence as r298 action or
    logit authority.  Keep this named helper only so old importers fail loudly
    rather than silently reactivating the historical calibration overlay.
    """

    raise R289RunnerError(
        "legacy r288 calibration overlays are retired for r298 BO250; "
        "a receipt-sealed strict r298 stage wrapper is required"
    )

    # Historical implementation intentionally unreachable.  It is retained
    # temporarily only to keep an old source diff auditable; no executable
    # entrypoint reaches it after the fail-closed return above.
    artifact_identity = file_identity(
        calibration_artifact_path, label="r288 calibration artifact"
    )
    receipt = _json_object(calibration_receipt_path, label="r288 calibration receipt")
    artifact = _json_object(calibration_artifact_path, label="r288 calibration artifact")
    receipt_config = receipt.get("config")
    receipt_artifact = receipt.get("artifact")
    if (
        not isinstance(receipt_config, Mapping)
        or not isinstance(receipt_artifact, Mapping)
        or receipt_config.get("file_sha256") != base_config_identity.get("sha256")
        or receipt_artifact.get("file_sha256") != artifact_identity.get("sha256")
        or receipt.get("receipt_sha256") != _semantic_digest(receipt, "receipt_sha256")
    ):
        raise R289RunnerError("r288 calibration receipt does not bind this config/artifact")
    if (
        artifact.get("schema")
        != "poke_bot.alakazam_turn_checklist_gate_calibration_artifact/v1"
        or artifact.get("owner_decision_revision") != 288
        or artifact.get("status") != "offline_validation_only_not_runtime_active"
        or artifact.get("runtime_active") is not False
        or artifact.get("requires_separate_receipt_backed_activation") is not True
        or artifact.get("artifact_sha256")
        != _semantic_digest(artifact, "artifact_sha256")
        or artifact.get("source_disjoint_exact_new_list_data") is not True
    ):
        raise R289RunnerError("r288 calibration artifact is not an inert exact-list receipt")
    normalized_seeds = {str(key): str(value) for key, value in comparison_seed_identities.items()}
    if set(normalized_seeds) != set(COMPARISON_IDS):
        raise R289RunnerError("calibration overlay requires all A/B/C seed identities")
    disjoint = artifact.get("bo250_seed_disjointness")
    frozen_policy = artifact.get("frozen_neural_policy")
    contract = artifact.get("runtime_contract")
    authority = artifact.get("authority")
    if (
        not isinstance(disjoint, Mapping)
        or disjoint.get("bo250_seed_disjoint") is not True
        or not set(normalized_seeds.values()).issubset(
            set(disjoint.get("bo250_seed_identities_excluded") or [])
        )
        or not isinstance(frozen_policy, Mapping)
        or frozen_policy.get("checkpoint_sha256") != candidate_checkpoint_sha256
        or frozen_policy.get("all_neural_model_and_checkpoint_tensors_frozen")
        is not True
        or not isinstance(contract, Mapping)
        or contract.get("config_file_sha256") != base_config_identity.get("sha256")
        or not isinstance(authority, Mapping)
        or any(
            authority.get(key) is not False
            for key in (
                "production_loop",
                "learner_or_checkpoint_training",
                "selector",
                "submission",
                "elmo_production_readmission",
                "managed_service_change",
                "search",
                "rollout",
                "mcts",
                "rtp",
            )
        )
    ):
        raise R289RunnerError(
            "r288 calibration artifact lacks candidate/frozen/BO250-disjoint parity"
        )
    base = _json_object(base_config_path, label="r288 base checklist config")
    runtime = base.get("runtime")
    gates = artifact.get("gates")
    grouped_aggregation = contract.get("grouped_aggregation")
    winner_trace = artifact.get("fitted_group_winner_identity_trace")
    activation_requirements = artifact.get("activation_requirements")
    if not isinstance(runtime, Mapping) or not isinstance(gates, Mapping):
        raise R289RunnerError("r288 calibration runtime/gate payload is malformed")
    gate_order = runtime.get("gate_order")
    eligible_gate_order = runtime.get("residual_enabled_gate_order")
    base_gates = runtime.get("scalar_gates")
    if (
        not isinstance(gate_order, list)
        or tuple(gate_order) != CHECKLIST_GATE_NAMES
        or not isinstance(eligible_gate_order, list)
        or tuple(eligible_gate_order) != CALIBRATION_ELIGIBLE_GATE_NAMES
        or not isinstance(base_gates, Mapping)
        or set(base_gates) != set(CHECKLIST_GATE_NAMES)
        or set(gates) != set(CHECKLIST_GATE_NAMES)
        or contract.get("layer_schema")
        != "poke_bot.alakazam_turn_checklist_heuristic_logit_layer/v1"
        or contract.get("config_file_sha256") != base_config_identity.get("sha256")
        or contract.get("exact_new_list_multiset_sha256")
        != "sha256:a42e047c45c419a599a31f2e20a6209d324558082f27e12091ade8918376d182"
        or contract.get("corrected_guide_attachment_sha256")
        != R295_CORRECTED_GUIDE_ATTACHMENT_SHA256
        or contract.get("corrected_guide_exact_inventory")
        != {"pokemon": 17, "trainers": 36, "energy": 7, "alakazam": 3}
        or contract.get("channel_order") != list(CHECKLIST_CHANNELS)
        or contract.get("guide_support_channel") != "guide_support"
        or contract.get("gate_order") != list(CHECKLIST_GATE_NAMES)
        or contract.get("fitted_gate_order") != list(CALIBRATION_ELIGIBLE_GATE_NAMES)
        or contract.get("fixed_trace_only_gates") != FIXED_TRACE_ONLY_GATES
        or contract.get("gate_channel_map") != EXPECTED_GATE_CHANNEL_MAP
        or not isinstance(grouped_aggregation, Mapping)
        or grouped_aggregation.get("kind") != GROUPED_RESIDUAL_AGGREGATION_KIND
        or grouped_aggregation.get("group_order")
        != list(RESIDUAL_GROUP_GATE_MEMBERS)
        or grouped_aggregation.get("group_gate_members")
        != {
            name: list(gates)
            for name, gates in RESIDUAL_GROUP_GATE_MEMBERS.items()
        }
        or grouped_aggregation.get("tie_break") != "runtime.channel_order_first"
        or grouped_aggregation.get("exact_zero_group_winner") is not None
        or grouped_aggregation.get("final_aggregation")
        != "sum_group_winner_contributions_then_global_clamp"
        or contract.get("vector_stage")
        != "centered_linf_normalized_pre_gate"
        or contract.get("total_residual_cap") != 0.1
        or contract.get("gate_bounds") != {"minimum": 0.0, "maximum": 0.1}
        or contract.get("guide_support_trace_only") is not True
        or contract.get("guide_support_calibration_eligible") is not False
        or contract.get("post_deduplication_vectors_required") is not True
        or not isinstance(winner_trace, Mapping)
        or winner_trace.get("group_order") != list(RESIDUAL_GROUP_GATE_MEMBERS)
        or winner_trace.get("tie_break") != "runtime.channel_order_first"
        or winner_trace.get("exact_zero_group_winner") is not None
        or not isinstance(winner_trace.get("rows"), list)
        or not isinstance(winner_trace.get("rows_sha256"), str)
        or winner_trace.get("rows_sha256")
        != canonical_digest(winner_trace.get("rows"))
        or not isinstance(activation_requirements, Mapping)
        or activation_requirements.get("runtime_activation_authority") is not False
        or activation_requirements.get("bo250_bound_calibration_receipt_required_before_launch")
        is not True
        or activation_requirements.get("current_receipt_is_bo250_bound") is not True
        or activation_requirements.get("unbound_artifact_may_launch_bo250") is not False
    ):
        raise R289RunnerError(
            "r288 calibration artifact does not bind the six-gate trace-only runtime contract"
        )
    normalized_eligible_gates: dict[str, float] = {}
    for name in CHECKLIST_GATE_NAMES:
        value = gates.get(name)
        if type(value) is bool or not isinstance(value, (int, float)):
            raise R289RunnerError(f"r288 calibrated gate {name} is non-numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 0.10:
            raise R289RunnerError(f"r288 calibrated gate {name} exceeds its bounds")
        if name in FIXED_TRACE_ONLY_GATES:
            if number != FIXED_TRACE_ONLY_GATES[name]:
                raise R289RunnerError(
                    f"r288 trace-only gate {name} must remain exact zero in calibration"
                )
            continue
        if name not in CALIBRATION_ELIGIBLE_GATE_NAMES:
            raise R289RunnerError(f"r288 calibration has an unknown eligible gate {name}")
        normalized_eligible_gates[name] = number
    derived = json.loads(json.dumps(base))
    derived_runtime = derived.get("runtime")
    if not isinstance(derived_runtime, dict):  # protected by base config validation.
        raise R289RunnerError("r288 base runtime unexpectedly changed while deriving gates")
    derived_scalar_gates = derived_runtime.get("scalar_gates")
    if not isinstance(derived_scalar_gates, dict):
        raise R289RunnerError("r288 base scalar-gate mapping unexpectedly changed")
    for name in CALIBRATION_ELIGIBLE_GATE_NAMES:
        derived_scalar_gates[name] = normalized_eligible_gates[name]
    for name, fixed_value in FIXED_TRACE_ONLY_GATES.items():
        if derived_scalar_gates.get(name) != fixed_value:
            raise R289RunnerError(
                f"r288 base trace-only gate {name} is not exact zero before overlay"
            )
        # Preserve the base's fixed value rather than copying it from a
        # fitted artifact.  This makes the six-value activation boundary
        # explicit in the generated Elmo-only config.
        derived_scalar_gates[name] = fixed_value
    derived["r289_elmo_calibration_overlay"] = {
        "runtime_selected_explicitly_by": EVALUATION_ID,
        "base_config_file_sha256": base_config_identity["sha256"],
        "calibration_receipt_file_sha256": calibration_receipt_identity["sha256"],
        "calibration_artifact_file_sha256": artifact_identity["sha256"],
        "calibration_artifact_semantic_sha256": artifact["artifact_sha256"],
        "comparison_seed_identities": normalized_seeds,
        "eligible_gate_order": list(CALIBRATION_ELIGIBLE_GATE_NAMES),
        "fixed_trace_only_gates": dict(FIXED_TRACE_ONLY_GATES),
        "elmo_only": True,
        "training_eligible": False,
        "promotion_authority": False,
        "production_authority": False,
    }
    encoded = canonical_json_bytes(derived)
    virtual_identity = {
        "path": "generated:r289-elmo-calibrated-checklist-runtime-config.json",
        "sha256": canonical_digest(derived),
        "size_bytes": len(encoded),
    }
    return derived, virtual_identity, artifact_identity


def _assert_default_uncalibrated_gates(path: Path) -> None:
    """Retired legacy r288 runtime activation guard.

    Any caller attempting the pre-rev4 direct r288 runtime path is blocked,
    including the old nominal 0.01 defaults.  The live three-cohort entrypoint
    separately requires every legacy gate to be exact zero for trace-only
    evidence and cannot invoke this helper.
    """

    raise R289RunnerError(
        "legacy r288 runtime activation is prohibited in r298 BO250"
    )

    payload = _json_object(path, label="uncalibrated r288 checklist config")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise R289RunnerError("uncalibrated r288 runtime section is malformed")
    gate_order = runtime.get("gate_order")
    eligible_gate_order = runtime.get("residual_enabled_gate_order")
    gates = runtime.get("scalar_gates")
    if (
        not isinstance(gate_order, list)
        or tuple(gate_order) != CHECKLIST_GATE_NAMES
        or not isinstance(eligible_gate_order, list)
        or tuple(eligible_gate_order) != CALIBRATION_ELIGIBLE_GATE_NAMES
        or not isinstance(gates, Mapping)
        or set(gates) != set(CHECKLIST_GATE_NAMES)
    ):
        raise R289RunnerError("uncalibrated r288 gate inventory is malformed")
    if "r289_elmo_calibration_overlay" in payload:
        raise R289RunnerError("calibrated checklist config requires its artifact and receipt")
    for name in CALIBRATION_ELIGIBLE_GATE_NAMES:
        value = gates.get(name)
        if type(value) is bool or not isinstance(value, (int, float)) or float(value) != 0.01:
            raise R289RunnerError(
                "non-default checklist gates require a disjoint r288 calibration artifact"
            )
    for name, expected in FIXED_TRACE_ONLY_GATES.items():
        value = gates.get(name)
        if type(value) is bool or not isinstance(value, (int, float)) or value != expected:
            raise R289RunnerError(
                f"trace-only r288 gate {name} must remain exact zero"
            )


@contextmanager
def _direct_policy_environment() -> Iterator[dict[str, list[str]]]:
    """Suppress inherited action authorities in this isolated process only.

    The explicit ``PolicyAgent`` constructor below remains the source of truth;
    no ambient setting can turn on RTP, search, remote inference, or the
    checklist control arm.  Original values are restored on normal exit so
    importing/running the runner cannot mutate a caller's interactive shell.
    """

    def forbidden(key: str) -> bool:
        upper = key.upper()
        prefixes = (
            "POKEBOT_RTP",
            "POKEBOT_USE_RECURSIVE",
            "POKEBOT_MCTS",
            "POKEBOT_SEARCH",
            "POKEBOT_ROLLOUT",
            "POKEBOT_POKE_RLM",
            "POKEBOT_SLOWKING",
            "POKEBOT_GUIDE",
            "POKEBOT_ALAKAZAM_TURN_CHECKLIST",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME",
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH",
            "POKEBOT_COMBO_STATE_ROUTE",
        )
        exact = {
            "POKEBOT_LIBCG_PATH",
            "POKEBOT_ALLOW_ORACLE_DECK",
            "POKEBOT_ORACLE_MODE",
        }
        return upper in exact or upper.startswith(prefixes)

    previous: dict[str, str] = {}
    removed: list[str] = []
    for key in sorted(list(os.environ)):
        if forbidden(key):
            previous[key] = os.environ.pop(key)
            removed.append(key)
    forced = {"POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0"}
    for key, value in forced.items():
        if key in os.environ:
            previous[key] = os.environ[key]
        os.environ[key] = value
    try:
        yield {"removed": removed, "forced": sorted(forced)}
    finally:
        for key in forced:
            os.environ.pop(key, None)
        os.environ.update(previous)


def _load_seeded_engine(path: Path) -> tuple[Any, dict[str, Any]]:
    """Load exactly the staged native library and configure only its safe ABI."""

    try:
        from poke_bot import cg_env

        cg_env.ensure_cg_importable()
        from cg import sim  # type: ignore
    except Exception as exc:  # noqa: BLE001 - native type import is required.
        raise R289RunnerError("cannot import the local libcg ctypes contract") from exc
    try:
        lib = ctypes.cdll.LoadLibrary(str(path))
        lib.GameInitialize.restype = None
        lib.GameInitialize.argtypes = []
        lib.GameInitialize()
        configure_battle_start_seeded(lib, sim.StartData)
        lib.BattleFinish.restype = None
        lib.BattleFinish.argtypes = [ctypes.c_void_p]
        lib.GetBattleData.restype = sim.SerialData
        lib.GetBattleData.argtypes = [ctypes.c_void_p]
        lib.Select.restype = ctypes.c_int
        lib.Select.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
    except (AttributeError, OSError, SeededMirrorHarnessError) as exc:
        raise R289RunnerError(
            "staged engine lacks the required BattleStartSeeded direct-policy ABI"
        ) from exc
    if hasattr(lib, "GetHiddenSnapshot") or hasattr(lib, "HiddenSnapshotAbiVersion"):
        raise R289RunnerError("staged engine exposes a forbidden hidden-state API")
    return lib, {
        "engine_path": str(path),
        "battle_start_seeded_available": True,
        "hidden_snapshot_api_available": False,
        "called_apis": [
            "GameInitialize",
            "BattleStartSeeded",
            "GetBattleData",
            "Select",
            "BattleFinish",
        ],
    }


def _model_fingerprint(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    bank = getattr(model, "matchup_adapter_bank", None)
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "training": bool(getattr(model, "training", True)),
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "own_deck_ledger_enabled": bool(
            getattr(model, "own_deck_ledger_enabled", False)
        ),
        "decision_fusion_enabled": bool(
            getattr(model, "decision_fusion_enabled", False)
        ),
        "matchup_adapter_bank_present": bank is not None,
        "matchup_adapter_bank_enabled": bool(getattr(bank, "enabled", False)),
    }


def _assert_direct_policy(
    policy: Any,
    *,
    arm: str,
    checklist_enabled: bool,
    checklist_config: Path | None,
    matchup_tree: Path,
) -> dict[str, Any]:
    if bool(getattr(policy, "use_mcts", False)):
        raise R289RunnerError(f"{arm} unexpectedly enabled MCTS")
    if bool(getattr(policy, "use_recursive_turn_planner", False)) or getattr(
        policy, "_rtp_bridge", None
    ) is not None:
        raise R289RunnerError(f"{arm} unexpectedly enabled RTP")
    if bool(getattr(policy, "oracle_mode", False)) or bool(
        getattr(policy, "belief_mcts", False)
    ):
        raise R289RunnerError(f"{arm} unexpectedly enabled oracle/belief search")
    if getattr(policy, "leaf_backend", None) is not None:
        raise R289RunnerError(f"{arm} unexpectedly enabled a remote leaf backend")
    if bool(getattr(policy, "sample_actions", True)):
        raise R289RunnerError(f"{arm} unexpectedly enables stochastic action sampling")
    if bool(getattr(policy, "turn_checklist_logit_layer_enabled", False)) != checklist_enabled:
        raise R289RunnerError(f"{arm} checklist-layer setting drifted")
    if not bool(getattr(policy, "matchup_adapter_runtime", False)):
        raise R289RunnerError(f"{arm} did not activate its explicit matchup tree")
    actual_tree = Path(str(getattr(policy, "matchup_adapter_tree_path", ""))).resolve()
    if actual_tree != matchup_tree.resolve():
        raise R289RunnerError(f"{arm} matchup tree path drifted at runtime")
    if checklist_enabled:
        if checklist_config is None:
            raise R289RunnerError("candidate lacks an explicit checklist config")
        actual_config = Path(
            str(getattr(policy, "turn_checklist_logit_layer_config_path", ""))
        ).resolve()
        if actual_config != checklist_config.resolve():
            raise R289RunnerError("candidate checklist config path drifted at runtime")
    elif getattr(policy, "turn_checklist_logit_layer_config_path", None) is not None:
        raise R289RunnerError("r195 control unexpectedly has a checklist config")
    for config_name, bridge_name in (
        ("poke_rlm_config", "_poke_rlm_bridge"),
        ("slowking_distill_config", "_slowking_distill_bridge"),
    ):
        config = getattr(policy, config_name, None)
        if config is not None and bool(getattr(config, "selects_actions", False)):
            raise R289RunnerError(f"{arm} unexpectedly grants {config_name} action authority")
        if getattr(policy, bridge_name, None) is not None:
            raise R289RunnerError(f"{arm} unexpectedly initialized {bridge_name}")
    model = getattr(policy, "model", None)
    if model is None or bool(getattr(model, "training", True)):
        raise R289RunnerError(f"{arm} model is not frozen/eval-only")
    fingerprint = _model_fingerprint(model)
    if fingerprint["trainable_parameter_count"] != 0:
        raise R289RunnerError(f"{arm} model still has trainable parameters")
    return {
        "arm": arm,
        "checklist_layer_enabled": checklist_enabled,
        "matchup_tree_path": str(matchup_tree.resolve()),
        "strict_runtime": bool(getattr(policy, "strict_runtime", False)),
        "model": fingerprint,
    }


def _load_direct_policies(
    *,
    candidate_checkpoint: Path,
    control_checkpoint: Path,
    candidate_deck: Sequence[int],
    control_deck: Sequence[int],
    candidate_tree: Path,
    control_tree: Path,
    checklist_config: Path,
    device_name: str,
    control_arm: str,
) -> tuple[Any, Any, dict[str, Any]]:
    try:
        import torch
        from poke_bot.agent import PolicyAgent
        from poke_bot.train import load_model_from_checkpoint
    except Exception as exc:  # noqa: BLE001 - a model runtime is mandatory.
        raise R289RunnerError("r289 requires an importable local Torch policy runtime") from exc
    try:
        device = torch.device(device_name)
        candidate_model = load_model_from_checkpoint(candidate_checkpoint, device=device)
        control_model = load_model_from_checkpoint(control_checkpoint, device=device)
        for model in (candidate_model, control_model):
            model.eval()
            model.requires_grad_(False)
        candidate = PolicyAgent(
            model=candidate_model,
            deck=list(candidate_deck),
            use_mcts=False,
            use_recursive_turn_planner=False,
            oracle_mode=False,
            belief_mcts=False,
            max_sims=0,
            move_time_s=0.0,
            collect_targets=False,
            sample_actions=False,
            leaf_backend=None,
            strict_runtime=True,
            matchup_adapter_runtime=True,
            matchup_adapter_tree_path=str(candidate_tree),
            turn_checklist_logit_layer_enabled=True,
            turn_checklist_logit_layer_config_path=str(checklist_config),
        )
        control = PolicyAgent(
            model=control_model,
            deck=list(control_deck),
            use_mcts=False,
            use_recursive_turn_planner=False,
            oracle_mode=False,
            belief_mcts=False,
            max_sims=0,
            move_time_s=0.0,
            collect_targets=False,
            sample_actions=False,
            leaf_backend=None,
            strict_runtime=True,
            matchup_adapter_runtime=True,
            matchup_adapter_tree_path=str(control_tree),
            turn_checklist_logit_layer_enabled=False,
            turn_checklist_logit_layer_config_path=None,
        )
    except Exception as exc:  # noqa: BLE001 - no fallback policy is permitted.
        raise R289RunnerError("receipt-bound direct policies could not be constructed") from exc
    candidate_runtime = _assert_direct_policy(
        candidate,
        arm=CANDIDATE_ARM,
        checklist_enabled=True,
        checklist_config=checklist_config,
        matchup_tree=candidate_tree,
    )
    control_runtime = _assert_direct_policy(
        control,
        arm=control_arm,
        checklist_enabled=False,
        checklist_config=None,
        matchup_tree=control_tree,
    )
    if not candidate_runtime["model"]["own_deck_ledger_enabled"]:
        raise R289RunnerError("r298 candidate lost its frozen new-list OwnDeck runtime")
    return candidate, control, {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "candidate": candidate_runtime,
        "control": control_runtime,
    }


def _nonzero(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_nonzero(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonzero(item) for item in value)
    if type(value) is bool or value is None:
        return False
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise R289RunnerError("checklist telemetry contains a non-finite value")
        return abs(number) > 0.0
    raise R289RunnerError("checklist telemetry contains a non-numeric score")


def _assert_residual_cap(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_residual_cap(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_residual_cap(item)
        return
    if type(value) is bool or not isinstance(value, (int, float)):
        raise R289RunnerError("checklist residual telemetry is malformed")
    number = float(value)
    if not math.isfinite(number) or abs(number) > 0.1000001:
        raise R289RunnerError("checklist residual exceeded its immutable 0.10 cap")


def _require_exact_stage_gate_boundary(stage: Mapping[str, Any]) -> None:
    """Prove Q5/Q6/guide were trace-only in this applied stage."""

    gates = stage.get("scalar_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(CHECKLIST_GATE_NAMES):
        raise R289RunnerError("candidate checklist stage has no exact scalar-gate inventory")
    for name in CHECKLIST_GATE_NAMES:
        value = gates.get(name)
        if type(value) is bool or not isinstance(value, (int, float)):
            raise R289RunnerError(f"candidate checklist gate {name} is malformed")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 0.10:
            raise R289RunnerError(f"candidate checklist gate {name} exceeds bounds")
        if name in FIXED_TRACE_ONLY_GATES and number != FIXED_TRACE_ONLY_GATES[name]:
            raise R289RunnerError(
                f"candidate trace-only checklist gate {name} is nonzero"
            )
    # This contract makes it impossible for a stale config to introduce a
    # nonzero Q5/Q6/guide residual after the offline receipt was issued.
    eligible = [
        name for name in CHECKLIST_GATE_NAMES if name not in FIXED_TRACE_ONLY_GATES
    ]
    if eligible != list(CALIBRATION_ELIGIBLE_GATE_NAMES):
        raise R289RunnerError("candidate checklist six-gate contract drifted")


def _channel_available(
    stage: Mapping[str, Any], channel: Mapping[str, Any], name: str
) -> bool:
    available = stage.get("available")
    if isinstance(available, Mapping):
        value = available.get(name, channel.get("available", False))
    elif isinstance(available, list) and len(available) == len(CHECKLIST_CHANNELS):
        value = available[CHECKLIST_CHANNELS.index(name)]
    else:
        value = channel.get("available", available if type(available) is bool else False)
    if type(value) is not bool:
        raise R289RunnerError(f"checklist availability for {name} is malformed")
    return bool(value)


def _validate_grouped_stage_residuals(
    *,
    stage: Mapping[str, Any],
    channels: Mapping[str, Mapping[str, Any]],
    width: int,
) -> None:
    """Recompute the r293 grouped residual from recorded causal vectors.

    This is deliberately independent of the checklist implementation: the
    candidate trace must demonstrate that its actual raw residual retained at
    most the strongest absolute gated contribution in each approved group
    (with deterministic channel-order ties), before the agent applies its
    selected-path whole-decision budget.
    """

    gates = stage["scalar_gates"]
    if not isinstance(gates, Mapping):  # protected earlier; keep local proof.
        raise R289RunnerError("candidate scalar gates are unavailable for grouped audit")
    raw_residuals = stage.get("module_residuals_before_whole_decision_cap")
    applied_residuals = stage.get("residuals")
    if (
        not isinstance(raw_residuals, list)
        or not isinstance(applied_residuals, list)
        or len(raw_residuals) != width
        or len(applied_residuals) != width
    ):
        raise R289RunnerError("candidate grouped residual vectors are misaligned")
    before = float(stage["whole_decision_budget_before"])
    for option_index in range(width):
        expected_raw = 0.0
        for members in RESIDUAL_GROUP_GATE_MEMBERS.values():
            winner = 0.0
            # ``members`` is already in the immutable channel-order tie
            # order, so a strict comparison retains the first exact tie.
            for gate_name in members:
                channel_name = EXPECTED_GATE_CHANNEL_MAP[gate_name]
                normalized = channels[channel_name].get("normalized")
                if not isinstance(normalized, list) or len(normalized) != width:
                    raise R289RunnerError(
                        f"candidate normalized channel {channel_name} is misaligned"
                    )
                value = normalized[option_index]
                if type(value) is bool or not isinstance(value, (int, float)):
                    raise R289RunnerError(
                        f"candidate normalized channel {channel_name} is non-numeric"
                    )
                contribution = float(gates[gate_name]) * float(value)
                if not math.isfinite(contribution):
                    raise R289RunnerError("candidate grouped contribution is non-finite")
                if abs(contribution) > abs(winner):
                    winner = contribution
            expected_raw += winner
        expected_raw = max(-0.10, min(0.10, expected_raw))
        value = raw_residuals[option_index]
        if type(value) is bool or not isinstance(value, (int, float)):
            raise R289RunnerError("candidate module residual is non-numeric")
        actual_raw = float(value)
        if not math.isfinite(actual_raw) or abs(actual_raw - expected_raw) > 1e-6:
            raise R289RunnerError(
                "candidate module residual does not match r293 grouped aggregation"
            )
        expected_applied = max(-before, min(before, actual_raw))
        applied = applied_residuals[option_index]
        if type(applied) is bool or not isinstance(applied, (int, float)):
            raise R289RunnerError("candidate applied residual is non-numeric")
        if not math.isfinite(float(applied)) or abs(float(applied) - expected_applied) > 1e-6:
            raise R289RunnerError(
                "candidate applied residual does not match whole-decision budget"
            )


def _validate_r293_stage_overlap(stage: Mapping[str, Any]) -> None:
    """Require the live r293 per-channel de-duplication audit.

    The r288 residual is intentionally additive to frozen neural/Fusion/
    OwnDeck/Adapter paths.  Revision 293 permits any attenuation only inside
    this layer, so a BO250 stage without the new layer's audit cannot be
    outcome-eligible even when its action is otherwise legal.
    """

    audit = stage.get("channel_overlap_audit")
    if not isinstance(audit, Mapping) or set(audit) != set(CHECKLIST_CHANNELS):
        raise R289RunnerError("candidate trace lacks the r293 per-channel overlap audit")
    for name in CHECKLIST_CHANNELS:
        row = audit[name]
        if (
            not isinstance(row, Mapping)
            or not isinstance(
                row.get("existing_route_overlap_or_distinct_reason"), str
            )
            or not row["existing_route_overlap_or_distinct_reason"].strip()
            or not isinstance(
                row.get("attenuation_or_suppression_decision"), str
            )
            or not row["attenuation_or_suppression_decision"].strip()
            or "post_deduplication_signed_residual" not in row
        ):
            raise R289RunnerError(f"candidate r293 overlap audit is malformed for {name}")
        _assert_residual_cap(row["post_deduplication_signed_residual"])
    _require_bool(
        stage.get("guide_support_trace_only"), label="checklist guide trace-only gate"
    )
    guide_residual = stage.get("guide_support_runtime_residual")
    if isinstance(guide_residual, (Mapping, list, tuple)) and not guide_residual:
        raise R289RunnerError("legacy broad guide residual may not be empty")
    _assert_residual_cap(guide_residual)
    if _nonzero(guide_residual):
        raise R289RunnerError("legacy broad guide changed an r289 policy logit")


def _record_candidate_trace(
    *,
    telemetry: dict[str, Any],
    trace: object,
    selected_action: Sequence[int],
    game_step: int,
) -> dict[str, Any]:
    """Validate and aggregate the policy-provided factorized checklist audit."""

    if not isinstance(trace, Mapping) or trace.get("enabled") is not True:
        raise R289RunnerError("candidate made a decision without an enabled checklist trace")
    selected = trace.get("selected_action")
    if not isinstance(selected, list) or selected != list(selected_action):
        raise R289RunnerError("candidate checklist trace does not bind its selected action")
    stages = trace.get("stages")
    if not isinstance(stages, list) or not stages:
        raise R289RunnerError("candidate checklist trace does not expose factorized stages")
    top_budget: dict[str, float] = {}
    for name in (
        "whole_decision_budget_initial",
        "whole_decision_budget_consumed",
        "whole_decision_budget_remaining",
    ):
        value = trace.get(name)
        if type(value) is bool or not isinstance(value, (int, float)):
            raise R289RunnerError(f"candidate {name} is malformed")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 0.1000001:
            raise R289RunnerError(f"candidate {name} exceeds whole-decision cap")
        top_budget[name] = number
    if top_budget["whole_decision_budget_initial"] != 0.10:
        raise R289RunnerError("candidate whole-decision budget initial value drifted")
    telemetry["policy_decisions"] += 1
    stable_stages: list[dict[str, Any]] = []
    expected_budget_before = 0.10
    total_consumed = 0.0
    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            raise R289RunnerError("candidate checklist stage is not an object")
        stage = dict(raw_stage)
        required = {
            "enabled",
            "applied",
            "evaluation_mode",
            "action_authority",
            "channels",
            "guide_support",
            "scalar_gates",
            "module_residuals_before_whole_decision_cap",
            "residuals",
            "facts",
            "available",
            "reason",
            "active",
            "score_space",
            "base_argmax_index",
            "adjusted_argmax_index",
            "action_changed_from_base_policy",
            "channel_overlap_audit",
            "guide_support_trace_only",
            "guide_support_runtime_residual",
            "whole_decision_budget_initial",
            "whole_decision_budget_before",
            "whole_decision_budget_consumed",
            "whole_decision_budget_remaining",
            "factorized_stage_index",
            "factorized_prefix",
            "candidate_rows",
            "selected_candidate_index",
            "selected_stage_index",
            "selected_candidate",
            "stage_argmax_changed_at_prefix",
        }
        missing = sorted(required - set(stage))
        if missing:
            raise R289RunnerError(
                "candidate checklist stage is missing stable telemetry fields: "
                + ", ".join(missing)
            )
        if stage.get("applied") is False or stage.get("reason") == "checklist_layer_unavailable":
            raise R289RunnerError("candidate checklist layer fell back instead of applying")
        if stage.get("score_space") != "local_logits":
            raise R289RunnerError("direct-only r289 candidate attempted remote policy priors")
        for key in ("base_argmax_index", "adjusted_argmax_index"):
            value = stage.get(key)
            if type(value) is not int or value < 0:
                raise R289RunnerError(f"candidate checklist {key} is malformed")
        _require_bool(
            stage.get("stage_argmax_changed_at_prefix"),
            label="checklist stage argmax-change",
        )
        if (
            stage.get("action_changed_from_base_policy")
            != stage.get("stage_argmax_changed_at_prefix")
        ):
            raise R289RunnerError("deprecated checklist action-change alias drifted")
        _require_bool(stage.get("active"), label="checklist active")
        if stage.get("evaluation_mode") != "evaluated" or stage.get("action_authority") is not True:
            raise R289RunnerError("candidate checklist stage was not action-authoritative")
        stage_index = stage.get("factorized_stage_index")
        if type(stage_index) is not int or stage_index != index:
            raise R289RunnerError("candidate factorized stage index drifted")
        if (
            not isinstance(stage.get("factorized_prefix"), list)
            or not isinstance(stage.get("candidate_rows"), list)
            or not isinstance(stage.get("selected_candidate_index"), int)
            or not isinstance(stage.get("selected_stage_index"), int)
            or not isinstance(stage.get("selected_candidate"), list)
        ):
            raise R289RunnerError("candidate factorized stage selection telemetry is malformed")
        rows = stage["candidate_rows"]
        selected_index = stage["selected_candidate_index"]
        if (
            selected_index != stage["selected_stage_index"]
            or not 0 <= selected_index < len(rows)
            or not all(isinstance(row, list) for row in rows)
            or rows[selected_index] != stage["selected_candidate"]
        ):
            raise R289RunnerError("candidate selected factorized row does not bind trace")
        for name in (
            "whole_decision_budget_initial",
            "whole_decision_budget_before",
            "whole_decision_budget_consumed",
            "whole_decision_budget_remaining",
        ):
            value = stage.get(name)
            if type(value) is bool or not isinstance(value, (int, float)):
                raise R289RunnerError(f"candidate {name} is malformed")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 0.1000001:
                raise R289RunnerError(f"candidate {name} exceeds whole-decision cap")
        if stage.get("whole_decision_budget_initial") != 0.10:
            raise R289RunnerError("candidate whole-decision budget initial value drifted")
        budget_before = float(stage["whole_decision_budget_before"])
        budget_consumed = float(stage["whole_decision_budget_consumed"])
        budget_remaining = float(stage["whole_decision_budget_remaining"])
        if abs(budget_before - expected_budget_before) > 1e-8:
            raise R289RunnerError("candidate stage whole-decision budget sequence drifted")
        _require_exact_stage_gate_boundary(stage)
        _validate_r293_stage_overlap(stage)
        raw_channels = stage.get("channels")
        channel_names = stage.get("channel_names")
        if (
            not isinstance(raw_channels, list)
            or channel_names != list(CHECKLIST_CHANNELS)
            or len(raw_channels) != len(CHECKLIST_CHANNELS)
        ):
            raise R289RunnerError("candidate trace does not report all eight checklist channels")
        channels: dict[str, Mapping[str, Any]] = {}
        for raw_channel in raw_channels:
            if not isinstance(raw_channel, Mapping):
                raise R289RunnerError("candidate trace channel is not an object")
            name = raw_channel.get("name")
            if not isinstance(name, str) or name in channels:
                raise R289RunnerError("candidate trace channel name is malformed")
            channels[name] = raw_channel
        if set(channels) != set(CHECKLIST_CHANNELS):
            raise R289RunnerError("candidate trace channel names drifted")
        guide = stage.get("guide_support")
        if not isinstance(guide, Mapping) or set(guide) < {"raw", "normalized"}:
            raise R289RunnerError("candidate trace guide-support telemetry is malformed")
        if not isinstance(stage.get("scalar_gates"), Mapping):
            raise R289RunnerError("candidate trace scalar gates are malformed")
        if not isinstance(stage.get("facts"), Mapping):
            raise R289RunnerError("candidate trace public facts are malformed")
        _assert_residual_cap(stage["module_residuals_before_whole_decision_cap"])
        _assert_residual_cap(stage["residuals"])
        if (
            not isinstance(stage["module_residuals_before_whole_decision_cap"], list)
            or len(stage["module_residuals_before_whole_decision_cap"]) != len(rows)
            or not isinstance(stage["residuals"], list)
            or len(stage["residuals"]) != len(rows)
        ):
            raise R289RunnerError("candidate applied residuals do not align with factorized rows")
        _validate_grouped_stage_residuals(
            stage=stage,
            channels=channels,
            width=len(rows),
        )
        expected_consumed = abs(float(stage["residuals"][selected_index]))
        if abs(budget_consumed - expected_consumed) > 1e-8:
            raise R289RunnerError("candidate stage budget consumption does not bind selected residual")
        expected_remaining = max(0.0, budget_before - budget_consumed)
        if abs(budget_remaining - expected_remaining) > 1e-8:
            raise R289RunnerError("candidate stage whole-decision remaining budget drifted")
        expected_budget_before = budget_remaining
        total_consumed += budget_consumed
        telemetry["factorized_stages"] += 1
        if stage["active"]:
            telemetry["active_stages"] += 1
        if _nonzero(stage["residuals"]):
            telemetry["residual_nonzero_stages"] += 1
        if stage["stage_argmax_changed_at_prefix"]:
            telemetry["action_changed_stages"] += 1
        any_unavailable = False
        for name in CHECKLIST_CHANNELS:
            channel = channels[name]
            if not isinstance(channel, Mapping) or set(channel) < {"raw", "normalized"}:
                raise R289RunnerError(f"candidate trace channel {name} is malformed")
            available = _channel_available(stage, channel, name)
            if available:
                telemetry["channels"][name]["available_stages"] += 1
            else:
                telemetry["channels"][name]["unavailable_stages"] += 1
                any_unavailable = True
            if _nonzero(channel["raw"]):
                telemetry["channels"][name]["raw_nonzero_stages"] += 1
            if _nonzero(channel["normalized"]):
                telemetry["channels"][name]["normalized_nonzero_stages"] += 1
        if any_unavailable:
            telemetry["unavailable_stages"] += 1
        if _nonzero(guide["raw"]):
            telemetry["guide_support"]["raw_nonzero_stages"] += 1
        if _nonzero(guide["normalized"]):
            telemetry["guide_support"]["normalized_nonzero_stages"] += 1
        stable_stages.append(stage)
    stable = {
        "game_step": int(game_step),
        "selected_action": list(selected_action),
        "stages": stable_stages,
        "whole_decision_budget_initial": top_budget[
            "whole_decision_budget_initial"
        ],
        "whole_decision_budget_consumed": top_budget[
            "whole_decision_budget_consumed"
        ],
        "whole_decision_budget_remaining": top_budget[
            "whole_decision_budget_remaining"
        ],
    }
    if abs(top_budget["whole_decision_budget_consumed"] - total_consumed) > 1e-8:
        raise R289RunnerError("candidate whole-decision consumed budget differs from stages")
    if abs(top_budget["whole_decision_budget_remaining"] - expected_budget_before) > 1e-8:
        raise R289RunnerError("candidate whole-decision remaining budget differs from stages")
    # Verify the full trace is canonical JSON now, while a failed turn is still
    # ineligible.  This avoids writing a receipt whose trace cannot be audited.
    canonical_json_bytes(stable)
    return stable


def _pair_specs(schedule: Sequence[SeededMirrorGameSpec], pair_index: int) -> tuple[SeededMirrorGameSpec, SeededMirrorGameSpec]:
    rows = sorted(
        (spec for spec in schedule if spec.pair_index == pair_index),
        key=lambda spec: spec.game_index,
    )
    if len(rows) != 2 or [row.game_index for row in rows] != [0, 1]:
        raise R289RunnerError(f"r289 schedule pair {pair_index} is malformed")
    return rows[0], rows[1]


def _seal_document(
    *,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
    seal: PairFirstPlayerSeal,
    setup_actions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    actions = [list(action) for action in setup_actions]
    return {
        "schema": SCRIPT_SCHEMA,
        "kind": PAIR_SEAL_KIND,
        "evaluation_id": seal.evaluation_id,
        "run_identity_sha256": run_identity_sha256,
        "runtime_preflight_sha256": runtime_preflight_sha256,
        "pair_first_player_seal": seal.as_payload(),
        "setup_actions": actions,
        "setup_actions_sha256": canonical_digest(actions),
        "training_eligible": False,
        "promotion_authority": False,
        "production_authority": False,
    }


def _decode_pair_seal(
    payload: Mapping[str, Any], *, pair: Sequence[SeededMirrorGameSpec], run_identity_sha256: str, runtime_preflight_sha256: str) -> PairFirstPlayerSeal:
    if (
        payload.get("schema") != SCRIPT_SCHEMA
        or payload.get("kind") != PAIR_SEAL_KIND
        or payload.get("evaluation_id") != pair[0].evaluation_id
        or payload.get("run_identity_sha256") != run_identity_sha256
        or payload.get("runtime_preflight_sha256") != runtime_preflight_sha256
        or payload.get("training_eligible") is not False
        or payload.get("promotion_authority") is not False
        or payload.get("production_authority") is not False
    ):
        raise R289RunnerError("persisted pair first-player seal has foreign authority/binding")
    actions = payload.get("setup_actions")
    if not isinstance(actions, list) or payload.get("setup_actions_sha256") != canonical_digest(actions):
        raise R289RunnerError("persisted pair first-player setup actions drifted")
    raw = payload.get("pair_first_player_seal")
    if not isinstance(raw, Mapping):
        raise R289RunnerError("persisted pair seal has no PairFirstPlayerSeal payload")
    expected_keys = {
        "evaluation_id",
        "pair_index",
        "pair_id",
        "pair_nonce_sha256",
        "engine_seed_u32",
        "deck_order_seed_u32",
        "first_player_seat",
        "post_turn_order_observation_sha256",
        "identity_sha256",
    }
    if set(raw) != expected_keys:
        raise R289RunnerError("persisted pair seal fields drifted")
    try:
        seal = PairFirstPlayerSeal(
            evaluation_id=raw["evaluation_id"],
            pair_index=raw["pair_index"],
            pair_id=raw["pair_id"],
            pair_nonce_sha256=raw["pair_nonce_sha256"],
            engine_seed_u32=raw["engine_seed_u32"],
            deck_order_seed_u32=raw["deck_order_seed_u32"],
            first_player_seat=raw["first_player_seat"],
            post_turn_order_observation_sha256=raw["post_turn_order_observation_sha256"],
        )
        validate_pair_first_player_seal(pair, seal)
    except (SeededMirrorHarnessError, TypeError, ValueError) as exc:
        raise R289RunnerError("persisted pair seal no longer binds its schedule") from exc
    if raw != seal.as_payload():
        raise R289RunnerError("persisted pair seal identity digest drifted")
    return seal


def _capture_pair_seal(
    *,
    env: Any,
    pair: Sequence[SeededMirrorGameSpec],
    candidate_deck: Sequence[int],
    control_deck: Sequence[int],
    spec: SeededMirrorGameSpec,
) -> tuple[PairFirstPlayerSeal, list[list[int]]]:
    from poke_bot.engine_rebuild.interfaces import ResetSpec

    first, second = pair
    if spec not in pair:
        raise R289RunnerError("first-player seal spec is foreign to its pair")
    if first.engine_seed_u32 != second.engine_seed_u32:
        raise R289RunnerError("pair cannot share an engine seed")
    deck0 = candidate_deck if spec.experimental_seat == 0 else control_deck
    deck1 = candidate_deck if spec.experimental_seat == 1 else control_deck
    state = env.reset([ResetSpec(deck0=list(deck0), deck1=list(deck1), seed=int(first.engine_seed_u32))]).envs[0]
    setup_actions: list[list[int]] = []
    for _ in range(8):
        if state.done:
            raise R289RunnerError("engine ended before first-player sealing")
        observation = state.obs
        forced = _forced_turn_order_action(observation)
        if forced is not None:
            setup_actions.append(forced)
            state = env.step_batch([forced]).envs[0]
            continue
        first_player = _first_player_from_observation(observation)
        if first_player is None:
            raise R289RunnerError("seeded engine did not resolve an IsFirst prompt")
        seal = PairFirstPlayerSeal(
            evaluation_id=first.evaluation_id,
            pair_index=first.pair_index,
            pair_id=first.pair_id,
            pair_nonce_sha256=first.pair_nonce_sha256,
            engine_seed_u32=first.engine_seed_u32,
            deck_order_seed_u32=first.deck_order_seed_u32,
            first_player_seat=first_player,
            post_turn_order_observation_sha256=canonical_digest(observation),
        )
        try:
            validate_pair_first_player_seal(pair, seal)
        except SeededMirrorHarnessError as exc:
            raise R289RunnerError("captured first-player seal does not bind pair") from exc
        return seal, setup_actions
    raise R289RunnerError("seeded engine did not resolve first player within setup limit")


def _load_or_capture_pair_seal(
    *,
    path: Path,
    env: Any,
    pair: Sequence[SeededMirrorGameSpec],
    candidate_deck: Sequence[int],
    control_deck: Sequence[int],
    spec: SeededMirrorGameSpec,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
) -> PairFirstPlayerSeal:
    if path.exists():
        return _decode_pair_seal(
            _json_object(path, label="pair first-player seal"),
            pair=pair,
            run_identity_sha256=run_identity_sha256,
            runtime_preflight_sha256=runtime_preflight_sha256,
        )
    seal, setup_actions = _capture_pair_seal(
        env=env,
        pair=pair,
        candidate_deck=candidate_deck,
        control_deck=control_deck,
        spec=spec,
    )
    write_create_only_json(
        path,
        _seal_document(
            run_identity_sha256=run_identity_sha256,
            runtime_preflight_sha256=runtime_preflight_sha256,
            seal=seal,
            setup_actions=setup_actions,
        ),
        label="pair first-player seal",
    )
    return seal


def _trace_document(
    *,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
    spec: SeededMirrorGameSpec,
    stage_traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    traces = [dict(trace) for trace in stage_traces]
    digest = canonical_digest(traces)
    return {
        "schema": SCRIPT_SCHEMA,
        "kind": TRACE_KIND,
        "evaluation_id": spec.evaluation_id,
        "run_identity_sha256": run_identity_sha256,
        "runtime_preflight_sha256": runtime_preflight_sha256,
        "game_nonce_sha256": spec.game_nonce_sha256,
        "candidate_stage_traces": traces,
        "candidate_stage_trace_sha256": digest,
        "training_eligible": False,
        "promotion_authority": False,
        "production_authority": False,
    }


def _validate_trace_document(
    payload: Mapping[str, Any],
    *,
    spec: SeededMirrorGameSpec,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if (
        payload.get("schema") != SCRIPT_SCHEMA
        or payload.get("kind") != TRACE_KIND
        or payload.get("evaluation_id") != spec.evaluation_id
        or payload.get("run_identity_sha256") != run_identity_sha256
        or payload.get("runtime_preflight_sha256") != runtime_preflight_sha256
        or payload.get("game_nonce_sha256") != spec.game_nonce_sha256
        or payload.get("training_eligible") is not False
        or payload.get("promotion_authority") is not False
        or payload.get("production_authority") is not False
    ):
        raise R289RunnerError("candidate trace document has a foreign authority/binding")
    traces = payload.get("candidate_stage_traces")
    if not isinstance(traces, list):
        raise R289RunnerError("candidate trace document has no trace list")
    digest = canonical_digest(traces)
    if payload.get("candidate_stage_trace_sha256") != digest:
        raise R289RunnerError("candidate trace document digest mismatch")
    telemetry = empty_checklist_telemetry()
    for entry in traces:
        if not isinstance(entry, Mapping):
            raise R289RunnerError("candidate trace journal entry is not an object")
        selected_action = entry.get("selected_action")
        game_step = entry.get("game_step")
        if (
            not isinstance(selected_action, list)
            or not isinstance(game_step, int)
            or game_step < 0
        ):
            raise R289RunnerError("candidate trace journal entry lacks action/step binding")
        # The persisted journal intentionally omits the redundant top-level
        # enabled flag, so restore it only to exercise the same strict live
        # validator that produced the trace.  This prevents a changed trace
        # and receipt from becoming creditable merely because their digests
        # agree with one another.
        reconstructed = _record_candidate_trace(
            telemetry=telemetry,
            trace={"enabled": True, **dict(entry)},
            selected_action=selected_action,
            game_step=game_step,
        )
        if reconstructed != dict(entry):
            raise R289RunnerError("candidate trace journal semantic binding drifted")
    return digest, telemetry


def _play_one_game(
    *,
    env: Any,
    spec: SeededMirrorGameSpec,
    seal: PairFirstPlayerSeal,
    candidate_deck: Sequence[int],
    control_deck: Sequence[int],
    candidate: Any,
    control: Any,
    max_steps: int,
) -> tuple[int, int, int, dict[str, Any], list[dict[str, Any]]]:
    """Play one terminal game.  Errors leave it entirely outcome-ineligible."""

    from poke_bot.engine_rebuild.interfaces import ResetSpec

    candidate.reset_game()
    control.reset_game()
    candidate.rng = random.Random(int(spec.experimental_rng_seed_u32))
    control.rng = random.Random(int(spec.control_rng_seed_u32))
    deck0 = candidate_deck if spec.experimental_seat == 0 else control_deck
    deck1 = candidate_deck if spec.experimental_seat == 1 else control_deck
    state = env.reset([ResetSpec(deck0=list(deck0), deck1=list(deck1), seed=int(spec.engine_seed_u32))]).envs[0]
    steps = 0
    first_player_verified = False
    telemetry = empty_checklist_telemetry()
    stage_traces: list[dict[str, Any]] = []
    while not state.done and steps < max_steps:
        observation = state.obs
        forced = _forced_turn_order_action(observation)
        if forced is not None:
            state = env.step_batch([forced]).envs[0]
            steps += 1
            continue
        # The pair seal is a start-state fact, not an invariant of every
        # later board state.  Check it exactly once at the first actionable
        # observation; subsequent atomic actions necessarily change the
        # observation digest.
        if not first_player_verified:
            first_player = _first_player_from_observation(observation)
            if first_player is None:
                raise R289RunnerError(
                    "game left the IsFirst prompt without a first-player fact"
                )
            if first_player != seal.first_player_seat:
                raise R289RunnerError(
                    "game first player disagrees with its pre-game pair seal"
                )
            if canonical_digest(observation) != seal.post_turn_order_observation_sha256:
                raise R289RunnerError(
                    "game post-turn-order public state disagrees with its pair seal"
                )
            first_player_verified = True
        current = observation.get("current")
        if not isinstance(current, Mapping):
            raise R289RunnerError("game has no current-seat payload")
        seat = _require_int(current.get("yourIndex"), label="engine acting seat", allowed={0, 1})
        if seat == spec.experimental_seat:
            action = _validate_legal_action(observation, candidate(dict(observation)))
            stage_traces.append(
                _record_candidate_trace(
                    telemetry=telemetry,
                    trace=getattr(candidate, "last_turn_checklist_logit_trace", None),
                    selected_action=action,
                    game_step=steps,
                )
            )
        elif seat == spec.control_seat:
            action = _validate_legal_action(observation, control(dict(observation)))
            if getattr(control, "last_turn_checklist_logit_trace", None) is not None:
                raise R289RunnerError("r195 control emitted checklist telemetry while disabled")
        else:  # pragma: no cover - both legal seats are checked above.
            raise R289RunnerError("engine emitted a seat outside the pair")
        state = env.step_batch([action]).envs[0]
        steps += 1
    if not first_player_verified:
        raise R289RunnerError("terminal game never verified its first-player pair seal")
    if not state.done:
        raise R289RunnerError("game reached max atomic actions without a terminal result")
    winner = _require_int(state.winner, label="terminal winner", allowed={0, 1, 2})
    if steps < 1:
        raise R289RunnerError("terminal game has no atomic actions")
    return seal.first_player_seat, winner, steps, telemetry, stage_traces


def _game_paths(output_dir: Path, spec: SeededMirrorGameSpec) -> tuple[Path, Path]:
    return (
        output_dir / "games" / f"{spec.game_nonce_sha256[7:]}.json",
        output_dir / "traces" / f"{spec.game_nonce_sha256[7:]}.json",
    )


def _read_existing_game_receipt(
    *,
    output_dir: Path,
    spec: SeededMirrorGameSpec,
    run_identity_sha256: str,
    runtime_preflight_sha256: str,
) -> dict[str, Any] | None:
    receipt_path, trace_path = _game_paths(output_dir, spec)
    if not receipt_path.exists():
        if trace_path.exists():
            # A process may be interrupted after the immutable trace journal is
            # fsynced but before an outcome receipt exists.  It has no credit;
            # validate its binding, then permit a deterministic replay.  The
            # create-only trace writer will reject any non-identical replay.
            _validate_trace_document(
                _json_object(trace_path, label="orphan candidate checklist trace"),
                spec=spec,
                run_identity_sha256=run_identity_sha256,
                runtime_preflight_sha256=runtime_preflight_sha256,
            )
        return None
    receipt = _json_object(receipt_path, label="persisted r289 game receipt")
    try:
        validated = validate_game_receipt(
            receipt,
            spec=spec,
            run_identity_sha256=run_identity_sha256,
            runtime_preflight_sha256=runtime_preflight_sha256,
        )
    except (R289BO250Error, TypeError, ValueError) as exc:
        raise R289RunnerError("persisted r289 game receipt is invalid") from exc
    if not trace_path.exists():
        raise R289RunnerError("outcome receipt lacks its required candidate trace")
    trace_digest, trace_telemetry = _validate_trace_document(
        _json_object(trace_path, label="persisted candidate checklist trace"),
        spec=spec,
        run_identity_sha256=run_identity_sha256,
        runtime_preflight_sha256=runtime_preflight_sha256,
    )
    if trace_digest != validated["candidate_stage_trace_sha256"]:
        raise R289RunnerError("game receipt trace digest differs from its staged trace")
    if trace_telemetry != validated["candidate_checklist_telemetry"]:
        raise R289RunnerError("game receipt checklist telemetry differs from trace journal")
    return validated


def _validate_completed_game_binding(
    *,
    spec: SeededMirrorGameSpec,
    seal: PairFirstPlayerSeal,
    receipt: Mapping[str, Any] | None,
) -> None:
    """Require one credited game to bind its deck-specific pre-game seal.

    A/C happen to use one deck on both seats, but B deliberately puts the
    r195 native deck opposite the derivative's new list.  Pairing still shares
    RNG and seats, while each game's public post-setup observation needs a
    separate seal.  Treating it as a single shared observation would mask that
    deck distinction.
    """

    if receipt is None:
        return
    if receipt.get("pair_first_player_seal_sha256") != seal.identity_sha256:
        raise R289RunnerError("completed game receipt is bound to a different game seal")
    if receipt.get("first_player_seat") != seal.first_player_seat:
        raise R289RunnerError("completed game receipt first player differs from game seal")
    if receipt.get("game_nonce_sha256") != spec.game_nonce_sha256:
        raise R289RunnerError("completed game receipt does not match its sealed game")


def _preflight_body(
    *,
    run_identity_sha256: str,
    input_identities: Mapping[str, Any],
    staged_identities: Mapping[str, Any],
    sanitized_environment: Mapping[str, Any],
    runtime: Mapping[str, Any],
    engine_runtime: Mapping[str, Any],
    host: str,
    comparison_id: str,
    candidate_deck_identity: Mapping[str, Any],
    control_deck_identity: Mapping[str, Any],
    control_arm: str,
) -> dict[str, Any]:
    return {
        "schema": SCRIPT_SCHEMA,
        "kind": PREFLIGHT_KIND,
        "evaluation_id": comparison_evaluation_id(comparison_id),
        "comparison_id": comparison_id,
        "run_identity_sha256": run_identity_sha256,
        "input_identities": dict(input_identities),
        "staged_identities": dict(staged_identities),
        "sanitized_environment": dict(sanitized_environment),
        "runtime": dict(runtime),
        "engine_runtime": dict(engine_runtime),
        "host_scope": {"required": "elmo_only", "observed_hostname": host},
        "candidate_arm": CANDIDATE_ARM,
        "control_arm": control_arm,
        "candidate_deck": dict(candidate_deck_identity),
        "control_deck": dict(control_deck_identity),
        "deck_shift_confound": comparison_id in {"B", "C"},
        "direct_policy_boundary": dict(DIRECT_POLICY_FLAGS),
        "training_eligible": False,
        "promotion_authority": False,
        "kaggle_authority": False,
        "production_authority": False,
        "production_selector_authority": False,
        "elmo_production_readmission_authority": False,
    }


def _runtime_preflight_payload(body: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    digest = canonical_digest(body)
    return {**dict(body), "runtime_preflight_sha256": digest}, digest


def _validate_runtime_preflight(
    payload: Mapping[str, Any], *, run_identity_sha256: str, comparison_id: str
) -> str:
    raw = dict(payload)
    digest = raw.pop("runtime_preflight_sha256", None)
    if not isinstance(digest, str) or digest != canonical_digest(raw):
        raise R289RunnerError("runtime preflight digest mismatch")
    if (
        raw.get("schema") != SCRIPT_SCHEMA
        or raw.get("kind") != PREFLIGHT_KIND
        or raw.get("evaluation_id") != comparison_evaluation_id(comparison_id)
        or raw.get("comparison_id") != comparison_id
        or raw.get("run_identity_sha256") != run_identity_sha256
        or raw.get("direct_policy_boundary") != DIRECT_POLICY_FLAGS
        or raw.get("training_eligible") is not False
        or raw.get("promotion_authority") is not False
        or raw.get("production_authority") is not False
        or raw.get("control_arm") != CONTROL_ARMS[comparison_id]
    ):
        raise R289RunnerError("runtime preflight is not an r289 direct-only diagnostic")
    return digest


def _stage_inputs(
    *,
    output_dir: Path,
    sources: Mapping[str, Path],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    staged: dict[str, dict[str, Any]] = {}
    for name in sorted(sources):
        source = sources[name]
        relative = {
            "candidate_checkpoint": "artifacts/candidate-r298-derivative.pt",
            "r298_validation_receipt": "artifacts/r298-validation-receipt.json",
            "raw_corpus_receipt": "artifacts/r298-raw-corpus-receipt.json",
            "collision_census_receipt": "artifacts/r298-collision-census-receipt.json",
            "goal_contract": "artifacts/r298-goal-contract.json",
            "baseline_r274_checkpoint": "artifacts/baseline-r274.pt",
            "checklist_base_config": "artifacts/checklist-r288-base.json",
            "r293_overlap_audit_receipt": "artifacts/r293-r295-overlap-audit.json",
            "calibration_artifact": "artifacts/calibration-r288-gates.json",
            "r195_checkpoint": "artifacts/control-r195-no-rtp.pt",
            "r195_contract": "artifacts/control-r195-contract.json",
            "r195_no_rtp_bundle": "artifacts/control-r195-no-rtp-bundle.bin",
            "exact_new_list_deck": "artifacts/exact-new-list.csv",
            "r195_native_deck": "artifacts/r195-native-operational.csv",
            "candidate_matchup_tree": "artifacts/candidate-matchup-tree.json",
            "baseline_r274_matchup_tree": "artifacts/baseline-r274-matchup-tree.json",
            "r195_matchup_tree": "artifacts/r195-matchup-tree.json",
            "seeded_engine": "artifacts/libcg-seeded.so",
            "seeded_engine_receipt": "artifacts/libcg-seeded-receipt.json",
            "r289_config": "artifacts/r289-config.json",
            "calibration_receipt": "artifacts/calibration-r288.json",
        }.get(name)
        if relative is None:
            if not name.startswith("runtime_source:"):
                raise R289RunnerError(f"unknown r289 input for staging: {name}")
            relative = "artifacts/runtime-source/" + name.split(":", 1)[1]
        staged[name] = stage_verified_copy(
            source,
            output_dir / relative,
            expected_identity=identities[name],
            label=f"r289 {name}",
        )
    return staged


def _assert_legacy_checklist_trace_only(path: Path) -> None:
    """Ensure the historic r288 guide cannot influence an r298 benchmark.

    Until a separately typed r298 runtime-provenance artifact binds a narrow
    public-rule implementation for Q1/Q2/Q8, the old guide/CSV-derived r288
    layer may supply trace shape only.  This intentionally blocks the supplied
    nonzero r288 defaults rather than treating their historical calibration as
    authority for the derivative.
    """

    payload = _json_object(path, label="legacy checklist trace-only config")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("scalar_gates"), Mapping):
        raise R289RunnerError("legacy checklist runtime has no scalar-gate mapping")
    gates = runtime["scalar_gates"]
    if set(gates) != set(CHECKLIST_GATE_NAMES):
        raise R289RunnerError("legacy checklist gate inventory drifted")
    for name in CHECKLIST_GATE_NAMES:
        value = gates.get(name)
        if type(value) is bool or not isinstance(value, (int, float)) or float(value) != 0.0:
            raise R289RunnerError(
                "legacy r288 checklist/guide cannot arm any r298 derivative residual; "
                "supply the future strict r298 runtime-provenance integration instead"
            )


def _run_comparison(
    *,
    comparison_id: str,
    output_dir: Path,
    run_identity_sha256: str,
    comparison_seed_identity_sha256: str,
    schedule: Sequence[SeededMirrorGameSpec],
    schedule_sha256: str,
    input_identities: Mapping[str, Any],
    staged: Mapping[str, Mapping[str, Any]],
    sanitized_environment: Mapping[str, Any],
    lib: Any,
    engine_runtime: Mapping[str, Any],
    candidate_deck: Sequence[int],
    candidate_deck_identity: Mapping[str, Any],
    control_deck: Sequence[int],
    control_deck_identity: Mapping[str, Any],
    control_checkpoint_key: str,
    control_tree_key: str,
    runtime_sources: Mapping[str, Mapping[str, Any]],
    host: str,
    device_name: str,
    max_steps: int,
    remaining_new_games: int | None,
    preflight_only: bool,
) -> tuple[dict[str, Any] | None, int | None, int, int]:
    """Run or resume one isolated cohort without cross-crediting another."""

    evaluation_id = comparison_evaluation_id(comparison_id)
    validate_schedule(
        schedule,
        seed_identity_sha256=comparison_seed_identity_sha256,
        comparison_id=comparison_id,
    )
    if output_dir.is_symlink():
        raise R289RunnerError("comparison output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_create_only_json(
        output_dir / "output-index.json",
        output_index_payload(
            run_identity_sha256=run_identity_sha256,
            output_dir=output_dir,
            schedule_sha256=schedule_sha256,
            input_identities=input_identities,
            evaluation_id=evaluation_id,
            comparison_id=comparison_id,
        ),
        label=f"r289 comparison {comparison_id} output index",
    )
    write_create_only_json(
        output_dir / "schedule.json",
        {
            "schema": SCRIPT_SCHEMA,
            "kind": SCHEDULE_KIND,
            "evaluation_id": evaluation_id,
            "comparison_id": comparison_id,
            "run_identity_sha256": run_identity_sha256,
            "comparison_seed_identity_sha256": comparison_seed_identity_sha256,
            "schedule_sha256": schedule_sha256,
            "games": [spec.as_payload() for spec in schedule],
            "pair_count": 125,
            "game_count": 250,
            "training_eligible": False,
            "promotion_authority": False,
            "production_authority": False,
        },
        label=f"r289 comparison {comparison_id} seed schedule",
    )
    candidate, control, policy_runtime = _load_direct_policies(
        candidate_checkpoint=Path(staged["candidate_checkpoint"]["path"]),
        control_checkpoint=Path(staged[control_checkpoint_key]["path"]),
        candidate_deck=candidate_deck,
        control_deck=control_deck,
        candidate_tree=Path(staged["candidate_matchup_tree"]["path"]),
        control_tree=Path(staged[control_tree_key]["path"]),
        checklist_config=Path(staged["checklist_base_config"]["path"]),
        device_name=device_name,
        control_arm=CONTROL_ARMS[comparison_id],
    )
    preflight_body = _preflight_body(
        run_identity_sha256=run_identity_sha256,
        input_identities=input_identities,
        staged_identities=staged,
        sanitized_environment=sanitized_environment,
        runtime=policy_runtime,
        engine_runtime=engine_runtime,
        host=host,
        comparison_id=comparison_id,
        candidate_deck_identity=candidate_deck_identity,
        control_deck_identity=control_deck_identity,
        control_arm=CONTROL_ARMS[comparison_id],
    )
    preflight_payload, runtime_preflight_sha256 = _runtime_preflight_payload(preflight_body)
    preflight_path = output_dir / "runtime-preflight.json"
    write_create_only_json(
        preflight_path, preflight_payload, label=f"r289 comparison {comparison_id} preflight"
    )
    persisted_preflight = _json_object(preflight_path, label="r289 runtime preflight")
    if _validate_runtime_preflight(
        persisted_preflight,
        run_identity_sha256=run_identity_sha256,
        comparison_id=comparison_id,
    ) != runtime_preflight_sha256:
        raise R289RunnerError("persisted comparison runtime preflight address drifted")
    if preflight_only:
        return None, remaining_new_games, 0, 0

    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(1, lib=lib)
    try:
        existing: dict[str, dict[str, Any]] = {}
        for spec in schedule:
            row = _read_existing_game_receipt(
                output_dir=output_dir,
                spec=spec,
                run_identity_sha256=run_identity_sha256,
                runtime_preflight_sha256=runtime_preflight_sha256,
            )
            if row is not None:
                existing[spec.game_nonce_sha256] = row
        new_games = 0
        stopped = False
        for pair_index in range(125):
            pair = _pair_specs(schedule, pair_index)
            for spec in pair:
                if spec.game_nonce_sha256 in existing:
                    seal_path = output_dir / "game-seals" / f"{spec.game_nonce_sha256[7:]}.json"
                    if not seal_path.exists():
                        raise R289RunnerError("completed game lacks its deck-specific pre-game seal")
                    seal = _load_or_capture_pair_seal(
                        path=seal_path,
                        env=env,
                        pair=pair,
                        candidate_deck=candidate_deck,
                        control_deck=control_deck,
                        spec=spec,
                        run_identity_sha256=run_identity_sha256,
                        runtime_preflight_sha256=runtime_preflight_sha256,
                    )
                    _validate_completed_game_binding(
                        spec=spec, seal=seal, receipt=existing[spec.game_nonce_sha256]
                    )
                    continue
                if remaining_new_games is not None and remaining_new_games <= 0:
                    stopped = True
                    break
                seal_path = output_dir / "game-seals" / f"{spec.game_nonce_sha256[7:]}.json"
                seal = _load_or_capture_pair_seal(
                    path=seal_path,
                    env=env,
                    pair=pair,
                    candidate_deck=candidate_deck,
                    control_deck=control_deck,
                    spec=spec,
                    run_identity_sha256=run_identity_sha256,
                    runtime_preflight_sha256=runtime_preflight_sha256,
                )
                first_player, winner, steps, telemetry, stage_traces = _play_one_game(
                    env=env,
                    spec=spec,
                    seal=seal,
                    candidate_deck=candidate_deck,
                    control_deck=control_deck,
                    candidate=candidate,
                    control=control,
                    max_steps=max_steps,
                )
                trace_payload = _trace_document(
                    run_identity_sha256=run_identity_sha256,
                    runtime_preflight_sha256=runtime_preflight_sha256,
                    spec=spec,
                    stage_traces=stage_traces,
                )
                receipt_path, trace_path = _game_paths(output_dir, spec)
                write_create_only_json(
                    trace_path,
                    trace_payload,
                    label=f"r289 comparison {comparison_id} candidate trace",
                )
                receipt = make_game_receipt(
                    run_identity_sha256=run_identity_sha256,
                    spec=spec,
                    first_player_seat=first_player,
                    winner_seat=winner,
                    steps=steps,
                    checklist_telemetry=telemetry,
                    runtime_preflight_sha256=runtime_preflight_sha256,
                    direct_policy_flags=DIRECT_POLICY_FLAGS,
                    pair_first_player_seal_sha256=seal.identity_sha256,
                    stage_trace_digest=trace_payload["candidate_stage_trace_sha256"],
                )
                write_create_only_json(
                    receipt_path,
                    receipt,
                    label=f"r289 comparison {comparison_id} completed game receipt",
                )
                persisted = _read_existing_game_receipt(
                    output_dir=output_dir,
                    spec=spec,
                    run_identity_sha256=run_identity_sha256,
                    runtime_preflight_sha256=runtime_preflight_sha256,
                )
                if persisted is None:
                    raise R289RunnerError("newly written game receipt was not readable")
                _validate_completed_game_binding(spec=spec, seal=seal, receipt=persisted)
                existing[spec.game_nonce_sha256] = persisted
                new_games += 1
                if remaining_new_games is not None:
                    remaining_new_games -= 1
                print(
                    json.dumps(
                        {
                            "comparison_id": comparison_id,
                            "completed_game_nonce_sha256": spec.game_nonce_sha256,
                            "new_games_this_comparison": new_games,
                            "output_dir": str(output_dir),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if stopped:
                break
        _assert_runtime_sources_unchanged(runtime_sources)
        completed = [
            existing[spec.game_nonce_sha256]
            for spec in schedule
            if spec.game_nonce_sha256 in existing
        ]
        if len(completed) != 250:
            return None, remaining_new_games, new_games, len(completed)
        report = compile_report(
            run_identity_sha256=run_identity_sha256,
            runtime_preflight_sha256=runtime_preflight_sha256,
            seed_identity_sha256=comparison_seed_identity_sha256,
            schedule=schedule,
            game_receipts=completed,
            input_identities=input_identities,
            comparison_id=comparison_id,
        )
        write_create_only_json(
            output_dir / "report.json", report, label=f"r289 comparison {comparison_id} report"
        )
        return report, remaining_new_games, new_games, len(completed)
    finally:
        env.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Retired single-cohort parser; no legacy r288 route may be invoked."""

    raise R289RunnerError(
        "retired legacy r288 single-cohort invocation is prohibited; use the "
        "revision-4 r298 receipt-bound entrypoint"
    )

    # Historical parser intentionally unreachable.  It remains only until the
    # surrounding source is mechanically pruned, never as an alternate path.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/evaluations/alakazam-turn-checklist-bo250-r289.json",
    )
    parser.add_argument(
        "--owner-contract",
        type=Path,
        default=ROOT / "state/alakazam-new-list-direct-policy-r241.json",
    )
    parser.add_argument(
        "--candidate-checkpoint", type=Path, required=True,
        help="Elmo-readable source for exact r274 iter_00000; copied create-only after validation.",
    )
    parser.add_argument(
        "--candidate-receipt",
        type=Path,
        default=ROOT / "state/alakazam-r274-iteration1-boundary-r284.json",
    )
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--r195-contract",
        type=Path,
        default=ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
    )
    parser.add_argument("--r195-no-rtp-bundle", type=Path, required=True)
    parser.add_argument(
        "--deck",
        type=Path,
        default=ROOT / "decks/archetype-samples/alakazam-new-list-direct-r241.csv",
    )
    parser.add_argument(
        "--checklist-config",
        type=Path,
        default=ROOT / "config/policy_layers/alakazam-turn-checklist-r288.json",
    )
    parser.add_argument(
        "--r293-overlap-audit-receipt",
        type=Path,
        required=True,
        help=(
            "Completed Elmo-only r292/r293/r295 audit receipt. It must bind the "
            "base checklist config, corrected attachment, per-channel dedup trace "
            "contract, and trace-only broad-guide gate."
        ),
    )
    parser.add_argument("--candidate-matchup-tree", type=Path, required=True)
    parser.add_argument("--control-matchup-tree", type=Path, required=True)
    parser.add_argument("--seeded-engine", type=Path, required=True)
    parser.add_argument("--seeded-engine-receipt", type=Path, required=True)
    parser.add_argument(
        "--seed-identity-sha256",
        required=True,
        help="Fresh immutable identity for the 125-pair r289 seed schedule.",
    )
    parser.add_argument(
        "--calibration-receipt",
        type=Path,
        default=None,
        help="Optional r288 scalar-gate receipt; must explicitly exclude this seed identity.",
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=None,
        help=(
            "Optional inert r288 gate artifact paired with --calibration-receipt; "
            "the runner derives an Elmo-only immutable runtime config from it."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Elmo-local diagnostic root; runner creates one content-addressed child only.",
    )
    parser.add_argument("--device", default="cpu", help="Local Torch device; never a remote backend.")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--max-new-games",
        type=int,
        default=0,
        help="Bound one resumable invocation; 0 means play all pending games.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Stage and construct the exact isolated runtime without starting a game.",
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.max_new_games < 0:
        parser.error("--max-new-games cannot be negative")
    if (args.calibration_receipt is None) != (args.calibration_artifact is None):
        parser.error("--calibration-receipt and --calibration-artifact must be supplied together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    # Kept only as a compatibility entrypoint for callers that imported the
    # original r289 script.  The previous r274-vs-r195 body remains below as
    # unreachable historical context while the executable path is revision-4
    # contract-bound only.
    return main_r298(argv)

    # pragma: no cover - legacy body intentionally unreachable.
    args = _parse_args(argv)
    try:
        elmo_hostname = _require_elmo_host()
        config_path = _path(args.config)
        owner_contract_path = _path(args.owner_contract)
        candidate_checkpoint_path = _path(args.candidate_checkpoint)
        candidate_receipt_path = _path(args.candidate_receipt)
        control_checkpoint_path = _path(args.control_checkpoint)
        r195_contract_path = _path(args.r195_contract)
        r195_bundle_path = _path(args.r195_no_rtp_bundle)
        deck_path = _path(args.deck)
        checklist_config_path = _path(args.checklist_config)
        r293_overlap_audit_receipt_path = _path(args.r293_overlap_audit_receipt)
        candidate_tree_path = _path(args.candidate_matchup_tree)
        control_tree_path = _path(args.control_matchup_tree)
        seeded_engine_path = _path(args.seeded_engine)
        seeded_engine_receipt_path = _path(args.seeded_engine_receipt)
        calibration_path = (
            None if args.calibration_receipt is None else _path(args.calibration_receipt)
        )
        calibration_artifact_path = (
            None if args.calibration_artifact is None else _path(args.calibration_artifact)
        )

        # Everything that could grant authority is checked before output input
        # staging or a native reset.  We pass the exact same new list into both
        # agent instances and both engine seats below.
        _config, config_identity = load_r289_config(config_path)
        owner_identity = validate_owner_contract(owner_contract_path)
        deck, deck_identity = read_exact_new_list_deck(deck_path)
        base_checklist_identity = validate_checklist_config(checklist_config_path)
        r293_overlap_audit_identity = validate_r293_overlap_audit_receipt(
            r293_overlap_audit_receipt_path,
            checklist_config_identity=base_checklist_identity,
        )
        candidate_identity = file_identity(candidate_checkpoint_path, label="r274 candidate checkpoint")
        candidate_receipt_identity = validate_candidate_receipt(
            candidate_receipt_path, candidate_identity
        )
        control_identity = file_identity(control_checkpoint_path, label="r195 control checkpoint")
        r195_contract_identity = validate_r195_contract(r195_contract_path, control_identity)
        r195_bundle_identity = file_identity(r195_bundle_path, label="r195 NO-RTP bundle")
        if r195_bundle_identity.get("sha256") != R195_NO_RTP_BUNDLE_SHA256:
            raise R289RunnerError("control bundle is not immutable r195 NO-RTP")
        # The control's public tree is part of its immutable r195 submission
        # contract, not a user-tunable comparison knob.
        r195_contract_payload = _json_object(r195_contract_path, label="r195 contract")
        expected_r195_tree = (
            r195_contract_payload.get("completion", {}).get("matchup_tree_sha256")
            if isinstance(r195_contract_payload.get("completion"), Mapping)
            else None
        )
        control_tree_identity = file_identity(control_tree_path, label="r195 matchup tree")
        if control_tree_identity.get("sha256") != expected_r195_tree:
            raise R289RunnerError("control matchup tree is not the immutable r195 tree")
        candidate_tree_identity = file_identity(candidate_tree_path, label="r274 matchup tree")
        if (
            candidate_tree_identity.get("sha256")
            != R274_DIRECT_POLICY_MATCHUP_TREE_SHA256
        ):
            raise R289RunnerError(
                "candidate matchup tree is not the receipt-bound r274 direct-policy tree"
            )
        seeded_engine_identity = file_identity(seeded_engine_path, label="seeded engine")
        seeded_engine_receipt_identity = validate_seeded_engine_receipt(
            seeded_engine_receipt_path, engine_identity=seeded_engine_identity
        )
        calibration_identity = validate_optional_calibration_receipt(
            calibration_path,
            evaluation_seed_identity_sha256=args.seed_identity_sha256,
        )
        calibrated_runtime_payload: dict[str, Any] | None = None
        calibration_artifact_identity: dict[str, Any] | None = None
        runtime_checklist_identity: dict[str, Any] = dict(base_checklist_identity)
        if calibration_path is not None and calibration_identity is not None:
            if calibration_artifact_path is None:
                raise R289RunnerError("calibration artifact was not supplied")
            (
                calibrated_runtime_payload,
                runtime_checklist_identity,
                calibration_artifact_identity,
            ) = _prepare_calibrated_runtime_config(
                base_config_path=checklist_config_path,
                base_config_identity=base_checklist_identity,
                calibration_receipt_path=calibration_path,
                calibration_receipt_identity=calibration_identity,
                calibration_artifact_path=calibration_artifact_path,
                candidate_checkpoint_sha256=str(candidate_identity["sha256"]),
                evaluation_seed_identity_sha256=args.seed_identity_sha256,
            )
        else:
            _assert_default_uncalibrated_gates(checklist_config_path)
        runtime_sources = _runtime_source_identities()
        schedule = build_schedule(args.seed_identity_sha256)
        validate_schedule(schedule, seed_identity_sha256=args.seed_identity_sha256)
        schedule_sha256 = schedule_identity(
            schedule, seed_identity_sha256=args.seed_identity_sha256
        )
        run_identity = build_run_identity(
            config_identity=config_identity,
            owner_contract_identity=owner_identity,
            candidate_checkpoint=candidate_identity,
            candidate_receipt=candidate_receipt_identity,
            control_checkpoint=control_identity,
            r195_contract=r195_contract_identity,
            r195_bundle=r195_bundle_identity,
            exact_deck=deck_identity,
            checklist_config=runtime_checklist_identity,
            candidate_matchup_tree=candidate_tree_identity,
            control_matchup_tree=control_tree_identity,
            seeded_engine=seeded_engine_identity,
            seeded_engine_receipt=seeded_engine_receipt_identity,
            r293_overlap_audit_receipt=r293_overlap_audit_identity,
            calibration_receipt=calibration_identity,
            runtime_sources=runtime_sources,
            seed_identity_sha256=args.seed_identity_sha256,
            schedule_sha256=schedule_sha256,
        )

        output_root = _path(args.output_root)
        _assert_isolated_output_root(output_root)
        output_dir = output_root / f"{EVALUATION_ID}-{run_identity[7:]}"
        if output_dir.is_symlink():
            raise R289RunnerError("content-addressed r289 output may not be a symlink")
        input_identities: dict[str, Any] = {
            "r289_config": config_identity,
            "owner_contract": owner_identity,
            "candidate_checkpoint": candidate_identity,
            "candidate_receipt": candidate_receipt_identity,
            "control_checkpoint": control_identity,
            "r195_contract": r195_contract_identity,
            "r195_no_rtp_bundle": r195_bundle_identity,
            "exact_new_list_deck": deck_identity,
            "checklist_base_config": base_checklist_identity,
            "r293_overlap_audit_receipt": r293_overlap_audit_identity,
            "checklist_runtime_config": runtime_checklist_identity,
            "candidate_matchup_tree": candidate_tree_identity,
            "control_matchup_tree": control_tree_identity,
            "seeded_engine": seeded_engine_identity,
            "seeded_engine_receipt": seeded_engine_receipt_identity,
            "calibration_receipt": calibration_identity,
            "calibration_artifact": calibration_artifact_identity,
            "runtime_sources": runtime_sources,
            "seed_identity_sha256": args.seed_identity_sha256,
            "schedule_sha256": schedule_sha256,
        }
        content_input_identities = _content_only(input_identities)
        sources: dict[str, Path] = {
            "r289_config": config_path,
            "owner_contract": owner_contract_path,
            "candidate_checkpoint": candidate_checkpoint_path,
            "candidate_receipt": candidate_receipt_path,
            "control_checkpoint": control_checkpoint_path,
            "r195_contract": r195_contract_path,
            "r195_no_rtp_bundle": r195_bundle_path,
            "exact_new_list_deck": deck_path,
            "checklist_base_config": checklist_config_path,
            "r293_overlap_audit_receipt": r293_overlap_audit_receipt_path,
            "candidate_matchup_tree": candidate_tree_path,
            "control_matchup_tree": control_tree_path,
            "seeded_engine": seeded_engine_path,
            "seeded_engine_receipt": seeded_engine_receipt_path,
        }
        staging_identities: dict[str, Mapping[str, Any]] = {
            key: input_identities[key]
            for key in sources
            if isinstance(input_identities.get(key), Mapping)
        }
        if calibration_path is not None and calibration_identity is not None:
            sources["calibration_receipt"] = calibration_path
            staging_identities["calibration_receipt"] = calibration_identity
        if calibration_artifact_path is not None and calibration_artifact_identity is not None:
            sources["calibration_artifact"] = calibration_artifact_path
            staging_identities["calibration_artifact"] = calibration_artifact_identity
        for relative, identity in runtime_sources.items():
            key = f"runtime_source:{relative}"
            sources[key] = ROOT / relative
            staging_identities[key] = identity

        output_dir.mkdir(parents=True, exist_ok=True)
        write_create_only_json(
            output_dir / "output-index.json",
            output_index_payload(
                run_identity_sha256=run_identity,
                output_dir=output_dir,
                schedule_sha256=schedule_sha256,
                input_identities=content_input_identities,
            ),
            label="r289 output index",
        )
        write_create_only_json(
            output_dir / "schedule.json",
            {
                "schema": SCRIPT_SCHEMA,
                "kind": SCHEDULE_KIND,
                "evaluation_id": EVALUATION_ID,
                "run_identity_sha256": run_identity,
                "seed_identity_sha256": args.seed_identity_sha256,
                "schedule_sha256": schedule_sha256,
                "games": [spec.as_payload() for spec in schedule],
                "pair_count": 125,
                "game_count": 250,
                "training_eligible": False,
                "promotion_authority": False,
                "production_authority": False,
            },
            label="r289 seed schedule",
        )
        staged = _stage_inputs(
            output_dir=output_dir, sources=sources, identities=staging_identities
        )
        if calibrated_runtime_payload is not None:
            runtime_config_path = (
                output_dir / "artifacts" / "checklist-r288-calibrated-runtime.json"
            )
            staged_runtime_config = write_create_only_json(
                runtime_config_path,
                calibrated_runtime_payload,
                label="r289 Elmo-only calibrated checklist config",
            )
            if (
                staged_runtime_config.get("sha256")
                != runtime_checklist_identity.get("sha256")
                or staged_runtime_config.get("size_bytes")
                != runtime_checklist_identity.get("size_bytes")
            ):
                raise R289RunnerError("derived calibrated checklist config identity drifted")
            validate_checklist_config(runtime_config_path)
            staged["checklist_runtime_config"] = staged_runtime_config
        else:
            staged["checklist_runtime_config"] = staged["checklist_base_config"]
        runtime_deck, runtime_deck_identity = read_exact_new_list_deck(
            Path(staged["exact_new_list_deck"]["path"])
        )
        if (
            runtime_deck_identity.get("sha256") != deck_identity.get("sha256")
            or runtime_deck_identity.get("canonical_multiset_sha256")
            != deck_identity.get("canonical_multiset_sha256")
            or runtime_deck != deck
        ):
            raise R289RunnerError("staged exact new-list deck no longer matches its source")

        with _direct_policy_environment() as sanitized_environment:
            staged_candidate = Path(staged["candidate_checkpoint"]["path"])
            staged_control = Path(staged["control_checkpoint"]["path"])
            staged_candidate_tree = Path(staged["candidate_matchup_tree"]["path"])
            staged_control_tree = Path(staged["control_matchup_tree"]["path"])
            staged_checklist = Path(staged["checklist_runtime_config"]["path"])
            staged_engine = Path(staged["seeded_engine"]["path"])
            lib, engine_runtime = _load_seeded_engine(staged_engine)
            candidate, control, policy_runtime = _load_direct_policies(
                candidate_checkpoint=staged_candidate,
                control_checkpoint=staged_control,
                deck=runtime_deck,
                candidate_tree=staged_candidate_tree,
                control_tree=staged_control_tree,
                checklist_config=staged_checklist,
                device_name=args.device,
            )
            preflight_body = _preflight_body(
                run_identity_sha256=run_identity,
                input_identities=content_input_identities,
                staged_identities=staged,
                sanitized_environment=sanitized_environment,
                runtime=policy_runtime,
                engine_runtime=engine_runtime,
                host=elmo_hostname,
            )
            preflight_payload, runtime_preflight_sha256 = _runtime_preflight_payload(
                preflight_body
            )
            preflight_path = output_dir / "runtime-preflight.json"
            write_create_only_json(
                preflight_path, preflight_payload, label="r289 runtime preflight"
            )
            persisted_preflight = _json_object(preflight_path, label="r289 runtime preflight")
            if _validate_runtime_preflight(
                persisted_preflight, run_identity_sha256=run_identity
            ) != runtime_preflight_sha256:
                raise R289RunnerError("persisted runtime preflight address drifted")

            if args.preflight_only:
                print(
                    json.dumps(
                        {
                            "status": "preflight_passed_no_games_started",
                            "evaluation_id": EVALUATION_ID,
                            "output_dir": str(output_dir),
                            "run_identity_sha256": run_identity,
                        },
                        sort_keys=True,
                    )
                )
                return 0

            from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

            env = LibcgMultiEnv(1, lib=lib)
            try:
                existing: dict[str, dict[str, Any]] = {}
                for spec in schedule:
                    row = _read_existing_game_receipt(
                        output_dir=output_dir,
                        spec=spec,
                        run_identity_sha256=run_identity,
                        runtime_preflight_sha256=runtime_preflight_sha256,
                    )
                    if row is not None:
                        existing[spec.game_nonce_sha256] = row
                new_games = 0
                stop = False
                for pair_index in range(125):
                    pair = _pair_specs(schedule, pair_index)
                    pending = [
                        spec
                        for spec in pair
                        if spec.game_nonce_sha256 not in existing
                    ]
                    if not pending:
                        seal_path = output_dir / "pair-seals" / f"{pair[0].pair_id}.json"
                        if not seal_path.exists():
                            raise R289RunnerError(
                                "completed game receipts lack their pre-game pair seal"
                            )
                        completed_seal = _load_or_capture_pair_seal(
                            path=seal_path,
                            env=env,
                            pair=pair,
                            deck=runtime_deck,
                            run_identity_sha256=run_identity,
                            runtime_preflight_sha256=runtime_preflight_sha256,
                        )
                        _validate_completed_pair_binding(
                            pair=pair,
                            seal=completed_seal,
                            receipts_by_nonce=existing,
                        )
                        continue
                    if args.max_new_games and new_games >= args.max_new_games:
                        stop = True
                        break
                    seal_path = output_dir / "pair-seals" / f"{pair[0].pair_id}.json"
                    seal = _load_or_capture_pair_seal(
                        path=seal_path,
                        env=env,
                        pair=pair,
                        deck=runtime_deck,
                        run_identity_sha256=run_identity,
                        runtime_preflight_sha256=runtime_preflight_sha256,
                    )
                    _validate_completed_pair_binding(
                        pair=pair,
                        seal=seal,
                        receipts_by_nonce=existing,
                    )
                    for spec in pending:
                        if args.max_new_games and new_games >= args.max_new_games:
                            stop = True
                            break
                        first_player, winner, steps, telemetry, stage_traces = _play_one_game(
                            env=env,
                            spec=spec,
                            seal=seal,
                            deck=runtime_deck,
                            candidate=candidate,
                            control=control,
                            max_steps=args.max_steps,
                        )
                        trace_payload = _trace_document(
                            run_identity_sha256=run_identity,
                            runtime_preflight_sha256=runtime_preflight_sha256,
                            spec=spec,
                            stage_traces=stage_traces,
                        )
                        receipt_path, trace_path = _game_paths(output_dir, spec)
                        write_create_only_json(
                            trace_path, trace_payload, label="r289 candidate checklist trace"
                        )
                        receipt = make_game_receipt(
                            run_identity_sha256=run_identity,
                            spec=spec,
                            first_player_seat=first_player,
                            winner_seat=winner,
                            steps=steps,
                            checklist_telemetry=telemetry,
                            runtime_preflight_sha256=runtime_preflight_sha256,
                            direct_policy_flags=DIRECT_POLICY_FLAGS,
                            pair_first_player_seal_sha256=seal.identity_sha256,
                            stage_trace_digest=trace_payload[
                                "candidate_stage_trace_sha256"
                            ],
                        )
                        write_create_only_json(
                            receipt_path, receipt, label="r289 completed game receipt"
                        )
                        # Reparse immediately so no in-memory row is counted
                        # unless the persisted receipt and trace agree exactly.
                        existing[spec.game_nonce_sha256] = _read_existing_game_receipt(
                            output_dir=output_dir,
                            spec=spec,
                            run_identity_sha256=run_identity,
                            runtime_preflight_sha256=runtime_preflight_sha256,
                        ) or {}
                        new_games += 1
                        print(
                            json.dumps(
                                {
                                    "completed_game_nonce_sha256": spec.game_nonce_sha256,
                                    "new_games": new_games,
                                    "output_dir": str(output_dir),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if stop:
                        break
                _assert_runtime_sources_unchanged(runtime_sources)
                receipts = [
                    _read_existing_game_receipt(
                        output_dir=output_dir,
                        spec=spec,
                        run_identity_sha256=run_identity,
                        runtime_preflight_sha256=runtime_preflight_sha256,
                    )
                    for spec in schedule
                ]
                completed = [row for row in receipts if row is not None]
                if len(completed) == 250:
                    for pair_index in range(125):
                        pair = _pair_specs(schedule, pair_index)
                        seal_path = output_dir / "pair-seals" / f"{pair[0].pair_id}.json"
                        if not seal_path.exists():
                            raise R289RunnerError(
                                "completed report is missing a required pair seal"
                            )
                        report_seal = _load_or_capture_pair_seal(
                            path=seal_path,
                            env=env,
                            pair=pair,
                            deck=runtime_deck,
                            run_identity_sha256=run_identity,
                            runtime_preflight_sha256=runtime_preflight_sha256,
                        )
                        _validate_completed_pair_binding(
                            pair=pair,
                            seal=report_seal,
                            receipts_by_nonce={
                                spec.game_nonce_sha256: row
                                for spec, row in zip(schedule, receipts)
                                if row is not None
                            },
                        )
                    report = compile_report(
                        run_identity_sha256=run_identity,
                        runtime_preflight_sha256=runtime_preflight_sha256,
                        seed_identity_sha256=args.seed_identity_sha256,
                        schedule=schedule,
                        game_receipts=completed,
                        input_identities=content_input_identities,
                    )
                    write_create_only_json(
                        output_dir / "report.json", report, label="r289 completed report"
                    )
                    print(
                        json.dumps(
                            {
                                "status": "completed_diagnostic_only",
                                "evaluation_id": EVALUATION_ID,
                                "completed_games": 250,
                                "output_dir": str(output_dir),
                                "report": str(output_dir / "report.json"),
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "status": "partial_resumable_no_report_credit",
                                "evaluation_id": EVALUATION_ID,
                                "completed_games": len(completed),
                                "pending_games": 250 - len(completed),
                                "new_games": new_games,
                                "output_dir": str(output_dir),
                            },
                            sort_keys=True,
                        )
                    )
            finally:
                env.close()
        return 0
    except (R289BO250Error, R289RunnerError) as exc:
        print(f"r289 BO250 failed closed: {exc}", file=sys.stderr)
        return 2


def _parse_r298_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only the revision-4 receipt-bound derivative invocation surface."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/evaluations/alakazam-turn-checklist-bo250-r289.json",
    )
    parser.add_argument(
        "--goal-contract",
        type=Path,
        default=ROOT / "goals/alakazam-elmo-rule-derivative/contract.json",
    )
    parser.add_argument("--r298-validation-receipt", type=Path, required=True)
    parser.add_argument("--raw-corpus-receipt", type=Path, required=True)
    parser.add_argument("--collision-census-receipt", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-r274-checkpoint", type=Path, required=True)
    parser.add_argument("--r195-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--r195-contract",
        type=Path,
        default=ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
    )
    parser.add_argument("--r195-no-rtp-bundle", type=Path, required=True)
    parser.add_argument(
        "--exact-new-list-deck",
        type=Path,
        default=ROOT / "decks/archetype-samples/alakazam-new-list-direct-r241.csv",
    )
    parser.add_argument(
        "--r195-native-deck",
        type=Path,
        default=ROOT / "decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv",
    )
    parser.add_argument(
        "--checklist-config",
        type=Path,
        required=True,
        help=(
            "Temporary trace-only legacy config. Every gate must be exact zero; "
            "it cannot arm r298 logits."
        ),
    )
    parser.add_argument("--candidate-matchup-tree", type=Path, required=True)
    parser.add_argument("--baseline-r274-matchup-tree", type=Path, required=True)
    parser.add_argument("--r195-matchup-tree", type=Path, required=True)
    parser.add_argument("--seeded-engine", type=Path, required=True)
    parser.add_argument("--seeded-engine-receipt", type=Path, required=True)
    parser.add_argument(
        "--benchmark-seed-identity-sha256",
        "--seed-identity-sha256",
        dest="benchmark_seed_identity_sha256",
        required=True,
        help="Fresh immutable root; the runner derives and records distinct A/B/C seeds.",
    )
    parser.add_argument(
        "--calibration-receipt",
        type=Path,
        required=True,
        help="Receipt-bound evidence only; legacy r288 gates are never armed here.",
    )
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--max-new-games",
        type=int,
        default=0,
        help="Global resumable cap across all 750 games; 0 means all pending games.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Stage and build all three receipt-bound direct runtimes without a game.",
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.max_new_games < 0:
        parser.error("--max-new-games cannot be negative")
    return args


def _assert_r298_phase_links(
    *, r298_receipt_path: Path, collision_identity: Mapping[str, Any]
) -> None:
    """Make the r298 aggregate receipt point at the exact census receipt."""

    payload = _json_object(r298_receipt_path, label="r298 validation receipt")
    phases = payload.get("phase_evidence")
    collision = phases.get("collision_census") if isinstance(phases, Mapping) else None
    artifact = collision.get("artifact") if isinstance(collision, Mapping) else None
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("sha256") != collision_identity.get("sha256")
        or artifact.get("size_bytes") != collision_identity.get("size_bytes")
    ):
        raise R289RunnerError(
            "r298 validation receipt does not bind the exact completed collision census"
        )


def _require_wired_strict_r298_runtime_wrapper(*, r298_receipt_path: Path) -> None:
    """Refuse the r288 fallback while no sealed r298 PolicyAgent route exists.

    The receipt can prove that a strict action-time wrapper was tested, but
    the current PolicyAgent constructor only exposes the legacy r288 checklist
    API.  A receipt is not authorization to reinterpret that API as r298.
    Keep the benchmark blocked before any staging until the wrapper owner adds
    a typed runtime integration and this runner can validate it directly.
    """

    receipt = _json_object(r298_receipt_path, label="r298 validation receipt")
    evidence = receipt.get("checklist_provenance")
    if not isinstance(evidence, Mapping):
        raise R289RunnerError("r298 receipt lacks strict checklist wrapper evidence")
    required_true = (
        "candidate_action_time_wrapper_sealed",
        "acting_player_public_information_only",
        "strict_q1_q2_q8_provenance_enforced",
        "q5_q6_exact_zero",
    )
    if (
        any(evidence.get(name) is not True for name in required_true)
        or evidence.get("legacy_r288_runtime_residual") != 0.0
    ):
        raise R289RunnerError("r298 strict checklist wrapper evidence is incomplete")
    raise R289RunnerError(
        "sealed r298 action-time checklist wrapper is not yet wired into PolicyAgent; "
        "refusing the legacy r288 fallback"
    )


def main_r298(argv: Sequence[str] | None = None) -> int:
    """Run the revision-4-gated three-comparison diagnostic or fail before staging."""

    args = _parse_r298_args(argv)
    try:
        elmo_hostname = _require_elmo_host()
        config_path = _path(args.config)
        goal_contract_path = _path(args.goal_contract)
        validation_receipt_path = _path(args.r298_validation_receipt)
        raw_corpus_receipt_path = _path(args.raw_corpus_receipt)
        collision_receipt_path = _path(args.collision_census_receipt)
        candidate_checkpoint_path = _path(args.candidate_checkpoint)
        baseline_checkpoint_path = _path(args.baseline_r274_checkpoint)
        r195_checkpoint_path = _path(args.r195_checkpoint)
        r195_contract_path = _path(args.r195_contract)
        r195_bundle_path = _path(args.r195_no_rtp_bundle)
        new_list_path = _path(args.exact_new_list_deck)
        r195_native_deck_path = _path(args.r195_native_deck)
        checklist_config_path = _path(args.checklist_config)
        candidate_tree_path = _path(args.candidate_matchup_tree)
        baseline_tree_path = _path(args.baseline_r274_matchup_tree)
        r195_tree_path = _path(args.r195_matchup_tree)
        seeded_engine_path = _path(args.seeded_engine)
        seeded_engine_receipt_path = _path(args.seeded_engine_receipt)
        calibration_receipt_path = _path(args.calibration_receipt)
        calibration_artifact_path = _path(args.calibration_artifact)

        _config, config_identity = load_r289_config(config_path)
        goal_identity = validate_derivative_goal_contract(goal_contract_path)
        if goal_identity.get("sha256") != DERIVATIVE_GOAL_CONTRACT_SHA256:
            raise R289RunnerError("dedicated derivative contract digest drifted")
        comparison_seeds = {
            comparison_id: derive_comparison_seed_identity(
                args.benchmark_seed_identity_sha256, comparison_id=comparison_id
            )
            for comparison_id in COMPARISON_IDS
        }
        schedules = {
            comparison_id: build_schedule(
                comparison_seeds[comparison_id], comparison_id=comparison_id
            )
            for comparison_id in COMPARISON_IDS
        }
        schedule_digests = {
            comparison_id: schedule_identity(
                schedules[comparison_id],
                seed_identity_sha256=comparison_seeds[comparison_id],
                comparison_id=comparison_id,
            )
            for comparison_id in COMPARISON_IDS
        }
        raw_corpus_identity = validate_r298_raw_corpus_receipt(raw_corpus_receipt_path)
        collision_identity = validate_r298_collision_census_receipt(
            collision_receipt_path,
            raw_corpus_receipt_identity=raw_corpus_identity,
        )
        candidate_identity = file_identity(
            candidate_checkpoint_path, label="r298 derivative candidate checkpoint"
        )
        baseline_identity = file_identity(
            baseline_checkpoint_path, label="unchanged r241/r274 baseline checkpoint"
        )
        validation_identity = validate_r298_validation_receipt(
            validation_receipt_path,
            goal_contract_identity=goal_identity,
            candidate_checkpoint=candidate_identity,
            baseline_r274_checkpoint=baseline_identity,
            comparison_seed_identities=comparison_seeds,
        )
        _require_wired_strict_r298_runtime_wrapper(
            r298_receipt_path=validation_receipt_path
        )
        _assert_r298_phase_links(
            r298_receipt_path=validation_receipt_path,
            collision_identity=collision_identity,
        )
        r195_identity = file_identity(r195_checkpoint_path, label="immutable r195 checkpoint")
        r195_contract_identity = validate_r195_contract(r195_contract_path, r195_identity)
        r195_bundle_identity = file_identity(r195_bundle_path, label="r195 NO-RTP bundle")
        if r195_bundle_identity.get("sha256") != R195_NO_RTP_BUNDLE_SHA256:
            raise R289RunnerError("r195 bundle is not the immutable NO-RTP bundle")
        new_list, new_list_identity = read_exact_new_list_deck(new_list_path)
        r195_native, r195_native_identity = read_r195_native_deck(r195_native_deck_path)
        checklist_identity = validate_checklist_config(checklist_config_path)
        _assert_legacy_checklist_trace_only(checklist_config_path)
        candidate_tree_identity = file_identity(candidate_tree_path, label="r298 candidate matchup tree")
        baseline_tree_identity = file_identity(baseline_tree_path, label="r274 baseline matchup tree")
        if candidate_tree_identity.get("sha256") != baseline_tree_identity.get("sha256"):
            raise R289RunnerError("r298 frozen matchup adapters must use the exact baseline tree")
        r195_tree_identity = file_identity(r195_tree_path, label="r195 matchup tree")
        r195_payload = _json_object(r195_contract_path, label="r195 contract")
        completion = r195_payload.get("completion")
        expected_r195_tree = completion.get("matchup_tree_sha256") if isinstance(completion, Mapping) else None
        if r195_tree_identity.get("sha256") != expected_r195_tree:
            raise R289RunnerError("r195 tree does not match its immutable operational contract")
        seeded_engine_identity = file_identity(seeded_engine_path, label="seeded engine")
        seeded_engine_receipt_identity = validate_seeded_engine_receipt(
            seeded_engine_receipt_path, engine_identity=seeded_engine_identity
        )
        calibration_identity = validate_required_calibration_receipt(
            calibration_receipt_path, comparison_seed_identities=comparison_seeds
        )
        calibration_artifact_identity = file_identity(
            calibration_artifact_path, label="legacy calibration artifact (evidence only)"
        )
        runtime_sources = _runtime_source_identities()
        run_identity = build_three_cohort_run_identity(
            config_identity=config_identity,
            goal_contract_identity=goal_identity,
            r298_validation_receipt=validation_identity,
            raw_corpus_receipt=raw_corpus_identity,
            collision_census_receipt=collision_identity,
            candidate_checkpoint=candidate_identity,
            baseline_r274_checkpoint=baseline_identity,
            r195_checkpoint=r195_identity,
            r195_contract=r195_contract_identity,
            r195_bundle=r195_bundle_identity,
            exact_new_list_deck=new_list_identity,
            r195_native_deck=r195_native_identity,
            checklist_config=checklist_identity,
            candidate_matchup_tree=candidate_tree_identity,
            baseline_r274_matchup_tree=baseline_tree_identity,
            r195_matchup_tree=r195_tree_identity,
            seeded_engine=seeded_engine_identity,
            seeded_engine_receipt=seeded_engine_receipt_identity,
            calibration_receipt=calibration_identity,
            calibration_artifact=calibration_artifact_identity,
            runtime_sources=runtime_sources,
            benchmark_seed_identity_sha256=args.benchmark_seed_identity_sha256,
            comparison_seed_identities=comparison_seeds,
            comparison_schedule_sha256s=schedule_digests,
        )
        output_root = _path(args.output_root)
        _assert_isolated_output_root(output_root)
        output_dir = output_root / f"{EVALUATION_ID}-{run_identity[7:]}"
        if output_dir.is_symlink():
            raise R289RunnerError("content-addressed r289 output may not be a symlink")
        input_identities: dict[str, Any] = {
            "r289_config": config_identity,
            "goal_contract": goal_identity,
            "r298_validation_receipt": validation_identity,
            "raw_corpus_receipt": raw_corpus_identity,
            "collision_census_receipt": collision_identity,
            "candidate_checkpoint": candidate_identity,
            "baseline_r274_checkpoint": baseline_identity,
            "r195_checkpoint": r195_identity,
            "r195_contract": r195_contract_identity,
            "r195_no_rtp_bundle": r195_bundle_identity,
            "exact_new_list_deck": new_list_identity,
            "r195_native_deck": r195_native_identity,
            "checklist_base_config_trace_only": checklist_identity,
            "candidate_matchup_tree": candidate_tree_identity,
            "baseline_r274_matchup_tree": baseline_tree_identity,
            "r195_matchup_tree": r195_tree_identity,
            "seeded_engine": seeded_engine_identity,
            "seeded_engine_receipt": seeded_engine_receipt_identity,
            "calibration_receipt_evidence_only": calibration_identity,
            "calibration_artifact_evidence_only": calibration_artifact_identity,
            "runtime_sources": runtime_sources,
            "benchmark_seed_identity_sha256": args.benchmark_seed_identity_sha256,
            "comparison_seed_identities": comparison_seeds,
            "comparison_schedule_sha256s": schedule_digests,
        }
        content_input_identities = _content_only(input_identities)
        sources: dict[str, Path] = {
            "r289_config": config_path,
            "goal_contract": goal_contract_path,
            "r298_validation_receipt": validation_receipt_path,
            "raw_corpus_receipt": raw_corpus_receipt_path,
            "collision_census_receipt": collision_receipt_path,
            "candidate_checkpoint": candidate_checkpoint_path,
            "baseline_r274_checkpoint": baseline_checkpoint_path,
            "r195_checkpoint": r195_checkpoint_path,
            "r195_contract": r195_contract_path,
            "r195_no_rtp_bundle": r195_bundle_path,
            "exact_new_list_deck": new_list_path,
            "r195_native_deck": r195_native_deck_path,
            "checklist_base_config": checklist_config_path,
            "candidate_matchup_tree": candidate_tree_path,
            "baseline_r274_matchup_tree": baseline_tree_path,
            "r195_matchup_tree": r195_tree_path,
            "seeded_engine": seeded_engine_path,
            "seeded_engine_receipt": seeded_engine_receipt_path,
            "calibration_receipt": calibration_receipt_path,
            "calibration_artifact": calibration_artifact_path,
        }
        staging_identities: dict[str, Mapping[str, Any]] = {
            "r289_config": config_identity,
            "goal_contract": goal_identity,
            "r298_validation_receipt": validation_identity,
            "raw_corpus_receipt": raw_corpus_identity,
            "collision_census_receipt": collision_identity,
            "candidate_checkpoint": candidate_identity,
            "baseline_r274_checkpoint": baseline_identity,
            "r195_checkpoint": r195_identity,
            "r195_contract": r195_contract_identity,
            "r195_no_rtp_bundle": r195_bundle_identity,
            "exact_new_list_deck": new_list_identity,
            "r195_native_deck": r195_native_identity,
            "checklist_base_config": checklist_identity,
            "candidate_matchup_tree": candidate_tree_identity,
            "baseline_r274_matchup_tree": baseline_tree_identity,
            "r195_matchup_tree": r195_tree_identity,
            "seeded_engine": seeded_engine_identity,
            "seeded_engine_receipt": seeded_engine_receipt_identity,
            "calibration_receipt": calibration_identity,
            "calibration_artifact": calibration_artifact_identity,
        }
        for relative, identity in runtime_sources.items():
            key = f"runtime_source:{relative}"
            sources[key] = ROOT / relative
            staging_identities[key] = identity
        output_dir.mkdir(parents=True, exist_ok=True)
        study_schedule_digest = canonical_digest(schedule_digests)
        write_create_only_json(
            output_dir / "output-index.json",
            output_index_payload(
                run_identity_sha256=run_identity,
                output_dir=output_dir,
                schedule_sha256=study_schedule_digest,
                input_identities=content_input_identities,
            ),
            label="r289 three-cohort output index",
        )
        staged = _stage_inputs(
            output_dir=output_dir, sources=sources, identities=staging_identities
        )
        staged_new_list, staged_new_list_identity = read_exact_new_list_deck(
            Path(staged["exact_new_list_deck"]["path"])
        )
        staged_native, staged_native_identity = read_r195_native_deck(
            Path(staged["r195_native_deck"]["path"])
        )
        if staged_new_list != new_list or staged_new_list_identity.get("sha256") != new_list_identity.get("sha256"):
            raise R289RunnerError("staged new-list deck drifted")
        if staged_native != r195_native or staged_native_identity.get("sha256") != r195_native_identity.get("sha256"):
            raise R289RunnerError("staged r195 native deck drifted")
        with _direct_policy_environment() as sanitized_environment:
            lib, engine_runtime = _load_seeded_engine(Path(staged["seeded_engine"]["path"]))
            remaining = None if args.max_new_games == 0 else args.max_new_games
            reports: dict[str, dict[str, Any]] = {}
            completed_by_comparison: dict[str, int] = {}
            new_games_total = 0
            for comparison_id in COMPARISON_IDS:
                if comparison_id == "A":
                    control_checkpoint_key = "baseline_r274_checkpoint"
                    control_tree_key = "baseline_r274_matchup_tree"
                    control_deck = staged_new_list
                    control_deck_identity = staged_new_list_identity
                elif comparison_id == "B":
                    control_checkpoint_key = "r195_checkpoint"
                    control_tree_key = "r195_matchup_tree"
                    control_deck = staged_native
                    control_deck_identity = staged_native_identity
                else:
                    control_checkpoint_key = "r195_checkpoint"
                    control_tree_key = "r195_matchup_tree"
                    control_deck = staged_new_list
                    control_deck_identity = staged_new_list_identity
                report, remaining, new_games, completed_games = _run_comparison(
                    comparison_id=comparison_id,
                    output_dir=output_dir / "comparisons" / comparison_id,
                    run_identity_sha256=run_identity,
                    comparison_seed_identity_sha256=comparison_seeds[comparison_id],
                    schedule=schedules[comparison_id],
                    schedule_sha256=schedule_digests[comparison_id],
                    input_identities=content_input_identities,
                    staged=staged,
                    sanitized_environment=sanitized_environment,
                    lib=lib,
                    engine_runtime=engine_runtime,
                    candidate_deck=staged_new_list,
                    candidate_deck_identity=staged_new_list_identity,
                    control_deck=control_deck,
                    control_deck_identity=control_deck_identity,
                    control_checkpoint_key=control_checkpoint_key,
                    control_tree_key=control_tree_key,
                    runtime_sources=runtime_sources,
                    host=elmo_hostname,
                    device_name=args.device,
                    max_steps=args.max_steps,
                    remaining_new_games=remaining,
                    preflight_only=args.preflight_only,
                )
                new_games_total += new_games
                completed_by_comparison[comparison_id] = completed_games
                if report is not None:
                    reports[comparison_id] = report
            _assert_runtime_sources_unchanged(runtime_sources)
        if args.preflight_only:
            print(json.dumps({"status": "preflight_passed_no_games_started", "evaluation_id": EVALUATION_ID, "output_dir": str(output_dir), "run_identity_sha256": run_identity}, sort_keys=True))
            return 0
        if set(reports) == set(COMPARISON_IDS):
            combined = compile_three_comparison_report(
                run_identity_sha256=run_identity,
                benchmark_seed_identity_sha256=args.benchmark_seed_identity_sha256,
                comparison_reports=reports,
                input_identities=content_input_identities,
            )
            write_create_only_json(
                output_dir / "report.json", combined, label="r289 three-comparison report"
            )
            print(json.dumps({"status": "completed_three_comparison_diagnostic_only", "evaluation_id": EVALUATION_ID, "completed_games": 750, "new_games": new_games_total, "output_dir": str(output_dir), "report": str(output_dir / "report.json")}, sort_keys=True))
        else:
            completed_games = sum(completed_by_comparison.values())
            print(json.dumps({"status": "partial_resumable_no_combined_report_credit", "evaluation_id": EVALUATION_ID, "completed_games": completed_games, "pending_games": 750 - completed_games, "new_games": new_games_total, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except (R289BO250Error, R289RunnerError) as exc:
        print(f"r289 derivative BO250 failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main_r298())
