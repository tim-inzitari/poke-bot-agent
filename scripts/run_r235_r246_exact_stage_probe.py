#!/usr/bin/env python3
"""Run the actual-only R240/R246 exact-stage probe route suite.

This is deliberately package-external.  It starts every scenario through a
fresh :class:`R235ExactChildWatchdog` child/session/group, never uses a shell,
and emits the final R240 probe object only when every required witness was
actually observed.  A normal physical game is useful evidence, but it is not
silently relabelled as high-confidence, continuation, actor-boundary, or R246
terminal-win coverage.  Missing evidence is a hard failure.

The default scenario is the sealed physical-game smoke runner.  The reviewed
controlled-parent worker always runs in a separate exact child for the narrow
direct/continuation regressions that the owner permits it to cover.  Optional
additional stock-route workers are explicit project-script argv arrays, each
of which must bind the same archive, manifest, and contracts.  The terminal-
win witness is never controlled: it must link to a real stock ``SearchStep``
terminal result, or the probe fails closed.

The command writes no receipt itself.  Invoke it under
``run_r235_r238_phase1_preflight.py --probe-command-json``; that preflight
owns the immutable primary and derived binder-gate receipts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r235_exact_child_watchdog import (  # noqa: E402
    ExactChildOutcome,
    R235ExactChildWatchdog,
)
from poke_bot import r235_kaggle_phase1_preflight as preflight  # noqa: E402
from poke_bot import r244_handle_scoped_search_id_preflight as r244_preflight  # noqa: E402
from scripts.r228_kaggle_r244_harness_common import load_binding_identity  # noqa: E402


RAW_SMOKE_SCHEMA = "poke_bot.r244_exact_package_raw_physical_game_telemetry/v1"
SCENARIO_EVIDENCE_SCHEMA = "poke_bot.r235_r246_exact_stage_scenario_evidence/v1"
CONTROLLED_PARENT_ROUTE_SCHEMA = "poke_bot.r240_controlled_parent_route_worker/v1"
CONTROLLED_PARENT_ROUTE_PREFIX = "R240_CONTROLLED_PARENT_ROUTE_RESULT "
ACTUAL_STOCK_RUNTIME_OBSERVATION_SCHEMA = (
    "poke_bot.r242_actual_stock_actor_change_runtime_observation/v1"
)
ACTUAL_STOCK_RUNTIME_OBSERVATION_ORIGIN = "actual_stock_runtime_observation"
ACTUAL_STOCK_RUNTIME_EVIDENCE_KIND = (
    "actual_stock_actor_change_boundary_resource_startup_preflight_observation"
)
ACTUAL_STOCK_RUNTIME_SEARCH_STEP_ORIGIN = (
    "fresh_official_r236_search_step_actor_change_successor"
)
ACTUAL_STOCK_RUNTIME_WORKER = (
    ROOT / "scripts/run_r246_actual_stock_boundary_resource_route.py"
)
R244_WITNESS_SCHEMA = "poke_bot.r244_handle_scoped_search_id_identity_probe/v1"
R244_CONTRACT_PROJECTION_KIND = (
    "r225_r244_static_handle_namespace_contract_projection"
)
MAX_SCENARIO_OUTPUT_BYTES = 8 * 1024 * 1024

# The intentionally narrow in-memory parent worker is allowed to establish
# only these parent-owned regressions.  It must never be promoted into a
# physical libcg, resource, topology, actor-boundary, or terminal-win fact.
CONTROLLED_PARENT_ALLOWED_WITNESSES = frozenset(
    {
        "synthetic_high_confidence_direct",
        "full_game_cumulative",
    }
)
STOCK_ONLY_WITNESSES = frozenset(
    {
        "synthetic_ambiguous_two_lane_mcts",
        preflight.R246_TERMINAL_WIN_PROBE_KEY,
        "actor_change_end_turn_boundary",
        "observed_resource_probe",
        "startup_seconds",
    }
)


class ExactStageProbeError(RuntimeError):
    """The actual route suite lacks one required, independently observed fact."""


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    payload: dict[str, Any]
    outcome: ExactChildOutcome


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha256_bytes(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactStageProbeError(f"{label} must be an object")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExactStageProbeError(f"{label} must be a nonempty string")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExactStageProbeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ExactStageProbeError(f"{label} must be a nonnegative finite number")
    return result


def _physical_directory(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise ExactStageProbeError(f"{label} must be an existing physical directory")
    return raw.resolve()


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ExactStageProbeError(f"{label} must be an existing regular non-symlink file")
    return raw.resolve()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _stage_disk_bytes(stage: Path) -> int:
    total = 0
    for member in stage.rglob("*"):
        if member.is_symlink():
            raise ExactStageProbeError("sealed stage contains a symlink")
        if member.is_file():
            total += member.stat().st_size
    return total


def _load_exact_binding_identity(
    *, stage: Path, archive: Path, manifest: Path, r225: Path, r236: Path
) -> dict[str, Any]:
    """Recompute the package binding before any staged route is launched."""

    if _sha256_file(r225) != preflight.R225_CANONICAL_SHA256:
        raise ExactStageProbeError("r225 contract is not the frozen R246 canonical bytes")
    try:
        raw = load_binding_identity(
            stage=stage,
            candidate_archive=archive,
            member_manifest=manifest,
            r225_contract=r225,
            r236_contract=r236,
        )
    except Exception as exc:  # noqa: BLE001 - foreign validator has typed local errors
        raise ExactStageProbeError(f"exact archive/stage binding failed: {exc}") from exc
    identity = dict(_mapping(raw, label="recomputed package binding identity"))
    for field in ("common_identity", "exact_package", "stage_contract"):
        identity[field] = dict(_mapping(identity.get(field), label=f"binding {field}"))
    return identity


def _require_binding_identity(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    label: str,
    require_common_identity: bool = False,
) -> None:
    """Require a worker to link every claim to the exact sealed package."""

    observed_exact = _mapping(payload.get("exact_package_identity"), label=f"{label} exact identity")
    observed_stage = _mapping(payload.get("stage_contract"), label=f"{label} stage contract")
    if dict(observed_exact) != dict(expected["exact_package"]):
        raise ExactStageProbeError(f"{label} exact package identity does not match this run")
    if dict(observed_stage) != dict(expected["stage_contract"]):
        raise ExactStageProbeError(f"{label} stage contract does not match this run")
    # Generic actual stock-route workers carry common identity directly.  The
    # raw smoke envelope predates that field, so its stage/archive identity is
    # checked above instead.
    if require_common_identity and "common_identity" not in payload:
        raise ExactStageProbeError(f"{label} lacks common identity")
    if "common_identity" in payload:
        observed_common = _mapping(payload.get("common_identity"), label=f"{label} common identity")
        if dict(observed_common) != dict(expected["common_identity"]):
            raise ExactStageProbeError(f"{label} common identity does not match this run")


def _read_argv_json(path: Path, *, label: str) -> list[str]:
    source = _regular_file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactStageProbeError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, list) or not payload or not all(
        isinstance(item, str) and item for item in payload
    ):
        raise ExactStageProbeError(f"{label} must be a nonempty string argv array")
    return list(payload)


def _require_explicit_stage_argument(argv: Sequence[str], *, stage: Path, label: str) -> None:
    """Keep auxiliary scenario commands pinned to this exact staged package."""

    try:
        index = list(argv).index("--stage")
        declared = Path(argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise ExactStageProbeError(f"{label} must declare --stage /exact/stage") from exc
    if declared != stage:
        raise ExactStageProbeError(f"{label} is not pinned to the requested stage")


def _require_exact_path_argument(
    argv: Sequence[str], *, flag: str, expected: Path, label: str
) -> None:
    try:
        index = list(argv).index(flag)
        declared = Path(argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise ExactStageProbeError(f"{label} must declare {flag} with this exact path") from exc
    if declared != expected:
        raise ExactStageProbeError(f"{label} is not pinned to the requested {flag} path")


def _require_owned_actual_route_command(
    argv: Sequence[str],
    *,
    stage: Path,
    archive: Path,
    manifest: Path,
    r225: Path,
    r236: Path,
    label: str,
) -> None:
    """Forbid ad-hoc ``-c``/shell claims as a stock-route evidence source."""

    if len(argv) < 2 or argv[1].startswith("-"):
        raise ExactStageProbeError(f"{label} must invoke a reviewed project Python script")
    script = _regular_file(Path(argv[1]), label=f"{label} script")
    scripts_root = (ROOT / "scripts").resolve()
    if not _inside(script, scripts_root) or script.suffix != ".py":
        raise ExactStageProbeError(f"{label} script must be a physical project scripts/*.py file")
    _require_explicit_stage_argument(argv, stage=stage, label=label)
    for flag, expected in (
        ("--candidate-archive", archive),
        ("--member-manifest", manifest),
        ("--r225-contract", r225),
        ("--r236-contract", r236),
    ):
        _require_exact_path_argument(argv, flag=flag, expected=expected, label=label)


def _parse_json_output(
    outcome: ExactChildOutcome, *, label: str, required_prefix: str | None = None
) -> dict[str, Any]:
    if not outcome.completed:
        raise ExactStageProbeError(f"{label} did not complete cleanly: {outcome.status}")
    if outcome.stdout_truncated or outcome.stderr_truncated:
        raise ExactStageProbeError(f"{label} exceeded bounded exact-child output capture")
    if len(outcome.stdout) > MAX_SCENARIO_OUTPUT_BYTES:
        raise ExactStageProbeError(f"{label} emitted an overlarge scenario payload")
    try:
        text = outcome.stdout.decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactStageProbeError(f"{label} did not emit UTF-8 JSON output") from exc
    if required_prefix is not None:
        lines = text.splitlines()
        if len(lines) != 1 or not lines[0].startswith(required_prefix):
            raise ExactStageProbeError(
                f"{label} did not emit exactly one required controlled-evidence row"
            )
        text = lines[0][len(required_prefix) :]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExactStageProbeError(f"{label} did not emit exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ExactStageProbeError(f"{label} JSON is not an object")
    return payload


def _run_fresh_scenario(
    *,
    name: str,
    argv: Sequence[str],
    stage: Path,
    timeout_seconds: float,
    term_grace_seconds: float,
    kill_grace_seconds: float,
    required_stdout_prefix: str | None = None,
    watchdog_factory: type[R235ExactChildWatchdog] = R235ExactChildWatchdog,
) -> ScenarioResult:
    """Run one command only through a fresh owned child/session/process group."""

    _require_explicit_stage_argument(argv, stage=stage, label=f"scenario {name}")
    watchdog = watchdog_factory(
        timeout_seconds=timeout_seconds,
        term_grace_seconds=term_grace_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )
    outcome = watchdog.run(list(argv), cwd=stage)
    return ScenarioResult(
        name=name,
        payload=_parse_json_output(
            outcome,
            label=f"scenario {name}",
            required_prefix=required_stdout_prefix,
        ),
        outcome=outcome,
    )


def _validate_raw_smoke(
    raw: Mapping[str, Any],
    *,
    stage: Path,
    binding_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject anything less than a sealed, terminal physical-game observation."""

    if raw.get("schema") != RAW_SMOKE_SCHEMA:
        raise ExactStageProbeError("physical smoke did not emit the raw R240 telemetry schema")
    if raw.get("status") != "passed" or raw.get("failure") is not None:
        raise ExactStageProbeError("physical smoke did not pass cleanly")
    mutation = _mapping(raw.get("package_mutation_check"), label="physical smoke mutation check")
    if mutation.get("unchanged") is not True:
        raise ExactStageProbeError("physical smoke mutated the sealed package")
    stock = _mapping(raw.get("stock_game"), label="physical smoke stock game")
    terminal = _mapping(stock.get("terminal"), label="physical smoke terminal")
    if terminal.get("physical_terminal_confirmed") is not True:
        raise ExactStageProbeError("physical smoke did not reach a stock terminal game")
    full_markers = raw.get("full_game_success_markers")
    if not isinstance(full_markers, list) or len(full_markers) != 1:
        raise ExactStageProbeError("physical smoke lacks exactly one full-game success marker")
    for field in ("hard_failure_markers", "degraded_fallback_markers"):
        rows = raw.get(field)
        if not isinstance(rows, list) or rows:
            raise ExactStageProbeError(f"physical smoke has forbidden {field}")
    callbacks = raw.get("callbacks")
    if not isinstance(callbacks, list) or not callbacks:
        raise ExactStageProbeError("physical smoke lacks callback telemetry")
    for index, callback_raw in enumerate(callbacks):
        callback = _mapping(callback_raw, label=f"physical smoke callback {index}")
        if callback.get("stock_action_accepted") is not True:
            raise ExactStageProbeError(f"physical smoke callback {index} was not accepted by stock libcg")
    _require_binding_identity(raw, expected=binding_identity, label="physical smoke")
    return {
        "raw_sha256": _sha256_bytes(raw),
        "stage_disk_bytes": _stage_disk_bytes(stage),
        "callbacks": [dict(_mapping(row, label="physical smoke callback")) for row in callbacks],
        "decision_markers": [
            dict(_mapping(row, label="physical smoke decision marker"))
            for row in raw.get("decision_markers", [])
            if isinstance(row, Mapping)
        ],
        "elapsed_seconds": _finite_nonnegative(raw.get("elapsed_seconds"), label="physical smoke elapsed seconds"),
        "process_observation": dict(
            _mapping(raw.get("process_observation"), label="physical smoke process observation")
        ),
    }


def _marker_matches_callback(marker: Mapping[str, Any], callbacks: Sequence[Mapping[str, Any]]) -> int:
    """Each route claim must be linked to a stock-accepted callback marker."""

    marker_digest = _sha256_bytes(marker)
    matches: list[int] = []
    for index, callback in enumerate(callbacks):
        authority = callback.get("decision_marker_or_containment")
        if isinstance(authority, Mapping) and _sha256_bytes(authority) == marker_digest:
            matches.append(index)
    if len(matches) != 1:
        raise ExactStageProbeError(
            "route marker is not linked to exactly one stock-accepted physical callback"
        )
    callback = callbacks[matches[0]]
    selected = marker.get("selected_action")
    accepted = callback.get("action")
    if not isinstance(selected, list) or not isinstance(accepted, list) or selected != accepted:
        raise ExactStageProbeError(
            "route marker selected action does not equal its stock-accepted callback action"
        )
    return matches[0]


def _required_marker_fields(marker: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    missing = [field for field in fields if field not in marker]
    if missing:
        raise ExactStageProbeError(
            f"{label} lacks actual marker evidence for: {', '.join(sorted(missing))}"
        )


def _complete_cuda_observation(raw: object, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ExactStageProbeError(f"{label} lacks a complete actual CUDA observation")
    try:
        return dict(preflight._validate_cuda_runtime_before_search(raw))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - preflight owns the detailed schema error
        raise ExactStageProbeError(f"{label} CUDA observation is incomplete or invalid") from exc


def _cuda_semantic_identity(observation: Mapping[str, Any]) -> dict[str, object]:
    """Compare CUDA placement/topology without treating free-memory samples as static."""

    devices = observation.get("devices")
    if not isinstance(devices, list):  # defensive after the complete validator
        raise ExactStageProbeError("complete CUDA observation devices are malformed")
    return {
        "schema": observation.get("schema"),
        "phase": observation.get("phase"),
        "torch_imported": observation.get("torch_imported"),
        "cuda_available": observation.get("cuda_available"),
        "cuda_initialized": observation.get("cuda_initialized"),
        "device_count": observation.get("device_count"),
        "model_device": observation.get("model_device"),
        "devices": [
            {
                "device_index": device.get("device_index"),
                "device_name": device.get("device_name"),
                "total_memory_bytes": device.get("total_memory_bytes"),
            }
            for device in devices
            if isinstance(device, Mapping)
        ],
    }


def _require_matching_marker_cuda(marker: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """A searched decision needs complete, semantically matching parent/child CUDA facts."""

    parent = _complete_cuda_observation(
        marker.get("parent_cuda_runtime_before_search"), label=f"{label} parent"
    )
    broker = _mapping(marker.get("broker"), label=f"{label} broker")
    child_identity = _mapping(broker.get("child_identity"), label=f"{label} broker child identity")
    child = _complete_cuda_observation(
        child_identity.get("cuda_runtime_before_search"), label=f"{label} child"
    )
    if _cuda_semantic_identity(parent) != _cuda_semantic_identity(child):
        raise ExactStageProbeError(f"{label} parent and child CUDA observations do not match")
    return parent


def _actor_seat(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ExactStageProbeError(f"{label} must be an actor seat 0 or 1")
    return int(value)


def _action_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ExactStageProbeError(f"{label} must be a nonempty action-index list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ExactStageProbeError(f"{label}[{index}] must be an action index")
        result.append(int(item))
    return result


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExactStageProbeError(f"{label} must be a nonnegative integer")
    return int(value)


def _handle_identity(value: object, *, label: str) -> int | str:
    """Accept the official handle identity domain without conflating IDs.

    Official libcg SearchId values are scoped to an agent-start handle.  A
    handle itself may be represented by a numeric opaque identity or a
    nonempty string, but booleans must never pass as numeric handles.
    """

    if isinstance(value, bool):
        raise ExactStageProbeError(f"{label} must be a non-boolean handle identity")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise ExactStageProbeError(f"{label} must be an integer or nonempty string handle identity")


def _require_exact_sha256(value: object, *, expected: str, label: str) -> None:
    if value != expected:
        raise ExactStageProbeError(f"{label} does not match its exact observed source")


def _actual_stock_runtime_observation_witnesses(
    payload: Mapping[str, Any],
    *,
    name: str,
    binding_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Normalize one owner-authorized, actual stock runtime observation.

    The staged parent currently records aggregate actor-boundary counts but
    not the leaf-level rows required by R242.  This separate worker is allowed
    to observe a fresh official-r236 ``SearchStep`` successor through the
    exact staged runtime's frozen evaluator.  The worker has no action
    authority, and this function builds the legacy boundary receipt only from
    its raw observed leaf facts.  It never accepts a hand-authored
    ``actor_change_end_turn_boundary`` wrapper.
    """

    _require_binding_identity(
        payload,
        expected=binding_identity,
        label=f"scenario {name}",
        require_common_identity=True,
    )
    if payload.get("evidence_kind") != ACTUAL_STOCK_RUNTIME_EVIDENCE_KIND:
        raise ExactStageProbeError(f"scenario {name} has the wrong runtime-observation kind")
    mutation = _mapping(payload.get("stage_mutation_check"), label=f"scenario {name} mutation")
    if mutation.get("unchanged") is not True:
        raise ExactStageProbeError(f"scenario {name} mutated the sealed stage")

    runtime = _mapping(
        payload.get("actual_stock_runtime_observation"),
        label=f"scenario {name} actual stock runtime observation",
    )
    if runtime.get("schema") != ACTUAL_STOCK_RUNTIME_OBSERVATION_SCHEMA:
        raise ExactStageProbeError(f"scenario {name} runtime observation schema is not recognized")
    if runtime.get("observation_origin") != ACTUAL_STOCK_RUNTIME_SEARCH_STEP_ORIGIN:
        raise ExactStageProbeError(
            f"scenario {name} did not use a fresh official-r236 actor-change SearchStep successor"
        )
    if runtime.get("sealed_stage_runtime_module") != "poke_bot.r228_kaggle_async_runtime":
        raise ExactStageProbeError(f"scenario {name} did not observe the sealed staged runtime module")
    if runtime.get("sealed_runtime_evaluator_method") != "R228AsyncGameplay._evaluate_batch":
        raise ExactStageProbeError(f"scenario {name} did not observe the staged frozen evaluator")
    for field, expected in (
        ("official_r236_search_step_succeeded", True),
        ("model_value_evaluated", True),
        ("stage_mutation_unchanged", True),
        ("action_authority_granted", False),
        ("opponent_action_selected_or_planned", False),
        ("opponent_action_cached", False),
    ):
        if runtime.get(field) != expected:
            raise ExactStageProbeError(f"scenario {name} runtime observation {field} is not {expected!r}")
    if _exact_nonnegative_int(
        runtime.get("frozen_evaluator_value_call_count"),
        label=f"scenario {name} frozen evaluator value call count",
    ) != 1:
        raise ExactStageProbeError(f"scenario {name} did not make exactly one frozen evaluator value call")
    for field in (
        "expanded_legal_action_count",
        "expanded_child_count",
        "search_steps_beyond_boundary",
    ):
        if _exact_nonnegative_int(runtime.get(field), label=f"scenario {name} {field}") != 0:
            raise ExactStageProbeError(f"scenario {name} runtime observation expanded beyond actor boundary")
    root_actor = _actor_seat(runtime.get("root_actor_seat"), label=f"scenario {name} root actor")
    leaf_actor = _actor_seat(runtime.get("leaf_actor_seat"), label=f"scenario {name} leaf actor")
    if root_actor == leaf_actor:
        raise ExactStageProbeError(f"scenario {name} SearchStep successor did not change actor")
    for field in ("root_observation_fingerprint", "successor_observation_fingerprint"):
        _nonempty_string(runtime.get(field), label=f"scenario {name} {field}")
    search_step = _mapping(
        runtime.get("official_r236_search_step"),
        label=f"scenario {name} official r236 SearchStep observation",
    )
    for field in ("search_begin_succeeded", "search_step_succeeded"):
        if search_step.get(field) is not True:
            raise ExactStageProbeError(f"scenario {name} official SearchStep {field} is not true")
    _handle_identity(
        search_step.get("lane_handle_identity"),
        label=f"scenario {name} official SearchStep lane handle identity",
    )
    _exact_nonnegative_int(
        search_step.get("root_search_id"),
        label=f"scenario {name} official SearchStep root SearchId",
    )
    if _actor_seat(
        search_step.get("root_actor_seat"),
        label=f"scenario {name} official SearchStep root actor",
    ) != root_actor:
        raise ExactStageProbeError(f"scenario {name} official SearchStep root actor drifted")
    if _actor_seat(
        search_step.get("successor_actor_seat"),
        label=f"scenario {name} official SearchStep successor actor",
    ) != leaf_actor:
        raise ExactStageProbeError(f"scenario {name} official SearchStep successor actor drifted")
    expected_libcg_sha256 = _mapping(
        binding_identity.get("common_identity"), label=f"scenario {name} common identity"
    ).get("linux_x86_64_libcg_sha256")
    if not isinstance(expected_libcg_sha256, str) or not expected_libcg_sha256:
        raise ExactStageProbeError(f"scenario {name} common identity lacks the exact Linux r236 libcg SHA-256")
    _require_exact_sha256(
        search_step.get("official_linux_x86_64_libcg_sha256"),
        expected=expected_libcg_sha256,
        label=f"scenario {name} official SearchStep Linux r236 libcg SHA-256",
    )

    # The worker additionally runs one fresh staged parent MCTS callback solely
    # to produce the standalone R244 topology witness.  It must retain that
    # marker verbatim and link it to a stock-accepted callback.
    marker = _mapping(payload.get("literal_staged_marker"), label=f"scenario {name} literal marker")
    marker_sha256 = _sha256_bytes(marker)
    _require_exact_sha256(
        payload.get("literal_staged_marker_sha256"),
        expected=marker_sha256,
        label=f"scenario {name} literal marker SHA-256",
    )
    if marker.get("mode") != "shared_tree_mcts" or marker.get("mcts_action_authority") is not True:
        raise ExactStageProbeError(f"scenario {name} literal marker is not an action-authoritative MCTS marker")
    if marker.get("degraded") is not False:
        raise ExactStageProbeError(f"scenario {name} literal marker is degraded")
    callback = _mapping(payload.get("physical_stock_callback"), label=f"scenario {name} stock callback")
    if callback.get("stock_action_accepted") is not True:
        raise ExactStageProbeError(f"scenario {name} marker is not linked to a stock-accepted callback")
    if _action_list(callback.get("action"), label=f"scenario {name} callback action") != _action_list(
        marker.get("selected_action"), label=f"scenario {name} marker selected action"
    ):
        raise ExactStageProbeError(f"scenario {name} marker action differs from the stock callback")
    if _action_list(
        search_step.get("selected_action"), label=f"scenario {name} official SearchStep selected action"
    ) != _action_list(callback.get("action"), label=f"scenario {name} callback action"):
        raise ExactStageProbeError(
            f"scenario {name} official SearchStep action differs from the stock callback"
        )
    _finite_nonnegative(
        callback.get("callback_elapsed_seconds"), label=f"scenario {name} callback elapsed"
    )

    resources = _mapping(payload.get("observed_resource_probe"), label=f"scenario {name} resources")
    resource_source = _mapping(
        payload.get("actual_parent_broker_resource_startup_observation"),
        label=f"scenario {name} parent/broker resource observation",
    )
    if resource_source.get("measurement_origin") != "fresh_sealed_parent_and_exact_broker_child":
        raise ExactStageProbeError(f"scenario {name} resource observation origin is not exact parent/broker")
    for field, expected in (
        ("startup_ready_before_first_search", True),
        ("broker_child_observed_while_alive", True),
    ):
        if resource_source.get(field) is not expected:
            raise ExactStageProbeError(f"scenario {name} resource observation {field} is not {expected!r}")
    parent_rss = _exact_nonnegative_int(
        resource_source.get("parent_peak_rss_bytes"), label=f"scenario {name} parent peak RSS"
    )
    broker_rss = _exact_nonnegative_int(
        resource_source.get("broker_child_peak_rss_bytes"),
        label=f"scenario {name} broker child peak RSS",
    )
    nested_peak = _exact_nonnegative_int(
        resource_source.get("combined_nested_parent_broker_peak_rss_bytes"),
        label=f"scenario {name} nested parent/broker peak RSS",
    )
    if nested_peak != parent_rss + broker_rss:
        raise ExactStageProbeError(f"scenario {name} nested parent/broker RSS sum drifted")
    if _exact_nonnegative_int(
        resources.get("child_peak_rss_bytes"), label=f"scenario {name} conservative nested RSS"
    ) != nested_peak:
        raise ExactStageProbeError(
            f"scenario {name} resource probe does not conservatively carry nested parent/broker RSS"
        )
    startup_seconds = _finite_nonnegative(
        payload.get("startup_seconds"), label=f"scenario {name} startup seconds"
    )
    if _finite_nonnegative(
        resource_source.get("startup_seconds"), label=f"scenario {name} measured startup seconds"
    ) != startup_seconds:
        raise ExactStageProbeError(f"scenario {name} startup seconds drifted from the raw measurement")
    if _exact_nonnegative_int(
        resources.get("runtime_disk_bytes"), label=f"scenario {name} runtime disk bytes"
    ) != _exact_nonnegative_int(
        resource_source.get("runtime_disk_bytes"), label=f"scenario {name} raw runtime disk bytes"
    ):
        raise ExactStageProbeError(f"scenario {name} resource runtime disk bytes drifted")
    if resources.get("phase1_target") != resource_source.get("phase1_target"):
        raise ExactStageProbeError(f"scenario {name} resource target drifted from raw observation")
    runtime_topology = _mapping(resources.get("runtime"), label=f"scenario {name} runtime topology")
    for field in (
        "configured_vcpus",
        "configured_simulator_lane_count",
        "maximum_simulator_lanes",
        "observed_active_simulator_lane_count",
        "receipt_lane_count",
        "receipt_schema",
        "maximum_simulator_calls_in_flight",
    ):
        if runtime_topology.get(field) != resource_source.get(field):
            raise ExactStageProbeError(f"scenario {name} resource runtime {field} drifted")
    parent_threads = _exact_nonnegative_int(
        resource_source.get("parent_worker_thread_count_peak"),
        label=f"scenario {name} parent worker thread peak",
    )
    broker_threads = _exact_nonnegative_int(
        resource_source.get("broker_child_worker_thread_count_peak"),
        label=f"scenario {name} broker worker thread peak",
    )
    max_threads = max(parent_threads, broker_threads)
    for field in ("worker_thread_count", "observed_peak_worker_threads"):
        if _exact_nonnegative_int(runtime_topology.get(field), label=f"scenario {name} {field}") != max_threads:
            raise ExactStageProbeError(f"scenario {name} resource {field} does not bind measured peaks")
    source_cuda = _complete_cuda_observation(
        resource_source.get("parent_cuda_runtime_before_search"),
        label=f"scenario {name} resource parent",
    )
    source_child_cuda = _complete_cuda_observation(
        resource_source.get("broker_child_cuda_runtime_before_search"),
        label=f"scenario {name} resource broker child",
    )
    if _cuda_semantic_identity(source_cuda) != _cuda_semantic_identity(source_child_cuda):
        raise ExactStageProbeError(f"scenario {name} resource parent/child CUDA observations differ")
    resource_cuda = _complete_cuda_observation(
        resources.get("cuda_runtime_before_search"), label=f"scenario {name} resource CUDA"
    )
    if _cuda_semantic_identity(resource_cuda) != _cuda_semantic_identity(source_cuda):
        raise ExactStageProbeError(f"scenario {name} resource CUDA observation drifted")

    leaf = {
        "model_value_evaluated": True,
        "expanded_legal_action_count": 0,
        "expanded_child_count": 0,
        "search_steps_beyond_boundary": 0,
        "opponent_action_selected_or_planned": False,
        "opponent_action_cached": False,
    }
    return {
        "actor_change_end_turn_boundary": {
            "actor_change_end_turn_boundary_regression_passed": True,
            "declared_opponent_actor_leaf_count": 1,
            "value_evaluated_opponent_actor_leaf_count": 1,
            "expanded_legal_action_count": 0,
            "expanded_child_count": 0,
            "search_steps_beyond_boundary": 0,
            "opponent_action_selected_or_planned_count": 0,
            "opponent_action_cached_count": 0,
            "opponent_actor_leaves": [leaf],
        },
        "observed_resource_probe": dict(resources),
        "startup_seconds": startup_seconds,
        "_actual_stock_runtime_observation": dict(runtime),
        "_literal_staged_marker": dict(marker),
        "_literal_staged_marker_sha256": marker_sha256,
        "_r244_semantic_contract_source": payload.get("semantic_contract_source"),
    }


def _validate_written_r244_actual_witness(
    *,
    witness_path: Path,
    scenario_payload: Mapping[str, Any],
    binding_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the standalone R244 input to one fresh physical staged marker.

    The three namespace fields are contract semantics, not values returned by
    native libcg.  Their explicit R225 projection is accepted only alongside
    topology vectors copied byte-for-byte from the actual staged marker.
    """

    source = _regular_file(witness_path, label="actual R244 witness output")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactStageProbeError("actual R244 witness output is unreadable JSON") from exc
    witness = dict(_mapping(raw, label="actual R244 witness output"))
    if witness.get("schema") != R244_WITNESS_SCHEMA:
        raise ExactStageProbeError("actual R244 witness output has the wrong schema")
    if witness.get("witness_origin") != (
        "actual_staged_mcts_marker_topology_with_r225_contract_namespace_projection"
    ):
        raise ExactStageProbeError("actual R244 witness output has the wrong evidence origin")
    _require_binding_identity(
        witness,
        expected=binding_identity,
        label="actual R244 witness output",
        require_common_identity=True,
    )
    marker = _mapping(
        scenario_payload.get("literal_staged_marker"), label="actual R244 source marker"
    )
    marker_sha256 = _sha256_bytes(marker)
    _require_exact_sha256(
        scenario_payload.get("literal_staged_marker_sha256"),
        expected=marker_sha256,
        label="actual R244 scenario marker SHA-256",
    )
    _require_exact_sha256(
        witness.get("literal_staged_marker_sha256"),
        expected=marker_sha256,
        label="actual R244 witness marker SHA-256",
    )
    for field in (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "arena_count",
        "unique_handle_count",
        "search_begin_calls",
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
    ):
        if witness.get(field) != marker.get(field):
            raise ExactStageProbeError(f"actual R244 witness {field} drifted from its literal staged marker")
    semantic = _mapping(
        witness.get("semantic_contract_source"), label="actual R244 semantic contract source"
    )
    if semantic.get("kind") != R244_CONTRACT_PROJECTION_KIND:
        raise ExactStageProbeError("actual R244 witness semantic contract source is not recognized")
    expected_r225_sha = binding_identity["common_identity"].get("r225_contract_sha256")
    if semantic.get("r225_contract_sha256") != expected_r225_sha:
        raise ExactStageProbeError("actual R244 witness semantic contract source has the wrong R225 SHA-256")
    static_contract = {
        "owner_handle_scoped_search_id_revision": r244_preflight.R244_OWNER_REVISION,
        "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }
    for field, expected in static_contract.items():
        if semantic.get(field) != expected:
            raise ExactStageProbeError(f"actual R244 semantic contract {field} drifted")
    for field, expected in {
        "search_id_numeric_namespace": "per_distinct_agent_start_handle",
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
    }.items():
        if witness.get(field) != expected:
            raise ExactStageProbeError(f"actual R244 witness {field} does not match its R225 projection")
    try:
        r244_preflight._validate_witness(witness)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - preserve the standalone gate's precise parser
        raise ExactStageProbeError(f"actual R244 witness fails standalone validation: {exc}") from exc
    return witness


def _r244_witness_output_target(path: Path, *, stage: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise ExactStageProbeError("actual R244 witness output already exists; refusing overwrite")
    target = raw.resolve()
    if _inside(target, stage):
        raise ExactStageProbeError("actual R244 witness output must be outside the sealed stage")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ExactStageProbeError("actual R244 witness output parent must be physical")
    return target


def _actual_boundary_resource_worker_argv(
    *,
    stage: Path,
    archive: Path,
    manifest: Path,
    r225: Path,
    r236: Path,
    r244_witness_output: Path,
) -> list[str]:
    """Build the only automatic boundary/resource worker invocation.

    This remains a normal argv (never a shell) and is still placed under a
    fresh exact-child watchdog by the caller.
    """

    return [
        sys.executable,
        str(ACTUAL_STOCK_RUNTIME_WORKER),
        "--stage",
        str(stage),
        "--candidate-archive",
        str(archive),
        "--member-manifest",
        str(manifest),
        "--r225-contract",
        str(r225),
        "--r236-contract",
        str(r236),
        "--r244-witness-output",
        str(r244_witness_output),
    ]


def _scenario_witness_bundle(
    payload: Mapping[str, Any], *, name: str, binding_identity: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Accept only a typed, exact-bound physical stock-route worker bundle."""

    if payload.get("schema") != SCENARIO_EVIDENCE_SCHEMA:
        return None
    if payload.get("status") != "passed" or payload.get("passed") is not True:
        raise ExactStageProbeError(f"scenario {name} did not pass")
    origin = payload.get("witness_origin")
    if origin == "controlled_parent_route":
        bundle = _mapping(payload.get("r240_witnesses"), label=f"scenario {name} controlled witnesses")
        forbidden = sorted(set(bundle).intersection(STOCK_ONLY_WITNESSES))
        if forbidden:
            raise ExactStageProbeError(
                "controlled_parent_route may not contribute stock-only witnesses: "
                + ", ".join(forbidden)
            )
        raise ExactStageProbeError(
            "controlled_parent_route must use the reviewed controlled-parent worker schema"
        )
    if origin == ACTUAL_STOCK_RUNTIME_OBSERVATION_ORIGIN:
        return _actual_stock_runtime_observation_witnesses(
            payload,
            name=name,
            binding_identity=binding_identity,
        )
    if origin != "actual_stock_search_route":
        raise ExactStageProbeError(f"scenario {name} has an unrecognized evidence origin")
    _require_binding_identity(
        payload,
        expected=binding_identity,
        label=f"scenario {name}",
        require_common_identity=True,
    )
    return _mapping(payload.get("r240_witnesses"), label=f"scenario {name} R240 witnesses")


def _controlled_parent_witnesses(
    payload: Mapping[str, Any], *, stage: Path
) -> Mapping[str, Any] | None:
    """Validate the reviewed nonphysical worker and retain its narrow scope."""

    if payload.get("schema") != CONTROLLED_PARENT_ROUTE_SCHEMA:
        return None
    required = {
        "status": "passed",
        "controlled": True,
        "evidence_kind": "controlled_parent_route",
        "controlled_parent_route": True,
        "evidence_class": (
            "controlled_in_memory_parent_route_regression_not_physical_game_"
            "not_preflight_eligible"
        ),
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ExactStageProbeError(f"controlled parent worker {field} is not the required value")
    mutation = _mapping(payload.get("stage_mutation_check"), label="controlled parent mutation check")
    if mutation.get("unchanged") is not True:
        raise ExactStageProbeError("controlled parent worker mutated the sealed stage")
    raw_normalized = _mapping(
        payload.get("normalized_parent_route_evidence"),
        label="controlled parent normalized route evidence",
    )
    for field in ("controlled_only", "nonphysical", "not_r240_final_schema"):
        if raw_normalized.get(field) is not True:
            raise ExactStageProbeError(f"controlled parent evidence {field} is not true")
    forbidden = sorted(set(raw_normalized).intersection(STOCK_ONLY_WITNESSES))
    if forbidden:
        raise ExactStageProbeError(
            "controlled_parent_route may not contribute stock-only witnesses: "
            + ", ".join(forbidden)
        )
    wrapper_fields = {"controlled_only", "nonphysical", "not_r240_final_schema"}
    unsupported = sorted(
        set(raw_normalized).difference(CONTROLLED_PARENT_ALLOWED_WITNESSES | wrapper_fields)
    )
    if unsupported:
        raise ExactStageProbeError(
            "controlled_parent_route supplied unsupported witnesses: " + ", ".join(unsupported)
        )
    route_results = payload.get("route_results")
    if not isinstance(route_results, list) or not route_results:
        raise ExactStageProbeError("controlled parent worker lacks route results")
    for index, raw_result in enumerate(route_results):
        result = _mapping(raw_result, label=f"controlled parent route {index}")
        if result.get("status") != "passed":
            raise ExactStageProbeError(f"controlled parent route {index} did not pass")
        for field in (
            "network_accessed",
            "kaggle_api_called",
            "kaggle_upload_used",
            "gpu_used",
            "simulator_started",
            "model_loaded",
        ):
            if result.get(field) is not False:
                raise ExactStageProbeError(
                    f"controlled parent route {index} {field} is not false"
                )
        result_mutation = _mapping(
            result.get("stage_mutation_check"), label=f"controlled parent route {index} mutation"
        )
        if result_mutation.get("unchanged") is not True:
            raise ExactStageProbeError(f"controlled parent route {index} mutated the sealed stage")
        details = _mapping(result.get("result"), label=f"controlled parent route {index} details")
        imports = _mapping(details.get("stage_import"), label=f"controlled parent route {index} imports")
        for field in ("main", "features"):
            raw_path = imports.get(field)
            if not isinstance(raw_path, str) or not _inside(Path(raw_path).resolve(), stage):
                raise ExactStageProbeError(
                    f"controlled parent route {index} imported {field} outside the sealed stage"
                )
    return {
        key: raw_normalized[key]
        for key in CONTROLLED_PARENT_ALLOWED_WITNESSES
        if key in raw_normalized
    }


def _one_explicit_witness(
    bundles: Sequence[Mapping[str, Any]], *, key: str, label: str
) -> Mapping[str, Any] | None:
    rows = [bundle[key] for bundle in bundles if key in bundle]
    if not rows:
        return None
    if len(rows) != 1:
        raise ExactStageProbeError(f"multiple {label} witnesses were supplied")
    return _mapping(rows[0], label=label)


def _controlled_event(
    events: Sequence[object], *, mode: str, route_case: str | None = None, label: str
) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for raw in events:
        if not isinstance(raw, Mapping) or raw.get("mode") != mode:
            continue
        if route_case is not None and raw.get("route_case") != route_case:
            continue
        matches.append(raw)
    if len(matches) != 1:
        raise ExactStageProbeError(f"controlled continuation evidence lacks one {label} event")
    event = matches[0]
    if event.get("controlled_only") is not True:
        raise ExactStageProbeError(f"controlled {label} event lost its controlled-only label")
    return event


def _controlled_root_fingerprint(event: Mapping[str, Any], *, label: str) -> str:
    fingerprint = event.get("controlled_root_observation_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ExactStageProbeError(f"controlled {label} event lacks a staged-main root fingerprint")
    if event.get("controlled_root_observation_fingerprint_source") != (
        "staged_main._canonical_observation_fingerprint"
    ):
        raise ExactStageProbeError(f"controlled {label} fingerprint source is not staged main")
    return fingerprint


def _normalize_controlled_full_game(
    payload: Mapping[str, Any], *, high: Mapping[str, Any]
) -> dict[str, Any]:
    """Map owner-authorized controlled parent facts into legacy R240 fields.

    Only parent direct/continuation behavior is represented here.  The output
    intentionally carries controlled-only provenance on every synthetic row;
    no field from this function is used as the stock topology, resource,
    actor-boundary, CUDA, or R246 terminal witness.
    """

    for field in ("controlled_only", "nonphysical"):
        if payload.get(field) is not True:
            raise ExactStageProbeError(f"controlled continuation evidence {field} is not true")
    source_events = payload.get("decision_events")
    if not isinstance(source_events, list):
        raise ExactStageProbeError("controlled continuation evidence lacks decision events")
    high_event = _controlled_event(
        source_events,
        mode="high_confidence_frozen_direct",
        label="high-confidence direct",
    )
    mcts_event = _controlled_event(
        source_events,
        mode="new_adaptive_two_lane_mcts",
        route_case="continuation_consume",
        label="plan-extraction MCTS",
    )
    continuation_event = _controlled_event(
        source_events,
        mode="cached_deterministic_continuation",
        route_case="continuation_consume",
        label="continuation consume",
    )
    high_fingerprint = _controlled_root_fingerprint(high_event, label="high-confidence direct")
    mcts_fingerprint = _controlled_root_fingerprint(mcts_event, label="plan-extraction MCTS")
    continuation_fingerprint = _controlled_root_fingerprint(
        continuation_event, label="continuation consume"
    )
    if payload.get("controlled_high_confidence_root_observation_fingerprint") != high_fingerprint:
        raise ExactStageProbeError("controlled high-direct fingerprint convenience field drifted")
    if payload.get("controlled_plan_extraction_root_observation_fingerprint") != mcts_fingerprint:
        raise ExactStageProbeError("controlled MCTS fingerprint convenience field drifted")
    source_marker = _mapping(
        payload.get("controlled_plan_extraction_marker"),
        label="controlled verbatim plan-extraction marker",
    )
    if payload.get("controlled_plan_extraction_marker_is_verbatim_staged_parent_marker") is not True:
        raise ExactStageProbeError("controlled plan-extraction marker is not declared verbatim")
    if source_marker.get("mode") != "shared_tree_mcts":
        raise ExactStageProbeError("controlled plan-extraction source is not a staged MCTS marker")
    if source_marker.get("mcts_child_started") is not True or source_marker.get(
        "mcts_child_call_count"
    ) != 1:
        raise ExactStageProbeError("controlled plan-extraction marker lacks one parent broker call")
    if source_marker.get("mcts_action_authority") is not True:
        raise ExactStageProbeError("controlled plan-extraction marker lacks parent MCTS authority")
    if source_marker.get("selected_action") != mcts_event.get("selected_action"):
        raise ExactStageProbeError("controlled plan-extraction marker/action link drifted")

    source_plans = payload.get("deterministic_continuation_plans")
    if not isinstance(source_plans, list) or len(source_plans) != 1:
        raise ExactStageProbeError("controlled continuation evidence lacks one plan")
    source_plan = _mapping(source_plans[0], label="controlled continuation plan")
    plan_id = _nonempty_string(source_plan.get("plan_id"), label="controlled continuation plan id")
    turn_id = _nonempty_string(source_plan.get("actual_turn_id"), label="controlled continuation turn id")
    source_steps = source_plan.get("steps")
    if not isinstance(source_steps, list) or len(source_steps) != 1:
        raise ExactStageProbeError("controlled continuation plan lacks exactly one consumed step")
    source_step = _mapping(source_steps[0], label="controlled continuation step")
    planned_fingerprint = _nonempty_string(
        source_step.get("canonical_observation_fingerprint"),
        label="controlled planned observation fingerprint",
    )
    planned_action = source_step.get("planned_action")
    if (
        not isinstance(planned_action, list)
        or not planned_action
        or any(isinstance(value, bool) or not isinstance(value, int) for value in planned_action)
    ):
        raise ExactStageProbeError("controlled continuation planned action is malformed")
    if planned_fingerprint != continuation_fingerprint:
        raise ExactStageProbeError("controlled continuation plan does not match its consumed prompt")
    if continuation_event.get("selected_action") != planned_action:
        raise ExactStageProbeError("controlled continuation did not select its planned action")

    high_elapsed = _finite_nonnegative(
        high_event.get("parent_action_elapsed_seconds"), label="controlled high-direct elapsed"
    )
    mcts_elapsed = _finite_nonnegative(
        mcts_event.get("parent_action_elapsed_seconds"), label="controlled MCTS elapsed"
    )
    continuation_elapsed = _finite_nonnegative(
        continuation_event.get("parent_action_elapsed_seconds"),
        label="controlled continuation elapsed",
    )
    for field in (
        "selected_factorized_stage_probabilities",
        "selected_factorized_stage_probability_threshold",
        "all_selected_factorized_stages_meet_threshold",
        "mcts_child_started_for_this_decision",
        "mcts_select_call_count",
        "mcts_search_call_count",
        "mcts_model_call_count",
        "mcts_simulator_call_count",
        "history_only_existing_child_journal_count",
        "degraded",
    ):
        if field not in high:
            raise ExactStageProbeError(f"controlled high-direct witness lacks {field}")

    # The staged marker carries these topology records verbatim from the
    # controlled two-lane receipt that the parent independently validated.
    topology_fields = (
        "requested_simulator_lane_count",
        "active_simulator_lane_count",
        "per_lane_handle_identities",
        "per_lane_search_id_chains",
        "per_lane_first_search_ids",
        "handle_scoped_first_search_id_composite_states",
    )
    _required_marker_fields(
        source_marker, topology_fields, label="controlled plan-extraction topology"
    )
    controlled_high_event = {
        "controlled_only": True,
        "nonphysical": True,
        "controlled_evidence_origin": "staged_parent_direct_route",
        "mode": "high_confidence_frozen_direct",
        "actual_turn_id": "controlled-parent-route-high-turn-001",
        "canonical_observation_fingerprint": high_fingerprint,
        "selected_action": high.get("selected_action"),
        "selected_factorized_stage_probabilities": high[
            "selected_factorized_stage_probabilities"
        ],
        "selected_factorized_stage_probability_threshold": high[
            "selected_factorized_stage_probability_threshold"
        ],
        "all_selected_factorized_stages_meet_threshold": high[
            "all_selected_factorized_stages_meet_threshold"
        ],
        "mcts_child_started_for_this_decision": high[
            "mcts_child_started_for_this_decision"
        ],
        "mcts_select_call_count": high["mcts_select_call_count"],
        "mcts_search_call_count": high["mcts_search_call_count"],
        "mcts_model_call_count": high["mcts_model_call_count"],
        "mcts_simulator_call_count": high["mcts_simulator_call_count"],
        "history_only_existing_child_journal_count": high[
            "history_only_existing_child_journal_count"
        ],
        "degraded": high["degraded"],
        "parent_action_elapsed_seconds": high_elapsed,
    }
    controlled_mcts_event = {
        "controlled_only": True,
        "nonphysical": True,
        "controlled_evidence_origin": "staged_parent_controlled_broker_route",
        "mode": "new_adaptive_two_lane_mcts",
        # This name deliberately distinguishes controlled plan extraction from
        # the later exact planned prompt consumed below.
        "actual_turn_id": "controlled-parent-route-plan-extraction-turn-001",
        "canonical_observation_fingerprint": mcts_fingerprint,
        "broker_started": True,
        "mcts_child_started": True,
        "mcts_child_called": True,
        "new_mcts_search_started": True,
        "requested_simulator_lane_count": source_marker[
            "requested_simulator_lane_count"
        ],
        "active_simulator_lane_count": source_marker["active_simulator_lane_count"],
        "per_lane_handle_identities": source_marker["per_lane_handle_identities"],
        "per_lane_search_id_chains": source_marker["per_lane_search_id_chains"],
        "per_lane_first_search_ids": source_marker["per_lane_first_search_ids"],
        "handle_scoped_first_search_id_composite_states": source_marker[
            "handle_scoped_first_search_id_composite_states"
        ],
        "child_search_budget_seconds": preflight.R240_CHILD_SEARCH_HARD_SECONDS,
        "child_search_elapsed_seconds": 0.0,
        "parent_action_deadline_seconds": preflight.R240_PARENT_ACTION_HARD_SECONDS,
        "parent_action_elapsed_seconds": mcts_elapsed,
    }
    controlled_continuation_event = {
        "controlled_only": True,
        "nonphysical": True,
        "controlled_evidence_origin": "staged_parent_continuation_consume_route",
        "mode": "cached_deterministic_continuation",
        "actual_turn_id": turn_id,
        "canonical_observation_fingerprint": planned_fingerprint,
        "plan_id": plan_id,
        "selected_action": list(planned_action),
        "exact_fingerprint_match": True,
        "same_actor": True,
        "action_in_complete_legal_order": True,
        "two_lane_agreed_backed_leader": True,
        "no_chance_boundary_or_opponent_transition": True,
        "crossed_actor_change_end_turn_boundary": False,
        "new_mcts_search_started": False,
        "mcts_child_called": False,
        "mcts_child_started_for_this_decision": False,
        "mcts_select_call_count": 0,
        "mcts_search_call_count": 0,
        "mcts_model_call_count": 0,
        "mcts_simulator_call_count": 0,
        "history_only_existing_child_journal_count": continuation_event.get(
            "history_only_existing_child_journal_count"
        ),
        "degraded": continuation_event.get("degraded"),
        "parent_action_elapsed_seconds": continuation_elapsed,
    }
    normalized_plan = {
        "controlled_only": True,
        "nonphysical": True,
        "plan_id": plan_id,
        "actual_turn_id": turn_id,
        "extracted_from_mode": "adaptive_two_lane_mcts",
        "exact_fingerprint_proven": True,
        "two_lane_agreed_backed_leader": True,
        "no_chance_boundary_or_opponent_transition": True,
        "crossed_actor_change_end_turn_boundary": False,
        "steps": [
            {
                "canonical_observation_fingerprint": planned_fingerprint,
                "planned_action": list(planned_action),
            }
        ],
    }
    regression = _mapping(
        payload.get("deterministic_continuation_regression"),
        label="controlled continuation mismatch regression",
    )
    for field in (
        "chance_disagreement_clears_entire_plan",
        "fingerprint_disagreement_clears_entire_plan",
        "action_disagreement_clears_entire_plan",
        "actor_disagreement_clears_entire_plan",
        "precomputed_direct_action_and_history_correction_retained",
    ):
        if regression.get(field) is not True:
            raise ExactStageProbeError(f"controlled continuation regression lacks {field}")
    parent_total = high_elapsed + mcts_elapsed + continuation_elapsed
    return {
        "controlled_only": True,
        "nonphysical": True,
        "controlled_evidence_origin": "owner_authorized_staged_parent_routes",
        "cumulative_parent_wall_seconds": parent_total,
        "cumulative_child_search_seconds": 0.0,
        "new_mcts_search_count": 1,
        "cached_deterministic_continuation_count": 1,
        "high_confidence_frozen_direct_count": 1,
        "deterministic_continuation_plans": [normalized_plan],
        "decision_events": [
            controlled_high_event,
            controlled_mcts_event,
            controlled_continuation_event,
        ],
        "deterministic_continuation_regression": dict(regression),
        "controlled_source_plan_extraction_marker_sha256": _sha256_bytes(source_marker),
    }


def _one_marker_or_none(
    markers: Sequence[Mapping[str, Any]], *, predicate: Any
) -> Mapping[str, Any] | None:
    matches = [marker for marker in markers if predicate(marker)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ExactStageProbeError("multiple physical markers match one required route")
    return matches[0]


def _require_witness_fields(
    witness: Mapping[str, Any] | None, *, label: str, fields: Sequence[str]
) -> Mapping[str, Any]:
    if witness is None:
        raise ExactStageProbeError(f"no {label} witness was observed")
    _required_marker_fields(witness, fields, label=label)
    return witness


def _normalize_actual_stock_mcts_witness(
    witness: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Project only the route spelling needed by the legacy probe reader.

    This is intentionally narrow.  The staged parent must already contain all
    operational facts (child calls, budgets/elapsed values, and R246 proof
    fields) literally.  In particular, this function never turns a count into
    a call fact, aliases a timer, or invents broker/start evidence.
    """

    result = dict(witness)
    if result.get("mode") == "shared_tree_mcts":
        result["mode"] = "adaptive_two_lane_mcts"
        result["source_mode"] = "shared_tree_mcts"
    if result.get("stop_reason") == "stable_root_leader":
        leaders = result.get("deterministic_root_leader_observations")
        if (
            isinstance(leaders, list)
            and len(leaders) >= preflight.R240_STABLE_ROOT_LEADER_OBSERVATIONS
            and all(isinstance(value, str) and value for value in leaders)
            and len(set(leaders[-preflight.R240_STABLE_ROOT_LEADER_OBSERVATIONS :])) == 1
        ):
            result["stop_reason"] = "adaptive_early_stop"
            result["source_stop_reason"] = "stable_root_leader"
    if result.get("mode") not in {
        "adaptive_two_lane_mcts",
        "shared_tree_mcts",
    }:
        raise ExactStageProbeError(f"{label} is not a stock two-lane MCTS marker")
    return result


def _collect_witnesses(
    *,
    smoke: Mapping[str, Any],
    scenario_results: Sequence[ScenarioResult],
    binding_identity: Mapping[str, Any],
    stage: Path,
) -> dict[str, Any]:
    """Collect explicit route evidence, keeping controlled and stock facts apart."""

    markers = [marker for marker in smoke["decision_markers"] if isinstance(marker, Mapping)]
    callbacks = [row for row in smoke["callbacks"] if isinstance(row, Mapping)]
    for marker in markers:
        _marker_matches_callback(marker, callbacks)

    actual_bundles: list[Mapping[str, Any]] = []
    controlled_bundles: list[Mapping[str, Any]] = []
    for result in scenario_results:
        actual = _scenario_witness_bundle(
            result.payload, name=result.name, binding_identity=binding_identity
        )
        if actual is not None:
            actual_bundles.append(actual)
        controlled = _controlled_parent_witnesses(result.payload, stage=stage)
        if controlled is not None:
            controlled_bundles.append(controlled)

    controlled_high = _one_explicit_witness(
        controlled_bundles,
        key="synthetic_high_confidence_direct",
        label="controlled high-confidence direct",
    )
    controlled_full_game = _one_explicit_witness(
        controlled_bundles,
        key="full_game_cumulative",
        label="controlled continuation mismatch-and-consume",
    )
    high = controlled_high or _one_marker_or_none(
        markers, predicate=lambda row: row.get("mode") == "high_confidence_frozen_direct"
    )
    normal_raw = _one_marker_or_none(
        markers,
        predicate=lambda row: row.get("mode") == "shared_tree_mcts"
        and row.get("stop_reason") in {"adaptive_early_stop", "stable_root_leader"},
    ) or _one_explicit_witness(
        actual_bundles,
        key="synthetic_ambiguous_two_lane_mcts",
        label="actual ordinary adaptive two-lane MCTS",
    )
    terminal_raw = _one_marker_or_none(
        markers,
        predicate=lambda row: row.get("mode") == "shared_tree_mcts"
        and row.get("stop_reason") == preflight.R246_TERMINAL_WIN_STOP_REASON,
    ) or _one_explicit_witness(
        actual_bundles,
        key=preflight.R246_TERMINAL_WIN_PROBE_KEY,
        label="actual R246 stock terminal-win",
    )
    normal = (
        None
        if normal_raw is None
        else _normalize_actual_stock_mcts_witness(
            normal_raw, label="ordinary adaptive two-lane MCTS"
        )
    )
    terminal = (
        None
        if terminal_raw is None
        else _normalize_actual_stock_mcts_witness(terminal_raw, label="R246 stock terminal-win")
    )
    high = _require_witness_fields(
        high,
        label="high-confidence direct",
        fields=(
            "direct_action_precomputed_and_validated",
            "mcts_child_started_for_this_decision",
            "mcts_select_call_count",
            "mcts_search_call_count",
            "mcts_model_call_count",
            "mcts_simulator_call_count",
            "history_only_existing_child_journal_count",
            "degraded",
            "parent_action_elapsed_seconds",
        ),
    )
    normal = _require_witness_fields(
        normal,
        label="ordinary adaptive two-lane MCTS",
        fields=(
            "broker_started",
            "mcts_child_started",
            "mcts_child_called",
            "mcts_action_authority",
            "both_lanes_progressed",
            "deterministic_root_leader_observations",
            "child_search_elapsed_seconds",
            "parent_action_elapsed_seconds",
        ),
    )
    terminal = _require_witness_fields(
        terminal,
        label="R246 stock terminal-win",
        fields=(
            "direct_action_precomputed_and_validated",
            "broker_started",
            "mcts_child_started",
            "mcts_child_called",
            "mcts_action_authority",
            "two_lane_topology_initialized_before_terminal_win_override",
            "terminal_win_proof_backed_up_into_shared_root_tree",
            "terminal_leaf_returned_by_exact_stock_simulator",
            "parent_validated_current_root_observation_legal_fingerprint_and_actor",
            "all_owned_lane_resources_reservations_and_child_cleanup_complete",
            "completed_root_backup_count",
            "terminal_win_proof_count",
            "proven_deterministic_terminal_win_this_turn_stop_count",
            "child_search_elapsed_seconds",
            "parent_action_elapsed_seconds",
        ),
    )
    boundary = _one_explicit_witness(
        actual_bundles,
        key="actor_change_end_turn_boundary",
        label="actual stock leaf-level actor-boundary",
    )
    resources = _one_explicit_witness(
        actual_bundles,
        key="observed_resource_probe",
        label="actual parent/broker resource",
    )
    startup = _one_explicit_witness(
        actual_bundles,
        key="startup_seconds",
        label="actual parent/broker startup",
    )
    if boundary is None:
        raise ExactStageProbeError("no actual stock leaf-level actor-boundary witness was observed")
    if controlled_full_game is None:
        raise ExactStageProbeError("no controlled continuation mismatch-and-consume witness was observed")
    if resources is None or startup is None:
        raise ExactStageProbeError("no actual parent/broker resource and startup witness was observed")
    normalized_full_game = _normalize_controlled_full_game(
        controlled_full_game,
        high=high,
    )
    return {
        "high": dict(high),
        "normal": dict(normal),
        "terminal": dict(terminal),
        "boundary": dict(_mapping(boundary, label="actor-boundary witness")),
        "full_game": normalized_full_game,
        "resources": dict(_mapping(resources, label="resource witness")),
        "startup_seconds": _finite_nonnegative(startup, label="actual startup seconds"),
        "source_marker_sha256": {
            "high": _sha256_bytes(high),
            "normal": _sha256_bytes(normal),
            "terminal": _sha256_bytes(terminal),
        },
    }


def build_probe_from_actual_scenarios(
    *,
    smoke_result: ScenarioResult,
    scenario_results: Sequence[ScenarioResult],
    stage: Path,
    archive: Path,
    manifest: Path,
    r225: Path,
    r236: Path,
    phase1_full_game_budget_seconds: float,
    binding_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the legacy R240 shape solely from actual linked witnesses."""

    binding = (
        dict(binding_identity)
        if binding_identity is not None
        else _load_exact_binding_identity(
            stage=stage, archive=archive, manifest=manifest, r225=r225, r236=r236
        )
    )
    smoke = _validate_raw_smoke(
        smoke_result.payload,
        stage=stage,
        binding_identity=binding,
    )
    witnesses = _collect_witnesses(
        smoke=smoke,
        scenario_results=scenario_results,
        binding_identity=binding,
        stage=stage,
    )
    callbacks = smoke["callbacks"]
    elapsed_values = [
        _finite_nonnegative(row.get("callback_elapsed_seconds"), label="physical callback elapsed")
        for row in callbacks
    ]
    ordered = sorted(elapsed_values)
    p50 = ordered[(len(ordered) - 1) // 2]
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    elapsed = smoke["elapsed_seconds"]
    if elapsed <= 0.0:
        raise ExactStageProbeError("physical smoke elapsed time is zero")
    parent_cuda = _require_matching_marker_cuda(
        witnesses["normal"], label="actual ordinary adaptive two-lane MCTS"
    )
    # The resource worker must bind staged parent/broker measurements.  The
    # raw smoke's own process RSS is retained only as source provenance; it is
    # not misrepresented as broker-child RSS.
    resource_probe = witnesses["resources"]
    if resource_probe.get("runtime_disk_bytes") != smoke["stage_disk_bytes"]:
        raise ExactStageProbeError("actual resource witness does not match sealed stage disk bytes")
    resource_cuda = _complete_cuda_observation(
        resource_probe.get("cuda_runtime_before_search"), label="actual resource parent"
    )
    if _cuda_semantic_identity(resource_cuda) != _cuda_semantic_identity(parent_cuda):
        raise ExactStageProbeError("actual resource witness does not match physical parent CUDA observation")
    return {
        "schema": preflight.R240_PROBE_SCHEMA,
        "witness_origin": "actual_exact_package_scenarios",
        "candidate_archive_sha256": _sha256_file(archive),
        "member_manifest_sha256": _sha256_file(manifest),
        "r225_contract_sha256": _sha256_file(r225),
        "canonical_libcg_contract_sha256": _sha256_file(r236),
        "actual_scenario_source": {
            "physical_smoke_stdout_sha256": _sha256_bytes(smoke_result.payload),
            "physical_smoke_watchdog": smoke_result.outcome.as_dict(),
            "additional_exact_child_scenarios": [
                {
                    "name": result.name,
                    "schema": result.payload.get("schema"),
                    "evidence_kind": result.payload.get("evidence_kind"),
                    "stdout_sha256": _sha256_bytes(result.payload),
                    "watchdog": result.outcome.as_dict(),
                }
                for result in scenario_results
            ],
            "source_marker_sha256": witnesses["source_marker_sha256"],
        },
        "observed_resource_probe": resource_probe,
        "startup_seconds": witnesses["startup_seconds"],
        "decision_latency_seconds": {
            "sample_count": len(elapsed_values),
            "p50": p50,
            "p95": p95,
            "max": max(elapsed_values),
        },
        "throughput": {
            "decision_count": len(callbacks),
            "elapsed_seconds": elapsed,
            "decisions_per_second": len(callbacks) / elapsed,
        },
        preflight.R240_HYBRID_PROBE_KEY: {
            "owner_decision_revision": preflight.R246_OWNER_REVISION,
            "configuration": {
                "high_confidence_threshold_owner_revision": preflight.R242_OWNER_REVISION,
                "high_confidence_threshold": preflight.R242_HIGH_CONFIDENCE_THRESHOLD,
                "child_search_hard_seconds": preflight.R240_CHILD_SEARCH_HARD_SECONDS,
                "parent_action_hard_seconds": preflight.R240_PARENT_ACTION_HARD_SECONDS,
                "minimum_backups_before_stability": preflight.R240_MINIMUM_BACKUPS_BEFORE_STABILITY,
                "stable_root_leader_observations": preflight.R240_STABLE_ROOT_LEADER_OBSERVATIONS,
                "maximum_backups_per_decision": preflight.R240_MAXIMUM_BACKUPS_PER_DECISION,
                "maximum_deterministic_continuation_actions": preflight.R240_MAX_DETERMINISTIC_CONTINUATION_ACTIONS,
                "legacy_fixed_eight_second_branching_windows_rejected": True,
                "historical_r240_0_90_threshold_draft_and_preflight_rejected": True,
                "proven_deterministic_terminal_win_this_turn_owner_revision": preflight.R246_OWNER_REVISION,
            },
            "synthetic_high_confidence_direct": witnesses["high"],
            "synthetic_ambiguous_two_lane_mcts": witnesses["normal"],
            preflight.R246_TERMINAL_WIN_PROBE_KEY: witnesses["terminal"],
            "actor_change_end_turn_boundary": witnesses["boundary"],
            "actor_change_boundary_leaf_count": witnesses["normal"].get(
                "actor_change_boundary_leaf_count"
            ),
            "chance_boundary_leaf_count": witnesses["normal"].get(
                "chance_boundary_leaf_count"
            ),
            "boundary_leaf_count": witnesses["normal"].get("boundary_leaf_count"),
            "full_game_cumulative": {
                **witnesses["full_game"],
                "phase1_full_game_budget_seconds": phase1_full_game_budget_seconds,
            },
        },
    }


def _positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--r225-contract", type=Path, default=ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json")
    parser.add_argument("--r236-contract", type=Path, default=ROOT / "state/canonical-libcg-r236.json")
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--game-timeout-seconds", type=_positive, default=900.0)
    parser.add_argument("--per-action-timeout-seconds", type=_positive, default=4.0)
    parser.add_argument("--scenario-timeout-seconds", type=_positive, required=True)
    parser.add_argument("--term-grace-seconds", type=_positive, required=True)
    parser.add_argument("--kill-grace-seconds", type=_positive, required=True)
    parser.add_argument("--phase1-full-game-budget-seconds", type=_positive, required=True)
    parser.add_argument(
        "--controlled-parent-routes",
        action="store_true",
        help=(
            "run the reviewed, explicitly nonphysical staged-parent direct/"
            "continuation regression worker in its own exact child"
        ),
    )
    parser.add_argument(
        "--actual-boundary-resource-route",
        action="store_true",
        help=(
            "run the reviewed fresh-stock actor-boundary/resource/startup worker "
            "in its own exact child"
        ),
    )
    parser.add_argument(
        "--r244-witness-output",
        type=Path,
        help=(
            "new immutable standalone R244 witness output written only by the "
            "reviewed actual boundary/resource worker"
        ),
    )
    parser.add_argument(
        "--actual-stock-route-command-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "reviewed project-script argv JSON for one additional exact-stage "
            "physical stock-route worker; it must bind stage/archive/contracts"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if int(args.max_actions) < 1:
            raise ExactStageProbeError("--max-actions must be positive")
        stage = _physical_directory(args.stage, label="stage")
        archive = _regular_file(args.candidate_archive, label="candidate archive")
        manifest = _regular_file(args.member_manifest, label="member manifest")
        r225 = _regular_file(args.r225_contract, label="r225 contract")
        r236 = _regular_file(args.r236_contract, label="r236 contract")
        if not _inside(manifest, stage):
            raise ExactStageProbeError("member manifest must be an exact staged member")
        binding_identity = _load_exact_binding_identity(
            stage=stage, archive=archive, manifest=manifest, r225=r225, r236=r236
        )
        if not args.controlled_parent_routes:
            raise ExactStageProbeError(
                "actual route suite requires --controlled-parent-routes for its "
                "explicitly nonphysical parent-only witnesses"
            )
        if not args.actual_boundary_resource_route:
            raise ExactStageProbeError(
                "actual route suite requires --actual-boundary-resource-route for "
                "fresh stock actor-boundary/resource/startup evidence"
            )
        if args.r244_witness_output is None:
            raise ExactStageProbeError(
                "--actual-boundary-resource-route requires --r244-witness-output"
            )
        r244_witness_output = _r244_witness_output_target(
            args.r244_witness_output,
            stage=stage,
        )
        smoke_receipt = Path(args.smoke_receipt).expanduser().resolve()
        if smoke_receipt.exists() or smoke_receipt.is_symlink():
            raise ExactStageProbeError("raw smoke receipt already exists; refusing overwrite")
        if _inside(smoke_receipt, stage):
            raise ExactStageProbeError("raw smoke receipt must be outside the sealed stage")
        if not smoke_receipt.parent.is_dir() or smoke_receipt.parent.is_symlink():
            raise ExactStageProbeError("raw smoke receipt parent must be a physical directory")
        smoke_argv = [
            sys.executable,
            str(ROOT / "scripts/run_r228_async_eight_worker_packaged_smoke.py"),
            "--stage",
            str(stage),
            "--candidate-archive",
            str(archive),
            "--member-manifest",
            str(manifest),
            "--r225-contract",
            str(r225),
            "--r236-contract",
            str(r236),
            "--receipt",
            str(smoke_receipt),
            "--max-actions",
            str(int(args.max_actions)),
            "--game-timeout-seconds",
            str(float(args.game_timeout_seconds)),
            "--per-action-timeout-seconds",
            str(float(args.per_action_timeout_seconds)),
            "--emit-r240-probe",
        ]
        smoke_result = _run_fresh_scenario(
            name="physical_stock_full_game",
            argv=smoke_argv,
            stage=stage,
            timeout_seconds=float(args.scenario_timeout_seconds),
            term_grace_seconds=float(args.term_grace_seconds),
            kill_grace_seconds=float(args.kill_grace_seconds),
        )
        scenario_results: list[ScenarioResult] = [
            _run_fresh_scenario(
                name="controlled_parent_routes",
                argv=[
                    sys.executable,
                    str(ROOT / "scripts/run_r240_controlled_parent_routes.py"),
                    "--stage",
                    str(stage),
                    "--case",
                    "all",
                ],
                stage=stage,
                timeout_seconds=float(args.scenario_timeout_seconds),
                term_grace_seconds=float(args.term_grace_seconds),
                kill_grace_seconds=float(args.kill_grace_seconds),
                required_stdout_prefix=CONTROLLED_PARENT_ROUTE_PREFIX,
            )
        ]
        boundary_resource_argv = _actual_boundary_resource_worker_argv(
            stage=stage,
            archive=archive,
            manifest=manifest,
            r225=r225,
            r236=r236,
            r244_witness_output=r244_witness_output,
        )
        _require_owned_actual_route_command(
            boundary_resource_argv,
            stage=stage,
            archive=archive,
            manifest=manifest,
            r225=r225,
            r236=r236,
            label="actual boundary/resource route",
        )
        boundary_resource_result = _run_fresh_scenario(
            name="actual_stock_boundary_resource_route",
            argv=boundary_resource_argv,
            stage=stage,
            timeout_seconds=float(args.scenario_timeout_seconds),
            term_grace_seconds=float(args.term_grace_seconds),
            kill_grace_seconds=float(args.kill_grace_seconds),
        )
        _validate_written_r244_actual_witness(
            witness_path=r244_witness_output,
            scenario_payload=boundary_resource_result.payload,
            binding_identity=binding_identity,
        )
        scenario_results.append(boundary_resource_result)
        for index, command_path in enumerate(args.actual_stock_route_command_json):
            command = _read_argv_json(command_path, label=f"actual stock route {index}")
            _require_owned_actual_route_command(
                command,
                stage=stage,
                archive=archive,
                manifest=manifest,
                r225=r225,
                r236=r236,
                label=f"actual stock route {index}",
            )
            scenario_results.append(
                _run_fresh_scenario(
                    name=f"actual_stock_route_{index}",
                    argv=command,
                    stage=stage,
                    timeout_seconds=float(args.scenario_timeout_seconds),
                    term_grace_seconds=float(args.term_grace_seconds),
                    kill_grace_seconds=float(args.kill_grace_seconds),
                )
            )
        probe = build_probe_from_actual_scenarios(
            smoke_result=smoke_result,
            scenario_results=scenario_results,
            stage=stage,
            archive=archive,
            manifest=manifest,
            r225=r225,
            r236=r236,
            phase1_full_game_budget_seconds=float(args.phase1_full_game_budget_seconds),
            binding_identity=binding_identity,
        )
    except Exception as exc:  # noqa: BLE001 - stdout must stay one probe object or empty
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "r225_contract_sha256": (
                        _sha256_file(Path(args.r225_contract))
                        if Path(args.r225_contract).is_file() and not Path(args.r225_contract).is_symlink()
                        else None
                    ),
                    "kaggle_api_called": False,
                    "kaggle_upload_used": False,
                    "kaggle_queue_used": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(probe, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
